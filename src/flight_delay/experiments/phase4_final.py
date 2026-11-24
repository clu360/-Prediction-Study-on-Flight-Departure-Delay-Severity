"""Fit one frozen model per arm/seed and independently verify its outputs."""

import hashlib
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import joblib
import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, precision_recall_curve

from .config import digest, file_hash
from .evaluation import airport_metrics, check_probabilities, evaluate
from .inputs import table_scan
from .phase4_checkpoint import compatible_store
from .phase4_inputs import BASE, HOLDOUT, ID, WEATHER
from .results import (
    PROB_COLUMNS, atomic, exclusive, prediction_frame, read_json, write_frame,
    write_json, write_model,
)
from .runner import ids_hash
from ..models.state import state_hash

ARMS = {
    "XGB-base": (BASE, False), "XGB-weather": (WEATHER, False),
    "MLP-A": (BASE, False), "MLP-B": (BASE, True),
    "MLP-C": (WEATHER, False), "MLP-D": (WEATHER, True),
}
AIRPORTS = ["ATL", "DFW", "DEN", "ORD", "LAX"]


def xgb_config():
    manifest = Path("reports/phase3/experiment_manifest.json")
    selected = Path("reports/phase3/selected_configs.json")
    report, selection = read_json(manifest), read_json(selected)
    signature = selection.pop("selection_hash")
    if digest(selection) != signature or signature != report["selected_configs_hash"]:
        raise ValueError("Phase III selection checksum differs")
    key = "xgboost/G1_G2_G3"
    params = report["chosen_parameters"][key]
    if selection["selected"][key]["params"] != params:
        raise ValueError("XGBoost documentation disagrees with the locked selection")
    resolved = report["resolved_config"]
    return dict(params=params, seed=resolved["seed"], threads=resolved["threads"],
                manifest_sha256=file_hash(manifest), selected_sha256=file_hash(selected))


def booster_hash(estimator):
    return hashlib.sha256(estimator.get_booster().save_raw(raw_format="ubj")).hexdigest()


def selection_config(inputs):
    selection = read_json(inputs.path("output_dir") / "mlp_selection.json")
    signature = selection.pop("sha256")
    if digest(selection) != signature or selection["settings"] != inputs.mlp or file_hash(
        inputs.path("reports_dir") / "mlp_tuning_summary.csv"
    ) != selection["tuning_summary_sha256"]:
        raise ValueError("MLP development selection integrity check failed")
    return selection | {"sha256": signature}


def weather_signature(inputs, upstream_sha):
    root = inputs.path("weather_dir")
    record = read_json(root / "audit" / "verified.json")
    if digest(record["context"]) != record["signature"]:
        raise ValueError("G4 context checksum differs")
    if record["context"]["input_sha256"] != upstream_sha or len(record["files"]) != 12:
        raise ValueError("G4 source population is not validated")
    for relative, checksum in record["files"].items():
        if file_hash(root / relative) != checksum:
            raise ValueError(f"G4 checksum differs: {relative}")
    return record["signature"]


def risk_metrics(y, p, metrics=None):
    result = evaluate(y, p) if metrics is None else dict(metrics)
    severe = y == 2
    result["severe_pr_auc_ap"] = float(average_precision_score(severe, p[:, 2])) if severe.any() else None
    if severe.any():
        precision, recall, _ = precision_recall_curve(severe, p[:, 2])
        # Publish a bounded numerical curve. Average precision uses the full data.
        indices = np.unique(np.linspace(0, len(precision) - 1, min(len(precision), 501), dtype=int))
        result["severe_pr_curve"] = [[float(recall[i]), float(precision[i])] for i in indices]
    return result


def metadata(inputs, month, ids):
    meta = table_scan(inputs.path("features_dir"), "full_features", [month]).select(
        ID, "g1_origin", "g2_prior_dep_delay_min", "g2_minutes_until_crs_dep_at_prediction"
    ).collect()
    aligned = ids.join(meta, on=ID, how="left", maintain_order="left", validate="1:1")
    if aligned["g1_origin"].null_count():
        raise ValueError("Missing evaluation metadata")
    return aligned


def subgroup_scores(y, p, meta, weather_names):
    score = {"overall": risk_metrics(y, p)}
    airports = meta["g1_origin"].to_numpy()
    score.update({f"airport:{name}": risk_metrics(y[airports == name], p[airports == name], values)
                  for name, values in airport_metrics(y, p, airports, AIRPORTS).items()})
    lead = meta["g2_minutes_until_crs_dep_at_prediction"].to_numpy()
    incoming = meta["g2_prior_dep_delay_min"].to_numpy()
    selectors = {
        "lead:<=120": lead <= 120, "lead:120–360": (lead > 120) & (lead <= 360),
        "lead:>360": lead > 360, "incoming:<15": incoming < 15,
    }
    if set(weather_names).issubset(meta.columns):
        origin, frozen, thunder, _, destination = meta.select(weather_names).to_numpy().T
        adverse = (origin > 0) | (destination > 0) | (frozen == 1) | (thunder == 1)
        known = np.isfinite(np.column_stack([origin, destination, frozen, thunder])).all(axis=1)
        selectors |= {"weather:adverse": adverse, "weather:benign": known & ~adverse,
                      "weather:unknown": ~known & ~adverse}
    score.update({name: risk_metrics(y[mask], p[mask]) for name, mask in selectors.items() if mask.any()})
    return score


def execute_arm(inputs, arm, seed, device="cpu"):
    upstream = inputs.verify_source("all")
    view, weighted = ARMS[arm]
    neural = arm.startswith("MLP")
    if neural:
        from .phase4_mlp import environment
        from ..models import mlp

        protocol = selection_config(inputs)
        if seed not in inputs.mlp["final_seeds"]:
            raise ValueError("Seed is outside the prespecified final seed set")
        hardware = environment(device)
    else:
        import platform
        import xgboost

        protocol = xgb_config()
        if seed != protocol["seed"]:
            raise ValueError("XGBoost seed differs from Phase III")
        hardware = dict(platform=platform.system(), xgboost=xgboost.__version__, device="cpu")
    g4 = weather_signature(inputs, upstream) if view == WEATHER else None
    context = dict(phase="phase4-frozen-holdout", input_sha256=upstream, weather_sha256=g4,
                   model=arm, view=view, seed=seed, protocol=protocol, environment=hardware,
                   code_sha256=file_hash(__file__), config_sha256=file_hash(inputs.config_path))
    store = compatible_store(inputs.path("output_dir") / "final" / arm / str(seed), context, Path(__file__))
    with exclusive(store.root):
        fitted = store.load("fit")
        if fitted is None:
            train = inputs.rows(11, "train", view)
            transformer = inputs.transformer(view, train)
            matrix = transformer.transform(train)
            y_train = train["target_class"].to_numpy()
            train_ids = ids_hash(train[ID])
            original = read_json(Path("reports/phase3/experiment_manifest.json"))
            if train_ids != original["frozen_fit_hashes"]["xgboost/G1_G2_G3"]["train_ids_hash"]:
                raise ValueError("Final training IDs differ from the Phase III reference")
            del train
            attempt = uuid4().hex
            root = store.root / "attempts" / attempt
            timestamp = perf_counter()
            curves = []
            status = root / "status.json"
            write_json(status, dict(state="running", arm=arm, seed=seed, curves=curves))

            def on_epoch(epoch):
                curves.append(epoch)
                write_json(status, dict(state="running", arm=arm, seed=seed, curves=curves))

            if neural:
                estimator, _, _ = mlp.train(matrix, y_train, protocol["candidate"], inputs.mlp,
                                             seed, device, protocol["epochs"], weighted=weighted,
                                             on_epoch=on_epoch)
                model_path = root / "model.pt"
                atomic(model_path, lambda p: mlp.torch.save(estimator.state_dict(), p))
                model_sha = mlp.state_sha(estimator)
                weights = mlp.training_weights(y_train).tolist() if weighted else None
            else:
                from ..models.xgboost import create

                estimator = create(protocol["params"], seed, protocol["threads"]).fit(matrix, y_train)
                if estimator.classes_.tolist() != [0, 1, 2]:
                    raise ValueError("XGBoost class order changed")
                model_path = root / "model.ubj"
                atomic(model_path, estimator.save_model)
                model_sha, weights = booster_hash(estimator), None
            transform_path = root / "transformer.joblib"
            write_model(transform_path, transformer)
            write_json(status, dict(state="complete", arm=arm, seed=seed, curves=curves))
            store.save("fit", dict(attempt=attempt, train_ids_sha256=train_ids, rows=len(y_train),
                                   train_class_counts=np.bincount(y_train, minlength=3).tolist(),
                                   model_hash=model_sha, transformer_hash=state_hash(transformer),
                                   training_seconds=perf_counter() - timestamp,
                                   epochs=protocol.get("epochs"), weights=weights,
                                   curves=curves), (model_path, transform_path, status))
            fitted = store.load("fit")
            del matrix, y_train
        root = store.root / "attempts" / fitted["attempt"]
        transformer = joblib.load(root / "transformer.joblib")
        if neural:
            estimator = mlp.create(len(transformer.feature_names_out_), protocol["candidate"], seed,
                                   inputs.mlp["threads"], device)
            estimator.load_state_dict(mlp.torch.load(root / "model.pt", map_location=device, weights_only=True))
            get_sha = lambda: mlp.state_sha(estimator)
            predict = lambda X: mlp.predict(estimator, X, inputs.mlp["batch_size"], device)
        else:
            from ..models.xgboost import create

            estimator = create(protocol["params"], seed, protocol["threads"])
            estimator.load_model(root / "model.ubj")
            get_sha = lambda: booster_hash(estimator)
            predict = estimator.predict_proba
        for month in HOLDOUT:
            key = f"month-{month}"
            if store.load(key) is not None:
                continue
            frame = inputs.rows(month, "test", view)
            actual = predict(transformer.transform(frame))
            y = frame["target_class"].to_numpy()
            ids = frame[ID].to_numpy()
            check_probabilities(actual, len(y))
            state = (get_sha(), state_hash(transformer))
            if state != (fitted["model_hash"], fitted["transformer_hash"]):
                raise ValueError("Fitted model or transformation changed during holdout")
            path = store.root / f"predictions-{month}.parquet"
            write_frame(path, prediction_frame(store.signature, arm, view, "frozen", month,
                                               ids, y, actual, digest(fitted)))
            store.save(key, dict(month=month, fit_hash=digest(fitted), rows=len(y),
                                 ids_sha256=ids_hash(ids), model_hash=state[0],
                                 transformer_hash=state[1]), (path,))
    return store


def load_predictions(store, month):
    record, fit = store.load(f"month-{month}"), store.load("fit")
    if not record or not fit or record["fit_hash"] != digest(fit) or any(
        record[key] != fit[key] for key in ("model_hash", "transformer_hash")
    ):
        raise ValueError("Month lacks matching frozen-fit evidence")
    frame = pl.read_parquet(store.root / f"predictions-{month}.parquet")
    if frame.height != record["rows"] or ids_hash(frame[ID]) != record["ids_sha256"]:
        raise ValueError("Prediction ID checksum differs")
    p, y = frame.select(PROB_COLUMNS).to_numpy(), frame["y_true"].to_numpy()
    check_probabilities(p, len(y))
    return frame, record
