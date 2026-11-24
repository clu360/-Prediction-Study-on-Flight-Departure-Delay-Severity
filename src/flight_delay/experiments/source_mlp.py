"""Follow-up MLP experiment with saved chronological selection and frozen fits."""

from pathlib import Path
from statistics import median
from time import perf_counter, sleep
from uuid import uuid4

import joblib
import numpy as np
import polars as pl
import torch

from .config import digest, file_hash
from .evaluation import check_probabilities, evaluate
from .phase4_final import load_predictions, risk_metrics, selection_config, weather_signature
from .phase4_inputs import BASE, DEV, HOLDOUT, ID, WEATHER
from .phase4_mlp import candidate_rank, environment
from .phase4_results import reporting_checkpoints
from .results import (
    PROB_COLUMNS, ResultStore, atomic, exclusive, prediction_frame, read_json, write_frame,
    write_json, write_model,
)
from .runner import ids_hash
from ..models import mlp, source_mlp
from ..models.state import state_hash

PUBLIC = Path("reports/mlp_source")
PRIVATE = Path("data/model_outputs/mlp_source")
VIEWS = (BASE, WEATHER)
ARMS = {BASE: ("MLP-A", "MLP-B"), WEATHER: ("MLP-C", "MLP-D")}
REGIMES = ("fixed", "tuned")


def read_retry(operation, *args, **kwargs):
    for attempt in range(4):
        try:
            return operation(*args, **kwargs)
        except OSError as error:
            if attempt == 3 or not (isinstance(error, PermissionError) or "os error 32" in str(error)):
                raise
            print(f"Transient data read error, retry {attempt + 1}/3", flush=True)
            sleep(2 ** attempt)


def locked_weather(inputs, scope, upstream):
    if scope == "all":
        return read_retry(weather_signature, inputs, upstream)
    root = inputs.path("weather_dir")
    gate = read_json(root / "audit/verified.json")
    published = read_json(Path("reports/phase4/weather_g4_verification.json"))
    if gate["signature"] != published["signature"] or digest(gate["context"]) != gate["signature"]:
        raise ValueError("The G4 development source differs from the verified Phase IV archive")
    files = {name: value for name, value in gate["files"].items()
             if int(Path(name).parent.name.split("=")[1]) <= 10}
    if len(files) != 10 or any(read_retry(file_hash, root / name) != value for name, value in files.items()):
        raise ValueError("G4 development bytes failed verification")
    return digest(files)


def context(inputs, view, scope, device):
    upstream = read_retry(inputs.verify_source, scope)
    files = (
        Path(__file__), Path(source_mlp.__file__), Path(mlp.__file__),
        Path(__file__).with_name("phase4_inputs.py"),
        Path(__file__).with_name("phase4_mlp.py"),
    )
    code = {p.name: digest(p.read_text(encoding="utf-8")) for p in files}
    return dict(phase="mlp-source-inputs", scope=scope, view=view, input_sha256=upstream,
                g4=locked_weather(inputs, scope, upstream) if view == WEATHER else None,
                code=code, settings=inputs.mlp, environment=environment(device))


def matrices(inputs, view, train, test):
    adapter = source_mlp.SourceAdapter.fit(inputs.columns[view], train)
    return adapter, adapter.transform(train), train["target_class"].to_numpy(), adapter.transform(test)


def training(store, key, adapter, matrix, labels, candidate, settings, seed, device,
             epochs, *, stop=None, weighted=False):
    attempt = store.root / "attempts" / uuid4().hex
    path = attempt / "status.json"
    curves = []
    started = perf_counter()

    def status(state, error=None):
        write_json(path, dict(key=key, state=state, curves=curves, error=error,
                              seconds=perf_counter() - started))

    def on_epoch(row):
        curves.append(row)
        status("running")

    status("running")
    try:
        model, count, _ = mlp.train(
            matrix, labels, candidate, settings, seed, device, epochs,
            weighted=weighted, stop=stop, model_factory=adapter.factory(), on_epoch=on_epoch,
        )
    except Exception as error:
        status("failed", type(error).__name__)
        raise
    status("complete")
    return model, count, path, curves, perf_counter() - started


def evaluate_fold(inputs, store, view, month, indices, records, device):
    pending = []
    for index in indices:
        key = f"candidate-{index + 1}/month-{month:02}"
        record = store.load(key)
        if record is None:
            pending.append((index, key))
        else:
            records[index, month] = record
    if not pending:
        return
    gradient = read_retry(inputs.rows, month, "train", view, [month - 3, month - 2])
    stop = read_retry(inputs.rows, month, "train", view, [month - 1])
    validation = read_retry(inputs.rows, month, "test", view)
    outer_train = pl.concat([gradient, stop]).sort(ID)
    inner, x1, y1, x_stop = matrices(inputs, view, gradient, stop)
    outer, x2, y2, x_val = matrices(inputs, view, outer_train, validation)
    ids = {name: ids_hash(frame[ID]) for name, frame in (
        ("gradient", gradient), ("stop", stop), ("refit", outer_train), ("score", validation)
    )}
    y_stop, y_val = stop["target_class"].to_numpy(), validation["target_class"].to_numpy()
    del gradient, stop, outer_train, validation
    settings = inputs.mlp
    seed, batch_size = settings["development_seed"], settings["batch_size"]
    for index, key in pending:
        candidate = settings["candidates"][index]
        _, epochs, inner_status, inner_curve, inner_seconds = training(
            store, key + "/inner", inner, x1, y1, candidate, settings, seed, device,
            settings["max_epochs"], stop=(x_stop, y_stop),
        )
        model, _, outer_status, outer_curve, outer_seconds = training(
            store, key + "/refit", outer, x2, y2, candidate, settings, seed, device, epochs,
        )
        p = mlp.predict(model, x_val, batch_size, device)
        check_probabilities(p, len(y_val))
        metrics = evaluate(y_val, p)
        record = dict(candidate=index + 1, month=month, epochs=epochs, ids=ids,
                      seconds=inner_seconds + outer_seconds,
                      metrics={name: metrics[name] for name in ("log_loss", "macro_f1")},
                      inner_curve=inner_curve, outer_curve=outer_curve,
                      model_hash=mlp.state_sha(model), adapter_hash=state_hash(outer))
        store.save(key, record, (inner_status, outer_status))
        records[index, month] = store.load(key)
        print(f"{view} {key} complete: {record['metrics']}, epochs={epochs}", flush=True)


def tune(inputs, view, device):
    provenance = context(inputs, view, "development", device)
    store = ResultStore(PRIVATE / "development" / view, provenance, resume=True)
    settings = inputs.mlp
    candidates, screen, months = (settings[k] for k in ("candidates", "screening_months", "development_months"))
    if screen != [4, 7, 10] or months != list(DEV):
        raise ValueError("Unapproved development calendar")
    records = {}
    with exclusive(store.root):
        for month in screen:
            evaluate_fold(inputs, store, view, month, range(len(candidates)), records, device)
        finalists = candidate_rank(records, candidates, screen, range(len(candidates)))[:2]
        for month in sorted(set(months) - set(screen)):
            evaluate_fold(inputs, store, view, month, finalists, records, device)
        winner = candidate_rank(records, candidates, months, finalists)[0]
        rows = [dict(candidate=index + 1, month=month, epochs=row["epochs"], **row["metrics"],
                     seconds=row["seconds"], checkpoint_sha256=digest(row))
                for (index, month), row in sorted(records.items())]
        summary = PUBLIC / f"tuning_{view}.csv"
        write_frame(summary, pl.DataFrame(rows))
        lock = dict(candidate=candidates[winner], candidate_id=winner + 1,
                    epochs=int(median(records[winner, month]["epochs"] for month in months)),
                    finalists=[i + 1 for i in finalists], development=provenance,
                    signature=store.signature, tuning_sha256=file_hash(summary),
                    mean_log_loss=float(np.mean([records[winner, m]["metrics"]["log_loss"] for m in months])))
        lock["sha256"] = digest(lock)
        path = PUBLIC / f"selection_{view}.json"
        if path.exists() and read_json(path) != lock:
            raise ValueError("An incompatible development lock already exists")
        write_json(path, lock)
    return lock


def protocol(inputs, view, regime):
    if regime == "fixed":
        prior = selection_config(inputs)
        return {name: prior[name] for name in ("candidate", "candidate_id", "epochs", "sha256")}
    path = PUBLIC / f"selection_{view}.json"
    choice = read_json(path)
    signature = choice.pop("sha256")
    if (digest(choice) != signature or choice["development"]["view"] != view
            or choice["tuning_sha256"] != file_hash(PUBLIC / f"tuning_{view}.csv")):
        raise ValueError("Development selection evidence failed verification")
    return choice | {"sha256": signature}


def final_fit(inputs, view, regime, arm, seed, device, provenance):
    weighted = arm == ARMS[view][1]
    selected = protocol(inputs, view, regime)
    candidate, epochs = selected["candidate"], selected["epochs"]
    ctx = provenance | dict(arm=arm, regime=regime, seed=seed, protocol=selected, weighted=weighted)
    store = ResultStore(PRIVATE / "final" / view / regime / arm / str(seed), ctx, resume=True)
    with exclusive(store.root):
        fit = store.load("fit")
        if fit is None:
            train = read_retry(inputs.rows, 11, "train", view)
            train_hash = ids_hash(train[ID])
            original = read_json(Path("reports/phase3/experiment_manifest.json"))
            if train_hash != original["frozen_fit_hashes"]["xgboost/G1_G2_G3"]["train_ids_hash"]:
                raise ValueError("Training IDs do not match the locked pre-existing population")
            adapter = source_mlp.SourceAdapter.fit(inputs.columns[view], train)
            labels = train["target_class"].to_numpy()
            model, _, status, curves, seconds = training(
                store, "fit", adapter, adapter.transform(train), labels, candidate, inputs.mlp,
                seed, device, epochs, weighted=weighted,
            )
            del train
            root = status.parent
            model_path, adapter_path = root / "model.pt", root / "adapter.joblib"
            atomic(model_path, lambda path: torch.save(model.state_dict(), path))
            write_model(adapter_path, adapter)
            store.save("fit", dict(attempt=root.name, train_ids_sha256=train_hash, rows=len(labels),
                                   model_hash=mlp.state_sha(model), transformer_hash=state_hash(adapter),
                                   class_counts=np.bincount(labels, minlength=3).tolist(), curves=curves,
                                   weights=mlp.training_weights(labels).tolist() if weighted else None,
                                   epochs=epochs, seconds=seconds), (model_path, adapter_path, status))
            fit = store.load("fit")
        root = store.root / "attempts" / fit["attempt"]
        adapter = joblib.load(root / "adapter.joblib")
        model = adapter.factory()(adapter.input_width, candidate, seed, inputs.mlp["threads"], device)
        model.load_state_dict(torch.load(root / "model.pt", map_location=device, weights_only=True))
        for month in HOLDOUT:
            key = f"month-{month}"
            if store.load(key) is not None:
                continue
            frame = read_retry(inputs.rows, month, "test", view)
            p = mlp.predict(model, adapter.transform(frame), inputs.mlp["batch_size"], device)
            if (mlp.state_sha(model), state_hash(adapter)) != (fit["model_hash"], fit["transformer_hash"]):
                raise ValueError("Frozen fitted model or input adapter changed")
            ids, y = frame[ID].to_numpy(), frame["target_class"].to_numpy()
            artifact = store.root / f"predictions-{month}.parquet"
            write_frame(artifact, prediction_frame(store.signature, arm, view, "frozen", month,
                                                   ids, y, p, digest(fit)))
            store.save(key, dict(month=month, rows=len(y), fit_hash=digest(fit), ids_sha256=ids_hash(ids),
                                 model_hash=fit["model_hash"], transformer_hash=fit["transformer_hash"]),
                       (artifact,))
    print(f"{view} {regime} {arm} seed={seed} verified: {store.signature}", flush=True)


def publish(inputs):
    historical = reporting_checkpoints(inputs)
    expected = {(view, regime, arm, seed) for view in VIEWS for regime in REGIMES
                for arm in ARMS[view] for seed in inputs.mlp["final_seeds"]}
    rows, fits, completed = [], [], set()
    for path in sorted((PRIVATE / "final").glob("*/*/*/*/*/context.json")):
        ctx = read_json(path)
        unit = (ctx["view"], ctx["regime"], ctx["arm"], ctx["seed"])
        store = ResultStore(path.parent.parent, ctx, resume=True)
        if not all(store.load(f"month-{month}") for month in HOLDOUT):
            continue
        if unit not in expected or unit in completed:
            raise ValueError("Unexpected or duplicate finished run")
        completed.add(unit)
        view, regime, arm, seed = unit
        reference = historical[arm, seed]
        fit = store.load("fit")
        reference_fit = reference.load("fit")
        if (fit["train_ids_sha256"], fit["weights"]) != (reference_fit["train_ids_sha256"], reference_fit["weights"]):
            raise ValueError("Training IDs or loss weights differ from matched Phase IV reference")
        pairs, month_hashes = {}, {}
        for month in HOLDOUT:
            actual, record = load_predictions(store, month)
            prior, _ = load_predictions(reference, month)
            if not actual.select(ID, "y_true").equals(prior.select(ID, "y_true")):
                raise ValueError("Holdout IDs or labels differ from Phase IV")
            for frame in (actual, prior):
                if not np.array_equal(frame["y_pred"].to_numpy(), frame.select(PROB_COLUMNS).to_numpy().argmax(axis=1)):
                    raise ValueError("Saved class differs from probability argmax")
            pairs[str(month)] = (actual, prior)
            month_hashes[str(month)] = dict(ids_sha256=record["ids_sha256"], record_sha256=digest(record))
        pairs["pooled"] = tuple(pl.concat(parts) for parts in zip(*pairs.values()))
        for month, pair in pairs.items():
            vals = [risk_metrics(frame["y_true"].to_numpy(), frame.select(PROB_COLUMNS).to_numpy())
                    for frame in pair]
            new, old = vals
            fields = ("accuracy", "macro_f1", "weighted_f1", "log_loss", "brier", "macro_auc",
                      "severe_precision", "severe_recall", "severe_pr_auc_ap")
            rows.append(dict(view=view, regime=regime, arm=arm, seed=seed, month=month, n=new["n"],
                             **{k: new[k] for k in fields},
                             **{f"delta_{k}": new[k] - old[k] for k in fields},
                             severe_prevalence=new["observed_proportions"][2],
                             mean_severe_probability=new["mean_probabilities"][2],
                             severe_argmax_share=new["argmax_proportions"][2]))
        fits.append(dict(context=ctx, fit={k: v for k, v in fit.items() if k != "artifacts"},
                         signature=store.signature, months=month_hashes))
    if not rows:
        raise ValueError("No complete frozen fits")
    detail = pl.DataFrame(rows).sort("view", "regime", "arm", "seed", "month")
    keys = ["view", "regime", "arm", "month"]
    values = [name for name in detail.columns if name not in (*keys, "seed", "n")]
    rollup = detail.group_by(keys).agg(pl.len().alias("seeds"), pl.col("n").first(),
                                     *[pl.col(name).mean() for name in values],
                                     pl.col("macro_f1").std().alias("macro_f1_seed_sd")).sort(keys)
    write_frame(PUBLIC / "metrics_by_seed.csv", detail)
    write_frame(PUBLIC / "metrics_summary.csv", rollup)
    manifest = dict(status="complete" if completed == expected else "partial",
                    completed=len(completed), expected=len(expected), fits=fits,
                    metrics_sha256=file_hash(PUBLIC / "metrics_by_seed.csv"))
    write_json(PUBLIC / "manifest.json", manifest)
    return manifest
