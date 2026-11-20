"""Validate private prediction checkpoints and publish bounded aggregate tables."""

import numpy as np
import polars as pl

from .config import file_hash
from .phase4_final import ARMS, load_predictions, metadata, subgroup_scores, xgb_config
from .phase4_inputs import HOLDOUT, ID
from .results import PROB_COLUMNS, ResultStore, exclusive, read_json, write_frame, write_json
from .runner import ids_hash

METRICS = ("accuracy", "macro_f1", "weighted_f1", "log_loss", "brier", "macro_auc",
           "severe_precision", "severe_recall", "severe_pr_auc_ap")
RATIOS = ("actual_severe_prevalence", "mean_predicted_severe_probability", "argmax_severe_share")


def reporting_checkpoints(inputs):
    root = inputs.path("output_dir") / "final"
    by_arm = {}
    xgb_seed = xgb_config()["seed"]
    for arm in ARMS:
        seeds = inputs.mlp["final_seeds"] if arm.startswith("MLP") else [xgb_seed]
        for seed in seeds:
            checkpoint_root = root / arm / str(seed)
            contexts = sorted(checkpoint_root.glob("*/context.json"))
            complete = []
            for path in contexts:
                store = ResultStore(checkpoint_root, read_json(path), resume=True)
                if store.root.name != path.parent.name:
                    raise ValueError("Mismatched checkpoint folder name")
                if all(store.load(f"month-{month}") for month in HOLDOUT):
                    complete.append(store)
            if len(complete) > 1:
                raise ValueError(f"Multiple completed science results for {arm}/{seed}")
            if complete:
                by_arm[arm, seed] = complete[0]
    return by_arm


def checked_metadata(inputs, month, frame, weather):
    meta = metadata(inputs, month, frame.select(ID))
    if weather is not None:
        values = pl.read_parquet(weather / "g4_features" / "year=2025" / f"month={month:02}" / "part-00.parquet")
        if values.height != meta.height or meta.select(ID).join(values.select(ID), on=ID, how="anti").height:
            raise ValueError("Weather strata IDs do not match frozen holdout")
        meta = meta.join(values, on=ID, how="left", validate="1:1", maintain_order="left")
    return meta


def publish(inputs):
    from .phase4_final import weather_signature

    root = inputs.path("reports_dir")
    stores = reporting_checkpoints(inputs)
    if not stores:
        raise ValueError("No real-data final checkpoints have passed")
    upstream = inputs.verify_source("all")
    weather_root = inputs.path("weather_dir")
    if (weather_root / "audit" / "verified.json").exists():
        weather_sha = weather_signature(inputs, upstream)
    else:
        weather_root, weather_sha = None, None
    shared = {}
    row_records, detailed, provenance = [], {}, []
    for (arm, seed), store in stores.items():
        fit = store.load("fit")
        if store.context["input_sha256"] != upstream:
            raise ValueError("Stale frozen-study inputs")
        expected_weights = np.array(fit["train_class_counts"])
        if ARMS[arm][1]:
            expected_weights = expected_weights.sum() / (3 * expected_weights)
            if not np.allclose(fit["weights"], expected_weights):
                raise ValueError("Weighted fit does not match its training-class weights")
        elif fit["weights"] is not None:
            raise ValueError("Unweighted fit contains weights")
        if shared.setdefault("train", fit["train_ids_sha256"]) != fit["train_ids_sha256"]:
            raise ValueError("Training ID hashes differ across arms")
        scored = {}
        for month in HOLDOUT:
            frame, record = load_predictions(store, month)
            if shared.setdefault(str(month), record["ids_sha256"]) != record["ids_sha256"]:
                raise ValueError("Evaluation ID hashes differ across arms")
            scored[str(month)] = (frame, checked_metadata(inputs, month, frame, weather_root))
        scored["pooled"] = tuple(pl.concat(pair) for pair in zip(*scored.values()))
        for month, (frame, meta) in scored.items():
            y, p = frame["y_true"].to_numpy(), frame.select(PROB_COLUMNS).to_numpy()
            results = subgroup_scores(y, p, meta, inputs.weather["features"])
            for subgroup, metrics in results.items():
                common = dict(arm=arm, seed=seed, month=month, subgroup=subgroup,
                              n=metrics["n"], view=ARMS[arm][0],
                              fit_id=fit["record_hash"], epochs=fit["epochs"],
                              train_ids_sha256=fit["train_ids_sha256"],
                              test_ids_sha256=ids_hash(frame[ID]), input_sha256=upstream)
                row_records.append(dict(common, **{key: metrics[key] for key in METRICS},
                                        actual_severe_prevalence=metrics["observed_proportions"][2],
                                        mean_predicted_severe_probability=metrics["mean_probabilities"][2],
                                        argmax_severe_share=metrics["argmax_proportions"][2]))
                detailed[f"{arm}/{seed}/{month}/{subgroup}"] = metrics
        provenance.append(dict(arm=arm, seed=seed, signature=store.signature, context=store.context,
                               fit={k: v for k, v in fit.items() if k != "artifacts"},
                               month_checkpoints={str(month): store.load(f"month-{month}")["record_hash"]
                                                  for month in HOLDOUT}))
    rows = pl.DataFrame(row_records, infer_schema_length=None)
    expected_units = {(arm, seed) for arm in ARMS for seed in (
        inputs.mlp["final_seeds"] if arm.startswith("MLP") else [xgb_config()["seed"]])}
    complete = set(stores) == expected_units
    seed_summary = rows.group_by("arm", "month", "subgroup").agg(
        pl.len().alias("seed_count"), *[pl.col(m).mean().alias(m) for m in (*METRICS, *RATIOS)],
        *[pl.col(m).std().alias(f"{m}_seed_sd") for m in METRICS],
    ).sort("arm", "month", "subgroup")
    factorial = []
    for key, group in rows.filter(pl.col("arm").str.starts_with("MLP")).group_by("seed", "month", "subgroup"):
        arms = {row["arm"][-1]: row for row in group.iter_rows(named=True)}
        if set(arms) != set("ABCD"):
            continue
        for metric in (*METRICS, *RATIOS):
            if any(row[metric] is None for row in arms.values()):
                continue
            a, b, c, d = (arms[arm][metric] for arm in "ABCD")
            factorial.append(dict(seed=key[0], month=key[1], subgroup=key[2], metric=metric,
                                  B_minus_A=b-a, C_minus_A=c-a, D_minus_C=d-c,
                                  D_minus_B=d-b, interaction=(d-c)-(b-a)))
    with exclusive(root):
        write_frame(root / "metrics_summary.csv", rows.sort("arm", "seed", "month", "subgroup"))
        write_frame(root / "seed_summary.csv", seed_summary)
        write_json(root / "metric_details.json", detailed)
        if factorial:
            write_frame(root / "factorial_contrasts.csv", pl.DataFrame(factorial).sort(
                "seed", "month", "subgroup", "metric"))
        out = dict(status="all_model_checkpoints_verified" if complete else "partial_model_checkpoints",
                   class_order=[0, 1, 2], shared_population=shared, weather_sha256=weather_sha,
                   input_sha256=upstream, fits=provenance,
                   public_metrics_sha256=file_hash(root / "metrics_summary.csv"),
                   exploratory_weather_and_weighting=True)
        write_json(root / "experiment_manifest.json", out)
    return out
