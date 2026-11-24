"""Validate private prediction checkpoints and publish bounded aggregate tables."""

from pathlib import Path

import numpy as np
import polars as pl

from .config import digest, file_hash
from .phase4_final import ARMS, load_predictions, metadata, subgroup_scores, xgb_config
from .inputs import memberships, table_scan
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
    labels = {}
    for month in HOLDOUT:
        expected = memberships(inputs.path("processed_dir"), f"2025-{month:02}", "test").sort(ID)
        target = table_scan(inputs.path("features_dir"), "targets", [month]).select(ID, "target_class").collect()
        labels[month] = expected.join(target, on=ID, how="left", validate="1:1",
                                      maintain_order="left").rename({"target_class": "y_true"})
        if labels[month]["y_true"].null_count():
            raise ValueError("Canonical holdout labels do not align with their published fold")
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
            if not frame.select(ID, "y_true").equals(labels[month], null_equal=False):
                # Prediction files use Int8 and upstream schemas may use another integer width.
                if not frame.select(ID, pl.col("y_true").cast(pl.Int64)).equals(
                    labels[month].with_columns(pl.col("y_true").cast(pl.Int64)), null_equal=False
                ):
                    raise ValueError("Frozen predictions differ from exact holdout IDs or labels")
            if not np.array_equal(frame["y_pred"].to_numpy(), frame.select(PROB_COLUMNS).to_numpy().argmax(axis=1)):
                raise ValueError("Saved learned-model classes differ from probability argmax")
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
                              fit_id=digest(fit), epochs=fit["epochs"],
                              train_ids_sha256=fit["train_ids_sha256"],
                              test_ids_sha256=ids_hash(frame[ID]), input_sha256=upstream)
                row_records.append(dict(common, **{key: metrics[key] for key in METRICS},
                                        actual_severe_prevalence=metrics["observed_proportions"][2],
                                        mean_predicted_severe_probability=metrics["mean_probabilities"][2],
                                        argmax_severe_share=metrics["argmax_proportions"][2]))
                detailed[f"{arm}/{seed}/{month}/{subgroup}"] = {
                    key: value for key, value in metrics.items()
                    if key != "severe_pr_curve" or subgroup == "overall"
                }
        checkpoint_hashes = {}
        for month in HOLDOUT:
            record = store.load(f"month-{month}")
            checkpoint_hashes[str(month)] = dict(record_sha256=digest(record),
                                                predictions_sha256=record["artifacts"][f"predictions-{month}.parquet"])
        provenance.append(dict(arm=arm, seed=seed, signature=store.signature, context=store.context,
                               fit={k: v for k, v in fit.items() if k != "artifacts"},
                               month_checkpoints=checkpoint_hashes))
    rows = pl.DataFrame(row_records, infer_schema_length=None)
    expected_units = {(arm, seed) for arm in ARMS for seed in (
        inputs.mlp["final_seeds"] if arm.startswith("MLP") else [xgb_config()["seed"]])}
    complete = set(stores) == expected_units
    seed_summary = rows.group_by("arm", "month", "subgroup").agg(
        pl.len().alias("seed_count"), *[pl.col(m).mean().alias(m) for m in (*METRICS, *RATIOS)],
        *[pl.col(m).std().alias(f"{m}_seed_sd") for m in METRICS],
    ).sort("arm", "month", "subgroup")
    contrasts = []
    reference = pl.read_csv(Path("reports/phase3/metrics_summary.csv"), schema_overrides={"scored_month": pl.String})
    historical = reference.filter(
        (pl.col("model") == "xgboost") & (pl.col("feature_view") == "G1_G2_G3")
        & (pl.col("airport") == "overall") & (pl.col("evaluation_stage") == "frozen_holdout")
    )
    for month in ("11", "12", "pooled"):
        data = seed_summary.filter((pl.col("month") == month) & (pl.col("subgroup") == "overall"))
        matched = {r["arm"]: r for r in data.iter_rows(named=True)}
        old_month = "2025-11_12_pooled" if month == "pooled" else f"2025-{month}"
        old = historical.filter(pl.col("scored_month") == old_month)
        if old.height == 1:
            matched["XGB-phase3-reference"] = old.row(0, named=True)
        for left, right in (("XGB-weather", "XGB-base"), ("MLP-A", "XGB-base"),
                            ("XGB-base", "XGB-phase3-reference"), ("MLP-A", "XGB-phase3-reference")):
            if left in matched and right in matched:
                for metric in ("macro_f1", "log_loss", "severe_precision", "severe_recall"):
                    contrasts.append(dict(month=month, arm=left, reference=right, metric=metric,
                                          raw_difference=matched[left][metric] - matched[right][metric]))
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
        if contrasts:
            write_frame(root / "model_contrasts.csv", pl.DataFrame(contrasts))
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
