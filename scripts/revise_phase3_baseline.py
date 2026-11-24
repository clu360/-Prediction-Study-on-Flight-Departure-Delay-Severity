"""Re-score only the baseline under an explicitly post-holdout CDF revision.

The original signed experiment, selection locks, learned fits and predictions
are read-only. Current comparison reports overlay the revised baseline only.
"""

import argparse
from copy import deepcopy
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
import polars as pl
from flight_delay.models.baseline import CDF_RULE, sampled_classes
from flight_delay.experiments.config import (
    Phase3Config,
    analysis_fingerprint,
    code_fingerprints,
    digest,
    file_hash,
)
from flight_delay.experiments.evaluation import evaluate, airport_metrics
from flight_delay.experiments.inputs import gate_path, origins_for
from flight_delay.experiments.results import (
    ResultStore,
    PROB_COLUMNS,
    read_json,
    write_frame,
    atomic,
    exclusive,
)
from flight_delay.experiments.runner import baseline_rows


def write_json(path, value):
    """Canonical UTF-8/LF exports keep cross-platform report diffs reviewable."""
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path = Path(path).resolve()
    if os.name == "nt" and not str(path).startswith("\\\\?\\"):
        path = Path("\\\\?\\" + str(path))
    atomic(path, lambda temporary: temporary.write_bytes(payload.encode("utf-8")))


def read_with_retry(operation):
    """Retry only transient Windows sharing violations, never invalid data."""
    for attempt in range(5):
        try:
            return operation()
        except OSError as error:
            sharing = getattr(error, "winerror", None) == 32 or "os error 32" in str(
                error
            )
            if not sharing or attempt == 4:
                raise
            time.sleep(0.25 * 2**attempt)


def independent_classes(priors, ids, seed):
    """Independent scalar reference: first 53 SHA-256 bits and CDF search."""
    import bisect

    cdf = np.cumsum(priors).tolist()
    cdf[-1] = 1.0
    prefix = f"{CDF_RULE}:{seed}:".encode()
    return np.fromiter(
        (
            bisect.bisect_right(
                cdf,
                (
                    int.from_bytes(
                        hashlib.sha256(prefix + value.encode()).digest(), "big"
                    )
                    >> 203
                )
                / 2**53,
            )
            for value in ids
        ),
        dtype=np.int8,
    )


def snapshot_reports(reports, destination):
    destination.mkdir(parents=True, exist_ok=True)
    index = destination / "snapshot.json"
    if not index.exists():
        files = {}
        for path in sorted(reports.iterdir()):
            if path.suffix in {".json", ".csv"}:
                shutil.copyfile(path, destination / path.name)
                files[path.name] = file_hash(path)
        write_json(index, files)
    files = read_json(index)
    for name, expected in files.items():
        if file_hash(destination / name) != expected:
            raise ValueError("Original report snapshot changed")
    return files


def revise(cfg, source_run, apply=False):
    started = time.monotonic()
    if not source_run.startswith("phase3-") or Path(source_run).name != source_run:
        raise ValueError("Use a simple original run ID")
    original_root = cfg.path("output_dir") / source_run
    context = read_json(original_root / "context.json")
    source = ResultStore(cfg.path("output_dir"), context, True)
    if source.run_id != source_run:
        raise ValueError("Original run identity mismatch")
    gate = read_json(gate_path(cfg))
    if not gate["passed"] or gate["input_digest"] != context["input_digest"]:
        raise ValueError("Original upstream audit does not match the source run")
    private = cfg.path("output_dir") / "baseline_revisions" / CDF_RULE
    original_reports = private / "original_reports"
    original_hashes = snapshot_reports(cfg.path("reports_dir"), original_reports)
    old_manifest = read_json(original_reports / "experiment_manifest.json")
    if (
        old_manifest["run_id"] != source_run
        or old_manifest.get("report_revision_id")
        or old_manifest["completion_status"] != "complete"
    ):
        raise ValueError("Baseline revision must start from the original study reports")
    selection = read_json(original_reports / "selected_configs.json")
    if not selection["complete"] or len(selection["selected"]) != 13:
        raise ValueError(
            "The original complete thirteen-configuration selection is required"
        )
    if (original_reports / "selected_configs.json").read_bytes() != (
        original_root / "development/selected_configs.json"
    ).read_bytes():
        raise ValueError("Frozen selected configurations changed")
    print(
        "Verifying original target, fold and origin-metadata input hashes", flush=True
    )
    checked = {}
    for relative, expected in gate["input_files"].items():
        if any(f"/{table}/" in relative for table in ["targets", "folds", "metadata"]):
            if read_with_retry(lambda: file_hash(cfg.root / relative)) != expected:
                raise ValueError("Baseline input changed after the original audit")
            checked[relative] = expected
    if not checked:
        raise ValueError("No original baseline input fingerprints found")
    revision_id = (
        "baseline-cdf-"
        + digest(
            dict(
                source_run=source_run,
                rule=CDF_RULE,
                seed=cfg["seed"],
                original_reports=original_hashes,
            )
        )[:20]
    )
    output = private / revision_id
    output.mkdir(parents=True, exist_ok=True)
    records, frames, origin_arrays, prior_evidence = [], [], [], []
    final_priors = None
    prediction_rows = 0
    for month in range(4, 13):
        period = f"2025-{month:02}"
        record_key = (
            f"development/selected/baseline_independent_{period}"
            if month < 11
            else f"final/months/baseline_independent_{period}"
        )
        old = read_with_retry(lambda: source.load(record_key))
        if old is None:
            raise ValueError("Original baseline checkpoint missing")
        relative = (
            f"development/selected_predictions/baseline_independent_{period}.parquet"
            if month < 11
            else f"final/predictions/baseline_independent_{period}.parquet"
        )
        frame = read_with_retry(lambda: pl.read_parquet(source.root / relative))
        if month != 12:
            train = read_with_retry(lambda: baseline_rows(cfg, period, "train"))
            counts = np.bincount(train["target_class"].to_numpy(), minlength=3)
            priors = counts / counts.sum()
            priors /= priors.sum()
            prior_evidence.append(
                dict(
                    scored_period=period if month < 11 else "2025-11_and_2025-12",
                    training_months=list(range(month - 3, month)),
                    training_rows=train.height,
                    training_class_counts=counts.tolist(),
                    probabilities=priors.tolist(),
                    cdf=np.cumsum(priors).tolist(),
                )
            )
            if month == 11:
                final_priors = priors.copy()
        else:
            priors = final_priors
        expected = read_with_retry(lambda: baseline_rows(cfg, period, "test"))
        if (
            frame["target_record_id"].to_list()
            != expected["target_record_id"].to_list()
        ):
            raise ValueError("Original baseline prediction IDs differ from memberships")
        y = expected["target_class"].to_numpy()
        if not np.array_equal(y, frame["y_true"].to_numpy()):
            raise ValueError("Original baseline labels differ from targets")
        p = frame.select(PROB_COLUMNS).to_numpy()
        if not np.allclose(p, priors, atol=1e-15, rtol=0):
            raise ValueError(
                "Saved probabilities differ from training-only class priors"
            )
        ids = frame["target_record_id"].to_numpy()
        predicted = sampled_classes(p, ids, cfg["seed"])
        reference = independent_classes(priors, ids, cfg["seed"])
        if not np.array_equal(predicted, reference):
            raise ValueError("CDF samples differ from independent reference")
        revised = frame.with_columns(pl.Series("y_pred", predicted))
        path = output / f"predictions/{period}.parquet"
        write_frame(path, revised)
        replay = pl.read_parquet(path)
        if not replay.equals(revised):
            raise ValueError("Persisted revision predictions changed on reload")
        origins = read_with_retry(lambda: origins_for(cfg, ids, month))
        record = dict(
            old,
            prediction_rule=CDF_RULE,
            report_revision_id=revision_id,
            metrics=evaluate(y, p, predicted=predicted),
            airport_metrics=airport_metrics(
                y, p, origins, cfg["evaluation_airports"], predicted=predicted
            ),
        )
        cm = np.bincount(3 * y.astype(int) + reference, minlength=9).reshape(3, 3)
        if record["metrics"]["confusion_matrix"] != cm.tolist():
            raise ValueError("Independent classification confusion matrix differs")
        records.append(record)
        prediction_rows += len(y)
        if month >= 11:
            frames.append(revised)
            origin_arrays.append(origins)
        print(f"Verified CDF baseline {period}: {len(y):,} rows", flush=True)
    pooled = pl.concat(frames)
    if pooled["target_record_id"].n_unique() != pooled.height:
        raise ValueError("Duplicate pooled baseline IDs")
    y, p, predicted = (
        pooled["y_true"].to_numpy(),
        pooled.select(PROB_COLUMNS).to_numpy(),
        pooled["y_pred"].to_numpy(),
    )
    records.append(
        dict(
            records[-2],
            scored_month="2025-11_12_pooled",
            metrics=evaluate(y, p, predicted=predicted),
            airport_metrics=airport_metrics(
                y,
                p,
                np.concatenate(origin_arrays),
                cfg["evaluation_airports"],
                predicted=predicted,
            ),
        )
    )
    by_period = {row["scored_month"]: row for row in records}
    original_details = read_json(original_reports / "metric_details.json")
    details = deepcopy(original_details)
    for row in details:
        if row["model"] == "baseline":
            replacement = by_period[row["scored_month"]]
            for key in [
                "metrics",
                "airport_metrics",
                "prediction_rule",
                "report_revision_id",
            ]:
                row[key] = replacement[key]
    if [r for r in details if r["model"] != "baseline"] != [
        r for r in original_details if r["model"] != "baseline"
    ]:
        raise ValueError("Learned-model metrics changed")
    staged = output / "reports"
    staged.mkdir(exist_ok=True)
    write_json(staged / "metric_details.json", details)
    with (original_reports / "metrics_summary.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        reader = csv.DictReader(stream)
        columns, csv_rows = reader.fieldnames, list(reader)
    for row in csv_rows:
        if row["model"] == "baseline":
            record = by_period[row["scored_month"]]
            metrics = (
                record["metrics"]
                if row["airport"] == "overall"
                else record["airport_metrics"][row["airport"]]
            )
            for key in row.keys() & metrics.keys():
                value = metrics[key]
                row[key] = (
                    ""
                    if value is None
                    else str(value)
                    if isinstance(value, int)
                    else f"{value:.12f}"
                )
            row["provenance_hash"] = digest(revision_id)

    def write_csv(path):
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(csv_rows)

    atomic(staged / "metrics_summary.csv", write_csv)
    candidates = read_json(original_reports / "candidate_scores.json")
    for row in candidates:
        if row["model"] == "baseline":
            metrics = by_period[row["scored_month"]]["metrics"]
            row.update(macro_f1=metrics["macro_f1"], log_loss=metrics["log_loss"])
    write_json(staged / "candidate_scores.json", candidates)
    dev = [
        row["metrics"] for row in records if row["evaluation_stage"] == "development"
    ]
    verification = dict(
        passed=True,
        report_revision_id=revision_id,
        baseline_provenance_hash=digest(revision_id),
        original_run_id=source_run,
        prediction_rows=prediction_rows,
        holdout_prediction_rows=pooled.height,
        periods_verified=9,
        independent_full_population_cdf_replay=True,
        input_files_rehashed=len(checked),
        input_subset_digest=digest(checked),
        learned_model_metric_records_unchanged=sum(
            row["model"] != "baseline" for row in original_details
        ),
        selected_configuration_bytes_unchanged=True,
        probabilities_match_original_training_priors=True,
        pooled_predictions_reused_without_resampling=True,
        original_result_verification_scope="Original argmax study; revised baseline classes verified separately here",
    )
    write_json(staged / "baseline_revision_verification.json", verification)
    revision = dict(
        report_revision_id=revision_id,
        original_run_id=source_run,
        prediction_rule=CDF_RULE,
        post_holdout_revision=True,
        seed=cfg["seed"],
        seed_selected_using_results=False,
        uniform_method="First 53 bits of SHA256(prior_cdf_sha256_v1:6740:stable_ID) / 2^53; IDs only address random draws",
        cdf_intervals="[0,p0), [p0,p0+p1), [p0+p1,1)",
        probability_metrics="Unchanged historical probabilities; never one-hot sampled predictions",
        final_training_months=[8, 9, 10],
        frozen_for_months=[11, 12],
        prior_evidence=prior_evidence,
        learned_models_refitted=0,
        hyperparameters_reselected=False,
        original_selection_hash=selection["selection_hash"],
        original_report_sha256=original_hashes,
        report_source_code_hashes=code_fingerprints(cfg.root),
        report_analysis_code_hash=analysis_fingerprint(cfg.root),
        revised_report_sha256={
            name: file_hash(staged / name)
            for name in [
                "metrics_summary.csv",
                "metric_details.json",
                "candidate_scores.json",
                "baseline_revision_verification.json",
            ]
        },
        development_summary=dict(
            mean_macro_f1=float(np.mean([m["macro_f1"] for m in dev])),
            macro_f1_std=float(np.std([m["macro_f1"] for m in dev])),
            mean_log_loss=float(np.mean([m["log_loss"] for m in dev])),
        ),
        performance=[
            dict(
                evaluation_stage=row["evaluation_stage"],
                period=row["scored_month"],
                **{
                    k: row["metrics"][k]
                    for k in [
                        "n",
                        "accuracy",
                        "macro_f1",
                        "precision",
                        "recall",
                        "f1",
                        "support",
                        "severe_precision",
                        "severe_recall",
                        "log_loss",
                        "brier",
                        "predicted_proportions",
                    ]
                },
            )
            for row in records
        ],
        original_argmax_performance=[
            dict(
                period=row["scored_month"],
                accuracy=row["metrics"]["accuracy"],
                macro_f1=row["metrics"]["macro_f1"],
            )
            for row in original_details
            if row["model"] == "baseline"
        ],
        limits=[
            "Post-holdout user-requested reference change, not a preregistered comparison",
            "One fixed seeded draw realization; no seed or threshold search",
            "Original saved study and verification retained; only baseline reporting is replaced",
        ],
        elapsed_seconds=time.monotonic() - started,
    )
    write_json(staged / "baseline_revision.json", revision)
    manifest = dict(
        old_manifest,
        report_revision_id=revision_id,
        baseline_prediction_rule=CDF_RULE,
        baseline_revision_file="baseline_revision.json",
        baseline_revision_sha256=file_hash(staged / "baseline_revision.json"),
        baseline_revision_verification_sha256=file_hash(
            staged / "baseline_revision_verification.json"
        ),
        report_source_code_hashes=code_fingerprints(cfg.root),
        report_analysis_code_hash=analysis_fingerprint(cfg.root),
    )
    manifest["limits"] = old_manifest["limits"] + revision["limits"]
    write_json(staged / "experiment_manifest.json", manifest)
    if apply:
        for path in staged.iterdir():
            atomic(
                cfg.path("reports_dir") / path.name,
                lambda target, path=path: shutil.copyfile(path, target),
            )
    print(
        f"CDF baseline revision {'published' if apply else 'staged only'}: {revision_id}",
        flush=True,
    )
    return revision


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase3_pace.yaml"))
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    cfg = Phase3Config.load(args.config)
    with exclusive(cfg.path("output_dir") / "baseline_revisions" / CDF_RULE):
        revise(cfg, args.source_run, args.apply)
