"""Publish aggregate-only CSV readiness evidence; never publish prediction rows."""

from pathlib import Path
import argparse
import json
import polars as pl
from flight_delay.experiments.results import write_json, write_frame


def export(readiness: Path, reports: Path):
    result = json.loads(readiness.read_text(encoding="utf-8"))
    if (
        result.get("study_results") is not False
        or result.get("evaluation_stage") != "csv_readiness_smoke"
        or result.get("status") != "csv_models_verified"
    ):
        raise ValueError(
            "Only completed, explicitly non-study CSV readiness diagnostics can be exported"
        )
    rows, details = [], []
    for record in result["models"]:
        metrics = record["metrics"]
        rows.append(
            dict(
                model=record["model"],
                feature_view=record["feature_view"],
                evaluation_stage="csv_readiness_smoke",
                study_results=False,
                scored_month="2025-04",
                train_rows=result["train_rows"],
                test_rows=metrics["n"],
                **{
                    key: metrics[key]
                    for key in [
                        "accuracy",
                        "macro_f1",
                        "weighted_f1",
                        "severe_recall",
                        "severe_precision",
                        "log_loss",
                        "brier",
                        "macro_auc",
                    ]
                },
                **{
                    key: record[key]
                    for key in [
                        "preprocessing_seconds",
                        "fit_seconds",
                        "inference_seconds",
                        "peak_rss_bytes",
                    ]
                },
                warning_count=len(record["warnings"]),
            )
        )
        details.append(
            {
                key: record[key]
                for key in [
                    "model",
                    "feature_view",
                    "params",
                    "metrics",
                    "unknown_categories",
                    "warnings",
                    "train_shape",
                    "test_shape",
                ]
            }
        )
    frame = pl.DataFrame(rows, infer_schema_length=None).sort("model", "feature_view")
    if (
        frame.select("model", "feature_view").n_unique() != frame.height
        or frame.height != result["model_view_combinations"]
    ):
        raise ValueError("Duplicate or missing diagnostic model/view results")
    summary = {
        key: result[key]
        for key in [
            "status",
            "study_results",
            "evaluation_stage",
            "scope",
            "split",
            "rows_per_month",
            "original_fold_manifest_verified",
            "limitation",
            "train_rows",
            "test_rows",
            "model_view_combinations",
            "csv_verification",
        ]
    }
    summary["prediction_rows"] = frame["test_rows"].sum()
    summary["diagnostic_run_id"] = readiness.parent.name
    write_frame(reports / "csv_readiness_metrics.csv", frame)
    write_json(reports / "csv_readiness_summary.json", summary)
    write_json(reports / "csv_readiness_details.json", details)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    paths = list(
        (root / "data/model_outputs/phase3/csv_readiness").glob("*/readiness.json")
    )
    readiness = args.readiness or (
        max(paths, key=lambda p: p.stat().st_mtime) if paths else None
    )
    if readiness is None:
        raise ValueError("No completed CSV readiness diagnostic is available")
    summary = export(readiness, root / "reports/phase3")
    print(
        json.dumps(
            dict(
                model_rows=summary["model_view_combinations"],
                prediction_rows=summary["prediction_rows"],
                study_results=False,
            )
        )
    )


if __name__ == "__main__":
    main()
