"""Export complete source-input MLP class/risk metrics from frozen predictions.

No fitting or model selection occurs. Existing headline reports and scientific
checkpoints remain unchanged; this fills the missing notebook detail aggregate.
"""

import csv
import json
from pathlib import Path
import time
import numpy as np
import polars as pl
from flight_delay.experiments.config import file_hash
from flight_delay.experiments.phase4_final import load_predictions, risk_metrics
from flight_delay.experiments.results import ResultStore, PROB_COLUMNS, atomic


def export(root):
    start = time.monotonic()
    public = root / "reports/mlp_source"
    manifest_path = public / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verification = json.loads(
        (public / "holdout_verification.json").read_text(encoding="utf-8")
    )
    if (
        manifest["status"] != "complete"
        or manifest["completed"] != 24
        or manifest["expected"] != 24
    ):
        raise ValueError("All 24 original fits must be complete")
    if verification["status"] != "passed" or verification[
        "manifest_sha256"
    ] != file_hash(manifest_path):
        raise ValueError(
            "Source-input holdout verification does not match the manifest"
        )
    with (public / "metrics_by_seed.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected = {
        (r["view"], r["regime"], r["arm"], int(r["seed"]), r["month"]): r for r in rows
    }
    if len(expected) != 72:
        raise ValueError("Expected exactly 72 per-seed/month metric records")
    details, prediction_rows = {}, 0
    for unit in manifest["fits"]:
        ctx = unit["context"]
        base = (
            root
            / "data/model_outputs/mlp_source/final"
            / ctx["view"]
            / ctx["regime"]
            / ctx["arm"]
            / str(ctx["seed"])
        )
        store = ResultStore(base, ctx, True)
        if (
            store.signature != unit["signature"]
            or store.load("fit")["model_hash"] != unit["fit"]["model_hash"]
        ):
            raise ValueError("Signed source-input fit identity changed")
        frames = []
        for month in [11, 12]:
            frame, _ = load_predictions(store, month)
            frames.append(frame)
            prediction_rows += frame.height
        frames.append(pl.concat(frames))
        for month, frame in zip(["11", "12", "pooled"], frames):
            if frame["target_record_id"].n_unique() != frame.height:
                raise ValueError("Duplicate prediction IDs")
            y, p = frame["y_true"].to_numpy(), frame.select(PROB_COLUMNS).to_numpy()
            predicted = p.argmax(axis=1)
            if not np.array_equal(predicted, frame["y_pred"].to_numpy()):
                raise ValueError("Saved source-input classes differ from argmax")
            values = risk_metrics(y, p)
            prior = expected[ctx["view"], ctx["regime"], ctx["arm"], ctx["seed"], month]
            for name, value in values.items():
                if (
                    name in prior
                    and prior[name] != ""
                    and isinstance(value, (int, float))
                ):
                    if not np.isclose(
                        value, float(prior[name]), atol=1e-10, rtol=1e-10
                    ):
                        raise ValueError(
                            f"Published source-input metric changed: {name}"
                        )
            confusion = np.bincount(3 * y.astype(int) + predicted, minlength=9).reshape(
                3, 3
            )
            if values["confusion_matrix"] != confusion.tolist():
                raise ValueError("Independent class-count reconstruction failed")
            details[
                f"{ctx['view']}/{ctx['regime']}/{ctx['arm']}/{ctx['seed']}/{month}"
            ] = values
        print(
            f"Verified complete class metrics: {ctx['view']} {ctx['regime']} {ctx['arm']} {ctx['seed']}",
            flush=True,
        )
    if len(details) != 72:
        raise ValueError("Incomplete source-input details")
    target = public / "metric_details.json"
    atomic(
        target,
        lambda p: p.write_bytes(
            (
                json.dumps(details, indent=2, sort_keys=True, allow_nan=False) + "\n"
            ).encode()
        ),
    )
    audit = dict(
        status="passed",
        scope="Read-only export from signed frozen predictions; no refits",
        completed_fits=24,
        metric_records=72,
        class_metric_records=216,
        prediction_rows_verified=prediction_rows,
        original_headline_metrics_unchanged=True,
        source_manifest_sha256=file_hash(manifest_path),
        source_per_seed_metrics_sha256=file_hash(public / "metrics_by_seed.csv"),
        details_sha256=file_hash(target),
        elapsed_seconds=time.monotonic() - start,
    )
    atomic(
        public / "detail_export_verification.json",
        lambda p: p.write_bytes(
            (json.dumps(audit, indent=2, sort_keys=True) + "\n").encode()
        ),
    )
    return audit


if __name__ == "__main__":
    print(json.dumps(export(Path(__file__).resolve().parents[1]), indent=2))
