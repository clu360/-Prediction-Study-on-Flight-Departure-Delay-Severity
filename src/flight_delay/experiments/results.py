"""Atomic, content-checked checkpoints; only explicit aggregates go public."""

from contextlib import contextmanager
from pathlib import Path
import json
import os
import uuid
import joblib
import polars as pl
from .config import digest, file_hash

SUMMARY_COLUMNS = [
    "run_id",
    "model",
    "feature_view",
    "evaluation_stage",
    "scored_month",
    "airport",
    "n",
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "severe_recall",
    "severe_precision",
    "log_loss",
    "brier",
    "macro_auc",
    "selected_config_id",
    "model_fit_id",
    "provenance_hash",
]
PROB_COLUMNS = ["p_minimal", "p_moderate", "p_severe"]


def atomic(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp{path.suffix}")
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, value):
    atomic(
        path,
        lambda p: p.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        ),
    )


def write_frame(path, frame):
    path = Path(path)
    atomic(
        path,
        lambda p: frame.write_csv(p, float_precision=12)
        if path.suffix == ".csv"
        else frame.write_parquet(p),
    )


def write_model(path, model):
    atomic(path, lambda p: joblib.dump(model, p, compress=3))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@contextmanager
def exclusive(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".writer.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise ValueError(
            "Another experiment writer or interrupted lock exists; inspect its PID before removing"
        ) from exc
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(str(os.getpid()))
        yield
    finally:
        lock.unlink(missing_ok=True)


class ResultStore:
    def __init__(self, root, context, resume=False):
        self.context = context
        self.signature = digest(context)
        self.run_id = f"phase3-{self.signature[:20]}"
        base = Path(root).resolve()
        if os.name == "nt" and not str(base).startswith("\\\\?\\"):
            base = Path("\\\\?\\" + str(base))
        self.root = base / self.run_id
        self.resume = resume
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / "context.json"
        if path.exists() and read_json(path) != context:
            raise ValueError("Stale or incompatible run context")
        if not path.exists():
            write_json(path, context)

    def load(self, key):
        path = self.root / f"{key}.json"
        if not path.exists():
            return None
        record = read_json(path)
        checksum = record.pop("record_hash", None)
        if record.get("run_signature") != self.signature or digest(record) != checksum:
            raise ValueError("Corrupted or incompatible checkpoint")
        for name, expected in record.get("artifacts", {}).items():
            target = (self.root / name).resolve()
            if (
                not target.is_relative_to(self.root.resolve())
                or not target.is_file()
                or file_hash(target) != expected
            ):
                raise ValueError("Missing/corrupted checkpoint artifact")
        return record

    def save(self, key, value, artifacts=()):
        value = dict(
            value,
            run_signature=self.signature,
            artifacts={
                str(Path(p).relative_to(self.root)).replace("\\", "/"): file_hash(p)
                for p in artifacts
            },
        )
        value["record_hash"] = digest(value)
        write_json(self.root / f"{key}.json", value)


def prediction_frame(run_id, model, view, stage, month, ids, y, p, fit_id):
    import numpy as np
    from .evaluation import check_probabilities

    check_probabilities(p, len(ids))
    if len(set(ids)) != len(ids) or len(y) != len(ids):
        raise ValueError("Duplicate or misaligned prediction IDs")
    return pl.DataFrame(
        dict(
            run_id=[run_id] * len(ids),
            model=[model] * len(ids),
            feature_view=[view] * len(ids),
            evaluation_stage=[stage] * len(ids),
            scored_month=[month] * len(ids),
            target_record_id=ids,
            y_true=np.asarray(y, dtype=np.int8),
            **{
                name: np.asarray(p[:, c], dtype=np.float64)
                for c, name in enumerate(PROB_COLUMNS)
            },
            y_pred=np.argmax(p, axis=1).astype(np.int8),
            model_fit_id=[fit_id] * len(ids),
        )
    )


def summarize(record, metrics, airport="overall"):
    row = {k: record.get(k) for k in SUMMARY_COLUMNS}
    row.update({k: metrics.get(k) for k in SUMMARY_COLUMNS if k in metrics})
    row["airport"] = airport
    return row


def public_tables(root, records):
    """Records contain metrics and fit summaries, never prediction rows."""
    root = Path(root)
    rows, details = [], []
    for record in records:
        rows.append(summarize(record, record["metrics"]))
        for airport, metrics in record.get("airport_metrics", {}).items():
            rows.append(summarize(record, metrics, airport))
        details.append(
            {
                k: record[k]
                for k in [
                    "run_id",
                    "model",
                    "feature_view",
                    "evaluation_stage",
                    "scored_month",
                    "metrics",
                    "airport_metrics",
                    "resources",
                    "transform",
                    "warnings",
                ]
                if k in record
            }
        )
    frame = (
        pl.DataFrame(rows, infer_schema_length=None)
        if rows
        else pl.DataFrame(schema={c: pl.String for c in SUMMARY_COLUMNS})
    )
    if rows:
        keys = [
            "run_id",
            "model",
            "feature_view",
            "evaluation_stage",
            "scored_month",
            "airport",
        ]
        if frame.select(keys).n_unique() != frame.height:
            raise ValueError("Duplicate public metric rows")
        frame = frame.select(SUMMARY_COLUMNS).sort(keys)
    write_frame(root / "metrics_summary.csv", frame)
    write_json(root / "metric_details.json", details)
