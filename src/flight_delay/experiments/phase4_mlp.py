"""Screen and lock the unweighted MLP using chronological inner validation."""

from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from time import perf_counter
from uuid import uuid4
import platform

import numpy as np
import polars as pl
import torch

from .config import digest, file_hash
from .evaluation import evaluate
from .phase4_inputs import BASE, DEV, ID
from .results import ResultStore, exclusive, read_json, write_frame, write_json
from .runner import ids_hash
from ..models import mlp
from ..models.state import state_hash


def environment(device):
    return dict(python=platform.python_version(), platform=platform.system(),
                torch=torch.__version__, device=device, cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(),
                gpu=torch.cuda.get_device_name(device) if device.startswith("cuda") else None)


def matrices(inputs, train, validation):
    transformer = inputs.transformer(BASE, train)
    return (transformer.transform(train), train["target_class"].to_numpy(),
            transformer.transform(validation), validation["target_class"].to_numpy(),
            transformer)


def candidate_rank(records, candidates, months, indices):
    def score(index):
        selected = [records[index, month] for month in months]
        return (np.mean([r["metrics"]["log_loss"] for r in selected]),
                -np.mean([r["metrics"]["macro_f1"] for r in selected]),
                tuple(candidates[index]["widths"]),
                np.median([r["seconds"] for r in selected]), index)
    return sorted(indices, key=score)


def source_context(inputs, device, fingerprint):
    code = {p.name: file_hash(p) for p in (
        Path(__file__), Path(mlp.__file__), Path(__file__).with_name("phase4_inputs.py")
    )}
    return dict(input_sha256=fingerprint, code=code, settings=inputs.mlp,
                environment=environment(device), phase="phase4-mlp-development")


def tuning_fold(inputs, store, records, indices, month, device):
    pending = []
    for index in indices:
        key = f"tuning/candidate-{index + 1:02}/month-{month:02}"
        record = store.load(key)
        if record is None:
            pending.append((index, key))
        else:
            records[index, month] = record
    if not pending:
        return
    # Scope both learned preprocessing and gradient updates inside this outer fold.
    gradient = inputs.rows(month, "train", months=[month - 3, month - 2])
    stop = inputs.rows(month, "train", months=[month - 1])
    next_month = inputs.rows(month, "test")
    ids = {name: ids_hash(frame[ID]) for name, frame in
           [("inner_gradient", gradient), ("inner_stop", stop), ("outer_score", next_month)]}
    train = pl.concat([gradient, stop]).sort(ID)
    ids["outer_train"] = ids_hash(train[ID])
    inner = matrices(inputs, gradient, stop)
    del gradient, stop
    outer = matrices(inputs, train, next_month)
    del train
    settings = inputs.mlp
    seed, batch_size = settings["development_seed"], settings["batch_size"]
    for index, key in pending:
        attempt = uuid4().hex
        attempt_dir = store.root / key / attempt
        attempt_path = attempt_dir / "status.json"
        candidate = settings["candidates"][index]
        curves = {"inner": [], "refit": []}
        started = perf_counter()
        print(f"Phase IV MLP candidate={index + 1} fold={month:02} attempt={attempt}", flush=True)

        def log(stage, epoch):
            curves[stage].append(epoch)
            status("running")

        def status(state, error=None):
            write_json(attempt_path, dict(state=state, candidate=candidate, fold=month, seed=seed,
                                          updated_utc=datetime.now(timezone.utc).isoformat(),
                                          seconds=perf_counter() - started, curves=curves, error=error))

        status("running")
        try:
            _, epochs, _ = mlp.train(*inner[:2], candidate, settings, seed, device,
                                      settings["max_epochs"], stop=inner[2:4],
                                      on_epoch=lambda epoch: log("inner", epoch))
            model, _, _ = mlp.train(*outer[:2], candidate, settings, seed, device, epochs,
                                    on_epoch=lambda epoch: log("refit", epoch))
            probabilities = mlp.predict(model, outer[2], batch_size, device)
            metrics = evaluate(outer[3], probabilities)
            prediction_path = attempt_dir / "predictions.parquet"
            write_frame(prediction_path, next_month.select(ID, "target_class").with_columns(
                pl.Series("p_minimal", probabilities[:, 0]), pl.Series("p_moderate", probabilities[:, 1]),
                pl.Series("p_severe", probabilities[:, 2])))
            status("complete")
            store.save(key, dict(candidate=candidate, candidate_id=index + 1, score_month=month,
                                 seed=seed, epochs=epochs, curves=curves, metrics=metrics,
                                 stop_reason="epoch_cap" if len(curves["inner"]) == settings["max_epochs"] else "patience",
                                 seconds=perf_counter() - started, attempt_id=attempt,
                                 transformer_hash=state_hash(outer[4]), model_hash=mlp.state_sha(model),
                                 ids=ids), (prediction_path, attempt_path))
        except Exception as exc:
            status("failed", f"{type(exc).__name__}: {exc}")
            raise
        records[index, month] = store.load(key)


def tune(inputs, device):
    fingerprint = inputs.verify_source("development")
    context = source_context(inputs, device, fingerprint)
    store = ResultStore(inputs.path("output_dir") / "development", context, resume=True)
    settings = inputs.mlp
    candidates = settings["candidates"]
    screen, months = settings["screening_months"], settings["development_months"]
    if screen != [4, 7, 10] or months != list(DEV) or len(candidates) != 6:
        raise ValueError("Development calendar or candidate count changed")
    records = {}
    with exclusive(store.root):
        for month in screen:
            tuning_fold(inputs, store, records, range(len(candidates)), month, device)
        finalists = candidate_rank(records, candidates, screen, range(len(candidates)))[:2]
        for month in sorted(set(months) - set(screen)):
            tuning_fold(inputs, store, records, finalists, month, device)
        winner = candidate_rank(records, candidates, months, finalists)[0]
        rows = []
        for (index, month), record in sorted(records.items()):
            rows.append(dict(candidate_id=index + 1, month=month, epochs=record["epochs"],
                             log_loss=record["metrics"]["log_loss"], macro_f1=record["metrics"]["macro_f1"],
                             seconds=record["seconds"], stop_reason=record["stop_reason"],
                             attempt_id=record["attempt_id"], checkpoint_sha256=record["record_hash"]))
        summary = inputs.path("reports_dir") / "mlp_tuning_summary.csv"
        write_frame(summary, pl.DataFrame(rows))
        selection = dict(candidate=candidates[winner], candidate_id=winner + 1,
                         epochs=int(median(records[winner, month]["epochs"] for month in months)),
                         finalists=[index + 1 for index in finalists], settings=settings,
                         epoch_rule="median of seven inner stopping epochs for the winning candidate",
                         development_signature=store.signature, context=context,
                         tuning_summary_sha256=file_hash(summary))
        selection["sha256"] = digest(selection)
        path = inputs.path("output_dir") / "mlp_selection.json"
        if path.exists() and read_json(path) != selection:
            raise ValueError("MLP architecture lock already exists with different provenance")
        write_json(path, selection)
        return selection

