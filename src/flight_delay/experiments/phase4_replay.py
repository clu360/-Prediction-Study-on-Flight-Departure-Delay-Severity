"""Replay a fixed sample of both months using each verified saved model."""

import joblib
import numpy as np

from .config import file_hash
from .phase4_final import ARMS, booster_hash, load_predictions
from .phase4_inputs import HOLDOUT, ID
from .phase4_results import reporting_checkpoints
from .results import PROB_COLUMNS, write_json
from ..models.state import state_hash


def replay(inputs, tolerance=2e-6):
    verified = reporting_checkpoints(inputs)
    if not verified:
        raise ValueError("Frozen predictions are required for replay")
    result = []
    for (arm, seed), store in verified.items():
        fit = store.load("fit")
        saved = store.root / "attempts" / fit["attempt"]
        transformer = joblib.load(saved / "transformer.joblib")
        protocol = store.context["protocol"]
        device = store.context["environment"]["device"]
        neural = arm.startswith("MLP")
        if neural:
            from ..models import mlp

            model = mlp.create(len(transformer.feature_names_out_), protocol["candidate"],
                               seed, inputs.mlp["threads"], device)
            model.load_state_dict(mlp.torch.load(saved / "model.pt", map_location=device, weights_only=True))

            def predict(matrix):
                return mlp.predict(model, matrix, inputs.mlp["batch_size"], device)

            def learned_hash():
                return mlp.state_sha(model)
        else:
            from ..models.xgboost import create

            model = create(protocol["params"], seed, protocol["threads"])
            model.load_model(saved / "model.ubj")
            predict = model.predict_proba

            def learned_hash():
                return booster_hash(model)

        for month in HOLDOUT:
            predictions, _ = load_predictions(store, month)
            sample_index = np.linspace(0, predictions.height - 1, min(predictions.height, 2048), dtype=int)
            sample = predictions[sample_index.tolist()]
            frame = inputs.rows(month, "test", ARMS[arm][0])
            rows = sample.select(ID).join(frame, on=ID, how="left", validate="1:1", maintain_order="left")
            if rows["target_class"].null_count():
                raise ValueError("Replay sample lacks canonical IDs")
            probability = predict(transformer.transform(rows))
            maximum = float(np.abs(probability - sample.select(PROB_COLUMNS).to_numpy()).max())
            if maximum > tolerance or learned_hash() != fit["model_hash"] or state_hash(transformer) != fit["transformer_hash"]:
                raise ValueError(f"Frozen-state replay changed for {arm}/{seed}/{month}: {maximum}")
            result.append(dict(arm=arm, seed=seed, month=month, rows=len(sample_index),
                               maximum_absolute_probability_difference=maximum,
                               signature=store.signature))
    out = inputs.path("reports_dir") / "frozen_replay.json"
    write_json(out, dict(tolerance=tolerance, status="passed", repetitions=result,
                        metrics_summary_sha256=file_hash(inputs.path("reports_dir") / "metrics_summary.csv")))
    return result

