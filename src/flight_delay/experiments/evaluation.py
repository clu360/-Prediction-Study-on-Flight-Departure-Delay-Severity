"""Pure, fixed-class metrics. Brier = mean sum of three squared errors."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)


def check_probabilities(p, n):
    p = np.asarray(p)
    if (
        p.shape != (n, 3)
        or not n
        or not np.isfinite(p).all()
        or (p < 0).any()
        or (p > 1).any()
        or not np.allclose(p.sum(axis=1), 1, rtol=0, atol=1e-6)
    ):
        raise ValueError(
            "Expected finite (N,3) probabilities in [0,1], row sums=1 within 1e-6"
        )


def evaluate(y, p, bins=10):
    y, p = np.asarray(y), np.asarray(p, dtype=np.float64)
    check_probabilities(p, len(y))
    p = p / p.sum(axis=1, keepdims=True)
    if not np.isin(y, [0, 1, 2]).all() or bins != 10:
        raise ValueError("Locked labels [0,1,2] and 10 equal-width reliability bins")
    predicted = np.argmax(p, axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, predicted, labels=[0, 1, 2], zero_division=0
    )
    auc, curves = [], []
    for c in range(3):
        truth = y == c
        auc.append(
            float(roc_auc_score(truth, p[:, c])) if 0 < truth.sum() < len(y) else None
        )
        index = np.minimum((p[:, c] * bins).astype(int), bins - 1)
        for b in range(bins):
            selected = index == b
            curves.append(
                dict(
                    class_id=c,
                    bin_id=b,
                    left=b / bins,
                    right=(b + 1) / bins,
                    n=int(selected.sum()),
                    mean_probability=float(p[selected, c].mean())
                    if selected.any()
                    else None,
                    observed_rate=float(truth[selected].mean())
                    if selected.any()
                    else None,
                )
            )
    return dict(
        n=len(y),
        accuracy=float(accuracy_score(y, predicted)),
        macro_f1=float(
            f1_score(y, predicted, labels=[0, 1, 2], average="macro", zero_division=0)
        ),
        weighted_f1=float(
            f1_score(
                y, predicted, labels=[0, 1, 2], average="weighted", zero_division=0
            )
        ),
        log_loss=float(log_loss(y, p, labels=[0, 1, 2])),
        brier=float(np.square(p - np.eye(3)[y.astype(int)]).sum(axis=1).mean()),
        severe_recall=float(recall[2]),
        severe_precision=float(precision[2]),
        precision=precision.tolist(),
        recall=recall.tolist(),
        f1=f1.tolist(),
        support=support.tolist(),
        auc=auc,
        macro_auc=float(np.mean(auc)) if all(v is not None for v in auc) else None,
        confusion_matrix=confusion_matrix(y, predicted, labels=[0, 1, 2]).tolist(),
        observed_proportions=(support / len(y)).tolist(),
        mean_probabilities=p.mean(axis=0).tolist(),
        argmax_proportions=(np.bincount(predicted, minlength=3) / len(y)).tolist(),
        calibration=curves,
        uncertainty="Counts only; flights are correlated, no IID confidence interval claimed",
    )


def airport_metrics(y, p, origins, airports):
    origins = np.asarray(origins)
    if len(origins) != len(y):
        raise ValueError("Origin metadata is misaligned")
    return {
        airport: evaluate(
            np.asarray(y)[origins == airport], np.asarray(p)[origins == airport]
        )
        for airport in airports
        if np.any(origins == airport)
    }
