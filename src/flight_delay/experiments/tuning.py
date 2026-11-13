"""Deterministic screening/shortlisting and equal-month development selection."""

import numpy as np
from .config import DEV_FOLDS, canonical, digest


def config_id(params):
    return digest(params)[:16]


def aggregate(records, expected_folds):
    if len(records) != len(expected_folds) or {
        r["scored_month"] for r in records
    } != set(expected_folds):
        raise ValueError("Incomplete or duplicate development folds")
    if any(
        r["evaluation_stage"] != "development" or r["scored_month"] not in DEV_FOLDS
        for r in records
    ):
        raise ValueError("Selection cannot consume holdout results")
    if (
        len({canonical(r["params"]) for r in records}) != 1
        or len({(r["model"], r["feature_view"]) for r in records}) != 1
    ):
        raise ValueError("Hyperparameters/model/view changed across folds")
    scores = [r["metrics"]["macro_f1"] for r in records]
    result = dict(
        params=records[0]["params"],
        config_id=config_id(records[0]["params"]),
        mean_macro_f1=float(np.mean(scores)),
        mean_log_loss=float(np.mean([r["metrics"]["log_loss"] for r in records])),
        macro_f1_std=float(np.std(scores, ddof=0)),
        macro_f1_variance=float(np.var(scores, ddof=0)),
        median_runtime=float(
            np.median([r["resources"]["fit_seconds"] for r in records])
        ),
        fold_scores={r["scored_month"]: r["metrics"]["macro_f1"] for r in records},
        evidence_hash=digest(
            [
                {
                    k: r[k]
                    for k in ["scored_month", "params", "metrics", "provenance_hash"]
                }
                for r in sorted(records, key=lambda x: x["scored_month"])
            ]
        ),
    )
    if not np.isfinite(
        [
            result[k]
            for k in [
                "mean_macro_f1",
                "mean_log_loss",
                "macro_f1_std",
                "median_runtime",
            ]
        ]
    ).all():
        raise ValueError("Nonfinite selection statistics")
    return result


def rank(candidates, tolerance=1e-6):
    """Anchor each tolerance tie to the current best, avoiding nontransitive comparators."""
    remaining = list(candidates)
    output = []
    while remaining:
        best = max(r["mean_macro_f1"] for r in remaining)
        tied = [r for r in remaining if best - r["mean_macro_f1"] <= tolerance]
        minimum = min(r["mean_log_loss"] for r in tied)
        tied = [r for r in tied if r["mean_log_loss"] - minimum <= tolerance]
        minimum = min(r["macro_f1_std"] for r in tied)
        tied = [r for r in tied if r["macro_f1_std"] - minimum <= tolerance]
        minimum = min(r["median_runtime"] for r in tied)
        tied = [r for r in tied if r["median_runtime"] - minimum <= tolerance]
        chosen = min(tied, key=lambda r: canonical(r["params"]))
        output.append(chosen)
        remaining = [r for r in remaining if r["config_id"] != chosen["config_id"]]
    return output
