"""Validated name-to-factory contract and explicit probability class mapping."""

from typing import Protocol, Callable
import numpy as np
from .baseline import PriorBaseline
from . import logistic, random_forest, xgboost


class Estimator(Protocol):
    def fit(self, X, y): ...
    def predict_proba(self, X): ...


FACTORIES: dict[str, Callable] = {
    "baseline": lambda params, seed, threads: PriorBaseline(),
    "logistic": logistic.create,
    "random_forest": random_forest.create,
    "xgboost": xgboost.create,
}
KEYS = {
    "baseline": set(),
    "logistic": {"C", "class_weight", "solver", "penalty", "max_iter", "tol"},
    "random_forest": {
        "n_estimators",
        "max_depth",
        "min_samples_leaf",
        "max_features",
        "bootstrap",
    },
    "xgboost": {
        "max_depth",
        "learning_rate",
        "n_estimators",
        "subsample",
        "colsample_bytree",
        "tree_method",
        "objective",
        "num_class",
        "eval_metric",
    },
}


def validate_params(name: str, params: dict):
    if name not in KEYS or set(params) != KEYS[name]:
        raise ValueError(
            f"{name}: expected exactly hyperparameter keys {sorted(KEYS.get(name, []))}"
        )
    positive = {
        "C",
        "max_iter",
        "tol",
        "n_estimators",
        "max_depth",
        "min_samples_leaf",
        "learning_rate",
    }
    for key in positive & params.keys():
        if (
            isinstance(params[key], bool)
            or not isinstance(params[key], (int, float))
            or params[key] <= 0
        ):
            raise ValueError(f"{name}.{key} must be positive")
    for key in {
        "n_estimators",
        "max_depth",
        "min_samples_leaf",
        "max_iter",
    } & params.keys():
        if not isinstance(params[key], int):
            raise ValueError(f"{name}.{key} must be integer")
    for key in {"subsample", "colsample_bytree"} & params.keys():
        if not isinstance(params[key], (int, float)) or not 0 < params[key] <= 1:
            raise ValueError(f"{name}.{key} must be in (0,1]")
    if name == "logistic" and (
        params["solver"] != "saga"
        or params["penalty"] != "l2"
        or params["class_weight"] not in [None, "balanced"]
    ):
        raise ValueError(
            "Locked logistic solver=saga, penalty=l2, class_weight=null/balanced"
        )
    if name == "random_forest" and (
        params["bootstrap"] is not True or params["max_features"] != "sqrt"
    ):
        raise ValueError("Locked forest bootstrap=True, max_features=sqrt")
    if name == "xgboost" and any(
        params[k] != v
        for k, v in {
            "tree_method": "hist",
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
        }.items()
    ):
        raise ValueError("Locked XGBoost multiclass CPU histogram contract")


def create(name: str, params: dict, seed: int = 6740, threads: int = 4) -> Estimator:
    validate_params(name, params)
    if not 1 <= threads <= 4 or seed != 6740:
        raise ValueError("Require seed=6740 and 1..4 estimator threads")
    return FACTORIES[name](dict(params), seed, threads)


def check_training_labels(name: str, y):
    classes = np.unique(y)
    if not len(classes) or not np.isin(classes, [0, 1, 2]).all():
        raise ValueError("Invalid training severity labels")
    if name != "baseline" and len(classes) < 2:
        raise ValueError(f"{name}: at least two training classes required")
    if name == "xgboost" and classes.tolist() != [0, 1, 2]:
        raise ValueError(
            "XGBoost requires all three training classes; do not fabricate rows"
        )


def ordered_probabilities(model, X):
    raw = np.asarray(model.predict_proba(X), dtype=np.float64)
    classes = np.asarray(model.classes_)
    if (
        len(set(classes.tolist())) != len(classes)
        or not np.isin(classes, [0, 1, 2]).all()
        or raw.shape != (X.shape[0], len(classes))
    ):
        raise ValueError("Estimator probability/class mapping is invalid")
    result = np.zeros((len(raw), 3), dtype=np.float64)
    result[:, classes.astype(int)] = raw
    from ..experiments.evaluation import check_probabilities

    check_probabilities(result, X.shape[0])
    # Remove admitted float32 summation roundoff only AFTER checking the 1e-6
    # contract. This is numerical normalization, not probability calibration.
    result /= result.sum(axis=1, keepdims=True)
    return result
