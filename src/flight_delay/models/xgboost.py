"""CPU histogram multiclass XGBoost; no scored-month early stopping."""


def create(params: dict, seed: int, threads: int):
    from xgboost import XGBClassifier

    return XGBClassifier(**params, random_state=seed, n_jobs=threads)
