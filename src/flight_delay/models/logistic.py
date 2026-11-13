"""Sparse multinomial logistic regression."""

from sklearn.linear_model import LogisticRegression


def create(params: dict, seed: int, threads: int):
    return LogisticRegression(**params, random_state=seed)
