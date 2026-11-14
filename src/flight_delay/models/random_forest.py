"""Bounded sparse random forest."""

from sklearn.ensemble import RandomForestClassifier


def create(params: dict, seed: int, threads: int):
    return RandomForestClassifier(**params, random_state=seed, n_jobs=threads)
