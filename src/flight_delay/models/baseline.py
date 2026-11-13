"""Unsmoothed training-window priors in the fixed severity order."""

import numpy as np


class PriorBaseline:
    def fit(self, X, y):
        y = np.asarray(y)
        if not len(y) or not np.isin(y, [0, 1, 2]).all():
            raise ValueError("Baseline requires nonempty labels in [0,1,2]")
        self.classes_ = np.arange(3)
        self.priors_ = np.bincount(y.astype(int), minlength=3) / len(y)
        return self

    def predict_proba(self, X):
        return np.tile(self.priors_, (X.shape[0], 1))

    def get_params(self, deep=True):
        return {"strategy": "prior", "smoothing": 0}
