"""Training priors with reproducible inverse-CDF sampled class predictions."""

import hashlib
import numpy as np

CDF_RULE = "prior_cdf_sha256_v1"


def uniform_draws(ids, seed=6740):
    """Stateless Uniform[0,1) variates, stable across ordering and batching.

    IDs only address the pseudorandom stream; they are not predictors. The
    domain-separated SHA-256 prefix uses 53 bits, exactly representable as a
    float64 fraction. Neither features nor labels enter the random draw.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError("A nonnegative integer random seed is required")
    prefix = f"{CDF_RULE}:{seed}:".encode()

    def draw(value):
        if not isinstance(value, (str, np.str_)) or not value:
            raise ValueError("Sampling requires nonempty stable string IDs")
        word = int.from_bytes(
            hashlib.sha256(prefix + value.encode()).digest()[:8], "big"
        )
        return (word >> 11) * (2.0**-53)

    return np.fromiter((draw(value) for value in ids), dtype=np.float64)


def inverse_cdf(probabilities, uniforms):
    """Class 0: U<p0; class 1: p0<=U<p0+p1; class 2: otherwise."""
    from ..experiments.evaluation import check_probabilities

    p = np.asarray(probabilities, dtype=np.float64)
    u = np.asarray(uniforms, dtype=np.float64)
    check_probabilities(p, len(u))
    if u.shape != (len(p),) or not np.isfinite(u).all() or ((u < 0) | (u >= 1)).any():
        raise ValueError("Expected one finite Uniform[0,1) draw per observation")
    p = p / p.sum(axis=1, keepdims=True)
    return (u >= p[:, 0]).astype(np.int8) + (u >= p[:, :2].sum(axis=1)).astype(np.int8)


def sampled_classes(probabilities, ids, seed=6740):
    return inverse_cdf(probabilities, uniform_draws(ids, seed))


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

    def predict(self, X, *, ids):
        return sampled_classes(self.predict_proba(X), ids)

    def get_params(self, deep=True):
        return {"strategy": CDF_RULE, "smoothing": 0, "seed": 6740}
