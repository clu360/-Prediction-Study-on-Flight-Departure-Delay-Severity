"""Hash logical fitted state, independently of array strides or storage padding."""

import hashlib
import joblib
import numpy as np


def _logical(value):
    if isinstance(value, np.ndarray):
        if value.dtype.names:
            # Tree node records include unused alignment bytes. Hash named fields.
            payload = [(name, _logical(value[name])) for name in value.dtype.names]
        elif value.dtype.hasobject:
            payload = _logical(value.tolist())
        else:
            payload = hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
        return ("array", value.dtype.descr, value.shape, payload)
    if isinstance(value, np.generic):
        return ("numpy_scalar", value.dtype.str, value.item())
    if value is None or isinstance(value, (bool, int, float, str, bytes, bytearray)):
        return bytes(value) if isinstance(value, bytearray) else value
    if isinstance(value, dict):
        return {key: _logical(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, [_logical(item) for item in value])
    if isinstance(value, (set, frozenset)):
        return (type(value).__name__, sorted(joblib.hash(_logical(x)) for x in value))
    if isinstance(value, type):
        return ("type", value.__module__, value.__qualname__)
    if isinstance(value, np.dtype):
        return ("dtype", value.descr)
    state = getattr(value, "__getstate__", lambda: None)()
    if state is not None:
        return (type(value).__module__, type(value).__qualname__, _logical(state))
    return (
        "opaque",
        type(value).__module__,
        type(value).__qualname__,
        joblib.hash(value),
    )


def state_hash(value):
    """Include parameters and learned values; exclude incidental memory layout.

    Estimator/transformer states are acyclic. NumPy object arrays are traversed
    by value, structured arrays by named field, and XGBoost's pickle state
    supplies its serialized booster bytes. No fitted object is modified.
    """
    return joblib.hash(("logical-fitted-state-v1", _logical(value)))
