"""Shared env-side utilities: feature-file caching."""

from __future__ import annotations

from pathlib import Path

import numpy as np


# In-process cache so multiple envs in the same process share the (big) feature
# arrays in memory. np.load on .npz with no mmap loads arrays on attribute
# access; we dict-copy them so each instance points at the same buffers.
_FEATURE_CACHE: dict[str, dict[str, np.ndarray]] = {}


def load_features_cached(path: str | Path) -> dict[str, np.ndarray]:
    key = str(Path(path).resolve())
    if key not in _FEATURE_CACHE:
        with np.load(path, allow_pickle=False) as d:
            _FEATURE_CACHE[key] = {
                "features":      np.asarray(d["features"]),
                "open":          np.asarray(d["open"]),
                "close":         np.asarray(d["close"]),
                "timestamps":    np.asarray(d["timestamps"]),
                "feature_names": np.asarray(d["feature_names"]),
                "symbols":       np.asarray(d["symbols"]),
            }
    return _FEATURE_CACHE[key]
