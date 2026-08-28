from __future__ import annotations

import numpy as np
import pandas as pd


def counts_to_props(counts: pd.DataFrame) -> pd.DataFrame:
    """Convert yearly counts to row-wise proportions."""
    totals = counts.sum(axis=1).replace(0, np.nan)
    return counts.div(totals, axis=0).fillna(0.0)


def morisita_horn(left: np.ndarray, right: np.ndarray) -> float:
    """Compute the Morisita-Horn similarity between two compositions."""
    left = np.asarray(left, dtype=float).ravel()
    right = np.asarray(right, dtype=float).ravel()
    denominator = float(np.sum(left * left) + np.sum(right * right))
    return 0.0 if denominator <= 0 else float(2.0 * np.dot(left, right) / denominator)


def pair_indices(years: np.ndarray, max_lag: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return valid year-pair lags and their row indices."""
    years = np.asarray(years, dtype=int).ravel()
    lags: list[int] = []
    left_indices: list[int] = []
    right_indices: list[int] = []
    for left in range(years.size):
        for right in range(left + 1, years.size):
            lag = int(years[right] - years[left])
            if lag <= 0 or (max_lag is not None and lag > int(max_lag)):
                continue
            lags.append(lag)
            left_indices.append(left)
            right_indices.append(right)
    return np.asarray(lags, dtype=float), np.asarray(left_indices, dtype=int), np.asarray(right_indices, dtype=int)


def entity_pair_data(counts: pd.DataFrame, max_lag: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Compute lag-pair MH similarities for one entity’s count matrix."""
    totals = counts.sum(axis=1)
    counts = counts.loc[totals > 0]
    if len(counts.index) < 2:
        return np.array([], dtype=float), np.array([], dtype=float)
    years = counts.index.to_numpy(dtype=int)
    props = counts_to_props(counts).to_numpy(dtype=float)
    x_values, left_indices, right_indices = pair_indices(years, max_lag)
    if not len(x_values):
        return np.array([], dtype=float), np.array([], dtype=float)
    y_values = np.asarray(
        [morisita_horn(props[left], props[right]) for left, right in zip(left_indices, right_indices, strict=True)],
        dtype=float,
    )
    return x_values, y_values
