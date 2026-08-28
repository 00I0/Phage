from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .config import AnalysisConfig


@dataclass(frozen=True)
class EntityYearSelection:
    """Record available, selected, excluded, and missing years for an entity."""

    entity: str
    available_years: tuple[int, ...]
    qualifying_years: frozenset[int]
    selected_years: frozenset[int]
    excluded_years: frozenset[int]
    missing_years: frozenset[int] = frozenset()

    def __post_init__(self) -> None:
        available = frozenset(self.available_years)
        if not self.selected_years.issubset(available):
            raise ValueError(f"{self.entity}: selected years must be available years.")
        if self.selected_years.intersection(self.excluded_years):
            raise ValueError(f"{self.entity}: selected and excluded years overlap.")
        if self.selected_years.union(self.excluded_years) != available:
            raise ValueError(f"{self.entity}: selected plus excluded years must equal available years.")


@dataclass(frozen=True)
class EntityGrouping:
    """Record selected and grouped serotypes for an entity."""

    entity: str
    selected_serotypes: tuple[str, ...]
    transient_serotypes: tuple[str, ...]
    include_other: bool
    include_transient: bool

    def __post_init__(self) -> None:
        overlap = set(self.selected_serotypes).intersection(self.transient_serotypes)
        if overlap:
            raise ValueError(f"{self.entity}: selected and transient serotypes overlap: {sorted(overlap)}")


@dataclass(frozen=True)
class PreparedEntityData:
    """Hold all prepared matrices and metadata for one entity."""

    entity: str
    raw_counts: pd.DataFrame
    display_counts: pd.DataFrame
    analysis_counts: pd.DataFrame
    grouped_display_counts: pd.DataFrame
    mh_counts: pd.DataFrame
    grouping: EntityGrouping
    year_selection: EntityYearSelection
    display_total: float
    analysis_total: float


@dataclass(frozen=True)
class PreparedData:
    """Hold prepared data for the complete analysis."""

    entity_order: tuple[str, ...]
    entities: Mapping[str, PreparedEntityData]
    master_display_columns: tuple[str, ...]
    palette: Mapping[str, str]


@dataclass(frozen=True)
class DecayFit:
    """Store posterior draws for an exponential decay model."""

    y0_samples: np.ndarray
    b_samples: np.ndarray
    c_samples: np.ndarray
    divergences: int = 0
    sigma_samples: np.ndarray | None = None

    def __post_init__(self) -> None:
        arrays = tuple(np.asarray(value, dtype=float).reshape(-1) for value in (self.y0_samples, self.b_samples, self.c_samples))
        if len({len(value) for value in arrays}) != 1:
            raise ValueError("DecayFit posterior arrays must have equal lengths.")
        object.__setattr__(self, "y0_samples", arrays[0])
        object.__setattr__(self, "b_samples", arrays[1])
        object.__setattr__(self, "c_samples", arrays[2])
        if self.sigma_samples is not None:
            sigma = np.asarray(self.sigma_samples, dtype=float).reshape(-1)
            if len(sigma) != len(arrays[0]):
                raise ValueError("DecayFit sigma samples must align with parameter samples.")
            object.__setattr__(self, "sigma_samples", sigma)

    def curve_draws(self, x_values: np.ndarray) -> np.ndarray:
        """Evaluate every posterior draw at the supplied lags.

        Args:
            x_values: One-dimensional lag values.

        Returns:
            Array shaped as ``(draws, len(x_values))``.
        """
        x_values = np.asarray(x_values, dtype=float).reshape(-1)
        return self.c_samples[:, None] + (self.y0_samples[:, None] - self.c_samples[:, None]) * np.exp(
            -self.b_samples[:, None] * x_values[None, :]
        )

    def median_curve(self, x_values: np.ndarray) -> np.ndarray:
        """Return the pointwise median of the posterior curves."""
        return np.median(self.curve_draws(x_values), axis=0)

    def interval(self, x_values: np.ndarray, q: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
        """Return a central posterior interval on the original scale.

        Args:
            x_values: One-dimensional lag values.
            q: Interval mass in ``(0, 1]``.

        Returns:
            Lower and upper pointwise interval bounds.
        """
        if not 0 < float(q) <= 1:
            raise ValueError("q must be in (0, 1].")
        alpha = 0.5 * (1.0 - float(q))
        draws = self.curve_draws(x_values)
        return np.quantile(draws, alpha, axis=0), np.quantile(draws, 1.0 - alpha, axis=0)

    def normalized_curve_draws(self, x_values: np.ndarray) -> np.ndarray:
        """Evaluate posterior curves after dividing each draw by ``y0``."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.divide(
                self.curve_draws(x_values),
                self.y0_samples[:, None],
                out=np.full((len(self.y0_samples), len(np.asarray(x_values).reshape(-1))), np.nan),
                where=self.y0_samples[:, None] != 0,
            )

    def normalized_median_curve(self, x_values: np.ndarray) -> np.ndarray:
        """Return the pointwise median of normalized posterior curves."""
        return np.nanmedian(self.normalized_curve_draws(x_values), axis=0)

    def normalized_interval(self, x_values: np.ndarray, q: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
        """Return a central posterior interval on the normalized scale."""
        if not 0 < float(q) <= 1:
            raise ValueError("q must be in (0, 1].")
        alpha = 0.5 * (1.0 - float(q))
        draws = self.normalized_curve_draws(x_values)
        return np.nanquantile(draws, alpha, axis=0), np.nanquantile(draws, 1.0 - alpha, axis=0)


@dataclass(frozen=True)
class EntityDecayData:
    """Store observed lag-pair similarities and an optional fitted model."""

    x: np.ndarray
    y: np.ndarray
    fit: DecayFit | None = None


@dataclass(frozen=True)
class RunResult:
    """Return prepared data, decay results, and the rendered output path."""

    config: AnalysisConfig
    prepared: PreparedData
    decay_data: Mapping[str, EntityDecayData]
    output_path: Path | None


@dataclass(frozen=True)
class CoveragePlan:
    """Describe one requested coverage level and its selected taxa."""

    coverage_percent: float
    country_n: Mapping[str, int]
    achieved_share: Mapping[str, float]
    eligible_count: Mapping[str, int]
    global_selected_serotypes: tuple[str, ...]


@dataclass(frozen=True)
class SensitivityResult:
    """Collect coverage plans, runs, fit summaries, and stability summaries."""

    plans: tuple[CoveragePlan, ...]
    runs: tuple[RunResult, ...]
    fit_summary: pd.DataFrame
    stability_summary: pd.DataFrame


@dataclass(frozen=True)
class CurveParameters:
    """Parameters of ``c + (y0 - c) * exp(-b * x)``."""

    y0: float
    b: float
    c: float

    def evaluate(self, x_values: np.ndarray) -> np.ndarray:
        """Evaluate the curve at lag values.

        Args:
            x_values: Lag values at which to evaluate the curve.
        Returns:
            Curve values with the shape of ``x_values``.
        """
        return self.c + (self.y0 - self.c) * np.exp(-self.b * np.asarray(x_values, dtype=float))
