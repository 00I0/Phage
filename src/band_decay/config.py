"""Immutable configuration and policy composition for decay analyses."""

from __future__ import annotations

import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from .policies import (
    CollapseTaxa,
    DecayDisplayPolicy,
    DisplayAxisPolicy,
    EntityAvailableYears,
    GlobalTopN,
    GlobalTransientTaxa,
    LargestContiguousBlock,
    SelectedYearRanking,
    TaxonGroupingPolicy,
    TaxonRankingPolicy,
    TopNSelectionPolicy,
    TransientPolicy,
    UnionAvailableYears,
    YearSelectionPolicy,
)
from .policies import OriginalDecay
from .priors import DecayPriorConfig


SUPPORTED_FILENAME_FIELDS = frozenset({"top_n", "top_n_scope"})


@dataclass(frozen=True)
class InputConfig:
    """Configure the source dataset and countries to analyze."""

    data_path: Path | None = None
    countries: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.data_path is not None:
            object.__setattr__(self, "data_path", Path(self.data_path))
        countries = tuple(str(country) for country in self.countries)
        if not countries or any(not country for country in countries):
            raise ValueError("input.countries must contain at least one non-empty country.")
        if len(set(countries)) != len(countries):
            raise ValueError("input.countries must not contain duplicates.")
        object.__setattr__(self, "countries", countries)


@dataclass(frozen=True)
class YearSelectionConfig:
    """Compose year-selection and display-axis policies.

    Attributes:
        min_count_per_year: Default yearly count threshold.
        per_country_min_count_per_year: Country-specific thresholds.
        selection: Policy selecting analytical years.
        display_axis: Policy selecting displayed years.
    """

    min_count_per_year: int = 10
    per_country_min_count_per_year: Mapping[str, int] = field(default_factory=dict)
    selection: YearSelectionPolicy = field(default_factory=LargestContiguousBlock)
    display_axis: DisplayAxisPolicy = field(default_factory=EntityAvailableYears)

    def __post_init__(self) -> None:
        if int(self.min_count_per_year) < 0:
            raise ValueError("year_selection.min_count_per_year must be nonnegative.")
        overrides = {str(country): int(value) for country, value in self.per_country_min_count_per_year.items()}
        if any(not country for country in overrides) or any(value < 0 for value in overrides.values()):
            raise ValueError("year_selection country overrides must use nonnegative values and non-empty names.")
        object.__setattr__(self, "per_country_min_count_per_year", overrides)

    def min_count_for_country(self, country: str) -> int:
        """Return the threshold configured for one country."""
        return int(self.per_country_min_count_per_year.get(country, self.min_count_per_year))


@dataclass(frozen=True)
class TopNConfig:
    """Compose taxon ranking, selection, transient, and threshold policies."""

    n: int = 18
    per_country_n: Mapping[str, int] = field(default_factory=dict)
    global_selected_serotypes: tuple[str, ...] | None = None
    selection: TopNSelectionPolicy = field(default_factory=GlobalTopN)
    ranking: TaxonRankingPolicy = field(default_factory=SelectedYearRanking)
    transient: TransientPolicy = field(default_factory=GlobalTransientTaxa)
    min_year_count: float = 0.0
    per_country_min_year_count: Mapping[str, float] = field(default_factory=dict)
    min_year_percent: float = 0.0
    per_country_min_year_percent: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.n) < 0:
            raise ValueError("top_n.n must be nonnegative.")
        n_overrides = {str(country): int(value) for country, value in self.per_country_n.items()}
        count_overrides = {str(country): float(value) for country, value in self.per_country_min_year_count.items()}
        percent_overrides = {str(country): float(value) for country, value in self.per_country_min_year_percent.items()}
        if any(not country or value < 0 for country, value in n_overrides.items()):
            raise ValueError("top_n.per_country_n values must be nonnegative with non-empty country names.")
        if float(self.min_year_count) < 0 or any(not country or value < 0 for country, value in count_overrides.items()):
            raise ValueError("top_n minimum count values must be nonnegative.")
        if not 0 <= float(self.min_year_percent) <= 100:
            raise ValueError("top_n.min_year_percent must be in [0, 100].")
        if any(not country or not 0 <= value <= 100 for country, value in percent_overrides.items()):
            raise ValueError("top_n percentage overrides must be in [0, 100].")
        selected = None if self.global_selected_serotypes is None else tuple(str(value) for value in self.global_selected_serotypes)
        if selected is not None and (any(not value for value in selected) or len(set(selected)) != len(selected)):
            raise ValueError("top_n.global_selected_serotypes must contain unique non-empty labels.")
        object.__setattr__(self, "per_country_n", n_overrides)
        object.__setattr__(self, "per_country_min_year_count", count_overrides)
        object.__setattr__(self, "per_country_min_year_percent", percent_overrides)
        object.__setattr__(self, "global_selected_serotypes", selected)

    def n_for_country(self, country: str) -> int:
        """Return the Top-N value configured for one country."""
        return int(self.per_country_n.get(country, self.n))

    def min_year_count_for_country(self, country: str) -> float:
        """Return the minimum count configured for one country."""
        return float(self.per_country_min_year_count.get(country, self.min_year_count))

    def min_year_percent_for_country(self, country: str) -> float:
        """Return the minimum percentage configured for one country."""
        return float(self.per_country_min_year_percent.get(country, self.min_year_percent))


@dataclass(frozen=True)
class MorisitaHornConfig:
    """Compose grouping policies and an optional maximum MH lag."""

    other_grouping: TaxonGroupingPolicy = field(default_factory=CollapseTaxa)
    transient_grouping: TaxonGroupingPolicy = field(default_factory=CollapseTaxa)
    max_lag: int | None = None

    def __post_init__(self) -> None:
        if self.max_lag is not None and int(self.max_lag) <= 0:
            raise ValueError("mh.max_lag must be positive when provided.")
        if self.max_lag is not None:
            object.__setattr__(self, "max_lag", int(self.max_lag))


@dataclass(frozen=True)
class SamplingConfig:
    """Configure PyMC sampling and explicit decay priors."""

    draws: int = 500
    tune: int = 500
    chains: int = 4
    cores: int = 4
    target_accept: float = 0.95
    seed: int = 0
    observation_sigma: float | None = None
    priors: DecayPriorConfig = field(default_factory=DecayPriorConfig.legacy)

    def __post_init__(self) -> None:
        if int(self.draws) < 1 or int(self.tune) < 0 or int(self.chains) < 1 or int(self.cores) < 1:
            raise ValueError("sampling draws/chains/cores must be positive and tune must be nonnegative.")
        if not 0 < float(self.target_accept) < 1:
            raise ValueError("sampling.target_accept must be between 0 and 1.")
        if self.observation_sigma is not None and float(self.observation_sigma) <= 0:
            raise ValueError("sampling.observation_sigma must be positive when provided.")


@dataclass(frozen=True)
class PlotConfig:
    """Configure combined figure rendering."""

    dpi: int = 300
    show: bool = False
    max_legend_labels: int = 42
    count_label_max_y_fraction: float = 0.95
    count_label_max_years: int = 14
    strike_excluded_year_labels: bool = True
    excluded_year_alpha: float = 0.28
    excluded_year_hatch: str = "////"
    decay_display: DecayDisplayPolicy = field(default_factory=OriginalDecay)

    def __post_init__(self) -> None:
        if int(self.dpi) < 1 or int(self.max_legend_labels) < 0 or int(self.count_label_max_years) < 0:
            raise ValueError("plot dpi, legend limit, and label years must be nonnegative; dpi must be positive.")
        if float(self.count_label_max_y_fraction) < 0 or not 0 <= float(self.excluded_year_alpha) <= 1:
            raise ValueError("plot label fraction must be nonnegative and mask alpha must be in [0, 1].")


@dataclass(frozen=True)
class OutputConfig:
    """Configure output directory and filename template."""

    output_directory: Path = Path("plots")
    filename_template: str = "whitelisted_top_{top_n}_global.pdf"

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_directory", Path(self.output_directory))
        fields = {field_name for _, field_name, _, _ in string.Formatter().parse(self.filename_template) if field_name}
        unsupported = fields.difference(SUPPORTED_FILENAME_FIELDS)
        if unsupported:
            raise ValueError(f"output.filename_template contains unsupported fields: {sorted(unsupported)}")


@dataclass(frozen=True)
class AnalysisConfig:
    """Complete policy composition for one decay analysis."""

    input: InputConfig
    year_selection: YearSelectionConfig = field(default_factory=YearSelectionConfig)
    top_n: TopNConfig = field(default_factory=TopNConfig)
    mh: MorisitaHornConfig = field(default_factory=MorisitaHornConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    plot: PlotConfig = field(default_factory=PlotConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


@dataclass(frozen=True)
class SensitivityConfig:
    """Configure coverage sensitivity runs and stability summaries."""

    coverage_percentages: tuple[float, ...] = (80.0, 90.0, 95.0)
    output_directory: Path = Path("plots/sensitivity")
    filename_template: str = "region_country_band_decay2_{coverage_percent:g}pct.png"
    stability_horizon_years: float = 20.0
    stability_targets: tuple[float, ...] = (0.3, 0.5, 0.7, 0.8, 0.9, 0.95)
    minimum_pairs_for_supported_lag: int = 4

    def __post_init__(self) -> None:
        coverage = tuple(float(value) for value in self.coverage_percentages)
        targets = tuple(float(value) for value in self.stability_targets)
        if not coverage or any(not 0 < value <= 100 for value in coverage):
            raise ValueError("sensitivity coverage percentages must be in (0, 100].")
        if not targets or any(not 0 <= value <= 1 for value in targets):
            raise ValueError("sensitivity stability targets must be in [0, 1].")
        if not float(self.stability_horizon_years) > 0 or int(self.minimum_pairs_for_supported_lag) < 1:
            raise ValueError("sensitivity horizon must be positive and minimum pair count at least one.")
        object.__setattr__(self, "coverage_percentages", coverage)
        object.__setattr__(self, "stability_targets", targets)
        object.__setattr__(self, "output_directory", Path(self.output_directory))


@dataclass(frozen=True)
class CurvePlotConfig:
    """Configure standalone fitted-curve rendering."""

    output_path: Path = Path("plots/region_country_band_decay2_fitted_curves.png")
    horizon_years: float = 20.0
    point_count: int = 400
    dpi: int = 300
    show: bool = False
    display: DecayDisplayPolicy = field(default_factory=lambda: OriginalDecay())

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_path", Path(self.output_path))
        if float(self.horizon_years) <= 0 or int(self.point_count) < 2 or int(self.dpi) < 1:
            raise ValueError("curve horizon must be positive, point count at least two, and dpi positive.")
