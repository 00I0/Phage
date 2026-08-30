"""Fit country decay curves and export summaries for the R plot."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from band_decay import (
    AnalysisConfig,
    BetaPrior,
    DecayAnalysis,
    DecayPriorConfig,
    DirectAsymptote,
    FixedObservationNoise,
    HalfNormalPrior,
    InputConfig,
    MorisitaHornConfig,
    NoTransientTaxa,
    OutputConfig,
    PerEntityTopN,
    PlotConfig,
    QualifyingYears,
    SamplingConfig,
    SkipTaxa,
    TopNConfig,
    UnionAvailableYears,
    YearSelectionConfig,
)
from band_decay.domain import EntityDecayData


COUNTRIES = (
    "Greece",
    "Italy",
    "Spain",
    "Russia",
    "United Kingdom",
    "France",
    "Germany",
    "Switzerland",
)
MIN_YEAR_COUNT_BY_COUNTRY = {
    "United Kingdom": 2.0,
    "Switzerland": 2.0,
    "Germany": 2.0,
    "Greece": 3.0,
    "Italy": 3.0,
    "Spain": 3.0,
    "Russia": 3.0,
    "France": 3.0,
}
HORIZON_YEARS = 20.0
POINT_COUNT = 396
CONFIDENCE_LEVEL = 0.95
OUTPUT_RELATIVE_PATH = Path("data/fitted_country_curves_data.R")
R_VALUES_PER_LINE = 12
R_VALUE_PRECISION = 4


def build_config(project_directory: Path) -> AnalysisConfig:
    """Build the independently configurable analysis configuration."""
    priors = DecayPriorConfig(
        y0=BetaPrior(alpha=2.0, beta=2.0),
        b=HalfNormalPrior(sigma=2.0),
        asymptote=DirectAsymptote(BetaPrior(alpha=2.0, beta=2.0)),
        noise=FixedObservationNoise(),
    )
    return AnalysisConfig(
        input=InputConfig(
            data_path=project_directory / "data/serotype_counts_country_ds_geodate2-2.tsv",
            countries=COUNTRIES,
        ),
        year_selection=YearSelectionConfig(
            min_count_per_year=10,
            selection=QualifyingYears(),
            display_axis=UnionAvailableYears(),
        ),
        top_n=TopNConfig(
            n=0,
            per_country_n={},
            selection=PerEntityTopN(),
            per_country_min_year_count=MIN_YEAR_COUNT_BY_COUNTRY,
            transient=NoTransientTaxa(),
        ),
        mh=MorisitaHornConfig(other_grouping=SkipTaxa(), transient_grouping=SkipTaxa()),
        sampling=SamplingConfig(
            draws=2_500,
            tune=2_500,
            chains=8,
            cores=8,
            target_accept=0.975,
            seed=0,
            observation_sigma=None,
            priors=priors,
        ),
        plot=PlotConfig(),
        output=OutputConfig(output_directory=project_directory / "plots"),
    )


def finite_fit_end(decay: EntityDecayData, country_name: str) -> float:
    """Return the largest finite observed lag for one country."""
    finite_lags = np.asarray(decay.x, dtype=float).reshape(-1)
    finite_lags = finite_lags[np.isfinite(finite_lags)]
    if not len(finite_lags):
        raise ValueError(f"No finite fitted lags are available for {country_name}.")
    return float(np.max(finite_lags))


def format_r_vector(values: np.ndarray, field_name: str) -> list[str]:
    """Format one numeric vector as readable R source lines."""
    numeric_values = np.asarray(values, dtype=float).reshape(-1)
    formatted_values = [f"{value:.{R_VALUE_PRECISION}f}" for value in numeric_values]
    rows = [
        formatted_values[start : start + R_VALUES_PER_LINE]
        for start in range(0, len(formatted_values), R_VALUES_PER_LINE)
    ]
    lines = [f"      {field_name} = c("]
    lines.extend(
        f"        {', '.join(row)}{',' if row_index < len(rows) - 1 else ''}"
        for row_index, row in enumerate(rows)
    )
    lines.append("      )")
    return lines


def build_r_data(
    decay_data: Mapping[str, EntityDecayData],
    entity_order: tuple[str, ...],
    *,
    horizon_years: float,
    point_count: int,
    confidence_level: float,
) -> str:
    """Build the R data file from fitted posterior curve summaries."""
    if not 0 < confidence_level <= 1:
        raise ValueError("confidence_level must be in (0, 1].")
    if point_count < 2:
        raise ValueError("point_count must be at least 2.")

    lag_years = np.linspace(0.0, horizon_years, point_count)
    tail_probability = 0.5 * (1.0 - confidence_level)
    lines = [
        "curve_data <- list(",
        f"  horizon_years = {horizon_years:.8g},",
        f"  lag_years = seq(0, {horizon_years:.8g}, length.out = {point_count}),",
        "  country_curves = list(",
    ]

    for country_index, country_name in enumerate(entity_order):
        decay = decay_data[country_name]
        if decay.fit is None:
            raise ValueError(f"No fitted decay curve is available for {country_name}.")

        curve_draws = decay.fit.normalized_curve_draws(lag_years)
        median_curve = np.nanmedian(curve_draws, axis=0)
        lower_curve = np.nanquantile(curve_draws, tail_probability, axis=0)
        upper_curve = np.nanquantile(curve_draws, 1.0 - tail_probability, axis=0)
        fitted_lag = finite_fit_end(decay, country_name)
        median_at_fitted_lag = np.nanmedian(
            decay.fit.normalized_curve_draws(np.array([fitted_lag]))[:, 0]
        )
        if not all(np.all(np.isfinite(values)) for values in (median_curve, lower_curve, upper_curve)):
            raise ValueError(f"Non-finite curve summary values are present for {country_name}.")

        lines.extend(
            [
                f"    `{country_name}` = list(",
                f"      extrapolation_start = {fitted_lag:.{R_VALUE_PRECISION}f},",
                f"      median_at_extrapolation_start = {median_at_fitted_lag:.{R_VALUE_PRECISION}f},",
            ]
        )
        for field_name, values in (
            ("median", median_curve),
            ("lower", lower_curve),
            ("upper", upper_curve),
        ):
            vector_lines = format_r_vector(values, field_name)
            if field_name != "upper":
                vector_lines[-1] += ","
            lines.extend(vector_lines)
        country_suffix = "," if country_index < len(entity_order) - 1 else ""
        lines.append(f"    ){country_suffix}")

    lines.extend(["  )", ")", ""])
    return "\n".join(lines)


def main() -> None:
    """Fit configured countries and update the R plotting data file."""
    project_directory = Path(__file__).resolve().parents[1]
    config = build_config(project_directory)
    analysis = DecayAnalysis(config)
    prepared = analysis.prepare()
    decay_data = analysis.fit(prepared)
    entity_order = prepared.entity_order[1:]
    data_file = project_directory / OUTPUT_RELATIVE_PATH
    data_file.parent.mkdir(parents=True, exist_ok=True)
    data_file.write_text(
        build_r_data(
            decay_data,
            entity_order,
            horizon_years=HORIZON_YEARS,
            point_count=POINT_COUNT,
            confidence_level=CONFIDENCE_LEVEL,
        ),
        encoding="utf-8",
    )
    print(f"Updated {data_file}")


if __name__ == "__main__":
    main()
