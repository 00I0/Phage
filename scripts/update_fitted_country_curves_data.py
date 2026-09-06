"""Fit decay curves and export all data needed by the R plots."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
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
    NormalizedDecay,
    OutputConfig,
    PerEntityTopN,
    PlotConfig,
    PosteriorMedian,
    QualifyingYears,
    SamplingConfig,
    SensitivityConfig,
    SensitivityRunner,
    SkipTaxa,
    TopNConfig,
    UnionAvailableYears,
    YearSelectionConfig,
)
from band_decay.data import load_counts
from band_decay.domain import EntityDecayData, RunResult
from band_decay.preparation import counts_to_props


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
MIN_YEAR_COUNT = 2.0
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
OUTPUT_RELATIVE_PATH = Path("data/band_decay_data.R")
PALETTE_OUTPUT_RELATIVE_PATH = Path("data/palette.R")
BAND_COVERAGE_PERCENT = 90.0
BAND_STABILITY_TARGETS = (0.3, 0.5, 0.7, 0.8, 0.9, 0.95)
BAND_MINIMUM_PAIRS_FOR_SUPPORTED_LAG = 4
R_VALUES_PER_LINE = 12
R_VALUE_PRECISION = 4
R_CURVE_PRECISION = 8
COUNTRY_COLORS = {
    "GLOBAL": "#000000",
    "GLOBAL_ALL": "#000000",
    "Greece": "#1f77b4",
    "Italy": "#ff7f0e",
    "Spain": "#2ca02c",
    "Russia": "#d62728",
    "United Kingdom": "#9467bd",
    "France": "#8c564b",
    "Germany": "#e377c2",
    "Switzerland": "#7f7f7f",
}


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
            min_year_count=MIN_YEAR_COUNT,
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


def fit_all_countries_aggregate(config: AnalysisConfig) -> EntityDecayData:
    """Fit only the aggregate using every country present in the input file."""
    counts = load_counts(config.input.data_path)
    all_countries_config = replace(
        config,
        input=replace(config.input, countries=tuple(sorted(counts["country"].unique()))),
    )
    analysis = DecayAnalysis(all_countries_config)
    prepared = analysis.prepare(counts)
    aggregate_prepared = replace(
        prepared,
        entity_order=("GLOBAL",),
        entities={"GLOBAL": prepared.entities["GLOBAL"]},
    )
    return analysis.fit(aggregate_prepared)["GLOBAL"]


def finite_fit_end(decay: EntityDecayData, country_name: str) -> float:
    """Return the largest finite observed lag for one country."""
    finite_lags = np.asarray(decay.x, dtype=float).reshape(-1)
    finite_lags = finite_lags[np.isfinite(finite_lags)]
    if not len(finite_lags):
        raise ValueError(f"No finite fitted lags are available for {country_name}.")
    return float(np.max(finite_lags))


def format_r_scalar(value: float | int, *, precision: int = R_VALUE_PRECISION) -> str:
    """Format one scalar as an R numeric literal."""
    numeric_value = float(value)
    if np.isnan(numeric_value):
        return "NA_real_"
    if np.isposinf(numeric_value):
        return "Inf"
    if np.isneginf(numeric_value):
        return "-Inf"
    return f"{numeric_value:.{precision}f}"


def format_r_vector(values: np.ndarray, field_name: str, *, precision: int = R_VALUE_PRECISION) -> list[str]:
    """Format one numeric vector as readable R source lines."""
    numeric_values = np.asarray(values, dtype=float).reshape(-1)
    formatted_values = [format_r_scalar(value, precision=precision) for value in numeric_values]
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


def format_r_string_vector(values: Sequence[str], field_name: str, *, indent: str = "      ") -> list[str]:
    """Format one character vector as readable R source lines."""
    formatted_values = [f'"{str(value).replace(chr(34), chr(92) + chr(34))}"' for value in values]
    rows = [
        formatted_values[start : start + R_VALUES_PER_LINE]
        for start in range(0, len(formatted_values), R_VALUES_PER_LINE)
    ]
    lines = [f"{indent}{field_name} = c("]
    lines.extend(
        f"{indent}  {', '.join(row)}{',' if row_index < len(rows) - 1 else ''}"
        for row_index, row in enumerate(rows)
    )
    lines.append(f"{indent})")
    return lines


def r_name(name: str) -> str:
    """Return a quoted R list name that supports arbitrary labels."""
    return f"`{str(name).replace('`', '')}`"


def format_named_r_vector(values: Mapping[str, str], field_name: str, *, indent: str = "  ") -> list[str]:
    """Format a named character vector as readable R source lines."""
    lines = [f"{indent}{field_name} = c("]
    items = list(values.items())
    for index, (name, value) in enumerate(items):
        suffix = "," if index < len(items) - 1 else ""
        escaped_value = str(value).replace(chr(34), chr(92) + chr(34))
        lines.append(f'{indent}  {r_name(name)} = "{escaped_value}"{suffix}')
    lines.append(f"{indent})")
    return lines


def _finite_vector(values: np.ndarray, field_name: str, context: str) -> np.ndarray:
    """Return a finite vector or raise a useful export error."""
    numeric_values = np.asarray(values, dtype=float).reshape(-1)
    if not np.all(np.isfinite(numeric_values)):
        raise ValueError(f"Non-finite {field_name} values are present for {context}.")
    return numeric_values


def _decay_plot_data(
    decay: EntityDecayData,
    entity_name: str,
    *,
    display,
    summary,
) -> dict[str, object]:
    """Build displayed-scale curve data matching the combined Python renderer."""
    x_values = _finite_vector(decay.x, "lag", entity_name)
    y_values = _finite_vector(decay.y, "similarity", entity_name)
    xmax = max(1.0, float(np.max(x_values))) if len(x_values) else 1.0
    result: dict[str, object] = {
        "lag": x_values,
        "similarity": y_values,
        "fit": None,
    }
    if decay.fit is None:
        return result

    query = np.linspace(0.0, xmax, 200)
    median = _finite_vector(summary.curve(display, decay.fit, query), "median curve", entity_name)
    lower, upper = display.interval(decay.fit, query)
    lower = _finite_vector(lower, "lower curve", entity_name)
    upper = _finite_vector(upper, "upper curve", entity_name)
    fit_data = {
        "query": query,
        "median": median,
        "lower": lower,
        "upper": upper,
        "y0": summary.scalar(decay.fit.y0_samples),
        "b": summary.scalar(decay.fit.b_samples),
        "c": summary.scalar(decay.fit.c_samples),
        "divergences": int(decay.fit.divergences),
    }
    if decay.fit.sigma_samples is not None:
        fit_data["sigma"] = summary.scalar(decay.fit.sigma_samples)
    result["fit"] = fit_data
    return result


def build_band_decay_data(run: RunResult, *, coverage_percent: float) -> list[str]:
    """Build the R payload needed to recreate the combined band-decay figure."""
    prepared = run.prepared
    display = run.config.plot.decay_display
    summary = run.config.plot.fit_summary
    plot_config = run.config.plot
    lines = [
        "  band_decay = list(",
        f"    coverage_percent = {format_r_scalar(coverage_percent, precision=R_VALUE_PRECISION)},",
    ]
    lines.extend(format_r_string_vector(prepared.entity_order, "entity_order", indent="    "))
    lines[-1] += ","
    lines.extend(format_r_string_vector(prepared.master_display_columns, "display_columns", indent="    "))
    lines[-1] += ","
    lines.extend(
        [
            "    plot = list(",
            f"      decay_display = \"{display.option_name()}\",",
            f"      fit_summary = \"{summary.label()}\",",
            "      confidence_level = 0.95,",
            "      curve_point_count = 200,",
            f"      max_legend_labels = {int(plot_config.max_legend_labels)},",
            f"      count_label_max_y_fraction = {format_r_scalar(plot_config.count_label_max_y_fraction, precision=R_CURVE_PRECISION)},",
            f"      count_label_max_years = {int(plot_config.count_label_max_years)},",
            f"      strike_excluded_year_labels = {'TRUE' if plot_config.strike_excluded_year_labels else 'FALSE'},",
            f"      excluded_year_alpha = {format_r_scalar(plot_config.excluded_year_alpha, precision=R_CURVE_PRECISION)},",
            f"      excluded_year_hatch = \"{plot_config.excluded_year_hatch}\"",
            "    ),",
            "    entities = list(",
        ]
    )

    for entity_index, entity_name in enumerate(prepared.entity_order):
        entity = prepared.entities[entity_name]
        selection = entity.year_selection
        grouping = entity.grouping
        grouped_counts = entity.grouped_display_counts
        proportions = counts_to_props(grouped_counts)
        decay = _decay_plot_data(
            run.decay_data[entity_name],
            entity_name,
            display=display,
            summary=summary,
        )
        lines.extend(
            [
                f"      {r_name(entity_name)} = list(",
                f"        title = {chr(34)}{'GLOBAL aggregate' if entity_name == 'GLOBAL' else entity_name}{chr(34)},",
                f"        analysis_total = {format_r_scalar(entity.analysis_total)},",
            ]
        )
        for field_name, values in (
            ("available_years", selection.available_years),
            ("selected_years", sorted(selection.selected_years)),
            ("qualifying_years", sorted(selection.qualifying_years)),
            ("excluded_years", sorted(selection.excluded_years)),
            ("missing_years", sorted(selection.missing_years)),
        ):
            lines.append(f"        {field_name} = c({', '.join(str(int(value)) for value in values)}),")
        display_years = grouped_counts.index.to_numpy(dtype=int)
        lines.append(f"        display_years = c({', '.join(str(int(value)) for value in display_years)}),")
        lines.extend(["        grouping = list("])
        lines.extend(format_r_string_vector(grouping.selected_serotypes, "selected_serotypes", indent="          "))
        lines[-1] += ","
        lines.extend(format_r_string_vector(grouping.transient_serotypes, "transient_serotypes", indent="          "))
        lines[-1] += ","
        lines.extend(
            [
                f"          include_other = {'TRUE' if grouping.include_other else 'FALSE'},",
                f"          include_transient = {'TRUE' if grouping.include_transient else 'FALSE'}",
                "        ),",
            ]
        )
        lines.extend(["        grouped_counts = list("])
        for label_index, label in enumerate(prepared.master_display_columns):
            suffix = "," if label_index < len(prepared.master_display_columns) - 1 else ""
            values = grouped_counts[label].to_numpy(dtype=float)
            lines.extend(format_r_vector(values, r_name(label), precision=R_VALUE_PRECISION))
            lines[-1] += suffix
        lines.append("        ),")
        lines.extend(["        grouped_proportions = list("])
        for label_index, label in enumerate(prepared.master_display_columns):
            suffix = "," if label_index < len(prepared.master_display_columns) - 1 else ""
            values = proportions[label].to_numpy(dtype=float)
            lines.extend(format_r_vector(values, r_name(label), precision=R_CURVE_PRECISION))
            lines[-1] += suffix
        lines.extend(["        ),", "        decay = list("])
        for field_name in ("lag", "similarity"):
            lines.extend(format_r_vector(decay[field_name], field_name, precision=R_VALUE_PRECISION))
            lines[-1] += ","
        fit_data = decay["fit"]
        if fit_data is None:
            lines.append("          fit = NULL")
        else:
            lines.extend(
                [
                    "          fit = list(",
                ]
            )
            for field_name in ("query", "median", "lower", "upper"):
                lines.extend(format_r_vector(fit_data[field_name], field_name, precision=R_CURVE_PRECISION))
                lines[-1] += ","
            for field_name in ("y0", "b", "c"):
                lines.append(f"            {field_name} = {format_r_scalar(fit_data[field_name])},")
            if "sigma" in fit_data:
                lines.append(f"            sigma = {format_r_scalar(fit_data['sigma'])},")
            lines.append(f"            divergences = {int(fit_data['divergences'])}")
            lines.append("          )")
        lines.append("        )")
        entity_suffix = "," if entity_index < len(prepared.entity_order) - 1 else ""
        lines.append(f"      ){entity_suffix}")
    lines.extend(["    )", "  )"])
    return lines


def build_palette_r(prepared_palette: Mapping[str, str]) -> str:
    """Build the standalone R palette file."""
    lines = ["palette_data <- list("]
    lines.extend(format_named_r_vector(COUNTRY_COLORS, "country_colors", indent="  "))
    lines[-1] += ","
    lines.extend(format_named_r_vector(prepared_palette, "serotype_colors", indent="  "))
    lines.extend([")", ""])
    return "\n".join(lines)


def build_r_data(
    decay_data: Mapping[str, EntityDecayData],
    entity_order: tuple[str, ...],
    *,
    horizon_years: float,
    point_count: int,
    confidence_level: float,
    band_run: RunResult | None = None,
    band_coverage_percent: float | None = None,
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

    lines.append("  )" + ("," if band_run is not None else ""))
    if band_run is not None:
        if band_coverage_percent is None:
            raise ValueError("band_coverage_percent is required when band_run is provided.")
        lines.extend(build_band_decay_data(band_run, coverage_percent=band_coverage_percent))
    lines.extend([")", ""])
    return "\n".join(lines)


def build_band_config(project_directory: Path) -> AnalysisConfig:
    """Build the analysis configuration used by the combined Python figure."""
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
            min_year_count=MIN_YEAR_COUNT,
            per_country_min_year_count=MIN_YEAR_COUNT_BY_COUNTRY,
            transient=NoTransientTaxa(),
        ),
        mh=MorisitaHornConfig(other_grouping=SkipTaxa(), transient_grouping=SkipTaxa()),
        sampling=SamplingConfig(
            draws=500,
            tune=500,
            chains=8,
            cores=8,
            target_accept=0.975,
            seed=0,
            observation_sigma=None,
            priors=priors,
        ),
        plot=PlotConfig(
            max_legend_labels=200,
            decay_display=NormalizedDecay(),
            fit_summary=PosteriorMedian(),
        ),
        output=OutputConfig(output_directory=project_directory / "plots"),
    )


def build_sensitivity_config(project_directory: Path) -> SensitivityConfig:
    """Build the coverage-sensitivity configuration used by the combined figure."""
    return SensitivityConfig(
        coverage_percentages=(BAND_COVERAGE_PERCENT,),
        output_directory=project_directory / "plots",
        filename_template="band_decay_{coverage_percent:g}pct.png",
        stability_horizon_years=20.0,
        stability_targets=BAND_STABILITY_TARGETS,
        minimum_pairs_for_supported_lag=BAND_MINIMUM_PAIRS_FOR_SUPPORTED_LAG,
        fit_summary=PosteriorMedian(),
        stability_summary=PosteriorMedian(),
    )


def main() -> None:
    """Fit configured curves and update the R plotting data files."""
    project_directory = Path(__file__).resolve().parents[1]
    config = build_config(project_directory)
    analysis = DecayAnalysis(config)
    prepared = analysis.prepare()
    decay_data = analysis.fit(prepared)
    decay_data["GLOBAL_ALL"] = fit_all_countries_aggregate(config)
    band_config = build_band_config(project_directory)
    band_result = SensitivityRunner(band_config, build_sensitivity_config(project_directory)).run(fit=True)
    band_run = band_result.runs[0]
    entity_order = ("GLOBAL", "GLOBAL_ALL", *prepared.entity_order[1:])
    data_file = project_directory / OUTPUT_RELATIVE_PATH
    palette_file = project_directory / PALETTE_OUTPUT_RELATIVE_PATH
    data_file.parent.mkdir(parents=True, exist_ok=True)
    data_file.write_text(
        build_r_data(
            decay_data,
            entity_order,
            horizon_years=HORIZON_YEARS,
            point_count=POINT_COUNT,
            confidence_level=CONFIDENCE_LEVEL,
            band_run=band_run,
            band_coverage_percent=BAND_COVERAGE_PERCENT,
        ),
        encoding="utf-8",
    )
    palette_file.write_text(build_palette_r(band_run.prepared.palette), encoding="utf-8")
    print(f"Updated {data_file}")
    print(f"Updated {palette_file}")


if __name__ == "__main__":
    main()
