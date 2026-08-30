from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from .config import AnalysisConfig
from .data import load_counts
from .domain import CoveragePlan, DecayFit, PreparedData, RunResult, SensitivityResult
from .pipeline import DecayAnalysis
from .policies import DecayDisplayPolicy, PosteriorMedian, PosteriorSummaryPolicy, TaxonRanker

FIT_SUMMARY_COLUMNS = (
    "coverage_percent", "country", "top_n", "eligible_taxa",
    "achieved_eligible_share_percent", "analysis_n", "y0", "b", "c", "divergences",
)
STABILITY_COLUMNS = (
    "coverage_percent", "country", "auc_summary", "auc_ci_low", "auc_ci_high",
    "extrapolated_auc_summary", "extrapolated_auc_fraction_summary", "target_mh",
    "p_never_hit", "time_to_target_summary", "time_to_target_ci_low", "time_to_target_ci_high",
)


def top_n_for_coverage(entity, config, coverage_percent: float) -> tuple[int, float, int]:
    """Find the smallest Top-N reaching a target eligible-taxa share.

    Args:
        entity: Prepared entity data.
        config: Top-N configuration.
        coverage_percent: Target cumulative share in percent.

    Returns:
        Top-N value, achieved share, and eligible-taxonomy count.
    """
    counts = config.ranking.counts_for(entity)
    ranker = TaxonRanker()
    eligible = ranker.eligible(
        counts, entity.grouping.transient_serotypes,
        config.min_year_count_for_country(entity.entity),
        config.min_year_percent_for_country(entity.entity),
    )
    ranked = ranker.ranked(counts, eligible)
    totals = counts.reindex(columns=list(ranked), fill_value=0.0).sum(axis=0).to_numpy(dtype=float)
    denominator = float(totals.sum())
    if denominator <= 0 or not ranked:
        return 0, 0.0, len(ranked)
    target = float(coverage_percent) / 100.0
    cumulative = np.cumsum(totals) / denominator
    reached = np.flatnonzero(cumulative >= target)
    count = int(reached[0] + 1) if len(reached) else len(ranked)
    return count, float(cumulative[count - 1]), len(ranked)


def selected_taxa_for_country(entity, config, n: int) -> tuple[str, ...]:
    """Return the highest-ranked eligible taxa for one entity."""
    counts = config.ranking.counts_for(entity)
    ranker = TaxonRanker()
    eligible = ranker.eligible(
        counts, entity.grouping.transient_serotypes,
        config.min_year_count_for_country(entity.entity),
        config.min_year_percent_for_country(entity.entity),
    )
    ranked = ranker.ranked(counts, eligible)
    return ranked if n == 0 else ranked[:n]


def build_coverage_plan(prepared: PreparedData, config: AnalysisConfig, coverage_percent: float) -> CoveragePlan:
    """Build one country-specific Top-N plan for a coverage target."""
    country_n: dict[str, int] = {}
    achieved: dict[str, float] = {}
    eligible: dict[str, int] = {}
    for country in config.input.countries:
        if country not in prepared.entities:
            continue
        count, share, eligible_count = top_n_for_coverage(prepared.entities[country], config.top_n, coverage_percent)
        country_n[country] = count
        achieved[country] = share
        eligible[country] = eligible_count
    selected_union = tuple(sorted({
        taxon
        for country, count in country_n.items()
        for taxon in selected_taxa_for_country(prepared.entities[country], config.top_n, count)
    }))
    return CoveragePlan(
        coverage_percent=float(coverage_percent),
        country_n=country_n,
        achieved_share=achieved,
        eligible_count=eligible,
        global_selected_serotypes=selected_union,
    )


def _fit_summary_rows(
    result: RunResult,
    plan: CoveragePlan,
    summary_policy: PosteriorSummaryPolicy,
) -> list[dict[str, object]]:
    rows = []
    for country in result.config.input.countries:
        if country not in result.prepared.entities:
            continue
        fit = result.decay_data[country].fit
        row: dict[str, object] = {
            "coverage_percent": plan.coverage_percent,
            "country": country,
            "top_n": int(plan.country_n[country]),
            "eligible_taxa": int(plan.eligible_count[country]),
            "achieved_eligible_share_percent": 100 * float(plan.achieved_share[country]),
            "analysis_n": int(round(result.prepared.entities[country].analysis_total)),
            "y0": np.nan,
            "b": np.nan,
            "c": np.nan,
            "divergences": np.nan,
        }
        if fit is not None:
            row.update({
                "y0": summary_policy.scalar(fit.y0_samples),
                "b": summary_policy.scalar(fit.b_samples),
                "c": summary_policy.scalar(fit.c_samples),
                "divergences": int(fit.divergences),
            })
        rows.append(row)
    return rows


def _integral_of_decay(duration: float, b: np.ndarray) -> np.ndarray:
    if duration <= 0:
        return np.zeros_like(b, dtype=float)
    scaled = b * float(duration)
    small = np.abs(scaled) < 1e-5
    result = np.empty_like(b, dtype=float)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        result[small] = float(duration) * (1 - scaled[small] / 2 + scaled[small] ** 2 / 6 - scaled[small] ** 3 / 24 + scaled[small] ** 4 / 120)
        result[~small] = -np.expm1(-scaled[~small]) / b[~small]
    return result


def _auc_draws(y0: np.ndarray, b: np.ndarray, c: np.ndarray, start: float, end: float) -> np.ndarray:
    # Integrate the exponential analytically for every posterior draw.
    duration = max(0.0, float(end) - float(start))
    if duration == 0:
        return np.zeros_like(y0, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        start_decay = np.exp(-b * float(start))
    return c * duration + (y0 - c) * start_decay * _integral_of_decay(duration, b)


def largest_supported_lag(x_values: np.ndarray, minimum_pairs: int) -> float:
    """Return the largest lag supported by the requested pair count."""
    values = np.asarray(x_values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if not len(values):
        return 0.0
    lags, counts = np.unique(values, return_counts=True)
    supported = lags[counts >= int(minimum_pairs)]
    return float(np.max(supported if len(supported) else lags))


def _displayed_decay_parameters(fit: DecayFit, display_policy: DecayDisplayPolicy) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return display_policy.posterior_parameters(fit)


def stability_metrics(
    fit: DecayFit,
    x_pairs: np.ndarray,
    *,
    horizon_years: float,
    target_mh: float,
    minimum_pairs_for_supported_lag: int,
    display_policy: DecayDisplayPolicy,
    summary_policy: PosteriorSummaryPolicy | None = None,
) -> dict[str, float]:
    """Summarize fitted decay stability, AUC, and target-crossing time.

    Args:
        fit: Posterior decay fit.
        x_pairs: Observed lag values.
        horizon_years: Integration horizon.
        target_mh: Similarity target for crossing-time metrics.
        minimum_pairs_for_supported_lag: Pair threshold for extrapolation.
        display_policy: Policy selecting the displayed curve scale.
        summary_policy: Policy selecting the scalar summary statistic.

    Returns:
        Scalar stability metrics suitable for tabular reporting.
    """
    summary = summary_policy or PosteriorMedian()
    y0, b, c = _displayed_decay_parameters(fit, display_policy)
    auc = _auc_draws(y0, b, c, 0.0, horizon_years)
    supported_lag = largest_supported_lag(x_pairs, minimum_pairs_for_supported_lag)
    extrapolated_auc = _auc_draws(y0, b, c, supported_lag, horizon_years)
    fraction = np.divide(extrapolated_auc, auc, out=np.zeros_like(auc), where=auc != 0)
    never_hit = np.ones(len(b), dtype=bool)
    times = np.full(len(b), np.nan, dtype=float)
    already = y0 <= float(target_mh)
    times[already] = 0.0
    never_hit[already] = False
    crossing = (~already) & (b > 0) & (c < float(target_mh))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        times[crossing] = np.log1p((y0[crossing] - target_mh) / (target_mh - c[crossing])) / b[crossing]
    valid_times = times[np.isfinite(times) & (times >= 0)]
    time_summary = summary.scalar(valid_times) if len(valid_times) else np.nan
    time_low, time_high = np.quantile(valid_times, [0.025, 0.975]) if len(valid_times) else (np.nan, np.nan)
    finite_auc = auc[np.isfinite(auc)]
    finite_extra = extrapolated_auc[np.isfinite(extrapolated_auc)]
    finite_fraction = fraction[np.isfinite(fraction)]
    if not len(finite_auc):
        raise ValueError("DecayFit produced no finite AUC draws.")
    return {
        "auc_summary": summary.scalar(finite_auc),
        "auc_ci_low": float(np.quantile(finite_auc, 0.025)),
        "auc_ci_high": float(np.quantile(finite_auc, 0.975)),
        "extrapolated_auc_summary": summary.scalar(finite_extra) if len(finite_extra) else np.nan,
        "extrapolated_auc_fraction_summary": summary.scalar(finite_fraction) if len(finite_fraction) else np.nan,
        "target_mh": float(target_mh),
        "p_never_hit": float(np.mean(never_hit)),
        "time_to_target_summary": time_summary,
        "time_to_target_ci_low": float(time_low),
        "time_to_target_ci_high": float(time_high),
    }


def build_stability_table(
    fitted_results: Sequence[tuple[float, RunResult]],
    *,
    horizon_years: float = 20.0,
    target_mh: float | Iterable[float] = (0.3, 0.5, 0.7, 0.8, 0.9, 0.95),
    minimum_pairs_for_supported_lag: int = 4,
    summary_policy: PosteriorSummaryPolicy | None = None,
) -> pd.DataFrame:
    """Build stability metrics using the configured posterior summary policy."""
    targets = (float(target_mh),) if np.isscalar(target_mh) else tuple(float(value) for value in target_mh)
    if not targets or any(not 0 <= target <= 1 for target in targets):
        raise ValueError("target_mh must contain values in [0, 1].")
    if float(horizon_years) <= 0 or int(minimum_pairs_for_supported_lag) < 1:
        raise ValueError("horizon_years must be positive and minimum pair count at least one.")
    summary = summary_policy or PosteriorMedian()
    rows = []
    for coverage_percent, result in fitted_results:
        for target in targets:
            for country in result.prepared.entity_order:
                if country == "GLOBAL":
                    continue
                row = {"coverage_percent": float(coverage_percent), "country": country, "target_mh": target}
                row.update({column: np.nan for column in STABILITY_COLUMNS if column not in row})
                decay = result.decay_data[country]
                if decay.fit is not None:
                    row.update(stability_metrics(
                        decay.fit, decay.x, horizon_years=float(horizon_years), target_mh=target,
                        minimum_pairs_for_supported_lag=int(minimum_pairs_for_supported_lag),
                        display_policy=result.config.plot.decay_display,
                        summary_policy=summary,
                    ))
                rows.append(row)
    return pd.DataFrame(rows, columns=list(STABILITY_COLUMNS))


class SensitivityRunner:
    """Run coverage plans and collect fit and stability summaries."""

    def __init__(self, config: AnalysisConfig, sensitivity_config, *, fitter=None, palette_builder=None):
        """Create a sensitivity runner with optional injected services."""
        self.config = config
        self.sensitivity_config = sensitivity_config
        self.fitter = fitter
        self.palette_builder = palette_builder

    def run(self, raw_counts: pd.DataFrame | None = None, *, fit: bool = False) -> SensitivityResult:
        """Execute all configured coverage levels.

        Args:
            raw_counts: Optional long-form count dataframe.
            fit: Whether to fit decay curves for each coverage level.

        Returns:
            Coverage plans, run results, and summary tables.
        """
        if raw_counts is None:
            if self.config.input.data_path is None:
                raise ValueError("input.data_path is required when raw_counts is not provided.")
            raw_counts = load_counts(self.config.input.data_path)
        base_analysis = DecayAnalysis(self.config, palette_builder=self.palette_builder)
        prepared = base_analysis.prepare(raw_counts)
        plans = tuple(build_coverage_plan(prepared, self.config, value) for value in self.sensitivity_config.coverage_percentages)
        stable_labels = tuple(dict.fromkeys(label for plan in plans for label in plan.global_selected_serotypes))
        runs = []
        fit_rows = []
        # Reuse one label universe so colors remain stable across coverage runs.
        for plan in plans:
            variant = self.config.with_top_n(
                n=len(plan.global_selected_serotypes),
                per_country_n=dict(plan.country_n),
                global_selected_serotypes=plan.global_selected_serotypes,
            )
            output_path = Path(self.sensitivity_config.output_directory) / self.sensitivity_config.filename_template.format(
                coverage=plan.coverage_percent,
                coverage_percent=plan.coverage_percent,
            )
            result = DecayAnalysis(variant, fitter=self.fitter, palette_builder=self.palette_builder).run(
                raw_counts, fit=fit, output_path=output_path, palette_master_labels=stable_labels,
            )
            runs.append(result)
            fit_rows.extend(_fit_summary_rows(result, plan, self.sensitivity_config.fit_summary))
        fit_summary = pd.DataFrame(fit_rows, columns=list(FIT_SUMMARY_COLUMNS))
        stability_summary = build_stability_table(
            tuple(zip((plan.coverage_percent for plan in plans), runs, strict=True)),
            horizon_years=self.sensitivity_config.stability_horizon_years,
            target_mh=self.sensitivity_config.stability_targets,
            minimum_pairs_for_supported_lag=self.sensitivity_config.minimum_pairs_for_supported_lag,
            summary_policy=self.sensitivity_config.stability_summary,
        )
        return SensitivityResult(plans=plans, runs=tuple(runs), fit_summary=fit_summary, stability_summary=stability_summary)


def run_sensitivity(config: AnalysisConfig, sensitivity_config, *, raw_counts: pd.DataFrame | None = None, fit: bool = False, fitter=None) -> SensitivityResult:
    """Run sensitivity analysis through the functional API."""
    return SensitivityRunner(config, sensitivity_config, fitter=fitter).run(raw_counts, fit=fit)
