from __future__ import annotations

import numpy as np
import pandas as pd

from band_decay import (
    AnalysisConfig,
    DecayFit,
    AllAvailableYears,
    EntityAvailableYears,
    CollapseTaxa,
    NoTransientTaxa,
    OriginalDecay,
    PerEntityTopN,
    InputConfig,
    MorisitaHornConfig,
    OutputConfig,
    PaletteBuilder,
    PaletteSettings,
    PlotConfig,
    SamplingConfig,
    SensitivityConfig,
    SensitivityRunner,
    TopNConfig,
    YearSelectionConfig,
    stability_metrics,
)


def _frame() -> pd.DataFrame:
    rows = []
    for country in ("A", "B"):
        for year in (2000, 2001, 2002, 2003):
            for label, count in (("S1", 100), ("S2", 50), ("S3", 25)):
                rows.append({"country": country, "collection_year": year, "serotype": label, "count": count})
    return pd.DataFrame(rows)


def _config(tmp_path) -> AnalysisConfig:
    return AnalysisConfig(
        input=InputConfig(countries=("A", "B")),
        year_selection=YearSelectionConfig(min_count_per_year=0, selection=AllAvailableYears(), display_axis=EntityAvailableYears()),
        top_n=TopNConfig(n=0, selection=PerEntityTopN(), transient=NoTransientTaxa()),
        mh=MorisitaHornConfig(other_grouping=CollapseTaxa(), transient_grouping=CollapseTaxa()),
        sampling=SamplingConfig(draws=1, tune=0, chains=1, cores=1),
        plot=PlotConfig(dpi=60),
        output=OutputConfig(output_directory=tmp_path, filename_template="ignored.pdf"),
    )


class FakeFitter:
    def fit(self, x, y, sampling, seed):
        return DecayFit(
            y0_samples=np.asarray([0.9, 0.92, 0.88]),
            b_samples=np.asarray([0.1, 0.11, 0.09]),
            c_samples=np.asarray([0.1, 0.12, 0.08]),
        )


def test_sensitivity_returns_coverage_plans_runs_and_tables(tmp_path) -> None:
    sensitivity = SensitivityConfig(
        coverage_percentages=(50.0, 90.0),
        output_directory=tmp_path,
        stability_targets=(0.5,),
        stability_horizon_years=20,
        minimum_pairs_for_supported_lag=1,
    )
    result = SensitivityRunner(
        _config(tmp_path), sensitivity, fitter=FakeFitter(),
        palette_builder=PaletteBuilder(PaletteSettings(candidate_count=32)),
    ).run(_frame(), fit=True)
    assert [plan.coverage_percent for plan in result.plans] == [50.0, 90.0]
    assert len(result.runs) == 2
    assert set(result.fit_summary["country"]) == {"A", "B"}
    assert set(result.stability_summary["target_mh"]) == {0.5}
    for plan, run in zip(result.plans, result.runs, strict=True):
        assert run.prepared.entities["GLOBAL"].grouping.selected_serotypes == plan.global_selected_serotypes
    assert all(run.output_path is not None and run.output_path.exists() for run in result.runs)


def test_stability_handles_zero_decay_and_already_reached_target() -> None:
    fit = DecayFit(
        y0_samples=np.asarray([0.4, 0.8]),
        b_samples=np.asarray([0.0, 0.0]),
        c_samples=np.asarray([0.2, 0.2]),
    )
    metrics = stability_metrics(
        fit, np.asarray([1, 1, 2]), horizon_years=10, target_mh=0.5,
        minimum_pairs_for_supported_lag=1, display_policy=OriginalDecay(),
    )
    assert metrics["time_to_target_mean"] == 0.0
    assert metrics["auc_median"] > 0
