"""Fit and plot one summarized decay curve for each configured country."""

from __future__ import annotations

from pathlib import Path

from band_decay import (
    AnalysisConfig,
    BetaPrior,
    CentralConfidenceInterval,
    CurvePlotConfig,
    DecayAnalysis,
    DecayPriorConfig,
    DashedExtrapolation,
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
    SamplingConfig,
    SkipTaxa,
    TopNConfig,
    QualifyingYears,
    UnionAvailableYears,
    YearSelectionConfig,
    render_fitted_curves, PosteriorMedian,
)


def build_config(project_directory: Path) -> AnalysisConfig:
    """Build the analysis configuration edited by the user."""
    countries = (
        "Greece",
        "Italy",
        "Spain",
        "Russia",
        "United Kingdom",
        "France",
        "Germany",
        "Switzerland",
    )
    min_year_count_by_country = {
        "United Kingdom": 2.0,
        "Switzerland": 2.0,
        "Germany": 2.0,
        "Greece": 3.0,
        "Italy": 3.0,
        "Spain": 3.0,
        "Russia": 3.0,
        "France": 3.0,
    }
    priors = DecayPriorConfig(
        y0=BetaPrior(alpha=2.0, beta=2.0),
        b=HalfNormalPrior(sigma=2.0),
        asymptote=DirectAsymptote(BetaPrior(alpha=2.0, beta=2.0)),
        noise=FixedObservationNoise(),
    )
    return AnalysisConfig(
        input=InputConfig(
            data_path=project_directory / "data/serotype_counts_country_ds_geodate2-2.tsv",
            countries=countries,
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
            per_country_min_year_count=min_year_count_by_country,
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
        plot=PlotConfig(),
        output=OutputConfig(output_directory=project_directory / "plots"),
    )


def main() -> None:
    """Fit configured countries and save their separate curve plot."""
    project_directory = Path(__file__).resolve().parents[1]
    config = build_config(project_directory)
    analysis = DecayAnalysis(config)
    prepared = analysis.prepare()
    decay_data = analysis.fit(prepared)
    render_fitted_curves(
        decay_data,
        CurvePlotConfig(
            output_path=project_directory / "plots/fitted_country_curves.png",
            horizon_years=20.0,
            display=NormalizedDecay(),
            fit_summary=PosteriorMedian(),
            confidence_interval=CentralConfidenceInterval(),
            extrapolation=DashedExtrapolation(),
        ),
        entity_order=prepared.entity_order[1:],
    )



if __name__ == "__main__":
    main()
