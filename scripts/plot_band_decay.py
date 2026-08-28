"""Run the configured Top-N coverage sensitivity workflow."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from band_decay import (
    AnalysisConfig,
    BetaPrior,
    DecayPriorConfig,
    DirectAsymptote,
    HalfNormalPrior,
    NoTransientTaxa,
    NormalizedDecay,
    PerEntityTopN,
    SensitivityConfig,
    SensitivityResult,
    SensitivityRunner,
    SamplingConfig,
    default_config,
    FixedObservationNoise,
)


class SensitivityApplication:
    """Run and report one configured coverage-sensitivity workflow."""

    def __init__(self, config: AnalysisConfig, sensitivity_config: SensitivityConfig, *, fit: bool = True):
        """Create an application around the library sensitivity runner.

        Args:
            config: Base analysis configuration.
            sensitivity_config: Coverage and stability configuration.
            fit: Whether to run the optional PyMC fits.
        """
        self.runner = SensitivityRunner(config, sensitivity_config)
        self.fit = fit

    def run(self) -> SensitivityResult:
        """Run the workflow and print its summary tables."""
        result = self.runner.run(fit=self.fit)
        for plan in result.plans:
            print(
                f"Coverage {plan.coverage_percent:g}%: "
                f"global local-union Top-N={len(plan.global_selected_serotypes)}; "
                f"country Top-N={dict(plan.country_n)}"
            )
        if not result.fit_summary.empty:
            print("\nSensitivity results (posterior medians):")
            print(result.fit_summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
        print("\nStability results (draw-wise posterior summaries):")
        print(result.stability_summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
        return result


def main() -> None:
    """Configure and run the standalone sensitivity analysis."""
    project_directory = Path(__file__).resolve().parents[1]
    data_path = project_directory / "data/serotype_counts_country_ds_geodate2-2.tsv"
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
    coverage_percentages = (90.0,)
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
    output_directory = project_directory / "plots"
    stability_horizon_years = 20.0
    stability_targets = (0.3, 0.5, 0.7, 0.8, 0.9, 0.95)
    minimum_pairs_for_supported_lag = 4
    fit = True

    priors = DecayPriorConfig(
        y0=BetaPrior(alpha=2.0, beta=2.0),
        b=HalfNormalPrior(sigma=2.0),
        asymptote=DirectAsymptote(BetaPrior(alpha=2.0, beta=2.0)),
        noise=FixedObservationNoise(),
    )
    base_config = default_config(data_path, countries, output_directory=output_directory)
    config = replace(
        base_config,
        top_n=replace(
            base_config.top_n,
            n=0,
            per_country_n={},
            selection=PerEntityTopN(),
            per_country_min_year_count=min_year_count_by_country,
            transient=NoTransientTaxa(),
        ),
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
        plot=replace(
            base_config.plot,
            max_legend_labels=200,
            decay_display=NormalizedDecay(),
        ),
    )
    sensitivity_config = SensitivityConfig(
        coverage_percentages=coverage_percentages,
        output_directory=output_directory,
        filename_template="band_decay_{coverage_percent:g}pct.png",
        stability_horizon_years=stability_horizon_years,
        stability_targets=stability_targets,
        minimum_pairs_for_supported_lag=minimum_pairs_for_supported_lag,
    )
    SensitivityApplication(config, sensitivity_config, fit=fit).run()


if __name__ == "__main__":
    main()
