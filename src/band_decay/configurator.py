"""Notebook controls for composing standalone analysis policies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace
from pathlib import Path

import pandas as pd

from .config import (
    AnalysisConfig,
    InputConfig,
    MorisitaHornConfig,
    OutputConfig,
    PlotConfig,
    SamplingConfig,
    TopNConfig,
    YearSelectionConfig,
)
from .pipeline import DecayAnalysis
from .priors import (
    BetaPrior,
    DecayPriorConfig,
    DirectAsymptote,
    FixedObservationNoise,
    FixedValuePrior,
    HalfNormalPrior,
    RelativeAsymptote,
    SampledObservationNoise,
)
from .policies import (
    AllAvailableYears,
    AvailableYearRanking,
    CollapseTaxa,
    EntityAvailableYears,
    GlobalTopN,
    GlobalTransientTaxa,
    KeepTaxa,
    LargestContiguousBlock,
    NoTransientTaxa,
    NormalizedDecay,
    OriginalDecay,
    PerEntityTopN,
    PerEntityTransientTaxa,
    QualifyingYears,
    SelectedYearRanking,
    SkipTaxa,
    UnionAvailableYears,
)


class _Y0PriorEditor(ABC):
    """Create one of the supported intercept prior objects."""

    @abstractmethod
    def create(self, *, alpha: float, beta: float, value: float):
        """Create the configured intercept prior."""


class _BetaY0PriorEditor(_Y0PriorEditor):
    """Edit a Beta prior for ``y0``."""

    def option_name(self) -> str:
        """Return the represented prior type."""
        return "BetaPrior"

    def create(self, *, alpha: float, beta: float, value: float):
        """Create the configured Beta prior."""
        return BetaPrior(alpha=alpha, beta=beta)


class _FixedY0PriorEditor(_Y0PriorEditor):
    """Edit a fixed prior for ``y0``."""

    def option_name(self) -> str:
        """Return the represented prior type."""
        return "FixedValuePrior"

    def create(self, *, alpha: float, beta: float, value: float):
        """Create the configured fixed prior."""
        return FixedValuePrior(value=value)


class _AsymptoteEditor(ABC):
    """Create a direct or relative asymptote strategy."""

    @abstractmethod
    def create(self, prior):
        """Create the configured asymptote strategy."""


class _DirectAsymptoteEditor(_AsymptoteEditor):
    """Edit a direct asymptote strategy."""

    def option_name(self) -> str:
        """Return the represented asymptote type."""
        return "DirectAsymptote"

    def create(self, prior):
        """Create the configured direct strategy."""
        return DirectAsymptote(prior)


class _RelativeAsymptoteEditor(_AsymptoteEditor):
    """Edit a relative asymptote strategy."""

    def option_name(self) -> str:
        """Return the represented asymptote type."""
        return "RelativeAsymptote"

    def create(self, prior):
        """Create the configured relative strategy."""
        return RelativeAsymptote(prior)


class _NoiseEditor(ABC):
    """Create fixed or sampled observation noise."""

    @abstractmethod
    def create(self, sigma: float):
        """Create the configured observation-noise strategy."""


class _FixedNoiseEditor(_NoiseEditor):
    """Edit fixed observation noise."""

    def option_name(self) -> str:
        """Return the represented noise type."""
        return "FixedObservationNoise"

    def create(self, sigma: float):
        """Create fixed observation noise."""
        return FixedObservationNoise()


class _SampledNoiseEditor(_NoiseEditor):
    """Edit sampled observation noise."""

    def option_name(self) -> str:
        """Return the represented noise type."""
        return "SampledObservationNoise"

    def create(self, sigma: float):
        """Create sampled observation noise."""
        return SampledObservationNoise(HalfNormalPrior(sigma=sigma))


def _policy_options(candidates, current):
    """Return labeled policy instances with the current choice selected."""
    options = list(candidates)
    if not any(option.option_name() == current.option_name() for option in options):
        options.insert(0, current)
    return tuple((option.option_name(), option) for option in options), next(
        option for option in options if option.option_name() == current.option_name()
    )


class WidgetConfigurator:
    """Compose immutable analysis configuration from notebook widgets."""

    def __init__(self, config: AnalysisConfig, *, fit_enabled: bool = False):
        """Create notebook controls for an analysis configuration.

        Args:
            config: Initial immutable analysis configuration.
            fit_enabled: Whether preview actions initially enable PyMC fitting.

        Raises:
            ImportError: If notebook dependencies are unavailable.
        """
        try:
            import ipywidgets as widgets
        except ImportError as exc:
            raise ImportError("The widget configurator requires the dependencies installed with: pip install -r requirements.txt") from exc
        self._widgets = widgets
        self.base_config = config
        self._per_country_widgets = {}
        self._per_country_cache = {}
        self.country_data_path = widgets.Text(
            value=str(config.input.data_path or ""),
            description="Data path",
            layout=widgets.Layout(width="620px"),
        )
        self.refresh_countries_button = widgets.Button(description="Refresh countries")
        options = self.available_countries(self.country_data_path.value)
        selected = tuple(country for country in config.input.countries if country in options)
        self.countries = widgets.SelectMultiple(
            options=options,
            value=selected,
            description="Countries",
            rows=10,
            layout=widgets.Layout(width="420px"),
        )
        self.min_count_per_year = widgets.IntText(value=config.year_selection.min_count_per_year, description="Min/year")
        self.year_selection = self._dropdown(
            (AllAvailableYears(), QualifyingYears(), LargestContiguousBlock()), config.year_selection.selection, "Year policy"
        )
        self.display_axis = self._dropdown(
            (EntityAvailableYears(), UnionAvailableYears()), config.year_selection.display_axis, "Axis policy"
        )
        self.top_n = widgets.IntSlider(value=config.top_n.n, min=0, max=50, step=1, description="Top N", continuous_update=False)
        self.top_n_selection = self._dropdown(
            (GlobalTopN(), PerEntityTopN()), config.top_n.selection, "Top-N policy"
        )
        self.ranking = self._dropdown(
            (SelectedYearRanking(), AvailableYearRanking()), config.top_n.ranking, "Rank policy"
        )
        self.min_year_count = widgets.FloatText(value=config.top_n.min_year_count, description="Min count")
        self.min_year_percent = widgets.FloatSlider(
            value=config.top_n.min_year_percent,
            min=0.0,
            max=100.0,
            step=0.5,
            description="Min %",
            continuous_update=False,
        )
        self.transient = self._dropdown(
            (NoTransientTaxa(), GlobalTransientTaxa(), PerEntityTransientTaxa()), config.top_n.transient, "Transient policy"
        )
        self.per_country_top_n_box = widgets.VBox()
        self.mh_other_grouping = self._dropdown(
            (CollapseTaxa(), KeepTaxa(), SkipTaxa()), config.mh.other_grouping, "Other grouping"
        )
        self.mh_transient_grouping = self._dropdown(
            (CollapseTaxa(), KeepTaxa(), SkipTaxa()), config.mh.transient_grouping, "Transient grouping"
        )
        self.no_max_lag = widgets.Checkbox(value=config.mh.max_lag is None, description="No max lag")
        self.max_lag = widgets.IntText(value=config.mh.max_lag or 10, description="Max lag")
        self.run_pymc_fit = widgets.Checkbox(value=fit_enabled, description="Run PyMC fitting")
        self.draws = widgets.IntText(value=config.sampling.draws, description="Draws")
        self.tune = widgets.IntText(value=config.sampling.tune, description="Tune")
        self.chains = widgets.IntText(value=config.sampling.chains, description="Chains")
        self.cores = widgets.IntText(value=config.sampling.cores, description="Cores")
        self.target_accept = widgets.FloatSlider(
            value=config.sampling.target_accept,
            min=0.5,
            max=0.999,
            step=0.001,
            readout_format=".3f",
            description="Target acc.",
            continuous_update=False,
        )
        self.seed = widgets.IntText(value=config.sampling.seed, description="Seed")
        sigma_value = "" if config.sampling.observation_sigma is None else str(config.sampling.observation_sigma)
        self.observation_sigma = widgets.Text(value=sigma_value, description="Obs sigma")
        y0_prior = config.sampling.priors.y0
        asymptote_prior = config.sampling.priors.asymptote
        noise_prior = config.sampling.priors.noise
        asymptote_parameter_prior = getattr(
            asymptote_prior, "prior", getattr(asymptote_prior, "ratio_prior", BetaPrior(1.0, 1.0))
        )
        noise_parameter_prior = getattr(noise_prior, "prior", HalfNormalPrior(0.1))
        self.y0_prior = self._dropdown(
            (_BetaY0PriorEditor(), _FixedY0PriorEditor()), y0_prior, "y0 prior"
        )
        self.y0_alpha = widgets.FloatText(value=float(getattr(y0_prior, "alpha", 1.0)), description="y0 α")
        self.y0_beta = widgets.FloatText(value=float(getattr(y0_prior, "beta", 1.0)), description="y0 β")
        self.y0_value = widgets.FloatText(value=float(getattr(y0_prior, "value", 1.0)), description="y0 value")
        self.b_sigma = widgets.FloatText(value=float(getattr(config.sampling.priors.b, "sigma", 2.0)), description="b sigma")
        self.asymptote = self._dropdown(
            (_DirectAsymptoteEditor(), _RelativeAsymptoteEditor()),
            asymptote_prior,
            "Asymptote",
        )
        self.asymptote_alpha = widgets.FloatText(value=float(getattr(asymptote_parameter_prior, "alpha", 1.0)), description="c α")
        self.asymptote_beta = widgets.FloatText(value=float(getattr(asymptote_parameter_prior, "beta", 1.0)), description="c β")
        self.noise = self._dropdown(
            (_FixedNoiseEditor(), _SampledNoiseEditor()), noise_prior, "Noise"
        )
        self.noise_sigma = widgets.FloatText(value=float(getattr(noise_parameter_prior, "sigma", 0.1)), description="noise sigma")
        self.dpi = widgets.IntText(value=config.plot.dpi, description="DPI")
        self.max_legend_labels = widgets.IntText(value=config.plot.max_legend_labels, description="Legend max")
        self.count_label_max_y_fraction = widgets.FloatSlider(
            value=config.plot.count_label_max_y_fraction,
            min=0.0,
            max=1.5,
            step=0.05,
            description="Label frac.",
            continuous_update=False,
        )
        self.count_label_max_years = widgets.IntText(value=config.plot.count_label_max_years, description="Label years")
        self.strike_excluded_year_labels = widgets.Checkbox(
            value=config.plot.strike_excluded_year_labels, description="Strike excluded labels"
        )
        self.excluded_year_alpha = widgets.FloatSlider(
            value=config.plot.excluded_year_alpha,
            min=0.0,
            max=1.0,
            step=0.02,
            description="Mask alpha",
            continuous_update=False,
        )
        self.excluded_year_hatch = widgets.Text(value=config.plot.excluded_year_hatch, description="Mask hatch")
        self.decay_display = self._dropdown(
            (OriginalDecay(), NormalizedDecay()), config.plot.decay_display, "Decay display"
        )
        self.output_directory = widgets.Text(
            value=str(config.output.output_directory), description="Output dir", layout=widgets.Layout(width="500px")
        )
        self.save_button = widgets.Button(description="Save plot", button_style="primary")
        self.run_output = widgets.Output()
        self.refresh_countries_button.on_click(self.refresh_countries)
        self.save_button.on_click(self.save_current_config)
        self.countries.observe(self._rebuild_per_country_controls, names="value")
        self.top_n_selection.observe(self._rebuild_per_country_controls, names="value")
        self._rebuild_per_country_controls()
        for control in self._auto_render_controls():
            control.observe(self.render_current_config, names="value")

    def _dropdown(self, candidates, current, description):
        options, value = _policy_options(candidates, current)
        return self._widgets.Dropdown(options=options, value=value, description=description)

    def available_countries(self, path: str) -> tuple[str, ...]:
        """Read available country labels from a dataset path."""
        dataset = Path(path)
        if not dataset.exists():
            return tuple(self.base_config.input.countries)
        frame = pd.read_csv(dataset, sep="\t", usecols=["country"])
        return tuple(sorted(frame["country"].dropna().astype(str).unique()))

    def refresh_countries(self, _button=None) -> None:
        """Refresh country choices from the current data path."""
        options = self.available_countries(self.country_data_path.value)
        self.countries.options = options
        self.countries.value = tuple(country for country in self._selected_countries() if country in options)

    def _selected_countries(self) -> tuple[str, ...]:
        selected = tuple(str(country) for country in self.countries.value)
        return selected if selected else self.base_config.input.countries

    def _store_country_values(self) -> None:
        for country, controls in self._per_country_widgets.items():
            top_n, minimum_count, minimum_percent = controls
            self._per_country_cache[country] = (int(top_n.value), float(minimum_count.value), float(minimum_percent.value))

    def _country_overrides(self) -> tuple[dict[str, int], dict[str, float], dict[str, float]]:
        self._store_country_values()
        countries = set(self._selected_countries())
        values = {country: data for country, data in self._per_country_cache.items() if country in countries}
        return (
            {country: data[0] for country, data in values.items()},
            {country: data[1] for country, data in values.items()},
            {country: data[2] for country, data in values.items()},
        )

    def _rebuild_per_country_controls(self, _change=None) -> None:
        widgets = self._widgets
        self._store_country_values()
        rows = [widgets.HTML("<b>Per-country Top-N overrides</b>")]
        self._per_country_widgets.clear()
        for country in self._selected_countries():
            defaults = self._per_country_cache.get(
                country,
                (
                    self.base_config.top_n.per_country_n.get(country, int(self.top_n.value)),
                    self.base_config.top_n.per_country_min_year_count.get(country, float(self.min_year_count.value)),
                    self.base_config.top_n.per_country_min_year_percent.get(country, float(self.min_year_percent.value)),
                ),
            )
            controls = (
                widgets.IntText(value=int(defaults[0]), description="Top N", layout=widgets.Layout(width="140px")),
                widgets.FloatText(value=float(defaults[1]), description="Min count", layout=widgets.Layout(width="175px")),
                widgets.FloatSlider(
                    value=float(defaults[2]),
                    min=0.0,
                    max=100.0,
                    step=0.5,
                    description="Min %",
                    continuous_update=False,
                    layout=widgets.Layout(width="280px"),
                ),
            )
            self._per_country_widgets[country] = controls
            for control in controls:
                control.observe(self.render_current_config, names="value")
            rows.append(widgets.HBox([widgets.HTML(f"<b>{country}</b>", layout=widgets.Layout(width="150px")), *controls]))
        self.per_country_top_n_box.children = tuple(rows)
        self.per_country_top_n_box.layout.display = "" if self.top_n_selection.value.supports_country_overrides() else "none"

    def build_config(self) -> AnalysisConfig:
        """Build an immutable analysis configuration from widget values."""
        per_country_n, per_country_count, per_country_percent = self._country_overrides()
        sigma_text = self.observation_sigma.value.strip()
        sigma = float(sigma_text) if sigma_text else None
        max_lag = None if self.no_max_lag.value else int(self.max_lag.value)
        asymptote_parameter_prior = BetaPrior(
            alpha=float(self.asymptote_alpha.value), beta=float(self.asymptote_beta.value)
        )
        priors = DecayPriorConfig(
            y0=self.y0_prior.value.create(
                alpha=float(self.y0_alpha.value), beta=float(self.y0_beta.value), value=float(self.y0_value.value)
            ),
            b=HalfNormalPrior(sigma=float(self.b_sigma.value)),
            asymptote=self.asymptote.value.create(asymptote_parameter_prior),
            noise=self.noise.value.create(float(self.noise_sigma.value)),
        )
        return AnalysisConfig(
            input=InputConfig(data_path=Path(self.country_data_path.value), countries=self._selected_countries()),
            year_selection=YearSelectionConfig(
                min_count_per_year=int(self.min_count_per_year.value),
                selection=self.year_selection.value,
                display_axis=self.display_axis.value,
            ),
            top_n=TopNConfig(
                n=int(self.top_n.value),
                per_country_n=per_country_n,
                selection=self.top_n_selection.value,
                ranking=self.ranking.value,
                min_year_count=float(self.min_year_count.value),
                per_country_min_year_count=per_country_count,
                min_year_percent=float(self.min_year_percent.value),
                per_country_min_year_percent=per_country_percent,
                transient=self.transient.value,
            ),
            mh=MorisitaHornConfig(
                other_grouping=self.mh_other_grouping.value,
                transient_grouping=self.mh_transient_grouping.value,
                max_lag=max_lag,
            ),
            sampling=SamplingConfig(
                draws=int(self.draws.value),
                tune=int(self.tune.value),
                chains=int(self.chains.value),
                cores=int(self.cores.value),
                target_accept=float(self.target_accept.value),
                seed=int(self.seed.value),
                observation_sigma=sigma,
                priors=priors,
            ),
            plot=PlotConfig(
                dpi=int(self.dpi.value),
                show=False,
                max_legend_labels=int(self.max_legend_labels.value),
                count_label_max_y_fraction=float(self.count_label_max_y_fraction.value),
                count_label_max_years=int(self.count_label_max_years.value),
                strike_excluded_year_labels=bool(self.strike_excluded_year_labels.value),
                excluded_year_alpha=float(self.excluded_year_alpha.value),
                excluded_year_hatch=str(self.excluded_year_hatch.value),
                decay_display=self.decay_display.value,
            ),
            output=OutputConfig(
                output_directory=Path(self.output_directory.value), filename_template="region_country_band_decay2.png"
            ),
        )

    def _auto_render_controls(self):
        return (
            self.country_data_path, self.countries, self.min_count_per_year, self.year_selection, self.display_axis,
            self.top_n, self.top_n_selection, self.ranking, self.min_year_count, self.min_year_percent, self.transient,
            self.mh_other_grouping, self.mh_transient_grouping, self.no_max_lag, self.max_lag, self.run_pymc_fit,
            self.draws, self.tune, self.chains, self.cores, self.target_accept, self.seed, self.observation_sigma,
            self.y0_prior, self.y0_alpha, self.y0_beta, self.y0_value, self.b_sigma, self.asymptote,
            self.asymptote_alpha, self.asymptote_beta, self.noise, self.noise_sigma,
            self.dpi, self.max_legend_labels, self.count_label_max_y_fraction, self.count_label_max_years,
            self.strike_excluded_year_labels, self.excluded_year_alpha, self.excluded_year_hatch, self.output_directory,
            self.decay_display,
        )

    def _run(self, output_path=None) -> None:
        from IPython.display import clear_output

        with self.run_output:
            clear_output(wait=True)
            try:
                config = self.build_config()
                config = replace(config, plot=replace(config.plot, show=True))
                result = DecayAnalysis(config).run(
                    fit=bool(self.run_pymc_fit.value), output_path=output_path, save=output_path is not None
                )
            except Exception as exc:
                print(f"{type(exc).__name__}: {exc}")
                return
            print("Saved: " + str(result.output_path) if output_path is not None else "Preview rendered; click Save plot to write the image.")
            print(f"Entities: {len(result.prepared.entity_order)} ({', '.join(result.prepared.entity_order)})")
            print(f"Layers: {len(result.prepared.master_display_columns)}")

    def render_current_config(self, _change=None) -> None:
        """Render a preview using the current widget values."""
        self._run()

    def save_current_config(self, _button=None) -> None:
        """Save a plot using the current widget values."""
        self._run(Path(self.output_directory.value) / "region_country_band_decay2.png")

    @property
    def layout(self):
        """Return the complete widget layout."""
        widgets = self._widgets
        return widgets.VBox([
            widgets.VBox([widgets.HTML("<b>Input</b>"), widgets.HBox([self.country_data_path, self.refresh_countries_button]), self.countries]),
            widgets.VBox([widgets.HTML("<b>Year Selection</b>"), widgets.HBox([self.min_count_per_year, self.year_selection, self.display_axis])]),
            widgets.VBox([widgets.HTML("<b>Top-N Grouping</b>"), widgets.HBox([self.top_n, self.top_n_selection, self.ranking]), widgets.HBox([self.min_year_count, self.min_year_percent, self.transient]), self.per_country_top_n_box]),
            widgets.VBox([widgets.HTML("<b>Morisita-Horn</b>"), widgets.HBox([self.mh_other_grouping, self.mh_transient_grouping, self.no_max_lag, self.max_lag])]),
            widgets.VBox([widgets.HTML("<b>Sampling</b>"), self.run_pymc_fit, widgets.HBox([self.draws, self.tune, self.chains, self.cores]), widgets.HBox([self.target_accept, self.seed, self.observation_sigma]), widgets.HBox([self.y0_prior, self.y0_alpha, self.y0_beta, self.y0_value, self.b_sigma]), widgets.HBox([self.asymptote, self.asymptote_alpha, self.asymptote_beta, self.noise, self.noise_sigma])]),
            widgets.VBox([widgets.HTML("<b>Plot</b>"), widgets.HBox([self.dpi, self.max_legend_labels, self.count_label_max_y_fraction, self.count_label_max_years]), widgets.HBox([self.strike_excluded_year_labels, self.excluded_year_alpha, self.excluded_year_hatch, self.decay_display])]),
            widgets.VBox([widgets.HTML("<b>Output</b>"), self.output_directory, self.save_button]),
            self.run_output,
        ])

    def display(self):
        """Display the controls and render the initial preview."""
        from IPython.display import display

        display(self.layout)
        self.render_current_config()


def create_configurator(config: AnalysisConfig, *, fit_enabled: bool = False) -> WidgetConfigurator:
    """Create a notebook configurator for an analysis configuration."""
    return WidgetConfigurator(config, fit_enabled=fit_enabled)
