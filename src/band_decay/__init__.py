"""Standalone object-oriented regional decay analysis library."""

from .config import (
    AnalysisConfig,
    CurvePlotConfig,
    InputConfig,
    MorisitaHornConfig,
    OutputConfig,
    PlotConfig,
    SamplingConfig,
    SensitivityConfig,
    TopNConfig,
    YearSelectionConfig,
)
from .constants import GLOBAL_ENTITY, OTHER_TAXON, TRANSIENT_TAXON
from .curves import DEFAULT_MEAN_CURVES, CurveSet, FittedCurvesRenderer, render_fitted_curves
from .configurator import WidgetConfigurator, create_configurator
from .data import (
    build_global_analysis_frame,
    build_raw_entity_count_matrices,
    load_counts,
    raw_entity_counts,
    resolve_configured_countries,
    validate_counts_frame,
)
from .defaults import default_config
from .domain import (
    CoveragePlan,
    CurveParameters,
    DecayFit,
    EntityDecayData,
    EntityGrouping,
    EntityYearSelection,
    PreparedData,
    PreparedEntityData,
    RunResult,
    SensitivityResult,
)
from .fitting import DecayFitter, NoOpFitter, PyMCDecayFitter, fit_entities
from .palette import DEFAULT_PALETTE_SETTINGS, PaletteBuilder, PaletteSettings, build_taxon_palette, palette_diagnostics
from .pipeline import DecayAnalysis, prepare_data, run_top_n_sweep
from .policies import (
    AllAvailableYears,
    AvailableYearRanking,
    CollapseTaxa,
    DecayDisplayPolicy,
    DisplayAxisPolicy,
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
    TaxonGroupingPolicy,
    TaxonRanker,
    TaxonRankingPolicy,
    TopNSelectionPolicy,
    TransientPolicy,
    UnionAvailableYears,
    YearSelectionPolicy,
)
from .priors import (
    AsymptotePrior,
    BetaPrior,
    BuiltNoise,
    DecayPriorConfig,
    DirectAsymptote,
    FixedNoiseVariable,
    FixedObservationNoise,
    FixedValuePrior,
    HalfNormalPrior,
    NoisePrior,
    ParameterPrior,
    RelativeAsymptote,
    SampledNoiseVariable,
    SampledObservationNoise,
)
from .preparation import (
    DataPreparer,
    OTHER_NAME,
    TRANSIENT_NAME,
    build_display_and_analysis_matrices,
    build_entity_groupings,
    build_master_display_columns,
    build_mh_counts,
    counts_to_props,
    determine_transient_serotypes,
    eligible_serotypes_for_top_n,
    group_display_counts,
    largest_contiguous_block,
    rank_serotypes,
    select_years_for_entity,
)
from .rendering import AnalysisFigureRenderer, active_legend_labels, masked_display_years, output_path_for_config, render_figure
from .sensitivity import (
    SensitivityRunner,
    build_coverage_plan,
    build_stability_table,
    largest_supported_lag,
    run_sensitivity,
    stability_metrics,
    top_n_for_coverage,
)

__all__ = [
    "AllAvailableYears", "AnalysisConfig", "AnalysisFigureRenderer", "AsymptotePrior", "AvailableYearRanking",
    "BetaPrior", "BuiltNoise", "CollapseTaxa", "CoveragePlan", "CurveParameters", "CurvePlotConfig", "CurveSet",
    "DataPreparer", "DecayAnalysis", "DecayDisplayPolicy", "DecayFitter", "DecayFit", "DecayPriorConfig",
    "DirectAsymptote", "DisplayAxisPolicy", "EntityAvailableYears", "EntityDecayData", "EntityGrouping",
    "EntityYearSelection", "FittedCurvesRenderer", "FixedNoiseVariable", "FixedObservationNoise", "FixedValuePrior",
    "GLOBAL_ENTITY", "GlobalTopN", "GlobalTransientTaxa", "HalfNormalPrior", "InputConfig", "KeepTaxa",
    "LargestContiguousBlock", "MorisitaHornConfig", "NoOpFitter", "NoTransientTaxa", "NormalizedDecay", "NoisePrior",
    "OriginalDecay", "OTHER_NAME", "OTHER_TAXON", "OutputConfig", "PaletteBuilder", "PaletteSettings", "ParameterPrior",
    "PerEntityTopN", "PerEntityTransientTaxa", "PlotConfig", "PreparedData", "PreparedEntityData", "PyMCDecayFitter",
    "QualifyingYears", "RelativeAsymptote", "RunResult", "SampledNoiseVariable", "SampledObservationNoise",
    "SamplingConfig", "SelectedYearRanking", "SensitivityConfig", "SensitivityResult", "SensitivityRunner", "SkipTaxa",
    "TaxonGroupingPolicy", "TaxonRanker", "TaxonRankingPolicy", "TopNConfig", "TopNSelectionPolicy", "TRANSIENT_NAME",
    "TRANSIENT_TAXON", "TransientPolicy", "UnionAvailableYears", "WidgetConfigurator", "YearSelectionConfig",
    "YearSelectionPolicy", "build_coverage_plan", "build_display_and_analysis_matrices", "build_global_analysis_frame",
    "build_master_display_columns", "build_mh_counts", "build_raw_entity_count_matrices", "build_stability_table",
    "build_taxon_palette", "counts_to_props", "create_configurator", "default_config", "determine_transient_serotypes",
    "eligible_serotypes_for_top_n", "fit_entities", "group_display_counts", "largest_contiguous_block", "largest_supported_lag",
    "load_counts", "masked_display_years", "output_path_for_config", "palette_diagnostics", "prepare_data", "rank_serotypes",
    "raw_entity_counts", "render_figure", "render_fitted_curves", "resolve_configured_countries", "run_sensitivity",
    "run_top_n_sweep", "select_years_for_entity", "stability_metrics", "top_n_for_coverage", "validate_counts_frame",
    "DEFAULT_MEAN_CURVES", "DEFAULT_PALETTE_SETTINGS",
]
