from __future__ import annotations

from pathlib import Path

from .config import AnalysisConfig, InputConfig, MorisitaHornConfig, OutputConfig, PlotConfig, SamplingConfig, TopNConfig, YearSelectionConfig
from .policies import GlobalTopN, GlobalTransientTaxa, QualifyingYears, SkipTaxa, UnionAvailableYears, OriginalDecay


def default_config(data_path: Path | str, countries: tuple[str, ...], *, output_directory: Path | str = "plots") -> AnalysisConfig:
    """Create the standard configuration used by the standalone workflow.

    Args:
        data_path: Path to the tab-separated count dataset.
        countries: Country labels to include.
        output_directory: Directory for rendered figures.

    Returns:
        A ready-to-run analysis configuration.
    """
    return AnalysisConfig(
        input=InputConfig(data_path=Path(data_path), countries=tuple(countries)),
        year_selection=YearSelectionConfig(
            min_count_per_year=10,
            selection=QualifyingYears(),
            display_axis=UnionAvailableYears(),
        ),
        top_n=TopNConfig(selection=GlobalTopN(), transient=GlobalTransientTaxa()),
        mh=MorisitaHornConfig(other_grouping=SkipTaxa(), transient_grouping=SkipTaxa()),
        sampling=SamplingConfig(),
        plot=PlotConfig(decay_display=OriginalDecay()),
        output=OutputConfig(output_directory=Path(output_directory)),
    )
