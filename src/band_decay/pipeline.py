from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Sequence

import pandas as pd

from .config import AnalysisConfig
from .domain import PreparedData, RunResult
from .fitting import DecayFitter, NoOpFitter, PyMCDecayFitter, fit_entities
from .palette import PaletteBuilder
from .preparation import DataPreparer
from .rendering import AnalysisFigureRenderer, output_path_for_config


class DecayAnalysis:
    """Coordinate preparation, optional fitting, and figure rendering."""

    def __init__(self, config: AnalysisConfig, *, fitter: DecayFitter | None = None, renderer=None, palette_builder=None):
        """Create an analysis facade with optional injectable services.

        Args:
            config: Immutable analysis configuration.
            fitter: Optional custom decay fitter.
            renderer: Optional custom figure renderer.
            palette_builder: Optional custom palette builder.
        """
        self.config = config
        self.fitter = fitter
        self.renderer = renderer or AnalysisFigureRenderer()
        self.palette_builder = palette_builder or PaletteBuilder()

    def prepare(self, raw_counts: pd.DataFrame | None = None, *, palette_master_labels: Sequence[str] | None = None) -> PreparedData:
        """Prepare input data without fitting or rendering.

        Args:
            raw_counts: Optional long-form count dataframe.
            palette_master_labels: Optional stable palette label universe.

        Returns:
            Prepared matrices, groupings, and palette.
        """
        return DataPreparer(self.config).prepare(
            raw_counts,
            palette_builder=self.palette_builder,
            palette_master_labels=palette_master_labels,
        )

    def run(
        self,
        raw_counts: pd.DataFrame | None = None,
        *,
        fit: bool = False,
        fitter: DecayFitter | None = None,
        output_path: Path | str | None = None,
        save: bool = True,
        palette_master_labels: Sequence[str] | None = None,
    ) -> RunResult:
        """Run preparation, optional fitting, and figure rendering.

        Args:
            raw_counts: Optional long-form count dataframe.
            fit: Whether to fit decay curves.
            fitter: Optional per-call fitter override.
            output_path: Optional explicit output path.
            save: Whether to derive and save the default output path.
            palette_master_labels: Optional stable palette label universe.

        Returns:
            Complete run result including prepared data and decay data.
        """
        prepared = self.prepare(raw_counts, palette_master_labels=palette_master_labels)
        selected_fitter = fitter if fitter is not None else self.fitter
        if fit and selected_fitter is None:
            selected_fitter = PyMCDecayFitter()
        # Keep fitting optional so preparation and preview runs stay lightweight.
        decay_data = fit_entities(prepared, self.config, selected_fitter if fit else NoOpFitter())
        resolved_path = Path(output_path) if output_path is not None else output_path_for_config(self.config) if save else None
        rendered_path = self.renderer.render(prepared, decay_data, self.config, resolved_path)
        return RunResult(config=self.config, prepared=prepared, decay_data=decay_data, output_path=rendered_path)


def prepare_data(config: AnalysisConfig, raw_counts: pd.DataFrame | None = None, *, palette_master_labels=None, palette_builder=None) -> PreparedData:
    """Prepare data through the functional API.

    Args:
        config: Immutable analysis configuration.
        raw_counts: Optional long-form count dataframe.
        palette_master_labels: Optional stable palette label universe.
        palette_builder: Optional custom palette builder.

    Returns:
        Prepared analysis data.
    """
    return DataPreparer(config).prepare(raw_counts, palette_master_labels=palette_master_labels, palette_builder=palette_builder)


def run_top_n_sweep(
    base_config: AnalysisConfig,
    top_n_values,
    *,
    raw_counts: pd.DataFrame | None = None,
    fit: bool = False,
    fitter: DecayFitter | None = None,
) -> list[RunResult]:
    """Run the same analysis configuration for several Top-N values.

    Args:
        base_config: Configuration to copy for each sweep point.
        top_n_values: Values assigned to ``config.top_n.n``.
        raw_counts: Optional shared long-form count dataframe.
        fit: Whether to fit decay curves at each point.
        fitter: Optional fitter shared by all runs.

    Returns:
        Run results in the order of ``top_n_values``.
    """
    results = []
    for value in top_n_values:
        config = replace(base_config, top_n=replace(base_config.top_n, n=int(value)))
        results.append(DecayAnalysis(config, fitter=fitter).run(raw_counts, fit=fit))
    return results
