from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from band_decay import (
    DecayFit,
    EntityDecayData,
    EntityYearSelection,
    NormalizedDecay,
    PlotConfig,
    PosteriorMedian,
)
from scripts.update_fitted_country_curves_data import build_palette_r, build_r_data


def _band_run() -> SimpleNamespace:
    fit = DecayFit(
        y0_samples=np.asarray([0.9, 0.92, 0.91]),
        b_samples=np.asarray([0.1, 0.11, 0.105]),
        c_samples=np.asarray([0.2, 0.21, 0.19]),
        divergences=2,
        sigma_samples=np.asarray([0.05, 0.06, 0.055]),
    )
    counts = pd.DataFrame(
        {"S1": [10.0, 8.0], "Other": [2.0, 4.0]},
        index=[2000, 2001],
    )
    entity = SimpleNamespace(
        analysis_total=24.0,
        grouped_display_counts=counts,
        grouping=SimpleNamespace(
            selected_serotypes=("S1",),
            transient_serotypes=(),
            include_other=True,
            include_transient=False,
        ),
        year_selection=EntityYearSelection(
            entity="A",
            available_years=(2000, 2001),
            qualifying_years=frozenset({2000}),
            selected_years=frozenset({2000}),
            excluded_years=frozenset({2001}),
            missing_years=frozenset(),
        ),
    )
    prepared = SimpleNamespace(
        entity_order=("GLOBAL", "A"),
        master_display_columns=("S1", "Other"),
        entities={"GLOBAL": entity, "A": entity},
        palette={"S1": "#112233", "Other": "#445566"},
    )
    decay_data = {
        "GLOBAL": EntityDecayData(np.asarray([1.0, 2.0]), np.asarray([0.8, 0.6]), fit),
        "A": EntityDecayData(np.asarray([1.0, 2.0]), np.asarray([0.7, 0.5]), fit),
    }
    return SimpleNamespace(
        config=SimpleNamespace(
            plot=PlotConfig(decay_display=NormalizedDecay(), fit_summary=PosteriorMedian())
        ),
        prepared=prepared,
        decay_data=decay_data,
    )


def test_build_r_data_includes_combined_band_decay_payload() -> None:
    fit = _band_run().decay_data["A"]
    output = build_r_data(
        {"A": fit},
        ("A",),
        horizon_years=20.0,
        point_count=4,
        confidence_level=0.95,
        band_run=_band_run(),
        band_coverage_percent=90.0,
    )

    assert "band_decay = list(" in output
    assert "grouped_counts = list(" in output
    assert "grouped_proportions = list(" in output
    assert "grouping = list(" in output
    assert "similarity = c(" in output
    assert "fit = list(" in output
    assert "sigma =" in output
    assert "divergences = 2" in output


def test_build_palette_r_exports_country_and_serotype_colors() -> None:
    output = build_palette_r({"S1": "#112233"})

    assert "country_colors = c(" in output
    assert "`Greece` = \"#1f77b4\"" in output
    assert "serotype_colors = c(" in output
    assert "`S1` = \"#112233\"" in output
