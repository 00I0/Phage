from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from band_decay import (
    CentralConfidenceInterval,
    CurvePlotConfig,
    DashedExtrapolation,
    OriginalDecay,
    PaletteBuilder,
    PaletteSettings,
    PlotConfig,
    PosteriorMean,
    PosteriorMedian,
    render_fitted_curves,
)
from band_decay.domain import DecayFit, EntityDecayData
import band_decay.curves as curve_module


def test_palette_is_deterministic_and_master_labels_anchor_subset() -> None:
    labels = ("A", "B", "C", "Other", "Transient")
    stack = pd.DataFrame({"A": [100.0], "B": [50.0], "C": [20.0], "Other": [5.0], "Transient": [2.0]})
    settings = PaletteSettings(candidate_count=32)
    first = PaletteBuilder(settings).build(labels, displayed_stacks=[stack])
    second = PaletteBuilder(settings).build(labels[::-1], displayed_stacks=[stack.loc[:, labels[::-1]]])
    assert first == {label: second[label] for label in labels}
    subset = PaletteBuilder(settings).build(("A", "B"), master_labels=labels, master_displayed_stacks=[stack])
    assert subset["A"] == first["A"]
    assert subset["B"] == first["B"]
    assert first["Other"] == settings.other_color


def test_default_palette_matches_legacy_reference_colors() -> None:
    stack = pd.DataFrame(
        {
            "Taxon A": [12, 4, 0, 8],
            "Taxon B": [2, 7, 3, 0],
            "Taxon C": [0, 3, 9, 1],
            "Taxon D": [1, 0, 2, 5],
            "Other": [3, 2, 1, 2],
        }
    )
    palette = PaletteBuilder().build(
        tuple(stack.columns),
        displayed_stacks=[stack],
        master_labels=tuple(stack.columns),
        master_displayed_stacks=[stack],
    )
    assert palette == {
        "Taxon A": "#cc25d6",
        "Taxon B": "#834706",
        "Taxon C": "#21b418",
        "Taxon D": "#1da1f5",
        "Other": "#d9d9d9",
    }


def test_plot_count_label_default_matches_legacy() -> None:
    assert PlotConfig().count_label_max_y_fraction == 0.95


def test_fitted_curves_render_only_requested_entities(tmp_path) -> None:
    fit = DecayFit(
        y0_samples=[0.8, 1.0],
        b_samples=[0.2, 0.2],
        c_samples=[0.1, 0.3],
    )
    output = render_fitted_curves(
        {
            "GLOBAL": EntityDecayData(x=[], y=[], fit=fit),
            "Greece": EntityDecayData(x=[], y=[], fit=fit),
        },
        CurvePlotConfig(output_path=tmp_path / "mean-curves.png", point_count=32, dpi=60),
        entity_order=("Greece",),
    )
    assert output.exists()
    np.testing.assert_allclose(PosteriorMean().curve(OriginalDecay(), fit, np.array([0.0])), np.array([0.9]))
    np.testing.assert_allclose(PosteriorMedian().curve(OriginalDecay(), fit, np.array([0.0])), np.array([0.9]))


def test_posterior_summary_policy_selects_mean_or_median() -> None:
    fit = DecayFit(
        y0_samples=[0.1, 0.9, 0.9],
        b_samples=[0.2, 0.2, 0.2],
        c_samples=[0.1, 0.1, 0.1],
    )
    x_values = np.array([0.0])
    mean = PosteriorMean().curve(OriginalDecay(), fit, x_values)
    median = PosteriorMedian().curve(OriginalDecay(), fit, x_values)
    np.testing.assert_allclose(mean, np.array([19 / 30]))
    np.testing.assert_allclose(median, np.array([0.9]))


def test_confidence_interval_policy_returns_requested_central_band() -> None:
    fit = DecayFit(
        y0_samples=[0.8, 0.9, 1.0],
        b_samples=[0.2, 0.2, 0.2],
        c_samples=[0.1, 0.2, 0.3],
    )
    lower, upper = CentralConfidenceInterval().bounds(OriginalDecay(), fit, np.array([0.0]))
    assert 0.8 <= lower[0] < 0.9
    assert 0.9 < upper[0] <= 1.0


def test_dashed_extrapolation_splits_after_largest_fitted_lag() -> None:
    decay = EntityDecayData(x=np.array([1.0, 2.0, 4.0]), y=np.array([0.9, 0.7, 0.4]))
    fitted, extrapolated = DashedExtrapolation().split(decay, np.array([0.0, 2.0, 4.0, 6.0]))
    np.testing.assert_array_equal(fitted, np.array([0.0, 2.0, 4.0]))
    np.testing.assert_array_equal(extrapolated, np.array([6.0]))


def test_mean_fitted_curves_support_confidence_and_extrapolation_options(tmp_path) -> None:
    fit = DecayFit(
        y0_samples=[0.8, 1.0],
        b_samples=[0.2, 0.2],
        c_samples=[0.1, 0.3],
    )
    output = render_fitted_curves(
        {"Greece": EntityDecayData(x=np.array([1.0, 2.0]), y=np.array([0.8, 0.6]), fit=fit)},
        CurvePlotConfig(
            output_path=tmp_path / "configured-curves.png",
            point_count=32,
            dpi=60,
            confidence_interval=CentralConfidenceInterval(),
            extrapolation=DashedExtrapolation(),
        ),
    )
    assert output.exists()


def test_configured_curve_legend_hides_auxiliary_styles_and_uses_integer_ticks(tmp_path, monkeypatch) -> None:
    fit = DecayFit(
        y0_samples=[0.8, 1.0],
        b_samples=[0.2, 0.2],
        c_samples=[0.1, 0.3],
    )
    figures = []
    original_subplots = curve_module.plt.subplots

    def capture_subplots(*args, **kwargs):
        figure, axes = original_subplots(*args, **kwargs)
        figures.append(figure)
        return figure, axes

    monkeypatch.setattr(curve_module.plt, "subplots", capture_subplots)
    render_fitted_curves(
        {"Greece": EntityDecayData(x=np.array([1.0, 2.0]), y=np.array([0.8, 0.6]), fit=fit)},
        CurvePlotConfig(
            output_path=tmp_path / "legend-and-ticks.png",
            point_count=32,
            dpi=60,
            confidence_interval=CentralConfidenceInterval(),
            extrapolation=DashedExtrapolation(),
        ),
    )
    axis = figures[0].axes[0]
    _, labels = axis.get_legend_handles_labels()
    assert labels == ["Greece"]
    assert isinstance(axis.xaxis.get_major_locator(), MaxNLocator)
    assert np.allclose(axis.get_xticks(), np.round(axis.get_xticks()))
