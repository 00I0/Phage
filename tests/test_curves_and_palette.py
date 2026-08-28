from __future__ import annotations

import pandas as pd

from band_decay import (
    CurvePlotConfig,
    PaletteBuilder,
    PaletteSettings,
    PlotConfig,
    render_fitted_curves,
)


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


def test_fitted_curves_render(tmp_path) -> None:
    output = render_fitted_curves(
        {"GLOBAL": (0.9, 0.1, 0.1), "A": (0.8, 0.2, 0.2)},
        CurvePlotConfig(output_path=tmp_path / "curves.png", point_count=32, dpi=60),
    )
    assert output.exists()
