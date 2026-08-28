from __future__ import annotations

import pytest
import pandas as pd

from band_decay import PaletteBuilder, PaletteSettings, build_taxon_palette, palette_diagnostics
from band_decay.palette import diagnose_palette


def _stack() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "A": [100.0, 0.0, 20.0],
            "B": [50.0, 40.0, 0.0],
            "C": [20.0, 30.0, 10.0],
            "Other": [5.0, 2.0, 1.0],
            "Transient": [2.0, 0.0, 1.0],
        }
    )


def test_special_only_palette_returns_configured_colors_in_input_order() -> None:
    settings = PaletteSettings(other_color="#010203", transient_color="#a0b0c0")

    palette = PaletteBuilder(settings).build(("Transient", "Other"))

    assert list(palette) == ["Transient", "Other"]
    assert palette == {"Transient": "#a0b0c0", "Other": "#010203"}


def test_mapping_and_generator_stacks_match_list_input() -> None:
    labels = tuple(_stack().columns)
    settings = PaletteSettings(candidate_count=32)
    expected = build_taxon_palette(labels, displayed_stacks=[_stack()], settings=settings)

    mapping_result = build_taxon_palette(
        labels,
        displayed_stacks={"one": _stack()},
        settings=settings,
    )
    generator_result = build_taxon_palette(
        labels,
        displayed_stacks=(stack for stack in [_stack()]),
        settings=settings,
    )

    assert mapping_result == expected
    assert generator_result == expected


def test_custom_palette_builder_matches_function_and_is_deterministic() -> None:
    labels = ("A", "B", "C")
    settings = PaletteSettings(
        candidate_count=32,
        cvd_conditions=("normal", "protanopia", "deuteranopia"),
    )

    first = PaletteBuilder(settings).build(labels, displayed_stacks=[_stack()])
    second = build_taxon_palette(labels[::-1], displayed_stacks=[_stack()[list(labels[::-1])]], settings=settings)

    assert first == {label: second[label] for label in labels}


def test_master_labels_anchor_subset_colors() -> None:
    stack = _stack()
    settings = PaletteSettings(candidate_count=32)
    full = build_taxon_palette(
        tuple(stack.columns),
        displayed_stacks=[stack],
        master_labels=tuple(stack.columns),
        master_displayed_stacks=[stack],
        settings=settings,
    )

    subset = build_taxon_palette(
        ("A", "B"),
        master_labels=tuple(stack.columns),
        master_displayed_stacks=[stack],
        settings=settings,
    )

    assert subset == {"A": full["A"], "B": full["B"]}


def test_diagnostics_preserve_current_aliases_and_alias_function() -> None:
    settings = PaletteSettings(candidate_count=32)
    palette = PaletteBuilder(settings).build(("A", "B", "C"), displayed_stacks=[_stack()])

    diagnostics = palette_diagnostics(palette, displayed_stacks=[_stack()], settings=settings)
    alias_diagnostics = diagnose_palette(palette, displayed_stacks=[_stack()], settings=settings)

    assert diagnostics == alias_diagnostics
    assert diagnostics["top_10_min_distance"] == diagnostics["top_10_minimum_distance"]
    assert diagnostics["top_20_min_distance"] == diagnostics["top_20_minimum_distance"]
    assert diagnostics["global_min_distance"] == diagnostics["global_minimum_distance"]
    assert diagnostics["lightness_range"] == diagnostics["used_lightness_range"]
    assert diagnostics["chroma_range"] == diagnostics["used_chroma_range"]
    assert len(diagnostics["closest_pairs"]) == 3


def test_invalid_labels_and_stacks_raise_current_errors() -> None:
    with pytest.raises(ValueError, match="labels must be unique"):
        PaletteBuilder(PaletteSettings(candidate_count=32)).build(("A", "A"))

    with pytest.raises(TypeError, match="pandas DataFrames"):
        PaletteBuilder(PaletteSettings(candidate_count=32)).build(("A",), displayed_stacks=[object()])

    with pytest.raises(ValueError, match="candidate_count"):
        PaletteSettings(candidate_count=1)


def test_zero_hero_count_can_build_a_palette() -> None:
    settings = PaletteSettings(candidate_count=32, hero_taxon_count=0)

    palette = PaletteBuilder(settings).build(("A", "B", "C"))

    assert set(palette) == {"A", "B", "C"}
    assert len(set(palette.values())) == 3
