from __future__ import annotations

import pytest
import pandas as pd

from band_decay import PaletteBuilder, PaletteSettings, build_taxon_palette, palette_diagnostics, reconcile_palette
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


def test_reconcile_palette_preserves_reference_colors_with_color_switching() -> None:
    # Serotype A had blue, B had red. In reference, A is red, B is blue.
    candidate = {"A": "#0000ff", "B": "#ff0000", "NewTaxon": "#0000ff"}
    reference = {"A": "#ff0000", "B": "#0000ff"}

    reconciled = reconcile_palette(candidate, reference)

    # Both A and B must strictly match reference
    assert reconciled["A"] == "#ff0000"
    assert reconciled["B"] == "#0000ff"
    # NewTaxon must have a unique color not colliding with A or B
    assert reconciled["NewTaxon"] not in ("#ff0000", "#0000ff")
    # All serotypes have distinct colors
    assert len(set(reconciled.values())) == len(reconciled)


def test_reconcile_palette_integration_with_builder() -> None:
    settings = PaletteSettings(candidate_count=32)
    reference = {"A": "#112233", "B": "#445566"}

    palette = PaletteBuilder(settings).build(
        ("A", "B", "C", "Other"),
        reference_palette=reference,
    )

    assert palette["A"] == "#112233"
    assert palette["B"] == "#445566"
    assert palette["Other"] == settings.other_color
    assert len(set(palette.values())) == 4


def test_palette_2_matches_palette_reference_and_has_unique_colors() -> None:
    from pathlib import Path
    import re

    palette_ref_path = Path("data/palette.R")
    palette_2_path = Path("data/palette_2.R")
    if not (palette_ref_path.exists() and palette_2_path.exists()):
        pytest.skip("Palette R files not present.")

    def parse_serotypes(p: Path) -> dict[str, str]:
        m = re.search(r"serotype_colors = c\((.*?)\)\n\)", p.read_text(encoding="utf-8"), re.DOTALL)
        assert m is not None
        res = {}
        for line in m.group(1).splitlines():
            line = line.strip().rstrip(",")
            if not line:
                continue
            sm = re.match(r"`(.*?)` = \"(#[0-9a-fA-F]+)\"", line)
            if sm:
                res[sm.group(1)] = sm.group(2)
        return res

    ref_colors = parse_serotypes(palette_ref_path)
    p2_colors = parse_serotypes(palette_2_path)

    # 1. All serotypes in palette.R must be present in palette_2.R with the EXACT same color
    for serotype, col in ref_colors.items():
        assert serotype in p2_colors, f"Missing {serotype} in palette_2.R"
        assert p2_colors[serotype] == col, f"Color mismatch for {serotype}: {p2_colors[serotype]} != {col}"

    # 2. palette_2.R must have the additional serotypes
    assert len(p2_colors) > len(ref_colors)

    # 3. Every single color in palette_2.R must be unique
    assert len(set(p2_colors.values())) == len(p2_colors)
