from __future__ import annotations

from dataclasses import replace

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from band_decay import (
    AnalysisConfig,
    AllAvailableYears,
    DecayAnalysis,
    EntityAvailableYears,
    UnionAvailableYears,
    CollapseTaxa,
    PerEntityTopN,
    PerEntityTransientTaxa,
    QualifyingYears,
    InputConfig,
    MorisitaHornConfig,
    OutputConfig,
    PaletteBuilder,
    PaletteSettings,
    PlotConfig,
    TopNConfig,
    YearSelectionConfig,
    largest_contiguous_block,
    prepare_data,
    validate_counts_frame,
)


def rows(country: str, years: list[int], values: dict[str, float]) -> list[dict[str, object]]:
    return [
        {"country": country, "collection_year": year, "serotype": label, "count": count}
        for year in years
        for label, count in values.items()
    ]


def config_for(countries: tuple[str, ...], *, top_n: int = 2, mode=None, axis=None, scope=None) -> AnalysisConfig:
    mode = mode or AllAvailableYears()
    axis = axis or EntityAvailableYears()
    scope = scope or PerEntityTopN()
    return AnalysisConfig(
        input=InputConfig(countries=countries),
        year_selection=YearSelectionConfig(min_count_per_year=0, selection=mode, display_axis=axis),
        top_n=TopNConfig(n=top_n, selection=scope, transient=PerEntityTransientTaxa()),
        mh=MorisitaHornConfig(other_grouping=CollapseTaxa(), transient_grouping=CollapseTaxa()),
        plot=PlotConfig(dpi=60),
        output=OutputConfig(),
    )


def test_validation_aggregates_duplicates_and_rejects_bad_counts() -> None:
    frame = pd.DataFrame([
        {"country": "A", "collection_year": 2000, "serotype": "S1", "count": 2},
        {"country": "A", "collection_year": 2000, "serotype": "S1", "count": 3},
    ])
    result = validate_counts_frame(frame)
    assert result.iloc[0].to_dict() == {"country": "A", "collection_year": 2000, "serotype": "S1", "count": 5.0}
    with pytest.raises(ValueError, match="negative"):
        validate_counts_frame(frame.assign(count=-1))


def test_preparation_preserves_display_years_and_zero_fills_gaps() -> None:
    frame = pd.DataFrame(rows("A", [2000, 2002], {"S1": 10}) + rows("B", [2000, 2001, 2002], {"S1": 10}))
    prepared = prepare_data(config_for(("A", "B"), axis=UnionAvailableYears()), frame)
    entity = prepared.entities["A"]
    assert entity.display_counts.index.tolist() == [2000, 2001, 2002]
    assert entity.display_counts["S1"].tolist() == [10.0, 0.0, 10.0]
    assert entity.year_selection.missing_years == frozenset({2001})


def test_per_entity_top_n_conserves_totals_and_uses_shared_palette() -> None:
    frame = pd.DataFrame(
        rows("A", [2000, 2001], {"S1": 100, "S2": 90, "S3": 10})
        + rows("B", [2000, 2001], {"S1": 10, "S2": 100, "S3": 90})
    )
    config = config_for(("A", "B"))
    prepared = prepare_data(config, frame, palette_builder=PaletteBuilder(PaletteSettings(candidate_count=32)))
    assert prepared.entities["A"].grouping.selected_serotypes == ("S1", "S2")
    assert prepared.entities["B"].grouping.selected_serotypes == ("S2", "S3")
    for entity in prepared.entity_order:
        pd.testing.assert_series_equal(
            prepared.entities[entity].display_counts.sum(axis=1),
            prepared.entities[entity].grouped_display_counts.sum(axis=1),
            check_names=False,
        )
    assert prepared.palette["S2"]


def test_year_block_tie_prefers_earliest() -> None:
    assert largest_contiguous_block([2000, 2001, 2003, 2004]) == frozenset({2000, 2001})


def test_transient_detection_uses_available_timeline_when_analysis_is_filtered() -> None:
    frame = pd.DataFrame(
        rows("A", [2000], {"persistent": 10, "transient": 5})
        + rows("A", [2001], {"persistent": 10})
    )
    config = replace(
        config_for(("A",), top_n=0),
        year_selection=YearSelectionConfig(
            min_count_per_year=10,
            selection=QualifyingYears(),
            display_axis=EntityAvailableYears(),
        ),
    )
    prepared = prepare_data(config, frame)
    assert prepared.entities["A"].grouping.selected_serotypes == ("persistent",)


def test_analysis_run_without_fit_writes_figure(tmp_path) -> None:
    frame = pd.DataFrame(rows("A", [2000, 2001, 2002], {"S1": 10, "S2": 5}))
    config = replace(config_for(("A",), top_n=1), output=OutputConfig(output_directory=tmp_path, filename_template="decay.pdf"))
    result = DecayAnalysis(config).run(frame, fit=False)
    assert result.output_path == tmp_path / "decay.pdf"
    assert result.output_path.exists()
    assert result.decay_data["A"].fit is None
