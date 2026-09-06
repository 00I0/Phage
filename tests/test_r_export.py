from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from band_decay import (
    DecayFit,
    EntityDecayData,
    EntityYearSelection,
    NormalizedDecay,
    PlotConfig,
    PosteriorMedian, DecayAnalysis, SensitivityRunner,
)
from scripts.update_fitted_country_curves_data import build_palette_r, build_r_data
import scripts.update_fitted_country_curves_data as export_module


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
    assert "`GLOBAL` = \"#000000\"" in output
    assert "`GLOBAL_ALL` = \"#000000\"" in output
    assert "`Greece` = \"#1f77b4\"" in output
    assert "serotype_colors = c(" in output
    assert "`S1` = \"#112233\"" in output


def test_main_always_exports_both_aggregate_curves_and_colors(tmp_path, monkeypatch) -> None:
    run = _band_run()
    analysis = SimpleNamespace(
        prepare=lambda: run.prepared,
        fit=lambda prepared: run.decay_data,
    )
    sensitivity_runner = SimpleNamespace(run=lambda fit: SimpleNamespace(runs=[run]))
    data_file = tmp_path / "band_decay_data.R"
    palette_file = tmp_path / "palette.R"
    monkeypatch.setattr(export_module, "DecayAnalysis", lambda config: analysis)
    monkeypatch.setattr(export_module, "SensitivityRunner", lambda *args: sensitivity_runner)
    all_countries_decay = EntityDecayData(
        np.asarray([1.0, 7.0]), np.asarray([0.8, 0.6]), run.decay_data["GLOBAL"].fit
    )
    monkeypatch.setattr(export_module, "fit_all_countries_aggregate", lambda config: all_countries_decay)
    monkeypatch.setattr(export_module, "OUTPUT_RELATIVE_PATH", data_file)
    monkeypatch.setattr(export_module, "PALETTE_OUTPUT_RELATIVE_PATH", palette_file)

    export_module.main()

    curve_payload = data_file.read_text(encoding="utf-8").split("  band_decay = list(")[0]
    assert "    `GLOBAL` = list(" in curve_payload
    assert "    `GLOBAL_ALL` = list(" in curve_payload
    assert "    `A` = list(" in curve_payload
    assert curve_payload.count("      extrapolation_start = ") == 3
    all_countries_payload = curve_payload.split("    `GLOBAL_ALL` = list(")[1].split("    `A` = list(")[0]
    assert "extrapolation_start = 7.0000" in all_countries_payload
    assert "`GLOBAL` = \"#000000\"" in palette_file.read_text(encoding="utf-8")
    assert "`GLOBAL_ALL` = \"#000000\"" in palette_file.read_text(encoding="utf-8")


@pytest.mark.parametrize("config_builder", [export_module.build_config, export_module.build_band_config])
def test_export_configs_default_minimum_year_count_to_two(tmp_path, config_builder) -> None:
    config = config_builder(tmp_path)

    assert config.top_n.min_year_count == 2.0
    assert config.top_n.min_year_count_for_country("Austria") == 2.0
    assert config.top_n.min_year_count_for_country("GLOBAL") == 2.0
    assert config.top_n.min_year_count_for_country("Greece") == 3.0


def test_all_countries_aggregate_pools_extra_countries_and_fits_only_global(tmp_path, monkeypatch) -> None:
    counts = pd.DataFrame(
        [
            {"country": country, "collection_year": year, "serotype": "common", "count": 10}
            for country in ("Greece", "Austria")
            for year in (2000, 2001, 2002)
        ]
        + [
            {"country": "Austria", "collection_year": year, "serotype": "extra", "count": count}
            for year, count in ((2000, 4), (2001, 8), (2002, 12))
        ]
        + [
            {"country": "Austria", "collection_year": 2000, "serotype": "singleton", "count": 1},
            {"country": "Austria", "collection_year": 1999, "serotype": "excluded", "count": 3},
        ]
    )
    input_file = tmp_path / "counts.tsv"
    counts.to_csv(input_file, sep="\t", index=False)
    base_config = export_module.build_config(tmp_path)
    config = replace(
        base_config,
        input=replace(base_config.input, data_path=input_file, countries=("Greece",)),
    )
    selected_prepared = DecayAnalysis(config).prepare()
    assert selected_prepared.entities["GLOBAL"].analysis_total == 30.0
    expected_decay = _band_run().decay_data["GLOBAL"]
    fit_calls = []

    def capture_fit(analysis, prepared):
        fit_calls.append(prepared)
        assert analysis.config.input.countries == ("Austria", "Greece")
        assert prepared.entity_order == ("GLOBAL",)
        assert tuple(prepared.entities) == ("GLOBAL",)
        aggregate = prepared.entities["GLOBAL"]
        assert aggregate.analysis_total == 85.0
        assert aggregate.mh_counts.index.tolist() == [2000, 2001, 2002]
        assert aggregate.mh_counts["common"].tolist() == [20.0, 20.0, 20.0]
        assert aggregate.mh_counts["extra"].tolist() == [4.0, 8.0, 12.0]
        assert "singleton" not in aggregate.mh_counts
        assert "excluded" not in aggregate.mh_counts
        return {"GLOBAL": expected_decay}

    monkeypatch.setattr(DecayAnalysis, "fit", capture_fit)

    assert export_module.fit_all_countries_aggregate(config) is expected_decay
    assert len(fit_calls) == 1
    assert config.input.countries == ("Greece",)


def test_r_aggregate_modes_use_exported_curves(tmp_path) -> None:
    rscript = shutil.which("Rscript")
    if rscript is None:
        pytest.skip("Rscript is not installed.")
    dependency_check = subprocess.run(
        [rscript, "--vanilla", "-e", 'quit(status = if (requireNamespace("ggplot2", quietly = TRUE)) 0 else 1)'],
        capture_output=True,
        text=True,
        check=False,
    )
    if dependency_check.returncode != 0:
        pytest.skip("R package ggplot2 is not installed.")

    decay = _band_run().decay_data["GLOBAL"]
    all_countries_decay = EntityDecayData(
        np.asarray([1.0, 7.0]),
        np.asarray([0.7, 0.4]),
        DecayFit(y0_samples=[0.9], b_samples=[0.4], c_samples=[0.1]),
    )
    data_directory = tmp_path / "data"
    data_directory.mkdir()
    (data_directory / "band_decay_data.R").write_text(
        build_r_data(
            {"GLOBAL": decay, "GLOBAL_ALL": all_countries_decay, "Greece": decay},
            ("GLOBAL", "GLOBAL_ALL", "Greece"),
            horizon_years=20.0,
            point_count=8,
            confidence_level=0.95,
        ),
        encoding="utf-8",
    )
    (data_directory / "palette.R").write_text(build_palette_r({}), encoding="utf-8")
    project_directory = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            rscript, "--vanilla",
            str(project_directory / "tests/test_r_curve_modes.R"),
            str(project_directory / "scripts/plot_fitted_country_curves.R"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
