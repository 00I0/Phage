#!/usr/bin/env python3
"""List serotypes ordered by their total abundance in the raw count data."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Optional

PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_DIRECTORY / "data/serotype_counts_country_ds_geodate2-2.tsv"
SELECTED_COUNTRIES = (
    "Greece",
    "Italy",
    "Spain",
    "Russia",
    "United Kingdom",
    "France",
    "Germany",
    "Switzerland",
)
LIMIT:Optional[int] = None
REQUIRED_COLUMNS = {"country", "collection_year", "serotype", "count"}


def read_serotype_totals(input_path: Path, selected_countries: tuple[str, ...]) -> dict[str, int]:
    """Sum counts across selected countries and all collection years."""
    selected_country_set = set(selected_countries)
    if len(selected_country_set) != len(selected_countries):
        raise ValueError("SELECTED_COUNTRIES must contain unique country names.")

    with input_path.open(newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file, delimiter="\t")
        fieldnames = set(reader.fieldnames or ())
        missing_columns = REQUIRED_COLUMNS.difference(fieldnames)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Input file is missing required columns: {missing}")

        totals: defaultdict[str, int] = defaultdict(int)
        observed_countries: set[str] = set()
        for row_number, row in enumerate(reader, start=2):
            country = (row["country"] or "").strip()
            observed_countries.add(country)
            if country not in selected_country_set:
                continue
            serotype = (row["serotype"] or "").strip()
            if not serotype:
                raise ValueError(f"Row {row_number} has an empty serotype.")
            try:
                count = int(row["count"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Row {row_number} has an invalid count: {row['count']!r}") from error
            if count < 0:
                raise ValueError(f"Row {row_number} has a negative count: {count}")
            totals[serotype] += count
    missing_countries = selected_country_set.difference(observed_countries)
    if missing_countries:
        missing = ", ".join(sorted(missing_countries))
        raise ValueError(f"Selected countries were not found in the input file: {missing}")
    return dict(totals)


def format_report(totals: dict[str, int], input_path: Path, selected_countries: tuple[str, ...], limit: int | None) -> str:
    """Format the abundance totals as an aligned plain-text report."""
    ranked_totals = sorted(totals.items(), key=lambda item: (-item[1], item[0]))
    displayed_totals = ranked_totals if limit is None else ranked_totals[:limit]
    try:
        source_display = input_path.relative_to(PROJECT_DIRECTORY)
    except ValueError:
        source_display = input_path
    rank_width = max(4, len(str(len(displayed_totals))))
    serotype_width = max(len("Serotype"), *(len(serotype) for serotype, _ in displayed_totals)) if displayed_totals else len("Serotype")
    total_width = len("Total abundance")
    separator = f"{'-' * rank_width}  {'-' * serotype_width}  {'-' * total_width}"

    lines = [
        "Serotype abundance totals",
        f"Source: {source_display}",
        f"Countries: {', '.join(selected_countries)}",
        f"Serotypes: {len(totals):,}",
        "",
        f"{'Rank':>{rank_width}}  {'Serotype':<{serotype_width}}  {'Total abundance':>{total_width}}",
        separator,
    ]
    lines.extend(
        f"{rank:>{rank_width},}  {serotype:<{serotype_width}}  {total:>{total_width},}"
        for rank, (serotype, total) in enumerate(displayed_totals, start=1)
    )
    return "\n".join(lines)


def main() -> None:
    """Read the raw data and print the ranked abundance report."""
    if not DEFAULT_INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {DEFAULT_INPUT_PATH}")
    if LIMIT is not None and LIMIT < 1:
        raise ValueError("LIMIT must be at least 1 when set.")
    totals = read_serotype_totals(DEFAULT_INPUT_PATH, SELECTED_COUNTRIES)
    print(format_report(totals, DEFAULT_INPUT_PATH, SELECTED_COUNTRIES, LIMIT))


if __name__ == "__main__":
    main()
