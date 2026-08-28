from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping

import pandas as pd

from .constants import GLOBAL_ENTITY

LOGGER = logging.getLogger(__name__)
REQUIRED_COLUMNS = frozenset({"country", "collection_year", "serotype", "count"})


def validate_counts_frame(raw: pd.DataFrame, *, source: str = "input dataframe") -> pd.DataFrame:
    """Validate and canonicalize a long-form count dataframe.

    Args:
        raw: Dataframe with country, year, serotype, and count columns.
        source: Label used in validation errors.

    Returns:
        Validated, aggregated, and sorted count records.

    Raises:
        TypeError: If ``raw`` is not a dataframe.
        ValueError: If required columns or values are invalid.
    """
    if not isinstance(raw, pd.DataFrame):
        raise TypeError(f"{source} must be a pandas DataFrame.")
    missing = REQUIRED_COLUMNS.difference(raw.columns)
    if missing:
        raise ValueError(f"{source} is missing required columns: {sorted(missing)}")

    frame = raw.loc[:, ["country", "collection_year", "serotype", "count"]].copy()
    for column in ("country", "collection_year", "serotype"):
        if frame[column].isna().any():
            raise ValueError(f"{source} contains null {column} values.")

    years = pd.to_numeric(frame["collection_year"], errors="coerce")
    invalid_years = years.isna() | (years % 1 != 0)
    if invalid_years.any():
        values = frame.loc[invalid_years, "collection_year"].astype(str).unique().tolist()
        raise ValueError(f"{source} contains invalid collection_year values: {values[:5]}")

    counts = pd.to_numeric(frame["count"], errors="coerce")
    if counts.isna().any():
        values = frame.loc[counts.isna(), "count"].astype(str).unique().tolist()
        raise ValueError(f"{source} contains invalid count values: {values[:5]}")
    if (counts < 0).any():
        raise ValueError(f"{source} contains negative count values.")

    frame["country"] = frame["country"].astype(str)
    frame["collection_year"] = years.astype(int)
    frame["serotype"] = frame["serotype"].astype(str)
    frame["count"] = counts.astype(float)
    if (frame["country"].str.len() == 0).any() or (frame["serotype"].str.len() == 0).any():
        raise ValueError(f"{source} contains empty country or serotype identifiers.")
    return (
        frame.groupby(["country", "collection_year", "serotype"], as_index=False, sort=True)["count"]
        .sum()
        .sort_values(["country", "collection_year", "serotype"], kind="mergesort")
        .reset_index(drop=True)
    )


def load_counts(path: Path | str) -> pd.DataFrame:
    """Load and validate a tab-separated count dataset.

    Args:
        path: Path to the tab-separated input file.

    Returns:
        Validated long-form count records.

    Raises:
        FileNotFoundError: If the dataset does not exist.
        ValueError: If the dataset fails validation.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return validate_counts_frame(pd.read_csv(path, sep="\t"), source=str(path))


def resolve_configured_countries(frame: pd.DataFrame, configured_countries: tuple[str, ...]) -> tuple[str, ...]:
    """Keep configured countries that are present in the input.

    Args:
        frame: Validated long-form count records.
        configured_countries: Requested country labels.

    Returns:
        Present countries in configured order.
    """
    present = set(frame["country"].astype(str).unique())
    selected = tuple(country for country in configured_countries if country in present)
    missing = tuple(country for country in configured_countries if country not in present)
    if missing:
        LOGGER.warning("Skipping configured countries not found in input: %s", list(missing))
    if not selected:
        raise ValueError("No configured countries are present in the input data.")
    return selected


def raw_entity_counts(frame: pd.DataFrame, entity: str) -> pd.DataFrame:
    """Pivot long-form records into yearly serotype counts for an entity."""
    subset = frame[frame["country"] == entity]
    if subset.empty:
        return pd.DataFrame(dtype=float)
    return (
        subset.pivot_table(
            index="collection_year",
            columns="serotype",
            values="count",
            aggfunc="sum",
            fill_value=0.0,
        )
        .sort_index()
        .sort_index(axis=1)
        .astype(float)
    )


def build_raw_entity_count_matrices(frame: pd.DataFrame, countries: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    """Build count matrices for the aggregate and requested countries."""
    country_frame = frame[frame["country"].isin(countries)].copy()
    global_frame = country_frame.copy()
    global_frame["country"] = GLOBAL_ENTITY
    matrices = {GLOBAL_ENTITY: raw_entity_counts(global_frame, GLOBAL_ENTITY)}
    matrices.update({country: raw_entity_counts(country_frame, country) for country in countries})
    return matrices


def build_global_analysis_frame(
    country_frame: pd.DataFrame,
    country_year_selection: Mapping[str, "EntityYearSelection"],
) -> pd.DataFrame:
    """Combine each country’s selected years into the global analysis frame."""
    parts = []
    for country, selection in country_year_selection.items():
        if selection.selected_years:
            parts.append(
                country_frame[
                    (country_frame["country"] == country)
                    & country_frame["collection_year"].isin(selection.selected_years)
                ]
            )
    if not parts:
        return country_frame.iloc[0:0].copy()
    result = pd.concat(parts, ignore_index=True)
    result["country"] = GLOBAL_ENTITY
    return result


from .domain import EntityYearSelection
