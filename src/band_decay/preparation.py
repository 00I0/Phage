from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
import numpy as np
import pandas as pd

from .config import TopNConfig, YearSelectionConfig
from .constants import GLOBAL_ENTITY, OTHER_TAXON, TRANSIENT_TAXON
from .data import (
    build_global_analysis_frame,
    build_raw_entity_count_matrices,
    load_counts,
    raw_entity_counts,
    resolve_configured_countries,
    validate_counts_frame,
)
from .domain import EntityGrouping, EntityYearSelection, PreparedData, PreparedEntityData
from .policies import (
    CollapseTaxa,
    DisplayAxisPolicy,
    TaxonGroupingPolicy,
    TransientPolicy,
)

OTHER_NAME = OTHER_TAXON
TRANSIENT_NAME = TRANSIENT_TAXON


def counts_to_props(counts: pd.DataFrame) -> pd.DataFrame:
    """Convert yearly counts to row-wise proportions."""
    totals = counts.sum(axis=1).replace(0, np.nan)
    return counts.div(totals, axis=0).fillna(0.0)


def largest_contiguous_block(years: Iterable[int]) -> frozenset[int]:
    """Return the longest consecutive-year block in an iterable."""
    from .policies import LargestContiguousBlock

    return LargestContiguousBlock().select(tuple(int(year) for year in years), frozenset(int(year) for year in years))


def select_years_for_entity(counts: pd.DataFrame, config: YearSelectionConfig, entity: str) -> EntityYearSelection:
    """Select an entity’s analytical years according to its year policy.

    Args:
        counts: Year-indexed serotype counts.
        config: Year-selection configuration.
        entity: Entity label used for country-specific overrides.

    Returns:
        The available, qualifying, selected, and excluded years.
    """
    yearly_totals = counts.sum(axis=1).astype(float)
    available = tuple(int(year) for year in yearly_totals.index[yearly_totals > 0])
    threshold = float(config.min_count_for_country(entity))
    qualifying = frozenset(int(year) for year in yearly_totals.index[yearly_totals >= threshold]) if threshold > 0 else frozenset(available)
    selected = config.selection.select(available, qualifying)
    return EntityYearSelection(
        entity=entity,
        available_years=available,
        qualifying_years=qualifying,
        selected_years=selected,
        excluded_years=frozenset(set(available).difference(selected)),
    )


def _continuous_year_span(years: Iterable[int]) -> tuple[int, ...]:
    values = tuple(sorted({int(year) for year in years}))
    return tuple(range(values[0], values[-1] + 1)) if values else tuple()


def _reindex_counts(counts: pd.DataFrame, years: tuple[int, ...], serotypes: tuple[str, ...]) -> pd.DataFrame:
    return counts.reindex(index=list(years), fill_value=0.0).reindex(columns=list(serotypes), fill_value=0.0).astype(float)


def _global_year_selection(display_counts: pd.DataFrame, analysis_counts: pd.DataFrame) -> EntityYearSelection:
    display_totals = display_counts.sum(axis=1).astype(float)
    analysis_totals = analysis_counts.sum(axis=1).astype(float) if not analysis_counts.empty else pd.Series(dtype=float)
    available = tuple(int(year) for year in display_totals.index[display_totals > 0])
    selected = frozenset(int(year) for year in analysis_totals.index[analysis_totals > 0])
    return EntityYearSelection(
        entity=GLOBAL_ENTITY,
        available_years=available,
        qualifying_years=selected,
        selected_years=selected,
        excluded_years=frozenset(set(available).difference(selected)),
    )


def _display_year_index(selection: EntityYearSelection, union_years: tuple[int, ...], policy: DisplayAxisPolicy) -> tuple[int, ...]:
    return policy.years(selection, union_years)


def build_display_and_analysis_matrices(
    *,
    raw_counts_by_entity: Mapping[str, pd.DataFrame],
    country_frame: pd.DataFrame,
    countries: tuple[str, ...],
    country_year_selection: Mapping[str, EntityYearSelection],
    config: YearSelectionConfig,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, EntityYearSelection]]:
    """Build display and filtered analytical matrices for each entity."""
    # Keep broad display timelines separate from filtered analytical years.
    all_serotypes = tuple(sorted({label for counts in raw_counts_by_entity.values() for label in counts.columns.astype(str)}))
    global_analysis = raw_entity_counts(build_global_analysis_frame(country_frame, country_year_selection), GLOBAL_ENTITY)
    selections = {GLOBAL_ENTITY: _global_year_selection(raw_counts_by_entity[GLOBAL_ENTITY], global_analysis), **country_year_selection}
    union_years = _continuous_year_span(year for selection in selections.values() for year in selection.available_years)
    display_counts: dict[str, pd.DataFrame] = {}
    analysis_counts: dict[str, pd.DataFrame] = {}
    updated: dict[str, EntityYearSelection] = {}
    for entity in (GLOBAL_ENTITY, *countries):
        selection = selections[entity]
        display_years = _display_year_index(selection, union_years, config.display_axis)
        missing = frozenset(set(display_years).difference(selection.available_years))
        updated[entity] = replace(selection, missing_years=missing)
        display_counts[entity] = _reindex_counts(raw_counts_by_entity[entity], display_years, all_serotypes)
        if entity == GLOBAL_ENTITY:
            analysis_counts[entity] = _reindex_counts(global_analysis, tuple(sorted(selection.selected_years)), all_serotypes)
        else:
            analysis_counts[entity] = _reindex_counts(
                raw_counts_by_entity[entity],
                tuple(year for year in selection.available_years if year in selection.selected_years),
                all_serotypes,
            )
    return display_counts, analysis_counts, updated


def _transient_serotypes_from_counts(counts: pd.DataFrame) -> tuple[str, ...]:
    if counts.empty:
        return tuple()
    present_years = (counts > 0).sum(axis=0)
    return tuple(sorted(present_years[present_years == 1].index.astype(str)))


def determine_transient_serotypes(
    entity_order: tuple[str, ...],
    analysis_counts: Mapping[str, pd.DataFrame],
    scope: TransientPolicy,
    display_counts: Mapping[str, pd.DataFrame] | None = None,
) -> dict[str, tuple[str, ...]]:
    """Identify serotypes observed in only one year at the requested scope."""
    source = display_counts if display_counts is not None else analysis_counts
    return scope.resolve(entity_order, analysis_counts, source)


def eligible_serotypes_for_top_n(
    counts: pd.DataFrame,
    *,
    transient_serotypes: Iterable[str] = (),
    min_year_count: float = 0.0,
    min_year_percent: float = 0.0,
) -> tuple[str, ...]:
    """Return taxa meeting transient, count, and percentage filters."""
    transient = set(transient_serotypes)
    non_transient = [label for label in counts.columns.astype(str) if label not in transient]
    if not non_transient:
        return tuple()
    if min_year_count <= 0 and min_year_percent <= 0:
        return tuple(sorted(non_transient))
    values = counts.reindex(columns=non_transient, fill_value=0.0).to_numpy(dtype=float)
    totals = counts.sum(axis=1).to_numpy(dtype=float)
    shares = np.divide(values, totals[:, None], out=np.zeros_like(values), where=totals[:, None] > 0)
    eligible = np.any((values >= float(min_year_count)) & (shares * 100 >= float(min_year_percent)), axis=0)
    return tuple(sorted(np.asarray(non_transient, dtype=str)[eligible].tolist()))


def rank_serotypes(counts: pd.DataFrame, eligible_serotypes: Iterable[str] | None = None) -> tuple[str, ...]:
    """Rank eligible serotypes by total count with deterministic tie-breaking."""
    if counts.empty:
        return tuple()
    eligible = set(counts.columns.astype(str)) if eligible_serotypes is None else {str(value) for value in eligible_serotypes}
    totals = counts.reindex(columns=sorted(eligible), fill_value=0.0).sum(axis=0).astype(float)
    return tuple(sorted(totals.index.astype(str), key=lambda label: (-float(totals[label]), label)))


def build_entity_groupings(
    *,
    entity_order: tuple[str, ...],
    display_counts: Mapping[str, pd.DataFrame],
    analysis_counts: Mapping[str, pd.DataFrame],
    transient_serotypes: Mapping[str, tuple[str, ...]],
    config: TopNConfig,
) -> dict[str, EntityGrouping]:
    """Select taxa and grouping flags for every analysis entity."""
    # Resolve selection once per entity, then derive display grouping flags.
    selected_by_entity = config.selection.select(
        entity_order, display_counts, analysis_counts, transient_serotypes, config
    )

    groupings = {}
    for entity in entity_order:
        transient = tuple(transient_serotypes[entity])
        selected = tuple(label for label in selected_by_entity[entity] if label not in set(transient))
        original = set(display_counts[entity].columns.astype(str))
        selected_set = set(selected)
        transient_set = set(transient)
        groupings[entity] = EntityGrouping(
            entity=entity,
            selected_serotypes=selected,
            transient_serotypes=transient,
            include_other=bool(original.difference(selected_set).difference(transient_set)),
            include_transient=bool(original.intersection(transient_set)),
        )
    return groupings


def build_master_display_columns(
    display_counts: Mapping[str, pd.DataFrame],
    groupings: Mapping[str, EntityGrouping],
) -> tuple[str, ...]:
    """Build the shared display-column order across all entities."""
    selected_union = {label for grouping in groupings.values() for label in grouping.selected_serotypes}
    global_totals = display_counts[GLOBAL_ENTITY].reindex(columns=sorted(selected_union), fill_value=0.0).sum(axis=0)
    selected_order = tuple(sorted(selected_union, key=lambda label: (-float(global_totals[label]), label)))
    columns = list(selected_order)
    if any(grouping.include_other for grouping in groupings.values()):
        columns.append(OTHER_NAME)
    if any(grouping.include_transient for grouping in groupings.values()):
        columns.append(TRANSIENT_NAME)
    return tuple(columns)


def _assert_totals_equal(left: pd.Series, right: pd.Series, context: str) -> None:
    if not np.allclose(left.to_numpy(dtype=float), right.to_numpy(dtype=float)):
        raise AssertionError(f"{context}: grouped totals do not equal raw totals.")


def group_display_counts(counts: pd.DataFrame, grouping: EntityGrouping, master_columns: tuple[str, ...]) -> pd.DataFrame:
    """Group display counts into selected, other, and transient columns."""
    selected = set(grouping.selected_serotypes)
    transient = set(grouping.transient_serotypes)
    output_columns = {
        label: counts[label] if label in counts.columns else pd.Series(0.0, index=counts.index)
        for label in grouping.selected_serotypes
    }
    other = [label for label in counts.columns.astype(str) if label not in selected and label not in transient]
    if grouping.include_other:
        output_columns[OTHER_NAME] = counts[other].sum(axis=1) if other else 0.0
    transient_present = [label for label in grouping.transient_serotypes if label in counts.columns]
    if grouping.include_transient:
        output_columns[TRANSIENT_NAME] = counts[transient_present].sum(axis=1) if transient_present else 0.0
    output = pd.DataFrame(output_columns, index=counts.index)
    grouped = output.reindex(columns=list(master_columns), fill_value=0.0).astype(float)
    _assert_totals_equal(counts.sum(axis=1), grouped.sum(axis=1), f"{grouping.entity} grouped display totals")
    return grouped


def build_mh_counts(
    counts: pd.DataFrame,
    grouping: EntityGrouping,
    *,
    other_grouping: TaxonGroupingPolicy,
    transient_grouping: TaxonGroupingPolicy,
) -> pd.DataFrame:
    """Build the MH input matrix using configured grouping modes."""
    selected = set(grouping.selected_serotypes)
    transient = set(grouping.transient_serotypes)
    columns: dict[str, pd.Series | float] = {
        label: counts[label].astype(float) for label in grouping.selected_serotypes if label in counts.columns
    }
    other = [label for label in counts.columns.astype(str) if label not in selected and label not in transient]
    transient_present = [label for label in grouping.transient_serotypes if label in counts.columns]
    other_grouping.add(columns, counts, other, OTHER_NAME)
    transient_grouping.add(columns, counts, transient_present, TRANSIENT_NAME)
    return pd.DataFrame(columns, index=counts.index).astype(float) if columns else pd.DataFrame(index=counts.index, dtype=float)


class DataPreparer:
    """Prepare validated counts, groupings, palettes, and MH matrices."""

    def __init__(self, config):
        self.config = config

    def prepare(
        self,
        raw_counts: pd.DataFrame | None = None,
        *,
        palette_builder=None,
        palette_master_labels: Sequence[str] | None = None,
    ) -> PreparedData:
        """Prepare one analysis dataset from a dataframe or configured path.

        Args:
            raw_counts: Optional validated or raw long-form count dataframe.
            palette_builder: Optional custom palette service.
            palette_master_labels: Optional stable label universe for colors.

        Returns:
            Prepared data for the global aggregate and configured countries.

        Raises:
            ValueError: If input data or configuration cannot produce an analysis.
        """
        from .palette import PaletteBuilder

        if raw_counts is None:
            if self.config.input.data_path is None:
                raise ValueError("input.data_path is required when raw_counts is not provided.")
            counts = load_counts(self.config.input.data_path)
        else:
            counts = validate_counts_frame(raw_counts)
        countries = resolve_configured_countries(counts, self.config.input.countries)
        country_frame = counts[counts["country"].isin(countries)].copy()
        entity_order = (GLOBAL_ENTITY, *countries)
        # Build raw matrices once so every downstream step shares the same schema.
        raw_by_entity = build_raw_entity_count_matrices(counts, countries)
        country_selection = {country: select_years_for_entity(raw_by_entity[country], self.config.year_selection, country) for country in countries}
        display_counts, analysis_counts, selections = build_display_and_analysis_matrices(
            raw_counts_by_entity=raw_by_entity,
            country_frame=country_frame,
            countries=countries,
            country_year_selection=country_selection,
            config=self.config.year_selection,
        )
        transient = determine_transient_serotypes(
            entity_order, analysis_counts, self.config.top_n.transient, display_counts
        )
        groupings = build_entity_groupings(
            entity_order=entity_order,
            display_counts=display_counts,
            analysis_counts=analysis_counts,
            transient_serotypes=transient,
            config=self.config.top_n,
        )
        master_columns = build_master_display_columns(display_counts, groupings)
        grouped = {entity: group_display_counts(display_counts[entity], groupings[entity], master_columns) for entity in entity_order}
        master_labels = tuple(dict.fromkeys(str(label) for label in (palette_master_labels or master_columns)))
        builder = palette_builder or PaletteBuilder()
        palette = builder.build(master_columns, displayed_stacks=grouped, master_labels=master_labels, master_displayed_stacks=display_counts)
        entities = {}
        for entity in entity_order:
            mh_counts = build_mh_counts(
                analysis_counts[entity], groupings[entity],
                other_grouping=self.config.mh.other_grouping,
                transient_grouping=self.config.mh.transient_grouping,
            )
            if not set(mh_counts.index.astype(int)).issubset(selections[entity].selected_years):
                raise AssertionError(f"{entity}: MH counts include years outside selected analytical years.")
            entities[entity] = PreparedEntityData(
                entity=entity,
                raw_counts=raw_by_entity[entity],
                display_counts=display_counts[entity],
                analysis_counts=analysis_counts[entity],
                grouped_display_counts=grouped[entity],
                mh_counts=mh_counts,
                grouping=groupings[entity],
                year_selection=selections[entity],
                display_total=float(display_counts[entity].to_numpy(dtype=float).sum()),
                analysis_total=float(analysis_counts[entity].to_numpy(dtype=float).sum()),
            )
        return PreparedData(entity_order=entity_order, entities=entities, master_display_columns=master_columns, palette=palette)
