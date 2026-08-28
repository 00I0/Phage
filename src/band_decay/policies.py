"""Behavior policies used by the analysis pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .constants import GLOBAL_ENTITY, OTHER_TAXON, TRANSIENT_TAXON

if TYPE_CHECKING:
    from .domain import CurveParameters, DecayFit, EntityYearSelection


class YearSelectionPolicy(ABC):
    """Select analytical years from available and qualifying years."""

    def option_name(self) -> str:
        """Return the concise name used by configuration controls."""
        return self.__class__.__name__

    @abstractmethod
    def select(self, available: tuple[int, ...], qualifying: frozenset[int]) -> frozenset[int]:
        """Return selected years."""


class AllAvailableYears(YearSelectionPolicy):
    """Select every available year."""

    def select(self, available: tuple[int, ...], qualifying: frozenset[int]) -> frozenset[int]:
        """Select every available year."""
        return frozenset(available)


class QualifyingYears(YearSelectionPolicy):
    """Select only years meeting the configured threshold."""

    def select(self, available: tuple[int, ...], qualifying: frozenset[int]) -> frozenset[int]:
        """Select only qualifying years."""
        return qualifying


class LargestContiguousBlock(YearSelectionPolicy):
    """Select the earliest longest contiguous qualifying-year block."""

    def select(self, available: tuple[int, ...], qualifying: frozenset[int]) -> frozenset[int]:
        """Select the earliest longest qualifying block."""
        values = sorted(qualifying)
        if not values:
            return frozenset()
        best_start = current_start = 0
        best_length = current_length = 1
        for index in range(1, len(values)):
            if values[index] == values[index - 1] + 1:
                current_length += 1
                continue
            if current_length > best_length:
                best_start, best_length = current_start, current_length
            current_start, current_length = index, 1
        if current_length > best_length:
            best_start, best_length = current_start, current_length
        return frozenset(values[best_start:best_start + best_length])


class DisplayAxisPolicy(ABC):
    """Choose the years represented on a display axis."""

    def option_name(self) -> str:
        """Return the concise name used by configuration controls."""
        return self.__class__.__name__

    @abstractmethod
    def years(self, selection: EntityYearSelection, union_years: tuple[int, ...]) -> tuple[int, ...]:
        """Return the entity’s continuous available-year span."""
        """Return display years."""


class EntityAvailableYears(DisplayAxisPolicy):
    """Use each entity’s continuous available-year span."""

    def years(self, selection: EntityYearSelection, union_years: tuple[int, ...]) -> tuple[int, ...]:
        """Return the entity’s continuous available-year span."""
        values = tuple(sorted(selection.available_years))
        return tuple(range(values[0], values[-1] + 1)) if values else tuple()


class UnionAvailableYears(DisplayAxisPolicy):
    """Use one continuous year axis shared by all entities."""

    def years(self, selection: EntityYearSelection, union_years: tuple[int, ...]) -> tuple[int, ...]:
        """Return the shared continuous year axis."""
        return union_years


class TransientPolicy(ABC):
    """Identify transient serotypes at a chosen scope."""

    def option_name(self) -> str:
        """Return the concise name used by configuration controls."""
        return self.__class__.__name__

    @abstractmethod
    def resolve(
        self,
        entity_order: tuple[str, ...],
        analysis_counts: Mapping[str, pd.DataFrame],
        display_counts: Mapping[str, pd.DataFrame],
    ) -> dict[str, tuple[str, ...]]:
        """Return transient labels by entity."""


def _transient_labels(counts: pd.DataFrame) -> tuple[str, ...]:
    if counts.empty:
        return tuple()
    years_present = (counts > 0).sum(axis=0)
    return tuple(sorted(years_present[years_present == 1].index.astype(str)))


class NoTransientTaxa(TransientPolicy):
    """Never classify a serotype as transient."""

    def resolve(self, entity_order, analysis_counts, display_counts):
        """Return no transient taxa for any entity."""
        return {entity: tuple() for entity in entity_order}


class GlobalTransientTaxa(TransientPolicy):
    """Use the aggregate display timeline for every entity."""

    def resolve(self, entity_order, analysis_counts, display_counts):
        """Return aggregate transient taxa for every entity."""
        labels = _transient_labels(display_counts[GLOBAL_ENTITY])
        return {entity: labels for entity in entity_order}


class PerEntityTransientTaxa(TransientPolicy):
    """Detect transient labels independently for every entity."""

    def resolve(self, entity_order, analysis_counts, display_counts):
        """Return transient taxa independently for each entity."""
        return {entity: _transient_labels(display_counts[entity]) for entity in entity_order}


class TaxonRankingPolicy(ABC):
    """Choose the count matrix used to rank taxa."""

    def option_name(self) -> str:
        """Return the concise name used by configuration controls."""
        return self.__class__.__name__

    @abstractmethod
    def counts(self, entity: str, display_counts: Mapping[str, pd.DataFrame], analysis_counts: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        """Return the ranking matrix for an entity."""

    def counts_for(self, entity) -> pd.DataFrame:
        """Return the matrix used to rank one prepared entity."""
        return self.counts(entity.entity, {entity.entity: entity.display_counts}, {entity.entity: entity.analysis_counts})


class SelectedYearRanking(TaxonRankingPolicy):
    """Rank from selected analytical years."""

    def counts(self, entity, display_counts, analysis_counts):
        """Return selected-year counts for an entity."""
        return analysis_counts[entity]


class AvailableYearRanking(TaxonRankingPolicy):
    """Rank from all displayed available years."""

    def counts(self, entity, display_counts, analysis_counts):
        """Return available-year counts for an entity."""
        return display_counts[entity]


class TaxonRanker:
    """Apply eligibility filters and deterministic total-count ranking."""

    def eligible(self, counts: pd.DataFrame, transient: Iterable[str], minimum_count: float, minimum_percent: float) -> tuple[str, ...]:
        """Return labels passing transient and abundance filters."""
        excluded = set(transient)
        labels = [label for label in counts.columns.astype(str) if label not in excluded]
        if not labels or (minimum_count <= 0 and minimum_percent <= 0):
            return tuple(sorted(labels))
        values = counts.reindex(columns=labels, fill_value=0.0).to_numpy(dtype=float)
        totals = counts.sum(axis=1).to_numpy(dtype=float)
        shares = np.divide(values, totals[:, None], out=np.zeros_like(values), where=totals[:, None] > 0)
        eligible = np.any((values >= float(minimum_count)) & (shares * 100 >= float(minimum_percent)), axis=0)
        return tuple(sorted(np.asarray(labels, dtype=str)[eligible].tolist()))

    def ranked(self, counts: pd.DataFrame, labels: Iterable[str]) -> tuple[str, ...]:
        """Return labels ordered by total count and name."""
        labels = tuple(str(label) for label in labels)
        totals = counts.reindex(columns=sorted(set(labels)), fill_value=0.0).sum(axis=0).astype(float)
        return tuple(sorted(totals.index.astype(str), key=lambda label: (-float(totals[label]), label)))

    def select(self, counts: pd.DataFrame, transient: Iterable[str], n: int, minimum_count: float, minimum_percent: float) -> tuple[str, ...]:
        """Return the requested number of ranked eligible labels."""
        eligible = self.eligible(counts, transient, minimum_count, minimum_percent)
        ranked = self.ranked(counts, eligible)
        return ranked if n == 0 else ranked[:n]


class TopNSelectionPolicy(ABC):
    """Select labels globally or independently per entity."""

    def option_name(self) -> str:
        """Return the concise name used by configuration controls."""
        return self.__class__.__name__

    @abstractmethod
    def select(
        self,
        entity_order: tuple[str, ...],
        display_counts: Mapping[str, pd.DataFrame],
        analysis_counts: Mapping[str, pd.DataFrame],
        transient: Mapping[str, tuple[str, ...]],
        config,
    ) -> dict[str, tuple[str, ...]]:
        """Return selected labels by entity."""

    @abstractmethod
    def describe(self, n: int) -> str:
        """Return a concise description for plot titles."""

    @abstractmethod
    def supports_country_overrides(self) -> bool:
        """Return whether per-entity overrides are meaningful."""


class GlobalTopN(TopNSelectionPolicy):
    """Select one aggregate ranking for every entity."""

    def select(self, entity_order, display_counts, analysis_counts, transient, config):
        """Apply one aggregate ranking to every entity."""
        ranker = TaxonRanker()
        source = config.ranking.counts(GLOBAL_ENTITY, display_counts, analysis_counts)
        selected = ranker.select(source, transient[GLOBAL_ENTITY], config.n, config.min_year_count, config.min_year_percent)
        return {entity: selected for entity in entity_order}

    def describe(self, n: int) -> str:
        """Describe global Top-N selection."""
        return f"top-{n}"

    def supports_country_overrides(self) -> bool:
        """Return whether global selection uses country overrides."""
        return False


class PerEntityTopN(TopNSelectionPolicy):
    """Select a separate ranked set for every entity."""

    def select(self, entity_order, display_counts, analysis_counts, transient, config):
        """Apply an independent ranking to each entity."""
        ranker = TaxonRanker()
        selected = {}
        for entity in entity_order:
            source = config.ranking.counts(entity, display_counts, analysis_counts)
            selected[entity] = ranker.select(
                source,
                transient[entity],
                config.n_for_country(entity),
                config.min_year_count_for_country(entity),
                config.min_year_percent_for_country(entity),
            )
        if config.global_selected_serotypes is not None:
            selected[GLOBAL_ENTITY] = config.global_selected_serotypes
        return selected

    def describe(self, n: int) -> str:
        """Describe per-entity Top-N selection."""
        return "per-country Top-N"

    def supports_country_overrides(self) -> bool:
        """Return whether per-entity selection uses country overrides."""
        return True


class TaxonGroupingPolicy(ABC):
    """Add a group of labels to an output count mapping."""

    def option_name(self) -> str:
        """Return the concise name used by configuration controls."""
        return self.__class__.__name__

    @abstractmethod
    def add(self, output: dict[str, pd.Series | float], counts: pd.DataFrame, labels: Iterable[str], output_label: str) -> None:
        """Apply this grouping policy."""


class CollapseTaxa(TaxonGroupingPolicy):
    """Collapse labels into one output column."""

    def add(self, output, counts, labels, output_label):
        """Add labels as one summed column."""
        labels = tuple(label for label in labels if label in counts.columns)
        output[output_label] = counts[list(labels)].sum(axis=1) if labels else 0.0


class KeepTaxa(TaxonGroupingPolicy):
    """Keep every label as its own output column."""

    def add(self, output, counts, labels, output_label):
        """Add labels as separate columns."""
        output.update({label: counts[label].astype(float) for label in labels if label in counts.columns})


class SkipTaxa(TaxonGroupingPolicy):
    """Exclude a group of labels from the output."""

    def add(self, output, counts, labels, output_label):
        """Skip the requested labels."""
        return None


class DecayDisplayPolicy(ABC):
    """Render fitted curves on a chosen scale."""

    def option_name(self) -> str:
        """Return the concise name used by configuration controls."""
        return self.__class__.__name__

    @abstractmethod
    def curve_draws(self, fit: DecayFit, x_values: np.ndarray) -> np.ndarray:
        """Return posterior curve draws."""

    @abstractmethod
    def median_curve(self, fit: DecayFit, x_values: np.ndarray) -> np.ndarray:
        """Return the pointwise median curve."""

    @abstractmethod
    def interval(self, fit: DecayFit, x_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the central posterior band."""

    @abstractmethod
    def asymptote(self, fit: DecayFit) -> float:
        """Return the displayed asymptote."""

    @abstractmethod
    def y_label(self) -> str:
        """Return the y-axis label."""

    @abstractmethod
    def parameter_label(self) -> str:
        """Return the asymptote parameter label."""

    @abstractmethod
    def includes_direct_parameter(self) -> bool:
        """Return whether direct ``c`` should be shown in summaries."""

    @abstractmethod
    def posterior_parameters(self, fit: DecayFit) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return posterior parameters on the displayed scale."""

    @abstractmethod
    def parameter_curve(self, parameters: CurveParameters, x_values: np.ndarray) -> np.ndarray:
        """Evaluate standalone curve parameters on the displayed scale."""


def _finite_posterior_parameters(fit: DecayFit) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y0 = np.asarray(fit.y0_samples, dtype=float).reshape(-1)
    b = np.asarray(fit.b_samples, dtype=float).reshape(-1)
    c = np.asarray(fit.c_samples, dtype=float).reshape(-1)
    if not len(y0) or len(y0) != len(b) or len(y0) != len(c):
        raise ValueError("DecayFit posterior sample arrays must have equal nonzero lengths.")
    valid = np.isfinite(y0) & np.isfinite(b) & np.isfinite(c)
    if not np.any(valid):
        raise ValueError("DecayFit does not contain finite joint posterior draws.")
    return y0[valid], b[valid], c[valid]


class OriginalDecay(DecayDisplayPolicy):
    """Display fitted curves on the original MH scale."""

    def curve_draws(self, fit, x_values):
        """Return posterior curves on the original scale."""
        return fit.curve_draws(x_values)

    def median_curve(self, fit, x_values):
        """Return the original-scale posterior median curve."""
        return fit.median_curve(x_values)

    def interval(self, fit, x_values):
        """Return the original-scale posterior interval."""
        return fit.interval(x_values)

    def asymptote(self, fit):
        """Return the original-scale asymptote."""
        return float(np.median(self.posterior_parameters(fit)[2]))

    def y_label(self):
        """Return the original-scale axis label."""
        return "Morisita-Horn similarity"

    def parameter_label(self):
        """Return the original-scale parameter label."""
        return "c"

    def includes_direct_parameter(self):
        """Return whether the displayed scale includes direct ``c``."""
        return True

    def posterior_parameters(self, fit):
        """Return finite posterior parameters unchanged."""
        return _finite_posterior_parameters(fit)

    def parameter_curve(self, parameters, x_values):
        """Evaluate standalone parameters on the original scale."""
        return parameters.evaluate(x_values)


class NormalizedDecay(DecayDisplayPolicy):
    """Display fitted curves normalized by each draw’s ``y0``."""

    def curve_draws(self, fit, x_values):
        """Return posterior curves normalized by each draw’s intercept."""
        return fit.normalized_curve_draws(x_values)

    def median_curve(self, fit, x_values):
        """Return the normalized posterior median curve."""
        return fit.normalized_median_curve(x_values)

    def interval(self, fit, x_values):
        """Return the normalized posterior interval."""
        return fit.normalized_interval(x_values)

    def asymptote(self, fit):
        """Return the normalized asymptote."""
        return float(np.median(self.posterior_parameters(fit)[2]))

    def y_label(self):
        """Return the normalized axis label."""
        return "Morisita-Horn similarity (y₀-normalized fit)"

    def parameter_label(self):
        """Return the normalized parameter label."""
        return "c/y0"

    def includes_direct_parameter(self):
        """Return whether the displayed scale includes direct ``c``."""
        return False

    def posterior_parameters(self, fit):
        """Return finite posterior parameters on the normalized scale."""
        y0, b, c = _finite_posterior_parameters(fit)
        valid = y0 != 0
        if not np.any(valid):
            raise ValueError("DecayFit has no nonzero y0 posterior draws.")
        return np.ones(int(valid.sum())), b[valid], c[valid] / y0[valid]

    def parameter_curve(self, parameters, x_values):
        """Evaluate standalone parameters on the normalized scale."""
        if parameters.y0 == 0:
            raise ValueError("Cannot normalize a curve with y0=0.")
        return parameters.evaluate(x_values) / parameters.y0
