"""Palette construction and staged candidate optimization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .candidates import _candidate_pool
from .colors import _srgb_to_hex
from .metrics import (
    _abundance_order,
    _canonical_taxa,
    _display_metrics,
    _labels_from_stacks,
    _materialize_stacks,
    _metrics_digest,
    _pairwise_distances,
    _sector_counts,
    _stable_label_key,
    _hue_sectors,
)
from .settings import (
    OTHER_TAXON,
    _SPECIAL_TAXA,
    _TIER_HERO,
    _TIER_INTERMEDIATE,
    _TIER_RARE,
    DEFAULT_PALETTE_SETTINGS,
    PaletteSettings,
    TRANSIENT_TAXON,
)
from .types import CandidatePool, PaletteState


_PALETTE_CACHE: dict[tuple[tuple[str, ...], str, PaletteSettings], tuple[str, ...]] = {}


def build_taxon_palette(
        labels: Sequence[str],
        *,
        displayed_stacks: Mapping[Any, pd.DataFrame] | Iterable[pd.DataFrame] | None = None,
        settings: PaletteSettings = DEFAULT_PALETTE_SETTINGS,
        master_labels: Sequence[str] | None = None,
        master_displayed_stacks: Mapping[Any, pd.DataFrame] | Iterable[pd.DataFrame] | None = None,
) -> dict[str, str]:
    """Build a deterministic, hierarchical abundance- and adjacency-aware palette.

    displayed_stacks should contain the exact count matrices passed to the
    stackplots. master_labels and master_displayed_stacks optionally describe a
    stable superset used to anchor colors when a subset of taxa is plotted.
    When master data is supplied, the returned mapping is still restricted to
    labels, while high-priority colors come from the stable full universe.
    """

    ordered_labels = tuple(dict.fromkeys(str(label) for label in labels))
    if len(ordered_labels) != len(labels):
        raise ValueError("Palette labels must be unique.")
    # Optimize against the stable master universe while returning requested labels only.
    displayed_stack_tuple = _materialize_stacks(displayed_stacks)
    master_stack_tuple = _materialize_stacks(master_displayed_stacks)
    inferred_master_labels = _labels_from_stacks(master_stack_tuple or displayed_stack_tuple)
    supplied_master_labels = (
        tuple(dict.fromkeys(str(label) for label in master_labels)) if master_labels is not None else tuple()
    )
    optimization_label_set = set(ordered_labels)
    if supplied_master_labels:
        optimization_label_set.update(supplied_master_labels)
    else:
        optimization_label_set.update(inferred_master_labels)
    optimization_taxa = _canonical_taxa(
        (label for label in optimization_label_set if label not in _SPECIAL_TAXA),
        settings,
    )
    special_colors = {
        OTHER_TAXON: settings.other_color,
        TRANSIENT_TAXON: settings.transient_color,
    }
    palette = {label: special_colors[label] for label in ordered_labels if label in special_colors}
    dynamic_requested = tuple(label for label in ordered_labels if label not in _SPECIAL_TAXA)
    if not dynamic_requested:
        return {label: palette[label] for label in ordered_labels}

    metric_labels = (
        tuple(dict.fromkeys((*supplied_master_labels, *ordered_labels)))
        if supplied_master_labels
        else inferred_master_labels or ordered_labels
    )
    metric_stacks = master_stack_tuple if master_stack_tuple is not None else displayed_stack_tuple
    abundance, adjacency = _display_metrics(optimization_taxa, metric_labels, metric_stacks, settings)
    cache_key = (optimization_taxa, _metrics_digest(abundance, adjacency), settings)
    dynamic_colors = _PALETTE_CACHE.get(cache_key) if settings.cache_enabled else None
    if dynamic_colors is None:
        dynamic_colors = _optimize_palette(optimization_taxa, abundance, adjacency, settings)
        if settings.cache_enabled:
            _PALETTE_CACHE[cache_key] = dynamic_colors
    color_by_taxon = dict(zip(optimization_taxa, dynamic_colors, strict=True))
    palette.update({taxon: color_by_taxon[taxon] for taxon in dynamic_requested})
    return {label: palette[label] for label in ordered_labels}


def _optimize_palette(
        taxa: tuple[str, ...],
        abundance: np.ndarray,
        adjacency: np.ndarray,
        settings: PaletteSettings,
) -> tuple[str, ...]:
    """Build the palette in protected hero, intermediate, and rare stages."""

    # Allocate protected hero colors first, then assign and refine lower tiers.
    if not taxa:
        return tuple()
    if len(taxa) > int(settings.candidate_count):
        raise ValueError("candidate_count must be at least the number of dynamic taxa.")

    taxon_order, _ = _abundance_order(taxa, abundance, settings)
    hero_count = min(int(settings.hero_taxon_count), len(taxa))
    pool_size = max(int(settings.candidate_count), int(settings.hero_taxon_count) + 8)
    pools = {
        tier: _candidate_pool(settings, tier, pool_size)
        for tier in (_TIER_HERO, _TIER_INTERMEDIATE, _TIER_RARE)
    }
    hero_slots = _select_hero_candidates(
        pools[_TIER_HERO],
        int(settings.hero_taxon_count),
        settings,
    )

    condition_count = pools[_TIER_HERO].labs.shape[0]
    state = PaletteState(
        srgb=np.zeros((len(taxa), 3), dtype=float),
        oklch=np.zeros((len(taxa), 3), dtype=float),
        labs=np.zeros((condition_count, len(taxa), 3), dtype=float),
        tiers=[None] * len(taxa),
        pool_indices=np.full(len(taxa), -1, dtype=int),
        used_by_tier={tier: set() for tier in pools},
    )

    for rank, taxon_index in enumerate(taxon_order[:hero_count]):
        _set_selected_candidate(
            taxon_index,
            _TIER_HERO,
            int(hero_slots[rank]),
            pools,
            state,
        )

    for rank, taxon_index in enumerate(taxon_order[hero_count:], start=hero_count):
        tier = _tier_for_rank(rank, settings)
        candidate_index = _choose_candidate_for_taxon(
            taxon_index=taxon_index,
            tier=tier,
            pools=pools,
            assigned_indices=taxon_order[:rank],
            top_indices=taxon_order[:hero_count],
            selected_oklch=state.oklch,
            selected_labs=state.labs,
            used_by_tier=state.used_by_tier,
            adjacency=adjacency,
            settings=settings,
        )
        _set_selected_candidate(
            taxon_index,
            tier,
            candidate_index,
            pools,
            state,
        )

    _refine_nonhero_assignments(
        taxon_order=taxon_order,
        hero_count=hero_count,
        pools=pools,
        state=state,
        adjacency=adjacency,
        settings=settings,
    )
    return tuple(_srgb_to_hex(color) for color in state.srgb)


def _tier_for_rank(rank: int, settings: PaletteSettings) -> str:
    if rank < int(settings.hero_taxon_count):
        return _TIER_HERO
    if rank < int(settings.intermediate_taxon_count):
        return _TIER_INTERMEDIATE
    return _TIER_RARE


def _abundance_order(
        taxa: tuple[str, ...],
        abundance: np.ndarray,
        settings: PaletteSettings,
) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    """Rank abundance with hash-permuted exact and near-tie groups."""

    values = np.asarray(abundance, dtype=float)
    safe_values = np.where(np.isfinite(values), values, 0.0)
    preliminary = sorted(range(len(taxa)), key=lambda index: -float(safe_values[index]))
    groups: list[tuple[int, ...]] = []
    position = 0
    tolerance = float(settings.abundance_tie_tolerance)
    while position < len(preliminary):
        group = [preliminary[position]]
        anchor = float(safe_values[group[0]])
        position += 1
        while position < len(preliminary):
            candidate = float(safe_values[preliminary[position]])
            denominator = max(abs(anchor), abs(candidate), 1.0e-12)
            if abs(anchor - candidate) > tolerance * denominator:
                break
            group.append(preliminary[position])
            position += 1
        group.sort(
            key=lambda index: _stable_label_key(
                taxa[index],
                int(settings.tie_randomization_seed),
            )
        )
        groups.append(tuple(group))
    order = np.asarray([index for group in groups for index in group], dtype=int)
    return order, tuple(groups)


def _select_hero_candidates(
        pool: CandidatePool,
        target_count: int,
        settings: PaletteSettings,
) -> np.ndarray:
    """Select a stable hero set by maximin distance, then balance its hue."""

    # Greedily maximize separation while maintaining hue-sector balance.
    target_count = min(int(target_count), pool.srgb.shape[0])
    if target_count <= 0:
        return np.empty(0, dtype=int)
    normal_labs = pool.labs[0]
    chroma = pool.oklch[:, 1]
    central_lightness = 1.0 - np.abs(pool.oklch[:, 0] - 0.61) / 0.22
    first_quality = 0.7 * chroma + 0.3 * np.clip(central_lightness, 0.0, 1.0)
    first_index = int(np.argmax(first_quality))
    selected = [first_index]
    selected_mask = np.zeros(pool.srgb.shape[0], dtype=bool)
    selected_mask[first_index] = True
    sector_counts = np.zeros(int(settings.hue_sector_count), dtype=int)
    sector_counts[_hue_sector(pool.oklch[first_index, 2], settings)] += 1

    while len(selected) < target_count:
        available = np.flatnonzero(~selected_mask)
        distances = _worst_case_distances_to_selected(pool.labs, np.asarray(selected, dtype=int), settings)
        all_minimum_distance = np.min(distances[available], axis=1)
        best_minimum_distance = float(np.max(all_minimum_distance))
        distance_tolerance = float(settings.hue_balance_distance_tolerance)
        eligible = all_minimum_distance >= best_minimum_distance - distance_tolerance
        available = available[eligible]
        minimum_distance = all_minimum_distance[eligible]
        candidate_sectors = _hue_sectors(pool.oklch[available, 2], settings)
        least_occupied = int(np.min(sector_counts))
        underrepresented = sector_counts[candidate_sectors] <= least_occupied
        if np.any(underrepresented):
            available = available[underrepresented]
            minimum_distance = minimum_distance[underrepresented]
            candidate_sectors = candidate_sectors[underrepresented]
        hue_quality = _hue_quality_after_candidates(
            sector_counts,
            candidate_sectors,
            settings,
        )
        lightness_coverage = _range_after_candidates(
            pool.oklch[available, 0],
            pool.oklch[np.asarray(selected, dtype=int), 0],
            settings.lightness_min,
            settings.lightness_max,
        )
        chroma_coverage = _range_after_candidates(
            pool.oklch[available, 1],
            pool.oklch[np.asarray(selected, dtype=int), 1],
            settings.chroma_min,
            settings.chroma_max,
        )
        average_distance = np.mean(distances[available], axis=1)
        components = (
            float(settings.top_minimum_distance_priority) * minimum_distance,
            float(settings.top_hue_balance_strength) * hue_quality,
            lightness_coverage + chroma_coverage,
            average_distance,
        )
        chosen_position = _best_position(components)
        chosen_index = int(available[chosen_position])
        selected.append(chosen_index)
        selected_mask[chosen_index] = True
        sector_counts[_hue_sector(pool.oklch[chosen_index, 2], settings)] += 1

    selected = _refine_hero_set(pool, np.asarray(selected, dtype=int), settings)
    return selected


def _refine_hero_set(
        pool: CandidatePool,
        initial: np.ndarray,
        settings: PaletteSettings,
) -> np.ndarray:
    if len(initial) < 2 or int(settings.optimization_iterations) <= 0:
        return initial
    # Use seeded replacement search to improve the greedy hero set.
    generator = np.random.default_rng(int(settings.seed) + 101)
    current = initial.copy()
    current_key = _hero_key(pool, current, settings)
    distance_tolerance = float(settings.hue_balance_distance_tolerance)
    minimum_floor = current_key[0] - distance_tolerance
    iterations = min(800, max(1, int(settings.optimization_iterations) // 2))
    for _ in range(iterations):
        slot = int(generator.integers(len(current)))
        selected_mask = np.zeros(pool.srgb.shape[0], dtype=bool)
        selected_mask[current] = True
        available = np.flatnonzero(~selected_mask)
        if available.size == 0:
            break
        replacement = int(generator.choice(available))
        proposed = current.copy()
        proposed[slot] = replacement
        proposed_key = _hero_key(pool, proposed, settings)
        if proposed_key[0] >= minimum_floor and _hero_move_is_better(proposed_key, current_key, settings):
            current = proposed
            current_key = proposed_key
    return current


def _hero_move_is_better(
        proposed_key: tuple[float, ...],
        current_key: tuple[float, ...],
        settings: PaletteSettings,
) -> bool:
    if proposed_key[0] > current_key[0] + 1.0e-12:
        return True
    allowed_loss = float(settings.top_minimum_distance_priority) * float(settings.hue_balance_distance_tolerance)
    if proposed_key[0] < current_key[0] - allowed_loss:
        return False
    return proposed_key[1:] > current_key[1:]


def _hero_key(pool: CandidatePool, selected: np.ndarray, settings: PaletteSettings) -> tuple[float, ...]:
    labs = pool.labs[:, selected, :]
    distances = _pairwise_distances(labs, settings)
    pair_distances = distances[np.triu_indices(len(selected), k=1)]
    minimum_distance = float(np.min(pair_distances)) if pair_distances.size else 0.0
    sector_counts = _sector_counts(pool.oklch[selected, 2], settings)
    hue_quality = _hue_quality(sector_counts)
    lightness_range = _normalized_range(pool.oklch[selected, 0], settings.lightness_min, settings.lightness_max)
    chroma_range = _normalized_range(pool.oklch[selected, 1], settings.chroma_min, settings.chroma_max)
    average_distance = float(np.mean(pair_distances)) if pair_distances.size else 0.0
    return (
        float(settings.top_minimum_distance_priority) * minimum_distance,
        float(settings.top_hue_balance_strength) * hue_quality,
        lightness_range + chroma_range,
        average_distance,
    )


def _choose_candidate_for_taxon(
        *,
        taxon_index: int,
        tier: str,
        pools: Mapping[str, CandidatePool],
        assigned_indices: np.ndarray,
        top_indices: np.ndarray,
        selected_oklch: np.ndarray,
        selected_labs: np.ndarray,
        used_by_tier: Mapping[str, set[int]],
        adjacency: np.ndarray,
        settings: PaletteSettings,
) -> int:
    # Rank candidates by hierarchy protection, coverage, adjacency, and contrast.
    pool = pools[tier]
    used = used_by_tier[tier]
    available = np.asarray([index for index in range(pool.srgb.shape[0]) if index not in used], dtype=int)
    if available.size == 0:
        raise ValueError(f"Not enough {tier} palette candidates.")
    if assigned_indices.size == 0:
        return int(available[0])

    available = _hue_balanced_candidate_indices(
        available,
        pool.oklch[:, 2],
        selected_oklch[assigned_indices, 2],
        settings,
    )
    assigned_labs = selected_labs[:, assigned_indices, :]
    distances = _distances_to_labs(pool.labs, assigned_labs, settings)
    top_positions = np.asarray(top_indices, dtype=int)
    top_distance = (
        np.min(_distances_to_labs(pool.labs, selected_labs[:, top_positions, :], settings), axis=1)
        if top_positions.size
        else np.zeros(pool.srgb.shape[0])
    )
    global_distance = np.min(distances, axis=1)
    sector_counts = _sector_counts(selected_oklch[assigned_indices, 2], settings)
    hue_quality = _hue_quality_after_candidates(
        sector_counts,
        _hue_sectors(pool.oklch[:, 2], settings),
        settings,
    )
    lightness_coverage = _range_after_candidates(
        pool.oklch[:, 0],
        selected_oklch[assigned_indices, 0],
        settings.lightness_min,
        settings.lightness_max,
    )
    chroma_coverage = _range_after_candidates(
        pool.oklch[:, 1],
        selected_oklch[assigned_indices, 1],
        settings.chroma_min,
        settings.chroma_max,
    )
    adjacency_weights = adjacency[taxon_index, assigned_indices]
    weight_total = float(np.sum(adjacency_weights))
    if weight_total > 0.0:
        adjacency_distance = np.sum(distances * adjacency_weights[None, :], axis=1) / weight_total
        lightness_distance = np.sum(
            np.abs(pool.oklch[:, None, 0] - selected_oklch[assigned_indices, 0][None, :])
            * adjacency_weights[None, :],
            axis=1,
        ) / weight_total
    else:
        adjacency_distance = np.zeros(pool.srgb.shape[0], dtype=float)
        lightness_distance = np.zeros(pool.srgb.shape[0], dtype=float)
    average_distance = np.mean(distances, axis=1)
    components = (
        float(settings.top_to_other_minimum_distance_priority) * top_distance,
        float(settings.global_minimum_distance_priority) * global_distance,
        float(settings.hue_balance_strength) * hue_quality + lightness_coverage + chroma_coverage,
        float(settings.adjacency_strength) * adjacency_distance,
        float(settings.lightness_contrast_strength) * lightness_distance,
        average_distance,
    )
    return int(
        available[
            _best_position_with_distance_tolerance(
                tuple(component[available] for component in components),
                distance_component_count=2,
                settings=settings,
            )
        ]
    )


def _set_selected_candidate(
        taxon_index: int,
        tier: str,
        candidate_index: int,
        pools: Mapping[str, CandidatePool],
        state: PaletteState,
) -> None:
    pool = pools[tier]
    state.srgb[taxon_index] = pool.srgb[candidate_index]
    state.oklch[taxon_index] = pool.oklch[candidate_index]
    state.labs[:, taxon_index, :] = pool.labs[:, candidate_index, :]
    state.tiers[taxon_index] = tier
    state.pool_indices[taxon_index] = int(candidate_index)
    state.used_by_tier[tier].add(int(candidate_index))


def _refine_nonhero_assignments(
        *,
        taxon_order: np.ndarray,
        hero_count: int,
        pools: Mapping[str, CandidatePool],
        state: PaletteState,
        adjacency: np.ndarray,
        settings: PaletteSettings,
) -> None:
    if len(taxon_order) <= hero_count or int(settings.optimization_iterations) <= 0:
        return

    # Improve lower-tier assignments without sacrificing protected top-tier distances.
    generator = np.random.default_rng(int(settings.seed) + 211)
    top_indices = taxon_order[:hero_count]
    current_key = _palette_hierarchy_key(
        state.oklch,
        state.labs,
        top_indices,
        adjacency,
        settings,
    )
    protected_top_to_other_key = current_key[1]
    protected_top_to_other = current_key[1] / max(
        float(settings.top_to_other_minimum_distance_priority),
        np.finfo(float).eps,
    )
    mutable = taxon_order[hero_count:]
    iterations = int(settings.optimization_iterations)

    for iteration in range(iterations):
        taxon_index = int(mutable[iteration % len(mutable)])
        tier = state.tiers[taxon_index]
        if tier is None:
            continue

        pool = pools[tier]
        current_candidate = int(state.pool_indices[taxon_index])
        state.used_by_tier[tier].discard(current_candidate)
        available = np.asarray(
            [index for index in range(pool.srgb.shape[0]) if index not in state.used_by_tier[tier]],
            dtype=int,
        )
        if available.size == 0:
            state.used_by_tier[tier].add(current_candidate)
            continue

        if generator.random() < 0.25:
            proposed_candidate = int(generator.choice(available))
            proposed_tier = tier
        elif generator.random() < 0.70:
            global_repair = _choose_global_repair_candidate(
                taxon_index=taxon_index,
                tier=tier,
                pools=pools,
                top_indices=top_indices,
                selected_oklch=state.oklch,
                selected_labs=state.labs,
                selected_pool_indices=state.pool_indices,
                used_by_tier=state.used_by_tier,
                minimum_top_distance=protected_top_to_other
                                     * (1.0 - float(settings.global_repair_top_distance_loss)),
                adjacency=adjacency,
                settings=settings,
            )
            if global_repair is None:
                proposed_candidate = int(generator.choice(available))
                proposed_tier = tier
            else:
                proposed_tier, proposed_candidate = global_repair
        else:
            others = np.asarray([index for index in taxon_order if index != taxon_index], dtype=int)
            proposed_candidate = _choose_candidate_for_taxon(
                taxon_index=taxon_index,
                tier=tier,
                pools=pools,
                assigned_indices=others,
                top_indices=top_indices,
                selected_oklch=state.oklch,
                selected_labs=state.labs,
                used_by_tier=state.used_by_tier,
                adjacency=adjacency,
                settings=settings,
            )
            proposed_tier = tier

        proposal_pool = pools[proposed_tier]
        previous_srgb = state.srgb[taxon_index].copy()
        previous_oklch = state.oklch[taxon_index].copy()
        previous_labs = state.labs[:, taxon_index, :].copy()
        state.srgb[taxon_index] = proposal_pool.srgb[proposed_candidate]
        state.oklch[taxon_index] = proposal_pool.oklch[proposed_candidate]
        state.labs[:, taxon_index, :] = proposal_pool.labs[:, proposed_candidate, :]

        proposed_key = _palette_hierarchy_key(
            state.oklch,
            state.labs,
            top_indices,
            adjacency,
            settings,
        )
        if proposed_key > current_key or _global_repair_is_better(
                proposed_key,
                current_key,
                protected_top_to_other_key,
                settings,
        ):
            state.used_by_tier[tier].add(proposed_candidate)
            if proposed_tier != tier:
                state.used_by_tier[tier].discard(proposed_candidate)
                state.used_by_tier[proposed_tier].add(proposed_candidate)
            state.pool_indices[taxon_index] = proposed_candidate
            state.tiers[taxon_index] = proposed_tier
            current_key = proposed_key
        else:
            state.srgb[taxon_index] = previous_srgb
            state.oklch[taxon_index] = previous_oklch
            state.labs[:, taxon_index, :] = previous_labs
            state.used_by_tier[tier].add(current_candidate)


def _global_repair_is_better(
        proposed_key: tuple[float, ...],
        current_key: tuple[float, ...],
        protected_top_to_other: float,
        settings: PaletteSettings,
) -> bool:
    if proposed_key[0] < current_key[0] - 1.0e-12:
        return False
    if protected_top_to_other > 0.0:
        allowed_loss = float(settings.global_repair_top_distance_loss) * protected_top_to_other
        if proposed_key[1] < protected_top_to_other - allowed_loss:
            return False
    if proposed_key[2] > current_key[2] + 1.0e-12:
        return True
    if abs(proposed_key[2] - current_key[2]) > 1.0e-12:
        return False
    return proposed_key[3:] > current_key[3:]


def _choose_global_repair_candidate(
        *,
        taxon_index: int,
        tier: str,
        pools: Mapping[str, CandidatePool],
        top_indices: np.ndarray,
        selected_oklch: np.ndarray,
        selected_labs: np.ndarray,
        selected_pool_indices: np.ndarray,
        used_by_tier: Mapping[str, set[int]],
        minimum_top_distance: float,
        adjacency: np.ndarray,
        settings: PaletteSettings,
) -> tuple[str, int] | None:
    # Search both lower tiers for a candidate that repairs global separation.
    current_candidate = int(selected_pool_indices[taxon_index])
    others = np.asarray([index for index in range(selected_labs.shape[1]) if index != taxon_index], dtype=int)
    if not others.size:
        return None
    other_oklch = selected_oklch[others]
    adjacency_weights = adjacency[taxon_index, others]
    sector_counts = _sector_counts(other_oklch[:, 2], settings)
    all_tiers = (_TIER_INTERMEDIATE, _TIER_RARE)
    candidate_tiers: list[str] = []
    candidate_indices: list[int] = []
    candidate_components: list[tuple[float, ...]] = []
    for candidate_tier in all_tiers:
        pool = pools[candidate_tier]
        available = np.asarray(
            [
                index
                for index in range(pool.srgb.shape[0])
                if index not in used_by_tier[candidate_tier]
                   or (candidate_tier == tier and index == current_candidate)
            ],
            dtype=int,
        )
        if not available.size:
            continue
        distances = _distances_to_labs(pool.labs, selected_labs[:, others, :], settings)
        global_distance = np.min(distances, axis=1)
        if top_indices.size:
            top_distance = np.min(
                _distances_to_labs(pool.labs, selected_labs[:, top_indices, :], settings),
                axis=1,
            )
        else:
            top_distance = np.zeros(pool.srgb.shape[0], dtype=float)
        valid = available[top_distance[available] >= float(minimum_top_distance)]
        if not valid.size:
            continue
        valid = _hue_balanced_candidate_indices(
            valid,
            pool.oklch[:, 2],
            other_oklch[:, 2],
            settings,
        )
        hue_quality = _hue_quality_after_candidates(
            sector_counts,
            _hue_sectors(pool.oklch[:, 2], settings),
            settings,
        )
        lightness_coverage = _range_after_candidates(
            pool.oklch[:, 0],
            other_oklch[:, 0],
            settings.lightness_min,
            settings.lightness_max,
        )
        chroma_coverage = _range_after_candidates(
            pool.oklch[:, 1],
            other_oklch[:, 1],
            settings.chroma_min,
            settings.chroma_max,
        )
        if float(np.sum(adjacency_weights)) > 0.0:
            adjacency_distance = np.sum(
                distances * adjacency_weights[None, :],
                axis=1,
            ) / np.sum(adjacency_weights)
            lightness_distance = np.sum(
                np.abs(pool.oklch[:, None, 0] - other_oklch[None, :, 0])
                * adjacency_weights[None, :],
                axis=1,
            ) / np.sum(adjacency_weights)
        else:
            adjacency_distance = np.zeros(pool.srgb.shape[0], dtype=float)
            lightness_distance = np.zeros(pool.srgb.shape[0], dtype=float)
        components = (
            global_distance[valid],
            top_distance[valid],
            float(settings.hue_balance_strength) * hue_quality[valid]
            + lightness_coverage[valid]
            + chroma_coverage[valid],
            float(settings.adjacency_strength) * adjacency_distance[valid],
            float(settings.lightness_contrast_strength) * lightness_distance[valid],
        )
        for component_index in range(len(valid)):
            candidate_tiers.append(candidate_tier)
            candidate_indices.append(int(valid[component_index]))
            candidate_components.append(
                tuple(float(component[component_index]) for component in components)
            )
    if not candidate_indices:
        return None
    component_arrays = tuple(
        np.asarray([components[index] for components in candidate_components], dtype=float)
        for index in range(len(candidate_components[0]))
    )
    best_position = _best_position_with_distance_tolerance(
        component_arrays,
        distance_component_count=1,
        settings=settings,
    )
    return candidate_tiers[best_position], candidate_indices[best_position]


def _palette_hierarchy_key(
        selected_oklch: np.ndarray,
        selected_labs: np.ndarray,
        top_indices: np.ndarray,
        adjacency: np.ndarray,
        settings: PaletteSettings,
) -> tuple[float, ...]:
    # Return a lexicographic objective with higher-priority separations first.
    taxon_count = selected_oklch.shape[0]
    distances = _pairwise_distances(selected_labs, settings)
    top_indices = np.asarray(top_indices, dtype=int)
    other_indices = np.asarray(
        [index for index in range(taxon_count) if index not in set(top_indices.tolist())],
        dtype=int,
    )
    if top_indices.size >= 2:
        top_pair_distances = distances[np.ix_(top_indices, top_indices)][
            np.triu_indices(top_indices.size, k=1)
        ]
        top_minimum = float(np.min(top_pair_distances))
    else:
        top_minimum = 0.0
    if top_indices.size and other_indices.size:
        top_to_other_minimum = float(np.min(distances[np.ix_(top_indices, other_indices)]))
    else:
        top_to_other_minimum = 0.0
    if taxon_count >= 2:
        global_pair_distances = distances[np.triu_indices(taxon_count, k=1)]
        global_minimum = float(np.min(global_pair_distances))
        average_distance = float(np.mean(global_pair_distances))
    else:
        global_minimum = 0.0
        average_distance = 0.0
    sector_counts = _sector_counts(selected_oklch[:, 2], settings)
    hue_quality = _hue_quality(sector_counts)
    lightness_coverage = _normalized_range(
        selected_oklch[:, 0],
        settings.lightness_min,
        settings.lightness_max,
    )
    chroma_coverage = _normalized_range(
        selected_oklch[:, 1],
        settings.chroma_min,
        settings.chroma_max,
    )
    coverage_score = (
            float(settings.hue_balance_strength) * hue_quality
            + lightness_coverage
            + chroma_coverage
    )
    upper_triangle = np.triu_indices(taxon_count, k=1)
    adjacency_weights = adjacency[upper_triangle]
    if float(np.sum(adjacency_weights)) > 0.0:
        adjacent_distances = distances[upper_triangle]
        adjacency_score = float(
            np.sum(adjacent_distances * adjacency_weights) / np.sum(adjacency_weights)
        )
        lightness_difference = np.abs(
            selected_oklch[:, None, 0] - selected_oklch[None, :, 0]
        )[upper_triangle]
        lightness_score = float(
            np.sum(lightness_difference * adjacency_weights) / np.sum(adjacency_weights)
        )
    else:
        adjacency_score = 0.0
        lightness_score = 0.0
    return (
        float(settings.top_minimum_distance_priority) * top_minimum,
        float(settings.top_to_other_minimum_distance_priority) * top_to_other_minimum,
        float(settings.global_minimum_distance_priority) * global_minimum,
        coverage_score,
        float(settings.adjacency_strength) * adjacency_score,
        float(settings.lightness_contrast_strength) * lightness_score,
        average_distance,
    )


def _best_position(components: tuple[np.ndarray, ...]) -> int:
    if not components:
        raise ValueError("At least one candidate-selection component is required.")
    arrays = tuple(np.asarray(component, dtype=float) for component in components)
    candidate_count = len(arrays[0])
    if any(len(array) != candidate_count for array in arrays):
        raise ValueError("Candidate-selection components must have equal lengths.")
    order = np.lexsort(tuple(array for array in arrays[::-1]))
    return int(order[-1])


def _best_position_with_distance_tolerance(
        components: tuple[np.ndarray, ...],
        *,
        distance_component_count: int,
        settings: PaletteSettings,
) -> int:
    """Prefer coverage among candidates with effectively equal separation."""

    arrays = tuple(np.asarray(component, dtype=float) for component in components)
    if not arrays:
        raise ValueError("At least one candidate-selection component is required.")
    candidate_count = len(arrays[0])
    if any(len(array) != candidate_count for array in arrays):
        raise ValueError("Candidate-selection components must have equal lengths.")
    if not 0 <= int(distance_component_count) < len(arrays):
        raise ValueError("distance_component_count must leave at least one tie-break component.")

    # Keep near-tied distance candidates so coverage can decide the final choice.
    positions = np.arange(candidate_count, dtype=int)
    for distances in arrays[: int(distance_component_count)]:
        best_distance = float(np.max(distances[positions]))
        allowed_loss = min(
            float(settings.hue_balance_distance_tolerance),
            max(1.0e-6, 0.15 * max(best_distance, 0.0)),
        )
        positions = positions[distances[positions] >= best_distance - allowed_loss]
    tie_breakers = tuple(component[positions] for component in arrays[int(distance_component_count):])
    return int(positions[_best_position(tie_breakers)])


def _worst_case_distances_to_selected(
        candidate_labs: np.ndarray,
        selected_candidates: np.ndarray,
        settings: PaletteSettings,
) -> np.ndarray:
    """Return each pool candidate's distance to each selected pool color."""

    if len(selected_candidates) == 0:
        return np.empty((candidate_labs.shape[1], 0), dtype=float)
    return _distances_to_labs(
        candidate_labs,
        candidate_labs[:, selected_candidates, :],
        settings,
    )


def _distances_to_labs(
        candidate_labs: np.ndarray,
        selected_labs: np.ndarray,
        settings: PaletteSettings,
) -> np.ndarray:
    # Collapse vision-condition distances to the configured worst-case or mean metric.
    differences = candidate_labs[:, :, None, :] - selected_labs[:, None, :, :]
    condition_distances = np.linalg.norm(differences, axis=3)
    if settings.use_worst_case_cvd_distance:
        return np.min(condition_distances, axis=0)
    return np.mean(condition_distances, axis=0)


def _hue_sector(hue: float, settings: PaletteSettings) -> int:
    return int(_hue_sectors(np.asarray([hue]), settings)[0])


def _hue_quality(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    total = float(np.sum(counts))
    if total <= 0.0 or len(counts) <= 1:
        return 1.0
    probabilities = counts / total
    hhi = float(np.sum(probabilities ** 2))
    minimum_hhi = 1.0 / len(counts)
    return float(np.clip(1.0 - (hhi - minimum_hhi) / (1.0 - minimum_hhi), 0.0, 1.0))


def _hue_quality_after_candidates(
        counts: np.ndarray,
        candidate_sectors: np.ndarray,
        settings: PaletteSettings,
) -> np.ndarray:
    counts = np.asarray(counts, dtype=int)
    qualities = np.empty(len(candidate_sectors), dtype=float)
    for sector in range(int(settings.hue_sector_count)):
        positions = np.flatnonzero(candidate_sectors == sector)
        if positions.size == 0:
            continue
        updated = counts.copy()
        updated[sector] += 1
        qualities[positions] = _hue_quality(updated)
    return qualities


def _hue_balanced_candidate_indices(
        candidate_indices: np.ndarray,
        candidate_hues: np.ndarray,
        selected_hues: np.ndarray,
        settings: PaletteSettings,
) -> np.ndarray:
    """Avoid overfilling a hue family while leaving useful slack for optimization."""

    candidate_indices = np.asarray(candidate_indices, dtype=int)
    if not candidate_indices.size:
        return candidate_indices
    sector_counts = _sector_counts(selected_hues, settings)
    target_occupancy = int(np.ceil((len(selected_hues) + 1) / int(settings.hue_sector_count)))
    maximum_occupancy = target_occupancy + int(settings.hue_sector_max_excess)
    candidate_sectors = _hue_sectors(candidate_hues[candidate_indices], settings)
    permitted = sector_counts[candidate_sectors] < maximum_occupancy
    return candidate_indices[permitted] if np.any(permitted) else candidate_indices


def _normalized_range(values: np.ndarray, lower: float, upper: float) -> float:
    values = np.asarray(values, dtype=float)
    if values.size < 2 or not np.isfinite(values).any():
        return 0.0
    span = max(float(upper) - float(lower), np.finfo(float).eps)
    return float(np.clip((np.nanmax(values) - np.nanmin(values)) / span, 0.0, 1.0))


def _range_after_candidates(
        candidates: np.ndarray,
        selected: np.ndarray,
        lower: float,
        upper: float,
) -> np.ndarray:
    candidates = np.asarray(candidates, dtype=float)
    selected = np.asarray(selected, dtype=float)
    if selected.size == 0:
        return np.zeros(len(candidates), dtype=float)
    current_min = float(np.min(selected))
    current_max = float(np.max(selected))
    span = max(float(upper) - float(lower), np.finfo(float).eps)
    proposed_min = np.minimum(current_min, candidates)
    proposed_max = np.maximum(current_max, candidates)
    return np.clip((proposed_max - proposed_min) / span, 0.0, 1.0)


class PaletteBuilder:
    """Build palettes through the legacy-compatible palette strategy."""

    def __init__(self, settings: PaletteSettings = DEFAULT_PALETTE_SETTINGS):
        """Initialize a palette builder with explicit generation settings."""
        self.settings = settings

    def build(
        self,
        labels: Sequence[str],
        *,
        displayed_stacks: Mapping[Any, pd.DataFrame] | Iterable[pd.DataFrame] | None = None,
        master_labels: Sequence[str] | None = None,
        master_displayed_stacks: Mapping[Any, pd.DataFrame] | Iterable[pd.DataFrame] | None = None,
    ) -> dict[str, str]:
        """Build a deterministic palette for the requested labels.

        Args:
            labels: Labels that must appear in the returned palette.
            displayed_stacks: Displayed count stacks used for optimization.
            master_labels: Stable label universe used to anchor colors.
            master_displayed_stacks: Displayed stacks for the stable universe.

        Returns:
            Mapping from each requested label to a hexadecimal color.
        """
        return build_taxon_palette(
            labels,
            displayed_stacks=displayed_stacks,
            settings=self.settings,
            master_labels=master_labels,
            master_displayed_stacks=master_displayed_stacks,
        )
