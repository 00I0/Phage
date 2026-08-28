"""CVD-aware, abundance-tiered categorical palettes for stacked taxon plots.

The public build_taxon_palette function accepts taxon labels plus the displayed
stack data and returns one deterministic {taxon: color} mapping. The optimizer
is independent of any plotting layout. Plotting code only supplies labels and
data frames; all color generation, optimization, and diagnostics live here.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

OTHER_TAXON = "Other"
TRANSIENT_TAXON = "Transient"
_SPECIAL_TAXA = frozenset({OTHER_TAXON, TRANSIENT_TAXON})
_TIER_HERO = "hero"
_TIER_INTERMEDIATE = "intermediate"
_TIER_RARE = "rare"
_CVD_MATRICES = {
    "protanopia": np.asarray(
        [
            [0.152286, 1.052583, -0.204868],
            [0.114503, 0.786281, 0.099216],
            [-0.003882, -0.048116, 1.051998],
        ],
        dtype=float,
    ),
    "deuteranopia": np.asarray(
        [
            [0.367322, 0.860646, -0.227968],
            [0.280085, 0.672501, 0.047413],
            [-0.011820, 0.042940, 0.968881],
        ],
        dtype=float,
    ),
}


@dataclass(frozen=True)
class PaletteSettings:
    """Controls for the reusable hierarchical categorical palette optimizer."""

    lightness_min: float = 0.40
    lightness_max: float = 0.78
    chroma_min: float = 0.055
    chroma_max: float = 0.30
    hero_lightness_range: tuple[float, float] = (0.46, 0.69)
    hero_chroma_range: tuple[float, float] = (0.095, 0.28)
    intermediate_lightness_range: tuple[float, float] = (0.42, 0.73)
    intermediate_chroma_range: tuple[float, float] = (0.09, 0.24)
    rare_lightness_range: tuple[float, float] = (0.41, 0.77)
    rare_chroma_range: tuple[float, float] = (0.075, 0.20)
    hero_taxon_count: int = 10
    intermediate_taxon_count: int = 20
    abundance_tie_tolerance: float = 0.012
    tie_randomization_seed: int | None = None
    hue_sector_count: int = 10
    hue_balance_strength: float = 3.0
    top_hue_balance_strength: float = 3.0
    hue_balance_distance_tolerance: float = 0.010
    hue_sector_max_excess: int = 2
    top_minimum_distance_priority: float = 1.25
    top_to_other_minimum_distance_priority: float = 1.15
    global_minimum_distance_priority: float = 1.0
    global_repair_top_distance_loss: float = 0.04
    candidate_count: int = 3_000
    optimization_iterations: int = 2_500
    seed: int = 20_260_807
    adjacency_strength: float = 0.75
    adjacency_area_exponent: float = 0.5
    lightness_contrast_strength: float = 0.95
    minimum_distance: float = 0.045
    minimum_distance_weight: float = 3_000.0
    distance_cap: float = 0.30
    minimum_relative_luminance: float = 0.055
    minimum_background_contrast: float = 1.35
    cvd_conditions: tuple[str, ...] = ("normal",)  # ("normal", "protanopia", "deuteranopia")
    use_worst_case_cvd_distance: bool = True
    cache_enabled: bool = True
    other_color: str = "#d9d9d9"
    transient_color: str = "#6c757d"
    abundance_weight: float = 2.0
    abundance_exponent: float = 0.55
    top_priority_count: int | None = None
    top_priority_weight: float = 3.0
    adjacency_weight: float | None = None
    lightness_contrast_weight: float | None = None
    tie_tolerance: float | None = None
    tie_seed: int | None = None

    def __post_init__(self) -> None:
        default_hero_count = type(self).__dataclass_fields__["hero_taxon_count"].default
        if self.top_priority_count is not None:
            if int(self.hero_taxon_count) != int(default_hero_count) and int(self.hero_taxon_count) != int(
                    self.top_priority_count
            ):
                raise ValueError("hero_taxon_count and top_priority_count disagree.")
            resolved_hero_count = int(self.top_priority_count)
        else:
            resolved_hero_count = int(self.hero_taxon_count)
        if resolved_hero_count < 0:
            raise ValueError("hero_taxon_count must be nonnegative.")
        if int(self.intermediate_taxon_count) < resolved_hero_count:
            raise ValueError("intermediate_taxon_count must be at least hero_taxon_count.")
        object.__setattr__(self, "hero_taxon_count", resolved_hero_count)
        object.__setattr__(self, "top_priority_count", resolved_hero_count)

        resolved_adjacency_strength = (
            float(self.adjacency_weight)
            if self.adjacency_strength == type(self).__dataclass_fields__["adjacency_strength"].default
               and self.adjacency_weight is not None
            else float(self.adjacency_strength)
        )
        resolved_lightness_strength = (
            float(self.lightness_contrast_weight)
            if self.lightness_contrast_strength
               == type(self).__dataclass_fields__["lightness_contrast_strength"].default
               and self.lightness_contrast_weight is not None
            else float(self.lightness_contrast_strength)
        )
        object.__setattr__(self, "adjacency_strength", resolved_adjacency_strength)
        object.__setattr__(self, "adjacency_weight", resolved_adjacency_strength)
        object.__setattr__(self, "lightness_contrast_strength", resolved_lightness_strength)
        object.__setattr__(self, "lightness_contrast_weight", resolved_lightness_strength)

        resolved_tie_tolerance = (
            float(self.tie_tolerance) if self.tie_tolerance is not None else float(self.abundance_tie_tolerance)
        )
        resolved_tie_seed = (
            int(self.tie_seed)
            if self.tie_seed is not None
            else int(self.seed if self.tie_randomization_seed is None else self.tie_randomization_seed)
        )
        object.__setattr__(self, "abundance_tie_tolerance", resolved_tie_tolerance)
        object.__setattr__(self, "tie_tolerance", resolved_tie_tolerance)
        object.__setattr__(self, "tie_randomization_seed", resolved_tie_seed)
        object.__setattr__(self, "tie_seed", resolved_tie_seed)

        if not 0.0 < float(self.lightness_min) < float(self.lightness_max) < 1.0:
            raise ValueError("Palette lightness bounds must satisfy 0 < minimum < maximum < 1.")
        if not 0.0 <= float(self.chroma_min) < float(self.chroma_max):
            raise ValueError("Palette chroma bounds must satisfy 0 <= minimum < maximum.")
        for name, bounds, lower, upper in (
                ("hero_lightness_range", self.hero_lightness_range, 0.0, 1.0),
                ("intermediate_lightness_range", self.intermediate_lightness_range, 0.0, 1.0),
                ("rare_lightness_range", self.rare_lightness_range, 0.0, 1.0),
                ("hero_chroma_range", self.hero_chroma_range, 0.0, np.inf),
                ("intermediate_chroma_range", self.intermediate_chroma_range, 0.0, np.inf),
                ("rare_chroma_range", self.rare_chroma_range, 0.0, np.inf),
        ):
            if len(bounds) != 2 or not float(bounds[0]) < float(bounds[1]):
                raise ValueError(f"{name} must contain increasing lower and upper bounds.")
            if float(bounds[0]) < lower or float(bounds[1]) > upper:
                raise ValueError(f"{name} contains values outside its valid range.")
            object.__setattr__(self, name, (float(bounds[0]), float(bounds[1])))
        if int(self.candidate_count) < 2:
            raise ValueError("candidate_count must be at least 2.")
        if int(self.optimization_iterations) < 0:
            raise ValueError("optimization_iterations must be nonnegative.")
        if float(self.abundance_weight) < 0.0 or float(self.top_priority_weight) < 0.0:
            raise ValueError("Palette abundance weights must be nonnegative.")
        if float(self.abundance_exponent) <= 0.0 or float(self.adjacency_area_exponent) <= 0.0:
            raise ValueError("Palette weighting exponents must be positive.")
        if float(self.abundance_tie_tolerance) < 0.0:
            raise ValueError("abundance_tie_tolerance must be nonnegative.")
        if int(self.hue_sector_count) < 2:
            raise ValueError("hue_sector_count must be at least 2.")
        if int(self.hue_sector_max_excess) < 0:
            raise ValueError("hue_sector_max_excess must be nonnegative.")
        for name in (
                "hue_balance_strength",
                "top_hue_balance_strength",
                "top_minimum_distance_priority",
                "top_to_other_minimum_distance_priority",
                "global_minimum_distance_priority",
                "adjacency_strength",
                "lightness_contrast_strength",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be nonnegative.")
        if float(self.hue_balance_distance_tolerance) < 0.0:
            raise ValueError("hue_balance_distance_tolerance must be nonnegative.")
        if not 0.0 <= float(self.global_repair_top_distance_loss) < 1.0:
            raise ValueError("global_repair_top_distance_loss must be in [0, 1).")
        if float(self.minimum_distance) < 0.0 or float(self.minimum_distance_weight) < 0.0:
            raise ValueError("Palette minimum-distance settings must be nonnegative.")
        if float(self.distance_cap) <= 0.0:
            raise ValueError("distance_cap must be positive.")
        if not 0.0 <= float(self.minimum_relative_luminance) <= 1.0:
            raise ValueError("minimum_relative_luminance must be in [0, 1].")
        if float(self.minimum_background_contrast) < 1.0:
            raise ValueError("minimum_background_contrast must be at least 1.")
        invalid_conditions = set(self.cvd_conditions).difference({"normal", *_CVD_MATRICES})
        if invalid_conditions:
            raise ValueError(f"Unsupported cvd_conditions: {sorted(invalid_conditions)}")
        if "normal" not in self.cvd_conditions:
            raise ValueError("cvd_conditions must include normal vision.")
        if len(set(self.cvd_conditions)) != len(self.cvd_conditions):
            raise ValueError("cvd_conditions must not contain duplicates.")


DEFAULT_PALETTE_SETTINGS = PaletteSettings()
_PALETTE_CACHE: dict[tuple[tuple[str, ...], str, PaletteSettings], tuple[str, ...]] = {}


@dataclass(frozen=True)
class _CandidatePool:
    tier: str
    oklch: np.ndarray
    srgb: np.ndarray
    labs: np.ndarray
    hexes: tuple[str, ...]


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


def _materialize_stacks(
        displayed_stacks: Mapping[Any, pd.DataFrame] | Iterable[pd.DataFrame] | None,
) -> tuple[pd.DataFrame, ...] | None:
    if displayed_stacks is None:
        return None
    stacks = displayed_stacks.values() if isinstance(displayed_stacks, Mapping) else displayed_stacks
    return tuple(stacks)


def _labels_from_stacks(stacks: tuple[pd.DataFrame, ...] | None) -> tuple[str, ...]:
    if stacks is None:
        return tuple()
    labels: list[str] = []
    seen: set[str] = set()
    for stack in stacks:
        if not isinstance(stack, pd.DataFrame):
            raise TypeError("displayed_stacks values must be pandas DataFrames.")
        for label in stack.columns:
            string_label = str(label)
            if string_label not in seen:
                seen.add(string_label)
                labels.append(string_label)
    return tuple(labels)


def _stable_label_key(label: str, seed: int) -> bytes:
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(int(seed)).encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(label).encode("utf-8"))
    return digest.digest()


def _canonical_taxa(labels: Iterable[str], settings: PaletteSettings) -> tuple[str, ...]:
    return tuple(
        sorted(
            set(str(label) for label in labels),
            key=lambda label: _stable_label_key(label, int(settings.tie_randomization_seed)),
        )
    )


def _display_metrics(
        taxa: tuple[str, ...],
        displayed_labels: tuple[str, ...],
        displayed_stacks: Mapping[Any, pd.DataFrame] | Iterable[pd.DataFrame] | None,
        settings: PaletteSettings,
) -> tuple[np.ndarray, np.ndarray]:
    """Return total displayed abundance and normalized observed adjacency weights."""

    taxon_count = len(taxa)
    taxon_index = {taxon: index for index, taxon in enumerate(taxa)}
    abundance = np.zeros(taxon_count, dtype=float)
    adjacency = np.zeros((taxon_count, taxon_count), dtype=float)
    if displayed_stacks is None:
        return np.ones(taxon_count, dtype=float), adjacency

    stacks = displayed_stacks.values() if isinstance(displayed_stacks, Mapping) else displayed_stacks
    for stack in stacks:
        if not isinstance(stack, pd.DataFrame):
            raise TypeError("displayed_stacks values must be pandas DataFrames.")
        values = stack.reindex(columns=list(displayed_labels), fill_value=0.0).to_numpy(dtype=float)
        values = np.where(np.isfinite(values) & (values > 0.0), values, 0.0)
        for label_index, label in enumerate(displayed_labels):
            dynamic_index = taxon_index.get(label)
            if dynamic_index is not None:
                abundance[dynamic_index] += values[:, label_index].sum()
        for row_values in values:
            present_labels = np.flatnonzero(row_values > 0.0)
            if present_labels.size < 2:
                continue
            for left_label_index, right_label_index in zip(
                    present_labels[:-1],
                    present_labels[1:],
                    strict=True,
            ):
                left_taxon_index = taxon_index.get(displayed_labels[left_label_index])
                right_taxon_index = taxon_index.get(displayed_labels[right_label_index])
                if left_taxon_index is None or right_taxon_index is None:
                    continue
                boundary_area = np.sqrt(row_values[left_label_index] * row_values[right_label_index])
                boundary_weight = boundary_area ** float(settings.adjacency_area_exponent)
                adjacency[left_taxon_index, right_taxon_index] += boundary_weight
                adjacency[right_taxon_index, left_taxon_index] += boundary_weight

    maximum_adjacency = float(np.max(adjacency)) if adjacency.size else 0.0
    if maximum_adjacency > 0.0:
        adjacency /= maximum_adjacency
    if not np.any(abundance > 0.0):
        abundance[:] = 1.0
    return abundance, adjacency


def _metrics_digest(abundance: np.ndarray, adjacency: np.ndarray) -> str:
    """Create a compact cache key for the palette-driving displayed data."""

    digest = hashlib.blake2b(digest_size=16)
    digest.update(np.ascontiguousarray(abundance, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(adjacency, dtype=np.float64).tobytes())
    return digest.hexdigest()


def _tier_ranges(settings: PaletteSettings, tier: str) -> tuple[tuple[float, float], tuple[float, float]]:
    if tier == _TIER_HERO:
        lightness_range = settings.hero_lightness_range
        chroma_range = settings.hero_chroma_range
    elif tier == _TIER_INTERMEDIATE:
        lightness_range = settings.intermediate_lightness_range
        chroma_range = settings.intermediate_chroma_range
    elif tier == _TIER_RARE:
        lightness_range = settings.rare_lightness_range
        chroma_range = settings.rare_chroma_range
    else:
        raise ValueError(f"Unsupported palette tier: {tier!r}")
    bounded_lightness = (
        max(float(settings.lightness_min), float(lightness_range[0])),
        min(float(settings.lightness_max), float(lightness_range[1])),
    )
    bounded_chroma = (
        max(float(settings.chroma_min), float(chroma_range[0])),
        min(float(settings.chroma_max), float(chroma_range[1])),
    )
    if not bounded_lightness[0] < bounded_lightness[1]:
        raise ValueError(f"{tier} lightness range does not overlap the global bounds.")
    if not bounded_chroma[0] < bounded_chroma[1]:
        raise ValueError(f"{tier} chroma range does not overlap the global bounds.")
    return bounded_lightness, bounded_chroma


def _candidate_seed(settings: PaletteSettings, tier: str, pool_size: int) -> int:
    digest = hashlib.blake2b(digest_size=8)
    digest.update(str(int(settings.seed)).encode("utf-8"))
    digest.update(b"\0")
    digest.update(tier.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(int(pool_size)).encode("utf-8"))
    return int.from_bytes(digest.digest(), byteorder="little", signed=False) % (2 ** 63 - 1)


@lru_cache(maxsize=48)
def _candidate_pool(settings: PaletteSettings, tier: str, pool_size: int) -> _CandidatePool:
    """Generate a deterministic, stratified, usable in-gamut OKLCH pool."""

    lightness_range, chroma_range = _tier_ranges(settings, tier)
    generator = np.random.default_rng(_candidate_seed(settings, tier, pool_size))
    sector_count = int(settings.hue_sector_count)
    target_per_sector = np.full(sector_count, int(pool_size) // sector_count, dtype=int)
    target_per_sector[: int(pool_size) % sector_count] += 1
    collected_per_sector = np.zeros(sector_count, dtype=int)
    oklch_chunks: list[list[np.ndarray]] = [[] for _ in range(sector_count)]
    srgb_chunks: list[list[np.ndarray]] = [[] for _ in range(sector_count)]
    attempts = 0
    while np.any(collected_per_sector < target_per_sector) and attempts < 500:
        needed = int(np.sum(target_per_sector - collected_per_sector))
        batch_size = max(2_048, needed * 12)
        sector = generator.integers(0, sector_count, size=batch_size)
        hue = (sector + generator.random(batch_size)) * (360.0 / sector_count)
        lightness = generator.uniform(lightness_range[0], lightness_range[1], size=batch_size)
        chroma = generator.uniform(chroma_range[0], chroma_range[1], size=batch_size)
        oklab = _oklch_to_oklab(lightness, chroma, hue)
        srgb = _oklab_to_srgb(oklab)
        finite = np.all(np.isfinite(srgb), axis=1)
        in_gamut = np.all((srgb >= 0.0) & (srgb <= 1.0), axis=1)
        clipped_srgb = np.clip(srgb, 0.0, 1.0)
        luminance = _relative_luminance(clipped_srgb)
        contrast = 1.05 / (luminance + 0.05)
        usable = (
                finite
                & in_gamut
                & (luminance >= float(settings.minimum_relative_luminance))
                & (contrast >= float(settings.minimum_background_contrast))
        )
        candidate_oklch = np.stack([lightness, chroma, hue], axis=1)
        for sector_index in range(sector_count):
            remaining = int(target_per_sector[sector_index] - collected_per_sector[sector_index])
            if remaining <= 0:
                continue
            valid_indices = np.flatnonzero(usable & (sector == sector_index))
            if not valid_indices.size:
                continue
            chosen_indices = valid_indices[:remaining]
            oklch_chunks[sector_index].append(candidate_oklch[chosen_indices])
            srgb_chunks[sector_index].append(srgb[chosen_indices])
            collected_per_sector[sector_index] += int(chosen_indices.size)
        attempts += 1
    if np.any(collected_per_sector < target_per_sector):
        missing = np.flatnonzero(collected_per_sector < target_per_sector).tolist()
        raise RuntimeError(
            f"Unable to generate balanced {tier} palette candidates for hue sectors {missing}."
        )
    oklch = np.concatenate([np.concatenate(chunks, axis=0) for chunks in oklch_chunks], axis=0)
    srgb = np.concatenate([np.concatenate(chunks, axis=0) for chunks in srgb_chunks], axis=0)
    ordering = generator.permutation(int(pool_size))
    oklch = oklch[ordering]
    srgb = srgb[ordering]
    labs = _candidate_labs_for_conditions(srgb, settings.cvd_conditions)
    hexes = tuple(_srgb_to_hex(color) for color in srgb)
    return _CandidatePool(tier=tier, oklch=oklch, srgb=srgb, labs=labs, hexes=hexes)


@lru_cache(maxsize=16)
def _candidate_srgb_pool(settings: PaletteSettings) -> np.ndarray:
    """Return a compatibility view of the tiered candidate pool."""

    pool_size = max(int(settings.candidate_count), int(settings.hero_taxon_count) + 8)
    pools = [
        _candidate_pool(settings, tier, pool_size)
        for tier in (_TIER_HERO, _TIER_INTERMEDIATE, _TIER_RARE)
    ]
    return np.concatenate([pool.srgb for pool in pools], axis=0)[: int(settings.candidate_count)]


def _optimize_palette(
        taxa: tuple[str, ...],
        abundance: np.ndarray,
        adjacency: np.ndarray,
        settings: PaletteSettings,
) -> tuple[str, ...]:
    """Build the palette in protected hero, intermediate, and rare stages."""

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
    hero_slots = _select_hero_candidates(pools[_TIER_HERO], int(settings.hero_taxon_count), settings)

    condition_count = pools[_TIER_HERO].labs.shape[0]
    selected_srgb = np.zeros((len(taxa), 3), dtype=float)
    selected_oklch = np.zeros((len(taxa), 3), dtype=float)
    selected_labs = np.zeros((condition_count, len(taxa), 3), dtype=float)
    selected_tiers: list[str | None] = [None] * len(taxa)
    selected_pool_indices = np.full(len(taxa), -1, dtype=int)
    used_by_tier = {tier: set() for tier in pools}

    for rank, taxon_index in enumerate(taxon_order[:hero_count]):
        candidate_index = int(hero_slots[rank])
        _set_selected_candidate(
            taxon_index,
            _TIER_HERO,
            candidate_index,
            pools,
            selected_srgb,
            selected_oklch,
            selected_labs,
            selected_tiers,
            selected_pool_indices,
            used_by_tier,
        )

    for rank, taxon_index in enumerate(taxon_order[hero_count:], start=hero_count):
        tier = _tier_for_rank(rank, settings)
        assigned_indices = taxon_order[:rank]
        candidate_index = _choose_candidate_for_taxon(
            taxon_index=taxon_index,
            tier=tier,
            pools=pools,
            assigned_indices=assigned_indices,
            top_indices=taxon_order[:hero_count],
            selected_oklch=selected_oklch,
            selected_labs=selected_labs,
            selected_tiers=selected_tiers,
            selected_pool_indices=selected_pool_indices,
            used_by_tier=used_by_tier,
            abundance=abundance,
            adjacency=adjacency,
            settings=settings,
        )
        _set_selected_candidate(
            taxon_index,
            tier,
            candidate_index,
            pools,
            selected_srgb,
            selected_oklch,
            selected_labs,
            selected_tiers,
            selected_pool_indices,
            used_by_tier,
        )

    _refine_nonhero_assignments(
        taxon_order=taxon_order,
        hero_count=hero_count,
        pools=pools,
        selected_srgb=selected_srgb,
        selected_oklch=selected_oklch,
        selected_labs=selected_labs,
        selected_tiers=selected_tiers,
        selected_pool_indices=selected_pool_indices,
        used_by_tier=used_by_tier,
        adjacency=adjacency,
        settings=settings,
    )
    return tuple(_srgb_to_hex(color) for color in selected_srgb)


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
        pool: _CandidatePool,
        target_count: int,
        settings: PaletteSettings,
) -> np.ndarray:
    """Select a stable hero set by maximin distance, then balance its hue."""

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
        pool: _CandidatePool,
        initial: np.ndarray,
        settings: PaletteSettings,
) -> np.ndarray:
    if len(initial) < 2 or int(settings.optimization_iterations) <= 0:
        return initial
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


def _hero_key(pool: _CandidatePool, selected: np.ndarray, settings: PaletteSettings) -> tuple[float, ...]:
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
        pools: Mapping[str, _CandidatePool],
        assigned_indices: np.ndarray,
        top_indices: np.ndarray,
        selected_oklch: np.ndarray,
        selected_labs: np.ndarray,
        selected_tiers: list[str | None],
        selected_pool_indices: np.ndarray,
        used_by_tier: Mapping[str, set[int]],
        abundance: np.ndarray,
        adjacency: np.ndarray,
        settings: PaletteSettings,
) -> int:
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
        else np.zeros(available.size)
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
        pools: Mapping[str, _CandidatePool],
        selected_srgb: np.ndarray,
        selected_oklch: np.ndarray,
        selected_labs: np.ndarray,
        selected_tiers: list[str | None],
        selected_pool_indices: np.ndarray,
        used_by_tier: Mapping[str, set[int]],
) -> None:
    pool = pools[tier]
    selected_srgb[taxon_index] = pool.srgb[candidate_index]
    selected_oklch[taxon_index] = pool.oklch[candidate_index]
    selected_labs[:, taxon_index, :] = pool.labs[:, candidate_index, :]
    selected_tiers[taxon_index] = tier
    selected_pool_indices[taxon_index] = int(candidate_index)
    used_by_tier[tier].add(int(candidate_index))


def _refine_nonhero_assignments(
        *,
        taxon_order: np.ndarray,
        hero_count: int,
        pools: Mapping[str, _CandidatePool],
        selected_srgb: np.ndarray,
        selected_oklch: np.ndarray,
        selected_labs: np.ndarray,
        selected_tiers: list[str | None],
        selected_pool_indices: np.ndarray,
        used_by_tier: Mapping[str, set[int]],
        adjacency: np.ndarray,
        settings: PaletteSettings,
) -> None:
    if len(taxon_order) <= hero_count or int(settings.optimization_iterations) <= 0:
        return
    generator = np.random.default_rng(int(settings.seed) + 211)
    top_indices = taxon_order[:hero_count]
    current_key = _palette_hierarchy_key(
        selected_oklch,
        selected_labs,
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
        tier = selected_tiers[taxon_index]
        if tier is None:
            continue
        pool = pools[tier]
        current_candidate = int(selected_pool_indices[taxon_index])
        used_by_tier[tier].discard(current_candidate)
        available = np.asarray(
            [index for index in range(pool.srgb.shape[0]) if index not in used_by_tier[tier]],
            dtype=int,
        )
        if available.size == 0:
            used_by_tier[tier].add(current_candidate)
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
                selected_oklch=selected_oklch,
                selected_labs=selected_labs,
                selected_pool_indices=selected_pool_indices,
                used_by_tier=used_by_tier,
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
                selected_oklch=selected_oklch,
                selected_labs=selected_labs,
                selected_tiers=selected_tiers,
                selected_pool_indices=selected_pool_indices,
                used_by_tier=used_by_tier,
                abundance=np.zeros(len(taxon_order), dtype=float),
                adjacency=adjacency,
                settings=settings,
            )
            proposed_tier = tier
        proposal_pool = pools[proposed_tier]
        previous_srgb = selected_srgb[taxon_index].copy()
        previous_oklch = selected_oklch[taxon_index].copy()
        previous_labs = selected_labs[:, taxon_index, :].copy()
        selected_srgb[taxon_index] = proposal_pool.srgb[proposed_candidate]
        selected_oklch[taxon_index] = proposal_pool.oklch[proposed_candidate]
        selected_labs[:, taxon_index, :] = proposal_pool.labs[:, proposed_candidate, :]
        proposed_key = _palette_hierarchy_key(
            selected_oklch,
            selected_labs,
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
            used_by_tier[tier].add(proposed_candidate)
            if proposed_tier != tier:
                used_by_tier[tier].discard(proposed_candidate)
                used_by_tier[proposed_tier].add(proposed_candidate)
            selected_pool_indices[taxon_index] = proposed_candidate
            selected_tiers[taxon_index] = proposed_tier
            current_key = proposed_key
        else:
            selected_srgb[taxon_index] = previous_srgb
            selected_oklch[taxon_index] = previous_oklch
            selected_labs[:, taxon_index, :] = previous_labs
            used_by_tier[tier].add(current_candidate)


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
        pools: Mapping[str, _CandidatePool],
        top_indices: np.ndarray,
        selected_oklch: np.ndarray,
        selected_labs: np.ndarray,
        selected_pool_indices: np.ndarray,
        used_by_tier: Mapping[str, set[int]],
        minimum_top_distance: float,
        adjacency: np.ndarray,
        settings: PaletteSettings,
) -> tuple[str, int] | None:
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


def _pair_importance(
        taxa: tuple[str, ...],
        abundance: np.ndarray,
        settings: PaletteSettings,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the legacy abundance-pair view for callers using old helpers."""

    maximum_abundance = float(np.max(abundance)) if len(abundance) else 0.0
    normalized_abundance = (
        abundance / maximum_abundance if maximum_abundance > 0.0 else np.ones_like(abundance)
    )
    priority = 1.0 + float(settings.abundance_weight) * normalized_abundance ** float(settings.abundance_exponent)
    taxon_order, _ = _abundance_order(taxa, abundance, settings)
    top_count = min(int(settings.hero_taxon_count), len(taxa))
    for rank, taxon_index in enumerate(taxon_order[:top_count]):
        tier_weight = (top_count - rank) / max(1, top_count)
        priority[taxon_index] += float(settings.top_priority_weight) * tier_weight
    return np.maximum.outer(priority, priority), taxon_order


def _farthest_point_assignment(
        candidate_labs: np.ndarray,
        taxon_order: np.ndarray,
        pair_importance: np.ndarray,
        adjacency: np.ndarray,
        settings: PaletteSettings,
) -> np.ndarray:
    """Compatibility helper implementing CVD-aware farthest-point assignment."""

    assignment = np.full(len(taxon_order), -1, dtype=int)
    available = np.ones(candidate_labs.shape[1], dtype=bool)
    for order_index, taxon_index in enumerate(taxon_order):
        candidates = np.flatnonzero(available)
        if order_index == 0:
            chosen = int(candidates[0])
        else:
            distances = _worst_case_distances_to_selected(
                candidate_labs,
                assignment[taxon_order[:order_index]],
                settings,
            )
            chosen = int(candidates[np.argmax(np.min(distances[candidates], axis=1))])
        assignment[taxon_index] = chosen
        available[chosen] = False
    return assignment


def _anneal_assignment(
        candidate_labs: np.ndarray,
        initial_assignment: np.ndarray,
        pair_importance: np.ndarray,
        adjacency: np.ndarray,
        settings: PaletteSettings,
) -> np.ndarray:
    """Compatibility wrapper retaining the former helper name."""

    if len(initial_assignment) < 2 or int(settings.optimization_iterations) == 0:
        return initial_assignment
    generator = np.random.default_rng(int(settings.seed) + 1)
    current = initial_assignment.copy()
    current_score = _assignment_score(candidate_labs, current, pair_importance, adjacency, settings)
    best = current.copy()
    best_score = current_score
    for _ in range(int(settings.optimization_iterations)):
        proposed = current.copy()
        left, right = generator.choice(len(proposed), size=2, replace=False)
        proposed[left], proposed[right] = proposed[right], proposed[left]
        score = _assignment_score(candidate_labs, proposed, pair_importance, adjacency, settings)
        if score > current_score:
            current, current_score = proposed, score
            if score > best_score:
                best, best_score = proposed.copy(), score
    return best


def _assignment_score(
        candidate_labs: np.ndarray,
        assignment: np.ndarray,
        pair_importance: np.ndarray,
        adjacency: np.ndarray,
        settings: PaletteSettings,
) -> float:
    """Return a legacy scalar score without driving the staged optimizer."""

    distances = _pairwise_distances(candidate_labs[:, assignment, :], settings)
    upper_triangle = np.triu_indices(len(assignment), k=1)
    pair_distances = distances[upper_triangle]
    pair_weights = pair_importance[upper_triangle]
    adjacency_weights = adjacency[upper_triangle]
    capped_distances = np.minimum(pair_distances, float(settings.distance_cap))
    score = float(np.sum(pair_weights * capped_distances))
    score += float(settings.adjacency_strength) * float(
        np.sum(adjacency_weights * capped_distances)
    )
    return score - float(settings.minimum_distance_weight) * float(
        np.sum(np.maximum(0.0, float(settings.minimum_distance) - pair_distances) ** 2)
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
    differences = candidate_labs[:, :, None, :] - selected_labs[:, None, :, :]
    condition_distances = np.linalg.norm(differences, axis=3)
    if settings.use_worst_case_cvd_distance:
        return np.min(condition_distances, axis=0)
    return np.mean(condition_distances, axis=0)


def _pairwise_distances(candidate_labs: np.ndarray, settings: PaletteSettings) -> np.ndarray:
    differences = candidate_labs[:, :, None, :] - candidate_labs[:, None, :, :]
    condition_distances = np.linalg.norm(differences, axis=3)
    if settings.use_worst_case_cvd_distance:
        return np.min(condition_distances, axis=0)
    return np.mean(condition_distances, axis=0)


def _sector_counts(hues: np.ndarray, settings: PaletteSettings) -> np.ndarray:
    sectors = _hue_sectors(hues, settings)
    return np.bincount(sectors, minlength=int(settings.hue_sector_count)).astype(int)


def _hue_sectors(hues: np.ndarray, settings: PaletteSettings) -> np.ndarray:
    values = np.mod(np.asarray(hues, dtype=float), 360.0)
    return np.floor(values / (360.0 / int(settings.hue_sector_count))).astype(int)


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


def _candidate_labs_for_conditions(candidate_srgb: np.ndarray, conditions: tuple[str, ...]) -> np.ndarray:
    """Return OKLab coordinates for normal and simulated color-vision states."""

    labs = []
    ordered_conditions = ("normal", *(condition for condition in conditions if condition != "normal"))
    for condition in ordered_conditions:
        if condition == "normal":
            condition_srgb = candidate_srgb
        else:
            condition_srgb = _simulate_cvd_srgb(candidate_srgb, condition)
        labs.append(_srgb_to_oklab(condition_srgb))
    return np.stack(labs, axis=0)


def _simulate_cvd_srgb(srgb: np.ndarray, condition: str) -> np.ndarray:
    """Simulate full protanopia or deuteranopia in linear sRGB."""

    transformed = _srgb_to_linear(srgb) @ _CVD_MATRICES[condition].T
    return _linear_to_srgb(np.clip(transformed, 0.0, 1.0))


def _oklch_to_oklab(lightness: np.ndarray, chroma: np.ndarray, hue_degrees: np.ndarray) -> np.ndarray:
    hue_radians = np.deg2rad(hue_degrees)
    return np.stack(
        [lightness, chroma * np.cos(hue_radians), chroma * np.sin(hue_radians)],
        axis=-1,
    )


def _oklab_to_oklch(oklab: np.ndarray) -> np.ndarray:
    oklab = np.asarray(oklab, dtype=float)
    return np.stack(
        [
            oklab[..., 0],
            np.hypot(oklab[..., 1], oklab[..., 2]),
            np.mod(np.rad2deg(np.arctan2(oklab[..., 2], oklab[..., 1])), 360.0),
        ],
        axis=-1,
    )


def _oklab_to_srgb(oklab: np.ndarray) -> np.ndarray:
    lightness = oklab[..., 0]
    axis_a = oklab[..., 1]
    axis_b = oklab[..., 2]
    l_prime = lightness + 0.3963377774 * axis_a + 0.2158037573 * axis_b
    m_prime = lightness - 0.1055613458 * axis_a - 0.0638541728 * axis_b
    s_prime = lightness - 0.0894841775 * axis_a - 1.2914855480 * axis_b
    lms = np.stack([l_prime ** 3, m_prime ** 3, s_prime ** 3], axis=-1)
    linear_srgb = np.stack(
        [
            4.0767416621 * lms[..., 0] - 3.3077115913 * lms[..., 1] + 0.2309699292 * lms[..., 2],
            -1.2684380046 * lms[..., 0] + 2.6097574011 * lms[..., 1] - 0.3413193965 * lms[..., 2],
            -0.0041960863 * lms[..., 0] - 0.7034186147 * lms[..., 1] + 1.7076147010 * lms[..., 2],
        ],
        axis=-1,
    )
    return _linear_to_srgb(linear_srgb)


def _srgb_to_oklab(srgb: np.ndarray) -> np.ndarray:
    linear_srgb = _srgb_to_linear(np.asarray(srgb, dtype=float))
    lms = np.stack(
        [
            0.4122214708 * linear_srgb[..., 0]
            + 0.5363325363 * linear_srgb[..., 1]
            + 0.0514459929 * linear_srgb[..., 2],
            0.2119034982 * linear_srgb[..., 0]
            + 0.6806995451 * linear_srgb[..., 1]
            + 0.1073969566 * linear_srgb[..., 2],
            0.0883024619 * linear_srgb[..., 0]
            + 0.2817188376 * linear_srgb[..., 1]
            + 0.6299787005 * linear_srgb[..., 2],
        ],
        axis=-1,
    )
    lms_root = np.cbrt(np.clip(lms, 0.0, None))
    return np.stack(
        [
            0.2104542553 * lms_root[..., 0]
            + 0.7936177850 * lms_root[..., 1]
            - 0.0040720468 * lms_root[..., 2],
            1.9779984951 * lms_root[..., 0]
            - 2.4285922050 * lms_root[..., 1]
            + 0.4505937099 * lms_root[..., 2],
            0.0259040371 * lms_root[..., 0]
            + 0.7827717662 * lms_root[..., 1]
            - 0.8086757660 * lms_root[..., 2],
        ],
        axis=-1,
    )


def _srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    srgb = np.asarray(srgb, dtype=float)
    linear_srgb = np.empty_like(srgb, dtype=float)
    low_values = srgb <= 0.04045
    linear_srgb[low_values] = srgb[low_values] / 12.92
    linear_srgb[~low_values] = ((srgb[~low_values] + 0.055) / 1.055) ** 2.4
    return linear_srgb


def _linear_to_srgb(linear_srgb: np.ndarray) -> np.ndarray:
    linear_srgb = np.asarray(linear_srgb, dtype=float)
    srgb = np.empty_like(linear_srgb, dtype=float)
    low_values = linear_srgb <= 0.0031308
    srgb[low_values] = 12.92 * linear_srgb[low_values]
    srgb[~low_values] = 1.055 * np.maximum(linear_srgb[~low_values], 0.0) ** (1.0 / 2.4) - 0.055
    return srgb


def _relative_luminance(srgb: np.ndarray) -> np.ndarray:
    linear = _srgb_to_linear(np.clip(np.asarray(srgb, dtype=float), 0.0, 1.0))
    return (
            0.2126 * linear[..., 0]
            + 0.7152 * linear[..., 1]
            + 0.0722 * linear[..., 2]
    )


def _srgb_to_hex(srgb: np.ndarray) -> str:
    rgb_uint8 = np.rint(np.clip(srgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    return f"#{rgb_uint8[0]:02x}{rgb_uint8[1]:02x}{rgb_uint8[2]:02x}"


def _hex_to_srgb(color: str) -> np.ndarray:
    value = str(color).strip().lstrip("#")
    if len(value) == 3:
        value = "".join(character * 2 for character in value)
    if len(value) != 6:
        raise ValueError(f"Expected a six-digit hexadecimal color, got {color!r}.")
    try:
        return np.asarray([int(value[index: index + 2], 16) / 255.0 for index in (0, 2, 4)], dtype=float)
    except ValueError as error:
        raise ValueError(f"Invalid hexadecimal color {color!r}.") from error


def palette_diagnostics(
        palette: Mapping[str, str],
        *,
        displayed_stacks: Mapping[Any, pd.DataFrame] | Iterable[pd.DataFrame] | None = None,
        abundance: Mapping[str, float] | None = None,
        settings: PaletteSettings = DEFAULT_PALETTE_SETTINGS,
        closest_pair_count: int = 8,
) -> dict[str, Any]:
    """Return tuning diagnostics for a finished palette.

    Distances use the configured worst-case normal/protanopia/deuteranopia
    metric. Closest-pair entries include abundance rank and observed adjacency.
    """

    dynamic_taxa = tuple(str(label) for label in palette if str(label) not in _SPECIAL_TAXA)
    if not dynamic_taxa:
        return {
            "top_10_min_distance": 0.0,
            "top_10_minimum_distance": 0.0,
            "top_20_min_distance": 0.0,
            "top_20_minimum_distance": 0.0,
            "top_20_to_other_min_distance": 0.0,
            "top_20_to_other_minimum_distance": 0.0,
            "global_min_distance": 0.0,
            "global_minimum_distance": 0.0,
            "top_20_hue_sector_occupancy": {},
            "hue_sector_occupancy_top_20": {},
            "full_hue_sector_occupancy": {},
            "hue_sector_occupancy_full": {},
            "lightness_range": (0.0, 0.0),
            "chroma_range": (0.0, 0.0),
            "closest_pairs": [],
        }
    stack_tuple = _materialize_stacks(displayed_stacks)
    metric_labels = tuple(dict.fromkeys((*dynamic_taxa, *_labels_from_stacks(stack_tuple))))
    if abundance is None:
        abundance_values, adjacency = _display_metrics(
            dynamic_taxa,
            metric_labels,
            stack_tuple,
            settings,
        )
    else:
        abundance_values = np.asarray([float(abundance.get(taxon, 0.0)) for taxon in dynamic_taxa], dtype=float)
        _, adjacency = _display_metrics(dynamic_taxa, metric_labels, stack_tuple, settings)
    srgb = np.stack([_hex_to_srgb(palette[taxon]) for taxon in dynamic_taxa], axis=0)
    labs = _candidate_labs_for_conditions(srgb, settings.cvd_conditions)
    distances = _pairwise_distances(labs, settings)
    oklch = _oklab_to_oklch(_srgb_to_oklab(srgb))
    taxon_order, _ = _abundance_order(dynamic_taxa, abundance_values, settings)
    abundance_rank = {int(taxon_index): rank + 1 for rank, taxon_index in enumerate(taxon_order)}
    top_10 = taxon_order[: min(10, len(taxon_order))]
    top_20 = taxon_order[: min(20, len(taxon_order))]
    other = np.asarray([index for index in range(len(dynamic_taxa)) if index not in set(top_20.tolist())], dtype=int)
    top_10_minimum = _minimum_distance_for_indices(distances, top_10)
    top_20_minimum = _minimum_distance_for_indices(distances, top_20)
    if top_20.size and other.size:
        top_20_to_other = float(np.min(distances[np.ix_(top_20, other)]))
    else:
        top_20_to_other = 0.0
    global_minimum = _minimum_distance_for_indices(distances, np.arange(len(dynamic_taxa), dtype=int))
    closest_pairs = _closest_pairs(
        dynamic_taxa,
        distances,
        adjacency,
        abundance_rank,
        int(closest_pair_count),
    )
    top_hue_sector_occupancy = _occupancy_dict(oklch[top_20, 2], settings)
    full_hue_sector_occupancy = _occupancy_dict(oklch[:, 2], settings)
    diagnostics = {
        "top_10_min_distance": top_10_minimum,
        "top_10_minimum_distance": top_10_minimum,
        "top_20_min_distance": top_20_minimum,
        "top_20_minimum_distance": top_20_minimum,
        "top_20_to_other_min_distance": top_20_to_other,
        "top_20_to_other_minimum_distance": top_20_to_other,
        "global_min_distance": global_minimum,
        "global_minimum_distance": global_minimum,
        "top_20_hue_sector_occupancy": top_hue_sector_occupancy,
        "hue_sector_occupancy_top_20": top_hue_sector_occupancy,
        "full_hue_sector_occupancy": full_hue_sector_occupancy,
        "hue_sector_occupancy_full": full_hue_sector_occupancy,
        "lightness_range": (float(np.min(oklch[:, 0])), float(np.max(oklch[:, 0]))),
        "chroma_range": (float(np.min(oklch[:, 1])), float(np.max(oklch[:, 1]))),
        "closest_pairs": closest_pairs,
    }
    diagnostics["used_lightness_range"] = diagnostics["lightness_range"]
    diagnostics["used_chroma_range"] = diagnostics["chroma_range"]
    return diagnostics


def diagnose_palette(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Alias for palette_diagnostics."""

    return palette_diagnostics(*args, **kwargs)


def _minimum_distance_for_indices(distances: np.ndarray, indices: np.ndarray) -> float:
    indices = np.asarray(indices, dtype=int)
    if indices.size < 2:
        return 0.0
    values = distances[np.ix_(indices, indices)][np.triu_indices(indices.size, k=1)]
    return float(np.min(values)) if values.size else 0.0


def _occupancy_dict(hues: np.ndarray, settings: PaletteSettings) -> dict[int, int]:
    counts = _sector_counts(hues, settings)
    return {sector: int(count) for sector, count in enumerate(counts)}


def _closest_pairs(
        taxa: tuple[str, ...],
        distances: np.ndarray,
        adjacency: np.ndarray,
        abundance_rank: Mapping[int, int],
        pair_count: int,
) -> list[dict[str, Any]]:
    upper_triangle = np.triu_indices(len(taxa), k=1)
    order = np.argsort(distances[upper_triangle], kind="stable")
    result: list[dict[str, Any]] = []
    for flat_index in order[: max(0, int(pair_count))]:
        left_index = int(upper_triangle[0][flat_index])
        right_index = int(upper_triangle[1][flat_index])
        result.append(
            {
                "left_taxon": taxa[left_index],
                "right_taxon": taxa[right_index],
                "distance": float(distances[left_index, right_index]),
                "left_rank": int(abundance_rank[left_index]),
                "right_rank": int(abundance_rank[right_index]),
                "adjacent": bool(adjacency[left_index, right_index] > 0.0),
            }
        )
    return result


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

