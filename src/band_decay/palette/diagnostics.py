"""Diagnostics for generated categorical palettes."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from .colors import (
    _candidate_labs_for_conditions,
    _hex_to_srgb,
    _oklab_to_oklch,
    _srgb_to_oklab,
)
from .metrics import (
    _abundance_order,
    _display_metrics,
    _labels_from_stacks,
    _materialize_stacks,
    _pairwise_distances,
    _sector_counts,
)
from .settings import _SPECIAL_TAXA, DEFAULT_PALETTE_SETTINGS, PaletteSettings


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

    # Recompute the optimizer's perceptual metrics for inspection and reporting.
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
    # Flatten the upper triangle so each unordered pair is reported once.
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
