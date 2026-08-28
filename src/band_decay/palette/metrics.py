"""Displayed-stack metrics and deterministic taxon ordering."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

from .settings import PaletteSettings


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

    # Convert positive stack values into abundance and adjacent-boundary weights.
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
