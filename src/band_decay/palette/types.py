"""Internal data structures for palette generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CandidatePool:
    tier: str
    oklch: np.ndarray
    srgb: np.ndarray
    labs: np.ndarray
    hexes: tuple[str, ...]


@dataclass
class PaletteState:
    srgb: np.ndarray
    oklch: np.ndarray
    labs: np.ndarray
    tiers: list[str | None]
    pool_indices: np.ndarray
    used_by_tier: dict[str, set[int]]
