"""Deterministic, gamut-safe palette candidate generation."""

from __future__ import annotations

import hashlib
from functools import lru_cache

import numpy as np

from .colors import (
    _candidate_labs_for_conditions,
    _oklch_to_oklab,
    _oklab_to_srgb,
    _relative_luminance,
    _srgb_to_hex,
)
from .settings import (
    _TIER_HERO,
    _TIER_INTERMEDIATE,
    _TIER_RARE,
    PaletteSettings,
)
from .types import CandidatePool


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
    # Intersect tier bounds with the global search limits before sampling.
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
def _candidate_pool(settings: PaletteSettings, tier: str, pool_size: int) -> CandidatePool:
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
    # Fill every hue sector before shuffling to keep candidate coverage balanced.
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
    return CandidatePool(tier=tier, oklch=oklch, srgb=srgb, labs=labs, hexes=hexes)
