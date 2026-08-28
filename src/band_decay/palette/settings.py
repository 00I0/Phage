"""Configuration for palette generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


OTHER_TAXON = "Other"
TRANSIENT_TAXON = "Transient"
_SPECIAL_TAXA = frozenset({OTHER_TAXON, TRANSIENT_TAXON})
_TIER_HERO = "hero"
_TIER_INTERMEDIATE = "intermediate"
_TIER_RARE = "rare"


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
    minimum_relative_luminance: float = 0.055
    minimum_background_contrast: float = 1.35
    cvd_conditions: tuple[str, ...] = ("normal",)
    use_worst_case_cvd_distance: bool = True
    cache_enabled: bool = True
    other_color: str = "#d9d9d9"
    transient_color: str = "#6c757d"

    def __post_init__(self) -> None:
        # Normalize dependent values before validating the complete configuration.
        resolved_hero_count = int(self.hero_taxon_count)
        if resolved_hero_count < 0:
            raise ValueError("hero_taxon_count must be nonnegative.")
        if int(self.intermediate_taxon_count) < resolved_hero_count:
            raise ValueError("intermediate_taxon_count must be at least hero_taxon_count.")
        object.__setattr__(self, "hero_taxon_count", resolved_hero_count)

        resolved_tie_seed = (
            int(self.seed)
            if self.tie_randomization_seed is None
            else int(self.tie_randomization_seed)
        )
        object.__setattr__(self, "tie_randomization_seed", resolved_tie_seed)

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
        if float(self.hue_balance_distance_tolerance) < 0.0:
            raise ValueError("hue_balance_distance_tolerance must be nonnegative.")
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
        if not 0.0 <= float(self.global_repair_top_distance_loss) < 1.0:
            raise ValueError("global_repair_top_distance_loss must be in [0, 1).")
        if not 0.0 <= float(self.minimum_relative_luminance) <= 1.0:
            raise ValueError("minimum_relative_luminance must be in [0, 1].")
        if float(self.minimum_background_contrast) < 1.0:
            raise ValueError("minimum_background_contrast must be at least 1.")
        invalid_conditions = set(self.cvd_conditions).difference(
            {"normal", "protanopia", "deuteranopia"}
        )
        if invalid_conditions:
            raise ValueError(f"Unsupported cvd_conditions: {sorted(invalid_conditions)}")
        if "normal" not in self.cvd_conditions:
            raise ValueError("cvd_conditions must include normal vision.")
        if len(set(self.cvd_conditions)) != len(self.cvd_conditions):
            raise ValueError("cvd_conditions must not contain duplicates.")


DEFAULT_PALETTE_SETTINGS = PaletteSettings()
