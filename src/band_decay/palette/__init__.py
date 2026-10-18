"""CVD-aware, abundance-tiered categorical palettes."""

from .diagnostics import diagnose_palette, palette_diagnostics
from .loader import load_palette
from .optimizer import (
    PaletteBuilder,
    PredefinedPaletteBuilder,
    build_predefined_palette,
    build_taxon_palette,
    reconcile_palette,
)
from .settings import DEFAULT_PALETTE_SETTINGS, PaletteSettings

__all__ = [
    "DEFAULT_PALETTE_SETTINGS",
    "PaletteBuilder",
    "PaletteSettings",
    "PredefinedPaletteBuilder",
    "build_predefined_palette",
    "build_taxon_palette",
    "diagnose_palette",
    "load_palette",
    "palette_diagnostics",
    "reconcile_palette",
]
