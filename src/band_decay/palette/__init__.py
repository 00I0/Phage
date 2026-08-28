"""CVD-aware, abundance-tiered categorical palettes."""

from .diagnostics import diagnose_palette, palette_diagnostics
from .optimizer import PaletteBuilder, build_taxon_palette
from .settings import DEFAULT_PALETTE_SETTINGS, PaletteSettings

__all__ = [
    "DEFAULT_PALETTE_SETTINGS",
    "PaletteBuilder",
    "PaletteSettings",
    "build_taxon_palette",
    "diagnose_palette",
    "palette_diagnostics",
]
