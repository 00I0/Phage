from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .config import CurvePlotConfig
from .domain import CurveParameters


DEFAULT_MEAN_CURVES: dict[str, CurveParameters] = {
    "GLOBAL": CurveParameters(0.90, 0.10, 0.12),
    "Greece": CurveParameters(0.9402, 0.0768, 0.2225),
    "Italy": CurveParameters(0.8127, 0.1142, 0.2335),
    "Spain": CurveParameters(0.9290, 0.1110, 0.1658),
    "Russia": CurveParameters(0.8634, 0.4945, 0.6264),
    "United Kingdom": CurveParameters(0.6350, 0.4352, 0.0903),
    "France": CurveParameters(0.6857, 0.3618, 0.1143),
    "Germany": CurveParameters(0.6907, 0.1434, 0.1327),
    "Switzerland": CurveParameters(0.7419, 0.3946, 0.1136),
}


class CurveSet:
    """Normalize fitted-curve mappings into a predictable value object."""

    def __init__(self, curves: Mapping[str, CurveParameters | tuple[float, float, float]]):
        """Create a curve set from parameter objects or ``(y0, b, c)`` tuples."""
        self._curves = {
            str(entity): value if isinstance(value, CurveParameters) else CurveParameters(*map(float, value))
            for entity, value in curves.items()
        }

    @property
    def curves(self) -> Mapping[str, CurveParameters]:
        """Return a copy of the entity-to-parameters mapping."""
        return dict(self._curves)

    def ordered(self, entity_order: tuple[str, ...] | None = None) -> tuple[tuple[str, CurveParameters], ...]:
        """Return curves in a requested or insertion-preserving order."""
        if entity_order is None:
            entities = tuple(self._curves)
        else:
            entities = tuple(entity_order)
        missing = [entity for entity in entities if entity not in self._curves]
        if missing:
            raise ValueError(f"Missing fitted curves for entities: {missing}")
        return tuple((entity, self._curves[entity]) for entity in entities)


class FittedCurvesRenderer:
    """Render standalone fitted MH decay curves."""

    def render(self, curves: CurveSet | Mapping[str, CurveParameters | tuple[float, float, float]], config: CurvePlotConfig, *, entity_order: tuple[str, ...] | None = None) -> Path:
        """Render fitted curves and save the configured image.

        Args:
            curves: Curve set or entity-to-parameter mapping.
            config: Curve rendering configuration.
            entity_order: Optional plotting order.

        Returns:
            Path to the saved curve figure.
        """
        curve_set = curves if isinstance(curves, CurveSet) else CurveSet(curves)
        ordered = curve_set.ordered(entity_order)
        x_values = np.linspace(0.0, float(config.horizon_years), int(config.point_count))
        colors = plt.get_cmap("tab10").colors
        figure, axis = plt.subplots(figsize=(10.0, 6.0), dpi=int(config.dpi))
        for index, (entity, parameters) in enumerate(ordered):
            curve = config.display.parameter_curve(parameters, x_values)
            label = "Aggregate" if entity == "GLOBAL" else entity
            axis.plot(x_values, curve, color=colors[index % len(colors)], linewidth=2.0, label=label)
        axis.set_xlabel("Year lag")
        axis.set_ylabel(config.display.y_label())
        axis.set_title("Fitted MH exponential decay curves")
        axis.set_xlim(0.0, float(config.horizon_years))
        axis.set_ylim(bottom=0.0)
        axis.grid(alpha=0.2, linewidth=0.6)
        axis.legend(frameon=False)
        figure.tight_layout()
        output_path = Path(config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=int(config.dpi), bbox_inches="tight")
        if config.show:
            plt.show()
        else:
            plt.close(figure)
        return output_path


def render_fitted_curves(
    curves: CurveSet | Mapping[str, CurveParameters | tuple[float, float, float]],
    config: CurvePlotConfig = CurvePlotConfig(),
    *,
    entity_order: tuple[str, ...] | None = None,
) -> Path:
    """Render standalone fitted curves through the default renderer."""
    return FittedCurvesRenderer().render(curves, config, entity_order=entity_order)
