from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from .config import CurvePlotConfig
from .domain import EntityDecayData


class FittedCurvesRenderer:
    """Render summarized MH decay curves from fitted entity data."""

    def render(
        self,
        decay_data: Mapping[str, EntityDecayData],
        config: CurvePlotConfig,
        *,
        entity_order: tuple[str, ...] | None = None,
    ) -> Path:
        """Render pointwise summarized curves from fitted decay data."""
        entities = tuple(entity_order) if entity_order is not None else tuple(decay_data)
        missing = [entity for entity in entities if entity not in decay_data]
        if missing:
            raise ValueError(f"Missing decay data for entities: {missing}")
        x_values = np.linspace(0.0, float(config.horizon_years), int(config.point_count))
        colors = plt.get_cmap("tab10").colors
        figure, axis = plt.subplots(figsize=(10.0, 6.0), dpi=int(config.dpi))
        plotted = 0
        fitted_entity_count = sum(decay_data[entity].fit is not None for entity in entities)
        interval_alpha = min(
            config.confidence_interval.fill_alpha(),
            0.5 / max(1, fitted_entity_count),
        )
        for entity in entities:
            fit = decay_data[entity].fit
            if fit is None:
                continue
            label = "Aggregate" if entity == "GLOBAL" else entity
            color = colors[plotted % len(colors)]
            fitted_x, extrapolated_x = config.extrapolation.split(decay_data[entity], x_values)
            axis.plot(
                fitted_x,
                config.fit_summary.curve(config.display, fit, fitted_x),
                color=color,
                linewidth=2.0,
                label=label,
                zorder=3,
            )
            if len(extrapolated_x):
                axis.plot(
                    extrapolated_x,
                    config.fit_summary.curve(config.display, fit, extrapolated_x),
                    color=color,
                    linewidth=2.0,
                    linestyle="--",
                    label="_nolegend_",
                    zorder=3,
                )
            bounds = config.confidence_interval.bounds(config.display, fit, x_values)
            if bounds is not None:
                lower, upper = bounds
                axis.fill_between(
                    x_values,
                    lower,
                    upper,
                    color=color,
                    alpha=interval_alpha,
                    linewidth=0.0,
                    label="_nolegend_",
                    zorder=1,
                )
            plotted += 1
        if not plotted:
            raise ValueError("No fitted decay curves are available to plot.")
        axis.set_xlabel("Year lag")
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.set_ylabel(config.display.y_label())
        axis.set_title(f"Posterior-{config.fit_summary.label()} MH exponential decay curves")
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
    decay_data: Mapping[str, EntityDecayData],
    config: CurvePlotConfig,
    *,
    entity_order: tuple[str, ...] | None = None,
) -> Path:
    """Render summarized curves from fitted entity decay data."""
    return FittedCurvesRenderer().render(decay_data, config, entity_order=entity_order)
