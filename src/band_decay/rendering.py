from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch, Rectangle
from matplotlib.textpath import TextPath
from matplotlib.ticker import FuncFormatter

from .config import AnalysisConfig
from .constants import GLOBAL_ENTITY
from .domain import EntityDecayData, EntityYearSelection, PreparedData
from .policies import DecayDisplayPolicy, PosteriorSummaryPolicy
from .preparation import counts_to_props


def _strike_text(text: str) -> str:
    return "".join(f"{character}\u0336" for character in text)


def masked_display_years(selection: EntityYearSelection) -> frozenset[int]:
    """Return displayed years marked as excluded or missing."""
    return selection.excluded_years.union(selection.missing_years)


def _mask_marked_years(axis, years: frozenset[int], *, height: float, config) -> None:
    for year in sorted(years):
        axis.add_patch(
            Rectangle(
                (float(year) - 0.5, 0.0),
                1.0,
                height,
                facecolor="#b0b0b0",
                edgecolor="#666666",
                hatch=config.excluded_year_hatch,
                alpha=float(config.excluded_year_alpha),
                linewidth=0.0,
                zorder=6.0,
            )
        )


def _plot_band_pair(
    relative_axis,
    absolute_axis,
    counts,
    labels: tuple[str, ...],
    palette: Mapping[str, str],
    *,
    title: str,
    absolute_ylim: float,
    show_titles: bool,
    show_y_labels: bool,
    show_x_labels: bool,
    row_label: str | None,
    year_selection: EntityYearSelection,
    config,
) -> None:
    # Render relative and absolute views from the same grouped counts.
    years = counts.index.to_numpy(dtype=int)
    marked_years = masked_display_years(year_selection)
    if not len(years):
        for axis in (relative_axis, absolute_axis):
            axis.set_xlim(0.0, 1.0)
            axis.set_ylim(0.0, 1.0)
            axis.set_xticks([])
            axis.grid(axis="y", alpha=0.2, linewidth=0.6)
            axis.text(0.5, 0.5, "No data", transform=axis.transAxes, ha="center", va="center", color="#555555")
        if show_titles:
            relative_axis.set_title(f"{title} - Relative")
            absolute_axis.set_title(f"{title} - Absolute")
        relative_axis.set_ylabel("Share" if show_y_labels else "")
        absolute_axis.set_ylabel("Count" if show_y_labels else "")
        if row_label:
            _add_row_label(relative_axis, row_label)
        return

    totals = counts.sum(axis=1).to_numpy(dtype=float)
    proportions = counts_to_props(counts)
    colors = [palette[label] for label in labels]
    absolute_ymax = absolute_ylim * 1.04 if absolute_ylim > 0 else 1.0
    relative_axis.stackplot(
        years,
        [proportions[label].to_numpy(dtype=float) for label in labels],
        colors=colors,
        baseline="zero",
        edgecolor="white",
        linewidth=0.4,
        alpha=1.0,
    )
    relative_axis.set_ylim(0.0, 1.0)
    if show_titles:
        relative_axis.set_title(f"{title} - Relative")
    relative_axis.set_ylabel("Share" if show_y_labels else "")
    relative_axis.set_yticks(np.linspace(0.0, 1.0, 5))
    relative_axis.set_yticklabels([f"{int(value * 100)}%" for value in np.linspace(0.0, 1.0, 5)])
    relative_axis.grid(axis="y", alpha=0.2, linewidth=0.6)

    absolute_axis.stackplot(
        years,
        [counts[label].to_numpy(dtype=float) for label in labels],
        colors=colors,
        baseline="zero",
        edgecolor="white",
        linewidth=0.4,
        alpha=1.0,
    )
    absolute_axis.set_ylim(0.0, absolute_ymax)
    if show_titles:
        absolute_axis.set_title(f"{title} - Absolute")
    absolute_axis.set_ylabel("Count" if show_y_labels else "")
    absolute_axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(round(value)):,}"))
    absolute_axis.yaxis.offsetText.set_visible(False)
    absolute_axis.grid(axis="y", alpha=0.2, linewidth=0.6)

    if len(years) <= int(config.count_label_max_years):
        cutoff = float(config.count_label_max_y_fraction) * absolute_ymax
        offset = 0.015 * absolute_ymax
        for year, total in zip(years, totals, strict=True):
            if total <= 0 or total > cutoff:
                continue
            absolute_axis.text(int(year), float(total) + offset, f"{int(round(total)):,}", ha="center", va="bottom", fontsize=6, color="#333333", clip_on=True)

    for axis in (relative_axis, absolute_axis):
        axis.set_xticks(years)
        tick_labels = []
        for year in years:
            label = str(int(year))
            if config.strike_excluded_year_labels and int(year) in marked_years:
                label = _strike_text(label)
            tick_labels.append(label)
        axis.set_xticklabels(tick_labels)
        axis.set_xlim(float(np.min(years)) - 0.5, float(np.max(years)) + 0.5)
        axis.set_xlabel("Collection year" if show_x_labels else "")
        for tick in axis.get_xticklabels():
            tick.set_rotation(45)
            tick.set_ha("right")
    _mask_marked_years(relative_axis, marked_years, height=1.0, config=config)
    _mask_marked_years(absolute_axis, marked_years, height=absolute_ymax, config=config)
    if row_label:
        _add_row_label(relative_axis, row_label)


def _add_row_label(axis, row_label: str) -> None:
    axis.text(-0.16, 0.5, row_label, transform=axis.transAxes, ha="right", va="center", rotation=90, fontsize=9, color="#333333")


def _plot_decay(
    axis,
    decay: EntityDecayData,
    *,
    title: str,
    show_title: bool,
    show_y_label: bool,
    show_x_label: bool,
    display_policy: DecayDisplayPolicy,
    summary_policy: PosteriorSummaryPolicy,
) -> None:
    fit = decay.fit
    axis.scatter(decay.x, decay.y, s=18, alpha=0.65, color="#222222", linewidths=0.0, label="Lag-pair MH similarity")
    xmax = max(1.0, float(np.max(decay.x))) if len(decay.x) else 1.0
    axis.set_xlim(0.0, xmax * 1.04)
    axis.set_ylim(0.0, 1.02)
    if show_title:
        axis.set_title(f"{title} - MH decay")
    axis.set_xlabel("Year lag" if show_x_label else "")
    axis.set_ylabel(display_policy.y_label() if show_y_label else "")
    axis.grid(alpha=0.2, linewidth=0.6)
    if fit is None:
        axis.text(0.5, 0.5, "Not fit\n< 3 lag pairs", transform=axis.transAxes, ha="center", va="center", fontsize=9, color="#555555")
        return
    query = np.linspace(0.0, xmax, 200)
    summary_curve = summary_policy.curve(display_policy, fit, query)
    lower, upper = display_policy.interval(fit, query)
    observed_max = float(np.max(decay.y)) if len(decay.y) else 1.0
    observed_min = float(np.min(decay.y)) if len(decay.y) else 0.0
    upper_limit = max(1.02, observed_max, float(np.nanmax(upper)))
    lower_limit = min(0.0, observed_min, float(np.nanmin(lower)))
    axis.set_ylim(lower_limit, upper_limit * 1.05 if upper_limit > 0 else 1.0)
    y0, b, asymptote_draws = display_policy.posterior_parameters(fit)
    asymptote = summary_policy.scalar(asymptote_draws)
    axis.fill_between(query, lower, upper, color="#4c78a8", alpha=0.22, linewidth=0.0, label="95% posterior band")
    axis.plot(query, summary_curve, color="#1f4e79", linewidth=1.8, label=f"{summary_policy.label().capitalize()} fit")
    text = f"y0={summary_policy.scalar(y0):.2f}, b={summary_policy.scalar(b):.2f}, {display_policy.parameter_label()}={asymptote:.2f}"
    if fit.sigma_samples is not None:
        text += f", sigma={summary_policy.scalar(fit.sigma_samples):.2f}"
    axis.text(0.03, 0.05, text, transform=axis.transAxes, ha="left", va="bottom", fontsize=8, color="#333333")


def active_legend_labels(prepared: PreparedData) -> tuple[str, ...]:
    """Return display labels with nonzero abundance in any entity."""
    return tuple(
        label for label in prepared.master_display_columns
        if any(float(prepared.entities[entity].grouped_display_counts[label].sum()) > 0 for entity in prepared.entity_order)
    )


def _legend_column_width(labels: tuple[str, ...], font_size: float) -> float:
    if not labels:
        return 0.7
    properties = FontProperties(size=font_size)
    widest = max(TextPath((0.0, 0.0), label, prop=properties).get_extents().width / 72.0 for label in labels)
    return widest + 0.7


def _legend_columns(labels: tuple[str, ...], available_width: float, font_size: float) -> int:
    return 1 if not labels else max(1, min(len(labels), int(available_width // _legend_column_width(labels, font_size))))


def _parameter_rows(
    prepared: PreparedData,
    decay_data: Mapping[str, EntityDecayData],
    include_sigma: bool,
    include_c: bool,
    summary_policy: PosteriorSummaryPolicy,
) -> list[list[str]]:
    rows = []
    for index, entity in enumerate(prepared.entity_order):
        data = prepared.entities[entity]
        label = "European aggregate" if index == 0 and entity == GLOBAL_ENTITY else entity
        fit = decay_data[entity].fit
        if fit is None:
            row = [label, f"{int(round(data.analysis_total)):,}", "not fit", ""]
            if include_c:
                row.append("")
            if include_sigma:
                row.append("")
        else:
            row = [
                label,
                f"{int(round(data.analysis_total)):,}",
                f"{summary_policy.scalar(fit.y0_samples):.2f}",
                f"{summary_policy.scalar(fit.b_samples):.2f}",
            ]
            if include_c:
                row.append(f"{summary_policy.scalar(fit.c_samples):.2f}")
            if include_sigma:
                row.append(f"{summary_policy.scalar(fit.sigma_samples):.2f}" if fit.sigma_samples is not None else "")
        rows.append(row)
    return rows


class AnalysisFigureRenderer:
    """Render the combined abundance and MH-decay figure."""

    def render(self, prepared: PreparedData, decay_data: Mapping[str, EntityDecayData], config: AnalysisConfig, output_path: Path | None = None) -> Path | None:
        """Render prepared data and optional fits to a figure.

        Args:
            prepared: Prepared matrices, groupings, and palette.
            decay_data: Observed and fitted decay data by entity.
            config: Analysis and plot configuration.
            output_path: Optional path for saving the figure.

        Returns:
            The saved path, or ``None`` when no path was requested.
        """
        nrows = len(prepared.entity_order)
        height_ratios = [2.4] + [1.0] * (nrows - 1)
        legend_labels = active_legend_labels(prepared)[:max(0, int(config.plot.max_legend_labels))]
        handles = [Patch(facecolor=prepared.palette[label], edgecolor="none", label=label) for label in legend_labels]
        table_width = 6.0
        footer_gap = 0.35
        font_size = 8.0
        minimum_legend_width = _legend_column_width(legend_labels, font_size)
        figure_width = max(18.0, (table_width + footer_gap + minimum_legend_width) / 0.92)
        footer_width = figure_width * 0.92
        legend_width = footer_width - table_width - footer_gap
        legend_columns = _legend_columns(legend_labels, legend_width, font_size)
        legend_rows = int(math.ceil(len(legend_labels) / legend_columns)) if legend_labels else 0
        legend_height = 0.34 + 0.24 * max(1, legend_rows)
        table_height = max(1.45, 0.19 * (nrows + 1) + 0.2)
        footer_height = max(legend_height, table_height)
        footer_bottom = 0.16
        top_gap = 0.15
        main_height = 4.2 + 1.65 * max(1, nrows - 1)
        figure_height = main_height + footer_height + footer_bottom + top_gap
        footer_left = (figure_width - footer_width) / 2.0
        table_left = footer_left + legend_width + footer_gap
        figure, axes = plt.subplots(
            nrows=nrows,
            ncols=3,
            figsize=(figure_width, figure_height),
            gridspec_kw={"height_ratios": height_ratios, "width_ratios": [1.35, 1.35, 1.5]},
            constrained_layout=False,
            dpi=int(config.plot.dpi),
        )
        if nrows == 1:
            axes = np.asarray([axes])
        absolute_candidates = [float(prepared.entities[entity].grouped_display_counts.sum(axis=1).max()) for entity in prepared.entity_order if len(prepared.entities[entity].grouped_display_counts)]
        absolute_ylim = max(absolute_candidates) if absolute_candidates else 1.0
        for row_index, entity in enumerate(prepared.entity_order):
            data = prepared.entities[entity]
            main_row = row_index == 0
            last_row = row_index == nrows - 1
            title = "GLOBAL aggregate" if row_index == 0 and entity == GLOBAL_ENTITY else entity
            _plot_band_pair(
                axes[row_index, 0], axes[row_index, 1], data.grouped_display_counts, prepared.master_display_columns,
                prepared.palette, title=title, absolute_ylim=absolute_ylim, show_titles=main_row,
                show_y_labels=main_row, show_x_labels=last_row, row_label=None if main_row else entity,
                year_selection=data.year_selection, config=config.plot,
            )
            _plot_decay(
                axes[row_index, 2], decay_data[entity], title=title, show_title=main_row,
                show_y_label=main_row, show_x_label=last_row, display_policy=config.plot.decay_display,
                summary_policy=config.plot.fit_summary,
            )
        minimum_year_label = str(config.year_selection.min_count_per_year)
        if config.year_selection.per_country_min_count_per_year:
            minimum_year_label += " (country overrides)"
        grouping_label = config.top_n.selection.describe(config.top_n.n)
        figure.suptitle(f"Whitelisted countries + selected-years global aggregate (min/year={minimum_year_label}) + Grouping {grouping_label}", fontsize=16, y=0.995)
        footer_bottom_fraction = footer_bottom / figure_height
        footer_height_fraction = footer_height / figure_height
        plot_bottom = (footer_bottom + footer_height + top_gap) / figure_height
        figure.tight_layout(rect=(0.04, plot_bottom, 1.0, 0.98))
        if handles:
            legend_axis = figure.add_axes([footer_left / figure_width, footer_bottom_fraction, legend_width / figure_width, footer_height_fraction])
            legend_axis.axis("off")
            legend_axis.legend(handles=handles, loc="center", ncol=legend_columns, frameon=False, fontsize=font_size, title="Serotype groups")
        table_axis = figure.add_axes([(table_left) / figure_width, footer_bottom_fraction, table_width / figure_width, footer_height_fraction])
        table_axis.axis("off")
        include_sigma = any(data.fit is not None and data.fit.sigma_samples is not None for data in decay_data.values())
        include_c = (
            config.plot.decay_display.includes_direct_parameter()
            or config.sampling.priors.asymptote.includes_direct_parameter()
        )
        column_labels = ["Country", "n", "y0", "b"]
        if include_c:
            column_labels.append("c")
        if include_sigma:
            column_labels.append("sigma")
        table = table_axis.table(
            cellText=_parameter_rows(prepared, decay_data, include_sigma, include_c, config.plot.fit_summary),
            colLabels=column_labels, loc="center", cellLoc="center", colLoc="center",
            colWidths=[0.20] * len(column_labels),
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7.0)
        table.scale(1.0, 0.92)
        for (row, column), cell in table.get_celld().items():
            cell.set_linewidth(0.3)
            cell.set_edgecolor("#cccccc")
            if row == 0:
                cell.set_facecolor("#f2f2f2")
                cell.set_text_props(weight="bold")
            cell.set_text_props(ha="center")
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(output_path, dpi=int(config.plot.dpi), bbox_inches="tight")
        backend = str(plt.get_backend()).lower()
        if config.plot.show and "agg" not in backend:
            plt.show()
        else:
            plt.close(figure)
        return output_path


def output_path_for_config(config: AnalysisConfig) -> Path:
    """Resolve the configured output filename for an analysis."""
    filename = config.output.filename_template.format(top_n=config.top_n.n, top_n_scope=config.top_n.selection.describe(config.top_n.n))
    return config.output.output_directory / filename


def render_figure(prepared: PreparedData, decay_data: Mapping[str, EntityDecayData], config: AnalysisConfig, output_path: Path | None = None) -> Path | None:
    """Render a combined figure through the default renderer."""
    return AnalysisFigureRenderer().render(prepared, decay_data, config, output_path)
