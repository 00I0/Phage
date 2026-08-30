#!/usr/bin/env Rscript

INTERVAL_ALPHA <- 0.0625
COUNTRY_COLORS <- c(
  Greece = "#1f77b4",
  Italy = "#ff7f0e",
  Spain = "#2ca02c",
  Russia = "#d62728",
  `United Kingdom` = "#9467bd",
  France = "#8c564b",
  Germany = "#e377c2",
  Switzerland = "#7f7f7f"
)

load_curve_data <- function(data_file) {
  if (!file.exists(data_file)) {
    stop(sprintf("Curve data file not found: %s", data_file))
  }
  data_environment <- new.env(parent = baseenv())
  sys.source(data_file, envir = data_environment)
  if (!exists("curve_data", envir = data_environment, inherits = FALSE)) {
    stop("Curve data file must define curve_data.")
  }
  data_environment$curve_data
}

curve_data <- load_curve_data("data/fitted_country_curves_data.R")
HORIZON_YEARS <- curve_data$horizon_years
lag_years <- curve_data$lag_years
country_curves <- curve_data$country_curves
output_path <- "plots/fitted_country_curves_r.png"

if (!requireNamespace("ggplot2", quietly = TRUE)) {
  stop("The ggplot2 package is required to render this plot.")
}

library(ggplot2)

validate_curve_data <- function(curves, x_values) {
  required_fields <- c(
    "extrapolation_start",
    "median_at_extrapolation_start",
    "median",
    "lower",
    "upper"
  )

  for (country_name in names(curves)) {
    curve <- curves[[country_name]]
    if (!all(required_fields %in% names(curve))) {
      stop(sprintf("Missing plotting constants for %s.", country_name))
    }
    if (!all(vapply(curve[required_fields], is.numeric, logical(1)))) {
      stop(sprintf("Non-numeric plotting constants for %s.", country_name))
    }
    if (any(vapply(curve[c("median", "lower", "upper")], length, integer(1)) != length(x_values))) {
      stop(sprintf("Curve lengths do not match the lag grid for %s.", country_name))
    }
    if (curve$extrapolation_start < min(x_values) || curve$extrapolation_start > max(x_values)) {
      stop(sprintf("Invalid extrapolation start for %s.", country_name))
    }
  }
}

split_median_curve <- function(curve, x_values) {
  fitted_indices <- x_values <= curve$extrapolation_start
  extrapolated_indices <- x_values > curve$extrapolation_start

  fitted_x <- x_values[fitted_indices]
  fitted_y <- curve$median[fitted_indices]
  extrapolated_x <- x_values[extrapolated_indices]
  extrapolated_y <- curve$median[extrapolated_indices]

  if (
    length(fitted_x) > 0 &&
      length(extrapolated_x) > 0 &&
      tail(fitted_x, 1) < curve$extrapolation_start
  ) {
    fitted_x <- c(fitted_x, curve$extrapolation_start)
    fitted_y <- c(fitted_y, curve$median_at_extrapolation_start)
    extrapolated_x <- c(curve$extrapolation_start, extrapolated_x)
    extrapolated_y <- c(curve$median_at_extrapolation_start, extrapolated_y)
  }

  list(
    fitted_x = fitted_x,
    fitted_y = fitted_y,
    extrapolated_x = extrapolated_x,
    extrapolated_y = extrapolated_y
  )
}

build_plot_data <- function(curves, x_values) {
  country_names <- names(curves)
  band_data <- do.call(rbind, lapply(country_names, function(country_name) {
    curve <- curves[[country_name]]
    data.frame(
      country = country_name,
      lag = x_values,
      median = curve$median,
      lower = curve$lower,
      upper = curve$upper,
      stringsAsFactors = FALSE
    )
  }))

  line_data <- do.call(rbind, lapply(country_names, function(country_name) {
    segments <- split_median_curve(curves[[country_name]], x_values)
    fitted_data <- data.frame(
      country = country_name,
      lag = segments$fitted_x,
      median = segments$fitted_y,
      section = "Fitted",
      stringsAsFactors = FALSE
    )
    if (length(segments$extrapolated_x) == 0) {
      return(fitted_data)
    }
    rbind(
      fitted_data,
      data.frame(
        country = country_name,
        lag = segments$extrapolated_x,
        median = segments$extrapolated_y,
        section = "Extrapolated",
        stringsAsFactors = FALSE
      )
    )
  }))

  list(band = band_data, lines = line_data)
}

render_curve_plot <- function(curves, x_values, output_file, show_confidence_intervals = TRUE) {
  if (
    !is.logical(show_confidence_intervals) ||
      length(show_confidence_intervals) != 1L ||
      is.na(show_confidence_intervals)
  ) {
    stop("show_confidence_intervals must be a single TRUE or FALSE value.")
  }
  validate_curve_data(curves, x_values)
  plot_data <- build_plot_data(curves, x_values)
  country_names <- names(curves)
  country_colors <- unname(COUNTRY_COLORS[country_names])
  names(country_colors) <- country_names
  maximum_y <- max(plot_data$band$upper)

  output_directory <- dirname(output_file)
  if (!dir.exists(output_directory)) {
    dir.create(output_directory, recursive = TRUE)
  }

  plot <- ggplot(plot_data$band, aes(x = lag, y = median, color = country))
  if (show_confidence_intervals) {
    plot <- plot + geom_ribbon(
      aes(ymin = lower, ymax = upper, fill = country),
      alpha = INTERVAL_ALPHA,
      color = NA
    )
  }
  plot <- plot + geom_line(
      data = plot_data$lines,
      aes(linetype = section, group = interaction(country, section)),
      linewidth = 0.9
    ) +
    scale_color_manual(values = country_colors, breaks = country_names) +
    scale_fill_manual(values = country_colors, guide = "none") +
    scale_linetype_manual(
      values = c(Fitted = "solid", Extrapolated = "dashed"),
      guide = "none"
    ) +
    scale_x_continuous(breaks = seq(0, HORIZON_YEARS, by = 2), expand = c(0, 0)) +
    scale_y_continuous(breaks = seq(0, 1, by = 0.2), expand = c(0, 0)) +
    coord_cartesian(ylim = c(0, maximum_y * 1.05), xlim = c(0, HORIZON_YEARS), expand = FALSE) +
    labs(
      title = "MH exponential decay curves",
      x = "Year lag",
      y = expression("Morisita-Horn similarity (" * y[0] * "-normalized fit)"),
      color = NULL
    ) +
    theme_minimal(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold", hjust = 0, margin = margin(b = 10)),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(color = "#d9dee5", linewidth = 0.35),
      axis.title = element_text(face = "bold"),
      legend.position = "bottom",
      legend.direction = "horizontal",
      legend.box = "vertical",
      legend.key.width = grid::unit(1.4, "lines")
    ) +
    guides(color = guide_legend(nrow = 2, byrow = TRUE, override.aes = list(linewidth = 1)))

  ggsave(output_file, plot = plot, width = 10, height = 6, units = "in", dpi = 300, bg = "white")
}

render_curve_plot(country_curves, lag_years, output_path, show_confidence_intervals = FALSE)
