# Region/Country Band Decay

This project provides a Python workflow for country and global serotype
composition plots, Morisita-Horn lag decay, Bayesian exponential fits,
coverage sensitivity analysis, and fitted-curve figures.

## Project structure

- `src/band_decay/` contains the analysis package.
- `data/` contains the input data and fitted curve data. The examples expect
  `serotype_counts_country_ds_geodate2-2.tsv`, a tab-separated file with
  `country`, `collection_year`, `serotype`, and `count` columns. Generated plots
  are written to `plots/` and the exported R curve data stays in `data/`.
- `scripts/` contains the standalone Python and R workflows.
- `notebooks/` contains an interactive version of the analysis.
- `plots/` stores generated figures, while `tests/` contains automated tests.

## Scripts

- `plot_band_decay.py` runs the exponential curve fitting and creates the stack plots.
- `plot_fitted_country_curves.py` fits and plots one decay curve per country on the same plot.
- `update_fitted_country_curves_data.py` updates the R curve data file.
- `plot_fitted_country_curves.R` reads that file and creates the R curve plot.

## Installation

From the project directory, install all runtime, fitting, and notebook
dependencies:

```bash
python -m pip install -r requirements.txt
```

The package uses a `src` layout and does not provide an editable package
installation. Run commands from the project directory with `src` on
`PYTHONPATH`, or mark `src` as a source root in your IDE.

## Basic usage

```python
from pathlib import Path

from band_decay import AnalysisConfig, DecayAnalysis, InputConfig

config = AnalysisConfig(
    input=InputConfig(
        data_path=Path("data/serotype_counts_country_ds_geodate2-2.tsv"),
        countries=("Greece", "Italy"),
    )
)

result = DecayAnalysis(config).run(fit=False, output_path=Path("plots/decay.pdf"))
print(result.prepared.entity_order)
```

Use `PyMCDecayFitter` explicitly when a Bayesian fit is wanted. The notebook
`notebooks/band_decay.ipynb` provides an interactive configuration surface
over the same API; its widget implementation lives in a hidden notebook cell.

## Sensitivity and curves

```python
from band_decay import SensitivityConfig, SensitivityRunner

sensitivity = SensitivityRunner(
    config,
    SensitivityConfig(coverage_percentages=(80, 90, 95)),
).run(fit=True)
print(sensitivity.stability_summary)
```

Fitted curve plotting consumes posterior results produced by the analysis. The
`config` below is the one created in the basic usage example:

```python
from band_decay import CurvePlotConfig, DecayAnalysis, PosteriorMedian, render_fitted_curves

analysis = DecayAnalysis(config)
prepared = analysis.prepare()
decay_data = analysis.fit(prepared)

render_fitted_curves(
    decay_data,
    CurvePlotConfig(
        output_path="plots/fitted-curves.png",
        fit_summary=PosteriorMedian(),
    ),
)
```

`PosteriorMean()` and `PosteriorMedian()` can be selected independently on
`CurvePlotConfig.fit_summary`, `PlotConfig.fit_summary`, and the
`SensitivityConfig.fit_summary` / `SensitivityConfig.stability_summary`
settings.

## Scripts and notebook

Run the scripts from the project directory with:

```bash
PYTHONPATH=src python scripts/plot_band_decay.py
```

Or with:

```bash
PYTHONPATH=src python scripts/plot_fitted_country_curves.py
```

Update the data used by the R plot and then render it with:

```bash
PYTHONPATH=src python scripts/update_fitted_country_curves_data.py
Rscript scripts/plot_fitted_country_curves.R
```

The R script requires R and the `ggplot2` package. 

Start the notebook from the
project directory so its relative data path and `src` package path resolve:

```bash
PYTHONPATH="$PWD/src" jupyter notebook notebooks/band_decay.ipynb
```