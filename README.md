# Region/Country Band Decay

This project provides a Python workflow for country and global serotype
composition plots, Morisita-Horn lag decay, Bayesian exponential fits,
coverage sensitivity analysis, and fitted-curve figures.

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
over the same API; its widget implementation lives in a hidden notebook cell
instead of the analysis package.

## Sensitivity and curves

```python
from band_decay import SensitivityConfig, SensitivityRunner

sensitivity = SensitivityRunner(
    config,
    SensitivityConfig(coverage_percentages=(80, 90, 95)),
).run(fit=True)
print(sensitivity.stability_summary)
```

Fitted curve plotting consumes posterior results produced by the analysis:

```python
from band_decay import CurvePlotConfig, PosteriorMedian, render_fitted_curves

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

Run the coverage sensitivity workflow from the project directory with:

```bash
PYTHONPATH=src python scripts/plot_band_decay.py
```

Run the country-only posterior curve plot with:

```bash
PYTHONPATH=src python scripts/plot_fitted_country_curves.py
```

Choose the curve scale by changing the `display` policy in `main()` in
`scripts/plot_fitted_country_curves.py` to either `NormalizedDecay()` or
`OriginalDecay()`.

Choose the curve summary by changing `fit_summary` in the same call between
`PosteriorMean()` and `PosteriorMedian()`.

The same `main()` configuration can opt into `CentralConfidenceInterval()`
and `DashedExtrapolation()` in place of `NoConfidenceInterval()` and
`NoExtrapolation()`.

Open `notebooks/band_decay.ipynb` in Jupyter after installing the dependencies.
The notebook uses `data/serotype_counts_country_ds_geodate2-2.tsv` by default.
Edit the configuration values in the notebook or in `main()` in the script
before running an analysis.
