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
over the same API.

## Sensitivity and curves

```python
from band_decay import SensitivityConfig, SensitivityRunner

sensitivity = SensitivityRunner(
    config,
    SensitivityConfig(coverage_percentages=(80, 90, 95)),
).run(fit=True)
print(sensitivity.stability_summary)
```

Standalone fitted curves accept either `CurveParameters` objects or
`(y0, b, c)` tuples:

```python
from band_decay import CurvePlotConfig, render_fitted_curves

render_fitted_curves(
    {"GLOBAL": (0.9, 0.1, 0.12), "Greece": (0.94, 0.08, 0.22)},
    CurvePlotConfig(output_path="plots/fitted-curves.png"),
)
```

## Scripts and notebook

Run the coverage sensitivity workflow from the project directory with:

```bash
PYTHONPATH=src python scripts/plot_band_decay.py
```

Open `notebooks/band_decay.ipynb` in Jupyter after installing the dependencies.
The notebook uses `data/serotype_counts_country_ds_geodate2-2.tsv` by default.
Edit the configuration values in the notebook or in `main()` in the script
before running an analysis.
