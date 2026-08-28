from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .config import AnalysisConfig, SamplingConfig
from .domain import DecayFit, EntityDecayData, PreparedData
from .priors import BuiltNoise, DecayPriorConfig
from .similarity import entity_pair_data

LOGGER = logging.getLogger(__name__)


class DecayFitter(Protocol):
    """Interface for fitting one MH decay curve."""

    def fit(self, x: np.ndarray, y: np.ndarray, sampling: SamplingConfig, seed: int) -> DecayFit | None:
        """Fit a decay curve from lag values and similarities."""
        ...


class NoOpFitter:
    """Fitter implementation that deliberately skips model fitting."""

    def fit(self, x: np.ndarray, y: np.ndarray, sampling: SamplingConfig, seed: int) -> DecayFit | None:
        """Return no fit, preserving preparation-only workflows."""
        return None


@dataclass(frozen=True)
class _ModelTerms:
    y0: object
    b: object
    c: object
    noise: BuiltNoise


class _DecayModelBuilder:
    """Translate explicit prior objects into one PyMC model specification."""

    def __init__(self, priors: DecayPriorConfig):
        self.priors = priors

    def build(self, pm, *, y0_init: float, c_init: float, b_init: float, fixed_sigma: float) -> _ModelTerms:
        """Build model variables from the configured prior objects."""
        y0 = self.priors.y0.build(pm, "y0", init=y0_init)
        b = self.priors.b.build(pm, "b", init=b_init)
        c = self.priors.asymptote.build(pm, y0=y0, y0_init=y0_init, c_init=c_init)
        noise = self.priors.noise.build(pm, fixed_sigma=fixed_sigma)
        return _ModelTerms(y0=y0, b=b, c=c, noise=noise)


class PyMCDecayFitter:
    """Fit exponential MH decay curves with optional PyMC sampling."""

    def fit(self, x: np.ndarray, y: np.ndarray, sampling: SamplingConfig, seed: int) -> DecayFit | None:
        """Fit one curve and return posterior parameter draws.

        Args:
            x: Lag values for observed year pairs.
            y: MH similarities for the corresponding pairs.
            sampling: Sampling and prior configuration.
            seed: Random seed for this entity’s model.

        Returns:
            Posterior draws, or ``None`` when fewer than three finite pairs exist.

        Raises:
            ImportError: If the optional PyMC dependency is unavailable.
        """
        try:
            import pymc as pm
        except ImportError as exc:
            raise ImportError("PyMC fitting requires the dependency installed with: pip install -r requirements.txt") from exc

        x = np.asarray(x, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite]
        y = np.clip(y[finite], 0.0, 1.0)
        if len(x) < 3:
            return None
        # Use the observed endpoints to initialize the selected prior strategy.
        order = np.argsort(x)
        y_sorted = y[order]
        y0_init = float(np.clip(y_sorted[0], 1e-3, 1 - 1e-3))
        c_init = float(np.clip(np.median(y_sorted[-max(1, len(y_sorted) // 4):]), 1e-3, 1 - 1e-3))
        b_init = 1.0 / max(1.0, float(np.max(x)))
        fixed_sigma = (
            float(sampling.observation_sigma)
            if sampling.observation_sigma is not None
            else max(0.03, 0.5 * float(np.std(y, ddof=1)))
        )
        model_builder = _DecayModelBuilder(sampling.priors)
        with pm.Model() as model:
            terms = model_builder.build(
                pm,
                y0_init=y0_init,
                c_init=c_init,
                b_init=b_init,
                fixed_sigma=fixed_sigma,
            )
            mu = terms.c + (terms.y0 - terms.c) * pm.math.exp(-terms.b * x)
            pm.Normal("obs", mu=mu, sigma=terms.noise.variable, observed=y)
            idata = pm.sample(
                draws=int(sampling.draws),
                tune=int(sampling.tune),
                chains=int(sampling.chains),
                cores=min(int(sampling.cores), int(sampling.chains)),
                target_accept=float(sampling.target_accept),
                progressbar=False,
                random_seed=int(seed),
            )
        posterior = idata.posterior
        b_samples = np.asarray(posterior["b"], dtype=float).reshape(-1)
        sigma_samples = terms.noise.posterior_samples(posterior, len(b_samples))
        divergences = int(np.asarray(idata.sample_stats["diverging"], dtype=int).sum())
        return DecayFit(
            y0_samples=sampling.priors.y0.posterior_samples(posterior, len(b_samples), "y0"),
            b_samples=b_samples,
            c_samples=np.asarray(posterior["c"], dtype=float).reshape(-1),
            divergences=divergences,
            sigma_samples=sigma_samples,
        )


def fit_entities(
    prepared: PreparedData,
    config: AnalysisConfig,
    fitter: DecayFitter | None,
) -> dict[str, EntityDecayData]:
    """Compute pair data and optionally fit every prepared entity.

    Args:
        prepared: Prepared entity matrices.
        config: Analysis configuration controlling lag and sampling behavior.
        fitter: Fitter implementation, or ``None`` to skip fitting.

    Returns:
        Mapping from entity labels to observed and fitted decay data.
    """
    result: dict[str, EntityDecayData] = {}
    for index, entity in enumerate(prepared.entity_order):
        x_values, y_values = entity_pair_data(prepared.entities[entity].mh_counts, config.mh.max_lag)
        fit = None
        if fitter is not None and len(x_values) >= 3:
            fit = fitter.fit(x_values, y_values, config.sampling, int(config.sampling.seed) + index)
            if fit is not None and fit.divergences:
                LOGGER.warning("%s fit reported %d divergences.", entity, fit.divergences)
        result[entity] = EntityDecayData(x=x_values, y=y_values, fit=fit)
    return result
