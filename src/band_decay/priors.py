"""Object-based prior specifications for exponential decay models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


class ParameterPrior(ABC):
    """Prior capable of creating a model variable and reading its draws."""

    def option_name(self) -> str:
        """Return the concise name used by configuration controls."""
        return self.__class__.__name__

    def validate(self) -> None:
        """Validate this prior specification."""
        return None

    @abstractmethod
    def build(self, pm, name: str, *, init: float):
        """Create a model variable or fixed value."""

    @abstractmethod
    def posterior_samples(self, posterior, sample_count: int, name: str) -> np.ndarray:
        """Read aligned posterior samples for the parameter."""


@dataclass(frozen=True)
class BetaPrior(ParameterPrior):
    """A Beta prior defined by alpha and beta shape parameters."""

    alpha: float
    beta: float

    def __post_init__(self) -> None:
        if float(self.alpha) <= 0 or float(self.beta) <= 0:
            raise ValueError("Beta prior alpha and beta must be positive.")

    def build(self, pm, name: str, *, init: float):
        """Create a PyMC Beta variable."""
        return pm.Beta(name, alpha=float(self.alpha), beta=float(self.beta), initval=float(init))

    def posterior_samples(self, posterior, sample_count: int, name: str) -> np.ndarray:
        """Read Beta posterior samples."""
        return np.asarray(posterior[name], dtype=float).reshape(-1)


@dataclass(frozen=True)
class HalfNormalPrior(ParameterPrior):
    """A HalfNormal prior defined by its scale."""

    sigma: float

    def __post_init__(self) -> None:
        if float(self.sigma) <= 0:
            raise ValueError("HalfNormal prior sigma must be positive.")

    def build(self, pm, name: str, *, init: float):
        """Create a PyMC HalfNormal variable."""
        return pm.HalfNormal(name, sigma=float(self.sigma), initval=float(init))

    def posterior_samples(self, posterior, sample_count: int, name: str) -> np.ndarray:
        """Read HalfNormal posterior samples."""
        return np.asarray(posterior[name], dtype=float).reshape(-1)


@dataclass(frozen=True)
class FixedValuePrior(ParameterPrior):
    """A parameter fixed at one scalar value."""

    value: float

    def build(self, pm, name: str, *, init: float):
        """Return the fixed value without creating a random variable."""
        return float(self.value)

    def posterior_samples(self, posterior, sample_count: int, name: str) -> np.ndarray:
        """Return one fixed value for each posterior draw."""
        return np.full(int(sample_count), float(self.value), dtype=float)


class AsymptotePrior(ABC):
    """Strategy for constructing the decay asymptote."""

    def option_name(self) -> str:
        """Return the concise name used by configuration controls."""
        return self.__class__.__name__

    def validate(self) -> None:
        """Validate this asymptote strategy."""
        return None

    @abstractmethod
    def build(self, pm, *, y0, y0_init: float, c_init: float):
        """Create the model's ``c`` value."""

    @abstractmethod
    def parameter_label(self) -> str:
        """Return the displayed asymptote parameter label."""

    @abstractmethod
    def includes_direct_parameter(self) -> bool:
        """Return whether direct ``c`` should be shown in summaries."""


@dataclass(frozen=True)
class DirectAsymptote(AsymptotePrior):
    """Use a direct prior for the asymptote ``c``."""

    prior: ParameterPrior

    def validate(self) -> None:
        """Validate the wrapped parameter prior."""
        self.prior.validate()

    def build(self, pm, *, y0, y0_init: float, c_init: float):
        """Create a directly distributed ``c`` variable."""
        return self.prior.build(pm, "c", init=c_init)

    def parameter_label(self) -> str:
        """Return the direct asymptote label."""
        return "c"

    def includes_direct_parameter(self) -> bool:
        """Return whether the direct asymptote is available."""
        return True


@dataclass(frozen=True)
class RelativeAsymptote(AsymptotePrior):
    """Model ``c`` as ``y0`` multiplied by a ratio prior."""

    ratio_prior: ParameterPrior

    def validate(self) -> None:
        """Validate the wrapped ratio prior."""
        self.ratio_prior.validate()

    def build(self, pm, *, y0, y0_init: float, c_init: float):
        """Create the ratio variable and deterministic ``c`` value."""
        ratio_init = float(np.clip(c_init / max(float(y0_init), 1e-6), 1e-3, 1 - 1e-3))
        ratio = self.ratio_prior.build(pm, "c_ratio", init=ratio_init)
        return pm.Deterministic("c", y0 * ratio)

    def parameter_label(self) -> str:
        """Return the relative asymptote label."""
        return "c/y0"

    def includes_direct_parameter(self) -> bool:
        """Return whether a direct asymptote is available."""
        return False


class NoisePrior(ABC):
    """Strategy for fixed or sampled observation noise."""

    def option_name(self) -> str:
        """Return the concise name used by configuration controls."""
        return self.__class__.__name__

    def validate(self) -> None:
        """Validate this noise strategy."""
        return None

    @abstractmethod
    def build(self, pm, *, fixed_sigma: float):
        """Return model noise and its posterior reader."""


@dataclass(frozen=True)
class FixedObservationNoise(NoisePrior):
    """Keep the fitted observation scale fixed from the data."""

    def build(self, pm, *, fixed_sigma: float):
        """Return the data-derived fixed scale."""
        return FixedNoiseVariable(float(fixed_sigma))


@dataclass(frozen=True)
class SampledObservationNoise(NoisePrior):
    """Sample observation noise from a HalfNormal prior."""

    prior: HalfNormalPrior

    def validate(self) -> None:
        """Validate the wrapped noise prior."""
        self.prior.validate()

    def build(self, pm, *, fixed_sigma: float):
        """Create a sampled noise variable."""
        variable = self.prior.build(pm, "sigma", init=fixed_sigma)
        return SampledNoiseVariable(variable)


class BuiltNoise(ABC):
    """Result of materializing an observation-noise strategy."""

    @property
    @abstractmethod
    def variable(self):
        """Return the value used by the likelihood."""

    @abstractmethod
    def posterior_samples(self, posterior, sample_count: int) -> np.ndarray | None:
        """Read noise draws or return ``None`` for fixed noise."""


@dataclass(frozen=True)
class FixedNoiseVariable(BuiltNoise):
    """Materialized fixed observation noise."""

    value: float

    @property
    def variable(self):
        """Return the fixed likelihood scale."""
        return self.value

    def posterior_samples(self, posterior, sample_count: int) -> np.ndarray | None:
        """Return no posterior noise samples."""
        return None


@dataclass(frozen=True)
class SampledNoiseVariable(BuiltNoise):
    """Materialized sampled observation noise."""

    value: object

    @property
    def variable(self):
        """Return the PyMC likelihood scale."""
        return self.value

    def posterior_samples(self, posterior, sample_count: int) -> np.ndarray | None:
        """Read sampled observation noise from the posterior."""
        return np.asarray(posterior["sigma"], dtype=float).reshape(-1)


@dataclass(frozen=True)
class DecayPriorConfig:
    """Compose all priors needed by one exponential decay model."""

    y0: ParameterPrior
    b: ParameterPrior
    asymptote: AsymptotePrior
    noise: NoisePrior

    def __post_init__(self) -> None:
        # Validate each composed strategy through its own implementation.
        for prior in (self.y0, self.b, self.asymptote, self.noise):
            prior.validate()

    @classmethod
    def legacy(cls) -> "DecayPriorConfig":
        """Return explicit priors matching the original legacy fit."""
        return cls(
            y0=BetaPrior(2.0, 2.0),
            b=HalfNormalPrior(2.0),
            asymptote=DirectAsymptote(BetaPrior(2.0, 2.0)),
            noise=FixedObservationNoise(),
        )

    @classmethod
    def constrained(cls) -> "DecayPriorConfig":
        """Return explicit priors with ``c / y0`` and sampled noise."""
        return cls(
            y0=BetaPrior(8.0, 2.0),
            b=HalfNormalPrior(2.0),
            asymptote=RelativeAsymptote(BetaPrior(2.0, 2.0)),
            noise=SampledObservationNoise(HalfNormalPrior(0.1)),
        )

    @classmethod
    def informative(cls) -> "DecayPriorConfig":
        """Return explicit informative priors matching the old profile."""
        return cls(
            y0=BetaPrior(5.0, 1.0),
            b=HalfNormalPrior(2.0),
            asymptote=DirectAsymptote(BetaPrior(1.0, 5.0)),
            noise=FixedObservationNoise(),
        )
