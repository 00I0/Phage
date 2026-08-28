from band_decay import (
    BetaPrior,
    DecayPriorConfig,
    DirectAsymptote,
    FixedObservationNoise,
    HalfNormalPrior,
    NormalizedDecay,
    RelativeAsymptote,
)


def test_decay_prior_configuration_is_composed_from_objects() -> None:
    config = DecayPriorConfig(
        y0=BetaPrior(2.0, 2.0),
        b=HalfNormalPrior(2.0),
        asymptote=DirectAsymptote(BetaPrior(2.0, 2.0)),
        noise=FixedObservationNoise(),
    )
    assert config.y0.alpha == 2.0
    assert config.asymptote.parameter_label() == "c"
    assert NormalizedDecay().parameter_label() == "c/y0"


def test_relative_asymptote_is_a_separate_dispatch_strategy() -> None:
    config = DecayPriorConfig(
        y0=BetaPrior(8.0, 2.0),
        b=HalfNormalPrior(2.0),
        asymptote=RelativeAsymptote(BetaPrior(2.0, 2.0)),
        noise=FixedObservationNoise(),
    )
    assert config.asymptote.parameter_label() == "c/y0"
