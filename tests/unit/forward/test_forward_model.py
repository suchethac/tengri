"""Tests for ForwardModel (forward-model architecture §5).

Tracer-bullet scope: single-population only. Multi-population lives
in the ADR-0012 plan.
"""

from __future__ import annotations

import pytest

from tengri.forward.forward_model import ForwardModel
from tengri.forward.population import Population


@pytest.fixture
def sed_model_minimal(synthetic_ssp, simple_observation):
    from tengri import FIXED, SEDModel

    return SEDModel.build(
        ssp_data=synthetic_ssp,
        observation=simple_observation,
        sfh={"type": "dpl", "*": FIXED},
        dust={"type": "two_component", "law_bc": "calzetti", "*": FIXED},
        neb={"type": "none"},
    )


def test_build_single_population_from_sed_kwarg(sed_model_minimal, simple_observation) -> None:
    forward = ForwardModel.build(
        sed=sed_model_minimal,
        observation=simple_observation,
    )
    assert isinstance(forward, ForwardModel)
    assert len(forward.populations) == 1
    assert forward.populations[0].name == "default"


def test_build_rejects_no_sed_no_populations(simple_observation) -> None:
    with pytest.raises(ValueError, match=r"sed=.*populations="):
        ForwardModel.build(observation=simple_observation)


def test_build_accepts_explicit_populations(sed_model_minimal, simple_observation) -> None:
    from tengri.forward._sed_submodel_adapter import _LegacySEDSubModel

    pop = Population(name="only", sed=_LegacySEDSubModel(sed_model_minimal))
    forward = ForwardModel.build(populations=[pop], observation=simple_observation)
    assert forward.populations[0].name == "only"


def test_build_rejects_multi_population_in_tracer_bullet(
    sed_model_minimal, simple_observation
) -> None:
    """Multi-population is deferred to ADR-0012 plan. Tracer-bullet ships single-pop."""
    from tengri.forward._sed_submodel_adapter import _LegacySEDSubModel

    pops = [
        Population(name="a", sed=_LegacySEDSubModel(sed_model_minimal)),
        Population(name="b", sed=_LegacySEDSubModel(sed_model_minimal)),
    ]
    with pytest.raises(NotImplementedError, match="ADR-0012"):
        ForwardModel.build(populations=pops, observation=simple_observation)
