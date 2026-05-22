"""Tests for _LegacySEDSubModel — wraps existing SEDModel as a SubModel."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from tengri.forward._sed_submodel_adapter import _LegacySEDSubModel
from tengri.protocols import SubModel

pytestmark = pytest.mark.contract


@pytest.fixture
def sed_model_minimal(synthetic_ssp, simple_observation):
    """A minimal SEDModel for shape testing. Reuses existing test fixtures."""
    from tengri import FIXED, SEDModel

    return SEDModel.build(
        ssp_data=synthetic_ssp,
        observation=simple_observation,
        sfh={"type": "dpl", "*": FIXED},
        dust={"type": "two_component", "law_bc": "calzetti", "*": FIXED},
        neb={"type": "none"},
    )


def test_adapter_satisfies_submodel_protocol(sed_model_minimal) -> None:
    sub = _LegacySEDSubModel(sed_model_minimal)
    assert isinstance(sub, SubModel)


def test_adapter_name_is_sed(sed_model_minimal) -> None:
    sub = _LegacySEDSubModel(sed_model_minimal)
    assert sub.name == "sed"


def test_adapter_declared_parameters_delegates(sed_model_minimal) -> None:
    sub = _LegacySEDSubModel(sed_model_minimal)
    declared = sub.declared_parameters()
    assert {d.name for d in declared} == set(sed_model_minimal.spec.free_params)


def test_adapter_holds_sed_by_reference(sed_model_minimal) -> None:
    sub = _LegacySEDSubModel(sed_model_minimal)
    assert sub.sed_model is sed_model_minimal


def test_adapter_run_returns_forward_state(sed_model_minimal) -> None:
    from tengri.protocols.component import ForwardState

    sub = _LegacySEDSubModel(sed_model_minimal)
    state = ForwardState(wave=jnp.array([3000.0, 5000.0, 8000.0]))
    params = {name: 0.5 for name in sed_model_minimal.spec.free_params}
    result = sub.run(state, params)
    assert isinstance(result, ForwardState)
    assert result.wave is not None
