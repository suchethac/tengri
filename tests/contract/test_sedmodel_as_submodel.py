# SPDX-License-Identifier: BSD-3-Clause
"""SEDModel directly satisfies the SubModel Protocol.

This makes the _LegacySEDSubModel adapter unnecessary; that file is
deleted in this same plan. The architecture spec at
``docs/dev/archive/forward-model-architecture.md`` §4 has `SEDModel` listed
as one of the three sub-models satisfying SubModel.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from tengri import Fixed
from tengri.protocols import SubModel
from tengri.protocols.component import ForwardState

pytestmark = pytest.mark.contract


@pytest.fixture
def sed_model_minimal(synthetic_ssp, simple_observation):
    from tengri import FIXED, SEDModel

    return SEDModel.build(
        ssp_data=synthetic_ssp,
        observation=simple_observation,
        sfh={"type": "dpl", "all_params": FIXED},
        dust_attenuation={"type": "two_component", "law": "calzetti", "all_params": FIXED},
        neb={"type": "none"},
        redshift=Fixed(0.1),
    )


def test_sedmodel_satisfies_submodel_protocol(sed_model_minimal) -> None:
    assert isinstance(sed_model_minimal, SubModel)


def test_sedmodel_declared_parameters_matches_spec(sed_model_minimal) -> None:
    declared = sed_model_minimal.declared_parameters()
    assert {d.name for d in declared} == set(sed_model_minimal.spec.free_params)


def test_sedmodel_run_returns_forward_state(sed_model_minimal) -> None:
    state = ForwardState(wave=jnp.array([3000.0, 5000.0, 8000.0]))
    params = {name: 0.5 for name in sed_model_minimal.spec.free_params}
    out = sed_model_minimal.run(state, params)
    assert isinstance(out, ForwardState)


def test_sedmodel_name_attribute(sed_model_minimal) -> None:
    assert sed_model_minimal.name == "sed"
