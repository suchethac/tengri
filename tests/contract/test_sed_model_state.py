# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the Phase 1 SEDModelState frozen bundle."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from tengri.forward.sed_model import SEDModel
from tengri.forward.sed_model_types import SEDModelState
from tengri.parameters.parameters import Parameters
from tengri.parameters.priors import Fixed, Uniform

# ── Paths for SSP data ──────────────────────────────────────
_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_SSP_EXISTS = len(list(_DATA_DIR.glob("ssp_*.h5"))) > 0

# One assignment holding both. Assigning `pytestmark` twice rebinds the name,
# which silently dropped the `contract` taxonomy marker.
pytestmark = [
    pytest.mark.contract,
    pytest.mark.skipif(
        not _SSP_EXISTS,
        reason="SSP data file not found — tests require data/ssp_*.h5",
    ),
]


@pytest.fixture(scope="module")
def minimal_spec():
    """Minimal parameter spec for testing state construction."""
    return Parameters(
        mean_sfh_type="dpl",
        sfh_dpl_alpha=Uniform(0.5, 4.0),
        sfh_dpl_beta=Uniform(0.5, 4.0),
        met_logzsol=Fixed(0.0),
        dust_tau_bc=Fixed(0.0),
        dust_tau_diff=Uniform(0.0, 2.0),
        redshift=Fixed(0.1),
    )


@pytest.fixture(scope="module")
def synthetic_ssp():
    """Minimal synthetic SSP for state tests."""
    from tengri.components.stellar.sps.dsps_wrapper import SSPData

    n_met, n_age, n_wave = 3, 20, 100
    wave = jnp.linspace(3000.0, 10000.0, n_wave)
    ages_gyr = jnp.linspace(-1.0, 1.14, n_age)
    key = jax.random.PRNGKey(456)
    flux = jnp.abs(jax.random.normal(key, (n_met, n_age, n_wave))) * 1e-3 + 1e-5
    lgmet = jnp.array([-1.5, -0.5, 0.0])
    return SSPData(ssp_wave=wave, ssp_flux=flux, ssp_lg_age_gyr=ages_gyr, ssp_lgmet=lgmet)


@pytest.fixture(scope="module")
def minimal_model(minimal_spec, synthetic_ssp):
    """Build minimal SEDModel with state attribute."""
    return SEDModel(minimal_spec, synthetic_ssp, precompute=False, filters=None)


@pytest.mark.unit
def test_sed_model_has_state_attribute(minimal_model):
    """SEDModel should have _state attribute that is a SEDModelState instance."""
    assert hasattr(minimal_model, "_state"), "model._state attribute missing"
    assert isinstance(minimal_model._state, SEDModelState), (
        f"model._state is {type(minimal_model._state)}, not SEDModelState"
    )


@pytest.mark.unit
def test_sed_model_state_is_frozen(minimal_model):
    """SEDModelState should be a frozen dataclass."""
    assert dataclasses.is_dataclass(minimal_model._state), "model._state is not a dataclass"
    assert minimal_model._state.__dataclass_params__.frozen, "model._state dataclass is not frozen"

    # Attempt to mutate should raise FrozenInstanceError
    with pytest.raises(dataclasses.FrozenInstanceError):
        minimal_model._state.spec = None


@pytest.mark.unit
def test_state_field_count(minimal_model):
    """SEDModelState should bundle approximately 35+ fields."""
    n_fields = len(dataclasses.fields(minimal_model._state))
    assert n_fields >= 35, (
        f"SEDModelState has {n_fields} fields; expected >= 35 (was the bundle truncated?)"
    )
