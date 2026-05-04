# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for :mod:`tengri.core.component`.

These tests assert structural properties of the
:class:`tengri.core.SEDComponent` Protocol against the first two real
adapters (radio + IGM). They are deliberately *cheap* — no SSP grids,
no inference, no fits — so they run on every PR and catch drift
between the scaffold and live adapters.

If you add a third adapter, register it in :data:`ADAPTERS` and these
tests run for it automatically.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from tengri.components.dust.component import DustAttenuationSEDComponent
from tengri.components.dust.emission_component import DustEmissionSEDComponent
from tengri.components.igm.component import IGMSEDComponent
from tengri.components.nebular.component import NebularSEDComponent
from tengri.components.radio.component import RadioSEDComponent
from tengri.components.xray.component import XRaySEDComponent
from tengri.core import (
    BARE_NAME_ALLOWLIST,
    ParamDeclaration,
    PipelineState,
    SEDComponent,
    SEDComponentState,
)
from tengri.forward.orchestrator import slice_params_for_component
from tengri.parameters.priors import Distribution

ADAPTERS: list[SEDComponent] = [
    RadioSEDComponent(),
    IGMSEDComponent(),
    XRaySEDComponent(),
    DustAttenuationSEDComponent(),
    NebularSEDComponent(),
    DustEmissionSEDComponent(),
]


@pytest.fixture
def wave():
    return jnp.linspace(1e3, 1e9, 256)


@pytest.fixture
def state(wave):
    return PipelineState(
        wave=wave,
        sed_intrinsic=jnp.zeros_like(wave),
        sed_observed=jnp.ones_like(wave),
    )


def _full_params() -> dict:
    return {
        "redshift": 0.5,
        "radio_q_ir": 2.64,
        "radio_alpha_sf": 0.8,
        "radio_loudness": 0.0,
        "radio_alpha_agn": 0.7,
        "radio_T_e": 1e4,
        "radio_alpha_ff": -0.1,
        "igm_z_mid": 7.0,
        "igm_dz": 0.5,
        "igm_log_nhi": 20.0,
        "xray_gamma_hmxb": 2.0,
        "xray_gamma_lmxb": 1.6,
        "xray_gamma_agn": 1.8,
        "xray_E_cut": 300.0,
        "xray_alpha_ox": -1.4,
        "dust_tau_v": 0.3,
        "dust_T": 30.0,
        "dust_beta_ir": 1.8,
    }


@pytest.mark.parametrize("adapter", ADAPTERS, ids=lambda a: a.name)
def test_adapter_satisfies_protocol(adapter):
    """Every adapter must duck-type as :class:`SEDComponent`."""
    assert isinstance(adapter, SEDComponent)


@pytest.mark.parametrize("adapter", ADAPTERS, ids=lambda a: a.name)
def test_parameter_prefix_is_nonempty(adapter):
    """Empty prefixes break the orchestrator's slicing rule.

    ``parameter_prefix`` may be a single ``str`` or a tuple
    ``tuple[str, ...]`` for adapters that span multiple prefix
    domains (StellarSEDComponent owns ``("sfh_", "met_", "chem_")``;
    NebularSEDComponent owns ``("neb_", "shock_", "ionspec_", "gas_")``).
    Every entry must be non-empty and end with ``_``.
    """
    prefix = adapter.parameter_prefix
    prefixes = (prefix,) if isinstance(prefix, str) else tuple(prefix)
    assert len(prefixes) > 0, f"{adapter.name}: empty parameter_prefix"
    for p in prefixes:
        assert p != "", f"{adapter.name}: empty entry in parameter_prefix"
        assert p.endswith("_"), f"{adapter.name}: prefix {p!r} must end with '_'"


@pytest.mark.parametrize("adapter", ADAPTERS, ids=lambda a: a.name)
def test_declared_parameters_obey_prefix_rule(adapter):
    """Names start with parameter_prefix or are in BARE_NAME_ALLOWLIST.

    Zero-parameter adapters (e.g. ``NebularSEDComponent`` with the
    BakedIn backend) are valid — they return an empty list and the
    prefix-rule loop is a no-op.
    """
    decls = adapter.declared_parameters()
    for decl in decls:
        assert isinstance(decl, ParamDeclaration)
        assert isinstance(decl.prior, Distribution)
        assert (
            decl.name.startswith(adapter.parameter_prefix) or decl.name in BARE_NAME_ALLOWLIST
        ), f"{adapter.name}: parameter {decl.name!r} violates the prefix rule"


@pytest.mark.parametrize("adapter", ADAPTERS, ids=lambda a: a.name)
def test_precompute_returns_typed_marker(adapter):
    """precompute() returns a SEDComponentState, even when it's a no-op."""
    cstate = adapter.precompute()
    assert isinstance(cstate, SEDComponentState)


@pytest.mark.parametrize("adapter", ADAPTERS, ids=lambda a: a.name)
def test_apply_returns_new_state(adapter, state):
    """apply() returns a new PipelineState — never mutates the input."""
    sliced = slice_params_for_component(adapter, _full_params())
    new_state = adapter.apply(state, sliced)
    assert isinstance(new_state, PipelineState)
    assert new_state is not state


def test_two_adapters_exist():
    """The two-adapter rule: at least two SEDComponent-conforming classes."""
    assert len(ADAPTERS) >= 2
    prefixes = {a.parameter_prefix for a in ADAPTERS}
    assert len(prefixes) >= 2, "two adapters must use distinct prefixes"


def test_two_adapters_touch_distinct_pipeline_state_slots(wave):
    """The two adapters must exercise different PipelineState slots.

    Radio writes :attr:`sed_intrinsic`; IGM writes :attr:`sed_observed`.
    If both wrote the same slot the seam would be under-tested.
    """
    state = PipelineState(
        wave=wave,
        sed_intrinsic=jnp.zeros_like(wave),
        sed_observed=jnp.ones_like(wave),
        derived={"L_ir": 1e44, "L_agn_bol": 0.0, "log_mstar": 10.0},
    )
    params = _full_params()
    radio = RadioSEDComponent()
    igm = IGMSEDComponent()

    after_radio = radio.apply(state, slice_params_for_component(radio, params))
    after_igm = igm.apply(state, slice_params_for_component(igm, params))

    assert not jnp.array_equal(after_radio.sed_intrinsic, state.sed_intrinsic)
    assert jnp.array_equal(after_radio.sed_observed, state.sed_observed)

    assert jnp.array_equal(after_igm.sed_intrinsic, state.sed_intrinsic)
    assert not jnp.array_equal(after_igm.sed_observed, state.sed_observed)


def test_orchestrator_rejects_empty_prefix(state):
    """slice_params_for_component must refuse parameter_prefix=''."""

    class BadComponent:
        name = "bad"
        parameter_prefix = ""
        config = None

        def declared_parameters(self):
            return []

        def precompute(self, ssp_data=None, wave_grid=None):
            return SEDComponentState()

        def apply(self, state, params):
            return state

    with pytest.raises(ValueError, match="empty parameter_prefix"):
        slice_params_for_component(BadComponent(), _full_params())


def test_orchestrator_passes_bare_redshift_to_components():
    """Even with no 'redshift_' prefix, every component sees redshift."""
    sliced_radio = slice_params_for_component(RadioSEDComponent(), _full_params())
    sliced_igm = slice_params_for_component(IGMSEDComponent(), _full_params())
    assert "redshift" in sliced_radio
    assert "redshift" in sliced_igm
