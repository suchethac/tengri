# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for ``RadioSEDComponentConfig.agn_radio_model`` dispatch.

Covers the four-way dispatch added when AGN-radio gained selectable
sub-models:

- ``"powerlaw"`` (default, current behaviour) — :func:`radio_total`.
- ``"dpl"`` — AGNfitter-rx broken double power-law,
  :func:`radio_total_dpl`.
- ``"JP"``, ``"KP"``, ``"tribble"`` — reserved physical aging kernels;
  raise :class:`NotImplementedError` until the follow-up PR lands.

Also asserts that the parameter registry exposes the new free
parameters under the canonical ``radio_*`` prefix per
``docs/dev/NAMING_CONTRACT.md`` §3.2.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from tengri.components.radio.component import (
    AGN_RADIO_MODELS,
    RadioSEDComponent,
    RadioSEDComponentConfig,
)
from tengri.components.radio.radio import radio_total, radio_total_dpl
from tengri.core import PipelineState
from tengri.parameters._param_defs import _RADIO_PARAMS


@pytest.fixture
def wave():
    return jnp.linspace(1e3, 1e9, 256)


@pytest.fixture
def state(wave):
    return PipelineState(
        wave=wave,
        sed_intrinsic=jnp.zeros_like(wave),
        sed_observed=jnp.ones_like(wave),
        derived={"L_ir": 1e44, "L_agn_bol": 1e45, "log_mstar": 10.5},
    )


def _all_radio_params() -> dict:
    """Build a complete radio_* params dict with literature defaults."""
    return {
        "redshift": 0.5,
        "radio_q_ir": 2.64,
        "radio_alpha_sf": 0.8,
        "radio_loudness": 1.0,
        "radio_alpha_agn": 0.7,
        "radio_T_e": 1e4,
        "radio_alpha_ff": -0.1,
        "radio_alpha_thin": -0.75,
        "radio_alpha_thick": -0.1,
        "radio_log_nu_t": 10.0,
        "radio_log_nu_cut": 13.0,
    }


@pytest.mark.unit
class TestParameterRegistry:
    """The new params are declared under the radio_ prefix."""

    @pytest.mark.parametrize(
        "name",
        [
            "radio_alpha_thin",
            "radio_alpha_thick",
            "radio_log_nu_t",
            "radio_log_nu_cut",
        ],
    )
    def test_registered_in_param_defs(self, name):
        assert name in _RADIO_PARAMS, f"{name!r} missing from _RADIO_PARAMS"

    @pytest.mark.parametrize(
        "name",
        [
            "radio_alpha_thin",
            "radio_alpha_thick",
            "radio_log_nu_t",
            "radio_log_nu_cut",
        ],
    )
    def test_declared_by_component(self, name):
        decls = {d.name for d in RadioSEDComponent().declared_parameters()}
        assert name in decls

    @pytest.mark.parametrize("name", ["radio_alpha_inj", "radio_log_nu_break"])
    def test_aging_kernel_params_not_yet_registered(self, name):
        # Reserved-params cleanup: until the JP/KP/Tribble physics lands
        # the two free parameters they would own are absent from both the
        # registry and the component declaration. This test pins that
        # contract so a partial re-introduction (params without physics,
        # or vice versa) is caught.
        assert name not in _RADIO_PARAMS
        decls = {d.name for d in RadioSEDComponent().declared_parameters()}
        assert name not in decls

    def test_naming_follows_radio_prefix(self):
        for name in _RADIO_PARAMS:
            assert name.startswith("radio_"), f"{name!r} violates radio_ prefix rule"


@pytest.mark.unit
class TestConfigValidation:
    """``agn_radio_model`` must be one of the documented values."""

    def test_default_is_powerlaw(self):
        assert RadioSEDComponentConfig().agn_radio_model == "powerlaw"

    @pytest.mark.parametrize("model", AGN_RADIO_MODELS)
    def test_all_documented_models_construct(self, model):
        cfg = RadioSEDComponentConfig(agn_radio_model=model)
        assert cfg.agn_radio_model == model

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown agn_radio_model"):
            RadioSEDComponentConfig(agn_radio_model="not_a_real_model")


@pytest.mark.unit
class TestPowerLawDispatch:
    """``agn_radio_model="powerlaw"`` matches direct radio_total call."""

    def test_default_behaviour_unchanged(self, state, wave):
        comp = RadioSEDComponent()  # default config
        params = _all_radio_params()
        new_state = comp.apply(state, params)

        expected = radio_total(
            wave,
            L_ir=1e44,
            L_agn_bol=1e45,
            q_ir=2.64,
            alpha_sf=0.8,
            radio_loudness=1.0,
            alpha_agn=0.7,
            sfr_mode="bell2003",
            log_mstar=10.5,
            redshift=0.5,
            include_freefree=True,
            T_e=1e4,
            alpha_ff=-0.1,
        )
        assert jnp.allclose(new_state.derived["sed_radio"], expected, rtol=1e-12)


@pytest.mark.unit
class TestDplDispatch:
    """``agn_radio_model="dpl"`` matches direct radio_total_dpl call."""

    def test_dpl_routes_to_radio_total_dpl(self, state, wave):
        cfg = RadioSEDComponentConfig(agn_radio_model="dpl")
        comp = RadioSEDComponent(config=cfg)
        params = _all_radio_params()
        new_state = comp.apply(state, params)

        expected = radio_total_dpl(
            wave,
            L_ir=1e44,
            L_agn_bol=1e45,
            q_ir=2.64,
            alpha_sf=0.8,
            radio_loudness=1.0,
            alpha1=-0.75,
            alpha2=-0.1,
            log_nu_t=10.0,
            log_nu_cut=13.0,
            sfr_mode="bell2003",
            log_mstar=10.5,
            redshift=0.5,
            include_freefree=True,
            T_e=1e4,
            alpha_ff=-0.1,
        )
        assert jnp.allclose(new_state.derived["sed_radio"], expected, rtol=1e-12)

    def test_dpl_differs_from_powerlaw(self, state):
        """At high frequencies the cutoff makes DPL diverge from powerlaw."""
        params = _all_radio_params()
        # Drive cutoff into the radio band so the two models must disagree.
        params["radio_log_nu_cut"] = 9.5  # 3 GHz cutoff
        params["radio_loudness"] = 2.0  # boost AGN to dominate

        powerlaw = RadioSEDComponent().apply(state, params).derived["sed_radio"]
        dpl = (
            RadioSEDComponent(config=RadioSEDComponentConfig(agn_radio_model="dpl"))
            .apply(state, params)
            .derived["sed_radio"]
        )

        assert not jnp.allclose(powerlaw, dpl)


@pytest.mark.unit
class TestAgingKernelsRejectedAtConstruction:
    """JP/KP/Tribble are not yet implemented — construction must fail
    early with a clean ValueError, not silently succeed and then raise
    NotImplementedError deep inside apply()."""

    @pytest.mark.parametrize("model", ["JP", "KP", "tribble"])
    def test_aging_kernels_raise_at_construction(self, model):
        with pytest.raises(ValueError, match="Unknown agn_radio_model"):
            RadioSEDComponentConfig(agn_radio_model=model)
