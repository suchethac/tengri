# SPDX-License-Identifier: BSD-3-Clause
"""End-to-end regression guards for the Phase II orchestrator JIT path.

After Phase II-2.2-followup (ForwardState registered as a JAX pytree),
the full ``Stellar + Radio + XRay + IGM`` adapter chain is supposed to
flow through ``jax.jit`` end-to-end with bit-exact match to the eager
path. This module guards that invariant — if a future change breaks
the pytree registration, derived-dict propagation, or JIT
compatibility of any of the four shipped adapters, this file fails.

Tests intentionally use real SSP data
(``data/ssp_prsc_miles_chabrier_*.h5``); they skip when unavailable
(matches the convention in :mod:`tests/integration`).
"""

from __future__ import annotations

import pathlib

import chex
import jax
import jax.numpy as jnp
import pytest

from tengri.components.igm.component import IGMSEDComponent
from tengri.components.radio.component import RadioSEDComponent
from tengri.components.stellar import StellarSEDComponent
from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
from tengri.components.xray.component import XRaySEDComponent
from tengri.forward.orchestrator import default_params_dict, run_components
from tengri.protocols.component import ForwardState
from tests._jit_parity import assert_jit_matches_eager

_SSP_PATH = pathlib.Path("data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5").resolve()


@pytest.fixture(scope="module")
def ssp():
    if not _SSP_PATH.exists():
        pytest.skip(f"SSP file not present at {_SSP_PATH}")
    return load_ssp_data(str(_SSP_PATH))


@pytest.fixture(scope="module")
def base_params():
    return {
        # tsnorm SFH
        "sfh_tsnorm_log_total_mass": jnp.asarray(1.0),
        "sfh_tsnorm_peak_lbt_gyr": jnp.asarray(2.0),
        "sfh_tsnorm_width_gyr": jnp.asarray(1.0),
        "sfh_tsnorm_skew": jnp.asarray(0.0),
        "sfh_tsnorm_trunc": jnp.asarray(3.0),
        # delta metallicity
        "met_logzsol": jnp.asarray(-0.5),
        # Radio + X-ray at their declared defaults. Spelling them out was a copy
        # of the declaration: all twelve literals matched their default exactly,
        # and the copy was already missing thirteen of the twenty-five declared
        # parameters when xray_det_hmxb became the first of the thirteen that a
        # component went on to index (#1832).
        **default_params_dict([RadioSEDComponent(), XRaySEDComponent()]),
        # The one deliberate departure from the declared defaults.
        "xray_delta_alpha_ox": jnp.asarray(-1.4),
        # observed at z=0 to dodge the upstream-DSPS NaN edge case
        # (t_obs < ssp_lg_age_gyr.max() ⇒ DSPS triweight kernel vanishes).
        "redshift": jnp.asarray(0.0),
    }


def test_orchestrator_chain_eager(ssp, base_params):
    """All four adapters chain through run_components without errors."""
    components = [
        StellarSEDComponent(ssp_data=ssp),
        RadioSEDComponent(),
        XRaySEDComponent(),
        IGMSEDComponent(),
    ]
    state0 = ForwardState(wave=ssp.ssp_wave, sed_observed=jnp.ones(len(ssp.ssp_wave)))
    s = run_components(components, state0, base_params)

    assert s.sed_intrinsic is not None
    chex.assert_tree_all_finite(s.sed_intrinsic)
    # Stellar contract keys present
    for k in ("log_mstar", "log_mstar_formed", "sfr", "lnu_age", "nion"):
        assert k in s.derived
    # Per-component derived publishes
    assert "sed_radio" in s.derived
    assert "sed_xray" in s.derived
    assert "igm_transmission" in s.derived


def test_orchestrator_chain_jit_matches_eager(ssp, base_params):
    """JIT-compiled orchestrator produces bit-exact match vs the eager path."""
    components = [
        StellarSEDComponent(ssp_data=ssp),
        RadioSEDComponent(),
        XRaySEDComponent(),
        IGMSEDComponent(),
    ]
    state0 = ForwardState(wave=ssp.ssp_wave, sed_observed=jnp.ones(len(ssp.ssp_wave)))

    s_eager = run_components(components, state0, base_params)
    s_jit = assert_jit_matches_eager(lambda p: run_components(components, state0, p), base_params)

    assert jnp.allclose(s_jit.sed_intrinsic, s_eager.sed_intrinsic, rtol=1e-12)
    # Spot-check derived scalars too
    for k in ("log_mstar", "nion"):
        assert jnp.allclose(
            jnp.asarray(s_jit.derived[k]),
            jnp.asarray(s_eager.derived[k]),
            rtol=1e-12,
        )


def test_stellar_grad_finite(ssp, base_params):
    """Gradients flow through the JIT-compiled apply()."""
    stellar = StellarSEDComponent(ssp_data=ssp)
    state0 = ForwardState(wave=ssp.ssp_wave)

    def loss(p):
        return jnp.sum(stellar.apply(state0, p).sed_intrinsic)

    g = jax.jit(jax.grad(loss))(base_params)
    # The peak-SFR gradient must be positive (more SFR → more L).
    assert float(g["sfh_tsnorm_log_total_mass"]) > 0.0
    # All gradient leaves are finite.
    chex.assert_tree_all_finite(g)


def test_orchestrator_chain_traces_once(ssp, base_params):
    """The full Stellar→Radio→XRay→IGM chain must not retrace on repeated
    same-shape calls — the recompile of a 4-component chain costs ~30 s on
    real SSP grids, so an accidental retrace caused by a Python-scalar leak
    in any apply() would be immediately user-visible. ``n=1`` is tight by
    design; the only way to pass is to compile once and then hit the cache."""
    chex.clear_trace_counter()
    components = [
        StellarSEDComponent(ssp_data=ssp),
        RadioSEDComponent(),
        XRaySEDComponent(),
        IGMSEDComponent(),
    ]
    state0 = ForwardState(wave=ssp.ssp_wave, sed_observed=jnp.ones(len(ssp.ssp_wave)))

    @jax.jit
    @chex.assert_max_traces(n=1)
    def chain(p):
        return run_components(components, state0, p)

    chain(base_params)
    chain(base_params)
    chain(base_params)


def test_orchestrator_jit_grad_traces_once(ssp, base_params):
    """jit ∘ grad over the orchestrator must not retrace either. This is the
    hot path under every gradient-based fitter (MAP, VI, NUTS warmup); a
    retrace here is the most expensive class of regression in the codebase
    since the AD-compiled XLA graph is roughly 3-4× the forward graph."""
    chex.clear_trace_counter()
    components = [
        StellarSEDComponent(ssp_data=ssp),
        RadioSEDComponent(),
        XRaySEDComponent(),
        IGMSEDComponent(),
    ]
    state0 = ForwardState(wave=ssp.ssp_wave, sed_observed=jnp.ones(len(ssp.ssp_wave)))

    def loss(p):
        return jnp.sum(run_components(components, state0, p).sed_intrinsic)

    @jax.jit
    @chex.assert_max_traces(n=1)
    def grad_chain(p):
        return jax.grad(loss)(p)

    grad_chain(base_params)
    grad_chain(base_params)


@pytest.fixture(scope="module")
def full_chain_params(base_params):
    """``base_params`` extended with the dust + AGN keys."""
    return {
        **base_params,
        # dust two-component
        "dust_tau_bc": jnp.asarray(1.0),
        "dust_tau_diff": jnp.asarray(0.3),
        "dust_slope": jnp.asarray(-0.7),
        "dust_T": jnp.asarray(35.0),
        "dust_beta_ir": jnp.asarray(1.6),
        # dust emission component params (modified_blackbody + casey2012)
        "dust_epsilon_mbb": jnp.asarray(1.0),
        "dust_alpha_mir": jnp.asarray(2.0),
        # AGN
        "agn_log_lbol": jnp.asarray(11.0),
        "agn_lum_ratio": jnp.asarray(0.1),
    }


@pytest.mark.parametrize("agn_model", ["multicolor_agn", "kubota_done"])
@pytest.mark.parametrize("dust_law", ["power_law", "calzetti", "smc", "cardelli"])
@pytest.mark.parametrize("emission_model", ["modified_blackbody", "casey2012"])
def test_full_chain_composability(ssp, full_chain_params, agn_model, dust_law, emission_model):
    """Stellar + Nebular(BakedIn) + AGN + Dust + Radio + XRay + IGM
    composes for any registered AGN model × dust law × emission template.

    Verifies the architectural promise: physics components are
    independently swappable via their config keys, and the orchestrator
    JIT path produces bit-exact match to the eager path.
    """
    from tengri.components.agn.component import AGNSEDComponent, AGNSEDComponentConfig
    from tengri.components.dust.two_component import (
        DustSEDComponent,
        DustSEDComponentConfig,
    )
    from tengri.components.nebular.component import (
        NebularSEDComponent,
        NebularSEDComponentConfig,
    )
    from tengri.components.sed_model_component import _REGISTRY

    chain = [
        StellarSEDComponent(ssp_data=ssp),
        NebularSEDComponent(config=NebularSEDComponentConfig(backend="baked_in")),
        AGNSEDComponent(config=AGNSEDComponentConfig(model=agn_model)),
        DustSEDComponent(config=DustSEDComponentConfig(law_bc=dust_law, law_diff=dust_law)),
        # Emission is a separate registry component now; placed after the attenuator so
        # it reads the published L_ir (energy balance).
        _REGISTRY[emission_model](),
        RadioSEDComponent(),
        XRaySEDComponent(),
        IGMSEDComponent(),
    ]
    state0 = ForwardState(wave=ssp.ssp_wave, sed_observed=jnp.ones(len(ssp.ssp_wave)))

    s_eager = run_components(chain, state0, full_chain_params)
    s_jit = jax.jit(lambda p: run_components(chain, state0, p))(full_chain_params)

    chex.assert_tree_all_finite(s_eager.sed_intrinsic)
    # JIT determinism check across diverse model paths. rtol=1e-6 (not 1e-12):
    # the tabulated dust-emission templates (e.g. casey2012) accumulate ~1e-9
    # float64 round-off from XLA op-reassociation under jit — benign and far
    # below physical relevance. A real JIT bug produces O(1) divergence, which
    # this still catches. The single-model strict path is covered at 1e-12 by
    # ``test_orchestrator_chain_jit_matches_eager``.
    #
    # The atol floor covers near-zero pixels: the EUV tail of the kubota_done
    # disc sits ~9 dex below the SED scale, where any graph-shape change (a
    # new constant leaf, a fusion re-order) moves rounding past a pure-rtol
    # threshold while remaining ~1e-14 of the SED itself. Scale-tied atol
    # keeps the check meaningful at physical pixels and blind to the floor.
    _scale = jnp.max(jnp.abs(s_eager.sed_intrinsic))
    assert jnp.allclose(s_jit.sed_intrinsic, s_eager.sed_intrinsic, rtol=1e-6, atol=1e-12 * _scale)
    # Cross-component publications all populated:
    for k in ("L_ir", "L_agn_bol", "sed_radio", "sed_xray", "igm_transmission"):
        assert k in s_eager.derived
