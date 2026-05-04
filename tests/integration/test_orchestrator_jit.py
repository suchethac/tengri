# SPDX-License-Identifier: BSD-3-Clause
"""End-to-end regression guards for the Phase II orchestrator JIT path.

After Phase II-2.2-followup (PipelineState registered as a JAX pytree),
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

import jax
import jax.numpy as jnp
import pytest

from tengri.components.igm.component import IGMSEDComponent
from tengri.components.radio.component import RadioSEDComponent
from tengri.components.stellar import StellarSEDComponent
from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
from tengri.components.xray.component import XRaySEDComponent
from tengri.core.component import PipelineState
from tengri.forward.orchestrator import run_components

_SSP_PATH = pathlib.Path(
    "data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
).resolve()


@pytest.fixture(scope="module")
def ssp():
    if not _SSP_PATH.exists():
        pytest.skip(f"SSP file not present at {_SSP_PATH}")
    return load_ssp_data(str(_SSP_PATH))


@pytest.fixture(scope="module")
def base_params():
    return {
        # tsnorm SFH
        "sfh_tsnorm_log_peak_sfr": jnp.asarray(1.0),
        "sfh_tsnorm_peak_lbt_gyr": jnp.asarray(2.0),
        "sfh_tsnorm_width_gyr": jnp.asarray(1.0),
        "sfh_tsnorm_skew": jnp.asarray(0.0),
        "sfh_tsnorm_trunc": jnp.asarray(3.0),
        # delta metallicity
        "met_logzsol": jnp.asarray(-0.5),
        # Radio
        "radio_q_ir": jnp.asarray(2.64),
        "radio_alpha_sf": jnp.asarray(0.8),
        "radio_loudness": jnp.asarray(0.0),
        "radio_alpha_agn": jnp.asarray(0.7),
        "radio_T_e": jnp.asarray(1e4),
        "radio_alpha_ff": jnp.asarray(-0.1),
        # X-ray
        "xray_gamma_hmxb": jnp.asarray(2.0),
        "xray_gamma_lmxb": jnp.asarray(1.6),
        "xray_gamma_agn": jnp.asarray(1.8),
        "xray_E_cut": jnp.asarray(300.0),
        "xray_alpha_ox": jnp.asarray(-1.4),
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
    state0 = PipelineState(
        wave=ssp.ssp_wave, sed_observed=jnp.ones(len(ssp.ssp_wave))
    )
    s = run_components(components, state0, base_params)

    assert s.sed_intrinsic is not None
    assert bool(jnp.all(jnp.isfinite(s.sed_intrinsic)))
    # Stellar contract keys present
    for k in ("log_mstar", "log_mstar_formed", "sfr", "lnu_age", "nion"):
        assert k in s.derived
    # Per-component derived publishes
    assert "L_radio" in s.derived
    assert "L_xray" in s.derived
    assert "igm_transmission" in s.derived


def test_orchestrator_chain_jit_matches_eager(ssp, base_params):
    """JIT-compiled orchestrator produces bit-exact match vs the eager path."""
    components = [
        StellarSEDComponent(ssp_data=ssp),
        RadioSEDComponent(),
        XRaySEDComponent(),
        IGMSEDComponent(),
    ]
    state0 = PipelineState(
        wave=ssp.ssp_wave, sed_observed=jnp.ones(len(ssp.ssp_wave))
    )

    s_eager = run_components(components, state0, base_params)
    pipeline_jit = jax.jit(lambda p: run_components(components, state0, p))
    s_jit = pipeline_jit(base_params)

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
    state0 = PipelineState(wave=ssp.ssp_wave)

    def loss(p):
        return jnp.sum(stellar.apply(state0, p).sed_intrinsic)

    g = jax.jit(jax.grad(loss))(base_params)
    # The peak-SFR gradient must be positive (more SFR → more L).
    assert float(g["sfh_tsnorm_log_peak_sfr"]) > 0.0
    # All gradient leaves are finite.
    leaves = jax.tree.leaves(g)
    assert all(bool(jnp.all(jnp.isfinite(jnp.asarray(x)))) for x in leaves)


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
        # AGN
        "agn_log_lbol": jnp.asarray(11.0),
        "agn_frac": jnp.asarray(0.1),
    }


@pytest.mark.parametrize("agn_model", ["simple", "standard"])
@pytest.mark.parametrize("dust_law", ["power_law", "calzetti", "smc", "cardelli"])
@pytest.mark.parametrize("emission_model", ["modified_blackbody", "casey2012"])
def test_full_chain_composability(
    ssp, full_chain_params, agn_model, dust_law, emission_model
):
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

    chain = [
        StellarSEDComponent(ssp_data=ssp),
        NebularSEDComponent(config=NebularSEDComponentConfig(backend="baked_in")),
        AGNSEDComponent(config=AGNSEDComponentConfig(model=agn_model)),
        DustSEDComponent(
            config=DustSEDComponentConfig(
                law_bc=dust_law, law_diff=dust_law, emission_model=emission_model
            )
        ),
        RadioSEDComponent(),
        XRaySEDComponent(),
        IGMSEDComponent(),
    ]
    state0 = PipelineState(
        wave=ssp.ssp_wave, sed_observed=jnp.ones(len(ssp.ssp_wave))
    )

    s_eager = run_components(chain, state0, full_chain_params)
    s_jit = jax.jit(lambda p: run_components(chain, state0, p))(full_chain_params)

    assert bool(jnp.all(jnp.isfinite(s_eager.sed_intrinsic)))
    assert jnp.allclose(s_jit.sed_intrinsic, s_eager.sed_intrinsic, rtol=1e-12)
    # Cross-component publications all populated:
    for k in ("L_ir", "L_agn_bol", "L_radio", "L_xray", "igm_transmission"):
        assert k in s_eager.derived
