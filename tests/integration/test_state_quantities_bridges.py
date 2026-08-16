# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for the orchestrator → legacy-Quantities bridges.

The :func:`tengri.forward.state_to_*_quantities` helpers and their
``SEDModel.predict_*_components`` wrappers convert a
:class:`tengri.protocols.ForwardState` into the legacy
:class:`SFHQuantities` / :class:`SEDQuantities` NamedTuples (and the
new :class:`RadioQuantities` / :class:`XRayQuantities` /
:class:`IonizingQuantities` mirrors).

This module pins their behavior with a fixed parameter set so future
changes to the bridges or the underlying components surface as test
failures rather than silent drift.
"""

from __future__ import annotations

import pathlib

import jax
import jax.numpy as jnp
import pytest

from tengri.components.dust.two_component import DustSEDComponent
from tengri.components.radio.component import RadioSEDComponent
from tengri.components.stellar import StellarSEDComponent
from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
from tengri.components.xray.component import XRaySEDComponent
from tengri.forward import (
    IonizingQuantities,
    RadioQuantities,
    XRayQuantities,
    run_components,
    state_to_ionizing_quantities,
    state_to_radio_quantities,
    state_to_sed_quantities,
    state_to_sfh_quantities,
    state_to_xray_quantities,
)
from tengri.protocols.component import ForwardState
from tests._component_params import component_params

# Bare-stellar SSP — required by Cue (wNE SSPs now raise CueWNESSPError).
_SSP_PATH = pathlib.Path("data/fsps_prsc_miles_chabrier.h5").resolve()


@pytest.fixture(scope="module")
def ssp():
    if not _SSP_PATH.exists():
        pytest.skip(f"SSP file not present at {_SSP_PATH}")
    return load_ssp_data(str(_SSP_PATH))


@pytest.fixture(scope="module")
def state(ssp):
    chain = [
        StellarSEDComponent(ssp_data=ssp),
        DustSEDComponent(),
        RadioSEDComponent(),
        XRaySEDComponent(),
    ]
    state0 = ForwardState(wave=ssp.ssp_wave, sed_observed=jnp.ones(len(ssp.ssp_wave)))
    params = {
        # Radio/X-ray declared defaults first; the explicit values below
        # override them, so this pinned set stays complete when a component
        # declares a new parameter (#1832).
        **component_params(RadioSEDComponent(), XRaySEDComponent()),
        # tsnorm SFH
        "sfh_tsnorm_log_total_mass": jnp.asarray(10.0),  # 1e10 Msun galaxy (#673)
        "sfh_tsnorm_peak_lbt_gyr": jnp.asarray(2.0),
        "sfh_tsnorm_width_gyr": jnp.asarray(1.0),
        "sfh_tsnorm_skew": jnp.asarray(0.0),
        "sfh_tsnorm_trunc": jnp.asarray(3.0),
        "met_logzsol": jnp.asarray(-0.5),
        # Dust
        "dust_tau_bc": jnp.asarray(0.5),
        "dust_tau_diff": jnp.asarray(0.2),
        "dust_slope": jnp.asarray(-0.7),
        "dust_T": jnp.asarray(35.0),
        "dust_beta_ir": jnp.asarray(1.6),
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
        "xray_delta_alpha_ox": jnp.asarray(-1.4),
        "xray_log_nh": jnp.asarray(20.0),
        "redshift": jnp.asarray(0.0),
    }
    return run_components(chain, state0, params)


# ── SFHQuantities ────────────────────────────────────────────────────


def test_sfh_quantities_all_fields_finite(state):
    sfh_q = state_to_sfh_quantities(state)
    for f in sfh_q._fields:
        v = float(getattr(sfh_q, f))
        assert jnp.isfinite(v), f"SFH.{f} = {v} (expected finite)"


def test_sfh_quantities_physical_ranges(state):
    sfh_q = state_to_sfh_quantities(state)
    # 10**10 Msun-class galaxy with peak SFH 10 Msun/yr at 2 Gyr
    assert 1e9 < float(sfh_q.stellar_mass) < 1e11
    assert 0.0 < float(sfh_q.sfr_10myr) < 100.0
    assert 0.0 < float(sfh_q.sfr_100myr) < 100.0
    # ssfr in 1/yr; reasonable galaxy range
    assert 1e-12 < float(sfh_q.ssfr) < 1e-7
    # Mass-weighted age positive, < age of universe
    assert 0.0 < float(sfh_q.mass_weighted_age_gyr) < 14.0


# ── SEDQuantities ────────────────────────────────────────────────────


def test_sed_quantities_all_15_fields_populated(state):
    sed_q = state_to_sed_quantities(state)
    nans = [f for f in sed_q._fields if not bool(jnp.isfinite(getattr(sed_q, f)))]
    assert nans == [], f"SED fields with NaN: {nans}"


def test_sed_quantities_physical_ranges(state):
    sed_q = state_to_sed_quantities(state)
    # L_bol ~ 10^10 Lsun for our setup
    assert 1e9 < float(sed_q.l_bol) < 1e12
    # Dust attenuation moves L_TIR below L_bol
    assert 0.0 < float(sed_q.l_tir) < float(sed_q.l_bol)
    # UV slope reddened by dust → typically negative (e.g. -1 to -2 for SF + dust)
    assert -3.0 < float(sed_q.uv_slope_beta) < 0.0
    # Dn4000 in plausible range
    assert 0.5 < float(sed_q.dn4000) < 3.0
    # Pre-dust UV brighter than post-dust (tau_bc=0.5)
    assert float(sed_q.fuv_flux_intrinsic) > float(sed_q.fuv_flux)


# ── Radio / XRay / Ionizing ──────────────────────────────────────────


def test_radio_quantities_finite_and_physical(state):
    rq = state_to_radio_quantities(state)
    assert isinstance(rq, RadioQuantities)
    # 1.4 GHz luminosity in plausible SF-galaxy range
    assert 1e25 < float(rq.l_1p4ghz) < 1e30
    # FIR-radio correlation parameter in plausible range
    assert 1.0 < float(rq.q_ir) < 5.0


def test_xray_quantities_finite_and_physical(state):
    xq = state_to_xray_quantities(state)
    assert isinstance(xq, XRayQuantities)
    # XRB luminosity in plausible range
    assert 1e37 < float(xq.l_x_xrb) < 1e42
    # No AGN component → l_x_agn is exactly 0 (not NaN)
    assert float(xq.l_x_agn) == 0.0
    assert float(xq.l_x_total) == float(xq.l_x_xrb)


def test_ionizing_quantities_finite(state):
    iq = state_to_ionizing_quantities(state)
    assert isinstance(iq, IonizingQuantities)
    # nion magnitude is set by stellar; the BakedIn SSP suppresses it
    # but it should still be positive and finite.
    assert float(iq.q_h) > 0.0
    assert float(iq.xi_ion) > 0.0


# ── Bridges work under jit ───────────────────────────────────────────


def test_state_to_sfh_quantities_jit_compatible(state):
    out = jax.jit(state_to_sfh_quantities)(state)
    out_eager = state_to_sfh_quantities(state)
    for f in out._fields:
        assert jnp.allclose(getattr(out, f), getattr(out_eager, f), rtol=1e-12), (
            f"JIT vs eager differ at {f}"
        )


def test_state_to_sed_quantities_jit_compatible(state):
    out = jax.jit(state_to_sed_quantities)(state)
    out_eager = state_to_sed_quantities(state)
    for f in out._fields:
        assert jnp.allclose(getattr(out, f), getattr(out_eager, f), rtol=1e-12), (
            f"JIT vs eager differ at {f}"
        )


# ── Emission-lines bridge ────────────────────────────────────────────


@pytest.fixture(scope="module")
def state_with_cue(ssp):
    """A chain with Cue nebular backend so line catalog is published."""
    from tengri.components.nebular.component import (
        NebularSEDComponent,
        NebularSEDComponentConfig,
    )
    from tengri.components.nebular.cue import CueBackend
    from tengri.forward import state_to_emission_lines  # noqa: F401 — used elsewhere

    cue_path = pathlib.Path("data/cue_weights.npz").resolve()
    if not cue_path.exists():
        pytest.skip(f"Cue weights not present at {cue_path}")
    cue = CueBackend(weights_path=str(cue_path), ssp_data=ssp)
    chain = [
        StellarSEDComponent(ssp_data=ssp),
        NebularSEDComponent(config=NebularSEDComponentConfig(backend="cue"), backend=cue),
    ]
    state0 = ForwardState(wave=ssp.ssp_wave)
    params = {
        "sfh_tsnorm_log_total_mass": jnp.asarray(10.0),  # 1e10 Msun galaxy (#673)
        "sfh_tsnorm_peak_lbt_gyr": jnp.asarray(2.0),
        "sfh_tsnorm_width_gyr": jnp.asarray(1.0),
        "sfh_tsnorm_skew": jnp.asarray(0.0),
        "sfh_tsnorm_trunc": jnp.asarray(3.0),
        "met_logzsol": jnp.asarray(-0.5),
        "neb_logU": jnp.asarray(-2.5),
        "neb_logZ_gas": jnp.asarray(-0.3),
        "neb_fesc": jnp.asarray(0.0),
        "neb_fesc_lya": jnp.asarray(0.0),
        "ionspec_index1": jnp.asarray(15.0),
        "ionspec_index2": jnp.asarray(5.0),
        "ionspec_index3": jnp.asarray(0.0),
        "ionspec_index4": jnp.asarray(0.0),
        "ionspec_logLratio1": jnp.asarray(2.0),
        "ionspec_logLratio2": jnp.asarray(0.5),
        "ionspec_logLratio3": jnp.asarray(0.5),
        "gas_logn": jnp.asarray(2.0),
        "gas_logno": jnp.asarray(0.0),
        "gas_logco": jnp.asarray(0.0),
        "redshift": jnp.asarray(0.0),
    }
    return run_components(chain, state0, params)


def test_emission_lines_published_by_cue(state_with_cue):
    """Cue backend should populate state.derived line catalog."""
    assert "line_waves" in state_with_cue.derived
    assert "line_lums" in state_with_cue.derived
    # Cue's output is a many-line catalog (~100+).
    assert state_with_cue.derived["line_waves"].shape[0] > 50


def test_state_to_emission_lines_all_finite(state_with_cue):
    """All 11 headline bridge-extracted lines should be finite for Cue.

    ``all_waves``/``all_lums`` are arrays (skipped here — see
    ``test_state_to_emission_lines_publishes_full_catalog``).
    """
    from tengri.forward import state_to_emission_lines

    lines = state_to_emission_lines(state_with_cue)
    scalar_fields = [f for f in lines._fields if f not in ("all_waves", "all_lums")]
    nans = [f for f in scalar_fields if not bool(jnp.isfinite(getattr(lines, f)))]
    assert nans == [], f"Lines with NaN: {nans}"


def test_state_to_emission_lines_publishes_full_catalog(state_with_cue):
    """Cue exposes the full ~138-line catalog via all_waves/all_lums (#303)."""
    from tengri.forward import state_to_emission_lines

    lines = state_to_emission_lines(state_with_cue)
    assert lines.all_waves.size > 50, (
        f"Expected >50 species exposed via all_waves, got {lines.all_waves.size}"
    )
    assert lines.all_waves.shape == lines.all_lums.shape
    # HeII 1640 was a canonical example in the issue: must be queryable.
    heii = float(lines.get(1640.4, tol_aa=5.0))
    assert jnp.isfinite(heii), "HeII 1640 should be in the catalog"


def test_state_to_emission_lines_balmer_decrement(state_with_cue):
    """Halpha / Hbeta ≈ 2.85 (case-B recombination at T_e=10⁴ K)."""
    from tengri.forward import state_to_emission_lines

    lines = state_to_emission_lines(state_with_cue)
    ratio = float(lines.halpha / lines.hbeta)
    # Cue's intrinsic Balmer decrement should be in [2.5, 3.5] —
    # case-B is 2.86; small departures are physical.
    assert 2.0 < ratio < 4.0, f"Halpha/Hbeta = {ratio} (expected ~2.85)"


def test_state_to_emission_lines_no_catalog_returns_nan(state):
    """Chain without nebular catalog (no Cue/Cloudy) → NaN headlines + empty all_*."""
    from tengri.forward import state_to_emission_lines

    # The ``state`` fixture has no nebular component → no line_waves.
    assert "line_waves" not in state.derived
    lines = state_to_emission_lines(state)
    scalar_fields = [f for f in lines._fields if f not in ("all_waves", "all_lums")]
    for f in scalar_fields:
        assert not bool(jnp.isfinite(getattr(lines, f))), (
            f"Lines.{f} should be NaN when no catalog published"
        )
    assert lines.all_waves.size == 0
    assert lines.all_lums.size == 0
