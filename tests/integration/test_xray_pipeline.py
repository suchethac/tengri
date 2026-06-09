# SPDX-License-Identifier: BSD-3-Clause
"""Numerical-equivalence tests for the X-ray Phase II-1 adapter.

Asserts that running ``[XRaySEDComponent]`` through
:func:`tengri.forward.orchestrator.run_components` produces the same
SED as calling :func:`xray_total` directly. Holds the orchestrator +
adapter honest as a pair: any future Protocol revision that breaks
this contract is caught here, not in a science fit weeks later.

Mirrors :mod:`tests.integration.test_radio_igm_pipeline` so the two
files are easy to diff side-by-side.
"""

from __future__ import annotations

import chex
import jax.numpy as jnp
import pytest

from tengri.components.xray.component import XRaySEDComponent
from tengri.components.xray.xray import xray_total
from tengri.forward.orchestrator import run_components
from tengri.protocols import ForwardState

REL_TOL = 1e-10


@pytest.mark.parametrize(
    ("z", "sfr", "stellar_mass", "l_2500", "delta_alpha_ox"),
    [
        (0.1, 1.0, 1e10, 0.0, 0.0),  # XRB-only, no AGN (l_2500 = 0)
        (0.5, 5.0, 5e10, 1e29, 0.0),  # XRB + AGN corona, self-consistent alpha_ox
        (2.0, 50.0, 1e11, 1e31, -0.2),  # powerful AGN, harder UV-X coupling
        (4.0, 0.1, 1e9, 0.0, 0.0),  # quiescent low-mass, no AGN
        (6.0, 10.0, 1e10, 5e29, 0.3),  # AGN at z=6, softer coupling
    ],
)
def test_orchestrator_matches_direct_call(z, sfr, stellar_mass, l_2500, delta_alpha_ox):
    """Pipeline output equals direct xray_total call (corona driven by L_2500)."""
    wave = jnp.linspace(1e0, 1e3, 1024)  # X-ray range: 1 to 1000 Å

    initial_state = ForwardState(
        wave=wave,
        sed_intrinsic=jnp.zeros_like(wave),
        derived={"sfr": sfr, "log_mstar": jnp.log10(stellar_mass), "l_2500": l_2500},
    )

    params = {
        "redshift": z,
        "xray_gamma_hmxb": 2.0,
        "xray_gamma_lmxb": 1.6,
        "xray_gamma_agn": 1.8,
        "xray_E_cut": 300.0,
        "xray_delta_alpha_ox": delta_alpha_ox,
    }

    final = run_components([XRaySEDComponent()], initial_state, params)

    expected = xray_total(
        wave,
        sfr=sfr,
        stellar_mass=stellar_mass,
        l_2500_30deg=l_2500,
        gamma_hmxb=2.0,
        gamma_lmxb=1.6,
        gamma_agn=1.8,
        E_cut=300.0,
        delta_alpha_ox=delta_alpha_ox,
    )

    assert jnp.allclose(final.sed_intrinsic, expected, rtol=REL_TOL, atol=0.0)
    # Adapter must publish sed_xray for downstream readers.
    assert "sed_xray" in final.derived
    assert jnp.allclose(final.derived["sed_xray"], expected, rtol=REL_TOL, atol=0.0)


def test_corona_scales_with_l2500_regression_746():
    """#746: the AGN X-ray corona was silently dropped.

    The component passed ``L_agn_bol=`` / ``alpha_ox=`` kwargs that
    ``xray_total`` does not accept (swallowed by ``**_kwargs``), so
    ``l_2500_30deg`` was never set and the corona contributed nothing —
    ``sed_xray`` was byte-identical regardless of AGN luminosity. The fix
    wires the AGN-published ``l_2500`` into ``l_2500_30deg``; the corona must
    now scale with it. Corona normalisation: Just+2007 alpha_ox relation,
    ``L_2keV = L_2500 * 10**(alpha_ox / 0.3838)`` (Yang+2020, MNRAS 491, 740).
    """
    wave = jnp.linspace(1e0, 1e3, 1024)  # Å; 2 keV ≈ 6.2 Å sits inside
    params = {
        "redshift": 0.0,
        "xray_gamma_hmxb": 2.0,
        "xray_gamma_lmxb": 1.6,
        "xray_gamma_agn": 1.8,
        "xray_E_cut": 300.0,
        "xray_delta_alpha_ox": 0.0,
    }

    def _sed_xray(l_2500):
        state = ForwardState(
            wave=wave,
            sed_intrinsic=jnp.zeros_like(wave),
            derived={"sfr": 1.0, "log_mstar": 10.0, "l_2500": l_2500},
        )
        return run_components([XRaySEDComponent()], state, params).derived["sed_xray"]

    i_2kev = int(jnp.argmin(jnp.abs(wave - 6.2)))
    none = _sed_xray(0.0)  # no AGN -> XRB + hot gas only
    faint = _sed_xray(1e29)
    bright = _sed_xray(1e31)  # 100x brighter disc UV

    # Corona is present and scales with L_2500. At the default delta=0 the
    # Just+2007 relation steepens alpha_ox with luminosity, so a 100x rise in
    # L_2500 gives a ~19x (sub-linear) corona rise — but it is decisively NOT
    # the byte-identical 1.0x of the #746 bug.
    assert bright[i_2kev] > 10.0 * faint[i_2kev]
    # With no AGN the corona vanishes (only XRB/hot-gas remain).
    assert none[i_2kev] < faint[i_2kev]


def test_xray_no_agn_upstream_falls_back_to_zero():
    """Without AGN upstream, X-ray uses its documented fallback (L_agn_bol=0)."""
    wave = jnp.linspace(1e0, 1e3, 64)
    state = ForwardState(wave=wave, sed_intrinsic=jnp.zeros_like(wave))
    xray = XRaySEDComponent()

    params = {
        "redshift": 0.0,
        "xray_gamma_hmxb": 2.0,
        "xray_gamma_lmxb": 1.6,
        "xray_gamma_agn": 1.8,
        "xray_E_cut": 300.0,
        "xray_delta_alpha_ox": 0.0,
    }
    out = xray.apply(state, params)

    expected = xray_total(
        wave,
        sfr=1.0,
        stellar_mass=1e10,
        l_2500_30deg=0.0,
        gamma_hmxb=2.0,
        gamma_lmxb=1.6,
        gamma_agn=1.8,
        E_cut=300.0,
        delta_alpha_ox=0.0,
    )
    assert jnp.allclose(out.sed_intrinsic, expected, rtol=REL_TOL, atol=0.0)


def test_xray_pipeline_preserves_input_state_immutability():
    """The orchestrator must not mutate the input ForwardState."""
    wave = jnp.linspace(1e0, 1e3, 64)
    initial = ForwardState(
        wave=wave,
        sed_intrinsic=jnp.zeros_like(wave),
        derived={"sfr": 1.0, "log_mstar": 10.0, "L_agn_bol": 1e44},
    )
    snapshot_intrinsic = initial.sed_intrinsic
    snapshot_derived = dict(initial.derived)

    _ = run_components(
        [XRaySEDComponent()],
        initial,
        {
            "redshift": 1.0,
            "xray_gamma_hmxb": 2.0,
            "xray_gamma_lmxb": 1.6,
            "xray_gamma_agn": 1.8,
            "xray_E_cut": 300.0,
            "xray_delta_alpha_ox": 0.0,
        },
    )

    assert jnp.array_equal(initial.sed_intrinsic, snapshot_intrinsic)
    # ``initial.derived`` is a DerivedState (not a plain dict) since the
    # state-bundle refactor; compare via dict() so the immutability check
    # actually exercises content equality rather than DerivedState-vs-dict
    # type inequality (#673).
    assert dict(initial.derived) == snapshot_derived


def test_three_adapter_chain_runs_end_to_end():
    """Radio + IGM + X-ray composed in a single pipeline.

    Covers the case where multiple additive emission components write to
    ``sed_intrinsic`` and IGM applies to ``sed_observed`` separately.
    Doesn't assert on numerical values (those are covered by the
    single-adapter tests) — just that the chain runs and produces a
    finite, non-trivial state.
    """
    from tengri.components.igm.component import IGMSEDComponent
    from tengri.components.radio.component import RadioSEDComponent

    # Log-spaced grid so UV/Lyα-region points exist (where IGM attenuates).
    wave = jnp.logspace(0, 9, 256)
    state = ForwardState(
        wave=wave,
        sed_intrinsic=jnp.zeros_like(wave),
        sed_observed=jnp.ones_like(wave) * 1e30,
        derived={
            "L_ir": 1e44,
            "L_agn_bol": 1e44,
            "log_mstar": 10.5,  # XRay reads this, exponentiates internally
            "sfr": 5.0,
        },
    )

    # Use z high enough that IGM produces non-trivial attenuation
    # (reionization midpoint igm_z_mid=7.0 → galaxies AT z=8 see real
    # transmission < 1).
    params = {
        "redshift": 8.0,
        # radio
        "radio_q_ir": 2.64,
        "radio_alpha_sf": 0.8,
        "radio_loudness": 0.0,
        "radio_alpha_agn": 0.7,
        "radio_T_e": 1e4,
        "radio_alpha_ff": -0.1,
        # igm
        "igm_z_mid": 7.0,
        "igm_dz": 0.5,
        "igm_log_nhi": 20.0,
        # xray
        "xray_gamma_hmxb": 2.0,
        "xray_gamma_lmxb": 1.6,
        "xray_gamma_agn": 1.8,
        "xray_E_cut": 300.0,
        "xray_delta_alpha_ox": 0.0,
    }

    final = run_components(
        [RadioSEDComponent(), XRaySEDComponent(), IGMSEDComponent()],
        state,
        params,
    )

    assert final.sed_intrinsic is not None
    chex.assert_tree_all_finite(final.sed_intrinsic)
    assert "sed_radio" in final.derived
    assert "sed_xray" in final.derived
    # IGM transmission at z=8 (above the z_mid=7 reionization midpoint)
    # must reduce sed_observed at rest-frame Lyα-blue wavelengths.
    assert jnp.any(final.sed_observed < state.sed_observed)
