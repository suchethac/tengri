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
from tengri.forward.orchestrator import default_params_dict, run_components
from tengri.protocols import ForwardState

REL_TOL = 1e-10


@pytest.mark.parametrize(
    ("z", "sfr", "stellar_mass", "L_agn_bol"),
    [
        (0.1, 1.0, 1e10, 0.0),  # XRB-only, low-z star-forming
        (0.5, 5.0, 5e10, 1e44),  # XRB + AGN corona, intermediate-z
        (2.0, 50.0, 1e11, 1e46),  # high-SFR + powerful AGN
        (4.0, 0.1, 1e9, 0.0),  # quiescent low-mass at high-z
        (6.0, 10.0, 1e10, 5e44),  # AGN-dominated at z=6
    ],
)
def test_orchestrator_matches_direct_call(z, sfr, stellar_mass, L_agn_bol):
    """Pipeline output equals direct xray_total call."""
    wave = jnp.linspace(1e0, 1e3, 1024)  # X-ray range: 1 to 1000 Å

    initial_state = ForwardState(
        wave=wave,
        sed_intrinsic=jnp.zeros_like(wave),
        derived={"sfr": sfr, "log_mstar": jnp.log10(stellar_mass), "L_agn_bol": L_agn_bol},
    )

    # Declared defaults, with the one deliberate departure spelled out. The
    # literal this replaces named six xray_* keys and was complete when written;
    # it broke the day xray_det_hmxb gained a reader (#1832).
    params = default_params_dict(
        [XRaySEDComponent()],
        overrides={"redshift": z, "xray_delta_alpha_ox": -1.4},
    )

    final = run_components([XRaySEDComponent()], initial_state, params)

    # The component derives l_2500 from L_agn_bol via the Hopkins+2007 BC
    # fallback (no L_2500_intrinsic / L_2500_30deg in this minimal state) and
    # passes the xray_delta_alpha_ox offset through as delta_alpha_ox. Replicate
    # both so the equivalence contract holds (#722/#746: the old expected used
    # L_agn_bol=/alpha_ox= kwargs that xray_total silently swallows).
    #
    # Read the spectral parameters off ``params`` rather than repeating them:
    # this test is about the orchestrator agreeing with the direct call, not
    # about any particular photon index, and a second literal is a second thing
    # to forget.
    l_2500 = L_agn_bol / (5.15 * 1.199e15)
    expected = xray_total(
        wave,
        sfr=sfr,
        stellar_mass=stellar_mass,
        l_2500_30deg=l_2500,
        gamma_hmxb=float(params["xray_gamma_hmxb"]),
        gamma_lmxb=float(params["xray_gamma_lmxb"]),
        gamma_agn=float(params["xray_gamma_agn"]),
        E_cut=float(params["xray_E_cut"]),
        delta_alpha_ox=float(params["xray_delta_alpha_ox"]),
    )

    assert jnp.allclose(final.sed_intrinsic, expected, rtol=REL_TOL, atol=0.0)
    # Adapter must publish sed_xray for downstream readers.
    assert "sed_xray" in final.derived
    assert jnp.allclose(final.derived["sed_xray"], expected, rtol=REL_TOL, atol=0.0)


def test_xray_no_agn_upstream_falls_back_to_zero():
    """Without AGN upstream, X-ray uses its documented fallback (L_agn_bol=0)."""
    wave = jnp.linspace(1e0, 1e3, 64)
    state = ForwardState(wave=wave, sed_intrinsic=jnp.zeros_like(wave))
    xray = XRaySEDComponent()

    params = default_params_dict(
        [xray],
        overrides={"redshift": 0.0, "xray_delta_alpha_ox": -1.4},
    )
    out = xray.apply(state, params)

    expected = xray_total(
        wave,
        sfr=1.0,
        stellar_mass=1e10,
        L_agn_bol=0.0,
        gamma_hmxb=2.0,
        gamma_lmxb=1.6,
        gamma_agn=1.8,
        E_cut=300.0,
        alpha_ox=-1.4,
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
        default_params_dict(
            [XRaySEDComponent()],
            overrides={"redshift": 1.0, "xray_delta_alpha_ox": -1.4},
        ),
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
    chain = [RadioSEDComponent(), XRaySEDComponent(), IGMSEDComponent()]

    # Everything at its declared default except the four the test chose
    # deliberately: z=8 above the reionization midpoint, and the IGM knobs the
    # comment above pins.
    params = default_params_dict(
        chain,
        overrides={
            "redshift": 8.0,
            "igm_z_mid": 7.0,
            "igm_dz": 0.5,
            "xray_delta_alpha_ox": -1.4,
        },
    )

    final = run_components(chain, state, params)

    assert final.sed_intrinsic is not None
    chex.assert_tree_all_finite(final.sed_intrinsic)
    assert "sed_radio" in final.derived
    assert "sed_xray" in final.derived
    # IGM transmission at z=8 (above the z_mid=7 reionization midpoint)
    # must reduce sed_observed at rest-frame Lyα-blue wavelengths.
    assert jnp.any(final.sed_observed < state.sed_observed)
