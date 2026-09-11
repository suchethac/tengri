# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for the inclination-dependent torus screen (#294).

A dusty torus obscures the AGN central engine (disc + broad/narrow lines +
FeII) along edge-on (Type-2) sightlines, while its own IR emission is not
re-extinguished by that screen. These tests pin the screen's behavior:

* face-on (Type-1) sightlines see ~unit transmission — so a default-inclination
  model is unchanged (the screen is opt-in via inclination, never a silent
  multiplicative hit);
* edge-on (Type-2) sightlines suppress the rest-UV continuum;
* the suppression is wavelength-dependent (SMC curve: bluer = more extinction);
* the screen is differentiable everywhere (gradient-safe for VI/HMC);
* only genuine dusty-torus blocks (skirtor, fritz) carry the screen.

Markers (see tests/TESTING.md)
------------------------------
- ``@pytest.mark.contract`` — pins the screen's transmission contract
- ``@pytest.mark.gradient`` — gradient finiteness across the Type-1/2 edge
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.agn.blocks.torus_screen import (
    TORUS_SCREEN_PARAMS,
    torus_screen_transmission,
)

# cos(30 deg) — the runner's default inclination (a clearly face-on sightline).
COS_INC_DEFAULT = 0.86602540378443864
# A wavelength grid spanning rest-UV (1000 A) to mid-IR (1e5 A).
WAVE = np.logspace(3.0, 5.0, 200)


def _uv(trans: np.ndarray) -> float:
    """Transmission near rest-frame 1500 A (UV continuum)."""
    return float(trans[np.argmin(np.abs(WAVE - 1500.0))])


def _ir(trans: np.ndarray) -> float:
    """Transmission near rest-frame 5 um (well redward of the optical)."""
    return float(trans[np.argmin(np.abs(WAVE - 5.0e4))])


@pytest.mark.contract
def test_face_on_transmission_is_unity():
    """Face-on (default inclination) leaves the central engine essentially
    unscreened, even at a large equatorial optical depth — the screen must
    never silently dim a Type-1 model."""
    trans = np.asarray(
        torus_screen_transmission(WAVE, cos_inc=COS_INC_DEFAULT, oa_deg=40.0, tau_v=7.0)
    )
    assert np.all(trans <= 1.0 + 1e-9)
    assert _uv(trans) > 0.99
    assert _ir(trans) > 0.999


@pytest.mark.contract
def test_edge_on_suppresses_uv():
    """Edge-on (cos_inc < sin(oa)) drives the disc/UV continuum down by orders
    of magnitude at tau_v = 7."""
    trans = np.asarray(torus_screen_transmission(WAVE, cos_inc=0.0, oa_deg=40.0, tau_v=7.0))
    assert _uv(trans) < 1e-3
    # IR is far less extinguished than UV (SMC curve falls toward the red).
    assert _ir(trans) > _uv(trans)


@pytest.mark.contract
def test_zero_optical_depth_is_transparent():
    """tau_v = 0 is a no-op at every inclination and wavelength."""
    for cos_inc in (0.0, 0.5, COS_INC_DEFAULT, 1.0):
        trans = np.asarray(
            torus_screen_transmission(WAVE, cos_inc=cos_inc, oa_deg=40.0, tau_v=0.0)
        )
        np.testing.assert_allclose(trans, 1.0, atol=1e-12)


@pytest.mark.contract
def test_monotonic_in_inclination():
    """More face-on (larger cos_inc) ⇒ more transmission — the Type-1/Type-2
    edge is monotone, no overshoot."""
    cos_grid = np.linspace(0.0, 1.0, 21)
    uv_trans = np.array(
        [
            _uv(np.asarray(torus_screen_transmission(WAVE, c, oa_deg=40.0, tau_v=7.0)))
            for c in cos_grid
        ]
    )
    assert np.all(np.diff(uv_trans) >= -1e-9)


@pytest.mark.contract
def test_smc_vs_calzetti_law_selectable():
    """Both reddening curves are accepted and produce edge-on suppression."""
    smc = np.asarray(torus_screen_transmission(WAVE, 0.0, 40.0, 7.0, law="smc"))
    cal = np.asarray(torus_screen_transmission(WAVE, 0.0, 40.0, 7.0, law="calzetti"))
    assert _uv(smc) < 1.0 and _uv(cal) < 1.0
    assert np.all(np.isfinite(smc)) and np.all(np.isfinite(cal))


@pytest.mark.contract
def test_only_dusty_torus_blocks_carry_screen():
    """skirtor & fritz map to (opening-angle, V-band-tau) param names; toy
    blocks are deliberately absent so they get no screen."""
    assert set(TORUS_SCREEN_PARAMS) == {"skirtor", "fritz"}
    assert TORUS_SCREEN_PARAMS["skirtor"] == ("agn_oa_skirtor", "agn_tau_skirtor")
    assert TORUS_SCREEN_PARAMS["fritz"] == ("agn_fritz_oa", "agn_fritz_tau")
    for toy in ("two_temperature", "simple", "grahsp"):
        assert toy not in TORUS_SCREEN_PARAMS


@pytest.mark.gradient
def test_gradient_finite_across_edge():
    """The sigmoid edge makes the screen differentiable in cos_inc and tau_v
    everywhere, including right at the Type-1/Type-2 transition."""

    def uv_loss_cos(cos_inc):
        trans = torus_screen_transmission(jnp.asarray(WAVE), cos_inc, 40.0, 7.0)
        return jnp.sum(trans)

    def uv_loss_tau(tau_v):
        trans = torus_screen_transmission(jnp.asarray(WAVE), 0.0, 40.0, tau_v)
        return jnp.sum(trans)

    # sin(40 deg) ~ 0.643 — sample on and around the edge.
    for c in (0.0, 0.6428, 0.9):
        g = jax.grad(uv_loss_cos)(c)
        assert np.isfinite(float(g))
        assert np.any(float(g) != 0.0), (
            "`float(g)` is identically zero — finite is not enough, "
            "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
        )
    assert np.isfinite(float(jax.grad(uv_loss_tau)(7.0)))
