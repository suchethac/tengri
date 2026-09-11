# SPDX-License-Identifier: BSD-3-Clause
"""Intrinsic nebular emission-line broadening (velocity triweight profiles).

Tengri renders nebular emission lines with an intrinsic velocity-width triweight
profile (``neb_eline_sigma_kms``, default 100 km/s) rather than delta functions —
matching Prospector's ``eline_sigma`` treatment. These tests lock the renderer's
physics (flux conservation, velocity scaling, compact support), its JIT/gradient
safety (the width is a *fittable* parameter), and the parameter wiring.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.nebular._shared import (
    place_line_profiles_velocity,
    render_nebular_lines,
)
from tengri.utils.physics_constants import C_CGS, C_KM_S

pytestmark = pytest.mark.contract

_LW = jnp.array([4861.0, 5007.0, 6563.0])  # Hβ, [OIII], Hα
_LL = jnp.array([1.0, 3.0, 2.5])


def _integrate_nu(wave, sed):
    nu = C_CGS / (np.asarray(wave) * 1e-8)
    o = np.argsort(nu)
    return np.trapezoid(np.asarray(sed)[o], nu[o])


@pytest.mark.conservation
def test_velocity_profile_conserves_flux_across_widths():
    """∫ profile dν == Σ line_lum, independent of the line width."""
    wave = jnp.geomspace(3000.0, 9000.0, 30000)  # all three lines well inside
    totals = [
        _integrate_nu(wave, place_line_profiles_velocity(_LW, _LL, wave, s))
        for s in (50.0, 100.0, 300.0)
    ]
    for t in totals:
        assert t == pytest.approx(float(_LL.sum()), rel=2e-3)
    # width-independence
    assert totals[0] == pytest.approx(totals[2], rel=2e-3)


@pytest.mark.bounds
def test_velocity_scaling_equal_fwhm_in_kms():
    """A constant km/s width gives every line the same velocity FWHM (Δλ ∝ λ)."""
    sig = 150.0
    wave = jnp.linspace(4700.0, 6700.0, 60000)
    sed = np.asarray(place_line_profiles_velocity(_LW, _LL, wave, sig))
    w = np.asarray(wave)

    def fwhm_kms(c):
        m = (w > c - 60) & (w < c + 60)
        ww, seg = w[m], sed[m]
        above = ww[seg >= seg.max() / 2]
        return (above.max() - above.min()) / c * C_KM_S

    f_hb, f_ha = fwhm_kms(4861.0), fwhm_kms(6563.0)
    # Equal in velocity space (grid-limited tolerance); triweight FWHM ≈ 2.72σ.
    assert f_hb == pytest.approx(f_ha, rel=0.05)
    assert f_ha == pytest.approx(2.72 * sig, rel=0.1)


@pytest.mark.bounds
def test_compact_support_zero_far_from_lines():
    """Triweight has finite support: zero well away from any line."""
    wave = jnp.linspace(3900.0, 4100.0, 500)  # no line within (nearest is Hβ 4861)
    sed = np.asarray(place_line_profiles_velocity(_LW, _LL, wave, 100.0))
    assert np.all(sed == 0.0)


@pytest.mark.gradient
def test_velocity_profile_jit_and_grad_safe():
    """The width may be a *traced* (fittable) parameter under jax.jit / jax.grad."""
    wave = jnp.linspace(4800.0, 5100.0, 4000)
    out = jax.jit(lambda s: place_line_profiles_velocity(_LW, _LL, wave, s))(100.0)
    assert np.all(np.isfinite(np.asarray(out)))
    g = jax.jit(jax.grad(lambda s: jnp.sum(place_line_profiles_velocity(_LW, _LL, wave, s))))(
        100.0
    )
    assert np.isfinite(g)
    assert np.any(g != 0.0), (
        "`g` is identically zero — finite is not enough, "
        "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
    )


def test_zero_width_floors_to_finite_profile():
    """σ→0 floors to a ~1-pixel profile: finite (no NaN/Inf), positive flux.

    The floored degenerate width is not flux-exact on a variable (geomspace)
    grid — that is the σ=0 corner. The guarantee is finiteness; flux exactness
    holds for the realistic σ ≳ grid-resolution regime (see the conservation
    test, which checks σ = 50/100/300 km/s).
    """
    wave = jnp.geomspace(3000.0, 9000.0, 30000)
    sed = place_line_profiles_velocity(_LW, _LL, wave, 0.0)
    assert np.all(np.isfinite(np.asarray(sed)))
    assert _integrate_nu(wave, sed) > 0.0


def test_render_dispatch_prefers_velocity_then_aa():
    """render_nebular_lines: velocity by default; legacy Å Gaussian when set."""
    wave = jnp.linspace(4800.0, 5100.0, 4000)
    vel = np.asarray(render_nebular_lines(_LW, _LL, wave, 0.0, 100.0))
    aa = np.asarray(render_nebular_lines(_LW, _LL, wave, 5.0, 0.0))  # legacy Å Gaussian path
    # Both render real lines but with different profiles → different peak heights.
    assert vel.max() > 0 and aa.max() > 0
    # atol=0: the absolute luminosities are tiny (~1e-12), so the default atol
    # would mask a real difference in peak height.
    assert not np.isclose(vel.max(), aa.max(), rtol=0.1, atol=0.0)


def test_param_registers_with_nebular_fittable():
    """neb_eline_sigma_kms registers with nebular (default 100), is fittable, off otherwise."""
    import tengri
    from tengri.parameters.parameters import Parameters

    p = Parameters(mean_sfh_type="dpl", nebular_cue=True)
    assert "neb_eline_sigma_kms" in p.all_params
    assert float(p.get_distribution("neb_eline_sigma_kms").sample(jax.random.PRNGKey(0))) == 100.0

    p_free = Parameters(
        mean_sfh_type="dpl", nebular_cue=True, neb_eline_sigma_kms=tengri.Uniform(30.0, 300.0)
    )
    assert "neb_eline_sigma_kms" in p_free.free_params

    p_off = Parameters(mean_sfh_type="dpl")
    assert "neb_eline_sigma_kms" not in p_off.all_params
