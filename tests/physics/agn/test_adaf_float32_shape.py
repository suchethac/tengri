# SPDX-License-Identifier: BSD-3-Clause
"""ADAF spectrum shape agrees between pure float32 and float64 over the declared box (#2783).

Conservation of the normalized shape: the spectrum :math:`L_\\nu` is normalized to the
target power, so its *shape* is what must not depend on the working precision. Before the
fix the float32 solve of the synchrotron self-absorption parameter ``x_M`` overflowed at the
clipped-accretion-rate corner (``alpha = 0.5``, ``T_e`` at its floor), and the spectrum
differed from float64 by up to 0.997 relative. Pure float32 is a supported mode, so the
shape must agree at every point of the declared parameter box.

Grid
----
Every axis is ``numpy.linspace`` over the declared prior bounds of
:mod:`tengri.components.agn._params`, so both declared edges are always included. The point
counts are chosen so that the step is a round number (``alpha`` in steps of 0.05, so
``alpha = 0.1`` is a grid point; ``log L_bol`` in steps of 2, so the declared lower edge 8 is).

Tolerance derivation
--------------------
Each float32 spectral point is ``exp(a)`` of a sum of logs, and the relative error of the
point is the absolute error of the log argument. Two terms set the measured budget:

1. **Level error of the logarithms.** ``log_numer`` and ``log_integral`` are natural logs of
   erg/s powers, :math:`|\\ln P| \\approx 80\\text{--}110` across the box. Each carries
   :math:`\\varepsilon\\,|\\ln P| \\approx 6\\times10^{-8}\\times 110 \\approx 7\\times10^{-6}`
   of float32 rounding, and their difference is the level error of the spectrum.
2. **Cancellation in :math:`\\ln \\tau_{\\rm es}`.** The Compton slope :math:`\\alpha_c` is
   obtained from :math:`\\ln\\tau_{\\rm es}`, with :math:`\\tau_{\\rm es}` near 1 on the
   clipped-accretion-rate edge (``alpha = 0.5``, :math:`\\tau_{\\rm es} = 1.0023`). The
   float32 error is
   :math:`\\varepsilon/|\\ln\\tau_{\\rm es}| \\approx 6\\times10^{-8}/2.3\\times10^{-3}
   \\approx 2.6\\times10^{-5}`, and it enters the shape through :math:`\\alpha_c\\ln(\\nu/\\nu_p)`.

The worst grid point therefore has a shape error of about :math:`7\\times10^{-6} +
2.6\\times10^{-5} \\approx 3.3\\times10^{-5}`. The tolerance :math:`\\mathrm{RTOL} = 10^{-4}` is
about three times that budget. The term in 2 is bounded only because
:math:`\\tau_{\\rm es}` stays away from 1 on this grid: for :math:`\\alpha \\to 0.4988` at the
clipped rate it is unbounded. That corner is governed by the Eq. 43 temperature branch
(a separate issue), not by this tolerance.

Below the significance floor the tolerance is taken relative to the floor, so every point
uses the same formula: ``|f32 - f64| <= RTOL * max(|f64|, FLOOR * peak)``.

Physical bound: ADAF spectrum, Mahadevan (1997, ApJ 477, 585) Eqs. 21-23, 28, 30 and the
normalization of :func:`tengri.components.agn.adaf.adaf_scalar_state`.
"""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.agn import adaf as adaf_mod
from tengri.components.agn._params import PARAMS

pytestmark = pytest.mark.regression_bug


def _declared_bounds(name: str) -> tuple[float, float]:
    """Return the declared ``(lo, hi)`` of a parameter's Uniform prior."""
    (decl,) = [p for p in PARAMS if p.name == name]
    return tuple(float(x) for x in decl.prior.bounds)


def _axis(name: str, n: int) -> tuple[float, ...]:
    lo, hi = _declared_bounds(name)
    return tuple(float(x) for x in np.linspace(lo, hi, n))


LOG_MBH = _axis("agn_log_mbh", 3)  # 6, 8, 10
ALPHA = _axis("agn_adaf_alpha", 10)  # 0.05, 0.10, ..., 0.50 (step 0.05)
BETA = _axis("agn_adaf_beta", 3)  # 0.1, 0.5, 0.9
DELTA = _axis("agn_adaf_delta", 3)  # 0.001, 0.2505, 0.5
SHAPE = _axis("agn_log_lbol", 4)  # 8, 10, 12, 14

# Reference luminosity (float32 AGN path) and wavelength grid [Angstrom] covering the
# support of the spectrum at every corner.
_REFERENCE_LOG_LBOL = 2.0
_WAVE_AA = np.logspace(-3, 9, 1500)

RTOL = 1e-4
FLOOR = 1e-6

_CORNERS = list(itertools.product(LOG_MBH, ALPHA, BETA, DELTA, SHAPE))


def _make_spectrum_fn():
    """Jitted ``(wave, log_mbh, alpha, beta, delta, shape) -> L_nu`` at the wave dtype."""

    def spectrum(wave, log_mbh, alpha, beta, delta, shape):
        state = adaf_mod.adaf_scalar_state(
            _REFERENCE_LOG_LBOL,
            1.0,
            log_mbh,
            alpha,
            beta,
            delta,
            shape,
            dtype=wave.dtype,
        )
        return adaf_mod.adaf_spectrum_from_state(wave, state)

    return jax.jit(spectrum)


@pytest.fixture(scope="module")
def spectrum_f64():
    fn = _make_spectrum_fn()
    wave = jnp.asarray(_WAVE_AA, dtype=jnp.float64)

    def run(log_mbh, alpha, beta, delta, shape):
        out = fn(wave, log_mbh, alpha, beta, delta, shape)
        return np.asarray(out, dtype=np.float64)

    return run


@pytest.fixture(scope="module")
def spectrum_f32():
    with jax.enable_x64(False):
        fn = _make_spectrum_fn()
        wave = jnp.asarray(_WAVE_AA, dtype=jnp.float32)

    def run(log_mbh, alpha, beta, delta, shape):
        with jax.enable_x64(False):
            f32 = jnp.float32
            out = fn(
                wave,
                f32(log_mbh),
                f32(alpha),
                f32(beta),
                f32(delta),
                f32(shape),
            )
            return np.asarray(out, dtype=np.float64)

    return run


def _ids(corner):
    log_mbh, alpha, beta, delta, shape = corner
    return f"mbh{log_mbh:g}-a{alpha:g}-b{beta:g}-d{delta:g}-s{shape:g}"


@pytest.mark.parametrize("corner", _CORNERS, ids=[_ids(c) for c in _CORNERS])
def test_adaf_shape_float32_matches_float64_on_box(corner, spectrum_f64, spectrum_f32):
    """Pure float32 spectrum agrees with float64 pointwise at every box grid point.

    Asserts ``|f32 - f64| <= RTOL * max(|f64|, FLOOR * peak)`` at every wavelength, with
    ``RTOL = 1e-4`` and ``FLOOR = 1e-6`` (see the module docstring for the derivation).
    """
    ref = spectrum_f64(*corner)
    got = spectrum_f32(*corner)
    assert np.all(np.isfinite(got)), f"non-finite float32 spectrum at {corner}"
    peak = np.max(np.abs(ref))
    scale = np.maximum(np.abs(ref), FLOOR * peak)
    err = np.abs(got - ref)
    worst = np.max(err / scale)
    assert np.all(err <= RTOL * scale), (
        f"float32 shape differs from float64 at {corner}: "
        f"worst |f32-f64|/max(|f64|, {FLOOR:g} peak) = {worst:.3e} > RTOL = {RTOL:g}"
    )


def _loss_fn(dtype):
    """Summed spectrum as a function of ``p = (shape, log_mbh, alpha, beta, delta, ratio)``."""
    wave = jnp.asarray(_WAVE_AA, dtype=dtype)

    def loss(p):
        state = adaf_mod.adaf_scalar_state(
            _REFERENCE_LOG_LBOL,
            p[5],
            p[1],
            p[2],
            p[3],
            p[4],
            p[0],
            dtype=dtype,
        )
        return jnp.sum(adaf_mod.adaf_spectrum_from_state(wave, state))

    return loss


# Four corners that together put every declared edge (mass, alpha, beta, delta) at one of
# them; reverse mode through the unrolled solves costs about a minute per corner on CPU.
_GRAD_CORNERS = [
    (6.0, 0.05, 0.1, 0.001, 14.0),
    (6.0, 0.5, 0.9, 0.5, 14.0),
    (10.0, 0.05, 0.9, 0.5, 14.0),
    (10.0, 0.5, 0.1, 0.001, 14.0),
]


@pytest.mark.parametrize("corner", _GRAD_CORNERS, ids=[_ids(c) for c in _GRAD_CORNERS])
def test_adaf_gradient_finite_at_box_corners_both_precisions(corner):
    """``jax.grad`` of the spectrum w.r.t. ``(shape, log_mbh, alpha, beta, delta)`` is finite.

    Checked in float64 and in pure float32 at four box corners (every declared edge appears
    in at least one; alpha = 0.5 with shape 14 is the clipped-mdot regime). Finiteness is the
    claim: the clipped-mdot regime has a zero derivative in ``mdot``, not a non-finite one.
    """
    log_mbh, alpha, beta, delta, shape = corner
    p = [shape, log_mbh, alpha, beta, delta, 1.0]
    g64 = np.asarray(jax.grad(_loss_fn(jnp.float64))(jnp.asarray(p, jnp.float64)))
    assert np.all(np.isfinite(g64[:5])), f"float64 gradient not finite at {corner}: {g64}"
    with jax.enable_x64(False):
        g32 = np.asarray(jax.grad(_loss_fn(jnp.float32))(jnp.asarray(p, dtype=jnp.float32)))
    assert np.all(np.isfinite(g32[:5])), f"float32 gradient not finite at {corner}: {g32}"


# Reproducer for the linear luminosity ratio: a zero ratio must give the exact zero spectrum
# and the exact derivative with respect to the ratio, which is the spectrum at unit ratio.
_RATIO_POINT = (12.0, 8.0, 0.3, 0.5, 0.1)  # (shape, log_mbh, alpha, beta, delta)


def _central_fd_grad(loss, p, rel_step=1e-5):
    """Central finite-difference gradient of a float64 scalar ``loss`` at ``p``."""
    p = np.asarray(p, dtype=np.float64)
    grad = np.zeros_like(p)
    for i in range(p.size):
        h = rel_step * max(1.0, abs(p[i]))
        up, dn = p.copy(), p.copy()
        up[i] += h
        dn[i] -= h
        grad[i] = (float(loss(jnp.asarray(up))) - float(loss(jnp.asarray(dn)))) / (2.0 * h)
    return grad


@pytest.mark.parametrize("ratio", [0.0, 1e-3], ids=["ratio0", "ratio1e-3"])
def test_adaf_ratio_gradient_matches_finite_differences_both_precisions(ratio):
    """``d(sum L_nu)/d(...)`` including ``agn_lum_ratio`` matches central differences.

    The spectrum is linear in ``agn_lum_ratio``, so at ratio 0 the derivative with respect to
    the ratio is the spectrum at unit ratio, not zero (the main branch returned that value; a
    log-space ratio returned 0). Float64 analytic gradients are checked against float64
    central differences at 1e-6 relative to the largest component; float32 gradients against
    the same float64 differences at 1e-3 relative to the largest component.
    """
    shape, log_mbh, alpha, beta, delta = _RATIO_POINT
    p = np.array([shape, log_mbh, alpha, beta, delta, ratio], dtype=np.float64)

    fd = _central_fd_grad(_loss_fn(jnp.float64), p)
    g64 = np.asarray(jax.grad(_loss_fn(jnp.float64))(jnp.asarray(p, jnp.float64)))
    scale = max(np.max(np.abs(fd)), 1e-300)
    assert np.all(np.isfinite(g64)), f"float64 gradient not finite: {g64}"
    assert np.max(np.abs(g64 - fd)) <= 1e-6 * scale, f"f64 AD vs FD: {g64} vs {fd}"

    with jax.enable_x64(False):
        g32 = np.asarray(
            jax.grad(_loss_fn(jnp.float32))(jnp.asarray(p, dtype=jnp.float32))
        ).astype(np.float64)
    assert np.all(np.isfinite(g32)), f"float32 gradient not finite: {g32}"
    assert np.max(np.abs(g32 - fd)) <= 1e-3 * scale, f"f32 AD vs f64 FD: {g32} vs {fd}"
