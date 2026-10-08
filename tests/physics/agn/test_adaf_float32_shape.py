# SPDX-License-Identifier: BSD-3-Clause
"""ADAF spectrum shape agrees between pure float32 and float64 over the declared box (#2783).

Conservation of the normalized shape: the spectrum :math:`L_\\nu` is normalized to the
target power, so its *shape* is what must not depend on the working precision. Before the
fix the float32 solve of the synchrotron self-absorption parameter ``x_M`` overflowed at the
clipped-accretion-rate corner (``alpha = 0.5``, ``T_e`` at its floor), and the spectrum
differed from float64 by up to 0.997 relative. Pure float32 is a supported mode, so the
shape must agree at every corner of the declared parameter box.

Tolerance derivation
--------------------
Float32 unit roundoff is :math:`\\varepsilon = 2^{-24} \\approx 6\\times10^{-8}`. Each
spectral point is a chain of at most ~50 correctly rounded float32 operations (the
Newton and fixed-point solves are fixed-count and contribute ~10 operations each per
step). Every exponential and power-law argument in the significant part of the
spectrum has magnitude at most :math:`|a| \\le 14`: the significance floor
:math:`10^{-6}` of the peak caps the exponential cutoffs at :math:`e^{-14}`. The
absolute error of an argument :math:`a` is :math:`|a|\\varepsilon`, so the worst-case
shape error is about :math:`50\\,\\varepsilon\\,14 \\approx 4\\times10^{-5}`. The tolerance
is :math:`\\mathrm{RTOL} = 10^{-4}`, a factor 2.5 above that bound. The temperature
solve is well conditioned (its float32 relative error is measured at ~1 :math:`\\varepsilon`
on healthy corners), so the bound is set by the arithmetic, not by the solve. Below the
significance floor the tolerance is taken relative to the floor, so every point uses the
same formula: ``|f32 - f64| <= RTOL * max(|f64|, FLOOR * peak)``.

Physical bound: ADAF spectrum, Mahadevan (1997, ApJ 477, 585) Eqs. 21-23, 28, 30 and
the normalization of :func:`tengri.components.agn.adaf.adaf_scalar_state`.
"""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.agn import adaf as adaf_mod

pytestmark = pytest.mark.regression_bug

# Declared box (tengri.components.agn._params): agn_log_mbh Uniform(6, 10),
# agn_adaf_alpha Uniform(0.05, 0.5), agn_adaf_beta Uniform(0.1, 0.9),
# agn_adaf_delta Uniform(0.001, 0.5). The grid takes both edges and an interior point
# for the mass; the shape luminosities span the unclipped and the clipped-mdot regimes.
LOG_MBH = (6.0, 8.0, 10.0)
ALPHA = (0.05, 0.5)
BETA = (0.1, 0.9)
DELTA = (0.001, 0.5)
SHAPE = (10.0, 12.0, 14.0)

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
    """Pure float32 spectrum agrees with float64 pointwise at every box corner.

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


def _grad_fn(dtype):
    """Gradient of the summed spectrum w.r.t. ``(shape, log_mbh, alpha, beta, delta)``.

    Eager reverse mode: a jit of the unrolled solves compiles for minutes and several GB.
    """
    wave = jnp.asarray(_WAVE_AA, dtype=dtype)

    def loss(p):
        state = adaf_mod.adaf_scalar_state(
            _REFERENCE_LOG_LBOL,
            1.0,
            p[1],
            p[2],
            p[3],
            p[4],
            p[0],
            dtype=dtype,
        )
        return jnp.sum(adaf_mod.adaf_spectrum_from_state(wave, state))

    return jax.grad(loss)


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
    g64 = np.asarray(
        _grad_fn(jnp.float64)(jnp.asarray([shape, log_mbh, alpha, beta, delta], jnp.float64))
    )
    assert np.all(np.isfinite(g64)), f"float64 gradient not finite at {corner}: {g64}"
    with jax.enable_x64(False):
        p32 = jnp.asarray([shape, log_mbh, alpha, beta, delta], dtype=jnp.float32)
        g32 = np.asarray(_grad_fn(jnp.float32)(p32))
    assert np.all(np.isfinite(g32)), f"float32 gradient not finite at {corner}: {g32}"
