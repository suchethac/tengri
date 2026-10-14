# SPDX-License-Identifier: BSD-3-Clause
"""The Cue trained-box clamp keeps a finite, non-zero gradient (#2539, #2943).

``clamp_ionspec_straight_through`` pins the seven ionizing-shape parameters to the
Cue reference code's training box with value ``clip(x)`` and gradient equal to the
identity, so a fit driven past the box keeps a signal. This pins that gradient for
(a) an in-range input, (b) a finite out-of-box input and (c) a component that is
non-finite upstream of the clamp (a segment with zero luminosity makes
``log10(L_k)`` equal ``-inf``): the upstream gradient is NaN free and the other
six components keep their unit gradient. Through ``CueBackend`` the Jacobian of
the clamped shape with respect to the SSP weights is finite and non-zero on the
parameters the weights drive.

Tolerance: the unit gradient is exact (``==``); the Jacobian checks are
finiteness and non-vanishing.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.nebular.cue import CueBackend
from tengri.components.nebular.ionizing_spectrum import (
    _CLIP_RANGES,
    IONSPEC_NAMES,
    clamp_ionspec_straight_through,
)

pytestmark = pytest.mark.gradient

_CUE_WEIGHTS = "data/cue_weights.npz"
_LO = np.asarray([v[0] for v in _CLIP_RANGES.values()])
_HI = np.asarray([v[1] for v in _CLIP_RANGES.values()])
_MID = jnp.asarray((_LO + _HI) / 2.0)
_COEF = jnp.arange(1.0, 8.0)  # distinct per-component weights: a swap or a zero shows up


def _weighted_grad(x):
    return jax.grad(lambda v: jnp.sum(clamp_ionspec_straight_through(v) * _COEF))(x)


def test_in_range_gradient_is_finite_and_nonzero():
    """Inside the box d(out_i)/d(x_i) is exactly 1, so the gradient is the weights."""
    g = np.asarray(_weighted_grad(_MID))
    assert np.all(np.isfinite(g))
    assert np.all(g != 0.0)
    np.testing.assert_array_equal(g, np.asarray(_COEF))


@pytest.mark.parametrize("side", ["below", "above"])
@pytest.mark.parametrize("k", range(7))
def test_out_of_box_finite_gradient_is_identity(k, side):
    """A finite value past one bound is pinned there but keeps the unit gradient."""
    x = _MID.at[k].set(_LO[k] - 7.0 if side == "below" else _HI[k] + 7.0)
    out = np.asarray(clamp_ionspec_straight_through(x))
    assert out[k] == (_LO[k] if side == "below" else _HI[k])
    g = np.asarray(_weighted_grad(x))
    assert np.all(np.isfinite(g))
    assert np.all(g != 0.0)  # the clamped component keeps its signal
    np.testing.assert_array_equal(g, np.asarray(_COEF))


def _poisoned(u, sign):
    """Shape whose logLratio1 is non-finite upstream: ``+/-inf`` from a degenerate log."""
    raw = _MID * u
    # log10 of a zero segment luminosity is -inf; its derivative at 0 is infinite, so
    # the tangent of this component is NaN (inf * 0) before the clamp sees it.
    bad = sign * jnp.inf * (u / u) if sign > 0 else jnp.log10(0.0 * u)
    return clamp_ionspec_straight_through(raw.at[4].set(bad))


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_non_finite_upstream_component_has_nan_free_gradient(sign):
    """A component that is +/-inf upstream never turns the gradient into NaN.

    ``log10`` of a zero segment luminosity is ``-inf``; the clamp lands that
    component on a bound with zero gradient while the other six keep their unit
    gradient with respect to the upstream scalar that drives them. Forward mode
    would carry the NaN tangent of the poisoned component through the clamp's
    non-finite branch before the double-where.
    """
    u0 = jnp.asarray(1.0)
    out = np.asarray(_poisoned(u0, sign))
    assert out[4] == (_HI[4] if sign > 0 else _LO[4])
    g = np.asarray(jax.jacfwd(lambda u: _poisoned(u, sign))(u0))
    assert np.all(np.isfinite(g))
    keep = np.arange(7) != 4
    assert np.all(g[keep] != 0.0)
    np.testing.assert_array_equal(g[keep], np.asarray(_MID)[keep])
    assert g[4] == 0.0  # pinned at a bound by an infinity: no signal, and no NaN


@pytest.mark.parametrize("bad", [jnp.inf, -jnp.inf, jnp.nan])
def test_non_finite_constant_component_reverse_gradient_is_nan_free(bad):
    """Reverse mode: a guarded upstream +/-inf/NaN gives a finite, non-zero gradient."""

    def loss(u):
        raw = _MID * u
        poisoned = jnp.where(u > 0.0, bad, 0.0)  # non-finite, built the way the backend does
        return jnp.sum(clamp_ionspec_straight_through(raw.at[4].set(poisoned)) * _COEF)

    g = float(jax.grad(loss)(jnp.asarray(1.0)))
    assert np.isfinite(g)
    assert g != 0.0
    keep = np.arange(7) != 4
    assert g == pytest.approx(float(np.sum(np.asarray(_MID * _COEF)[keep])), rel=1e-6)


@pytest.mark.skipif(not Path(_CUE_WEIGHTS).is_file(), reason=f"missing {_CUE_WEIGHTS}")
def test_clamped_weighted_shape_jacobian_is_finite(synthetic_ssp_wide):
    """On the wide synthetic SSP the clamp binds; the weight Jacobian stays finite.

    The synthetic SSP has no ionizing flux, so its fitted ``logLratio1`` and
    ``logLratio2`` lie below the box and are clamped. The Jacobian of the seven
    shape parameters with respect to the SSP weights is finite everywhere and
    non-zero on the clamped components, equal to the raw Jacobian there.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        be = CueBackend(_CUE_WEIGHTS, ssp_data=synthetic_ssp_wide)
    log_age = jnp.asarray(be._ssp_log_age_yr)
    w0 = jnp.ones(log_age.shape[0])
    log_z = float(jnp.asarray(synthetic_ssp_wide.ssp_lgmet)[1])

    def stack(method, w):
        out = getattr(be, method)(w, log_age, log_z)
        return jnp.stack([out[k] for k in IONSPEC_NAMES])

    raw = np.asarray(stack("_weighted_cue_params_raw", w0))
    clamped = np.asarray(stack("_compute_weighted_cue_params", w0))
    binds = (raw < _LO) | (raw > _HI)
    assert binds[4] and binds[5]  # the premise: the clamp is active on this SSP
    np.testing.assert_array_equal(clamped[binds], np.where(raw < _LO, _LO, _HI)[binds])

    jac_raw = np.asarray(jax.jacrev(lambda w: stack("_weighted_cue_params_raw", w))(w0))
    jac = np.asarray(jax.jacrev(lambda w: stack("_compute_weighted_cue_params", w))(w0))
    assert np.all(np.isfinite(jac))
    np.testing.assert_array_equal(jac, jac_raw)
    for k in np.flatnonzero(binds & (np.abs(jac_raw).sum(axis=1) > 0)):
        assert np.abs(jac[k]).sum() > 0.0
    assert np.abs(jac[4]).sum() > 0.0 and np.abs(jac[5]).sum() > 0.0
