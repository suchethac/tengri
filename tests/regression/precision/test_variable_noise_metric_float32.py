# SPDX-License-Identifier: BSD-3-Clause
r"""The variable-covariance Hessian must survive pure float32 (#1617).

``variable_noise_metric_vec`` builds the Gauss-Newton Hessian of the
VariableCovarianceGaussian energy from four diagonal blocks, with
:math:`\tau = 1/\sigma_{\rm eff}`:

.. code-block:: python

    H_ff = tau**2
    H_tt = residual**2 + 1.0 / tau**2
    H_ft = -2.0 * residual * tau

At a real photometric :math:`\sigma \sim 3\times10^{-30}`, **two of the four are
destroyed in float32, in opposite directions** — measured, not inferred:

=========================  ===========  ==========
block                      float64      float32
=========================  ===========  ==========
``tau``                    3.333e+29    3.333e+29
``H_ff = tau**2``          1.111e+59    **inf**
``H_tt = r**2 + 1/tau**2`` 3.611e-56    **0.0**
``H_ft = -2*r*tau``        -1.267e+02   -1.267e+02
=========================  ===========  ==========

``H_ff`` poisons ``w_f`` with ``inf`` (``NaN`` wherever ``Jv_f`` is exactly
zero). ``H_tt`` collapsing to ``0.0`` is the more dangerous of the two: the
curvature along the noise direction is silently *removed* rather than poisoned,
so the metric stays finite and looks usable.

This is #1588's defect class but not #1588's fix. There the overflowing
quantity was :math:`N^{-1}` inside :math:`J^\mathsf{T} N^{-1} J`, and whitening
twice removed it. Here the blocks *are* the Hessian, and ``H_tt`` needs
:math:`1/\tau^2` **added to** :math:`r^2` with both ~1e-56. The way out is that
each block factors into representable pieces:

.. math::

    H_{ff} J_f = \tau^2 J_f = (J_f/\sigma)/\sigma, \qquad \sigma = 1/\tau

    H_{tt} J_\tau = \bigl((r\tau)^2 + 1\bigr)\, J_\tau/\tau^2

:math:`r\tau` is the standardized residual — O(1) by construction — so the
second form never forms either underflowing term.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.observation.noise import variable_noise_metric_vec
from tengri.utils.scale import whiten

pytestmark = pytest.mark.regression_bug

_SIGMA = 3.0e-30  # erg/s/cm^2/Hz — a real photometric uncertainty
_FLUX = 1.9e-28
_N_BAND, _N_PARAM = 8, 4


def _problem(rng):
    """A linear (f, tau) response at realistic magnitudes.

    Row 3 of the flux Jacobian is exactly zero so that ``0 * inf = NaN`` is
    reachable — an all-nonzero Jacobian would report ``inf`` and hide the NaN.
    """
    a_f = rng.standard_normal((_N_BAND, _N_PARAM)) * _FLUX
    a_f[3, :] = 0.0
    a_t = rng.standard_normal((_N_BAND, _N_PARAM)) / _SIGMA
    data = jnp.asarray(_FLUX * (1.0 + 0.05 * rng.standard_normal(_N_BAND)))
    return jnp.asarray(a_f), jnp.asarray(a_t), data


def _run_with(a_f, a_t, data, xi, v):
    """``variable_noise_metric_vec`` on an explicit problem (no fixture rebuild)."""

    def signal_noise_fn(x):
        return a_f @ x, jnp.abs(a_t @ x) + 1.0 / _SIGMA

    return variable_noise_metric_vec(
        xi, v, signal_noise_fn, data, unflatten=lambda z: z, flatten=lambda z: z
    )


def _run(dtype_x64, seed=0):
    """Return ``metric_vec(xi, v)`` for the shared linear problem."""
    with jax.enable_x64(dtype_x64):
        rng = np.random.default_rng(seed)
        a_f, a_t, data = _problem(rng)
        xi = jnp.asarray(rng.standard_normal(_N_PARAM))
        v = jnp.asarray(rng.standard_normal(_N_PARAM))

        def signal_noise_fn(x):
            """Map params -> (predicted flux, tau = 1/sigma_eff)."""
            return a_f @ x, jnp.abs(a_t @ x) + 1.0 / _SIGMA

        out = variable_noise_metric_vec(
            xi, v, signal_noise_fn, data, unflatten=lambda z: z, flatten=lambda z: z
        )
        return np.asarray(out, dtype=np.float64)


def test_the_hessian_blocks_really_do_overflow_in_float32():
    """Precondition: without it, everything below could pass vacuously."""
    with jax.enable_x64(False):
        tau = 1.0 / jnp.full((4,), _SIGMA)
        h_ff = float(jnp.max(tau**2))
        h_tt = float(jnp.max(jnp.full((4,), _FLUX) ** 2 + 1.0 / tau**2))
    assert np.isinf(h_ff), (
        f"H_ff = tau**2 is {h_ff:.3e}, finite in float32 — this fixture no longer "
        "exercises the defect; lower sigma rather than deleting the test"
    )
    assert h_tt == 0.0, (
        f"H_tt is {h_tt:.3e}, not the exact 0.0 the underflow produces — the "
        "silent half of #1617 is no longer reproduced here"
    )


def test_variable_noise_metric_is_finite_in_float32():
    """The metric-vector product must be finite at a real photometric sigma."""
    got = _run(False)
    assert np.all(np.isfinite(got)), (
        f"{np.sum(~np.isfinite(got))}/{got.size} entries of the variable-noise "
        "metric are non-finite in float32 — H_ff = tau**2 overflows (#1617)"
    )
    assert np.any(got != 0.0), (
        "`got` is identically zero — finite is not enough, "
        "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
    )


def _metric_with_h_tt_zeroed(a_f, a_t, data, xi, v):
    """The metric with the noise-direction curvature deleted.

    This is the *shape* of the silent failure: ``H_tt`` underflowing to exactly
    0.0 removes this block rather than poisoning it, so the metric stays finite
    and inverts happily.
    """

    def signal_noise_fn(x):
        return a_f @ x, jnp.abs(a_t @ x) + 1.0 / _SIGMA

    (f, tau), (jv_f, jv_tau) = jax.jvp(signal_noise_fn, (xi,), (v,))
    residual = data - f
    r_std = residual * tau
    h_ft = -2.0 * r_std
    sigma_eff = 1.0 / tau
    w_f = whiten(whiten(jv_f, sigma_eff), sigma_eff) + h_ft * jv_tau
    w_t = h_ft * jv_f  # + H_tt * Jv_tau, deliberately dropped
    _, vjp_fn = jax.vjp(signal_noise_fn, xi)
    (jtw,) = vjp_fn((w_f, w_t))
    return jtw + v


def test_the_noise_direction_curvature_is_actually_present():
    """H_tt collapsing to 0.0 is finite and wrong — pin the *varying* part.

    An earlier version of this test asserted ``any(|metric| > 0)``. That cannot
    fail: the metric returns ``flatten(JTw) + v``, so the identity prior ``v``
    alone satisfies it before the Hessian contributes anything — measured, the
    check passed with the noise-direction curvature removed entirely. **An
    existence check on a quantity carrying an additive constant cannot test the
    varying part.**

    So compare against the collapse itself rather than against zero.
    """
    with jax.enable_x64(False):
        rng = np.random.default_rng(0)
        a_f, a_t, data = _problem(rng)
        xi = jnp.asarray(rng.standard_normal(_N_PARAM))
        v = jnp.asarray(rng.standard_normal(_N_PARAM))
        real = np.asarray(_run_with(a_f, a_t, data, xi, v), dtype=np.float64)
        collapsed = np.asarray(_metric_with_h_tt_zeroed(a_f, a_t, data, xi, v), dtype=np.float64)

    denom = np.maximum(np.abs(real), 1e-300)
    rel = np.max(np.abs(real - collapsed) / denom)
    assert rel > 1e-2, (
        f"the float32 metric is within {rel:.3e} of one computed with the "
        "noise-direction curvature deleted, so H_tt is contributing nothing — "
        "the silent half of #1617"
    )


def _metric_old(a_f, a_t, data, xi, v):
    """The pre-#1617 arithmetic, frozen here as the parity reference.

    Comparing the fixed function against *itself* would only prove determinism.
    The claim under test is that the factored form is the same number, so the
    reference has to be the expression it replaced.
    """

    def signal_noise_fn(x):
        return a_f @ x, jnp.abs(a_t @ x) + 1.0 / _SIGMA

    (f, tau), (jv_f, jv_tau) = jax.jvp(signal_noise_fn, (xi,), (v,))
    residual = data - f
    h_ff = tau**2
    h_tt = residual**2 + 1.0 / tau**2
    h_ft = -2.0 * residual * tau
    w_f = h_ff * jv_f + h_ft * jv_tau
    w_t = h_ft * jv_f + h_tt * jv_tau
    _, vjp_fn = jax.vjp(signal_noise_fn, xi)
    (jtw,) = vjp_fn((w_f, w_t))
    return jtw + v


def test_variable_noise_metric_preserves_float64():
    """The factored form must reproduce the original expression (rtol 1e-12)."""
    with jax.enable_x64(True):
        rng = np.random.default_rng(0)
        a_f, a_t, data = _problem(rng)
        xi = jnp.asarray(rng.standard_normal(_N_PARAM))
        v = jnp.asarray(rng.standard_normal(_N_PARAM))
        old = np.asarray(_metric_old(a_f, a_t, data, xi, v), dtype=np.float64)
    new = _run(True, seed=0)
    np.testing.assert_allclose(new, old, rtol=1e-12)


def test_the_old_arithmetic_still_fails_in_float32():
    """Negative control: the reference arm must still be broken.

    Without this, the parity test above could pass on a build where ``tau**2``
    had stopped overflowing, and the factoring would look load-bearing when it
    was not.
    """
    with jax.enable_x64(False):
        rng = np.random.default_rng(0)
        a_f, a_t, data = _problem(rng)
        xi = jnp.asarray(rng.standard_normal(_N_PARAM))
        v = jnp.asarray(rng.standard_normal(_N_PARAM))
        old = np.asarray(_metric_old(a_f, a_t, data, xi, v), dtype=np.float64)
    assert not np.all(np.isfinite(old)), (
        f"the pre-fix arithmetic is finite in float32 ({old}) — the defect this "
        "module exists for is no longer reproducible; re-measure before deleting"
    )


def test_float32_tracks_float64():
    """And the float32 answer must be the same number, not merely finite."""
    f64, f32 = _run(True), _run(False)
    denom = np.maximum(np.abs(f64), 1e-300)
    rel = np.max(np.abs(f32 - f64) / denom)
    assert rel < 1e-3, f"float32 metric departs from float64 by {rel:.3e} (rel)"
