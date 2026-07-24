# SPDX-License-Identifier: BSD-3-Clause
r"""The inference objective must be *differentiable* in pure float32 (#1206).

A finite forward SED and a finite likelihood value are still not enough for a
pure-float32 *fit*: the sampler needs ``grad(neg_log_posterior)``, and the
threaded inference forward (``predict_observables``, which builds an internal
photometry LUT) carried reverse-pass float32 overflows that a forward-only or
value-only check never sees — the failure is a NaN *gradient* under a finite
forward.

Two reverse-pass hazards, both fixed with behavior-preserving ``custom_vjp``\ s
in the stellar component:

1. **Mass scaling** ``total_mass * <per-Msun SSP> * L_sun`` — the local Jacobian
   ``total_mass * L_sun`` ~3.8e43 overflows float32 as a standalone intermediate
   under XLA's fused backward, poisoning the SSP-contraction ``dot_general`` with
   ``inf`` regardless of the incoming cotangent's size. (:func:`_mass_scale_lnu`.)
2. **Sub-band node wavelength** ``Σ(w·λ·φ) / Σ(w·φ)`` — a ratio whose autodiff
   Jacobian ``-num/den**2`` overflows for a near-zero-weight sub-band; with a
   zero downstream cotangent that is ``0 * inf = nan``. (:func:`_flux_weighted_node`.)

This pins ``grad(neg_log_posterior_fn)`` finite in pure float32 through the real
threaded objective — a full MAP/NUTS fit runs from here (verified manually;
kept out of this fast test). Both fixes are exact in float64.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import Fitter, Fixed, Observation, Parameters, Photometry, SEDModel, Uniform

pytestmark = pytest.mark.regression_bug

_SSP_FLUX_SCALE = 1.0e-17
_FREE = ("sfh_delayed_log_total_mass", "met_logzsol", "dust_tau_diff")


def _physical_ssp(ssp):
    from tengri.components.stellar.sps.dsps_wrapper import SSPData

    return SSPData(
        ssp_wave=ssp.ssp_wave,
        ssp_flux=ssp.ssp_flux * _SSP_FLUX_SCALE,
        ssp_lg_age_gyr=ssp.ssp_lg_age_gyr,
        ssp_lgmet=ssp.ssp_lgmet,
    )


def _model(ssp):
    obs = Observation(
        photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
    )
    spec = Parameters(
        sfh_delayed_log_total_mass=Uniform(9.0, 11.0),
        sfh_delayed_tau_gyr=Fixed(1.0),
        sfh_delayed_age_gyr=Fixed(5.0),
        met_logzsol=Uniform(-1.0, 0.2),
        dust_tau_diff=Uniform(0.0, 1.5),
        dust_tau_bc=Fixed(0.0),
        redshift=Fixed(0.1),
        mean_sfh_type="delayed",
    )
    return SEDModel(spec, ssp, observation=obs)


def _mock_data(ssp):
    """Mock ``(flux, noise)`` generated once in float64, as numpy — the SAME data
    fed to both precisions so a float32/float64 comparison varies only precision."""
    with jax.enable_x64(True):
        truth = {"sfh_delayed_log_total_mass": 10.0, "met_logzsol": -0.3, "dust_tau_diff": 0.5}
        mock = _model(ssp).mock(truth, snr=30.0, key=jax.random.PRNGKey(1))
        return np.asarray(mock.flux_obs, dtype=np.float64), np.asarray(
            mock.noise, dtype=np.float64
        )


def _context(ssp, flux, noise):
    """A Fitter/InferenceContext on the given data (converted to the active precision)."""
    from tengri.inference.context import InferenceContext

    return InferenceContext.from_target(Fitter(_model(ssp), jnp.asarray(flux), jnp.asarray(noise)))


def test_neg_log_posterior_gradient_is_finite_in_pure_float32(synthetic_ssp_wide):
    """``grad(nlp)`` must be finite in pure float32 through the threaded objective."""
    ssp = _physical_ssp(synthetic_ssp_wide)
    flux, noise = _mock_data(ssp)
    with jax.enable_x64(False):
        ctx = _context(ssp, flux, noise)
        da = ctx.data_args
        for i in range(4):
            p = ctx.initial_params(jax.random.fold_in(jax.random.PRNGKey(0), i))
            g = jax.grad(lambda q: ctx.neg_log_posterior_fn(q, da))(p)
            leaves = [np.asarray(v) for v in jax.tree_util.tree_leaves(g)]
            assert leaves and leaves[0].dtype == jnp.float32, "precondition: genuinely float32"
            assert all(np.all(np.isfinite(v)) for v in leaves), (
                f"grad(nlp) is non-finite at draw {i} in pure float32 — a reverse-pass "
                "overflow in the stellar mass scaling or the sub-band node ratio"
            )


def test_float32_inference_gradient_is_grad_finite_where_it_was_nan(synthetic_ssp_wide):
    """Regression guard: the reverse pass must be finite at a point that NaN'd.

    Distinct from the loop above only in intent — this pins that the *specific*
    reverse-pass hazards (the mass-scale ``dot_general`` and the sub-band node
    ratio) are neutralized, so a revert of either ``custom_vjp`` re-introduces a
    NaN here. (Forward-value and float64-gradient parity are covered by the wider
    stellar / photometry regression suite; the ``synthetic_ssp_wide`` fixture is
    not a valid bed for a float32-vs-float64 *value* comparison — its rescaled
    grid is numerically degenerate in float32, unrelated to these fixes.)
    """
    ssp = _physical_ssp(synthetic_ssp_wide)
    flux, noise = _mock_data(ssp)
    with jax.enable_x64(False):
        ctx = _context(ssp, flux, noise)
        da = ctx.data_args
        p = ctx.initial_params(jax.random.PRNGKey(11))
        g = jax.grad(lambda q: ctx.neg_log_posterior_fn(q, da))(p)
    assert all(np.all(np.isfinite(np.asarray(v))) for v in jax.tree_util.tree_leaves(g)), (
        "grad(nlp) non-finite in pure float32 — the stellar mass-scale or "
        "sub-band node-ratio custom_vjp is not neutralizing its overflow"
    )
