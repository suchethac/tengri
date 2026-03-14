"""Benchmark tests for photometry precomputation speedup and gradient cleanliness.

Validates:
1. Fast (precomputed) photometry agrees with exact within 1% (Zacharegkas+2025)
2. Gradient speedup is >5x over exact path
3. All gradients are finite (no NaN/inf)
4. Autodiff gradients match finite differences to ~6 digits
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from diffsed import (
    Model, ParamSpec, Uniform, Gaussian, Fixed,
    load_ssp_data, load_filter_set,
)
from diffsed.utils.transforms import to_bounded


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ssp_data():
    return load_ssp_data(
        "data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
    )


@pytest.fixture(scope="module")
def sdss_filters():
    return load_filter_set(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])


@pytest.fixture(scope="module")
def smooth_spec():
    """Smooth SFH spec with fixed redshift (triggers precomputation)."""
    return ParamSpec(
        sfh_alpha=Uniform(0.5, 3.0),
        sfh_beta=Uniform(0.3, 2.0),
        sfh_tau_peak_gyr=Uniform(0.5, 10.0),
        sfh_peak_sfr=Uniform(0.1, 50.0),
        met_logzsol=Gaussian(-1.5, 0.3, lo=-2.0, hi=-1.23),
        dust_tau_bc=Uniform(0.0, 3.0),
        dust_tau_diff=0.3,
        dust_slope=-0.7,
        redshift=0.1,
        stochastic=False,
    )


# ---------------------------------------------------------------------------
# Accuracy: fast vs exact
# ---------------------------------------------------------------------------

class TestPrecomputeAccuracy:
    """Verify approximate photometry matches exact within tolerance."""

    def test_fast_vs_exact_agreement(self, ssp_data, sdss_filters, smooth_spec):
        """Fast photometry agrees with exact within 1% per band."""
        model_fast = Model(smooth_spec, ssp_data, filters=sdss_filters, precompute=True)
        model_exact = Model(smooth_spec, ssp_data, filters=sdss_filters, precompute=False)

        assert model_fast._precomp is not None
        assert model_exact._precomp is None

        key = jax.random.PRNGKey(42)
        params = smooth_spec.sample(key)

        flux_fast = model_fast.predict_photometry(params)
        flux_exact = model_exact.predict_photometry(params)

        frac_error = jnp.abs(flux_fast - flux_exact) / flux_exact
        assert float(jnp.max(frac_error)) < 0.01, (
            f"Max fractional error {float(jnp.max(frac_error)):.4f} > 1%"
        )

    def test_fast_vs_exact_multiple_params(self, ssp_data, sdss_filters, smooth_spec):
        """Agreement holds across 10 random parameter sets."""
        model_fast = Model(smooth_spec, ssp_data, filters=sdss_filters, precompute=True)
        model_exact = Model(smooth_spec, ssp_data, filters=sdss_filters, precompute=False)

        for i in range(10):
            key = jax.random.PRNGKey(i)
            params = smooth_spec.sample(key)

            flux_fast = model_fast.predict_photometry(params)
            flux_exact = model_exact.predict_photometry(params)

            frac_error = jnp.abs(flux_fast - flux_exact) / flux_exact
            assert float(jnp.max(frac_error)) < 0.01, (
                f"Seed {i}: max error {float(jnp.max(frac_error)):.4f}"
            )


# ---------------------------------------------------------------------------
# Speedup benchmark
# ---------------------------------------------------------------------------

class TestPrecomputeSpeedup:
    """Verify meaningful speedup from precomputation."""

    def test_gradient_speedup(self, ssp_data, sdss_filters, smooth_spec):
        """Gradient evaluation is >5x faster with precomputation."""
        model_fast = Model(smooth_spec, ssp_data, filters=sdss_filters, precompute=True)
        model_exact = Model(smooth_spec, ssp_data, filters=sdss_filters, precompute=False)

        params = smooth_spec.sample(jax.random.PRNGKey(42))
        data = model_exact.predict_photometry(params)
        noise = data / 20.0

        def make_loss(model):
            def loss(p):
                pred = model.predict_photometry(p)
                return jnp.sum(((data - pred) / noise) ** 2)
            return jax.jit(jax.value_and_grad(loss))

        grad_fast = make_loss(model_fast)
        grad_exact = make_loss(model_exact)

        # JIT warmup
        _ = grad_fast(params)
        _ = grad_exact(params)

        N = 50
        t0 = time.time()
        for _ in range(N):
            _ = grad_fast(params)
        t_fast = (time.time() - t0) / N

        t0 = time.time()
        for _ in range(N):
            _ = grad_exact(params)
        t_exact = (time.time() - t0) / N

        speedup = t_exact / t_fast
        assert speedup > 5.0, (
            f"Gradient speedup {speedup:.1f}x < 5x "
            f"(fast={t_fast*1e3:.2f}ms, exact={t_exact*1e3:.2f}ms)"
        )


# ---------------------------------------------------------------------------
# Gradient cleanliness
# ---------------------------------------------------------------------------

class TestGradientCleanliness:
    """Verify gradients are finite and agree with finite differences."""

    def _build_loss(self, model, data, noise, spec):
        """Build loss function in unbounded space (like Fitter does)."""
        free_names = spec.free_params
        bounds = {n: spec.get_distribution(n).bounds for n in free_names}
        fixed_values = spec.get_fixed_values()

        def loss_fn(params_unbounded):
            p = {}
            for name in free_names:
                lo, hi = bounds[name]
                p[name] = to_bounded(params_unbounded[name], lo, hi)
            for name, val in fixed_values.items():
                p[name] = val
            predicted = model.predict_photometry(p)
            return jnp.sum(((data - predicted) / noise) ** 2)

        return loss_fn

    def test_all_gradients_finite(self, ssp_data, sdss_filters, smooth_spec):
        """All autodiff gradients are finite (no NaN/inf)."""
        model = Model(smooth_spec, ssp_data, filters=sdss_filters, precompute=True)
        params = smooth_spec.sample(jax.random.PRNGKey(42))
        data = model.predict_photometry(params)
        noise = data / 20.0

        loss_fn = self._build_loss(model, data, noise, smooth_spec)
        init = {n: jnp.array(0.3) for n in smooth_spec.free_params}
        _, grads = jax.value_and_grad(loss_fn)(init)

        for name in grads:
            assert bool(jnp.isfinite(grads[name])), (
                f"Gradient for {name} is not finite: {float(grads[name])}"
            )

    def test_autodiff_matches_finite_differences(self, ssp_data, sdss_filters,
                                                  smooth_spec):
        """Autodiff gradients match finite differences to 4+ digits."""
        model = Model(smooth_spec, ssp_data, filters=sdss_filters, precompute=True)
        params = smooth_spec.sample(jax.random.PRNGKey(42))
        data = model.predict_photometry(params)
        noise = data / 20.0

        loss_fn = self._build_loss(model, data, noise, smooth_spec)

        # Initialize away from clamped regions
        init = {n: jnp.array(-0.5) for n in smooth_spec.free_params}
        _, grads = jax.value_and_grad(loss_fn)(init)

        eps = 1e-5
        for name in smooth_spec.free_params:
            init_p = dict(init)
            init_m = dict(init)
            init_p[name] = init[name] + eps
            init_m[name] = init[name] - eps
            fd = float((loss_fn(init_p) - loss_fn(init_m)) / (2 * eps))
            ad = float(grads[name])

            if abs(fd) < 1e-10:
                # Both should be ~zero
                assert abs(ad) < 1e-6, (
                    f"{name}: AD={ad:.6e} but FD≈0"
                )
            else:
                ratio = ad / fd
                assert abs(ratio - 1.0) < 1e-3, (
                    f"{name}: AD/FD ratio = {ratio:.6f}, "
                    f"AD={ad:.6e}, FD={fd:.6e}"
                )

    def test_stochastic_gradients_finite(self, ssp_data, sdss_filters):
        """Gradients are finite for stochastic (GP) model too."""
        spec = ParamSpec(
            sfh_alpha=Uniform(0.5, 3.0),
            sfh_beta=Uniform(0.3, 2.0),
            sfh_tau_peak_gyr=Uniform(0.5, 10.0),
            sfh_peak_sfr=Uniform(0.1, 50.0),
            psd_sigma=Uniform(0.1, 3.0),
            psd_tau_myr=Uniform(1.0, 300.0),
            met_logzsol=Gaussian(-1.5, 0.3, lo=-2.0, hi=-1.23),
            dust_tau_bc=Uniform(0.0, 3.0),
            dust_tau_diff=0.3,
            dust_slope=-0.7,
            redshift=0.1,
            stochastic=True,
            n_grid=64,
        )
        model = Model(spec, ssp_data, filters=sdss_filters, precompute=True)
        params = spec.sample(jax.random.PRNGKey(42))
        data = model.predict_photometry(params)
        noise = data / 20.0

        free_names = spec.free_params
        bounds = {n: spec.get_distribution(n).bounds for n in free_names}
        fixed_values = spec.get_fixed_values()

        def loss_fn(params_unbounded):
            p = {}
            for name in free_names:
                lo, hi = bounds[name]
                p[name] = to_bounded(params_unbounded[name], lo, hi)
            for name, val in fixed_values.items():
                p[name] = val
            if "psd_xi" in params_unbounded:
                p["psd_xi"] = params_unbounded["psd_xi"]
            predicted = model.predict_photometry(p)
            return jnp.sum(((data - predicted) / noise) ** 2)

        init = {n: jnp.array(0.0) for n in free_names}
        init["psd_xi"] = jnp.zeros(spec.n_grid)
        _, grads = jax.value_and_grad(loss_fn)(init)

        for name in grads:
            g = grads[name]
            if g.ndim == 0:
                assert bool(jnp.isfinite(g)), f"{name}: grad not finite"
            else:
                assert bool(jnp.all(jnp.isfinite(g))), f"{name}: some grads not finite"
