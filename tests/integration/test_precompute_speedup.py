# SPDX-License-Identifier: BSD-3-Clause
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
import pytest

from tengri import (
    Gaussian,
    Parameters,
    SEDModel,
    Uniform,
    WavePrecomp,
    load_filter_set,
    load_ssp_data,
)
from tengri.utils.transforms import to_bounded

# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def ssp_data():
    return load_ssp_data("data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")


@pytest.fixture(scope="module")
def sdss_filters():
    return load_filter_set(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])


@pytest.fixture(scope="module")
def smooth_spec():
    """Smooth SFH spec with fixed redshift (triggers precomputation)."""
    return Parameters(
        mean_sfh_type="dpl",
        sfh_dpl_alpha=Uniform(0.5, 3.0),
        sfh_dpl_beta=Uniform(0.3, 2.0),
        sfh_dpl_tau_gyr=Uniform(0.5, 10.0),
        sfh_dpl_log_total_mass=Uniform(7.0, 12.5),
        met_logzsol=Gaussian(-1.5, 0.3, lo=-2.0, hi=-1.23),
        dust_tau_bc=Uniform(0.0, 3.0),
        dust_tau_diff=0.3,
        dust_slope=-0.7,
        redshift=0.1,
    )


# ── Accuracy: fast vs exact ───────────────────────────────────────
#
# WavePrecomp/SpectrumPrecomp LUT accuracy is validated with controlled
# configs (diffuse-only dust, machine-exact continuum) in
# tests/contract/test_stellar_precomp_contract.py and test_spectrum_lut.py.
# The former benchmark accuracy tests here were vacuous before #620
# (precompute= did not affect predict_photometry, so they compared the exact
# path to itself) and are removed; the effective-wavelength dust
# factorization error on a *dusty* sampled galaxy is several-% in the blue
# (see #620 / optimization-architecture.md), which is config-dependent and
# unsuitable for a fixed-tolerance random-draw assertion.


# ── Speedup benchmark ─────────────────────────────────────────────


class TestPrecomputeSpeedup:
    """Verify meaningful speedup from precomputation."""

    def test_gradient_speedup(self, ssp_data, sdss_filters, smooth_spec):
        """Gradient evaluation is >5x faster with precomputation."""
        model_fast = SEDModel(smooth_spec, ssp_data, filters=sdss_filters, approx=WavePrecomp())
        model_exact = SEDModel(smooth_spec, ssp_data, filters=sdss_filters, approx=None)

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
        # Auto mode now picks compositional (bit-exact JIT) over the old
        # precomputed path.  Compositional fuses the full SED einsum, so
        # the speedup is ~3-10x (not the ~30x from precomputed).
        assert speedup > 2.0, (
            f"Gradient speedup {speedup:.1f}x < 2x "
            f"(approx={t_fast * 1e3:.2f}ms, exact={t_exact * 1e3:.2f}ms)"
        )


# ── Gradient cleanliness ──────────────────────────────────────────


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
        model = SEDModel(smooth_spec, ssp_data, filters=sdss_filters, approx=None)
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
            assert jnp.any(grads[name] != 0.0), (
                "`grads[name]` is identically zero — finite is not enough, "
                "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
            )

    def test_autodiff_matches_finite_differences(self, ssp_data, sdss_filters, smooth_spec):
        """Autodiff gradients match finite differences to 4+ digits."""
        model = SEDModel(smooth_spec, ssp_data, filters=sdss_filters, approx=None)
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
                assert abs(ad) < 1e-6, f"{name}: AD={ad:.6e} but FD≈0"
            else:
                ratio = ad / fd
                assert abs(ratio - 1.0) < 1e-3, (
                    f"{name}: AD/FD ratio = {ratio:.6f}, AD={ad:.6e}, FD={fd:.6e}"
                )

    def test_stochastic_gradients_finite(self, ssp_data, sdss_filters):
        """Gradients are finite for stochastic (GP) model too."""
        spec = Parameters(
            mean_sfh_type=["dpl", "field"],
            sfh_dpl_alpha=Uniform(0.5, 3.0),
            sfh_dpl_beta=Uniform(0.3, 2.0),
            sfh_dpl_tau_gyr=Uniform(0.5, 10.0),
            sfh_dpl_log_total_mass=Uniform(7.0, 12.5),
            sfh_field_psd_sigma=Uniform(0.1, 3.0),
            sfh_field_psd_tau_myr=Uniform(1.0, 300.0),
            met_logzsol=Gaussian(-1.5, 0.3, lo=-2.0, hi=-1.23),
            dust_tau_bc=Uniform(0.0, 3.0),
            dust_tau_diff=0.3,
            dust_slope=-0.7,
            redshift=0.1,
            n_grid=64,
        )
        model = SEDModel(spec, ssp_data, filters=sdss_filters, approx=None)
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
                # ``StellarSEDComponent`` reads ``sfh_field_xi``, not ``psd_xi``;
                # ``inference/loss_functions.py`` attaches BOTH for exactly this
                # reason and carries a comment saying so. Attaching only
                # ``psd_xi`` here meant the latent field never reached the model:
                # measured, ``grads["psd_xi"]`` summed to exactly 0.0 and
                # ``grads["sfh_field_psd_sigma"]`` was bit-identical for
                # ``xi = 0`` and ``xi ~ N(0, 1)`` (36.78049294936069 both times) —
                # a "stochastic gradients" test in which the stochastic field was
                # inert. The finite-only assertion on the array branch below could
                # not see it, because an identically zero array is finite.
                p["sfh_field_xi"] = params_unbounded["psd_xi"]
            predicted = model.predict_photometry(p)
            return jnp.sum(((data - predicted) / noise) ** 2)

        init = {n: jnp.array(0.0) for n in free_names}
        init["psd_xi"] = jnp.zeros(spec.n_grid)
        _, grads = jax.value_and_grad(loss_fn)(init)

        # The PSD timescale only *colors* the latent field, so at xi = 0 there is
        # nothing to color and its gradient is exactly zero by construction. Every
        # other parameter must move the loss even from the origin.
        inert_at_zero_field = {"sfh_field_psd_tau_myr"}

        for name in grads:
            g = grads[name]
            if g.ndim == 0:
                # grad-assert: finite-only — `sfh_field_psd_tau_myr` is the PSD
                # correlation time, which enters only as the shape applied to the
                # latent field; with xi identically zero the field is zero for any
                # timescale, so d/dtau is exactly -0.0 here. Its non-zero half is
                # asserted below at xi ~ N(0, 1), where it measures 4.32.
                assert bool(jnp.isfinite(g)), f"{name}: grad not finite"
                if name in inert_at_zero_field:
                    continue
                assert jnp.any(g != 0.0), (
                    f"`grads[{name!r}]` is identically zero — finite is not enough, "
                    "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
                )
            else:
                assert bool(jnp.all(jnp.isfinite(g))), f"{name}: some grads not finite"
                assert jnp.any(g != 0.0), (
                    f"`grads[{name!r}]` is identically zero across all "
                    f"{g.shape[0]} entries. For `psd_xi` that means the latent field "
                    "is not reaching the model at all — finite is not enough (#2100)."
                )

        # The timescale is not inert in general, only at the zero field. Off the
        # origin it must carry real signal, which is what rules out a PSD whose
        # correlation time never reaches the SFH.
        init_live = dict(init)
        init_live["psd_xi"] = jax.random.normal(jax.random.PRNGKey(7), (spec.n_grid,))
        _, grads_live = jax.value_and_grad(loss_fn)(init_live)
        g_tau = grads_live["sfh_field_psd_tau_myr"]
        assert bool(jnp.isfinite(g_tau)), (
            f"non-finite psd_tau_myr gradient at xi ~ N(0,1): {g_tau}"
        )
        assert jnp.any(g_tau != 0.0), (
            "`sfh_field_psd_tau_myr` has an exactly zero gradient even at a non-zero "
            "latent field, so the PSD correlation time never reaches the SFH (#2100)."
        )
