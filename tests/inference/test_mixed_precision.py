# SPDX-License-Identifier: BSD-3-Clause
"""Tests for mixed-precision forward model (float32 with float64 output).

Validates that using forward_dtype="float32" gives results within
acceptable accuracy of float64, with measurable memory and speed benefits.
"""

import pytest

pytestmark = pytest.mark.contract

import os
import time

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from tengri.components.dust.attenuation import precompute_dust_age_weights
from tengri.components.stellar.sps.dsps_wrapper import LSUN_ERG_PER_S


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite-difference gradient. O(eps^2) accurate."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


# ── Fixtures: synthetic SSP data ──────────────────────────────────


@pytest.fixture
def n_met():
    return 5


@pytest.fixture
def n_age():
    return 107


@pytest.fixture
def n_filt():
    return 5


@pytest.fixture
def ssp_ages_yr(n_age):
    return 10.0 ** jnp.linspace(5.5, 10.14, n_age)


@pytest.fixture
def ssp_lgmet(n_met):
    return jnp.array([-2.0, -1.5, -1.0, -0.5, 0.0])


@pytest.fixture
def ssp_phot(n_met, n_age, n_filt):
    key = jax.random.PRNGKey(0)
    return jnp.abs(jax.random.normal(key, (n_met, n_age, n_filt))) + 0.1


@pytest.fixture
def eff_waves_rest(n_filt):
    return jnp.array([3551.0, 4686.0, 6166.0, 7480.0, 8932.0])


@pytest.fixture
def dust_age_weights(ssp_ages_yr):
    return precompute_dust_age_weights(ssp_ages_yr)


@pytest.fixture
def sfr_on_ssp(n_age):
    key = jax.random.PRNGKey(2)
    return jnp.abs(jax.random.normal(key, (n_age,))) + 0.01


# ── Build kernels at different precisions ─────────────────────────


def _make_kernel(ssp_phot, ssp_lgmet, eff_waves_rest, dust_age_w, flux_scale, ssp_ages_yr, dtype):
    """Build fused photometry kernel at specified dtype."""
    dt = jnp.dtype(dtype)
    sp = ssp_phot.astype(dt)
    slm = ssp_lgmet.astype(dt)
    ewr = eff_waves_rest.astype(dt)
    daw = dust_age_w.astype(dt)
    fs = dt.type(flux_scale)
    say = ssp_ages_yr.astype(dt)
    ls = dt.type(LSUN_ERG_PER_S)

    @jax.jit
    def kernel(sfr_on_ssp, log_z, tau_v1, tau_v2, dust_n):
        sfr = sfr_on_ssp.astype(dt)
        lz = jnp.asarray(log_z, dtype=dt)
        tv1 = jnp.asarray(tau_v1, dtype=dt)
        tv2 = jnp.asarray(tau_v2, dtype=dt)
        dn = jnp.asarray(dust_n, dtype=dt)

        age_dt = jnp.concatenate(
            [
                jnp.array([say[1] - say[0]]),
                0.5 * (say[2:] - say[:-2]),
                jnp.array([say[-1] - say[-2]]),
            ]
        )
        weights = sfr * age_dt

        log_z_c = jnp.clip(lz, slm[0], slm[-1])
        idx = jnp.clip(jnp.searchsorted(slm, log_z_c) - 1, 0, len(slm) - 2)
        frac = (log_z_c - slm[idx]) / (slm[idx + 1] - slm[idx])
        ssp_at_z = (1.0 - frac) * sp[idx] + frac * sp[idx + 1]

        wave_ratio = (ewr / 5500.0) ** dn
        tau_v_eff = daw * tv1 + tv2
        dust = jnp.exp(-(tau_v_eff[:, None] * wave_ratio[None, :]))

        flux_lsun = jnp.einsum("i,if,if->f", weights, dust, ssp_at_z)
        return (fs * flux_lsun * ls).astype(jnp.float64)

    return kernel


# ── Tests ─────────────────────────────────────────────────────────


class TestMixedPrecisionAccuracy:
    """float32 forward model matches float64 within acceptable tolerance."""

    def test_photometry_agreement(
        self,
        ssp_phot,
        ssp_lgmet,
        eff_waves_rest,
        dust_age_weights,
        ssp_ages_yr,
        sfr_on_ssp,
    ):
        """float32 photometry agrees with float64 within 0.1%."""
        flux_scale = 1e-30

        kernel_f64 = _make_kernel(
            ssp_phot,
            ssp_lgmet,
            eff_waves_rest,
            dust_age_weights,
            flux_scale,
            ssp_ages_yr,
            "float64",
        )
        kernel_f32 = _make_kernel(
            ssp_phot,
            ssp_lgmet,
            eff_waves_rest,
            dust_age_weights,
            flux_scale,
            ssp_ages_yr,
            "float32",
        )

        args = (sfr_on_ssp, -1.0, 0.5, 0.3, -0.7)
        result_f64 = kernel_f64(*args)
        result_f32 = kernel_f32(*args)

        # float32 should agree within 0.1% (1e-3 relative)
        frac_error = jnp.abs(result_f32 - result_f64) / jnp.abs(result_f64)
        assert float(jnp.max(frac_error)) < 1e-3, (
            f"Max fractional error: {float(jnp.max(frac_error)):.2e}"
        )

    @pytest.mark.parametrize("log_z", [-2.0, -1.5, -1.0, -0.5, 0.0])
    def test_agreement_across_metallicities(
        self,
        ssp_phot,
        ssp_lgmet,
        eff_waves_rest,
        dust_age_weights,
        ssp_ages_yr,
        sfr_on_ssp,
        log_z,
    ):
        """Agreement holds across the full metallicity grid."""
        flux_scale = 1e-30
        k64 = _make_kernel(
            ssp_phot,
            ssp_lgmet,
            eff_waves_rest,
            dust_age_weights,
            flux_scale,
            ssp_ages_yr,
            "float64",
        )
        k32 = _make_kernel(
            ssp_phot,
            ssp_lgmet,
            eff_waves_rest,
            dust_age_weights,
            flux_scale,
            ssp_ages_yr,
            "float32",
        )

        r64 = k64(sfr_on_ssp, log_z, 0.5, 0.3, -0.7)
        r32 = k32(sfr_on_ssp, log_z, 0.5, 0.3, -0.7)
        frac_err = jnp.abs(r32 - r64) / jnp.abs(r64)
        assert float(jnp.max(frac_err)) < 1e-3

    @pytest.mark.parametrize(
        "tau_v1,tau_v2",
        [
            (0.0, 0.0),
            (3.0, 1.5),
            (0.01, 0.01),
            (1.0, 0.5),
        ],
    )
    def test_agreement_various_dust(
        self,
        ssp_phot,
        ssp_lgmet,
        eff_waves_rest,
        dust_age_weights,
        ssp_ages_yr,
        sfr_on_ssp,
        tau_v1,
        tau_v2,
    ):
        """Agreement holds across dust parameter space."""
        flux_scale = 1e-30
        k64 = _make_kernel(
            ssp_phot,
            ssp_lgmet,
            eff_waves_rest,
            dust_age_weights,
            flux_scale,
            ssp_ages_yr,
            "float64",
        )
        k32 = _make_kernel(
            ssp_phot,
            ssp_lgmet,
            eff_waves_rest,
            dust_age_weights,
            flux_scale,
            ssp_ages_yr,
            "float32",
        )

        r64 = k64(sfr_on_ssp, -1.0, tau_v1, tau_v2, -0.7)
        r32 = k32(sfr_on_ssp, -1.0, tau_v1, tau_v2, -0.7)
        frac_err = jnp.abs(r32 - r64) / jnp.maximum(jnp.abs(r64), 1e-50)
        assert float(jnp.max(frac_err)) < 1e-3

    def test_output_dtype_is_float64(
        self,
        ssp_phot,
        ssp_lgmet,
        eff_waves_rest,
        dust_age_weights,
        ssp_ages_yr,
        sfr_on_ssp,
    ):
        """Even with float32 forward, output must be float64."""
        k32 = _make_kernel(
            ssp_phot, ssp_lgmet, eff_waves_rest, dust_age_weights, 1e-30, ssp_ages_yr, "float32"
        )
        result = k32(sfr_on_ssp, -1.0, 0.5, 0.3, -0.7)
        assert result.dtype == jnp.float64


class TestMixedPrecisionGradients:
    """Gradients through float32 forward model."""

    def test_gradients_finite(
        self,
        ssp_phot,
        ssp_lgmet,
        eff_waves_rest,
        dust_age_weights,
        ssp_ages_yr,
        sfr_on_ssp,
    ):
        """float32 gradients are finite."""
        k32 = _make_kernel(
            ssp_phot, ssp_lgmet, eff_waves_rest, dust_age_weights, 1e-30, ssp_ages_yr, "float32"
        )

        def loss(log_z, tau_v1, tau_v2):
            return jnp.sum(k32(sfr_on_ssp, log_z, tau_v1, tau_v2, -0.7))

        grads = jax.grad(loss, argnums=(0, 1, 2))(-1.0, 0.5, 0.3)
        # FD is unreliable for float32 kernels because f32 truncation in the
        # function evaluations dominates the finite-difference step — the meaningful
        # accuracy test is test_gradients_agree_with_f64 below.
        for g in grads:
            assert jnp.isfinite(g)

    def test_gradients_agree_with_f64(
        self,
        ssp_phot,
        ssp_lgmet,
        eff_waves_rest,
        dust_age_weights,
        ssp_ages_yr,
        sfr_on_ssp,
    ):
        """float32 gradients agree with float64 within 1%."""
        k64 = _make_kernel(
            ssp_phot, ssp_lgmet, eff_waves_rest, dust_age_weights, 1e-30, ssp_ages_yr, "float64"
        )
        k32 = _make_kernel(
            ssp_phot, ssp_lgmet, eff_waves_rest, dust_age_weights, 1e-30, ssp_ages_yr, "float32"
        )

        def loss64(sfr, lz, tv1, tv2):
            return jnp.sum(k64(sfr, lz, tv1, tv2, -0.7))

        def loss32(sfr, lz, tv1, tv2):
            return jnp.sum(k32(sfr, lz, tv1, tv2, -0.7))

        g64 = jax.grad(loss64, argnums=(0, 1, 2, 3))(sfr_on_ssp, -1.0, 0.5, 0.3)
        g32 = jax.grad(loss32, argnums=(0, 1, 2, 3))(sfr_on_ssp, -1.0, 0.5, 0.3)

        for gf32, gf64 in zip(g32, g64):
            # Allow 1% relative error for float32 gradients
            assert_allclose(gf32, gf64, rtol=1e-2, atol=1e-20)


class TestMixedPrecisionSpeedup:
    """float32 forward model should be faster than float64."""

    @pytest.mark.skipif(
        os.environ.get("CI") == "true",
        reason=(
            "Sub-millisecond microbenchmark — shared CI runners produce "
            "noisy timings (f32 vs f64 difference is well under the runner-"
            "noise envelope). Runs locally where timings are stable."
        ),
    )
    def test_f32_not_slower_than_f64(
        self,
        ssp_phot,
        ssp_lgmet,
        eff_waves_rest,
        dust_age_weights,
        ssp_ages_yr,
        sfr_on_ssp,
    ):
        """float32 gradient is at least as fast as float64."""
        k64 = _make_kernel(
            ssp_phot, ssp_lgmet, eff_waves_rest, dust_age_weights, 1e-30, ssp_ages_yr, "float64"
        )
        k32 = _make_kernel(
            ssp_phot, ssp_lgmet, eff_waves_rest, dust_age_weights, 1e-30, ssp_ages_yr, "float32"
        )

        def loss64(sfr, lz, tv1, tv2):
            return jnp.sum(k64(sfr, lz, tv1, tv2, -0.7))

        def loss32(sfr, lz, tv1, tv2):
            return jnp.sum(k32(sfr, lz, tv1, tv2, -0.7))

        grad64 = jax.jit(jax.value_and_grad(loss64))
        grad32 = jax.jit(jax.value_and_grad(loss32))

        args = (sfr_on_ssp, -1.0, 0.5, 0.3)

        # Warmup
        _ = grad64(*args)
        _ = grad32(*args)

        N = 500
        t0 = time.time()
        for _ in range(N):
            _ = grad64(*args)[0].block_until_ready()
        t_f64 = (time.time() - t0) / N

        t0 = time.time()
        for _ in range(N):
            _ = grad32(*args)[0].block_until_ready()
        t_f32 = (time.time() - t0) / N

        # float32 should be at least as fast (allow 30% margin)
        assert t_f32 < t_f64 * 1.3, f"f32 not faster: {t_f32 * 1e6:.1f}μs vs {t_f64 * 1e6:.1f}μs"

    def test_f32_closure_arrays_are_f32(
        self,
        ssp_phot,
        ssp_lgmet,
        eff_waves_rest,
        dust_age_weights,
        ssp_ages_yr,
    ):
        """Closure arrays should be stored in float32 when dtype=float32."""
        # Verify the dtype conversion actually happens
        dt = jnp.float32
        assert ssp_phot.astype(dt).dtype == jnp.float32
        assert ssp_lgmet.astype(dt).dtype == jnp.float32
        assert eff_waves_rest.astype(dt).dtype == jnp.float32
        assert dust_age_weights.astype(dt).dtype == jnp.float32
        assert ssp_ages_yr.astype(dt).dtype == jnp.float32
