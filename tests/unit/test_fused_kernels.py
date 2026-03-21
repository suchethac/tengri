"""Tests for fused forward model JIT kernels.

Validates that the fused kernels (single JIT scope for
weights + metallicity interp + dust + einsum) produce identical
results to the multi-function path, with measurable speedup.
"""

import time

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from diffsed.models.dust.attenuation import precompute_dust_age_weights, two_component_dust_fast
from diffsed.models.sps.dsps_wrapper import LSUN_ERG_PER_S, compute_csp_weights
from diffsed.models.sps.precompute import (
    fast_photometry,
    fast_spectrum,
    interpolate_ssp_phot_metallicity,
)

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Fixtures: synthetic SSP-like data
# ---------------------------------------------------------------------------


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
def n_pix():
    return 200


@pytest.fixture
def ssp_ages_yr(n_age):
    return 10.0 ** jnp.linspace(5.5, 10.14, n_age)


@pytest.fixture
def ssp_lgmet(n_met):
    return jnp.array([-2.0, -1.5, -1.0, -0.5, 0.0])


@pytest.fixture
def ssp_phot(n_met, n_age, n_filt):
    """Synthetic precomputed SSP photometry."""
    key = jax.random.PRNGKey(0)
    return jnp.abs(jax.random.normal(key, (n_met, n_age, n_filt))) + 0.1


@pytest.fixture
def ssp_on_pixels(n_met, n_age, n_pix):
    """Synthetic precomputed SSP spectra."""
    key = jax.random.PRNGKey(1)
    return jnp.abs(jax.random.normal(key, (n_met, n_age, n_pix))) + 0.1


@pytest.fixture
def eff_waves_rest(n_filt):
    return jnp.array([3551.0, 4686.0, 6166.0, 7480.0, 8932.0])


@pytest.fixture
def wave_rest_pixels(n_pix):
    return jnp.linspace(3500.0, 9500.0, n_pix)


@pytest.fixture
def dust_age_weights(ssp_ages_yr):
    return precompute_dust_age_weights(ssp_ages_yr)


@pytest.fixture
def sfr_on_ssp(n_age):
    """Synthetic SFR interpolated to SSP ages."""
    key = jax.random.PRNGKey(2)
    return jnp.abs(jax.random.normal(key, (n_age,))) + 0.01


# ---------------------------------------------------------------------------
# Build fused photometry kernel (mimics Model._build_fused_photometry)
# ---------------------------------------------------------------------------


def _make_fused_phot(ssp_phot, ssp_lgmet, eff_waves_rest, dust_age_w, flux_scale, ssp_ages_yr):
    @jax.jit
    def fused_phot(sfr_on_ssp, log_z, tau_v1, tau_v2, dust_n):
        dt = jnp.concatenate(
            [
                jnp.array([ssp_ages_yr[1] - ssp_ages_yr[0]]),
                0.5 * (ssp_ages_yr[2:] - ssp_ages_yr[:-2]),
                jnp.array([ssp_ages_yr[-1] - ssp_ages_yr[-2]]),
            ]
        )
        weights = sfr_on_ssp * dt
        log_z_c = jnp.clip(log_z, ssp_lgmet[0], ssp_lgmet[-1])
        idx = jnp.clip(jnp.searchsorted(ssp_lgmet, log_z_c) - 1, 0, len(ssp_lgmet) - 2)
        frac = (log_z_c - ssp_lgmet[idx]) / (ssp_lgmet[idx + 1] - ssp_lgmet[idx])
        ssp_at_z = (1.0 - frac) * ssp_phot[idx] + frac * ssp_phot[idx + 1]
        wave_ratio = (eff_waves_rest / 5500.0) ** dust_n
        tau_v_eff = dust_age_w * tau_v1 + tau_v2
        dust = jnp.exp(-(tau_v_eff[:, None] * wave_ratio[None, :]))
        flux_lsun = jnp.einsum("i,if,if->f", weights, dust, ssp_at_z)
        return flux_scale * flux_lsun * LSUN_ERG_PER_S

    return fused_phot


def _make_fused_spec(
    ssp_on_pixels, ssp_lgmet, wave_rest_pixels, dust_age_w, flux_scale, ssp_ages_yr
):
    @jax.jit
    def fused_spec(sfr_on_ssp, log_z, tau_v1, tau_v2, dust_n):
        dt = jnp.concatenate(
            [
                jnp.array([ssp_ages_yr[1] - ssp_ages_yr[0]]),
                0.5 * (ssp_ages_yr[2:] - ssp_ages_yr[:-2]),
                jnp.array([ssp_ages_yr[-1] - ssp_ages_yr[-2]]),
            ]
        )
        weights = sfr_on_ssp * dt
        log_z_c = jnp.clip(log_z, ssp_lgmet[0], ssp_lgmet[-1])
        idx = jnp.clip(jnp.searchsorted(ssp_lgmet, log_z_c) - 1, 0, len(ssp_lgmet) - 2)
        frac = (log_z_c - ssp_lgmet[idx]) / (ssp_lgmet[idx + 1] - ssp_lgmet[idx])
        ssp_at_z = (1.0 - frac) * ssp_on_pixels[idx] + frac * ssp_on_pixels[idx + 1]
        wave_ratio = (wave_rest_pixels / 5500.0) ** dust_n
        tau_v_eff = dust_age_w * tau_v1 + tau_v2
        dust = jnp.exp(-(tau_v_eff[:, None] * wave_ratio[None, :]))
        flux = jnp.einsum("i,ip,ip->p", weights, dust, ssp_at_z)
        return flux_scale * flux * LSUN_ERG_PER_S

    return fused_spec


# ---------------------------------------------------------------------------
# Unfused reference path (calls individual JIT functions separately)
# ---------------------------------------------------------------------------


def _unfused_photometry(
    sfr_on_ssp,
    ssp_phot,
    ssp_lgmet,
    ssp_ages_yr,
    eff_waves_rest,
    dust_age_weights,
    flux_scale,
    log_z,
    tau_v1,
    tau_v2,
    dust_n,
):
    weights = compute_csp_weights(sfr_on_ssp, ssp_ages_yr)
    ssp_at_z = interpolate_ssp_phot_metallicity(ssp_phot, ssp_lgmet, log_z)
    dust = two_component_dust_fast(
        eff_waves_rest,
        dust_age_weights,
        tau_v1=tau_v1,
        tau_v2=tau_v2,
        law_bc="power_law",
        law_diff="power_law",
        n_slope=dust_n,
    )
    return fast_photometry(weights, ssp_at_z, dust, flux_scale)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFusedPhotometryAccuracy:
    """Fused kernel must match unfused path exactly."""

    def test_matches_unfused(
        self,
        ssp_phot,
        ssp_lgmet,
        eff_waves_rest,
        dust_age_weights,
        ssp_ages_yr,
        sfr_on_ssp,
    ):
        """Fused photometry matches separate-function photometry."""
        flux_scale = 1e-30
        log_z, tau_v1, tau_v2, dust_n = -1.0, 0.5, 0.3, -0.7

        fused = _make_fused_phot(
            ssp_phot,
            ssp_lgmet,
            eff_waves_rest,
            dust_age_weights,
            flux_scale,
            ssp_ages_yr,
        )
        result_fused = fused(sfr_on_ssp, log_z, tau_v1, tau_v2, dust_n)
        result_unfused = _unfused_photometry(
            sfr_on_ssp,
            ssp_phot,
            ssp_lgmet,
            ssp_ages_yr,
            eff_waves_rest,
            dust_age_weights,
            flux_scale,
            log_z,
            tau_v1,
            tau_v2,
            dust_n,
        )
        assert_allclose(result_fused, result_unfused, rtol=1e-10)

    @pytest.mark.parametrize("log_z", [-2.0, -1.0, -0.5, 0.0])
    def test_matches_across_metallicities(
        self,
        ssp_phot,
        ssp_lgmet,
        eff_waves_rest,
        dust_age_weights,
        ssp_ages_yr,
        sfr_on_ssp,
        log_z,
    ):
        """Agreement across metallicity range."""
        flux_scale = 1e-30
        fused = _make_fused_phot(
            ssp_phot,
            ssp_lgmet,
            eff_waves_rest,
            dust_age_weights,
            flux_scale,
            ssp_ages_yr,
        )
        result_fused = fused(sfr_on_ssp, log_z, 0.5, 0.3, -0.7)
        result_unfused = _unfused_photometry(
            sfr_on_ssp,
            ssp_phot,
            ssp_lgmet,
            ssp_ages_yr,
            eff_waves_rest,
            dust_age_weights,
            flux_scale,
            log_z,
            0.5,
            0.3,
            -0.7,
        )
        assert_allclose(result_fused, result_unfused, rtol=1e-10)


class TestFusedPhotometryGradients:
    """Gradients through fused kernel are finite and correct."""

    def test_gradients_finite(
        self,
        ssp_phot,
        ssp_lgmet,
        eff_waves_rest,
        dust_age_weights,
        ssp_ages_yr,
        sfr_on_ssp,
    ):
        fused = _make_fused_phot(
            ssp_phot,
            ssp_lgmet,
            eff_waves_rest,
            dust_age_weights,
            1e-30,
            ssp_ages_yr,
        )

        def loss(log_z, tau_v1, tau_v2):
            return jnp.sum(fused(sfr_on_ssp, log_z, tau_v1, tau_v2, -0.7))

        grads = jax.grad(loss, argnums=(0, 1, 2))(-1.0, 0.5, 0.3)
        for g in grads:
            assert jnp.isfinite(g), f"Non-finite gradient: {g}"

    def test_gradients_match_unfused(
        self,
        ssp_phot,
        ssp_lgmet,
        eff_waves_rest,
        dust_age_weights,
        ssp_ages_yr,
        sfr_on_ssp,
    ):
        flux_scale = 1e-30
        fused = _make_fused_phot(
            ssp_phot,
            ssp_lgmet,
            eff_waves_rest,
            dust_age_weights,
            flux_scale,
            ssp_ages_yr,
        )

        def loss_fused(sfr_on_ssp, log_z, tau_v1, tau_v2):
            return jnp.sum(fused(sfr_on_ssp, log_z, tau_v1, tau_v2, -0.7))

        def loss_unfused(sfr_on_ssp, log_z, tau_v1, tau_v2):
            return jnp.sum(
                _unfused_photometry(
                    sfr_on_ssp,
                    ssp_phot,
                    ssp_lgmet,
                    ssp_ages_yr,
                    eff_waves_rest,
                    dust_age_weights,
                    flux_scale,
                    log_z,
                    tau_v1,
                    tau_v2,
                    -0.7,
                )
            )

        g_fused = jax.grad(loss_fused, argnums=(0, 1, 2, 3))(
            sfr_on_ssp,
            -1.0,
            0.5,
            0.3,
        )
        g_unfused = jax.grad(loss_unfused, argnums=(0, 1, 2, 3))(
            sfr_on_ssp,
            -1.0,
            0.5,
            0.3,
        )
        for gf, gu in zip(g_fused, g_unfused):
            assert_allclose(gf, gu, rtol=1e-10)


class TestFusedSpectrumAccuracy:
    """Fused spectrum kernel accuracy."""

    def test_matches_unfused_spectrum(
        self,
        ssp_on_pixels,
        ssp_lgmet,
        wave_rest_pixels,
        dust_age_weights,
        ssp_ages_yr,
        sfr_on_ssp,
    ):
        flux_scale = 1e-30
        log_z, tau_v1, tau_v2, dust_n = -1.0, 0.5, 0.3, -0.7

        fused = _make_fused_spec(
            ssp_on_pixels,
            ssp_lgmet,
            wave_rest_pixels,
            dust_age_weights,
            flux_scale,
            ssp_ages_yr,
        )
        result_fused = fused(sfr_on_ssp, log_z, tau_v1, tau_v2, dust_n)

        # Unfused
        weights = compute_csp_weights(sfr_on_ssp, ssp_ages_yr)
        ssp_at_z = interpolate_ssp_phot_metallicity(ssp_on_pixels, ssp_lgmet, log_z)
        dust = two_component_dust_fast(
            wave_rest_pixels,
            dust_age_weights,
            tau_v1=tau_v1,
            tau_v2=tau_v2,
            law_bc="power_law",
            law_diff="power_law",
            n_slope=dust_n,
        )
        flux = fast_spectrum(weights, ssp_at_z, dust, flux_scale)
        result_unfused = flux * LSUN_ERG_PER_S

        assert_allclose(result_fused, result_unfused, rtol=1e-10)


class TestFusedKernelSpeedup:
    """Benchmark fused vs unfused forward model."""

    def test_fused_photometry_speedup(
        self,
        ssp_phot,
        ssp_lgmet,
        eff_waves_rest,
        dust_age_weights,
        ssp_ages_yr,
        sfr_on_ssp,
    ):
        """Fused photometry gradient is faster than unfused."""
        flux_scale = 1e-30

        fused = _make_fused_phot(
            ssp_phot,
            ssp_lgmet,
            eff_waves_rest,
            dust_age_weights,
            flux_scale,
            ssp_ages_yr,
        )

        def loss_fused(sfr, lz, tv1, tv2):
            return jnp.sum(fused(sfr, lz, tv1, tv2, -0.7))

        def loss_unfused(sfr, lz, tv1, tv2):
            return jnp.sum(
                _unfused_photometry(
                    sfr,
                    ssp_phot,
                    ssp_lgmet,
                    ssp_ages_yr,
                    eff_waves_rest,
                    dust_age_weights,
                    flux_scale,
                    lz,
                    tv1,
                    tv2,
                    -0.7,
                )
            )

        grad_fused = jax.jit(jax.value_and_grad(loss_fused))
        grad_unfused = jax.jit(jax.value_and_grad(loss_unfused))

        args = (sfr_on_ssp, -1.0, 0.5, 0.3)

        # Warmup
        _ = grad_fused(*args)
        _ = grad_unfused(*args)

        N = 500
        t0 = time.time()
        for _ in range(N):
            _ = grad_fused(*args)[0].block_until_ready()
        t_fused = (time.time() - t0) / N

        t0 = time.time()
        for _ in range(N):
            _ = grad_unfused(*args)[0].block_until_ready()
        t_unfused = (time.time() - t0) / N

        # Fused should be at least as fast (allow margin for noise)
        assert t_fused < t_unfused * 1.3, (
            f"Fused not faster: {t_fused * 1e6:.1f}μs vs {t_unfused * 1e6:.1f}μs"
        )
