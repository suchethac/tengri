# SPDX-License-Identifier: BSD-3-Clause
"""Tests for fused forward model JIT kernels.

Validates that the fused kernels (single JIT scope for
weights + metallicity interp + dust + einsum) produce identical
results to the multi-function path, and that the fused path
compiles to strictly less work.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from tengri.components.dust.attenuation import precompute_dust_age_weights, two_component_dust_fast
from tengri.components.stellar.sps.dsps_wrapper import LSUN_ERG_PER_S, compute_csp_weights
from tengri.components.stellar.sps.precompute import (
    fast_photometry,
    fast_spectrum,
    interpolate_ssp_phot_metallicity,
)
from tengri.utils.scale import apply_log10_scale

pytestmark = pytest.mark.bounds


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite difference: (f(x+eps) - f(x-eps)) / (2*eps)."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


def setup_module(_module):
    """Clear XLA JIT cache before this module runs.

    Some earlier test (in random pytest ordering) may leave compiled
    functions with different shapes/dtypes in the in-process cache.
    The fused kernel tests use synthetic data with specific shapes;
    a stale cache entry causes shape-mismatch failures that only appear
    when pytest randomly places certain tests before this module.
    """
    jax.clear_caches()


# ── Fixtures: synthetic SSP-like data ─────────────────────────────


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


# ── Build compositional photometry kernel ─────────────────────────


def _make_fused_phot(
    ssp_phot, ssp_lgmet, eff_waves_rest, dust_age_w, log10_flux_scale, ssp_ages_yr
):
    @jax.jit
    def fused_phot(sfr_on_ssp, log_z, tau_v1, tau_v2, dust_n):
        dt = jnp.concatenate(
            [
                jnp.array([0.5 * (ssp_ages_yr[1] - ssp_ages_yr[0])]),
                0.5 * (ssp_ages_yr[2:] - ssp_ages_yr[:-2]),
                jnp.array([0.5 * (ssp_ages_yr[-1] - ssp_ages_yr[-2])]),
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
        return apply_log10_scale(flux_lsun * LSUN_ERG_PER_S, log10_flux_scale)

    return fused_phot


def _make_fused_spec(
    ssp_on_pixels, ssp_lgmet, wave_rest_pixels, dust_age_w, log10_flux_scale, ssp_ages_yr
):
    @jax.jit
    def fused_spec(sfr_on_ssp, log_z, tau_v1, tau_v2, dust_n):
        dt = jnp.concatenate(
            [
                jnp.array([0.5 * (ssp_ages_yr[1] - ssp_ages_yr[0])]),
                0.5 * (ssp_ages_yr[2:] - ssp_ages_yr[:-2]),
                jnp.array([0.5 * (ssp_ages_yr[-1] - ssp_ages_yr[-2])]),
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
        return apply_log10_scale(flux, log10_flux_scale) * LSUN_ERG_PER_S

    return fused_spec


# ── Unfused reference path (calls individual JIT functions separately)


def _unfused_photometry(
    sfr_on_ssp,
    ssp_phot,
    ssp_lgmet,
    ssp_ages_yr,
    eff_waves_rest,
    dust_age_weights,
    log10_flux_scale,
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
    return fast_photometry(weights, ssp_at_z, dust, log10_flux_scale)


# ── Tests ─────────────────────────────────────────────────────────


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
        log10_flux_scale = -30.0
        log_z, tau_v1, tau_v2, dust_n = -1.0, 0.5, 0.3, -0.7

        fused = _make_fused_phot(
            ssp_phot,
            ssp_lgmet,
            eff_waves_rest,
            dust_age_weights,
            log10_flux_scale,
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
            log10_flux_scale,
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
        log10_flux_scale = -30.0
        fused = _make_fused_phot(
            ssp_phot,
            ssp_lgmet,
            eff_waves_rest,
            dust_age_weights,
            log10_flux_scale,
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
            log10_flux_scale,
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

        # Use -1.2 (interior of [-1.5, -1.0] cell), NOT -1.0 (an exact grid node).
        # At grid nodes, jnp.searchsorted places x+eps and x-eps in different cells,
        # so FD straddles a kink in the bilinear interpolant while autodiff stays within
        # one cell — giving systematically different values (~30% mismatch is typical).
        grads = jax.grad(loss, argnums=(0, 1, 2))(-1.2, 0.5, 0.3)

        # Test each gradient against FD approximation
        def loss_lz(lz):
            return jnp.sum(fused(sfr_on_ssp, lz, 0.5, 0.3, -0.7))

        def loss_tv1(tv1):
            return jnp.sum(fused(sfr_on_ssp, -1.2, tv1, 0.3, -0.7))

        def loss_tv2(tv2):
            return jnp.sum(fused(sfr_on_ssp, -1.2, 0.5, tv2, -0.7))

        grad_lz_jax = float(grads[0])
        grad_lz_fd = fd_grad(loss_lz, -1.2)
        np.testing.assert_allclose(
            grad_lz_jax,
            grad_lz_fd,
            rtol=1e-3,
            err_msg=f"log_z: autodiff={grad_lz_jax:.4e}, FD={grad_lz_fd:.4e}",
        )

        grad_tv1_jax = float(grads[1])
        grad_tv1_fd = fd_grad(loss_tv1, 0.5)
        np.testing.assert_allclose(
            grad_tv1_jax,
            grad_tv1_fd,
            rtol=1e-3,
            err_msg=f"tau_v1: autodiff={grad_tv1_jax:.4e}, FD={grad_tv1_fd:.4e}",
        )

        grad_tv2_jax = float(grads[2])
        grad_tv2_fd = fd_grad(loss_tv2, 0.3)
        np.testing.assert_allclose(
            grad_tv2_jax,
            grad_tv2_fd,
            rtol=1e-3,
            err_msg=f"tau_v2: autodiff={grad_tv2_jax:.4e}, FD={grad_tv2_fd:.4e}",
        )

    def test_gradients_match_unfused(
        self,
        ssp_phot,
        ssp_lgmet,
        eff_waves_rest,
        dust_age_weights,
        ssp_ages_yr,
        sfr_on_ssp,
    ):
        log10_flux_scale = -30.0
        fused = _make_fused_phot(
            ssp_phot,
            ssp_lgmet,
            eff_waves_rest,
            dust_age_weights,
            log10_flux_scale,
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
                    log10_flux_scale,
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
        log10_flux_scale = -30.0
        log_z, tau_v1, tau_v2, dust_n = -1.0, 0.5, 0.3, -0.7

        fused = _make_fused_spec(
            ssp_on_pixels,
            ssp_lgmet,
            wave_rest_pixels,
            dust_age_weights,
            log10_flux_scale,
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
        flux = fast_spectrum(weights, ssp_at_z, dust, log10_flux_scale)
        result_unfused = flux * LSUN_ERG_PER_S

        assert_allclose(result_fused, result_unfused, rtol=1e-10)


class TestFusedKernelSpeedup:
    """Fused kernel must compile to strictly less work than unfused."""

    def test_fused_photometry_compiles_to_fewer_flops(
        self,
        ssp_phot,
        ssp_lgmet,
        eff_waves_rest,
        dust_age_weights,
        ssp_ages_yr,
        sfr_on_ssp,
    ):
        """The fused photometry gradient compiles to strictly fewer FLOPs than unfused.

        This guard exists to catch the fused kernel silently falling back to the
        unfused path. It used to assert a wall-clock ratio (``t_fused < t_unfused * 3.0``)
        and was marked ``benchmark`` so it ran serially. That formulation could not do
        the job because a fused kernel that degrades to an unfused call would compile
        to an identical cost (the two paths are bit-identical under XLA constant folding),
        making the regression undetectable by timing alone.

        Passing arrays as **traced arguments** blocks constant folding and the real
        difference appears: compiled FLOP counts come from the compiled executable, are
        identical across recompiles, and cannot be moved by scheduler load — so this
        runs in the ordinary parallel sweep and the ``benchmark`` marker no longer applies.

        This pattern mirrors #1696: trace the arrays that define path selection.
        """

        def _flops(fn, *args) -> float:
            """Compiled FLOP count — deterministic, unlike wall clock."""
            return jax.jit(fn).lower(*args).compile().cost_analysis()["flops"]

        log10_flux_scale = -30.0

        # The fused kernel must be TRACED, not closed over. Closed over, XLA
        # folds both the fused and unfused paths to identical cost, which is
        # what made the old wall-clock form blind. We pass the precomputed
        # tables as traced arguments so the two paths have distinct code.
        flops_unfused = _flops(
            lambda sfr, lz, tv1, tv2, ssp_p, ssp_m, ages, dw: jnp.sum(
                _unfused_photometry(
                    sfr,
                    ssp_p,
                    ssp_m,
                    ages,
                    eff_waves_rest,
                    dw,
                    log10_flux_scale,
                    lz,
                    tv1,
                    tv2,
                    -0.7,
                )
            ),
            sfr_on_ssp,
            -1.0,
            0.5,
            0.3,
            ssp_phot,
            ssp_lgmet,
            ssp_ages_yr,
            dust_age_weights,
        )

        # Build the fused fn INSIDE the traced function so the SSP tables and
        # dust weights are traced operands in this arm too. Binding them into
        # the closure before lowering would hand XLA constants to fold -- the
        # exact #1696 blindness this guard exists to prevent -- and make the
        # two arms incomparable (folded constants vs traced arrays).
        flops_fused = _flops(
            lambda sfr, lz, tv1, tv2, ssp_p, ssp_m, ages, dw: jnp.sum(
                _make_fused_phot(
                    ssp_p,
                    ssp_m,
                    eff_waves_rest,
                    dw,
                    log10_flux_scale,
                    ages,
                )(sfr, lz, tv1, tv2, -0.7)
            ),
            sfr_on_ssp,
            -1.0,
            0.5,
            0.3,
            ssp_phot,
            ssp_lgmet,
            ssp_ages_yr,
            dust_age_weights,
        )

        assert flops_fused < flops_unfused, (
            f"Fused kernel does not do less work: fused {flops_fused:,.0f} FLOPs "
            f"vs unfused {flops_unfused:,.0f}. Equal counts mean the fused kernel "
            f"is not actually fusing the computation."
        )

    def test_fused_photometry_mutation_degrades_path(
        self,
        ssp_phot,
        ssp_lgmet,
        eff_waves_rest,
        dust_age_weights,
        ssp_ages_yr,
        sfr_on_ssp,
    ):
        """Mutation: when fused path degrades to unfused, the FLOP assert fails.

        This verifies the guard catches the regression it exists for. If the
        fused kernel is bypassed and the unfused path is called instead, the
        compiled FLOPs become equal (or more), and the assertion must fail.
        """

        def _flops(fn, *args) -> float:
            """Compiled FLOP count — deterministic, unlike wall clock."""
            return jax.jit(fn).lower(*args).compile().cost_analysis()["flops"]

        log10_flux_scale = -30.0

        # Unfused path (unchanged)
        flops_unfused = _flops(
            lambda sfr, lz, tv1, tv2, ssp_p, ssp_m, ages, dw: jnp.sum(
                _unfused_photometry(
                    sfr,
                    ssp_p,
                    ssp_m,
                    ages,
                    eff_waves_rest,
                    dw,
                    log10_flux_scale,
                    lz,
                    tv1,
                    tv2,
                    -0.7,
                )
            ),
            sfr_on_ssp,
            -1.0,
            0.5,
            0.3,
            ssp_phot,
            ssp_lgmet,
            ssp_ages_yr,
            dust_age_weights,
        )

        # MUTANT: Fused path replaced with unfused implementation
        # This simulates the degradation the guard should catch
        flops_fused_mutant = _flops(
            lambda sfr, lz, tv1, tv2, ssp_p, ssp_m, ages, dw: jnp.sum(
                _unfused_photometry(  # <-- MUTANT: unfused instead of fused
                    sfr,
                    ssp_p,
                    ssp_m,
                    ages,
                    eff_waves_rest,
                    dw,
                    log10_flux_scale,
                    lz,
                    tv1,
                    tv2,
                    -0.7,
                )
            ),
            sfr_on_ssp,
            -1.0,
            0.5,
            0.3,
            ssp_phot,
            ssp_lgmet,
            ssp_ages_yr,
            dust_age_weights,
        )

        # The FLOP counts must be equal (or close) under the mutation
        assert flops_fused_mutant >= flops_unfused * 0.95, (
            f"Mutation sanity check failed: mutant {flops_fused_mutant:,.0f} "
            f"should equal or exceed unfused {flops_unfused:,.0f}"
        )

        # The assertion should fail with the mutant (showing the guard works)
        with pytest.raises(AssertionError, match="does not do less work"):
            assert flops_fused_mutant < flops_unfused, (
                f"Fused kernel does not do less work: "
                f"fused {flops_fused_mutant:,.0f} FLOPs "
                f"vs unfused {flops_unfused:,.0f}"
            )


# ── Regression: CSP trapezoidal endpoint weights (2026-04 bug fix)


class TestCSPEndpointWeights:
    """Regression: CSP endpoint weights must be half-width, not full-width.

    Previously the youngest and oldest SSP bins had full-width trapezoidal
    weights, over-counting their contribution by ~2x. Fixed in csp_age_dt().
    """

    def test_uniform_grid_endpoints_are_half_interior(self):
        """On a uniform grid, endpoints dt should be half the interior dt."""
        from tengri.components.stellar.sps.dsps_wrapper import csp_age_dt

        ages = jnp.linspace(1e6, 1e10, 100)
        dt = csp_age_dt(ages, method="trapz")
        interior = dt[1:-1]
        assert_allclose(
            float(dt[0]),
            float(interior[0]) / 2.0,
            rtol=1e-10,
            err_msg="Left endpoint should be half interior width",
        )
        assert_allclose(
            float(dt[-1]),
            float(interior[-1]) / 2.0,
            rtol=1e-10,
            err_msg="Right endpoint should be half interior width",
        )

    def test_log_grid_endpoints_are_half_interior(self):
        """On a log grid with log_trapz, same half-width rule holds."""
        from tengri.components.stellar.sps.dsps_wrapper import csp_age_dt

        ages = jnp.logspace(6, 10, 100)
        dt = csp_age_dt(ages, method="log_trapz")
        # First and last should be ~half of their nearest interior neighbors
        ratio_left = float(dt[0] / dt[1])
        ratio_right = float(dt[-1] / dt[-2])
        assert 0.4 < ratio_left < 0.6, f"Left endpoint ratio {ratio_left:.3f}, expected ~0.5"
        assert 0.4 < ratio_right < 0.6, f"Right endpoint ratio {ratio_right:.3f}, expected ~0.5"

    def test_constant_sfr_mass_integral_accurate(self):
        """With constant SFR=1 Msun/yr, total mass = age span in years."""
        from tengri.components.stellar.sps.dsps_wrapper import compute_csp_weights

        ages = jnp.logspace(6, 10, 200)
        sfr = jnp.ones_like(ages)  # 1 Msun/yr
        weights = compute_csp_weights(sfr, ages, method="trapz")
        total_mass = float(jnp.sum(weights))
        expected = float(ages[-1] - ages[0])  # years
        assert_allclose(
            total_mass,
            expected,
            rtol=0.01,
            err_msg=f"Total mass {total_mass:.2e} != age span {expected:.2e}",
        )
