# SPDX-License-Identifier: BSD-3-Clause
"""Tests for time-evolving metallicity support.

Verifies:
1. interpolate_metallicity_evolving gives correct per-age-bin results
2. compute_log_z_evolving produces correct ramp shapes
3. Backward compatibility: constant Z(t) matches scalar interpolation
4. Gradients flow through the evolving metallicity path
5. Parameters correctly adds/removes parameters based on setting
"""

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from tengri.components.stellar.sps.dsps_wrapper import (
    compute_log_z_evolving,
    interpolate_mass_remaining,
    interpolate_mass_remaining_evolving,
    interpolate_metallicity,
    interpolate_metallicity_evolving,
)
from tengri.parameters.parameters import Parameters
from tests._grad_parity import assert_grad_matches_fd

pytestmark = pytest.mark.bounds


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite difference: (f(x+eps) - f(x-eps)) / (2*eps)."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def ssp_lgmet():
    """Simple 5-point metallicity grid in log10(Z)."""
    return jnp.array([-4.0, -3.0, -2.0, -1.0, 0.0])


@pytest.fixture
def ssp_flux(ssp_lgmet):
    """Synthetic SSP flux: (n_met, n_age, n_wave).

    Flux increases linearly with metallicity index so interpolation
    results are analytically predictable.
    """
    n_met = len(ssp_lgmet)
    n_age = 10
    n_wave = 20
    # flux[m, a, w] = (m + 1) * (a + 1) * (w + 1) — easy to verify
    met_idx = jnp.arange(n_met, dtype=jnp.float64)[:, None, None]
    age_idx = jnp.arange(n_age, dtype=jnp.float64)[None, :, None]
    wave_idx = jnp.arange(n_wave, dtype=jnp.float64)[None, None, :]
    return (met_idx + 1.0) * (age_idx + 1.0) * (wave_idx + 1.0)


@pytest.fixture
def ssp_mass_remaining(ssp_lgmet):
    """Synthetic mass-remaining: (n_met, n_age).

    Linearly increasing with metallicity index.
    """
    n_met = len(ssp_lgmet)
    n_age = 10
    met_idx = jnp.arange(n_met, dtype=jnp.float64)[:, None]
    age_idx = jnp.arange(n_age, dtype=jnp.float64)[None, :]
    return 0.1 + 0.1 * met_idx + 0.01 * age_idx


@pytest.fixture
def ssp_lg_age_gyr():
    """Log10(age/Gyr) grid with 10 age bins spanning 1 Myr to 13 Gyr."""
    return jnp.linspace(-3.0, 1.114, 10)


# ── Tests: interpolate_metallicity_evolving ───────────────────────


class TestInterpolateMetallicityEvolving:
    """Tests for per-age-bin metallicity interpolation."""

    def test_constant_z_matches_scalar(self, ssp_flux, ssp_lgmet):
        """Constant Z(t) = z0 should match scalar interpolation exactly."""
        log_z = -1.5
        n_age = ssp_flux.shape[1]
        log_z_per_age = jnp.full(n_age, log_z)

        result_evolving = interpolate_metallicity_evolving(ssp_flux, ssp_lgmet, log_z_per_age)
        result_scalar = interpolate_metallicity(ssp_flux, ssp_lgmet, log_z)

        assert_allclose(result_evolving, result_scalar, atol=1e-12)

    def test_on_grid_points(self, ssp_flux, ssp_lgmet):
        """When Z(t) lands exactly on grid points, use that grid slice."""
        n_age = ssp_flux.shape[1]
        # Each age bin uses a different grid point
        log_z_per_age = jnp.array([ssp_lgmet[i % len(ssp_lgmet)] for i in range(n_age)])
        result = interpolate_metallicity_evolving(ssp_flux, ssp_lgmet, log_z_per_age)

        for i in range(n_age):
            met_idx = i % len(ssp_lgmet)
            expected = ssp_flux[met_idx, i, :]
            assert_allclose(result[i], expected, atol=1e-12, err_msg=f"Age bin {i} mismatch")

    def test_output_shape(self, ssp_flux, ssp_lgmet):
        """Output should be (n_age, n_wave)."""
        n_age = ssp_flux.shape[1]
        log_z_per_age = jnp.zeros(n_age)
        result = interpolate_metallicity_evolving(ssp_flux, ssp_lgmet, log_z_per_age)
        chex.assert_shape(result, (ssp_flux.shape[1], ssp_flux.shape[2]))

    def test_clamping_at_bounds(self, ssp_flux, ssp_lgmet):
        """Values outside grid should clamp to boundary."""
        n_age = ssp_flux.shape[1]

        # All below minimum
        log_z_low = jnp.full(n_age, -10.0)
        result_low = interpolate_metallicity_evolving(ssp_flux, ssp_lgmet, log_z_low)
        expected_low = ssp_flux[0]  # lowest metallicity slice
        assert_allclose(result_low, expected_low, atol=1e-12)

        # All above maximum
        log_z_high = jnp.full(n_age, 5.0)
        result_high = interpolate_metallicity_evolving(ssp_flux, ssp_lgmet, log_z_high)
        expected_high = ssp_flux[-1]  # highest metallicity slice
        assert_allclose(result_high, expected_high, atol=1e-12)

    def test_different_z_per_age(self, ssp_flux, ssp_lgmet):
        """Different metallicities per age bin should give different results."""
        n_age = ssp_flux.shape[1]
        # Low Z for young stars, high Z for old stars
        log_z_per_age = jnp.linspace(-3.5, -0.5, n_age)
        result = interpolate_metallicity_evolving(ssp_flux, ssp_lgmet, log_z_per_age)

        # Young bins (low Z) should have lower flux than old bins (high Z)
        # because our synthetic flux increases with metallicity index
        assert float(jnp.mean(result[0])) < float(jnp.mean(result[-1]))


# ── Tests: interpolate_mass_remaining_evolving ────────────────────


class TestInterpolateMassRemainingEvolving:
    """Tests for per-age-bin mass-remaining interpolation."""

    def test_constant_z_matches_scalar(self, ssp_mass_remaining, ssp_lgmet):
        """Constant Z(t) should match scalar interpolation."""
        log_z = -1.5
        n_age = ssp_mass_remaining.shape[1]
        log_z_per_age = jnp.full(n_age, log_z)

        result_evolving = interpolate_mass_remaining_evolving(
            ssp_mass_remaining, ssp_lgmet, log_z_per_age
        )
        result_scalar = interpolate_mass_remaining(ssp_mass_remaining, ssp_lgmet, log_z)

        assert_allclose(result_evolving, result_scalar, atol=1e-12)

    def test_output_shape(self, ssp_mass_remaining, ssp_lgmet):
        """Output should be (n_age,)."""
        n_age = ssp_mass_remaining.shape[1]
        log_z_per_age = jnp.zeros(n_age)
        result = interpolate_mass_remaining_evolving(ssp_mass_remaining, ssp_lgmet, log_z_per_age)
        chex.assert_shape(result, (n_age,))


# ── Tests: compute_log_z_evolving ─────────────────────────────────


class TestComputeLogZEvolving:
    """Tests for the linear-in-log metallicity ramp."""

    def test_endpoints(self, ssp_lg_age_gyr):
        """Youngest bin -> final, oldest bin -> initial (when age = t_universe)."""
        t_universe_gyr = 13.0
        log_z_initial = -3.0
        log_z_final = -1.0

        result = compute_log_z_evolving(ssp_lg_age_gyr, log_z_initial, log_z_final, t_universe_gyr)

        # Youngest age (smallest lookback) should be closest to final
        # Oldest age should approach initial (if age ~ t_universe)
        assert float(result[0]) == pytest.approx(float(log_z_final), abs=0.5)

        # The trend should be: younger bins closer to final, older bins closer to initial
        # (monotonically decreasing from final toward initial as age increases)
        assert float(result[0]) > float(result[-1])

    def test_constant_when_equal(self, ssp_lg_age_gyr):
        """When initial == final, all bins get the same metallicity."""
        log_z = -1.5
        result = compute_log_z_evolving(ssp_lg_age_gyr, log_z, log_z, 13.0)
        assert_allclose(result, log_z, atol=1e-12)

    def test_linear_ramp_shape(self, ssp_lg_age_gyr):
        """Metallicity should be linear in lookback time (not log-age)."""
        t_universe_gyr = 13.0
        log_z_initial = -2.0
        log_z_final = 0.0

        result = compute_log_z_evolving(ssp_lg_age_gyr, log_z_initial, log_z_final, t_universe_gyr)

        age_gyr = 10.0**ssp_lg_age_gyr
        t_frac = jnp.clip(age_gyr / t_universe_gyr, 0.0, 1.0)
        expected = log_z_final + (log_z_initial - log_z_final) * t_frac

        assert_allclose(result, expected, atol=1e-12)

    def test_clamping_old_ages(self):
        """Ages > t_universe should clamp to initial metallicity."""
        # SSP ages that exceed the universe age
        ssp_lg_age_gyr = jnp.array([-1.0, 0.0, 1.0, 1.5])  # 0.1, 1, 10, ~31 Gyr
        t_universe_gyr = 5.0
        log_z_initial = -2.0
        log_z_final = 0.0

        result = compute_log_z_evolving(ssp_lg_age_gyr, log_z_initial, log_z_final, t_universe_gyr)

        # Ages > 5 Gyr should clamp to initial
        assert float(result[-1]) == pytest.approx(log_z_initial, abs=1e-10)
        assert float(result[-2]) == pytest.approx(log_z_initial, abs=1e-10)


# ── Tests: Gradient flow ──────────────────────────────────────────


class TestEvolvingMetallicityGradients:
    """Verify gradients flow through the evolving metallicity path."""

    def test_grad_wrt_log_z_per_age(self, ssp_flux, ssp_lgmet):
        """Gradients w.r.t. log_z_per_age should be finite and non-zero."""
        n_age = ssp_flux.shape[1]
        log_z_per_age = jnp.linspace(-3.0, -1.0, n_age)

        def loss(log_z_arr):
            flux = interpolate_metallicity_evolving(ssp_flux, ssp_lgmet, log_z_arr)
            return jnp.sum(flux**2)

        grad = assert_grad_matches_fd(loss, log_z_per_age)
        chex.assert_tree_all_finite(grad)
        assert jnp.any(grad != 0.0), "Gradients must be non-zero"
        assert jnp.all(jnp.isfinite(grad)), (
            "`grad` is non-finite — non-zero is not enough, `nan != 0.0` is True "
            "and a NaN satisfies a non-zero assertion (#2178)"
        )

    def test_grad_wrt_ramp_params(self, ssp_flux, ssp_lgmet, ssp_lg_age_gyr):
        """Gradients w.r.t. ramp parameters (initial, final) should flow."""
        t_universe_gyr = 13.0

        def loss_init(log_z_initial):
            log_z_per_age = compute_log_z_evolving(
                ssp_lg_age_gyr, log_z_initial, -0.5, t_universe_gyr
            )
            flux = interpolate_metallicity_evolving(ssp_flux, ssp_lgmet, log_z_per_age)
            return jnp.sum(flux**2)

        def loss_final(log_z_final):
            log_z_per_age = compute_log_z_evolving(
                ssp_lg_age_gyr, -2.0, log_z_final, t_universe_gyr
            )
            flux = interpolate_metallicity_evolving(ssp_flux, ssp_lgmet, log_z_per_age)
            return jnp.sum(flux**2)

        g_jax_init = float(jax.grad(loss_init)(-2.0))
        g_fd_init = fd_grad(loss_init, -2.0)
        np.testing.assert_allclose(
            g_jax_init,
            g_fd_init,
            rtol=1e-3,
            err_msg=f"autodiff={g_jax_init:.4e}, FD={g_fd_init:.4e}",
        )

        g_jax_final = float(jax.grad(loss_final)(-0.5))
        g_fd_final = fd_grad(loss_final, -0.5)
        np.testing.assert_allclose(
            g_jax_final,
            g_fd_final,
            rtol=1e-3,
            err_msg=f"autodiff={g_jax_final:.4e}, FD={g_fd_final:.4e}",
        )

        assert g_jax_init != 0.0, "Gradient w.r.t. initial Z must be non-zero"
        assert np.all(np.isfinite(g_jax_init)), (
            "`g_jax_init` is non-finite — non-zero is not enough, `nan != 0.0` is True "
            "and a NaN satisfies a non-zero assertion (#2178)"
        )
        assert g_jax_final != 0.0, "Gradient w.r.t. final Z must be non-zero"
        assert np.all(np.isfinite(g_jax_final)), (
            "`g_jax_final` is non-finite — non-zero is not enough, `nan != 0.0` is True "
            "and a NaN satisfies a non-zero assertion (#2178)"
        )

    def test_grad_autodiff_vs_finite_diff(self, ssp_flux, ssp_lgmet, ssp_lg_age_gyr):
        """Autodiff gradient matches central finite differences."""
        t_universe_gyr = 13.0

        def loss(log_z_final):
            log_z_per_age = compute_log_z_evolving(
                ssp_lg_age_gyr, -2.0, log_z_final, t_universe_gyr
            )
            flux = interpolate_metallicity_evolving(ssp_flux, ssp_lgmet, log_z_per_age)
            return jnp.sum(flux**2)

        val = -0.5
        grad_auto = float(jax.grad(loss)(val))

        eps = 1e-5
        grad_fd = float((loss(val + eps) - loss(val - eps)) / (2 * eps))

        assert_allclose(grad_auto, grad_fd, rtol=1e-3, err_msg="Autodiff vs finite-diff mismatch")


# ── Tests: Parameters integration ──────────────────────────────────


class TestParamSpecEvolvingMetallicity:
    """Verify Parameters correctly handles evolving_metallicity setting."""

    def test_default_has_met_logzsol(self):
        """Default Parameters should have met_logzsol, not evolving params."""
        spec = Parameters()
        assert "met_logzsol" in spec.all_params
        assert "met_logzsol_0" not in spec.all_params
        assert "met_logzsol_final" not in spec.all_params

    def test_evolving_replaces_scalar(self):
        """evolving_metallicity=True should replace met_logzsol with two params."""
        spec = Parameters(evolving_metallicity=True)
        assert "met_logzsol" not in spec.all_params
        assert "met_logzsol_0" in spec.all_params
        assert "met_logzsol_final" in spec.all_params

    def test_evolving_custom_priors(self):
        """Custom priors should work for evolving metallicity params."""
        from tengri.parameters.priors import Uniform

        spec = Parameters(
            evolving_metallicity=True,
            met_logzsol_0=Uniform(-3.0, -1.0),
            met_logzsol_final=Uniform(-1.0, 0.5),
        )
        d0 = spec.get_distribution("met_logzsol_0")
        d1 = spec.get_distribution("met_logzsol_final")
        assert d0.bounds == (-3.0, -1.0)
        assert d1.bounds == (-1.0, 0.5)

    def test_sample_has_evolving_keys(self):
        """Sampling from evolving Parameters should produce the right keys."""
        spec = Parameters(evolving_metallicity=True)
        params = spec.sample(jax.random.PRNGKey(0))
        assert "met_logzsol_0" in params
        assert "met_logzsol_final" in params
        assert "met_logzsol" not in params

    def test_backward_compat_no_setting(self):
        """Omitting evolving_metallicity should give old behavior."""
        spec = Parameters(met_logzsol=-0.3)
        assert "met_logzsol" in spec.all_params
        assert spec.get_distribution("met_logzsol").is_fixed
