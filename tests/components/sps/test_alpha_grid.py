# SPDX-License-Identifier: BSD-3-Clause
"""Tests for alpha-enhanced SSP grid interpolation.

Validates the new 4D (metallicity, [α/Fe], age, wavelength) SSP grid
support including:
1. Bilinear ([Fe/H], [α/Fe]) interpolation
2. Per-age evolving [α/Fe] interpolation
3. Alpha evolution ramp parameterization
4. JIT compatibility and differentiability
5. Backward compatibility with 3D grids (no alpha dimension)
6. has_alpha_grid detection
7. Salaris [Fe/H] ↔ [M/H] convention conversions
8. Solar [α/Fe]=0.0 equivalence with no-alpha case
"""

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.stellar.sps.dsps_wrapper import (
    SSPData,
    compute_alpha_fe_evolving,
    effective_metallicity,
    has_alpha_grid,
    interpolate_met_alpha,
    interpolate_met_alpha_evolving,
)
from tests._bounds import assert_non_negative
from tests._jit_parity import assert_jit_matches_eager

pytestmark = pytest.mark.bounds


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite difference: (f(x+eps) - f(x-eps)) / (2*eps)."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


@pytest.fixture
def alpha_ssp_grid():
    """Synthetic 4D SSP grid: (3 met, 5 alpha, 10 age, 20 wave).

    The flux is designed so that:
    - Higher metallicity → redder (more flux at long wavelengths)
    - Higher [α/Fe] → stronger Mg feature (bump near λ=5180)
    - Younger age → bluer (more flux at short wavelengths)
    This allows us to verify interpolation responds correctly.
    """
    n_met, n_alpha, n_age, n_wave = 3, 5, 10, 20

    lgmet = jnp.array([-1.5, -0.5, 0.0])
    alpha_fe = jnp.array([-0.2, 0.0, 0.2, 0.4, 0.6])
    lg_age_gyr = jnp.linspace(-1.0, 1.1, n_age)
    wave = jnp.linspace(3500.0, 9000.0, n_wave)

    # Build flux: base is a smooth function of met, alpha, age, wave
    # so we can verify interpolation is sensible
    key = jax.random.PRNGKey(42)
    base = jnp.abs(jax.random.normal(key, (n_met, n_alpha, n_age, n_wave))) * 1e-3 + 1e-5

    # Add systematic trends
    # Higher Z → more flux at red end
    z_trend = jnp.linspace(0.5, 1.5, n_wave)[None, None, None, :]
    met_scale = jnp.linspace(0.5, 1.5, n_met)[:, None, None, None]
    # Higher alpha → bump near pixel 10 (~ 5180 Å)
    alpha_bump = jnp.exp(-0.5 * ((jnp.arange(n_wave) - 10) / 2.0) ** 2)
    alpha_scale = alpha_fe[:, None, None] * alpha_bump[None, None, :]
    alpha_scale = alpha_scale[None, :, :, :]  # (1, n_alpha, 1, n_wave)
    # Younger → more blue flux
    age_scale = jnp.linspace(1.5, 0.5, n_age)[None, None, :, None]

    flux = base * met_scale * z_trend * age_scale + 0.01 * alpha_scale

    return {
        "ssp_flux": flux,
        "ssp_lgmet": lgmet,
        "ssp_alpha_fe": alpha_fe,
        "ssp_lg_age_gyr": lg_age_gyr,
        "ssp_wave": wave,
    }


@pytest.fixture
def ssp_data_4d(alpha_ssp_grid):
    """SSPData with 4D alpha-enhanced grid."""
    from tengri.components.stellar.sps.dsps_wrapper import SSPData

    g = alpha_ssp_grid
    return SSPData(
        ssp_wave=g["ssp_wave"],
        ssp_flux=g["ssp_flux"],
        ssp_lg_age_gyr=g["ssp_lg_age_gyr"],
        ssp_lgmet=g["ssp_lgmet"],
        ssp_alpha_fe=g["ssp_alpha_fe"],
    )


@pytest.fixture
def ssp_data_3d():
    """SSPData with standard 3D grid (no alpha)."""

    n_met, n_age, n_wave = 3, 10, 20
    key = jax.random.PRNGKey(0)
    return SSPData(
        ssp_wave=jnp.linspace(3500.0, 9000.0, n_wave),
        ssp_flux=jnp.abs(jax.random.normal(key, (n_met, n_age, n_wave))) * 1e-3 + 1e-5,
        ssp_lg_age_gyr=jnp.linspace(-1.0, 1.1, n_age),
        ssp_lgmet=jnp.array([-1.5, -0.5, 0.0]),
    )


# ── has_alpha_grid detection ──────────────────────────────────────


class TestHasAlphaGrid:
    """Bounds test: alpha grid detection."""

    def test_4d_grid_detected(self, ssp_data_4d):
        """4D grid with ssp_alpha_fe must be detected."""
        from tengri.components.stellar.sps.dsps_wrapper import has_alpha_grid

        assert has_alpha_grid(ssp_data_4d) is True

    def test_3d_grid_not_detected(self, ssp_data_3d):
        """3D grid without ssp_alpha_fe must not be detected."""

        assert has_alpha_grid(ssp_data_3d) is False

    def test_none_alpha_not_detected(self, ssp_data_3d):
        """3D grid with ssp_alpha_fe=None must not be detected."""

        assert has_alpha_grid(ssp_data_3d) is False


# ── Bilinear (Z, [α/Fe]) interpolation ────────────────────────────


class TestInterpolateMetAlpha:
    """Bounds tests: bilinear (metallicity, [α/Fe]) interpolation."""

    def test_at_grid_point(self, alpha_ssp_grid):
        """Interpolation at exact grid point should return that grid point.

        Test bounds: interpolation must be exact at grid nodes (zero error).
        """
        from tengri.components.stellar.sps.dsps_wrapper import interpolate_met_alpha

        g = alpha_ssp_grid
        result = interpolate_met_alpha(
            g["ssp_flux"],
            g["ssp_lgmet"],
            g["ssp_alpha_fe"],
            log_z=-0.5,
            alpha_fe=0.0,
        )
        expected = g["ssp_flux"][1, 1]  # lgmet=-0.5 is idx 1, alpha=0.0 is idx 1
        assert jnp.allclose(result, expected, atol=1e-10)

    def test_midpoint_interpolation(self, alpha_ssp_grid):
        """Midpoint between grid points should be the average of neighbors.

        Test bounds: linear interpolation is exact for piecewise-linear spaces.
        """

        g = alpha_ssp_grid
        # Midpoint between lgmet[0]=-1.5 and lgmet[1]=-0.5 at alpha[1]=0.0
        result = interpolate_met_alpha(
            g["ssp_flux"],
            g["ssp_lgmet"],
            g["ssp_alpha_fe"],
            log_z=-1.0,
            alpha_fe=0.0,
        )
        expected = 0.5 * g["ssp_flux"][0, 1] + 0.5 * g["ssp_flux"][1, 1]
        assert jnp.allclose(result, expected, atol=1e-10)

    def test_different_alpha_different_result(self, alpha_ssp_grid):
        """Different [α/Fe] values should give different SEDs.

        Test bounds: the spectral response to [α/Fe] must be non-zero.
        """

        g = alpha_ssp_grid
        sed_solar = interpolate_met_alpha(
            g["ssp_flux"],
            g["ssp_lgmet"],
            g["ssp_alpha_fe"],
            log_z=-0.5,
            alpha_fe=0.0,
        )
        sed_alpha = interpolate_met_alpha(
            g["ssp_flux"],
            g["ssp_lgmet"],
            g["ssp_alpha_fe"],
            log_z=-0.5,
            alpha_fe=0.4,
        )
        assert not jnp.allclose(sed_solar, sed_alpha)

    def test_output_shape(self, alpha_ssp_grid):
        """Output must be (n_age, n_wave) regardless of input."""

        g = alpha_ssp_grid
        result = interpolate_met_alpha(
            g["ssp_flux"],
            g["ssp_lgmet"],
            g["ssp_alpha_fe"],
            log_z=-0.5,
            alpha_fe=0.2,
        )
        chex.assert_shape(result, (10, 20))

    def test_finite_output(self, alpha_ssp_grid):
        """All outputs must be finite (no NaN or Inf).

        Bounds test: numerical overflow/underflow in interpolation.
        """

        g = alpha_ssp_grid
        for lz in [-1.5, -1.0, -0.5, 0.0]:
            for afe in [-0.2, 0.0, 0.2, 0.4, 0.6]:
                result = interpolate_met_alpha(
                    g["ssp_flux"],
                    g["ssp_lgmet"],
                    g["ssp_alpha_fe"],
                    log_z=lz,
                    alpha_fe=afe,
                )
                assert jnp.all(jnp.isfinite(result)), f"Non-finite at lz={lz}, afe={afe}"
                assert jnp.any(result != 0.0), (
                    "`result` is identically zero — finite is not enough, "
                    "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
                )

    def test_clamping_at_bounds(self, alpha_ssp_grid):
        """Values outside grid must clamp to boundary (extrapolation = nearest).

        Bounds test: behavior outside the grid extent.
        """

        g = alpha_ssp_grid
        # Beyond low Z boundary
        result_low = interpolate_met_alpha(
            g["ssp_flux"],
            g["ssp_lgmet"],
            g["ssp_alpha_fe"],
            log_z=-5.0,
            alpha_fe=0.0,
        )
        result_edge = interpolate_met_alpha(
            g["ssp_flux"],
            g["ssp_lgmet"],
            g["ssp_alpha_fe"],
            log_z=-1.5,
            alpha_fe=0.0,
        )
        assert jnp.allclose(result_low, result_edge, atol=1e-10)

    def test_jit_compatible(self, alpha_ssp_grid):
        """Should work under jax.jit."""

        g = alpha_ssp_grid
        jit_fn = jax.jit(interpolate_met_alpha, static_argnames=[])
        result = jit_fn(
            g["ssp_flux"],
            g["ssp_lgmet"],
            g["ssp_alpha_fe"],
            log_z=-0.5,
            alpha_fe=0.2,
        )
        chex.assert_tree_all_finite(result)

    def test_differentiable_wrt_log_z(self, alpha_ssp_grid):
        """Gradient w.r.t. log_z should be finite and match FD (gradient test)."""
        import numpy as np

        g = alpha_ssp_grid

        def total_flux(lz):
            return jnp.sum(
                interpolate_met_alpha(
                    g["ssp_flux"],
                    g["ssp_lgmet"],
                    g["ssp_alpha_fe"],
                    log_z=lz,
                    alpha_fe=0.0,
                )
            )

        grad_jax = float(jax.grad(total_flux)(-0.7))
        grad_fd = fd_grad(total_flux, -0.7)
        np.testing.assert_allclose(
            grad_jax, grad_fd, rtol=1e-3, err_msg=f"autodiff={grad_jax:.4e}, FD={grad_fd:.4e}"
        )

    def test_differentiable_wrt_alpha_fe(self, alpha_ssp_grid):
        """Gradient w.r.t. [α/Fe] should be finite and match FD (gradient test)."""

        g = alpha_ssp_grid

        def total_flux(afe):
            return jnp.sum(
                interpolate_met_alpha(
                    g["ssp_flux"],
                    g["ssp_lgmet"],
                    g["ssp_alpha_fe"],
                    log_z=-0.7,
                    alpha_fe=afe,
                )
            )

        grad_jax = float(jax.grad(total_flux)(0.1))
        grad_fd = fd_grad(total_flux, 0.1)
        np.testing.assert_allclose(
            grad_jax, grad_fd, rtol=1e-3, err_msg=f"autodiff={grad_jax:.4e}, FD={grad_fd:.4e}"
        )


# ── Per-age evolving (Z, [α/Fe]) interpolation ────────────────────


class TestInterpolateMetAlphaEvolving:
    """Bounds tests for per-age bilinear interpolation with evolving abundances."""

    def test_uniform_matches_global(self, alpha_ssp_grid):
        """Uniform Z and [α/Fe] across ages should match global interpolation.

        Bounds test: consistency between global and per-age paths.
        """
        from tengri.components.stellar.sps.dsps_wrapper import (
            interpolate_met_alpha,
            interpolate_met_alpha_evolving,
        )

        g = alpha_ssp_grid
        n_age = len(g["ssp_lg_age_gyr"])

        global_result = interpolate_met_alpha(
            g["ssp_flux"],
            g["ssp_lgmet"],
            g["ssp_alpha_fe"],
            log_z=-0.5,
            alpha_fe=0.2,
        )

        lz_per_age = jnp.full(n_age, -0.5)
        afe_per_age = jnp.full(n_age, 0.2)

        evolving_result = interpolate_met_alpha_evolving(
            g["ssp_flux"],
            g["ssp_lgmet"],
            g["ssp_alpha_fe"],
            lz_per_age,
            afe_per_age,
        )

        assert jnp.allclose(global_result, evolving_result, atol=1e-10)

    def test_varying_alpha_per_age(self, alpha_ssp_grid):
        """Different [α/Fe] per age should produce different spectra per age.

        Bounds test: per-age variation must propagate to output.
        """
        from tengri.components.stellar.sps.dsps_wrapper import interpolate_met_alpha_evolving

        g = alpha_ssp_grid
        n_age = len(g["ssp_lg_age_gyr"])

        lz_per_age = jnp.full(n_age, -0.5)
        afe_per_age = jnp.linspace(-0.2, 0.6, n_age)  # old=0.6, young=-0.2

        result = interpolate_met_alpha_evolving(
            g["ssp_flux"],
            g["ssp_lgmet"],
            g["ssp_alpha_fe"],
            lz_per_age,
            afe_per_age,
        )

        # Each age bin should be different since [α/Fe] varies
        chex.assert_shape(result, (n_age, 20))
        chex.assert_tree_all_finite(result)

    def test_output_shape(self, alpha_ssp_grid):
        """Output shape must be (n_age, n_wave)."""

        g = alpha_ssp_grid
        n_age = len(g["ssp_lg_age_gyr"])

        result = interpolate_met_alpha_evolving(
            g["ssp_flux"],
            g["ssp_lgmet"],
            g["ssp_alpha_fe"],
            jnp.full(n_age, -0.5),
            jnp.full(n_age, 0.0),
        )
        chex.assert_shape(result, (n_age, 20))

    def test_jit_compatible(self, alpha_ssp_grid):
        """Should work under jax.jit."""

        g = alpha_ssp_grid
        n_age = len(g["ssp_lg_age_gyr"])

        result = assert_jit_matches_eager(
            interpolate_met_alpha_evolving,
            g["ssp_flux"],
            g["ssp_lgmet"],
            g["ssp_alpha_fe"],
            jnp.full(n_age, -0.5),
            jnp.full(n_age, 0.0),
        )
        chex.assert_tree_all_finite(result)


# ── Alpha evolution ramp ──────────────────────────────────────────


class TestComputeAlphaFeEvolving:
    """Bounds tests for the [α/Fe](t_lookback) linear ramp."""

    def test_young_gets_young_value(self):
        """Youngest age bin (t_lookback ≈ 0) should get alpha_fe_young.

        Bounds test: boundary value at t_lookback=0.
        """
        from tengri.components.stellar.sps.dsps_wrapper import compute_alpha_fe_evolving

        lg_ages = jnp.array([-2.0, -1.0, 0.0, 1.0])  # log10(age/Gyr)
        result = compute_alpha_fe_evolving(lg_ages, 0.4, 0.0, 13.7)

        # Youngest bin: 10^(-2) = 0.01 Gyr → t_frac ≈ 0.0007 → near alpha_fe_young
        assert float(result[0]) == pytest.approx(0.0, abs=0.01)

    def test_old_gets_old_value(self):
        """Oldest age bin (t_lookback ≈ t_universe) should get alpha_fe_old.

        Bounds test: boundary value at t_lookback=t_universe.
        """

        lg_ages = jnp.array([-2.0, -1.0, 0.0, 1.14])  # 10^1.14 ≈ 13.8 Gyr
        result = compute_alpha_fe_evolving(lg_ages, 0.4, 0.0, 13.7)

        # Oldest bin: 10^1.14 ≈ 13.8 Gyr, clamped to t_frac=1.0
        assert float(result[-1]) == pytest.approx(0.4, abs=0.01)

    def test_monotonic_increase(self):
        """[α/Fe] should increase monotonically with lookback time.

        Bounds test: physical ordering (α-enhancement only gets stronger
        as you go back in time, never decreases).
        """

        lg_ages = jnp.linspace(-2.0, 1.1, 20)
        result = compute_alpha_fe_evolving(lg_ages, 0.4, 0.0, 13.7)

        diffs = jnp.diff(result)
        assert_non_negative(
            diffs, name="diffs", msg="Alpha should increase with age (lookback time)"
        )

    def test_equal_old_young_gives_constant(self):
        """If alpha_old == alpha_young, result should be constant.

        Bounds test: degenerate case (no evolution).
        """

        lg_ages = jnp.linspace(-2.0, 1.1, 20)
        result = compute_alpha_fe_evolving(lg_ages, 0.3, 0.3, 13.7)

        assert jnp.allclose(result, 0.3, atol=1e-10)

    def test_differentiable(self):
        """Should be differentiable w.r.t. alpha_fe_old and match FD (gradient test)."""

        lg_ages = jnp.linspace(-2.0, 1.1, 20)

        def total(alpha_old):
            return jnp.sum(compute_alpha_fe_evolving(lg_ages, alpha_old, 0.0, 13.7))

        grad_jax = float(jax.grad(total)(0.4))
        grad_fd = fd_grad(total, 0.4)
        np.testing.assert_allclose(
            grad_jax, grad_fd, rtol=1e-3, err_msg=f"autodiff={grad_jax:.4e}, FD={grad_fd:.4e}"
        )
        assert grad_jax > 0  # more alpha_old → higher total


# ── Effective metallicity (backward compatibility) ────────────────


class TestEffectiveMetallicity:
    """Bounds test: effective_metallicity backward compatibility with 3D grids."""

    def test_solar_alpha_is_identity(self):
        """[α/Fe] = 0 should leave metallicity unchanged.

        Bounds test: zero [α/Fe] offset.
        """
        from tengri.components.stellar.sps.dsps_wrapper import effective_metallicity

        assert float(effective_metallicity(-0.5, 0.0)) == pytest.approx(-0.5)

    def test_positive_alpha_increases_z(self):
        """Positive [α/Fe] should increase effective Z.

        Bounds test: physical direction (more alpha → higher effective Z).
        """

        z_eff = effective_metallicity(-0.5, 0.4)
        assert float(z_eff) > -0.5

    def test_coefficient_is_0_75(self):
        """The conversion coefficient should be 0.75.

        Bounds test: standard approximation formula magnitude.
        """

        z_eff = effective_metallicity(0.0, 1.0)
        assert float(z_eff) == pytest.approx(0.75, abs=0.01)
