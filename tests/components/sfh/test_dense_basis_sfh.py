# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for the dense_basis GP-SFH model.

Tests the Matérn 3/2 + Linear kernel, GP interpolation, and the
dense_basis() function for correctness, JIT-compatibility,
differentiability, and registry integration.

References
----------
- Iyer & Gawiser (2017), ApJ 838, 127.
- Iyer et al. (2019), ApJ 879, 116.
"""

import chex
import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

pytestmark = pytest.mark.bounds


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite-difference gradient. O(eps^2) accurate."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


from tengri.components.stellar.sfh.dense_basis import (
    _build_quantile_points,
    _george_combined_kernel,
    _george_linear_kernel,
    dense_basis,
    gp_interpolate,
    linear_kernel,
    matern32_kernel,
)
from tengri.components.stellar.sfh.registry import resolve_sfh
from tests._jit_parity import assert_jit_matches_eager

# Shared test fixtures
AGE_YR = jnp.geomspace(1e6, 13.7e9, 200)
DEFAULT_KW = {
    "log_total_mass": 10.0,
    "log_sfr_inst": 0.0,
    "tx_frac_0": 0.3,
    "tx_frac_1": 0.55,
    "tx_frac_2": 0.8,
}


# ── Kernel tests ──────────────────────────────────────────────────


class TestMatern32Kernel:
    """Tests for the Matérn 3/2 covariance kernel."""

    def test_symmetric(self) -> None:
        """K(x1, x2) == K(x2, x1).T."""
        x1 = jnp.array([0.1, 0.3, 0.7])
        x2 = jnp.array([0.2, 0.5, 0.8, 0.9])
        k12 = matern32_kernel(x1, x2, variance=1.0, length_scale=0.3)
        k21 = matern32_kernel(x2, x1, variance=1.0, length_scale=0.3)
        assert jnp.allclose(k12, k21.T, atol=1e-12)

    def test_positive_definite(self) -> None:
        """Kernel matrix has all positive eigenvalues."""
        x = jnp.linspace(0.0, 1.0, 10)
        k = matern32_kernel(x, x, variance=1.0, length_scale=0.3)
        eigvals = jnp.linalg.eigvalsh(k)
        assert jnp.all(eigvals > -1e-10)

    def test_diagonal_equals_variance(self) -> None:
        """K(x, x) diagonal should equal the variance."""
        x = jnp.array([0.1, 0.5, 0.9])
        k = matern32_kernel(x, x, variance=2.5, length_scale=0.3)
        assert jnp.allclose(jnp.diag(k), 2.5, atol=1e-12)

    def test_decays_with_distance(self) -> None:
        """Covariance decreases with increasing distance."""
        x = jnp.array([0.0])
        y = jnp.array([0.1, 0.5, 1.0, 2.0])
        k = matern32_kernel(x, y, variance=1.0, length_scale=0.5)
        assert jnp.all(jnp.diff(k[0]) < 0)


class TestLinearKernel:
    """Tests for the linear covariance kernel."""

    def test_symmetric(self) -> None:
        x1 = jnp.array([0.1, 0.3, 0.7])
        x2 = jnp.array([0.2, 0.5])
        k12 = linear_kernel(x1, x2, variance=1.0, length_scale=0.3)
        k21 = linear_kernel(x2, x1, variance=1.0, length_scale=0.3)
        assert jnp.allclose(k12, k21.T, atol=1e-12)


class TestCombinedKernel:
    """Tests for the george-faithful Matérn 3/2 + Linear kernel.

    ``dense_basis`` builds ``var(y) * (Matern32(median(y)) + Linear(median(y),
    order=2))``; the hyperparameters come from the training values, so these
    tests pass ``y_train`` rather than a variance/length-scale pair.
    """

    def test_positive_definite(self) -> None:
        x = jnp.linspace(0.1, 1.0, 8)
        y = jnp.linspace(0.0, 1.0, 8)
        k = _george_combined_kernel(x, x, y)
        eigvals = jnp.linalg.eigvalsh(k)
        assert jnp.all(eigvals > -1e-10)

    def test_is_sum_of_components(self) -> None:
        x1 = jnp.array([0.1, 0.5, 0.9])
        x2 = jnp.array([0.2, 0.6])
        y = jnp.array([0.0, 0.4, 1.0])
        variance = jnp.var(y)
        med = jnp.median(y)
        k_comb = _george_combined_kernel(x1, x2, y)
        # george's Matern32 metric is a squared length scale, so ell = sqrt(med).
        k_m = matern32_kernel(x1, x2, variance, jnp.sqrt(med))
        k_l = variance * _george_linear_kernel(x1, x2, log_gamma2=med, order=2)
        assert jnp.allclose(k_comb, k_m + k_l, atol=1e-12)


# ── GP interpolation tests ────────────────────────────────────────


class TestGPInterpolation:
    """Tests for GP conditional mean prediction."""

    def test_passes_through_training_points(self) -> None:
        """GP mean at training points should match training values (small noise)."""
        x_train = jnp.array([0.0, 0.3, 0.6, 1.0])
        y_train = jnp.array([0.0, 0.3, 0.8, 1.0])
        y_err = jnp.full(4, 0.001)
        y_pred = gp_interpolate(x_train, y_train, y_err, x_train)
        assert jnp.allclose(y_pred, y_train, atol=0.01)

    def test_interpolation_monotonic_for_monotonic_data(self) -> None:
        """GP interpolation of monotonic data should be approximately monotonic."""
        x_train = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])
        y_train = jnp.array([0.0, 0.2, 0.5, 0.8, 1.0])
        y_err = jnp.full(5, 0.001)
        x_eval = jnp.linspace(0.01, 0.99, 100)
        y_pred = gp_interpolate(x_train, y_train, y_err, x_eval)
        # Allow small non-monotonicity from GP smoothing
        n_violations = jnp.sum(jnp.diff(y_pred) < -0.01)
        assert n_violations <= 2, f"Too many monotonicity violations: {n_violations}"


# ── Quantile point construction ───────────────────────────────────


class TestBuildQuantilePoints:
    """Tests for quantile point assembly."""

    def test_shapes(self) -> None:
        tx = jnp.array([0.3, 0.5, 0.8])
        time_q, mass_q, yerr = _build_quantile_points(
            tx,
            n_param=3,
            log_total_mass=10.0,
            log_sfr_inst=0.0,
            age_universe_yr=13.47e9,
        )
        # n_param+2 (endpoints) + 1 (BB) + 3 (SFR constraints) = 9
        chex.assert_shape(time_q, (9,))
        chex.assert_shape(mass_q, (9,))
        chex.assert_shape(yerr, (9,))

    def test_boundary_values(self) -> None:
        tx = jnp.array([0.3, 0.5, 0.8])
        time_q, mass_q, _yerr = _build_quantile_points(
            tx,
            n_param=3,
            log_total_mass=10.0,
            log_sfr_inst=0.0,
            age_universe_yr=13.47e9,
        )
        # First point is t=0, M=0
        assert jnp.isclose(time_q[0], 0.0)
        assert jnp.isclose(mass_q[0], 0.0)
        # Last point is observation epoch (t=1, M=1)
        assert jnp.isclose(time_q[-1], 1.0)
        assert jnp.isclose(mass_q[-1], 1.0)


class TestGPCumulativeMassAccuracy:
    """Verify GP passes through quantile constraint points."""

    def test_cumulative_mass_at_quantile_times(self) -> None:
        """Cumulative mass at tx should match target mass fractions."""
        from tengri.components.stellar.sfh.dense_basis import (
            _build_quantile_points,
            gp_interpolate,
        )

        tx = jnp.array([0.3, 0.55, 0.8])
        time_q, mass_q, yerr = _build_quantile_points(
            tx,
            n_param=3,
            log_total_mass=10.0,
            log_sfr_inst=0.0,
            age_universe_yr=13.47e9,
        )
        m_pred = gp_interpolate(time_q, mass_q, yerr, tx)
        # Should match 25%, 50%, 75% mass fractions
        expected = jnp.array([0.25, 0.5, 0.75])
        assert jnp.allclose(m_pred, expected, atol=0.05), (
            f"GP at quantile times: {m_pred} vs expected {expected}"
        )


# ── Dense basis SFH function tests ────────────────────────────────


class TestDenseBasisSFH:
    """Tests for the main dense_basis function."""

    def test_non_negative(self) -> None:
        sfr = dense_basis(AGE_YR, **DEFAULT_KW)
        assert jnp.all(sfr >= 0)

    def test_mass_conservation(self) -> None:
        """Integrated SFR should match 10^log_total_mass within 15%."""
        sfr = dense_basis(AGE_YR, **DEFAULT_KW)
        mass = jnp.trapezoid(sfr, AGE_YR)
        target = 10.0 ** DEFAULT_KW["log_total_mass"]
        ratio = mass / target
        assert 0.85 < ratio < 1.15, f"Mass ratio {ratio:.3f} outside [0.85, 1.15]"

    def test_mass_scales_with_log_total_mass(self) -> None:
        """Increasing log_total_mass by 1 dex → ~10x more mass."""
        tx_kw = {
            "tx_frac_0": 0.3,
            "tx_frac_1": 0.55,
            "tx_frac_2": 0.8,
        }
        # Scale both mass and SFR together (they're coupled via
        # the SFR constraint points)
        sfr_10 = dense_basis(
            AGE_YR,
            log_total_mass=10.0,
            log_sfr_inst=0.0,
            **tx_kw,
        )
        sfr_11 = dense_basis(
            AGE_YR,
            log_total_mass=11.0,
            log_sfr_inst=1.0,
            **tx_kw,
        )
        m10 = jnp.trapezoid(sfr_10, AGE_YR)
        m11 = jnp.trapezoid(sfr_11, AGE_YR)
        ratio = m11 / m10
        assert 5.0 < ratio < 20.0, f"Mass ratio {ratio:.1f} not ~10"

    def test_jit_parity_vs_eager(self) -> None:
        """JIT output matches eager evaluation (JAX correctness)."""
        sfr_eager = dense_basis(AGE_YR, **DEFAULT_KW)
        sfr_jit = jax.jit(dense_basis)(AGE_YR, **DEFAULT_KW)
        chex.assert_trees_all_close(sfr_eager, sfr_jit, rtol=1e-6)

    def test_has_gradient_log_total_mass(self) -> None:
        def _sfr_sum(m: float) -> float:
            return jnp.sum(
                dense_basis(
                    AGE_YR,
                    log_total_mass=m,
                    log_sfr_inst=0.0,
                    tx_frac_0=0.3,
                    tx_frac_1=0.55,
                    tx_frac_2=0.8,
                )
            )

        g_val = float(jax.grad(_sfr_sum)(10.0))

        def f_scalar(m):
            return float(_sfr_sum(m))

        assert_allclose(
            g_val,
            fd_grad(f_scalar, 10.0, eps=0.01),
            rtol=1e-3,
            err_msg="dense_basis: FD check ∂(∑SFR)/∂log_total_mass",
        )

    def test_has_gradient_tx_frac(self) -> None:
        def _sfr_sum(t0: float) -> float:
            return jnp.sum(
                dense_basis(
                    AGE_YR,
                    log_total_mass=10.0,
                    log_sfr_inst=0.0,
                    tx_frac_0=t0,
                    tx_frac_1=0.55,
                    tx_frac_2=0.8,
                )
            )

        g_val = float(jax.grad(_sfr_sum)(0.3))

        def f_scalar(t0):
            return float(_sfr_sum(t0))

        assert_allclose(
            g_val,
            fd_grad(f_scalar, 0.3, eps=0.01),
            rtol=5e-3,  # dense_basis uses GP-style spline interp; 0.5% agreement is sufficient
            err_msg="dense_basis: FD check ∂(∑SFR)/∂tx_frac_0",
        )

    def test_dense_basis_tx_ordering(self) -> None:
        """Mass assembly times must be monotone: t(20%) < t(50%) < t(80%).

        Iyer et al. 2019, ApJ 879, 116: tx_frac_i are cumulative-mass quantiles,
        so the times at which those mass fractions are reached must be ordered.
        """
        AGE_GRID = jnp.geomspace(1e6, 13.7e9, 10000)
        sfr = dense_basis(
            AGE_GRID,
            log_total_mass=10.0,
            log_sfr_inst=0.0,
            age_universe_yr=13.7e9,
            tx_frac_0=0.2,
            tx_frac_1=0.5,
            tx_frac_2=0.8,
        )
        dt = jnp.diff(AGE_GRID, prepend=AGE_GRID[0])
        cum_mass = jnp.cumsum(sfr * dt)
        total = float(cum_mass[-1])
        t20 = float(AGE_GRID[int(jnp.searchsorted(cum_mass / total, 0.2))])
        t50 = float(AGE_GRID[int(jnp.searchsorted(cum_mass / total, 0.5))])
        t80 = float(AGE_GRID[int(jnp.searchsorted(cum_mass / total, 0.8))])
        assert t20 < t50 < t80, (
            f"Dense basis: mass assembly times not monotone: "
            f"t20={t20 / 1e9:.2f} Gyr, t50={t50 / 1e9:.2f} Gyr, t80={t80 / 1e9:.2f} Gyr"
        )

    def test_dense_basis_mass_conservation(self) -> None:
        """Integral of SFR over time must equal 10^log_total_mass within 1%.

        Iyer et al. 2019, ApJ 879, 116: dense_basis is normalized so that
        ∫ SFR(t) dt = 10^log_total_mass Msun.
        """
        AGE_GRID = jnp.geomspace(1e6, 13.7e9, 10000)
        sfr = dense_basis(
            AGE_GRID,
            log_total_mass=10.0,
            log_sfr_inst=0.0,
            age_universe_yr=13.7e9,
            tx_frac_0=0.2,
            tx_frac_1=0.5,
            tx_frac_2=0.8,
        )
        mass = float(jnp.trapezoid(sfr, AGE_GRID))
        assert_allclose(
            mass,
            1e10,
            rtol=0.01,
            err_msg="dense_basis: ∫SFR dt != 10^10 Msun (mass conservation violated)",
        )

    def test_ordering_enforced(self) -> None:
        """Passing unordered tx fractions should still produce valid output."""
        sfr = dense_basis(
            AGE_YR,
            log_total_mass=10.0,
            log_sfr_inst=0.0,
            tx_frac_0=0.8,
            tx_frac_1=0.3,
            tx_frac_2=0.55,
        )
        assert jnp.all(sfr >= 0)
        chex.assert_equal_shape([sfr, AGE_YR])


# ── SFH shape tests (tutorial shapes from Iyer+2019) ──────────────


def _sfh_for_tx(t0: float, t1: float, t2: float) -> jnp.ndarray:
    """Helper: compute SFR on AGE_YR for given tx fractions."""
    return dense_basis(
        AGE_YR,
        log_total_mass=10.0,
        log_sfr_inst=0.0,
        tx_frac_0=t0,
        tx_frac_1=t1,
        tx_frac_2=t2,
    )


class TestDenseBasisShapes:
    """Test that different tx_frac configurations produce qualitatively correct SFH shapes."""

    def test_rising_sfh_peaks_recently(self) -> None:
        """Late tx values → rising SFH peaking at recent times."""
        sfr = _sfh_for_tx(0.5, 0.7, 0.85)
        peak_age = AGE_YR[jnp.argmax(sfr)]
        assert peak_age < 7e9, f"Rising SFH peaks too early: {peak_age / 1e9:.1f} Gyr"

    def test_quenched_sfh_peaks_early(self) -> None:
        """Early tx values → old quenched galaxy with early peak."""
        sfr = _sfh_for_tx(0.15, 0.3, 0.5)
        peak_age = AGE_YR[jnp.argmax(sfr)]
        assert peak_age > 5e9, f"Quenched SFH peaks too late: {peak_age / 1e9:.1f} Gyr"

    def test_non_negative_all_shapes(self) -> None:
        """All 6 tutorial shapes produce non-negative SFR."""
        shapes = [
            (0.5, 0.7, 0.85),  # Rising
            (0.3, 0.55, 0.8),  # Regular SF
            (0.5, 0.8, 0.9),  # Post-starburst
            (0.15, 0.3, 0.5),  # Quenched
            (0.25, 0.30, 0.7),  # Double-peaked SF
            (0.1, 0.6, 0.7),  # Double-peaked Q
        ]
        for t0, t1, t2 in shapes:
            sfr = _sfh_for_tx(t0, t1, t2)
            assert jnp.all(sfr >= 0), f"Negative SFR for tx=({t0}, {t1}, {t2})"


# ── Registry integration tests ────────────────────────────────────


class TestDenseBasisRegistry:
    """Tests for SFH registry integration."""

    def test_resolve_dense_basis(self) -> None:
        fn, params, _pmap, _settings = resolve_sfh("dense_basis")
        assert fn is not None
        assert "sfh_db_log_total_mass" in params
        assert "sfh_db_tx_frac_0" in params
        assert "sfh_db_tx_frac_1" in params
        assert "sfh_db_tx_frac_2" in params

    def test_alias_db(self) -> None:
        fn1, _, _, _ = resolve_sfh("dense_basis")
        fn2, _, _, _ = resolve_sfh("db")
        # Both should resolve successfully (may not be same object due to closure)
        assert fn1 is not None
        assert fn2 is not None

    def test_internal_param_map(self) -> None:
        _, _, pmap, _ = resolve_sfh("dense_basis")
        assert pmap["sfh_db_log_total_mass"] == ("log_total_mass", 1.0, 0.0)
        assert pmap["sfh_db_tx_frac_0"] == ("tx_frac_0", 1.0, 0.0)

    def test_composition_with_burst(self) -> None:
        """Auto-swaps to dense_basis_pure when burst is present."""
        _fn, params, _, _ = resolve_sfh(["dense_basis", "burst"])
        assert "sfh_dbp_log_total_mass" in params
        assert "sfh_db_log_sfr_inst" not in params
        assert "sfh_burst_log_fburst" in params

    def test_composition_with_field(self) -> None:
        """Auto-swaps to dense_basis_pure when field is present."""
        _fn, params, _, settings = resolve_sfh(["dense_basis", "field"])
        # Auto-swapped to pure variant (sfh_dbp_ prefix, no log_sfr_inst)
        assert "sfh_dbp_log_total_mass" in params
        assert "sfh_db_log_sfr_inst" not in params
        assert "sfh_field_psd_sigma" in params
        assert "sfh_field_ngrid" in settings

    def test_settings_contain_nparam(self) -> None:
        _, _, _, settings = resolve_sfh("dense_basis")
        assert settings["sfh_db_nparam"] == 3
        assert settings["sfh_db_age_universe_gyr"] == 13.47


# ── Edge case tests ───────────────────────────────────────────────


class TestDenseBasisEdgeCases:
    """Edge cases and boundary conditions."""

    def test_missing_tx_param_raises(self) -> None:
        """Missing tx_frac key raises ValueError."""
        import pytest

        with pytest.raises(ValueError, match="Missing required"):
            dense_basis(
                AGE_YR,
                log_total_mass=10.0,
                tx_frac_0=0.3,
                tx_frac_2=0.8,  # missing tx_frac_1
            )

    def test_no_tx_params_raises(self) -> None:
        """No tx_frac kwargs raises ValueError."""
        import pytest

        with pytest.raises(ValueError, match="at least one"):
            dense_basis(AGE_YR, log_total_mass=10.0)

    def test_extreme_tx_near_zero(self) -> None:
        """Very early mass assembly (all tx near 0)."""
        sfr = _sfh_for_tx(0.05, 0.1, 0.15)
        chex.assert_tree_all_finite(sfr)
        assert jnp.all(sfr >= 0)

    def test_extreme_tx_near_one(self) -> None:
        """Very late mass assembly (all tx near 1)."""
        sfr = _sfh_for_tx(0.85, 0.9, 0.95)
        chex.assert_tree_all_finite(sfr)
        assert jnp.all(sfr >= 0)

    def test_low_mass_galaxy(self) -> None:
        """log_total_mass = 8 (dwarf galaxy)."""
        sfr = dense_basis(
            AGE_YR,
            log_total_mass=8.0,
            log_sfr_inst=-2.0,
            tx_frac_0=0.3,
            tx_frac_1=0.55,
            tx_frac_2=0.8,
        )
        mass = jnp.trapezoid(sfr, AGE_YR)
        assert 0.5e8 < mass < 1.5e8

    def test_high_mass_galaxy(self) -> None:
        """log_total_mass = 12 (massive galaxy)."""
        sfr = dense_basis(
            AGE_YR,
            log_total_mass=12.0,
            log_sfr_inst=2.0,
            tx_frac_0=0.15,
            tx_frac_1=0.3,
            tx_frac_2=0.5,
        )
        mass = jnp.trapezoid(sfr, AGE_YR)
        assert 0.7e12 < mass < 1.3e12

    def test_custom_age_universe(self) -> None:
        """Higher redshift: age_universe < 13.8 Gyr."""
        age_z1 = jnp.geomspace(1e6, 5.9e9, 200)  # z~1: age ~5.9 Gyr
        sfr = dense_basis(
            age_z1,
            log_total_mass=10.0,
            age_universe_yr=5.9e9,
            tx_frac_0=0.3,
            tx_frac_1=0.55,
            tx_frac_2=0.8,
        )
        chex.assert_tree_all_finite(sfr)
        assert jnp.all(sfr >= 0)


class TestJITNaNRegression:
    """Regression tests for JIT-only NaN bug in gp_interpolate.

    When log_sfr_inst is large, all three SFR constraint time points clip to
    the same upper bound (0.999), making the GP kernel matrix singular.
    jnp.linalg.solve returns NaN silently under JIT but raises a RuntimeError
    in eager mode — causing a JIT-only phantom NaN in the forward model.

    The fix: distinct clip bounds per SFR constraint point (0.997, 0.998, 0.999)
    and a 1e-8 nugget on the kernel diagonal.
    """

    def test_high_sfr_no_nan_jit(self) -> None:
        """log_sfr_inst=2.93 (near prior upper bound) must not produce NaN under JIT."""
        sfr = assert_jit_matches_eager(
            lambda log_sfr: dense_basis(
                AGE_YR,
                log_total_mass=10.0,
                log_sfr_inst=log_sfr,
                tx_frac_0=0.408,
                tx_frac_1=0.610,
            ),
            2.93,
        )
        chex.assert_tree_all_finite(sfr)

    def test_very_high_sfr_no_nan_jit(self) -> None:
        """Extreme SFR (log=3.0) must not produce NaN under JIT."""
        sfr = assert_jit_matches_eager(
            lambda log_sfr: dense_basis(
                AGE_YR,
                log_total_mass=8.0,
                log_sfr_inst=log_sfr,
                tx_frac_0=0.3,
                tx_frac_1=0.6,
            ),
            3.0,
        )
        chex.assert_tree_all_finite(sfr)

    def test_low_sfr_no_nan_jit(self) -> None:
        """log_sfr_inst=-2.0 (prior lower bound) must not produce NaN under JIT."""
        sfr = assert_jit_matches_eager(
            lambda log_sfr: dense_basis(
                AGE_YR,
                log_total_mass=12.0,
                log_sfr_inst=log_sfr,
                tx_frac_0=0.3,
                tx_frac_1=0.6,
                tx_frac_2=0.85,
            ),
            -2.0,
        )
        chex.assert_tree_all_finite(sfr)

    def test_random_prior_samples_no_nan_jit(self) -> None:
        """50 random prior samples must produce zero NaN under JIT."""
        key = jax.random.PRNGKey(42)
        jit_sfh = jax.jit(
            lambda lm, ls, t0, t1, t2: dense_basis(
                AGE_YR,
                log_total_mass=lm,
                log_sfr_inst=ls,
                tx_frac_0=t0,
                tx_frac_1=t1,
                tx_frac_2=t2,
            )
        )
        n_nan = 0
        for _ in range(50):
            key, k1, k2, k3, k4, k5 = jax.random.split(key, 6)
            sfr = jit_sfh(
                float(jax.random.uniform(k1, minval=8.0, maxval=12.0)),
                float(jax.random.uniform(k2, minval=-2.0, maxval=3.0)),
                float(jax.random.uniform(k3, minval=0.05, maxval=0.95)),
                float(jax.random.uniform(k4, minval=0.05, maxval=0.95)),
                float(jax.random.uniform(k5, minval=0.05, maxval=0.95)),
            )
            if jnp.any(jnp.isnan(sfr)):
                n_nan += 1
        assert n_nan == 0, f"{n_nan}/50 prior samples produced NaN under JIT"
