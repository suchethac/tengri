"""Tests for SFH positivity and mass conservation bounds.

Tests verify that all SFH models produce non-negative star formation rates
and correctly conserve the total stellar mass.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tengri.components.stellar.sfh.nonparametric import (
    continuity,
    dirichlet,
    make_agebins_from_zred,
    psb_continuity,
)

pytestmark = pytest.mark.bounds

AGE_YR = jnp.linspace(1e7, 13.5e9, 200)


class TestContinuitySFHMassConservation:
    """Test mass conservation in continuity SFH."""

    def test_total_mass_conserved(self):
        """Integrated SFR * dt should equal the specified total mass."""
        log_mass = 10.0
        kwargs = {f"ratio_{i}": 0.0 for i in range(6)}
        sfr = continuity(AGE_YR, log_total_mass=log_mass, **kwargs)

        # Numerically integrate SFR over the age grid
        integrated_mass = jnp.trapezoid(sfr, AGE_YR)

        # Should match 10^log_mass to ~10% (interpolation introduces small errors)
        expected_mass = 10.0**log_mass
        relative_error = jnp.abs(integrated_mass - expected_mass) / expected_mass
        assert relative_error < 0.15, f"Mass error: {relative_error:.3f}"

    def test_custom_bin_edges(self):
        """Custom bin edges should preserve non-negativity."""
        custom_edges = jnp.array([0.0, 0.1, 1.0, 5.0, 13.7])
        n_bins = 4
        kwargs = {f"ratio_{i}": 0.0 for i in range(n_bins - 1)}
        sfr = continuity(AGE_YR, log_total_mass=10.0, bin_edges_gyr=custom_edges, **kwargs)
        assert sfr.shape == AGE_YR.shape
        assert jnp.all(sfr >= 0)


class TestDirichletSFHMassConservation:
    """Test mass conservation in Dirichlet SFH."""

    def test_mass_fractions_sum_to_one(self):
        """Stick-breaking mass fractions must sum to 1."""
        from tengri.components.stellar.sfh.nonparametric import _stick_breaking

        z = jnp.array([0.3, 0.5, 0.7, 0.2, 0.8, 0.4])
        fracs = _stick_breaking(z)
        assert jnp.allclose(jnp.sum(fracs), 1.0, atol=1e-10)

    def test_total_mass_conserved(self):
        """Integrated SFR * dt should match specified total mass."""
        log_mass = 10.0
        equal_z = [1 / 7, 1 / 6, 1 / 5, 1 / 4, 1 / 3, 1 / 2]
        kwargs = {f"z_frac_{i}": equal_z[i] for i in range(6)}
        sfr = dirichlet(AGE_YR, log_total_mass=log_mass, **kwargs)

        integrated_mass = jnp.trapezoid(sfr, AGE_YR)
        expected_mass = 10.0**log_mass
        relative_error = jnp.abs(integrated_mass - expected_mass) / expected_mass
        assert relative_error < 0.25, f"Mass error: {relative_error:.3f}"


# ── Regression: step-function behavior (searchsorted fix, 2026-04)


class TestStepFunctionRegression:
    """Regression: continuity/dirichlet SFH must be piecewise-constant (Leja+2019).

    Previously used linear interpolation on bin centers, giving smoothly varying
    SFR instead of the intended step function. Fixed by using searchsorted on bin
    edges.
    """

    def test_continuity_constant_within_bins(self):
        """SFR must be constant within each age bin (step function)."""
        bin_edges = jnp.array([0.0, 0.1, 0.5, 1.0, 3.0, 6.0, 10.0, 13.7])
        bin_edges_yr = bin_edges * 1e9

        # Two ages well inside the same bin (e.g., bin 3: 1.0-3.0 Gyr)
        age_mid1 = jnp.array([1.5e9])
        age_mid2 = jnp.array([2.5e9])

        kwargs = {f"ratio_{i}": 0.3 * (i - 3) for i in range(7)}
        sfr1 = continuity(age_mid1, log_total_mass=10.0, bin_edges_gyr=bin_edges, **kwargs)
        sfr2 = continuity(age_mid2, log_total_mass=10.0, bin_edges_gyr=bin_edges, **kwargs)

        assert float(jnp.abs(sfr1[0] - sfr2[0])) < 1e-10, (
            f"SFR should be constant within a bin: {float(sfr1[0]):.6e} vs {float(sfr2[0]):.6e}"
        )

    def test_dirichlet_constant_within_bins(self):
        """Dirichlet SFR must also be step-function within bins."""
        bin_edges = jnp.array([0.0, 0.1, 0.5, 1.0, 3.0, 6.0, 10.0, 13.7])

        age_mid1 = jnp.array([1.5e9])
        age_mid2 = jnp.array([2.5e9])

        kwargs = {f"z_frac_{i}": 0.3 + 0.1 * i for i in range(7)}
        sfr1 = dirichlet(age_mid1, log_total_mass=10.0, bin_edges_gyr=bin_edges, **kwargs)
        sfr2 = dirichlet(age_mid2, log_total_mass=10.0, bin_edges_gyr=bin_edges, **kwargs)

        assert float(jnp.abs(sfr1[0] - sfr2[0])) < 1e-10, (
            "Dirichlet SFR should be constant within a bin"
        )

    def test_sfr_changes_across_bin_boundary(self):
        """SFR must change across bin boundaries (not interpolated)."""
        bin_edges = jnp.array([0.0, 0.1, 0.5, 1.0, 3.0, 6.0, 10.0, 13.7])

        # Ages straddling the 1.0 Gyr boundary between bin 2 (0.5-1.0) and bin 3 (1.0-3.0)
        age_before = jnp.array([0.99e9])
        age_after = jnp.array([1.01e9])

        # ratio_i = log(SFR_{i+1} / SFR_i). Setting ratio_2 = 1.0 makes
        # SFR in bin 3 = 10^1.0 × SFR in bin 2 — a clear step.
        kwargs = {f"ratio_{i}": 1.0 if i == 2 else 0.0 for i in range(7)}
        sfr_before = continuity(age_before, log_total_mass=10.0, bin_edges_gyr=bin_edges, **kwargs)
        sfr_after = continuity(age_after, log_total_mass=10.0, bin_edges_gyr=bin_edges, **kwargs)

        # With ratio_2 = 1.0, SFR should differ by ~10x across the boundary
        assert float(jnp.abs(sfr_before[0] - sfr_after[0])) > 1e-3, (
            f"SFR should change discontinuously at bin boundaries: "
            f"before={float(sfr_before[0]):.4e}, after={float(sfr_after[0]):.4e}"
        )


class TestDirichletSFH:
    """Tests for Dirichlet SFH mass conservation."""

    def test_mass_fractions_sum_to_one(self):
        """Stick-breaking mass fractions must sum to 1."""
        from tengri.components.stellar.sfh.nonparametric import _stick_breaking

        z = jnp.array([0.3, 0.5, 0.7, 0.2, 0.8, 0.4])
        fracs = _stick_breaking(z)
        assert jnp.allclose(jnp.sum(fracs), 1.0, atol=1e-10)

    def test_total_mass_conserved(self):
        """Integrated SFR * dt should match specified total mass."""
        log_mass = 10.0
        equal_z = [1 / 7, 1 / 6, 1 / 5, 1 / 4, 1 / 3, 1 / 2]
        kwargs = {f"z_frac_{i}": equal_z[i] for i in range(6)}
        sfr = dirichlet(AGE_YR, log_total_mass=log_mass, **kwargs)

        integrated_mass = jnp.trapezoid(sfr, AGE_YR)
        expected_mass = 10.0**log_mass
        relative_error = jnp.abs(integrated_mass - expected_mass) / expected_mass
        assert relative_error < 0.25, f"Mass error: {relative_error:.3f}"


# ── make_agebins_from_zred ────────────────────────────────────────


class TestMakeAgebinsFromZred:
    """Tests for Prospector-β redshift-aware age bin construction."""

    def test_edges_monotone(self):
        edges = make_agebins_from_zred(1.0)
        assert np.all(np.diff(edges) >= 0.0), "bin edges must be monotonically non-decreasing"

    def test_starts_at_zero(self):
        edges = make_agebins_from_zred(2.0)
        assert edges[0] == 0.0

    def test_capped_at_tuniv_z2(self):
        edges = make_agebins_from_zred(2.0)
        # Age of universe at z=2 is ~3.3 Gyr; edges must not exceed it
        assert edges[-1] <= 3.5, f"edges exceed tuniv at z=2: {edges[-1]:.2f} Gyr"

    def test_capped_at_tuniv_z4(self):
        edges = make_agebins_from_zred(4.0)
        assert edges[-1] <= 1.8, f"edges exceed tuniv at z=4: {edges[-1]:.2f} Gyr"

    def test_capped_at_tuniv_z6(self):
        edges = make_agebins_from_zred(6.0)
        assert edges[-1] <= 1.0, f"edges exceed tuniv at z=6: {edges[-1]:.2f} Gyr"

    def test_returns_numpy_not_jax(self):
        edges = make_agebins_from_zred(1.0)
        assert isinstance(edges, np.ndarray), "should return numpy array (setup-time utility)"

    def test_n_bins_argument(self):
        edges = make_agebins_from_zred(1.0, n_bins=5)
        assert len(edges) == 6, f"n_bins=5 → 6 edges, got {len(edges)}"

    def test_low_zred_has_young_bins(self):
        edges = make_agebins_from_zred(0.5)
        # Should include ~30 Myr and ~100 Myr young edges
        assert any(0.02 < e < 0.05 for e in edges), "missing ~30 Myr young bin edge"
        assert any(0.08 < e < 0.15 for e in edges), "missing ~100 Myr young bin edge"


# ── PSB Continuity SFH ────────────────────────────────────────────


class TestPSBContinuitySFH:
    """Tests for Suess+2021 PSB nonparametric SFH."""

    @pytest.fixture
    def age_yr(self):
        return jnp.linspace(1e6, 10e9, 200)

    @pytest.fixture
    def default_edges(self):
        return jnp.array([0.1, 1.0, 3.0, 6.0, 13.7])

    def test_non_negative(self, age_yr, default_edges):
        sfr = psb_continuity(
            age_yr,
            10.0,
            tlast_gyr=0.5,
            tflex_gyr=2.0,
            bin_edges_gyr=default_edges,
            ratio_young=0.0,
            ratio_old_0=0.0,
        )
        assert jnp.all(sfr >= 0.0)

    def test_finite(self, age_yr, default_edges):
        sfr = psb_continuity(
            age_yr,
            10.0,
            tlast_gyr=0.5,
            tflex_gyr=2.0,
            bin_edges_gyr=default_edges,
            ratio_young=0.0,
            ratio_old_0=0.0,
        )
        assert jnp.all(jnp.isfinite(sfr))

    def test_mass_scales_with_log_total_mass(self, age_yr, default_edges):
        sfr10 = psb_continuity(
            age_yr,
            10.0,
            tlast_gyr=0.5,
            tflex_gyr=2.0,
            bin_edges_gyr=default_edges,
            ratio_young=0.0,
            ratio_old_0=0.0,
        )
        sfr11 = psb_continuity(
            age_yr,
            11.0,
            tlast_gyr=0.5,
            tflex_gyr=2.0,
            bin_edges_gyr=default_edges,
            ratio_young=0.0,
            ratio_old_0=0.0,
        )
        ratio = float(jnp.sum(sfr11) / jnp.sum(sfr10))
        assert abs(ratio - 10.0) < 0.5, f"10x mass increase should give ~10x SFR, got {ratio:.2f}"

    def test_jit_compatible(self, age_yr, default_edges):
        import functools

        # bin_edges_gyr is a fixed structural arg — bake it in via partial before JIT
        fn = jax.jit(functools.partial(psb_continuity, bin_edges_gyr=default_edges))
        sfr = fn(age_yr, 10.0, tlast_gyr=0.5, tflex_gyr=2.0, ratio_young=0.0, ratio_old_0=0.0)
        assert jnp.all(jnp.isfinite(sfr))

    def test_grad_wrt_log_total_mass(self, age_yr, default_edges):
        g = jax.grad(
            lambda m: jnp.sum(
                psb_continuity(
                    age_yr,
                    m,
                    tlast_gyr=0.5,
                    tflex_gyr=2.0,
                    bin_edges_gyr=default_edges,
                    ratio_young=0.0,
                    ratio_old_0=0.0,
                )
            )
        )(10.0)
        assert jnp.isfinite(g) and g > 0

    def test_grad_wrt_ratio_young(self, age_yr, default_edges):
        g = jax.grad(
            lambda r: jnp.sum(
                psb_continuity(
                    age_yr,
                    10.0,
                    tlast_gyr=0.5,
                    tflex_gyr=2.0,
                    bin_edges_gyr=default_edges,
                    ratio_young=r,
                    ratio_old_0=0.0,
                )
            )
        )(0.0)
        assert jnp.isfinite(g)


# ── Registry bin_edges_gyr support ───────────────────────────────


class TestRegistryBinEdges:
    """Tests for resolve_sfh bin_edges_gyr argument."""

    def test_custom_edges_passed_through(self):
        from tengri.components.stellar.sfh.registry import resolve_sfh

        edges = make_agebins_from_zred(2.0, n_bins=6)
        fn, params, _, _ = resolve_sfh("continuity", bin_edges_gyr=edges)
        age_yr = jnp.linspace(1e6, 3.3e9, 100)
        kwargs = {v[0]: 0.0 for v in params.values() if v[0] != "log_total_mass"}
        sfr = fn(age_yr, log_total_mass=10.0, **kwargs)
        assert jnp.all(jnp.isfinite(sfr))
        assert jnp.any(sfr > 0)

    def test_none_uses_default_edges(self):
        from tengri.components.stellar.sfh.registry import resolve_sfh

        fn, params, _, _ = resolve_sfh("continuity", bin_edges_gyr=None)
        age_yr = jnp.linspace(1e6, 13.7e9, 100)
        kwargs = {v[0]: 0.0 for v in params.values() if v[0] != "log_total_mass"}
        sfr = fn(age_yr, log_total_mass=10.0, **kwargs)
        assert jnp.all(jnp.isfinite(sfr))

    def test_dirichlet_custom_edges(self):
        from tengri.components.stellar.sfh.registry import resolve_sfh

        edges = make_agebins_from_zred(3.0, n_bins=6)
        fn, _, param_map, _ = resolve_sfh("dirichlet", bin_edges_gyr=edges)
        age_yr = jnp.linspace(1e6, 2.0e9, 100)
        # param_map: {public_name: (internal_name, scale, offset)}
        kwargs = {v[0]: 0.5 for v in param_map.values() if v[0] != "log_total_mass"}
        sfr = fn(age_yr, log_total_mass=10.0, **kwargs)
        assert jnp.all(jnp.isfinite(sfr))
