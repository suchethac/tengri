# SPDX-License-Identifier: BSD-3-Clause
"""Tests for models/sps/mass_remaining.py — stellar remnant mass fractions.

Covers all three IMFs, main-sequence lifetime, Newton-Raphson turnoff
inversion, remnant mass recipes, and the top-level
compute_mass_remaining_fraction integration.
"""

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests._jit_parity import assert_jit_matches_eager

pytestmark = pytest.mark.conservation


# ── IMF shape tests ───────────────────────────────────────────────


class TestChabrierIMF:
    """_chabrier_imf uses a lognormal below 1 Msun and a power-law m^{-1.3} above."""

    def test_positive_output(self):
        from tengri.components.stellar.sps.mass_remaining import _chabrier_imf

        for log_m in [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]:
            assert float(_chabrier_imf(log_m)) > 0.0

    def test_above_one_msun_slope_is_minus_1p3(self):
        """Above 1 Msun the slope d(log phi)/d(log m) should be -1.3 (Chabrier, not Salpeter)."""
        from tengri.components.stellar.sps.mass_remaining import _chabrier_imf

        # Evaluate at log_m=0.5 and log_m=1.0 (m=3.16 and 10 Msun)
        log_m1, log_m2 = 0.5, 1.0
        phi1 = float(_chabrier_imf(log_m1))
        phi2 = float(_chabrier_imf(log_m2))
        # d log(phi) / d log(m) = log10(phi2/phi1) / (log_m2 - log_m1)
        slope = np.log10(phi2 / phi1) / (log_m2 - log_m1)
        np.testing.assert_allclose(slope, -1.3, atol=0.05)

    def test_above_one_msun_slope_not_salpeter(self):
        """Slope above 1 Msun is -1.3, NOT -1.35 (which would be Salpeter)."""
        from tengri.components.stellar.sps.mass_remaining import _chabrier_imf

        log_m1, log_m2 = 0.5, 1.5
        phi1 = float(_chabrier_imf(log_m1))
        phi2 = float(_chabrier_imf(log_m2))
        slope = np.log10(phi2 / phi1) / (log_m2 - log_m1)
        # Confirm it's closer to -1.3 than -1.35
        assert abs(slope - (-1.3)) < abs(slope - (-1.35))

    def test_continuity_at_one_msun(self):
        """IMF is continuous at the lognormal/power-law boundary (m=1)."""
        from tengri.components.stellar.sps.mass_remaining import _chabrier_imf

        eps = 1e-6
        phi_below = float(_chabrier_imf(-eps))  # just below log_m=0 (m=1)
        phi_above = float(_chabrier_imf(eps))  # just above log_m=0
        np.testing.assert_allclose(phi_below, phi_above, rtol=1e-4)

    def test_decreasing_above_peak(self):
        """Chabrier IMF decreases with mass above the lognormal peak (~0.1 Msun)."""
        from tengri.components.stellar.sps.mass_remaining import _chabrier_imf

        # At m=1 Msun (log_m=0) down to m=100 Msun (log_m=2), phi should decrease
        phi_1 = float(_chabrier_imf(0.0))
        phi_100 = float(_chabrier_imf(2.0))
        assert phi_1 > phi_100


class TestSalpeterIMF:
    """_salpeter_imf is a pure power law m^{-1.35} (Salpeter 1955)."""

    def test_positive_output(self):
        from tengri.components.stellar.sps.mass_remaining import _salpeter_imf

        for log_m in [-1.0, 0.0, 1.0, 2.0]:
            assert float(_salpeter_imf(log_m)) > 0.0

    def test_slope_is_minus_1p35(self):
        """d(log phi)/d(log m) = -1.35 across all masses."""
        from tengri.components.stellar.sps.mass_remaining import _salpeter_imf

        log_m1, log_m2 = 0.0, 1.0
        phi1 = float(_salpeter_imf(log_m1))
        phi2 = float(_salpeter_imf(log_m2))
        slope = np.log10(phi2 / phi1) / (log_m2 - log_m1)
        np.testing.assert_allclose(slope, -1.35, atol=0.01)

    def test_monotonically_decreasing(self):
        """Salpeter IMF is strictly decreasing with mass."""
        from tengri.components.stellar.sps.mass_remaining import _salpeter_imf

        log_masses = jnp.linspace(-1.0, 2.0, 20)
        phis = jnp.array([_salpeter_imf(float(lm)) for lm in log_masses])
        diffs = jnp.diff(phis)
        assert jnp.all(diffs < 0.0)


class TestKroupaIMF:
    """_kroupa_imf has three power-law slopes with exact continuity at the breakpoints."""

    def test_positive_output(self):
        from tengri.components.stellar.sps.mass_remaining import _kroupa_imf

        for log_m in [np.log10(0.05), np.log10(0.1), np.log10(0.3), np.log10(1.0), 1.0]:
            assert float(_kroupa_imf(log_m)) > 0.0

    def test_continuity_at_0p08_msun(self):
        """Kroupa IMF is continuous at the 0.08 Msun breakpoint (c1 = 0.08^{0.7-0.7} = 1?)."""
        from tengri.components.stellar.sps.mass_remaining import _kroupa_imf

        # At m=0.08: phi_low(0.08) = 0.08^0.7, phi_mid(0.08) = c1 * 0.08^{-0.3}
        # With c1=0.08: phi_mid = 0.08 * 0.08^{-0.3} = 0.08^{0.7} = phi_low ✓
        eps = 1e-6
        log_m_break = np.log10(0.08)
        phi_low = float(_kroupa_imf(log_m_break - eps))
        phi_mid = float(_kroupa_imf(log_m_break + eps))
        np.testing.assert_allclose(phi_low, phi_mid, rtol=1e-3)

    def test_continuity_at_0p5_msun(self):
        """Kroupa IMF is continuous at the 0.5 Msun breakpoint."""
        from tengri.components.stellar.sps.mass_remaining import _kroupa_imf

        eps = 1e-6
        log_m_break = np.log10(0.5)
        phi_mid = float(_kroupa_imf(log_m_break - eps))
        phi_high = float(_kroupa_imf(log_m_break + eps))
        np.testing.assert_allclose(phi_mid, phi_high, rtol=1e-3)

    def test_high_mass_slope_minus_1p3(self):
        """Above 0.5 Msun the slope is -1.3."""
        from tengri.components.stellar.sps.mass_remaining import _kroupa_imf

        log_m1, log_m2 = np.log10(1.0), np.log10(10.0)
        phi1 = float(_kroupa_imf(log_m1))
        phi2 = float(_kroupa_imf(log_m2))
        slope = np.log10(phi2 / phi1) / (log_m2 - log_m1)
        np.testing.assert_allclose(slope, -1.3, atol=0.05)


# ── Main-sequence lifetime ────────────────────────────────────────


class TestMsLifetime:
    """_ms_lifetime_gyr = 10.0 * m^{-2.5} + 0.1 * m^{-0.75}."""

    def test_sun_lifetime(self):
        """Solar lifetime should be ~10.1 Gyr."""
        from tengri.components.stellar.sps.mass_remaining import _ms_lifetime_gyr

        t_sun = float(_ms_lifetime_gyr(1.0))
        # 10.0 * 1^{-2.5} + 0.1 * 1^{-0.75} = 10.1 Gyr
        np.testing.assert_allclose(t_sun, 10.1, rtol=1e-6)

    def test_massive_star_short_lifetime(self):
        """A 25 Msun star should have a lifetime < 15 Myr.

        The simplified Hurley+2000 formula gives
        t = 10 * 25^{-2.5} + 0.1 * 25^{-0.75} ≈ 12.1 Myr,
        which is ~20% above the 10 Myr rule-of-thumb but within the
        ~20% accuracy stated in the Hurley+2000 paper.
        """
        from tengri.components.stellar.sps.mass_remaining import _ms_lifetime_gyr

        t = float(_ms_lifetime_gyr(25.0))
        assert t < 0.015  # < 15 Myr in Gyr

    def test_low_mass_star_long_lifetime(self):
        """A 0.3 Msun star should have a lifetime > 100 Gyr."""
        from tengri.components.stellar.sps.mass_remaining import _ms_lifetime_gyr

        t = float(_ms_lifetime_gyr(0.3))
        assert t > 100.0

    def test_decreasing_with_mass(self):
        """Lifetime decreases monotonically with mass."""
        from tengri.components.stellar.sps.mass_remaining import _ms_lifetime_gyr

        masses = [0.5, 1.0, 2.0, 5.0, 10.0, 25.0]
        lifetimes = [float(_ms_lifetime_gyr(m)) for m in masses]
        for i in range(len(lifetimes) - 1):
            assert lifetimes[i] > lifetimes[i + 1]

    def test_analytic_formula(self):
        """Verify exact implementation: 10.0 * m^{-2.5} + 0.1 * m^{-0.75}."""
        from tengri.components.stellar.sps.mass_remaining import _ms_lifetime_gyr

        m = 3.0
        expected = 10.0 * m ** (-2.5) + 0.1 * m ** (-0.75)
        np.testing.assert_allclose(float(_ms_lifetime_gyr(m)), expected, rtol=1e-10)


# ── Turnoff mass (Newton-Raphson inversion of lifetime) ───────────


class TestTurnoffMass:
    """_turnoff_mass(age_gyr) inverts _ms_lifetime_gyr via Newton's method."""

    def test_sun_at_10_gyr(self):
        """At t=10 Gyr turnoff mass should be near 1 Msun."""
        from tengri.components.stellar.sps.mass_remaining import _turnoff_mass

        m_to = float(_turnoff_mass(10.0))
        # Not exact because 10.1 Gyr is the solar lifetime, but should be close
        np.testing.assert_allclose(m_to, 1.0, rtol=0.05)

    def test_roundtrip_consistency(self):
        """lifetime(_turnoff_mass(t)) ≈ t for several ages."""
        from tengri.components.stellar.sps.mass_remaining import _ms_lifetime_gyr, _turnoff_mass

        for age in [1.0, 5.0, 10.0, 13.0]:
            m_to = float(_turnoff_mass(age))
            t_back = float(_ms_lifetime_gyr(m_to))
            np.testing.assert_allclose(t_back, age, rtol=0.01)

    def test_young_age_high_turnoff_mass(self):
        """Young stellar populations (0.01 Gyr) have high turnoff masses."""
        from tengri.components.stellar.sps.mass_remaining import _turnoff_mass

        m_to = float(_turnoff_mass(0.01))
        assert m_to > 10.0

    def test_old_age_low_turnoff_mass(self):
        """Old populations (13.8 Gyr) have low turnoff masses (~0.9 Msun)."""
        from tengri.components.stellar.sps.mass_remaining import _turnoff_mass

        m_to = float(_turnoff_mass(13.8))
        assert 0.5 < m_to < 1.5

    def test_decreasing_with_age(self):
        """Turnoff mass decreases monotonically with time."""
        from tengri.components.stellar.sps.mass_remaining import _turnoff_mass

        ages = [0.1, 1.0, 5.0, 10.0, 13.0]
        m_tos = [float(_turnoff_mass(a)) for a in ages]
        for i in range(len(m_tos) - 1):
            assert m_tos[i] > m_tos[i + 1]

    def test_clipped_to_valid_range(self):
        """Output is always within [0.08, 300] Msun regardless of age input."""
        from tengri.components.stellar.sps.mass_remaining import _turnoff_mass

        # Very young (t→0): clipped to 300 Msun
        m_young = float(_turnoff_mass(1e-6))
        assert 0.08 <= m_young <= 300.0

        # Very old (t→1000 Gyr): clipped to 0.08 Msun
        m_old = float(_turnoff_mass(1000.0))
        assert 0.08 <= m_old <= 300.0


# ── Remnant mass (Kalirai+2008 WDs, NS, BH) ───────────────────────


class TestRemnantMass:
    """_remnant_mass implements Kalirai+2008 WDs / 1.4 Msun NS / 0.4*m_init BH."""

    def test_below_0p5_returns_initial_mass(self):
        """Sub-stellar objects (m<0.5) return their initial mass (no remnant)."""
        from tengri.components.stellar.sps.mass_remaining import _remnant_mass

        for m in [0.1, 0.3, 0.49]:
            np.testing.assert_allclose(float(_remnant_mass(m)), m, rtol=1e-8)

    def test_wd_regime_kalirai(self):
        """For 0.5 <= m < 8 Msun, remnant = 0.394 + 0.109 * m (Kalirai+2008 WD)."""
        from tengri.components.stellar.sps.mass_remaining import _remnant_mass

        for m in [0.5, 1.0, 2.0, 4.0, 7.0]:
            expected = 0.394 + 0.109 * m
            np.testing.assert_allclose(float(_remnant_mass(m)), expected, rtol=1e-6)

    def test_ns_regime(self):
        """For 8 <= m < 25 Msun, remnant = 1.4 Msun (neutron star).

        The boundary condition uses strict less-than at m=25: ``m < 25.0``
        routes to BH, so 25.0 is the first BH mass, not NS.
        """
        from tengri.components.stellar.sps.mass_remaining import _remnant_mass

        for m in [8.0, 10.0, 15.0, 20.0, 24.9]:
            np.testing.assert_allclose(float(_remnant_mass(m)), 1.4, rtol=1e-6)

    def test_bh_regime_fryer2012(self):
        """For m > 25 Msun, remnant = 0.4 * m_init (Fryer+2012 BH, NOT 0.5)."""
        from tengri.components.stellar.sps.mass_remaining import _remnant_mass

        for m in [30.0, 50.0, 100.0]:
            expected = 0.4 * m
            np.testing.assert_allclose(float(_remnant_mass(m)), expected, rtol=1e-6)

    def test_bh_regime_not_half(self):
        """Confirm BH fraction is 0.4, not 0.5 (common mistake vs Fryer+2012)."""
        from tengri.components.stellar.sps.mass_remaining import _remnant_mass

        m = 50.0
        result = float(_remnant_mass(m))
        # 0.4 * 50 = 20, NOT 0.5 * 50 = 25
        np.testing.assert_allclose(result, 20.0, rtol=1e-6)
        assert abs(result - 25.0) > 1.0  # definitely not 0.5 * m

    def test_remnant_less_than_initial(self):
        """Remnant mass is always less than initial mass for m > 0.5."""
        from tengri.components.stellar.sps.mass_remaining import _remnant_mass

        for m in [0.8, 2.0, 10.0, 50.0, 100.0]:
            r = float(_remnant_mass(m))
            assert r < m, f"Remnant {r} >= initial {m} at m={m} Msun"


# ── Top-level compute_mass_remaining_fraction ─────────────────────


class TestComputeMassRemainingFraction:
    """Tests for the main integration function."""

    def test_default_output_shape(self):
        """Output shape matches input age_gyr array shape."""
        from tengri.components.stellar.sps.mass_remaining import compute_mass_remaining_fraction

        ages = jnp.array([1.0, 5.0, 10.0, 13.0])
        result = compute_mass_remaining_fraction(ages)
        chex.assert_equal_shape([result, ages])

    def test_fraction_in_zero_one(self):
        """Mass remaining fraction must be in (0, 1]."""
        from tengri.components.stellar.sps.mass_remaining import compute_mass_remaining_fraction

        ages = jnp.linspace(0.1, 13.8, 20)
        fractions = compute_mass_remaining_fraction(ages)
        assert jnp.all(fractions > 0.0)
        assert jnp.all(fractions <= 1.0)

    def test_finite_output(self):
        """No NaN or Inf values for physically sensible ages."""
        from tengri.components.stellar.sps.mass_remaining import compute_mass_remaining_fraction

        ages = jnp.logspace(-1, 1.2, 30)  # 0.1 to 15.8 Gyr
        fractions = compute_mass_remaining_fraction(ages)
        chex.assert_tree_all_finite(fractions)

    def test_decreasing_with_age(self):
        """Older populations have more mass locked up in remnants (lower fraction)."""
        from tengri.components.stellar.sps.mass_remaining import compute_mass_remaining_fraction

        ages = jnp.array([0.5, 2.0, 8.0, 13.0])
        fracs = compute_mass_remaining_fraction(ages)
        # Should be monotonically decreasing
        diffs = jnp.diff(fracs)
        assert jnp.all(diffs < 0.0), f"Fractions not decreasing: {fracs}"

    def test_all_three_imfs_run(self):
        """All registered IMF names produce finite output."""
        from tengri.components.stellar.sps.mass_remaining import compute_mass_remaining_fraction

        ages = jnp.array([1.0, 5.0, 10.0])
        for imf_name in ["chabrier", "salpeter", "kroupa"]:
            frac = compute_mass_remaining_fraction(ages, imf=imf_name)
            assert jnp.all(jnp.isfinite(frac)), f"NaN/Inf for IMF '{imf_name}'"

    def test_unknown_imf_raises_valueerror(self):
        """Unknown IMF name raises ValueError naming the bad IMF."""
        from tengri.components.stellar.sps.mass_remaining import compute_mass_remaining_fraction

        with pytest.raises(ValueError, match="badimf"):
            compute_mass_remaining_fraction(jnp.array([5.0]), imf="badimf")

    def test_salpeter_greater_than_chabrier(self):
        """Salpeter (steeper at high mass) retains less mass than Chabrier.

        Salpeter over-weights low-mass long-lived stars relative to
        Chabrier's lognormal, so remaining fraction should be higher for
        Salpeter at intermediate ages when high-mass stars are gone.
        """
        from tengri.components.stellar.sps.mass_remaining import compute_mass_remaining_fraction

        ages = jnp.array([5.0])
        f_sal = float(compute_mass_remaining_fraction(ages, imf="salpeter")[0])
        f_cha = float(compute_mass_remaining_fraction(ages, imf="chabrier")[0])
        # Salpeter weights the low-mass (surviving) end more heavily
        assert f_sal > f_cha

    def test_young_population_high_fraction(self):
        """Young populations (100 Myr) should retain >75% of initial stellar mass.

        At 100 Myr the stellar turnoff is ~5-6 Msun.  Stars above that mass
        have already evolved off the main sequence and deposited compact
        remnants.  The mass-weighted integral over a Chabrier IMF yields
        a surviving fraction around 81%, well above 75% but below 90%.
        """
        from tengri.components.stellar.sps.mass_remaining import compute_mass_remaining_fraction

        frac = float(compute_mass_remaining_fraction(jnp.array([0.1]))[0])
        assert frac > 0.75

    def test_old_population_retains_most_mass(self):
        """Even at 13 Gyr a significant fraction of initial mass remains
        in remnants and low-mass stars."""
        from tengri.components.stellar.sps.mass_remaining import compute_mass_remaining_fraction

        frac = float(compute_mass_remaining_fraction(jnp.array([13.0]))[0])
        # Should be substantial (not close to zero) — stars < 1 Msun are still alive
        assert frac > 0.4

    def test_jit_compatible(self):
        """Function is JIT-compilable."""
        from tengri.components.stellar.sps.mass_remaining import compute_mass_remaining_fraction

        ages = jnp.array([1.0, 5.0, 10.0])
        result = assert_jit_matches_eager(compute_mass_remaining_fraction, ages)
        chex.assert_tree_all_finite(result)
        chex.assert_equal_shape([result, ages])

    def test_n_mass_parameter(self):
        """Changing n_mass (integration resolution) gives consistent results."""
        from tengri.components.stellar.sps.mass_remaining import compute_mass_remaining_fraction

        ages = jnp.array([5.0])
        f_coarse = float(compute_mass_remaining_fraction(ages, n_mass=100)[0])
        f_fine = float(compute_mass_remaining_fraction(ages, n_mass=1000)[0])
        # Both are valid; agreement within a few percent
        np.testing.assert_allclose(f_coarse, f_fine, rtol=0.02)


class TestMassRemainingGradient:
    """JAX autodiff gradient tests for differentiable sub-functions.

    ``compute_mass_remaining_fraction`` uses ``jnp.where(m_grid < m_to, ...)``
    where both branch values are independent of ``m_to`` (they are constants
    from the mass grid). JAX AD therefore returns zero for the gradient of the
    integral w.r.t. age — this is a Heaviside step function, and JAX AD does
    not differentiate through the boolean condition, only through the values.

    We test the two genuinely-differentiable sub-functions instead:
    ``_ms_lifetime_gyr`` (analytic power-law) and ``_turnoff_mass``
    (Newton-Raphson iteration using only JAX arithmetic).
    """

    def test_ms_lifetime_gradient_negative(self):
        """Lifetime decreases with mass: d(t_ms)/d(m) < 0."""
        from tengri.components.stellar.sps.mass_remaining import _ms_lifetime_gyr

        g = float(jax.grad(_ms_lifetime_gyr)(2.0))
        assert g < 0.0, f"d(t_ms)/d(m) should be negative, got {g}"

    def test_ms_lifetime_gradient_fd(self):
        """JAX gradient of _ms_lifetime_gyr matches central FD."""
        from tengri.components.stellar.sps.mass_remaining import _ms_lifetime_gyr

        m0 = 2.0
        eps = 0.01
        g_jax = float(jax.grad(_ms_lifetime_gyr)(m0))
        g_fd = (float(_ms_lifetime_gyr(m0 + eps)) - float(_ms_lifetime_gyr(m0 - eps))) / (
            2.0 * eps
        )
        np.testing.assert_allclose(g_jax, g_fd, rtol=1e-3)

    def test_turnoff_mass_gradient_negative(self):
        """Turnoff mass decreases with age: d(m_to)/d(age) < 0."""
        from tengri.components.stellar.sps.mass_remaining import _turnoff_mass

        g = float(jax.grad(_turnoff_mass)(5.0))
        assert g < 0.0, f"d(m_turnoff)/d(age) should be negative, got {g}"

    def test_turnoff_mass_gradient_fd(self):
        """JAX gradient of _turnoff_mass (Newton solver) matches central FD."""
        from tengri.components.stellar.sps.mass_remaining import _turnoff_mass

        age0 = 5.0
        eps = 0.05
        g_jax = float(jax.grad(_turnoff_mass)(age0))
        g_fd = (float(_turnoff_mass(age0 + eps)) - float(_turnoff_mass(age0 - eps))) / (2.0 * eps)
        np.testing.assert_allclose(g_jax, g_fd, rtol=1e-2)
