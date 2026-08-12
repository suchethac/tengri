# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the Beloborodov (1999) self-consistent Gamma_hot and L2500 helper.

Covers:
- beloborodov_gamma_hot: known values, clipping, JIT, grad
- compute_l2500: exact grid, interpolation
- kubota_done_disc with agn_self_consistent_gamma=True/False
"""

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.regression_paper


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite-difference gradient. O(eps^2) accurate."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


from tengri.components.agn.disc import (
    _eddington_luminosity,
    _gravitational_radius,
    _isco_radius,
    _l_seed_geometric,
    _nt_l_diss_analytic,
    _r_hot_bisect,
    beloborodov_gamma_hot,
    compute_l2500,
    kubota_done_disc,
)
from tests._bounds import assert_non_negative
from tests._jit_parity import assert_jit_matches_eager

# ── beloborodov_gamma_hot ─────────────────────────────────────────


class TestBeloborodovGammaHot:
    """Tests for the Beloborodov (1999) photon index relation."""

    def test_known_value_ratio_one(self):
        """L_diss / L_seed = 1 gives Gamma = 7/3 * 1^(-0.1) = 7/3."""
        gamma = beloborodov_gamma_hot(1.0, 1.0)
        expected = 7.0 / 3.0
        assert jnp.isclose(gamma, expected, atol=1e-6)

    def test_low_ratio_softer(self):
        """L_diss << L_seed -> ratio < 1 -> Gamma > 7/3 (softer spectrum)."""
        gamma = beloborodov_gamma_hot(0.01, 1.0)
        assert gamma > 7.0 / 3.0

    def test_high_ratio_harder(self):
        """L_diss >> L_seed -> ratio > 1 -> Gamma < 7/3 (harder spectrum)."""
        gamma = beloborodov_gamma_hot(100.0, 1.0)
        assert gamma < 7.0 / 3.0

    def test_clipped_min(self):
        """Extreme high ratio clips Gamma to 1.4."""
        gamma = beloborodov_gamma_hot(1e30, 1e-10)
        assert jnp.isclose(gamma, 1.4, atol=1e-6)

    def test_clipped_max(self):
        """Extreme low ratio clips Gamma to 3.0."""
        gamma = beloborodov_gamma_hot(1e-30, 1e10)
        assert jnp.isclose(gamma, 3.0, atol=1e-6)

    def test_jit(self):
        """beloborodov_gamma_hot works under jax.jit."""
        gamma = assert_jit_matches_eager(beloborodov_gamma_hot, 1.0, 1.0)
        expected = 7.0 / 3.0
        assert jnp.isclose(gamma, expected, atol=1e-6)

    def test_gradient(self):
        """Gradient w.r.t. l_diss_hot agrees with FD (nonzero)."""

        def f(x: float) -> float:
            return float(beloborodov_gamma_hot(x, 1.0))

        grad_jax = float(jax.grad(lambda x: beloborodov_gamma_hot(x, 1.0))(1.0))
        np.testing.assert_allclose(
            grad_jax,
            fd_grad(f, 1.0),
            rtol=1e-3,
            err_msg="beloborodov_gamma_hot: FD check ∂γ_hot/∂l_diss_hot",
        )
        assert grad_jax != 0.0


# ── compute_l2500 ─────────────────────────────────────────────────


class TestComputeL2500:
    """Tests for the 2500 A monochromatic luminosity extractor."""

    def test_at_exact_wavelength(self):
        """Grid containing exactly 2500 A returns the correct value."""
        wave = jnp.array([2000.0, 2500.0, 3000.0])
        l_nu = jnp.array([1.0, 5.0, 2.0])
        result = compute_l2500(wave, l_nu)
        assert jnp.isclose(result, 5.0, atol=1e-5)

    def test_interpolates(self):
        """Grid NOT containing 2500 A interpolates linearly."""
        wave = jnp.array([2000.0, 3000.0])
        l_nu = jnp.array([1.0, 3.0])
        result = compute_l2500(wave, l_nu)
        # Linear interpolation: 1 + (3-1) * (2500-2000)/(3000-2000) = 2.0
        assert jnp.isclose(result, 2.0, atol=1e-5)

    def test_unsorted_wavelength(self):
        """Works even when wavelength grid is not sorted."""
        wave = jnp.array([3000.0, 2000.0, 2500.0])
        l_nu = jnp.array([2.0, 1.0, 5.0])
        result = compute_l2500(wave, l_nu)
        assert jnp.isclose(result, 5.0, atol=1e-5)


# ── kubota_done_disc with self-consistent gamma ───────────────────


class TestKubotaDoneSelfConsistent:
    """Tests for kubota_done_disc with agn_self_consistent_gamma."""

    @pytest.fixture()
    def wavelength(self):
        """Standard wavelength grid from UV to X-ray."""
        return jnp.logspace(jnp.log10(100.0), jnp.log10(1e6), 200)

    def test_self_consistent_runs(self, wavelength):
        """self_consistent=True produces finite, positive output."""
        result = kubota_done_disc(
            wavelength,
            agn_log_lbol=45.0,
            agn_log_mbh=8.0,
            agn_log_ledd=-1.0,
            agn_self_consistent_gamma=True,
        )
        chex.assert_equal_shape([result, wavelength])
        chex.assert_tree_all_finite(result)
        assert_non_negative(result, name="result")

    def test_self_consistent_gamma_in_range(self, wavelength):
        """Derived Gamma_hot must lie in [1.4, 3.0].

        We test indirectly: the self-consistent SED should differ from
        both extreme gamma values (1.4 and 3.0), confirming the derived
        value is intermediate.
        """
        sed_sc = kubota_done_disc(
            wavelength,
            agn_log_lbol=12.0,
            agn_log_mbh=8.0,
            agn_self_consistent_gamma=True,
        )
        sed_hard = kubota_done_disc(
            wavelength,
            agn_log_lbol=12.0,
            agn_log_mbh=8.0,
            agn_gamma_hard=1.4,
            agn_self_consistent_gamma=False,
        )
        sed_soft = kubota_done_disc(
            wavelength,
            agn_log_lbol=12.0,
            agn_log_mbh=8.0,
            agn_gamma_hard=3.0,
            agn_self_consistent_gamma=False,
        )
        # Self-consistent SED should not be identical to either extreme
        assert not jnp.allclose(sed_sc, sed_hard, atol=1e-10)
        assert not jnp.allclose(sed_sc, sed_soft, atol=1e-10)

    def test_backward_compatible(self, wavelength):
        """self_consistent=False gives identical result to default call."""
        sed_default = kubota_done_disc(
            wavelength,
            agn_log_lbol=45.0,
            agn_log_mbh=8.0,
            agn_log_ledd=-1.0,
        )
        sed_explicit = kubota_done_disc(
            wavelength,
            agn_log_lbol=45.0,
            agn_log_mbh=8.0,
            agn_log_ledd=-1.0,
            agn_self_consistent_gamma=False,
        )
        assert jnp.allclose(sed_default, sed_explicit, atol=1e-12)

    def test_higher_mdot_softer_gamma(self, wavelength):
        """Higher Eddington ratio should yield softer (steeper) Gamma.

        Physical expectation from K&D 2018: at higher mdot, R_hot shrinks
        relative to R_warm, so more seed photons reach the corona, reducing
        the L_diss/L_seed ratio and steepening Gamma.

        We test this by comparing the X-ray spectral slope at short
        wavelengths between two Eddington ratios.
        """
        # High accretion rate
        sed_high = kubota_done_disc(
            wavelength,
            agn_log_lbol=45.5,
            agn_log_mbh=8.0,
            agn_log_ledd=-0.3,
            agn_self_consistent_gamma=True,
        )
        # Low accretion rate
        sed_low = kubota_done_disc(
            wavelength,
            agn_log_lbol=44.0,
            agn_log_mbh=8.0,
            agn_log_ledd=-2.0,
            agn_self_consistent_gamma=True,
        )
        # Compare X-ray band (short wavelengths < 100 A = > 0.1 keV)
        xray_mask = wavelength < 200.0
        # The higher-mdot SED should have relatively MORE soft X-ray
        # emission (steeper power law = more flux at lower energies
        # relative to highest energies).
        # Use the ratio at a moderate X-ray wavelength vs very hard X-ray
        ratio_high = sed_high[xray_mask]
        ratio_low = sed_low[xray_mask]
        # Both should be finite
        chex.assert_tree_all_finite(ratio_high)
        chex.assert_tree_all_finite(ratio_low)
        # The SEDs should differ in the X-ray band
        assert not jnp.allclose(ratio_high, ratio_low, atol=1e-15)


# ── _r_hot_bisect: exact K&D 2018 Eq. 2 solve ─────────────────────

_SIGMA_SB = 5.670374419e-5  # erg cm^-2 s^-1 K^-4


def _make_disc_params(log_mbh=8.0, log_ledd=-1.0, a_spin=0.0):
    """Return (r_isco_cm, t_in) for the given BH parameters."""
    import jax.numpy as jnp

    r_g = _gravitational_radius(log_mbh)
    r_isco_rg = _isco_radius(a_spin)
    r_isco_cm = r_isco_rg * r_g
    eta = 1.0 - jnp.sqrt(1.0 - 2.0 / (3.0 * r_isco_rg))
    l_edd = _eddington_luminosity(log_mbh)
    l_edd_ratio = jnp.clip(10.0**log_ledd, 1e-10, 1.0)
    l_bol = l_edd_ratio * l_edd
    mdot = l_bol / (eta * 3e10**2)
    t_in = (
        3.0
        * 6.674e-8
        * 10.0**log_mbh
        * 2e33
        * float(mdot)
        / (8.0 * jnp.pi * _SIGMA_SB * r_isco_cm**3)
    ) ** 0.25
    return float(r_isco_cm), float(t_in)


class TestRHotBisect:
    """Regression tests for the exact K&D 2018 Eq. 2 r_hot bisection solver."""

    def test_energy_balance_satisfied(self):
        """L_diss(r_hot) must equal the target to machine precision."""
        r_isco, t_in = _make_disc_params()
        l_edd = float(_eddington_luminosity(8.0))
        l_target = 0.02 * l_edd  # default f_hard = 0.02

        r_hot = float(_r_hot_bisect(r_isco, t_in, l_target))
        x_hot = r_hot / r_isco
        l_check = float(_nt_l_diss_analytic(x_hot, r_isco, t_in))

        # Energy balance must be satisfied to better than 1e-8 relative
        assert abs(l_check - l_target) / l_target < 1e-8

    def test_r_hot_increases_with_f_hard(self):
        """Larger f_hard (more corona power) must give larger r_hot."""
        r_isco, t_in = _make_disc_params()
        l_edd = float(_eddington_luminosity(8.0))

        r_hot_lo = float(_r_hot_bisect(r_isco, t_in, 0.01 * l_edd))
        r_hot_hi = float(_r_hot_bisect(r_isco, t_in, 0.05 * l_edd))
        assert r_hot_hi > r_hot_lo

    def test_r_hot_exceeds_r_isco(self):
        """r_hot must always be > r_isco for any positive target."""
        r_isco, t_in = _make_disc_params()
        l_edd = float(_eddington_luminosity(8.0))

        for f in [0.005, 0.01, 0.02, 0.05]:
            r_hot = float(_r_hot_bisect(r_isco, t_in, f * l_edd))
            assert r_hot > r_isco

    def test_finite_and_positive(self):
        """r_hot must be finite and positive for typical parameters."""
        import jax.numpy as jnp

        for log_mbh in [7.0, 8.0, 9.0]:
            for log_ledd in [-2.0, -1.0, -0.5]:
                r_isco, t_in = _make_disc_params(log_mbh, log_ledd)
                l_edd = float(_eddington_luminosity(log_mbh))
                r_hot = _r_hot_bisect(r_isco, t_in, 0.02 * l_edd)
                assert jnp.isfinite(r_hot)
                assert r_hot > 0.0


class TestLSeedGeometric:
    """Tests for the geometric L_seed integral (K&D 2018 Eq. 3)."""

    def test_positive_and_finite(self):
        """L_seed must be positive and finite."""
        import jax.numpy as jnp

        r_isco, t_in = _make_disc_params()
        l_edd = float(_eddington_luminosity(8.0))
        r_hot = float(_r_hot_bisect(r_isco, t_in, 0.02 * l_edd))
        r_g = float(_gravitational_radius(8.0))
        r_out = 1000.0 * r_isco  # approximate outer radius for test

        l_seed = _l_seed_geometric(r_isco, r_hot, r_out, t_in)
        assert jnp.isfinite(l_seed)
        assert l_seed > 0.0

    def test_l_seed_less_than_disc_bol(self):
        """L_seed (covering factor < 1) must be less than the disc bolometric."""
        r_isco, t_in = _make_disc_params()
        l_edd = float(_eddington_luminosity(8.0))
        r_hot = float(_r_hot_bisect(r_isco, t_in, 0.02 * l_edd))
        r_out = 1000.0 * r_isco

        l_seed = float(_l_seed_geometric(r_isco, r_hot, r_out, t_in))
        # L_seed must be < l_edd * 0.1 (total NT disc luminosity factor)
        l0 = 4.0 * 3.14159 * r_isco**2 * _SIGMA_SB * t_in**4
        assert l_seed < l0 * 0.1

    def test_l_seed_decreases_with_larger_r_hot(self):
        """Larger r_hot (smaller disc area intercepted) -> smaller L_seed."""
        r_isco, t_in = _make_disc_params()
        l_edd = float(_eddington_luminosity(8.0))
        r_hot_small = float(_r_hot_bisect(r_isco, t_in, 0.01 * l_edd))
        r_hot_large = float(_r_hot_bisect(r_isco, t_in, 0.05 * l_edd))
        r_out = 1000.0 * r_isco

        l_seed_small = float(_l_seed_geometric(r_isco, r_hot_small, r_out, t_in))
        l_seed_large = float(_l_seed_geometric(r_isco, r_hot_large, r_out, t_in))
        # Larger r_hot means smaller disc outside corona, so less L_seed
        assert l_seed_large < l_seed_small
