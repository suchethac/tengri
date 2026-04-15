"""Tests for Narayanan+2018 redshift-dependent dust attenuation priors."""

import numpy as np

from tengri.parameters.priors import Gaussian
from tengri.components.dust.priors import narayanan_prior, narayanan_tau_prior


class TestNarayananPrior:
    """Test narayanan_prior(z) returns correct z-dependent distributions."""

    def test_z0_baseline_delta(self):
        """At z=0, delta mean should be ~-0.2."""
        priors = narayanan_prior(z=0.0)
        np.testing.assert_allclose(priors["dust_delta"].mu, -0.2, atol=1e-10)

    def test_z0_baseline_bump(self):
        """At z=0, bump strength mean should be ~1.0."""
        priors = narayanan_prior(z=0.0)
        np.testing.assert_allclose(priors["dust_bump_strength"].mu, 1.0, atol=1e-10)

    def test_delta_more_negative_at_high_z(self):
        """Higher z should give more negative (steeper) delta."""
        priors_z0 = narayanan_prior(z=0.0)
        priors_z4 = narayanan_prior(z=4.0)
        assert priors_z4["dust_delta"].mu < priors_z0["dust_delta"].mu

    def test_z4_delta_value(self):
        """At z=4, delta mean should be -0.2 - 0.1*4 = -0.6."""
        priors = narayanan_prior(z=4.0)
        np.testing.assert_allclose(priors["dust_delta"].mu, -0.6, atol=1e-10)

    def test_bump_floors_at_zero(self):
        """Bump strength mean should never go negative."""
        # At z=10, formula gives 1.0 - 0.15*10 = -0.5, but should floor at 0
        priors = narayanan_prior(z=10.0)
        assert priors["dust_bump_strength"].mu >= 0.0

    def test_bump_floors_exact_at_threshold(self):
        """At z = 1/0.15 ~ 6.67, bump should be exactly 0."""
        z_threshold = 1.0 / 0.15
        priors = narayanan_prior(z=z_threshold)
        np.testing.assert_allclose(priors["dust_bump_strength"].mu, 0.0, atol=1e-10)

    def test_returns_gaussian_distributions(self):
        """Both values should be Gaussian distribution instances."""
        priors = narayanan_prior(z=1.0)
        assert isinstance(priors["dust_delta"], Gaussian)
        assert isinstance(priors["dust_bump_strength"], Gaussian)

    def test_returns_dict_with_expected_keys(self):
        """Should return exactly dust_delta and dust_bump_strength."""
        priors = narayanan_prior(z=2.0)
        assert set(priors.keys()) == {"dust_delta", "dust_bump_strength"}

    def test_sigma_values(self):
        """Sigma should be 0.15 for delta, 0.3 for bump (z-independent)."""
        priors = narayanan_prior(z=1.5)
        np.testing.assert_allclose(priors["dust_delta"].sigma, 0.15, atol=1e-10)
        np.testing.assert_allclose(priors["dust_bump_strength"].sigma, 0.3, atol=1e-10)


class TestNarayananTauPrior:
    """Test narayanan_tau_prior(z, log_mstar)."""

    def test_returns_dict_with_tau_diff(self):
        """Should return dict with dust_tau_diff key."""
        priors = narayanan_tau_prior(z=0.0)
        assert set(priors.keys()) == {"dust_tau_diff"}
        assert isinstance(priors["dust_tau_diff"], Gaussian)

    def test_default_mstar(self):
        """Default log_mstar=10.0 at z=0: tau_mean = 0.5 * 1.0 * 1.0 = 0.5."""
        priors = narayanan_tau_prior(z=0.0, log_mstar=10.0)
        np.testing.assert_allclose(priors["dust_tau_diff"].mu, 0.5, atol=1e-10)

    def test_tau_scales_with_stellar_mass(self):
        """Higher stellar mass should give higher tau."""
        priors_low = narayanan_tau_prior(z=1.0, log_mstar=9.0)
        priors_high = narayanan_tau_prior(z=1.0, log_mstar=11.0)
        assert priors_high["dust_tau_diff"].mu > priors_low["dust_tau_diff"].mu

    def test_tau_scales_with_redshift(self):
        """Higher redshift should give higher tau."""
        priors_z0 = narayanan_tau_prior(z=0.0, log_mstar=10.0)
        priors_z2 = narayanan_tau_prior(z=2.0, log_mstar=10.0)
        assert priors_z2["dust_tau_diff"].mu > priors_z0["dust_tau_diff"].mu

    def test_sigma_proportional_to_mean(self):
        """Sigma should be 0.3*tau_mean + 0.1."""
        priors = narayanan_tau_prior(z=1.0, log_mstar=10.5)
        tau_mu = priors["dust_tau_diff"].mu
        expected_sigma = 0.3 * tau_mu + 0.1
        np.testing.assert_allclose(priors["dust_tau_diff"].sigma, expected_sigma, atol=1e-10)

    def test_massive_high_z_galaxy(self):
        """Massive high-z galaxy: tau should be substantially larger."""
        priors = narayanan_tau_prior(z=3.0, log_mstar=11.0)
        tau = priors["dust_tau_diff"].mu
        # 0.5 * (10^1)^0.5 * (4)^0.5 = 0.5 * sqrt(10) * 2 ~ 3.16
        expected = 0.5 * (10**1) ** 0.5 * (4.0) ** 0.5
        np.testing.assert_allclose(tau, expected, rtol=1e-10)
