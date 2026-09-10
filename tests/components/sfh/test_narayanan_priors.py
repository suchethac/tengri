# SPDX-License-Identifier: BSD-3-Clause
"""Narayanan+2018 redshift-dependent dust attenuation priors.

The slope and bump centers moved with #2199. They used to be
``delta = -0.2 - 0.1 z`` and ``bump = max(0, 1 - 0.15 z)``, closed forms that
appear nowhere in Narayanan, Conroy, Davé, Johnson & Popping 2018 (ApJ 869, 70,
arXiv:1805.06905, doi:10.3847/1538-4357/aaed25) and whose slope term steepened
the curve with redshift, opposite the paper's own Section 5.1. They are now the
published median curves, through the same fitted table
``narayanan_z`` interpolates, so this file pins the *direction* the paper states
plus agreement with that single source, and no longer a formula.
"""

import numpy as np
import pytest

from tengri.components.dust.attenuation import (
    _NARAYANAN_BUMP_STRENGTH,
    _NARAYANAN_DELTA,
)
from tengri.components.dust.priors import narayanan_prior, narayanan_tau_prior
from tengri.parameters.priors import Gaussian

pytestmark = pytest.mark.bounds


class TestNarayananPrior:
    """narayanan_prior(z) centers on the published median at z."""

    def test_centers_are_the_fitted_table_at_the_nodes(self):
        """At every tabulated redshift the center must be the table entry.

        One source for the medians. A second copy of the numbers here would be
        free to drift from the one the law evaluates, and a user would then get
        a prior centered somewhere the model cannot reach.
        """
        for index, z in enumerate((0, 1, 2, 3, 4, 5, 6)):
            priors = narayanan_prior(z=float(z))
            np.testing.assert_allclose(
                priors["dust_delta"].mu, float(_NARAYANAN_DELTA[index]), atol=0.0, rtol=1e-12
            )
            np.testing.assert_allclose(
                priors["dust_bump_strength"].mu,
                float(_NARAYANAN_BUMP_STRENGTH[index]),
                atol=0.0,
                rtol=1e-12,
            )

    def test_delta_grows_less_negative_with_redshift(self):
        """The curve gets grayer with z, which is a rising delta.

        Narayanan et al. 2018 Section 5.1. The old center did the opposite.
        """
        assert narayanan_prior(z=6.0)["dust_delta"].mu > narayanan_prior(z=0.0)["dust_delta"].mu

    def test_the_bump_weakens_with_redshift(self):
        """E_b = m (0.85 - 1.9 delta) must fall from z=0 to z=6, and stay positive.

        Measured on the fitted table: 6.36 at z=0 against 1.96 at z=6. The bump
        does not vanish, which the old ``max(0, 1 - 0.15 z)`` center forced it
        to above z = 6.67.
        """

        def e_b(z: float) -> float:
            priors = narayanan_prior(z=z)
            delta = priors["dust_delta"].mu
            return float(priors["dust_bump_strength"].mu * (0.85 - 1.9 * delta))

        assert e_b(0.0) > e_b(6.0) > 0.0
        np.testing.assert_allclose(e_b(0.0), 6.3576, atol=1e-3)
        np.testing.assert_allclose(e_b(6.0), 1.9637, atol=1e-3)

    def test_outside_the_tabulated_range_the_end_node_is_held(self):
        """No extrapolation past the redshifts the paper tabulates."""
        for z_out, z_end in ((-1.0, 0.0), (10.0, 6.0)):
            out = narayanan_prior(z=z_out)
            end = narayanan_prior(z=z_end)
            assert out["dust_delta"].mu == end["dust_delta"].mu
            assert out["dust_bump_strength"].mu == end["dust_bump_strength"].mu

    def test_between_nodes_the_center_interpolates(self):
        """A half-integer redshift must land between its two neighbors."""
        lo = narayanan_prior(z=2.0)["dust_delta"].mu
        mid = narayanan_prior(z=2.5)["dust_delta"].mu
        hi = narayanan_prior(z=3.0)["dust_delta"].mu
        assert min(lo, hi) <= mid <= max(lo, hi)
        np.testing.assert_allclose(mid, 0.5 * (lo + hi), rtol=1e-12)

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

    def test_the_prior_is_usable_on_the_law_it_names(self):
        """The keys must be exactly what ``kriek_conroy`` reads.

        These are priors on ``kriek_conroy``'s two shape parameters, not on
        ``narayanan_z``, which since #2199 reads redshift and nothing else.
        """
        from tengri.components.dust.laws._registry import law_kwarg_names

        assert set(narayanan_prior(z=1.0)) <= law_kwarg_names("kriek_conroy")
        assert not set(narayanan_prior(z=1.0)) & law_kwarg_names("narayanan_z")


class TestNarayananTauPrior:
    """narayanan_tau_prior(z, log_mstar). Unchanged by #2199."""

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
