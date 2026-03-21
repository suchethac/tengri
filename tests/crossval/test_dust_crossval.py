"""Cross-validate dust attenuation against bagpipes.

Charlot & Fall (2000) attenuation is a power-law curve:
    A(lambda) / A_V = (5500 / lambda)^n

Both codes implement this. The parameterization differs:
- diffsed: optical depth tau_V, transmission = exp(-tau_V * (lam/5500)^n)
- bagpipes CF00: A_V magnitudes, transmission = 10^(-A_V * A_lam / 2.5)

Mapping: tau_V = A_V * ln(10) / 2.5  (i.e., A_V = 1.086 * tau_V)

For old stars (age >> t_birth=10 Myr) in diffsed's Charlot & Fall:
    tau_eff = tau_v2 only (birth cloud weight = 0)
For young stars (age << t_birth):
    tau_eff = tau_v1 + tau_v2

We compare the attenuation CURVE shape (wavelength dependence),
not the age dependence, since bagpipes CF00 doesn't separate
birth-cloud vs diffuse — it returns a single A(lambda)/A_V curve.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

pytestmark = pytest.mark.crossval

bagpipes_dust = pytest.importorskip(
    "bagpipes.models.dust_attenuation_model",
    reason="bagpipes not installed",
)

from diffsed.models.dust.attenuation import two_component_dust


class TestDustCurveCrossval:
    """Compare power-law attenuation curve shape."""

    @pytest.mark.parametrize("n_slope", [-0.7, -1.0, -1.3])
    def test_power_law_shape_matches(self, optical_wavelengths, n_slope):
        """The wavelength dependence (lambda/5500)^n should be identical.

        bagpipes CF00 returns A(lambda)/A_V = (5500/lambda)^n,
        diffsed uses (lambda/5500)^n in the exponent. Since
        (5500/lambda)^n = (lambda/5500)^(-n), we need to be careful
        with sign conventions.
        """
        wavs = optical_wavelengths

        # bagpipes CF00 curve: A(lam)/A_V
        bp_dust = bagpipes_dust.dust_attenuation(wavs, {"type": "CF00", "n": -n_slope})
        a_curve_bp = bp_dust.A_cont  # A(lam)/A_V = (5500/lam)^n_bp

        # diffsed: for old stars (weight=0), transmission = exp(-tau_v2 * (lam/5500)^n)
        # The curve shape is (lam/5500)^n = (5500/lam)^(-n)
        # Mapping: A(lam)/A_V ∝ (5500/lam)^(-n_diffsed)
        # diffsed n_slope = -0.7 means (lam/5500)^(-0.7) = (5500/lam)^(0.7)
        # bagpipes n = 0.7 means (5500/lam)^(0.7)
        # So: bagpipes n_bp = -n_diffsed

        # Build diffsed attenuation curve shape for comparison
        curve_diffsed = (wavs / 5500.0) ** n_slope  # (lam/5500)^n

        # Normalize both to V-band (5500 A) for shape comparison
        # At 5500A, both should be 1.0
        idx_v = np.argmin(np.abs(wavs - 5500.0))

        ratio_bp = a_curve_bp / a_curve_bp[idx_v]
        ratio_ds = curve_diffsed / curve_diffsed[idx_v]

        np.testing.assert_allclose(
            ratio_ds,
            ratio_bp,
            rtol=1e-4,
            err_msg=f"Power-law curve shape mismatch for n={n_slope}",
        )

    def test_transmission_mapping_tau_to_av(self, optical_wavelengths):
        """Verify tau_V <-> A_V mapping gives identical transmission.

        For a single-component dust (old stars only, weight=0):
            diffsed: T = exp(-tau_v2 * (lam/5500)^n)
            bagpipes: T = 10^(-A_V * A_curve / 2.5)

        With A_V = tau_V * 2.5/ln(10), these should be identical.
        """
        wavs = optical_wavelengths
        n_slope = -0.7
        tau_v = 0.5

        # diffsed transmission for old stars
        ages_old = np.array([1e10])  # 10 Gyr >> t_birth
        trans_diffsed = np.asarray(
            two_component_dust(
                jnp.array(wavs),
                jnp.array(ages_old),
                tau_v1=0.0,  # no birth cloud
                tau_v2=tau_v,
                law_bc="power_law",
                law_diff="power_law",
                n_slope=n_slope,
            )
        )[0]  # shape (1, n_wave) -> (n_wave,)

        # bagpipes: A_V = tau_V * 2.5 / ln(10) = tau_V * 1.0857
        a_v = tau_v * 2.5 / np.log(10.0)
        bp_dust = bagpipes_dust.dust_attenuation(wavs, {"type": "CF00", "n": -n_slope})
        trans_bagpipes = 10.0 ** (-a_v * bp_dust.A_cont / 2.5)

        np.testing.assert_allclose(
            trans_diffsed,
            trans_bagpipes,
            rtol=1e-4,
            err_msg="Transmission mismatch after tau<->A_V mapping",
        )

    @pytest.mark.parametrize("tau_v2", [0.1, 0.5, 1.0, 2.0])
    def test_transmission_range_consistency(self, optical_wavelengths, tau_v2):
        """Both codes should produce T in (0, 1] for physical tau values."""
        wavs = optical_wavelengths
        n_slope = -0.7

        # diffsed (old stars)
        ages_old = np.array([1e10])
        trans_ds = np.asarray(
            two_component_dust(
                jnp.array(wavs),
                jnp.array(ages_old),
                0.0,
                tau_v2,
                law_bc="power_law",
                law_diff="power_law",
                n_slope=n_slope,
            )
        )[0]

        # bagpipes
        a_v = tau_v2 * 2.5 / np.log(10.0)
        bp_dust = bagpipes_dust.dust_attenuation(wavs, {"type": "CF00", "n": -n_slope})
        trans_bp = 10.0 ** (-a_v * bp_dust.A_cont / 2.5)

        # Both in (0, 1]
        assert np.all(trans_ds > 0) and np.all(trans_ds <= 1.0)
        assert np.all(trans_bp > 0) and np.all(trans_bp <= 1.0)

        # Blue more attenuated than red in both
        assert trans_ds[0] < trans_ds[-1], "diffsed: blue should be more attenuated"
        assert trans_bp[0] < trans_bp[-1], "bagpipes: blue should be more attenuated"
