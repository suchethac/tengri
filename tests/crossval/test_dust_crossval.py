"""Cross-validate dust attenuation and emission against published references.

Part 1 (bagpipes): Charlot & Fall (2000) power-law curve vs bagpipes.
Part 2 (reference): Calzetti+2000, CCM89, Pei92, KC13, CF00 physics,
    WG00 geometries, energy balance, Casey 2012 FIR peak.

Part 2 tests require NO external dependencies (pure reference values).

Usage:
    pytest -m crossval tests/crossval/test_dust_crossval.py -v
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

pytestmark = pytest.mark.crossval

from tengri.models.dust.attenuation import (
    calzetti,
    cardelli,
    kriek_conroy,
    lmc,
    smc,
    two_component_dust,
    wg00_cloudy,
    wg00_dusty,
    wg00_shell,
)
from tengri.models.dust.emission import (
    casey2012,
    compute_absorbed_luminosity,
    modified_blackbody,
)

# bagpipes is optional — only Part 1 (TestDustCurveCrossval) requires it
try:
    import bagpipes.models.dust_attenuation_model as bagpipes_dust

    HAS_BAGPIPES = True
except ImportError:
    HAS_BAGPIPES = False
    bagpipes_dust = None  # type: ignore[assignment]

_skip_no_bagpipes = pytest.mark.skipif(
    not HAS_BAGPIPES, reason="bagpipes not installed"
)


@_skip_no_bagpipes
class TestDustCurveCrossval:
    """Compare power-law attenuation curve shape (requires bagpipes)."""

    @pytest.mark.parametrize("n_slope", [-0.7, -1.0, -1.3])
    def test_power_law_shape_matches(self, optical_wavelengths, n_slope):
        """The wavelength dependence (lambda/5500)^n should be identical.

        bagpipes CF00 returns A(lambda)/A_V = (5500/lambda)^n,
        tengri uses (lambda/5500)^n in the exponent. Since
        (5500/lambda)^n = (lambda/5500)^(-n), we need to be careful
        with sign conventions.
        """
        wavs = optical_wavelengths

        # bagpipes CF00 curve: A(lam)/A_V
        bp_dust = bagpipes_dust.dust_attenuation(wavs, {"type": "CF00", "n": -n_slope})
        a_curve_bp = bp_dust.A_cont  # A(lam)/A_V = (5500/lam)^n_bp

        # tengri: for old stars (weight=0), transmission = exp(-tau_v2 * (lam/5500)^n)
        # The curve shape is (lam/5500)^n = (5500/lam)^(-n)
        # Mapping: A(lam)/A_V ∝ (5500/lam)^(-n_tengri)
        # tengri n_slope = -0.7 means (lam/5500)^(-0.7) = (5500/lam)^(0.7)
        # bagpipes n = 0.7 means (5500/lam)^(0.7)
        # So: bagpipes n_bp = -n_tengri

        # Build tengri attenuation curve shape for comparison
        curve_tengri = (wavs / 5500.0) ** n_slope  # (lam/5500)^n

        # Normalize both to V-band (5500 A) for shape comparison
        # At 5500A, both should be 1.0
        idx_v = np.argmin(np.abs(wavs - 5500.0))

        ratio_bp = a_curve_bp / a_curve_bp[idx_v]
        ratio_ds = curve_tengri / curve_tengri[idx_v]

        np.testing.assert_allclose(
            ratio_ds,
            ratio_bp,
            rtol=1e-4,
            err_msg=f"Power-law curve shape mismatch for n={n_slope}",
        )

    def test_transmission_mapping_tau_to_av(self, optical_wavelengths):
        """Verify tau_V <-> A_V mapping gives identical transmission.

        For a single-component dust (old stars only, weight=0):
            tengri: T = exp(-tau_v2 * (lam/5500)^n)
            bagpipes: T = 10^(-A_V * A_curve / 2.5)

        With A_V = tau_V * 2.5/ln(10), these should be identical.
        """
        wavs = optical_wavelengths
        n_slope = -0.7
        tau_v = 0.5

        # tengri transmission for old stars
        ages_old = np.array([1e10])  # 10 Gyr >> t_birth
        trans_tengri = np.asarray(
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
            trans_tengri,
            trans_bagpipes,
            rtol=1e-4,
            err_msg="Transmission mismatch after tau<->A_V mapping",
        )

    @pytest.mark.parametrize("tau_v2", [0.1, 0.5, 1.0, 2.0])
    def test_transmission_range_consistency(self, optical_wavelengths, tau_v2):
        """Both codes should produce T in (0, 1] for physical tau values."""
        wavs = optical_wavelengths
        n_slope = -0.7

        # tengri (old stars)
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
        assert trans_ds[0] < trans_ds[-1], "tengri: blue should be more attenuated"
        assert trans_bp[0] < trans_bp[-1], "bagpipes: blue should be more attenuated"
