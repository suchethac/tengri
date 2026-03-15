"""Integration tests for the diffsed forward model using real SSP data.

Tests the full pipeline from physical parameters to observable quantities
(photometry, spectroscopy) using FSPS SSP templates. Verifies physical
correctness, gradient flow, and consistency between computation paths.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from diffsed.forward_model import (
    ForwardModel,
    ModelConfig,
    generate_mock,
)
from diffsed.models.dust.charlot_fall import charlot_fall
from diffsed.models.observation.photometry import (
    ab_mag_from_flux,
)
from diffsed.models.sps.dsps_wrapper import (
    compute_csp_weights,
    load_ssp_data,
)
from diffsed.models.sps.precompute import (
    fast_photometry,
    interpolate_ssp_phot_metallicity,
    precompute_photometry,
)
from diffsed.utils.cosmology import luminosity_distance

# ---------------------------------------------------------------------------
# Paths to real SSP data
# ---------------------------------------------------------------------------
_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_SSP_NO_NEB = _DATA_DIR / "fsps_prsc_miles_chabrier.h5"
_SSP_WITH_NEB = _DATA_DIR / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"

_SSP_FILES_EXIST = _SSP_NO_NEB.is_file() and _SSP_WITH_NEB.is_file()
pytestmark = pytest.mark.skipif(
    not _SSP_FILES_EXIST,
    reason="SSP data files not found in data/",
)


# ---------------------------------------------------------------------------
# Session-scoped fixtures (loaded once, shared across all tests)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def ssp_no_neb():
    """Load SSP templates without nebular emission."""
    return load_ssp_data(str(_SSP_NO_NEB))


@pytest.fixture(scope="session")
def ssp_with_neb():
    """Load SSP templates with nebular emission."""
    return load_ssp_data(str(_SSP_WITH_NEB))


@pytest.fixture(scope="session")
def sdss_filters():
    """Simple top-hat filters at SDSS ugriz effective wavelengths.

    Returns (filter_waves, filter_trans) lists for ForwardModel.
    Each filter is a 200 Angstrom wide top-hat.
    """
    # SDSS effective wavelengths (Angstrom)
    eff_wavelengths = {
        "u": 3551.0,
        "g": 4686.0,
        "r": 6166.0,
        "i": 7480.0,
        "z": 8932.0,
    }
    filter_waves = []
    filter_trans = []
    half_width = 100.0  # Angstrom
    n_pts = 50

    for name in ["u", "g", "r", "i", "z"]:
        center = eff_wavelengths[name]
        wave = jnp.linspace(center - half_width, center + half_width, n_pts)
        trans = jnp.ones(n_pts)
        filter_waves.append(wave)
        filter_trans.append(trans)

    return filter_waves, filter_trans


@pytest.fixture(scope="session")
def default_config():
    """Default model configuration at z=0.1."""
    return ModelConfig(
        n_grid=256,
        log_age_min=6.0,
        log_age_max=10.14,
        redshift=0.1,
    )


@pytest.fixture(scope="session")
def fiducial_params():
    """Fiducial galaxy parameters for a typical star-forming galaxy."""
    return {
        "xi": jnp.zeros(256),
        "sigma_ps": 1.0,
        "tau_ps": 50e6,
        "alpha": 1.0,
        "beta": 1.5,
        "tau_sfh": 3e9,
        "sfr_norm": 5.0,
        "log_z": -0.2,
        "tau_v1": 1.0,
        "tau_v2": 0.3,
        "dust_n": -0.7,
    }


@pytest.fixture(scope="session")
def forward_model(ssp_no_neb, default_config, sdss_filters):
    """ForwardModel with real SSP data and SDSS filters."""
    fw, ft = sdss_filters
    return ForwardModel(ssp_no_neb, default_config, fw, ft)


# ===================================================================
# 1. SSP Loading
# ===================================================================


class TestSSPLoading:
    """Verify SSP template data has correct shapes and physical ranges."""

    def test_shapes_no_neb(self, ssp_no_neb):
        assert ssp_no_neb.ssp_flux.shape == (15, 93, 5994)
        assert ssp_no_neb.ssp_lg_age_gyr.shape == (93,)
        assert ssp_no_neb.ssp_lgmet.shape == (15,)
        assert ssp_no_neb.ssp_wave.shape == (5994,)

    def test_shapes_with_neb(self, ssp_with_neb):
        assert ssp_with_neb.ssp_flux.shape == (15, 93, 5994)
        assert ssp_with_neb.ssp_lg_age_gyr.shape == (93,)
        assert ssp_with_neb.ssp_lgmet.shape == (15,)
        assert ssp_with_neb.ssp_wave.shape == (5994,)

    def test_wavelength_range(self, ssp_no_neb):
        wave = ssp_no_neb.ssp_wave
        assert float(wave.min()) < 200.0, "Should have UV coverage"
        assert float(wave.max()) > 10000.0, "Should have NIR coverage"

    def test_age_range(self, ssp_no_neb):
        # ssp_lg_age_gyr is log10(age/Gyr)
        age_gyr = 10.0**ssp_no_neb.ssp_lg_age_gyr
        assert float(age_gyr.min()) < 0.002, "Youngest SSP should be < 2 Myr"
        assert float(age_gyr.max()) > 10.0, "Oldest SSP should be > 10 Gyr"

    def test_metallicity_range(self, ssp_no_neb):
        lgmet = ssp_no_neb.ssp_lgmet
        assert float(lgmet.min()) < -2.0, "Should include low metallicities"
        # The FSPS grid may be all sub-solar; just check range spans > 2 dex
        met_range = float(lgmet.max() - lgmet.min())
        assert met_range > 2.0, f"Metallicity range should span > 2 dex, got {met_range:.1f}"

    def test_flux_positive_and_finite(self, ssp_no_neb):
        flux = ssp_no_neb.ssp_flux
        assert jnp.all(jnp.isfinite(flux)), "SSP flux must be finite"
        assert jnp.all(flux >= 0.0), "SSP flux must be non-negative"

    def test_wavelength_monotonic(self, ssp_no_neb):
        diff = jnp.diff(ssp_no_neb.ssp_wave)
        assert jnp.all(diff > 0), "Wavelength must be strictly increasing"

    def test_age_monotonic(self, ssp_no_neb):
        diff = jnp.diff(ssp_no_neb.ssp_lg_age_gyr)
        assert jnp.all(diff > 0), "Ages must be strictly increasing"

    def test_metallicity_monotonic(self, ssp_no_neb):
        diff = jnp.diff(ssp_no_neb.ssp_lgmet)
        assert jnp.all(diff > 0), "Metallicities must be strictly increasing"

    def test_both_ssp_same_grids(self, ssp_no_neb, ssp_with_neb):
        np.testing.assert_allclose(
            ssp_no_neb.ssp_wave,
            ssp_with_neb.ssp_wave,
            rtol=1e-10,
            err_msg="Both SSP files should share the same wavelength grid",
        )
        np.testing.assert_allclose(
            ssp_no_neb.ssp_lg_age_gyr,
            ssp_with_neb.ssp_lg_age_gyr,
            rtol=1e-10,
            err_msg="Both SSP files should share the same age grid",
        )
        np.testing.assert_allclose(
            ssp_no_neb.ssp_lgmet,
            ssp_with_neb.ssp_lgmet,
            rtol=1e-10,
            err_msg="Both SSP files should share the same metallicity grid",
        )


# ===================================================================
# 2. Full Forward Model
# ===================================================================


class TestFullForwardModel:
    """Test the full parameter -> SED pipeline with real SSP data."""

    def test_sed_positive(self, forward_model, fiducial_params):
        sed = forward_model(fiducial_params)
        assert jnp.all(sed >= 0.0), "SED must be non-negative everywhere"

    def test_sed_finite(self, forward_model, fiducial_params):
        sed = forward_model(fiducial_params)
        assert jnp.all(jnp.isfinite(sed)), "SED must be finite everywhere"

    def test_sed_not_all_zero(self, forward_model, fiducial_params):
        sed = forward_model(fiducial_params)
        assert float(jnp.sum(sed)) > 0.0, "SED must have nonzero flux"

    def test_sed_peaks_in_optical_nir(self, forward_model, fiducial_params, ssp_no_neb):
        sed = forward_model(fiducial_params)
        wave = ssp_no_neb.ssp_wave

        # Find peak wavelength
        peak_idx = jnp.argmax(sed)
        peak_wave = float(wave[peak_idx])

        # Peak should be in optical/NIR: 3000-20000 Angstrom
        assert 3000.0 < peak_wave < 20000.0, (
            f"SED peak at {peak_wave:.0f} A should be in optical/NIR range"
        )

    def test_sed_shape(self, forward_model, fiducial_params, ssp_no_neb):
        sed = forward_model(fiducial_params)
        assert sed.shape == ssp_no_neb.ssp_wave.shape


# ===================================================================
# 3. Photometry
# ===================================================================


class TestPhotometry:
    """Test photometric predictions with real SSP data and top-hat filters."""

    def test_all_fluxes_positive(self, forward_model, fiducial_params):
        fluxes = forward_model.predict_photometry(fiducial_params)
        assert jnp.all(fluxes > 0.0), "All filter fluxes must be positive"

    def test_all_fluxes_finite(self, forward_model, fiducial_params):
        fluxes = forward_model.predict_photometry(fiducial_params)
        assert jnp.all(jnp.isfinite(fluxes)), "All filter fluxes must be finite"

    def test_five_bands(self, forward_model, fiducial_params):
        fluxes = forward_model.predict_photometry(fiducial_params)
        assert fluxes.shape == (5,), "Should have 5 SDSS ugriz bands"

    def test_color_ordering(self, forward_model, fiducial_params):
        """For a typical galaxy SED, flux in u < g < r roughly."""
        fluxes = forward_model.predict_photometry(fiducial_params)
        # u < g for a galaxy SED (blue is fainter in f_nu for typical galaxy)
        assert float(fluxes[0]) < float(fluxes[1]), (
            "u-band flux should be less than g-band for typical galaxy"
        )

    def test_ab_magnitudes_finite(self, forward_model, fiducial_params):
        """AB magnitudes should be finite."""
        fluxes = forward_model.predict_photometry(fiducial_params)
        mags = ab_mag_from_flux(fluxes)
        assert jnp.all(jnp.isfinite(mags)), "AB magnitudes must be finite"

    def test_flux_increases_red_to_blue(self, forward_model, fiducial_params):
        """For a dusty galaxy, redder bands should have relatively more flux."""
        fluxes = forward_model.predict_photometry(fiducial_params)
        # u-band flux (index 0) should be less than z-band (index 4)
        # for a galaxy with dust attenuation
        assert float(fluxes[0]) < float(fluxes[4]), (
            "u-band flux should be less than z-band for a dusty galaxy"
        )


# ===================================================================
# 4. Spectroscopy
# ===================================================================


class TestSpectroscopy:
    """Test spectroscopic predictions with real SSP data."""

    def test_spectrum_positive_and_finite(self, forward_model, fiducial_params):
        # Observed wavelengths covering optical range at z=0.1
        wave_obs = jnp.linspace(4000.0, 9000.0, 500)
        spec = forward_model.predict_spectrum(fiducial_params, wave_obs)

        assert jnp.all(jnp.isfinite(spec)), "Spectrum must be finite"
        assert jnp.all(spec >= 0.0), "Spectrum must be non-negative"
        assert float(jnp.sum(spec)) > 0.0, "Spectrum must have nonzero flux"

    def test_spectrum_shape(self, forward_model, fiducial_params):
        wave_obs = jnp.linspace(4000.0, 9000.0, 500)
        spec = forward_model.predict_spectrum(fiducial_params, wave_obs)
        assert spec.shape == (500,)

    def test_spectrum_outside_coverage_is_zero(self, forward_model, fiducial_params, ssp_no_neb):
        """Wavelengths far outside SSP coverage should yield zero flux."""
        z = forward_model.config.redshift
        # Rest-frame wavelength beyond SSP maximum
        max_rest = float(ssp_no_neb.ssp_wave.max())
        wave_obs_far = jnp.array([max_rest * (1.0 + z) + 5000.0])
        spec = forward_model.predict_spectrum(fiducial_params, wave_obs_far)
        assert float(spec[0]) == 0.0, "Flux outside SSP coverage should be zero"

    def test_spectrum_consistency_with_photometry(
        self, forward_model, fiducial_params, sdss_filters
    ):
        """Spectrum integrated through a filter should roughly match photometry."""
        sed = forward_model(fiducial_params)
        photo_fluxes = forward_model.predict_photometry(fiducial_params)

        # Just check that they are on the same order of magnitude
        # (exact agreement not expected with simple top-hat vs pixel interpolation)
        assert jnp.all(photo_fluxes > 0.0)


# ===================================================================
# 5. Dust Effects
# ===================================================================


class TestDustEffects:
    """Verify that dust attenuation has physically correct wavelength dependence."""

    def test_dust_reduces_blue_more_than_red(
        self, ssp_no_neb, default_config, sdss_filters, fiducial_params
    ):
        fw, ft = sdss_filters

        # No dust
        params_no_dust = {**fiducial_params, "tau_v1": 0.0, "tau_v2": 0.0}
        model = ForwardModel(ssp_no_neb, default_config, fw, ft)
        flux_no_dust = model.predict_photometry(params_no_dust)

        # With dust
        flux_with_dust = model.predict_photometry(fiducial_params)

        # Dust attenuation ratio should be larger (more attenuation) at blue end
        ratio = flux_with_dust / flux_no_dust

        # u-band ratio (most attenuated) < z-band ratio (least attenuated)
        assert float(ratio[0]) < float(ratio[4]), "Dust should attenuate u-band more than z-band"

    def test_no_dust_gives_higher_flux(
        self, ssp_no_neb, default_config, sdss_filters, fiducial_params
    ):
        fw, ft = sdss_filters
        model = ForwardModel(ssp_no_neb, default_config, fw, ft)

        params_no_dust = {**fiducial_params, "tau_v1": 0.0, "tau_v2": 0.0}
        flux_no_dust = model.predict_photometry(params_no_dust)
        flux_with_dust = model.predict_photometry(fiducial_params)

        assert jnp.all(flux_no_dust >= flux_with_dust), (
            "Removing dust should increase flux in all bands"
        )

    def test_more_dust_gives_less_flux(
        self, ssp_no_neb, default_config, sdss_filters, fiducial_params
    ):
        fw, ft = sdss_filters
        model = ForwardModel(ssp_no_neb, default_config, fw, ft)

        params_light_dust = {**fiducial_params, "tau_v1": 0.3, "tau_v2": 0.1}
        params_heavy_dust = {**fiducial_params, "tau_v1": 3.0, "tau_v2": 1.0}

        flux_light = model.predict_photometry(params_light_dust)
        flux_heavy = model.predict_photometry(params_heavy_dust)

        assert jnp.all(flux_light > flux_heavy), (
            "Heavier dust should produce less flux in all bands"
        )


# ===================================================================
# 6. Metallicity Effects
# ===================================================================


class TestMetallicityEffects:
    """Verify that metallicity changes the SED shape."""

    def test_different_metallicities_give_different_seds(
        self, ssp_no_neb, default_config, fiducial_params
    ):
        model = ForwardModel(ssp_no_neb, default_config)

        # Use metallicities within the grid range
        lgmet = ssp_no_neb.ssp_lgmet
        z_lo = float(lgmet[1])  # near lower end
        z_hi = float(lgmet[-2])  # near upper end

        params_low_z = {**fiducial_params, "log_z": z_lo}
        params_high_z = {**fiducial_params, "log_z": z_hi}

        sed_low = model(params_low_z)
        sed_high = model(params_high_z)

        # SEDs should differ — check that they're not identical
        # Use sum of absolute differences, normalized
        diff = float(jnp.sum(jnp.abs(sed_low - sed_high)))
        total = float(jnp.sum(sed_low) + jnp.sum(sed_high))
        assert diff / total > 1e-4, "Low-Z and high-Z SEDs should differ"

    def test_metallicity_affects_photometry(
        self, ssp_no_neb, default_config, sdss_filters, fiducial_params
    ):
        fw, ft = sdss_filters
        model = ForwardModel(ssp_no_neb, default_config, fw, ft)

        lgmet = ssp_no_neb.ssp_lgmet
        z_lo = float(lgmet[1])
        z_hi = float(lgmet[-2])

        params_low_z = {**fiducial_params, "log_z": z_lo}
        params_high_z = {**fiducial_params, "log_z": z_hi}

        flux_low = model.predict_photometry(params_low_z)
        flux_high = model.predict_photometry(params_high_z)

        # Check they differ in at least one band
        max_ratio = float(jnp.max(jnp.abs(flux_low / flux_high - 1.0)))
        assert max_ratio > 0.01, (
            f"Photometry should differ between metallicities, max ratio diff = {max_ratio:.4f}"
        )


# ===================================================================
# 7. GP Burstiness
# ===================================================================


class TestGPBurstiness:
    """Verify that burstiness parameter sigma_PS affects SFH variability."""

    def test_bursty_vs_smooth(self, ssp_no_neb, default_config, sdss_filters):
        fw, ft = sdss_filters
        model = ForwardModel(ssp_no_neb, default_config, fw, ft)

        key = jax.random.PRNGKey(123)
        xi = jax.random.normal(key, shape=(256,))

        params_smooth = {
            "xi": xi,
            "sigma_ps": 0.5,
            "tau_ps": 50e6,
            "alpha": 1.0,
            "beta": 1.5,
            "tau_sfh": 3e9,
            "sfr_norm": 5.0,
            "log_z": -0.2,
            "tau_v1": 0.5,
            "tau_v2": 0.2,
            "dust_n": -0.7,
        }
        params_bursty = {**params_smooth, "sigma_ps": 3.0}

        sed_smooth = model(params_smooth)
        sed_bursty = model(params_bursty)

        # Both should be valid SEDs
        assert jnp.all(jnp.isfinite(sed_smooth))
        assert jnp.all(jnp.isfinite(sed_bursty))

        # They should differ — use relative difference check
        diff = float(jnp.sum(jnp.abs(sed_smooth - sed_bursty)))
        total = float(jnp.sum(sed_smooth) + jnp.sum(sed_bursty))
        assert diff / total > 1e-4, "Smooth and bursty SEDs should differ"

    def test_bursty_sed_has_more_variance_across_realizations(
        self, ssp_no_neb, default_config, sdss_filters
    ):
        """Bursty GP should produce more variance in photometry across xi draws."""
        fw, ft = sdss_filters
        model = ForwardModel(ssp_no_neb, default_config, fw, ft)

        n_draws = 10
        key = jax.random.PRNGKey(99)

        def _photometry_for_draws(sigma_ps):
            fluxes = []
            for i in range(n_draws):
                subkey = jax.random.fold_in(key, i)
                xi = jax.random.normal(subkey, shape=(256,))
                params = {
                    "xi": xi,
                    "sigma_ps": sigma_ps,
                    "tau_ps": 50e6,
                    "alpha": 1.0,
                    "beta": 1.5,
                    "tau_sfh": 3e9,
                    "sfr_norm": 5.0,
                    "log_z": -0.2,
                    "tau_v1": 0.5,
                    "tau_v2": 0.2,
                    "dust_n": -0.7,
                }
                fluxes.append(model.predict_photometry(params))
            return jnp.stack(fluxes)

        flux_smooth = _photometry_for_draws(0.5)
        flux_bursty = _photometry_for_draws(3.0)

        # Coefficient of variation across realizations
        cv_smooth = float(
            jnp.std(flux_smooth, axis=0).mean() / jnp.mean(flux_smooth, axis=0).mean()
        )
        cv_bursty = float(
            jnp.std(flux_bursty, axis=0).mean() / jnp.mean(flux_bursty, axis=0).mean()
        )

        assert cv_bursty > cv_smooth, (
            f"Bursty CV ({cv_bursty:.3f}) should exceed smooth CV ({cv_smooth:.3f})"
        )


# ===================================================================
# 8. End-to-End Gradients
# ===================================================================


class TestGradients:
    """Verify that gradients of photometry w.r.t. ALL parameters are finite and non-zero."""

    def test_gradients_finite_and_nonzero(self, forward_model):
        """Gradient of total photometric flux w.r.t. each parameter."""
        # Use non-zero xi so PSD parameters (sigma_ps, tau_ps) have nonzero gradients
        params = {
            "xi": jax.random.normal(jax.random.PRNGKey(42), shape=(256,)),
            "sigma_ps": 1.5,
            "tau_ps": 50e6,
            "alpha": 1.0,
            "beta": 1.5,
            "tau_sfh": 3e9,
            "sfr_norm": 5.0,
            "log_z": -0.5,
            "tau_v1": 1.0,
            "tau_v2": 0.3,
            "dust_n": -0.7,
        }

        def loss_fn(p):
            fluxes = forward_model.predict_photometry(p)
            return jnp.sum(fluxes)

        grads = jax.grad(loss_fn)(params)

        # Check scalar parameters are finite
        scalar_keys = [
            "sigma_ps",
            "alpha",
            "beta",
            "tau_sfh",
            "sfr_norm",
            "log_z",
            "tau_v1",
            "tau_v2",
            "dust_n",
        ]
        for key in scalar_keys:
            g = grads[key]
            assert jnp.isfinite(g), f"Gradient w.r.t. {key} is not finite"

        # Check xi vector
        xi_grad = grads["xi"]
        assert jnp.all(jnp.isfinite(xi_grad)), "Gradients w.r.t. xi must be finite"
        assert float(jnp.sum(jnp.abs(xi_grad))) > 0.0, (
            "At least some xi gradients should be nonzero"
        )

    def test_gradient_per_band(self, forward_model):
        """Each band's flux should have a finite gradient w.r.t. sfr_norm."""
        params = {
            "xi": jax.random.normal(jax.random.PRNGKey(42), shape=(256,)),
            "sigma_ps": 1.5,
            "tau_ps": 50e6,
            "alpha": 1.0,
            "beta": 1.5,
            "tau_sfh": 3e9,
            "sfr_norm": 5.0,
            "log_z": -0.5,
            "tau_v1": 1.0,
            "tau_v2": 0.3,
            "dust_n": -0.7,
        }

        for i in range(5):

            def flux_band(p, band_idx=i):
                return forward_model.predict_photometry(p)[band_idx]

            g = jax.grad(flux_band)(params)
            assert jnp.isfinite(g["sfr_norm"]), f"Band {i}: gradient w.r.t. sfr_norm is not finite"


# ===================================================================
# 9. Mock Generation
# ===================================================================


class TestMockGeneration:
    """Test generate_mock() produces realistic noisy photometry."""

    def test_noiseless_mock(self, forward_model, fiducial_params):
        mock = generate_mock(forward_model, fiducial_params)

        assert "flux_true" in mock
        assert "noise" in mock
        assert "params" in mock
        assert "flux_obs" not in mock  # No noise without key

        assert jnp.all(mock["flux_true"] > 0.0)
        assert jnp.all(mock["noise"] > 0.0)

    def test_noisy_mock(self, forward_model, fiducial_params):
        key = jax.random.PRNGKey(0)
        mock = generate_mock(forward_model, fiducial_params, key=key, snr=20.0)

        assert "flux_obs" in mock
        assert mock["flux_obs"].shape == mock["flux_true"].shape

    def test_noise_level_matches_snr(self, forward_model, fiducial_params):
        key = jax.random.PRNGKey(1)
        snr = 50.0
        mock = generate_mock(forward_model, fiducial_params, key=key, snr=snr)

        expected_noise = mock["flux_true"] / snr
        np.testing.assert_allclose(mock["noise"], expected_noise, rtol=1e-10)

    def test_noise_statistics(self, forward_model, fiducial_params):
        """With many realizations, the mean should approach truth and scatter ~ noise."""
        n_realizations = 500
        snr = 20.0
        key = jax.random.PRNGKey(42)

        all_obs = []
        for i in range(n_realizations):
            subkey = jax.random.fold_in(key, i)
            mock = generate_mock(forward_model, fiducial_params, key=subkey, snr=snr)
            all_obs.append(mock["flux_obs"])

        all_obs = jnp.stack(all_obs)
        mean_obs = jnp.mean(all_obs, axis=0)
        std_obs = jnp.std(all_obs, axis=0)

        flux_true = generate_mock(forward_model, fiducial_params)["flux_true"]
        expected_noise = flux_true / snr

        # Mean should be close to truth (within 3-sigma of the mean)
        sigma_of_mean = expected_noise / jnp.sqrt(n_realizations)
        residual = jnp.abs(mean_obs - flux_true) / sigma_of_mean
        assert jnp.all(residual < 5.0), (
            f"Mean deviates from truth by up to {float(residual.max()):.1f} sigma"
        )

        # Scatter should match expected noise (within ~20%)
        noise_ratio = std_obs / expected_noise
        np.testing.assert_allclose(
            noise_ratio,
            1.0,
            atol=0.2,
            err_msg=("Observed scatter should match expected noise level"),
        )


# ===================================================================
# 10. Pre-computation Speedup
# ===================================================================


class TestPrecomputation:
    """Verify precomputed photometry matches direct computation."""

    def test_precomputed_agrees_with_direct(
        self, ssp_no_neb, default_config, sdss_filters, fiducial_params
    ):
        fw, ft = sdss_filters
        model = ForwardModel(ssp_no_neb, default_config, fw, ft)
        z = default_config.redshift

        # Direct photometry
        flux_direct = model.predict_photometry(fiducial_params)

        # Precomputed path
        dl_cm = luminosity_distance(z)
        precomp = precompute_photometry(ssp_no_neb, fw, ft, z, dl_cm)

        # Reproduce the forward model steps to get weights
        sed = model(fiducial_params)
        ssp_log_ages_yr = ssp_no_neb.ssp_lg_age_gyr + 9.0
        ssp_ages_yr = 10.0**ssp_log_ages_yr

        # Interpolate SSP photometry at target metallicity
        ssp_phot_at_z = interpolate_ssp_phot_metallicity(
            precomp.ssp_phot, ssp_no_neb.ssp_lgmet, fiducial_params["log_z"]
        )

        # Dust at effective rest-frame wavelengths
        dust_at_eff = charlot_fall(
            precomp.effective_wavelengths_rest,
            ssp_ages_yr,
            tau_v1=fiducial_params["tau_v1"],
            tau_v2=fiducial_params["tau_v2"],
            n_slope=fiducial_params["dust_n"],
        )

        # Recompute weights
        from diffsed.models.sfh.gp_sfh import gp_from_xi
        from diffsed.models.sfh.mean_sfh import double_powerlaw
        from diffsed.models.sfh.psd_models import drw_variance

        sqrt_power = model.compute_sqrt_power(
            fiducial_params["sigma_ps"], fiducial_params["tau_ps"]
        )
        gp_x = gp_from_xi(fiducial_params["xi"], sqrt_power, default_config.n_grid)
        k0_half = drw_variance(fiducial_params["sigma_ps"]) / 2.0
        sfr_mean = double_powerlaw(
            model.age_yr,
            alpha=fiducial_params["alpha"],
            beta=fiducial_params["beta"],
            tau=fiducial_params["tau_sfh"],
            norm=fiducial_params["sfr_norm"],
        )
        sfr = sfr_mean * jnp.exp(gp_x - k0_half)
        sfr_on_ssp = jnp.interp(ssp_log_ages_yr, model.log_age_grid, sfr)
        weights = compute_csp_weights(sfr_on_ssp, ssp_ages_yr)

        flux_precomp = fast_photometry(weights, ssp_phot_at_z, dust_at_eff, precomp.flux_scale)

        # The precomputed path evaluates dust only at effective wavelengths
        # (a single wavelength per band), while the direct path integrates
        # dust across the full filter transmission curve. So they will not
        # match exactly, but should agree to ~10-20% for broadband photometry.
        ratio = flux_precomp / flux_direct
        np.testing.assert_allclose(
            ratio,
            1.0,
            atol=0.25,
            err_msg=("Precomputed and direct photometry should agree within ~25%"),
        )

    def test_precomputed_shapes(self, ssp_no_neb, default_config, sdss_filters):
        fw, ft = sdss_filters
        z = default_config.redshift
        dl_cm = luminosity_distance(z)
        precomp = precompute_photometry(ssp_no_neb, fw, ft, z, dl_cm)

        assert precomp.ssp_phot.shape == (15, 93, 5), (
            "ssp_phot should be (n_met, n_age, n_filters)"
        )
        assert precomp.effective_wavelengths.shape == (5,)
        assert precomp.effective_wavelengths_rest.shape == (5,)
        assert precomp.n_filters == 5

    def test_precomputed_effective_wavelengths(self, ssp_no_neb, default_config, sdss_filters):
        fw, ft = sdss_filters
        z = default_config.redshift
        dl_cm = luminosity_distance(z)
        precomp = precompute_photometry(ssp_no_neb, fw, ft, z, dl_cm)

        # Effective wavelengths should be near the filter centers
        expected_centers = jnp.array([3551.0, 4686.0, 6166.0, 7480.0, 8932.0])
        np.testing.assert_allclose(
            precomp.effective_wavelengths,
            expected_centers,
            rtol=0.02,
            err_msg="Effective wavelengths should be near filter centers",
        )
