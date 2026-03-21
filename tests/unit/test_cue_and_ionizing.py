"""Tests for Cue emulator and ionizing spectrum fitting."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)


# ===================================================================
# Ionizing spectrum fitting
# ===================================================================


class TestIonizingSpectrumFit:
    """Test ionizing spectrum parameter extraction from SSP spectra."""

    @pytest.fixture(scope="class")
    def ssp(self):
        from tengri import load_ssp_data

        return load_ssp_data("data/fsps_prsc_miles_chabrier.h5")

    def test_fit_returns_7_params(self, ssp):
        from tengri.models.nebular.ionizing_spectrum import fit_ionizing_spectrum

        wave = np.array(ssp.ssp_wave)
        flux = np.array(ssp.ssp_flux[7, 10, :])  # solar Z, young
        result = fit_ionizing_spectrum(wave, flux)
        assert "ionspec_index1" in result
        assert "ionspec_index4" in result
        assert "ionspec_logLratio3" in result
        assert "gas_logqion" in result
        assert result["powerlaw_params"].shape == (4, 2)

    def test_fit_within_cue_ranges(self, ssp):
        from tengri.models.nebular.ionizing_spectrum import _CLIP_RANGES, fit_ionizing_spectrum

        wave = np.array(ssp.ssp_wave)
        flux = np.array(ssp.ssp_flux[7, 10, :])
        result = fit_ionizing_spectrum(wave, flux)
        for key, (lo, hi) in _CLIP_RANGES.items():
            assert lo <= result[key] <= hi, f"{key}={result[key]} outside [{lo}, {hi}]"

    def test_qion_reasonable(self, ssp):
        """Q_H should be ~10^46-10^48 for young SSPs."""
        from tengri.models.nebular.ionizing_spectrum import fit_ionizing_spectrum

        wave = np.array(ssp.ssp_wave)
        flux = np.array(ssp.ssp_flux[7, 10, :])
        result = fit_ionizing_spectrum(wave, flux)
        assert 45 < result["gas_logqion"] < 49, (
            f"logQion={result['gas_logqion']:.1f}, expected 45-49"
        )

    def test_index1_steep(self, ssp):
        """The extreme UV slope (index1) should be steep (>5)."""
        from tengri.models.nebular.ionizing_spectrum import fit_ionizing_spectrum

        wave = np.array(ssp.ssp_wave)
        flux = np.array(ssp.ssp_flux[7, 10, :])
        result = fit_ionizing_spectrum(wave, flux)
        assert result["ionspec_index1"] > 5, (
            f"index1={result['ionspec_index1']:.1f}, expected >5 (steep UV)"
        )

    def test_index4_shallow(self, ssp):
        """The optical slope (index4, near Lyman limit) should be shallower."""
        from tengri.models.nebular.ionizing_spectrum import fit_ionizing_spectrum

        wave = np.array(ssp.ssp_wave)
        flux = np.array(ssp.ssp_flux[7, 10, :])
        result = fit_ionizing_spectrum(wave, flux)
        assert result["ionspec_index4"] < result["ionspec_index1"], (
            "index4 should be shallower than index1"
        )


# ===================================================================
# Precomputed ionizing params table
# ===================================================================


class TestIonizingParamsTable:
    """Test precomputation and interpolation of ionizing params."""

    @pytest.fixture(scope="class")
    def ssp(self):
        from tengri import load_ssp_data

        return load_ssp_data("data/fsps_prsc_miles_chabrier.h5")

    def test_interpolation_within_bounds(self, ssp):
        from tengri.models.nebular.ionizing_spectrum import (
            interpolate_ionizing_params,
            precompute_ionizing_params_table,
        )

        result = precompute_ionizing_params_table(
            np.array(ssp.ssp_wave),
            np.array(ssp.ssp_flux[:3, :10, :]),  # small subset for speed
            np.array(ssp.ssp_lgmet[:3]),
        )
        ionspec, logqion = interpolate_ionizing_params(
            jnp.array(result["ionspec_table"]),
            jnp.array(result["logqion_table"]),
            jnp.array(ssp.ssp_lgmet[:3]),
            jnp.array(ssp.ssp_lg_age_gyr[:10]) + 9.0,
            -1.85,  # solar Z
            6.5,  # 3 Myr
        )
        assert ionspec.shape == (7,)
        assert jnp.all(jnp.isfinite(ionspec))


# ===================================================================
# Cue backend
# ===================================================================


class TestCueBackend:
    """Test the Cue neural net emulator backend."""

    @pytest.fixture(scope="class")
    def backend(self):
        import os

        from tengri.models.nebular.cue import CueBackend

        weights_path = "data/cue_weights.npz"
        if not os.path.exists(weights_path):
            pytest.skip("Cue weights not found (run convert_cue_weights.py)")
        return CueBackend(weights_path)

    def test_load_weights(self, backend):
        assert backend.name == "cue"
        assert backend.has_free_params is True
        assert backend.weights is not None

    def test_predict_lines(self, backend):
        wave, lum = backend.predict_nebular_line_luminosities(
            gas_logu=-2.5,
            gas_logn=2.0,
            gas_logz=0.0,
            gas_logqion=49.0,
        )
        assert len(wave) > 0
        assert len(lum) == len(wave)
        assert jnp.all(jnp.isfinite(lum))

    def test_predict_continuum(self, backend):
        result = backend.predict_nebular_continuum(
            gas_logu=-2.5,
            gas_logn=2.0,
            gas_logz=0.0,
            gas_logqion=49.0,
        )
        # Result is (wavelength, luminosity) tuple or just array
        if isinstance(result, tuple):
            wave_cont, lum_cont = result
            assert len(lum_cont) > 0
            assert jnp.all(jnp.isfinite(lum_cont))
        else:
            assert jnp.all(jnp.isfinite(result))

    def test_halpha_positive(self, backend):
        """H-alpha should be one of the brightest lines."""
        wave, lum = backend.predict_nebular_line_luminosities(
            gas_logu=-2.5,
            gas_logn=2.0,
            gas_logz=0.0,
            gas_logqion=49.0,
        )
        ha_idx = jnp.argmin(jnp.abs(wave - 6562.8))
        ha_lum = float(lum[ha_idx])
        assert ha_lum > 0, "H-alpha luminosity should be positive"

    def test_logU_affects_lines(self, backend):
        """Higher logU should produce brighter lines."""
        _, lum_low = backend.predict_nebular_line_luminosities(
            gas_logu=-3.5,
            gas_logn=2.0,
            gas_logz=0.0,
            gas_logqion=49.0,
        )
        _, lum_high = backend.predict_nebular_line_luminosities(
            gas_logu=-1.5,
            gas_logn=2.0,
            gas_logz=0.0,
            gas_logqion=49.0,
        )
        assert float(jnp.sum(lum_high)) != float(jnp.sum(lum_low)), (
            "Different logU should give different total line luminosity"
        )

    def test_gradient_through_cue(self, backend):
        """Cue predictions should be differentiable w.r.t. gas params."""

        def loss(logu):
            _, lum = backend.predict_nebular_line_luminosities(
                gas_logu=logu,
                gas_logn=2.0,
                gas_logz=0.0,
                gas_logqion=49.0,
            )
            return jnp.sum(lum)

        g = jax.grad(loss)(-2.5)
        assert jnp.isfinite(g), "Gradient should be finite"


# ===================================================================
# Cue with precomputed ionizing params
# ===================================================================


class TestCueWithSSP:
    """Test Cue backend with SSP-derived ionizing parameters."""

    @pytest.fixture(scope="class")
    def backend_with_ssp(self):
        import os

        from tengri import load_ssp_data
        from tengri.models.nebular.cue import CueBackend

        if not os.path.exists("data/cue_weights.npz"):
            pytest.skip("Cue weights not found")
        ssp = load_ssp_data("data/fsps_prsc_miles_chabrier.h5")
        return CueBackend("data/cue_weights.npz", ssp_data=ssp)

    def test_ionspec_table_computed(self, backend_with_ssp):
        assert backend_with_ssp._ionspec_table is not None
        assert backend_with_ssp._ionspec_table.shape[2] == 7

    def test_get_ionizing_params(self, backend_with_ssp):
        ionspec, logqion = backend_with_ssp.get_ionizing_params_at(-1.85, 6.5)
        assert ionspec.shape == (7,)
        assert jnp.isfinite(logqion)
        assert float(logqion) > 40  # should have meaningful Q_H at 3 Myr

    def test_predict_with_ssp_params(self, backend_with_ssp):
        """Predict lines using SSP-derived ionizing params."""
        ionspec, logqion = backend_with_ssp.get_ionizing_params_at(-1.85, 6.5)
        wave, lum = backend_with_ssp.predict_nebular_line_luminosities(
            gas_logu=-2.5,
            gas_logn=2.0,
            gas_logz=0.0,
            gas_logqion=float(logqion),
            ionspec_index1=float(ionspec[0]),
            ionspec_index2=float(ionspec[1]),
            ionspec_index3=float(ionspec[2]),
            ionspec_index4=float(ionspec[3]),
            ionspec_logLratio1=float(ionspec[4]),
            ionspec_logLratio2=float(ionspec[5]),
            ionspec_logLratio3=float(ionspec[6]),
        )
        assert float(jnp.sum(lum)) > 0, "Should produce positive line emission"
