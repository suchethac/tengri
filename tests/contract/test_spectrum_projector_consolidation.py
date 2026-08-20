# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for spectrum projection consolidation.

Verifies that the consolidated `project_spectrum` function replaces four
previously-duplicated compose patterns:

1. Observation.observe_spectrum
2. Observation.predict (spectroscopy branch)
3. SEDModel._predict_spectrum_on_grid
4. (simulate.spectrum_from_sfh uses velocity_broaden, not apply_lsf — left alone)

Taxonomy marker: contract (consolidation of public seams).
"""

from __future__ import annotations

import chex
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import Fixed, SEDModel
from tengri.cosmology import luminosity_distance
from tengri.observation import Observation, Photometry, Spectroscopy
from tengri.observation.photometry import FilterCurve
from tengri.observation.spectrum import apply_lsf, compute_spectrum, project_spectrum

pytestmark = pytest.mark.contract


@pytest.fixture(scope="module")
def spectrum_inputs(synthetic_ssp_wide):
    """Prepare test inputs: SED, rest wavelengths, observed wavelengths, z, dl."""
    # Create synthetic SED and wavelength grids
    wave_rest = np.linspace(1000, 10000, 500)
    sed_rest = np.exp(-((np.log10(wave_rest) - 3.5) ** 2) / 0.5) + 0.1 * np.exp(
        -((np.log10(wave_rest) - 4.0) ** 2) / 0.3
    )

    # Create a small observed-wavelength grid
    wave_obs = np.linspace(4000, 8000, 200)

    z = 0.1
    dl_cm = 1.0e27  # arbitrary

    return {
        "sed_rest": jnp.asarray(sed_rest),
        "wave_rest": jnp.asarray(wave_rest),
        "wave_obs": jnp.asarray(wave_obs),
        "z": z,
        "dl_cm": dl_cm,
    }


class TestProjectSpectrumDecomposition:
    """Test 1: project_spectrum == compute_spectrum + apply_lsf (manually composed)."""

    def test_project_spectrum_no_resolution_equals_compute_spectrum(self, spectrum_inputs):
        """resolution=None should match compute_spectrum exactly."""
        flux_direct = compute_spectrum(
            spectrum_inputs["sed_rest"],
            spectrum_inputs["wave_rest"],
            spectrum_inputs["wave_obs"],
            spectrum_inputs["z"],
            spectrum_inputs["dl_cm"],
        )

        flux_via_project = project_spectrum(
            spectrum_inputs["sed_rest"],
            spectrum_inputs["wave_rest"],
            spectrum_inputs["wave_obs"],
            spectrum_inputs["z"],
            spectrum_inputs["dl_cm"],
            resolution=None,
        )

        chex.assert_trees_all_close(flux_direct, flux_via_project, rtol=0, atol=0)

    def test_project_spectrum_scalar_resolution(self, spectrum_inputs):
        """Scalar R: project_spectrum should match manual compose."""
        resolution = 100.0
        sigma_lib_kms = 70.0

        flux_compute = compute_spectrum(
            spectrum_inputs["sed_rest"],
            spectrum_inputs["wave_rest"],
            spectrum_inputs["wave_obs"],
            spectrum_inputs["z"],
            spectrum_inputs["dl_cm"],
        )
        flux_manual = apply_lsf(
            flux_compute,
            spectrum_inputs["wave_obs"],
            resolution,
            sigma_lib_kms=sigma_lib_kms,
        )

        flux_via_project = project_spectrum(
            spectrum_inputs["sed_rest"],
            spectrum_inputs["wave_rest"],
            spectrum_inputs["wave_obs"],
            spectrum_inputs["z"],
            spectrum_inputs["dl_cm"],
            resolution=resolution,
            sigma_lib_kms=sigma_lib_kms,
        )

        chex.assert_trees_all_close(flux_manual, flux_via_project, rtol=0, atol=0)

    def test_project_spectrum_array_resolution(self, spectrum_inputs):
        """Array R: project_spectrum should match manual compose."""
        wave_um = spectrum_inputs["wave_obs"] / 1e4
        resolution = 30.0 + 55.0 * (wave_um - 0.6)  # NIRSPEC PRISM-like
        resolution = jnp.clip(resolution, 30.0, 330.0)

        flux_compute = compute_spectrum(
            spectrum_inputs["sed_rest"],
            spectrum_inputs["wave_rest"],
            spectrum_inputs["wave_obs"],
            spectrum_inputs["z"],
            spectrum_inputs["dl_cm"],
        )
        flux_manual = apply_lsf(
            flux_compute,
            spectrum_inputs["wave_obs"],
            resolution,
            sigma_lib_kms=70.0,
            n_bins=16,
        )

        flux_via_project = project_spectrum(
            spectrum_inputs["sed_rest"],
            spectrum_inputs["wave_rest"],
            spectrum_inputs["wave_obs"],
            spectrum_inputs["z"],
            spectrum_inputs["dl_cm"],
            resolution=resolution,
            sigma_lib_kms=70.0,
            n_bins=16,
        )

        chex.assert_trees_all_close(flux_manual, flux_via_project, rtol=0, atol=0)

    def test_project_spectrum_with_sigma_v(self, spectrum_inputs):
        """With intrinsic velocity dispersion."""
        resolution = 100.0
        sigma_v_kms = 200.0

        flux_compute = compute_spectrum(
            spectrum_inputs["sed_rest"],
            spectrum_inputs["wave_rest"],
            spectrum_inputs["wave_obs"],
            spectrum_inputs["z"],
            spectrum_inputs["dl_cm"],
        )
        flux_manual = apply_lsf(
            flux_compute,
            spectrum_inputs["wave_obs"],
            resolution,
            sigma_lib_kms=0.0,
            sigma_v_kms=sigma_v_kms,
        )

        flux_via_project = project_spectrum(
            spectrum_inputs["sed_rest"],
            spectrum_inputs["wave_rest"],
            spectrum_inputs["wave_obs"],
            spectrum_inputs["z"],
            spectrum_inputs["dl_cm"],
            resolution=resolution,
            sigma_v_kms=sigma_v_kms,
        )

        chex.assert_trees_all_close(flux_manual, flux_via_project, rtol=0, atol=0)


class TestPublicSurfaceAgreement:
    """Test 2: model.predict_spectrum == model.predict_observables.spec_fnu."""

    def test_predict_spectrum_agrees_with_predict_observables(self, synthetic_ssp_wide):
        """Joint model: predict_spectrum matches predict_observables.spec_fnu.

        Pre-existing float64 ULP divergence (~4e-16 max relative difference)
        from the predict_obs_sed wavelength round-trip (wavelength*(1+z)/(1+z))
        vs direct rest-frame path in Observation.predict. This divergence is
        measured identical before and after consolidation (#1044), so we assert
        rtol=1e-12 (well above ULP noise) not rtol=0.
        """

        # Build synthetic tophat photometry (5 filters)
        def _tophat(center, frac=0.16, n=40):
            wave = jnp.linspace(center * (1.0 - frac), center * (1.0 + frac), n)
            trans = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
            return FilterCurve(wave=wave, trans=trans, name=f"b{int(center)}")

        phot_curves = tuple(_tophat(c) for c in (3500.0, 4800.0, 6200.0, 7600.0, 9000.0))
        photometry = Photometry(filters=phot_curves)

        # Add spectroscopy: constant R=800
        wave_obs_spec = jnp.linspace(4000.0, 8000.0, 120)
        spectroscopy = Spectroscopy(wave_obs=wave_obs_spec, resolution=800.0)

        # Joint observation
        obs = Observation(photometry=photometry, spectroscopy=spectroscopy)

        # Build joint model
        model = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=obs,
            sfh={"type": "dpl"},
            dust={
                "type": "two_component",
                "law": "calzetti",
                "tau_bc": 0.3,
                "tau_diff": 0.1,
            },
            neb={"type": "none"},
            redshift=Fixed(0.1),
        )

        params = {}

        # Path 1: predict_spectrum
        spec_via_predict_spectrum = model.predict_spectrum(params, wave_obs=wave_obs_spec)

        # Path 2: predict_observables
        spec_via_predict_obs = model.predict_observables(params).spec_fnu

        # Assert close agreement at float64 ULP level
        # rtol=1e-12 is well above the ~4e-16 ULP noise from predict_obs_sed
        # wavelength round-trip (pre-existing, identical before/after #1044).
        chex.assert_trees_all_close(
            spec_via_predict_spectrum, spec_via_predict_obs, rtol=1e-12, atol=0
        )


class TestIntegrationViaForwardModel:
    """Test 3: Verify project_spectrum delegation in _predict_spectrum_on_grid."""

    def test_predict_spectrum_on_grid_matches_manual_projection(self, synthetic_ssp_wide):
        """_predict_spectrum_on_grid bit-exact equals manual project_spectrum call.

        Bit-exactness against the LSF-bearing manual composition also pins
        that the LSF is applied on this path (a skipped LSF would not match).
        """

        # Build synthetic tophat photometry (5 filters)
        def _tophat(center, frac=0.16, n=40):
            wave = jnp.linspace(center * (1.0 - frac), center * (1.0 + frac), n)
            trans = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
            return FilterCurve(wave=wave, trans=trans, name=f"b{int(center)}")

        phot_curves = tuple(_tophat(c) for c in (3500.0, 4800.0, 6200.0, 7600.0, 9000.0))
        photometry = Photometry(filters=phot_curves)

        # Add spectroscopy: constant R=800
        wave_obs_spec = jnp.linspace(4000.0, 8000.0, 120)
        spectroscopy = Spectroscopy(wave_obs=wave_obs_spec, resolution=800.0)

        # Joint observation
        obs = Observation(photometry=photometry, spectroscopy=spectroscopy)

        # Build joint model
        model = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=obs,
            sfh={"type": "dpl"},
            dust={
                "type": "two_component",
                "law": "calzetti",
                "tau_bc": 0.3,
                "tau_diff": 0.1,
            },
            neb={"type": "none"},
            redshift=Fixed(0.1),
        )

        params = {}
        z = 0.1
        dl_cm = jnp.asarray(luminosity_distance(z)).reshape(())

        # Get the observed-frame SED
        sed_obs = model.predict_obs_sed(params)
        wave_rest = sed_obs.wavelength / (1.0 + z)

        # Manual composition using project_spectrum
        flux_manual = project_spectrum(
            sed_obs.sed,
            wave_rest,
            wave_obs_spec,
            z,
            dl_cm,
            resolution=spectroscopy.resolution,
            sigma_lib_kms=spectroscopy.sigma_lib_kms,
            sigma_v_kms=0.0,
        )

        # Via _predict_spectrum_on_grid (the delegation target)
        flux_via_grid_method = model._predict_spectrum_on_grid(params, wave_obs_spec)

        # Assert bit-exact equality (rtol=0, atol=0)
        # This verifies that project_spectrum is correctly delegated
        # and that the spectrum projection seam is consolidated.
        chex.assert_trees_all_close(flux_manual, flux_via_grid_method, rtol=0, atol=0)
