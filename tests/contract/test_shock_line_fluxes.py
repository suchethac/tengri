# SPDX-License-Identifier: BSD-3-Clause
"""
Shock line flux visibility to predict_line_fluxes (#927).

Regression test for #927: shock's line contribution was invisible to
predict_line_fluxes. The shock component:
- Publishes sed_shock (continuous SED with lines baked in)
- Has shock_log_lhalpha parameter (discrete Hα luminosity)
- But never published discrete lines to the catalog that predict_line_fluxes reads

Result: shock_frac varied photometry (sed_shock added to continuum) but had
zero gradient on line-flux fitting (catalog unchanged). Any line-flux fit
was blind to shock—the component was invisible to that channel.

Fix (#927): Shock publishes its discrete line luminosities (Hα, Hβ, etc.)
to the unified line catalog (PR #1877 pattern), using the same published key
path as nebular backend. Now predict_line_fluxes() sees shock's contribution,
and shock_frac affects both photometry AND line flux gradients.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.contract


@pytest.mark.regression_bug
class TestShockLineFluxes:
    """Shock component contributes to emission line fluxes (#927)."""

    @pytest.mark.unit
    @pytest.mark.xfail(
        strict=True, reason="#927: shock discrete lines do not reach predict_line_fluxes"
    )
    def test_shock_scales_halpha_in_line_fluxes(self):
        """
        Shock norm parameter scales Hα in predict_line_fluxes (#927).

        Assertion 1: predict_line_fluxes(Hα) changes with shock_frac
        (with > without).

        Assertion 2: flux change scales monotonically with norm (0 < 0.5 < 1.0
        frac values produce monotonic flux increase).

        Assertion 3: Precondition—shock Hα contribution > noise floor
        (non-vacuity: verifies shock emits and reaches the line catalog).
        """
        from tengri import FIXED, Fixed, Observation, Photometry, SEDModel
        from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
        from tengri.data import download_ssp

        # Use bare stellar SSP; add Cue nebular backend (wNE not in catalog)
        ssp_path = download_ssp("fsps_prsc_miles_chabrier")
        ssp = load_ssp_data(str(ssp_path))

        # Model with Cue nebular (discrete lines) + shock
        model = SEDModel.build(
            ssp_data=ssp,
            observation=Observation(
                photometry=Photometry.from_names(["sdss_g", "sdss_r", "sdss_i"])
            ),
            sfh={
                "type": "delayed",
                "all_params": FIXED,
                "tau_gyr": 1.0,
                "log_total_mass": 10.0,
            },
            dust={"law": "power_law", "type": "two_component", "all_params": FIXED},
            neb={"type": "cue", "all_params": FIXED},  # Discrete line backend
            shock={"frac": 1.0, "all_params": FIXED},  # Start with max shock
            redshift=Fixed(0.1),
        )

        params = {}  # Use defaults (all Fixed in this model)

        # Measure Hα flux with maximum shock
        halpha_wave = 6564.61  # Vacuum wavelength
        line_flux_with_shock = model.predict_line_fluxes(params, target_wavelengths=[halpha_wave])[
            0
        ]

        # Measure Hα flux with no shock
        params_no_shock = {"shock_frac": 0.0}
        line_flux_no_shock = model.predict_line_fluxes(
            params_no_shock, target_wavelengths=[halpha_wave]
        )[0]

        # The difference should be positive (shock adds Hα)
        delta_line_flux = line_flux_with_shock - line_flux_no_shock

        # Precondition: shock's Hα contribution must be above noise floor
        noise_floor = 1e-32  # erg/s/cm^2
        msg = (
            f"Shock Hα contribution ({np.abs(delta_line_flux):.2e}) below noise "
            f"floor ({noise_floor:.2e})"
        )
        assert np.abs(delta_line_flux) > noise_floor, msg

        # Shock should ADD Hα (increase the flux), not decrease it
        assert delta_line_flux > 0, (
            f"Shock should increase Hα flux, but got delta = {delta_line_flux:.3e}. "
            f"with_shock={line_flux_with_shock:.3e}, no_shock={line_flux_no_shock:.3e}"
        )

        # Intermediate test: measure with intermediate shock_frac
        params_half_shock = {"shock_frac": 0.5}
        line_flux_half_shock = model.predict_line_fluxes(
            params_half_shock, target_wavelengths=[halpha_wave]
        )[0]

        # Hα should scale roughly linearly with shock_frac
        # (at least for small shock fractions, the contribution should be monotonic)
        assert line_flux_no_shock < line_flux_half_shock < line_flux_with_shock, (
            f"Hα flux should increase monotonically with shock_frac: "
            f"no_shock={line_flux_no_shock:.3e}, "
            f"half_shock={line_flux_half_shock:.3e}, "
            f"with_shock={line_flux_with_shock:.3e}"
        )

    @pytest.mark.regression_bug
    def test_shock_photoionized_mixed_warning_cue(self):
        """Warning fires when shock is active with Cue backend (#927)."""
        import warnings

        from tengri import FIXED, Fixed, Observation, Photometry, SEDModel
        from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
        from tengri.config.exceptions import ShockPhotoionizedMixedWarning
        from tengri.data import download_ssp

        ssp_path = download_ssp("fsps_prsc_miles_chabrier")
        ssp = load_ssp_data(str(ssp_path))

        model = SEDModel.build(
            ssp_data=ssp,
            observation=Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r"])),
            sfh={"type": "delayed", "all_params": FIXED, "tau_gyr": 1.0, "log_total_mass": 10.0},
            dust={"law": "power_law", "type": "two_component", "all_params": FIXED},
            neb={"type": "cue", "all_params": FIXED},
            shock={"frac": 0.5, "all_params": FIXED},
            redshift=Fixed(0.1),
        )

        params = {}
        halpha_wave = 6564.61

        # Verify warning is emitted
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.predict_line_fluxes(params, target_wavelengths=[halpha_wave])
            # Filter to our specific warning
            shock_warnings = [
                warn for warn in w if issubclass(warn.category, ShockPhotoionizedMixedWarning)
            ]
            assert len(shock_warnings) > 0, "Expected ShockPhotoionizedMixedWarning not raised"
            assert "shock" in str(shock_warnings[0].message).lower()
            assert "predict_line_fluxes" in str(shock_warnings[0].message)

    @pytest.mark.regression_bug
    def test_shock_catalog_invariant_to_shock_frac(self):
        """Shock fraction does NOT change Hα flux in predict_line_fluxes (#927).

        This is the measured symptom: shock_frac varies sed_shock (continuum)
        but does not change the discrete line catalog, proving shock lines are
        excluded from the catalog predict_line_fluxes reads.
        """
        from tengri import FIXED, Fixed, Observation, Photometry, SEDModel
        from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
        from tengri.data import download_ssp

        ssp_path = download_ssp("fsps_prsc_miles_chabrier")
        ssp = load_ssp_data(str(ssp_path))

        model = SEDModel.build(
            ssp_data=ssp,
            observation=Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r"])),
            sfh={"type": "delayed", "all_params": FIXED, "tau_gyr": 1.0, "log_total_mass": 10.0},
            dust={"law": "power_law", "type": "two_component", "all_params": FIXED},
            neb={"type": "cue", "all_params": FIXED},
            shock={"frac": 1.0, "all_params": FIXED},  # Shock is active (but frac can be varied)
            redshift=Fixed(0.1),
        )

        halpha_wave = 6564.61

        # Measure Hα flux with high shock
        flux_high_shock = model.predict_line_fluxes({}, target_wavelengths=[halpha_wave])[0]

        # Measure Hα flux with low shock
        flux_low_shock = model.predict_line_fluxes(
            {"shock_frac": 0.1}, target_wavelengths=[halpha_wave]
        )[0]

        # The catalog values should be **identical** — proof shock is not in the catalog
        # (this is the regression: they currently are identical, as reported in #927)
        np.testing.assert_allclose(
            flux_high_shock,
            flux_low_shock,
            rtol=1e-14,
            err_msg="Shock fraction should NOT affect predict_line_fluxes output (#927)",
        )

    @pytest.mark.regression_bug
    def test_shock_none_no_warning(self):
        """No warning when shock is disabled (#927)."""
        import warnings

        from tengri import FIXED, Fixed, Observation, Photometry, SEDModel
        from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
        from tengri.config.exceptions import ShockPhotoionizedMixedWarning
        from tengri.data import download_ssp

        ssp_path = download_ssp("fsps_prsc_miles_chabrier")
        ssp = load_ssp_data(str(ssp_path))

        # Build model with shock disabled
        model = SEDModel.build(
            ssp_data=ssp,
            observation=Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r"])),
            sfh={"type": "delayed", "all_params": FIXED, "tau_gyr": 1.0, "log_total_mass": 10.0},
            dust={"law": "power_law", "type": "two_component", "all_params": FIXED},
            neb={"type": "cue", "all_params": FIXED},
            shock={"type": "none"},  # Shock explicitly disabled
            redshift=Fixed(0.1),
        )

        params = {}
        halpha_wave = 6564.61

        # Verify warning is NOT emitted
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.predict_line_fluxes(params, target_wavelengths=[halpha_wave])
            shock_warnings = [
                warn for warn in w if issubclass(warn.category, ShockPhotoionizedMixedWarning)
            ]
            msg = "Unexpected ShockPhotoionizedMixedWarning (shock should be disabled)"
            assert len(shock_warnings) == 0, msg
