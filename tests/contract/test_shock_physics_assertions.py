# SPDX-License-Identifier: BSD-3-Clause
"""
Physics-level assertions for shock dust attenuation (#1434).

These tests verify the PHYSICS of shock attenuation, not cross-path agreement:
1. Attenuated form is less than intrinsic (dust screen reduces flux)
2. Band-dependent dimming ratios follow Calzetti k(λ) ordering (g < r < i)
"""

from functools import lru_cache

import numpy as np
import pytest

pytestmark = pytest.mark.contract


@lru_cache(maxsize=1)
def _build_shock_dust_model():
    """Build shock+two_component model once, reuse to avoid SSP download overhead."""
    from tengri import FIXED, Fixed, Observation, Photometry, SEDModel, WavePrecomp
    from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
    from tengri.data import download_ssp

    ssp_path = download_ssp("fsps_prsc_miles_chabrier")
    ssp = load_ssp_data(str(ssp_path))

    model = SEDModel.build(
        ssp_data=ssp,
        observation=Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r", "sdss_i"])),
        sfh={
            "type": "delayed",
            "all_params": FIXED,
            "tau_gyr": 1.0,
            "log_total_mass": 10.0,
        },
        dust={
            "law": "power_law",
            "type": "two_component",
            "all_params": FIXED,
            "tau_bc": 2.0,
            "tau_diff": 1.0,
        },
        neb={"type": "none"},
        shock={"frac": 1.0, "all_params": FIXED},
        redshift=Fixed(0.5),
        approx=WavePrecomp(),
    )
    return model


@pytest.mark.regression_bug
class TestShockAttenuationPhysics:
    """Shock attenuation obeys physical laws (#1434)."""

    @pytest.mark.unit
    def test_attenuated_shock_less_than_intrinsic(self):
        """
        Attenuated shock photometry < intrinsic per band (#1434).

        Assertion: dust screen reduces flux. For every filter,
        shock_phot_lnu_attenuated_precomp[b] < shock_phot_lnu_precomp[b].

        This is a sanity check: if attenuation multiplies by factors < 1,
        the product must be smaller. Fails if dust component is not
        publishing the attenuated form (silently returns zeros, or
        returns unattenuated form).
        """
        from tengri import FIXED, Fixed, Observation, Photometry, SEDModel, WavePrecomp
        from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
        from tengri.data import download_ssp

        ssp_path = download_ssp("fsps_prsc_miles_chabrier")
        ssp = load_ssp_data(str(ssp_path))

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
            dust={
                "law": "power_law",
                "type": "two_component",
                "all_params": FIXED,
                "tau_bc": 2.0,
                "tau_diff": 1.0,
            },
            neb={"type": "none"},
            shock={"frac": 1.0, "all_params": FIXED},
            redshift=Fixed(0.5),
            approx=WavePrecomp(),
        )

        params = {}
        state = model.predict_state(params)

        shock_intrinsic = state.derived.get("shock_phot_lnu_precomp")
        shock_attenuated = state.derived.get("shock_phot_lnu_attenuated_precomp")

        assert shock_intrinsic is not None, (
            "shock_phot_lnu_precomp not in state.derived (shock component missing?)"
        )
        assert shock_attenuated is not None, (
            "shock_phot_lnu_attenuated_precomp not in state.derived (dust publication failed?)"
        )

        # Per-band check: attenuated < intrinsic
        ratio = np.asarray(shock_attenuated) / np.maximum(np.asarray(shock_intrinsic), 1e-99)
        assert np.all(ratio < 1.0), (
            f"Dust screen must attenuate shock (multiply by factors ≤ 1). "
            f"Got ratios (attenuated/intrinsic) = {ratio}. "
            f"If all ratios ≈ 1, attenuation is not being applied. "
            f"If ratios > 1, the 'attenuated' form is actually amplified."
        )

    @pytest.mark.unit
    def test_band_dependent_dimming_ratio_ordering(self):
        """
        Per-band dimming ratios r_b must be band-dependent, ordered g < r < i (#1434).

        Calzetti attenuation law: k(λ) decreases with wavelength. Shorter wavelength
        (g-band) has steeper attenuation; longer wavelength (i-band) has gentler
        attenuation. The dimming ratio (how much flux survives) should therefore
        be lowest in g (most dimming), highest in i (least dimming).

        Measurement: r_b = delta_phot(tau_bc=2, tau_diff=1) / delta_phot(tau=0)
        where delta_phot = photometry_with_shock - photometry_without_shock.

        Pre-fix (broken code): both paths unattenuated, so r_b = 1.0 for all bands
        (no dimming, ratio = 1.0 everywhere).

        Post-fix: r_b must satisfy r_g < r_r < r_i and minimum spread > 5%.
        """
        from tengri import FIXED, Fixed, Observation, Photometry, SEDModel
        from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
        from tengri.data import download_ssp

        ssp_path = download_ssp("fsps_prsc_miles_chabrier")
        ssp = load_ssp_data(str(ssp_path))

        # Two models: tau=0 (no attenuation) and tau=2/1 (full attenuation)
        model_notau = SEDModel.build(
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
            dust={
                "law": "power_law",
                "type": "two_component",
                "all_params": FIXED,
                "tau_bc": 0.0,
                "tau_diff": 0.0,
            },
            neb={"type": "none"},
            shock={"frac": 1.0, "all_params": FIXED},
            redshift=Fixed(0.05),
        )

        model_withtau = SEDModel.build(
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
            dust={
                "law": "power_law",
                "type": "two_component",
                "all_params": FIXED,
                "tau_bc": 2.0,
                "tau_diff": 1.0,
            },
            neb={"type": "none"},
            shock={"frac": 1.0, "all_params": FIXED},
            redshift=Fixed(0.05),
        )

        params = {}
        params_no_shock = {"shock_frac": 0.0}

        # Measure shock deltas
        phot_withtau_with_shock = model_withtau.predict_photometry(params)
        phot_withtau_no_shock = model_withtau.predict_photometry(params_no_shock)
        delta_withtau = phot_withtau_with_shock - phot_withtau_no_shock

        phot_notau_with_shock = model_notau.predict_photometry(params)
        phot_notau_no_shock = model_notau.predict_photometry(params_no_shock)
        delta_notau = phot_notau_with_shock - phot_notau_no_shock

        # Dimming ratios: how much shock contributes with attenuation vs without
        r_bands = np.asarray(delta_withtau) / np.maximum(np.asarray(delta_notau), 1e-99)

        # Extract per-band (g, r, i)
        r_g, r_r, r_i = r_bands[0], r_bands[1], r_bands[2]

        # Physics assertions
        msg_ordering = (
            f"Band-dependent dimming ratios must obey Calzetti k(λ) ordering: "
            f"r_g < r_r < r_i (shorter wavelength attenuates more). "
            f"Got r_g={r_g:.4f}, r_r={r_r:.4f}, r_i={r_i:.4f}. "
            f"If all ratios ≈ 1.0, shock attenuation is not working (pre-fix bug). "
            f"If ordering is wrong, dust screen is not band-dependent."
        )
        assert r_g < r_r, msg_ordering
        assert r_r < r_i, msg_ordering

        # Spread check: minimum 5% variation between g and i
        spread = (r_i - r_g) / r_g if r_g > 0 else 0.0
        min_spread = 0.05
        msg_spread = (
            f"Band dependence too weak. Calzetti curve should produce >5% spread. "
            f"Got {spread:.1%} spread from g to i. "
            f"If spread < 1%, attenuation factors may be constant across bands."
        )
        assert spread >= min_spread, msg_spread

    @pytest.mark.unit
    def test_missing_attenuated_form_raises_gate(self):
        """
        Structural gate: if two-component dust is active but
        shock_phot_lnu_attenuated_precomp is missing, raise KeyError (#1434).

        This test verifies the gate can fire. The gate exists to catch silent
        failures: if dust is active and shock exists but the attenuated form
        is absent (e.g., due to a key rename or publication bug), the exact
        and precomp paths would silently disagree (shock unattenuated in precomp,
        attenuated in exact). The gate raises immediately, making the failure
        loud and discoverable.

        Pre-vacuity checks: shock_phot_lnu_precomp and dust_bc_attenuation_precomp
        must both be present in the undoctored state (proving the gate arms are ready).
        Control: undoctored state completes without raising (gate doesn't fire).
        Test: doctored state (missing attenuated form) raises KeyError with #1434.
        """
        model = _build_shock_dust_model()
        params = {}

        # Get the real state with all keys intact
        state = model.predict_state(params)

        # Non-vacuity preconditions: verify gate arms are armed
        assert state.derived.get("shock_phot_lnu_precomp") is not None, (
            "Precondition failed: shock_phot_lnu_precomp missing. "
            "Either shock not in model or prediction failed."
        )
        assert state.derived.get("dust_bc_attenuation_precomp") is not None, (
            "Precondition failed: dust_bc_attenuation_precomp missing. "
            "Either dust not in model or prediction failed."
        )

        # Control: undoctored state should complete without error
        obs = model.observation
        # predict_via_precomp requires full params including Fixed values (e.g., redshift)
        full_params = {**model.spec.get_fixed_values(), **params}
        phot_undoctored = obs.predict_via_precomp(state, full_params)  # Should not raise
        assert phot_undoctored is not None, "Control failed: undoctored predict returned None"

        # Doctor the state: remove shock_phot_lnu_attenuated_precomp
        # DerivedState.from_dict REPLACES (doesn't merge), so we must
        # build the doctored form explicitly using with_()
        doctored_derived = state.derived.with_(shock_phot_lnu_attenuated_precomp=None)
        bad_state = state.with_(derived=doctored_derived)

        # Verify the doctoring worked
        assert bad_state.derived.get("shock_phot_lnu_attenuated_precomp") is None
        assert bad_state.derived.get("shock_phot_lnu_precomp") is not None  # Still armed

        # Test: doctored state must raise KeyError with #1434 in the message
        with pytest.raises(KeyError, match="1434"):
            obs.predict_via_precomp(bad_state, full_params)

    @pytest.mark.unit
    def test_single_component_dust_shock_nonzero(self):
        """
        Single-component dust + shock: precomp delta must be nonzero (#1434).

        Regression: after unified seam fix, single-component dust was broken.
        The precomp path unconditionally subtracted shock from unattenuated_phi
        but only added it back in two-component branch, losing shock entirely.

        Fix: restore legacy behavior for single-component — include shock in
        nebular_phi_for_dust so it gets screened via a_lut * shock at λ_eff.

        Assertion: precomp shock delta is nonzero and scales roughly with exact
        (within 3x measured λ_eff vs exact band-integration overhead).
        """
        from tengri import FIXED, Fixed, Observation, Photometry, SEDModel, WavePrecomp
        from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
        from tengri.data import download_ssp

        ssp_path = download_ssp("fsps_prsc_miles_chabrier")
        ssp = load_ssp_data(str(ssp_path))

        # Two models: exact and precomp with SINGLE-component dust
        model_exact = SEDModel.build(
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
            dust={
                "law": "power_law",
                "type": "single_component",
                "all_params": FIXED,
                "tau_v": 1.0,
            },
            neb={"type": "none"},
            shock={"frac": 1.0, "all_params": FIXED},
            redshift=Fixed(0.5),
            approx=None,
        )

        model_precomp = SEDModel.build(
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
            dust={
                "law": "power_law",
                "type": "single_component",
                "all_params": FIXED,
                "tau_v": 1.0,
            },
            neb={"type": "none"},
            shock={"frac": 1.0, "all_params": FIXED},
            redshift=Fixed(0.5),
            approx=WavePrecomp(),
        )

        params = {}
        params_no_shock = {"shock_frac": 0.0}

        # Compute deltas
        phot_exact_with = model_exact.predict_photometry(params)
        phot_exact_without = model_exact.predict_photometry(params_no_shock)
        delta_exact = phot_exact_with - phot_exact_without

        phot_precomp_with = model_precomp.predict_photometry(params)
        phot_precomp_without = model_precomp.predict_photometry(params_no_shock)
        delta_precomp = phot_precomp_with - phot_precomp_without

        # Precondition: shocks must be nonzero (non-vacuity)
        assert np.any(delta_exact != 0), "Exact shock delta is zero (test is vacuous)"
        assert np.any(delta_precomp != 0), (
            "Precomp shock delta is ZERO — single-component dust fix broken. "
            "Shock should be included in nebular_phi_for_dust when attenuated form "
            "is not available."
        )

        # Assertion: precomp and exact deltas should scale roughly the same.
        # Single-component λ_eff screen vs exact band-integration: ~3x overhead.
        ratio = np.abs(delta_precomp) / np.maximum(np.abs(delta_exact), 1e-40)
        max_ratio = np.max(ratio)
        msg = (
            f"Single-component dust shock: precomp/exact ratio is {max_ratio:.2f}. "
            f"Expected ~1 (both paths screen at λ_eff for single-component). "
            f"Ratio > 3: drift or double-counting. Ratio ≈ 0: shock suppressed."
        )
        assert 0.3 < max_ratio < 3.0, msg
