# SPDX-License-Identifier: BSD-3-Clause
"""Regression: power-law AGN radio jet reads radio_log_nu_cut parameter (#2689).

The default AGN radio model (radio_agn_model="powerlaw") applies a
synchrotron-aging cutoff exp(-nu/nu_cut) with nu_cut hard-wired at 10^13 Hz,
even though radio_log_nu_cut is declared and accepted by the builder. This
leaves the cutoff unreachable on the default model and creates a flat
(unobserved) posterior direction when freed.

Fix: radio_total_terms reads log_nu_cut and passes it to radio_agn; the
component's power-law branch reads params["radio_log_nu_cut"].

Test coverage:
1. Function-level: radio_total_terms with log_nu_cut in {11.0, 13.0, 40.0}
   produces the expected closed-form spectrum.
2. Default identity: with and without log_nu_cut=13.0 match bit-exactly.
3. Through SEDModel.build with radio_log_nu_cut=Fixed (two cases).
4. Default model (no explicit cutoff) matches the old behavior.
5. Float32 at log_nu_cut=40.0 is finite and matches float64.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tengri
from tengri.components.radio.radio import radio_total_terms
from tengri.utils.physics_constants import C_AA as _C_AA_PER_S

pytestmark = pytest.mark.regression_bug


@pytest.fixture(scope="module")
def ssp():
    """Load the SSP data once for the module."""
    return tengri.load_ssp()


class TestRadioPowerlawCutoff:
    """Power-law AGN radio jet reads the cutoff parameter."""

    @pytest.mark.parametrize(
        "log_nu_cut,nu_hz",
        [
            (11.0, 30e9),
            (11.0, 100e9),
            (11.0, 300e9),
            (11.0, 1e12),
            (13.0, 30e9),
            (13.0, 100e9),
            (13.0, 300e9),
            (13.0, 1e12),
            (40.0, 30e9),
            (40.0, 100e9),
            (40.0, 300e9),
            (40.0, 1e12),
        ],
    )
    def test_radio_total_terms_agn_cutoff_closed_form(self, log_nu_cut, nu_hz):
        """AGN term equals closed-form (nu / 5e9)**(-0.7) * exp(-nu / 10^log_nu_cut).

        Uses minimal arguments: l_bband=1.0 (forces direct disc use),
        radio_loudness=0.0 (AGN coefficient), L_ir=0.0 (no SF), alpha_agn=0.7.
        """
        wavelength = _C_AA_PER_S / nu_hz  # C / nu in Angstrom
        wavelength_array = jnp.asarray([wavelength])

        result = radio_total_terms(
            wavelength_array,
            L_ir=0.0,
            L_agn_bol=0.0,
            radio_loudness=0.0,
            alpha_agn=0.7,
            l_bband=1.0,  # Direct disc luminosity in erg/s/Hz at 4400 A
            log_nu_cut=log_nu_cut,
        )

        agn_lnu = float(result["agn"][0])

        # Closed-form expectation: L_nu = (nu / 5e9)**(-0.7) * exp(-nu / 10^log_nu_cut)
        # With l_bband=1.0 and radio_loudness=0.0, L_B=1.0 and L_5GHz=1.0*10^0=1.0
        nu_ref = 5e9
        nu_cut = 10.0**log_nu_cut
        expected = (nu_hz / nu_ref) ** (-0.7) * np.exp(-nu_hz / nu_cut)

        assert np.allclose(agn_lnu, expected, rtol=1e-6), (
            f"log_nu_cut={log_nu_cut}, nu={nu_hz:.2e}: "
            f"got {agn_lnu:.6e}, expected {expected:.6e}, "
            f"rel error {abs(agn_lnu - expected) / expected:.6e}"
        )

    def test_radio_total_terms_default_is_log_nu_cut_13(self):
        """radio_total_terms without log_nu_cut equals the call with log_nu_cut=13.0.

        All returned terms must be bit-exactly equal.
        """
        wavelength = jnp.linspace(1000, 30000, 50)  # Arbitrary rest grid
        kwargs = {
            "L_ir": 0.0,
            "L_agn_bol": 0.0,
            "radio_loudness": 0.0,
            "alpha_agn": 0.7,
            "l_bband": 1.0,
        }

        # Call without log_nu_cut (uses default)
        result_default = radio_total_terms(wavelength, **kwargs)

        # Call with explicit log_nu_cut=13.0
        result_explicit = radio_total_terms(wavelength, log_nu_cut=13.0, **kwargs)

        # All three terms must match bit-exactly
        for key in ("sf", "ff", "agn"):
            assert np.array_equal(result_default[key], result_explicit[key]), (
                f"Term '{key}' differs: "
                f"default {np.max(np.abs(result_default[key]))} vs "
                f"explicit {np.max(np.abs(result_explicit[key]))}"
            )

    def test_sedmodel_build_radio_log_nu_cut_varies_agn(self, ssp):
        """SEDModel with radio_log_nu_cut Fixed produces the expected AGN variation.

        Reproducer from issue #2689: AGN term at 30, 100, 300 GHz and 1 THz
        with cutoff 11.0 and 40.0 equals default * exp(-nu/10^cut) / exp(-nu/1e13).
        """
        m_base = tengri.SEDModel.build(
            ssp_data=ssp,
            met={"logzsol": tengri.Fixed(0.0), "all_params": tengri.Fixed(tengri.DEFAULT)},
            sfh={
                "type": "delayed",
                "tau_gyr": tengri.Fixed(1.0),
                "age_gyr": tengri.Fixed(3.0),
                "log_total_mass": tengri.Fixed(10.0),
                "all_params": tengri.Fixed(tengri.DEFAULT),
            },
            dust_attenuation={
                "type": "single_component",
                "law": "calzetti",
                "all_params": tengri.Fixed(tengri.DEFAULT),
            },
            dust_emission={"type": "none"},
            neb={"type": "none"},
            agn={"type": "composable", "all_params": tengri.Fixed(tengri.DEFAULT)},
            radio={
                "all_params": tengri.Fixed(tengri.DEFAULT),
                "agn": {
                    "all_params": tengri.Fixed(tengri.DEFAULT),
                    "radio_loudness": tengri.Fixed(3.0),
                },
            },
            redshift=tengri.Fixed(0.0),
        )

        st_base = m_base.predict_state({})
        sed_base = np.asarray(st_base.derived["sed_radio"])
        wave = np.asarray(m_base.wave if hasattr(m_base, "wave") else st_base.wave)
        nu = _C_AA_PER_S / wave  # C / wave in Hz

        # Test log_nu_cut=40.0 (no cutoff) relative to default (13.0)
        m_40 = tengri.SEDModel.build(
            ssp_data=ssp,
            met={"logzsol": tengri.Fixed(0.0), "all_params": tengri.Fixed(tengri.DEFAULT)},
            sfh={
                "type": "delayed",
                "tau_gyr": tengri.Fixed(1.0),
                "age_gyr": tengri.Fixed(3.0),
                "log_total_mass": tengri.Fixed(10.0),
                "all_params": tengri.Fixed(tengri.DEFAULT),
            },
            dust_attenuation={
                "type": "single_component",
                "law": "calzetti",
                "all_params": tengri.Fixed(tengri.DEFAULT),
            },
            dust_emission={"type": "none"},
            neb={"type": "none"},
            agn={"type": "composable", "all_params": tengri.Fixed(tengri.DEFAULT)},
            radio={
                "all_params": tengri.Fixed(tengri.DEFAULT),
                "agn": {
                    "all_params": tengri.Fixed(tengri.DEFAULT),
                    "radio_loudness": tengri.Fixed(3.0),
                    "radio_log_nu_cut": tengri.Fixed(40.0),
                },
            },
            redshift=tengri.Fixed(0.0),
        )

        st_40 = m_40.predict_state({})
        sed_40 = np.asarray(st_40.derived["sed_radio"])

        # AGN term = sed_radio(R) - sed_radio(R=-30) where R < -30 has no jet
        m_nojet = tengri.SEDModel.build(
            ssp_data=ssp,
            met={"logzsol": tengri.Fixed(0.0), "all_params": tengri.Fixed(tengri.DEFAULT)},
            sfh={
                "type": "delayed",
                "tau_gyr": tengri.Fixed(1.0),
                "age_gyr": tengri.Fixed(3.0),
                "log_total_mass": tengri.Fixed(10.0),
                "all_params": tengri.Fixed(tengri.DEFAULT),
            },
            dust_attenuation={
                "type": "single_component",
                "law": "calzetti",
                "all_params": tengri.Fixed(tengri.DEFAULT),
            },
            dust_emission={"type": "none"},
            neb={"type": "none"},
            agn={"type": "composable", "all_params": tengri.Fixed(tengri.DEFAULT)},
            radio={
                "all_params": tengri.Fixed(tengri.DEFAULT),
                "agn": {
                    "all_params": tengri.Fixed(tengri.DEFAULT),
                    "radio_loudness": tengri.Fixed(-30.0),
                },
            },
            redshift=tengri.Fixed(0.0),
        )

        st_nojet = m_nojet.predict_state({})
        sed_nojet = np.asarray(st_nojet.derived["sed_radio"])

        agn_base = sed_base - sed_nojet
        agn_40 = sed_40 - sed_nojet

        # Element-wise comparison on model's wavelength grid
        # Select grid points with 30e9 <= nu <= 1e12 and agn_base > 0
        nu_cut_13 = 1e13
        sel_40 = (nu >= 30e9) & (nu <= 1e12) & (agn_base > 0)
        assert np.sum(sel_40) >= 5, (
            f"cut=40: only {np.sum(sel_40)} grid points selected; need >= 5"
        )

        expected_40 = np.exp(-nu[sel_40] / 10.0**40.0) / np.exp(-nu[sel_40] / nu_cut_13)
        ratio_40 = agn_40[sel_40] / agn_base[sel_40]
        np.testing.assert_allclose(ratio_40, expected_40, rtol=1e-5)

        # Test log_nu_cut=11.0 (aggressive cutoff)
        m_11 = tengri.SEDModel.build(
            ssp_data=ssp,
            met={"logzsol": tengri.Fixed(0.0), "all_params": tengri.Fixed(tengri.DEFAULT)},
            sfh={
                "type": "delayed",
                "tau_gyr": tengri.Fixed(1.0),
                "age_gyr": tengri.Fixed(3.0),
                "log_total_mass": tengri.Fixed(10.0),
                "all_params": tengri.Fixed(tengri.DEFAULT),
            },
            dust_attenuation={
                "type": "single_component",
                "law": "calzetti",
                "all_params": tengri.Fixed(tengri.DEFAULT),
            },
            dust_emission={"type": "none"},
            neb={"type": "none"},
            agn={"type": "composable", "all_params": tengri.Fixed(tengri.DEFAULT)},
            radio={
                "all_params": tengri.Fixed(tengri.DEFAULT),
                "agn": {
                    "all_params": tengri.Fixed(tengri.DEFAULT),
                    "radio_loudness": tengri.Fixed(3.0),
                    "radio_log_nu_cut": tengri.Fixed(11.0),
                },
            },
            redshift=tengri.Fixed(0.0),
        )

        st_11 = m_11.predict_state({})
        sed_11 = np.asarray(st_11.derived["sed_radio"])
        agn_11 = sed_11 - sed_nojet

        # For cut=11, at high nu the ratio becomes tiny; restrict to nu <= 300e9
        sel_11 = (nu >= 30e9) & (nu <= 300e9) & (agn_base > 0)
        assert np.sum(sel_11) >= 5, (
            f"cut=11: only {np.sum(sel_11)} grid points selected; need >= 5"
        )

        expected_11 = np.exp(-nu[sel_11] / 10.0**11.0) / np.exp(-nu[sel_11] / nu_cut_13)
        ratio_11 = agn_11[sel_11] / agn_base[sel_11]
        np.testing.assert_allclose(ratio_11, expected_11, rtol=1e-5)

    def test_default_model_unchanged(self, ssp):
        """Default model (no explicit radio_log_nu_cut) matches bit-identical
        behavior: sed_radio with default equals sed_radio with Fixed(13.0)."""
        m_default = tengri.SEDModel.build(
            ssp_data=ssp,
            met={"logzsol": tengri.Fixed(0.0), "all_params": tengri.Fixed(tengri.DEFAULT)},
            sfh={
                "type": "delayed",
                "tau_gyr": tengri.Fixed(1.0),
                "age_gyr": tengri.Fixed(3.0),
                "log_total_mass": tengri.Fixed(10.0),
                "all_params": tengri.Fixed(tengri.DEFAULT),
            },
            dust_attenuation={
                "type": "single_component",
                "law": "calzetti",
                "all_params": tengri.Fixed(tengri.DEFAULT),
            },
            dust_emission={"type": "none"},
            neb={"type": "none"},
            agn={"type": "composable", "all_params": tengri.Fixed(tengri.DEFAULT)},
            radio={"all_params": tengri.Fixed(tengri.DEFAULT)},
            redshift=tengri.Fixed(0.0),
        )

        st_default = m_default.predict_state({})
        sed_default = np.asarray(st_default.derived["sed_radio"])

        m_explicit = tengri.SEDModel.build(
            ssp_data=ssp,
            met={"logzsol": tengri.Fixed(0.0), "all_params": tengri.Fixed(tengri.DEFAULT)},
            sfh={
                "type": "delayed",
                "tau_gyr": tengri.Fixed(1.0),
                "age_gyr": tengri.Fixed(3.0),
                "log_total_mass": tengri.Fixed(10.0),
                "all_params": tengri.Fixed(tengri.DEFAULT),
            },
            dust_attenuation={
                "type": "single_component",
                "law": "calzetti",
                "all_params": tengri.Fixed(tengri.DEFAULT),
            },
            dust_emission={"type": "none"},
            neb={"type": "none"},
            agn={"type": "composable", "all_params": tengri.Fixed(tengri.DEFAULT)},
            radio={
                "all_params": tengri.Fixed(tengri.DEFAULT),
                "agn": {"radio_log_nu_cut": tengri.Fixed(13.0)},
            },
            redshift=tengri.Fixed(0.0),
        )

        st_explicit = m_explicit.predict_state({})
        sed_explicit = np.asarray(st_explicit.derived["sed_radio"])

        assert np.array_equal(sed_default, sed_explicit), (
            f"Default model differs from explicit 13.0:\n"
            f"max diff {np.max(np.abs(sed_default - sed_explicit))}"
        )

    def test_float32_at_cutoff_40_is_finite(self):
        """Float32 at log_nu_cut=40.0 and nu=300 GHz is finite and matches float64."""
        with jax.enable_x64(False):
            wavelength = jnp.asarray([_C_AA_PER_S / 300e9])
            result = radio_total_terms(
                wavelength,
                L_ir=0.0,
                L_agn_bol=0.0,
                radio_loudness=0.0,
                alpha_agn=0.7,
                l_bband=1.0,
                log_nu_cut=40.0,
            )
            agn_f32 = float(result["agn"][0])

        # Also compute float64 for comparison
        wavelength_f64 = jnp.asarray([_C_AA_PER_S / 300e9])
        result_f64 = radio_total_terms(
            wavelength_f64,
            L_ir=0.0,
            L_agn_bol=0.0,
            radio_loudness=0.0,
            alpha_agn=0.7,
            l_bband=1.0,
            log_nu_cut=40.0,
        )
        agn_f64 = float(result_f64["agn"][0])

        # f32 should be finite
        assert np.isfinite(agn_f32), f"f32 result not finite: {agn_f32}"

        # f32 should match f64 to 1e-6 relative
        assert np.allclose(agn_f32, agn_f64, rtol=1e-6), (
            f"f32 vs f64 mismatch at log_nu_cut=40, nu=300 GHz: "
            f"f32={agn_f32:.6e}, f64={agn_f64:.6e}"
        )

    @pytest.mark.parametrize("log_nu_cut", [11.0, 40.0])
    def test_radio_total_honors_cutoff(self, log_nu_cut):
        """radio_total honors the log_nu_cut parameter.

        At nu = 300 GHz, with L_ir=0.0, L_agn_bol=0.0, radio_loudness=0.0,
        alpha_agn=0.7, l_bband=1.0 and include_freefree=False, the returned
        L_nu equals (nu/5e9)**(-0.7) * exp(-nu/10^log_nu_cut) to 1e-6 relative.
        """
        from tengri.components.radio.radio import radio_total

        nu_hz = 300e9
        wavelength = _C_AA_PER_S / nu_hz
        wavelength_array = jnp.asarray([wavelength])

        result = radio_total(
            wavelength_array,
            L_ir=0.0,
            L_agn_bol=0.0,
            radio_loudness=0.0,
            alpha_agn=0.7,
            l_bband=1.0,
            log_nu_cut=log_nu_cut,
            include_freefree=False,
        )

        lnu = float(result[0])

        # Closed-form expectation: L_nu = (nu / 5e9)**(-0.7) * exp(-nu / 10^log_nu_cut)
        nu_ref = 5e9
        nu_cut = 10.0**log_nu_cut
        expected = (nu_hz / nu_ref) ** (-0.7) * np.exp(-nu_hz / nu_cut)

        assert np.allclose(lnu, expected, rtol=1e-6), (
            f"log_nu_cut={log_nu_cut}, nu={nu_hz:.2e}: "
            f"got {lnu:.6e}, expected {expected:.6e}, "
            f"rel error {abs(lnu - expected) / expected:.6e}"
        )

    @pytest.mark.parametrize("log_nu_cut", [11.0, 40.0])
    def test_compute_radio_components_honors_cutoff(self, log_nu_cut):
        """compute_radio_components honors the log_nu_cut parameter.

        Its AGN term for log_nu_cut in {11.0, 40.0} divided by its AGN term
        at the default equals exp(-nu/10**c) / exp(-nu/1e13) to 1e-6 at
        nu = 300e9 Hz. The default call equals the call with log_nu_cut=13.0
        exactly (using l_bband=1.0).
        """
        from tengri.components.radio.radio import compute_radio_components

        nu_hz = 300e9
        wavelength = _C_AA_PER_S / nu_hz
        wavelength_array = jnp.asarray([wavelength])

        # Minimal valid call with l_bband to ensure AGN is non-zero
        result_default = compute_radio_components(
            wavelength_array,
            L_ir=0.0,
            L_agn_bol=0.0,
            radio_loudness=0.0,
            alpha_agn=0.7,
            l_bband=1.0,
        )

        result_custom = compute_radio_components(
            wavelength_array,
            L_ir=0.0,
            L_agn_bol=0.0,
            radio_loudness=0.0,
            alpha_agn=0.7,
            l_bband=1.0,
            log_nu_cut=log_nu_cut,
        )

        agn_default = float(result_default["agn"][0])
        agn_custom = float(result_custom["agn"][0])

        # Expected ratio: exp(-nu/10^c) / exp(-nu/1e13)
        nu_cut_custom = 10.0**log_nu_cut
        nu_cut_default = 1e13
        expected_ratio = np.exp(-nu_hz / nu_cut_custom) / np.exp(-nu_hz / nu_cut_default)

        ratio = agn_custom / agn_default
        assert np.allclose(ratio, expected_ratio, rtol=1e-6), (
            f"log_nu_cut={log_nu_cut}, nu={nu_hz:.2e}: "
            f"got ratio {ratio:.6e}, expected {expected_ratio:.6e}, "
            f"rel error {abs(ratio - expected_ratio) / expected_ratio:.6e}"
        )

    def test_compute_radio_components_default_is_13(self):
        """compute_radio_components without log_nu_cut equals the call with
        log_nu_cut=13.0 exactly.
        """
        from tengri.components.radio.radio import compute_radio_components

        wavelength = jnp.linspace(1000, 30000, 50)

        kwargs = {
            "L_ir": 0.0,
            "L_agn_bol": 0.0,
            "radio_loudness": 0.0,
            "alpha_agn": 0.7,
        }

        # Call without log_nu_cut (uses default)
        result_default = compute_radio_components(wavelength, **kwargs)

        # Call with explicit log_nu_cut=13.0
        result_explicit = compute_radio_components(wavelength, log_nu_cut=13.0, **kwargs)

        # All component terms must match bit-exactly
        for key in ("synchrotron", "freefree", "agn", "total"):
            assert np.array_equal(result_default[key], result_explicit[key]), (
                f"Term '{key}' differs between default and explicit 13.0"
            )

    def test_wildcard_decision_pinned(self):
        """The `*` wildcard on the power-law model frees the loudness and the
        slope; the cutoff is read by both models and freed only when named.
        """
        from tengri.parameters.groups import _RADIO_AGN_PARAMS_BY_MODEL

        # Power-law model has only loudness and slope freed by wildcard
        assert _RADIO_AGN_PARAMS_BY_MODEL["powerlaw"] == frozenset(
            {"radio_loudness", "radio_alpha_agn"}
        ), "Power-law model wildcard should only free radio_loudness and radio_alpha_agn"

        # DPL model includes the cutoff in its parameter set
        assert "radio_log_nu_cut" in _RADIO_AGN_PARAMS_BY_MODEL["dpl"], (
            "DPL model should have radio_log_nu_cut in its parameter set"
        )
