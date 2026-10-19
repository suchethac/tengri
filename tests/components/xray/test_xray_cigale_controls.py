# SPDX-License-Identifier: BSD-3-Clause
"""CIGALE yang20 controls on the tengri ``xray`` block (#2661).

Three controls CIGALE's ``yang20`` module exposes and tengri could not express:

- the absolute ``alpha_ox`` (``xray_alpha_ox``), an alternative to the offset
  ``xray_delta_alpha_ox`` from the Just+2007 relation;
- the anisotropy coefficients ``(a1, a2)`` of ``Lx(i)/Lx(0)``;
- CIGALE's ``max_dev_alpha_ox``, whose soft form is a Gaussian prior on the offset.

The per-term X-ray luminosities that CIGALE publishes are checked for finiteness and
for the HMXB + LMXB sum.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from tengri import DEFAULT, Fixed, SEDModel

pytestmark = pytest.mark.contract

#: Log10 of the disc L_nu at 2500 A used by the absolute <-> offset mapping [erg/s/Hz].
#: The issue's table is printed at 28.92 (the reproducer's model); its offsets match this
#: value to 1e-3, and 28.90 would be 2.7e-3 off for alpha_ox = -1.9.
LOG_L_2500 = 28.92
#: Yang+2020 reference inclination: cos(30 deg).
COS_30 = 0.8660254037844387
#: The issue's table: CIGALE absolute alpha_ox nodes and the offsets they need at 28.92.
ALPHA_OX_NODES = (-1.9, -1.6, -1.4, -1.1)
DELTA_TABLE = (-0.576, -0.276, -0.076, 0.224)
JUST07_SLOPE, JUST07_INTERCEPT = -0.137, 2.638

_XRAY_KEYS = {
    "xray_gamma_agn": 1.8,
    "xray_gamma_hmxb": 2.0,
    "xray_gamma_lmxb": 1.56,
    "xray_E_cut": 300.0,
    "xray_log_nh": 20.0,
    "xray_det_hmxb": 0.0,
    "xray_det_lmxb": 0.0,
    "xray_alpha_irx": 0.3,
    "xray_delta_alpha_ox": 0.0,
    "xray_alpha_ox": -1.5,
    "xray_a1": 0.5,
    "xray_a2": 0.0,
}


def _just07(log_l_2500: float) -> float:
    return JUST07_SLOPE * log_l_2500 + JUST07_INTERCEPT


def _component(alpha_ox_mode: str = "delta"):
    from tengri.components.xray.component import XRaySEDComponent, XRaySEDComponentConfig

    config = XRaySEDComponentConfig(model="yang20", alpha_ox_mode=alpha_ox_mode)
    return XRaySEDComponent(config=config)


def _params(**overrides):
    return {**_XRAY_KEYS, **overrides}


def _inputs(l_2500: float = 10.0**LOG_L_2500, cos_inc: float = COS_30):
    return {
        "sfr": jnp.asarray(1.0),
        "stellar_mass": jnp.asarray(1.0e10),
        "stellar_age_gyr": jnp.asarray(3.0),
        "metallicity_z": jnp.asarray(0.02),
        "l_2500": jnp.asarray(l_2500),
        "cos_inc": jnp.asarray(cos_inc),
        "log_l_12um": jnp.asarray(-jnp.inf),
    }


# A rest-frame grid across the corona band, 0.3-40 keV (the 2 keV anchor is in it).
_WAVE = jnp.asarray(np.geomspace(12.398419843320026 / 40.0, 12.398419843320026 / 0.3, 48))
_WAVE_2KEV = jnp.asarray([12.398419843320026 / 2.0])


@pytest.fixture(scope="module")
def ssp():
    import tengri

    return tengri.load_ssp()


def _agn_model(ssp, xray_type="yang20", **xray):
    """A composable-AGN model with an X-ray corona, the minimal panchromatic build."""
    agn = {
        "type": "composable",
        "disc": {"type": "schartmann2005", "all_params": Fixed(DEFAULT)},
        "torus": {"type": "skirtor", "agn_torus_frac": Fixed(1.0), "all_params": Fixed(DEFAULT)},
        "atten": {"type": "none"},
        "agn_log_lbol": Fixed(12.0),
        "agn_cos_inc": Fixed(COS_30),
        "all_params": Fixed(DEFAULT),
    }
    return SEDModel.build(
        ssp_data=ssp,
        met={"logzsol": Fixed(0.0), "all_params": Fixed(DEFAULT)},
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(1.0),
            "age_gyr": Fixed(5.0),
            "log_total_mass": Fixed(10.0),
            "all_params": Fixed(DEFAULT),
        },
        dust_attenuation={"type": "none"},
        dust_emission={"type": "none"},
        neb={"type": "none"},
        agn=agn,
        xray={"type": xray_type, "all_params": Fixed(DEFAULT), **xray},
        redshift=Fixed(0.0),
    )


class TestAbsoluteAlphaOx:
    """(a) Absolute alpha_ox equals the offset that CIGALE's node implies."""

    def test_offsets_implied_by_the_issue_table(self):
        for alpha, table in zip(ALPHA_OX_NODES, DELTA_TABLE, strict=True):
            delta = alpha - _just07(LOG_L_2500)
            assert abs(delta - table) < 1e-3, (alpha, delta, table)

    @pytest.mark.parametrize("alpha", ALPHA_OX_NODES)
    def test_absolute_mode_spectrum_is_the_offset_spectrum(self, alpha):
        """Absolute alpha_ox gives the same corona as the offset it converts to.

        The absolute-mode params carry a deliberately wrong ``xray_delta_alpha_ox`` (0.0),
        so the test fails if the component ignores the conversion.
        """
        delta = alpha - _just07(LOG_L_2500)
        absolute = _component("absolute").emission_terms(
            _params(xray_alpha_ox=alpha), _WAVE, **_inputs()
        )
        offset = _component("delta").emission_terms(
            _params(xray_delta_alpha_ox=delta), _WAVE, **_inputs()
        )
        for term in ("agn", "hmxb", "lmxb", "hotgas"):
            assert_allclose(
                np.asarray(absolute[term]), np.asarray(offset[term]), rtol=1e-12, atol=0.0
            )

    def test_absolute_mode_luminosity_matches_offset_mode(self):
        """The same conversion holds on the band-luminosity path the properties read."""
        alpha = -1.4
        delta = alpha - _just07(LOG_L_2500)
        absolute = _component("absolute").log_band_luminosities(
            _params(xray_alpha_ox=alpha), **_inputs()
        )
        offset = _component("delta").log_band_luminosities(
            _params(xray_delta_alpha_ox=delta), **_inputs()
        )
        assert_allclose(float(absolute["agn"]), float(offset["agn"]), rtol=1e-12)
        assert_allclose(float(absolute["agn_total"]), float(offset["agn_total"]), rtol=1e-12)

    def test_absolute_alpha_ox_sets_the_2kev_anchor(self):
        """L_nu(2 keV) = L_2500 * 10**(alpha_ox / 0.3838) at 30 degrees."""
        alpha = -1.4
        lnu = _component("absolute").log_band_luminosities(
            _params(xray_alpha_ox=alpha), **_inputs()
        )["lnu_2kev_30deg"]
        expected = LOG_L_2500 + alpha / 0.3838
        assert_allclose(float(lnu), expected, atol=1e-10)

    def test_grammar_absolute_model_matches_offset_model(self, ssp):
        """The alpha_ox key on the xray group gives the offset model's SED, end to end."""
        absolute = _agn_model(ssp, alpha_ox=Fixed(-1.4)).predict_state({})
        l_2500 = float(absolute.derived["L_2500_intrinsic"])
        delta = -1.4 - _just07(np.log10(l_2500))
        offset = _agn_model(ssp, delta_alpha_ox=Fixed(delta)).predict_state({})
        assert_allclose(
            np.asarray(absolute.derived["sed_xray"]),
            np.asarray(offset.derived["sed_xray"]),
            rtol=1e-10,
            atol=0.0,
        )


class TestAlphaOxDeclarationGrammar:
    """(b) Both keys are refused; neither keeps today's default."""

    def test_both_keys_raise_naming_both(self, ssp):
        from tengri.config.exceptions import ParameterError

        with pytest.raises(ParameterError) as info:
            _agn_model(ssp, alpha_ox=Fixed(-1.4), delta_alpha_ox=Fixed(0.0))
        message = str(info.value)
        assert "delta_alpha_ox" in message
        assert "alpha_ox" in message

    def test_absolute_alpha_ox_refused_under_lopez24(self, ssp):
        """Only the yang20 corona reads alpha_ox; lopez24 would ignore it silently."""
        from tengri.config.exceptions import ParameterError

        with pytest.raises(ParameterError, match="lopez24"):
            _agn_model(ssp, xray_type="lopez24", alpha_ox=Fixed(-1.4))

    def test_neither_key_is_the_default_offset_mode(self, ssp):
        """No key given: the corona reads the offset, which defaults to 0."""
        default = _agn_model(ssp)
        explicit = _agn_model(ssp, delta_alpha_ox=Fixed(0.0))
        sed_default = np.asarray(default.predict_state({}).derived["sed_xray"])
        sed_explicit = np.asarray(explicit.predict_state({}).derived["sed_xray"])
        np.testing.assert_array_equal(sed_default, sed_explicit)


class TestAnisotropyCoefficients:
    """(c) (a1, a2) = (0, 0) is isotropic; the default is Yang+2022's (0.5, 0)."""

    @staticmethod
    def _ratio(cos_inc, a1, a2):
        params = _params(xray_a1=a1, xray_a2=a2)
        face_on = _component().emission_terms(params, _WAVE_2KEV, **_inputs(cos_inc=1.0))
        tilted = _component().emission_terms(params, _WAVE_2KEV, **_inputs(cos_inc=cos_inc))
        return float(tilted["agn"][0] / face_on["agn"][0])

    @pytest.mark.parametrize("cos_inc", [0.5, 0.1])
    def test_isotropic_when_a1_a2_are_zero(self, cos_inc):
        assert_allclose(self._ratio(cos_inc, 0.0, 0.0), 1.0, rtol=1e-6)

    @pytest.mark.parametrize(("cos_inc", "expected"), [(0.5, 0.75), (0.1, 0.55)])
    def test_default_coefficients_match_yang2022(self, cos_inc, expected):
        assert_allclose(self._ratio(cos_inc, 0.5, 0.0), expected, rtol=1e-6)


class TestMaxDevAsGaussianPrior:
    """(d) max_dev_alpha_ox as a Gaussian prior on the offset with sigma = max_dev."""

    def test_gaussian_offset_builds(self, ssp):
        from tengri import Gaussian

        max_dev = 0.2
        model = _agn_model(ssp, delta_alpha_ox=Gaussian(0.0, max_dev))
        derived = model.predict_state({"xray_delta_alpha_ox": 0.1}).derived
        sed = np.asarray(derived["sed_xray"])
        assert np.all(np.isfinite(sed)), "a non-finite X-ray SED under a Gaussian offset"
        assert np.any(sed != 0.0), "an identically zero X-ray SED: the AGN corona is missing"

    def test_log_prior_drops_by_half_at_max_dev(self):
        from tengri import Gaussian

        prior = Gaussian(0.0, 0.2)
        at_zero = float(prior.log_prob(jnp.asarray(0.0)))
        assert_allclose(float(prior.log_prob(jnp.asarray(0.2))) - at_zero, -0.5, atol=1e-12)
        assert_allclose(float(prior.log_prob(jnp.asarray(-0.2))) - at_zero, -0.5, atol=1e-12)


class TestPerTermLuminosities:
    """(e) The new log_L_* keys are finite and the XRB terms add to the XRB total."""

    def test_total_band_is_the_unabsorbed_corona_band_integral(self):
        """agn_total = L_nu(2 keV) x integral of the unabsorbed shape over 0.25-1240 keV.

        Checked against a 20000-node quadrature of the same shape in numpy; the 2-10 keV
        band the emitted corona is reported over sits inside the total band.
        """
        from tengri.utils.physics_constants import KEV_TO_HZ

        lum = _component().log_band_luminosities(_params(), **_inputs())
        E_lo, E_hi = 1.2398419843320026 / (10.0**0.7), 1.2398419843320026 / 1.0e-3
        E = np.geomspace(E_lo, E_hi, 20000)
        shape = (E / 2.0) ** (1.0 - 1.8) * np.exp(-(E - 2.0) / 300.0)
        expected = float(np.log10(np.trapezoid(shape, E * KEV_TO_HZ)))
        measured = float(lum["agn_total"]) - float(lum["lnu_2kev_30deg"])
        # The 400-node quadrature is within 4e-5 dex of the 20000-node reference.
        assert_allclose(measured, expected, atol=1e-4)
        assert float(lum["agn_total"]) > float(lum["agn"]), "the total band must exceed 2-10 keV"

    def test_new_keys_are_finite_and_xrb_terms_add(self, ssp):
        from tengri.utils.scale import log10_add

        derived = _agn_model(ssp).predict_state({}).derived
        new_keys = (
            "log_L_x_hmxb_2_10",
            "log_L_x_lmxb_2_10",
            "log_L_x_hotgas_0p5_2",
            "log_L_x_agn_total",
            "log_Lnu_2keV_30deg",
        )
        for key in new_keys:
            # grad-assert: finite-only — these are log10 luminosities in dex; a zero dex
            # value is a physical luminosity of 1 erg/s, so zero is not a failure here.
            assert np.isfinite(float(derived[key])), key
        xrb = log10_add(derived["log_L_x_hmxb_2_10"], derived["log_L_x_lmxb_2_10"])
        assert_allclose(float(xrb), float(derived["log_L_x_xrb_2_10"]), rtol=0.0, atol=1e-10)
