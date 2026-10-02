# SPDX-License-Identifier: BSD-3-Clause
"""Pins for the conventions where tengri and CIGALE (pcigale 2025.1) differ by design.

The model-reference pages state these differences; each is pinned here to the code
that defines it, not to tengri's own output:

* equivalent-width sign and continuum (``restframe_parameters.py`` ``EW``: emission
  positive, linear continuum through the sideband centers; tengri's
  ``SpectralIndexDef`` is the Lick form, absorption positive; ``equivalent_width``
  is emission positive on a constant continuum),
* the star-forming radio normalization (CIGALE ``radio.py``: ``qir_sf`` = 2.58 anchored at
  the 21 cm wavelength; tengri: ``radio_q_ir`` = 2.64, Bell 2003, anchored at 1.4 GHz),
* the AGN jet: tengri's loudness is ``log10(L_5GHz / L_4400)`` (CIGALE: ``L_5GHz / L_2500``,
  linear), and tengri applies the cutoff ``exp(-nu/10^13 Hz)`` that CIGALE does not.
"""

import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.crossval

_C_NM = 2.99792458e17  # nm/s
_C_AA = 2.99792458e18  # A/s


# ── equivalent widths ─────────────────────────────────────────────

_PCIGALE_WINDOWS = "Halpha/650.0/652.5/653.5/660.0/661.0/663.5"  # nm: blue / line / red

_CASES = [(1.0, 0.0), (1.0, -4.0), (-0.5, 0.0), (-0.5, -4.0)]  # (line amplitude, continuum slope)


def _gaussian_line_on_power_law(amplitude, slope):
    """L_lambda ~ (lambda/6563 A)^slope * (1 + A exp(-(d lambda)^2 / 2 sigma^2)), sigma = 1.5 A."""
    wl_nm = np.linspace(640.0, 672.0, 12801)
    llam = (
        1e30
        * (wl_nm / 656.3) ** slope
        * (1.0 + amplitude * np.exp(-0.5 * ((wl_nm - 656.3) / 0.15) ** 2))
    )
    return wl_nm, llam


def _ew_pcigale_angstrom(wl_nm, llam):
    from pcigale.sed import SED
    from pcigale.sed_modules.restframe_parameters import RestframeParam

    sed = SED()
    sed.add_contribution("x", wl_nm.copy(), llam.copy())
    kwargs = dict.fromkeys(RestframeParam.parameters, False)  # beta, Dn4000, IRX off
    kwargs.update({k: "" for k in kwargs if k.endswith("_filters")})
    kwargs["EW"] = _PCIGALE_WINDOWS
    module = RestframeParam(name="rf", **kwargs)
    return module.EW(sed)["Halpha"] * 10.0


@pytest.mark.parametrize(("amplitude", "slope"), _CASES)
def test_equivalent_width_signs_and_continua(amplitude, slope):
    pytest.importorskip("pcigale")
    from tengri.analysis.diagnostics.spectral import equivalent_width
    from tengri.observation.spectral_indices import SpectralIndexDef, measure_index_jax

    wl_nm, llam = _gaussian_line_on_power_law(amplitude, slope)
    truth = amplitude * 1.5 * np.sqrt(2.0 * np.pi)  # Angstrom, emission positive
    wave_aa = jnp.asarray(wl_nm * 10.0)

    pcigale = _ew_pcigale_angstrom(wl_nm, llam)
    lick = float(
        measure_index_jax(
            wave_aa,
            jnp.asarray(llam),
            SpectralIndexDef(
                name="halpha_pcigale_windows",
                index_type="EW",
                continuum=((6500.0, 6525.0), (6610.0, 6635.0)),
                feature=(6535.0, 6600.0),
            ),
        )
    )
    lnu = llam * wl_nm**2 * 1e7 / _C_NM
    diag = float(equivalent_width(wave_aa, jnp.asarray(lnu), 6563.0, 20.0, 50.0))

    # CIGALE: emission positive (the code, not its parameter help), linear continuum
    assert np.sign(pcigale) == np.sign(truth)
    assert pcigale == pytest.approx(truth, rel=3e-2)
    if slope == 0.0:
        assert pcigale == pytest.approx(truth, rel=1e-3)
    # tengri SpectralIndexDef: Lick convention, absorption positive; same continuum as CIGALE here
    # (sidebands symmetric about the line window), so the two are equal and opposite
    assert lick == pytest.approx(-pcigale, rel=1e-3)
    # tengri equivalent_width: emission positive, constant continuum from two sidebands
    assert np.sign(diag) == np.sign(truth)
    assert diag == pytest.approx(truth, rel=1e-2)


# ── star-forming radio normalization ──────────────────────────────


@pytest.mark.parametrize(("qir", "alpha"), [(2.58, 0.8), (2.2, 0.6), (3.0, 1.0)])
def test_sf_radio_normalization_differs_by_q_and_anchor_frequency(qir, alpha):
    pytest.importorskip("pcigale")
    from pcigale.sed import SED
    from pcigale.sed_modules import radio as pcigale_radio

    from tengri.components.radio import radio as tengri_radio

    nu = np.array([0.15e9, 1.4e9, 5e9, 30e9, 100e9])
    l_ir_w = 1e37
    sed = SED()
    sed.add_info("dust.luminosity", l_ir_w, True, unit="W")
    sed.add_info("agn.intrin_Lnu_2500A_30deg", 0.0, True, unit="W/Hz")
    pcigale_radio.Radio(
        name="radio", qir_sf=qir, alpha_sf=alpha, R_agn=10.0, alpha_agn=0.7
    ).process(sed)
    w = sed.wavelength_grid
    pcigale_lnu = (
        np.interp(_C_NM / nu, w, sed.luminosities["radio.sf_nonthermal"] * w**2 / _C_NM) * 1e7
    )

    nu_21cm = _C_NM / 2.1e8  # CIGALE's anchor: lambda = 21.0 cm, 1.42758 GHz
    default_q = 2.64
    expected = 10.0 ** (qir - default_q) * (1.4e9 / nu_21cm) ** alpha  # tengri default / CIGALE
    got = (
        np.asarray(
            tengri_radio.radio_sfr_bell2003(
                _C_AA / nu, l_ir_w * 1e7, q_ir=default_q, alpha_sf=alpha
            )
        )
        / pcigale_lnu
    )
    np.testing.assert_allclose(got, expected, rtol=1e-4)
    # with CIGALE's q and its anchor, tengri reproduces CIGALE
    matched = (
        np.asarray(
            tengri_radio.radio_sfr_bell2003(
                _C_AA / nu, l_ir_w * 1e7, q_ir=qir, alpha_sf=alpha, nu_ref=nu_21cm
            )
        )
        / pcigale_lnu
    )
    np.testing.assert_allclose(matched, 1.0, rtol=2e-5)


# ── AGN jet cutoff ────────────────────────────────────────────────


@pytest.mark.parametrize("nu_hz", [30e9, 100e9, 300e9])
def test_agn_jet_cutoff_factor(nu_hz):
    """tengri's default AGN jet carries exp(-nu / 10^13 Hz); ``log_nu_cut=40`` removes it."""
    from tengri.components.radio import radio as tengri_radio

    wave = np.array([_C_AA / nu_hz])
    with_cut = float(tengri_radio.radio_agn(wave, 0.0, 0.0, 0.7, l_bband=1.0)[0])
    no_cut = float(tengri_radio.radio_agn(wave, 0.0, 0.0, 0.7, l_bband=1.0, log_nu_cut=40.0)[0])
    assert with_cut / no_cut == pytest.approx(np.exp(-nu_hz / 1e13), rel=1e-6)


def test_agn_loudness_is_log10_of_the_5ghz_to_4400_angstrom_ratio():
    """``radio_loudness = log10(L_5GHz / L_4400)``; CIGALE's R is the linear ratio on 2500 A."""
    from tengri.components.radio import radio as tengri_radio

    l_4400 = 2.0e28  # erg/s/Hz
    loudness = 1.5
    lnu_5ghz = float(
        tengri_radio.radio_agn(
            np.array([_C_AA / 5e9]), 0.0, loudness, 0.7, l_bband=l_4400, log_nu_cut=40.0
        )[0]
    )
    assert lnu_5ghz == pytest.approx(l_4400 * 10.0**loudness, rel=1e-6)
