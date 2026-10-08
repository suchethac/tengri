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


def _window_average_conventions(wave, flam):
    """(CIGALE-form, Lick-form) EW [A] of F_lambda, both on the line through the sideband means.

    Windows [A]: blue 6500-6525, line 6535-6600, red 6610-6635. CIGALE: W (<F>/<C> - 1),
    emission positive. Lick (Trager et al. 1998, Eq. 3): W (1 - <F/C>), absorption positive.
    """
    def mean(lo, hi, y):
        m = (wave >= lo) & (wave <= hi)
        return np.trapezoid(y[m], wave[m]) / (wave[m][-1] - wave[m][0])

    f_b, f_r = mean(6500.0, 6525.0, flam), mean(6610.0, 6635.0, flam)
    cont = f_b + (f_r - f_b) * (wave - 6512.5) / (6622.5 - 6512.5)
    width = 65.0
    cigale = width * (mean(6535.0, 6600.0, flam) / mean(6535.0, 6600.0, cont) - 1.0)
    lick = width * (1.0 - mean(6535.0, 6600.0, flam / cont))
    return cigale, lick


@pytest.mark.parametrize(("amplitude", "slope"), _CASES)
def test_equivalent_width_signs_and_continua(amplitude, slope):
    pytest.importorskip("pcigale")
    from tengri.analysis.diagnostics.spectral import equivalent_width
    from tengri.observation.spectral_indices import SpectralIndexDef, measure_index_jax

    wl_nm, llam = _gaussian_line_on_power_law(amplitude, slope)
    truth = amplitude * 1.5 * np.sqrt(2.0 * np.pi)  # Angstrom, emission positive
    wave_aa = jnp.asarray(wl_nm * 10.0)

    pcigale = _ew_pcigale_angstrom(wl_nm, llam)
    # The index operator takes a flux density per unit frequency and converts it to F_lambda
    # itself (Trager et al. 1998), so it is given L_nu; pcigale's input is L_lambda.
    lnu = llam * wl_nm**2 * 1e7 / _C_NM
    lick = float(
        measure_index_jax(
            wave_aa,
            jnp.asarray(lnu),
            SpectralIndexDef(
                name="halpha_pcigale_windows",
                index_type="EW",
                continuum=((6500.0, 6525.0), (6610.0, 6635.0)),
                feature=(6535.0, 6600.0),
            ),
        )
    )
    diag = float(equivalent_width(wave_aa, jnp.asarray(lnu), 6563.0, 20.0, 50.0))

    # CIGALE: emission positive (the code, not its parameter help), linear continuum
    assert np.sign(pcigale) == np.sign(truth)
    assert pcigale == pytest.approx(truth, rel=3e-2)
    if slope == 0.0:
        assert pcigale == pytest.approx(truth, rel=1e-3)
    # tengri SpectralIndexDef: Lick convention, absorption positive, on the same F_lambda and the
    # same straight-line continuum (sideband means at the sideband midpoints). The two differ in
    # how the line window is averaged: CIGALE forms W (<F>/<C> - 1) (ratio of window means), the
    # Lick index W (1 - <F/C>) (Trager et al. 1998, Eq. 3, mean of the ratio). They are equal and
    # opposite on a flat continuum and differ at second order in the continuum slope across the
    # window. Both are evaluated below from their definitions, on the fine input grid.
    ref_pcigale, ref_lick = _window_average_conventions(wl_nm * 10.0, llam)
    assert pcigale == pytest.approx(ref_pcigale, rel=1e-3)
    assert lick == pytest.approx(ref_lick, rel=1e-3)
    if slope == 0.0:
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


# ── equivalent-width numbers from the review ──────────────────────


@pytest.mark.parametrize("continuum_width", [20, 50, 100])
def test_equivalent_width_on_steep_continuum_pinned_numbers(continuum_width):
    """Gaussian line (sigma=1.5 A, unit amplitude, at 6563 A) on L_lambda ~ lambda^-4.

    `equivalent_width` returns pinned numbers for different continuum windows
    (standard: 25 A wide sidebands; also tested: 20, 100 A).
    """
    from tengri.analysis.diagnostics.spectral import equivalent_width

    wl_nm, llam = _gaussian_line_on_power_law(1.0, -4.0)
    wave_aa = jnp.asarray(wl_nm * 10.0)
    lnu = llam * wl_nm**2 * 1e7 / _C_NM

    result = float(equivalent_width(wave_aa, jnp.asarray(lnu), 6563.0, 20.0, continuum_width))
    expected = {20: 3.752, 50: 3.739, 100: 3.704}[continuum_width]
    assert result == pytest.approx(expected, abs=5e-3)


def test_spectral_index_def_on_l_nu_steep_continuum():
    """SpectralIndexDef on L_nu array gives negative value (Lick absorption convention).

    Same line as the EW tests, measured with SpectralIndexDef using the standard
    windows (sidebands 25 A wide, centered 50 A from the line window center).

    The reference is Trager et al. (1998, ApJS 116, 1, Eqs. 1-3) evaluated here in
    numpy on F_lambda: the pseudo-continuum is the straight line through the two
    sideband means at the sideband mid-wavelengths. On this lambda^-4 continuum that
    line sits above the curved true continuum, so the index (-3.716 A) is 1.2 % smaller
    in magnitude than the equivalent width against the exact power law (-3.760 A),
    and differs from the constant mean-of-sidebands continuum of
    ``equivalent_width`` (-3.752 A at the same 20 A windows).
    """
    from tengri.observation.spectral_indices import SpectralIndexDef, measure_index_jax

    wl_nm, llam = _gaussian_line_on_power_law(1.0, -4.0)
    wave_aa = jnp.asarray(wl_nm * 10.0)
    lnu = llam * wl_nm**2 * 1e7 / _C_NM
    blue, red, feature = (6500.0, 6525.0), (6610.0, 6635.0), (6535.0, 6600.0)

    result = float(
        measure_index_jax(
            wave_aa,
            jnp.asarray(lnu),
            SpectralIndexDef(
                name="halpha_lnu",
                index_type="EW",
                continuum=(blue, red),
                feature=feature,
            ),
        )
    )

    wave = wl_nm * 10.0
    flam = np.asarray(llam)

    def _band_mean(lo, hi):
        m = (wave >= lo) & (wave <= hi)
        return np.trapezoid(flam[m], wave[m]) / (wave[m][-1] - wave[m][0])

    blue_mid, red_mid = np.mean(blue), np.mean(red)
    blue_mean, red_mean = _band_mean(*blue), _band_mean(*red)
    in_feature = (wave >= feature[0]) & (wave <= feature[1])
    pseudo = blue_mean + (red_mean - blue_mean) * (wave[in_feature] - blue_mid) / (
        red_mid - blue_mid
    )
    lick = np.trapezoid(1.0 - flam[in_feature] / pseudo, wave[in_feature])

    assert lick == pytest.approx(-3.716, abs=5e-3)
    assert result == pytest.approx(lick, abs=1e-3)


def test_equivalent_width_and_spectral_index_on_flat_continuum():
    """On a flat continuum, all three implementations agree to ±3.76 A.

    Tests `equivalent_width`, `SpectralIndexDef`, and CIGALE (pcigale).
    """
    pytest.importorskip("pcigale")
    from tengri.analysis.diagnostics.spectral import equivalent_width
    from tengri.observation.spectral_indices import SpectralIndexDef, measure_index_jax

    wl_nm, llam = _gaussian_line_on_power_law(1.0, 0.0)
    wave_aa = jnp.asarray(wl_nm * 10.0)
    lnu = llam * wl_nm**2 * 1e7 / _C_NM

    # CIGALE result
    pcigale_ew = _ew_pcigale_angstrom(wl_nm, llam)

    # tengri equivalent_width
    tengri_ew = float(equivalent_width(wave_aa, jnp.asarray(lnu), 6563.0, 20.0, 50.0))

    # tengri SpectralIndexDef on L_nu (converted to the F_lambda CIGALE is given)
    lick_llam = float(
        measure_index_jax(
            wave_aa,
            jnp.asarray(lnu),
            SpectralIndexDef(
                name="halpha_flat",
                index_type="EW",
                continuum=((6500.0, 6525.0), (6610.0, 6635.0)),
                feature=(6535.0, 6600.0),
            ),
        )
    )

    assert pcigale_ew == pytest.approx(3.760, abs=5e-3)
    assert tengri_ew == pytest.approx(3.760, abs=5e-3)
    assert lick_llam == pytest.approx(-3.760, abs=5e-3)


def test_equivalent_width_on_flat_lnu_and_llam_continua():
    """equivalent_width on flat continua: L_nu gives 3.754, L_lambda gives 3.760.

    Tests the Gaussian line (sigma=1.5 A, unit amplitude, at 6563 A) on
    continua flat in L_nu and L_lambda, with sidebands 50 A wide.
    No pcigale needed; no importorskip.
    """
    from tengri.analysis.diagnostics.spectral import equivalent_width

    # Flat L_nu: slope = -2 in L_lambda (since L_nu = L_lambda * wl_nm^2)
    wl_nm_lnu, llam_lnu = _gaussian_line_on_power_law(1.0, -2.0)
    wave_aa_lnu = jnp.asarray(wl_nm_lnu * 10.0)
    lnu_flat = llam_lnu * wl_nm_lnu**2 * 1e7 / _C_NM

    # Flat L_lambda: slope = 0
    wl_nm_llam, llam_flat = _gaussian_line_on_power_law(1.0, 0.0)
    wave_aa_llam = jnp.asarray(wl_nm_llam * 10.0)
    lnu_llam = llam_flat * wl_nm_llam**2 * 1e7 / _C_NM

    # equivalent_width on L_nu flat continuum
    ew_lnu = float(equivalent_width(wave_aa_lnu, jnp.asarray(lnu_flat), 6563.0, 20.0, 50.0))

    # equivalent_width on L_lambda flat continuum
    ew_llam = float(equivalent_width(wave_aa_llam, jnp.asarray(lnu_llam), 6563.0, 20.0, 50.0))

    assert ew_lnu == pytest.approx(3.754, abs=5e-3)
    assert ew_llam == pytest.approx(3.760, abs=5e-3)
