# SPDX-License-Identifier: BSD-3-Clause
"""Star-formation synchrotron extends to 100 um, as in pcigale's radio module.

The non-thermal power law L_nu = L_nu(nu_ref) (nu / nu_ref)^-alpha has no physical
break at 1 mm. pcigale's ``sed_modules/radio.py`` evaluates it on
``np.logspace(5, 11, 1000)`` nm, i.e. down to 100 um, where it is 1.3 % of the CIGALE
SED at 1 mm and 0.7 % at 850 um. Zeroing it shortward of 1 mm (the former edge) removed
that flux and, in the CIGALE reproduction, offset a nebular excess.

The thermal free-free (Murphy et al. 2011 term, and the split mode's thermal law) shares
the same 100 um edge by owner ruling: an optically thin power law has no break at 1 mm,
and the overlap with the nebular continuum is gated by ``include_freefree`` and
``freefree_wave_min``, not by the edge.
"""

from __future__ import annotations

import warnings

import jax
import numpy as np
import pytest
from tests._radio_band import band_l_ir

from tengri import FREE, Fixed, SEDModel
from tengri.components.radio import radio as R, radio_precompute as RP

pytestmark = pytest.mark.regression_bug

_C_AA = 2.99792458e18
_C_M = 299_792_458.0
_NU_21CM = _C_M / 0.21
_L_IR = 1e44  # erg/s
_Q, _ALPHA = 2.58, 0.8
_UM = 1e4  # Angstrom per micron
_SUB_MM_UM = np.array([100.5, 160.0, 250.0, 350.0, 500.0, 850.0, 990.0])


def _power_law(wave_aa, nu_ref=R._NU_REF_BELL2003_HZ, q=_Q, alpha=_ALPHA):
    nu = _C_AA / wave_aa
    return _L_IR / (3.75e12 * 10.0**q) * (nu / nu_ref) ** (-alpha)


@pytest.mark.parametrize("um", _SUB_MM_UM)
def test_bell2003_sub_mm_is_the_power_law(um):
    """Bell (2003) Eq. 1 anchor times (nu/nu_ref)^-alpha at 100-1000 um."""
    wave = np.array([um * _UM])
    got = np.asarray(R.radio_sfr_bell2003(wave, _L_IR, q_ir=_Q, alpha_sf=_ALPHA))
    np.testing.assert_allclose(got, _power_law(wave), rtol=1e-12)


def test_other_sf_modes_share_the_extended_edge():
    wave = np.array([250.0 * _UM, 850.0 * _UM])
    for fn in (R.radio_sfr_delvecchio2021, R.radio_sfr_mccheyne2022):
        out = np.asarray(fn(wave, _L_IR, 10.0, 1.0))
        assert np.all(out > 0.0)
    split = np.asarray(R.radio_sfr_bell2003_split(wave, _L_IR))
    nonthermal_only = np.asarray(R.radio_sfr_bell2003_split(wave, _L_IR, include_thermal=False))
    assert np.all(nonthermal_only > 0.0)
    # The thermal law shares the 100 um edge: the split exceeds the non-thermal part by f_thermal.
    nu = _C_AA / wave
    thermal = (
        R._F_THERMAL_AGNFITTER
        * (nu / R._NU_REF_BELL2003_HZ) ** (-R._ALPHA_THERMAL_AGNFITTER)
        * _L_IR
        / (3.75e12 * 10.0**2.64)
    )
    np.testing.assert_allclose(split - nonthermal_only, thermal, rtol=1e-12)


def test_below_100_um_is_zero_and_edge_is_included():
    wave = np.array([5.0e5, 9.9e5, 1.0e6, 1.0e7 - 1.0])
    got = np.asarray(R.radio_sfr_bell2003(wave, _L_IR, q_ir=_Q, alpha_sf=_ALPHA))
    assert got[0] == 0.0 and got[1] == 0.0
    assert got[2] > 0.0 and got[3] > 0.0


def test_values_longward_of_1mm_are_unchanged():
    """Frozen from the 1 mm-edge code (origin/main 2876cf562): bit-identical beyond 1 mm."""
    wave = np.array([1.0e7 + 1.0, 3.0e7, 2.14e9, 1.0e10])
    got = np.asarray(R.radio_sfr_bell2003(wave, _L_IR, q_ir=_Q, alpha_sf=_ALPHA))
    np.testing.assert_array_equal(got, _power_law(wave))


def test_free_free_at_500_um_is_the_murphy_power_law():
    """Murphy+2011 Eqs. 4 and 11 at T_e = 1e4 K: L_nu = SFR/4.6e-28 (nu/GHz)^-0.1."""
    wave = np.array([500.0 * _UM, 100.0 * _UM])
    nu = _C_AA / wave
    expect = (1.0 / 4.6e-28) * (nu / 1.0e9) ** -0.1 * 3.88e-44 * _L_IR
    np.testing.assert_allclose(np.asarray(R.radio_freefree(wave, _L_IR)), expect, rtol=1e-6)
    assert float(R.radio_freefree(np.array([9.9e5]), _L_IR)[0]) == 0.0


@pytest.mark.parametrize("mode", ["bell2003", "bell2003_split"])
def test_radio_block_is_continuous_through_1_mm(mode):
    """The kernel sum across 1 mm follows the power-law slope, with no step."""
    below, above = 1.0e7 * (1 - 1e-6), 1.0e7 * (1 + 1e-6)
    wave = np.array([below, above])
    if mode == "bell2003":
        out = np.asarray(R.radio_sfr_bell2003(wave, _L_IR, q_ir=_Q, alpha_sf=_ALPHA))
        out = out + np.asarray(R.radio_freefree(wave, _L_IR))
    else:
        out = np.asarray(R.radio_sfr_bell2003_split(wave, _L_IR))
    ratio = out[1] / out[0]
    # nu changes by 2e-6 across the cell: the ratio is 1 - 2e-6 alpha_eff, alpha_eff in [-0.1, 0.8]
    # (the former 1 mm edge gave 6.7x for bell2003 with free-free and 4.6x for the split).
    assert 1.0 - 0.3e-6 < ratio < 1.0 + 1.7e-6, ratio


def test_matches_pcigale_radio_module_below_1mm():
    """Same q, alpha, 21 cm anchor: tengri / pcigale = 1 at every pcigale node >= 100 um."""
    pytest.importorskip("pcigale")
    from pcigale.sed import SED
    from pcigale.sed_modules import radio as pcigale_radio

    sed = SED()
    sed.add_info("dust.luminosity", _L_IR / 1e7, True, unit="W")
    sed.add_info("agn.intrin_Lnu_2500A_30deg", 0.0, True, unit="W/Hz")
    pcigale_radio.Radio(
        name="radio", qir_sf=_Q, alpha_sf=_ALPHA, R_agn=10.0, alpha_agn=0.7
    ).process(sed)
    w_nm = sed.wavelength_grid
    pc = sed.luminosities["radio.sf_nonthermal"] * w_nm**2 / (_C_M * 1e9) * 1e7  # erg/s/Hz
    sub_mm = (w_nm >= 1.0e5) & (w_nm < 1.0e6)
    assert sub_mm.sum() > 100
    ours = np.asarray(
        R.radio_sfr_bell2003(w_nm[sub_mm] * 10.0, _L_IR, q_ir=_Q, alpha_sf=_ALPHA, nu_ref=_NU_21CM)
    )
    np.testing.assert_allclose(ours, pc[sub_mm], rtol=1e-6)
    # 850 and 500 um by interpolation of the pcigale curve
    for um in (500.0, 850.0):
        ref = np.exp(np.interp(np.log(um * 1e3), np.log(w_nm), np.log(pc)))
        mine = float(
            R.radio_sfr_bell2003(
                np.array([um * _UM]), _L_IR, q_ir=_Q, alpha_sf=_ALPHA, nu_ref=_NU_21CM
            )[0]
        )
        assert mine / ref == pytest.approx(1.0, abs=2e-5)


def test_precompute_grid_covers_100_um_and_keeps_old_nodes():
    wave = np.asarray(RP._WAVE_REST)
    assert wave.min() <= 1.0e6
    old = np.logspace(7, 13, 1024, dtype=np.float64)
    assert np.all(np.isin(old, wave))


def _build(ssp, obs, with_radio=True, sf_extra=None, neb=None):
    radio = {
        "sf": {"type": "bell2003", **({"freefree": False} if sf_extra is None else sf_extra)},
        "agn": {"type": "none"},
        "all_params": FREE,
        "q_ir": Fixed(_Q),
        "alpha_sf": Fixed(_ALPHA),
    }
    if sf_extra is not None and sf_extra.get("freefree"):
        radio = {k: v for k, v in radio.items() if k != "all_params"} | {
            "T_e": Fixed(1.0e4),  # Murphy+2011 reference temperature
            "alpha_ff": Fixed(-0.1),
        }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            redshift=Fixed(0.0),
            sfh={"type": "const", "all_params": FREE},
            dust_attenuation={"type": "two_component", "law": "calzetti"},
            dust_emission={"type": "draine_li2014"},
            **({"neb": neb} if neb is not None else {}),
            **({"radio": radio} if with_radio else {}),
        )


def test_model_sub_mm_synchrotron_and_ir_luminosities_unchanged(
    synthetic_ssp_wide, synthetic_tophat_obs
):
    """The SED holds the power law at 850 and 500 um; L_IR, the TIR input and sed_dust_ir
    are the dust block's, untouched by the radio block (it is integrated from sed_dust_ir)."""
    with_r = _build(synthetic_ssp_wide, synthetic_tophat_obs)
    without = _build(synthetic_ssp_wide, synthetic_tophat_obs, with_radio=False)
    key = jax.random.PRNGKey(0)
    s1 = with_r.predict_state(with_r.spec.sample(key))
    s0 = without.predict_state(without.spec.sample(key))
    wave = np.asarray(s1.wave)
    sed = np.asarray(s1.derived["sed_radio"])
    l_tir = float(np.asarray(s1.derived["radio_L_ir_input"])) * 3.828e33
    for um in (500.0, 850.0):
        i = int(np.argmin(np.abs(wave - um * _UM)))
        expect = l_tir / (3.75e12 * 10.0**_Q) * (_C_AA / wave[i] / 1.4e9) ** (-_ALPHA)
        assert sed[i] == pytest.approx(expect, rel=1e-3)
    assert sed[wave < 0.99e6].max() == 0.0
    # The radio block extends the wavelength grid; on the shared nodes the dust SED is identical.
    w0 = np.asarray(s0.wave)
    in1 = np.isin(wave, w0)
    assert in1.sum() == w0.size
    np.testing.assert_allclose(
        np.asarray(s1.derived["sed_dust_ir"])[in1],
        np.asarray(s0.derived["sed_dust_ir"]),
        rtol=1e-8,  # grid-dependent normalization, present with or without the edge change
    )
    assert float(s1.derived["L_ir"]) == float(s0.derived["L_ir"])
    assert float(np.asarray(s1.derived["radio_L_ir_input"])) * 3.828e33 == pytest.approx(
        band_l_ir(s1, "tir"), rel=1e-4
    )


def test_final_sed_l_tir_gains_the_sub_mm_synchrotron(synthetic_ssp_wide, synthetic_tophat_obs):
    """``l_tir`` (8-1000 um of the final SED) now holds the synchrotron between 100 and 1000 um.

    It is a share ~1e-5 of the TIR input: with L_nu(nu_ref) = L_TIR / (3.75e12 10^q) and a
    power law, the 100-1000 um integral is
    L_TIR / (3.75e12 10^q) nu_ref^alpha [nu^(1-alpha) / (1-alpha)] from 0.3 to 3 THz.
    The energy-balance ``L_ir`` and ``sed_dust_ir`` are not affected (test above).
    """
    from tengri.utils.sed_quantities import compute_l_tir

    def l_tir(model):
        s = model.predict_state(model.spec.sample(jax.random.PRNGKey(0)))
        return float(compute_l_tir(s.sed_intrinsic, s.wave)), s

    with_r, without = (
        _build(synthetic_ssp_wide, synthetic_tophat_obs, with_radio=r) for r in (True, False)
    )
    l1, s1 = l_tir(with_r)
    l0, _ = l_tir(without)
    l_in = float(np.asarray(s1.derived["radio_L_ir_input"])) * 3.828e33  # erg/s
    nu_lo, nu_hi = _C_AA / 1.0e7, _C_AA / 1.0e6
    analytic = (
        l_in
        / (3.75e12 * 10.0**_Q)
        * 1.4e9**_ALPHA
        * (nu_hi ** (1 - _ALPHA) - nu_lo ** (1 - _ALPHA))
        / (1 - _ALPHA)
    )
    gain = (l1 - l0) * 3.828e33  # erg/s
    assert gain == pytest.approx(analytic, rel=0.03)
    assert 1e-6 < gain / l_in < 3e-5


def _thermal_share():
    """Murphy+2011 free-free over Bell (2003) Eq. 1 at 1.4 GHz, q = 2.58 (L_IR cancels)."""
    return 3.88e-44 / 4.6e-28 * 1.4**-0.1 * 3.75e12 * 10.0**_Q


def test_final_sed_l_tir_gain_includes_free_free(synthetic_ssp_wide, synthetic_tophat_obs):
    """Free-free on (the default): l_tir gains the synchrotron (total minus thermal) and the
    Murphy term over 100-1000 um, both analytic power-law integrals."""
    from tengri.utils.sed_quantities import compute_l_tir

    def l_tir(model):
        s = model.predict_state(model.spec.sample(jax.random.PRNGKey(0)))
        return float(compute_l_tir(s.sed_intrinsic, s.wave)), s

    on = _build(synthetic_ssp_wide, synthetic_tophat_obs, sf_extra={"freefree": True})
    off = _build(synthetic_ssp_wide, synthetic_tophat_obs, with_radio=False)
    l1, s1 = l_tir(on)
    l0, _ = l_tir(off)
    l_in = float(np.asarray(s1.derived["radio_L_ir_input"])) * 3.828e33
    nu_lo, nu_hi = _C_AA / 1.0e7, _C_AA / 1.0e6
    ref = l_in / (3.75e12 * 10.0**_Q)  # total L_nu at 1.4 GHz
    ff_ref = _thermal_share() * ref
    sync = (ref - ff_ref) * 1.4e9**_ALPHA * (nu_hi ** (1 - _ALPHA) - nu_lo ** (1 - _ALPHA))
    sync /= 1 - _ALPHA
    ff = ff_ref * 1.4e9**0.1 * (nu_hi**0.9 - nu_lo**0.9) / 0.9  # L_nu ~ nu^-0.1
    assert (l1 - l0) * 3.828e33 == pytest.approx(sync + ff, rel=0.03)
    # The energy-balance L_ir is the dust block's: free-free does not touch it.
    _, s0 = l_tir(off)
    assert float(s1.derived["L_ir"]) == float(s0.derived["L_ir"])


@pytest.mark.parametrize("sf", [{"type": "bell2003"}, {"type": "bell2003_split"}])
def test_bare_ssp_radio_block_is_continuous_through_1_mm(ssp_data_bc03, synthetic_tophat_obs, sf):
    """Default radio block on a bare SSP, no nebular continuum: no step at 1 mm.

    The step was 6.7x (bell2003) and 4.6x (split) with the 1 mm free-free edge. Across the
    grid cell around 1 mm the block now changes by its power-law slope only.
    """
    del synthetic_tophat_obs
    model = _build_bare(ssp_data_bc03, sf)
    state = model.predict_state(model.spec.sample(jax.random.PRNGKey(0)))
    wave = np.asarray(state.wave)
    sed = np.asarray(state.derived["sed_radio"])
    lo = int(np.nonzero(wave <= 1.0e7)[0][-1])
    nu_ratio = wave[lo + 1] / wave[lo]
    ratio = sed[lo + 1] / sed[lo]
    # L ~ wave^alpha_eff, alpha_eff in [-0.1, 0.8]: ratio between the two pure power laws
    assert nu_ratio**-0.1 <= ratio <= nu_ratio**0.8, (ratio, nu_ratio)


def _build_bare(ssp, sf, neb=None):
    from tengri.observation import Photometry

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp,
            observation=Photometry.from_names(["sdss_g", "sdss_r"]),
            redshift=Fixed(0.0),
            sfh={"type": "const", "all_params": FREE},
            dust_attenuation={"type": "two_component", "law": "calzetti"},
            dust_emission={"type": "draine_li2014"},
            **({"neb": neb} if neb is not None else {}),
            radio={"sf": sf, "agn": {"type": "none"}},
        )


@pytest.mark.parametrize("neb", ["cue", "cloudy"])
def test_free_free_gate_holds_with_a_nebular_continuum(ssp_data_bc03, neb):
    """A nebular backend carrying free-free switches the radio free-free off at every node:
    the radio block is the synchrotron alone, (1 - f_th) of the ``freefree: False`` one."""
    from tengri.components.radio.component import RadioSEDComponentConfig as Cfg  # noqa: F401

    key = jax.random.PRNGKey(0)
    gated = _build_bare(ssp_data_bc03, {"type": "bell2003"}, neb={"type": neb})
    plain = _build_bare(ssp_data_bc03, {"type": "bell2003", "freefree": False}, neb={"type": neb})
    s_g = gated.predict_state(gated.spec.sample(key))
    s_p = plain.predict_state(plain.spec.sample(key))
    wave = np.asarray(s_g.wave)
    a, b = np.asarray(s_g.derived["sed_radio"]), np.asarray(s_p.derived["sed_radio"])
    on = wave >= 1.0e6
    assert on.sum() > 100
    q64 = 3.88e-44 / 4.6e-28 * 1.4**-0.1 * 3.75e12 * 10.0**2.64
    np.testing.assert_allclose(a[on], (1.0 - q64) * b[on], rtol=1e-10)
    assert np.all(a[~on] == 0.0)
