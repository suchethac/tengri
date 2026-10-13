# SPDX-License-Identifier: BSD-3-Clause
r"""The XRB cutoff is fixed at 100 keV and the LMXB age is surviving-mass weighted.

Two defects found by the CIGALE head-to-head (notebook ``01_cigale``):

1. ``xray_total_terms`` handed the AGN corona's ``E_cut`` (``xray_E_cut``, 300 keV
   in the notebook) to ``xray_xrb_terms``, so the HMXB and LMXB spectra were
   rolled over at the corona's cutoff. Yang et al. 2020 (MNRAS 491, 740, Sect.
   2.2.2) fix ``E_cut`` = 100 keV for both XRB populations (citing Zhang 1997 and
   Motta et al. 2009) and use the free ``E_cut`` for the AGN corona only.
2. The LMXB age was weighted by the FORMED mass per SSP age bin, while the LMXB
   luminosity multiplies the SURVIVING stellar mass. Fragos et al. 2013 (ApJL
   776, L31, Eq. 4) give :math:`L_X/M_\star` for the 2-10 keV band against the
   "mass-weighted mean stellar age of the population", the stellar mass that the
   population holds now, so the age and the mass must be weighted the same way.

The 2-10 keV amplitudes are Fragos et al. 2013, ApJL 776, L31, Eqs. 3 and 4 with
the coefficients of their Table 2 (2-10 keV, interstellar absorption included).
"""

import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, SEDModel, SSPData
from tengri.components.xray.component import XRaySEDComponent
from tengri.components.xray.xray import (
    XRB_E_CUT_KEV,
    xray_total_lopez24_terms,
    xray_total_terms,
    xray_xrb_terms,
)
from tengri.utils.physics_constants import LOG10_ZSUN

pytestmark = pytest.mark.regression_bug

_HC_KEV_AA = 12.398419843  # [keV Angstrom]
_ERG_PER_KEV = 1.602176634e-9

#: Fragos et al. 2013 (ApJL 776, L31) Eq. 3, Table 2: beta_0..beta_4, 2-10 keV.
_BETA = (40.28, -62.12, 569.44, -1833.80, 1968.33)
#: Fragos et al. 2013 Eq. 4, Table 2: gamma_0..gamma_4, 2-10 keV.
_GAMMA = (40.276, -1.503, -0.423, 0.425, 0.136)


def _fragos_hmxb(z_metal: float) -> float:
    """Eq. 3: L_X(HMXB, 2-10 keV) per unit SFR [erg/s/(Msun/yr)]."""
    return 10.0 ** sum(b * z_metal**i for i, b in enumerate(_BETA))


def _fragos_lmxb(t_gyr: float) -> float:
    """Eq. 4: L_X(LMXB, 2-10 keV) per 1e10 Msun [erg/s]."""
    lt = np.log10(t_gyr)
    return 10.0 ** sum(g * lt**i for i, g in enumerate(_GAMMA))


def _band_luminosity(l_nu_of_wave, e_lo=2.0, e_hi=10.0, n=20001) -> float:
    """Direct trapezoid of L_nu over [e_lo, e_hi] keV on a dense energy grid [erg/s]."""
    energy = np.linspace(e_lo, e_hi, n)
    wave = _HC_KEV_AA / energy
    l_nu = np.asarray(l_nu_of_wave(jnp.asarray(wave)))
    nu = energy * _ERG_PER_KEV / 6.62607015e-27
    return float(np.trapezoid(l_nu, nu))


_WAVE = jnp.asarray(_HC_KEV_AA / np.geomspace(0.2, 400.0, 600))
_KW = dict(sfr=3.0, stellar_mass=2e10, metallicity_z=0.02, stellar_age_gyr=3.0)


def test_xrb_cutoff_constant_is_100_kev():
    """The named constant holds Yang+2020 Sect. 2.2.2's E_cut for LMXB and HMXB."""
    assert XRB_E_CUT_KEV == 100.0


def test_total_terms_xrb_is_independent_of_the_corona_cutoff():
    """Changing the AGN E_cut leaves the HMXB and LMXB terms bit-identical."""
    base = xray_total_terms(_WAVE, l_2500_30deg=1e29, E_cut=100.0, **_KW)
    other = xray_total_terms(_WAVE, l_2500_30deg=1e29, E_cut=1000.0, **_KW)
    for term in ("hmxb", "lmxb", "hotgas"):
        assert np.array_equal(np.asarray(base[term]), np.asarray(other[term])), term
    # The knob still drives the corona it belongs to.
    assert not np.allclose(np.asarray(base["agn"]), np.asarray(other["agn"]))


def test_lopez24_terms_xrb_is_independent_of_the_corona_cutoff():
    """The lopez24 path shares the XRB terms and the same decoupling."""
    kw = dict(log_l_12um_erg=43.0, **_KW)
    base = xray_total_lopez24_terms(_WAVE, E_cut=100.0, **kw)
    other = xray_total_lopez24_terms(_WAVE, E_cut=1000.0, **kw)
    for term in ("hmxb", "lmxb"):
        assert np.array_equal(np.asarray(base[term]), np.asarray(other[term])), term


@pytest.mark.parametrize(("gamma_key", "term"), [("gamma_hmxb", "hmxb"), ("gamma_lmxb", "lmxb")])
def test_xrb_spectral_shape_rolls_over_at_100_kev(gamma_key, term):
    r"""L_nu(E1)/L_nu(E2) = (E1/E2)^(1-Gamma) exp(-(E1-E2)/100 keV) (Yang+2020 Eq. 2)."""
    gamma = {"gamma_hmxb": 2.0, "gamma_lmxb": 1.56}[gamma_key]
    e1, e2 = 60.0, 20.0
    waves = jnp.asarray([_HC_KEV_AA / e1, _HC_KEV_AA / e2])
    out = np.asarray(xray_total_terms(waves, l_2500_30deg=0.0, E_cut=300.0, **_KW)[term])
    expected = (e1 / e2) ** (1.0 - gamma) * np.exp(-(e1 - e2) / 100.0)
    # 1e-4: the keV <-> Angstrom constant differs from the code's CODATA h*c at 6e-6;
    # a 300 keV cutoff would be off by 33%.
    assert out[0] / out[1] == pytest.approx(expected, rel=1e-4)


def test_xrb_band_normalization_is_the_fragos_luminosity():
    """The 2-10 keV integral of each term equals Eq. 3 (HMXB) and Eq. 4 (LMXB)."""
    terms = xray_xrb_terms(_WAVE, **_KW)
    hmxb = _band_luminosity(lambda w: xray_xrb_terms(w, **_KW)["hmxb"])
    lmxb = _band_luminosity(lambda w: xray_xrb_terms(w, **_KW)["lmxb"])
    assert set(terms) == {"hmxb", "lmxb"}
    assert hmxb == pytest.approx(_fragos_hmxb(0.02) * _KW["sfr"], rel=2e-3)
    assert lmxb == pytest.approx(_fragos_lmxb(3.0) * _KW["stellar_mass"] / 1e10, rel=2e-3)


# ── LMXB age ──────────────────────────────────────────────────────────────

_TAU_GYR = 1.0
_AGE_GYR = 5.0


@pytest.fixture(scope="module")
def ssp_with_mass_remaining():
    """Smooth synthetic SSP with a mass-remaining table, so the age tests run in CI.

    120 log-spaced ages (1 Myr to 13.8 Gyr) keep the CIC age binning within the
    test tolerance of a fine quadrature. The surviving fraction falls from 1 to
    0.55 with log age and varies mildly with metallicity, like a real SSP.
    """
    n_age = 120
    wave = jnp.logspace(2.0, 7.0, 200)
    lg_age = jnp.linspace(-3.0, 1.14, n_age)
    lgmet = jnp.array([-4.0, -2.65, -1.3])
    flux = (5000.0 / wave)[None, None, :] ** 2 * jnp.ones((3, n_age, 1)) + 1e-12
    frac = 1.0 - 0.45 * (lg_age + 3.0) / 4.14
    mass_remaining = frac[None, :] * (1.0 + 0.05 * (lgmet - lgmet.mean()))[:, None]
    return SSPData(
        ssp_wave=wave,
        ssp_flux=flux,
        ssp_lg_age_gyr=lg_age,
        ssp_lgmet=lgmet,
        ssp_mass_remaining=mass_remaining,
    )


def _model(ssp, logzsol):
    return SEDModel.build(
        ssp_data=ssp,
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(_TAU_GYR),
            "age_gyr": Fixed(_AGE_GYR),
            "log_total_mass": Fixed(10.0),
            "all_params": Fixed(DEFAULT),
        },
        met={"type": "delta", "logzsol": Fixed(logzsol), "all_params": Fixed(DEFAULT)},
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "tau_bc": Fixed(0.0),
            "tau_diff": Fixed(0.0),
            "all_params": Fixed(DEFAULT),
        },
        neb={"type": "none"},
        xray={"type": "yang20", "all_params": Fixed(DEFAULT)},
        redshift=Fixed(0.0),
    )


def test_lmxb_age_is_weighted_by_surviving_mass(ssp_with_mass_remaining):
    r"""The LMXB age is :math:`\int t\,\mathrm{SFR}\,m\,dt / \int \mathrm{SFR}\,m\,dt`.

    ``m`` is the SSP surviving-mass fraction at the galaxy's metallicity, the same
    mass that multiplies the LMXB luminosity (Fragos+2013 Eq. 4, "mass-weighted
    mean stellar age of the population"). Independent reference: a 2e5-point
    quadrature of a delayed-tau SFH against the SSP mass-remaining row. The
    formed-mass weighting differs from it by more than 1%.
    """
    ssp = ssp_with_mass_remaining
    node = 1
    logzsol = float(np.asarray(ssp.ssp_lgmet)[node]) - LOG10_ZSUN
    state = _model(ssp, logzsol).predict_state({})
    age = float(XRaySEDComponent().emitter_inputs(state.derived)["stellar_age_gyr"])

    ages = 10.0 ** np.asarray(ssp.ssp_lg_age_gyr)
    lookback = np.linspace(0.0, _AGE_GYR, 200001)
    sfr = (_AGE_GYR - lookback) * np.exp(-(_AGE_GYR - lookback) / _TAU_GYR)
    m_rem = np.interp(lookback, ages, np.asarray(ssp.ssp_mass_remaining)[node])
    surviving = np.trapezoid(lookback * sfr * m_rem, lookback) / np.trapezoid(
        sfr * m_rem, lookback
    )
    formed = np.trapezoid(lookback * sfr, lookback) / np.trapezoid(sfr, lookback)

    assert abs(formed / surviving - 1.0) > 0.010  # the two definitions are distinguishable
    assert age == pytest.approx(surviving, rel=2e-3)


def test_lmxb_age_and_mass_share_one_weighting(ssp_with_mass_remaining):
    """The age weights sum to the surviving mass the LMXB luminosity multiplies."""
    state = _model(ssp_with_mass_remaining, -0.5).predict_state({})
    weights = np.asarray(state.derived["age_weights_surviving"])
    surviving = 10.0 ** float(state.derived["log_mstar_surviving"])
    assert weights.sum() == pytest.approx(surviving, rel=1e-9)
