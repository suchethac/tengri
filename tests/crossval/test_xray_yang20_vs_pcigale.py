# SPDX-License-Identifier: BSD-3-Clause
"""Cross-validate the galaxy X-ray terms against CIGALE's ``yang20`` module.

Both codes receive the same galaxy (SFR over 100 Myr, surviving stellar mass,
mass-weighted age, metallicity) and the notebook's X-ray configuration
(``gam`` = 1.8, ``E_cut`` = 300 keV, no AGN disc, so the corona is zero). The
XRB + hot-gas spectrum, tengri over CIGALE, is compared at 1 and 5 keV.

Before the XRB cutoff was decoupled from ``xray_E_cut`` it was 300 keV, against
CIGALE's fixed 100 keV (Yang et al. 2020, MNRAS 491, 740, Sect. 2.2.2), and the
ratio at 1 keV for this galaxy was 0.9697 (the notebook head-to-head, with its
own galaxy, went 0.9599 to 0.9817). The remaining ~1% is CIGALE's band
normalization, which sums only the grid points inside the 2-10 keV band
(``yang20.py``, ``np.trapezoid`` over ``lam_idxs`` on a 1000-point grid), against
tengri's integral of the same band on a finer grid. It is a property of
CIGALE's grid, not of the model.

Skipped unless ``pcigale`` is installed (``pytest -m crossval``).
"""

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.xray.component import XRaySEDComponent

pytestmark = [pytest.mark.crossval, pytest.mark.regression_paper]

pytest.importorskip("pcigale")

_C_NM = 2.99792458e17  # speed of light [nm/s]
_HC_KEV_AA = 12.398419843  # [keV Angstrom]
_ERG_PER_JOULE = 1.0e7

#: The galaxy fed to both codes.
_SFR_100 = 3.0  # [Msun/yr]
_M_STAR = 5.0e10  # [Msun], surviving
_AGE_MYR = 3127.0  # mass-weighted age [Myr]
_Z = 0.02

#: Notebook ``yang20`` configuration; ``alpha_ox`` is inert without an AGN disc.
_CIGALE = dict(
    gam=1.8,
    E_cut=300.0,
    alpha_ox=-1.4,
    max_dev_alpha_ox=0.2,
    angle_coef="0.5 & 0",
    det_lmxb=0.0,
    det_hmxb=0.0,
)
_TENGRI = {
    "xray_gamma_agn": 1.8,
    "xray_gamma_hmxb": 2.0,
    "xray_gamma_lmxb": 1.56,
    "xray_E_cut": 300.0,
    "xray_delta_alpha_ox": 0.0,
    "xray_log_nh": 20.0,
    "xray_alpha_irx": 0.3,
    "xray_det_hmxb": 0.0,
    "xray_det_lmxb": 0.0,
}


def _cigale_galaxy_lnu() -> tuple[np.ndarray, np.ndarray]:
    """CIGALE's ``xray.galaxy`` contribution: ``(wavelength [nm], L_nu [erg/s/Hz])``."""
    from pcigale.sed import SED
    from pcigale.sed_modules import get_module

    sed = SED()
    sed.add_info("sfh.sfr100Myrs", _SFR_100)
    sed.add_info("stellar.m_star", _M_STAR)
    sed.add_info("stellar.age_m_star", _AGE_MYR)
    sed.add_info("stellar.metallicity", _Z)
    get_module("yang20", **_CIGALE).process(sed)
    wave_nm = np.asarray(sed.wavelength_grid)
    l_lambda = np.asarray(sed.luminosities["xray.galaxy"]) * _ERG_PER_JOULE  # [erg/s/nm]
    return wave_nm, l_lambda * wave_nm**2 / _C_NM


def _tengri_galaxy_lnu(wave_aa: np.ndarray) -> np.ndarray:
    terms = XRaySEDComponent().emission_terms(
        _TENGRI,
        jnp.asarray(wave_aa),
        sfr=jnp.asarray(_SFR_100),
        stellar_mass=jnp.asarray(_M_STAR),
        stellar_age_gyr=jnp.asarray(_AGE_MYR / 1000.0),
        metallicity_z=jnp.asarray(_Z),
        l_2500=jnp.asarray(0.0),
        cos_inc=jnp.asarray(0.9),
        log_l_12um=jnp.asarray(-jnp.inf),
    )
    return np.asarray(terms["hmxb"] + terms["lmxb"] + terms["hotgas"])


@pytest.mark.parametrize("energy_kev", [1.0, 5.0])
def test_galaxy_xray_matches_pcigale_yang20(energy_kev):
    """tengri/CIGALE L_nu at 1 and 5 keV is 0.991 +/- 0.004 (CIGALE's grid-truncated band norm)."""
    wave_nm, l_nu_cigale = _cigale_galaxy_lnu()
    j = int(np.argmin(np.abs(wave_nm - _HC_KEV_AA / energy_kev / 10.0)))
    l_nu_tengri = _tengri_galaxy_lnu(wave_nm[j : j + 1] * 10.0)[0]
    ratio = float(l_nu_tengri / l_nu_cigale[j])
    assert ratio == pytest.approx(0.991, abs=0.004), f"{energy_kev} keV: {ratio:.4f}"
