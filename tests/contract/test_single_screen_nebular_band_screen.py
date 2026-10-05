# SPDX-License-Identifier: BSD-3-Clause
"""Contract: the single-screen dust component reddens the nebular band flux at
the wavelengths of the emission, not at the filter's effective wavelength.

Nebular emission is line-dominated and a line sits where it sits. Under
`approx=WavePrecomp()`, the dust screen must be applied to the nebular
continuum integrated through each band where the emission is (the sub-band
sum), not at the filter effective wavelength.

Before the screen was applied at the emission, `WavePrecomp()` against the
exact model (`model.with_approx(None)`) deviated for single-component Calzetti
dust with Cue nebular emission (paper configuration II, z = 1.012, tau_v = 2.32)
by 1.9e-2 (F606W, F814W), 4.0e-2 (F850LP) and 3.8e-2 (F125W); with every optical
depth at zero the same comparison gave 1.3e-4.

Measured worst relative deviations at the tested point (z = 1, tau_v = 2.3,
age 0.05 Gyr, neb_logU = -2.5, six bands including the IRAC 3.6 micron band)
and the tolerances set (twice the measurement, never below 1e-3):
observed frame 1.2e-3 (young) and 2.1e-5 (control) with tolerances 2.5e-3 and
1e-3; grid-served 2.4e-3 and 1.4e-3 with 4.8e-3 and 2.8e-3; rest frame 1.2e-3
with 2.4e-3. The rest-frame models carry no dust emission: the rest-band dust
emission of the WavePrecomp path deviates from the exact model on its own
(2 to 6 percent in the optical, 0.6 at 3.6 micron at this point) and would mask
the nebular screen.

Data-gated like `tests/contract/test_dusty_nebular_grid_wiring.py`: with
TENGRI_DATA_DIR set every test runs.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import jax
import numpy as np
import pytest

from tengri import (
    DEFAULT,
    Fixed,
    Observation,
    Photometry,
    SEDModel,
    Uniform,
    WavePrecomp,
    load_ssp_data,
)
from tengri.inference import Fitter

pytestmark = pytest.mark.contract

_BARE = "data/fsps_prsc_miles_chabrier.h5"
_DUST = {
    "type": "single_component",
    "law": "calzetti",
    "all_params": Fixed(DEFAULT),
    "tau_v": Uniform(0.0, 3.0),
}
_BANDS = ["hst_f606w", "hst_f814w", "hst_f850lp", "hst_f125w", "hst_f160w", "irac_36"]


def _worst(a, b) -> float:
    """Largest relative deviation of ``a`` from ``b`` across bands."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.max(np.abs(a - b) / np.abs(b)))


def _require():
    for f in (_BARE, "data/cue_weights.npz"):
        if not Path(f).is_file():
            pytest.skip(f"missing {f}")


def _model(ssp, *, approx, dust, dust_emission=True):
    extra = (
        {"dust_emission": {"type": "dale2014", "all_params": Fixed(DEFAULT)}}
        if dust_emission
        else {}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp,
            observation=Observation(photometry=Photometry.from_names(_BANDS)),
            redshift=Fixed(1.0),
            sfh={
                "type": "dpl",
                "all_params": Fixed(DEFAULT),
                "age_gyr": Uniform(0.05, 2.0),
                "log_total_mass": Uniform(8, 12),
            },
            dust_attenuation=dust,
            **extra,
            neb={"type": "cue", "all_params": Fixed(DEFAULT), "neb_logU": Uniform(-3.5, -2.0)},
            approx=approx,
            met={"logzsol": Fixed(0.0)},
        )


@pytest.fixture(scope="module")
def ssp():
    _require()
    return load_ssp_data(_BARE)


@pytest.fixture(scope="module")
def m_wave(ssp):
    return _model(
        ssp,
        approx=WavePrecomp(),
        dust=_DUST,
    )


@pytest.fixture(scope="module")
def m_exact(ssp):
    return _model(
        ssp,
        approx=None,
        dust=_DUST,
    )


@pytest.fixture(scope="module")
def m_wave_rest(ssp):
    """WavePrecomp model without dust emission, for the rest-frame comparison."""
    return _model(ssp, approx=WavePrecomp(), dust=_DUST, dust_emission=False)


@pytest.fixture(scope="module")
def m_exact_rest(ssp):
    """Exact twin of ``m_wave_rest``."""
    return _model(ssp, approx=None, dust=_DUST, dust_emission=False)


@pytest.fixture(scope="module")
def p_warm(m_wave):
    """Young, dusty point: age at the young end, tau_v = 2.3, neb_logU = -2.5."""
    p = dict(m_wave.spec.sample(jax.random.PRNGKey(0)))
    p["sfh_dpl_age_gyr"] = 0.05
    p["dust_tau_v"] = 2.3
    p["neb_logU"] = -2.5
    return p


@pytest.fixture(scope="module")
def p_control(p_warm):
    """Control: same parameters with dust_tau_v = 0."""
    p = dict(p_warm)
    p["dust_tau_v"] = 0.0
    return p


def test_nebular_shares_at_young_point(m_wave, p_warm):
    """Report the nebular band shares at the young point for documentation."""
    state = m_wave.predict_state(p_warm)
    neb_shares = state.derived["nebular_phot_lnu_precomp"] / (
        state.derived.get("stellar_phot_lnu_precomp", 0.0)
        + state.derived["nebular_phot_lnu_precomp"]
    )
    print(f"Nebular band shares at young point: {neb_shares}")


def test_wave_precomp_matches_the_exact_model_under_a_thick_screen(
    m_wave, m_exact, p_warm, p_control
):
    """WavePrecomp flux matches the exact path at the young point and control.

    The screen is applied at the emission wavelengths, not at lambda_eff.
    Measured worst relative deviation: 1.238e-3 (young), 2.071e-5 (control).
    """
    m_exact.predict_photometry(p_warm)
    dev_warm = _worst(m_wave.predict_photometry(p_warm), m_exact.predict_photometry(p_warm))
    dev_ctrl = _worst(m_wave.predict_photometry(p_control), m_exact.predict_photometry(p_control))
    print(f"MEASURED test1 young={dev_warm:.3e} control={dev_ctrl:.3e}")
    assert dev_warm <= 2.5e-3, dev_warm
    assert dev_ctrl <= 1e-3, dev_ctrl


def test_grid_served_single_screen_matches_the_exact_model(m_wave, m_exact, p_warm, p_control):
    """Grid-served single screen matches exact path after Fitter attaches the grid.

    Fitter(..., approx='auto') attaches a grid that serves dust, which flags
    the dust component nebular_from_grid=True and zeroes sed_nebular. This
    test verifies the grid-based path also matches exact. Measured worst
    relative deviation: 2.365e-3 (young), 1.390e-3 (control).
    """
    flux = m_wave.predict_photometry(p_warm)
    m_fast = Fitter(
        m_wave, data=flux, noise=0.05 * flux, data_type="photometry", approx="auto"
    ).model

    assert m_fast._nebular_grid_table.serves_dust

    dev_warm = _worst(m_fast.predict_photometry(p_warm), m_exact.predict_photometry(p_warm))
    dev_ctrl = _worst(m_fast.predict_photometry(p_control), m_exact.predict_photometry(p_control))
    print(f"MEASURED test2 young={dev_warm:.3e} control={dev_ctrl:.3e}")
    assert dev_warm <= 4.8e-3, dev_warm
    assert dev_ctrl <= 2.8e-3, dev_ctrl


def test_rest_band_twin_gets_the_same_screen(m_wave_rest, m_exact_rest, p_warm):
    """The rest-frame band photometry carries the same screen at the emission.

    ``phot_rest_fnu`` of the WavePrecomp model (served by the LUT projector)
    against the exact model at the young point. Measured worst relative
    deviation: 1.196e-3. The models carry no dust
    emission so the comparison isolates the nebular screen.
    """
    rest_wave = np.asarray(m_wave_rest.predict_observables_jit(p_warm).phot_rest_fnu)
    rest_exact = np.asarray(m_exact_rest.predict_observables(p_warm).phot_rest_fnu)
    dev = _worst(rest_wave, rest_exact)
    print(f"MEASURED test3 young={dev:.3e}")
    assert dev <= 2.4e-3, dev


def test_published_screened_flux_is_the_band_integral_of_the_reddened_continuum(m_wave, p_warm):
    """The published nebular_phot_lnu_attenuated_precomp keys are present and correct.

    The dust component must publish:
    - nebular_phot_lnu_attenuated_precomp: the reddened nebular continuum
      integrated through each observed-frame band.
    - nebular_restband_lnu_attenuated_precomp: the rest-frame twin.

    Both must be strictly less than the unscreened flux (the dust is applied),
    and the ratio must differ from the unscreened attenuation factor by >1e-3
    in at least one band (the screen is evaluated at the emission, not lambda_eff).
    """
    st = m_wave.predict_state(p_warm)

    assert "nebular_phot_lnu_attenuated_precomp" in st.derived
    assert "nebular_restband_lnu_attenuated_precomp" in st.derived

    neb_unreddened = st.derived["nebular_phot_lnu_precomp"]
    neb_reddened = st.derived["nebular_phot_lnu_attenuated_precomp"]
    a_eff = st.derived["dust_attenuation_precomp"]

    assert np.all(neb_reddened < neb_unreddened)

    ratio_screen_at_emission = neb_reddened / neb_unreddened
    max_diff = np.max(np.abs(ratio_screen_at_emission - a_eff))
    assert max_diff > 1e-3, (
        f"Screen at emission differs from screen at lambda_eff by only {max_diff:.3e}; "
        "the fix may not be working."
    )
