# SPDX-License-Identifier: BSD-3-Clause
"""Rest-frame band fluxes keep the dust emission under WavePrecomp (#2559).

``phot_rest_fnu`` is the SED reprojected at z=0, so the rest band is the unit-luminosity
integral at redshift 0. The rest-band projector sums the ``*_restband_lnu_precomp`` family,
and the dust emission component published only the observed-band key
``dust_emission_phot_lnu_precomp``. The infrared was therefore missing from rest-frame
photometry on the LUT path while the exact path kept it.

Measured on a single-screen Calzetti model (tau_v = 1.5, DPL history, ``neb`` off,
``redshift=Fixed(1.0)``, ``dust_emission={"type": "dale2014"}``), before the fix,
``phot_rest_fnu`` WavePrecomp against exact, as ``wave/exact - 1``: irac_36 -0.15,
irac_80 -0.91; the optical rest bands were within a few per cent. Observed-band
``phot_fnu`` was unaffected.

This module pins the rest-band agreement with the exact path, that the dust emission
share in rest irac_80 is large enough that dropping it cannot pass, and that the observed
bands did not move.

Every accuracy assertion compares the LUT against the exact path (``approx=None``).
"""

import os
from pathlib import Path

import jax
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel, Uniform, WavePrecomp

pytestmark = pytest.mark.regression_bug

KEY = jax.random.PRNGKey(0)

#: The FSPS PARSEC/C3K grid the issue measured on.
_SSP_NAME = "fsps_prsc_c3k_a_chabrier.h5"
_DATA_DIR = Path(os.environ.get("TENGRI_DATA_DIR", "data"))

BANDS = ["hst_f435w", "hst_f814w", "hst_f160w", "irac_36", "irac_80"]
IRAC_36, IRAC_80 = 3, 4

#: Worst per-band rest-frame relative error of the LUT against exact, after the fix,
#: times two (measured below; never below 1e-3).
_RTOL_REST = 1e-3

#: Observed-band tolerance, the same one the #1148 test uses.
_RTOL_OBS = 0.02


@pytest.fixture(scope="module")
def ssp():
    path = _DATA_DIR / _SSP_NAME
    if not path.is_file():
        pytest.skip(f"SSP data not found: {path}")
    from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

    return load_ssp_data(str(path))


def _build(ssp, *, dust_emission, approx) -> SEDModel:
    return SEDModel.build(
        ssp_data=ssp,
        observation=Observation(photometry=Photometry.from_names(BANDS)),
        sfh={
            "type": "dpl",
            "all_params": Fixed(DEFAULT),
            "log_total_mass": Uniform(9.0, 11.0),
        },
        dust_attenuation={
            "type": "single_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
            "tau_v": Fixed(1.5),
        },
        dust_emission=dust_emission,
        neb={"type": "none"},
        redshift=Fixed(1.0),
        approx=approx,
    )


DUSTY = {"type": "dale2014", "all_params": Fixed(DEFAULT)}
NO_EMISSION = {"type": "none"}


def test_rest_frame_photometry_agrees_with_exact_on_every_band(ssp):
    """The LUT's rest-frame fluxes match the exact path, including the dust IR."""
    m_exact = _build(ssp, dust_emission=DUSTY, approx=None)
    p = dict(m_exact.spec.sample(KEY))
    exact = np.asarray(m_exact.predict_observables(p).phot_rest_fnu)
    lut = np.asarray(
        _build(ssp, dust_emission=DUSTY, approx=WavePrecomp()).predict_observables(p).phot_rest_fnu
    )
    ratio = lut / exact - 1.0
    print(f"\n#2559 phot_rest_fnu WavePrecomp/exact - 1 per band {BANDS}: {ratio}")
    worst = float(np.max(np.abs(ratio)))
    assert worst < _RTOL_REST, (
        f"rest-frame LUT is {worst:.3e} off the exact path (per band {BANDS}: {ratio})"
    )


def test_rest_frame_dust_emission_is_material_in_irac_80(ssp):
    """Dropping the dust emission from the rest band must be visible at test tolerance.

    Without this, a LUT that omits the emission on both sides could still pass (a).
    """
    m_dusty = _build(ssp, dust_emission=DUSTY, approx=None)
    m_none = _build(ssp, dust_emission=NO_EMISSION, approx=None)
    p = dict(m_dusty.spec.sample(KEY))
    dusty = np.asarray(m_dusty.predict_observables(p).phot_rest_fnu)
    bare = np.asarray(m_none.predict_observables(p).phot_rest_fnu)
    share = abs(dusty[IRAC_80] / bare[IRAC_80] - 1.0)
    print(f"\n#2559 rest irac_80 dust-emission share: {share:.3e}")
    assert share > 10.0 * _RTOL_REST, (
        f"dust emission changes rest irac_80 by only {share:.3e}; the test cannot tell "
        "a dropped emission from a correct one"
    )


def test_observed_photometry_is_untouched(ssp):
    """phot_fnu is the likelihood channel. The rest-band fix must not move it."""
    m_exact = _build(ssp, dust_emission=DUSTY, approx=None)
    p = dict(m_exact.spec.sample(KEY))
    exact = np.asarray(m_exact.predict_photometry(p))
    lut = np.asarray(_build(ssp, dust_emission=DUSTY, approx=WavePrecomp()).predict_photometry(p))
    worst = float(np.max(np.abs(lut / exact - 1.0)))
    assert worst < _RTOL_OBS, f"observed photometry moved {worst * 100:.2f}%"
