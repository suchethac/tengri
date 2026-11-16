# SPDX-License-Identifier: BSD-3-Clause
"""#3541: the rest-band photometry reads the SSP metallicity by the native-slope Hermite.

The rest band (``phot_rest_fnu``, and so ``Observables.mag_absolute``) was read across
metallicity by PCHIP of its own stored values, with no native-slope sibling, so an
off-node log Z moved it off the exact path by 1.25e-4 in GALEX FUV on the real SSP. The
rest-band tables now carry their slope sibling, and the read is the Hermite of both.

Every accuracy assertion compares the LUT against the exact path (approx=None), as in
``test_bug_1148_restband_photometry.py``. The metallicity is off the grid nodes: the
absolute log Z is met_logzsol - 1.848 (solar-relative), so -0.052 is log Z -1.900, between
the -2.0 and -1.854 nodes of ``fsps_prsc_miles_chabrier``.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel, WavePrecomp

pytestmark = pytest.mark.regression_bug

#: Relative tolerance of a table read against the exact read (round-off).
RTOL = 1e-12
_BANDS = ["galex_fuv", "sdss_g", "sdss_r", "2mass_ks"]
#: Off-node met_logzsol values (absolute log Z -1.900 and -2.148).
_OFF_NODE = (-0.052, -0.300)


def _build(ssp, approx, met_logzsol):
    return SEDModel.build(
        ssp_data=ssp,
        observation=Observation(photometry=Photometry.from_names(_BANDS)),
        approx=approx,
        redshift=Fixed(0.1),
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(1.0),
            "age_frac": Fixed(0.6),
            "log_total_mass": Fixed(10.0),
            "all_params": Fixed(DEFAULT),
            "age_kernel": "cic",
        },
        met={"type": "delta", "met_logzsol": Fixed(met_logzsol), "all_params": Fixed(DEFAULT)},
        neb={"type": "none"},
        dust_attenuation={"type": "none"},
        dust_emission={"type": "none"},
    )


@pytest.mark.parametrize("met_logzsol", _OFF_NODE)
def test_rest_frame_photometry_matches_exact_path_at_off_node_z(ssp_data_fsps, met_logzsol):
    exact_model = _build(ssp_data_fsps, None, met_logzsol)
    params = dict(exact_model.spec.sample(jax.random.PRNGKey(0)))
    exact = np.asarray(exact_model.predict_observables(params).phot_rest_fnu, dtype=np.float64)
    lut_model = _build(ssp_data_fsps, WavePrecomp(), met_logzsol)
    lut = np.asarray(lut_model.predict_observables(params).phot_rest_fnu, dtype=np.float64)
    rel = np.abs(lut / exact - 1.0)
    assert rel.max() <= RTOL, f"met_logzsol={met_logzsol}: rest band per band {rel}"
