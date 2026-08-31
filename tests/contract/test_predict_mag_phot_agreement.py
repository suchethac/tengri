# SPDX-License-Identifier: BSD-3-Clause
"""Issue #436: predict_magnitudes and predict_photometry must agree.

Before the fix, ``predict_magnitudes`` routed through
``dsps.calc_obs_mag`` which uses Bessell & Murphy 2012's
photon-counting form (``int T F_nu dlambda/lambda``), while
``predict_photometry`` uses Tokunaga & Vacca 2005's form
(``int lambda T F_nu dlambda``). The two conventions give
5--40 mmag offsets in SDSS bands -- a band-dependent
zero-point error larger than typical photometric precision.

``predict_magnitudes`` now derives from ``predict_photometry``
via the AB definition so the two are consistent by construction.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

import tengri

pytestmark = pytest.mark.contract


@pytest.fixture(scope="module")
def model(synthetic_ssp_wide, synthetic_tophat_obs):
    # #613: synthetic SSP + synthetic filters so this mag↔phot self-consistency
    # check runs on CI (both APIs share the same filter convolution regardless
    # of SSP, so the agreement is SSP-independent).
    return tengri.SEDModel.build(
        synthetic_ssp_wide,
        observation=synthetic_tophat_obs,
        sfh={"type": "tsnorm", "all_params": tengri.Fixed(tengri.DEFAULT)},
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "all_params": tengri.Fixed(tengri.DEFAULT),
            "tau_diff": 0.3,
            "tau_bc": 0.2,
        },
        redshift=tengri.Fixed(0.05),
    )


def test_predict_mag_matches_predict_phot_within_one_mmag(model):
    """Issue #436: two APIs must agree within float64 round-off.

    The issue used ``-48.6`` for the AB zeropoint; tengri uses the
    exact ``MAGGIES_ZP_CGS = 3.631e-20`` (which is ~48.6002),
    so the residual is the ~0.07 mmag difference between those two
    zero-point conventions -- well under the 1 mmag tolerance set
    by the issue.
    """
    p = dict(model.spec.sample(jax.random.PRNGKey(0)))
    flux = np.asarray(model.predict_photometry(p))
    mag = np.asarray(model.predict_magnitudes(p))
    mag_from_flux = -2.5 * np.log10(flux) - 48.6
    diff_mmag = (mag_from_flux - mag) * 1000.0
    assert np.max(np.abs(diff_mmag)) < 1.0, (
        f"predict_magnitudes vs predict_photometry diverge by "
        f"{np.max(np.abs(diff_mmag)):.3f} mmag (per-band: {diff_mmag})"
    )
