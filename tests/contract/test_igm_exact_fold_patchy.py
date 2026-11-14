# SPDX-License-Identifier: BSD-3-Clause
"""``igm_fold='exact'`` serves patchy reionization and DLA absorption (#3505, row B20).

The exact fold used to refuse any transmission with a free parameter. It now folds
the fixed part exactly (the mean IGM, or the Inoue base of the patchy model) into
the sub-band quadrature, and multiplies the free factor (the damping wing and the
DLA absorber) at the exact fold's own sub-band nodes at every call. The contract:

* the build succeeds and ``igm_fold='auto'`` resolves to the exact fold;
* at zero patchiness the result equals the plain exact fold to round-off;
* at non-zero patchiness the error against the dense wavelength-grid integrator
  (``approx=None``) is no larger than the error of the node fold.

Bare-stellar models throughout: no dust and no nebular group, so the comparison
isolates the IGM, as in ``test_igm_exact_fold.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

import tengri
from tengri import DEFAULT, SEDModel, WavePrecomp
from tengri.parameters import Fixed

pytestmark = pytest.mark.contract

#: Sub-band count for the LUT under test. The exact fold needs the sub-band tensors.
N_SUBBANDS = 3

#: Patchy case: z = 6.5 puts Lyman-alpha (9120 A observed) inside the sdss_z band, so
#: the damping wing lands on the band and the node fold has a real error to lose.
PATCHY_Z = 6.5
PATCHY_BANDS = ["sdss_z", "2mass_j"]
PATCHY_X_HI = 0.5

#: DLA case: at z = 1 the Lyman limit (1824 A observed) lies inside galex_nuv, and
#: the absorber sits at the source redshift.
DLA_Z = 1.0
DLA_BANDS = ["galex_nuv", "sdss_r"]
DLA_LOG_N_HI = 21.5


@pytest.fixture(scope="module")
def ssp(ssp_data_wne):
    return ssp_data_wne


def _sfh(age_gyr):
    """Delayed SFH with its onset before the universe's age at the model redshift."""
    return {
        "type": "delayed",
        "all_params": Fixed(DEFAULT),
        "tau_gyr": Fixed(0.2),
        "age_gyr": Fixed(age_gyr),
        "log_total_mass": Fixed(10.0),
        "met_logzsol": Fixed(0.0),
    }


def _model(ssp, bands, z, igm, approx):
    # Onset at 0.5 Gyr is before age(z=6.5) = 0.84 Gyr and before age(z=1) = 5.8 Gyr.
    return SEDModel.build(
        ssp_data=ssp,
        observation=tengri.Observation(photometry=tengri.Photometry.from_names(bands)),
        sfh=_sfh(0.5),
        redshift=Fixed(z),
        igm=igm,
        approx=approx,
    )


def _phot(model):
    return np.asarray(model.predict_photometry({}), dtype=np.float64)


def _patchy_igm(x_hi):
    return {"type": "inoue", "patchy": True, "x_HI": Fixed(x_hi), "bubble_mpc": Fixed(1.0)}


def _dla_igm():
    return {"type": "inoue", "dla": True, "log_n_hi": Fixed(DLA_LOG_N_HI)}


def _exact(fold_n=N_SUBBANDS):
    return WavePrecomp(igm_fold="exact", n_subbands=fold_n)


def _node(fold_n=N_SUBBANDS):
    return WavePrecomp(igm_fold="node", n_subbands=fold_n)


# --- patchy reionization -------------------------------------------------------


def test_patchy_exact_fold_builds_and_is_finite(ssp):
    """The exact fold no longer refuses a free patchy transmission."""
    f = _phot(_model(ssp, PATCHY_BANDS, PATCHY_Z, _patchy_igm(PATCHY_X_HI), _exact()))
    assert np.all(np.isfinite(f)) and np.all(f > 0)


def test_patchy_auto_resolves_to_the_exact_fold(ssp):
    """``auto`` takes the exact fold for patchy too: the same numbers as ``exact``."""
    auto = _phot(
        _model(
            ssp,
            PATCHY_BANDS,
            PATCHY_Z,
            _patchy_igm(PATCHY_X_HI),
            WavePrecomp(n_subbands=N_SUBBANDS),
        )
    )
    exact = _phot(_model(ssp, PATCHY_BANDS, PATCHY_Z, _patchy_igm(PATCHY_X_HI), _exact()))
    np.testing.assert_array_equal(auto, exact)


def test_zero_patchiness_equals_the_plain_exact_fold(ssp):
    """At x_HI = 0 the damping factor is exactly one, so the fold is the plain exact one."""
    patchy0 = _phot(_model(ssp, PATCHY_BANDS, PATCHY_Z, _patchy_igm(0.0), _exact()))
    plain = _phot(_model(ssp, PATCHY_BANDS, PATCHY_Z, {"type": "inoue"}, _exact()))
    np.testing.assert_allclose(patchy0, plain, rtol=1e-10, atol=0.0)


def test_patchy_exact_fold_is_no_worse_than_the_node_fold(ssp):
    """At x_HI = 0.5 the exact fold tracks the integrator at least as well as the node fold."""
    truth = _phot(_model(ssp, PATCHY_BANDS, PATCHY_Z, _patchy_igm(PATCHY_X_HI), None))
    exact = _phot(_model(ssp, PATCHY_BANDS, PATCHY_Z, _patchy_igm(PATCHY_X_HI), _exact()))
    node = _phot(_model(ssp, PATCHY_BANDS, PATCHY_Z, _patchy_igm(PATCHY_X_HI), _node()))

    err_exact = float(np.max(np.abs(exact / truth - 1.0)))
    err_node = float(np.max(np.abs(node / truth - 1.0)))
    assert err_exact <= err_node, (
        f"patchy x_HI={PATCHY_X_HI} at z={PATCHY_Z}: exact fold max error "
        f"{err_exact:.3e} exceeds node fold max error {err_node:.3e}"
    )


# --- DLA absorber --------------------------------------------------------------


def test_dla_exact_fold_builds_and_is_finite(ssp):
    """The exact fold no longer refuses a DLA absorber."""
    f = _phot(_model(ssp, DLA_BANDS, DLA_Z, _dla_igm(), _exact()))
    assert np.all(np.isfinite(f)) and np.all(f > 0)


def test_dla_exact_fold_is_no_worse_than_the_node_fold(ssp):
    """With a DLA the exact fold tracks the integrator at least as well as the node fold."""
    truth = _phot(_model(ssp, DLA_BANDS, DLA_Z, _dla_igm(), None))
    exact = _phot(_model(ssp, DLA_BANDS, DLA_Z, _dla_igm(), _exact()))
    node = _phot(_model(ssp, DLA_BANDS, DLA_Z, _dla_igm(), _node()))

    err_exact = float(np.max(np.abs(exact / truth - 1.0)))
    err_node = float(np.max(np.abs(node / truth - 1.0)))
    assert err_exact <= err_node, (
        f"DLA log_n_hi={DLA_LOG_N_HI} at z={DLA_Z}: exact fold max error "
        f"{err_exact:.3e} exceeds node fold max error {err_node:.3e}"
    )
