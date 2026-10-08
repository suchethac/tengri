# SPDX-License-Identifier: BSD-3-Clause
"""#2683 follow-up: the ``dsps`` kernel places structure finer than the SSP node spacing.

The histogram kernel assigns every mass parcel to the single node whose
log-midpoint bin holds it. A burst narrower than the local node spacing is then
put on a node whose age can differ by half a spacing from the burst's, which in
the far UV (flux falls steeply with age) is a 14-19 % error for a 30 Myr burst
at z = 2.5, ~20 % for periodic bursts, 5.4 % for Gaussian-in-lookback peaks.

The kernel now integrates each bin's mass exactly (the same dense integrand as
cic, with the bin edges as exact knots) and deposits it by its mass-weighted
mean log-age, linearly between the two bracketing nodes, which preserves the
first moment of log-age: a burst lands where it belongs.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import tengri
from tengri import DEFAULT, Fixed, SEDModel
from tengri.components.stellar import component as stellar_component

pytestmark = pytest.mark.regression_bug

BANDS = ["galex_fuv", "sdss_u", "sdss_r", "2mass_h"]
TRUTH_FACTOR = 256


def _const_burst(start, end):
    return {
        "type": "const",
        "start_gyr": Fixed(start),
        "end_gyr": Fixed(end),
        "log_total_mass": Fixed(9.0),
    }


@pytest.fixture(scope="module")
def ssp():
    return tengri.load_ssp()


@pytest.fixture(scope="module")
def obs():
    return tengri.Observation(photometry=tengri.Photometry.from_names(BANDS))


def _model(ssp, obs, kernel, z, sfh):
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        neb={"type": "none"},
        redshift=Fixed(z),
        sfh={**sfh, "all_params": Fixed(DEFAULT), "age_kernel": kernel},
    )


def _phot(model):
    return np.asarray(model.predict_photometry({}))


def _truth(monkeypatch, ssp, obs, z, sfh):
    with monkeypatch.context() as m:
        m.setattr(stellar_component, "INTEGRAND_FACTOR_PARAMETRIC", TRUTH_FACTOR)
        return _phot(_model(ssp, obs, "cic", z, sfh))


@pytest.mark.parametrize(
    ("label", "z", "sfh", "tol"),
    [
        ("burst 30 Myr, lookback 1.00-1.03 Gyr, z=2.5", 2.5, _const_burst(1.03, 1.0), 0.005),
        ("burst 30 Myr, lookback 0.50-0.53 Gyr, z=2.5", 2.5, _const_burst(0.53, 0.5), 0.005),
        ("burst 10 Myr, lookback 1.00-1.01 Gyr, z=2.5", 2.5, _const_burst(1.01, 1.0), 0.005),
        ("burst 10 Myr, lookback 2.00-2.01 Gyr, z=0", 0.0, _const_burst(2.01, 2.0), 0.005),
        ("periodic, z=0", 0.0, {"type": "periodic"}, 0.02),
        ("periodic, z=2.5", 2.5, {"type": "periodic"}, 0.02),
        ("norm, z=2.5", 2.5, {"type": "norm"}, 0.015),
        ("tsnorm, z=2.5", 2.5, {"type": "tsnorm"}, 0.015),
        ("snorm, z=2.5", 2.5, {"type": "snorm"}, 0.015),
        ("continuity, z=0", 0.0, {"type": "continuity"}, 0.002),
        ("psb_flex, z=0", 0.0, {"type": "psb_flex"}, 0.002),
    ],
)
def test_a_dsps_flux_matches_converged_quadrature(monkeypatch, ssp, obs, label, z, sfh, tol):
    """dsps flux vs the 256x converged quadrature, worst of FUV/u/r/H."""
    truth = _truth(monkeypatch, ssp, obs, z, sfh)
    err = np.abs(_phot(_model(ssp, obs, "dsps", z, sfh)) / truth - 1.0)
    assert np.all(err <= tol), f"{label}: |dsps/truth-1| = {np.round(err * 100, 3)} % in {BANDS}"


@pytest.mark.parametrize("kernel", ["cic", "dsps"])
@pytest.mark.parametrize(
    ("start", "end"), [(1.03, 1.0), (0.53, 0.5), (3.007, 3.0), (2.5, 0.12)]
)
def test_b_log_age_first_moment_is_preserved(ssp, obs, kernel, start, end):
    """sum_a w_a log10(age_a) equals the SFH's own mean log-age (exact integral).

    For a constant-SFR window [end, start] (lookback, Gyr) the mean of log10 t is
    (F(start) - F(end)) / (start - end) with F(t) = t (log10 t - 1/ln 10). First
    order sharing between log-spaced nodes preserves it to the curvature of the
    log-age interpolant (~1e-5 dex); assigning the parcel to one node cannot.
    """
    model = _model(ssp, obs, kernel, 0.0, _const_burst(start, end))
    w = np.asarray(model.predict_state({}).derived["joint_weights"]).sum(0)
    w = w / w.sum()
    lg = np.asarray(ssp.ssp_lg_age_gyr)

    def big_f(t):
        return t * (np.log10(t) - 1.0 / np.log(10.0))

    exact = (big_f(start) - big_f(end)) / (start - end)
    got = float(np.sum(w * lg))
    assert abs(got - exact) <= 5e-5, f"{kernel}: mean log-age {got:.6f} vs exact {exact:.6f} dex"


def test_c_unresolved_warning_is_gone():
    """The kernel resolves the structure, so there is nothing left to warn about."""
    assert not hasattr(stellar_component, "DSPSUnresolvedHistoryWarning")
