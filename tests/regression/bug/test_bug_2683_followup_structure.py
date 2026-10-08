# SPDX-License-Identifier: BSD-3-Clause
"""#2683 follow-up: the ``dsps`` age kernel places structure finer than the SSP node spacing.

The histogram kernel assigned every mass parcel to the single node whose
log-midpoint bin holds it. A burst narrower than the local node spacing was then
put on a node whose age can differ by half a spacing from the burst's, which in
the far UV (flux falls steeply with age) is a 14-19 % error for a 30 Myr burst
at z = 2.5, 26 % for periodic bursts, 5.6 % for Gaussian-in-lookback peaks, and
no refinement of the SFR table removes it. First-order (log-age linear) sharing
of the exact integrand is the exact answer, so ``age_kernel='dsps'`` now selects
the same integration as ``'cic'``: the weights agree to round-off for every SFH
family, and both reproduce the converged quadrature (the integrand at 256x).
"""

from __future__ import annotations

import jax
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
    """Photometry on the 256x integrand. The compiled kernels are cached per
    function, not per module constant, so the cache is cleared around the patch."""
    jax.clear_caches()
    try:
        with monkeypatch.context() as m:
            m.setattr(stellar_component, "INTEGRAND_FACTOR_PARAMETRIC", TRUTH_FACTOR)
            m.setattr(stellar_component, "INTEGRAND_FACTOR_SAWTOOTH", TRUTH_FACTOR)
            return _phot(_model(ssp, obs, "cic", z, sfh))
    finally:
        jax.clear_caches()


CASES = [
    ("burst 30 Myr, lookback 1.00-1.03 Gyr, z=2.5", 2.5, _const_burst(1.03, 1.0)),
    ("burst 30 Myr, lookback 0.50-0.53 Gyr, z=2.5", 2.5, _const_burst(0.53, 0.5)),
    ("burst 10 Myr, lookback 1.00-1.01 Gyr, z=2.5", 2.5, _const_burst(1.01, 1.0)),
    ("burst 10 Myr, lookback 2.00-2.01 Gyr, z=0", 0.0, _const_burst(2.01, 2.0)),
    ("periodic, z=0", 0.0, {"type": "periodic"}),
    ("periodic, z=2.5", 2.5, {"type": "periodic"}),
    ("norm, z=2.5", 2.5, {"type": "norm"}),
    ("tsnorm, z=2.5", 2.5, {"type": "tsnorm"}),
    ("snorm, z=2.5", 2.5, {"type": "snorm"}),
    ("continuity, z=0", 0.0, {"type": "continuity"}),
    ("continuity, z=2.5", 2.5, {"type": "continuity"}),
    ("psb_flex, z=0", 0.0, {"type": "psb_flex"}),
    ("psb_flex, z=2.5", 2.5, {"type": "psb_flex"}),
]


@pytest.mark.parametrize(("label", "z", "sfh"), CASES, ids=[c[0] for c in CASES])
def test_a_dsps_is_the_first_order_integration(ssp, obs, label, z, sfh):
    """dsps flux equals cic flux to round-off (was up to 26 % in the FUV)."""
    cic = _phot(_model(ssp, obs, "cic", z, sfh))
    dsps = _phot(_model(ssp, obs, "dsps", z, sfh))
    err = np.abs(dsps / cic - 1.0)
    assert np.all(err <= 1e-9), f"{label}: |dsps/cic-1| = {np.round(err * 100, 4)} % in {BANDS}"


@pytest.mark.parametrize(
    ("label", "z", "sfh", "tol"),
    [
        ("burst 30 Myr, lookback 1.00-1.03 Gyr, z=2.5", 2.5, _const_burst(1.03, 1.0), 0.002),
        ("burst 30 Myr, lookback 0.50-0.53 Gyr, z=2.5", 2.5, _const_burst(0.53, 0.5), 0.003),
        ("burst 10 Myr, lookback 1.00-1.01 Gyr, z=2.5", 2.5, _const_burst(1.01, 1.0), 0.005),
        ("periodic, z=0", 0.0, {"type": "periodic"}, 0.0015),
        ("periodic, z=2.5", 2.5, {"type": "periodic"}, 0.0015),
        ("norm, z=2.5", 2.5, {"type": "norm"}, 0.002),
        ("tsnorm, z=2.5", 2.5, {"type": "tsnorm"}, 0.002),
        ("continuity, z=0", 0.0, {"type": "continuity"}, 1e-4),
        ("psb_flex, z=0", 0.0, {"type": "psb_flex"}, 1e-4),
    ],
)
def test_b_dsps_flux_matches_converged_quadrature(monkeypatch, ssp, obs, label, z, sfh, tol):
    """dsps flux vs the 256x converged quadrature, worst of FUV/u/r/H.

    The cic integrand is exact for a top-hat or step with its edges as knots, so
    the burst and step cases converge to 1e-4; the periodic family's onsets are
    exact knots on a 128x integrand (-0.10 % FUV at z = 0; 16x left -4.7 %); the
    Gaussian peaks and the 10 Myr burst are limited by the integrand's own
    ~30 Myr resolution.
    """
    truth = _truth(monkeypatch, ssp, obs, z, sfh)
    err = np.abs(_phot(_model(ssp, obs, "dsps", z, sfh)) / truth - 1.0)
    assert np.all(err <= tol), f"{label}: |dsps/truth-1| = {np.round(err * 100, 3)} % in {BANDS}"


@pytest.mark.parametrize("kernel", ["cic", "dsps"])
@pytest.mark.parametrize(
    ("start", "end"), [(1.03, 1.0), (0.53, 0.5), (3.2, 3.0), (2.5, 0.12)]
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


def test_c_no_unresolved_history_warning_is_defined():
    """Nothing is left unresolved, so there is no warning class to emit."""
    assert not hasattr(stellar_component, "DSPSUnresolvedHistoryWarning")
