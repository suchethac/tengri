# SPDX-License-Identifier: BSD-3-Clause
r"""Bug #2378 / #3505: float32 Laplace is served and agrees with float64.

The fitter used to refuse ``fitter.run("laplace")`` in float32 on the claim that the
photometry Hessian is all-NaN there. That was measured on one narrow fixture and does
not hold: the Hessian of the standardized log posterior at the MAP is finite in float32
and agrees with float64.

Two fixtures: the minimal one-band recipe (7 free parameters), and five bands with a free
calibration floor, so the noise enters the likelihood as a parameter-dependent term, not
only the data.

Tolerances come from a measured error budget, not from the Monte Carlo noise. Both runs
draw ``N_SAMPLES`` from the same key, so the standard-normal draws are identical and the
sampling noise (``sigma / sqrt(N) = 0.007 sigma`` on a mean, ~1 % on a covariance) cancels
between the dtypes. What differs is float32 arithmetic, in two places:

* the Hessian at a given point, ``eps_H = max|dH| / max|H|``, which perturbs the covariance
  ``H^-1`` by at most ``kappa(H) eps_H / (1 - kappa(H) eps_H)`` (relative, spectral norm);
* the mode, since the float32 MAP lands a little away from the float64 one and the
  Hessian of a non-Gaussian posterior moves with it.

Measured (float64 MAP cast to float32 for ``eps_H``; two independent fits for the gaps):

=====================  ===  =========  =======  ==========  =========  =============
fixture                  D  eps_H      kappa    kappa*eps   mean gap   covariance gap
=====================  ===  =========  =======  ==========  =========  =============
one band                 7  5.9e-04    3.2      1.9e-03    0.026 sigma  0.0099
five bands, free floor   3  4.6e-06    986      4.5e-03    0.018 sigma  0.0058
=====================  ===  =========  =======  ==========  =========  =============

The Hessian budget alone is 0.2-0.5 %; the remainder of the covariance gap is the mode
shift. The tolerances are twice the largest measured gap (0.05 sigma, 0.02), which is a few
times the cancelled sampling noise and 3-5x tighter than a bound set from it. The test
also fails if the Hessian budget itself exceeds half the covariance tolerance.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import (
    DEFAULT,
    Fixed,
    ForwardModel,
    SEDModel,
    Uniform,
    generate_mock,
    recipes,
)
from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
from tengri.inference.backends.map_dispatch import run_map
from tengri.inference.backends.mcmc._shared import _get_flat_logdensity
from tengri.inference.context import InferenceContext
from tengri.inference.curvature import curvature
from tengri.inference.fitter import Fitter
from tengri.observation import NoiseModel, Observation, Photometry
from tengri.observation.photometry import FilterCurve

pytestmark = pytest.mark.regression_bug

N_SAMPLES = 20000
CASES = ("one_band", "five_band_free_floor")
#: Margins over the measured gaps (see ``test_budget_is_measured``), set from the numbers
#: recorded in each test's docstring.
MEAN_TOL_SIGMA = 0.05
COV_TOL = 0.02
CENTERS = (3800.0, 4800.0, 6200.0, 7600.0, 9000.0)
SSP = "data/fsps_prsc_miles_chabrier.h5"


def _model(ssp, case):
    if case == "one_band":
        obs = Observation(photometry=Photometry.from_names(["sdss_r"]))
        return SEDModel.build(ssp_data=ssp, observation=obs, **recipes.mock_recovery_minimal())
    curves = []
    for i, centre in enumerate(CENTERS):
        wave = np.linspace(centre - 500, centre + 500, 32)
        curves.append(FilterCurve(wave=wave, trans=np.ones_like(wave), name=f"b{i}"))
    obs = Observation(
        photometry=Photometry(filters=tuple(curves)),
        noise=NoiseModel(calibration_floor=Uniform(0.01, 0.2)),
    )
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={
            "type": "dpl",
            "all_params": Fixed(DEFAULT),
            "log_total_mass": Uniform(8.0, 12.0),
            "tau_gyr": Uniform(0.5, 8.0),
        },
        dust_attenuation={"type": "none"},
        neb={"type": "none"},
        redshift=Fixed(0.1),
    )


@pytest.fixture(scope="module")
def mocks():
    """Float64 mock data per case, shared by both dtypes."""
    ssp = load_ssp_data(SSP)
    out = {}
    with jax.enable_x64(True):
        for case in CASES:
            model = _model(ssp, case)
            truth = model.spec.sample(jax.random.PRNGKey(0))
            data = generate_mock(model, truth, key=jax.random.PRNGKey(7), snr=30.0)
            out[case] = (np.asarray(data["flux_obs"]), np.asarray(data["noise"]))
    return ssp, out


def _fitter(mocks, case):
    ssp, data = mocks
    flux, noise = data[case]
    model = _model(ssp, case)
    return Fitter(
        ForwardModel.build(sed=model),
        data=jnp.asarray(flux),
        noise=jnp.asarray(noise),
        profile_mass="auto",
    )


def _laplace(mocks, case, x64):
    with jax.enable_x64(x64):
        assert jnp.result_type(float) == (jnp.float64 if x64 else jnp.float32)
        fitter = _fitter(mocks, case)
        post = fitter.run("laplace", key=jax.random.PRNGKey(2), n_samples=N_SAMPLES)
        names = sorted(post.samples)
        draws = np.stack([np.asarray(post.samples[n], dtype=np.float64) for n in names], 1)
        return names, draws, bool(fitter._profile_mass)


@pytest.fixture(scope="module")
def runs(mocks):
    return {(c, x64): _laplace(mocks, c, x64) for c in CASES for x64 in (True, False)}


@pytest.mark.parametrize("case", CASES)
def test_laplace_runs_in_float32_and_matches_float64(case, runs):
    names32, d32, _ = runs[(case, False)]
    names64, d64, _ = runs[(case, True)]
    assert names32 == names64
    assert np.all(np.isfinite(d32))
    mean_gap = np.max(np.abs(d32.mean(0) - d64.mean(0)) / d64.std(0))
    assert mean_gap <= MEAN_TOL_SIGMA, f"mean gap {mean_gap:.4f} sigma"
    c32, c64 = np.cov(d32, rowvar=False), np.cov(d64, rowvar=False)
    cov_gap = np.max(np.abs(c32 - c64)) / np.max(np.abs(c64))
    assert cov_gap <= COV_TOL, f"covariance relative gap {cov_gap:.4f}"


@pytest.mark.parametrize("case", CASES)
def test_profile_auto_resolves_identically_in_both_dtypes(case, runs):
    assert runs[(case, False)][2] == runs[(case, True)][2]


def _flat(mocks, case, x64, mode):
    with jax.enable_x64(x64):
        fitter = _fitter(mocks, case)
        dtype = jnp.float64 if x64 else jnp.float32
        if mode is None:
            ctx = InferenceContext.from_target(fitter)
            res = run_map(ctx, key=jax.random.PRNGKey(1), n_steps=1000, verbose=False)
            mode = jax.tree.map(np.asarray, ctx.unbounded_from_posterior(res))
        point = jax.tree.map(lambda a: jnp.asarray(a, dtype), mode)
        logd, _, x, data_args = _get_flat_logdensity(fitter, point)
        curv = curvature(logd, x, data_args)
        return mode, np.asarray(curv.hess, dtype=np.float64), np.asarray(curv.grad, np.float64)


@pytest.mark.parametrize("case", CASES)
def test_hessian_budget_leaves_room_in_the_covariance_tolerance(case, mocks):
    """``kappa * eps_H`` (the Hessian's share of the covariance error) is under half the bar."""
    mode, h64, _ = _flat(mocks, case, True, None)
    _, h32, _ = _flat(mocks, case, False, mode)
    eps_h = np.max(np.abs(h32 - h64)) / np.max(np.abs(h64))
    eig = np.linalg.eigvalsh(-0.5 * (h64 + h64.T))
    kappa = eig.max() / eig.min()
    budget = kappa * eps_h / (1.0 - kappa * eps_h)
    assert eps_h <= 1e-3, f"eps_H {eps_h:.2e}"
    assert budget <= COV_TOL / 2, f"budget {budget:.2e} (eps_H={eps_h:.2e}, kappa={kappa:.1f})"
