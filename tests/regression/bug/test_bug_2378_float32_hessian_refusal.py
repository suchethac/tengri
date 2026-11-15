# SPDX-License-Identifier: BSD-3-Clause
r"""Bug #2378 / #3505: float32 Laplace is served and agrees with float64.

The fitter used to refuse ``fitter.run("laplace")`` in float32 on the claim that the
photometry Hessian is all-NaN there. That was measured on one narrow fixture and does
not hold: the Hessian of the standardized log posterior at the MAP is finite in float32
and agrees with float64 (#3505, ``test_dtype_curvature_sweep.py``).

Assertions

(a) Under float32, ``fitter.run("laplace")`` runs with ``profile_mass="auto"``.

(b) The float32 posterior mean is within 0.1 sigma of the float64 one and the covariance
    within 5 %. Both runs draw 20000 samples from the same key, so the Monte Carlo
    noise is about 0.007 sigma on the mean and about 1 % on the covariance.

(c) ``profile_mass="auto"`` engages identically in both dtypes.

(d) A non-finite preconditioning input is reported as an upstream failure in both dtypes,
    with no dtype named as the cause.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import ForwardModel, SEDModel, generate_mock, recipes
from tengri.inference.fitter import Fitter
from tengri.inference.preconditioning import prepare_preconditioning
from tengri.observation import Observation, Photometry

pytestmark = pytest.mark.regression_bug

N_SAMPLES = 20000
MEAN_TOL_SIGMA = 0.1
COV_TOL = 0.05


def _ssp():
    from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

    return load_ssp_data("data/fsps_prsc_miles_chabrier.h5")


def _build_fit(ssp, flux, noise):
    obs = Observation(photometry=Photometry.from_names(["sdss_r"]))
    model = SEDModel.build(ssp_data=ssp, observation=obs, **recipes.mock_recovery_minimal())
    fitter = Fitter(
        ForwardModel.build(sed=model),
        data=jnp.asarray(flux),
        noise=jnp.asarray(noise),
        profile_mass="auto",
    )
    return model, fitter


@pytest.fixture(scope="module")
def mock():
    ssp = _ssp()
    with jax.enable_x64(True):
        obs = Observation(photometry=Photometry.from_names(["sdss_r"]))
        model = SEDModel.build(ssp_data=ssp, observation=obs, **recipes.mock_recovery_minimal())
        key_truth, key_mock = jax.random.split(jax.random.PRNGKey(0))
        truth = model.spec.sample(key_truth)
        data = generate_mock(model, truth, key=key_mock, snr=30.0)
        return ssp, np.asarray(data["flux_obs"]), np.asarray(data["noise"])


def _run(mock, x64):
    ssp, flux, noise = mock
    with jax.enable_x64(x64):
        assert jnp.result_type(float) == (jnp.float64 if x64 else jnp.float32)
        _, fitter = _build_fit(ssp, flux, noise)
        post = fitter.run("laplace", key=jax.random.PRNGKey(2), n_samples=N_SAMPLES)
        names = sorted(post.samples)
        draws = np.stack([np.asarray(post.samples[n], dtype=np.float64) for n in names], 1)
        return names, draws, bool(fitter._profile_mass)


@pytest.fixture(scope="module")
def laplace_f64(mock):
    return _run(mock, True)


@pytest.fixture(scope="module")
def laplace_f32(mock):
    return _run(mock, False)


def test_laplace_runs_in_float32_and_matches_float64(laplace_f32, laplace_f64):
    names32, d32, _ = laplace_f32
    names64, d64, _ = laplace_f64
    assert names32 == names64
    assert np.all(np.isfinite(d32))
    mean_gap = np.max(np.abs(d32.mean(0) - d64.mean(0)) / d64.std(0))
    assert mean_gap <= MEAN_TOL_SIGMA, f"mean gap {mean_gap:.3f} sigma"
    c32, c64 = np.cov(d32, rowvar=False), np.cov(d64, rowvar=False)
    cov_gap = np.max(np.abs(c32 - c64)) / np.max(np.abs(c64))
    assert cov_gap <= COV_TOL, f"covariance relative gap {cov_gap:.3f}"


def test_profile_auto_resolves_identically_in_both_dtypes(laplace_f32, laplace_f64):
    assert laplace_f32[2] == laplace_f64[2]


@pytest.mark.parametrize("x64", [True, False], ids=["float64", "float32"])
def test_non_finite_preconditioning_input_blames_no_dtype(x64):
    def quartic(xi):
        return -0.5 * jnp.sum(xi**4)

    bad_init = jnp.asarray([0.0, jnp.nan, 0.0, 0.0])
    with jax.enable_x64(x64), pytest.raises(ValueError) as excinfo:
        prepare_preconditioning(quartic, bad_init, 0.0, precondition=True)
    assert "float32" not in str(excinfo.value).lower()
