# SPDX-License-Identifier: BSD-3-Clause
"""Every spectrum-prediction surface shares one kernel (#2519, #2526).

``project_spectrum_kernel_split`` in ``src/tengri/observation/observation.py``
is the single function that decides which SED components get the stellar
kernel and which get the instrument-only kernel, and how ``lsf_scale``
enters. Four surfaces call it (directly, or through
``Observation.predict``): the eager orchestrator path
(``SEDModel._spectrum_via_state``), the arbitrary-grid projector
(``SEDModel._predict_spectrum_on_grid``, reached by
``predict_spectrum(params, wave_obs=...)``), the compiled
``predict_observables`` kernel (reached by ``predict_spectrum(params)`` when
a spectroscopy channel is configured), and the cached-``ForwardState``
accessor (``Prediction.spectrum()``, reached by
``model.predict(params).spectrum()``). A model whose free parameters are
identical across all four, evaluated on the same pixel grid, must predict
the same spectrum from any of them.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import pytest

from tengri import DEFAULT, FREE, Fixed, Observation, SEDModel, Uniform, load_ssp_data
from tengri.observation.banded import gaussian_resolution_bands
from tengri.observation.spectroscopy import Spectroscopy

pytestmark = pytest.mark.contract

_MILES_BARE = "data/fsps_prsc_miles_chabrier.h5"


def _ssp_or_skip(path):
    if not Path(path).is_file():
        pytest.skip(f"missing SSP grid {path}")
    return load_ssp_data(path)


def _cue_or_skip():
    if not Path("data/cue_weights.npz").is_file():
        pytest.skip("Cue weights (data/cue_weights.npz) not present")


def _build_model(ssp, neb, wave_obs, *, resolution=None, resolution_matrix=None):
    spec = Spectroscopy(
        wave_obs=wave_obs,
        resolution=resolution,
        sigma_lib_kms=70.0,
        resolution_matrix=resolution_matrix,
    )
    obs = Observation(spectroscopy=spec)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={"type": "dpl", "all_params": FREE},
            dust_attenuation={"type": "two_component", "law": "calzetti", "all_params": FREE},
            neb=neb,
            redshift=Fixed(0.05),
        )
        merged = base.spec.merge_observation_params(
            sigma_v_kms=Uniform(0.0, 2000.0), lsf_scale=Uniform(0.8, 1.2)
        )
        return SEDModel(merged, ssp, observation=obs)


def _params(model, sigma_v, lsf_scale):
    p = dict(model.spec.sample(jax.random.PRNGKey(7)))
    p["sfh_dpl_log_total_mass"] = jnp.asarray(10.0)
    p["dust_tau_bc"] = jnp.asarray(0.0)
    p["dust_tau_diff"] = jnp.asarray(0.0)
    p["sigma_v_kms"] = jnp.asarray(sigma_v)
    p["lsf_scale"] = jnp.asarray(lsf_scale)
    return p


def _four_paths(model, params, wave_obs):
    """The eager, arbitrary-grid, compiled-kernel, and Prediction accessor
    surfaces, on one grid.
    """
    flux_eager = model._spectrum_via_state(params, wave_obs=wave_obs)
    flux_grid = model.predict_spectrum(params, wave_obs=wave_obs)
    flux_jit = model.predict_spectrum(params)  # configured grid IS wave_obs
    pred = model.predict(params)
    flux_pred = pred.spectrum()  # configured grid IS wave_obs
    flux_pred_grid = pred.spectrum(wave_obs=wave_obs)
    return flux_eager, flux_grid, flux_jit, flux_pred, flux_pred_grid


@pytest.mark.parametrize("neb_type", ["cue", "none"])
@pytest.mark.parametrize("sigma_v", [0.0, 300.0])
@pytest.mark.parametrize("lsf_scale", [1.0, 1.1])
def test_gaussian_lsf_paths_agree_to_1e10(sigma_v, lsf_scale, neb_type):
    """The default Gaussian LSF path: same spectrum from all three surfaces."""
    if neb_type == "cue":
        _cue_or_skip()
    ssp = _ssp_or_skip(_MILES_BARE)
    wave_obs = jnp.linspace(6800.0, 6990.0, 400)
    neb = {"type": "cue", "all_params": Fixed(DEFAULT)} if neb_type == "cue" else {"type": "none"}
    model = _build_model(ssp, neb, wave_obs, resolution=3000.0)
    p = _params(model, sigma_v, lsf_scale)

    flux_eager, flux_grid, flux_jit, flux_pred, flux_pred_grid = _four_paths(model, p, wave_obs)
    chex.assert_trees_all_close(flux_eager, flux_grid, rtol=1e-10, atol=0.0)
    chex.assert_trees_all_close(flux_eager, flux_jit, rtol=1e-10, atol=0.0)
    chex.assert_trees_all_close(flux_eager, flux_pred, rtol=1e-10, atol=0.0)
    chex.assert_trees_all_close(flux_eager, flux_pred_grid, rtol=1e-10, atol=0.0)


@pytest.mark.parametrize("sigma_v", [0.0, 300.0])
def test_banded_path_agrees_across_surfaces(sigma_v):
    """Banded resolution-matrix path: same #2519 split, same four-way agreement.

    The stellar piece is broadened by sigma_v before ``R @ model`` (the
    matrix carries no galaxy-kinematics term of its own, #2506); the
    instrument-only piece is not.
    """
    _cue_or_skip()
    ssp = _ssp_or_skip(_MILES_BARE)
    wave_obs = jnp.linspace(6800.0, 6990.0, 400)
    bm = gaussian_resolution_bands(wave_obs, resolution=300.0, n_diag=21)
    neb = {"type": "cue", "all_params": Fixed(DEFAULT)}
    model = _build_model(ssp, neb, wave_obs, resolution=None, resolution_matrix=bm)
    p = _params(model, sigma_v, 1.0)

    flux_eager, flux_grid, flux_jit, flux_pred, flux_pred_grid = _four_paths(model, p, wave_obs)
    chex.assert_trees_all_close(flux_eager, flux_grid, rtol=1e-10, atol=0.0)
    chex.assert_trees_all_close(flux_eager, flux_jit, rtol=1e-10, atol=0.0)
    chex.assert_trees_all_close(flux_eager, flux_pred, rtol=1e-10, atol=0.0)
    chex.assert_trees_all_close(flux_eager, flux_pred_grid, rtol=1e-10, atol=0.0)


def test_lsf_scale_excluded_from_banded_path():
    """lsf_scale does not reach the banded path: the matrix has no scalar
    resolution parameter for it to rescale (unlike the Gaussian path's
    resolution / lsf_scale trick). The banded operator is the measured
    instrument response directly; its own calibration uncertainty, if
    modeled, would need to act on the matrix itself, not on this parameter.
    """
    _cue_or_skip()
    ssp = _ssp_or_skip(_MILES_BARE)
    wave_obs = jnp.linspace(6800.0, 6990.0, 400)
    bm = gaussian_resolution_bands(wave_obs, resolution=300.0, n_diag=21)
    neb = {"type": "cue", "all_params": Fixed(DEFAULT)}
    model = _build_model(ssp, neb, wave_obs, resolution=None, resolution_matrix=bm)
    p_one = _params(model, 200.0, 1.0)
    p_scaled = _params(model, 200.0, 1.1)

    flux_one = model._spectrum_via_state(p_one, wave_obs=wave_obs)
    flux_scaled = model._spectrum_via_state(p_scaled, wave_obs=wave_obs)
    chex.assert_trees_all_close(flux_one, flux_scaled, rtol=0.0, atol=0.0)
