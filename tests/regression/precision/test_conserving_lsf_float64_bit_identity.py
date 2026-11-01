# SPDX-License-Identifier: BSD-3-Clause
r"""The flux-conserving LSF in float64 matches the unscaled computation (#3149).

``compute_spectrum_conserving_lsf`` divides the rest SED by its binary peak before the
line-spread convolution and restores the factor on the pixel luminosity, so float32
never forms an erg/s-scale sum. The reference here is the unscaled computation: the
line-spread convolution, then the bin-integrated resample and the flux conversion as
one jitted program, with the same arguments the scaled path passes.

(a) forward values agree to 2e-15 relative to the peak on a real-SSP model and on
    synthetic grids (array and scalar resolution, linear and log-uniform grids);
(b) the gradient of ``sum(flux * w)`` with respect to the rest SED is bit-identical.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import (
    DEFAULT,
    FREE,
    Fixed,
    Observation,
    SEDModel,
    Spectroscopy,
    load_ssp_data,
)
from tengri._data_setup import find_data
from tengri.cosmology import luminosity_distance
from tengri.observation.spectrum import (
    _resample_peak_factored,
    apply_lsf,
    compute_spectrum_conserving_lsf,
)
from tengri.utils.conversions import lnu_to_fnu

pytestmark = pytest.mark.regression_bug

_SSP_NAME = "fsps_prsc_miles_chabrier.h5"
_Z = 0.1
_RTOL_FORWARD = 2e-15
# Measured on this branch: 3.1e-16 (array R, real-SSP model and synthetic grid),
# 1.6e-16 (scalar R). Eager tail on the same inputs: 1.5e-13 to 1.8e-13.


def _real_ssp_path():
    path = find_data(_SSP_NAME)
    if path is None:
        pytest.skip(f"real SSP grid {_SSP_NAME} not on this machine")
    return path


def _resolution_on(wave_obs):
    return jnp.linspace(1800.0, 3200.0, wave_obs.shape[0])


def _reference_lsf(sed, wave_rest, wave_obs, redshift, dl_cm, resolution, sigma_lib=0.0):
    """Unscaled LSF then bin integral and flux conversion, without any power-of-two factor."""
    wave_model_obs = wave_rest * (1.0 + redshift)

    def _onto_model(x):
        x = jnp.asarray(x)
        return x if x.ndim == 0 else jnp.interp(wave_model_obs, wave_obs, x)

    sed_broad = apply_lsf(
        sed,
        wave_rest,
        _onto_model(resolution),
        sigma_lib_kms=_onto_model(sigma_lib),
        n_bins=16,
        sigma_v_kms=0.0,
    )
    return _reference_tail(sed_broad, wave_rest, wave_obs, redshift, dl_cm)


@jax.jit
def _reference_tail(sed_broad, wave_rest, wave_obs, redshift, dl_cm):
    """Bin-integrated resample and flux conversion, one jitted program."""
    wave_rest_query = wave_obs / (1.0 + redshift)
    pixel_lnu = _resample_peak_factored(wave_rest, sed_broad, wave_rest_query, None)
    return lnu_to_fnu(pixel_lnu, dl_cm, redshift)


def _max_relative(values, reference):
    values = np.asarray(values)
    reference = np.asarray(reference)
    return float(np.max(np.abs(values - reference)) / np.max(np.abs(reference)))


def _synthetic_sed(wave_rest):
    rng = np.random.default_rng(0)
    lam = np.asarray(wave_rest)
    sed = 1e31 * (
        np.exp(-0.5 * ((np.log(lam) - np.log(4000.0)) / 0.3) ** 2)
        + 0.2 * np.exp(-0.5 * ((np.log(lam) - np.log(1500.0)) / 0.2) ** 2)
        + 0.01
    )
    return jnp.asarray(sed * (1.0 + 0.05 * rng.standard_normal(lam.shape[0]) ** 2))


def _grids():
    wave_rest = jnp.asarray(np.logspace(np.log10(900.0), np.log10(50000.0), 6000))
    lin_obs = jnp.linspace(3600.0, 9800.0, 1200)
    log_obs = jnp.asarray(np.logspace(np.log10(3600.0), np.log10(9800.0), 1200))
    return wave_rest, lin_obs, log_obs


@pytest.mark.parametrize(
    ("grid", "resolution"),
    [("linear", "array"), ("linear", "scalar"), ("log", "scalar")],
    ids=["linear_array_R", "linear_scalar_R", "loguniform_scalar_R"],
)
def test_synthetic_lsf_forward_matches_unscaled(grid, resolution):
    wave_rest, lin_obs, log_obs = _grids()
    wave_obs = lin_obs if grid == "linear" else log_obs
    sed = _synthetic_sed(wave_rest)
    res = _resolution_on(wave_obs) if resolution == "array" else jnp.asarray(2500.0)
    dl_cm = 1.5e28
    got = compute_spectrum_conserving_lsf(sed, wave_rest, wave_obs, _Z, dl_cm, resolution=res)
    ref = _reference_lsf(sed, wave_rest, wave_obs, _Z, dl_cm, res)
    assert _max_relative(got, ref) <= _RTOL_FORWARD


def _spectroscopy_model(ssp):
    wave_obs = jnp.linspace(3600.0, 9800.0, 1200)
    observation = Observation(
        spectroscopy=Spectroscopy(
            wave_obs=wave_obs,
            resolution=_resolution_on(wave_obs),
            resample="conserving",
        )
    )
    return SEDModel.build(
        ssp_data=ssp,
        observation=observation,
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT), "log_total_mass": FREE},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "tau_bc": 0.5,
            "tau_diff": 0.3,
            "other_params": Fixed(DEFAULT),
        },
        neb={"type": "none"},
        redshift=Fixed(_Z),
    )


def _real_model_inputs():
    """Real-SSP model at log10 M = 10: rest SED, rest grid and observed grid."""
    path = _real_ssp_path()
    model = _spectroscopy_model(load_ssp_data(path))
    state = model.predict_state({"sfh_dpl_log_total_mass": 10.0})
    wave_obs = jnp.linspace(3600.0, 9800.0, 1200)
    return model, state, wave_obs


def test_real_model_lsf_forward_matches_unscaled():
    model, state, wave_obs = _real_model_inputs()
    spectrum = np.asarray(model.predict_spectrum({"sfh_dpl_log_total_mass": 10.0}))
    assert np.all(np.isfinite(spectrum))
    assert np.any(spectrum != 0.0), "predicted spectrum is identically zero"
    wave_rest = state.wave
    sed = state.sed_intrinsic
    dl_cm = jnp.asarray(luminosity_distance(_Z)).reshape(())
    res = _resolution_on(wave_obs)
    got = compute_spectrum_conserving_lsf(sed, wave_rest, wave_obs, _Z, dl_cm, resolution=res)
    ref = _reference_lsf(sed, wave_rest, wave_obs, _Z, dl_cm, res)
    assert _max_relative(got, ref) <= _RTOL_FORWARD


def test_real_model_lsf_gradient_is_bit_identical():
    _, state, wave_obs = _real_model_inputs()
    wave_rest = state.wave
    sed = state.sed_intrinsic
    dl_cm = jnp.asarray(luminosity_distance(_Z)).reshape(())
    res = _resolution_on(wave_obs)
    w = jnp.asarray(np.random.default_rng(3).standard_normal(wave_obs.shape[0]))

    def new_loss(s):
        return jnp.sum(
            compute_spectrum_conserving_lsf(s, wave_rest, wave_obs, _Z, dl_cm, resolution=res) * w
        )

    def ref_loss(s):
        return jnp.sum(_reference_lsf(s, wave_rest, wave_obs, _Z, dl_cm, res) * w)

    grad_new = np.asarray(jax.grad(new_loss)(sed))
    grad_ref = np.asarray(jax.grad(ref_loss)(sed))
    assert np.array_equal(grad_new, grad_ref)
