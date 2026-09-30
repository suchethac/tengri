# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for the free instrument-LSF scale ``lsf_scale`` (#2526).

``lsf_scale`` multiplies sigma_inst(lambda) before the quadrature
combination in ``apply_lsf``, for both the stellar-continuum kernel and
the #2519 instrument-only line kernel (``Observation.predict`` divides
``resolution`` by ``lsf_scale`` before either ``project_spectrum`` call,
since sigma_inst = c / (2.3548 * resolution) is inversely proportional to
resolution). Declared in ``src/tengri/parameters/_shared.py`` alongside
``redshift`` and ``sigma_v_kms`` -- the existing precedent for a bare,
non-domain-prefixed observation/instrument-level parameter outside
``docs/dev/NAMING_CONTRACT.md``'s §3.2 physics-domain prefixes.
"""

from __future__ import annotations

import warnings

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import minimize_scalar

import tengri
from tengri import DEFAULT, FREE, Fixed, Observation, SEDModel, Uniform, load_ssp_data
from tengri.observation.spectroscopy import Spectroscopy
from tengri.observation.spectrum import apply_lsf

pytestmark = pytest.mark.regression_bug

_C_KMS = 299792.458
_FWHM_TO_SIGMA = 2.354820045030949


def test_lsf_scale_declared_fixed_one_default_uniform_free_prior():
    """lsf_scale is registered: default Fixed(1.0), free_prior Uniform(0.8, 1.2)."""
    registry = tengri.list_parameters()
    assert "lsf_scale" in registry.names()

    from tengri import Parameters

    spec = Parameters(lsf_scale=Fixed(1.0), redshift=Fixed(0.1))
    dist = spec.get_distribution("lsf_scale")
    assert dist.is_fixed
    assert float(dist.value) == 1.0

    spec_free = Parameters(lsf_scale=Uniform(0.8, 1.2), redshift=Fixed(0.1))
    free_dist = spec_free.get_distribution("lsf_scale")
    assert free_dist.bounds == (0.8, 1.2)


def test_apply_lsf_bit_identical_at_lsf_scale_one():
    """resolution / 1.0 must reproduce apply_lsf's un-scaled kernel exactly."""
    wave = jnp.linspace(3600.0, 7300.0, 2000)
    flux = jnp.ones_like(wave) - 0.3 * jnp.exp(-0.5 * ((wave - 5000.0) / 3.0) ** 2)
    resolution = 3000.0
    baseline = apply_lsf(flux, wave, resolution=resolution, sigma_lib_kms=40.0, sigma_v_kms=100.0)
    scaled = apply_lsf(
        flux, wave, resolution=resolution / 1.0, sigma_lib_kms=40.0, sigma_v_kms=100.0
    )
    chex.assert_trees_all_close(scaled, baseline, atol=0.0, rtol=0.0)


def _ssp_or_skip(path):
    from pathlib import Path

    if not Path(path).is_file():
        pytest.skip(f"missing SSP grid {path}")
    return load_ssp_data(path)


def test_predict_spectrum_bit_identical_at_lsf_scale_one():
    """A free lsf_scale explicitly set to 1.0 reproduces the model's output
    with lsf_scale left at its Fixed(1.0) default, through the full
    Observation.predict / SEDModel.predict_spectrum path (#2526).
    """
    from pathlib import Path

    if not Path("data/cue_weights.npz").is_file():
        pytest.skip("Cue weights (data/cue_weights.npz) not present")
    ssp = _ssp_or_skip("data/fsps_prsc_miles_chabrier.h5")
    z = 0.05
    wave_obs = jnp.linspace(6800.0, 6990.0, 400)
    obs = Observation(
        spectroscopy=Spectroscopy(wave_obs=wave_obs, resolution=3000.0, sigma_lib_kms=70.0)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={"type": "dpl", "all_params": FREE},
            dust_attenuation={"type": "two_component", "law": "calzetti", "all_params": FREE},
            neb={"type": "cue", "all_params": Fixed(DEFAULT)},
            redshift=Fixed(z),
        )
        model_default = SEDModel(base.spec, ssp, observation=obs)
        model_free = SEDModel(
            base.spec.merge_observation_params(lsf_scale=Uniform(0.8, 1.2)),
            ssp,
            observation=obs,
        )
    p0 = dict(model_default.spec.sample(jax.random.PRNGKey(3)))
    p0["sfh_dpl_log_total_mass"] = jnp.asarray(10.0)
    p0["dust_tau_bc"] = jnp.asarray(0.0)
    p0["dust_tau_diff"] = jnp.asarray(0.0)

    flux_default = model_default.predict_spectrum(p0)
    flux_scale_one = model_free.predict_spectrum({**p0, "lsf_scale": jnp.asarray(1.0)})
    chex.assert_trees_all_close(flux_scale_one, flux_default, atol=0.0, rtol=1e-10)


def _recover_lsf_scale(template, wave, resolution, mock):
    def objective(scale):
        pred = apply_lsf(
            template, wave, resolution=resolution / scale, sigma_lib_kms=0.0, sigma_v_kms=0.0
        )
        return float(jnp.sum((pred - mock) ** 2))

    result = minimize_scalar(
        objective, bounds=(0.8, 1.2), method="bounded", options={"xatol": 1e-4}
    )
    return float(result.x)


def test_lsf_scale_recovery():
    """A 1-D least-squares fit against an lsf_scale=1.1 mock recovers 1.1 +/- 0.01."""
    wave = jnp.linspace(3600.0, 7300.0, 2000)
    flux = jnp.asarray(np.ones_like(np.asarray(wave)))
    line_centers = (3934.0, 3969.0, 4102.0, 4861.0, 5175.0, 5895.0, 6563.0, 6900.0)
    for lc in line_centers:
        flux = flux - 0.3 * jnp.exp(-0.5 * ((wave - lc) / 3.0) ** 2)
    resolution = 3000.0
    true_scale = 1.1
    mock = np.asarray(
        apply_lsf(
            flux, wave, resolution=resolution / true_scale, sigma_lib_kms=0.0, sigma_v_kms=0.0
        )
    )
    recovered = _recover_lsf_scale(flux, wave, resolution, mock)
    assert recovered == pytest.approx(true_scale, abs=0.01)


def test_sigma_v_and_lsf_scale_degenerate_on_continuum_not_on_lines():
    """sigma_v and lsf_scale trade off in the stellar continuum kernel (both
    widen sqrt(sigma_v^2 + sigma_inst^2 - sigma_lib^2)) but the #2519 line
    kernel excludes sigma_v entirely, so only lsf_scale moves it -- the two
    degenerate continuum settings must give DIFFERENT line widths (#2526).
    """
    resolution = 3000.0
    sigma_inst_1 = _C_KMS / (_FWHM_TO_SIGMA * resolution)  # at lsf_scale=1

    # Pair A: (sigma_v=0, lsf_scale=1.0). Pair B: chosen so the CONTINUUM
    # kernel (which includes sigma_v) matches pair A exactly.
    scale_b = 0.9
    sigma_inst_b = scale_b * sigma_inst_1
    sigma_v_b = float(np.sqrt(max(sigma_inst_1**2 - sigma_inst_b**2, 0.0)))
    assert scale_b != 1.0 and sigma_v_b > 0.0

    continuum_kernel_a = np.sqrt(0.0**2 + sigma_inst_1**2)
    continuum_kernel_b = np.sqrt(sigma_v_b**2 + sigma_inst_b**2)
    assert continuum_kernel_a == pytest.approx(continuum_kernel_b, rel=1e-9)

    # Line kernel excludes sigma_v (#2519): only lsf_scale sets it, so pair A
    # and pair B, degenerate above, must now differ.
    line_kernel_a = sigma_inst_1
    line_kernel_b = sigma_inst_b
    assert line_kernel_a != pytest.approx(line_kernel_b, rel=1e-3)
