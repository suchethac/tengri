# SPDX-License-Identifier: BSD-3-Clause
"""#3343: emission-line profiles use the resolution at each line.

A NIRSpec grating changes R by about a factor of two end to end. The line
profiles used the median, so the instrumental width of a line near either end
was wrong by up to that factor, and a fitted intrinsic width absorbed the error.
Mock spectra here are NumPy Gaussians, not the design-matrix builder's output.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, Observation, SEDModel, Spectroscopy, Uniform
from tengri.inference.batched_spectro import batched_log_likelihood
from tengri.inference.fitter import Fitter
from tengri.inference.likelihood import _build_eline_G_eff
from tengri.inference.loss_functions import build_loglikelihood_fn
from tengri.observation.batched import GalaxySpectrum, build_spectro_batches
from tengri.observation.eline_marginalization import (
    build_eline_design_matrix,
    eline_amplitude_names,
    line_resolution,
)
from tengri.observation.line_list import LineList

pytestmark = pytest.mark.contract

_C_KMS = 299792.458
_Z = 0.1
_WAVE = np.linspace(4000.0, 7500.0, 3000)
# R rises by a factor of two across the grid, as on a NIRSpec grating.
_R_CURVE = np.linspace(1000.0, 2000.0, _WAVE.size)
_LINES = ["Hbeta", "OIII_5007", "Halpha"]
_SIGMA_TRUE = 60.0
_SIGMA_NOISE = 2.0e-19
_AMPS = {"Hbeta": 4.0e-16, "OIII_4959": 6.0e-16 / 2.98, "OIII_5007": 6.0e-16, "Halpha": 1.2e-15}


def _gaussian(rest, sigma_kms, resolution):
    lam = rest * (1.0 + _Z)
    width = np.hypot(lam / (2.355 * resolution), lam * sigma_kms / _C_KMS)
    return np.exp(-0.5 * ((_WAVE - lam) / width) ** 2) / (np.sqrt(2.0 * np.pi) * width)


def _ssp():
    from tengri.components.stellar.sps.dsps_wrapper import SSPData

    flux = jnp.abs(jax.random.normal(jax.random.PRNGKey(0), (3, 20, 100))) * 1e-3 + 1e-5
    return SSPData(
        ssp_wave=jnp.linspace(3000.0, 10000.0, 100),
        ssp_flux=flux,
        ssp_lg_age_gyr=jnp.linspace(-1.0, 1.14, 20),
        ssp_lgmet=jnp.array([-2.0, -1.0, 0.0]),
    )


def _model(resolution, redshift=None):
    spectroscopy = Spectroscopy(
        wave_obs=jnp.asarray(_WAVE),
        resolution=resolution,
        eline_mode="marginalized",
        eline_catalog=LineList.from_names(_LINES),
        eline_prior_sigma=1.0e-14,
    )
    return SEDModel.build(
        ssp_data=_ssp(),
        observation=Observation(spectroscopy=spectroscopy),
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        neb={"type": "none"},
        redshift=Fixed(_Z) if redshift is None else redshift,
        eline_sigma_kms=Uniform(0.0, 300.0),
    )


@pytest.fixture(scope="module")
def mock():
    model = _model(jnp.asarray(_R_CURVE))
    continuum = np.asarray(model.predict_spectrum({"eline_sigma_kms": 0.0}))
    catalog = model.observation.spectroscopy.effective_catalog
    lines = np.zeros_like(_WAVE)
    for name, rest in zip(catalog.names, np.asarray(catalog.wavelengths)):
        r_line = np.interp(rest * (1.0 + _Z), _WAVE, _R_CURVE)
        lines += _AMPS[name] * _gaussian(rest, _SIGMA_TRUE, r_line)
    flux = continuum + lines + np.random.default_rng(3343).normal(0.0, _SIGMA_NOISE, _WAVE.size)
    return model, flux


def _best_sigma(model, flux):
    fitter = Fitter(
        model,
        jnp.asarray(flux),
        jnp.full(_WAVE.size, _SIGMA_NOISE),
        data_type="spectroscopy",
        approx=None,
    )
    fn = build_loglikelihood_fn(fitter)
    grid = np.arange(20.0, 120.0, 1.0)
    values = [float(fn({"eline_sigma_kms": s}, fitter._data_args)) for s in grid]
    return float(grid[int(np.argmax(values))])


def test_design_columns_use_the_resolution_at_each_line():
    model = _model(jnp.asarray(_R_CURVE))
    catalog = model.observation.spectroscopy.effective_catalog
    rest = np.asarray(catalog.wavelengths)
    design = np.asarray(
        _build_eline_G_eff(
            {"eline_sigma_kms": _SIGMA_TRUE},
            {"redshift": _Z},
            model,
            jnp.asarray(rest),
            jnp.eye(rest.size),
        )
    )
    for j, lam in enumerate(rest):
        expected = _gaussian(lam, _SIGMA_TRUE, np.interp(lam * (1 + _Z), _WAVE, _R_CURVE))
        np.testing.assert_allclose(design[:, j], expected, rtol=1e-6, atol=1e-12)


def test_intrinsic_width_is_recovered_with_a_varying_resolution(mock):
    model, flux = mock
    assert abs(_best_sigma(model, flux) - _SIGMA_TRUE) <= 3.0


def test_the_median_resolution_biases_the_width(mock):
    """The same data fitted with the median R, the old rule, miss the width."""
    _, flux = mock
    median_model = _model(float(np.median(_R_CURVE)))
    assert abs(_best_sigma(median_model, flux) - _SIGMA_TRUE) > 10.0


def test_scalar_resolution_keeps_the_scalar_kernel():
    """A scalar R takes the scalar kernel: bit-identical to a direct call."""
    model = _model(1500.0)
    rest = jnp.asarray(LineList.from_names(_LINES).wavelengths)
    eye = jnp.eye(rest.shape[0])
    got = _build_eline_G_eff({"eline_sigma_kms": 80.0}, {"redshift": _Z}, model, rest, eye)
    ref = build_eline_design_matrix(model.wave_obs, rest, 1500.0, _Z, 80.0, 0.0)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))


def test_resolution_gradient_with_respect_to_redshift_is_finite():
    rest = jnp.asarray([4862.68, 6564.61])

    def width_sum(z):
        return jnp.sum(line_resolution(jnp.asarray(_R_CURVE), jnp.asarray(_WAVE), rest, z))

    grad = float(jax.grad(width_sum)(0.1))
    slope = (_R_CURVE[-1] - _R_CURVE[0]) / (_WAVE[-1] - _WAVE[0])
    assert grad == pytest.approx(slope * float(jnp.sum(rest)), rel=1e-4)


def _fitted_model(redshift):
    spectroscopy = Spectroscopy(
        wave_obs=jnp.asarray(_WAVE),
        resolution=jnp.asarray(_R_CURVE),
        eline_mode="fitted",
        eline_catalog=LineList.from_names(_LINES),
        eline_prior_sigma=1.0e-14,
    )
    model = SEDModel.build(
        ssp_data=_ssp(),
        observation=Observation(spectroscopy=spectroscopy),
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        neb={"type": "none"},
        redshift=redshift,
    )
    spec = model.spec.with_fixed_value("eline_sigma_kms", _SIGMA_TRUE)
    return SEDModel(spec, model.ssp_data, observation=model.observation)


def test_batched_and_per_galaxy_likelihoods_agree_with_per_pixel_resolution(mock):
    """Both paths read R at each line, so they agree on a per-pixel resolution."""
    _, flux = mock
    template = _fitted_model(Uniform(0.0, 0.5))
    catalog = template.observation.spectroscopy.effective_catalog
    names = eline_amplitude_names(catalog)
    amps = {n: _AMPS[n.removeprefix("eline_amp_")] for n in names}
    ivar = np.full(_WAVE.size, 1.0 / _SIGMA_NOISE**2)
    galaxy = GalaxySpectrum(wave=_WAVE, flux=flux, ivar=ivar, z=_Z, resolution=_R_CURVE)
    spec, batch, _ = build_spectro_batches([galaxy], model=template)[0]
    obs = batch.galaxy(0)
    expected_r = np.interp(np.asarray(catalog.wavelengths) * (1 + _Z), _WAVE, _R_CURVE)
    np.testing.assert_allclose(np.asarray(obs.eline_resolution), expected_r, rtol=1e-12)
    got = float(batched_log_likelihood(template, spec)(amps, obs))

    single = _fitted_model(Fixed(_Z))
    fitter = Fitter(
        single,
        jnp.asarray(flux),
        jnp.full(_WAVE.size, _SIGMA_NOISE),
        data_type="spectroscopy",
        approx=None,
    )
    ref = float(build_loglikelihood_fn(fitter)(amps, fitter._data_args))
    assert got == pytest.approx(ref, rel=1e-8)
