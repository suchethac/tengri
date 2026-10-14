# SPDX-License-Identifier: BSD-3-Clause
"""``approx="auto"`` selects the spectrum LUT where the LUT is exact (#2958).

Since the SpectrumPrecomp LUT applies sigma_v and the instrument stage like the
exact path (#2958, 6.6e-4 of exact at R = 2000), ``auto`` takes it for a
spectroscopy or joint fit when it can serve the pixels exactly:

(i)   the pixel integral is not required (point-sampled pixels, or pixels no
      wider than the model grid, #2530);
(ii)  ``sigma_v_kms`` is Fixed (a free one broadens per draw, ~15x the exact path);
(iii) there is no banded resolution matrix (the LUT cannot carry it).

Otherwise ``auto`` stays exact, and says so once for a free sigma_v. ``None``
and an explicit ``SpectrumPrecomp()`` are unchanged.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import (
    DEFAULT,
    Fixed,
    ForwardModel,
    Observation,
    Photometry,
    SEDModel,
    Uniform,
)
from tengri.inference.fitter import Fitter, _resolve_batch_fit_approx
from tengri.observation.banded import gaussian_resolution_bands
from tengri.observation.spectroscopy import Spectroscopy

pytestmark = pytest.mark.regression_bug

_WAVE_OBS = np.linspace(4000.0, 9000.0, 2000)
_Z = 0.1
_MADAU = {"type": "madau"}


def _spec(resample="point", wave_obs=_WAVE_OBS, resolution_matrix=None):
    return Spectroscopy(
        resample=resample,
        wave_obs=wave_obs,
        resolution=None if resolution_matrix is not None else 2000.0,
        resolution_matrix=resolution_matrix,
    )


def _build(ssp, *, spectro, sigma_v, photometry=None, approx=None):
    obs = Observation(photometry=photometry, spectroscopy=spectro)
    base = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        redshift=Fixed(_Z),
        igm=_MADAU,
        sfh={"type": "tsnorm", "all_params": Fixed(DEFAULT), "log_total_mass": Fixed(10.0)},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
        },
        neb={"type": "none"},
    )
    spec = base.spec.merge_observation_params(sigma_v_kms=sigma_v)
    return SEDModel(spec, ssp, observation=obs, approx=approx)


def _fwd(model):
    return ForwardModel.build(sed=model, observation=model.observation)


def _flux(model):
    fixed = model.spec.get_distribution("sigma_v_kms").is_fixed
    params = {} if fixed else {"sigma_v_kms": 150.0}
    return np.asarray(model.predict_spectrum(params), dtype=float)


def _fit(model, *, data_type="spectroscopy", **kwargs):
    flux = _flux(model)
    noise = np.full(flux.shape, 0.05 * np.max(np.abs(flux)))
    return Fitter(_fwd(model), flux, noise, data_type=data_type, **kwargs)


# (a) fixed sigma_v, point pixels: auto engages the spectrum LUT ---------------


def test_a_auto_fixed_sigma_resolves_to_the_spectrum_lut(ssp_data_wne):
    model = _build(ssp_data_wne, spectro=_spec(), sigma_v=Fixed(150.0))
    fitter = _fit(model)  # approx="auto" default
    assert fitter.model.approx.spectrum_precomp


# (b) free sigma_v: auto stays exact and says so once -------------------------


def test_b_auto_free_sigma_stays_exact_and_warns_once(ssp_data_wne):
    model = _build(ssp_data_wne, spectro=_spec(), sigma_v=Uniform(0.0, 2000.0))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fitter = _fit(model)
    assert not fitter.model.approx.spectrum_precomp
    hits = [w for w in caught if "faster for it" in str(w.message)]
    assert len(hits) == 1, [str(w.message) for w in caught]


# (c) pixels wider than the model grid: auto stays exact ----------------------


def test_c_auto_pixel_integral_stays_exact(ssp_data_wne):
    coarse = np.linspace(4000.0, 9000.0, 50)
    model = _build(
        ssp_data_wne, spectro=_spec(resample="auto", wave_obs=coarse), sigma_v=Fixed(150.0)
    )
    fitter = _fit(model)
    assert not fitter.model.approx.spectrum_precomp


# (d) banded resolution matrix: auto stays exact ------------------------------


def test_d_auto_banded_resolution_matrix_stays_exact(ssp_data_wne):
    rm = gaussian_resolution_bands(jnp.asarray(_WAVE_OBS), 2000.0, n_diag=11)
    model = _build(ssp_data_wne, spectro=_spec(resolution_matrix=rm), sigma_v=Fixed(150.0))
    fitter = _fit(model)
    assert not fitter.model.approx.spectrum_precomp


# (e) joint observation: both LUTs for fixed sigma_v, photometry only for free --


def _joint_obs_model(ssp, sigma_v):
    phot = Photometry.from_names(["des_g"])
    return _build(ssp, spectro=_spec(), sigma_v=sigma_v, photometry=phot)


def test_e_joint_fixed_sigma_resolves_to_both_luts(ssp_data_wne):
    model = _joint_obs_model(ssp_data_wne, Fixed(150.0))
    fitter = _fit(model, data_type="joint")
    assert fitter.model.approx.wave_precomp
    assert fitter.model.approx.spectrum_precomp


def test_e_joint_free_sigma_keeps_photometry_lut_only(ssp_data_wne):
    model = _joint_obs_model(ssp_data_wne, Uniform(0.0, 2000.0))
    fitter = _fit(model, data_type="joint")
    assert fitter.model.approx.wave_precomp
    assert not fitter.model.approx.spectrum_precomp


# (g) batch mirror: same decision as (a)-(d) ----------------------------------


@pytest.mark.parametrize(
    ("case", "expect_lut"),
    [("fixed", True), ("free", False), ("coarse", False), ("banded", False)],
)
def test_g_batch_resolver_matches_the_single_fit_decision(ssp_data_wne, case, expect_lut):
    if case == "fixed":
        model = _build(ssp_data_wne, spectro=_spec(), sigma_v=Fixed(150.0))
    elif case == "free":
        model = _build(ssp_data_wne, spectro=_spec(), sigma_v=Uniform(0.0, 2000.0))
    elif case == "coarse":
        coarse = np.linspace(4000.0, 9000.0, 50)
        model = _build(
            ssp_data_wne, spectro=_spec(resample="auto", wave_obs=coarse), sigma_v=Fixed(150.0)
        )
    else:
        rm = gaussian_resolution_bands(jnp.asarray(_WAVE_OBS), 2000.0, n_diag=11)
        model = _build(ssp_data_wne, spectro=_spec(resolution_matrix=rm), sigma_v=Fixed(150.0))
    resolved = _resolve_batch_fit_approx(model, "auto", "spectroscopy")
    assert bool(resolved.approx.spectrum_precomp) is expect_lut


# (f) accuracy at the fit surface: MAP under the LUT vs exact ------------------

_FREE = ("sfh_tsnorm_log_total_mass", "sfh_tsnorm_peak_lbt_frac", "dust_tau_diff")
_TRUTH = {"sfh_tsnorm_log_total_mass": 10.0, "sfh_tsnorm_peak_lbt_frac": 0.5, "dust_tau_diff": 0.4}


def _free_model(ssp, *, approx=None):
    """Exact-spec model with three free SFH/dust parameters and a fixed sigma_v."""
    obs = Observation(spectroscopy=_spec())
    base = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        redshift=Fixed(_Z),
        igm=_MADAU,
        sfh={
            "type": "tsnorm",
            "all_params": Fixed(DEFAULT),
            "log_total_mass": Uniform(9.0, 11.0),
            "peak_lbt_frac": Uniform(0.2, 0.9),
        },
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
            "tau_diff": Uniform(0.0, 2.0),
        },
        neb={"type": "none"},
    )
    spec = base.spec.merge_observation_params(sigma_v_kms=Fixed(150.0))
    return SEDModel(spec, ssp, observation=obs, approx=approx)


def _laplace_std(model, flux, noise, x0):
    def chi2(x):
        params = dict(zip(_FREE, x, strict=True))
        pred = model.predict_spectrum(params)
        return jnp.sum(((pred - flux) / noise) ** 2)

    x0 = jnp.asarray([x0[k] for k in _FREE], dtype=jnp.float64)
    hess = np.asarray(jax.hessian(chi2)(x0))
    cov = np.linalg.inv(0.5 * hess)  # -log L = chi2 / 2
    return np.sqrt(np.diag(cov))


def test_f_map_under_the_lut_agrees_with_exact(ssp_data_wne):
    exact = _free_model(ssp_data_wne, approx=None)
    truth = dict(_TRUTH)
    clean = np.asarray(exact.predict_spectrum(truth), dtype=float)
    rng = np.random.default_rng(2958)
    noise = 0.02 * np.abs(clean)
    flux = clean + noise * rng.standard_normal(clean.shape)

    sigma = _laplace_std(exact, jnp.asarray(flux), jnp.asarray(noise), truth)

    lut_fit = Fitter(_fwd(_free_model(ssp_data_wne)), flux, noise, data_type="spectroscopy")
    exact_fit = Fitter(_fwd(exact), flux, noise, data_type="spectroscopy", approx=None)
    assert lut_fit.model.approx.spectrum_precomp
    assert not exact_fit.model.approx.spectrum_precomp

    key = jax.random.PRNGKey(7)
    lut_map = lut_fit.run(method="map", key=key, n_restarts=2, n_steps=300, verbose=False)
    exact_map = exact_fit.run(method="map", key=key, n_restarts=2, n_steps=300, verbose=False)
    for i, name in enumerate(_FREE):
        delta = abs(
            float(np.asarray(lut_map.params[name])) - float(np.asarray(exact_map.params[name]))
        )
        print(
            f"(f) {name}: lut={float(np.asarray(lut_map.params[name])):.5f} "
            f"exact={float(np.asarray(exact_map.params[name])):.5f} "
            f"delta/sigma={delta / sigma[i]:.4f}"
        )
        assert delta <= 0.1 * sigma[i], name
