# SPDX-License-Identifier: BSD-3-Clause
"""#2941: the measured line-flux likelihood carries Cue's per-line emulator error.

End-to-end on the shipped Cue weights (no stubs): the Fitter packs Cue's
published fractional sigma for each measured line, the line-flux channel is
scored with sigma_eff = hypot(sigma_obs, s |F_model|) and the ln sigma_eff
term kept, and the same channel with ``line_model_sigma=False`` reproduces the
observation-only chi-square. The C IV line, whose published error is 144 per
cent, is strongly down-weighted against H-alpha.

The model is built as in ``test_line_flux_scaling.py``: Cue nebular, all
parameters Fixed, z = 0.2, constant SFH, a few DES bands. Truth fluxes come
from ``predict_line_fluxes``; the observed fluxes are scaled by 1.1 so the
model and the data differ.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tengri
from tengri import DEFAULT, Fixed, LogNormal, Observation, Photometry, parse_groups
from tengri.inference.fitter import Fitter
from tengri.inference.loss_functions import build_loglikelihood_fn
from tengri.observation.line_flux_data import LineFluxData
from tests._data_skip import requires_cue_weights

pytestmark = pytest.mark.contract

_BANDS = ["des_g", "des_r", "des_i"]
_LINE_NAMES = ("Halpha", "Hbeta", "CIV_1549")
_SIGMA_FRAC_OBS = 0.02
_DATA_SCALE = 1.1


def _model(ssp, obs, *, free_scale=False):
    groups = dict(
        redshift=Fixed(0.2),
        sfh={
            "type": "const",
            "all_params": Fixed(DEFAULT),
            "start_gyr": Fixed(1.0),
            "end_gyr": Fixed(0.0),
        },
        neb={"type": "cue", "all_params": Fixed(DEFAULT)},
    )
    spec = parse_groups(**groups)
    if free_scale:
        spec = spec.merge_observation_params(line_flux_scaling=LogNormal(mu=0.0, sigma=0.05))
    return tengri.SEDModel(spec, ssp, observation=obs)


def _cue_sigma_by_wavelength(model, target_waves):
    """Cue's published fractional sigma matched to each target by nearest wavelength."""
    backend = model._nebular_backend
    catalog_waves = np.asarray(backend.published_line_wavelengths(cloudyfsps_only=False))
    catalog_sigma = np.asarray(backend.published_line_model_sigma(cloudyfsps_only=False))
    idx = np.argmin(np.abs(catalog_waves[None, :] - target_waves[:, None]), axis=1)
    return catalog_sigma[idx].astype(np.float64)


@pytest.fixture(scope="module")
def ssp():
    return tengri.load_ssp("fsps_prsc_miles_chabrier", download=False)


@pytest.fixture(scope="module")
def truth_setup(ssp):
    """Truth model, its photometry, and the model's own line fluxes at truth."""
    lines = LineFluxData.from_dict({n: (1e-16, 1e-17) for n in _LINE_NAMES})
    obs = Observation(photometry=Photometry.from_names(_BANDS), line_fluxes=lines)
    model = _model(ssp, obs)
    truth = dict(model.spec.sample(jax.random.PRNGKey(0)))
    waves = np.asarray(lines.wavelengths, dtype=np.float64)
    model_lines = np.asarray(model.predict_line_fluxes(truth, target_wavelengths=waves))
    phot = np.asarray(model.predict_photometry(truth))
    return {
        "ssp": ssp,
        "truth": truth,
        "waves": waves,
        "model_lines": model_lines,
        "phot": phot,
        "names": lines.names,
    }


def _line_data(setup, *, names=_LINE_NAMES, scale=_DATA_SCALE):
    """Observed line fluxes at ``scale`` times the truth, 2 per cent errors."""
    keep = [i for i, n in enumerate(setup["names"]) if n in names]
    flux = setup["model_lines"][keep] * scale
    err = _SIGMA_FRAC_OBS * setup["model_lines"][keep]
    waves = setup["waves"][keep]
    return LineFluxData(
        names=tuple(setup["names"][i] for i in keep),
        fluxes=jnp.asarray(flux),
        errors=jnp.asarray(err),
        wavelengths=jnp.asarray(waves),
    )


def _fitter(setup, *, line_model_sigma=None, names=_LINE_NAMES, free_scale=False):
    obs = Observation(
        photometry=Photometry.from_names(_BANDS),
        line_fluxes=_line_data(setup, names=names),
    )
    model = _model(setup["ssp"], obs, free_scale=free_scale)
    noise = 0.05 * np.abs(setup["phot"])
    # ``None`` leaves the Fitter default in force, so the default path is what is tested.
    extra = {} if line_model_sigma is None else {"line_model_sigma": line_model_sigma}
    return Fitter(
        model,
        data=jnp.asarray(setup["phot"]),
        noise=jnp.asarray(noise),
        **extra,
    )


def _line_channel_ll(setup, fitter, params):
    """Line-channel log-likelihood: full minus the photometry-only value (as in #2527)."""
    phot_only = Observation(photometry=Photometry.from_names(_BANDS))
    model_photonly = _model(setup["ssp"], phot_only)
    fitter_photonly = Fitter(
        model_photonly,
        data=jnp.asarray(setup["phot"]),
        noise=jnp.asarray(0.05 * np.abs(setup["phot"])),
    )
    ll_full = build_loglikelihood_fn(fitter)
    ll_photonly = build_loglikelihood_fn(fitter_photonly)
    base_photonly = {
        k: v for k, v in setup["truth"].items() if k in model_photonly.spec.free_params
    }
    full = float(ll_full(params, fitter._data_args))
    photometry = float(ll_photonly(base_photonly, fitter_photonly._data_args))
    return full - photometry


@requires_cue_weights
def test_packed_sigma_is_cues_published_error_by_wavelength(truth_setup):
    fitter = _fitter(truth_setup)
    packed = np.asarray(fitter._data_args["line_flux_model_sigma_frac"])
    published = _cue_sigma_by_wavelength(fitter.model, truth_setup["waves"])
    np.testing.assert_allclose(packed, published, rtol=1e-6)

    # Pinned to three significant figures: Halpha, Hbeta, C IV 1548 (matched by wavelength).
    assert f"{packed[0]:.3g}" == "0.0119"
    assert f"{packed[1]:.3g}" == "0.00865"
    assert f"{packed[2]:.3g}" == "1.44"


@requires_cue_weights
def test_line_channel_is_censored_with_hypot_model_error(truth_setup):
    fitter = _fitter(truth_setup)
    m = np.asarray(
        fitter.model.predict_line_fluxes(
            truth_setup["truth"], target_wavelengths=truth_setup["waves"]
        )
    )
    d = np.asarray(fitter._data_args["line_flux_obs"])
    sigma_obs = np.asarray(fitter._data_args["line_flux_err"])
    s = np.asarray(fitter._data_args["line_flux_model_sigma_frac"])

    se = np.hypot(sigma_obs, s * np.abs(m))
    expected = -np.sum(0.5 * ((d - m) / se) ** 2 + np.log(se))

    free = set(fitter.model.spec.free_params)
    params = {k: v for k, v in truth_setup["truth"].items() if k in free}
    got = _line_channel_ll(truth_setup, fitter, params)
    assert got == pytest.approx(expected, rel=1e-6)


@requires_cue_weights
def test_line_channel_without_model_error_is_the_chi_square(truth_setup):
    fitter = _fitter(truth_setup, line_model_sigma=False)
    assert "line_flux_model_sigma_frac" not in fitter._data_args
    m = np.asarray(
        fitter.model.predict_line_fluxes(
            truth_setup["truth"], target_wavelengths=truth_setup["waves"]
        )
    )
    d = np.asarray(fitter._data_args["line_flux_obs"])
    sigma_obs = np.asarray(fitter._data_args["line_flux_err"])

    expected = -0.5 * np.sum(((d - m) / sigma_obs) ** 2)
    free = set(fitter.model.spec.free_params)
    params = {k: v for k, v in truth_setup["truth"].items() if k in free}
    got = _line_channel_ll(truth_setup, fitter, params)
    assert got == pytest.approx(expected, rel=1e-6)


@requires_cue_weights
def test_civ_gradient_is_suppressed_by_its_model_error(truth_setup):
    """C IV: first-order data weight is sigma_obs^2 / se^2, which is ~2e-4 here."""
    civ = ("CIV_1549",)
    fitter = _fitter(truth_setup, names=civ, free_scale=True)
    m = float(
        np.asarray(
            fitter.model.predict_line_fluxes(
                truth_setup["truth"], target_wavelengths=truth_setup["waves"][2:3]
            )
        )[0]
    )
    d = float(np.asarray(fitter._data_args["line_flux_obs"])[0])
    sigma_obs = float(np.asarray(fitter._data_args["line_flux_err"])[0])
    f = float(np.asarray(fitter._data_args["line_flux_model_sigma_frac"])[0])

    b = sigma_obs**2 + f**2 * m**2
    data_term = m * (d - m) / b
    cross_term = (d - m) ** 2 * f**2 * m**2 / b**2
    penalty = -(f**2) * m**2 / b
    expected = data_term + cross_term + penalty

    ll = build_loglikelihood_fn(fitter)
    base = {
        k: v for k, v in truth_setup["truth"].items() if k in set(fitter.model.spec.free_params)
    }

    def ll_of_scale(s):
        p = dict(base)
        p["line_flux_scaling"] = s
        return ll(p, fitter._data_args)

    grad = float(jax.grad(ll_of_scale)(jnp.asarray(1.0)))
    assert grad == pytest.approx(expected, rel=1e-6)

    suppression = sigma_obs**2 / b
    assert suppression < 1e-3
    assert suppression == pytest.approx(1.9e-4, rel=0.05)
