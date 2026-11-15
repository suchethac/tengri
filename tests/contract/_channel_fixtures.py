# SPDX-License-Identifier: BSD-3-Clause
"""Small Fitters, one per likelihood structure, for the channel contract tests.

Each builder returns a :class:`~tengri.Fitter` on the 8x15x200 synthetic SSP grid whose data
sit within a few sigma of the model at the prior's reference draw, so every channel is scored
at an O(1) residual rather than at the underflowed ``data = 1`` of the cache-key fixtures.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from tengri import Fitter, Fixed, Parameters, SEDModel, Uniform
from tengri.observation.line_list import LineList
from tengri.observation.observation import Observation
from tengri.observation.photometry_config import Photometry
from tengri.observation.spectroscopy import Spectroscopy
from tests.contract.test_inference_cache_keys import _small_ssp

BANDS = ["sdss_g", "sdss_r", "sdss_i"]
N_PIX = 40
WAVE_OBS = np.linspace(4000.0, 8000.0, N_PIX)

#: Fitter kinds the builders produce.
KINDS = (
    "phot",
    "phot_student_t",
    "phot_floor",
    "presence",
    "mask",
    "mask_student_t",
    "mask_floor",
    "spec",
    "spec_student_t",
    "spec_dense",
    "spec_banded",
    "spec_cal",
    "spec_lines",
    "spec_lines_fitted",
    "spec_cal_lines",
    "joint",
    "joint_mask",
)

#: Structures with no golden values, checked against independent references instead.
EXTRA_KINDS = (
    "presence_student_t",
    "presence_floor",
    "mask_presence",
    "spec_dense_cal",
    "spec_banded_cal",
    "spec_dense_lines",
    "spec_dense_lines_fitted",
    "spec_student_t_cal",
)


def _parameters(noise: str | None) -> Parameters:
    extra: dict = {}
    if noise == "student_t":
        extra = {"noise_dof": Fixed(4.0), "noise_frac_cal": Uniform(0.0, 0.3)}
    elif noise == "floor":
        extra = {"noise_frac_cal": Uniform(0.0, 0.3)}
    return Parameters(
        redshift=0.1, sfh_dpl_alpha=Uniform(0.5, 4.0), sfh_dpl_beta=Uniform(0.3, 3.0), **extra
    )


def _reference_free(model) -> dict:
    """The prior's reference draw, the point the data are built around."""
    return dict(model.spec.sample(jax.random.PRNGKey(0)))


def _perturb(prediction, scale: float):
    """Data a few sigma off the prediction, with errors that are a fixed fraction of it."""
    prediction = np.asarray(prediction)
    wiggle = 1.0 + 0.2 * np.sin(np.arange(prediction.size))
    return jnp.asarray(prediction * wiggle), jnp.asarray(np.abs(prediction) * scale)


def _covariance(noise: np.ndarray, *, banded: bool) -> np.ndarray:
    n = noise.size
    cov = np.diag(noise**2)
    for lag, rho in ((1, 0.3), (2, 0.1)) if banded else tuple((k, 0.5**k) for k in range(1, n)):
        idx = np.arange(n - lag)
        cov[idx, idx + lag] = rho * noise[idx] * noise[idx + lag]
        cov[idx + lag, idx] = cov[idx, idx + lag]
    return cov


def _photometry_model(noise: str | None):
    obs = Observation(photometry=Photometry.from_names(BANDS))
    return SEDModel(_parameters(noise), _small_ssp(), observation=obs)


def _photometry_fitter(noise=None, *, mask=None, presence=None) -> Fitter:
    model = _photometry_model(noise)
    pred = model.predict_photometry(_reference_free(model))
    data, err = _perturb(pred, 0.1)
    if presence is not None:
        # An absent band is stored with zero flux and zero noise.
        absent = np.asarray(presence) == 0
        data = jnp.where(absent, 0.0, data)
        err = jnp.where(absent, 0.0, err)
    return Fitter(model, data, err, data_type="photometry", data_mask=mask, presence=presence)


def _spectroscopy_model(noise, *, eline=None, covariance=None):
    spec_cfg = Spectroscopy(
        wave_obs=WAVE_OBS,
        eline_mode=eline or "off",
        eline_catalog=LineList.default_13() if eline else None,
        covariance=covariance,
    )
    obs = Observation(spectroscopy=spec_cfg)
    return SEDModel(_parameters(noise), _small_ssp(), observation=obs)


def _spectroscopy_fitter(noise=None, *, eline=None, covariance="none", **fit_kw) -> Fitter:
    cov = None
    if covariance != "none":
        # The error floor that makes the covariance proper depends on the model, so the
        # reference spectrum is taken from an uncorrelated model first.
        plain = _spectroscopy_model(noise, eline=eline)
        _, err0 = _perturb(plain.predict_spectrum(_reference_free(plain)), 0.1)
        cov = _covariance(np.asarray(err0), banded=covariance == "banded")
    model = _spectroscopy_model(noise, eline=eline, covariance=cov)
    data, err = _perturb(model.predict_spectrum(_reference_free(model)), 0.1)
    return Fitter(model, data, err, data_type="spectroscopy", approx=None, **fit_kw)


def _joint_fitter(*, mask=None) -> Fitter:
    obs = Observation(
        photometry=Photometry.from_names(BANDS), spectroscopy=Spectroscopy(wave_obs=WAVE_OBS)
    )
    model = SEDModel(_parameters(None), _small_ssp(), observation=obs)
    free = _reference_free(model)
    pd, pe = _perturb(model.predict_photometry(free), 0.1)
    sd, se = _perturb(model.predict_spectrum(free), 0.1)
    return Fitter(
        model,
        jnp.concatenate([pd, sd]),
        jnp.concatenate([pe, se]),
        data_type="joint",
        data_mask=None if mask is None else jnp.asarray(mask),
        approx=None,
    )


def fitter_of(kind: str) -> Fitter:
    """Build the Fitter of one structure; see :data:`KINDS`."""
    if kind == "phot":
        return _photometry_fitter()
    if kind == "phot_student_t":
        return _photometry_fitter("student_t")
    if kind == "phot_floor":
        return _photometry_fitter("floor")
    if kind == "presence":
        return _photometry_fitter(presence=jnp.array([1.0, 0.0, 1.0]))
    if kind in ("presence_student_t", "presence_floor"):
        noise = kind.removeprefix("presence_")
        return _photometry_fitter(noise, presence=jnp.array([1.0, 0.0, 1.0]))
    if kind == "mask_presence":
        return _photometry_fitter(mask=jnp.array([0, 1, -1]), presence=jnp.array([1.0, 1.0, 0.0]))
    if kind in ("spec_dense_cal", "spec_banded_cal"):
        return _spectroscopy_fitter(covariance=kind.split("_")[1], calibration_marginalize=True)
    if kind == "spec_dense_lines":
        return _spectroscopy_fitter(covariance="dense", eline="marginalized")
    if kind == "spec_dense_lines_fitted":
        return _spectroscopy_fitter(covariance="dense", eline="fitted")
    if kind == "spec_student_t_cal":
        return _spectroscopy_fitter("student_t", calibration_marginalize=True)
    if kind.startswith("mask"):
        noise = {"mask": None, "mask_student_t": "student_t", "mask_floor": "floor"}[kind]
        return _photometry_fitter(noise, mask=jnp.array([0, 1, -1]))
    if kind == "spec":
        return _spectroscopy_fitter()
    if kind == "spec_student_t":
        return _spectroscopy_fitter("student_t")
    if kind in ("spec_dense", "spec_banded"):
        return _spectroscopy_fitter(covariance=kind.removeprefix("spec_"))
    if kind == "spec_cal":
        return _spectroscopy_fitter(calibration_marginalize=True)
    if kind == "spec_lines":
        return _spectroscopy_fitter(eline="marginalized")
    if kind == "spec_lines_fitted":
        return _spectroscopy_fitter(eline="fitted")
    if kind == "spec_cal_lines":
        return _spectroscopy_fitter(eline="marginalized", calibration_marginalize=True)
    if kind == "joint":
        return _joint_fitter()
    if kind == "joint_mask":
        return _joint_fitter(mask=np.r_[[0, 1, 0], np.zeros(N_PIX, dtype=int)])
    raise KeyError(kind)


def physical_params(fitter: Fitter) -> dict:
    """The resolved physical parameters of the prior's reference draw for ``fitter``."""
    physical = dict(fitter.spec.sample(jax.random.PRNGKey(0)))
    physical.update(fitter._fixed_values)
    return fitter.spec.resolve_mirrors(physical)


def build_term(fitter: Fitter):
    """The data term the Fitter builds, with its physical parameters at the reference draw."""
    from tengri.inference.data_term import build_data_term

    term = build_data_term(
        fitter.model,
        spec=fitter.spec,
        data_args=fitter._data_args,
        options=fitter.data_term_options(),
        likelihood=fitter._user_likelihood,
    )
    return term, physical_params(fitter)
