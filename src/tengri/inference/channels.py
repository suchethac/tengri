# SPDX-License-Identifier: BSD-3-Clause
"""The one per-galaxy likelihood scorer: model observables in, log-likelihood out.

An inference engine has two jobs. *Prediction* produces the model observables (band fluxes,
line fluxes, spectrum pixels, design columns) and is specific to the engine's batching and
compilation. *Scoring* turns those observables and the data into ``log p(d | params)`` and
is shared. :func:`score` is the scoring half: it sums every :class:`Channel` whose data keys
are present, under one :class:`NoiseSpec`. A data channel or noise model added here is
therefore available to every engine that calls :func:`score`.

Each channel's math is a pure function of arrays (``*_term``). The adapters in
:mod:`tengri.inference.likelihoods` and the fall-through of
:mod:`tengri.inference.data_term` call the same functions, so a term has one implementation.

Notes
-----
:func:`score` drops the Gaussian constants of the diagonal and censored terms, as the
posterior does not need them; :func:`log_normalization` returns what it dropped, so
``score + log_normalization`` is the exact density of the data. The marginalized terms
carry their normalization inside :func:`score` and add nothing here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from tengri.inference.likelihoods.gaussian import (
    LOG_2PI,
    diag_gaussian_log_normalization,
    diag_gaussian_log_prob,
    student_t_log_norm_per_point,
)
from tengri.inference.likelihoods.terms import (
    calibration_eline_marginal_term,
    calibration_marginal_term,
    censored_term,
    correlated_noise_term,
    correlated_term,
    eline_cloudy_marginal_term,
    eline_flat_marginal_term,
    eline_model_marginal_term,
    fitted_line_model,
    student_t_term,
)

__all__ = [
    "BOOKKEEPING_KEYS",
    "CHANNELS",
    "ENGINES_ON_SCORER",
    "CalibrationSpec",
    "Channel",
    "LineSpec",
    "NoiseSpec",
    "Predictions",
    "channel_keys",
    "data_keys",
    "log_normalization",
    "score",
    "selected_channels",
    "validate",
]

#: ``data_args`` entries that carry no data channel: sizes, derived whiteners and the traced
#: model inputs the prediction reads.
BOOKKEEPING_KEYS = frozenset({"n_data", "sqrt_noise_inv", "redshift", "_jit_inputs"})

_PHOT_FIELDS = ("phot_fnu", "spec_fnu", "line_fluxes", "line_ratios", "indices")


# ── Predictions and noise specification ───────────────────────────────────────────────


@dataclass(frozen=True)
class Predictions:
    """Model observables one galaxy's channels read, produced by any engine.

    Parameters
    ----------
    phot_fnu : array_like, shape (n_bands,), optional
        Predicted band flux densities [erg/s/cm2/Hz].
    spec_fnu : array_like, shape (n_pix,), optional
        Predicted spectrum pixels [erg/s/cm2/Hz].
    line_fluxes : array_like, shape (n_lines,), optional
        Predicted line fluxes, already scaled by ``line_flux_scaling``.
    line_ratios, indices : array_like, optional
        Predicted line ratios and spectral indices.
    log_line_lums_attenuated : array_like, optional
        Model line luminosities [log10 Lsun] centering a model-centered line prior.
    eline_design : array_like, shape (n_pix, n_lines), optional
        Emission-line design columns with the doublet constraints applied. Produced by the
        engine, so redshift and line width move it like any other observable.
    eline_prior_inputs : tuple, optional
        ``(redshift, dl_cm, aperture_frac, transmission)`` of a model-centered line prior.
    params : mapping, optional
        Resolved physical parameters the noise terms read (``noise_frac_cal``,
        ``cal_log_q``, line amplitudes, ``met_logzsol``, ``neb_logU``).

    Notes
    -----
    A plain container with no model attached: an engine fills the fields it has. It is
    registered as a pytree, so it passes through ``jit`` and ``vmap``.
    """

    phot_fnu: Any = None
    spec_fnu: Any = None
    line_fluxes: Any = None
    line_ratios: Any = None
    indices: Any = None
    log_line_lums_attenuated: Any = None
    eline_design: Any = None
    eline_prior_inputs: Any = None
    params: Mapping = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, prediction, params=None, *, eline_design=None, eline_prior_inputs=None):
        """Build from the channel dictionary of ``_build_prediction``."""
        return cls(
            phot_fnu=prediction.get("phot_fnu"),
            spec_fnu=prediction.get("spec_fnu"),
            line_fluxes=prediction.get("line_fluxes"),
            line_ratios=prediction.get("line_ratios"),
            indices=prediction.get("indices"),
            log_line_lums_attenuated=prediction.get("log_line_lums_attenuated"),
            eline_design=eline_design,
            eline_prior_inputs=eline_prior_inputs,
            params={} if params is None else params,
        )

    def as_mapping(self) -> dict:
        """The prediction dictionary the likelihood adapters read."""
        keys = (*_PHOT_FIELDS, "log_line_lums_attenuated")
        return {k: getattr(self, k) for k in keys if getattr(self, k) is not None}


jax.tree_util.register_dataclass(
    Predictions, data_fields=[f.name for f in fields(Predictions)], meta_fields=[]
)


@dataclass(frozen=True, eq=False)
class CalibrationSpec:
    """Calibration-polynomial marginalization of the spectrum.

    Parameters
    ----------
    wavelength : array_like, shape (n_pix,)
        Observed-frame pixel wavelengths [Angstrom] of the Chebyshev coordinate.
    n_poly : int, optional
        Highest Chebyshev order.
    prior_sigma : float, optional
        Gaussian prior width of each coefficient.
    segments, n_segments, cal_prior, q_window, has_q
        Camera-segment layout, prior family and fiber-fraction window, as in
        :func:`~tengri.observation.calibration.marginalize_calibration_segments`.
    """

    wavelength: Any
    n_poly: int = 3
    prior_sigma: float = 1.0
    segments: Any = None
    n_segments: int = 0
    cal_prior: str = "gaussian"
    q_window: Any = None
    has_q: bool = False


@dataclass(frozen=True, eq=False)
class LineSpec:
    """Emission-line amplitudes of the spectrum, marginalized or fitted.

    Parameters
    ----------
    mode : {"marginalized", "fitted"}
        Integrate the amplitudes out, or add them (``params[name]``) to the model.
    prior_type : {"flat", "cloudy", "model"}, optional
        Prior of marginalized amplitudes.
    prior_sigma : float or None, optional
        Width of the flat prior; ``None`` is ``1e10``. The marginal is exact at this finite
        width, not the improper-prior limit: each constrained line lowers the evidence by
        about ``ln prior_sigma`` nats, so compare models with different numbers of
        marginalized lines at the same width.
    prior_variance : float or None, optional
        Variance of the flat prior, taking precedence over ``prior_sigma``.
    line_wavelengths : array_like, optional
        Independent catalog wavelengths [Angstrom] of a Cloudy or model prior.
    prior_width_dex : float, optional
        Scatter [dex] of a Cloudy or model prior.
    match_index, matched : array_like, optional
        Catalog-to-model line matching of a model prior.
    amplitude_names : tuple of str, optional
        Parameters holding fitted amplitudes.
    """

    mode: str
    prior_type: str = "flat"
    prior_sigma: float | None = None
    prior_variance: float | None = None
    line_wavelengths: Any = None
    prior_width_dex: float = 0.3
    match_index: Any = None
    matched: Any = None
    amplitude_names: tuple = ()


@dataclass(frozen=True, eq=False)
class NoiseSpec:
    """The noise model one galaxy is scored under.

    Parameters
    ----------
    data_type : {"photometry", "spectroscopy", "joint"}
        Which blocks the data vector holds. A joint vector is photometry then spectrum.
    n_phot : int, optional
        Length of the photometry block of a joint vector.
    student_t : bool, optional
        Score the main data with the variable-noise energy (heavy tails and the
        calibration floor) instead of a plain Gaussian.
    dof : float or None, optional
        Student-t degrees of freedom; ``None`` with ``student_t`` is the floored Gaussian.
    f_cal_param : str or None, optional
        Parameter holding the fractional calibration floor.
    covariance : {"diagonal", "dense", "banded"}, optional
        Pixel covariance of the spectrum. Dense reads ``spec_cov_inv``, banded reads
        ``spec_chol_data`` with ``chol_offsets``.
    chol_offsets : tuple of int, optional
        Diagonal offsets of the banded factor.
    calibration : CalibrationSpec or None, optional
        Marginalize the calibration polynomial.
    lines : LineSpec or None, optional
        Marginalize or fit the emission-line amplitudes.

    Notes
    -----
    Static structure only; arrays inside it are model-level inputs, not per-galaxy data.
    """

    data_type: str = "photometry"
    n_phot: int = 0
    student_t: bool = False
    dof: float | None = None
    f_cal_param: str | None = None
    covariance: str = "diagonal"
    chol_offsets: tuple | None = None
    calibration: CalibrationSpec | None = None
    lines: LineSpec | None = None


# ── Channels ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Channel:
    """One additive term of the log-likelihood.

    Parameters
    ----------
    name : str
        Registry name.
    keys : tuple of str
        ``data`` entries the channel needs; it is scored only when all are present.
    loglik : callable
        ``loglik(preds, data, noise)``, a scalar for one galaxy.
    gaussian_linear : bool
        Whether the term is linear-Gaussian in the predictions, so a linear-Gaussian
        engine can score it in closed form.
    optional_keys : tuple of str, optional
        ``data`` entries it reads when present.
    applies : callable, optional
        ``applies(data, noise)``: whether the noise model selects this channel.
    supersedes : tuple of str, optional
        Channels it replaces when it applies.
    log_norm : callable, optional
        ``log_norm(data, noise)``: the constant ``loglik`` omits [nats]; zero for a term
        that already returns its normalization.
    """

    name: str
    keys: tuple
    loglik: Callable
    gaussian_linear: bool
    optional_keys: tuple = ()
    applies: Callable = lambda data, noise: True
    supersedes: tuple = ()
    log_norm: Callable = lambda data, noise: 0.0


def _part(arr, block, noise):
    """The photometry or spectrum block of a data array."""
    if noise.data_type != "joint":
        return arr
    return arr[: noise.n_phot] if block == "phot" else arr[noise.n_phot :]


def _f_cal(preds, noise):
    name = noise.f_cal_param
    if name is not None and name in preds.params:
        return preds.params[name]
    return 0.0


def _floor(preds, noise):
    """The calibration floor a marginal whitens with, or ``None`` when there is none."""
    if noise.student_t and noise.dof is None:
        return _f_cal(preds, noise)
    return None


def _has_mask(data):
    return "data_mask" in data


def _is_phot(noise):
    return noise.data_type in ("photometry", "joint")


def _is_spec(noise):
    return noise.data_type in ("spectroscopy", "joint")


# photometry ---------------------------------------------------------------------------


def _photometry_applies(data, noise):
    return _is_phot(noise)


def _present(data, noise):
    """Boolean presence of the photometric bands, or ``None`` when every band is present."""
    if "presence" in data:
        return data["presence"] > 0
    return None


def _present_over_data(data, noise):
    """Presence over the whole data vector: spectrum pixels of a joint vector are present."""
    present = _present(data, noise)
    if present is None or noise.data_type != "joint":
        return present
    n_spec = jnp.shape(data["data"])[0] - noise.n_phot
    return jnp.concatenate([present, jnp.ones(n_spec, dtype=bool)])


def _check_presence(data, noise):
    """Refuse a ``presence`` entry that no photometric block can apply."""
    if "presence" not in data:
        return
    if not _is_phot(noise):
        raise ValueError(
            "a presence entry masks photometric bands, and this data has none "
            f"(data_type={noise.data_type!r}). The data term has no per-pixel mask for a "
            "spectrum; drop the pixels, or mask them in the batched path with pix_mask."
        )
    n_bands = jnp.shape(data["data"])[0] if noise.data_type == "photometry" else noise.n_phot
    if jnp.shape(data["presence"])[0] != n_bands:
        raise ValueError(
            f"presence has {jnp.shape(data['presence'])[0]} entries; the photometric block "
            f"has {n_bands} bands"
        )


def _phot_inputs(preds, data, noise):
    """Photometry prediction, data and error with absent bands neutralized.

    An absent band stores zero flux and zero noise, so it is replaced before any division,
    logarithm or ``hypot``: prediction, data and mask-free residual are zero and the error
    is one. It then adds exactly zero to every noise family and carries no gradient.
    """
    obs, err = _part(data["data"], "phot", noise), _part(data["noise"], "phot", noise)
    present = _present(data, noise)
    pred = preds.phot_fnu
    if present is None:
        return pred, obs, err, None
    return (
        jnp.where(present, pred, 0.0),
        jnp.where(present, obs, 0.0),
        jnp.where(present, err, 1.0),
        present,
    )


def _student_t_with_presence(pred, obs, err, f_cal, dof, present):
    value = student_t_term(pred, obs, err, f_cal, dof)
    if present is None:
        return value
    # A neutralized band still carries the per-point Student-t constant; remove it.
    zero = jnp.zeros(1)
    baseline = student_t_term(zero, zero, jnp.ones(1), 0.0, dof)
    return value - jnp.sum(jnp.logical_not(present)) * baseline


def _n_present(obs, present):
    return jnp.size(obs) if present is None else jnp.sum(present)


def _photometry_loglik(preds, data, noise):
    pred, obs, err, present = _phot_inputs(preds, data, noise)
    if noise.student_t:
        return _student_t_with_presence(pred, obs, err, _f_cal(preds, noise), noise.dof, present)
    return diag_gaussian_log_prob(pred, obs, err)


def _photometry_norm(data, noise):
    obs = _part(data["data"], "phot", noise)
    err = _part(data["noise"], "phot", noise)
    present = _present(data, noise)
    if noise.student_t:
        return -0.5 * _n_present(obs, present) * LOG_2PI
    if present is not None:
        err = jnp.where(present, err, 1.0)
    return diag_gaussian_log_normalization(jnp.hypot(err, 0.0 * obs), present)


def _censoring_applies(data, noise):
    return _has_mask(data)


def _censored_prediction(preds, noise):
    if noise.data_type == "photometry":
        return preds.phot_fnu
    if noise.data_type == "spectroscopy":
        return _spectrum_model(preds, noise)
    return jnp.concatenate([preds.phot_fnu, _spectrum_model(preds, noise)])


def _censoring_loglik(preds, data, noise):
    pred, obs, err = _censored_prediction(preds, noise), data["data"], data["noise"]
    mask = data["data_mask"]
    present = _present_over_data(data, noise)
    if present is not None:
        pred = jnp.where(present, pred, 0.0)
        obs = jnp.where(present, obs, 0.0)
        err = jnp.where(present, err, 1.0)
        mask = jnp.where(present, mask, 0)
    return censored_term(pred, obs, err, mask, _f_cal(preds, noise), noise.dof)


def _censoring_norm(data, noise):
    detected = data["data_mask"] == 0
    present = _present_over_data(data, noise)
    if present is not None:
        detected = jnp.logical_and(detected, present)
    n_detected = jnp.sum(detected)
    if noise.dof is not None:
        return n_detected * student_t_log_norm_per_point(noise.dof)
    return -0.5 * n_detected * LOG_2PI


# spectrum -----------------------------------------------------------------------------


def _fitted(noise):
    return noise.lines is not None and noise.lines.mode == "fitted"


def _marginalizes(noise):
    lines = noise.lines is not None and noise.lines.mode == "marginalized"
    return noise.calibration is not None or lines


def _spectrum_applies(data, noise):
    if not _is_spec(noise) or _has_mask(data):
        return False
    return not _marginalizes(noise)


def _spectrum_model(preds, noise):
    """The predicted spectrum, with fitted line amplitudes added when they are latent."""
    if not _fitted(noise):
        return preds.spec_fnu
    amplitudes = jnp.array([preds.params[name] for name in noise.lines.amplitude_names])
    return fitted_line_model(preds.spec_fnu, preds.eline_design, amplitudes)


def _chol(data, noise):
    """Cholesky bands and offsets of the pixel covariance a marginal whitens with."""
    if noise.covariance == "banded":
        return data["spec_chol_data"], noise.chol_offsets
    if noise.covariance == "dense":
        bands = data["spec_cov_chol_data"]
        return bands, tuple(-k for k in range(bands.shape[1]))
    return None, None


def _spectrum_loglik(preds, data, noise):
    obs, err = _part(data["data"], "spec", noise), _part(data["noise"], "spec", noise)
    model = _spectrum_model(preds, noise)
    if noise.student_t and noise.covariance != "diagonal":
        banded = noise.covariance == "banded"
        return correlated_noise_term(
            model,
            obs,
            cov_inv=None if banded else data["spec_cov_inv"],
            chol_data=data["spec_chol_data"] if banded else None,
            chol_offsets=noise.chol_offsets if banded else None,
            f_cal=_f_cal(preds, noise),
            dof=noise.dof,
        )
    if noise.student_t:
        return student_t_term(model, obs, err, _f_cal(preds, noise), noise.dof)
    if noise.covariance == "diagonal":
        return diag_gaussian_log_prob(model, obs, err)
    if noise.covariance == "banded":
        return correlated_term(model, obs, None, data["spec_chol_data"], noise.chol_offsets)
    return correlated_term(model, obs, data["spec_cov_inv"], None, None)


def _spectrum_norm(data, noise):
    obs = _part(data["data"], "spec", noise)
    if noise.student_t and noise.covariance != "diagonal":
        return 0.0
    if noise.student_t:
        return -0.5 * jnp.size(obs) * LOG_2PI
    if noise.covariance == "diagonal":
        err = _part(data["noise"], "spec", noise)
        return diag_gaussian_log_normalization(jnp.hypot(err, 0.0 * obs), None)
    if noise.covariance == "dense":
        log_det = -jnp.linalg.slogdet(data["spec_cov_inv"])[1]
    else:
        chol = data["spec_chol_data"]
        diagonal = chol[int(np.where(np.asarray(noise.chol_offsets, dtype=int) == 0)[0][0])]
        log_det = 2.0 * jnp.sum(jnp.log(diagonal))
    return -0.5 * log_det - 0.5 * jnp.size(obs) * LOG_2PI


def _marginal_active(data, noise):
    heavy_tailed = noise.student_t and noise.dof is not None
    return _is_spec(noise) and not _has_mask(data) and not heavy_tailed


def _lines_marginalized(noise):
    return noise.lines is not None and noise.lines.mode == "marginalized"


def _calibration_applies(data, noise):
    return (
        _marginal_active(data, noise)
        and noise.calibration is not None
        and not _lines_marginalized(noise)
    )


def _calibration_loglik(preds, data, noise):
    cal = noise.calibration
    chol, offsets = _chol(data, noise)
    return calibration_marginal_term(
        preds.spec_fnu,
        _part(data["data"], "spec", noise),
        _part(data["noise"], "spec", noise),
        cal.wavelength,
        n_poly=cal.n_poly,
        prior_sigma=cal.prior_sigma,
        segments=cal.segments,
        n_segments=cal.n_segments,
        cal_prior=cal.cal_prior,
        log_q=preds.params["cal_log_q"] if cal.has_q else None,
        q_window=cal.q_window,
        has_q=cal.has_q,
        chol_data=chol,
        chol_offsets=offsets,
        f_cal=_floor(preds, noise),
    )


def _flat_variance(lines):
    if lines.prior_variance is not None:
        return lines.prior_variance
    return (lines.prior_sigma or 1e10) ** 2


def _lines_applies(data, noise):
    return (
        _marginal_active(data, noise) and _lines_marginalized(noise) and noise.calibration is None
    )


def _lines_loglik(preds, data, noise):
    lines = noise.lines
    obs, err = _part(data["data"], "spec", noise), _part(data["noise"], "spec", noise)
    model = preds.spec_fnu
    design = preds.eline_design
    chol, offsets = _chol(data, noise)
    if lines.prior_type == "cloudy":
        return eline_cloudy_marginal_term(
            model,
            obs,
            err,
            design,
            log_z=preds.params["met_logzsol"],
            neb_logU=preds.params["neb_logU"],
            line_wavelengths=lines.line_wavelengths,
            prior_width_dex=lines.prior_width_dex,
            chol_data=chol,
            chol_offsets=offsets,
            f_cal=_floor(preds, noise),
        )
    if lines.prior_type == "model":
        return eline_model_marginal_term(
            model,
            obs,
            err,
            design,
            log_line_lums=preds.log_line_lums_attenuated,
            match_index=lines.match_index,
            matched=lines.matched,
            line_wavelengths=lines.line_wavelengths,
            prior_inputs=preds.eline_prior_inputs,
            prior_width_dex=lines.prior_width_dex,
            chol_data=chol,
            chol_offsets=offsets,
            f_cal=_floor(preds, noise),
        )
    return eline_flat_marginal_term(
        model,
        obs,
        err,
        design,
        prior_variance=_flat_variance(lines),
        chol_data=chol,
        chol_offsets=offsets,
        f_cal=_floor(preds, noise),
    )


def _calibration_lines_applies(data, noise):
    return (
        _marginal_active(data, noise)
        and _lines_marginalized(noise)
        and noise.calibration is not None
    )


def _calibration_lines_loglik(preds, data, noise):
    cal, lines = noise.calibration, noise.lines
    chol, offsets = _chol(data, noise)
    return calibration_eline_marginal_term(
        preds.spec_fnu,
        _part(data["data"], "spec", noise),
        _part(data["noise"], "spec", noise),
        preds.eline_design,
        cal.wavelength,
        n_poly=cal.n_poly,
        prior_sigma=cal.prior_sigma,
        prior_type=lines.prior_type,
        line_prior_sigma=lines.prior_sigma or 1e10,
        line_wavelengths=lines.line_wavelengths,
        prior_width_dex=lines.prior_width_dex,
        params=preds.params,
        log_line_lums=preds.log_line_lums_attenuated,
        match_index=lines.match_index,
        matched=lines.matched,
        prior_inputs=preds.eline_prior_inputs,
        segments=cal.segments,
        n_segments=cal.n_segments,
        cal_prior=cal.cal_prior,
        q_window=cal.q_window,
        has_q=cal.has_q,
        chol_data=chol,
        chol_offsets=offsets,
        f_cal=_floor(preds, noise),
    )


# constraint channels ------------------------------------------------------------------


def _line_gaussian_applies(data, noise):
    return "line_flux_limit_mask" not in data and "line_flux_model_sigma_frac" not in data


def _line_gaussian_loglik(preds, data, noise):
    return diag_gaussian_log_prob(preds.line_fluxes, data["line_flux_obs"], data["line_flux_err"])


def _line_gaussian_norm(data, noise):
    err, obs = data["line_flux_err"], data["line_flux_obs"]
    return diag_gaussian_log_normalization(jnp.hypot(err, 0.0 * obs), None)


def _line_censor_applies(data, noise):
    return not _line_gaussian_applies(data, noise)


def _line_mask(data):
    if "line_flux_limit_mask" in data:
        return data["line_flux_limit_mask"]
    return jnp.zeros(data["line_flux_obs"].shape[0], dtype=jnp.int32)


def _line_censor_loglik(preds, data, noise):
    return censored_term(
        preds.line_fluxes,
        data["line_flux_obs"],
        data["line_flux_err"],
        _line_mask(data),
        0.0,
        None,
        data.get("line_flux_model_sigma_frac"),
    )


def _line_censor_norm(data, noise):
    return -0.5 * jnp.sum(_line_mask(data) == 0) * LOG_2PI


def _ratios_loglik(preds, data, noise):
    return diag_gaussian_log_prob(
        preds.line_ratios, data["line_ratio_obs"], data["line_ratio_err"]
    )


def _ratios_norm(data, noise):
    err, obs = data["line_ratio_err"], data["line_ratio_obs"]
    return diag_gaussian_log_normalization(jnp.hypot(err, 0.0 * obs), None)


def _indices_loglik(preds, data, noise):
    return diag_gaussian_log_prob(preds.indices, data["index_obs"], data["index_err"])


def _indices_norm(data, noise):
    err, obs = data["index_err"], data["index_obs"]
    return diag_gaussian_log_normalization(jnp.hypot(err, 0.0 * obs), None)


_BASE = ("data", "noise")

#: Every channel, in summation order: the main data, the spectrum, then the constraints.
CHANNELS: tuple = (
    Channel(
        "photometry",
        _BASE,
        _photometry_loglik,
        True,
        optional_keys=("presence",),
        applies=_photometry_applies,
        log_norm=_photometry_norm,
    ),
    Channel(
        "censoring",
        (*_BASE, "data_mask"),
        _censoring_loglik,
        False,
        optional_keys=("presence",),
        applies=_censoring_applies,
        supersedes=("photometry",),
        log_norm=_censoring_norm,
    ),
    Channel(
        "spectrum",
        _BASE,
        _spectrum_loglik,
        True,
        optional_keys=("spec_cov_inv", "spec_chol_data", "spec_cov_chol_data"),
        applies=_spectrum_applies,
        log_norm=_spectrum_norm,
    ),
    Channel(
        "calibration",
        _BASE,
        _calibration_loglik,
        True,
        optional_keys=("spec_chol_data", "spec_cov_chol_data"),
        applies=_calibration_applies,
    ),
    Channel(
        "lines",
        _BASE,
        _lines_loglik,
        True,
        optional_keys=("spec_chol_data", "spec_cov_chol_data"),
        applies=_lines_applies,
    ),
    Channel(
        "calibration_lines",
        _BASE,
        _calibration_lines_loglik,
        True,
        optional_keys=("spec_chol_data", "spec_cov_chol_data"),
        applies=_calibration_lines_applies,
    ),
    Channel(
        "line_fluxes",
        ("line_flux_obs", "line_flux_err", "line_flux_waves"),
        _line_gaussian_loglik,
        True,
        applies=_line_gaussian_applies,
        log_norm=_line_gaussian_norm,
    ),
    Channel(
        "line_censor",
        ("line_flux_obs", "line_flux_err", "line_flux_waves"),
        _line_censor_loglik,
        False,
        optional_keys=("line_flux_limit_mask", "line_flux_model_sigma_frac"),
        applies=_line_censor_applies,
        supersedes=("line_fluxes",),
        log_norm=_line_censor_norm,
    ),
    Channel(
        "line_ratios",
        ("line_ratio_obs", "line_ratio_err"),
        _ratios_loglik,
        True,
        log_norm=_ratios_norm,
    ),
    Channel(
        "spectral_indices",
        ("index_obs", "index_err"),
        _indices_loglik,
        True,
        log_norm=_indices_norm,
    ),
)


def channel_keys() -> dict:
    """The channel registry: channel name to :class:`Channel`, derived from :data:`CHANNELS`."""
    return {channel.name: channel for channel in CHANNELS}


def data_keys() -> frozenset:
    """Every ``data`` entry some channel reads, bookkeeping entries excluded."""
    keys = set()
    for channel in CHANNELS:
        keys.update(channel.keys, channel.optional_keys)
    return frozenset(keys)


def selected_channels(data: Mapping, noise: NoiseSpec) -> list:
    """The channels that score this ``data`` under ``noise``, superseded ones removed."""
    active = [c for c in CHANNELS if all(k in data for k in c.keys) and c.applies(data, noise)]
    superseded = {name for c in active for name in c.supersedes}
    return [c for c in active if c.name not in superseded]


def _refuse_unscorable(noise: NoiseSpec) -> None:
    """Refuse noise models with no closed-form score, naming the conflict."""
    lines = noise.lines
    if noise.calibration is not None and lines is not None and lines.mode == "fitted":
        raise NotImplementedError(
            "calibration_marginalize=True cannot be combined with eline_mode='fitted': the "
            "two cannot share one marginal. Use eline_mode='marginalized' (with optional "
            "eline_prior_type) for the standard Prospector-style configuration, or set "
            "calibration_marginalize=False."
        )
    marginalized = noise.calibration is not None or (
        lines is not None and lines.mode == "marginalized"
    )
    if noise.student_t and marginalized and noise.dof is not None:
        raise NotImplementedError(
            "A Student-t noise_dof cannot be combined with calibration_marginalize=True or "
            "eline_mode='marginalized': the marginal has no closed form under heavy-tailed "
            "noise. Drop noise_dof for Gaussian noise, or fit the nuisance coefficients "
            "explicitly (calibration_order > 0 with calibration_marginalize=False, "
            "eline_mode='fitted')."
        )


def validate(noise: NoiseSpec, data_keys_present) -> None:
    """Refuse a noise model that no channel combination can score.

    Parameters
    ----------
    noise : NoiseSpec
        The noise model.
    data_keys_present : iterable of str
        The ``data`` keys the galaxy carries.

    Raises
    ------
    NotImplementedError
        Calibration marginalization together with fitted line amplitudes, or Student-t
        noise together with marginalized nuisance coefficients: neither has a closed form.
    ValueError
        A ``data`` key that is neither a channel key nor bookkeeping, or a spectrum
        covariance without its data keys.
    """
    present = set(data_keys_present)
    unknown = present - data_keys() - BOOKKEEPING_KEYS
    if unknown:
        raise ValueError(f"data keys read by no channel: {sorted(unknown)}")
    _refuse_unscorable(noise)
    if "presence" in present and not _is_phot(noise):
        _check_presence({"presence": np.zeros(1)}, noise)
    if noise.covariance == "banded" and noise.chol_offsets is None:
        raise ValueError("a banded covariance needs chol_offsets")
    if noise.covariance == "dense" and "spec_cov_inv" not in present:
        raise ValueError("a dense covariance needs the spec_cov_inv data key")
    if noise.covariance == "banded" and "spec_chol_data" not in present:
        raise ValueError("a banded covariance needs the spec_chol_data data key")
    marginalized = noise.calibration is not None or _lines_marginalized(noise)
    if noise.covariance == "dense" and marginalized and "spec_cov_chol_data" not in present:
        raise ValueError("a dense covariance with marginalization needs spec_cov_chol_data")


def score(preds: Predictions, data: Mapping, noise: NoiseSpec):
    """Log-likelihood of one galaxy: the sum over the channels present in ``data``.

    Parameters
    ----------
    preds : Predictions
        Model observables from any engine.
    data : mapping
        The data bundle; its key set selects the channels.
    noise : NoiseSpec
        The noise model.

    Returns
    -------
    scalar
        ``log p(d | params)`` [nats] without the constants :func:`log_normalization` returns.

    Notes
    -----
    **JIT-compatible**: pure in ``(preds, data)``; ``noise`` is static structure.
    """
    _refuse_unscorable(noise)
    _check_presence(data, noise)
    total = jnp.asarray(0.0)
    for channel in selected_channels(data, noise):
        total = total + channel.loglik(preds, data, noise)
    return total


def log_normalization(data: Mapping, noise: NoiseSpec):
    """The constant :func:`score` omits, so ``score + log_normalization`` is the exact density.

    Parameters
    ----------
    data : mapping
        The data bundle.
    noise : NoiseSpec
        The noise model.

    Returns
    -------
    scalar
        The normalization [nats], summed over the selected channels.
    """
    total = jnp.asarray(0.0)
    for channel in selected_channels(data, noise):
        total = total + channel.log_norm(data, noise)
    return total


#: Engines whose likelihood is :func:`score`. The engine sweep in ``tests/contract`` expects
#: every engine outside this set to fail its cells, and every engine inside it to pass.
ENGINES_ON_SCORER = frozenset({"sequential"})
