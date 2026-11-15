# SPDX-License-Identifier: BSD-3-Clause
"""The one per-galaxy likelihood scorer: model observables in, log-likelihood out (#3505).

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

import warnings
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
    correlated_term,
    eline_cloudy_marginal_term,
    eline_fitted_term,
    eline_flat_marginal_term,
    eline_model_marginal_term,
    student_t_term,
)

__all__ = [
    "BOOKKEEPING_KEYS",
    "CHANNELS",
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
        Width of the flat prior; ``None`` is the uninformative ``1e10``.
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
        ``log_norm(data, noise)``: the constant ``loglik`` omits [nats]. ``None`` means the
        term has no separate constant and a warning says so.
    """

    name: str
    keys: tuple
    loglik: Callable
    gaussian_linear: bool
    optional_keys: tuple = ()
    applies: Callable = lambda data, noise: True
    supersedes: tuple = ()
    log_norm: Callable | None = None
    legacy_name: Callable = lambda noise: ""


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


def _has_mask(data):
    return "data_mask" in data


def _is_phot(noise):
    return noise.data_type in ("photometry", "joint")


def _is_spec(noise):
    return noise.data_type in ("spectroscopy", "joint")


# photometry ---------------------------------------------------------------------------


def _photometry_applies(data, noise):
    return _is_phot(noise)


def _photometry_loglik(preds, data, noise):
    obs, err = _part(data["data"], "phot", noise), _part(data["noise"], "phot", noise)
    if noise.student_t:
        return student_t_term(preds.phot_fnu, obs, err, _f_cal(preds, noise), noise.dof)
    return diag_gaussian_log_prob(preds.phot_fnu, obs, err)


def _photometry_norm(data, noise):
    obs, err = _part(data["data"], "phot", noise), _part(data["noise"], "phot", noise)
    if noise.student_t:
        return -0.5 * jnp.size(obs) * LOG_2PI
    return diag_gaussian_log_normalization(jnp.hypot(err, 0.0 * obs), None)


def _presence_applies(data, noise):
    return noise.data_type == "photometry" and not noise.student_t and not _has_mask(data)


def _presence_loglik(preds, data, noise):
    return diag_gaussian_log_prob(
        preds.phot_fnu, data["data"], data["noise"], presence=data["presence"]
    )


def _presence_norm(data, noise):
    sigma_eff = jnp.hypot(data["noise"], 0.0 * data["data"])
    return diag_gaussian_log_normalization(sigma_eff, data["presence"])


def _censoring_applies(data, noise):
    return _has_mask(data)


def _censored_prediction(preds, noise):
    if noise.data_type == "photometry":
        return preds.phot_fnu
    if noise.data_type == "spectroscopy":
        return preds.spec_fnu
    return jnp.concatenate([preds.phot_fnu, preds.spec_fnu])


def _censoring_loglik(preds, data, noise):
    return censored_term(
        _censored_prediction(preds, noise),
        data["data"],
        data["noise"],
        data["data_mask"],
        _f_cal(preds, noise),
        noise.dof,
    )


def _censoring_norm(data, noise):
    n_detected = jnp.sum(data["data_mask"] == 0)
    if noise.dof is not None:
        return n_detected * student_t_log_norm_per_point(noise.dof)
    return -0.5 * n_detected * LOG_2PI


# spectrum -----------------------------------------------------------------------------


def _marginalizes(noise):
    return noise.calibration is not None or noise.lines is not None


def _spectrum_applies(data, noise):
    if not _is_spec(noise) or _has_mask(data):
        return False
    return noise.student_t or noise.covariance == "dense" or not _marginalizes(noise)


def _chol(data, noise):
    if noise.covariance != "banded":
        return None, None
    return data["spec_chol_data"], noise.chol_offsets


def _spectrum_loglik(preds, data, noise):
    obs, err = _part(data["data"], "spec", noise), _part(data["noise"], "spec", noise)
    if noise.student_t:
        return student_t_term(preds.spec_fnu, obs, err, _f_cal(preds, noise), noise.dof)
    if noise.covariance == "diagonal":
        return diag_gaussian_log_prob(preds.spec_fnu, obs, err)
    chol, offsets = _chol(data, noise)
    return correlated_term(preds.spec_fnu, obs, data.get("spec_cov_inv"), chol, offsets)


def _spectrum_norm(data, noise):
    obs, err = _part(data["data"], "spec", noise), _part(data["noise"], "spec", noise)
    if noise.student_t:
        return -0.5 * jnp.size(obs) * LOG_2PI
    if noise.covariance == "diagonal":
        return diag_gaussian_log_normalization(jnp.hypot(err, 0.0 * obs), None)
    if noise.covariance == "dense":
        log_det = -jnp.linalg.slogdet(data["spec_cov_inv"])[1]
    else:
        chol, offsets = _chol(data, noise)
        diagonal = chol[int(np.where(np.asarray(offsets, dtype=int) == 0)[0][0])]
        log_det = 2.0 * jnp.sum(jnp.log(diagonal))
    return -0.5 * log_det - 0.5 * jnp.size(obs) * LOG_2PI


def _marginal_active(data, noise):
    return (
        _is_spec(noise)
        and not _has_mask(data)
        and not noise.student_t
        and noise.covariance != "dense"
    )


def _calibration_applies(data, noise):
    return _marginal_active(data, noise) and noise.calibration is not None and noise.lines is None


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
    )


def _flat_variance(lines):
    if lines.prior_variance is not None:
        return lines.prior_variance
    return (lines.prior_sigma or 1e10) ** 2


def _lines_applies(data, noise):
    return _marginal_active(data, noise) and noise.lines is not None and noise.calibration is None


def _lines_loglik(preds, data, noise):
    lines = noise.lines
    obs, err = _part(data["data"], "spec", noise), _part(data["noise"], "spec", noise)
    model = preds.spec_fnu
    design = preds.eline_design
    if lines.mode == "fitted":
        amplitudes = jnp.array([preds.params[nm] for nm in lines.amplitude_names])
        return eline_fitted_term(model, obs, err, design, amplitudes)
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
        )
    return eline_flat_marginal_term(
        model,
        obs,
        err,
        design,
        prior_variance=_flat_variance(lines),
        chol_data=chol,
        chol_offsets=offsets,
    )


def _calibration_lines_applies(data, noise):
    return (
        _marginal_active(data, noise)
        and noise.lines is not None
        and noise.lines.mode == "marginalized"
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


def _lines_legacy_name(noise):
    lines = noise.lines
    if lines.mode == "fitted":
        return "ELineFittedLikelihood"
    if lines.prior_type == "cloudy":
        return "CloudyELineMarginalizedLikelihood"
    if lines.prior_type == "model":
        return "ModelELineMarginalizedLikelihood"
    return "ELineMarginalizedLikelihood"


_BASE = ("data", "noise")

#: Every channel, in summation order: the main data, the spectrum, then the constraints.
CHANNELS: tuple = (
    Channel(
        "photometry",
        _BASE,
        _photometry_loglik,
        True,
        applies=_photometry_applies,
        log_norm=_photometry_norm,
    ),
    Channel(
        "presence",
        (*_BASE, "presence"),
        _presence_loglik,
        True,
        applies=_presence_applies,
        supersedes=("photometry",),
        log_norm=_presence_norm,
    ),
    Channel(
        "censoring",
        (*_BASE, "data_mask"),
        _censoring_loglik,
        False,
        applies=_censoring_applies,
        supersedes=("photometry", "presence"),
        log_norm=_censoring_norm,
    ),
    Channel(
        "spectrum",
        _BASE,
        _spectrum_loglik,
        True,
        optional_keys=("spec_cov_inv", "spec_chol_data"),
        applies=_spectrum_applies,
        log_norm=_spectrum_norm,
    ),
    Channel(
        "calibration",
        _BASE,
        _calibration_loglik,
        True,
        optional_keys=("spec_chol_data",),
        applies=_calibration_applies,
        legacy_name=lambda noise: "CalibrationMarginalizedLikelihood",
    ),
    Channel(
        "lines",
        _BASE,
        _lines_loglik,
        True,
        optional_keys=("spec_chol_data",),
        applies=_lines_applies,
        legacy_name=_lines_legacy_name,
    ),
    Channel(
        "calibration_lines",
        _BASE,
        _calibration_lines_loglik,
        True,
        optional_keys=("spec_chol_data",),
        applies=_calibration_lines_applies,
        legacy_name=lambda noise: "CalibrationELineMarginalizedLikelihood",
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
        Calibration marginalization together with fitted line amplitudes: the two cannot
        share one marginal.
    ValueError
        A ``data`` key that is neither a channel key nor bookkeeping, or a spectrum
        covariance key without its noise setting.
    """
    present = set(data_keys_present)
    unknown = present - data_keys() - BOOKKEEPING_KEYS - {"spec_cov_inv", "spec_chol_data"}
    if unknown:
        raise ValueError(f"data keys read by no channel: {sorted(unknown)}")
    if noise.calibration is not None and noise.lines is not None and noise.lines.mode == "fitted":
        raise NotImplementedError(
            "Combined calibration marginalization + fitted (non-marginalized) "
            "emission-line amplitudes is not currently supported. "
            "Use eline_marginalize=True (with optional eline_prior_type) for "
            "the standard Prospector-style configuration, or disable "
            "calibration_marginalize."
        )
    if noise.covariance == "banded" and noise.chol_offsets is None:
        raise ValueError("a banded covariance needs chol_offsets")
    if noise.covariance == "dense" and "spec_cov_inv" not in present:
        raise ValueError("a dense covariance needs the spec_cov_inv data key")
    if noise.covariance == "banded" and "spec_chol_data" not in present:
        raise ValueError("a banded covariance needs the spec_chol_data data key")


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
        The normalization [nats]. A censored joint or spectrum fit scores its constraint
        channels without constants and warns.
    """
    selected = selected_channels(data, noise)
    legacy_censored = _has_mask(data) and noise.data_type != "photometry"
    total = jnp.asarray(0.0)
    skipped = False
    for channel in selected:
        if legacy_censored and channel.name not in ("censoring",):
            skipped = True
            continue
        if channel.log_norm is None:
            warnings.warn(
                f"{channel.legacy_name(noise)} has no exact log normalization: log_evidence "
                "omits its constant (#3427).",
                stacklevel=3,
            )
            continue
        total = total + channel.log_norm(data, noise)
    if skipped:
        warnings.warn(
            "The censored fall-through has line-flux, line-ratio or index channels; their "
            "normalization constants are not in log_evidence (#3427).",
            stacklevel=3,
        )
    return total


#: Engines whose likelihood is :func:`score`. An engine joins this set in the change that
#: moves it onto the scorer; the engine sweep in ``tests/contract`` then requires its cells.
ENGINES_ON_SCORER = frozenset({"sequential"})
