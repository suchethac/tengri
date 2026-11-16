# SPDX-License-Identifier: BSD-3-Clause
"""Engine x channel x noise sweep of the one likelihood scorer.

Every inference engine must score a galaxy through :func:`tengri.inference.channels.score`,
so a data channel or noise model added once works in every engine. The matrix is generated
from the channel registry, not a hand list. Only the sequential engine (``DataTerm`` through
the Fitter) is on the scorer; every other engine's cells are strict ``xfail`` and flip to a
failure the moment that engine is registered in ``ENGINES_ON_SCORER``.

Live cells assert, for each channel, noise and presence variant:

1. ``score`` on the model's own predictions equals ``DataTerm.log_prob`` (rtol 1e-12), where
   a Fitter of that structure exists;
2. ``score + log_normalization`` equals an independent numpy/scipy density written below
   (rtol 1e-9). Those references share no code with the scorer;
3. the gradient of ``score`` matches central finite differences with respect to every
   prediction array and every scalar parameter the noise model reads, and an absent band
   carries exactly zero gradient.
"""

from __future__ import annotations

import gc

import jax
import numpy as np
import pytest
import scipy.stats

from tengri.inference import channels
from tengri.inference.channels import (
    CalibrationSpec,
    LineSpec,
    NoiseSpec,
    Predictions,
    channel_keys,
    log_normalization,
    score,
    selected_channels,
)
from tengri.observation.banded import banded_cholesky, dense_to_banded
from tengri.observation.noise import compute_effective_noise
from tests.contract import _channel_fixtures as fixtures

pytestmark = pytest.mark.contract

ENGINES = (
    "sequential",
    "catalog_mcmc_vmap",
    "native_vi_linear",
    "native_vi_nonlinear",
    "batched_spectro",
    "hierarchical",
    "profiled_map",
)
COVARIANCES = ("dense", "banded")
NOISES = (
    "gaussian",
    "student_t",
    "cal_floor",
    "dense_cov",
    "banded_cov",
    "cal_marg",
    "cal_marg_banded",
    "cal_marg_dense",
    "cal_marg_segments",
    "lines_marginalized",
    "lines_marginalized_banded",
    "lines_marginalized_dense",
    "lines_marginalized_cloudy",
    "lines_marginalized_model",
    "lines_fitted",
    "lines_fitted_banded",
    "lines_fitted_dense",
    "lines_fitted_student_t",
    "cal_marg_lines",
    "cal_marg_lines_banded",
    "cal_marg_lines_dense",
    "cal_marg_lines_segments",
    "cal_marg_lines_cloudy",
    "cal_marg_lines_model",
    "cal_marg_lines_segments_model",
    "student_t_dense",
    "student_t_banded",
    "cal_floor_dense",
    "cal_floor_banded",
    "lines_fitted_student_t_dense",
    "floor_cal_marg",
    "floor_cal_marg_banded",
    "floor_cal_marg_dense",
    "floor_cal_marg_segments",
    "floor_lines_marginalized",
    "floor_lines_marginalized_dense",
    "floor_lines_marginalized_cloudy",
    "floor_lines_marginalized_model",
    "floor_cal_marg_lines",
    "floor_cal_marg_lines_banded",
    "floor_cal_marg_lines_segments_model",
    "floor_lines_fitted",
    "cal_marg_fitted_lines",
)
REFUSED = ("cal_marg_fitted_lines",)
DOF = 4.0
F_CAL = 0.07
N_BAND, N_PIX, N_LINE, N_CONSTRAINT = 5, 24, 3, 4
CAL_SIGMA, LINE_SIGMA = 0.3, 2.0
PRESENCE = np.array([1.0, 0.0, 1.0, 1.0, 0.0])

SPECTRUM_BLOCK = ("spectrum", "calibration", "lines", "calibration_lines")
SPECTRUM_ONLY_NOISE = NOISES[3:]
CONSTRAINTS = ("line_fluxes", "line_censor", "line_ratios", "spectral_indices")
PRESENCE_CHANNELS = tuple(c.name for c in channels.CHANNELS if "presence" in c.optional_keys)
MAIN_NOISES = ("gaussian", "student_t", "cal_floor")
FITTED_NOISES = tuple(n for n in NOISES if n.startswith("lines_fitted"))
CAL_NOISES = (
    "cal_marg",
    "cal_marg_banded",
    "cal_marg_dense",
    "cal_marg_segments",
    "floor_cal_marg",
    "floor_cal_marg_banded",
    "floor_cal_marg_dense",
    "floor_cal_marg_segments",
)
LINE_NOISES = (
    "lines_marginalized",
    "lines_marginalized_banded",
    "lines_marginalized_dense",
    "lines_marginalized_cloudy",
    "lines_marginalized_model",
    "floor_lines_marginalized",
    "floor_lines_marginalized_dense",
    "floor_lines_marginalized_cloudy",
    "floor_lines_marginalized_model",
)
JOINT_NOISES = (
    "cal_marg_lines",
    "cal_marg_lines_banded",
    "cal_marg_lines_dense",
    "cal_marg_lines_segments",
    "cal_marg_lines_cloudy",
    "cal_marg_lines_model",
    "cal_marg_lines_segments_model",
    "floor_cal_marg_lines",
    "floor_cal_marg_lines_banded",
    "floor_cal_marg_lines_segments_model",
)
SPECTRUM_NOISES = (
    "gaussian",
    "student_t",
    "cal_floor",
    "dense_cov",
    "banded_cov",
    "student_t_dense",
    "student_t_banded",
    "cal_floor_dense",
    "cal_floor_banded",
    "floor_lines_fitted",
    *FITTED_NOISES,
)
REFUSED_REASON = {
    "cal_marg_fitted_lines": (
        "calibration marginalization with fitted line amplitudes cannot share one marginal; "
        "validate() refuses it (test_unscorable_noise_is_refused_by_name)"
    ),
}


def exclusion_reason(channel: str, noise: str, variant: str = "") -> str | None:
    """Why a cell is contradictory, or ``None`` when it is live."""
    if noise in REFUSED:
        return REFUSED_REASON[noise]
    if noise in SPECTRUM_ONLY_NOISE and channel not in SPECTRUM_BLOCK:
        return f"{noise} acts on the spectrum pixels; {channel} does not read them"
    if channel in CONSTRAINTS and noise != "gaussian":
        return (
            f"{channel} is a Gaussian constraint on a few measured numbers; the {noise} noise "
            "model applies to the main data block only"
        )
    if variant == "presence" and noise not in MAIN_NOISES:
        return "presence masks photometry bands; the noise model is a spectrum one"
    table = {
        "spectrum": SPECTRUM_NOISES,
        "calibration": CAL_NOISES,
        "lines": LINE_NOISES,
        "calibration_lines": JOINT_NOISES,
    }
    if channel in table and noise not in table[channel]:
        return (
            f"a {noise} spectrum is scored by the channel its noise model selects, not {channel}"
        )
    return None


# ── Independent references (numpy and scipy only) ────────────────────────────────────


def _blocks(data, noise, block):
    obs, err = np.asarray(data["data"]), np.asarray(data["noise"])
    if noise.data_type != "joint":
        return obs, err
    cut = noise.n_phot
    return (obs[:cut], err[:cut]) if block == "phot" else (obs[cut:], err[cut:])


def _f_cal_of(preds, noise):
    name = noise.f_cal_param
    return float(preds.params[name]) if name and name in preds.params else 0.0


def _ref_main(pred, obs, err, noise, preds, keep=None):
    """Gaussian, floored Gaussian or Student-t density of the present data."""
    if keep is not None:
        pred, obs, err = pred[keep], obs[keep], err[keep]
    sigma = np.hypot(err, _f_cal_of(preds, noise) * np.abs(pred)) if noise.student_t else err
    r = (obs - pred) / sigma
    if noise.student_t and noise.dof is not None:
        return float(np.sum(scipy.stats.t.logpdf(r, noise.dof) - np.log(sigma)))
    return float(np.sum(scipy.stats.norm.logpdf(r) - np.log(sigma)))


def _ref_censored(pred, obs, err, mask, noise, preds, sigma_frac=None, f_cal=None, keep=None):
    if keep is not None:
        pred, obs, err, mask = pred[keep], obs[keep], err[keep], mask[keep]
        sigma_frac = None if sigma_frac is None else np.asarray(sigma_frac)[keep]
    f_cal = _f_cal_of(preds, noise) if f_cal is None else f_cal
    model_sigma = 0.0 if sigma_frac is None else np.asarray(sigma_frac)
    sigma_eff = np.sqrt(err**2 + (f_cal * pred) ** 2 + (model_sigma * pred) ** 2)
    sigma_lim = np.sqrt(err**2 + (model_sigma * pred) ** 2)
    r = (obs - pred) / sigma_eff
    if noise.dof is not None:
        detected = scipy.stats.t.logpdf(r, noise.dof) - np.log(sigma_eff)
    else:
        detected = scipy.stats.norm.logpdf(r) - np.log(sigma_eff)
    upper = scipy.stats.norm.logcdf((obs - pred) / sigma_lim)
    lower = scipy.stats.norm.logcdf((pred - obs) / sigma_lim)
    return float(np.sum(np.where(mask == 1, upper, np.where(mask == -1, lower, detected))))


def _chebyshev(wave, n_poly):
    x = (2 * wave - wave.min() - wave.max()) / (wave.max() - wave.min())
    cols = [np.polynomial.chebyshev.Chebyshev.basis(k)(x) for k in range(1, n_poly + 1)]
    return np.stack(cols, 1)


def _calibration_basis(cal):
    basis = _chebyshev(np.asarray(cal.wavelength), cal.n_poly)
    if cal.n_segments > 0:
        ids = np.asarray(cal.segments)
        indicators = (ids[:, None] == np.arange(cal.n_segments)[None, :]).astype(float)
        basis = np.concatenate([indicators, basis], axis=1)
    return basis


def _pixel_covariance(data, noise, err):
    if noise.covariance == "dense":
        return np.linalg.inv(np.asarray(data["spec_cov_inv"]))
    if noise.covariance == "banded":
        offsets = np.asarray(noise.chol_offsets, dtype=int)
        band = np.asarray(data["spec_chol_data"])
        n = band.shape[1]
        lower = np.zeros((n, n))
        for k, off in enumerate(offsets):
            for i in range(n):
                if 0 <= i + off < n:
                    lower[i, i + off] = band[k, i]
        return lower @ lower.T
    return np.diag(err**2)


def _mvn(resid, cov):
    """Zero-mean Gaussian log-density, in units of the typical error to span any flux scale."""
    scale = float(np.sqrt(np.mean(np.diag(cov))))
    n = len(resid)
    dense = scipy.stats.multivariate_normal.logpdf(
        resid / scale, mean=np.zeros(n), cov=cov / scale**2
    )
    return float(dense) - n * np.log(scale)


def _fitted_shift(preds, noise):
    if noise.lines is None or noise.lines.mode != "fitted":
        return 0.0
    amps = np.array([float(preds.params[n]) for n in noise.lines.amplitude_names])
    return np.asarray(preds.eline_design) @ amps


def _ref_spectrum(preds, data, noise):
    obs, err = _blocks(data, noise, "spec")
    model = np.asarray(preds.spec_fnu) + _fitted_shift(preds, noise)
    if noise.student_t and noise.covariance != "diagonal":
        # Convention: the floor adds (f * model)^2 to the diagonal of the covariance.
        cov = _pixel_covariance(data, noise, err) + np.diag((_f_cal_of(preds, noise) * model) ** 2)
        if noise.dof is None:
            return _mvn(obs - model, cov)
        return float(scipy.stats.multivariate_t.logpdf(obs, loc=model, shape=cov, df=noise.dof))
    if noise.student_t or noise.covariance == "diagonal":
        return _ref_main(model, obs, err, noise, preds)
    return _mvn(obs - model, _pixel_covariance(data, noise, err))


def _noise_cov(preds, data, noise, err, m0):
    """N: the pixel covariance plus the floor variance (f * m0)^2 on its diagonal.

    ``m0`` is the model before the nuisance terms, the spectrum the calibration polynomial
    multiplies and the line amplitudes are added to.
    """
    cov = _pixel_covariance(data, noise, err)
    if noise.student_t and noise.dof is None:
        cov = cov + np.diag((_f_cal_of(preds, noise) * m0) ** 2)
    return cov


def _ref_calibration(preds, data, noise):
    obs, err = _blocks(data, noise, "spec")
    pred, cal = np.asarray(preds.spec_fnu), noise.calibration
    basis = pred[:, None] * _calibration_basis(cal)
    cov = _noise_cov(preds, data, noise, err, pred) + cal.prior_sigma**2 * basis @ basis.T
    return _mvn(obs - pred, cov)


def _line_moments(preds, noise):
    """Mean and width of each line amplitude's Gaussian prior, from the prior's definition."""
    lines = noise.lines
    if lines.prior_type == "cloudy":
        from tengri.observation.eline_priors import cloudy_line_priors

        mean, sigma = cloudy_line_priors(
            log_z=preds.params["met_logzsol"],
            neb_logU=preds.params["neb_logU"],
            line_wavelengths=np.asarray(lines.line_wavelengths),
            prior_width_dex=lines.prior_width_dex,
        )
        return np.asarray(mean), np.asarray(sigma)
    if lines.prior_type == "model":
        from tengri.observation.eline_marginalization import model_line_prior

        redshift, dl_cm, aperture, transmission = preds.eline_prior_inputs
        prior = model_line_prior(
            preds.log_line_lums_attenuated,
            lines.match_index,
            lines.matched,
            lines.line_wavelengths,
            redshift,
            dl_cm,
            aperture,
            lines.prior_width_dex,
            transmission=transmission,
        )
        assert np.all(np.asarray(prior.gaussian) == 1.0)
        return np.asarray(prior.prior_mean), np.asarray(prior.prior_width)
    sigma = lines.prior_variance**0.5 if lines.prior_variance is not None else lines.prior_sigma
    n_lines = np.shape(preds.eline_design)[1]
    return np.zeros(n_lines), np.full(n_lines, sigma)


def _ref_lines(preds, data, noise):
    obs, err = _blocks(data, noise, "spec")
    pred, design = np.asarray(preds.spec_fnu), np.asarray(preds.eline_design)
    mean, width = _line_moments(preds, noise)
    cov = _noise_cov(preds, data, noise, err, pred) + (design * width) @ (design * width).T
    return _mvn(obs - pred - design @ mean, cov)


def _ref_calibration_lines(preds, data, noise):
    obs, err = _blocks(data, noise, "spec")
    pred, design = np.asarray(preds.spec_fnu), np.asarray(preds.eline_design)
    cal = noise.calibration
    basis = pred[:, None] * _calibration_basis(cal)
    mean, width = _line_moments(preds, noise)
    cov = (
        _noise_cov(preds, data, noise, err, pred)
        + cal.prior_sigma**2 * basis @ basis.T
        + (design * width) @ (design * width).T
    )
    return _mvn(obs - pred - design @ mean, cov)


def reference_density(channel, preds, data, noise):
    """The independent log-density a channel must reproduce, constants included."""
    if channel == "photometry":
        obs, err = _blocks(data, noise, "phot")
        keep = None if "presence" not in data else np.asarray(data["presence"]) > 0
        return _ref_main(np.asarray(preds.phot_fnu), obs, err, noise, preds, keep)
    if channel == "censoring":
        keep = None if "presence" not in data else np.asarray(data["presence"]) > 0
        return _ref_censored(
            np.asarray(preds.phot_fnu),
            np.asarray(data["data"]),
            np.asarray(data["noise"]),
            np.asarray(data["data_mask"]),
            noise,
            preds,
            keep=keep,
        )
    if channel == "spectrum":
        return _ref_spectrum(preds, data, noise)
    if channel == "calibration":
        return _ref_calibration(preds, data, noise)
    if channel == "lines":
        return _ref_lines(preds, data, noise)
    if channel == "calibration_lines":
        return _ref_calibration_lines(preds, data, noise)
    return _reference_constraint(channel, preds, data)


def _reference_constraint(channel, preds, data):
    plain = NoiseSpec()
    if channel == "line_fluxes":
        return _ref_main(
            np.asarray(preds.line_fluxes),
            np.asarray(data["line_flux_obs"]),
            np.asarray(data["line_flux_err"]),
            plain,
            preds,
        )
    if channel == "line_censor":
        return _ref_censored(
            np.asarray(preds.line_fluxes),
            np.asarray(data["line_flux_obs"]),
            np.asarray(data["line_flux_err"]),
            np.asarray(data["line_flux_limit_mask"]),
            plain,
            preds,
            sigma_frac=data["line_flux_model_sigma_frac"],
            f_cal=0.0,
        )
    names = {
        "line_ratios": ("line_ratio_obs", "line_ratio_err", "line_ratios"),
        "spectral_indices": ("index_obs", "index_err", "indices"),
    }[channel]
    return _ref_main(
        np.asarray(getattr(preds, names[2])),
        np.asarray(data[names[0]]),
        np.asarray(data[names[1]]),
        plain,
        preds,
    )


# ── Synthetic cells ───────────────────────────────────────────────────────────────────


def _dense_bands(lower):
    n = lower.shape[0]
    bands = np.zeros((n, n))
    for lag in range(n):
        bands[lag, lag:] = np.diagonal(lower, -lag)
    return bands


def _covariance_inputs(label, err):
    """Data entries and noise fields of the pixel covariance a noise label names."""
    kind = "dense" if "dense" in label else "banded" if "banded" in label else None
    if kind is None:
        return {}, {}
    n = err.size
    cov = np.diag(err**2)
    lags = tuple((k, 0.5**k) for k in range(1, n)) if kind == "dense" else ((1, 0.3), (2, 0.1))
    for lag, rho in lags:
        idx = np.arange(n - lag)
        cov[idx, idx + lag] = rho * err[idx] * err[idx + lag]
        cov[idx + lag, idx] = cov[idx, idx + lag]
    if kind == "dense":
        data = {
            "spec_cov_inv": np.linalg.inv(cov),
            "spec_cov_chol_data": _dense_bands(np.linalg.cholesky(cov)),
        }
        return data, {"covariance": "dense"}
    factor = banded_cholesky(dense_to_banded(cov))
    offsets = tuple(int(o) for o in np.asarray(factor.offsets))
    return {"spec_chol_data": np.asarray(factor.data)}, {
        "covariance": "banded",
        "chol_offsets": offsets,
    }


def _student_fields(label):
    if label in REFUSED:
        return {}
    if label.startswith("student_t") or "student_t" in label:
        return {"student_t": True, "dof": DOF, "f_cal_param": "noise_frac_cal"}
    if label.startswith(("cal_floor", "floor_")):
        return {"student_t": True, "dof": None, "f_cal_param": "noise_frac_cal"}
    return {}


def _synthetic(channel: str, label: str, variant: str = "", seed: int = 0):
    """Predictions, data and noise spec of one live cell, drawn at random."""
    rng = np.random.default_rng(seed)

    def draw(n):
        pred = rng.uniform(1.0, 2.0, n)
        return pred, pred * (1.0 + 0.1 * rng.standard_normal(n)), rng.uniform(0.05, 0.15, n)

    params = {
        "noise_frac_cal": F_CAL,
        "a0": 0.3,
        "a1": -0.2,
        "a2": 0.1,
        "cal_log_q": 0.05,
        "met_logzsol": -0.3,
        "neb_logU": -2.5,
    }
    kw = _student_fields(label)
    if channel in ("photometry", "censoring"):
        pred, obs, err = draw(N_BAND)
        if variant == "presence":
            obs, err = np.where(PRESENCE > 0, obs, 0.0), np.where(PRESENCE > 0, err, 0.0)
        data = {"data": obs, "noise": err}
        if variant == "presence":
            data["presence"] = PRESENCE
        if channel == "censoring":
            data["data_mask"] = np.array([0, 1, -1, 0, 1])
            kw.update(dof=DOF if label == "student_t" else None, student_t=False)
            kw["f_cal_param"] = "noise_frac_cal" if label != "gaussian" else None
        preds_kw = {"phot_fnu": pred}
        kw["data_type"] = "photometry"
    elif channel in SPECTRUM_BLOCK:
        data, extra, preds_kw = _synthetic_spectrum(label, rng, draw)
        kw.update(extra, data_type="spectroscopy")
    else:
        data, preds_kw = _synthetic_constraint(channel, rng)
        kw["data_type"] = "photometry"
    return Predictions(params=params, **preds_kw), data, NoiseSpec(**kw)


def _synthetic_constraint(channel, rng):
    pred = rng.uniform(1.0, 2.0, N_CONSTRAINT)
    obs = pred * (1 + 0.1 * rng.standard_normal(N_CONSTRAINT))
    err = rng.uniform(0.05, 0.15, N_CONSTRAINT)
    if channel in ("line_fluxes", "line_censor"):
        data = {
            "line_flux_obs": obs,
            "line_flux_err": err,
            "line_flux_waves": np.arange(N_CONSTRAINT),
        }
        if channel == "line_censor":
            data["line_flux_limit_mask"] = np.array([0, 1, -1, 0])
            data["line_flux_model_sigma_frac"] = np.array([0.05, 0.1, 0.0, 0.2])
        return data, {"line_fluxes": pred}
    if channel == "line_ratios":
        return {"line_ratio_obs": obs, "line_ratio_err": err}, {"line_ratios": pred}
    return {"index_obs": obs, "index_err": err}, {"indices": pred}


LINE_WAVELENGTHS = np.array([4861.0, 5007.0, 6563.0])
MODEL_PRIOR_INPUTS = (0.1, 1.0e27, 1.0, np.ones(N_LINE))
MODEL_LOG_LUMS = np.array([8.0, 8.3, 8.6])


def _line_spec(label, calibrated):
    """The line noise spec a label names, and the prediction fields its prior needs."""
    if label.endswith("_cloudy"):
        spec = LineSpec(
            mode="marginalized",
            prior_type="cloudy",
            line_wavelengths=LINE_WAVELENGTHS,
            prior_width_dex=0.3,
        )
        return spec, {}
    if label.endswith("_model"):
        spec = LineSpec(
            mode="marginalized",
            prior_type="model",
            line_wavelengths=LINE_WAVELENGTHS,
            prior_width_dex=0.3,
            match_index=np.arange(N_LINE),
            matched=np.ones(N_LINE),
        )
        extra = {
            "log_line_lums_attenuated": MODEL_LOG_LUMS,
            "eline_prior_inputs": MODEL_PRIOR_INPUTS,
        }
        return spec, extra
    if calibrated:
        return LineSpec(mode="marginalized", prior_sigma=LINE_SIGMA), {}
    return LineSpec(mode="marginalized", prior_variance=LINE_SIGMA**2), {}


def _synthetic_spectrum(label, rng, draw):
    label = label.removeprefix("floor_")
    spec_p, spec_d, spec_e = draw(N_PIX)
    wave = np.linspace(4000.0, 8000.0, N_PIX)
    data = {"data": spec_d, "noise": spec_e}
    cov_data, kw = _covariance_inputs(label, spec_e)
    data.update(cov_data)
    segmented = "segments" in label
    cal = CalibrationSpec(
        wavelength=wave,
        n_poly=3,
        prior_sigma=CAL_SIGMA,
        segments=(np.arange(N_PIX) % 2) if segmented else None,
        n_segments=2 if segmented else 0,
    )
    preds_kw = {"spec_fnu": spec_p}
    if label.startswith("cal_marg") and "lines" not in label:
        kw["calibration"] = cal
    if label.startswith("lines_fitted"):
        kw["lines"] = LineSpec(mode="fitted", amplitude_names=("a0", "a1", "a2"))
    if label.startswith("lines_marginalized") or label.startswith("cal_marg_lines"):
        kw["lines"], extra = _line_spec(label, label.startswith("cal_marg"))
        preds_kw.update(extra)
        if label.startswith("cal_marg"):
            kw["calibration"] = cal
    if "lines" in label:
        scale = np.abs(spec_p.mean()) * 0.2
        design = rng.standard_normal((N_PIX, N_LINE)) * scale
        if label.endswith(("_cloudy", "_model")):
            design = design / _prior_mean_scale(label)
        preds_kw["eline_design"] = design
    return data, kw, preds_kw


def _prior_mean_scale(label):
    """Typical prior-mean amplitude, so design columns put the prior on the data's scale."""
    if label.endswith("_cloudy"):
        from tengri.observation.eline_priors import cloudy_line_priors

        mean, _ = cloudy_line_priors(
            log_z=-0.3, neb_logU=-2.5, line_wavelengths=LINE_WAVELENGTHS, prior_width_dex=0.3
        )
        return float(np.mean(np.asarray(mean)))
    from tengri.observation.eline_marginalization import model_line_prior

    prior = model_line_prior(
        MODEL_LOG_LUMS,
        np.arange(N_LINE),
        np.ones(N_LINE),
        LINE_WAVELENGTHS,
        *MODEL_PRIOR_INPUTS[:3],
        0.3,
        transmission=MODEL_PRIOR_INPUTS[3],
    )
    return float(np.mean(np.asarray(prior.prior_mean)))


# ── Cell checks ───────────────────────────────────────────────────────────────────────


def _full_density(preds, data, noise, channel):
    """``score + log_normalization``, after checking the cell engages only its channel."""
    chosen = [c.name for c in selected_channels(data, noise)]
    assert chosen == [channel], (channel, chosen)
    return float(score(preds, data, noise) + log_normalization(data, noise))


def cell_discrepancy(channel, label, variant="", cell=None) -> float:
    """Relative difference between the scorer and the independent reference of one cell."""
    preds, data, noise = _synthetic(channel, label, variant) if cell is None else cell
    expected = reference_density(channel, preds, data, noise)
    actual = _full_density(preds, data, noise, channel)
    return abs(actual - expected) / max(abs(expected), 1e-300)


ARRAY_FIELDS = ("phot_fnu", "spec_fnu", "line_fluxes", "line_ratios", "indices", "eline_design")


def _scalar_params(preds, noise):
    names = []
    if noise.f_cal_param and noise.f_cal_param in preds.params:
        names.append(noise.f_cal_param)
    if noise.calibration is not None and noise.calibration.has_q:
        names.append("cal_log_q")
    if noise.lines is not None and noise.lines.mode == "fitted":
        names.extend(noise.lines.amplitude_names)
    return names


def _total_of(preds, data, noise, fields, scalars):
    def total(arrays, values):
        replaced = {**preds.__dict__, **dict(zip(fields, arrays, strict=True))}
        params = {**preds.params, **dict(zip(scalars, values, strict=True))}
        return score(Predictions(**{**replaced, "params": params}), data, noise)

    return total


def check_gradients(preds, data, noise, *, eps=1e-6, skip=()):
    """Autodiff gradients match central differences; returns the gradient of the prediction."""
    fields = [n for n in ARRAY_FIELDS if getattr(preds, n) is not None]
    scalars = [n for n in _scalar_params(preds, noise) if n not in skip]
    total = _total_of(preds, data, noise, fields, scalars)
    arrays = [np.asarray(getattr(preds, n), dtype=float) for n in fields]
    values = [float(preds.params[n]) for n in scalars]
    grads_a, grads_v = jax.grad(total, argnums=(0, 1))(arrays, values)
    rng = np.random.default_rng(1)
    for i, array in enumerate(arrays):
        direction = rng.standard_normal(array.shape) * (np.max(np.abs(array)) or 1.0)
        up = [a + eps * direction if j == i else a for j, a in enumerate(arrays)]
        dn = [a - eps * direction if j == i else a for j, a in enumerate(arrays)]
        fd = (float(total(up, values)) - float(total(dn, values))) / (2 * eps)
        auto = float(np.sum(np.asarray(grads_a[i]) * direction))
        np.testing.assert_allclose(auto, fd, rtol=1e-5, atol=1e-9 * abs(fd))
    for i, name in enumerate(scalars):
        up = [v + eps if j == i else v for j, v in enumerate(values)]
        dn = [v - eps if j == i else v for j, v in enumerate(values)]
        fd = (float(total(arrays, up)) - float(total(arrays, dn))) / (2 * eps)
        np.testing.assert_allclose(
            float(grads_v[i]), fd, rtol=1e-5, atol=1e-9 * abs(fd), err_msg=name
        )
    return dict(zip(fields, grads_a, strict=True))


# Fitter structure that realizes a cell on a real model, for assertion (i).
FITTER_KINDS = {
    ("photometry", "gaussian", ""): "phot",
    ("photometry", "student_t", ""): "phot_student_t",
    ("photometry", "cal_floor", ""): "phot_floor",
    ("photometry", "gaussian", "presence"): "presence",
    ("photometry", "student_t", "presence"): "presence_student_t",
    ("photometry", "cal_floor", "presence"): "presence_floor",
    ("censoring", "gaussian", ""): "mask",
    ("censoring", "student_t", ""): "mask_student_t",
    ("censoring", "cal_floor", ""): "mask_floor",
    ("censoring", "gaussian", "presence"): "mask_presence",
    ("spectrum", "gaussian", ""): "spec",
    ("spectrum", "student_t", ""): "spec_student_t",
    ("spectrum", "dense_cov", ""): "spec_dense",
    ("spectrum", "banded_cov", ""): "spec_banded",
    ("spectrum", "lines_fitted", ""): "spec_lines_fitted",
    ("spectrum", "lines_fitted_dense", ""): "spec_dense_lines_fitted",
    ("spectrum", "lines_fitted_student_t", ""): "spec_student_t_lines_fitted",
    ("spectrum", "student_t_dense", ""): "spec_student_t_dense",
    ("spectrum", "cal_floor_banded", ""): "spec_floor_banded",
    ("spectrum", "lines_fitted_student_t_dense", ""): "spec_student_t_dense_lines_fitted",
    ("calibration", "cal_marg", ""): "spec_cal",
    ("calibration", "floor_cal_marg", ""): "spec_floor_cal",
    ("lines", "floor_lines_marginalized", ""): "spec_floor_lines",
    ("calibration_lines", "floor_cal_marg_lines", ""): "spec_floor_cal_lines",
    ("calibration", "cal_marg_dense", ""): "spec_dense_cal",
    ("calibration", "cal_marg_banded", ""): "spec_banded_cal",
    ("lines", "lines_marginalized", ""): "spec_lines",
    ("lines", "lines_marginalized_dense", ""): "spec_dense_lines",
    ("calibration_lines", "cal_marg_lines", ""): "spec_cal_lines",
}
_FITTER_CACHE: dict = {}


def _term_of(kind):
    """The Fitter, data term and reference point of one structure.

    Only the most recent structure is kept. Each Fitter owns compiled programs of a few
    hundred megabytes, and a worker that held every structure at once (the repository's
    default is one worker per file) grew until the machine's memory guard killed it.
    """
    if kind not in _FITTER_CACHE:
        _FITTER_CACHE.clear()
        gc.collect()
        jax.clear_caches()
        fitter = fixtures.fitter_of(kind)
        term, physical = fixtures.build_term(fitter)
        _FITTER_CACHE[kind] = (fitter, term, physical)
    return _FITTER_CACHE[kind]


def _moderate_line_prior(noise, preds, data):
    """The noise spec with a line prior that a dense reference can resolve.

    A fit's uninformative line prior (variance 1e20) makes the pixel covariance singular to
    machine precision; the same marginal at a prior ten errors wide is checked instead.
    """
    lines = noise.lines
    if lines is None or lines.mode != "marginalized":
        return noise
    width = (
        10.0
        * float(np.median(np.asarray(data["noise"])))
        / float(np.max(np.abs(np.asarray(preds.eline_design))))
    )
    if noise.calibration is None:
        moderate = LineSpec(mode="marginalized", prior_variance=width**2)
    else:
        moderate = LineSpec(mode="marginalized", prior_sigma=width)
    return NoiseSpec(**{**noise.__dict__, "lines": moderate})


def _sequential_cell(channel, label, variant):
    cell = _synthetic(channel, label, variant)
    preds, data, noise = cell
    assert cell_discrepancy(channel, label, variant, cell) < 1e-9
    grads = check_gradients(preds, data, noise)
    if variant == "presence":
        absent = np.asarray(data["presence"]) == 0
        assert np.all(np.asarray(grads["phot_fnu"])[absent] == 0.0)
    kind = FITTER_KINDS.get((channel, label, variant))
    if kind is None:
        return
    fitter, term, physical = _term_of(kind)
    data_args = fitter._data_args
    amplitudes = getattr(fitter, "_eline_amplitude_names", ())
    at_point = {**physical, **{name: 0.0 for name in amplitudes}}
    real = term.predictions(at_point, data_args)
    scored = float(score(real, data_args, term.noise))
    np.testing.assert_allclose(scored, float(term.log_prob(at_point, data_args)), rtol=1e-12)
    resolvable = _moderate_line_prior(term.noise, real, data_args)
    assert cell_discrepancy(channel, label, variant, (real, data_args, resolvable)) < 1e-9
    check_gradients(real, data_args, resolvable, skip=amplitudes)


def _cells():
    for channel in channel_keys():
        variants = ("", "presence") if channel in PRESENCE_CHANNELS else ("",)
        for noise in NOISES:
            for variant in variants:
                yield channel, noise, variant


LIVE = [(c, n, v) for c, n, v in _cells() if exclusion_reason(c, n, v) is None]
EXCLUDED = [(c, n, v) for c, n, v in _cells() if exclusion_reason(c, n, v) is not None]
ENGINE_REASON = "engine does not score through channels.score"
BATCHED_FLOOR_REASON = "the batched path does not take a calibration floor"


def _params():
    for engine in ENGINES:
        for channel, noise, variant in LIVE:
            marks = []
            if engine != "sequential":
                reason = (
                    BATCHED_FLOOR_REASON
                    if engine == "batched_spectro" and noise.startswith("floor_")
                    else ENGINE_REASON
                )
                marks = [pytest.mark.xfail(strict=True, raises=AssertionError, reason=reason)]
            suffix = f"-{variant}" if variant else ""
            yield pytest.param(
                engine,
                channel,
                noise,
                variant,
                marks=marks,
                id=f"{engine}-{channel}-{noise}{suffix}",
            )


@pytest.mark.parametrize(("engine", "channel", "noise", "variant"), list(_params()))
def test_engine_scores_channel_under_noise(engine, channel, noise, variant):
    if engine == "sequential":
        _sequential_cell(channel, noise, variant)
        return
    assert engine in channels.ENGINES_ON_SCORER


def test_the_matrix_is_complete_and_every_exclusion_is_explained():
    assert len(LIVE) + len(EXCLUDED) == len(list(_cells()))
    assert all(len(exclusion_reason(c, n, v)) > 20 for c, n, v in EXCLUDED)
    assert {c for c, _, _ in LIVE} == set(channel_keys())
    assert {v for _, _, v in LIVE} == {"", "presence"}


def test_only_the_sequential_engine_is_on_the_scorer():
    assert set(channels.ENGINES_ON_SCORER) == {"sequential"}
    assert set(ENGINES) >= set(channels.ENGINES_ON_SCORER)


# ── Registry ──────────────────────────────────────────────────────────────────────────


def test_registry_is_derived_and_complete():
    registry = channel_keys()
    assert list(registry) == [c.name for c in channels.CHANNELS]
    assert len(set(registry)) == len(registry)
    assert {"photometry", "censoring", "line_fluxes", "line_censor"} <= set(registry)
    assert PRESENCE_CHANNELS == ("photometry", "censoring")


def _emitted_data_keys() -> set:
    """Every ``data_args`` key ``build_data_args`` can write, read off its source."""
    import inspect
    import re

    from tengri.inference import data_term

    source = inspect.getsource(data_term.build_data_args) + inspect.getsource(
        data_term._add_observation_channels
    )
    literal = {"data", "noise", "sqrt_noise_inv", "n_data"}
    return literal | set(re.findall(r'args\["(\w+)"\]', source))


def test_channel_keys_and_data_term_keys_are_the_same_set():
    """Every channel key is one the data term accepts; every key it accepts has a channel."""
    emitted = _emitted_data_keys() - channels.BOOKKEEPING_KEYS
    assert channels.data_keys() == emitted, sorted(channels.data_keys() ^ emitted)


# ── Absent bands, with the stored convention: zero flux and zero noise ────────────────


@pytest.mark.parametrize("label", MAIN_NOISES)
@pytest.mark.parametrize("channel", PRESENCE_CHANNELS)
def test_an_absent_band_is_masked_once_for_every_noise_family(channel, label):
    preds, data, noise = _synthetic(channel, label, "presence")
    assert np.all(np.asarray(data["noise"])[PRESENCE == 0] == 0.0)
    value = float(score(preds, data, noise))
    assert np.isfinite(value)
    moved = np.asarray(preds.phot_fnu).copy()
    moved[PRESENCE == 0] += 3.0
    shifted = Predictions(**{**preds.__dict__, "phot_fnu": moved})
    assert float(score(shifted, data, noise)) == value
    dropped = {k: np.asarray(v)[PRESENCE > 0] for k, v in data.items()}
    dropped_preds = Predictions(
        **{**preds.__dict__, "phot_fnu": np.asarray(preds.phot_fnu)[PRESENCE > 0]}
    )
    absent_free = {k: v for k, v in dropped.items() if k != "presence"}
    np.testing.assert_allclose(
        value + float(log_normalization(data, noise)),
        float(score(dropped_preds, absent_free, noise) + log_normalization(absent_free, noise)),
        rtol=1e-12,
    )


# ── Normalization: every marginal carries its constant once ───────────────────────────


def _gaussian_density(preds, data, noise):
    obs, err = np.asarray(data["data"]), np.asarray(data["noise"])
    return float(np.sum(scipy.stats.norm.logpdf(obs, loc=np.asarray(preds.spec_fnu), scale=err)))


def test_flat_lines_with_an_unused_column_equal_the_plain_gaussian_density():
    preds, data, noise = _synthetic("lines", "lines_marginalized")
    unused = Predictions(**{**preds.__dict__, "eline_design": np.zeros((N_PIX, N_LINE))})
    value = float(score(unused, data, noise) + log_normalization(data, noise))
    np.testing.assert_allclose(value, _gaussian_density(unused, data, noise), rtol=1e-10)


@pytest.mark.parametrize("label", ("cal_marg_lines", "cal_marg_lines_segments"))
def test_calibration_with_unused_lines_equals_calibration_alone(label):
    preds, data, noise = _synthetic("calibration_lines", label)
    unused = Predictions(**{**preds.__dict__, "eline_design": np.zeros((N_PIX, N_LINE))})
    alone = NoiseSpec(data_type="spectroscopy", calibration=noise.calibration)
    both = float(score(unused, data, noise))
    np.testing.assert_allclose(both, float(score(unused, data, alone)), rtol=1e-10)


def test_marginal_values_differ_from_the_primitive_by_a_parameter_free_constant():
    """Within a model the offset is data-only, so the posterior does not depend on it."""
    from tengri.observation.eline_marginalization import marginalize_emission_lines

    preds, data, noise = _synthetic("lines", "lines_marginalized")
    err = np.asarray(data["noise"])
    constant = 0.5 * N_PIX * np.log(2 * np.pi) + np.sum(np.log(err))
    offsets = []
    for shift in (0.0, 0.4):
        moved = Predictions(**{**preds.__dict__, "spec_fnu": np.asarray(preds.spec_fnu) + shift})
        primitive, _, _ = marginalize_emission_lines(
            np.asarray(data["data"]) - np.asarray(moved.spec_fnu),
            err,
            np.asarray(preds.eline_design),
            prior_variance=LINE_SIGMA**2,
        )
        offsets.append(float(primitive) - float(score(moved, data, noise)))
    np.testing.assert_allclose(offsets, [constant, constant], rtol=1e-10)


def test_flat_line_prior_limit_is_the_log_width_asymptote():
    """For a very wide prior each line costs 1/2 ln(width ratio squared): d lnL/d ln Λ = -k/2."""
    preds, data, _noise = _synthetic("lines", "lines_marginalized")
    values = {}
    for variance in (1e16, 1e18):
        spec = NoiseSpec(
            data_type="spectroscopy",
            lines=LineSpec(mode="marginalized", prior_variance=variance),
        )
        values[variance] = float(score(preds, data, spec))
    expected = -0.5 * N_LINE * np.log(1e18 / 1e16)
    np.testing.assert_allclose(values[1e18] - values[1e16], expected, rtol=1e-6)


# ── Line columns of very different norms, and a collinear pair ─────────────────────────


def _out_of_band_lines():
    """The marginalized-line fixture of the golden values: 11 columns, 3 of them informative.

    Two columns are exactly zero (lines outside the wavelength range), five have whitened
    norms from 1e-30 to 1e-150, and two are exactly collinear (the whitened correlation is
    1 to the last float64 digit). The amplitude prior is a width of 100 against fluxes of 1e-10.
    """
    fitter = fixtures.fitter_of("spec_lines")
    term, physical = fixtures.build_term(fitter)
    preds = term.predictions(physical, fitter._data_args)
    data = fitter._data_args
    return (
        np.asarray(preds.spec_fnu),
        np.asarray(data["data"]),
        np.asarray(data["noise"]),
        np.asarray(preds.eline_design),
        float(term.noise.lines.prior_variance),
    )


def _svd_line_density(model, obs, err, design, variance):
    """Full normalized log-density of the lines marginalized under ``a ~ N(0, variance)``.

    From the singular values ``s`` of ``B = G sqrt(variance) / sigma``: the data covariance is
    ``sigma (I + B B^T) sigma``, so ``-1/2 sum log1p(s^2)`` is its log-determinant and the
    residual is ``r^T r - sum s^2/(1+s^2) (u . r)^2``. The prior enters as ``1 + s^2``, never as
    a difference of squares, so a null or collinear column costs nothing. Agrees with a 120-digit
    mpmath evaluation of the dense covariance to 6e-13 nat on the fixture.
    """
    err = np.asarray(err, dtype=float)
    whitened = np.asarray(design, dtype=float) * np.sqrt(variance) / err[:, None]
    r = (np.asarray(obs, dtype=float) - np.asarray(model, dtype=float)) / err
    u, sv, _ = np.linalg.svd(whitened, full_matrices=False)
    chi = r @ r - np.sum(sv**2 / (1.0 + sv**2) * (u.T @ r) ** 2)
    n = r.size
    return float(
        -0.5 * n * np.log(2 * np.pi)
        - np.sum(np.log(err))
        - 0.5 * np.sum(np.log1p(sv**2))
        - 0.5 * chi
    )


def _lines_value(model, obs, err, design, variance):
    from tengri.inference.likelihoods.terms import eline_flat_marginal_term

    return eline_flat_marginal_term(model, obs, err, design, prior_variance=variance)


def _well_conditioned_lines():
    """Five lines with columns of comparable norm, no collinearity, a prior width of 100."""
    rng = np.random.default_rng(3)
    model = 1e-10 * (1.0 + 0.3 * rng.random(N_PIX))
    err = 0.1 * model
    design = rng.standard_normal((N_PIX, 5)) * 0.2 * model.mean()
    obs = model * (1.0 + 0.1 * rng.standard_normal(N_PIX)) + design @ (0.5 * np.ones(5))
    return model, obs, err, design, 1e4


_LINE_FIXTURES = {"out_of_band": _out_of_band_lines, "well_conditioned": _well_conditioned_lines}

#: Measured: the value against the stable reference, float64 and eager or jitted, is within
#: 3e-16 relative on the first fixture; this leaves four orders of margin.
LINES_RTOL = 1e-12


@pytest.mark.parametrize("label", _LINE_FIXTURES)
def test_line_marginal_matches_the_stable_reference_eager_and_jitted(label):
    args = _LINE_FIXTURES[label]()
    reference = _svd_line_density(*args)
    eager = float(_lines_value(*args))
    jitted = float(jax.jit(_lines_value)(*args))
    np.testing.assert_allclose(eager, reference, rtol=LINES_RTOL)
    np.testing.assert_allclose(jitted, reference, rtol=LINES_RTOL)
    np.testing.assert_allclose(eager, jitted, rtol=LINES_RTOL)


def _moved(args, v, a, m):
    model, obs, err, design, variance = args
    return model * m, obs, err * a, design, variance * v


@pytest.mark.parametrize("label", _LINE_FIXTURES)
def test_line_marginal_gradients_are_finite_nonzero_and_match_finite_differences(label):
    """Gradient with respect to the prior variance, the noise scale and the model scale."""
    args = _LINE_FIXTURES[label]()

    def value(x):
        return _lines_value(*_moved(args, *x))

    point = np.ones(3)
    grad = np.asarray(jax.grad(value)(point))
    jitted = np.asarray(jax.jit(jax.grad(value))(point))
    assert np.all(np.isfinite(grad)), "a non-finite gradient would break every sampler step"
    assert np.all(grad != 0.0), "an identically zero gradient would freeze that parameter"
    np.testing.assert_allclose(jitted, grad, rtol=1e-9)
    step = 1e-4
    fd = np.empty(3)
    for k in range(3):
        up, down = point.copy(), point.copy()
        up[k] += step
        down[k] -= step
        fd[k] = (
            _svd_line_density(*_moved(args, *up)) - _svd_line_density(*_moved(args, *down))
        ) / (2 * step)
    np.testing.assert_allclose(grad, fd, rtol=1e-5)


#: Measured at float32 against the float64 reference: the value is within 1.6e-7 relative (the
#: spacing of float32 at 857 is 6e-5 nat), the gradients within 2.9e-6. Tolerances leave a factor
#: of 6 and 7.
LINES_RTOL_FLOAT32 = 1e-6
LINES_GRAD_RTOL_FLOAT32 = 2e-5


@pytest.mark.parametrize("label", _LINE_FIXTURES)
def test_line_marginal_in_float32_stays_within_its_measured_error(label):
    args = _LINE_FIXTURES[label]()
    reference = _svd_line_density(*args)
    single = (*(np.asarray(x, dtype=np.float32) for x in args[:4]), args[4])
    for run in (_lines_value, jax.jit(_lines_value)):
        np.testing.assert_allclose(float(run(*single)), reference, rtol=LINES_RTOL_FLOAT32)

    def value(x, inputs):
        return _lines_value(*_moved(inputs, *x))

    grad64 = np.asarray(jax.grad(value)(np.ones(3), args))
    grad32 = np.asarray(jax.grad(value)(np.ones(3, dtype=np.float32), single))
    assert np.all(np.isfinite(grad32)) and np.all(grad32 != 0.0)
    np.testing.assert_allclose(grad32, grad64, rtol=LINES_GRAD_RTOL_FLOAT32)


# ── Refusals by name ──────────────────────────────────────────────────────────────────


def test_unscorable_noise_is_refused_by_name():
    _preds, data, noise = _synthetic("calibration", "cal_marg")
    fitted = LineSpec(mode="fitted", amplitude_names=("a0",))
    both = NoiseSpec(data_type="spectroscopy", calibration=noise.calibration, lines=fitted)
    with pytest.raises(NotImplementedError, match="fitted"):
        channels.validate(both, data.keys())


def test_the_fitter_takes_heavy_tailed_noise_and_a_floor_with_marginalization():
    assert fixtures.fitter_of("spec_student_t_cal") is not None
    assert fixtures.fitter_of("spec_floor_cal") is not None


# ── Seams that read the mask outside the loss ─────────────────────────────────────────


def test_mask_reaches_the_sampler_log_likelihood_seams():
    """The nested-sampling and elliptical-slice log-likelihoods score the censoring mask.

    Both read ``Fitter._get_or_build_loglikelihood_fn`` and its unbounded twin, a seam beside
    the loss and the flat log-density. The path is checked as engaged (the censoring channel
    is the one selected, and the value differs from the unmasked Gaussian by more than a nat,
    the control that must fail) as well as numerically against the closed form.
    """
    from tengri.inference.loss_functions import _unstandardize_parameters

    fitter, term, physical = _term_of("mask")
    data_args = fitter._data_args
    assert [c.name for c in selected_channels(data_args, term.noise)] == ["censoring"]

    free = {k: physical[k] for k in fitter._free_names}
    predicted = np.asarray(fitter.model.predict_photometry(free))
    obs, err = np.asarray(data_args["data"]), np.asarray(data_args["noise"])
    mask = np.asarray(data_args["data_mask"])
    plain = NoiseSpec()
    expected = _ref_censored(predicted, obs, err, mask, plain, Predictions(), f_cal=0.0)
    unmasked = _ref_main(predicted, obs, err, plain, Predictions())
    assert abs(expected - unmasked) > 1.0

    constant = float(term.log_normalization(data_args))
    seam = fitter._get_or_build_loglikelihood_fn()
    np.testing.assert_allclose(float(seam(free, data_args)) + constant, expected, rtol=1e-9)

    zeros = {name: 0.0 for name in fitter._free_names}
    resolved = _unstandardize_parameters(
        zeros, fitter.spec, fitter._free_names, fitter._fixed_values, fitter.spec.stochastic
    )
    at_zero = {k: resolved[k] for k in fitter._free_names}
    predicted0 = np.asarray(fitter.model.predict_photometry(at_zero))
    expected0 = _ref_censored(predicted0, obs, err, mask, plain, Predictions(), f_cal=0.0)
    unbounded = fitter._get_or_build_loglikelihood_unbounded_fn()
    np.testing.assert_allclose(float(unbounded(zeros, data_args)) + constant, expected0, rtol=1e-9)


def test_photometry_and_line_limits_share_one_censoring_convention():
    """Upper and lower limits are the normal log-CDF of the same standardized residual.

    The censoring channel (photometry), the line-censor channel and the standalone
    ``LineFluxData.log_likelihood`` all read one function and agree, constants included.
    """
    from tengri.observation.line_flux_data import LineFluxData

    preds, data, noise = _synthetic("line_censor", "gaussian")
    sigma_frac = np.array([0.05, 0.1, 0.0, 0.2])
    line_value = float(score(preds, data, noise))
    constant = float(log_normalization(data, noise))

    as_photometry = {
        "data": data["line_flux_obs"],
        "noise": data["line_flux_err"],
        "data_mask": data["line_flux_limit_mask"],
    }
    zero_sigma = {**data, "line_flux_model_sigma_frac": np.zeros(N_CONSTRAINT)}
    no_model_error = float(score(preds, zero_sigma, noise))
    phot_value = float(
        score(
            Predictions(phot_fnu=preds.line_fluxes),
            as_photometry,
            NoiseSpec(data_type="photometry"),
        )
    )
    np.testing.assert_allclose(no_model_error, phot_value, rtol=1e-12)

    lines = LineFluxData(
        names=tuple(f"l{i}" for i in range(N_CONSTRAINT)),
        fluxes=data["line_flux_obs"],
        errors=data["line_flux_err"],
        wavelengths=np.linspace(4000.0, 7000.0, N_CONSTRAINT),
        is_upper_limit=data["line_flux_limit_mask"] == 1,
        is_lower_limit=data["line_flux_limit_mask"] == -1,
    )
    np.testing.assert_allclose(
        line_value + constant,
        float(lines.log_likelihood(preds.line_fluxes, model_sigma_frac=sigma_frac)),
        rtol=1e-12,
    )


def test_the_scored_error_is_the_one_a_noisy_mock_draws_from():
    """``compute_effective_noise`` is the error the floored likelihood scores with.

    The floored Gaussian density rebuilt from ``compute_effective_noise`` equals ``score`` plus its
    normalization; the control rebuilds it from the bare measurement error and must miss by
    many nats, so the comparison can fail.
    """
    preds, data, noise = _synthetic("photometry", "cal_floor")
    obs, err = np.asarray(data["data"]), np.asarray(data["noise"])
    pred = np.asarray(preds.phot_fnu)
    sigma = np.asarray(compute_effective_noise(err, pred, F_CAL))
    np.testing.assert_allclose(sigma, np.hypot(err, F_CAL * np.abs(pred)), rtol=1e-14)

    def gaussian(s):
        return float(np.sum(scipy.stats.norm.logpdf(obs, loc=pred, scale=s)))

    total = _full_density(preds, data, noise, "photometry")
    np.testing.assert_allclose(total, gaussian(sigma) if noise.dof is None else total, rtol=1e-9)
    control_miss = abs(total - gaussian(err))
    assert control_miss > 1.0, f"control differs by only {control_miss:.3g} nats"


@pytest.mark.parametrize(
    ("channel", "label"),
    [("calibration", "cal_marg_segments"), ("calibration_lines", "cal_marg_lines_segments")],
)
def test_gradient_wrt_the_log_mean_multiplier_matches_finite_differences(channel, label):
    """``cal_log_q`` enters the marginal through the fiber-fraction window; check its gradient."""
    preds, data, noise = _synthetic(channel, label)
    cal = noise.calibration
    pinned = CalibrationSpec(
        wavelength=cal.wavelength,
        n_poly=cal.n_poly,
        prior_sigma=cal.prior_sigma,
        segments=cal.segments,
        n_segments=cal.n_segments,
        q_window=np.ones(N_PIX),
        has_q=True,
    )
    spec = NoiseSpec(**{**noise.__dict__, "calibration": pinned})
    assert "cal_log_q" in _scalar_params(preds, spec)
    assert np.isfinite(float(score(preds, data, spec)))
    check_gradients(preds, data, spec)


# ── The full product of noise family x covariance x nuisance ──────────────────────────

FAMILIES = ("gaussian", "floor", "student_t")
COVARIANCE_KINDS = ("diagonal", "dense", "banded")
NUISANCES = ("none", "fitted", "calibration", "lines", "calibration_lines")
TERM_NAMES = (
    "diag_gaussian_log_prob",
    "student_t_term",
    "correlated_term",
    "correlated_noise_term",
    "calibration_marginal_term",
    "eline_flat_marginal_term",
    "calibration_eline_marginal_term",
)


def _product_cell(family, covariance, nuisance):
    rng = np.random.default_rng(0)
    pred = rng.uniform(1.0, 2.0, N_PIX)
    err = rng.uniform(0.05, 0.15, N_PIX)
    data = {"data": pred * (1 + 0.1 * rng.standard_normal(N_PIX)), "noise": err}
    cov_label = {"diagonal": "", "dense": "dense", "banded": "banded"}[covariance]
    cov_data, kw = _covariance_inputs(cov_label, err)
    data.update(cov_data)
    kw.update(data_type="spectroscopy")
    if family == "floor":
        kw.update(student_t=True, dof=None, f_cal_param="noise_frac_cal")
    if family == "student_t":
        kw.update(student_t=True, dof=DOF, f_cal_param="noise_frac_cal")
    wave = np.linspace(4000.0, 8000.0, N_PIX)
    calibration = CalibrationSpec(wavelength=wave, n_poly=2, prior_sigma=CAL_SIGMA)
    marginalized = LineSpec(mode="marginalized", prior_variance=LINE_SIGMA**2)
    fitted = LineSpec(mode="fitted", amplitude_names=("a0", "a1", "a2"))
    kw.update(
        {
            "none": {},
            "fitted": {"lines": fitted},
            "calibration": {"calibration": calibration},
            "lines": {"lines": marginalized},
            "calibration_lines": {"calibration": calibration, "lines": marginalized},
        }[nuisance]
    )
    preds_kw = {"spec_fnu": pred}
    if nuisance != "none" and nuisance != "calibration":
        preds_kw["eline_design"] = rng.standard_normal((N_PIX, N_LINE)) * 0.2
    params = {"noise_frac_cal": F_CAL, "a0": 0.1, "a1": 0.0, "a2": -0.1}
    return Predictions(params=params, **preds_kw), data, NoiseSpec(**kw)


def _expected_term(family, covariance, nuisance):
    if nuisance == "calibration":
        return "calibration_marginal_term"
    if nuisance == "lines":
        return "eline_flat_marginal_term"
    if nuisance == "calibration_lines":
        return "calibration_eline_marginal_term"
    heavy = family != "gaussian"
    if covariance == "diagonal":
        return "student_t_term" if heavy else "diag_gaussian_log_prob"
    return "correlated_noise_term" if heavy else "correlated_term"


@pytest.mark.parametrize("nuisance", NUISANCES)
@pytest.mark.parametrize("covariance", COVARIANCE_KINDS)
@pytest.mark.parametrize("family", FAMILIES)
def test_every_noise_family_covariance_and_nuisance_has_a_term_or_a_named_error(
    family, covariance, nuisance, monkeypatch
):
    """No combination is scored by a term that ignores part of the noise model."""
    preds, data, noise = _product_cell(family, covariance, nuisance)
    marginalized = nuisance in ("calibration", "lines", "calibration_lines")
    used = []
    for name in TERM_NAMES:
        original = getattr(channels, name)

        def spy(*args, _name=name, _original=original, **kwargs):
            used.append((_name, kwargs.get("chol_data") is not None))
            return _original(*args, **kwargs)

        monkeypatch.setattr(channels, name, spy)
    channels.validate(noise, data.keys())
    assert np.isfinite(float(score(preds, data, noise)))
    assert [name for name, _ in used] == [_expected_term(family, covariance, nuisance)]
    if marginalized:
        assert used[0][1] == (covariance != "diagonal")


# ── Presence on joint and spectroscopic data ──────────────────────────────────────────


def _joint_cell(family, with_mask):
    rng = np.random.default_rng(2)
    bands = rng.uniform(1.0, 2.0, N_BAND)
    pixels = rng.uniform(1.0, 2.0, N_PIX)
    obs = np.concatenate([bands, pixels]) * (1 + 0.1 * rng.standard_normal(N_BAND + N_PIX))
    err = np.concatenate([rng.uniform(0.05, 0.15, N_BAND), rng.uniform(0.05, 0.15, N_PIX)])
    obs[:N_BAND] = np.where(PRESENCE > 0, obs[:N_BAND], 0.0)
    err[:N_BAND] = np.where(PRESENCE > 0, err[:N_BAND], 0.0)
    data = {"data": obs, "noise": err, "presence": PRESENCE}
    kw = {"data_type": "joint", "n_phot": N_BAND}
    if family != "gaussian":
        kw.update(
            student_t=True,
            dof=DOF if family == "student_t" else None,
            f_cal_param="noise_frac_cal",
        )
    if with_mask:
        mask = np.zeros(N_BAND + N_PIX, dtype=int)
        mask[:N_BAND] = [0, 1, -1, 0, 1]
        mask[N_BAND + 3] = 1
        data["data_mask"] = mask
    preds = Predictions(phot_fnu=bands, spec_fnu=pixels, params={"noise_frac_cal": F_CAL})
    return preds, data, NoiseSpec(**kw)


@pytest.mark.parametrize("family", FAMILIES)
def test_presence_masks_the_photometric_block_of_joint_data(family):
    preds, data, noise = _joint_cell(family, with_mask=False)
    expected = reference_density("photometry", preds, data, noise) + reference_density(
        "spectrum", preds, data, noise
    )
    actual = float(score(preds, data, noise) + log_normalization(data, noise))
    np.testing.assert_allclose(actual, expected, rtol=1e-9)
    shifted = np.asarray(preds.phot_fnu).copy()
    shifted[PRESENCE == 0] += 2.0
    moved = Predictions(**{**preds.__dict__, "phot_fnu": shifted})
    assert float(score(moved, data, noise)) == float(score(preds, data, noise))
    grads = jax.grad(
        lambda p: score(Predictions(**{**preds.__dict__, "phot_fnu": p}), data, noise)
    )(np.asarray(preds.phot_fnu))
    assert np.all(np.asarray(grads)[PRESENCE == 0] == 0.0)


@pytest.mark.parametrize("family", FAMILIES)
def test_presence_masks_the_photometric_block_of_censored_joint_data(family):
    preds, data, noise = _joint_cell(family, with_mask=True)
    keep = np.concatenate([PRESENCE > 0, np.ones(N_PIX, dtype=bool)])
    model = np.concatenate([np.asarray(preds.phot_fnu), np.asarray(preds.spec_fnu)])
    expected = _ref_censored(
        model,
        np.asarray(data["data"]),
        np.asarray(data["noise"]),
        np.asarray(data["data_mask"]),
        noise,
        preds,
        keep=keep,
    )
    actual = float(score(preds, data, noise) + log_normalization(data, noise))
    np.testing.assert_allclose(actual, expected, rtol=1e-9)


def test_presence_on_spectroscopy_is_rejected_by_name():
    preds, data, noise = _synthetic("spectrum", "gaussian")
    with_presence = {**data, "presence": np.ones(N_PIX)}
    with pytest.raises(ValueError, match="presence"):
        score(preds, with_presence, noise)
    with pytest.raises(ValueError, match="presence"):
        channels.validate(noise, with_presence.keys())


def test_a_presence_entry_of_the_wrong_length_is_rejected():
    preds, data, noise = _joint_cell("gaussian", with_mask=False)
    bad = {**data, "presence": np.ones(N_BAND + 1)}
    with pytest.raises(ValueError, match="presence has"):
        score(preds, bad, noise)


# ── The calibration floor with marginalized nuisance terms ────────────────────────────

FLOOR_PAIRS = (
    ("calibration", "floor_cal_marg", "cal_marg"),
    ("calibration", "floor_cal_marg_segments", "cal_marg_segments"),
    ("lines", "floor_lines_marginalized", "lines_marginalized"),
    ("lines", "floor_lines_marginalized_cloudy", "lines_marginalized_cloudy"),
    ("lines", "floor_lines_marginalized_model", "lines_marginalized_model"),
    ("calibration_lines", "floor_cal_marg_lines", "cal_marg_lines"),
    ("calibration", "floor_cal_marg_dense", "cal_marg_dense"),
    ("calibration_lines", "floor_cal_marg_lines_banded", "cal_marg_lines_banded"),
)


@pytest.mark.parametrize(("channel", "floored", "plain"), FLOOR_PAIRS)
def test_a_vanishing_floor_is_the_marginal_without_a_floor(channel, floored, plain):
    preds, data, noise = _synthetic(channel, floored)
    _, plain_data, plain_noise = _synthetic(channel, plain)
    none = Predictions(**{**preds.__dict__, "params": {**preds.params, "noise_frac_cal": 1e-12}})
    with_floor = float(score(none, data, noise) + log_normalization(data, noise))
    without = float(
        score(none, plain_data, plain_noise) + log_normalization(plain_data, plain_noise)
    )
    np.testing.assert_allclose(with_floor, without, rtol=1e-9)


def test_the_floor_is_scaled_by_the_model_before_the_nuisance_terms():
    """N = diag(sigma^2 + (f m0)^2) with m0 the model the nuisance multiplies or adds to.

    A floor scaled by another model, here one half larger, is a different density that the
    reference rejects by far more than its tolerance.
    """
    preds, data, noise = _synthetic("calibration_lines", "floor_cal_marg_lines")
    obs, err = _blocks(data, noise, "spec")
    m0, design = np.asarray(preds.spec_fnu), np.asarray(preds.eline_design)
    cal = noise.calibration
    basis = m0[:, None] * _calibration_basis(cal)
    width = np.full(design.shape[1], noise.lines.prior_sigma)

    def density(floor_model):
        cov = (
            _noise_cov(preds, data, noise, err, floor_model)
            + cal.prior_sigma**2 * basis @ basis.T
            + (design * width) @ (design * width).T
        )
        return _mvn(obs - m0, cov)

    actual = float(score(preds, data, noise) + log_normalization(data, noise))
    np.testing.assert_allclose(actual, density(m0), rtol=1e-10)
    assert abs(density(1.5 * m0) - actual) > 0.05, (
        "the control must differ by many tenths of a nat"
    )


def test_the_floored_marginal_depends_on_the_model_through_its_log_determinant():
    """Moving m0 changes ln det N, so the value is not the no-floor value plus a constant."""
    preds, data, noise = _synthetic("calibration", "floor_cal_marg")
    plain_noise = NoiseSpec(**{**noise.__dict__, "student_t": False, "f_cal_param": None})
    shifted = Predictions(**{**preds.__dict__, "spec_fnu": 1.4 * np.asarray(preds.spec_fnu)})
    floored_change = float(score(shifted, data, noise) - score(preds, data, noise))
    plain_change = float(score(shifted, data, plain_noise) - score(preds, data, plain_noise))
    assert abs(floored_change - plain_change) > 0.05


@pytest.mark.parametrize(("channel", "floored", "plain"), FLOOR_PAIRS)
def test_the_gradient_through_the_model_includes_the_floored_log_determinant(
    channel, floored, plain
):
    """d score / d (scale of m0) matches central differences and is not the no-floor gradient."""
    preds, data, noise = _synthetic(channel, floored)
    _, plain_data, plain_noise = _synthetic(channel, plain)
    base = np.asarray(preds.spec_fnu, dtype=float)

    def value(scale, spec_data, spec_noise):
        shifted = Predictions(**{**preds.__dict__, "spec_fnu": scale * base})
        return score(shifted, spec_data, spec_noise)

    grad = float(jax.grad(value)(1.0, data, noise))
    eps = 1e-6
    fd = (float(value(1.0 + eps, data, noise)) - float(value(1.0 - eps, data, noise))) / (2 * eps)
    np.testing.assert_allclose(grad, fd, rtol=1e-5)
    plain_grad = float(jax.grad(value)(1.0, plain_data, plain_noise))
    assert abs(grad - plain_grad) > 1e-3 * abs(plain_grad)
