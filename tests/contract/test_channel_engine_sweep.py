# SPDX-License-Identifier: BSD-3-Clause
"""Engine x channel x noise sweep of the one likelihood scorer (#3505).

Every inference engine must score a galaxy through :func:`tengri.inference.channels.score`,
so a data channel or noise model added once works in every engine. The matrix is generated
from the channel registry, not a hand list. Only the sequential engine (``DataTerm`` through
the Fitter) is on the scorer today; every other engine's cells are strict ``xfail`` and flip
to a failure the moment that engine is registered in ``ENGINES_ON_SCORER``.

Live cells assert, for each channel and noise pair:

1. ``score`` on the model's own predictions equals ``DataTerm.log_prob`` (rtol 1e-12), where
   a Fitter of that structure exists;
2. ``score + log_normalization`` equals an independent numpy/scipy density written below
   (rtol 1e-9). Those references share no code with the scorer;
3. the gradient of ``score`` with respect to the predictions is finite.
"""

from __future__ import annotations

import warnings

import jax
import numpy as np
import pytest
import scipy.linalg
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
NOISES = (
    "gaussian",
    "student_t",
    "cal_floor",
    "dense_cov",
    "banded_cov",
    "cal_marg",
    "cal_marg_banded",
    "lines_marginalized",
    "lines_marginalized_banded",
    "lines_fitted",
    "cal_marg_lines",
    "cal_marg_lines_banded",
    "cal_marg_fitted_lines",
)
DOF = 4.0
F_CAL = 0.07
N_BAND, N_PIX, N_LINE, N_CONSTRAINT = 5, 24, 3, 4
CAL_SIGMA, LINE_SIGMA = 0.3, 2.0

SPECTRUM_BLOCK = ("spectrum", "calibration", "lines", "calibration_lines")
SPECTRUM_ONLY_NOISE = NOISES[3:]
CONSTRAINTS = ("line_fluxes", "line_censor", "line_ratios", "spectral_indices")

# Cells where an independent reference disagrees with the implementation beyond 1e-9.
# Each entry carries the measured discrepancy, and the cell is a strict xfail.
LINES_CONSTANT = (
    "the flat-prior line marginal returns the density without its Gaussian constant "
    "(reference - score = -(0.5 N ln 2pi + sum ln sigma), to 1e-12) and log_normalization "
    "supplies none, so the evidence omits it; measured relative difference "
)
JOINT_CONSTANT = (
    "the joint calibration + line marginal adds 0.5 N ln 2pi + sum ln sigma to a primitive that "
    "already carries its normalization (the primitive alone equals the reference to 1e-13), "
    "so the value is off by that data-only constant; measured relative difference "
)
KNOWN_DISCREPANCIES: dict = {
    ("lines", "lines_marginalized"): LINES_CONSTANT + "2.96",
    ("lines", "lines_marginalized_banded"): LINES_CONSTANT + "3.07",
    ("calibration_lines", "cal_marg_lines"): JOINT_CONSTANT + "5.93",
    ("calibration_lines", "cal_marg_lines_banded"): JOINT_CONSTANT + "4.08",
    ("presence", "student_t"): (
        "Student-t photometry scores every band: an absent band (presence 0) enters the "
        "density, where the independent reference drops it; measured relative difference 3.7e-2"
    ),
    ("presence", "cal_floor"): (
        "the calibration-floor photometry term scores every band: an absent band enters the "
        "density, where the independent reference drops it; measured relative difference 7.2e-3"
    ),
}


def exclusion_reason(channel: str, noise: str) -> str | None:
    """Why a channel and noise pair is contradictory, or ``None`` when it is a live cell."""
    if noise == "cal_marg_fitted_lines":
        return (
            "calibration marginalization with fitted line amplitudes cannot share one "
            "marginal; validate() refuses it (test_contradictory_noise_is_refused)"
        )
    if noise in SPECTRUM_ONLY_NOISE and channel not in SPECTRUM_BLOCK:
        return f"{noise} acts on the spectrum pixels; {channel} does not read them"
    if channel in CONSTRAINTS and noise != "gaussian":
        return (
            f"{channel} is a Gaussian constraint on a few measured numbers; the {noise} noise "
            "model applies to the main data block only"
        )
    if channel == "presence" and noise not in ("gaussian", "student_t", "cal_floor"):
        return "presence masks photometry bands; the noise model is a spectrum one"
    if channel == "spectrum" and noise in SPECTRUM_ONLY_NOISE[2:]:
        return f"a {noise} spectrum is scored by its marginalization channel, not `spectrum`"
    if channel == "calibration" and noise not in ("cal_marg", "cal_marg_banded"):
        return "the calibration channel needs a calibration-marginalization noise model"
    if channel == "lines" and noise not in (
        "lines_marginalized",
        "lines_marginalized_banded",
        "lines_fitted",
    ):
        return "the lines channel needs a line-amplitude noise model"
    if channel == "calibration_lines" and noise not in ("cal_marg_lines", "cal_marg_lines_banded"):
        return "the joint channel needs calibration and marginalized lines together"
    return None


# ── Independent references (numpy and scipy only) ────────────────────────────────────


def _blocks(data, noise, block):
    obs, err = np.asarray(data["data"]), np.asarray(data["noise"])
    if noise.data_type != "joint":
        return obs, err
    cut = noise.n_phot
    return (obs[:cut], err[:cut]) if block == "phot" else (obs[cut:], err[cut:])


def _sigma_eff(err, pred, f_cal):
    return np.sqrt(err**2 + (f_cal * np.abs(pred)) ** 2)


def _f_cal_of(preds, noise):
    name = noise.f_cal_param
    return float(preds.params[name]) if name and name in preds.params else 0.0


def _ref_main(pred, obs, err, noise, preds, presence=None):
    """Gaussian, floored Gaussian or Student-t density of the main data."""
    sigma = _sigma_eff(err, pred, _f_cal_of(preds, noise)) if noise.student_t else err
    r = (obs - pred) / sigma
    if noise.student_t and noise.dof is not None:
        per = scipy.stats.t.logpdf(r, noise.dof) - np.log(sigma)
    else:
        per = scipy.stats.norm.logpdf(r) - np.log(sigma)
    if presence is not None:
        per = np.where(np.asarray(presence) > 0, per, 0.0)
    return float(np.sum(per))


def _ref_censored(pred, obs, err, mask, noise, preds, model_sigma_frac=None, f_cal=None):
    f_cal = _f_cal_of(preds, noise) if f_cal is None else f_cal
    model_sigma = 0.0 if model_sigma_frac is None else np.asarray(model_sigma_frac)
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
    return np.stack(
        [np.polynomial.chebyshev.Chebyshev.basis(k)(x) for k in range(1, n_poly + 1)], 1
    )


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
    return float(scipy.stats.multivariate_normal.logpdf(resid, mean=np.zeros(len(resid)), cov=cov))


def _ref_spectrum(preds, data, noise):
    obs, err = _blocks(data, noise, "spec")
    pred = np.asarray(preds.spec_fnu)
    if noise.student_t or noise.covariance == "diagonal":
        return _ref_main(pred, obs, err, noise, preds)
    return _mvn(obs - pred, _pixel_covariance(data, noise, err))


def _ref_calibration(preds, data, noise):
    obs, err = _blocks(data, noise, "spec")
    pred, cal = np.asarray(preds.spec_fnu), noise.calibration
    basis = pred[:, None] * _chebyshev(np.asarray(cal.wavelength), cal.n_poly)
    cov = _pixel_covariance(data, noise, err) + cal.prior_sigma**2 * basis @ basis.T
    return _mvn(obs - pred, cov)


def _ref_lines(preds, data, noise):
    obs, err = _blocks(data, noise, "spec")
    pred, design = np.asarray(preds.spec_fnu), np.asarray(preds.eline_design)
    lines = noise.lines
    if lines.mode == "fitted":
        amps = np.array([float(preds.params[n]) for n in lines.amplitude_names])
        return float(-0.5 * np.sum(((obs - pred - design @ amps) / err) ** 2))
    sigma = LINE_SIGMA if lines.prior_variance is None else lines.prior_variance**0.5
    cov = _pixel_covariance(data, noise, err) + sigma**2 * design @ design.T
    return _mvn(obs - pred, cov)


def _ref_calibration_lines(preds, data, noise):
    obs, err = _blocks(data, noise, "spec")
    pred, design = np.asarray(preds.spec_fnu), np.asarray(preds.eline_design)
    cal = noise.calibration
    basis = pred[:, None] * _chebyshev(np.asarray(cal.wavelength), cal.n_poly)
    sigma = noise.lines.prior_sigma
    cov = (
        _pixel_covariance(data, noise, err)
        + cal.prior_sigma**2 * basis @ basis.T
        + sigma**2 * design @ design.T
    )
    return _mvn(obs - pred, cov)


def reference_density(channel, preds, data, noise):
    """The independent log-density a channel must reproduce, constants included.

    Returns ``(value, includes_constants)``: a channel without a normalization of its own
    (fitted lines) is compared on the part ``score`` carries.
    """
    if channel == "photometry":
        obs, err = _blocks(data, noise, "phot")
        return _ref_main(np.asarray(preds.phot_fnu), obs, err, noise, preds), True
    if channel == "presence":
        obs, err = np.asarray(data["data"]), np.asarray(data["noise"])
        pred = np.asarray(preds.phot_fnu)
        return _ref_main(pred, obs, err, noise, preds, data["presence"]), True
    if channel == "censoring":
        pred = np.asarray(preds.phot_fnu)
        mask = np.asarray(data["data_mask"])
        return _ref_censored(
            pred, np.asarray(data["data"]), np.asarray(data["noise"]), mask, noise, preds
        ), True
    if channel == "spectrum":
        return _ref_spectrum(preds, data, noise), True
    if channel == "calibration":
        return _ref_calibration(preds, data, noise), True
    if channel == "lines":
        return _ref_lines(preds, data, noise), noise.lines.mode != "fitted"
    if channel == "calibration_lines":
        return _ref_calibration_lines(preds, data, noise), True
    return _reference_constraint(channel, preds, data, noise), True


def _reference_constraint(channel, preds, data, noise):
    if channel == "line_fluxes":
        return _ref_main(
            np.asarray(preds.line_fluxes),
            np.asarray(data["line_flux_obs"]),
            np.asarray(data["line_flux_err"]),
            NoiseSpec(),
            preds,
        )
    if channel == "line_censor":
        return _ref_censored(
            np.asarray(preds.line_fluxes),
            np.asarray(data["line_flux_obs"]),
            np.asarray(data["line_flux_err"]),
            np.asarray(data["line_flux_limit_mask"]),
            NoiseSpec(),
            preds,
            model_sigma_frac=data["line_flux_model_sigma_frac"],
            f_cal=0.0,
        )
    key = {"line_ratios": ("line_ratio_obs", "line_ratio_err", "line_ratios")}.get(
        channel, ("index_obs", "index_err", "indices")
    )
    return _ref_main(
        np.asarray(getattr(preds, key[2])),
        np.asarray(data[key[0]]),
        np.asarray(data[key[1]]),
        NoiseSpec(),
        preds,
    )


# ── Synthetic cells ───────────────────────────────────────────────────────────────────


def _synthetic(channel: str, noise_label: str, seed: int = 0):
    """Predictions, data and noise spec of one live cell, drawn at random."""
    rng = np.random.default_rng(seed)
    wave = np.linspace(4000.0, 8000.0, N_PIX)

    def draw(n, scale=1.0):
        pred = rng.uniform(1.0, 2.0, n) * scale
        return (
            pred,
            pred * (1.0 + 0.1 * rng.standard_normal(n)),
            rng.uniform(0.05, 0.15, n) * scale,
        )

    phot_p, phot_d, phot_e = draw(N_BAND)
    spec_p, spec_d, spec_e = draw(N_PIX)
    params = {"noise_frac_cal": F_CAL, "a0": 0.3, "a1": -0.2, "a2": 0.1}
    data: dict = {}
    kw: dict = {}
    preds_kw: dict = {}
    if noise_label in ("student_t", "cal_floor"):
        kw = {
            "student_t": True,
            "dof": DOF if noise_label == "student_t" else None,
            "f_cal_param": "noise_frac_cal",
        }
    if channel in ("photometry", "presence", "censoring"):
        data = {"data": phot_d, "noise": phot_e}
        kw.update(data_type="photometry")
        preds_kw = {"phot_fnu": phot_p}
        if channel == "presence":
            data["presence"] = np.array([1.0, 0.0, 1.0, 1.0, 0.0])
        if channel == "censoring":
            data["data_mask"] = np.array([0, 1, -1, 0, 1])
            kw["dof"] = DOF if noise_label == "student_t" else None
            kw["f_cal_param"] = "noise_frac_cal" if noise_label != "gaussian" else None
    elif channel in SPECTRUM_BLOCK:
        data, kw2, preds_kw = _synthetic_spectrum(
            channel, noise_label, spec_p, spec_d, spec_e, wave, rng
        )
        kw.update(kw2, data_type="spectroscopy")
    else:
        data, preds_kw = _synthetic_constraint(channel, rng)
        kw.update(data_type="photometry")
    return Predictions(params=params, **preds_kw), data, NoiseSpec(**kw)


def _synthetic_constraint(channel, rng):
    pred = rng.uniform(1.0, 2.0, N_CONSTRAINT)
    obs, err = (
        pred * (1 + 0.1 * rng.standard_normal(N_CONSTRAINT)),
        rng.uniform(0.05, 0.15, N_CONSTRAINT),
    )
    if channel == "line_fluxes":
        extra = {
            "line_flux_obs": obs,
            "line_flux_err": err,
            "line_flux_waves": np.arange(N_CONSTRAINT),
        }
        return extra, {"line_fluxes": pred}
    if channel == "line_censor":
        extra = {
            "line_flux_obs": obs,
            "line_flux_err": err,
            "line_flux_waves": np.arange(N_CONSTRAINT),
            "line_flux_limit_mask": np.array([0, 1, -1, 0]),
            "line_flux_model_sigma_frac": np.array([0.05, 0.1, 0.0, 0.2]),
        }
        return extra, {"line_fluxes": pred}
    if channel == "line_ratios":
        return {"line_ratio_obs": obs, "line_ratio_err": err}, {"line_ratios": pred}
    return {"index_obs": obs, "index_err": err}, {"indices": pred}


def _covariance_inputs(label, err):
    """Data entries and noise fields of the pixel covariance of a noise label."""
    n = err.size
    cov = np.diag(err**2)
    if "dense" in label or "banded" in label:
        lags = (
            tuple((k, 0.5**k) for k in range(1, n)) if "dense" in label else ((1, 0.3), (2, 0.1))
        )
        for lag, rho in lags:
            idx = np.arange(n - lag)
            cov[idx, idx + lag] = rho * err[idx] * err[idx + lag]
            cov[idx + lag, idx] = cov[idx, idx + lag]
    if "dense" in label:
        return {"spec_cov_inv": np.linalg.inv(cov)}, {"covariance": "dense"}
    if "banded" in label:
        factor = banded_cholesky(dense_to_banded(cov))
        offsets = tuple(int(o) for o in np.asarray(factor.offsets))
        return {"spec_chol_data": np.asarray(factor.data)}, {
            "covariance": "banded",
            "chol_offsets": offsets,
        }
    return {}, {}


def _synthetic_spectrum(channel, label, spec_p, spec_d, spec_e, wave, rng):
    data = {"data": spec_d, "noise": spec_e}
    cov_data, kw = _covariance_inputs(label, spec_e)
    data.update(cov_data)
    preds_kw = {"spec_fnu": spec_p}
    cal = CalibrationSpec(wavelength=wave, n_poly=3, prior_sigma=CAL_SIGMA)
    if label.startswith("cal_marg") and "lines" not in label:
        kw["calibration"] = cal
    if label.startswith("lines_marginalized"):
        kw["lines"] = LineSpec(mode="marginalized", prior_variance=LINE_SIGMA**2)
    if label == "lines_fitted":
        kw["lines"] = LineSpec(mode="fitted", amplitude_names=("a0", "a1", "a2"))
    if label.startswith("cal_marg_lines"):
        kw["calibration"] = cal
        kw["lines"] = LineSpec(mode="marginalized", prior_sigma=LINE_SIGMA)
    if "lines" in label:
        preds_kw["eline_design"] = (
            rng.standard_normal((N_PIX, N_LINE)) * np.abs(spec_p.mean()) * 0.2
        )
    return data, kw, preds_kw


# ── Cell checks ───────────────────────────────────────────────────────────────────────


def noise_label_of(noise):
    """The sweep label of the Student-t and floor noise models the presence cells use."""
    if noise.student_t:
        return "student_t" if noise.dof is not None else "cal_floor"
    return "gaussian"


def _full_density(preds, data, noise, channel):
    """``score + log_normalization`` of one channel, constants included when it has them."""
    chosen = [c.name for c in selected_channels(data, noise)]
    if (channel, noise_label_of(noise)) not in KNOWN_DISCREPANCIES:
        assert chosen == [channel], (channel, chosen)
    value = score(preds, data, noise)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(value + log_normalization(data, noise))


def cell_discrepancy(channel: str, noise_label: str, preds=None, data=None, noise=None) -> float:
    """Relative difference between the scorer and the independent reference of one cell."""
    if preds is None:
        preds, data, noise = _synthetic(channel, noise_label)
    expected, with_constants = reference_density(channel, preds, data, noise)
    actual = _full_density(preds, data, noise, channel)
    if not with_constants:
        actual = float(score(preds, data, noise))
    return abs(actual - expected) / max(abs(expected), 1e-300)


def _gradient_is_finite(preds, data, noise):
    names = [
        n
        for n in ("phot_fnu", "spec_fnu", "line_fluxes", "line_ratios", "indices", "eline_design")
        if getattr(preds, n) is not None
    ]

    def total(arrays):
        replaced = Predictions(**{**preds.__dict__, **dict(zip(names, arrays, strict=True))})
        return score(replaced, data, noise)

    grads = jax.grad(total)([np.asarray(getattr(preds, n), dtype=float) for n in names])
    return all(np.all(np.isfinite(np.asarray(g))) for g in grads)


# Fitter structure that realizes a (channel, noise) cell on a real model, for assertion (i).
FITTER_KINDS = {
    ("photometry", "gaussian"): "phot",
    ("photometry", "student_t"): "phot_student_t",
    ("photometry", "cal_floor"): "phot_floor",
    ("presence", "gaussian"): "presence",
    ("censoring", "gaussian"): "mask",
    ("censoring", "student_t"): "mask_student_t",
    ("censoring", "cal_floor"): "mask_floor",
    ("spectrum", "gaussian"): "spec",
    ("spectrum", "student_t"): "spec_student_t",
    ("spectrum", "dense_cov"): "spec_dense",
    ("spectrum", "banded_cov"): "spec_banded",
    ("calibration", "cal_marg"): "spec_cal",
    ("lines", "lines_marginalized"): "spec_lines",
    ("lines", "lines_fitted"): "spec_lines_fitted",
    ("calibration_lines", "cal_marg_lines"): "spec_cal_lines",
}
_FITTER_CACHE: dict = {}


def _term_of(kind):
    if kind not in _FITTER_CACHE:
        fitter = fixtures.fitter_of(kind)
        term, physical = fixtures.build_term(fitter)
        _FITTER_CACHE[kind] = (fitter, term, physical)
    return _FITTER_CACHE[kind]


def _sequential_cell(channel, noise_label):
    preds, data, noise = _synthetic(channel, noise_label)
    assert cell_discrepancy(channel, noise_label, preds, data, noise) < 1e-9
    assert _gradient_is_finite(preds, data, noise)
    kind = FITTER_KINDS.get((channel, noise_label))
    if kind is None:
        return
    fitter, term, physical = _term_of(kind)
    data_args = fitter._data_args
    real = term.predictions(physical, data_args)
    scored = float(score(real, data_args, term.noise))
    np.testing.assert_allclose(scored, float(term.log_prob(physical, data_args)), rtol=1e-12)
    assert cell_discrepancy(channel, noise_label, real, data_args, term.noise) < 1e-9
    assert _gradient_is_finite(real, data_args, term.noise)


def _cells():
    for channel in channel_keys():
        for noise in NOISES:
            yield channel, noise


LIVE = [(c, n) for c, n in _cells() if exclusion_reason(c, n) is None]
EXCLUDED = [(c, n) for c, n in _cells() if exclusion_reason(c, n) is not None]
NOT_ON_SCORER = "engine not yet on the shared scorer (#3505)"


def _params():
    for engine in ENGINES:
        for channel, noise in LIVE:
            marks = []
            if engine != "sequential":
                marks = [pytest.mark.xfail(strict=True, reason=NOT_ON_SCORER)]
            elif (channel, noise) in KNOWN_DISCREPANCIES:
                marks = [
                    pytest.mark.xfail(strict=True, reason=KNOWN_DISCREPANCIES[(channel, noise)])
                ]
            yield pytest.param(
                engine, channel, noise, marks=marks, id=f"{engine}-{channel}-{noise}"
            )


@pytest.mark.parametrize(("engine", "channel", "noise"), list(_params()))
def test_engine_scores_channel_under_noise(engine, channel, noise):
    if engine == "sequential":
        _sequential_cell(channel, noise)
        return
    assert engine in channels.ENGINES_ON_SCORER


def test_the_matrix_is_complete_and_every_exclusion_is_explained():
    assert len(LIVE) + len(EXCLUDED) == len(channel_keys()) * len(NOISES)
    assert all(len(exclusion_reason(c, n)) > 20 for c, n in EXCLUDED)
    assert {c for c, _ in LIVE} == set(channel_keys())


def test_only_the_sequential_engine_is_on_the_scorer():
    assert set(channels.ENGINES_ON_SCORER) == {"sequential"}
    assert set(ENGINES) >= set(channels.ENGINES_ON_SCORER)


def test_contradictory_noise_is_refused():
    _, data, noise = _synthetic("calibration", "cal_marg")
    contradictory = NoiseSpec(
        data_type="spectroscopy",
        calibration=noise.calibration,
        lines=LineSpec(mode="fitted", amplitude_names=("a0",)),
    )
    with pytest.raises(NotImplementedError, match="fitted"):
        channels.validate(contradictory, data.keys())


# ── Registry ──────────────────────────────────────────────────────────────────────────


def test_registry_is_derived_and_complete():
    registry = channel_keys()
    assert list(registry) == [c.name for c in channels.CHANNELS]
    assert len(set(registry)) == len(registry)
    assert {"photometry", "presence", "censoring", "line_fluxes", "line_censor"} <= set(registry)


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
