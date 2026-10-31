# SPDX-License-Identifier: BSD-3-Clause
"""Cue's ionizing shape is the power-law fit of the composite ionizing spectrum.

For a stellar population with weights ``w_a`` over SSP ages, the nebular emission
sees the composite ``sum_a w_a f_a(lambda)``, interpolated in metallicity. The shape
is the four-segment power-law fit of that composite (Li et al. 2025, ApJ 986, 9,
arXiv:2405.04598). tengri solves each segment in JAX to its stationary point and
differentiates through that stationarity condition.

The reference is :func:`fit_ionizing_spectrum`, whose SciPy SLSQP answer is polished
to the stationary point of the same objective. The tests compare against that
converged minimum and assert agreement to 1e-8. The SLSQP answer on its own is not a
reference for that bound, because SLSQP stops at its own tolerance.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import minimize

from tengri.components.nebular._constants import _C_AA, _H_PLANCK
from tengri.components.nebular.cue import CueBackend
from tengri.components.nebular.ionizing_spectrum import (
    SEGMENT_EDGES,
    _compute_segment_luminosities,
    _np_trapz,
    fit_ionizing_spectrum,
)
from tengri.config.exceptions import CueIonizingShapeWarning

pytestmark = pytest.mark.regression_paper

_CUE_WEIGHTS = "data/cue_weights.npz"
_LOG_H = np.log10(_H_PLANCK)
_LOG_Z_SOLAR = float(np.log10(0.0142))  # absolute log10(Z) at solar
_LOG_Z_TENTH_SOLAR = _LOG_Z_SOLAR - 1.0
_MAX_LOG_AGE_YR = 8.0  # the young window the Cue weighting uses
_IONSPEC_KEYS = (
    "ionspec_index1",
    "ionspec_index2",
    "ionspec_index3",
    "ionspec_index4",
    "ionspec_logLratio1",
    "ionspec_logLratio2",
    "ionspec_logLratio3",
)
#: Agreement of the JAX shape with the reference, which is the converged minimum.
_REF_TOL = 1e-8


def _backend(ssp):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CueIonizingShapeWarning)
        return CueBackend(_CUE_WEIGHTS, ssp_data=ssp)


def _ages(ssp):
    return np.asarray(ssp.ssp_lg_age_gyr) + 9.0


def _weights(ssp, kind):
    """SSP-age weights (Msun per bin) for the named star-formation history."""
    ages = _ages(ssp)
    t_myr = 10.0 ** (ages - 6.0)
    dt = np.gradient(10.0**ages)
    young = ages <= _MAX_LOG_AGE_YR
    if kind == "constant":
        shape = np.ones_like(t_myr)
    elif kind == "burst_3myr":
        shape = np.exp(-0.5 * ((t_myr - 3.0) / 0.75) ** 2)
    elif kind == "burst_15myr":
        shape = np.exp(-0.5 * ((t_myr - 15.0) / 3.75) ** 2)
    elif kind == "exp_rise":
        shape = np.exp(t_myr / 50.0)
    else:  # pragma: no cover - guard against a typo in a parametrization
        raise ValueError(kind)
    return np.where(young & (shape > 0), shape * dt, 0.0)


def _composite_linear(ssp, weights, log_z):
    """Linear composite spectrum, interpolated in metallicity, summed over ages."""
    flux = np.asarray(ssp.ssp_flux, dtype=np.float64)
    lgmet = np.asarray(ssp.ssp_lgmet, dtype=np.float64)
    iz = int(np.clip(np.searchsorted(lgmet, log_z) - 1, 0, len(lgmet) - 2))
    fz = (log_z - lgmet[iz]) / (lgmet[iz + 1] - lgmet[iz])
    f = (1.0 - fz) * flux[iz] + fz * flux[iz + 1]  # (n_age, n_wave)
    return np.sum(weights[:, None] * f, axis=0)


def _ind_bin(wave):
    ind = np.array([max(np.where(wave <= e)[0]) for e in SEGMENT_EDGES[1:]]) + 1
    return np.insert(ind, 0, 0)


def _segment_problems(wave, comp):
    """Per segment: normalized log flux, log Q, log wave and the pixel endpoints.

    Mirrors :func:`fit_ionizing_spectrum` and :func:`_fit_segment`, so that the
    objective evaluated here is the one the reference minimizes.
    """
    ind_bin = _ind_bin(wave)
    ref_flux = np.median(comp[ind_bin[-1] - 1 : ind_bin[-1] + 1])
    norm = 1e-18 / max(ref_flux, 1e-99)
    normalized = np.clip(comp * norm, 1e-70 * norm, np.inf)
    problems = []
    for k in range(4):
        seg_wave = wave[ind_bin[k] : ind_bin[k + 1]]
        seg_flux = normalized[ind_bin[k] : ind_bin[k + 1]]
        nu = (_C_AA / seg_wave).astype(np.float64)
        integrand = seg_flux.astype(np.float64) / (_H_PLANCK * nu)
        q_seg = np.abs(_np_trapz(integrand[::-1], x=nu[::-1]))
        problems.append(
            dict(
                log_wave=np.log10(seg_wave),
                log_flux=np.log10(np.maximum(seg_flux, 1e-99)),
                log_Q=np.log10(max(q_seg, 1e-99)),
                xmin=seg_wave[0],
                xmax=seg_wave[-1],
                log_norm=np.log10(norm),
            )
        )
    return problems


def _objective(p_norm, prob):
    """The reference objective of one segment, at normalized ``(alpha, b)``."""
    alpha, b = p_norm
    n = len(prob["log_wave"])
    pred = b + alpha * prob["log_wave"]
    log_q_pred = (
        b - _LOG_H + np.log10(np.abs((prob["xmax"] ** alpha - prob["xmin"] ** alpha) / alpha))
    )
    return (
        0.5 * np.sum((prob["log_flux"] - pred) ** 2) + 0.5 * n * (prob["log_Q"] - log_q_pred) ** 2
    )


def _polished_coeff(wave, comp, ref_powerlaw):
    """Each segment minimized to convergence from the SLSQP answer, denormalized."""
    out = np.zeros((4, 2))
    for k, prob in enumerate(_segment_problems(wave, comp)):
        x0 = np.array([ref_powerlaw[k, 0], ref_powerlaw[k, 1] + prob["log_norm"]])
        res = minimize(
            _objective, x0, args=(prob,), method="BFGS", options={"gtol": 1e-12, "maxiter": 5000}
        )
        out[k] = [res.x[0], res.x[1] - prob["log_norm"]]
    return out


def _seven_from_coeff(coeff, edges=SEGMENT_EDGES):
    log_L = _compute_segment_luminosities(coeff, edges)
    return np.concatenate([coeff[:, 0], np.diff(log_L)])


def _reference(ssp, weights, log_z):
    """The 7-vector of fit_ionizing_spectrum on the linear composite, and its inputs."""
    wave = np.asarray(ssp.ssp_wave, dtype=np.float64)
    comp = _composite_linear(ssp, weights, log_z)
    ref = fit_ionizing_spectrum(wave, comp)
    ref_pl = np.asarray(ref["powerlaw_params"])
    ref7 = np.concatenate([ref_pl[:, 0], np.diff(ref["log_seglum"])])
    return ref7, ref_pl, wave, comp


def _jax_seven(be, weights, log_z):
    ages = jnp.asarray(be._ssp_log_age_yr)
    raw = be._weighted_cue_params_raw(jnp.asarray(weights), ages, log_z)
    return np.array([float(raw[k]) for k in _IONSPEC_KEYS], dtype=np.float64)


# ── Matches the reference fit ─────────────────────────────────────────────


@pytest.mark.parametrize("log_z", [_LOG_Z_SOLAR, _LOG_Z_TENTH_SOLAR], ids=["solar", "tenth"])
@pytest.mark.parametrize("kind", ["constant", "burst_3myr", "burst_15myr", "exp_rise"])
def test_matches_reference_fit_of_composite(ssp_data_fsps, kind, log_z):
    """The unclipped JAX 7-vector equals fit_ionizing_spectrum on the composite, 1e-8."""
    be = _backend(ssp_data_fsps)
    w = _weights(ssp_data_fsps, kind)
    ref7, _, _, _ = _reference(ssp_data_fsps, w, log_z)
    np.testing.assert_allclose(_jax_seven(be, w, log_z), ref7, rtol=0, atol=_REF_TOL)


@pytest.mark.parametrize("kind", ["constant", "burst_15myr"])
def test_reference_is_the_converged_minimum(ssp_data_fsps, kind):
    """An independent BFGS polish from the reference answer agrees with it.

    fit_ionizing_spectrum finishes SLSQP with a converged Newton solve, so the BFGS
    polish (gtol 1e-12, its own objective evaluation) must land on the same point.
    """
    w = _weights(ssp_data_fsps, kind)
    _, ref_pl, wave, comp = _reference(ssp_data_fsps, w, _LOG_Z_SOLAR)
    polished = _polished_coeff(wave, comp, ref_pl)
    np.testing.assert_allclose(polished[:, 0], ref_pl[:, 0], rtol=0, atol=1e-6)
    np.testing.assert_allclose(polished[:, 1], ref_pl[:, 1], rtol=0, atol=1e-6)


@pytest.mark.parametrize("kind", ["constant", "burst_15myr"])
def test_jax_objective_is_at_least_as_converged(ssp_data_fsps, kind):
    """Per segment, the JAX (alpha, b) has objective <= the reference's, + 1e-9."""
    be = _backend(ssp_data_fsps)
    log_z = _LOG_Z_SOLAR
    w = _weights(ssp_data_fsps, kind)
    _, ref_pl, wave, comp = _reference(ssp_data_fsps, w, log_z)
    jax_coeff = np.asarray(be._fit_composite_shape(jnp.asarray(w), log_z, jnp.zeros(4))[1])
    for k, prob in enumerate(_segment_problems(wave, comp)):
        jax_p = np.array([jax_coeff[k, 0], jax_coeff[k, 1] + prob["log_norm"]])
        ref_p = np.array([ref_pl[k, 0], ref_pl[k, 1] + prob["log_norm"]])
        assert _objective(jax_p, prob) <= _objective(ref_p, prob) + 1e-9


# ── Single age ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("pick", ["middle", "quarter"])
def test_single_age_reproduces_table_row(ssp_data_fsps, pick):
    """A composite of one SSP is that SSP: its table row, to 1e-8."""
    ssp = ssp_data_fsps
    be = _backend(ssp)
    im = int(np.argmin(np.abs(np.asarray(ssp.ssp_lgmet) - _LOG_Z_SOLAR)))
    log_z = float(np.asarray(ssp.ssp_lgmet)[im])
    young = np.flatnonzero(_ages(ssp) <= _MAX_LOG_AGE_YR)
    age_index = int(young[len(young) // 2] if pick == "middle" else young[len(young) // 4])
    w = np.zeros(len(_ages(ssp)))
    w[age_index] = 1.0
    jax7 = _jax_seven(be, w, log_z)
    row = np.asarray(be._ionspec_table)[im, age_index]
    np.testing.assert_allclose(jax7, row, rtol=0, atol=_REF_TOL)


# ── Gradient ──────────────────────────────────────────────────────────────


def test_gradient_wrt_weights_matches_finite_differences(ssp_data_fsps):
    """d(scalar of the 7-vector)/d(w) is the implicit derivative, not a finite diff."""
    ssp = ssp_data_fsps
    be = _backend(ssp)
    log_z = _LOG_Z_SOLAR
    ages = jnp.asarray(_ages(ssp))
    c = jnp.arange(1.0, 8.0)

    def scalar(w):
        raw = be._weighted_cue_params_raw(w, ages, log_z)
        return jnp.sum(c * jnp.stack([raw[k] for k in _IONSPEC_KEYS]))

    f = jax.jit(scalar)
    f0 = float(f(jnp.asarray(_weights(ssp, "burst_15myr"))))
    g_fn = jax.jit(jax.grad(scalar))
    w0 = _weights(ssp, "burst_15myr")
    g = np.asarray(g_fn(jnp.asarray(w0)))
    assert np.all(np.isfinite(g))
    assert np.any(g != 0.0)
    # Bins with a tiny weight move the scalar below float64 resolution; sample the rest.
    active = np.flatnonzero(w0 > 1e-3 * w0.max())
    for i in active[:: max(1, len(active) // 12)]:
        # Relative step 1e-4: the Newton root is exact to ~1e-12 in gradient, so a
        # much smaller step would difference that residual, not the function.
        h = 1e-4 * w0[i]
        wp = w0.copy()
        wm = w0.copy()
        wp[i] += h
        wm[i] -= h
        fd = (float(f(jnp.asarray(wp))) - float(f(jnp.asarray(wm)))) / (2 * h)
        # Round-off in the scalar bounds what a difference quotient can resolve.
        resolution = 4 * np.finfo(float).eps * abs(f0) / (2 * h)
        np.testing.assert_allclose(g[i], fd, rtol=1e-5, atol=max(1e-12, resolution))


# ── Smoothness ────────────────────────────────────────────────────────────


def test_burst_age_sweep_is_smooth(ssp_data_fsps):
    """Sweeping a burst age finely gives no isolated jump.

    The slope index2 makes a steep S-shaped transition as the burst passes about
    15 Myr (a step of ~0.6 per 0.1 Myr), so the largest step is compared with its
    neighbors, not with the median: a jump is a step several times larger than
    both steps beside it. The converged minimum agrees with the fit along the sweep
    (checked with the reference objective, not asserted here).
    """
    ssp = ssp_data_fsps
    be = _backend(ssp)
    ages = _ages(ssp)
    t_myr = 10.0 ** (ages - 6.0)
    dt = np.gradient(10.0**ages)
    young = ages <= _MAX_LOG_AGE_YR
    t0_grid = np.linspace(2.0, 30.0, 150)
    ages_j = jnp.asarray(ages)
    jit_raw = jax.jit(lambda w: be._weighted_cue_params_raw(w, ages_j, _LOG_Z_SOLAR))
    path = []
    for t0 in t0_grid:
        w = np.where(young, np.exp(-0.5 * ((t_myr - t0) / 2.0) ** 2) * dt, 0.0)
        raw = jit_raw(jnp.asarray(w))
        path.append([float(raw[k]) for k in _IONSPEC_KEYS])
    path = np.asarray(path)
    steps = np.abs(np.diff(path, axis=0))
    assert np.all(np.isfinite(path))
    left = np.vstack([np.zeros((1, 7)), steps[:-1]])
    right = np.vstack([steps[1:], np.zeros((1, 7))])
    neighbor = np.maximum(left, right)
    assert np.all(steps <= 3.0 * neighbor + 1e-2)


# ── float32 ───────────────────────────────────────────────────────────────


def test_float32_matches_float64(ssp_data_fsps):
    """Under float32 the shape is finite and within 1e-3 of the float64 result."""
    ssp = ssp_data_fsps
    be64 = _backend(ssp)
    w64 = _weights(ssp, "burst_15myr")
    ref = _jax_seven(be64, w64, _LOG_Z_SOLAR)
    with jax.enable_x64(False):
        be32 = _backend(ssp)
        raw = be32._weighted_cue_params_raw(
            jnp.asarray(w64, dtype=jnp.float32),
            jnp.asarray(_ages(ssp), dtype=jnp.float32),
            jnp.float32(_LOG_Z_SOLAR),
        )
        out = np.array([float(raw[k]) for k in _IONSPEC_KEYS])
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out, ref, rtol=0, atol=1e-3)


# ── No young bins ─────────────────────────────────────────────────────────


def test_population_without_young_bins_is_finite_with_finite_gradient(ssp_data_fsps):
    """All weight older than 100 Myr: finite Q_H and shape, finite gradient.

    The composite is empty, so the fit runs on a flat floor and the output is taken
    from the dominant-bin fallback through ``jnp.where``. The gradient must stay
    finite through that branch (no NaN leaking from the unselected arm).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CueIonizingShapeWarning)
        be = CueBackend(_CUE_WEIGHTS, ssp_data=ssp_data_fsps, lyc_max_age_yr=1.0e8)
    ages = np.asarray(be._ssp_log_age_yr)
    w_old = jnp.asarray(np.where(ages > _MAX_LOG_AGE_YR, 1.0, 0.0))
    ages_j = jnp.asarray(ages)

    raw = be._weighted_cue_params_raw(w_old, ages_j, _LOG_Z_SOLAR)
    assert float(raw["gas_logqion"]) == -99.0
    shape = np.array([float(raw[k]) for k in _IONSPEC_KEYS])
    assert np.all(np.isfinite(shape))

    def scalar(w):
        out = be._weighted_cue_params_raw(w, ages_j, _LOG_Z_SOLAR)
        return jnp.sum(jnp.stack([out[k] for k in _IONSPEC_KEYS]))

    g = np.asarray(jax.grad(scalar)(w_old))
    # grad-assert: finite-only - with no young bin the weights do not reach the output,
    # so a zero gradient is the correct value; only NaN/inf would be a leak.
    assert np.all(np.isfinite(g))
