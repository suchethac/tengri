# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for #2619 -- the calibration floor entered upper/lower limits.

The calibration floor ``noise_frac_cal * |model|`` inflates the error of a
measurement. The CIGALE implementation (``pcigale`` ``_add_model_error``) adds it
in quadrature to detections only; a limit keeps its own ``sigma_obs`` and is
scored with the Gaussian CDF at that sigma (Boquien et al. 2019, Eq. 15).
``censored_neg_log_likelihood`` computed
``sigma_eff = hypot(sigma_obs, f * |m|)`` once and used it in the limit branches
as well, so a free ``noise_frac_cal`` could absorb a limit violated at 3 sigma.

Every expected value below is written out in this file with ``scipy.stats`` or
``numpy``; nothing is read back from tengri.

Cells
-----
a  upper-limit term equals ``-2 ln Phi((U - m)/sigma_obs)`` for every f (27 ids)
b  detection term ``0.5 sum r^2 + sum ln sigma_eff`` (control, unchanged)
c  lower-limit term, mirrored statement (27 ids)
d  a free ``noise_frac_cal`` stays at 0 with one violated limit (S/N 3, 10, 50)
e  ``variable_noise_hamiltonian`` has no limit branch: detection-term control,
   Gaussian and Student-t; plus the Student-t detections of the censored
   function scored beside limits (limit term independent of ``dof``)
f  public path: ``SEDModel`` + ``Fitter(data_mask=...)`` log likelihood; the
   f-dependence is the detection ``ln sigma_eff`` terms only, float32 agrees
   with float64, ``jax.grad`` w.r.t. ``noise_frac_cal`` is the analytic value
g  detections and both kinds of limit in one call
h  limit gradient is the Mills-ratio value, independent of f; float32 at z = -30, -300
i  a zero sigma on a limit band is refused by the Fitter before the likelihood

https://github.com/suchethac/tengri/issues/2619
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

import tengri
from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel, Uniform, parse_groups
from tengri.inference.fitter import Fitter
from tengri.inference.likelihoods import CensoredLikelihood
from tengri.inference.loss_functions import _build_prediction, build_loglikelihood_fn
from tengri.observation.noise import (
    DETECTED,
    LOWER_LIMIT,
    UPPER_LIMIT,
    censored_neg_log_likelihood,
    variable_noise_hamiltonian,
)

pytestmark = pytest.mark.regression_bug

_SNR = (3, 10, 50)
_FCAL = (0.0, 0.05, 0.1)
_K = (-3.0, 0.0, 3.0)


def _hypot_sigma(sigma, model, f):
    """sigma_eff = sqrt(sigma_obs^2 + (f |m|)^2), written out in numpy."""
    return np.sqrt(np.asarray(sigma) ** 2 + (f * np.abs(np.asarray(model))) ** 2)


def _detection_energy(data, sigma, model, f):
    """0.5 sum r^2 + sum ln sigma_eff for detected bands."""
    s = _hypot_sigma(sigma, model, f)
    r = (np.asarray(data) - np.asarray(model)) / s
    return 0.5 * np.sum(r**2) + np.sum(np.log(s))


@pytest.mark.parametrize("k", _K)
@pytest.mark.parametrize("f", _FCAL)
@pytest.mark.parametrize("snr", _SNR)
def test_upper_limit_term_is_scored_at_sigma_obs(snr, f, k):
    """2E of one upper limit is -2 ln Phi((U - m)/sigma_obs) for every f."""
    m, sg = 1.0, 1.0 / snr
    u = m + k * sg
    e = censored_neg_log_likelihood(
        jnp.array([u]),
        jnp.array([sg]),
        jnp.array([m]),
        jnp.array([UPPER_LIMIT]),
        f_cal=f,
    )
    assert e.dtype == jnp.float64
    expected = -2.0 * stats.norm.logcdf((u - m) / sg)
    np.testing.assert_allclose(2.0 * float(e), expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("snr", _SNR)
@pytest.mark.parametrize("f", _FCAL)
def test_detection_term_is_unchanged(f, snr):
    """Control: detections carry 0.5 sum r^2 + sum ln hypot(sigma_obs, f |m|)."""
    m = np.array([1.0, 1.5, 2.0, 1.2, 0.6])
    eps = np.array([0.5, -1.0, 0.3, 1.2, -0.4])
    sg = m / snr
    d = m + eps * sg
    e = censored_neg_log_likelihood(
        jnp.array(d), jnp.array(sg), jnp.array(m), jnp.zeros(5, dtype=int), f_cal=f
    )
    np.testing.assert_allclose(float(e), _detection_energy(d, sg, m, f), rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("k", _K)
@pytest.mark.parametrize("f", _FCAL)
@pytest.mark.parametrize("snr", _SNR)
def test_lower_limit_term_is_scored_at_sigma_obs(snr, f, k):
    """2E of one lower limit is -2 ln Phi((m - L)/sigma_obs) for every f."""
    m, sg = 1.0, 1.0 / snr
    lo = m - k * sg
    e = censored_neg_log_likelihood(
        jnp.array([lo]),
        jnp.array([sg]),
        jnp.array([m]),
        jnp.array([LOWER_LIMIT]),
        f_cal=f,
    )
    expected = -2.0 * stats.norm.logcdf((m - lo) / sg)
    np.testing.assert_allclose(2.0 * float(e), expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("snr", _SNR)
def test_free_noise_frac_cal_is_not_pulled_up_by_a_violated_limit(snr):
    """Five exactly-fitted detections plus one upper limit violated at 3 sigma.

    With the floor in the limit term the argmin over f was 0.446 / 0.134 / 0.027
    at S/N 3 / 10 / 50; the limit at sigma_obs gives no reason to leave f = 0.
    """
    m = np.array([1.0, 1.5, 2.0, 1.2, 0.6])
    sg = m / snr
    sl = 1.0 / snr
    data = jnp.array(np.concatenate([m, [1.0 - 3.0 * sl]]))  # detections fit exactly
    noise = jnp.array(np.concatenate([sg, [sl]]))
    pred = jnp.array(np.concatenate([m, [1.0]]))
    mask = jnp.array([DETECTED] * 5 + [UPPER_LIMIT])
    fgrid = np.linspace(0.0, 0.5, 2001)
    energy = np.asarray(
        jax.vmap(lambda f: censored_neg_log_likelihood(data, noise, pred, mask, f_cal=f))(
            jnp.asarray(fgrid)
        )
    )
    assert abs(fgrid[np.argmin(energy)]) < 1e-3, (
        f"S/N {snr}: argmin f = {fgrid[np.argmin(energy)]:.4f}, expected 0.000"
    )


@pytest.mark.parametrize("dof", [None, 2.0, 5.0])
@pytest.mark.parametrize("f", [0.0, 0.1])
def test_variable_noise_hamiltonian_detection_term_control(f, dof):
    """``variable_noise_hamiltonian`` has no mask or limit branch (detections only).

    Control for the sibling path that shares ``compute_effective_noise``: its
    energy is the detection energy. Gaussian: 0.5 sum r^2 + sum ln sigma_eff.
    Student-t: -sum ln[t_pdf(r; dof)/sigma_eff] - 0.5 n ln(2 pi) (the
    Gaussian-normalized convention, which tends to the Gaussian energy as
    dof -> infinity).
    """
    m = np.array([1.0, 2.0, 1.5, 0.8])
    sg = np.array([0.1, 0.2, 0.15, 0.08])
    d = m + np.array([0.7, -1.2, 0.3, 2.0]) * sg
    e = float(variable_noise_hamiltonian(jnp.array(d), jnp.array(sg), jnp.array(m), f, dof=dof))
    if dof is None:
        expected = _detection_energy(d, sg, m, f)
    else:
        s = _hypot_sigma(sg, m, f)
        r = (d - m) / s
        expected = -np.sum(stats.t.logpdf(r, dof) - np.log(s)) - 0.5 * len(d) * np.log(2.0 * np.pi)
    np.testing.assert_allclose(e, expected, rtol=1e-12)


@pytest.mark.parametrize("dof", [None, 2.0, 5.0])
@pytest.mark.parametrize("f", [0.0, 0.1])
def test_censored_limit_term_is_independent_of_f_and_dof(f, dof):
    """Student-t detections beside both limit kinds: limits stay at sigma_obs.

    The censored function adds no Student-t normalization constant, so its
    detection energy is 0.5 (nu+1) sum ln(1 + r^2/nu) + sum ln sigma_eff.
    """
    m = np.array([1.0, 2.0, 1.5, 0.8, 1.1])
    sg = np.array([0.1, 0.2, 0.15, 0.08, 0.11])
    d = np.array([1.07, 1.76, 1.545, 0.8 - 3.0 * 0.08, 1.1 + 3.0 * 0.11])
    mask = np.array([DETECTED, DETECTED, DETECTED, UPPER_LIMIT, LOWER_LIMIT])
    e = float(
        censored_neg_log_likelihood(
            jnp.array(d), jnp.array(sg), jnp.array(m), jnp.array(mask), f_cal=f, dof=dof
        )
    )
    det = mask == DETECTED
    s = _hypot_sigma(sg[det], m[det], f)
    r = (d[det] - m[det]) / s
    if dof is None:
        e_det = 0.5 * np.sum(r**2) + np.sum(np.log(s))
    else:
        e_det = 0.5 * (dof + 1.0) * np.sum(np.log1p(r**2 / dof)) + np.sum(np.log(s))
    e_upper = -stats.norm.logcdf((d[3] - m[3]) / sg[3])
    e_lower = -stats.norm.logcdf((m[4] - d[4]) / sg[4])
    np.testing.assert_allclose(e, e_det + e_upper + e_lower, rtol=1e-12)


@pytest.mark.parametrize("f", _FCAL)
@pytest.mark.parametrize("snr", _SNR)
def test_detections_and_both_limit_kinds_in_one_call(snr, f):
    """Total energy is the detection energy plus two limit terms at sigma_obs."""
    m = np.array([1.0, 1.5, 2.0, 1.2, 0.6])
    sg = m / snr
    d = m + np.array([0.5, -1.0, 0.3, 0.0, 0.0]) * sg
    d[3] = m[3] - 3.0 * sg[3]  # upper limit violated at 3 sigma
    d[4] = m[4] + 3.0 * sg[4]  # lower limit violated at 3 sigma
    mask = np.array([DETECTED, DETECTED, DETECTED, UPPER_LIMIT, LOWER_LIMIT])
    e = float(
        censored_neg_log_likelihood(
            jnp.array(d), jnp.array(sg), jnp.array(m), jnp.array(mask), f_cal=f
        )
    )
    expected = (
        _detection_energy(d[:3], sg[:3], m[:3], f)
        - stats.norm.logcdf((d[3] - m[3]) / sg[3])
        - stats.norm.logcdf((m[4] - d[4]) / sg[4])
    )
    np.testing.assert_allclose(e, expected, rtol=1e-12)


# --- cell f: the public path --------------------------------------------------

_LOG_MASS = 10.0


@pytest.fixture(scope="module")
def ssp():
    return tengri.load_ssp("fsps_prsc_miles_chabrier", download=False)


@pytest.fixture(scope="module")
def observation():
    return Observation(photometry=Photometry.from_names(["des_g", "des_r", "des_i"]))


_PUBLIC_SNR = 10.0
_LIMIT_FRACTION = 0.7  # limit = 0.7 m at S/N 10: violated at exactly 3 sigma


def _public_path(ssp, observation, x64):
    """Build SEDModel + Fitter(data_mask=...) in one precision; return the pieces.

    The detections (bands 0, 1) equal the model prediction; band 2 is an upper
    limit at ``0.7 m`` with ``sigma_obs = m / 10`` (violated at 3 sigma).
    """
    with jax.enable_x64(x64):
        spec = parse_groups(
            sfh={
                "type": "delayed",
                "all_params": Fixed(DEFAULT),
                "log_total_mass": Uniform(9.0, 11.0),
            },
            dust_attenuation={
                "type": "single_component",
                "law": "calzetti",
                "all_params": Fixed(DEFAULT),
            },
            neb={"type": "none"},
            redshift=Fixed(0.1),
        ).merge_observation_params(noise_frac_cal=Uniform(0.0, 0.3))
        model = SEDModel(spec, ssp, observation=observation)
        free = {
            "sfh_delayed_log_total_mass": jnp.asarray(_LOG_MASS),
            "noise_frac_cal": jnp.asarray(0.0),
        }
        mask = jnp.array([DETECTED, DETECTED, UPPER_LIMIT])
        # The fit's own forward (threaded orchestrator) is what the likelihood
        # compares against, so the data are taken from it, not from the eager
        # ``predict_photometry``.
        probe = model.predict_photometry(free)
        probe_fitter = Fitter(model, data=probe, noise=probe / _PUBLIC_SNR, data_mask=mask)
        fit_model = probe_fitter.model
        pred = _build_prediction(
            fit_model,
            {**free, **probe_fitter._fixed_values},
            "photometry",
            has_line_fluxes=False,
            has_indices=False,
            index_defs=None,
            data_args=probe_fitter._data_args,
            jit_inputs=probe_fitter._data_args["_jit_inputs"],
            threaded_impl=fit_model._get_or_build_predict_observables_jit(),
        )[1]
        noise = pred / _PUBLIC_SNR
        data = pred.at[2].set(_LIMIT_FRACTION * pred[2])
        fitter = Fitter(model, data=data, noise=noise, data_mask=mask)
        ll = build_loglikelihood_fn(fitter)

        def neg_ll(f):
            params = {"sfh_delayed_log_total_mass": jnp.asarray(_LOG_MASS), "noise_frac_cal": f}
            return -ll(params, fitter._data_args)

        return fitter, neg_ll, np.asarray(pred, dtype=np.float64)


def test_public_path_reaches_the_censored_likelihood(ssp, observation):
    """Fitter(data_mask=...) on photometry builds CensoredLikelihood with the free f."""
    fitter, _, _ = _public_path(ssp, observation, True)
    assert isinstance(fitter._user_likelihood, CensoredLikelihood)
    assert fitter._user_likelihood.f_cal_param == "noise_frac_cal"
    assert np.asarray(fitter._data_args["data_mask"]).tolist() == [0, 0, 1]


def test_public_path_f_dependence_is_detections_only(ssp, observation):
    """E(f=0.1) - E(f=0) equals the detection ln sigma_eff terms; the limit adds none."""
    _, neg_ll, pred = _public_path(ssp, observation, True)
    sg = pred / _PUBLIC_SNR
    f = 0.1
    e0, ef = float(neg_ll(0.0)), float(neg_ll(f))
    expected = np.sum(np.log(_hypot_sigma(sg[:2], pred[:2], f)) - np.log(sg[:2]))
    assert expected > 0.1  # non-vacuous: the detections do move
    np.testing.assert_allclose(ef - e0, expected, rtol=0, atol=1e-8)
    # The whole f = 0 energy: detection ln sigma_obs plus -ln Phi(-3) for the limit.
    np.testing.assert_allclose(
        e0, np.sum(np.log(sg[:2])) - stats.norm.logcdf(-3.0), rtol=0, atol=1e-8
    )


def test_public_path_float32_agrees_with_float64(ssp, observation):
    """float32 energies (and their f-difference) agree with float64 to 1e-4."""
    _, nll64, _ = _public_path(ssp, observation, True)
    _, nll32, _ = _public_path(ssp, observation, False)
    with jax.enable_x64(False):
        e32 = [nll32(jnp.asarray(f, dtype=jnp.float32)) for f in (0.0, 0.1)]
    assert all(v.dtype == jnp.float32 for v in e32), "precondition: genuinely float32"
    e64 = [float(nll64(f)) for f in (0.0, 0.1)]
    for a, b in zip(e32, e64):
        assert np.isfinite(float(a))
        np.testing.assert_allclose(float(a), b, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(float(e32[1] - e32[0]), e64[1] - e64[0], rtol=1e-4, atol=1e-4)


def test_public_path_gradient_wrt_noise_frac_cal(ssp, observation):
    """d(-ln L)/df is finite, non-zero and the analytic detection value.

    With the detections fitted exactly, dE/df = sum_det f m^2 / (sigma^2 + f^2 m^2).
    """
    _, neg_ll, pred = _public_path(ssp, observation, True)
    sg = pred / _PUBLIC_SNR
    f = 0.1
    g = float(jax.grad(neg_ll)(jnp.asarray(f)))
    assert np.isfinite(g) and g != 0.0
    expected = np.sum(f * pred[:2] ** 2 / (sg[:2] ** 2 + f**2 * pred[:2] ** 2))
    np.testing.assert_allclose(g, expected, rtol=1e-8)


# --- limit gradient and strongly violated limits -------------------------------


def _upper_limit_energy(m, u, sg, f):
    return censored_neg_log_likelihood(
        jnp.atleast_1d(u),
        jnp.atleast_1d(sg),
        jnp.atleast_1d(m),
        jnp.array([UPPER_LIMIT]),
        f_cal=f,
    )


@pytest.mark.parametrize("f", _FCAL)
def test_upper_limit_gradient_is_the_mills_ratio_and_independent_of_f(f):
    """dE/dm = phi(z) / (sigma Phi(z)) at z = -3, sigma = 0.1 (32.8310), for every f."""
    m, sg = 1.0, 0.1
    u = m - 3.0 * sg
    g = float(jax.grad(_upper_limit_energy)(jnp.asarray(m), u, sg, f))
    z = (u - m) / sg
    expected = stats.norm.pdf(z) / (sg * stats.norm.cdf(z))
    np.testing.assert_allclose(expected, 32.8310, rtol=1e-5)
    np.testing.assert_allclose(g, expected, rtol=1e-8)


@pytest.mark.parametrize("z", [-30.0, -300.0])
def test_strongly_violated_limit_is_finite_in_float32(z):
    """Energy and gradient stay finite in pure float32; the energy matches float64 to 1e-4."""
    m, sg, f = 1.0, 0.1, 0.1
    u = m + z * sg
    e64, g64 = jax.value_and_grad(_upper_limit_energy)(jnp.asarray(m), u, sg, f)
    with jax.enable_x64(False):
        e32, g32 = jax.value_and_grad(_upper_limit_energy)(
            jnp.asarray(m, dtype=jnp.float32),
            jnp.asarray(u, dtype=jnp.float32),
            jnp.asarray(sg, dtype=jnp.float32),
            jnp.asarray(f, dtype=jnp.float32),
        )
    assert e32.dtype == jnp.float32 and g32.dtype == jnp.float32
    assert np.isfinite(float(e32)) and float(e32) > 0.0, "float32 energy collapsed"
    assert np.isfinite(float(g32)) and float(g32) != 0.0, "float32 gradient collapsed"
    np.testing.assert_allclose(float(e64), -stats.norm.logcdf(z), rtol=1e-10)
    np.testing.assert_allclose(float(e32), float(e64), rtol=1e-4)
    assert float(g32) > 0.0 and float(g64) > 0.0


def test_zero_sigma_on_a_limit_band_is_refused_by_the_fitter(ssp, observation):
    """The Fitter's boundary check covers limit bands: sigma = 0 raises, naming the band."""
    model = SEDModel(
        parse_groups(
            sfh={"type": "delayed", "all_params": Fixed(DEFAULT)},
            dust_attenuation={
                "type": "single_component",
                "law": "calzetti",
                "all_params": Fixed(DEFAULT),
            },
            neb={"type": "none"},
            redshift=Fixed(0.1),
        ),
        ssp,
        observation=observation,
    )
    data = jnp.array([1.0e-28, 1.0e-28, 7.0e-29])
    noise = jnp.array([1.0e-29, 1.0e-29, 0.0])  # band 2 is the limit
    mask = jnp.array([DETECTED, DETECTED, UPPER_LIMIT])
    with pytest.raises(ValueError, match=r"Non-positive noise at index/indices \[2\]"):
        Fitter(model, data=data, noise=noise, data_mask=mask)
