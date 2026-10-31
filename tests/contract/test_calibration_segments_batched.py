# SPDX-License-Identifier: BSD-3-Clause
"""Calibration segments and the fiber-fraction latent on the batched path (#3134).

The batched spectral marginal must reproduce the per-galaxy
:class:`CalibrationMarginalizedLikelihood` galaxy by galaxy, for galaxies with
different segment layouts and different ``cal_scale_prior`` centers. Padding, the
compile cache and float32 behave as they do on the plain calibration marginal.

Runs on the synthetic narrow SSP (no ``data/ssp_*.h5`` needed).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src import test_util as jtu

from tengri import DEFAULT, Fixed, Observation, SEDModel, Spectroscopy, Uniform
from tengri.inference.batched_spectro import (
    CAL_LOG_Q,
    _latent_spec,
    batched_log_likelihood,
    batched_neg_log_posterior,
)
from tengri.inference.likelihoods.marginalized import CalibrationMarginalizedLikelihood
from tengri.observation.batched import build_spectro_batches
from tests.contract.test_batched_cal_marginal import (
    _F32_TRUE_MASS,
    _mock_catalog,
    _record,
)
from tests.contract.test_batched_spectrum_forward import _banded, _grid

pytestmark = pytest.mark.contract

_MASS = "sfh_dpl_log_total_mass"
_WINDOW = (4600.0, 6000.0)
_PRIOR_CENTERS = (0.0, 0.08)
_SEGMENT_CUTS = (0.45, 0.6)
_MASSES = (10.02, 10.28)


def _segment_ids_for(n: int, cut: float) -> np.ndarray:
    ids = np.zeros(n, dtype=int)
    ids[int(cut * n) :] = 1
    return ids


@pytest.fixture(scope="module")
def template(synthetic_ssp):
    wave = _grid(300, 4000.0, 7000.0)
    spectroscopy = Spectroscopy(
        wave_obs=jnp.asarray(wave),
        resample="conserving",
        calibration_segments=(135, 165),
        cal_scale_prior=(0.0, 0.3),
        cal_scale_window=_WINDOW,
        **_banded(wave),
    )
    return SEDModel.build(
        ssp_data=synthetic_ssp,
        observation=Observation(spectroscopy=spectroscopy),
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT), _MASS: Uniform(8.0, 12.0)},
        neb={"type": "none"},
        redshift=Uniform(0.0, 0.5),
    )


@pytest.fixture(scope="module")
def mock(synthetic_ssp):
    return _mock_catalog(synthetic_ssp)[:2]


def _records(galaxies):
    return [
        _record(g)._replace(
            segment_ids=_segment_ids_for(g["wave"].size, cut), cal_scale_prior=(center, 0.3)
        )
        for g, cut, center in zip(galaxies, _SEGMENT_CUTS, _PRIOR_CENTERS)
    ]


def _per_galaxy_log_prob(g, ids, log_q, mass):
    n = g["wave"].size
    window = ((g["wave"] >= _WINDOW[0]) & (g["wave"] <= _WINDOW[1])).astype(float)
    model_flux = g["ref"].predict_spectrum({_MASS: mass, "redshift": g["z"]})[:n]
    lik = CalibrationMarginalizedLikelihood(
        jnp.asarray(g["flux"]),
        jnp.asarray(g["sigma"]),
        wavelength=jnp.asarray(g["wave"]),
        n_poly=2,
        prior_sigma=1.0,
        segments=jnp.asarray(ids),
        n_segments=2,
        q_window=jnp.asarray(window),
        has_q=True,
    )
    return lik.log_prob({"spec_fnu": model_flux}, {"cal_log_q": log_q})


def _loglik(template, spec):
    return batched_log_likelihood(template, spec, calibration_marginalize=True, cal_n_poly=2)


def test_batched_equals_per_galaxy_with_different_layouts_and_prior_centers(template, mock):
    records = _records(mock)
    [(spec, batch, _)] = build_spectro_batches(records, quantum=420, model=template)
    assert spec.n_segments == 2
    params = {_MASS: jnp.asarray(_MASSES), "cal_log_q": jnp.asarray([0.05, -0.1])}
    values = np.asarray(jax.jit(jax.vmap(_loglik(template, spec)))(params, batch))
    for i, g in enumerate(mock):
        ids = _segment_ids_for(g["wave"].size, _SEGMENT_CUTS[i])
        ref = _per_galaxy_log_prob(g, ids, params["cal_log_q"][i], params[_MASS][i])
        np.testing.assert_allclose(values[i], float(ref), rtol=1e-10)

    for spec2, batch2, index in build_spectro_batches(records, quantum=64, model=template):
        sel = np.asarray(index)
        sub = {k: v[sel] for k, v in params.items()}
        values2 = np.asarray(jax.vmap(_loglik(template, spec2))(sub, batch2))
        np.testing.assert_allclose(values2, values[sel], rtol=1e-10)


def test_one_compile_per_spec_and_per_galaxy_prior_enters_the_posterior(template, mock):
    [(spec, batch, _)] = build_spectro_batches(_records(mock), quantum=420, model=template)
    nlp = jax.jit(
        jax.vmap(
            batched_neg_log_posterior(template, spec, calibration_marginalize=True, cal_n_poly=2)
        )
    )
    latent = _latent_spec(template)
    q_dist, mass_dist = latent.get_distribution(CAL_LOG_Q), latent.get_distribution(_MASS)

    def latents(log_q):
        return {
            _MASS: mass_dist.standardize(jnp.asarray(_MASSES)),
            CAL_LOG_Q: q_dist.standardize(jnp.asarray(log_q)),
        }

    base = np.asarray(nlp(latents([0.05, -0.1]), batch))
    with jtu.count_jit_compilation_cache_miss() as count:
        shifted = np.asarray(nlp(latents([0.15, 0.0]), batch))
    assert count() == 0
    for i, g in enumerate(mock):
        ids = _segment_ids_for(g["wave"].size, _SEGMENT_CUTS[i])
        ln_c, ln_s = math.log(10.0) * _PRIOR_CENTERS[i], math.log(10.0) * 0.3

        def log_post(lq, g=g, ids=ids, i=i, ln_c=ln_c, ln_s=ln_s):
            like = float(_per_galaxy_log_prob(g, ids, jnp.asarray(lq), jnp.asarray(_MASSES[i])))
            return like - 0.5 * ((lq - ln_c) / ln_s) ** 2

        first, second = [(0.05, 0.15), (-0.1, 0.0)][i]
        np.testing.assert_allclose(
            base[i] - shifted[i], log_post(second) - log_post(first), rtol=1e-9
        )


def test_float32_value_and_gradient_with_log_q_are_finite_and_nonzero(synthetic_ssp, template):
    galaxies = _mock_catalog(synthetic_ssp, true_masses=_F32_TRUE_MASS)[:2]
    [(spec, batch, _)] = build_spectro_batches(_records(galaxies), quantum=420, model=template)
    with jax.enable_x64(False):
        batch32 = jax.tree_util.tree_map(
            lambda a: (
                jnp.asarray(a, dtype=jnp.float32) if jnp.issubdtype(a.dtype, jnp.floating) else a
            ),
            batch,
        )
        params = {
            _MASS: jnp.asarray([4.05, 4.35], dtype=jnp.float32),
            "cal_log_q": jnp.asarray([0.02, -0.03], dtype=jnp.float32),
        }
        values, grads = jax.vmap(jax.value_and_grad(_loglik(template, spec)))(params, batch32)
        # grad-assert: finite-only — the value is a log-likelihood, not a gradient
        assert np.all(np.isfinite(np.asarray(values)))
        for name in (_MASS, "cal_log_q"):
            assert np.all(np.isfinite(np.asarray(grads[name])))
            assert np.all(np.asarray(grads[name]) != 0.0)


def test_per_galaxy_fitter_declares_log_q_and_scores_the_same_marginal(template, mock):
    import tengri
    from tengri.inference.context import InferenceContext

    g = mock[0]
    fitter = tengri.Fitter(
        template,
        data=jnp.asarray(g["flux"]),
        noise=jnp.asarray(g["sigma"]),
        data_type="spectroscopy",
        calibration_marginalize=True,
        cal_n_poly=2,
    )
    context = InferenceContext.from_target(fitter)
    assert CAL_LOG_Q in context.free_names

    physical = {n: jnp.asarray(0.0) for n in context.free_names}
    physical.update(
        {_MASS: jnp.asarray(10.02), "redshift": jnp.asarray(g["z"]), CAL_LOG_Q: jnp.asarray(0.05)}
    )
    unbounded = {
        n: fitter.spec.get_distribution(n).standardize(physical[n]) for n in context.free_names
    }
    value = context.log_likelihood_fn(unbounded, context.data_args)
    ids = np.repeat([0, 1], [135, 165])
    reference = _per_galaxy_log_prob(g, ids, jnp.asarray(0.05), jnp.asarray(10.02))
    np.testing.assert_allclose(float(value), float(reference), rtol=1e-8)
