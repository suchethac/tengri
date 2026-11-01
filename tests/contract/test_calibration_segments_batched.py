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


# ── (e2) the joint with lines, segments and the ln q latent ─────────────────

_J_WAVE = np.linspace(4000.0, 7500.0, 160)
_J_R = 300.0
_J_SIGMA_KMS = 120.0
_J_PRIOR_SIGMA = 1.0e-15
_J_Z = (0.1, 0.145)
_J_CUTS = (0.4, 0.65)
_J_CENTERS = (0.0, 0.08)
_J_SIGMA_DEX = 0.3
_J_WINDOW = (4600.0, 6000.0)
_J_LOG_Q = (0.05, -0.1)
_J_LINES = ("Hbeta", "Halpha")


def _joint_template(ssp, cal_prior):
    from tengri.observation.line_list import LineList

    n = _J_WAVE.size
    spectroscopy = Spectroscopy(
        wave_obs=jnp.asarray(_J_WAVE),
        resolution=_J_R,
        resample="conserving",
        eline_mode="marginalized",
        eline_catalog=LineList.from_names(list(_J_LINES)),
        eline_prior_sigma=_J_PRIOR_SIGMA,
        eline_prior_type="flat",
        eline_broad=False,
        calibration_segments=(int(_J_CUTS[0] * n), n - int(_J_CUTS[0] * n)),
        cal_prior=cal_prior,
        cal_scale_prior=(0.0, _J_SIGMA_DEX),
        cal_scale_window=_J_WINDOW,
    )
    model = SEDModel.build(
        ssp_data=ssp,
        observation=Observation(spectroscopy=spectroscopy),
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT), _MASS: Uniform(8.0, 12.0)},
        neb={"type": "none"},
        redshift=Uniform(0.0, 0.5),
    )
    spec = model.spec.with_fixed_value("eline_sigma_kms", float(_J_SIGMA_KMS))
    return SEDModel(spec, model.ssp_data, observation=model.observation)


def _joint_line_design(model, z):
    from tengri.observation.eline_marginalization import (
        apply_doublet_constraints,
        build_eline_design_matrix,
    )

    catalog = model.observation.spectroscopy.effective_catalog
    g = build_eline_design_matrix(
        jnp.asarray(_J_WAVE), jnp.asarray(catalog.wavelengths), _J_R, z, _J_SIGMA_KMS, 0.0
    )
    return apply_doublet_constraints(g, catalog.build_constraint_matrix())


@pytest.mark.parametrize("cal_prior", ["gaussian", "scale_free"])
def test_batched_joint_with_segments_lines_and_log_q_equals_per_galaxy(synthetic_ssp, cal_prior):
    from tengri.inference.batched_spectro import _log_q_log_normal
    from tengri.inference.likelihoods.marginalized import CalibrationELineMarginalizedLikelihood
    from tengri.observation.batched import GalaxySpectrum

    model = _joint_template(synthetic_ssp, cal_prior)
    n = _J_WAVE.size
    rng = np.random.default_rng(3)
    flux = rng.normal(1.3e-15, 0.1e-15, n)
    sigma = np.full(n, 0.1e-15)
    cuts = [int(c * n) for c in _J_CUTS]
    records = [
        GalaxySpectrum(
            wave=_J_WAVE,
            flux=flux,
            ivar=1.0 / sigma**2,
            z=z,
            resolution=_J_R,
            segment_ids=np.repeat([0, 1], [cut, n - cut]),
            cal_scale_prior=(center, _J_SIGMA_DEX),
        )
        for z, cut, center in zip(_J_Z, cuts, _J_CENTERS)
    ]
    [(spec, batch, _)] = build_spectro_batches(records, model=model)
    assert not np.array_equal(np.asarray(batch.line_mask[0]), np.asarray(batch.line_mask[1]))

    masses = jnp.asarray([10.0, 10.1])
    log_q = jnp.asarray(_J_LOG_Q)
    loglik = batched_log_likelihood(
        model, spec, calibration_marginalize=True, cal_n_poly=2, cal_prior_sigma=1.0
    )
    values = np.asarray(jax.vmap(loglik)({_MASS: masses, CAL_LOG_Q: log_q}, batch))
    window = ((_J_WINDOW[0] <= _J_WAVE) & (_J_WINDOW[1] >= _J_WAVE)).astype(float)
    for i, (z, cut) in enumerate(zip(_J_Z, cuts)):
        on_grid = np.asarray(batch.line_mask[i]) > 0
        design = _joint_line_design(model, z)[:, on_grid]
        reference = CalibrationELineMarginalizedLikelihood(
            fnu_obs=jnp.asarray(flux),
            fnu_err=jnp.asarray(sigma),
            wavelength=jnp.asarray(_J_WAVE),
            design_matrix_builder=lambda _p, design=design: design,
            n_poly=2,
            prior_sigma=1.0,
            eline_prior_type="flat",
            eline_prior_sigma=_J_PRIOR_SIGMA,
            segments=jnp.asarray(np.repeat([0, 1], [cut, n - cut])),
            n_segments=2,
            cal_prior=cal_prior,
            q_window=jnp.asarray(window),
            has_q=True,
        )
        model_flux = model.predict_spectrum({_MASS: masses[i], "redshift": z, CAL_LOG_Q: log_q[i]})
        expected = reference.log_prob({"spec_fnu": model_flux}, params={CAL_LOG_Q: log_q[i]})
        np.testing.assert_allclose(values[i], float(expected), rtol=1e-10)

        per_galaxy = Spectroscopy(
            wave_obs=jnp.asarray(_J_WAVE),
            cal_scale_prior=(_J_CENTERS[i], _J_SIGMA_DEX),
            cal_scale_window=_J_WINDOW,
        ).get_calibration_params()[CAL_LOG_Q]
        np.testing.assert_allclose(
            float(_log_q_log_normal(batch.galaxy(i), log_q[i])),
            float(per_galaxy.log_prob(log_q[i])),
            rtol=1e-9,
        )
