# SPDX-License-Identifier: BSD-3-Clause
"""Batched MAP runs L-BFGS to gradient tolerance, as the per-galaxy Fitter does (#3193).

The batched MAP must agree with the per-galaxy :class:`Fitter` MAP in every
standardized parameter, and both must be stationary by the same Newton-decrement
measure. The Adam driver is kept as ``optimizer="adam"`` and must reproduce the
previous driver at a fixed step count.

Runs on the synthetic narrow SSP (no ``data/ssp_*.h5`` needed).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

from tengri import FREE, Fixed, Observation, SEDModel, Spectroscopy, Uniform
from tengri.inference.batched_spectro import batched_neg_log_posterior, fit_spectra_map_vmap
from tengri.inference.fitter import Fitter
from tengri.observation.batched import build_spectro_batches
from tests.contract.test_batched_spectro_loglik import _MASS, _mock_catalog, _record
from tests.contract.test_batched_spectrum_forward import _banded
from tests.inference.test_profile_mass import _newton_decrement_and_sigma_xi

pytestmark = pytest.mark.contract

#: Newton-decrement bar in nats, the same build-independent stationarity measure as
#: the #3083 parity test in ``tests/inference/test_profile_mass.py``.
_DECREMENT_BAR = 1e-8
#: Agreement with the per-galaxy Fitter MAP in standardized units.
_STANDARDIZED_TOL = 1e-4

#: Adam driver on the three mocks, seed 0, 25 steps, recorded on the previous driver.
_ADAM25_MASS = np.array([9.991730669643628, 10.307567535163958, 9.473208382015576])
_ADAM25_NLP = np.array([311651.6955499946, 2650388.9314064053, 19696851.196080387])


def _dpl_template(ssp, wave, resolution, *, redshift):
    """Spectroscopic model with every DPL shape parameter free and the mass free."""
    spectroscopy = Spectroscopy(wave_obs=jnp.asarray(wave), resample="conserving", **resolution)
    return SEDModel.build(
        ssp_data=ssp,
        observation=Observation(spectroscopy=spectroscopy),
        sfh={"type": "dpl", "all_params": FREE, _MASS: Uniform(8.0, 12.0)},
        neb={"type": "none"},
        redshift=redshift,
    )


@pytest.fixture(scope="module")
def mock(synthetic_ssp):
    return _mock_catalog(synthetic_ssp)


@pytest.fixture(scope="module")
def template(synthetic_ssp, mock):
    return _dpl_template(
        synthetic_ssp, mock[0]["wave"], _banded(mock[0]["wave"]), redshift=Uniform(0.0, 0.5)
    )


@pytest.fixture(scope="module")
def batches(mock, template):
    return build_spectro_batches([_record(g) for g in mock], quantum=420, model=template)


def _batched_decrement(nlp, obs, xi: dict) -> float:
    """Newton decrement [nats] of one galaxy's negative log-posterior at standardized ``xi``."""
    flat, unravel = ravel_pytree(xi)

    def loss(vec):
        return nlp(unravel(vec), obs)

    grad = jax.jit(jax.grad(loss))(flat)
    hessian = jax.jit(jax.hessian(loss))(flat)
    eigenvalues, eigenvectors = jnp.linalg.eigh(hessian)
    assert bool(jnp.all(eigenvalues > 0)), "Hessian at the batched MAP is not positive definite"
    return 0.5 * float(jnp.sum((eigenvectors.T @ grad) ** 2 / eigenvalues))


def test_batched_map_matches_fitter_map_and_is_stationary(synthetic_ssp, template, mock, batches):
    [(spec, batch, index)] = batches
    np.testing.assert_array_equal(np.asarray(index), [0, 1, 2])
    result = fit_spectra_map_vmap(template, batches)
    nlp = batched_neg_log_posterior(template, spec)
    names = list(result["params"])
    assert _MASS in names and "sfh_dpl_alpha" in names

    for i, g in enumerate(mock):
        fixed = _dpl_template(synthetic_ssp, g["wave"], _banded(g["wave"]), redshift=Fixed(g["z"]))
        fitter = Fitter(
            fixed,
            data=jnp.asarray(g["flux"]),
            noise=jnp.asarray(g["sigma"]),
            data_type="spectroscopy",
            approx=None,
            profile_mass=False,
        )
        post = fitter.run("map")
        xi_batch, xi_fit = {}, {}
        for name in names:
            dist = template.spec.get_distribution(name)
            xi_batch[name] = dist.standardize(result["params"][name][i])
            xi_fit[name] = dist.standardize(jnp.asarray(post.params[name]))

        obs = batch.galaxy(i)
        decrement = _batched_decrement(nlp, obs, xi_batch)
        assert np.isfinite(decrement) and decrement < _DECREMENT_BAR, (i, decrement)
        fit_decrement, _ = _newton_decrement_and_sigma_xi(fitter, post)
        if fit_decrement < _DECREMENT_BAR:
            for name in names:
                gap = abs(float(xi_batch[name]) - float(xi_fit[name]))
                assert gap < _STANDARDIZED_TOL, (i, name, gap)
        else:
            # The Fitter's point is not stationary, so it is not a reference to match.
            # The batched point must then be at least as low in the negative log-posterior.
            assert float(nlp(xi_batch, obs)) <= float(nlp(xi_fit, obs)) + 1e-8, (i, fit_decrement)


def test_adam_option_reproduces_the_previous_driver(template, batches):
    result = fit_spectra_map_vmap(template, batches, optimizer="adam", n_steps=25)
    np.testing.assert_allclose(np.asarray(result["params"][_MASS]), _ADAM25_MASS, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(result["nlp"]), _ADAM25_NLP, rtol=1e-12)
