# SPDX-License-Identifier: BSD-3-Clause
"""Fitter MAP ends at a stationary point where scipy stops on a crease (#3205).

On the galaxy-2 mock, scipy's L-BFGS-B stops at a crease of the objective with
``ABNORMAL`` and a large gradient, far above the smooth optimum. The Fitter must
not return that point: its Newton decrement is below the bar, its Hessian is
positive definite, and its negative log-posterior matches the batched optimum.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

from tengri import Fixed, Uniform
from tengri.inference.batched_spectro import fit_spectra_map_vmap
from tengri.inference.context import InferenceContext
from tengri.inference.fitter import Fitter
from tengri.inference.map_stationarity import MAP_DECREMENT_BAR, stationarity
from tengri.observation.batched import build_spectro_batches
from tests.contract.test_batched_map_lbfgs import _dpl_template
from tests.contract.test_batched_spectro_loglik import _mock_catalog, _record
from tests.contract.test_batched_spectrum_forward import _banded

pytestmark = pytest.mark.contract

_CREASE_GALAXY = 2
#: Agreement with the batched optimum in nats of the negative log-posterior.
_NLP_TOL = 1e-6


@pytest.fixture(scope="module")
def mock(synthetic_ssp):
    return _mock_catalog(synthetic_ssp)


@pytest.fixture(scope="module")
def batched_nlp(synthetic_ssp, mock):
    """Negative log-posterior at the batched MAP of the crease galaxy."""
    template = _dpl_template(
        synthetic_ssp, mock[0]["wave"], _banded(mock[0]["wave"]), redshift=Uniform(0.0, 0.5)
    )
    batches = build_spectro_batches([_record(g) for g in mock], quantum=420, model=template)
    result = fit_spectra_map_vmap(template, batches)
    return float(np.asarray(result["nlp"])[_CREASE_GALAXY])


def test_fitter_map_is_stationary_on_the_crease_galaxy(synthetic_ssp, mock, batched_nlp):
    g = mock[_CREASE_GALAXY]
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

    ctx = InferenceContext.from_target(fitter)
    flat, unravel = ravel_pytree(ctx.unbounded_from_posterior(post))

    def loss(vec):
        return ctx.neg_log_posterior_fn(unravel(vec), ctx.data_args)

    point = stationarity(loss, flat)
    assert point.positive_definite, "Fitter MAP Hessian is not positive definite"
    assert point.decrement < MAP_DECREMENT_BAR, point.decrement
    assert abs(point.nlp - batched_nlp) < _NLP_TOL, (point.nlp, batched_nlp)

    assert post.diagnostics["map_converged"] is True
    assert post.diagnostics["map_decrement"] < MAP_DECREMENT_BAR
    assert post.diagnostics["map_attempts"] >= 1
    assert isinstance(post.diagnostics["map_message"], str)
