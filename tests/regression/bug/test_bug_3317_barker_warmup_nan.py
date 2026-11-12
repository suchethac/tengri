# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for #3317: a Barker warm-up must not go NaN into a frozen chain.

On the photometry fixture of ``test_inference_cache_keys.py`` with ``PRNGKey(0)``, the
warm-up's second step proposes a position whose log density and gradient are NaN. The
kernel's acceptance is then NaN, which used to poison dual averaging and freeze every
draw at the post-warmup state. A non-finite acceptance is now a rejection, so the
adapted step size is finite and the chain moves.
"""

import numpy as np
import pytest
from jax import random

from tests.contract.test_inference_cache_keys import fitter_photometry

pytestmark = pytest.mark.regression_bug


@pytest.fixture(scope="module")
def barker_posterior():
    """Barker run on the issue fixture: 20 warm-up steps and 20 draws."""
    fitter = fitter_photometry()
    return fitter.run("mcmc_barker", key=random.PRNGKey(0), n_warmup=20, n_samples=20)


class TestBarkerWarmupFinite:
    """The adapted step size and acceptance are finite and the chain moves."""

    def test_step_size_is_finite(self, barker_posterior):
        """The adapted step size is a finite number, not NaN."""
        assert np.isfinite(barker_posterior.diagnostics["step_size"])

    def test_acceptance_strictly_inside_unit_interval(self, barker_posterior):
        """The mean acceptance is strictly between 0 and 1."""
        acceptance = barker_posterior.diagnostics["acceptance_rate"]
        assert 0.0 < acceptance < 1.0

    def test_every_parameter_has_multiple_unique_draws(self, barker_posterior):
        """Each free parameter takes more than one value over the 20 draws."""
        for name, values in barker_posterior.samples.items():
            assert len(np.unique(np.asarray(values))) > 1, name

    def test_no_non_finite_draws(self, barker_posterior):
        """No draw is NaN or infinite."""
        for name, values in barker_posterior.samples.items():
            assert np.all(np.isfinite(np.asarray(values))), name
