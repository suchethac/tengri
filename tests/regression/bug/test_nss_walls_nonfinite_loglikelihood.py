# SPDX-License-Identifier: BSD-3-Clause
"""Nested sampling walls non-finite log-likelihoods and refuses dead or unconverged runs.

A NaN log-likelihood handed to the nested sampler corrupts logZ without any
error, so ``run_nss`` maps non-finite values to -inf (the same wall the gradient
samplers use, #3135). A run whose initial live set is mostly non-finite has no
mass to integrate and raises ``DeadFitError``, and a run that stops at its
iteration cap before the evidence converged is refused by name, as the
hierarchical seam already does (#1429). Related open issues: #3162, #3174.

The toy problem has a 2-D unit-box prior and a Gaussian log-likelihood that is
NaN wherever ``x0 < 0.5``, so half the prior is non-finite. It runs through the
public ``run_nss`` driver with a minimal stand-in for the fitter.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.config.exceptions import DeadFitError
from tengri.inference.backends.evidence import run_nss

pytestmark = pytest.mark.regression_bug

_NAMES = ("x0", "x1")
_N_LIVE = 50
_NUM_DELETE = 10


class _ToyModel:
    """Weak-referenceable stand-in for the SEDModel the nested-sampler cache keys on."""


class _ToySpec:
    """Prior sampler and stochastic flag the driver reads from the spec."""

    stochastic = False

    def sample_batch(self, key, n):
        """Draw ``n`` points uniformly from the unit box."""
        keys = jax.random.split(key, len(_NAMES))
        return {name: jax.random.uniform(keys[i], (n,)) for i, name in enumerate(_NAMES)}


class _ToyFitter:
    """Duck-typed fitter exposing only what ``run_nss`` and ``_get_nss_fns`` read."""

    def __init__(self, loglikelihood_fn):
        self.model = _ToyModel()
        self.spec = _ToySpec()
        self._free_names = list(_NAMES)
        self._fixed_values = {}
        self._data_args = {}
        self._loglikelihood_fn = loglikelihood_fn

    def _engine_cache_key(self):
        """Return a key that is unique per toy likelihood."""
        return ("toy_nss", id(self._loglikelihood_fn))

    def _build_logprior_fn(self):
        """Return the uniform unit-box log-prior."""

        def logprior(params):
            return sum(jax.scipy.stats.uniform.logpdf(params[k], 0.0, 1.0) for k in _NAMES)

        return logprior

    def _get_or_build_loglikelihood_fn(self):
        """Return the toy log-likelihood ``(params, data_args) -> scalar``."""
        return self._loglikelihood_fn


def _gaussian_loglik(params, data_args):
    """Finite Gaussian log-likelihood centered inside the prior."""
    return -0.5 * ((params["x0"] - 0.75) ** 2 + (params["x1"] - 0.5) ** 2) / 0.05**2


def _half_nan_loglik(params, data_args):
    """Gaussian log-likelihood that is NaN wherever ``x0 < 0.5``."""
    return jnp.where(params["x0"] < 0.5, jnp.nan, _gaussian_loglik(params, data_args))


def _all_nan_loglik(params, data_args):
    """Log-likelihood that is NaN everywhere on the prior."""
    return params["x0"] * jnp.nan


def _mostly_nan_loglik(params, data_args):
    """Gaussian log-likelihood that is finite only in a 5% slice, ``x0 > 0.95``."""
    return jnp.where(params["x0"] > 0.95, _gaussian_loglik(params, data_args), jnp.nan)


def _run(loglikelihood_fn, *, max_iterations=2000):
    """Run the nested sampler on the toy problem through the public driver."""
    return run_nss(
        _ToyFitter(loglikelihood_fn),
        key=jax.random.PRNGKey(0),
        n_live=_N_LIVE,
        num_delete=_NUM_DELETE,
        max_iterations=max_iterations,
        n_posterior_samples=200,
        verbose=False,
    )


def test_nonfinite_loglikelihood_is_walled_and_logz_stays_finite():
    """Half the prior returning NaN still gives a finite evidence and no NaN in the output."""
    posterior = _run(_half_nan_loglik)

    assert np.isfinite(posterior.log_evidence)
    assert np.isfinite(posterior.diagnostics["log_evidence_err"])
    assert not np.any(np.isnan(np.asarray(posterior.samples["x0"])))


@pytest.mark.parametrize("loglikelihood_fn", [_all_nan_loglik, _mostly_nan_loglik])
def test_dead_initial_live_set_raises_dead_fit_error(loglikelihood_fn):
    """Fewer than 10% finite initial live points raise DeadFitError with the counts."""
    with pytest.raises(DeadFitError, match=rf"of {_N_LIVE} initial live points have a finite"):
        _run(loglikelihood_fn, max_iterations=5)


def test_unconverged_run_at_iteration_cap_is_refused():
    """Stopping at max_iterations before the evidence converged raises instead of returning."""
    with pytest.raises(RuntimeError, match="nested sampling hit nss_max_iterations=1 "):
        _run(_half_nan_loglik, max_iterations=1)
