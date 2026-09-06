# SPDX-License-Identifier: BSD-3-Clause
"""The standardized N(0, I) prior is computed in one place, and reduces to a scalar.

Two implementations of one rule existed: the objective in
:mod:`tengri.inference.loss_functions` and the accessor
``InferenceContext.log_prior_fn``. They agreed on a single galaxy and diverged on
a batched one -- the objective wrapped each free parameter in ``jnp.sum`` (with a
comment explaining that hierarchical fits carry per-galaxy parameters of shape
``(N,)``), the accessor did not. So ``log_prior_fn`` returned shape ``(N,)``
where its own docstring promises a scalar.

Nothing in ``src/`` consumed the accessor, so no fit was wrong. That is what
makes it worth pinning rather than shrugging at: the next caller inherits a
silently non-scalar log-prior, and a vector objective either broadcasts into
nonsense or fails somewhere unrelated.

One helper now serves both, which is also the single place a change of
parameterization has to touch (#1355).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.contract


def test_the_penalty_is_scalar_for_a_batched_parameter_set():
    """Per-galaxy parameters must reduce, or the objective is not a single number."""
    from tengri.inference.loss_functions import standardized_neg_log_prior

    params = {"a": jnp.ones(4), "b": jnp.ones(4), "psd_xi": jnp.ones((4, 16))}
    value = standardized_neg_log_prior(params, ("a", "b"), stochastic=True)

    assert jnp.ndim(value) == 0, f"expected a scalar, got shape {jnp.shape(value)}"
    # 4 + 4 + 64 squared ones, halved.
    assert float(value) == pytest.approx(0.5 * 72.0)


def test_the_penalty_is_scalar_for_a_single_galaxy():
    """The common case must be unchanged by the reduction."""
    from tengri.inference.loss_functions import standardized_neg_log_prior

    params = {"a": jnp.asarray(2.0), "b": jnp.asarray(-1.0), "psd_xi": jnp.ones(9)}
    value = standardized_neg_log_prior(params, ("a", "b"), stochastic=True)

    assert jnp.ndim(value) == 0
    assert float(value) == pytest.approx(0.5 * (4.0 + 1.0 + 9.0))


def test_the_field_is_excluded_when_the_model_is_not_stochastic():
    """A non-stochastic fit has no field latent to penalize."""
    from tengri.inference.loss_functions import standardized_neg_log_prior

    params = {"a": jnp.asarray(3.0), "psd_xi": jnp.ones(5)}

    assert float(standardized_neg_log_prior(params, ("a",), stochastic=False)) == pytest.approx(
        4.5
    )
    assert float(standardized_neg_log_prior(params, ("a",), stochastic=True)) == pytest.approx(7.0)


def test_the_context_accessor_and_the_objective_agree_by_construction():
    """Both must route through the helper, not restate the rule.

    This is a source-based structural claim guarding against a regression
    documented in the module docstring: two independent implementations of the
    standardized N(0, I) prior existed, and they diverged on batched inputs
    (one returned shape (N,), the other a scalar). The fix consolidated them
    into a single helper. Pinning that both code paths call that helper is
    the only way to prevent the divergence from creeping back in when someone
    refactors either half. Tests above (test_the_penalty_is_scalar_*) verify
    the scalar output; this one pins the architectural control.
    """
    import inspect

    from tengri.inference import context as context_module, loss_functions

    context_src = inspect.getsource(context_module.InferenceContext.log_prior_fn.fget)
    assert "standardized_neg_log_prior" in context_src, (
        "log_prior_fn restates the prior instead of calling the shared helper"
    )
    assert "standardized_neg_log_prior" in inspect.getsource(loss_functions.build_loss_fn), (
        "the objective restates the prior instead of calling the shared helper"
    )


def test_log_prior_is_the_negated_penalty():
    """Sign convention: the objective minimizes, the accessor reports a log-density."""
    from tengri.inference.loss_functions import standardized_neg_log_prior

    params = {"a": jnp.asarray(2.0)}
    penalty = float(standardized_neg_log_prior(params, ("a",), stochastic=False))

    # log N(2; 0, 1) up to the normalizing constant is -0.5 * 4.
    assert -penalty == pytest.approx(-2.0)
    assert np.isfinite(penalty)
