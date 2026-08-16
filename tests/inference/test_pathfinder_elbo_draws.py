# SPDX-License-Identifier: BSD-3-Clause
"""Regression: Pathfinder must not silently use BlackJAX's 200 ELBO draws (#1028).

Pathfinder draws two unrelated sets of samples:

* **posterior draws** -- one multivariate-normal sample each, essentially free;
* **ELBO draws** -- used to pick which Gaussian along the L-BFGS path is best.
  Each one is a full forward-model evaluation, and they are vmapped.

BlackJAX calls both ``num_samples``. ``vi.pathfinder.approximate`` defaults its ELBO
draws to 200, and neither ``run_pathfinder`` nor ``pathfinder_adaptation`` used to
set it -- so a 7-parameter photometry fit vmapped 200 SED evaluations and peaked at
**25.6 GB**, OOM-killing the slow test tier. With 25 draws (Stan's ``num_elbo_draws``)
the same fit peaks at 4.9 GB.

These tests use a toy Gaussian log-density: they assert the *plumbing*, not the
memory, so they are fast and cannot themselves OOM. The memory claim is recorded in
``run_pathfinder``'s docstring.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.regression_bug

blackjax = pytest.importorskip("blackjax")


def _spy_on_approximate(monkeypatch, seen):
    """Record the ``num_samples`` blackjax's ``approximate`` is actually called with.

    ``blackjax.pathfinder`` (a ``GeneratePathfinderAPI`` instance) and
    ``blackjax.vi.pathfinder`` (the module) expose the *same* function object but are
    different namespaces, so rebinding one does not affect the other. ``run_pathfinder``
    resolves the former, ``pathfinder_adaptation`` the latter — spy on both.
    """
    import blackjax
    from blackjax import vi

    original = vi.pathfinder.approximate
    assert blackjax.pathfinder.approximate is original, "namespaces diverged upstream"

    def spy(rng_key, logdensity_fn, initial_position, num_samples=200, **kwargs):
        seen.append(num_samples)
        return original(rng_key, logdensity_fn, initial_position, num_samples, **kwargs)

    monkeypatch.setattr(vi.pathfinder, "approximate", spy)
    monkeypatch.setattr(blackjax.pathfinder, "approximate", spy)


def test_blackjax_still_defaults_elbo_draws_to_200():
    """Pin the upstream default this workaround exists for.

    If BlackJAX ever lowers it (or exposes it through ``pathfinder_adaptation``),
    this fails and ``_bounded_pathfinder_elbo_draws`` can be deleted.
    """
    import inspect

    from blackjax import vi

    default = inspect.signature(vi.pathfinder.approximate).parameters["num_samples"].default
    assert default == 200, f"upstream default changed to {default} — revisit #1028"


def test_warmstart_caps_the_elbo_draws_blackjax_would_otherwise_default(monkeypatch):
    """``pathfinder_adaptation`` must see 25 ELBO draws, not 200.

    ``pathfinder_adaptation`` forwards ``**extra_parameters`` to the *sampler*, not to
    ``approximate``, so the only seam is the module attribute it resolves at call time.
    """
    from blackjax.adaptation.pathfinder_adaptation import pathfinder_adaptation

    from tengri.inference.backends.mcmc._shared import (
        _PATHFINDER_ELBO_DRAWS,
        _bounded_pathfinder_elbo_draws,
    )

    seen: list[int] = []
    _spy_on_approximate(monkeypatch, seen)

    def logdensity(x):
        return -0.5 * jnp.sum(x**2)

    # without the guard, blackjax takes its own default
    warmup = pathfinder_adaptation(blackjax.nuts, logdensity)
    warmup.run(jax.random.PRNGKey(0), jnp.zeros(3), num_steps=10)
    assert seen == [200], f"expected the unguarded default, saw {seen}"

    # with it, the ELBO draws are bounded
    seen.clear()
    with _bounded_pathfinder_elbo_draws():
        warmup = pathfinder_adaptation(blackjax.nuts, logdensity)
        warmup.run(jax.random.PRNGKey(0), jnp.zeros(3), num_steps=10)
    assert seen == [_PATHFINDER_ELBO_DRAWS], f"ELBO draws not capped, saw {seen}"


def test_bounded_elbo_draws_restores_the_original_on_exit(monkeypatch):
    """The rebind is scoped -- including when the body raises."""
    from blackjax import vi

    from tengri.inference.backends.mcmc._shared import _bounded_pathfinder_elbo_draws

    original = vi.pathfinder.approximate

    with _bounded_pathfinder_elbo_draws():
        assert vi.pathfinder.approximate is not original
    assert vi.pathfinder.approximate is original

    with pytest.raises(RuntimeError), _bounded_pathfinder_elbo_draws():
        raise RuntimeError("boom")
    assert vi.pathfinder.approximate is original, "rebind leaked after an exception"


def test_standalone_pathfinder_passes_its_elbo_draws_through(monkeypatch):
    """``run_pathfinder(n_elbo_draws=...)`` must reach ``approximate``, not be dropped."""
    import contextlib

    from tengri.inference.backends.pathfinder import run_pathfinder

    seen: list[int] = []
    _spy_on_approximate(monkeypatch, seen)

    def logdensity(x):
        return -0.5 * jnp.sum(x**2)

    # Only the call into ``approximate`` is under test; whatever ``run_pathfinder``
    # does afterwards needs a real model, so let it fail there.
    with contextlib.suppress(Exception):
        run_pathfinder(
            key=jax.random.PRNGKey(0),
            log_posterior_flat=logdensity,
            init_flat=jnp.zeros(3),
            unravel_fn=lambda v: {"x": v},
            to_physical_fn=lambda d: d,
            model=None,
            n_samples=10,
            maxiter=3,
            n_elbo_draws=7,
            verbose=False,
        )
    assert seen == [7], f"n_elbo_draws was dropped on the way to approximate, saw {seen}"
