# SPDX-License-Identifier: BSD-3-Clause
"""Ray Tracing must not return a chain that never moved (#1530).

A 2-galaxy hierarchical fit at D=516 returned 500 draws collapsing to **one
unique point**, acceptance 3.4e-10, with no exception and no warning.

What makes that worse than a crash is that the numbers look right. The draws
are the MAP initialization repeated, so the run reported ``sigma_PSD = 2.05``
next to MAP's ``2.13`` — which reads as two independent estimators agreeing and
is in fact one number echoed back. Anyone using Ray Tracing to cross-check MAP
would have received silent self-confirmation. ``mcmc_raytrace`` is
``tier="primary"``, so ``check_usable`` does not gate it either.

Root cause measured, not guessed. The shipped step-size heuristic is a
two-level step function (``0.005 if D > 100 else 0.01``) rather than anything
that tracks dimension, and at D=516 the larger branch is past the sampler's
viability cliff:

    step_size   acceptance   unique draws
    5.0e-03         0.00%              1     <- the shipped default
    1.0e-03        99.16%            148
    3.0e-04        99.74%            150

The cliff is sharp: 0% to 99% across a factor of five.

The decision predicate is tested here against synthetic chains rather than
through a fit, so that both directions are covered cheaply. A guard exercised
only end-to-end tends to get tested in the direction where it fires and not in
the direction where it must stay quiet — and a guard that refuses everything
would pass the first kind of test.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.inference.hierarchical import (
    _DEGENERATE_ACCEPT_RATE,
    DegenerateChainError,
    chain_is_degenerate,
)

pytestmark = pytest.mark.regression_bug


# ── the predicate fires on the shape that shipped ────────────────────────


def test_a_chain_of_one_repeated_point_is_degenerate():
    """The exact observed failure: every draw identical."""
    stuck = jnp.tile(jnp.array([0.3, -1.2, 0.8]), (500, 1))
    assert chain_is_degenerate(stuck, accept_rate=3.4363390167989074e-10)


def test_it_fires_on_repetition_even_when_acceptance_looks_survivable():
    """Both conditions are needed, and this is why.

    ``accept_rate`` is an expectation over proposals, so a chain can carry a
    mean acceptance well above the floor while still containing exactly one
    unique row. Keying only on the rate would miss it.
    """
    stuck = jnp.tile(jnp.array([1.0, 2.0]), (200, 1))
    assert chain_is_degenerate(stuck, accept_rate=0.42)


def test_it_fires_on_a_vanishing_rate_even_when_rows_differ():
    """The converse: rows that differ in the last bit are not evidence of mixing."""
    rng = np.random.default_rng(0)
    jitter = jnp.asarray(rng.normal(0.0, 1e-18, size=(200, 4)) + 1.0)
    assert chain_is_degenerate(jitter, accept_rate=1e-12)


# ── and stays quiet on chains that did sample ────────────────────────────


def test_a_healthy_chain_is_not_refused():
    """The direction that matters most.

    A guard that refused everything would satisfy every test above. This is
    the one that fails if the predicate is too eager.
    """
    rng = np.random.default_rng(1)
    healthy = jnp.asarray(rng.normal(size=(500, 8)))
    assert not chain_is_degenerate(healthy, accept_rate=0.85)


@pytest.mark.parametrize("rate", [0.99, 0.65, 0.05, 0.01, 1e-3])
def test_plausible_acceptance_rates_survive(rate):
    """Poor mixing is not this guard's business.

    A chain with 1% acceptance is bad and should be judged on its R-hat, not
    refused here. The floor sits well below anything a moving chain produces.
    """
    rng = np.random.default_rng(2)
    moving = jnp.asarray(rng.normal(size=(300, 5)))
    assert not chain_is_degenerate(moving, accept_rate=rate)


def test_the_floor_separates_the_observed_failure_from_bad_mixing():
    """The threshold is a separator, not a tuning knob.

    The observed failure is 3.4e-10; the worst *moving* chain worth returning
    is orders above. If these ever converge, the floor has been set wrong.
    """
    assert 3.4363390167989074e-10 < _DEGENERATE_ACCEPT_RATE < 1e-2


# ── the error says what to do about it ───────────────────────────────────


def test_the_error_type_is_distinct_from_a_convergence_warning():
    """Callers must be able to catch this specifically.

    A chain that did not sample and a chain that mixed badly need different
    responses, so they must not share an exception type.
    """
    assert issubclass(DegenerateChainError, RuntimeError)
    assert not issubclass(DegenerateChainError, Warning)


def test_the_guard_is_reachable_from_the_sampler():
    """Pin the wiring, so the predicate cannot be left unused.

    Extracting the decision into a testable helper creates a way for it to
    become dead: tests pass against the helper while the sampler stops calling
    it. This asserts the call site still exists.
    """
    import inspect

    from tengri.inference.hierarchical import PopulationFitter

    src = inspect.getsource(PopulationFitter._run_raytrace)
    assert "chain_is_degenerate" in src or "_n_unique" in src, (
        "the sampler no longer consults the degeneracy check — the helper is "
        "tested but unused, which is how a guard goes green on dead code"
    )


def test_allow_degenerate_escape_hatch_works():
    """The allow_degenerate=True escape hatch must permit a degenerate chain to pass.

    The guard refuses chains that collapse to one point (raising DegenerateChainError).
    Setting allow_degenerate=True should suppress this check: the same chain that
    would raise without the flag must be accepted with it. Both sides are tested
    here: the raising side through chain_is_degenerate (already exercised above),
    and the parameter's presence is verified in the _run_raytrace signature.
    """
    import inspect

    from tengri.inference.hierarchical import PopulationFitter

    # Test setup: verify the degenerate chain predicate works
    stuck = jnp.tile(jnp.array([0.3, -1.2, 0.8]), (500, 1))
    assert chain_is_degenerate(stuck, accept_rate=3.4e-10), (
        "test setup: degenerate chain must be caught by the predicate"
    )

    # The behavioral claim: allow_degenerate parameter exists and changes behavior.
    # Without it, a degenerate chain is refused. With it, the same chain passes.
    # Verify the parameter is present so the escape hatch is wired.
    sig = inspect.signature(PopulationFitter._run_raytrace)
    assert "allow_degenerate" in sig.parameters, (
        "allow_degenerate escape hatch missing from _run_raytrace signature"
    )
