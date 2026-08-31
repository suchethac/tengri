# SPDX-License-Identifier: BSD-3-Clause
"""#1029 -- the Pathfinder ELBO-draw cap did not cover BlackJAX's multi-path route.

``_bounded_pathfinder_elbo_draws`` exists because ``pathfinder_adaptation``
exposes no knob for the ELBO draws, and BlackJAX defaults them to **200** -- a
number that is a *memory* knob, since each draw is a full forward-model
evaluation vmapped across every L-BFGS iterate. #1029 measured that default at
**25.65 GB and a SIGKILL** on a 7-parameter photometry fit, which is why the cap
exists at all.

BlackJAX 1.6.2 grew a multi-path Pathfinder, and ``pathfinder_adaptation`` now
dispatches on ``(num_chains, effective_n_paths)`` where::

    effective_n_paths = n_paths if n_paths is not None else num_chains

* ``num_chains == 1 and effective_n_paths == 1`` (PATH A) calls
  ``vi.pathfinder.approximate(key, logdensity_fn, position)`` with no
  ``num_samples`` -- the case the cap was written for.
* ``effective_n_paths >= 2`` (PATH C) calls
  ``multipathfinder.multi_approximate(..., num_samples=num_samples_per_path)``,
  default 200, which ``jax.vmap``s ``approximate`` across every path. The
  exposure is ``n_paths x 200`` ELBO draws.

tengri's two call sites pass neither ``num_chains`` nor ``n_paths``, so both take
PATH A and the cap held -- **by coincidence of call shape, not by construction**.
``n_paths`` defaults to ``num_chains``, and the fixtures this library is measured
on run ``n_chains=2``, so forwarding the chain count into the warm-up is the
obvious next edit to those very lines.

PATH C defeated the cap **twice, independently**:

1. ``blackjax/vi/multipathfinder.py`` does
   ``from blackjax.vi.pathfinder import ... approximate ...`` at *import* time,
   so it holds its own module-global binding and ``multi_approximate`` calls that
   bare name. Rebinding ``vi.pathfinder.approximate`` never reached it.
2. ``multi_approximate`` forwards ``num_samples`` **positionally**, so a cap
   expressed as a parameter *default* is overridden even in the right namespace.

**One test per cause, deliberately.** A single assertion on the draw count that
finally reaches BlackJAX passes if *either* cause is fixed -- a check that
succeeds for two reasons, which is the same shape as the bug. Verified by
mutation: reverting the clamp to a default fails only
:func:`test_the_cap_clamps_a_positional_num_samples`; dropping ``multipathfinder``
from the patch list fails only
:func:`test_the_cap_reaches_the_multipathfinder_namespace`.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.regression_bug


def test_the_cap_reaches_the_multipathfinder_namespace():
    """CAUSE 1 alone: the import-time binding must be rebound, and restored.

    Asserts only the rebinding, never the resulting draw count, so it cannot pass
    for the other cause's reason.
    """
    pytest.importorskip("blackjax")
    multipathfinder = pytest.importorskip("blackjax.vi.multipathfinder")

    from tengri.inference.backends.mcmc._shared import _bounded_pathfinder_elbo_draws

    before = multipathfinder.approximate
    with _bounded_pathfinder_elbo_draws(7):
        during = multipathfinder.approximate
    after = multipathfinder.approximate

    assert during is not before, (
        "blackjax.vi.multipathfinder.approximate was NOT rebound inside the cap. "
        "It holds an import-time binding, so patching vi.pathfinder alone leaves "
        "the multi-path route (effective_n_paths >= 2) running at 200 ELBO draws "
        "per path -- the configuration #1029 measured at 25.65 GB and a SIGKILL."
    )
    assert after is before, "the cap must restore every namespace it patched"


def test_the_cap_clamps_a_positional_num_samples():
    """CAUSE 2 alone: a positional 200 must be clamped, not merely defaulted past.

    Exercises one namespace only, so it cannot pass for the other cause's reason.
    The third call pins that the clamp is a ``min`` and not a hard overwrite: a
    caller asking for fewer draws than the cap keeps their smaller number.
    """
    pytest.importorskip("blackjax")
    import blackjax.vi.pathfinder as pathfinder_module

    from tengri.inference.backends.mcmc._shared import _bounded_pathfinder_elbo_draws

    seen = []

    def _spy(rng_key, logdensity_fn, initial_position, num_samples=200, **kwargs):
        seen.append(num_samples)
        return "state", "info"

    original = pathfinder_module.approximate
    pathfinder_module.approximate = _spy
    try:
        with _bounded_pathfinder_elbo_draws(7):
            pathfinder_module.approximate(None, None, None, 200)  # positional, as upstream
            pathfinder_module.approximate(None, None, None)  # PATH A, no num_samples
            pathfinder_module.approximate(None, None, None, 3)  # caller wants fewer
    finally:
        pathfinder_module.approximate = original

    assert seen == [7, 7, 3], (
        f"ELBO draws reaching blackjax were {seen}, expected [7, 7, 3]: an explicit "
        "200 clamped down, an absent value capped, and a smaller request honored."
    )
