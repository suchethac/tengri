# SPDX-License-Identifier: BSD-3-Clause
"""HMC must not run a dense mass matrix where NUTS refuses to (#1454, #1413).

``mcmc_hmc`` defaulted to ``dense_mass_matrix=True`` and gated it on
``n_dim <= 30``, so it allocated a dense O(D^2) mass matrix across the whole
D = 8-30 band — precisely the band where NUTS's auto-policy switches to
diagonal to dodge the documented 20+ GB warmup spike (#319). The shared high-D
advisory could not catch it either: it fires *above* D = 30, by which point HMC
has already fallen back to diagonal. The guard was inverted with respect to the
cost it guards.

Measured consequence: ``mcmc_hmc`` at D = 9 peaked at 13.47 GB and was
SIGKILLed while ``mcmc_nuts`` at the same D was already diagonal — and
CLAUDE.md recommends ``mcmc_hmc`` as the remedy for exactly that OOM, so the
method users were steered toward was the worse one in that band.

The advisory's method set is asserted directly rather than through a fit: a
memory-load test on a shared machine measures the machine, not the policy.
"""

import pytest

from tengri.inference._dimension_guard import NUTS_LIKE
from tengri.inference.backends.mcmc.nuts import _resolve_dense_mass_matrix

pytestmark = pytest.mark.regression_bug


@pytest.mark.parametrize("n_dim", [8, 9, 12, 20, 30])
def test_hmc_is_diagonal_wherever_nuts_is(n_dim):
    """The two samplers must agree on the dense/diagonal policy.

    They carry the same O(D^2) mass-matrix cost — fixed-length HMC differs from
    NUTS in trajectory length, not in mass-matrix adaptation — so a D at which
    one avoids dense and the other does not is a defect by construction.
    """
    nuts_dense = _resolve_dense_mass_matrix(None, n_dim)
    hmc_dense = _resolve_dense_mass_matrix(None, n_dim) and n_dim <= 30
    assert hmc_dense == nuts_dense, (
        f"at D={n_dim} NUTS uses dense={nuts_dense} but HMC uses "
        f"dense={hmc_dense}; HMC pays O(D^2) where NUTS declines to"
    )


@pytest.mark.parametrize("n_dim", [8, 9, 20])
def test_the_auto_policy_is_diagonal_at_and_above_eight(n_dim):
    """Pin the cliff itself, not just agreement between the two paths.

    Both could agree on 'dense everywhere' and satisfy the test above.
    """
    assert _resolve_dense_mass_matrix(None, n_dim) is False


@pytest.mark.parametrize("n_dim", [1, 5, 7])
def test_small_problems_keep_the_dense_matrix(n_dim):
    """Guard the fix against over-reaching: dense converges better below 8."""
    assert _resolve_dense_mass_matrix(None, n_dim) is True


def test_explicit_choices_still_round_trip():
    """``True``/``False`` are the user's call and must not be overridden."""
    assert _resolve_dense_mass_matrix(True, 50) is True
    assert _resolve_dense_mass_matrix(False, 2) is False


def test_hmc_defaults_to_the_auto_policy_not_true():
    """The signature default is the thing that regressed."""
    import inspect

    from tengri.inference.backends.mcmc.hmc import run_hmc

    default = inspect.signature(run_hmc).parameters["dense_mass_matrix"].default
    assert default is None, (
        f"run_hmc(dense_mass_matrix={default!r}) — a True default reinstates the "
        "dense matrix across D = 8-30, where NUTS deliberately uses diagonal"
    )


def test_the_advisory_covers_hmc():
    """``mcmc_hmc`` runs a NUTS-style warmup, so the advisory must reach it."""
    assert "mcmc_hmc" in NUTS_LIKE
    assert "mcmc_nuts" in NUTS_LIKE


def test_the_advisory_excludes_ghmc_because_ghmc_is_always_diagonal():
    """GHMC's ``dense_mass_matrix=True`` default is inert — do not warn on it.

    Its momentum generator treats ``momentum_inverse_scale`` as a diagonal
    vector, so ``ghmc.py`` pins ``adapt_key = ("ghmc", True)`` — always diagonal
    — regardless of the flag. Including it would advertise a cost the method
    never pays. This is the one place where reading the signature default
    rather than the code gives the wrong answer.

    What is asserted is the *literal* ``True``, not the whole key. The namespace
    prefix is orthogonal: it was ``"hmc"`` until #1442 renamed it, because sharing
    a prefix with ``hmc.py`` at the same arity made the two samplers collide in
    the warmup cache. Pinning the prefix here would make this test fail on a
    change it has no opinion about — the prefix is pinned on its own by
    ``test_each_mcmc_backend_owns_its_adaptation_cache_namespace``.
    """
    assert "mcmc_ghmc" not in NUTS_LIKE

    import inspect
    import re

    from tengri.inference.backends.mcmc.ghmc import run_ghmc

    source = inspect.getsource(run_ghmc)
    adapt_key = re.search(r"adapt_key\s*=\s*\((.*?)\)", source)
    assert adapt_key is not None, (
        "ghmc no longer sets an adapt_key — if it can now allocate a dense mass "
        "matrix, it belongs in NUTS_LIKE after all"
    )
    assert adapt_key.group(1).split(",")[-1].strip() == "True", (
        f"ghmc no longer pins a diagonal adapt_key (got ({adapt_key.group(1)})) — "
        "if it can now allocate a dense mass matrix, it belongs in NUTS_LIKE after all"
    )
