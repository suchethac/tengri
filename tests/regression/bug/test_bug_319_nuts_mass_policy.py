# SPDX-License-Identifier: BSD-3-Clause
"""Regression: NUTS ``dense_mass_matrix=None`` auto-policy (#319).

Default ``dense_mass_matrix=None`` now resolves to ``False`` (diagonal)
at D >= 8, dodging the 20+ GB warmup spike documented in the issue for
photometry fits with ``mean_sfh_type="dense_basis"``. Below D = 8 the
dense matrix converges faster and stays under 10 GB, so the default
keeps it on.

The heuristic is pulled into :func:`_resolve_dense_mass_matrix` so the
policy is unit-testable without spinning up an actual NUTS warmup.
"""

from __future__ import annotations

import inspect

import pytest

from tengri.inference.backends.mcmc.nuts import _resolve_dense_mass_matrix, run_nuts

pytestmark = pytest.mark.regression_bug


def test_auto_policy_dense_below_threshold():
    """At D < 8 the auto-policy returns ``True`` (dense)."""
    for n_dim in (1, 2, 5, 7):
        assert _resolve_dense_mass_matrix(None, n_dim) is True, (
            f"D={n_dim}: expected dense, got diagonal"
        )


def test_auto_policy_diagonal_at_and_above_threshold():
    """At D >= 8 the auto-policy returns ``False`` (diagonal)."""
    for n_dim in (8, 9, 12, 20, 50, 137):
        assert _resolve_dense_mass_matrix(None, n_dim) is False, (
            f"D={n_dim}: expected diagonal, got dense"
        )


def test_explicit_true_honoured_at_high_dim():
    """User passing ``dense_mass_matrix=True`` at D >= 8 must be respected.
    The warning fires (locked by a separate test) but the value goes
    through unchanged."""
    for n_dim in (4, 8, 20):
        assert _resolve_dense_mass_matrix(True, n_dim) is True


def test_explicit_false_honoured_at_low_dim():
    """User passing ``dense_mass_matrix=False`` at D < 8 must be respected
    (e.g. user already has a converged init and wants diagonal for speed)."""
    for n_dim in (1, 5, 7):
        assert _resolve_dense_mass_matrix(False, n_dim) is False


def test_run_nuts_default_is_none():
    """``run_nuts`` signature defaults ``dense_mass_matrix=None`` so the
    auto-policy is the path users get by default. Pinning the signature
    locks the policy entry point."""
    sig = inspect.signature(run_nuts)
    param = sig.parameters["dense_mass_matrix"]
    assert param.default is None, (
        f"dense_mass_matrix default changed from None to {param.default!r} — "
        f"the #319 auto-policy entry point regressed."
    )
