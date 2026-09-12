# SPDX-License-Identifier: BSD-3-Clause
"""Regression: NUTS ``dense_mass_matrix=None`` auto-policy (#319).

Revised 2026-09-11: the D < 8 cliff measured photometry fits with
``mean_sfh_type="dense_basis"`` and generalized the 20+ GB warmup finding to
every SFH at D >= 8. Measured on ``ctl-dpl`` (D=8 photometry, 14 bands, a
DPL, non-``dense_basis`` SFH): the dense window adaptation uses 1.1-1.9 GB
RSS and costs 3.4x fewer gradients per effective sample than diagonal (dense
34 g/draw, ESS 83; diagonal 550 g/draw, ESS 113). The policy is now dense at
D <= 12 unless the spec's SFH is ``dense_basis`` (diagonal at any D in that
case, since it is ``dense_basis``'s per-sample derived-quantity publishing,
not dimensionality, that drives the 22.78 GB spike); diagonal above D = 12
regardless of SFH.

The heuristic is pulled into :func:`_resolve_dense_mass_matrix` so the
policy is unit-testable without spinning up an actual NUTS warmup. See
``tests/inference/test_dense_auto_policy.py`` for the ``dense_basis``
exception and the D=12 boundary in detail.
"""

from __future__ import annotations

import inspect

import pytest

from tengri.inference.backends.mcmc.nuts import _resolve_dense_mass_matrix, run_nuts

pytestmark = pytest.mark.regression_bug


def test_auto_policy_dense_at_and_below_twelve():
    """At D <= 12, with no ``dense_basis`` spec, the auto-policy is dense."""
    for n_dim in (1, 2, 5, 7, 8, 9, 12):
        assert _resolve_dense_mass_matrix(None, n_dim) is True, (
            f"D={n_dim}: expected dense, got diagonal"
        )


def test_auto_policy_diagonal_above_twelve():
    """Above D = 12 the auto-policy returns ``False`` (diagonal)."""
    for n_dim in (13, 20, 50, 137):
        assert _resolve_dense_mass_matrix(None, n_dim) is False, (
            f"D={n_dim}: expected diagonal, got dense"
        )


def test_explicit_true_honored_at_high_dim():
    """User passing ``dense_mass_matrix=True`` at D > 12 must be respected.
    The warning fires (locked by a separate test) but the value goes
    through unchanged."""
    for n_dim in (4, 8, 20):
        assert _resolve_dense_mass_matrix(True, n_dim) is True


def test_explicit_false_honored_at_low_dim():
    """User passing ``dense_mass_matrix=False`` at D <= 12 must be respected
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
