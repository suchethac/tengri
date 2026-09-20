# SPDX-License-Identifier: BSD-3-Clause
"""Self-whitening backends refuse precondition=, raising before sampling (#2196).

Two whitenings compose catastrophically: the analytic metric whitens geometry
and a backend that learns its own metric (MCLMC, low-rank HMC) fits noise when
applied afterward, multiplying to produce the worst sampling configuration.
Measured twice: 472 divergences in low-rank+precond on a stochastic-field fit
(bench/reports/2026-09-06_low_rank_metric_d74.md), and flagged in MCLMC
(#1994, #2166).

The fix: backends that whiten internally declare self_whitening=True and raise
ValueError in check_capabilities when precondition= is also truthy, before any
sampling starts.
"""

from __future__ import annotations

import pytest

import tengri  # noqa: F401 (registers backends)
from tengri.inference._backend_registry import (
    all_backends,
    check_capabilities,
    get_backend,
)

pytestmark = pytest.mark.regression_bug


def _runner_self_whitens(entry) -> bool:
    """Check if runner internally applies diagonal_preconditioning or learns a metric.

    Enumerate by reading runner source: look for diagonal_preconditioning=True
    or calls to adaptation functions that learn a metric (window_adaptation_low_rank).
    """
    # For now, check the self_whitening flag; this will be set during implementation
    return getattr(entry, "self_whitening", False)


class TestSelfWhiteningBackendsDeclareIt:
    """Every backend whose runner passes diagonal_preconditioning=True declares self_whitening."""

    @pytest.mark.parametrize("entry", all_backends(), ids=lambda e: e.name)
    def test_every_self_whitening_backend_declares_it(self, entry):
        """Enumerate runners that whiten internally and verify declaration."""
        # Backends with diagonal_preconditioning=True (learned from code inspection):
        self_whitening_names = {"mcmc_mclmc", "mcmc_adjusted_mclmc", "mcmc_hmc_lowrank"}

        if entry.name in self_whitening_names:
            assert entry.self_whitening, (
                f"{entry.name!r} whitens internally but does not declare "
                f"self_whitening=True. This omission allows precondition= to "
                f"compose two whitenings, degrading sampling (#2196)."
            )


class TestSelfWhiteningAndPreconditionRaise:
    """Using precondition= with a self-whitening backend raises ValueError before sampling."""

    def test_mclmc_with_precondition_raises(self):
        """mcmc_mclmc internally whitens; precondition= composes two whitenings."""
        entry = get_backend("mcmc_mclmc")
        with pytest.raises(
            ValueError,
            match=r".*self.whitening.*precondition.*whiten.*#2196",
        ):
            check_capabilities(entry, {"precondition": True})

    def test_hmc_lowrank_with_precondition_raises(self):
        """mcmc_hmc_lowrank learns a metric from warmup; precondition= composes two."""
        entry = get_backend("mcmc_hmc_lowrank")
        with pytest.raises(
            ValueError,
            match=r".*self.whitening.*precondition.*whiten.*#2196",
        ):
            check_capabilities(entry, {"precondition": True})

    def test_self_whitening_without_precondition_allowed(self):
        """Self-whitening backends alone (no precondition=) are allowed."""
        entry = get_backend("mcmc_mclmc")
        # Must not raise
        check_capabilities(entry, {})

    def test_self_whitening_with_precondition_false_allowed(self):
        """precondition=False asks for the default behavior (no whitening)."""
        entry = get_backend("mcmc_mclmc")
        # Must not raise
        check_capabilities(entry, {"precondition": False})

    def test_non_self_whitening_with_precondition_allowed(self):
        """Non-self-whitening backends accept precondition= normally."""
        entry = get_backend("mcmc_nuts")
        # Must not raise
        check_capabilities(entry, {"precondition": True})
