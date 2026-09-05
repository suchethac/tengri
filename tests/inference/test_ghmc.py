# SPDX-License-Identifier: BSD-3-Clause
import pytest

pytestmark = pytest.mark.contract

"""Tests for Generalized HMC (GHMC) sampler integration."""

import pytest

try:
    import blackjax  # noqa: F401

    HAS_BLACKJAX = True
except ImportError:
    HAS_BLACKJAX = False


@pytest.mark.skipif(not HAS_BLACKJAX, reason="blackjax not available")
class TestGHMCImports:
    def test_can_import_run_ghmc(self):
        from tengri.inference.backends.mcmc.ghmc import run_ghmc

        assert callable(run_ghmc)

    def test_can_import_posterior(self):
        from tengri.inference.posterior import Posterior

        assert Posterior is not None


@pytest.mark.skipif(not HAS_BLACKJAX, reason="blackjax not available")
class TestGHMCSignatures:
    def test_run_ghmc_signature(self):
        import inspect

        from tengri.inference.backends.mcmc.ghmc import run_ghmc

        sig = inspect.signature(run_ghmc)
        assert "context" in sig.parameters
        assert "key" in sig.parameters
        assert "n_warmup" in sig.parameters
        assert "n_burnin" in sig.parameters
        assert "n_samples" in sig.parameters
        assert "alpha" in sig.parameters
        assert "delta" in sig.parameters
        assert "verbose" in sig.parameters

    def test_run_ghmc_alpha_defaults_to_adapted(self):
        """``0.8`` was a guess; MEADS derives the damping from the ensemble.

        Was ``assert alpha_param.default == 0.8`` until the 2026-08-30 MEADS
        switch. Window adaptation cannot see ``alpha`` at all, so the one
        parameter that governs GHMC's
        mixing was the one nothing tuned -- the diagnosis behind ``tier='broken'``.
        ``None`` means "let ``blackjax.meads_adaptation`` decide"; a float still
        pins it, which is what the old default did unconditionally.
        """
        import inspect

        from tengri.inference.backends.mcmc.ghmc import run_ghmc

        sig = inspect.signature(run_ghmc)
        assert sig.parameters["alpha"].default is None

    def test_run_ghmc_delta_defaults_to_adapted(self):
        """MEADS Algorithm 3 sets ``delta = alpha / 2``; ``0.65`` was unrelated."""
        import inspect

        from tengri.inference.backends.mcmc.ghmc import run_ghmc

        sig = inspect.signature(run_ghmc)
        assert sig.parameters["delta"].default is None
