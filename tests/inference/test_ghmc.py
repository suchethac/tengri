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

    def test_run_ghmc_alpha_default_is_reasonable(self):
        import inspect

        from tengri.inference.backends.mcmc.ghmc import run_ghmc

        sig = inspect.signature(run_ghmc)
        alpha_param = sig.parameters["alpha"]
        assert alpha_param.default == 0.8

    def test_run_ghmc_delta_default_is_reasonable(self):
        import inspect

        from tengri.inference.backends.mcmc.ghmc import run_ghmc

        sig = inspect.signature(run_ghmc)
        delta_param = sig.parameters["delta"]
        assert delta_param.default == 0.65
