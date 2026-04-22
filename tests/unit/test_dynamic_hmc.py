"""Tests for Dynamic HMC sampler integration."""

import jax
import pytest

jax.config.update("jax_enable_x64", True)

try:
    import blackjax  # noqa: F401

    HAS_BLACKJAX = True
except ImportError:
    HAS_BLACKJAX = False


@pytest.mark.skipif(not HAS_BLACKJAX, reason="blackjax not available")
class TestDynamicHMCImports:
    def test_can_import_run_dynamic_hmc(self):
        from tengri.inference.backends.mcmc.dynamic_hmc import run_dynamic_hmc

        assert callable(run_dynamic_hmc)

    def test_can_import_posterior(self):
        from tengri.inference.posterior import Posterior

        assert Posterior is not None


@pytest.mark.skipif(not HAS_BLACKJAX, reason="blackjax not available")
class TestDynamicHMCSignatures:
    def test_run_dynamic_hmc_signature(self):
        import inspect

        from tengri.inference.backends.mcmc.dynamic_hmc import run_dynamic_hmc

        sig = inspect.signature(run_dynamic_hmc)
        assert "fitter" in sig.parameters
        assert "key" in sig.parameters
        assert "n_warmup" in sig.parameters
        assert "n_burnin" in sig.parameters
        assert "n_samples" in sig.parameters
        assert "verbose" in sig.parameters
