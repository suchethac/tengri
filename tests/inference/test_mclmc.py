# SPDX-License-Identifier: BSD-3-Clause
import pytest

pytestmark = pytest.mark.contract

"""Tests for MCLMC and Adjusted MCLMC sampler integration."""

import pytest

try:
    import blackjax  # noqa: F401

    HAS_BLACKJAX = True
except ImportError:
    HAS_BLACKJAX = False


@pytest.mark.skipif(not HAS_BLACKJAX, reason="blackjax not available")
class TestMCLMCImports:
    def test_can_import_run_mclmc(self):
        from tengri.inference.backends.mcmc.mclmc import run_mclmc

        assert callable(run_mclmc)

    def test_can_import_run_adjusted_mclmc(self):
        from tengri.inference.backends.mcmc.mclmc import run_adjusted_mclmc

        assert callable(run_adjusted_mclmc)

    def test_can_import_posterior(self):
        from tengri.inference.posterior import Posterior

        assert Posterior is not None


@pytest.mark.skipif(not HAS_BLACKJAX, reason="blackjax not available")
class TestMCLMCSignatures:
    def test_run_mclmc_signature(self):
        import inspect

        from tengri.inference.backends.mcmc.mclmc import run_mclmc

        sig = inspect.signature(run_mclmc)
        assert "context" in sig.parameters
        assert "key" in sig.parameters
        assert "n_warmup" in sig.parameters
        assert "n_samples" in sig.parameters
        assert "desired_energy_var" in sig.parameters
        assert "verbose" in sig.parameters

    def test_run_adjusted_mclmc_signature(self):
        import inspect

        from tengri.inference.backends.mcmc.mclmc import run_adjusted_mclmc

        sig = inspect.signature(run_adjusted_mclmc)
        assert "context" in sig.parameters
        assert "key" in sig.parameters
        assert "n_warmup" in sig.parameters
        assert "n_samples" in sig.parameters
        assert "target_accept_rate" in sig.parameters
        assert "verbose" in sig.parameters

    def test_run_mclmc_defaults(self):
        """The defaults are counted in integrator steps, not NUTS-comparable draws.

        One MCLMC draw is one integrator step (two gradient evaluations), where
        one NUTS draw is a whole trajectory — ~50 gradient evaluations on the
        D=7 photometry mocks. The old 500 / 2000 defaults were NUTS-shaped and
        bought ~4000 gradient evaluations for a posterior whose worst direction
        has an autocorrelation time of a few thousand *steps*, which is the
        arithmetic behind this backend's "ESS ~ 1" quarantine.
        """
        import inspect

        from tengri.inference.backends.mcmc.mclmc import run_mclmc

        sig = inspect.signature(run_mclmc)
        assert sig.parameters["n_warmup"].default == 5000
        assert sig.parameters["n_samples"].default == 20000
        assert sig.parameters["desired_energy_var"].default == 5e-4
        assert sig.parameters["verbose"].default is True

    def test_run_adjusted_mclmc_defaults(self):
        import inspect

        from tengri.inference.backends.mcmc.mclmc import run_adjusted_mclmc

        sig = inspect.signature(run_adjusted_mclmc)
        assert sig.parameters["n_warmup"].default == 500
        assert sig.parameters["n_samples"].default == 2000
        assert sig.parameters["target_accept_rate"].default == 0.65
        assert sig.parameters["verbose"].default is True
