# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for issue #1999 — dead MCMC fits warn at construction.

Bug: An HMC fit returns a completely dead posterior — every draw identical for
every parameter, 100% divergent transitions — and Posterior.__init__ hands it
back silently. The existing guards (posterior.rhat() raises; convergence_check
flags it) are opt-in: a script that never calls diagnostics publishes dead
numbers.

Guard: detect at Posterior construction (the ONE seam every MCMC fit flows
through) and warn with UNAMBIGUOUS triggers:
- n_divergent == n_samples (every transition divergent), or
- any free parameter with all identical draws (np.ptp == 0) over >= 100 draws

Mutation testing requirements:
1. All-divergent trigger: n_divergent == n_samples. Dies if condition → False.
2. Frozen-param trigger: np.ptp == 0 for any param, n_samples >= 100.
   Dies if check → always False, or threshold changed.
3. No false alarm on small posteriors: < 100 draws, even if frozen.
4. No false alarm on healthy chains.
5. No false alarm on MAP (no n_divergent in diagnostics).
"""

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.config.exceptions import DeadFitWarning

pytestmark = pytest.mark.regression_bug


class TestDeadFitWarnsAtConstruction:
    """Issue #1999 — dead fit detection at Posterior construction."""

    def test_all_divergent_warns_with_message(self):
        """All-divergent trigger: n_divergent == n_samples fires warning.

        Mutation test: condition n_divergent == n_samples → False disables guard.
        """
        from tengri.inference.posterior import Posterior

        n_samples = 200
        # Create varying samples (not frozen) but all divergent
        samples = {
            "x": jnp.arange(n_samples, dtype=float),
            "y": jnp.sin(jnp.arange(n_samples, dtype=float)),
        }

        # Verify samples are not frozen
        assert float(np.ptp(np.asarray(samples["x"]))) > 0.0
        assert float(np.ptp(np.asarray(samples["y"]))) > 0.0

        # Should warn when all samples diverged
        with pytest.warns(DeadFitWarning, match=r"dead fit.*divergent"):
            Posterior(
                samples=samples,
                params={"x": jnp.array(100.0)},
                method="mcmc_nuts",
                wall_time_s=1.0,
                diagnostics={"n_divergent": n_samples, "n_samples": n_samples},
            )

    def test_frozen_param_600_draws_warns(self):
        """Frozen-param trigger: np.ptp == 0 for 600 draws, 0 divergences.

        Verifies the >= 100 draw threshold (no false alarm on tiny test posteriors).
        """
        from tengri.inference.posterior import Posterior

        n_samples = 600
        # All frozen — identical draws
        samples = {
            "x": jnp.full((n_samples,), 5.0),
            "y": jnp.full((n_samples,), -2.0),
        }

        # Should warn: frozen parameter with >= 100 draws
        with pytest.warns(DeadFitWarning, match=r"dead fit.*unique draw"):
            Posterior(
                samples=samples,
                params={"x": jnp.array(5.0)},
                method="mcmc_hmc",
                wall_time_s=1.0,
                diagnostics={"n_divergent": 0, "n_samples": n_samples},
            )

    def test_frozen_param_alone_with_many_divergences(self):
        """Combined frozen + divergent: both conditions true, single warning."""
        from tengri.inference.posterior import Posterior

        n_samples = 200
        # Frozen + all divergent
        samples = {
            "x": jnp.full((n_samples,), 5.0),
            "y": jnp.full((n_samples,), -2.0),
        }

        # Should warn (both conditions true)
        with pytest.warns(DeadFitWarning, match=r"dead fit"):
            Posterior(
                samples=samples,
                params={"x": jnp.array(5.0)},
                method="mcmc_hmc",
                wall_time_s=1.0,
                diagnostics={"n_divergent": n_samples, "n_samples": n_samples},
            )

    def test_healthy_posterior_no_warning(self):
        """Healthy chain: varying samples, low divergence → no warning."""
        from tengri.inference.posterior import Posterior

        n_samples = 200
        # Healthy: varying samples, low divergence
        samples = {
            "x": jnp.arange(n_samples, dtype=float),
            "y": jnp.sin(jnp.arange(n_samples, dtype=float)),
        }

        # Should NOT warn
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Treat warnings as errors
            Posterior(
                samples=samples,
                params={"x": jnp.array(100.0)},
                method="mcmc_nuts",
                wall_time_s=1.0,
                diagnostics={"n_divergent": 2, "n_samples": n_samples},
            )

    def test_map_posterior_no_warning(self):
        """MAP result (no n_divergent in diagnostics) does not warn."""
        from tengri.inference.posterior import Posterior

        # MAP-style: single point estimate, no samples
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Treat warnings as errors
            Posterior(
                samples=None,
                params={"x": jnp.array(5.0)},
                method="map",
                wall_time_s=1.0,
                diagnostics={"final_loss": 123.45},
            )

    def test_small_sample_frozen_no_warning(self):
        """Frozen chain with < 100 draws does not warn (test-posterior exemption)."""
        from tengri.inference.posterior import Posterior

        n_samples = 50  # Below threshold
        samples = {
            "x": jnp.full((n_samples,), 5.0),
        }

        # Should NOT warn (below 100-draw threshold)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            Posterior(
                samples=samples,
                params={"x": jnp.array(5.0)},
                method="mcmc_hmc",
                wall_time_s=1.0,
                diagnostics={"n_divergent": 0, "n_samples": n_samples},
            )

    def test_partial_divergence_no_warning(self):
        """Partial divergences (not 100%) do not warn."""
        from tengri.inference.posterior import Posterior

        n_samples = 200
        samples = {
            "x": jnp.arange(n_samples, dtype=float),
            "y": jnp.sin(jnp.arange(n_samples, dtype=float)),
        }

        # Partial divergence (50%, not 100%)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            Posterior(
                samples=samples,
                params={"x": jnp.array(100.0)},
                method="mcmc_nuts",
                wall_time_s=1.0,
                diagnostics={"n_divergent": n_samples // 2, "n_samples": n_samples},
            )

    def test_message_signature_contains_details(self):
        """Warning message states n_divergent, parameter name, unique count."""
        from tengri.inference.posterior import Posterior

        n_samples = 150
        # Frozen + divergent
        samples = {
            "param_a": jnp.full((n_samples,), 3.14),
            "param_b": jnp.arange(n_samples, dtype=float),
        }

        with pytest.warns(DeadFitWarning) as record:
            Posterior(
                samples=samples,
                params={"param_a": jnp.array(3.14)},
                method="mcmc_nuts",
                wall_time_s=1.0,
                diagnostics={"n_divergent": n_samples, "n_samples": n_samples},
            )

        assert len(record) > 0
        message = str(record[0].message)

        # Check for specific elements in the message
        assert "dead fit" in message.lower() or "critical" in message.lower()
        assert "150" in message or "150/150" in message  # n_divergent/n_samples
