# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for issue #1437 — frozen HMC chain reports converged R-hat.

Bug: A frozen chain (all 200 samples identical, 200/200 divergences) reports
split-R-hat ~0.995 and is read as converged. The diagnostic is structurally
blind to the failure: R-hat compares within-chain to between-chain variance;
both are ~0 for a frozen chain, so the ratio is ~1.

The guard: detect frozen chains (n_unique == 1 per parameter) and all-divergent
chains (n_divergent == n_samples), surface with warnings and machine-readable
convergence flag set to False.

Mutation testing requirements:
1. Frozen-isolation: frozen but NO divergences. Asserts frozen_params flag and
   "FROZEN" in warnings. Dies under Mutant A (frozen-check → `if False:`).
2. All-divergent-flag: varying samples, n_divergent == n_samples. Asserts
   all_samples_divergent flag and "CRITICAL" in warnings. Dies under Mutant B
   (n_div == n_samples → n_div == n_samples + 1).
3. Healthy control and rhat raise test kept for regression coverage.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug


class TestFrozenChainDiagnostics:
    """Issue #1437 — frozen chain detection and convergence verdict."""

    def test_frozen_chain_raises_on_rhat(self):
        """rhat() should raise ValueError when all samples are identical."""
        from tengri.inference.posterior import Posterior

        # Create frozen chain: all 200 samples identical
        key = jax.random.PRNGKey(0)
        n_samples = 200
        identical_value = 0.737
        frozen_samples = {
            "dust_tau_bc": jnp.full((n_samples,), identical_value),
            "dust_tau_diff": jnp.full((n_samples,), 0.281),
            "met_logzsol": jnp.full((n_samples,), -1.171),
        }

        p = Posterior(
            samples=frozen_samples,
            params={"dust_tau_bc": jnp.array(identical_value)},
            method="mcmc_hmc",
            wall_time_s=1.0,
            diagnostics={"n_divergent": 0, "n_samples": n_samples},
        )

        # rhat() should raise, not return an empty dict silently
        with pytest.raises(ValueError, match="chain did not move"):
            p.rhat()

    def test_frozen_isolation_zero_divergences(self):
        """Frozen-chain guard fires: 200 identical samples, ZERO divergences.

        Mutation test: Mutant A replaces frozen-detection condition with `if False:`
        and this test should FAIL (guard disabled, converged=True).

        Asserts specifically on frozen_params flag and "FROZEN" warning,
        not just converged=False (which could come from other warnings).
        """
        from tengri.analysis.plotting.convergence import convergence_check
        from tengri.inference.posterior import Posterior

        # Create frozen chain with ZERO divergences (pure frozen case)
        n_samples = 200
        frozen_samples = {
            "dust_tau_bc": jnp.full((n_samples,), 0.737),
            "dust_tau_diff": jnp.full((n_samples,), 0.281),
            "met_logzsol": jnp.full((n_samples,), -1.171),
        }

        p = Posterior(
            samples=frozen_samples,
            params={"dust_tau_bc": jnp.array(0.737)},
            method="mcmc_hmc",
            wall_time_s=1.0,
            diagnostics={"n_divergent": 0, "n_samples": n_samples},
        )

        info = convergence_check(p, method_name="HMC", verbose=False)

        # --- Assertion 1: frozen_params flag is set ---
        assert "frozen_params" in info, (
            "frozen_params key should be in info dict when frozen parameters detected"
        )
        frozen_list = info["frozen_params"]
        assert len(frozen_list) == 3, (
            f"Expected 3 frozen parameters, got {len(frozen_list)}: {frozen_list}"
        )
        assert set(frozen_list) == {
            "dust_tau_bc",
            "dust_tau_diff",
            "met_logzsol",
        }, f"Frozen params list does not match: {frozen_list}"

        # --- Assertion 2: "FROZEN" appears in warnings ---
        warning_str = " ".join(info["warnings"])
        assert "FROZEN" in warning_str, (
            f"Warning should contain 'FROZEN' marker. Got: {info['warnings']}"
        )

        # --- Assertion 3: converged is False (consequence of guard firing) ---
        assert not info["converged"], "Frozen chain should not be marked converged"

    def test_all_divergent_flag_varying_samples(self):
        """All-divergent guard fires: varying samples, n_divergent == n_samples.

        Mutation test: Mutant B changes `if n_div == n_samples and n_samples > 0:`
        to `n_div == n_samples + 1` and this test should FAIL (guard disabled,
        all_samples_divergent flag not set).

        Asserts specifically on all_samples_divergent flag and "CRITICAL" warning.
        """
        from tengri.analysis.plotting.convergence import convergence_check
        from tengri.inference.posterior import Posterior

        # Create chain with VARYING samples (not frozen) but all divergent
        key = jax.random.PRNGKey(0)
        n_samples = 200
        varying_samples = {
            "x": jax.random.normal(key, (n_samples,)) + 5.0,
            "y": jax.random.normal(key, (n_samples,)) - 2.0,
            "z": jax.random.normal(key, (n_samples,)) * 0.1,
        }

        # Verify samples are NOT frozen (ptp > 0)
        for name, arr in varying_samples.items():
            ptp = float(np.ptp(np.asarray(arr)))
            assert ptp > 0.0, f"Samples for {name} should be varying (ptp={ptp})"

        p = Posterior(
            samples=varying_samples,
            params={"x": jnp.array(5.0)},
            method="mcmc_nuts",
            wall_time_s=1.0,
            diagnostics={"n_divergent": n_samples, "n_samples": n_samples},
        )

        info = convergence_check(p, method_name="NUTS", verbose=False)

        # --- Assertion 1: all_samples_divergent flag is True ---
        assert info.get("all_samples_divergent") is True, (
            "all_samples_divergent flag should be True when n_divergent == n_samples. "
            f"Got: {info.get('all_samples_divergent')}"
        )

        # --- Assertion 2: "CRITICAL" appears in warnings ---
        warning_str = " ".join(info["warnings"])
        assert "CRITICAL" in warning_str, (
            "Warning should contain 'CRITICAL' marker for 100% divergence. "
            f"Got: {info['warnings']}"
        )

        # --- Assertion 3: converged is False ---
        assert not info["converged"], "All-divergent chain should not be marked converged"

    def test_healthy_chain_convergence_check_passes(self):
        """convergence_check() should pass normal varying chains with no divergences."""
        from tengri.analysis.plotting.convergence import convergence_check
        from tengri.inference.posterior import Posterior

        # Create healthy chain with varying samples and no divergences
        key = jax.random.PRNGKey(42)
        n_samples = 200
        healthy_samples = {
            "dust_tau_bc": jax.random.normal(key, (n_samples,)) * 0.05 + 0.737,
            "dust_tau_diff": jax.random.normal(key, (n_samples,)) * 0.05 + 0.281,
            "met_logzsol": jax.random.normal(key, (n_samples,)) * 0.05 - 1.171,
        }

        p = Posterior(
            samples=healthy_samples,
            params={"dust_tau_bc": jnp.array(0.737)},
            method="mcmc_hmc",
            wall_time_s=1.0,
            diagnostics={"n_divergent": 0, "n_samples": n_samples},
        )

        info = convergence_check(p, method_name="HMC", verbose=False)

        # Should have no frozen_params and no all_samples_divergent issues
        assert "frozen_params" not in info or len(info.get("frozen_params", [])) == 0, (
            "Healthy chain should not have frozen parameters"
        )
        assert info.get("all_samples_divergent") is False, (
            "Healthy chain should have all_samples_divergent=False"
        )

        # Should not have FROZEN or CRITICAL in warnings
        warning_str = " ".join(info.get("warnings", [])).lower()
        assert "frozen" not in warning_str, "Healthy chain should not be marked frozen"
        assert "critical" not in warning_str, "Healthy chain should not have CRITICAL warning"

    def test_frozen_chain_high_rhat_equivalence(self):
        """Verify that frozen chain rhat() would be ~1.0 if computed.

        This test documents WHY the guard is necessary: split-R-hat cannot
        detect a frozen chain (both within and between-chain variance are ~0,
        so the ratio is ~1).
        """
        from tengri.analysis.diagnostics.autocorrelation import rhat as compute_rhat

        # Create frozen chain
        n_samples = 100
        identical_value = 0.737
        frozen_samples = {
            "x": np.full((n_samples,), identical_value),
        }

        # Compute rhat manually — should produce no entries (static params excluded)
        rhat_result = compute_rhat(frozen_samples)

        # Frozen chain should produce NO entries (static params excluded)
        assert len(rhat_result) == 0, (
            "rhat() should exclude static (zero-variance) parameters; "
            "frozen chain with identical samples is static"
        )

        # This is the key insight: rhat() returns empty dict when all params are frozen.
        # The guard in Posterior.rhat() must check for this and raise/warn rather than
        # returning silently. With only some params frozen, those frozen ones are
        # silently dropped, which is dangerous if the user calls max(rhat().values()).
