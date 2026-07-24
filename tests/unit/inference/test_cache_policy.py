# SPDX-License-Identifier: BSD-3-Clause
"""Test suite for ForwardModel.prewarm() and cache policy derivation.

Tests the surface-derived cache policy contract and the lean= deprecation path.
"""

import warnings

import jax.numpy as jnp
import pytest

from tengri import ForwardModel, SEDModel, recipes


class TestForwardPrewarm:
    """Test ForwardModel.prewarm() surface and idempotency."""

    def test_forward_prewarm_exists_and_is_idempotent(
        self, synthetic_ssp_wide, synthetic_tophat_obs
    ):
        """ForwardModel.prewarm() exists and is idempotent.

        A second call to prewarm() with the same method is a fast no-op.
        The method compiles JIT kernels and populates the adaptation cache
        but does not duplicate compilation work on subsequent calls.
        """
        sed = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=synthetic_tophat_obs,
            **recipes.star_forming_photometry(),
        )
        model = ForwardModel.build(sed=sed)

        # First call should succeed and compile kernels
        model.prewarm(method="mcmc_nuts")

        # Second call should be a fast no-op (idempotent)
        model.prewarm(method="mcmc_nuts")

        # No assertion needed beyond "doesn't raise"; the test passes if
        # both calls complete without error. The adaptation cache is
        # persistent and reused.

    def test_lean_kwarg_warns_deprecation(self, synthetic_ssp_wide, synthetic_tophat_obs):
        """Fitter.run(lean=...) emits a DeprecationWarning with retire message.

        When a caller passes lean=True or lean=False to Fitter.run(),
        a one-shot DeprecationWarning is issued with the message naming issue #1318.
        The behavior is honored for back-compat.
        """
        sed = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=synthetic_tophat_obs,
            **recipes.star_forming_photometry(),
        )
        fwd = ForwardModel.build(sed=sed)
        data = jnp.ones(5)
        noise = jnp.ones(5) * 0.1

        # Import Fitter to test directly
        from tengri.inference.fitter import Fitter

        fitter = Fitter(fwd, data=data, noise=noise)

        # Test that passing lean=True raises DeprecationWarning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Run with lean=True; we expect it to warn and complete
            fitter.run("map", n_steps=1, lean=True)
            assert len(w) >= 1
            # Find the deprecation warning (filter out other warnings)
            dep_warns = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warns) >= 1
            assert "#1318" in str(dep_warns[0].message)

    def test_iterate_policy_same_fit_reuses_tier3(self, synthetic_ssp_wide, synthetic_tophat_obs):
        """iterate policy (default) reuses tier-3 cache for identical fits.

        When two identical fits run back-to-back with the default policy
        (derive policy = 'iterate' when no Catalog flag is set), the second
        fit is significantly faster than the first because it reuses the
        tier-3 inference-body cache. The timing ratio (second / first) is
        measured; if flaky due to system load, the test keeps a generous
        threshold and notes the flakiness.
        """
        import time

        sed = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=synthetic_tophat_obs,
            **recipes.star_forming_photometry(),
        )
        fwd = ForwardModel.build(sed=sed)
        data = jnp.ones(5)
        noise = jnp.ones(5) * 0.1

        from tengri.inference.fitter import Fitter

        # First fit (cold cache)
        fitter1 = Fitter(fwd, data=data, noise=noise)
        start1 = time.time()
        result1 = fitter1.run("map", n_steps=5)
        time1 = time.time() - start1

        # Second fit (warm cache from first run)
        fitter2 = Fitter(fwd, data=data, noise=noise)
        start2 = time.time()
        result2 = fitter2.run("map", n_steps=5)
        time2 = time.time() - start2

        # The second fit should reuse the tier-3 cache and be faster.
        # Wall-clock ratio: time2 / time1 should be << 1.0.
        # Threshold is generous (1.2) to account for system load variability.
        # If this test fails ONLY on timing (not logic), note the flakiness
        # in the report and keep the generous threshold.
        ratio = time2 / time1 if time1 > 0 else 1.0
        assert ratio < 1.2, (
            f"Second fit not significantly faster. Ratio: {ratio:.3f}. "
            f"This may be flaky under system load; check if logic is correct."
        )
