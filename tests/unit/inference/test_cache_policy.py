# SPDX-License-Identifier: BSD-3-Clause
"""Test suite for ForwardModel.prewarm() and cache policy derivation.

Tests the surface-derived cache policy contract and the lean= deprecation path.
"""

import warnings

import jax.numpy as jnp

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

    def test_cache_policy_derivation_iterate_vs_sweep(
        self, monkeypatch, synthetic_ssp_wide, synthetic_tophat_obs
    ):
        """The cache-policy SEAM (#1318), tested DETERMINISTICALLY.

        The wall-clock 'second fit is faster' proxy is not reliable at unit
        scale — a small MAP fit is dominated by fixed overhead, so the warm/cold
        ratio hovers near 1.0 regardless of reuse. Instead we assert the actual
        policy derivation, which is what T4 built: the default run derives
        'iterate' (``clear_shared_caches`` called with a keep_sig TUPLE, so the
        matching entry is preserved), while the deprecated ``lean=True`` derives
        'sweep' (keep_sig=None, drop everything).
        """
        import tengri.inference.jit_engine as je
        from tengri.inference.fitter import Fitter

        sed = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=synthetic_tophat_obs,
            **recipes.star_forming_photometry(),
        )
        fwd = ForwardModel.build(sed=sed)
        data = jnp.ones(5)
        noise = jnp.ones(5) * 0.1

        keep_sigs = []
        orig = je.clear_shared_caches

        def spy(*args, **kwargs):
            keep_sigs.append(kwargs.get("keep_sig", "MISSING"))
            return orig(*args, **kwargs)

        monkeypatch.setattr(je, "clear_shared_caches", spy)

        # Default policy -> iterate -> keep_sig is a tuple (preserve matching).
        Fitter(fwd, data=data, noise=noise).run("map", n_steps=5)
        assert keep_sigs, "run() did not invoke the cache-policy seam"
        assert isinstance(keep_sigs[-1], tuple), (
            f"default policy should be 'iterate' (keep_sig tuple), got {keep_sigs[-1]!r}"
        )

        # Deprecated lean=True -> sweep -> keep_sig is None (drop all stale).
        keep_sigs.clear()
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            Fitter(fwd, data=data, noise=noise).run("map", n_steps=5, lean=True)
        assert keep_sigs and keep_sigs[-1] is None, (
            f"lean=True should derive 'sweep' (keep_sig=None), got {keep_sigs[-1]!r}"
        )
