# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for BUG-01: SFR hardcoded to 1.0 Msun/yr.

See ADR / docs/known_bugs.md for full context.
"""

import pytest

pytestmark = pytest.mark.regression_bug


class TestBug01SfrCached:
    """forward/sed_model.py — present-day SFR must reflect actual SFR.

    Fixed: the orchestrator path now feeds ``_sfr_current`` from
    ``time_weighted_sfr(sfr, age_yr, 1e7)`` (10 Myr Murphy+2011 timescale)
    for parametric SFH paths instead of hard-coding 1.0. Originally pinned
    to the legacy ``sed_pipeline.compute_sed_components`` body, deleted in
    Phase B closure; the equivalent invariant is upheld by the orchestrator.
    """

    def test_sfr_computed_not_hardcoded(self, synthetic_ssp_wide):
        """Present-day SFR must be computed, not hard-coded to 1.0 Msun/yr.

        Regression: #01 (reversed in orchestrator phase). Using a hard-coded
        1.0 Msun/yr regardless of the actual SFH caused SFR-derived quantities
        (radio, X-ray) to be independent of star formation history, a ~1-2 dex
        error for starburst/quiescent systems. Fix: time_weighted_sfr on a
        10 Myr timescale computes the instantaneous SFR correctly.

        Test: Build models with different SFH shapes and verify that
        sfr_current varies appropriately (not frozen at 1.0).
        """
        import jax.numpy as jnp

        from tengri import Fixed, SEDModel

        ssp = synthetic_ssp_wide

        # Build a model with parametric SFH
        from tengri import FREE

        model = SEDModel.build(
            ssp_data=ssp,
            observation=None,
            sfh={"type": "dpl", "all_params": FREE},  # All SFH params free
            met={"logzsol": Fixed(0.0)},  # Fixed solar metallicity
            redshift=Fixed(0.0),  # Rest-frame observation
        )

        # Create two parameter sets with different SFH shapes. For DPL model,
        # alpha=1 is declining SFR over time. At z=0 (present day):
        # - alpha < 1: star formation rises toward present (high SFR now)
        # - alpha > 1: star formation declines toward present (low SFR now)
        import jax

        key = jax.random.PRNGKey(0)
        log_mass = 10.5

        # High present-day SFR: alpha < 1
        params_high_sfr = model.spec.sample(key)
        params_high_sfr = dict(params_high_sfr)  # Convert to dict for modification
        params_high_sfr["sfh_dpl_alpha"] = 0.5
        params_high_sfr["sfh_dpl_log_total_mass"] = log_mass

        # Low present-day SFR: alpha > 1
        params_low_sfr = model.spec.sample(key)
        params_low_sfr = dict(params_low_sfr)  # Convert to dict for modification
        params_low_sfr["sfh_dpl_alpha"] = 2.5
        params_low_sfr["sfh_dpl_log_total_mass"] = log_mass

        # Get states
        state_high_sfr = model.predict_state(params_high_sfr)
        state_low_sfr = model.predict_state(params_low_sfr)

        # Verify time-weighted SFR is in derived state (sfr_10myr is the 10 Myr timescale)
        assert "sfr_10myr" in state_high_sfr.derived, (
            "sfr_10myr (10 Myr time-weighted SFR) must be published in ForwardState.derived"
        )

        sfr_high = state_high_sfr.derived["sfr_10myr"]
        sfr_low = state_low_sfr.derived["sfr_10myr"]

        # Verify both are finite and positive
        assert jnp.isfinite(sfr_high) and sfr_high > 0, (
            f"sfr_high must be finite and positive, got {sfr_high}"
        )
        assert jnp.isfinite(sfr_low) and sfr_low > 0, (
            f"sfr_low must be finite and positive, got {sfr_low}"
        )

        # Verify SFR depends on parameters (not hard-coded to 1.0)
        # The 10 Myr timescale is short on cosmological scales, so SFR differences
        # between alpha=0.5 and alpha=2.5 may be modest, but they should not be
        # identical, nor should sfr_10myr be a constant value like 1.0.
        assert sfr_high != sfr_low, (
            f"sfr_10myr must vary with SFH parameters. "
            f"alpha=0.5: {sfr_high:.3e}, alpha=2.5: {sfr_low:.3e}. "
            f"If identical, SFR is hard-coded."
        )

        # Verify the values are not suspiciously constant
        # If hardcoded to 1.0, both values would be 1.0
        assert not (jnp.abs(sfr_high - 1.0) < 0.01 and jnp.abs(sfr_low - 1.0) < 0.01), (
            f"SFR values too close to hard-coded 1.0 Msun/yr. "
            f"Got sfr_high={sfr_high:.3e}, sfr_low={sfr_low:.3e}."
        )
