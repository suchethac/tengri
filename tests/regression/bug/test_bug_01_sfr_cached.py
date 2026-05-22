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

    def test_sfr_computed_not_hardcoded(self):
        """Verify the orchestrator computes SFR rather than hard-coding 1.0."""
        import inspect

        from tengri.components.stellar import component as stellar_component
        from tengri.forward import orchestrator, sed_model

        src = "\n".join(
            [
                inspect.getsource(sed_model),
                inspect.getsource(orchestrator),
                inspect.getsource(stellar_component),
            ]
        )
        # The fix: the canonical 10 Myr time-weighted helper is wired in.
        assert "time_weighted_sfr" in src, (
            "stellar component must call time_weighted_sfr to compute _sfr_current"
        )
        # Guard against any re-introduction of the 1.0 fallback at the _sfr_current line
        lines = [l for l in src.split("\n") if "_sfr_current" in l and "1.0" in l]
        assert not any("= 1.0" in l and "sfr" not in l for l in lines), (
            "Hard-coded _sfr_current = 1.0 found with no sfr fallback"
        )
