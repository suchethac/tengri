# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for #1796: sfh wildcard must not free met_*.

The sfh group's 'all_params: FREE' wildcard should not apply to met_*
parameters when there is no explicit met block. These parameters logically
belong to the "met" group and should use their registry defaults.
"""

from __future__ import annotations

import warnings

import pytest

from tengri import FIXED, FREE, Fixed, Uniform
from tengri.config.exceptions import WildcardPartialFreeWarning
from tengri.parameters import parse_groups

pytestmark = pytest.mark.regression_bug


class TestSFHWildcardMustNotFreeMet:
    """#1796: sfh wildcard scope must exclude met_* parameters."""

    def test_sfh_free_no_met_block_excludes_met_params(self):
        """sfh={'all_params': FREE} with no met block should NOT free met_logzsol (#1796).

        When there is no explicit met block, met_* parameters should be pinned
        at their Fixed defaults (as if met={'all_params': FIXED} was implicit).
        """
        spec = parse_groups(
            sfh={"type": "dpl", "all_params": FREE},
            dust={"type": "none"},
            neb={"type": "none"},
            agn={"type": "none"},
            redshift=0.1,
        )

        # met_logzsol should NOT be free (no met block means implicit FIXED)
        assert "met_logzsol" not in spec.free_params, (
            "met_logzsol should be Fixed when no met block is provided"
        )

    def test_sfh_fixed_fixes_met_params(self):
        """sfh={'all_params': FIXED} should also fix met_* parameters."""
        spec = parse_groups(
            sfh={"type": "dpl", "all_params": FIXED},
            dust={"type": "none"},
            neb={"type": "none"},
            agn={"type": "none"},
            redshift=0.1,
        )

        # met_logzsol should be fixed when sfh wildcard is FIXED
        assert "met_logzsol" not in spec.free_params

    def test_met_block_prevents_sfh_from_freeing_met_params(self):
        """With explicit met block, sfh wildcard never touches met_* params."""
        # met_logzsol in a met block with FIXED default
        spec = parse_groups(
            sfh={"type": "dpl", "all_params": FREE},
            met={"type": "table"},  # Default behavior
            dust={"type": "none"},
            neb={"type": "none"},
            agn={"type": "none"},
            redshift=0.1,
        )

        # With an explicit met block, met_logzsol stays fixed (table mode)
        assert "met_logzsol" not in spec.free_params

        # Confirm met_logzsol is partitioned to "met" group, not "sfh"
        # (This is implicit but assured by the fact it's fixed despite sfh=FREE)

    def test_explicit_met_override_in_sfh_raises_no_error(self):
        """Explicitly overriding a met param in sfh should work correctly."""
        spec = parse_groups(
            sfh={
                "type": "dpl",
                "all_params": FREE,
                "logzsol": Uniform(-1.0, 0.5),  # Explicit override
            },
            dust={"type": "none"},
            neb={"type": "none"},
            agn={"type": "none"},
            redshift=0.1,
        )

        # The explicit override should be respected
        assert "met_logzsol" in spec.free_params
        prov = getattr(spec, "_group_provenance", {})
        # Explicit overrides take precedence over wildcard
        assert prov.get("met_logzsol") == "user_prior"

    def test_roundtrip_preserves_sfh_wildcard_free_behavior(self):
        """to_groups() should preserve the sfh wildcard FREE behavior (#1796)."""
        spec1 = parse_groups(
            sfh={"type": "dpl", "all_params": FREE},
            redshift=Fixed(0.1),
        )

        # Convert back and re-parse
        groups = spec1.to_groups()
        spec2 = parse_groups(**groups)

        # Should have identical free params
        assert spec1.free_params == spec2.free_params

        # sfh_dpl params should still be free
        sfh_params = [p for p in spec2.free_params if "sfh_dpl" in p]
        assert len(sfh_params) == 5, f"Expected 5 sfh_dpl params, got {len(sfh_params)}"

        # met_logzsol should still be Fixed (no met block)
        assert "met_logzsol" not in spec2.free_params

    def test_migration_warning_fires_for_sfh_free_no_met_block(self):
        """sfh={'all_params': FREE} with no met block should emit migration warning (#1796)."""
        with pytest.warns(WildcardPartialFreeWarning, match="met_logzsol"):
            parse_groups(
                sfh={"type": "dpl", "all_params": FREE},
                dust={"type": "none"},
                neb={"type": "none"},
                agn={"type": "none"},
                redshift=0.1,
            )

    def test_no_warning_for_sfh_fixed_no_met_block(self):
        """sfh={'all_params': FIXED} with no met block should NOT emit warning."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", WildcardPartialFreeWarning)
            # Should not raise because the warning should not fire
            parse_groups(
                sfh={"type": "dpl", "all_params": FIXED},
                dust={"type": "none"},
                neb={"type": "none"},
                agn={"type": "none"},
                redshift=0.1,
            )

    def test_no_warning_with_met_block(self):
        """sfh={'all_params': FREE} WITH met block should NOT emit warning (#1796)."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", WildcardPartialFreeWarning)
            # Should not raise because the warning should not fire when met block exists
            parse_groups(
                sfh={"type": "dpl", "all_params": FREE},
                met={"type": "table"},
                dust={"type": "none"},
                neb={"type": "none"},
                agn={"type": "none"},
                redshift=0.1,
            )
