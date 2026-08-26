# SPDX-License-Identifier: BSD-3-Clause
"""Tests for tools/check_param_prefixes.py.

Covers the registry-based prefix validation added as part of ADR-0005
follow-up #2. Verifies that:
1. Known-good parameters pass both registry and prefix checks
2. Unregistered parameters (even if prefix-valid) fail the registry check
3. Registered but prefix-invalid parameters fail the prefix check
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add tools/ to path so we can import the check script.
# Layout: tests/contract/<this_file> -> repo root is 2 levels up.
tools_dir = Path(__file__).parent.parent.parent / "tools"
sys.path.insert(0, str(tools_dir))

from check_param_prefixes import is_valid_param_name

pytestmark = pytest.mark.contract


class TestParameterValidation:
    """Test is_valid_param_name with and without registry."""

    def test_known_good_param_passes_without_registry(self):
        """Parameters with valid prefixes pass even without registry context."""
        assert is_valid_param_name("sfh_dpl_alpha")
        assert is_valid_param_name("dust_tau_bc")
        assert is_valid_param_name("agn_lum_ratio")
        assert is_valid_param_name("redshift")

    def test_unknown_prefix_fails(self):
        """Parameters with unknown prefixes always fail."""
        assert not is_valid_param_name("foo_bar")
        assert not is_valid_param_name("xyz_param")
        assert not is_valid_param_name("bad_prefix")

    def test_valid_prefix_requires_registry_check(self):
        """Parameters that look valid by prefix must be checked against registry."""
        import tengri

        registered = set(tengri.list_parameters().names())

        # dust_tau_bc is a known good registered parameter
        assert "dust_tau_bc" in registered
        assert is_valid_param_name("dust_tau_bc", registered)

        # Deliberately fabricated parameter with valid prefix but not registered
        fake_param = "dust_fake_unknown_param_xyz"
        assert fake_param not in registered
        assert not is_valid_param_name(fake_param, registered)

        # The same fake parameter passes prefix check without registry context
        assert is_valid_param_name(fake_param, None)

    def test_registry_membership_check(self):
        """Registry membership is necessary when registered_params is provided."""
        import tengri

        registered = set(tengri.list_parameters().names())

        # redshift is in the registry
        assert "redshift" in registered
        assert is_valid_param_name("redshift", registered)

        # A valid-looking but unregistered param fails
        unregistered = "dust_nonexistent_param"
        assert unregistered not in registered
        assert not is_valid_param_name(unregistered, registered)

    def test_exact_match_redshift(self):
        """'redshift' is a valid bare-name match (NAMING_CONTRACT §3.2)."""
        assert is_valid_param_name("redshift")

        # Test with registry context
        import tengri

        registered = set(tengri.list_parameters().names())
        assert "redshift" in registered
        assert is_valid_param_name("redshift", registered)

    def test_preset_params_pass_prefix_check(self):
        """All free parameters in presets must satisfy the prefix rule."""
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
        from check_param_prefixes import collect_registered_params

        import tengri.presets as presets

        registered = collect_registered_params()

        # Derived from the registry, not retyped. The hand-written list this
        # replaces omitted synthesizer_default, and every preset was wrapped in
        # `except Exception: pass`, so a preset that failed to construct was
        # skipped in silence — with all six failing, the assertion below would
        # still have passed on an empty dict.
        preset_names = sorted(row["name"] for row in presets.list_presets())
        assert len(preset_names) >= 7, f"preset registry collapsed to {preset_names}"

        all_failures = {}
        for preset_name in preset_names:
            preset_fn = getattr(presets, preset_name, None)
            assert preset_fn is not None, (
                f"{preset_name} is registered but not exposed on tengri.presets"
            )
            returned = preset_fn()

            # Presets do not agree on tuple order: most return
            # (Parameters, ...), synthesizer_default returns (config,
            # Parameters). Unpacking positionally is what made the latter raise
            # AttributeError, which the swallow then hid. Pick by contract.
            params = next((r for r in returned if hasattr(r, "free_params")), None)
            assert params is not None, (
                f"{preset_name} returned no object exposing free_params: "
                f"{[type(r).__name__ for r in returned]}"
            )

            failures = [p for p in params.free_params if not is_valid_param_name(p, registered)]
            if failures:
                all_failures[preset_name] = failures

        assert not all_failures, f"Preset free params fail validation: {all_failures}"
