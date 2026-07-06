# SPDX-License-Identifier: BSD-3-Clause
"""Contract test: SEDModelComponent registry has stable baseline entries.

This test verifies that a bare `import tengri` registers a stable expected
set of component names. It enforces that new registrations do not regress,
while allowing for growth (new components).

The test uses a subset check (not exact equality) so later migrations can
add new domains to the registry without changing this test.
"""

import pytest


@pytest.mark.contract
def test_registry_baseline_keys_registered():
    """Verify baseline component keys are in the unified registry."""
    from tengri.components.sed_model_component import _REGISTRY

    # Baseline: components that must remain registered. The "mappings" port
    # was deleted in the 2026-07 dead-code purge (grammar-unreachable no-op;
    # canonical shock component is "shock", #851).
    expected_baseline = {
        "cat3d_wind",
        "draine2021_pah_ir",
        "kd18_disc",
        "powerlaw_disc",
        "radio_powerlaw",
        "schreiber2016_ir",
        "silva04",
        "skirtor",
        "xray_aird",
    }

    # Check that all baseline entries are in the registry
    actual_keys = set(_REGISTRY.keys())
    assert expected_baseline <= actual_keys, (
        f"baseline keys {expected_baseline - actual_keys} missing from registry"
    )

    # Sanity check: registry should have at least the baseline
    assert len(actual_keys) >= len(expected_baseline), "registry has fewer entries than expected"
