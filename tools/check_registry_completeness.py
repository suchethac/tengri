#!/usr/bin/env python3
"""CI guard for SEDModelComponent registry coverage (part of migration).

After migrating a component domain from a legacy dispatch registry to the
unified SEDModelComponent registry, this guard verifies that all expected
types are registered in tengri.components.sed_model_component._REGISTRY.

Domains marked as status='pending' are skipped (not yet enforced).

Usage
-----
    python tools/check_registry_completeness.py

Exit code 0 if all migrated domains' expected types are registered; non-zero
if any are missing.

Architecture migration workflow:
1. List expected component names in migration_manifest.json under
   expected_types for the domain.
2. When migration is complete, flip status to 'migrated'.
3. This guard then verifies all expected names are keys in _REGISTRY.
4. Regressions (deleted components, failed registrations) become visible.
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "tools" / "migration_manifest.json"


def main() -> int:
    """Run the registry completeness guard.

    Returns
    -------
    int
        0 if all migrated domains' types are in _REGISTRY; 1 if missing.
    """
    # Load manifest
    try:
        with open(MANIFEST_PATH) as f:
            manifest = json.load(f)
    except Exception as e:
        print(f"ERROR: Could not load manifest at {MANIFEST_PATH}: {e}", file=sys.stderr)
        return 1

    # Import tengri and get the registry
    try:
        from tengri.components.sed_model_component import _REGISTRY
    except Exception as e:
        print(f"ERROR: Could not import tengri: {e}", file=sys.stderr)
        return 1

    violations = []
    domains = manifest.get("domains", {})

    for domain_name, domain_cfg in domains.items():
        status = domain_cfg.get("status", "pending")

        # Skip pending domains
        if status == "pending":
            continue

        # For migrated domains, check that all expected_types are in _REGISTRY
        if status == "migrated":
            expected_types = domain_cfg.get("expected_types", [])
            for type_name in expected_types:
                if type_name not in _REGISTRY:
                    violations.append((domain_name, type_name))

    # Report findings
    if violations:
        print(f"FAIL: {len(violations)} expected type(s) missing from _REGISTRY:\n")
        by_domain = {}
        for domain_name, type_name in violations:
            if domain_name not in by_domain:
                by_domain[domain_name] = []
            by_domain[domain_name].append(type_name)

        for domain_name in sorted(by_domain.keys()):
            print(f"  Domain: {domain_name}")
            for type_name in sorted(by_domain[domain_name]):
                print(f"    - {type_name}")
        print()
        print(
            "Fix: Ensure all components are registered via SEDModelComponent.__init_subclass__()."
        )
        return 1

    migrated_count = sum(1 for d in domains.values() if d.get("status") == "migrated")
    if migrated_count > 0:
        print(f"OK: all {migrated_count} migrated domain(s) have complete type coverage")
    else:
        print("OK: no migrated domains yet (all pending)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
