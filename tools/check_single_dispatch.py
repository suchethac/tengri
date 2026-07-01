#!/usr/bin/env python3
"""CI guard for legacy registry cleanup (part of architecture migration).

After migrating a component domain from a legacy dispatch registry to the
unified SEDModelComponent registry, this guard verifies that the legacy
registry (marked as status='migrated' in migration_manifest.json) is now
empty or absent.

Domains marked as status='pending' are skipped (not yet enforced).

Usage
-----
    python tools/check_single_dispatch.py

Exit code 0 if all migrated registries are empty; non-zero if stale entries
remain.

Architecture migration workflow:
1. Domains start as status='pending' (this guard skips them).
2. When migration is complete, flip status to 'migrated'.
3. This guard then verifies the legacy registry is empty (no entries left).
4. This makes regressions visible: stale entries or new additions revert to
   'pending' and allow re-migration planning.
"""

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "tools" / "migration_manifest.json"


def safe_import_attr(module_path: str, attr_name: str) -> tuple[bool, Any | None]:
    """Safely import a module attribute.

    Returns (exists, value). exists=False means the attribute is missing
    (treated as pass); exists=True means it exists (check if empty).
    """
    try:
        # Import the module named by module_path, then read attr_name off it.
        module = __import__(module_path, fromlist=[attr_name])
        if not hasattr(module, attr_name):
            return False, None
        return True, getattr(module, attr_name)
    except (ImportError, AttributeError):
        return False, None


def main() -> int:
    """Run the single-dispatch cleanup guard.

    Returns
    -------
    int
        0 if all migrated registries are empty/absent; 1 if stale entries found.
    """
    # Load manifest
    try:
        with open(MANIFEST_PATH) as f:
            manifest = json.load(f)
    except Exception as e:
        print(f"ERROR: Could not load manifest at {MANIFEST_PATH}: {e}", file=sys.stderr)
        return 1

    violations = []
    domains = manifest.get("domains", {})

    for domain_name, domain_cfg in domains.items():
        status = domain_cfg.get("status", "pending")

        # Skip pending domains
        if status == "pending":
            continue

        # For migrated domains, enforce BOTH (whichever are configured):
        #  - legacy_dispatch: the legacy dispatch FUNCTION must be ABSENT
        #    (unimportable) — the honest single-dispatch invariant. A loader
        #    cache can survive under its own name, but no second dispatch entry
        #    point may remain reachable.
        #  - legacy_registry: the legacy dispatch REGISTRY must be empty/absent.
        if status == "migrated":
            checked = False

            dispatch_cfg = domain_cfg.get("legacy_dispatch", {})
            d_module, d_attr = dispatch_cfg.get("module"), dispatch_cfg.get("attr")
            if d_module and d_attr:
                checked = True
                exists, _ = safe_import_attr(d_module, d_attr)
                if exists:
                    violations.append(
                        (domain_name, d_module, d_attr, ["<dispatch symbol still importable>"])
                    )

            legacy_cfg = domain_cfg.get("legacy_registry", {})
            module_path, attr_name = legacy_cfg.get("module"), legacy_cfg.get("attr")
            if module_path and attr_name:
                checked = True
                exists, obj = safe_import_attr(module_path, attr_name)
                if exists:
                    if isinstance(obj, dict) and obj:
                        violations.append((domain_name, module_path, attr_name, list(obj.keys())))
                    elif hasattr(obj, "__len__") and len(obj) > 0:
                        violations.append((domain_name, module_path, attr_name, list(obj)))

            if not checked:
                print(
                    f"WARNING: Domain '{domain_name}' marked migrated but has neither "
                    f"legacy_dispatch nor legacy_registry configured",
                    file=sys.stderr,
                )

    # Report findings
    if violations:
        print(f"FAIL: {len(violations)} migrated registry/registries still have entries:\n")
        for domain_name, module_path, attr_name, entries in violations:
            print(f"  Domain: {domain_name}")
            print(f"    Registry: {module_path}.{attr_name}")
            print(f"    Stale entries ({len(entries)}):")
            for entry in sorted(entries):
                print(f"      - {entry}")
            print()
        print("Fix: Complete the architecture migration or revert domain status to 'pending'.")
        return 1

    migrated_count = sum(1 for d in domains.values() if d.get("status") == "migrated")
    print(f"OK: all {migrated_count} migrated domain(s) have empty/absent legacy registries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
