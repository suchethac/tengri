#!/usr/bin/env python
"""CI guard: verify grammar documentation completeness.

Ensures that:
1. Every structural key documented in docs/model_configuration.md exists
   in tengri.parameters.groups._GROUP_STRUCTURAL_KEYS.
2. Every user-facing structural key appears somewhere in the documentation.

The internal '*' wildcard is replaced by 'all_params' for user documentation.

Exit codes:
- 0: All structural keys documented and accounted for.
- 1: Documentation is missing or structural keys are undocumented/extra.
"""

from __future__ import annotations

import pathlib
import re
import sys
from typing import NamedTuple


class KeyMismatch(NamedTuple):
    """A discrepancy between doc and code."""

    key: str
    issue: str  # 'missing_from_code', 'undocumented', etc.


def _read_doc_keys_from_file(doc_path: pathlib.Path) -> dict[str, set[str]]:
    """Extract structural keys from the documentation.

    Parses docs/model_configuration.md for sections like:
    "### Star-formation history: `sfh`"
    followed by "**Structural keys:**" and a list of keys.

    Returns
    -------
    dict[str, set[str]]
        Mapping of domain name to set of documented keys (with '*' replaced
        by 'all_params' to match user-facing spelling).
    """
    content = doc_path.read_text(encoding="utf-8")

    # Find "### ... `domain` ..." followed by "**Structural keys:**"
    domain_keys: dict[str, set[str]] = {}

    # Split into sections by ### headings
    sections = re.split(r"^### ", content, flags=re.MULTILINE)

    for section in sections[1:]:  # Skip the part before the first ###
        lines = section.split("\n", 1)
        if len(lines) < 2:
            continue

        heading = lines[0]
        body = lines[1]

        # Extract domain name from heading like "Star-formation history: `sfh`"
        domain_match = re.search(r"`([a-z_]+)`", heading)
        if not domain_match:
            continue

        domain = domain_match.group(1)

        # Look for "**Structural keys:**" section
        keys_pattern = r"\*\*Structural keys:\*\*\n(.*?)(?=\n\n|$)"
        keys_match = re.search(keys_pattern, body, re.DOTALL)
        if not keys_match:
            continue

        keys_section = keys_match.group(1)

        # Extract keys from bullet lines like:
        # - `'key_name'` —
        # - `'key1'`, `'key2'`, `'key3'` —
        # Only match keys that appear at the start of a bullet line (after "- ")
        bullet_lines = re.findall(r"^- .+$", keys_section, re.MULTILINE)
        keys = set()
        for line in bullet_lines:
            # Only add keys that appear before the em-dash (—) separator.
            # In practice, keys come before the em-dash, examples after.
            em_dash_pos = line.find("—")
            if em_dash_pos == -1:
                em_dash_pos = len(line)
            before_dash = line[:em_dash_pos]
            key_matches = re.findall(r"`['\"]([a-z_A-Z0-9]+)['\"]`", before_dash)
            keys.update(key_matches)

        # Translate '*' to 'all_params' for user-facing comparison
        keys = {("all_params" if k == "*" else k) for k in keys}
        domain_keys[domain] = keys

    return domain_keys


def _get_code_keys() -> dict[str, set[str]]:
    """Extract structural keys from the code.

    Reads tengri.parameters.groups._GROUP_STRUCTURAL_KEYS and returns
    a mapping of domain to keys. The internal '*' is kept as-is in the code
    mapping (since we'll translate it to 'all_params' in the comparison).

    Returns
    -------
    dict[str, set[str]]
        Mapping of domain name to set of structural keys (with '*' included).
    """
    try:
        from tengri.parameters.groups import _GROUP_STRUCTURAL_KEYS
    except ImportError:
        print("Error: Could not import _GROUP_STRUCTURAL_KEYS from tengri", file=sys.stderr)
        return {}

    # Top-level groups only (no '.' in the name)
    return {group: set(keys) for group, keys in _GROUP_STRUCTURAL_KEYS.items() if "." not in group}


def _check_consistency(
    doc_keys: dict[str, set[str]], code_keys: dict[str, set[str]]
) -> list[KeyMismatch]:
    """Compare documented vs code-defined structural keys.

    Returns a list of issues found.
    """
    issues: list[KeyMismatch] = []

    # For each domain in code, check that it's documented
    for domain, keys in code_keys.items():
        if domain not in doc_keys:
            issues.append(KeyMismatch(domain, "missing_from_docs"))
            continue

        # Translate '*' to 'all_params' for comparison
        code_keys_translated = {("all_params" if k == "*" else k) for k in keys}
        doc_keys_for_domain = doc_keys[domain]

        # Check for undocumented keys
        for key in code_keys_translated:
            if key not in doc_keys_for_domain:
                issues.append(KeyMismatch(f"{domain}.{key}", "undocumented"))

        # Check for documented keys that don't exist in code
        for key in doc_keys_for_domain:
            if key not in code_keys_translated:
                issues.append(KeyMismatch(f"{domain}.{key}", "extra_in_docs"))

    # Check for domains documented but not in code
    for domain in doc_keys:
        if domain not in code_keys:
            issues.append(KeyMismatch(domain, "extra_domain_in_docs"))

    return issues


def main() -> int:
    """Check grammar documentation completeness.

    Returns 0 if all structural keys are documented and consistent, 1 otherwise.
    """
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    doc_path = repo_root / "docs" / "model_configuration.md"

    if not doc_path.exists():
        print(f"Error: {doc_path} not found", file=sys.stderr)
        return 1

    doc_keys = _read_doc_keys_from_file(doc_path)
    code_keys = _get_code_keys()

    if not code_keys:
        print("Error: Could not extract structural keys from code", file=sys.stderr)
        return 1

    issues = _check_consistency(doc_keys, code_keys)

    if not issues:
        print("✓ Grammar documentation is complete and consistent")
        return 0

    # Report issues
    print(f"✗ Found {len(issues)} grammar documentation issue(s):\n")
    for issue in sorted(issues, key=lambda x: (x.key, x.issue)):
        if issue.issue == "missing_from_docs":
            print(f"  Missing from docs: domain '{issue.key}' exists in code but not documented")
        elif issue.issue == "undocumented":
            print(f"  Undocumented: key '{issue.key}' exists in code but not documented")
        elif issue.issue == "extra_in_docs":
            print(f"  Extra in docs: key '{issue.key}' is documented but not in code")
        elif issue.issue == "extra_domain_in_docs":
            print(f"  Extra domain in docs: '{issue.key}' is documented but doesn't exist in code")

    return 1


if __name__ == "__main__":
    sys.exit(main())
