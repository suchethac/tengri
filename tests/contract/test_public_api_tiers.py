# SPDX-License-Identifier: BSD-3-Clause
"""``tengri.__all__`` is organized into tiers, and the tiers must stay honest.

Phase 6 (2026-07) grouped the flat ~120-name ``__all__`` into commented tiers
(core / physics submodules / toolkit / introspection / exceptions / layer
modules) so a reader can see the shape of the public surface instead of an
alphabetical wall. The tiers are *comments*, which means nothing stops a later
edit from appending a name below the last tier header and quietly leaving it
out of the documented structure.

These tests assert the properties the comments claim:

* every exported name resolves (a typo in ``__all__`` breaks ``import *`` and
  Sphinx, but nothing else notices);
* no name is listed twice;
* the tier blocks *partition* ``__all__`` — every name sits under exactly one
  tier header, so the documented map is complete.

The *membership* of the public surface is deliberately NOT re-asserted here.
Two contract tests already own that baseline
(``test_public_api_surface.py::test_all_is_within_allowed_top_level`` and
``test_public_surface.py::test_all_matches_expected``). A third copy would be a
maintenance trap: three lists to update, and nothing to say which is right when
they disagree.
"""

from __future__ import annotations

import pathlib
import re

import pytest

import tengri

pytestmark = pytest.mark.contract

_INIT = pathlib.Path(tengri.__file__)
_SOURCE = _INIT.read_text()

# A tier header looks like:  # ========== Tier 1: CORE (...) ==========
_TIER_HEADER = re.compile(r"^\s*#\s*=+\s*(.+?)\s*=+\s*$")


def _all_block_lines() -> list[str]:
    """The raw source lines of the ``__all__ = [...]`` literal."""
    lines = _SOURCE.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("__all__ = ["))
    depth = 0
    for i in range(start, len(lines)):
        depth += lines[i].count("[") - lines[i].count("]")
        if depth == 0:
            return lines[start : i + 1]
    raise AssertionError("unterminated __all__ literal")


def _names_by_tier() -> dict[str, list[str]]:
    """Map each tier header to the names listed beneath it."""
    tiers: dict[str, list[str]] = {}
    current: str | None = None
    for line in _all_block_lines():
        header = _TIER_HEADER.match(line)
        if header:
            current = header.group(1)
            tiers.setdefault(current, [])
            continue
        name = re.fullmatch(r'\s*"([A-Za-z_]\w*)",\s*', line)
        if name and current is not None:
            tiers[current].append(name.group(1))
    return tiers


def test_every_exported_name_resolves():
    """A typo in ``__all__`` breaks ``import *`` and Sphinx, silently."""
    missing = [name for name in tengri.__all__ if not hasattr(tengri, name)]
    assert not missing, (
        f"names in tengri.__all__ that do not exist on the package: {missing}. "
        "Either import them in __init__.py or remove them from __all__."
    )


def test_no_name_is_exported_twice():
    duplicates = sorted({n for n in tengri.__all__ if tengri.__all__.count(n) > 1})
    assert not duplicates, f"duplicated in tengri.__all__: {duplicates}"


def test_the_tiers_partition_the_public_surface():
    """Every exported name sits under exactly one tier header.

    A name appended after the last tier header, or above the first one, is
    exported but undocumented — the tier map would be a lie.
    """
    tiers = _names_by_tier()
    tiered = [name for names in tiers.values() for name in names]

    orphans = sorted(set(tengri.__all__) - set(tiered))
    assert not orphans, (
        f"exported but not under any tier header: {orphans}. Put each name "
        "under the tier it belongs to in src/tengri/__init__.py."
    )

    phantom = sorted(set(tiered) - set(tengri.__all__))
    assert not phantom, f"listed under a tier but not exported: {phantom}"

    doubled = sorted({n for n in tiered if tiered.count(n) > 1})
    assert not doubled, f"listed under more than one tier: {doubled}"


def test_the_tier_parser_is_not_vacuous():
    """Guard the guard: a regex that matches nothing would pass everything."""
    tiers = _names_by_tier()
    assert len(tiers) >= 4, (
        f"only {len(tiers)} tier headers parsed — the header regex has probably "
        "stopped matching, so test_the_tiers_partition_the_public_surface proves nothing."
    )
    assert len(tengri.__all__) >= 100, "public surface unexpectedly small"
    # The partition test must be able to FAIL: an orphan name is detectable.
    tiered = {name for names in tiers.values() for name in names}
    assert "SEDModel" in tiered, "the core tier should contain SEDModel"
