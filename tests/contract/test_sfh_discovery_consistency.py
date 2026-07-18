# SPDX-License-Identifier: BSD-3-Clause
"""Contract: the two SFH discovery surfaces must differ by exactly one known set.

``tengri.list_sfh_models()`` reports every *registered* SFH type (34), while
``builders.sfh.available()`` reports only the *buildable* ones (26). Two numbers
for "what SFH models are there?" looks like drift, so the difference is pinned
here rather than left to be rediscovered: it must be exactly
``UNVALIDATED_SFH_TYPES`` — types registered but not yet wired into the DSPS
forward path, which ``SEDModel.build`` rejects with a specific error.

Audit note (#1210): the surfaces were reported as inconsistent. They are not —
``list_sfh_models()`` marks the extras ``status='unvalidated'`` in a prominent
column and documents the semantics, and the build error names the reason. What
*was* wrong was ``available()``'s docstring, which claimed to mirror the
canonical registry keys while silently excluding eight of them.
"""

from __future__ import annotations

import pytest

from tengri import builders, list_sfh_models
from tengri.components.stellar.sfh.registry import SFH_REGISTRY, UNVALIDATED_SFH_TYPES

pytestmark = pytest.mark.contract


def test_available_is_registry_minus_unvalidated():
    """The gap between the two surfaces is exactly the unvalidated set.

    Neuter-check: drop a name from ``UNVALIDATED_SFH_TYPES`` without wiring it
    into the forward path and this goes red, rather than the discovery surfaces
    quietly disagreeing by one more model.
    """
    registered = set(SFH_REGISTRY)
    buildable = set(builders.sfh.available())
    assert registered - buildable == set(UNVALIDATED_SFH_TYPES), (
        "builders.sfh.available() no longer differs from SFH_REGISTRY by exactly "
        "UNVALIDATED_SFH_TYPES — the two discovery surfaces have drifted"
    )
    assert not buildable - registered, "available() surfaces a name absent from the registry"


def test_list_sfh_models_marks_the_unvalidated_ones():
    """Every type absent from ``available()`` is labelled, not silently listed.

    This is what keeps the larger count honest: a user scanning the table sees
    ``unvalidated`` beside the name rather than assuming all 34 are usable.
    """
    rows = {row["name"]: row for row in list_sfh_models()}
    for name in UNVALIDATED_SFH_TYPES:
        assert name in rows, f"{name} is registered but missing from list_sfh_models()"
        assert rows[name]["status"] == "unvalidated", (
            f"{name} cannot be built but is advertised as {rows[name]['status']!r}"
        )


def test_buildable_models_are_not_marked_unvalidated():
    """The complement: nothing buildable is scared off with a false label."""
    rows = {row["name"]: row for row in list_sfh_models()}
    mislabelled = [
        n for n in builders.sfh.available() if rows.get(n, {}).get("status") == "unvalidated"
    ]
    assert not mislabelled, f"buildable models labelled unvalidated: {mislabelled}"
