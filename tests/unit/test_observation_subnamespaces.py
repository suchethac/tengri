# SPDX-License-Identifier: BSD-3-Clause
"""Binding-identity guard for tengri.observation sub-namespaces.

Phase E added :mod:`tengri.observation.containers`,
:mod:`tengri.observation.physics`, and
:mod:`tengri.observation.constants` as additive sub-namespaces. Each
re-exports a subset of the flat ``tengri.observation`` surface.

This test pins two invariants:

1. Every name listed in a sub-namespace's ``__all__`` resolves to an
   object that is :keyword:`is`-identical to the corresponding name on
   the flat ``tengri.observation`` surface (no drift, no copy).

2. The sub-namespaces are jointly exhaustive over the flat surface's
   ``__all__`` minus the sub-namespace module names themselves —
   guaranteeing no flat-surface symbol is silently orphaned outside
   any sub-namespace.

If a future PR adds a new symbol to ``tengri.observation.__all__``,
this test forces a deliberate decision about which sub-namespace it
belongs to.
"""

from __future__ import annotations

import pytest

import tengri.observation as flat
from tengri.observation import constants, containers, physics


@pytest.mark.parametrize("subns", [containers, physics, constants])
def test_subnamespace_bindings_match_flat_surface(subns):
    """Every sub-namespace name resolves to the same object on the flat surface."""
    for name in subns.__all__:
        sub_obj = getattr(subns, name)
        flat_obj = getattr(flat, name, None)
        assert flat_obj is not None, (
            f"{name} is exported from tengri.observation.{subns.__name__.split('.')[-1]} "
            f"but missing from flat tengri.observation surface."
        )
        assert sub_obj is flat_obj, (
            f"{name} differs between tengri.observation.{subns.__name__.split('.')[-1]} "
            f"and tengri.observation flat surface — re-export drift detected."
        )


def test_subnamespaces_jointly_cover_flat_all():
    """Every name in tengri.observation.__all__ is in exactly one sub-namespace."""
    sub_names = {"constants", "containers", "physics"}
    flat_names = set(flat.__all__) - sub_names

    union = set(containers.__all__) | set(physics.__all__) | set(constants.__all__)

    orphaned = flat_names - union
    assert not orphaned, (
        f"Flat tengri.observation surface has names not covered by any sub-namespace: "
        f"{sorted(orphaned)}. Add them to containers/physics/constants."
    )

    # No name should appear in more than one sub-namespace.
    seen: dict[str, str] = {}
    for label, names in [
        ("containers", containers.__all__),
        ("physics", physics.__all__),
        ("constants", constants.__all__),
    ]:
        for name in names:
            if name in seen:
                pytest.fail(
                    f"{name} appears in both {seen[name]} and {label} — "
                    f"sub-namespaces should be disjoint."
                )
            seen[name] = label


def test_subnamespaces_attached_to_observation():
    """The sub-namespaces are accessible via the flat tengri.observation namespace."""
    assert flat.containers is containers
    assert flat.physics is physics
    assert flat.constants is constants
