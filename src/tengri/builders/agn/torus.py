# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Suchetha Cooray

"""AGN torus sub-block factories.

Variants from ``tengri.parameters.groups._VALID_AGN_TORUS_TYPES``.
All variants share the same param partition under ``agn.torus``
(``T_torus``, ``tau_skirtor``, ``frac_hot``, etc.); the variant
string selects the physics model.
"""

from __future__ import annotations

from tengri.builders.agn._sub import build_axis_factories
from tengri.parameters.groups import _VALID_AGN_TORUS_TYPES

_FACTORIES = build_axis_factories(
    axis="torus",
    variants=_VALID_AGN_TORUS_TYPES,
    representative_variant="skirtor",
)
globals().update(_FACTORIES)


def available() -> list[str]:
    """Return the list of AGN-torus variants exposed by this module."""
    return sorted(_FACTORIES)


__all__ = ["available", *sorted(_FACTORIES)]
