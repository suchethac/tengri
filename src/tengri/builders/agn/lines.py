# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Suchetha Cooray

"""AGN emission-lines sub-block factories.

Variants from :data:`tengri.parameters.groups._VALID_AGN_LINES_TYPES`:
``none``, ``blr``, ``nlr``, ``grahsp``, ``qsogen``. Shared params
under ``agn.lines`` partition include ``blr_cf``, ``nlr_cf``,
``alpha_ion``, ``feltre_cf``.
"""

from __future__ import annotations

from tengri.builders.agn._sub import build_axis_factories
from tengri.parameters.groups import _VALID_AGN_LINES_TYPES

_FACTORIES = build_axis_factories(
    axis="lines",
    variants=_VALID_AGN_LINES_TYPES,
    representative_variant="nlr",
)
globals().update(_FACTORIES)


def available() -> list[str]:
    """Return the list of AGN emission-lines variants exposed by this module."""
    return sorted(_FACTORIES)


__all__ = ["available", *sorted(_FACTORIES)]
