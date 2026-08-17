# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Suchetha Cooray

"""AGN FeII sub-block factories.

Variants from ``tengri.parameters.groups._VALID_AGN_FEII_TYPES``:
``none``, ``grahsp``, ``qsogen_balmer``. Shared param under
``agn.feii``: ``fe2_strength``.
"""

from __future__ import annotations

from tengri.builders.agn._sub import build_axis_factories
from tengri.parameters.groups import _VALID_AGN_FEII_TYPES

_FACTORIES = build_axis_factories(
    axis="feii",
    variants=_VALID_AGN_FEII_TYPES,
    representative_variant="grahsp",
)
globals().update(_FACTORIES)


def available() -> list[str]:
    """Return the list of AGN-FeII variants exposed by this module."""
    return sorted(_FACTORIES)


__all__ = ["available", *sorted(_FACTORIES)]
