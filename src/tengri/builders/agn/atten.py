# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Suchetha Cooray

"""AGN polar-dust attenuation sub-block factories.

Variants from ``tengri.parameters.groups._VALID_AGN_ATTEN_TYPES``:
``none``, ``smc_prevot``, ``polar_dust``, ``grahsp_biatten``,
``qsogen_smc``. Shared params under ``agn.atten``: ``polar_ebv``,
``polar_oa``.
"""

from __future__ import annotations

from tengri.builders.agn._sub import build_axis_factories
from tengri.parameters.groups import _VALID_AGN_ATTEN_TYPES

_FACTORIES = build_axis_factories(
    axis="atten",
    variants=_VALID_AGN_ATTEN_TYPES,
    representative_variant="smc_prevot",
)
globals().update(_FACTORIES)


def available() -> list[str]:
    """Return the list of AGN-attenuation variants exposed by this module."""
    return sorted(_FACTORIES)


__all__ = ["available", *sorted(_FACTORIES)]
