# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Suchetha Cooray

"""AGN disc sub-block factories.

Variants from ``tengri.parameters.groups._VALID_AGN_DISC_TYPES``:
``none``, ``powerlaw``, ``multicolor``, ``kubota_done``, ``adaf``,
``qsogen``, ``grahsp_sbpl``. The disc-shaping parameters
(``log_mbh``, ``a_spin``, ``alpha``, …) live in the top-level
``agn`` partition, not in ``agn.disc``, so individual disc factories
expose only the variant selector itself — pass those parameters to
``builders.agn.composable(...)`` directly.
"""

from __future__ import annotations

from tengri.builders.agn._sub import build_axis_factories
from tengri.parameters.groups import _VALID_AGN_DISC_TYPES

_FACTORIES = build_axis_factories(
    axis="disc",
    variants=_VALID_AGN_DISC_TYPES,
    representative_variant="powerlaw",
)
globals().update(_FACTORIES)


def available() -> list[str]:
    """Return the list of AGN-disc variants exposed by this module."""
    return sorted(_FACTORIES)


__all__ = ["available", *sorted(_FACTORIES)]
