# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Suchetha Cooray

"""AGN narrow-line region (NLR) sub-block factories.

Variants from ``tengri.parameters.groups._VALID_AGN_NLR_TYPES``:
``none``, ``analytic``, ``synthesizer``, ``synthesizer_spectra``, ``grahsp``.
Shared params under ``agn.nlr`` partition include ``nlr_cf``,
``alpha_ion``.
"""

from __future__ import annotations

from tengri.builders.agn._sub import build_axis_factories
from tengri.parameters.groups import _VALID_AGN_NLR_TYPES

_FACTORIES = build_axis_factories(
    axis="nlr",
    variants=_VALID_AGN_NLR_TYPES,
    representative_variant="analytic",
)
globals().update(_FACTORIES)


def available() -> list[str]:
    """Return the list of AGN narrow-line region variants exposed by this module."""
    return sorted(_FACTORIES)


__all__ = ["available", *sorted(_FACTORIES)]
