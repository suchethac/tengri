# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Suchetha Cooray

"""AGN polar-dust attenuation sub-block factories.

Variants from ``tengri.parameters.groups._VALID_AGN_ATTEN_TYPES``:
``none``, ``smc_prevot``, ``polar_dust``, ``grahsp_biatten``,
``qsogen_smc``. Shared params under ``agn.atten``: ``polar_ebv``,
``polar_oa``.
"""

from __future__ import annotations

from typing import Any

from tengri.builders.agn._sub import build_axis_factories
from tengri.parameters.groups import _VALID_AGN_ATTEN_TYPES

_FACTORIES = build_axis_factories(
    axis="atten",
    variants=_VALID_AGN_ATTEN_TYPES,
    representative_variant="smc_prevot",
)

# Special handling for smc_prevot: wrap it to emit law key instead of type
_original_smc_prevot = _FACTORIES["smc_prevot"]


def smc_prevot(**kwargs: Any) -> dict:
    """AGN atten sub-block: smc_prevot (via law='prevot_smc').

    This factory wraps the smc_prevot attenuation model to emit the new
    law-based grammar instead of the deprecated type-based form.
    """
    # Call the original factory
    output = _original_smc_prevot(**kwargs)

    # Convert type='smc_prevot' to law='prevot_smc'
    output.pop("type", None)
    output["law"] = "prevot_smc"

    return output


# Update the factories dict with the wrapped version
_FACTORIES["smc_prevot"] = smc_prevot
globals().update(_FACTORIES)


def available() -> list[str]:
    """Return the list of AGN-attenuation variants exposed by this module."""
    return sorted(_FACTORIES)


__all__ = ["available", *sorted(_FACTORIES)]
