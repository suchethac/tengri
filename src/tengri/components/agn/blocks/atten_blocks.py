# SPDX-License-Identifier: BSD-3-Clause
"""Re-export attenuation-stage blocks from per-model modules.

This module re-exports the polar-dust attenuation block and its graybody
reemission helper from their per-model module (``polar_dust_atten``).
Importing this module side-effects the registration of the attenuation block.
"""

from __future__ import annotations

from tengri.components.agn.blocks.polar_dust_atten import (
    polar_dust_attenuation_block as polar_dust_attenuation_block,
    polar_dust_reemission_lnu as polar_dust_reemission_lnu,
)

__all__ = [
    "polar_dust_attenuation_block",
    "polar_dust_reemission_lnu",
]
