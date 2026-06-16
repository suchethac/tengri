# SPDX-License-Identifier: BSD-3-Clause
"""Re-export FeII blocks from per-model modules.

This module re-exports the FeII pseudo-continuum block from its per-model
module (``boroson_green_feii``). Importing this module side-effects the
registration of the FeII block.
"""

from __future__ import annotations

from tengri.components.agn.blocks.boroson_green_feii import (
    DEFAULT_F_BOL_5100 as DEFAULT_F_BOL_5100,
    boroson_green_feii_block as boroson_green_feii_block,
)

__all__ = [
    "DEFAULT_F_BOL_5100",
    "boroson_green_feii_block",
]
