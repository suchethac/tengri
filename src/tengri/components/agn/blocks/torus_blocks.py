# SPDX-License-Identifier: BSD-3-Clause
"""Re-export torus blocks from per-model modules.

This module imports and re-exports individual torus block implementations
(Nenkova, SKIRTOR, Silva04, CAT3D-wind, SKIRTOR_mean_3p, Fritz) from
their respective per-model modules. Importing this module side-effects
the registration of all torus blocks.
"""

from __future__ import annotations

from tengri.components.agn.blocks.cat3d_wind_torus import (
    cat3d_wind_torus_block as cat3d_wind_torus_block,
)
from tengri.components.agn.blocks.fritz_torus import (
    fritz_torus_block as fritz_torus_block,
)
from tengri.components.agn.blocks.nenkova_torus import (
    nenkova_torus_block as nenkova_torus_block,
)
from tengri.components.agn.blocks.silva04_torus import (
    silva04_torus_block as silva04_torus_block,
)
from tengri.components.agn.blocks.skirtor_agnfitter_torus import (
    skirtor_agnfitter_torus_block as skirtor_agnfitter_torus_block,
)
from tengri.components.agn.blocks.skirtor_torus import (
    skirtor_torus_block as skirtor_torus_block,
)

__all__ = [
    "cat3d_wind_torus_block",
    "fritz_torus_block",
    "nenkova_torus_block",
    "silva04_torus_block",
    "skirtor_agnfitter_torus_block",
    "skirtor_torus_block",
]
