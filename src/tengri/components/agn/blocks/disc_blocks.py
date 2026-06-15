# SPDX-License-Identifier: BSD-3-Clause
"""Re-export disc blocks from per-model modules.

This module imports and re-exports individual disc block implementations
(multicolor, Kubota-Done, ADAF, CIGALE variants, Slone & Netzer, RELAGN,
Richards2006) from their respective per-model modules. Importing this module
side-effects the registration of all disc blocks.
"""

from __future__ import annotations

from tengri.components.agn.blocks._disc_common import (
    _C_AA_PER_S as _C_AA_PER_S,
    _L_SUN_ERG as _L_SUN_ERG,
    _cigale_disc_lambda as _cigale_disc_lambda,
)
from tengri.components.agn.blocks.adaf_disc import (
    adaf_disc_block as adaf_disc_block,
)
from tengri.components.agn.blocks.cigale_adaf_disc import (
    cigale_adaf_disc_block as cigale_adaf_disc_block,
)
from tengri.components.agn.blocks.cigale_schartmann_disc import (
    cigale_schartmann_disc_block as cigale_schartmann_disc_block,
)
from tengri.components.agn.blocks.cigale_schartmann_skirtor_attenuated_disc import (
    cigale_schartmann_skirtor_attenuated_disc_block,
)
from tengri.components.agn.blocks.cigale_skirtor_disc import (
    cigale_skirtor_disc_block as cigale_skirtor_disc_block,
)
from tengri.components.agn.blocks.kubota_done_disc import (
    kubota_done_disc_block as kubota_done_disc_block,
)
from tengri.components.agn.blocks.multicolor_disc import (
    multicolor_disc_block as multicolor_disc_block,
)
from tengri.components.agn.blocks.relagn_disc import (
    relagn_disc_block as relagn_disc_block,
)
from tengri.components.agn.blocks.richards2006_disc import (
    richards2006_disc_block as richards2006_disc_block,
)
from tengri.components.agn.blocks.slone_netzer_disc import (
    slone_netzer_disc_block as slone_netzer_disc_block,
)

__all__ = [
    "adaf_disc_block",
    "cigale_adaf_disc_block",
    "cigale_schartmann_disc_block",
    "cigale_schartmann_skirtor_attenuated_disc_block",
    "cigale_skirtor_disc_block",
    "kubota_done_disc_block",
    "multicolor_disc_block",
    "relagn_disc_block",
    "richards2006_disc_block",
    "slone_netzer_disc_block",
]
