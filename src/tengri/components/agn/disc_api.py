"""AGN disc models (powerlaw, multicolor, K&D, ADAF, qsogen).

This module is a re-export grouping. The actual implementations live in
`tengri.components.agn._disc`, `tengri.components.agn.disc`, and
`tengri.components.agn.qsogen`. Importing from here is the canonical
physics-grouped path: `from tengri.components.agn.disc_api import ...`.

For backwards compatibility, use `from tengri.components.agn import ...`
to access these symbols.
"""

from tengri.components.agn.disc import (
    adaf_disc,
    beloborodov_gamma_hot,
    compute_l2500,
    kubota_done_disc,
    multicolor_disc,
    powerlaw_disc,
)
from tengri.components.agn.qsogen import compute_qsogen_sed, qsogen

__all__ = [
    "adaf_disc",
    "beloborodov_gamma_hot",
    "compute_l2500",
    "compute_qsogen_sed",
    "kubota_done_disc",
    "multicolor_disc",
    "powerlaw_disc",
    "qsogen",
]
