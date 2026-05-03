"""AGN narrow-line region (NLR) and broad-line region (BLR) emission.

This module is a re-export grouping of NLR and BLR helpers. The actual
implementations live in `tengri.components.agn.nlr` and
`tengri.components.agn.blr`. Importing from here is the canonical
physics-grouped path: `from tengri.components.agn.lines import ...`.

For backwards compatibility, use `from tengri.components.agn import ...`
to access these symbols.
"""

from tengri.components.agn.blr import compute_blr_sed
from tengri.components.agn.nlr import (
    compute_nlr_sed,
    compute_nlr_sed_richardson2014,
)

__all__ = [
    "compute_blr_sed",
    "compute_nlr_sed",
    "compute_nlr_sed_richardson2014",
]
