"""High-level AGN SED combiners (unified model builder).

This module is a re-export grouping. The actual implementations live in
`tengri.components.agn.unified`. Importing from here is the canonical
physics-grouped path: `from tengri.components.agn.compose import ...`.

These are the top-level functions that combine disc, torus, and line
emission into a complete AGN SED model.

For backwards compatibility, use `from tengri.components.agn import ...`
to access these symbols.
"""

from tengri.components.agn.unified import (
    adaf_agn,
    kubota_done_full_agn,
    unified_agn,
    unified_nlr_blr,
)

__all__ = [
    "adaf_agn",
    "kubota_done_full_agn",
    "unified_agn",
    "unified_nlr_blr",
]
