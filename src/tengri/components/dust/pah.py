"""Polycyclic aromatic hydrocarbon (PAH) emission helpers.

This module is a re-export grouping. The actual implementations live in
`tengri.components.dust.drude_profiles`. Importing from here is the
canonical physics-grouped path:
`from tengri.components.dust.pah import ...`.

Provides Drude profiles for PAH features and decomposition tools for
PAH template spectra.

For backwards compatibility, use `from tengri.components.dust import ...`
to access these symbols.
"""

from tengri.components.dust.drude_profiles import (
    N_PAH_FEATURES,
    SMITH2007_PAH_FEATURES,
    compute_pah_template,
    decompose_pah,
    drude_profile,
)

__all__ = [
    "N_PAH_FEATURES",
    "SMITH2007_PAH_FEATURES",
    "compute_pah_template",
    "decompose_pah",
    "drude_profile",
]
