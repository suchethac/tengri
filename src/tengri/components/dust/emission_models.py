"""Dust infrared emission models (modified blackbody, Casey 2012, Dale 2014, etc.).

This module is a re-export grouping. The actual implementations live in
`tengri.components.dust.emission`. Importing from here is the canonical
physics-grouped path:
`from tengri.components.dust.emission_models import ...`.

Provides parametric and tabulated IR emission models with energy-balance
normalization to match absorbed UV/optical radiation.

For backwards compatibility, use `from tengri.components.dust import ...`
to access these symbols.
"""

from tengri.components.dust.emission import (
    astrodust,
    bosa,
    casey2012,
    compute_absorbed_luminosity,
    compute_absorbed_luminosity_from_tau,
    create_astrodust_from_grid,
    create_bosa_from_grid,
    create_dale2014_from_grid,
    create_dl07_from_grid,
    create_dl14_from_grid,
    create_themis_from_grid,
    dale2014,
    draine_li2007,
    draine_li2014,
    energy_balance_split,
    load_astrodust_templates,
    load_bosa_templates,
    load_dale2014_templates,
    load_dl14_templates,
    load_draine_li_templates,
    load_themis_templates,
    modified_blackbody,
    register_astrodust_tabulated,
    register_bosa_tabulated,
    register_dale2014_tabulated,
    register_dl07_tabulated,
    register_dl14_tabulated,
    register_themis_tabulated,
    themis,
)

__all__ = [
    "astrodust",
    "bosa",
    "casey2012",
    "compute_absorbed_luminosity",
    "compute_absorbed_luminosity_from_tau",
    "create_astrodust_from_grid",
    "create_bosa_from_grid",
    "create_dale2014_from_grid",
    "create_dl07_from_grid",
    "create_dl14_from_grid",
    "create_themis_from_grid",
    "dale2014",
    "draine_li2007",
    "draine_li2014",
    "energy_balance_split",
    "load_astrodust_templates",
    "load_bosa_templates",
    "load_dale2014_templates",
    "load_dl14_templates",
    "load_draine_li_templates",
    "load_themis_templates",
    "modified_blackbody",
    "register_astrodust_tabulated",
    "register_bosa_tabulated",
    "register_dale2014_tabulated",
    "register_dl07_tabulated",
    "register_dl14_tabulated",
    "register_themis_tabulated",
    "themis",
]
