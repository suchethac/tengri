"""Dust attenuation laws (Calzetti, Cardelli, SMC, LMC, etc.).

This module is a re-export grouping. The actual implementations live in
`tengri.components.dust.attenuation`. Importing from here is the canonical
physics-grouped path:
`from tengri.components.dust.attenuation_models import ...`.

Provides single-component and two-component dust attenuation models with
configurable reddening laws and obscuration fractions.

For backwards compatibility, use `from tengri.components.dust import ...`
to access these symbols.
"""

from tengri.components.dust.attenuation import (
    calzetti,
    cardelli,
    d03_mwrv31,
    hd23_mwrv31,
    li08,
    lmc,
    prevot_smc,
    register_dust_law,
    resolve_dust_law,
    single_component_dust,
    single_component_dust_fast,
    smc,
    two_component_dust,
    two_component_dust_fast,
    vw07_bc,
    vw07_diff,
    wd01_mwrv31,
    wg00_cloudy,
    wg00_dusty,
    wg00_shell,
)

__all__ = [
    "calzetti",
    "cardelli",
    "d03_mwrv31",
    "hd23_mwrv31",
    "li08",
    "lmc",
    "prevot_smc",
    "register_dust_law",
    "resolve_dust_law",
    "single_component_dust",
    "single_component_dust_fast",
    "smc",
    "two_component_dust",
    "two_component_dust_fast",
    "vw07_bc",
    "vw07_diff",
    "wd01_mwrv31",
    "wg00_cloudy",
    "wg00_dusty",
    "wg00_shell",
]
