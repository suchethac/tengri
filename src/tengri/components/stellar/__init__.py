# SPDX-License-Identifier: BSD-3-Clause
"""Stellar SEDComponent: star formation history and population synthesis.

Converts star formation history parameters (age, metallicity, mass,
dust extinction) into rest-frame optical and UV specific luminosity L_nu
using DSPS stellar population synthesis grids. Publishes derived
quantities (current SFR, stellar mass) to downstream components.
"""

from __future__ import annotations

from tengri.components.stellar.component import (
    StellarSEDComponent,
    StellarSEDComponentConfig,
    StellarSEDComponentState,
)

__all__ = [
    "StellarSEDComponent",
    "StellarSEDComponentConfig",
    "StellarSEDComponentState",
]
