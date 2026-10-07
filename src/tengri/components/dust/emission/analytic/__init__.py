# SPDX-License-Identifier: BSD-3-Clause
"""Analytic dust emission SEDModelComponents.

Auto-registers the analytic dust emission templates as SEDModelComponent
subclasses when imported.
"""

from tengri.components.dust.emission.analytic.casey2012 import (
    Casey2012IRSEDComponent,
)
from tengri.components.dust.emission.analytic.energy_balance_split import (
    EnergyBalanceSplitIRSEDComponent,
)
from tengri.components.dust.emission.analytic.graybody import (
    GraybodyIRSEDComponent,
)
from tengri.components.dust.emission.analytic.modified_blackbody import (
    ModifiedBlackbodyIRSEDComponent,
)
from tengri.components.dust.emission.analytic.pah_drude import (
    PAHDrudeIRSEDComponent,
)

__all__ = [
    "Casey2012IRSEDComponent",
    "EnergyBalanceSplitIRSEDComponent",
    "GraybodyIRSEDComponent",
    "ModifiedBlackbodyIRSEDComponent",
    "PAHDrudeIRSEDComponent",
]
