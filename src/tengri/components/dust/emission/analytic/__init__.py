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
from tengri.components.dust.emission.analytic.greybody import (
    GreybodyIRSEDComponent,
)
from tengri.components.dust.emission.analytic.modified_blackbody import (
    ModifiedBlackbodyIRSEDComponent,
)
from tengri.components.dust.emission.analytic.pah_drude import (
    PAHDrudeIRSEDComponent,
)
from tengri.components.dust.emission.analytic.schreiber2016 import (
    Schreiber2016AnalyticIRSEDComponent,
)

__all__ = [
    "Casey2012IRSEDComponent",
    "EnergyBalanceSplitIRSEDComponent",
    "GreybodyIRSEDComponent",
    "ModifiedBlackbodyIRSEDComponent",
    "PAHDrudeIRSEDComponent",
    "Schreiber2016AnalyticIRSEDComponent",
]
