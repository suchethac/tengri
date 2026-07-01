# SPDX-License-Identifier: BSD-3-Clause
"""Analytic dust emission SEDModelComponent ports.

Auto-registers the 4 analytic dust emission templates as SEDModelComponent
subclasses when imported.
"""

from tengri.components.dust.emission.analytic.casey2012 import (
    Casey2012IRSEDComponent,
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
    "ModifiedBlackbodyIRSEDComponent",
    "PAHDrudeIRSEDComponent",
    "Schreiber2016AnalyticIRSEDComponent",
]
