# SPDX-License-Identifier: BSD-3-Clause
"""Grid/template dust-emission SEDModelComponents.

This package provides SEDModelComponents for tabulated dust IR
emission templates. Each model wraps a closure from the parent `emission`
module and registers itself for automatic discovery in the SEDModelComponent
registry.

Available models (auto-registered):

- ``dale2014`` — Dale et al. (2014) 1-parameter template
- ``dale2014_cigale`` — CIGALE variant of Dale et al. (2014)
- ``draine_li2007`` — Draine & Li (2007) 3-parameter template
- ``draine_li2014`` — Draine & Li (2014) 4-parameter template
- ``astrodust`` — Hensley & Draine (2023) Astrodust+PAH
- ``bosa`` — Boquien & Salim (2021) (L_TIR, sSFR)-parameterized
- ``themis`` — Jones et al. (2017) THEMIS/DustEM
- ``schreiber2018`` — Schreiber et al. (2018) dust emission templates
- ``dh02_ce01`` — Dale & Helou (2002) + Chary & Elbaz (2001) cold dust

Notes
-----
Each component auto-loads its HDF5 templates on first instantiation. The
closures are imported lazily in the `predict` method to avoid circular imports
and unnecessary template loading.
"""

from __future__ import annotations

from tengri.components.dust.emission.templates.astrodust import (
    AstrodustIRSEDComponent,
)
from tengri.components.dust.emission.templates.bosa import BosaIRSEDComponent
from tengri.components.dust.emission.templates.dale import (
    Dale2014CigaleIRSEDComponent,
    Dale2014IRSEDComponent,
)
from tengri.components.dust.emission.templates.dh02_ce01 import (
    DH02CE01IRSEDComponent,
)
from tengri.components.dust.emission.templates.draine_li import (
    DraineLi2007IRSEDComponent,
    DraineLi2014IRSEDComponent,
)
from tengri.components.dust.emission.templates.schreiber2018 import (
    Schreiber2018IRSEDComponent,
)
from tengri.components.dust.emission.templates.themis import ThemisIRSEDComponent

__all__ = [
    "AstrodustIRSEDComponent",
    "BosaIRSEDComponent",
    "DH02CE01IRSEDComponent",
    "Dale2014CigaleIRSEDComponent",
    "Dale2014IRSEDComponent",
    "DraineLi2007IRSEDComponent",
    "DraineLi2014IRSEDComponent",
    "Schreiber2018IRSEDComponent",
    "ThemisIRSEDComponent",
]
