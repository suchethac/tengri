# SPDX-License-Identifier: BSD-3-Clause
"""Free-parameter declarations owned by the AGN X-ray corona component.

Single source of truth for the ``agn_xray_*`` priors.
:meth:`AGNXRayCoronaSEDComponent.declared_parameters` returns these
directly via the base class auto-discovery from class-level Distribution
attributes. This module provides the registry with ParamDeclaration
entries so that introspection functions like :func:`tengri.describe_parameter`
can find them.
"""

from __future__ import annotations

from tengri.parameters.priors import Fixed, Uniform
from tengri.protocols.component import ParamDeclaration

PARAMS: tuple[ParamDeclaration, ...] = (
    ParamDeclaration(
        "agn_xray_gamma",
        Uniform(1.4, 2.4, default=1.9),
        "X-ray photon index",
        units="dimensionless",
    ),
    ParamDeclaration(
        "agn_xray_delta_alpha_ox",
        Fixed(0.0),
        "Offset on Just+2007 alpha_ox(L_2500)",
        units="dex",
    ),
    ParamDeclaration(
        "agn_xray_e_cut",
        Fixed(300.0),
        "High-energy cutoff",
        units="keV",
    ),
)

__all__ = ["PARAMS"]
