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
        # AGNfitter-rX (Martinez-Ramirez et al. 2024) fits Δα_ox over ±0.4;
        # the intrinsic scatter of α_ox about the L_2500 relation is σ =
        # 0.146 dex (31 < log L < 32) and 0.131 dex (32 < log L < 33) (Just
        # et al. 2007, ApJ 665, 1004, Table 4 footnote, quoting Steffen et
        # al. 2006, AJ 131, 2826, Table 5), so Uniform(-0.4, 0.4) spans ≈3σ.
        free_prior=Uniform(
            -0.4, 0.4, "Offset applied to the L_2500-derived alpha_ox", default=0.0
        ),
    ),
    ParamDeclaration(
        "agn_xray_e_cut",
        Fixed(300.0),
        "High-energy cutoff",
        units="keV",
        # Matches its sibling ``xray_E_cut``, whose description states the
        # typical 100-500 keV interval; the two spellings must agree.
        free_prior=Uniform(100.0, 500.0, "AGN X-ray cutoff energy", units="keV", default=300.0),
    ),
)

__all__ = ["PARAMS"]
