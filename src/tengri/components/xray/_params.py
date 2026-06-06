# SPDX-License-Identifier: BSD-3-Clause
"""Free-parameter declarations owned by the X-ray component.

Single source of truth for the ``xray_*`` priors.
``tengri.parameters._param_defs`` derives its legacy ``_XRAY_PARAMS``
bucket from this tuple, and :meth:`XRaySEDComponent.declared_parameters`
returns it directly. Drift between the two paths is structurally
impossible because they share the same in-memory list.
"""

from __future__ import annotations

from tengri.parameters.priors import Fixed
from tengri.protocols.component import ParamDeclaration

PARAMS: tuple[ParamDeclaration, ...] = (
    ParamDeclaration(
        "xray_gamma_agn",
        Fixed(1.8),
        "AGN X-ray photon index Gamma (typical 1.4-2.4)",
        lambda lo, hi: lo > 0,
        "must be > 0",
    ),
    ParamDeclaration(
        "xray_alpha_ox",
        Fixed(0.0),
        "Offset [dex] applied to the L_2500-derived alpha_ox (Just+2007,"
        " CIGALE convention). 0 (default) = pure empirical alpha_ox(L_2500);"
        " negative hardens the X-ray corona, positive softens it.",
    ),
    ParamDeclaration(
        "xray_gamma_hmxb",
        Fixed(2.0),
        "HMXB photon index (typical 2.0)",
        lambda lo, hi: lo > 0,
        "must be > 0",
    ),
    ParamDeclaration(
        "xray_gamma_lmxb",
        Fixed(1.6),
        "LMXB photon index (typical 1.6)",
        lambda lo, hi: lo > 0,
        "must be > 0",
    ),
    ParamDeclaration(
        "xray_E_cut",
        Fixed(300.0),
        "Exponential cutoff energy [keV] for AGN X-ray spectrum (typical 100-500)",
        lambda lo, hi: lo > 0,
        "must be > 0",
    ),
)

__all__ = ["PARAMS"]
