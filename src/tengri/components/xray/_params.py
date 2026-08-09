# SPDX-License-Identifier: BSD-3-Clause
"""Free-parameter declarations owned by the X-ray component.

Single source of truth for the ``xray_*`` priors.
``tengri.parameters._builders`` derives its legacy ``_XRAY_PARAMS``
bucket from this tuple, and :meth:`XRaySEDComponent.declared_parameters`
returns it directly. Drift between the two paths is structurally
impossible because they share the same in-memory list.
"""

from __future__ import annotations

from tengri.parameters.priors import Fixed, Uniform
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
        "xray_delta_alpha_ox",
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
        units="keV",
    ),
    ParamDeclaration(
        "xray_log_nh",
        Fixed(20.0),
        "Line-of-sight equivalent hydrogen column density [log10(cm^-2)]. "
        "Typical range 20 (unobscured) to 24 (Compton-thick). Controls "
        "photoelectric absorption below ~2 keV (Morrison & McCammon 1983).",
        lambda lo, hi: 0 <= lo <= 26,
        "must be in [0, 26]",
        # The [0, 26] bound is what the absorption model tolerates, not a
        # sensible prior: it would put most of the mass on columns no galaxy
        # has. Free over the range this declaration itself calls typical —
        # 20 (unobscured) to 24 (Compton-thick).
        free_prior=Uniform(
            20.0, 24.0, "Line-of-sight hydrogen column", units="log10(cm^-2)", default=20.0
        ),
        units="log10(cm^-2)",
    ),
    ParamDeclaration(
        "xray_alpha_irx",
        Fixed(0.3),
        "alpha_IRX = log10(nu*Lnu(12um) / Lx(2-10 keV)) for the lopez24 corona "
        "(Asmus+2015 / Lopez+2024). Higher -> fainter X-ray. Ignored by yang20. "
        "Typical range 0.0-0.6.",
    ),
    # X-ray binary offsets (xray_aird component, #1307)
    ParamDeclaration(
        "xray_det_hmxb",
        Uniform(
            -2.0,
            2.0,
            default=0.0,
        ),
        "Deviation from expected HMXB log L_X (Yang+2020 [1]_). "
        "Positive = brighter X-ray. Allows intrinsic scatter or evolution "
        "around the Lehmer+2016 SFR relation.",
        units="dex",
    ),
    ParamDeclaration(
        "xray_det_lmxb",
        Uniform(
            -2.0,
            2.0,
            default=0.0,
        ),
        "Deviation from expected LMXB log L_X (Yang+2020 [1]_). "
        "Positive = brighter X-ray. Allows intrinsic scatter or evolution "
        "around the Lehmer+2016 age/mass relation.",
        units="dex",
    ),
)

__all__ = ["PARAMS"]
