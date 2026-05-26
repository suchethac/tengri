# SPDX-License-Identifier: BSD-3-Clause
"""X-ray emission: XRBs (HMXB + LMXB) + AGN corona."""

from tengri.components.xray.xray import (
    alpha_ox_from_l2500,
    xray_agn_corona,
    xray_agn_corona_bolometric,
    xray_agn_corona_from_disc,
    xray_anisotropy,
    xray_total,
    xray_xrb,
)
from tengri.components.xray.xray_model import (
    XRayAirdSEDComponent,
    XRayAirdSEDComponentConfig,
)

__all__ = [
    "XRayAirdSEDComponent",
    "XRayAirdSEDComponentConfig",
    "alpha_ox_from_l2500",
    "xray_agn_corona",
    "xray_agn_corona_bolometric",
    "xray_agn_corona_from_disc",
    "xray_anisotropy",
    "xray_total",
    "xray_xrb",
]
