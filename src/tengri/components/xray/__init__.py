# SPDX-License-Identifier: BSD-3-Clause
"""X-ray emission: XRBs (HMXB + LMXB) + AGN corona."""

from tengri.components.xray._models import (
    XRAY_MODELS,
    XRayRegistryEntry,
    register_xray_model,
)
from tengri.components.xray.xray import (
    alpha_ox_from_l2500,
    compton_scattering_transmission,
    pexrav_reflection,
    tbabs_transmission,
    xray_agn_corona,
    xray_agn_corona_from_disc,
    xray_anisotropy,
    xray_hotgas,
    xray_total,
    xray_xrb,
)
from tengri.components.xray.xray_model import (
    XRayAirdSEDComponent,
    XRayAirdSEDComponentConfig,
)

# Populate the runtime registry. ``'none'`` is the disable-toggle; new
# variants register here at module import. ``_VALID_XRAY_TYPES`` in
# ``parameters/groups.py`` derives from ``XRAY_MODELS.keys()``.
register_xray_model(
    "none",
    short_doc="Disable X-ray emission",
)(None)
register_xray_model(
    "simple",
    citation="Yang+2020 / X-CIGALE (MNRAS 491, 740)",
    short_doc="AGN corona via alpha_ox(L_2500) + Lehmer+16 XRBs + thermal hot gas",
)(xray_total)
# ``yang20`` is a CIGALE-compatible alias of ``simple``: the underlying
# tengri X-ray component already implements the Yang+2020 physics
# (alpha_ox corona + Morrison & McCammon 1983 N_H + Compton scattering +
# Lehmer+2016 XRBs). The alias closes the discoverability half of #440.
register_xray_model(
    "yang20",
    citation="Yang et al. 2020 (MNRAS 491, 4276)",
    short_doc="Alias of 'simple'; use this name for CIGALE pcigale.sed_modules.xray parity",
)(xray_total)

__all__ = [
    "XRAY_MODELS",
    "XRayAirdSEDComponent",
    "XRayAirdSEDComponentConfig",
    "XRayRegistryEntry",
    "alpha_ox_from_l2500",
    "compton_scattering_transmission",
    "pexrav_reflection",
    "register_xray_model",
    "tbabs_transmission",
    "xray_agn_corona",
    "xray_agn_corona_from_disc",
    "xray_anisotropy",
    "xray_hotgas",
    "xray_total",
    "xray_xrb",
]
