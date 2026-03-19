"""AGN emission models for diffsed.

Provides modular accretion disc and dust torus components that combine
into a unified AGN SED. Three complexity levels:

- **simple**: power-law disc + single-temperature torus (3 params)
- **standard**: multi-color Shakura-Sunyaev disc + two-temperature torus (6 params)
- **kubota_done**: Kubota & Done (2018) disc + clumpy torus (8+ params)
- **unified_nlr_blr**: kubota_done + NLR/BLR decomposition with geometric masking (12+ params)

Usage::

    from diffsed.models.agn import get_agn_model, unified_agn

    # Named model
    model_fn = get_agn_model("simple")
    l_nu = model_fn(wavelength, agn_log_lbol=44.0, agn_frac=0.1)

    # Generic combiner
    l_nu = unified_agn(wavelength, agn_log_lbol=44.0, disc_model="multicolor")

References
----------
- Shakura & Sunyaev 1973, A&A, 24, 337
- Kubota & Done 2018, MNRAS, 480, 1247
- Nenkova et al. 2008, ApJ, 685, 147
- Stalevski et al. 2012, MNRAS, 420, 2756
"""

from diffsed.models.agn.blr import blr_emission
from diffsed.models.agn.disc import multicolor_disc, powerlaw_disc
from diffsed.models.agn.nlr import nlr_emission
from diffsed.models.agn.torus import simple_torus, two_temperature_torus
from diffsed.models.agn.unified import (
    AGN_MODELS,
    get_agn_model,
    register_agn_model,
    unified_agn,
    unified_nlr_blr,
)

__all__ = [
    # Registry
    "AGN_MODELS",
    "get_agn_model",
    "register_agn_model",
    # Unified
    "unified_agn",
    "unified_nlr_blr",
    # Disc models
    "multicolor_disc",
    "powerlaw_disc",
    # Torus models
    "simple_torus",
    "two_temperature_torus",
    # Line region models
    "nlr_emission",
    "blr_emission",
]
