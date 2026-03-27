"""AGN emission models for tengri.

Provides modular accretion disc and dust torus components that combine
into a unified AGN SED. Three complexity levels:

- **simple**: power-law disc + single-temperature torus (3 params)
- **standard**: multi-color Shakura-Sunyaev disc + two-temperature torus (6 params)
- **kubota_done**: Kubota & Done (2018) disc (outer zone only) + clumpy torus (8+ params)
- **kubota_done_full**: Kubota & Done (2018) full 3-zone disc + torus (13+ params)
- **unified_nlr_blr**: kubota_done + NLR/BLR decomposition with geometric masking (12+ params)
- **skirtor**: power-law disc + SKIRTOR clumpy torus (Stalevski+2012, 2016) (7 params)
- **qsogen**: Temple, Hewett & Banerji (2021) empirical quasar SED (7 params)

Usage::

    from tengri.models.agn import get_agn_model, unified_agn

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
- Stalevski et al. 2016, MNRAS, 458, 2288
- Temple, Hewett & Banerji 2021, MNRAS, 508, 737
"""

from tengri.models.agn.blr import blr_emission
from tengri.models.agn.disc import kubota_done_disc, multicolor_disc, powerlaw_disc
from tengri.models.agn.nlr import nlr_emission

# QSOgen must be imported after unified (needs register_agn_model)
from tengri.models.agn.qsogen import qsogen, qsogen_sed
from tengri.models.agn.skirtor import create_skirtor_from_grid, skirtor_analytic
from tengri.models.agn.torus import simple_torus, two_temperature_torus
from tengri.models.agn.unified import (
    AGN_MODELS,
    get_agn_model,
    kubota_done_full_agn,
    register_agn_model,
    unified_agn,
    unified_nlr_blr,
)

__all__ = [
    # Registry
    "AGN_MODELS",
    "blr_emission",
    "create_skirtor_from_grid",
    "get_agn_model",
    # Disc models
    "kubota_done_disc",
    "kubota_done_full_agn",
    "multicolor_disc",
    # Line region models
    "nlr_emission",
    "powerlaw_disc",
    # QSOgen
    "qsogen",
    "qsogen_sed",
    "register_agn_model",
    # Torus models
    "simple_torus",
    "two_temperature_torus",
    # Unified
    "unified_agn",
    "unified_nlr_blr",
]
