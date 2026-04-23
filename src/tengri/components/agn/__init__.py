"""AGN emission models for tengri.

Provides modular accretion disc and dust torus components that combine
into a unified AGN SED. Three complexity levels:

- **simple**: power-law disc + single-temperature torus (3 params)
- **standard**: multi-color Shakura-Sunyaev disc + two-temperature torus (6 params)
- **kubota_done**: Kubota & Done (2018) disc (outer zone only) + clumpy torus (8+ params)
- **kubota_done_full**: Kubota & Done (2018) full 3-zone disc + torus (13+ params)
- **adaf**: ADAF + truncated disc for low-luminosity AGN (Lopez+2024, Mahadevan 1997) (6 params)
- **unified_nlr_blr**: kubota_done + NLR/BLR decomposition with geometric masking (12+ params)
- **skirtor**: power-law disc + SKIRTOR clumpy torus (Stalevski+2012, 2016) (7 params)
- **qsogen**: Temple, Hewett & Banerji (2021) empirical quasar SED (7 params)

Usage::

    from tengri.components.agn import resolve_agn_model, unified_agn

    # Named model
    model_fn = resolve_agn_model("simple")
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

from tengri.components.agn.blr import blr_emission, compute_blr_sed
from tengri.components.agn.cat3d_wind import cat3d_wind_analytic, create_cat3d_wind_from_grid
from tengri.components.agn.disc import (
    adaf_disc,
    beloborodov_gamma_hot,
    compute_l2500,
    kubota_done_disc,
    multicolor_disc,
    powerlaw_disc,
)
from tengri.components.agn.nlr import (
    compute_nlr_sed,
    nlr_emission,
    nlr_emission_richardson2014,
)

# QSOgen must be imported after unified (needs register_agn_model)
from tengri.components.agn.qsogen import compute_qsogen_sed, qsogen, qsogen_sed
from tengri.components.agn.silva04 import create_silva04_from_grid, silva04_analytic
from tengri.components.agn.skirtor import create_skirtor_from_grid, skirtor_analytic
from tengri.components.agn.torus import nenkova_torus, simple_torus, two_temperature_torus
from tengri.components.agn.unified import (
    AGN_MODELS,
    adaf_agn,
    get_agn_model,
    kubota_done_full_agn,
    register_agn_model,
    resolve_agn_model,
    unified_agn,
    unified_nlr_blr,
)

__all__ = [
    "AGN_MODELS",
    "AGNConfig",
    "adaf_agn",
    "adaf_disc",
    "beloborodov_gamma_hot",
    "blr_emission",
    "cat3d_wind_analytic",
    "compute_blr_sed",
    "compute_l2500",
    "compute_nlr_sed",
    "compute_qsogen_sed",
    "create_cat3d_wind_from_grid",
    "create_silva04_from_grid",
    "create_skirtor_from_grid",
    "get_agn_model",
    "kubota_done_disc",
    "kubota_done_full_agn",
    "multicolor_disc",
    "nenkova_torus",
    "nlr_emission",
    "nlr_emission_richardson2014",
    "powerlaw_disc",
    "qsogen",
    "qsogen_sed",
    "register_agn_model",
    "resolve_agn_model",
    "silva04_analytic",
    "simple_torus",
    "two_temperature_torus",
    "unified_agn",
    "unified_nlr_blr",
]

# Convenience re-exports for `from tengri.agn import ...`
from tengri.config.settings import AGNConfig
