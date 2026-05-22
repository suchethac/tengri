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

    # ``agn_log_lbol`` is log10(L_bol / L_sun) — see ``unified.py`` for the
    # convention warning. agn_log_lbol = 11 corresponds to a bright Seyfert
    # (L_bol ~ 4e44 erg/s); 13 to a bright quasar.

    # Named model
    model_fn = resolve_agn_model("simple")
    l_nu = model_fn(wavelength, agn_log_lbol=11.0, agn_frac=0.1)

    # Generic combiner
    l_nu = unified_agn(wavelength, agn_log_lbol=11.0, disc_model="multicolor")

References
----------
- Shakura & Sunyaev 1973, A&A, 24, 337
- Kubota & Done 2018, MNRAS, 480, 1247
- Nenkova et al. 2008, ApJ, 685, 147
- Stalevski et al. 2012, MNRAS, 420, 2756
- Stalevski et al. 2016, MNRAS, 458, 2288
- Temple, Hewett & Banerji 2021, MNRAS, 508, 737
"""

# Importing ``blocks`` side-effects all @register_agn_block calls plus
# AGN_MODELS["composable"]. Must come after grahsp import so GRAHSP
# blocks see the GRAHSP package fully initialised.
from tengri.components.agn import blocks
from tengri.components.agn._phys import planck_lnu
from tengri.components.agn.blocks import (
    AGN_BLOCKS,
    BLOCK_CATEGORIES,
    RecipeWarning,
    composable_agn_l_nu,
    register_agn_block,
    resolve_agn_block,
    validate_block_recipe,
)
from tengri.components.agn.blr import compute_blr_sed

# SEDModelComponent adapters (Phase II-4)
from tengri.components.agn.cat3d_torus_model import CAT3DTorus
from tengri.components.agn.cat3d_wind import cat3d_wind_analytic, create_cat3d_wind_from_grid
from tengri.components.agn.disc import (
    adaf_disc,
    beloborodov_gamma_hot,
    compute_l2500,
    create_relagn_disc_from_grid,
    kubota_done_disc,
    multicolor_disc,
    powerlaw_disc,
)
from tengri.components.agn.grahsp import (
    GRAHSPSED,
    GRAHSPParams,
    GRAHSPSEDComponent,
    GRAHSPSEDComponentConfig,
    compute_grahsp_sed,
    evaluate_grahsp_agn,
    grahsp,
)
from tengri.components.agn.kd18_disc_model import KD18Disc
from tengri.components.agn.nlr import (
    compute_nlr_sed,
    compute_nlr_sed_richardson2014,
)
from tengri.components.agn.powerlaw_disc_model import PowerLawDisc

# QSOgen must be imported after unified (needs register_agn_model)
from tengri.components.agn.qsogen import compute_qsogen_sed, qsogen
from tengri.components.agn.silva04 import create_silva04_from_grid, silva04_analytic
from tengri.components.agn.silva04_model import Silva04Torus
from tengri.components.agn.skirtor import create_skirtor_from_grid, skirtor_analytic
from tengri.components.agn.skirtor_model import SKIRTORTorus
from tengri.components.agn.torus import nenkova_torus, simple_torus, two_temperature_torus
from tengri.components.agn.unified import (
    AGN_MODELS,
    adaf_agn,
    kubota_done_full_agn,
    register_agn_model,
    resolve_agn_model,
    unified_agn,
    unified_nlr_blr,
)

# New names

__all__ = [
    "AGN_BLOCKS",
    "AGN_MODELS",
    "BLOCK_CATEGORIES",
    "GRAHSPSED",
    "AGNConfig",
    "CAT3DTorus",
    "GRAHSPParams",
    "GRAHSPSEDComponent",
    "GRAHSPSEDComponentConfig",
    "KD18Disc",
    "PowerLawDisc",
    "RecipeWarning",
    "SKIRTORTorus",
    "Silva04Torus",
    "adaf_agn",
    "adaf_disc",
    "beloborodov_gamma_hot",
    "blocks",
    "cat3d_wind_analytic",
    "composable_agn_l_nu",
    "compute_blr_sed",
    "compute_grahsp_sed",
    "compute_l2500",
    "compute_nlr_sed",
    "compute_nlr_sed_richardson2014",
    "compute_qsogen_sed",
    "create_cat3d_wind_from_grid",
    "create_relagn_disc_from_grid",
    "create_silva04_from_grid",
    "create_skirtor_from_grid",
    "evaluate_grahsp_agn",
    "grahsp",
    "kubota_done_disc",
    "kubota_done_full_agn",
    "multicolor_disc",
    "nenkova_torus",
    "planck_lnu",
    "powerlaw_disc",
    "qsogen",
    "register_agn_block",
    "register_agn_model",
    "resolve_agn_block",
    "resolve_agn_model",
    "silva04_analytic",
    "simple_torus",
    "two_temperature_torus",
    "unified_agn",
    "unified_nlr_blr",
    "validate_block_recipe",
]


def __dir__() -> list[str]:
    """Curated tab-completion list — only the names in ``__all__``."""
    return list(__all__)


# Convenience re-exports for `from tengri.agn import ...`
from tengri.config.settings import AGNConfig
