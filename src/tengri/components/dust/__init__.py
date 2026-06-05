# SPDX-License-Identifier: BSD-3-Clause
"""Dust attenuation and emission models.

Attenuation
-----------
``attenuation.py`` — generalized two-component model with pluggable curves
and f_obscuration.

Emission
--------
``emission.py`` — IR re-emission models (modified blackbody, Casey 2012,
Dale 2014, Draine & Li 2007, Draine & Li 2014 update, Astrodust+PAH,
BOSA, THEMIS) with energy-balance normalization.

Priors
------
``priors.py`` — redshift-dependent dust attenuation priors from
Narayanan+2018 cosmological RT simulations.
"""

# Convenience re-exports for `from tengri.dust import ...`
# Dust block in the SEDComponent pipeline — combines UV–optical attenuation
# with IR re-emission via ``DustEmissionSEDComponentConfig.template``.
from tengri.components.dust.astrodust_ir import (
    AstrodustIRConfig as AstrodustIRConfig,
    AstrodustIRSEDComponent as AstrodustIRSEDComponent,
)
from tengri.components.dust.attenuation import (
    DUST_LAWS,
    apply_lyman_cutoff as apply_lyman_cutoff,
    calzetti as calzetti,
    cardelli as cardelli,
    d03_mwrv31 as d03_mwrv31,
    hd23_mwrv31 as hd23_mwrv31,
    li08 as li08,
    list_laws,
    lmc as lmc,
    precompute_dust_age_mask,
    precompute_dust_age_weights,
    prevot_smc as prevot_smc,
    register_dust_law,
    resolve_dust_law,
    single_component_dust,
    single_component_dust_fast,
    smc as smc,
    two_component_dust,
    two_component_dust_fast,
    vw07_bc as vw07_bc,
    vw07_diff as vw07_diff,
    wd01_mwrv31 as wd01_mwrv31,
    wd01_smcbar as wd01_smcbar,
    wg00_cloudy,
    wg00_dusty,
    wg00_shell,
)

# SEDModelComponent-style attenuation ports
from tengri.components.dust.calzetti_model import Calzetti as Calzetti
from tengri.components.dust.dale2014_ir import (
    Dale2014IRConfig as Dale2014IRConfig,
    Dale2014IRSEDComponent as Dale2014IRSEDComponent,
)
from tengri.components.dust.dl14_ir import (
    DL14IRConfig as DL14IRConfig,
    DL14IRSEDComponent as DL14IRSEDComponent,
)
from tengri.components.dust.draine2021_pah_ir import (
    Draine2021PAHIRConfig as Draine2021PAHIRConfig,
    Draine2021PAHIRSEDComponent as Draine2021PAHIRSEDComponent,
)

# New names
from tengri.components.dust.drude_profiles import (
    N_PAH_FEATURES,
    SMITH2007_PAH_FEATURES,
    compute_pah_template,
    decompose_pah,
    drude_profile,
)
from tengri.components.dust.emission import (
    DUST_EMISSION_MODELS,
    astrodust,
    bosa,
    casey2012,
    cmb_contrast_factor,
    cmb_corrected_temperature,
    compute_absorbed_luminosity,
    compute_absorbed_luminosity_from_tau,
    create_astrodust_from_grid,
    create_bosa_from_grid,
    create_dale2014_from_grid,
    create_dl07_from_grid,
    create_dl14_from_grid,
    create_themis_from_grid,
    dale2014 as dale2014,
    draine_li2007 as draine_li2007,
    draine_li2014 as draine_li2014,
    energy_balance_split as energy_balance_split,
    load_astrodust_templates,
    load_bosa_templates,
    load_dale2014_templates,
    load_dl14_templates,
    load_draine_li_templates,
    load_themis_templates,
    modified_blackbody as modified_blackbody,
    planck_bnu,
    register_astrodust_tabulated,
    register_bosa_tabulated,
    register_dale2014_tabulated,
    register_dl07_tabulated,
    register_dl14_tabulated,
    register_emission_model,
    register_themis_tabulated,
    resolve_emission_model,
    themis,
)
from tengri.components.dust.emission_component import (
    DustEmissionSEDComponent,
    DustEmissionSEDComponentConfig,
    DustEmissionSEDComponentState,
)
from tengri.components.dust.emission_templates import (
    Draine2021PAHTemplates,
    load_draine2021_pahspec_templates,
)
from tengri.components.dust.mw_model import MilkyWay as MilkyWay

# New names
from tengri.components.dust.priors import (
    narayanan_prior,
    narayanan_tau_prior,
)
from tengri.components.dust.salim18_model import Salim18 as Salim18
from tengri.components.dust.schreiber2016_ir import (
    Schreiber2016IRConfig as Schreiber2016IRConfig,
    Schreiber2016IRSEDComponent as Schreiber2016IRSEDComponent,
)
from tengri.components.dust.smc_model import SMC as SMC
from tengri.components.dust.wg00 import (
    WG00_DUST_CURVES as WG00_DUST_CURVES,
    WG00_GEOMETRIES as WG00_GEOMETRIES,
    WG00_STRUCTURES as WG00_STRUCTURES,
    wg00_attenuation as wg00_attenuation,
)
from tengri.components.dust.wg00_model import (
    WG00AttenuationSEDComponent as WG00AttenuationSEDComponent,
    WG00AttenuationSEDComponentConfig as WG00AttenuationSEDComponentConfig,
)

# ──────────────────────────────────────────────────────────────────
# Curated tab-completion surface for `tengri.dust.<TAB>`.
# Hides internal helpers (compute_*, create_*_from_grid, load_*_templates,
# precompute_*, drude_profile, etc.) and constants. Everything is still
# importable via attribute access — only `dir(tengri.dust)` is filtered.
# ──────────────────────────────────────────────────────────────────
_CURATED_DIR = (
    # Registries
    "DUST_LAWS",
    "DUST_EMISSION_MODELS",
    # Named attenuation laws
    "calzetti",
    "cardelli",
    "li08",
    "lmc",
    "smc",
    "prevot_smc",
    "vw07_bc",
    "vw07_diff",
    "d03_mwrv31",
    "hd23_mwrv31",
    "wd01_mwrv31",
    "wd01_smcbar",
    "wg00_attenuation",
    "wg00_cloudy",
    "wg00_dusty",
    "wg00_shell",
    # Named emission templates
    "astrodust",
    "bosa",
    "casey2012",
    "dale2014",
    "draine_li2007",
    "draine_li2014",
    "modified_blackbody",
    "themis",
    # SEDModelComponent-style attenuation ports
    "Calzetti",
    "WG00AttenuationSEDComponent",
    "WG00AttenuationSEDComponentConfig",
    "MilkyWay",
    "Salim18",
    "SMC",
    # Composable SEDComponent adapter (template= dispatch)
    "DustEmissionSEDComponent",
    "DustEmissionSEDComponentConfig",
    "DustEmissionSEDComponentState",
    # Standalone IR emission SEDComponent backends
    "DL14IRSEDComponent",
    "DL14IRConfig",
    "Dale2014IRSEDComponent",
    "Dale2014IRConfig",
    "Schreiber2016IRSEDComponent",
    "Schreiber2016IRConfig",
    "AstrodustIRSEDComponent",
    "AstrodustIRConfig",
    "Draine2021PAHIRSEDComponent",
    "Draine2021PAHIRConfig",
    # Draine+2021 PAHspec template loader
    "Draine2021PAHTemplates",
    "load_draine2021_pahspec_templates",
    # Pipeline helpers
    "two_component_dust",
    "single_component_dust",
    "energy_balance_split",
    "apply_lyman_cutoff",
    # Registration / resolver
    "list_laws",
    "register_dust_law",
    "resolve_dust_law",
    "register_emission_model",
    "resolve_emission_model",
    "register_astrodust_tabulated",
    "register_bosa_tabulated",
    "register_dale2014_tabulated",
    "register_dl07_tabulated",
    "register_dl14_tabulated",
    "register_themis_tabulated",
    # Priors
    "narayanan_prior",
    "narayanan_tau_prior",
    # Submodules
    "attenuation",
    "emission",
    "pah",
    "priors",
)


def __dir__() -> list[str]:
    """Curated tab-completion list. Filtering only — everything remains
    accessible via attribute access."""
    return list(_CURATED_DIR)
