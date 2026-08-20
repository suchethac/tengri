# SPDX-License-Identifier: BSD-3-Clause
"""Dust attenuation and emission models.

Layout
------
Naming rule: ``*_model.py`` / ``*_ir.py`` files are SEDModelComponent
components (things the model grammar can select); same-stem files without
the suffix are the underlying physics (curves, template loaders).

Components (selected via the model grammar):

- ``component.py``: ``DustAttenuationSEDComponent`` (attenuation only).
- ``two_component.py``: ``DustSEDComponent`` (attenuation + IR
  re-emission with energy balance).
- ``wg00_model.py``: WG00 radiative-transfer attenuation component
  (physics in ``wg00.py``).
- ``schreiber2016_ir.py``, ``draine2021_pah_ir.py`` — standalone IR
  emission components (Draine+2021 physics in ``draine2021_pah.py``).

Physics libraries:

- ``attenuation.py``: k(λ) attenuation laws and the ``DUST_LAWS``
  registry; individual curve families live in ``laws/``.
- ``emission/``: IR re-emission package (analytic models + tabulated
  template components + shared ``_physics.py`` integrals) with
  energy-balance normalization.
- ``astrodust_hd23.py``, ``emission_templates.py`` — template grid
  loaders (not components).
- ``drude_profiles.py``: PAH Drude decomposition helpers.
- ``priors.py``: redshift-dependent attenuation priors from
  Narayanan+2018 cosmological RT simulations.

Internal plumbing:

- ``_apply.py``: applies ``DUST_LAWS`` curves (two-component /
  single-screen transmission, age weights, Lyman cutoff).
- ``_params.py``: free-parameter declarations owned by dust.
- ``_protocol.py``: structural Protocols for laws and templates.
- ``*_precompute.py``: build-time LUT construction for the
  ``approx=WavePrecomp(...)`` path.

"""

# Convenience re-exports for `from tengri.dust import ...`
# Dust block in the SEDComponent pipeline — combines UV–optical attenuation
# with IR re-emission via per-template emission components (``type='astrodust'``,
# ``'draine2021_pah'``, ``'modified_blackbody'``, ``'dale2014'``, …).
from tengri._completion import curated_dir
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

# SEDModelComponent-style attenuation components
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
    themis,
)
from tengri.components.dust.emission_templates import (
    Draine2021PAHTemplates,
    load_draine2021_pahspec_templates,
)

# New names
from tengri.components.dust.priors import (
    narayanan_prior,
    narayanan_tau_prior,
)
from tengri.components.dust.schreiber2016_ir import (
    Schreiber2016IRConfig as Schreiber2016IRConfig,
    Schreiber2016IRSEDComponent as Schreiber2016IRSEDComponent,
)
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
    # SEDModelComponent-style attenuation components
    "WG00AttenuationSEDComponent",
    "WG00AttenuationSEDComponentConfig",
    # Standalone IR emission SEDComponent backends
    "Schreiber2016IRSEDComponent",
    "Schreiber2016IRConfig",
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


__dir__ = curated_dir(_CURATED_DIR)


#: Names removed in #871 when the monolithic ``DustEmissionSEDComponent`` adapter
#: was retired for per-template emission components. No 1:1 successor (the adapter
#: dispatched three templates), so the old names raise with a migration path
#: rather than silently aliasing to one component.
_REMOVED_DUST_EMISSION_NAMES = frozenset(
    {"DustEmissionSEDComponent", "DustEmissionSEDComponentConfig", "DustEmissionSEDComponentState"}
)


def __getattr__(name: str):
    if name in _REMOVED_DUST_EMISSION_NAMES:
        raise AttributeError(
            f"{name!r} was removed in tengri #871. Dust IR emission is now authored "
            "as SEDModelComponents selected via the model grammar, e.g. "
            "SEDModel.build(dust_attenuation={'law': 'calzetti'}, "
            "dust_emission={'type': 'astrodust'}) "
            "with type in {'modified_blackbody', 'draine2021_pah', 'astrodust', "
            "'dale2014', 'casey2012', 'schreiber2016', ...}. To import a component "
            "directly use e.g. "
            "tengri.components.dust.emission.templates.astrodust.AstrodustIRSEDComponent."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
