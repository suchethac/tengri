# SPDX-License-Identifier: BSD-3-Clause
"""Parameter registry builder and lazy bucket resolution.

This module contains the aggregator logic that converts component-owned
:class:`ParamDeclaration` tuples into the legacy 4-tuple bucket-dict format
consumed by :class:`~tengri.parameters.parameters.Parameters` and related code.

Prior to ADR-0005 migration completion, these functions remain the API bridge
between the canonical component parameter definitions and the remaining legacy
code paths.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterable

from tengri.parameters.priors import Fixed
from tengri.protocols.component import ParamDeclaration


def _bucket_from_declarations(
    decls: Iterable[ParamDeclaration],
) -> dict[str, tuple[str, object, str, object]]:
    """Adapter: component-owned :class:`ParamDeclaration` tuple → legacy
    4-tuple bucket dict consumed by :func:`_build_param_registry` and by
    :mod:`tengri.parameters.translate`.

    A ``bound_check`` of ``None`` is normalized to ``lambda lo, hi: True``
    so the downstream code that always calls the check stays branch-free.
    """
    return {
        d.name: (
            d.description,
            d.bound_check if d.bound_check is not None else (lambda lo, hi: True),
            d.bound_error,
            d.prior,
        )
        for d in decls
    }


# ── AGN extras: neb_xid parameter merged at bucket resolution time ────────
#
# ``neb_xid`` is nebular-prefixed but owned/consumed by the Feltre NLR backend
# and appears alongside ``agn_alpha_ion`` in the AGN bucket. Legacy code expects
# it bundled with _AGN_PARAMS, so it is merged at bucket-resolution time rather
# than moving it into ``components/agn/_params.py`` (which would violate the
# prefix invariant checked by ``tools/check_param_prefixes.py``).

# ``neb_xid`` carries no ``free_prior`` (#887), for two reasons that compound.
# It is read only by the Feltre NLR backend, and the ``neb`` group wildcard is
# not backend-scoped the way ``dust.emission`` has been since #1482, so freeing
# it under any other nebular backend would add an inert dimension. Its grid is
# also three nodes ({0.1, 0.3, 0.5}) inside a validator interval of
# [0.05, 0.6], so the admissible interval and the tabulated one are not the same
# object and a uniform over the former would spend most of its mass off-grid.
#
# This tuple is a *third* declaration shape besides ``ParamDeclaration`` and
# ``ParamDef``/``MetParamDef``, and the only one still without a ``free_prior``
# slot. Adding one here is not worth it for a single parameter that should not
# be freed by wildcard anyway; fold it into a real declaration if that changes.
_AGN_EXTRAS: dict = {
    "neb_xid": (
        "Dust-to-metal ratio (Feltre NLR backend) [0.1, 0.3, 0.5]",
        lambda lo, hi: lo >= 0.05 and hi <= 0.6,
        "must be in [0.05, 0.6] (grid values: 0.1, 0.3, 0.5)",
        Fixed(0.3),
    ),
}


# ── Lazy bucket resolution ──────────────────────────────────────────────────
#
# Component-owned ``PARAMS`` and sub-tuple declarations are imported lazily
# here to avoid circular imports: ``tengri.components/__init__.py`` eagerly
# loads every component subpackage, and some transitively re-enter
# :mod:`tengri.parameters`. Resolving on first access defers the import until
# this module has finished initializing.
#
# Maps legacy bucket name → (component module path, attribute name on that module).

_LAZY_DECL_SOURCES: dict[str, tuple[str, str]] = {
    "_RADIO_PARAMS": ("tengri.components.radio._params", "PARAMS"),
    "_XRAY_PARAMS": ("tengri.components.xray._params", "PARAMS"),
    "_AGN_PARAMS": ("tengri.components.agn._params", "PARAMS"),
    "_NEBULAR_PARAMS": ("tengri.components.nebular._params", "PARAMS"),
    "_DUST_EMISSION_PARAMS": ("tengri.components.dust._params", "PARAMS"),
    # PR4: dust attenuation + single-component dust + IGM patchy + DLA
    "_DUST_EXTRA_PARAMS": ("tengri.components.dust._params", "ATTENUATION_PARAMS"),
    "_SINGLE_COMPONENT_DUST_PARAMS": (
        "tengri.components.dust._params",
        "SINGLE_COMPONENT_PARAMS",
    ),
    "_IGM_PATCHY_PARAMS": ("tengri.components.igm._params", "PATCHY_PARAMS"),
    "_DLA_PARAMS": ("tengri.components.igm._params", "DLA_PARAMS"),
    # PR5: nebular sub-buckets + shock + stellar α/Fe
    "_CB19_PARAMS": ("tengri.components.nebular._params", "CB19_PARAMS"),
    "_ELINE_PARAMS": ("tengri.components.nebular._params", "ELINE_PARAMS"),
    "_ELINE_BROAD_PARAMS": (
        "tengri.components.nebular._params",
        "ELINE_BROAD_PARAMS",
    ),
    "_CUE_IONSPEC_PARAMS": (
        "tengri.components.nebular._params",
        "CUE_IONSPEC_PARAMS",
    ),
    "_CUE_GAS_EXTRA_PARAMS": (
        "tengri.components.nebular._params",
        "CUE_GAS_EXTRA_PARAMS",
    ),
    "_SHOCK_PARAMS": ("tengri.components.nebular._params", "SHOCK_PARAMS"),
    "_ALPHA_FE_PARAMS": ("tengri.components.stellar._params", "ALPHA_FE_PARAMS"),
    "_EVOLVING_ALPHA_PARAMS": (
        "tengri.components.stellar._params",
        "EVOLVING_ALPHA_PARAMS",
    ),
}

#: Extra entries merged into a lazily-resolved bucket after the
#: component-owned declarations are converted. Keyed by bucket name.
_LAZY_DECL_EXTRAS: dict[str, dict] = {
    "_AGN_PARAMS": _AGN_EXTRAS,
}


def _resolve_lazy_bucket(name: str) -> dict:
    """Resolve a lazily-imported parameter bucket.

    Loads the component module, converts its declarations to the legacy 4-tuple
    format, merges any extras, and caches the result.
    """
    src = _LAZY_DECL_SOURCES.get(name)
    if src is None:
        raise AttributeError(f"No lazy source for bucket {name!r}")
    module_path, attr = src
    mod = importlib.import_module(module_path)
    bucket = _bucket_from_declarations(getattr(mod, attr))
    extras = _LAZY_DECL_EXTRAS.get(name)
    if extras:
        bucket = {**bucket, **extras}
    return bucket


# ── Non-SFH parameter bucket (derived from canonical _shared.PARAMS) ────────
#
# Includes observation-layer noise params (``noise_frac_cal``, ``noise_dof``)
# declared in ``observation/_params.py`` under the ADR-0005 component-owned
# pattern. They are merged here so the legacy ``_NON_SFH_PARAMS`` bucket
# remains the single registry entry consumed by ``_build_param_registry``.

from tengri.observation import _params as _obs_params_module
from tengri.parameters import _shared as _shared_module

_NON_SFH_PARAMS = {
    **_bucket_from_declarations(_shared_module.PARAMS),
    **_bucket_from_declarations(_obs_params_module.PARAMS),
}


# ── Two-component dust parameters ───────────────────────────────────────────

from tengri.components.dust._params import ATTENUATION_TWO_COMPONENT_ONLY

# ── Settings keys (non-parameters) ─────────────────────────────────────────

# Settings keys that are not model parameters
SETTINGS_KEYS = frozenset(
    {
        "stochastic",
        "n_grid",
        "mean_sfh_type",
        # Nebular emission
        "nebular",
        "nebular_ssp",
        "nebular_cue",
        "neb_ionization",
        "cloudy_grid_path",
        "cue_weights_path",
        # Dust model & law
        "dust_model",
        "dust_approx",
        "dust_law",
        "dust_law_bc",
        "dust_law_diff",
        # Dust emission
        "dust_emission",
        "dl07_grid_path",
        # AGN
        "agn_model",
        # AGN block-recipe selectors (consumed by agn_model="composable")
        "agn_disc_block",
        "agn_nlr_block",
        "agn_blr_block",
        "agn_feii_block",
        "agn_torus_block",
        "agn_attenuation_block",
        # AGN composable-precompute axes: dict[param_name → np.ndarray grid].
        # When set with agn_model="composable", SEDModel builds the
        # composable_precompute lookup at construction time and threads
        # it through the hybrid kernel via PrecomputedData.
        "agn_axis_grids",
        # Radio & X-ray
        "radio",
        "xray",
        # Shock emission
        "shock",
        # Metallicity mode (registry-based, replaces evolving_metallicity/chem_evol)
        "met_mode",
        # Older boolean flags (resolved to met_mode internally)
        "evolving_metallicity",
        "alpha_fe_evolving",
        "chem_evol",
        # Metallicity interpolation
        "met_interp",
        "lgmet_scatter",
        # Emission line fitting mode
        "eline_mode",  # "off", "fixed", "marginalized", "fitted"
        "eline_broad",  # bool: enable broad AGN emission line component
    }
)


# ── Parameter registry builder ──────────────────────────────────────────────


def _build_param_registry(
    mean_sfh_type,
    nebular=False,
    dust_model="two_component",
    dust_law_bc="power_law",
    dust_law_diff=None,
    dust_emission=None,
    agn_model=None,
    radio=False,
    xray=False,
    shock=False,
    igm_patchy=False,
    dla=False,
    evolving_metallicity=False,
    alpha_fe_evolving=False,
    chem_evol=False,
    met_mode="delta",
    eline_mode="off",
    eline_broad=False,
):
    """Build the parameter registry for a given model configuration.

    Parameters
    ----------
    mean_sfh_type : list[str]
        SFH model components.
    nebular : bool or str
        Enable nebular parameters. True or "cloudy" adds neb_logU, neb_logZ_gas, neb_fesc.
    dust_model : str
        Dust geometry model: ``"two_component"`` (Charlot & Fall) or
        ``"single_component"`` (uniform screen).
    dust_law_bc : str
        Birth cloud dust law name. Non-power-law laws may add extra parameters.
    dust_law_diff : str or None
        Diffuse ISM dust law. None = same as bc.
    evolving_metallicity : bool
        If True, replace met_logzsol with met_logzsol_0 and met_logzsol_final.
    chem_evol : bool
        If True, derive Z(t) from SFH via gas-regulator model. Replaces
        met_logzsol with chem_yield, chem_eta_outflow, etc.

    Returns
    -------
    registry : dict
        param_name -> (description, bound_check, bound_error)
    defaults : dict
        param_name -> default Distribution
    """
    from tengri.components.stellar.sfh.met_registry import resolve_met
    from tengri.components.stellar.sfh.registry import resolve_sfh

    _, sfh_params, _, _ = resolve_sfh(mean_sfh_type)

    registry = {}
    defaults = {}

    # SFH params from registry
    for pname, pdef in sfh_params.items():
        registry[pname] = (pdef.description, pdef.bound_check, pdef.bound_error)
        defaults[pname] = pdef.default

    # Non-SFH global params (redshift, noise, sigma_v_kms; met_logzsol
    # is injected separately by the metallicity registry below).
    for pname, (desc, check, err, default) in _NON_SFH_PARAMS.items():
        if pname == "met_logzsol":
            continue
        registry[pname] = (desc, check, err)
        defaults[pname] = default

    # Dust attenuation params (PR4: moved to components/dust/_params.py).
    # The two Charlot-Fall optical depths are skipped under the single-screen
    # models (``dust_model="single_component"`` and WG00 ``dust_model="wg00"``,
    # FSPS dust_type=3); ``dust_tau_v`` from ``_SINGLE_COMPONENT_DUST_PARAMS``
    # takes their place there.
    _is_single = dust_model in ("single_component", "wg00")
    from tengri.components.dust._params import ATTENUATION_PARAMS

    for decl in ATTENUATION_PARAMS:
        if _is_single and decl.name in ATTENUATION_TWO_COMPONENT_ONLY:
            continue
        check = decl.bound_check if decl.bound_check is not None else (lambda lo, hi: True)
        registry[decl.name] = (decl.description, check, decl.bound_error)
        defaults[decl.name] = decl.prior

    if _is_single:
        single_comp_bucket = _resolve_lazy_bucket("_SINGLE_COMPONENT_DUST_PARAMS")
        for pname, (desc, check, err, default) in single_comp_bucket.items():
            registry[pname] = (desc, check, err)
            defaults[pname] = default

    # Metallicity params from registry (replaces ad-hoc evolving_metallicity/chem_evol)
    _, met_params, _, _ = resolve_met(met_mode)
    for pname, pdef in met_params.items():
        registry[pname] = (pdef.description, pdef.bound_check, pdef.bound_error)
        defaults[pname] = pdef.default

    # Nebular params (CLOUDY, Cue, or CB_19; not BakedIn/ssp/off).
    if nebular in ("cloudy", "cue", "cb19"):
        nebular_bucket = _resolve_lazy_bucket("_NEBULAR_PARAMS")
        for pname, (desc, check, err, default) in nebular_bucket.items():
            registry[pname] = (desc, check, err)
            defaults[pname] = default

    # CB_19, alpha-Fe, shock, eline buckets are all resolved lazily.
    if nebular == "cb19":
        cb19_bucket = _resolve_lazy_bucket("_CB19_PARAMS")
        for pname, (desc, check, err, default) in cb19_bucket.items():
            registry[pname] = (desc, check, err)
            defaults[pname] = default

    if alpha_fe_evolving:
        evolving_alpha_bucket = _resolve_lazy_bucket("_EVOLVING_ALPHA_PARAMS")
        for pname, (desc, check, err, default) in evolving_alpha_bucket.items():
            registry[pname] = (desc, check, err)
            defaults[pname] = default
    else:
        alpha_fe_bucket = _resolve_lazy_bucket("_ALPHA_FE_PARAMS")
        for pname, (desc, check, err, default) in alpha_fe_bucket.items():
            registry[pname] = (desc, check, err)
            defaults[pname] = default

    # (Dust attenuation extras now registered above with the rest of the
    # ATTENUATION_PARAMS tuple from components/dust/_params.py.)

    # Dust emission params (only when dust emission is enabled).
    if dust_emission:
        dust_emission_bucket = _resolve_lazy_bucket("_DUST_EMISSION_PARAMS")
        for pname, (desc, check, err, default) in dust_emission_bucket.items():
            registry[pname] = (desc, check, err)
            defaults[pname] = default

    # AGN, radio, X-ray buckets are resolved lazily.
    if agn_model:
        agn_bucket = _resolve_lazy_bucket("_AGN_PARAMS")
        for pname, (desc, check, err, default) in agn_bucket.items():
            registry[pname] = (desc, check, err)
            defaults[pname] = default

    if radio:
        radio_bucket = _resolve_lazy_bucket("_RADIO_PARAMS")
        for pname, (desc, check, err, default) in radio_bucket.items():
            registry[pname] = (desc, check, err)
            defaults[pname] = default

    if xray:
        xray_bucket = _resolve_lazy_bucket("_XRAY_PARAMS")
        for pname, (desc, check, err, default) in xray_bucket.items():
            registry[pname] = (desc, check, err)
            defaults[pname] = default

    if shock:
        shock_bucket = _resolve_lazy_bucket("_SHOCK_PARAMS")
        for pname, (desc, check, err, default) in shock_bucket.items():
            registry[pname] = (desc, check, err)
            defaults[pname] = default

    # Patchy IGM + DLA buckets (PR4: now derived from components/igm).
    if igm_patchy:
        igm_patchy_bucket = _resolve_lazy_bucket("_IGM_PATCHY_PARAMS")
        for pname, (desc, check, err, default) in igm_patchy_bucket.items():
            registry[pname] = (desc, check, err)
            defaults[pname] = default

    if dla:
        dla_bucket = _resolve_lazy_bucket("_DLA_PARAMS")
        for pname, (desc, check, err, default) in dla_bucket.items():
            registry[pname] = (desc, check, err)
            defaults[pname] = default

    if eline_mode in ("marginalized", "fitted"):
        eline_bucket = _resolve_lazy_bucket("_ELINE_PARAMS")
        for pname, (desc, check, err, default) in eline_bucket.items():
            registry[pname] = (desc, check, err)
            defaults[pname] = default

    if eline_broad:
        eline_broad_bucket = _resolve_lazy_bucket("_ELINE_BROAD_PARAMS")
        for pname, (desc, check, err, default) in eline_broad_bucket.items():
            registry[pname] = (desc, check, err)
            defaults[pname] = default

    return registry, defaults
