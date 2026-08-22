# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Suchetha Cooray
# Parameter-transform conventions follow Prospector (Johnson et al. 2021)

"""Nested-dict model builder for parameter specification.

Provides a Bagpipes-style nested-dictionary interface to the Parameters
class. Instead of flat kwargs (e.g., ``sfh_dpl_alpha=..., sfh_dpl_beta=...``),
users can organize parameters into semantic groups::

    from tengri.parameters import parse_groups, FREE, FIXED

    params = parse_groups(
        sfh={"type": "dpl", "all_params": FREE, "beta": 0.5},
        dust_attenuation={"type": "two_component", "law": "calzetti", "all_params": FIXED},
        dust_emission={"type": "dale2014"},
        neb={"type": "cue"},
        redshift=FREE,
    )

The parser translates group structure and parameter overrides into a
canonical Parameters object via a two-pass algorithm:

**Pass 1** (structural translation): Map group names and type values to
model configuration kwargs (e.g., ``sfh['type'] = 'dpl'`` → ``mean_sfh_type='dpl'``).

**Pass 2** (parameter resolution): Declare a structural Parameters, then
for each declared parameter, decide its final prior/value by:

1. Checking for per-parameter override in the user's group dict.
2. Checking for a wildcard directive ('all_params': FREE / FIXED).
3. Using the registry's default (fixed at median for free defaults).

Parameters
----------
**kwargs : keyword arguments
    Model configuration. Keys are either group names (sfh, dust, neb, etc.),
    top-level settings (redshift, apply_igm), or both.

Returns
-------
Parameters
    A fully resolved Parameters object ready for use with SEDModel.

Raises
------
ValueError
    If unknown group key, unknown type value, or unknown sub-key is provided.

Navigation
----------
This file stays one module by design; use the ``# ── <section> ──``
marker lines to jump. In order:

- registry loading helpers and ``_valid_*_types()`` menus
- ``Constants``
- ``Main API``: ``parse_groups()``
- ``Internal helpers``: one ``_translate_<group>()`` per group (sfh,
  dust_attenuation, dust_emission, neb, shock, igm, radio, foreground, xray, agn),
  plus _translate_dust_retired for the old dust= form, then key validation and
  per-parameter resolution
- ``Inverse: Parameters to nested-dict form``: ``parameters_to_groups()``

Notes
-----
**Not JAX-traced**: Like Parameters itself, parse_groups is a pure Python
translator and cannot be called inside a JAX gradient tape.

**Wildcard semantics**: The 'all_params' key in a group dict applies a default
(FREE or FIXED) to all parameters in that group not explicitly overridden.
'all_params' is the only accepted user-facing spelling; '*' is an internal key
the normalizer rewrites to, not a user input synonym.

**Sentinels**: FREE and FIXED are singleton objects that preserve identity
across copy and pickle operations.

References
----------
.. [1] Bagpipes model builder (Carnall et al., 2018, arXiv:1712.04452).

Examples
--------
>>> from tengri.parameters import parse_groups, FREE, FIXED
>>> from tengri.parameters import Uniform
>>> params = parse_groups(
...     sfh={"type": "dpl", "all_params": FREE, "alpha": Uniform(0.5, 3.0)},
...     redshift=0.1,
... )
>>> "sfh_dpl_alpha" in params.free_params
True
"""

from __future__ import annotations

import difflib
import inspect
import warnings
from collections.abc import Callable
from functools import cache, lru_cache
from typing import NamedTuple

from tengri.config.exceptions import (
    AdvisoryWarning,
    DefaultFixedParametersWarning,
    ParameterError,
    WildcardPartialFreeWarning,
    warn_measured,
)
from tengri.parameters._builders import _resolve_lazy_bucket
from tengri.parameters.parameters import Parameters
from tengri.parameters.priors import Distribution, Fixed
from tengri.parameters.sentinels import FIXED, FREE, WILDCARD_ALIAS, WILDCARD_KEY

__all__ = ["parameters_to_groups", "parse_groups"]


# Canonical fixed-default values that override the prior-midpoint rule
# (``registry_default.unstandardize(0.0)``) used by the wildcard-FIXED
# resolver. The midpoint is rarely what the user actually wants for a
# default — e.g. ``Uniform(-2.0, 0.2)`` for ``met_logzsol`` gives -0.9,
# which silently injected a ~0.85 dex Z offset in CIGALE comparisons
# (#412). Conventions here are chosen to match the canonical
# external-library defaults (FSPS / Bagpipes / Prospector).
_CANONICAL_FIXED_DEFAULTS: dict[str, float] = {
    # met_logzsol = log10(Z/Z⊙); 0.0 = solar — matches FSPS ``logzsol=0.0``
    # and Bagpipes ``metallicity=1.0 Z⊙``.
    "met_logzsol": 0.0,
    "met_logzsol_0": 0.0,
    "met_logzsol_final": 0.0,
    "met_logzsol_old": 0.0,
    "met_logzsol_young": 0.0,
    "met_logzsol_burst": 0.0,
    # Lognormal metallicity-scatter sigma [dex] (#506). Pinned to the historical
    # fixed ``config.lgmet_scatter`` (0.1) so ``*: FIXED`` delta models are
    # byte-unchanged; free it to fit the MDF width.
    "met_logzsol_scatter": 0.1,
}


def _expand_free(param_name: str, registry_default: Distribution) -> Distribution:
    """Resolve ``FREE`` for one parameter: its declared range, else the default.

    ``FREE`` used to resolve straight to ``registry_default``. For the ~half of
    the registry whose default is a ``Fixed`` scalar that silently freed
    nothing — the fit then ran with that physics pinned while the caller
    believed it was being sampled (#1264).

    Now a parameter may declare ``free_prior``: the admissible range to open up
    when asked to be free, normally the same range ``bound_check`` enforces.
    When it does, ``FREE`` genuinely frees. When it does not, this returns the
    Fixed default unchanged and
    :func:`_check_wildcard_freed_something` says so rather than pretending it
    worked — refusing outright when the group freed nothing, and warning with
    :class:`WildcardPartialFreeWarning` when it freed only a subset.

    Parameters
    ----------
    param_name : str
        Canonical (fully-prefixed) parameter name, e.g. ``"neb_logU"``.
    registry_default : Distribution
        The parameter's registry default.

    Returns
    -------
    Distribution
        ``free_prior`` when one is declared and the default is Fixed;
        otherwise ``registry_default`` unchanged.
    """
    if not registry_default.is_fixed:
        # Already free — FREE means "leave the registry's range alone".
        return registry_default
    # Local import: registry imports parse_groups (for recipe introspection),
    # so a module-level import here would close the cycle.
    from tengri.parameters.registry import registry

    record = registry().get(param_name)
    free_prior = getattr(record, "free_prior", None) if record is not None else None
    return free_prior if free_prior is not None else registry_default


def _default_fixed_value(param_name: str, registry_default: Distribution) -> float:
    """Pick the fixed value used when wildcard-FIXED collapses a free param.

    Resolution order:
    1. ``_CANONICAL_FIXED_DEFAULTS`` — hand-curated per-name override for
       parameters whose name carries CIGALE / FSPS / Bagpipes conventions
       that take precedence over a generic prior default (e.g. ``met_logzsol``
       across the several stellar metallicity models).
    2. ``registry_default.default`` — the per-declaration default set on the
       ``Distribution`` itself via #478 (``Uniform(lo, hi, default=...)``).
    3. ``ParameterDefaultMissingError`` — every declared free parameter must
       carry one of the above. The pre-#478 fallback to
       ``registry_default.unstandardize(0.0)`` (the prior midpoint) is gone:
       for symmetric ranges the midpoint often coincided with the physical
       default (``Uniform(-2, 2)`` → 0.0 = solar for ``gas_logno``) but for
       skewed ranges it silently injected non-physical values (``Uniform(0, 5)``
       for ``gas_logn = log10(n_H/cm^-3)`` → 2.5 = 316 cm⁻³, three times the
       CIGALE-canonical 100 cm⁻³). Forcing an explicit default at the
       declaration site eliminates that class of silent footgun.
    """
    if param_name in _CANONICAL_FIXED_DEFAULTS:
        return _CANONICAL_FIXED_DEFAULTS[param_name]
    if registry_default.default is not None:
        return float(registry_default.default)
    # Defense-in-depth: the contract test
    # ``tests/contract/test_param_defaults.py`` enforces every declared
    # parameter carries an explicit ``default=``. If we get here it means
    # something slipped through (or the user constructed a Distribution by
    # hand without one). Warn loudly so the gap is visible in stderr / CI
    # logs, then fall back to the prior midpoint to keep introspection paths
    # (e.g. ``recipe_parameters`` invoked by ``builders/agn/atten.py`` at
    # import time) from crashing the whole package. The warning category is
    # ``UserWarning`` so it shows up by default; do not silence it without
    # adding the default at the declaration site.
    warn_measured(
        f"{param_name!r} has no curated default, so 'all_params': FIXED pins it at its "
        f"prior midpoint ({float(registry_default.unstandardize(0.0)):.4g}) — "
        f"an arbitrary rather than physically motivated value. Pass an "
        f"explicit value for it in the group dict to silence this, or leave "
        f"it FREE. (Curating registry defaults is tracked in #1007.)",
        UserWarning,
        stacklevel=3,
        prior_midpoint=float(registry_default.unstandardize(0.0)),
    )
    return float(registry_default.unstandardize(0.0))


# Ensure SEDModelComponent subclasses are imported and registered.
# This populates _REGISTRY so the resolver can consult it.
def _ensure_registry_loaded() -> None:
    """Import all SEDModelComponent subclasses to populate _REGISTRY."""
    try:
        # Import key component modules to trigger __init_subclass__ registration
        import tengri.components.agn
        import tengri.components.dust
        import tengri.components.nebular
        import tengri.components.radio
        import tengri.components.xray

        # Force use of imports so they're not removed as unused
        _ = (
            tengri.components.agn,
            tengri.components.dust,
            tengri.components.nebular,
            tengri.components.radio,
            tengri.components.xray,
        )
    except ImportError:
        # If imports fail (missing dependencies), gracefully continue.
        # The registry just won't have those types available.
        pass


_ensure_registry_loaded()


# ── Constants ──────────────────────────────────────────────────────────────


#: Dust emission parameter names that belong to the 'dust.emission' subgroup.
_DUST_EMISSION_PARAM_NAMES = frozenset(_resolve_lazy_bucket("_DUST_EMISSION_PARAMS").keys())

#: Optional Cue nebular knobs beyond logU/logZ_gas — gas density / abundance
#: ratios (``gas_logn``, ``gas_logno``, ``gas_logco``) and the broken-power-law
#: ionizing-spectrum shape (``ionspec_index1..4``, ``ionspec_logLratio1..3``).
#: They carry ``None`` priors in the registry (registered only when the user
#: supplies them), so they are absent from a structural ``Parameters`` and the
#: partition. The nested-dict builder recognizes them as ``neb`` keys (when
#: ``type='cue'``) and forwards user-provided values to the flat constructor,
#: which registers them on demand (#653).
_OPTIONAL_NEB_PARAM_NAMES = frozenset(
    {
        *_resolve_lazy_bucket("_CUE_GAS_EXTRA_PARAMS").keys(),
        *_resolve_lazy_bucket("_CUE_IONSPEC_PARAMS").keys(),
    }
)


def _valid_sfh_types() -> frozenset[str]:
    """Return the set of accepted ``sfh.type`` values, derived from the registry.

    Mirrors ``SFH_REGISTRY.keys()`` minus
    :data:`~tengri.components.stellar.sfh.registry.UNVALIDATED_SFH_TYPES` so the
    dict-grammar validator and the auto-generated ``tengri.builders.sfh.*``
    factories share one source of truth (per ADR-0005 / ADR-0008) and only
    advertise SFHs that actually forward-model. Includes alias keys (``db``,
    ``dbp``) that point at the same spec. Looked up at call time so newly-
    registered SFHs (e.g. plugins) are picked up without re-importing.
    """
    from tengri.components.stellar.sfh.registry import (
        SFH_REGISTRY,
        UNVALIDATED_SFH_TYPES,
    )

    return frozenset(SFH_REGISTRY.keys()) - UNVALIDATED_SFH_TYPES


#: Valid dust model types.
_VALID_DUST_TYPES = {
    "two_component",
    "single_component",
    "wg00",
}

#: Valid ``shock={'type': ...}`` values. Named here rather than built inline
#: at the point of validation so :func:`tengri.list_shock_models` can derive
#: its menu from the very set the builder checks against — the menu and the
#: validator then cannot drift (the failure mode behind #1273 and #1276).
_VALID_SHOCK_TYPES = frozenset({"none", "mappings"})

#: Witt & Gordon (2000) structural selectors (dust_model="wg00", FSPS dust_type=3).
_WG00_DUST_CURVES = ("mw", "smc")
_WG00_GEOMETRIES = ("shell", "cloudy", "dusty")
_WG00_STRUCTURES = ("homogeneous", "clumpy")

#: Dust emission types that register lazily via ``register_*_tabulated``
#: helpers (template data loaded from HDF5 only on first call). These names
#: are valid even before any tabulated grid has been resolved, so they must
#: be unioned into the live ``DUST_EMISSION_MODELS`` view when the validator
#: builds its accepted set. Keeping them as a small, explicit constant
#: rather than introspecting the loader plumbing keeps the validator path
#: a pure function of static state.
_LAZY_DUST_EMISSION_TYPES = frozenset(
    {
        "dl07_tabulated",
        # AGNfitter-rX DH02_CE01 legacy cold-dust library: an engine-only
        # tabulated model registered lazily in DUST_EMISSION_MODELS (no
        # SEDModelComponent), so it is not discovered via the _REGISTRY
        # scan and must be declared here to be an accepted dust.emission type.
        "dh02_ce01",
    }
)


def _valid_dust_emission_types() -> frozenset[str]:
    """Return accepted ``dust.emission.type`` values, derived from the registry.

    Reads the ``_REGISTRY`` (populated by SEDModelComponent.__init_subclass__
    at import time) and returns names of all components whose ``outputs`` include
    ``"sed_dust_ir"``. This automatically picks up all dust emission components
    (modified_blackbody, dale2014, draine_li2007, etc.) without manual
    maintenance. Union with the alias map keys (e.g., draine2021_pah →
    draine2021_pah_ir) so grammar type names resolve correctly.

    Includes ``energy_balance_split`` (a registered two-temperature + AGN-IR
    component publishing ``sed_dust_ir``, Kokorev+2021 — a real model, not a
    helper). Also includes ``_LAZY_DUST_EMISSION_TYPES`` (ADR-0005 / ADR-0008).
    """
    from tengri.components.sed_model_component import _REGISTRY
    from tengri.forward.component_factory import _EMISSION_TYPE_ALIASES

    # Registry names whose outputs include "sed_dust_ir" (getattr skips non-emission entries, #844)
    dust_ir_components = frozenset(
        name
        for name, cls in _REGISTRY.items()
        if "sed_dust_ir" in {o.name for o in getattr(cls, "_outputs_tuple", ())}
    )

    # Add alias keys (grammar names that map to registry names)
    alias_keys = frozenset(_EMISSION_TYPE_ALIASES.keys())

    return dust_ir_components | alias_keys | _LAZY_DUST_EMISSION_TYPES


def _valid_nebular_types() -> frozenset[str]:
    """Derive accepted ``neb.type`` values from :data:`NEBULAR_MODELS`.

    Mirrors the IGM / radio / X-ray derivation (#355): the validator
    reads the runtime registry rather than maintaining a parallel
    hand-written set. Adding a new nebular backend = one
    ``register_nebular_model`` call in
    ``components/nebular/__init__.py``; this validator picks it up
    automatically. ADR-0005 / ADR-0008.
    """
    from tengri.components.nebular import NEBULAR_MODELS

    return frozenset(NEBULAR_MODELS.keys())


def _valid_igm_types() -> frozenset[str]:
    """Derive accepted ``igm.type`` values from :data:`IGM_MODELS`.

    Following ADR-0005 / ADR-0008 (single source of truth), the
    grammar-layer validator views the runtime registry directly rather
    than maintaining a parallel hand-written set. Adding a new IGM
    transmission model = one ``register_igm_model`` call in
    ``components/igm/__init__.py``; this validator picks it up
    automatically.
    """
    from tengri.components.igm import IGM_MODELS

    return frozenset(IGM_MODELS.keys())


#: Map grammar-layer IGM names to the canonical form consumed by
#: :meth:`SEDModel._init_igm` (which only accepts ``'inoue'`` / ``'madau'``).
_IGM_TYPE_ALIASES = {
    "inoue14": "inoue",
    "inoue": "inoue",
    "madau": "madau",
    "meiksin06": "meiksin06",
    "asada25": "asada25",
}


def _valid_radio_types() -> frozenset[str]:
    """Derive accepted ``radio={'type': ...}`` values from :data:`RADIO_MODELS`
    and SEDModelComponent radio variants.

    RADIO_MODELS contains the legacy ``radio={'type': 'condon92'|'none'}``
    path; SEDModelComponent models like ``radio_powerlaw`` and ``radio_dpl``
    are also valid ``type`` values and must be discoverable to users.
    Both derive from the registry so the menu and the builder cannot drift
    (the failure mode behind #1120).
    """
    from tengri.components.radio import RADIO_MODELS
    from tengri.forward.component_factory import _REGISTRY

    # Legacy function-based models: 'condon92', 'none'
    valid_types = set(RADIO_MODELS.keys())

    # SEDModelComponent models: 'radio_powerlaw', 'radio_dpl'
    # Filter for components that start with 'radio_' to avoid including the
    # main 'radio' dispatcher component
    radio_component_models = {name for name in _REGISTRY if name.startswith("radio_")}
    valid_types.update(radio_component_models)

    return frozenset(valid_types)


def _valid_xray_types() -> frozenset[str]:
    """Derive accepted ``xray={'type': ...}`` values from :data:`XRAY_MODELS`
    and SEDModelComponent X-ray variants.

    XRAY_MODELS contains the function-based models (``simple``, ``lopez24``,
    ``yang20`` alias, ``none`` disable). SEDModelComponent models like
    ``xray_aird`` and ``agn_xray_corona`` are also valid ``type`` values and
    must be discoverable to users. Both derive from the registry so the menu
    and the builder cannot drift (the failure mode behind #1120).

    ``"yang20"`` is registered in :data:`XRAY_MODELS` as an alias of
    ``"simple"``: tengri's X-ray component already implements the
    Yang+2020 physics (alpha_ox corona + Morrison & McCammon 1983 N_H +
    Compton/Thomson scattering) -- only the user-facing name was
    missing. See ``components/xray/xray.py`` for the formulas.
    """
    from tengri.components.xray import XRAY_MODELS
    from tengri.forward.component_factory import _REGISTRY

    # Function-based models: 'none', 'simple', 'yang20', 'lopez24'
    valid_types = set(XRAY_MODELS.keys())

    # SEDModelComponent models: 'xray_aird', 'agn_xray_corona'
    # Filter for components that contain 'xray_' to include both xray-prefixed
    # and agn_xray components
    xray_component_models = {name for name in _REGISTRY if "xray_" in name}
    valid_types.update(xray_component_models)

    return frozenset(valid_types)


def _valid_dust_laws() -> frozenset[str]:
    """Return accepted ``dust_attenuation.law`` values from the registry.

    Mirrors ``DUST_LAWS.keys()``, which the ``@register_dust_law`` decorator
    populates eagerly at import time of ``components.dust.attenuation``.
    No lazy-load wrinkle here (cf. dust emission), so the derivation is a
    direct view per ADR-0005 / ADR-0008.
    """
    from tengri.components.dust.attenuation import DUST_LAWS

    return frozenset(DUST_LAWS.keys())


def _agn_block_types(category: str) -> frozenset[str]:
    """Derive valid AGN block-type names for ``category`` from the registry.

    Previously each ``_VALID_AGN_*_TYPES`` set was hand-maintained, which
    silently drifted whenever a new block was registered without updating
    this file (#488's ``disc.skirtor`` / ``disc.schartmann2005`` /
    ``disc.adaf_lopez2024`` landed in :data:`AGN_BLOCKS` but were missing
    here, so the validator rejected them at build time with "Unknown
    agn_disc_block type 'skirtor'"). Derive from ``AGN_BLOCKS`` so the
    block-registration decorator is the single source of truth — the same
    fix pattern that closed the IGM ``meiksin06`` drift earlier.
    """
    # Force-import every block module so its ``@register_agn_block``
    # decorators have fired. Mirrors the eager imports done by AGN
    # ``unified.py`` at module-load time; safe to redo here.
    import tengri.components.agn.blocks.alternates
    import tengri.components.agn.blocks.atten
    import tengri.components.agn.blocks.blr
    import tengri.components.agn.blocks.disc
    import tengri.components.agn.blocks.feii
    import tengri.components.agn.blocks.grahsp_blocks
    import tengri.components.agn.blocks.nlr
    import tengri.components.agn.blocks.qsogen_blocks
    import tengri.components.agn.blocks.torus  # noqa: F401
    from tengri.components.agn.blocks._protocol import AGN_BLOCKS

    return frozenset(AGN_BLOCKS.get(category, {}).keys()) | {"none"}


#: Valid AGN disc block types (derived from ``AGN_BLOCKS['disc']``).
_VALID_AGN_DISC_TYPES = _agn_block_types("disc")

#: Valid AGN torus block types (derived from ``AGN_BLOCKS['torus']``).
_VALID_AGN_TORUS_TYPES = _agn_block_types("torus")

#: Valid AGN narrow-line region block types (derived from
#: ``AGN_BLOCKS['nlr']``).
_VALID_AGN_NLR_TYPES = _agn_block_types("nlr")

#: Valid AGN broad-line region block types (derived from ``AGN_BLOCKS['blr']``).
#: Includes ``"qsogen"`` which lives on the qsogen model.
_VALID_AGN_BLR_TYPES = _agn_block_types("blr") | {"qsogen"}

#: Valid AGN feii block types (derived from ``AGN_BLOCKS['feii']`` — the
#: block-registration decorator is the single source of truth, so a new feii
#: block like ``boroson_green`` is picked up automatically without editing here).
_VALID_AGN_FEII_TYPES = _agn_block_types("feii")

#: Valid AGN attenuation block types (derived from ``AGN_BLOCKS['attenuation']``).
_VALID_AGN_ATTEN_TYPES = _agn_block_types("attenuation")

#: Top-level groups whose ``type`` the round-trip must be able to emit even when
#: the group declares no parameters of its own. Exported (rather than inlined in
#: :func:`parameters_to_groups`) so the contract test's census is *derived* from
#: the emitter instead of retyped beside it.
_TOP_LEVEL_TYPED_GROUPS: frozenset[str] = frozenset(
    {"sfh", "dust_attenuation", "dust_emission", "neb", "shock", "igm", "radio", "xray", "agn"}
)

#: AGN sub-block name -> the ``Parameters`` attribute holding its selected type.
#:
#: Read in BOTH directions: ``_translate_agn_composable`` writes these attributes
#: from ``agn={'torus': {'type': ...}}``, and ``_extract_group_type`` reads them
#: back on the round-trip. One table, so the parser and the emitter cannot
#: disagree about where a sub-block's choice is stored — they did, and the
#: emitter simply returned ``None`` for the whole family (#1777).
_AGN_BLOCK_TO_KWARG: dict[str, str] = {
    "disc": "agn_disc_block",
    "torus": "agn_torus_block",
    "nlr": "agn_nlr_block",
    "blr": "agn_blr_block",
    "feii": "agn_feii_block",
    "atten": "agn_attenuation_block",
}

#: Partition table: agn_* param name -> group path (for sub-block routing).
#: Maps full agn_* param names to their owning group (agn, agn.disc, agn.torus, etc.)
_AGN_PARTITION = {
    # Shared params (no sub-block prefix)
    "agn_lum_ratio": "agn",
    "agn_log_lbol": "agn",
    "agn_alpha": "agn",
    "agn_log_mbh": "agn",
    "agn_log_ledd": "agn",
    "agn_a_spin": "agn",
    "agn_cos_inc": "agn",
    # Disc dust obscuration (Prevot SMC; AGNfitter EBVbbb). Shared (not
    # agn.disc): redden_disc applies it at the runner disc stage for every
    # disc type, mirroring agn_log_lbol.
    "agn_ebv_disc": "agn",
    # Torus
    "agn_T_torus": "agn.torus",
    "agn_T_hot": "agn.torus",
    "agn_T_warm": "agn.torus",
    "agn_frac_hot": "agn.torus",
    "agn_tau_torus": "agn.torus",
    "agn_tau": "agn.torus",  # Nenkova+2008 CLUMPY equatorial optical depth
    "agn_tau_skirtor": "agn.torus",
    "agn_p_skirtor": "agn.torus",
    "agn_q_skirtor": "agn.torus",
    "agn_oa_skirtor": "agn.torus",
    "agn_radius_ratio": "agn.torus",
    # SKIRTOR_mean_3p (AGNfitter-rX averaged) torus
    "agn_incl_skirtor": "agn.torus",
    "agn_tv_skirtor": "agn.torus",
    # CAT3D-Wind clumpy torus (Hönig & Kishimoto 2017)
    "agn_a_cat3d": "agn.torus",
    "agn_fwd_cat3d": "agn.torus",
    # Silva+04 obscured-torus column density
    "agn_log_nh_silva": "agn.torus",
    "agn_torus_frac": "agn.torus",
    # Fritz et al. (2006) smooth-dust torus
    "agn_fritz_r_ratio": "agn.torus",
    "agn_fritz_tau": "agn.torus",
    "agn_fritz_beta": "agn.torus",
    "agn_fritz_gamma": "agn.torus",
    "agn_fritz_oa": "agn.torus",
    "agn_fritz_psy": "agn.torus",
    # Narrow-line region
    "agn_nlr_cf": "agn.nlr",
    "agn_alpha_ion": "agn.nlr",  # NLR photoionization knob
    "neb_xid": "agn.nlr",  # Nebular ionization for NLR
    # Broad-line region
    "agn_blr_cf": "agn.blr",
    # FeII
    "agn_fe2_strength": "agn.feii",
    # Attenuation
    "agn_polar_ebv": "agn.atten",
    "agn_polar_oa": "agn.atten",
    "agn_polar_T": "agn.atten",
    "agn_polar_beta": "agn.atten",
    "agn_attenuation_ebv": "agn.atten",  # smc_prevot block E(B-V)
    # Radiation physics (shared disc normalization)
    "agn_f_hard": "agn",
    "agn_gamma_warm": "agn",
    "agn_kt_warm": "agn",
    "agn_gamma_hard": "agn",
    "agn_kt_hot": "agn",
    "agn_r_warm_ratio": "agn",
}

#: Top-level kwargs that are not groups (passed through to Parameters).
_TOP_LEVEL_SETTINGS = {
    "redshift",
    "n_grid",
    # Emission-line velocity mode. Activating it (``"fixed"``/``"marginalized"``/
    # ``"fitted"``) registers the line-velocity params (``eline_sigma_kms``,
    # ``eline_delta_v_kms``). ``SEDModel.build`` auto-propagates this from a
    # ``Spectroscopy`` observation so it need not be set twice (#653).
    "eline_mode",
}

#: Top-level kwargs that are SEDModel-only settings (silently ignored here).
#:
#: Some recipes (and user-facing nested dicts) carry build-time SEDModel
#: kwargs like ``approx=WavePrecomp()`` so that the same dict can be splatted
#: into either ``parse_groups(**d)`` (parameters only) or
#: ``SEDModel.build(**d)`` (parameters + model construction). These keys
#: are valid at the SEDModel layer but have no meaning for Parameters; we
#: drop them here rather than raise ``Unknown group key`` so the splat-both
#: pattern stays ergonomic.
_SEDMODEL_PASSTHROUGH = {
    "approx",
    "filters",
    "observation",
    "ssp_data",
}


# ── Main API ───────────────────────────────────────────────────────────────


def _normalize_wildcard_keys(group: object) -> object:
    """Rewrite the user-facing ``all_params`` wildcard key to internal ``'*'``.

    ``all_params`` is the one spelling the grammar accepts; ``'*'`` is an
    internal detail and is refused on input. This normalizer rewrites the
    user's key to the internal one once at the parser boundary, so every
    downstream site keeps operating on the single ``'*'`` invariant without
    that invariant ever being something a user has to know. Anything printing
    a normalized dict back to a user must undo this with
    :func:`_wildcard_keys_for_display`. Recurses through nested sub-block dicts
    (``dust.emission``, the ``agn.*`` selectors, ``igm.dla``, ``radio.sf`` /
    ``radio.agn``, …); per-parameter values are never dicts, so recursing into
    every dict-valued entry is safe.

    Parameters
    ----------
    group : object
        A group dict (or any value). Non-dict values pass through unchanged.

    Returns
    -------
    object
        A new dict with ``all_params`` keys rewritten to ``'*'`` (the input is
        never mutated), or the original value if it is not a dict.

    Raises
    ------
    ValueError
        If a dict carries ``'*'`` at all -- the key is retired, whether or not
        ``all_params`` is present beside it.

    Notes
    -----
    **JIT-compatible**: no — pure Python, runs at build time only.
    """
    if not isinstance(group, dict):
        return group
    # One rule, whether or not the dict also carries the preferred spelling:
    # the advice for both cases is to drop the star. Keeping a separate
    # "you set both" branch would have to name ``'*'`` as an option to explain
    # itself, which is a retirement error teaching the retired form.
    if WILDCARD_KEY in group:
        raise ValueError(
            f"The wildcard key {WILDCARD_KEY!r} has been retired; the wildcard "
            f"is spelled {WILDCARD_ALIAS!r}. Write "
            f"{{{WILDCARD_ALIAS!r}: FREE}} instead of {{{WILDCARD_KEY!r}: FREE}}."
        )
    normalized: dict[object, object] = {}
    for key, value in group.items():
        canonical_key = WILDCARD_KEY if key == WILDCARD_ALIAS else key
        normalized[canonical_key] = _normalize_wildcard_keys(value)
    return normalized


def _wildcard_keys_for_display(group: object) -> object:
    """Undo :func:`_normalize_wildcard_keys` for anything shown to a user.

    Normalization runs at the parser boundary, so every dict the translators
    hold already spells the wildcard ``'*'`` -- the internal key, which is
    refused on input. A message that formats such a dict verbatim therefore
    hands back a suggestion the parser rejects: exactly the retirement error
    that teaches the retired form. Render through this first.

    Parameters
    ----------
    group : object
        A group dict (or any value). Non-dict values pass through unchanged.

    Returns
    -------
    object
        A new dict with ``'*'`` keys rewritten to ``all_params`` (the input is
        never mutated), or the original value if it is not a dict.

    Notes
    -----
    **JIT-compatible**: no -- pure Python, runs at build time only.
    """
    if not isinstance(group, dict):
        return group
    return {
        (WILDCARD_ALIAS if key == WILDCARD_KEY else key): _wildcard_keys_for_display(value)
        for key, value in group.items()
    }


def parse_groups(**kwargs) -> Parameters:
    """Translate nested-dict model specification to Parameters.

    Parameters
    ----------
    **kwargs : keyword arguments
        Model configuration. Keys are group names (sfh, dust_attenuation,
        dust_emission, neb, igm, radio, xray, agn) or top-level settings
        (redshift, apply_igm, n_grid).
    _allow_empty_wildcard : bool, optional
        Private. When True, an ``all_params: FREE`` that frees nothing is
        permitted instead of raising :class:`~tengri.config.exceptions.ParameterError`.
        Reserved for introspection callers (:func:`~tengri.recipe_parameters`)
        that read ``all_params`` and do not care whether a parameter is free.
        Never set this when building a model to fit.

    Returns
    -------
    Parameters
        A fully initialized Parameters object ready for inference.

    Raises
    ------
    ValueError
        If an unknown group key, unknown type value, or unknown parameter
        name is provided. Also raised if both deprecated 'lines' and new
        'nlr'/'blr' sub-blocks are provided under 'agn'.

    Notes
    -----
    **JIT-compatible**: no — this is a pure Python translator.

    **AGN blocks**: The new composable AGN grammar supports independent
    ``nlr`` (narrow-line region) and ``blr`` (broad-line region) sub-blocks
    under the ``agn`` group. The deprecated ``lines`` sub-block is expanded
    to an (nlr, blr) pair with a ``DeprecationWarning``.

    Examples
    --------
    >>> from tengri.parameters import parse_groups, FREE, FIXED
    >>> params = parse_groups(
    ...     sfh={"type": "dpl", "all_params": FREE},
    ...     redshift=0.1,
    ... )
    >>> assert "sfh_dpl_alpha" in params.free_params
    """
    # Private introspection escape hatch — popped before group parsing so it is
    # never mistaken for a group name.
    allow_empty_wildcard = bool(kwargs.pop("_allow_empty_wildcard", False))

    # Redshift is required, and the question asked here is whether the caller
    # PASSED it -- not what its value is. A value-based sentinel cannot answer
    # that: any object standing in for "absent" is also a legal prior. The one
    # tried first was ``Uniform(0.0, 10.0)``, which is the most natural photo-z
    # prior in this package's target science and compares equal to a user's own
    # ``Uniform(0, 10)`` with an identical repr. That left two silent failures,
    # one on each side of the identity check: compare with ``is`` and any path
    # that REBUILDS the default (a copy, a serializer, a to_groups round-trip)
    # yields a free z in [0, 10] fit where no redshift was given; tighten it to
    # ``==`` and every genuine photo-z fit over that range is refused as
    # missing. Presence has neither failure mode. Introspection callers pass
    # ``_allow_empty_wildcard`` and legitimately have no redshift.
    redshift_was_given = "redshift" in kwargs

    # ── Pass 0a: Normalize the preferred ``all_params`` wildcard alias ──
    # Rewrite ``all_params`` -> ``'*'`` in every group dict (and nested
    # sub-block) so all downstream logic operates on the single canonical key.
    kwargs = {key: _normalize_wildcard_keys(value) for key, value in kwargs.items()}
    # ── Pass 0b: Normalize the sfh 'field' modulator sub-block ─────────
    # Rewrite sfh={'type': t, 'field': {...}} → sfh={'type': [t, 'field'], ...}
    # so the stochastic field is reachable from the natural nested-dict idiom.
    # Runs after wildcard normalization, so a field sub-block written with the
    # ``all_params`` alias is already canonicalized to ``'*'`` before scoping.
    kwargs = _normalize_sfh_field(kwargs)

    # ── Pass 1: Translate structural choices ──────────────────────────

    structural_kwargs = _translate_structural(kwargs)

    # ── Pass 2: Resolve per-parameter values ──────────────────────────

    # Build a structural Parameters to get the declared parameter list.
    #
    # Advisories are silenced here: this spec exists only to enumerate which
    # parameters the selected components declare. Its values are registry
    # defaults that pass 2 has not resolved and grid-narrowing has not touched
    # yet, so any advisory it raises describes a state the caller never asked
    # for and is about to be superseded — it would report the *pre-narrowing*
    # range as a defect after that range has already been fixed (#1586).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AdvisoryWarning)
        structural_params = Parameters(**structural_kwargs)

    # Partition declared params by owning group. ``met_*`` lands in
    # ``"stellar"`` when the user opted into the new top-level slot
    # (issue #311); otherwise it stays in ``"sfh"`` so the legacy
    # ``sfh={'*': FIXED}`` wildcard keeps cascading over met_* params
    # — preserves pre-#311 behavior for every fixture/recipe that didn't
    # pass a ``met={}`` block.
    dust_emission_active = structural_params.dust_emission is not None
    has_met_block = isinstance(kwargs.get("met"), dict)
    # ``met`` owns met_* when present (#1720), else ``sfh`` — the pre-#311
    # default, kept so a legacy ``sfh={'*': FIXED}`` wildcard still cascades
    # over met_* for fixtures and recipes that pass no metallicity block.
    param_partition = _partition_by_group(
        structural_params.all_params,
        dust_emission_active,
        met_group="met" if has_met_block else "sfh",
    )

    # Resolve each parameter's final distribution
    resolved_kwargs = dict(structural_kwargs)
    provenance: dict[str, str] = {}

    # Which parameters each group's ``all_params: FREE`` may free, scoped to the
    # structural variant that group selected. Computed once, consulted once in
    # the resolve loop below — see :func:`_wildcard_scopes` for why every group
    # with a structural axis needs an entry and what happens when one lacks it.
    wildcard_scopes = _wildcard_scopes(structural_kwargs, structural_params, param_partition)

    # Outcome of every *active* ``all_params: FREE`` wildcard, keyed by the
    # group it was written in. ``FREE`` resolves to the registry default, which
    # for most parameters is a ``Fixed`` scalar — so a wildcard can legally
    # resolve without freeing anything. That silently produces a model whose
    # physics is pinned at defaults while the user believes it is being fitted.
    # Collected here and adjudicated by ``_check_wildcard_freed_something``.
    wildcard_free_outcome: dict[str, list[tuple[str, bool]]] = {}

    # Track if we've emitted the met_* suppression warning for this parse (#1796).
    # Emit once per parse even if multiple met_* params are suppressed.
    met_suppression_warned = False

    for param_name in structural_params.all_params:
        group = param_partition.get(param_name, None)
        if group is None:
            # _structural or unknown; skip
            continue

        # Handle top-level parameters specially (sentinels, direct values)
        if group == "_toplevel":
            if param_name in kwargs:
                val = kwargs[param_name]
                # Resolve sentinels
                if val is FREE:
                    resolved_kwargs[param_name] = structural_params.get_distribution(param_name)
                    provenance[param_name] = "user_free"
                elif val is FIXED:
                    registry_default = structural_params.get_distribution(param_name)
                    if registry_default.is_fixed:
                        resolved_kwargs[param_name] = registry_default
                    else:
                        resolved_kwargs[param_name] = Fixed(
                            _default_fixed_value(param_name, registry_default)
                        )
                    provenance[param_name] = "user_fixed"
                else:
                    if isinstance(val, Distribution):
                        resolved_kwargs[param_name] = val
                        provenance[param_name] = "user_fixed" if val.is_fixed else "user_prior"
                    else:
                        resolved_kwargs[param_name] = Fixed(val)
                        provenance[param_name] = "user_fixed"
            continue

        # Get the group dict from the input for group-based parameters
        if group == "igm.dla":
            # DLA sub-block: prefer the new nested-dict form
            # ``igm={'dla': {'log_n_hi': ..., ...}}`` (closes #507), but
            # fall back to the flat form where the builder factories emit
            # ``igm={'dla': True, 'log_n_hi': ...}`` with the DLA params
            # at the same level as the ``dla`` flag.
            igm_dict = kwargs.get("igm") if isinstance(kwargs.get("igm"), dict) else {}
            dla_val = igm_dict.get("dla", False) if igm_dict else False
            group_dict = dla_val if isinstance(dla_val, dict) else igm_dict
        elif group == "agn" or group.startswith("agn."):
            # AGN params live in a two-level nest: shared (`agn` itself) and
            # five sub-blocks (`agn.disc`, `agn.torus`, `agn.lines`, `agn.feii`,
            # `agn.atten`). Users naturally place a shared parameter inside
            # a sub-block (`agn={'disc': {'agn_log_lbol': Uniform(...)}}`)
            # or — less commonly — a sub-block parameter at the top level.
            # Both should work. Build a merged search view across the
            # canonical location and the sibling locations; conflicts raise.
            group_dict = _build_agn_search_view(param_name, kwargs.get("agn", {}), group)
        elif group == "radio.sf" or group == "radio.agn":
            # Radio sub-blocks: descend into radio={'sf': {...}} / {'agn': {...}}
            # (mirrors the dust.emission sub-group path above).
            radio_top = kwargs.get("radio")
            radio_top = radio_top if isinstance(radio_top, dict) else {}
            subkey = "sf" if group == "radio.sf" else "agn"
            sub = radio_top.get(subkey, {})
            group_dict = sub if isinstance(sub, dict) else {}
        else:
            group_dict = kwargs.get(group, {})

        # Determine final distribution
        if not isinstance(group_dict, dict):
            # Group was not a dict (or is a sub-key); use structural default
            continue

        # Special case: met_* parameters in sfh group (no met block) should be
        # pinned at Fixed defaults, not freed by sfh wildcard (#1796).
        # These params logically belong to "met" group. When there's an explicit
        # sfh override (e.g., sfh={'logzsol': Uniform(...)}) honor it with a
        # migration warning. Otherwise, treat as implicit FIXED wildcard.
        if group == "sfh" and param_name.startswith("met_"):
            short_name = _extract_short_name(param_name, group_dict)
            has_explicit_override = short_name in group_dict or param_name in group_dict

            if not has_explicit_override and group_dict.get("*") is FREE and not has_met_block:
                # sfh wildcard is FREE, but met_* should not be freed (no met
                # block means implicit FIXED). Override with FIXED for this param.
                group_dict = {"*": FIXED}

                # Emit migration warning once per parse (#1796)
                if not met_suppression_warned:
                    warnings.warn(
                        (
                            "sfh={'all_params': FREE} no longer frees metallicity "
                            "parameters when there is no explicit met block. "
                            "Before this change, met_logzsol (and other met_* params) "
                            "were freed by the sfh wildcard.\n\n"
                            "To free metallicity parameters explicitly, pass either:\n"
                            "  met={'all_params': FREE}\n"
                            "or:\n"
                            "  met={'logzsol': Uniform(-2, 0.2)}\n\n"
                            "Issue #1796"
                        ),
                        WildcardPartialFreeWarning,
                        stacklevel=2,
                    )
                    met_suppression_warned = True

        # A group whose wildcard is scoped to its selected structural variant
        # frees only what that variant reads; a parameter outside the scope
        # resolves to ``wildcard_fixed_inactive`` and stays declared-but-Fixed.
        # ``None`` (the group has no structural axis, or its variant declares
        # nothing introspectable) leaves the wildcard unnarrowed.
        is_agn = group == "agn" or group.startswith("agn.")
        scope = wildcard_scopes.get(group)
        wildcard_active = scope is None or param_name in scope
        final_dist, tag = _resolve_value(
            param_name,
            group_dict,
            structural_params.get_distribution(param_name),
            wildcard_active=wildcard_active,
        )

        # Record whether an active wildcard-FREE actually freed this parameter.
        # ``wildcard_fixed_inactive`` is deliberate block scoping, not a failure,
        # so only the active branch is tracked.
        if tag == "wildcard_free":
            freed = not final_dist.is_fixed
            wildcard_free_outcome.setdefault(group, []).append((param_name, freed))
            # Report the *outcome*, not the request. A wildcard-FREE that found
            # no declared prior leaves the parameter Fixed, and tagging it
            # "[all_params FREE]" put a row reading FREE inside the Fixed block
            # of ``spec.summary()`` — the one table a user consults to answer
            # "what did I hold constant?" (#1726). WildcardPartialFreeWarning
            # says so at build time, but a notebook can miss or filter it, and
            # the summary is what gets read afterwards. Same principle as the
            # "_grid" tags below: shown, never silent (#1586).
            if not freed:
                tag = "wildcard_free_pinned"

        # Apply the resolved distribution when the user addressed this group
        # (a per-param override or wildcard), or for any AGN param whenever the
        # AGN group is configured. The AGN clause is essential: AGN parameters
        # now carry *free* Uniform/LogUniform registry defaults (so FREE can
        # expand them), which means an untouched AGN param would otherwise keep
        # its free prior — violating the grammar's FIXED-by-default contract.
        # ``_resolve_value`` collapses such untouched params to Fixed(default)
        # via its registry-default branch, so applying it here pins them fixed
        # unless explicitly/wildcard-freed. (Non-AGN params keep Fixed scalar
        # registry defaults, so the original ``group_dict`` guard is correct.)
        if group_dict or group == "_toplevel" or is_agn:
            resolved_kwargs[param_name] = final_dist
            provenance[param_name] = tag

    # ── Optional Cue knobs (density / abundances / ionizing spectrum) ──
    # These carry None priors so they never appear on ``structural_params``
    # or in ``param_partition`` above — the resolution loop skips them.
    # Forward any the user set in a ``type='cue'`` neb group straight to the
    # flat constructor, which registers them on demand (#653). Only explicit
    # priors / values are accepted (no '*' wildcard or bare sentinel, since
    # these params have no registry default to expand a FREE/FIXED against).
    neb_group = kwargs.get("neb")
    if isinstance(neb_group, dict) and neb_group.get("type") == "cue":
        for pname in _OPTIONAL_NEB_PARAM_NAMES:
            if pname not in neb_group:
                continue
            val = neb_group[pname]
            if isinstance(val, Distribution):
                resolved_kwargs[pname] = val
                provenance[pname] = "user_prior"
            elif val is FREE or val is FIXED:
                raise ValueError(
                    f"neb[{pname!r}] needs an explicit prior or value "
                    f"(e.g. Uniform(lo, hi) or a number); the 'all_params' wildcard "
                    f"(also '*') and "
                    f"bare FREE/FIXED are unsupported for optional Cue knobs "
                    f"because they carry no registry default."
                )
            else:
                resolved_kwargs[pname] = Fixed(val)
                provenance[pname] = "user_fixed"

    # ── Validate every key the user supplied was recognized ───────────
    # The resolution loop above silently uses the registry default when
    # a parameter override is not found, so typos like
    # ``dust={'tau_qpah': 5}`` (instead of ``dust_qpah``) used to vanish
    # without trace. Walk the user's dicts now and raise a friendly
    # "Did you mean ...?" error on any unrecognized key.
    _validate_user_keys(kwargs, structural_params, param_partition)

    # ── Validate every ``all_params: FREE`` actually freed something ───
    # Runs after key validation so a typo is reported before this, which is
    # the more fundamental error.
    if not allow_empty_wildcard:
        _check_wildcard_freed_something(
            _narrow_outcome_to_selected_component(wildcard_free_outcome, structural_params)
        )

    # ── Construct final Parameters ────────────────────────────────────

    _narrow_free_priors_to_grid(resolved_kwargs, provenance, structural_params)

    final_params = Parameters(**resolved_kwargs)
    # Fill in provenance for params not touched by user/wildcard
    for name in list(final_params._distributions.keys()):
        provenance.setdefault(name, "registry_default")
    object.__setattr__(final_params, "_group_provenance", provenance)

    # Raised HERE, after the groups have been translated and validated, not at
    # the top. A caller with a malformed group AND no redshift should hear about
    # the group: it is the more specific of the two complaints, and the one they
    # can act on. Checking first made `parse_groups(radio=True)` answer with
    # "redshift is required" instead of the bool-gate advice that explains what
    # `radio=True` should have been.
    if not redshift_was_given and not allow_empty_wildcard:
        raise ValueError(
            "redshift is required. Specify one of:\n"
            "  - redshift=Fixed(z)             a known redshift\n"
            "  - redshift=Uniform(lo, hi)      a photo-z fit\n"
            "  - redshift=<any Distribution>   any other prior\n"
            "\n"
            "It used to default to Fixed(0.1), which put every model that "
            "omitted it at z=0.1 without saying so."
        )

    _warn_silently_fixed_parameters(final_params, param_partition, kwargs)
    _warn_firrc_slope_degeneracy(final_params)

    return final_params


#: Marks a provenance tag whose prior was intersected with a component's grid.
_GRID_NARROWED_SUFFIX = "_grid"

#: Marks a wildcard-FREE that found no declared prior and left the parameter
#: Fixed. Both suffixes annotate the *outcome* of a resolution whose *intent*
#: is carried by the base tag, so :func:`_base_provenance` strips either.
_WILDCARD_PINNED_SUFFIX = "_pinned"

#: Least fraction of a declared range that may survive an automatic narrowing.
#:
#: Trimming a modest dead tail is a tidy-up. Cutting a 2.5 dex prior down to
#: 0.04 dex is not — it pins the parameter in all but name, and doing that
#: silently is worse than the dead tail it removes. Below this, the narrowing
#: is declined and the overhang is reported instead.
_MIN_RETAINED_FRACTION = 0.10


def _base_provenance(tag: str) -> str:
    """Strip outcome markers, leaving how the value was *chosen*.

    ``wildcard_free_grid`` still means "this came from ``all_params: FREE``";
    the suffix only records that the declared range was then intersected with
    the selected component's grid. Round-tripping through
    :func:`parameters_to_groups` has to see the base tag, or a narrowed
    parameter loses its wildcard intent and gets emitted as an explicit
    override instead of collapsing back into ``all_params``.

    ``wildcard_free_pinned`` is the same shape (#1726): the wildcard did reach
    the parameter, it simply found no declared prior and left it Fixed. The
    suffix exists so ``spec.summary()`` can report the outcome rather than the
    request, and must not make the parameter round-trip as an explicit override
    — the user asked for ``all_params: FREE`` and that is what ``to_groups()``
    should hand back.

    Parameters
    ----------
    tag : str
        Raw provenance tag.

    Returns
    -------
    str
        The tag without its outcome marker.
    """
    for suffix in (_GRID_NARROWED_SUFFIX, _WILDCARD_PINNED_SUFFIX):
        if tag.endswith(suffix):
            return tag[: -len(suffix)]
    return tag


#: Provenance tags whose prior came from the *declaration*, not from the user.
#:
#: ``FREE`` is a request to sample a parameter, not a statement about its range
#: — the range comes from the declared ``free_prior``. Those are the only ones
#: safe to narrow. ``user_prior`` means the caller wrote a distribution by hand;
#: overriding that would be silently substituting a different model for the one
#: they asked for, so it is left alone and merely warned about.
_DECLARATION_SOURCED_FREE = frozenset({"user_free", "wildcard_free"})


def _selected_component(selector: str, structural) -> str | None:
    """Which component is selected for ``selector``, or ``None``.

    Derives the structural attribute from the selector rather than consulting a
    second table: ``"dust.emission"`` -> ``dust_emission``, ``"agn.disc"`` ->
    ``agn_disc_block``. An earlier draft hand-maintained that mapping, which
    made it a second census that had to agree with
    :data:`~tengri.components.grid_support.GRID_SUPPORT` — registering a
    component there would silently not be narrowed, with nothing failing. That
    is the same drift this whole area exists to remove, so it is derived.

    Parameters
    ----------
    selector : str
        Dotted selector, e.g. ``'agn.disc'``.
    structural : Parameters
        Structural-only spec carrying the component selections.

    Returns
    -------
    str or None
        The selected component name, or ``None`` when this spec makes no such
        selection.

    Notes
    -----
    **JIT-compatible**: not applicable — composition-time only.

    Narrowing keys on the *selected* component, which is what makes it safe for
    a shared parameter: ``agn_log_mbh`` is consumed by the analytic discs too,
    and they are untouched because they carry no grid.
    """
    base = selector.replace(".", "_")
    for attr in (base, f"{base}_block"):
        value = getattr(structural, attr, None)
        if isinstance(value, str):
            return value
    return None


def _narrow_free_priors_to_grid(
    resolved: dict, provenance: dict[str, str], structural: dict
) -> None:
    """Intersect a declaration-supplied free prior with its component's grid.

    A template-backed component clips its parameters onto the grid axes it
    interpolates over, where ``jnp.clip`` is flat: the SED is bit-identical and
    the gradient is exactly zero (#1586). ``astrodust`` ships the live case —
    ``dust_lgU`` declares ``free_prior=Uniform(0, 7)`` against a grid of
    ``[-3, 6]``, so ``all_params: FREE`` handed a sampler a range whose top
    14.3% could not move. Intersecting removes the dead region instead of only
    warning about it.

    Mutates ``resolved`` in place and retags ``provenance`` so
    :meth:`~tengri.parameters.parameters.Parameters.summary` shows the narrowing
    rather than silently reporting a range the caller never wrote.

    Parameters
    ----------
    resolved : dict
        Resolved ``{param_name: Distribution}`` kwargs, mutated in place.
    provenance : dict of str to str
        Resolution tag per parameter.
    structural : Parameters
        Structural-only spec carrying the component selections, e.g. its
        ``dust_emission`` attribute.

    Notes
    -----
    **JIT-compatible**: not applicable — composition-time only.

    Deliberately narrow in scope:

    - Only :class:`~tengri.parameters.priors.Uniform` is narrowed. Intersecting
      a Normal or LogUniform with a box is not a truncation of the same family,
      so those keep their prior and fall through to the warning.
    - The **declaration is never touched**. ``dust_lgU`` is shared with
      ``draine2021_pah_ir``, whose grid need not match ``astrodust``'s, and
      ``dust_umin``'s cap is 20 / 50 / 80 depending only on which backend is
      selected. That is why the constraint lives per ``(component, parameter)``
      — see :mod:`tengri.components.grid_support`.
    - Narrowing only ever shrinks. The grid may extend *below* a declared
      bound (astrodust reaches ``lgU = -3`` where the declaration floors at 0);
      widening there would assert physics the declaration deliberately excluded.
    """
    from tengri.components.grid_support import GRID_SUPPORT, grid_support
    from tengri.parameters.priors import Uniform

    # Drive off the registry itself, so registering a component is the only
    # step needed for it to be narrowed as well as reported.
    for selector in sorted({sel for sel, _ in GRID_SUPPORT}):
        name = _selected_component(selector, structural)
        if name is None:
            continue
        for pname, (g_lo, g_hi) in grid_support(selector, name).items():
            if provenance.get(pname) not in _DECLARATION_SOURCED_FREE:
                continue
            dist = resolved.get(pname)
            if not isinstance(dist, Uniform):
                continue
            lo, hi = dist.bounds
            new_lo, new_hi = max(lo, g_lo), min(hi, g_hi)
            if new_lo >= new_hi or (new_lo <= lo and new_hi >= hi):
                # Disjoint (nothing sensible to narrow to — let the warning
                # say so) or already contained.
                continue
            if (new_hi - new_lo) < _MIN_RETAINED_FRACTION * (hi - lo):
                # The declaration and the grid barely overlap, so "narrowing"
                # would replace a physical range with a sliver — effectively
                # pinning the parameter without saying so. agn_log_ledd is the
                # case: Uniform(-2, 0.5) against a grid ending at -1.9586
                # retains 1.7%, a 0.04 dex prior. A disagreement that large is
                # a modeling decision, not a tidy-up: leave it and let the
                # warning report it.
                continue
            default = dist.default
            if default is not None:
                default = min(max(default, new_lo), new_hi)
            resolved[pname] = Uniform(
                new_lo,
                new_hi,
                dist.description,
                units=dist.units,
                default=default,
            )
            provenance[pname] = provenance[pname] + "_grid"


#: Sub-block group name -> the ``structural_params`` attribute naming the
#: component selected for it. Only groups listed here are narrowed; add an
#: entry when another sub-block's parameter partition is wider than any one
#: component's declarations.
_SUBBLOCK_COMPONENT_ATTR: dict[str, str] = {"dust_emission": "dust_emission"}


def _declared_param_names(component_type: str) -> frozenset[str] | None:
    """Prefixed parameter names a registered component declares.

    Reads the class-level ``_priors`` that ``SEDModelComponent.__init_subclass__``
    populates, so no instance is built.

    Parameters
    ----------
    component_type : str
        Grammar type name as written in the spec, e.g. ``"dale2014"`` or
        ``"dl07"``. Grammar names that are aliases (``dl07`` -> ``draine_li2007``,
        ``mbb`` -> ``modified_blackbody``) are resolved through
        ``_EMISSION_TYPE_ALIASES`` first; looking them up raw misses the class
        and reports the engine as declaration-free, which silently disables both
        the guard and the wildcard scoping for five production engines.

    Returns
    -------
    frozenset of str or None
        Prefixed names (``dust_alpha_dale``, ...), or ``None`` when the type is
        not a registered component or declares nothing — the caller then leaves
        that group unnarrowed rather than guessing.

    Notes
    -----
    An empty ``_priors`` is **three** questions wearing one shape, and inferring
    from it is wrong for two of them. The component is asked instead:

    * ``declares_no_parameters = True``: genuinely parameter-free
      (``pah_drude``, a pure template shape). Narrowing to the empty set is
      correct: it is what every engine that *does* declare priors gets.
    * ``reads_parameters = {...}``: reads real knobs declared elsewhere.
      ``energy_balance_split``'s warm/cold knobs live in
      ``components/dust/_params.py`` because re-declaring them beside the
      attenuator's would raise a duplicate declaration.
    * neither — unknown, so return ``None`` and leave the wildcard alone rather
      than guess.

    Returning an empty frozenset for the second case would pin every one of that
    engine's parameters; returning ``None`` for the first leaves it freeing the
    whole static union, which is #1482. Both markers exist because the two
    failures are opposite and neither is introspectable.
    """
    from tengri.forward.component_factory import _EMISSION_TYPE_ALIASES, _REGISTRY

    cls = _REGISTRY.get(_EMISSION_TYPE_ALIASES.get(component_type, component_type))
    priors = getattr(cls, "_priors", None)
    if not priors:
        # Empty ``_priors`` is ambiguous, so ask rather than infer. A component
        # that declares ``declares_no_parameters`` is stating it reads nothing,
        # and narrowing it to the empty set is then correct -- it is what every
        # engine that *does* declare priors already gets. Without the marker the
        # wildcard is left alone, because the other reading of an empty _priors
        # is ``energy_balance_split``, whose six knobs live in
        # ``components/dust/_params.py`` and would all be pinned.
        if getattr(cls, "declares_no_parameters", False):
            return frozenset()
        # ...and a component that names the parameters it reads elsewhere is
        # answering the same question from the other side. Without this branch
        # ``energy_balance_split`` stays unnarrowed and ``'*': FREE`` hands the
        # sampler 20 parameters for the 6 it reads (measured).
        declared_elsewhere = getattr(cls, "reads_parameters", None)
        if declared_elsewhere:
            return frozenset(declared_elsewhere)
        return None
    prefix = getattr(cls, "parameter_prefix", "")
    return frozenset(prefix + name for name in priors)


def _narrow_outcome_to_selected_component(
    outcome: dict[str, list[tuple[str, bool]]],
    structural_params,
) -> dict[str, list[tuple[str, bool]]]:
    """Restrict a sub-block's wildcard outcome to its selected component's params.

    A sub-block's parameter partition spans every backend it can dispatch to —
    ``dust.emission`` covers 22 names across nine engines. Seven of those carry
    distribution registry defaults and so are freed by ``'*': FREE`` whichever
    engine is selected, which makes the group-level ``any(freed)`` test in
    :func:`_check_wildcard_freed_something` unfalsifiable: it cannot fire even
    when the selected engine got none of *its* parameters freed (#1482).

    ``dale2014`` declares ``dust_alpha_dale`` and ``dust_frac_agn``; both default
    to ``Fixed`` scalars, so ``emission={'type':'dale2014','*':FREE}`` freed seven
    parameters belonging to THEMIS, DL07/DL14, MBB and Schreiber, and none the
    engine reads. Narrowing to the declared set restores the guard.

    Parameters
    ----------
    outcome : dict
        Group name -> list of ``(param_name, was_freed)``.
    structural_params : StructuralParams
        Carries the selected component per sub-block.

    Returns
    -------
    dict
        ``outcome`` with narrowed sub-block entries; other groups pass through.
    """
    narrowed = dict(outcome)
    for group, attr in _SUBBLOCK_COMPONENT_ATTR.items():
        if group not in narrowed:
            continue
        component_type = getattr(structural_params, attr, None)
        if component_type is None:
            continue
        declared = _declared_param_names(component_type)
        if declared is None:
            continue  # unregistered or declaration-free: keep the old behavior
        freed = {name for name, was_freed in narrowed[group] if was_freed}
        # Rebuild from the declared set, not by filtering: a declared parameter
        # the wildcard never covered must count as not-freed, or dropping it
        # would empty the list and skip the check entirely.
        narrowed[group] = [(name, name in freed) for name in sorted(declared)]
    return narrowed


def _format_stuck(group: str, stuck: list[str]) -> tuple[str, str, str]:
    """Render the pinned-parameter list and a short-form example for a group.

    Parameters
    ----------
    group : str
        Group name, possibly dotted for a sub-block (``'agn.torus'``).
    stuck : list of str
        Fully-prefixed names of the parameters that stayed ``Fixed``.

    Returns
    -------
    shown : str
        Comma-joined names, truncated with a ``(+N more)`` tail.
    top : str
        Top-level group name — what the caller writes as the kwarg.
    example : str
        A ready-to-paste snippet freeing the first stuck parameter, nested at
        the level the grammar accepts.

    Notes
    -----
    The snippet used to be assembled by the callers as
    ``f"{top}={{{short!r}: Uniform(lo, hi)}}"``, which flattens a sub-block
    parameter to its top-level group: for ``dust.emission`` that produced
    ``dust={'alpha_dale': Uniform(lo, hi)}``. The dust level used to accept
    that spelling and silently discard it, so the advice appeared to work and
    changed nothing; it is now refused outright. Either way the recommendation
    has to name the nesting the parameter actually lives at.
    """
    # Show enough that the caller can see what they meant to free without
    # scrolling; groups run to ~22 params at the widest (dust.emission).
    _LIMIT = 12
    shown = ", ".join(stuck[:_LIMIT]) + (
        f", ... (+{len(stuck) - _LIMIT} more)" if len(stuck) > _LIMIT else ""
    )
    top = group.split(".")[0]
    prefix = f"{top}_"
    first = stuck[0]
    short = first[len(prefix) :] if first.startswith(prefix) else first
    sub = group.split(".", 1)[1] if "." in group else None
    inner = f"{{{short!r}: Uniform(lo, hi)}}"
    example = f"{top}={{{sub!r}: {inner}}}" if sub else f"{top}={inner}"
    return shown, top, example


def _check_wildcard_freed_something(
    outcome: dict[str, list[tuple[str, bool]]],
) -> None:
    """Adjudicate what an ``all_params: FREE`` wildcard actually freed.

    ``FREE`` resolves each parameter to its declared ``free_prior``. A parameter
    with no ``free_prior`` falls back to its ``prior`` — a ``Fixed`` scalar — and
    stays pinned. A group can therefore hold both kinds, and the wildcard frees
    one subset while leaving the rest frozen. The fit then runs to completion
    with that physics constant, which is indistinguishable from success at the
    call site.

    Three outcomes, three responses:

    * freed everything — silent, the request was honored;
    * freed nothing — :class:`ParameterError`, since that is never intended;
    * freed some — :class:`WildcardPartialFreeWarning` naming what stayed pinned
      (issue #1474). A warning rather than a refusal because a partial free is
      sometimes correct: ``dust_Rv`` is fixed by definition under a Calzetti law,
      and six of the ten shipped recipes free a strict subset today.

    Parameters
    ----------
    outcome : dict
        Group name -> list of ``(param_name, was_freed)`` for every parameter an
        *active* wildcard-FREE touched. Blocks scoped out by an inactive model
        selection are excluded by the caller and never reach here.

    Raises
    ------
    ParameterError
        If any group's wildcard freed zero of the parameters it covered.

    Warns
    -----
    WildcardPartialFreeWarning
        If a group's wildcard freed some, but not all, of them.
    """
    for group, entries in sorted(outcome.items()):
        if not entries:
            continue
        stuck = [name for name, freed in entries if not freed]
        if not stuck:
            # Freed everything it covered — exactly what was asked for.
            continue

        shown, _top, example = _format_stuck(group, stuck)

        if len(stuck) < len(entries):
            warnings.warn(
                f"'all_params: FREE' freed {len(entries) - len(stuck)} of "
                f"{len(entries)} parameters in group {group!r}. These have no "
                f"declared prior, only Fixed defaults, so they stay pinned:\n"
                f"  {shown}\n"
                f"The fit will run with that physics held constant. Pass "
                f"explicit priors for the ones you meant to vary, e.g. "
                f"{example}, or filter "
                f"WildcardPartialFreeWarning if this is deliberate.",
                WildcardPartialFreeWarning,
                stacklevel=3,
            )
            continue

        raise ParameterError(
            f"'all_params: FREE' freed 0 of {len(stuck)} parameters in group "
            f"{group!r}. These have no declared prior, only Fixed defaults:\n"
            f"  {shown}\n"
            f"FREE resolves to each parameter's registry default, and these "
            f"default to Fixed — so the wildcard would leave every one of them "
            f"pinned and the fit would silently not vary this physics.\n"
            f"Pass explicit priors instead, e.g. {example}."
        )


def _warn_silently_fixed_parameters(
    final_params: Parameters, param_partition: dict[str, str], kwargs: dict
) -> None:
    """Warn when a parameter group silently fixes parameters (no disposition).

    Parameters
    ----------
    final_params : Parameters
        The resolved Parameters object with _group_provenance set.
    param_partition : dict[str, str]
        Maps parameter names to their group (group name or "_toplevel", "_structural").
    kwargs : dict
        The original user-provided kwargs to parse_groups.

    Notes
    -----
    Emits DefaultFixedParametersWarning when a group's parameters are marked as
    "registry_default" provenance AND are Fixed, meaning the user stated no
    'all_params' disposition and those params were pinned at defaults.
    One warning per group, listing the first ~8 parameters and their values.
    """
    provenance = getattr(final_params, "_group_provenance", {})

    # Collect parameters fixed by default (registry_default + Fixed)
    # grouped by their parameter group
    default_fixed_by_group: dict[str, list[tuple[str, float]]] = {}

    # Check whether user provided an explicit met block
    has_met_block = isinstance(kwargs.get("met"), dict)

    for param_name in final_params._distributions:
        # Skip if not in provenance (shouldn't happen) or if not registry_default
        if provenance.get(param_name) != "registry_default":
            continue

        # Skip if not Fixed
        dist = final_params._distributions[param_name]
        if not dist.is_fixed:
            continue

        # Get the group this parameter belongs to
        group = param_partition.get(param_name)
        if group is None or group == "_structural" or group == "_toplevel":
            # Skip structural and toplevel parameters
            continue

        # ``met_*`` sits in the ``sfh`` partition when the user passed no ``met``
        # block, by design (#311/#1720, see ``met_group=`` above). Warning about
        # it under the ``sfh`` label would name a group the user never wrote and
        # hand them a remedy that does not apply: ``sfh={'all_params': FIXED}``
        # says nothing about metallicity. Naming the wrong group is worse than
        # staying quiet, so stay quiet.
        if group == "sfh" and param_name.startswith("met_") and not has_met_block:
            continue

        # Collect this parameter as silently-fixed
        value = dist.default
        default_fixed_by_group.setdefault(group, []).append((param_name, value))

    # A group that yielded at least one free parameter cannot have hit the
    # failure this warning exists to catch -- "I configured a group and got
    # nothing free out of it" (#1995). Counting free parameters per group is
    # the test; engagement deliberately is not, because a group can be engaged
    # and still yield nothing free, which is the footgun itself:
    #
    #     sfh={"type": "dpl", "alpha": Fixed(1.5)}   # engaged, n_free == 0
    #
    # Before this, the condition was "did the group state a disposition", a
    # proxy that misfires on the standard ``met={"logzsol": Uniform(...)}``
    # spelling and put 115 warnings into 9 published renders.
    free_by_group: dict[str, int] = {}
    for param_name, dist in final_params._distributions.items():
        if dist.is_fixed:
            continue
        owning_group = param_partition.get(param_name)
        if owning_group is None:
            continue
        free_by_group[owning_group] = free_by_group.get(owning_group, 0) + 1

    # For each group with silently-fixed parameters, check if the user
    # explicitly stated a disposition. Only warn if they didn't.
    for group, params_and_values in default_fixed_by_group.items():
        if free_by_group.get(group, 0):
            continue
        # Determine if the user actually provided this group in kwargs
        if group.startswith("dust."):
            # Sub-group like dust.emission
            parent_group = "dust"
            user_provided = parent_group in kwargs
            group_dict = kwargs.get(parent_group, {})
            if isinstance(group_dict, dict):
                subkey = group.replace("dust.", "")
                group_dict = group_dict.get(subkey, {})
            else:
                group_dict = {}
        elif group == "agn" or group.startswith("agn."):
            user_provided = "agn" in kwargs
            group_dict = kwargs.get("agn", {})
        else:
            user_provided = group in kwargs
            group_dict = kwargs.get(group, {})

        # Only warn if the user explicitly provided this group
        if not user_provided:
            continue

        if not isinstance(group_dict, dict):
            # Group was provided but not as a dict, so skip
            continue

        # Check if user stated a disposition in their provided dict
        has_explicit_disposition = (
            "all_params" in group_dict and group_dict["all_params"] in (FREE, FIXED)
        ) or ("*" in group_dict and group_dict["*"] in (FREE, FIXED))

        if has_explicit_disposition:
            # User explicitly stated a disposition, so don't warn
            continue

        # Format the parameter list: first ~8 params with values, then ellipsis if more
        formatted_params = []
        for i, (pname, value) in enumerate(params_and_values):
            if i >= 8:
                formatted_params.append(f"... and {len(params_and_values) - 8} more")
                break
            formatted_params.append(f"{pname}={value:.4g}")

        # Say how many actually defaulted, never "all": a group commonly sets
        # some parameters explicitly and leaves the rest to the default.
        n_params = len(params_and_values)
        subject = "parameter" if n_params == 1 else f"{n_params} parameters"
        verb = "was" if n_params == 1 else "were"

        message = (
            f"Group {group!r} states no 'all_params' disposition, so its remaining "
            f"{subject} {verb} fixed at declared defaults:\n"
            f"  {', '.join(formatted_params)}\n\n"
            f"To fit them, pass 'all_params': FREE:\n"
            f"  {group}={{'all_params': FREE, ...}}\n"
            f"To keep them fixed and silence this warning, say so explicitly:\n"
            f"  {group}={{'all_params': FIXED, ...}}"
        )

        warnings.warn(
            message,
            DefaultFixedParametersWarning,
            stacklevel=4,  # Point to user's parse_groups call
        )


def _warn_firrc_slope_degeneracy(final_params: Parameters) -> None:
    """Warn when a FIRRC *slope* coefficient is freed (per-galaxy degeneracy).

    The mass/redshift FIRRC slopes vary q_IR *across* a sample; at one
    galaxy's fixed (M*, z) they collapse to a single scalar, degenerate with
    the ``radio_*_q0`` normalization. Freeing them only makes sense as
    ``PopulationFitter`` hyperparameters — see ADR-0018 §8a. The
    :class:`RadioFIRRCDegeneracyWarning` category is filterable so a
    deliberate hierarchical fit can silence it.
    """
    from tengri.components.radio._params import (
        FIRRC_SLOPE_PARAMS,
        RadioFIRRCDegeneracyWarning,
    )

    freed = sorted(FIRRC_SLOPE_PARAMS.intersection(final_params.free_params))
    if not freed:
        return
    warnings.warn(
        f"Radio FIRRC slope coefficient(s) {freed} are free, but the FIR-radio "
        f"correlation slopes are degenerate with the normalization at a single "
        f"galaxy's fixed (M*, z) — they map to one scalar q_IR, so a per-galaxy "
        f"fit cannot constrain them. Free the normalization instead "
        f"('radio_delv_q0' / 'radio_mcch_q0' / 'radio_q_ir') for the radio-excess "
        f"amplitude, and reserve the slopes for PopulationFitter hyperparameters "
        f"(see ADR-0018 §8a). Filter RadioFIRRCDegeneracyWarning to silence this "
        f"for a deliberate hierarchical fit.",
        RadioFIRRCDegeneracyWarning,
        stacklevel=3,
    )


# ── Internal helpers ───────────────────────────────────────────────────────


def _agn_active_param_set(structural_kwargs: dict) -> frozenset[str]:
    """Params a group-level AGN wildcard should free, scoped to active blocks.

    Thin wrapper over
    ``tengri.components.agn.blocks._consumes.agn_active_param_set``,
    lazy-imported to avoid an import cycle (the agn package imports the priors
    layer that ultimately re-exports this module).
    """
    from tengri.components.agn.blocks._consumes import agn_active_param_set

    return agn_active_param_set(structural_kwargs)


def _law_shape_params(law_name: str) -> frozenset[str]:
    """Declared parameters the named attenuation law reads, from its signature.

    Parameters
    ----------
    law_name : str
        Key in ``DUST_LAWS`` as written in the grammar, e.g. ``"calzetti"``.

    Returns
    -------
    frozenset of str
        Flat parameter names (``dust_Rv``, ``dust_slope``, ...). Empty for a law
        that reads nothing beyond wavelength, and for an unregistered name.

    Notes
    -----
    **JIT-compatible**: no — signature introspection at build time.

    Read off the function signature rather than a maintained table, so a law
    registered later is scoped without editing this module. Every law also
    takes ``**kwargs``, which is exactly why the signature is the only honest
    source: ``def calzetti(wavelength, **_kwargs)`` *accepts* ``dust_Rv`` and
    silently discards it, so "does the call succeed?" cannot answer "does this
    law read this parameter?" — only the named parameters can.

    Two spellings reach the same quantity: the law kwarg (``n_slope``) and the
    flat parameter (``dust_slope``). ``_TWO_COMPONENT_LAW_PARAMS`` is the
    existing map between them; a signature name already spelled ``dust_*`` is
    its own flat name.
    """
    from tengri.components.dust._apply import _TWO_COMPONENT_LAW_PARAMS
    from tengri.components.dust.laws._registry import DUST_LAWS

    entry = DUST_LAWS.get(law_name)
    if entry is None:
        return frozenset()
    fn = entry["fn"] if isinstance(entry, dict) else entry

    kwarg_to_flat = {kwarg: flat for kwarg, flat, _ in _TWO_COMPONENT_LAW_PARAMS}
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):  # pragma: no cover - builtins have no signature
        return frozenset()

    names = set()
    for param in sig.parameters.values():
        if param.kind is param.VAR_KEYWORD:
            continue
        flat = kwarg_to_flat.get(param.name)
        if flat is not None:
            names.add(flat)
        elif param.name.startswith("dust_"):
            names.add(param.name)
    return frozenset(names)


@lru_cache(maxsize=1)
def _all_law_shape_params() -> frozenset[str]:
    """Every parameter any registered attenuation law reads.

    Returns
    -------
    frozenset of str
        Union of :func:`_law_shape_params` over ``DUST_LAWS``.

    Notes
    -----
    **JIT-compatible**: no — signature introspection, cached after the first
    call because the law registry is populated at import time and every
    ``parse_groups`` would otherwise re-introspect all 22 laws.

    The ``dust`` wildcard frees the group's parameters *minus* the members of
    this set that the selected laws do not read, so parameters no law takes as
    a shape argument — the Charlot & Fall optical depths — are never narrowed
    away by a law that happens to be shape-free.
    """
    from tengri.components.dust.laws._registry import DUST_LAWS

    return frozenset().union(*(_law_shape_params(name) for name in DUST_LAWS), frozenset())


#: Law slots a two-component/single-component dust model can select. Each is an
#: independent choice, so a shape parameter is live if *any* slot's law reads it.
_DUST_LAW_SLOTS: tuple[str, ...] = ("dust_law_bc", "dust_law_diff", "dust_law_neb")


def _wildcard_scopes(
    structural_kwargs: dict,
    structural_params: Parameters,
    param_partition: dict[str, str],
) -> dict[str, frozenset[str] | None]:
    """Per-group scope for ``all_params: FREE``, keyed by group name.

    Parameters
    ----------
    structural_kwargs : dict
        Translated structural choices (``dust_law_bc``, ``shock_norm``, ...).
    structural_params : Parameters
        Structural-only spec, used for the selected ``dust_emission`` engine.
    param_partition : dict
        Parameter name -> owning group, from :func:`_partition_by_group`.

    Returns
    -------
    dict
        Group name -> the parameters its wildcard may free, or ``None`` to leave
        the wildcard unnarrowed. Groups absent from the mapping are unnarrowed.

    Notes
    -----
    **JIT-compatible**: no — build-time introspection.

    A group's declared parameters span every structural variant it can dispatch
    to, but a build selects one. Freeing the whole superset hands the sampler
    dimensions the selected variant never reads: flat directions explored at
    full cost whose posterior comes back equal to the prior, with nothing in the
    fit saying why. That is #1482, measured there as six inert free dimensions
    under ``schreiber2016`` while the engine's own ``dust_T`` — worth 94% of the
    dust IR SED — stayed pinned.

    The failure recurred because each group was scoped where it was noticed:
    AGN, then ``radio.sf``/``radio.agn``, then ``dust.emission``, three
    mechanisms behind three branches in the resolver. A fourth group needing it
    was a fourth branch, and until someone wrote that branch the group failed
    silently. Collecting the scopes here makes the resolver's use of them
    single and uniform, so adding a group is an entry rather than a branch.

    ``None`` and ``frozenset()`` are different answers and the distinction is
    load-bearing: ``None`` means "not scoped, free whatever the group declares",
    while an empty set would pin every parameter in the group. A variant that
    declares nothing introspectable must map to ``None`` — narrowing it to the
    empty set would cause the very failure this exists to prevent (see
    ``_declared_param_names`` on ``energy_balance_split``).
    """
    scopes: dict[str, frozenset[str] | None] = {}

    # ── AGN: every agn.* sub-block shares the active-block scope ──
    agn_active = _agn_active_param_set(structural_kwargs)
    for group in set(param_partition.values()):
        if group == "agn" or group.startswith("agn."):
            scopes[group] = agn_active

    # ── radio: the selected sf mode / agn model ──
    scopes["radio.sf"] = _RADIO_SF_PARAMS_BY_MODE.get(
        structural_kwargs.get("radio_sfr_mode"), frozenset()
    )
    scopes["radio.agn"] = _RADIO_AGN_PARAMS_BY_MODEL.get(
        structural_kwargs.get("radio_agn_model"), frozenset()
    )

    # ── dust_emission: the selected IR engine's own declarations ──
    scopes["dust_emission"] = (
        _declared_param_names(structural_params.dust_emission)
        if structural_params.dust_emission is not None
        else None
    )

    # ── dust: the attenuation laws the selected slots name ──
    # The group owns both the optical depths (which every law consumes through
    # the Charlot & Fall geometry, not as a curve-shape argument) and the four
    # curve-shape modifiers, which only some laws read. Narrow by removing the
    # shape parameters no selected law names, leaving everything else free.
    dust_group_params = {
        name for name, grp in param_partition.items() if grp == "dust_attenuation"
    }
    if dust_group_params:
        # Read the slots off ``structural_params``, not ``structural_kwargs``:
        # a slot the user did not name is absent from the kwargs but still
        # resolves to a real law (both default to ``power_law``, which reads
        # ``dust_slope``). Consulting the raw kwargs would narrow away a
        # parameter the law in force does read — the failure this prevents,
        # inverted.
        active_shape = frozenset().union(
            *(
                _law_shape_params(law)
                for law in (getattr(structural_params, slot, None) for slot in _DUST_LAW_SLOTS)
                if law is not None
            ),
            frozenset(),
        )
        scopes["dust_attenuation"] = frozenset(
            dust_group_params - (_all_law_shape_params() - active_shape)
        )

    # ── xray: the selected corona model ──
    # Read the model off ``structural_params`` for the same reason the dust
    # slots are: an ``xray`` group that names no model still resolves to one.
    xray_group_params = {name for name, grp in param_partition.items() if grp == "xray"}
    if xray_group_params:
        xray_model = getattr(structural_params, "xray_model", None) or structural_kwargs.get(
            "xray_model"
        )
        reads = _XRAY_PARAMS_BY_MODEL.get(xray_model, _XRAY_DEFAULT_MODEL_PARAMS)
        scopes["xray"] = frozenset(
            xray_group_params - (_XRAY_VARIANT_PARAMS - reads) - _XRAY_UNREACHABLE_PARAMS
        )

    # ── shock: the selected normalization ──
    # ``norm='frac'`` scales the galaxy Halpha via ``shock_frac``; ``'lhalpha'``
    # sets an absolute ``shock_log_lhalpha``. Each reads one and ignores the
    # other, so an unscoped wildcard always frees one inert luminosity scale.
    shock_group_params = {name for name, grp in param_partition.items() if grp == "shock"}
    if shock_group_params:
        unused = (
            "shock_log_lhalpha"
            if structural_kwargs.get("shock_norm", "frac") != "lhalpha"
            else "shock_frac"
        )
        scopes["shock"] = frozenset(shock_group_params - {unused})

    # Note: sfh scoping is handled differently in parse_groups() —
    # met_* parameters in sfh (when no met block) skip the group dict to avoid
    # forcing them through wildcard scoping (see line ~850 for the check).

    return scopes


def _translate_structural(groups: dict) -> dict:
    """Resolve each group's `type` choice into the matching Parameters kwargs."""
    # Derived, not restated. This was a second hand-maintained copy of the group
    # list, and the two had to be edited together with nothing checking that they
    # were: a group added to ``_GROUP_STRUCTURAL_KEYS`` alone would be rejected
    # here as unknown, and one added only here would accept any key at all.
    # Dotted entries are sub-blocks (``dust.emission``, ``igm.dla``), reached
    # through their parent rather than named at top level.
    valid_groups = {k for k in _GROUP_STRUCTURAL_KEYS if "." not in k}
    # Add "dust" to valid_groups so old dust= syntax can be caught and given a helpful error
    valid_groups.add("dust")
    result = {}

    # Check for ambiguity: both old dust= and new dust_attenuation=/dust_emission=
    has_dust_old = "dust" in groups and isinstance(groups.get("dust"), dict)
    has_dust_atten = "dust_attenuation" in groups and isinstance(
        groups.get("dust_attenuation"), dict
    )
    has_dust_emis = "dust_emission" in groups and isinstance(groups.get("dust_emission"), dict)
    if has_dust_old and (has_dust_atten or has_dust_emis):
        raise ValueError(
            "Ambiguous dust specification: both old `dust=` and new `dust_attenuation=` / "
            "`dust_emission=` groups are present. Choose one style:\n"
            "Old (retired): dust={...} with optional dust={'emission': {...}} nested\n"
            "New: dust_attenuation={...}, dust_emission={...} as separate top-level entries.\n"
            "Remove the old `dust=` form."
        )

    # Suggested model when someone tries the (unsupported) bool form for an
    # additive gate group — used only to make the error message actionable.
    # radio: Condon (1992) radio-IR correlation; xray: 'simple' (X-ray binaries
    # + AGN corona; 'simple' ≡ 'yang20' physics); shock: MAPPINGS V.
    _GATE_SUGGESTED_TYPE = {"radio": "condon92", "xray": "simple", "shock": "mappings"}

    for group_name, group_dict in groups.items():
        if group_name in _TOP_LEVEL_SETTINGS:
            continue
        if group_name in _SEDMODEL_PASSTHROUGH:
            # SEDModel-only kwarg (e.g. ``approx=WavePrecomp()``) splatted
            # in from a recipe — ignore at the parameters layer.
            continue

        if group_name == "stellar":
            raise ValueError(
                "the 'stellar' group is gone (#1720); its one setting was the "
                "metallicity mode, and that now lives in 'met', which selects "
                "with 'type' like every other group. "
                "A 'met_mode' key becomes met={'type': ...} — so the tabulated "
                "mode is met={'type': 'table'} — and a 'met_logzsol' key becomes "
                "met={'logzsol': ...}. "
                "Two spellings of one setting was the maintenance cost this "
                "removes; tengri.list_metallicity_modes() shows the current form."
            )

        if group_name == "apply_igm":
            raise ValueError(
                "apply_igm is retired. IGM activation is now derived from the igm dict: "
                "pass igm={'type': 'inoue'} (or 'madau', 'meiksin06') to enable IGM, or "
                "omit the igm dict (or pass igm={'type': 'none'}) to disable it."
            )

        if group_name not in valid_groups:
            suggestions = difflib.get_close_matches(group_name, valid_groups, n=2, cutoff=0.6)
            suggest_str = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
            raise ValueError(
                f"Unknown group key '{group_name}'. "
                f"Valid groups: {', '.join(sorted(valid_groups))}.{suggest_str}"
            )

        # Every component is declared the same way — a dict selecting the
        # model, e.g. ``neb={'type': 'cue'}``. The bool form (``xray=True``)
        # is NOT supported: it used to fall through the ``isinstance(dict)``
        # skip below and leave the component silently absent, so the
        # panchromatic recipes shipped with no radio / X-ray at all. Reject it
        # loudly and point at the consistent dict grammar rather than guessing.
        if group_name in _GATE_SUGGESTED_TYPE and isinstance(group_dict, bool):
            suggested = _GATE_SUGGESTED_TYPE[group_name]
            if group_name == "radio":
                # Radio uses the composable form, not the legacy type form
                sf_spec = "{'type': 'bell2003'}"
                agn_spec = "{'type': 'powerlaw'}"
                raise ValueError(
                    f"{group_name}={group_dict!r} is not a valid declaration — declare "
                    f"radio with the composable surface using 'sf' and 'agn' keys: "
                    f"radio={{'sf': {sf_spec}, 'agn': {agn_spec}}} "
                    f"to enable (add per-param priors), or omit to disable."
                )
            raise ValueError(
                f"{group_name}={group_dict!r} is not a valid declaration — declare "
                f"{group_name} like every other component, with a dict selecting the "
                f"model: {group_name}={{'type': '{suggested}'}} to enable (add per-param "
                f"priors as needed), or {group_name}={{'type': 'none'}} / omit to disable."
            )

        if not isinstance(group_dict, dict):
            continue

        if group_name == "sfh":
            _translate_sfh(group_dict, result)
        elif group_name == "met":
            _translate_met(group_dict, result)
        elif group_name == "dust":
            # Old dust= form is retired; handle with loud error
            _translate_dust_retired(group_dict, result)
        elif group_name == "dust_attenuation":
            _translate_dust_attenuation(group_dict, result)
        elif group_name == "dust_emission":
            _translate_dust_emission(group_dict, result)
        elif group_name == "neb":
            _translate_neb(group_dict, result)
        elif group_name == "shock":
            _translate_shock(group_dict, result)
        elif group_name == "igm":
            _translate_igm(group_dict, result)
        elif group_name == "radio":
            _translate_radio(group_dict, result)
        elif group_name == "xray":
            _translate_xray(group_dict, result)
        elif group_name == "foreground":
            _translate_foreground(group_dict, result)
        elif group_name == "agn":
            _translate_agn(group_dict, result)

    # Top-level settings win over group-derived ones; sentinels (FREE/FIXED)
    # are resolved later in _resolve_value, not here.
    for key in list(groups.keys()):
        if key in _TOP_LEVEL_SETTINGS:
            val = groups[key]
            if val is not FREE and val is not FIXED:
                result[key] = val

    return result


#: Field (stochastic IFT modulator) PSD parameter short names. The field is a
#: DRW-governed correlated field; these are the priors a ``field`` sub-block
#: scopes its ``'*'`` wildcard over. Keep in sync with the field SFH registry.
_FIELD_PARAM_SHORT = ("psd_sigma", "psd_tau_myr")


def _normalize_sfh_field(kwargs: dict) -> dict:
    """Accept ``sfh={'type': <smooth>, 'field': {...}}`` for the stochastic field.

    The IFT correlated-field burstiness is a *modulator* composed with a smooth
    SFH, so internally it lives in ``mean_sfh_type`` as the list
    ``[<smooth>, 'field']``. That list form has always worked
    (``sfh={'type': ['dpl', 'field']}``) but is not the natural nested-dict
    idiom — a user mirroring ``dust={'emission': {...}}`` reaches for
    ``sfh={'type': 'dpl', 'field': {...}}`` and hit a bare "Unknown key 'field'".

    Rewrite that sub-block form into the list-``type`` composition *before* the
    validator and translator run, so the whole existing grammar (validation,
    partition, wildcard/param resolution) handles it unchanged. The ``field``
    sub-dict carries the PSD priors (``psd_sigma``, ``psd_tau_myr``) and an
    optional ``'*'`` wildcard scoped to just those field params (the group-level
    ``'*'`` still governs the smooth SFH).
    """
    sfh = kwargs.get("sfh")
    if not isinstance(sfh, dict) or "field" not in sfh:
        return kwargs

    sfh = dict(sfh)  # copy — never mutate the caller's dict
    field_block = sfh.pop("field")

    base = sfh.get("type", "dpl")
    types = list(base) if isinstance(base, (list, tuple)) else [base]
    if "field" not in types:
        types.append("field")
    sfh["type"] = types

    if field_block in (True, None):
        pass  # enable with default field priors
    elif isinstance(field_block, dict):
        star = field_block.get("*")
        if star is not None:
            # Scope the field wildcard to the field params only (so the smooth
            # SFH keeps its own '*' / defaults).
            for short in _FIELD_PARAM_SHORT:
                sfh.setdefault(short, star)
        for key, value in field_block.items():
            if key == "*":
                continue
            sfh[key] = value  # explicit per-param prior overrides the wildcard
    else:
        raise ValueError(
            "sfh 'field' must be a dict of field/PSD priors (e.g. "
            "sfh={'type': 'dpl', 'field': {'*': FREE}}), or True to enable it "
            "with defaults."
        )

    return {**kwargs, "sfh": sfh}


def _validate_sfh_bin_edges(sfh_type, edges) -> None:
    """Validate ``sfh['bin_edges_gyr']` at build time.

    Delegates to the registry so the grammar and the stellar component apply one
    rule; a second copy is what let #1975 ship with a green suite.
    """
    from tengri.components.stellar.sfh.registry import validate_bin_edges_gyr

    validate_bin_edges_gyr(sfh_type, edges)


def _translate_sfh(sfh_dict: dict, result: dict) -> None:
    """Resolve `sfh.type` (or a list composition) into `mean_sfh_type`.

    Also forwards the (non-parametric only) ``bin_edges_gyr`` structural
    kwarg through to :func:`resolve_sfh` so users can override the
    bin layout for ``prospector_beta`` / ``continuity`` / etc. from the
    nested-dict grammar (#337).
    """
    sfh_type = sfh_dict.get("type")

    # ``bin_edges_gyr`` is a structural setting (array of bin edges in
    # Gyr) that only applies to non-parametric SFHs. Surface it as a
    # top-level kwarg so ``Parameters.__init__`` can pop it and forward
    # to ``resolve_sfh(mean_sfh_type, bin_edges_gyr=...)`` via
    # ``_build_legacy``. The wildcard ``'*': FREE / FIXED`` does NOT
    # apply to this — it's a config, not a free parameter.
    if "bin_edges_gyr" in sfh_dict:
        _validate_sfh_bin_edges(sfh_type, sfh_dict["bin_edges_gyr"])
        result["bin_edges_gyr"] = sfh_dict["bin_edges_gyr"]

    # ``age_kernel`` is likewise a structural setting, not a free parameter:
    # which kernel integrates the SFH onto the SSP age grid ("cic" / "dsps").
    # Validated here so a typo fails at build time with the valid set, rather
    # than silently falling back to the default at the first prediction (#964).
    if "age_kernel" in sfh_dict:
        from tengri.components.stellar.component import VALID_AGE_KERNELS

        age_kernel = sfh_dict["age_kernel"]
        if age_kernel is not None and age_kernel not in VALID_AGE_KERNELS:
            raise ValueError(
                f"Unknown sfh age_kernel {age_kernel!r}. "
                f"Valid: {', '.join(repr(k) for k in VALID_AGE_KERNELS)} "
                f"(or None to auto-select). 'cic' is the accuracy default; "
                f"'dsps' selects DSPS's histogram kernel for cross-code "
                f"comparison (biases the optical CSP +1.2 %, #964)."
            )
        # Pass 0b has already folded any ``sfh={'field': {...}}`` sub-block into
        # the type list, so the incompatible pair is knowable HERE — at
        # ``SEDModel.build`` — rather than at the first prediction, which for a
        # fit means after warmup has already started. The component-level
        # ``_resolve_age_kernel`` still guards direct construction.
        _types = sfh_dict.get("type") or []
        if age_kernel == "cic" and "field" in (
            _types if isinstance(_types, (list, tuple)) else [_types]
        ):
            raise NotImplementedError(
                "sfh age_kernel='cic' is not supported with a GP-field SFH — "
                "the field draw is defined on its own coarse lookback grid, so "
                "there is no dense integrand to cloud-in-cell (#964). Drop the "
                "field modulator to use the CIC kernel, or set "
                "age_kernel='dsps' explicitly to acknowledge the field path's "
                "kernel."
            )
        result["age_kernel"] = age_kernel

    # ``field_centering`` is a structural setting too: WHICH COORDINATES the GP
    # field latent is sampled in, not a physical parameter. ``a = 1`` is the
    # shipped non-centered map ``s = L(sigma, tau) xi``; ``a < 1`` moves
    # amplitude dependence out of that map (#1355). Validated here, beside
    # ``age_kernel``, so an out-of-range value or a request on a field-less SFH
    # fails at ``SEDModel.build`` rather than at the first prediction.
    if "field_centering" in sfh_dict:
        centering = sfh_dict["field_centering"]
        try:
            centering = float(centering)
        except (TypeError, ValueError):
            raise ValueError(
                f"sfh field_centering must be a number between 0 and 1, got "
                f"{centering!r}. 1.0 (default) is the non-centered "
                f"parameterization; 0.0 is fully centered (#1355)."
            ) from None
        if not 0.0 <= centering <= 1.0:
            raise ValueError(
                f"sfh field_centering must be between 0 and 1, got {centering!r}. "
                f"It interpolates the parameterization: 1.0 (default) samples the "
                f"standardized latent, 0.0 samples the field itself (#1355)."
            )
        _types = sfh_dict.get("type") or []
        _types = _types if isinstance(_types, (list, tuple)) else [_types]
        if "field" not in _types:
            # An explicit request the model cannot serve must raise, not no-op:
            # there is no GP field here to reparameterize, and silently keeping
            # the value would be #1488's "selectable but inert" exactly.
            raise ValueError(
                f"sfh field_centering={centering!r} needs a GP-field SFH — it "
                f"reparameterizes the field latent, and this SFH has no field "
                f"component. Add it with sfh={{'type': ['dpl', 'field'], ...}}, "
                f"or drop field_centering (#1355)."
            )
        result["field_centering"] = centering

    if sfh_type is None:
        result["mean_sfh_type"] = ["dpl", "field"]
        return

    valid = _valid_sfh_types()
    if not isinstance(sfh_type, (str, list)):
        raise TypeError(
            f"sfh 'type' must be a string (or a list of strings for a "
            f"composition), got {type(sfh_type).__name__}: {sfh_type!r}. "
            f"Example: sfh={{'type': 'delayed', 'all_params': FIXED}}."
        )
    if isinstance(sfh_type, list):
        for type_name in sfh_type:
            if type_name not in valid:
                suggestions = difflib.get_close_matches(type_name, valid, n=3, cutoff=0.6)
                suggest_str = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
                raise ValueError(f"Unknown SFH type '{type_name}' in composition.{suggest_str}")
        result["mean_sfh_type"] = sfh_type
        return

    from tengri.components.stellar.sfh.registry import UNVALIDATED_SFH_TYPES

    if sfh_type in UNVALIDATED_SFH_TYPES:
        raise ValueError(
            f"SFH type '{sfh_type}' is registered but not yet validated against "
            f"the DSPS forward path, so it is not available via the builder. Use a "
            f"validated SFH (see tengri.builders.sfh / list of accepted types)."
        )

    if sfh_type not in valid:
        suggestions = difflib.get_close_matches(sfh_type, valid, n=3, cutoff=0.6)
        suggest_str = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        raise ValueError(f"Unknown SFH type '{sfh_type}'.{suggest_str}")

    result["mean_sfh_type"] = sfh_type


def _translate_met(met_dict: dict, result: dict) -> None:
    """Resolve ``met.type`` into ``met_mode`` — the parallel of ``sfh.type`` (#1720).

    ``sfh`` and ``met`` describe the same thing from two angles: how much mass
    formed when, and at what metallicity. They should read the same at the call
    site, and until #1720 they did not — the metallicity mode lived under
    ``stellar`` and used ``met_mode`` where every other group uses ``type``.

    That asymmetry is what produced #1677: ``Catalog.from_histories`` advised
    ``met={'type': 'table'}``, the form both conventions imply, and the grammar
    rejected it. The advice was right and the grammar was the outlier, so the
    grammar moved.

    ``stellar={'met_mode': ...}`` (the #311 spelling) is **gone**, not
    deprecated: two spellings of one setting is a maintenance cost with no
    upside, and every call site in the repo moved with this change. Passing
    ``stellar=`` now raises with the one-line translation.

    Parameters
    ----------
    met_dict : dict
        The ``met=`` group. ``'type'`` selects a mode from ``MET_REGISTRY``.
    result : dict
        Parameters kwargs being assembled; ``met_mode`` is written into it.

    Raises
    ------
    ValueError
        If ``type`` is not a registered metallicity mode.

    Notes
    -----
    **JIT-compatible**: no — construction-time grammar translation.
    """
    if "met_mode" in met_dict:
        raise ValueError(
            "the met group selects its mode with 'type', like every other group "
            "— not with 'met_mode'. Write met={'type': 'table'}. ('met_mode' is "
            "the key of the older met={'type': ...} spelling, which also "
            "still works; this mixes the two.)"
        )
    _set_met_mode(met_dict.get("type"), result, key="met={'type': ...}")


def _set_met_mode(met_mode, result: dict, *, key: str) -> None:
    """Validate a metallicity mode and record it, whichever spelling supplied it."""
    from tengri.components.stellar.sfh.met_registry import MET_REGISTRY

    if met_mode is None:
        # No explicit mode; let auto-inference (from per-param keys) decide.
        return

    valid_modes = sorted(MET_REGISTRY.keys())
    if met_mode not in MET_REGISTRY:
        suggestions = difflib.get_close_matches(met_mode, valid_modes, n=2, cutoff=0.6)
        suggest_str = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        raise ValueError(
            f"Unknown metallicity mode '{met_mode}' in {key}. Valid modes: "
            f"{', '.join(valid_modes)}.{suggest_str}"
        )

    result["met_mode"] = met_mode


def _translate_dust_attenuation(dust_atten_dict: dict, result: dict) -> None:
    """Translate dust_attenuation group to dust_model and law settings.

    Resolves dust type against the set of supported dust models
    (two_component, single_component, wg00) and extracts structural
    configuration and law selections. Preserves the law validation rules
    from PR #1984: law XOR (law_bc AND law_diff); single_component takes
    only law; wg00 takes none.
    """
    dust_type = dust_atten_dict.get("type", "two_component")

    # 'none'/'off' disable the dust block entirely — parity with neb/agn/radio/
    # xray/igm/shock, all of which accept type='none' (and the generic grammar
    # error even promises it). The forward model reads dust_model=='off' as
    # use_dust=False, so normalize both spellings onto that sentinel and skip
    # law parsing (there is nothing to attenuate).
    if dust_type in ("none", "off"):
        result["dust_model"] = "off"
        return

    # Lyman-limit clip is wired only through the two-component screen. Flag any
    # other type rather than silently dropping the request (single-component,
    # WG00, and SEDModelComponents do not route through it yet).
    if dust_atten_dict.get("lyman_cutoff") and dust_type != "two_component":
        raise ValueError(
            f"dust_attenuation 'lyman_cutoff' is only supported for type='two_component' "
            f"(got type={dust_type!r}). Use a two-component dust_attenuation block, or drop "
            f"'lyman_cutoff'."
        )

    # Validate type against hard-coded dust model types
    if dust_type not in _VALID_DUST_TYPES:
        # A common mistake (#664): passing an attenuation *law* name as the dust
        # ``type``. Laws (calzetti, smc, salim_sbl18, …) are not standalone dust
        # models — they are selected with the ``law`` key inside a
        # ``single_component`` or ``two_component`` block. Point there instead of
        # emitting a bare "unknown type" so the request is not lost.
        if dust_type in _valid_dust_laws():
            raise ValueError(
                f"'{dust_type}' is a dust attenuation *law*, not a dust model type. "
                f"Select it with the 'law' key on a dust_attenuation model, e.g. a single "
                f"screen dust_attenuation={{'type': 'single_component', 'law': '{dust_type}', "
                f"'tau_v': ...}}, or birth-cloud + ISM "
                f"dust_attenuation={{'type': 'two_component', 'law': '{dust_type}', "
                f"'tau_bc': ..., 'tau_diff': ...}} -- use 'law_bc' and 'law_diff' instead only "
                f"to give the two screens different laws. Valid dust types are: "
                f"{', '.join(sorted(_VALID_DUST_TYPES))}."
            )
        suggestions = difflib.get_close_matches(dust_type, _VALID_DUST_TYPES, n=2, cutoff=0.6)
        suggest_str = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        raise ValueError(f"Unknown dust_attenuation type '{dust_type}'.{suggest_str}")

    result["dust_model"] = dust_type

    # Reject nested dust_attenuation={'emission': ...} — emission is now a top-level group
    if "emission" in dust_atten_dict:
        # Build a helpful translation showing the concrete split
        atten_part = {k: v for k, v in dust_atten_dict.items() if k != "emission"}
        emission_part = dust_atten_dict["emission"]

        # Merge legacy law forms (same rule as dust= retirement)
        dust_type = atten_part.get("type", "two_component")
        atten_part_translated = {
            k: v for k, v in atten_part.items() if k not in ("law", "law_bc", "law_diff")
        }

        law_note = ""

        if dust_type == "two_component":
            has_law = "law" in atten_part
            has_law_bc = "law_bc" in atten_part
            has_law_diff = "law_diff" in atten_part

            if has_law:
                atten_part_translated["law"] = atten_part["law"]
            elif has_law_bc and has_law_diff:
                atten_part_translated["law_bc"] = atten_part["law_bc"]
                atten_part_translated["law_diff"] = atten_part["law_diff"]
            elif has_law_bc:
                atten_part_translated["law"] = atten_part["law_bc"]
                law_note = (
                    "\n\nNote: The old form had 'law_bc' alone, which pre-#1989 applied "
                    "to both screens. We merged it to 'law' to preserve that behavior."
                )
            elif has_law_diff:
                atten_part_translated["law_diff"] = atten_part["law_diff"]
                atten_part_translated["law_bc"] = "power_law"
                law_note = (
                    "\n\nNote: The old form had only 'law_diff'; the 'law_bc' screen "
                    "defaulted to power_law. Verify whether power_law was intended for "
                    "the birth cloud."
                )
            else:
                atten_part_translated["law"] = "power_law"
                law_note = (
                    "\n\nNote: The dust_attenuation suggestion includes 'law': "
                    "'power_law' — before PR #1989, a missing law defaulted to power_law. "
                    "This reproduces the old result exactly, but you should verify that "
                    "power_law was your intended choice and not just an "
                    "accidentally-omitted setting."
                )

        elif dust_type == "single_component":
            has_law = "law" in atten_part
            has_law_bc = "law_bc" in atten_part

            if has_law:
                atten_part_translated["law"] = atten_part["law"]
            elif has_law_bc:
                atten_part_translated["law"] = atten_part["law_bc"]
                law_note = (
                    "\n\nNote: The old form allowed 'law_bc' on single_component; "
                    "we merged it to 'law'."
                )
            else:
                atten_part_translated["law"] = "power_law"
                law_note = (
                    "\n\nNote: The dust_attenuation suggestion includes 'law': "
                    "'power_law' — before PR #1989, a missing law defaulted to power_law. "
                    "This reproduces the old result exactly, but you should verify that "
                    "power_law was your intended choice and not just an "
                    "accidentally-omitted setting."
                )

        atten_str = (
            "dust_attenuation={"
            + ", ".join(
                f"'{k}': {v!r}"
                for k, v in sorted(_wildcard_keys_for_display(atten_part_translated).items())
            )
            + "}"
        )
        emission_str = (
            "dust_emission={"
            + ", ".join(
                f"'{k}': {v!r}"
                for k, v in sorted(_wildcard_keys_for_display(emission_part).items())
            )
            + "}"
        )

        message = (
            "dust_attenuation={'emission': ...} is retired; "
            "IR emission is now a separate top-level group. "
            "Replace\n"
            f"    dust_attenuation={{..., 'emission': {{...}}}}\n"
            "with\n"
            f"    {atten_str},\n"
            f"    {emission_str}"
        )
        if law_note:
            message += law_note

        raise ValueError(message)

    # Witt & Gordon (2000) screen (FSPS dust_type=3): capture and validate the
    # three structural selectors. They are static structural choices (not free
    # params); the fitted depth is ``tau_v`` (→ dust_tau_v), shared with the
    # single-component screen.
    if dust_type == "wg00":
        _wg00_axes = {
            "dust_curve": ("dust_wg00_curve", _WG00_DUST_CURVES),
            "geometry": ("dust_wg00_geometry", _WG00_GEOMETRIES),
            "structure": ("dust_wg00_structure", _WG00_STRUCTURES),
        }
        for key, (result_key, allowed) in _wg00_axes.items():
            if key in dust_atten_dict:
                val = dust_atten_dict[key]
                if val not in allowed:
                    raise ValueError(f"Invalid WG00 {key} {val!r}; choose one of {allowed}.")
                result[result_key] = val
        return

    # Extract and validate dust laws. Attenuation laws are now EXPLICIT and required.
    # For single_component: 'law' is required (singular, one screen).
    # For two_component: either 'law' (shared by both screens) XOR both 'law_bc' AND 'law_diff'.
    valid_laws = _valid_dust_laws()

    dust_law = dust_atten_dict.get("law")
    dust_law_bc = dust_atten_dict.get("law_bc")
    dust_law_diff = dust_atten_dict.get("law_diff")
    dust_law_neb = dust_atten_dict.get("law_neb")

    # Paired per-screen keys follow the same rule as the laws: on a two-screen
    # model, naming one screen and leaving the other implicit is an incomplete
    # specification. A lone `tau_bc` used to leave `tau_diff` pinned at its
    # declared 0.3 while the birth cloud was fitted -- and the diffuse screen
    # usually dominates the total attenuation, so that is rarely what anyone
    # means. A wildcard is still an accepted way to say "free the partner too".
    if dust_type == "two_component" and not ({"all_params", "*"} & set(dust_atten_dict)):
        for stem in ("tau", "Rv", "delta", "slope", "bump_strength"):
            bc, diff = f"{stem}_bc", f"{stem}_diff"
            has_bc = dust_atten_dict.get(bc) is not None
            has_diff = dust_atten_dict.get(diff) is not None
            if has_bc == has_diff:
                continue
            named, missing = (bc, diff) if has_bc else (diff, bc)
            raise ValueError(
                f"dust_attenuation type='two_component' names {named!r} but not "
                f"{missing!r}. A two-screen model needs both, or a wildcard to free "
                f"them together -- otherwise {missing!r} silently keeps its declared "
                f"default while {named!r} is fitted. Give {missing!r} explicitly, or "
                f"pass 'all_params': FREE. Accepted: "
                f"{{'{bc}': ..., '{diff}': ...}}, or "
                f"{{'{named}': ..., 'all_params': FREE}}, or neither."
            )

    # For single_component: require 'law', reject law_bc/law_diff
    if dust_type == "single_component":
        if dust_law_bc is not None or dust_law_diff is not None:
            raise ValueError(
                "dust_attenuation type='single_component' has a single attenuation screen. "
                "Use 'law' to set the attenuation law, not 'law_bc' or 'law_diff'. "
                "Example: dust_attenuation={'type': 'single_component', 'law': 'calzetti', ...}"
            )
        if dust_law is None:
            laws_list = ", ".join(sorted(valid_laws))
            raise ValueError(
                f"dust_attenuation type='single_component' requires 'law' to be specified. "
                f"Valid laws: {laws_list}. "
                f"Example: dust_attenuation={{'type': 'single_component', "
                f"'law': 'calzetti', 'tau_v': ...}}"
            )
        if dust_law not in valid_laws:
            suggestions = difflib.get_close_matches(dust_law, valid_laws, n=2, cutoff=0.6)
            suggest_str = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
            raise ValueError(f"Unknown dust law '{dust_law}'.{suggest_str}")
        # Store law on both _bc and _diff for consistency (single screen)
        result["dust_law_bc"] = dust_law
        result["dust_law_diff"] = dust_law

    # For two_component: either 'law' XOR (law_bc AND law_diff)
    elif dust_type == "two_component":
        has_law = dust_law is not None
        has_law_bc = dust_law_bc is not None
        has_law_diff = dust_law_diff is not None

        # Check for invalid combinations
        if has_law and (has_law_bc or has_law_diff):
            raise ValueError(
                "dust_attenuation type='two_component' grammar is ambiguous: "
                "cannot specify both 'law' and 'law_bc'/'law_diff'. "
                "Use EITHER 'law' (shared by both screens) "
                "OR both 'law_bc' and 'law_diff' (per-screen). "
                "Example 1: dust_attenuation={'type': 'two_component', 'law': 'calzetti', ...} "
                "Example 2: dust_attenuation={'type': 'two_component', 'law_bc': 'calzetti', "
                "'law_diff': 'power_law', ...}"
            )

        if not has_law and not (has_law_bc and has_law_diff):
            if has_law_bc or has_law_diff:
                given = [k for k in ("law_bc", "law_diff") if dust_atten_dict.get(k) is not None]
                raise ValueError(
                    f"dust_attenuation type='two_component' requires BOTH 'law_bc' and 'law_diff' "
                    f"for per-screen specification, or use 'law' for a shared law. "
                    f"You gave: {given}. "
                    f"Example 1: dust_attenuation={{'type': 'two_component', "
                    f"'law': 'calzetti', ...}} "
                    f"Example 2: dust_attenuation={{'type': 'two_component', "
                    f"'law_bc': 'calzetti', "
                    f"'law_diff': 'power_law', ...}}"
                )
            # Neither form given
            laws_list = ", ".join(sorted(valid_laws))
            raise ValueError(
                f"dust_attenuation type='two_component' requires either 'law' "
                f"(applied to both screens) "
                f"or both 'law_bc' and 'law_diff' (per-screen). "
                f"Valid laws: {laws_list}. "
                f"Example 1: dust_attenuation={{'type': 'two_component', 'law': 'calzetti', "
                f"'tau_bc': ..., 'tau_diff': ...}} "
                f"Example 2: dust_attenuation={{'type': 'two_component', 'law_bc': 'calzetti', "
                f"'law_diff': 'power_law', 'tau_bc': ..., 'tau_diff': ...}}"
            )

        # Resolve to the two-screen form
        if has_law:
            if dust_law not in valid_laws:
                suggestions = difflib.get_close_matches(dust_law, valid_laws, n=2, cutoff=0.6)
                suggest_str = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
                raise ValueError(f"Unknown dust law '{dust_law}'.{suggest_str}")
            result["dust_law_bc"] = dust_law
            result["dust_law_diff"] = dust_law
        else:
            # Both law_bc and law_diff are given (already checked above)
            if dust_law_bc not in valid_laws:
                suggestions = difflib.get_close_matches(dust_law_bc, valid_laws, n=2, cutoff=0.6)
                suggest_str = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
                raise ValueError(f"Unknown dust law '{dust_law_bc}'.{suggest_str}")
            if dust_law_diff not in valid_laws:
                suggestions = difflib.get_close_matches(dust_law_diff, valid_laws, n=2, cutoff=0.6)
                suggest_str = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
                raise ValueError(f"Unknown dust law '{dust_law_diff}'.{suggest_str}")
            result["dust_law_bc"] = dust_law_bc
            result["dust_law_diff"] = dust_law_diff

    # law_neb: optional per-screen override for nebular birth cloud. None -> inherit dust_law_bc
    if dust_law_neb is not None:
        if dust_law_neb not in valid_laws:
            suggestions = difflib.get_close_matches(dust_law_neb, valid_laws, n=2, cutoff=0.6)
            suggest_str = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
            raise ValueError(f"Unknown dust law '{dust_law_neb}'.{suggest_str}")
        result["dust_law_neb"] = dust_law_neb

    # Per-component law-parameter overrides: slope_bc / slope_diff / slope_neb /
    # bump_strength_bc / delta_diff / delta_neb / Rv_bc / … route onto a nested
    # dict {'bc': {law_kwarg: value}, 'diff': {...}, 'neb': {...}} consumed by
    # DustSEDComponent. The 'neb' channel reddens only the nebular birth cloud
    # (shares the diffuse ISM screen with the stars).
    from tengri.components.dust.attenuation import TWO_COMPONENT_OVERRIDE_KEYS

    overrides: dict[str, dict[str, float]] = {}
    for short, law_kw in TWO_COMPONENT_OVERRIDE_KEYS.items():
        for comp in ("bc", "diff", "neb"):
            key = f"{short}_{comp}"
            if key in dust_atten_dict:
                overrides.setdefault(comp, {})[law_kw] = float(dust_atten_dict[key])
    if overrides:
        result["dust_law_overrides"] = overrides

    # Lyman-limit clip (912 Å). Boolean in the grammar; stored as the cutoff
    # wavelength so the forward model and compile_signature carry a single float.
    if dust_atten_dict.get("lyman_cutoff"):
        result["dust_lyman_cutoff_aa"] = 912.0

    # Whether ALL stellar LyC is absorbed by neb_fesc (FSPS/CIGALE) or only the
    # young/birth-cloud population (default; bagpipes). See DustSEDComponent.
    if "lyc_absorb_all" in dust_atten_dict:
        result["dust_lyc_absorb_all"] = bool(dust_atten_dict["lyc_absorb_all"])

    # Include the LyC in the dust energy-balance integral (FSPS/Prospector
    # parity) vs the canonical LyC-masked L_absorbed (default; #922/#961).
    if "eb_include_lyc" in dust_atten_dict:
        result["dust_eb_include_lyc"] = bool(dust_atten_dict["eb_include_lyc"])


def _translate_dust_retired(dust_dict: dict, result: dict) -> None:
    """Handle retired dust= syntax with loud error and translation.

    The dust group has been split into dust_attenuation and dust_emission
    as separate top-level groups. This function rejects the old form and
    provides a helpful error message showing the translation.

    Before PR #1989, a missing attenuation law defaulted to 'power_law'.
    When translating old dust= blocks, we add that default to the
    dust_attenuation suggestion so it parses and reproduces the old behavior.
    """
    # Passing both the old and the new form is caught in _translate_structural,
    # which sees every group; this function only ever runs for `dust=` and its
    # job is to translate the caller's own dict into the two replacements.
    has_emission = "emission" in dust_dict

    # Extract the attenuation half (everything except 'emission')
    atten_dict = {k: v for k, v in dust_dict.items() if k != "emission"}

    # Build the translated attenuation dict by merging legacy law forms into modern form.
    # Before PR #1989, dust allowed partial law specs; we must translate them cleanly.
    dust_type = atten_dict.get("type", "two_component")
    atten_dict_translated = {
        k: v for k, v in atten_dict.items() if k not in ("law", "law_bc", "law_diff")
    }

    law_note = ""  # Explanation if we modified the law spec

    if dust_type == "two_component":
        has_law = "law" in atten_dict
        has_law_bc = "law_bc" in atten_dict
        has_law_diff = "law_diff" in atten_dict

        if has_law:
            # Already modern form; keep it
            atten_dict_translated["law"] = atten_dict["law"]
        elif has_law_bc and has_law_diff:
            # Already modern form; keep both
            atten_dict_translated["law_bc"] = atten_dict["law_bc"]
            atten_dict_translated["law_diff"] = atten_dict["law_diff"]
        elif has_law_bc:
            # Lone law_bc: pre-#1989 it applied to both screens; merge to 'law'
            atten_dict_translated["law"] = atten_dict["law_bc"]
            law_note = (
                "\n\nNote: The old form had 'law_bc' alone, which pre-#1989 applied to "
                "both screens. We merged it to 'law' to preserve that behavior."
            )
        elif has_law_diff:
            # Lone law_diff: the other screen used power_law by default
            atten_dict_translated["law_diff"] = atten_dict["law_diff"]
            atten_dict_translated["law_bc"] = "power_law"
            law_note = (
                "\n\nNote: The old form had only 'law_diff'; the 'law_bc' screen defaulted "
                "to power_law. Verify whether power_law was intended for the birth cloud."
            )
        else:
            # No law info: add default
            atten_dict_translated["law"] = "power_law"
            law_note = (
                "\n\nNote: The suggestion includes 'law': 'power_law' — before PR #1989, "
                "a missing law defaulted to power_law. This reproduces the old result "
                "exactly, but you should verify that power_law was your intended choice "
                "and not just an accidentally-omitted setting."
            )

    elif dust_type == "single_component":
        has_law = "law" in atten_dict
        has_law_bc = "law_bc" in atten_dict

        if has_law:
            # Already modern form
            atten_dict_translated["law"] = atten_dict["law"]
        elif has_law_bc:
            # Pre-#1989 allowed law_bc on single_component; use it as 'law'
            atten_dict_translated["law"] = atten_dict["law_bc"]
            law_note = (
                "\n\nNote: The old form allowed 'law_bc' on single_component; "
                "we merged it to 'law'."
            )
        else:
            # No law: add default
            atten_dict_translated["law"] = "power_law"
            law_note = (
                "\n\nNote: The suggestion includes 'law': 'power_law' — before PR #1989, "
                "a missing law defaulted to power_law. This reproduces the old result "
                "exactly, but you should verify that power_law was your intended choice "
                "and not just an accidentally-omitted setting."
            )

    atten_str = (
        "dust_attenuation={"
        + ", ".join(
            f"'{k}': {v!r}"
            for k, v in sorted(_wildcard_keys_for_display(atten_dict_translated).items())
        )
        + "}"
    )

    emission_str = ""
    if has_emission:
        emission_dict = dust_dict["emission"]
        if isinstance(emission_dict, dict):
            emis_str_parts = []
            for k, v in sorted(_wildcard_keys_for_display(emission_dict).items()):
                if isinstance(v, str):
                    emis_str_parts.append(f"'{k}': {v!r}")
                else:
                    emis_str_parts.append(f"'{k}': {v!r}")
            emission_str = "dust_emission={" + ", ".join(emis_str_parts) + "}"

    dust_str_parts = []
    for k, v in sorted(_wildcard_keys_for_display(dust_dict).items()):
        if isinstance(v, str):
            dust_str_parts.append(f"'{k}': {v!r}")
        else:
            dust_str_parts.append(f"'{k}': {v!r}")
    dust_str = "dust={" + ", ".join(dust_str_parts) + "}"

    message = (
        "`dust=` is retired; attenuation and IR emission are now separate groups. "
        "Replace\n"
        f"    {dust_str}\n"
        "with\n"
        f"    {atten_str}"
    )
    if emission_str:
        message += f",\n    {emission_str}"
    else:
        message += "."

    if law_note:
        message += law_note

    raise ValueError(message)


def _translate_dust_emission(dust_emis_dict: dict, result: dict) -> None:
    """Translate dust_emission group to dust_emission type and settings.

    Handles IR re-emission model selection and associated structural
    configuration (e.g., astrodust spinning dust and f_cnm).
    """
    emission_type = dust_emis_dict.get("type")
    if emission_type in ("none", "off"):
        # Explicitly disable IR re-emission — parity with the group-level
        # 'none'. Leave result['dust_emission'] unset (its off default).
        emission_type = None
    if emission_type is not None:
        # Dust IR emission types are engine names (modified_blackbody, dale2014,
        # dl07, dl14, astrodust, etc.) resolved by the DUST_EMISSION_MODELS loader cache.
        valid_emission_types = _valid_dust_emission_types()
        if emission_type not in valid_emission_types:
            suggestions = difflib.get_close_matches(
                emission_type, valid_emission_types, n=3, cutoff=0.6
            )
            suggest_str = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
            raise ValueError(f"Unknown dust_emission type '{emission_type}'.{suggest_str}")
        result["dust_emission"] = emission_type

        # Astrodust+PAH (HD23) optional configuration: spinning dust (AME)
        # and cold-neutral-medium filling fraction. These are structural
        # configuration (not free parameters), stored with astrodust_ prefix
        # to distinguish from parameter registry.
        if emission_type == "astrodust":
            if "spinning_dust" in dust_emis_dict:
                result["astrodust_spinning_dust"] = bool(dust_emis_dict["spinning_dust"])
            if "f_cnm" in dust_emis_dict:
                result["astrodust_f_cnm"] = float(dust_emis_dict["f_cnm"])


# Backend *implementations* are named after their physics (BakedInBackend,
# CloudyGridBackend) and the internal NebularConfig.backend enum spells two of
# them "baked_in" / "cloudy". The builder grammar instead names each backend for
# where its emission comes from ("ssp", "cloudy"). Users reasonably guess the
# class/config spelling, and difflib cannot bridge "baked_in" -> "ssp" (no shared
# substring), so map the guessable spellings explicitly to keep the error useful.
_NEBULAR_TYPE_HINTS = {
    "baked_in": "ssp",
    "bakedin": "ssp",
    "cloudy_grid": "cloudy",
    "cloudygrid": "cloudy",
}


def _translate_neb(neb_dict: dict, result: dict) -> None:
    """Translate neb group to nebular settings."""
    neb_type = neb_dict.get("type", "none")

    # Validate type
    valid_neb = _valid_nebular_types()
    if neb_type not in valid_neb:
        hint = _NEBULAR_TYPE_HINTS.get(str(neb_type).lower())
        suggestions = (
            [hint]
            if hint in valid_neb
            else difflib.get_close_matches(neb_type, valid_neb, n=2, cutoff=0.6)
        )
        suggest_str = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        raise ValueError(
            f"Unknown nebular type '{neb_type}'.{suggest_str} "
            f"Available: {', '.join(sorted(valid_neb))}."
        )

    # Map type to nebular settings
    if neb_type == "none":
        result["nebular"] = False
        result["nebular_ssp"] = False
        result["nebular_cue"] = False
    elif neb_type == "ssp":
        result["nebular_ssp"] = True
    elif neb_type == "cue":
        result["nebular_cue"] = True
        # #303: opt into the full Cue catalog (~271 species) instead
        # of the default 128 CLOUDY/FSPS subset so users can read
        # HeII 1640, HeI 10830, etc. via pred.lines.get(wavelength).
        if neb_dict.get("full_catalog", False):
            result["cue_full_catalog"] = True
    elif neb_type == "cloudy":
        result["nebular"] = True
        # Optional explicit grid; without it Parameters auto-resolves
        # data/cloudy_grid_mist.h5 (matching the default MIST/FSPS SSP
        # family) and raises with the available-grid listing otherwise.
        if "grid" in neb_dict:
            result["cloudy_grid_path"] = str(neb_dict["grid"])
    elif neb_type == "cb19":
        result["nebular"] = "cb19"


def _translate_shock(shock_dict: dict, result: dict) -> None:
    """Translate the top-level ``shock`` group to shock settings (#851).

    Activates the composable MAPPINGS V shock component, which is *additive*:
    it composes with whatever photoionized backend the ``neb`` group selects
    (a model may run both). ``type='none'`` disables it.

    Recognized structural keys:

    - ``norm``: ``"frac"`` (relative to the galaxy Halpha, default) or
      ``"lhalpha"`` (absolute ``shock_log_lhalpha``).
    - ``abundance``: MAPPINGS abundance set (``"solar"``, ``"2xsolar"``, …).
    - ``component``: ``"shock"`` | ``"precursor"`` | ``"combined"``.

    Per-parameter overrides (``frac``, ``log_lhalpha``, ``velocity``,
    ``log_density``, ``b_over_sqrt_n``) resolve to the ``shock_*`` bucket
    params in :func:`parse_groups`.
    """
    shock_type = shock_dict.get("type", "mappings")
    valid_shock = _VALID_SHOCK_TYPES
    if shock_type not in valid_shock:
        suggestions = difflib.get_close_matches(shock_type, valid_shock, n=2, cutoff=0.6)
        suggest_str = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        raise ValueError(f"Unknown shock type '{shock_type}'.{suggest_str}")

    if shock_type == "none":
        result["shock"] = False
        return

    result["shock"] = True

    norm = shock_dict.get("norm", "frac")
    if norm not in ("frac", "lhalpha"):
        raise ValueError(
            f"shock norm must be 'frac' or 'lhalpha', got {norm!r}. "
            "'frac' scales the galaxy Halpha; 'lhalpha' sets an absolute "
            "log10(L_Halpha/[erg/s]) via shock_log_lhalpha."
        )
    result["shock_norm"] = norm
    result["shock_abundance"] = shock_dict.get("abundance", "solar")
    result["shock_component"] = shock_dict.get("component", "combined")


def _translate_igm(igm_dict: dict, result: dict) -> None:
    """Translate igm group to igm_model and related settings.

    IGM activation is derived from the igm group presence: if igm={'type': ...}
    is provided, IGM is activated. If igm={'type': 'none'} is provided, IGM is
    deactivated. The apply_igm secondary switch is retired.

    The typeless default is ``inoue14``, changed here from ``madau``. Those are
    different transmission curves, not two names for one -- but the two entry
    points disagreed about which the grammar meant. ``Parameters.__init__``
    defaults ``igm_model="inoue"`` (an alias of ``inoue14``), so the same model
    written flat and written as a group got different IGM physics, and
    ``components/igm/igm.py`` documents the intent as "the dict-grammar API
    consistently used ``inoue14``" -- which this function contradicted. The
    comment described the design; the code had drifted from it.

    Measured: no call site in the tree omits ``type``, so this moves nothing
    today. That is also why it survived -- a latent default is invisible until
    someone relies on it.
    """
    igm_type = igm_dict.get("type", "inoue14")

    # Validate type
    valid_igm = _valid_igm_types()
    if igm_type not in valid_igm:
        suggestions = difflib.get_close_matches(igm_type, valid_igm, n=2, cutoff=0.6)
        suggest_str = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        raise ValueError(f"Unknown IGM type '{igm_type}'.{suggest_str}")

    # IGM is activated by the presence of the igm dict.
    # If type='none', it is explicitly deactivated.
    if igm_type == "none":
        result["apply_igm"] = False
    else:
        # madau, inoue14, meiksin06 -> apply_igm=True
        result["apply_igm"] = True
        # Propagate the model choice. _init_igm speaks 'inoue'/'madau'/
        # 'meiksin06'; 'inoue14' is the grammar-level name — normalize to
        # the canonical form here so the user's selection isn't silently
        # dropped (#344, #440).
        result["igm_model"] = _IGM_TYPE_ALIASES[igm_type]

    # Handle optional IGM subkeys
    if igm_dict.get("patchy", False):
        result["igm_patchy"] = True

    # ``dla`` accepts either the legacy boolean form (``dla=True``) or the
    # nested-dict form ``dla={'log_n_hi': Uniform(...), '*': FREE, ...}`` —
    # both activate the DLA absorber (closes #507). The per-parameter
    # overrides inside the dict are resolved by the ``igm.dla`` sub-group
    # path in :func:`parse_groups`.
    dla_spec = igm_dict.get("dla", False)
    if dla_spec:
        result["dla"] = True


def _legacy_radio_type_to_blocks(radio_type: str) -> tuple[str, str]:
    """Resolve a legacy ``radio={'type': X}`` name onto ``(sf_mode, agn_model)``.

    Parameters
    ----------
    radio_type : str
        A member of :func:`_valid_radio_types` other than ``"none"``.

    Returns
    -------
    tuple of (str, str)
        ``(radio_sfr_mode, radio_agn_model)`` — the two attributes that
        decide which radio physics runs.

    Notes
    -----
    ``condon92`` predates the SF/AGN split and names the *composite*
    (``radio_total``), not a third AGN model, so it resolves to both
    defaults. The SEDModelComponent variants that
    :func:`_valid_radio_types` folds in — ``radio_dpl``,
    ``radio_powerlaw`` — each name exactly one axis, so strip the
    ``radio_`` prefix and route the remainder to whichever axis
    registers it.

    Deriving the mapping from ``AGN_RADIO_MODELS`` / ``SF_RADIO_MODELS``
    rather than spelling it out keeps this in step with the registry the
    validator already reads: a hand-written third list is how the radio
    error message and the dust menu each drifted out of agreement with
    the builder. A ``radio_*`` component whose stripped name matches
    neither axis raises here rather than silently taking the defaults —
    that silent path is #1461, where ``radio_dpl`` was accepted and the
    single power-law ran in place of the Martinez-Ramirez+2024 double
    power-law.
    """
    from tengri.components.radio.component import AGN_RADIO_MODELS, SF_RADIO_MODELS

    sf_default, agn_default = "bell2003", "powerlaw"
    if radio_type == "condon92":
        return sf_default, agn_default

    # ``"none"`` is the only name in both tuples and never reaches here,
    # so the AGN-first order below cannot be ambiguous.
    block = radio_type.removeprefix("radio_")
    if block in AGN_RADIO_MODELS:
        return sf_default, block
    if block in SF_RADIO_MODELS:
        return block, agn_default

    raise ValueError(
        f"radio type '{radio_type}' is accepted by the grammar but names "
        f"neither a star-forming model {SF_RADIO_MODELS} nor an AGN model "
        f"{AGN_RADIO_MODELS}. It was registered as a radio component "
        "without a matching sf/agn block, so there is nothing for the "
        "forward model to run. Select it explicitly instead, e.g. "
        "radio={'agn': {'type': 'dpl'}}."
    )


def _translate_radio(radio_dict: dict, result: dict) -> None:
    """Translate radio group with composable SF + AGN sub-blocks.

    Supports two grammar styles:

    **New composable form (preferred)**:

    .. code-block:: python

        radio = {
            "sf": {"type": "delvecchio2021"},  # SF variant
            "agn": {"type": "dpl"},  # AGN variant
        }

    **Legacy form (RETIRED, #1980)** — ``radio={'type': 'condon92'}`` now
    raises with the composable equivalent in the message.
    :func:`_legacy_radio_type_to_blocks` survives only to compute that
    equivalent for the error text (naming the mapping rather than a
    generic default pair — accepting a name and then ignoring it is #1461).

    Raises if both 'type' and 'sf'/'agn' sub-blocks are present.
    """
    has_legacy_type = "type" in radio_dict
    has_sf_block = "sf" in radio_dict
    has_agn_block = "agn" in radio_dict

    if has_legacy_type and (has_sf_block or has_agn_block):
        # #1980: the legacy 'type' key is retired, so the recovery advice must
        # not offer it as one of two valid spellings — drop it and keep the
        # composable form only.
        raise ValueError(
            "radio: the legacy 'type' key is retired and cannot be mixed with "
            "'sf'/'agn' sub-blocks. Remove 'type' and use the composable form: "
            "radio={'sf': {'type': 'bell2003'}, 'agn': {'type': 'powerlaw'}}."
        )

    # Legacy form retired: radio={'type': 'X'} is no longer accepted.
    # Users must use the composable surface: radio={'sf': {...}, 'agn': {...}}
    if has_legacy_type:
        radio_type = radio_dict["type"]
        # Provide helpful guidance: show what the legacy type maps to
        sf_variant, agn_variant = "bell2003", "powerlaw"  # defaults
        if radio_type == "condon92":
            # condon92 used both defaults
            raise ValueError(
                f"radio legacy type form is retired. "
                f"radio={{'type': '{radio_type}'}} → use the composable form: "
                f"radio={{'sf': {{'type': '{sf_variant}'}}, 'agn': {{'type': '{agn_variant}'}}}}"
            )
        elif radio_type != "none":
            # Try to find what this maps to for a better error message
            try:
                sf_variant, agn_variant = _legacy_radio_type_to_blocks(radio_type)
            except ValueError:
                sf_variant, agn_variant = "bell2003", "powerlaw"
            raise ValueError(
                f"radio legacy type form is retired. "
                f"radio={{'type': '{radio_type}'}} → use the composable form: "
                f"radio={{'sf': {{'type': '{sf_variant}'}}, 'agn': {{'type': '{agn_variant}'}}}}"
            )
        else:  # radio_type == 'none'
            raise ValueError(
                "radio legacy type form is retired. "
                "radio={'type': 'none'} → use the composable form with both 'none': "
                "radio={'sf': {'type': 'none'}, 'agn': {'type': 'none'}}"
            )

    # New composable form: extract SF and AGN sub-blocks
    sf_variant = "bell2003"  # default
    agn_variant = "powerlaw"  # default
    radio_enabled = False

    if has_sf_block:
        sf_dict = radio_dict["sf"]
        if isinstance(sf_dict, dict):
            sf_variant = sf_dict.get("type", "bell2003")
            from tengri.components.radio.component import SF_RADIO_MODELS

            valid_sf = frozenset(SF_RADIO_MODELS)
            if sf_variant not in valid_sf:
                suggestions = difflib.get_close_matches(sf_variant, valid_sf, n=2, cutoff=0.6)
                suggest_str = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
                raise ValueError(f"Unknown radio sf type '{sf_variant}'.{suggest_str}")
        else:
            raise TypeError(f"radio['sf'] must be a dict, got {type(sf_dict).__name__}.")

    if has_agn_block:
        agn_dict = radio_dict["agn"]
        if isinstance(agn_dict, dict):
            agn_variant = agn_dict.get("type", "powerlaw")
            from tengri.components.radio.component import AGN_RADIO_MODELS

            valid_agn = frozenset(AGN_RADIO_MODELS)
            if agn_variant not in valid_agn:
                suggestions = difflib.get_close_matches(agn_variant, valid_agn, n=2, cutoff=0.6)
                suggest_str = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
                raise ValueError(f"Unknown radio agn type '{agn_variant}'.{suggest_str}")
        else:
            raise TypeError(f"radio['agn'] must be a dict, got {type(agn_dict).__name__}.")

    # Determine if radio is overall enabled: True if either SF or AGN is not 'none'
    radio_enabled = sf_variant != "none" or agn_variant != "none"

    result["radio"] = radio_enabled
    if radio_enabled:
        result["radio_sfr_mode"] = sf_variant
        result["radio_agn_model"] = agn_variant


#: Model-specific radio sub-block params, routed to the ``radio.sf`` /
#: ``radio.agn`` sub-groups so they resolve when nested in
#: ``radio={'sf': {...}, 'agn': {...}}`` (mirrors the ``dust.emission``
#: sub-group). Keyed by the active mode: a sub-block ``'*'`` wildcard frees
#: only the active model's params (block-scoped, like the AGN grammar), so
#: selecting ``delvecchio2021`` never frees the McCheyne coefficients and a
#: ``powerlaw`` AGN radio never frees the DPL turnover knobs.
_RADIO_SF_PARAMS_BY_MODE: dict[str, frozenset[str]] = {
    "delvecchio2021": frozenset({"radio_delv_q0", "radio_delv_mass_slope", "radio_delv_z_slope"}),
    "mccheyne2022": frozenset({"radio_mcch_q0", "radio_mcch_mass_slope", "radio_mcch_z_slope"}),
}
_RADIO_AGN_PARAMS_BY_MODEL: dict[str, frozenset[str]] = {
    "powerlaw": frozenset({"radio_loudness", "radio_alpha_agn"}),
    "dpl": frozenset(
        {
            "radio_loudness",
            "radio_alpha_thin",
            "radio_alpha_thick",
            "radio_log_nu_t",
            "radio_log_nu_cut",
        }
    ),
}
#: X-ray corona params that only *some* models read. ``XRaySEDComponent`` picks
#: one of two argument lists on ``config.model``: the ``lopez24`` corona passes
#: ``alpha_irx`` (the 12um -> L_X ratio) and no ``delta_alpha_ox``; every other
#: model passes ``delta_alpha_ox`` (the Just+2007 alpha_ox offset) and never
#: passes ``alpha_irx``.
#:
#: Params every model reads -- ``gamma_agn``, ``E_cut``, ``log_nh``, the two XRB
#: photon indices -- are deliberately absent: only variant-specific names belong
#: here, so a shared param added later cannot be pinned by omission.
#:
#: The ``xray_alpha_irx`` declaration has stated this since it was written
#: ("read only by the lopez24 corona -- yang20 ignores it -- so freeing it under
#: a wildcard is only meaningful with that corona selected"). Nothing consulted
#: it: measured on a UV-to-X-ray fixture with a luminous AGN, ``alpha_irx``
#: moves the SED by 62% under ``lopez24`` and by *exactly* zero under every
#: other model, while the wildcard freed it for all of them.
#: **The "shared params are deliberately absent" rule above ended with #1684.**
#: It held while every X-ray type resolved to one component, so ``gamma_agn`` /
#: ``E_cut`` / ``log_nh`` / the XRB offsets were read by all of them and could
#: not be pinned by omission. ``xray_aird`` now builds ``XRayAirdSEDComponent``,
#: which reads the XRB offsets and none of the corona parameters — measured:
#: under ``xray_aird`` the wildcard freed ``xray_E_cut``, ``xray_gamma_agn`` and
#: ``xray_log_nh`` while none of the three could move the SED, beside
#: ``xray_det_hmxb`` / ``xray_det_lmxb`` which could.
#:
#: So every entry now lists what its model reads in full, rather than only the
#: names unique to it. Omission no longer means "shared", it means "not read" —
#: which is what the narrowing needs to be able to say.
_XRAY_PARAMS_BY_MODEL: dict[str, frozenset[str]] = {
    "lopez24": frozenset(
        {
            "xray_alpha_irx",
            "xray_E_cut",
            "xray_gamma_agn",
            "xray_log_nh",
            "xray_det_hmxb",
            "xray_det_lmxb",
        }
    ),
    # Aird+2015 XRB scaling: the Lehmer+2016 offsets, and no corona.
    "xray_aird": frozenset({"xray_det_hmxb", "xray_det_lmxb"}),
    # The alpha_ox corona and nothing else: no XRB channel, so the Lehmer+2016
    # offsets are genuinely unread here, and no N_H screen.
    "agn_xray_corona": frozenset({"xray_gamma_agn", "xray_E_cut", "xray_delta_alpha_ox"}),
}
#: What the shared ``XRaySEDComponent`` corona reads on its non-``lopez24``
#: branch (``yang20`` / ``simple`` / ``agn_xray_corona``, which all resolve to
#: it). ``xray_alpha_irx`` is absent because only the lopez24 branch reads it.
_XRAY_DEFAULT_MODEL_PARAMS: frozenset[str] = frozenset(
    {
        "xray_delta_alpha_ox",
        "xray_E_cut",
        "xray_gamma_agn",
        "xray_log_nh",
        "xray_det_hmxb",
        "xray_det_lmxb",
    }
)
#: Union of the above: the names the scope may narrow away.
_XRAY_VARIANT_PARAMS: frozenset[str] = _XRAY_DEFAULT_MODEL_PARAMS | frozenset().union(
    *_XRAY_PARAMS_BY_MODEL.values()
)
#: Declared and freeable, but read by no model the grammar can currently build.
#:
#: **Empty since #1706.** It held ``xray_det_hmxb`` / ``xray_det_lmxb``, the
#: Lehmer+2016 XRB luminosity offsets, which were inert under every X-ray type
#: because ``XRaySEDComponent._terms()`` never passed them on to
#: ``xray_total_terms`` / ``xray_total_lopez24_terms``. Both call sites now do,
#: so every model reads them and there is no variant to scope them to.
#:
#: One prediction in the old note did not survive measurement, and is recorded
#: here because it is the kind that costs a large refactor: it held that fixing
#: the offsets also required splitting the precompute XRB grid, since
#: ``_build_grid_xrb`` bakes HMXB and LMXB into one summed template. It does
#: not. The band-response precompute derives its amplitudes by calling
#: ``emission_terms`` at reference wavelengths on every predict, and the offsets
#: are pure scalar amplitudes, so both accelerated paths inherit the call-site
#: fix untouched -- verified under ``WavePrecomp()`` and ``precompute=True`` on
#: both ``yang20`` and ``lopez24`` in
#: ``tests/regression/bug/test_xray_xrb_offsets_wired.py``.
#:
#: Kept (empty) rather than deleted: the narrowing step in ``parse_groups`` is
#: the right home for a genuinely unreadable parameter, and the next one should
#: land here rather than re-deriving the mechanism.
_XRAY_UNREACHABLE_PARAMS: frozenset[str] = frozenset()

#: Union of every param owned by each radio sub-group, used by the partition
#: to route names away from the flat ``radio`` group.
_RADIO_SF_PARAM_NAMES: frozenset[str] = frozenset().union(*_RADIO_SF_PARAMS_BY_MODE.values())
_RADIO_AGN_PARAM_NAMES: frozenset[str] = frozenset().union(*_RADIO_AGN_PARAMS_BY_MODEL.values())


#: Valid laws for the MW foreground screen (#297). Only the closed-form
#: laws that take a single ``R_V`` parameter are usable as a foreground
#: screen — host-dust laws with two free knobs (slope, bump, ...) would
#: collide with the host ``dust`` block's parameter prefix.
_VALID_FOREGROUND_LAWS = frozenset({"cardelli"})

#: Laws the AGN attenuation-stage block actually implements. One entry, and that
#: is the honest count: ``components/agn/reddening.py`` imports ``prevot_smc``
#: and applies it unconditionally, so the block is single-curve by construction
#: -- its own name (``smc_prevot``) and its E(B-V) parameter's description say so.
#:
#: This was validated against ``DUST_LAWS`` (22 entries) with every accepted
#: name mapped to the same block, so ``agn={'atten': {'law': 'calzetti'}}`` was
#: validated, accepted, and silently given the Prevot SMC curve. Measured: five
#: distinct law names produced bit-identical SEDs at E(B-V)=0.4, while the same
#: comparison saw E(B-V) itself change the SED. Validating against a menu the
#: physics does not honor is worse than not offering the choice -- the careful
#: "did you mean" error taught users the choice was real (#2012).
#:
#: ``smc`` is deliberately NOT here: it is a different curve from ``prevot_smc``
#: (20% apart over 1000-20000 A), so accepting it would be the same silent
#: substitution at smaller magnitude. Wiring more laws into the block widens
#: this set; until then it describes what the block does.
_VALID_AGN_ATTEN_LAWS = frozenset({"prevot_smc"})


def _translate_foreground(fg_dict: dict, result: dict) -> None:
    """Translate the ``foreground`` group (MW screen) — see #297.

    Flat layout: ``foreground={'ebmv_mw': 0.05, 'law': 'cardelli', 'rv': 3.1}``.
    Surfaces three top-level kwargs on ``Parameters`` so the SEDModel can
    apply the screen in the observed-frame SED path, after IGM and
    redshifting, independently from the host-galaxy ``dust`` block.

    **Intentional design**: ``foreground`` is deliberately settings-only
    (no ``type`` key, no free parameters). Milky Way dust extinction is not a
    fittable model choice — it is observational / astronomical data — so it
    remains a bare configuration dictionary. Unlike other groups (dust, AGN,
    radio, etc.), foreground carries no sub-blocks and no structural type.
    """
    ebmv = fg_dict.get("ebmv_mw", 0.0)
    law = fg_dict.get("law", "cardelli")
    rv = fg_dict.get("rv", 3.1)
    if law not in _VALID_FOREGROUND_LAWS:
        suggestions = difflib.get_close_matches(law, _VALID_FOREGROUND_LAWS, n=2, cutoff=0.6)
        suggest_str = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        raise ValueError(
            f"Unknown foreground law {law!r}. Valid: "
            f"{sorted(_VALID_FOREGROUND_LAWS)}.{suggest_str}"
        )
    if float(ebmv) < 0:
        raise ValueError(f"foreground.ebmv_mw must be >= 0, got {ebmv}")
    if float(rv) <= 0:
        raise ValueError(f"foreground.rv must be > 0, got {rv}")
    result["foreground_ebmv_mw"] = float(ebmv)
    result["foreground_law"] = law
    result["foreground_rv"] = float(rv)


def _translate_xray(xray_dict: dict, result: dict) -> None:
    """Translate xray group to xray=True/False."""
    xray_type = xray_dict.get("type", "none")

    # Validate type
    valid_xray = _valid_xray_types()
    if xray_type not in valid_xray:
        suggestions = difflib.get_close_matches(xray_type, valid_xray, n=2, cutoff=0.6)
        suggest_str = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        raise ValueError(f"Unknown X-ray type '{xray_type}'.{suggest_str}")

    result["xray"] = xray_type != "none"
    # Thread the corona prescription (yang20 / simple / lopez24) to the
    # component so it can dispatch; "simple" is the yang20 alias.
    if xray_type != "none":
        result["xray_model"] = xray_type


#: AGN sub-block keys recognized by the nested-dict grammar. Used when
#: walking a user's top-level ``agn`` dict to tell sub-block dicts apart
#: from per-parameter overrides. The new split includes independent ``nlr``
#: and ``blr`` categories; ``lines`` is deprecated (expanded to an (nlr, blr)
#: pair via expand_lines_alias).
_AGN_SUBBLOCK_KEYS = frozenset({"disc", "torus", "nlr", "blr", "feii", "atten", "lines"})

#: Per-group structural keys the grammar accepts on top of declared params.
#: Keys nested in a sub-block (e.g. ``dust.emission``) appear separately.
# Do NOT remove entries from _GROUP_STRUCTURAL_KEYS. The '*' key (WILDCARD_KEY)
# is required for post-normalization acceptance — validation runs after
# _normalize_wildcard_keys converts user-facing 'all_params' to '*', and the
# acceptance set must contain '*' for the converted dict to pass validation.
# The 'all_params' key is required as user-facing vocabulary for typo
# suggestions and displayed key lists in error messages. Removing either breaks
# a different consumer.
_GROUP_STRUCTURAL_KEYS: dict[str, frozenset[str]] = {
    "sfh": frozenset(
        {"type", "*", "all_params", "bin_edges_gyr", "age_kernel", "field_centering"}
    ),
    # ``met`` is the parallel of ``sfh``: both describe the stellar population's
    # history, and both select a model with ``type`` like every other group
    # (#1720). It replaces ``met={'type': ...}`` (#311) outright — two
    # spellings of one setting is the maintenance cost this removes.
    "met": frozenset({"type", "*", "all_params"}),
    "dust_attenuation": frozenset(
        {
            "type",
            "*",
            "all_params",
            "law",
            "law_bc",
            "law_diff",
            "law_neb",
            # WG00 screen structural selectors (FSPS dust_type=3).
            "dust_curve",
            "geometry",
            "structure",
            # Per-component law-parameter overrides (TWO_COMPONENT_OVERRIDE_KEYS
            # × {bc, diff, neb}); routed to dust_law_overrides, not declared
            # params. The 'neb' channel reddens only the nebular birth cloud.
            "slope_bc",
            "slope_diff",
            "slope_neb",
            "bump_strength_bc",
            "bump_strength_diff",
            "bump_strength_neb",
            "delta_bc",
            "delta_diff",
            "delta_neb",
            "Rv_bc",
            "Rv_diff",
            "Rv_neb",
            # Lyman-limit clip: zero the attenuation curve below 912 Å (CIGALE
            # parity). Two-component only; routed to dust_lyman_cutoff_aa.
            "lyman_cutoff",
            # Absorb ALL stellar LyC by neb_fesc (FSPS/CIGALE) vs young-only
            # (default; bagpipes). Two-component only.
            "lyc_absorb_all",
            # Include LyC in the dust energy-balance integral (FSPS/Prospector
            # parity) vs the canonical LyC-masked L_absorbed (#922/#961).
            "eb_include_lyc",
        }
    ),
    "dust_emission": frozenset(
        {"type", "*", "all_params", "spinning_dust", "f_cnm", "eta_balance"}
    ),
    "neb": frozenset({"type", "*", "all_params", "full_catalog", "grid"}),
    "shock": frozenset({"type", "*", "all_params", "norm", "abundance", "component"}),
    "igm": frozenset({"type", "*", "all_params", "patchy", "dla"}),
    "igm.dla": frozenset({"type", "*", "all_params"}),
    "radio": frozenset({"type", "*", "all_params", "sf", "agn"}),
    "radio.sf": frozenset({"type", "*", "all_params"}),
    "radio.agn": frozenset({"type", "*", "all_params"}),
    "xray": frozenset({"type", "*", "all_params"}),
    "agn": frozenset({"type", "*", "all_params", "norm"}) | _AGN_SUBBLOCK_KEYS,
    "agn.disc": frozenset({"type", "*", "all_params"}),
    "agn.torus": frozenset({"type", "*", "all_params"}),
    "agn.nlr": frozenset({"type", "*", "all_params"}),
    "agn.blr": frozenset({"type", "*", "all_params"}),
    "agn.feii": frozenset({"type", "*", "all_params"}),
    "agn.atten": frozenset({"type", "*", "all_params", "law"}),
    # Deprecated: agn.lines is expanded to (agn.nlr, agn.blr) via expand_lines_alias
    "agn.lines": frozenset({"type", "*", "all_params"}),
    "foreground": frozenset({"ebmv_mw", "law", "rv"}),
}


class _Structural(NamedTuple):
    """One structural (non-parameter) group setting and how it round-trips.

    ``_GROUP_STRUCTURAL_KEYS`` says which keys :func:`parse_groups` *accepts*;
    this says how :func:`parameters_to_groups` gives each one *back*. Keeping
    both declarative is the point — when the emit side was hand-written
    per-group it silently covered only three of the eight groups, so
    ``sfh['age_kernel']``, ``agn['norm']`` and the WG00 dust selectors were
    accepted, stored, and then dropped on the next ``to_groups()`` (#964).

    Attributes
    ----------
    key : str
        Name the setting carries inside its group dict (grammar side).
    attr : str
        Attribute :class:`~tengri.parameters.parameters.Parameters` stores it
        on (spec side).
    default : object
        Value the spec holds when the user did not set the key. The key is
        emitted only when the spec differs from this, so an untouched group
        stays absent from the round-trip rather than growing noise.
    only_types : tuple of str or None
        Emit only when the group's resolved ``type`` is one of these. Guards
        settings whose mere presence *implies* a backend — a ``neb['grid']``
        emitted onto a non-CLOUDY spec would switch the backend on re-parse.
    resolved_default : callable or None
        ``callable(spec)`` returning the value the spec would hold had the
        user not set the key, for defaults resolved at construction rather
        than fixed literals (CLOUDY's auto-located grid path). Overrides
        ``default`` when given.
    """

    key: str
    attr: str
    default: object
    only_types: tuple[str, ...] | None = None
    resolved_default: Callable[[Parameters], object] | None = None


#: How every plain structural setting round-trips: group -> settings.
#:
#: "Plain" means the grammar value and the stored value are the same object,
#: so a default comparison is enough to decide whether to emit. The dust
#: attenuation laws and their per-component overrides are deliberately absent:
#: they carry inheritance rules and a flattened override dict, and stay
#: hand-written in :func:`_add_structural_settings`.
_STRUCTURAL_ROUNDTRIP: dict[str, tuple[_Structural, ...]] = {
    "sfh": (
        _Structural("bin_edges_gyr", "bin_edges_gyr", None),
        _Structural("age_kernel", "age_kernel", None),
        _Structural("field_centering", "field_centering", 1.0),
    ),
    "met": (
        # Emitted as 'type', the key every other group uses (#1720). Defaulting
        # to 'delta' keeps it off the round-trip for the ordinary model, so a
        # met={} entry is never forced onto call sites that diff against
        # from_groups.
        _Structural("type", "met_mode", "delta"),
    ),
    # No 'stellar' entry: that group is gone (#1720). Its one setting was the
    # metallicity mode, and it is emitted above as met={'type': ...}.
    "dust_attenuation": (
        # The two law attributes are emitted by hand in
        # _emit_declared_structural (one 'law' when both screens agree, the
        # pair when they differ), which then skips these entries. They stay
        # because test_structural_settings_roundtrip asserts every structural
        # attribute is named here -- they are coverage, not logic.
        _Structural(
            "law_bc", "dust_law_bc", None, only_types=("single_component", "two_component")
        ),
        _Structural(
            "law_diff", "dust_law_diff", None, only_types=("single_component", "two_component")
        ),
        _Structural("law_neb", "dust_law_neb", None),
        # Witt & Gordon (2000) screen selectors (FSPS dust_type=3). Only read
        # by the parser when the dust type is wg00, so a non-WG00 spec always
        # holds the defaults and never emits them.
        _Structural("dust_curve", "dust_wg00_curve", "mw", only_types=("wg00",)),
        _Structural("geometry", "dust_wg00_geometry", "shell", only_types=("wg00",)),
        _Structural("structure", "dust_wg00_structure", "homogeneous", only_types=("wg00",)),
        # Per-component law-parameter overrides (slope/bump_strength/delta/Rv x
        # bc/diff/neb), the Lyman flags and the law keys are emitted by
        # _add_structural_settings / _emit_declared_structural, not by this table.
        # They are covered by the `hand_written` allowlist in
        # test_structural_settings_roundtrip, which is where a hand-emitted key
        # belongs: an entry here must name a real attribute on Parameters.
    ),
    "dust_emission": (
        # Astrodust+PAH (HD23): spinning dust (AME) enable + cold-neutral-medium
        # fraction. Structural config, forwarded to component_factory (#1093).
        _Structural("spinning_dust", "astrodust_spinning_dust", False, only_types=("astrodust",)),
        _Structural("f_cnm", "astrodust_f_cnm", 0.28, only_types=("astrodust",)),
        # eta_balance is a PARAMETER (dust_eta_balance), not a settings attribute,
        # so it has no attribute for this table to target — it is covered by the
        # test's hand_written allowlist instead.
    ),
    "neb": (
        _Structural("full_catalog", "cue_full_catalog", False, only_types=("cue",)),
        _Structural(
            "grid",
            "cloudy_grid_path",
            None,
            only_types=("cloudy",),
            # Parameters fills an unset path with the grid matching the default
            # SSP family; emitting *that* would bake a machine-specific
            # absolute path into a portable grammar dict, so compare against it.
            resolved_default=lambda spec: spec._default_cloudy_grid(),
        ),
    ),
    "shock": (
        _Structural("norm", "shock_norm", "frac"),
        _Structural("abundance", "shock_abundance", "solar"),
        _Structural("component", "shock_component", "combined"),
    ),
    "igm": (
        # Without this the patchy-reionization params (bubble_mpc, x_HI) were
        # emitted while the toggle that legalizes them was not, so re-parsing
        # a patchy spec raised "Unknown key 'bubble_mpc' in group 'igm'".
        _Structural("patchy", "igm_patchy", False),
    ),
    "agn": (_Structural("norm", "agn_norm", "cigale_joint"),),
    "foreground": (
        # The MW screen declares no fitted parameters, so its group never
        # entered the per-group emit loop at all — see the no-parameter pass
        # at the end of parameters_to_groups.
        _Structural("ebmv_mw", "foreground_ebmv_mw", 0.0),
        _Structural("law", "foreground_law", "cardelli"),
        _Structural("rv", "foreground_rv", 3.1),
    ),
}


def _differs_from_default(value: object, default: object) -> bool:
    """True when a structural setting has been moved off its default.

    Parameters
    ----------
    value : object
        Value read off the spec.
    default : object
        Value the spec holds when the user did not set the key.

    Returns
    -------
    bool
        Whether the setting must be emitted. Array-valued settings
        (``sfh['bin_edges_gyr']``) compare elementwise, so a plain ``!=``
        would raise in a boolean context.
    """
    if value is None or default is None:
        return value is not default
    if isinstance(value, (list, tuple)) or hasattr(value, "shape"):
        import numpy as np

        return not np.array_equal(np.asarray(value), np.asarray(default))
    return bool(value != default)


def _emit_declared_structural(group_name: str, group_output: dict, spec: Parameters) -> None:
    """Emit every non-default plain structural setting for one group.

    Parameters
    ----------
    group_name : str
        Group name (e.g. ``'sfh'``).
    group_output : dict
        Group dict to fill (modified in place). Its ``'type'`` entry, when
        already present, gates the ``only_types`` settings.
    spec : Parameters
        The Parameters object to read settings from.
    """
    group_type = group_output.get("type")
    entries_to_skip = set()

    # Special handling for dust laws: emit 'law' when both screens share,
    # or 'law_bc'/'law_diff' otherwise
    if group_name == "dust_attenuation" and group_type in ("single_component", "two_component"):
        law_bc = getattr(spec, "dust_law_bc", None)
        law_diff = getattr(spec, "dust_law_diff", None)
        if law_bc is not None or law_diff is not None:
            if group_type == "single_component":
                # single_component: emit as 'law'
                if law_bc is not None:
                    group_output["law"] = law_bc
            elif law_bc == law_diff and law_bc is not None:
                # two_component with shared law: emit as 'law'
                group_output["law"] = law_bc
            else:
                # two_component with different laws: emit as 'law_bc' and 'law_diff'
                if law_bc is not None:
                    group_output["law_bc"] = law_bc
                if law_diff is not None:
                    group_output["law_diff"] = law_diff
            # Skip the normal law_bc/law_diff handling below to avoid duplication
            entries_to_skip.add("law_bc")
            entries_to_skip.add("law_diff")

    for entry in _STRUCTURAL_ROUNDTRIP.get(group_name, ()):
        if entry.key in entries_to_skip:
            continue
        if entry.only_types is not None and group_type not in entry.only_types:
            continue
        default = entry.default if entry.resolved_default is None else entry.resolved_default(spec)
        value = getattr(spec, entry.attr, default)
        if _differs_from_default(value, default):
            group_output[entry.key] = value


def _short_names_for_group(group: str, param_partition: dict[str, str]) -> set[str]:
    """Return the set of short and full names every declared param exposes
    under ``group`` (e.g. ``"agn.torus"`` → ``{"tau_skirtor", "agn_tau_skirtor", ...}``).

    Used by :func:`_validate_user_keys` to recognize per-parameter overrides
    when walking a user's group dict.
    """
    from tengri.parameters._aliases import legacy_names_for

    out: set[str] = set()
    for full_name, owner in param_partition.items():
        if owner != group:
            continue
        out.add(full_name)
        out.add(_extract_short_name(full_name, {}))
        # Legacy spellings stay accepted so a rename does not turn a working
        # group dict into "Unknown key" (#1296). The override lookup warns
        # when one is actually used; admitting them here only stops the
        # validator rejecting them first.
        for legacy_full in legacy_names_for(full_name):
            out.add(legacy_full)
            out.add(_extract_short_name(legacy_full, {}))
    return out


def _short_names_for_registered_type(type_name: str | None) -> set[str]:
    """Short + full param names declared by a user-registered SEDModelComponent
    subclass selected via ``type=<type_name>``.

    The per-group validator only sees params that already live on the
    structural ``Parameters`` instance. User-registered subclasses
    (``class MyDust(SEDModelComponent): T = Uniform(...)``) aren't in
    that pool yet, so a per-parameter override like ``"T": Fixed(35)``
    is rejected before the build can wire the subclass in (#391).

    Returns both the short name (``T``) and the prefixed full name
    (``dust_T``) so either spelling is accepted in the user's group dict.
    """
    if not type_name:
        return set()
    # First-party import: swallowing a failure here would return an empty set and
    # reject the user's custom-component params as "unknown keys" with no hint why.
    from tengri.components.sed_model_component import _REGISTRY

    cls = _REGISTRY.get(type_name)
    if cls is None:
        return set()
    prefix = getattr(cls, "parameter_prefix", "")
    priors = getattr(cls, "_priors", {}) or {}
    out: set[str] = set()
    for short in priors:
        out.add(short)
        out.add(f"{prefix}{short}")
    return out


def _validate_user_keys(
    kwargs: dict,
    structural_params: Parameters,
    param_partition: dict[str, str],
) -> None:
    """Validate that every key the user supplied is recognized.

    Walks each group dict (and any sub-block dicts) and checks each key
    against the union of:

    1. Structural keys for that group (``"type"``, ``"*"``, etc.).
    2. Short and full names of every parameter partitioned to that group.
    3. For AGN: short/full names of *shared* AGN params (cross-level
       acceptance — see :func:`_build_agn_search_view`).

    Unknown keys raise :class:`ValueError` with a "Did you mean ...?"
    hint generated via :mod:`difflib`. Silent typos were the dominant
    "AI slop" failure mode of the nested-dict API before this validator
    was added (issue tracked in the forward-model cleanup arc).
    """
    valid_top_groups = {
        "sfh",
        "met",
        "dust_attenuation",
        "dust_emission",
        "neb",
        "shock",
        "igm",
        "radio",
        "xray",
        "agn",
    }

    # Build the AGN cross-level acceptance set (shared params land in any sub-block).
    agn_shared_names = _short_names_for_group("agn", param_partition)

    for top_key, top_val in kwargs.items():
        if top_key in _TOP_LEVEL_SETTINGS:
            continue
        if top_key in _SEDMODEL_PASSTHROUGH:
            continue
        if top_key in structural_params._distributions:
            # A top-level free-form override like ``redshift=Fixed(0.1)``.
            continue
        if top_key not in valid_top_groups:
            # _translate_structural already raised on unknown top-level keys;
            # this branch only fires when a dist/sentinel slipped through.
            continue

        if not isinstance(top_val, dict):
            continue

        # Validate the top-level group dict.
        group_allowed = _GROUP_STRUCTURAL_KEYS.get(top_key, frozenset({"type", "*"}))
        param_names = _short_names_for_group(top_key, param_partition)
        if top_key == "agn":
            # AGN top-level accepts only the shared param short/full names
            # (agn_log_lbol, agn_lum_ratio), not sub-block-owned params.
            # Sub-block params written at the top level raise with guidance
            # on correct nesting (e.g., agn={'torus': {'tau_skirtor': ...}}).
            # This matches dust.emission strictness: parameters must be nested
            # under their owning sub-block.
            param_names = param_names | agn_shared_names
        # NOTE: the dust top level deliberately does NOT accept dust.emission
        # short names. It used to, "for legacy code that flattens emission
        # params at the dust level ... still resolved via the dust.emission
        # group path" — but the resolution half was never wired. Measured:
        # **22 of 22** emission params written at the dust level were accepted
        # and silently discarded, with no error and no warning, so
        # ``dust={'emission': {...}, 'qpah': Uniform(1, 4)}`` ran the fit with
        # qpah pinned at its default and one fewer free dimension than the
        # author wrote. A form that silently does nothing can have no working
        # caller, so refusing it cannot break code that works today.
        # (``agn`` genuinely resolves its cross-level names — 14/14 applied —
        # which is why that union below stays.)
        elif top_key == "igm":
            # The IGM top-level accepts DLA param short names for the
            # builder-factory output form ``igm={'dla': True, 'log_n_hi': ...}``,
            # which flattens DLA params at the igm level. The new nested
            # ``igm={'dla': {...}}`` form is validated separately below.
            param_names = param_names | _short_names_for_group("igm.dla", param_partition)

        # User-registered SEDModelComponent subclasses (#391): if the
        # group dict picks a custom ``type``, add that subclass's
        # declared short/full param names to the accepted set.
        if isinstance(top_val.get("type"), str):
            param_names = param_names | _short_names_for_registered_type(top_val["type"])

        # Optional Cue knobs (#653): a ``type='cue'`` neb group also accepts
        # the density / abundance / ionizing-spectrum params, which are
        # registered on demand and so absent from ``param_partition``.
        if top_key == "neb" and top_val.get("type") == "cue":
            param_names = param_names | _OPTIONAL_NEB_PARAM_NAMES

        _check_dict_keys(top_key, top_val, group_allowed | param_names, param_partition)

        # Recurse into sub-block dicts.
        if top_key == "igm" and isinstance(top_val.get("dla"), dict):
            sub_allowed = frozenset({"type", "*"})
            sub_params = _short_names_for_group("igm.dla", param_partition)
            _check_dict_keys("igm.dla", top_val["dla"], sub_allowed | sub_params, param_partition)
        if top_key == "dust_attenuation" and isinstance(top_val.get("emission"), dict):
            # The nested dust_attenuation={'emission': ...} form is retired; dust_emission
            # is now a top-level group. _translate_dust_attenuation already raises
            # but we validate it here for completeness in case validation runs before translation.
            raise ValueError(
                "dust_attenuation={'emission': ...} is retired; "
                "IR emission is now a separate top-level group. "
                "Rewrite dust_attenuation={...}, dust_emission={...} as separate top-level "
                "entries."
            )
        elif top_key == "agn":
            for sub_name in _AGN_SUBBLOCK_KEYS:
                sub = top_val.get(sub_name)
                if not isinstance(sub, dict):
                    continue
                sub_group = f"agn.{sub_name}"
                sub_allowed = _GROUP_STRUCTURAL_KEYS[sub_group]
                sub_params = _short_names_for_group(sub_group, param_partition)
                sub_params = sub_params | _short_names_for_registered_type(sub.get("type"))
                # Cross-level: sub-block dict may also legitimately carry
                # shared AGN param names.
                _check_dict_keys(
                    sub_group,
                    sub,
                    sub_allowed | sub_params | agn_shared_names,
                    param_partition,
                )
        elif top_key == "radio":
            # Validate nested radio={'sf': {...}} / {'agn': {...}} sub-blocks
            # so a typo'd FIRRC / DPL key raises instead of silently vanishing.
            for sub_name in ("sf", "agn"):
                sub = top_val.get(sub_name)
                if not isinstance(sub, dict):
                    continue
                sub_group = f"radio.{sub_name}"
                sub_allowed = _GROUP_STRUCTURAL_KEYS[sub_group]
                sub_params = _short_names_for_group(sub_group, param_partition)
                sub_params = sub_params | _short_names_for_registered_type(sub.get("type"))
                _check_dict_keys(sub_group, sub, sub_allowed | sub_params, param_partition)


def _check_dict_keys(
    group: str,
    user_dict: dict,
    allowed: set,
    param_partition: dict[str, str],
) -> None:
    """Raise ``ValueError`` on any unrecognized key in ``user_dict``."""
    for key in user_dict:
        if key in allowed:
            continue

        # Special case: user wrote 'defaults' instead of 'all_params'
        if key == "defaults":
            raise ValueError(
                f"Unknown key 'defaults' in group {group!r}. Did you mean 'all_params'? "
                f"The nested-dict grammar uses ``'all_params': FREE`` / ``FIXED`` to set the "
                f"wildcard policy (matching the builder factories' ``all_params=`` parameter)."
            )

        # Suggestion pool: same group's allowed keys + every short name
        # across all groups (helps the "wrong group, right name" case).
        suggestion_pool = set(allowed)
        for full_name in param_partition:
            suggestion_pool.add(_extract_short_name(full_name, {}))
            suggestion_pool.add(full_name)
        # A parameter written one level too high is the common case, and
        # "Unknown key 'alpha' ... Did you mean: alpha?" — the message this
        # produced before — tells the reader to write exactly what they wrote.
        # Name the sub-block instead.
        owner = _subblock_owning(str(key), group, param_partition)
        if owner is not None:
            sub = owner.split(".", 1)[1]
            raise ValueError(
                f"{key!r} is a {owner!r} parameter, not a {group!r} one, so writing "
                f"it here would be silently ignored. Nest it: "
                f"{group}={{{sub!r}: {{{key!r}: ...}}}}."
            )
        suggestions = difflib.get_close_matches(str(key), list(suggestion_pool), n=2, cutoff=0.6)
        # A suggestion identical to the rejected key is noise, not help.
        suggestions = [s for s in suggestions if s != str(key)]
        # Filter out the internal '*' key from suggestions defensively.
        suggestions = [s for s in suggestions if s != WILDCARD_KEY]
        suggest_str = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        # Display user-facing keys only (exclude internal WILDCARD_KEY '*').
        displayed_keys = sorted(
            {
                k
                for k in _GROUP_STRUCTURAL_KEYS.get(group, frozenset({"type"}))
                if k != WILDCARD_KEY
            }
        )
        raise ValueError(
            f"Unknown key {key!r} in group {group!r}.{suggest_str} "
            f"Valid structural keys for this group are: "
            f"{displayed_keys}."
        )


def _subblock_owning(key: str, group: str, param_partition: dict[str, str]) -> str | None:
    """The ``group.sub`` block that declares ``key``, if one does.

    Parameters
    ----------
    key : str
        The rejected key, short or fully prefixed.
    group : str
        The group the key was written under.
    param_partition : dict
        Full parameter name -> owning group.

    Returns
    -------
    str or None
        ``"dust.emission"``-style owner, or ``None`` when no sub-block of
        ``group`` declares it.
    """
    for sub in _GROUP_STRUCTURAL_KEYS:
        if not sub.startswith(f"{group}."):
            continue
        if key in _short_names_for_group(sub, param_partition):
            return sub
    return None


def _build_agn_search_view(param_name: str, agn_dict: dict, group: str) -> dict:
    """Build the resolution view for one AGN parameter.

    AGN parameters live in a two-level nest: the top-level ``agn`` dict
    plus up to six sub-block dicts (``disc``/``torus``/``nlr``/``blr``/
    ``feii``/``atten``). To keep the API friendly, a shared parameter can be
    supplied at *either* level — the partition table records the canonical
    location, but a user who writes ``agn={'disc': {'agn_log_lbol': Uniform(...)}}``
    expects the value to take effect even though ``agn_log_lbol`` is nominally
    a shared (top-level) param.

    A SUB-BLOCK parameter, however, must be nested under its OWNING sub-block
    per the partition table. Writing a sub-block-owned parameter under a
    non-owning sub-block (one that consumes but does not own it) raises
    with guidance.

    This helper assembles a single dict the caller can pass to
    :func:`_resolve_value`:

    1. The canonical location for ``param_name`` (top level if
       ``group == "agn"``; the matching sub-block if ``group.startswith("agn.")``).
    2. Every sibling location that also carries an override for the same
       short name (with validation that the parameter is allowed there).

    If a parameter appears in more than one location with conflicting
    values, raise :class:`ValueError` to flag the ambiguity instead of
    silently picking one.

    Parameters
    ----------
    param_name : str
        Full parameter name (e.g. ``"agn_log_lbol"``).
    agn_dict : dict
        User's top-level ``agn`` dict.
    group : str
        Partition group for ``param_name`` (``"agn"`` for shared,
        ``"agn.<subblock>"`` for sub-block).

    Returns
    -------
    dict
        Search dict for :func:`_resolve_value`. The wildcard ``'*'``
        from the canonical location is preserved when present.

    Raises
    ------
    ValueError
        If a parameter is written under a non-owning sub-block.
    """
    if not isinstance(agn_dict, dict):
        return {}

    # The short name the resolver expects (`_extract_short_name` strips
    # the `agn_` prefix from full AGN param names; pre-compute it here
    # so we can search every candidate dict by either spelling).
    short_name = param_name[4:] if param_name.startswith("agn_") else param_name

    # Canonical and sibling dicts to scan.
    canonical_subkey = group.replace("agn.", "") if group.startswith("agn.") else None
    canonical_dict = (
        agn_dict.get(canonical_subkey, {}) if canonical_subkey is not None else agn_dict
    )
    if not isinstance(canonical_dict, dict):
        canonical_dict = {}

    siblings = []
    if canonical_subkey is None:
        # Shared param: check every sub-block dict for a stray override.
        for k in _AGN_SUBBLOCK_KEYS:
            sub = agn_dict.get(k)
            if isinstance(sub, dict):
                siblings.append((k, sub))
    else:
        # Sub-block param: also check the top level and OTHER sub-blocks for
        # stray overrides. The top level is allowed (shared params can land there);
        # other sub-blocks are not allowed (a sub-block param belongs only in its owner).
        siblings.append(("<top>", agn_dict))
        # Check other sub-blocks for stray overrides (will be rejected if found)
        for k in _AGN_SUBBLOCK_KEYS:
            if k != canonical_subkey:
                sub = agn_dict.get(k)
                if isinstance(sub, dict):
                    siblings.append((k, sub))

    # Collect (location, value) for every place this param appears.
    hits = []
    for key in (short_name, param_name):
        if key in canonical_dict and key not in ("type", "*"):
            hits.append(("<canonical>", canonical_dict[key]))
            break
    for location, sub in siblings:
        for key in (short_name, param_name):
            if key in sub and key not in ("type", "*"):
                # VALIDATION: Check if this parameter is allowed in this sibling location.
                # RULE: Sub-block-owned parameters must be in their owner sub-block.
                # Shared parameters can go anywhere (top level or any sub-block).
                param_owner = _AGN_PARTITION.get(param_name, "agn")
                if location != "<top>" and location != "none":
                    # location is a sub-block name (e.g., "torus", "disc", "atten")
                    sibling_group = f"agn.{location}"
                    # Only reject if this is a sub-block-owned parameter in the wrong sub-block
                    if param_owner != "agn" and param_owner != sibling_group:
                        # Sub-block parameter in wrong sub-block
                        owner_subblock = param_owner.replace("agn.", "")
                        raise ValueError(
                            f"{key!r} is a {param_owner!r} parameter, not a "
                            f"{sibling_group!r} one. Nest it: "
                            f"agn={{{owner_subblock!r}: {{{key!r}: ...}}}}."
                        )
                    # Otherwise: shared param in sub-block (OK) or right sub-block (OK)
                hits.append((location, sub[key]))
                break

    if len(hits) > 1:
        locs = ", ".join(h[0] for h in hits)
        raise ValueError(
            f"AGN parameter {param_name!r} is set in multiple locations ({locs}). "
            f"Set it in exactly one place — either the top-level ``agn`` dict "
            f"or one sub-block dict — to avoid ambiguity."
        )

    if not hits:
        # Nothing found anywhere; surface the canonical dict so its wildcard
        # ('*') applies. For a sub-block param whose sub-block carries no '*',
        # inherit the top-level agn '*' so a top-level ``agn={'*': FREE}``
        # governs sub-block params too. This is scoped downstream by
        # ``wildcard_active`` (block-scoped wildcard), so it frees only the
        # parameters the active blocks actually consume — e.g. ``agn_polar_ebv``
        # is partitioned to ``agn.atten`` but consumed by the SKIRTOR torus, so
        # a top-level wildcard must be able to reach it.
        if canonical_subkey is not None and "*" not in canonical_dict and "*" in agn_dict:
            merged = dict(canonical_dict)
            merged["*"] = agn_dict["*"]
            return merged
        return canonical_dict

    # Single hit: return a synthetic dict carrying that one override
    # plus the wildcard from the canonical location (so '*': FREE inside
    # a sub-block still controls shared params landed via this view).
    # A sub-block wildcard takes precedence over the top-level one.
    _, found_val = hits[0]
    view: dict = {short_name: found_val}
    if "*" in canonical_dict:
        view["*"] = canonical_dict["*"]
    elif canonical_subkey is not None and "*" in agn_dict:
        view["*"] = agn_dict["*"]
    return view


def _translate_agn(agn_dict: dict, result: dict) -> None:
    """Translate agn group to AGN composable block selectors.

    Activates agn_model='composable' and sets per-block selectors
    (agn_disc_block, agn_torus_block, agn_nlr_block, agn_blr_block,
    agn_feii_block, agn_attenuation_block). Omitted blocks default to 'none'.

    A top-level ``'type'`` key picks a named non-composable AGN model
    (e.g. ``'richards2006'``, ``'kubota_done'``, ``'multicolor_agn'``),
    validated lazily by ``resolve_agn_model`` at predict time. When
    ``type='composable'`` (or absent), the sub-block selectors are honored.
    Mixing a non-composable ``type`` with sub-blocks is an error.

    The deprecated ``lines`` sub-block (which combined NLR and BLR) is
    expanded to independent ``nlr`` and ``blr`` sub-blocks via
    ``expand_lines_alias``, with a ``DeprecationWarning`` emitted.

    Parameters
    ----------
    agn_dict : dict
        User's agn group specification.
    result : dict
        Structural kwargs dict (modified in-place).

    Raises
    ------
    ValueError
        If unknown block type, unknown model type, or invalid block
        specification. Also raised if both ``lines`` and ``nlr``/``blr``
        are provided.
    """
    # Top-level 'type' selects a monolithic AGN model when not 'composable'.
    # Previously this key was silently dropped and the model collapsed to
    # composable-with-all-none-blocks, which emits identically zero — a
    # silent-failure footgun (closes #417 second case).
    top_type = agn_dict.get("type")
    if top_type is not None and top_type != "composable":
        # Reject mixing a monolithic ``type`` with sub-block selectors —
        # the two surfaces are mutually exclusive.
        used_blocks = sorted(k for k in _AGN_SUBBLOCK_KEYS if k in agn_dict)
        if used_blocks:
            raise ValueError(
                f"agn['type']={top_type!r} selects a monolithic AGN model, "
                f"but sub-block keys {used_blocks} are also present. Drop "
                f"the sub-blocks, or remove 'type' and let the composable "
                f"runner use the per-block selectors."
            )
        # Forward ``type`` to ``agn_model`` and skip the block-selector
        # plumbing. Unknown model names are validated lazily by
        # ``resolve_agn_model`` at predict time, where the available list
        # is fully populated (some models register late through plugins).
        result["agn_model"] = top_type
        return

    # Activate composable model
    result["agn_model"] = "composable"

    # Handle deprecated `lines` sub-block: expand to independent nlr/blr
    # and emit a DeprecationWarning.
    agn_dict = dict(agn_dict)  # Shallow copy to avoid modifying the original
    if "lines" in agn_dict:
        lines_spec = agn_dict.pop("lines")
        # Check for conflicting specification: both lines and nlr/blr
        if isinstance(lines_spec, dict) and ("nlr" in agn_dict or "blr" in agn_dict):
            raise ValueError(
                "Cannot specify both deprecated 'lines' sub-block and "
                "the new 'nlr' and/or 'blr' sub-blocks. Use only 'nlr' "
                "and 'blr', or update your code to use the new grammar."
            )
        # Expand the deprecated lines type to (nlr_type, blr_type)
        if isinstance(lines_spec, dict):
            lines_type = lines_spec.get("type", "none")
            from tengri.components.agn.blocks._aliases import expand_lines_alias

            try:
                nlr_type, blr_type = expand_lines_alias(lines_type)
            except ValueError as e:
                raise ValueError(f"Invalid 'lines' type in agn dict: {e}") from e

            # Emit the deprecation warning
            warnings.warn(
                f"The AGN 'lines' slot is deprecated; use independent 'nlr' "
                f"and 'blr' slots. 'lines' type '{lines_type}' maps to "
                f"nlr='{nlr_type}', blr='{blr_type}'.",
                DeprecationWarning,
                stacklevel=2,
            )

            # Construct nlr and blr sub-blocks from the expanded types.
            # Carry over any wildcard and per-param overrides from lines.
            nlr_spec = {"type": nlr_type}
            blr_spec = {"type": blr_type}

            # Copy over wildcard and per-param overrides from lines_spec.
            # Heuristic: parameters with "blr" in the name go to blr only;
            # those with "nlr" go to nlr only; wildcard and other params
            # replicate to both.
            for key, val in lines_spec.items():
                if key == "type":
                    continue
                if key == "*":
                    # Wildcard applies to both sub-blocks
                    nlr_spec["*"] = val
                    blr_spec["*"] = val
                elif isinstance(key, str) and "blr" in key:
                    # BLR-specific parameter
                    blr_spec[key] = val
                elif isinstance(key, str) and "nlr" in key:
                    # NLR-specific parameter
                    nlr_spec[key] = val
                else:
                    # Ambiguous or shared: replicate to both
                    nlr_spec[key] = val
                    blr_spec[key] = val

            agn_dict["nlr"] = nlr_spec
            agn_dict["blr"] = blr_spec

    # Define the six canonical sub-blocks and their type validator sets
    block_specs = {
        "disc": _VALID_AGN_DISC_TYPES,
        "torus": _VALID_AGN_TORUS_TYPES,
        "nlr": _VALID_AGN_NLR_TYPES,
        "blr": _VALID_AGN_BLR_TYPES,
        "feii": _VALID_AGN_FEII_TYPES,
        "atten": _VALID_AGN_ATTEN_TYPES,
    }

    # Map sub-block names to their result kwargs (module-level: the round-trip
    # emitter reads the same table, so the two directions cannot drift).
    block_to_kwarg = _AGN_BLOCK_TO_KWARG

    # Process each sub-block
    for block_name, valid_types in block_specs.items():
        if block_name not in agn_dict:
            # Not specified: default to 'none'
            result[block_to_kwarg[block_name]] = "none"
            continue

        block_spec = agn_dict[block_name]
        if not isinstance(block_spec, dict):
            # Malformed: skip or raise
            raise ValueError(
                f"agn['{block_name}'] must be a dict with 'type' and optional parameters, "
                f"got {type(block_spec).__name__}."
            )

        # Special handling for atten: 'law' key selects smc_prevot via DUST_LAWS,
        # while 'type' selects genuine attenuation models (polar_dust, grahsp_biatten, etc)
        block_type = None
        if block_name == "atten":
            law_key = block_spec.get("law")
            type_key = block_spec.get("type")

            # Check for conflicting law and type keys
            if law_key is not None and type_key is not None:
                raise ValueError(
                    "agn['atten'] cannot specify both 'law' and 'type' keys. "
                    "Use 'law' for DUST_LAWS curves (e.g. law='prevot_smc') or "
                    "'type' for genuine attenuation models (e.g. type='polar_dust')."
                )

            # Reject old law-as-type spelling: type='smc_prevot'
            if type_key == "smc_prevot":
                raise ValueError(
                    "agn['atten'] type='smc_prevot' is no longer supported. "
                    "Use the new form with law key instead:\n"
                    "  agn={'atten': {'law': 'prevot_smc', 'ebv': Uniform(...)}}\n"
                    "'prevot_smc' is the only law this block implements -- it applies "
                    "that curve unconditionally, so the rename is a spelling change, "
                    "not a new choice."
                )

            if law_key is not None:
                # Validate against the laws the block IMPLEMENTS, not against
                # every name in DUST_LAWS. The old check accepted all 22 and
                # mapped them to the same single-curve block, so a user who
                # selected Calzetti silently got Prevot SMC (#2012). Same policy
                # as `foreground`, which has the same single-curve limitation
                # and has always refused the laws it does not wire.
                if law_key not in _VALID_AGN_ATTEN_LAWS:
                    from tengri.components.dust.laws._registry import DUST_LAWS

                    valid = sorted(_VALID_AGN_ATTEN_LAWS)
                    detail = (
                        f"'{law_key}' is a real attenuation law, but the AGN "
                        f"attenuation stage does not implement it"
                        if law_key in DUST_LAWS
                        else f"Unknown dust law '{law_key}'"
                    )
                    suggestions = difflib.get_close_matches(law_key, valid, n=2, cutoff=0.6)
                    suggest_str = (
                        f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
                    )
                    raise ValueError(
                        f"{detail}.{suggest_str}\n"
                        f"Valid agn['atten'] laws: {valid}.\n"
                        f"The block applies the Prevot SMC curve unconditionally, so "
                        f"accepting another name would silently substitute this one. "
                        f"For a different AGN attenuation curve, use "
                        f"agn={{'atten': {{'type': 'polar_dust', 'polar_ebv': 0.1}}}} "
                        f"to apply polar dust extinction (using Pei 1992 SMC curve)."
                    )
                block_type = "smc_prevot"
            else:
                block_type = type_key

        else:
            block_type = block_spec.get("type")

        if block_type is None:
            # Assume 'none' if no type/law given
            result[block_to_kwarg[block_name]] = "none"
            continue

        # Validate type
        if block_type not in valid_types:
            suggestions = difflib.get_close_matches(block_type, valid_types, n=2, cutoff=0.6)
            suggest_str = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
            raise ValueError(f"Unknown agn_{block_name}_block type '{block_type}'.{suggest_str}")

        result[block_to_kwarg[block_name]] = block_type

    # Cross-block normalization policy (``agn['norm']``): menu single-sourced
    # from AGN_NORM_POLICIES so the grammar can't drift from the runner (#556).
    if "norm" in agn_dict:
        from tengri.components.agn.blocks import AGN_NORM_POLICIES

        _norm = agn_dict["norm"]
        if _norm not in AGN_NORM_POLICIES:
            raise ValueError(f"Unknown agn['norm']={_norm!r}. Valid: {sorted(AGN_NORM_POLICIES)}")
        result["agn_norm"] = _norm


def _partition_by_group(
    all_param_names: list[str],
    dust_emission_active: bool,
    *,
    met_group: str = "sfh",
    agn_flat: bool = False,
) -> dict[str, str]:
    """Partition parameter names by their owning group.

    Returns a dict: param_name -> group_name. For sub-groups, group_name
    includes a dot (e.g., "dust.emission").

    Parameters
    ----------
    all_param_names : list[str]
        All declared parameter names from the structural Parameters.
    dust_emission_active : bool
        If True, dust_emission params belong to "dust.emission"; else ignored.
    met_group : str, optional
        Group that ``met_*`` parameters belong to.
    agn_flat : bool, optional
        Route every ``agn_*`` parameter to a flat ``"agn"`` group instead of
        its ``agn.<block>`` sub-block. Set by the round-trip emitter when a
        *monolithic* AGN model is selected: ``_translate_agn`` raises on a
        non-composable ``agn['type']`` that appears alongside sub-block keys,
        so a nested emission would be a dict the grammar refuses to read back.
        Flat per-parameter keys are accepted there, and the short names are
        unaffected (both forms strip only the ``agn_`` prefix).

    Returns
    -------
    dict[str, str]
        Mapping from param name to owning group.
    """
    partition = {}

    for name in all_param_names:
        if name == "redshift" or name == "apply_igm":
            partition[name] = "_toplevel"
        elif name.startswith("agn_"):
            if agn_flat:
                partition[name] = "agn"
                continue
            # Use partition table for fine-grained routing
            partition[name] = _AGN_PARTITION.get(name, "agn")
            # Catch-all for grahsp_* -> disc
            if partition[name] == "agn" and "grahsp" in name:
                partition[name] = "agn.disc"
        elif name.startswith("xray_"):
            partition[name] = "xray"
        elif name in _RADIO_SF_PARAM_NAMES:
            # Model-specific FIRRC evolution coefficients live in the radio.sf
            # sub-block (radio={'sf': {...}}).
            partition[name] = "radio.sf"
        elif name in _RADIO_AGN_PARAM_NAMES:
            # AGN radio loudness / power-law / DPL knobs live in radio.agn.
            partition[name] = "radio.agn"
        elif name.startswith("radio_"):
            partition[name] = "radio"
        elif name.startswith("dla_"):
            partition[name] = "igm.dla"
        elif name.startswith("igm_"):
            partition[name] = "igm"
        elif name.startswith("neb_") or name.startswith("ionspec_") or name.startswith("gas_log"):
            partition[name] = "neb"
        elif name.startswith("shock_"):
            partition[name] = "shock"
        elif dust_emission_active and name in _DUST_EMISSION_PARAM_NAMES:
            partition[name] = "dust_emission"
        elif name.startswith("dust_"):
            partition[name] = "dust_attenuation"
        elif name.startswith("met_"):
            partition[name] = met_group
        elif name.startswith("sfh_"):
            partition[name] = "sfh"
        else:
            # _structural (settings like mean_sfh_type, dust_model, etc.)
            partition[name] = "_structural"

    return partition


def _resolve_value(
    param_name: str,
    group_dict: dict,
    registry_default: Distribution,
    *,
    wildcard_active: bool = True,
) -> tuple[Distribution, str]:
    """Resolve the final distribution for a single parameter.

    Checks (in order):
    1. Per-parameter override in group_dict (including bare values)
    2. Wildcard '*' (FREE or FIXED)
    3. Registry default

    ``wildcard_active`` (keyword-only, default ``True``) gates the
    wildcard-``FREE`` branch only. When ``False`` a ``'*': FREE`` is treated
    as ``'*': FIXED`` for *this* parameter — used by the AGN group so a group
    wildcard frees only the parameters the active disc/torus/nlr/blr/feii/atten
    blocks actually consume, not the full declared superset (which would
    otherwise create unconstrained no-op nuisance dimensions). An *explicit*
    per-parameter ``FREE`` is handled earlier and is never gated, so the user
    can always force a specific parameter free.

    Parameters
    ----------
    param_name : str
        Full parameter name (e.g., 'sfh_dpl_alpha').
    group_dict : dict
        The user's group dict (e.g., sfh[...]).
    registry_default : Distribution
        The registry's default prior/value for this param.

    Returns
    -------
    distribution : Distribution
        The resolved distribution (or Fixed wrapper).
    provenance : str
        Tag describing why this resolution was chosen. One of:
        ``"user_prior"``, ``"user_fixed"``, ``"user_free"``,
        ``"wildcard_free"``, ``"wildcard_fixed"``, ``"registry_default"``.

    Raises
    ------
    ValueError
        If a parameter name in group_dict is unknown for this group.
    """
    # Extract the short name (e.g., 'alpha' from 'sfh_dpl_alpha')
    # by removing the group prefix
    short_name = _extract_short_name(param_name, group_dict)

    # Accept either the short form ('logU') or the full-prefixed form
    # ('neb_logU') as a per-param override key. The validator already
    # admits both names (see _short_names_for_group), so silently
    # dropping the full-prefix form here would be a footgun (issue #424).
    override_key = None
    if short_name in group_dict:
        override_key = short_name
    elif param_name != short_name and param_name in group_dict:
        override_key = param_name
    else:
        # A renamed parameter also invalidates its *short* key: after
        # agn_frac -> agn_lum_ratio, `agn={'frac': 0.5}` became "Unknown key"
        # (#1296). Accept the legacy spelling, both short and full, and warn
        # -- the full-name alias map alone does not cover the grammar's short
        # form, because the short form is derived by stripping the prefix.
        from tengri.parameters._aliases import _warn_once_if_legacy, legacy_names_for

        for legacy_full in legacy_names_for(param_name):
            legacy_short = _extract_short_name(legacy_full, group_dict)
            for candidate in (legacy_short, legacy_full):
                if candidate in group_dict:
                    _warn_once_if_legacy(candidate, short_name)
                    override_key = candidate
                    break
            if override_key is not None:
                break

    # Check for per-param override
    if override_key is not None:
        val = group_dict[override_key]

        # Validate that this key is actually a parameter (not 'type', '*', etc.)
        structural_keys = {
            "type",
            "*",
            "law_bc",
            "law_diff",
            "law_neb",
            "emission",
            "patchy",
            "dla",
            "slope_bc",
            "slope_diff",
            "slope_neb",
            "bump_strength_bc",
            "bump_strength_diff",
            "bump_strength_neb",
            "delta_bc",
            "delta_diff",
            "delta_neb",
            "Rv_bc",
            "Rv_diff",
            "Rv_neb",
            "lyman_cutoff",
            "lyc_absorb_all",
            "eb_include_lyc",
        }
        if override_key in structural_keys:
            # These are structural keys, not parameters
            return registry_default, "registry_default"

        if val is FREE:
            return _expand_free(param_name, registry_default), "user_free"
        elif val is FIXED:
            # FIXED: convert registry default to Fixed at its center
            if registry_default.is_fixed:
                return registry_default, "user_fixed"
            else:
                return (
                    Fixed(_default_fixed_value(param_name, registry_default)),
                    "user_fixed",
                )
        elif isinstance(val, Distribution):
            # Explicit distribution: use as-is
            tag = "user_fixed" if val.is_fixed else "user_prior"
            return val, tag
        else:
            # Bare value: wrap in Fixed
            return Fixed(val), "user_fixed"

    # Check for wildcard
    if "*" in group_dict:
        wildcard = group_dict["*"]
        if wildcard is FREE:
            if wildcard_active:
                return _expand_free(param_name, registry_default), "wildcard_free"
            # Param is not consumed by the active block selection. A wildcard
            # FREE here would create an unconstrained no-op nuisance dimension,
            # so collapse it to its fixed default instead (block-scoped
            # wildcard). The param stays declared (no missing keys) — it is
            # simply held fixed.
            if registry_default.is_fixed:
                return registry_default, "wildcard_fixed_inactive"
            return (
                Fixed(_default_fixed_value(param_name, registry_default)),
                "wildcard_fixed_inactive",
            )
        elif wildcard is FIXED:
            if registry_default.is_fixed:
                return registry_default, "wildcard_fixed"
            else:
                return (
                    Fixed(_default_fixed_value(param_name, registry_default)),
                    "wildcard_fixed",
                )
        else:
            # Bad wildcard value — only FREE or FIXED are accepted in the
            # wildcard slot ('all_params', or its synonym '*').
            raise ValueError(
                f"The 'all_params' wildcard (also '*') must be FREE or FIXED "
                f"(the sentinels exported from tengri), got {wildcard!r}. "
                f"Did you mean ``'all_params': FREE`` or ``'all_params': FIXED``? "
                f"Note: string 'free'/'fixed' is not accepted — use the sentinel."
            )

    # No override, no wildcard: fall through to registry default (auto-fixed)
    if registry_default.is_fixed:
        return registry_default, "registry_default"
    else:
        return (
            Fixed(_default_fixed_value(param_name, registry_default)),
            "registry_default",
        )


def _extract_short_name(full_param_name: str, group_dict: dict) -> str:
    """Extract short parameter name by removing group prefix.

    E.g., 'sfh_dpl_alpha' -> 'alpha' (for sfh group).
    Handles nested sub-keys: dust.emission params, AGN sub-blocks.
    For SFH composition with ambiguous short names, checks if user
    provided a full-prefix name.

    Parameters
    ----------
    full_param_name : str
        Full parameter name.
    group_dict : dict
        The group dict (to check for full-prefix overrides and short-name ambiguities).

    Returns
    -------
    str
        The short name (or full prefixed name if provided by user).

    Raises
    ------
    ValueError
        If a short name is ambiguous in SFH composition.
    """
    # For SFH composition: check if user provided a full-prefix name
    # If so, prefer that. Otherwise, extract the short name and check for ambiguity.
    if full_param_name.startswith("sfh_"):
        rest = full_param_name[4:]  # Remove 'sfh_'
        parts = rest.split("_", 1)
        if len(parts) == 2:
            short = parts[1]
            # Check if user provided the full param name
            if full_param_name in group_dict:
                return full_param_name
            # Check for short name in composition
            # (If mean_sfh_type is a list, we need to check all types)
            if short in group_dict:
                # User provided the short name; check for ambiguity
                # A short name is ambiguous if it exists in multiple composition types
                sfh_type = group_dict.get("type")
                if isinstance(sfh_type, list):
                    # Multiple types in composition; ambiguity possible
                    # For now, defer to Parameters validation
                    pass
                return short
            # Extract short name as usual
            return parts[1]
        return rest
    elif full_param_name.startswith("met_"):
        return full_param_name[4:]
    elif full_param_name.startswith("dust_"):
        return full_param_name[5:]
    elif full_param_name.startswith("neb_"):
        return full_param_name[4:]
    elif full_param_name.startswith("shock_"):
        return full_param_name[6:]
    elif full_param_name.startswith("ionspec_"):
        return full_param_name[8:]
    elif full_param_name.startswith("gas_log"):
        return full_param_name[4:]
    elif full_param_name.startswith("radio_"):
        return full_param_name[6:]
    elif full_param_name.startswith("xray_"):
        return full_param_name[5:]
    elif full_param_name.startswith("agn_"):
        # AGN params: check partition table to determine prefix stripping
        # For sub-blocks like agn.torus, strip appropriate prefix
        if full_param_name in _AGN_PARTITION:
            group_path = _AGN_PARTITION[full_param_name]
            # If it's a shared agn param, strip 'agn_'; if sub-block, strip more
            if group_path.startswith("agn."):
                # Sub-block: strip 'agn_' and block name prefix if appropriate
                # e.g., 'agn_tau_skirtor' in agn.torus -> 'tau_skirtor'
                return full_param_name[4:]
        # Catch-all for agn_grahsp_* -> disc
        if "grahsp" in full_param_name:
            return full_param_name[4:]
        # Default: just strip 'agn_'
        return full_param_name[4:]
    elif full_param_name.startswith(("igm_", "dla_")):
        return full_param_name[4:]
    else:
        # Top-level (e.g., redshift)
        return full_param_name


# ── Inverse: Parameters to nested-dict form ────────────────────────────────


def parameters_to_groups(spec: Parameters) -> dict:
    """Convert a Parameters back to nested-dict form.

    Inverts parse_groups() by reconstructing the nested-dict
    structure that would reproduce the same Parameters when passed back to
    parse_groups(). Uses provenance tags (if available) to collapse wildcard-
    expanded parameters and preserve explicit overrides.

    Parameters
    ----------
    spec : Parameters
        The Parameters object to convert.

    Returns
    -------
    dict
        Nested-dict suitable for re-passing to parse_groups(**result).

    Notes
    -----
    **Provenance-aware collapsing**: If spec has _group_provenance metadata,
    parameters sharing the same wildcard tag ('wildcard_free' or 'wildcard_fixed')
    are collapsed into a single 'all_params': FREE or 'all_params': FIXED entry,
    with explicit overrides listed separately.

    **Flat-built fallback**: If spec was built via flat-kwarg Parameters(...),
    all parameters are listed explicitly (no wildcard).

    **Roundtrip guarantee**: The output dict, when passed to
    parse_groups(**output), produces a Parameters with identical
    free/fixed partitions and distributions, *and* identical structural
    settings — every non-parameter key ``parse_groups`` accepts
    (``_GROUP_STRUCTURAL_KEYS``) is emitted back whenever it differs from
    its default.

    Structural settings were outside this guarantee until #964, and the
    narrow wording is why that went unnoticed: ``sfh['age_kernel']``,
    ``agn['norm']`` and the WG00 dust selectors were accepted, stored, and
    then silently reverted to their defaults on the next round-trip. The
    rules now live in ``_STRUCTURAL_ROUNDTRIP``, and
    ``tests/contract/test_structural_settings_roundtrip.py`` asserts the two
    tables cannot drift apart again.

    The ``type`` key itself is *not* in that table — it is emitted by
    :func:`_extract_group_type` — and that exemption hid the same class of
    loss for a second round (#1777). Two causes, both now pinned by
    ``tests/contract/test_structural_types_survive_the_roundtrip.py``:

    * the AGN family had one arm returning ``None`` for every ``agn*`` group,
      so the top-level model and all six sub-block selectors were dropped;
      28 of 79 structural selections rebuilt as different physics (up to 98%
      in photometry) and the two AGN recipes lost 14 and 17 free parameters;
    * a group whose non-default type declares no parameters of its own never
      reaches the per-group walk, and the fallback that catches those tested a
      hand-written list naming only ``dust`` and ``igm`` — so
      ``neb={'type': 'ssp'}`` rebuilt with nebular emission off. The rule is
      now "differs from :func:`_default_group_type`", read off a bare spec.

    Only *non-default* values are emitted, so an untouched group stays
    absent from the output rather than growing noise.

    Examples
    --------
    >>> spec = parse_groups(
    ...     sfh={"type": "dpl", "all_params": FREE, "beta": Uniform(1, 3)},
    ...     redshift=Fixed(0.05),
    ... )
    >>> groups = spec.to_groups()
    >>> assert "all_params" in groups["sfh"]  # preferred spelling on output
    >>> roundtripped = parse_groups(**groups)
    >>> spec.free_params == roundtripped.free_params
    True
    """
    result = {}
    # Emit a ``met`` block on the round-trip only when ``met_mode`` is
    # non-default. Otherwise keep met_* under ``sfh`` for back-compat with the
    # legacy fixtures that pre-#311 expected.
    #
    # The block emitted is ``met`` (#1720), which is the only spelling now:
    # ``met={'type': ...}``, the parallel of ``sfh={'type': ...}``.
    use_met_block = getattr(spec, "met_mode", "delta") != "delta"
    # A monolithic AGN model and the six sub-block selectors are mutually
    # exclusive surfaces — ``_translate_agn`` raises when a non-composable
    # ``agn['type']`` appears next to sub-block keys. So the moment the type is
    # emitted (it was not, before #1777), the nested form has to go.
    agn_model = getattr(spec, "agn_model", None)
    agn_flat = agn_model is not None and agn_model != "composable"
    partition = _partition_by_group(
        spec.all_params,
        spec.dust_emission is not None,
        met_group="met" if use_met_block else "sfh",
        agn_flat=agn_flat,
    )
    provenance = getattr(spec, "_group_provenance", {})

    # Group parameters by their owning group
    groups_dict = {}
    for param_name in spec.all_params:
        group = partition.get(param_name, "_structural")

        if group == "_structural" or group == "_toplevel":
            continue

        if group not in groups_dict:
            groups_dict[group] = []
        groups_dict[group].append(param_name)

    # Process each group
    for group_name in sorted(groups_dict.keys()):
        param_names = sorted(groups_dict[group_name])

        # Handle nested groups (dust.emission, agn.*)
        if "." in group_name:
            parent, subkey = group_name.split(".", 1)
            if parent not in result:
                result[parent] = {}
            group_output = result[parent].setdefault(subkey, {})
        else:
            group_output = result.setdefault(group_name, {})

        # Add type from the spec's settings
        type_value = _extract_group_type(group_name, spec)
        if type_value is not None:
            # Special handling for agn.atten: emit 'law' key for smc_prevot
            # (the DUST_LAWS wrapper), 'type' for genuine models
            if group_name == "agn.atten" and type_value == "smc_prevot":
                group_output["law"] = "prevot_smc"
            else:
                group_output["type"] = type_value

        # Add other structural settings
        _add_structural_settings(group_name, group_output, spec)

        # Determine if we can use a wildcard
        wildcard_intent = _analyze_wildcard_intent(param_names, spec, provenance)

        if wildcard_intent is not None:
            # Emit the preferred wildcard spelling for collapsed params.
            group_output[WILDCARD_ALIAS] = wildcard_intent
            explicit_params = _get_explicit_overrides(
                param_names, spec, provenance, wildcard_intent
            )
        else:
            # No wildcard; list all params explicitly
            explicit_params = {p: spec.get_distribution(p) for p in param_names}

        # Add explicit params to the group dict
        for full_name, distribution in explicit_params.items():
            short_name = _extract_short_name(full_name, {})
            group_output[short_name] = distribution

    # Also add groups that have no params but have a configured type (e.g., neb='none')
    for group_name in sorted(_TOP_LEVEL_TYPED_GROUPS):
        if group_name not in result:
            type_value = _extract_group_type(group_name, spec)
            # Normalize a list type to the tuple the cached default holds, so
            # a composed SFH is not reported as differing from itself.
            comparable = tuple(type_value) if isinstance(type_value, list) else type_value
            if type_value is not None and (
                type_value == "none" or comparable != _default_group_type(group_name)
            ):
                result[group_name] = {"type": type_value}

    # Groups carrying structural settings but owning no declared parameters
    # never entered the per-group loop above, so their settings would vanish
    # from the round-trip entirely. The MW foreground screen (#297) is the
    # standing case: it has three settings and no fitted parameters.
    for group_name in sorted(_STRUCTURAL_ROUNDTRIP):
        if group_name in result:
            continue
        type_value = _extract_group_type(group_name, spec)
        pending: dict = {} if type_value is None else {"type": type_value}
        n_before = len(pending)
        _emit_declared_structural(group_name, pending, spec)
        # Emit the group only when a setting actually fired — a bare type is
        # the business of the block above, which knows which are non-default.
        if len(pending) > n_before:
            result[group_name] = pending

    # Handle top-level settings. No `apply_igm`: the igm group carries
    # activation on its own (``type: "none"`` when off, omitted when off and
    # nothing else is set, which now means the same thing), and emitting the
    # retired keyword beside it made every round-trip raise on the way back in.
    # A round-trip that emits a key its own parser refuses is the clearest sign
    # the second switch was redundant.
    if "redshift" in spec.all_params:
        result["redshift"] = spec.get_distribution("redshift")

    if spec.n_grid != 256:
        # Only include if non-default
        result["n_grid"] = spec.n_grid

    return result


@cache
def _default_group_type(group_name: str) -> str | tuple | None:
    """The type a group falls back to when the round-trip omits it.

    Read off a bare ``Parameters()`` through :func:`_extract_group_type`, so it
    is the *same* code path that reports a spec's type — including that
    function's boundary translations (``nebular_mode='off'`` -> ``"none"``, the
    ``shock`` boolean -> ``"mappings"``). Any hand-written copy of these
    defaults is a second source of truth that drifts.

    That is not hypothetical. The emitter used to decide "is this worth
    emitting?" from a literal table naming only ``dust`` and ``igm``, so a
    group whose non-default type happened to declare **no parameters of its
    own** never reached the round-trip at all: ``neb={'type': 'ssp'}`` rebuilt
    with nebular emission switched **off** (#1777).

    Parameters
    ----------
    group_name : str
        Group name, e.g. ``"neb"`` or ``"agn.torus"``.

    Returns
    -------
    str or tuple or None
        The default type. Lists are returned as tuples so the result stays
        hashable for the cache.
    """
    value = _extract_group_type(group_name, Parameters())
    return tuple(value) if isinstance(value, list) else value


def _extract_group_type(group_name: str, spec: Parameters) -> str | list[str] | None:
    """Extract the type value for a group from spec settings.

    Notes
    -----
    ``None`` means "this group has no type axis, or none is selected", and the
    round-trip omits the key. That makes an *unimplemented* arm indistinguishable
    from a genuinely absent axis, which is how the AGN family went unemitted:
    a single ``elif group_name.startswith("agn"): return None`` covered the
    top-level model and all six sub-blocks, annotated "more complex composition
    handled in tests" — no test held it (#1777). Return ``None`` only when the
    axis really is absent.

    Parameters
    ----------
    group_name : str
        Group name (e.g., 'sfh', 'dust_attenuation', 'neb', 'dust_emission', 'agn.disc').
    spec : Parameters
        The Parameters object.

    Returns
    -------
    str or list[str] or None
        The type value, or None if not applicable.
    """
    if group_name == "sfh":
        sfh_type = spec.mean_sfh_type
        # Normalize: if single-element list, return string
        if isinstance(sfh_type, list):
            return sfh_type[0] if len(sfh_type) == 1 else sfh_type
        return sfh_type
    elif group_name == "dust_attenuation":
        return spec.dust_model
    elif group_name == "dust_emission":
        return spec.dust_emission
    elif group_name == "neb":
        # ``Parameters`` stores ``nebular_mode == "off"`` to mean "no nebular
        # contribution"; the dict grammar's canonical name for that state is
        # ``"none"``. Map at the boundary so to_groups() / parse_groups()
        # round-trip cleanly.
        return "none" if spec.nebular_mode == "off" else spec.nebular_mode
    elif group_name == "shock":
        # ``shock`` is a boolean toggle on Parameters; the grammar type is
        # ``"mappings"`` when active and ``"none"`` when off (#851).
        return "mappings" if getattr(spec, "shock", False) else "none"
    elif group_name == "igm":
        # ``apply_igm`` is the on/off switch; ``igm_model`` stores the
        # internal spelling (e.g. ``"inoue"``), which is also a registered
        # grammar alias, so it round-trips through parse_groups unchanged.
        if not getattr(spec, "apply_igm", True):
            return "none"
        return spec.igm_model if hasattr(spec, "igm_model") else None
    elif group_name == "radio":
        # The composable radio grammar carries its types on the ``sf`` /
        # ``agn`` sub-blocks — parse_groups raises on a top-level ``type``
        # mixed with sub-blocks, so the round-trip must not emit one here.
        return None
    elif group_name == "radio.sf":
        return spec.radio_sfr_mode if getattr(spec, "radio", False) else None
    elif group_name == "radio.agn":
        return spec.radio_agn_model if getattr(spec, "radio", False) else None
    elif group_name == "xray":
        return spec.xray_model if hasattr(spec, "xray_model") else None
    elif group_name == "agn":
        # ``None`` means no AGN component at all, and the group is then absent
        # from the round-trip entirely. Otherwise this is either the literal
        # ``"composable"`` or a monolithic model name, and both are grammar
        # types that ``_translate_agn`` accepts back verbatim.
        return getattr(spec, "agn_model", None)
    elif group_name.startswith("agn."):
        block = group_name.split(".", 1)[1]
        attr = _AGN_BLOCK_TO_KWARG.get(block)
        if attr is None:
            return None
        value = getattr(spec, attr, None)
        # ``"none"`` is every sub-block's default, so omitting it keeps an
        # untouched block absent from the output rather than growing noise.
        return None if value in (None, "none") else value
    return None


def _add_structural_settings(group_name: str, group_output: dict, spec: Parameters) -> None:
    """Add non-type structural settings to a group dict.

    Parameters
    ----------
    group_name : str
        Group name.
    group_output : dict
        The group dict to fill (modified in place).
    spec : Parameters
        The Parameters object.

    Notes
    -----
    Plain settings come from the declarative ``_STRUCTURAL_ROUNDTRIP`` table.
    The dust attenuation block stays hand-written below, because three of its
    settings do not survive a straight attribute read: ``law_neb`` falls back to
    the birth-cloud law when unset and so is emitted only when it was given, the
    per-screen law-parameter overrides are stored in one flattened dict, and
    ``lyman_cutoff`` persists as a float wavelength rather than the boolean the
    grammar takes.
    """
    _emit_declared_structural(group_name, group_output, spec)

    if group_name == "dust_attenuation":
        # Dust laws are now handled in _emit_declared_structural with the new 'law' key.
        # Nebular birth-cloud law (None -> inherits bc; only emit when set).
        if getattr(spec, "dust_law_neb", None) is not None:
            group_output["law_neb"] = spec.dust_law_neb
        # Round-trip per-component law-parameter overrides (slope_bc, delta_diff,
        # slope_neb…).
        from tengri.components.dust.attenuation import TWO_COMPONENT_OVERRIDE_KEYS

        _law_kw_to_short = {v: k for k, v in TWO_COMPONENT_OVERRIDE_KEYS.items()}
        for comp in ("bc", "diff", "neb"):
            for law_kw, value in (getattr(spec, "dust_law_overrides", {}).get(comp) or {}).items():
                short = _law_kw_to_short.get(law_kw)
                if short is not None:
                    group_output[f"{short}_{comp}"] = value
        # Round-trip the Lyman-limit clip back to its boolean grammar form.
        if float(getattr(spec, "dust_lyman_cutoff_aa", 0.0) or 0.0) > 0.0:
            group_output["lyman_cutoff"] = True
        # Round-trip the absorb-all LyC toggle (only emit when non-default).
        if bool(getattr(spec, "dust_lyc_absorb_all", False)):
            group_output["lyc_absorb_all"] = True
        # Round-trip the FSPS-parity energy-balance toggle (non-default only).
        if bool(getattr(spec, "dust_eb_include_lyc", False)):
            group_output["eb_include_lyc"] = True


def _analyze_wildcard_intent(
    param_names: list[str], spec: Parameters, provenance: dict
) -> str | None:
    """Determine if a group can be represented with a wildcard.

    Returns FREE or FIXED if all non-explicit params share the same wildcard
    tag, otherwise returns None (use explicit listing).

    Parameters
    ----------
    param_names : list[str]
        Full parameter names in this group.
    spec : Parameters
        The Parameters object.
    provenance : dict
        The _group_provenance dict (or empty dict).

    Returns
    -------
    str or None
        FREE, FIXED, or None.
    """
    if not provenance:
        # No provenance: don't use wildcard
        return None

    # Collect provenance tags
    tags = set()
    for param_name in param_names:
        tag = _base_provenance(provenance.get(param_name, "registry_default"))
        # Only consider wildcard tags
        if tag in ("wildcard_free", "wildcard_fixed"):
            tags.add(tag)

    # If all params share the same wildcard tag, use it
    if len(tags) == 1:
        tag = tags.pop()
        if tag == "wildcard_free":
            return FREE
        elif tag == "wildcard_fixed":
            return FIXED

    return None


def _get_explicit_overrides(
    param_names: list[str],
    spec: Parameters,
    provenance: dict,
    wildcard_intent: str | None,
) -> dict[str, Distribution]:
    """Extract parameters that should be explicit (not collapsed by wildcard).

    Parameters
    ----------
    param_names : list[str]
        Full parameter names in the group.
    spec : Parameters
        The Parameters object.
    provenance : dict
        The _group_provenance dict.
    wildcard_intent : str or None
        The wildcard intent (FREE, FIXED, or None).

    Returns
    -------
    dict[str, Distribution]
        Mapping of full param name to distribution for explicit listing.
    """
    explicit = {}

    for param_name in param_names:
        # Base tag: a grid-narrowed parameter still came from the wildcard, so
        # it must collapse back into it rather than surface as an override.
        tag = _base_provenance(provenance.get(param_name, "registry_default"))

        # If there's a wildcard intent, exclude params that match it
        if wildcard_intent is not None:
            if wildcard_intent is FREE and tag == "wildcard_free":
                continue
            if wildcard_intent is FIXED and tag == "wildcard_fixed":
                continue

        # Include: per-param overrides, mismatched tags, or defaults
        explicit[param_name] = spec.get_distribution(param_name)

    return explicit
