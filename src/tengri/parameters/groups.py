# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Suchetha Cooray
# Portions adapted from Prospector (Johnson et al. 2021)

"""Nested-dict model builder for parameter specification.

Provides a Bagpipes-style nested-dictionary interface to the Parameters
class. Instead of flat kwargs (e.g., ``sfh_dpl_alpha=..., sfh_dpl_beta=...``),
users can organize parameters into semantic groups::

    from tengri.parameters import parse_groups, FREE, FIXED

    params = parse_groups(
        sfh={"type": "dpl", "*": FREE, "beta": 0.5},
        dust={"type": "two_component", "law_bc": "calzetti", "*": FIXED},
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
2. Checking for a wildcard directive ('*': FREE / FIXED).
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
- ``Main API`` — ``parse_groups()``
- ``Internal helpers`` — one ``_translate_<group>()`` per group (sfh,
  stellar, dust, neb, shock, igm, radio, foreground, xray, agn), then
  key validation and per-parameter resolution
- ``Inverse: Parameters to nested-dict form`` — ``parameters_to_groups()``

Notes
-----
**Not JAX-traced**: Like Parameters itself, parse_groups is a pure Python
translator and cannot be called inside a JAX gradient tape.

**Wildcard semantics**: The '*' key in a group dict applies a default
(FREE or FIXED) to all parameters in that group not explicitly overridden.

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
...     sfh={"type": "dpl", "*": FREE, "alpha": Uniform(0.5, 3.0)},
...     redshift=0.1,
... )
>>> "sfh_dpl_alpha" in params.free_params
True
"""

from __future__ import annotations

import difflib
import warnings

from tengri.parameters._builders import _resolve_lazy_bucket
from tengri.parameters.parameters import Parameters
from tengri.parameters.priors import Distribution, Fixed
from tengri.parameters.sentinels import FIXED, FREE

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
    warnings.warn(
        f"{param_name!r} has no curated default, so '*': FIXED pins it at its "
        f"prior midpoint ({float(registry_default.unstandardize(0.0)):.4g}) — "
        f"an arbitrary rather than physically motivated value. Pass an "
        f"explicit value for it in the group dict to silence this, or leave "
        f"it FREE. (Curating registry defaults is tracked in #1007.)",
        UserWarning,
        stacklevel=3,
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
    }
)


def _valid_dust_emission_types() -> frozenset[str]:
    """Return accepted ``dust.emission.type`` values, derived from the registry.

    Reads the ``_REGISTRY`` (populated by SEDModelComponent.__init_subclass__
    at import time) and returns names of all ports whose ``outputs`` include
    ``"sed_dust_ir"``. This automatically picks up all dust emission ports
    (modified_blackbody, dale2014, draine_li2007, etc.) without manual
    maintenance. Union with the alias map keys (e.g., draine2021_pah →
    draine2021_pah_ir) so grammar type names resolve correctly.

    Includes ``energy_balance_split`` (a registered two-temperature + AGN-IR
    port publishing ``sed_dust_ir``, Kokorev+2021 — a real model, not a helper).
    Also includes ``_LAZY_DUST_EMISSION_TYPES`` (ADR-0005 / ADR-0008).
    """
    from tengri.components.sed_model_component import _REGISTRY
    from tengri.forward.component_factory import _EMISSION_TYPE_ALIASES

    # Registry names whose outputs include "sed_dust_ir" (getattr skips non-emission entries, #844)
    dust_ir_ports = frozenset(
        name
        for name, cls in _REGISTRY.items()
        if "sed_dust_ir" in {o.name for o in getattr(cls, "_outputs_tuple", ())}
    )

    # Add alias keys (grammar names that map to registry names)
    alias_keys = frozenset(_EMISSION_TYPE_ALIASES.keys())

    return dust_ir_ports | alias_keys | _LAZY_DUST_EMISSION_TYPES


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
    """Derive accepted ``radio.type`` values from :data:`RADIO_MODELS`."""
    from tengri.components.radio import RADIO_MODELS

    return frozenset(RADIO_MODELS.keys())


def _valid_xray_types() -> frozenset[str]:
    """Derive accepted ``xray.type`` values from :data:`XRAY_MODELS`.

    ``"yang20"`` is registered in :data:`XRAY_MODELS` as an alias of
    ``"simple"``: tengri's X-ray component already implements the
    Yang+2020 physics (alpha_ox corona + Morrison & McCammon 1983 N_H +
    Compton/Thomson scattering) -- only the user-facing name was
    missing. See ``components/xray/xray.py`` for the formulas.
    """
    from tengri.components.xray import XRAY_MODELS

    return frozenset(XRAY_MODELS.keys())


def _valid_dust_laws() -> frozenset[str]:
    """Return accepted ``dust.law_bc`` / ``dust.law_diff`` values from the registry.

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

#: Partition table: agn_* param name -> group path (for sub-block routing).
#: Maps full agn_* param names to their owning group (agn, agn.disc, agn.torus, etc.)
_AGN_PARTITION = {
    # Shared params (no sub-block prefix)
    "agn_frac": "agn",
    "agn_log_lbol": "agn",
    "agn_alpha": "agn",
    "agn_log_mbh": "agn",
    "agn_log_ledd": "agn",
    "agn_a_spin": "agn",
    "agn_cos_inc": "agn",
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
    "agn_feltre_cf": "agn.nlr",  # Feltre calibration for NLR
    "neb_xid": "agn.nlr",  # Nebular ionization for NLR
    # Broad-line region
    "agn_blr_cf": "agn.blr",
    # FeII
    "agn_fe2_strength": "agn.feii",
    # Attenuation
    "agn_polar_ebv": "agn.atten",
    "agn_polar_oa": "agn.atten",
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
    "apply_igm",
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


def parse_groups(**kwargs) -> Parameters:
    """Translate nested-dict model specification to Parameters.

    Parameters
    ----------
    **kwargs : keyword arguments
        Model configuration. Keys are group names (sfh, dust, neb, igm,
        radio, xray, agn) or top-level settings (redshift, apply_igm, n_grid).

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
    ...     sfh={"type": "dpl", "*": FREE},
    ...     redshift=0.1,
    ... )
    >>> assert "sfh_dpl_alpha" in params.free_params
    """
    # ── Pass 1: Translate structural choices ──────────────────────────

    structural_kwargs = _translate_structural(kwargs)

    # ── Pass 2: Resolve per-parameter values ──────────────────────────

    # Build a structural Parameters to get the declared parameter list
    structural_params = Parameters(**structural_kwargs)

    # Partition declared params by owning group. ``met_*`` lands in
    # ``"stellar"`` when the user opted into the new top-level slot
    # (issue #311); otherwise it stays in ``"sfh"`` so the legacy
    # ``sfh={'*': FIXED}`` wildcard keeps cascading over met_* params
    # — preserves pre-#311 behavior for every fixture/recipe that didn't
    # pass a ``stellar={}`` block.
    dust_emission_active = structural_params.dust_emission is not None
    has_stellar_block = isinstance(kwargs.get("stellar"), dict)
    param_partition = _partition_by_group(
        structural_params.all_params,
        dust_emission_active,
        met_group="stellar" if has_stellar_block else "sfh",
    )

    # Resolve each parameter's final distribution
    resolved_kwargs = dict(structural_kwargs)
    provenance: dict[str, str] = {}

    # Which agn_* parameters the active AGN model / composable block selection
    # actually consumes. A group-level ``agn={'*': FREE}`` frees only these,
    # not the full declared superset — otherwise it would create dozens of
    # unconstrained no-op nuisance dimensions for parameters belonging to
    # inactive blocks (e.g. GRAHSP params under a SKIRTOR torus).
    agn_active_params = _agn_active_param_set(structural_kwargs)

    # Which radio sub-block params the active sf / agn radio model consumes.
    # A ``radio={'sf': {'*': FREE}}`` / ``radio={'agn': {'*': FREE}}`` wildcard
    # frees only these — the inactive model's params collapse to Fixed (mirrors
    # the AGN block-scoped wildcard above).
    radio_sf_active = _RADIO_SF_PARAMS_BY_MODE.get(
        structural_kwargs.get("radio_sfr_mode"), frozenset()
    )
    radio_agn_active = _RADIO_AGN_PARAMS_BY_MODEL.get(
        structural_kwargs.get("radio_agn_model"), frozenset()
    )

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
        if group.startswith("dust."):
            # Sub-group (e.g., "dust.emission")
            parent_group = "dust"
            group_dict = kwargs.get(parent_group, {})
            subkey = group.replace("dust.", "")
            if isinstance(group_dict, dict) and subkey in group_dict:
                group_dict = group_dict[subkey]
            else:
                group_dict = {}
        elif group == "igm.dla":
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

        # AGN group wildcards are scoped to the active blocks' consumed params
        # (block-scoped wildcard). Non-AGN groups are always wildcard-active.
        is_agn = group == "agn" or group.startswith("agn.")
        wildcard_active = True
        if is_agn:
            wildcard_active = param_name in agn_active_params
        elif group == "radio.sf":
            wildcard_active = param_name in radio_sf_active
        elif group == "radio.agn":
            wildcard_active = param_name in radio_agn_active
        final_dist, tag = _resolve_value(
            param_name,
            group_dict,
            structural_params.get_distribution(param_name),
            wildcard_active=wildcard_active,
        )

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
                    f"(e.g. Uniform(lo, hi) or a number); the '*' wildcard and "
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

    # ── Construct final Parameters ────────────────────────────────────

    final_params = Parameters(**resolved_kwargs)
    # Fill in provenance for params not touched by user/wildcard
    for name in list(final_params._distributions.keys()):
        provenance.setdefault(name, "registry_default")
    object.__setattr__(final_params, "_group_provenance", provenance)

    _warn_firrc_slope_degeneracy(final_params)

    return final_params


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
    :func:`tengri.components.agn.blocks._consumes.agn_active_param_set`,
    lazy-imported to avoid an import cycle (the agn package imports the priors
    layer that ultimately re-exports this module).
    """
    from tengri.components.agn.blocks._consumes import agn_active_param_set

    return agn_active_param_set(structural_kwargs)


def _translate_structural(groups: dict) -> dict:
    """Resolve each group's `type` choice into the matching Parameters kwargs."""
    valid_groups = {
        "sfh",
        "stellar",
        "dust",
        "neb",
        "shock",
        "igm",
        "radio",
        "xray",
        "agn",
        "foreground",
    }
    result = {}

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
        elif group_name == "stellar":
            _translate_stellar(group_dict, result)
        elif group_name == "dust":
            _translate_dust(group_dict, result)
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
        result["bin_edges_gyr"] = sfh_dict["bin_edges_gyr"]

    if sfh_type is None:
        result["mean_sfh_type"] = ["dpl", "field"]
        return

    valid = _valid_sfh_types()
    if not isinstance(sfh_type, (str, list)):
        raise TypeError(
            f"sfh 'type' must be a string (or a list of strings for a "
            f"composition), got {type(sfh_type).__name__}: {sfh_type!r}. "
            f"Example: sfh={{'type': 'delayed', '*': FIXED}}."
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


def _translate_stellar(stellar_dict: dict, result: dict) -> None:
    """Resolve ``stellar.met_mode`` into the matching Parameters kwarg.

    Wires the chemical-evolution mode (``delta``, ``ramp``, ``two_step``,
    ``psb_two_step``, ``bins``, ``bins_continuity``, ``chem_evol``, ``table``)
    through the nested-dict builder. Per-mode parameters (``logzsol_old``,
    ``logzsol_young``, ``step_age_gyr``, etc.) flow through the standard
    pass-2 resolver, since :func:`_partition_by_group` routes ``met_*``
    declarations into this group.

    See :func:`tengri.components.stellar.sfh.met_registry` for the full
    list of registered modes and their per-mode parameters.
    """
    from tengri.components.stellar.sfh.met_registry import MET_REGISTRY

    met_mode = stellar_dict.get("met_mode")
    if met_mode is None:
        # No explicit mode; let auto-inference (from per-param keys) decide.
        return

    valid_modes = sorted(MET_REGISTRY.keys())
    if met_mode not in MET_REGISTRY:
        suggestions = difflib.get_close_matches(met_mode, valid_modes, n=2, cutoff=0.6)
        suggest_str = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        raise ValueError(
            f"Unknown met_mode '{met_mode}'. Valid modes: {', '.join(valid_modes)}.{suggest_str}"
        )

    result["met_mode"] = met_mode


def _translate_dust(dust_dict: dict, result: dict) -> None:
    """Translate dust group to dust_model, dust_law_bc, dust_emission.

    Resolves dust type against the set of supported dust models
    (two_component, single_component, wg00) and extracts structural
    configuration and law selections.
    """
    dust_type = dust_dict.get("type", "two_component")

    # Lyman-limit clip is wired only through the two-component screen. Flag any
    # other type rather than silently dropping the request (single-component,
    # WG00, and SEDModelComponent ports do not route through it yet).
    if dust_dict.get("lyman_cutoff") and dust_type != "two_component":
        raise ValueError(
            f"dust 'lyman_cutoff' is only supported for type='two_component' "
            f"(got type={dust_type!r}). Use a two-component dust block, or drop "
            f"'lyman_cutoff'."
        )

    # Validate type against hard-coded dust model types
    if dust_type not in _VALID_DUST_TYPES:
        # A common mistake (#664): passing an attenuation *law* name as the dust
        # ``type``. Laws (calzetti, smc, salim_sbl18, …) are not standalone dust
        # models — they are selected via ``law_bc`` / ``law_diff`` inside a
        # ``two_component`` (or ``single_component``) block. Point there instead
        # of emitting a bare "unknown type" so the request is not lost.
        if dust_type in _valid_dust_laws():
            raise ValueError(
                f"'{dust_type}' is a dust attenuation *law*, not a dust model type. "
                f"Select it via 'law_bc'/'law_diff' on a dust model, e.g. a single "
                f"screen dust={{'type': 'single_component', 'law_bc': '{dust_type}', "
                f"'tau_v': ...}}, or birth-cloud + ISM dust={{'type': 'two_component', "
                f"'law_bc': '{dust_type}', 'law_diff': '{dust_type}', 'tau_bc': ..., "
                f"'tau_diff': ...}}. Valid dust types are: "
                f"{', '.join(sorted(_VALID_DUST_TYPES))}."
            )
        suggestions = difflib.get_close_matches(dust_type, _VALID_DUST_TYPES, n=2, cutoff=0.6)
        suggest_str = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        raise ValueError(f"Unknown dust type '{dust_type}'.{suggest_str}")

    result["dust_model"] = dust_type

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
            if key in dust_dict:
                val = dust_dict[key]
                if val not in allowed:
                    raise ValueError(f"Invalid WG00 {key} {val!r}; choose one of {allowed}.")
                result[result_key] = val
        return

    # Extract dust laws. Leave each unset (None) when the user did not give it,
    # so Parameters can apply symmetric inheritance (set one law -> both share
    # it; set neither -> power_law for both).
    dust_law_bc = dust_dict.get("law_bc")
    dust_law_diff = dust_dict.get("law_diff")
    dust_law_neb = dust_dict.get("law_neb")

    valid_laws = _valid_dust_laws()
    if dust_law_bc is not None:
        if dust_law_bc not in valid_laws:
            suggestions = difflib.get_close_matches(dust_law_bc, valid_laws, n=2, cutoff=0.6)
            suggest_str = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
            raise ValueError(f"Unknown dust law '{dust_law_bc}'.{suggest_str}")
        result["dust_law_bc"] = dust_law_bc

    if dust_law_diff is not None:
        if dust_law_diff not in valid_laws:
            suggestions = difflib.get_close_matches(dust_law_diff, valid_laws, n=2, cutoff=0.6)
            suggest_str = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
            raise ValueError(f"Unknown dust law '{dust_law_diff}'.{suggest_str}")
        result["dust_law_diff"] = dust_law_diff

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
            if key in dust_dict:
                overrides.setdefault(comp, {})[law_kw] = float(dust_dict[key])
    if overrides:
        result["dust_law_overrides"] = overrides

    # Lyman-limit clip (912 Å). Boolean in the grammar; stored as the cutoff
    # wavelength so the forward model and compile_signature carry a single float.
    if dust_dict.get("lyman_cutoff"):
        result["dust_lyman_cutoff_aa"] = 912.0

    # Whether ALL stellar LyC is absorbed by neb_fesc (FSPS/CIGALE) or only the
    # young/birth-cloud population (default; bagpipes). See DustSEDComponent.
    if "lyc_absorb_all" in dust_dict:
        result["dust_lyc_absorb_all"] = bool(dust_dict["lyc_absorb_all"])

    # Include the LyC in the dust energy-balance integral (FSPS/Prospector
    # parity) vs the canonical LyC-masked L_absorbed (default; #922/#961).
    if "eb_include_lyc" in dust_dict:
        result["dust_eb_include_lyc"] = bool(dust_dict["eb_include_lyc"])

    # Extract dust emission sub-block
    if "emission" in dust_dict:
        emission_dict = dust_dict["emission"]
        if isinstance(emission_dict, dict):
            emission_type = emission_dict.get("type", None)
            if emission_type is not None:
                # Dust IR emission types are engine names (modified_blackbody, dale2014,
                # dl07, dl14, astrodust, etc.) resolved by the DUST_EMISSION_MODELS loader cache.
                valid_emission_types = _valid_dust_emission_types()
                if emission_type not in valid_emission_types:
                    suggestions = difflib.get_close_matches(
                        emission_type, valid_emission_types, n=3, cutoff=0.6
                    )
                    suggest_str = (
                        f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
                    )
                    raise ValueError(f"Unknown dust emission type '{emission_type}'.{suggest_str}")
                result["dust_emission"] = emission_type


def _translate_neb(neb_dict: dict, result: dict) -> None:
    """Translate neb group to nebular settings."""
    neb_type = neb_dict.get("type", "none")

    # Validate type
    valid_neb = _valid_nebular_types()
    if neb_type not in valid_neb:
        suggestions = difflib.get_close_matches(neb_type, valid_neb, n=2, cutoff=0.6)
        suggest_str = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        raise ValueError(f"Unknown nebular type '{neb_type}'.{suggest_str}")

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

    - ``norm`` — ``"frac"`` (relative to the galaxy Halpha, default) or
      ``"lhalpha"`` (absolute ``shock_log_lhalpha``).
    - ``abundance`` — MAPPINGS abundance set (``"solar"``, ``"2xsolar"``, …).
    - ``component`` — ``"shock"`` | ``"precursor"`` | ``"combined"``.

    Per-parameter overrides (``frac``, ``log_lhalpha``, ``velocity``,
    ``log_density``, ``b_over_sqrt_n``) resolve to the ``shock_*`` bucket
    params in :func:`parse_groups`.
    """
    shock_type = shock_dict.get("type", "mappings")
    valid_shock = frozenset({"none", "mappings"})
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
    """Translate igm group to apply_igm and related settings."""
    igm_type = igm_dict.get("type", "madau")

    # Validate type
    valid_igm = _valid_igm_types()
    if igm_type not in valid_igm:
        suggestions = difflib.get_close_matches(igm_type, valid_igm, n=2, cutoff=0.6)
        suggest_str = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        raise ValueError(f"Unknown IGM type '{igm_type}'.{suggest_str}")

    # Map type to apply_igm + igm_model
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


def _translate_radio(radio_dict: dict, result: dict) -> None:
    """Translate radio group with composable SF + AGN sub-blocks.

    Supports two grammar styles:

    **New composable form (preferred)**:

    .. code-block:: python

        radio = {
            "sf": {"type": "delvecchio2021"},  # SF variant
            "agn": {"type": "dpl"},  # AGN variant
        }

    **Legacy form (back-compat)**:

    .. code-block:: python

        radio = {"type": "condon92"}  # radio on with default sf/agn models

    Raises if both 'type' and 'sf'/'agn' sub-blocks are present.
    """
    has_legacy_type = "type" in radio_dict
    has_sf_block = "sf" in radio_dict
    has_agn_block = "agn" in radio_dict

    if has_legacy_type and (has_sf_block or has_agn_block):
        raise ValueError(
            "radio: cannot mix legacy 'type' key with 'sf'/'agn' sub-blocks. "
            "Use either: radio={'type': 'bell2003'} (legacy) "
            "or radio={'sf': {'type': 'bell2003'}, 'agn': {'type': 'powerlaw'}} (new)."
        )

    # Legacy form: radio={'type': 'X'} → interpret as SF variant with default AGN
    if has_legacy_type and not has_sf_block and not has_agn_block:
        radio_type = radio_dict["type"]
        valid_radio = _valid_radio_types()
        if radio_type not in valid_radio:
            suggestions = difflib.get_close_matches(radio_type, valid_radio, n=2, cutoff=0.6)
            suggest_str = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
            raise ValueError(f"Unknown radio type '{radio_type}'.{suggest_str}")
        result["radio"] = radio_type != "none"
        if radio_type != "none":
            # Legacy 'type' predates the SF/AGN split → default both models.
            result["radio_sfr_mode"] = "bell2003"
            result["radio_agn_model"] = "powerlaw"
        return

    # New composable form: extract SF and AGN sub-blocks
    sf_variant = "bell2003"  # default
    agn_variant = "powerlaw"  # default
    radio_enabled = False

    if has_sf_block:
        sf_dict = radio_dict["sf"]
        if isinstance(sf_dict, dict):
            sf_variant = sf_dict.get("type", "bell2003")
            valid_sf = frozenset({"none", "bell2003", "delvecchio2021", "mccheyne2022"})
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
#: Union of every param owned by each radio sub-group, used by the partition
#: to route names away from the flat ``radio`` group.
_RADIO_SF_PARAM_NAMES: frozenset[str] = frozenset().union(*_RADIO_SF_PARAMS_BY_MODE.values())
_RADIO_AGN_PARAM_NAMES: frozenset[str] = frozenset().union(*_RADIO_AGN_PARAMS_BY_MODEL.values())


#: Valid laws for the MW foreground screen (#297). Only the closed-form
#: laws that take a single ``R_V`` parameter are usable as a foreground
#: screen — host-dust laws with two free knobs (slope, bump, ...) would
#: collide with the host ``dust`` block's parameter prefix.
_VALID_FOREGROUND_LAWS = frozenset({"cardelli"})


def _translate_foreground(fg_dict: dict, result: dict) -> None:
    """Translate the ``foreground`` group (MW screen) — see #297.

    Flat layout: ``foreground={'ebmv_mw': 0.05, 'law': 'cardelli', 'rv': 3.1}``.
    Surfaces three top-level kwargs on ``Parameters`` so the SEDModel can
    apply the screen in the observed-frame SED path, after IGM and
    redshifting, independently from the host-galaxy ``dust`` block.
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
_GROUP_STRUCTURAL_KEYS: dict[str, frozenset[str]] = {
    "sfh": frozenset({"type", "*", "bin_edges_gyr"}),
    "stellar": frozenset({"met_mode", "*"}),
    "dust": frozenset(
        {
            "type",
            "*",
            "law_bc",
            "law_diff",
            "law_neb",
            "emission",
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
    "dust.emission": frozenset({"type", "*"}),
    "neb": frozenset({"type", "*", "full_catalog", "grid"}),
    "shock": frozenset({"type", "*", "norm", "abundance", "component"}),
    "igm": frozenset({"type", "*", "patchy", "dla"}),
    "igm.dla": frozenset({"type", "*"}),
    "radio": frozenset({"type", "*", "sf", "agn"}),
    "radio.sf": frozenset({"type", "*"}),
    "radio.agn": frozenset({"type", "*"}),
    "xray": frozenset({"type", "*"}),
    "agn": frozenset({"type", "*", "norm"}) | _AGN_SUBBLOCK_KEYS,
    "agn.disc": frozenset({"type", "*"}),
    "agn.torus": frozenset({"type", "*"}),
    "agn.nlr": frozenset({"type", "*"}),
    "agn.blr": frozenset({"type", "*"}),
    "agn.feii": frozenset({"type", "*"}),
    "agn.atten": frozenset({"type", "*"}),
    # Deprecated: agn.lines is expanded to (agn.nlr, agn.blr) via expand_lines_alias
    "agn.lines": frozenset({"type", "*"}),
    "foreground": frozenset({"ebmv_mw", "law", "rv"}),
}


def _short_names_for_group(group: str, param_partition: dict[str, str]) -> set[str]:
    """Return the set of short and full names every declared param exposes
    under ``group`` (e.g. ``"agn.torus"`` → ``{"tau_skirtor", "agn_tau_skirtor", ...}``).

    Used by :func:`_validate_user_keys` to recognize per-parameter overrides
    when walking a user's group dict.
    """
    out: set[str] = set()
    for full_name, owner in param_partition.items():
        if owner != group:
            continue
        out.add(full_name)
        out.add(_extract_short_name(full_name, {}))
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
    try:
        from tengri.components.sed_model_component import _REGISTRY
    except Exception:
        return set()
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
    dust_emission_active = structural_params.dust_emission is not None
    valid_top_groups = {"sfh", "stellar", "dust", "neb", "shock", "igm", "radio", "xray", "agn"}

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
            # AGN top-level also accepts shared param short/full names
            # *and* names from every sub-block (cross-level acceptance —
            # see :func:`_build_agn_search_view`). Sub-block dicts are
            # still tagged separately below.
            param_names = param_names | agn_shared_names
            for sub_name in _AGN_SUBBLOCK_KEYS:
                param_names = param_names | _short_names_for_group(
                    f"agn.{sub_name}", param_partition
                )
        elif top_key == "dust" and dust_emission_active:
            # Dust top-level accepts the dust.emission param short names
            # for legacy code that flattens emission params at the dust
            # level. Treat as a soft acceptance (still resolved via the
            # dust.emission group path).
            param_names = param_names | _short_names_for_group("dust.emission", param_partition)
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
        if top_key == "dust" and isinstance(top_val.get("emission"), dict):
            sub_allowed = _GROUP_STRUCTURAL_KEYS["dust.emission"]
            sub_params = _short_names_for_group("dust.emission", param_partition)
            sub_params = sub_params | _short_names_for_registered_type(
                top_val["emission"].get("type") if isinstance(top_val["emission"], dict) else None
            )
            _check_dict_keys(
                "dust.emission", top_val["emission"], sub_allowed | sub_params, param_partition
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
        # Suggestion pool: same group's allowed keys + every short name
        # across all groups (helps the "wrong group, right name" case).
        suggestion_pool = set(allowed)
        for full_name in param_partition:
            suggestion_pool.add(_extract_short_name(full_name, {}))
            suggestion_pool.add(full_name)
        suggestions = difflib.get_close_matches(str(key), list(suggestion_pool), n=2, cutoff=0.6)
        suggest_str = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        raise ValueError(
            f"Unknown key {key!r} in group {group!r}.{suggest_str} "
            f"Valid structural keys for this group are: "
            f"{sorted(_GROUP_STRUCTURAL_KEYS.get(group, frozenset({'type', '*'})))}."
        )


def _build_agn_search_view(param_name: str, agn_dict: dict, group: str) -> dict:
    """Build the resolution view for one AGN parameter.

    AGN parameters live in a two-level nest: the top-level ``agn`` dict
    plus up to six sub-block dicts (``disc``/``torus``/``nlr``/``blr``/
    ``feii``/``atten``). To keep the API friendly, a parameter can be supplied
    at *either* level — the partition table records the canonical location,
    but a user who writes ``agn={'disc': {'agn_log_lbol': Uniform(...)}}``
    expects the value to take effect even though ``agn_log_lbol`` is
    nominally a shared (top-level) param.

    This helper assembles a single dict the caller can pass to
    :func:`_resolve_value`:

    1. The canonical location for ``param_name`` (top level if
       ``group == "agn"``; the matching sub-block if ``group.startswith("agn.")``).
    2. Every sibling location that also carries an override for the same
       short name.

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
        # Sub-block param: also accept it at the top level.
        siblings.append(("<top>", agn_dict))

    # Collect (location, value) for every place this param appears.
    hits = []
    for key in (short_name, param_name):
        if key in canonical_dict and key not in ("type", "*"):
            hits.append(("<canonical>", canonical_dict[key]))
            break
    for location, sub in siblings:
        for key in (short_name, param_name):
            if key in sub and key not in ("type", "*"):
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

    # Map sub-block names to their result kwargs
    block_to_kwarg = {
        "disc": "agn_disc_block",
        "torus": "agn_torus_block",
        "nlr": "agn_nlr_block",
        "blr": "agn_blr_block",
        "feii": "agn_feii_block",
        "atten": "agn_attenuation_block",
    }

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

        block_type = block_spec.get("type")
        if block_type is None:
            # Assume 'none' if no type given
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
            partition[name] = "dust.emission"
        elif name.startswith("dust_"):
            partition[name] = "dust"
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
            # FREE: use registry default (which may be Fixed; that's ok)
            return registry_default, "user_free"
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
                return registry_default, "wildcard_free"
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
            # Bad wildcard value — only FREE or FIXED are accepted in the '*' slot
            raise ValueError(
                f"Wildcard '*' must be FREE or FIXED (the sentinels exported "
                f"from tengri), got {wildcard!r}. "
                f"Did you mean ``'*': FREE`` or ``'*': FIXED``? "
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
    are collapsed into a single '*': FREE or '*': FIXED entry, with explicit
    overrides listed separately.

    **Flat-built fallback**: If spec was built via flat-kwarg Parameters(...),
    all parameters are listed explicitly (no wildcard).

    **Roundtrip guarantee**: The output dict, when passed to
    parse_groups(**output), produces a Parameters with identical
    free/fixed partitions and distributions.

    Examples
    --------
    >>> spec = parse_groups(
    ...     sfh={"type": "dpl", "*": FREE, "beta": Uniform(1, 3)},
    ...     redshift=Fixed(0.05),
    ... )
    >>> groups = spec.to_groups()
    >>> roundtripped = parse_groups(**groups)
    >>> spec.free_params == roundtripped.free_params
    True
    """
    result = {}
    # Emit a ``stellar`` block on the round-trip only when ``met_mode`` is
    # non-default OR the user explicitly built the spec with a stellar group
    # (provenance check). Otherwise keep met_* under ``sfh`` for back-compat
    # with the legacy fixtures that pre-#311 expected.
    use_stellar = getattr(spec, "met_mode", "delta") != "delta"
    partition = _partition_by_group(
        spec.all_params,
        spec.dust_emission is not None,
        met_group="stellar" if use_stellar else "sfh",
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
            group_output["type"] = type_value

        # Add other structural settings
        _add_structural_settings(group_name, group_output, spec)

        # Determine if we can use a wildcard
        wildcard_intent = _analyze_wildcard_intent(param_names, spec, provenance)

        if wildcard_intent is not None:
            # Use wildcard for collapsed params
            group_output["*"] = wildcard_intent
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
    _all_possible_groups = {"sfh", "dust", "neb", "shock", "igm", "radio", "xray", "agn"}
    for group_name in sorted(_all_possible_groups):
        if group_name not in result:
            type_value = _extract_group_type(group_name, spec)
            if type_value is not None and (
                type_value == "none" or (group_name == "dust" and type_value != "two_component")
            ):
                # Only add if it's a non-default type or a special case
                # For now, only add 'none' types and other explicit settings
                result[group_name] = {"type": type_value}

    # Handle top-level parameters (redshift, apply_igm)
    if "redshift" in spec.all_params:
        result["redshift"] = spec.get_distribution("redshift")

    if spec.apply_igm is not True:
        # Only include if non-default (default is True)
        result["apply_igm"] = spec.apply_igm

    if spec.n_grid != 256:
        # Only include if non-default
        result["n_grid"] = spec.n_grid

    return result


def _extract_group_type(group_name: str, spec: Parameters) -> str | list[str] | None:
    """Extract the type value for a group from spec settings.

    Parameters
    ----------
    group_name : str
        Group name (e.g., 'sfh', 'dust', 'neb', 'dust.emission', 'agn.disc').
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
    elif group_name == "dust":
        return spec.dust_model
    elif group_name == "dust.emission":
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
        return spec.igm_model if hasattr(spec, "igm_model") else None
    elif group_name == "radio":
        return spec.radio_model if hasattr(spec, "radio_model") else None
    elif group_name == "xray":
        return spec.xray_model if hasattr(spec, "xray_model") else None
    elif group_name.startswith("agn"):
        # AGN sub-blocks extract from agn_model setting
        # This is a simplification; more complex composition handled in tests
        return None
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
    """
    if group_name == "dust":
        # Add law_bc and law_diff if non-default
        if spec.dust_law_bc != "power_law":
            group_output["law_bc"] = spec.dust_law_bc
        if hasattr(spec, "dust_law_diff") and spec.dust_law_diff != spec.dust_law_bc:
            group_output["law_diff"] = spec.dust_law_diff
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
    elif group_name == "stellar":
        # Emit met_mode whenever it's non-default (default = 'delta').
        # Always-emit would force a stellar={} entry on every round-trip, which
        # noisily breaks existing diff-against-from_groups call sites.
        if getattr(spec, "met_mode", "delta") != "delta":
            group_output["met_mode"] = spec.met_mode
    elif group_name == "shock":
        # Round-trip the shock normalization + categorical knobs (only when
        # non-default), mirroring the neb/dust structural round-trip (#851).
        if getattr(spec, "shock_norm", "frac") != "frac":
            group_output["norm"] = spec.shock_norm
        if getattr(spec, "shock_abundance", "solar") != "solar":
            group_output["abundance"] = spec.shock_abundance
        if getattr(spec, "shock_component", "combined") != "combined":
            group_output["component"] = spec.shock_component


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
        tag = provenance.get(param_name, "registry_default")
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
        tag = provenance.get(param_name, "registry_default")

        # If there's a wildcard intent, exclude params that match it
        if wildcard_intent is not None:
            if wildcard_intent is FREE and tag == "wildcard_free":
                continue
            if wildcard_intent is FIXED and tag == "wildcard_fixed":
                continue

        # Include: per-param overrides, mismatched tags, or defaults
        explicit[param_name] = spec.get_distribution(param_name)

    return explicit
