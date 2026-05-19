"""Parameter translation between public and internal names.

Naming Conventions
------------------
Metallicity:
    Public:   met_logzsol     = log10(Z/Zsun) (relative to solar)
    Internal: log_z_abs       = log10(Z) (absolute)
    Offset:   LOG10_ZSUN = -1.8477 (Asplund 2009)

Dust:
    Public:   dust_tau_bc, dust_tau_diff, dust_slope
    Internal: tau_bc, tau_diff, dust_slope

PSD:
    Public:   sfh_field_psd_sigma, sfh_field_psd_tau_myr
    Internal: psd_sigma, psd_tau_yr (years, not Myr)

Ages:
    ssp_log_ages_yr = log10(age/yr) on SSP grid
    ssp_lg_age_gyr  = log10(age/Gyr) from DSPS (conversion: + 9.0)
    log_age_grid    = log10(age/yr) on GP grid

Function Prefixes:
    predict_*  = public methods returning observable quantities
    _compute_* = internal computation steps
    build_*    = factory functions creating JIT kernels
    load_*     = file I/O
    get_*      = retrieve from registry
    register_* = add to registry
"""

from __future__ import annotations

import warnings

from tengri.parameters._aliases import (
    _REVERSE_ALIASES,
    find_short_param,
)
from tengri.parameters._builders import _resolve_lazy_bucket

# ── Constants ─────────────────────────────────────────────────────

# Solar metallicity: log10(Zsun) = log10(0.0142) ≈ -1.848 (Asplund 2009)
LOG10_ZSUN = -1.8477116556169435

# ── Parameter maps: public → (internal, unit_scale, offset) ───────

_EVOLVING_MET_PARAM_MAP = {
    "met_logzsol_0": ("log_z_abs_initial", 1.0, LOG10_ZSUN),  # log(Z/Zsun) → log(Z)
    "met_logzsol_final": ("log_z_abs_final", 1.0, LOG10_ZSUN),
}

_EVOLVING_ALPHA_PARAM_MAP = {
    "met_alpha_fe_old": ("alpha_fe_old", 1.0, 0.0),  # [alpha/Fe] of oldest stars
    "met_alpha_fe_young": ("alpha_fe_young", 1.0, 0.0),  # [alpha/Fe] at present day
}

# ── Identity param lists ──────────────────────────────────────────
#
# After Step B (ADR-deepening 2026-05-18) ``_build_param_map`` reads
# identity entries directly from the parameter registry (ADR-0008), so
# the six per-domain ``_*_IDENTITY_PARAMS`` lists previously published
# here (``_DUST_EMISSION_IDENTITY_PARAMS`` etc.) are gone. The two Cue
# lists below stay because :class:`SEDModel._init_nebular` consults
# them at construction time to register only the Cue free params the
# user actually opted into via the spec — that's a deliberate filter
# the auto-derive can't express.
#
# Exception list: any name in ``_PARAMS_WITH_UNIT_CONVERSION`` is filtered
# out of the identity lists because it goes through a real unit-converting
# entry in ``_NON_SFH_PARAM_MAP`` (or one of the other explicit maps) and
# must NOT also receive a passthrough identity entry.
_PARAMS_WITH_UNIT_CONVERSION: frozenset[str] = frozenset(
    {
        # ── Stellar/dust handled in _NON_SFH_PARAM_MAP ────────────
        "met_logzsol",  # Z/Zsun → log(Z), LOG10_ZSUN offset
        "met_alpha_fe",
        "dust_tau_bc",  # rename to internal `tau_bc`
        "dust_tau_diff",  # rename to internal `tau_diff`
        "dust_slope",
        "redshift",
        "noise_frac_cal",
        "dust_f_obscuration",  # rename to internal `f_obscuration`
        "dust_bump_strength",
        "dust_delta",
        "dust_Rv",
        "noise_dof",
        "sigma_v_kms",
        # ── Nebular gas metallicity ───────────────────────────────
        # Declared in _param_defs as log10(Z_gas/Zsun); consumed in
        # ``components/nebular/{cloudy_cb19,feltre_precompute,
        # mappings_photo_precompute}.py`` as absolute log10(Z).
        # Conversion lives in _NON_SFH_PARAM_MAP below.
        "neb_logZ_gas",
    }
)


def _identity_params_from_bucket(bucket: dict) -> list[str]:
    """Derive identity-mapping param list from a `_param_defs.py` bucket.

    Retained only for the two Cue conditional lists below — every other
    caller went away in Step B (ADR-deepening 2026-05-18).
    """
    return sorted(name for name in bucket if name not in _PARAMS_WITH_UNIT_CONVERSION)


_CUE_GAS_IDENTITY_PARAMS = _identity_params_from_bucket(
    _resolve_lazy_bucket("_CUE_GAS_EXTRA_PARAMS")
)
_CUE_IONSPEC_IDENTITY_PARAMS = _identity_params_from_bucket(
    _resolve_lazy_bucket("_CUE_IONSPEC_PARAMS")
)


# ── Non-SFH param map (includes real unit conversions) ───────────────

_NON_SFH_PARAM_MAP = {
    "met_logzsol": ("log_z_abs", 1.0, LOG10_ZSUN),  # log(Z/Zsun) → log(Z)
    "met_alpha_fe": ("alpha_fe", 1.0, 0.0),  # [alpha/Fe] in dex (global)
    "dust_tau_bc": ("tau_bc", 1.0, 0.0),
    "dust_tau_diff": ("tau_diff", 1.0, 0.0),
    "dust_slope": ("dust_slope", 1.0, 0.0),
    "redshift": ("redshift", 1.0, 0.0),
    "noise_frac_cal": ("noise_frac_cal", 1.0, 0.0),
    "dust_f_obscuration": ("f_obscuration", 1.0, 0.0),
    "dust_bump_strength": ("dust_bump_strength", 1.0, 0.0),
    "dust_delta": ("dust_delta", 1.0, 0.0),
    "dust_Rv": ("dust_Rv", 1.0, 0.0),
    "noise_dof": ("noise_dof", 1.0, 0.0),
    "sigma_v_kms": ("sigma_v_kms", 1.0, 0.0),
    # neb_logZ_gas is declared as log10(Z_gas/Zsun) in _param_defs but
    # consumed downstream as absolute log10(Z); LOG10_ZSUN bridges the two
    # (same convention as met_logzsol → log_z_abs).
    "neb_logZ_gas": ("neb_logZ_gas", 1.0, LOG10_ZSUN),
}

# (Reverse alias map now managed in _aliases.py — imported at top)

# ── High-level API: short name → full prefixed name ──────────────

# sfh_type token → {short_name: full_prefixed_name}
# Used by Model.from_config() to expand user-supplied short priors.
_SFH_SHORT_NAMES: dict[str, dict[str, str]] = {
    "tsnorm": {
        "log_peak_sfr": "sfh_tsnorm_log_peak_sfr",
        "peak_lbt_gyr": "sfh_tsnorm_peak_lbt_gyr",
        "width_gyr": "sfh_tsnorm_width_gyr",
        "skew": "sfh_tsnorm_skew",
        "trunc": "sfh_tsnorm_trunc",
    },
    "snorm": {
        "log_peak_sfr": "sfh_snorm_log_peak_sfr",
        "peak_lbt_gyr": "sfh_snorm_peak_lbt_gyr",
        "width_gyr": "sfh_snorm_width_gyr",
        "skew": "sfh_snorm_skew",
    },
    "lnorm": {
        "log_peak_sfr": "sfh_lnorm_log_peak_sfr",
        "peak_lbt_gyr": "sfh_lnorm_peak_lbt_gyr",
        "width_gyr": "sfh_lnorm_width_gyr",
    },
    "dpl": {
        "alpha": "sfh_dpl_alpha",
        "beta": "sfh_dpl_beta",
        "log_peak_sfr": "sfh_dpl_log_peak_sfr",
        "tau_gyr": "sfh_dpl_tau_gyr",
    },
    "delayed": {
        "tau_gyr": "sfh_delayed_tau_gyr",
        "log_peak_sfr": "sfh_delayed_log_peak_sfr",
    },
    # "field" additions apply to any sfh that includes "+field"
    "field": {
        "psd_sigma": "sfh_field_psd_sigma",
        "psd_tau_myr": "sfh_field_psd_tau_myr",
    },
    "dense_basis": {
        "log_total_mass": "sfh_db_log_total_mass",
        "log_sfr_inst": "sfh_db_log_sfr_inst",
        "tx_frac_0": "sfh_db_tx_frac_0",
        "tx_frac_1": "sfh_db_tx_frac_1",
        "tx_frac_2": "sfh_db_tx_frac_2",
    },
    "db": {
        "log_total_mass": "sfh_db_log_total_mass",
        "log_sfr_inst": "sfh_db_log_sfr_inst",
        "tx_frac_0": "sfh_db_tx_frac_0",
        "tx_frac_1": "sfh_db_tx_frac_1",
        "tx_frac_2": "sfh_db_tx_frac_2",
    },
}

# Universal short names valid for any SFH type
_UNIVERSAL_SHORT_NAMES: dict[str, str] = {
    "logzsol": "met_logzsol",
    "tau_bc": "dust_tau_bc",
    "tau_diff": "dust_tau_diff",
}


def resolve_short_names(sfh_type: str | list[str], priors: dict) -> dict:
    """Expand short parameter names to full prefixed names.

    Parameters
    ----------
    sfh_type : str or list of str
        SFH type tokens, e.g. ``"tsnorm"`` or ``["dpl", "field"]``.
        Determines which short names are valid.
    priors : dict
        User-supplied prior dict, may contain short names like ``"log_peak_sfr"``
        or full names like ``"sfh_tsnorm_log_peak_sfr"``. Full names pass through
        unchanged.

    Returns
    -------
    dict
        New dict with all short names expanded to full prefixed names.
        Unknown keys that are neither short nor full names pass through unchanged.

    Examples
    --------
    >>> resolve_short_names("tsnorm", {"log_peak_sfr": Uniform(-1, 2.5)})
    {"sfh_tsnorm_log_peak_sfr": Uniform(-1, 2.5)}
    """
    if isinstance(sfh_type, str):
        tokens = [t.strip() for t in sfh_type.replace("+", " ").split()]
    else:
        tokens = list(sfh_type)

    # Build combined short→full map for this sfh_type
    short_map: dict[str, str] = {}
    for token in tokens:
        if token in _SFH_SHORT_NAMES:
            short_map.update(_SFH_SHORT_NAMES[token])
    short_map.update(_UNIVERSAL_SHORT_NAMES)

    expanded: dict = {}
    for key, val in priors.items():
        if key in short_map:
            expanded[short_map[key]] = val
        else:
            # Assume it's already a full name (pass through)
            expanded[key] = val

    return expanded


# Module-level PARAM_MAP exported for tests and external tooling
PARAM_MAP = {
    "sfh_alpha": ("alpha", 1.0, 0.0),
    "sfh_beta": ("beta", 1.0, 0.0),
    "sfh_tau_peak_gyr": ("tau_sfh", 1e9, 0.0),
    "sfh_peak_sfr": ("sfr_norm", 1.0, 0.0),
    "psd_sigma": ("psd_sigma", 1.0, 0.0),
    "psd_tau_myr": ("psd_tau_yr", 1e6, 0.0),
    "met_logzsol": ("log_z_abs", 1.0, LOG10_ZSUN),
    "dust_tau_bc": ("tau_bc", 1.0, 0.0),
    "dust_tau_diff": ("tau_diff", 1.0, 0.0),
    "dust_slope": ("dust_slope", 1.0, 0.0),
    "redshift": ("redshift", 1.0, 0.0),
}

# ── Factory ────────────────────────────────────────────────────────


_SINGLE_COMPONENT_DUST_PARAM_MAP = {
    "dust_tau_v": ("tau_v", 1.0, 0.0),
}


def _build_param_map(mean_sfh_type, dust_model="two_component"):
    """Build complete param map from SFH registry + non-SFH params + auto-derived components.

    Parameters
    ----------
    mean_sfh_type : list[str]
        SFH type tokens, e.g. ``["tsnorm"]`` or ``["tsnorm", "field"]``.
    dust_model : str
        ``"two_component"`` or ``"single_component"``.

    Returns
    -------
    dict
        public_name -> (internal_name, scale, offset)

    Notes
    -----
    As of Phase II-2, registered SEDComponent instances auto-declare parameters
    via :meth:`declared_parameters`. This function auto-derives identity mappings
    from those declarations (if not already in the map) to keep the param_map
    synchronized without manual editing.

    Identity entries come from ``tengri.parameters.registry.as_param_map``
    (ADR-0008) — walks every ``components/*/_params.py`` directly. Manual
    entries above (``resolve_sfh``, ``_NON_SFH_PARAM_MAP``, etc.) always
    take precedence because they carry unit conversions that identity
    mapping would otherwise silently break.
    """
    from tengri.components.stellar.sfh.registry import resolve_sfh

    _, _, sfh_param_map, _ = resolve_sfh(mean_sfh_type)
    result = dict(sfh_param_map)
    if dust_model == "single_component":
        # Skip tau_bc/tau_diff, add tau_v
        for k, v in _NON_SFH_PARAM_MAP.items():
            if k not in ("dust_tau_bc", "dust_tau_diff"):
                result[k] = v
        result.update(_SINGLE_COMPONENT_DUST_PARAM_MAP)
    else:
        result.update(_NON_SFH_PARAM_MAP)

    # Auto-derive identity entries from the parameter registry (ADR-0008).
    # The registry walks every ``components/*/_params.py`` directly and reads
    # the static ``ParamDeclaration`` tuples — it does not instantiate
    # components with default configs, so we get the full declared-parameter
    # universe regardless of which variant a default ``comp_cls()`` would
    # have picked. This means the registry is strictly safer than the older
    # ``_get_registered_components()``-based auto-derive that needed a
    # ``_SKIP_AUTO_DERIVE`` list (Stellar/DustAttenuation/DustEmission)
    # because their default-config ``declared_parameters()`` would have
    # injected parameters from the wrong variant.
    #
    # Manual entries above (``resolve_sfh``, ``_NON_SFH_PARAM_MAP``, etc.)
    # always take precedence — they carry unit conversions that identity
    # mapping would otherwise silently break.
    try:
        from tengri.parameters.registry import as_param_map as _registry_as_param_map

        for pub_name, (internal, scale, offset, _units) in _registry_as_param_map().items():
            if pub_name not in result:
                result[pub_name] = (internal, scale, offset)
    except ImportError:
        # registry module unavailable (very early bootstrap)
        pass

    return result


# ── Translation functions ──────────────────────────────────────────


# (find_short_param moved to _aliases.py — imported at top)


def get_internal_params(params, param_map, spec, has_field, *, strict_unknown_params: bool = True):
    """Translate a public parameter dict to internal names with unit conversion.

    Conversion applied element-wise: ``internal = public * scale + offset``.

    Also accepts short-form parameter names (sfh_alpha, psd_sigma, etc.) via
    reverse alias lookup, and fills in fixed values from the spec when a
    parameter is absent from ``params``.

    Parameters
    ----------
    params : dict
        Public parameter dict, e.g. from ``spec.sample(key)``.
    param_map : dict
        Mapping ``public_name -> (internal_name, scale, offset)``, as built
        by ``_build_param_map``.
    spec : Parameters
        The parameter specification (used to look up fixed defaults).
    has_field : bool
        Whether the model uses a stochastic field component. When ``True``
        the latent vector ``xi`` is passed through from ``params``.
    strict_unknown_params : bool, optional
        When ``True`` (default), raise :class:`ValueError` if ``params`` contains
        keys that aren't in ``param_map``, the reverse-alias map, or the latent
        vector slots. When ``False``, emit a :class:`UserWarning` instead — used
        by JIT kernel call sites to avoid double-flagging keys already validated
        by the outer ``Model._get_internal_params`` entry point.

    Returns
    -------
    dict
        Internal parameter dict ready for the low-level forward model.

    Raises
    ------
    KeyError
        If a free parameter is absent from ``params`` and not in ``spec``.
    ValueError
        If ``strict_unknown_params`` is ``True`` and ``params`` contains
        unrecognized keys (typos, deleted params, or params from a component
        that isn't active in this spec).
    """
    internal = {}
    for pub_name, (int_name, scale, offset) in param_map.items():
        if pub_name in params:
            value = params[pub_name]
            # String-typed Fixed params (e.g. shock_abundance="solar") are config
            # enums, not numeric values — pass them through verbatim so downstream
            # code that branches on the string still sees it. Numeric params get
            # the standard scale/offset translation.
            if isinstance(value, str):
                internal[int_name] = value
            else:
                internal[int_name] = value * scale + offset
        else:
            # Check short-form alias: find short name that maps to pub_name
            alias_val = find_short_param(params, pub_name)
            if alias_val is not None:
                if isinstance(alias_val, str):
                    internal[int_name] = alias_val
                else:
                    internal[int_name] = alias_val * scale + offset
            else:
                # Fall back to fixed value from spec, or skip if absent.
                #
                # ``param_map`` is built from a registry that may include
                # auto-derived entries from Phase II SEDComponents (AGN,
                # Radio, IGM, X-ray) regardless of whether the active
                # spec actually uses them. When a spec doesn't use a
                # given component, its parameters are absent from both
                # ``params`` and ``spec`` — silently skipping the entry
                # is correct (the param has no internal use either, since
                # the SEDModel doesn't dispatch to that component). Free
                # params that ARE in spec but missing from params still
                # fail loudly.
                try:
                    dist = spec.get_distribution(pub_name)
                except KeyError:
                    # Param isn't in spec at all → treat as inactive,
                    # skip silently. Auto-derived AGN/Radio/IGM/X-ray
                    # entries fall here when the user opts out of those
                    # components by simply not setting their params.
                    continue
                if dist.is_fixed:
                    fixed_val = dist.bounds[0]
                    if isinstance(fixed_val, str) or fixed_val is None:
                        # String-typed Fixed config (e.g. shock_abundance="solar"):
                        # bounds[0] is the literal value, not a numeric range. Pass
                        # through verbatim — downstream code branches on the string.
                        # None handles enum-typed Fixed where bounds isn't populated.
                        internal[int_name] = fixed_val if fixed_val is not None else dist.value
                    else:
                        internal[int_name] = fixed_val * scale + offset
                else:
                    raise KeyError(f"Free parameter '{pub_name}' not found in params dict")

    # Handle field latent vector (both full and short names)
    if has_field:
        if "sfh_field_xi" in params:
            internal["xi"] = params["sfh_field_xi"]
        elif "psd_xi" in params:
            internal["xi"] = params["psd_xi"]

    # Warn about unrecognized keys (silent bugs when wrong names used)
    recognized = set(param_map.keys())
    recognized.update(_REVERSE_ALIASES.keys())
    recognized.update({"sfh_field_xi", "psd_xi"})
    # Array-data inputs consumed by ``forward/pipeline.py`` directly (not via
    # the scalar param-map): tabulated SFH (``sfh_t_gyr`` + ``sfh_sfr``) and
    # tabulated metallicity history (``met_history``) for chem-evolution runs.
    recognized.update({"sfh_t_gyr", "sfh_sfr", "met_history"})
    # Also recognize internal names (for backwards compat)
    recognized.update(int_name for _, (int_name, _, _) in param_map.items())
    unrecognized = set(params.keys()) - recognized
    if unrecognized:
        msg = (
            f"Unrecognized parameter names passed to Model: {sorted(unrecognized)}. "
            f"Did you mean one of: {sorted(param_map.keys())}? "
            f"(Pass strict_unknown_params=False to downgrade to a warning.)"
        )
        if strict_unknown_params:
            raise ValueError(msg)
        warnings.warn(msg, UserWarning, stacklevel=3)

    return internal
