# SPDX-License-Identifier: BSD-3-Clause
"""Parameter translation between public and internal names.

Naming Conventions
------------------
Metallicity:
    Public:   met_logzsol     = log10(Z/Zsun) (relative to solar)
    Internal: log_z_abs       = log10(Z) (absolute)
    Offset:   LOG10_ZSUN = -1.8477 (Asplund 2009, Zsun = 0.0142)

    Tengri uses **Asplund 2009 Zsun = 0.0142** as the single normalizing
    constant for the public ``met_logzsol`` axis. This matches the MIST
    isochrone family. Other SSP libraries adopt different "solar"
    references:

    =========  =======  =======================================
    Library    Zsun     LOG10_ZSUN  (= log10(Zsun))
    =========  =======  =======================================
    MIST       0.0142   -1.8477   ← matches tengri's constant
    PARSEC     0.0152   -1.8181
    Padova     0.0190   -1.7212   ← BC03, default in CIGALE
    BASTI      0.0200   -1.6990
    =========  =======  =======================================

    Practical note: when working entirely in solar-normalized
    ``met_logzsol`` units **and** the SSP file's tabulated Z grid was
    generated against the *same* Zsun, the round-trip is self-consistent.
    When comparing against a code that uses a different Zsun (e.g. CIGALE
    BC03 on Padova), reason in **absolute** ``log_z_abs = met_logzsol +
    LOG10_ZSUN`` and pin that: see ``reproduction/cigale/_drivers/
    consistency_audit.py`` for the canonical CIGALE-comparison pattern.
    See also #412.

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
from tengri.utils.physics_constants import LOG10_ZSUN

# ── Constants ─────────────────────────────────────────────────────

# Solar metallicity convention: tengri uses **Asplund 2009 Zsun = 0.0142**,
# i.e. LOG10_ZSUN = log10(0.0142) = -1.8477. This matches the MIST isochrone
# family. SSP libraries built on Padova (BC03, default CIGALE), PARSEC, or
# BASTI use different Zsun: see module docstring for the table. Reason in
# absolute ``log_z_abs`` (not solar-normalized) for cross-code comparisons.
# ``LOG10_ZSUN`` is imported from physics_constants above and re-exported
# here: this remains the canonical import path for the rest of the tree.

# Per-SSP-library solar Z values (kept as a reference dict so downstream
# code or audits can look up the right Zsun if they need to translate
# between conventions). Not consumed by the forward model directly: the
# public surface uses LOG10_ZSUN above.
LOG10_ZSUN_BY_LIBRARY: dict[str, float] = {
    "mist": -1.8477,  # Asplund 2009, Zsun = 0.0142
    "parsec": -1.8181,  # Bressan+ 2012, Zsun = 0.0152
    "padova": -1.7212,  # BC03 / CIGALE default, Zsun = 0.0190
    "basti": -1.6990,  # Pietrinferni+ 2004, Zsun = 0.0200
}

# ── Parameter maps: public → (internal, unit_scale, offset) ───────

_EVOLVING_MET_PARAM_MAP = {
    "met_logzsol_0": ("log_z_abs_initial", 1.0, LOG10_ZSUN),  # log(Z/Zsun) → log(Z)
    "met_logzsol_final": ("log_z_abs_final", 1.0, LOG10_ZSUN),
}

_EVOLVING_ALPHA_PARAM_MAP = {
    "met_alpha_fe_old": ("alpha_fe_old", 1.0, 0.0),  # [alpha/Fe] of oldest stars
    "met_alpha_fe_young": ("alpha_fe_young", 1.0, 0.0),  # [alpha/Fe] at present day
}

# ── Cue identity param lists ──────────────────────────────────────
#
# After Step B (ADR-deepening 2026-05-18) ``_build_param_map`` reads
# identity entries directly from the parameter registry (ADR-0008), so
# the six per-domain ``_*_IDENTITY_PARAMS`` lists previously published
# here are gone. The two Cue lists below stay because
# :class:`SEDModel._init_nebular` consults them at construction time
# to register only the Cue free params the user actually opted into
# via the spec: that's a deliberate filter the auto-derive can't
# express.
#
# No unit-conversion filter is needed: every Cue parameter name is
# ``gas_*`` or ``ionspec_*`` and maps identity. The legacy
# ``_PARAMS_WITH_UNIT_CONVERSION`` frozenset (``met_*``, ``dust_*``,
# ``redshift``, ``noise_*``, ``sigma_v_kms``, ``neb_logZ_gas``) had
# zero overlap with these two Cue buckets: the filter never fired.
# Verified empty by name-set intersection on 2026-05-19; retired in
# the same pass.
_CUE_GAS_IDENTITY_PARAMS: list[str] = sorted(_resolve_lazy_bucket("_CUE_GAS_EXTRA_PARAMS"))
_CUE_IONSPEC_IDENTITY_PARAMS: list[str] = sorted(_resolve_lazy_bucket("_CUE_IONSPEC_PARAMS"))


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

# (Reverse alias map now managed in _aliases.py: imported at top)

# ── High-level API: short name → full prefixed name ──────────────

# sfh_type token → {short_name: full_prefixed_name}
# Used by Model.from_config() to expand user-supplied short priors.
_SFH_SHORT_NAMES: dict[str, dict[str, str]] = {
    "tsnorm": {
        "log_total_mass": "sfh_tsnorm_log_total_mass",
        "peak_lbt_gyr": "sfh_tsnorm_peak_lbt_gyr",
        "width_gyr": "sfh_tsnorm_width_gyr",
        "skew": "sfh_tsnorm_skew",
        "trunc": "sfh_tsnorm_trunc",
    },
    "snorm": {
        "log_total_mass": "sfh_snorm_log_total_mass",
        "peak_lbt_gyr": "sfh_snorm_peak_lbt_gyr",
        "width_gyr": "sfh_snorm_width_gyr",
        "skew": "sfh_snorm_skew",
    },
    "lnorm": {
        "log_total_mass": "sfh_lnorm_log_total_mass",
        "peak_gyr": "sfh_lnorm_peak_gyr",
        "width_gyr": "sfh_lnorm_width_gyr",
        "age_gyr": "sfh_lnorm_age_gyr",
    },
    "dpl": {
        "alpha": "sfh_dpl_alpha",
        "beta": "sfh_dpl_beta",
        "log_total_mass": "sfh_dpl_log_total_mass",
        "tau_gyr": "sfh_dpl_tau_gyr",
        "age_gyr": "sfh_dpl_age_gyr",
    },
    "delayed": {
        "tau_gyr": "sfh_delayed_tau_gyr",
        "log_total_mass": "sfh_delayed_log_total_mass",
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
        User-supplied prior dict, may contain short names like ``"log_total_mass"``
        or full names like ``"sfh_tsnorm_log_total_mass"``. Full names pass through
        unchanged.

    Returns
    -------
    dict
        New dict with all short names expanded to full prefixed names.
        Unknown keys that are neither short nor full names pass through unchanged.

    Examples
    --------
    >>> resolve_short_names("tsnorm", {"log_total_mass": Uniform(8, 12)})
    {"sfh_tsnorm_log_total_mass": Uniform(8, 12)}
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
# NOTE: This legacy map has been superseded by the registry-driven _build_param_map()
# function. Retained here for backwards compatibility with tests that may reference it.
PARAM_MAP = {
    "sfh_alpha": ("alpha", 1.0, 0.0),
    "sfh_beta": ("beta", 1.0, 0.0),
    "sfh_tau_peak_gyr": ("tau_sfh", 1e9, 0.0),
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
    Registered SEDComponent instances auto-declare parameters
    via :meth:`declared_parameters`. This function auto-derives identity mappings
    from those declarations (if not already in the map) to keep the param_map
    synchronized without manual editing.

    Identity entries come from ``tengri.parameters.registry.as_param_map``
    (ADR-0008): walks every ``components/*/_params.py`` directly. Manual
    entries above (``resolve_sfh``, ``_NON_SFH_PARAM_MAP``, etc.) always
    take precedence because they carry unit conversions that identity
    mapping would otherwise silently break.
    """
    from tengri.components.stellar.sfh.registry import resolve_sfh

    _, _, sfh_param_map, _ = resolve_sfh(mean_sfh_type)
    result = dict(sfh_param_map)
    if dust_model in ("single_component", "wg00"):
        # Single screen (Calzetti-style or WG00): skip tau_bc/tau_diff, add tau_v.
        for k, v in _NON_SFH_PARAM_MAP.items():
            if k not in ("dust_tau_bc", "dust_tau_diff"):
                result[k] = v
        result.update(_SINGLE_COMPONENT_DUST_PARAM_MAP)
    else:
        result.update(_NON_SFH_PARAM_MAP)

    # Add manual metallicity param maps that carry unit conversions.
    # These MUST come before auto-derivation so they take precedence.
    # _EVOLVING_ALPHA_PARAM_MAP renames met_alpha_fe_old/young (public)
    # to alpha_fe_old/young (internal), which would otherwise conflict
    # with identity auto-derivation (#1767).
    result.update(_EVOLVING_MET_PARAM_MAP)
    result.update(_EVOLVING_ALPHA_PARAM_MAP)

    # Auto-derive identity entries from the parameter registry (ADR-0008).
    # The registry walks every ``components/*/_params.py`` directly and reads
    # the static ``ParamDeclaration`` tuples: it does not instantiate
    # components with default configs, so we get the full declared-parameter
    # universe regardless of which variant a default ``comp_cls()`` would
    # have picked. This means the registry is strictly safer than the older
    # ``_get_registered_components()``-based auto-derive that needed a
    # ``_SKIP_AUTO_DERIVE`` list (Stellar/DustAttenuation/DustEmission)
    # because their default-config ``declared_parameters()`` would have
    # injected parameters from the wrong variant.
    #
    # Manual entries above (``resolve_sfh``, ``_NON_SFH_PARAM_MAP``, etc.)
    # always take precedence: they carry unit conversions that identity
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


# (find_short_param moved to _aliases.py: imported at top)


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
        vector slots. When ``False``, emit a :class:`UserWarning` instead; used
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
    # Per-spec SFH params (sfh_X_*) are *also* preserved under their public name so
    # the composer in resolve_sfh can dispatch per-component without collisions
    # when multiple SFHs share the same internal kwarg name (e.g.
    # ``log_total_mass`` for any two parametric SFHs after the 2026-05-25
    # normalization refactor). See ``composed_fn`` in
    # ``components/stellar/sfh/registry.py`` and #372.
    for pub_name, (int_name, scale, offset) in param_map.items():
        is_sfh = pub_name.startswith("sfh_")
        if pub_name in params:
            value = params[pub_name]
            # String-typed Fixed params (e.g. shock_abundance="solar") are config
            # enums, not numeric values: pass them through verbatim so downstream
            # code that branches on the string still sees it. Numeric params get
            # the standard scale/offset translation.
            if isinstance(value, str):
                internal[int_name] = value
                if is_sfh:
                    internal[pub_name] = value
            else:
                translated = value * scale + offset
                internal[int_name] = translated
                if is_sfh:
                    internal[pub_name] = translated
        else:
            # Check short-form alias: find short name that maps to pub_name
            alias_val = find_short_param(params, pub_name)
            if alias_val is not None:
                if isinstance(alias_val, str):
                    internal[int_name] = alias_val
                    if is_sfh:
                        internal[pub_name] = alias_val
                else:
                    translated = alias_val * scale + offset
                    internal[int_name] = translated
                    if is_sfh:
                        internal[pub_name] = translated
            else:
                # Fall back to fixed value from spec, or skip if absent.
                #
                # ``param_map`` is built from a registry that may include
                # auto-derived entries from SEDComponents (AGN, Radio,
                # IGM, X-ray) regardless of whether the active
                # spec actually uses them. When a spec doesn't use a
                # given component, its parameters are absent from both
                # ``params`` and ``spec``: silently skipping the entry
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
                        # through verbatim: downstream code branches on the string.
                        # None handles enum-typed Fixed where bounds isn't populated.
                        resolved = fixed_val if fixed_val is not None else dist.value
                        internal[int_name] = resolved
                        if is_sfh:
                            internal[pub_name] = resolved
                    else:
                        translated = fixed_val * scale + offset
                        internal[int_name] = translated
                        if is_sfh:
                            internal[pub_name] = translated
                else:
                    raise KeyError(f"Free parameter '{pub_name}' not found in params dict")

    # Handle field latent vector (both full and short names)
    if has_field:
        if "sfh_field_xi" in params:
            internal["xi"] = params["sfh_field_xi"]
        elif "psd_xi" in params:
            internal["xi"] = params["psd_xi"]

    # Raise/warn about unrecognized keys (silent bugs when wrong names used)
    unrecognized = _unknown_param_keys(params, param_map)
    if unrecognized:
        msg = _unknown_param_msg(unrecognized, param_map)
        if strict_unknown_params:
            from tengri.config.exceptions import UnknownParameterError

            raise UnknownParameterError(msg)
        warnings.warn(msg, UserWarning, stacklevel=3)

    return internal


def _recognized_param_keys(param_map):
    """Set of keys ``params`` may legally contain for a given param_map.

    Combines public names, short-form aliases, latent-vector slots, array-data
    inputs (tabulated SFH, metallicity history), and internal names (for
    backwards compat). Pure-Python; cheap enough to call per predict.
    """
    recognized = set(param_map.keys())
    recognized.update(_REVERSE_ALIASES.keys())
    recognized.update({"sfh_field_xi", "psd_xi"})
    recognized.update({"sfh_t_gyr", "sfh_sfr", "met_history"})
    recognized.update(int_name for _, (int_name, _, _) in param_map.items())
    return recognized


#: Public-key prefixes that belong to sibling sub-models composed alongside
#: ``SEDModel`` in higher-level wrappers (``ForwardModel`` + ``SpatialModel``,
#: etc.) and pass through the params dict on their way to a different
#: handler. We recognize them so the SED-side validator doesn't false-flag
#: legitimate sub-model kwargs as typos (#314).
_PASSTHROUGH_PARAM_PREFIXES: tuple[str, ...] = ("spatial_",)


def _unknown_param_keys(params, param_map):
    """Return sorted list of param keys not recognized for ``param_map``."""
    recognized = _recognized_param_keys(param_map)
    return sorted(
        k for k in params if k not in recognized and not k.startswith(_PASSTHROUGH_PARAM_PREFIXES)
    )


def _unknown_param_msg(unrecognized, param_map):
    """Build the ``UnknownParameterError`` message with did-you-mean hints."""
    import difflib

    public = sorted(param_map.keys())
    hints = []
    for bad in unrecognized:
        close = difflib.get_close_matches(bad, public, n=3, cutoff=0.6)
        if close:
            hints.append(f"  {bad!r} → did you mean {close}?")
        else:
            hints.append(f"  {bad!r} → no close match (this param is not in the live model)")
    return (
        "Unrecognized parameter names passed to Model: "
        f"{list(unrecognized)}.\n"
        + "\n".join(hints)
        + "\n\nValid free + fixed parameter names: see ``model.spec.summary()`` "
        "or ``list(model.spec.param_map_public_keys())``."
    )


def check_unknown_params(params, param_map):
    """Validate ``params`` keys against the live param_map; raise on unknowns.

    Lightweight set-difference validator suitable for use at JIT entry points
    (no value translation, no tracing concerns). Used by
    :meth:`SEDModel.predict_observables_jit` and friends to surface typo /
    stale-override bugs that the JIT path would otherwise silently drop.

    Parameters
    ----------
    params : Mapping
        User-supplied parameter dict.
    param_map : Mapping
        ``public_name -> (internal_name, scale, offset)`` from the SEDModel.

    Raises
    ------
    UnknownParameterError
        If any key in ``params`` is not a recognized public, alias, latent,
        array-data, or internal name.
    """
    unrecognized = _unknown_param_keys(params, param_map)
    if unrecognized:
        from tengri.config.exceptions import UnknownParameterError

        raise UnknownParameterError(_unknown_param_msg(unrecognized, param_map))


def check_missing_free_params(params, spec, param_map=None):
    """Raise when a free (non-Fixed) parameter has no value in ``params``.

    Companion to :func:`check_unknown_params` at the ``predict_state``
    entry. Without it, a missing free parameter survives the
    ``{**fixed_values, **params}`` merge and surfaces deep inside a
    component as a bare ``KeyError`` (e.g. ``'dust_tau_bc'``) with no hint
    that the model simply expected a value for every free parameter;
    commonly hit by ``model.mock({})`` on a model whose default dust group
    carries free optical depths.

    Parameters
    ----------
    params : Mapping
        User-supplied parameter dict (public names, short-form aliases, or
        legacy internal names).
    spec : Parameters
        The model's parameter specification.
    param_map : Mapping, optional
        ``public_name -> (internal_name, scale, offset)``. When given, a
        value supplied under the parameter's internal name also counts
        (mirrors the backwards-compat acceptance in
        :func:`check_unknown_params`).

    Raises
    ------
    MissingParameterError
        If any free parameter of ``spec`` is absent from ``params`` under
        its public name, short-form alias, and internal name.

    Notes
    -----
    Specs built by the flat ``Parameters(...)`` expert path auto-register
    the full parameter universe of the AGN/Radio/X-ray/IGM component
    families as non-Fixed distributions regardless of the selected
    variant; their runners read those with per-variant defaults by
    design. For such specs (no ``_group_provenance``), those families are
    exempt from the missing check. Grammar-built specs
    (``SEDModel.build``) register only what the user selected, so every
    free parameter is deliberate and enforced.
    """
    legacy_flat_spec = getattr(spec, "_group_provenance", None) is None
    missing = []
    for pub_name in spec.free_params:
        if pub_name in params:
            continue
        if legacy_flat_spec and pub_name.startswith(("agn_", "radio_", "xray_", "igm_")):
            continue
        if pub_name == "sfh_field_xi" and "psd_xi" in params:
            continue
        if find_short_param(params, pub_name) is not None:
            continue
        if param_map is not None and pub_name in param_map and param_map[pub_name][0] in params:
            continue
        missing.append(pub_name)
    if missing:
        from tengri.config.exceptions import MissingParameterError

        raise MissingParameterError(
            f"Missing values for free parameters: {missing}. Every non-Fixed "
            "parameter needs a value at predict time. Draw a complete set with "
            "``params = model.spec.sample(jax.random.PRNGKey(0))``, or fix "
            "parameters at build time (``'all_params': FIXED`` in the group dict, or "
            "``param=Fixed(value)``). ``model.spec.summary()`` shows which "
            "parameters are free."
        )
