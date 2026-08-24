# SPDX-License-Identifier: BSD-3-Clause
"""Legacy parameter alias resolution and deprecation warnings.

This module consolidates all legacy parameter name handling:

- Alias tables (_LEGACY_PARAM_ALIASES, _LEGACY_SFH_TYPE_ALIASES)
- Reverse alias map for lookup (_REVERSE_ALIASES)
- Unified resolver functions with deprecation warnings

All alias resolution is one-shot at Parameters construction time (not in traced
functions), so no JIT incompatibilities.
"""

import warnings
from typing import Any

# ── Legacy parameter name aliases (user input) ──────────────────────────────

_LEGACY_PARAM_ALIASES = {
    "sfh_alpha": "sfh_dpl_alpha",
    "sfh_beta": "sfh_dpl_beta",
    "sfh_tau_peak_gyr": "sfh_dpl_tau_gyr",
    "psd_sigma": "sfh_field_psd_sigma",
    "psd_tau_myr": "sfh_field_psd_tau_myr",
    # Dust-emission name unification (#849): the Schreiber tabulated components used
    # divergent spellings for the same physics: dust temperature and PAH mass
    # fraction. Canonical: ``dust_T`` (shared with modified_blackbody / casey2012
    # / the schreiber2016 closure) and ``dust_f_pah`` (matches the dust_f_cold /
    # dust_f_obscuration underscore family).
    "dust_tdust": "dust_T",
    "dust_fpah": "dust_f_pah",
    # AGN normalization disambiguation (#1296). Four parameters carried four
    # *different* physical definitions behind near-identical names, two of them
    # distinguishable only by capitalization:
    #
    #   agn_fracAGN    Uniform(0, 0.99)  AGN fraction of the total dust IR
    #   agn_frac_agn   Uniform(0, 1.0)   L_AGN / L_total in a configurable band
    #   agn_frac       Uniform(0, 5.0)   L_AGN / L_stellar_bol
    #   dust_frac_agn  Fixed(0.0)        Dale 2014 additive AGN-heated dust
    #
    # A user reaching for "the AGN fraction" had four wrong answers available
    # and no error to tell them. Each canonical name now states *which*
    # normalization it is.
    #
    # Note the trap this avoids: the obvious camelCase fix, agn_fracAGN ->
    # agn_frac_agn, targets a name that already existed as a different
    # quantity with a different prior in the same module. Applying it would
    # have silently merged two AGN normalizations.
    #
    # agn_frac -> agn_lum_ratio is a second correction: its range runs to 5.0,
    # so it is a ratio, not a fraction, and the old name said otherwise.
    #
    # dust_frac_agn keeps its name -- the ``dust_`` prefix already says which
    # component owns it, so it was never ambiguous.
    "agn_fracAGN": "agn_ir_frac",
    "agn_frac_agn": "agn_band_frac",
    "agn_frac": "agn_lum_ratio",
}

# SFH type short-form aliases
_LEGACY_SFH_TYPE_ALIASES = {
    "double_powerlaw": "dpl",
}

# Reverse alias map: canonical name → short-form alias
# Used for lookup when constructing internal parameter dicts.
_REVERSE_ALIASES = {
    "sfh_dpl_alpha": "sfh_alpha",
    "sfh_dpl_beta": "sfh_beta",
    "sfh_dpl_tau_gyr": "sfh_tau_peak_gyr",
    "sfh_field_psd_sigma": "psd_sigma",
    "sfh_field_psd_tau_myr": "psd_tau_myr",
}


# ── Deprecation warning cache (avoid spam) ────────────────────────────────

_WARNED_ALIASES: set[str] = set()


def _warn_once_if_legacy(name: str, canonical_name: str) -> None:
    """Emit a DeprecationWarning if a legacy alias is used (once per alias).

    Parameters
    ----------
    name : str
        The legacy parameter name used by the user.
    canonical_name : str
        The canonical parameter name it maps to.
    """
    if name not in _WARNED_ALIASES:
        _WARNED_ALIASES.add(name)
        warnings.warn(
            f"Parameter alias '{name}' is deprecated; use '{canonical_name}' instead.",
            DeprecationWarning,
            stacklevel=3,
        )


# ── Resolution functions ───────────────────────────────────────────────────


def resolve_param_name(name: str) -> str:
    """Resolve a parameter name, applying legacy alias mapping.

    If the name is a legacy alias, emits a one-time DeprecationWarning and
    returns the canonical name. Otherwise returns the input name unchanged.

    Parameters
    ----------
    name : str
        The parameter name (possibly a legacy alias).

    Returns
    -------
    str
        The canonical parameter name.
    """
    if name in _LEGACY_PARAM_ALIASES:
        canonical = _LEGACY_PARAM_ALIASES[name]
        _warn_once_if_legacy(name, canonical)
        return canonical
    return name


#: canonical name -> every legacy name that resolves to it. Derived, so it
#: cannot drift from ``_LEGACY_PARAM_ALIASES``.
_LEGACY_NAMES_BY_CANONICAL: dict[str, list[str]] = {}
for _legacy, _canonical in _LEGACY_PARAM_ALIASES.items():
    _LEGACY_NAMES_BY_CANONICAL.setdefault(_canonical, []).append(_legacy)


def legacy_names_for(canonical: str) -> list[str]:
    """Every deprecated spelling of ``canonical``.

    The nested-dict grammar accepts a *short* per-parameter key
    (``agn={'frac': 0.5}``) derived by stripping the group prefix. Renaming a
    parameter therefore silently invalidates its short form too -- the #1296
    AGN renames broke ``agn={'frac': ...}`` and ``agn={'fracAGN': ...}`` in 34
    places. Callers use this to keep accepting the old short key and warn,
    rather than rejecting it with "Unknown key".

    Parameters
    ----------
    canonical : str
        A current parameter name.

    Returns
    -------
    list of str
        Legacy full names mapping to it; empty if it was never renamed.
    """
    return _LEGACY_NAMES_BY_CANONICAL.get(canonical, [])


def resolve_sfh_type(raw_sfh_type: str | list[str] | None) -> list[str]:
    """Resolve SFH type names, applying legacy aliases and emitting warnings.

    Accepts a single name or list of names. Resolves legacy aliases
    (e.g., "double_powerlaw" -> "dpl") and emits one-time DeprecationWarning
    for each unique legacy name used.

    Parameters
    ----------
    raw_sfh_type : str, list of str, or None
        The SFH type(s) to resolve.

    Returns
    -------
    list of str
        The resolved SFH type names (always a list, even if input was a single str).
    """
    if raw_sfh_type is None:
        return []

    if isinstance(raw_sfh_type, str):
        result = [raw_sfh_type]
    else:
        result = list(raw_sfh_type)

    # Resolve aliases and warn
    resolved = []
    for s in result:
        if s in _LEGACY_SFH_TYPE_ALIASES:
            canonical = _LEGACY_SFH_TYPE_ALIASES[s]
            _warn_once_if_legacy(s, canonical)
            resolved.append(canonical)
        else:
            resolved.append(s)

    return resolved


def find_short_param(params: dict[str, Any], target_public_name: str) -> Any | None:
    """Look up a short-form alias value from a parameter dict.

    Used by translate.get_internal_params to resolve legacy names when
    constructing the internal parameter dict.

    Parameters
    ----------
    params : dict
        Parameter dict (may contain legacy names).
    target_public_name : str
        The canonical public name to look up, e.g., ``"sfh_dpl_alpha"``.

    Returns
    -------
    float or None
        The value from params under the short-form alias, or None if not found.
    """
    old_name = _REVERSE_ALIASES.get(target_public_name)
    if old_name and old_name in params:
        return params[old_name]
    return None
