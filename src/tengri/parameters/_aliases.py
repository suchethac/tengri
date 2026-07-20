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
    # divergent spellings for the same physics — dust temperature and PAH mass
    # fraction. Canonical: ``dust_T`` (shared with modified_blackbody / casey2012
    # / the schreiber2016 closure) and ``dust_f_pah`` (matches the dust_f_cold /
    # dust_f_obscuration underscore family).
    "dust_tdust": "dust_T",
    "dust_fpah": "dust_f_pah",
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
