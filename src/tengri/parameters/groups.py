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
NotImplementedError
    If AGN group is provided (deferred to PR4).
    If SFH type is a list (composition lands in PR4).
ValueError
    If unknown group key, unknown type value, or unknown sub-key is provided.

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

from tengri.parameters._param_defs import _DUST_EMISSION_PARAMS
from tengri.parameters.parameters import Parameters
from tengri.parameters.priors import Distribution, Fixed
from tengri.parameters.sentinels import FIXED, FREE

__all__ = ["parse_groups"]


# ── Constants ──────────────────────────────────────────────────────────────


#: Dust emission parameter names that belong to the 'dust.emission' subgroup.
_DUST_EMISSION_PARAM_NAMES = frozenset(_DUST_EMISSION_PARAMS.keys())

#: Valid SFH model types (from the registry).
_VALID_SFH_TYPES = {
    "tsnorm",
    "snorm",
    "norm",
    "lnorm",
    "const",
    "exp",
    "dexp",
    "burst",
    "field",
    "dpl",
    "psb",
    "dense_basis",
    "dense_basis_pure",
    # Aliases
    "truncated_skewnormal_sfh",
    "skewnormal_sfh",
    "gaussian_sfh",
    "lognormal_sfh",
    "const_exp",
    "constant_then_exponential",
    "psb_wild2020",
    "db",
    "dbp",
}

#: Valid dust model types.
_VALID_DUST_TYPES = {
    "two_component",
    "single_component",
}

#: Valid dust emission types.
_VALID_DUST_EMISSION_TYPES = {
    "modified_blackbody",
    "casey2012",
    "dale2014",
    "draine_li2007",
    "draine_li2014",
    "dl07_tabulated",
    "astrodust",
    "bosa",
    "themis",
    "draine2021_pah",
}

#: Valid nebular types.
_VALID_NEBULAR_TYPES = {
    "none",
    "ssp",
    "cue",
    "cloudy",
    "cb19",
}

#: Valid IGM types.
_VALID_IGM_TYPES = {
    "none",
    "madau",
    "inoue14",
}

#: Valid radio types.
_VALID_RADIO_TYPES = {
    "none",
    "condon92",
}

#: Valid X-ray types.
_VALID_XRAY_TYPES = {
    "none",
    "simple",
}

#: Valid dust attenuation laws.
_VALID_DUST_LAWS = {
    "power_law",
    "calzetti",
    "kriek_conroy",
    "smc",
    "cardelli",
    "salim",
    "li08",
}

#: Valid AGN disc block types.
_VALID_AGN_DISC_TYPES = {
    "none",
    "powerlaw",
    "multicolor",
    "kubota_done",
    "adaf",
    "qsogen",
    "grahsp_sbpl",
}

#: Valid AGN torus block types.
_VALID_AGN_TORUS_TYPES = {
    "none",
    "simple",
    "two_temperature",
    "nenkova",
    "skirtor",
    "silva04",
    "cat3d_wind",
    "qsogen",
    "grahsp",
}

#: Valid AGN lines block types.
_VALID_AGN_LINES_TYPES = {
    "none",
    "blr",
    "nlr",
    "grahsp",
    "qsogen",
}

#: Valid AGN feii block types.
_VALID_AGN_FEII_TYPES = {
    "none",
    "grahsp",
    "qsogen_balmer",
}

#: Valid AGN attenuation block types.
_VALID_AGN_ATTEN_TYPES = {
    "none",
    "smc_prevot",
    "polar_dust",
    "grahsp_biatten",
    "qsogen_smc",
}

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
    "agn_tau_skirtor": "agn.torus",
    "agn_p_skirtor": "agn.torus",
    "agn_q_skirtor": "agn.torus",
    "agn_oa_skirtor": "agn.torus",
    "agn_torus_frac": "agn.torus",
    # Lines
    "agn_blr_cf": "agn.lines",
    "agn_nlr_cf": "agn.lines",
    "agn_alpha_ion": "agn.lines",
    "agn_feltre_cf": "agn.lines",
    "neb_xid": "agn.lines",
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
}


# ── Main API ───────────────────────────────────────────────────────────────


def parse_groups(**kwargs) -> Parameters:
    """Translate nested-dict model specification to Parameters.

    Parameters
    ----------
    **kwargs : keyword arguments
        Model configuration. Keys are group names (sfh, dust, neb, igm,
        radio, xray) or top-level settings (redshift, apply_igm, n_grid).

    Returns
    -------
    Parameters
        A fully initialized Parameters object ready for inference.

    Raises
    ------
    ValueError
        If an unknown group key, unknown type value, or unknown parameter
        name is provided.
    NotImplementedError
        If AGN group is provided or if SFH type is a list.

    Notes
    -----
    **JIT-compatible**: no — this is a pure Python translator.

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

    # Partition declared params by owning group
    dust_emission_active = structural_params.dust_emission is not None
    param_partition = _partition_by_group(structural_params.all_params, dust_emission_active)

    # Resolve each parameter's final distribution
    resolved_kwargs = dict(structural_kwargs)
    provenance: dict[str, str] = {}

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
                        center = registry_default.unstandardize(0.0)
                        resolved_kwargs[param_name] = Fixed(float(center))
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
        elif group.startswith("agn."):
            # AGN sub-block (e.g., "agn.disc", "agn.torus")
            parent_group = "agn"
            group_dict = kwargs.get(parent_group, {})
            subkey = group.replace("agn.", "")
            if isinstance(group_dict, dict) and subkey in group_dict:
                group_dict = group_dict[subkey]
            else:
                group_dict = {}
        else:
            group_dict = kwargs.get(group, {})

        # Determine final distribution
        if not isinstance(group_dict, dict):
            # Group was not a dict (or is a sub-key); use structural default
            continue

        final_dist, tag = _resolve_value(
            param_name, group_dict, structural_params.get_distribution(param_name)
        )

        # Only override if there's an actual per-param or wildcard in the group dict
        if group_dict or group == "_toplevel":
            resolved_kwargs[param_name] = final_dist
            provenance[param_name] = tag

    # ── Construct final Parameters ────────────────────────────────────

    final_params = Parameters(**resolved_kwargs)
    # Fill in provenance for params not touched by user/wildcard
    for name in list(final_params._distributions.keys()):
        provenance.setdefault(name, "registry_default")
    object.__setattr__(final_params, "_group_provenance", provenance)
    return final_params


# ── Internal helpers ───────────────────────────────────────────────────────


def _translate_structural(groups: dict) -> dict:
    """Translate group structure to Parameters constructor kwargs.

    Pass 1 of the algorithm: extract type values and convert to structural
    settings (mean_sfh_type, dust_model, dust_law_bc, agn_model, etc.).

    Parameters
    ----------
    groups : dict
        User input from parse_groups(...).

    Returns
    -------
    dict
        Structural kwargs ready to pass to Parameters(...).

    Raises
    ------
    ValueError
        If unknown group keys or type values.
    """
    valid_groups = {"sfh", "dust", "neb", "igm", "radio", "xray", "agn"}
    result = {}

    # Validate and process each group
    for group_name, group_dict in groups.items():
        # Skip top-level settings (process them later)
        if group_name in _TOP_LEVEL_SETTINGS:
            continue

        # Check for unknown groups
        if group_name not in valid_groups:
            suggestions = difflib.get_close_matches(group_name, valid_groups, n=2, cutoff=0.6)
            suggest_str = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
            raise ValueError(
                f"Unknown group key '{group_name}'. "
                f"Valid groups: {', '.join(sorted(valid_groups))}.{suggest_str}"
            )

        # Skip absent/None groups
        if not isinstance(group_dict, dict):
            continue

        # Translate group to structural kwargs
        if group_name == "sfh":
            _translate_sfh(group_dict, result)
        elif group_name == "dust":
            _translate_dust(group_dict, result)
        elif group_name == "neb":
            _translate_neb(group_dict, result)
        elif group_name == "igm":
            _translate_igm(group_dict, result)
        elif group_name == "radio":
            _translate_radio(group_dict, result)
        elif group_name == "xray":
            _translate_xray(group_dict, result)
        elif group_name == "agn":
            _translate_agn(group_dict, result)

    # Apply top-level settings AFTER groups, so they override
    # Skip sentinels; they'll be handled in parameter resolution
    for key in list(groups.keys()):
        if key in _TOP_LEVEL_SETTINGS:
            val = groups[key]
            # Skip sentinels here; they'll be resolved in _resolve_value
            if val is not FREE and val is not FIXED:
                result[key] = val

    return result


def _translate_sfh(sfh_dict: dict, result: dict) -> None:
    """Translate sfh group to mean_sfh_type (supporting single type or list composition)."""
    sfh_type = sfh_dict.get("type")

    if sfh_type is None:
        # Default: dpl + field
        result["mean_sfh_type"] = ["dpl", "field"]
        return

    # Check for composition (list of types)
    if isinstance(sfh_type, list):
        # Validate each type in the composition
        for type_name in sfh_type:
            if type_name not in _VALID_SFH_TYPES:
                suggestions = difflib.get_close_matches(
                    type_name, _VALID_SFH_TYPES, n=3, cutoff=0.6
                )
                suggest_str = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
                raise ValueError(f"Unknown SFH type '{type_name}' in composition.{suggest_str}")
        result["mean_sfh_type"] = sfh_type
        return

    # Validate single type
    if sfh_type not in _VALID_SFH_TYPES:
        suggestions = difflib.get_close_matches(sfh_type, _VALID_SFH_TYPES, n=3, cutoff=0.6)
        suggest_str = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        raise ValueError(f"Unknown SFH type '{sfh_type}'.{suggest_str}")

    result["mean_sfh_type"] = sfh_type


def _translate_dust(dust_dict: dict, result: dict) -> None:
    """Translate dust group to dust_model, dust_law_bc, dust_emission."""
    dust_type = dust_dict.get("type", "two_component")

    # Validate type
    if dust_type not in _VALID_DUST_TYPES:
        suggestions = difflib.get_close_matches(dust_type, _VALID_DUST_TYPES, n=2, cutoff=0.6)
        suggest_str = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        raise ValueError(f"Unknown dust type '{dust_type}'.{suggest_str}")

    result["dust_model"] = dust_type

    # Extract dust law (can be in dust_dict or dust['law_bc'])
    dust_law_bc = dust_dict.get("law_bc", "power_law")
    dust_law_diff = dust_dict.get("law_diff")

    if dust_law_bc not in _VALID_DUST_LAWS:
        suggestions = difflib.get_close_matches(dust_law_bc, _VALID_DUST_LAWS, n=2, cutoff=0.6)
        suggest_str = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        raise ValueError(f"Unknown dust law '{dust_law_bc}'.{suggest_str}")

    result["dust_law_bc"] = dust_law_bc

    if dust_law_diff is not None:
        if dust_law_diff not in _VALID_DUST_LAWS:
            suggestions = difflib.get_close_matches(
                dust_law_diff, _VALID_DUST_LAWS, n=2, cutoff=0.6
            )
            suggest_str = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
            raise ValueError(f"Unknown dust law '{dust_law_diff}'.{suggest_str}")
        result["dust_law_diff"] = dust_law_diff

    # Extract dust emission sub-block
    if "emission" in dust_dict:
        emission_dict = dust_dict["emission"]
        if isinstance(emission_dict, dict):
            emission_type = emission_dict.get("type", None)
            if emission_type is not None:
                if emission_type not in _VALID_DUST_EMISSION_TYPES:
                    suggestions = difflib.get_close_matches(
                        emission_type, _VALID_DUST_EMISSION_TYPES, n=3, cutoff=0.6
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
    if neb_type not in _VALID_NEBULAR_TYPES:
        suggestions = difflib.get_close_matches(neb_type, _VALID_NEBULAR_TYPES, n=2, cutoff=0.6)
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
    elif neb_type == "cloudy":
        result["nebular"] = True
    elif neb_type == "cb19":
        result["nebular"] = "cb19"


def _translate_igm(igm_dict: dict, result: dict) -> None:
    """Translate igm group to apply_igm and related settings."""
    igm_type = igm_dict.get("type", "madau")

    # Validate type
    if igm_type not in _VALID_IGM_TYPES:
        suggestions = difflib.get_close_matches(igm_type, _VALID_IGM_TYPES, n=2, cutoff=0.6)
        suggest_str = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        raise ValueError(f"Unknown IGM type '{igm_type}'.{suggest_str}")

    # Map type to apply_igm
    if igm_type == "none":
        result["apply_igm"] = False
    else:
        # Both madau and inoue14 -> apply_igm=True
        result["apply_igm"] = True

    # Handle optional IGM subkeys
    if igm_dict.get("patchy", False):
        result["igm_patchy"] = True

    if igm_dict.get("dla", False):
        result["dla"] = True


def _translate_radio(radio_dict: dict, result: dict) -> None:
    """Translate radio group to radio=True/False."""
    radio_type = radio_dict.get("type", "none")

    # Validate type
    if radio_type not in _VALID_RADIO_TYPES:
        suggestions = difflib.get_close_matches(radio_type, _VALID_RADIO_TYPES, n=2, cutoff=0.6)
        suggest_str = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        raise ValueError(f"Unknown radio type '{radio_type}'.{suggest_str}")

    result["radio"] = radio_type != "none"


def _translate_xray(xray_dict: dict, result: dict) -> None:
    """Translate xray group to xray=True/False."""
    xray_type = xray_dict.get("type", "none")

    # Validate type
    if xray_type not in _VALID_XRAY_TYPES:
        suggestions = difflib.get_close_matches(xray_type, _VALID_XRAY_TYPES, n=2, cutoff=0.6)
        suggest_str = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        raise ValueError(f"Unknown X-ray type '{xray_type}'.{suggest_str}")

    result["xray"] = xray_type != "none"


def _translate_agn(agn_dict: dict, result: dict) -> None:
    """Translate agn group to AGN composable block selectors.

    Activates agn_model='composable' and sets per-block selectors
    (agn_disc_block, agn_torus_block, agn_lines_block, agn_feii_block,
    agn_attenuation_block). Omitted blocks default to 'none'.

    Parameters
    ----------
    agn_dict : dict
        User's agn group specification.
    result : dict
        Structural kwargs dict (modified in-place).

    Raises
    ------
    ValueError
        If unknown block type or invalid block specification.
    """
    # Activate composable model
    result["agn_model"] = "composable"

    # Define the five canonical sub-blocks and their type validator sets
    block_specs = {
        "disc": _VALID_AGN_DISC_TYPES,
        "torus": _VALID_AGN_TORUS_TYPES,
        "lines": _VALID_AGN_LINES_TYPES,
        "feii": _VALID_AGN_FEII_TYPES,
        "atten": _VALID_AGN_ATTEN_TYPES,
    }

    # Map sub-block names to their result kwargs
    block_to_kwarg = {
        "disc": "agn_disc_block",
        "torus": "agn_torus_block",
        "lines": "agn_lines_block",
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


def _partition_by_group(all_param_names: list[str], dust_emission_active: bool) -> dict[str, str]:
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
        elif name.startswith("radio_"):
            partition[name] = "radio"
        elif name.startswith("igm_") or name.startswith("dla_"):
            partition[name] = "igm"
        elif name.startswith("neb_") or name.startswith("ionspec_") or name.startswith("gas_log"):
            partition[name] = "neb"
        elif dust_emission_active and name in _DUST_EMISSION_PARAM_NAMES:
            partition[name] = "dust.emission"
        elif name.startswith("dust_"):
            partition[name] = "dust"
        elif name.startswith("sfh_") or name.startswith("met_"):
            partition[name] = "sfh"
        else:
            # _structural (settings like mean_sfh_type, dust_model, etc.)
            partition[name] = "_structural"

    return partition


def _resolve_value(
    param_name: str, group_dict: dict, registry_default: Distribution
) -> tuple[Distribution, str]:
    """Resolve the final distribution for a single parameter.

    Checks (in order):
    1. Per-parameter override in group_dict (including bare values)
    2. Wildcard '*' (FREE or FIXED)
    3. Registry default

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

    # Check for per-param override
    if short_name in group_dict:
        val = group_dict[short_name]

        # Validate that this key is actually a parameter (not 'type', '*', etc.)
        structural_keys = {"type", "*", "law_bc", "law_diff", "emission", "patchy", "dla"}
        if short_name in structural_keys:
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
                center = registry_default.unstandardize(0.0)
                return Fixed(float(center)), "user_fixed"
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
            return registry_default, "wildcard_free"
        elif wildcard is FIXED:
            if registry_default.is_fixed:
                return registry_default, "wildcard_fixed"
            else:
                center = registry_default.unstandardize(0.0)
                return Fixed(float(center)), "wildcard_fixed"
        else:
            return registry_default, "registry_default"

    # No override, no wildcard: fall through to registry default (auto-fixed)
    if registry_default.is_fixed:
        return registry_default, "registry_default"
    else:
        center = registry_default.unstandardize(0.0)
        return Fixed(float(center)), "registry_default"


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
