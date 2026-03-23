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

from tengri.models.sfh.registry import resolve_sfh

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Solar metallicity: log10(Zsun) = log10(0.0142) ≈ -1.848 (Asplund 2009)
LOG10_ZSUN = -1.8477116556169435

# ---------------------------------------------------------------------------
# Parameter maps: public → (internal, unit_scale, offset)
# ---------------------------------------------------------------------------

_EVOLVING_MET_PARAM_MAP = {
    "met_logzsol_0": ("log_z_abs_initial", 1.0, LOG10_ZSUN),  # log(Z/Zsun) → log(Z)
    "met_logzsol_final": ("log_z_abs_final", 1.0, LOG10_ZSUN),
}

_EVOLVING_ALPHA_PARAM_MAP = {
    "met_alpha_fe_old": ("alpha_fe_old", 1.0, 0.0),  # [alpha/Fe] of oldest stars
    "met_alpha_fe_young": ("alpha_fe_young", 1.0, 0.0),  # [alpha/Fe] at present day
}

_NON_SFH_PARAM_MAP = {
    "met_logzsol": ("log_z_abs", 1.0, LOG10_ZSUN),  # log(Z/Zsun) → log(Z)
    "met_alpha_fe": ("alpha_fe", 1.0, 0.0),  # [alpha/Fe] in dex (global)
    "dust_tau_bc": ("tau_bc", 1.0, 0.0),
    "dust_tau_diff": ("tau_diff", 1.0, 0.0),
    "dust_slope": ("dust_slope", 1.0, 0.0),
    "redshift": ("redshift", 1.0, 0.0),
    "noise_frac_cal": ("noise_frac_cal", 1.0, 0.0),
    # Dust extra params (identity mapping — no unit conversion)
    "dust_f_obscuration": ("f_obscuration", 1.0, 0.0),
    "dust_bump_strength": ("dust_bump_strength", 1.0, 0.0),
    "dust_delta": ("dust_delta", 1.0, 0.0),
    "dust_Rv": ("dust_Rv", 1.0, 0.0),
    # Noise model
    "noise_dof": ("noise_dof", 1.0, 0.0),
}

# Reverse alias map: new canonical name → old legacy name
# Used by find_legacy_param to accept old-style param dicts.
# TODO(future): Remove once all callers use new parameter names.
_REVERSE_ALIASES = {
    "sfh_dpl_alpha": "sfh_alpha",
    "sfh_dpl_beta": "sfh_beta",
    "sfh_dpl_tau_gyr": "sfh_tau_peak_gyr",
    "sfh_dpl_log_peak_sfr": "sfh_peak_sfr",
    "sfh_field_psd_sigma": "psd_sigma",
    "sfh_field_psd_tau_myr": "psd_tau_myr",
}

# Legacy module-level PARAM_MAP for backward compatibility with imports
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

# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _build_param_map(mean_sfh_type):
    """Build complete param map from SFH registry + non-SFH params.

    Parameters
    ----------
    mean_sfh_type : list[str]
        SFH type tokens, e.g. ``["tsnorm"]`` or ``["tsnorm", "field"]``.

    Returns
    -------
    dict
        public_name -> (internal_name, scale, offset)
    """
    _, _, sfh_param_map, _ = resolve_sfh(mean_sfh_type)
    result = dict(sfh_param_map)
    result.update(_NON_SFH_PARAM_MAP)
    return result


# ---------------------------------------------------------------------------
# Translation functions
# ---------------------------------------------------------------------------


def find_legacy_param(params, target_public_name):
    """Check if params contains a legacy alias for target_public_name.

    Parameters
    ----------
    params : dict
        Parameter dict (may contain old-style names).
    target_public_name : str
        The canonical public name to look up, e.g. ``"sfh_dpl_alpha"``.

    Returns
    -------
    float or None
        The value from params under the legacy alias, or ``None`` if not found.
    """
    old_name = _REVERSE_ALIASES.get(target_public_name)
    if old_name and old_name in params:
        return params[old_name]
    return None


def get_internal_params(params, param_map, spec, has_field):
    """Translate a public parameter dict to internal names with unit conversion.

    Conversion applied element-wise: ``internal = public * scale + offset``.

    Also accepts legacy parameter names (sfh_alpha, psd_sigma, etc.) via
    reverse alias lookup, and fills in fixed values from the spec when a
    parameter is absent from ``params``.

    Parameters
    ----------
    params : dict
        Public parameter dict, e.g. from ``spec.sample(key)``.
    param_map : dict
        Mapping ``public_name -> (internal_name, scale, offset)``, as built
        by ``_build_param_map``.
    spec : ParamSpec
        The parameter specification (used to look up fixed defaults).
    has_field : bool
        Whether the model uses a stochastic field component. When ``True``
        the latent vector ``xi`` is passed through from ``params``.

    Returns
    -------
    dict
        Internal parameter dict ready for the low-level forward model.

    Raises
    ------
    KeyError
        If a free parameter is absent from ``params`` and not in ``spec``.
    """
    internal = {}
    for pub_name, (int_name, scale, offset) in param_map.items():
        if pub_name in params:
            internal[int_name] = params[pub_name] * scale + offset
        else:
            # Check legacy alias: find old name that maps to pub_name
            legacy_val = find_legacy_param(params, pub_name)
            if legacy_val is not None:
                internal[int_name] = legacy_val * scale + offset
            else:
                # Fall back to fixed value from spec
                try:
                    dist = spec.get_distribution(pub_name)
                    if dist.is_fixed:
                        internal[int_name] = dist.bounds[0] * scale + offset
                    else:
                        raise KeyError(f"Free parameter '{pub_name}' not found in params dict")
                except KeyError as err:
                    raise KeyError(
                        f"Parameter '{pub_name}' not found in params dict and not in spec"
                    ) from err

    # Handle field latent vector (both new and legacy names)
    if has_field:
        if "sfh_field_xi" in params:
            internal["xi"] = params["sfh_field_xi"]
        elif "psd_xi" in params:
            internal["xi"] = params["psd_xi"]

    # Warn about unrecognized keys (silent bugs when wrong names used)
    recognized = set(param_map.keys())
    recognized.update(_REVERSE_ALIASES.keys())
    recognized.update({"sfh_field_xi", "psd_xi"})
    # Also recognize internal names (for backwards compat)
    recognized.update(int_name for _, (int_name, _, _) in param_map.items())
    unrecognized = set(params.keys()) - recognized
    if unrecognized:
        warnings.warn(
            f"Unrecognized parameter names passed to Model: {sorted(unrecognized)}. "
            f"These will be silently ignored. Did you mean one of: "
            f"{sorted(param_map.keys())}?",
            UserWarning,
            stacklevel=3,
        )

    return internal
