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

# ── Identity param lists (no unit conversion, scale=1, offset=0) ─────

_DUST_EMISSION_IDENTITY_PARAMS = [
    "dust_T",
    "dust_beta_ir",
    "dust_alpha_mir",
    "dust_alpha_dale",
    "dust_umin",
    "dust_gamma_dl",
    "dust_qpah",
    "dust_eta_balance",
    "dust_alpha_dl14",
]

_AGN_IDENTITY_PARAMS = [
    "agn_frac",
    "agn_log_lbol",
    "agn_alpha",
    "agn_T_torus",
    "agn_tau_torus",
    "agn_torus_frac",
    "agn_log_mbh",
    "agn_log_ledd",
    "agn_tau_skirtor",
    "agn_p_skirtor",
    "agn_q_skirtor",
    "agn_oa_skirtor",
    "agn_cos_inc",
    "agn_a_spin",
    "agn_T_hot",
    "agn_T_warm",
    "agn_frac_hot",
    "agn_f_hard",
    "agn_gamma_warm",
    "agn_kt_warm",
    "agn_gamma_hard",
    "agn_kt_hot",
    "agn_r_warm_ratio",
    "agn_polar_ebv",
    "agn_polar_oa",
]

_RADIO_IDENTITY_PARAMS = [
    "radio_q_ir",
    "radio_alpha_sf",
    "radio_loudness",
    "radio_alpha_agn",
    "radio_T_e",
    "radio_alpha_ff",
]

_XRAY_IDENTITY_PARAMS = [
    "xray_gamma_agn",
    "xray_alpha_ox",
    "xray_gamma_hmxb",
    "xray_gamma_lmxb",
    "xray_E_cut",
]

_SHOCK_IDENTITY_PARAMS = [
    "shock_frac",
    "shock_velocity",
    "shock_log_density",
    "shock_b_over_sqrt_n",
]

_NEBULAR_IDENTITY_PARAMS = [
    "neb_logU",
    "neb_fesc",
    "neb_fesc_lya",
]


def identity_param_map(names: list[str]) -> dict[str, tuple[str, float, float]]:
    """Build param_map entries for parameters that pass through without unit conversion.

    Parameters
    ----------
    names : list of str
        Parameter names to include.

    Returns
    -------
    dict
        Mapping of ``{name: (name, 1.0, 0.0)}`` for each name.
    """
    return {p: (p, 1.0, 0.0) for p in names}


# ── Non-SFH param map (includes real unit conversions) ───────────────

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
    """Build complete param map from SFH registry + non-SFH params.

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
    """
    from tengri.components.sfh.registry import resolve_sfh

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
    return result


# ── Translation functions ──────────────────────────────────────────


# (find_short_param moved to _aliases.py — imported at top)


def get_internal_params(params, param_map, spec, has_field):
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
            # Check short-form alias: find short name that maps to pub_name
            alias_val = find_short_param(params, pub_name)
            if alias_val is not None:
                internal[int_name] = alias_val * scale + offset
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
