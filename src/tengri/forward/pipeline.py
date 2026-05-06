"""Core SED computation pipeline.

This module implements the forward model engine that translates
physical parameters into rest-frame SEDs. It handles SFH dispatch,
metallicity interpolation, dust attenuation, nebular emission,
AGN contribution, dust emission, and IGM absorption.

All functions take a ``model`` argument (the :class:`~tengri.model.Model`
instance) instead of ``self``, allowing the heavy computation to live
outside the class while preserving access to model state.
"""

from __future__ import annotations

import jax.numpy as jnp

from tengri.components.dust.attenuation import (
    single_component_dust_fast,
    two_component_dust_fast,
)
from tengri.components.stellar.sps.dsps_wrapper import (
    compute_csp_sed,
    compute_log_z_evolving,
    compute_surviving_mass,
    effective_metallicity,
    has_alpha_grid,
    interpolate_mass_remaining,
    interpolate_met_alpha,
    interpolate_met_alpha_evolving,
    interpolate_metallicity,
    interpolate_metallicity_evolving,
    interpolate_metallicity_smooth,
    interpolate_metallicity_smooth_evolving,
)


def interp_metallicity(model, log_z, ssp_flux=None, ssp_lgmet=None):
    """Dispatch metallicity interpolation on SSP grid (single Z value).

    Parameters
    ----------
    model : SEDModel
        The model instance.
    log_z : float
        log10(Z) absolute metallicity [dimensionless].
    ssp_flux : ndarray, optional
        Traced override for ``model.ssp_data.ssp_flux``. When provided
        with ``ssp_lgmet``, the SSP arrays enter the JIT graph as
        runtime tensors instead of closure-captured constants
        (Phase II-2 trace path; see ``docs/dev/quickstart_oom_diagnosis.md``).
    ssp_lgmet : ndarray, optional
        Traced override for ``model.ssp_data.ssp_lgmet``.

    Returns
    -------
    ndarray, shape (n_age, n_wave)
        SSP flux interpolated to target metallicity [erg/s/Hz].

    Notes
    -----
    **JIT-compatible**: yes — uses smooth or nearest-neighbor interpolation.
    """
    flux = ssp_flux if ssp_flux is not None else model.ssp_data.ssp_flux
    lgmet = ssp_lgmet if ssp_lgmet is not None else model.ssp_data.ssp_lgmet
    if model._met_interp == "smooth":
        return interpolate_metallicity_smooth(
            flux,
            lgmet,
            log_z,
            model._lgmet_scatter,
        )
    return interpolate_metallicity(flux, lgmet, log_z)


def interp_metallicity_evolving(model, log_z_per_age, ssp_flux=None, ssp_lgmet=None):
    """Dispatch per-age metallicity interpolation on SSP grid.

    Parameters
    ----------
    model : SEDModel
        The model instance.
    log_z_per_age : ndarray, shape (n_age,)
        log10(Z) absolute metallicity at each SSP age bin [dimensionless].

    Returns
    -------
    ndarray, shape (n_age, n_wave)
        SSP flux with age-dependent metallicity [erg/s/Hz].

    Notes
    -----
    **JIT-compatible**: yes — uses smooth or nearest-neighbor interpolation.
    """
    flux = ssp_flux if ssp_flux is not None else model.ssp_data.ssp_flux
    lgmet = ssp_lgmet if ssp_lgmet is not None else model.ssp_data.ssp_lgmet
    if model._met_interp == "smooth":
        return interpolate_metallicity_smooth_evolving(
            flux,
            lgmet,
            log_z_per_age,
            model._lgmet_scatter,
        )
    return interpolate_metallicity_evolving(flux, lgmet, log_z_per_age)


def _closure_a_sfh_prep(model, p, sfr, t_obs_gyr_for_weights):
    """Mirror :class:`StellarSEDComponent.apply` for closure-path-A.

    Builds the orchestrator-canonical inputs to DSPS:
    ``(t_cosmic_asc, sfr_asc, total_mass)`` and the orchestrator's
    64-pt SFH grid. The three legacy ``compute_sed_components``
    branches (no-α delta-Z, α-aware, chem_evol) all share these
    inputs to feed ``calc_rest_sed_sfh_table_lognormal_mdf`` /
    ``calc_rest_sed_sfh_table_met_table`` exactly the same way the
    orchestrator's component does.

    The three required alignments:

    1. Grid resolution: ``n_grid=64`` regardless of ``spec.stochastic``
       (legacy uses 256 for non-stochastic). For stochastic SFHs the
       legacy ``sfr`` already lives on a 64-pt grid with the GP draw
       baked in, so reuse it directly to preserve the realisation.
    2. Linear lookback-time interpolation to SSP ages (NOT legacy's
       log10-age interpolation).
    3. NaN-safe cosmic-time prep: a strictly-monotonic ramp + SFR-
       zeroing for invalid SSP bins (``ssp_age > t_obs``). Bare
       ``jnp.clip(min=1e-3)`` collapses multiple invalid bins to the
       same boundary value, producing a degenerate ``gal_t_table``
       that DSPS NaNs on at z>0.05 with old SSPs.

    Parameters
    ----------
    model : SEDModel
    p : dict
        Internal-name parameters (``model._get_internal_params(...)``).
    sfr : array, shape (n_internal_grid,)
        SFR on the legacy internal log_age_grid (256-pt non-stochastic,
        64-pt stochastic). Used directly when stochastic; otherwise
        re-evaluated on the 64-pt orchestrator grid.
    t_obs_gyr_for_weights : float
        Age of the universe at the observation redshift [Gyr].

    Returns
    -------
    t_cosmic_asc : array, shape (n_ssp_age,)
        Ascending cosmic-time grid for DSPS (Gyr).
    sfr_asc : array, shape (n_ssp_age,)
        SFR on the SSP age grid, ascending in cosmic time, with
        invalid-SSP-bin SFR zeroed.
    total_mass : scalar array
        Total stellar mass formed [Msun] (trapezoid integral).
    sfh_lbt_grid_orch : array, shape (64,)
        The orchestrator-style 64-pt SFH lookback-time grid [yr];
        useful when the caller needs to recompute mode-specific
        quantities (e.g. chem_evol's ``log_z_per_age``) on the same
        grid for bit-exact match.
    """
    from tengri.components.stellar.sfh.gp_sfh import make_log_age_grid

    _n_grid = 64
    _log_age_grid = make_log_age_grid(_n_grid)
    sfh_lbt_grid_orch = jnp.power(10.0, _log_age_grid)
    if model._uses_stochastic_sfh:
        # Stochastic SFHs already use n_grid=64 in legacy; reuse to
        # preserve the GP draw baked into ``sfr``.
        _sfr_orch_grid = sfr
    else:
        _kw = {k: v for k, v in p.items() if k in model._sfh_internal_names}
        _sfr_orch_grid = model._sfh_fn(sfh_lbt_grid_orch, **_kw)

    _sfr_on_ssp_orch = jnp.interp(model.ssp_ages_yr, sfh_lbt_grid_orch, _sfr_orch_grid)

    _ssp_age_gyr = model.ssp_ages_yr / 1e9
    _T_TABLE_MIN = 0.01
    _t_cosmic_raw = t_obs_gyr_for_weights - _ssp_age_gyr
    _n_ssp = model.ssp_ages_yr.shape[0]
    _t_cosmic_floor = jnp.maximum(_t_cosmic_raw, _T_TABLE_MIN)
    _valid = _t_cosmic_raw > 0.0
    _t_cosmic_asc_raw = _t_cosmic_floor[::-1]
    _sfr_asc_raw = _sfr_on_ssp_orch[::-1]
    _n_invalid = jnp.sum(~_valid[::-1])
    _idx_pos = jnp.arange(_n_ssp)
    _is_invalid_pos = _idx_pos < _n_invalid
    _ramp = _T_TABLE_MIN + (_T_TABLE_MIN * 0.5) * (_idx_pos + 1) / jnp.maximum(_n_invalid, 1)
    t_cosmic_asc = jnp.where(_is_invalid_pos, _ramp, _t_cosmic_asc_raw)
    sfr_asc = jnp.where(_is_invalid_pos, 0.0, _sfr_asc_raw)
    total_mass = jnp.maximum(jnp.trapezoid(sfr_asc, t_cosmic_asc * 1e9), 0.0)

    return t_cosmic_asc, sfr_asc, total_mass, sfh_lbt_grid_orch


def interp_met_alpha_dispatch(
    model, log_z, alpha_fe, ssp_flux=None, ssp_lgmet=None, ssp_alpha_fe=None
):
    """Dispatch metallicity+alpha interpolation based on SSP grid dimensionality.

    Uses 4D bilinear interpolation if alpha-enhanced SSPs are available,
    otherwise falls back to effective_metallicity approximation on 3D grid.

    Parameters
    ----------
    model : SEDModel
        The model instance.
    log_z : float
        Iron abundance [Fe/H] or log10(Z/Zsun) [dimensionless].
    alpha_fe : float
        Alpha-element enhancement [α/Fe] [dex].

    Returns
    -------
    ndarray, shape (n_age, n_wave)
        SSP flux with metallicity and alpha enhancement [erg/s/Hz].

    Notes
    -----
    **JIT-compatible**: yes — uses bilinear or effective-Z interpolation.
    """
    if has_alpha_grid(model.ssp_data):
        return interpolate_met_alpha(
            ssp_flux if ssp_flux is not None else model.ssp_data.ssp_flux,
            ssp_lgmet if ssp_lgmet is not None else model.ssp_data.ssp_lgmet,
            ssp_alpha_fe if ssp_alpha_fe is not None else model.ssp_data.ssp_alpha_fe,
            log_z,
            alpha_fe,
        )
    # Fallback: effective_metallicity on 3D grid, with DSPS-canonical
    # lognormal MDF triweight kernel (Hearin+ 2021 Eq. 11). Matches
    # the no-α-enhancement path so ``alpha_fe = 0`` reduces exactly
    # to the non-α SED. See
    # ``docs/dev/20260504-csp-integral-canonicalization.md``.
    from dsps.sed.metallicity_weights import calc_lgmet_weights_from_lognormal_mdf

    log_z_eff = effective_metallicity(log_z, alpha_fe)
    flux = ssp_flux if ssp_flux is not None else model.ssp_data.ssp_flux
    lgmet = ssp_lgmet if ssp_lgmet is not None else model.ssp_data.ssp_lgmet
    lgmet_w = calc_lgmet_weights_from_lognormal_mdf(log_z_eff, model._lgmet_scatter, lgmet)
    return jnp.einsum("m,maw->aw", lgmet_w, flux)


def interp_met_alpha_evolving_dispatch(
    model,
    log_z_per_age,
    alpha_fe_per_age,
    ssp_flux=None,
    ssp_lgmet=None,
    ssp_alpha_fe=None,
):
    """Dispatch per-age metallicity+alpha interpolation on SSP grid.

    Uses per-age 4D bilinear if alpha-enhanced SSPs available,
    otherwise effective-Z approximation on 3D grid.

    Parameters
    ----------
    model : SEDModel
        The model instance.
    log_z_per_age : ndarray, shape (n_age,)
        [Fe/H] at each SSP age bin [dimensionless].
    alpha_fe_per_age : ndarray or float
        [α/Fe] at each age (array or scalar) [dex].

    Returns
    -------
    ndarray, shape (n_age, n_wave)
        SSP flux with per-age metallicity and alpha [erg/s/Hz].

    Notes
    -----
    **JIT-compatible**: yes — uses bilinear or effective-Z interpolation.
    """
    if has_alpha_grid(model.ssp_data):
        # Broadcast scalar alpha_fe to per-age array
        if not hasattr(alpha_fe_per_age, "shape") or alpha_fe_per_age.ndim == 0:
            alpha_fe_per_age = jnp.full_like(log_z_per_age, alpha_fe_per_age)
        return interpolate_met_alpha_evolving(
            ssp_flux if ssp_flux is not None else model.ssp_data.ssp_flux,
            ssp_lgmet if ssp_lgmet is not None else model.ssp_data.ssp_lgmet,
            ssp_alpha_fe if ssp_alpha_fe is not None else model.ssp_data.ssp_alpha_fe,
            log_z_per_age,
            alpha_fe_per_age,
        )
    # Fallback: effective_metallicity on 3D grid
    log_z_eff = effective_metallicity(log_z_per_age, alpha_fe_per_age)
    return interp_metallicity_evolving(model, log_z_eff, ssp_flux=ssp_flux, ssp_lgmet=ssp_lgmet)


def get_dust_kwargs(model, p):
    """Extract dust law and emission keyword arguments from internal params.

    Parameters
    ----------
    model : SEDModel
        The model instance.
    p : dict
        Internal parameter dictionary from ``_get_internal_params()``.

    Returns
    -------
    dict
        Keyword arguments for fused dust kernels. Keys may include
        ``f_obscuration``, ``dust_bump_strength``, ``dust_delta``,
        ``dust_Rv``, ``alpha_fe``, and (if dust emission enabled)
        ``dust_T``, ``dust_beta_ir``, ``dust_eta_balance``.

    Notes
    -----
    **JIT-compatible**: yes — returns a dict of JAX-compatible values.
    """
    kw = {
        "f_obscuration": p.get("f_obscuration", 0.0),
        "dust_bump_strength": p.get("dust_bump_strength", 0.0),
        "dust_delta": p.get("dust_delta", 0.0),
        "dust_Rv": p.get("dust_Rv", 3.1),
        "alpha_fe": p.get("alpha_fe", 0.0),
    }
    # Dust emission params (fused kernel handles MBB/dale2014 inline)
    if model._dust_emission_model in ("modified_blackbody", "dale2014"):
        kw["dust_T"] = p.get("dust_T", 35.0)
        kw["dust_beta_ir"] = p.get("dust_beta_ir", 1.6)
        kw["dust_eta_balance"] = p.get("dust_eta_balance", 1.0)
    return kw


def _compute_dust_atten(model, wave_dt, p):
    """Compute dust attenuation curve dispatched on model type.

    Returns shape (n_ages, n_wave) for both single and two-component dust.

    Parameters
    ----------
    model : SEDModel
        Model instance with dust configuration.
    wave_dt : ndarray, shape (n_wave,)
        Wavelength grid [Angstrom].
    p : dict
        Internal parameter dictionary.

    Returns
    -------
    ndarray, shape (n_age, n_wave)
        Dust transmission factor (fraction transmitted) [dimensionless].

    Notes
    -----
    **JIT-compatible**: yes — all operations use ``jnp`` primitives.
    """
    law_kw = {
        "f_obscuration": p.get("f_obscuration", 0.0),
        "n_slope": p.get("dust_slope", -0.7),
        "dust_bump_strength": p.get("dust_bump_strength", 0.0),
        "dust_delta": p.get("dust_delta", 0.0),
        "dust_Rv": p.get("dust_Rv", 3.1),
    }
    if model._dust_model == "single_component":
        return single_component_dust_fast(
            wave_dt,
            n_ages=len(model.ssp_ages_yr),
            tau_v=p["tau_v"],
            law=model._dust_law_bc,
            **law_kw,
        )
    # For the non-fused fallback path, always use smooth sigmoid weights
    # (this path is only hit for evolving-Z etc., not performance-critical)
    from tengri.components.dust.attenuation import precompute_dust_age_weights

    if model._precomputed.dust_age_weights is not None:
        dust_age_w = model._precomputed.dust_age_weights.astype(wave_dt.dtype)
    else:
        dust_age_w = precompute_dust_age_weights(model.ssp_ages_yr).astype(wave_dt.dtype)
    return two_component_dust_fast(
        wave_dt,
        dust_age_w,
        tau_v1=p["tau_bc"],
        tau_v2=p["tau_diff"],
        law_bc=model._dust_law_bc,
        law_diff=model._dust_law_diff,
        **law_kw,
    )


def get_agn_kwargs(model, p):
    """Extract AGN keyword arguments from internal params for fused kernel.

    Parameters
    ----------
    model : SEDModel
        The model instance.
    p : dict
        Internal parameter dictionary from ``_get_internal_params()``.

    Returns
    -------
    dict
        Keyword arguments for fused AGN kernel. If AGN is disabled, returns
        an empty dict. Otherwise contains keys like ``agn_log_lbol``,
        ``agn_alpha``, ``agn_T_torus``, ``agn_tau_torus``, ``agn_torus_frac``,
        ``agn_log_mbh``, ``agn_log_ledd`` [various dimensionless units].

    Notes
    -----
    **JIT-compatible**: yes — returns a dict of JAX-compatible values.
    """
    if not (model._agn_model is not None and model._agn_luminosity_mode):
        return {}
    return {
        "agn_log_lbol": p.get("agn_log_lbol", 10.0),
        "agn_alpha": p.get("agn_alpha", -1.0),
        "agn_T_torus": p.get("agn_T_torus", 1000.0),
        "agn_tau_torus": p.get("agn_tau_torus", 5.0),
        "agn_torus_frac": p.get("agn_torus_frac", 0.5),
        "agn_log_mbh": p.get("agn_log_mbh", 7.0),
        "agn_log_ledd": p.get("agn_log_ledd", -1.0),
    }


def compute_sed_components(
    model, params, _sfr=None, _weights=None, need_intrinsic=False, rest_wavelength=None
):
    """Compute full SED and all intermediates (shared engine for forward model).

    Orchestrates stellar population synthesis, dust attenuation, and non-stellar
    components (nebular, AGN, radio, X-ray). Returns all intermediates to enable
    derived-quantity calculations without re-running forward model.

    Parameters
    ----------
    model : SEDModel
        Model instance.
    params : dict
        Parameter values (public names).
    _sfr : ndarray, optional
        Pre-computed SFR on log-age grid [Msun/yr].
    _weights : ndarray, optional
        Pre-computed CSP mass weights [Msun].
    need_intrinsic : bool, optional
        If True, compute intrinsic (unattenuated) SED. Default False.
    rest_wavelength : ndarray, optional
        Custom rest wavelength grid [Angstrom]. If None, use model grid.

    Returns
    -------
    dict
        Complete SED computation with keys:

        - ``sed_total``: ndarray, shape (n_wave,) — final rest-frame SED [erg/s/Hz]
        - ``sed_attenuated``: ndarray, shape (n_wave,) — stellar after dust [erg/s/Hz]
        - ``sed_intrinsic``: ndarray, shape (n_wave,) — stellar unattenuated [erg/s/Hz]
        - ``ssp_flux_at_z``: ndarray, shape (n_age, n_wave) — Z-interpolated SSP [erg/s/Hz]
        - ``weights``: ndarray, shape (n_age,) — CSP mass weights [Msun]
        - ``sfr``: ndarray, shape (n_grid,) — SFR on log-age grid [Msun/yr]
        - ``p``: dict — internal parameters
        - ``agn_bol_erg``: float — AGN bolometric luminosity [erg/s]

    Notes
    -----
    **JIT-compatible**: no — uses Python-level dispatch and model introspection.
    """
    p = model._get_internal_params(params)

    _LSUN = 3.828e33  # erg/s/Hz per Lsun — used in stellar + polar-dust blocks
    _dsps_weights_2d = None  # set by DSPS table path if used
    _use_dsps_table = False

    # SFH: parametric (from params) or tabulated (from sfh_t_gyr + sfh_sfr)
    if _sfr is not None:
        sfr = _sfr
    elif "sfh_t_gyr" in params and "sfh_sfr" in params:
        # Tabulated SFH from simulation -- use DSPS table functions
        # which properly handle time->age conversion, trapezoidal
        # weighting, and metallicity distribution (lognormal MDF).
        t_cosmic_gyr = jnp.asarray(params["sfh_t_gyr"])
        sfr_table = jnp.asarray(params["sfh_sfr"])
        z = p.get("redshift", 0.0)
        t_obs_gyr = model._t_universe_gyr(z) if hasattr(model, "_t_universe_gyr") else 13.7

        _use_dsps_table = True
        # Also compute sfr on internal grid for SFH plotting
        t_lookback_yr = jnp.maximum((t_obs_gyr - t_cosmic_gyr) * 1e9, 1.0)
        log_t_lookback = jnp.log10(t_lookback_yr)
        sfr_on_ssp = jnp.interp(
            model.ssp_log_ages_yr,
            log_t_lookback[::-1],
            sfr_table[::-1],
        )
        sfr = jnp.interp(
            model.log_age_grid,
            log_t_lookback[::-1],
            sfr_table[::-1],
        )
    else:
        sfr = model._compute_sfr(p)

    _dsps_mode = model._csp_integration in ("dsps_native", "dsps_met_table")

    # SFH integration: trapz / log_trapz fall through to the
    # DSPS-canonical trapezoidal-in-cosmic-time helper
    # (``compute_dsps_age_weights``); log_interp keeps its
    # Johnson+2021 matrix path; the dsps_* modes compute weights
    # inside the metallicity dispatch block below.
    #
    # Replaces the legacy lookback-rectangle ``sfr_on_ssp * _csp_age_dt``
    # rule (Hearin+ 2021 Eq. 9). See
    # ``docs/dev/20260504-csp-integral-canonicalization.md``.
    from tengri.components.stellar.sps.dsps_wrapper import compute_dsps_age_weights

    _z_for_t_obs = p.get("redshift", 0.0)
    _t_obs_gyr_for_weights = (
        model._t_universe_gyr(_z_for_t_obs) if hasattr(model, "_t_universe_gyr") else 13.7
    )

    # Per-call CSP-3D ssp_flux. Set by closure-path-A in the α-aware
    # branch to the α-interpolated cube; otherwise the original 3D
    # grid. Read by the 3D-einsum fallback at line ~752 below.
    _ssp_flux_for_csp_3d = None

    if _weights is not None:
        weights = _weights
        _use_dsps_table = False
    elif "sfh_t_gyr" in params:
        if model._csp_integration == "log_interp":
            weights = model._csp_matrix @ sfr_on_ssp
        elif _dsps_mode:
            weights = None  # computed in metallicity dispatch block below
        else:
            weights = compute_dsps_age_weights(
                sfr_on_ssp,
                model.ssp_ages_yr,
                model.ssp_data.ssp_lg_age_gyr,
                _t_obs_gyr_for_weights,
            )
    else:
        sfr_on_ssp = jnp.interp(model.ssp_log_ages_yr, model.log_age_grid, sfr)
        if model._csp_integration == "log_interp":
            weights = model._csp_matrix @ sfr_on_ssp
        elif _dsps_mode:
            weights = None  # computed in metallicity dispatch block below
        else:
            weights = compute_dsps_age_weights(
                sfr_on_ssp,
                model.ssp_ages_yr,
                model.ssp_data.ssp_lg_age_gyr,
                _t_obs_gyr_for_weights,
            )
        _use_dsps_table = False

    # Alpha-element enhancement
    # When 4D alpha-enhanced SSPs are loaded, proper bilinear (Z, [α/Fe])
    # interpolation is used. Otherwise falls back to effective_metallicity,
    # but ONLY when the user explicitly opts in by making met_alpha_fe a free
    # parameter (or enabling alpha_fe_evolving). When alpha_fe is Fixed(0),
    # effective_metallicity correction is skipped and plain interp_metallicity
    # is used instead.
    _has_alpha = has_alpha_grid(model.ssp_data)
    _use_alpha_fe = (
        _has_alpha
        or getattr(model.spec, "alpha_fe_evolving", False)
        or "met_alpha_fe" in model.spec.free_params
    )

    # Resolve [α/Fe]: either global scalar or per-age evolving ramp
    _alpha_evolving = getattr(model, "_alpha_fe_evolving", False)
    if _alpha_evolving:
        from tengri.components.stellar.sps.dsps_wrapper import compute_alpha_fe_evolving

        alpha_fe_old = p.get("alpha_fe_old", 0.3)
        alpha_fe_young = p.get("alpha_fe_young", 0.0)
        z_val = p.get("redshift", 0.0)
        t_universe = model._t_universe_gyr(z_val) if hasattr(model, "_t_universe_gyr") else 13.7
        alpha_fe = compute_alpha_fe_evolving(
            model.ssp_data.ssp_lg_age_gyr,
            alpha_fe_old,
            alpha_fe_young,
            t_universe,
        )  # (n_age,) per-age values
    else:
        alpha_fe = p.get("alpha_fe", 0.0)  # scalar

    # --- Metallicity + CSP integral ---
    # For tabulated SFH with met_history: use DSPS
    # calc_ssp_weights_sfh_table_met_table which properly handles
    # time->age + metallicity distribution -> 2D weights (n_met, n_age).
    # We then apply dust using these 2D weights for correct
    # age-dependent attenuation.
    if _use_dsps_table and "met_history" in params:
        # Use DSPS for the FULL CSP integral with Z(t) history.
        # This properly handles time->age, trapezoidal weighting,
        # and lognormal metallicity distribution at each age.
        from dsps.sed.stellar_sed import calc_rest_sed_sfh_table_met_table

        met_table = jnp.asarray(params["met_history"])
        # DSPS calc_rest_sed_sfh_table_met_table only supports 3D grids;
        # use effective_metallicity for the Z(t) table regardless of grid dim.
        log_z_abs = effective_metallicity(met_table + (-1.8477), alpha_fe)
        lgmet_scatter = float(params.get("lgmet_scatter", 0.2))
        dsps_result = calc_rest_sed_sfh_table_met_table(
            gal_t_table=t_cosmic_gyr,
            gal_sfr_table=sfr_table,
            gal_lgmet_table=log_z_abs,
            gal_lgmet_scatter=lgmet_scatter,
            ssp_lgmet=model.ssp_data.ssp_lgmet,
            ssp_lg_age_gyr=model.ssp_data.ssp_lg_age_gyr,
            ssp_flux=model.ssp_data.ssp_flux,
            t_obs=t_obs_gyr,
        )
        # DSPS gives the intrinsic (no-dust) SED and 2D weights
        _dsps_intrinsic_sed = dsps_result.rest_sed
        weights = dsps_result.age_weights  # (n_age,) normalized
        # For dust: use Z-marginalized SSP flux per age
        lgmet_w = dsps_result.lgmet_weights  # (n_met, n_age)
        lgmet_w_safe = jnp.maximum(jnp.sum(lgmet_w, axis=0, keepdims=True), 1e-30)
        ssp_flux_at_z = jnp.einsum("ma,maw->aw", lgmet_w / lgmet_w_safe, model.ssp_data.ssp_flux)
        # DSPS age_weights are fractional. Total stellar mass =
        # integral(SFR * dt). Multiply weights by total mass.
        _total_mass_formed = jnp.trapezoid(sfr_table, t_cosmic_gyr * 1e9)
        weights = weights * _total_mass_formed

    elif _use_dsps_table:
        # Tabulated SFH path: use the user's ``sfh_t_gyr`` /
        # ``sfh_sfr`` cosmic-time grid for SFH integration via DSPS,
        # and apply DSPS canonical lognormal-MDF metallicity weighting
        # (Hearin+ 2021, Eq. 11) — replacing the previous
        # ``interp_metallicity`` bilinear path. The user's t_cosmic
        # grid does not derive from the SSP age axis, so this avoids
        # the negative-cosmic-time NaN failure mode that affects
        # ``compute_dsps_native_weights`` when SSP ages exceed t_obs.
        from dsps.sed.ssp_weights import calc_age_weights_from_sfh_table

        dsps_age_w = calc_age_weights_from_sfh_table(
            gal_t_table=t_cosmic_gyr,
            gal_sfr_table=sfr_table,
            ssp_lg_age_gyr=model.ssp_data.ssp_lg_age_gyr,
            t_obs=t_obs_gyr,
        )
        _total_mass_formed = jnp.trapezoid(sfr_table, t_cosmic_gyr * 1e9)
        weights = dsps_age_w * _total_mass_formed  # (n_age,) Msun

        log_z_abs_scalar = p.get("log_z_abs", -1.8477)
        if _use_alpha_fe:
            # α-enhancement still uses the 4D bilinear path; α-aware
            # joint einsum is the next migration step.
            ssp_flux_at_z = interp_met_alpha_dispatch(model, log_z_abs_scalar, alpha_fe)
        else:
            from dsps.sed.metallicity_weights import (
                calc_lgmet_weights_from_lognormal_mdf,
            )

            lgmet_scatter = float(p.get("lgmet_scatter", getattr(model, "_lgmet_scatter", 0.2)))
            lgmet_w = calc_lgmet_weights_from_lognormal_mdf(
                log_z_abs_scalar, lgmet_scatter, model.ssp_data.ssp_lgmet
            )
            ssp_flux_at_z = jnp.einsum("m,maw->aw", lgmet_w, model.ssp_data.ssp_flux)

    # Metallicity interpolation (non-table path)
    # Priority: met_history (array) > evolving_metallicity > single Z
    if "met_history" in params and not _use_dsps_table:
        # Tabulated metallicity history Z(t) from simulation
        # Expects array of log10(Z/Zsun) at same time grid as sfh_t_gyr
        met_table = jnp.asarray(params["met_history"])
        t_cosmic_gyr = jnp.asarray(
            params.get(
                "sfh_t_gyr",
                jnp.linspace(0.1, 13.7, len(met_table)),
            )
        )
        z_val = p.get("redshift", 0.0)
        t_obs_gyr = model._t_universe_gyr(z_val) if hasattr(model, "_t_universe_gyr") else 13.7
        t_lookback_yr = jnp.maximum((t_obs_gyr - t_cosmic_gyr) * 1e9, 1.0)
        log_t_lookback = jnp.log10(t_lookback_yr)
        # Interpolate Z(t) onto SSP age grid
        log_z_on_ssp = jnp.interp(
            model.ssp_log_ages_yr,
            log_t_lookback[::-1],
            met_table[::-1],
        )
        log_z_abs = log_z_on_ssp + (-1.8477)  # solar offset
        if _use_alpha_fe:
            ssp_flux_at_z = interp_met_alpha_evolving_dispatch(model, log_z_abs, alpha_fe)
        else:
            ssp_flux_at_z = interp_metallicity_evolving(model, log_z_abs)
    elif model._met_mode == "ramp":
        z = p.get("redshift", 0.0)
        t_universe_gyr = model._t_universe_gyr(z)
        log_z_per_age = compute_log_z_evolving(
            model.ssp_data.ssp_lg_age_gyr,
            p["log_z_abs_initial"],
            p["log_z_abs_final"],
            t_universe_gyr,
        )
        if model._csp_integration == "dsps_met_table":
            from tengri.components.stellar.sps.dsps_wrapper import compute_dsps_met_table_weights

            lgmet_scatter = float(p.get("lgmet_scatter", getattr(model, "_lgmet_scatter", 0.2)))
            weights, ssp_flux_at_z = compute_dsps_met_table_weights(
                sfr_on_ssp,
                log_z_per_age,
                model.ssp_ages_yr,
                model.ssp_data.ssp_lgmet,
                model.ssp_data.ssp_lg_age_gyr,
                model.ssp_data.ssp_flux,
                t_universe_gyr,
                lgmet_scatter,
            )
        elif _use_alpha_fe:
            ssp_flux_at_z = interp_met_alpha_evolving_dispatch(model, log_z_per_age, alpha_fe)
        else:
            # Closure-path-A for ramp metallicity (default
            # ``csp_integration='trapz'``): use
            # ``calc_rest_sed_sfh_table_met_table`` with the per-age
            # ramp-interpolated ``log_z_per_age``. Mirrors
            # :class:`StellarSEDComponent.apply` (component.py ramp
            # branch) so the two paths produce bit-exact equal SEDs.
            # Replaces the previous ``interp_metallicity_evolving``
            # 2-point bilinear path which drove a ~32% per-wavelength
            # divergence (now pinned at rtol=1e-2 by
            # ``test_ramp_orchestrator_rest_sed_close_to_legacy``).
            from dsps.sed.stellar_sed import calc_rest_sed_sfh_table_met_table

            lgmet_scatter_ramp = float(
                p.get("lgmet_scatter", getattr(model, "_lgmet_scatter", 0.2))
            )

            (
                _t_cosmic_asc_ramp,
                _sfr_asc_ramp,
                _total_mass_ramp,
                _sfh_lbt_grid_ramp,
            ) = _closure_a_sfh_prep(model, p, sfr, _t_obs_gyr_for_weights)

            # ``log_z_per_age`` is on the SSP age grid already (built
            # via ``compute_log_z_evolving(ssp_lg_age_gyr, ...)`` above).
            # Ramp Z(t) is grid-resolution-independent (closed-form
            # linear interpolation between two endpoints), so the
            # legacy ``log_z_per_age`` is fine to reuse — no need to
            # recompute on the orchestrator grid like chem_evol.
            _dsps_result_ramp = calc_rest_sed_sfh_table_met_table(
                gal_t_table=_t_cosmic_asc_ramp,
                gal_sfr_table=_sfr_asc_ramp,
                gal_lgmet_table=log_z_per_age[::-1],
                gal_lgmet_scatter=lgmet_scatter_ramp,
                ssp_lgmet=model.ssp_data.ssp_lgmet,
                ssp_lg_age_gyr=model.ssp_data.ssp_lg_age_gyr,
                ssp_flux=model.ssp_data.ssp_flux,
                t_obs=_t_obs_gyr_for_weights,
            )
            _dsps_weights_2d = _dsps_result_ramp.weights * _total_mass_ramp
            ssp_flux_at_z = jnp.einsum(
                "ma,maw->aw", _dsps_result_ramp.weights, model.ssp_data.ssp_flux
            )
    elif model._met_mode == "chem_evol":
        # Chemical evolution: derive Z(t) from SFH via gas-regulator model
        from tengri.components.stellar.sfh.chemical_evolution import (
            chem_evol_metallicity_on_ssp_grid,
        )

        log_z_per_age = chem_evol_metallicity_on_ssp_grid(
            model.ssp_log_ages_yr,
            model.log_age_grid,
            sfr,
            yield_y=p.get("chem_yield", 0.03),
            eta_outflow=p.get("chem_eta_outflow", 0.0),
            f_gas_init=p.get("chem_f_gas_init", 0.9),
            return_frac=p.get("chem_return_frac", 0.4),
        )
        if model._csp_integration == "dsps_met_table":
            from tengri.components.stellar.sps.dsps_wrapper import compute_dsps_met_table_weights

            z = p.get("redshift", 0.0)
            _has_t_uni = hasattr(model, "_t_universe_gyr")
            t_universe_gyr = model._t_universe_gyr(z) if _has_t_uni else 13.7
            lgmet_scatter = float(p.get("lgmet_scatter", getattr(model, "_lgmet_scatter", 0.2)))
            weights, ssp_flux_at_z = compute_dsps_met_table_weights(
                sfr_on_ssp,
                log_z_per_age,
                model.ssp_ages_yr,
                model.ssp_data.ssp_lgmet,
                model.ssp_data.ssp_lg_age_gyr,
                model.ssp_data.ssp_flux,
                t_universe_gyr,
                lgmet_scatter,
            )
        elif _use_alpha_fe:
            ssp_flux_at_z = interp_met_alpha_evolving_dispatch(model, log_z_per_age, alpha_fe)
        else:
            # Closure-path-A for chem_evol (default ``csp_integration='trapz'``):
            # use ``calc_rest_sed_sfh_table_met_table`` with the per-age
            # ``log_z_per_age`` from the gas-regulator model. Mirrors
            # :class:`StellarSEDComponent.apply` (component.py chem_evol
            # branch) so the two paths produce bit-exact equal SEDs.
            #
            # Replaces the previous ``interp_metallicity_evolving`` 2-point
            # bilinear path which drove the ~50% UV-side divergence
            # documented in the (formerly xfail) test
            # ``test_chem_evol_orchestrator_rest_sed_close_to_legacy``.
            from dsps.sed.stellar_sed import calc_rest_sed_sfh_table_met_table

            lgmet_scatter_chem = float(
                p.get("lgmet_scatter", getattr(model, "_lgmet_scatter", 0.2))
            )

            # Orchestrator-canonical SFH prep — shared helper.
            (
                _t_cosmic_asc_chem,
                _sfr_asc_chem,
                _total_mass_chem,
                _sfh_lbt_grid_chem,
            ) = _closure_a_sfh_prep(model, p, sfr, _t_obs_gyr_for_weights)

            # Recompute log_z_per_age on the orchestrator-style 64-pt
            # SFH grid + its SFR realisation, since chem_evol's input
            # grid affects the integrated metallicity profile.
            _log_age_grid_chem = jnp.log10(_sfh_lbt_grid_chem)
            if model._uses_stochastic_sfh:
                _sfr_orch_grid_chem = sfr
            else:
                _kw_orch_chem = {
                    k: v for k, v in p.items() if k in model._sfh_internal_names
                }
                _sfr_orch_grid_chem = model._sfh_fn(_sfh_lbt_grid_chem, **_kw_orch_chem)
            _log_z_per_age_chem = chem_evol_metallicity_on_ssp_grid(
                model.ssp_log_ages_yr,
                _log_age_grid_chem,
                _sfr_orch_grid_chem,
                yield_y=p.get("chem_yield", 0.03),
                eta_outflow=p.get("chem_eta_outflow", 0.0),
                f_gas_init=p.get("chem_f_gas_init", 0.9),
                return_frac=p.get("chem_return_frac", 0.4),
            )

            _dsps_result_chem = calc_rest_sed_sfh_table_met_table(
                gal_t_table=_t_cosmic_asc_chem,
                gal_sfr_table=_sfr_asc_chem,
                gal_lgmet_table=_log_z_per_age_chem[::-1],
                gal_lgmet_scatter=lgmet_scatter_chem,
                ssp_lgmet=model.ssp_data.ssp_lgmet,
                ssp_lg_age_gyr=model.ssp_data.ssp_lg_age_gyr,
                ssp_flux=model.ssp_data.ssp_flux,
                t_obs=_t_obs_gyr_for_weights,
            )
            _dsps_weights_2d = _dsps_result_chem.weights * _total_mass_chem
            ssp_flux_at_z = jnp.einsum(
                "ma,maw->aw", _dsps_result_chem.weights, model.ssp_data.ssp_flux
            )
    else:
        if model._csp_integration == "dsps_native":
            from tengri.components.stellar.sps.dsps_wrapper import compute_dsps_native_weights

            z = p.get("redshift", 0.0)
            _has_t_uni = hasattr(model, "_t_universe_gyr")
            t_universe_gyr = model._t_universe_gyr(z) if _has_t_uni else 13.7
            lgmet = p.get("log_z_abs", -1.8477)
            lgmet_scatter = float(p.get("lgmet_scatter", getattr(model, "_lgmet_scatter", 0.2)))
            weights, ssp_flux_at_z = compute_dsps_native_weights(
                sfr_on_ssp,
                model.ssp_ages_yr,
                model.ssp_data.ssp_lgmet,
                model.ssp_data.ssp_lg_age_gyr,
                model.ssp_data.ssp_flux,
                t_universe_gyr,
                lgmet,
                lgmet_scatter,
            )
        elif model._csp_integration == "dsps_met_table":
            from tengri.components.stellar.sps.dsps_wrapper import compute_dsps_met_table_weights

            z = p.get("redshift", 0.0)
            _has_t_uni = hasattr(model, "_t_universe_gyr")
            t_universe_gyr = model._t_universe_gyr(z) if _has_t_uni else 13.7
            lgmet_scatter = float(p.get("lgmet_scatter", getattr(model, "_lgmet_scatter", 0.2)))
            lgmet_per_age = jnp.full_like(model.ssp_ages_yr, p.get("log_z_abs", -1.8477))
            weights, ssp_flux_at_z = compute_dsps_met_table_weights(
                sfr_on_ssp,
                lgmet_per_age,
                model.ssp_ages_yr,
                model.ssp_data.ssp_lgmet,
                model.ssp_data.ssp_lg_age_gyr,
                model.ssp_data.ssp_flux,
                t_universe_gyr,
                lgmet_scatter,
            )
        elif _use_alpha_fe:
            # Closure-path-A for the α-aware branch. Mirrors the no-α
            # branch below but feeds DSPS an α-interpolated 3D ssp_flux
            # cube instead of the bare 3D grid:
            #
            # 1. If a 4D α-grid is loaded (``has_alpha_grid``), do
            #    bilinear-only-in-α interp on ``ssp_flux`` at the
            #    requested ``alpha_fe`` → 3D (n_met, n_age, n_wave)
            #    cube. Use the canonical ``log_z`` for DSPS lognormal
            #    MDF (no ``effective_metallicity`` shift — α is a real
            #    grid axis).
            # 2. If no α-grid, use the original 3D ssp_flux and apply
            #    ``effective_metallicity(log_z, α)`` for DSPS lognormal
            #    MDF (matches the no-α-grid fallback in
            #    :func:`interp_met_alpha_dispatch`).
            #
            # At ``α=0``, both branches reduce exactly to the no-α
            # delta-Z path — closes ``test_alpha_zero_matches_no_alpha``.
            from dsps.sed.stellar_sed import (
                calc_rest_sed_sfh_table_lognormal_mdf as _calc_rest_sed_α,
            )

            _lgmet = p.get("log_z_abs", p.get("log_z_abs_final", -1.8477))
            lgmet_scatter = float(p.get("lgmet_scatter", getattr(model, "_lgmet_scatter", 0.2)))

            if has_alpha_grid(model.ssp_data):
                # α-only bilinear interp on 4D ssp_flux at requested α.
                # Shape (n_met, n_alpha, n_age, n_wave) → (n_met, n_age, n_wave).
                _ssp_alpha_fe_grid = model.ssp_data.ssp_alpha_fe
                _afe_clipped = jnp.clip(
                    alpha_fe, _ssp_alpha_fe_grid[0], _ssp_alpha_fe_grid[-1]
                )
                _ia = jnp.clip(
                    jnp.searchsorted(_ssp_alpha_fe_grid, _afe_clipped) - 1,
                    0,
                    _ssp_alpha_fe_grid.shape[0] - 2,
                )
                _fa = (_afe_clipped - _ssp_alpha_fe_grid[_ia]) / (
                    _ssp_alpha_fe_grid[_ia + 1] - _ssp_alpha_fe_grid[_ia]
                )
                _ssp_flux_3d_alpha = (1.0 - _fa) * model.ssp_data.ssp_flux[
                    :, _ia
                ] + _fa * model.ssp_data.ssp_flux[:, _ia + 1]
                _lgmet_for_dsps = _lgmet
            else:
                # 3D fallback: use effective_metallicity α correction +
                # original 3D ssp_flux.
                _ssp_flux_3d_alpha = model.ssp_data.ssp_flux
                _lgmet_for_dsps = effective_metallicity(_lgmet, alpha_fe)

            # Orchestrator-canonical SFH prep, shared with the no-α
            # and chem_evol closure-A blocks.
            _t_cosmic_asc_α, _sfr_asc_α, _total_mass_α, _ = _closure_a_sfh_prep(
                model, p, sfr, _t_obs_gyr_for_weights
            )

            _dsps_result_α = _calc_rest_sed_α(
                gal_t_table=_t_cosmic_asc_α,
                gal_sfr_table=_sfr_asc_α,
                gal_lgmet=_lgmet_for_dsps,
                gal_lgmet_scatter=lgmet_scatter,
                ssp_lgmet=model.ssp_data.ssp_lgmet,
                ssp_lg_age_gyr=model.ssp_data.ssp_lg_age_gyr,
                ssp_flux=_ssp_flux_3d_alpha,
                t_obs=_t_obs_gyr_for_weights,
            )
            _dsps_weights_2d = _dsps_result_α.weights * _total_mass_α
            ssp_flux_at_z = jnp.einsum(
                "ma,maw->aw", _dsps_result_α.weights, _ssp_flux_3d_alpha
            )
            # Plumb the α-interpolated cube into the 3D-einsum
            # fallback below so it sums over (m, a) at the requested α.
            _ssp_flux_for_csp_3d = _ssp_flux_3d_alpha
        else:
            # DSPS-canonical FULL CSP integral (closure path A from
            # ``docs/dev/20260504-csp-integral-canonicalization.md``):
            # call ``calc_rest_sed_sfh_table_lognormal_mdf`` directly,
            # mirroring :class:`StellarSEDComponent.apply` exactly so
            # the legacy and orchestrator paths produce literally
            # identical SEDs (closes ``test_orchestrator_rest_sed_bit_exact_to_legacy``
            # to ``rtol=1e-6``).
            #
            # The joint ``(n_met, n_age)`` weights are stored as
            # ``_dsps_weights_2d`` so the 3D-einsum fallback path
            # below (line ~752, ``einsum("ma,aw,maw->w", ...)``)
            # handles the SED construction in the same reduction
            # order as the orchestrator. The JIT kernel
            # ``_compositional.exact_sed`` is bypassed for this
            # branch — it uses a different two-step einsum that
            # accumulates ~1e-3 float64 reduction-order residual
            # against the orchestrator. ``ssp_flux_at_z`` is still
            # set (to the orchestrator-style age-marginalised cube)
            # for diagnostic continuity with downstream consumers.
            from dsps.sed.stellar_sed import (
                calc_rest_sed_sfh_table_lognormal_mdf,
            )

            _lgmet2 = p.get("log_z_abs", p.get("log_z_abs_final", -1.8477))
            lgmet_scatter = float(p.get("lgmet_scatter", getattr(model, "_lgmet_scatter", 0.2)))

            # Orchestrator-canonical SFH prep (n_grid=64, linear
            # lookback interp, NaN-safe cosmic-time ramp).
            _t_cosmic_asc, _sfr_asc, _total_mass, _ = _closure_a_sfh_prep(
                model, p, sfr, _t_obs_gyr_for_weights
            )

            _dsps_result = calc_rest_sed_sfh_table_lognormal_mdf(
                gal_t_table=_t_cosmic_asc,
                gal_sfr_table=_sfr_asc,
                gal_lgmet=_lgmet2,
                gal_lgmet_scatter=lgmet_scatter,
                ssp_lgmet=model.ssp_data.ssp_lgmet,
                ssp_lg_age_gyr=model.ssp_data.ssp_lg_age_gyr,
                ssp_flux=model.ssp_data.ssp_flux,
                t_obs=_t_obs_gyr_for_weights,
            )
            # ``dsps_result.weights`` is the joint (n_met, n_age)
            # probability distribution (sums to 1). Scale to absolute
            # Msun so the 3D-einsum fallback returns the SED in
            # erg/s/Hz directly via the existing ``LSUN`` factor.
            _dsps_weights_2d = _dsps_result.weights * _total_mass
            # Age-marginalised SSP flux cube — used downstream for
            # diagnostic publication and for non-JIT branches that
            # reach for ``ssp_flux_at_z``.
            ssp_flux_at_z = jnp.einsum(
                "ma,maw->aw", _dsps_result.weights, model.ssp_data.ssp_flux
            )

        # --- Fast JIT path: dust + einsum in one compiled kernel ---
        # Eliminates ~78% Python dispatch overhead (4-14x speedup).
        if (
            _dsps_weights_2d is None
            and model._met_mode != "ramp"
            and model._compositional.exact_sed is not None
        ):
            sed_attenuated, sed_intrinsic_jit = model._compositional.exact_sed(
                weights,
                ssp_flux_at_z,
                p.get("tau_bc", 0.0),
                p.get("tau_diff", 0.0),
                n_slope=p.get("dust_slope", -0.7),
                dust_bump_strength=p.get("dust_bump_strength", 0.0),
                dust_delta=p.get("dust_delta", 0.0),
                dust_Rv=p.get("dust_Rv", 3.1),
                f_obscuration=p.get("f_obscuration", 0.0),
                tau_v=p.get("tau_v", 0.0),
            )
            sed_intrinsic = (
                sed_intrinsic_jit
                if (model._dust_emission_model is not None or need_intrinsic)
                else None
            )
            # Skip the non-JIT dust/einsum below
            _dsps_weights_2d = "jit_done"

    # --- Fallback: non-JIT path for DSPS/evolving-Z/met-history ---
    if _dsps_weights_2d is not None and _dsps_weights_2d != "jit_done":
        dt = model._forward_dtype
        ssp_flux_at_z = ssp_flux_at_z.astype(dt)
        wave_dt = model.ssp_data.ssp_wave.astype(dt)

        dust_atten = _compute_dust_atten(model, wave_dt, p)

        _ssp_flux_3d_for_einsum = (
            _ssp_flux_for_csp_3d if _ssp_flux_for_csp_3d is not None
            else model.ssp_data.ssp_flux
        ).astype(dt)
        sed_attenuated = (
            _LSUN
            * jnp.einsum(
                "ma,aw,maw->w",
                _dsps_weights_2d,
                dust_atten,
                _ssp_flux_3d_for_einsum,
            )
        ).astype(jnp.float64)

        sed_intrinsic = None
        if model._dust_emission_model is not None or need_intrinsic:
            sed_intrinsic = (
                _LSUN
                * jnp.einsum(
                    "ma,maw->w",
                    _dsps_weights_2d,
                    _ssp_flux_3d_for_einsum,
                )
            ).astype(jnp.float64)

    elif _dsps_weights_2d is None:
        # Non-DSPS fallback (evolving Z or met_history without DSPS)
        dt = model._forward_dtype
        ssp_flux_at_z = ssp_flux_at_z.astype(dt)
        wave_dt = model.ssp_data.ssp_wave.astype(dt)

        dust_atten = _compute_dust_atten(model, wave_dt, p)

        sed_attenuated = compute_csp_sed(weights.astype(dt), ssp_flux_at_z, dust_atten).astype(
            jnp.float64
        )

        sed_intrinsic = None
        if model._dust_emission_model is not None or need_intrinsic:
            ones_atten = jnp.ones_like(dust_atten)
            sed_intrinsic = compute_csp_sed(weights.astype(dt), ssp_flux_at_z, ones_atten).astype(
                jnp.float64
            )

    sed = sed_attenuated

    # Component tracking for decomposition plots
    _neb_sed = jnp.zeros_like(sed_attenuated)
    _shock_sed = jnp.zeros_like(sed_attenuated)

    # ── Emission components via shared helpers (emission_helpers.py) ────
    from tengri.forward.emission_helpers import (
        attenuate_emission,
        nebular_emission,
        shock_emission,
    )

    _ssp_wave = model.ssp_data.ssp_wave
    _law_bc_fn = getattr(model, "_dust_law_bc_fn", None)
    _law_diff_fn = getattr(model, "_dust_law_diff_fn", None)
    _neb_bc_fn = getattr(model, "_neb_dust_law_bc_fn", _law_bc_fn)
    _neb_dust_mode = getattr(model, "_neb_dust_mode", "bc")
    _dust_kw = {
        "dust_slope": p.get("dust_slope", -0.7),
        "dust_bump_strength": p.get("dust_bump_strength", 0.0),
    }
    _tau_bc = p.get("tau_bc", p.get("tau_v", 0.0))
    _tau_diff = p.get("tau_diff", 0.0)

    # Track absorbed luminosity for energy balance
    L_absorbed_extra = jnp.float64(0.0)

    # Nebular emission
    if model._nebular_backend is not None and model._nebular_backend.has_free_params:
        _sfr_for_neb = (
            sfr_table[-1] if "sfr_table" in dir() else (sfr[-1] if "sfr" in dir() else 1.0)
        )
        neb_raw = nebular_emission(
            model._nebular_backend,
            weights,
            _ssp_wave,
            model.ssp_log_ages_yr,
            p.get("log_z_abs", p.get("log_z_abs_final", -1.8477)),
            _sfr_for_neb,
            neb_logU=p.get("neb_logU", -3.0),
            neb_logZ_gas=p.get("neb_logZ_gas", None),
            neb_fesc=p.get("neb_fesc", 0.0),
            neb_fesc_lya=p.get("neb_fesc_lya", 0.0),
        )
        neb_sed, L_abs_neb = attenuate_emission(
            neb_raw,
            _ssp_wave,
            _neb_dust_mode,
            _tau_bc,
            _tau_diff,
            _law_bc_fn,
            _law_diff_fn,
            neb_bc_fn=_neb_bc_fn,
            **_dust_kw,
        )
        L_absorbed_extra = L_absorbed_extra + L_abs_neb
        _neb_sed = neb_sed
        sed = sed + neb_sed

    # Shock emission (diffuse dust only — shocks outside birth clouds)
    if getattr(model, "_uses_shock", False):
        shock_raw = shock_emission(
            _ssp_wave,
            sed,
            shock_frac=p.get("shock_frac", 0.0),
            shock_velocity=p.get("shock_velocity", 300.0),
            shock_log_density=p.get("shock_log_density", 0.0),
            shock_b_over_sqrt_n=p.get("shock_b_over_sqrt_n", 1.0),
            shock_abundance=p.get("shock_abundance", "solar"),
            shock_component=p.get("shock_component", "combined"),
        )
        shock_sed, _ = attenuate_emission(
            shock_raw,
            _ssp_wave,
            "diff",
            0.0,
            _tau_diff,
            _law_bc_fn,
            _law_diff_fn,
            **_dust_kw,
        )
        _shock_sed = shock_sed
        sed = sed + shock_sed

    # ── Energy balance on SSP grid BEFORE interpolation ──────────────
    # L_absorbed = stellar absorption + nebular/shock dust absorption
    _c_aa_const = 2.99792458e18
    L_ir = jnp.float64(0.0)
    if model._dust_emission_model is not None and sed_intrinsic is not None:
        nu_ssp = _c_aa_const / _ssp_wave
        L_absorbed_stellar = -jnp.trapezoid(sed_intrinsic - sed_attenuated, nu_ssp)
        # Guard against NaN/Inf from pure SSPs with zero continuum
        L_absorbed_stellar = jnp.where(jnp.isfinite(L_absorbed_stellar), L_absorbed_stellar, 0.0)
        L_absorbed = jnp.maximum(L_absorbed_stellar + L_absorbed_extra, 0.0)
        eta_balance = p.get("dust_eta_balance", 1.0)
        L_ir = jnp.maximum(L_absorbed * eta_balance, 0.0)

    # ── Compute AGN bolometric luminosity on SSP grid ─────────────────
    agn_bol_erg = 0.0
    agn_log_lbol = 0.0
    agn_frac_for_model = 0.0
    if model._agn_model is not None:
        if model._agn_luminosity_mode:
            agn_log_lbol = p.get("agn_log_lbol", 10.0)
            agn_frac_for_model = 1.0
            agn_bol_erg = 10.0**agn_log_lbol * 3.828e33  # Lsun → erg/s
        else:
            agn_frac_for_model = p.get("agn_frac", 0.0)
            nu_agn = _c_aa_const / model.ssp_data.ssp_wave
            L_bol_stellar = -jnp.trapezoid(sed, nu_agn)
            agn_bol_erg = L_bol_stellar * agn_frac_for_model
            # AGN model functions expect log10(L_bol / Lsun), convert from erg/s
            agn_log_lbol = jnp.log10(jnp.maximum(agn_bol_erg, 1e-50)) - jnp.log10(3.828e33)

    # ── Populate physical quantities for radio/X-ray ──────────────────
    _L_ir = L_ir if model._dust_emission_model is not None else p.get("_L_ir_cached", 0.0)
    _sfr_cached = (
        sfr_table[-1]
        if "sfr_table" in dir()
        else (sfr[-1] if "sfr" in dir() else p.get("_sfr_cached", 1.0))
    )
    _mstar_formed = jnp.sum(weights) if weights is not None else p.get("_mstar_cached", 1e10)
    if weights is not None and model.ssp_data.ssp_mass_remaining is not None:
        _mass_remaining = interpolate_mass_remaining(
            model.ssp_data.ssp_mass_remaining,
            model.ssp_data.ssp_lgmet,
            p.get("log_z_abs", -1.8477),
        )
        _mstar_surviving = compute_surviving_mass(weights, _mass_remaining)
    else:
        _mstar_surviving = _mstar_formed
    _mstar = _mstar_surviving

    # ── Zone 2: Interpolate to panchromatic grid if needed ────────────
    # Determine output wavelength grid
    rest_wave_target = rest_wavelength if rest_wavelength is not None else model.ssp_data.ssp_wave
    _needs_extension = rest_wave_target is not model.ssp_data.ssp_wave

    if _needs_extension:
        from tengri.utils.wavelength import interpolate_sed_to_grid

        ssp_wave = model.ssp_data.ssp_wave
        sed = interpolate_sed_to_grid(ssp_wave, sed, rest_wave_target)
        if sed_intrinsic is not None:
            sed_intrinsic = interpolate_sed_to_grid(ssp_wave, sed_intrinsic, rest_wave_target)
        sed_attenuated = interpolate_sed_to_grid(ssp_wave, sed_attenuated, rest_wave_target)
        _neb_sed = interpolate_sed_to_grid(ssp_wave, _neb_sed, rest_wave_target)
        _shock_sed = interpolate_sed_to_grid(ssp_wave, _shock_sed, rest_wave_target)

    # Use the target grid for all Zone 2 emission components
    wave_z2 = rest_wave_target
    _n_z2 = len(wave_z2)
    _dust_ir_sed = jnp.zeros(_n_z2)
    _agn_sed = jnp.zeros(_n_z2)
    _radio_sed = jnp.zeros(_n_z2)
    _xray_sed = jnp.zeros(_n_z2)

    # ── Emission components via shared helpers ──────────────────────────
    from tengri.forward.emission_helpers import (
        agn_emission,
        dust_ir_emission,
        radio_emission,
        xray_emission,
    )

    # ── Dust IR emission (energy-balanced) ────────────────────────────
    if model._dust_emission_model is not None:
        from tengri.components.dust.emission import resolve_emission_model

        # Ensure L_ir is finite before passing to dust emission model
        L_ir_safe = jnp.where(jnp.isfinite(L_ir) & (L_ir > 0.0), L_ir, 0.0)
        dust_ir = dust_ir_emission(
            resolve_emission_model(model._dust_emission_model),
            wave_z2,
            L_ir_safe,
            dust_T=p.get("dust_T", 35.0),
            dust_beta_ir=p.get("dust_beta_ir", 1.6),
            dust_alpha_mir=p.get("dust_alpha_mir", 2.0),
            dust_alpha_dale=p.get("dust_alpha_dale", 2.0),
            dust_umin=p.get("dust_umin", 1.0),
            dust_gamma_dl=p.get("dust_gamma_dl", 0.01),
            dust_qpah=p.get("dust_qpah", 2.5),
            dust_alpha_dl14=p.get("dust_alpha_dl14", 2.0),
        )
        _dust_ir_sed = dust_ir
        sed = sed + dust_ir

    # ── AGN contribution ──────────────────────────────────────────────
    if model._agn_model is not None:
        from tengri.components.agn import resolve_agn_model

        agn_sed = agn_emission(
            resolve_agn_model(model._agn_model),
            wave_z2,
            agn_log_lbol=agn_log_lbol,
            agn_frac=agn_frac_for_model,
            agn_polar_ebv=p.get("agn_polar_ebv", 0.0),
            agn_cos_inc=p.get("agn_cos_inc", 0.5),
            agn_polar_oa=p.get("agn_polar_oa", 45.0),
            agn_alpha=p.get("agn_alpha", -1.0),
            agn_T_torus=p.get("agn_T_torus", 1000.0),
            agn_tau_torus=p.get("agn_tau_torus", 5.0),
            agn_torus_frac=p.get("agn_torus_frac", 0.5),
            agn_log_mbh=p.get("agn_log_mbh", 7.0),
            agn_log_ledd=p.get("agn_log_ledd", -1.0),
            agn_a_spin=p.get("agn_a_spin", 0.0),
            agn_tau_skirtor=p.get("agn_tau_skirtor", 7.0),
            agn_p_skirtor=p.get("agn_p_skirtor", 1.0),
            agn_q_skirtor=p.get("agn_q_skirtor", 1.0),
            agn_oa_skirtor=p.get("agn_oa_skirtor", 40.0),
            agn_T_hot=p.get("agn_T_hot", 1200.0),
            agn_T_warm=p.get("agn_T_warm", 300.0),
            agn_frac_hot=p.get("agn_frac_hot", 0.3),
            agn_f_hard=p.get("agn_f_hard", 0.02),
            agn_gamma_warm=p.get("agn_gamma_warm", 2.5),
            agn_kt_warm=p.get("agn_kt_warm", 0.2),
            agn_gamma_hard=p.get("agn_gamma_hard", 1.8),
            agn_kt_hot=p.get("agn_kt_hot", 100.0),
            agn_r_warm_ratio=p.get("agn_r_warm_ratio", 2.0),
        )
        _agn_sed = agn_sed
        sed = sed + agn_sed

    # ── Radio emission (synchrotron from SF + AGN jets) ───────────────
    if model._uses_radio:
        radio_sed = radio_emission(
            wave_z2,
            L_ir=_L_ir,
            L_agn_bol=agn_bol_erg,
            q_ir=p.get("radio_q_ir", 2.64),
            alpha_sf=p.get("radio_alpha_sf", 0.8),
            radio_loudness=p.get("radio_loudness", 0.0),
            alpha_agn=p.get("radio_alpha_agn", 0.7),
            sfr_mode=model._radio_sfr_mode,
            log_mstar=float(jnp.log10(jnp.maximum(_mstar, 1e4))),
            redshift=float(getattr(model, "_redshift", 0.0)),
            include_freefree=model._radio_include_freefree,
            T_e=p.get("radio_T_e", 1e4),
            alpha_ff=p.get("radio_alpha_ff", -0.1),
        )
        _radio_sed = radio_sed
        sed = sed + radio_sed

    # ── X-ray emission (XRBs + AGN corona) ────────────────────────────
    if model._uses_xray:
        xray_sed = xray_emission(
            wave_z2,
            sfr=_sfr_cached,
            stellar_mass=_mstar,
            L_agn_bol=agn_bol_erg,
            gamma_agn=p.get("xray_gamma_agn", 1.8),
            alpha_ox=p.get("xray_alpha_ox", -1.4),
            gamma_hmxb=p.get("xray_gamma_hmxb", 2.0),
            gamma_lmxb=p.get("xray_gamma_lmxb", 1.6),
            E_cut=p.get("xray_E_cut", 300.0),
        )
        _xray_sed = xray_sed
        sed = sed + xray_sed

    return {
        "sed_total": sed,
        "rest_wavelength": rest_wave_target,
        "sed_attenuated": sed_attenuated,
        "sed_intrinsic": sed_intrinsic,
        "ssp_flux_at_z": ssp_flux_at_z,
        "weights": weights,
        "sfr": sfr,
        "p": p,
        "agn_bol_erg": agn_bol_erg,
        "mstar_formed": _mstar_formed,
        "mstar_surviving": _mstar_surviving,
        # Component SEDs (all erg/s/Hz on rest_wavelength grid)
        "sed_nebular": _neb_sed,
        "sed_shock": _shock_sed,
        "sed_dust_ir": _dust_ir_sed,
        "sed_agn": _agn_sed,
        "sed_radio": _radio_sed,
        "sed_xray": _xray_sed,
    }
