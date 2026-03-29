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

from tengri.models.dust.attenuation import single_component_dust_fast, two_component_dust_fast
from tengri.models.sps.dsps_wrapper import (
    compute_csp_sed,
    compute_csp_weights,
    compute_log_z_evolving,
    effective_metallicity,
    has_alpha_grid,
    interpolate_met_alpha,
    interpolate_met_alpha_evolving,
    interpolate_metallicity,
    interpolate_metallicity_evolving,
    interpolate_metallicity_smooth,
    interpolate_metallicity_smooth_evolving,
)


def interp_metallicity(model, log_z):
    """Dispatch metallicity interpolation (single Z value).

    Parameters
    ----------
    model : Model
        The tengri Model instance.
    log_z : float
        Log10 absolute metallicity.

    Returns
    -------
    array, shape (n_age, n_wave)
        SSP flux interpolated to target metallicity.
    """
    if model._met_interp == "smooth":
        return interpolate_metallicity_smooth(
            model.ssp_data.ssp_flux,
            model.ssp_data.ssp_lgmet,
            log_z,
            model._lgmet_scatter,
        )
    return interpolate_metallicity(model.ssp_data.ssp_flux, model.ssp_data.ssp_lgmet, log_z)


def interp_metallicity_evolving(model, log_z_per_age):
    """Dispatch evolving metallicity interpolation (per-age Z).

    Parameters
    ----------
    model : Model
        The tengri Model instance.
    log_z_per_age : array, shape (n_age,)
        Log10 absolute metallicity at each SSP age bin.

    Returns
    -------
    array, shape (n_age, n_wave)
        SSP flux with age-dependent metallicity.
    """
    if model._met_interp == "smooth":
        return interpolate_metallicity_smooth_evolving(
            model.ssp_data.ssp_flux,
            model.ssp_data.ssp_lgmet,
            log_z_per_age,
            model._lgmet_scatter,
        )
    return interpolate_metallicity_evolving(
        model.ssp_data.ssp_flux, model.ssp_data.ssp_lgmet, log_z_per_age
    )


def interp_met_alpha_dispatch(model, log_z, alpha_fe):
    """Dispatch metallicity+alpha interpolation based on SSP grid dimensionality.

    When 4D alpha-enhanced SSPs are loaded, uses proper bilinear (Z, [α/Fe])
    interpolation. Otherwise falls back to effective_metallicity approximation
    on the 3D grid.

    Parameters
    ----------
    model : Model
        The tengri Model instance.
    log_z : float
        Iron abundance [Fe/H] (or log10(Z/Zsun) for 3D grids).
    alpha_fe : float
        Alpha-element enhancement [α/Fe] in dex.

    Returns
    -------
    array, shape (n_age, n_wave)
    """
    if has_alpha_grid(model.ssp_data):
        return interpolate_met_alpha(
            model.ssp_data.ssp_flux,
            model.ssp_data.ssp_lgmet,
            model.ssp_data.ssp_alpha_fe,
            log_z,
            alpha_fe,
        )
    # Fallback: effective_metallicity on 3D grid
    log_z_eff = effective_metallicity(log_z, alpha_fe)
    return interp_metallicity(model, log_z_eff)


def interp_met_alpha_evolving_dispatch(model, log_z_per_age, alpha_fe_per_age):
    """Dispatch per-age metallicity+alpha interpolation.

    When 4D alpha-enhanced SSPs are loaded, uses per-age bilinear (Z, [α/Fe])
    interpolation. Otherwise applies effective_metallicity per age bin and
    interpolates on the 3D grid.

    Parameters
    ----------
    model : Model
        The tengri Model instance.
    log_z_per_age : array, shape (n_age,)
        [Fe/H] at each SSP age bin.
    alpha_fe_per_age : array or float
        [α/Fe] at each SSP age bin (array for evolving, scalar for constant).

    Returns
    -------
    array, shape (n_age, n_wave)
    """
    if has_alpha_grid(model.ssp_data):
        # Broadcast scalar alpha_fe to per-age array
        if not hasattr(alpha_fe_per_age, "shape") or alpha_fe_per_age.ndim == 0:
            alpha_fe_per_age = jnp.full_like(log_z_per_age, alpha_fe_per_age)
        return interpolate_met_alpha_evolving(
            model.ssp_data.ssp_flux,
            model.ssp_data.ssp_lgmet,
            model.ssp_data.ssp_alpha_fe,
            log_z_per_age,
            alpha_fe_per_age,
        )
    # Fallback: effective_metallicity on 3D grid
    log_z_eff = effective_metallicity(log_z_per_age, alpha_fe_per_age)
    return interp_metallicity_evolving(model, log_z_eff)


def get_dust_kwargs(model, p):
    """Extract dust law + emission kwargs from internal params dict.

    Parameters
    ----------
    model : Model
        The tengri Model instance.
    p : dict
        Internal parameter dict from ``_get_internal_params()``.

    Returns
    -------
    dict
        Keyword arguments for fused dust kernels.
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
    """Compute dust attenuation array, dispatching on dust model type.

    Returns shape ``(n_ages, n_wave)`` for both single- and two-component.
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
    from tengri.models.dust.attenuation import precompute_dust_age_weights

    if model._dust_age_weights is not None:
        dust_age_w = model._dust_age_weights.astype(wave_dt.dtype)
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
    """Extract AGN kwargs from internal params dict for fused kernel.

    Parameters
    ----------
    model : Model
        The tengri Model instance.
    p : dict
        Internal parameter dict from ``_get_internal_params()``.

    Returns
    -------
    dict
        Keyword arguments for fused AGN kernel (empty if AGN disabled).
    """
    if not (model._agn_model is not None and model._agn_parametric):
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


def compute_sed_components(model, params, _sfr=None, _weights=None, need_intrinsic=False):
    """Compute all SED intermediates.

    This is the shared computation engine behind ``predict_sed()``,
    ``predict_sed_quantities()``, and the lazy ``Prediction`` object.
    By returning all intermediates, downstream code can compute derived
    quantities without re-running the forward model.

    Parameters
    ----------
    model : Model
        The tengri Model instance.
    params : dict
        Parameter values (public names).
    _sfr : array, optional
        Pre-computed SFR on the log-age grid (avoids recomputation
        when called from ``predict_derived``).
    _weights : array, optional
        Pre-computed CSP weights.
    need_intrinsic : bool
        If True, always compute the unattenuated stellar SED even
        when no dust emission model is enabled. Required for
        ``l_dust_absorbed`` and intrinsic FUV/NUV.

    Returns
    -------
    dict with keys:
        ``"sed_total"`` : array (n_wave,) -- final rest-frame SED
        ``"sed_attenuated"`` : array (n_wave,) -- dust-attenuated stellar SED
        ``"sed_intrinsic"`` : array (n_wave,) or None -- unattenuated stellar SED
        ``"ssp_flux_at_z"`` : array (n_age, n_wave) -- Z-interpolated SSP
        ``"weights"`` : array (n_age,) -- CSP mass weights
        ``"sfr"`` : array (n_grid,) -- SFR on log-age grid
        ``"p"`` : dict -- internal parameter dict
        ``"agn_bol_erg"`` : float -- AGN bolometric luminosity (erg/s)
    """
    p = model._get_internal_params(params)

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

    if _weights is not None:
        weights = _weights
        _use_dsps_table = False
    elif "sfh_t_gyr" in params:
        weights = compute_csp_weights(sfr_on_ssp, model.ssp_ages_yr)
    else:
        sfr_on_ssp = jnp.interp(model.ssp_log_ages_yr, model.log_age_grid, sfr)
        weights = compute_csp_weights(sfr_on_ssp, model.ssp_ages_yr)
        _use_dsps_table = False

    # Alpha-element enhancement
    # When 4D alpha-enhanced SSPs are loaded, proper bilinear (Z, [α/Fe])
    # interpolation is used. Otherwise falls back to effective_metallicity.
    _has_alpha = has_alpha_grid(model.ssp_data)

    # Resolve [α/Fe]: either global scalar or per-age evolving ramp
    _alpha_evolving = getattr(model, "_alpha_fe_evolving", False)
    if _alpha_evolving:
        from tengri.models.sps.dsps_wrapper import compute_alpha_fe_evolving

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
        # Use DSPS calc_age_weights for proper trapezoidal integration
        # of SFH within each SSP age bin (Hearin+2023 Eq. 9).
        # Then our standard interpolate_metallicity + compute_csp_sed.
        # This gives ~1-2% accuracy at observable wavelengths in 0.5 ms.
        from dsps.sed.ssp_weights import calc_age_weights_from_sfh_table

        dsps_age_w = calc_age_weights_from_sfh_table(
            gal_t_table=t_cosmic_gyr,
            gal_sfr_table=sfr_table,
            ssp_lg_age_gyr=model.ssp_data.ssp_lg_age_gyr,
            t_obs=t_obs_gyr,
        )
        # DSPS returns normalized weights (sum=1); scale to absolute mass
        _total_mass_formed = jnp.trapezoid(sfr_table, t_cosmic_gyr * 1e9)
        weights = dsps_age_w * _total_mass_formed  # (n_age,) Msun

        # Metallicity + alpha: dispatch to 4D bilinear or 3D + effective_met
        log_z_solar = p.get("log_z_abs", -1.8477)
        ssp_flux_at_z = interp_met_alpha_dispatch(model, log_z_solar, alpha_fe)

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
        ssp_flux_at_z = interp_met_alpha_evolving_dispatch(model, log_z_abs, alpha_fe)
    elif model._evolving_metallicity:
        z = p.get("redshift", 0.0)
        t_universe_gyr = model._t_universe_gyr(z)
        log_z_per_age = compute_log_z_evolving(
            model.ssp_data.ssp_lg_age_gyr,
            p["log_z_abs_initial"],
            p["log_z_abs_final"],
            t_universe_gyr,
        )
        ssp_flux_at_z = interp_met_alpha_evolving_dispatch(model, log_z_per_age, alpha_fe)
    elif getattr(model, "_chem_evol_enabled", False):
        # Chemical evolution: derive Z(t) from SFH via gas-regulator model
        from tengri.models.sfh.chemical_evolution import chem_evol_metallicity_on_ssp_grid

        log_z_per_age = chem_evol_metallicity_on_ssp_grid(
            model.ssp_log_ages_yr,
            model.log_age_grid,
            sfr,
            yield_y=p.get("chem_yield", 0.03),
            eta_outflow=p.get("chem_eta_outflow", 0.0),
            f_gas_init=p.get("chem_f_gas_init", 0.9),
            return_frac=p.get("chem_return_frac", 0.4),
        )
        ssp_flux_at_z = interp_met_alpha_evolving_dispatch(model, log_z_per_age, alpha_fe)
    else:
        ssp_flux_at_z = interp_met_alpha_dispatch(model, p["log_z_abs"], alpha_fe)

        # --- Fast JIT path: dust + einsum in one compiled kernel ---
        # Eliminates ~78% Python dispatch overhead (4-14x speedup).
        if (
            _dsps_weights_2d is None
            and not model._evolving_metallicity
            and model._jit_exact_sed is not None
        ):
            sed_attenuated, sed_intrinsic_jit = model._jit_exact_sed(
                weights,
                ssp_flux_at_z,
                p["tau_bc"],
                p["tau_diff"],
                n_slope=p.get("dust_slope", -0.7),
                dust_bump_strength=p.get("dust_bump_strength", 0.0),
                dust_delta=p.get("dust_delta", 0.0),
                dust_Rv=p.get("dust_Rv", 3.1),
                f_obscuration=p.get("f_obscuration", 0.0),
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

        _LSUN = 3.828e33
        sed_attenuated = (
            _LSUN
            * jnp.einsum(
                "ma,aw,maw->w",
                _dsps_weights_2d,
                dust_atten,
                model.ssp_data.ssp_flux.astype(dt),
            )
        ).astype(jnp.float64)

        sed_intrinsic = None
        if model._dust_emission_model is not None or need_intrinsic:
            sed_intrinsic = (
                _LSUN
                * jnp.einsum(
                    "ma,maw->w",
                    _dsps_weights_2d,
                    model.ssp_data.ssp_flux.astype(dt),
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

    # Nebular emission (if backend provides it)
    if model._nebular_backend is not None and model._nebular_backend.has_free_params:
        neb_sed = model._nebular_backend.predict_nebular_sed(
            ssp_weights=weights,
            ssp_wave=model.ssp_data.ssp_wave,
            ssp_log_ages_yr=model.ssp_log_ages_yr,
            log_z=p["log_z_abs"],
            neb_logU=p.get("neb_logU", -3.0),
            neb_logZ_gas=p.get("neb_logZ_gas", None),
            neb_fesc=p.get("neb_fesc", 0.0),
            neb_fesc_lya=p.get("neb_fesc_lya", 0.0),
        )
        sed = sed + neb_sed

    # Shock emission (MAPPINGS V — Allen+2008)
    # Mix: replace shock_frac of nebular Halpha with shock emission lines
    if getattr(model, "_shock_enabled", False):
        from tengri.models.nebular.shock import shock_emission_sed

        shock_frac = p.get("shock_frac", 0.0)
        shock_velocity = p.get("shock_velocity", 300.0)
        shock_log_density = p.get("shock_log_density", 0.0)

        # Estimate L_halpha from the SED near 6563 A as a proxy.
        # Use a fraction of the bolometric luminosity scaled by shock_frac.
        _c_aa_shock = 2.99792458e18  # c in Angstrom/s
        _nu_shock = _c_aa_shock / model.ssp_data.ssp_wave
        l_bol = -jnp.trapezoid(sed, _nu_shock)
        # Approximate: L_Halpha ~ 1e-3 * L_bol for a star-forming galaxy
        l_halpha_approx = jnp.maximum(l_bol * 1e-3, 1e-30)
        l_shock_halpha = shock_frac * l_halpha_approx

        shock_sed = shock_emission_sed(
            model.ssp_data.ssp_wave,
            shock_velocity,
            l_shock_halpha,
            shock_log_density=shock_log_density,
        )
        sed = sed + shock_sed

    # Dust IR emission (energy-balanced)
    if model._dust_emission_model is not None and sed_intrinsic is not None:
        from tengri.models.dust.emission import get_emission_model

        _c_aa_em = 2.99792458e18  # c in Angstrom/s
        nu_em = _c_aa_em / model.ssp_data.ssp_wave
        L_absorbed = -jnp.trapezoid(sed_intrinsic - sed_attenuated, nu_em)
        eta_balance = p.get("dust_eta_balance", 1.0)
        L_ir = jnp.maximum(L_absorbed * eta_balance, 0.0)
        dust_ir = get_emission_model(model._dust_emission_model)(
            model.ssp_data.ssp_wave,
            L_ir,
            dust_T=p.get("dust_T", 35.0),
            dust_beta_ir=p.get("dust_beta_ir", 1.6),
            dust_alpha_mir=p.get("dust_alpha_mir", 2.0),
            dust_alpha_dale=p.get("dust_alpha_dale", 2.0),
            dust_umin=p.get("dust_umin", 1.0),
            dust_gamma_dl=p.get("dust_gamma_dl", 0.01),
            dust_qpah=p.get("dust_qpah", 2.5),
        )
        sed = sed + dust_ir

    # AGN contribution
    agn_bol_erg = 0.0
    if model._agn_model is not None:
        from tengri.models.agn import get_agn_model

        if model._agn_parametric:
            agn_log_lbol = p.get("agn_log_lbol", 10.0)
            agn_frac_for_model = 1.0
            agn_bol_erg = 10.0**agn_log_lbol
        else:
            agn_frac_for_model = p.get("agn_frac", 0.0)
            _c_aa = 2.99792458e18
            nu = _c_aa / model.ssp_data.ssp_wave
            L_bol_stellar = -jnp.trapezoid(sed, nu)
            agn_log_lbol = jnp.log10(jnp.maximum(L_bol_stellar * agn_frac_for_model, 1e-50))
            agn_bol_erg = L_bol_stellar * agn_frac_for_model
        agn_sed = get_agn_model(model._agn_model)(
            model.ssp_data.ssp_wave,
            agn_log_lbol=agn_log_lbol,
            agn_frac=agn_frac_for_model,
            agn_alpha=p.get("agn_alpha", -1.0),
            agn_T_torus=p.get("agn_T_torus", 1000.0),
            agn_tau_torus=p.get("agn_tau_torus", 5.0),
            agn_torus_frac=p.get("agn_torus_frac", 0.5),
            agn_log_mbh=p.get("agn_log_mbh", 7.0),
            agn_log_ledd=p.get("agn_log_ledd", -1.0),
        )
        sed = sed + agn_sed

    # Populate cached physical quantities for radio/X-ray models
    # L_ir: total IR luminosity from dust energy balance (Lsun)
    _L_ir = L_ir if "L_ir" in dir() else p.get("_L_ir_cached", 0.0)
    # SFR: instantaneous SFR at z_obs (Msun/yr) from the SFH
    _sfr_cached = sfr_table[-1] if "sfr_table" in dir() else p.get("_sfr_cached", 1.0)
    # M*: total stellar mass formed (Msun)
    _mstar = jnp.sum(weights) if weights is not None else p.get("_mstar_cached", 1e10)

    # Radio emission (synchrotron from SF + AGN jets)
    if model._radio_enabled:
        from tengri.models.radio import radio_total
        radio_sed = radio_total(
            model.ssp_data.ssp_wave,
            L_ir=_L_ir,
            L_agn_bol=agn_bol_erg,
            q_ir=p.get("radio_q_ir", 2.64),
            alpha_sf=p.get("radio_alpha_sf", 0.8),
            radio_loudness=p.get("radio_loudness", 0.0),
            alpha_agn=p.get("radio_alpha_agn", 0.7),
        )
        sed = sed + radio_sed

    # X-ray emission (XRBs + AGN corona)
    if model._xray_enabled:
        from tengri.models.xray import xray_total

        # Use pipeline-computed SFR and M* (populated above)
        xray_sed = xray_total(
            model.ssp_data.ssp_wave,
            sfr=_sfr_cached,
            stellar_mass=_mstar,
            L_agn_bol=agn_bol_erg,
            gamma_agn=p.get("xray_gamma_agn", 1.8),
            alpha_ox=p.get("xray_alpha_ox", -1.4),
        )
        sed = sed + xray_sed

    return {
        "sed_total": sed,
        "sed_attenuated": sed_attenuated,
        "sed_intrinsic": sed_intrinsic,
        "ssp_flux_at_z": ssp_flux_at_z,
        "weights": weights,
        "sfr": sfr,
        "p": p,
        "agn_bol_erg": agn_bol_erg,
    }
