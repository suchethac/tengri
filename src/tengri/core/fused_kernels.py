"""Fused JIT kernel builders for fast photometry and spectroscopy.

These factory functions build @jax.jit closures that capture precomputed
arrays (SSP grids, dust weights, effective wavelengths) at build time.
The returned functions take only per-call parameters (SFR weights, dust
params) as arguments.

Extracted from ``SEDModel`` methods to keep model.py focused on orchestration.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

# -------------------------------------------------------------------
# Hybrid kernel: precomputed SSP + exact non-stellar
# -------------------------------------------------------------------


def build_hybrid_photometry(model):
    """Build hybrid photometry kernel: precomputed stellar + exact non-stellar.

    Stellar CSP evaluated via precomputed SSP×filter einsum (fast, ~0.4% error).
    Non-stellar components evaluated at full wavelength resolution via
    emission_helpers, then integrated through filters (exact).

    This kernel bridges Tier 1 (fast but approximate) and Tier 2 (exact but slow):
    - Use for science models where stellar photometry speed matters but non-stellar
      accuracy is critical (nebular lines, AGN variability, etc.)
    - Non-stellar components must be available for full-wavelength evaluation
      (emission_helpers functions)

    Parameters
    ----------
    model : SEDModel
        The model instance providing config and precomputed arrays.

    Returns
    -------
    callable
        JIT-compiled function: (sfr_on_ssp, log_z_abs, tau_bc, tau_diff,
        dust_slope, ..., neb_logU=..., shock_frac=..., etc.)
        -> photometry array (n_filters,) in erg/s/cm^2/Hz.
    """
    from tengri.core.emission_helpers import (
        agn_emission,
        attenuate_emission,
        dust_ir_emission,
        nebular_emission,
        radio_emission,
        shock_emission,
        xray_emission,
    )
    from tengri.models.dust.attenuation import resolve_dust_law
    from tengri.models.observation.photometry import compute_flux_density
    from tengri.models.sps.dsps_wrapper import LSUN_ERG_PER_S

    dt = model._forward_dtype
    precomp = model._precomputed.photometry
    ssp_phot = precomp.ssp_phot.astype(dt)
    ssp_lgmet = model.ssp_data.ssp_lgmet.astype(dt)
    eff_waves_rest = precomp.effective_wavelengths_rest.astype(dt)
    _use_taylor = precomp.ssp_phot_moment is not None
    if _use_taylor:
        ssp_phot_moment = precomp.ssp_phot_moment.astype(dt)
    _is_single_dust = model._dust_model == "single_component"
    _dust_exact = getattr(model, "_dust_approx", "fast") == "exact"
    if not _is_single_dust:
        if _dust_exact:
            dust_age_w = model._precomputed.dust_age_weights.astype(dt)
        else:
            _t_birth = 1e7
            young_mask = (model.ssp_ages_yr < _t_birth).astype(dt)
            old_mask = dt.type(1.0) - young_mask
    flux_scale = dt.type(precomp.flux_scale)
    _csp_use_matrix = model._csp_integration == "log_interp"
    if _csp_use_matrix:
        _csp_mat = model._csp_matrix.astype(dt)
    else:
        _age_dt = model._csp_age_dt.astype(dt)
    lsun = dt.type(LSUN_ERG_PER_S)

    # Voronoi frequency bandwidths for L_absorbed broadband estimate (Hz).
    # Without these weights, sum(L_ν) is dimensionally wrong and
    # catastrophically underestimates L_absorbed for panchromatic filter sets.
    _eff_bw = model._precomputed.effective_bandwidths_hz
    if _eff_bw is not None:
        _eff_bw = _eff_bw.astype(dt)

    # Capture dust law functions
    law_bc_fn = resolve_dust_law(model._dust_law_bc)
    if not _is_single_dust:
        law_diff_fn = resolve_dust_law(model._dust_law_diff)

    from tengri.models.sps.dsps_wrapper import (
        _ALPHA_TO_Z_COEFF as _A2Z,
        has_alpha_grid,
    )

    _has_alpha = has_alpha_grid(model.ssp_data)
    if _has_alpha:
        ssp_alpha_fe = model.ssp_data.ssp_alpha_fe.astype(dt)

    _use_alpha_fe = model.spec.alpha_fe_evolving or "met_alpha_fe" in model.spec.free_params

    _use_smooth_z = model._met_interp == "smooth"
    _lgmet_scat = dt.type(model._lgmet_scatter)
    if _use_smooth_z:
        from tengri.models.sps.dsps_wrapper import compute_lgmet_weights as _clw

    # IGM: precomputed at effective wavelengths
    has_igm = model._precomputed.igm_at_effective_wavelengths is not None
    if has_igm:
        igm_trans = model._precomputed.igm_at_effective_wavelengths.astype(dt)

    # Note: has_dust_em flag checked for stellar L_absorbed only; full-wavelength
    # dust emission happens in non-stellar section
    # AGN at effective wavelengths (parametric mode only) — not used in hybrid
    # since non-stellar AGN is evaluated at full wavelength

    # === Non-stellar components (full wavelength) ===
    ssp_wave_f64 = model.ssp_data.ssp_wave
    rest_wave_f64 = model._rest_wavelength
    _needs_extension = rest_wave_f64 is not model.ssp_data.ssp_wave

    # Nebular
    has_nebular = model._nebular_backend is not None and getattr(
        model._nebular_backend, "has_free_params", False
    )
    if has_nebular:
        nebular_backend = model._nebular_backend
        ssp_log_ages_yr = model.ssp_log_ages_yr
        _neb_dust_mode = getattr(model, "_neb_dust", "bc")
        _neb_bc_fn = getattr(model, "_neb_dust_law_bc_fn", law_bc_fn)

    # Shock
    has_shock = getattr(model, "_shock_enabled", False)

    # Dust emission (full wavelength)
    has_dust_em_full = model._dust_emission_model is not None
    if has_dust_em_full:
        from tengri.models.dust.emission import resolve_emission_model

        dust_emission_fn = resolve_emission_model(model._dust_emission_model)

    # AGN (full wavelength)
    has_agn_full = model._agn_model is not None
    agn_parametric = model._agn_parametric if has_agn_full else False
    if has_agn_full:
        from tengri.models.agn import resolve_agn_model

        agn_model_fn_full = resolve_agn_model(model._agn_model)

    # Radio
    has_radio = model._radio_enabled
    if has_radio:
        _radio_sfr_mode = model._radio_sfr_mode
        _include_freefree = model._radio_include_freefree
        _redshift = float(getattr(model, "_redshift", 0.0))

    # X-ray
    has_xray = model._xray_enabled

    # Constants for energy balance
    _c_aa = dt.type(2.99792458e18)

    # Redshift for filter integration
    z_fixed = model._z_fixed
    dl_cm_fixed = model._dl_cm_fixed

    # Filter information
    n_filters = len(model.filter_waves) if model.filter_waves else 0
    filter_waves_list = model.filter_waves if model.filter_waves else []
    filter_trans_list = model.filter_trans if model.filter_trans else []

    # === Define kernel signatures (single vs two-component dust) ===

    if _is_single_dust:

        @jax.jit
        def hybrid_phot(
            sfr_on_ssp,
            log_z_abs,
            tau_v,
            dust_slope,
            f_obscuration=0.0,
            dust_bump_strength=0.0,
            dust_delta=0.0,
            dust_Rv=3.1,
            alpha_fe=0.0,
            dust_T=35.0,
            dust_beta_ir=1.6,
            dust_eta_balance=1.0,
            agn_log_lbol=10.0,
            agn_alpha=-1.0,
            agn_T_torus=1000.0,
            agn_tau_torus=5.0,
            agn_torus_frac=0.5,
            agn_log_mbh=7.0,
            agn_log_ledd=-1.0,
            # Non-stellar parameters
            neb_logU=-3.0,
            neb_logZ_gas=None,
            neb_fesc=0.0,
            neb_fesc_lya=0.0,
            shock_frac=0.0,
            shock_velocity=300.0,
            shock_log_density=0.0,
            shock_b_over_sqrt_n=1.0,
            shock_abundance="solar",
            shock_component="combined",
            dust_alpha_mir=2.0,
            dust_alpha_dale=2.0,
            dust_umin=1.0,
            dust_gamma_dl=0.01,
            dust_qpah=2.5,
            agn_polar_ebv=0.0,
            agn_cos_inc=0.5,
            agn_polar_oa=45.0,
            agn_a_spin=0.0,
            agn_tau_skirtor=7.0,
            agn_p_skirtor=1.0,
            agn_q_skirtor=1.0,
            agn_oa_skirtor=40.0,
            agn_frac=0.0,
            agn_T_hot=1200.0,
            agn_T_warm=300.0,
            agn_frac_hot=0.3,
            agn_f_hard=0.02,
            agn_gamma_warm=2.5,
            agn_kt_warm=0.2,
            agn_gamma_hard=1.8,
            agn_kt_hot=100.0,
            agn_r_warm_ratio=2.0,
            radio_q_ir=2.64,
            radio_alpha_sf=0.8,
            radio_loudness=0.0,
            radio_alpha_agn=0.7,
            radio_T_e=1e4,
            radio_alpha_ff=-0.1,
            xray_gamma_agn=1.8,
            xray_alpha_ox=-1.4,
            xray_gamma_hmxb=2.0,
            xray_gamma_lmxb=1.6,
            xray_E_cut=300.0,
        ):
            return _hybrid_phot_body(
                sfr_on_ssp,
                log_z_abs,
                0.0,
                0.0,
                dust_slope,
                f_obscuration,
                dust_bump_strength,
                dust_delta,
                dust_Rv,
                alpha_fe,
                dust_T,
                dust_beta_ir,
                dust_eta_balance,
                agn_log_lbol,
                agn_alpha,
                agn_T_torus,
                agn_tau_torus,
                agn_torus_frac,
                agn_log_mbh,
                agn_log_ledd,
                neb_logU,
                neb_logZ_gas,
                neb_fesc,
                neb_fesc_lya,
                shock_frac,
                shock_velocity,
                shock_log_density,
                shock_b_over_sqrt_n,
                shock_abundance,
                shock_component,
                dust_alpha_mir,
                dust_alpha_dale,
                dust_umin,
                dust_gamma_dl,
                dust_qpah,
                agn_polar_ebv,
                agn_cos_inc,
                agn_polar_oa,
                agn_frac,
                agn_a_spin,
                agn_tau_skirtor,
                agn_p_skirtor,
                agn_q_skirtor,
                agn_oa_skirtor,
                agn_T_hot,
                agn_T_warm,
                agn_frac_hot,
                agn_f_hard,
                agn_gamma_warm,
                agn_kt_warm,
                agn_gamma_hard,
                agn_kt_hot,
                agn_r_warm_ratio,
                radio_q_ir,
                radio_alpha_sf,
                radio_loudness,
                radio_alpha_agn,
                radio_T_e,
                radio_alpha_ff,
                xray_gamma_agn,
                xray_alpha_ox,
                xray_gamma_hmxb,
                xray_gamma_lmxb,
                xray_E_cut,
                tau_v=tau_v,
            )

    else:

        @jax.jit
        def hybrid_phot(
            sfr_on_ssp,
            log_z_abs,
            tau_bc,
            tau_diff,
            dust_slope,
            f_obscuration=0.0,
            dust_bump_strength=0.0,
            dust_delta=0.0,
            dust_Rv=3.1,
            alpha_fe=0.0,
            dust_T=35.0,
            dust_beta_ir=1.6,
            dust_eta_balance=1.0,
            agn_log_lbol=10.0,
            agn_alpha=-1.0,
            agn_T_torus=1000.0,
            agn_tau_torus=5.0,
            agn_torus_frac=0.5,
            agn_log_mbh=7.0,
            agn_log_ledd=-1.0,
            # Non-stellar parameters
            neb_logU=-3.0,
            neb_logZ_gas=None,
            neb_fesc=0.0,
            neb_fesc_lya=0.0,
            shock_frac=0.0,
            shock_velocity=300.0,
            shock_log_density=0.0,
            shock_b_over_sqrt_n=1.0,
            shock_abundance="solar",
            shock_component="combined",
            dust_alpha_mir=2.0,
            dust_alpha_dale=2.0,
            dust_umin=1.0,
            dust_gamma_dl=0.01,
            dust_qpah=2.5,
            agn_polar_ebv=0.0,
            agn_cos_inc=0.5,
            agn_polar_oa=45.0,
            agn_a_spin=0.0,
            agn_tau_skirtor=7.0,
            agn_p_skirtor=1.0,
            agn_q_skirtor=1.0,
            agn_oa_skirtor=40.0,
            agn_frac=0.0,
            agn_T_hot=1200.0,
            agn_T_warm=300.0,
            agn_frac_hot=0.3,
            agn_f_hard=0.02,
            agn_gamma_warm=2.5,
            agn_kt_warm=0.2,
            agn_gamma_hard=1.8,
            agn_kt_hot=100.0,
            agn_r_warm_ratio=2.0,
            radio_q_ir=2.64,
            radio_alpha_sf=0.8,
            radio_loudness=0.0,
            radio_alpha_agn=0.7,
            radio_T_e=1e4,
            radio_alpha_ff=-0.1,
            xray_gamma_agn=1.8,
            xray_alpha_ox=-1.4,
            xray_gamma_hmxb=2.0,
            xray_gamma_lmxb=1.6,
            xray_E_cut=300.0,
        ):
            return _hybrid_phot_body(
                sfr_on_ssp,
                log_z_abs,
                tau_bc,
                tau_diff,
                dust_slope,
                f_obscuration,
                dust_bump_strength,
                dust_delta,
                dust_Rv,
                alpha_fe,
                dust_T,
                dust_beta_ir,
                dust_eta_balance,
                agn_log_lbol,
                agn_alpha,
                agn_T_torus,
                agn_tau_torus,
                agn_torus_frac,
                agn_log_mbh,
                agn_log_ledd,
                neb_logU,
                neb_logZ_gas,
                neb_fesc,
                neb_fesc_lya,
                shock_frac,
                shock_velocity,
                shock_log_density,
                shock_b_over_sqrt_n,
                shock_abundance,
                shock_component,
                dust_alpha_mir,
                dust_alpha_dale,
                dust_umin,
                dust_gamma_dl,
                dust_qpah,
                agn_polar_ebv,
                agn_cos_inc,
                agn_polar_oa,
                agn_frac,
                agn_a_spin,
                agn_tau_skirtor,
                agn_p_skirtor,
                agn_q_skirtor,
                agn_oa_skirtor,
                agn_T_hot,
                agn_T_warm,
                agn_frac_hot,
                agn_f_hard,
                agn_gamma_warm,
                agn_kt_warm,
                agn_gamma_hard,
                agn_kt_hot,
                agn_r_warm_ratio,
                radio_q_ir,
                radio_alpha_sf,
                radio_loudness,
                radio_alpha_agn,
                radio_T_e,
                radio_alpha_ff,
                xray_gamma_agn,
                xray_alpha_ox,
                xray_gamma_hmxb,
                xray_gamma_lmxb,
                xray_E_cut,
            )

    def _hybrid_phot_body(
        sfr_on_ssp,
        log_z_abs,
        tau_bc,
        tau_diff,
        dust_slope,
        f_obscuration,
        dust_bump_strength,
        dust_delta,
        dust_Rv,
        alpha_fe,
        dust_T,
        dust_beta_ir,
        dust_eta_balance,
        agn_log_lbol,
        agn_alpha,
        agn_T_torus,
        agn_tau_torus,
        agn_torus_frac,
        agn_log_mbh,
        agn_log_ledd,
        neb_logU,
        neb_logZ_gas,
        neb_fesc,
        neb_fesc_lya,
        shock_frac,
        shock_velocity,
        shock_log_density,
        shock_b_over_sqrt_n,
        shock_abundance,
        shock_component,
        dust_alpha_mir,
        dust_alpha_dale,
        dust_umin,
        dust_gamma_dl,
        dust_qpah,
        agn_polar_ebv,
        agn_cos_inc,
        agn_polar_oa,
        agn_frac,
        agn_a_spin,
        agn_tau_skirtor,
        agn_p_skirtor,
        agn_q_skirtor,
        agn_oa_skirtor,
        agn_T_hot,
        agn_T_warm,
        agn_frac_hot,
        agn_f_hard,
        agn_gamma_warm,
        agn_kt_warm,
        agn_gamma_hard,
        agn_kt_hot,
        agn_r_warm_ratio,
        radio_q_ir,
        radio_alpha_sf,
        radio_loudness,
        radio_alpha_agn,
        radio_T_e,
        radio_alpha_ff,
        xray_gamma_agn,
        xray_alpha_ox,
        xray_gamma_hmxb,
        xray_gamma_lmxb,
        xray_E_cut,
        tau_v=0.0,
    ):
        """Hybrid kernel body: stellar (precomputed) + non-stellar (exact)."""
        # === STEP 1: Stellar photometry (COPIED from _fused_phot_body) ===
        sfr = sfr_on_ssp.astype(dt)
        lz = jnp.asarray(log_z_abs, dtype=dt)
        tv1 = jnp.asarray(tau_bc, dtype=dt)
        tv2 = jnp.asarray(tau_diff, dtype=dt)
        tv = jnp.asarray(tau_v, dtype=dt)
        dn = jnp.asarray(dust_slope, dtype=dt)
        f_obs = jnp.asarray(f_obscuration, dtype=dt)
        bump = jnp.asarray(dust_bump_strength, dtype=dt)
        delta = jnp.asarray(dust_delta, dtype=dt)
        rv = jnp.asarray(dust_Rv, dtype=dt)
        afe = jnp.asarray(alpha_fe, dtype=dt)

        # CSP weights
        weights = _csp_mat @ sfr if _csp_use_matrix else sfr * _age_dt

        # Metallicity + alpha interpolation
        if _has_alpha:
            lz_c = jnp.clip(lz, ssp_lgmet[0], ssp_lgmet[-1])
            iz = jnp.clip(jnp.searchsorted(ssp_lgmet, lz_c) - 1, 0, len(ssp_lgmet) - 2)
            fz = (lz_c - ssp_lgmet[iz]) / (ssp_lgmet[iz + 1] - ssp_lgmet[iz])
            afe_c = jnp.clip(afe, ssp_alpha_fe[0], ssp_alpha_fe[-1])
            ia = jnp.clip(jnp.searchsorted(ssp_alpha_fe, afe_c) - 1, 0, len(ssp_alpha_fe) - 2)
            fa = (afe_c - ssp_alpha_fe[ia]) / (ssp_alpha_fe[ia + 1] - ssp_alpha_fe[ia])
            ssp_at_z = (
                (1 - fz) * (1 - fa) * ssp_phot[iz, ia]
                + fz * (1 - fa) * ssp_phot[iz + 1, ia]
                + (1 - fz) * fa * ssp_phot[iz, ia + 1]
                + fz * fa * ssp_phot[iz + 1, ia + 1]
            )
        else:
            if _use_alpha_fe:
                lz = lz + _A2Z * afe
            if _use_smooth_z:
                zw = _clw(lz, ssp_lgmet, _lgmet_scat)
                ssp_at_z = jnp.einsum("m,maf->af", zw, ssp_phot)
            else:
                log_z_c = jnp.clip(lz, ssp_lgmet[0], ssp_lgmet[-1])
                idx = jnp.clip(jnp.searchsorted(ssp_lgmet, log_z_c) - 1, 0, len(ssp_lgmet) - 2)
                frac = (log_z_c - ssp_lgmet[idx]) / (ssp_lgmet[idx + 1] - ssp_lgmet[idx])
                ssp_at_z = (1.0 - frac) * ssp_phot[idx] + frac * ssp_phot[idx + 1]

        # Dust at effective wavelengths
        _law_kw = dict(n_slope=dn, dust_bump_strength=bump, dust_delta=delta, dust_Rv=rv)
        if _is_single_dust:
            k = law_bc_fn(eff_waves_rest, **_law_kw)
            trans_1d = f_obs + (1.0 - f_obs) * jnp.exp(-tv * k)
            flux_attenuated = jnp.einsum("i,if->f", weights, ssp_at_z) * trans_1d
        elif _dust_exact:
            k_bc = law_bc_fn(eff_waves_rest, **_law_kw)
            k_diff = law_diff_fn(eff_waves_rest, **_law_kw)
            tau = dust_age_w[:, None] * tv1 * k_bc[None, :] + tv2 * k_diff[None, :]
            dust = f_obs + (1.0 - f_obs) * jnp.exp(-tau)
            flux_attenuated = jnp.einsum("i,if,if->f", weights, dust, ssp_at_z)
        else:
            # Fast two-CSP decomposition
            k_bc = law_bc_fn(eff_waves_rest, **_law_kw)
            k_diff = law_diff_fn(eff_waves_rest, **_law_kw)
            trans_bc = jnp.exp(-tv1 * k_bc)
            trans_diff = jnp.exp(-tv2 * k_diff)

            csp_young = jnp.einsum("i,if->f", weights * young_mask, ssp_at_z)
            csp_old = jnp.einsum("i,if->f", weights * old_mask, ssp_at_z)

            if _use_taylor:
                _dl = dt.type(1.0)
                k_bc_p = law_bc_fn(eff_waves_rest + _dl, **_law_kw)
                k_bc_m = law_bc_fn(eff_waves_rest - _dl, **_law_kw)
                k_diff_p = law_diff_fn(eff_waves_rest + _dl, **_law_kw)
                k_diff_m = law_diff_fn(eff_waves_rest - _dl, **_law_kw)

                if _has_alpha:
                    ssp_moment_at_z = (
                        (1 - fz) * (1 - fa) * ssp_phot_moment[iz, ia]
                        + fz * (1 - fa) * ssp_phot_moment[iz + 1, ia]
                        + (1 - fz) * fa * ssp_phot_moment[iz, ia + 1]
                        + fz * fa * ssp_phot_moment[iz + 1, ia + 1]
                    )
                elif _use_smooth_z:
                    ssp_moment_at_z = jnp.einsum("m,maf->af", zw, ssp_phot_moment)
                else:
                    ssp_moment_at_z = (1.0 - frac) * ssp_phot_moment[idx] + frac * ssp_phot_moment[
                        idx + 1
                    ]

                csp_young_m = jnp.einsum("i,if->f", weights * young_mask, ssp_moment_at_z)
                csp_old_m = jnp.einsum("i,if->f", weights * old_mask, ssp_moment_at_z)

                A_young = trans_bc * trans_diff
                A_young_deriv = (
                    jnp.exp(-tv1 * k_bc_p) * jnp.exp(-tv2 * k_diff_p)
                    - jnp.exp(-tv1 * k_bc_m) * jnp.exp(-tv2 * k_diff_m)
                ) / (2 * _dl)
                A_old = trans_diff
                A_old_deriv = (jnp.exp(-tv2 * k_diff_p) - jnp.exp(-tv2 * k_diff_m)) / (2 * _dl)

                flux_no_geom = (
                    A_young * csp_young
                    + A_young_deriv * csp_young_m
                    + A_old * csp_old
                    + A_old_deriv * csp_old_m
                )
            else:
                flux_no_geom = trans_diff * (trans_bc * csp_young + csp_old)

            flux_intrinsic_for_geom = csp_young + csp_old
            flux_attenuated = f_obs * flux_intrinsic_for_geom + (1.0 - f_obs) * flux_no_geom

        # Estimate L_absorbed from stellar broadband.
        # Weight each filter by its Voronoi frequency bandwidth to convert
        # the sum of L_ν values into a proper ∫L_ν dν quadrature (erg/s).
        flux_intrinsic = jnp.einsum("i,if->f", weights, ssp_at_z)
        diff_flux = (flux_intrinsic - flux_attenuated) * lsun  # erg/s/Hz per band
        if _eff_bw is not None:
            L_absorbed_stellar = jnp.sum(diff_flux * _eff_bw)  # erg/s
        else:
            L_absorbed_stellar = jnp.sum(diff_flux)  # fallback (wrong units)
        L_absorbed_stellar = jnp.maximum(L_absorbed_stellar, dt.type(0.0))

        # === STEP 2: Non-stellar SED at full wavelength resolution ===
        non_stellar_sed = jnp.zeros_like(ssp_wave_f64)
        L_abs_neb = jnp.float64(0.0)

        # 2a: Nebular emission
        if has_nebular:
            _sfr_last = weights[-1]
            neb_raw = nebular_emission(
                nebular_backend,
                weights,
                ssp_wave_f64,
                ssp_log_ages_yr,
                jnp.float64(log_z_abs),
                _sfr_last,
                neb_logU=neb_logU,
                neb_logZ_gas=neb_logZ_gas,
                neb_fesc=neb_fesc,
                neb_fesc_lya=neb_fesc_lya,
            )
            _tau_bc_neb = tau_bc if not _is_single_dust else tau_v
            _tau_diff_neb = tau_diff if not _is_single_dust else jnp.float64(0.0)
            _dust_kw_neb = {
                "dust_slope": jnp.float64(dust_slope),
                "dust_bump_strength": jnp.float64(dust_bump_strength),
            }
            neb_sed, L_abs_neb = attenuate_emission(
                neb_raw,
                ssp_wave_f64,
                _neb_dust_mode,
                _tau_bc_neb,
                _tau_diff_neb,
                law_bc_fn,
                law_diff_fn if not _is_single_dust else law_bc_fn,
                neb_bc_fn=_neb_bc_fn,
                **_dust_kw_neb,
            )
            non_stellar_sed = non_stellar_sed + neb_sed

        # 2b: Shock emission
        if has_shock:
            shock_raw = shock_emission(
                ssp_wave_f64,
                non_stellar_sed,
                shock_frac=shock_frac,
                shock_velocity=shock_velocity,
                shock_log_density=shock_log_density,
                shock_b_over_sqrt_n=shock_b_over_sqrt_n,
                shock_abundance=shock_abundance,
                shock_component=shock_component,
            )
            _tau_diff_shock = tau_diff if not _is_single_dust else jnp.float64(0.0)
            _dust_kw_shock = {
                "dust_slope": jnp.float64(dust_slope),
                "dust_bump_strength": jnp.float64(dust_bump_strength),
            }
            shock_sed, _ = attenuate_emission(
                shock_raw,
                ssp_wave_f64,
                "diff",
                jnp.float64(0.0),
                _tau_diff_shock,
                law_bc_fn,
                law_diff_fn if not _is_single_dust else law_bc_fn,
                **_dust_kw_shock,
            )
            non_stellar_sed = non_stellar_sed + shock_sed

        # 2c: Dust IR emission (energy-balanced)
        if has_dust_em_full:
            L_ir = jnp.maximum(
                (L_absorbed_stellar + L_abs_neb) * jnp.float64(dust_eta_balance), 0.0
            )
            dust_ir = dust_ir_emission(
                dust_emission_fn,
                rest_wave_f64,
                L_ir,
                dust_T=jnp.float64(dust_T),
                dust_beta_ir=jnp.float64(dust_beta_ir),
                dust_alpha_mir=jnp.float64(dust_alpha_mir),
                dust_alpha_dale=jnp.float64(dust_alpha_dale),
                dust_umin=jnp.float64(dust_umin),
                dust_gamma_dl=jnp.float64(dust_gamma_dl),
                dust_qpah=jnp.float64(dust_qpah),
            )
            # Interpolate to panchromatic grid if needed
            if _needs_extension:
                from tengri.utils.wavelength import interpolate_sed_to_grid

                dust_ir = interpolate_sed_to_grid(rest_wave_f64, dust_ir, rest_wave_f64)
            non_stellar_sed = non_stellar_sed + dust_ir
        else:
            L_ir = jnp.float64(0.0)

        # 2d: AGN emission
        if has_agn_full:
            if agn_parametric:
                _agn_lbol = agn_log_lbol
                _agn_frac = 1.0
            else:
                _agn_frac = jnp.float64(0.0)  # Not parametric in hybrid
                _agn_lbol = 10.0
            agn_sed = agn_emission(
                agn_model_fn_full,
                rest_wave_f64,
                agn_log_lbol=_agn_lbol,
                agn_frac=_agn_frac,
                agn_polar_ebv=jnp.float64(agn_polar_ebv),
                agn_cos_inc=jnp.float64(agn_cos_inc),
                agn_polar_oa=jnp.float64(agn_polar_oa),
                agn_alpha=jnp.float64(agn_alpha),
                agn_T_torus=jnp.float64(agn_T_torus),
                agn_tau_torus=jnp.float64(agn_tau_torus),
                agn_torus_frac=jnp.float64(agn_torus_frac),
                agn_log_mbh=jnp.float64(agn_log_mbh),
                agn_log_ledd=jnp.float64(agn_log_ledd),
                agn_a_spin=jnp.float64(agn_a_spin),
                agn_tau_skirtor=jnp.float64(agn_tau_skirtor),
                agn_p_skirtor=jnp.float64(agn_p_skirtor),
                agn_q_skirtor=jnp.float64(agn_q_skirtor),
                agn_oa_skirtor=jnp.float64(agn_oa_skirtor),
                agn_T_hot=jnp.float64(agn_T_hot),
                agn_T_warm=jnp.float64(agn_T_warm),
                agn_frac_hot=jnp.float64(agn_frac_hot),
                agn_f_hard=jnp.float64(agn_f_hard),
                agn_gamma_warm=jnp.float64(agn_gamma_warm),
                agn_kt_warm=jnp.float64(agn_kt_warm),
                agn_gamma_hard=jnp.float64(agn_gamma_hard),
                agn_kt_hot=jnp.float64(agn_kt_hot),
                agn_r_warm_ratio=jnp.float64(agn_r_warm_ratio),
            )
            non_stellar_sed = non_stellar_sed + agn_sed

        # 2e: Radio emission
        if has_radio:
            _L_ir = L_ir
            _agn_bol = (
                10.0 ** (jnp.float64(agn_log_lbol)) * LSUN_ERG_PER_S if has_agn_full else 0.0
            )
            _log_mstar = jnp.log10(jnp.maximum(jnp.sum(weights), 1e-10))
            radio_sed = radio_emission(
                rest_wave_f64,
                L_ir=_L_ir,
                L_agn_bol=_agn_bol,
                q_ir=jnp.float64(radio_q_ir),
                alpha_sf=jnp.float64(radio_alpha_sf),
                radio_loudness=jnp.float64(radio_loudness),
                alpha_agn=jnp.float64(radio_alpha_agn),
                sfr_mode=_radio_sfr_mode,
                log_mstar=_log_mstar,
                redshift=_redshift,
                include_freefree=_include_freefree,
                T_e=jnp.float64(radio_T_e),
                alpha_ff=jnp.float64(radio_alpha_ff),
            )
            non_stellar_sed = non_stellar_sed + radio_sed

        # 2f: X-ray emission
        if has_xray:
            sfr_now = weights[-1]
            mstar = jnp.sum(weights)
            _agn_bol_xray = (
                10.0 ** (jnp.float64(agn_log_lbol)) * LSUN_ERG_PER_S if has_agn_full else 0.0
            )
            xray_sed = xray_emission(
                rest_wave_f64,
                sfr=sfr_now,
                stellar_mass=mstar,
                L_agn_bol=_agn_bol_xray,
                gamma_agn=jnp.float64(xray_gamma_agn),
                alpha_ox=jnp.float64(xray_alpha_ox),
                gamma_hmxb=jnp.float64(xray_gamma_hmxb),
                gamma_lmxb=jnp.float64(xray_gamma_lmxb),
                E_cut=jnp.float64(xray_E_cut),
            )
            non_stellar_sed = non_stellar_sed + xray_sed

        # === STEP 3: Integrate non-stellar through filters ===
        non_stellar_phot = jnp.zeros(n_filters, dtype=jnp.float64)

        # Loop over filters (unrolled by JAX tracer)
        ns_fluxes = []
        for fw, ft in zip(filter_waves_list, filter_trans_list):
            f = compute_flux_density(non_stellar_sed, ssp_wave_f64, fw, ft, z_fixed, dl_cm_fixed)
            ns_fluxes.append(f)
        if n_filters > 0:
            non_stellar_phot = jnp.array(ns_fluxes)

        # === STEP 4: Combine stellar + non-stellar ===
        stellar_phot = flux_attenuated
        if has_igm:
            stellar_phot = stellar_phot * igm_trans

        # Scale stellar to erg/s/cm^2/Hz
        stellar_phot = (flux_scale * stellar_phot * lsun).astype(jnp.float64)

        return stellar_phot + non_stellar_phot

    return hybrid_phot


# Exact-path SED kernel
# -------------------------------------------------------------------


def build_exact_sed(model):
    """Build a JIT-compiled function for exact-path dust + CSP SED.

    Without JIT, the exact path dispatches ~15 JAX operations through
    Python individually.  Each dispatch costs ~100-300 us -- totalling
    ~78% of the measured dust cost.  This wraps dust curve evaluation,
    age-dependent attenuation, and the CSP einsum in a single
    ``@jax.jit`` scope, eliminating Python dispatch overhead and
    enabling XLA kernel fusion (exp + einsum in one kernel).

    Optimizations baked in:

    - **Mixed precision**: all intermediates use ``forward_dtype``
      (halves memory traffic when float32).
    - **Duplicate law skip**: when ``law_bc == law_diff`` (common
      Charlot & Fall case), the curve is evaluated once, not twice.
    - **Fused dust + einsum**: XLA can fuse ``exp(-tau)`` into the
      downstream ``einsum("i,iw,iw->w")``, avoiding a full
      ``(n_age, n_wave)`` materialization.

    Parameters
    ----------
    model : SEDModel
        The model instance providing config and precomputed arrays.

    Returns
    -------
    callable
        JIT-compiled function: (weights, ssp_at_z, tau_bc, tau_diff,
        ...) -> (sed_atten, sed_intr).

    Notes
    -----
    Typical speedup: 4-14x vs un-JIT'd exact path.
    """
    from tengri.models.sps.dsps_wrapper import LSUN_ERG_PER_S

    dt = model._forward_dtype
    ssp_wave = model.ssp_data.ssp_wave.astype(dt)
    _is_single_dust_exact = model._dust_model == "single_component"
    _dust_exact_sed = getattr(model, "_dust_approx", "fast") == "exact"
    if not _is_single_dust_exact:
        if _dust_exact_sed:
            dust_age_w = model._precomputed.dust_age_weights.astype(dt)
        else:
            _t_birth_exact = 1e7
            young_mask_exact = (model.ssp_ages_yr < _t_birth_exact).astype(dt)
            old_mask_exact = dt.type(1.0) - young_mask_exact
    lsun = dt.type(LSUN_ERG_PER_S)

    law_bc_fn = model._dust_law_bc_fn
    if not _is_single_dust_exact:
        law_diff_fn = model._dust_law_diff_fn
        same_law = model._dust_law_bc == model._dust_law_diff

    @jax.jit
    def exact_sed(
        weights,
        ssp_at_z,
        tau_bc=0.0,
        tau_diff=0.0,
        n_slope=-0.7,
        dust_bump_strength=0.0,
        dust_delta=0.0,
        dust_Rv=3.1,
        f_obscuration=0.0,
        tau_v=0.0,
    ):
        w = weights.astype(dt)
        ssp_z = ssp_at_z.astype(dt)

        _law_kw = dict(
            n_slope=n_slope,
            dust_bump_strength=dust_bump_strength,
            dust_delta=dust_delta,
            dust_Rv=dust_Rv,
        )

        if _is_single_dust_exact:
            k = law_bc_fn(ssp_wave, **_law_kw)
            trans_1d = f_obscuration + (1.0 - f_obscuration) * jnp.exp(-tau_v * k)
            sed_atten = (lsun * jnp.einsum("i,iw->w", w, ssp_z) * trans_1d).astype(jnp.float64)
        elif _dust_exact_sed:
            # Exact: smooth sigmoid — full (n_ages, n_wave) outer product
            k_bc = law_bc_fn(ssp_wave, **_law_kw)
            k_diff = k_bc if same_law else law_diff_fn(ssp_wave, **_law_kw)
            tau = dust_age_w[:, None] * tau_bc * k_bc[None, :] + tau_diff * k_diff[None, :]
            dust_trans = f_obscuration + (1.0 - f_obscuration) * jnp.exp(-tau)
            sed_atten = (lsun * jnp.einsum("i,iw,iw->w", w, ssp_z, dust_trans)).astype(jnp.float64)
        else:
            # Fast two-CSP decomposition
            k_bc = law_bc_fn(ssp_wave, **_law_kw)
            k_diff = k_bc if same_law else law_diff_fn(ssp_wave, **_law_kw)
            trans_bc = jnp.exp(-tau_bc * k_bc)
            trans_diff = jnp.exp(-tau_diff * k_diff)

            csp_young = jnp.einsum("i,iw->w", w * young_mask_exact, ssp_z)
            csp_old = jnp.einsum("i,iw->w", w * old_mask_exact, ssp_z)

            flux_no_geom = trans_diff * (trans_bc * csp_young + csp_old)
            flux_intr = csp_young + csp_old
            sed_atten = (
                lsun * (f_obscuration * flux_intr + (1.0 - f_obscuration) * flux_no_geom)
            ).astype(jnp.float64)

        sed_intr = (lsun * jnp.einsum("i,iw->w", w, ssp_z)).astype(jnp.float64)
        return sed_atten, sed_intr

    return exact_sed


def is_tier2_compatible(model):
    """Check if the compositional rest-frame SED kernel can be built.

    Tier 2 supports ALL physics components (unlike Tier 1 fused kernels).
    It only falls back when the model uses features that require non-standard
    SFH/metallicity paths (tabulated SFH, DSPS table, evolving Z).

    Parameters
    ----------
    model : SEDModel
        The model instance to check.

    Returns
    -------
    bool
        True if Tier 2 kernel can be built.
    """
    # Evolving metallicity needs per-age Z interpolation — complex but supported
    # Chemical evolution derives Z(t) from SFH — self-referential, supported
    # The only hard blockers are DSPS table path and tabulated SFH
    # (those are detected at call time, not build time)
    return True


def build_fused_rest_sed(model):
    """Build a JIT'd function: internal params -> rest-frame SED.

    Composes all enabled physics components into a single JIT'd function.
    Disabled components are excluded from the XLA graph at trace time
    (Python ``if`` on captured booleans).

    This is the **Tier 2** kernel — full wavelength resolution like the
    exact path but JIT-compiled end-to-end like the fused kernels.
    The observation model (redshift, filter integration, IGM) is applied
    separately by thin wrappers.

    Parameters
    ----------
    model : SEDModel
        The model instance providing config and precomputed arrays.

    Returns
    -------
    callable
        JIT-compiled function:
        ``(weights, ssp_flux_at_z, p_dict) -> rest_sed``
        where ``p_dict`` contains internal dust/AGN/nebular/radio/X-ray
        parameters.
    """
    from tengri.models.dust.attenuation import resolve_dust_law
    from tengri.models.sps.dsps_wrapper import LSUN_ERG_PER_S

    dt = model._forward_dtype
    ssp_wave = model.ssp_data.ssp_wave.astype(dt)
    _is_single_dust = model._dust_model == "single_component"
    _dust_exact = getattr(model, "_dust_approx", "fast") == "exact"
    if not _is_single_dust:
        if _dust_exact:
            dust_age_w = model._precomputed.dust_age_weights.astype(dt)
        else:
            _t_birth = 1e7  # 10 Myr — Charlot & Fall (2000)
            young_mask = (model.ssp_ages_yr < _t_birth).astype(dt)
            old_mask = dt.type(1.0) - young_mask
    lsun = dt.type(LSUN_ERG_PER_S)

    # Capture dust law functions (pure JAX, JIT-traceable)
    law_bc_fn = resolve_dust_law(model._dust_law_bc)
    if not _is_single_dust:
        law_diff_fn = resolve_dust_law(model._dust_law_diff)
        same_law = model._dust_law_bc == model._dust_law_diff

    # --- Optional components (Python if at trace time) ---
    # Nebular
    has_nebular = model._nebular_backend is not None and getattr(
        model._nebular_backend, "has_free_params", False
    )
    # Full-precision wavelength array for components that need float64
    ssp_wave_f64 = model.ssp_data.ssp_wave
    # Panchromatic grid for Zone 2 (may differ from ssp_wave when radio/xray enabled)
    rest_wave_f64 = model._rest_wavelength
    _needs_extension = rest_wave_f64 is not model.ssp_data.ssp_wave

    if has_nebular:
        from tengri.core.emission_helpers import attenuate_emission, nebular_emission

        nebular_backend = model._nebular_backend
        ssp_log_ages_yr = model.ssp_log_ages_yr
        _neb_dust_mode = getattr(model, "_neb_dust", "bc")
        _neb_bc_fn = getattr(model, "_neb_dust_law_bc_fn", law_bc_fn)

    # Shock emission
    has_shock = getattr(model, "_shock_enabled", False)
    if has_shock:
        from tengri.core.emission_helpers import shock_emission

    # Dust emission
    has_dust_em = model._dust_emission_model is not None
    if has_dust_em:
        from tengri.core.emission_helpers import dust_ir_emission
        from tengri.models.dust.emission import resolve_emission_model

        dust_emission_fn = resolve_emission_model(model._dust_emission_model)

    # AGN
    has_agn = model._agn_model is not None
    agn_parametric = model._agn_parametric if has_agn else False
    if has_agn:
        from tengri.core.emission_helpers import agn_emission
        from tengri.models.agn import resolve_agn_model

        agn_model_fn = resolve_agn_model(model._agn_model)

    # Radio
    has_radio = model._radio_enabled
    if has_radio:
        from tengri.core.emission_helpers import radio_emission

        _radio_sfr_mode = model._radio_sfr_mode
        _include_freefree = model._radio_include_freefree
        _redshift = float(getattr(model, "_redshift", 0.0))

    # X-ray
    has_xray = model._xray_enabled
    if has_xray:
        from tengri.core.emission_helpers import xray_emission

    # Constants for energy balance
    _c_aa = dt.type(2.99792458e18)  # c in Angstrom/s

    @jax.jit
    def rest_sed_kernel(weights, ssp_flux_at_z, p):
        """Compute rest-frame SED from CSP weights and Z-interpolated SSP.

        Parameters
        ----------
        weights : array, shape (n_age,)
            CSP mass weights (Msun).
        ssp_flux_at_z : array, shape (n_age, n_wave)
            SSP flux interpolated to target metallicity.
        p : dict
            Internal parameters (dust, AGN, nebular, radio, X-ray).

        Returns
        -------
        array, shape (n_wave,)
            Rest-frame SED in erg/s/Hz.
        """
        w = weights.astype(dt)
        ssp_z = ssp_flux_at_z.astype(dt)

        # --- 1. Dust attenuation ---
        tau_bc = jnp.asarray(p.get("tau_bc", 0.0), dtype=dt)
        tau_diff = jnp.asarray(p.get("tau_diff", 0.0), dtype=dt)
        tau_v = jnp.asarray(p.get("tau_v", 0.0), dtype=dt)
        dust_slope = jnp.asarray(p.get("dust_slope", -0.7), dtype=dt)
        f_obs = jnp.asarray(p.get("f_obscuration", 0.0), dtype=dt)
        bump = jnp.asarray(p.get("dust_bump_strength", 0.0), dtype=dt)
        delta = jnp.asarray(p.get("dust_delta", 0.0), dtype=dt)
        rv = jnp.asarray(p.get("dust_Rv", 3.1), dtype=dt)

        _law_kw = dict(
            n_slope=dust_slope,
            dust_bump_strength=bump,
            dust_delta=delta,
            dust_Rv=rv,
        )

        if _is_single_dust:
            k = law_bc_fn(ssp_wave, **_law_kw)
            trans_1d = f_obs + (1.0 - f_obs) * jnp.exp(-tau_v * k)
            sed_atten = (lsun * jnp.einsum("i,iw->w", w, ssp_z) * trans_1d).astype(jnp.float64)
            sed_intr = (lsun * jnp.einsum("i,iw->w", w, ssp_z)).astype(jnp.float64)
        elif _dust_exact:
            k_bc = law_bc_fn(ssp_wave, **_law_kw)
            k_diff = k_bc if same_law else law_diff_fn(ssp_wave, **_law_kw)
            tau = dust_age_w[:, None] * tau_bc * k_bc[None, :] + tau_diff * k_diff[None, :]
            dust_trans = f_obs + (1.0 - f_obs) * jnp.exp(-tau)
            sed_atten = (lsun * jnp.einsum("i,iw,iw->w", w, ssp_z, dust_trans)).astype(jnp.float64)
            sed_intr = (lsun * jnp.einsum("i,iw->w", w, ssp_z)).astype(jnp.float64)
        else:
            # Fast two-CSP decomposition
            k_bc = law_bc_fn(ssp_wave, **_law_kw)
            k_diff = k_bc if same_law else law_diff_fn(ssp_wave, **_law_kw)
            trans_bc = jnp.exp(-tau_bc * k_bc)
            trans_diff = jnp.exp(-tau_diff * k_diff)

            csp_young = jnp.einsum("i,iw->w", w * young_mask, ssp_z)
            csp_old = jnp.einsum("i,iw->w", w * old_mask, ssp_z)

            flux_no_geom = trans_diff * (trans_bc * csp_young + csp_old)
            flux_intr = csp_young + csp_old
            sed_atten = (lsun * (f_obs * flux_intr + (1.0 - f_obs) * flux_no_geom)).astype(
                jnp.float64
            )
            sed_intr = (lsun * flux_intr).astype(jnp.float64)

        sed = sed_atten

        # --- 2. Nebular emission ---
        L_abs_neb = jnp.float64(0.0)  # Initialize for energy balance
        if has_nebular:
            _sfr_last = weights[-1]  # approximate current SFR
            neb_raw = nebular_emission(
                nebular_backend,
                weights,
                ssp_wave_f64,
                ssp_log_ages_yr,
                p["log_z_abs"],
                _sfr_last,
                neb_logU=p.get("neb_logU", -3.0),
                neb_logZ_gas=p.get("neb_logZ_gas", None),
                neb_fesc=p.get("neb_fesc", 0.0),
                neb_fesc_lya=p.get("neb_fesc_lya", 0.0),
            )
            _tau_bc = p.get("tau_bc", p.get("tau_v", 0.0))
            _tau_diff = p.get("tau_diff", 0.0)
            _dust_kw = {
                "dust_slope": p.get("dust_slope", -0.7),
                "dust_bump_strength": p.get("dust_bump_strength", 0.0),
            }
            neb_sed, L_abs_neb = attenuate_emission(
                neb_raw,
                ssp_wave_f64,
                _neb_dust_mode,
                _tau_bc,
                _tau_diff,
                law_bc_fn,
                law_diff_fn if not _is_single_dust else law_bc_fn,
                neb_bc_fn=_neb_bc_fn,
                **_dust_kw,
            )
            sed = sed + neb_sed

        # --- 3. Shock emission ---
        if has_shock:
            shock_raw = shock_emission(
                ssp_wave_f64,
                sed,
                shock_frac=p.get("shock_frac", 0.0),
                shock_velocity=p.get("shock_velocity", 300.0),
                shock_log_density=p.get("shock_log_density", 0.0),
                shock_b_over_sqrt_n=p.get("shock_b_over_sqrt_n", 1.0),
                shock_abundance=p.get("shock_abundance", "solar"),
                shock_component=p.get("shock_component", "combined"),
            )
            _tau_diff_s = p.get("tau_diff", 0.0)
            _dust_kw_s = {
                "dust_slope": p.get("dust_slope", -0.7),
                "dust_bump_strength": p.get("dust_bump_strength", 0.0),
            }
            shock_sed, _ = attenuate_emission(
                shock_raw,
                ssp_wave_f64,
                "diff",
                0.0,
                _tau_diff_s,
                law_bc_fn,
                law_diff_fn if not _is_single_dust else law_bc_fn,
                **_dust_kw_s,
            )
            sed = sed + shock_sed

        # --- 4. Energy balance + AGN L_bol on SSP grid (before interp) ---
        if has_dust_em:
            nu_em = _c_aa / ssp_wave.astype(jnp.float64)
            L_absorbed_stellar = -jnp.trapezoid(sed_intr - sed_atten, nu_em)
            L_absorbed = L_absorbed_stellar
            if has_nebular:
                L_absorbed = L_absorbed + L_abs_neb
            eta_balance = p.get("dust_eta_balance", 1.0)
            L_ir = jnp.maximum(L_absorbed * eta_balance, 0.0)
        else:
            L_ir = jnp.float64(0.0)

        agn_bol_erg = jnp.float64(0.0)
        agn_log_lbol = jnp.float64(0.0)
        agn_frac_val = jnp.float64(0.0)
        if has_agn:
            if agn_parametric:
                agn_log_lbol = p.get("agn_log_lbol", 10.0)
                agn_frac_val = 1.0
                agn_bol_erg = 10.0**agn_log_lbol * LSUN_ERG_PER_S
            else:
                agn_frac_val = p.get("agn_frac", 0.0)
                nu_agn = _c_aa / ssp_wave.astype(jnp.float64)
                L_bol_stellar = -jnp.trapezoid(sed, nu_agn)
                agn_bol_erg = L_bol_stellar * agn_frac_val
                # AGN model functions expect log10(L_bol / Lsun), convert from erg/s
                _log_lsun = jnp.log10(LSUN_ERG_PER_S)
                agn_log_lbol = jnp.log10(jnp.maximum(agn_bol_erg, 1e-50)) - _log_lsun

        # --- 5. Interpolate to panchromatic grid if needed ---
        if _needs_extension:
            from tengri.utils.wavelength import interpolate_sed_to_grid

            sed = interpolate_sed_to_grid(ssp_wave_f64, sed, rest_wave_f64)

        # Zone 2 wavelength grid (panchromatic or SSP)
        wave_z2 = rest_wave_f64

        # --- 6. Dust IR emission (energy-balanced) ---
        if has_dust_em:
            dust_ir = dust_ir_emission(
                dust_emission_fn,
                wave_z2,
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

        # --- 7. AGN ---
        if has_agn:
            agn_sed = agn_emission(
                agn_model_fn,
                wave_z2,
                agn_log_lbol=agn_log_lbol,
                agn_frac=agn_frac_val,
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
            sed = sed + agn_sed

        # --- 8. Radio ---
        if has_radio:
            _log_mstar = jnp.log10(jnp.maximum(jnp.sum(weights), 1e-10))
            radio_sed = radio_emission(
                wave_z2,
                L_ir=L_ir,
                L_agn_bol=agn_bol_erg,
                q_ir=p.get("radio_q_ir", 2.64),
                alpha_sf=p.get("radio_alpha_sf", 0.8),
                radio_loudness=p.get("radio_loudness", 0.0),
                alpha_agn=p.get("radio_alpha_agn", 0.7),
                sfr_mode=_radio_sfr_mode,
                log_mstar=_log_mstar,
                redshift=_redshift,
                include_freefree=_include_freefree,
                T_e=p.get("radio_T_e", 1e4),
                alpha_ff=p.get("radio_alpha_ff", -0.1),
            )
            sed = sed + radio_sed

        # --- 9. X-ray ---
        if has_xray:
            sfr_current = p.get("_sfr_current", 1.0)
            mstar = jnp.sum(weights)
            xray_sed = xray_emission(
                wave_z2,
                sfr=sfr_current,
                stellar_mass=mstar,
                L_agn_bol=agn_bol_erg,
                gamma_agn=p.get("xray_gamma_agn", 1.8),
                alpha_ox=p.get("xray_alpha_ox", -1.4),
                gamma_hmxb=p.get("xray_gamma_hmxb", 2.0),
                gamma_lmxb=p.get("xray_gamma_lmxb", 1.6),
                E_cut=p.get("xray_E_cut", 300.0),
            )
            sed = sed + xray_sed

        return sed

    return rest_sed_kernel


# -------------------------------------------------------------------
# Tier 2 observation wrappers
# -------------------------------------------------------------------


def observe_photometry_from_rest_sed(
    rest_sed,
    wave_rest,
    z,
    dl_cm,
    filter_waves,
    filter_trans,
    apply_igm=False,
):
    """Apply redshift + filter integration to a rest-frame SED.

    Thin wrapper that converts a Tier 2 rest-frame SED into observed
    photometric flux densities. Not JIT'd itself (loops over filters),
    but each filter integration is a fast JAX operation.

    Parameters
    ----------
    rest_sed : array, shape (n_wave,)
        Rest-frame SED in erg/s/Hz.
    wave_rest : array, shape (n_wave,)
        Rest-frame wavelength grid (Angstrom).
    z : float
        Redshift.
    dl_cm : float
        Luminosity distance (cm).
    filter_waves : list of arrays
        Filter wavelength arrays.
    filter_trans : list of arrays
        Filter transmission arrays.
    apply_igm : bool
        Whether to apply IGM absorption.

    Returns
    -------
    array, shape (n_filters,)
        Observed flux densities in erg/s/cm^2/Hz.
    """
    from tengri.models.observation.photometry import compute_flux_density

    sed = rest_sed
    if apply_igm:
        from tengri.models.igm import igm_transmission

        wave_obs = wave_rest * (1.0 + z)
        igm_trans = igm_transmission(wave_obs, z)
        sed = sed * igm_trans

    fluxes = []
    for fw, ft in zip(filter_waves, filter_trans):
        f = compute_flux_density(sed, wave_rest, fw, ft, z, dl_cm)
        fluxes.append(f)
    return jnp.array(fluxes)


def observe_spectrum_from_rest_sed(
    rest_sed,
    wave_rest,
    wave_obs,
    z,
    dl_cm,
):
    """Apply redshift + interpolation to a rest-frame SED.

    Thin wrapper that converts a Tier 2 rest-frame SED into an
    observed spectrum at specified wavelength pixels.

    Parameters
    ----------
    rest_sed : array, shape (n_wave,)
        Rest-frame SED in erg/s/Hz.
    wave_rest : array, shape (n_wave,)
        Rest-frame wavelength grid (Angstrom).
    wave_obs : array, shape (n_pix,)
        Observed wavelength grid (Angstrom).
    z : float
        Redshift.
    dl_cm : float
        Luminosity distance (cm).

    Returns
    -------
    array, shape (n_pix,)
        Spectral flux density in erg/s/cm^2/Hz.
    """
    from tengri.models.observation.spectrum import compute_spectrum

    return compute_spectrum(rest_sed, wave_rest, wave_obs, z, dl_cm)


# -------------------------------------------------------------------
# Fused Tier 2 end-to-end kernels (params → photometry/spectrum)
# -------------------------------------------------------------------


def build_fused_tier2_photometry(model):
    """Build a single JIT function: params dict → observed photometry.

    Fuses the entire Tier 2 pipeline into one ``@jax.jit`` scope:

    1. Parameter translation (public → internal names + unit conversion)
    2. SFH computation (registry-dispatched composed function)
    3. Metallicity interpolation (linear or smooth, single-Z)
    4. Compositional rest-frame SED kernel (dust, nebular, AGN, ...)
    5. Filter integration (loop unrolled by JAX tracer)
    6. Optional IGM absorption

    Eliminates all Python dispatch overhead between steps. The filter
    loop executes 5 ``compute_flux_density`` calls in Python, but the
    JAX tracer unrolls them into a single XLA program.

    Parameters
    ----------
    model : SEDModel
        Fully initialized Model with filters and fixed redshift.

    Returns
    -------
    callable or None
        JIT-compiled function: ``params_dict -> photometry_array``.
        Returns None if prerequisites are not met (no filters, no
        fixed z, no Tier 2 kernel).
    """
    if model._compositional.rest_sed is None:
        return None
    if model.filter_waves is None:
        return None

    from tengri.core.param_translate import get_internal_params
    from tengri.core.sed_pipeline import interp_met_alpha_dispatch, interp_metallicity
    from tengri.models.observation.photometry import compute_flux_density
    from tengri.models.sfh.registry import compute_field_gp
    from tengri.models.sps.dsps_wrapper import compute_csp_weights

    _use_dsps_native = model._csp_integration == "dsps_native"
    if _use_dsps_native:
        from tengri.models.sps.dsps_wrapper import compute_dsps_native_weights

    # effective_metallicity correction is opt-in (see fused_kernels tier1 note).
    _use_alpha_fe_t2 = model.spec.alpha_fe_evolving or "met_alpha_fe" in model.spec.free_params

    # Capture model state at build time
    rest_sed_kernel = model._compositional.rest_sed
    param_map = model._param_map
    spec = model.spec
    has_field = model._has_field
    sfh_fn = model._sfh_fn
    sfh_internal_names = model._sfh_internal_names
    field_model = model._field_model
    n_grid = model._n_grid
    d_log_age = float(model.d_log_age)
    ssp_log_ages_yr = model.ssp_log_ages_yr
    log_age_grid = model.log_age_grid
    age_yr = model.age_yr
    ssp_ages_yr = model.ssp_ages_yr
    ssp_wave = model.ssp_data.ssp_wave
    xray_enabled = model._xray_enabled
    filter_waves = model.filter_waves
    filter_trans = model.filter_trans
    apply_igm = model._apply_igm

    # dsps_native: capture SSP arrays for DSPS triweight kernel
    if _use_dsps_native:
        _ssp_lgmet = model.ssp_data.ssp_lgmet
        _ssp_lg_age_gyr = model.ssp_data.ssp_lg_age_gyr
        _ssp_flux = model.ssp_data.ssp_flux
        _lgmet_scatter_native = float(model._lgmet_scatter)
        from tengri.utils.cosmology import age_at_z as _age_at_z_fn

    # Redshift: fixed (precompute IGM once) or free (traced through)
    z_fixed = model._z_fixed
    dl_cm_fixed = model._dl_cm_fixed
    is_free_z = z_fixed is None

    # For dsps_native + fixed z: precompute t_obs_gyr once at closure build
    _t_obs_gyr_fixed = None
    if _use_dsps_native and not is_free_z:
        _t_obs_gyr_fixed = float(_age_at_z_fn(z_fixed))

    # IGM at full wavelength grid (only for fixed z)
    igm_trans_full = None
    if apply_igm and not is_free_z:
        from tengri.models.igm import igm_transmission

        wave_obs_full = ssp_wave * (1.0 + z_fixed)
        igm_trans_full = igm_transmission(wave_obs_full, z_fixed)

    # For free-z: need luminosity_distance inside JIT
    if is_free_z:
        from tengri.utils.cosmology import luminosity_distance as _lum_dist

        if apply_igm:
            from tengri.models.igm import igm_transmission as _igm_fn

    # --- Shared SFH + SED computation (used by both fixed-z and free-z) ---
    def _compute_rest_sed(params):
        """params → (rest_sed, redshift_value)."""
        p = get_internal_params(params, param_map, spec, has_field)

        kw = {k: v for k, v in p.items() if k in sfh_internal_names}
        if has_field and "xi" in p:
            gp_x, k0_half = compute_field_gp(
                xi=p["xi"],
                psd_sigma=p["psd_sigma"],
                psd_tau_yr=p["psd_tau_yr"],
                n_grid=n_grid,
                d_log_age=d_log_age,
                field_model=field_model,
            )
            kw["gp_x"] = gp_x
            kw["k0_half"] = k0_half
        sfr = sfh_fn(age_yr, **kw)

        sfr_on_ssp = jnp.interp(ssp_log_ages_yr, log_age_grid, sfr)

        if _use_dsps_native:
            z = p.get("redshift", z_fixed if z_fixed is not None else 0.0)
            t_obs_gyr = _t_obs_gyr_fixed if _t_obs_gyr_fixed is not None else _age_at_z_fn(z)
            lgmet = p.get("log_z_abs", -1.8477)
            lgmet_scatter = float(p.get("lgmet_scatter", _lgmet_scatter_native))
            weights, ssp_flux_at_z = compute_dsps_native_weights(
                sfr_on_ssp,
                ssp_ages_yr,
                _ssp_lgmet,
                _ssp_lg_age_gyr,
                _ssp_flux,
                t_obs_gyr,
                lgmet,
                lgmet_scatter,
            )
        else:
            weights = compute_csp_weights(sfr_on_ssp, ssp_ages_yr)
            if _use_alpha_fe_t2:
                alpha_fe = p.get("alpha_fe", 0.0)
                ssp_flux_at_z = interp_met_alpha_dispatch(model, p["log_z_abs"], alpha_fe)
            else:
                ssp_flux_at_z = interp_metallicity(model, p["log_z_abs"])

        if xray_enabled:
            p = {**p, "_sfr_current": sfr[-1]}

        rest_sed = rest_sed_kernel(weights, ssp_flux_at_z, p)
        z = p.get("redshift", z_fixed if z_fixed is not None else 0.0)
        return rest_sed, z

    if is_free_z:

        @jax.jit
        def fused_tier2_phot(params):
            """params dict → observed photometry (free z)."""
            rest_sed, z = _compute_rest_sed(params)
            dl_cm = _lum_dist(z)

            if apply_igm:
                wave_obs = ssp_wave * (1.0 + z)
                rest_sed = rest_sed * _igm_fn(wave_obs, z)

            fluxes = []
            for fw, ft in zip(filter_waves, filter_trans):
                f = compute_flux_density(rest_sed, ssp_wave, fw, ft, z, dl_cm)
                fluxes.append(f)
            return jnp.array(fluxes)

    else:

        @jax.jit
        def fused_tier2_phot(params):
            """params dict → observed photometry (fixed z)."""
            rest_sed, _z = _compute_rest_sed(params)

            if igm_trans_full is not None:
                rest_sed = rest_sed * igm_trans_full

            fluxes = []
            for fw, ft in zip(filter_waves, filter_trans):
                f = compute_flux_density(rest_sed, ssp_wave, fw, ft, z_fixed, dl_cm_fixed)
                fluxes.append(f)
            return jnp.array(fluxes)

    return fused_tier2_phot


def build_fused_tier2_spectrum(model):
    """Build a single JIT function: params dict → observed spectrum.

    Like :func:`build_fused_tier2_photometry` but for spectroscopy.
    Requires ``precompute_spectroscopy()`` to have been called (for
    the wavelength grid) or an Observation with spectroscopy config.

    Parameters
    ----------
    model : SEDModel
        Fully initialized Model with spectroscopy config.

    Returns
    -------
    callable or None
        JIT-compiled function: ``params_dict -> spectrum_array``.
    """
    if model._compositional.rest_sed is None:
        return None

    from tengri.core.param_translate import get_internal_params
    from tengri.core.sed_pipeline import interp_met_alpha_dispatch, interp_metallicity
    from tengri.models.observation.spectrum import compute_spectrum
    from tengri.models.sfh.registry import compute_field_gp
    from tengri.models.sps.dsps_wrapper import compute_csp_weights

    _use_dsps_native_spec = model._csp_integration == "dsps_native"
    if _use_dsps_native_spec:
        from tengri.models.sps.dsps_wrapper import compute_dsps_native_weights

    # effective_metallicity correction is opt-in: only applied when the user
    # explicitly makes met_alpha_fe (or evolving variant) a free parameter.
    _use_alpha_fe_spec2 = model.spec.alpha_fe_evolving or "met_alpha_fe" in model.spec.free_params

    # Must have a wavelength grid
    wave_obs = None
    if model._precomputed.spectroscopy is not None:
        wave_obs = model._precomputed.spectroscopy.wave_obs_pixels
    elif hasattr(model, "_wave_obs"):
        wave_obs = model._wave_obs
    if wave_obs is None:
        return None

    # Capture model state
    rest_sed_kernel = model._compositional.rest_sed
    param_map = model._param_map
    spec = model.spec
    has_field = model._has_field
    sfh_fn = model._sfh_fn
    sfh_internal_names = model._sfh_internal_names
    field_model = model._field_model
    n_grid = model._n_grid
    d_log_age = float(model.d_log_age)
    ssp_log_ages_yr = model.ssp_log_ages_yr
    log_age_grid = model.log_age_grid
    age_yr = model.age_yr
    ssp_ages_yr = model.ssp_ages_yr
    ssp_wave = model.ssp_data.ssp_wave
    xray_enabled = model._xray_enabled

    if _use_dsps_native_spec:
        _ssp_lgmet_spec = model.ssp_data.ssp_lgmet
        _ssp_lg_age_gyr_spec = model.ssp_data.ssp_lg_age_gyr
        _ssp_flux_spec = model.ssp_data.ssp_flux
        _lgmet_scatter_spec = float(model._lgmet_scatter)
        from tengri.utils.cosmology import age_at_z as _age_at_z_spec

    z_fixed = model._z_fixed
    dl_cm_fixed = model._dl_cm_fixed
    is_free_z = z_fixed is None

    _t_obs_gyr_fixed_spec = None
    if _use_dsps_native_spec and not is_free_z:
        _t_obs_gyr_fixed_spec = float(_age_at_z_spec(z_fixed))

    if is_free_z:
        from tengri.utils.cosmology import luminosity_distance as _lum_dist_spec

    # Shared SFH+SED (same as photometry)
    def _compute_rest_sed_spec(params):
        p = get_internal_params(params, param_map, spec, has_field)
        kw = {k: v for k, v in p.items() if k in sfh_internal_names}
        if has_field and "xi" in p:
            gp_x, k0_half = compute_field_gp(
                xi=p["xi"],
                psd_sigma=p["psd_sigma"],
                psd_tau_yr=p["psd_tau_yr"],
                n_grid=n_grid,
                d_log_age=d_log_age,
                field_model=field_model,
            )
            kw["gp_x"] = gp_x
            kw["k0_half"] = k0_half
        sfr = sfh_fn(age_yr, **kw)
        sfr_on_ssp = jnp.interp(ssp_log_ages_yr, log_age_grid, sfr)
        if _use_dsps_native_spec:
            z = p.get("redshift", z_fixed if z_fixed is not None else 0.0)
            t_obs_gyr = (
                _t_obs_gyr_fixed_spec if _t_obs_gyr_fixed_spec is not None else _age_at_z_spec(z)
            )
            lgmet = p.get("log_z_abs", -1.8477)
            lgmet_scatter = float(p.get("lgmet_scatter", _lgmet_scatter_spec))
            weights, ssp_flux_at_z = compute_dsps_native_weights(
                sfr_on_ssp,
                ssp_ages_yr,
                _ssp_lgmet_spec,
                _ssp_lg_age_gyr_spec,
                _ssp_flux_spec,
                t_obs_gyr,
                lgmet,
                lgmet_scatter,
            )
        else:
            weights = compute_csp_weights(sfr_on_ssp, ssp_ages_yr)
            if _use_alpha_fe_spec2:
                alpha_fe = p.get("alpha_fe", 0.0)
                ssp_flux_at_z = interp_met_alpha_dispatch(model, p["log_z_abs"], alpha_fe)
            else:
                ssp_flux_at_z = interp_metallicity(model, p["log_z_abs"])
        if xray_enabled:
            p = {**p, "_sfr_current": sfr[-1]}
        rest_sed = rest_sed_kernel(weights, ssp_flux_at_z, p)
        z = p.get("redshift", z_fixed if z_fixed is not None else 0.0)
        return rest_sed, z

    if is_free_z:

        @jax.jit
        def fused_tier2_spec(params):
            """params dict → observed spectrum (free z)."""
            rest_sed, z = _compute_rest_sed_spec(params)
            dl_cm = _lum_dist_spec(z)
            return compute_spectrum(rest_sed, ssp_wave, wave_obs, z, dl_cm)

    else:

        @jax.jit
        def fused_tier2_spec(params):
            """params dict → observed spectrum (fixed z)."""
            rest_sed, _z = _compute_rest_sed_spec(params)
            return compute_spectrum(rest_sed, ssp_wave, wave_obs, z_fixed, dl_cm_fixed)

    return fused_tier2_spec
