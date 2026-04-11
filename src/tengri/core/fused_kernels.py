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
    # TODO(refactor): The inline per-component blocks below (~200 lines) mirror
    # the pattern that build_fused_rest_sed was refactored away from via
    # build_nonstell_fn() in core/nonstell.py.  Migrating this kernel is deferred
    # because _hybrid_phot_body has two preintegrated photometry-level shortcuts —
    # dust-IR triweight lookup (_has_preint_dust_ir) and nebular preintegration
    # (_has_preint_neb) — that return filter-integrated quantities rather than a
    # per-wavelength SED.  build_nonstell_fn() returns a full-wavelength SED, so
    # the preintegrated paths cannot be incorporated without either (a) dropping
    # the fast paths (performance loss) or (b) extending build_nonstell_fn to
    # optionally return photometry shortcuts.  Revisit once the preintegrated paths
    # are verified and stabilised.  Tracked in docs/dev/sessions/.
    ssp_wave_f64 = model.ssp_data.ssp_wave
    rest_wave_f64 = model._rest_wavelength
    _needs_extension = rest_wave_f64 is not model.ssp_data.ssp_wave

    # Nebular
    has_nebular = model._nebular_backend is not None and getattr(
        model._nebular_backend, "has_free_params", False
    )
    # Check if preintegrated nebular data is available
    _has_preint_neb = has_nebular and getattr(
        model._nebular_backend, "_has_preint_photometry", False
    )
    if has_nebular:
        nebular_backend = model._nebular_backend
        ssp_log_ages_yr = model.ssp_log_ages_yr
        _neb_dust_mode = getattr(model, "_neb_dust", "bc")
        _neb_bc_fn = getattr(model, "_neb_dust_law_bc_fn", law_bc_fn)
    if _has_preint_neb:
        # Capture preintegrated CLOUDY data for fast nebular photometry
        from tengri.core.preintegrate import interp_nd_triweight

        _neb_cont_phot = nebular_backend._preint_continuum.phot  # (n_Z, n_age, n_logU, n_filt)
        _neb_cont_axes = nebular_backend._preint_continuum.axes
        _neb_cont_edges = nebular_backend._preint_continuum.edges
        _neb_line_weights = nebular_backend._preint_lines.line_filter_weights  # (n_lines, n_filt)
        _neb_line_axes = nebular_backend._preint_lines.axes
        _neb_line_edges = nebular_backend._preint_lines.edges
        # Line luminosity grid (still in log10 space for triweight interp)
        _neb_line_lum = nebular_backend.grid.line_luminosity  # (n_Z, n_age, n_logU, n_lines)
        # Young SSP indices and CLOUDY age grid for age-sum
        _neb_young_idx = jnp.array(nebular_backend._young_idx)
        _neb_qh_table = nebular_backend._qh_table  # (n_met_ssp, n_age_ssp)
        _neb_qh_log_met = nebular_backend._qh_log_met
        _neb_qh_log_age = nebular_backend._qh_log_age
        _neb_lya_idx = int(jnp.argmin(jnp.abs(nebular_backend.grid.line_wavelengths - 1215.67)))

    # Shock
    has_shock = getattr(model, "_shock_enabled", False)

    # Dust emission (full wavelength or preintegrated)
    has_dust_em_full = model._dust_emission_model is not None
    _has_preint_dust_ir = False
    _dust_model_name = None
    if has_dust_em_full:
        from tengri.models.dust.emission import resolve_emission_model

        dust_emission_fn = resolve_emission_model(model._dust_emission_model)

        # Check if preintegrated dust IR lookup is available (for fast photometry)
        # NOTE: Disable preintegration if radio/X-ray is enabled, since dust IR
        # must be computed on the panchromatic grid to match other non-stellar
        # components. Preintegrated lookup was computed on SSP grid only.
        if model._precomputed.dust_ir_lookup is not None and not _needs_extension:
            _has_preint_dust_ir = True
            _dust_ir_lookup = model._precomputed.dust_ir_lookup
            _dust_model_name = model._dust_emission_model

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

    def _stellar_phot(
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
        tau_v=0.0,
    ):
        """Preintegrated stellar photometry + L_absorbed estimate.

        Computes stellar CSP through preintegrated SSP×filter tensor with
        metallicity + alpha interpolation, dust attenuation at effective
        wavelengths (single/two-component, exact/fast, with Taylor expansion),
        and L_absorbed estimation via Voronoi bandwidth weighting.

        Parameters
        ----------
        sfr_on_ssp : array-like
            SFR weights on SSP grid.
        log_z_abs : float
            log10(Z) absolute metallicity.
        tau_bc : float
            Birth-cloud optical depth (for two-component dust).
        tau_diff : float
            Diffuse ISM optical depth (for two-component dust).
        dust_slope : float
            Dust law power-law index.
        f_obscuration : float
            Obscured fraction (geometry).
        dust_bump_strength : float
            2175 Å bump strength.
        dust_delta : float
            Mid-UV dust delta parameter.
        dust_Rv : float
            Dust extinction curve Rv parameter.
        alpha_fe : float
            Alpha-to-iron enhancement ratio.
        tau_v : float, optional
            Optical depth (single-component dust case).

        Returns
        -------
        flux_attenuated : (n_filters,) array
            Stellar photometry in Lsun/Hz units.
        L_absorbed_stellar : scalar
            Absorbed stellar luminosity in erg/s.
        weights : (n_ages,) array
            CSP weights (needed for non-stellar calculations).
        """
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
        # Guard against NaN/Inf from pure SSPs with zero/negligible continuum
        L_absorbed_stellar = jnp.where(
            jnp.isfinite(L_absorbed_stellar), L_absorbed_stellar, dt.type(0.0)
        )
        L_absorbed_stellar = jnp.maximum(L_absorbed_stellar, dt.type(0.0))

        return flux_attenuated, L_absorbed_stellar, weights

    def _nebular_phot_preintegrated(
        weights,
        log_z_abs,
        neb_logU,
        neb_logZ_gas,
        neb_fesc,
        neb_fesc_lya,
        tau_bc,
        tau_diff,
        dust_slope,
        dust_bump_strength,
        tau_v=0.0,
    ):
        """Preintegrated nebular photometry (continuum + lines).

        Same pattern as _stellar_phot: age-weighted sum over preintegrated
        grid, with dust at effective wavelengths.  Operates on (n_filters,)
        instead of (n_wave,) — no filter integration loop needed.
        """
        _gas_z = jnp.where(neb_logZ_gas is None, log_z_abs, neb_logZ_gas)

        # --- Nebular continuum: age-sum over preintegrated CLOUDY grid ---
        # For each young SSP age, interp CLOUDY grid at (Z_gas, age, logU)
        # to get (n_filters,) per Q_H, then scale by Q_H × weight × (1-fesc)
        cont_phot = jnp.zeros(n_filters, dtype=jnp.float64)

        # Q_H interpolation helper (bilinear in SSP met × age)
        def _get_qh(log_z, log_age_yr):
            iz = jnp.clip(
                jnp.searchsorted(_neb_qh_log_met, log_z) - 1, 0, len(_neb_qh_log_met) - 2
            )
            ia = jnp.clip(
                jnp.searchsorted(_neb_qh_log_age, log_age_yr) - 1, 0, len(_neb_qh_log_age) - 2
            )
            fz = (log_z - _neb_qh_log_met[iz]) / jnp.maximum(
                _neb_qh_log_met[iz + 1] - _neb_qh_log_met[iz], 1e-30
            )
            fa = (log_age_yr - _neb_qh_log_age[ia]) / jnp.maximum(
                _neb_qh_log_age[ia + 1] - _neb_qh_log_age[ia], 1e-30
            )
            fz = jnp.clip(fz, 0.0, 1.0)
            fa = jnp.clip(fa, 0.0, 1.0)
            return (
                (1 - fz) * (1 - fa) * _neb_qh_table[iz, ia]
                + (1 - fz) * fa * _neb_qh_table[iz, ia + 1]
                + fz * (1 - fa) * _neb_qh_table[iz + 1, ia]
                + fz * fa * _neb_qh_table[iz + 1, ia + 1]
            )

        # Continuum: vmap over young ages
        young_ages = ssp_log_ages_yr[_neb_young_idx]
        young_weights = weights[_neb_young_idx]

        def _cont_one_age(log_age_i, weight_i):
            qh_i = _get_qh(log_z_abs, log_age_i)
            # Triweight interp in (Z_gas, age_cloudy, logU) → (n_filters,) per Q_H
            # The preint grid is in log10 linear space (10^log10_lum was done at preint time)
            phot_per_qh = interp_nd_triweight(
                _neb_cont_phot,
                _neb_cont_axes,
                _neb_cont_edges,
                (_gas_z, log_age_i, neb_logU),
            )
            return weight_i * qh_i * phot_per_qh * (1.0 - neb_fesc)

        cont_contribs = jax.vmap(_cont_one_age)(young_ages, young_weights)
        cont_phot = jnp.sum(cont_contribs, axis=0)  # (n_filters,)

        # --- Lines: same age-sum but on (n_lines,), then project to filters ---
        def _line_one_age(log_age_i, weight_i):
            qh_i = _get_qh(log_z_abs, log_age_i)
            # Interp line luminosities in (Z_gas, age, logU) → (n_lines,) log10
            log_lum_per_qh = interp_nd_triweight(
                _neb_line_lum,
                _neb_line_axes,
                _neb_line_edges,
                (_gas_z, log_age_i, neb_logU),
            )
            return weight_i * qh_i * (10.0**log_lum_per_qh) * (1.0 - neb_fesc)

        line_contribs = jax.vmap(_line_one_age)(young_ages, young_weights)
        total_line_lum = jnp.sum(line_contribs, axis=0)  # (n_lines,)

        # Lya escape fraction correction
        lya_scale = (1.0 - neb_fesc_lya) / jnp.maximum(1.0 - neb_fesc, 1e-10)
        total_line_lum = total_line_lum.at[_neb_lya_idx].multiply(lya_scale)

        # Project lines to filter space: (n_lines,) × (n_lines, n_filt) → (n_filt,)
        line_phot = jnp.einsum("l,lf->f", total_line_lum, _neb_line_weights)

        # Total nebular photometry (before dust)
        neb_phot = cont_phot + line_phot

        # Apply nebular dust at effective wavelengths
        # (same approximation as stellar: evaluate dust curve at λ_eff per filter)
        _law_kw = {"n_slope": dust_slope, "dust_bump_strength": dust_bump_strength}
        _tau_bc_neb = tau_bc if not _is_single_dust else tau_v
        _tau_diff_neb = tau_diff if not _is_single_dust else jnp.float64(0.0)
        if _neb_dust_mode in ("bc", "neb"):
            k_bc_neb = law_bc_fn(eff_waves_rest, **_law_kw)
            neb_phot = neb_phot * jnp.exp(-_tau_bc_neb * k_bc_neb)
        if _neb_dust_mode != "none":
            if not _is_single_dust:
                k_diff_neb = law_diff_fn(eff_waves_rest, **_law_kw)
            else:
                k_diff_neb = law_bc_fn(eff_waves_rest, **_law_kw)
            neb_phot = neb_phot * jnp.exp(-_tau_diff_neb * k_diff_neb)

        return neb_phot

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
        # === STEP 1: Stellar photometry ===
        flux_attenuated, L_absorbed_stellar, weights = _stellar_phot(
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
            tau_v=tau_v,
        )

        # === STEP 2: Non-stellar SED at full wavelength resolution ===
        # TODO: preintegrated non-stellar paths (CLOUDY, DL07, SKIRTOR)
        # are implemented in _nebular_phot_preintegrated() but need
        # calibration verification before enabling. For now, all
        # non-stellar components use the full-wavelength path.
        _has_any_nonstell = (
            has_nebular or has_shock or has_dust_em_full or has_agn_full or has_radio or has_xray
        )

        non_stellar_phot = jnp.zeros(n_filters, dtype=jnp.float64)

        if _has_any_nonstell:
            non_stellar_sed = jnp.zeros_like(ssp_wave_f64)
            L_abs_neb = jnp.float64(0.0)

            if has_nebular:
                # Use SFR (Msun/yr), NOT CSP mass weight (Msun).
                _sfr_last = sfr_on_ssp[-1]
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

            # 2b5: Extend to panchromatic grid if radio/X-ray enabled
            # (before components that need extended wavelength range)
            if _needs_extension:
                from tengri.utils.wavelength import interpolate_sed_to_grid

                non_stellar_sed = interpolate_sed_to_grid(
                    ssp_wave_f64, non_stellar_sed, rest_wave_f64
                )

            # 2c: Dust IR emission (energy-balanced)
            dust_ir_phot_preint = jnp.zeros(n_filters, dtype=jnp.float64)
            if has_dust_em_full:
                L_ir = jnp.maximum(
                    (L_absorbed_stellar + L_abs_neb) * jnp.float64(dust_eta_balance), 0.0
                )

                if _has_preint_dust_ir:
                    # Use preintegrated template lookup (fast triweight interp)
                    # Signature varies by dust model
                    if _dust_model_name == "draine_li2007":
                        # DL07: (L_absorbed, dust_umin, dust_gamma_dl, dust_qpah)
                        dust_ir_phot_preint = _dust_ir_lookup(
                            L_ir,
                            jnp.float64(dust_umin),
                            jnp.float64(dust_gamma_dl),
                            jnp.float64(dust_qpah),
                        )
                    elif _dust_model_name == "dale2014":
                        # Dale2014: (L_absorbed, dust_alpha_dale)
                        dust_ir_phot_preint = _dust_ir_lookup(
                            L_ir,
                            jnp.float64(dust_alpha_dale),
                        )
                    else:
                        # Generic template model: (L_absorbed, *grid_params)
                        # For now, fall back to full-wavelength for other models
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
                        non_stellar_sed = non_stellar_sed + dust_ir
                else:
                    # Full-wavelength computation (fallback or if preintegration disabled)
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
                sfr_now = sfr_on_ssp[-1]  # SFR (Msun/yr), not mass weight
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
            # Loop over filters (unrolled by JAX tracer)
            ns_fluxes = []
            for fw, ft in zip(filter_waves_list, filter_trans_list):
                f = compute_flux_density(
                    non_stellar_sed, rest_wave_f64, fw, ft, z_fixed, dl_cm_fixed
                )
                ns_fluxes.append(f)
            if n_filters > 0:
                non_stellar_phot = jnp.array(ns_fluxes)

        # === STEP 4: Combine stellar + non-stellar ===
        stellar_phot = flux_attenuated
        if has_igm:
            stellar_phot = stellar_phot * igm_trans

        # Scale stellar to erg/s/cm^2/Hz
        stellar_phot = (flux_scale * stellar_phot * lsun).astype(jnp.float64)

        # Add preintegrated dust IR photometry if available.
        # The lookup returns L_ν (erg/s/Hz); convert to flux density
        # (erg/s/cm²/Hz) with the same (1+z)/(4π d_L²) scaling as stellar.
        if _has_preint_dust_ir:
            non_stellar_phot = non_stellar_phot + dust_ir_phot_preint * flux_scale

        return stellar_phot + non_stellar_phot

    # --- Fused end-to-end wrapper: params dict → photometry ---
    # Fuse param translation + SFH computation into the JIT scope,
    # eliminating ~240 μs of Python dispatch overhead per call.
    from tengri.core.param_translate import get_internal_params
    from tengri.models.sfh.registry import compute_field_gp

    param_map = model._param_map
    spec = model.spec
    has_field = model._has_field
    sfh_fn = model._sfh_fn
    sfh_internal_names = model._sfh_internal_names
    field_model = model._field_model
    n_grid = model._n_grid
    d_log_age = float(model.d_log_age)
    ssp_log_ages_yr_cap = model.ssp_log_ages_yr
    log_age_grid = model.log_age_grid
    age_yr = model.age_yr

    @jax.jit
    def hybrid_phot_fused(params):
        """params dict → photometry (end-to-end JIT, no Python dispatch)."""
        p = get_internal_params(params, param_map, spec, has_field)

        # SFH computation (same as build_fused_tier2_photometry)
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
        sfr_on_ssp = jnp.interp(ssp_log_ages_yr_cap, log_age_grid, sfr)

        # Dispatch to the inner kernel (already defined above)
        if _is_single_dust:
            return hybrid_phot(
                sfr_on_ssp,
                p["log_z_abs"],
                p.get("tau_v", 0.0),
                p.get("dust_slope", -0.7),
                **{
                    k: p.get(k, v)
                    for k, v in [
                        ("f_obscuration", 0.0),
                        ("dust_bump_strength", 0.0),
                        ("dust_delta", 0.0),
                        ("dust_Rv", 3.1),
                        ("alpha_fe", 0.0),
                        ("dust_T", 35.0),
                        ("dust_beta_ir", 1.6),
                        ("dust_eta_balance", 1.0),
                        ("agn_log_lbol", 10.0),
                        ("agn_alpha", -1.0),
                        ("agn_T_torus", 1000.0),
                        ("agn_tau_torus", 5.0),
                        ("agn_torus_frac", 0.5),
                        ("agn_log_mbh", 7.0),
                        ("agn_log_ledd", -1.0),
                    ]
                },
            )
        return hybrid_phot(
            sfr_on_ssp,
            p["log_z_abs"],
            p.get("tau_bc", 0.0),
            p.get("tau_diff", 0.0),
            p.get("dust_slope", -0.7),
            **{
                k: p.get(k, v)
                for k, v in [
                    ("f_obscuration", 0.0),
                    ("dust_bump_strength", 0.0),
                    ("dust_delta", 0.0),
                    ("dust_Rv", 3.1),
                    ("alpha_fe", 0.0),
                    ("dust_T", 35.0),
                    ("dust_beta_ir", 1.6),
                    ("dust_eta_balance", 1.0),
                    ("agn_log_lbol", 10.0),
                    ("agn_alpha", -1.0),
                    ("agn_T_torus", 1000.0),
                    ("agn_tau_torus", 5.0),
                    ("agn_torus_frac", 0.5),
                    ("agn_log_mbh", 7.0),
                    ("agn_log_ledd", -1.0),
                ]
            },
        )

    return hybrid_phot_fused


def build_hybrid_photometry_ztable(model):
    """Build hybrid photometry kernel for free-z inference using a z-table.

    Like build_hybrid_photometry but with SSP photometry interpolated from
    a precomputed redshift grid instead of fixed at model.redshift.
    Stellar photometry uses precomputed SSP×filter einsum (fast, ~0.4% error).
    Non-stellar components evaluated at full wavelength resolution via
    emission_helpers, then integrated through filters (exact).

    For free-redshift inference, this kernel interpolates the precomputed
    z-table to the current redshift at each step, maintaining the same
    speedup as fixed-z precomputation while allowing z to vary.

    Parameters
    ----------
    model : SEDModel
        The model instance providing config and precomputed z-table arrays.

    Returns
    -------
    callable
        JIT-compiled function: (params_dict) -> photometry array (n_filters,)
        in erg/s/cm^2/Hz. Parameters include redshift as a free variable.

    Raises
    ------
    ValueError
        If photometry_ztable has not been precomputed via model.precompute_ztable().
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
    from tengri.core.param_translate import get_internal_params
    from tengri.models.dust.attenuation import resolve_dust_law
    from tengri.models.sfh.registry import compute_field_gp, resolve_sfh
    from tengri.models.sps.dsps_wrapper import LSUN_ERG_PER_S
    from tengri.models.sps.precompute import interpolate_ztable

    # Validate that z-table has been precomputed
    if model._precomputed.photometry_ztable is None:
        raise ValueError("Z-table not precomputed. Call model.precompute_ztable() first.")

    dt = model._forward_dtype
    ztable = model._precomputed.photometry_ztable
    ssp_lgmet = model.ssp_data.ssp_lgmet.astype(dt)
    _is_single_dust = model._dust_model == "single_component"
    _dust_exact = getattr(model, "_dust_approx", "fast") == "exact"
    if not _is_single_dust:
        if _dust_exact:
            dust_age_w = model._precomputed.dust_age_weights.astype(dt)
        else:
            _t_birth = 1e7
            young_mask = (model.ssp_ages_yr < _t_birth).astype(dt)
            old_mask = dt.type(1.0) - young_mask
    _csp_use_matrix = model._csp_integration == "log_interp"
    if _csp_use_matrix:
        _csp_mat = model._csp_matrix.astype(dt)
    else:
        _age_dt = model._csp_age_dt.astype(dt)
    lsun = dt.type(LSUN_ERG_PER_S)

    # Voronoi frequency bandwidths for L_absorbed broadband estimate (Hz).
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

    # IGM: precomputed on z-table
    has_igm = ztable.igm_trans_table is not None
    if has_igm:
        # Ensure igm_trans_table is all ones (expected for z-table)
        # Full wavelength IGM will be evaluated in non-stellar section
        pass

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
        _neb_dust_mode = getattr(model, "_neb_dust", "bc")
        _neb_bc_fn = getattr(model, "_neb_dust_law_bc_fn", law_bc_fn)
    _has_preint_neb = has_nebular and getattr(
        model._nebular_backend, "_has_preint_photometry", False
    )
    if _has_preint_neb:
        _neb_cont_phot = nebular_backend._preint_continuum.phot
        _neb_cont_axes = nebular_backend._preint_continuum.axes
        _neb_cont_edges = nebular_backend._preint_continuum.edges
        _neb_line_weights = nebular_backend._preint_lines.line_filter_weights
        _neb_line_axes = nebular_backend._preint_lines.axes
        _neb_line_edges = nebular_backend._preint_lines.edges
        _neb_line_lum = nebular_backend.grid.line_luminosity
        _neb_young_idx = jnp.array(nebular_backend._young_idx)
        _neb_qh_table = nebular_backend._qh_table
        _neb_qh_log_met = nebular_backend._qh_log_met
        _neb_qh_log_age = nebular_backend._qh_log_age
        _neb_lya_idx = int(jnp.argmin(jnp.abs(nebular_backend.grid.line_wavelengths - 1215.67)))

    # Shock
    has_shock = getattr(model, "_shock_enabled", False)

    # Dust emission (full wavelength or preintegrated)
    has_dust_em_full = model._dust_emission_model is not None
    _dust_model_name = model._dust_emission_model if has_dust_em_full else None
    if has_dust_em_full:
        from tengri.models.dust.emission import resolve_emission_model

        resolve_emission_model(model._dust_emission_model)

    # AGN (full wavelength)
    has_agn_full = model._agn_model is not None
    if has_agn_full:
        from tengri.models.agn import resolve_agn_model

        agn_model_fn_full = resolve_agn_model(model._agn_model)

    # Radio
    has_radio = model._radio_enabled
    if has_radio:
        _radio_sfr_mode = model._radio_sfr_mode
        _include_freefree = model._radio_include_freefree

    # X-ray
    has_xray = model._xray_enabled

    # Constants for energy balance
    _c_aa = dt.type(2.99792458e18)

    # Filter information
    n_filters = len(model.filter_waves) if model.filter_waves else 0
    filter_waves_list = model.filter_waves if model.filter_waves else []
    filter_trans_list = model.filter_trans if model.filter_trans else []

    # Build SFH function and parameters from model
    spec = model.spec
    param_map = model._param_map
    age_yr = model._age_yr
    log_age_grid = model._log_age_grid
    d_log_age = model._d_log_age
    ssp_log_ages_yr_cap = model._ssp_log_ages_yr_cap
    has_field = "xi" in model.spec.free_params
    sfh_internal_names = model._sfh_internal_names
    n_grid = model._age_grid_n_pts
    field_model = model._field_model
    sfh_fn = resolve_sfh(model._sfh_model)

    # === Define kernel signature (parametric-style, same as fixed-z) ===

    if _is_single_dust:

        @jax.jit
        def hybrid_phot_ztable(
            sfr_on_ssp,
            log_z_abs,
            redshift,
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
            return _hybrid_phot_ztable_body(
                sfr_on_ssp,
                log_z_abs,
                redshift,
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
        def hybrid_phot_ztable(
            sfr_on_ssp,
            log_z_abs,
            redshift,
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
            return _hybrid_phot_ztable_body(
                sfr_on_ssp,
                log_z_abs,
                redshift,
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

    def _stellar_phot_ztable(
        sfr_on_ssp,
        log_z_abs,
        redshift,
        tau_bc,
        tau_diff,
        dust_slope,
        f_obscuration,
        dust_bump_strength,
        dust_delta,
        dust_Rv,
        alpha_fe,
        tau_v=0.0,
    ):
        """Stellar photometry from z-table: metallicity interp + z-table lookup.

        Interpolates precomputed z-table to the current redshift and
        metallicity to get SSP photometry, then applies dust attenuation
        and computes L_absorbed estimate (same as fixed-z version).
        """
        sfr = sfr_on_ssp.astype(dt)
        lz = jnp.asarray(log_z_abs, dtype=dt)
        z = jnp.asarray(redshift, dtype=dt)
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

        # === Interpolate z-table to current redshift ===
        ssp_phot_at_z, eff_waves_rest, _flux_scale = interpolate_ztable(
            ztable.ssp_phot_table,
            ztable.eff_waves_rest_table,
            ztable.flux_scale_table,
            ztable.z_grid,
            z,
        )
        # ssp_phot_at_z: shape (n_met, n_age, n_filters)
        # eff_waves_rest: shape (n_filters,)
        # _flux_scale: scalar (not used in z-table, photometry already scaled)

        # === Interpolate metallicity ===
        if _has_alpha:
            lz_c = jnp.clip(lz, ssp_lgmet[0], ssp_lgmet[-1])
            iz = jnp.clip(jnp.searchsorted(ssp_lgmet, lz_c) - 1, 0, len(ssp_lgmet) - 2)
            fz = (lz_c - ssp_lgmet[iz]) / (ssp_lgmet[iz + 1] - ssp_lgmet[iz])
            afe_c = jnp.clip(afe, ssp_alpha_fe[0], ssp_alpha_fe[-1])
            ia = jnp.clip(jnp.searchsorted(ssp_alpha_fe, afe_c) - 1, 0, len(ssp_alpha_fe) - 2)
            fa = (afe_c - ssp_alpha_fe[ia]) / (ssp_alpha_fe[ia + 1] - ssp_alpha_fe[ia])
            ssp_at_z_met = (
                (1 - fz) * (1 - fa) * ssp_phot_at_z[iz, ia]
                + fz * (1 - fa) * ssp_phot_at_z[iz + 1, ia]
                + (1 - fz) * fa * ssp_phot_at_z[iz, ia + 1]
                + fz * fa * ssp_phot_at_z[iz + 1, ia + 1]
            )
        else:
            if _use_alpha_fe:
                lz = lz + _A2Z * afe
            if _use_smooth_z:
                zw = _clw(lz, ssp_lgmet, _lgmet_scat)
                ssp_at_z_met = jnp.einsum("m,maf->af", zw, ssp_phot_at_z)
            else:
                log_z_c = jnp.clip(lz, ssp_lgmet[0], ssp_lgmet[-1])
                idx = jnp.clip(jnp.searchsorted(ssp_lgmet, log_z_c) - 1, 0, len(ssp_lgmet) - 2)
                frac = (log_z_c - ssp_lgmet[idx]) / (ssp_lgmet[idx + 1] - ssp_lgmet[idx])
                ssp_at_z_met = (1.0 - frac) * ssp_phot_at_z[idx] + frac * ssp_phot_at_z[idx + 1]

        # Dust at effective wavelengths (same as fixed-z)
        _law_kw = dict(n_slope=dn, dust_bump_strength=bump, dust_delta=delta, dust_Rv=rv)
        if _is_single_dust:
            k = law_bc_fn(eff_waves_rest.astype(dt), **_law_kw)
            trans_1d = f_obs + (1.0 - f_obs) * jnp.exp(-tv * k)
            flux_attenuated = jnp.einsum("i,if->f", weights, ssp_at_z_met) * trans_1d
        elif _dust_exact:
            k_bc = law_bc_fn(eff_waves_rest.astype(dt), **_law_kw)
            k_diff = law_diff_fn(eff_waves_rest.astype(dt), **_law_kw)
            tau = dust_age_w[:, None] * tv1 * k_bc[None, :] + tv2 * k_diff[None, :]
            dust = f_obs + (1.0 - f_obs) * jnp.exp(-tau)
            flux_attenuated = jnp.einsum("i,if,if->f", weights, dust, ssp_at_z_met)
        else:
            # Fast two-component dust (same as fixed-z)
            k_bc = law_bc_fn(eff_waves_rest.astype(dt), **_law_kw)
            k_diff = law_diff_fn(eff_waves_rest.astype(dt), **_law_kw)
            trans_bc = jnp.exp(-tv1 * k_bc)
            trans_diff = jnp.exp(-tv2 * k_diff)

            csp_young = jnp.einsum("i,if->f", weights * young_mask, ssp_at_z_met)
            csp_old = jnp.einsum("i,if->f", weights * old_mask, ssp_at_z_met)

            flux_no_geom = trans_diff * (trans_bc * csp_young + csp_old)
            flux_intrinsic_for_geom = csp_young + csp_old
            flux_attenuated = f_obs * flux_intrinsic_for_geom + (1.0 - f_obs) * flux_no_geom

        # Estimate L_absorbed
        flux_intrinsic = jnp.einsum("i,if->f", weights, ssp_at_z_met)
        diff_flux = (flux_intrinsic - flux_attenuated) * lsun  # erg/s/Hz per band
        if _eff_bw is not None:
            L_absorbed_stellar = jnp.sum(diff_flux * _eff_bw)
        else:
            L_absorbed_stellar = jnp.sum(diff_flux)

        return flux_attenuated, L_absorbed_stellar, weights

    def _hybrid_phot_ztable_body(
        sfr_on_ssp,
        log_z_abs,
        redshift,
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
        """Hybrid z-table kernel body: stellar + exact non-stellar."""
        # === STEP 1: Stellar photometry from z-table ===
        flux_attenuated, L_absorbed_stellar, _weights = _stellar_phot_ztable(
            sfr_on_ssp,
            log_z_abs,
            redshift,
            tau_bc,
            tau_diff,
            dust_slope,
            f_obscuration,
            dust_bump_strength,
            dust_delta,
            dust_Rv,
            alpha_fe,
            tau_v=tau_v,
        )

        # === STEP 2: Non-stellar SED (same as build_hybrid_photometry) ===
        # NOTE: This is exact same non-stellar computation as the fixed-z hybrid.
        # The only difference is that stellar photometry is interpolated from
        # z-table at each step, while fixed-z version uses precomputed values.
        _has_any_nonstell = (
            has_nebular or has_shock or has_dust_em_full or has_agn_full or has_radio or has_xray
        )

        non_stellar_phot = jnp.zeros(n_filters, dtype=jnp.float64)

        if _has_any_nonstell:
            non_stellar_sed = jnp.zeros_like(ssp_wave_f64)
            L_abs_neb = jnp.float64(0.0)

            if has_nebular:
                _sfr_last = sfr_on_ssp[-1]  # SFR (Msun/yr), not mass weight
                neb_sed, _neb_lines = nebular_emission(
                    sfr_on_ssp=sfr_on_ssp,
                    logU=neb_logU,
                    log_z=log_z_abs,
                    fesc=neb_fesc,
                    fesc_lya=neb_fesc_lya,
                    backend=nebular_backend,
                    return_lines=True,
                )
                non_stellar_sed = non_stellar_sed + neb_sed

            if has_shock:
                shock_sed = shock_emission(
                    frac=shock_frac,
                    velocity=shock_velocity,
                    log_density=shock_log_density,
                    b_over_sqrt_n=shock_b_over_sqrt_n,
                    abundance=shock_abundance,
                    component=shock_component,
                )
                non_stellar_sed = non_stellar_sed + shock_sed

            if has_agn_full:
                agn_sed = agn_emission(
                    log_lbol=agn_log_lbol,
                    alpha=agn_alpha,
                    T_torus=agn_T_torus,
                    tau_torus=agn_tau_torus,
                    torus_frac=agn_torus_frac,
                    log_mbh=agn_log_mbh,
                    log_ledd=agn_log_ledd,
                    polar_ebv=agn_polar_ebv,
                    cos_inc=agn_cos_inc,
                    polar_oa=agn_polar_oa,
                    frac=agn_frac,
                    a_spin=agn_a_spin,
                    tau_skirtor=agn_tau_skirtor,
                    p_skirtor=agn_p_skirtor,
                    q_skirtor=agn_q_skirtor,
                    oa_skirtor=agn_oa_skirtor,
                    T_hot=agn_T_hot,
                    T_warm=agn_T_warm,
                    frac_hot=agn_frac_hot,
                    f_hard=agn_f_hard,
                    gamma_warm=agn_gamma_warm,
                    kt_warm=agn_kt_warm,
                    gamma_hard=agn_gamma_hard,
                    kt_hot=agn_kt_hot,
                    r_warm_ratio=agn_r_warm_ratio,
                    func=agn_model_fn_full,
                )
                non_stellar_sed = non_stellar_sed + agn_sed

            if has_radio:
                mstar = jnp.exp(10.0 * jnp.log(10.0))  # dummy, not used in hybrid
                radio_sed = radio_emission(
                    sfr=sfr_on_ssp[-1],
                    mstar=mstar,
                    log_lbol_agn=agn_log_lbol if has_agn_full else 10.0,
                    frac_agn=agn_frac if has_agn_full else 0.0,
                    mode=_radio_sfr_mode,
                    q_ir=radio_q_ir,
                    alpha_sf=radio_alpha_sf,
                    loudness=radio_loudness,
                    alpha_agn=radio_alpha_agn,
                    T_e=radio_T_e,
                    freefree=_include_freefree,
                    alpha_ff=radio_alpha_ff,
                    z=redshift,
                )
                non_stellar_sed = non_stellar_sed + radio_sed

            if has_xray:
                mstar = jnp.exp(10.0 * jnp.log(10.0))  # dummy
                xray_sed = xray_emission(
                    sfr=sfr_on_ssp[-1],
                    mstar=mstar,
                    log_lbol_agn=agn_log_lbol if has_agn_full else 10.0,
                    gamma_agn=xray_gamma_agn,
                    alpha_ox=xray_alpha_ox,
                    gamma_hmxb=xray_gamma_hmxb,
                    gamma_lmxb=xray_gamma_lmxb,
                    E_cut=xray_E_cut,
                )
                non_stellar_sed = non_stellar_sed + xray_sed

            # Dust attenuation on non-stellar
            if has_nebular or has_shock:
                attenuated_nonstell_sed, L_abs_neb = attenuate_emission(
                    non_stellar_sed,
                    dust_tau_bc=tau_bc,
                    dust_tau_diff=tau_diff,
                    dust_slope=dust_slope,
                    dust_bump=dust_bump_strength,
                    dust_delta=dust_delta,
                    dust_Rv=dust_Rv,
                    redshift=redshift,
                    law_bc_fn=_neb_bc_fn,
                    law_diff_fn=law_diff_fn if not _is_single_dust else None,
                    mode=_neb_dust_mode,
                )
                non_stellar_sed = attenuated_nonstell_sed
            else:
                L_abs_neb = jnp.float64(0.0)

            # Dust IR emission (uses L_absorbed)
            if has_dust_em_full:
                L_abs_total = L_absorbed_stellar + L_abs_neb
                ir_sed = dust_ir_emission(
                    L_absorbed=L_abs_total,
                    T=dust_T,
                    beta_ir=dust_beta_ir,
                    eta=dust_eta_balance,
                    alpha_mir=dust_alpha_mir,
                    alpha_dale=dust_alpha_dale,
                    umin=dust_umin,
                    gamma_dl=dust_gamma_dl,
                    qpah=dust_qpah,
                    model=_dust_model_name,
                )
                non_stellar_sed = non_stellar_sed + ir_sed

            # Integrate non-stellar through filters
            # Interpolate non-stellar SED to observed frame + filter integration
            from tengri.models.observation.photometry import compute_flux_density

            non_stellar_phot = compute_flux_density(
                wave_rest=ssp_wave_f64,
                sed=non_stellar_sed,
                filter_waves=filter_waves_list,
                filter_trans=filter_trans_list,
                redshift=redshift,
                use_wave_res=False,
            )

        # === STEP 3: Combine and convert to erg/s/cm²/Hz ===
        # Interpolate z-table to get flux_scale at current z
        _, _, flux_scale = interpolate_ztable(
            ztable.ssp_phot_table,
            ztable.eff_waves_rest_table,
            ztable.flux_scale_table,
            ztable.z_grid,
            jnp.asarray(redshift, dtype=dt),
        )
        flux_scale = jnp.asarray(flux_scale, dtype=jnp.float64)

        # Stellar contribution: flux_attenuated is in Lsun/Hz (from einsum)
        stellar_phot = flux_scale * flux_attenuated * lsun  # erg/s/cm²/Hz

        # Total photometry
        total_phot = stellar_phot + non_stellar_phot

        return total_phot

    @jax.jit
    def hybrid_phot_ztable_fused(params):
        """params dict → photometry (end-to-end JIT for z-table path)."""
        p = get_internal_params(params, param_map, spec, has_field)

        # SFH computation
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
        sfr_on_ssp = jnp.interp(ssp_log_ages_yr_cap, log_age_grid, sfr)

        # Get redshift from params
        z = p.get("redshift", 0.1)

        # Dispatch to the inner kernel
        if _is_single_dust:
            return hybrid_phot_ztable(
                sfr_on_ssp,
                p["log_z_abs"],
                z,
                p.get("tau_v", 0.0),
                p.get("dust_slope", -0.7),
                **{
                    k: p.get(k, v)
                    for k, v in [
                        ("f_obscuration", 0.0),
                        ("dust_bump_strength", 0.0),
                        ("dust_delta", 0.0),
                        ("dust_Rv", 3.1),
                        ("alpha_fe", 0.0),
                        ("dust_T", 35.0),
                        ("dust_beta_ir", 1.6),
                        ("dust_eta_balance", 1.0),
                        ("agn_log_lbol", 10.0),
                        ("agn_alpha", -1.0),
                        ("agn_T_torus", 1000.0),
                        ("agn_tau_torus", 5.0),
                        ("agn_torus_frac", 0.5),
                        ("agn_log_mbh", 7.0),
                        ("agn_log_ledd", -1.0),
                    ]
                },
            )
        return hybrid_phot_ztable(
            sfr_on_ssp,
            p["log_z_abs"],
            z,
            p.get("tau_bc", 0.0),
            p.get("tau_diff", 0.0),
            p.get("dust_slope", -0.7),
            **{
                k: p.get(k, v)
                for k, v in [
                    ("f_obscuration", 0.0),
                    ("dust_bump_strength", 0.0),
                    ("dust_delta", 0.0),
                    ("dust_Rv", 3.1),
                    ("alpha_fe", 0.0),
                    ("dust_T", 35.0),
                    ("dust_beta_ir", 1.6),
                    ("dust_eta_balance", 1.0),
                    ("agn_log_lbol", 10.0),
                    ("agn_alpha", -1.0),
                    ("agn_T_torus", 1000.0),
                    ("agn_tau_torus", 5.0),
                    ("agn_torus_frac", 0.5),
                    ("agn_log_mbh", 7.0),
                    ("agn_log_ledd", -1.0),
                ]
            },
        )

    return hybrid_phot_ztable_fused


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

    # Full-precision wavelength arrays for non-stellar components
    ssp_wave_f64 = model.ssp_data.ssp_wave
    rest_wave_f64 = model._rest_wavelength
    _needs_extension = rest_wave_f64 is not model.ssp_data.ssp_wave

    # Build the non-stellar sub-closure once (outside JIT).
    # All per-component flags, imports, and callables are captured inside.
    from tengri.core.nonstell import build_nonstell_fn

    _law_diff_for_nonstell = law_diff_fn if not _is_single_dust else law_bc_fn
    nonstell_fn = build_nonstell_fn(
        model, law_bc_fn, _law_diff_for_nonstell, ssp_wave_f64, rest_wave_f64
    )

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

        # --- 2–9. All non-stellar components (nebular, shock, dust IR, AGN, radio, X-ray) ---
        # nonstell_fn was built once at closure time by build_nonstell_fn(); calling it
        # here adds all enabled components and returns the full panchromatic SED.
        return nonstell_fn(weights, p, sed_atten, sed_intr)

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
    """Build a single JIT function: (sfr_on_ssp, params dict) → observed photometry.

    Fuses the Tier 2 pipeline into one ``@jax.jit`` scope.  SFH is evaluated
    *outside* the JIT (caller computes ``sfr_on_ssp`` and passes it as a traced
    array), which avoids re-triggering XLA recompilation when the SFH type changes.

    Steps inside the JIT:

    1. Parameter translation (public → internal names + unit conversion)
    2. Metallicity interpolation (linear or smooth, single-Z)
    3. Compositional rest-frame SED kernel (dust, nebular, AGN, ...)
    4. Filter integration (loop unrolled by JAX tracer)
    5. Optional IGM absorption

    Parameters
    ----------
    model : SEDModel
        Fully initialized Model with filters and fixed redshift.

    Returns
    -------
    callable or None
        JIT-compiled function: ``(sfr_on_ssp, params_dict) -> photometry_array``.
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
    ssp_ages_yr = model.ssp_ages_yr
    # Panchromatic wavelength grid (extended if radio/xray enabled)
    rest_wave = model._rest_wavelength

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
    # Use panchromatic grid if available (when radio/xray enabled), else SSP grid
    igm_trans_full = None
    if apply_igm and not is_free_z:
        from tengri.models.igm import igm_transmission

        wave_obs_full = rest_wave * (1.0 + z_fixed)
        igm_trans_full = igm_transmission(wave_obs_full, z_fixed)

    # For free-z: need luminosity_distance inside JIT
    if is_free_z:
        from tengri.utils.cosmology import luminosity_distance as _lum_dist

        if apply_igm:
            from tengri.models.igm import igm_transmission as _igm_fn

    # --- Shared SED computation (sfr_on_ssp pre-computed by caller) ---
    def _compute_rest_sed(sfr_on_ssp, params):
        """sfr_on_ssp, params → (rest_sed, redshift_value).

        ``sfr_on_ssp`` is the SFH already evaluated on the SSP age grid.
        Keeping SFH outside this function prevents the SFH type from entering
        the JIT closure, so switching SFH models does not cause recompilation.
        """
        p = get_internal_params(params, param_map, spec, has_field)

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

        # Always pass current SFR — needed by nebular (Q_H scaling) and X-ray.
        p = {**p, "_sfr_current": sfr_on_ssp[-1]}

        rest_sed = rest_sed_kernel(weights, ssp_flux_at_z, p)
        z = p.get("redshift", z_fixed if z_fixed is not None else 0.0)
        return rest_sed, z

    if is_free_z:

        @jax.jit
        def fused_tier2_phot(sfr_on_ssp, params):
            """sfr_on_ssp, params dict → observed photometry (free z)."""
            rest_sed, z = _compute_rest_sed(sfr_on_ssp, params)
            dl_cm = _lum_dist(z)

            if apply_igm:
                wave_obs = rest_wave * (1.0 + z)
                rest_sed = rest_sed * _igm_fn(wave_obs, z)

            fluxes = []
            for fw, ft in zip(filter_waves, filter_trans):
                f = compute_flux_density(rest_sed, rest_wave, fw, ft, z, dl_cm)
                fluxes.append(f)
            return jnp.array(fluxes)

    else:

        @jax.jit
        def fused_tier2_phot(sfr_on_ssp, params):
            """sfr_on_ssp, params dict → observed photometry (fixed z)."""
            rest_sed, _z = _compute_rest_sed(sfr_on_ssp, params)

            if igm_trans_full is not None:
                rest_sed = rest_sed * igm_trans_full

            fluxes = []
            for fw, ft in zip(filter_waves, filter_trans):
                f = compute_flux_density(rest_sed, rest_wave, fw, ft, z_fixed, dl_cm_fixed)
                fluxes.append(f)
            return jnp.array(fluxes)

    return fused_tier2_phot


def build_fused_tier2_spectrum(model):
    """Build a single JIT function: (sfr_on_ssp, params dict) → observed spectrum.

    Like :func:`build_fused_tier2_photometry` but for spectroscopy.
    SFH is evaluated *outside* the JIT; the caller passes ``sfr_on_ssp``
    as a traced array so the JIT closure is SFH-type-independent.

    Requires ``precompute_spectroscopy()`` to have been called (for
    the wavelength grid) or an Observation with spectroscopy config.

    Parameters
    ----------
    model : SEDModel
        Fully initialized Model with spectroscopy config.

    Returns
    -------
    callable or None
        JIT-compiled function: ``(sfr_on_ssp, params_dict) -> spectrum_array``.
    """
    if model._compositional.rest_sed is None:
        return None

    from tengri.core.param_translate import get_internal_params
    from tengri.core.sed_pipeline import interp_met_alpha_dispatch, interp_metallicity
    from tengri.models.observation.spectrum import compute_spectrum
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
    ssp_ages_yr = model.ssp_ages_yr
    # Panchromatic wavelength grid (extended if radio/xray enabled)
    rest_wave = model._rest_wavelength


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

    # Shared SED computation (sfr_on_ssp pre-computed by caller)
    def _compute_rest_sed_spec(sfr_on_ssp, params):
        p = get_internal_params(params, param_map, spec, has_field)
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
        p = {**p, "_sfr_current": sfr_on_ssp[-1]}
        rest_sed = rest_sed_kernel(weights, ssp_flux_at_z, p)
        z = p.get("redshift", z_fixed if z_fixed is not None else 0.0)
        return rest_sed, z

    if is_free_z:

        @jax.jit
        def fused_tier2_spec(sfr_on_ssp, params):
            """sfr_on_ssp, params dict → observed spectrum (free z)."""
            rest_sed, z = _compute_rest_sed_spec(sfr_on_ssp, params)
            dl_cm = _lum_dist_spec(z)
            return compute_spectrum(rest_sed, rest_wave, wave_obs, z, dl_cm)

    else:

        @jax.jit
        def fused_tier2_spec(sfr_on_ssp, params):
            """sfr_on_ssp, params dict → observed spectrum (fixed z)."""
            rest_sed, _z = _compute_rest_sed_spec(sfr_on_ssp, params)
            return compute_spectrum(rest_sed, rest_wave, wave_obs, z_fixed, dl_cm_fixed)

    return fused_tier2_spec


def build_hybrid_spectrum(model):
    """Build hybrid spectrum kernel: precomputed SSP stellar + exact non-stellar.

    Stellar spectrum evaluated via precomputed SSP interpolated to spectral
    pixels (exact on the pixel grid). Non-stellar components evaluated at
    full wavelength resolution via rest_sed_kernel, then interpolated to
    spectral pixels.

    This kernel bridges precomputed stellar (fast) and exact non-stellar
    (accurate):
    - Stellar CSP: precomputed on spectral pixels, no wavelength integral
    - Non-stellar: rest_sed_kernel at full wavelength, interpolated to pixels
    - Use for science models where non-stellar accuracy is critical

    Parameters
    ----------
    model : SEDModel
        The model instance with spectroscopy precomputation.

    Returns
    -------
    callable or None
        JIT-compiled function: (sfr_on_ssp, params) → spectrum array.
        Returns None if spectroscopy is not precomputed.
    """
    if model._precomputed.spectroscopy is None:
        return None

    from tengri.core.param_translate import get_internal_params
    from tengri.core.sed_pipeline import (
        interp_met_alpha_dispatch,
        interp_metallicity,
    )
    from tengri.models.dust.attenuation import resolve_dust_law
    from tengri.models.sps.dsps_wrapper import compute_csp_weights
    from tengri.utils.conversions import lnu_to_fnu

    # Precomputed spectroscopic data
    precomp_spec = model._precomputed.spectroscopy
    ssp_on_pixels = precomp_spec.ssp_on_pixels.astype(model._forward_dtype)
    wave_rest_pixels = precomp_spec.wave_rest_pixels
    z_fixed = model._z_fixed
    dl_cm_fixed = model._dl_cm_fixed

    # Model configuration
    ssp_ages_yr = model.ssp_ages_yr
    _is_single_dust = model._dust_model == "single_component"
    law_bc_fn = resolve_dust_law(model._dust_law_bc)
    if not _is_single_dust:
        law_diff_fn = resolve_dust_law(model._dust_law_diff)

    # Alpha enhancement
    _use_alpha_fe = model.spec.alpha_fe_evolving or "met_alpha_fe" in model.spec.free_params
    if _use_alpha_fe:
        from tengri.models.sps.dsps_wrapper import has_alpha_grid

        _has_alpha = has_alpha_grid(model.ssp_data)
    else:
        _has_alpha = False

    # Non-stellar kernel
    rest_sed_kernel = model._compositional.rest_sed
    param_map = model._param_map
    spec = model.spec
    has_field = model._has_field
    rest_wave = model._rest_wavelength

    # Redshift handling
    is_free_z = z_fixed is None
    if is_free_z:
        from tengri.utils.cosmology import luminosity_distance as _lum_dist

    # === Core kernel body (shared for single and two-component dust) ===

    def _hybrid_spec_body(
        sfr_on_ssp,
        params,
        log_z_abs,
        dust_params,  # tuple: (tau_bc,) or (tau_bc, tau_diff)
        alpha_fe=0.0,
    ):
        """Compute hybrid spectrum: precomputed stellar + non-stellar."""
        p = get_internal_params(params, param_map, spec, has_field)
        weights = compute_csp_weights(sfr_on_ssp, ssp_ages_yr)

        # Metallicity interpolation
        if _use_alpha_fe and _has_alpha:
            alpha_fe_val = p.get("alpha_fe", 0.0)
            ssp_flux_at_z = interp_met_alpha_dispatch(model, p["log_z_abs"], alpha_fe_val)
        else:
            ssp_flux_at_z = interp_metallicity(model, p["log_z_abs"])

        # Stellar spectrum on pixels
        stellar_spec = jnp.dot(weights, ssp_on_pixels[0])  # (n_pix,)

        # Apply dust attenuation to stellar spectrum
        if _is_single_dust:
            k_lambda = law_bc_fn(wave_rest_pixels)
            atten = jnp.exp(-dust_params[0] * k_lambda)
        else:
            k_bc = law_bc_fn(wave_rest_pixels)
            k_diff = law_diff_fn(wave_rest_pixels)
            atten = jnp.exp(-dust_params[0] * k_bc - dust_params[1] * k_diff)

        stellar_spec_dust = stellar_spec * atten

        # Non-stellar SED at full wavelength
        p_ns = p.copy()
        if model._xray_enabled:
            p_ns["_sfr_current"] = sfr_on_ssp[-1]
        rest_sed_full = rest_sed_kernel(weights, ssp_flux_at_z, p_ns)

        # Interpolate to spectral pixels
        non_stellar_spec_rest = jnp.interp(
            wave_rest_pixels, rest_wave, rest_sed_full, left=0.0, right=0.0
        )

        # Combine stellar + non-stellar
        total_spec_rest = stellar_spec_dust + non_stellar_spec_rest

        # Cosmological scaling
        if is_free_z:
            z = p.get("redshift", 0.0)
            dl_cm = _lum_dist(z)
        else:
            z = z_fixed
            dl_cm = dl_cm_fixed

        flux_scale = lnu_to_fnu(1.0, dl_cm, z)
        return flux_scale * total_spec_rest

    # === Define kernel with correct dust signature ===

    if _is_single_dust:

        @jax.jit
        def hybrid_spec(sfr_on_ssp, params):
            """Single-dust hybrid spectrum."""
            p = get_internal_params(params, param_map, spec, has_field)
            return _hybrid_spec_body(sfr_on_ssp, params, p["log_z_abs"], (p.get("tau_v", 0.0),))

    else:

        @jax.jit
        def hybrid_spec(sfr_on_ssp, params):
            """Two-dust hybrid spectrum."""
            p = get_internal_params(params, param_map, spec, has_field)
            return _hybrid_spec_body(
                sfr_on_ssp,
                params,
                p["log_z_abs"],
                (p.get("tau_bc", 0.0), p.get("tau_diff", 0.0)),
            )

    return hybrid_spec
