"""Fused JIT kernel builders for fast photometry and spectroscopy.

These factory functions build @jax.jit closures that capture precomputed
arrays (SSP grids, dust weights, effective wavelengths) at build time.
The returned functions take only per-call parameters (SFR weights, dust
params) as arguments.

Extracted from ``Model`` methods to keep model.py focused on orchestration.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp

# -------------------------------------------------------------------
# Compatibility check
# -------------------------------------------------------------------


def is_fused_compatible(model):
    """Check if the fused JIT kernel can handle the current model config.

    Respects ``model._approx`` settings: if a component's approximation
    is disabled, the fused kernel cannot handle it and the model falls
    back to the exact path.

    Emits warnings for each active approximation so the user knows
    what trade-offs are being made.

    Parameters
    ----------
    model : Model
        The model instance to check.

    Returns
    -------
    bool
        True if the fused kernel can be used.
    """
    reasons = []  # reasons to fall back to exact

    # Dust attenuation approximation
    if not model._approx["dust_attenuation"]:
        reasons.append("dust_attenuation approx disabled by user")

    # Nebular: Cloudy with free params can't be fused
    neb_ok = model._nebular_backend is None or not getattr(
        model._nebular_backend, "has_free_params", False
    )
    if not neb_ok:
        reasons.append("nebular emission (Cloudy) requires full SED")

    # Dust emission: MBB/dale2014 supported if approx enabled
    if model._dust_emission_model is not None:
        if not model._approx["dust_emission"]:
            reasons.append("dust_emission approx disabled by user")
        elif model._dust_emission_model not in ("modified_blackbody", "dale2014"):
            reasons.append(
                f"dust_emission='{model._dust_emission_model}' not supported "
                f"in fused kernel (only modified_blackbody, dale2014)"
            )

    # AGN: legacy mode (agn_frac) forces exact path (needs L_bol from
    # full SED integral). Parametric mode (agn_log_lbol) is fused-compatible.
    if model._agn_model is not None and not model._agn_parametric:
        reasons.append("AGN (legacy agn_frac mode) requires full SED for bolometric luminosity")

    # IGM: can be precomputed at effective wavelengths if approx enabled
    if model._apply_igm and not model._approx["igm"]:
        reasons.append("igm approx disabled by user")

    if reasons:
        if model._precomp is not None:
            warnings.warn(
                "Fused kernel disabled, using exact path (slower). "
                f"Reasons: {'; '.join(reasons)}. "
                "Set approx=True or remove incompatible components for "
                "~10-50x speedup.",
                UserWarning,
                stacklevel=3,
            )
        return False

    # Emit approximation warnings for active components
    active_approx = []
    if model._approx["dust_attenuation"]:
        active_approx.append(
            "dust attenuation at filter effective wavelengths "
            "(<3% error for most laws, ~36% for SMC)"
        )
    if model._dust_emission_model in ("modified_blackbody", "dale2014"):
        active_approx.append(
            f"dust emission ({model._dust_emission_model}) with approximate "
            "L_absorbed from broadband fluxes"
        )
    if model._apply_igm and model._approx["igm"]:
        active_approx.append("IGM absorption precomputed at filter effective wavelengths")
    if model._agn_model is not None and model._agn_parametric:
        active_approx.append(
            "AGN (parametric agn_log_lbol) evaluated at filter effective "
            "wavelengths. The AGN SED shape (power-law disc + blackbody "
            "torus) varies strongly across optical-IR; effective-wavelength "
            "approximation may be less accurate than for dust (~10-20% error "
            "in broadband fluxes for AGN-dominated bands)"
        )

    if active_approx:
        warnings.warn(
            "Fused kernel active with approximations: "
            + "; ".join(active_approx)
            + ". Set approx=False to disable.",
            UserWarning,
            stacklevel=3,
        )

    return True


# -------------------------------------------------------------------
# Fused photometry kernel (fixed redshift)
# -------------------------------------------------------------------


def build_fused_photometry(model):
    """Build a single JIT function: SFR-on-SSP -> photometry.

    Captures all constants (SSP grid, precomp, dust weights) in the
    closure so XLA can fuse metallicity interpolation, dust, and
    weighted sum into one optimized kernel with no intermediate
    array materializations.

    Supports all registered dust laws (calzetti, kriek_conroy, smc, etc.)
    via captured law functions. For power-law dust, XLA constant-folds
    the curve evaluation to identical code as the old hardcoded path.

    Parameters
    ----------
    model : Model
        The model instance providing config and precomputed arrays.

    Returns
    -------
    callable
        JIT-compiled function: (sfr_on_ssp, log_z_abs, tau_bc, tau_diff,
        dust_slope, ...) -> photometry array.
    """
    from tengri.models.dust.attenuation import get_dust_law
    from tengri.models.sps.dsps_wrapper import LSUN_ERG_PER_S

    dt = model._forward_dtype
    precomp = model._precomp
    ssp_phot = precomp.ssp_phot.astype(dt)
    ssp_lgmet = model.ssp_data.ssp_lgmet.astype(dt)
    eff_waves_rest = precomp.effective_wavelengths_rest.astype(dt)
    _is_single_dust = model._dust_model == "single_component"
    _dust_exact = getattr(model, "_dust_approx", "fast") == "exact"
    if not _is_single_dust:
        if _dust_exact:
            # Smooth sigmoid weights for exact two-component dust
            dust_age_w = model._dust_age_weights.astype(dt)
        else:
            # Hard threshold for fast two-CSP decomposition (default)
            _t_birth = 1e7  # 10 Myr — Charlot & Fall (2000)
            young_mask = (model.ssp_ages_yr < _t_birth).astype(dt)
            old_mask = dt.type(1.0) - young_mask
    flux_scale = dt.type(precomp.flux_scale)
    _csp_use_matrix = model._csp_integration == "log_interp"
    if _csp_use_matrix:
        _csp_mat = model._csp_matrix.astype(dt)
    else:
        _age_dt = model._csp_age_dt.astype(dt)  # precomputed CSP bin widths
    lsun = dt.type(LSUN_ERG_PER_S)

    # Capture dust law functions (pure JAX, JIT-traceable)
    law_bc_fn = get_dust_law(model._dust_law_bc)
    if not _is_single_dust:
        law_diff_fn = get_dust_law(model._dust_law_diff)

    from tengri.models.sps.dsps_wrapper import (
        _ALPHA_TO_Z_COEFF as _A2Z,
        has_alpha_grid,
    )

    # Alpha-enhanced SSP grid detection (4D vs 3D)
    _has_alpha = has_alpha_grid(model.ssp_data)
    if _has_alpha:
        ssp_alpha_fe = model.ssp_data.ssp_alpha_fe.astype(dt)

    # effective_metallicity correction is opt-in: only applied when the user
    # explicitly makes met_alpha_fe (or evolving variant) a free parameter.
    # When alpha_fe is Fixed(0), skip the dispatch and use plain interpolation.
    _use_alpha_fe = model.spec.alpha_fe_evolving or "met_alpha_fe" in model.spec.free_params

    # Metallicity interpolation mode for fused kernel
    _use_smooth_z = model._met_interp == "smooth"
    _lgmet_scat = dt.type(model._lgmet_scatter)
    if _use_smooth_z:
        from tengri.models.sps.dsps_wrapper import compute_lgmet_weights as _clw

    # IGM: precomputed at effective wavelengths (constant for fixed z)
    has_igm = model._igm_at_eff is not None
    if has_igm:
        igm_trans = model._igm_at_eff.astype(dt)

    # Dust emission: precompute constants for MBB at effective wavelengths
    has_dust_em = model._dust_emission_model in ("modified_blackbody", "dale2014")
    if has_dust_em:
        # Precompute frequency at effective wavelengths (constant)
        eff_waves_cm = eff_waves_rest * dt.type(1e-8)
        eff_nu = dt.type(2.99792458e10) / eff_waves_cm  # Hz
        nu_ref_250um = dt.type(2.99792458e10 / 250.0e-4)

    # AGN: capture model function for evaluation at effective wavelengths
    has_agn = model._agn_model is not None and model._agn_parametric
    if has_agn:
        from tengri.models.agn import get_agn_model

        agn_model_fn = get_agn_model(model._agn_model)

    # Single-component: signature is (sfr, log_z, tau_v, dust_slope, ...)
    # Two-component:    signature is (sfr, log_z, tau_bc, tau_diff, dust_slope, ...)
    # Both defined as separate closures to keep signatures clean for callers.

    if _is_single_dust:

        @jax.jit
        def fused_phot(
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
        ):
            return _fused_phot_body(
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
                tau_v=tau_v,
            )

    else:

        @jax.jit
        def fused_phot(
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
        ):
            return _fused_phot_body(
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
            )

    def _fused_phot_body(
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
        tau_v=0.0,
    ):
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

        # CSP weights (precomputed bin widths; method set at model init)
        weights = _csp_mat @ sfr if _csp_use_matrix else sfr * _age_dt

        # Metallicity + alpha interpolation
        if _has_alpha:
            # 4D bilinear: (Z, [α/Fe]) interpolation on precomputed photometry
            # ssp_phot shape: (n_met, n_alpha, n_age, n_filt)
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
            # 3D grid: apply effective_metallicity correction only when alpha_fe
            # is explicitly a free parameter (opt-in, not default).
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

        # Dust: evaluate configurable curves at effective wavelengths
        _law_kw = dict(n_slope=dn, dust_bump_strength=bump, dust_delta=delta, dust_Rv=rv)
        if _is_single_dust:
            # Single-component: age-independent screen → factor out of einsum
            k = law_bc_fn(eff_waves_rest, **_law_kw)
            trans_1d = f_obs + (1.0 - f_obs) * jnp.exp(-tv * k)
            flux_attenuated = jnp.einsum("i,if->f", weights, ssp_at_z) * trans_1d
        elif _dust_exact:
            # Exact: smooth sigmoid age weights — full (n_ages, n_wave) outer product
            k_bc = law_bc_fn(eff_waves_rest, **_law_kw)
            k_diff = law_diff_fn(eff_waves_rest, **_law_kw)
            tau = dust_age_w[:, None] * tv1 * k_bc[None, :] + tv2 * k_diff[None, :]
            dust = f_obs + (1.0 - f_obs) * jnp.exp(-tau)
            flux_attenuated = jnp.einsum("i,if,if->f", weights, dust, ssp_at_z)
        else:
            # Fast two-CSP decomposition (Charlot & Fall hard threshold):
            #   flux = trans_diff * (trans_bc * CSP_young + CSP_old)
            # Two 1D einsums + 1D dust — no (n_ages, n_wave) intermediate.
            k_bc = law_bc_fn(eff_waves_rest, **_law_kw)
            k_diff = law_diff_fn(eff_waves_rest, **_law_kw)
            trans_bc = jnp.exp(-tv1 * k_bc)  # (n_filt,)
            trans_diff = jnp.exp(-tv2 * k_diff)  # (n_filt,)

            csp_young = jnp.einsum("i,if->f", weights * young_mask, ssp_at_z)
            csp_old = jnp.einsum("i,if->f", weights * old_mask, ssp_at_z)

            flux_no_geom = trans_diff * (trans_bc * csp_young + csp_old)
            flux_intrinsic_for_geom = csp_young + csp_old
            flux_attenuated = f_obs * flux_intrinsic_for_geom + (1.0 - f_obs) * flux_no_geom

        if has_dust_em:
            # Approximate dust emission in the fused kernel:
            # 1. L_stellar (intrinsic, no dust) at effective wavelengths
            flux_intrinsic = jnp.einsum("i,if->f", weights, ssp_at_z)
            # 2. L_absorbed ~ sum(L_intrinsic - L_attenuated) across bands
            #    This is an approximation -- exact would integrate over full SED
            L_absorbed_approx = jnp.sum(flux_intrinsic - flux_attenuated) * lsun
            L_absorbed_approx = jnp.maximum(L_absorbed_approx, dt.type(0.0))
            L_ir = L_absorbed_approx * jnp.asarray(dust_eta_balance, dtype=dt)

            # 3. Modified blackbody at effective wavelengths
            T = jnp.asarray(dust_T, dtype=dt)
            beta = jnp.asarray(dust_beta_ir, dtype=dt)
            emissivity = (eff_nu / nu_ref_250um) ** beta
            x = jnp.clip(
                dt.type(6.62607015e-27) * eff_nu / (dt.type(1.380649e-16) * T),
                dt.type(0.0),
                dt.type(500.0),
            )
            bnu = (
                dt.type(2.0)
                * dt.type(6.62607015e-27)
                * eff_nu**3
                / dt.type(2.99792458e10) ** 2
                / (jnp.exp(x) - dt.type(1.0))
            )
            mbb_shape = emissivity * bnu  # (n_filters,)

            # 4. Normalize MBB to L_ir (approximate: use sum over filters)
            mbb_norm = jnp.sum(mbb_shape)
            mbb_norm_safe = jnp.maximum(mbb_norm, dt.type(1e-100))
            dust_em_flux = L_ir / lsun * mbb_shape / mbb_norm_safe

            flux_total = flux_attenuated + dust_em_flux
        else:
            flux_total = flux_attenuated

        # AGN contribution at effective wavelengths (parametric mode)
        if has_agn:
            # Evaluate AGN SED at filter effective wavelengths
            agn_lnu = agn_model_fn(
                eff_waves_rest,
                agn_log_lbol=agn_log_lbol,
                agn_frac=1.0,  # L_bol fully specified by agn_log_lbol
                agn_alpha=agn_alpha,
                agn_T_torus=agn_T_torus,
                agn_tau_torus=agn_tau_torus,
                agn_torus_frac=agn_torus_frac,
                agn_log_mbh=agn_log_mbh,
                agn_log_ledd=agn_log_ledd,
            )
            # agn_lnu is in Lsun/Hz, flux_total is in Lsun at eff wavelengths
            # Convert: L_nu [Lsun/Hz] -> add to broadband flux [Lsun]
            # The precomp SSP photometry is L_nu*dnu integrated through
            # the filter, so AGN L_nu is treated the same way (evaluated
            # at the effective wavelength as a representative value).
            flux_total = flux_total + agn_lnu

        # IGM absorption (precomputed at effective wavelengths)
        if has_igm:
            flux_total = flux_total * igm_trans

        return (flux_scale * flux_total * lsun).astype(jnp.float64)

    return fused_phot


# -------------------------------------------------------------------
# Fused spectrum kernel (fixed redshift)
# -------------------------------------------------------------------


def build_fused_spectrum(model):
    """Build a single JIT function: SFR-on-SSP -> spectrum.

    Same fusion approach as photometry but for spectroscopic pixels.
    Supports all dust laws, f_obscuration, and optional velocity broadening.

    Parameters
    ----------
    model : Model
        The model instance providing config and precomputed arrays.

    Returns
    -------
    callable
        JIT-compiled function: (sfr_on_ssp, log_z_abs, tau_bc, tau_diff,
        dust_slope, ...) -> spectrum array.
    """
    from tengri.models.dust.attenuation import get_dust_law
    from tengri.models.sps.dsps_wrapper import (
        _ALPHA_TO_Z_COEFF as _A2Z,
        LSUN_ERG_PER_S,
        has_alpha_grid,
    )

    fdt = model._forward_dtype
    precomp = model._spec_precomp
    ssp_on_pixels = precomp.ssp_on_pixels.astype(fdt)
    ssp_lgmet = model.ssp_data.ssp_lgmet.astype(fdt)

    _has_alpha_zt = has_alpha_grid(model.ssp_data)
    if _has_alpha_zt:
        ssp_alpha_fe_zt = model.ssp_data.ssp_alpha_fe.astype(fdt)
    _use_alpha_fe_spec = model.spec.alpha_fe_evolving or "met_alpha_fe" in model.spec.free_params
    wave_obs_pixels = precomp.wave_obs_pixels.astype(fdt)
    wave_rest_pixels = precomp.wave_rest_pixels.astype(fdt)
    _is_single_dust_spec = model._dust_model == "single_component"
    _dust_exact_spec = getattr(model, "_dust_approx", "fast") == "exact"
    if not _is_single_dust_spec:
        if _dust_exact_spec:
            dust_age_w = model._dust_age_weights.astype(fdt)
        else:
            _t_birth_spec = 1e7
            young_mask_spec = (model.ssp_ages_yr < _t_birth_spec).astype(fdt)
            old_mask_spec = fdt.type(1.0) - young_mask_spec
    flux_scale = fdt.type(precomp.flux_scale)
    _csp_use_matrix_spec = model._csp_integration == "log_interp"
    if _csp_use_matrix_spec:
        _csp_mat_spec = model._csp_matrix.astype(fdt)
    else:
        _age_dt_spec = model._csp_age_dt.astype(fdt)  # precomputed CSP bin widths
    lsun = fdt.type(LSUN_ERG_PER_S)
    n_pix = len(wave_obs_pixels)
    has_sigma_v = model._has_sigma_v

    # Capture dust law functions
    law_bc_fn_spec = get_dust_law(model._dust_law_bc)
    if not _is_single_dust_spec:
        law_diff_fn_spec = get_dust_law(model._dust_law_diff)

    # Precompute FFT frequencies for velocity broadening (only if needed)
    if has_sigma_v:
        fft_freq = jnp.fft.rfftfreq(n_pix).astype(fdt)
        dlnwave = jnp.log(wave_obs_pixels[1] / wave_obs_pixels[0]).astype(fdt)
        c_km_s = fdt.type(299792.458)

    # AGN: capture model function for evaluation at pixel wavelengths
    has_agn = model._agn_model is not None and model._agn_parametric
    if has_agn:
        from tengri.models.agn import get_agn_model

        agn_model_fn = get_agn_model(model._agn_model)

    def _fused_spec_body(
        sfr_on_ssp,
        log_z_abs,
        tau_bc,
        tau_diff,
        dust_slope,
        sigma_v,
        f_obscuration,
        dust_bump_strength,
        dust_delta,
        dust_Rv,
        alpha_fe,
        agn_log_lbol,
        agn_alpha,
        agn_T_torus,
        agn_tau_torus,
        agn_torus_frac,
        agn_log_mbh,
        agn_log_ledd,
        tau_v=0.0,
    ):
        sfr = sfr_on_ssp.astype(fdt)
        lz = jnp.asarray(log_z_abs, dtype=fdt)
        tv1 = jnp.asarray(tau_bc, dtype=fdt)
        tv2 = jnp.asarray(tau_diff, dtype=fdt)
        tv = jnp.asarray(tau_v, dtype=fdt)
        dn = jnp.asarray(dust_slope, dtype=fdt)
        f_obs = jnp.asarray(f_obscuration, dtype=fdt)
        bump = jnp.asarray(dust_bump_strength, dtype=fdt)
        delta = jnp.asarray(dust_delta, dtype=fdt)
        rv = jnp.asarray(dust_Rv, dtype=fdt)
        afe = jnp.asarray(alpha_fe, dtype=fdt)

        # CSP weights (precomputed bin widths; method set at model init)
        weights = _csp_mat_spec @ sfr if _csp_use_matrix_spec else sfr * _age_dt_spec

        # Metallicity + alpha interpolation
        if _has_alpha_zt:
            lz_c = jnp.clip(lz, ssp_lgmet[0], ssp_lgmet[-1])
            iz = jnp.clip(jnp.searchsorted(ssp_lgmet, lz_c) - 1, 0, len(ssp_lgmet) - 2)
            fz = (lz_c - ssp_lgmet[iz]) / (ssp_lgmet[iz + 1] - ssp_lgmet[iz])
            afe_c = jnp.clip(afe, ssp_alpha_fe_zt[0], ssp_alpha_fe_zt[-1])
            n_afe = len(ssp_alpha_fe_zt)
            ia = jnp.clip(jnp.searchsorted(ssp_alpha_fe_zt, afe_c) - 1, 0, n_afe - 2)
            fa = (afe_c - ssp_alpha_fe_zt[ia]) / (ssp_alpha_fe_zt[ia + 1] - ssp_alpha_fe_zt[ia])
            ssp_at_z = (
                (1 - fz) * (1 - fa) * ssp_on_pixels[iz, ia]
                + fz * (1 - fa) * ssp_on_pixels[iz + 1, ia]
                + (1 - fz) * fa * ssp_on_pixels[iz, ia + 1]
                + fz * fa * ssp_on_pixels[iz + 1, ia + 1]
            )
        else:
            if _use_alpha_fe_spec:
                lz = lz + _A2Z * afe
            log_z_c = jnp.clip(lz, ssp_lgmet[0], ssp_lgmet[-1])
            idx = jnp.clip(jnp.searchsorted(ssp_lgmet, log_z_c) - 1, 0, len(ssp_lgmet) - 2)
            frac = (log_z_c - ssp_lgmet[idx]) / (ssp_lgmet[idx + 1] - ssp_lgmet[idx])
            ssp_at_z = (1.0 - frac) * ssp_on_pixels[idx] + frac * ssp_on_pixels[idx + 1]

        # Dust: configurable curves at pixel wavelengths
        _law_kw = dict(n_slope=dn, dust_bump_strength=bump, dust_delta=delta, dust_Rv=rv)
        if _is_single_dust_spec:
            k = law_bc_fn_spec(wave_rest_pixels, **_law_kw)
            trans_1d = f_obs + (1.0 - f_obs) * jnp.exp(-tv * k)
            flux = jnp.einsum("i,ip->p", weights, ssp_at_z) * trans_1d
        elif _dust_exact_spec:
            # Exact: smooth sigmoid — full (n_ages, n_pix) outer product
            k_bc = law_bc_fn_spec(wave_rest_pixels, **_law_kw)
            k_diff = law_diff_fn_spec(wave_rest_pixels, **_law_kw)
            tau = dust_age_w[:, None] * tv1 * k_bc[None, :] + tv2 * k_diff[None, :]
            dust = f_obs + (1.0 - f_obs) * jnp.exp(-tau)
            flux = jnp.einsum("i,ip,ip->p", weights, dust, ssp_at_z)
        else:
            # Fast two-CSP decomposition for spectroscopy
            k_bc = law_bc_fn_spec(wave_rest_pixels, **_law_kw)
            k_diff = law_diff_fn_spec(wave_rest_pixels, **_law_kw)
            trans_bc = jnp.exp(-tv1 * k_bc)
            trans_diff = jnp.exp(-tv2 * k_diff)

            csp_young = jnp.einsum("i,ip->p", weights * young_mask_spec, ssp_at_z)
            csp_old = jnp.einsum("i,ip->p", weights * old_mask_spec, ssp_at_z)

            flux_no_geom = trans_diff * (trans_bc * csp_young + csp_old)
            flux_intrinsic = csp_young + csp_old
            flux = f_obs * flux_intrinsic + (1.0 - f_obs) * flux_no_geom

        # AGN contribution at pixel wavelengths (parametric mode)
        if has_agn:
            agn_lnu = agn_model_fn(
                wave_rest_pixels,
                agn_log_lbol=agn_log_lbol,
                agn_frac=1.0,  # L_bol fully specified by agn_log_lbol
                agn_alpha=agn_alpha,
                agn_T_torus=agn_T_torus,
                agn_tau_torus=agn_tau_torus,
                agn_torus_frac=agn_torus_frac,
                agn_log_mbh=agn_log_mbh,
                agn_log_ledd=agn_log_ledd,
            )
            flux = flux + agn_lnu

        flux = flux_scale * flux * lsun

        if has_sigma_v:
            sv = jnp.asarray(sigma_v, dtype=fdt)
            sigma_pix = (sv / c_km_s) / dlnwave
            kernel_ft = jnp.exp(-2.0 * jnp.pi**2 * sigma_pix**2 * fft_freq**2)
            flux = jnp.fft.irfft(jnp.fft.rfft(flux) * kernel_ft, n=n_pix)

        return flux.astype(jnp.float64)

    if _is_single_dust_spec:

        @jax.jit
        def fused_spec(
            sfr_on_ssp,
            log_z_abs,
            tau_v,
            dust_slope,
            sigma_v=0.0,
            f_obscuration=0.0,
            dust_bump_strength=0.0,
            dust_delta=0.0,
            dust_Rv=3.1,
            alpha_fe=0.0,
            agn_log_lbol=10.0,
            agn_alpha=-1.0,
            agn_T_torus=1000.0,
            agn_tau_torus=5.0,
            agn_torus_frac=0.5,
            agn_log_mbh=7.0,
            agn_log_ledd=-1.0,
        ):
            return _fused_spec_body(
                sfr_on_ssp,
                log_z_abs,
                0.0,
                0.0,
                dust_slope,
                sigma_v,
                f_obscuration,
                dust_bump_strength,
                dust_delta,
                dust_Rv,
                alpha_fe,
                agn_log_lbol,
                agn_alpha,
                agn_T_torus,
                agn_tau_torus,
                agn_torus_frac,
                agn_log_mbh,
                agn_log_ledd,
                tau_v=tau_v,
            )

    else:

        @jax.jit
        def fused_spec(
            sfr_on_ssp,
            log_z_abs,
            tau_bc,
            tau_diff,
            dust_slope,
            sigma_v=0.0,
            f_obscuration=0.0,
            dust_bump_strength=0.0,
            dust_delta=0.0,
            dust_Rv=3.1,
            alpha_fe=0.0,
            agn_log_lbol=10.0,
            agn_alpha=-1.0,
            agn_T_torus=1000.0,
            agn_tau_torus=5.0,
            agn_torus_frac=0.5,
            agn_log_mbh=7.0,
            agn_log_ledd=-1.0,
        ):
            return _fused_spec_body(
                sfr_on_ssp,
                log_z_abs,
                tau_bc,
                tau_diff,
                dust_slope,
                sigma_v,
                f_obscuration,
                dust_bump_strength,
                dust_delta,
                dust_Rv,
                alpha_fe,
                agn_log_lbol,
                agn_alpha,
                agn_T_torus,
                agn_tau_torus,
                agn_torus_frac,
                agn_log_mbh,
                agn_log_ledd,
            )

    return fused_spec


# -------------------------------------------------------------------
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
    model : Model
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
            dust_age_w = model._dust_age_weights.astype(dt)
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


# -------------------------------------------------------------------
# Fused photometry kernel (free redshift, z-table)
# -------------------------------------------------------------------


def build_fused_photometry_ztable(model):
    """Build fused JIT kernel with z-table interpolation.

    Like :func:`build_fused_photometry` but redshift is a free parameter.
    Supports all dust laws via captured law functions.

    Parameters
    ----------
    model : Model
        The model instance providing config and precomputed arrays.

    Returns
    -------
    callable
        JIT-compiled function: (sfr_on_ssp, log_z_abs, tau_bc, tau_diff,
        dust_slope, redshift, ...) -> photometry array.
    """
    from tengri.models.dust.attenuation import get_dust_law
    from tengri.models.sps.dsps_wrapper import (
        _ALPHA_TO_Z_COEFF as _A2Z,
        LSUN_ERG_PER_S,
        has_alpha_grid,
    )

    fdt = model._forward_dtype
    zt = model._ztable
    ssp_phot_table = zt.ssp_phot_table.astype(fdt)
    eff_rest_table = zt.eff_waves_rest_table.astype(fdt)
    flux_scale_table = zt.flux_scale_table.astype(fdt)
    z_grid = zt.z_grid.astype(fdt)
    ssp_lgmet = model.ssp_data.ssp_lgmet.astype(fdt)
    _is_single_dust_zt = model._dust_model == "single_component"
    _dust_exact_zt = getattr(model, "_dust_approx", "fast") == "exact"
    if not _is_single_dust_zt:
        if _dust_exact_zt:
            dust_age_w = model._dust_age_weights.astype(fdt)
        else:
            _t_birth_zt = 1e7
            young_mask_zt = (model.ssp_ages_yr < _t_birth_zt).astype(fdt)
            old_mask_zt = fdt.type(1.0) - young_mask_zt
    _csp_use_matrix_zt = model._csp_integration == "log_interp"
    if _csp_use_matrix_zt:
        _csp_mat_zt = model._csp_matrix.astype(fdt)
    else:
        _age_dt_zt = model._csp_age_dt.astype(fdt)  # precomputed CSP bin widths
    lsun = fdt.type(LSUN_ERG_PER_S)

    _has_alpha_zt = has_alpha_grid(model.ssp_data)
    if _has_alpha_zt:
        ssp_alpha_fe_zt = model.ssp_data.ssp_alpha_fe.astype(fdt)
    _use_alpha_fe_zt = model.spec.alpha_fe_evolving or "met_alpha_fe" in model.spec.free_params

    # IGM: precomputed on z-grid when apply_igm + approx["igm"]
    igm_trans_table = zt.igm_trans_table.astype(fdt)
    has_igm_ztable = bool(model._apply_igm and model._approx.get("igm", True))

    law_bc_fn_zt = get_dust_law(model._dust_law_bc)
    if not _is_single_dust_zt:
        law_diff_fn_zt = get_dust_law(model._dust_law_diff)

    # Metallicity interpolation mode (ztable variant)
    _use_smooth_z_zt = model._met_interp == "smooth"
    _lgmet_scat_zt = fdt.type(model._lgmet_scatter)
    if _use_smooth_z_zt:
        from tengri.models.sps.dsps_wrapper import (
            compute_lgmet_weights as _clw_zt,
        )

    # AGN: capture model function for evaluation at effective wavelengths
    has_agn = model._agn_model is not None and model._agn_parametric
    if has_agn:
        from tengri.models.agn import get_agn_model

        agn_model_fn = get_agn_model(model._agn_model)

    def _fused_zt_body(
        sfr_on_ssp,
        log_z_abs,
        tau_bc,
        tau_diff,
        dust_slope,
        redshift,
        f_obscuration,
        dust_bump_strength,
        dust_delta,
        dust_Rv,
        alpha_fe,
        agn_log_lbol,
        agn_alpha,
        agn_T_torus,
        agn_tau_torus,
        agn_torus_frac,
        agn_log_mbh,
        agn_log_ledd,
        tau_v=0.0,
    ):
        sfr = sfr_on_ssp.astype(fdt)
        lz = jnp.asarray(log_z_abs, dtype=fdt)
        tv1 = jnp.asarray(tau_bc, dtype=fdt)
        tv2 = jnp.asarray(tau_diff, dtype=fdt)
        tv = jnp.asarray(tau_v, dtype=fdt)
        dn = jnp.asarray(dust_slope, dtype=fdt)
        z = jnp.asarray(redshift, dtype=fdt)
        f_obs = jnp.asarray(f_obscuration, dtype=fdt)
        bump = jnp.asarray(dust_bump_strength, dtype=fdt)
        delta = jnp.asarray(dust_delta, dtype=fdt)
        rv = jnp.asarray(dust_Rv, dtype=fdt)
        afe = jnp.asarray(alpha_fe, dtype=fdt)

        # Interpolate z-table to current redshift
        z_c = jnp.clip(z, z_grid[0], z_grid[-1])
        zi = jnp.clip(jnp.searchsorted(z_grid, z_c) - 1, 0, len(z_grid) - 2)
        zf = (z_c - z_grid[zi]) / (z_grid[zi + 1] - z_grid[zi])

        ssp_phot = (1.0 - zf) * ssp_phot_table[zi] + zf * ssp_phot_table[zi + 1]
        eff_rest = (1.0 - zf) * eff_rest_table[zi] + zf * eff_rest_table[zi + 1]
        flux_scale = (1.0 - zf) * flux_scale_table[zi] + zf * flux_scale_table[zi + 1]

        # CSP weights (precomputed bin widths; method set at model init)
        weights = _csp_mat_zt @ sfr if _csp_use_matrix_zt else sfr * _age_dt_zt

        # Metallicity + alpha interpolation
        if _has_alpha_zt:
            lz_c = jnp.clip(lz, ssp_lgmet[0], ssp_lgmet[-1])
            iz = jnp.clip(jnp.searchsorted(ssp_lgmet, lz_c) - 1, 0, len(ssp_lgmet) - 2)
            fz = (lz_c - ssp_lgmet[iz]) / (ssp_lgmet[iz + 1] - ssp_lgmet[iz])
            afe_c = jnp.clip(afe, ssp_alpha_fe_zt[0], ssp_alpha_fe_zt[-1])
            n_afe = len(ssp_alpha_fe_zt)
            ia = jnp.clip(jnp.searchsorted(ssp_alpha_fe_zt, afe_c) - 1, 0, n_afe - 2)
            fa = (afe_c - ssp_alpha_fe_zt[ia]) / (ssp_alpha_fe_zt[ia + 1] - ssp_alpha_fe_zt[ia])
            ssp_at_z = (
                (1 - fz) * (1 - fa) * ssp_phot[iz, ia]
                + fz * (1 - fa) * ssp_phot[iz + 1, ia]
                + (1 - fz) * fa * ssp_phot[iz, ia + 1]
                + fz * fa * ssp_phot[iz + 1, ia + 1]
            )
        else:
            if _use_alpha_fe_zt:
                lz = lz + _A2Z * afe
            if _use_smooth_z_zt:
                zw = _clw_zt(lz, ssp_lgmet, _lgmet_scat_zt)
                ssp_at_z = jnp.einsum("m,maf->af", zw, ssp_phot)
            else:
                log_z_c = jnp.clip(lz, ssp_lgmet[0], ssp_lgmet[-1])
                idx = jnp.clip(
                    jnp.searchsorted(ssp_lgmet, log_z_c) - 1,
                    0,
                    len(ssp_lgmet) - 2,
                )
                frac = (log_z_c - ssp_lgmet[idx]) / (ssp_lgmet[idx + 1] - ssp_lgmet[idx])
                ssp_at_z = (1.0 - frac) * ssp_phot[idx] + frac * ssp_phot[idx + 1]

        # Dust: configurable curves at effective wavelengths
        _law_kw = dict(n_slope=dn, dust_bump_strength=bump, dust_delta=delta, dust_Rv=rv)
        if _is_single_dust_zt:
            k = law_bc_fn_zt(eff_rest, **_law_kw)
            trans_1d = f_obs + (1.0 - f_obs) * jnp.exp(-tv * k)
            flux_lsun = jnp.einsum("i,if->f", weights, ssp_at_z) * trans_1d
        elif _dust_exact_zt:
            # Exact: smooth sigmoid — full (n_ages, n_filt) outer product
            k_bc = law_bc_fn_zt(eff_rest, **_law_kw)
            k_diff = law_diff_fn_zt(eff_rest, **_law_kw)
            tau = dust_age_w[:, None] * tv1 * k_bc[None, :] + tv2 * k_diff[None, :]
            dust = f_obs + (1.0 - f_obs) * jnp.exp(-tau)
            flux_lsun = jnp.einsum("i,if,if->f", weights, dust, ssp_at_z)
        else:
            # Fast two-CSP decomposition
            k_bc = law_bc_fn_zt(eff_rest, **_law_kw)
            k_diff = law_diff_fn_zt(eff_rest, **_law_kw)
            trans_bc = jnp.exp(-tv1 * k_bc)
            trans_diff = jnp.exp(-tv2 * k_diff)

            csp_young = jnp.einsum("i,if->f", weights * young_mask_zt, ssp_at_z)
            csp_old = jnp.einsum("i,if->f", weights * old_mask_zt, ssp_at_z)

            flux_no_geom = trans_diff * (trans_bc * csp_young + csp_old)
            flux_intr = csp_young + csp_old
            flux_lsun = f_obs * flux_intr + (1.0 - f_obs) * flux_no_geom

        # AGN contribution at effective wavelengths (parametric mode)
        if has_agn:
            agn_lnu = agn_model_fn(
                eff_rest,
                agn_log_lbol=agn_log_lbol,
                agn_frac=1.0,  # L_bol fully specified by agn_log_lbol
                agn_alpha=agn_alpha,
                agn_T_torus=agn_T_torus,
                agn_tau_torus=agn_tau_torus,
                agn_torus_frac=agn_torus_frac,
                agn_log_mbh=agn_log_mbh,
                agn_log_ledd=agn_log_ledd,
            )
            flux_lsun = flux_lsun + agn_lnu

        # IGM absorption (interpolated from precomputed z-table)
        if has_igm_ztable:
            igm_trans = (1.0 - zf) * igm_trans_table[zi] + zf * igm_trans_table[zi + 1]
            flux_lsun = flux_lsun * igm_trans

        return (flux_scale * flux_lsun * lsun).astype(jnp.float64)

    if _is_single_dust_zt:

        @jax.jit
        def fused_phot_ztable(
            sfr_on_ssp,
            log_z_abs,
            tau_v,
            dust_slope,
            redshift,
            f_obscuration=0.0,
            dust_bump_strength=0.0,
            dust_delta=0.0,
            dust_Rv=3.1,
            alpha_fe=0.0,
            agn_log_lbol=10.0,
            agn_alpha=-1.0,
            agn_T_torus=1000.0,
            agn_tau_torus=5.0,
            agn_torus_frac=0.5,
            agn_log_mbh=7.0,
            agn_log_ledd=-1.0,
        ):
            return _fused_zt_body(
                sfr_on_ssp,
                log_z_abs,
                0.0,
                0.0,
                dust_slope,
                redshift,
                f_obscuration,
                dust_bump_strength,
                dust_delta,
                dust_Rv,
                alpha_fe,
                agn_log_lbol,
                agn_alpha,
                agn_T_torus,
                agn_tau_torus,
                agn_torus_frac,
                agn_log_mbh,
                agn_log_ledd,
                tau_v=tau_v,
            )

    else:

        @jax.jit
        def fused_phot_ztable(
            sfr_on_ssp,
            log_z_abs,
            tau_bc,
            tau_diff,
            dust_slope,
            redshift,
            f_obscuration=0.0,
            dust_bump_strength=0.0,
            dust_delta=0.0,
            dust_Rv=3.1,
            alpha_fe=0.0,
            agn_log_lbol=10.0,
            agn_alpha=-1.0,
            agn_T_torus=1000.0,
            agn_tau_torus=5.0,
            agn_torus_frac=0.5,
            agn_log_mbh=7.0,
            agn_log_ledd=-1.0,
        ):
            return _fused_zt_body(
                sfr_on_ssp,
                log_z_abs,
                tau_bc,
                tau_diff,
                dust_slope,
                redshift,
                f_obscuration,
                dust_bump_strength,
                dust_delta,
                dust_Rv,
                alpha_fe,
                agn_log_lbol,
                agn_alpha,
                agn_T_torus,
                agn_tau_torus,
                agn_torus_frac,
                agn_log_mbh,
                agn_log_ledd,
            )

    return fused_phot_ztable


# -------------------------------------------------------------------
# Compositional rest-frame SED kernel (Tier 2)
# -------------------------------------------------------------------


def is_tier2_compatible(model):
    """Check if the compositional rest-frame SED kernel can be built.

    Tier 2 supports ALL physics components (unlike Tier 1 fused kernels).
    It only falls back when the model uses features that require non-standard
    SFH/metallicity paths (tabulated SFH, DSPS table, evolving Z).

    Parameters
    ----------
    model : Model
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
    model : Model
        The model instance providing config and precomputed arrays.

    Returns
    -------
    callable
        JIT-compiled function:
        ``(weights, ssp_flux_at_z, p_dict) -> rest_sed``
        where ``p_dict`` contains internal dust/AGN/nebular/radio/X-ray
        parameters.
    """
    from tengri.models.dust.attenuation import get_dust_law
    from tengri.models.sps.dsps_wrapper import LSUN_ERG_PER_S

    dt = model._forward_dtype
    ssp_wave = model.ssp_data.ssp_wave.astype(dt)
    _is_single_dust = model._dust_model == "single_component"
    _dust_exact = getattr(model, "_dust_approx", "fast") == "exact"
    if not _is_single_dust:
        if _dust_exact:
            dust_age_w = model._dust_age_weights.astype(dt)
        else:
            _t_birth = 1e7  # 10 Myr — Charlot & Fall (2000)
            young_mask = (model.ssp_ages_yr < _t_birth).astype(dt)
            old_mask = dt.type(1.0) - young_mask
    lsun = dt.type(LSUN_ERG_PER_S)

    # Capture dust law functions (pure JAX, JIT-traceable)
    law_bc_fn = get_dust_law(model._dust_law_bc)
    if not _is_single_dust:
        law_diff_fn = get_dust_law(model._dust_law_diff)
        same_law = model._dust_law_bc == model._dust_law_diff

    # --- Optional components (Python if at trace time) ---
    # Nebular
    has_nebular = model._nebular_backend is not None and getattr(
        model._nebular_backend, "has_free_params", False
    )
    # Full-precision wavelength array for components that need float64
    ssp_wave_f64 = model.ssp_data.ssp_wave

    if has_nebular:
        nebular_backend = model._nebular_backend
        ssp_log_ages_yr = model.ssp_log_ages_yr

    # Shock emission
    has_shock = getattr(model, "_shock_enabled", False)
    if has_shock:
        from tengri.models.nebular.shock import shock_emission_sed

    # Dust emission
    has_dust_em = model._dust_emission_model is not None
    if has_dust_em:
        from tengri.models.dust.emission import get_emission_model

        dust_emission_fn = get_emission_model(model._dust_emission_model)

    # AGN
    has_agn = model._agn_model is not None
    agn_parametric = model._agn_parametric if has_agn else False
    if has_agn:
        from tengri.models.agn import get_agn_model

        agn_model_fn = get_agn_model(model._agn_model)

    # Radio
    has_radio = model._radio_enabled
    if has_radio:
        from tengri.models.radio import radio_total

    # X-ray
    has_xray = model._xray_enabled
    if has_xray:
        from tengri.models.xray import xray_total

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
        if has_nebular:
            neb_sed = nebular_backend.predict_nebular_sed(
                ssp_weights=weights,
                ssp_wave=ssp_wave_f64,
                ssp_log_ages_yr=ssp_log_ages_yr,
                log_z=p["log_z_abs"],
                neb_logU=p.get("neb_logU", -3.0),
                neb_logZ_gas=p.get("neb_logZ_gas", None),
                neb_fesc=p.get("neb_fesc", 0.0),
                neb_fesc_lya=p.get("neb_fesc_lya", 0.0),
            )
            sed = sed + neb_sed

        # --- 3. Shock emission ---
        if has_shock:
            shock_frac = p.get("shock_frac", 0.0)
            shock_velocity = p.get("shock_velocity", 300.0)
            shock_log_density = p.get("shock_log_density", 0.0)
            nu_shock = _c_aa / ssp_wave.astype(jnp.float64)
            l_bol = -jnp.trapezoid(sed, nu_shock)
            l_halpha_approx = jnp.maximum(l_bol * 1e-3, 1e-30)
            l_shock_halpha = shock_frac * l_halpha_approx
            shock_sed = shock_emission_sed(
                ssp_wave_f64,
                shock_velocity,
                l_shock_halpha,
                shock_log_density=shock_log_density,
            )
            sed = sed + shock_sed

        # --- 4. Dust IR emission (energy-balanced) ---
        if has_dust_em:
            nu_em = _c_aa / ssp_wave.astype(jnp.float64)
            L_absorbed = -jnp.trapezoid(sed_intr - sed_atten, nu_em)
            eta_balance = p.get("dust_eta_balance", 1.0)
            L_ir = jnp.maximum(L_absorbed * eta_balance, 0.0)
            dust_ir = dust_emission_fn(
                ssp_wave_f64,
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
        else:
            L_ir = jnp.float64(0.0)

        # --- 5. AGN ---
        agn_bol_erg = jnp.float64(0.0)
        if has_agn:
            if agn_parametric:
                agn_log_lbol = p.get("agn_log_lbol", 10.0)
                agn_frac_val = 1.0
                agn_bol_erg = 10.0**agn_log_lbol
            else:
                agn_frac_val = p.get("agn_frac", 0.0)
                nu_agn = _c_aa / ssp_wave.astype(jnp.float64)
                L_bol_stellar = -jnp.trapezoid(sed, nu_agn)
                agn_log_lbol = jnp.log10(jnp.maximum(L_bol_stellar * agn_frac_val, 1e-50))
                agn_bol_erg = L_bol_stellar * agn_frac_val
            agn_sed = agn_model_fn(
                ssp_wave_f64,
                agn_log_lbol=agn_log_lbol,
                agn_frac=agn_frac_val,
                agn_alpha=p.get("agn_alpha", -1.0),
                agn_T_torus=p.get("agn_T_torus", 1000.0),
                agn_tau_torus=p.get("agn_tau_torus", 5.0),
                agn_torus_frac=p.get("agn_torus_frac", 0.5),
                agn_log_mbh=p.get("agn_log_mbh", 7.0),
                agn_log_ledd=p.get("agn_log_ledd", -1.0),
            )
            sed = sed + agn_sed

        # --- 6. Radio ---
        if has_radio:
            radio_sed = radio_total(
                ssp_wave_f64,
                L_ir=L_ir,
                L_agn_bol=agn_bol_erg,
                q_ir=p.get("radio_q_ir", 2.64),
                alpha_sf=p.get("radio_alpha_sf", 0.8),
                radio_loudness=p.get("radio_loudness", 0.0),
                alpha_agn=p.get("radio_alpha_agn", 0.7),
            )
            sed = sed + radio_sed

        # --- 7. X-ray ---
        if has_xray:
            # SFR: use last weight as proxy for current SFR
            sfr_current = p.get("_sfr_current", 1.0)
            mstar = jnp.sum(weights)
            xray_sed = xray_total(
                ssp_wave_f64,
                sfr=sfr_current,
                stellar_mass=mstar,
                L_agn_bol=agn_bol_erg,
                gamma_agn=p.get("xray_gamma_agn", 1.8),
                alpha_ox=p.get("xray_alpha_ox", -1.4),
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
    from tengri.models.observation.spectroscopy import compute_spectrum

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
    model : Model
        Fully initialized Model with filters and fixed redshift.

    Returns
    -------
    callable or None
        JIT-compiled function: ``params_dict -> photometry_array``.
        Returns None if prerequisites are not met (no filters, no
        fixed z, no Tier 2 kernel).
    """
    if model._fused_rest_sed is None:
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
    rest_sed_kernel = model._fused_rest_sed
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
        _t_obs_gyr_fixed = float(_age_at_z_fn(z_fixed) / 1e9)

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
            t_obs_gyr = _t_obs_gyr_fixed if _t_obs_gyr_fixed is not None else _age_at_z_fn(z) / 1e9
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
    model : Model
        Fully initialized Model with spectroscopy config.

    Returns
    -------
    callable or None
        JIT-compiled function: ``params_dict -> spectrum_array``.
    """
    if model._fused_rest_sed is None:
        return None

    from tengri.core.param_translate import get_internal_params
    from tengri.core.sed_pipeline import interp_met_alpha_dispatch, interp_metallicity
    from tengri.models.observation.spectroscopy import compute_spectrum
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
    if model._spec_precomp is not None:
        wave_obs = model._spec_precomp.wave_obs_pixels
    elif hasattr(model, "_wave_obs"):
        wave_obs = model._wave_obs
    if wave_obs is None:
        return None

    # Capture model state
    rest_sed_kernel = model._fused_rest_sed
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
        _t_obs_gyr_fixed_spec = float(_age_at_z_spec(z_fixed) / 1e9)

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
                _t_obs_gyr_fixed_spec
                if _t_obs_gyr_fixed_spec is not None
                else _age_at_z_spec(z) / 1e9
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
