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
    dust_age_w = model._dust_age_weights.astype(dt)
    flux_scale = dt.type(precomp.flux_scale)
    ssp_ages_yr = model.ssp_ages_yr.astype(dt)
    lsun = dt.type(LSUN_ERG_PER_S)

    # Capture dust law functions (pure JAX, JIT-traceable)
    law_bc_fn = get_dust_law(model._dust_law_bc)
    law_diff_fn = get_dust_law(model._dust_law_diff)

    from tengri.models.sps.dsps_wrapper import (
        _ALPHA_TO_Z_COEFF as _A2Z,
        has_alpha_grid,
    )

    # Alpha-enhanced SSP grid detection (4D vs 3D)
    _has_alpha = has_alpha_grid(model.ssp_data)
    if _has_alpha:
        ssp_alpha_fe = model.ssp_data.ssp_alpha_fe.astype(dt)

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
        sfr = sfr_on_ssp.astype(dt)
        lz = jnp.asarray(log_z_abs, dtype=dt)
        tv1 = jnp.asarray(tau_bc, dtype=dt)
        tv2 = jnp.asarray(tau_diff, dtype=dt)
        dn = jnp.asarray(dust_slope, dtype=dt)
        f_obs = jnp.asarray(f_obscuration, dtype=dt)
        bump = jnp.asarray(dust_bump_strength, dtype=dt)
        delta = jnp.asarray(dust_delta, dtype=dt)
        rv = jnp.asarray(dust_Rv, dtype=dt)
        afe = jnp.asarray(alpha_fe, dtype=dt)

        # CSP weights
        age_dt = jnp.concatenate(
            [
                jnp.array([ssp_ages_yr[1] - ssp_ages_yr[0]]),
                0.5 * (ssp_ages_yr[2:] - ssp_ages_yr[:-2]),
                jnp.array([ssp_ages_yr[-1] - ssp_ages_yr[-2]]),
            ]
        )
        weights = sfr * age_dt

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
            # 3D: effective_metallicity fallback
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
        k_bc = law_bc_fn(
            eff_waves_rest,
            n_slope=dn,
            dust_bump_strength=bump,
            dust_delta=delta,
            dust_Rv=rv,
        )
        k_diff = law_diff_fn(
            eff_waves_rest,
            n_slope=dn,
            dust_bump_strength=bump,
            dust_delta=delta,
            dust_Rv=rv,
        )
        tau = dust_age_w[:, None] * tv1 * k_bc[None, :] + tv2 * k_diff[None, :]
        dust = f_obs + (1.0 - f_obs) * jnp.exp(-tau)

        # Attenuated stellar flux (Lsun)
        flux_attenuated = jnp.einsum("i,if,if->f", weights, dust, ssp_at_z)

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
    wave_obs_pixels = precomp.wave_obs_pixels.astype(fdt)
    wave_rest_pixels = precomp.wave_rest_pixels.astype(fdt)
    dust_age_w = model._dust_age_weights.astype(fdt)
    flux_scale = fdt.type(precomp.flux_scale)
    ssp_ages_yr = model.ssp_ages_yr.astype(fdt)
    lsun = fdt.type(LSUN_ERG_PER_S)
    n_pix = len(wave_obs_pixels)
    has_sigma_v = model._has_sigma_v

    # Capture dust law functions
    law_bc_fn = get_dust_law(model._dust_law_bc)
    law_diff_fn = get_dust_law(model._dust_law_diff)

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
        sfr = sfr_on_ssp.astype(fdt)
        lz = jnp.asarray(log_z_abs, dtype=fdt)
        tv1 = jnp.asarray(tau_bc, dtype=fdt)
        tv2 = jnp.asarray(tau_diff, dtype=fdt)
        dn = jnp.asarray(dust_slope, dtype=fdt)
        f_obs = jnp.asarray(f_obscuration, dtype=fdt)
        bump = jnp.asarray(dust_bump_strength, dtype=fdt)
        delta = jnp.asarray(dust_delta, dtype=fdt)
        rv = jnp.asarray(dust_Rv, dtype=fdt)
        afe = jnp.asarray(alpha_fe, dtype=fdt)

        # CSP weights
        age_dt = jnp.concatenate(
            [
                jnp.array([ssp_ages_yr[1] - ssp_ages_yr[0]]),
                0.5 * (ssp_ages_yr[2:] - ssp_ages_yr[:-2]),
                jnp.array([ssp_ages_yr[-1] - ssp_ages_yr[-2]]),
            ]
        )
        weights = sfr * age_dt

        # Metallicity + alpha interpolation
        if _has_alpha_zt:
            lz_c = jnp.clip(lz, ssp_lgmet[0], ssp_lgmet[-1])
            iz = jnp.clip(jnp.searchsorted(ssp_lgmet, lz_c) - 1, 0, len(ssp_lgmet) - 2)
            fz = (lz_c - ssp_lgmet[iz]) / (ssp_lgmet[iz + 1] - ssp_lgmet[iz])
            afe_c = jnp.clip(afe, ssp_alpha_fe_zt[0], ssp_alpha_fe_zt[-1])
            n_afe = len(ssp_alpha_fe_zt)
            ia = jnp.clip(
                jnp.searchsorted(ssp_alpha_fe_zt, afe_c) - 1, 0, n_afe - 2
            )
            fa = (afe_c - ssp_alpha_fe_zt[ia]) / (ssp_alpha_fe_zt[ia + 1] - ssp_alpha_fe_zt[ia])
            ssp_at_z = (
                (1 - fz) * (1 - fa) * ssp_on_pixels[iz, ia]
                + fz * (1 - fa) * ssp_on_pixels[iz + 1, ia]
                + (1 - fz) * fa * ssp_on_pixels[iz, ia + 1]
                + fz * fa * ssp_on_pixels[iz + 1, ia + 1]
            )
        else:
            lz = lz + _A2Z * afe
            log_z_c = jnp.clip(lz, ssp_lgmet[0], ssp_lgmet[-1])
            idx = jnp.clip(jnp.searchsorted(ssp_lgmet, log_z_c) - 1, 0, len(ssp_lgmet) - 2)
            frac = (log_z_c - ssp_lgmet[idx]) / (ssp_lgmet[idx + 1] - ssp_lgmet[idx])
            ssp_at_z = (1.0 - frac) * ssp_on_pixels[idx] + frac * ssp_on_pixels[idx + 1]

        # Dust: configurable curves at pixel wavelengths
        k_bc = law_bc_fn(
            wave_rest_pixels,
            n_slope=dn,
            dust_bump_strength=bump,
            dust_delta=delta,
            dust_Rv=rv,
        )
        k_diff = law_diff_fn(
            wave_rest_pixels,
            n_slope=dn,
            dust_bump_strength=bump,
            dust_delta=delta,
            dust_Rv=rv,
        )
        tau = dust_age_w[:, None] * tv1 * k_bc[None, :] + tv2 * k_diff[None, :]
        dust = f_obs + (1.0 - f_obs) * jnp.exp(-tau)

        # Weighted sum
        flux = jnp.einsum("i,ip,ip->p", weights, dust, ssp_at_z)

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
    dust_age_w = model._dust_age_weights.astype(dt)
    lsun = dt.type(LSUN_ERG_PER_S)

    law_bc_fn = model._dust_law_bc_fn
    law_diff_fn = model._dust_law_diff_fn
    same_law = model._dust_law_bc == model._dust_law_diff

    @jax.jit
    def exact_sed(
        weights,
        ssp_at_z,
        tau_bc,
        tau_diff,
        n_slope=-0.7,
        dust_bump_strength=0.0,
        dust_delta=0.0,
        dust_Rv=3.1,
        f_obscuration=0.0,
    ):
        w = weights.astype(dt)
        ssp_z = ssp_at_z.astype(dt)

        # Dust curves -- skip duplicate when bc == diff
        k_bc = law_bc_fn(
            ssp_wave,
            n_slope=n_slope,
            dust_bump_strength=dust_bump_strength,
            dust_delta=dust_delta,
            dust_Rv=dust_Rv,
        )
        k_diff = (
            k_bc
            if same_law
            else law_diff_fn(
                ssp_wave,
                n_slope=n_slope,
                dust_bump_strength=dust_bump_strength,
                dust_delta=dust_delta,
                dust_Rv=dust_Rv,
            )
        )

        # Dust + CSP SED: XLA fuses broadcast + exp + einsum
        tau = dust_age_w[:, None] * tau_bc * k_bc[None, :] + tau_diff * k_diff[None, :]
        dust_trans = f_obscuration + (1.0 - f_obscuration) * jnp.exp(-tau)

        sed_atten = (lsun * jnp.einsum("i,iw,iw->w", w, ssp_z, dust_trans)).astype(jnp.float64)
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
    dust_age_w = model._dust_age_weights.astype(fdt)
    ssp_ages_yr = model.ssp_ages_yr.astype(fdt)
    lsun = fdt.type(LSUN_ERG_PER_S)

    _has_alpha_zt = has_alpha_grid(model.ssp_data)
    if _has_alpha_zt:
        ssp_alpha_fe_zt = model.ssp_data.ssp_alpha_fe.astype(fdt)

    # IGM: precomputed on z-grid when apply_igm + approx["igm"]
    igm_trans_table = zt.igm_trans_table.astype(fdt)
    has_igm_ztable = bool(model._apply_igm and model._approx.get("igm", True))

    law_bc_fn = get_dust_law(model._dust_law_bc)
    law_diff_fn = get_dust_law(model._dust_law_diff)

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
        sfr = sfr_on_ssp.astype(fdt)
        lz = jnp.asarray(log_z_abs, dtype=fdt)
        tv1 = jnp.asarray(tau_bc, dtype=fdt)
        tv2 = jnp.asarray(tau_diff, dtype=fdt)
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

        # CSP weights
        age_dt = jnp.concatenate(
            [
                jnp.array([ssp_ages_yr[1] - ssp_ages_yr[0]]),
                0.5 * (ssp_ages_yr[2:] - ssp_ages_yr[:-2]),
                jnp.array([ssp_ages_yr[-1] - ssp_ages_yr[-2]]),
            ]
        )
        weights = sfr * age_dt

        # Metallicity + alpha interpolation
        if _has_alpha_zt:
            lz_c = jnp.clip(lz, ssp_lgmet[0], ssp_lgmet[-1])
            iz = jnp.clip(jnp.searchsorted(ssp_lgmet, lz_c) - 1, 0, len(ssp_lgmet) - 2)
            fz = (lz_c - ssp_lgmet[iz]) / (ssp_lgmet[iz + 1] - ssp_lgmet[iz])
            afe_c = jnp.clip(afe, ssp_alpha_fe_zt[0], ssp_alpha_fe_zt[-1])
            n_afe = len(ssp_alpha_fe_zt)
            ia = jnp.clip(
                jnp.searchsorted(ssp_alpha_fe_zt, afe_c) - 1, 0, n_afe - 2
            )
            fa = (afe_c - ssp_alpha_fe_zt[ia]) / (ssp_alpha_fe_zt[ia + 1] - ssp_alpha_fe_zt[ia])
            ssp_at_z = (
                (1 - fz) * (1 - fa) * ssp_phot[iz, ia]
                + fz * (1 - fa) * ssp_phot[iz + 1, ia]
                + (1 - fz) * fa * ssp_phot[iz, ia + 1]
                + fz * fa * ssp_phot[iz + 1, ia + 1]
            )
        else:
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
        k_bc = law_bc_fn(
            eff_rest,
            n_slope=dn,
            dust_bump_strength=bump,
            dust_delta=delta,
            dust_Rv=rv,
        )
        k_diff = law_diff_fn(
            eff_rest,
            n_slope=dn,
            dust_bump_strength=bump,
            dust_delta=delta,
            dust_Rv=rv,
        )
        tau = dust_age_w[:, None] * tv1 * k_bc[None, :] + tv2 * k_diff[None, :]
        dust = f_obs + (1.0 - f_obs) * jnp.exp(-tau)

        # Weighted sum
        flux_lsun = jnp.einsum("i,if,if->f", weights, dust, ssp_at_z)

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

    return fused_phot_ztable
