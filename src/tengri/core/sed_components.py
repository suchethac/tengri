"""Compositional rest-frame SED component builders.

Each builder function takes a ``model`` instance, captures static config
(SSP grids, dust laws, enabled flags) in a closure, and returns a pure
JAX function that operates on per-call parameters only.

The returned functions are JIT-compatible and compose into the
:func:`build_fused_rest_sed` kernel in ``fused_kernels.py`` (Tier 2).

All functions produce rest-frame luminosity SEDs in erg/s/Hz on the
full SSP wavelength grid.
"""

from __future__ import annotations

import jax.numpy as jnp

# -------------------------------------------------------------------
# SSP component: metallicity interpolation + CSP weighted sum
# -------------------------------------------------------------------


def build_ssp_component(model):
    """Build SSP flux + CSP weight computation.

    Captures SSP grid, metallicity mode, and dust age weights.

    Parameters
    ----------
    model : Model
        The model instance providing config and precomputed arrays.

    Returns
    -------
    callable
        ``(sfr_on_ssp, log_z_abs, alpha_fe) -> (ssp_flux_at_z, weights)``

        - ``ssp_flux_at_z``: shape (n_age, n_wave), Z-interpolated SSP.
        - ``weights``: shape (n_age,), CSP mass weights (Msun).
    """
    from tengri.models.sps.dsps_wrapper import (
        _ALPHA_TO_Z_COEFF as _A2Z,
        compute_lgmet_weights,
    )

    dt = model._forward_dtype
    ssp_flux = model.ssp_data.ssp_flux.astype(dt)
    ssp_lgmet = model.ssp_data.ssp_lgmet.astype(dt)
    ssp_ages_yr = model.ssp_ages_yr.astype(dt)
    use_smooth = model._met_interp == "smooth"
    lgmet_scatter = dt.type(model._lgmet_scatter)

    def ssp_fn(sfr_on_ssp, log_z_abs, alpha_fe=0.0):
        sfr = sfr_on_ssp.astype(dt)
        lz = jnp.asarray(log_z_abs, dtype=dt)
        afe = jnp.asarray(alpha_fe, dtype=dt)

        # CSP weights (trapezoidal rule, consistent with fused kernel)
        age_dt = jnp.concatenate(
            [
                jnp.array([0.5 * (ssp_ages_yr[1] - ssp_ages_yr[0])]),
                0.5 * (ssp_ages_yr[2:] - ssp_ages_yr[:-2]),
                jnp.array([0.5 * (ssp_ages_yr[-1] - ssp_ages_yr[-2])]),
            ]
        )
        weights = sfr * age_dt

        # Effective metallicity with alpha-element shift
        lz_eff = lz + _A2Z * afe

        # Metallicity interpolation
        if use_smooth:
            zw = compute_lgmet_weights(lz_eff, ssp_lgmet, lgmet_scatter)
            ssp_at_z = jnp.einsum("m,maw->aw", zw, ssp_flux)
        else:
            log_z_c = jnp.clip(lz_eff, ssp_lgmet[0], ssp_lgmet[-1])
            idx = jnp.clip(
                jnp.searchsorted(ssp_lgmet, log_z_c) - 1,
                0,
                len(ssp_lgmet) - 2,
            )
            frac = (log_z_c - ssp_lgmet[idx]) / (ssp_lgmet[idx + 1] - ssp_lgmet[idx])
            ssp_at_z = (1.0 - frac) * ssp_flux[idx] + frac * ssp_flux[idx + 1]

        return ssp_at_z, weights

    return ssp_fn


# -------------------------------------------------------------------
# Dust attenuation: two-component model
# -------------------------------------------------------------------


def build_dust_atten_component(model):
    """Build dust attenuation applied to age-resolved SSP fluxes.

    Captures dust law functions and precomputed age weights.

    Parameters
    ----------
    model : Model
        The model instance.

    Returns
    -------
    callable
        ``(ssp_flux_at_z, weights, tau_bc, tau_diff, dust_slope, ...) ->
        (sed_attenuated, sed_intrinsic, L_absorbed)``

        All outputs in erg/s/Hz. ``L_absorbed`` is in erg/s (integrated).
    """
    from tengri.models.sps.dsps_wrapper import LSUN_ERG_PER_S

    dt = model._forward_dtype
    ssp_wave = model.ssp_data.ssp_wave.astype(dt)
    dust_age_w = model._dust_age_weights.astype(dt)
    lsun = dt.type(LSUN_ERG_PER_S)
    law_bc_fn = model._dust_law_bc_fn
    law_diff_fn = model._dust_law_diff_fn
    same_law = model._dust_law_bc == model._dust_law_diff
    _c_aa = dt.type(2.99792458e18)

    def dust_fn(
        ssp_flux_at_z,
        weights,
        tau_bc,
        tau_diff,
        dust_slope=-0.7,
        f_obscuration=0.0,
        dust_bump_strength=0.0,
        dust_delta=0.0,
        dust_Rv=3.1,
    ):
        w = weights.astype(dt)
        ssp_z = ssp_flux_at_z.astype(dt)

        # Dust curves
        k_bc = law_bc_fn(
            ssp_wave,
            n_slope=dust_slope,
            dust_bump_strength=dust_bump_strength,
            dust_delta=dust_delta,
            dust_Rv=dust_Rv,
        )
        k_diff = (
            k_bc
            if same_law
            else law_diff_fn(
                ssp_wave,
                n_slope=dust_slope,
                dust_bump_strength=dust_bump_strength,
                dust_delta=dust_delta,
                dust_Rv=dust_Rv,
            )
        )

        tau = dust_age_w[:, None] * tau_bc * k_bc[None, :] + tau_diff * k_diff[None, :]
        dust_trans = f_obscuration + (1.0 - f_obscuration) * jnp.exp(-tau)

        sed_atten = lsun * jnp.einsum("i,iw,iw->w", w, ssp_z, dust_trans)
        sed_intr = lsun * jnp.einsum("i,iw->w", w, ssp_z)

        # Absorbed luminosity (erg/s): integrate (L_intr - L_atten) over nu
        nu = _c_aa / ssp_wave
        L_absorbed = -jnp.trapezoid(sed_intr - sed_atten, nu)
        L_absorbed = jnp.maximum(L_absorbed, dt.type(0.0))

        return (
            sed_atten.astype(jnp.float64),
            sed_intr.astype(jnp.float64),
            L_absorbed.astype(jnp.float64),
        )

    return dust_fn


# -------------------------------------------------------------------
# Dust IR emission (energy-balanced)
# -------------------------------------------------------------------


def build_dust_emission_component(model):
    """Build dust IR emission from absorbed luminosity.

    Captures the emission model name and dispatches at trace time.

    Parameters
    ----------
    model : Model
        The model instance.

    Returns
    -------
    callable or None
        ``(wave, L_ir, dust_params) -> dust_ir_sed`` in erg/s/Hz,
        or None if dust emission is disabled.
    """
    if model._dust_emission_model is None:
        return None

    from tengri.models.dust.emission import get_emission_model

    emission_fn = get_emission_model(model._dust_emission_model)
    wave = model.ssp_data.ssp_wave

    def dust_em_fn(
        L_ir,
        dust_T=35.0,
        dust_beta_ir=1.6,
        dust_alpha_mir=2.0,
        dust_alpha_dale=2.0,
        dust_umin=1.0,
        dust_gamma_dl=0.01,
        dust_qpah=2.5,
        dust_eta_balance=1.0,
    ):
        L_ir_scaled = jnp.maximum(L_ir * dust_eta_balance, 0.0)
        return emission_fn(
            wave,
            L_ir_scaled,
            dust_T=dust_T,
            dust_beta_ir=dust_beta_ir,
            dust_alpha_mir=dust_alpha_mir,
            dust_alpha_dale=dust_alpha_dale,
            dust_umin=dust_umin,
            dust_gamma_dl=dust_gamma_dl,
            dust_qpah=dust_qpah,
        )

    return dust_em_fn


# -------------------------------------------------------------------
# AGN contribution
# -------------------------------------------------------------------


def build_agn_component(model):
    """Build AGN SED contribution (parametric mode only).

    Parameters
    ----------
    model : Model
        The model instance.

    Returns
    -------
    callable or None
        ``(agn_params) -> agn_sed`` in erg/s/Hz (Lsun/Hz units),
        or None if AGN is disabled.
    """
    if model._agn_model is None:
        return None

    from tengri.models.agn import get_agn_model

    agn_model_fn = get_agn_model(model._agn_model)
    wave = model.ssp_data.ssp_wave
    is_parametric = model._agn_parametric

    def agn_fn(
        sed_so_far=None,
        agn_log_lbol=10.0,
        agn_frac=0.0,
        agn_alpha=-1.0,
        agn_T_torus=1000.0,
        agn_tau_torus=5.0,
        agn_torus_frac=0.5,
        agn_log_mbh=7.0,
        agn_log_ledd=-1.0,
    ):
        if is_parametric:
            frac_for_model = 1.0
            bol_erg = 10.0**agn_log_lbol
        else:
            frac_for_model = agn_frac
            _c_aa = 2.99792458e18
            nu = _c_aa / wave
            L_bol_stellar = -jnp.trapezoid(sed_so_far, nu)
            agn_log_lbol = jnp.log10(jnp.maximum(L_bol_stellar * agn_frac, 1e-50))
            bol_erg = L_bol_stellar * agn_frac

        agn_sed = agn_model_fn(
            wave,
            agn_log_lbol=agn_log_lbol,
            agn_frac=frac_for_model,
            agn_alpha=agn_alpha,
            agn_T_torus=agn_T_torus,
            agn_tau_torus=agn_tau_torus,
            agn_torus_frac=agn_torus_frac,
            agn_log_mbh=agn_log_mbh,
            agn_log_ledd=agn_log_ledd,
        )
        return agn_sed, bol_erg

    return agn_fn


# -------------------------------------------------------------------
# Nebular emission
# -------------------------------------------------------------------


def build_nebular_component(model):
    """Build nebular emission from backend.

    Parameters
    ----------
    model : Model
        The model instance.

    Returns
    -------
    callable or None
        ``(weights, log_z, neb_params) -> neb_sed`` in erg/s/Hz,
        or None if nebular backend has no free params.
    """
    backend = model._nebular_backend
    if backend is None or not backend.has_free_params:
        return None

    ssp_wave = model.ssp_data.ssp_wave
    ssp_log_ages_yr = model.ssp_log_ages_yr

    def neb_fn(
        weights,
        log_z,
        neb_logU=-3.0,
        neb_logZ_gas=None,
        neb_fesc=0.0,
        neb_fesc_lya=0.0,
    ):
        return backend.predict_nebular_sed(
            ssp_weights=weights,
            ssp_wave=ssp_wave,
            ssp_log_ages_yr=ssp_log_ages_yr,
            log_z=log_z,
            neb_logU=neb_logU,
            neb_logZ_gas=neb_logZ_gas,
            neb_fesc=neb_fesc,
            neb_fesc_lya=neb_fesc_lya,
        )

    return neb_fn


# -------------------------------------------------------------------
# Radio emission
# -------------------------------------------------------------------


def build_radio_component(model):
    """Build radio emission (synchrotron from SF + AGN jets).

    Parameters
    ----------
    model : Model
        The model instance.

    Returns
    -------
    callable or None
        ``(L_ir, L_agn_bol, radio_params) -> radio_sed`` in erg/s/Hz,
        or None if radio is disabled.
    """
    if not model._radio_enabled:
        return None

    from tengri.models.radio import radio_total

    wave = model.ssp_data.ssp_wave

    def radio_fn(
        L_ir,
        L_agn_bol,
        radio_q_ir=2.64,
        radio_alpha_sf=0.8,
        radio_loudness=0.0,
        radio_alpha_agn=0.7,
    ):
        return radio_total(
            wave,
            L_ir=L_ir,
            L_agn_bol=L_agn_bol,
            q_ir=radio_q_ir,
            alpha_sf=radio_alpha_sf,
            radio_loudness=radio_loudness,
            alpha_agn=radio_alpha_agn,
        )

    return radio_fn


# -------------------------------------------------------------------
# X-ray emission
# -------------------------------------------------------------------


def build_xray_component(model):
    """Build X-ray emission (XRBs + AGN corona).

    Parameters
    ----------
    model : Model
        The model instance.

    Returns
    -------
    callable or None
        ``(sfr, mstar, L_agn_bol, xray_params) -> xray_sed``
        in erg/s/Hz, or None if X-ray is disabled.
    """
    if not model._xray_enabled:
        return None

    from tengri.models.xray import xray_total

    wave = model.ssp_data.ssp_wave

    def xray_fn(
        sfr,
        stellar_mass,
        L_agn_bol,
        xray_gamma_agn=1.8,
        xray_alpha_ox=-1.4,
    ):
        return xray_total(
            wave,
            sfr=sfr,
            stellar_mass=stellar_mass,
            L_agn_bol=L_agn_bol,
            gamma_agn=xray_gamma_agn,
            alpha_ox=xray_alpha_ox,
        )

    return xray_fn
