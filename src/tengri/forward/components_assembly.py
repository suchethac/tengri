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

# ── SSP component: metallicity interpolation + CSP weighted sum ───


def build_ssp_component(model):
    """Build SSP flux and CSP weight computation closure.

    Captures SSP grid, metallicity mode, and dust age weights, returning a
    pure JAX function for metallicity interpolation and CSP integration.

    Parameters
    ----------
    model : SEDModel
        The model instance providing config and precomputed arrays.

    Returns
    -------
    callable
        A function with signature
        ``(sfr_on_ssp, log_z_abs, alpha_fe) -> (ssp_flux_at_z, weights)``.

        - ``ssp_flux_at_z``: ndarray, shape (n_age, n_wave) — Z-interpolated
          SSP [erg/s/Hz]
        - ``weights``: ndarray, shape (n_age,) — CSP mass weights [Msun]

    Notes
    -----
    **JIT-compatible**: yes — all operations use ``jnp`` primitives.
    """
    from tengri.components.stellar.sps.dsps_wrapper import (
        _ALPHA_TO_Z_COEFF as _A2Z,
        compute_lgmet_weights,
    )

    dt = model._forward_dtype
    ssp_flux = model.ssp_data.ssp_flux.astype(dt)
    ssp_lgmet = model.ssp_data.ssp_lgmet.astype(dt)
    use_smooth = model._met_interp == "smooth"
    lgmet_scatter = dt.type(model._lgmet_scatter)
    _use_matrix = model._csp_integration == "log_interp"
    if _use_matrix:
        _csp_mat = model._csp_matrix.astype(dt)  # precomputed Johnson+2021 matrix
    else:
        _age_dt = model._csp_age_dt.astype(dt)  # precomputed bin widths

    def ssp_fn(sfr_on_ssp, log_z_abs, alpha_fe=0.0):
        """Interpolate SSP at fixed metallicity and integrate SFR profile.

        Parameters
        ----------
        sfr_on_ssp : array_like, shape (n_age,)
            Star formation rate on SSP age grid [Msun/yr].
        log_z_abs : float
            Absolute log10 metallicity [dimensionless].
        alpha_fe : float, optional
            Alpha-element enhancement [dex]. Default 0.0.

        Returns
        -------
        ssp_at_z : ndarray, shape (n_age, n_wave)
            Z-interpolated SSP flux [erg/s/Hz].
        weights : ndarray, shape (n_age,)
            CSP mass weights [Msun].

        Notes
        -----
        **JIT-compatible**: yes — all operations use ``jnp`` primitives.
        """
        sfr = sfr_on_ssp.astype(dt)
        lz = jnp.asarray(log_z_abs, dtype=dt)
        afe = jnp.asarray(alpha_fe, dtype=dt)

        weights = _csp_mat @ sfr if _use_matrix else sfr * _age_dt

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


# ── Dust attenuation: two-component model ─────────────────────────


def build_dust_atten_component(model):
    """Build dust attenuation closure for age-resolved SSP fluxes.

    Captures dust law functions and precomputed age weights, returning a
    pure JAX function for Charlot & Fall two-component attenuation.

    Parameters
    ----------
    model : SEDModel
        The model instance.

    Returns
    -------
    callable
        A function with signature
        ``(ssp_flux_at_z, weights, tau_bc, tau_diff, ...) -> (sed_atten, sed_intr, L_abs)``.

        - ``sed_attenuated``: ndarray, shape (n_wave,) — attenuated SED
          [erg/s/Hz]
        - ``sed_intrinsic``: ndarray, shape (n_wave,) — intrinsic (unattenuated)
          SED [erg/s/Hz]
        - ``L_absorbed``: ndarray, shape () — integrated absorbed luminosity
          [erg/s]

    Notes
    -----
    **JIT-compatible**: yes — all operations use ``jnp`` primitives.
    """
    from tengri.components.stellar.sps.dsps_wrapper import LSUN_ERG_PER_S

    dt = model._forward_dtype
    ssp_wave = model.ssp_data.ssp_wave.astype(dt)
    dust_age_w = model._precomputed.dust_age_weights.astype(dt)
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
        """Apply two-component dust attenuation and compute absorbed energy.

        Parameters
        ----------
        ssp_flux_at_z : array_like, shape (n_age, n_wave)
            Z-interpolated SSP flux [erg/s/Hz].
        weights : array_like, shape (n_age,)
            CSP mass weights [Msun].
        tau_bc : float
            Birth-cloud dust optical depth [dimensionless].
        tau_diff : float
            Diffuse ISM dust optical depth [dimensionless].
        dust_slope : float, optional
            Dust attenuation slope [dimensionless]. Default -0.7.
        f_obscuration : float, optional
            Foreground obscuration fraction [dimensionless]. Default 0.0.
        dust_bump_strength : float, optional
            Dust bump strength (2175 A feature) [dimensionless]. Default 0.0.
        dust_delta : float, optional
            Dust delta parameter [dimensionless]. Default 0.0.
        dust_Rv : float, optional
            Dust Rv parameter [dimensionless]. Default 3.1.

        Returns
        -------
        sed_atten : ndarray, shape (n_wave,)
            Attenuated SED [erg/s/Hz].
        sed_intr : ndarray, shape (n_wave,)
            Intrinsic (unattenuated) SED [erg/s/Hz].
        L_absorbed : ndarray, shape ()
            Integrated absorbed luminosity [erg/s].

        Notes
        -----
        **JIT-compatible**: yes — all operations use ``jnp`` primitives.
        """
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


# ── Dust IR emission (energy-balanced) ────────────────────────────


def build_dust_emission_component(model):
    """Build dust IR emission closure from absorbed luminosity.

    Captures the emission model type and returns a pure JAX function
    for energy-balanced dust re-radiation.

    Parameters
    ----------
    model : SEDModel
        The model instance.

    Returns
    -------
    callable or None
        A function with signature ``(L_ir, dust_T, dust_beta_ir, ...) -> dust_ir_sed``
        in [erg/s/Hz], or None if dust emission is disabled.

    Notes
    -----
    **JIT-compatible**: yes — all operations use ``jnp`` primitives.
    """
    if model._dust_emission_model is None:
        return None

    from tengri.components.dust.emission import resolve_emission_model

    emission_fn = resolve_emission_model(model._dust_emission_model)
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
        """Compute dust IR emission SED from absorbed luminosity scaling.

        Parameters
        ----------
        L_ir : float
            Absorbed luminosity [erg/s].
        dust_T : float, optional
            Dust temperature [K]. Default 35.0.
        dust_beta_ir : float, optional
            Dust emissivity index [dimensionless]. Default 1.6.
        dust_alpha_mir : float, optional
            MIR slope parameter [dimensionless]. Default 2.0.
        dust_alpha_dale : float, optional
            Dale et al. template alpha parameter [dimensionless]. Default 2.0.
        dust_umin : float, optional
            Minimum radiation field strength [dimensionless]. Default 1.0.
        dust_gamma_dl : float, optional
            DL07 gamma parameter [dimensionless]. Default 0.01.
        dust_qpah : float, optional
            PAH mass fraction [%]. Default 2.5.
        dust_eta_balance : float, optional
            Energy balance correction factor [dimensionless]. Default 1.0.

        Returns
        -------
        dust_ir_sed : ndarray, shape (n_wave,)
            Dust IR emission SED [erg/s/Hz].

        Notes
        -----
        **JIT-compatible**: yes — all operations use ``jnp`` primitives.
        """
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


# ── AGN contribution ──────────────────────────────────────────────


def build_agn_component(model):
    """Build AGN SED contribution closure.

    Captures AGN model type and luminosity mode, returning a pure JAX function
    for AGN spectral energy distribution synthesis.

    Parameters
    ----------
    model : SEDModel
        The model instance.

    Returns
    -------
    callable or None
        A function with signature ``(sed_so_far, agn_log_lbol, ...) ->
        (agn_sed, bol_erg)`` returning AGN SED in [erg/s/Hz] and bolometric
        luminosity in [erg/s], or None if AGN is disabled.

    Notes
    -----
    **JIT-compatible**: yes — all operations use ``jnp`` primitives.
    """
    if model._agn_model is None:
        return None

    from tengri.components.agn import resolve_agn_model

    agn_model_fn = resolve_agn_model(model._agn_model)
    wave = model.ssp_data.ssp_wave
    is_parametric = model._agn_luminosity_mode

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
        """Synthesize AGN spectrum and derive bolometric luminosity.

        Parameters
        ----------
        sed_so_far : array_like, shape (n_wave,), optional
            Prior SED for deriving AGN fraction [erg/s/Hz]. Default None.
        agn_log_lbol : float, optional
            Log10 AGN bolometric luminosity [log10(Lsun)]. Default 10.0.
        agn_frac : float, optional
            AGN fraction in total (stellar + AGN) luminosity [dimensionless].
            Default 0.0.
        agn_alpha : float, optional
            AGN power-law slope [dimensionless]. Default -1.0.
        agn_T_torus : float, optional
            Torus dust temperature [K]. Default 1000.0.
        agn_tau_torus : float, optional
            Torus dust optical depth [dimensionless]. Default 5.0.
        agn_torus_frac : float, optional
            Torus-obscured fraction [dimensionless]. Default 0.5.
        agn_log_mbh : float, optional
            Log10 black hole mass [log10(Msun)]. Default 7.0.
        agn_log_ledd : float, optional
            Log10 Eddington ratio [dimensionless]. Default -1.0.

        Returns
        -------
        agn_sed : ndarray, shape (n_wave,)
            AGN SED [erg/s/Hz].
        bol_erg : ndarray, shape ()
            AGN bolometric luminosity [erg/s].

        Notes
        -----
        **JIT-compatible**: yes — all operations use ``jnp`` primitives.
        """
        if is_parametric:
            frac_for_model = 1.0
            bol_erg = 10.0**agn_log_lbol * 3.828e33  # Lsun → erg/s
        else:
            frac_for_model = agn_frac
            _c_aa = 2.99792458e18
            nu = _c_aa / wave
            L_bol_stellar = -jnp.trapezoid(sed_so_far, nu)
            bol_erg = L_bol_stellar * agn_frac
            # AGN model functions expect log10(L_bol / Lsun), convert from erg/s
            agn_log_lbol = jnp.log10(jnp.maximum(bol_erg, 1e-50)) - jnp.log10(3.828e33)

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


# ── Nebular emission ──────────────────────────────────────────────


def build_nebular_component(model):
    """Build nebular emission closure from backend.

    Captures nebular backend and returns a pure JAX function for
    nebular line and continuum synthesis.

    Parameters
    ----------
    model : SEDModel
        The model instance.

    Returns
    -------
    callable or None
        A function with signature ``(weights, log_z, neb_logU, ...) -> neb_sed``
        in [erg/s/Hz], or None if nebular backend has no free parameters.

    Notes
    -----
    **JIT-compatible**: yes — all operations use ``jnp`` primitives.
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
        """Synthesize nebular lines and continuum emission.

        Parameters
        ----------
        weights : array_like, shape (n_age,)
            CSP mass weights [Msun].
        log_z : float
            Log10 absolute metallicity [dimensionless].
        neb_logU : float, optional
            Log10 ionization parameter [dimensionless]. Default -3.0.
        neb_logZ_gas : float, optional
            Log10 gas-phase metallicity Z/Zsun [dimensionless].
            If None, derived from log_z. Default None.
        neb_fesc : float, optional
            Ionizing photon escape fraction [dimensionless]. Default 0.0.
        neb_fesc_lya : float, optional
            Lyman-alpha escape fraction [dimensionless]. Default 0.0.

        Returns
        -------
        neb_sed : ndarray, shape (n_wave,)
            Nebular emission SED [erg/s/Hz].

        Notes
        -----
        **JIT-compatible**: yes — all operations use ``jnp`` primitives.
        """
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


# ── Radio emission ────────────────────────────────────────────────


def build_radio_component(model):
    """Build radio emission closure from synchrotron and free-free.

    Captures radio model configuration and returns a pure JAX function
    for radio SED synthesis.

    Parameters
    ----------
    model : SEDModel
        The model instance.

    Returns
    -------
    callable or None
        A function with signature ``(L_ir, L_agn_bol, radio_q_ir, ...) -> radio_sed``
        in [erg/s/Hz], or None if radio is disabled.

    Notes
    -----
    **JIT-compatible**: yes — all operations use ``jnp`` primitives.
    """
    if not model._uses_radio:
        return None

    from tengri.components.radio import radio_total

    wave = model.ssp_data.ssp_wave

    def radio_fn(
        L_ir,
        L_agn_bol,
        radio_q_ir=2.64,
        radio_alpha_sf=0.8,
        radio_loudness=0.0,
        radio_alpha_agn=0.7,
    ):
        """Synthesize radio SED from synchrotron and free-free sources.

        Parameters
        ----------
        L_ir : float
            Star-formation-related luminosity (IR or bolometric) [erg/s].
        L_agn_bol : float
            AGN bolometric luminosity [erg/s].
        radio_q_ir : float, optional
            FIR-to-radio luminosity ratio [dimensionless]. Default 2.64.
        radio_alpha_sf : float, optional
            Star-formation synchrotron slope [dimensionless]. Default 0.8.
        radio_loudness : float, optional
            AGN radio loudness parameter [dimensionless]. Default 0.0.
        radio_alpha_agn : float, optional
            AGN radio slope [dimensionless]. Default 0.7.

        Returns
        -------
        radio_sed : ndarray, shape (n_wave,)
            Radio emission SED [erg/s/Hz].

        Notes
        -----
        **JIT-compatible**: yes — all operations use ``jnp`` primitives.
        """
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


# ── X-ray emission ────────────────────────────────────────────────


def build_xray_component(model):
    """Build X-ray emission closure from XRBs and AGN corona.

    Captures X-ray model configuration and returns a pure JAX function
    for X-ray SED synthesis.

    Parameters
    ----------
    model : SEDModel
        The model instance.

    Returns
    -------
    callable or None
        A function with signature ``(sfr, stellar_mass, L_agn_bol, ...)
        -> xray_sed`` in [erg/s/Hz], or None if X-ray is disabled.

    Notes
    -----
    **JIT-compatible**: yes — all operations use ``jnp`` primitives.
    """
    if not model._uses_xray:
        return None

    from tengri.components.xray import xray_total

    wave = model.ssp_data.ssp_wave

    def xray_fn(
        sfr,
        stellar_mass,
        L_agn_bol,
        xray_gamma_agn=1.8,
        xray_alpha_ox=-1.4,
    ):
        """Synthesize X-ray SED from XRB and AGN emission.

        Parameters
        ----------
        sfr : float
            Current star formation rate [Msun/yr].
        stellar_mass : float
            Total stellar mass [Msun].
        L_agn_bol : float
            AGN bolometric luminosity [erg/s].
        xray_gamma_agn : float, optional
            AGN X-ray spectral index (photon index) [dimensionless].
            Default 1.8.
        xray_alpha_ox : float, optional
            X-ray to optical slope (alpha_ox) [dimensionless]. Default -1.4.

        Returns
        -------
        xray_sed : ndarray, shape (n_wave,)
            X-ray emission SED [erg/s/Hz].

        Notes
        -----
        **JIT-compatible**: yes — all operations use ``jnp`` primitives.
        """
        return xray_total(
            wave,
            sfr=sfr,
            stellar_mass=stellar_mass,
            L_agn_bol=L_agn_bol,
            gamma_agn=xray_gamma_agn,
            alpha_ox=xray_alpha_ox,
        )

    return xray_fn
