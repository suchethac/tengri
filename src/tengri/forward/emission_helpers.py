"""Shared emission-component helpers for the SED forward model.

Both the non-fused pipeline (``sed_pipeline.py``) and the fused JIT
kernel (``fused_kernels.py``) call these same functions, guaranteeing
identical physics.  All helpers are **pure functions** taking explicit
arguments (no ``model`` object), so they work inside ``@jax.jit``
closures as well as plain Python.

Each helper computes one emission component and returns the SED
(erg/s/Hz).  Orchestration (branching on ``has_nebular``, component
tracking, wavelength interpolation) stays in the caller.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

# ── Constants ────────────────────────────────────────────────────────────────

_C_AA: float = 2.99792458e18  # speed of light in Å/s
_LSUN: float = 3.828e33  # erg/s  (IAU 2015 nominal solar luminosity)
_QH_PER_SFR: float = 4.2e53  # phot/s per Msun/yr  (Leitherer+1999, Chabrier IMF)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Nebular emission
# ═══════════════════════════════════════════════════════════════════════════


def nebular_emission(
    backend,
    weights: jnp.ndarray,
    ssp_wave: jnp.ndarray,
    ssp_log_ages_yr: jnp.ndarray,
    log_z_abs: float,
    sfr_current: float,
    *,
    neb_logU: float = -3.0,
    neb_logZ_gas: float | None = None,
    neb_fesc: float = 0.0,
    neb_fesc_lya: float = 0.0,
) -> jnp.ndarray:
    """Synthesize nebular emission (lines + continuum) with automatic Q_H mode selection.

    Parameters
    ----------
    backend : nebular backend instance
        Nebular emission backend (CueBackend, CloudyGridBackend, etc.).
    weights : ndarray, shape (n_age,)
        CSP mass weights [Msun].
    ssp_wave : ndarray, shape (n_wave,)
        SSP wavelength grid [Angstrom].
    ssp_log_ages_yr : ndarray, shape (n_age,)
        SSP age bin centers log10(age/yr) [dimensionless].
    log_z_abs : float
        Stellar metallicity log10(Z) absolute [dimensionless].
    sfr_current : float
        Current star formation rate [Msun/yr].
    neb_logU : float, optional
        Ionization parameter log10(U). Default -3.0.
    neb_logZ_gas : float, optional
        Gas-phase metallicity log10(Z/Zsun). If None, inferred from stellar.
    neb_fesc : float, optional
        Escape fraction of LyC photons (0-1). Default 0.0.
    neb_fesc_lya : float, optional
        Lyman-alpha escape fraction (0-1). Default 0.0.

    Returns
    -------
    ndarray, shape (n_wave,)
        Nebular SED [erg/s/Hz] before dust attenuation.

    Notes
    -----
    **JIT-compatible**: yes — uses ``jax.lax.cond`` for Q_H mode branching.
    """
    # SFR-based Q_H
    qh_from_sfr = _QH_PER_SFR * sfr_current
    gas_logqion_sfr = jnp.log10(jnp.maximum(qh_from_sfr, 1.0))

    # Detect wNE SSPs: if SSP-derived Q_H < 1% of SFR-based Q_H,
    # the ionizing spectrum is pre-absorbed → use low-level mode
    # (default ionspec).  Use jax.lax.cond for JIT-safe branching.
    if hasattr(backend, "_compute_weighted_cue_params"):
        derived = backend._compute_weighted_cue_params(
            weights,
            ssp_log_ages_yr,
            log_z_abs,
            neb_logU=neb_logU,
        )
        ssp_logqion = derived.get("gas_logqion", jnp.float64(0.0))
        ssp_qh_ok = ssp_logqion > gas_logqion_sfr - 2.0  # within 1%

        def _ssp_path(_):
            """Use SSP-derived ionizing photon spectrum."""
            return backend.predict_nebular_sed(
                ssp_weights=weights,
                ssp_wave=ssp_wave,
                ssp_log_ages_yr=ssp_log_ages_yr,
                log_z=log_z_abs,
                neb_logU=neb_logU,
                neb_logZ_gas=neb_logZ_gas,
                neb_fesc=neb_fesc,
                neb_fesc_lya=neb_fesc_lya,
                gas_logqion=gas_logqion_sfr,
            )

        def _fallback_path(_):
            """Use SFR-derived ionizing photon spectrum."""
            return backend.predict_nebular_sed(
                ssp_wave=ssp_wave,
                log_z=log_z_abs,
                neb_logU=neb_logU,
                neb_logZ_gas=neb_logZ_gas,
                neb_fesc=neb_fesc,
                neb_fesc_lya=neb_fesc_lya,
                gas_logqion=gas_logqion_sfr,
            )

        return jax.lax.cond(ssp_qh_ok, _ssp_path, _fallback_path, None)

    # Non-Cue backends: always pass ssp_weights
    return backend.predict_nebular_sed(
        ssp_weights=weights,
        ssp_wave=ssp_wave,
        ssp_log_ages_yr=ssp_log_ages_yr,
        log_z=log_z_abs,
        neb_logU=neb_logU,
        neb_logZ_gas=neb_logZ_gas,
        neb_fesc=neb_fesc,
        neb_fesc_lya=neb_fesc_lya,
        gas_logqion=gas_logqion_sfr,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 2. Dust attenuation of emission components (nebular / shock)
# ═══════════════════════════════════════════════════════════════════════════


def attenuate_emission(
    sed: jnp.ndarray,
    wave: jnp.ndarray,
    mode: str,
    tau_bc: float,
    tau_diff: float,
    law_bc_fn,
    law_diff_fn,
    *,
    neb_bc_fn=None,
    dust_slope: float = -0.7,
    dust_bump_strength: float = 0.0,
) -> tuple[jnp.ndarray, float]:
    """Apply dust attenuation to emission and integrate absorbed luminosity.

    Parameters
    ----------
    sed : ndarray, shape (n_wave,)
        Input SED [erg/s/Hz] (before dust).
    wave : ndarray, shape (n_wave,)
        Wavelength grid [Angstrom].
    mode : str
        Attenuation mode: ``"bc"`` (birth-cloud + diffuse), ``"diff"`` (diffuse only),
        ``"neb"`` (separate BC law + diffuse), or ``"none"`` (no attenuation).
    tau_bc : float
        Birth-cloud V-band optical depth [dimensionless].
    tau_diff : float
        Diffuse V-band optical depth [dimensionless].
    law_bc_fn : callable
        Birth-cloud dust law ``(wave, n_slope, dust_bump_strength) -> k(λ)`` [1/mag].
    law_diff_fn : callable
        Diffuse dust law ``(wave, n_slope, dust_bump_strength) -> k(λ)`` [1/mag].
    neb_bc_fn : callable, optional
        Separate BC law for ``mode="neb"``. Falls back to ``law_bc_fn`` if None.
    dust_slope : float, optional
        Dust law slope parameter. Default -0.7.
    dust_bump_strength : float, optional
        Dust law bump strength parameter. Default 0.0.

    Returns
    -------
    sed_attenuated : ndarray, shape (n_wave,)
        Attenuated SED [erg/s/Hz].
    L_absorbed : float
        Integrated absorbed luminosity [erg/s].

    Notes
    -----
    **JIT-compatible**: yes — all operations use ``jnp`` primitives.
    """
    if mode == "none":
        return sed, jnp.float64(0.0)

    dust_kw = {"n_slope": dust_slope, "dust_bump_strength": dust_bump_strength}
    sed_out = sed

    # Birth-cloud attenuation (modes "bc" and "neb").
    # Always compute (exp(-0*k)=1 is a no-op); avoids Python boolean
    # conversion on traced tau_bc which fails inside @jax.jit.
    if mode in ("bc", "neb"):
        bc_fn = neb_bc_fn if (mode == "neb" and neb_bc_fn is not None) else law_bc_fn
        if bc_fn is not None:
            k_bc = bc_fn(wave, **dust_kw)
            sed_out = sed_out * jnp.exp(-tau_bc * k_bc)

    # Diffuse ISM attenuation (all modes except "none").
    # Same: always compute, let XLA optimize exp(-0*k)=1.
    if law_diff_fn is not None:
        k_diff = law_diff_fn(wave, **dust_kw)
        sed_out = sed_out * jnp.exp(-tau_diff * k_diff)

    # Absorbed luminosity: integrate (input − output) over frequency
    nu = _C_AA / wave
    L_absorbed = -jnp.trapezoid(sed - sed_out, nu)
    L_absorbed = jnp.maximum(L_absorbed, 0.0)

    return sed_out, L_absorbed


# ═══════════════════════════════════════════════════════════════════════════
# 3. Shock emission
# ═══════════════════════════════════════════════════════════════════════════


def shock_emission(
    wave: jnp.ndarray,
    sed_so_far: jnp.ndarray,
    shock_frac: float,
    shock_velocity: float = 300.0,
    shock_log_density: float = 0.0,
    shock_b_over_sqrt_n: float = 1.0,
    shock_abundance: str = "solar",
    shock_component: str = "combined",
) -> jnp.ndarray:
    """Synthesize shock emission SED (MAPPINGS V).

    Returns raw shock SED before dust attenuation.

    Parameters
    ----------
    wave : ndarray, shape (n_wave,)
        Wavelength grid [Angstrom].
    sed_so_far : ndarray, shape (n_wave,)
        Current cumulative SED [erg/s/Hz] for bolometric luminosity estimation.
    shock_frac : float
        Fraction of Halpha luminosity channeled into shocks [dimensionless].
    shock_velocity : float, optional
        Shock velocity [km/s]. Default 300.0.
    shock_log_density : float, optional
        Log10 of electron density [cm^-3]. Default 0.0.
    shock_b_over_sqrt_n : float, optional
        Magnetic parameter B/sqrt(n). Default 1.0.
    shock_abundance : str, optional
        Abundance set ("solar", etc.). Default "solar".
    shock_component : str, optional
        Component to return ("combined", etc.). Default "combined".

    Returns
    -------
    ndarray, shape (n_wave,)
        Shock emission SED [erg/s/Hz] before dust attenuation.

    Notes
    -----
    **JIT-compatible**: yes — all operations use ``jnp`` primitives.
    """
    from tengri.components.nebular.shock import compute_shock_sed

    nu = _C_AA / wave
    l_bol = -jnp.trapezoid(sed_so_far, nu)
    l_halpha_approx = jnp.maximum(l_bol * 1e-3, 1e-30)
    l_shock_halpha = shock_frac * l_halpha_approx

    return compute_shock_sed(
        wave,
        shock_velocity,
        l_shock_halpha,
        shock_log_density=shock_log_density,
        shock_b_over_sqrt_n=shock_b_over_sqrt_n,
        shock_abundance=shock_abundance,
        shock_component=shock_component,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 4. AGN emission
# ═══════════════════════════════════════════════════════════════════════════


def agn_emission(
    agn_fn,
    wave: jnp.ndarray,
    agn_log_lbol: float,
    agn_frac: float,
    *,
    agn_polar_ebv: float = 0.0,
    agn_cos_inc: float = 0.5,
    agn_polar_oa: float = 45.0,
    **agn_params,
) -> jnp.ndarray:
    """Synthesize AGN emission with optional polar dust reddening.

    Parameters
    ----------
    agn_fn : callable
        Resolved AGN model function (e.g., ``kubota_done_full_agn``).
    wave : ndarray, shape (n_wave,)
        Wavelength grid [Angstrom].
    agn_log_lbol : float
        log10(L_bol / Lsun) bolometric luminosity [dimensionless].
    agn_frac : float
        AGN SED fraction (1.0 for parametric mode) [dimensionless].
    agn_polar_ebv : float, optional
        Polar dust reddening E(B-V) [mag]. Default 0.0 (no polar dust).
    agn_cos_inc : float, optional
        cos(inclination angle) for viewing geometry [dimensionless]. Default 0.5.
    agn_polar_oa : float, optional
        Polar opening angle [degrees]. Default 45.0.
    **agn_params
        Additional AGN parameters forwarded to ``agn_fn``.

    Returns
    -------
    ndarray, shape (n_wave,)
        AGN SED [erg/s/Hz].

    Notes
    -----
    **JIT-compatible**: yes — uses ``jax.lax.cond`` for polar dust branching.
    """
    agn_sed = agn_fn(
        wave,
        agn_log_lbol=agn_log_lbol,
        agn_frac=agn_frac,
        agn_cos_inc=agn_cos_inc,
        **agn_params,
    )

    # Polar dust: use jax.lax.cond for JIT compatibility (agn_polar_ebv
    # is a traced value inside @jit, so Python `if` would fail).
    from tengri.components.agn.polar_dust import polar_dust_total

    def _apply_polar_dust(sed):
        """Attenuate and re-emit through polar dust."""
        agn_lsun = sed / _LSUN
        att_lsun, reemit_lsun = polar_dust_total(
            agn_lsun,
            wave,
            cos_inc=agn_cos_inc,
            opening_angle_deg=agn_polar_oa,
            ebv=agn_polar_ebv,
        )
        return (att_lsun + reemit_lsun) * _LSUN

    agn_sed = jax.lax.cond(
        jnp.asarray(agn_polar_ebv) > 0.0,
        _apply_polar_dust,
        lambda sed: sed,
        agn_sed,
    )

    return agn_sed


# ═══════════════════════════════════════════════════════════════════════════
# 5. Dust IR emission (energy-balanced)
# ═══════════════════════════════════════════════════════════════════════════


def dust_ir_emission(
    emission_fn,
    wave: jnp.ndarray,
    L_ir: float,
    **dust_params,
) -> jnp.ndarray:
    """Synthesize dust IR re-emission from absorbed luminosity.

    Parameters
    ----------
    emission_fn : callable
        Resolved dust emission function (e.g., DL07 grid).
    wave : ndarray, shape (n_wave,)
        Wavelength grid [Angstrom].
    L_ir : float
        Total absorbed luminosity from dust attenuation [erg/s].
    **dust_params
        Dust emission parameters (T, beta_ir, alpha_mir, etc.).

    Returns
    -------
    ndarray, shape (n_wave,)
        Dust IR SED [erg/s/Hz].

    Notes
    -----
    **JIT-compatible**: yes — all operations use ``jnp`` primitives.
    """
    # Guard against NaN/Inf L_ir (can occur with pure SSPs)
    L_ir_safe = jnp.where(jnp.isfinite(L_ir), L_ir, 0.0)

    return emission_fn(
        wave,
        L_ir_safe,
        dust_T=dust_params.get("dust_T", 35.0),
        dust_beta_ir=dust_params.get("dust_beta_ir", 1.6),
        dust_alpha_mir=dust_params.get("dust_alpha_mir", 2.0),
        dust_alpha_dale=dust_params.get("dust_alpha_dale", 2.0),
        dust_umin=dust_params.get("dust_umin", 1.0),
        dust_gamma_dl=dust_params.get("dust_gamma_dl", 0.01),
        dust_qpah=dust_params.get("dust_qpah", 2.5),
        dust_alpha_dl14=dust_params.get("dust_alpha_dl14", 2.0),
    )


# ═══════════════════════════════════════════════════════════════════════════
# 6. Radio emission
# ═══════════════════════════════════════════════════════════════════════════


def radio_emission(
    wave: jnp.ndarray,
    L_ir: float,
    L_agn_bol: float,
    *,
    q_ir: float = 2.64,
    alpha_sf: float = 0.8,
    radio_loudness: float = 0.0,
    alpha_agn: float = 0.7,
    sfr_mode: str = "bell2003",
    log_mstar: float = 10.0,
    redshift: float = 0.0,
    include_freefree: bool = False,
    T_e: float = 1e4,
    alpha_ff: float = -0.1,
) -> jnp.ndarray:
    """Synthesize radio emission SED from SF synchrotron and AGN jets.

    Parameters
    ----------
    wave : ndarray, shape (n_wave,)
        Wavelength grid [Angstrom].
    L_ir : float
        Infrared luminosity [erg/s].
    L_agn_bol : float
        AGN bolometric luminosity [erg/s].
    q_ir : float, optional
        q-parameter for IR-radio correlation. Default 2.64.
    alpha_sf : float, optional
        Star formation synchrotron spectral index. Default 0.8.
    radio_loudness : float, optional
        AGN radio loudness offset. Default 0.0.
    alpha_agn : float, optional
        AGN radio spectral index. Default 0.7.
    sfr_mode : str, optional
        SFR→radio conversion model. Default "bell2003".
    log_mstar : float, optional
        log10(stellar mass / Msun). Default 10.0.
    redshift : float, optional
        Redshift for K-correction. Default 0.0.
    include_freefree : bool, optional
        Include free-free contribution. Default False.
    T_e : float, optional
        Electron temperature [K]. Default 1e4.
    alpha_ff : float, optional
        Free-free spectral index. Default -0.1.

    Returns
    -------
    ndarray, shape (n_wave,)
        Radio SED [erg/s/Hz].

    Notes
    -----
    **JIT-compatible**: yes — all operations use ``jnp`` primitives.
    """
    from tengri.components.radio import radio_total

    return radio_total(
        wave,
        L_ir=L_ir,
        L_agn_bol=L_agn_bol,
        q_ir=q_ir,
        alpha_sf=alpha_sf,
        radio_loudness=radio_loudness,
        alpha_agn=alpha_agn,
        sfr_mode=sfr_mode,
        log_mstar=log_mstar,
        redshift=redshift,
        include_freefree=include_freefree,
        T_e=T_e,
        alpha_ff=alpha_ff,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 7. X-ray emission
# ═══════════════════════════════════════════════════════════════════════════


def xray_emission(
    wave: jnp.ndarray,
    sfr: float,
    stellar_mass: float,
    L_agn_bol: float,
    *,
    gamma_agn: float = 1.8,
    alpha_ox: float = -1.4,
    gamma_hmxb: float = 2.0,
    gamma_lmxb: float = 1.6,
    E_cut: float = 300.0,
) -> jnp.ndarray:
    """Synthesize X-ray emission SED from XRBs and AGN corona.

    Parameters
    ----------
    wave : ndarray, shape (n_wave,)
        Wavelength grid [Angstrom].
    sfr : float
        Star formation rate [Msun/yr].
    stellar_mass : float
        Stellar mass [Msun].
    L_agn_bol : float
        AGN bolometric luminosity [erg/s].
    gamma_agn : float, optional
        AGN photon index. Default 1.8.
    alpha_ox : float, optional
        UV-to-X-ray slope. Default -1.4.
    gamma_hmxb : float, optional
        HMXB photon index. Default 2.0.
    gamma_lmxb : float, optional
        LMXB photon index. Default 1.6.
    E_cut : float, optional
        X-ray cutoff energy [keV]. Default 300.0.

    Returns
    -------
    ndarray, shape (n_wave,)
        X-ray SED [erg/s/Hz].

    Notes
    -----
    **JIT-compatible**: yes — all operations use ``jnp`` primitives.
    """
    from tengri.components.xray import xray_total

    return xray_total(
        wave,
        sfr=sfr,
        stellar_mass=stellar_mass,
        L_agn_bol=L_agn_bol,
        gamma_agn=gamma_agn,
        alpha_ox=alpha_ox,
        gamma_hmxb=gamma_hmxb,
        gamma_lmxb=gamma_lmxb,
        E_cut=E_cut,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 8. IGM absorption
# ═══════════════════════════════════════════════════════════════════════════


def igm_absorption(
    wave_obs: jnp.ndarray,
    z: float,
    igm_x_HI: float = 0.0,
    igm_bubble_mpc: float = 10.0,
    igm_patchy: bool = False,
    igm_model: str = "inoue",
) -> jnp.ndarray:
    """Compute IGM transmission with optional patchy reionization.

    Computes transmission using the selected mean-IGM model, optionally
    modified by Miralda-Escudé (1998) / Mason+2018 patchy reionization.

    Parameters
    ----------
    wave_obs : ndarray, shape (n_wave,)
        Observed-frame wavelength [Angstrom].
    z : float
        Redshift [dimensionless].
    igm_x_HI : float, optional
        Volume-averaged neutral hydrogen fraction (0-1). Default 0.0.
        Only used when ``igm_patchy=True``.
    igm_bubble_mpc : float, optional
        Ionized bubble radius [proper Mpc]. Default 10.0.
        Only used when ``igm_patchy=True``.
    igm_patchy : bool, optional
        Enable patchy reionization damping wing model. Default False.
    igm_model : str, optional
        Mean IGM model: ``"inoue"`` (Inoue+2014, default) or
        ``"madau"`` (Madau+1995).

    Returns
    -------
    ndarray, shape (n_wave,)
        Transmission fraction [dimensionless, 0-1].

    Notes
    -----
    **JIT-compatible**: yes — all operations use ``jnp`` primitives.
    Patchy reionization always uses Inoue+2014 as the mean-IGM base.
    """
    if igm_patchy and igm_x_HI > 0.0:
        from tengri.components.igm import igm_transmission_patchy

        return igm_transmission_patchy(wave_obs, z, x_HI=igm_x_HI, R_bubble=igm_bubble_mpc)

    if igm_model == "madau":
        from tengri.components.igm import igm_transmission_madau

        return igm_transmission_madau(wave_obs, z)

    from tengri.components.igm import igm_transmission

    return igm_transmission(wave_obs, z)
