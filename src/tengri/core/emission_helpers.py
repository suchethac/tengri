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
    """Compute nebular emission SED (lines + continuum) with wNE fallback.

    Parameters
    ----------
    backend
        Nebular backend instance (CueBackend, CloudyGridBackend, etc.).
    weights : array (n_age,)
        CSP mass weights.
    ssp_wave : array (n_wave,)
        SSP wavelength grid in Å.
    ssp_log_ages_yr : array (n_age,)
        log10(age/yr) of SSP age bins.
    log_z_abs : float
        Stellar metallicity log10(Z) (absolute).
    sfr_current : float
        Current SFR in Msun/yr (for SFR-based Q_H).

    Returns
    -------
    array (n_wave,)
        Nebular SED in erg/s/Hz on the SSP wavelength grid.
        **Before** dust attenuation — caller applies via :func:`attenuate_emission`.
    """
    # SFR-based Q_H
    qh_from_sfr = _QH_PER_SFR * sfr_current
    gas_logqion_sfr = jnp.log10(jnp.maximum(qh_from_sfr, 1.0))

    # Always use SSP-derived ionizing spectrum when available.
    # The wNE detection (SSP Q_H << SFR Q_H) previously branched with
    # a Python `if`, which fails inside @jax.jit.  Instead, always
    # pass ssp_weights — the backend handles the wNE case internally
    # (Cue uses default ionspec when SSP Q_H is negligible).
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
    """Apply dust attenuation and return absorbed luminosity.

    Parameters
    ----------
    sed : array (n_wave,)
        Input SED in erg/s/Hz (before dust).
    wave : array (n_wave,)
        Wavelength grid in Å.
    mode : str
        ``"bc"`` — birth-cloud + diffuse (Charlot & Fall 2000 default).
        ``"diff"`` — diffuse ISM only.
        ``"neb"`` — separate BC law (*neb_bc_fn*) + same diffuse law.
        ``"none"`` — no dust (returns input unchanged, L_absorbed=0).
    tau_bc, tau_diff : float
        Birth-cloud and diffuse V-band optical depths.
    law_bc_fn, law_diff_fn : callable
        Dust law functions ``(wave, n_slope=..., dust_bump_strength=...) -> k(λ)``.
    neb_bc_fn : callable or None
        Separate BC law for ``mode="neb"``.  Falls back to *law_bc_fn*.
    dust_slope, dust_bump_strength : float
        Dust law shape parameters.

    Returns
    -------
    sed_attenuated : array (n_wave,)
        Attenuated SED in erg/s/Hz.
    L_absorbed : float
        Total absorbed luminosity in erg/s (for energy balance).
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
    """Compute shock emission SED (MAPPINGS V).

    Returns the **raw** shock SED before dust attenuation.
    Caller applies diffuse dust via :func:`attenuate_emission`
    with ``mode="diff"``.

    Parameters
    ----------
    wave : array (n_wave,)
        Wavelength grid in Å.
    sed_so_far : array (n_wave,)
        Current cumulative SED (erg/s/Hz) for L_bol estimation.
    shock_frac : float
        Fraction of L_Halpha channelled into shock emission.

    Returns
    -------
    array (n_wave,)
        Shock emission SED in erg/s/Hz (before dust).
    """
    from tengri.models.nebular.shock import compute_shock_sed

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
    # Polar dust
    agn_polar_ebv: float = 0.0,
    agn_cos_inc: float = 0.5,
    agn_polar_oa: float = 45.0,
    # All other AGN params forwarded via kwargs
    **agn_params,
) -> jnp.ndarray:
    """Compute AGN emission including polar dust reddening.

    Parameters
    ----------
    agn_fn : callable
        Resolved AGN model function (e.g., ``kubota_done_full_agn``).
    wave : array (n_wave,)
        Wavelength grid in Å.
    agn_log_lbol : float
        log10(L_bol / Lsun).
    agn_frac : float
        AGN fraction (1.0 for parametric mode).
    agn_polar_ebv : float
        Polar dust E(B-V).  0 → no polar dust.

    Returns
    -------
    array (n_wave,)
        AGN SED in erg/s/Hz.
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
    from tengri.models.agn.polar_dust import polar_dust_total

    def _apply_polar_dust(sed):
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
    """Compute dust IR re-emission SED.

    Parameters
    ----------
    emission_fn : callable
        Resolved dust emission function (e.g., DL07 tabulated).
    wave : array (n_wave,)
        Wavelength grid in Å.
    L_ir : float
        Total absorbed luminosity in erg/s (energy balance input).

    Returns
    -------
    array (n_wave,)
        Dust IR SED in erg/s/Hz.
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
    """Compute radio emission SED (SF synchrotron + AGN jets + free-free).

    Returns
    -------
    array (n_wave,)
        Radio SED in erg/s/Hz.
    """
    from tengri.models.radio import radio_total

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
    """Compute X-ray emission SED (XRBs + AGN corona).

    Returns
    -------
    array (n_wave,)
        X-ray SED in erg/s/Hz.
    """
    from tengri.models.xray import xray_total

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
) -> jnp.ndarray:
    """Compute IGM transmission (Inoue+2014, optionally patchy).

    When ``igm_patchy=True`` and ``igm_x_HI > 0``, applies the
    Miralda-Escudé (1998) / Mason+2018 damping wing model on top
    of the mean Inoue+2014 transmission.

    Parameters
    ----------
    wave_obs : array (n_wave,)
        Observed-frame wavelength in Å.
    z : float
        Redshift.
    igm_x_HI : float
        Volume-averaged neutral hydrogen fraction (0–1).
        Only used when ``igm_patchy=True``.
    igm_bubble_mpc : float
        Ionized bubble radius in proper Mpc.
        Only used when ``igm_patchy=True``.
    igm_patchy : bool
        If True, apply patchy reionization damping wing.

    Returns
    -------
    array (n_wave,)
        Transmission fraction (0–1).
    """
    if igm_patchy and igm_x_HI > 0.0:
        from tengri.models.igm import igm_transmission_patchy

        return igm_transmission_patchy(wave_obs, z, x_HI=igm_x_HI, R_bubble=igm_bubble_mpc)

    from tengri.models.igm import igm_transmission

    return igm_transmission(wave_obs, z)
