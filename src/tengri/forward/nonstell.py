"""Non-stellar SED component infrastructure.

Provides:

* :class:`NonStellarSlot` — lightweight registry entry describing one enabled
  non-stellar SED component (nebular, shock, dust IR, AGN, radio, X-ray).
* :func:`collect_nonstell` — inspect a ``SEDModel`` and return the ordered list of
  enabled :class:`NonStellarSlot` objects.
* :func:`build_nonstell_fn` — factory that captures all component callables and
  model flags into a single JAX-traceable function
  ``nonstell_fn(weights, p, stellar_sed, stellar_sed_intr) -> sed``.

Both :func:`build_fused_rest_sed` (compositional kernel) and future refactors of
the hybrid kernel can call :func:`build_nonstell_fn` instead of duplicating the
~165-line per-component dispatch blocks.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NonStellarSlot:
    """Registry entry for one enabled non-stellar SED component.

    Attributes
    ----------
    name : str
        Canonical component name.  One of:
        ``"nebular"``, ``"shock"``, ``"dust_ir"``, ``"agn"``,
        ``"radio"``, ``"xray"``.
    dust_mode : str
        Dust attenuation mode applied to this component's emission:
        ``"bc"`` | ``"diff"`` | ``"neb"`` | ``"none"``.
    on_ssp_grid : bool
        ``True`` if the component lives on the SSP UV–NIR wavelength grid.
        ``False`` if it lives on the panchromatic (extended) grid.
    """

    name: str
    dust_mode: str = "none"
    on_ssp_grid: bool = True


# Canonical ordering enforced by energy-balance dependencies:
#   nebular → shock → (energy balance) → dust_ir → agn → radio → xray
_NONSTELL_ORDER = ("nebular", "shock", "dust_ir", "agn", "radio", "xray")


def collect_nonstell(model) -> list[NonStellarSlot]:
    """Inspect model and return ordered list of enabled non-stellar components.

    Parameters
    ----------
    model : SEDModel
        Fully initialized model instance.

    Returns
    -------
    list[NonStellarSlot]
        One entry per enabled non-stellar component, in canonical order.

    Notes
    -----
    **JIT-compatible**: no — uses Python-level attribute introspection.
    """
    slots: list[NonStellarSlot] = []

    # Nebular (requires backend + free params flag)
    if model._nebular_backend is not None and getattr(
        model._nebular_backend, "has_free_params", False
    ):
        slots.append(
            NonStellarSlot(
                "nebular",
                dust_mode=getattr(model, "_neb_dust_mode", "bc"),
                on_ssp_grid=True,
            )
        )

    # Shock
    if getattr(model, "_uses_shock", False):
        slots.append(NonStellarSlot("shock", dust_mode="diff", on_ssp_grid=True))

    # Dust IR (energy-balanced thermal re-emission)
    if model._dust_emission_model is not None:
        slots.append(NonStellarSlot("dust_ir", on_ssp_grid=False))

    # AGN
    if model._agn_model is not None:
        slots.append(NonStellarSlot("agn", on_ssp_grid=False))

    # Radio
    if model._uses_radio:
        slots.append(NonStellarSlot("radio", on_ssp_grid=False))

    # X-ray
    if model._uses_xray:
        slots.append(NonStellarSlot("xray", on_ssp_grid=False))

    return slots


def build_nonstell_fn(model, law_bc_fn, law_diff_fn, ssp_wave_f64, rest_wave_f64):
    """Build a JAX-traceable closure that adds all non-stellar SED components.

    Called once at model-build time to capture Python-level flags and component
    callables. The returned function can be called inside ``@jax.jit`` code.

    Parameters
    ----------
    model : SEDModel
        Fully initialized model instance.
    law_bc_fn : callable
        Birth-cloud dust extinction law [1/mag].
    law_diff_fn : callable
        Diffuse ISM dust extinction law [1/mag].
    ssp_wave_f64 : ndarray, shape (n_wave_ssp,)
        SSP wavelength grid [Angstrom].
    rest_wave_f64 : ndarray, shape (n_wave_rest,)
        Panchromatic rest-frame wavelength grid [Angstrom].
        May equal ``ssp_wave_f64`` when radio/X-ray disabled.

    Returns
    -------
    callable
        Function with signature ``(weights, p, stellar_sed, stellar_sed_intr) -> sed``.

        - ``weights``: ndarray, shape (n_age,) — CSP mass weights [Msun]
        - ``p``: dict — internal parameters
        - ``stellar_sed``: ndarray, shape (n_wave_ssp,) — attenuated stellar SED [erg/s/Hz]
        - ``stellar_sed_intr``: ndarray, shape (n_wave_ssp,) — intrinsic stellar SED [erg/s/Hz]
        - returns: ndarray, shape (n_wave_rest,) — full SED [erg/s/Hz]

    Notes
    -----
    **JIT-compatible**: yes — all operations use ``jnp`` primitives after closure creation.
    """
    import jax.numpy as jnp

    from tengri.components.sps.dsps_wrapper import LSUN_ERG_PER_S

    _is_single_dust = model._dust_model == "single_component"
    _c_aa = 2.99792458e18  # c in Angstrom/s
    _needs_extension = rest_wave_f64 is not ssp_wave_f64

    # --- Nebular ---
    has_nebular = model._nebular_backend is not None and getattr(
        model._nebular_backend, "has_free_params", False
    )
    if has_nebular:
        from tengri.forward.emission_helpers import attenuate_emission, nebular_emission

        nebular_backend = model._nebular_backend
        ssp_log_ages_yr = model.ssp_log_ages_yr
        _neb_dust_mode = getattr(model, "_neb_dust_mode", "bc")
        _neb_bc_fn = getattr(model, "_neb_dust_law_bc_fn", law_bc_fn)

    # --- Shock ---
    has_shock = getattr(model, "_uses_shock", False)
    if has_shock:
        from tengri.forward.emission_helpers import (
            attenuate_emission as _atten_shock,
            shock_emission,
        )

    # --- Dust IR ---
    has_dust_em = model._dust_emission_model is not None
    if has_dust_em:
        from tengri.components.dust.emission import preload_emission_model
        from tengri.forward.emission_helpers import dust_ir_emission

        # preload_emission_model ensures templates are loaded outside JIT,
        # preventing DynamicJaxprTracer leaks into closures.
        dust_emission_fn = preload_emission_model(model._dust_emission_model)

    # --- AGN ---
    has_agn = model._agn_model is not None
    agn_parametric = model._agn_luminosity_mode if has_agn else False
    if has_agn:
        from tengri.components.agn import resolve_agn_model
        from tengri.forward.emission_helpers import agn_emission

        agn_model_fn = resolve_agn_model(model._agn_model)

    # --- Radio ---
    has_radio = model._uses_radio
    if has_radio:
        from tengri.forward.emission_helpers import radio_emission

        _radio_sfr_mode = model._radio_sfr_mode
        _include_freefree = model._radio_include_freefree
        _redshift = float(getattr(model, "_redshift", 0.0))

    # --- X-ray ---
    has_xray = model._uses_xray
    if has_xray:
        from tengri.forward.emission_helpers import xray_emission

    def nonstell_fn(weights, p, stellar_sed, stellar_sed_intr):
        """Synthesize full SED by adding all enabled non-stellar components.

        Parameters
        ----------
        weights : ndarray, shape (n_age,)
            CSP mass weights [Msun].
        p : dict
            Internal parameter dictionary.
        stellar_sed : ndarray, shape (n_wave_ssp,)
            Dust-attenuated stellar SED [erg/s/Hz].
        stellar_sed_intr : ndarray, shape (n_wave_ssp,)
            Intrinsic (unattenuated) stellar SED [erg/s/Hz].

        Returns
        -------
        ndarray, shape (n_wave_rest,)
            Full SED (stellar + nebular + dust + AGN + radio + X-ray) [erg/s/Hz].
        """
        sed = stellar_sed
        L_abs_neb = jnp.float64(0.0)

        # 1. Nebular emission (on SSP grid)
        if has_nebular:
            # Current SFR from params (set by caller), NOT weights[-1] which
            # is total mass in the youngest bin (Msun), not SFR (Msun/yr).
            _sfr_last = p.get("_sfr_current", weights[-1])
            neb_raw = nebular_emission(
                nebular_backend,
                weights,
                ssp_wave_f64,
                ssp_log_ages_yr,
                p.get("log_z_abs", p.get("log_z_abs_final", -1.8477)),
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

        # 2. Shock emission (on SSP grid)
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
            shock_sed, _ = _atten_shock(
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

        # 3. Energy balance: L_ir = (L_absorbed_stellar + L_abs_neb) × η
        if has_dust_em or has_agn:
            nu_ssp = _c_aa / ssp_wave_f64
            L_absorbed_stellar = -jnp.trapezoid(stellar_sed_intr - stellar_sed, nu_ssp)
            # Guard against Inf/NaN at extreme metallicities
            L_absorbed_stellar = jnp.where(
                jnp.isfinite(L_absorbed_stellar), L_absorbed_stellar, 0.0
            )
            L_absorbed_stellar = jnp.maximum(L_absorbed_stellar, 0.0)
            L_absorbed = L_absorbed_stellar + L_abs_neb
            eta_balance = p.get("dust_eta_balance", 1.0)
            L_ir = jnp.maximum(L_absorbed * eta_balance, 0.0)
        else:
            L_ir = jnp.float64(0.0)

        # 4. AGN bolometric luminosity (needed before extending to panchromatic grid)
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
                nu_agn = _c_aa / ssp_wave_f64
                L_bol_stellar = -jnp.trapezoid(sed, nu_agn)
                agn_bol_erg = L_bol_stellar * agn_frac_val
                _log_lsun = jnp.log10(LSUN_ERG_PER_S)
                agn_log_lbol = jnp.log10(jnp.maximum(agn_bol_erg, 1e-50)) - _log_lsun

        # 5. Extend to panchromatic grid (if radio/X-ray enabled)
        if _needs_extension:
            from tengri.utils.wavelength import interpolate_sed_to_grid

            sed = interpolate_sed_to_grid(ssp_wave_f64, sed, rest_wave_f64)

        wave_z2 = rest_wave_f64

        # 6. Dust IR emission (energy-balanced, on panchromatic grid)
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

        # 7. AGN emission (on panchromatic grid)
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

        # 8. Radio emission (SF synchrotron + AGN jets + free-free)
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

        # 9. X-ray emission (HMXBs + LMXBs + AGN corona)
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

    return nonstell_fn
