"""Fused JIT kernel builders for fast photometry and spectroscopy.

These factory functions build @jax.jit closures that capture precomputed
arrays (SSP grids, dust weights, effective wavelengths) at build time.
The returned functions take only per-call parameters (SFR weights, dust
params) as arguments.

Extracted from ``SEDModel`` methods to keep model.py focused on orchestration.
"""

from __future__ import annotations

import jax.numpy as jnp

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
    from tengri.components.sps.dsps_wrapper import LSUN_ERG_PER_S

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
    from tengri.components.dust.attenuation import resolve_dust_law
    from tengri.components.sps.dsps_wrapper import LSUN_ERG_PER_S

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
    from tengri.forward.nonstell import build_nonstell_fn

    _law_diff_for_nonstell = law_diff_fn if not _is_single_dust else law_bc_fn
    nonstell_fn = build_nonstell_fn(
        model, law_bc_fn, _law_diff_for_nonstell, ssp_wave_f64, rest_wave_f64
    )

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
