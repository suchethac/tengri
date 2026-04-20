"""Fused JIT kernel builders for fast photometry and spectroscopy.

These factory functions build @jax.jit closures that capture precomputed
arrays (SSP grids, dust weights, effective wavelengths) at build time.
The returned functions take only per-call parameters (SFR weights, dust
params) as arguments.

Extracted from ``SEDModel`` methods to keep model.py focused on orchestration.
"""

from __future__ import annotations

import jax.numpy as jnp


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
    from tengri.observation.photometry import (
        compute_flux_density_batch,
        pad_filters,
    )

    sed = rest_sed
    if apply_igm:
        from tengri.components.igm import igm_transmission

        wave_obs = wave_rest * (1.0 + z)
        igm_trans = igm_transmission(wave_obs, z)
        sed = sed * igm_trans

    fw_pad, ft_pad, _n_valid = pad_filters(filter_waves, filter_trans)
    return compute_flux_density_batch(sed, wave_rest, fw_pad, ft_pad, z, dl_cm)


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
    from tengri.observation.spectrum import compute_spectrum

    return compute_spectrum(rest_sed, wave_rest, wave_obs, z, dl_cm)


# ── Fused Tier 2 end-to-end kernels (params → photometry/spectrum)


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

    from tengri.components.sps.dsps_wrapper import compute_csp_weights
    from tengri.forward.pipeline import interp_met_alpha_dispatch, interp_metallicity
    from tengri.observation.photometry import (
        compute_flux_density_batch,
        pad_filters,
    )
    from tengri.parameters.translate import get_internal_params

    _use_dsps_native = model._csp_integration == "dsps_native"
    if _use_dsps_native:
        from tengri.components.sps.dsps_wrapper import compute_dsps_native_weights

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
    # Pad filters for vectorized integration
    fw_padded_t2, ft_padded_t2, _filt_nv_t2 = pad_filters(filter_waves, filter_trans)

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
        from tengri.components.igm import igm_transmission

        wave_obs_full = rest_wave * (1.0 + z_fixed)
        igm_trans_full = igm_transmission(wave_obs_full, z_fixed)

    # For free-z: need luminosity_distance inside JIT
    if is_free_z:
        from tengri.utils.cosmology import luminosity_distance as _lum_dist

        if apply_igm:
            from tengri.components.igm import igm_transmission as _igm_fn

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
            lgmet = p.get("log_z_abs", p.get("log_z_abs_final", -1.8477))
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
            _lgmet = p.get("log_z_abs", p.get("log_z_abs_final", -1.8477))
            if _use_alpha_fe_t2:
                alpha_fe = p.get("alpha_fe", 0.0)
                ssp_flux_at_z = interp_met_alpha_dispatch(model, _lgmet, alpha_fe)
            else:
                ssp_flux_at_z = interp_metallicity(model, _lgmet)

        # Always pass current SFR — needed by nebular (Q_H scaling) and X-ray.
        p = {**p, "_sfr_current": sfr_on_ssp[-1]}

        rest_sed = rest_sed_kernel(weights, ssp_flux_at_z, p)
        z = p.get("redshift", z_fixed if z_fixed is not None else 0.0)
        return rest_sed, z

    if is_free_z:

        def fused_tier2_phot(sfr_on_ssp, params):
            """sfr_on_ssp, params dict → observed photometry (free z)."""
            rest_sed, z = _compute_rest_sed(sfr_on_ssp, params)
            dl_cm = _lum_dist(z)

            if apply_igm:
                wave_obs = rest_wave * (1.0 + z)
                rest_sed = rest_sed * _igm_fn(wave_obs, z)

            return compute_flux_density_batch(
                rest_sed,
                rest_wave,
                fw_padded_t2,
                ft_padded_t2,
                z,
                dl_cm,
            )

    else:

        def fused_tier2_phot(sfr_on_ssp, params):
            """sfr_on_ssp, params dict → observed photometry (fixed z)."""
            rest_sed, _z = _compute_rest_sed(sfr_on_ssp, params)

            if igm_trans_full is not None:
                rest_sed = rest_sed * igm_trans_full

            return compute_flux_density_batch(
                rest_sed,
                rest_wave,
                fw_padded_t2,
                ft_padded_t2,
                z_fixed,
                dl_cm_fixed,
            )

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

    from tengri.components.sps.dsps_wrapper import compute_csp_weights
    from tengri.forward.pipeline import interp_met_alpha_dispatch, interp_metallicity
    from tengri.observation.spectrum import compute_spectrum
    from tengri.parameters.translate import get_internal_params

    _use_dsps_native_spec = model._csp_integration == "dsps_native"
    if _use_dsps_native_spec:
        from tengri.components.sps.dsps_wrapper import compute_dsps_native_weights

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
                _lgmet_s = p.get("log_z_abs", p.get("log_z_abs_final", -1.8477))
                ssp_flux_at_z = interp_met_alpha_dispatch(model, _lgmet_s, alpha_fe)
            else:
                ssp_flux_at_z = interp_metallicity(
                    model, p.get("log_z_abs", p.get("log_z_abs_final", -1.8477))
                )
        p = {**p, "_sfr_current": sfr_on_ssp[-1]}
        rest_sed = rest_sed_kernel(weights, ssp_flux_at_z, p)
        z = p.get("redshift", z_fixed if z_fixed is not None else 0.0)
        return rest_sed, z

    if is_free_z:

        def fused_tier2_spec(sfr_on_ssp, params):
            """sfr_on_ssp, params dict → observed spectrum (free z)."""
            rest_sed, z = _compute_rest_sed_spec(sfr_on_ssp, params)
            dl_cm = _lum_dist_spec(z)
            return compute_spectrum(rest_sed, rest_wave, wave_obs, z, dl_cm)

    else:

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

    from tengri.components.dust.attenuation import resolve_dust_law
    from tengri.components.sps.dsps_wrapper import compute_csp_weights
    from tengri.forward.pipeline import (
        interp_met_alpha_dispatch,
        interp_metallicity,
    )
    from tengri.parameters.translate import get_internal_params
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
        from tengri.components.sps.dsps_wrapper import has_alpha_grid

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
        _lgmet_hs = p.get("log_z_abs", p.get("log_z_abs_final", -1.8477))
        if _use_alpha_fe and _has_alpha:
            alpha_fe_val = p.get("alpha_fe", 0.0)
            ssp_flux_at_z = interp_met_alpha_dispatch(model, _lgmet_hs, alpha_fe_val)
        else:
            ssp_flux_at_z = interp_metallicity(model, _lgmet_hs)

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

        def hybrid_spec(sfr_on_ssp, params):
            """Single-dust hybrid spectrum."""
            p = get_internal_params(params, param_map, spec, has_field)
            _lgmet = p.get("log_z_abs", p.get("log_z_abs_final", -1.8477))
            return _hybrid_spec_body(sfr_on_ssp, params, _lgmet, (p.get("tau_v", 0.0),))

    else:

        def hybrid_spec(sfr_on_ssp, params):
            """Two-dust hybrid spectrum."""
            p = get_internal_params(params, param_map, spec, has_field)
            _lgmet = p.get("log_z_abs", p.get("log_z_abs_final", -1.8477))
            return _hybrid_spec_body(
                sfr_on_ssp,
                params,
                _lgmet,
                (p.get("tau_bc", 0.0), p.get("tau_diff", 0.0)),
            )

    return hybrid_spec
