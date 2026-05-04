"""Fused JIT kernel builders for fast photometry and spectroscopy.

These factory functions build @jax.jit closures that capture precomputed
arrays (SSP grids, dust weights, effective wavelengths) at build time.
The returned functions take only per-call parameters (SFR weights, dust
params) as arguments.

Extracted from ``SEDModel`` methods to keep model.py focused on orchestration.
"""

from __future__ import annotations

import jax.numpy as jnp

from tengri.forward._kernels.exact import build_fused_rest_sed
from tengri.forward.sed_model_types import SEDModelState


def observe_photometry_from_rest_sed(
    rest_sed,
    wave_rest,
    z,
    dl_cm,
    filter_waves,
    filter_trans,
    apply_igm=False,
    igm_fn=None,
):
    """Apply redshift + filter integration to a rest-frame SED.

    Thin wrapper that converts a Tier 2 rest-frame SED into observed
    photometric flux densities. Not JIT'd itself (loops over filters),
    but each filter integration is a fast JAX operation.

    Parameters
    ----------
    rest_sed : array_like, shape (n_wave,)
        Rest-frame SED. [erg/s/Hz]
    wave_rest : array_like, shape (n_wave,)
        Rest-frame wavelength grid. [Angstrom]
    z : float
        Redshift. [dimensionless]
    dl_cm : float
        Luminosity distance. [cm]
    filter_waves : list of array_like
        Filter wavelength arrays. [Angstrom]
    filter_trans : list of array_like
        Filter transmission arrays (normalized to 1). [dimensionless]
    apply_igm : bool, optional
        Whether to apply IGM absorption. Default: False.
    igm_fn : callable, optional
        IGM transmission function (wave_obs, z) → transmission.
        If None, defaults to ``igm_transmission``. Default: None.

    Returns
    -------
    ndarray, shape (n_filters,)
        Observed flux densities. [erg/s/cm^2/Hz]

    Notes
    -----
    **JIT-compatible**: no — loops over filters (filter integration is fast JAX ops).
    """
    from tengri.observation.photometry import (
        compute_flux_density_batch,
        pad_filters,
    )

    sed = rest_sed
    if apply_igm:
        if igm_fn is None:
            from tengri.components.igm import igm_transmission as igm_fn
        wave_obs = wave_rest * (1.0 + z)
        igm_trans = igm_fn(wave_obs, z)
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
    rest_sed : array_like, shape (n_wave,)
        Rest-frame SED. [erg/s/Hz]
    wave_rest : array_like, shape (n_wave,)
        Rest-frame wavelength grid. [Angstrom]
    wave_obs : array_like, shape (n_pix,)
        Observed wavelength grid. [Angstrom]
    z : float
        Redshift. [dimensionless]
    dl_cm : float
        Luminosity distance. [cm]

    Returns
    -------
    ndarray, shape (n_pix,)
        Spectral flux density. [erg/s/cm^2/Hz]

    Notes
    -----
    **JIT-compatible**: no — uses log-linear interpolation and cosmological scaling.
    """
    from tengri.observation.spectrum import compute_spectrum

    return compute_spectrum(rest_sed, wave_rest, wave_obs, z, dl_cm)


# ── Fused Tier 2 end-to-end kernels (params → photometry/spectrum)


def build_fused_tier2_photometry(state: SEDModelState, model=None, rest_sed_kernel=None):
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
    state : SEDModelState
        Frozen state bundle providing config and precomputed arrays.
    model : SEDModel, optional
        Legacy model reference for metallicity interpolation functions.
        If None, interpolation will fail; this parameter is temporary
        pending pipeline.py refactoring.
    rest_sed_kernel : callable, optional
        Pre-built rest-frame SED kernel (from build_fused_rest_sed).
        If None, built from state.

    Returns
    -------
    callable or None
        JIT-compiled function: ``(sfr_on_ssp, params_dict) -> photometry_array``,
        where ``sfr_on_ssp`` has shape (n_age,) [Msun/yr] and ``photometry_array``
        has shape (n_filters,) [erg/s/cm^2/Hz]. Returns None if no filters or
        Tier 2 kernel is available.

    Notes
    -----
    **JIT-compatible**: yes — entire pipeline (parameter translation, metallicity
    interpolation, compositional SED, filter integration) fused into one
    ``@jax.jit`` scope. SFH evaluation remains outside JIT (caller-computed).

    **Gradient-safe**: yes — differentiable w.r.t. all parameters and sfr_on_ssp.
    """
    if rest_sed_kernel is None:
        rest_sed_kernel = build_fused_rest_sed(state, model)
    if rest_sed_kernel is None:
        return None
    if state.filter_waves is None:
        return None

    from tengri.components.stellar.sps.dsps_wrapper import compute_csp_weights
    from tengri.forward.pipeline import interp_met_alpha_dispatch, interp_metallicity
    from tengri.observation.photometry import (
        compute_flux_density_batch,
        pad_filters,
    )
    from tengri.parameters.translate import get_internal_params

    _use_dsps_native = state.csp_integration == "dsps_native"
    if _use_dsps_native:
        from tengri.components.stellar.sps.dsps_wrapper import compute_dsps_native_weights

    # effective_metallicity correction is opt-in (see fused_kernels tier1 note).
    _use_alpha_fe_t2 = state.spec.alpha_fe_evolving or "met_alpha_fe" in state.spec.free_params

    # Capture state at build time
    param_map = state.param_map
    spec = state.spec
    has_field = state.uses_stochastic_sfh
    ssp_ages_yr = state.ssp_ages_yr
    # Metallicity mode: "delta" (scalar), "ramp" (evolving), or "chem_evol"
    _met_mode = state.met_mode
    # Panchromatic wavelength grid (extended if radio/xray enabled)
    rest_wave = state.rest_wavelength

    filter_waves = state.filter_waves
    filter_trans = state.filter_trans
    apply_igm = state.uses_igm
    # Pad filters for vectorized integration
    fw_padded_t2, ft_padded_t2, _filt_nv_t2 = pad_filters(filter_waves, filter_trans)

    # dsps_native: capture SSP arrays for DSPS triweight kernel
    if _use_dsps_native:
        _ssp_lgmet = state.ssp_data.ssp_lgmet
        _ssp_lg_age_gyr = state.ssp_data.ssp_lg_age_gyr
        _ssp_flux = state.ssp_data.ssp_flux
        _lgmet_scatter_native = float(state.lgmet_scatter)
        from tengri.utils.cosmology import age_at_z as _age_at_z_fn

    # BUG-NSS-02: For ramp mode (evolving metallicity), capture SSP age grid
    # to compute per-age metallicity evolution
    if _met_mode == "ramp":
        if not _use_dsps_native:
            _ssp_lg_age_gyr = state.ssp_data.ssp_lg_age_gyr
        from tengri.utils.cosmology import age_at_z as _age_at_z_fn

    # Redshift: fixed (precompute IGM once) or free (traced through)
    z_fixed = state.z_fixed
    dl_cm_fixed = state.dl_cm_fixed
    is_free_z = z_fixed is None

    # For dsps_native + fixed z: precompute t_obs_gyr once at closure build
    _t_obs_gyr_fixed = None
    if _use_dsps_native and not is_free_z:
        _t_obs_gyr_fixed = float(_age_at_z_fn(z_fixed))

    # Resolve IGM function once at factory build time based on state config.
    _igm_fn = state.igm_fn
    if _igm_fn is None:
        from tengri.components.igm import igm_transmission as _igm_fn

    # IGM at full wavelength grid (only for fixed z)
    # Use panchromatic grid if available (when radio/xray enabled), else SSP grid
    igm_trans_full = None
    if apply_igm and not is_free_z:
        wave_obs_full = rest_wave * (1.0 + z_fixed)
        igm_trans_full = _igm_fn(wave_obs_full, z_fixed)

    # For free-z: need luminosity_distance inside JIT
    if is_free_z:
        from tengri.utils.cosmology import luminosity_distance as _lum_dist

    # Phase II-2 (focused): SSP arrays passed as JIT-traced inputs rather than
    # captured by closure. Without this, the simple non-evolving-Z trapz path
    # below would close over `model.ssp_data.ssp_flux` (~114 MB at MIST grid
    # size) and XLA's constant-folder would bake transposes of it into the
    # compiled HLO, blowing past the 2 GB protobuf serialization limit. With
    # `ssp_flux_traced` and `ssp_lgmet_traced` as JIT inputs, the array stays a
    # runtime tensor — closed paths below that still take `model` (alpha_fe,
    # ramp metallicity) are unchanged for now and remain on the closure-capture
    # path; those branches are not on the quickstart's photometry path. See
    # `docs/dev/quickstart_oom_diagnosis.md`.

    # Closure-captured fallbacks for SSP arrays (used when callers don't pass
    # traced inputs — e.g. legacy callsites and the non-_traceable predict_*
    # methods that go through the lazily-built logged_jit wrapper).
    _ssp_flux_closure = state.ssp_data.ssp_flux
    _ssp_lgmet_closure = state.ssp_data.ssp_lgmet
    _lgmet_scatter_closure = float(state.lgmet_scatter)
    from tengri.components.stellar.sps.dsps_wrapper import (
        interpolate_metallicity_smooth as _interp_metallicity_smooth,
    )

    _met_use_smooth = state.met_interp == "smooth"

    # --- Shared SED computation (sfr_on_ssp pre-computed by caller) ---
    def _compute_rest_sed(sfr_on_ssp, params, ssp_flux_traced=None, ssp_lgmet_traced=None):
        """sfr_on_ssp, params → (rest_sed, redshift_value).

        ``sfr_on_ssp`` is the SFH already evaluated on the SSP age grid.
        Keeping SFH outside this function prevents the SFH type from entering
        the JIT closure, so switching SFH models does not cause recompilation.

        When ``ssp_flux_traced`` and ``ssp_lgmet_traced`` are provided
        (Phase II-2 trace path), the metallicity-interpolation step uses
        these as JIT-traced inputs rather than the closure-captured SSP
        arrays — which keeps XLA from baking the 114 MB SSP flux grid
        into the compiled HLO.
        """
        p = get_internal_params(params, param_map, spec, has_field)

        if _use_dsps_native:
            z = p.get("redshift", z_fixed if z_fixed is not None else 0.0)
            t_obs_gyr = _t_obs_gyr_fixed if _t_obs_gyr_fixed is not None else _age_at_z_fn(z)
            # BUG-NSS-02 fix: For ramp mode, compute per-age metallicity from initial/final
            if _met_mode == "ramp":
                from tengri.components.stellar.sps.dsps_wrapper import compute_log_z_evolving

                lgmet_per_age = compute_log_z_evolving(
                    _ssp_lg_age_gyr, p["log_z_abs_initial"], p["log_z_abs_final"], t_obs_gyr
                )
                # For dsps_native with ramp, pass the per-age array
                # Note: compute_dsps_native_weights expects scalar lgmet, so this path
                # is not fully supported yet. Fall back to scalar (use final value).
                # TODO: extend compute_dsps_native_weights to handle per-age metallicity
                lgmet = jnp.mean(lgmet_per_age)  # Approximate with mean
            else:
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
            # BUG-NSS-02 fix: For ramp mode, compute per-age metallicity and vmap interpolation
            if _met_mode == "ramp":
                from tengri.components.stellar.sps.dsps_wrapper import compute_log_z_evolving
                from tengri.forward.pipeline import (
                    interp_met_alpha_evolving_dispatch,
                    interp_metallicity_evolving,
                )

                z = p.get("redshift", z_fixed if z_fixed is not None else 0.0)
                t_obs_gyr = _t_obs_gyr_fixed if _t_obs_gyr_fixed is not None else _age_at_z_fn(z)
                lgmet_per_age = compute_log_z_evolving(
                    _ssp_lg_age_gyr, p["log_z_abs_initial"], p["log_z_abs_final"], t_obs_gyr
                )
                if _use_alpha_fe_t2:
                    alpha_fe = p.get("alpha_fe", 0.0)
                    ssp_flux_at_z = interp_met_alpha_evolving_dispatch(
                        model,
                        lgmet_per_age,
                        alpha_fe,
                        ssp_flux=ssp_flux_traced,
                        ssp_lgmet=ssp_lgmet_traced,
                    )
                else:
                    ssp_flux_at_z = interp_metallicity_evolving(
                        model,
                        lgmet_per_age,
                        ssp_flux=ssp_flux_traced,
                        ssp_lgmet=ssp_lgmet_traced,
                    )
            else:
                _lgmet = p.get("log_z_abs", -1.8477)
                if _use_alpha_fe_t2:
                    alpha_fe = p.get("alpha_fe", 0.0)
                    ssp_flux_at_z = interp_met_alpha_dispatch(
                        model,
                        _lgmet,
                        alpha_fe,
                        ssp_flux=ssp_flux_traced,
                        ssp_lgmet=ssp_lgmet_traced,
                    )
                else:
                    # Phase II-2 trace path: prefer traced SSP arrays when
                    # provided, fall back to closure-captured otherwise.
                    if (
                        _met_use_smooth
                        and ssp_flux_traced is not None
                        and ssp_lgmet_traced is not None
                    ):
                        ssp_flux_at_z = _interp_metallicity_smooth(
                            ssp_flux_traced,
                            ssp_lgmet_traced,
                            _lgmet,
                            _lgmet_scatter_closure,
                        )
                    else:
                        ssp_flux_at_z = interp_metallicity(model, _lgmet)

        # Always pass current SFR — needed by nebular (Q_H scaling) and X-ray.
        p = {**p, "_sfr_current": sfr_on_ssp[-1]}

        rest_sed = rest_sed_kernel(weights, ssp_flux_at_z, p)
        z = p.get("redshift", z_fixed if z_fixed is not None else 0.0)
        return rest_sed, z

    if is_free_z:

        def fused_tier2_phot(sfr_on_ssp, params, ssp_flux_traced=None, ssp_lgmet_traced=None):
            """sfr_on_ssp, params dict → observed photometry (free z)."""
            rest_sed, z = _compute_rest_sed(sfr_on_ssp, params, ssp_flux_traced, ssp_lgmet_traced)
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

        def fused_tier2_phot(sfr_on_ssp, params, ssp_flux_traced=None, ssp_lgmet_traced=None):
            """sfr_on_ssp, params dict → observed photometry (fixed z)."""
            rest_sed, _z = _compute_rest_sed(sfr_on_ssp, params, ssp_flux_traced, ssp_lgmet_traced)

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


def build_fused_tier2_spectrum(state: SEDModelState, model=None, rest_sed_kernel=None):
    """Build a single JIT function: (sfr_on_ssp, params dict) → observed spectrum.

    Like :func:`build_fused_tier2_photometry` but for spectroscopy.
    SFH is evaluated *outside* the JIT; the caller passes ``sfr_on_ssp``
    as a traced array so the JIT closure is SFH-type-independent.

    Requires ``precompute_spectroscopy()`` to have been called (for
    the wavelength grid) or an Observation with spectroscopy config.

    Parameters
    ----------
    state : SEDModelState
        Frozen state bundle providing config and precomputed arrays.
    model : SEDModel, optional
        Legacy model reference for metallicity interpolation functions.
        If None, interpolation will fail; this parameter is temporary
        pending pipeline.py refactoring.
    rest_sed_kernel : callable, optional
        Pre-built rest-frame SED kernel (from build_fused_rest_sed).
        If None, built from state.

    Returns
    -------
    callable or None
        JIT-compiled function: ``(sfr_on_ssp, params_dict) -> spectrum_array``,
        where ``sfr_on_ssp`` has shape (n_age,) [Msun/yr] and ``spectrum_array``
        has shape (n_pix,) [erg/s/cm^2/Hz]. Returns None if no Tier 2 kernel
        or wavelength grid is available.

    Notes
    -----
    **JIT-compatible**: yes — entire pipeline fused into one ``@jax.jit`` scope.
    Requires precomputed wavelength grid from ``precompute_spectroscopy()`` or
    Observation config.

    **Gradient-safe**: yes — differentiable w.r.t. all parameters and sfr_on_ssp.
    """
    if rest_sed_kernel is None:
        rest_sed_kernel = build_fused_rest_sed(state, model)
    if rest_sed_kernel is None:
        return None

    from tengri.components.stellar.sps.dsps_wrapper import compute_csp_weights
    from tengri.forward.pipeline import interp_met_alpha_dispatch, interp_metallicity
    from tengri.observation.spectrum import compute_spectrum
    from tengri.parameters.translate import get_internal_params

    _use_dsps_native_spec = state.csp_integration == "dsps_native"
    if _use_dsps_native_spec:
        from tengri.components.stellar.sps.dsps_wrapper import compute_dsps_native_weights

    # effective_metallicity correction is opt-in: only applied when the user
    # explicitly makes met_alpha_fe (or evolving variant) a free parameter.
    _use_alpha_fe_spec2 = state.spec.alpha_fe_evolving or "met_alpha_fe" in state.spec.free_params

    # Must have a wavelength grid
    wave_obs = None
    if state.precomputed.spectroscopy is not None:
        wave_obs = state.precomputed.spectroscopy.wave_obs_pixels
    elif state.wave_obs is not None:
        wave_obs = state.wave_obs
    if wave_obs is None:
        return None

    # Capture state
    param_map = state.param_map
    spec = state.spec
    has_field = state.uses_stochastic_sfh
    ssp_ages_yr = state.ssp_ages_yr
    # Panchromatic wavelength grid (extended if radio/xray enabled)
    rest_wave = state.rest_wavelength
    # Metallicity mode: "delta" (scalar), "ramp" (evolving), or "chem_evol"
    _met_mode_spec = state.met_mode

    if _use_dsps_native_spec:
        _ssp_lgmet_spec = state.ssp_data.ssp_lgmet
        _ssp_lg_age_gyr_spec = state.ssp_data.ssp_lg_age_gyr
        _ssp_flux_spec = state.ssp_data.ssp_flux
        _lgmet_scatter_spec = float(state.lgmet_scatter)
        from tengri.utils.cosmology import age_at_z as _age_at_z_spec

    # BUG-NSS-02: For ramp mode (evolving metallicity), capture SSP age grid
    # to compute per-age metallicity evolution
    if _met_mode_spec == "ramp":
        if not _use_dsps_native_spec:
            _ssp_lg_age_gyr_spec = state.ssp_data.ssp_lg_age_gyr
        from tengri.utils.cosmology import age_at_z as _age_at_z_spec

    z_fixed = state.z_fixed
    dl_cm_fixed = state.dl_cm_fixed
    is_free_z = z_fixed is None

    _t_obs_gyr_fixed_spec = None
    if _use_dsps_native_spec and not is_free_z:
        _t_obs_gyr_fixed_spec = float(_age_at_z_spec(z_fixed))

    if is_free_z:
        from tengri.utils.cosmology import luminosity_distance as _lum_dist_spec

    # Phase II-2: closure fallbacks for SSP arrays so legacy callers
    # (logged_jit wrapper, batch fitter) still work when traced kwargs
    # are absent.
    _ssp_flux_closure_spec = state.ssp_data.ssp_flux
    _ssp_lgmet_closure_spec = state.ssp_data.ssp_lgmet
    _lgmet_scatter_closure_spec = float(state.lgmet_scatter)
    _met_use_smooth_spec = state.met_interp == "smooth"
    from tengri.components.stellar.sps.dsps_wrapper import (
        interpolate_metallicity_smooth as _interp_metallicity_smooth_spec,
    )

    # Shared SED computation (sfr_on_ssp pre-computed by caller)
    def _compute_rest_sed_spec(sfr_on_ssp, params, ssp_flux_traced=None, ssp_lgmet_traced=None):
        """Compute rest-frame SED for the spectroscopy kernel given pre-computed SFR weights.

        ``ssp_flux_traced`` / ``ssp_lgmet_traced`` (Phase II-2 trace path):
        when provided, the smooth-Z metallicity-interpolation step uses
        them as JIT-traced inputs instead of closure-captured SSP arrays.
        Keeps XLA from baking the 114 MB SSP grid into the compiled HLO.
        """
        p = get_internal_params(params, param_map, spec, has_field)
        if _use_dsps_native_spec:
            z = p.get("redshift", z_fixed if z_fixed is not None else 0.0)
            t_obs_gyr = (
                _t_obs_gyr_fixed_spec if _t_obs_gyr_fixed_spec is not None else _age_at_z_spec(z)
            )
            # BUG-NSS-02 fix: For ramp mode, compute per-age metallicity from initial/final
            if _met_mode_spec == "ramp":
                from tengri.components.stellar.sps.dsps_wrapper import compute_log_z_evolving

                lgmet_per_age = compute_log_z_evolving(
                    _ssp_lg_age_gyr_spec,
                    p["log_z_abs_initial"],
                    p["log_z_abs_final"],
                    t_obs_gyr,
                )
                # For dsps_native with ramp, approximate with mean
                # TODO: extend compute_dsps_native_weights to handle per-age metallicity
                lgmet = jnp.mean(lgmet_per_age)
            else:
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
            # BUG-NSS-02 fix: For ramp mode, compute per-age metallicity and vmap interpolation
            if _met_mode_spec == "ramp":
                from tengri.components.stellar.sps.dsps_wrapper import compute_log_z_evolving
                from tengri.forward.pipeline import (
                    interp_met_alpha_evolving_dispatch,
                    interp_metallicity_evolving,
                )

                z = p.get("redshift", z_fixed if z_fixed is not None else 0.0)
                t_obs_gyr = (
                    _t_obs_gyr_fixed_spec
                    if _t_obs_gyr_fixed_spec is not None
                    else _age_at_z_spec(z)
                )
                lgmet_per_age = compute_log_z_evolving(
                    _ssp_lg_age_gyr_spec,
                    p["log_z_abs_initial"],
                    p["log_z_abs_final"],
                    t_obs_gyr,
                )
                if _use_alpha_fe_spec2:
                    alpha_fe = p.get("alpha_fe", 0.0)
                    ssp_flux_at_z = interp_met_alpha_evolving_dispatch(
                        model,
                        lgmet_per_age,
                        alpha_fe,
                        ssp_flux=ssp_flux_traced,
                        ssp_lgmet=ssp_lgmet_traced,
                    )
                else:
                    ssp_flux_at_z = interp_metallicity_evolving(
                        model,
                        lgmet_per_age,
                        ssp_flux=ssp_flux_traced,
                        ssp_lgmet=ssp_lgmet_traced,
                    )
            else:
                if _use_alpha_fe_spec2:
                    alpha_fe = p.get("alpha_fe", 0.0)
                    _lgmet_s = p.get("log_z_abs", -1.8477)
                    ssp_flux_at_z = interp_met_alpha_dispatch(
                        model,
                        _lgmet_s,
                        alpha_fe,
                        ssp_flux=ssp_flux_traced,
                        ssp_lgmet=ssp_lgmet_traced,
                    )
                else:
                    _lgmet_s = p.get("log_z_abs", -1.8477)
                    if (
                        _met_use_smooth_spec
                        and ssp_flux_traced is not None
                        and ssp_lgmet_traced is not None
                    ):
                        # Phase II-2 trace path: SSP arrays as JIT inputs
                        ssp_flux_at_z = _interp_metallicity_smooth_spec(
                            ssp_flux_traced,
                            ssp_lgmet_traced,
                            _lgmet_s,
                            _lgmet_scatter_closure_spec,
                        )
                    else:
                        ssp_flux_at_z = interp_metallicity(model, _lgmet_s)
        p = {**p, "_sfr_current": sfr_on_ssp[-1]}
        rest_sed = rest_sed_kernel(weights, ssp_flux_at_z, p)
        z = p.get("redshift", z_fixed if z_fixed is not None else 0.0)
        return rest_sed, z

    if is_free_z:

        def fused_tier2_spec(sfr_on_ssp, params, ssp_flux_traced=None, ssp_lgmet_traced=None):
            """sfr_on_ssp, params dict → observed spectrum (free z)."""
            rest_sed, z = _compute_rest_sed_spec(
                sfr_on_ssp, params, ssp_flux_traced, ssp_lgmet_traced
            )
            dl_cm = _lum_dist_spec(z)
            return compute_spectrum(rest_sed, rest_wave, wave_obs, z, dl_cm)

    else:

        def fused_tier2_spec(sfr_on_ssp, params, ssp_flux_traced=None, ssp_lgmet_traced=None):
            """sfr_on_ssp, params dict → observed spectrum (fixed z)."""
            rest_sed, _z = _compute_rest_sed_spec(
                sfr_on_ssp, params, ssp_flux_traced, ssp_lgmet_traced
            )
            return compute_spectrum(rest_sed, rest_wave, wave_obs, z_fixed, dl_cm_fixed)

    return fused_tier2_spec


def build_hybrid_spectrum(state: SEDModelState, model=None):
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
    state : SEDModelState
        Frozen state bundle providing config and precomputed arrays.
    model : SEDModel, optional
        Legacy model reference for metallicity interpolation functions.
        If None, interpolation will fail; this parameter is temporary
        pending pipeline.py refactoring.

    Returns
    -------
    callable or None
        JIT-compiled function: ``(sfr_on_ssp, params) -> spectrum_array``,
        where ``sfr_on_ssp`` has shape (n_age,) [Msun/yr] and
        ``spectrum_array`` has shape (n_pix,) [erg/s/cm^2/Hz].
        Returns None if spectroscopy is not precomputed.

    Notes
    -----
    **JIT-compatible**: yes — fuses precomputed stellar interpolation (fast, on
    pixel grid) with exact non-stellar evaluation (full wavelength, then
    interpolated to pixels). This balances speed and accuracy for science models.

    **Gradient-safe**: yes — differentiable w.r.t. all parameters and sfr_on_ssp.
    """
    if state.precomputed.spectroscopy is None:
        return None

    from tengri.components.dust.attenuation import resolve_dust_law
    from tengri.components.stellar.sps.dsps_wrapper import compute_csp_weights
    from tengri.forward.pipeline import (
        interp_met_alpha_dispatch,
        interp_metallicity,
    )
    from tengri.parameters.translate import get_internal_params
    from tengri.utils.conversions import lnu_to_fnu

    # Precomputed spectroscopic data
    precomp_spec = state.precomputed.spectroscopy
    ssp_on_pixels = precomp_spec.ssp_on_pixels.astype(state.forward_dtype)
    wave_rest_pixels = precomp_spec.wave_rest_pixels
    z_fixed = state.z_fixed
    dl_cm_fixed = state.dl_cm_fixed

    # Model configuration
    ssp_ages_yr = state.ssp_ages_yr
    _is_single_dust = state.dust_model == "single_component"
    law_bc_fn = resolve_dust_law(state.dust_law_bc)
    if not _is_single_dust:
        law_diff_fn = resolve_dust_law(state.dust_law_diff)

    # Alpha enhancement
    _use_alpha_fe = state.spec.alpha_fe_evolving or "met_alpha_fe" in state.spec.free_params
    if _use_alpha_fe:
        from tengri.components.stellar.sps.dsps_wrapper import has_alpha_grid

        _has_alpha = has_alpha_grid(state.ssp_data)
    else:
        _has_alpha = False

    # Non-stellar kernel (build if not provided)
    if (
        model is not None
        and hasattr(model, "_compositional")
        and model._compositional.rest_sed is not None
    ):
        rest_sed_kernel = model._compositional.rest_sed
    else:
        rest_sed_kernel = build_fused_rest_sed(state, model)
    param_map = state.param_map
    spec = state.spec
    has_field = state.uses_stochastic_sfh
    rest_wave = state.rest_wavelength

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
        ssp_flux_traced=None,
        ssp_lgmet_traced=None,
    ):
        """Compute hybrid spectrum: precomputed stellar + non-stellar.

        Phase II-2: when ``ssp_flux_traced`` and ``ssp_lgmet_traced`` are
        supplied, the metallicity interpolation step uses them as
        JIT-traced inputs rather than closure-captured constants — keeps
        XLA from baking the full SSP grid into the compiled HLO.
        """
        p = get_internal_params(params, param_map, spec, has_field)
        weights = compute_csp_weights(sfr_on_ssp, ssp_ages_yr)

        # Metallicity interpolation
        # BUG-NSS-02: Use the provided log_z_abs, don't silently fall back
        _lgmet_hs = log_z_abs
        if _use_alpha_fe and _has_alpha:
            alpha_fe_val = p.get("alpha_fe", 0.0)
            ssp_flux_at_z = interp_met_alpha_dispatch(
                model,
                _lgmet_hs,
                alpha_fe_val,
                ssp_flux=ssp_flux_traced,
                ssp_lgmet=ssp_lgmet_traced,
            )
        else:
            ssp_flux_at_z = interp_metallicity(
                model,
                _lgmet_hs,
                ssp_flux=ssp_flux_traced,
                ssp_lgmet=ssp_lgmet_traced,
            )

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
        if state.uses_xray:
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

    # BUG-NSS-02: For ramp mode, use mean metallicity (approximate) since
    # the hybrid kernel's precomputed stellar pixels are not per-age.
    # Ideally, hybrid kernel would not be used for evolving metallicity.
    if _is_single_dust:

        def hybrid_spec(sfr_on_ssp, params, ssp_flux_traced=None, ssp_lgmet_traced=None):
            """Single-dust hybrid spectrum."""
            p = get_internal_params(params, param_map, spec, has_field)
            if state.met_mode == "ramp":
                from tengri.components.stellar.sps.dsps_wrapper import compute_log_z_evolving
                from tengri.utils.cosmology import age_at_z

                z = p.get("redshift", z_fixed if z_fixed is not None else 0.0)
                t_obs_gyr = age_at_z(z)
                lgmet_per_age = compute_log_z_evolving(
                    state.ssp_data.ssp_lg_age_gyr,
                    p["log_z_abs_initial"],
                    p["log_z_abs_final"],
                    t_obs_gyr,
                )
                # Use mean metallicity as approximation for precomputed stellar grid
                _lgmet = jnp.mean(lgmet_per_age)
            else:
                _lgmet = p.get("log_z_abs", -1.8477)
            return _hybrid_spec_body(
                sfr_on_ssp,
                params,
                _lgmet,
                (p.get("tau_v", 0.0),),
                ssp_flux_traced=ssp_flux_traced,
                ssp_lgmet_traced=ssp_lgmet_traced,
            )

    else:

        def hybrid_spec(sfr_on_ssp, params, ssp_flux_traced=None, ssp_lgmet_traced=None):
            """Two-dust hybrid spectrum."""
            p = get_internal_params(params, param_map, spec, has_field)
            if state.met_mode == "ramp":
                from tengri.components.stellar.sps.dsps_wrapper import compute_log_z_evolving
                from tengri.utils.cosmology import age_at_z

                z = p.get("redshift", z_fixed if z_fixed is not None else 0.0)
                t_obs_gyr = age_at_z(z)
                lgmet_per_age = compute_log_z_evolving(
                    state.ssp_data.ssp_lg_age_gyr,
                    p["log_z_abs_initial"],
                    p["log_z_abs_final"],
                    t_obs_gyr,
                )
                # Use mean metallicity as approximation for precomputed stellar grid
                _lgmet = jnp.mean(lgmet_per_age)
            else:
                _lgmet = p.get("log_z_abs", -1.8477)
            return _hybrid_spec_body(
                sfr_on_ssp,
                params,
                _lgmet,
                (p.get("tau_bc", 0.0), p.get("tau_diff", 0.0)),
                ssp_flux_traced=ssp_flux_traced,
                ssp_lgmet_traced=ssp_lgmet_traced,
            )

    return hybrid_spec
