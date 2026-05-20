"""Fused JIT kernel builders for fast photometry and spectroscopy.

These factory functions build @jax.jit closures that capture precomputed
arrays (SSP grids, dust weights, effective wavelengths) at build time.
The returned functions take only per-call parameters (SFR weights, dust
params) as arguments.

Extracted from ``SEDModel`` methods to keep model.py focused on orchestration.
"""

from __future__ import annotations

import jax.numpy as jnp

from tengri.components.stellar.sfh.sfr_window import time_weighted_sfr
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
    from tengri.forward.pipeline import interp_met_alpha_dispatch
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

    # SSP arrays are passed as JIT-traced inputs rather than captured by
    # closure. Without this, the simple non-evolving-Z trapz path below
    # would close over `model.ssp_data.ssp_flux` (~114 MB at MIST grid
    # size) and XLA's constant-folder would bake transposes of it into
    # the compiled HLO, blowing past the 2 GB protobuf serialization
    # limit. With `ssp_flux_traced` and `ssp_lgmet_traced` as JIT inputs
    # the array stays a runtime tensor — closed paths below that still
    # take `model` (alpha_fe, ramp metallicity) are unchanged for now and
    # remain on the closure-capture path; those branches are not on the
    # quickstart's photometry path. See
    # `docs/dev/quickstart_oom_diagnosis.md`.

    # Closure-captured fallbacks for SSP arrays (used when callers don't pass
    # traced inputs — e.g. legacy callsites and the non-traced predict_*
    # methods that go through the lazily-built logged_jit wrapper).
    _ssp_flux_closure = state.ssp_data.ssp_flux
    _ssp_lgmet_closure = state.ssp_data.ssp_lgmet
    _lgmet_scatter_closure = float(state.lgmet_scatter)

    _met_use_smooth = state.met_interp == "smooth"

    # ── Closure-path-A captures (no-α / non-ramp / non-native default) ──
    # The exact path's ``compute_sed_components`` uses
    # ``calc_rest_sed_sfh_table_lognormal_mdf`` with trapezoidal cosmic-time
    # SFH integration on the orchestrator's 64-pt grid + lognormal-MDF
    # metallicity weighting. To match bit-exactly we replicate those
    # semantics here. SFH type re-enters the JIT closure: switching SFH
    # models triggers recompilation.
    # See ``forward/pipeline.py:_closure_a_sfh_prep`` for the canonical impl.
    _ssp_lg_age_gyr_ca = state.ssp_data.ssp_lg_age_gyr
    from tengri.components.stellar.sfh.gp_sfh import (
        make_log_age_grid as _make_log_age_grid_ca,
    )
    from tengri.utils.cosmology import age_at_z as _age_at_z_fn_ca

    # Read ``n_grid`` from spec to match orchestrator's
    # ``StellarSEDComponentConfig.n_grid``. Variable name keeps the legacy
    # ``_64`` suffix purely as a label; actual value is spec-driven.
    _orch_n_grid_ca = int(getattr(state.spec, "n_grid", 64))
    _sfh_lbt_grid_orch_64 = jnp.power(10.0, _make_log_age_grid_ca(_orch_n_grid_ca))
    _t_obs_gyr_fixed_ca = None if is_free_z else float(_age_at_z_fn_ca(z_fixed))
    _sfh_fn_ca = model._sfh_fn
    _sfh_internal_names_ca = model._sfh_internal_names

    # --- Shared SED computation (sfr_on_ssp pre-computed by caller) ---
    def _compute_rest_sed(
        sfr_on_ssp,
        params,
        ssp_flux_traced=None,
        ssp_lgmet_traced=None,
        sfr_internal=None,
    ):
        """sfr_on_ssp, params → (rest_sed, redshift_value).

        ``sfr_on_ssp`` is the SFH already evaluated on the SSP age grid.
        Keeping SFH outside this function prevents the SFH type from entering
        the JIT closure, so switching SFH models does not cause recompilation.

        When ``ssp_flux_traced`` and ``ssp_lgmet_traced`` are provided,
        the metallicity-interpolation step uses these as JIT-traced
        inputs rather than the closure-captured SSP arrays — which keeps
        XLA from baking the 114 MB SSP flux grid into the compiled HLO.
        """
        p = get_internal_params(params, param_map, spec, has_field, strict_unknown_params=False)

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
                    # CLOSURE-A: replace legacy ``compute_csp_weights`` (rectangle
                    # rule, set above at line ~313) + ``interp_metallicity``
                    # (single-Z bilinear, no MDF) with DSPS canonical
                    # ``calc_rest_sed_sfh_table_lognormal_mdf``: trapezoidal
                    # cosmic-time SFH integration on the orchestrator's 64-pt
                    # grid + lognormal-MDF metallicity weighting (σ=lgmet_scatter).
                    # Mirrors ``StellarSEDComponent.apply`` and
                    # ``pipeline.py``'s closure-A branch (line ~954-1010) so the
                    # compositional and exact paths produce bit-identical SEDs.
                    #
                    # Marginalisation back to the existing ``rest_sed_kernel``
                    # (1D age weights × 2D age,wave SSP) preserves the kernel
                    # signature: the joint ``einsum("ma,maw->w")`` decomposes
                    # into ``einsum("a,aw->w")`` after collapsing m via the
                    # per-age MDF average — so dust attenuation in the kernel
                    # (``trans_1d`` or two-CSP age-mask) still applies correctly.
                    from dsps.sed.stellar_sed import (
                        calc_rest_sed_sfh_table_lognormal_mdf,
                    )

                    _ssp_flux_use = (
                        ssp_flux_traced if ssp_flux_traced is not None else _ssp_flux_closure
                    )
                    _ssp_lgmet_use = (
                        ssp_lgmet_traced if ssp_lgmet_traced is not None else _ssp_lgmet_closure
                    )
                    lgmet_scatter_ca = float(p.get("lgmet_scatter", _lgmet_scatter_closure))

                    # Re-evaluate SFH on 64-pt orchestrator grid (closure-A
                    # canonical). For stochastic SFHs we prefer the caller-
                    # supplied ``sfr_internal`` (the GP draw on the model's
                    # internal grid) so we don't lose precision through a
                    # double interp via ``sfr_on_ssp``. Mirrors the helper
                    # ``_closure_a_sfh_prep``'s stochastic branch.
                    if has_field:
                        if sfr_internal is not None and sfr_internal.shape[0] == _orch_n_grid_ca:
                            # GP draw already on the orchestrator grid —
                            # reuse directly to preserve the realisation.
                            _sfr_orch_grid = sfr_internal
                        elif sfr_internal is not None:
                            # Defensive fallback: model's internal grid
                            # differs from orchestrator's spec.n_grid.
                            # Should not happen with the n_grid plumbing
                            # in place, but interp keeps the kernel safe.
                            _sfr_orch_grid = jnp.interp(
                                _sfh_lbt_grid_orch_64,
                                model.log_age_grid,
                                sfr_internal,
                            )
                        else:
                            _sfr_orch_grid = jnp.interp(
                                _sfh_lbt_grid_orch_64, ssp_ages_yr, sfr_on_ssp
                            )
                    else:
                        _sfh_kw = {k: v for k, v in p.items() if k in _sfh_internal_names_ca}
                        _sfr_orch_grid = _sfh_fn_ca(_sfh_lbt_grid_orch_64, **_sfh_kw)

                    _sfr_on_ssp_orch = jnp.interp(
                        ssp_ages_yr, _sfh_lbt_grid_orch_64, _sfr_orch_grid
                    )

                    # NaN-safe cosmic-time prep (mirrors _closure_a_sfh_prep).
                    _z_internal_ca = p.get("redshift", z_fixed if z_fixed is not None else 0.0)
                    _t_obs_gyr_ca = (
                        _t_obs_gyr_fixed_ca
                        if _t_obs_gyr_fixed_ca is not None
                        else _age_at_z_fn_ca(_z_internal_ca)
                    )
                    _T_TABLE_MIN = 0.01
                    _ssp_age_gyr = ssp_ages_yr / 1e9
                    _t_cosmic_raw = _t_obs_gyr_ca - _ssp_age_gyr
                    _n_ssp = ssp_ages_yr.shape[0]
                    _t_cosmic_floor = jnp.maximum(_t_cosmic_raw, _T_TABLE_MIN)
                    _valid = _t_cosmic_raw > 0.0
                    _t_cosmic_asc_raw = _t_cosmic_floor[::-1]
                    _sfr_asc_raw = _sfr_on_ssp_orch[::-1]
                    _n_invalid = jnp.sum(~_valid[::-1])
                    _idx_pos = jnp.arange(_n_ssp)
                    _is_invalid_pos = _idx_pos < _n_invalid
                    _ramp_ca = _T_TABLE_MIN + (_T_TABLE_MIN * 0.5) * (_idx_pos + 1) / jnp.maximum(
                        _n_invalid, 1
                    )
                    _t_cosmic_asc = jnp.where(_is_invalid_pos, _ramp_ca, _t_cosmic_asc_raw)
                    _sfr_asc = jnp.where(_is_invalid_pos, 0.0, _sfr_asc_raw)
                    _total_mass_ca = jnp.maximum(jnp.trapezoid(_sfr_asc, _t_cosmic_asc * 1e9), 0.0)

                    _dsps_result_ca = calc_rest_sed_sfh_table_lognormal_mdf(
                        gal_t_table=_t_cosmic_asc,
                        gal_sfr_table=_sfr_asc,
                        gal_lgmet=_lgmet,
                        gal_lgmet_scatter=lgmet_scatter_ca,
                        ssp_lgmet=_ssp_lgmet_use,
                        ssp_lg_age_gyr=_ssp_lg_age_gyr_ca,
                        ssp_flux=_ssp_flux_use,
                        t_obs=_t_obs_gyr_ca,
                    )
                    # weights_2d shape (n_met, n_age), ∑=1; scale to Msun.
                    _weights_2d_ca = _dsps_result_ca.weights * _total_mass_ca
                    # Marginalise to (1D age weights, 2D MDF-averaged SSP)
                    # so the existing rest_sed_kernel reduction
                    # ``einsum("i,iw->w", w, ssp_z)`` reproduces the joint
                    # ``einsum("ma,maw->w", weights_2d, ssp_flux)`` exactly.
                    weights = _weights_2d_ca.sum(axis=0)
                    _w_safe_ca = jnp.maximum(weights, 1e-30)
                    ssp_flux_at_z = jnp.einsum(
                        "ma,maw->aw",
                        _weights_2d_ca / _w_safe_ca[None, :],
                        _ssp_flux_use,
                    )

        # Always pass the canonical 10 Myr time-weighted SFR — needed by
        # nebular (Q_H scaling) and X-ray. Murphy+2011 timescale.
        p = {**p, "_sfr_current": time_weighted_sfr(sfr_on_ssp, ssp_ages_yr, 1e7)}

        rest_sed = rest_sed_kernel(weights, ssp_flux_at_z, p)
        z = p.get("redshift", z_fixed if z_fixed is not None else 0.0)
        return rest_sed, z

    if is_free_z:

        def fused_tier2_phot(
            sfr_on_ssp,
            params,
            ssp_flux_traced=None,
            ssp_lgmet_traced=None,
            sfr_internal=None,
        ):
            """sfr_on_ssp, params dict → observed photometry (free z)."""
            rest_sed, z = _compute_rest_sed(
                sfr_on_ssp,
                params,
                ssp_flux_traced,
                ssp_lgmet_traced,
                sfr_internal=sfr_internal,
            )
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

        def fused_tier2_phot(
            sfr_on_ssp,
            params,
            ssp_flux_traced=None,
            ssp_lgmet_traced=None,
            sfr_internal=None,
        ):
            """sfr_on_ssp, params dict → observed photometry (fixed z)."""
            rest_sed, _z = _compute_rest_sed(
                sfr_on_ssp,
                params,
                ssp_flux_traced,
                ssp_lgmet_traced,
                sfr_internal=sfr_internal,
            )

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
    from tengri.forward.pipeline import interp_met_alpha_dispatch
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

    # Closure fallbacks for SSP arrays so legacy callers (logged_jit
    # wrapper, batch fitter) still work when traced kwargs are absent.
    _ssp_flux_closure_spec = state.ssp_data.ssp_flux
    _ssp_lgmet_closure_spec = state.ssp_data.ssp_lgmet
    _lgmet_scatter_closure_spec = float(state.lgmet_scatter)
    _met_use_smooth_spec = state.met_interp == "smooth"

    # Closure-path-A captures for spectrum kernel (mirrors photometry kernel).
    _ssp_lg_age_gyr_ca_spec = state.ssp_data.ssp_lg_age_gyr
    from tengri.components.stellar.sfh.gp_sfh import (
        make_log_age_grid as _make_log_age_grid_ca_spec,
    )
    from tengri.utils.cosmology import age_at_z as _age_at_z_fn_ca_spec

    _orch_n_grid_ca_spec = int(getattr(state.spec, "n_grid", 64))
    _sfh_lbt_grid_orch_64_spec = jnp.power(10.0, _make_log_age_grid_ca_spec(_orch_n_grid_ca_spec))
    _t_obs_gyr_fixed_ca_spec = None if is_free_z else float(_age_at_z_fn_ca_spec(z_fixed))
    _sfh_fn_ca_spec = model._sfh_fn
    _sfh_internal_names_ca_spec = model._sfh_internal_names

    # Shared SED computation (sfr_on_ssp pre-computed by caller)
    def _compute_rest_sed_spec(
        sfr_on_ssp,
        params,
        ssp_flux_traced=None,
        ssp_lgmet_traced=None,
        sfr_internal=None,
    ):
        """Compute rest-frame SED for the spectroscopy kernel given pre-computed SFR weights.

        When ``ssp_flux_traced`` / ``ssp_lgmet_traced`` are provided,
        the smooth-Z metallicity-interpolation step uses them as
        JIT-traced inputs instead of closure-captured SSP arrays —
        keeps XLA from baking the 114 MB SSP grid into the compiled HLO.
        """
        p = get_internal_params(params, param_map, spec, has_field, strict_unknown_params=False)
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
                    # CLOSURE-A spectrum: same migration as photometry kernel
                    # (see ``build_fused_tier2_photometry`` for the equivalence
                    # proof). Mirrors ``StellarSEDComponent.apply`` /
                    # ``pipeline.py``'s closure-A branch so compositional
                    # spectrum is bit-identical to the exact path.
                    from dsps.sed.stellar_sed import (
                        calc_rest_sed_sfh_table_lognormal_mdf as _calc_rest_sed_ln_spec,
                    )

                    _lgmet_s = p.get("log_z_abs", -1.8477)
                    _ssp_flux_use_spec = (
                        ssp_flux_traced if ssp_flux_traced is not None else _ssp_flux_closure_spec
                    )
                    _ssp_lgmet_use_spec = (
                        ssp_lgmet_traced
                        if ssp_lgmet_traced is not None
                        else _ssp_lgmet_closure_spec
                    )
                    lgmet_scatter_ca_spec = float(
                        p.get("lgmet_scatter", _lgmet_scatter_closure_spec)
                    )

                    if has_field:
                        if (
                            sfr_internal is not None
                            and sfr_internal.shape[0] == _orch_n_grid_ca_spec
                        ):
                            _sfr_orch_grid_spec = sfr_internal
                        elif sfr_internal is not None:
                            _sfr_orch_grid_spec = jnp.interp(
                                _sfh_lbt_grid_orch_64_spec,
                                model.log_age_grid,
                                sfr_internal,
                            )
                        else:
                            _sfr_orch_grid_spec = jnp.interp(
                                _sfh_lbt_grid_orch_64_spec, ssp_ages_yr, sfr_on_ssp
                            )
                    else:
                        _sfh_kw_spec = {
                            k: v for k, v in p.items() if k in _sfh_internal_names_ca_spec
                        }
                        _sfr_orch_grid_spec = _sfh_fn_ca_spec(
                            _sfh_lbt_grid_orch_64_spec, **_sfh_kw_spec
                        )
                    _sfr_on_ssp_orch_spec = jnp.interp(
                        ssp_ages_yr, _sfh_lbt_grid_orch_64_spec, _sfr_orch_grid_spec
                    )

                    _z_internal_ca_spec = p.get(
                        "redshift", z_fixed if z_fixed is not None else 0.0
                    )
                    _t_obs_gyr_ca_spec = (
                        _t_obs_gyr_fixed_ca_spec
                        if _t_obs_gyr_fixed_ca_spec is not None
                        else _age_at_z_fn_ca_spec(_z_internal_ca_spec)
                    )
                    _T_TABLE_MIN_spec = 0.01
                    _ssp_age_gyr_spec = ssp_ages_yr / 1e9
                    _t_cosmic_raw_spec = _t_obs_gyr_ca_spec - _ssp_age_gyr_spec
                    _n_ssp_spec = ssp_ages_yr.shape[0]
                    _t_cosmic_floor_spec = jnp.maximum(_t_cosmic_raw_spec, _T_TABLE_MIN_spec)
                    _valid_spec = _t_cosmic_raw_spec > 0.0
                    _t_cosmic_asc_raw_spec = _t_cosmic_floor_spec[::-1]
                    _sfr_asc_raw_spec = _sfr_on_ssp_orch_spec[::-1]
                    _n_invalid_spec = jnp.sum(~_valid_spec[::-1])
                    _idx_pos_spec = jnp.arange(_n_ssp_spec)
                    _is_invalid_pos_spec = _idx_pos_spec < _n_invalid_spec
                    _ramp_ca_spec = _T_TABLE_MIN_spec + (_T_TABLE_MIN_spec * 0.5) * (
                        _idx_pos_spec + 1
                    ) / jnp.maximum(_n_invalid_spec, 1)
                    _t_cosmic_asc_spec = jnp.where(
                        _is_invalid_pos_spec, _ramp_ca_spec, _t_cosmic_asc_raw_spec
                    )
                    _sfr_asc_spec = jnp.where(_is_invalid_pos_spec, 0.0, _sfr_asc_raw_spec)
                    _total_mass_ca_spec = jnp.maximum(
                        jnp.trapezoid(_sfr_asc_spec, _t_cosmic_asc_spec * 1e9), 0.0
                    )

                    _dsps_result_ca_spec = _calc_rest_sed_ln_spec(
                        gal_t_table=_t_cosmic_asc_spec,
                        gal_sfr_table=_sfr_asc_spec,
                        gal_lgmet=_lgmet_s,
                        gal_lgmet_scatter=lgmet_scatter_ca_spec,
                        ssp_lgmet=_ssp_lgmet_use_spec,
                        ssp_lg_age_gyr=_ssp_lg_age_gyr_ca_spec,
                        ssp_flux=_ssp_flux_use_spec,
                        t_obs=_t_obs_gyr_ca_spec,
                    )
                    _weights_2d_ca_spec = _dsps_result_ca_spec.weights * _total_mass_ca_spec
                    weights = _weights_2d_ca_spec.sum(axis=0)
                    _w_safe_ca_spec = jnp.maximum(weights, 1e-30)
                    ssp_flux_at_z = jnp.einsum(
                        "ma,maw->aw",
                        _weights_2d_ca_spec / _w_safe_ca_spec[None, :],
                        _ssp_flux_use_spec,
                    )
        # Canonical 10 Myr time-weighted SFR (Murphy+2011 timescale).
        p = {**p, "_sfr_current": time_weighted_sfr(sfr_on_ssp, ssp_ages_yr, 1e7)}
        rest_sed = rest_sed_kernel(weights, ssp_flux_at_z, p)
        z = p.get("redshift", z_fixed if z_fixed is not None else 0.0)
        return rest_sed, z

    if is_free_z:

        def fused_tier2_spec(
            sfr_on_ssp,
            params,
            ssp_flux_traced=None,
            ssp_lgmet_traced=None,
            sfr_internal=None,
        ):
            """sfr_on_ssp, params dict → observed spectrum (free z)."""
            rest_sed, z = _compute_rest_sed_spec(
                sfr_on_ssp,
                params,
                ssp_flux_traced,
                ssp_lgmet_traced,
                sfr_internal=sfr_internal,
            )
            dl_cm = _lum_dist_spec(z)
            return compute_spectrum(rest_sed, rest_wave, wave_obs, z, dl_cm)

    else:

        def fused_tier2_spec(
            sfr_on_ssp,
            params,
            ssp_flux_traced=None,
            ssp_lgmet_traced=None,
            sfr_internal=None,
        ):
            """sfr_on_ssp, params dict → observed spectrum (fixed z)."""
            rest_sed, _z = _compute_rest_sed_spec(
                sfr_on_ssp,
                params,
                ssp_flux_traced,
                ssp_lgmet_traced,
                sfr_internal=sfr_internal,
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

    # ── Closure-path-A captures (mirrors photometry/spectrum kernels) ──
    _ssp_lg_age_gyr_ca_hs = state.ssp_data.ssp_lg_age_gyr
    _ssp_flux_ca_hs = state.ssp_data.ssp_flux
    _ssp_lgmet_ca_hs = state.ssp_data.ssp_lgmet
    _lgmet_scatter_ca_hs = float(state.lgmet_scatter)
    from tengri.components.stellar.sfh.gp_sfh import (
        make_log_age_grid as _make_log_age_grid_ca_hs,
    )
    from tengri.utils.cosmology import age_at_z as _age_at_z_fn_ca_hs

    _orch_n_grid_ca_hs = int(getattr(state.spec, "n_grid", 64))
    _sfh_lbt_grid_orch_64_hs = jnp.power(10.0, _make_log_age_grid_ca_hs(_orch_n_grid_ca_hs))
    _t_obs_gyr_fixed_ca_hs = None if is_free_z else float(_age_at_z_fn_ca_hs(z_fixed))
    _sfh_fn_ca_hs = model._sfh_fn if model is not None else None
    _sfh_internal_names_ca_hs = model._sfh_internal_names if model is not None else set()
    _has_field_ca_hs = state.uses_stochastic_sfh
    _model_log_age_grid_ca_hs = model.log_age_grid if model is not None else None
    _closure_a_eligible_hs = not _use_alpha_fe and not _has_alpha and _sfh_fn_ca_hs is not None
    if _closure_a_eligible_hs:
        from dsps.sed.stellar_sed import (
            calc_rest_sed_sfh_table_lognormal_mdf as _calc_rest_sed_ln_hs,
        )

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

        When ``ssp_flux_traced`` and ``ssp_lgmet_traced`` are supplied,
        the metallicity interpolation step uses them as JIT-traced
        inputs rather than closure-captured constants — keeps XLA from
        baking the full SSP grid into the compiled HLO.
        """
        p = get_internal_params(params, param_map, spec, has_field, strict_unknown_params=False)

        # ── Closure-path-A migration ──────────────────────────────────
        # Replace legacy ``compute_csp_weights`` + ``interp_metallicity``
        # with DSPS lognormal-MDF + trapezoidal cosmic-time SFH integration
        # so the hybrid spectrum kernel agrees with the exact path. Also
        # produces an MDF-marginalised stellar pixel spectrum, fixing the
        # pre-existing line that selected ``ssp_on_pixels[0]`` (always
        # zeroth metallicity index) regardless of the requested ``log_z``.
        _lgmet_hs = log_z_abs
        _closure_a_runtime_hs = _closure_a_eligible_hs and _t_obs_gyr_fixed_ca_hs is not None
        if _closure_a_runtime_hs and _has_field_ca_hs:
            # Stochastic eligibility deferred (sfr_internal plumbing).
            _closure_a_runtime_hs = False
        if _closure_a_runtime_hs:
            _sfh_kw_hs = {k: v for k, v in p.items() if k in _sfh_internal_names_ca_hs}
            _sfr_orch_grid_hs = _sfh_fn_ca_hs(_sfh_lbt_grid_orch_64_hs, **_sfh_kw_hs)
            _sfr_on_ssp_orch_hs = jnp.interp(
                ssp_ages_yr, _sfh_lbt_grid_orch_64_hs, _sfr_orch_grid_hs
            )
            _T_TABLE_MIN_hs = 0.01
            _ssp_age_gyr_hs = ssp_ages_yr / 1e9
            _t_cosmic_raw_hs = _t_obs_gyr_fixed_ca_hs - _ssp_age_gyr_hs
            _n_ssp_hs = ssp_ages_yr.shape[0]
            _t_cosmic_floor_hs = jnp.maximum(_t_cosmic_raw_hs, _T_TABLE_MIN_hs)
            _valid_hs = _t_cosmic_raw_hs > 0.0
            _t_cosmic_asc_raw_hs = _t_cosmic_floor_hs[::-1]
            _sfr_asc_raw_hs = _sfr_on_ssp_orch_hs[::-1]
            _n_invalid_hs = jnp.sum(~_valid_hs[::-1])
            _idx_pos_hs = jnp.arange(_n_ssp_hs)
            _is_invalid_pos_hs = _idx_pos_hs < _n_invalid_hs
            _ramp_hs = _T_TABLE_MIN_hs + (_T_TABLE_MIN_hs * 0.5) * (_idx_pos_hs + 1) / jnp.maximum(
                _n_invalid_hs, 1
            )
            _t_cosmic_asc_hs = jnp.where(_is_invalid_pos_hs, _ramp_hs, _t_cosmic_asc_raw_hs)
            _sfr_asc_hs = jnp.where(_is_invalid_pos_hs, 0.0, _sfr_asc_raw_hs)
            _total_mass_hs = jnp.maximum(jnp.trapezoid(_sfr_asc_hs, _t_cosmic_asc_hs * 1e9), 0.0)
            _dsps_result_hs = _calc_rest_sed_ln_hs(
                gal_t_table=_t_cosmic_asc_hs,
                gal_sfr_table=_sfr_asc_hs,
                gal_lgmet=_lgmet_hs,
                gal_lgmet_scatter=_lgmet_scatter_ca_hs,
                ssp_lgmet=_ssp_lgmet_ca_hs,
                ssp_lg_age_gyr=_ssp_lg_age_gyr_ca_hs,
                ssp_flux=_ssp_flux_ca_hs,
                t_obs=_t_obs_gyr_fixed_ca_hs,
            )
            _weights_2d_hs = _dsps_result_hs.weights * _total_mass_hs
            weights = _weights_2d_hs.sum(axis=0)
            _w_safe_hs = jnp.maximum(weights, 1e-30)
            ssp_flux_at_z = jnp.einsum(
                "ma,maw->aw",
                _weights_2d_hs / _w_safe_hs[None, :],
                _ssp_flux_ca_hs,
            )
            # Metallicity-correct stellar pixel spectrum via joint
            # ``einsum("ma,map->p", weights_2d, ssp_on_pixels)``.
            stellar_spec = jnp.einsum("ma,map->p", _weights_2d_hs, ssp_on_pixels)
        else:
            weights = compute_csp_weights(sfr_on_ssp, ssp_ages_yr)
            # Metallicity interpolation
            # BUG-NSS-02: Use the provided log_z_abs, don't silently fall back
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
            # Stellar spectrum on pixels (legacy path: uses zeroth Z slice).
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
            # Canonical 10 Myr time-weighted SFR (Murphy+2011 timescale).
            p_ns["_sfr_current"] = time_weighted_sfr(sfr_on_ssp, ssp_ages_yr, 1e7)
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
            p = get_internal_params(
                params, param_map, spec, has_field, strict_unknown_params=False
            )
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
            p = get_internal_params(
                params, param_map, spec, has_field, strict_unknown_params=False
            )
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
