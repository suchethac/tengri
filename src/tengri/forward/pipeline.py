# SPDX-License-Identifier: BSD-3-Clause
"""Core SED computation pipeline.

This module implements the forward model engine that translates
physical parameters into rest-frame SEDs. It handles SFH dispatch,
metallicity interpolation, dust attenuation, nebular emission,
AGN contribution, dust emission, and IGM absorption.

All functions take a ``model`` argument (the :class:`~tengri.model.Model`
instance) instead of ``self``, allowing the heavy computation to live
outside the class while preserving access to model state.
"""

from __future__ import annotations

import jax.numpy as jnp

from tengri.components.stellar.sps.dsps_wrapper import (
    effective_metallicity,
    has_alpha_grid,
    interpolate_met_alpha,
    interpolate_met_alpha_evolving,
    interpolate_metallicity,
    interpolate_metallicity_evolving,
    interpolate_metallicity_smooth,
    interpolate_metallicity_smooth_evolving,
)


def interp_metallicity(model, log_z, ssp_flux=None, ssp_lgmet=None):
    """Dispatch metallicity interpolation on SSP grid (single Z value).

    Parameters
    ----------
    model : SEDModel
        The model instance.
    log_z : float
        log10(Z) absolute metallicity [dimensionless].
    ssp_flux : ndarray, optional
        Traced override for ``model.ssp_data.ssp_flux``. When provided
        with ``ssp_lgmet``, the SSP arrays enter the JIT graph as
        runtime tensors instead of closure-captured constants — the
        memory-efficient path; see
        ``docs/dev/quickstart_oom_diagnosis.md``.
    ssp_lgmet : ndarray, optional
        Traced override for ``model.ssp_data.ssp_lgmet``.

    Returns
    -------
    ndarray, shape (n_age, n_wave)
        SSP flux interpolated to target metallicity [erg/s/Hz].

    Notes
    -----
    **JIT-compatible**: yes — uses smooth or nearest-neighbor interpolation.
    """
    flux = ssp_flux if ssp_flux is not None else model.ssp_data.ssp_flux
    lgmet = ssp_lgmet if ssp_lgmet is not None else model.ssp_data.ssp_lgmet
    if model._met_interp == "smooth":
        return interpolate_metallicity_smooth(
            flux,
            lgmet,
            log_z,
            model._lgmet_scatter,
        )
    return interpolate_metallicity(flux, lgmet, log_z)


def interp_metallicity_evolving(model, log_z_per_age, ssp_flux=None, ssp_lgmet=None):
    """Dispatch per-age metallicity interpolation on SSP grid.

    Parameters
    ----------
    model : SEDModel
        The model instance.
    log_z_per_age : ndarray, shape (n_age,)
        log10(Z) absolute metallicity at each SSP age bin [dimensionless].

    Returns
    -------
    ndarray, shape (n_age, n_wave)
        SSP flux with age-dependent metallicity [erg/s/Hz].

    Notes
    -----
    **JIT-compatible**: yes — uses smooth or nearest-neighbor interpolation.
    """
    flux = ssp_flux if ssp_flux is not None else model.ssp_data.ssp_flux
    lgmet = ssp_lgmet if ssp_lgmet is not None else model.ssp_data.ssp_lgmet
    if model._met_interp == "smooth":
        return interpolate_metallicity_smooth_evolving(
            flux,
            lgmet,
            log_z_per_age,
            model._lgmet_scatter,
        )
    return interpolate_metallicity_evolving(flux, lgmet, log_z_per_age)


def _closure_a_sfh_prep(model, p, sfr, t_obs_gyr_for_weights):
    """Mirror :class:`StellarSEDComponent.apply` for closure-path-A.

    Builds the orchestrator-canonical inputs to DSPS:
    ``(t_cosmic_asc, sfr_asc, total_mass)`` and the orchestrator's
    64-pt SFH grid. The three legacy ``compute_sed_components``
    branches (no-α delta-Z, α-aware, chem_evol) all share these
    inputs to feed ``calc_rest_sed_sfh_table_lognormal_mdf`` /
    ``calc_rest_sed_sfh_table_met_table`` exactly the same way the
    orchestrator's component does.

    The three required alignments:

    1. Grid resolution: ``n_grid=64`` regardless of ``spec.stochastic``
       (legacy uses 256 for non-stochastic). For stochastic SFHs the
       legacy ``sfr`` already lives on a 64-pt grid with the GP draw
       baked in, so reuse it directly to preserve the realisation.
    2. Linear lookback-time interpolation to SSP ages (NOT legacy's
       log10-age interpolation).
    3. NaN-safe cosmic-time prep: a strictly-monotonic ramp + SFR-
       zeroing for invalid SSP bins (``ssp_age > t_obs``). Bare
       ``jnp.clip(min=1e-3)`` collapses multiple invalid bins to the
       same boundary value, producing a degenerate ``gal_t_table``
       that DSPS NaNs on at z>0.05 with old SSPs.

    Parameters
    ----------
    model : SEDModel
    p : dict
        Internal-name parameters (``model._get_internal_params(...)``).
    sfr : array, shape (n_internal_grid,)
        SFR on the legacy internal log_age_grid (256-pt non-stochastic,
        64-pt stochastic). Used directly when stochastic; otherwise
        re-evaluated on the 64-pt orchestrator grid.
    t_obs_gyr_for_weights : float
        Age of the universe at the observation redshift [Gyr].

    Returns
    -------
    t_cosmic_asc : array, shape (n_ssp_age,)
        Ascending cosmic-time grid for DSPS (Gyr).
    sfr_asc : array, shape (n_ssp_age,)
        SFR on the SSP age grid, ascending in cosmic time, with
        invalid-SSP-bin SFR zeroed.
    total_mass : scalar array
        Total stellar mass formed [Msun] (trapezoid integral).
    sfh_lbt_grid_orch : array, shape (64,)
        The orchestrator-style 64-pt SFH lookback-time grid [yr];
        useful when the caller needs to recompute mode-specific
        quantities (e.g. chem_evol's ``log_z_per_age``) on the same
        grid for bit-exact match.
    """
    from tengri.components.stellar.sfh.gp_sfh import make_log_age_grid

    # Read ``n_grid`` from the spec so the helper mirrors the orchestrator's
    # ``StellarSEDComponentConfig.n_grid`` (which is also sourced from
    # ``spec.n_grid`` in :func:`build_components`). For stochastic SFHs
    # this matches the size of the GP draw baked into ``sfr`` exactly, so
    # we can reuse ``sfr`` directly with no interpolation loss. For
    # parametric SFHs the helper re-evaluates ``sfh_fn`` on this grid.
    _n_grid = int(getattr(model.spec, "n_grid", 64))
    _log_age_grid = make_log_age_grid(_n_grid)
    sfh_lbt_grid_orch = jnp.power(10.0, _log_age_grid)
    if model.uses_stochastic_sfh:
        # Stochastic ``sfr`` is the GP draw on the spec's ``n_grid`` —
        # which equals ``_n_grid`` here, so reuse directly.
        _sfr_orch_grid = sfr
    else:
        # Include both internal and public SFH names so the composer
        # can dispatch per-component without colliding on shared internal
        # kwargs (e.g. ``log_total_mass``). See #372.
        _sfh_public_names = getattr(model, "_sfh_public_names", set())
        _kw = {
            k: v for k, v in p.items() if k in model._sfh_internal_names or k in _sfh_public_names
        }
        _sfr_orch_grid = model._sfh_fn(sfh_lbt_grid_orch, **_kw)

    _sfr_on_ssp_orch = jnp.interp(model.ssp_ages_yr, sfh_lbt_grid_orch, _sfr_orch_grid)

    _ssp_age_gyr = model.ssp_ages_yr / 1e9
    _T_TABLE_MIN = 0.01
    _t_cosmic_raw = t_obs_gyr_for_weights - _ssp_age_gyr
    _n_ssp = model.ssp_ages_yr.shape[0]
    _t_cosmic_floor = jnp.maximum(_t_cosmic_raw, _T_TABLE_MIN)
    _valid = _t_cosmic_raw > 0.0
    _t_cosmic_asc_raw = _t_cosmic_floor[::-1]
    _sfr_asc_raw = _sfr_on_ssp_orch[::-1]
    _n_invalid = jnp.sum(~_valid[::-1])
    _idx_pos = jnp.arange(_n_ssp)
    _is_invalid_pos = _idx_pos < _n_invalid
    _ramp = _T_TABLE_MIN + (_T_TABLE_MIN * 0.5) * (_idx_pos + 1) / jnp.maximum(_n_invalid, 1)
    t_cosmic_asc = jnp.where(_is_invalid_pos, _ramp, _t_cosmic_asc_raw)
    sfr_asc = jnp.where(_is_invalid_pos, 0.0, _sfr_asc_raw)
    total_mass = jnp.maximum(jnp.trapezoid(sfr_asc, t_cosmic_asc * 1e9), 0.0)

    return t_cosmic_asc, sfr_asc, total_mass, sfh_lbt_grid_orch


def interp_met_alpha_dispatch(
    model, log_z, alpha_fe, ssp_flux=None, ssp_lgmet=None, ssp_alpha_fe=None
):
    """Dispatch metallicity+alpha interpolation based on SSP grid dimensionality.

    Uses 4D bilinear interpolation if alpha-enhanced SSPs are available,
    otherwise falls back to effective_metallicity approximation on 3D grid.

    Parameters
    ----------
    model : SEDModel
        The model instance.
    log_z : float
        Iron abundance [Fe/H] or log10(Z/Zsun) [dimensionless].
    alpha_fe : float
        Alpha-element enhancement [α/Fe] [dex].

    Returns
    -------
    ndarray, shape (n_age, n_wave)
        SSP flux with metallicity and alpha enhancement [erg/s/Hz].

    Notes
    -----
    **JIT-compatible**: yes — uses bilinear or effective-Z interpolation.
    """
    if has_alpha_grid(model.ssp_data):
        return interpolate_met_alpha(
            ssp_flux if ssp_flux is not None else model.ssp_data.ssp_flux,
            ssp_lgmet if ssp_lgmet is not None else model.ssp_data.ssp_lgmet,
            ssp_alpha_fe if ssp_alpha_fe is not None else model.ssp_data.ssp_alpha_fe,
            log_z,
            alpha_fe,
        )
    # Fallback: effective_metallicity on 3D grid, with DSPS-canonical
    # lognormal MDF triweight kernel (Hearin+ 2021 Eq. 11). Matches
    # the no-α-enhancement path so ``alpha_fe = 0`` reduces exactly
    # to the non-α SED. See
    # ``docs/dev/20260504-csp-integral-canonicalization.md``.
    from dsps.sed.metallicity_weights import calc_lgmet_weights_from_lognormal_mdf

    log_z_eff = effective_metallicity(log_z, alpha_fe)
    flux = ssp_flux if ssp_flux is not None else model.ssp_data.ssp_flux
    lgmet = ssp_lgmet if ssp_lgmet is not None else model.ssp_data.ssp_lgmet
    lgmet_w = calc_lgmet_weights_from_lognormal_mdf(log_z_eff, model._lgmet_scatter, lgmet)
    return jnp.einsum("m,maw->aw", lgmet_w, flux)


def interp_met_alpha_evolving_dispatch(
    model,
    log_z_per_age,
    alpha_fe_per_age,
    ssp_flux=None,
    ssp_lgmet=None,
    ssp_alpha_fe=None,
):
    """Dispatch per-age metallicity+alpha interpolation on SSP grid.

    Uses per-age 4D bilinear if alpha-enhanced SSPs available,
    otherwise effective-Z approximation on 3D grid.

    Parameters
    ----------
    model : SEDModel
        The model instance.
    log_z_per_age : ndarray, shape (n_age,)
        [Fe/H] at each SSP age bin [dimensionless].
    alpha_fe_per_age : ndarray or float
        [α/Fe] at each age (array or scalar) [dex].

    Returns
    -------
    ndarray, shape (n_age, n_wave)
        SSP flux with per-age metallicity and alpha [erg/s/Hz].

    Notes
    -----
    **JIT-compatible**: yes — uses bilinear or effective-Z interpolation.
    """
    if has_alpha_grid(model.ssp_data):
        # Broadcast scalar alpha_fe to per-age array
        if not hasattr(alpha_fe_per_age, "shape") or alpha_fe_per_age.ndim == 0:
            alpha_fe_per_age = jnp.full_like(log_z_per_age, alpha_fe_per_age)
        return interpolate_met_alpha_evolving(
            ssp_flux if ssp_flux is not None else model.ssp_data.ssp_flux,
            ssp_lgmet if ssp_lgmet is not None else model.ssp_data.ssp_lgmet,
            ssp_alpha_fe if ssp_alpha_fe is not None else model.ssp_data.ssp_alpha_fe,
            log_z_per_age,
            alpha_fe_per_age,
        )
    # Fallback: effective_metallicity on 3D grid
    log_z_eff = effective_metallicity(log_z_per_age, alpha_fe_per_age)
    return interp_metallicity_evolving(model, log_z_eff, ssp_flux=ssp_flux, ssp_lgmet=ssp_lgmet)
