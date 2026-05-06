"""Fused JIT kernel builders for fast photometry and spectroscopy.

These factory functions build @jax.jit closures that capture precomputed
arrays (SSP grids, dust weights, effective wavelengths) at build time.
The returned functions take only per-call parameters (SFR weights, dust
params) as arguments.

Extracted from ``SEDModel`` methods to keep model.py focused on orchestration.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from tengri.forward.sed_model_types import SEDModelState

# ── Hybrid kernel: precomputed SSP + exact non-stellar ────────────


def build_hybrid_photometry(state: SEDModelState, model=None):
    """Build hybrid photometry kernel: precomputed stellar + exact non-stellar.

    Stellar CSP evaluated via precomputed SSP×filter einsum (fast, ~0.4% error).
    Non-stellar components evaluated at full wavelength resolution via
    emission_helpers, then integrated through filters (exact).

    This kernel bridges Tier 1 (fast but approximate) and Tier 2 (exact but slow):
    - Use for science models where stellar photometry speed matters but non-stellar
      accuracy is critical (nebular lines, AGN variability, etc.)
    - Non-stellar components must be available for full-wavelength evaluation
      (emission_helpers functions)

    Parameters
    ----------
    model : SEDModel
        The model instance providing config and precomputed arrays.

    Returns
    -------
    callable or None
        JIT-compiled function: ``(sfr_on_ssp, log_z_abs, tau_bc, tau_diff,
        dust_slope, ..., neb_logU=..., shock_frac=..., etc.)``
        → ``photometry_array``, where ``sfr_on_ssp`` has shape (n_age,)
        [Msun/yr] and ``photometry_array`` has shape (n_filters,)
        [erg/s/cm^2/Hz]. Returns None if prerequisites are not met.

    Notes
    -----
    **JIT-compatible**: yes — stellar photometry precomputed on filter grid,
    non-stellar components computed at full wavelength then integrated through
    filters.

    **Gradient-safe**: yes — differentiable w.r.t. all dust, AGN, nebular,
    and shock parameters.
    """
    from tengri.components.stellar.sps.dsps_wrapper import LSUN_ERG_PER_S
    from tengri.forward.emission_helpers import (
        agn_emission,
        attenuate_emission,
        dust_ir_emission,
        nebular_emission,
        radio_emission,
        shock_emission,
        xray_emission,
    )
    from tengri.observation.photometry import (
        compute_flux_density_batch,
        pad_filters,
    )

    dt = state.forward_dtype
    precomp = state.precomputed.photometry
    ssp_phot = precomp.ssp_phot.astype(dt)
    # True when metallicity axis was pre-collapsed at precompute time (fixed met_logzsol).
    # In that case ssp_phot has shape (n_age, n_filters) instead of (n_met, n_age, n_filters).
    _met_precomputed = ssp_phot.ndim == 2
    ssp_lgmet = state.ssp_data.ssp_lgmet.astype(dt)
    eff_waves_rest = precomp.effective_wavelengths_rest.astype(dt)
    _use_taylor = precomp.ssp_phot_moment is not None
    if _use_taylor:
        ssp_phot_moment = precomp.ssp_phot_moment.astype(dt)
    _is_single_dust = state.dust_model == "single_component"
    # Always use sigmoid age-dependent dust attenuation: the exact path
    # (``pipeline.py:_compute_dust_atten``) always uses smooth sigmoid
    # weights, so for hybrid to agree with exact within the 0.5%
    # dust-factorisation tolerance, the same age weighting is required.
    # Falls back to ``precompute_dust_age_weights`` when not pre-computed.
    _dust_exact = True
    # Define young/old masks unconditionally even though the smooth-sigmoid
    # branch (selected by ``_dust_exact = True``) is the only runtime path
    # used. The legacy hard-mask Taylor branch in ``_stellar_phot`` below
    # still references these names; keeping them defined keeps that branch
    # syntactically valid for callers that override ``_dust_exact`` to
    # False (e.g. by setting ``state.dust_scheme = "fast"`` in a future
    # patch). Cost is one ``jnp`` array materialisation at build time.
    _t_birth_h = 1e7
    young_mask = (state.ssp_ages_yr < _t_birth_h).astype(dt)
    old_mask = dt.type(1.0) - young_mask
    if not _is_single_dust:
        if state.precomputed.dust_age_weights is not None:
            dust_age_w = state.precomputed.dust_age_weights.astype(dt)
        else:
            from tengri.components.dust.attenuation import (
                precompute_dust_age_weights as _precompute_dust_age_weights_h,
            )

            dust_age_w = _precompute_dust_age_weights_h(state.ssp_ages_yr).astype(dt)
    flux_scale = dt.type(precomp.flux_scale)
    _csp_use_matrix = state.csp_integration == "log_interp"
    if _csp_use_matrix:
        _csp_mat = state.csp_matrix.astype(dt)
    else:
        _age_dt = state.csp_age_dt.astype(dt)
    lsun = dt.type(LSUN_ERG_PER_S)

    # Voronoi frequency bandwidths for L_absorbed broadband estimate (Hz).
    # Without these weights, sum(L_ν) is dimensionally wrong and
    # catastrophically underestimates L_absorbed for panchromatic filter sets.
    _eff_bw = state.precomputed.effective_bandwidths_hz
    if _eff_bw is not None:
        _eff_bw = _eff_bw.astype(dt)

    # Capture dust law functions
    law_bc_fn = state.dust_law_bc_fn
    if not _is_single_dust:
        law_diff_fn = state.dust_law_diff_fn

    from tengri.components.stellar.sps.dsps_wrapper import (
        _ALPHA_TO_Z_COEFF as _A2Z,
        has_alpha_grid,
    )

    _has_alpha = has_alpha_grid(state.ssp_data)
    if _has_alpha:
        ssp_alpha_fe = state.ssp_data.ssp_alpha_fe.astype(dt)

    _use_alpha_fe = state.spec.alpha_fe_evolving or "met_alpha_fe" in state.spec.free_params

    _use_smooth_z = state.met_interp == "smooth"
    _lgmet_scat = dt.type(state.lgmet_scatter)
    if _use_smooth_z:
        from tengri.components.stellar.sps.dsps_wrapper import compute_lgmet_weights as _clw

    # ── Closure-path-A captures (mirrors compositional kernel) ─────────
    # The exact path uses DSPS ``calc_rest_sed_sfh_table_lognormal_mdf``
    # for trapezoidal cosmic-time SFH integration + lognormal-MDF
    # metallicity weighting. Hybrid currently does rectangle-rule SFH
    # (``sfr * _age_dt``) + triweight-CDF metallicity weighting. To make
    # hybrid and exact agree to within the documented 0.5%
    # dust-factorisation tolerance, hybrid's stellar branch must use the
    # closure-A reduction for the no-α / non-met-precomputed delta-Z path.
    # Other branches (alpha, met_precomputed, evolving-Z) keep legacy
    # semantics.
    _ssp_lg_age_gyr_ca_h = state.ssp_data.ssp_lg_age_gyr
    _ssp_flux_ca_h = state.ssp_data.ssp_flux
    _ssp_lgmet_ca_h = state.ssp_data.ssp_lgmet
    _lgmet_scatter_ca_h = float(state.lgmet_scatter)
    from tengri.components.stellar.sfh.gp_sfh import (
        make_log_age_grid as _make_log_age_grid_ca_h,
    )
    from tengri.utils.cosmology import age_at_z as _age_at_z_fn_ca_h

    _orch_n_grid_ca_h = int(getattr(state.spec, "n_grid", 64))
    _sfh_lbt_grid_orch_64_h = jnp.power(10.0, _make_log_age_grid_ca_h(_orch_n_grid_ca_h))
    _t_obs_gyr_fixed_ca_h = (
        None if state.z_fixed is None else float(_age_at_z_fn_ca_h(state.z_fixed))
    )
    _sfh_fn_ca_h = model._sfh_fn if model is not None else None
    _sfh_internal_names_ca_h = model._sfh_internal_names if model is not None else set()
    _has_field_ca_h = state.uses_stochastic_sfh
    _model_log_age_grid_ca_h = model.log_age_grid if model is not None else None
    _ssp_ages_yr_ca_h = state.ssp_ages_yr
    # Closure-A is enabled only on the safest branch (delta-Z, no α, no
    # met_precomputed, default csp_integration). Other branches keep
    # legacy semantics for now.
    #
    # NB: ``_met_precomputed`` is allowed. When the metallicity axis was
    # collapsed at precompute time (Fixed ``met_logzsol``), DSPS's joint
    # MDF reduction collapses to a 1D age weighting that is trapezoidal
    # in cosmic time — exactly what we need to match the exact path's
    # SFH integration. The MDF marginalisation is then a no-op
    # (``ssp_at_z = ssp_phot`` directly), but the closure-A trapezoidal
    # weights still replace the legacy rectangle-rule ``sfr * _age_dt``.
    # This branch is critical for the DL07/Dale/THEMIS energy-balance
    # tests which all run with ``met_logzsol=Fixed`` (regression: 4.5%
    # mismatch in dust-IR was caused by the legacy rectangle-rule path
    # being used here).
    _closure_a_eligible_h = (
        not _use_alpha_fe and not _has_alpha and not _csp_use_matrix and _sfh_fn_ca_h is not None
    )
    if _closure_a_eligible_h:
        from dsps.sed.stellar_sed import (
            calc_rest_sed_sfh_table_lognormal_mdf as _calc_rest_sed_ln_h,
        )

    # IGM: full-wavelength transmission (compositional-quality, not approximate).
    # Precompute once at init on the rest-frame wavelength grid.
    _z_for_igm = state.z_fixed
    has_igm = state.uses_igm and _z_for_igm is not None
    _igm_fn = state.igm_fn
    if _igm_fn is None:
        from tengri.components.igm import igm_transmission as _igm_fn
    if has_igm:
        _wave_obs_igm = state.ssp_data.ssp_wave * (1.0 + _z_for_igm)
        igm_trans_full = jnp.asarray(_igm_fn(_wave_obs_igm, _z_for_igm), dtype=dt)
        # Per-filter effective IGM (for stellar preintegrated photometry)
        igm_trans_eff = state.precomputed.igm_at_effective_wavelengths
        if igm_trans_eff is not None:
            igm_trans_eff = igm_trans_eff.astype(dt)
        else:
            igm_trans_eff = jnp.ones(
                len(state.filter_waves) if state.filter_waves else 0, dtype=dt
            )

    # Note: has_dust_em flag checked for stellar L_absorbed only; full-wavelength
    # dust emission happens in non-stellar section
    # AGN at effective wavelengths (parametric mode only) — not used in hybrid
    # since non-stellar AGN is evaluated at full wavelength

    # === Non-stellar components (full wavelength) ===
    # TODO(refactor): The inline per-component blocks below (~200 lines) mirror
    # the pattern that build_fused_rest_sed was refactored away from via
    # build_nonstell_fn() in core/nonstell.py.  Migrating this kernel is deferred
    # because _hybrid_phot_body has two preintegrated photometry-level shortcuts —
    # dust-IR triweight lookup (_has_preint_dust_ir) and nebular preintegration
    # (_has_preint_neb) — that return filter-integrated quantities rather than a
    # per-wavelength SED.  build_nonstell_fn() returns a full-wavelength SED, so
    # the preintegrated paths cannot be incorporated without either (a) dropping
    # the fast paths (performance loss) or (b) extending build_nonstell_fn to
    # optionally return photometry shortcuts.  Revisit once the preintegrated paths
    # are verified and stabilised.  Tracked in docs/dev/sessions/.
    ssp_wave_f64 = state.ssp_data.ssp_wave
    rest_wave_f64 = state.rest_wavelength
    _needs_extension = rest_wave_f64 is not state.ssp_data.ssp_wave

    # Nebular
    has_nebular = state.nebular_backend is not None and getattr(
        state.nebular_backend, "has_free_params", False
    )
    # Check if preintegrated nebular data is available
    _has_preint_neb = has_nebular and getattr(
        state.nebular_backend, "_has_preint_photometry", False
    )
    if has_nebular:
        nebular_backend = state.nebular_backend
        ssp_log_ages_yr = state.ssp_log_ages_yr
        _neb_dust_mode = "bc"  # Fixed at state creation; not model-specific
        _neb_bc_fn = law_bc_fn  # Resolve from model._dust_law_bc_fn above
    if _has_preint_neb:
        # Capture preintegrated CLOUDY data for fast nebular photometry
        from tengri.utils.grid_interp import interp_nd_triweight

        _neb_cont_phot = nebular_backend._preint_continuum.phot  # (n_Z, n_age, n_logU, n_filt)
        _neb_cont_axes = nebular_backend._preint_continuum.axes
        _neb_cont_edges = nebular_backend._preint_continuum.edges
        _neb_line_weights = nebular_backend._preint_lines.line_filter_weights  # (n_lines, n_filt)
        _neb_line_axes = nebular_backend._preint_lines.axes
        _neb_line_edges = nebular_backend._preint_lines.edges
        # Line luminosity grid (log10 space; fixed axes pre-collapsed at init time so
        # ndim matches len(_neb_line_axes) for interp_nd_triweight)
        _neb_line_lum = nebular_backend._line_lum_collapsed
        # Young SSP indices and CLOUDY age grid for age-sum
        _neb_young_idx = jnp.array(nebular_backend._young_idx)
        _neb_qh_table = nebular_backend._qh_table  # (n_met_ssp, n_age_ssp)
        _neb_qh_log_met = nebular_backend._qh_log_met
        _neb_qh_log_age = nebular_backend._qh_log_age
        _neb_lya_idx = int(jnp.argmin(jnp.abs(nebular_backend.grid.line_wavelengths - 1215.67)))

    # Shock
    has_shock = getattr(model, "_uses_shock", False) if model is not None else False

    # Dust emission (full wavelength or preintegrated)
    has_dust_em_full = state.dust_emission_model is not None

    # Coarse SSP grid for energy-balance L_absorbed_stellar in hybrid kernel.
    # The Voronoi filter-band sum only covers filter wavelengths (e.g. ugriz at z=0.1
    # covers rest-frame ~2600–8800 Å), missing all UV absorption where dust peaks.
    # A 200-point log-spaced coarse grid captures the full SED at ~35x lower cost
    # than the full ~7000-point grid, matching the exact/compositional path formula
    # in nonstell.py:279: L_absorbed = -trapz(sed_intr - sed_att, nu).
    if has_dust_em_full:
        import numpy as _np  # factory-time only — not on JAX trace path

        _ssp_wave_full_np = _np.asarray(state.ssp_data.ssp_wave)
        _ssp_lnu_full_np = _np.asarray(state.ssp_data.ssp_flux)  # Lsun/Hz/Msun
        _N_COARSE = 200
        _wave_coarse_np = _np.geomspace(
            float(_ssp_wave_full_np[0]), float(_ssp_wave_full_np[-1]), _N_COARSE
        )
        _nu_coarse = jnp.asarray(2.99792458e18 / _wave_coarse_np, dtype=jnp.float64)
        _wave_coarse = jnp.asarray(_wave_coarse_np, dtype=jnp.float64)
        _ssp_lgmet_f64 = jnp.asarray(state.ssp_data.ssp_lgmet, dtype=jnp.float64)
        if _has_alpha:
            # ssp_flux shape: (n_met, n_alpha, n_age, n_wave)
            _nm, _na_fe, _na, _nw = _ssp_lnu_full_np.shape
            _ssp_lnu_coarse = jnp.asarray(
                _np.array(
                    [
                        [
                            [
                                _np.interp(
                                    _wave_coarse_np,
                                    _ssp_wave_full_np,
                                    _ssp_lnu_full_np[m, a_fe, a],
                                )
                                * LSUN_ERG_PER_S
                                for a in range(_na)
                            ]
                            for a_fe in range(_na_fe)
                        ]
                        for m in range(_nm)
                    ]
                ),
                dtype=jnp.float64,
            )  # (n_met, n_alpha, n_age, 200) erg/s/Hz/Msun
        else:
            # ssp_flux shape: (n_met, n_age, n_wave)
            _nm, _na, _nw = _ssp_lnu_full_np.shape
            _ssp_lnu_coarse = jnp.asarray(
                _np.array(
                    [
                        [
                            _np.interp(
                                _wave_coarse_np,
                                _ssp_wave_full_np,
                                _ssp_lnu_full_np[m, a],
                            )
                            * LSUN_ERG_PER_S
                            for a in range(_na)
                        ]
                        for m in range(_nm)
                    ]
                ),
                dtype=jnp.float64,
            )  # (n_met, n_age, 200) erg/s/Hz/Msun
        # Always precompute exact age weights for energy-balance trapz,
        # regardless of whether photometry uses the fast young/old split.
        if not _is_single_dust:
            _dust_age_w_f64 = dust_age_w.astype(jnp.float64)

    _has_preint_dust_ir = False
    _dust_model_name = None
    if has_dust_em_full:
        from tengri.components.dust.emission import preload_emission_model

        # preload_emission_model forces lazy template loading to happen NOW,
        # outside any JIT scope.  This prevents jnp.array() inside the loader
        # from creating DynamicJaxprTracers that would escape into closures and
        # cause UnexpectedTracerError when the JIT kernel is later called.
        dust_emission_fn = preload_emission_model(state.dust_emission_model)

        # Check if preintegrated dust IR lookup is available (for fast photometry)
        # NOTE: Disable preintegration if radio/X-ray is enabled, since dust IR
        # must be computed on the panchromatic grid to match other non-stellar
        # components. Preintegrated lookup was computed on SSP grid only.
        if state.precomputed.dust_ir_lookup is not None and not _needs_extension:
            _has_preint_dust_ir = True
            _dust_ir_lookup = state.precomputed.dust_ir_lookup
            _dust_model_name = state.dust_emission_model

    # AGN (full wavelength or preintegrated K&D disc)
    has_agn_full = state.agn_model is not None
    agn_parametric = state.agn_luminosity_mode if has_agn_full else False
    _has_preint_kd = False
    _has_preint_skirtor = False
    _kd_data_fn = None
    if has_agn_full:
        from tengri.components.agn import resolve_agn_model

        agn_model_fn_full = resolve_agn_model(state.agn_model)

        # Check if K&D disc preintegration is available (for fast photometry).
        # Preintegrated K&D is only valid when redshift is fixed and filters present.
        # Disable if radio/X-ray enabled, since K&D must be computed on panchromatic
        # grid to match other non-stellar components. Preintegration was computed on
        # SSP grid only.
        if (
            state.agn_model == "kubota_done_full"
            and state.precomputed.kd_preintegrated is not None
            and not _needs_extension
        ):
            _has_preint_kd = True
            from tengri.components.agn.kd_precompute import kubota_done_disc_preintegrated

            _kd_data_fn = kubota_done_disc_preintegrated
            _kd_data = state.precomputed.kd_preintegrated

        # SKIRTOR torus preintegration: filter-level torus lookup replaces the
        # full-wavelength SKIRTOR template at runtime.  Disc contribution is
        # computed at full wavelength via powerlaw_disc with the correct fraction.
        _skirtor_lookup = None
        _skirtor_disc_fn = None
        if (
            state.agn_model == "skirtor"
            and state.precomputed.skirtor_preintegrated is not None
            and not _needs_extension
        ):
            _has_preint_skirtor = True
            _skirtor_lookup = state.precomputed.skirtor_preintegrated
            from tengri.components.agn.disc import powerlaw_disc as _powerlaw_disc

            _skirtor_disc_fn = _powerlaw_disc

    # AGN-nebular emitter gates (BLR Gaussian, NLR Gaussian, Feltre NLR)
    # These are additive emitters controlled by state.agn_config flags.
    _has_blr_gaussian = (
        has_agn_full and state.agn_config is not None and state.agn_config.agn_blr_enabled
    )
    _has_nlr_gaussian = (
        has_agn_full and state.agn_config is not None and state.agn_config.agn_nlr_gaussian_enabled
    )
    _has_feltre_nlr = (
        has_agn_full
        and state.agn_config is not None
        and state.agn_config.agn_nlr_backend == "feltre"
    )

    # AGN-nebular preintegration gates
    _has_preint_blr = (
        has_agn_full
        and state.agn_config is not None
        and state.agn_config.agn_blr_enabled
        and state.precomputed.blr_lookup is not None
    )
    _has_preint_nlr_gaussian = (
        has_agn_full
        and state.agn_config is not None
        and state.agn_config.agn_nlr_gaussian_enabled
        and state.precomputed.nlr_gaussian_lookup is not None
    )
    _has_preint_feltre_nlr = (
        has_agn_full
        and state.agn_config is not None
        and state.agn_config.agn_nlr_backend == "feltre"
        and state.precomputed.feltre_nlr_lookup is not None
    )

    # Capture lookup closures
    _blr_lookup = None
    _nlr_gaussian_lookup = None
    _feltre_nlr_lookup = None
    if _has_preint_blr:
        _blr_lookup = state.precomputed.blr_lookup
    if _has_preint_nlr_gaussian:
        _nlr_gaussian_lookup = state.precomputed.nlr_gaussian_lookup
    if _has_preint_feltre_nlr:
        _feltre_nlr_lookup = state.precomputed.feltre_nlr_lookup

    # Disc/empirical AGN preintegration gates (mutually exclusive per agn_model)
    _has_preint_powerlaw_disc = False
    _has_preint_ss_disc = False
    _has_preint_cigale_disc = False
    _has_preint_qsogen = False
    _has_preint_silva04 = False
    _has_preint_cat3d = False
    _powerlaw_disc_lookup = None
    _ss_disc_lookup = None
    _cigale_disc_lookup = None
    _qsogen_lookup = None
    _silva04_lookup = None
    _cat3d_lookup = None

    if has_agn_full and not _needs_extension:
        # Disc models: only one per model at runtime
        if (
            state.agn_model == "powerlaw_disc"
            and state.precomputed.powerlaw_disc_preintegrated is not None
        ):
            _has_preint_powerlaw_disc = True
            _powerlaw_disc_lookup = state.precomputed.powerlaw_disc_preintegrated
        elif state.agn_model == "ss_disc" and state.precomputed.ss_disc_preintegrated is not None:
            _has_preint_ss_disc = True
            _ss_disc_lookup = state.precomputed.ss_disc_preintegrated
        elif (
            state.agn_model == "cigale_disc"
            and state.precomputed.cigale_disc_preintegrated is not None
        ):
            _has_preint_cigale_disc = True
            _cigale_disc_lookup = state.precomputed.cigale_disc_preintegrated
        elif state.agn_model == "qsogen" and state.precomputed.qsogen_preintegrated is not None:
            _has_preint_qsogen = True
            _qsogen_lookup = state.precomputed.qsogen_preintegrated
        # Torus models: alternative to skirtor
        elif state.agn_model == "silva04" and state.precomputed.silva04_preintegrated is not None:
            _has_preint_silva04 = True
            _silva04_lookup = state.precomputed.silva04_preintegrated
        elif state.agn_model == "cat3d_wind" and state.precomputed.cat3d_preintegrated is not None:
            _has_preint_cat3d = True
            _cat3d_lookup = state.precomputed.cat3d_preintegrated

    # Radio
    has_radio = state.uses_radio
    _has_preint_radio_synchrotron = False
    _has_preint_radio_freefree = False
    _has_preint_radio_agn_jet = False
    _radio_synchrotron_lookup = None
    _radio_freefree_lookup = None
    _radio_agn_jet_lookup = None
    if has_radio:
        _radio_sfr_mode = state.radio_sfr_mode
        _include_freefree = state.radio_include_freefree
        _redshift = float(getattr(state, "z_fixed", 0.0))

        # Check for preintegrated radio lookups (filter-level photometry)
        if (
            _radio_sfr_mode == "bell2003"
            and not _needs_extension
            and state.precomputed.radio_synchrotron_preintegrated is not None
        ):
            _has_preint_radio_synchrotron = True
            _radio_synchrotron_lookup = state.precomputed.radio_synchrotron_preintegrated
        if (
            _include_freefree
            and not _needs_extension
            and state.precomputed.radio_freefree_preintegrated is not None
        ):
            _has_preint_radio_freefree = True
            _radio_freefree_lookup = state.precomputed.radio_freefree_preintegrated
        if not _needs_extension and state.precomputed.radio_agn_jet_preintegrated is not None:
            _has_preint_radio_agn_jet = True
            _radio_agn_jet_lookup = state.precomputed.radio_agn_jet_preintegrated

    # X-ray (gate setup moved after n_filters definition below)
    has_xray = state.uses_xray

    # Constants for energy balance
    _c_aa = dt.type(2.99792458e18)

    # Redshift for filter integration
    z_fixed = state.z_fixed
    dl_cm_fixed = state.dl_cm_fixed

    # Filter information — pad to common length for vmap
    n_filters = len(state.filter_waves) if state.filter_waves else 0
    filter_waves_list = state.filter_waves if state.filter_waves else []
    filter_trans_list = state.filter_trans if state.filter_trans else []
    if n_filters > 0:
        fw_padded, ft_padded, _filt_n_valid = pad_filters(filter_waves_list, filter_trans_list)

    # X-ray precompute gates (after n_filters is defined)
    _has_preint_xray_xrb = False
    _has_preint_xray_corona = False
    _has_preint_xray_corona_lopez24 = False
    _xray_xrb_lookup = None
    _xray_corona_lookup = None
    _xray_corona_lopez24_lookup = None
    if has_xray and n_filters > 0:
        # Check for preintegrated X-ray lookups (filter-level photometry)
        # These are mutually exclusive: only one corona variant is active per SEDModel.
        if not _needs_extension and state.precomputed.xray_xrb_preintegrated is not None:
            _has_preint_xray_xrb = True
            _xray_xrb_lookup = state.precomputed.xray_xrb_preintegrated
        if not _needs_extension and state.precomputed.xray_corona_preintegrated is not None:
            _has_preint_xray_corona = True
            _xray_corona_lookup = state.precomputed.xray_corona_preintegrated
        if (
            not _needs_extension
            and state.precomputed.xray_corona_lopez24_preintegrated is not None
        ):
            _has_preint_xray_corona_lopez24 = True
            _xray_corona_lopez24_lookup = state.precomputed.xray_corona_lopez24_preintegrated

    # Shock MAPPINGS precompute gates (after n_filters is defined)
    _has_preint_shock_mappings = False
    _shock_mappings_lookup = None
    if (
        has_shock
        and n_filters > 0
        and not _needs_extension
        and state.precomputed.shock_mappings_preintegrated is not None
    ):
        # Check for preintegrated MAPPINGS shock lookup (filter-level line photometry)
        _has_preint_shock_mappings = True
        _shock_mappings_lookup = state.precomputed.shock_mappings_preintegrated

    # === Define kernel signatures (single vs two-component dust) ===

    if _is_single_dust:

        def hybrid_phot(
            sfr_on_ssp,
            log_z_abs,
            tau_v,
            dust_slope,
            f_obscuration=0.0,
            dust_bump_strength=0.0,
            dust_delta=0.0,
            dust_Rv=3.1,
            alpha_fe=0.0,
            dust_T=35.0,
            dust_beta_ir=1.6,
            dust_eta_balance=1.0,
            agn_log_lbol=10.0,
            agn_alpha=-1.0,
            agn_T_torus=1000.0,
            agn_tau_torus=5.0,
            agn_torus_frac=0.5,
            agn_log_mbh=7.0,
            agn_log_ledd=-1.0,
            # Non-stellar parameters
            neb_logU=-3.0,
            neb_logZ_gas=None,
            neb_log_nH=3.0,
            neb_fesc=0.0,
            neb_fesc_lya=0.0,
            shock_frac=0.0,
            shock_velocity=300.0,
            shock_log_density=0.0,
            shock_b_over_sqrt_n=1.0,
            shock_abundance="solar",
            shock_component="combined",
            dust_alpha_mir=2.0,
            dust_alpha_dale=2.0,
            dust_umin=1.0,
            dust_gamma_dl=0.01,
            dust_qpah=2.5,
            agn_polar_ebv=0.0,
            agn_cos_inc=0.5,
            agn_polar_oa=45.0,
            agn_a_spin=0.0,
            agn_tau_skirtor=7.0,
            agn_p_skirtor=1.0,
            agn_q_skirtor=1.0,
            agn_oa_skirtor=40.0,
            agn_frac=0.0,
            agn_T_hot=1200.0,
            agn_T_warm=300.0,
            agn_frac_hot=0.3,
            agn_f_hard=0.02,
            agn_gamma_warm=2.5,
            agn_kt_warm=0.2,
            agn_gamma_hard=1.8,
            agn_kt_hot=100.0,
            agn_r_warm_ratio=2.0,
            agn_blr_cf=0.1,
            agn_nlr_cf=0.1,
            agn_feltre_cf=0.1,
            agn_alpha_ion=-1.7,
            neb_xid=0.3,
            agn_fe2_strength=0.0,
            radio_q_ir=2.64,
            radio_alpha_sf=0.8,
            radio_loudness=0.0,
            radio_alpha_agn=0.7,
            radio_T_e=1e4,
            radio_alpha_ff=-0.1,
            xray_gamma_agn=1.8,
            xray_alpha_ox=-1.4,
            xray_gamma_hmxb=2.0,
            xray_gamma_lmxb=1.6,
            xray_E_cut=300.0,
            dust_qhac=0.17,
            dust_alpha_dl14=2.0,
            dust_log_ssfr=-10.0,
        ):
            """Compute hybrid photometry for single dust component."""
            return _hybrid_phot_body(
                sfr_on_ssp,
                log_z_abs,
                0.0,
                0.0,
                dust_slope,
                f_obscuration,
                dust_bump_strength,
                dust_delta,
                dust_Rv,
                alpha_fe,
                dust_T,
                dust_beta_ir,
                dust_eta_balance,
                agn_log_lbol,
                agn_alpha,
                agn_T_torus,
                agn_tau_torus,
                agn_torus_frac,
                agn_log_mbh,
                agn_log_ledd,
                neb_logU,
                neb_logZ_gas,
                neb_log_nH,
                neb_fesc,
                neb_fesc_lya,
                shock_frac,
                shock_velocity,
                shock_log_density,
                shock_b_over_sqrt_n,
                shock_abundance,
                shock_component,
                dust_alpha_mir,
                dust_alpha_dale,
                dust_umin,
                dust_gamma_dl,
                dust_qpah,
                agn_polar_ebv,
                agn_cos_inc,
                agn_polar_oa,
                agn_frac,
                agn_a_spin,
                agn_tau_skirtor,
                agn_p_skirtor,
                agn_q_skirtor,
                agn_oa_skirtor,
                agn_T_hot,
                agn_T_warm,
                agn_frac_hot,
                agn_f_hard,
                agn_gamma_warm,
                agn_kt_warm,
                agn_gamma_hard,
                agn_kt_hot,
                agn_r_warm_ratio,
                agn_blr_cf,
                agn_nlr_cf,
                agn_feltre_cf,
                agn_alpha_ion,
                neb_xid,
                agn_fe2_strength,
                radio_q_ir,
                radio_alpha_sf,
                radio_loudness,
                radio_alpha_agn,
                radio_T_e,
                radio_alpha_ff,
                xray_gamma_agn,
                xray_alpha_ox,
                xray_gamma_hmxb,
                xray_gamma_lmxb,
                xray_E_cut,
                dust_qhac,
                dust_alpha_dl14,
                dust_log_ssfr,
                tau_v=tau_v,
            )

    else:

        def hybrid_phot(
            sfr_on_ssp,
            log_z_abs,
            tau_bc,
            tau_diff,
            dust_slope,
            f_obscuration=0.0,
            dust_bump_strength=0.0,
            dust_delta=0.0,
            dust_Rv=3.1,
            alpha_fe=0.0,
            dust_T=35.0,
            dust_beta_ir=1.6,
            dust_eta_balance=1.0,
            agn_log_lbol=10.0,
            agn_alpha=-1.0,
            agn_T_torus=1000.0,
            agn_tau_torus=5.0,
            agn_torus_frac=0.5,
            agn_log_mbh=7.0,
            agn_log_ledd=-1.0,
            # Non-stellar parameters
            neb_logU=-3.0,
            neb_logZ_gas=None,
            neb_log_nH=3.0,
            neb_fesc=0.0,
            neb_fesc_lya=0.0,
            shock_frac=0.0,
            shock_velocity=300.0,
            shock_log_density=0.0,
            shock_b_over_sqrt_n=1.0,
            shock_abundance="solar",
            shock_component="combined",
            dust_alpha_mir=2.0,
            dust_alpha_dale=2.0,
            dust_umin=1.0,
            dust_gamma_dl=0.01,
            dust_qpah=2.5,
            agn_polar_ebv=0.0,
            agn_cos_inc=0.5,
            agn_polar_oa=45.0,
            agn_a_spin=0.0,
            agn_tau_skirtor=7.0,
            agn_p_skirtor=1.0,
            agn_q_skirtor=1.0,
            agn_oa_skirtor=40.0,
            agn_frac=0.0,
            agn_T_hot=1200.0,
            agn_T_warm=300.0,
            agn_frac_hot=0.3,
            agn_f_hard=0.02,
            agn_gamma_warm=2.5,
            agn_kt_warm=0.2,
            agn_gamma_hard=1.8,
            agn_kt_hot=100.0,
            agn_r_warm_ratio=2.0,
            agn_blr_cf=0.1,
            agn_nlr_cf=0.1,
            agn_feltre_cf=0.1,
            agn_alpha_ion=-1.7,
            neb_xid=0.3,
            agn_fe2_strength=0.0,
            radio_q_ir=2.64,
            radio_alpha_sf=0.8,
            radio_loudness=0.0,
            radio_alpha_agn=0.7,
            radio_T_e=1e4,
            radio_alpha_ff=-0.1,
            xray_gamma_agn=1.8,
            xray_alpha_ox=-1.4,
            xray_gamma_hmxb=2.0,
            xray_gamma_lmxb=1.6,
            xray_E_cut=300.0,
            dust_qhac=0.17,
            dust_alpha_dl14=2.0,
            dust_log_ssfr=-10.0,
        ):
            """Compute hybrid photometry for two-component dust."""
            return _hybrid_phot_body(
                sfr_on_ssp,
                log_z_abs,
                tau_bc,
                tau_diff,
                dust_slope,
                f_obscuration,
                dust_bump_strength,
                dust_delta,
                dust_Rv,
                alpha_fe,
                dust_T,
                dust_beta_ir,
                dust_eta_balance,
                agn_log_lbol,
                agn_alpha,
                agn_T_torus,
                agn_tau_torus,
                agn_torus_frac,
                agn_log_mbh,
                agn_log_ledd,
                neb_logU,
                neb_logZ_gas,
                neb_log_nH,
                neb_fesc,
                neb_fesc_lya,
                shock_frac,
                shock_velocity,
                shock_log_density,
                shock_b_over_sqrt_n,
                shock_abundance,
                shock_component,
                dust_alpha_mir,
                dust_alpha_dale,
                dust_umin,
                dust_gamma_dl,
                dust_qpah,
                agn_polar_ebv,
                agn_cos_inc,
                agn_polar_oa,
                agn_frac,
                agn_a_spin,
                agn_tau_skirtor,
                agn_p_skirtor,
                agn_q_skirtor,
                agn_oa_skirtor,
                agn_T_hot,
                agn_T_warm,
                agn_frac_hot,
                agn_f_hard,
                agn_gamma_warm,
                agn_kt_warm,
                agn_gamma_hard,
                agn_kt_hot,
                agn_r_warm_ratio,
                agn_blr_cf,
                agn_nlr_cf,
                agn_feltre_cf,
                agn_alpha_ion,
                neb_xid,
                agn_fe2_strength,
                radio_q_ir,
                radio_alpha_sf,
                radio_loudness,
                radio_alpha_agn,
                radio_T_e,
                radio_alpha_ff,
                xray_gamma_agn,
                xray_alpha_ox,
                xray_gamma_hmxb,
                xray_gamma_lmxb,
                xray_E_cut,
            )

    def _stellar_phot(
        sfr_on_ssp,
        log_z_abs,
        tau_bc,
        tau_diff,
        dust_slope,
        f_obscuration,
        dust_bump_strength,
        dust_delta,
        dust_Rv,
        alpha_fe,
        tau_v=0.0,
    ):
        """Preintegrated stellar photometry + L_absorbed_stellar.

        Computes stellar CSP through preintegrated SSP×filter tensor with
        metallicity + alpha interpolation, dust attenuation at effective
        wavelengths (single/two-component, exact/fast, with Taylor expansion),
        and L_absorbed_stellar via coarse 200-point trapz when dust emission
        is enabled (same formula as nonstell.py:279; replaces Voronoi band sum
        that missed UV absorption where dust attenuation peaks).

        Parameters
        ----------
        sfr_on_ssp : array-like
            SFR weights on SSP grid.
        log_z_abs : float
            log10(Z) absolute metallicity.
        tau_bc : float
            Birth-cloud optical depth (for two-component dust).
        tau_diff : float
            Diffuse ISM optical depth (for two-component dust).
        dust_slope : float
            Dust law power-law index.
        f_obscuration : float
            Obscured fraction (geometry).
        dust_bump_strength : float
            2175 Å bump strength.
        dust_delta : float
            Mid-UV dust delta parameter.
        dust_Rv : float
            Dust extinction curve Rv parameter.
        alpha_fe : float
            Alpha-to-iron enhancement ratio.
        tau_v : float, optional
            Optical depth (single-component dust case).

        Returns
        -------
        flux_attenuated : (n_filters,) array
            Stellar photometry in Lsun/Hz units.
        L_absorbed_stellar : scalar
            Absorbed stellar luminosity in erg/s.
        weights : (n_ages,) array
            CSP weights (needed for non-stellar calculations).
        """
        sfr = sfr_on_ssp.astype(dt)
        lz = jnp.asarray(log_z_abs, dtype=dt)
        tv1 = jnp.asarray(tau_bc, dtype=dt)
        tv2 = jnp.asarray(tau_diff, dtype=dt)
        tv = jnp.asarray(tau_v, dtype=dt)
        dn = jnp.asarray(dust_slope, dtype=dt)
        f_obs = jnp.asarray(f_obscuration, dtype=dt)
        bump = jnp.asarray(dust_bump_strength, dtype=dt)
        delta = jnp.asarray(dust_delta, dtype=dt)
        rv = jnp.asarray(dust_Rv, dtype=dt)
        afe = jnp.asarray(alpha_fe, dtype=dt)

        # CSP weights
        # ── Closure-path-A migration ──────────────────────────────────
        # When eligible (no-α / non-met-precomputed delta-Z, fixed z),
        # use DSPS lognormal-MDF + trapezoidal cosmic-time SFH integration
        # to bring hybrid into agreement with the exact path. Marginalise
        # back to (1D age weights, 2D MDF-averaged ssp_phot) so the
        # downstream dust einsums and Taylor moment block work unchanged.
        # See ``build_fused_tier2_photometry`` for the equivalence proof.
        _closure_a_runtime_h = _closure_a_eligible_h and _t_obs_gyr_fixed_ca_h is not None
        if _closure_a_runtime_h and _has_field_ca_h:
            # Stochastic SFHs need ``sfr_internal`` plumb-through similar
            # to the compositional kernel; not yet wired into the hybrid
            # caller. Fall back to legacy for stochastic.
            _closure_a_runtime_h = False
        # ``_stellar_phot`` only receives scalar args, not the params
        # dict. We reconstruct the orchestrator-grid SFR by interp'ing
        # ``sfr_on_ssp`` onto the 64-pt log-age grid. For parametric
        # SFHs this is one extra interp on top of the caller's
        # ``log_age_grid → ssp_log_ages_yr`` interp; the resulting cosmic-
        # time integration matches the exact path's closure-A reduction
        # to within the 256-vs-64-pt grid mismatch (small effect).
        if _closure_a_runtime_h:
            _sfr_orch_grid_h = jnp.interp(_sfh_lbt_grid_orch_64_h, _ssp_ages_yr_ca_h, sfr)
            _sfr_on_ssp_orch_h = jnp.interp(
                _ssp_ages_yr_ca_h, _sfh_lbt_grid_orch_64_h, _sfr_orch_grid_h
            )
            _T_TABLE_MIN_h = 0.01
            _ssp_age_gyr_h = _ssp_ages_yr_ca_h / 1e9
            _t_cosmic_raw_h = _t_obs_gyr_fixed_ca_h - _ssp_age_gyr_h
            _n_ssp_h = _ssp_ages_yr_ca_h.shape[0]
            _t_cosmic_floor_h = jnp.maximum(_t_cosmic_raw_h, _T_TABLE_MIN_h)
            _valid_h = _t_cosmic_raw_h > 0.0
            _t_cosmic_asc_raw_h = _t_cosmic_floor_h[::-1]
            _sfr_asc_raw_h = _sfr_on_ssp_orch_h[::-1]
            _n_invalid_h = jnp.sum(~_valid_h[::-1])
            _idx_pos_h = jnp.arange(_n_ssp_h)
            _is_invalid_pos_h = _idx_pos_h < _n_invalid_h
            _ramp_h = _T_TABLE_MIN_h + (_T_TABLE_MIN_h * 0.5) * (_idx_pos_h + 1) / jnp.maximum(
                _n_invalid_h, 1
            )
            _t_cosmic_asc_h = jnp.where(_is_invalid_pos_h, _ramp_h, _t_cosmic_asc_raw_h)
            _sfr_asc_h = jnp.where(_is_invalid_pos_h, 0.0, _sfr_asc_raw_h)
            _total_mass_h = jnp.maximum(jnp.trapezoid(_sfr_asc_h, _t_cosmic_asc_h * 1e9), 0.0)
            _dsps_result_h = _calc_rest_sed_ln_h(
                gal_t_table=_t_cosmic_asc_h,
                gal_sfr_table=_sfr_asc_h,
                gal_lgmet=lz,
                gal_lgmet_scatter=_lgmet_scatter_ca_h,
                ssp_lgmet=_ssp_lgmet_ca_h,
                ssp_lg_age_gyr=_ssp_lg_age_gyr_ca_h,
                ssp_flux=_ssp_flux_ca_h,
                t_obs=_t_obs_gyr_fixed_ca_h,
            )
            _weights_2d_h = _dsps_result_h.weights * _total_mass_h
            weights = _weights_2d_h.sum(axis=0)
            _w_safe_h = jnp.maximum(weights, 1e-30)
        else:
            weights = _csp_mat @ sfr if _csp_use_matrix else sfr * _age_dt
            _weights_2d_h = None
            _w_safe_h = None

        # Metallicity + alpha interpolation
        if _closure_a_runtime_h:
            if _met_precomputed:
                # Z axis already collapsed at precompute time → ssp_phot
                # is (n_age, n_filters). Closure-A's MDF marginalisation
                # is a no-op here; just use the precomputed table directly.
                # The trapezoidal cosmic-time ``weights`` (computed above)
                # still replaces the legacy rectangle-rule weights, which
                # is the dominant fix for the DL07/Dale/THEMIS regression.
                ssp_at_z = ssp_phot
            else:
                # MDF-marginalised ssp_at_z reproduces the joint
                # ``einsum("ma,maf->f", weights_2d, ssp_phot)`` reduction
                # downstream as ``einsum("a,af->f", weights, ssp_at_z)``.
                ssp_at_z = jnp.einsum(
                    "ma,maf->af",
                    _weights_2d_h / _w_safe_h[None, :],
                    ssp_phot,
                )
        elif _has_alpha:
            lz_c = jnp.clip(lz, ssp_lgmet[0], ssp_lgmet[-1])
            iz = jnp.clip(jnp.searchsorted(ssp_lgmet, lz_c) - 1, 0, ssp_lgmet.shape[0] - 2)
            fz = (lz_c - ssp_lgmet[iz]) / (ssp_lgmet[iz + 1] - ssp_lgmet[iz])
            afe_c = jnp.clip(afe, ssp_alpha_fe[0], ssp_alpha_fe[-1])
            ia = jnp.clip(jnp.searchsorted(ssp_alpha_fe, afe_c) - 1, 0, ssp_alpha_fe.shape[0] - 2)
            fa = (afe_c - ssp_alpha_fe[ia]) / (ssp_alpha_fe[ia + 1] - ssp_alpha_fe[ia])
            ssp_at_z = (
                (1 - fz) * (1 - fa) * ssp_phot[iz, ia]
                + fz * (1 - fa) * ssp_phot[iz + 1, ia]
                + (1 - fz) * fa * ssp_phot[iz, ia + 1]
                + fz * fa * ssp_phot[iz + 1, ia + 1]
            )
        else:
            if _met_precomputed:
                # Metallicity axis already collapsed at precompute time (fixed met_logzsol).
                # ssp_phot has shape (n_age, n_filters) — use directly.
                ssp_at_z = ssp_phot
            else:
                if _use_alpha_fe:
                    lz = lz + _A2Z * afe
                if _use_smooth_z:
                    zw = _clw(lz, ssp_lgmet, _lgmet_scat)
                    ssp_at_z = jnp.einsum("m,maf->af", zw, ssp_phot)
                else:
                    log_z_c = jnp.clip(lz, ssp_lgmet[0], ssp_lgmet[-1])
                    n_met = ssp_lgmet.shape[0]
                    idx = jnp.clip(jnp.searchsorted(ssp_lgmet, log_z_c) - 1, 0, n_met - 2)
                    frac = (log_z_c - ssp_lgmet[idx]) / (ssp_lgmet[idx + 1] - ssp_lgmet[idx])
                    ssp_at_z = (1.0 - frac) * ssp_phot[idx] + frac * ssp_phot[idx + 1]

        # Dust at effective wavelengths
        _law_kw = dict(n_slope=dn, dust_bump_strength=bump, dust_delta=delta, dust_Rv=rv)
        if _is_single_dust:
            k = law_bc_fn(eff_waves_rest, **_law_kw)
            trans_1d = f_obs + (1.0 - f_obs) * jnp.exp(-tv * k)
            flux_attenuated = jnp.einsum("i,if->f", weights, ssp_at_z) * trans_1d
        elif _dust_exact:
            k_bc = law_bc_fn(eff_waves_rest, **_law_kw)
            k_diff = law_diff_fn(eff_waves_rest, **_law_kw)
            tau = dust_age_w[:, None] * tv1 * k_bc[None, :] + tv2 * k_diff[None, :]
            dust = f_obs + (1.0 - f_obs) * jnp.exp(-tau)
            flux_attenuated = jnp.einsum("i,if,if->f", weights, dust, ssp_at_z)
        else:
            # Fast two-CSP decomposition
            k_bc = law_bc_fn(eff_waves_rest, **_law_kw)
            k_diff = law_diff_fn(eff_waves_rest, **_law_kw)
            trans_bc = jnp.exp(-tv1 * k_bc)
            trans_diff = jnp.exp(-tv2 * k_diff)

            csp_young = jnp.einsum("i,if->f", weights * young_mask, ssp_at_z)
            csp_old = jnp.einsum("i,if->f", weights * old_mask, ssp_at_z)

            if _use_taylor:
                _dl = dt.type(1.0)
                k_bc_p = law_bc_fn(eff_waves_rest + _dl, **_law_kw)
                k_bc_m = law_bc_fn(eff_waves_rest - _dl, **_law_kw)
                k_diff_p = law_diff_fn(eff_waves_rest + _dl, **_law_kw)
                k_diff_m = law_diff_fn(eff_waves_rest - _dl, **_law_kw)

                if _closure_a_runtime_h:
                    if _met_precomputed:
                        # Z axis pre-collapsed → moment is (n_age, n_filters).
                        ssp_moment_at_z = ssp_phot_moment
                    else:
                        # Closure-A: same MDF-marginalisation as ssp_at_z above.
                        ssp_moment_at_z = jnp.einsum(
                            "ma,maf->af",
                            _weights_2d_h / _w_safe_h[None, :],
                            ssp_phot_moment,
                        )
                elif _has_alpha:
                    ssp_moment_at_z = (
                        (1 - fz) * (1 - fa) * ssp_phot_moment[iz, ia]
                        + fz * (1 - fa) * ssp_phot_moment[iz + 1, ia]
                        + (1 - fz) * fa * ssp_phot_moment[iz, ia + 1]
                        + fz * fa * ssp_phot_moment[iz + 1, ia + 1]
                    )
                elif _met_precomputed:
                    # Metallicity axis pre-collapsed — moment grid is also (n_age, n_filters).
                    ssp_moment_at_z = ssp_phot_moment
                elif _use_smooth_z:
                    ssp_moment_at_z = jnp.einsum("m,maf->af", zw, ssp_phot_moment)
                else:
                    ssp_moment_at_z = (1.0 - frac) * ssp_phot_moment[idx] + frac * ssp_phot_moment[
                        idx + 1
                    ]

                csp_young_m = jnp.einsum("i,if->f", weights * young_mask, ssp_moment_at_z)
                csp_old_m = jnp.einsum("i,if->f", weights * old_mask, ssp_moment_at_z)

                A_young = trans_bc * trans_diff
                A_young_deriv = (
                    jnp.exp(-tv1 * k_bc_p) * jnp.exp(-tv2 * k_diff_p)
                    - jnp.exp(-tv1 * k_bc_m) * jnp.exp(-tv2 * k_diff_m)
                ) / (2 * _dl)
                A_old = trans_diff
                A_old_deriv = (jnp.exp(-tv2 * k_diff_p) - jnp.exp(-tv2 * k_diff_m)) / (2 * _dl)

                flux_no_geom = (
                    A_young * csp_young
                    + A_young_deriv * csp_young_m
                    + A_old * csp_old
                    + A_old_deriv * csp_old_m
                )
            else:
                flux_no_geom = trans_diff * (trans_bc * csp_young + csp_old)

            flux_intrinsic_for_geom = csp_young + csp_old
            flux_attenuated = f_obs * flux_intrinsic_for_geom + (1.0 - f_obs) * flux_no_geom

        # Compute L_absorbed_stellar via coarse SSP wavelength grid.
        # Replaces Voronoi-bandwidth filter-band sum, which only covered filter
        # wavelengths and missed UV absorption where dust attenuation peaks.
        # Same trapz formula as the exact/compositional path: nonstell.py:279.
        if has_dust_em_full:
            weights_f64 = weights.astype(jnp.float64)
            _kw_f64 = dict(
                n_slope=jnp.float64(dust_slope),
                dust_bump_strength=jnp.float64(dust_bump_strength),
                dust_delta=jnp.float64(dust_delta),
                dust_Rv=jnp.float64(dust_Rv),
            )
            f_obs_f64 = jnp.float64(f_obscuration)

            # Metallicity interpolation on coarse grid.
            # lz is already alpha-fe-corrected when _use_alpha_fe was applied above.
            # Closure-A is checked first because it covers both regular and
            # ``_met_precomputed`` cases uniformly via the joint
            # ``einsum("ma,maw->aw", weights_2d_norm, _ssp_lnu_coarse)``
            # reduction (``_ssp_lnu_coarse`` is always 3D regardless of
            # whether ``ssp_phot`` was collapsed at precompute time).
            if _has_alpha:
                # iz, fz, ia, fa computed above in the met+alpha interp block
                ssp_z_coarse = (
                    (1 - fz) * (1 - fa) * _ssp_lnu_coarse[iz, ia]
                    + fz * (1 - fa) * _ssp_lnu_coarse[iz + 1, ia]
                    + (1 - fz) * fa * _ssp_lnu_coarse[iz, ia + 1]
                    + fz * fa * _ssp_lnu_coarse[iz + 1, ia + 1]
                )  # (n_age, 200) erg/s/Hz/Msun
            elif _closure_a_runtime_h:
                # Closure-A energy-balance reduction: MDF-marginalise the
                # coarse SSP grid using the same ``weights_2d`` produced
                # for ``ssp_at_z``. Equivalent to the joint
                # ``einsum("ma,maw->w", weights_2d, ssp_lnu)`` reduction
                # the exact path uses, marginalised back into a (n_age, 200)
                # cube so the existing trapz/L_absorbed math is unchanged.
                # Works for both regular and ``_met_precomputed=True`` —
                # ``_ssp_lnu_coarse`` is built from the full 3D SSP flux
                # regardless of ``ssp_phot`` precompute collapse.
                ssp_z_coarse = jnp.einsum(
                    "ma,maw->aw",
                    (_weights_2d_h / _w_safe_h[None, :]).astype(jnp.float64),
                    _ssp_lnu_coarse,
                )  # (n_age, 200)
            elif _met_precomputed:
                # Fixed metallicity: lz is a constant; interpolate coarse grid directly.
                # (zw is not computed when _met_precomputed=True — mirrors Taylor block.)
                lz_f64 = jnp.float64(lz)
                lz_c64 = jnp.clip(lz_f64, _ssp_lgmet_f64[0], _ssp_lgmet_f64[-1])
                _ic = jnp.clip(
                    jnp.searchsorted(_ssp_lgmet_f64, lz_c64) - 1,
                    0,
                    _ssp_lgmet_f64.shape[0] - 2,
                )
                _fc = (lz_c64 - _ssp_lgmet_f64[_ic]) / (
                    _ssp_lgmet_f64[_ic + 1] - _ssp_lgmet_f64[_ic]
                )
                ssp_z_coarse = (1.0 - _fc) * _ssp_lnu_coarse[_ic] + _fc * _ssp_lnu_coarse[
                    _ic + 1
                ]  # (n_age, 200)
            elif _use_smooth_z:
                # zw computed above in the smooth met-interp block
                ssp_z_coarse = jnp.einsum(
                    "m,maw->aw", zw.astype(jnp.float64), _ssp_lnu_coarse
                )  # (n_age, 200)
            else:
                lz_f64 = jnp.float64(lz)
                lz_c64 = jnp.clip(lz_f64, _ssp_lgmet_f64[0], _ssp_lgmet_f64[-1])
                _ic = jnp.clip(
                    jnp.searchsorted(_ssp_lgmet_f64, lz_c64) - 1,
                    0,
                    _ssp_lgmet_f64.shape[0] - 2,
                )
                _fc = (lz_c64 - _ssp_lgmet_f64[_ic]) / (
                    _ssp_lgmet_f64[_ic + 1] - _ssp_lgmet_f64[_ic]
                )
                ssp_z_coarse = (1.0 - _fc) * _ssp_lnu_coarse[_ic] + _fc * _ssp_lnu_coarse[
                    _ic + 1
                ]  # (n_age, 200)

            # Attenuated CSP on coarse grid — same dust formula as filter-band path
            if _is_single_dust:
                csp_intr_coarse = jnp.einsum("i,iw->w", weights_f64, ssp_z_coarse)
                k_c = law_bc_fn(_wave_coarse, **_kw_f64)
                trans_c = f_obs_f64 + (1.0 - f_obs_f64) * jnp.exp(-jnp.float64(tau_v) * k_c)
                csp_att_coarse = csp_intr_coarse * trans_c
            elif _dust_exact:
                k_bc_c = law_bc_fn(_wave_coarse, **_kw_f64)
                k_diff_c = law_diff_fn(_wave_coarse, **_kw_f64)
                tau_c = (
                    _dust_age_w_f64[:, None] * jnp.float64(tau_bc) * k_bc_c[None, :]
                    + jnp.float64(tau_diff) * k_diff_c[None, :]
                )
                dust_c = f_obs_f64 + (1.0 - f_obs_f64) * jnp.exp(-tau_c)
                csp_intr_coarse = jnp.einsum("i,iw->w", weights_f64, ssp_z_coarse)
                csp_att_coarse = jnp.einsum("i,iw,iw->w", weights_f64, dust_c, ssp_z_coarse)
            else:
                # Energy balance requires accurate L_absorbed: always use exact
                # per-age dust_age_w weighting even when photometry uses the fast
                # young/old split.  The fast split underestimates UV absorption by
                # ~22% (binary threshold vs continuous sigmoid).
                k_bc_c = law_bc_fn(_wave_coarse, **_kw_f64)
                k_diff_c = law_diff_fn(_wave_coarse, **_kw_f64)
                tau_c = (
                    _dust_age_w_f64[:, None] * jnp.float64(tau_bc) * k_bc_c[None, :]
                    + jnp.float64(tau_diff) * k_diff_c[None, :]
                )
                dust_c = f_obs_f64 + (1.0 - f_obs_f64) * jnp.exp(-tau_c)
                csp_intr_coarse = jnp.einsum("i,iw->w", weights_f64, ssp_z_coarse)
                csp_att_coarse = jnp.einsum("i,iw,iw->w", weights_f64, dust_c, ssp_z_coarse)

            L_absorbed_stellar = -jnp.trapezoid(csp_intr_coarse - csp_att_coarse, _nu_coarse)
            L_absorbed_stellar = jnp.where(
                jnp.isfinite(L_absorbed_stellar), L_absorbed_stellar, jnp.float64(0.0)
            )
            L_absorbed_stellar = jnp.maximum(L_absorbed_stellar, jnp.float64(0.0))
        else:
            # Dust emission disabled; L_absorbed_stellar not used downstream.
            L_absorbed_stellar = dt.type(0.0)

        return flux_attenuated, L_absorbed_stellar, weights

    def _nebular_phot_preintegrated(
        weights,
        log_z_abs,
        neb_logU,
        neb_logZ_gas,
        neb_fesc,
        neb_fesc_lya,
        tau_bc,
        tau_diff,
        dust_slope,
        dust_bump_strength,
        tau_v=0.0,
    ):
        """Preintegrated nebular photometry (continuum + lines).

        Same pattern as _stellar_phot: age-weighted sum over preintegrated
        grid, with dust at effective wavelengths.  Operates on (n_filters,)
        instead of (n_wave,) — no filter integration loop needed.
        """
        _gas_z = jnp.where(neb_logZ_gas is None, log_z_abs, neb_logZ_gas)

        # --- Nebular continuum: age-sum over preintegrated CLOUDY grid ---
        # For each young SSP age, interp CLOUDY grid at (Z_gas, age, logU)
        # to get (n_filters,) per Q_H, then scale by Q_H × weight × (1-fesc)
        cont_phot = jnp.zeros(n_filters, dtype=jnp.float64)

        # Q_H interpolation helper (bilinear in SSP met × age)
        def _get_qh(log_z, log_age_yr):
            """Interpolate ionizing photon production rate from metallicity-age grid."""
            iz = jnp.clip(
                jnp.searchsorted(_neb_qh_log_met, log_z) - 1, 0, len(_neb_qh_log_met) - 2
            )
            ia = jnp.clip(
                jnp.searchsorted(_neb_qh_log_age, log_age_yr) - 1, 0, len(_neb_qh_log_age) - 2
            )
            fz = (log_z - _neb_qh_log_met[iz]) / jnp.maximum(
                _neb_qh_log_met[iz + 1] - _neb_qh_log_met[iz], 1e-30
            )
            fa = (log_age_yr - _neb_qh_log_age[ia]) / jnp.maximum(
                _neb_qh_log_age[ia + 1] - _neb_qh_log_age[ia], 1e-30
            )
            fz = jnp.clip(fz, 0.0, 1.0)
            fa = jnp.clip(fa, 0.0, 1.0)
            return (
                (1 - fz) * (1 - fa) * _neb_qh_table[iz, ia]
                + (1 - fz) * fa * _neb_qh_table[iz, ia + 1]
                + fz * (1 - fa) * _neb_qh_table[iz + 1, ia]
                + fz * fa * _neb_qh_table[iz + 1, ia + 1]
            )

        # Continuum: vmap over young ages
        young_ages = ssp_log_ages_yr[_neb_young_idx]
        young_weights = weights[_neb_young_idx]

        def _cont_one_age(log_age_i, weight_i):
            """Compute nebular continuum photometry contribution for one age bin."""
            qh_i = _get_qh(log_z_abs, log_age_i)
            # Triweight interp in (Z_gas, age_cloudy, logU) → (n_filters,) per Q_H
            # The preint grid is in log10 linear space (10^log10_lum was done at preint time)
            phot_per_qh = interp_nd_triweight(
                _neb_cont_phot,
                _neb_cont_axes,
                _neb_cont_edges,
                (_gas_z, log_age_i, neb_logU),
            )
            return weight_i * qh_i * phot_per_qh * (1.0 - neb_fesc)

        cont_contribs = jax.vmap(_cont_one_age)(young_ages, young_weights)
        cont_phot = jnp.sum(cont_contribs, axis=0)  # (n_filters,)

        # --- Lines: same age-sum but on (n_lines,), then project to filters ---
        def _line_one_age(log_age_i, weight_i):
            """Compute nebular emission line photometry contribution for one age bin."""
            qh_i = _get_qh(log_z_abs, log_age_i)
            # Interp line luminosities in (Z_gas, age, logU) → (n_lines,) log10
            log_lum_per_qh = interp_nd_triweight(
                _neb_line_lum,
                _neb_line_axes,
                _neb_line_edges,
                (_gas_z, log_age_i, neb_logU),
            )
            return weight_i * qh_i * (10.0**log_lum_per_qh) * (1.0 - neb_fesc)

        line_contribs = jax.vmap(_line_one_age)(young_ages, young_weights)
        total_line_lum = jnp.sum(line_contribs, axis=0)  # (n_lines,)

        # Lya escape fraction correction
        lya_scale = (1.0 - neb_fesc_lya) / jnp.maximum(1.0 - neb_fesc, 1e-10)
        total_line_lum = total_line_lum.at[_neb_lya_idx].multiply(lya_scale)

        # Project lines to filter space: (n_lines,) × (n_lines, n_filt) → (n_filt,)
        line_phot = jnp.einsum("l,lf->f", total_line_lum, _neb_line_weights)

        # Total nebular photometry (before dust)
        neb_phot = cont_phot + line_phot

        # Apply nebular dust at effective wavelengths
        # (same approximation as stellar: evaluate dust curve at λ_eff per filter)
        _law_kw = {"n_slope": dust_slope, "dust_bump_strength": dust_bump_strength}
        _tau_bc_neb = tau_bc if not _is_single_dust else tau_v
        _tau_diff_neb = tau_diff if not _is_single_dust else jnp.float64(0.0)
        if _neb_dust_mode in ("bc", "neb"):
            k_bc_neb = law_bc_fn(eff_waves_rest, **_law_kw)
            neb_phot = neb_phot * jnp.exp(-_tau_bc_neb * k_bc_neb)
        if _neb_dust_mode != "none":
            if not _is_single_dust:
                k_diff_neb = law_diff_fn(eff_waves_rest, **_law_kw)
            else:
                k_diff_neb = law_bc_fn(eff_waves_rest, **_law_kw)
            neb_phot = neb_phot * jnp.exp(-_tau_diff_neb * k_diff_neb)

        return neb_phot

    def _hybrid_phot_body(
        sfr_on_ssp,
        log_z_abs,
        tau_bc,
        tau_diff,
        dust_slope,
        f_obscuration,
        dust_bump_strength,
        dust_delta,
        dust_Rv,
        alpha_fe,
        dust_T,
        dust_beta_ir,
        dust_eta_balance,
        agn_log_lbol,
        agn_alpha,
        agn_T_torus,
        agn_tau_torus,
        agn_torus_frac,
        agn_log_mbh,
        agn_log_ledd,
        neb_logU,
        neb_logZ_gas,
        neb_log_nH,
        neb_fesc,
        neb_fesc_lya,
        shock_frac,
        shock_velocity,
        shock_log_density,
        shock_b_over_sqrt_n,
        shock_abundance,
        shock_component,
        dust_alpha_mir,
        dust_alpha_dale,
        dust_umin,
        dust_gamma_dl,
        dust_qpah,
        agn_polar_ebv,
        agn_cos_inc,
        agn_polar_oa,
        agn_frac,
        agn_a_spin,
        agn_tau_skirtor,
        agn_p_skirtor,
        agn_q_skirtor,
        agn_oa_skirtor,
        agn_T_hot,
        agn_T_warm,
        agn_frac_hot,
        agn_f_hard,
        agn_gamma_warm,
        agn_kt_warm,
        agn_gamma_hard,
        agn_kt_hot,
        agn_r_warm_ratio,
        agn_blr_cf,
        agn_nlr_cf,
        agn_feltre_cf,
        agn_alpha_ion,
        neb_xid,
        agn_fe2_strength,
        radio_q_ir,
        radio_alpha_sf,
        radio_loudness,
        radio_alpha_agn,
        radio_T_e,
        radio_alpha_ff,
        xray_gamma_agn,
        xray_alpha_ox,
        xray_gamma_hmxb,
        xray_gamma_lmxb,
        xray_E_cut,
        dust_qhac=0.17,
        dust_alpha_dl14=2.0,
        dust_log_ssfr=-10.0,
        tau_v=0.0,
    ):
        """Hybrid kernel body: stellar (precomputed) + non-stellar (exact)."""
        # === STEP 1: Stellar photometry ===
        flux_attenuated, L_absorbed_stellar, weights = _stellar_phot(
            sfr_on_ssp,
            log_z_abs,
            tau_bc,
            tau_diff,
            dust_slope,
            f_obscuration,
            dust_bump_strength,
            dust_delta,
            dust_Rv,
            alpha_fe,
            tau_v=tau_v,
        )

        # === STEP 2: Non-stellar SED at full wavelength resolution ===
        # TODO: preintegrated non-stellar paths (CLOUDY, DL07, SKIRTOR)
        # are implemented in _nebular_phot_preintegrated() but need
        # calibration verification before enabling. For now, all
        # non-stellar components use the full-wavelength path.
        _has_any_nonstell = (
            has_nebular or has_shock or has_dust_em_full or has_agn_full or has_radio or has_xray
        )

        non_stellar_phot = jnp.zeros(n_filters, dtype=jnp.float64)

        if _has_any_nonstell:
            non_stellar_sed = jnp.zeros_like(ssp_wave_f64)
            L_abs_neb = jnp.float64(0.0)

            if has_nebular:
                # Use SFR (Msun/yr), NOT CSP mass weight (Msun).
                _sfr_last = sfr_on_ssp[-1]
                neb_raw = nebular_emission(
                    nebular_backend,
                    weights,
                    ssp_wave_f64,
                    ssp_log_ages_yr,
                    jnp.float64(log_z_abs),
                    _sfr_last,
                    neb_logU=neb_logU,
                    neb_logZ_gas=neb_logZ_gas,
                    neb_fesc=neb_fesc,
                    neb_fesc_lya=neb_fesc_lya,
                )
                _tau_bc_neb = tau_bc if not _is_single_dust else tau_v
                _tau_diff_neb = tau_diff if not _is_single_dust else jnp.float64(0.0)
                _dust_kw_neb = {
                    "dust_slope": jnp.float64(dust_slope),
                    "dust_bump_strength": jnp.float64(dust_bump_strength),
                }
                neb_sed, L_abs_neb = attenuate_emission(
                    neb_raw,
                    ssp_wave_f64,
                    _neb_dust_mode,
                    _tau_bc_neb,
                    _tau_diff_neb,
                    law_bc_fn,
                    law_diff_fn if not _is_single_dust else law_bc_fn,
                    neb_bc_fn=_neb_bc_fn,
                    **_dust_kw_neb,
                )
                non_stellar_sed = non_stellar_sed + neb_sed

            # 2b: Shock emission (skip if preintegrated)
            if has_shock and not _has_preint_shock_mappings:
                shock_raw = shock_emission(
                    ssp_wave_f64,
                    non_stellar_sed,
                    shock_frac=shock_frac,
                    shock_velocity=shock_velocity,
                    shock_log_density=shock_log_density,
                    shock_b_over_sqrt_n=shock_b_over_sqrt_n,
                    shock_abundance=shock_abundance,
                    shock_component=shock_component,
                )
                _tau_diff_shock = tau_diff if not _is_single_dust else jnp.float64(0.0)
                _dust_kw_shock = {
                    "dust_slope": jnp.float64(dust_slope),
                    "dust_bump_strength": jnp.float64(dust_bump_strength),
                }
                shock_sed, _ = attenuate_emission(
                    shock_raw,
                    ssp_wave_f64,
                    "diff",
                    jnp.float64(0.0),
                    _tau_diff_shock,
                    law_bc_fn,
                    law_diff_fn if not _is_single_dust else law_bc_fn,
                    **_dust_kw_shock,
                )
                non_stellar_sed = non_stellar_sed + shock_sed

            # 2b5: Extend to panchromatic grid if radio/X-ray enabled
            # (before components that need extended wavelength range)
            if _needs_extension:
                from tengri.utils.wavelength import interpolate_sed_to_grid

                non_stellar_sed = interpolate_sed_to_grid(
                    ssp_wave_f64, non_stellar_sed, rest_wave_f64
                )

            # 2c: Dust IR emission (energy-balanced)
            dust_ir_phot_preint = jnp.zeros(n_filters, dtype=jnp.float64)
            if has_dust_em_full:
                L_ir = jnp.maximum(
                    (L_absorbed_stellar + L_abs_neb) * jnp.float64(dust_eta_balance), 0.0
                )

                if _has_preint_dust_ir:
                    # Use preintegrated template lookup (fast triweight interp)
                    # Signature varies by dust model
                    if _dust_model_name == "draine_li2007":
                        # DL07: (L_absorbed, dust_umin, dust_gamma_dl, dust_qpah)
                        dust_ir_phot_preint = _dust_ir_lookup(
                            L_ir,
                            jnp.float64(dust_umin),
                            jnp.float64(dust_gamma_dl),
                            jnp.float64(dust_qpah),
                        )
                    elif _dust_model_name == "dale2014":
                        # Dale2014: (L_absorbed, dust_alpha_dale)
                        dust_ir_phot_preint = _dust_ir_lookup(
                            L_ir,
                            jnp.float64(dust_alpha_dale),
                        )
                    elif _dust_model_name == "astrodust":
                        # Astrodust: (L_absorbed, dust_umin, dust_gamma_dl, dust_qpah)
                        dust_ir_phot_preint = _dust_ir_lookup(
                            L_ir,
                            jnp.float64(dust_umin),
                            jnp.float64(dust_gamma_dl),
                            jnp.float64(dust_qpah),
                        )
                    elif _dust_model_name == "themis":
                        # THEMIS: (L_absorbed, dust_umin, dust_gamma_dl, dust_qhac)
                        dust_ir_phot_preint = _dust_ir_lookup(
                            L_ir,
                            jnp.float64(dust_umin),
                            jnp.float64(dust_gamma_dl),
                            jnp.float64(dust_qhac),
                        )
                    elif _dust_model_name == "draine_li2014":
                        # DL14: (L, dust_umin, dust_gamma_dl, dust_qpah, dust_alpha_dl14)
                        dust_ir_phot_preint = _dust_ir_lookup(
                            L_ir,
                            jnp.float64(dust_umin),
                            jnp.float64(dust_gamma_dl),
                            jnp.float64(dust_qpah),
                            jnp.float64(dust_alpha_dl14),
                        )
                    elif _dust_model_name == "bosa":
                        # BOSA: (L_absorbed, dust_log_ssfr) — log_ltir is derived
                        dust_ir_phot_preint = _dust_ir_lookup(
                            L_ir,
                            jnp.float64(dust_log_ssfr),
                        )
                    else:
                        # Generic template model: (L_absorbed, *grid_params)
                        # For now, fall back to full-wavelength for other models
                        dust_ir = dust_ir_emission(
                            dust_emission_fn,
                            rest_wave_f64,
                            L_ir,
                            dust_T=jnp.float64(dust_T),
                            dust_beta_ir=jnp.float64(dust_beta_ir),
                            dust_alpha_mir=jnp.float64(dust_alpha_mir),
                            dust_alpha_dale=jnp.float64(dust_alpha_dale),
                            dust_umin=jnp.float64(dust_umin),
                            dust_gamma_dl=jnp.float64(dust_gamma_dl),
                            dust_qpah=jnp.float64(dust_qpah),
                        )
                        non_stellar_sed = non_stellar_sed + dust_ir
                else:
                    # Full-wavelength computation (fallback or if preintegration disabled)
                    dust_ir = dust_ir_emission(
                        dust_emission_fn,
                        rest_wave_f64,
                        L_ir,
                        dust_T=jnp.float64(dust_T),
                        dust_beta_ir=jnp.float64(dust_beta_ir),
                        dust_alpha_mir=jnp.float64(dust_alpha_mir),
                        dust_alpha_dale=jnp.float64(dust_alpha_dale),
                        dust_umin=jnp.float64(dust_umin),
                        dust_gamma_dl=jnp.float64(dust_gamma_dl),
                        dust_qpah=jnp.float64(dust_qpah),
                    )
                    non_stellar_sed = non_stellar_sed + dust_ir
            else:
                L_ir = jnp.float64(0.0)

            # 2d: AGN emission (full wavelength or preintegrated disc/torus)
            agn_phot_preint = jnp.zeros(n_filters, dtype=jnp.float64)
            skirtor_torus_preint = jnp.zeros(n_filters, dtype=jnp.float64)
            if has_agn_full:
                if agn_parametric:
                    _agn_lbol = agn_log_lbol
                    _agn_frac = 1.0
                else:
                    _agn_frac = jnp.float64(0.0)  # Not parametric in hybrid
                    _agn_lbol = 10.0

                # Check if ANY disc/empirical/torus precompute is available
                _has_any_disc_torus_preint = (
                    _has_preint_powerlaw_disc
                    or _has_preint_ss_disc
                    or _has_preint_cigale_disc
                    or _has_preint_qsogen
                    or _has_preint_silva04
                    or _has_preint_cat3d
                )

                if _has_any_disc_torus_preint:
                    # Preintegrated disc/empirical/torus: filter-level photometry.
                    # Polar dust is handled via effective-wavelength attenuation (see below).
                    # Signature: agn_log_lbol, (*free_axes), agn_torus_frac (torus only).
                    # Returns L_nu [erg/s/Hz]; scale to flux density via flux_scale.

                    disc_torus_lnu = jnp.zeros(n_filters, dtype=jnp.float64)
                    if _has_preint_powerlaw_disc:
                        disc_torus_lnu = _powerlaw_disc_lookup(
                            jnp.float64(_agn_lbol),
                        )
                    elif _has_preint_ss_disc:
                        disc_torus_lnu = _ss_disc_lookup(
                            jnp.float64(_agn_lbol),
                        )
                    elif _has_preint_cigale_disc:
                        disc_torus_lnu = _cigale_disc_lookup(
                            jnp.float64(_agn_lbol),
                        )
                    elif _has_preint_qsogen:
                        disc_torus_lnu = _qsogen_lookup(
                            jnp.float64(_agn_lbol),
                        )
                    elif _has_preint_silva04:
                        disc_torus_lnu = _silva04_lookup(
                            jnp.float64(_agn_lbol),
                            agn_torus_frac=jnp.float64(agn_torus_frac),
                        )
                    elif _has_preint_cat3d:
                        disc_torus_lnu = _cat3d_lookup(
                            jnp.float64(_agn_lbol),
                            agn_torus_frac=jnp.float64(agn_torus_frac),
                        )

                    # Scale disc/torus: L_ν [erg/s/Hz] * agn_frac → flux density [erg/s/cm²/Hz]
                    # Note: polar dust attenuation (agn_polar_ebv) is skipped for preintegrated
                    # models (agn_polar_ebv typically ~0 in practice for these SED fits).
                    # If polar dust is significant, revert to runtime agn_emission path
                    # by deleting the preintegration file or setting precompute=False.
                    agn_phot_preint = (
                        disc_torus_lnu * jnp.float64(_agn_frac) * jnp.float64(flux_scale)
                    )

                elif _has_preint_skirtor:
                    # Fast path: preintegrated SKIRTOR torus + full-wavelength disc.
                    # Torus: filter-level triweight lookup (bypasses 132-point template).
                    # Disc: powerlaw_disc at full wavelength with correct luminosity fraction.
                    skirtor_torus_lnu = _skirtor_lookup(
                        jnp.float64(_agn_lbol),
                        jnp.float64(agn_tau_skirtor),
                        jnp.float64(agn_p_skirtor),
                        jnp.float64(agn_q_skirtor),
                        jnp.float64(agn_oa_skirtor),
                        jnp.float64(agn_cos_inc),
                        jnp.float64(agn_torus_frac),
                    )
                    # Scale torus: L_ν [erg/s/Hz] * agn_frac → flux density [erg/s/cm²/Hz]
                    skirtor_torus_preint = (
                        skirtor_torus_lnu * jnp.float64(_agn_frac) * jnp.float64(flux_scale)
                    )

                    # Disc: powerlaw_disc with disc fraction of L_bol
                    # agn_frac_disc = _agn_frac * (1 - torus_frac) matches skirtor_agn:
                    #   l_disc = powerlaw_disc(..., agn_frac=1 - agn_torus_frac) * agn_frac
                    _disc_frac = jnp.float64(_agn_frac) * (1.0 - jnp.float64(agn_torus_frac))
                    disc_sed = agn_emission(
                        _skirtor_disc_fn,
                        rest_wave_f64,
                        agn_log_lbol=jnp.float64(_agn_lbol),
                        agn_frac=_disc_frac,
                        agn_polar_ebv=jnp.float64(agn_polar_ebv),
                        agn_cos_inc=jnp.float64(agn_cos_inc),
                        agn_polar_oa=jnp.float64(agn_polar_oa),
                        agn_alpha=jnp.float64(agn_alpha),
                    )
                    non_stellar_sed = non_stellar_sed + disc_sed

                else:
                    # Fast path: preintegrated K&D disc (if available).
                    # Computes filter-integrated photometry directly, bypassing full-wavelength
                    # computation for the outer disc component only. Torus, NLR, BLR still
                    # use full-wavelength path via agn_emission below.
                    if _has_preint_kd:
                        kd_disc_lnu = _kd_data_fn(
                            _kd_data,
                            agn_log_lbol=jnp.float64(_agn_lbol),
                            agn_frac=jnp.float64(_agn_frac),
                            agn_log_mbh=jnp.float64(agn_log_mbh),
                            agn_log_ledd=jnp.float64(agn_log_ledd),
                            agn_a_spin=jnp.float64(agn_a_spin),
                            agn_cos_inc=jnp.float64(agn_cos_inc),
                            agn_f_hard=jnp.float64(agn_f_hard),
                            agn_gamma_warm=jnp.float64(agn_gamma_warm),
                            agn_kt_warm=jnp.float64(agn_kt_warm),
                            agn_gamma_hard=jnp.float64(agn_gamma_hard),
                            agn_kt_hot=jnp.float64(agn_kt_hot),
                            agn_r_warm_ratio=jnp.float64(agn_r_warm_ratio),
                        )
                        # Convert L_nu (erg/s/Hz) to flux density (erg/s/cm^2/Hz)
                        agn_phot_preint = kd_disc_lnu * flux_scale

                    # Full-wavelength path: always computed to get torus, NLR, BLR, and
                    # hot corona. When K&D disc is preintegrated, this includes all AGN
                    # components except the outer disc (whose contribution is in agn_phot_preint).
                    agn_sed = agn_emission(
                        agn_model_fn_full,
                        rest_wave_f64,
                        agn_log_lbol=_agn_lbol,
                        agn_frac=_agn_frac,
                        agn_polar_ebv=jnp.float64(agn_polar_ebv),
                        agn_cos_inc=jnp.float64(agn_cos_inc),
                        agn_polar_oa=jnp.float64(agn_polar_oa),
                        agn_alpha=jnp.float64(agn_alpha),
                        agn_T_torus=jnp.float64(agn_T_torus),
                        agn_tau_torus=jnp.float64(agn_tau_torus),
                        agn_torus_frac=jnp.float64(agn_torus_frac),
                        agn_log_mbh=jnp.float64(agn_log_mbh),
                        agn_log_ledd=jnp.float64(agn_log_ledd),
                        agn_a_spin=jnp.float64(agn_a_spin),
                        agn_tau_skirtor=jnp.float64(agn_tau_skirtor),
                        agn_p_skirtor=jnp.float64(agn_p_skirtor),
                        agn_q_skirtor=jnp.float64(agn_q_skirtor),
                        agn_oa_skirtor=jnp.float64(agn_oa_skirtor),
                        agn_T_hot=jnp.float64(agn_T_hot),
                        agn_T_warm=jnp.float64(agn_T_warm),
                        agn_frac_hot=jnp.float64(agn_frac_hot),
                        agn_f_hard=jnp.float64(agn_f_hard),
                        agn_gamma_warm=jnp.float64(agn_gamma_warm),
                        agn_kt_warm=jnp.float64(agn_kt_warm),
                        agn_gamma_hard=jnp.float64(agn_gamma_hard),
                        agn_kt_hot=jnp.float64(agn_kt_hot),
                        agn_r_warm_ratio=jnp.float64(agn_r_warm_ratio),
                    )
                    non_stellar_sed = non_stellar_sed + agn_sed

                    # Add AGN-nebular emitters (BLR Gaussian, NLR Gaussian, Feltre NLR)
                    # These are additive to the base AGN SED, gated by config flags.
                    # BLR Gaussian: convert L_bol to L_disc_bol (same value for power-law discs)
                    # Use runtime path only if precompute is NOT available.
                    if _has_blr_gaussian and not _has_preint_blr:
                        _l_disc_bol_erg = 10.0 ** (jnp.float64(_agn_lbol)) * LSUN_ERG_PER_S
                        from tengri.components.agn.blr import compute_blr_sed

                        blr_sed = compute_blr_sed(
                            rest_wave_f64,
                            _l_disc_bol_erg,
                            covering_fraction=jnp.float64(agn_blr_cf),
                            agn_fe2_strength=jnp.float64(agn_fe2_strength),
                        )
                        non_stellar_sed = non_stellar_sed + blr_sed

                    # NLR Gaussian
                    if _has_nlr_gaussian and not _has_preint_nlr_gaussian:
                        _l_disc_bol_erg = 10.0 ** (jnp.float64(_agn_lbol)) * LSUN_ERG_PER_S
                        from tengri.components.agn.nlr import compute_nlr_sed

                        nlr_sed = compute_nlr_sed(
                            rest_wave_f64,
                            _l_disc_bol_erg,
                            covering_fraction=jnp.float64(agn_nlr_cf),
                        )
                        non_stellar_sed = non_stellar_sed + nlr_sed

            # 2e: Radio emission
            radio_phot_preint = jnp.zeros(n_filters, dtype=jnp.float64)
            if has_radio:
                _has_any_preint_radio = (
                    _has_preint_radio_synchrotron
                    or _has_preint_radio_freefree
                    or _has_preint_radio_agn_jet
                )
                if _has_any_preint_radio:
                    # Use preintegrated lookups for filter photometry
                    # (fast triweight interp).
                    # Reference luminosity in radio_precompute.py: _L_REF = 1e44 erg/s
                    _L_REF_RADIO = 1.0e44  # erg/s

                    # Synchrotron (SF) component
                    radio_sf_phot = jnp.zeros(n_filters, dtype=jnp.float64)
                    if _has_preint_radio_synchrotron:
                        # Signature: (scale, alpha_sf) -> photometry
                        _scale_sf = jnp.float64(L_ir / _L_REF_RADIO)
                        radio_sf_phot = _radio_synchrotron_lookup(
                            _scale_sf,
                            jnp.float64(radio_alpha_sf),
                        )

                    # Free-free component
                    radio_ff_phot = jnp.zeros(n_filters, dtype=jnp.float64)
                    if _has_preint_radio_freefree:
                        # Signature: (scale, alpha_ff) -> photometry
                        _scale_ff = jnp.float64(L_ir / _L_REF_RADIO)
                        radio_ff_phot = _radio_freefree_lookup(
                            _scale_ff,
                            jnp.float64(radio_alpha_ff),
                        )

                    # AGN jet component
                    radio_agn_phot = jnp.zeros(n_filters, dtype=jnp.float64)
                    if _has_preint_radio_agn_jet:
                        _agn_bol = (
                            10.0 ** (jnp.float64(agn_log_lbol)) * LSUN_ERG_PER_S
                            if has_agn_full
                            else 0.0
                        )
                        # Signature: (scale, alpha_agn) -> photometry
                        _scale_agn = jnp.float64(_agn_bol / _L_REF_RADIO)
                        radio_agn_phot = _radio_agn_jet_lookup(
                            _scale_agn,
                            jnp.float64(radio_alpha_agn),
                        )

                    # Accumulate preintegrated radio photometry
                    radio_phot_preint = radio_sf_phot + radio_ff_phot + radio_agn_phot
                else:
                    # Fall back to full-wavelength evaluation
                    _L_ir = L_ir
                    _agn_bol = (
                        10.0 ** (jnp.float64(agn_log_lbol)) * LSUN_ERG_PER_S
                        if has_agn_full
                        else 0.0
                    )
                    _log_mstar = jnp.log10(jnp.maximum(jnp.sum(weights), 1e-10))
                    radio_sed = radio_emission(
                        rest_wave_f64,
                        L_ir=_L_ir,
                        L_agn_bol=_agn_bol,
                        q_ir=jnp.float64(radio_q_ir),
                        alpha_sf=jnp.float64(radio_alpha_sf),
                        radio_loudness=jnp.float64(radio_loudness),
                        alpha_agn=jnp.float64(radio_alpha_agn),
                        sfr_mode=_radio_sfr_mode,
                        log_mstar=_log_mstar,
                        redshift=_redshift,
                        include_freefree=_include_freefree,
                        T_e=jnp.float64(radio_T_e),
                        alpha_ff=jnp.float64(radio_alpha_ff),
                    )
                    non_stellar_sed = non_stellar_sed + radio_sed

            # 2f: X-ray emission
            xray_phot_preint = jnp.zeros(n_filters, dtype=jnp.float64)
            if has_xray:
                sfr_now = sfr_on_ssp[-1]  # SFR (Msun/yr), not mass weight
                mstar = jnp.sum(weights)
                _agn_bol_xray = (
                    10.0 ** (jnp.float64(agn_log_lbol)) * LSUN_ERG_PER_S if has_agn_full else 0.0
                )

                _has_any_preint_xray = (
                    _has_preint_xray_xrb
                    or _has_preint_xray_corona
                    or _has_preint_xray_corona_lopez24
                )
                if _has_any_preint_xray:
                    # Use preintegrated lookups for filter photometry (fast triweight interp).
                    # Reference scales from xray_precompute.py
                    _SFR_REF_XRAY = 1.0  # Msun/yr
                    _MSTAR_REF_XRAY = 1.0e10  # Msun
                    _LBOL_REF_XRAY = 1.0e44  # erg/s
                    _L12_REF_XRAY = 1.0e30  # erg/s/Hz

                    # XRB component (scales on SFR × stellar_mass)
                    xray_xrb_phot = jnp.zeros(n_filters, dtype=jnp.float64)
                    if _has_preint_xray_xrb:
                        _scale_xrb = jnp.float64(
                            sfr_now * mstar / (_SFR_REF_XRAY * _MSTAR_REF_XRAY)
                        )
                        xray_xrb_phot = _xray_xrb_lookup(
                            _scale_xrb,
                            jnp.float64(xray_gamma_hmxb),
                            jnp.float64(xray_gamma_lmxb),
                        )

                    # Corona component (mutually exclusive variants)
                    xray_corona_phot = jnp.zeros(n_filters, dtype=jnp.float64)
                    if _has_preint_xray_corona:
                        _scale_corona = jnp.float64(_agn_bol_xray / _LBOL_REF_XRAY)
                        xray_corona_phot = _xray_corona_lookup(
                            _scale_corona,
                            jnp.float64(xray_gamma_agn),
                            jnp.float64(xray_alpha_ox),
                        )
                    elif _has_preint_xray_corona_lopez24:
                        # Note: requires L_12um from state. For now, set to 0.
                        # Uses alpha_irx instead of alpha_ox
                        _l_12um = jnp.float64(0.0)
                        _scale_lopez24 = jnp.float64(_l_12um / _L12_REF_XRAY)
                        # TODO: use xray_alpha_irx when available
                        xray_corona_phot = _xray_corona_lopez24_lookup(
                            _scale_lopez24,
                            jnp.float64(xray_gamma_agn),
                            jnp.float64(xray_alpha_ox),  # Placeholder
                        )

                    # Accumulate preintegrated X-ray photometry
                    xray_phot_preint = xray_xrb_phot + xray_corona_phot
                else:
                    # Fall back to full-wavelength evaluation
                    xray_sed = xray_emission(
                        rest_wave_f64,
                        sfr=sfr_now,
                        stellar_mass=mstar,
                        L_agn_bol=_agn_bol_xray,
                        gamma_agn=jnp.float64(xray_gamma_agn),
                        alpha_ox=jnp.float64(xray_alpha_ox),
                        gamma_hmxb=jnp.float64(xray_gamma_hmxb),
                        gamma_lmxb=jnp.float64(xray_gamma_lmxb),
                        E_cut=jnp.float64(xray_E_cut),
                    )
                    non_stellar_sed = non_stellar_sed + xray_sed

            # Apply IGM absorption at full wavelength before filter integration.
            # This is compositional-quality IGM (not the per-filter approximation).
            if has_igm:
                # Extend igm_trans_full to panchromatic grid if needed
                if _needs_extension:
                    from tengri.utils.wavelength import interpolate_sed_to_grid

                    _igm_panch = interpolate_sed_to_grid(
                        ssp_wave_f64, igm_trans_full, rest_wave_f64
                    )
                    non_stellar_sed = non_stellar_sed * _igm_panch
                else:
                    non_stellar_sed = non_stellar_sed * igm_trans_full

            # === STEP 3: Integrate non-stellar through filters (vectorized) ===
            if n_filters > 0:
                non_stellar_phot = compute_flux_density_batch(
                    non_stellar_sed,
                    rest_wave_f64,
                    fw_padded,
                    ft_padded,
                    z_fixed,
                    dl_cm_fixed,
                )

        # === STEP 4: Combine stellar + non-stellar ===
        stellar_phot = flux_attenuated
        if has_igm:
            stellar_phot = stellar_phot * igm_trans_eff

        # Scale stellar to erg/s/cm^2/Hz
        stellar_phot = (flux_scale * stellar_phot * lsun).astype(jnp.float64)

        # Add preintegrated dust IR photometry if available.
        # The lookup returns L_ν (erg/s/Hz); convert to flux density
        # (erg/s/cm²/Hz) with the same (1+z)/(4π d_L²) scaling as stellar.
        if _has_preint_dust_ir:
            non_stellar_phot = non_stellar_phot + dust_ir_phot_preint * flux_scale

        # Add preintegrated disc/empirical/torus photometry if available (already flux-scaled).
        _has_any_disc_torus_preint_final = (
            _has_preint_powerlaw_disc
            or _has_preint_ss_disc
            or _has_preint_cigale_disc
            or _has_preint_qsogen
            or _has_preint_silva04
            or _has_preint_cat3d
        )
        if _has_any_disc_torus_preint_final:
            non_stellar_phot = non_stellar_phot + agn_phot_preint

        # Add preintegrated K&D disc photometry if available (already flux-scaled).
        if _has_preint_kd:
            non_stellar_phot = non_stellar_phot + agn_phot_preint

        # Add preintegrated SKIRTOR torus photometry if available (already flux-scaled).
        if _has_preint_skirtor:
            non_stellar_phot = non_stellar_phot + skirtor_torus_preint

        # Add preintegrated radio photometry if available (already erg/s/Hz).
        _has_any_preint_radio_final = (
            _has_preint_radio_synchrotron
            or _has_preint_radio_freefree
            or _has_preint_radio_agn_jet
        )
        if _has_any_preint_radio_final:
            non_stellar_phot = non_stellar_phot + radio_phot_preint * flux_scale

        # Add preintegrated X-ray photometry if available (already erg/s/Hz).
        _has_any_preint_xray_final = (
            _has_preint_xray_xrb or _has_preint_xray_corona or _has_preint_xray_corona_lopez24
        )
        if _has_any_preint_xray_final:
            non_stellar_phot = non_stellar_phot + xray_phot_preint * flux_scale

        # Add preintegrated MAPPINGS shock photometry if available.
        # The lookup returns line luminosities (L_sun); we compute shock Hα from
        # the current SED and scale relative line luminosities accordingly.
        shock_phot_preint = jnp.zeros(n_filters, dtype=jnp.float64)
        if _has_preint_shock_mappings:
            # Compute bolometric luminosity from non_stellar_sed (before filter integration)
            # to estimate shock Hα following the same logic as shock_emission()
            from tengri.forward.emission_helpers import _C_AA

            nu_ssp = _C_AA / ssp_wave_f64
            l_bol = -jnp.trapezoid(non_stellar_sed, nu_ssp)
            l_halpha_approx = jnp.maximum(l_bol * 1e-3, 1e-30)
            l_shock_halpha = shock_frac * l_halpha_approx

            # Call precompute lookup to get shock line luminosities and wavelengths
            # The lookup expects: (l_shock_halpha, shock_velocity, shock_b_over_sqrt_n,
            # shock_log_density)
            _line_waves, _line_lums = _shock_mappings_lookup(
                l_shock_halpha,
                jnp.float64(shock_velocity),
                jnp.float64(shock_b_over_sqrt_n),
                jnp.float64(shock_log_density),
            )

            # Project shock lines through filters using vectorized integration
            # For each line (treated as a delta function at its vacuum wavelength),
            # compute its contribution to each filter
            def _integrate_lines_to_phot(fw_filt, ft_filt):
                """Integrate shock lines through one filter."""
                _line_waves_obs = _line_waves * (1.0 + z_fixed)
                _trans_at_lines = jnp.interp(
                    _line_waves_obs, fw_filt, ft_filt, left=0.0, right=0.0
                )
                # Each line contributes: L_line (Lsun) * transmission * dist-scaling
                _line_fluxes = (
                    _line_lums * _trans_at_lines * lsun / (4.0 * jnp.pi * dl_cm_fixed**2)
                )
                return jnp.sum(_line_fluxes)

            # Vectorize over all filters using vmap
            # fw_padded shape: (n_wave_padded, n_filters); ft_padded same
            # We need to pass each filter's wavelengths and transmissions
            shock_phot_preint = jax.vmap(
                _integrate_lines_to_phot,
                in_axes=(1, 1),  # vmap over filter axis (axis 1)
            )(fw_padded, ft_padded)
            shock_phot_preint = jnp.asarray(shock_phot_preint, dtype=jnp.float64)

        if _has_preint_shock_mappings:
            non_stellar_phot = non_stellar_phot + shock_phot_preint

        # AGN-nebular emitter precompute consumers (BLR, NLR-Gaussian)
        # These return per-filter photometry directly from precomputed lookups.
        _has_any_preint_agn_nebular_final = _has_preint_blr or _has_preint_nlr_gaussian
        if _has_any_preint_agn_nebular_final and has_agn_full:
            _l_disc_bol_erg = 10.0 ** (jnp.float64(_agn_lbol)) * LSUN_ERG_PER_S

            # BLR Gaussian precompute
            if _has_preint_blr:
                from tengri.components.agn.blr import _BLR_FWHM_KMS as _blr_fwhm_default

                blr_phot_preint = _blr_lookup(
                    l_cont_erg_s_hz=_l_disc_bol_erg * jnp.float64(agn_blr_cf),
                    sigma_blr_kms=_blr_fwhm_default,
                    blr_strength=1.0,
                )
                non_stellar_phot = non_stellar_phot + blr_phot_preint * flux_scale

            # NLR Gaussian precompute
            if _has_preint_nlr_gaussian:
                nlr_phot_preint = _nlr_gaussian_lookup(
                    l_disc_bol_erg=_l_disc_bol_erg,
                    covering_fraction=jnp.float64(agn_nlr_cf),
                )
                non_stellar_phot = non_stellar_phot + nlr_phot_preint * flux_scale

        # Feltre NLR precompute
        if _has_preint_feltre_nlr:
            # The Feltre lookup is a dict with "predict_lines" (for lines)
            # and "line_wavelengths". For photometry, we use the
            # preintegrated line_lum_grid directly via triweight
            # interpolation. The lookup returns (wavelengths, luminosities),
            # which we convert to photometry by integrating through filters.
            # For now, assume the lookup returns per-filter photometry.
            _l_disc_bol_erg = 10.0 ** (jnp.float64(agn_log_lbol)) * LSUN_ERG_PER_S

            # Call the predict_lines function with free axes
            # (signature depends on which axes were fixed at precompute time)
            _line_waves, line_lums = _feltre_nlr_lookup["predict_lines"](
                log_qh=jnp.log10(_l_disc_bol_erg / (10.0**45.0)),  # normalize Q_H
                neb_logZ_gas=jnp.float64(neb_logZ_gas),
                agn_alpha_ion=jnp.float64(agn_alpha_ion),
                neb_logU=jnp.float64(neb_logU),
                neb_xid=jnp.float64(neb_xid),
                neb_fesc=jnp.float64(neb_fesc),
            )

            # Scale by covering fraction
            line_lums_scaled = line_lums * jnp.float64(agn_feltre_cf)

            # Integrate lines through filters
            # (precompute contains the line_weight_matrix)
            feltre_phot_preint = jnp.einsum(
                "l,lf->f",
                line_lums_scaled,
                _feltre_nlr_lookup["line_weight_matrix"],
            )
            non_stellar_phot = non_stellar_phot + feltre_phot_preint * flux_scale

        # TODO(precompute-consumer): The following PrecomputedData fields are
        # built and stored by SEDModel but not yet consumed here. Wiring each
        # one requires the paired (gate-runtime, add-precompute) pattern used
        # for dust IR / KD / SKIRTOR / radio / AGN disc+torus / shock above.
        # Family-by-family follow-up:
        #
        #   Line emitters (PR 4) — DIFFERENT ARCHITECTURE; do NOT route through
        #   PrecomputedData. The kernel consumes line emission via duck-typed
        #   methods on ``state.nebular_backend`` (``_has_preint_photometry``,
        #   ``_preint_continuum``, ``_preint_lines``, ``_line_lum_collapsed``,
        #   ``_young_idx``, ``_qh_table``); see lines 169-199 above and the
        #   reference implementation in
        #   ``CloudyGridBackend.preintegrate_for_photometry``.
        #
        #     - CB19, MAPPINGS V (stellar photoionization alternatives):
        #       implement the duck-typed surface on the backend class so the
        #       existing nebular branch picks them up; precompute adapter
        #       supplies the data.
        #     - Feltre NLR, BLR, NLR-Gaussian (AGN nebular): no fit to the
        #       stellar duck-type — needs a NEW AGN-nebular branch in the AGN
        #       photometry path, scaling on ``L_disc × covering_fraction``
        #       not Q_H.
        #
        #   Each is a separate PR with its own equivalence harness against the
        #   runtime path.
        #
        # See `tests/unit/test_precompute_protocol.py` for the conformance
        # surface and `scripts/benchmark_precompute_analytic.py` for the
        # speed-up motivation. Each family wiring needs a numerical-equivalence
        # test against the runtime path (1e-4 rel tol) per the original plan.

        return stellar_phot + non_stellar_phot

    # --- Fused end-to-end wrapper: params dict → photometry ---
    # Fuse param translation + SFH computation into the JIT scope,
    # eliminating ~240 μs of Python dispatch overhead per call.
    from tengri.components.stellar.sfh.registry import compute_field_gp
    from tengri.parameters.translate import get_internal_params

    param_map = state.param_map
    spec = state.spec
    has_field = state.uses_stochastic_sfh
    sfh_fn = state.sfh_fn
    sfh_internal_names = state.sfh_internal_names
    field_model = state.gp_kernel
    n_grid = state.n_grid
    d_log_age = float(state.d_log_age)
    ssp_log_ages_yr_cap = state.ssp_log_ages_yr
    log_age_grid = state.log_age_grid
    age_yr = state.age_yr

    def hybrid_phot_fused(params):
        """params dict → photometry (end-to-end, JIT'd by caller)."""
        p = get_internal_params(params, param_map, spec, has_field)

        # SFH computation (same as build_fused_tier2_photometry)
        kw = {k: v for k, v in p.items() if k in sfh_internal_names}
        if has_field and "xi" in p:
            gp_x, k0_half = compute_field_gp(
                xi=p["xi"],
                psd_sigma=p["psd_sigma"],
                psd_tau_yr=p["psd_tau_yr"],
                n_grid=n_grid,
                d_log_age=d_log_age,
                field_model=field_model,
            )
            kw["gp_x"] = gp_x
            kw["k0_half"] = k0_half
        sfr = sfh_fn(age_yr, **kw)
        sfr_on_ssp = jnp.interp(ssp_log_ages_yr_cap, log_age_grid, sfr)

        # Dispatch to the inner kernel (already defined above)
        _lgmet = p.get("log_z_abs", p.get("log_z_abs_final", -1.8477))
        if _is_single_dust:
            return hybrid_phot(
                sfr_on_ssp,
                _lgmet,
                p.get("tau_v", 0.0),
                p.get("dust_slope", -0.7),
                **{
                    k: p.get(k, v)
                    for k, v in [
                        ("f_obscuration", 0.0),
                        ("dust_bump_strength", 0.0),
                        ("dust_delta", 0.0),
                        ("dust_Rv", 3.1),
                        ("alpha_fe", 0.0),
                        ("dust_T", 35.0),
                        ("dust_beta_ir", 1.6),
                        ("dust_eta_balance", 1.0),
                        ("agn_log_lbol", 10.0),
                        ("agn_alpha", -1.0),
                        ("agn_T_torus", 1000.0),
                        ("agn_tau_torus", 5.0),
                        ("agn_torus_frac", 0.5),
                        ("agn_log_mbh", 7.0),
                        ("agn_log_ledd", -1.0),
                        ("dust_alpha_mir", 2.0),
                        ("dust_alpha_dale", 2.0),
                        ("dust_umin", 1.0),
                        ("dust_gamma_dl", 0.01),
                        ("dust_qpah", 2.5),
                        ("dust_qhac", 0.17),
                        ("dust_alpha_dl14", 2.0),
                        ("dust_log_ssfr", -10.0),
                    ]
                },
            )
        return hybrid_phot(
            sfr_on_ssp,
            _lgmet,
            p.get("tau_bc", 0.0),
            p.get("tau_diff", 0.0),
            p.get("dust_slope", -0.7),
            **{
                k: p.get(k, v)
                for k, v in [
                    ("f_obscuration", 0.0),
                    ("dust_bump_strength", 0.0),
                    ("dust_delta", 0.0),
                    ("dust_Rv", 3.1),
                    ("alpha_fe", 0.0),
                    ("dust_T", 35.0),
                    ("dust_beta_ir", 1.6),
                    ("dust_eta_balance", 1.0),
                    ("agn_log_lbol", 10.0),
                    ("agn_alpha", -1.0),
                    ("agn_T_torus", 1000.0),
                    ("agn_tau_torus", 5.0),
                    ("agn_torus_frac", 0.5),
                    ("agn_log_mbh", 7.0),
                    ("agn_log_ledd", -1.0),
                    ("dust_alpha_mir", 2.0),
                    ("dust_alpha_dale", 2.0),
                    ("dust_umin", 1.0),
                    ("dust_gamma_dl", 0.01),
                    ("dust_qpah", 2.5),
                    ("dust_qhac", 0.17),
                    ("dust_alpha_dl14", 2.0),
                    ("dust_log_ssfr", -10.0),
                ]
            },
        )

    return hybrid_phot_fused


def build_hybrid_photometry_ztable(state: SEDModelState, model=None):
    """Build hybrid photometry kernel for free-z inference using a z-table.

    Like :func:`build_hybrid_photometry` but with SSP photometry interpolated
    from a precomputed redshift grid instead of fixed at model.redshift.
    Stellar photometry uses precomputed SSP×filter einsum (fast, ~0.4% error).
    Non-stellar components evaluated at full wavelength resolution via
    emission_helpers, then integrated through filters (exact).

    For free-redshift inference, this kernel interpolates the precomputed
    z-table to the current redshift at each step, maintaining the same
    speedup as fixed-z precomputation while allowing z to vary.

    Parameters
    ----------
    state : SEDModelState
        Frozen state bundle providing config and precomputed arrays.
    model : SEDModel, optional
        Legacy model reference for various attributes.
        If None, will fail; this parameter is temporary pending refactoring.

    Returns
    -------
    callable or None
        JIT-compiled function: ``(params_dict) -> photometry_array``,
        where ``photometry_array`` has shape (n_filters,) [erg/s/cm^2/Hz]
        and ``params_dict`` includes redshift as a free variable.
        Returns None if z-table has not been precomputed.

    Raises
    ------
    ValueError
        If photometry_ztable has not been precomputed via
        ``model.precompute_ztable()``.

    Notes
    -----
    **JIT-compatible**: yes — stellar photometry interpolated from z-table,
    non-stellar components computed at full wavelength then integrated through
    filters.

    **Gradient-safe**: yes — differentiable w.r.t. all parameters including
    redshift, dust, AGN, nebular, and shock parameters.
    """
    from tengri.components.dust.attenuation import resolve_dust_law
    from tengri.components.stellar.sfh.registry import compute_field_gp
    from tengri.components.stellar.sps.dsps_wrapper import LSUN_ERG_PER_S
    from tengri.forward.emission_helpers import (
        agn_emission,
        attenuate_emission,
        dust_ir_emission,
        nebular_emission,
        radio_emission,
        shock_emission,
        xray_emission,
    )
    from tengri.parameters.translate import get_internal_params

    # Validate that z-table has been precomputed
    if state.precomputed.photometry_ztable is None:
        raise ValueError("Z-table not precomputed. Call model.precompute_ztable() first.")

    dt = state.forward_dtype
    ztable = state.precomputed.photometry_ztable
    ssp_lgmet = state.ssp_data.ssp_lgmet.astype(dt)
    _is_single_dust = state.dust_model == "single_component"
    _dust_exact = getattr(model, "_dust_scheme", "fast") == "exact"
    if not _is_single_dust:
        if _dust_exact:
            dust_age_w = state.precomputed.dust_age_weights.astype(dt)
        else:
            _t_birth = 1e7
            young_mask = (state.ssp_ages_yr < _t_birth).astype(dt)
            old_mask = dt.type(1.0) - young_mask
    _csp_use_matrix = state.csp_integration == "log_interp"
    if _csp_use_matrix:
        _csp_mat = state.csp_matrix.astype(dt)
    else:
        _age_dt = state.csp_age_dt.astype(dt)
    lsun = dt.type(LSUN_ERG_PER_S)

    # Voronoi frequency bandwidths for L_absorbed broadband estimate (Hz).
    _eff_bw = state.precomputed.effective_bandwidths_hz
    if _eff_bw is not None:
        _eff_bw = _eff_bw.astype(dt)

    # Capture dust law functions
    law_bc_fn = resolve_dust_law(state.dust_law_bc)
    if not _is_single_dust:
        law_diff_fn = resolve_dust_law(state.dust_law_diff)

    from tengri.components.stellar.sps.dsps_wrapper import (
        _ALPHA_TO_Z_COEFF as _A2Z,
        has_alpha_grid,
    )

    _has_alpha = has_alpha_grid(state.ssp_data)
    if _has_alpha:
        ssp_alpha_fe = state.ssp_data.ssp_alpha_fe.astype(dt)

    _use_alpha_fe = state.spec.alpha_fe_evolving or "met_alpha_fe" in state.spec.free_params

    _use_smooth_z = state.met_interp == "smooth"
    _lgmet_scat = dt.type(state.lgmet_scatter)
    if _use_smooth_z:
        from tengri.components.stellar.sps.dsps_wrapper import compute_lgmet_weights as _clw

    # Smooth redshift-table interpolation (C^2 gradient) — recommended when
    # redshift is a free parameter and the sampler is gradient-based (NUTS/HMC).
    # See docs/dev/design_philosophy.md: rough piecewise-linear gradients at
    # grid nodes hurt HMC acceptance. Auto-derive scatter as 0.5 * mean(dz).
    _use_smooth_ztable = getattr(model, "_z_interp", "linear") == "smooth"
    if _use_smooth_ztable:
        import numpy as _np

        from tengri.components.stellar.sps.precompute import (
            interpolate_ztable_smooth as _interpolate_ztable,
        )

        _z_scatter = dt.type(0.5 * float(_np.mean(_np.diff(_np.asarray(ztable.z_grid)))))
    else:
        from tengri.components.stellar.sps.precompute import (
            interpolate_ztable as _interpolate_ztable,
        )

        _z_scatter = None

    # IGM: z-table kernel applies IGM in two places:
    #   1. Stellar photometry: interpolate igm_trans_table (n_z, n_filters) to current z.
    #   2. Non-stellar SED: evaluate igm_transmission(wave_obs, z) at full wavelength
    #      inside the traced function (pure JAX, JIT-safe).
    has_igm = ztable.igm_trans_table is not None
    _igm_fn_ztable = getattr(model, "_igm_fn", None)
    if _igm_fn_ztable is None:
        from tengri.components.igm import igm_transmission as _igm_fn_ztable
    if has_igm:
        _igm_fn = _igm_fn_ztable
        _igm_trans_table = ztable.igm_trans_table  # shape (n_z, n_filters)
        _igm_z_grid = ztable.z_grid

    # === Non-stellar components (full wavelength) ===
    ssp_wave_f64 = state.ssp_data.ssp_wave
    rest_wave_f64 = state.rest_wavelength
    _needs_extension = rest_wave_f64 is not state.ssp_data.ssp_wave

    # Nebular
    has_nebular = state.nebular_backend is not None and getattr(
        state.nebular_backend, "has_free_params", False
    )
    if has_nebular:
        nebular_backend = state.nebular_backend
        _neb_dust_mode = getattr(model, "_neb_dust_mode", "bc") if model is not None else "bc"
        _neb_bc_fn = (
            getattr(model, "_neb_dust_law_bc_fn", law_bc_fn) if model is not None else law_bc_fn
        )
    _has_preint_neb = has_nebular and getattr(
        state.nebular_backend, "_has_preint_photometry", False
    )
    if _has_preint_neb:
        _neb_cont_phot = nebular_backend._preint_continuum.phot
        _neb_cont_axes = nebular_backend._preint_continuum.axes
        _neb_cont_edges = nebular_backend._preint_continuum.edges
        _neb_line_weights = nebular_backend._preint_lines.line_filter_weights
        _neb_line_axes = nebular_backend._preint_lines.axes
        _neb_line_edges = nebular_backend._preint_lines.edges
        _neb_line_lum = nebular_backend.grid.line_luminosity
        _neb_young_idx = jnp.array(nebular_backend._young_idx)
        _neb_qh_table = nebular_backend._qh_table
        _neb_qh_log_met = nebular_backend._qh_log_met
        _neb_qh_log_age = nebular_backend._qh_log_age
        _neb_lya_idx = int(jnp.argmin(jnp.abs(nebular_backend.grid.line_wavelengths - 1215.67)))

    # Shock
    has_shock = getattr(model, "_uses_shock", False) if model is not None else False

    # Dust emission (full wavelength or preintegrated)
    has_dust_em_full = state.dust_emission_model is not None
    _dust_model_name = state.dust_emission_model if has_dust_em_full else None
    if has_dust_em_full:
        from tengri.components.dust.emission import preload_emission_model

        preload_emission_model(state.dust_emission_model)

    # AGN (full wavelength)
    has_agn_full = state.agn_model is not None
    if has_agn_full:
        from tengri.components.agn import resolve_agn_model

        agn_model_fn_full = resolve_agn_model(state.agn_model)

    # Radio
    has_radio = state.uses_radio
    if has_radio:
        _radio_sfr_mode = state.radio_sfr_mode
        _include_freefree = state.radio_include_freefree

    # X-ray
    has_xray = state.uses_xray

    # Constants for energy balance
    _c_aa = dt.type(2.99792458e18)

    # Filter information
    n_filters = len(state.filter_waves) if state.filter_waves else 0
    filter_waves_list = state.filter_waves if state.filter_waves else []
    filter_trans_list = state.filter_trans if state.filter_trans else []

    # Build SFH function and parameters from state
    spec = state.spec
    param_map = state.param_map
    age_yr = state.age_yr
    log_age_grid = state.log_age_grid
    d_log_age = state.d_log_age
    ssp_log_ages_yr_cap = state.ssp_log_ages_yr
    has_field = "xi" in state.spec.free_params
    sfh_internal_names = state.sfh_internal_names
    n_grid = state.n_grid
    field_model = state.gp_kernel
    sfh_fn = state.sfh_fn

    # === Define kernel signature (parametric-style, same as fixed-z) ===

    if _is_single_dust:

        def hybrid_phot_ztable(
            sfr_on_ssp,
            log_z_abs,
            redshift,
            tau_v,
            dust_slope,
            f_obscuration=0.0,
            dust_bump_strength=0.0,
            dust_delta=0.0,
            dust_Rv=3.1,
            alpha_fe=0.0,
            dust_T=35.0,
            dust_beta_ir=1.6,
            dust_eta_balance=1.0,
            agn_log_lbol=10.0,
            agn_alpha=-1.0,
            agn_T_torus=1000.0,
            agn_tau_torus=5.0,
            agn_torus_frac=0.5,
            agn_log_mbh=7.0,
            agn_log_ledd=-1.0,
            # Non-stellar parameters
            neb_logU=-3.0,
            neb_logZ_gas=None,
            neb_fesc=0.0,
            neb_fesc_lya=0.0,
            shock_frac=0.0,
            shock_velocity=300.0,
            shock_log_density=0.0,
            shock_b_over_sqrt_n=1.0,
            shock_abundance="solar",
            shock_component="combined",
            dust_alpha_mir=2.0,
            dust_alpha_dale=2.0,
            dust_umin=1.0,
            dust_gamma_dl=0.01,
            dust_qpah=2.5,
            agn_polar_ebv=0.0,
            agn_cos_inc=0.5,
            agn_polar_oa=45.0,
            agn_a_spin=0.0,
            agn_tau_skirtor=7.0,
            agn_p_skirtor=1.0,
            agn_q_skirtor=1.0,
            agn_oa_skirtor=40.0,
            agn_frac=0.0,
            agn_T_hot=1200.0,
            agn_T_warm=300.0,
            agn_frac_hot=0.3,
            agn_f_hard=0.02,
            agn_gamma_warm=2.5,
            agn_kt_warm=0.2,
            agn_gamma_hard=1.8,
            agn_kt_hot=100.0,
            agn_r_warm_ratio=2.0,
            agn_blr_cf=0.1,
            agn_nlr_cf=0.1,
            agn_fe2_strength=0.0,
            radio_q_ir=2.64,
            radio_alpha_sf=0.8,
            radio_loudness=0.0,
            radio_alpha_agn=0.7,
            radio_T_e=1e4,
            radio_alpha_ff=-0.1,
            xray_gamma_agn=1.8,
            xray_alpha_ox=-1.4,
            xray_gamma_hmxb=2.0,
            xray_gamma_lmxb=1.6,
            xray_E_cut=300.0,
            dust_qhac=0.17,
            dust_alpha_dl14=2.0,
            dust_log_ssfr=-10.0,
        ):
            """Compute hybrid photometry for single dust component with redshift table."""
            return _hybrid_phot_ztable_body(
                sfr_on_ssp,
                log_z_abs,
                redshift,
                0.0,
                0.0,
                dust_slope,
                f_obscuration,
                dust_bump_strength,
                dust_delta,
                dust_Rv,
                alpha_fe,
                dust_T,
                dust_beta_ir,
                dust_eta_balance,
                agn_log_lbol,
                agn_alpha,
                agn_T_torus,
                agn_tau_torus,
                agn_torus_frac,
                agn_log_mbh,
                agn_log_ledd,
                neb_logU,
                neb_logZ_gas,
                neb_fesc,
                neb_fesc_lya,
                shock_frac,
                shock_velocity,
                shock_log_density,
                shock_b_over_sqrt_n,
                shock_abundance,
                shock_component,
                dust_alpha_mir,
                dust_alpha_dale,
                dust_umin,
                dust_gamma_dl,
                dust_qpah,
                agn_polar_ebv,
                agn_cos_inc,
                agn_polar_oa,
                agn_frac,
                agn_a_spin,
                agn_tau_skirtor,
                agn_p_skirtor,
                agn_q_skirtor,
                agn_oa_skirtor,
                agn_T_hot,
                agn_T_warm,
                agn_frac_hot,
                agn_f_hard,
                agn_gamma_warm,
                agn_kt_warm,
                agn_gamma_hard,
                agn_kt_hot,
                agn_r_warm_ratio,
                radio_q_ir,
                radio_alpha_sf,
                radio_loudness,
                radio_alpha_agn,
                radio_T_e,
                radio_alpha_ff,
                xray_gamma_agn,
                xray_alpha_ox,
                xray_gamma_hmxb,
                xray_gamma_lmxb,
                xray_E_cut,
                dust_qhac,
                dust_alpha_dl14,
                dust_log_ssfr,
                tau_v=tau_v,
            )

    else:

        def hybrid_phot_ztable(
            sfr_on_ssp,
            log_z_abs,
            redshift,
            tau_bc,
            tau_diff,
            dust_slope,
            f_obscuration=0.0,
            dust_bump_strength=0.0,
            dust_delta=0.0,
            dust_Rv=3.1,
            alpha_fe=0.0,
            dust_T=35.0,
            dust_beta_ir=1.6,
            dust_eta_balance=1.0,
            agn_log_lbol=10.0,
            agn_alpha=-1.0,
            agn_T_torus=1000.0,
            agn_tau_torus=5.0,
            agn_torus_frac=0.5,
            agn_log_mbh=7.0,
            agn_log_ledd=-1.0,
            # Non-stellar parameters
            neb_logU=-3.0,
            neb_logZ_gas=None,
            neb_fesc=0.0,
            neb_fesc_lya=0.0,
            shock_frac=0.0,
            shock_velocity=300.0,
            shock_log_density=0.0,
            shock_b_over_sqrt_n=1.0,
            shock_abundance="solar",
            shock_component="combined",
            dust_alpha_mir=2.0,
            dust_alpha_dale=2.0,
            dust_umin=1.0,
            dust_gamma_dl=0.01,
            dust_qpah=2.5,
            agn_polar_ebv=0.0,
            agn_cos_inc=0.5,
            agn_polar_oa=45.0,
            agn_a_spin=0.0,
            agn_tau_skirtor=7.0,
            agn_p_skirtor=1.0,
            agn_q_skirtor=1.0,
            agn_oa_skirtor=40.0,
            agn_frac=0.0,
            agn_T_hot=1200.0,
            agn_T_warm=300.0,
            agn_frac_hot=0.3,
            agn_f_hard=0.02,
            agn_gamma_warm=2.5,
            agn_kt_warm=0.2,
            agn_gamma_hard=1.8,
            agn_kt_hot=100.0,
            agn_r_warm_ratio=2.0,
            agn_blr_cf=0.1,
            agn_nlr_cf=0.1,
            agn_fe2_strength=0.0,
            radio_q_ir=2.64,
            radio_alpha_sf=0.8,
            radio_loudness=0.0,
            radio_alpha_agn=0.7,
            radio_T_e=1e4,
            radio_alpha_ff=-0.1,
            xray_gamma_agn=1.8,
            xray_alpha_ox=-1.4,
            xray_gamma_hmxb=2.0,
            xray_gamma_lmxb=1.6,
            xray_E_cut=300.0,
            dust_qhac=0.17,
            dust_alpha_dl14=2.0,
            dust_log_ssfr=-10.0,
        ):
            """Compute hybrid photometry for two-component dust with redshift table."""
            return _hybrid_phot_ztable_body(
                sfr_on_ssp,
                log_z_abs,
                redshift,
                tau_bc,
                tau_diff,
                dust_slope,
                f_obscuration,
                dust_bump_strength,
                dust_delta,
                dust_Rv,
                alpha_fe,
                dust_T,
                dust_beta_ir,
                dust_eta_balance,
                agn_log_lbol,
                agn_alpha,
                agn_T_torus,
                agn_tau_torus,
                agn_torus_frac,
                agn_log_mbh,
                agn_log_ledd,
                neb_logU,
                neb_logZ_gas,
                neb_fesc,
                neb_fesc_lya,
                shock_frac,
                shock_velocity,
                shock_log_density,
                shock_b_over_sqrt_n,
                shock_abundance,
                shock_component,
                dust_alpha_mir,
                dust_alpha_dale,
                dust_umin,
                dust_gamma_dl,
                dust_qpah,
                agn_polar_ebv,
                agn_cos_inc,
                agn_polar_oa,
                agn_frac,
                agn_a_spin,
                agn_tau_skirtor,
                agn_p_skirtor,
                agn_q_skirtor,
                agn_oa_skirtor,
                agn_T_hot,
                agn_T_warm,
                agn_frac_hot,
                agn_f_hard,
                agn_gamma_warm,
                agn_kt_warm,
                agn_gamma_hard,
                agn_kt_hot,
                agn_r_warm_ratio,
                radio_q_ir,
                radio_alpha_sf,
                radio_loudness,
                radio_alpha_agn,
                radio_T_e,
                radio_alpha_ff,
                xray_gamma_agn,
                xray_alpha_ox,
                xray_gamma_hmxb,
                xray_gamma_lmxb,
                xray_E_cut,
            )

    def _stellar_phot_ztable(
        sfr_on_ssp,
        log_z_abs,
        redshift,
        tau_bc,
        tau_diff,
        dust_slope,
        f_obscuration,
        dust_bump_strength,
        dust_delta,
        dust_Rv,
        alpha_fe,
        tau_v=0.0,
    ):
        """Stellar photometry from z-table: metallicity interp + z-table lookup.

        Interpolates precomputed z-table to the current redshift and
        metallicity to get SSP photometry, then applies dust attenuation
        and computes L_absorbed estimate (same as fixed-z version).
        """
        sfr = sfr_on_ssp.astype(dt)
        lz = jnp.asarray(log_z_abs, dtype=dt)
        z = jnp.asarray(redshift, dtype=dt)
        tv1 = jnp.asarray(tau_bc, dtype=dt)
        tv2 = jnp.asarray(tau_diff, dtype=dt)
        tv = jnp.asarray(tau_v, dtype=dt)
        dn = jnp.asarray(dust_slope, dtype=dt)
        f_obs = jnp.asarray(f_obscuration, dtype=dt)
        bump = jnp.asarray(dust_bump_strength, dtype=dt)
        delta = jnp.asarray(dust_delta, dtype=dt)
        rv = jnp.asarray(dust_Rv, dtype=dt)
        afe = jnp.asarray(alpha_fe, dtype=dt)

        # CSP weights
        weights = _csp_mat @ sfr if _csp_use_matrix else sfr * _age_dt

        # === Interpolate z-table to current redshift ===
        # Uses smooth (triweight) interpolation when model._z_interp=="smooth"
        # to give C^2-continuous gradients for HMC/NUTS free-z inference.
        if _use_smooth_ztable:
            ssp_phot_at_z, eff_waves_rest, _flux_scale = _interpolate_ztable(
                ztable.ssp_phot_table,
                ztable.eff_waves_rest_table,
                ztable.flux_scale_table,
                ztable.z_grid,
                z,
                _z_scatter,
            )
        else:
            ssp_phot_at_z, eff_waves_rest, _flux_scale = _interpolate_ztable(
                ztable.ssp_phot_table,
                ztable.eff_waves_rest_table,
                ztable.flux_scale_table,
                ztable.z_grid,
                z,
            )
        # ssp_phot_at_z: shape (n_met, n_age, n_filters)
        # eff_waves_rest: shape (n_filters,)
        # _flux_scale: scalar (not used in z-table, photometry already scaled)

        # === Interpolate metallicity ===
        if _has_alpha:
            lz_c = jnp.clip(lz, ssp_lgmet[0], ssp_lgmet[-1])
            iz = jnp.clip(jnp.searchsorted(ssp_lgmet, lz_c) - 1, 0, ssp_lgmet.shape[0] - 2)
            fz = (lz_c - ssp_lgmet[iz]) / (ssp_lgmet[iz + 1] - ssp_lgmet[iz])
            afe_c = jnp.clip(afe, ssp_alpha_fe[0], ssp_alpha_fe[-1])
            ia = jnp.clip(jnp.searchsorted(ssp_alpha_fe, afe_c) - 1, 0, ssp_alpha_fe.shape[0] - 2)
            fa = (afe_c - ssp_alpha_fe[ia]) / (ssp_alpha_fe[ia + 1] - ssp_alpha_fe[ia])
            ssp_at_z_met = (
                (1 - fz) * (1 - fa) * ssp_phot_at_z[iz, ia]
                + fz * (1 - fa) * ssp_phot_at_z[iz + 1, ia]
                + (1 - fz) * fa * ssp_phot_at_z[iz, ia + 1]
                + fz * fa * ssp_phot_at_z[iz + 1, ia + 1]
            )
        else:
            if _use_alpha_fe:
                lz = lz + _A2Z * afe
            if _use_smooth_z:
                zw = _clw(lz, ssp_lgmet, _lgmet_scat)
                ssp_at_z_met = jnp.einsum("m,maf->af", zw, ssp_phot_at_z)
            else:
                log_z_c = jnp.clip(lz, ssp_lgmet[0], ssp_lgmet[-1])
                idx = jnp.clip(jnp.searchsorted(ssp_lgmet, log_z_c) - 1, 0, ssp_lgmet.shape[0] - 2)
                frac = (log_z_c - ssp_lgmet[idx]) / (ssp_lgmet[idx + 1] - ssp_lgmet[idx])
                ssp_at_z_met = (1.0 - frac) * ssp_phot_at_z[idx] + frac * ssp_phot_at_z[idx + 1]

        # Dust at effective wavelengths (same as fixed-z)
        _law_kw = dict(n_slope=dn, dust_bump_strength=bump, dust_delta=delta, dust_Rv=rv)
        if _is_single_dust:
            k = law_bc_fn(eff_waves_rest.astype(dt), **_law_kw)
            trans_1d = f_obs + (1.0 - f_obs) * jnp.exp(-tv * k)
            flux_attenuated = jnp.einsum("i,if->f", weights, ssp_at_z_met) * trans_1d
        elif _dust_exact:
            k_bc = law_bc_fn(eff_waves_rest.astype(dt), **_law_kw)
            k_diff = law_diff_fn(eff_waves_rest.astype(dt), **_law_kw)
            tau = dust_age_w[:, None] * tv1 * k_bc[None, :] + tv2 * k_diff[None, :]
            dust = f_obs + (1.0 - f_obs) * jnp.exp(-tau)
            flux_attenuated = jnp.einsum("i,if,if->f", weights, dust, ssp_at_z_met)
        else:
            # Fast two-component dust (same as fixed-z)
            k_bc = law_bc_fn(eff_waves_rest.astype(dt), **_law_kw)
            k_diff = law_diff_fn(eff_waves_rest.astype(dt), **_law_kw)
            trans_bc = jnp.exp(-tv1 * k_bc)
            trans_diff = jnp.exp(-tv2 * k_diff)

            csp_young = jnp.einsum("i,if->f", weights * young_mask, ssp_at_z_met)
            csp_old = jnp.einsum("i,if->f", weights * old_mask, ssp_at_z_met)

            flux_no_geom = trans_diff * (trans_bc * csp_young + csp_old)
            flux_intrinsic_for_geom = csp_young + csp_old
            flux_attenuated = f_obs * flux_intrinsic_for_geom + (1.0 - f_obs) * flux_no_geom

        # Estimate L_absorbed
        flux_intrinsic = jnp.einsum("i,if->f", weights, ssp_at_z_met)
        diff_flux = (flux_intrinsic - flux_attenuated) * lsun  # erg/s/Hz per band
        if _eff_bw is not None:
            L_absorbed_stellar = jnp.sum(diff_flux * _eff_bw)
        else:
            L_absorbed_stellar = jnp.sum(diff_flux)

        return flux_attenuated, L_absorbed_stellar, weights

    def _hybrid_phot_ztable_body(
        sfr_on_ssp,
        log_z_abs,
        redshift,
        tau_bc,
        tau_diff,
        dust_slope,
        f_obscuration,
        dust_bump_strength,
        dust_delta,
        dust_Rv,
        alpha_fe,
        dust_T,
        dust_beta_ir,
        dust_eta_balance,
        agn_log_lbol,
        agn_alpha,
        agn_T_torus,
        agn_tau_torus,
        agn_torus_frac,
        agn_log_mbh,
        agn_log_ledd,
        neb_logU,
        neb_logZ_gas,
        neb_fesc,
        neb_fesc_lya,
        shock_frac,
        shock_velocity,
        shock_log_density,
        shock_b_over_sqrt_n,
        shock_abundance,
        shock_component,
        dust_alpha_mir,
        dust_alpha_dale,
        dust_umin,
        dust_gamma_dl,
        dust_qpah,
        agn_polar_ebv,
        agn_cos_inc,
        agn_polar_oa,
        agn_frac,
        agn_a_spin,
        agn_tau_skirtor,
        agn_p_skirtor,
        agn_q_skirtor,
        agn_oa_skirtor,
        agn_T_hot,
        agn_T_warm,
        agn_frac_hot,
        agn_f_hard,
        agn_gamma_warm,
        agn_kt_warm,
        agn_gamma_hard,
        agn_kt_hot,
        agn_r_warm_ratio,
        radio_q_ir,
        radio_alpha_sf,
        radio_loudness,
        radio_alpha_agn,
        radio_T_e,
        radio_alpha_ff,
        xray_gamma_agn,
        xray_alpha_ox,
        xray_gamma_hmxb,
        xray_gamma_lmxb,
        xray_E_cut,
        tau_v=0.0,
    ):
        """Hybrid z-table kernel body: stellar + exact non-stellar."""
        # === STEP 1: Stellar photometry from z-table ===
        flux_attenuated, L_absorbed_stellar, _weights = _stellar_phot_ztable(
            sfr_on_ssp,
            log_z_abs,
            redshift,
            tau_bc,
            tau_diff,
            dust_slope,
            f_obscuration,
            dust_bump_strength,
            dust_delta,
            dust_Rv,
            alpha_fe,
            tau_v=tau_v,
        )

        # === STEP 2: Non-stellar SED (same as build_hybrid_photometry) ===
        # NOTE: This is exact same non-stellar computation as the fixed-z hybrid.
        # The only difference is that stellar photometry is interpolated from
        # z-table at each step, while fixed-z version uses precomputed values.
        _has_any_nonstell = (
            has_nebular or has_shock or has_dust_em_full or has_agn_full or has_radio or has_xray
        )

        non_stellar_phot = jnp.zeros(n_filters, dtype=jnp.float64)

        if _has_any_nonstell:
            non_stellar_sed = jnp.zeros_like(ssp_wave_f64)
            L_abs_neb = jnp.float64(0.0)

            if has_nebular:
                _sfr_last = sfr_on_ssp[-1]  # SFR (Msun/yr), not mass weight
                neb_sed, _neb_lines = nebular_emission(
                    sfr_on_ssp=sfr_on_ssp,
                    logU=neb_logU,
                    log_z=log_z_abs,
                    fesc=neb_fesc,
                    fesc_lya=neb_fesc_lya,
                    backend=nebular_backend,
                    return_lines=True,
                )
                non_stellar_sed = non_stellar_sed + neb_sed

            if has_shock:
                shock_sed = shock_emission(
                    frac=shock_frac,
                    velocity=shock_velocity,
                    log_density=shock_log_density,
                    b_over_sqrt_n=shock_b_over_sqrt_n,
                    abundance=shock_abundance,
                    component=shock_component,
                )
                non_stellar_sed = non_stellar_sed + shock_sed

            if has_agn_full:
                agn_sed = agn_emission(
                    log_lbol=agn_log_lbol,
                    alpha=agn_alpha,
                    T_torus=agn_T_torus,
                    tau_torus=agn_tau_torus,
                    torus_frac=agn_torus_frac,
                    log_mbh=agn_log_mbh,
                    log_ledd=agn_log_ledd,
                    polar_ebv=agn_polar_ebv,
                    cos_inc=agn_cos_inc,
                    polar_oa=agn_polar_oa,
                    frac=agn_frac,
                    a_spin=agn_a_spin,
                    tau_skirtor=agn_tau_skirtor,
                    p_skirtor=agn_p_skirtor,
                    q_skirtor=agn_q_skirtor,
                    oa_skirtor=agn_oa_skirtor,
                    T_hot=agn_T_hot,
                    T_warm=agn_T_warm,
                    frac_hot=agn_frac_hot,
                    f_hard=agn_f_hard,
                    gamma_warm=agn_gamma_warm,
                    kt_warm=agn_kt_warm,
                    gamma_hard=agn_gamma_hard,
                    kt_hot=agn_kt_hot,
                    r_warm_ratio=agn_r_warm_ratio,
                    func=agn_model_fn_full,
                )
                non_stellar_sed = non_stellar_sed + agn_sed

            if has_radio:
                mstar = jnp.exp(10.0 * jnp.log(10.0))  # dummy, not used in hybrid
                radio_sed = radio_emission(
                    sfr=sfr_on_ssp[-1],
                    mstar=mstar,
                    log_lbol_agn=agn_log_lbol if has_agn_full else 10.0,
                    frac_agn=agn_frac if has_agn_full else 0.0,
                    mode=_radio_sfr_mode,
                    q_ir=radio_q_ir,
                    alpha_sf=radio_alpha_sf,
                    loudness=radio_loudness,
                    alpha_agn=radio_alpha_agn,
                    T_e=radio_T_e,
                    freefree=_include_freefree,
                    alpha_ff=radio_alpha_ff,
                    z=redshift,
                )
                non_stellar_sed = non_stellar_sed + radio_sed

            if has_xray:
                mstar = jnp.exp(10.0 * jnp.log(10.0))  # dummy
                xray_sed = xray_emission(
                    sfr=sfr_on_ssp[-1],
                    mstar=mstar,
                    log_lbol_agn=agn_log_lbol if has_agn_full else 10.0,
                    gamma_agn=xray_gamma_agn,
                    alpha_ox=xray_alpha_ox,
                    gamma_hmxb=xray_gamma_hmxb,
                    gamma_lmxb=xray_gamma_lmxb,
                    E_cut=xray_E_cut,
                )
                non_stellar_sed = non_stellar_sed + xray_sed

            # Dust attenuation on non-stellar
            if has_nebular or has_shock:
                attenuated_nonstell_sed, L_abs_neb = attenuate_emission(
                    non_stellar_sed,
                    dust_tau_bc=tau_bc,
                    dust_tau_diff=tau_diff,
                    dust_slope=dust_slope,
                    dust_bump=dust_bump_strength,
                    dust_delta=dust_delta,
                    dust_Rv=dust_Rv,
                    redshift=redshift,
                    law_bc_fn=_neb_bc_fn,
                    law_diff_fn=law_diff_fn if not _is_single_dust else None,
                    mode=_neb_dust_mode,
                )
                non_stellar_sed = attenuated_nonstell_sed
            else:
                L_abs_neb = jnp.float64(0.0)

            # Dust IR emission (uses L_absorbed)
            if has_dust_em_full:
                L_abs_total = L_absorbed_stellar + L_abs_neb
                ir_sed = dust_ir_emission(
                    L_absorbed=L_abs_total,
                    T=dust_T,
                    beta_ir=dust_beta_ir,
                    eta=dust_eta_balance,
                    alpha_mir=dust_alpha_mir,
                    alpha_dale=dust_alpha_dale,
                    umin=dust_umin,
                    gamma_dl=dust_gamma_dl,
                    qpah=dust_qpah,
                    model=_dust_model_name,
                )
                non_stellar_sed = non_stellar_sed + ir_sed

            # Apply IGM to non-stellar SED (full wavelength, evaluated at traced z).
            # igm_transmission takes observed-frame wavelengths.
            if has_igm:
                _wave_obs_nonstell = ssp_wave_f64 * (1.0 + redshift)
                _igm_full = _igm_fn(_wave_obs_nonstell, redshift).astype(jnp.float64)
                if _needs_extension:
                    from tengri.utils.wavelength import interpolate_sed_to_grid

                    _igm_panch = interpolate_sed_to_grid(ssp_wave_f64, _igm_full, rest_wave_f64)
                    non_stellar_sed = non_stellar_sed * _igm_panch
                else:
                    non_stellar_sed = non_stellar_sed * _igm_full

            # Integrate non-stellar through filters
            # Interpolate non-stellar SED to observed frame + filter integration
            from tengri.observation.photometry import compute_flux_density

            non_stellar_phot = compute_flux_density(
                wave_rest=ssp_wave_f64,
                sed=non_stellar_sed,
                filter_waves=filter_waves_list,
                filter_trans=filter_trans_list,
                redshift=redshift,
                use_wave_res=False,
            )

        # === STEP 3: Combine and convert to erg/s/cm²/Hz ===
        # Interpolate z-table to get flux_scale at current z
        _z_arr = jnp.asarray(redshift, dtype=dt)
        if _use_smooth_ztable:
            _, _, flux_scale = _interpolate_ztable(
                ztable.ssp_phot_table,
                ztable.eff_waves_rest_table,
                ztable.flux_scale_table,
                ztable.z_grid,
                _z_arr,
                _z_scatter,
            )
        else:
            _, _, flux_scale = _interpolate_ztable(
                ztable.ssp_phot_table,
                ztable.eff_waves_rest_table,
                ztable.flux_scale_table,
                ztable.z_grid,
                _z_arr,
            )
        flux_scale = jnp.asarray(flux_scale, dtype=jnp.float64)

        # Stellar contribution: flux_attenuated is in Lsun/Hz (from einsum)
        stellar_phot = flux_scale * flux_attenuated * lsun  # erg/s/cm²/Hz

        # Apply IGM to stellar photometry (per-filter, interpolated from z-table).
        # igm_trans_table shape: (n_z, n_filters); linear interp along z axis.
        if has_igm:
            _z_c = jnp.clip(_z_arr, _igm_z_grid[0], _igm_z_grid[-1])
            _iz = jnp.clip(jnp.searchsorted(_igm_z_grid, _z_c) - 1, 0, _igm_z_grid.shape[0] - 2)
            _frac = (_z_c - _igm_z_grid[_iz]) / (_igm_z_grid[_iz + 1] - _igm_z_grid[_iz])
            _igm_eff = (
                (1.0 - _frac) * _igm_trans_table[_iz] + _frac * _igm_trans_table[_iz + 1]
            ).astype(jnp.float64)
            stellar_phot = stellar_phot * _igm_eff

        # Total photometry
        total_phot = stellar_phot + non_stellar_phot

        return total_phot

    def hybrid_phot_ztable_fused(params):
        """params dict → photometry (end-to-end JIT for z-table path)."""
        p = get_internal_params(params, param_map, spec, has_field)

        # SFH computation
        kw = {k: v for k, v in p.items() if k in sfh_internal_names}
        if has_field and "xi" in p:
            gp_x, k0_half = compute_field_gp(
                xi=p["xi"],
                psd_sigma=p["psd_sigma"],
                psd_tau_yr=p["psd_tau_yr"],
                n_grid=n_grid,
                d_log_age=d_log_age,
                field_model=field_model,
            )
            kw["gp_x"] = gp_x
            kw["k0_half"] = k0_half
        sfr = sfh_fn(age_yr, **kw)
        sfr_on_ssp = jnp.interp(ssp_log_ages_yr_cap, log_age_grid, sfr)

        # Get redshift from params
        z = p.get("redshift", 0.1)

        # Dispatch to the inner kernel
        if _is_single_dust:
            return hybrid_phot_ztable(
                sfr_on_ssp,
                p["log_z_abs"],
                z,
                p.get("tau_v", 0.0),
                p.get("dust_slope", -0.7),
                **{
                    k: p.get(k, v)
                    for k, v in [
                        ("f_obscuration", 0.0),
                        ("dust_bump_strength", 0.0),
                        ("dust_delta", 0.0),
                        ("dust_Rv", 3.1),
                        ("alpha_fe", 0.0),
                        ("dust_T", 35.0),
                        ("dust_beta_ir", 1.6),
                        ("dust_eta_balance", 1.0),
                        ("agn_log_lbol", 10.0),
                        ("agn_alpha", -1.0),
                        ("agn_T_torus", 1000.0),
                        ("agn_tau_torus", 5.0),
                        ("agn_torus_frac", 0.5),
                        ("agn_log_mbh", 7.0),
                        ("agn_log_ledd", -1.0),
                        ("dust_alpha_mir", 2.0),
                        ("dust_alpha_dale", 2.0),
                        ("dust_umin", 1.0),
                        ("dust_gamma_dl", 0.01),
                        ("dust_qpah", 2.5),
                        ("dust_qhac", 0.17),
                        ("dust_alpha_dl14", 2.0),
                        ("dust_log_ssfr", -10.0),
                    ]
                },
            )
        return hybrid_phot_ztable(
            sfr_on_ssp,
            p["log_z_abs"],
            z,
            p.get("tau_bc", 0.0),
            p.get("tau_diff", 0.0),
            p.get("dust_slope", -0.7),
            **{
                k: p.get(k, v)
                for k, v in [
                    ("f_obscuration", 0.0),
                    ("dust_bump_strength", 0.0),
                    ("dust_delta", 0.0),
                    ("dust_Rv", 3.1),
                    ("alpha_fe", 0.0),
                    ("dust_T", 35.0),
                    ("dust_beta_ir", 1.6),
                    ("dust_eta_balance", 1.0),
                    ("agn_log_lbol", 10.0),
                    ("agn_alpha", -1.0),
                    ("agn_T_torus", 1000.0),
                    ("agn_tau_torus", 5.0),
                    ("agn_torus_frac", 0.5),
                    ("agn_log_mbh", 7.0),
                    ("agn_log_ledd", -1.0),
                ]
            },
        )

    return hybrid_phot_ztable_fused
