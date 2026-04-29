"""SEDModel: high-level forward model wrapping the tengri SED pipeline.

SEDModel provides a clean API for:
- Forward predictions (SED, photometry, spectrum, SFH, derived quantities)
- Mock galaxy generation (single and batch)
- Convenience fitting (delegates to Fitter)

SEDModel translates between the user-facing parameter names and the
internal names used by the low-level functions, handling unit conversions
automatically. SFH computation is dispatched through the registry-driven
composed function, eliminating separate stochastic/parametric code paths.

Usage::

    from tengri import SEDModel, Parameters, Uniform, load_ssp_data, load_filter_set

    ssp = load_ssp_data("data/ssp.h5")
    filters = load_filter_set(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
    spec = Parameters(
        sfh_tsnorm_log_peak_sfr=Uniform(-1, 2),
        sfh_tsnorm_peak_lbt_gyr=Uniform(1, 12),
        sfh_tsnorm_width_gyr=Uniform(0.5, 5),
        sfh_tsnorm_skew=Uniform(-1, 1),
        sfh_tsnorm_trunc=Uniform(1, 10),
        sfh_field_psd_sigma=Uniform(0.01, 1.0),
        sfh_field_psd_tau_myr=Uniform(10, 500),
        redshift=0.1,
    )
    model = SEDModel(spec, ssp, filters=filters)
    params = spec.sample(jax.random.PRNGKey(0))
    photometry = model.predict_photometry(params)
"""

from __future__ import annotations

import contextlib
import dataclasses
import warnings
from typing import ClassVar

import jax
import jax.numpy as jnp

from tengri.components.dust.attenuation import precompute_dust_age_weights
from tengri.components.sfh.registry import compute_field_gp, resolve_sfh
from tengri.components.sps.dsps_wrapper import csp_age_dt
from tengri.components.sps.precompute import (
    precompute_photometry,
    precompute_photometry_ztable,
    precompute_spectroscopy,
)
from tengri.forward._kernels import (
    build_exact_sed,
    build_fused_rest_sed,
    build_fused_tier2_photometry,
    build_fused_tier2_spectrum,
    build_hybrid_photometry,
    build_hybrid_photometry_ztable,
    build_hybrid_spectrum,
    observe_photometry_from_rest_sed,
    observe_spectrum_from_rest_sed,
)
from tengri.forward.pipeline import (
    compute_sed_components,
    get_agn_kwargs,
    get_dust_kwargs,
    interp_metallicity,
    interp_metallicity_evolving,
)
from tengri.forward.sed_model_types import (
    CompositionalKernels,
    HybridKernels,
    MockData,
    PrecomputedData,
    PriorPredictive,
    SEDModelState,
)
from tengri.observation.photometry import ab_mag_from_flux
from tengri.observation.spectrum import apply_lsf, compute_spectrum
from tengri.parameters.translate import (
    _AGN_IDENTITY_PARAMS,
    _DUST_EMISSION_IDENTITY_PARAMS,
    _EVOLVING_ALPHA_PARAM_MAP,
    _NEBULAR_IDENTITY_PARAMS,
    _RADIO_IDENTITY_PARAMS,
    _SHOCK_IDENTITY_PARAMS,
    _XRAY_IDENTITY_PARAMS,
    LOG10_ZSUN,
    _build_param_map,
    get_internal_params,
    identity_param_map,
)
from tengri.utils.cosmology import age_at_z, luminosity_distance
from tengri.utils.grid import (
    grid_spacing,
    interpolate_to_linear_time,
    log_age_to_age_yr,
    make_log_age_grid,
)
from tengri.utils.jit_logging import logged_jit

# Re-export supporting types for backwards compatibility
__all__ = [
    "CompositionalKernels",
    "HybridKernels",
    "MockData",
    "PrecomputedData",
    "PriorPredictive",
    "SEDModel",
    "SEDModelState",
]


class SEDModel:
    """Differentiable SED forward model with modular physics and clean API.

    The forward model maps physical parameters (stellar mass, SFH, metallicity,
    dust, AGN, etc.) to observables: photometry, spectrum, and derived SED
    quantities. Internally, it decomposes the SED pipeline into independent
    physics modules (stellar populations, star formation history, dust,
    nebular, AGN, IGM) that are composed into prediction kernels at
    initialization time, enabling fast inference and flexibility in model
    configuration.

    The SFH is computed via a registry-driven composed function that handles
    additive smooth models, burst mixture, and correlated-field (GP) modulation
    in a single call. Three prediction modes (compositional, hybrid, exact) trade
    accuracy for speed, with automatic fallback.

    Parameters
    ----------
    spec : Parameters
        Parameter specification from ``tengri.Parameters``. Defines
        free/fixed parameters and their priors.
    ssp_data : SSPData
        Pre-loaded SSP templates (from ``load_ssp_data()``). Contains
        absolute SSP grid in ``log10(Z)`` absolute, age array, and
        optional mass-remaining tables for stellar mass surviving
        constraints.
    filters : list or tuple, optional
        Filter transmission curves for photometric prediction. Accepts either:

        - 3-tuple from :func:`load_filter_set`: ``(filter_waves, filter_trans, filter_curves)``
        - List of :class:`FilterCurve` namedtuples

        If provided, enables photometry prediction and automatic precomputation
        at initialization. Either ``filters`` or ``observation`` may be passed,
        not both.
    observation : Observation, optional
        Unified observation config (photometry + spectroscopy + emission lines).
        Mutually exclusive with ``filters``.
    precompute : bool, optional
        Whether to precompute SSP photometry and spectroscopy grids at
        initialization. Default True activates the Zacharegkas+2025
        fast-photometry path and enabling caching of spectroscopy grids.
        Set False to defer computation (useful for batch operations).
    forward_dtype : str or jnp.dtype, optional
        Dtype for forward model computation. Default ``"float64"`` preserves
        full precision. ``"float32"`` halves memory and gives ~1.5× speedup
        with <0.1% accuracy loss for photometry.

        Affects both fused (photometry + precomputation) and exact paths:

        - **Fused path**: captured arrays (SSP grid, dust weights, effective
          wavelengths) cast to ``forward_dtype`` at kernel build; outputs
          always cast back to float64 for cosmological distance scaling.
        - **Exact path** (spectroscopy, non-precomputed AGN): three largest intermediates
          — metallicity-interpolated SSP ``(n_age, n_wave)``, dust attenuation
          ``(n_age, n_wave)``, dust age weights ``(n_age,)`` — computed in
          ``forward_dtype``, halving the 4.5 MB memory traffic that dominates
          exact-path dust cost.

        Cosmological distances always use float64 (float32 overflows at z > 0.01).
    approx : dict or bool, optional
        Control which approximations the fused kernel uses. Default True enables
        all approximations (fastest). False disables all (forces exact path
        everywhere). A dict enables selective control:

        - ``"dust_attenuation"``: use dust at filter effective wavelengths (True, default)
        - ``"dust_emission"``: use MBB at filter effective wavelengths (True, default)
        - ``"igm"``: use IGM at filter effective wavelengths (True, default)

        Approximation accuracy (Zacharegkas+2025 [1]_):

        - dust_attenuation: <3% for most laws, ~36% for SMC
        - dust_emission: negligible for optical (MBB peak >50 μm)
        - igm: exact for fixed z (precomputed once)

    csp_integration : str, optional
        CSP age integration scheme. Default ``"trapz"`` (trapezoidal on
        linear time). Options: ``"log_trapz"``, ``"log_interp"`` (Dopita+2005
        interpolation), ``"dsps_native"`` (DSPS trapezoidal with automatic
        metallicity marginalization), ``"dsps_met_table"`` (time-evolving
        metallicity table). See Appendix A of the forward model paper [2]_.

    Attributes
    ----------
    observation : Observation or None
        Attached observation object containing photometry and/or spectroscopy
        configuration. Set by constructor if filters or observation= passed.
    spec : Parameters
        Parameter specification defining all free/fixed parameters and their priors.
    ssp_data : SSPData
        Pre-loaded stellar population synthesis templates (from ``load_ssp_data()``).
    config : ModelConfig
        Frozen model configuration (immutable after init).

    Notes
    -----
    **JIT-compatible**: yes — all prediction methods (except
    :meth:`predict` for lazy evaluation) are fully JAX differentiable
    and can be called inside :func:`jax.jit` and :func:`jax.vmap`.

    **Gradient-safe**: yes — all physical parameters are differentiable
    for inference via HMC, VI, and score-based methods.

    **Approximation scheme**: The forward model uses a three-tier kernel
    hierarchy to balance speed and accuracy:

    1. **Compositional** (preferred): Full-resolution JIT SED from all
       components → filter integration. XLA fuses entire graph (SFH → SED → photometry).
       Bit-exact and fastest.
    2. **Hybrid** (fallback): Precomputed SSP×filter stellar + exact
       non-stellar at full wavelength resolution.
    3. **Exact** (reference): Raw pipeline, no approximations or precomputation.

    Mode selection in :meth:`predict_photometry` and :meth:`predict_spectrum`:
    ``mode="auto"`` (default) cascades through available modes.

    **Physical units** (internal):

    - Time: years (yr). User-facing API converts to Myr/Gyr.
    - Wavelength: Angstrom (Å).
    - Luminosity (SED components): erg/s/Hz (L_ν).
    - Luminosity (photometry): erg/s/cm²/Hz (f_ν).
    - Metallicity (SSP grid): log₁₀(Z) absolute. User API uses log₁₀(Z/Z☉).
    - AGN bolometric luminosity: log₁₀(L_bol/L☉) at API level.

    **IGM absorption gotcha**: :meth:`predict_obs_sed` applies IGM transmission
    at observed-frame wavelengths (input to ``igm_transmission()`` is redshifted).
    This is automatic when ``igm=True`` in spec.

    References
    ----------
    .. [1] A. Zacharegkas et al., "Fast Photometry with Precomputed
       Stellar Population Grids," ApJ, (2025).
    .. [2] S. Cooray et al., "Forward Model for Differentiable SED Fitting
       with Correlated SFH," (2026).

    Examples
    --------
    Standard photometric fit with DPL SFH::

        from tengri import SEDModel, Parameters, Uniform, load_ssp_data, Photometry

        ssp = load_ssp_data("data/ssp_miles.h5")
        phot = Photometry.from_names(["sdss_r", "sdss_i", "sdss_z"])
        spec = Parameters(
            redshift=0.1,
            sfh_dpl_alpha=Uniform(0.5, 4.0),
            sfh_dpl_beta=Uniform(0.3, 3.0),
        )
        model = SEDModel(spec, ssp, observation=phot)
    """

    # Default approximation settings (immutable — used as template only)
    _DEFAULT_APPROX: ClassVar[dict] = {
        "dust_attenuation": True,
        "dust_emission": True,
        "igm": True,
    }

    _PREDICTION_MODES: ClassVar[frozenset] = frozenset(
        {"auto", "exact", "hybrid", "compositional"}
    )

    # ── Construction ──────────────────────────────────────────────────

    def __init__(
        self,
        spec,
        ssp_data,
        filters=None,
        observation=None,
        precompute=True,
        forward_dtype="float64",
        approx=None,
        csp_integration="trapz",
        wave_chunk_size=None,
    ):
        # ── Observation ────────────────────────────────────────────
        observation, spec = self._init_observation(spec, filters, observation)
        self.observation = observation
        self.spec = spec
        self.ssp_data = ssp_data
        self._forward_dtype = jnp.dtype(forward_dtype)
        self._wave_chunk_size = wave_chunk_size

        # ── Approximation settings ────────────────────────────────
        if approx is None or approx is True:
            self._approx = dict(self._DEFAULT_APPROX)
        elif approx is False:
            self._approx = {k: False for k in self._DEFAULT_APPROX}
        else:
            self._approx = {**self._DEFAULT_APPROX, **approx}

        # ── Stellar populations ───────────────────────────────────
        self._init_ssp(spec, ssp_data, csp_integration)

        # ── Star formation history ────────────────────────────────
        self._init_sfh(spec)

        # ── Metallicity ───────────────────────────────────────────
        self._init_metallicity(spec)

        # ── Dust (attenuation + emission) ─────────────────────────
        self._init_dust(spec)

        # ── IGM + DLA ─────────────────────────────────────────────
        self._init_igm(spec)

        # ── Nebular emission ──────────────────────────────────────
        self._init_nebular(spec, ssp_data)

        # ── AGN ───────────────────────────────────────────────────
        self._init_agn(spec)

        # ── Multiwavelength (radio, X-ray, shock) ─────────────────
        self._init_multiwavelength(spec, ssp_data)

        # ── Instrument (velocity dispersion, LSF) ─────────────────
        self._init_instrument(spec, observation)

        # ── Cosmology (luminosity distance) ───────────────────────
        self._init_cosmology(spec)

        # ── Kernel hierarchy ──────────────────────────────────────
        self._precomputed = self._build_precomputed_data(ssp_data, precompute)

        # ── Frozen runtime bundle for kernel layer (built BEFORE kernels) ──
        self._state = SEDModelState(
            spec=self.spec,
            ssp_data=self.ssp_data,
            precomputed=self._precomputed,
            filter_waves=self.filter_waves,
            filter_trans=self.filter_trans,
            rest_wavelength=self._rest_wavelength,
            log_age_grid=self.log_age_grid,
            age_yr=self.age_yr,
            d_log_age=self.d_log_age,
            n_grid=self._n_grid,
            ssp_log_ages_yr=self.ssp_log_ages_yr,
            ssp_ages_yr=self.ssp_ages_yr,
            csp_matrix=self._csp_matrix,
            csp_age_dt=self._csp_age_dt,
            csp_integration=self._csp_integration,
            forward_dtype=self._forward_dtype,
            met_interp=self._met_interp,
            z_interp=self._z_interp,
            lgmet_scatter=self._lgmet_scatter,
            sfh_fn=self._sfh_fn,
            sfh_internal_names=self._sfh_internal_names,
            uses_stochastic_sfh=self._uses_stochastic_sfh,
            gp_kernel=self._gp_kernel,
            dust_model=self._dust_model,
            dust_law_bc=self._dust_law_bc,
            dust_law_diff=self._dust_law_diff,
            dust_law_bc_fn=self._dust_law_bc_fn,
            dust_law_diff_fn=self._dust_law_diff_fn,
            dust_emission_model=self._dust_emission_model,
            nebular_backend=self._nebular_backend,
            agn_model=self._agn_model,
            agn_luminosity_mode=self._agn_luminosity_mode,
            uses_igm=self._uses_igm,
            uses_radio=self._uses_radio,
            uses_xray=self._uses_xray,
            radio_include_freefree=getattr(self, "_radio_include_freefree", None),
            radio_sfr_mode=getattr(self, "_radio_sfr_mode", None),
            z_fixed=self._z_fixed,
            dl_cm_fixed=self._dl_cm_fixed,
            param_map=self._param_map,
            igm_fn=self._igm_fn,
        )

        # ── Kernels (consume self._state) ─────────────────────────
        self._compositional_kernels = self._build_compositional_kernels()
        self._hybrid_kernels = self._build_hybrid_kernels()

        if (
            observation is not None
            and observation.can_do_spectroscopy
            and self._z_fixed is not None
            and precompute is not False
        ):
            self.precompute_spectroscopy(observation.spectroscopy.wave_obs)

    @staticmethod
    def _init_observation(spec, filters, observation):
        """Resolve observation/filters into a canonical Observation + spec."""
        if filters is not None and observation is not None:
            raise ValueError(
                "Cannot specify both filters= and observation=. "
                "Use observation=Observation(photometry=...) instead."
            )

        if observation is not None or filters is not None:
            from tengri.observation.observation import Observation

        if observation is not None:
            if not isinstance(observation, Observation):
                raise TypeError(
                    f"observation must be an Observation instance, got {type(observation)}"
                )
            obs_params = observation.get_all_params()
            if obs_params:
                spec = spec.with_params(**obs_params)
        elif filters is not None:
            from tengri.observation.photometry_config import Photometry

            observation = Observation(photometry=Photometry.from_filter_set(filters))

        return observation, spec

    def _init_ssp(self, spec, ssp_data, csp_integration):
        """Set up SSP grid, CSP integration, and log-age grid."""
        self._met_interp = getattr(spec, "met_interp", "linear")
        self._lgmet_scatter = float(getattr(spec, "lgmet_scatter", 0.1))
        # Redshift-table interpolation mode for free-z inference.
        # "linear" → piecewise-linear (C^0 gradient, kinks at grid nodes).
        # "smooth" → triweight kernel (C^2 gradient) — recommended for NUTS/HMC
        # when redshift is a free parameter. See `interpolate_ztable_smooth`
        # in components/sps/precompute.py.
        self._z_interp = getattr(spec, "z_interp", "linear")

        self.filter_waves = None
        self.filter_trans = None
        obs = self.observation
        if obs is not None and obs.can_do_photometry:
            self.filter_waves = list(obs.photometry.filter_waves)
            self.filter_trans = list(obs.photometry.filter_trans)

        self.ssp_log_ages_yr = ssp_data.ssp_lg_age_gyr + 9.0
        self.ssp_ages_yr = 10.0**self.ssp_log_ages_yr

        _valid_csp = ("trapz", "log_trapz", "log_interp", "dsps_native", "dsps_met_table")
        if csp_integration not in _valid_csp:
            raise ValueError(
                f"csp_integration must be one of {_valid_csp}, got {csp_integration!r}"
            )
        self._csp_integration = csp_integration
        if csp_integration == "log_interp":
            from tengri.components.sps.dsps_wrapper import csp_log_interp_matrix

            self._csp_matrix = jnp.array(csp_log_interp_matrix(self.ssp_ages_yr))
            self._csp_age_dt = None
        elif csp_integration in ("dsps_native", "dsps_met_table"):
            self._csp_age_dt = None
            self._csp_matrix = None
        else:
            self._csp_age_dt = csp_age_dt(self.ssp_ages_yr, csp_integration)
            self._csp_matrix = None

        n_grid = spec.n_grid if spec.stochastic else 256
        self.log_age_grid = make_log_age_grid(n_grid)
        self.d_log_age = grid_spacing(self.log_age_grid)
        self.age_yr = log_age_to_age_yr(self.log_age_grid)
        self._n_grid = n_grid

    def _init_sfh(self, spec):
        """Resolve SFH from registry and build the base param_map."""
        sfh_fn, _sfh_params, sfh_param_map, sfh_settings = resolve_sfh(spec.mean_sfh_type)
        self._sfh_fn = sfh_fn
        self._sfh_internal_names = {v[0] for v in sfh_param_map.values()}
        self._sfh_settings = sfh_settings
        self._param_map = _build_param_map(
            spec.mean_sfh_type,
            dust_model=getattr(spec, "dust_model", "two_component"),
        )
        self._uses_stochastic_sfh = spec.stochastic
        self._gp_kernel = sfh_settings.get("sfh_field_model", "drw")

    def _init_metallicity(self, spec):
        """Configure metallicity mode and evolving alpha-enhancement."""
        self._met_mode = getattr(spec, "met_mode", "delta")
        # _met_mode checked directly: "ramp" for evolving, "chem_evol" for chemical evolution

        if self._met_mode != "delta":
            self._param_map.pop("met_logzsol", None)
        from tengri.components.sfh.met_registry import resolve_met

        _, _, met_param_map, _ = resolve_met(self._met_mode)
        self._param_map.update(met_param_map)

        self._alpha_fe_evolving = getattr(spec, "alpha_fe_evolving", False)
        if self._alpha_fe_evolving:
            self._param_map.pop("met_alpha_fe", None)
            self._param_map.update(_EVOLVING_ALPHA_PARAM_MAP)

    def _init_dust(self, spec):
        """Configure dust attenuation laws, nebular dust, and dust emission."""
        self._dust_model = getattr(spec, "dust_model", "two_component")
        self._dust_scheme = getattr(spec, "dust_approx", "fast")

        self._dust_law_bc = spec.dust_law_bc
        self._dust_law_diff = spec.dust_law_diff
        from tengri.components.dust.attenuation import resolve_dust_law

        self._dust_law_bc_fn = resolve_dust_law(self._dust_law_bc)
        if self._dust_model == "single_component":
            self._dust_law_diff_fn = self._dust_law_bc_fn
        else:
            self._dust_law_diff_fn = resolve_dust_law(self._dust_law_diff)

        self._neb_dust_mode = getattr(spec, "neb_dust", "bc")
        _neb_bc_law_name = getattr(spec, "neb_dust_law_bc", None)
        if _neb_bc_law_name is not None:
            from tengri.components.dust.attenuation import resolve_dust_law as _rdl

            self._neb_dust_law_bc_fn = _rdl(_neb_bc_law_name)
        else:
            self._neb_dust_law_bc_fn = self._dust_law_bc_fn

        self._dust_emission_model = getattr(spec, "dust_emission", None)
        if self._dust_emission_model == "dl07_tabulated":
            warnings.warn(
                "'dl07_tabulated' is deprecated. Use 'draine_li2007' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self._dust_emission_model = "draine_li2007"
        if self._dust_emission_model:
            self._param_map.update(identity_param_map(_DUST_EMISSION_IDENTITY_PARAMS))

    def _init_igm(self, spec):
        """Configure IGM absorption and DLA."""
        self._uses_igm = spec.apply_igm
        self._uses_dla = getattr(spec, "dla", False)
        self._igm_patchy = getattr(spec, "igm_patchy", False)
        self._igm_model = getattr(spec, "igm_model", "inoue")
        _valid = {"inoue", "madau"}
        if self._igm_model not in _valid:
            raise ValueError(
                f"igm_model={self._igm_model!r} not recognised. Choose from: {sorted(_valid)}"
            )
        if self._igm_model == "madau":
            from tengri.components.igm import igm_transmission_madau as _igm_fn
        else:
            from tengri.components.igm import igm_transmission as _igm_fn
        self._igm_fn = _igm_fn

    def _init_nebular(self, spec, ssp_data):
        """Configure nebular emission backend and register param_map entries."""
        if spec.nebular_mode in ("cloudy", "cue"):
            self._param_map.update(identity_param_map(_NEBULAR_IDENTITY_PARAMS))
            self._param_map["neb_logZ_gas"] = ("neb_logZ_gas", 1.0, LOG10_ZSUN)

        self._nebular_backend = None
        if spec.nebular_mode == "cue":
            from tengri.components.nebular import CueBackend

            self._nebular_backend = CueBackend(spec.cue_weights_path, ssp_data=ssp_data)
        elif spec.nebular_mode == "cloudy":
            from tengri.components.nebular import CloudyGridBackend

            self._nebular_backend = CloudyGridBackend(spec.cloudy_grid_path, ssp_data)
        else:
            from tengri.components.nebular import BakedInBackend

            self._nebular_backend = BakedInBackend()

    def _init_agn(self, spec):
        """Configure AGN model and detect parametric vs. fraction mode."""
        self._agn_model = getattr(spec, "agn_model", None)
        self._agn_luminosity_mode = False
        if self._agn_model:
            agn_dists = getattr(spec, "_distributions", {})
            agn_lbol_dist = agn_dists.get("agn_log_lbol")
            agn_frac_dist = agn_dists.get("agn_frac")
            lbol_is_free = agn_lbol_dist is not None and not agn_lbol_dist.is_fixed
            frac_is_free = agn_frac_dist is not None and not agn_frac_dist.is_fixed
            self._agn_luminosity_mode = lbol_is_free and not frac_is_free
            self._param_map.update(identity_param_map(_AGN_IDENTITY_PARAMS))
            if self._agn_model == "skirtor":
                # Pre-warm the SKIRTOR template cache outside any JIT context.
                # Calling _load_skirtor_fn() lazily inside jit.trace causes a
                # tracer leak because create_skirtor_from_grid allocates jnp.array
                # objects that get captured as DynamicJaxprTracers.
                try:
                    from tengri.components.agn.unified import _load_skirtor_fn

                    _load_skirtor_fn()
                except Exception:
                    pass

    def _init_multiwavelength(self, spec, ssp_data):
        """Configure radio, X-ray, shock, and build wavelength grid."""
        self._uses_radio = getattr(spec, "radio", False)
        if self._uses_radio:
            self._param_map.update(identity_param_map(_RADIO_IDENTITY_PARAMS))
            self._radio_include_freefree = getattr(spec, "radio_include_freefree", True)
            self._radio_sfr_mode = getattr(spec, "radio_sfr_mode", "bell2003")

        self._uses_xray = getattr(spec, "xray", False)
        if self._uses_xray:
            self._param_map.update(identity_param_map(_XRAY_IDENTITY_PARAMS))

        if self._uses_radio or self._uses_xray:
            from tengri.utils.wavelength import make_panchromatic_grid

            self._rest_wavelength = make_panchromatic_grid(
                ssp_data.ssp_wave,
                extend_xray=self._uses_xray,
                extend_radio=self._uses_radio,
            )
        else:
            self._rest_wavelength = ssp_data.ssp_wave

        self._uses_shock = getattr(spec, "shock", False)
        if self._uses_shock:
            self._param_map.update(identity_param_map(_SHOCK_IDENTITY_PARAMS))

    def _init_instrument(self, spec, observation):
        """Configure velocity dispersion and LSF settings."""
        self._has_sigma_v = spec.has_param("sigma_v") if hasattr(spec, "has_param") else False
        if not self._has_sigma_v:
            try:
                spec.get_distribution("sigma_v")
                self._has_sigma_v = True
            except KeyError:
                self._has_sigma_v = False

        if observation is not None and observation.can_do_spectroscopy:
            sc = observation.spectroscopy
            self._sigma_lib_kms = sc.sigma_lib_kms
            self._lsf_resolution = sc.resolution
            self._lsf_n_bins = sc.lsf_n_bins
        else:
            self._sigma_lib_kms = getattr(spec, "sigma_lib_kms", 0.0)
            self._lsf_resolution = getattr(spec, "lsf_resolution", None)
            self._lsf_n_bins = getattr(spec, "lsf_n_bins", 16)

    def _init_cosmology(self, spec):
        """Precompute luminosity distance if redshift is fixed."""
        redshift_dist = spec.get_distribution("redshift")
        if redshift_dist.is_fixed:
            self._dl_cm_fixed = luminosity_distance(redshift_dist.bounds[0])
            self._z_fixed = redshift_dist.bounds[0]
        else:
            self._dl_cm_fixed = None
            self._z_fixed = None

    # ── Kernel management ─────────────────────────────────────────────

    @property
    def _compositional(self):
        """Lazily build compositional kernels on first access."""
        if self._compositional_kernels is None:
            self._compositional_kernels = self._build_compositional_kernels()
        return self._compositional_kernels

    @property
    def _hybrid(self):
        """Lazily build hybrid kernels on first access."""
        if self._hybrid_kernels is None:
            self._hybrid_kernels = self._build_hybrid_kernels()
        return self._hybrid_kernels

    def _invalidate_kernels(self):
        """Reset cached kernels so they're rebuilt on next access."""
        self._compositional_kernels = None
        self._hybrid_kernels = None

    def _precompute_dust_ir_photometry(self):
        """Precompute dust IR template photometry for fast hybrid kernel lookup.

        Delegates to the Precompute Protocol adapter at
        :mod:`tengri.components.dust.dust_emission_precompute`, which handles
        template loading, filter preintegration, and (per the Protocol) auto-
        collapse-on-Fixed for any ``AXIS_PARAMS`` marked :class:`Fixed` in
        ``self.spec``.  Returns ``None`` when the dust model is analytic
        (MBB / Casey) or template data is not available on disk — callers
        fall back to full-wavelength evaluation.

        Returns
        -------
        object or None
            JIT-compiled ``(L_absorbed, *grid_params) -> phot[n_filters]``
            lookup, or ``None`` for analytic / data-missing cases.
        """
        from tengri.components.dust.dust_emission_precompute import (
            build_lookup,
            precompute_for_model,
        )

        model_name = self._dust_emission_model
        try:
            precomp = precompute_for_model(
                model_name,
                filter_waves=self.filter_waves,
                filter_trans=self.filter_trans,
                redshift=float(self._z_fixed) if self._z_fixed is not None else 0.0,
                parameters=self.spec,
            )
            if precomp is None:
                return None
            return build_lookup(precomp, model_name=model_name)
        except Exception as e:
            import warnings

            warnings.warn(
                f"Failed to precompute dust IR photometry for {model_name}: {e}. "
                "Falling back to full-wavelength evaluation.",
                RuntimeWarning,
                stacklevel=2,
            )
            return None

    def _build_precomputed_data(self, ssp_data, precompute):
        """Build Level 1: precomputed SSP tensors."""
        # Photometry precomputation (Zacharegkas+2025 Section 3)
        phot = None
        if precompute and self._z_fixed is not None and self.filter_waves is not None:
            # Extract fixed SSP grid parameters and map to axis indices
            # SSP grid axes: [lgmet, lg_age_gyr]
            fixed_ssp = {}
            if "met_logzsol" in self.spec.fixed_params:
                # met_logzsol is fixed → collapse axis 0 (lgmet).
                # ssp_lgmet is in absolute log10(Z); convert met_logzsol
                # (log10 Z/Zsun) to absolute by adding LOG10_ZSUN.
                dist = self.spec._distributions["met_logzsol"]
                fixed_ssp[0] = float(dist.value) + LOG10_ZSUN

            phot = precompute_photometry(
                ssp_data,
                self.filter_waves,
                self.filter_trans,
                self._z_fixed,
                self._dl_cm_fixed,
                fixed=fixed_ssp if fixed_ssp else None,
            )

        # Dust age weights (sigmoid, for exact two-component dust)
        dust_age_w = None
        if self._dust_model != "single_component" and self._dust_scheme == "exact":
            dust_age_w = precompute_dust_age_weights(self.ssp_ages_yr)

        # IGM at filter effective wavelengths (for hybrid kernel, fixed z)
        igm_eff = None
        if (
            self._uses_igm
            and self._approx["igm"]
            and phot is not None
            and self._z_fixed is not None
        ):
            igm_eff = self._igm_fn(phot.effective_wavelengths, self._z_fixed)

        # Voronoi frequency bandwidths for L_absorbed broadband estimate.
        # Each filter is assigned a non-overlapping frequency interval via
        # Voronoi tessellation at the filter effective frequencies.  This
        # converts the naive sum(L_ν) into a proper ∫L_ν dν quadrature.
        eff_bw = None
        if phot is not None and self._z_fixed is not None:
            _c_aa = 2.998e18  # speed of light in Angstrom/s
            eff_rest = phot.effective_wavelengths / (1.0 + self._z_fixed)
            eff_nu = _c_aa / eff_rest  # Hz, decreasing order (UV first)

            sort_idx = jnp.argsort(eff_nu)
            nu_sorted = eff_nu[sort_idx]
            midpoints = 0.5 * (nu_sorted[:-1] + nu_sorted[1:])
            lower = nu_sorted[0] - 0.5 * (nu_sorted[1] - nu_sorted[0])
            upper = nu_sorted[-1] + 0.5 * (nu_sorted[-1] - nu_sorted[-2])
            edges = jnp.concatenate(
                [jnp.array([jnp.maximum(lower, 0.0)]), midpoints, jnp.array([upper])]
            )
            dnu_sorted = edges[1:] - edges[:-1]
            unsort_idx = jnp.argsort(sort_idx)
            eff_bw = dnu_sorted[unsort_idx]

        # CLOUDY nebular preintegration (continuum + lines through filters)
        if (
            precompute
            and self._z_fixed is not None
            and self.filter_waves is not None
            and self._nebular_backend is not None
            and hasattr(self._nebular_backend, "preintegrate_for_photometry")
        ):
            # Extract fixed grid parameters and map to axis indices
            # CLOUDY grid axes: [log_met, log_age, log_U]
            fixed_cloudy = {}
            if "met_logzsol" in self.spec.fixed_params:
                # met_logzsol is fixed → collapse axis 0 (log_met).
                # CLOUDY grid log_met is absolute log10(Z); convert
                # met_logzsol (log10 Z/Zsun) by adding LOG10_ZSUN.
                dist = self.spec._distributions["met_logzsol"]
                fixed_cloudy[0] = float(dist.value) + LOG10_ZSUN
            if "neb_logU" in self.spec.fixed_params:
                # neb_logU is fixed → collapse axis 2 (log_U)
                dist = self.spec._distributions["neb_logU"]
                fixed_cloudy[2] = float(dist.value)

            self._nebular_backend.preintegrate_for_photometry(
                self.filter_waves,
                self.filter_trans,
                self._z_fixed,
                self._dl_cm_fixed,
                fixed=fixed_cloudy if fixed_cloudy else None,
            )

        # Dust IR emission template preintegration (for hybrid kernel, fixed z)
        # For template-based dust models (DL07, Dale2014, etc.), pre-integrate
        # templates through filters at init time for fast runtime triweight lookup.
        dust_ir_lookup = None
        if (
            precompute
            and self._z_fixed is not None
            and self.filter_waves is not None
            and self._dust_emission_model is not None
        ):
            dust_ir_lookup = self._precompute_dust_ir_photometry()

        # K&D 2018 AGN disc preintegration (for hybrid kernel, fixed z, K&D models)
        # Pre-integrate the three K&D disc zones through filters at init time
        # for fast runtime filter-level lookup instead of wavelength-level computation.
        kd_preint = None
        if (
            precompute
            and self._z_fixed is not None
            and self.filter_waves is not None
            and self._agn_model in ("kubota_done_full", "kubota_done_disc")
        ):
            from tengri.components.agn.kd_precompute import preintegrate_kd_components

            kd_preint = preintegrate_kd_components(
                self.filter_waves,
                self.filter_trans,
                self._z_fixed,
            )

        # SKIRTOR torus preintegration (for hybrid kernel, fixed z, SKIRTOR models)
        # Pre-integrate SKIRTOR torus templates through filters at init time for
        # fast filter-level triweight lookup instead of wavelength-level computation.
        skirtor_preint = None
        if (
            precompute
            and self._z_fixed is not None
            and self.filter_waves is not None
            and self._agn_model == "skirtor"
        ):
            try:
                from tengri.components.agn.skirtor import _find_skirtor_grid
                from tengri.components.agn.skirtor_precompute import (
                    build_skirtor_photometry_lookup,
                    precompute_skirtor_photometry,
                )

                _grid_path = _find_skirtor_grid()
                _precomp = precompute_skirtor_photometry(
                    _grid_path,
                    self.filter_waves,
                    self.filter_trans,
                    redshift=float(self._z_fixed),
                )
                skirtor_preint = build_skirtor_photometry_lookup(_precomp)
            except Exception as e:
                import warnings

                warnings.warn(
                    f"SKIRTOR torus preintegration failed: {e}. "
                    "Falling back to full-wavelength evaluation.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        return PrecomputedData(
            photometry=phot,
            dust_age_weights=dust_age_w,
            igm_at_effective_wavelengths=igm_eff,
            effective_bandwidths_hz=eff_bw,
            dust_ir_lookup=dust_ir_lookup,
            kd_preintegrated=kd_preint,
            skirtor_preintegrated=skirtor_preint,
        )

    def _build_compositional_kernels(self):
        """Build Level 2: full-resolution JIT-compiled kernels."""
        exact_sed = build_exact_sed(self._state)

        rest_sed = None
        try:
            rest_sed = build_fused_rest_sed(self._state, self)
        except Exception as e:
            warnings.warn(
                f"Compositional rest-SED kernel build failed: {e}",
                UserWarning,
                stacklevel=2,
            )

        # Store partial result so build_fused_tier2_photometry can read
        # model._compositional.rest_sed during construction.
        self._compositional_kernels = CompositionalKernels(
            rest_sed=rest_sed,
            exact_sed=exact_sed,
        )

        fused_phot = None
        fused_phot_raw = None
        fused_spec = None
        fused_spec_raw = None
        if rest_sed is not None and self.filter_waves is not None:
            with contextlib.suppress(Exception):
                fused_phot_raw = build_fused_tier2_photometry(self._state, self)
                fused_phot = (
                    logged_jit(fused_phot_raw, name="compositional_phot")
                    if fused_phot_raw is not None
                    else None
                )

        if rest_sed is not None:
            with contextlib.suppress(Exception):
                fused_spec_raw = build_fused_tier2_spectrum(self._state, self)
                fused_spec = (
                    logged_jit(fused_spec_raw, name="compositional_spec")
                    if fused_spec_raw is not None
                    else None
                )

        ck = CompositionalKernels(
            rest_sed=rest_sed,
            photometry=fused_phot,
            spectrum=fused_spec,
            exact_sed=exact_sed,
        )
        # Store raw (un-JIT'd) versions for NIFTy tracing
        ck._photometry_raw = fused_phot_raw
        ck._spectrum_raw = fused_spec_raw
        return ck

    def _build_hybrid_kernels(self):
        """Build Level 3: precomputed SSP + exact non-stellar kernels.

        The hybrid kernel uses precomputed SSP×filter photometry for stellar
        (fast, ~0.4% error) and evaluates all non-stellar components at full
        wavelength resolution via emission_helpers.py, then integrates
        through filters (exact).
        """
        hybrid_phot = None
        hybrid_phot_raw = None
        if self._precomputed.photometry is not None and self._z_fixed is not None:
            with contextlib.suppress(Exception):
                hybrid_phot_raw = build_hybrid_photometry(self._state, self)
                hybrid_phot = (
                    logged_jit(hybrid_phot_raw, name="hybrid_phot")
                    if hybrid_phot_raw is not None
                    else None
                )

        hybrid_spec = None
        hybrid_spec_raw = None
        if self._precomputed.spectroscopy is not None and self._z_fixed is not None:
            with contextlib.suppress(Exception):
                hybrid_spec_raw = build_hybrid_spectrum(self._state, self)
                hybrid_spec = (
                    logged_jit(hybrid_spec_raw, name="hybrid_spec")
                    if hybrid_spec_raw is not None
                    else None
                )

        hk = HybridKernels(photometry=hybrid_phot, spectrum=hybrid_spec)
        hk._photometry_raw = hybrid_phot_raw
        hk._spectrum_raw = hybrid_spec_raw
        return hk

    # ── Parameter translation ─────────────────────────────────────────

    def _get_internal_params(self, params):
        """Translate public param dict to internal names with unit conversion.

        Thin wrapper around :func:`tengri._param_translate.get_internal_params`.
        """
        return get_internal_params(params, self._param_map, self.spec, self._uses_stochastic_sfh)

    def _get_redshift(self, params):
        """Get redshift value from params or fixed value."""
        if "redshift" in params:
            return params["redshift"]
        if self._z_fixed is not None:
            return self._z_fixed
        raise KeyError("Redshift not in params and not fixed in spec")

    def _get_dl_cm(self, params):
        """Get luminosity distance from params or precomputed value."""
        if self._dl_cm_fixed is not None:
            return self._dl_cm_fixed
        z = self._get_redshift(params)
        return luminosity_distance(z)

    # ── Core physics (SFH → SED pipeline) ─────────────────────────────

    def _compute_sfr(self, p):
        """Compute SFR via the composed SFH function.

        Single dispatch point for all SFH computation — replaces
        the old stochastic/parametric if/else branches.

        Parameters
        ----------
        p : dict
            Internal parameter dict from _get_internal_params().

        Returns
        -------
        array, shape (n_grid,)
            SFR(t) in Msun/yr on the log-age grid.
        """
        # Build kwargs for the composed SFH function
        kw = {k: v for k, v in p.items() if k in self._sfh_internal_names}

        # If field is present, compute GP and pass to composed fn
        if self._uses_stochastic_sfh and "xi" in p:
            gp_x, k0_half = compute_field_gp(
                xi=p["xi"],
                psd_sigma=p["psd_sigma"],
                psd_tau_yr=p["psd_tau_yr"],
                n_grid=self._n_grid,
                d_log_age=float(self.d_log_age),
                field_model=self._gp_kernel,
            )
            kw["gp_x"] = gp_x
            kw["k0_half"] = k0_half

        return self._sfh_fn(self.age_yr, **kw)

    def _compute_sfr_mean_and_full(self, p):
        """Compute both mean (no GP) and full (with GP) SFR.

        Used by predict_sfh which needs to return both.

        Returns
        -------
        sfr_mean : array
            SFR without GP modulation.
        sfr_full : array
            SFR with GP modulation (same as sfr_mean if no field).
        """
        kw = {k: v for k, v in p.items() if k in self._sfh_internal_names}
        sfr_mean = self._sfh_fn(self.age_yr, **kw)

        if self._uses_stochastic_sfh and "xi" in p:
            gp_x, k0_half = compute_field_gp(
                xi=p["xi"],
                psd_sigma=p["psd_sigma"],
                psd_tau_yr=p["psd_tau_yr"],
                n_grid=self._n_grid,
                d_log_age=float(self.d_log_age),
                field_model=self._gp_kernel,
            )
            kw["gp_x"] = gp_x
            kw["k0_half"] = k0_half
            sfr_full = self._sfh_fn(self.age_yr, **kw)
        else:
            sfr_full = sfr_mean

        return sfr_mean, sfr_full

    def _compute_sed_components(
        self, params, _sfr=None, _weights=None, need_intrinsic=False, rest_wavelength=None
    ):
        """Compute all SED intermediates.

        Delegates to :func:`tengri._sed_pipeline.compute_sed_components`.
        """
        return compute_sed_components(
            self, params, _sfr, _weights, need_intrinsic, rest_wavelength=rest_wavelength
        )

    def _get_dust_kwargs(self, p):
        """Extract dust law + emission kwargs from internal params dict."""
        return get_dust_kwargs(self, p)

    def _get_agn_kwargs(self, p):
        """Extract AGN kwargs from internal params dict for fused kernel."""
        return get_agn_kwargs(self, p)

    def _get_non_stellar_kwargs(self, p):
        """Extract non-stellar kwargs from internal params for hybrid kernel."""
        kw = {}
        # Nebular
        if self._nebular_backend is not None and getattr(
            self._nebular_backend, "has_free_params", False
        ):
            kw["neb_logU"] = p.get("neb_logU", -3.0)
            kw["neb_logZ_gas"] = p.get("neb_logZ_gas", None)
            kw["neb_fesc"] = p.get("neb_fesc", 0.0)
            kw["neb_fesc_lya"] = p.get("neb_fesc_lya", 0.0)
        # Shock
        if self._uses_shock:
            kw["shock_frac"] = p.get("shock_frac", 0.0)
            kw["shock_velocity"] = p.get("shock_velocity", 300.0)
            kw["shock_log_density"] = p.get("shock_log_density", 0.0)
            kw["shock_b_over_sqrt_n"] = p.get("shock_b_over_sqrt_n", 1.0)
        # Dust emission (all models, not just MBB)
        if self._dust_emission_model is not None:
            kw["dust_T"] = p.get("dust_T", 35.0)
            kw["dust_beta_ir"] = p.get("dust_beta_ir", 1.6)
            kw["dust_eta_balance"] = p.get("dust_eta_balance", 1.0)
            kw["dust_alpha_mir"] = p.get("dust_alpha_mir", 2.0)
            kw["dust_alpha_dale"] = p.get("dust_alpha_dale", 2.0)
            kw["dust_umin"] = p.get("dust_umin", 1.0)
            kw["dust_gamma_dl"] = p.get("dust_gamma_dl", 0.01)
            kw["dust_qpah"] = p.get("dust_qpah", 2.5)
        # AGN (full params for exact evaluation)
        if self._agn_model is not None:
            kw["agn_polar_ebv"] = p.get("agn_polar_ebv", 0.0)
            kw["agn_cos_inc"] = p.get("agn_cos_inc", 0.5)
            kw["agn_polar_oa"] = p.get("agn_polar_oa", 45.0)
            kw["agn_frac"] = p.get("agn_frac", 0.0)
            kw["agn_a_spin"] = p.get("agn_a_spin", 0.0)
            kw["agn_log_mbh"] = p.get("agn_log_mbh", 7.0)
            kw["agn_log_ledd"] = p.get("agn_log_ledd", -1.0)
            # K&D 3-zone disc params
            kw["agn_f_hard"] = p.get("agn_f_hard", 0.02)
            kw["agn_gamma_warm"] = p.get("agn_gamma_warm", 2.5)
            kw["agn_kt_warm"] = p.get("agn_kt_warm", 0.2)
            kw["agn_gamma_hard"] = p.get("agn_gamma_hard", 1.8)
            kw["agn_kt_hot"] = p.get("agn_kt_hot", 100.0)
            kw["agn_r_warm_ratio"] = p.get("agn_r_warm_ratio", 2.0)
            # Two-temperature torus
            kw["agn_T_hot"] = p.get("agn_T_hot", 1200.0)
            kw["agn_T_warm"] = p.get("agn_T_warm", 300.0)
            kw["agn_frac_hot"] = p.get("agn_frac_hot", 0.3)
            # SKIRTOR torus
            kw["agn_tau_skirtor"] = p.get("agn_tau_skirtor", 7.0)
            kw["agn_p_skirtor"] = p.get("agn_p_skirtor", 1.0)
            kw["agn_q_skirtor"] = p.get("agn_q_skirtor", 1.0)
            kw["agn_oa_skirtor"] = p.get("agn_oa_skirtor", 40.0)
        # Radio
        if self._uses_radio:
            kw["radio_loudness"] = p.get("radio_loudness", 0.0)
            kw["log_mstar"] = jnp.log10(jnp.maximum(p.get("mstar", 1e10), 1e-10))
        return kw

    # ── Predictions (public API) ──────────────────────────────────────

    def predict_sfh(self, params, n_linear=1000):
        """Compute SFH on uniform linear-time grid for visualization.

        Evaluates the SFH parameterization at ``n_linear`` evenly-spaced
        points in lookback time, returning both the smooth parametric
        component (``sfr_mean``) and the full SFH including GP-field
        modulation (``sfr_full``, if stochastic SFH enabled).

        Parameters
        ----------
        params : dict
            Parameter values using public parameter names.
        n_linear : int, optional
            Number of output grid points, evenly spaced in lookback time.
            Default 1000 (sufficient for smooth visualization).

        Returns
        -------
        dict with keys:

            - ``"t_gyr"`` : array, shape (n_linear,).
              Lookback time [Gyr], from 0 (now) to ~13.8 (Big Bang).
            - ``"sfr_mean"`` : array, shape (n_linear,).
              Parametric mean SFR [M☉/yr] (no GP modulation).
            - ``"sfr_full"`` : array, shape (n_linear,).
              Full SFH including GP field [M☉/yr]. Identical to ``sfr_mean``
              if stochastic SFH not enabled.

        Notes
        -----
        **JIT-compatible**: no — uses Python-side interpolation. For
        JIT-compatible SFH evaluation, use :meth:`predict_sfh_quantities`
        to get integrated quantities (stellar mass, age, etc.).

        **Time grid**: Output is on a uniform linear-time (lookback) grid,
        not the internal log-age grid. This makes visualization cleaner
        and suitable for plotting.

        **SFH mean vs. full**: When correlated-field (stochastic) SFH is enabled,
        ``sfr_mean`` shows the smooth parametric trend (e.g., exponential
        decline), while ``sfr_full`` adds GP modulation for realistic burstiness.
        If parametric-only SFH is used, they are identical.

        **Physical units**: Output SFR is in M☉/yr. Lookback time is in Gyr
        (cosmic time before today).

        Examples
        --------
        >>> sfh = model.predict_sfh(params)
        >>> print(sfh.keys())
        dict_keys(['t_gyr', 'sfr_mean', 'sfr_full'])
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(sfh["t_gyr"], sfh["sfr_mean"], label="Smooth")
        >>> if "sfr_full" in sfh:
        ...     plt.plot(sfh["t_gyr"], sfh["sfr_full"], alpha=0.5, label="With bursts")

        See Also
        --------
        predict_sfh_quantities : Integrated SFH quantities (JIT-compatible).
        predict : Lazy access to SFH and all derived quantities.
        """
        p = self._get_internal_params(params)
        sfr_mean, sfr_full = self._compute_sfr_mean_and_full(p)

        t_gyr_mean, sfr_mean_lin = interpolate_to_linear_time(
            self.log_age_grid, sfr_mean, n_linear
        )
        _, sfr_full_lin = interpolate_to_linear_time(self.log_age_grid, sfr_full, n_linear)

        return {
            "t_gyr": t_gyr_mean,
            "sfr_mean": sfr_mean_lin,
            "sfr_full": sfr_full_lin,
        }

    def predict_rest_sed(self, params, wave=None):
        """Compute rest-frame panchromatic SED luminosity spectrum.

        Evaluates all stellar populations, emission (nebular, AGN), and
        multi-wavelength (radio, X-ray) components in rest-frame coordinates.
        Returns the total SED integrated across the age distribution set by
        the SFH and stellar mass parameters.

        Parameters
        ----------
        params : dict
            Parameter values using public parameter names.
        wave : array, optional
            Custom rest-frame wavelength grid [Angstrom]. If None,
            uses the model's default: SSP wavelength grid
            (``ssp_data.ssp_wave``), or auto-extended grid if
            ``radio=True`` or ``xray=True`` in spec.

        Returns
        -------
        SEDResult
            NamedTuple with:

            - ``wavelength`` : array, shape (n_wave,). Rest-frame wavelength [Ångstrom]
            - ``sed`` : array, shape (n_wave,). Spectral luminosity density [erg/s/Hz]

        Notes
        -----
        **JIT-compatible**: no — computes SED components via
        :func:`_compute_sed_components` which is not JIT'd. For JIT-compatible
        SED access, use :meth:`predict_sed_quantities` instead.

        **Physical units**:

        - Wavelength: rest-frame Ångstrom (not redshifted)
        - SED: erg/s/Hz (L_ν), normalized to the total stellar mass
          implied by the SFH

        **SED components**: Total SED is the sum of:

        - Stellar continuum (CSP from SSP integration)
        - Nebular continuum (if nebular_mode ≠ 'baked-in')
        - Nebular emission lines (if ``neb_*`` params free)
        - AGN continuum (if ``agn_model`` set)
        - Dust attenuation (applied to stellar + AGN)
        - Dust emission (re-radiated IR, if dust_emission_model set)
        - Shock emission (if ``shock=True``)
        - Radio/X-ray (if ``radio=True`` or ``xray=True``)

        **Attenuation**: Applied via two-component (birth cloud + diffuse ISM)
        or single-screen dust law, parameterized by age-dependent optical depth.
        See ``components.dust`` for available laws.

        Examples
        --------
        >>> sed = model.predict_rest_sed(params)
        >>> import matplotlib.pyplot as plt
        >>> plt.loglog(sed.wavelength, sed.sed)
        >>> plt.xlabel("Rest-frame wavelength (Angstrom)")
        >>> plt.ylabel("SED (erg/s/Hz)")

        See Also
        --------
        predict_obs_sed : Observed-frame SED (redshifted + IGM).
        predict_sed_quantities : JIT-compatible SED-derived quantities.
        """
        from tengri.forward.result import SEDResult

        rest_wave = wave if wave is not None else self._rest_wavelength
        result = self._compute_sed_components(params, rest_wavelength=rest_wave)
        return SEDResult(wavelength=result["rest_wavelength"], sed=result["sed_total"])

    def predict_obs_sed(self, params, wave=None):
        """Compute observed-frame SED (redshifted + IGM + DLA transmission).

        Evaluates the rest-frame SED, redshifts to observed frame
        (wavelength × (1+z)), and applies IGM and DLA absorption where
        configured. At z=0, identical to :meth:`predict_rest_sed`.

        Parameters
        ----------
        params : dict
            Parameter values using public parameter names.
        wave : array, optional
            Custom rest-frame wavelength grid [Angstrom] before redshifting.
            If None, uses model default.

        Returns
        -------
        SEDResult
            NamedTuple with:

            - ``wavelength`` : array, shape (n_wave,).
              Observed-frame wavelength [Ångstrom]
            - ``sed`` : array, shape (n_wave,).
              Observed-frame spectral luminosity density [erg/s/Hz]

        Notes
        -----
        **JIT-compatible**: no — delegates to :meth:`predict_rest_sed`.

        **IGM absorption**: Applies transmission via
        :math:`T_{\\mathrm{IGM}}(\\lambda_{\\mathrm{obs}}, z)` when ``igm=True`` in spec.
        Uses Inoue+2014 [1]_ mean IGM with optional extensions for:

        - Reionization epoch: CGM damping wing (Asada+2025 [2]_)
        - Patchy reionization: parameterized neutral fraction (Mason+2018 [3]_)

        **CRITICAL GOTCHA**: IGM transmission takes **observed-frame** wavelengths
        as input. The redshifted ``wavelength`` in this SED is already in observed
        frame, so ``igm_transmission(wave_obs, z)`` is called correctly.

        **DLA absorption**: Applies Lyman-series damping wing when ``dla=True``.
        Parameterized by neutral column density log₁₀(N_HI) and temperature.
        See :func:`~tengri.components.igm.dla.dla_transmission_obs`.

        **Physical units**:

        - Wavelength: observed-frame Ångstrom (redshifted)
        - SED: erg/s/Hz (same as rest-frame), but now at redshifted
          wavelengths and reduced intensity by :math:`(1+z)` factor from
          cosmological redshift

        Examples
        --------
        >>> sed_obs = model.predict_obs_sed(params)
        >>> # IGM and redshift already applied
        >>> print(f"z={params['redshift']}: wavelength {sed_obs.wavelength[0]:.0f} Å")

        See Also
        --------
        predict_rest_sed : Rest-frame SED (before redshift/IGM).
        predict_photometry : Filter-integrated observed flux (uses this internally).

        References
        ----------
        .. [1] A. K. Inoue et al., "An updated analytic model for attenuation
           by the intergalactic medium," MNRAS, 442, 1805 (2014).
           arXiv:1402.0677. https://doi.org/10.1093/mnras/stu936
        .. [2] Y. Asada et al., "Improving Photometric Redshifts of Epoch of
           Reionization Galaxies: A New Empirical Transmission Curve with
           Neutral Hydrogen Damping Wing Ly-alpha Absorption," ApJL, 983, L2
           (2025). arXiv:2410.21543.
           https://doi.org/10.3847/2041-8213/adc388
        .. [3] C. A. Mason et al., "The Universe Is Reionizing at z ~ 7:
           Bayesian Inference of the IGM Neutral Fraction Using Ly-alpha
           Emission from Galaxies," ApJ, 856, 2 (2018).
           https://doi.org/10.3847/1538-4357/aab0a7
        """
        from tengri.forward.result import SEDResult

        rest_result = self.predict_rest_sed(params, wave=wave)
        z = self._get_redshift(params)
        wave_obs = rest_result.wavelength * (1.0 + z)
        sed_obs = rest_result.sed
        if self._uses_igm:
            from tengri.forward.emission_helpers import igm_absorption

            # Always apply IGM when enabled — igm_transmission returns
            # all-ones at z=0. Avoid z>0 comparison which fails under JIT.
            igm_trans = igm_absorption(
                wave_obs,
                z,
                igm_x_HI=params.get("igm_x_HI", 0.0),
                igm_bubble_mpc=params.get("igm_bubble_mpc", 10.0),
                igm_patchy=getattr(self, "_igm_patchy", False),
                igm_model=self._igm_model,
            )
            sed_obs = sed_obs * igm_trans
        if self._uses_dla:
            from tengri.components.igm.dla import dla_transmission_obs

            z_dla = params.get("dla_z", 0.0)
            z_dla = jnp.where(z_dla > 0.0, z_dla, z)
            sed_obs = sed_obs * dla_transmission_obs(
                wave_obs,
                z_dla=z_dla,
                log_n_hi=params.get("dla_log_n_hi", 20.0),
                temp=params.get("dla_temp", 1e4),
                b_turb_kms=params.get("dla_b_turb", 0.0),
            )
        return SEDResult(wavelength=wave_obs, sed=sed_obs)

    def predict(self, params):
        """Create a lazy prediction object for all derived physical quantities.

        Returns a :class:`Prediction` object that computes and caches
        derived quantities on first access. This is the recommended API
        for interactive exploration of a single galaxy's properties,
        trading speed for convenience.

        For batch computation over posterior chains or mock catalogs,
        use the JIT-compatible methods :meth:`predict_sfh_quantities`,
        :meth:`predict_sed_quantities`, or :meth:`predict_line_luminosities`
        with :func:`jax.vmap` instead (up to 1000× faster for large batches).

        Parameters
        ----------
        params : dict
            Parameter values using public parameter names.

        Returns
        -------
        Prediction
            Lazy caching wrapper with property groups:

            - ``.sfh`` : SFH-derived quantities (stellar mass, SFR, age, metallicity)
            - ``.sed`` : SED-derived quantities (luminosities, colors, indices)
            - ``.lines`` : Emission line properties (luminosities, fluxes, ratios)
            - ``.radio`` : Radio SED properties (if ``radio=True``)
            - ``.xray`` : X-ray SED properties (if ``xray=True``)
            - ``.ionizing`` : Ionizing photon budget properties

        Notes
        -----
        **Not JIT-compatible**: Uses Python-side caching and object
        attribute access. Useful for interactive exploration, not
        for inference loops. For inference, use
        :meth:`predict_sfh_quantities`, :meth:`predict_sed_quantities`,
        etc. with :func:`jax.vmap`.

        **Lazy evaluation**: Quantities are computed only when accessed.
        Repeated access to the same property reuses cached results.
        This is transparent to the user.

        **NaN handling**: Some quantities (e.g., ``stellar_mass_surviving``,
        ``l_dust_absorbed``) may return NaN if required data/parameters
        unavailable (e.g., no mass-remaining table, dust_model='none').
        The Prediction object handles NaN gracefully (returns None when
        data required to compute the quantity is absent).

        Examples
        --------
        **Single-galaxy exploration (lazy, on-demand):**

        >>> pred = model.predict(params)
        >>> pred.sfh.stellar_mass  # triggers SFH computation, caches result
        Array(1.23e10, dtype=float64)
        >>> pred.sfh.mass_weighted_age_gyr  # reuses cached SFH
        Array(2.34, dtype=float64)
        >>> pred.sed.l_bol  # triggers SED computation
        Array(2.5e10, dtype=float64)
        >>> pred.sed.uv_slope_beta  # reuses cached SED
        Array(-1.8, dtype=float64)
        >>> pred.lines.halpha  # triggers nebular computation
        Array(4.23e-15, dtype=float64)

        **Batch computation (JIT-compatible, faster for large N):**

        >>> import jax
        >>> params_batch = spec.sample(jax.random.PRNGKey(0), n=10000)
        >>> sfh_fn = jax.vmap(model.predict_sfh_quantities)
        >>> sfh_batch = sfh_fn(params_batch)
        >>> sfh_batch.stellar_mass  # shape (10000,)
        >>> sfh_batch.stellar_mass.mean()

        See Also
        --------
        predict_sfh_quantities : JIT-compatible SFH quantities for batch.
        predict_sed_quantities : JIT-compatible SED quantities for batch.
        predict_line_luminosities : JIT-compatible emission lines for batch.
        predict_rest_sed : Full rest-frame SED for custom analysis.
        """
        from tengri.forward.prediction import Prediction

        return Prediction(self, params)

    def predict_photometry(self, params, mode="auto", approx=None):
        """Compute observed photometric flux densities through all filters.

        Convolves the SED (redshifted and IGM-absorbed) through filter
        transmission curves, returning flux densities in the AB system
        at the source. Supports three prediction modes for speed/accuracy
        tradeoff: compositional (exact, XLA-fused), hybrid (precomputed
        stellar + exact non-stellar), and exact (full pipeline, slowest).

        Parameters
        ----------
        params : dict
            Parameter values using public parameter names (e.g.,
            ``sfh_tsnorm_log_peak_sfr``, ``met_logzsol``, ``redshift``).
            See :class:`Parameters` for canonical names.
        mode : str, optional
            Prediction strategy. Default ``"auto"`` selects fastest available.

            - ``"auto"`` — cascade through available: compositional → hybrid → exact
            - ``"compositional"`` — full-resolution JIT SED kernel (bit-exact,
              fastest). All components evaluated at full wavelength, integrated
              through filters in single XLA-fused graph. Preferred when available.
            - ``"hybrid"`` — precomputed SSP×filter photometry (stellar, ~0.4% error)
              + exact non-stellar (emission, AGN, dust) at full wavelength, integrated
              through filters. Fallback when compositional unavailable (e.g.,
              variable-redshift, evolving metallicity, tabulated SFH).
            - ``"exact"`` — raw forward pipeline, no kernel JIT, no precomputation.
              Reference accuracy, slowest (~5–10× slower than compositional).
        approx : bool, optional
            Maps ``True`` → ``mode="auto"`` and ``False`` → ``mode="exact"``.
            Prefer passing ``mode=`` directly.

        Returns
        -------
        flux_density : array, shape (n_filters,)
            Observed flux densities in erg/s/cm²/Hz (AB system, rest-frame
            reference frame corrected for luminosity distance and (1+z)
            redshift factor).

        Raises
        ------
        ValueError
            If no filters configured in the model (pass ``filters`` or
            ``observation=`` to constructor).

        Notes
        -----
        **JIT-compatible**: yes — compositional and hybrid modes are
        JIT'd at initialization. Exact mode is not JIT'd. All modes
        are safe inside :func:`jax.grad` for parameter gradients.

        **Approximate accuracy**: Compositional and hybrid modes produce
        predictions within 0.1%–0.4% of exact (see CLAUDE.md for
        mode-specific tolerances). Differences driven by:

        - Compositional: None (bit-exact vs. exact)
        - Hybrid: ~0.4% stellar photometry (Zacharegkas+2025 [1]_)
        - Approximations enabled via ``approx``: see :class:`SEDModel`
          for individual component tolerances

        **Filter wavelengths**: All filters loaded via :func:`load_filter_set`
        or :class:`Photometry` are assumed to be in observed frame (redshifted).
        The model auto-redshifts rest-frame SED by :math:`(1+z)` before
        filter integration.

        See Also
        --------
        predict : Lazy prediction object for all derived quantities.
        predict_spectrum : Spectral flux at arbitrary wavelengths.
        predict_magnitudes : AB magnitudes (uses photometry internally).

        Examples
        --------
        >>> flux = model.predict_photometry(params)
        >>> mags = model.predict_magnitudes(params)
        >>> flux_exact = model.predict_photometry(params, mode="exact")

        References
        ----------
        .. [1] A. Zacharegkas et al., "Fast Photometry with Precomputed
           Stellar Population Grids," ApJ, (2025).
        """
        if self.filter_waves is None:
            raise ValueError("No filters set. Pass filters or observation= to SEDModel().")

        # approx=bool is shorthand for mode= (for scripts that pre-date mode=)
        if approx is not None:
            mode = "auto" if approx else "exact"

        # _traceable: un-JIT'd path for NIFTy/VI tracing (not user-facing)
        if mode == "_traceable":
            return self._predict_photometry_traceable(params)

        if mode not in self._PREDICTION_MODES:
            raise ValueError(
                f"Unknown mode {mode!r}. Choose from: {sorted(self._PREDICTION_MODES)}"
            )

        if mode == "auto":
            return self._predict_photometry_auto(params)
        if mode == "_traceable":
            # Raw un-JIT'd path for use inside inference JIT scopes.
            # Picks hybrid (precomputed, tiny graph) if available,
            # otherwise falls back to auto.
            return self._predict_photometry_auto(params)
        if mode == "hybrid":
            return self._predict_photometry_hybrid(params)
        if mode == "precomputed":
            raise ValueError(
                "Precomputed mode has been removed. Use mode='hybrid' for fast "
                "approximate photometry or mode='compositional' for exact JIT."
            )
        if mode == "compositional":
            return self._predict_photometry_compositional(params)

        # mode == "exact"
        return self._predict_photometry_exact(params)

    def predict_spectrum(
        self,
        params,
        wave_obs=None,
        mode="auto",
        approx=None,
        wave_chunk_size=None,
    ):
        """Compute observed spectrum at given wavelengths with LSF convolution.

        Evaluates the full SED at custom wavelengths in observed frame,
        applies velocity dispersion broadening (if ``sigma_v`` in spec),
        convolves with instrument line-spread function, and optionally
        applies multiplicative Chebyshev calibration polynomial.

        Parameters
        ----------
        params : dict
            Parameter values using public parameter names.
        wave_obs : array, optional
            Observed-frame wavelength grid [Angstrom]. If None, uses:

            1. Grid from :meth:`precompute_spectroscopy()` if called
            2. Grid from ``observation.spectroscopy.wave_obs`` if set
            3. Raises ValueError if neither available

        mode : str, optional
            Prediction mode (same as :meth:`predict_photometry`).
            Default ``"auto"`` cascades through available kernels.
        approx : bool, optional
            Maps ``True`` → ``mode="auto"`` and ``False`` → ``mode="exact"``.
            Prefer passing ``mode=`` directly.
        wave_chunk_size : int, optional
            If specified, split observed-frame wavelength axis into chunks of
            this size and evaluate via ``jax.lax.map`` to reduce per-chunk HLO
            size for XLA compilation. Default None (no chunking, exact behavior).
            For spectroscopy with R~500 at N≥64 galaxies, typical value is 32–64
            to avoid XLA compilation wall-clock.

        Returns
        -------
        flux : array, shape (n_pix,)
            Observed spectral flux density [erg/s/cm²/Hz] in the AB system
            at the specified wavelengths.

        Raises
        ------
        ValueError
            If ``wave_obs`` is None and no precomputed wavelength grid available.

        Notes
        -----
        **JIT-compatible**: compositional and hybrid modes are JIT'd.
        Exact mode is not JIT'd.

        **Velocity dispersion**: When ``sigma_v`` is in free params,
        applies line-of-sight broadening via Gaussian convolution at
        FWHM = ``2.355 × sigma_v``. Implemented as wavelength-space
        Gaussian convolution (valid for linear pixels; use
        :func:`~tengri.observation.spectrum.apply_lsf` for
        log-wavelength pixels).

        **Line-spread function**: Composition of:

        - Velocity dispersion broadening (σ_v-dependent)
        - Instrument LSF (resolution R-dependent, Gaussian approximation)
        - Chebyshev multiplicative calibration (optional)

        All three are convolved in the forward model.

        **Precomputed wavelength grid**: For fixed-redshift models with
        fixed wavelength grid, call :meth:`precompute_spectroscopy(wave_obs)`
        at initialization to cache spectroscopy kernels. This enables the
        hybrid/compositional paths for ~10× speedup vs. exact.

        **Wavelength-axis chunking**: Set ``wave_chunk_size`` to split the
        observed-frame wavelength axis into ~N/chunk_size chunks and evaluate
        independently via lax.map. Each chunk's HLO is ~1/K of the full HLO
        (K = chunk_size / min_chunk_width), reducing XLA compile-time
        superlinearly. Numerical output is bitwise-identical to unchunked.
        Typical runtime overhead: +5–20% per galaxy due to map overhead.

        Examples
        --------
        >>> wave_obs = np.linspace(4000, 5500, 1000)  # observed frame [Å]
        >>> flux = model.predict_spectrum(params, wave_obs)
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(wave_obs, flux)
        >>> plt.xlabel("Wavelength (Å)")
        >>> plt.ylabel("Flux (erg/s/cm²/Hz)")

        For large spectroscopy sets with many galaxies, use chunking::

            >>> flux = model.predict_spectrum(params, wave_obs, wave_chunk_size=64)

        See Also
        --------
        predict_photometry : Filter-integrated flux (simpler, faster).
        predict : Lazy access to all SED and SFH quantities.
        precompute_spectroscopy : Cache spectroscopy kernels for this grid.
        """
        if wave_obs is None and self._precomputed.spectroscopy is not None:
            wave_obs = self._precomputed.spectroscopy.wave_obs_pixels
        elif wave_obs is None and hasattr(self, "_wave_obs"):
            wave_obs = self._wave_obs
        elif wave_obs is None:
            raise ValueError("No wavelength grid. Pass wave_obs or call precompute_spectroscopy()")

        if approx is not None:
            mode = "auto" if approx else "exact"

        # Use instance default if not overridden
        if wave_chunk_size is None:
            wave_chunk_size = self._wave_chunk_size

        if mode == "_traceable":
            return self._predict_spectrum_traceable(params, wave_obs, wave_chunk_size)

        if mode not in self._PREDICTION_MODES:
            raise ValueError(
                f"Unknown mode {mode!r}. Choose from: {sorted(self._PREDICTION_MODES)}"
            )

        if mode == "auto":
            return self._predict_spectrum_auto(params, wave_obs, wave_chunk_size)
        if mode == "precomputed":
            raise ValueError(
                "Precomputed mode has been removed. Use mode='compositional' for fast "
                "JIT spectrum or mode='exact' for full SED pipeline."
            )
        if mode == "compositional":
            return self._predict_spectrum_compositional(params, wave_obs, wave_chunk_size)
        if mode == "hybrid":
            return self._predict_spectrum_hybrid(params, wave_obs, wave_chunk_size)

        # mode == "exact"
        return self._predict_spectrum_exact(params, wave_obs, wave_chunk_size)

    def predict_magnitudes(self, params):
        """Compute observed AB magnitudes through all filters.

        Parameters
        ----------
        params : dict
            Parameter values using public parameter names.

        Returns
        -------
        magnitudes : ndarray, shape (n_filters,)
            Observed AB magnitudes [mag].

        Notes
        -----
        **JIT-compatible**: yes (via ``predict_photometry`` or ``predict_luminosity``).

        Uses :func:`dsps.calc_obs_mag` when available (cosmology-aware),
        falls back to conversion from photometric flux otherwise.
        """
        if self.filter_waves is None:
            raise ValueError("No filters set.")

        try:
            from dsps import calc_obs_mag

            from tengri.utils.cosmology import DEFAULT_COSMO

            sed_lsun = self.predict_luminosity(params)
            z = self._get_redshift(params)
            cosmo = DEFAULT_COSMO

            mags = []
            for fw, ft in zip(self.filter_waves, self.filter_trans):
                m = calc_obs_mag(
                    self.ssp_data.ssp_wave,
                    sed_lsun,
                    fw,
                    ft,
                    z,
                    cosmo.Om0,
                    cosmo.w0,
                    cosmo.wa,
                    cosmo.h,
                )
                mags.append(m)
            return jnp.array(mags)

        except ImportError:
            flux = self.predict_photometry(params)
            return ab_mag_from_flux(flux)

    def predict_luminosity(self, params):
        """Compute rest-frame luminosity SED in solar units.

        Parameters
        ----------
        params : dict
            Parameter values using public parameter names.

        Returns
        -------
        array, shape (n_wave,)
            Rest-frame luminosity [L_sun/Hz].

        Notes
        -----
        **JIT-compatible**: no — wraps :meth:`predict_rest_sed`.

        Divides rest-frame SED by :math:`L_{\\odot} = 3.828 \\times 10^{33}` erg/s
        (IAU 2015 definition).
        """
        LSUN_CGS = 3.828e33  # erg/s (IAU 2015)
        sed_erg = self.predict_rest_sed(params).sed
        return sed_erg / LSUN_CGS

    def predict_line_fluxes(self, params, target_wavelengths=None):
        """Predict observed emission line fluxes.

        Calls the nebular backend to compute line luminosities,
        selects target lines by wavelength matching, and converts
        from luminosity (Lsun) to observed flux (erg/s/cm^2).

        Parameters
        ----------
        params : dict
            Parameter values (public names).
        target_wavelengths : array, shape (n_target,), optional
            Rest-frame vacuum wavelengths (Angstrom) of lines to predict.
            Each wavelength is matched to the nearest backend line.
            If None, returns all lines from the nebular backend.

        Returns
        -------
        fluxes : array, shape (n_target,) or (n_all_lines,)
            Observed line fluxes in erg/s/cm^2.

        Raises
        ------
        ValueError
            If no nebular backend is configured.

        Notes
        -----
        **JIT-compatible**: no — delegates to nebular backend.

        Observed flux is calculated from luminosity via:

        .. math::

            F = \\frac{L_{\\odot}}{4\\pi d_L^2}

        where :math:`d_L` is the luminosity distance.
        """
        from tengri.utils.physics_constants import L_SUN

        backend = self._nebular_backend
        if backend is None or not hasattr(backend, "predict_nebular_line_luminosities"):
            raise ValueError(
                "No nebular backend with line prediction configured. Cannot compute line fluxes."
            )

        comp = self._compute_sed_components(params)
        weights = comp["weights"]
        p = comp["p"]

        all_waves, all_lums = backend.predict_nebular_line_luminosities(
            ssp_weights=weights,
            ssp_log_ages_yr=self.ssp_log_ages_yr,
            log_z=p.get("log_z_abs", 0.0),
            neb_logU=p.get("neb_logU", -3.0),
            neb_logZ_gas=p.get("neb_logZ_gas", None),
            neb_fesc=p.get("neb_fesc", 0.0),
        )

        if target_wavelengths is not None:
            target_wavelengths = jnp.asarray(target_wavelengths)
            indices = jnp.argmin(
                jnp.abs(all_waves[None, :] - target_wavelengths[:, None]),
                axis=1,
            )
            selected_lums = all_lums[indices]
        else:
            selected_lums = all_lums

        dl_cm = self._get_dl_cm(params)
        flux = selected_lums * L_SUN / (4.0 * jnp.pi * dl_cm**2)
        return flux

    def predict_spectral_indices(self, params, index_defs, mode="_traceable"):
        """Predict spectral index values from the model SED.

        Generates a rest-frame spectrum covering the index wavelength
        ranges and measures each index (EW or break ratio).

        Parameters
        ----------
        params : dict
            Parameter values (public names).
        index_defs : tuple of SpectralIndexDef
            Index definitions to measure.
        mode : str, optional
            Forward model prediction mode.

        Returns
        -------
        jnp.ndarray, shape (n_indices,)
            Predicted index values.

        Notes
        -----
        **JIT-compatible**: depends on ``mode`` (``"_traceable"`` by default).

        Measures spectral indices (equivalent width or break ratio) from a
        rest-frame spectrum covering all wavelength ranges in ``index_defs``.
        """
        from tengri.observation.spectral_indices import measure_index_jax

        wave_min = min(d.wave_min for d in index_defs)
        wave_max = max(d.wave_max for d in index_defs)

        z = params.get("redshift", 0.0)
        wave_obs = jnp.linspace(
            wave_min * (1.0 + z) * 0.98,
            wave_max * (1.0 + z) * 1.02,
            2000,
        )

        flux_obs = self.predict_spectrum(params, wave_obs, mode=mode)
        wave_rest = wave_obs / (1.0 + z)

        indices = []
        for idx_def in index_defs:
            val = measure_index_jax(wave_rest, flux_obs, idx_def)
            indices.append(val)
        return jnp.array(indices)

    def predict_hbeta(self, params: dict) -> float:
        """Predict Hβ luminosity for use with CLOUDY-informed emission line priors.

        Required by ``marginalize_emission_lines_cloudy()`` as the ``l_hbeta``
        argument, which scales CLOUDY's ratio-relative-to-Hβ priors to physical
        units.

        Hβ luminosity is computed via the Case B recombination approximation
        (Leitherer et al. 1999):

        .. math::

            L_{H\\beta} \\approx 5.22 \\times 10^7 \\times \\text{SFR}_{10} \\; [L_\\odot]

        where :math:`\\text{SFR}_{10}` is the SFR averaged over the last 10 Myr
        (the ionizing-photon relevant timescale), derived from
        Q_H ≈ 4.2 × 10⁵³ × SFR [photons/s] and
        L_Hβ = 4.76 × 10⁻¹³ × Q_H erg/s converted to L_sun.

        Parameters
        ----------
        params : dict
            Model parameters (from ``spec.sample()`` or a ``Posterior``).

        Returns
        -------
        float
            Hβ luminosity [Lsun].

        Examples
        --------
        >>> l_hbeta = model.predict_hbeta(params)
        >>> ln_L = marginalize_emission_lines_cloudy(
        ...     residual,
        ...     noise,
        ...     A,
        ...     log_z=params["met_logzsol"],
        ...     neb_logU=-3.0,
        ...     l_hbeta=l_hbeta,
        ... )

        Notes
        -----
        **JIT-compatible**: no — wraps :meth:`predict_sfh_quantities`.

        Uses Case B recombination coefficients (Leitherer et al. 1999 [1]_).
        If SFH computation fails (e.g., invalid params), returns safe fallback of 1 L_sun.

        See Also
        --------
        predict_sfh_quantities : JIT-compatible SFH quantities including sfr_10myr.

        References
        ----------
        .. [1] C. Leitherer et al., "Starburst99: Synthesis Models for Galaxies
           with Active Star Formation," ApJS, 123, 3 (1999).
           arXiv:astro-ph/9807340.
        """
        # Case B: L_Hbeta [Lsun] = 4.76e-13 * Q_H, Q_H = 4.2e53 * SFR [Msun/yr]
        # => L_Hbeta = 4.76e-13 * 4.2e53 / 3.828e33 * SFR ≈ 5.22e7 * SFR
        _L_HBETA_PER_SFR = 5.22e7  # Lsun per Msun/yr (Leitherer+1999)
        try:
            sfh_q = self.predict_sfh_quantities(params)
            sfr_10 = float(sfh_q.sfr_10myr)
            sfr_10 = max(sfr_10, 1e-10)
            return float(_L_HBETA_PER_SFR * sfr_10)
        except (AttributeError, TypeError, ValueError):
            # AttributeError: predict_sfh_quantities doesn't exist or sfr_10myr missing
            # TypeError: float() conversion failed (JAX tracer or wrong type)
            # ValueError: invalid params
            return 1.0  # 1 Lsun safe fallback

    def predict_derived(self, params):
        """Compute derived physical quantities as a flat dict.

        Convenience wrapper around :meth:`predict` that extracts the key
        SFH-derived scalars into a plain dict. Use :meth:`predict` for
        lazy on-demand access to all quantities, or
        :meth:`predict_sfh_quantities` for JIT-compatible batch computation.

        Parameters
        ----------
        params : dict
            Parameter values.

        Returns
        -------
        dict with keys:
            "stellar_mass": total mass formed [M_sun]
            "stellar_mass_surviving": surviving mass in living stars +
                remnants [M_sun] or None if mass-remaining table not loaded.
            "sfr_100myr": SFR averaged over last 100 Myr [M_sun/yr]
            "sfr_10myr": SFR averaged over last 10 Myr [M_sun/yr]
            "ssfr": specific SFR [yr^-1], uses surviving mass if
                available, else formed mass.

        Notes
        -----
        **JIT-compatible**: no — wraps :meth:`predict`.

        Convenience wrapper around the lazy :meth:`predict` object.
        For batch operations, use :meth:`predict_sfh_quantities` directly
        with :func:`jax.vmap`.
        """
        pred = self.predict(params)
        mass_surv = pred.sfh.stellar_mass_surviving
        # Return None (not NaN) when mass-remaining table is absent
        mass_surv_out = None if jnp.isnan(mass_surv) else mass_surv
        return {
            "stellar_mass": pred.sfh.stellar_mass,
            "stellar_mass_surviving": mass_surv_out,
            "sfr_100myr": pred.sfh.sfr_100myr,
            "sfr_10myr": pred.sfh.sfr_10myr,
            "ssfr": pred.sfh.ssfr,
        }

    def predict_sfh_quantities(self, params):
        """Compute SFH-derived quantities in JIT-compatible form.

        Integrates the SFH to compute stellar mass, recent SFR, specific SFR,
        and mass-weighted age/metallicity. Returns a :class:`SFHQuantities`
        NamedTuple that is fully JIT-compatible and vmap-ready for batch
        inference over posterior chains or mock catalogs.

        Parameters
        ----------
        params : dict
            Parameter values using public parameter names.

        Returns
        -------
        SFHQuantities
            NamedTuple with fields:

            - ``stellar_mass`` : float. Total stellar mass formed [M☉]
            - ``stellar_mass_surviving`` : float. Mass in living stars + remnants [M☉],
              or NaN if SSP mass-remaining tables not loaded.
            - ``sfr_100myr`` : float. SFR time-averaged over last 100 Myr [M☉/yr]
            - ``sfr_10myr`` : float. SFR time-averaged over last 10 Myr [M☉/yr]
            - ``ssfr`` : float. Specific SFR (SFR/M_surv or SFR/M_formed) [yr⁻¹]
            - ``mass_weighted_age_gyr`` : float. Mass-weighted age [Gyr]
            - ``mass_weighted_metallicity`` : float. Mass-weighted log₁₀(Z/Z☉) or
              absolute log₁₀(Z) depending on metallicity mode

        Notes
        -----
        **JIT-compatible**: yes — all operations use ``jnp`` primitives.
        Safe inside :func:`jax.jit`, :func:`jax.vmap`, and :func:`jax.grad`.

        **Gradient-safe**: yes — all quantities are differentiable w.r.t.
        SFH and metallicity parameters.

        **Surviving mass**: Requires SSP grid with ``ssp_mass_remaining``
        (e.g., FSPS grids). If unavailable, returns NaN. :meth:`predict`
        handles NaN gracefully when the quantity is unavailable.

        **SFR averaging**: Time-weighted mean over lookback-time window:

        .. math::

            \\langle\\mathrm{SFR}\\rangle_T =
                \\frac{\\sum_i \\mathrm{SFR}_i \\Delta t_i}{\\sum_i \\Delta t_i}

        where :math:`i` ranges over all ages :math:`\\leq T`. Uses symmetric
        bin widths (``jnp.gradient``) to avoid trapezoid boundary artifacts.

        **Mass-weighted age**: Computed as

        .. math::

            t_\\mathrm{mw} = \\frac{\\sum_i w_i t_i}{\\sum_i w_i}

        where :math:`w_i` are stellar population weights (age-integrated SFR).

        Examples
        --------
        **Single galaxy:**

        >>> sfh = model.predict_sfh_quantities(params)
        >>> sfh.stellar_mass
        Array(1.23e10, dtype=float64)

        **Batch over 10,000 posterior samples:**

        >>> import jax
        >>> sfh_fn = jax.vmap(model.predict_sfh_quantities)
        >>> sfh_batch = sfh_fn(params_batch)
        >>> sfh_batch.stellar_mass  # shape (10000,)
        >>> print(sfh_batch.stellar_mass.mean())

        See Also
        --------
        predict : Lazy prediction for single-galaxy exploration (non-JIT).
        predict_sfh : SFH on linear-time grid for visualization.
        predict_sed_quantities : JIT-compatible SED quantities.
        """
        from tengri.forward.prediction import SFHQuantities
        from tengri.utils.sed_quantities import (
            compute_mass_weighted_age,
            compute_mass_weighted_metallicity,
        )

        p = self._get_internal_params(params)
        sfr = self._compute_sfr(p)

        sfr_on_ssp = jnp.interp(self.ssp_log_ages_yr, self.log_age_grid, sfr)
        if self._csp_integration == "log_interp":
            weights = self._csp_matrix @ sfr_on_ssp
        elif self._csp_integration == "dsps_native":
            # For stellar_mass(), only age_weights matter (not ssp_flux_at_z).
            from tengri.components.sps.dsps_wrapper import compute_dsps_native_weights

            z_val = p.get("redshift", 0.1)
            t_obs_gyr = self._t_universe_gyr(z_val)
            lgmet = p.get("log_z_abs", -1.8477)
            lgmet_scatter = float(p.get("lgmet_scatter", self._lgmet_scatter))
            weights, _ = compute_dsps_native_weights(
                sfr_on_ssp,
                self.ssp_ages_yr,
                self.ssp_data.ssp_lgmet,
                self.ssp_data.ssp_lg_age_gyr,
                self.ssp_data.ssp_flux,
                t_obs_gyr,
                lgmet,
                lgmet_scatter,
            )
        elif self._csp_integration == "dsps_met_table":
            from tengri.components.sps.dsps_wrapper import compute_dsps_met_table_weights

            z_val = p.get("redshift", 0.1)
            t_obs_gyr = self._t_universe_gyr(z_val)
            lgmet_scatter = float(p.get("lgmet_scatter", self._lgmet_scatter))
            if self._met_mode == "ramp":
                from tengri.components.sps.dsps_wrapper import compute_log_z_evolving

                lgmet_per_age = compute_log_z_evolving(
                    self.ssp_data.ssp_lg_age_gyr,
                    p["log_z_abs_initial"],
                    p["log_z_abs_final"],
                    t_obs_gyr,
                )
            else:
                lgmet_per_age = jnp.full_like(self.ssp_ages_yr, p.get("log_z_abs", -1.8477))
            weights, _ = compute_dsps_met_table_weights(
                sfr_on_ssp,
                lgmet_per_age,
                self.ssp_ages_yr,
                self.ssp_data.ssp_lgmet,
                self.ssp_data.ssp_lg_age_gyr,
                self.ssp_data.ssp_flux,
                t_obs_gyr,
                lgmet_scatter,
            )
        else:
            weights = sfr_on_ssp * self._csp_age_dt
        mass_formed = jnp.sum(weights)

        # Surviving mass
        if self.ssp_data.ssp_mass_remaining is not None:
            from tengri.components.sps.dsps_wrapper import (
                compute_surviving_mass,
                interpolate_mass_remaining,
            )

            log_z = p.get("log_z_abs", 0.0)
            mr_at_met = interpolate_mass_remaining(
                self.ssp_data.ssp_mass_remaining,
                self.ssp_data.ssp_lgmet,
                log_z,
            )
            mass_surviving = compute_surviving_mass(weights, mr_at_met)
        else:
            mass_surviving = jnp.array(jnp.nan)

        # SFR averages — time-weighted mean over a lookback-time window.
        # <SFR>_T = sum(SFR_i * dt_i) / sum(dt_i)  for all age_i <= T.
        # Use jnp.gradient for symmetric bin widths; avoids the trapezoid boundary
        # artifact where zeroing SFR outside the window but keeping the full age
        # axis creates a phantom half-bin contribution at the window edge.
        dt = jnp.gradient(self.age_yr)
        mask_100 = self.age_yr <= 1e8
        numerator_100 = jnp.sum(jnp.where(mask_100, sfr * dt, 0.0))
        denom_100 = jnp.maximum(jnp.sum(jnp.where(mask_100, dt, 0.0)), 1.0)
        sfr_100myr = jnp.where(jnp.sum(mask_100) > 1, numerator_100 / denom_100, sfr[0])

        mask_10 = self.age_yr <= 1e7
        numerator_10 = jnp.sum(jnp.where(mask_10, sfr * dt, 0.0))
        denom_10 = jnp.maximum(jnp.sum(jnp.where(mask_10, dt, 0.0)), 1.0)
        sfr_10myr = jnp.where(jnp.sum(mask_10) > 1, numerator_10 / denom_10, sfr[0])

        # sSFR
        mass_for_ssfr = jnp.where(jnp.isnan(mass_surviving), mass_formed, mass_surviving)
        ssfr = sfr_100myr / jnp.maximum(mass_for_ssfr, 1.0)

        # Mass-weighted age and metallicity
        mw_age = compute_mass_weighted_age(weights, self.ssp_ages_yr)
        mw_z = compute_mass_weighted_metallicity(
            weights,
            self.ssp_ages_yr,
            p.get("log_z_abs", 0.0),
            log_z_initial=p.get("log_z_abs_initial"),
            log_z_final=p.get("log_z_abs_final"),
        )

        return SFHQuantities(
            stellar_mass=mass_formed,
            stellar_mass_surviving=mass_surviving,
            sfr_100myr=sfr_100myr,
            sfr_10myr=sfr_10myr,
            ssfr=ssfr,
            mass_weighted_age_gyr=mw_age,
            mass_weighted_metallicity=mw_z,
        )

    def predict_sed_quantities(self, params):
        """Compute SED-derived quantities in JIT-compatible form.

        Evaluates the full forward model and computes UV slope, spectral
        indices (D4000, Balmer break), bolometric/IR luminosities, dust
        attenuation, and luminosity-weighted age/metallicity. Returns
        a :class:`SEDQuantities` NamedTuple that is fully JIT-compatible
        and vmap-ready for batch inference.

        Parameters
        ----------
        params : dict
            Parameter values using public parameter names.

        Returns
        -------
        SEDQuantities
            NamedTuple with fields:

            - ``l_bol`` : float. Bolometric luminosity [L☉]
            - ``l_tir`` : float. Total infrared (8–1000 μm) luminosity [L☉]
            - ``l_dust_absorbed`` : float. Dust-absorbed luminosity [L☉]
              (intrinsic − attenuated), or NaN if intrinsic SED unavailable.
            - ``irx`` : float. Infrared excess := L_TIR / L_UV(1600 Å).
              Common probe of dust obscuration (Dale et al. 2001).
            - ``uv_slope_beta`` : float. UV slope (power-law index) in
              f_λ ∝ λ^β for 1200–2600 Å.
            - ``dn4000`` : float. D_n(4000) break ratio: flux average
              at 3750–3950 Å / 4050–4250 Å. Indicator of stellar age.
            - ``balmer_break`` : float. Balmer break: flux ratio
              ~3700 Å / ~4000 Å. Old stellar population signature.
            - ``m_uv`` : float. Absolute magnitude at 1500 Å
              (M_1500, standard reionization-era indicator).
            - ``fuv_flux`` : float. Flux at 1500 Å [erg/s/cm²]
            - ``nuv_flux`` : float. Flux at 2300 Å [erg/s/cm²]
            - ``fuv_flux_intrinsic`` : float. FUV flux, dust-free
              (intrinsic SED). NaN if unavailable.
            - ``nuv_flux_intrinsic`` : float. NUV flux, dust-free. NaN
              if unavailable.
            - ``rest_uv_color`` : float. Rest-frame UV color (f_1500 − f_2300).
            - ``luminosity_weighted_age_gyr`` : float. Luminosity-weighted
              age [Gyr] (∫L_λ age dλ / ∫L_λ dλ).
            - ``luminosity_weighted_metallicity`` : float. Luminosity-weighted
              log₁₀(Z/Z☉) or absolute log₁₀(Z).

        Notes
        -----
        **JIT-compatible**: yes — all operations use ``jnp`` primitives.
        Safe inside :func:`jax.jit`, :func:`jax.vmap`, and :func:`jax.grad`.

        **Gradient-safe**: yes — all quantities are differentiable w.r.t.
        SFH, metallicity, and dust parameters.

        **Spectral indices**: Computed directly on the rest-frame SED
        (not broadband-filtered). All wavelengths defined in rest frame.

        **Dust-absorbed luminosity**: Defined as L_dust = L_intrinsic − L_attenuated
        (i.e., the energy re-radiated in the IR). Requires the forward model
        to track both intrinsic and attenuated SEDs internally. Returns NaN if
        ``dust_model="none"`` or intrinsic SED not available.

        **Luminosity-weighted quantities**: Computed as:

        .. math::

            \\langle Q \\rangle_L = \\frac{\\int L_\\lambda(\\lambda) Q(\\lambda) d\\lambda}
                                        {\\int L_\\lambda(\\lambda) d\\lambda}

        More sensitive to young, UV-bright populations than mass-weighted age.

        Examples
        --------
        **Single galaxy:**

        >>> sed_q = model.predict_sed_quantities(params)
        >>> sed_q.l_bol
        Array(2.5e10, dtype=float64)
        >>> sed_q.dn4000
        Array(1.42, dtype=float64)
        >>> sed_q.irx
        Array(1.87, dtype=float64)

        **Batch over posterior samples:**

        >>> import jax
        >>> sed_fn = jax.vmap(model.predict_sed_quantities)
        >>> sed_batch = sed_fn(params_batch)
        >>> sed_batch.m_uv  # shape (n_samples,)
        >>> sed_batch.dn4000.mean()

        **Computing IRX − β relation:**

        >>> sed_q = sed_fn(params_batch)
        >>> irx = sed_q.irx
        >>> beta = sed_q.uv_slope_beta
        >>> # Compare to Meurer et al. (1999) IRX-β calibration

        See Also
        --------
        predict : Lazy prediction for single-galaxy exploration.
        predict_sfh_quantities : JIT-compatible SFH quantities.
        predict_rest_sed : Full rest-frame SED (for custom analysis).
        """
        from tengri.forward.prediction import SEDQuantities
        from tengri.utils.sed_quantities import (
            compute_balmer_break,
            compute_bolometric_luminosity,
            compute_dn4000,
            compute_fuv_flux,
            compute_irx,
            compute_l_dust_absorbed,
            compute_l_tir,
            compute_luminosity_weighted_age,
            compute_luminosity_weighted_metallicity,
            compute_m_uv,
            compute_nuv_flux,
            compute_rest_uv_color,
            compute_uv_luminosity_1600,
            compute_uv_slope_beta,
        )

        comp = self._compute_sed_components(params, need_intrinsic=True)
        sed = comp["sed_total"]
        wave = self.ssp_data.ssp_wave
        p = comp["p"]

        l_bol = compute_bolometric_luminosity(sed, wave)
        l_tir = compute_l_tir(sed, wave)

        sed_intr = comp["sed_intrinsic"]
        sed_atten = comp["sed_attenuated"]
        l_dust = (
            compute_l_dust_absorbed(sed_intr, sed_atten, wave)
            if sed_intr is not None
            else jnp.array(jnp.nan)
        )

        l_uv = compute_uv_luminosity_1600(sed, wave)
        irx = compute_irx(l_tir, l_uv)

        # Intrinsic UV fluxes
        fuv_intr = compute_fuv_flux(sed_intr, wave) if sed_intr is not None else jnp.array(jnp.nan)
        nuv_intr = compute_nuv_flux(sed_intr, wave) if sed_intr is not None else jnp.array(jnp.nan)

        # Luminosity-weighted quantities
        weights = comp["weights"]
        ssp_flux_at_z = comp["ssp_flux_at_z"]
        lw_age = compute_luminosity_weighted_age(weights, ssp_flux_at_z, self.ssp_ages_yr, wave)
        lw_z = compute_luminosity_weighted_metallicity(
            weights,
            ssp_flux_at_z,
            self.ssp_ages_yr,
            wave,
            p.get("log_z_abs", 0.0),
            log_z_initial=p.get("log_z_abs_initial"),
            log_z_final=p.get("log_z_abs_final"),
        )

        return SEDQuantities(
            l_bol=l_bol,
            l_tir=l_tir,
            l_dust_absorbed=l_dust,
            irx=irx,
            uv_slope_beta=compute_uv_slope_beta(sed, wave),
            dn4000=compute_dn4000(sed, wave),
            balmer_break=compute_balmer_break(sed, wave),
            m_uv=compute_m_uv(sed, wave),
            fuv_flux=compute_fuv_flux(sed, wave),
            nuv_flux=compute_nuv_flux(sed, wave),
            fuv_flux_intrinsic=fuv_intr,
            nuv_flux_intrinsic=nuv_intr,
            rest_uv_color=compute_rest_uv_color(sed, wave),
            luminosity_weighted_age_gyr=lw_age,
            luminosity_weighted_metallicity=lw_z,
        )

    # ── Batch operations ──────────────────────────────────────────────

    def predict_photometry_batch(self, params_batch):
        """Compute photometry for a batch of parameter sets via jax.vmap.

        Parameters
        ----------
        params_batch : dict of arrays
            Each value has shape (N, ...) with leading batch dimension.

        Returns
        -------
        array, shape (N, n_filters)
            Photometric flux for each galaxy.

        Notes
        -----
        **JIT-compatible**: yes — uses :func:`jax.vmap` over
        :meth:`predict_photometry`.

        Examples
        --------
        >>> import jax
        >>> key = jax.random.PRNGKey(0)
        >>> params_batch = {
        ...     k: jnp.tile(v[None], (100,) + (1,) * (len(v.shape)))
        ...     for k, v in posterior.samples.items()
        ... }
        >>> flux_batch = model.predict_photometry_batch(params_batch)
        """
        from tengri.forward.convenience import predict_photometry_batch as _fn

        return _fn(self, params_batch)

    def predict_spectrum_batch(self, params_batch):
        """Compute spectra for a batch of parameter sets via jax.vmap.

        Parameters
        ----------
        params_batch : dict of arrays
            Each value has leading batch dimension.

        Returns
        -------
        array, shape (N, n_pix)
            Spectral flux for each galaxy.

        Notes
        -----
        **JIT-compatible**: yes — uses :func:`jax.vmap` over
        :meth:`predict_spectrum`.

        Examples
        --------
        >>> params_batch = {
        ...     k: jnp.tile(v[None], (1000,) + (1,) * (len(v.shape)))
        ...     for k, v in posterior.samples.items()
        ... }
        >>> flux_batch = model.predict_spectrum_batch(params_batch)
        >>> flux_batch.shape
        (1000, n_pix)
        """
        from tengri.forward.convenience import predict_spectrum_batch as _fn

        return _fn(self, params_batch)

    # ── Private prediction dispatch ───────────────────────────────────

    def _predict_photometry_auto(self, params):
        """Auto mode: pick fastest available (Compositional → Hybrid → Exact).

        Compositional is preferred over hybrid because XLA fuses the
        entire graph (SFH → SED → filter integration) into one optimized
        kernel, which is faster than splitting into precomputed stellar
        + Python-dispatched non-stellar filter integration.  Hybrid is
        the fallback when compositional is unavailable.

        Tabulated SFH and standard parametric SFH are both handled by the
        compositional path: SFH is evaluated in Python before JIT entry, so
        the JIT closure is SFH-type-independent.  Evolving-metallicity and
        chem-evol models fall back inside ``_predict_photometry_compositional``.
        """
        import warnings

        # Compositional: full-resolution JIT (bit-exact, default)
        if self._compositional.photometry is not None:
            return self._predict_photometry_compositional(params)

        # Hybrid: precomputed SSP×filter (faster but ~0.2% approx, fallback)
        # Tabulated SFH not supported in hybrid (variable-size arrays).
        if self._hybrid.photometry is not None and "sfh_t_gyr" not in params:
            return self._predict_photometry_hybrid(params)

        warnings.warn(
            "mode='auto' requested but no fast path available, using exact path",
            stacklevel=3,
        )
        return self._predict_photometry_exact(params)

    def _predict_photometry_exact(self, params):
        """Exact photometry: full SED pipeline + filter integration."""
        obs_sed = self.predict_obs_sed(params)
        z = self._get_redshift(params)
        dl_cm = self._get_dl_cm(params)

        if self.observation is not None:
            return self.observation.observe_photometry(obs_sed, z, dl_cm)

        from tengri.observation.photometry import compute_flux_density

        wave_rest = obs_sed.wavelength / (1.0 + z)
        fluxes = []
        for fw, ft in zip(self.filter_waves, self.filter_trans):
            f = compute_flux_density(obs_sed.sed, wave_rest, fw, ft, z, dl_cm)
            fluxes.append(f)
        return jnp.array(fluxes)

    def _predict_photometry_hybrid(self, params):
        """Hybrid photometry: precomputed SSP + exact non-stellar.

        Uses precomputed SSP×filter for stellar (~0.4% error), and
        emission_helpers at full wavelength for non-stellar (exact).

        The kernel is fully fused: params dict → photometry in one JIT
        call (no Python-side SFH or param translation overhead).
        """
        if self._hybrid.photometry is None:
            return self._predict_photometry_auto(params)

        return self._hybrid.photometry(params)

    def _predict_photometry_traceable(self, params):
        """Un-JIT'd photometry for use inside JAX tracing (NIFTy VI).

        Returns the same result as hybrid/compositional but without
        @jax.jit on the outer wrapper, so it can be traced by an
        enclosing jax.jit (e.g. NIFTy's signal_response).

        IMPORTANT: uses _hybrid_kernels/_compositional_kernels directly
        (not the lazy property) to avoid building kernels inside a JIT scope.
        """
        # Prefer hybrid raw (already built at init)
        if self._hybrid_kernels is not None:
            raw = getattr(self._hybrid_kernels, "_photometry_raw", None)
            if raw is not None:
                return raw(params)
        # Fall back to compositional raw
        if self._compositional_kernels is not None:
            raw = getattr(self._compositional_kernels, "_photometry_raw", None)
            if raw is not None:
                p = self._get_internal_params(params)
                sfr = self._compute_sfr(p)
                sfr_on_ssp = jnp.interp(self.ssp_log_ages_yr, self.log_age_grid, sfr)
                return raw(sfr_on_ssp, params)
        # Last resort: exact (slow but always works)
        return self._predict_photometry_exact(params)

    def _predict_photometry_compositional(self, params):
        """Photometry via Compositional: rest SED + filter integration.

        Uses the compositional end-to-end JIT kernel when available.  SFH is
        evaluated in Python (before JIT entry) and passed as a traced array, so
        the JIT closure is SFH-type-independent and does not recompile on SFH
        type changes.

        Evolving-metallicity and chem-evol models cannot use the single-Z JIT
        and fall back to ``_compute_rest_sed_compositional`` + Python filter
        integration.
        """
        if self._compositional.photometry is not None:
            # Evolving-Z / chem-evol: the end-to-end JIT has no met-table path;
            # fall back to the REST-SED kernel + Python filter integration.
            if self._met_mode == "ramp" or self._met_mode == "chem_evol":
                rest_sed = self._compute_rest_sed_compositional(params)
                z = self._get_redshift(params)
                dl_cm = self._get_dl_cm(params)
                return observe_photometry_from_rest_sed(
                    rest_sed,
                    self._rest_wavelength,
                    z,
                    dl_cm,
                    self.filter_waves,
                    self.filter_trans,
                    apply_igm=self._uses_igm,
                    igm_fn=self._igm_fn,
                )
            # Standard path: compute sfr_on_ssp in Python, pass into JIT.
            p = self._get_internal_params(params)
            sfr = self._compute_sfr(p)
            sfr_on_ssp = jnp.interp(self.ssp_log_ages_yr, self.log_age_grid, sfr)
            return self._compositional.photometry(sfr_on_ssp, params)

        rest_sed = self._compute_rest_sed_compositional(params)
        z = self._get_redshift(params)
        dl_cm = self._get_dl_cm(params)
        return observe_photometry_from_rest_sed(
            rest_sed,
            self._rest_wavelength,
            z,
            dl_cm,
            self.filter_waves,
            self.filter_trans,
            apply_igm=self._uses_igm,
        )

    def _predict_spectrum_auto(self, params, wave_obs, wave_chunk_size=None):
        """Auto mode: pick fastest available spectrum path.

        The compositional path now handles all SFH types (including tabulated)
        because SFH is evaluated outside the JIT.  Evolving-metallicity and
        chem-evol models fall back inside ``_predict_spectrum_compositional``.
        """
        import warnings

        # Compositional (full resolution)
        if self._compositional.rest_sed is not None:
            return self._predict_spectrum_compositional(params, wave_obs, wave_chunk_size)

        warnings.warn(
            "mode='auto' requested but no fast path available, using exact path",
            stacklevel=3,
        )
        return self._predict_spectrum_exact(params, wave_obs, wave_chunk_size)

    def _predict_spectrum_exact(self, params, wave_obs, wave_chunk_size=None):
        """Exact spectrum: full SED pipeline + interpolation."""
        obs_sed = self.predict_obs_sed(params)
        z = self._get_redshift(params)
        dl_cm = self._get_dl_cm(params)

        if self.observation is not None and self.observation.spectroscopy is not None:
            return self.observation.observe_spectrum(obs_sed, z, dl_cm)

        wave_rest = obs_sed.wavelength / (1.0 + z)

        # Apply wavelength-axis chunking if requested
        if wave_chunk_size is not None and wave_chunk_size > 0:
            flux = self._compute_spectrum_chunked(
                obs_sed.sed, wave_rest, wave_obs, z, dl_cm, wave_chunk_size
            )
        else:
            flux = compute_spectrum(obs_sed.sed, wave_rest, wave_obs, z, dl_cm)

        resolution = self._lsf_resolution
        if resolution is not None:
            flux = apply_lsf(
                flux,
                wave_obs,
                resolution,
                sigma_lib_kms=self._sigma_lib_kms,
                n_bins=self._lsf_n_bins,
            )
        return flux

    def _predict_spectrum_hybrid(self, params, wave_obs, wave_chunk_size=None):
        """Hybrid spectrum: precomputed SSP + exact non-stellar.

        Uses precomputed SSP templates on spectral pixels for stellar
        (exact on the grid), and emission_helpers at full wavelength
        for non-stellar (exact).

        The kernel is fully fused: params dict → spectrum in one JIT call.
        """
        if self._hybrid.spectrum is None:
            return self._predict_spectrum_auto(params, wave_obs, wave_chunk_size)

        p = self._get_internal_params(params)
        sfr = self._compute_sfr(p)
        sfr_on_ssp = jnp.interp(self.ssp_log_ages_yr, self.log_age_grid, sfr)
        flux = self._hybrid.spectrum(sfr_on_ssp, params, **self._get_non_stellar_kwargs(p))

        # Apply LSF if needed
        resolution = self._lsf_resolution
        if resolution is not None:
            from tengri.observation.spectrum import apply_lsf

            flux = apply_lsf(
                flux,
                wave_obs,
                resolution,
                sigma_lib_kms=self._sigma_lib_kms,
                n_bins=self._lsf_n_bins,
            )

        return flux

    def _predict_spectrum_traceable(self, params, wave_obs=None, wave_chunk_size=None):
        """Un-JIT'd spectrum for use inside JAX tracing (NIFTy VI).

        Mirrors _predict_photometry_traceable: uses the hybrid spectrum kernel
        (precomputed SSPs on pixel grid, ~200 pts) rather than the full
        compositional rest-SED path (~10k wavelengths).  This reduces the
        XLA graph size during VI by ~50×.

        Requires spectroscopy precomputation.  Falls back to the compositional
        auto path if no precomputed spectrum kernel is available.
        """
        # Prefer hybrid spectrum raw (precomputed SSP on pixel grid)
        if self._hybrid_kernels is not None:
            raw = getattr(self._hybrid_kernels, "_spectrum_raw", None)
            if raw is not None:
                p = self._get_internal_params(params)
                sfr = self._compute_sfr(p)
                sfr_on_ssp = jnp.interp(self.ssp_log_ages_yr, self.log_age_grid, sfr)
                return raw(sfr_on_ssp, params)
        # Fall back to auto path (compositional rest-SED)
        return self._predict_spectrum_auto(params, wave_obs, wave_chunk_size)

    def _predict_spectrum_compositional(self, params, wave_obs, wave_chunk_size=None):
        """Spectrum via Compositional: rest SED + interpolation.

        Uses the compositional end-to-end JIT kernel when available and the
        wave_obs grid matches the precomputed grid.  SFH is evaluated in Python
        before JIT entry and passed as a traced array.  Evolving-metallicity and
        chem-evol models fall back to the rest-SED kernel path.
        """
        if (
            self._compositional.spectrum is not None
            and self._precomputed.spectroscopy is not None
            and wave_obs is self._precomputed.spectroscopy.wave_obs_pixels
            and self._met_mode != "ramp"
            and self._met_mode != "chem_evol"
        ):
            p = self._get_internal_params(params)
            sfr = self._compute_sfr(p)
            sfr_on_ssp = jnp.interp(self.ssp_log_ages_yr, self.log_age_grid, sfr)
            flux = self._compositional.spectrum(sfr_on_ssp, params)
            # Apply LSF if needed (below)
        else:
            rest_sed = self._compute_rest_sed_compositional(params)
            z = self._get_redshift(params)
            dl_cm = self._get_dl_cm(params)

            # Apply wavelength-axis chunking if requested
            if wave_chunk_size is not None and wave_chunk_size > 0:
                flux = self._observe_spectrum_from_rest_sed_chunked(
                    rest_sed, self._rest_wavelength, wave_obs, z, dl_cm, wave_chunk_size
                )
            else:
                flux = observe_spectrum_from_rest_sed(
                    rest_sed, self._rest_wavelength, wave_obs, z, dl_cm
                )

        # Apply LSF convolution if resolution profile is set
        resolution = self._lsf_resolution
        if resolution is not None:
            flux = apply_lsf(
                flux,
                wave_obs,
                resolution,
                sigma_lib_kms=self._sigma_lib_kms,
                n_bins=self._lsf_n_bins,
            )

        return flux

    def _compute_rest_sed_compositional(self, params):
        """Compute rest-frame SED via the compositional JIT kernel.

        Handles SFH computation, metallicity interpolation, and delegates
        the rest (dust, nebular, AGN, radio, X-ray) to the compositional kernel.

        Parameters
        ----------
        params : dict
            Parameter values (public names).

        Returns
        -------
        array, shape (n_wave,)
            Rest-frame SED in erg/s/Hz.
        """
        from tengri.forward.pipeline import interp_met_alpha_dispatch, interp_metallicity

        p = self._get_internal_params(params)
        sfr = self._compute_sfr(p)
        sfr_on_ssp = jnp.interp(self.ssp_log_ages_yr, self.log_age_grid, sfr)
        if self._csp_integration == "log_interp":
            weights = self._csp_matrix @ sfr_on_ssp
        elif self._csp_integration == "dsps_native":
            from tengri.components.sps.dsps_wrapper import compute_dsps_native_weights

            z_val = p.get("redshift", 0.0)
            t_obs_gyr = self._t_universe_gyr(z_val) if hasattr(self, "_t_universe_gyr") else 13.7
            lgmet = p.get("log_z_abs", -1.8477)
            lgmet_scatter = float(p.get("lgmet_scatter", self._lgmet_scatter))
            weights, ssp_flux_at_z = compute_dsps_native_weights(
                sfr_on_ssp,
                self.ssp_ages_yr,
                self.ssp_data.ssp_lgmet,
                self.ssp_data.ssp_lg_age_gyr,
                self.ssp_data.ssp_flux,
                t_obs_gyr,
                lgmet,
                lgmet_scatter,
            )
            if self._uses_xray:
                p = {**p, "_sfr_current": sfr[-1]}
            return self._compositional.rest_sed(weights, ssp_flux_at_z, p)
        elif self._csp_integration == "dsps_met_table":
            from tengri.components.sps.dsps_wrapper import compute_dsps_met_table_weights

            z_val = p.get("redshift", 0.0)
            t_obs_gyr = self._t_universe_gyr(z_val) if hasattr(self, "_t_universe_gyr") else 13.7
            lgmet_scatter = float(p.get("lgmet_scatter", self._lgmet_scatter))
            if self._met_mode == "ramp":
                from tengri.components.sps.dsps_wrapper import compute_log_z_evolving

                lgmet_per_age = compute_log_z_evolving(
                    self.ssp_data.ssp_lg_age_gyr,
                    p["log_z_abs_initial"],
                    p["log_z_abs_final"],
                    t_obs_gyr,
                )
            else:
                lgmet_per_age = jnp.full_like(self.ssp_ages_yr, p.get("log_z_abs", -1.8477))
            weights, ssp_flux_at_z = compute_dsps_met_table_weights(
                sfr_on_ssp,
                lgmet_per_age,
                self.ssp_ages_yr,
                self.ssp_data.ssp_lgmet,
                self.ssp_data.ssp_lg_age_gyr,
                self.ssp_data.ssp_flux,
                t_obs_gyr,
                lgmet_scatter,
            )
            if self._uses_xray:
                p = {**p, "_sfr_current": sfr[-1]}
            return self._compositional.rest_sed(weights, ssp_flux_at_z, p)
        else:
            weights = sfr_on_ssp * self._csp_age_dt

        # Metallicity interpolation (single Z, non-evolving path).
        # effective_metallicity alpha correction is opt-in: only applied when
        # met_alpha_fe is explicitly a free parameter.
        _use_alpha_fe = (
            getattr(self.spec, "alpha_fe_evolving", False)
            or "met_alpha_fe" in self.spec.free_params
        )
        # For evolving-Z / chem-evol with a non-dsps_met_table integration method,
        # the compositional kernel expects a single ssp_flux_at_z (no per-age grid),
        # so we derive a representative single metallicity.
        if self._met_mode == "ramp":
            # Use the final (present-day) metallicity as the representative value.
            _log_z = p.get("log_z_abs_final", p.get("log_z_abs", -1.8477))
        elif self._met_mode == "chem_evol":
            from tengri.components.sfh.chemical_evolution import chem_evol_metallicity_on_ssp_grid

            _log_z_per_age = chem_evol_metallicity_on_ssp_grid(
                self.ssp_log_ages_yr,
                self.log_age_grid,
                sfr,
                yield_y=p.get("chem_yield", 0.03),
                eta_outflow=p.get("chem_eta_outflow", 0.0),
                f_gas_init=p.get("chem_f_gas_init", 0.9),
                return_frac=p.get("chem_return_frac", 0.4),
            )
            _log_z = jnp.sum(weights * _log_z_per_age) / jnp.maximum(jnp.sum(weights), 1e-30)
        else:
            _log_z = p.get("log_z_abs", p.get("log_z_abs_final", -1.8477))
        if _use_alpha_fe:
            alpha_fe = p.get("alpha_fe", 0.0)
            ssp_flux_at_z = interp_met_alpha_dispatch(self, _log_z, alpha_fe)
        else:
            ssp_flux_at_z = interp_metallicity(self, _log_z)

        # Enrich p with current SFR for X-ray model
        if self._uses_xray:
            p = {**p, "_sfr_current": sfr[-1]}

        return self._compositional.rest_sed(weights, ssp_flux_at_z, p)

    # ── Wavelength-axis chunking for XLA compilation reduction ────────

    def _compute_spectrum_chunked(self, sed_rest, wave_rest, wave_obs, z, dl_cm, wave_chunk_size):
        """Compute spectrum with wavelength-axis chunking via lax.map.

        Splits observed-frame wavelengths into chunks and evaluates each chunk
        independently, reducing per-chunk HLO size for XLA compilation.
        Numerically equivalent to unchunked evaluation (bitwise identical).

        Parameters
        ----------
        sed_rest : array, shape (n_wave,)
            Rest-frame SED [erg/s/Hz].
        wave_rest : array, shape (n_wave,)
            Rest-frame wavelength grid [Angstrom].
        wave_obs : array, shape (n_pix,)
            Observed-frame wavelength pixels [Angstrom].
        z : float
            Redshift.
        dl_cm : float
            Luminosity distance [cm].
        wave_chunk_size : int
            Number of pixels per chunk.

        Returns
        -------
        flux : array, shape (n_pix,)
            Observed flux [erg/s/cm²/Hz].
        """
        from tengri.observation.spectrum import compute_spectrum

        n_pix = wave_obs.shape[0]
        n_chunks = int(jnp.ceil(n_pix / wave_chunk_size))

        # Pad wave_obs to a multiple of wave_chunk_size
        padded_size = n_chunks * wave_chunk_size
        wave_obs_padded = jnp.pad(wave_obs, (0, padded_size - n_pix), mode="edge")

        # Reshape into chunks: (n_chunks, wave_chunk_size)
        wave_obs_chunks = wave_obs_padded.reshape(n_chunks, wave_chunk_size)

        # Map over chunks
        def compute_chunk(wave_chunk):
            return compute_spectrum(sed_rest, wave_rest, wave_chunk, z, dl_cm)

        flux_chunks = jax.lax.map(compute_chunk, wave_obs_chunks)

        # Reshape back to 1D and trim padding
        flux_padded = flux_chunks.reshape(padded_size)
        return flux_padded[:n_pix]

    def _observe_spectrum_from_rest_sed_chunked(
        self, rest_sed, wave_rest, wave_obs, z, dl_cm, wave_chunk_size
    ):
        """Observe spectrum with wavelength-axis chunking.

        Wraps observe_spectrum_from_rest_sed with lax.map over wavelength chunks.
        Numerically equivalent to unchunked evaluation.

        Parameters
        ----------
        rest_sed : array, shape (n_wave,)
            Rest-frame SED [erg/s/Hz].
        wave_rest : array, shape (n_wave,)
            Rest-frame wavelength grid [Angstrom].
        wave_obs : array, shape (n_pix,)
            Observed-frame wavelength pixels [Angstrom].
        z : float
            Redshift.
        dl_cm : float
            Luminosity distance [cm].
        wave_chunk_size : int
            Number of pixels per chunk.

        Returns
        -------
        flux : array, shape (n_pix,)
            Observed flux [erg/s/cm²/Hz].
        """
        from tengri.forward._kernels.compositional import observe_spectrum_from_rest_sed

        n_pix = wave_obs.shape[0]
        n_chunks = int(jnp.ceil(n_pix / wave_chunk_size))

        # Pad wave_obs to a multiple of wave_chunk_size
        padded_size = n_chunks * wave_chunk_size
        wave_obs_padded = jnp.pad(wave_obs, (0, padded_size - n_pix), mode="edge")

        # Reshape into chunks: (n_chunks, wave_chunk_size)
        wave_obs_chunks = wave_obs_padded.reshape(n_chunks, wave_chunk_size)

        # Map over chunks
        def observe_chunk(wave_chunk):
            return observe_spectrum_from_rest_sed(rest_sed, wave_rest, wave_chunk, z, dl_cm)

        flux_chunks = jax.lax.map(observe_chunk, wave_obs_chunks)

        # Reshape back to 1D and trim padding
        flux_padded = flux_chunks.reshape(padded_size)
        return flux_padded[:n_pix]

    # ── Factories and convenience ─────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        ssp,
        sfh=...,
        dust=...,
        nebular=...,
        agn=...,
        redshift=...,
        filters: list[str] | None = None,
        wave_obs=None,
        priors: dict | None = None,
        **model_kwargs,
    ) -> SEDModel:
        """Build a SEDModel from a grouped configuration dict.

        Reduces boilerplate for the common case: instead of constructing
        ``Parameters``, ``SSPData``, ``Observation``, and ``SEDModel`` separately,
        provide a single grouped config and receive a fully configured ``SEDModel``.

        Parameters
        ----------
        ssp : str or SSPData
            Path to SSP HDF5 file, or a pre-loaded ``SSPData`` instance.
        sfh : str
            SFH family name, e.g. ``"tsnorm"``, ``"dpl"``, ``"dpl+field"``.
        dust : str
            Dust attenuation law. ``"charlot_fall"`` (default), ``"calzetti"``, etc.
        nebular : str or None
            Nebular emission backend. ``"baked_in"``, ``"cloudy"``, ``"cue"``, or None.
        agn : str or None
            AGN model. None (disabled) or any AGN model name.
        redshift : float or str
            Fixed redshift (float), or ``"free"`` to add a free redshift parameter.
        filters : list of str, optional
            Filter names for photometry, e.g. ``["sdss_u", "sdss_g", "sdss_r"]``.
        wave_obs : array, optional
            Observed-frame wavelength array for spectroscopy.
        priors : dict, optional
            Parameter priors. Keys may be short names (``"log_peak_sfr"``),
            universal short names (``"logzsol"``), or full prefixed names.
            Short names are expanded automatically.
        **model_kwargs
            Forwarded to ``SEDModel.__init__()``.

        Returns
        -------
        SEDModel
            Fully initialized model ready for prediction or fitting.

        Notes
        -----
        Ellipsis (``...``) placeholders in optional parameters map to
        defaults from ``defaults.toml``. For example, ``dust=...`` uses
        the default dust attenuation law.

        Examples
        --------
        >>> model = tengri.SEDModel.from_config(
        ...     ssp="data/ssp.h5",
        ...     sfh="dense_basis",
        ...     filters=["sdss_u", "sdss_g", "sdss_r"],
        ...     redshift=0.1,
        ...     priors=dict(
        ...         log_total_mass=tengri.Uniform(8, 12),
        ...         log_sfr_inst=tengri.Uniform(-2, 3),
        ...         logzsol=tengri.Uniform(-2, 0.2),
        ...     ),
        ... )
        """
        from tengri.forward.convenience import build_model_from_config
        from tengri.parameters.defaults import UNSET

        # Map Ellipsis (signature placeholder) → UNSET so build_model_from_config
        # knows to fall back to defaults.toml instead of hard-coded values.
        def _r(v):
            """Convert ellipsis to UNSET sentinel for optional config parameters."""
            return UNSET if v is ... else v

        return build_model_from_config(
            cls,
            ssp,
            sfh=_r(sfh),
            dust=_r(dust),
            nebular=_r(nebular),
            agn=_r(agn),
            redshift=_r(redshift),
            filters=filters,
            wave_obs=wave_obs,
            priors=priors,
            **model_kwargs,
        )

    def prior_predictive(self, n: int = 500, seed: int = 42) -> PriorPredictive:
        """Sample from the prior and evaluate forward model on each draw.

        Parameters
        ----------
        n : int
            Number of prior samples. Default 500.
        seed : int
            Random seed. Default 42.

        Returns
        -------
        PriorPredictive
            Object containing flux, SFH, and parameter draws with model reference.

        Notes
        -----
        Useful for prior predictive checks: visualizing what the model
        predicts under the prior without conditioning on data.

        Examples
        --------
        >>> pp = model.prior_predictive(n=100, seed=42)
        >>> # Access photometry, SFH, and parameters from the prior
        """
        from tengri.forward.convenience import prior_predictive as _fn

        return _fn(self, n=n, seed=seed)

    def fit(
        self,
        data=None,
        noise=None,
        method: str = "vi",
        data_type: str | None = None,
        *,
        photometry: tuple | None = None,
        spectrum: tuple | None = None,
        init: str | None = None,
        **kwargs,
    ):
        """Fit observed data.  Convenience wrapper — no Fitter construction needed.

        Parameters
        ----------
        data : array, optional
            Observed flux array (photometry or spectroscopy). For joint fitting,
            leave as ``None`` and use ``photometry=`` / ``spectrum=`` instead.
        noise : array, optional
            1-sigma uncertainties matching ``data``.
        method : str
            Inference method. Default ``"vi"`` (geoVI variational inference).
            Any canonical name accepted by ``Fitter.run()`` works here:
            ``"vi"``, ``"vi_linear"``, ``"mcmc"``, ``"mcmc_raytrace"``,
            ``"mcmc_nuts"``, ``"map"``, ``"laplace"``, ``"auto"``, etc.
        data_type : str or None
            ``"photometry"``, ``"spectroscopy"``, or ``"joint"``.
            When ``None`` (default), inferred from the model's ``observation``
            or from whether ``photometry=`` / ``spectrum=`` kwargs are used.
        photometry : tuple of (flux, noise), optional
            Photometric data for joint fitting. Pass alongside ``spectrum=``.
        spectrum : tuple of (flux, noise), optional
            Spectroscopic data for joint fitting. Pass alongside ``photometry=``.
        init : str or None
            Initialization strategy. ``"map"`` runs MAP optimization first, then
            uses the result to warm-start the requested method. ``None`` (default)
            uses the method's own default initialization.
        **kwargs
            Forwarded to ``Fitter.run()``.

        Returns
        -------
        Posterior
            Inference results.  ``._fitter`` is set so ``.refine()`` works.
            After this call, ``self.fitter_`` holds the ``Fitter`` instance.

        Notes
        -----
        Convenience wrapper around :class:`Fitter`. For advanced usage
        (custom loss, multiple refinement steps), use ``Fitter`` directly.

        Examples
        --------
        >>> result = model.fit(flux_obs, noise)
        >>> result = model.fit(flux_obs, noise, method="mcmc")
        >>> result = model.fit(photometry=(flux_p, noise_p), spectrum=(flux_s, noise_s))
        >>> result = model.fit(flux_obs, noise, init="map")
        >>> result = model.fit(flux_obs, noise).refine("mcmc_raytrace")
        """
        from tengri.forward.convenience import fit_model

        return fit_model(
            self,
            data=data,
            noise=noise,
            method=method,
            data_type=data_type,
            photometry=photometry,
            spectrum=spectrum,
            init=init,
            **kwargs,
        )

    def fit_batch(
        self,
        catalog,
        flux_cols: list[str],
        err_cols: list[str],
        redshift_col: str | None = None,
        method: str = "vi",
        n_workers: int = 1,
        verbose: bool = True,
        output_dir: str | None = None,
        id_col: str | None = None,
        **kwargs,
    ) -> list:
        """Fit a batch of galaxies from a catalog (DataFrame, Table, or list of dicts).

        Parameters
        ----------
        catalog : DataFrame, Table, or list of dict
            Input catalog.
        flux_cols : list of str
            Column names for per-band flux values.
        err_cols : list of str
            Column names for per-band 1-sigma uncertainties.
        redshift_col : str or None
            If provided, use this column as per-row redshift.
        method : str
            Inference method. Default ``"vi"``.
        n_workers : int
            Currently ignored (reserved for multiprocessing). Default 1.
        verbose : bool
            Print per-galaxy progress. Default True.
        output_dir : str or None
            If provided, save each Posterior to ``{output_dir}/{id}.h5``.
        id_col : str or None
            Column name for galaxy identifiers in checkpoint filenames.
        **kwargs
            Forwarded to Fitter.run().

        Returns
        -------
        list of Posterior
            One result per galaxy in catalog.

        Notes
        -----
        Sequential fitting (no parallelization yet). For 1000+ galaxies,
        consider using :meth:`fit` in a loop with a multiprocessing pool.

        Examples
        --------
        >>> import pandas as pd
        >>> cat = pd.read_csv("catalog.csv")
        >>> results = model.fit_batch(
        ...     cat,
        ...     flux_cols=["f_u", "f_g", "f_r", "f_i", "f_z"],
        ...     err_cols=["e_u", "e_g", "e_r", "e_i", "e_z"],
        ...     redshift_col="z",
        ... )
        """
        from tengri.forward.convenience import fit_batch as _fn

        return _fn(
            self,
            catalog,
            flux_cols,
            err_cols,
            redshift_col=redshift_col,
            method=method,
            n_workers=n_workers,
            verbose=verbose,
            output_dir=output_dir,
            id_col=id_col,
            **kwargs,
        )

    def fit_population(
        self,
        observations_list: list,
        method: str = "vi",
        population_prior: dict | None = None,
        **kwargs,
    ):
        """Fit a population of galaxies with shared PSD hyperparameters.

        Parameters
        ----------
        observations_list : list
            Each element is a (flux, noise) tuple or dict with flux_obs/noise keys.
        method : str
            Hierarchical inference method. Default ``"vi"``.
        population_prior : dict or None
            Hyperpriors on shared PSD parameters.
        **kwargs
            Forwarded to PopulationFitter.run().

        Returns
        -------
        PopulationPosterior
            Hierarchical inference results with population-level and per-galaxy posteriors.

        Notes
        -----
        Enables population-level constraints on shared PSD hyperparameters
        (e.g., shared burst timescale across a sample). All galaxies must
        use the same model configuration.

        Examples
        --------
        >>> obs_list = [(flux1, noise1), (flux2, noise2), ...]
        >>> result = model.fit_population(obs_list, method="vi")
        """
        from tengri.forward.convenience import fit_population as _fn

        return _fn(
            self,
            observations_list,
            method=method,
            population_prior=population_prior,
            **kwargs,
        )

    def mock(self, params, snr=20.0, key=None):
        """Generate mock photometric observation with noise.

        Parameters
        ----------
        params : dict
            Parameter values.
        snr : float
            Signal-to-noise ratio. Default 20.0.
        key : PRNGKey, optional
            Random key for noise. If None, returns noiseless.

        Returns
        -------
        MockData
            Mock photometric observation.

        Notes
        -----
        Requires model to have filters configured (``filters=`` or
        ``observation=`` in constructor).

        Examples
        --------
        >>> key = jax.random.PRNGKey(0)
        >>> mock = model.mock(params, snr=15.0, key=key)
        >>> print(mock.flux.shape)  # (n_filters,)
        """
        from tengri.forward.convenience import mock as _fn

        return _fn(self, params, snr=snr, key=key)

    def mock_spectrum(self, params, wave_obs, snr=30.0, key=None):
        """Generate mock spectroscopic observation with noise.

        Parameters
        ----------
        params : dict
            Parameter values.
        wave_obs : array
            Observed wavelength grid [Angstrom].
        snr : float
            Signal-to-noise ratio per pixel. Default 30.0.
        key : PRNGKey, optional
            Random key for noise. If None, returns noiseless.

        Returns
        -------
        MockData
            Mock spectroscopic observation.

        Notes
        -----
        Noise is drawn from Gaussian distribution with standard deviation = flux/snr.

        Examples
        --------
        >>> wave_obs = np.linspace(4000, 5500, 1000)
        >>> mock = model.mock_spectrum(params, wave_obs, snr=10.0, key=key)
        >>> print(mock.flux.shape)  # (1000,)
        """
        from tengri.forward.convenience import mock_spectrum as _fn

        return _fn(self, params, wave_obs, snr=snr, key=key)

    def mock_batch(self, params_batch, snr=20.0, key=None):
        """Generate batch of mock photometric observations.

        Parameters
        ----------
        params_batch : dict of arrays
            Each value has leading batch dimension.
        snr : float
            Signal-to-noise ratio. Default 20.0.
        key : PRNGKey, optional
            Random key for noise. If None, returns noiseless.

        Returns
        -------
        MockData
            Mock observations with shape (N, n_filters).

        Notes
        -----
        Uses :func:`jax.vmap` over :meth:`mock` for vectorized generation.

        Examples
        --------
        >>> params_batch = {
        ...     k: jnp.tile(v[None], (1000,) + (1,) * (len(v.shape)))
        ...     for k, v in posterior.samples.items()
        ... }
        >>> mocks = model.mock_batch(params_batch, snr=15.0, key=key)
        """
        from tengri.forward.convenience import mock_batch as _fn

        return _fn(self, params_batch, snr=snr, key=key)

    def plot_sfh_posterior(
        self, posterior, true_params=None, ax=None, n_draws=50, color="C0", label="Posterior"
    ):
        """Plot posterior SFH with percentile fill and sample lines.

        Parameters
        ----------
        posterior : Posterior
            Inference results with samples (if available) or params.
        true_params : dict, optional
            True parameter values (if known) to overlay on plot.
        ax : matplotlib.axes.Axes, optional
            Axes object to plot on. If None, creates new figure.
        n_draws : int
            Number of posterior samples to show as thin lines. Default 50.
        color : str
            Color for posterior lines. Default "C0" (first color in style).
        label : str
            Label for posterior. Default "Posterior".

        Returns
        -------
        ax : matplotlib.axes.Axes
            The matplotlib Axes object with the plot.

        Notes
        -----
        Shows 16th and 84th percentiles as filled region, with individual
        sample curves in light color. If ``true_params`` provided, shows
        truth in black with dashed line for smooth SFH (parametric part).

        Examples
        --------
        >>> result = model.fit(flux, noise)
        >>> ax = model.plot_sfh_posterior(result)
        >>> ax.set_yscale("log")
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 5))

        if posterior.samples is None:
            sfh = self.predict_sfh(posterior.params)
            ax.plot(sfh["t_gyr"], sfh["sfr_mean"], color=color, lw=2, label=label)
        else:
            n_total = len(next(iter(posterior.samples.values())))
            sfh_draws = []
            for i in range(n_total):
                s_i = {k: posterior.samples[k][i] for k in posterior.samples}
                sfh_i = self.predict_sfh(s_i)
                key = "sfr_full" if self.spec.stochastic else "sfr_mean"
                sfh_draws.append(sfh_i[key])

            import numpy as np

            sfh_arr = np.array(sfh_draws)
            t_gyr = np.array(self.predict_sfh(posterior.params)["t_gyr"])

            lo = np.percentile(sfh_arr, 16, axis=0)
            hi = np.percentile(sfh_arr, 84, axis=0)
            ax.fill_between(t_gyr, lo, hi, color=color, alpha=0.2)

            n_show = min(n_draws, n_total)
            indices = np.linspace(0, n_total - 1, n_show, dtype=int)
            for idx in indices:
                ax.plot(t_gyr, sfh_arr[idx], color=color, alpha=0.1, lw=0.4)

            sfh_mean = self.predict_sfh(posterior.params)
            key = "sfr_full" if self.spec.stochastic else "sfr_mean"
            ax.plot(t_gyr, sfh_mean[key], color=color, lw=2, label=label)

        if true_params is not None:
            sfh_true = self.predict_sfh(true_params)
            key = "sfr_full" if self.spec.stochastic else "sfr_mean"
            ax.plot(sfh_true["t_gyr"], sfh_true[key], "k-", lw=2.5, label="Truth", zorder=10)
            if self.spec.stochastic:
                ax.plot(sfh_true["t_gyr"], sfh_true["sfr_mean"], "k--", lw=1, alpha=0.3)

        ax.set_xlabel("Lookback time (Gyr)")
        ax.set_ylabel(r"SFR (M$_{\odot}$/yr)")
        ax.set_xlim(0, 13.5)
        ax.legend(fontsize=9)
        return ax

    # ── Utilities ─────────────────────────────────────────────────────

    @property
    def wavelengths(self):
        """Rest-frame wavelength grid (Angstrom).

        Returns the SSP grid by default, or the extended panchromatic grid
        when radio or X-ray emission is enabled.

        Returns
        -------
        ndarray, shape (n_wave,)
            Rest-frame wavelength grid [Angstrom].

        Notes
        -----
        This is the grid used by :meth:`predict_rest_sed` by default when
        no custom ``wave=`` is passed. Updated when radio/X-ray components
        are added to the model.

        Examples
        --------
        >>> print(model.wavelengths[0], model.wavelengths[-1])
        >>> # Default SSP range, e.g. 91.2 to 160000 Å
        """
        return self._rest_wavelength

    @staticmethod
    def _t_universe_gyr(z):
        """Age of the universe at redshift z in Gyr.

        Thin wrapper around age_at_z.

        Parameters
        ----------
        z : float or jnp.ndarray
            Redshift.

        Returns
        -------
        float
            Age of universe in Gyr.
        """
        return age_at_z(z)

    def _interp_metallicity(self, log_z):
        """Dispatch metallicity interpolation (single Z value)."""
        return interp_metallicity(self, log_z)

    def _interp_metallicity_evolving(self, log_z_per_age):
        """Dispatch evolving metallicity interpolation (per-age Z)."""
        return interp_metallicity_evolving(self, log_z_per_age)

    def precompute_spectroscopy(self, wave_obs):
        """Pre-interpolate SSP templates to observed wavelength grid.

        Call this before spectroscopic fitting to get a ~20x speedup.
        Requires fixed redshift.

        Parameters
        ----------
        wave_obs : array, shape (n_pix,)
            Observed wavelength grid (Angstrom).

        Returns
        -------
        self
            For chaining: ``model.precompute_spectroscopy(wave_obs)``

        Notes
        -----
        Caches precomputed SSP spectra at the fixed redshift,
        enabling ~10-20× speedup for repeated spectrum predictions.
        Requires fixed redshift (raises ValueError otherwise).

        Examples
        --------
        >>> wave_obs = np.linspace(3500, 7000, 2000)
        >>> model.precompute_spectroscopy(wave_obs)
        >>> flux = model.predict_spectrum(params)
        """
        if self._z_fixed is None:
            raise ValueError("Spectroscopy precomputation requires fixed redshift")
        spec_precomp = precompute_spectroscopy(
            self.ssp_data,
            jnp.asarray(wave_obs),
            self._z_fixed,
            self._dl_cm_fixed,
        )
        self._precomputed = dataclasses.replace(self._precomputed, spectroscopy=spec_precomp)
        self._wave_obs = jnp.asarray(wave_obs)
        self._state = dataclasses.replace(
            self._state, precomputed=self._precomputed, wave_obs=self._wave_obs
        )

        # Invalidate cached kernels and fitter loss-fn caches so any attached
        # Fitter picks up the new precomputed spectroscopy path on next run().
        # Compiled artefacts can always be regenerated, so we just drop them.
        from tengri.inference._model_cache import clear_model_cache

        self._invalidate_kernels()
        clear_model_cache(self)

        # Rebuild compositional spectrum kernel (full-resolution end-to-end JIT)
        if self._compositional.rest_sed is not None:
            with contextlib.suppress(Exception):
                _raw = build_fused_tier2_spectrum(self._state, self)
                self._compositional.spectrum = jax.jit(_raw) if _raw else None

        # Rebuild hybrid spectrum kernel (precomputed SSP + exact non-stellar)
        if self._compositional.rest_sed is not None:
            with contextlib.suppress(Exception):
                _raw = build_hybrid_spectrum(self._state, self)
                self._hybrid.spectrum = jax.jit(_raw) if _raw else None

        return self

    def precompute_ztable(self, z_grid=None, z_min=0.001, z_max=3.0, n_z=100):
        """Pre-compute SSP photometry on a redshift grid for free-z fitting.

        At inference time, the precomputed table is interpolated to the
        current z — same speedup as fixed-z precomputation, but z is free.
        Follows the DSPS ``precompute_ssp_obsmags_on_z_table`` approach.

        Parameters
        ----------
        z_grid : array, optional
            Custom redshift grid. If None, uses linspace(z_min, z_max, n_z).
        z_min : float
            Minimum redshift (default 0.001).
        z_max : float
            Maximum redshift (default 3.0).
        n_z : int
            Number of grid points (default 100). More points = more accurate
            interpolation. 100 gives <0.01% interpolation error for smooth
            filter transmission curves.

        Returns
        -------
        self
            For chaining: ``model.precompute_ztable().predict_photometry(params)``

        Notes
        -----
        Enables fast photometry prediction with free redshift (no fixed z).
        Interpolates precomputed SSP×filter grid to current z at inference time,
        achieving similar speedup as fixed-z precomputation.

        Examples
        --------
        >>> model.precompute_ztable(z_min=0.01, z_max=4.0, n_z=200)
        >>> flux = model.predict_photometry(params)  # z now free
        """
        if self.filter_waves is None:
            raise ValueError("Z-table precomputation requires filters to be set")
        ztable = precompute_photometry_ztable(
            self.ssp_data,
            self.filter_waves,
            self.filter_trans,
            z_grid=z_grid,
            z_min=z_min,
            z_max=z_max,
            n_z=n_z,
            apply_igm=self._uses_igm and self._approx.get("igm", True),
        )
        self._precomputed = dataclasses.replace(self._precomputed, photometry_ztable=ztable)
        self._state = dataclasses.replace(self._state, precomputed=self._precomputed)

        # Invalidate fitter loss-fn caches so any attached Fitter uses the
        # new ztable interpolation path on next run(). Compiled artefacts can
        # always be regenerated, so we just drop them.
        from tengri.inference._model_cache import clear_model_cache

        clear_model_cache(self)

        # Build hybrid z-table kernel if using hybrid mode
        if self._hybrid.photometry is not None or any(
            (
                getattr(self._nebular_backend, "has_free_params", False),
                getattr(self._shock_backend, "has_free_params", False),
                self._has_dust_ir_full,
                self._has_agn_full,
                self._has_radio,
                self._has_xray,
            )
        ):
            with contextlib.suppress(Exception):
                _raw = build_hybrid_photometry_ztable(self._state, self)
                self._hybrid.photometry = jax.jit(_raw) if _raw else None

        return self

    def _method_recommendation(self) -> tuple[str, str]:
        """Return (method_name, reason) for the recommended inference method."""
        from tengri.config.display import method_recommendation

        return method_recommendation(self)

    def tree(self) -> str:
        """Return a human-readable physics tree showing the model hierarchy.

        Shows the active sub-models at each physical layer (SFH, SPS, Dust,
        Nebular, AGN, Observation), the free parameters at each layer, and
        the recommended inference method.

        Returns
        -------
        str
            Multi-line formatted tree string.

        Notes
        -----
        Useful for inspecting model configuration before fitting or inference.

        Examples
        --------
        >>> print(model.tree())
        Model  [D=7, stochastic=False]
        ...
        """
        from tengri.config.display import tree as _tree

        return _tree(self)

    def recommend_method(self) -> str:
        """Return the recommended inference method string for this model.

        Returns
        -------
        str
            Canonical method name for ``Fitter.run()`` or ``model.fit()``.

        Notes
        -----
        Based on model dimensionality, complexity, and available precomputation.
        Use as input to ``model.fit(method=model.recommend_method())``.

        Examples
        --------
        >>> method = model.recommend_method()
        >>> result = model.fit(flux, noise, method=method)
        """
        method, _ = self._method_recommendation()
        return method

    def summary(self) -> str:
        """Return a human-readable summary of the model configuration.

        Returns
        -------
        str
            Formatted summary showing SSP grid, filters, precomputation,
            fused kernel status, and enabled components.

        Notes
        -----
        Similar to :meth:`tree` but focuses on computational configuration
        and precomputation status rather than physics parameters.

        Examples
        --------
        >>> print(model.summary())
        """
        from tengri.config.display import summary as _summary

        return _summary(self)


# Backward-compatibility alias
