"""High-level Model class wrapping the tengri forward model.

Model provides a clean API for:
- Forward predictions (SED, photometry, spectrum, SFH, derived quantities)
- Mock galaxy generation (single and batch)
- Convenience fitting (delegates to Fitter)

The Model translates between the user-facing parameter names and the
internal names used by the low-level functions, handling unit conversions
automatically. SFH computation is dispatched through the registry-driven
composed function, eliminating separate stochastic/parametric code paths.

Usage::

    from tengri import Model, ParamSpec, Uniform, load_ssp_data, load_filter_set

    ssp = load_ssp_data("data/ssp.h5")
    filters = load_filter_set(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
    spec = ParamSpec(
        sfh_tsnorm_log_peak_sfr=Uniform(-1, 2),
        sfh_tsnorm_peak_lbt_gyr=Uniform(1, 12),
        sfh_tsnorm_width_gyr=Uniform(0.5, 5),
        sfh_tsnorm_skew=Uniform(-1, 1),
        sfh_tsnorm_trunc=Uniform(1, 10),
        sfh_field_psd_sigma=Uniform(0.01, 1.0),
        sfh_field_psd_tau_myr=Uniform(10, 500),
        redshift=0.1,
    )
    model = Model(spec, ssp, filters=filters)
    params = spec.sample(jax.random.PRNGKey(0))
    photometry = model.predict_photometry(params)
"""

from __future__ import annotations

import contextlib
import dataclasses
import warnings
from typing import ClassVar, NamedTuple

import jax.numpy as jnp

from tengri.core.fused_kernels import (
    build_exact_sed,
    build_fused_photometry,
    build_fused_photometry_ztable,
    build_fused_rest_sed,
    build_fused_spectrum,
    build_fused_tier2_photometry,
    build_fused_tier2_spectrum,
    is_fused_compatible,
    is_tier2_compatible,
    observe_photometry_from_rest_sed,
    observe_spectrum_from_rest_sed,
)
from tengri.core.param_translate import (
    _EVOLVING_ALPHA_PARAM_MAP,
    _EVOLVING_MET_PARAM_MAP,
    LOG10_ZSUN,
    _build_param_map,
    get_internal_params,
)
from tengri.core.sed_pipeline import (
    compute_sed_components,
    get_agn_kwargs,
    get_dust_kwargs,
    interp_metallicity,
    interp_metallicity_evolving,
)
from tengri.models.dust.attenuation import precompute_dust_age_weights
from tengri.models.observation.photometry import ab_mag_from_flux, compute_flux_density
from tengri.models.observation.spectroscopy import apply_lsf, compute_spectrum
from tengri.models.sfh.registry import compute_field_gp, resolve_sfh
from tengri.models.sps.dsps_wrapper import csp_age_dt
from tengri.models.sps.precompute import (
    precompute_photometry,
    precompute_photometry_ztable,
    precompute_spectroscopy,
)
from tengri.utils.cosmology import age_at_z, luminosity_distance
from tengri.utils.grid import (
    grid_spacing,
    interpolate_to_linear_time,
    log_age_to_age_yr,
    make_log_age_grid,
)

# ---------------------------------------------------------------------------
# MockData container
# ---------------------------------------------------------------------------


class MockData(NamedTuple):
    """Container for mock galaxy observations."""

    flux_true: jnp.ndarray  # noiseless photometry (erg/s/cm²/Hz)
    flux_obs: jnp.ndarray  # noisy photometry
    noise: jnp.ndarray  # 1-sigma uncertainties
    params: dict  # input parameters

    def plot(self, filter_names=None, ax=None):
        """Plot mock photometry with errorbars.

        Parameters
        ----------
        filter_names : list of str, optional
            Filter labels for the x-axis. Falls back to integer indices.
        ax : matplotlib Axes, optional
            Axes to plot on. Creates new figure if None.

        Returns
        -------
        fig : matplotlib Figure
        """
        import matplotlib.pyplot as plt
        import numpy as np

        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 4))
        else:
            fig = ax.get_figure()

        n = len(self.flux_true)
        x = np.arange(n)
        labels = filter_names if filter_names is not None else [str(i) for i in x]

        ax.errorbar(
            x,
            np.array(self.flux_obs),
            yerr=np.array(self.noise),
            fmt="o",
            color="C0",
            label="observed (noisy)",
            capsize=3,
            zorder=3,
        )
        ax.plot(x, np.array(self.flux_true), "s--", color="C1", label="true (noiseless)", zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel(r"$F_\nu$ [erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$]", fontsize=11)
        ax.legend(fontsize=10, frameon=False)
        ax.set_title("Mock Photometry", fontsize=11)
        fig.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# PriorPredictive container
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class PriorPredictive:
    """Results of a prior predictive check.

    Attributes
    ----------
    flux : jnp.ndarray or None
        Predicted photometry draws, shape ``(n, n_filters)``.
        None if the model has no filters.
    sfh : jnp.ndarray
        SFH draws, shape ``(n, n_grid)``.
    params : dict
        Drawn parameter samples, each of shape ``(n,)``.
    _model : object
        Back-reference to the parent model.
    """

    flux: jnp.ndarray | None
    sfh: jnp.ndarray
    params: dict
    _model: object = dataclasses.field(default=None, repr=False)

    def check_finite(self) -> dict:
        """Check for NaN/Inf in flux draws.

        Returns
        -------
        dict
            ``{"n_nan": int, "n_inf": int, "frac_bad": float, "ok": bool}``
        """
        import warnings

        import numpy as np

        if self.flux is None:
            return {"n_nan": 0, "n_inf": 0, "frac_bad": 0.0, "ok": True}

        flux_np = np.array(self.flux)
        n_nan = int(np.sum(np.isnan(flux_np)))
        n_inf = int(np.sum(np.isinf(flux_np)))
        total = flux_np.size
        frac_bad = (n_nan + n_inf) / max(total, 1)
        if n_nan + n_inf > 0:
            warnings.warn(
                f"prior_predictive: {n_nan} NaN and {n_inf} Inf values in flux draws "
                f"({frac_bad:.1%} of total). Check priors for extreme parameter combinations.",
                UserWarning,
                stacklevel=2,
            )
        return {"n_nan": n_nan, "n_inf": n_inf, "frac_bad": frac_bad, "ok": (n_nan + n_inf == 0)}


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------


class Model:
    """Differentiable forward model with clean parameter API.

    The SFH is computed via a registry-driven composed function that
    handles additive smooth models, burst mixture, and GP field
    modulation in a single call.

    Parameters
    ----------
    spec : ParamSpec
        Parameter specification (defines free/fixed params and priors).
    ssp_data : SSPData
        Pre-loaded SSP templates.
    filters : list or tuple, optional
        Filter transmission curves. Accepts either:
        - 3-tuple from load_filter_set(): (filter_waves, filter_trans, filter_curves)
        - List of FilterCurve namedtuples
    precompute : bool or dict, optional
        Precomputation settings. True (default) = automatic.
    forward_dtype : str or jnp.dtype, optional
        Dtype for forward model computation. ``"float32"`` halves memory
        and gives ~1.5x speedup with <0.1% accuracy loss. Default
        ``"float64"`` preserves full precision.

        Affects **both** fused and exact paths:

        - **Fused path** (photometry with precomputation): all captured
          arrays (SSP grid, dust weights, effective wavelengths) are cast
          to ``forward_dtype`` at kernel build time. Outputs are always
          cast back to float64 for cosmological distance scaling.
        - **Exact path** (spectroscopy, legacy AGN): the three largest
          intermediates — metallicity-interpolated SSP ``(n_age, n_wave)``,
          dust attenuation ``(n_age, n_wave)``, and dust age weights
          ``(n_age,)`` — are computed in ``forward_dtype``. This halves
          the 4.5 MB memory traffic that dominates exact-path dust cost.

        Cosmological distances always use float64 (float32 overflows
        at z > 0.01).
    approx : dict, optional
        Control which approximations the fused kernel uses. Each key
        enables/disables a specific approximation. When a component's
        approximation is disabled, the model falls back to the exact
        path for that component.

        Keys and defaults::

            {
                "dust_attenuation": True,  # dust at filter eff. wavelengths
                "dust_emission": True,  # MBB at filter eff. wavelengths
                "igm": True,  # IGM at filter eff. wavelengths
            }

        Approximation accuracy (Zacharegkas+2025):
        - dust_attenuation: <3% for most laws, ~36% for SMC
        - dust_emission: negligible for optical (MBB peaks at >50μm)
        - igm: exact for fixed z (precomputed once)

        Set ``approx=False`` to disable all approximations (forces exact
        path for everything). Set ``approx=True`` (default) to use all.
    """

    # Default approximation settings (immutable — used as template only)
    _DEFAULT_APPROX: ClassVar[dict] = {
        "dust_attenuation": True,
        "dust_emission": True,
        "igm": True,
    }

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
    ):
        # --- Observation / filters resolution ---
        if filters is not None and observation is not None:
            raise ValueError(
                "Cannot specify both filters= and observation=. "
                "Use observation=Observation(photometry=...) instead."
            )

        if observation is not None or filters is not None:
            from tengri.models.observation.observation import Observation

        if observation is not None:
            if not isinstance(observation, Observation):
                raise TypeError(
                    f"observation must be an Observation instance, got {type(observation)}"
                )
            obs_params = observation.get_all_params()
            if obs_params:
                spec = spec.with_params(**obs_params)

        elif filters is not None:
            from tengri.models.observation.photometry_config import Photometry

            observation = Observation(photometry=Photometry.from_filter_set(filters))

        self.observation = observation
        self.spec = spec
        self.ssp_data = ssp_data
        self._forward_dtype = jnp.dtype(forward_dtype)

        # Initialize metallicity interpolation settings early so attribute
        # lookups never fail before the full-init code runs.
        self._met_interp = getattr(spec, "met_interp", "linear")
        self._lgmet_scatter = float(getattr(spec, "lgmet_scatter", 0.1))

        # Parse approximation settings
        if approx is None or approx is True:
            self._approx = dict(self._DEFAULT_APPROX)
        elif approx is False:
            self._approx = {k: False for k in self._DEFAULT_APPROX}
        else:
            self._approx = {**self._DEFAULT_APPROX, **approx}

        # Handle filter input formats
        self.filter_waves = None
        self.filter_trans = None
        if observation is not None and observation.can_do_photometry:
            self.filter_waves = list(observation.photometry.filter_waves)
            self.filter_trans = list(observation.photometry.filter_trans)
        elif filters is not None:
            if isinstance(filters, tuple) and len(filters) == 3:
                self.filter_waves = filters[0]
                self.filter_trans = filters[1]
            elif isinstance(filters, (list, tuple)):
                self.filter_waves = [f.wave for f in filters]
                self.filter_trans = [f.trans for f in filters]
            else:
                raise TypeError(
                    f"filters must be a list of FilterCurve or output of "
                    f"load_filter_set(), got {type(filters)}"
                )

        # SSP grid info
        self.ssp_log_ages_yr = ssp_data.ssp_lg_age_gyr + 9.0
        self.ssp_ages_yr = 10.0**self.ssp_log_ages_yr

        # CSP integration method (precompute bin widths once at model init)
        _valid_csp = ("trapz", "log_trapz", "log_interp", "dsps_native", "dsps_met_table")
        if csp_integration not in _valid_csp:
            raise ValueError(
                f"csp_integration must be one of {_valid_csp}, got {csp_integration!r}"
            )
        self._csp_integration = csp_integration
        if csp_integration == "log_interp":
            from tengri.models.sps.dsps_wrapper import csp_log_interp_matrix

            self._csp_matrix = jnp.array(csp_log_interp_matrix(self.ssp_ages_yr))
            self._csp_age_dt = None
        elif csp_integration in ("dsps_native", "dsps_met_table"):
            # No precomputed bin widths; DSPS handles integration internally.
            self._csp_age_dt = None
            self._csp_matrix = None
        else:
            self._csp_age_dt = csp_age_dt(self.ssp_ages_yr, csp_integration)
            self._csp_matrix = None

        # Log-age grid for SFH computation
        n_grid = spec.n_grid if spec.stochastic else 256
        self.log_age_grid = make_log_age_grid(n_grid)
        self.d_log_age = grid_spacing(self.log_age_grid)
        self.age_yr = log_age_to_age_yr(self.log_age_grid)
        self._n_grid = n_grid

        # Resolve SFH from registry
        sfh_fn, _sfh_params, sfh_param_map, sfh_settings = resolve_sfh(spec.mean_sfh_type)
        self._sfh_fn = sfh_fn
        self._sfh_internal_names = {v[0] for v in sfh_param_map.values()}
        self._sfh_settings = sfh_settings
        self._param_map = _build_param_map(
            spec.mean_sfh_type,
            dust_model=getattr(spec, "dust_model", "two_component"),
        )

        # Field settings
        self._has_field = spec.stochastic
        self._field_model = sfh_settings.get("sfh_field_model", "drw")

        # Dust model: "two_component" (Charlot & Fall) or "single_component" (screen)
        self._dust_model = getattr(spec, "dust_model", "two_component")
        # Dust approx: "fast" (hard threshold, 2-CSP) or "exact" (smooth sigmoid)
        self._dust_approx = getattr(spec, "dust_approx", "fast")

        # Dust law settings
        self._dust_law_bc = spec.dust_law_bc
        self._dust_law_diff = spec.dust_law_diff
        # Cache resolved dust law functions (avoid dict lookup per forward call)
        from tengri.models.dust.attenuation import get_dust_law

        self._dust_law_bc_fn = get_dust_law(self._dust_law_bc)
        if self._dust_model == "single_component":
            self._dust_law_diff_fn = self._dust_law_bc_fn  # not used, keep consistent
        else:
            self._dust_law_diff_fn = get_dust_law(self._dust_law_diff)

        # IGM absorption (Inoue+2014)
        self._apply_igm = spec.apply_igm

        # Dust emission model (None = disabled)
        # "dl07_tabulated" is a legacy alias — map to "draine_li2007" which
        # now auto-loads tabulated templates on first call.
        self._dust_emission_model = getattr(spec, "dust_emission", None)
        if self._dust_emission_model == "dl07_tabulated":
            self._dust_emission_model = "draine_li2007"
        if self._dust_emission_model:
            for p in [
                "dust_T",
                "dust_beta_ir",
                "dust_alpha_mir",
                "dust_alpha_dale",
                "dust_umin",
                "dust_gamma_dl",
                "dust_qpah",
                "dust_eta_balance",
            ]:
                self._param_map[p] = (p, 1.0, 0.0)

        # Evolving metallicity
        self._evolving_metallicity = getattr(spec, "evolving_metallicity", False)
        if self._evolving_metallicity:
            # Replace met_logzsol mapping with the two evolving-Z params
            self._param_map.pop("met_logzsol", None)
            self._param_map.update(_EVOLVING_MET_PARAM_MAP)

        # Chemical evolution: Z(t) derived from SFH via gas-regulator model
        self._chem_evol_enabled = getattr(spec, "chem_evol", False)
        if self._chem_evol_enabled:
            # Remove met_logzsol (metallicity derived from SFH)
            self._param_map.pop("met_logzsol", None)
            # Add chem_evol params (identity mapping, no unit conversion)
            for p in [
                "chem_yield",
                "chem_eta_outflow",
                "chem_f_gas_init",
                "chem_return_frac",
            ]:
                self._param_map[p] = (p, 1.0, 0.0)

        # Evolving alpha-enhancement: replaces global met_alpha_fe
        self._alpha_fe_evolving = getattr(spec, "alpha_fe_evolving", False)
        if self._alpha_fe_evolving:
            self._param_map.pop("met_alpha_fe", None)
            self._param_map.update(_EVOLVING_ALPHA_PARAM_MAP)

        # Store spec reference for pipeline access
        self._spec = spec

        # AGN model (None = disabled)
        self._agn_model = getattr(spec, "agn_model", None)
        # Detect parametric AGN mode: agn_log_lbol is a free (non-Fixed)
        # parameter. Parametric mode uses agn_log_lbol directly, enabling
        # fused-kernel evaluation. Legacy mode uses agn_frac to derive
        # L_bol from the full SED integral, forcing the exact path.
        self._agn_parametric = False
        if self._agn_model:
            agn_dists = getattr(spec, "_distributions", {})
            agn_lbol_dist = agn_dists.get("agn_log_lbol")
            agn_frac_dist = agn_dists.get("agn_frac")
            # Parametric if agn_log_lbol is free, or if agn_frac is Fixed(0)
            # (default) and agn_log_lbol exists with any non-zero default.
            lbol_is_free = agn_lbol_dist is not None and not agn_lbol_dist.is_fixed
            frac_is_free = agn_frac_dist is not None and not agn_frac_dist.is_fixed
            self._agn_parametric = lbol_is_free and not frac_is_free
            for p in [
                "agn_frac",
                "agn_log_lbol",
                "agn_alpha",
                "agn_T_torus",
                "agn_tau_torus",
                "agn_torus_frac",
                "agn_log_mbh",
                "agn_log_ledd",
            ]:
                self._param_map[p] = (p, 1.0, 0.0)

        # Radio and X-ray
        self._radio_enabled = getattr(spec, "radio", False)
        if self._radio_enabled:
            for p in ["radio_q_ir", "radio_alpha_sf", "radio_loudness", "radio_alpha_agn"]:
                self._param_map[p] = (p, 1.0, 0.0)

        self._xray_enabled = getattr(spec, "xray", False)
        if self._xray_enabled:
            for p in ["xray_gamma_agn", "xray_alpha_ox"]:
                self._param_map[p] = (p, 1.0, 0.0)

        # Shock emission (MAPPINGS V)
        self._shock_enabled = getattr(spec, "shock", False)
        if self._shock_enabled:
            for p in ["shock_frac", "shock_velocity", "shock_log_density"]:
                self._param_map[p] = (p, 1.0, 0.0)

        # Nebular emission backend + params
        if spec.nebular_mode in ("cloudy", "cue"):
            self._param_map["neb_logU"] = ("neb_logU", 1.0, 0.0)
            self._param_map["neb_logZ_gas"] = ("neb_logZ_gas", 1.0, LOG10_ZSUN)
            self._param_map["neb_fesc"] = ("neb_fesc", 1.0, 0.0)
            self._param_map["neb_fesc_lya"] = ("neb_fesc_lya", 1.0, 0.0)

        self._nebular_backend = None
        if spec.nebular_mode == "cue":
            from tengri.models.nebular import CueBackend

            self._nebular_backend = CueBackend(spec.cue_weights_path, ssp_data=ssp_data)
        elif spec.nebular_mode == "cloudy":
            from tengri.models.nebular import CloudyGridBackend

            self._nebular_backend = CloudyGridBackend(spec.cloudy_grid_path, ssp_data)
        elif spec.nebular_mode == "ssp":
            from tengri.models.nebular import BakedInBackend

            self._nebular_backend = BakedInBackend()
        else:
            from tengri.models.nebular import BakedInBackend

            self._nebular_backend = BakedInBackend()

        # Velocity dispersion: only apply if sigma_v is in the spec
        self._has_sigma_v = spec.has_param("sigma_v") if hasattr(spec, "has_param") else False
        if not self._has_sigma_v:
            # Check if sigma_v is in the param names directly
            try:
                spec.get_distribution("sigma_v")
                self._has_sigma_v = True
            except KeyError:
                self._has_sigma_v = False

        # SSP library velocity resolution for LSF subtraction (km/s)
        # Observation config takes precedence over spec attributes
        if observation is not None and observation.can_do_spectroscopy:
            sc = observation.spectroscopy
            self._sigma_lib_kms = sc.sigma_lib_kms
            self._lsf_resolution = sc.resolution
            self._lsf_n_bins = sc.lsf_n_bins
        else:
            self._sigma_lib_kms = getattr(spec, "sigma_lib_kms", 0.0)
            self._lsf_resolution = getattr(spec, "lsf_resolution", None)
            self._lsf_n_bins = getattr(spec, "lsf_n_bins", 16)

        # Precompute luminosity distance if redshift is fixed
        redshift_dist = spec.get_distribution("redshift")
        if redshift_dist.is_fixed:
            self._dl_cm_fixed = luminosity_distance(redshift_dist.bounds[0])
            self._z_fixed = redshift_dist.bounds[0]
        else:
            self._dl_cm_fixed = None
            self._z_fixed = None

        # Photometry precomputation (Zacharegkas+2025 Section 3)
        self._precomp = None
        if precompute is True and self._z_fixed is not None and self.filter_waves is not None:
            self._precomp = precompute_photometry(
                ssp_data,
                self.filter_waves,
                self.filter_trans,
                self._z_fixed,
                self._dl_cm_fixed,
            )

        # Precompute dust age weights (sigmoid depends only on age grid)
        # Dust precomputation depends on model and approximation mode
        if self._dust_model == "single_component":
            self._dust_age_weights = None
        elif self._dust_approx == "exact":
            # Smooth sigmoid weights for exact two-component dust
            self._dust_age_weights = precompute_dust_age_weights(self.ssp_ages_yr)
        else:
            # Fast mode: age weights not used (two-CSP decomposition in fused kernel)
            self._dust_age_weights = None

        # Precompute IGM at effective wavelengths (for fused kernel)
        self._igm_at_eff = None
        if (
            self._apply_igm
            and self._approx["igm"]
            and self._precomp is not None
            and self._z_fixed is not None
        ):
            from tengri.models.igm import igm_transmission

            eff_obs = self._precomp.effective_wavelengths
            self._igm_at_eff = igm_transmission(eff_obs, self._z_fixed)

        # Build fused JIT kernels for fast photometry/spectroscopy
        self._fused_photometry = None
        if self._precomp is not None and is_fused_compatible(self):
            self._fused_photometry = build_fused_photometry(self)

        # JIT-compiled exact-path SED kernel (eliminates Python dispatch overhead)
        self._jit_exact_sed = build_exact_sed(self)

        # Tier 2: Compositional rest-frame SED kernel (all components, JIT'd)
        self._fused_rest_sed = None
        if is_tier2_compatible(self):
            try:
                self._fused_rest_sed = build_fused_rest_sed(self)
            except Exception as e:
                warnings.warn(
                    f"Compositional SED kernel (Tier 2) build failed: {e}",
                    UserWarning,
                    stacklevel=2,
                )

        # Fused Tier 2 end-to-end photometry (params → photometry in one JIT)
        self._fused_tier2_phot = None
        self._fused_tier2_spec = None
        if self._fused_rest_sed is not None and self.filter_waves is not None:
            with contextlib.suppress(Exception):
                self._fused_tier2_phot = build_fused_tier2_photometry(self)

        # Spectroscopy precomputation (same idea: pre-interpolate SSPs)
        self._spec_precomp = None

        # Auto-precompute spectroscopy from Observation config
        if (
            observation is not None
            and observation.can_do_spectroscopy
            and self._z_fixed is not None
            and precompute is not False
        ):
            self.precompute_spectroscopy(observation.spectroscopy.wave_obs)

        # Z-table precomputation (for free-redshift fitting)
        self._ztable = None
        self._fused_photometry_ztable = None

    # -------------------------------------------------------------------
    # Internal parameter translation
    # -------------------------------------------------------------------

    def _get_internal_params(self, params):
        """Translate public param dict to internal names with unit conversion.

        Thin wrapper around :func:`tengri._param_translate.get_internal_params`.
        """
        return get_internal_params(params, self._param_map, self.spec, self._has_field)

    @staticmethod
    def _t_universe_gyr(z):
        """Age of the universe at redshift z in Gyr.

        Thin wrapper around age_at_z (which returns years).

        Parameters
        ----------
        z : float or jnp.ndarray
            Redshift.

        Returns
        -------
        float
            Age of universe in Gyr.
        """
        return age_at_z(z) / 1e9

    def _interp_metallicity(self, log_z):
        """Dispatch metallicity interpolation (single Z value)."""
        return interp_metallicity(self, log_z)

    def _interp_metallicity_evolving(self, log_z_per_age):
        """Dispatch evolving metallicity interpolation (per-age Z)."""
        return interp_metallicity_evolving(self, log_z_per_age)

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
        if self._has_field and "xi" in p:
            gp_x, k0_half = compute_field_gp(
                xi=p["xi"],
                psd_sigma=p["psd_sigma"],
                psd_tau_yr=p["psd_tau_yr"],
                n_grid=self._n_grid,
                d_log_age=float(self.d_log_age),
                field_model=self._field_model,
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

        if self._has_field and "xi" in p:
            gp_x, k0_half = compute_field_gp(
                xi=p["xi"],
                psd_sigma=p["psd_sigma"],
                psd_tau_yr=p["psd_tau_yr"],
                n_grid=self._n_grid,
                d_log_age=float(self.d_log_age),
                field_model=self._field_model,
            )
            kw["gp_x"] = gp_x
            kw["k0_half"] = k0_half
            sfr_full = self._sfh_fn(self.age_yr, **kw)
        else:
            sfr_full = sfr_mean

        return sfr_mean, sfr_full

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

    # -------------------------------------------------------------------
    # Forward predictions
    # -------------------------------------------------------------------

    def _compute_sed_components(self, params, _sfr=None, _weights=None, need_intrinsic=False):
        """Compute all SED intermediates.

        Delegates to :func:`tengri._sed_pipeline.compute_sed_components`.
        """
        return compute_sed_components(self, params, _sfr, _weights, need_intrinsic)

    def predict_sed(self, params):
        """Compute rest-frame luminosity SED.

        Parameters
        ----------
        params : dict
            Parameter values (public names).

        Returns
        -------
        array, shape (n_wave,)
            Rest-frame SED in erg/s/Hz.
        """
        return self._compute_sed_components(params)["sed_total"]

    def predict(self, params):
        """Create a lazy prediction object for derived physical quantities.

        Returns a :class:`~tengri.prediction.Prediction` object whose
        properties are computed on first access and cached. This is the
        recommended API for exploring derived quantities from a single
        galaxy.

        For batch computation over many parameter sets (posterior
        chains, mock catalogs), use the JIT-compatible group methods
        :meth:`predict_sfh_quantities`, :meth:`predict_sed_quantities`,
        or :meth:`predict_line_luminosities` with ``jax.vmap`` instead.

        Parameters
        ----------
        params : dict
            Parameter values (public names).

        Returns
        -------
        Prediction
            Lazy prediction object with ``.sfh``, ``.sed``, ``.lines``,
            ``.radio``, ``.xray``, and ``.ionizing`` property groups.

        Examples
        --------
        **Single-galaxy exploration (lazy, on-demand):**

        >>> pred = model.predict(params)
        >>> pred.sfh.stellar_mass  # triggers SFH computation
        >>> pred.sfh.mass_weighted_age_gyr  # reuses cached SFH
        >>> pred.sed.l_bol  # triggers SED computation
        >>> pred.sed.uv_slope_beta  # reuses cached SED
        >>> pred.lines.halpha  # triggers nebular computation
        >>> pred.lines.bpt_nii  # reuses cached lines

        **Batch computation (JIT + vmap):**

        >>> import jax
        >>> sfh_fn = jax.vmap(model.predict_sfh_quantities)
        >>> sfh_batch = sfh_fn(params_batch)
        >>> sfh_batch.stellar_mass  # shape (n_galaxies,)

        See Also
        --------
        predict_sfh_quantities : JIT-compatible SFH quantities.
        predict_sed_quantities : JIT-compatible SED quantities.
        predict_line_luminosities : JIT-compatible emission lines.
        """
        from tengri.core.prediction import Prediction

        return Prediction(self, params)

    def predict_sfh_quantities(self, params):
        """Compute SFH-derived quantities (JIT-compatible).

        Returns a :class:`~tengri.prediction.SFHQuantities` NamedTuple
        containing stellar mass, SFR, sSFR, and mass-weighted age and
        metallicity. This method is fully JIT-compatible and can be
        vectorized with ``jax.vmap`` for batch computation over
        posterior chains or mock catalogs.

        Parameters
        ----------
        params : dict
            Parameter values (public names).

        Returns
        -------
        SFHQuantities
            NamedTuple with fields: ``stellar_mass``,
            ``stellar_mass_surviving``, ``sfr_100myr``, ``sfr_10myr``,
            ``ssfr``, ``mass_weighted_age_gyr``,
            ``mass_weighted_metallicity``.

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

        See Also
        --------
        predict : Lazy prediction for single-galaxy exploration.
        predict_sed_quantities : JIT-compatible SED quantities.
        """
        from tengri.core.prediction import SFHQuantities
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
            from tengri.models.sps.dsps_wrapper import compute_dsps_native_weights

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
            from tengri.models.sps.dsps_wrapper import compute_dsps_met_table_weights

            z_val = p.get("redshift", 0.1)
            t_obs_gyr = self._t_universe_gyr(z_val)
            lgmet_scatter = float(p.get("lgmet_scatter", self._lgmet_scatter))
            if self._evolving_metallicity:
                from tengri.models.sps.dsps_wrapper import compute_log_z_evolving

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
            from tengri.models.sps.dsps_wrapper import (
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
        """Compute SED-derived quantities (JIT-compatible).

        Returns a :class:`~tengri.prediction.SEDQuantities` NamedTuple
        containing bolometric and IR luminosities, UV slope, spectral
        indices, and luminosity-weighted age/metallicity. Runs the
        full forward model internally.

        This method is fully JIT-compatible and can be vectorized with
        ``jax.vmap`` for batch computation.

        Parameters
        ----------
        params : dict
            Parameter values (public names).

        Returns
        -------
        SEDQuantities
            NamedTuple with fields: ``l_bol``, ``l_tir``,
            ``l_dust_absorbed``, ``irx``, ``uv_slope_beta``,
            ``dn4000``, ``balmer_break``, ``m_uv``, ``fuv_flux``,
            ``nuv_flux``, ``fuv_flux_intrinsic``, ``nuv_flux_intrinsic``,
            ``rest_uv_color``, ``luminosity_weighted_age_gyr``,
            ``luminosity_weighted_metallicity``.

        Examples
        --------
        **Single galaxy:**

        >>> sed_q = model.predict_sed_quantities(params)
        >>> sed_q.l_bol
        Array(2.5e10, dtype=float64)
        >>> sed_q.dn4000
        Array(1.42, dtype=float64)

        **Batch over posterior samples:**

        >>> import jax
        >>> sed_fn = jax.vmap(model.predict_sed_quantities)
        >>> sed_batch = sed_fn(params_batch)
        >>> sed_batch.m_uv  # shape (n_samples,)

        See Also
        --------
        predict : Lazy prediction for single-galaxy exploration.
        predict_sfh_quantities : JIT-compatible SFH quantities.
        """
        from tengri.core.prediction import SEDQuantities
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

    def predict_photometry(self, params):
        """Compute observed photometric flux densities.

        Automatically uses the fast precomputed path (Zacharegkas+2025)
        when available (redshift fixed + filters present + precompute=True).
        Falls back to exact wavelength integration otherwise.

        Parameters
        ----------
        params : dict
            Parameter values (public names).

        Returns
        -------
        array, shape (n_filters,)
            Observed flux densities in erg/s/cm^2/Hz.
        """
        if self.filter_waves is None:
            raise ValueError("No filters set. Pass filters to Model().")

        # Tabulated SFH bypasses fused kernel (needs full SED path)
        _has_tabulated_sfh = "sfh_t_gyr" in params

        if (
            self._precomp is not None
            and self._fused_photometry is not None
            and not _has_tabulated_sfh
        ):
            return self._predict_photometry_fast(params)

        if (
            self._ztable is not None
            and self._fused_photometry_ztable is not None
            and not _has_tabulated_sfh
        ):
            return self._predict_photometry_ztable(params)

        # Tier 2: compositional rest-frame SED + observation wrapper
        if (
            self._fused_rest_sed is not None
            and not _has_tabulated_sfh
            and not self._evolving_metallicity
            and not getattr(self, "_chem_evol_enabled", False)
        ):
            return self._predict_photometry_tier2(params)

        # Tier 3: exact path (Python dispatch)
        sed = self.predict_sed(params)
        z = self._get_redshift(params)
        dl_cm = self._get_dl_cm(params)

        # Apply IGM absorption (acts on observed-frame SED)
        if self._apply_igm:
            from tengri.models.igm import igm_transmission

            wave_obs = self.ssp_data.ssp_wave * (1.0 + z)
            igm_trans = igm_transmission(wave_obs, z)
            sed = sed * igm_trans

        fluxes = []
        for fw, ft in zip(self.filter_waves, self.filter_trans):
            f = compute_flux_density(
                sed,
                self.ssp_data.ssp_wave,
                fw,
                ft,
                z,
                dl_cm,
            )
            fluxes.append(f)
        return jnp.array(fluxes)

    def _get_dust_kwargs(self, p):
        """Extract dust law + emission kwargs from internal params dict."""
        return get_dust_kwargs(self, p)

    def _get_agn_kwargs(self, p):
        """Extract AGN kwargs from internal params dict for fused kernel."""
        return get_agn_kwargs(self, p)

    def _predict_photometry_fast(self, params):
        """Fast photometry using fused JIT kernel (fixed z)."""
        p = self._get_internal_params(params)
        sfr = self._compute_sfr(p)
        sfr_on_ssp = jnp.interp(self.ssp_log_ages_yr, self.log_age_grid, sfr)
        if self._dust_model == "single_component":
            return self._fused_photometry(
                sfr_on_ssp,
                p["log_z_abs"],
                p["tau_v"],
                p["dust_slope"],
                **self._get_dust_kwargs(p),
                **self._get_agn_kwargs(p),
            )
        return self._fused_photometry(
            sfr_on_ssp,
            p["log_z_abs"],
            p["tau_bc"],
            p["tau_diff"],
            p["dust_slope"],
            **self._get_dust_kwargs(p),
            **self._get_agn_kwargs(p),
        )

    def _predict_photometry_ztable(self, params):
        """Fast photometry using z-table interpolation (free z)."""
        p = self._get_internal_params(params)
        sfr = self._compute_sfr(p)
        sfr_on_ssp = jnp.interp(self.ssp_log_ages_yr, self.log_age_grid, sfr)
        z = self._get_redshift(params)
        if self._dust_model == "single_component":
            return self._fused_photometry_ztable(
                sfr_on_ssp,
                p["log_z_abs"],
                p["tau_v"],
                p["dust_slope"],
                z,
                **self._get_dust_kwargs(p),
                **self._get_agn_kwargs(p),
            )
        return self._fused_photometry_ztable(
            sfr_on_ssp,
            p["log_z_abs"],
            p["tau_bc"],
            p["tau_diff"],
            p["dust_slope"],
            z,
            **self._get_dust_kwargs(p),
            **self._get_agn_kwargs(p),
        )

    # -------------------------------------------------------------------
    # Tier 2: Compositional rest-frame SED dispatch
    # -------------------------------------------------------------------

    def _compute_rest_sed_tier2(self, params):
        """Compute rest-frame SED via the compositional JIT kernel (Tier 2).

        Handles SFH computation, metallicity interpolation, and delegates
        the rest (dust, nebular, AGN, radio, X-ray) to the fused kernel.

        Parameters
        ----------
        params : dict
            Parameter values (public names).

        Returns
        -------
        array, shape (n_wave,)
            Rest-frame SED in erg/s/Hz.
        """
        from tengri.core.sed_pipeline import interp_met_alpha_dispatch, interp_metallicity

        p = self._get_internal_params(params)
        sfr = self._compute_sfr(p)
        sfr_on_ssp = jnp.interp(self.ssp_log_ages_yr, self.log_age_grid, sfr)
        if self._csp_integration == "log_interp":
            weights = self._csp_matrix @ sfr_on_ssp
        elif self._csp_integration == "dsps_native":
            from tengri.models.sps.dsps_wrapper import compute_dsps_native_weights

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
            if self._xray_enabled:
                p = {**p, "_sfr_current": sfr[-1]}
            return self._fused_rest_sed(weights, ssp_flux_at_z, p)
        elif self._csp_integration == "dsps_met_table":
            from tengri.models.sps.dsps_wrapper import compute_dsps_met_table_weights

            z_val = p.get("redshift", 0.0)
            t_obs_gyr = self._t_universe_gyr(z_val) if hasattr(self, "_t_universe_gyr") else 13.7
            lgmet_scatter = float(p.get("lgmet_scatter", self._lgmet_scatter))
            if self._evolving_metallicity:
                from tengri.models.sps.dsps_wrapper import compute_log_z_evolving

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
            if self._xray_enabled:
                p = {**p, "_sfr_current": sfr[-1]}
            return self._fused_rest_sed(weights, ssp_flux_at_z, p)
        else:
            weights = sfr_on_ssp * self._csp_age_dt

        # Metallicity interpolation (single Z, non-evolving path).
        # effective_metallicity alpha correction is opt-in: only applied when
        # met_alpha_fe is explicitly a free parameter.
        _use_alpha_fe = (
            getattr(self.spec, "alpha_fe_evolving", False)
            or "met_alpha_fe" in self.spec.free_params
        )
        if _use_alpha_fe:
            alpha_fe = p.get("alpha_fe", 0.0)
            ssp_flux_at_z = interp_met_alpha_dispatch(self, p["log_z_abs"], alpha_fe)
        else:
            ssp_flux_at_z = interp_metallicity(self, p["log_z_abs"])

        # Enrich p with current SFR for X-ray model
        if self._xray_enabled:
            p = {**p, "_sfr_current": sfr[-1]}

        return self._fused_rest_sed(weights, ssp_flux_at_z, p)

    def _predict_photometry_tier2(self, params):
        """Photometry via Tier 2: compositional rest SED + filter integration.

        Uses the fused end-to-end JIT kernel when available (eliminates
        Python dispatch between SFH, metallicity, SED, and filter steps).
        Falls back to unfused path otherwise.
        """
        if self._fused_tier2_phot is not None:
            return self._fused_tier2_phot(params)

        rest_sed = self._compute_rest_sed_tier2(params)
        z = self._get_redshift(params)
        dl_cm = self._get_dl_cm(params)
        return observe_photometry_from_rest_sed(
            rest_sed,
            self.ssp_data.ssp_wave,
            z,
            dl_cm,
            self.filter_waves,
            self.filter_trans,
            apply_igm=self._apply_igm,
        )

    def _predict_spectrum_tier2(self, params, wave_obs):
        """Spectrum via Tier 2: compositional rest SED + interpolation.

        Uses the fused end-to-end JIT kernel when available and the
        wave_obs grid matches the precomputed grid. Falls back to
        unfused path otherwise.
        """
        if (
            self._fused_tier2_spec is not None
            and self._spec_precomp is not None
            and wave_obs is self._spec_precomp.wave_obs_pixels
        ):
            flux = self._fused_tier2_spec(params)
            # Apply LSF if needed (below)
        else:
            rest_sed = self._compute_rest_sed_tier2(params)
            z = self._get_redshift(params)
            dl_cm = self._get_dl_cm(params)
            flux = observe_spectrum_from_rest_sed(
                rest_sed, self.ssp_data.ssp_wave, wave_obs, z, dl_cm
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
        """
        if self._z_fixed is None:
            raise ValueError("Spectroscopy precomputation requires fixed redshift")
        self._spec_precomp = precompute_spectroscopy(
            self.ssp_data,
            jnp.asarray(wave_obs),
            self._z_fixed,
            self._dl_cm_fixed,
        )
        self._wave_obs = jnp.asarray(wave_obs)
        if is_fused_compatible(self):
            self._fused_spectrum = build_fused_spectrum(self)
        else:
            self._fused_spectrum = None

        # Build fused Tier 2 spectrum (end-to-end JIT)
        self._fused_tier2_spec = None
        if self._fused_rest_sed is not None:
            with contextlib.suppress(Exception):
                self._fused_tier2_spec = build_fused_tier2_spectrum(self)
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
        """
        if self.filter_waves is None:
            raise ValueError("Z-table precomputation requires filters to be set")
        self._ztable = precompute_photometry_ztable(
            self.ssp_data,
            self.filter_waves,
            self.filter_trans,
            z_grid=z_grid,
            z_min=z_min,
            z_max=z_max,
            n_z=n_z,
            apply_igm=self._apply_igm and self._approx.get("igm", True),
        )
        self._fused_photometry_ztable = build_fused_photometry_ztable(self)
        return self

    def predict_spectrum(self, params, wave_obs=None):
        """Compute observed spectrum at given wavelengths.

        Uses the fast precomputed path if ``precompute_spectroscopy()``
        was called, otherwise falls back to exact interpolation.

        Parameters
        ----------
        params : dict
            Parameter values.
        wave_obs : array, optional
            Observed wavelength grid (Angstrom). If None, uses the
            grid from ``precompute_spectroscopy()``.

        Returns
        -------
        array, shape (n_pix,)
            Spectral flux density in erg/s/cm^2/Hz.
        """
        if wave_obs is None and self._spec_precomp is not None:
            wave_obs = self._spec_precomp.wave_obs_pixels
        elif wave_obs is None and hasattr(self, "_wave_obs"):
            wave_obs = self._wave_obs
        elif wave_obs is None:
            raise ValueError("No wavelength grid. Pass wave_obs or call precompute_spectroscopy()")

        # Tier 1: use fused kernel if available and compatible
        if self._spec_precomp is not None and self._fused_spectrum is not None:
            return self._predict_spectrum_fast(params)

        # Tier 2: compositional rest-frame SED + observation wrapper
        _has_tabulated_sfh = "sfh_t_gyr" in params
        if (
            self._fused_rest_sed is not None
            and not _has_tabulated_sfh
            and not self._evolving_metallicity
            and not getattr(self, "_chem_evol_enabled", False)
        ):
            return self._predict_spectrum_tier2(params, wave_obs)

        # Tier 3: exact path (Python dispatch)
        sed = self.predict_sed(params)
        z = self._get_redshift(params)
        dl_cm = self._get_dl_cm(params)
        flux = compute_spectrum(
            sed,
            self.ssp_data.ssp_wave,
            wave_obs,
            z,
            dl_cm,
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

    def _predict_spectrum_fast(self, params):
        """Fast spectrum using fused JIT kernel."""
        p = self._get_internal_params(params)
        sfr = self._compute_sfr(p)
        sfr_on_ssp = jnp.interp(self.ssp_log_ages_yr, self.log_age_grid, sfr)
        sigma_v = params.get("sigma_v", 0.0) if self._has_sigma_v else 0.0
        if self._dust_model == "single_component":
            flux = self._fused_spectrum(
                sfr_on_ssp,
                p["log_z_abs"],
                p["tau_v"],
                p["dust_slope"],
                sigma_v,
                **self._get_dust_kwargs(p),
                **self._get_agn_kwargs(p),
            )
        else:
            flux = self._fused_spectrum(
                sfr_on_ssp,
                p["log_z_abs"],
                p["tau_bc"],
                p["tau_diff"],
                p["dust_slope"],
                sigma_v,
                **self._get_dust_kwargs(p),
                **self._get_agn_kwargs(p),
            )

        # Apply LSF convolution if resolution profile is set
        # (applied after the fused kernel, which handles velocity broadening)
        resolution = self._lsf_resolution
        if resolution is not None:
            wave_obs = self._spec_precomp.wave_obs_pixels
            flux = apply_lsf(
                flux,
                wave_obs,
                resolution,
                sigma_lib_kms=self._sigma_lib_kms,
                n_bins=self._lsf_n_bins,
            )

        return flux

    def predict_sfh(self, params, n_linear=1000):
        """Compute SFH on a uniform linear-time grid for plotting.

        Parameters
        ----------
        params : dict
            Parameter values.
        n_linear : int
            Number of output grid points.

        Returns
        -------
        dict with keys:
            "t_gyr": lookback time (Gyr), shape (n_linear,)
            "sfr_mean": mean SFH (Msun/yr), shape (n_linear,)
            "sfr_full": full SFH including GP (Msun/yr), shape (n_linear,)
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

    def predict_hbeta(self, params: dict) -> float:
        """Predict Hβ luminosity for use with CLOUDY-informed emission line priors.

        Required by ``marginalize_emission_lines_cloudy()`` as the ``l_hbeta``
        argument, which scales CLOUDY's ratio-relative-to-Hβ priors to physical
        units.

        Hβ luminosity is computed via the Case B recombination approximation
        (Leitherer et al. 1999):

        .. math::

            L_{H\\beta} \\approx 52.2 \\times \\text{SFR}_{10} \\; [L_\\odot]

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

        See Also
        --------
        predict_sfh_quantities : JIT-compatible SFH quantities including sfr_10myr.
        """
        # Case B: L_Hbeta [Lsun] = 4.76e-13 * Q_H, Q_H = 4.2e53 * SFR [Msun/yr]
        # => L_Hbeta = 4.76e-13 * 4.2e53 / 3.828e33 * SFR ≈ 52.2 * SFR
        _L_HBETA_PER_SFR = 52.2  # Lsun per Msun/yr (Leitherer+1999)
        try:
            sfh_q = self.predict_sfh_quantities(params)
            sfr_10 = float(sfh_q.sfr_10myr)
            sfr_10 = max(sfr_10, 1e-10)
            return float(_L_HBETA_PER_SFR * sfr_10)
        except Exception:
            return 1.0  # 1 Lsun safe fallback

    def predict_derived(self, params):
        """Compute derived physical quantities.

        .. deprecated::
            Use ``model.predict(params)`` for lazy on-demand access,
            or ``model.predict_sfh_quantities(params)`` for JIT-compatible
            batch computation.

        This method is kept for backward compatibility and returns a
        dict with the same keys as before.

        Parameters
        ----------
        params : dict
            Parameter values.

        Returns
        -------
        dict with keys:
            "stellar_mass": total mass formed (Msun)
            "stellar_mass_surviving": surviving mass in living stars +
                remnants (Msun). None if mass-remaining table not loaded.
            "sfr_100myr": SFR averaged over last 100 Myr (Msun/yr)
            "sfr_10myr": SFR averaged over last 10 Myr (Msun/yr)
            "ssfr": specific SFR (yr^-1), uses surviving mass if
                available, else formed mass.
        """
        pred = self.predict(params)
        mass_surv = pred.sfh.stellar_mass_surviving
        # Backward compat: return None instead of NaN
        mass_surv_out = None if jnp.isnan(mass_surv) else mass_surv
        return {
            "stellar_mass": pred.sfh.stellar_mass,
            "stellar_mass_surviving": mass_surv_out,
            "sfr_100myr": pred.sfh.sfr_100myr,
            "sfr_10myr": pred.sfh.sfr_10myr,
            "ssfr": pred.sfh.ssfr,
        }

    def predict_magnitudes(self, params):
        """Compute observed AB magnitudes through all filters."""
        if self.filter_waves is None:
            raise ValueError("No filters set.")

        try:
            from dsps import calc_obs_mag
            from dsps.cosmology import DEFAULT_COSMOLOGY

            sed_lsun = self.predict_luminosity(params)
            z = self._get_redshift(params)
            cosmo = DEFAULT_COSMOLOGY

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

        Returns
        -------
        array, shape (n_wave,)
            Rest-frame luminosity in Lsun/Hz.
        """
        LSUN_CGS = 3.828e33  # erg/s (IAU 2015)
        sed_erg = self.predict_sed(params)
        return sed_erg / LSUN_CGS

    def plot_sfh_posterior(
        self, posterior, true_params=None, ax=None, n_draws=50, color="C0", label="Posterior"
    ):
        """Plot posterior SFH with percentile fill and sample lines."""
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

    # -------------------------------------------------------------------
    # Mock generation
    # -------------------------------------------------------------------

    def mock(self, params, snr=20.0, key=None):
        from tengri.core.convenience import mock as _fn

        return _fn(self, params, snr=snr, key=key)

    def mock_spectrum(self, params, wave_obs, snr=30.0, key=None):
        from tengri.core.convenience import mock_spectrum as _fn

        return _fn(self, params, wave_obs, snr=snr, key=key)

    def mock_batch(self, params_batch, snr=20.0, key=None):
        from tengri.core.convenience import mock_batch as _fn

        return _fn(self, params_batch, snr=snr, key=key)

    # -------------------------------------------------------------------
    # Batch predictions (vmap over galaxies)
    # -------------------------------------------------------------------

    def predict_photometry_batch(self, params_batch):
        from tengri.core.convenience import predict_photometry_batch as _fn

        return _fn(self, params_batch)

    def predict_spectrum_batch(self, params_batch):
        from tengri.core.convenience import predict_spectrum_batch as _fn

        return _fn(self, params_batch)

    # -------------------------------------------------------------------
    # Factory classmethod
    # -------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        ssp,
        sfh: str = "dpl",
        dust: str = "charlot_fall",
        nebular: str | None = None,
        agn: str | None = None,
        redshift: float | str = 0.1,
        filters: list[str] | None = None,
        wave_obs=None,
        priors: dict | None = None,
        **model_kwargs,
    ) -> Model:
        """Build a Model from a grouped configuration dict.

        Reduces boilerplate for the common case: instead of constructing
        ``ParamSpec``, ``SSPData``, ``Observation``, and ``Model`` separately,
        provide a single grouped config and receive a fully configured ``Model``.

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
            Forwarded to ``Model.__init__()``.

        Returns
        -------
        Model

        Examples
        --------
        >>> model = tengri.Model.from_config(
        ...     ssp="data/ssp.h5",
        ...     sfh="tsnorm",
        ...     filters=["sdss_u", "sdss_g", "sdss_r"],
        ...     redshift=0.1,
        ...     priors=dict(
        ...         log_peak_sfr=tengri.Uniform(-1, 2.5),
        ...         logzsol=tengri.Uniform(-2, 0.2),
        ...     ),
        ... )
        """
        from tengri.core.convenience import build_model_from_config

        return build_model_from_config(
            cls,
            ssp,
            sfh=sfh,
            dust=dust,
            nebular=nebular,
            agn=agn,
            redshift=redshift,
            filters=filters,
            wave_obs=wave_obs,
            priors=priors,
            **model_kwargs,
        )

    # -------------------------------------------------------------------
    # Prior predictive check
    # -------------------------------------------------------------------

    def prior_predictive(self, n: int = 500, seed: int = 42) -> PriorPredictive:
        from tengri.core.convenience import prior_predictive as _fn

        return _fn(self, n=n, seed=seed)

    # -------------------------------------------------------------------
    # Convenience fit
    # -------------------------------------------------------------------

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

        Examples
        --------
        >>> result = model.fit(flux_obs, noise)
        >>> result = model.fit(flux_obs, noise, method="mcmc")
        >>> result = model.fit(photometry=(flux_p, noise_p), spectrum=(flux_s, noise_s))
        >>> result = model.fit(flux_obs, noise, init="map")
        >>> result = model.fit(flux_obs, noise).refine("mcmc_raytrace")
        """
        from tengri.core.convenience import fit_model

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

    def fit_catalog(
        self,
        catalog,
        flux_cols: list[str],
        err_cols: list[str],
        redshift_col: str | None = None,
        method: str = "vi",
        n_workers: int = 1,
        verbose: bool = True,
        **kwargs,
    ) -> list:
        from tengri.core.convenience import fit_catalog as _fn

        return _fn(
            self,
            catalog,
            flux_cols,
            err_cols,
            redshift_col=redshift_col,
            method=method,
            n_workers=n_workers,
            verbose=verbose,
            **kwargs,
        )

    def _method_recommendation(self) -> tuple[str, str]:
        """Return (method_name, reason) for the recommended inference method."""
        from tengri.core.display import method_recommendation

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

        Examples
        --------
        >>> print(model.tree())
        Model  [D=7, stochastic=False]
        ...
        """
        from tengri.core.display import tree as _tree

        return _tree(self)

    def recommend_method(self) -> str:
        """Return the recommended inference method string for this model.

        Returns
        -------
        str
            Canonical method name for ``Fitter.run()`` or ``model.fit()``.

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
        """
        from tengri.core.display import summary as _summary

        return _summary(self)

    # -------------------------------------------------------------------
    # Population fitting
    # -------------------------------------------------------------------

    def fit_population(
        self,
        observations_list: list,
        method: str = "vi",
        population_prior: dict | None = None,
        **kwargs,
    ):
        from tengri.core.convenience import fit_population as _fn

        return _fn(
            self,
            observations_list,
            method=method,
            population_prior=population_prior,
            **kwargs,
        )
