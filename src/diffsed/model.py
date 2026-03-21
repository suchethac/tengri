"""High-level Model class wrapping the diffsed forward model.

Model provides a clean API for:
- Forward predictions (SED, photometry, spectrum, SFH, derived quantities)
- Mock galaxy generation (single and batch)
- Convenience fitting (delegates to Fitter)

The Model translates between the user-facing parameter names and the
internal names used by the low-level functions, handling unit conversions
automatically. SFH computation is dispatched through the registry-driven
composed function, eliminating separate stochastic/parametric code paths.

Usage::

    from diffsed import Model, ParamSpec, Uniform, load_ssp_data, load_filter_set

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

from pathlib import Path
from typing import ClassVar, NamedTuple

import jax
import jax.numpy as jnp

from diffsed.models.dust.attenuation import (
    precompute_dust_age_weights,
    two_component_dust_fast,
)
from diffsed.models.observation.photometry import ab_mag_from_flux, compute_flux_density
from diffsed.models.observation.spectroscopy import apply_lsf, compute_spectrum
from diffsed.models.sfh.registry import compute_field_gp, resolve_sfh
from diffsed.models.sps.dsps_wrapper import (
    compute_csp_sed,
    compute_csp_weights,
    compute_log_z_evolving,
    effective_metallicity,
    interpolate_metallicity,
    interpolate_metallicity_evolving,
    interpolate_metallicity_smooth,
    interpolate_metallicity_smooth_evolving,
)
from diffsed.models.sps.precompute import (
    precompute_photometry,
    precompute_photometry_ztable,
    precompute_spectroscopy,
)
from diffsed.utils.cosmology import age_at_z, luminosity_distance
from diffsed.utils.grid import (
    grid_spacing,
    interpolate_to_linear_time,
    log_age_to_age_yr,
    make_log_age_grid,
)

# ---------------------------------------------------------------------------
# Non-SFH parameter mapping: public → (internal, unit_scale, offset)
# ---------------------------------------------------------------------------

# Solar metallicity: log10(Zsun) = log10(0.0142) ≈ -1.848 (Asplund 2009)
LOG10_ZSUN = -1.8477116556169435

_EVOLVING_MET_PARAM_MAP = {
    "met_logzsol_0": ("log_z_initial", 1.0, LOG10_ZSUN),  # log(Z/Zsun) → log(Z)
    "met_logzsol_final": ("log_z_final", 1.0, LOG10_ZSUN),
}

_NON_SFH_PARAM_MAP = {
    "met_logzsol": ("log_z", 1.0, LOG10_ZSUN),  # log(Z/Zsun) → log(Z)
    "met_alpha_fe": ("alpha_fe", 1.0, 0.0),  # [alpha/Fe] in dex
    "dust_tau_bc": ("tau_v1", 1.0, 0.0),
    "dust_tau_diff": ("tau_v2", 1.0, 0.0),
    "dust_slope": ("dust_n", 1.0, 0.0),
    "redshift": ("redshift", 1.0, 0.0),
    "noise_frac_cal": ("noise_frac_cal", 1.0, 0.0),
    # Dust extra params (identity mapping — no unit conversion)
    "dust_f_obscuration": ("f_obscuration", 1.0, 0.0),
    "dust_bump_strength": ("dust_bump_strength", 1.0, 0.0),
    "dust_delta": ("dust_delta", 1.0, 0.0),
    "dust_Rv": ("dust_Rv", 1.0, 0.0),
    # Noise model
    "noise_dof": ("noise_dof", 1.0, 0.0),
}


def _build_param_map(mean_sfh_type):
    """Build complete PARAM_MAP from SFH registry + non-SFH params.

    Returns
    -------
    dict
        public_name -> (internal_name, scale, offset)
    """
    _, _, sfh_param_map, _ = resolve_sfh(mean_sfh_type)
    result = dict(sfh_param_map)
    result.update(_NON_SFH_PARAM_MAP)
    return result


# Legacy module-level PARAM_MAP for backward compatibility with imports
PARAM_MAP = {
    "sfh_alpha": ("alpha", 1.0, 0.0),
    "sfh_beta": ("beta", 1.0, 0.0),
    "sfh_tau_peak_gyr": ("tau_sfh", 1e9, 0.0),
    "sfh_peak_sfr": ("sfr_norm", 1.0, 0.0),
    "psd_sigma": ("sigma_ps", 1.0, 0.0),
    "psd_tau_myr": ("tau_ps", 1e6, 0.0),
    "met_logzsol": ("log_z", 1.0, LOG10_ZSUN),
    "dust_tau_bc": ("tau_v1", 1.0, 0.0),
    "dust_tau_diff": ("tau_v2", 1.0, 0.0),
    "dust_slope": ("dust_n", 1.0, 0.0),
    "redshift": ("redshift", 1.0, 0.0),
}


# ---------------------------------------------------------------------------
# MockData container
# ---------------------------------------------------------------------------


class MockData(NamedTuple):
    """Container for mock galaxy observations."""

    flux_true: jnp.ndarray  # noiseless photometry (erg/s/cm²/Hz)
    flux_obs: jnp.ndarray  # noisy photometry
    noise: jnp.ndarray  # 1-sigma uncertainties
    params: dict  # input parameters


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
        self, spec, ssp_data, filters=None, precompute=True, forward_dtype="float64", approx=None
    ):
        self.spec = spec
        self.ssp_data = ssp_data
        self._forward_dtype = jnp.dtype(forward_dtype)

        # Initialize optional-component flags early (before any code that
        # might short-circuit or raise, so attribute lookups never fail).
        self._radio_enabled = getattr(spec, "radio", False)
        self._xray_enabled = getattr(spec, "xray", False)
        self._agn_model = getattr(spec, "agn_model", None)
        self._agn_parametric = False
        self._dust_emission_model = getattr(spec, "dust_emission", None)
        self._evolving_metallicity = getattr(spec, "evolving_metallicity", False)
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
        if filters is not None:
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
        self._param_map = _build_param_map(spec.mean_sfh_type)

        # Field settings
        self._has_field = spec.stochastic
        self._field_model = sfh_settings.get("sfh_field_model", "drw")

        # Dust law settings (generalized two-component model)
        self._dust_law_bc = spec.dust_law_bc
        self._dust_law_diff = spec.dust_law_diff
        # Cache resolved dust law functions (avoid dict lookup per forward call)
        from diffsed.models.dust.attenuation import get_dust_law

        self._dust_law_bc_fn = get_dust_law(self._dust_law_bc)
        self._dust_law_diff_fn = get_dust_law(self._dust_law_diff)

        # IGM absorption (Inoue+2014)
        self._apply_igm = spec.apply_igm

        # Dust emission model (None = disabled)
        self._dust_emission_model = getattr(spec, "dust_emission", None)
        if self._dust_emission_model == "dl07_tabulated":
            from diffsed.models.dust.emission import DUST_EMISSION_MODELS

            if "dl07_tabulated" not in DUST_EMISSION_MODELS:
                from diffsed.models.dust.emission import create_dl07_from_grid

                dl07_path = getattr(spec, "dl07_grid_path", None)
                if dl07_path is None:
                    # Try default locations
                    for candidate in [
                        Path(__file__).resolve().parents[2] / "data" / "dl07_templates.h5",
                        Path("data/dl07_templates.h5"),
                    ]:
                        if candidate.is_file():
                            dl07_path = str(candidate)
                            break
                if dl07_path is None or not Path(dl07_path).is_file():
                    raise FileNotFoundError(
                        "DL07 templates not found. Set dl07_grid_path in ParamSpec "
                        "or run: python scripts/convert_dl07_templates.py"
                    )
                DUST_EMISSION_MODELS["dl07_tabulated"] = create_dl07_from_grid(dl07_path)
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

        # Nebular emission backend + params
        if spec.nebular_mode in ("cloudy", "cue"):
            self._param_map["neb_logU"] = ("neb_logU", 1.0, 0.0)
            self._param_map["neb_logZ_gas"] = ("neb_logZ_gas", 1.0, LOG10_ZSUN)
            self._param_map["neb_fesc"] = ("neb_fesc", 1.0, 0.0)
            self._param_map["neb_fesc_lya"] = ("neb_fesc_lya", 1.0, 0.0)

        self._nebular_backend = None
        if spec.nebular_mode == "cue":
            from diffsed.models.nebular import CueBackend

            self._nebular_backend = CueBackend(spec.cue_weights_path, ssp_data=ssp_data)
        elif spec.nebular_mode == "cloudy":
            from diffsed.models.nebular import CloudyGridBackend

            self._nebular_backend = CloudyGridBackend(spec.cloudy_grid_path, ssp_data)
        elif spec.nebular_mode == "ssp":
            from diffsed.models.nebular import BakedInBackend

            self._nebular_backend = BakedInBackend()
        else:
            from diffsed.models.nebular import BakedInBackend

            self._nebular_backend = BakedInBackend()

        # For DL07 tabulated templates, load at init and register
        if self._dust_emission_model == "dl07_tabulated":
            dl07_path = getattr(spec, "dl07_grid_path", "data/dl07_templates.h5")
            from diffsed.models.dust.emission import DUST_EMISSION_MODELS, create_dl07_from_grid

            DUST_EMISSION_MODELS["dl07_tabulated"] = create_dl07_from_grid(dl07_path)

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
        self._sigma_lib_kms = getattr(spec, "sigma_lib_kms", 0.0)

        # Instrument LSF resolution profile (None = no LSF, scalar or array)
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
        self._dust_age_weights = precompute_dust_age_weights(self.ssp_ages_yr)

        # Precompute IGM at effective wavelengths (for fused kernel)
        self._igm_at_eff = None
        if (
            self._apply_igm
            and self._approx["igm"]
            and self._precomp is not None
            and self._z_fixed is not None
        ):
            from diffsed.models.igm import igm_transmission

            eff_obs = self._precomp.effective_wavelengths
            self._igm_at_eff = igm_transmission(eff_obs, self._z_fixed)

        # Build fused JIT kernels for fast photometry/spectroscopy
        self._fused_photometry = None
        if self._precomp is not None and self._fused_compatible():
            self._fused_photometry = self._build_fused_photometry()

        # JIT-compiled exact-path SED kernel (eliminates Python dispatch overhead)
        self._jit_exact_sed = self._build_exact_sed()

        # Spectroscopy precomputation (same idea: pre-interpolate SSPs)
        self._spec_precomp = None

        # Z-table precomputation (for free-redshift fitting)
        self._ztable = None
        self._fused_photometry_ztable = None

    def _fused_compatible(self):
        """Check if the fused JIT kernel can handle the current model config.

        Respects ``self._approx`` settings: if a component's approximation
        is disabled, the fused kernel cannot handle it and the model falls
        back to the exact path.

        Emits warnings for each active approximation so the user knows
        what trade-offs are being made.
        """
        import warnings

        reasons = []  # reasons to fall back to exact

        # Dust attenuation approximation
        if not self._approx["dust_attenuation"]:
            reasons.append("dust_attenuation approx disabled by user")

        # Nebular: Cloudy with free params can't be fused
        neb_ok = self._nebular_backend is None or not getattr(
            self._nebular_backend, "has_free_params", False
        )
        if not neb_ok:
            reasons.append("nebular emission (Cloudy) requires full SED")

        # Dust emission: MBB/dale2014 supported if approx enabled
        if self._dust_emission_model is not None:
            if not self._approx["dust_emission"]:
                reasons.append("dust_emission approx disabled by user")
            elif self._dust_emission_model not in ("modified_blackbody", "dale2014"):
                reasons.append(
                    f"dust_emission='{self._dust_emission_model}' not supported "
                    f"in fused kernel (only modified_blackbody, dale2014)"
                )

        # AGN: legacy mode (agn_frac) forces exact path (needs L_bol from
        # full SED integral). Parametric mode (agn_log_lbol) is fused-compatible.
        if self._agn_model is not None and not self._agn_parametric:
            reasons.append(
                "AGN (legacy agn_frac mode) requires full SED for bolometric luminosity"
            )

        # IGM: can be precomputed at effective wavelengths if approx enabled
        if self._apply_igm and not self._approx["igm"]:
            reasons.append("igm approx disabled by user")

        if reasons:
            if self._precomp is not None:
                warnings.warn(
                    "Fused kernel disabled, using exact path (slower). "
                    f"Reasons: {'; '.join(reasons)}. "
                    "Set approx=True or remove incompatible components for "
                    "~10-50x speedup.",
                    UserWarning,
                    stacklevel=3,
                )
            return False

        # Emit approximation warnings for active components
        active_approx = []
        if self._approx["dust_attenuation"]:
            active_approx.append(
                "dust attenuation at filter effective wavelengths "
                "(<3% error for most laws, ~36% for SMC)"
            )
        if self._dust_emission_model in ("modified_blackbody", "dale2014"):
            active_approx.append(
                f"dust emission ({self._dust_emission_model}) with approximate "
                "L_absorbed from broadband fluxes"
            )
        if self._apply_igm and self._approx["igm"]:
            active_approx.append("IGM absorption precomputed at filter effective wavelengths")
        if self._agn_model is not None and self._agn_parametric:
            active_approx.append(
                "AGN (parametric agn_log_lbol) evaluated at filter effective "
                "wavelengths. The AGN SED shape (power-law disc + blackbody "
                "torus) varies strongly across optical-IR; effective-wavelength "
                "approximation may be less accurate than for dust (~10-20% error "
                "in broadband fluxes for AGN-dominated bands)"
            )

        if active_approx:
            warnings.warn(
                "Fused kernel active with approximations: "
                + "; ".join(active_approx)
                + ". Set approx=False to disable.",
                UserWarning,
                stacklevel=3,
            )

        return True

    # -------------------------------------------------------------------
    # Internal parameter translation
    # -------------------------------------------------------------------

    def _get_internal_params(self, params):
        """Translate public param dict to internal names with unit conversion.

        Merges user-provided values with fixed defaults from the spec.
        Conversion: internal = public * scale + offset

        Also accepts legacy parameter names (sfh_alpha, psd_sigma, etc.)
        via reverse alias lookup.
        """
        internal = {}
        for pub_name, (int_name, scale, offset) in self._param_map.items():
            if pub_name in params:
                internal[int_name] = params[pub_name] * scale + offset
            else:
                # Check legacy alias: find old name that maps to pub_name
                legacy_val = self._find_legacy_param(params, pub_name)
                if legacy_val is not None:
                    internal[int_name] = legacy_val * scale + offset
                else:
                    # Fall back to fixed value from spec
                    try:
                        dist = self.spec.get_distribution(pub_name)
                        if dist.is_fixed:
                            internal[int_name] = dist.bounds[0] * scale + offset
                        else:
                            raise KeyError(f"Free parameter '{pub_name}' not found in params dict")
                    except KeyError as err:
                        raise KeyError(
                            f"Parameter '{pub_name}' not found in params dict and not in spec"
                        ) from err

        # Handle field latent vector (both new and legacy names)
        if self._has_field:
            if "sfh_field_xi" in params:
                internal["xi"] = params["sfh_field_xi"]
            elif "psd_xi" in params:
                internal["xi"] = params["psd_xi"]

        # Warn about unrecognized keys (silent bugs when wrong names used)
        recognized = set(self._param_map.keys())
        recognized.update(self._LEGACY_ALIASES.keys() if hasattr(self, "_LEGACY_ALIASES") else ())
        recognized.update({"sfh_field_xi", "psd_xi"})
        # Also recognize internal names (for backwards compat)
        recognized.update(int_name for _, (int_name, _, _) in self._param_map.items())
        unrecognized = set(params.keys()) - recognized
        if unrecognized:
            import warnings

            warnings.warn(
                f"Unrecognized parameter names passed to Model: {sorted(unrecognized)}. "
                f"These will be silently ignored. Did you mean one of: "
                f"{sorted(self._param_map.keys())}?",
                UserWarning,
                stacklevel=3,
            )

        return internal

    @staticmethod
    def _find_legacy_param(params, new_name):
        """Check if a legacy param name is in params that maps to new_name.

        TODO(future): Remove once all callers use new parameter names.
        Legacy callers (fitter.py, hierarchical.py, notebooks) still pass
        old-style param names like sfh_alpha, psd_sigma, etc.
        """
        # Reverse map: new_name -> old_name
        _REVERSE_ALIASES = {
            "sfh_dpl_alpha": "sfh_alpha",
            "sfh_dpl_beta": "sfh_beta",
            "sfh_dpl_tau_gyr": "sfh_tau_peak_gyr",
            "sfh_dpl_log_peak_sfr": "sfh_peak_sfr",
            "sfh_field_psd_sigma": "psd_sigma",
            "sfh_field_psd_tau_myr": "psd_tau_myr",
        }
        old_name = _REVERSE_ALIASES.get(new_name)
        if old_name and old_name in params:
            return params[old_name]
        return None

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
        if self._met_interp == "smooth":
            return interpolate_metallicity_smooth(
                self.ssp_data.ssp_flux, self.ssp_data.ssp_lgmet, log_z, self._lgmet_scatter
            )
        return interpolate_metallicity(self.ssp_data.ssp_flux, self.ssp_data.ssp_lgmet, log_z)

    def _interp_metallicity_evolving(self, log_z_per_age):
        """Dispatch evolving metallicity interpolation (per-age Z)."""
        if self._met_interp == "smooth":
            return interpolate_metallicity_smooth_evolving(
                self.ssp_data.ssp_flux, self.ssp_data.ssp_lgmet, log_z_per_age, self._lgmet_scatter
            )
        return interpolate_metallicity_evolving(
            self.ssp_data.ssp_flux, self.ssp_data.ssp_lgmet, log_z_per_age
        )

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
                psd_tau_myr=p["psd_tau_myr"],
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
                psd_tau_myr=p["psd_tau_myr"],
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
    # Fused JIT kernels
    # -------------------------------------------------------------------

    def _build_fused_photometry(self):
        """Build a single JIT function: SFR-on-SSP → photometry.

        Captures all constants (SSP grid, precomp, dust weights) in the
        closure so XLA can fuse metallicity interpolation, dust, and
        weighted sum into one optimized kernel with no intermediate
        array materializations.

        Supports all registered dust laws (calzetti, kriek_conroy, smc, etc.)
        via captured law functions. For power-law dust, XLA constant-folds
        the curve evaluation to identical code as the old hardcoded path.
        """
        from diffsed.models.dust.attenuation import get_dust_law
        from diffsed.models.sps.dsps_wrapper import LSUN_ERG_PER_S

        dt = self._forward_dtype
        precomp = self._precomp
        ssp_phot = precomp.ssp_phot.astype(dt)
        ssp_lgmet = self.ssp_data.ssp_lgmet.astype(dt)
        eff_waves_rest = precomp.effective_wavelengths_rest.astype(dt)
        dust_age_w = self._dust_age_weights.astype(dt)
        flux_scale = dt.type(precomp.flux_scale)
        ssp_ages_yr = self.ssp_ages_yr.astype(dt)
        lsun = dt.type(LSUN_ERG_PER_S)

        # Capture dust law functions (pure JAX, JIT-traceable)
        law_bc_fn = get_dust_law(self._dust_law_bc)
        law_diff_fn = get_dust_law(self._dust_law_diff)

        from diffsed.models.sps.dsps_wrapper import _ALPHA_TO_Z_COEFF as _A2Z

        # Metallicity interpolation mode for fused kernel
        _use_smooth_z = self._met_interp == "smooth"
        _lgmet_scat = dt.type(self._lgmet_scatter)
        if _use_smooth_z:
            from diffsed.models.sps.dsps_wrapper import compute_lgmet_weights as _clw

        # IGM: precomputed at effective wavelengths (constant for fixed z)
        has_igm = self._igm_at_eff is not None
        if has_igm:
            igm_trans = self._igm_at_eff.astype(dt)

        # Dust emission: precompute constants for MBB at effective wavelengths
        has_dust_em = self._dust_emission_model in ("modified_blackbody", "dale2014")
        if has_dust_em:
            # Precompute frequency at effective wavelengths (constant)
            eff_waves_cm = eff_waves_rest * dt.type(1e-8)
            eff_nu = dt.type(2.99792458e10) / eff_waves_cm  # Hz
            nu_ref_250um = dt.type(2.99792458e10 / 250.0e-4)

        # AGN: capture model function for evaluation at effective wavelengths
        has_agn = self._agn_model is not None and self._agn_parametric
        if has_agn:
            from diffsed.models.agn import get_agn_model

            agn_model_fn = get_agn_model(self._agn_model)

        @jax.jit
        def fused_phot(
            sfr_on_ssp,
            log_z,
            tau_v1,
            tau_v2,
            dust_n,
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
        ):
            sfr = sfr_on_ssp.astype(dt)
            lz = jnp.asarray(log_z, dtype=dt)
            tv1 = jnp.asarray(tau_v1, dtype=dt)
            tv2 = jnp.asarray(tau_v2, dtype=dt)
            dn = jnp.asarray(dust_n, dtype=dt)
            f_obs = jnp.asarray(f_obscuration, dtype=dt)
            bump = jnp.asarray(dust_bump_strength, dtype=dt)
            delta = jnp.asarray(dust_delta, dtype=dt)
            rv = jnp.asarray(dust_Rv, dtype=dt)
            afe = jnp.asarray(alpha_fe, dtype=dt)

            # CSP weights
            age_dt = jnp.concatenate(
                [
                    jnp.array([ssp_ages_yr[1] - ssp_ages_yr[0]]),
                    0.5 * (ssp_ages_yr[2:] - ssp_ages_yr[:-2]),
                    jnp.array([ssp_ages_yr[-1] - ssp_ages_yr[-2]]),
                ]
            )
            weights = sfr * age_dt

            # Alpha enhancement: shift effective metallicity
            lz = lz + _A2Z * afe

            # Metallicity interpolation (respects met_interp setting)
            if _use_smooth_z:
                # Triweight kernel: smooth C2 gradients (ssp_phot is tiny: n_met x n_age x n_filt)
                zw = _clw(lz, ssp_lgmet, _lgmet_scat)
                ssp_at_z = jnp.einsum("m,maf->af", zw, ssp_phot)
            else:
                # 2-point linear (FSPS-style)
                log_z_c = jnp.clip(lz, ssp_lgmet[0], ssp_lgmet[-1])
                idx = jnp.clip(jnp.searchsorted(ssp_lgmet, log_z_c) - 1, 0, len(ssp_lgmet) - 2)
                frac = (log_z_c - ssp_lgmet[idx]) / (ssp_lgmet[idx + 1] - ssp_lgmet[idx])
                ssp_at_z = (1.0 - frac) * ssp_phot[idx] + frac * ssp_phot[idx + 1]

            # Dust: evaluate configurable curves at effective wavelengths
            k_bc = law_bc_fn(
                eff_waves_rest,
                n_slope=dn,
                dust_bump_strength=bump,
                dust_delta=delta,
                dust_Rv=rv,
            )
            k_diff = law_diff_fn(
                eff_waves_rest,
                n_slope=dn,
                dust_bump_strength=bump,
                dust_delta=delta,
                dust_Rv=rv,
            )
            tau = dust_age_w[:, None] * tv1 * k_bc[None, :] + tv2 * k_diff[None, :]
            dust = f_obs + (1.0 - f_obs) * jnp.exp(-tau)

            # Attenuated stellar flux (Lsun)
            flux_attenuated = jnp.einsum("i,if,if->f", weights, dust, ssp_at_z)

            if has_dust_em:
                # Approximate dust emission in the fused kernel:
                # 1. L_stellar (intrinsic, no dust) at effective wavelengths
                flux_intrinsic = jnp.einsum("i,if->f", weights, ssp_at_z)
                # 2. L_absorbed ≈ sum(L_intrinsic - L_attenuated) across bands
                #    This is an approximation — exact would integrate over full SED
                L_absorbed_approx = jnp.sum(flux_intrinsic - flux_attenuated) * lsun
                L_absorbed_approx = jnp.maximum(L_absorbed_approx, dt.type(0.0))
                L_ir = L_absorbed_approx * jnp.asarray(dust_eta_balance, dtype=dt)

                # 3. Modified blackbody at effective wavelengths
                T = jnp.asarray(dust_T, dtype=dt)
                beta = jnp.asarray(dust_beta_ir, dtype=dt)
                emissivity = (eff_nu / nu_ref_250um) ** beta
                x = jnp.clip(
                    dt.type(6.62607015e-27) * eff_nu / (dt.type(1.380649e-16) * T),
                    dt.type(0.0),
                    dt.type(500.0),
                )
                bnu = (
                    dt.type(2.0)
                    * dt.type(6.62607015e-27)
                    * eff_nu**3
                    / dt.type(2.99792458e10) ** 2
                    / (jnp.exp(x) - dt.type(1.0))
                )
                mbb_shape = emissivity * bnu  # (n_filters,)

                # 4. Normalize MBB to L_ir (approximate: use sum over filters)
                mbb_norm = jnp.sum(mbb_shape)
                mbb_norm_safe = jnp.maximum(mbb_norm, dt.type(1e-100))
                dust_em_flux = L_ir / lsun * mbb_shape / mbb_norm_safe

                flux_total = flux_attenuated + dust_em_flux
            else:
                flux_total = flux_attenuated

            # AGN contribution at effective wavelengths (parametric mode)
            if has_agn:
                # Evaluate AGN SED at filter effective wavelengths
                agn_lnu = agn_model_fn(
                    eff_waves_rest,
                    agn_log_lbol=agn_log_lbol,
                    agn_frac=1.0,  # L_bol fully specified by agn_log_lbol
                    agn_alpha=agn_alpha,
                    agn_T_torus=agn_T_torus,
                    agn_tau_torus=agn_tau_torus,
                    agn_torus_frac=agn_torus_frac,
                    agn_log_mbh=agn_log_mbh,
                    agn_log_ledd=agn_log_ledd,
                )
                # agn_lnu is in Lsun/Hz, flux_total is in Lsun at eff wavelengths
                # Convert: L_nu [Lsun/Hz] → add to broadband flux [Lsun]
                # The precomp SSP photometry is L_nu*dnu integrated through
                # the filter, so AGN L_nu is treated the same way (evaluated
                # at the effective wavelength as a representative value).
                flux_total = flux_total + agn_lnu

            # IGM absorption (precomputed at effective wavelengths)
            if has_igm:
                flux_total = flux_total * igm_trans

            return (flux_scale * flux_total * lsun).astype(jnp.float64)

        return fused_phot

    def _build_fused_spectrum(self):
        """Build a single JIT function: SFR-on-SSP → spectrum.

        Same fusion approach as photometry but for spectroscopic pixels.
        Supports all dust laws, f_obscuration, and optional velocity broadening.
        """
        from diffsed.models.dust.attenuation import get_dust_law
        from diffsed.models.sps.dsps_wrapper import _ALPHA_TO_Z_COEFF as _A2Z, LSUN_ERG_PER_S

        fdt = self._forward_dtype
        precomp = self._spec_precomp
        ssp_on_pixels = precomp.ssp_on_pixels.astype(fdt)
        ssp_lgmet = self.ssp_data.ssp_lgmet.astype(fdt)
        wave_obs_pixels = precomp.wave_obs_pixels.astype(fdt)
        wave_rest_pixels = precomp.wave_rest_pixels.astype(fdt)
        dust_age_w = self._dust_age_weights.astype(fdt)
        flux_scale = fdt.type(precomp.flux_scale)
        ssp_ages_yr = self.ssp_ages_yr.astype(fdt)
        lsun = fdt.type(LSUN_ERG_PER_S)
        n_pix = len(wave_obs_pixels)
        has_sigma_v = self._has_sigma_v

        # Capture dust law functions
        law_bc_fn = get_dust_law(self._dust_law_bc)
        law_diff_fn = get_dust_law(self._dust_law_diff)

        # Precompute FFT frequencies for velocity broadening (only if needed)
        if has_sigma_v:
            fft_freq = jnp.fft.rfftfreq(n_pix).astype(fdt)
            dlnwave = jnp.log(wave_obs_pixels[1] / wave_obs_pixels[0]).astype(fdt)
            c_km_s = fdt.type(299792.458)

        # AGN: capture model function for evaluation at pixel wavelengths
        has_agn = self._agn_model is not None and self._agn_parametric
        if has_agn:
            from diffsed.models.agn import get_agn_model

            agn_model_fn = get_agn_model(self._agn_model)

        @jax.jit
        def fused_spec(
            sfr_on_ssp,
            log_z,
            tau_v1,
            tau_v2,
            dust_n,
            sigma_v=0.0,
            f_obscuration=0.0,
            dust_bump_strength=0.0,
            dust_delta=0.0,
            dust_Rv=3.1,
            alpha_fe=0.0,
            agn_log_lbol=10.0,
            agn_alpha=-1.0,
            agn_T_torus=1000.0,
            agn_tau_torus=5.0,
            agn_torus_frac=0.5,
            agn_log_mbh=7.0,
            agn_log_ledd=-1.0,
        ):
            sfr = sfr_on_ssp.astype(fdt)
            lz = jnp.asarray(log_z, dtype=fdt)
            tv1 = jnp.asarray(tau_v1, dtype=fdt)
            tv2 = jnp.asarray(tau_v2, dtype=fdt)
            dn = jnp.asarray(dust_n, dtype=fdt)
            f_obs = jnp.asarray(f_obscuration, dtype=fdt)
            bump = jnp.asarray(dust_bump_strength, dtype=fdt)
            delta = jnp.asarray(dust_delta, dtype=fdt)
            rv = jnp.asarray(dust_Rv, dtype=fdt)
            afe = jnp.asarray(alpha_fe, dtype=fdt)

            # CSP weights
            age_dt = jnp.concatenate(
                [
                    jnp.array([ssp_ages_yr[1] - ssp_ages_yr[0]]),
                    0.5 * (ssp_ages_yr[2:] - ssp_ages_yr[:-2]),
                    jnp.array([ssp_ages_yr[-1] - ssp_ages_yr[-2]]),
                ]
            )
            weights = sfr * age_dt

            # Alpha enhancement: shift effective metallicity
            lz = lz + _A2Z * afe

            # Metallicity interpolation
            log_z_c = jnp.clip(lz, ssp_lgmet[0], ssp_lgmet[-1])
            idx = jnp.clip(jnp.searchsorted(ssp_lgmet, log_z_c) - 1, 0, len(ssp_lgmet) - 2)
            frac = (log_z_c - ssp_lgmet[idx]) / (ssp_lgmet[idx + 1] - ssp_lgmet[idx])
            ssp_at_z = (1.0 - frac) * ssp_on_pixels[idx] + frac * ssp_on_pixels[idx + 1]

            # Dust: configurable curves at pixel wavelengths
            k_bc = law_bc_fn(
                wave_rest_pixels,
                n_slope=dn,
                dust_bump_strength=bump,
                dust_delta=delta,
                dust_Rv=rv,
            )
            k_diff = law_diff_fn(
                wave_rest_pixels,
                n_slope=dn,
                dust_bump_strength=bump,
                dust_delta=delta,
                dust_Rv=rv,
            )
            tau = dust_age_w[:, None] * tv1 * k_bc[None, :] + tv2 * k_diff[None, :]
            dust = f_obs + (1.0 - f_obs) * jnp.exp(-tau)

            # Weighted sum
            flux = jnp.einsum("i,ip,ip->p", weights, dust, ssp_at_z)

            # AGN contribution at pixel wavelengths (parametric mode)
            if has_agn:
                agn_lnu = agn_model_fn(
                    wave_rest_pixels,
                    agn_log_lbol=agn_log_lbol,
                    agn_frac=1.0,  # L_bol fully specified by agn_log_lbol
                    agn_alpha=agn_alpha,
                    agn_T_torus=agn_T_torus,
                    agn_tau_torus=agn_tau_torus,
                    agn_torus_frac=agn_torus_frac,
                    agn_log_mbh=agn_log_mbh,
                    agn_log_ledd=agn_log_ledd,
                )
                flux = flux + agn_lnu

            flux = flux_scale * flux * lsun

            if has_sigma_v:
                sv = jnp.asarray(sigma_v, dtype=fdt)
                sigma_pix = (sv / c_km_s) / dlnwave
                kernel_ft = jnp.exp(-2.0 * jnp.pi**2 * sigma_pix**2 * fft_freq**2)
                flux = jnp.fft.irfft(jnp.fft.rfft(flux) * kernel_ft, n=n_pix)

            return flux.astype(jnp.float64)

        return fused_spec

    # -------------------------------------------------------------------
    # JIT-compiled exact-path SED kernel
    # -------------------------------------------------------------------

    def _build_exact_sed(self):
        """Build a JIT-compiled function for exact-path dust + CSP SED.

        Without JIT, the exact path dispatches ~15 JAX operations through
        Python individually.  Each dispatch costs ~100-300 μs — totalling
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

        Returns a function::

            (weights, ssp_at_z, tau_v1, tau_v2, **kw) -> (sed_atten, sed_intr)

        Typical speedup: 4-14x vs un-JIT'd exact path.
        """
        from diffsed.models.sps.dsps_wrapper import LSUN_ERG_PER_S

        dt = self._forward_dtype
        ssp_wave = self.ssp_data.ssp_wave.astype(dt)
        dust_age_w = self._dust_age_weights.astype(dt)
        lsun = dt.type(LSUN_ERG_PER_S)

        law_bc_fn = self._dust_law_bc_fn
        law_diff_fn = self._dust_law_diff_fn
        same_law = self._dust_law_bc == self._dust_law_diff

        @jax.jit
        def exact_sed(
            weights,
            ssp_at_z,
            tau_v1,
            tau_v2,
            n_slope=-0.7,
            dust_bump_strength=0.0,
            dust_delta=0.0,
            dust_Rv=3.1,
            f_obscuration=0.0,
        ):
            w = weights.astype(dt)
            ssp_z = ssp_at_z.astype(dt)

            # Dust curves — skip duplicate when bc == diff
            k_bc = law_bc_fn(
                ssp_wave,
                n_slope=n_slope,
                dust_bump_strength=dust_bump_strength,
                dust_delta=dust_delta,
                dust_Rv=dust_Rv,
            )
            k_diff = (
                k_bc
                if same_law
                else law_diff_fn(
                    ssp_wave,
                    n_slope=n_slope,
                    dust_bump_strength=dust_bump_strength,
                    dust_delta=dust_delta,
                    dust_Rv=dust_Rv,
                )
            )

            # Dust + CSP SED: XLA fuses broadcast + exp + einsum
            tau = dust_age_w[:, None] * tau_v1 * k_bc[None, :] + tau_v2 * k_diff[None, :]
            dust_trans = f_obscuration + (1.0 - f_obscuration) * jnp.exp(-tau)

            sed_atten = (lsun * jnp.einsum("i,iw,iw->w", w, ssp_z, dust_trans)).astype(jnp.float64)
            sed_intr = (lsun * jnp.einsum("i,iw->w", w, ssp_z)).astype(jnp.float64)

            return sed_atten, sed_intr

        return exact_sed

    # -------------------------------------------------------------------
    # Forward predictions
    # -------------------------------------------------------------------

    def _compute_sed_components(self, params, _sfr=None, _weights=None, need_intrinsic=False):
        """Compute all SED intermediates.

        This is the shared computation engine behind :meth:`predict_sed`,
        :meth:`predict_sed_quantities`, and the lazy :class:`Prediction`
        object. By returning all intermediates, downstream code can
        compute derived quantities without re-running the forward model.

        Parameters
        ----------
        params : dict
            Parameter values (public names).
        _sfr : array, optional
            Pre-computed SFR on the log-age grid (avoids recomputation
            when called from :meth:`predict_derived`).
        _weights : array, optional
            Pre-computed CSP weights.
        need_intrinsic : bool
            If True, always compute the unattenuated stellar SED even
            when no dust emission model is enabled. Required for
            ``l_dust_absorbed`` and intrinsic FUV/NUV.

        Returns
        -------
        dict with keys:
            ``"sed_total"`` : array (n_wave,) — final rest-frame SED
            ``"sed_attenuated"`` : array (n_wave,) — dust-attenuated stellar SED
            ``"sed_intrinsic"`` : array (n_wave,) or None — unattenuated stellar SED
            ``"ssp_flux_at_z"`` : array (n_age, n_wave) — Z-interpolated SSP
            ``"weights"`` : array (n_age,) — CSP mass weights
            ``"sfr"`` : array (n_grid,) — SFR on log-age grid
            ``"p"`` : dict — internal parameter dict
            ``"agn_bol_erg"`` : float — AGN bolometric luminosity (erg/s)
        """
        p = self._get_internal_params(params)

        _dsps_weights_2d = None  # set by DSPS table path if used
        _use_dsps_table = False

        # SFH: parametric (from params) or tabulated (from sfh_t_gyr + sfh_sfr)
        if _sfr is not None:
            sfr = _sfr
        elif "sfh_t_gyr" in params and "sfh_sfr" in params:
            # Tabulated SFH from simulation — use DSPS table functions
            # which properly handle time→age conversion, trapezoidal
            # weighting, and metallicity distribution (lognormal MDF).
            t_cosmic_gyr = jnp.asarray(params["sfh_t_gyr"])
            sfr_table = jnp.asarray(params["sfh_sfr"])
            z = p.get("redshift", 0.0)
            t_obs_gyr = self._t_universe_gyr(z) if hasattr(self, "_t_universe_gyr") else 13.7

            _use_dsps_table = True
            # Also compute sfr on internal grid for SFH plotting
            t_lookback_yr = jnp.maximum((t_obs_gyr - t_cosmic_gyr) * 1e9, 1.0)
            log_t_lookback = jnp.log10(t_lookback_yr)
            sfr_on_ssp = jnp.interp(
                self.ssp_log_ages_yr,
                log_t_lookback[::-1],
                sfr_table[::-1],
            )
            sfr = jnp.interp(
                self.log_age_grid,
                log_t_lookback[::-1],
                sfr_table[::-1],
            )
        else:
            sfr = self._compute_sfr(p)

        if _weights is not None:
            weights = _weights
            _use_dsps_table = False
        elif "sfh_t_gyr" in params:
            weights = compute_csp_weights(sfr_on_ssp, self.ssp_ages_yr)
        else:
            sfr_on_ssp = jnp.interp(self.ssp_log_ages_yr, self.log_age_grid, sfr)
            weights = compute_csp_weights(sfr_on_ssp, self.ssp_ages_yr)
            _use_dsps_table = False

        # Alpha-element enhancement: shift effective metallicity
        alpha_fe = p.get("alpha_fe", 0.0)

        # --- Metallicity + CSP integral ---
        # For tabulated SFH with met_history: use DSPS calc_ssp_weights_sfh_table_met_table
        # which properly handles time→age + metallicity distribution → 2D weights (n_met, n_age).
        # We then apply dust using these 2D weights for correct age-dependent attenuation.
        if _use_dsps_table and "met_history" in params:
            # Use DSPS for the FULL CSP integral with Z(t) history.
            # This properly handles time→age, trapezoidal weighting,
            # and lognormal metallicity distribution at each age.
            from dsps.sed.stellar_sed import calc_rest_sed_sfh_table_met_table

            met_table = jnp.asarray(params["met_history"])
            log_z_abs = effective_metallicity(met_table + (-1.8477), alpha_fe)
            lgmet_scatter = float(params.get("lgmet_scatter", 0.2))
            dsps_result = calc_rest_sed_sfh_table_met_table(
                gal_t_table=t_cosmic_gyr,
                gal_sfr_table=sfr_table,
                gal_lgmet_table=log_z_abs,
                gal_lgmet_scatter=lgmet_scatter,
                ssp_lgmet=self.ssp_data.ssp_lgmet,
                ssp_lg_age_gyr=self.ssp_data.ssp_lg_age_gyr,
                ssp_flux=self.ssp_data.ssp_flux,
                t_obs=t_obs_gyr,
            )
            # DSPS gives the intrinsic (no-dust) SED and 2D weights
            _dsps_intrinsic_sed = dsps_result.rest_sed  # (n_wave,) in Lsun/Hz
            weights = dsps_result.age_weights  # (n_age,) normalized
            # For dust: use Z-marginalized SSP flux per age
            lgmet_w = dsps_result.lgmet_weights  # (n_met, n_age)
            lgmet_w_safe = jnp.maximum(jnp.sum(lgmet_w, axis=0, keepdims=True), 1e-30)
            ssp_flux_at_z = jnp.einsum(
                "ma,maw->aw", lgmet_w / lgmet_w_safe, self.ssp_data.ssp_flux
            )
            # Scale weights to absolute mass (DSPS normalizes to 1)
            total_mass = jnp.sum(dsps_result.weights) * jnp.trapezoid(
                sfr_table, t_cosmic_gyr * 1e9
            )
            # Actually: DSPS age_weights are fractional. Total stellar mass =
            # integral(SFR * dt). Multiply weights by total mass.
            _total_mass_formed = jnp.trapezoid(sfr_table, t_cosmic_gyr * 1e9)
            weights = weights * _total_mass_formed

        elif _use_dsps_table:
            # Use DSPS calc_age_weights for proper trapezoidal integration
            # of SFH within each SSP age bin (Hearin+2023 Eq. 9).
            # Then our standard interpolate_metallicity + compute_csp_sed.
            # This gives ~1-2% accuracy at observable wavelengths in 0.5 ms.
            from dsps.sed.ssp_weights import calc_age_weights_from_sfh_table

            dsps_age_w = calc_age_weights_from_sfh_table(
                gal_t_table=t_cosmic_gyr,
                gal_sfr_table=sfr_table,
                ssp_lg_age_gyr=self.ssp_data.ssp_lg_age_gyr,
                t_obs=t_obs_gyr,
            )
            # DSPS returns normalized weights (sum=1); scale to absolute mass
            _total_mass_formed = jnp.trapezoid(sfr_table, t_cosmic_gyr * 1e9)
            weights = dsps_age_w * _total_mass_formed  # (n_age,) Msun

            # Metallicity: dispatch to linear or smooth interpolation
            log_z_solar = p.get("log_z", -1.8477)
            log_z_eff = effective_metallicity(log_z_solar, alpha_fe)
            ssp_flux_at_z = self._interp_metallicity(log_z_eff)

        # Metallicity interpolation (non-table path)
        # Priority: met_history (array) > evolving_metallicity > single Z
        if "met_history" in params and not _use_dsps_table:
            # Tabulated metallicity history Z(t) from simulation
            # Expects array of log10(Z/Zsun) at same time grid as sfh_t_gyr
            met_table = jnp.asarray(params["met_history"])
            t_cosmic_gyr = jnp.asarray(
                params.get("sfh_t_gyr", jnp.linspace(0.1, 13.7, len(met_table)))
            )
            z_val = p.get("redshift", 0.0)
            t_obs_gyr = self._t_universe_gyr(z_val) if hasattr(self, "_t_universe_gyr") else 13.7
            t_lookback_yr = jnp.maximum((t_obs_gyr - t_cosmic_gyr) * 1e9, 1.0)
            log_t_lookback = jnp.log10(t_lookback_yr)
            # Interpolate Z(t) onto SSP age grid
            log_z_on_ssp = jnp.interp(
                self.ssp_log_ages_yr,
                log_t_lookback[::-1],
                met_table[::-1],
            )
            log_z_abs = log_z_on_ssp + (-1.8477)  # solar offset
            log_z_abs = effective_metallicity(log_z_abs, alpha_fe)
            ssp_flux_at_z = self._interp_metallicity_evolving(log_z_abs)
        elif self._evolving_metallicity:
            z = p.get("redshift", 0.0)
            t_universe_gyr = self._t_universe_gyr(z)
            log_z_per_age = compute_log_z_evolving(
                self.ssp_data.ssp_lg_age_gyr,
                p["log_z_initial"],
                p["log_z_final"],
                t_universe_gyr,
            )
            log_z_per_age = effective_metallicity(log_z_per_age, alpha_fe)
            ssp_flux_at_z = self._interp_metallicity_evolving(log_z_per_age)
        else:
            log_z_eff = effective_metallicity(p["log_z"], alpha_fe)
            ssp_flux_at_z = self._interp_metallicity(log_z_eff)

            # --- Fast JIT path: dust + einsum in one compiled kernel ---
            # Eliminates ~78% Python dispatch overhead (4-14x speedup).
            if (
                _dsps_weights_2d is None
                and not self._evolving_metallicity
                and self._jit_exact_sed is not None
            ):
                sed_attenuated, sed_intrinsic_jit = self._jit_exact_sed(
                    weights,
                    ssp_flux_at_z,
                    p["tau_v1"],
                    p["tau_v2"],
                    n_slope=p.get("dust_n", -0.7),
                    dust_bump_strength=p.get("dust_bump_strength", 0.0),
                    dust_delta=p.get("dust_delta", 0.0),
                    dust_Rv=p.get("dust_Rv", 3.1),
                    f_obscuration=p.get("f_obscuration", 0.0),
                )
                sed_intrinsic = (
                    sed_intrinsic_jit
                    if (self._dust_emission_model is not None or need_intrinsic)
                    else None
                )
                # Skip the non-JIT dust/einsum below
                _dsps_weights_2d = "jit_done"

        # --- Fallback: non-JIT path for DSPS/evolving-Z/met-history ---
        if _dsps_weights_2d is not None and _dsps_weights_2d != "jit_done":
            dt = self._forward_dtype
            ssp_flux_at_z = ssp_flux_at_z.astype(dt)
            dust_age_w = self._dust_age_weights.astype(dt)
            wave_dt = self.ssp_data.ssp_wave.astype(dt)

            dust_atten = two_component_dust_fast(
                wave_dt,
                dust_age_w,
                tau_v1=p["tau_v1"],
                tau_v2=p["tau_v2"],
                law_bc=self._dust_law_bc,
                law_diff=self._dust_law_diff,
                f_obscuration=p.get("f_obscuration", 0.0),
                n_slope=p.get("dust_n", -0.7),
                dust_bump_strength=p.get("dust_bump_strength", 0.0),
                dust_delta=p.get("dust_delta", 0.0),
                dust_Rv=p.get("dust_Rv", 3.1),
            )

            _LSUN = 3.828e33
            sed_attenuated = (
                _LSUN
                * jnp.einsum(
                    "ma,aw,maw->w",
                    _dsps_weights_2d,
                    dust_atten,
                    self.ssp_data.ssp_flux.astype(dt),
                )
            ).astype(jnp.float64)

            sed_intrinsic = None
            if self._dust_emission_model is not None or need_intrinsic:
                sed_intrinsic = (
                    _LSUN
                    * jnp.einsum(
                        "ma,maw->w",
                        _dsps_weights_2d,
                        self.ssp_data.ssp_flux.astype(dt),
                    )
                ).astype(jnp.float64)

        elif _dsps_weights_2d is None:
            # Non-DSPS fallback (evolving Z or met_history without DSPS)
            dt = self._forward_dtype
            ssp_flux_at_z = ssp_flux_at_z.astype(dt)
            dust_age_w = self._dust_age_weights.astype(dt)
            wave_dt = self.ssp_data.ssp_wave.astype(dt)

            dust_atten = two_component_dust_fast(
                wave_dt,
                dust_age_w,
                tau_v1=p["tau_v1"],
                tau_v2=p["tau_v2"],
                law_bc=self._dust_law_bc,
                law_diff=self._dust_law_diff,
                f_obscuration=p.get("f_obscuration", 0.0),
                n_slope=p.get("dust_n", -0.7),
                dust_bump_strength=p.get("dust_bump_strength", 0.0),
                dust_delta=p.get("dust_delta", 0.0),
                dust_Rv=p.get("dust_Rv", 3.1),
            )

            sed_attenuated = compute_csp_sed(weights.astype(dt), ssp_flux_at_z, dust_atten).astype(
                jnp.float64
            )

            sed_intrinsic = None
            if self._dust_emission_model is not None or need_intrinsic:
                ones_atten = jnp.ones_like(dust_atten)
                sed_intrinsic = compute_csp_sed(
                    weights.astype(dt), ssp_flux_at_z, ones_atten
                ).astype(jnp.float64)

        sed = sed_attenuated

        # Nebular emission (if backend provides it)
        if self._nebular_backend is not None and self._nebular_backend.has_free_params:
            neb_sed = self._nebular_backend.predict_nebular_sed(
                ssp_weights=weights,
                ssp_wave=self.ssp_data.ssp_wave,
                ssp_log_ages_yr=self.ssp_log_ages_yr,
                log_z=p["log_z"],
                neb_logU=p.get("neb_logU", -3.0),
                neb_logZ_gas=p.get("neb_logZ_gas", None),
                neb_fesc=p.get("neb_fesc", 0.0),
                neb_fesc_lya=p.get("neb_fesc_lya", 0.0),
            )
            sed = sed + neb_sed

        # Dust IR emission (energy-balanced)
        if self._dust_emission_model is not None and sed_intrinsic is not None:
            from diffsed.models.dust.emission import get_emission_model

            _c_aa_em = 2.99792458e18  # c in Angstrom/s
            nu_em = _c_aa_em / self.ssp_data.ssp_wave
            L_absorbed = -jnp.trapezoid(sed_intrinsic - sed_attenuated, nu_em)
            eta_balance = p.get("dust_eta_balance", 1.0)
            L_ir = jnp.maximum(L_absorbed * eta_balance, 0.0)
            dust_ir = get_emission_model(self._dust_emission_model)(
                self.ssp_data.ssp_wave,
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

        # AGN contribution
        agn_bol_erg = 0.0
        if self._agn_model is not None:
            from diffsed.models.agn import get_agn_model

            if self._agn_parametric:
                agn_log_lbol = p.get("agn_log_lbol", 10.0)
                agn_frac_for_model = 1.0
                agn_bol_erg = 10.0**agn_log_lbol
            else:
                agn_frac_for_model = p.get("agn_frac", 0.0)
                _c_aa = 2.99792458e18
                nu = _c_aa / self.ssp_data.ssp_wave
                L_bol_stellar = -jnp.trapezoid(sed, nu)
                agn_log_lbol = jnp.log10(jnp.maximum(L_bol_stellar * agn_frac_for_model, 1e-50))
                agn_bol_erg = L_bol_stellar * agn_frac_for_model
            agn_sed = get_agn_model(self._agn_model)(
                self.ssp_data.ssp_wave,
                agn_log_lbol=agn_log_lbol,
                agn_frac=agn_frac_for_model,
                agn_alpha=p.get("agn_alpha", -1.0),
                agn_T_torus=p.get("agn_T_torus", 1000.0),
                agn_tau_torus=p.get("agn_tau_torus", 5.0),
                agn_torus_frac=p.get("agn_torus_frac", 0.5),
                agn_log_mbh=p.get("agn_log_mbh", 7.0),
                agn_log_ledd=p.get("agn_log_ledd", -1.0),
            )
            sed = sed + agn_sed

        # Radio emission (synchrotron from SF + AGN jets)
        if self._radio_enabled:
            from diffsed.models.radio import radio_total

            _L_ir = p.get("_L_ir_cached", 0.0)
            radio_sed = radio_total(
                self.ssp_data.ssp_wave,
                L_ir=_L_ir,
                L_agn_bol=agn_bol_erg,
                q_ir=p.get("radio_q_ir", 2.64),
                alpha_sf=p.get("radio_alpha_sf", 0.8),
                radio_loudness=p.get("radio_loudness", 0.0),
                alpha_agn=p.get("radio_alpha_agn", 0.7),
            )
            sed = sed + radio_sed

        # X-ray emission (XRBs + AGN corona)
        if self._xray_enabled:
            from diffsed.models.xray import xray_total

            _sfr = p.get("_sfr_cached", 1.0)
            _mstar = p.get("_mstar_cached", 1e10)
            xray_sed = xray_total(
                self.ssp_data.ssp_wave,
                sfr=_sfr,
                stellar_mass=_mstar,
                L_agn_bol=agn_bol_erg,
                gamma_agn=p.get("xray_gamma_agn", 1.8),
                alpha_ox=p.get("xray_alpha_ox", -1.4),
            )
            sed = sed + xray_sed

        return {
            "sed_total": sed,
            "sed_attenuated": sed_attenuated,
            "sed_intrinsic": sed_intrinsic,
            "ssp_flux_at_z": ssp_flux_at_z,
            "weights": weights,
            "sfr": sfr,
            "p": p,
            "agn_bol_erg": agn_bol_erg,
        }

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

        Returns a :class:`~diffsed.prediction.Prediction` object whose
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
        from diffsed.prediction import Prediction

        return Prediction(self, params)

    def predict_sfh_quantities(self, params):
        """Compute SFH-derived quantities (JIT-compatible).

        Returns a :class:`~diffsed.prediction.SFHQuantities` NamedTuple
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
        from diffsed.prediction import SFHQuantities
        from diffsed.utils.sed_quantities import (
            compute_mass_weighted_age,
            compute_mass_weighted_metallicity,
        )

        p = self._get_internal_params(params)
        sfr = self._compute_sfr(p)

        sfr_on_ssp = jnp.interp(self.ssp_log_ages_yr, self.log_age_grid, sfr)
        weights = compute_csp_weights(sfr_on_ssp, self.ssp_ages_yr)
        mass_formed = jnp.sum(weights)

        # Surviving mass
        if self.ssp_data.ssp_mass_remaining is not None:
            from diffsed.models.sps.dsps_wrapper import (
                compute_surviving_mass,
                interpolate_mass_remaining,
            )

            log_z = p.get("log_z", 0.0)
            mr_at_met = interpolate_mass_remaining(
                self.ssp_data.ssp_mass_remaining,
                self.ssp_data.ssp_lgmet,
                log_z,
            )
            mass_surviving = compute_surviving_mass(weights, mr_at_met)
        else:
            mass_surviving = jnp.array(jnp.nan)

        # SFR averages
        mask_100 = self.age_yr <= 1e8
        sfr_100myr = jnp.where(
            jnp.sum(mask_100) > 0,
            jnp.sum(sfr * mask_100) / jnp.maximum(jnp.sum(mask_100), 1.0),
            sfr[0],
        )
        mask_10 = self.age_yr <= 1e7
        sfr_10myr = jnp.where(
            jnp.sum(mask_10) > 0,
            jnp.sum(sfr * mask_10) / jnp.maximum(jnp.sum(mask_10), 1.0),
            sfr[0],
        )

        # sSFR
        mass_for_ssfr = jnp.where(jnp.isnan(mass_surviving), mass_formed, mass_surviving)
        ssfr = sfr_100myr / jnp.maximum(mass_for_ssfr, 1.0)

        # Mass-weighted age and metallicity
        mw_age = compute_mass_weighted_age(weights, self.ssp_ages_yr)
        mw_z = compute_mass_weighted_metallicity(
            weights,
            self.ssp_ages_yr,
            p.get("log_z", 0.0),
            log_z_initial=p.get("log_z_initial"),
            log_z_final=p.get("log_z_final"),
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

        Returns a :class:`~diffsed.prediction.SEDQuantities` NamedTuple
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
        from diffsed.prediction import SEDQuantities
        from diffsed.utils.sed_quantities import (
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
            p.get("log_z", 0.0),
            log_z_initial=p.get("log_z_initial"),
            log_z_final=p.get("log_z_final"),
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

        sed = self.predict_sed(params)
        z = self._get_redshift(params)
        dl_cm = self._get_dl_cm(params)

        # Apply IGM absorption (acts on observed-frame SED)
        # Always compute (cheap), but only apply when enabled
        if self._apply_igm:
            from diffsed.models.igm import igm_transmission

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
        kw = {
            "f_obscuration": p.get("f_obscuration", 0.0),
            "dust_bump_strength": p.get("dust_bump_strength", 0.0),
            "dust_delta": p.get("dust_delta", 0.0),
            "dust_Rv": p.get("dust_Rv", 3.1),
            "alpha_fe": p.get("alpha_fe", 0.0),
        }
        # Dust emission params (fused kernel handles MBB/dale2014 inline)
        if self._dust_emission_model in ("modified_blackbody", "dale2014"):
            kw["dust_T"] = p.get("dust_T", 35.0)
            kw["dust_beta_ir"] = p.get("dust_beta_ir", 1.6)
            kw["dust_eta_balance"] = p.get("dust_eta_balance", 1.0)
        return kw

    def _get_agn_kwargs(self, p):
        """Extract AGN kwargs from internal params dict for fused kernel."""
        if not (self._agn_model is not None and self._agn_parametric):
            return {}
        return {
            "agn_log_lbol": p.get("agn_log_lbol", 10.0),
            "agn_alpha": p.get("agn_alpha", -1.0),
            "agn_T_torus": p.get("agn_T_torus", 1000.0),
            "agn_tau_torus": p.get("agn_tau_torus", 5.0),
            "agn_torus_frac": p.get("agn_torus_frac", 0.5),
            "agn_log_mbh": p.get("agn_log_mbh", 7.0),
            "agn_log_ledd": p.get("agn_log_ledd", -1.0),
        }

    def _predict_photometry_fast(self, params):
        """Fast photometry using fused JIT kernel (fixed z)."""
        p = self._get_internal_params(params)
        sfr = self._compute_sfr(p)
        sfr_on_ssp = jnp.interp(self.ssp_log_ages_yr, self.log_age_grid, sfr)
        return self._fused_photometry(
            sfr_on_ssp,
            p["log_z"],
            p["tau_v1"],
            p["tau_v2"],
            p["dust_n"],
            **self._get_dust_kwargs(p),
            **self._get_agn_kwargs(p),
        )

    def _predict_photometry_ztable(self, params):
        """Fast photometry using z-table interpolation (free z)."""
        p = self._get_internal_params(params)
        sfr = self._compute_sfr(p)
        sfr_on_ssp = jnp.interp(self.ssp_log_ages_yr, self.log_age_grid, sfr)
        z = self._get_redshift(params)
        return self._fused_photometry_ztable(
            sfr_on_ssp,
            p["log_z"],
            p["tau_v1"],
            p["tau_v2"],
            p["dust_n"],
            z,
            **self._get_dust_kwargs(p),
            **self._get_agn_kwargs(p),
        )

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
        if self._fused_compatible():
            self._fused_spectrum = self._build_fused_spectrum()
        else:
            self._fused_spectrum = None
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
        self._fused_photometry_ztable = self._build_fused_photometry_ztable()
        return self

    def _build_fused_photometry_ztable(self):
        """Build fused JIT kernel with z-table interpolation.

        Like _build_fused_photometry but redshift is a free parameter.
        Supports all dust laws via captured law functions.
        """
        from diffsed.models.dust.attenuation import get_dust_law
        from diffsed.models.sps.dsps_wrapper import _ALPHA_TO_Z_COEFF as _A2Z, LSUN_ERG_PER_S

        fdt = self._forward_dtype
        zt = self._ztable
        ssp_phot_table = zt.ssp_phot_table.astype(fdt)
        eff_rest_table = zt.eff_waves_rest_table.astype(fdt)
        flux_scale_table = zt.flux_scale_table.astype(fdt)
        z_grid = zt.z_grid.astype(fdt)
        ssp_lgmet = self.ssp_data.ssp_lgmet.astype(fdt)
        dust_age_w = self._dust_age_weights.astype(fdt)
        ssp_ages_yr = self.ssp_ages_yr.astype(fdt)
        lsun = fdt.type(LSUN_ERG_PER_S)

        # IGM: precomputed on z-grid when apply_igm + approx["igm"]
        igm_trans_table = zt.igm_trans_table.astype(fdt)
        has_igm_ztable = bool(self._apply_igm and self._approx.get("igm", True))

        law_bc_fn = get_dust_law(self._dust_law_bc)
        law_diff_fn = get_dust_law(self._dust_law_diff)

        # Metallicity interpolation mode (ztable variant)
        _use_smooth_z_zt = self._met_interp == "smooth"
        _lgmet_scat_zt = fdt.type(self._lgmet_scatter)
        if _use_smooth_z_zt:
            from diffsed.models.sps.dsps_wrapper import compute_lgmet_weights as _clw_zt

        # AGN: capture model function for evaluation at effective wavelengths
        has_agn = self._agn_model is not None and self._agn_parametric
        if has_agn:
            from diffsed.models.agn import get_agn_model

            agn_model_fn = get_agn_model(self._agn_model)

        @jax.jit
        def fused_phot_ztable(
            sfr_on_ssp,
            log_z,
            tau_v1,
            tau_v2,
            dust_n,
            redshift,
            f_obscuration=0.0,
            dust_bump_strength=0.0,
            dust_delta=0.0,
            dust_Rv=3.1,
            alpha_fe=0.0,
            agn_log_lbol=10.0,
            agn_alpha=-1.0,
            agn_T_torus=1000.0,
            agn_tau_torus=5.0,
            agn_torus_frac=0.5,
            agn_log_mbh=7.0,
            agn_log_ledd=-1.0,
        ):
            sfr = sfr_on_ssp.astype(fdt)
            lz = jnp.asarray(log_z, dtype=fdt)
            tv1 = jnp.asarray(tau_v1, dtype=fdt)
            tv2 = jnp.asarray(tau_v2, dtype=fdt)
            dn = jnp.asarray(dust_n, dtype=fdt)
            z = jnp.asarray(redshift, dtype=fdt)
            f_obs = jnp.asarray(f_obscuration, dtype=fdt)
            bump = jnp.asarray(dust_bump_strength, dtype=fdt)
            delta = jnp.asarray(dust_delta, dtype=fdt)
            rv = jnp.asarray(dust_Rv, dtype=fdt)
            afe = jnp.asarray(alpha_fe, dtype=fdt)

            # Interpolate z-table to current redshift
            z_c = jnp.clip(z, z_grid[0], z_grid[-1])
            zi = jnp.clip(jnp.searchsorted(z_grid, z_c) - 1, 0, len(z_grid) - 2)
            zf = (z_c - z_grid[zi]) / (z_grid[zi + 1] - z_grid[zi])

            ssp_phot = (1.0 - zf) * ssp_phot_table[zi] + zf * ssp_phot_table[zi + 1]
            eff_rest = (1.0 - zf) * eff_rest_table[zi] + zf * eff_rest_table[zi + 1]
            flux_scale = (1.0 - zf) * flux_scale_table[zi] + zf * flux_scale_table[zi + 1]

            # CSP weights
            age_dt = jnp.concatenate(
                [
                    jnp.array([ssp_ages_yr[1] - ssp_ages_yr[0]]),
                    0.5 * (ssp_ages_yr[2:] - ssp_ages_yr[:-2]),
                    jnp.array([ssp_ages_yr[-1] - ssp_ages_yr[-2]]),
                ]
            )
            weights = sfr * age_dt

            # Alpha enhancement: shift effective metallicity
            lz = lz + _A2Z * afe

            # Metallicity interpolation (respects met_interp setting)
            if _use_smooth_z_zt:
                zw = _clw_zt(lz, ssp_lgmet, _lgmet_scat_zt)
                ssp_at_z = jnp.einsum("m,maf->af", zw, ssp_phot)
            else:
                log_z_c = jnp.clip(lz, ssp_lgmet[0], ssp_lgmet[-1])
                idx = jnp.clip(jnp.searchsorted(ssp_lgmet, log_z_c) - 1, 0, len(ssp_lgmet) - 2)
                frac = (log_z_c - ssp_lgmet[idx]) / (ssp_lgmet[idx + 1] - ssp_lgmet[idx])
                ssp_at_z = (1.0 - frac) * ssp_phot[idx] + frac * ssp_phot[idx + 1]

            # Dust: configurable curves at effective wavelengths
            k_bc = law_bc_fn(
                eff_rest,
                n_slope=dn,
                dust_bump_strength=bump,
                dust_delta=delta,
                dust_Rv=rv,
            )
            k_diff = law_diff_fn(
                eff_rest,
                n_slope=dn,
                dust_bump_strength=bump,
                dust_delta=delta,
                dust_Rv=rv,
            )
            tau = dust_age_w[:, None] * tv1 * k_bc[None, :] + tv2 * k_diff[None, :]
            dust = f_obs + (1.0 - f_obs) * jnp.exp(-tau)

            # Weighted sum
            flux_lsun = jnp.einsum("i,if,if->f", weights, dust, ssp_at_z)

            # AGN contribution at effective wavelengths (parametric mode)
            if has_agn:
                agn_lnu = agn_model_fn(
                    eff_rest,
                    agn_log_lbol=agn_log_lbol,
                    agn_frac=1.0,  # L_bol fully specified by agn_log_lbol
                    agn_alpha=agn_alpha,
                    agn_T_torus=agn_T_torus,
                    agn_tau_torus=agn_tau_torus,
                    agn_torus_frac=agn_torus_frac,
                    agn_log_mbh=agn_log_mbh,
                    agn_log_ledd=agn_log_ledd,
                )
                flux_lsun = flux_lsun + agn_lnu

            # IGM absorption (interpolated from precomputed z-table)
            if has_igm_ztable:
                igm_trans = (1.0 - zf) * igm_trans_table[zi] + zf * igm_trans_table[zi + 1]
                flux_lsun = flux_lsun * igm_trans

            return (flux_scale * flux_lsun * lsun).astype(jnp.float64)

        return fused_phot_ztable

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

        # Fast path: use fused kernel if available and compatible
        if self._spec_precomp is not None and self._fused_spectrum is not None:
            return self._predict_spectrum_fast(params)

        # Exact path
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
        flux = self._fused_spectrum(
            sfr_on_ssp,
            p["log_z"],
            p["tau_v1"],
            p["tau_v2"],
            p["dust_n"],
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
        """Generate mock photometric observation."""
        flux_true = self.predict_photometry(params)
        noise = flux_true / snr

        if key is not None:
            flux_obs = flux_true + noise * jax.random.normal(key, shape=flux_true.shape)
        else:
            flux_obs = flux_true

        return MockData(
            flux_true=flux_true,
            flux_obs=flux_obs,
            noise=noise,
            params=params,
        )

    def mock_spectrum(self, params, wave_obs, snr=30.0, key=None):
        """Generate mock spectroscopic observation.

        Parameters
        ----------
        params : dict
            Parameter values.
        wave_obs : array
            Observed wavelength grid (Angstrom).
        snr : float
            Signal-to-noise ratio per pixel.
        key : PRNGKey, optional
            Random key for noise. If None, returns noiseless.

        Returns
        -------
        MockData
            Mock spectroscopic observation.
        """
        flux_true = self.predict_spectrum(params, wave_obs)
        noise = jnp.abs(flux_true) / snr

        if key is not None:
            flux_obs = flux_true + noise * jax.random.normal(key, shape=flux_true.shape)
        else:
            flux_obs = flux_true

        return MockData(
            flux_true=flux_true,
            flux_obs=flux_obs,
            noise=noise,
            params=params,
        )

    def mock_batch(self, params_batch, snr=20.0, key=None):
        """Generate batch of mock observations."""
        first_key = next(iter(params_batch))
        n_batch = params_batch[first_key].shape[0]

        def _get_single(i):
            return {k: v[i] for k, v in params_batch.items()}

        if key is not None:
            noise_keys = jax.random.split(key, n_batch)
        else:
            noise_keys = [None] * n_batch

        results = [self.mock(_get_single(i), snr=snr, key=noise_keys[i]) for i in range(n_batch)]

        return MockData(
            flux_true=jnp.stack([r.flux_true for r in results]),
            flux_obs=jnp.stack([r.flux_obs for r in results]),
            noise=jnp.stack([r.noise for r in results]),
            params=params_batch,
        )

    # -------------------------------------------------------------------
    # Batch predictions (vmap over galaxies)
    # -------------------------------------------------------------------

    def predict_photometry_batch(self, params_batch):
        """Compute photometry for a batch of galaxies via jax.vmap.

        Parameters
        ----------
        params_batch : dict of arrays
            Each value has a leading batch dimension: shape (N, ...).
            E.g. ``{"sfh_dpl_alpha": array([1.0, 1.5, 2.0]), ...}``

        Returns
        -------
        array, shape (N, n_filters)
            Photometric flux for each galaxy.
        """
        return jax.vmap(self.predict_photometry)(params_batch)

    def predict_spectrum_batch(self, params_batch):
        """Compute spectra for a batch of galaxies via jax.vmap.

        Requires ``precompute_spectroscopy()`` to have been called.

        Parameters
        ----------
        params_batch : dict of arrays
            Each value has leading batch dimension.

        Returns
        -------
        array, shape (N, n_pix)
            Spectral flux for each galaxy.
        """
        return jax.vmap(self.predict_spectrum)(params_batch)

    # -------------------------------------------------------------------
    # Convenience fit
    # -------------------------------------------------------------------

    def fit(self, data, noise, method="map", data_type="photometry", **kwargs):
        """Fit observed data (convenience wrapper around Fitter)."""
        from diffsed.fitter import Fitter

        fitter = Fitter(self, data, noise, data_type=data_type)
        return fitter.run(method, **kwargs)
