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

from typing import NamedTuple

import jax
import jax.numpy as jnp

from diffsed.models.dust.charlot_fall import charlot_fall  # backward compat
from diffsed.models.dust.two_component_dust import (
    precompute_dust_age_weights,
    two_component_dust,
)
from diffsed.models.observation.photometry import ab_mag_from_flux, compute_flux_density
from diffsed.models.observation.spectroscopy import compute_spectrum
from diffsed.models.sfh.registry import compute_field_gp, resolve_sfh
from diffsed.models.sps.dsps_wrapper import (
    compute_csp_sed,
    compute_csp_weights,
    interpolate_metallicity,
)
from diffsed.models.sps.precompute import (
    precompute_photometry,
    precompute_photometry_ztable,
    precompute_spectroscopy,
)
from diffsed.utils.cosmology import luminosity_distance
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

_NON_SFH_PARAM_MAP = {
    "met_logzsol": ("log_z", 1.0, LOG10_ZSUN),  # log(Z/Zsun) → log(Z)
    "dust_tau_bc": ("tau_v1", 1.0, 0.0),
    "dust_tau_diff": ("tau_v2", 1.0, 0.0),
    "dust_slope": ("dust_n", 1.0, 0.0),
    "redshift": ("redshift", 1.0, 0.0),
    "noise_frac_cal": ("noise_frac_cal", 1.0, 0.0),
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
        Dtype for forward model computation. "float32" halves memory
        and gives ~1.5x speedup with <0.1% accuracy loss. Default
        "float64" preserves full precision. Only affects the fused
        JIT kernels; cosmological distances always use float64.
    """

    def __init__(self, spec, ssp_data, filters=None, precompute=True, forward_dtype="float64"):
        self.spec = spec
        self.ssp_data = ssp_data
        self._forward_dtype = jnp.dtype(forward_dtype)

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

        # IGM absorption (Inoue+2014)
        self._apply_igm = spec.apply_igm

        # Nebular emission backend
        self._nebular_backend = None
        if spec.nebular and spec.cloudy_grid_path is not None:
            from diffsed.models.nebular import CloudyGridBackend
            self._nebular_backend = CloudyGridBackend(spec.cloudy_grid_path, ssp_data)
        elif spec.nebular:
            from diffsed.models.nebular import BakedInBackend
            self._nebular_backend = BakedInBackend()
        else:
            from diffsed.models.nebular import BakedInBackend
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

        # Build fused JIT kernels for fast photometry/spectroscopy
        self._fused_photometry = None
        if self._precomp is not None:
            self._fused_photometry = self._build_fused_photometry()

        # Spectroscopy precomputation (same idea: pre-interpolate SSPs)
        self._spec_precomp = None

        # Z-table precomputation (for free-redshift fitting)
        self._ztable = None
        self._fused_photometry_ztable = None

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

        If forward_dtype is float32, all closure arrays are cast to
        float32 for ~1.5x speed and 2x memory savings. The output
        is cast back to float64 for numerical stability in the likelihood.
        """
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

        @jax.jit
        def fused_phot(sfr_on_ssp, log_z, tau_v1, tau_v2, dust_n):
            # Cast inputs to forward dtype
            sfr = sfr_on_ssp.astype(dt)
            lz = jnp.asarray(log_z, dtype=dt)
            tv1 = jnp.asarray(tau_v1, dtype=dt)
            tv2 = jnp.asarray(tau_v2, dtype=dt)
            dn = jnp.asarray(dust_n, dtype=dt)

            # CSP weights
            age_dt = jnp.concatenate(
                [
                    jnp.array([ssp_ages_yr[1] - ssp_ages_yr[0]]),
                    0.5 * (ssp_ages_yr[2:] - ssp_ages_yr[:-2]),
                    jnp.array([ssp_ages_yr[-1] - ssp_ages_yr[-2]]),
                ]
            )
            weights = sfr * age_dt

            # Metallicity interpolation
            log_z_c = jnp.clip(lz, ssp_lgmet[0], ssp_lgmet[-1])
            idx = jnp.clip(jnp.searchsorted(ssp_lgmet, log_z_c) - 1, 0, len(ssp_lgmet) - 2)
            frac = (log_z_c - ssp_lgmet[idx]) / (ssp_lgmet[idx + 1] - ssp_lgmet[idx])
            ssp_at_z = (1.0 - frac) * ssp_phot[idx] + frac * ssp_phot[idx + 1]

            # Dust at effective wavelengths
            wave_ratio = (eff_waves_rest / 5500.0) ** dn
            tau_v_eff = dust_age_w * tv1 + tv2
            dust = jnp.exp(-(tau_v_eff[:, None] * wave_ratio[None, :]))

            # Weighted sum — output in float64 for likelihood stability
            flux_lsun = jnp.einsum("i,if,if->f", weights, dust, ssp_at_z)
            return (flux_scale * flux_lsun * lsun).astype(jnp.float64)

        return fused_phot

    def _build_fused_spectrum(self):
        """Build a single JIT function: SFR-on-SSP → spectrum.

        Same fusion approach as photometry but for spectroscopic pixels.
        Uses forward_dtype for mixed-precision support. Includes velocity
        dispersion broadening only if sigma_v is in the ParamSpec.
        """
        from diffsed.models.sps.dsps_wrapper import LSUN_ERG_PER_S

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

        # Precompute FFT frequencies for velocity broadening (only if needed)
        if has_sigma_v:
            fft_freq = jnp.fft.rfftfreq(n_pix).astype(fdt)
            dlnwave = jnp.log(wave_obs_pixels[1] / wave_obs_pixels[0]).astype(fdt)
            c_km_s = fdt.type(299792.458)

        @jax.jit
        def fused_spec(sfr_on_ssp, log_z, tau_v1, tau_v2, dust_n, sigma_v=0.0):
            sfr = sfr_on_ssp.astype(fdt)
            lz = jnp.asarray(log_z, dtype=fdt)
            tv1 = jnp.asarray(tau_v1, dtype=fdt)
            tv2 = jnp.asarray(tau_v2, dtype=fdt)
            dn = jnp.asarray(dust_n, dtype=fdt)

            # CSP weights
            age_dt = jnp.concatenate(
                [
                    jnp.array([ssp_ages_yr[1] - ssp_ages_yr[0]]),
                    0.5 * (ssp_ages_yr[2:] - ssp_ages_yr[:-2]),
                    jnp.array([ssp_ages_yr[-1] - ssp_ages_yr[-2]]),
                ]
            )
            weights = sfr * age_dt

            # Metallicity interpolation
            log_z_c = jnp.clip(lz, ssp_lgmet[0], ssp_lgmet[-1])
            idx = jnp.clip(jnp.searchsorted(ssp_lgmet, log_z_c) - 1, 0, len(ssp_lgmet) - 2)
            frac = (log_z_c - ssp_lgmet[idx]) / (ssp_lgmet[idx + 1] - ssp_lgmet[idx])
            ssp_at_z = (1.0 - frac) * ssp_on_pixels[idx] + frac * ssp_on_pixels[idx + 1]

            # Dust at pixel wavelengths
            wave_ratio = (wave_rest_pixels / 5500.0) ** dn
            tau_v_eff = dust_age_w * tv1 + tv2
            dust = jnp.exp(-(tau_v_eff[:, None] * wave_ratio[None, :]))

            # Weighted sum
            flux = jnp.einsum("i,ip,ip->p", weights, dust, ssp_at_z)
            flux = flux_scale * flux * lsun

            if has_sigma_v:
                # Velocity dispersion broadening (FFT convolution)
                sv = jnp.asarray(sigma_v, dtype=fdt)
                sigma_pix = (sv / c_km_s) / dlnwave
                kernel_ft = jnp.exp(-2.0 * jnp.pi**2 * sigma_pix**2 * fft_freq**2)
                flux = jnp.fft.irfft(jnp.fft.rfft(flux) * kernel_ft, n=n_pix)

            return flux.astype(jnp.float64)

        return fused_spec

    # -------------------------------------------------------------------
    # Forward predictions
    # -------------------------------------------------------------------

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
        p = self._get_internal_params(params)
        sfr = self._compute_sfr(p)

        # Interpolate SFR to SSP age grid
        sfr_on_ssp = jnp.interp(self.ssp_log_ages_yr, self.log_age_grid, sfr)
        weights = compute_csp_weights(sfr_on_ssp, self.ssp_ages_yr)

        # Metallicity interpolation
        ssp_flux_at_z = interpolate_metallicity(
            self.ssp_data.ssp_flux,
            self.ssp_data.ssp_lgmet,
            p["log_z"],
        )

        # Dust attenuation (generalized two-component model)
        dust_atten = two_component_dust(
            self.ssp_data.ssp_wave,
            self.ssp_ages_yr,
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

        sed = compute_csp_sed(weights, ssp_flux_at_z, dust_atten)

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
            )
            sed = sed + neb_sed

        return sed

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

        if self._precomp is not None:
            return self._predict_photometry_fast(params)

        if self._ztable is not None:
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

    def _predict_photometry_fast(self, params):
        """Fast photometry using fused JIT kernel (fixed z)."""
        p = self._get_internal_params(params)
        sfr = self._compute_sfr(p)
        sfr_on_ssp = jnp.interp(self.ssp_log_ages_yr, self.log_age_grid, sfr)
        return self._fused_photometry(
            sfr_on_ssp, p["log_z"], p["tau_v1"], p["tau_v2"], p["dust_n"]
        )

    def _predict_photometry_ztable(self, params):
        """Fast photometry using z-table interpolation (free z)."""
        p = self._get_internal_params(params)
        sfr = self._compute_sfr(p)
        sfr_on_ssp = jnp.interp(self.ssp_log_ages_yr, self.log_age_grid, sfr)
        z = self._get_redshift(params)
        return self._fused_photometry_ztable(
            sfr_on_ssp, p["log_z"], p["tau_v1"], p["tau_v2"], p["dust_n"], z
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
        self._fused_spectrum = self._build_fused_spectrum()
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
        )
        self._fused_photometry_ztable = self._build_fused_photometry_ztable()
        return self

    def _build_fused_photometry_ztable(self):
        """Build fused JIT kernel that interpolates the z-table at inference.

        Like _build_fused_photometry but redshift is a free parameter:
        interpolates precomputed SSP broadband fluxes, effective wavelengths,
        and flux scale from the z-table.
        """
        from diffsed.models.sps.dsps_wrapper import LSUN_ERG_PER_S

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

        @jax.jit
        def fused_phot_ztable(sfr_on_ssp, log_z, tau_v1, tau_v2, dust_n, redshift):
            sfr = sfr_on_ssp.astype(fdt)
            lz = jnp.asarray(log_z, dtype=fdt)
            tv1 = jnp.asarray(tau_v1, dtype=fdt)
            tv2 = jnp.asarray(tau_v2, dtype=fdt)
            dn = jnp.asarray(dust_n, dtype=fdt)
            z = jnp.asarray(redshift, dtype=fdt)

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

            # Metallicity interpolation
            log_z_c = jnp.clip(lz, ssp_lgmet[0], ssp_lgmet[-1])
            idx = jnp.clip(jnp.searchsorted(ssp_lgmet, log_z_c) - 1, 0, len(ssp_lgmet) - 2)
            frac = (log_z_c - ssp_lgmet[idx]) / (ssp_lgmet[idx + 1] - ssp_lgmet[idx])
            ssp_at_z = (1.0 - frac) * ssp_phot[idx] + frac * ssp_phot[idx + 1]

            # Dust at effective wavelengths (interpolated from z-table)
            wave_ratio = (eff_rest / 5500.0) ** dn
            tau_v_eff = dust_age_w * tv1 + tv2
            dust = jnp.exp(-(tau_v_eff[:, None] * wave_ratio[None, :]))

            # Weighted sum
            flux_lsun = jnp.einsum("i,if,if->f", weights, dust, ssp_at_z)
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

        # Fast path: use precomputed SSPs if wavelength grid matches
        if self._spec_precomp is not None:
            return self._predict_spectrum_fast(params)

        # Exact path
        sed = self.predict_sed(params)
        z = self._get_redshift(params)
        dl_cm = self._get_dl_cm(params)
        return compute_spectrum(
            sed,
            self.ssp_data.ssp_wave,
            wave_obs,
            z,
            dl_cm,
        )

    def _predict_spectrum_fast(self, params):
        """Fast spectrum using fused JIT kernel."""
        p = self._get_internal_params(params)
        sfr = self._compute_sfr(p)
        sfr_on_ssp = jnp.interp(self.ssp_log_ages_yr, self.log_age_grid, sfr)
        # Only pass sigma_v if it's a model parameter
        sigma_v = params.get("sigma_v", 0.0) if self._has_sigma_v else 0.0
        return self._fused_spectrum(
            sfr_on_ssp, p["log_z"], p["tau_v1"], p["tau_v2"], p["dust_n"], sigma_v
        )

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

        Parameters
        ----------
        params : dict
            Parameter values.

        Returns
        -------
        dict with keys:
            "stellar_mass": total mass formed (Msun)
            "sfr_100myr": SFR averaged over last 100 Myr (Msun/yr)
            "sfr_10myr": SFR averaged over last 10 Myr (Msun/yr)
            "ssfr": specific SFR (yr^-1)
        """
        p = self._get_internal_params(params)
        sfr = self._compute_sfr(p)

        sfr_on_ssp = jnp.interp(self.ssp_log_ages_yr, self.log_age_grid, sfr)
        weights = compute_csp_weights(sfr_on_ssp, self.ssp_ages_yr)
        stellar_mass = jnp.sum(weights)

        mask_100myr = self.age_yr <= 1e8
        sfr_100myr = jnp.where(
            jnp.sum(mask_100myr) > 0,
            jnp.sum(sfr * mask_100myr) / jnp.maximum(jnp.sum(mask_100myr), 1.0),
            sfr[0],
        )
        mask_10myr = self.age_yr <= 1e7
        sfr_10myr = jnp.where(
            jnp.sum(mask_10myr) > 0,
            jnp.sum(sfr * mask_10myr) / jnp.maximum(jnp.sum(mask_10myr), 1.0),
            sfr[0],
        )

        ssfr = sfr_100myr / jnp.maximum(stellar_mass, 1.0)

        return {
            "stellar_mass": stellar_mass,
            "sfr_100myr": sfr_100myr,
            "sfr_10myr": sfr_10myr,
            "ssfr": ssfr,
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
