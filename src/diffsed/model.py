"""High-level Model class wrapping the diffsed forward model.

Model provides a clean API for:
- Forward predictions (SED, photometry, spectrum, SFH, derived quantities)
- Mock galaxy generation (single and batch)
- Convenience fitting (delegates to Fitter)

The Model translates between the user-facing parameter names (sfh_alpha,
psd_sigma, etc.) and the internal names used by the low-level functions
(alpha, sigma_ps, etc.), handling unit conversions automatically.

Usage:
    from diffsed import Model, ParamSpec, Uniform, load_ssp_data, load_filter_set

    ssp = load_ssp_data("data/ssp.h5")
    filters = load_filter_set(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
    spec = ParamSpec(sfh_alpha=Uniform(0.5, 3.0), ..., redshift=0.1)
    model = Model(spec, ssp, filters=filters)

    params = spec.sample(jax.random.PRNGKey(0))
    photometry = model.predict_photometry(params)
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from diffsed.distributions import Fixed
from diffsed.param_spec import ParamSpec

from diffsed.models.sfh.mean_sfh import double_powerlaw
from diffsed.models.sfh.gp_sfh import gp_from_xi, compute_sqrt_power_drw
from diffsed.models.sfh.psd_models import drw_variance
from diffsed.models.dust.charlot_fall import charlot_fall
from diffsed.models.sps.dsps_wrapper import (
    compute_csp_weights,
    compute_csp_sed,
    interpolate_metallicity,
)
from diffsed.models.observation.photometry import compute_flux_density
from diffsed.models.observation.spectroscopy import compute_spectrum
from diffsed.utils.grid import (
    make_log_age_grid,
    log_age_to_age_yr,
    grid_spacing,
    interpolate_to_linear_time,
)
from diffsed.utils.cosmology import luminosity_distance
from diffsed.models.observation.photometry import ab_mag_from_flux
from diffsed.models.sps.precompute import (
    precompute_photometry,
    fast_photometry,
    interpolate_ssp_phot_metallicity,
)
from diffsed.models.dust.charlot_fall import charlot_fall_at_wavelengths


# ---------------------------------------------------------------------------
# Parameter name mapping: public → (internal, unit_scale, offset)
#
# Conversion: internal = public * scale + offset
# ---------------------------------------------------------------------------

# Solar metallicity: log10(Zsun) = log10(0.0142) ≈ -1.848 (Asplund 2009)
LOG10_ZSUN = -1.8477116556169435

PARAM_MAP = {
    "sfh_alpha":         ("alpha",    1.0, 0.0),
    "sfh_beta":          ("beta",     1.0, 0.0),
    "sfh_tau_peak_gyr":  ("tau_sfh",  1e9, 0.0),    # Gyr → yr
    "sfh_peak_sfr":      ("sfr_norm", 1.0, 0.0),
    "psd_sigma":         ("sigma_ps", 1.0, 0.0),
    "psd_tau_myr":       ("tau_ps",   1e6, 0.0),    # Myr → yr
    "met_logzsol":       ("log_z",    1.0, LOG10_ZSUN),  # log(Z/Zsun) → log(Z)
    "dust_tau_bc":       ("tau_v1",   1.0, 0.0),
    "dust_tau_diff":     ("tau_v2",   1.0, 0.0),
    "dust_slope":        ("dust_n",   1.0, 0.0),
    "redshift":          ("redshift", 1.0, 0.0),
}


# ---------------------------------------------------------------------------
# MockData container
# ---------------------------------------------------------------------------

class MockData(NamedTuple):
    """Container for mock galaxy observations."""
    flux_true: jnp.ndarray    # noiseless photometry (erg/s/cm²/Hz)
    flux_obs: jnp.ndarray     # noisy photometry
    noise: jnp.ndarray        # 1-sigma uncertainties
    params: dict               # input parameters


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------

class Model:
    """Differentiable forward model with clean parameter API.

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
    """

    def __init__(self, spec, ssp_data, filters=None, precompute=True):
        self.spec = spec
        self.ssp_data = ssp_data

        # Handle filter input formats
        self.filter_waves = None
        self.filter_trans = None
        if filters is not None:
            if isinstance(filters, tuple) and len(filters) == 3:
                # Output of load_filter_set(): (waves_list, trans_list, curves_list)
                self.filter_waves = filters[0]
                self.filter_trans = filters[1]
            elif isinstance(filters, (list, tuple)):
                # List of FilterCurve namedtuples
                self.filter_waves = [f.wave for f in filters]
                self.filter_trans = [f.trans for f in filters]
            else:
                raise TypeError(
                    f"filters must be a list of FilterCurve or output of "
                    f"load_filter_set(), got {type(filters)}"
                )

        # SSP grid info
        self.ssp_log_ages_yr = ssp_data.ssp_lg_age_gyr + 9.0
        self.ssp_ages_yr = 10.0 ** self.ssp_log_ages_yr

        # Log-age grid for SFH computation
        n_grid = spec.n_grid if spec.stochastic else 256
        self.log_age_grid = make_log_age_grid(n_grid)
        self.d_log_age = grid_spacing(self.log_age_grid)
        self.age_yr = log_age_to_age_yr(self.log_age_grid)
        self._n_grid = n_grid

        # Precompute luminosity distance if redshift is fixed
        redshift_dist = spec.get_distribution("redshift")
        if redshift_dist.is_fixed:
            self._dl_cm_fixed = luminosity_distance(redshift_dist.bounds[0])
            self._z_fixed = redshift_dist.bounds[0]
        else:
            self._dl_cm_fixed = None
            self._z_fixed = None

        # Photometry precomputation (Zacharegkas+2025 Section 3)
        # When redshift is fixed and filters are present, precompute
        # SSP broadband fluxes to eliminate wavelength integrals from
        # the inference loop. Gives 30-50x speedup.
        self._precomp = None
        if (precompute is True and self._z_fixed is not None
                and self.filter_waves is not None):
            self._precomp = precompute_photometry(
                ssp_data, self.filter_waves, self.filter_trans,
                self._z_fixed, self._dl_cm_fixed,
            )

    # -------------------------------------------------------------------
    # Internal parameter translation
    # -------------------------------------------------------------------

    def _get_internal_params(self, params):
        """Translate public param dict to internal names with unit conversion.

        Merges user-provided values with fixed defaults from the spec.
        Conversion: internal = public * scale + offset
        """
        internal = {}
        for pub_name, (int_name, scale, offset) in PARAM_MAP.items():
            if pub_name in params:
                internal[int_name] = params[pub_name] * scale + offset
            else:
                # Fall back to fixed value from spec
                dist = self.spec.get_distribution(pub_name)
                if dist.is_fixed:
                    internal[int_name] = dist.bounds[0] * scale + offset
                else:
                    raise KeyError(
                        f"Free parameter '{pub_name}' not found in params dict"
                    )

        # Handle psd_xi
        if self.spec.stochastic and "psd_xi" in params:
            internal["xi"] = params["psd_xi"]

        return internal

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

        if self.spec.stochastic:
            # GP stochastic path
            sqrt_power = compute_sqrt_power_drw(
                self._n_grid, float(self.d_log_age),
                p["sigma_ps"], p["tau_ps"]
            )
            gp_x = gp_from_xi(p["xi"], sqrt_power, self._n_grid)
            k0_half = drw_variance(p["sigma_ps"]) / 2.0

            sfr_mean = double_powerlaw(
                self.age_yr,
                alpha=p["alpha"], beta=p["beta"],
                tau=p["tau_sfh"], norm=p["sfr_norm"],
            )
            sfr = sfr_mean * jnp.exp(gp_x - k0_half)
        else:
            # Pure parametric path — no GP
            sfr = double_powerlaw(
                self.age_yr,
                alpha=p["alpha"], beta=p["beta"],
                tau=p["tau_sfh"], norm=p["sfr_norm"],
            )

        # Interpolate SFR to SSP age grid
        sfr_on_ssp = jnp.interp(self.ssp_log_ages_yr, self.log_age_grid, sfr)
        weights = compute_csp_weights(sfr_on_ssp, self.ssp_ages_yr)

        # Metallicity interpolation
        ssp_flux_at_z = interpolate_metallicity(
            self.ssp_data.ssp_flux,
            self.ssp_data.ssp_lgmet,
            p["log_z"],
        )

        # Dust attenuation
        dust_atten = charlot_fall(
            self.ssp_data.ssp_wave,
            self.ssp_ages_yr,
            tau_v1=p["tau_v1"],
            tau_v2=p["tau_v2"],
            n_slope=p["dust_n"],
        )

        # Compose CSP SED
        return compute_csp_sed(weights, ssp_flux_at_z, dust_atten)

    def predict_photometry(self, params):
        """Compute observed photometric flux densities.

        Automatically uses the fast precomputed path (Zacharegkas+2025)
        when available (redshift fixed + filters present + precompute=True).
        Falls back to exact wavelength integration otherwise.

        Parameters
        ----------
        params : dict
            Parameter values (public names). Must include 'redshift'
            if redshift is free.

        Returns
        -------
        array, shape (n_filters,)
            Observed flux densities in erg/s/cm^2/Hz.
        """
        if self.filter_waves is None:
            raise ValueError("No filters set. Pass filters to Model().")

        # Fast path: precomputed SSP photometry (no wavelength integrals)
        if self._precomp is not None:
            return self._predict_photometry_fast(params)

        # Exact path: full wavelength integration per filter
        sed = self.predict_sed(params)
        z = self._get_redshift(params)
        dl_cm = self._get_dl_cm(params)

        fluxes = []
        for fw, ft in zip(self.filter_waves, self.filter_trans):
            f = compute_flux_density(
                sed, self.ssp_data.ssp_wave, fw, ft, z, dl_cm,
            )
            fluxes.append(f)
        return jnp.array(fluxes)

    def _predict_photometry_fast(self, params):
        """Fast photometry using precomputed SSP broadband fluxes.

        Instead of integrating SED × filter × dust over wavelength,
        evaluates dust at filter effective wavelengths and uses a
        simple weighted sum. ~30-50x faster than exact computation.

        See Zacharegkas, Hearin & Benson (2025) Section 3, Eq. 6-7.
        """
        p = self._get_internal_params(params)
        precomp = self._precomp

        # 1. Compute SFH weights (same as predict_sed)
        if self.spec.stochastic and "xi" in p:
            sqrt_power = compute_sqrt_power_drw(
                self._n_grid, float(self.d_log_age),
                p["sigma_ps"], p["tau_ps"]
            )
            gp_x = gp_from_xi(p["xi"], sqrt_power, self._n_grid)
            k0_half = drw_variance(p["sigma_ps"]) / 2.0
            sfr_mean = double_powerlaw(
                self.age_yr,
                alpha=p["alpha"], beta=p["beta"],
                tau=p["tau_sfh"], norm=p["sfr_norm"],
            )
            sfr = sfr_mean * jnp.exp(gp_x - k0_half)
        else:
            sfr = double_powerlaw(
                self.age_yr,
                alpha=p["alpha"], beta=p["beta"],
                tau=p["tau_sfh"], norm=p["sfr_norm"],
            )

        sfr_on_ssp = jnp.interp(self.ssp_log_ages_yr, self.log_age_grid, sfr)
        weights = compute_csp_weights(sfr_on_ssp, self.ssp_ages_yr)

        # 2. Interpolate precomputed SSP photometry to target metallicity
        ssp_phot_at_z = interpolate_ssp_phot_metallicity(
            precomp.ssp_phot, self.ssp_data.ssp_lgmet, p["log_z"]
        )

        # 3. Evaluate dust at filter effective wavelengths (rest-frame)
        dust_at_eff = charlot_fall_at_wavelengths(
            precomp.effective_wavelengths_rest,
            self.ssp_ages_yr,
            tau_v1=p["tau_v1"], tau_v2=p["tau_v2"], n_slope=p["dust_n"],
        )

        # 4. Fast photometry: weighted sum (no wavelength integrals)
        return fast_photometry(weights, ssp_phot_at_z, dust_at_eff,
                               precomp.flux_scale)

    def predict_spectrum(self, params, wave_obs):
        """Compute observed spectrum at given wavelengths.

        Parameters
        ----------
        params : dict
            Parameter values.
        wave_obs : array
            Observed wavelength grid (Angstrom).

        Returns
        -------
        array, shape (n_pix,)
            Spectral flux density in erg/s/cm^2/Hz.
        """
        sed = self.predict_sed(params)
        z = self._get_redshift(params)
        dl_cm = self._get_dl_cm(params)
        return compute_spectrum(
            sed, self.ssp_data.ssp_wave, wave_obs, z, dl_cm,
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

        sfr_mean = double_powerlaw(
            self.age_yr,
            alpha=p["alpha"], beta=p["beta"],
            tau=p["tau_sfh"], norm=p["sfr_norm"],
        )

        if self.spec.stochastic and "xi" in p:
            sqrt_power = compute_sqrt_power_drw(
                self._n_grid, float(self.d_log_age),
                p["sigma_ps"], p["tau_ps"]
            )
            gp_x = gp_from_xi(p["xi"], sqrt_power, self._n_grid)
            k0_half = drw_variance(p["sigma_ps"]) / 2.0
            sfr_full = sfr_mean * jnp.exp(gp_x - k0_half)
        else:
            sfr_full = sfr_mean

        t_gyr_mean, sfr_mean_lin = interpolate_to_linear_time(
            self.log_age_grid, sfr_mean, n_linear
        )
        _, sfr_full_lin = interpolate_to_linear_time(
            self.log_age_grid, sfr_full, n_linear
        )

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

        sfr_mean = double_powerlaw(
            self.age_yr,
            alpha=p["alpha"], beta=p["beta"],
            tau=p["tau_sfh"], norm=p["sfr_norm"],
        )

        if self.spec.stochastic and "xi" in p:
            sqrt_power = compute_sqrt_power_drw(
                self._n_grid, float(self.d_log_age),
                p["sigma_ps"], p["tau_ps"]
            )
            gp_x = gp_from_xi(p["xi"], sqrt_power, self._n_grid)
            k0_half = drw_variance(p["sigma_ps"]) / 2.0
            sfr = sfr_mean * jnp.exp(gp_x - k0_half)
        else:
            sfr = sfr_mean

        # Interpolate to SSP grid and compute mass
        sfr_on_ssp = jnp.interp(self.ssp_log_ages_yr, self.log_age_grid, sfr)
        weights = compute_csp_weights(sfr_on_ssp, self.ssp_ages_yr)
        stellar_mass = jnp.sum(weights)

        # Average SFR over recent time windows
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
        """Compute observed AB magnitudes through all filters.

        Uses DSPS's calc_obs_mag for cosmologically correct magnitudes.
        Falls back to our own computation if DSPS is unavailable.

        Parameters
        ----------
        params : dict
            Parameter values.

        Returns
        -------
        array, shape (n_filters,)
            Observed AB magnitudes.
        """
        if self.filter_waves is None:
            raise ValueError("No filters set.")

        try:
            from dsps import calc_obs_mag
            from dsps.cosmology import DEFAULT_COSMOLOGY

            # DSPS expects rest-frame SED in Lsun/Hz
            sed_lsun = self.predict_luminosity(params)
            z = self._get_redshift(params)
            cosmo = DEFAULT_COSMOLOGY

            mags = []
            for fw, ft in zip(self.filter_waves, self.filter_trans):
                m = calc_obs_mag(
                    self.ssp_data.ssp_wave, sed_lsun, fw, ft,
                    z, cosmo.Om0, cosmo.w0, cosmo.wa, cosmo.h,
                )
                mags.append(m)
            return jnp.array(mags)

        except ImportError:
            # Fallback: use our own flux → AB mag conversion
            flux = self.predict_photometry(params)
            return ab_mag_from_flux(flux)

    def predict_luminosity(self, params):
        """Compute rest-frame luminosity SED in solar units.

        Parameters
        ----------
        params : dict
            Parameter values.

        Returns
        -------
        array, shape (n_wave,)
            Rest-frame luminosity in Lsun/Hz.
            This is the standard unit used by DSPS/FSPS.
        """
        LSUN_CGS = 3.828e33  # erg/s (IAU 2015)
        sed_erg = self.predict_sed(params)  # erg/s/Hz
        return sed_erg / LSUN_CGS

    def plot_sfh_posterior(self, posterior, true_params=None, ax=None,
                          n_draws=50, color="C0", label="Posterior"):
        """Plot posterior SFH with percentile fill and sample lines.

        Parameters
        ----------
        posterior : Posterior
            Inference result with samples.
        true_params : dict, optional
            True parameter values (for truth overlay).
        ax : matplotlib Axes, optional
            Axes to plot on.
        n_draws : int
            Number of sample lines to draw.
        color : str
            Color for posterior.
        label : str
            Label for the posterior mean line.

        Returns
        -------
        ax : matplotlib Axes
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 5))

        if posterior.samples is None:
            # MAP: just plot the point estimate
            sfh = self.predict_sfh(posterior.params)
            ax.plot(sfh["t_gyr"], sfh["sfr_mean"], color=color, lw=2, label=label)
        else:
            # Compute SFH for all samples
            n_total = len(next(iter(posterior.samples.values())))
            sfh_draws = []
            for i in range(n_total):
                s_i = {k: posterior.samples[k][i] for k in posterior.samples}
                sfh_i = self.predict_sfh(s_i)
                key = "sfr_full" if self.spec.stochastic else "sfr_mean"
                sfh_draws.append(sfh_i[key])

            import numpy as np
            sfh_arr = np.array(sfh_draws)  # (n_samples, n_linear)
            t_gyr = np.array(self.predict_sfh(posterior.params)["t_gyr"])

            # Percentile fill (16-84%)
            lo = np.percentile(sfh_arr, 16, axis=0)
            hi = np.percentile(sfh_arr, 84, axis=0)
            ax.fill_between(t_gyr, lo, hi, color=color, alpha=0.2)

            # Faint sample lines
            n_show = min(n_draws, n_total)
            indices = np.linspace(0, n_total - 1, n_show, dtype=int)
            for idx in indices:
                ax.plot(t_gyr, sfh_arr[idx], color=color, alpha=0.1, lw=0.4)

            # Posterior mean
            sfh_mean = self.predict_sfh(posterior.params)
            key = "sfr_full" if self.spec.stochastic else "sfr_mean"
            ax.plot(t_gyr, sfh_mean[key], color=color, lw=2, label=label)

        # Truth overlay
        if true_params is not None:
            sfh_true = self.predict_sfh(true_params)
            key = "sfr_full" if self.spec.stochastic else "sfr_mean"
            ax.plot(sfh_true["t_gyr"], sfh_true[key], "k-", lw=2.5,
                    label="Truth", zorder=10)
            if self.spec.stochastic:
                ax.plot(sfh_true["t_gyr"], sfh_true["sfr_mean"], "k--",
                        lw=1, alpha=0.3)

        ax.set_xlabel("Lookback time (Gyr)")
        ax.set_ylabel(r"SFR (M$_{\odot}$/yr)")
        ax.set_xlim(0, 13.5)
        ax.legend(fontsize=9)

        return ax

    # -------------------------------------------------------------------
    # Mock generation
    # -------------------------------------------------------------------

    def mock(self, params, snr=20.0, key=None):
        """Generate mock photometric observation.

        Parameters
        ----------
        params : dict
            Parameter values.
        snr : float
            Signal-to-noise ratio per band.
        key : PRNGKey, optional
            For noise. If None, return noiseless.

        Returns
        -------
        MockData
            flux_true, flux_obs, noise, params.
        """
        flux_true = self.predict_photometry(params)
        noise = flux_true / snr

        if key is not None:
            flux_obs = flux_true + noise * jax.random.normal(
                key, shape=flux_true.shape
            )
        else:
            flux_obs = flux_true

        return MockData(
            flux_true=flux_true,
            flux_obs=flux_obs,
            noise=noise,
            params=params,
        )

    def mock_batch(self, params_batch, snr=20.0, key=None):
        """Generate batch of mock observations.

        Parameters
        ----------
        params_batch : dict
            Dict of arrays, each with leading batch dimension.
        snr : float
            Signal-to-noise ratio per band.
        key : PRNGKey, optional
            For noise.

        Returns
        -------
        MockData
            Each field has leading batch dimension.
        """
        # Determine batch size from first array
        first_key = next(iter(params_batch))
        n_batch = params_batch[first_key].shape[0]

        def _single_mock(params_i, noise_key):
            return self.mock(params_i, snr=snr, key=noise_key)

        # Unbatch params: dict of (n,) arrays → n dicts of scalars
        def _get_single(i):
            return {k: v[i] for k, v in params_batch.items()}

        if key is not None:
            noise_keys = jax.random.split(key, n_batch)
        else:
            noise_keys = [None] * n_batch

        # Use Python loop (vmap would require all-JAX mock, which we can add later)
        results = [_single_mock(_get_single(i), noise_keys[i]) for i in range(n_batch)]

        return MockData(
            flux_true=jnp.stack([r.flux_true for r in results]),
            flux_obs=jnp.stack([r.flux_obs for r in results]),
            noise=jnp.stack([r.noise for r in results]),
            params=params_batch,
        )

    # -------------------------------------------------------------------
    # Convenience fit
    # -------------------------------------------------------------------

    def fit(self, data, noise, method="map", data_type="photometry", **kwargs):
        """Fit observed data (convenience wrapper around Fitter).

        Parameters
        ----------
        data : array
            Observed data.
        noise : array
            1-sigma uncertainties.
        method : str
            "map", "nuts", or "geovi".
        data_type : str
            "photometry", "spectroscopy", or "joint".
        **kwargs
            Passed to Fitter.run().

        Returns
        -------
        Posterior
            Inference results.
        """
        from diffsed.fitter import Fitter
        fitter = Fitter(self, data, noise, data_type=data_type)
        return fitter.run(method, **kwargs)
