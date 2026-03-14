"""Complete differentiable forward model: parameters -> predicted SED.

This module implements the full pipeline from latent parameters to
observable photometry/spectroscopy. The model is a pure JAX function
composing: PSD -> GP -> SFH -> DSPS CSP -> dust -> photometry.

The forward model can be used:
- Standalone: generate mock galaxies via generate_mock()
- With NIFTy.re: as the signal response in optimize_kl()
- With BlackJAX: via build_log_posterior()
- With optax: via build_loss()

Design: All model components are composed as pure functions.
The class is a thin stateful wrapper holding pre-computed grids
and SSP data. The __call__ method is JIT-compatible.
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple

from diffsed.models.sfh.psd_models import drw_variance
from diffsed.models.sfh.gp_sfh import gp_from_xi, compute_sqrt_power_drw
from diffsed.models.sfh.mean_sfh import double_powerlaw
from diffsed.models.dust.charlot_fall import charlot_fall
from diffsed.models.sps.dsps_wrapper import (
    compute_csp_weights,
    compute_csp_sed,
    interpolate_metallicity,
)
from diffsed.models.observation.photometry import compute_flux_density
from diffsed.models.observation.spectroscopy import (
    compute_spectrum,
    chebyshev_calibration,
)
from diffsed.utils.grid import (
    make_log_age_grid, log_age_to_age_yr, grid_spacing,
    interpolate_to_linear_time,
)
from diffsed.utils.cosmology import luminosity_distance


class ModelConfig(NamedTuple):
    """Static model configuration (does not change during inference).

    Attributes
    ----------
    n_grid : int
        GP grid size.
    log_age_min : float
        Minimum log10(age/yr).
    log_age_max : float
        Maximum log10(age/yr).
    redshift : float
        Source redshift.
    psd_type : str
        PSD model ("drw").
    mean_sfh_type : str
        Mean SFH model ("double_powerlaw").
    dust_type : str
        Dust model ("charlot_fall").
    """
    n_grid: int = 256
    log_age_min: float = 6.0
    log_age_max: float = 10.14
    redshift: float = 0.1
    psd_type: str = "drw"
    mean_sfh_type: str = "double_powerlaw"
    dust_type: str = "charlot_fall"


class ForwardModel:
    """Differentiable forward model: parameters -> predicted SED.

    Pipeline:
        1. PSD params -> amplitude operator sqrt(P/d)
        2. xi (latent) -> GP realization x(t) via IFFT
        3. Mean SFH * exp(x(t) - K(0)/2) -> full SFR(t) [lognormal correction]
        4. Interpolate SFR to SSP age grid -> SFH weights
        5. DSPS CSP integral with metallicity interpolation -> intrinsic SED
        6. Charlot & Fall dust -> attenuated SED
        7. Filter convolution -> photometry OR pixel interpolation -> spectrum

    Parameters
    ----------
    ssp_data : SSPData
        Pre-loaded SSP templates.
    config : ModelConfig
        Static model configuration.
    filter_waves : list of array, optional
        Filter wavelength grids (Angstrom).
    filter_trans : list of array, optional
        Filter transmission curves.
    """

    def __init__(self, ssp_data, config=None, filter_waves=None,
                 filter_trans=None, wave_obs=None):
        if config is None:
            config = ModelConfig()

        self.config = config
        self.ssp_data = ssp_data

        # Pre-compute grids
        self.log_age_grid = make_log_age_grid(
            config.n_grid, config.log_age_min, config.log_age_max
        )
        self.d_log_age = grid_spacing(self.log_age_grid)
        self.age_yr = log_age_to_age_yr(self.log_age_grid)

        # SSP ages in log10(yr)
        self.ssp_log_ages_yr = ssp_data.ssp_lg_age_gyr + 9.0  # Gyr -> yr
        self.ssp_ages_yr = 10.0 ** self.ssp_log_ages_yr

        # Luminosity distance
        self.dl_cm = luminosity_distance(config.redshift)

        # Filters and spectroscopic wavelength grid
        self.filter_waves = filter_waves
        self.filter_trans = filter_trans
        self._wave_obs = wave_obs

    def compute_sqrt_power(self, sigma_ps, tau_ps):
        """Pre-compute amplitude operator for given PSD params.

        Uses the Jacobian-corrected DRW PSD on the log-age grid:
        P_u(q) = P_t(q / (t_ref * ln10)) / (t_ref * ln10)

        Parameters
        ----------
        sigma_ps : float
            DRW PSD amplitude.
        tau_ps : float
            DRW damping timescale (yr).

        Returns
        -------
        array, shape (n_freq,)
            sqrt(P(omega) / d_log_age).
        """
        return compute_sqrt_power_drw(
            self.config.n_grid, float(self.d_log_age), sigma_ps, tau_ps
        )

    def __call__(self, params):
        """Run the full forward model.

        Parameters
        ----------
        params : dict
            Must contain:
            - "xi": latent vector, shape (n_grid,)
            - "sigma_ps": PSD amplitude
            - "tau_ps": PSD timescale (yr)
            - "alpha": double power law falling index
            - "beta": double power law rising index
            - "tau_sfh": double power law turnover (yr)
            - "sfr_norm": SFR normalization (Msun/yr)
            - "log_z": log10(Z/Zsun) metallicity
            - "tau_v1": birth cloud dust optical depth
            - "tau_v2": diffuse ISM dust optical depth
            - "dust_n": dust power-law index

        Returns
        -------
        array, shape (n_wave,)
            Attenuated rest-frame SED, or
        array, shape (n_filters,)
            Predicted photometric flux densities if filters are set.
        """
        # --- Step 1: Amplitude operator from PSD params ---
        sqrt_power = self.compute_sqrt_power(
            params["sigma_ps"], params["tau_ps"]
        )

        # --- Step 2: GP realization from standardized xi ---
        gp_x = gp_from_xi(params["xi"], sqrt_power, self.config.n_grid)

        # --- Step 3: Full SFH with lognormal correction ---
        # K(0) = sigma_PS^2 / 2 is the GP variance at zero lag
        k0_half = drw_variance(params["sigma_ps"]) / 2.0

        sfr_mean = double_powerlaw(
            self.age_yr,
            alpha=params["alpha"],
            beta=params["beta"],
            tau=params["tau_sfh"],
            norm=params["sfr_norm"],
        )
        sfr = sfr_mean * jnp.exp(gp_x - k0_half)

        # --- Step 4: Interpolate SFR to SSP age grid ---
        sfr_on_ssp = jnp.interp(
            self.ssp_log_ages_yr, self.log_age_grid, sfr
        )
        weights = compute_csp_weights(sfr_on_ssp, self.ssp_ages_yr)

        # --- Step 5: CSP integral with metallicity ---
        ssp_flux_at_z = interpolate_metallicity(
            self.ssp_data.ssp_flux,
            self.ssp_data.ssp_lgmet,
            params["log_z"],
        )

        # --- Step 6: Dust attenuation ---
        dust_atten = charlot_fall(
            self.ssp_data.ssp_wave,
            self.ssp_ages_yr,
            tau_v1=params["tau_v1"],
            tau_v2=params["tau_v2"],
            n_slope=params["dust_n"],
        )

        # --- Step 7: Compose CSP SED ---
        sed = compute_csp_sed(weights, ssp_flux_at_z, dust_atten)

        return sed

    def _compute_sfh(self, params):
        """Run the SFH pipeline and return mean SFR, full SFR, and related arrays.

        Parameters
        ----------
        params : dict
            Model parameters (must contain xi, sigma_ps, tau_ps, alpha,
            beta, tau_sfh, sfr_norm).

        Returns
        -------
        dict with keys:
            "sfr_mean": mean SFR(t) without GP (Msun/yr), on log-age grid
            "sfr_full": full SFR(t) including GP (Msun/yr), on log-age grid
            "sfr_mean_on_ssp": mean SFR interpolated to SSP ages (Msun/yr)
            "sfr_full_on_ssp": full SFR interpolated to SSP ages (Msun/yr)
        """
        sqrt_power = self.compute_sqrt_power(
            params["sigma_ps"], params["tau_ps"]
        )
        gp_x = gp_from_xi(params["xi"], sqrt_power, self.config.n_grid)
        k0_half = drw_variance(params["sigma_ps"]) / 2.0

        sfr_mean = double_powerlaw(
            self.age_yr,
            alpha=params["alpha"],
            beta=params["beta"],
            tau=params["tau_sfh"],
            norm=params["sfr_norm"],
        )
        sfr_full = sfr_mean * jnp.exp(gp_x - k0_half)

        sfr_mean_on_ssp = jnp.interp(
            self.ssp_log_ages_yr, self.log_age_grid, sfr_mean
        )
        sfr_full_on_ssp = jnp.interp(
            self.ssp_log_ages_yr, self.log_age_grid, sfr_full
        )

        return {
            "sfr_mean": sfr_mean,
            "sfr_full": sfr_full,
            "sfr_mean_on_ssp": sfr_mean_on_ssp,
            "sfr_full_on_ssp": sfr_full_on_ssp,
        }

    def predict_sfh_for_plot(self, params, n_linear=1000):
        """Compute SFH on a uniform linear-time grid for plotting.

        The GP lives on a log-age grid (uniform in log10(t)), which creates
        a visual artifact when plotted vs linear lookback time: the wiggles
        appear denser at large lookback time (old ages) where grid points
        are spaced farther apart in linear time. This method resamples
        to a uniform linear grid, giving visually honest SFH curves.

        Parameters
        ----------
        params : dict
            Model parameters.
        n_linear : int
            Number of points in the output linear grid.

        Returns
        -------
        dict with keys:
            "t_gyr": lookback time in Gyr, shape (n_linear,)
            "sfr_mean": mean SFH on linear grid (Msun/yr)
            "sfr_full": full SFH including GP on linear grid (Msun/yr)
        """
        sfh = self._compute_sfh(params)
        t_gyr_mean, sfr_mean_lin = interpolate_to_linear_time(
            self.log_age_grid, sfh["sfr_mean"], n_linear
        )
        t_gyr_full, sfr_full_lin = interpolate_to_linear_time(
            self.log_age_grid, sfh["sfr_full"], n_linear
        )
        return {
            "t_gyr": t_gyr_mean,
            "sfr_mean": sfr_mean_lin,
            "sfr_full": sfr_full_lin,
        }

    def compute_stellar_mass(self, params):
        """Compute total stellar mass formed by integrating the SFH.

        Returns both the mass from the mean SFH (analytic expectation)
        and from the full SFH (including the GP realization).

        Parameters
        ----------
        params : dict
            Model parameters.

        Returns
        -------
        dict with keys:
            "mstar_mean": mass formed from mean SFH only (Msun)
            "mstar_total": mass formed from full SFH including GP (Msun)
        """
        sfh = self._compute_sfh(params)

        weights_mean = compute_csp_weights(
            sfh["sfr_mean_on_ssp"], self.ssp_ages_yr
        )
        weights_full = compute_csp_weights(
            sfh["sfr_full_on_ssp"], self.ssp_ages_yr
        )

        return {
            "mstar_mean": jnp.sum(weights_mean),
            "mstar_total": jnp.sum(weights_full),
        }

    def compute_derived_quantities(self, params):
        """Compute derived physical quantities from model parameters.

        Parameters
        ----------
        params : dict
            Model parameters.

        Returns
        -------
        dict with keys:
            "mstar_formed": total stellar mass formed (Msun)
            "mstar_mean": mass from mean SFH only (Msun)
            "sfr_100myr": SFR averaged over last 100 Myr (Msun/yr)
            "sfr_10myr": SFR averaged over last 10 Myr (Msun/yr)
            "ssfr": specific SFR = sfr_100myr / mstar_formed (yr^-1)
        """
        sfh = self._compute_sfh(params)

        # Stellar masses
        weights_mean = compute_csp_weights(
            sfh["sfr_mean_on_ssp"], self.ssp_ages_yr
        )
        weights_full = compute_csp_weights(
            sfh["sfr_full_on_ssp"], self.ssp_ages_yr
        )
        mstar_mean = jnp.sum(weights_mean)
        mstar_formed = jnp.sum(weights_full)

        # Average SFR over recent time windows using the full SFH on
        # the log-age grid (finer resolution than SSP grid at young ages)
        age_yr = self.age_yr  # lookback time in years
        sfr_full = sfh["sfr_full"]

        # SFR averaged over last 100 Myr: mean SFR where age < 1e8 yr
        mask_100myr = age_yr <= 1e8
        sfr_100myr = jnp.where(
            jnp.sum(mask_100myr) > 0,
            jnp.sum(sfr_full * mask_100myr) / jnp.maximum(jnp.sum(mask_100myr), 1.0),
            sfr_full[0],
        )

        # SFR averaged over last 10 Myr: mean SFR where age < 1e7 yr
        mask_10myr = age_yr <= 1e7
        sfr_10myr = jnp.where(
            jnp.sum(mask_10myr) > 0,
            jnp.sum(sfr_full * mask_10myr) / jnp.maximum(jnp.sum(mask_10myr), 1.0),
            sfr_full[0],
        )

        ssfr = sfr_100myr / jnp.maximum(mstar_formed, 1.0)

        return {
            "mstar_formed": mstar_formed,
            "mstar_mean": mstar_mean,
            "sfr_100myr": sfr_100myr,
            "sfr_10myr": sfr_10myr,
            "ssfr": ssfr,
        }

    def predict_photometry(self, params):
        """Run forward model and compute photometry.

        Returns
        -------
        array, shape (n_filters,)
            Predicted flux densities (erg/s/cm^2/Hz).
        """
        sed = self(params)

        if self.filter_waves is None:
            raise ValueError("No filter curves set. Pass filter_waves/trans.")

        fluxes = []
        for fw, ft in zip(self.filter_waves, self.filter_trans):
            f = compute_flux_density(
                sed, self.ssp_data.ssp_wave, fw, ft,
                self.config.redshift, self.dl_cm,
            )
            fluxes.append(f)
        return jnp.array(fluxes)

    def predict_spectrum(self, params, wave_obs):
        """Run forward model and compute spectrum at observed wavelengths.

        Parameters
        ----------
        params : dict
            Model parameters.
        wave_obs : array
            Observed wavelength grid (Angstrom).

        Returns
        -------
        array, shape (n_pix,)
            Predicted flux at each spectral pixel.
        """
        sed = self(params)
        return compute_spectrum(
            sed, self.ssp_data.ssp_wave, wave_obs,
            self.config.redshift, self.dl_cm,
        )


# ---------------------------------------------------------------------------
# Likelihood functions
# ---------------------------------------------------------------------------

@jax.jit
def gaussian_log_likelihood(predicted, observed, noise):
    """Gaussian log-likelihood for photometry or spectroscopy.

    ln L = -0.5 * sum((d - m)^2 / sigma^2)

    Parameters
    ----------
    predicted : array
        Model predictions.
    observed : array
        Observed data.
    noise : array
        1-sigma uncertainties.

    Returns
    -------
    float
        Log-likelihood.
    """
    residual = (observed - predicted) / noise
    return -0.5 * jnp.sum(residual ** 2)


@jax.jit
def standard_normal_log_prior(xi):
    """Standard normal prior on latent variables: -0.5 * xi^T xi.

    This is the prior that NIFTy assumes for all standardized variables.
    """
    return -0.5 * jnp.sum(xi ** 2)


# ---------------------------------------------------------------------------
# Mock generation
# ---------------------------------------------------------------------------

def generate_mock(model, params, key=None, snr=20.0):
    """Generate mock photometry with Gaussian noise.

    Parameters
    ----------
    model : ForwardModel
        Configured forward model.
    params : dict
        True parameter values.
    key : PRNGKey, optional
        For adding noise. If None, return noiseless.
    snr : float
        Signal-to-noise ratio per band.

    Returns
    -------
    dict with keys:
        "flux_true": noiseless flux
        "flux_obs": noisy flux (if key given)
        "noise": 1-sigma uncertainties
        "params": input params
    """
    flux_true = model.predict_photometry(params)
    noise = flux_true / snr

    result = {
        "flux_true": flux_true,
        "noise": noise,
        "params": params,
    }

    if key is not None:
        flux_obs = flux_true + noise * jax.random.normal(key, shape=flux_true.shape)
        result["flux_obs"] = flux_obs

    return result
