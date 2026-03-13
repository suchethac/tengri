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

from diffsed.models.sfh.psd_models import psd_drw, psd_to_sqrt_power, drw_variance
from diffsed.models.sfh.gp_sfh import gp_from_xi, xi_to_complex
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
from diffsed.utils.grid import make_log_age_grid, log_age_to_age_yr, grid_spacing
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
                 filter_trans=None):
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

        # FFT frequencies (pre-computed, static)
        n_freq = config.n_grid // 2 + 1
        freqs = jnp.fft.rfftfreq(config.n_grid, d=float(self.d_log_age))
        self.omega = 2.0 * jnp.pi * freqs

        # Luminosity distance
        self.dl_cm = luminosity_distance(config.redshift)

        # Filters
        self.filter_waves = filter_waves
        self.filter_trans = filter_trans

    def compute_sqrt_power(self, sigma_ps, tau_ps):
        """Pre-compute amplitude operator for given PSD params.

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
        p_k = psd_drw(self.omega, sigma_ps, tau_ps)
        return psd_to_sqrt_power(p_k, self.d_log_age)

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
