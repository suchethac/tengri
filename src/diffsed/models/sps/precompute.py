"""Pre-computation of SSP observables at fixed redshift.

The key insight: at inference time, the redshift is known. Everything
that depends on z but NOT on model parameters (SFH weights, dust, Z)
can be computed once, outside the MCMC loop. This eliminates the
expensive wavelength dimension from the gradient tape.

Three levels of pre-computation:

1. Photometric: Pre-integrate SSP through filters -> c_SSP(age, Z, band)
   Galaxy photometry = weighted sum over age/Z, no wavelength integral.
   Speedup: 30-50x (Zacharegkas+2025 Section 3).

2. Spectroscopic: Pre-rebin SSP to observed pixel wavelengths.
   Reduces wavelength dimension from ~7000 to ~1000-3000.

3. Combined: Both simultaneously.

Usage:
    # At setup (once per galaxy / once per redshift)
    precomp = precompute_photometry(ssp_data, filters, redshift)

    # At each MCMC step (fast: no wavelength integrals)
    flux = fast_photometry(weights, precomp, dust_at_eff_wave)
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp


class PhotometricPrecomputation(NamedTuple):
    """Pre-computed SSP broadband fluxes for fast photometry.

    Attributes
    ----------
    ssp_phot : array, shape (n_met, n_age, n_filters)
        SSP broadband flux per metallicity, age, and filter.
        c_SSP = int T(lam|z) L_SSP(lam|age,Z) lam dlam / int T(lam|z) lam dlam
    effective_wavelengths : array, shape (n_filters,)
        Effective wavelength of each filter (Angstrom, observed frame).
        Used for evaluating dust at a single wavelength per band.
    effective_wavelengths_rest : array, shape (n_filters,)
        Effective wavelength in rest frame (Angstrom).
    flux_scale : float
        (1+z) / (4 pi dL^2) geometric factor.
    redshift : float
        Source redshift.
    n_filters : int
        Number of filters.
    """

    ssp_phot: jnp.ndarray
    effective_wavelengths: jnp.ndarray
    effective_wavelengths_rest: jnp.ndarray
    flux_scale: float
    redshift: float
    n_filters: int


class SpectroscopicPrecomputation(NamedTuple):
    """Pre-rebinned SSP templates for fast spectroscopy.

    Attributes
    ----------
    ssp_on_pixels : array, shape (n_met, n_age, n_pix)
        SSP flux interpolated to observed spectral pixel wavelengths
        (in rest frame).
    wave_rest_pixels : array, shape (n_pix,)
        Rest-frame wavelengths of spectral pixels.
    wave_obs_pixels : array, shape (n_pix,)
        Observed-frame wavelengths.
    flux_scale : float
        Geometric scaling factor.
    redshift : float
        Source redshift.
    """

    ssp_on_pixels: jnp.ndarray
    wave_rest_pixels: jnp.ndarray
    wave_obs_pixels: jnp.ndarray
    flux_scale: float
    redshift: float


def precompute_photometry(
    ssp_data, filter_waves, filter_trans, redshift, dl_cm
) -> PhotometricPrecomputation:
    """Pre-compute SSP broadband fluxes for all filters.

    This eliminates the wavelength integral from the MCMC loop.
    After this, galaxy photometry is just a weighted sum.

    Parameters
    ----------
    ssp_data : SSPData
        SSP templates with ssp_wave, ssp_flux, ssp_lg_age_gyr, ssp_lgmet.
    filter_waves : list of array
        Wavelength grid per filter (observed frame, Angstrom).
    filter_trans : list of array
        Transmission curve per filter.
    redshift : float
        Source redshift.
    dl_cm : float
        Luminosity distance (cm).

    Returns
    -------
    PhotometricPrecomputation
        Pre-computed data for fast_photometry().
    """
    n_met = ssp_data.ssp_flux.shape[0]
    n_age = ssp_data.ssp_flux.shape[1]
    n_filters = len(filter_waves)

    # Redshift SSP wavelengths to observed frame
    wave_obs = ssp_data.ssp_wave * (1.0 + redshift)

    # Effective wavelengths per filter
    eff_waves = []
    for fw, ft in zip(filter_waves, filter_trans):
        lam_eff = jnp.trapezoid(ft * fw**2, fw) / jnp.trapezoid(ft * fw, fw)
        eff_waves.append(lam_eff)
    eff_waves = jnp.array(eff_waves)
    eff_waves_rest = eff_waves / (1.0 + redshift)

    # Pre-integrate SSP through each filter for each (met, age)
    # Vectorized: interpolate all (met, age) SSPs to each filter grid at once
    import numpy as np

    ssp_flux_np = np.asarray(ssp_data.ssp_flux)  # (n_met, n_age, n_wave)
    wave_obs_np = np.asarray(wave_obs)
    ssp_phot_np = np.zeros((n_met, n_age, n_filters))

    for f_idx, (fw, ft) in enumerate(zip(filter_waves, filter_trans)):
        fw_np, ft_np = np.asarray(fw), np.asarray(ft)
        denom = np.trapezoid(ft_np * fw_np, fw_np)

        # Interpolate all SSPs onto this filter's wavelength grid
        # ssp_flux_np is (n_met, n_age, n_wave), we need (n_met, n_age, len(fw))
        ssp_on_filt = np.zeros((n_met, n_age, len(fw_np)))
        for m_idx in range(n_met):
            for a_idx in range(n_age):
                ssp_on_filt[m_idx, a_idx] = np.interp(
                    fw_np, wave_obs_np, ssp_flux_np[m_idx, a_idx], left=0.0, right=0.0
                )

        # Vectorized integration: (n_met, n_age, n_fw) * (n_fw,) → trapz
        integrand = ssp_on_filt * ft_np[None, None, :] * fw_np[None, None, :]
        num = np.trapezoid(integrand, fw_np, axis=-1)  # (n_met, n_age)
        ssp_phot_np[:, :, f_idx] = num / max(denom, 1e-30)

    ssp_phot = jnp.array(ssp_phot_np)

    flux_scale = (1.0 + redshift) / (4.0 * jnp.pi * dl_cm**2)

    return PhotometricPrecomputation(
        ssp_phot=ssp_phot,
        effective_wavelengths=eff_waves,
        effective_wavelengths_rest=eff_waves_rest,
        flux_scale=float(flux_scale),
        redshift=float(redshift),
        n_filters=n_filters,
    )


def precompute_spectroscopy(
    ssp_data, wave_obs_pixels, redshift, dl_cm
) -> SpectroscopicPrecomputation:
    """Pre-rebin SSP templates to observed spectral pixel wavelengths.

    Reduces the wavelength dimension from ~7000 (native SSP) to
    ~1000-3000 (observed pixels), cutting memory and compute.

    Parameters
    ----------
    ssp_data : SSPData
        SSP templates.
    wave_obs_pixels : array, shape (n_pix,)
        Observed-frame wavelengths of spectral pixels (Angstrom).
    redshift : float
        Source redshift.
    dl_cm : float
        Luminosity distance (cm).

    Returns
    -------
    SpectroscopicPrecomputation
        Pre-rebinned SSP data for fast_spectrum().
    """
    n_met = ssp_data.ssp_flux.shape[0]
    n_age = ssp_data.ssp_flux.shape[1]
    n_pix = len(wave_obs_pixels)

    wave_rest_pixels = wave_obs_pixels / (1.0 + redshift)

    # Interpolate all SSP spectra to the pixel rest-frame wavelengths
    import numpy as np

    ssp_flux_np = np.asarray(ssp_data.ssp_flux)
    wave_rest_np = np.asarray(wave_rest_pixels)
    wave_ssp_np = np.asarray(ssp_data.ssp_wave)
    ssp_on_pixels_np = np.zeros((n_met, n_age, n_pix))
    for m_idx in range(n_met):
        for a_idx in range(n_age):
            ssp_on_pixels_np[m_idx, a_idx] = np.interp(
                wave_rest_np,
                wave_ssp_np,
                ssp_flux_np[m_idx, a_idx],
                left=0.0,
                right=0.0,
            )
    ssp_on_pixels = jnp.array(ssp_on_pixels_np)

    flux_scale = (1.0 + redshift) / (4.0 * jnp.pi * dl_cm**2)

    return SpectroscopicPrecomputation(
        ssp_on_pixels=ssp_on_pixels,
        wave_rest_pixels=wave_rest_pixels,
        wave_obs_pixels=wave_obs_pixels,
        flux_scale=float(flux_scale),
        redshift=float(redshift),
    )


# -----------------------------------------------------------------------
# Fast inference-time functions (these run inside the MCMC loop)
# -----------------------------------------------------------------------


@jax.jit
def fast_photometry(
    weights: jnp.ndarray, ssp_phot_at_z: jnp.ndarray, dust_at_eff: jnp.ndarray, flux_scale: float
) -> jnp.ndarray:
    """Compute photometry from pre-computed SSP broadband fluxes.

    This is what runs at each MCMC step. No wavelength integrals —
    just a weighted sum over the age dimension.

    c_gal(band) = flux_scale * sum_i weights_i * dust_i(band) * c_SSP_i(band)

    Parameters
    ----------
    weights : array, shape (n_age,)
        Normalized SFH weights.
    ssp_phot_at_z : array, shape (n_age, n_filters)
        Pre-computed SSP broadband fluxes at the target metallicity.
        (Already interpolated in Z from the full grid.)
    dust_at_eff : array, shape (n_age, n_filters)
        Dust transmission evaluated at effective wavelengths per age/band.
    flux_scale : float
        Geometric factor (1+z) / (4 pi dL^2).

    Returns
    -------
    array, shape (n_filters,)
        Observed flux density per filter (erg/s/cm^2/Hz).
    """
    # Weighted sum: weights [Msun] * ssp [Lsun/Hz/Msun] * dust -> Lsun/Hz
    from diffsed.models.sps.dsps_wrapper import LSUN_ERG_PER_S

    flux_lsun = jnp.einsum("i,if,if->f", weights, dust_at_eff, ssp_phot_at_z)
    return flux_scale * flux_lsun * LSUN_ERG_PER_S


@jax.jit
def fast_spectrum(
    weights: jnp.ndarray,
    ssp_on_pixels_at_z: jnp.ndarray,
    dust_at_pixels: jnp.ndarray,
    flux_scale: float,
) -> jnp.ndarray:
    """Compute spectrum from pre-rebinned SSP templates.

    Parameters
    ----------
    weights : array, shape (n_age,)
        Normalized SFH weights.
    ssp_on_pixels_at_z : array, shape (n_age, n_pix)
        Pre-rebinned SSP flux at target metallicity.
    dust_at_pixels : array, shape (n_age, n_pix)
        Dust transmission at each pixel wavelength per age.
    flux_scale : float
        Geometric factor.

    Returns
    -------
    array, shape (n_pix,)
        Model flux at each spectral pixel.
    """
    flux = jnp.einsum("i,ip,ip->p", weights, dust_at_pixels, ssp_on_pixels_at_z)
    return flux_scale * flux


@jax.jit
def interpolate_ssp_phot_metallicity(
    ssp_phot: jnp.ndarray, ssp_lgmet: jnp.ndarray, log_z: float
) -> jnp.ndarray:
    """Interpolate pre-computed SSP photometry to target metallicity.

    Parameters
    ----------
    ssp_phot : array, shape (n_met, n_age, n_filters)
        Pre-computed SSP broadband fluxes.
    ssp_lgmet : array, shape (n_met,)
        Metallicity grid.
    log_z : float
        Target log10(Z/Zsun).

    Returns
    -------
    array, shape (n_age, n_filters)
        Interpolated SSP photometry.
    """
    log_z_clamped = jnp.clip(log_z, ssp_lgmet[0], ssp_lgmet[-1])
    idx = jnp.clip(jnp.searchsorted(ssp_lgmet, log_z_clamped) - 1, 0, len(ssp_lgmet) - 2)
    frac = (log_z_clamped - ssp_lgmet[idx]) / (ssp_lgmet[idx + 1] - ssp_lgmet[idx])
    return (1.0 - frac) * ssp_phot[idx] + frac * ssp_phot[idx + 1]
