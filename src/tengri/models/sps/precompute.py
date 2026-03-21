"""Pre-computation of SSP observables at fixed or tabulated redshift.

The key insight: at inference time, redshift is either known (fixed) or
varies slowly. Everything that depends on z but NOT on model parameters
(SFH weights, dust, Z) can be pre-computed, eliminating the expensive
wavelength dimension from the gradient tape.

Four levels of pre-computation:

1. Photometric (fixed z): Pre-integrate SSP through filters.
   Speedup: 30-50x (Zacharegkas+2025 Section 3).

2. Spectroscopic (fixed z): Pre-rebin SSP to observed pixel wavelengths.
   Reduces wavelength dimension from ~7000 to ~1000-3000.

3. Photometric z-table (free z): Pre-integrate SSP at each z in a grid.
   At inference: interpolate to current z. Same speedup, z is free.
   Follows DSPS precompute_ssp_obsmags_on_z_table approach.

4. Combined: Both simultaneously.

Usage:
    # Fixed redshift (once per galaxy)
    precomp = precompute_photometry(ssp_data, filters, redshift)

    # Free redshift (once per survey / filter set)
    ztable = precompute_photometry_ztable(ssp_data, filters)
    # At each MCMC step: interpolate to current z
    flux = fast_photometry_ztable(weights, ztable, z, dust_params)
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

# numpy >= 2.0 uses trapezoid; older versions used trapz
_np_trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def _vectorized_interp(
    xp_target: np.ndarray, xp_source: np.ndarray, yp_source: np.ndarray
) -> np.ndarray:
    """Vectorized linear interpolation: all (met, age) SSPs at once.

    Replaces the inner ``for m in n_met: for a in n_age: np.interp(...)``
    double loop with a single vectorized NumPy operation.  Computes
    interpolation indices and weights once, then applies via fancy indexing.

    Parameters
    ----------
    xp_target : array, shape (n_target,)
        Target x-coordinates (e.g. filter wavelengths).
    xp_source : array, shape (n_source,)
        Source x-coordinates (must be sorted ascending).
    yp_source : array, shape (n_met, n_age, n_source)
        Source y-values for all metallicities and ages.

    Returns
    -------
    array, shape (n_met, n_age, n_target)
        Interpolated values, with out-of-bounds set to 0.
    """
    n_source = len(xp_source)

    # Compute interpolation indices and fractional weights once
    idx = np.searchsorted(xp_source, xp_target) - 1
    idx = np.clip(idx, 0, n_source - 2)

    dx = xp_source[idx + 1] - xp_source[idx]
    # Guard against zero-width bins (shouldn't happen with sorted grids)
    dx = np.where(dx > 0, dx, 1.0)
    frac = (xp_target - xp_source[idx]) / dx

    # Vectorized gather: (n_met, n_age, n_target) via fancy indexing
    result = (1.0 - frac) * yp_source[:, :, idx] + frac * yp_source[:, :, idx + 1]

    # Zero out-of-bounds (left and right)
    oob = (xp_target < xp_source[0]) | (xp_target > xp_source[-1])
    result[:, :, oob] = 0.0

    return result


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
    # Vectorized: compute interpolation weights once, apply to all SSPs at once
    import numpy as np

    ssp_flux_np = np.asarray(ssp_data.ssp_flux)  # (n_met, n_age, n_wave)
    wave_obs_np = np.asarray(wave_obs)
    ssp_phot_np = np.zeros((n_met, n_age, n_filters))

    for f_idx, (fw, ft) in enumerate(zip(filter_waves, filter_trans)):
        fw_np, ft_np = np.asarray(fw), np.asarray(ft)
        denom = _np_trapezoid(ft_np * fw_np, fw_np)

        # Vectorized interpolation: compute weights once, apply to all (met, age)
        ssp_on_filt = _vectorized_interp(fw_np, wave_obs_np, ssp_flux_np)

        # Vectorized integration: (n_met, n_age, n_fw) * (n_fw,) → trapz
        integrand = ssp_on_filt * ft_np[None, None, :] * fw_np[None, None, :]
        num = _np_trapezoid(integrand, fw_np, axis=-1)  # (n_met, n_age)
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
    wave_rest_pixels = wave_obs_pixels / (1.0 + redshift)

    # Interpolate all SSP spectra to the pixel rest-frame wavelengths
    # Vectorized: compute weights once, apply to all (met, age) simultaneously
    import numpy as np

    ssp_flux_np = np.asarray(ssp_data.ssp_flux)
    wave_rest_np = np.asarray(wave_rest_pixels)
    wave_ssp_np = np.asarray(ssp_data.ssp_wave)
    ssp_on_pixels_np = _vectorized_interp(wave_rest_np, wave_ssp_np, ssp_flux_np)
    ssp_on_pixels = jnp.array(ssp_on_pixels_np)

    flux_scale = (1.0 + redshift) / (4.0 * jnp.pi * dl_cm**2)

    return SpectroscopicPrecomputation(
        ssp_on_pixels=ssp_on_pixels,
        wave_rest_pixels=wave_rest_pixels,
        wave_obs_pixels=wave_obs_pixels,
        flux_scale=float(flux_scale),
        redshift=float(redshift),
    )


class PhotometricZTable(NamedTuple):
    """Pre-computed SSP broadband fluxes on a redshift grid.

    For free-redshift inference: interpolate this table to the current z
    instead of computing the full wavelength integral each step.

    Attributes
    ----------
    ssp_phot_table : array, shape (n_z, n_met, n_age, n_filters)
        SSP broadband flux at each redshift, metallicity, age, and filter.
    eff_waves_rest_table : array, shape (n_z, n_filters)
        Rest-frame effective wavelengths at each redshift.
    flux_scale_table : array, shape (n_z,)
        Geometric factor (1+z)/(4π dL²) at each redshift.
    z_grid : array, shape (n_z,)
        Redshift grid.
    n_filters : int
        Number of filters.
    igm_trans_table : array, shape (n_z, n_filters)
        IGM transmission at effective observed wavelengths per redshift.
        All ones when IGM is not applied.
    """

    ssp_phot_table: jnp.ndarray
    eff_waves_rest_table: jnp.ndarray
    flux_scale_table: jnp.ndarray
    z_grid: jnp.ndarray
    n_filters: int
    igm_trans_table: jnp.ndarray


def precompute_photometry_ztable(
    ssp_data,
    filter_waves,
    filter_trans,
    z_grid=None,
    z_min=0.001,
    z_max=3.0,
    n_z=100,
    apply_igm=False,
) -> PhotometricZTable:
    """Pre-compute SSP broadband fluxes on a redshift grid.

    Evaluates the full wavelength integral at each z in the grid.
    At inference time, interpolate to the current z — same speedup
    as fixed-z precomputation, but z is now a free parameter.

    Parameters
    ----------
    ssp_data : SSPData
        SSP templates.
    filter_waves : list of array
        Wavelength grid per filter (observed frame, Angstrom).
    filter_trans : list of array
        Transmission curve per filter.
    z_grid : array, optional
        Custom redshift grid. If None, uses linspace(z_min, z_max, n_z).
    z_min : float
        Minimum redshift (default 0.001).
    z_max : float
        Maximum redshift (default 3.0).
    n_z : int
        Number of redshift grid points (default 100).
    apply_igm : bool
        If True, precompute IGM transmission (Inoue+2014) at the
        effective observed wavelengths for each z in the grid.
        Default False (igm_trans_table will be all ones).

    Returns
    -------
    PhotometricZTable
        Pre-computed table for fast_photometry_ztable().
    """
    from tengri.utils.cosmology import luminosity_distance

    if z_grid is None:
        z_grid = jnp.linspace(z_min, z_max, n_z)
    else:
        z_grid = jnp.asarray(z_grid)

    n_z_pts = len(z_grid)
    n_filters = len(filter_waves)
    n_met = ssp_data.ssp_flux.shape[0]
    n_age = ssp_data.ssp_flux.shape[1]

    import numpy as np

    ssp_phot_all = np.zeros((n_z_pts, n_met, n_age, n_filters))
    eff_waves_rest_all = np.zeros((n_z_pts, n_filters))
    flux_scale_all = np.zeros(n_z_pts)
    igm_trans_all = np.ones((n_z_pts, n_filters))

    ssp_flux_np = np.asarray(ssp_data.ssp_flux)
    wave_ssp_np = np.asarray(ssp_data.ssp_wave)

    # Precompute filter denominators and effective wavelengths (z-independent)
    filter_denoms = []
    eff_waves_obs = []  # observed-frame effective wavelengths (z-independent)
    for fw, ft in zip(filter_waves, filter_trans):
        fw_np, ft_np = np.asarray(fw), np.asarray(ft)
        filter_denoms.append(_np_trapezoid(ft_np * fw_np, fw_np))
        eff_waves_obs.append(
            _np_trapezoid(ft_np * fw_np**2, fw_np)
            / max(_np_trapezoid(ft_np * fw_np, fw_np), 1e-30)
        )
    eff_waves_obs = np.array(eff_waves_obs)

    for zi, z_val in enumerate(z_grid):
        z_val = float(z_val)
        wave_obs = wave_ssp_np * (1.0 + z_val)

        # Effective wavelengths: observed-frame are filter properties (z-independent)
        eff_waves_rest_all[zi] = eff_waves_obs / (1.0 + z_val)

        # IGM transmission at effective observed wavelengths
        if apply_igm:
            from tengri.models.igm import igm_transmission

            igm_trans_all[zi] = np.asarray(igm_transmission(jnp.asarray(eff_waves_obs), z_val))

        # Pre-integrate SSP through each filter (vectorized over met × age)
        for f_idx, (fw, ft) in enumerate(zip(filter_waves, filter_trans)):
            fw_np, ft_np = np.asarray(fw), np.asarray(ft)
            denom = filter_denoms[f_idx]

            ssp_on_filt = _vectorized_interp(fw_np, wave_obs, ssp_flux_np)
            integrand = ssp_on_filt * ft_np[None, None, :] * fw_np[None, None, :]
            num = _np_trapezoid(integrand, fw_np, axis=-1)
            ssp_phot_all[zi, :, :, f_idx] = num / max(denom, 1e-30)

        # Geometric flux scale
        dl_cm = float(luminosity_distance(z_val))
        flux_scale_all[zi] = (1.0 + z_val) / (4.0 * np.pi * dl_cm**2)

    return PhotometricZTable(
        ssp_phot_table=jnp.array(ssp_phot_all),
        eff_waves_rest_table=jnp.array(eff_waves_rest_all),
        flux_scale_table=jnp.array(flux_scale_all),
        z_grid=z_grid,
        n_filters=n_filters,
        igm_trans_table=jnp.array(igm_trans_all),
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
    from tengri.models.sps.dsps_wrapper import LSUN_ERG_PER_S

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


@jax.jit
def interpolate_ztable(ztable_ssp_phot, ztable_eff_rest, ztable_flux_scale, z_grid, z):
    """Interpolate z-table to a specific redshift (JIT-compatible).

    Linear interpolation along the z dimension of the precomputed table.

    Parameters
    ----------
    ztable_ssp_phot : array, shape (n_z, n_met, n_age, n_filters)
        Precomputed SSP photometry on z grid.
    ztable_eff_rest : array, shape (n_z, n_filters)
        Rest-frame effective wavelengths on z grid.
    ztable_flux_scale : array, shape (n_z,)
        Geometric flux scale on z grid.
    z_grid : array, shape (n_z,)
        Redshift grid.
    z : float
        Target redshift.

    Returns
    -------
    ssp_phot : array, shape (n_met, n_age, n_filters)
        Interpolated SSP photometry.
    eff_waves_rest : array, shape (n_filters,)
        Interpolated rest-frame effective wavelengths.
    flux_scale : float
        Interpolated geometric factor.
    """
    z_clamped = jnp.clip(z, z_grid[0], z_grid[-1])
    idx = jnp.clip(jnp.searchsorted(z_grid, z_clamped) - 1, 0, len(z_grid) - 2)
    frac = (z_clamped - z_grid[idx]) / (z_grid[idx + 1] - z_grid[idx])

    ssp_phot = (1.0 - frac) * ztable_ssp_phot[idx] + frac * ztable_ssp_phot[idx + 1]
    eff_rest = (1.0 - frac) * ztable_eff_rest[idx] + frac * ztable_eff_rest[idx + 1]
    flux_scale = (1.0 - frac) * ztable_flux_scale[idx] + frac * ztable_flux_scale[idx + 1]

    return ssp_phot, eff_rest, flux_scale


@jax.jit
def interpolate_igm_ztable(igm_trans_table, z_grid, z):
    """Interpolate precomputed IGM transmission to a specific redshift.

    Parameters
    ----------
    igm_trans_table : array, shape (n_z, n_filters)
        Precomputed IGM transmission on z grid.
    z_grid : array, shape (n_z,)
        Redshift grid.
    z : float
        Target redshift.

    Returns
    -------
    igm_trans : array, shape (n_filters,)
        Interpolated IGM transmission.
    """
    z_clamped = jnp.clip(z, z_grid[0], z_grid[-1])
    idx = jnp.clip(jnp.searchsorted(z_grid, z_clamped) - 1, 0, len(z_grid) - 2)
    frac = (z_clamped - z_grid[idx]) / (z_grid[idx + 1] - z_grid[idx])
    return (1.0 - frac) * igm_trans_table[idx] + frac * igm_trans_table[idx + 1]
