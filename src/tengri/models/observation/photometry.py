"""Photometric filter convolution.

Computes observed flux densities by convolving the rest-frame SED
(redshifted) through filter transmission curves.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from tengri.utils.conversions import lnu_to_fnu
from tengri.utils.magnitudes import fnu_to_ab_mag


class FilterCurve(NamedTuple):
    """A photometric filter transmission curve.

    Attributes
    ----------
    wave : array
        Wavelength grid (Angstrom).
    trans : array
        Transmission at each wavelength (dimensionless, 0-1).
    name : str
        Filter name (e.g., "SDSS_r", "F200W").
    """

    wave: jnp.ndarray
    trans: jnp.ndarray
    name: str = ""


@jax.jit
def compute_flux_density(
    sed_rest: jnp.ndarray,
    wave_rest: jnp.ndarray,
    filter_wave: jnp.ndarray,
    filter_trans: jnp.ndarray,
    redshift: float,
    dl_cm: float,
) -> float:
    """Compute observed flux density through a single filter.

    f_nu = (1+z) / (4*pi*dL^2) * int[L_nu(lam/(1+z)) * T(lam) * lam dlam]
                                  / int[T(lam) * lam dlam]

    Parameters
    ----------
    sed_rest : array, shape (n_wave,)
        Rest-frame SED (Lsun/Hz or erg/s/Hz).
    wave_rest : array, shape (n_wave,)
        Rest-frame wavelength grid (Angstrom).
    filter_wave : array, shape (n_filt,)
        Filter wavelength grid (observed frame, Angstrom).
    filter_trans : array, shape (n_filt,)
        Filter transmission.
    redshift : float
        Source redshift.
    dl_cm : float
        Luminosity distance (cm).

    Returns
    -------
    float
        Observed flux density (erg/s/cm^2/Hz).
    """
    # Redshift the SED: observed wavelength = rest * (1+z)
    wave_obs = wave_rest * (1.0 + redshift)

    # Interpolate SED onto filter wavelength grid
    sed_on_filter = jnp.interp(filter_wave, wave_obs, sed_rest, left=0.0, right=0.0)

    # Filter-weighted integral: int(SED * T * lam dlam) / int(T * lam dlam)
    numerator = jnp.trapezoid(sed_on_filter * filter_trans * filter_wave, filter_wave)
    denominator = jnp.trapezoid(filter_trans * filter_wave, filter_wave)

    # Scale: (1+z) / (4 pi dL^2) for flux density using lnu_to_fnu conversion
    flux_scale = lnu_to_fnu(1.0, dl_cm, redshift)

    return flux_scale * numerator / jnp.maximum(denominator, 1e-30)


def pad_filters(filter_waves: list, filter_trans: list):
    """Pad variable-length filter arrays to a common length and stack.

    Parameters
    ----------
    filter_waves : list of arrays
        Wavelength grid per filter (different lengths).
    filter_trans : list of arrays
        Transmission per filter (same lengths as filter_waves).

    Returns
    -------
    fw_padded : array, shape (n_filters, max_len)
        Zero-padded filter wavelengths.
    ft_padded : array, shape (n_filters, max_len)
        Zero-padded filter transmissions.
    n_valid : array, shape (n_filters,), int
        Number of valid (non-padded) points per filter.
    """
    max_len = max(len(fw) for fw in filter_waves)
    fw_padded = jnp.zeros((len(filter_waves), max_len))
    ft_padded = jnp.zeros((len(filter_trans), max_len))
    n_valid = jnp.array([len(fw) for fw in filter_waves])
    for i, (fw, ft) in enumerate(zip(filter_waves, filter_trans)):
        n = len(fw)
        fw_padded = fw_padded.at[i, :n].set(fw)
        ft_padded = ft_padded.at[i, :n].set(ft)
    return fw_padded, ft_padded, n_valid


def _compute_flux_density_padded(
    sed_rest, wave_rest, filter_wave_padded, filter_trans_padded, redshift, dl_cm
):
    """Compute flux density for a single padded filter.

    Zero-padded entries contribute zero to the integral (trans=0),
    so no masking is needed.
    """
    wave_obs = wave_rest * (1.0 + redshift)
    sed_on_filter = jnp.interp(
        filter_wave_padded, wave_obs, sed_rest, left=0.0, right=0.0
    )
    numerator = jnp.trapezoid(
        sed_on_filter * filter_trans_padded * filter_wave_padded, filter_wave_padded
    )
    denominator = jnp.trapezoid(
        filter_trans_padded * filter_wave_padded, filter_wave_padded
    )
    flux_scale = lnu_to_fnu(1.0, dl_cm, redshift)
    return flux_scale * numerator / jnp.maximum(denominator, 1e-30)


def compute_flux_density_batch(
    sed_rest, wave_rest, fw_padded, ft_padded, redshift, dl_cm
):
    """Compute flux densities through all filters at once via vmap.

    Parameters
    ----------
    sed_rest : array, shape (n_wave,)
        Rest-frame SED.
    wave_rest : array, shape (n_wave,)
        Rest-frame wavelength (Angstrom).
    fw_padded : array, shape (n_filters, max_len)
        Zero-padded filter wavelengths (from ``pad_filters``).
    ft_padded : array, shape (n_filters, max_len)
        Zero-padded filter transmissions (from ``pad_filters``).
    redshift : float
        Source redshift.
    dl_cm : float
        Luminosity distance (cm).

    Returns
    -------
    array, shape (n_filters,)
        Observed flux density per filter.
    """
    return jax.vmap(
        _compute_flux_density_padded, in_axes=(None, None, 0, 0, None, None)
    )(sed_rest, wave_rest, fw_padded, ft_padded, redshift, dl_cm)


def compute_photometry(
    sed_rest: jnp.ndarray, wave_rest: jnp.ndarray, filters: list, redshift: float, dl_cm: float
) -> jnp.ndarray:
    """Compute photometry through multiple filters.

    Parameters
    ----------
    sed_rest : array, shape (n_wave,)
        Rest-frame SED.
    wave_rest : array, shape (n_wave,)
        Rest-frame wavelength (Angstrom).
    filters : list of FilterCurve
        Filter transmission curves.
    redshift : float
        Source redshift.
    dl_cm : float
        Luminosity distance (cm).

    Returns
    -------
    array, shape (n_filters,)
        Observed flux density per filter.
    """
    fluxes = []
    for filt in filters:
        f = compute_flux_density(sed_rest, wave_rest, filt.wave, filt.trans, redshift, dl_cm)
        fluxes.append(f)
    return jnp.array(fluxes)


@jax.jit
def ab_mag_from_flux(flux_cgs: jnp.ndarray) -> jnp.ndarray:
    """Convert flux density (erg/s/cm^2/Hz) to AB magnitude.

    This is a backward-compatibility wrapper that delegates to
    :func:`tengri.utils.magnitudes.fnu_to_ab_mag`.
    """
    return fnu_to_ab_mag(flux_cgs)
