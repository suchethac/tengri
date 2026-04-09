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
