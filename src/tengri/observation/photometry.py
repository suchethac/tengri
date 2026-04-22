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
    """Photometric filter transmission curve.

    Represents a single broad-band photometric filter via its wavelength-dependent
    transmission profile. Used to convolve SEDs and compute observed flux densities.

    Attributes
    ----------
    wave : array, shape (n_wave,)
        Wavelength grid [Ångstrom]. Should be at least 10 points spanning
        the transmission curve from near zero to near zero.
    trans : array, shape (n_wave,)
        Transmission at each wavelength (dimensionless, 0.0–1.0).
        Typically peaked at 1.0 and falls to 0 at the filter edges.
    name : str, optional
        Filter identifier (e.g., ``"sdss_r"``, ``"jwst_f200w"``, ``"hsc_i"``).
        Used for diagnostic output and filter registry lookups. Default empty string.

    Notes
    -----
    **Standard sources**:

    - Optical/NIR: Spanish Virtual Observatory (SVO) Filter Profile Service
    - JWST: STScI filter definitions (via astropy.io.fits)
    - Custom: User-provided arrays

    **Filter conventions**:

    - Transmission is not flux-normalized (raw instrumental response)
    - Wavelength grid should be uniform or fine enough to resolve structure
    - Outside [wave[0], wave[-1]], transmission is assumed zero

    See Also
    --------
    load_filter_set : Load filter set from SVO database.
    compute_flux_density : Convolve SED through this filter.
    pad_filters : Stack variable-length filter arrays.

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
    r"""Compute observed flux density through a single photometric filter.

    Evaluates the rest-frame SED at observed wavelengths (redshifted), convolves
    with filter transmission, and integrates to produce a single observed flux
    density. Uses the standard flux-weighted approach: flux is the filter-weighted
    integral of redshifted SED, normalized by the filter response integral,
    and scaled by the luminosity distance and (1+z) redshift factor.

    Parameters
    ----------
    sed_rest : array, shape (n_wave,)
        Rest-frame spectral luminosity density [erg/s/Hz] or [L☉/Hz] at
        rest-frame wavelengths.
    wave_rest : array, shape (n_wave,)
        Rest-frame wavelength grid [Ångstrom].
    filter_wave : array, shape (n_filt,)
        Filter wavelength grid [Ångstrom], already in observed frame
        (redshifted by the model).
    filter_trans : array, shape (n_filt,)
        Filter transmission at each wavelength (dimensionless, 0–1).
    redshift : float
        Source redshift z. Used to redshift rest-frame wavelengths and
        scale flux by (1+z) factor.
    dl_cm : float
        Luminosity distance [cm]. Typically from :func:`luminosity_distance`.

    Returns
    -------
    flux_density : float
        Observed flux density [erg/s/cm²/Hz] in the AB system.

    Notes
    -----
    **JIT-compatible**: yes — all operations are ``jnp`` primitives.
    Safe to call inside :func:`jax.jit`.

    **Gradient-safe**: yes — differentiable w.r.t. all inputs except
    filter curves (considered fixed).

    **Filter convolution formula**:

    .. math::

        f_\\nu^{\\rm obs} = \\frac{1+z}{4\\pi d_L^2} \\;
        \\frac{\\int L_\\nu(\\lambda_\\mathrm{rest}) T(\\lambda_\\mathrm{obs})
               \\lambda_\\mathrm{obs} \\, d\\lambda_\\mathrm{obs}}
             {\\int T(\\lambda_\\mathrm{obs}) \\lambda_\\mathrm{obs}
              \\, d\\lambda_\\mathrm{obs}}

    where :math:`L_\\nu` is the rest-frame SED [erg/s/Hz],
    :math:`T(\\lambda_\\mathrm{obs})` is the filter transmission,
    :math:`z` is redshift, and :math:`d_L` is luminosity distance.
    This convention matches DSPS and is standard in SED fitting.

    **Interpolation**: The rest-frame SED is evaluated on the observed-frame
    filter grid via linear interpolation (``jnp.interp``). This assumes
    the filter grid adequately samples the SED; under-sampling (rare filters)
    may underestimate flux slightly.

    **Edge handling**: :math:`L_\\nu = 0` outside the SED wavelength domain
    (set via ``left=0.0, right=0.0``).

    See Also
    --------
    FilterCurve : Photometric filter transmission curve.
    pad_filters : Stack variable-length filter arrays.

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
        Wavelength grids per filter (different lengths, units [Angstrom]).
    filter_trans : list of arrays
        Transmission per filter (same lengths as filter_waves, dimensionless).

    Returns
    -------
    fw_padded : ndarray, shape (n_filters, max_len)
        Zero-padded filter wavelengths [Angstrom].
    ft_padded : ndarray, shape (n_filters, max_len)
        Zero-padded filter transmissions (dimensionless).
    n_valid : ndarray, shape (n_filters,), dtype int
        Number of valid (non-padded) points per filter.

    Notes
    -----
    Not JIT-compatible (uses Python list operations and loops).

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

    Parameters
    ----------
    sed_rest : array, shape (n_wave,)
        Rest-frame SED [erg/s/Hz].
    wave_rest : array, shape (n_wave,)
        Rest-frame wavelength [Angstrom].
    filter_wave_padded : array, shape (max_len,)
        Zero-padded filter wavelengths [Angstrom].
    filter_trans_padded : array, shape (max_len,)
        Zero-padded filter transmission (dimensionless).
    redshift : float
        Source redshift.
    dl_cm : float
        Luminosity distance [cm].

    Returns
    -------
    flux_density : float
        Observed flux density [erg/s/cm²/Hz].

    Notes
    -----
    Zero-padded entries contribute zero to the integral (trans=0),
    so no masking is needed. Private helper for compute_flux_density_batch.

    """
    wave_obs = wave_rest * (1.0 + redshift)
    sed_on_filter = jnp.interp(filter_wave_padded, wave_obs, sed_rest, left=0.0, right=0.0)
    numerator = jnp.trapezoid(
        sed_on_filter * filter_trans_padded * filter_wave_padded, filter_wave_padded
    )
    denominator = jnp.trapezoid(filter_trans_padded * filter_wave_padded, filter_wave_padded)
    flux_scale = lnu_to_fnu(1.0, dl_cm, redshift)
    return flux_scale * numerator / jnp.maximum(denominator, 1e-30)


def compute_flux_density_batch(sed_rest, wave_rest, fw_padded, ft_padded, redshift, dl_cm):
    """Compute flux densities through all filters at once via vmap.

    Parameters
    ----------
    sed_rest : array, shape (n_wave,)
        Rest-frame SED [erg/s/Hz].
    wave_rest : array, shape (n_wave,)
        Rest-frame wavelength [Angstrom].
    fw_padded : array, shape (n_filters, max_len)
        Zero-padded filter wavelengths [Angstrom] (from ``pad_filters``).
    ft_padded : array, shape (n_filters, max_len)
        Zero-padded filter transmissions (dimensionless, from ``pad_filters``).
    redshift : float
        Source redshift.
    dl_cm : float
        Luminosity distance [cm].

    Returns
    -------
    ndarray, shape (n_filters,)
        Observed flux density per filter [erg/s/cm²/Hz].

    Notes
    -----
    JIT-compatible: yes — vmapped over filters. Gradient-safe: yes.

    """
    return jax.vmap(_compute_flux_density_padded, in_axes=(None, None, 0, 0, None, None))(
        sed_rest, wave_rest, fw_padded, ft_padded, redshift, dl_cm
    )


def compute_photometry(
    sed_rest: jnp.ndarray, wave_rest: jnp.ndarray, filters: list, redshift: float, dl_cm: float
) -> jnp.ndarray:
    """Compute photometry through multiple filters.

    Convenience wrapper that calls :func:`compute_flux_density` for each filter
    in sequence and returns stacked flux densities.

    Parameters
    ----------
    sed_rest : array, shape (n_wave,)
        Rest-frame SED [erg/s/Hz].
    wave_rest : array, shape (n_wave,)
        Rest-frame wavelength [Angstrom].
    filters : list of FilterCurve
        Filter transmission curves to convolve.
    redshift : float
        Source redshift [dimensionless].
    dl_cm : float
        Luminosity distance [cm].

    Returns
    -------
    ndarray, shape (n_filters,)
        Observed flux density per filter [erg/s/cm²/Hz].

    Notes
    -----
    **JIT-compatible**: no — uses Python list comprehension and loops.
    For JIT-compiled photometry over many filters, use
    :func:`compute_flux_density_batch` with padded filter arrays.

    **Gradient-safe**: yes — each call to :func:`compute_flux_density`
    is differentiable w.r.t. ``sed_rest``.

    See Also
    --------
    compute_flux_density : Single filter convolution (JIT-compatible).
    compute_flux_density_batch : Vectorized convolution via vmap.
    """
    fluxes = []
    for filt in filters:
        f = compute_flux_density(sed_rest, wave_rest, filt.wave, filt.trans, redshift, dl_cm)
        fluxes.append(f)
    return jnp.array(fluxes)


@jax.jit
def ab_mag_from_flux(flux_cgs: jnp.ndarray) -> jnp.ndarray:
    """Convert flux density to AB magnitude.

    Parameters
    ----------
    flux_cgs : array, shape (n_band,)
        Flux density [erg/s/cm²/Hz].

    Returns
    -------
    ndarray, shape (n_band,)
        AB magnitude (dimensionless).

    Notes
    -----
    JIT-compatible: yes. Gradient-safe: yes.
    Delegates to :func:`tengri.utils.magnitudes.fnu_to_ab_mag`.

    """
    return fnu_to_ab_mag(flux_cgs)
