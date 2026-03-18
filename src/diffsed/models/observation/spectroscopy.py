"""Pixel-level spectroscopic forward model.

Fits every spectral pixel directly, with an optional multiplicative
calibration polynomial to absorb flux-calibration uncertainties
(following Prospector / Johnson+2021).
"""

import jax
import jax.numpy as jnp


@jax.jit
def compute_spectrum(
    sed_rest: jnp.ndarray,
    wave_rest: jnp.ndarray,
    wave_obs: jnp.ndarray,
    redshift: float,
    dl_cm: float,
) -> jnp.ndarray:
    """Compute observed spectrum at pixel wavelengths.

    f(lambda_j) = (1+z) / (4*pi*dL^2) * L_att(lambda_j / (1+z))

    Parameters
    ----------
    sed_rest : array, shape (n_wave,)
        Rest-frame attenuated SED (Lsun/Hz or erg/s/Hz).
    wave_rest : array, shape (n_wave,)
        Rest-frame wavelength grid (Angstrom).
    wave_obs : array, shape (n_pix,)
        Observed-frame wavelength at each spectral pixel (Angstrom).
    redshift : float
        Source redshift.
    dl_cm : float
        Luminosity distance (cm).

    Returns
    -------
    array, shape (n_pix,)
        Model flux at each spectral pixel (erg/s/cm^2/Hz).
    """
    # Map observed wavelengths to rest-frame
    wave_rest_query = wave_obs / (1.0 + redshift)

    # Interpolate rest-frame SED
    sed_at_pixels = jnp.interp(wave_rest_query, wave_rest, sed_rest, left=0.0, right=0.0)

    flux_scale = (1.0 + redshift) / (4.0 * jnp.pi * dl_cm**2)
    return flux_scale * sed_at_pixels


@jax.jit
def velocity_broaden(
    flux: jnp.ndarray,
    wave: jnp.ndarray,
    sigma_km_s: float,
) -> jnp.ndarray:
    """Broaden a spectrum by stellar velocity dispersion.

    Convolves with a Gaussian in log-wavelength space (equivalent to
    velocity space: Δv/c = Δln(λ)). Uses FFT convolution for speed.

    Parameters
    ----------
    flux : array, shape (n_pix,)
        Input spectrum on a uniform-in-wavelength grid.
    wave : array, shape (n_pix,)
        Wavelength grid (Angstrom). Must be uniformly spaced.
    sigma_km_s : float
        Velocity dispersion in km/s. Typical range: 50-300 km/s.

    Returns
    -------
    array, shape (n_pix,)
        Broadened spectrum.
    """
    c_km_s = 299792.458  # speed of light in km/s
    sigma_v = sigma_km_s / c_km_s  # fractional velocity dispersion

    # Pixel scale in log-wavelength
    dlnwave = jnp.log(wave[1] / wave[0])  # assumes uniform spacing

    # Gaussian kernel width in pixels
    sigma_pix = sigma_v / dlnwave

    # Build Gaussian kernel in Fourier space (faster than real-space)
    n = len(flux)
    freq = jnp.fft.rfftfreq(n)
    kernel_ft = jnp.exp(-2.0 * jnp.pi**2 * sigma_pix**2 * freq**2)

    # FFT convolution
    flux_ft = jnp.fft.rfft(flux)
    broadened = jnp.fft.irfft(flux_ft * kernel_ft, n=n)

    return broadened


@jax.jit
def chebyshev_calibration(
    wave_obs: jnp.ndarray, coeffs: jnp.ndarray, wave_min: float, wave_max: float
) -> jnp.ndarray:
    """Multiplicative Chebyshev calibration polynomial.

    C(lambda) = sum_k c_k * T_k(x)

    where x = (2*lambda - lambda_min - lambda_max) / (lambda_max - lambda_min)
    maps the wavelength range to [-1, 1], and T_k are Chebyshev polynomials
    of the first kind. c_0 is fixed to 1.

    Parameters
    ----------
    wave_obs : array, shape (n_pix,)
        Observed wavelengths (Angstrom).
    coeffs : array, shape (K,)
        Chebyshev coefficients c_1, ..., c_K (c_0 = 1 is implicit).
    wave_min, wave_max : float
        Wavelength range for normalization.

    Returns
    -------
    array, shape (n_pix,)
        Multiplicative calibration factor at each pixel.
    """
    x = (2.0 * wave_obs - wave_min - wave_max) / (wave_max - wave_min)

    # Start with c_0 = 1 (T_0 = 1)
    result = jnp.ones_like(x)

    # Add higher-order terms: T_1(x) = x, T_2(x) = 2x^2 - 1, etc.
    t_prev = jnp.ones_like(x)  # T_0
    t_curr = x  # T_1

    for k in range(len(coeffs)):
        if k == 0:
            result = result + coeffs[0] * t_curr  # c_1 * T_1
        else:
            t_next = 2.0 * x * t_curr - t_prev
            t_prev = t_curr
            t_curr = t_next
            result = result + coeffs[k] * t_curr  # c_{k+1} * T_{k+1}

    return result
