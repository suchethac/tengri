"""Pixel-level spectroscopic forward model.

Fits every spectral pixel directly, with an optional multiplicative
calibration polynomial to absorb flux-calibration uncertainties
(following Prospector / Johnson+2021).

Includes emission-line placement with instrument-resolution blending,
relevant for R < 1000 spectroscopy where close lines merge.
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


# ---------------------------------------------------------------------------
# Speed of light in Angstrom/s (for frequency conversions)
# ---------------------------------------------------------------------------
_C_AA_PER_S = 2.99792458e18


@jax.jit
def blend_emission_lines(
    line_wavelengths: jnp.ndarray,
    line_luminosities: jnp.ndarray,
    spectral_resolution: float,
    wave_out: jnp.ndarray,
    redshift: float = 0.0,
) -> jnp.ndarray:
    """Place emission lines onto a wavelength grid, blending by instrument resolution.

    Each line is represented as a Gaussian whose width is set by the
    instrument's spectral resolution R = lambda / delta_lambda. Lines
    closer than delta_lambda are effectively blended. The output is in
    Lsun/Hz, ready to be added to a continuum SED.

    Vectorized over all lines simultaneously using ``jax.vmap`` for
    efficient GPU/TPU execution.

    Parameters
    ----------
    line_wavelengths : array, shape (n_lines,)
        Rest-frame line wavelengths (Angstrom).
    line_luminosities : array, shape (n_lines,)
        Line luminosities (Lsun). Total integrated luminosity per line.
    spectral_resolution : float
        Instrument spectral resolution R = lambda / delta_lambda.
        Typical values: R ~ 100 for photometry, R ~ 1000 for low-res
        spectroscopy, R ~ 5000 for medium-res.
    wave_out : array, shape (n_pix,)
        Output wavelength grid (Angstrom, observed frame).
    redshift : float, optional
        Source redshift. Default is 0.0.

    Returns
    -------
    array, shape (n_pix,)
        Emission-line spectrum in Lsun/Hz on the output grid.
        Add to a continuum SED (also in Lsun/Hz) before applying
        cosmological dimming.

    Notes
    -----
    The Gaussian FWHM at each line is FWHM = lambda_obs / R, giving
    sigma = lambda_obs / (2.355 * R). The profile is normalized to
    integrate to 1 in wavelength space. To convert from Lsun (total
    line luminosity) to Lsun/Hz (spectral density), we divide by the
    frequency width delta_nu = c * sigma / lambda_obs^2.
    """
    def _single_line(lam_rest, lum):
        """Compute Gaussian profile for one line.

        Parameters
        ----------
        lam_rest : scalar
            Rest-frame wavelength (Angstrom).
        lum : scalar
            Line luminosity (Lsun).

        Returns
        -------
        array, shape (n_pix,)
            Contribution to the spectrum (Lsun/Hz).
        """
        lam_obs = lam_rest * (1.0 + redshift)
        sigma_aa = lam_obs / (2.355 * spectral_resolution)

        # Gaussian profile normalized in wavelength space: integral = 1
        profile = jnp.exp(
            -0.5 * ((wave_out - lam_obs) / sigma_aa) ** 2
        ) / (jnp.sqrt(2.0 * jnp.pi) * sigma_aa)

        # Convert Lsun (integrated over wavelength) to Lsun/Hz:
        # delta_nu = c / lam_obs^2 * sigma_aa  (characteristic freq width)
        # profile_nu = lum * profile_lambda / delta_nu
        # But more directly: profile is normalized in lambda, so
        # L_lambda = lum * profile  [Lsun/AA]
        # L_nu = L_lambda * lambda^2 / c  [Lsun/Hz]
        # At each pixel: L_nu = lum * profile * wave_out^2 / c
        return lum * profile * wave_out**2 / _C_AA_PER_S

    # Vectorize over all lines and sum
    all_profiles = jax.vmap(_single_line)(line_wavelengths, line_luminosities)
    return jnp.sum(all_profiles, axis=0)
