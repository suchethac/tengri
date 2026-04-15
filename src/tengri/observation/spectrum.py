"""Pixel-level spectroscopic forward model.

Fits every spectral pixel directly, with an optional multiplicative
calibration polynomial to absorb flux-calibration uncertainties
(following Prospector / Johnson+2021).

Includes emission-line placement with instrument-resolution blending,
relevant for R < 1000 spectroscopy where close lines merge.

Also provides wavelength-dependent Line Spread Function (LSF) convolution
for instruments with variable spectral resolution (e.g., JWST NIRSpec PRISM).
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from tengri.utils.conversions import lnu_to_fnu

# ---------------------------------------------------------------------------
# SSP library spectral resolutions (velocity dispersion in km/s)
# ---------------------------------------------------------------------------
SSP_LIBRARY_RESOLUTIONS: dict[str, float] = {
    "miles": 70.0,  # R ~ 2500 at 5000 A, sigma ~ 70 km/s
    "c3k": 15.0,  # R ~ 10000, sigma ~ 15 km/s
    "fsps_default": 70.0,  # MILES-based (default FSPS)
}


# ---------------------------------------------------------------------------
# Speed of light
# ---------------------------------------------------------------------------
_C_KM_S = 299792.458  # km/s
_FWHM_TO_SIGMA = 2.3548200  # 2*sqrt(2*ln(2))


# ---------------------------------------------------------------------------
# Instrument resolution profiles
# ---------------------------------------------------------------------------


def nirspec_prism_resolution(wave_um: jnp.ndarray) -> jnp.ndarray:
    """JWST NIRSpec PRISM R(lambda) -- ranges from ~30 to ~300.

    Approximate from NIRSpec documentation (Jakobsen et al. 2022).
    R increases roughly linearly from 0.6 to 5.3 um.

    Parameters
    ----------
    wave_um : array
        Observed wavelength in microns.

    Returns
    -------
    array
        Spectral resolution R = lambda / delta_lambda at each wavelength.
    """
    return jnp.clip(30.0 + 55.0 * (wave_um - 0.6), 30.0, 330.0)


def nirspec_g140m_resolution(wave_um: jnp.ndarray) -> jnp.ndarray:
    """JWST NIRSpec G140M grating -- roughly constant R~1000.

    Parameters
    ----------
    wave_um : array
        Observed wavelength in microns.

    Returns
    -------
    array
        Spectral resolution R ~ 1000 at each wavelength.
    """
    return 1000.0 * jnp.ones_like(wave_um)


# ---------------------------------------------------------------------------
# Line Spread Function (LSF) convolution
# ---------------------------------------------------------------------------


def _resolution_to_sigma_kms(resolution: jnp.ndarray) -> jnp.ndarray:
    """Convert spectral resolution R to velocity dispersion sigma (km/s).

    sigma = c / (FWHM_TO_SIGMA * R)

    Parameters
    ----------
    resolution : array or scalar
        Spectral resolution R = lambda / delta_lambda.

    Returns
    -------
    array or scalar
        Velocity dispersion in km/s.
    """
    return _C_KM_S / (_FWHM_TO_SIGMA * resolution)


@jax.jit
def _apply_lsf_constant_r(
    spectrum: jnp.ndarray,
    wave_obs: jnp.ndarray,
    sigma_eff_kms: float,
) -> jnp.ndarray:
    """FFT convolution in log-wavelength space for constant R.

    This is equivalent to velocity_broaden but with the effective
    (library-subtracted) sigma.

    Parameters
    ----------
    spectrum : array, shape (n_pix,)
        Input spectrum.
    wave_obs : array, shape (n_pix,)
        Observed wavelength grid (Angstrom). Must be uniformly spaced.
    sigma_eff_kms : float
        Effective velocity dispersion in km/s (after library subtraction).

    Returns
    -------
    array, shape (n_pix,)
        Smoothed spectrum.
    """
    sigma_v = sigma_eff_kms / _C_KM_S
    dlnwave = jnp.log(wave_obs[1] / wave_obs[0])
    sigma_pix = sigma_v / dlnwave

    n = spectrum.shape[0]
    freq = jnp.fft.rfftfreq(n)
    kernel_ft = jnp.exp(-2.0 * jnp.pi**2 * sigma_pix**2 * freq**2)

    flux_ft = jnp.fft.rfft(spectrum)
    return jnp.fft.irfft(flux_ft * kernel_ft, n=n)


@partial(jax.jit, static_argnums=(3,))
def _apply_lsf_variable_r(
    spectrum: jnp.ndarray,
    wave_obs: jnp.ndarray,
    sigma_eff_kms: jnp.ndarray,
    n_bins: int = 16,
) -> jnp.ndarray:
    """Piecewise-constant LSF convolution for variable R.

    Splits the wavelength range into ``n_bins`` segments. Within each
    segment the mean effective sigma is used for an FFT convolution.
    The segments are blended with smooth (raised-cosine) overlap to
    avoid discontinuities. Accurate to ~1% for typical instrument
    profiles and fully differentiable.

    Parameters
    ----------
    spectrum : array, shape (n_pix,)
        Input spectrum.
    wave_obs : array, shape (n_pix,)
        Observed wavelength grid (Angstrom).
    sigma_eff_kms : array, shape (n_pix,)
        Effective velocity dispersion at each pixel (km/s).
    n_bins : int
        Number of piecewise-constant segments. More bins = better
        accuracy but more FFTs. 10-20 is usually sufficient.

    Returns
    -------
    array, shape (n_pix,)
        Smoothed spectrum.
    """
    n_pix = spectrum.shape[0]
    dlnwave = jnp.log(wave_obs[1] / wave_obs[0])
    freq = jnp.fft.rfftfreq(n_pix)
    flux_ft = jnp.fft.rfft(spectrum)

    # Pixel indices for bin edges (uniform split)
    bin_edges = jnp.linspace(0, n_pix, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1] - bin_edges[0]

    # Pixel index array
    pix_idx = jnp.arange(n_pix, dtype=jnp.float64)

    def _convolve_bin(carry, bin_center):
        """Convolve with the mean sigma for one bin, weighted by overlap."""
        # Pixel index of bin center
        center = bin_center
        half_w = bin_width * 0.75  # overlap region for blending

        # Smooth weight: raised cosine (1 at center, 0 outside)
        dist = jnp.abs(pix_idx - center) / half_w
        weight = jnp.where(dist < 1.0, 0.5 * (1.0 + jnp.cos(jnp.pi * dist)), 0.0)

        # Mean sigma in this bin (weighted by the bin window)
        bin_mask = jnp.where(
            jnp.abs(pix_idx - center) < bin_width,
            1.0,
            0.0,
        )
        sigma_mean = jnp.sum(sigma_eff_kms * bin_mask) / jnp.maximum(jnp.sum(bin_mask), 1.0)

        # FFT convolution with this sigma
        sigma_pix = (sigma_mean / _C_KM_S) / dlnwave
        kernel_ft = jnp.exp(-2.0 * jnp.pi**2 * sigma_pix**2 * freq**2)
        convolved = jnp.fft.irfft(flux_ft * kernel_ft, n=n_pix)

        return carry + weight * convolved, None

    # Accumulate weighted contributions from all bins
    result = jnp.zeros(n_pix)
    result, _ = jax.lax.scan(_convolve_bin, result, bin_centers)

    # Normalize by total weight at each pixel
    def _weight_bin(carry, bin_center):
        center = bin_center
        half_w = bin_width * 0.75
        dist = jnp.abs(pix_idx - center) / half_w
        weight = jnp.where(dist < 1.0, 0.5 * (1.0 + jnp.cos(jnp.pi * dist)), 0.0)
        return carry + weight, None

    total_weight, _ = jax.lax.scan(_weight_bin, jnp.zeros(n_pix), bin_centers)
    total_weight = jnp.maximum(total_weight, 1e-30)

    return result / total_weight


def apply_lsf(
    spectrum: jnp.ndarray,
    wave_obs: jnp.ndarray,
    resolution: jnp.ndarray | float,
    sigma_lib_kms: float = 0.0,
    n_bins: int = 16,
) -> jnp.ndarray:
    """Apply wavelength-dependent Line Spread Function with library resolution subtraction.

    The effective kernel width at each pixel is::

        sigma_eff(lambda) = sqrt(sigma_inst(lambda)^2 - sigma_lib^2)

    where ``sigma_inst = c / (2.355 * R(lambda))`` in km/s.

    If ``sigma_inst < sigma_lib`` at some wavelengths, no smoothing is
    applied there (cannot sharpen what is already broader).

    For constant R (scalar), this reduces to a single FFT convolution
    in log-wavelength space (fast). For variable R (array), uses a
    piecewise-constant bin approximation with smooth blending (~10-20
    FFTs, accurate to ~1%).

    Parameters
    ----------
    spectrum : array, shape (n_pix,)
        Input flux spectrum.
    wave_obs : array, shape (n_pix,)
        Observed wavelength grid (Angstrom). Should be uniformly spaced
        (or close to it) for FFT convolution accuracy.
    resolution : array or float
        Spectral resolution R(lambda). Scalar for constant R, or
        per-pixel array for wavelength-dependent resolution.
    sigma_lib_kms : float
        SSP library velocity resolution (km/s). Subtracted in
        quadrature. Use ``SSP_LIBRARY_RESOLUTIONS["miles"]`` for
        MILES-based SSP libraries. Default 0.0 (no subtraction).
    n_bins : int
        Number of bins for variable-R piecewise approximation.
        Ignored for constant R. Default 16.

    Returns
    -------
    array, shape (n_pix,)
        Spectrum convolved with the effective LSF.

    Examples
    --------
    Constant R::

        smoothed = apply_lsf(spec, wave, resolution=100.0)

    JWST NIRSpec PRISM (variable R)::

        R_prism = nirspec_prism_resolution(wave / 1e4)  # Angstrom -> um
        smoothed = apply_lsf(spec, wave, resolution=R_prism, sigma_lib_kms=70.0)
    """
    resolution = jnp.asarray(resolution)

    # Compute instrument sigma at each pixel
    sigma_inst_kms = _C_KM_S / (_FWHM_TO_SIGMA * resolution)

    # Subtract library resolution in quadrature; clamp to zero
    sigma_lib2 = sigma_lib_kms**2
    sigma_eff_kms = jnp.sqrt(jnp.maximum(sigma_inst_kms**2 - sigma_lib2, 0.0))

    # Dispatch based on whether R is constant or variable
    if resolution.ndim == 0:
        # Scalar R: single FFT convolution (fast path)
        return _apply_lsf_constant_r(spectrum, wave_obs, sigma_eff_kms)
    else:
        # Per-pixel R: piecewise-constant approximation
        return _apply_lsf_variable_r(spectrum, wave_obs, sigma_eff_kms, n_bins)


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

    # Scale using lnu_to_fnu conversion for consistency with cosmological flux formula
    flux_scale = lnu_to_fnu(1.0, dl_cm, redshift)
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
    sigma_v = sigma_km_s / _C_KM_S  # fractional velocity dispersion

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
        profile = jnp.exp(-0.5 * ((wave_out - lam_obs) / sigma_aa) ** 2) / (
            jnp.sqrt(2.0 * jnp.pi) * sigma_aa
        )

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
