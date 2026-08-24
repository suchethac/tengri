# SPDX-License-Identifier: BSD-3-Clause
"""Differentiable post-inference emission line measurements.

Provides JIT-compatible, differentiable equivalents of FastSpecFit's
``populate_emtable()`` outputs (FLUX, EW, SIGMAINT).

References
----------
.. [1] Moustakas, J., Scholte, D., Dey, B., Khederlarian, A., 2023,
       "FastSpecFit: Fast spectral synthesis and emission-line fitting
       of DESI spectra", Astrophysics Source Code Library,
       record ascl:2308.005.
       https://ui.adsabs.harvard.edu/abs/2023ascl.soft08005M
.. [2] Steidel, C. C., et al., 1996, ApJ, 462, L17.

"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from tengri.utils.physics_constants import C_KM_S as _C_KMS


@jax.jit
def compute_line_fluxes(
    amplitudes: jnp.ndarray,
    line_rest_waves: jnp.ndarray,
    redshift: float,
    sigma_kms: float,
    spectral_resolution: float,
) -> jnp.ndarray:
    r"""Integrated Gaussian emission line fluxes from amplitude parameters.

    Converts per-line Gaussian amplitudes into integrated fluxes by
    multiplying by the Gaussian normalization :math:`\sqrt{2\pi}\,\sigma_{\rm ang}`,
    where :math:`\sigma_{\rm ang}` combines intrinsic velocity broadening and
    the instrumental line-spread function (LSF) in quadrature.

    Parameters
    ----------
    amplitudes: array_like, shape (n_lines,)
        Gaussian amplitude of each line [erg/s/cm²/Angstrom].
    line_rest_waves: array_like, shape (n_lines,)
        Rest-frame vacuum wavelengths [Angstrom].
    redshift: float
        Galaxy redshift [dimensionless].
    sigma_kms: float
        Intrinsic line velocity dispersion [km/s].
    spectral_resolution: float
        Spectral resolution :math:`R = \lambda/\Delta\lambda` [dimensionless].

    Returns
    -------
    ndarray, shape (n_lines,)
        Integrated line fluxes [erg/s/cm²].

    Notes
    -----
    **JIT-compatible**: yes, differentiable through ``amplitudes``,
    ``sigma_kms``, and ``spectral_resolution``.

    The observed-frame line center is :math:`\lambda_z = \lambda_{\rm rest}(1+z)`.
    The combined angular sigma in Angstrom is

    .. math::

        \sigma_{\rm ang} = \lambda_z \sqrt{
            \left(\frac{\sigma_{\rm kms}}{c}\right)^2
            +
            \left(\frac{1}{2.355\,R}\right)^2
        }

    where the second term is the LSF contribution (Gaussian FWHM = :math:`\lambda/R`).
    The integrated flux is then

    .. math::

        F_i = A_i \sqrt{2\pi}\,\sigma_{{\rm ang},i}

    Implements the same calculation as FastSpecFit ``populate_emtable()`` [1]_;
    validated against its output.

    References
    ----------
    .. [1] Moustakas, J., Scholte, D., Dey, B., Khederlarian, A., 2023,
           "FastSpecFit: Fast spectral synthesis and emission-line fitting
           of DESI spectra", Astrophysics Source Code Library,
           record ascl:2308.005.
           https://ui.adsabs.harvard.edu/abs/2023ascl.soft08005M

    """
    z_waves = line_rest_waves * (1.0 + redshift)
    sigma_vel_sq = (sigma_kms / _C_KMS) ** 2
    sigma_lsf_sq = (1.0 / (2.355 * spectral_resolution)) ** 2
    sigma_ang = z_waves * jnp.sqrt(sigma_vel_sq + sigma_lsf_sq)
    return amplitudes * jnp.sqrt(2.0 * jnp.pi) * sigma_ang


@jax.jit
def compute_equivalent_widths(
    line_fluxes: jnp.ndarray,
    continuum_at_lines: jnp.ndarray,
    redshift: float,
) -> jnp.ndarray:
    r"""Rest-frame equivalent widths from line fluxes and continuum levels.

    Follows the convention of Steidel et al. (1996) [2]_ and FastSpecFit [1]_:
    EW is positive for emission.

    Parameters
    ----------
    line_fluxes: array_like, shape (n_lines,)
        Integrated emission line fluxes [erg/s/cm²].
    continuum_at_lines: array_like, shape (n_lines,)
        Continuum spectral flux density evaluated at each line center
        [erg/s/cm²/Angstrom].
    redshift: float
        Galaxy redshift [dimensionless].

    Returns
    -------
    ndarray, shape (n_lines,)
        Rest-frame equivalent widths [Angstrom].  Zero where
        ``continuum_at_lines <= 0``.

    Notes
    -----
    **JIT-compatible**: yes; differentiable through ``line_fluxes`` and
    ``continuum_at_lines``.

    The rest-frame EW is

    .. math::

        \mathrm{EW}_i = \frac{F_i}{f_{\lambda,i}\,(1 + z)}

    where :math:`f_{\lambda,i}` is the observed-frame continuum at line
    :math:`i` and the :math:`(1+z)` factor converts from observed to
    rest-frame wavelength interval.

    Pixels with non-positive continuum return ``EW = 0`` (guarded with
    ``jnp.where``).

    Implements the same calculation as FastSpecFit ``populate_emtable()`` [1]_;
    validated against its output.

    References
    ----------
    .. [1] Moustakas, J., Scholte, D., Dey, B., Khederlarian, A., 2023,
           "FastSpecFit: Fast spectral synthesis and emission-line fitting
           of DESI spectra", Astrophysics Source Code Library,
           record ascl:2308.005.
           https://ui.adsabs.harvard.edu/abs/2023ascl.soft08005M
    .. [2] Steidel, C. C., et al., 1996, ApJ, 462, L17.

    """
    ew_obs = line_fluxes / jnp.where(
        continuum_at_lines > 0.0, continuum_at_lines, jnp.ones_like(continuum_at_lines)
    )
    ew_rest = ew_obs / (1.0 + redshift)
    return jnp.where(continuum_at_lines > 0.0, ew_rest, jnp.zeros_like(ew_rest))


@jax.jit
def compute_line_moments(
    wave_obs: jnp.ndarray,
    flux_residual: jnp.ndarray,
    ivar: jnp.ndarray,
    line_obs_wave: float,
    sigma_kms: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Flux-weighted velocity centroid and dispersion from residual spectrum.

    Computes the first and second velocity moments of a continuum-subtracted
    emission line using a soft Gaussian weight kernel.  Unlike FastSpecFit's
    hard :math:`\pm n_\sigma` pixel window, the soft kernel is differentiable
    through ``sigma_kms`` and fully JIT-compatible.  Can be ``jax.vmap``-ed
    over lines for batch measurement.

    Parameters
    ----------
    wave_obs: array_like, shape (n_pix,)
        Observed-frame wavelength array [Angstrom].
    flux_residual: array_like, shape (n_pix,)
        Continuum-subtracted flux [erg/s/cm²/Angstrom].
    ivar: array_like, shape (n_pix,)
        Inverse variance [(erg/s/cm²/Angstrom)^-2].
    line_obs_wave: float
        Observed-frame line center wavelength [Angstrom].
    sigma_kms: float
        Gaussian sigma for the soft integration window [km/s].

    Returns
    -------
    v_centroid: ndarray, shape ()
        Flux-weighted velocity centroid relative to ``line_obs_wave`` [km/s].
    sigma_int: ndarray, shape ()
        Flux-weighted velocity dispersion [km/s].

    Notes
    -----
    **JIT-compatible**: yes; differentiable through ``sigma_kms``.

    Velocity of pixel :math:`k` relative to the line center:

    .. math::

        v_k = c \frac{\lambda_k - \lambda_0}{\lambda_0}

    Soft Gaussian integration weight:

    .. math::

        w_k = \exp\!\left(-\frac{v_k^2}{2\,\sigma_{\rm kms}^2}\right)

    Combined weight incorporating inverse variance:

    .. math::

        W_k = w_k \cdot \max(f_k,\, 0) \cdot \mathrm{ivar}_k

    First moment (centroid):

    .. math::

        v_{\rm cent} = \frac{\sum_k W_k\,v_k}{\sum_k W_k + \epsilon}

    Second moment (dispersion):

    .. math::

        \sigma_{\rm int} = \sqrt{
            \frac{\sum_k W_k\,(v_k - v_{\rm cent})^2}{\sum_k W_k + \epsilon}
        }

    where :math:`\epsilon = 10^{-30}` prevents division by zero in empty windows.

    Improvement over FastSpecFit [1]_: the hard :math:`\pm n_\sigma` window in
    FastSpecFit requires dynamic boolean indexing (``flux[mask]``), which is
    incompatible with JAX JIT due to its variable-length output.  The soft
    kernel retains the full pixel array at fixed shape, enabling gradient-based
    inference through line widths.

    References
    ----------
    .. [1] Moustakas, J., Scholte, D., Dey, B., Khederlarian, A., 2023,
           "FastSpecFit: Fast spectral synthesis and emission-line fitting
           of DESI spectra", Astrophysics Source Code Library,
           record ascl:2308.005.
           https://ui.adsabs.harvard.edu/abs/2023ascl.soft08005M

    """
    v_pix = _C_KMS * (wave_obs - line_obs_wave) / line_obs_wave
    soft_weight = jnp.exp(-0.5 * (v_pix / sigma_kms) ** 2)
    # Positive-flux and ivar weighting suppresses noise spikes
    combined_weight = soft_weight * jnp.clip(flux_residual, 0.0, None) * ivar

    eps = 1e-30
    weight_sum = jnp.sum(combined_weight) + eps

    v_centroid = jnp.sum(combined_weight * v_pix) / weight_sum
    variance = jnp.sum(combined_weight * (v_pix - v_centroid) ** 2) / weight_sum
    sigma_int = jnp.sqrt(jnp.clip(variance, 0.0, None))

    return v_centroid, sigma_int
