# SPDX-License-Identifier: BSD-3-Clause
r"""Analytic line-spread reference for the point-sampled spectrum (#3500).

The reference is the exact Gaussian convolution, in :math:`\ln\lambda`, of the
piecewise-linear interpolant of the rest-frame SED, sampled at each pixel with the
kernel width of that pixel. It involves no grid refinement and no ``n_bins``
piecewise approximation, so it is converged by construction.

Closed form. Between native nodes :math:`\lambda_a < \lambda_b` the SED is
:math:`f(\lambda) = p + q\,\lambda` with :math:`q = (f_b - f_a)/(\lambda_b - \lambda_a)` and
:math:`p = f_a - q\lambda_a`. With :math:`u = \ln\lambda`, :math:`\mu = \ln x_0`,
:math:`s = \sigma_{\ln\lambda}` at the pixel and :math:`\Phi` the standard normal cdf,

.. math::

    \int_{u_a}^{u_b} f\,\mathcal{N}(u;\mu,s)\,du
    = p\,\big[\Phi(\tfrac{u_b-\mu}{s}) - \Phi(\tfrac{u_a-\mu}{s})\big]
    + q\,e^{\mu+s^2/2}\big[\Phi(\tfrac{u_b-\mu-s^2}{s}) - \Phi(\tfrac{u_a-\mu-s^2}{s})\big],

using :math:`e^{u}\mathcal{N}(u;\mu,s) = e^{\mu+s^2/2}\mathcal{N}(u;\mu+s^2,s)`. The
width is :math:`s = \sqrt{\sigma_v^2 + \max(\sigma_{\rm inst}^2 - \sigma_{\rm lib}^2, 0)}/c`
with :math:`\sigma_{\rm inst} = c/(2.3548\,R)`, the same kernel :func:`apply_lsf`
documents, evaluated at the pixel.

Verification. The closed form agrees with brute-force quadrature on a fine
:math:`\ln\lambda` grid (4 x 10^6 points) to 5 x 10^-9 relative at pixels
x0 = 3750, 3934 and 5501 A for sigma_v = 50 and 300 km/s (measured 2e-9 to 5e-9 on
the full wNE fixture, 2026-11-15).

The reference is evaluated at the pixel, so it is one value per pixel and does not
depend on the piecewise ``n_bins`` decomposition of :func:`apply_lsf`; the
difference between that decomposition and this reference is the error #3504 tracks.

Notes
-----
Pure numpy with :func:`jax.scipy.special.ndtr` for the normal cdf (float64 under the
suite's ``jax_enable_x64``). The reference covers the constant-R path: a scalar
``resolution`` and a scalar or per-pixel library width.
"""

from __future__ import annotations

import numpy as np
from jax.scipy.special import ndtr

#: Speed of light [km/s].
C_KMS = 299792.458

#: Pixels evaluated per block, to bound the (pixels x segments) work array.
_BLOCK = 100


def sigma_ln_at_pixels(
    wave_obs,
    redshift: float,
    *,
    resolution: float,
    sigma_v_kms: float,
    sigma_lib_curve=None,
    sigma_lib_kms: float = 0.0,
    lsf_scale: float = 1.0,
) -> np.ndarray:
    r"""Gaussian width :math:`\sigma_{\ln\lambda}` of the line-spread kernel per pixel.

    Parameters
    ----------
    wave_obs : array_like, shape (n_pix,)
        Observed-frame pixel wavelengths [Angstrom].
    redshift : float
        Source redshift.
    resolution : float
        Constant resolving power :math:`R`.
    sigma_v_kms : float
        Intrinsic velocity dispersion [km/s], added in quadrature.
    sigma_lib_curve : tuple of array_like or None
        ``(ssp_wave, ssp_resolution_kms)`` [Angstrom, km/s], the library's own curve, read
        at the rest-frame pixel wavelength. ``None`` uses the flat ``sigma_lib_kms``.
    sigma_lib_kms : float
        Flat library dispersion [km/s], used when ``sigma_lib_curve`` is ``None``.
    lsf_scale : float
        Multiplier on the instrument width (1 for the fixed-LSF tests).

    Returns
    -------
    ndarray, shape (n_pix,)
        Kernel width in :math:`\ln\lambda`, dimensionless.
    """
    wave_obs = np.asarray(wave_obs, dtype=float)
    wave_rest = wave_obs / (1.0 + redshift)
    if sigma_lib_curve is None:
        sig_lib = np.full(wave_obs.shape, float(sigma_lib_kms))
    else:
        sig_lib = np.interp(wave_rest, *(np.asarray(a, dtype=float) for a in sigma_lib_curve))
    sig_inst = C_KMS / (2.3548 * float(resolution))
    var = np.maximum((float(lsf_scale) * sig_inst) ** 2 - sig_lib**2, 0.0)
    return np.sqrt(float(sigma_v_kms) ** 2 + var) / C_KMS


def _segment_integrals(u_a, u_b, mu, s, p, q):
    """Sum over lambda-linear segments of the kernel-weighted SED, one block of pixels."""
    i0 = ndtr((u_b[None, :] - mu) / s) - ndtr((u_a[None, :] - mu) / s)
    e = np.exp(mu + 0.5 * s**2)
    i1 = e * (ndtr((u_b[None, :] - mu - s**2) / s) - ndtr((u_a[None, :] - mu - s**2) / s))
    return np.sum(p[None, :] * i0 + q[None, :] * i1, axis=1)


def _segment_integrals_nufnu(u_a, u_b, mu, s, p, q, x0):
    """Sum over segments of B(lambda F)(l_c) / lambda_c, with F = p + q lambda on each segment.

    lambda F = p e^u + q e^{2u}, so the segment moments are e^{k^2 s^2/2} times the Phi
    differences with shift k s^2 (k = 1, 2), and dividing by lambda_c = x0 leaves
    p e^{s^2/2} D_1 + q x0 e^{2 s^2} D_2. This integrates the linear F, not an interpolant of
    lambda F.
    """
    d1 = ndtr((u_b[None, :] - mu - s**2) / s) - ndtr((u_a[None, :] - mu - s**2) / s)
    d2 = ndtr((u_b[None, :] - mu - 2.0 * s**2) / s) - ndtr((u_a[None, :] - mu - 2.0 * s**2) / s)
    return np.sum(
        p[None, :] * np.exp(0.5 * s**2) * d1 + q[None, :] * x0 * np.exp(2.0 * s**2) * d2, axis=1
    )


def convolve_piecewise_linear(
    wave_nodes, sed_nodes, wave_obs, redshift, width_ln, *, quantity="fnu"
) -> np.ndarray:
    r"""Exact :math:`\ln\lambda`-Gaussian convolution of the lambda-linear SED, per pixel.

    Parameters
    ----------
    wave_nodes : array_like, shape (n_node,)
        Rest-frame native wavelengths, ascending [Angstrom].
    sed_nodes : array_like, shape (n_node,)
        SED at the native nodes [erg/s/Hz].
    wave_obs : array_like, shape (n_pix,)
        Observed-frame pixel wavelengths [Angstrom].
    redshift : float
        Source redshift; pixels are evaluated at
        :math:`\lambda_{\rm rest} = \lambda_{\rm obs}/(1+z)`.
    width_ln : array_like, shape (n_pix,)
        Kernel width :math:`\sigma_{\ln\lambda}` at each pixel, from
        :func:`sigma_ln_at_pixels`.

    Returns
    -------
    ndarray, shape (n_pix,)
        Convolved SED at the pixel rest wavelengths [erg/s/Hz].
    """
    wn = np.asarray(wave_nodes, dtype=float)
    fn = np.asarray(sed_nodes, dtype=float)
    if quantity not in ("fnu", "nufnu"):
        raise ValueError(f"quantity must be 'fnu' or 'nufnu', got {quantity!r}")
    x0 = np.asarray(wave_obs, dtype=float) / (1.0 + redshift)
    s_all = np.asarray(width_ln, dtype=float)
    lam_a, lam_b = wn[:-1], wn[1:]
    f_a, f_b = fn[:-1], fn[1:]
    q = (f_b - f_a) / (lam_b - lam_a)
    p = f_a - q * lam_a
    u_a, u_b = np.log(lam_a), np.log(lam_b)
    out = np.empty(x0.size)
    for k0 in range(0, x0.size, _BLOCK):
        sl = slice(k0, min(k0 + _BLOCK, x0.size))
        mu = np.log(x0[sl])[:, None]
        s = s_all[sl][:, None]
        if quantity == "nufnu":
            out[sl] = _segment_integrals_nufnu(u_a, u_b, mu, s, p, q, x0[sl][:, None])
        else:
            out[sl] = _segment_integrals(u_a, u_b, mu, s, p, q)
    return out


def analytic_reference(
    wave_nodes,
    sed_nodes,
    wave_obs,
    redshift: float,
    *,
    dimming,
    resolution: float,
    sigma_v_kms: float,
    sigma_lib_curve=None,
    sigma_lib_kms: float = 0.0,
    lsf_scale: float = 1.0,
) -> np.ndarray:
    r"""Analytic reference spectrum, in the units of the point-sampled prediction.

    Parameters
    ----------
    wave_nodes, sed_nodes : array_like
        Rest-frame native grid and SED, as in :func:`convolve_piecewise_linear`.
    wave_obs : array_like, shape (n_pix,)
        Observed-frame pixel wavelengths [Angstrom].
    redshift : float
        Source redshift.
    dimming : array_like, shape (n_pix,)
        Per-pixel cosmological factor that maps the SED to the flux density, taken as the
        resolution-free prediction divided by the interpolated SED (constant in practice).
    resolution, sigma_v_kms, sigma_lib_curve, sigma_lib_kms, lsf_scale
        Kernel parameters, as in :func:`sigma_ln_at_pixels`.

    Returns
    -------
    ndarray, shape (n_pix,)
        Reference flux density per pixel [erg/s/cm^2/Hz].
    """
    width = sigma_ln_at_pixels(
        wave_obs,
        redshift,
        resolution=resolution,
        sigma_v_kms=sigma_v_kms,
        sigma_lib_curve=sigma_lib_curve,
        sigma_lib_kms=sigma_lib_kms,
        lsf_scale=lsf_scale,
    )
    conv = convolve_piecewise_linear(wave_nodes, sed_nodes, wave_obs, redshift, width)
    return np.asarray(dimming, dtype=float) * conv


def check_igm_is_unity(igm_at_pixels, tol: float = 1e-9) -> None:
    """Raise if the IGM factor is not 1 at the pixels, which the reference assumes (#3500).

    Parameters
    ----------
    igm_at_pixels : array_like, shape (n_pix,)
        IGM transmission sampled at the pixel rest wavelengths.
    tol : float
        Largest allowed deviation from 1.

    Raises
    ------
    AssertionError
        If any pixel deviates from 1 by more than ``tol``.
    """
    dev = float(np.max(np.abs(np.asarray(igm_at_pixels, dtype=float) - 1.0)))
    assert dev <= tol, f"IGM transmission departs from 1 by {dev:.3e} in the window"
