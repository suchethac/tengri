# SPDX-License-Identifier: BSD-3-Clause
"""Quadrature on GRAHSP's own grid, for the published ``L_*`` luminosities.

A published luminosity integrates the SED the component emits, on a grid the component
owns, never on the caller's wavelength array. The grid is the union of:

* a log-uniform base grid over the emitted support, :data:`BASE_PER_DECADE` points per decade;
* the hard edges of the pieces (the X-ray floor, the Lyman limit, the Balmer convolution
  threshold, the SMC attenuation break);
* the tabulated template nodes of the selected pieces (their linear interpolants are
  non-smooth at every node);
* windows of +/- :data:`WINDOW_SIGMA` sigma around each narrow feature (Gaussian emission
  lines, the Balmer edge, the silicate feature), so every feature is resolved.

Each segment between consecutive breakpoints is integrated by a :data:`GL_ORDER`-point
Gauss-Legendre rule in ln(lambda). Within a segment every piece is smooth, so the rule
converges; the grid depends only on the component's configuration and its parameters, not
on the caller.

Notes
-----
**JIT-compatible**: yes. The breakpoints are a sorted concatenation of arrays whose shapes
are fixed by the configuration; the feature windows move with the traced linewidth through
the values only. **Gradient-safe**: yes, the sort is differentiable in its values.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
from jax import Array

from tengri.components.agn.grahsp.bolometric import LYMAN_LIMIT_NM
from tengri.utils.host_array import device_table

__all__ = [
    "BASE_PER_DECADE",
    "GL_ORDER",
    "PUBLICATION_HI_NM",
    "WINDOW_SIGMA",
    "bolometric_unit_nm",
    "gauss_legendre_segments",
    "linewidth_sigma_nm",
    "publication_breakpoints",
    "window_edges",
]

GL_ORDER: int = 4
"""Gauss-Legendre points per segment."""

BASE_PER_DECADE: int = 20
"""Base-grid points per decade of wavelength."""

WINDOW_SIGMA: float = 6.0
"""Half-width of a feature window in standard deviations."""

PUBLICATION_HI_NM: float = 1.0e14
"""Long-wavelength edge of the publication grid [nm].

The SMC-broken BBB falls as lambda**(alpha_2 - 1) beyond its cutoff, so a shallow
``plslope`` leaves a long tail: the luminosity above 91.2 nm that lies past an edge, over the
corners of the declared prior (plslope -2.7 to -1.0, bend 50-150 nm, width 0.1-10, cutoff
10 um), is at most 1.44e-4 past 1e8 nm and **1.9e-10 past 1e14 nm** (measured, with the
integral taken to 1e40 nm); the worst corner is plslope = -1 with bend width 10. The grid
therefore runs to 1e14 nm and the published ``L_agn_bol`` and ``L_agn_absorbed`` and the disc
block normalization are low by at most that fraction.
"""

_WINDOW_OFFSETS: np.ndarray = np.arange(-WINDOW_SIGMA, WINDOW_SIGMA + 1.0)
"""Window edge offsets in sigma units: 13 edges per feature (-6 sigma .. +6 sigma)."""

_FWHM_TO_SIGMA: float = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))
"""FWHM -> sigma for a Gaussian, 1 / (2 sqrt(2 ln 2))."""

_C_KMS: float = 299792.458
"""Speed of light [km/s]."""


def linewidth_sigma_nm(center_nm: Array, linewidth_kms: Array) -> Array:
    """Gaussian sigma [nm] of a feature whose FWHM is ``linewidth_kms`` at ``center_nm``.

    Parameters
    ----------
    center_nm : array_like, shape (n_feature,)
        Feature centers [nm].
    linewidth_kms : float
        FWHM of the features [km/s].

    Returns
    -------
    ndarray, shape (n_feature,)
        Standard deviation [nm], the same convention as :func:`gaussian_lines`.
    """
    return jnp.asarray(center_nm) * (linewidth_kms / _C_KMS) * _FWHM_TO_SIGMA


def window_edges(center_nm: Array, sigma_nm: Array) -> Array:
    """Breakpoints at +/- :data:`WINDOW_SIGMA` sigma around each feature center.

    Parameters
    ----------
    center_nm : array_like, shape (n_feature,)
        Feature centers [nm].
    sigma_nm : array_like, shape (n_feature,) or scalar
        Gaussian sigma of each feature [nm].

    Returns
    -------
    ndarray, shape (n_feature * 13,)
        Window edges [nm], unsorted.
    """
    center = jnp.atleast_1d(jnp.asarray(center_nm))
    sigma = jnp.broadcast_to(jnp.asarray(sigma_nm), center.shape)
    offsets = device_table(_WINDOW_OFFSETS)
    return (center[:, None] + offsets[None, :] * sigma[:, None]).ravel()


def publication_breakpoints(
    lo_nm: float,
    hi_nm: float,
    static_nm,
    arrays,
) -> Array:
    """Sorted, clipped breakpoints of the publication grid [nm].

    Parameters
    ----------
    lo_nm, hi_nm : float
        Support of the published integrals [nm]; both are breakpoints.
    static_nm : sequence of float
        Hard edges of the pieces [nm], fixed by the component.
    arrays : sequence of array_like
        Template nodes and feature-window edges [nm]; traced where they depend on a parameter.

    Returns
    -------
    ndarray, shape (n_breaks,)
        Ascending breakpoints in ``[lo_nm, hi_nm]``; duplicates give zero-width segments,
        which carry zero weight.
    """
    n_points = round(BASE_PER_DECADE * math.log10(hi_nm / lo_nm)) + 1
    base = np.logspace(math.log10(lo_nm), math.log10(hi_nm), n_points)
    fixed = np.concatenate([[lo_nm, hi_nm], base, np.asarray(static_nm, dtype=float)])
    parts = [jnp.asarray(fixed)] + [jnp.ravel(jnp.asarray(a)) for a in arrays]
    breaks = jnp.sort(jnp.concatenate(parts))
    return jnp.clip(breaks, lo_nm, hi_nm)


def gauss_legendre_segments(breaks_nm: Array) -> tuple[Array, Array]:
    """Gauss-Legendre nodes and wavelength weights on each segment of ``breaks_nm``.

    Parameters
    ----------
    breaks_nm : array_like, shape (n_breaks,)
        Ascending breakpoints [nm], all positive.

    Returns
    -------
    lam_nm : ndarray, shape (n_breaks - 1) * GL_ORDER
        Quadrature wavelengths [nm].
    weight_nm : ndarray, shape (n_breaks - 1) * GL_ORDER
        Weights in d(lambda) [nm], so ``sum(weight_nm * f(lam_nm))`` is ``int f dlambda``.

    Notes
    -----
    Each segment is integrated in ln(lambda): ``d lambda = lambda d ln(lambda)``.
    """
    x, w = np.polynomial.legendre.leggauss(GL_ORDER)
    lo = jnp.log(breaks_nm[:-1])
    hi = jnp.log(breaks_nm[1:])
    half = 0.5 * (hi - lo)[:, None]
    log_lam = lo[:, None] + half * (x[None, :] + 1.0)
    lam = jnp.exp(log_lam)
    weight = half * w[None, :] * lam
    return lam.ravel(), weight.ravel()


def bolometric_unit_nm(unit_fn, node_arrays=()) -> Array:
    r"""Bolometric integral above the Lyman limit of a unit-l5100 disc spectrum.

    The one converged normalization integral of the GRAHSP disc. Integrated on the
    component's own publication grid (segmented Gauss-Legendre in ln lambda between
    the template nodes in ``node_arrays``), never on the caller's wavelength array,
    so the normalization does not depend on how the caller samples wavelength.

    Parameters
    ----------
    unit_fn : callable
        ``unit_fn(wave_nm) -> L_lambda [erg/s/nm]`` of the disc at l5100 = 1.
    node_arrays : sequence of array_like, optional
        Template wavelength nodes [nm] that the spectrum is interpolated on.

    Returns
    -------
    ndarray, shape ()
        :math:`\int_{91.2\,\mathrm{nm}}^{\infty} L_\lambda\,\mathrm{d}\lambda` [erg/s].
    """
    breaks = publication_breakpoints(LYMAN_LIMIT_NM, PUBLICATION_HI_NM, (), node_arrays)
    lam_nm, weight_nm = gauss_legendre_segments(breaks)
    return jnp.sum(weight_nm * unit_fn(lam_nm))
