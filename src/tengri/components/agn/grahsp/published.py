# SPDX-License-Identifier: BSD-3-Clause
r"""Exact and fixed-node integrals of GRAHSP's default AGN pieces, for the published ``L_*``.

Every published luminosity is :math:`\int L_\lambda\,\mathrm{d}\lambda` [erg/s] of the emitted
AGN-side spectrum, with :math:`\lambda` in nm. Three identities cover the default pieces:

1. **Unit-area Gaussian** (lines, Si feature). For :math:`G(\lambda;c,s)=
   (\sqrt{2\pi}\,s)^{-1}\exp[-(\lambda-c)^2/(2s^2)]`, :math:`\int G\,\mathrm{d}\lambda = 1` and
   :math:`\int_a^\infty G\,\mathrm{d}\lambda = \tfrac12\,\mathrm{erfc}[(a-c)/(\sqrt2\,s)]`. With
   a weight :math:`h`, :math:`\int G h\,\mathrm{d}\lambda = \mathbb{E}_{\lambda\sim N(c,s^2)}[h]`,
   evaluated by Gauss-Hermite quadrature with nodes :math:`c+\sqrt2\,s\,x_i`.
2. **Log-normal weight** (the torus pair). Write :math:`u=\log_{10}(\lambda/\mu\mathrm{m})`
   and :math:`a=\ln 10`. Completing the square,
   :math:`\exp[-(u-\mu)^2/(2W^2)+au]=\exp(a\mu+a^2W^2/2)\,\exp[-(u-\mu')^2/(2W^2)]` with
   :math:`\mu'=\mu+aW^2`. Since :math:`\mathrm{d}\lambda=1000\,a\,10^u\,\mathrm{d}u` [nm],
   :math:`\int \exp[-(u-\mu)^2/(2W^2)]\,\mathrm{d}\lambda = C_0(W)\,10^{\mu}` with
   :math:`C_0(W)=1000\,a\sqrt{2\pi}\,W\,e^{a^2W^2/2}` [nm], the closed form over
   :math:`(0,\infty)`. With a weight :math:`h`, the same expectation is taken over
   :math:`u\sim N(\mu',W^2)`.
3. **Trapezoid on a template** (FeII). The FeII forest is linear in :math:`\lambda` between its
   nodes and zero outside, so :math:`\int L\,\mathrm{d}\lambda` is the trapezoid on its nodes.

The Gauss-Hermite rules are exact for polynomials of degree below :math:`2n` in the Gaussian
variable; their node count is chosen from a convergence measurement against the dense
reference (``tests/physics/test_agn_template_grid_independence.py``), not assumed.

Notes
-----
**JIT-compatible**: yes; shapes are fixed by the configuration. **Gradient-safe**: yes.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
from jax import Array

__all__ = [
    "line_expectation",
    "log_gaussian_expectation",
    "lognormal_c0",
]

_LN10: float = math.log(10.0)
_NM_PER_UM: float = 1000.0


def lognormal_c0(width_dex: Array) -> Array:
    r"""Closed-form weight :math:`C_0(W)` of a log-normal torus component [nm].

    .. math::

       C_0(W) = 1000\,\ln 10\,\sqrt{2\pi}\,W\,e^{(\ln 10)^2 W^2/2},

    so that :math:`\int_0^\infty \exp[-(\log_{10}(\lambda/\mu\mathrm{m})-\mu)^2/(2W^2)]\,
    \mathrm{d}\lambda = C_0(W)\,10^{\mu}` [nm].

    Parameters
    ----------
    width_dex : array_like
        Log-normal standard deviation :math:`W` [dex].

    Returns
    -------
    ndarray
        :math:`C_0(W)` [nm].
    """
    w = jnp.asarray(width_dex)
    return _NM_PER_UM * _LN10 * math.sqrt(2.0 * math.pi) * w * jnp.exp(0.5 * (_LN10 * w) ** 2)


def log_gaussian_expectation(
    h_fn,
    log_center_um: Array,
    width_dex: Array,
    split_nm: float,
    order: int,
    half_width: float = 6.0,
) -> Array:
    r"""Expectation of :math:`h(\lambda)` under the log-normal weight, in nm.

    Returns :math:`\mathbb{E}_{u\sim N(\mu',W^2)}[h(1000\cdot10^{u})]` with
    :math:`\mu'=\mu+(\ln 10)W^2`, the factor that multiplies :math:`C_0(W)\,10^{\mu}`.
    The expectation is taken by Gauss-Legendre over :math:`\mu'\pm 6W`, split at the
    attenuation break ``split_nm`` (where :math:`h` has a kink, so each side is smooth);
    the tail beyond :math:`6W` is :math:`\mathrm{erfc}(6/\sqrt2)\approx 2\times10^{-9}`.

    Parameters
    ----------
    h_fn : callable
        ``h_fn(wave_nm) -> ndarray``, the weight, applied to the nodes.
    log_center_um : array_like
        :math:`\mu=\log_{10}(\lambda_{\rm peak}/\mu\mathrm{m})`.
    width_dex : array_like
        :math:`W` [dex].
    split_nm : float
        Non-smooth point of ``h_fn`` [nm].
    order : int
        Gauss-Legendre points per side.
    half_width : float, optional
        Window half-width in units of :math:`W`.

    Returns
    -------
    ndarray, shape ()
        The expectation (dimensionless).
    """
    w_dex = jnp.asarray(width_dex)
    mu_prime = jnp.asarray(log_center_um) + _LN10 * w_dex**2
    lo = mu_prime - half_width * w_dex
    hi = mu_prime + half_width * w_dex
    split = jnp.clip(math.log10(split_nm / _NM_PER_UM), lo, hi)
    x, gl_w = np.polynomial.legendre.leggauss(order)
    total = jnp.zeros(())
    for p, q in ((lo, split), (split, hi)):
        mid = 0.5 * (p + q)
        half = 0.5 * (q - p)
        u = mid + half * jnp.asarray(x)
        density = jnp.exp(-0.5 * ((u - mu_prime) / w_dex) ** 2) / (
            math.sqrt(2.0 * math.pi) * w_dex
        )
        wave_nm = _NM_PER_UM * 10.0**u
        total = total + jnp.sum(half * jnp.asarray(gl_w) * density * h_fn(wave_nm))
    return total


def feii_segment_product(h_fn, template_wave_nm: Array, template_lumin: Array) -> Array:
    r"""Two-point Gauss-Legendre per FeII segment of :math:`L_{\rm FeII}(\lambda)\,h(\lambda)`.

    The template is linear between its nodes, so on each segment :math:`L` is linear and
    :math:`h` smooth; the two-point rule is exact for cubics, so the error falls as the
    fourth power of the node spacing.

    Parameters
    ----------
    h_fn : callable
        ``h_fn(wave_nm) -> ndarray``.
    template_wave_nm, template_lumin : array_like, shape (n_template,)
        FeII template nodes [nm] and luminosity density [erg/s/nm] per unit amplitude.

    Returns
    -------
    ndarray, shape ()
        :math:`\int L_{\rm FeII}\,h\,\mathrm{d}\lambda` per unit amplitude [erg/s].
    """
    wave = jnp.asarray(template_wave_nm)
    lumin = jnp.asarray(template_lumin)
    w0, w1 = wave[:-1], wave[1:]
    l0, l1 = lumin[:-1], lumin[1:]
    half = 0.5 * (w1 - w0)
    node = 0.5 / math.sqrt(3.0)
    total = jnp.zeros(())
    for sign in (-1.0, 1.0):
        lam = 0.5 * (w0 + w1) + sign * node * (w1 - w0)
        lum = l0 + (l1 - l0) * (lam - w0) / (w1 - w0)
        total = total + jnp.sum(half * lum * h_fn(lam))
    return total


def gauss_hermite(n: int) -> tuple[np.ndarray, np.ndarray]:
    r"""Gauss-Hermite nodes and weights for a standard normal.

    Parameters
    ----------
    n : int
        Number of points.

    Returns
    -------
    nodes : ndarray, shape (n,)
        Standard-normal nodes :math:`\sqrt2\,x_i`, where :math:`x_i` are the Hermite roots.
    weights : ndarray, shape (n,)
        Weights :math:`w_i/\sqrt\pi`, summing to 1, so
        :math:`\mathbb{E}[f]\approx\sum_i w_i f(\sqrt2 x_i)`.
    """
    x, w = np.polynomial.hermite.hermgauss(n)
    return math.sqrt(2.0) * x, w / math.sqrt(math.pi)


def line_expectation(h_fn, center_nm: Array, sigma_nm: Array, order: int) -> Array:
    r"""Expectation of :math:`h(\lambda)` under unit-area Gaussians, one per feature.

    Gauss-Hermite quadrature with nodes :math:`c_i+\sqrt2\,s_i\,x_k`. A kink of ``h`` inside
    a feature's window degrades the rule; the measured error is recorded with the component.

    Parameters
    ----------
    h_fn : callable
        ``h_fn(wave_nm) -> ndarray``; applied to the nodes, shape ``(n_feature, order)``.
    center_nm : array_like, shape (n_feature,)
        Feature centers [nm].
    sigma_nm : array_like, shape (n_feature,)
        Gaussian standard deviations [nm].
    order : int
        Gauss-Hermite points per feature.

    Returns
    -------
    ndarray, shape (n_feature,)
        :math:`\mathbb{E}_{\lambda\sim N(c_i,s_i^2)}[h]` for each feature.
    """
    nodes, weights = gauss_hermite(order)
    c = jnp.asarray(center_nm)
    s = jnp.asarray(sigma_nm)
    wave_nm = c[:, None] + s[:, None] * jnp.asarray(nodes)[None, :]
    return jnp.sum(jnp.asarray(weights)[None, :] * h_fn(wave_nm), axis=1)


def log_breaks(lo_nm: float, hi_nm: float, per_decade: int, extra=()) -> np.ndarray:
    """Log-uniform breakpoints [nm] from ``lo_nm`` to ``hi_nm``, plus fixed kinks ``extra``.

    Parameters
    ----------
    lo_nm, hi_nm : float
        Support [nm].
    per_decade : int
        Breakpoints per decade of wavelength.
    extra : sequence of float, optional
        Non-smooth points of the integrand [nm] inside the support (e.g. the SMC break).

    Returns
    -------
    ndarray, shape (n_breaks,)
        Ascending breakpoints, both ends included.
    """
    n = round(per_decade * math.log10(hi_nm / lo_nm)) + 1
    grid = np.logspace(math.log10(lo_nm), math.log10(hi_nm), n)
    inside = [e for e in extra if lo_nm < e < hi_nm]
    return np.unique(np.concatenate([grid, np.asarray(inside, dtype=float)]))
