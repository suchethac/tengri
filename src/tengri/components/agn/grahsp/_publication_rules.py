# SPDX-License-Identifier: BSD-3-Clause
r"""Quadrature rules and their measured orders for the published GRAHSP luminosities.

The rules here are the building blocks of the bolometric, torus and absorbed power integrals
(published as ``log_L_agn_bol``, ``log_L_agn_torus`` and ``log_L_agn_absorbed``) in
:mod:`tengri.components.agn.grahsp.published`: the log-normal expectation, the Gauss-Hermite
and Gauss-Legendre line expectations, the FeII segment product and the log-spaced breakpoints.
Each order and window below carries the error it was measured
at. None of this enters the emitted SED, which is integrated on the component's own grids.

Notes
-----
**JIT-compatible**: yes; node counts are fixed Python integers. **Gradient-safe**: yes.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
from jax import Array

from tengri.utils.host_array import device_table

_LN10: float = math.log(10.0)
_NM_PER_UM: float = 1000.0

#: Gauss-Legendre segments per decade for the smooth BBB quadrature (published L_* only).
#: Approximation error against the independent dense integral, BBB alone, 200 draws of the
#: declared box with the Balmer strength, line width and cutoff also drawn: L_agn_bol 1.3e-6 and
#: L_agn_absorbed 6.3e-7 at 10 per decade, 3.9e-9 and 2.8e-9 at 20 (this value), 2.8e-11 and
#: 2.2e-13 at 40.
SBPL_PER_DECADE: int = 20
#: Breakpoints per decade for the Balmer continuum quadrature (published L_* only).
#: Measured, Balmer alone over the same draws: worst error 1.3e-10 (L_agn_bol) and 2.3e-10
#: (L_agn_absorbed) at 20 per decade; 10 and 40 per decade give the same to 1e-9.
BALMER_PER_DECADE: int = 20
#: Gauss-Legendre points per side of the log-normal torus expectation (published L_* only).
#: Approximation of the weighted integral by a fixed-node rule over the window below. Measured
#: on the torus alone against the independent dense integral, 200 draws of the declared box
#: (E(B-V) in [3, 10]): worst L_agn_absorbed error 9.7e-5 at 20 points, 1.3e-5 at 24, 1.7e-7
#: at 32, 9.7e-10 at 40 (this value) and 4.5e-10 at 48; over the whole E(B-V) box the worst is
#: 3.0e-8 at 40 and above, a floor set by the 1100 nm kink, not by the order.
LOG_GL_ORDER: int = 40
#: Half-width of the log-normal expectation window in units of the torus width W (published L_*).
#: The absorbed power of a torus whose core sits where the attenuation is negligible comes from
#: its short-wavelength wing, so the window must reach far into it: measured at 40 points over
#: 200 draws of the box, a +/- 6 W window leaves 1.2e-5 in L_agn_absorbed of the torus alone and
#: +/- 8 W (this value) leaves 2.8e-9. The Gaussian mass beyond 8 W is 1.2e-15.
LOG_WINDOW_SIGMA: float = 8.0
#: Gauss-Hermite points per Si feature (published L_* only). The Si Gaussians sit more than 8
#: sigma from the 1100 nm attenuation kink, so the rule converges geometrically. Measured on the
#: torus plus Si, 200 draws of the box: worst L_agn_absorbed error 1.6e-5 at 3 points, 1.2e-7 at
#: 5 and 3.0e-8 (the torus floor above) from 7 points up; 13 is the value used.
SI_GH_ORDER: int = 13
#: Gauss-Legendre points per sub-segment of the line expectation (published L_* only).
#: The attenuation kinks at 1100 nm, inside the window of the 1083 nm line (1.1 sigma away at
#: the declared 10000 km/s) and of up to three lines at 20000 km/s, so a Gauss-Hermite rule
#: converges only algebraically (measured 5.7e-8 in the box and 1.2e-6 at wider lines at 13
#: points, 2.9e-7 at 61). The window is cut at the kink and each side in LINE_SUBSEGMENTS equal
#: sub-segments. Measured, lines alone, 200 draws with the line width drawn between 1e3 and
#: 2e4 km/s: worst L_agn_absorbed error 1.2e-4 at 6 points, 9.3e-7 at 8, 4.9e-9 at 10 (this
#: value; the same in the box).
LINE_GL_ORDER: int = 10
#: Sub-segments per side of the line expectation window (published L_* only).
LINE_SUBSEGMENTS: int = 3
#: Half-width of the line expectation window in units of the line sigma (published L_* only).
#: The Gaussian mass beyond 7 sigma is 2.6e-12.
LINE_WINDOW_SIGMA: float = 7.0
#: Segment width of the Si clip correction in units of the Si Gaussian sigma (published L_* only).
#: A dip of ``max(si, -torus)`` narrower than a segment, with positive ends, is invisible to the
#: sign test that places the kinks. Measured over 3000 random draws of the declared box
#: (about 700 binding): worst L_agn_torus error 4.6e-6 at 1.0 sigma, 9.5e-8 at 0.5, 1.6e-8 at
#: 0.25 and 3.7e-10 at 0.125.
CLIP_STEP_SIGMA: float = 0.125
#: Upper edge of the log grid laid between the Netzer-disc nodes [nm]; the template ends at
#: 1.24e6 nm and the integral is cut at its last node. The template has a 4.6 e-fold gap
#: between two nodes, across which a four-point rule on the node segment alone is not accurate.
_DISC_GRID_HI_NM: float = 2.0e6
#: Upper edge of the Balmer quadrature [nm]; its edge window ends below ~400 nm for any width.
_BALMER_HI_NM: float = 600.0


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
    half_width: float = LOG_WINDOW_SIGMA,
) -> Array:
    r"""Expectation of :math:`h(\lambda)` under the log-normal weight, in nm.

    Returns :math:`\mathbb{E}_{u\sim N(\mu',W^2)}[h(1000\cdot10^{u})]` with
    :math:`\mu'=\mu+(\ln 10)W^2`, the factor that multiplies :math:`C_0(W)\,10^{\mu}`.
    The expectation is taken by Gauss-Legendre over :math:`\mu'\pm wW`, split at the
    attenuation break ``split_nm`` (where :math:`h` has a kink, so each side is smooth);
    the Gaussian mass beyond the window is :math:`\mathrm{erfc}(w/\sqrt2)`.

    Parameters
    ----------
    h_fn : callable
        ``h_fn(wave_nm) -> ndarray``, the weight, applied to the nodes.
    log_center_um : array_like
        :math:`\mu=\log_{10}(\lambda_{\rm peak}/\mu\mathrm{m})` [dex re 1 um].
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
    wave = device_table(template_wave_nm)
    lumin = device_table(template_lumin)
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
    a feature's window degrades the rule to algebraic convergence; the error at
    :data:`SI_GH_ORDER` is recorded beside that constant.

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


def split_gaussian_expectation(
    h_fn,
    center_nm: Array,
    sigma_nm: Array,
    split_nm: float,
    order: int,
    n_sub: int,
    half_width: float,
) -> Array:
    r"""Expectation of :math:`h(\lambda)` under unit-area Gaussians, with ``h`` kinked at a point.

    Gauss-Legendre over :math:`c_i\pm w s_i`, cut at the kink of ``h`` clipped into the window
    (a zero-width side where the kink lies outside it) and each side cut again into ``n_sub``
    equal sub-segments, so every segment is smooth and at most a few sigma wide. The Gaussian
    mass beyond the window is :math:`\mathrm{erfc}(w/\sqrt2)`.

    Parameters
    ----------
    h_fn : callable
        ``h_fn(wave_nm) -> ndarray``, applied elementwise to the nodes.
    center_nm, sigma_nm : array_like, shape (n_feature,)
        Feature centers and standard deviations [nm].
    split_nm : float
        Non-smooth point of ``h_fn`` [nm].
    order : int
        Gauss-Legendre points per sub-segment.
    n_sub : int
        Sub-segments per side.
    half_width : float
        Window half-width :math:`w` in units of sigma.

    Returns
    -------
    ndarray, shape (n_feature,)
        :math:`\mathbb{E}_{\lambda\sim N(c_i,s_i^2)}[h]` for each feature.
    """
    c = jnp.asarray(center_nm)
    s = jnp.asarray(sigma_nm)
    lo = c - half_width * s
    hi = c + half_width * s
    split = jnp.clip(split_nm, lo, hi)
    x, gl_w = np.polynomial.legendre.leggauss(order)
    frac = jnp.arange(n_sub + 1) / n_sub
    total = jnp.zeros_like(c)
    for p, q in ((lo, split), (split, hi)):
        edges = p[:, None] + (q - p)[:, None] * frac[None, :]
        half = 0.5 * (edges[:, 1:] - edges[:, :-1])[..., None]
        lam = 0.5 * (edges[:, 1:] + edges[:, :-1])[..., None] + half * jnp.asarray(x)
        density = jnp.exp(-0.5 * ((lam - c[:, None, None]) / s[:, None, None]) ** 2) / (
            math.sqrt(2.0 * math.pi) * s[:, None, None]
        )
        total = total + jnp.sum(half * jnp.asarray(gl_w) * density * h_fn(lam), axis=(1, 2))
    return total


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
