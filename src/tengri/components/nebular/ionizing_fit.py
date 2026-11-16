# SPDX-License-Identifier: BSD-3-Clause
r"""Four-segment power-law fit of an ionizing spectrum, in JAX.

Cue (Li et al. 2025 [1]_) describes an ionizing spectrum by a power law
:math:`F_\nu = A_k \lambda^{\alpha_k}` on each of four segments bounded by the
HeII, OII, HeI and HI edges, and takes its seven shape parameters from a fit of
that form to the spectrum. This module performs that fit for a spectrum held as
log10 flux on a fixed wavelength layout, differentiably, so the shape follows
whatever the spectrum is built from (a stellar population, an AGN, or both).

Each segment minimizes the objective of the reference code,

.. math::

    L(\alpha, b) = \tfrac12 \sum_i \bigl(y_i - b - \alpha x_i\bigr)^2
        + \tfrac12 N \bigl(\log Q - \log Q_\mathrm{pred}(\alpha, b)\bigr)^2,

with :math:`y_i` the normalized log10 flux, :math:`x_i = \log_{10}\lambda_i`
[:math:`\lambda` in Angstrom], :math:`N` the pixel count, :math:`Q` the segment
photon rate of the normalized flux and
:math:`\log Q_\mathrm{pred} = b - \log h + \log_{10}|(\lambda_\mathrm{max}^\alpha -
\lambda_\mathrm{min}^\alpha)/\alpha|`.

With :math:`x` and :math:`y` centred on their segment means
(:math:`\tilde x_i = x_i - \bar x`, :math:`\tilde y_i = y_i - \bar y`) and the
intercept written :math:`b = \bar y + \delta - \alpha \bar x`, the optimum in
:math:`\delta` is closed form and the objective in the slope alone is

.. math::

    \phi(\alpha) = -\alpha \sum_i \tilde x_i \tilde y_i
        + \tfrac12 \alpha^2 \sum_i \tilde x_i^2 + N \delta(\alpha)^2,
    \qquad
    \delta(\alpha) = \tfrac12 \bigl(\log Q + \log h - \bar y - \tilde G(\alpha)\bigr),

up to a constant, with :math:`\tilde G(\alpha) = \log_{10}|(\lambda_\mathrm{max}^\alpha
- \lambda_\mathrm{min}^\alpha)/\alpha| - \alpha \bar x`. Every term is of order
unity, so the root is as accurate in float32 as the data allow. The data enter only
through :math:`\sum \tilde x_i \tilde y_i`, :math:`\bar y` and :math:`\log Q`; the
slope solves :math:`\phi'(\alpha) = 0` by Newton's method, the four segments
together, and the derivative of the root comes from that stationarity condition
(:func:`jax.lax.custom_root`), not from the iteration.

References
----------
.. [1] Y. Li et al., "Cue: A Fast and Flexible Photoionization Emulator for
   Modeling Nebular Emission Powered by Almost Any Ionizing Source," ApJ, 986, 9
   (2025). arXiv:2405.04598. https://doi.org/10.3847/1538-4357/adcab4
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

from tengri.components.nebular._constants import _C_AA, _H_PLANCK

#: Newton iterations of the scalar slope root. The start is the least-squares slope
#: of the segment, and the root is reached to round-off within this count for every
#: SSP row and population tested (``tests/components/nebular/test_ionizing_fit.py``).
NEWTON_ITERATIONS: int = 8

#: Below this ``|z|`` the series ``log1p(z/2 + z**2/6)`` replaces ``log(expm1(z)/z)``.
_EXPM1_SERIES_CUTOFF: float = 1e-4

#: log10 of the reference normalization target (``norm = 1e-18 / ref_flux``).
_LOG_NORM_TARGET: float = -18.0
#: log10 of the reference flux floor relative to the normalization (``1e-70 * norm``).
_LOG_CLIP_FLOOR: float = -70.0
#: log10 of the reference ``max(., 1e-99)`` floors on the log flux and on ``Q_seg``.
_LOG_TINY: float = -99.0

_LN10: float = float(np.log(10.0))
_LOG_H: float = float(np.log10(_H_PLANCK))
_LOG10_2: float = float(np.log10(2.0))


class IonizingLayout(NamedTuple):
    """Static pixel geometry of the four-segment fit (host NumPy, float64).

    Attributes
    ----------
    ref_pair : ndarray, shape (2,)
        Pixel indices whose mean flux sets the normalization (the two pixels at
        the HI edge, as in the reference fit).
    seg_index : ndarray, shape (4, n_max)
        Pixel index of each segment slot, padded with 0.
    seg_mask : ndarray, shape (4, n_max)
        True on real pixels of the segment.
    x_centred : ndarray, shape (4, n_max)
        ``log10(lambda) - mean`` per slot [dex], 0 on padding.
    log_qweight : ndarray, shape (4, n_max)
        ``log10`` of the trapezoid weight in frequency divided by ``h nu``, so the
        segment photon rate is ``sum 10**(y + log_qweight)`` [log10(Hz / (erg s))].
    n_pix, x_mean, sum_xx : ndarray, shape (4,)
        Pixel count, mean of ``log10(lambda)`` and ``sum x_centred**2`` per segment.
    log10_xmax, ell : ndarray, shape (4,)
        ``log10`` of the last pixel wavelength and ``ln(first/last)`` (negative),
        for the photon-rate term.
    lum_log10_hi, lum_ell : ndarray, shape (4,)
        The same two numbers for the segment edges, for the segment luminosities.
    """

    ref_pair: np.ndarray
    seg_index: np.ndarray
    seg_mask: np.ndarray
    x_centred: np.ndarray
    log_qweight: np.ndarray
    n_pix: np.ndarray
    x_mean: np.ndarray
    sum_xx: np.ndarray
    log10_xmax: np.ndarray
    ell: np.ndarray
    lum_log10_hi: np.ndarray
    lum_ell: np.ndarray


def ionizing_layout(wave: np.ndarray, edges: np.ndarray) -> IonizingLayout:
    """Static geometry of the four-segment fit on a wavelength grid.

    Parameters
    ----------
    wave : array_like, shape (n_wave,)
        Ascending wavelength grid [Angstrom], covering the four segments.
    edges : array_like, shape (5,)
        Segment edges [Angstrom]: 1, HeII, OII, HeI, HI.

    Returns
    -------
    IonizingLayout
        Pixel bounds and per-segment constants, as the reference fit takes them
        (a segment holds the pixels from one edge up to the next).
    """
    wave = np.asarray(wave, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)
    ind_bin = np.array([np.max(np.where(wave <= e)[0]) for e in edges[1:]]) + 1
    ind_bin = np.insert(ind_bin, 0, 0)
    counts = np.diff(ind_bin)
    if np.any(counts < 2):
        raise ValueError(
            f"every ionizing segment needs at least two pixels; got {counts.tolist()}"
        )
    n_max = int(counts.max())
    seg_index = np.zeros((4, n_max), dtype=np.int32)
    seg_mask = np.zeros((4, n_max), dtype=bool)
    x_centred = np.zeros((4, n_max))
    log_qweight = np.full((4, n_max), -np.inf)
    x_mean, sum_xx, log10_xmax, ell = (np.zeros(4) for _ in range(4))
    for k in range(4):
        idx = np.arange(ind_bin[k], ind_bin[k + 1])
        seg = wave[idx]
        x = np.log10(seg)
        nu = _C_AA / seg
        dnu = np.abs(np.diff(nu))
        trap = np.zeros(seg.size)
        trap[:-1] += 0.5 * dnu
        trap[1:] += 0.5 * dnu
        n = idx.size
        seg_index[k, :n] = idx
        seg_mask[k, :n] = True
        x_mean[k] = x.mean()
        x_centred[k, :n] = x - x_mean[k]
        log_qweight[k, :n] = np.log10(trap / (_H_PLANCK * nu))
        sum_xx[k] = np.sum((x - x_mean[k]) ** 2)
        log10_xmax[k] = np.log10(seg[-1])
        ell[k] = np.log(seg[0] / seg[-1])
    i_hi = int(ind_bin[-1])
    return IonizingLayout(
        ref_pair=np.array([i_hi - 1, i_hi], dtype=np.int32),
        seg_index=seg_index,
        seg_mask=seg_mask,
        x_centred=x_centred,
        log_qweight=log_qweight,
        n_pix=counts.astype(np.float64),
        x_mean=x_mean,
        sum_xx=sum_xx,
        log10_xmax=log10_xmax,
        ell=ell,
        lum_log10_hi=np.log10(edges[1:]),
        lum_ell=np.log(edges[:-1] / edges[1:]),
    )


_FIT_CACHE: dict[tuple[bytes, bytes], tuple[IonizingLayout, object]] = {}


def compiled_fit(wave: np.ndarray, edges: np.ndarray):
    """Layout and jitted :func:`fit_ionizing_shape` for a wavelength grid, memoized.

    Parameters
    ----------
    wave : array_like, shape (n_wave,)
        Ascending wavelength grid [Angstrom].
    edges : array_like, shape (5,)
        Segment edges [Angstrom].

    Returns
    -------
    tuple
        ``(layout, fit)`` where ``fit(log_flux)`` returns an :class:`IonizingShape`.
    """
    wave = np.ascontiguousarray(wave, dtype=np.float64)
    edges = np.ascontiguousarray(edges, dtype=np.float64)
    key = (wave.tobytes(), edges.tobytes())
    if key not in _FIT_CACHE:
        layout = ionizing_layout(wave, edges)
        _FIT_CACHE[key] = (layout, jax.jit(lambda lf: fit_ionizing_shape(lf, layout)))
    return _FIT_CACHE[key]


def _log10_expm1_ratio(z):
    """``log10(expm1(z) / z)``, finite and smooth for every real ``z``."""
    small = jnp.abs(z) < _EXPM1_SERIES_CUTOFF
    zs = jnp.where(small, 1.0, z)
    zp = jnp.where(zs > 0, zs, 1.0)
    zn = jnp.where(zs < 0, zs, -1.0)
    large = jnp.where(
        zs > 0,
        zp + jnp.log(-jnp.expm1(-zp)) - jnp.log(zp),
        jnp.log(-jnp.expm1(zn)) - jnp.log(-zn),
    )
    series = jnp.log1p(z / 2.0 + z * z / 6.0)
    return jnp.where(small, series, large) / _LN10


def _log10_power_integral(alpha, log10_hi, ell):
    """``log10 |(hi**alpha - lo**alpha) / alpha|`` with ``ell = ln(lo/hi) < 0``."""
    return alpha * log10_hi + jnp.log10(-ell) + _log10_expm1_ratio(alpha * ell)


def _delta(alpha, stats):
    """Closed-form optimum of the centred intercept offset at slope ``alpha``."""
    _, offset, _, x_mean, log10_xmax, ell, _ = stats
    g_centred = _log10_power_integral(alpha, log10_xmax, ell) - alpha * x_mean
    return 0.5 * (offset - g_centred)


def _profile(alpha, stats):
    """Segment objective in the slope alone (centred form, constant dropped)."""
    sum_xy, _, n, _, _, _, sum_xx = stats
    d = _delta(alpha, stats)
    return -alpha * sum_xy + 0.5 * alpha * alpha * sum_xx + n * d * d


def _solve_slope(stats, alpha0):
    """Stationary slope of one segment, differentiated by the implicit rule."""

    def slope_gradient(alpha):
        return jax.grad(_profile)(alpha, stats)

    def newton(f, x0):
        curvature = jax.grad(f)

        def body(_, a):
            d2 = curvature(a)
            # The profile is convex near its minimum; a non-positive curvature far
            # from it would step uphill, so take a unit-curvature step there instead.
            d2 = jnp.where(d2 > 0.0, d2, 1.0)
            return a - f(a) / d2

        return jax.lax.fori_loop(0, NEWTON_ITERATIONS, body, x0)

    def tangent_solve(g, y):
        # The implicit derivative divides by the curvature at the root. It is positive
        # at a minimum; a segment with no usable flux could leave it at zero, so the
        # same rule as the Newton step applies and the gradient stays finite.
        curvature = g(jnp.ones_like(y))
        return y / jnp.where(curvature > 0.0, curvature, 1.0)

    return jax.lax.custom_root(slope_gradient, alpha0, newton, tangent_solve)


class IonizingShape(NamedTuple):
    r"""Result of :func:`fit_ionizing_shape`.

    Attributes
    ----------
    i7 : jnp.ndarray, shape (7,)
        Cue's shape parameters, unclamped: the four slopes, then the three
        ``log10`` ratios of adjacent segment luminosities.
    slopes : jnp.ndarray, shape (4,)
        The four slopes ``alpha_k``.
    log_amp : jnp.ndarray, shape (4,)
        ``log10 A_k`` in the units of the input flux, so the fitted segment is
        ``10**log_amp * lambda**alpha`` [log10 of input L_nu units].
    log_seglum : jnp.ndarray, shape (4,)
        ``log10`` of :math:`\int A_k \lambda^{\alpha_k - 2}\,d\lambda` over the
        segment edges, in the input flux units times Angstrom^-1; multiply by
        :math:`c` [Angstrom/s] for a luminosity. Only differences enter ``i7``.
    """

    i7: jnp.ndarray
    slopes: jnp.ndarray
    log_amp: jnp.ndarray
    log_seglum: jnp.ndarray


def fit_ionizing_shape(log_flux, layout: IonizingLayout) -> IonizingShape:
    r"""Fit Cue's four-segment power law to an ionizing spectrum.

    Parameters
    ----------
    log_flux : array_like, shape (n_wave,)
        ``log10`` of the spectrum :math:`L_\nu` on the layout's grid, in any
        normalization (the shape does not depend on it) [log10(erg/s/Hz) + const].
        A pixel with no flux may be ``-inf``; it is floored as the reference fit
        floors it.
    layout : IonizingLayout
        From :func:`ionizing_layout` on the same grid.

    Returns
    -------
    IonizingShape
        The seven shape parameters (unclamped), the slopes, the amplitudes and
        the segment luminosities.

    Notes
    -----
    Normalization, floors and the objective follow the reference fit
    (``fit_4loglinear_ionparam`` in the Cue code; tengri's
    :func:`~tengri.components.nebular.ionizing_spectrum.fit_ionizing_spectrum`),
    and the minimizer is the exact stationary point of each segment's objective
    (see the module docstring for the centred form solved). The ratio parameters
    use the energy-integrated segment luminosity
    :math:`\int A_k \lambda^{\alpha_k - 2} c\,d\lambda` over the segment edges, as
    the reference code does.

    Implements the parameterization of Li et al. 2025 [1]_. JIT, grad and vmap
    safe; differentiable in ``log_flux``.

    References
    ----------
    .. [1] Y. Li et al., ApJ, 986, 9 (2025). arXiv:2405.04598.
       https://doi.org/10.3847/1538-4357/adcab4
    """
    log_flux = jnp.asarray(log_flux)
    if not jnp.issubdtype(log_flux.dtype, jnp.floating):
        log_flux = log_flux.astype(jnp.result_type(float))
    dtype = log_flux.dtype

    def const(a):
        # Layout constants follow the input precision, so a float32 spectrum is
        # fitted in float32 even in a float64 process.
        return jnp.asarray(a, dtype=dtype)

    pair = log_flux[layout.ref_pair]
    log_ref = jnp.maximum(logsumexp(_LN10 * pair) / _LN10 - _LOG10_2, _LOG_TINY)
    log_norm = _LOG_NORM_TARGET - log_ref
    lnorm = jnp.maximum(log_flux, _LOG_CLIP_FLOOR) + log_norm

    seg = lnorm[layout.seg_index]  # (4, n_max)
    mask = jnp.asarray(layout.seg_mask)
    n = const(layout.n_pix)
    y = jnp.where(mask, jnp.maximum(seg, _LOG_TINY), 0.0)
    y_mean = jnp.sum(y, axis=1) / n
    y_centred = jnp.where(mask, y - y_mean[:, None], 0.0)
    sum_xy = jnp.sum(y_centred * const(layout.x_centred), axis=1)
    log_qweight = jnp.where(mask, const(np.where(layout.seg_mask, layout.log_qweight, 0.0)), 0.0)
    log_q_terms = jnp.where(mask, _LN10 * (seg + log_qweight), -jnp.inf)
    log_q = jnp.maximum(logsumexp(log_q_terms, axis=1) / _LN10, _LOG_TINY)

    sum_xx = const(layout.sum_xx)
    x_mean = const(layout.x_mean)
    stats = (
        sum_xy,
        log_q + _LOG_H - y_mean,
        n,
        x_mean,
        const(layout.log10_xmax),
        const(layout.ell),
        sum_xx,
    )
    slopes = jax.vmap(_solve_slope)(stats, sum_xy / sum_xx)
    delta = jax.vmap(_delta)(slopes, stats)
    log_amp = y_mean + delta - slopes * x_mean - log_norm
    log_seglum = log_amp + _log10_power_integral(
        slopes - 1.0, const(layout.lum_log10_hi), const(layout.lum_ell)
    )
    i7 = jnp.concatenate([slopes, jnp.diff(log_seglum)])
    return IonizingShape(i7=i7, slopes=slopes, log_amp=log_amp, log_seglum=log_seglum)
