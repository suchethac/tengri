# SPDX-License-Identifier: BSD-3-Clause
r"""Gauss-Legendre band quadrature for the analytic dust precompute.

The band flux of an analytic spectrum through a filter is

.. math::

    \Phi = \frac{\int L_\nu(\lambda_{\rm obs}/(1+z))\, T(\lambda_{\rm obs})\,
           \lambda_{\rm obs}^{-1}\, d\lambda_{\rm obs}}
          {\int T(\lambda_{\rm obs})\, \lambda_{\rm obs}^{-1}\, d\lambda_{\rm obs}},

with the transmission ``T`` linear between the filter's nodes (the convention of
:func:`tengri.utils.grid_interp.preintegrate_grid`). Because the spectrum is a closed
form, the integral is evaluated by Gauss-Legendre rules on sub-segments of each
filter segment, sub-divided until the spectrum's logarithmic slope times the
sub-segment width is below a fixed bound. This resolves the Wien side of a cold
spectrum, where ``L_nu`` falls by ``h c / (lambda k T)`` e-folds per e-fold in
wavelength, and reads the spectrum at the filter's own wavelengths wherever they
fall: a filter outside any pre-set rest grid costs nothing.

Notes
-----
**JIT-compatible**: no, build-time NumPy.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import numpy as np

__all__ = ["BandQuadrature", "band_quadrature"]

#: Gauss-Legendre points per sub-segment.
GL_ORDER: int = 4

#: Bound on (log-slope of the spectrum) x (sub-segment width in ln lambda). A
#: four-point rule integrates exp(a x) over a sub-segment with a * width = 1.5 to
#: ~1e-7 relative, below the 1e-3 target by four orders.
SLOPE_WIDTH_BOUND: float = 1.5

#: Rest-frame wavelength [Angstrom] at which a filter segment that starts at a
#: non-positive wavelength is cut: the ``T / lambda`` measure diverges at 0, and the
#: dust spectra are zero (their Wien exponent is clipped) far below it.
WAVE_FLOOR_AA: float = 1.0e2


class BandQuadrature(NamedTuple):
    """Rest-frame quadrature nodes and weights of a filter set.

    Attributes
    ----------
    wave_rest : ndarray, shape (n_nodes,)
        Rest-frame wavelengths of every filter's nodes, concatenated [Angstrom].
    weights : ndarray, shape (n_nodes,)
        Weight of each node within its own filter; the weights of one filter sum to 1.
    owner : ndarray of int, shape (n_nodes,)
        Index of the filter each node belongs to.
    n_filters : int
        Number of filters.
    """

    wave_rest: np.ndarray
    weights: np.ndarray
    owner: np.ndarray
    n_filters: int


def _filter_nodes(
    fw: np.ndarray,
    ft: np.ndarray,
    redshift: float,
    log_slope: Callable[[np.ndarray], np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Observed-frame Gauss-Legendre nodes and ``T / lambda`` weights of one filter."""
    gx, gw = np.polynomial.legendre.leggauss(GL_ORDER)
    live = (fw[1:] > fw[:-1]) & ((ft[:-1] > 0.0) | (ft[1:] > 0.0))
    lo, hi = fw[:-1][live], fw[1:][live]
    lo = np.where(lo > 0.0, lo, WAVE_FLOOR_AA * (1.0 + redshift))
    keep = hi > lo
    lo, hi = lo[keep], hi[keep]
    if lo.size == 0:
        raise ValueError("filter transmission is zero everywhere: no band to integrate")
    slope = log_slope(0.5 * (lo + hi) / (1.0 + redshift))
    n_sub = np.maximum(1, np.ceil(np.log(hi / lo) * slope / SLOPE_WIDTH_BOUND)).astype(int)
    seg = np.repeat(np.arange(lo.size), n_sub)
    k = np.concatenate([np.arange(n) for n in n_sub])
    edge_lo = lo[seg] * (hi[seg] / lo[seg]) ** (k / n_sub[seg])
    edge_hi = lo[seg] * (hi[seg] / lo[seg]) ** ((k + 1) / n_sub[seg])
    half = 0.5 * (edge_hi - edge_lo)
    x = half[:, None] * gx[None, :] + 0.5 * (edge_hi + edge_lo)[:, None]
    w = half[:, None] * gw[None, :] * np.interp(x, fw, ft) / x
    return x.ravel(), w.ravel()


def band_quadrature(
    filter_waves: list,
    filter_trans: list,
    redshift: float,
    log_slope: Callable[[np.ndarray], np.ndarray],
) -> BandQuadrature:
    """Quadrature nodes of a filter set at redshift ``z``.

    Parameters
    ----------
    filter_waves : list of ndarray
        Observed-frame wavelength grid per filter [Angstrom].
    filter_trans : list of ndarray
        Transmission per filter.
    redshift : float
        Source redshift; the nodes are returned in the rest frame (``/ (1 + z)``).
    log_slope : callable
        Upper bound of ``|d ln L_nu / d ln lambda|`` at rest wavelengths [Angstrom],
        vectorized. Sets the sub-division of each filter segment.

    Returns
    -------
    BandQuadrature

    Raises
    ------
    ValueError
        If a filter has no positive transmission, or fewer than two nodes.
    """
    waves, owners, weights = [], [], []
    for i, (fw, ft) in enumerate(zip(filter_waves, filter_trans, strict=True)):
        fw_np = np.asarray(fw, dtype=np.float64)
        ft_np = np.asarray(ft, dtype=np.float64)
        if fw_np.size < 2 or fw_np.shape != ft_np.shape:
            raise ValueError(f"filter {i}: need matching wavelength and transmission, >= 2 nodes")
        x, w = _filter_nodes(fw_np, ft_np, float(redshift), log_slope)
        total = w.sum()
        if not total > 0.0:
            raise ValueError(f"filter {i}: transmission integrates to zero")
        waves.append(x / (1.0 + redshift))
        weights.append(w / total)
        owners.append(np.full(x.size, i, dtype=np.int64))
    return BandQuadrature(
        wave_rest=np.concatenate(waves),
        weights=np.concatenate(weights),
        owner=np.concatenate(owners),
        n_filters=len(filter_waves),
    )
