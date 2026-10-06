# SPDX-License-Identifier: BSD-3-Clause
"""Redshift grid of the photometric z-table: uniform nodes plus nodes on spectral-edge sweeps.

A band's flux :math:`F_b(z) = \\int S(\\lambda/(1+z))\\,T_b(\\lambda)\\,d\\lambda` is not
smooth where a sharp rest-frame edge of the SED (the Lyman limit, 911.76 A) sweeps
across the band's transmission curve. The observed edge sits at
:math:`\\lambda_e(1+z)`; as :math:`z` rises it enters the band at the blue cut-on,
runs over the flank where :math:`T_b` falls to zero and leaves at the red cutoff.
The flux below the edge is a small fraction of the flux above it, so :math:`F_b`
collapses by a large factor over the flank and bends sharply where the edge
leaves the support of :math:`T_b`. A uniform grid resolves that only by chance of
where its nodes land, so the interpolation error is *not monotone in the node
count* (11.6 / 1.5 / 0.7 / 1.2 % at 100 / 200 / 400 / 800 nodes for GALEX FUV,
#2749) and the grid that passed before one change fails after it.

The grid built here keeps the ``n_z`` uniform nodes the caller asked for exactly
and adds a graded cluster of nodes around every redshift at which a registered
edge crosses a band's support limit or one of its flanks (the redshifts at which
the transmission reaches given fractions of its peak). The table is read by a
local monotone cubic Hermite interpolant (:func:`tengri.utils.grid_interp.pchip_interp_local`,
Fritsch & Carlson 1980 [1]_), which is exact at every node and resolves a bend
that a node sits on; the cluster's local spacing ``h/R`` (``R`` =
:data:`REFINE_FACTOR`) shrinks the cubic's error across the bend by roughly
``R**2``. A band set and redshift range with no crossing gets the uniform grid
unchanged.

Only the Lyman limit is registered. Measured on the GALEX/SDSS/DES set and on
five NIRCam wide bands over z = 4-12 with the IGM on, Lyman-alpha nodes (1215.67 A)
leave the error at the default ``n_z`` unchanged and only matter at ``n_z`` = 100,
where they cut the F090W error in the IGM-extinguished tail from 10.8 % to 1.1 %
for 60-70 added nodes; extend :data:`EDGE_WAVELENGTHS_AA` to register another edge.

References
----------
.. [1] F. N. Fritsch and R. E. Carlson, "Monotone Piecewise Cubic
   Interpolation," SIAM J. Numer. Anal., 17(2), 238-246 (1980).
   https://doi.org/10.1137/0717021
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from tengri.utils.physics_constants import LYMAN_LIMIT_AA

__all__ = [
    "EDGE_WAVELENGTHS_AA",
    "band_edge_crossings",
    "build_edge_aware_z_grid",
    "edge_node_redshifts",
]

#: Rest wavelengths [Angstrom] of the sharp SED edges whose sweep across a band
#: cutoff the z-table must resolve.
EDGE_WAVELENGTHS_AA: tuple[float, ...] = (LYMAN_LIMIT_AA,)

#: Fractions of a band's peak transmission whose crossing (on the blue and the red
#: flank) is a kink-bearing redshift. 0.0 is the support limit itself.
FLANK_FRACTIONS: tuple[float, ...] = (0.0, 0.01, 0.1, 0.5)

#: Refinement factor: local node spacing at a crossing is ``h_uniform / R``.
REFINE_FACTOR: float = 8.0

#: Geometric growth of the spacing away from a crossing (back to ``h_uniform``).
GRADING_RATIO: float = 1.4

#: A node closer than this fraction of its own target spacing to a node already
#: kept is dropped (uniform nodes are kept first, so a uniform node sitting
#: within that distance of a crossing serves as its node).
MIN_GAP_FRACTION: float = 0.5


def band_edge_crossings(
    filter_waves,
    filter_trans,
    edges_aa=None,
    fractions=None,
) -> np.ndarray:
    """Redshifts at which a rest-frame edge crosses a band's support or flanks.

    For each band, the observed wavelengths at which the transmission first
    reaches (blue flank) and last stays above (red flank) each fraction of its
    peak, with the support limits for fraction 0, are mapped to the redshift
    :math:`z = \\lambda_{\\rm obs}/\\lambda_{\\rm edge} - 1` for each edge.

    Parameters
    ----------
    filter_waves, filter_trans : sequence of array_like
        One wavelength [Angstrom] and one transmission array per band.
    edges_aa : sequence of float, optional
        Rest wavelengths of the edges [Angstrom]. Default
        :data:`EDGE_WAVELENGTHS_AA`.
    fractions : sequence of float, optional
        Fractions of the peak transmission. Default :data:`FLANK_FRACTIONS`.

    Returns
    -------
    ndarray, shape (m,)
        Sorted unique crossing redshifts, all above ``-1``. [dimensionless]
    """
    edges_aa = EDGE_WAVELENGTHS_AA if edges_aa is None else tuple(edges_aa)
    fractions = FLANK_FRACTIONS if fractions is None else tuple(fractions)
    waves_obs = []
    for fw, ft in zip(filter_waves, filter_trans, strict=True):
        w = np.asarray(fw, dtype=float)
        t = np.asarray(ft, dtype=float)
        peak = float(np.max(t)) if t.size else 0.0
        if not peak > 0.0:
            continue
        for frac in fractions:
            blue, red = _flank_wavelengths(w, t, frac * peak)
            waves_obs.extend((blue, red))
    if not waves_obs or not edges_aa:
        return np.empty(0)
    z = np.concatenate([np.asarray(waves_obs) / lam - 1.0 for lam in edges_aa])
    return np.unique(z[z > -1.0])


def _flank_wavelengths(w: np.ndarray, t: np.ndarray, thr: float) -> tuple[float, float]:
    """Blue and red wavelengths at which ``t`` crosses ``thr`` (``t > 0`` if ``thr`` is 0)."""
    above = np.flatnonzero(t > thr) if thr <= 0.0 else np.flatnonzero(t >= thr)
    lo, hi = int(above[0]), int(above[-1])
    blue = _interp_crossing(w, t, lo, lo - 1, thr)
    red = _interp_crossing(w, t, hi, hi + 1, thr)
    return blue, red


def _interp_crossing(w: np.ndarray, t: np.ndarray, inside: int, outside: int, thr: float) -> float:
    """Wavelength at which ``t`` equals ``thr`` between samples ``inside`` and ``outside``."""
    if thr <= 0.0 or outside < 0 or outside >= w.size:
        return float(w[inside])
    t_in, t_out = float(t[inside]), float(t[outside])
    if t_in == t_out:
        return float(w[inside])
    frac = (t_in - thr) / (t_in - t_out)
    return float(w[inside] + frac * (w[outside] - w[inside]))


def edge_node_redshifts(
    crossings: np.ndarray, z_min: float, z_max: float, h_uniform: float
) -> list[tuple[float, float]]:
    """Graded cluster of ``(z, target spacing)`` around every crossing in range.

    Parameters
    ----------
    crossings : ndarray
        Crossing redshifts from :func:`band_edge_crossings`.
    z_min, z_max : float
        Table range; only nodes strictly inside are returned.
    h_uniform : float
        Spacing of the uniform grid.

    Returns
    -------
    list of (float, float)
        Candidate node and the spacing the grading intends there, crossing
        nodes first within each cluster.
    """
    out: list[tuple[float, float]] = []
    h0 = h_uniform / REFINE_FACTOR
    for zc in np.asarray(crossings, dtype=float):
        if not (z_min < zc < z_max):
            continue
        out.append((float(zc), h0))
        step, offset = h0, 0.0
        while step < h_uniform:
            offset += step
            for z in (zc - offset, zc + offset):
                if z_min < z < z_max:
                    out.append((float(z), step))
            step *= GRADING_RATIO
    return out


def build_edge_aware_z_grid(
    z_min: float,
    z_max: float,
    n_z: int,
    filter_waves=None,
    filter_trans=None,
    edges_aa=None,
) -> jnp.ndarray:
    """Uniform ``n_z`` nodes on ``[z_min, z_max]`` plus nodes on edge crossings.

    The ``n_z`` nodes of ``numpy.linspace(z_min, z_max, n_z)`` are always in the
    grid, unchanged. Nodes are added only where a registered edge crosses a
    band's support limit or flank inside the range (see the module docstring),
    so a band set without a crossing returns the uniform grid itself.

    Parameters
    ----------
    z_min, z_max : float
        Table range. [dimensionless]
    n_z : int
        Uniform node count, at least 2.
    filter_waves, filter_trans : sequence of array_like, optional
        Band curves; ``None`` returns the uniform grid.
    edges_aa : sequence of float, optional
        Rest wavelengths of the sharp edges [Angstrom]. Default
        :data:`EDGE_WAVELENGTHS_AA`.

    Returns
    -------
    ndarray, shape (n,)
        Strictly ascending nodes, ``n >= n_z``, in the JAX default float dtype.
    """
    if n_z < 2:
        raise ValueError(f"a redshift table needs at least 2 nodes, got n_z={n_z}")
    uniform = jnp.linspace(z_min, z_max, n_z)
    if filter_waves is None:
        return uniform
    crossings = band_edge_crossings(filter_waves, filter_trans, edges_aa)
    h_u = (float(z_max) - float(z_min)) / (int(n_z) - 1)
    cands = edge_node_redshifts(crossings, float(z_min), float(z_max), h_u)
    if not cands:
        return uniform
    kept = list(np.asarray(uniform, dtype=float))
    for z, spacing in sorted(cands, key=lambda c: c[1]):
        if np.min(np.abs(np.asarray(kept) - z)) >= MIN_GAP_FRACTION * spacing:
            kept.append(z)
    if len(kept) == int(n_z):
        return uniform
    return jnp.asarray(np.sort(np.asarray(kept)), dtype=uniform.dtype)
