# SPDX-License-Identifier: BSD-3-Clause
"""Static node windows of the pixels next to a dust-law breakpoint (#3413).

A pixel-moment spectrum table is exact except in the pixels whose rest-frame source range
contains a dust-law breakpoint: there the first-order expansion of the dust factor fails. Those
pixels are flagged at build, and each one keeps a static window of model nodes, padded to a
common length, so the per-draw path can evaluate them exactly on that window.

Everything here is build-time index arithmetic on numpy arrays. The window of a pixel is its
rest-frame source range, widened by ``pad_aa`` on both sides (the lattice window the exact path
uses, at least five sigma of the total broadening).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from tengri.components.lyc import LYMAN_LIMIT_AA
from tengri.utils.physics_constants import C_KM_S


def flagged_pixel_mask(
    edges_rest: np.ndarray,
    breakpoints: np.ndarray,
    pad_aa: np.ndarray | float,
) -> np.ndarray:
    """Which pixels' padded source range contains a breakpoint.

    Parameters
    ----------
    edges_rest : ndarray, shape (n_pix + 1,)
        Rest-frame pixel edges ``wave_obs_edges / (1 + z)`` [Angstrom], increasing.
    breakpoints : ndarray, shape (n_bp,)
        Rest-frame dust-law breakpoints [Angstrom].
    pad_aa : float or ndarray, shape (n_pix,)
        Half-width of the padding on each side of a pixel [Angstrom].

    Returns
    -------
    ndarray of bool, shape (n_pix,)
    """
    edges = np.asarray(edges_rest, dtype=np.float64)
    bps = np.asarray(breakpoints, dtype=np.float64).reshape(-1)
    pad = np.broadcast_to(np.asarray(pad_aa, dtype=np.float64), (edges.size - 1,))
    lo = edges[:-1] - pad
    hi = edges[1:] + pad
    if bps.size == 0:
        return np.zeros(edges.size - 1, dtype=bool)
    # A breakpoint lies in [lo, hi] when the count of breakpoints <= hi exceeds the count < lo.
    below_hi = np.searchsorted(bps, hi, side="right")
    below_lo = np.searchsorted(bps, lo, side="left")
    return below_hi > below_lo


def node_window(wave_nodes: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Indices of the model nodes inside ``[lo, hi]`` [Angstrom], increasing.

    The window must cover the slice of the source that the pixel reads; the caller pads ``lo``
    and ``hi`` by the broadening's reach before calling this.
    """
    w = np.asarray(wave_nodes, dtype=np.float64)
    i0 = int(np.searchsorted(w, lo, side="left"))
    i1 = int(np.searchsorted(w, hi, side="right"))
    return np.arange(i0, i1, dtype=np.int64)


def padded_node_slices(windows: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Pad per-pixel node windows to a common length ``L``.

    Parameters
    ----------
    windows : list of ndarray
        One node-index array per flagged pixel (from :func:`node_window`).

    Returns
    -------
    idx : ndarray of int, shape (n_flag, L)
        Node indices; masked entries point at node 0 and carry zero weight.
    mask : ndarray of bool, shape (n_flag, L)
        True where the entry is a real node.
    """
    n_flag = len(windows)
    length = max((w.size for w in windows), default=0)
    idx = np.zeros((n_flag, length), dtype=np.int64)
    mask = np.zeros((n_flag, length), dtype=bool)
    for row, w in enumerate(windows):
        idx[row, : w.size] = w
        mask[row, : w.size] = True
    return idx, mask


def slice_attenuated_stellar(dust, params, wave_rest, lnu_age, younger, idx, mask):
    """Per-age attenuated stellar SED on one pixel's node window (#3413).

    The exact path's contraction, on the window's nodes only: the sum over ages of
    ``lnu_age * T(age, lambda)``, with the transmission the dust component's own two-component
    screen (:meth:`compute_transmission`) evaluates on those wavelengths. The screen is
    elementwise in wavelength, so the window values equal the full-grid values at the same
    nodes.

    Parameters
    ----------
    dust : object
        A dust component with ``compute_transmission(params, wavelength, younger_fraction)``.
    params : mapping
        The draw's dust parameters.
    wave_rest : array, shape (n_wave,)
        Rest-frame model grid [Angstrom].
    lnu_age : array, shape (n_age, n_wave)
        The stellar per-age SED on the model grid (the exact path's ``lnu_age``).
    younger : array, shape (1, n_age)
        The stellar component's ``age_boundary_younger_fraction``.
    idx, mask : ndarray
        One pixel's padded node window from :func:`padded_node_slices`.

    Returns
    -------
    sed_attenuated : array, shape (n_slice,)
        Attenuated stellar SED on the window nodes, in the order of the window.
    sed_intrinsic : array, shape (n_slice,)
        Unattenuated stellar SED on the window nodes.
    """
    nodes = np.asarray(idx)[np.asarray(mask, dtype=bool)]
    lnu_s = lnu_age[:, nodes]
    transmission = dust.compute_transmission(params, wave_rest[nodes], younger)
    return jnp.sum(lnu_s * transmission, axis=0), jnp.sum(lnu_s, axis=0)


#: The velocity-dispersion bound [km/s] the lattice margin is sized for, and the number of
#: sigmas of it that a flagged pixel's node window reaches beyond the pixel's source range.
_SIGMA_REACH = 5.0
_WINDOW_SIGMAS = 8.0

#: A pixel is flagged when the second-order remainder of a dust factor's first-order
#: expansion across it, relative to the factor, exceeds this. A kink (a slope jump between
#: sub-grid samples) can cost the table about twice this bound, so laws with known kinks
#: declare them in ``DUST_LAW_BREAKPOINTS_AA`` instead of relying on the bound.
CURVATURE_BOUND = 1e-4
_CURVATURE_SUBNODES = 9


def curvature_flag_mask(edges_rest, curves, bound=CURVATURE_BOUND, pad_aa=None):
    r"""Pixels where a dust curve's curvature defeats the first-order screen expansion.

    The table carries :math:`T(c) M_0 + T'(c) M_1`; the remainder is
    :math:`\tfrac12 T''\,\overline{S(\mu-c)^2}`. Its size relative to the factor is
    estimated from second differences of each curve on a uniform sub-grid,
    :math:`8\,\max|\Delta^2 k| / \overline{|k|}` (the :math:`w^2/8` bound on the remainder
    of a quadratic over a pixel of width :math:`w`). The maximum runs over the pixel's source
    range, its edges widened by ``pad_aa`` (the broadening's reach), because a kink outside
    the pixel still moves the pixel's source mean; the scale is the curve over the pixel
    itself. Each pixel's sub-grid spacing is one eighth of its own width.

    Parameters
    ----------
    edges_rest : ndarray, shape (n_pix + 1,)
        Rest-frame pixel edges [Angstrom].
    curves : sequence of callable
        Each maps rest wavelengths [Angstrom] (ndarray) to a curve value ``k``.
    bound : float
        Relative remainder above which a pixel is flagged.
    pad_aa : float or ndarray, shape (n_pix,), optional
        Half-width [Angstrom] of each pixel's source reach. Default 0 (the pixel alone).

    Returns
    -------
    ndarray of bool, shape (n_pix,)
    """
    edges = np.asarray(edges_rest, dtype=np.float64)
    n_pix = edges.size - 1
    pad = np.zeros(n_pix) if pad_aa is None else np.broadcast_to(
        np.asarray(pad_aa, dtype=np.float64), (n_pix,)
    )
    u = np.linspace(0.0, 1.0, _CURVATURE_SUBNODES)
    flags = np.zeros(n_pix, dtype=bool)
    for curve in curves:
        for p in range(n_pix):
            width = edges[p + 1] - edges[p]
            step = width / (_CURVATURE_SUBNODES - 1)
            reach_lo = edges[p] - pad[p]
            n_reach = int(np.ceil((width + 2.0 * pad[p]) / step)) + 1
            lam = reach_lo + step * np.arange(n_reach)
            g = np.asarray(curve(jnp.asarray(lam)), dtype=np.float64)
            d2 = np.abs(g[:-2] - 2.0 * g[1:-1] + g[2:]).max(initial=0.0)
            own = np.asarray(curve(jnp.asarray(edges[p] + width * u)), dtype=np.float64)
            scale = float(np.abs(own).mean())
            if scale > 0.0 and 8.0 * d2 / scale > bound:
                flags[p] = True
    return flags


def kernel_reach_pad_aa(centres_rest, sigma_kms, reach=_SIGMA_REACH):
    """Per-pixel half-width [Angstrom] of ``reach`` sigmas at the pixel's wavelength."""
    return np.asarray(centres_rest, dtype=np.float64) * np.expm1(
        reach * float(sigma_kms) / C_KM_S
    )


@dataclass(frozen=True)
class FlaggedPixels:
    """Static layout of the pixels the table computes exactly per draw (#3413).

    Attributes
    ----------
    pixels : ndarray of int, shape (n_flag,)
        Flagged pixel indices.
    nodes : ndarray of int, shape (n_nodes,)
        Model-grid node indices of every flagged pixel's window, concatenated.
    offsets : tuple of int
        ``offsets[k]:offsets[k + 1]`` addresses pixel ``k``'s window in ``nodes``.
    spans : tuple of (float, int)
        Each window's ln-lambda lattice (:func:`~tengri.observation.spectrum.lattice_window`).
    wave_nodes : ndarray, shape (n_nodes,)
        Rest wavelength of every window node [Angstrom].
    ssp_cube : ndarray, shape (n_met, n_age, n_nodes)
        The library SSP cube interpolated onto the window nodes [Lsun/Hz/Msun].
    centres_rest : ndarray, shape (n_pix,)
        Rest-frame pixel centres [Angstrom].
    sigma_v_kms : float or None
        Velocity dispersion of the lattice correction, ``None`` when the table has none.
    instruments : tuple of (dict or None)
        Per flagged pixel, the instrument stage of the correction as
        :func:`~tengri.observation.spectrum.lattice_correction_rest` takes it. A kernel whose
        width varies with wavelength carries the pixel's bins of the full lattice (``plan``).
    """

    pixels: np.ndarray
    nodes: np.ndarray
    offsets: tuple
    spans: tuple
    wave_nodes: np.ndarray
    ssp_cube: np.ndarray
    centres_rest: np.ndarray
    sigma_v_kms: float | None
    instruments: tuple


def interpolate_cube_onto(wave_ssp, ssp_flux, wave_nodes):
    """Library cube linearly interpolated onto model nodes, zero outside the library grid.

    The exact path's projection of the per-age cube onto the model grid
    (``jnp.interp`` with zero outside), done once at build for the window nodes.

    Returns
    -------
    ndarray, shape (..., n_nodes)
    """
    ssp_w = np.asarray(wave_ssp, dtype=np.float64)
    w = np.asarray(wave_nodes, dtype=np.float64)
    cube = np.asarray(ssp_flux)
    i = np.clip(np.searchsorted(ssp_w, w, side="right") - 1, 0, ssp_w.size - 2)
    t = (w - ssp_w[i]) / (ssp_w[i + 1] - ssp_w[i])
    inside = (w >= ssp_w[0]) & (w <= ssp_w[-1])
    values = cube[..., i] + (cube[..., i + 1] - cube[..., i]) * t
    return np.where(inside, values, 0.0)


def build_flagged_pixels(
    wave_model,
    wave_ssp,
    ssp_flux,
    centres_rest,
    flags,
    sigma_v_kms,
    instrument,
    sigma_window_kms,
):
    """Layout of the flagged pixels, or ``None`` when no pixel is flagged.

    Each window is the pixel's rest-frame source range widened by ``_WINDOW_SIGMAS`` times
    the broadening width on both sides, plus the nodes that bracket it. The nodes are those
    of the model grid, the grid the exact path attenuates and takes the bin mean on (it can
    carry nodes the library does not), and the library cube is interpolated onto them.
    """
    from tengri.observation.spectrum import _pixel_edges, lattice_window

    wave = np.asarray(wave_model, dtype=np.float64)
    centres = np.asarray(centres_rest, dtype=np.float64)
    pixels = np.flatnonzero(np.asarray(flags))
    if pixels.size == 0:
        return None
    edges = np.asarray(_pixel_edges(jnp.asarray(centres)))
    pad = kernel_reach_pad_aa(centres, sigma_window_kms, _WINDOW_SIGMAS)
    kept, windows, spans, unwindowed = [], [], [], []
    for p in pixels:
        lo, hi = edges[p] - pad[p], edges[p + 1] + pad[p]
        win = node_window(wave, lo, hi)
        # A pixel with no model node in reach, or one whose window reaches the ionizing side
        # of the Lyman edge (the nebular gas gate is not in the window SED), cannot get an
        # exact window. It stays with the table and is reported, never dropped silently.
        if win.size == 0 or lo < LYMAN_LIMIT_AA:
            unwindowed.append(int(p))
            continue
        first, last = max(int(win[0]) - 1, 0), min(int(win[-1]) + 2, wave.size)
        kept.append(p)
        windows.append(np.arange(first, last, dtype=np.int64))
        spans.append(lattice_window(lo, hi))
    if unwindowed:
        warnings.warn(
            f"{len(unwindowed)} flagged spectrum pixel(s) {unwindowed} have no exact node window "
            "(no model node in reach, or the window reaches the Lyman edge) and are served by "
            "the first-order table (#3413).",
            stacklevel=2,
        )
    if not kept:
        return None
    pixels = np.asarray(kept)
    instruments = _window_instruments(instrument, spans, lattice_window(edges[0], edges[-1]))
    nodes = np.concatenate(windows)
    offsets = tuple(int(x) for x in np.concatenate([[0], np.cumsum([w.size for w in windows])]))
    return FlaggedPixels(
        pixels=pixels,
        nodes=nodes,
        offsets=offsets,
        spans=tuple(spans),
        wave_nodes=wave[nodes],
        ssp_cube=interpolate_cube_onto(wave_ssp, ssp_flux, wave[nodes]),
        centres_rest=centres,
        sigma_v_kms=sigma_v_kms,
        instruments=instruments,
    )


def _window_instruments(instrument, spans, span_global):
    """The instrument stage of each window; a wavelength-varying kernel gets its bin plan.

    The kernel's bins are those of the full lattice (``span_global``), so a window reproduces
    the full-lattice result only with the weights of those bins on its own cells.
    """
    from tengri.observation.spectrum import (
        _LATTICE_DLNLAMBDA,
        _lattice_instrument_sigma,
        lattice_bin_plan,
    )

    if instrument is None:
        return tuple(None for _ in spans)
    origin, n_cells = span_global
    lam_c = np.exp(origin + _LATTICE_DLNLAMBDA * (np.arange(n_cells) + 0.5))
    reads = {k: v for k, v in instrument.items() if k != "redshift"}
    sigma = np.asarray(
        _lattice_instrument_sigma(reads, jnp.asarray(lam_c * (1.0 + instrument["redshift"])))
    )
    if sigma.ndim == 0:
        return tuple(instrument for _ in spans)
    return tuple(
        {**instrument, "plan": lattice_bin_plan(span, span_global, sigma, instrument["n_bins"])}
        for span in spans
    )


def flagged_pixel_means(layout, sed_nodes):
    """Exact pixel means of an SED on the flagged windows, with the table's lattice correction.

    The exact conserving path's value of the pixel: the bin mean of the SED over the pixel's
    edges (:func:`~tengri.observation.spectrum._flux_conserving_resample`) plus, when the
    table carries a velocity dispersion or an instrument kernel, the lattice correction on
    the pixel's own window.

    Parameters
    ----------
    layout : FlaggedPixels
    sed_nodes : array, shape (n_nodes,)
        The SED at ``layout.nodes`` [erg/s/Hz].

    Returns
    -------
    array, shape (n_flag,)
        One value per flagged pixel [erg/s/Hz].
    """
    from tengri.observation.spectrum import _flux_conserving_resample, lattice_correction_rest

    n_pix = layout.centres_rest.size
    out = []
    for k, p in enumerate(layout.pixels):
        lo_pix, hi_pix = max(int(p) - 1, 0), min(int(p) + 2, n_pix)
        centres = jnp.asarray(layout.centres_rest[lo_pix:hi_pix])
        a, b = layout.offsets[k], layout.offsets[k + 1]
        wave_k = jnp.asarray(layout.wave_nodes[a:b])
        sed_k = sed_nodes[a:b]
        value = _flux_conserving_resample(wave_k, sed_k, centres)[int(p) - lo_pix]
        if layout.sigma_v_kms is not None:
            value = (
                value
                + lattice_correction_rest(
                    sed_k,
                    wave_k,
                    centres,
                    layout.sigma_v_kms,
                    instrument=layout.instruments[k],
                    span=layout.spans[k],
                )[int(p) - lo_pix]
            )
        out.append(value)
    return jnp.stack(out)
