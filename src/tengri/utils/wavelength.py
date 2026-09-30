# SPDX-License-Identifier: BSD-3-Clause
"""Wavelength grid construction and interpolation utilities.

Provides functions for building panchromatic wavelength grids that extend
the SSP grid to X-ray and radio wavelengths, and for interpolating SEDs
between grids.

All functions are pure JAX, JIT-compatible.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from tengri.utils.grid_interp import resample_template
from tengri.utils.scale import representable_denominator

# Wavelength ranges (Angstrom)
# 0.0413 Angstrom = hc / (300 keV) with hc = 12.398 keV.Angstrom: the hard edge
# is set at 300 keV so the Yang+2020 / X-CIGALE corona exponential cutoff
# (default E_cut = 300 keV) is sampled and visible, rather than clipped at the
# old ~120 keV grid edge where the rollover has barely begun.
XRAY_WAVE_MIN: float = 0.0413  # ~300 keV hard X-ray (matches corona E_cut)
XRAY_WAVE_MAX: float = 100.0  # ~0.12 keV soft X-ray
RADIO_WAVE_MIN: float = 1e5  # 10 μm: overlap with SSP IR tail
RADIO_WAVE_MAX: float = 3e11  # ~1 MHz radio
# 1 m (300 MHz): declared reach of a tabulated nebular continuum's free-free tail (#2346)
NEBULAR_CONTINUUM_WAVE_MAX: float = 1e10


def make_union_grid(
    *arrays: np.ndarray | jnp.ndarray,
    dedupe_tol_rel: float = 1e-9,
) -> jnp.ndarray:
    """Sorted, deduplicated union of several wavelength grids in Angstrom.

    Each component's native template grid (dust emission, AGN torus, etc.) is
    declared in :mod:`tengri.forward.wavelength_extension`; this helper unions
    them with the SSP grid into the master rest-frame grid that
    ``SEDModel._init_multiwavelength`` exposes as ``state.wave``. The result
    is static, sorted ascending, and contains every input node up to a
    relative floating-point tolerance.

    Parameters
    ----------
    *arrays : array_like
        Wavelength grids in Angstrom. Empty / ``None`` arrays are ignored.
    dedupe_tol_rel : float
        Relative tolerance for collapsing near-coincident points (defaults
        to 1e-9, i.e. one part in a billion). Two values are considered
        equal when their relative gap falls below this tolerance.

    Returns
    -------
    jnp.ndarray
        Sorted, unique wavelength grid in Angstrom.

    Notes
    -----
    The dedupe step is the only knob that affects JIT shape: identical inputs
    always produce identical-shape output, but adding a new component grid
    changes the grid size and forces one re-compile. This is acceptable
    because the union is computed at ``SEDModel.build`` time, not per-call.
    """
    clean: list[np.ndarray] = []
    for a in arrays:
        if a is None:
            continue
        arr = np.asarray(a, dtype=np.float64).ravel()
        if arr.size == 0:
            continue
        arr = arr[np.isfinite(arr) & (arr > 0.0)]
        if arr.size > 0:
            clean.append(arr)

    if not clean:
        return jnp.asarray(np.empty(0, dtype=np.float64))

    merged = np.sort(np.concatenate(clean))
    if dedupe_tol_rel > 0.0 and merged.size > 1:
        # Relative-gap dedupe: keep a point only if its relative jump from
        # the previous kept point exceeds the tolerance.
        keep = np.empty(merged.shape, dtype=bool)
        keep[0] = True
        # Use the larger of the two endpoints for the relative scale so the
        # check is symmetric.
        rel_gap = np.diff(merged) / np.maximum(merged[:-1], representable_denominator(1e-300))
        keep[1:] = rel_gap > dedupe_tol_rel
        merged = merged[keep]
    return jnp.asarray(merged)


def make_panchromatic_grid(
    ssp_wave: np.ndarray | jnp.ndarray,
    extend_xray: bool = True,
    extend_radio: bool = True,
    n_per_decade: int = 20,
) -> jnp.ndarray:
    """Extend SSP wavelength grid with X-ray and radio wings.

    Builds a panchromatic grid by concatenating log-spaced X-ray and/or
    radio wavelengths with the original SSP grid. SSP wavelength points
    are preserved exactly (no resampling) so stellar SED values at those
    points have zero interpolation error.

    Parameters
    ----------
    ssp_wave : array (n_ssp,)
        Base SSP wavelength grid in Angstrom, sorted ascending.
    extend_xray : bool
        If True, prepend log-spaced points from 0.1 Å to the SSP minimum.
    extend_radio : bool
        If True, append log-spaced points from the SSP maximum to 3×10¹¹ Å.
    n_per_decade : int
        Number of log-spaced points per decade in the X-ray and radio wings.

    Returns
    -------
    jnp.ndarray (n_total,)
        Sorted, unique wavelengths in Angstrom. If both flags are False,
        returns ``ssp_wave`` unchanged.
    """
    if not extend_xray and not extend_radio:
        return jnp.asarray(ssp_wave)

    ssp_np = np.asarray(ssp_wave)
    parts = []

    if extend_xray:
        wave_min = XRAY_WAVE_MIN  # hard X-ray edge (~300 keV)
        wave_max = ssp_np[0]  # up to first SSP point (exclusive)
        if wave_min < wave_max:
            n_decades = np.log10(wave_max) - np.log10(wave_min)
            n_pts = max(int(n_decades * n_per_decade), 2)
            xray_wing = np.logspace(np.log10(wave_min), np.log10(wave_max), n_pts, endpoint=False)
            parts.append(xray_wing)

    parts.append(ssp_np)

    if extend_radio:
        wave_min = ssp_np[-1]  # from last SSP point (exclusive)
        wave_max = RADIO_WAVE_MAX
        if wave_min < wave_max:
            n_decades = np.log10(wave_max) - np.log10(wave_min)
            n_pts = max(int(n_decades * n_per_decade), 2)
            radio_wing = np.logspace(np.log10(wave_min), np.log10(wave_max), n_pts, endpoint=True)[
                1:
            ]  # skip first point (== ssp_wave[-1])
            parts.append(radio_wing)

    grid = np.concatenate(parts)
    # Ensure sorted and unique (should already be, but defensive)
    grid = np.unique(grid)
    return jnp.asarray(grid)


def interpolate_sed_to_grid(
    wave_src: jnp.ndarray,
    sed_src: jnp.ndarray,
    wave_target: jnp.ndarray,
) -> jnp.ndarray:
    """Interpolate SED to a new wavelength grid in log-log space.

    Uses log-log interpolation which is natural for power-law spectra
    (radio synchrotron, X-ray power laws). Values outside the source
    wavelength range are set to zero (no extrapolation).

    Parameters
    ----------
    wave_src : array (n_src,)
        Source wavelengths (Angstrom), sorted ascending.
    sed_src : array (n_src,)
        SED on source grid (erg/s/Hz or Lsun/Hz).
    wave_target : array (n_tgt,)
        Target wavelengths (Angstrom), sorted ascending.

    Returns
    -------
    array (n_tgt,)
        Interpolated SED. Zero outside source range.
    """
    # Single-sourced on ``resample_template`` so there is one log-log resampler
    # in the codebase, not two. That helper falls back to linear-in-flux on any
    # interval with a non-positive endpoint, which is better than this function's
    # previous ``maximum(sed, 1e-300)`` floor: flooring makes the interval
    # between a zero and a real value a near-vertical geometric ramp, whereas
    # the fallback interpolates it sensibly.
    return resample_template(wave_target, wave_src, sed_src, left=0.0, right=0.0)


def lyman_edge_transmission(
    wave: jnp.ndarray, neb_fesc: jnp.ndarray, edge_aa: float = 912.0
) -> jnp.ndarray:
    r"""Node-wise Lyman-continuum transmission, exact under trapezoid quadrature.

    ``where(wave < edge_aa, neb_fesc, 1)`` places the Lyman-limit step at
    whichever grid node sits just below ``edge_aa`` rather than at the
    physical edge itself (#2447): on ``fsps_mist_c3k_a_chabrier`` the SSP
    nodes bracketing 912 Å are 911.5716 and 913.3967 Å, so a trapezoid band
    integral over the naively-masked spectrum ramps transmission linearly
    across that 1.8 Å interval instead of stepping at 912 Å. This replaces
    only the two nodes bracketing the edge with the trapezoid weights that
    make the SAME single-panel quadrature integrate the true split --
    ``neb_fesc`` below 912 Å, 1 above -- exactly, to round-off; every other
    node keeps the plain step.

    Generic over ``wave``: called on the rest-frame SSP grid by
    ``NebularSEDComponent`` (``components/nebular/component.py``) for the
    dense stellar SED, and again on the finer observed-frame photometric
    UNION grid (SED nodes unioned with the filter table, potentially much
    finer near 912 Å for a well-sampled real filter) by
    ``observation.photometry._filter_integral_union``'s exact-band-flux
    correction (#2447 residual on finely-sampled filters): the single-panel
    exactness claimed below is a property of whichever grid is passed in,
    not specific to the SSP spacing.

    Parameters
    ----------
    wave : array_like, shape (n_wave,)
        Ascending wavelength grid [Angstrom], any frame (rest or observed,
        as long as ``edge_aa`` is expressed in the SAME frame).
    neb_fesc : array_like, scalar
        Lyman-continuum escape fraction, in [0, 1].
    edge_aa : float, optional
        Lyman-limit wavelength in the SAME frame as ``wave`` [Angstrom].
        Default 912.0 (rest-frame).

    Returns
    -------
    ndarray, shape (n_wave,)
        Per-node transmission factor: ``neb_fesc`` below the edge and 1
        above it, except at the two nodes bracketing ``edge_aa``, which
        carry the trapezoid-exact split weights below.

    Notes
    -----
    **JIT/grad-safe.** Pure ``jnp`` primitives, fully vectorized via boolean
    masks and shifted-array differences (no data-dependent indexing). The two
    bracketing-interval widths are guarded against a zero denominator
    (``edge_aa`` outside the grid's range, never true for a real SSP grid) so
    the unselected ``jnp.where`` branch never carries a 0/0 that would poison
    the ``neb_fesc`` gradient.

    **Derivation.** Let :math:`[\lambda_a, \lambda_b]` (width :math:`h`) be
    the grid interval straddling the edge, :math:`s = (912 - \lambda_a)/h \in
    [0, 1]`, and :math:`F` the spectrum, linear between grid nodes by
    construction (the quadrature's own assumption). The exact split integral
    of :math:`F` over :math:`[\lambda_a, \lambda_b]` (weight ``neb_fesc``
    below 912 Å, 1 above) equals the ordinary trapezoid formula
    :math:`(h/2)(w_a F_a + w_b F_b)` with

    .. math::

        w_a &= 2\left[f_{\rm esc}\left(s - \frac{s^2}{2}\right)
               + \left(\frac{1}{2} - s + \frac{s^2}{2}\right)\right] \\
        w_b &= 2\left[f_{\rm esc}\frac{s^2}{2} + \frac{1}{2} - \frac{s^2}{2}\right]

    Node :math:`a` also closes the (fully-masked) interval to its left
    (width :math:`h_l`, weight ``neb_fesc``) and node :math:`b` the
    (fully-unmasked) interval to its right (width :math:`h_r`, weight 1), so
    the single per-node factors that make the *ordinary* (unmodified)
    trapezoid sum exact overall are

    .. math::

        T_a = \frac{f_{\rm esc} h_l + w_a h}{h_l + h}, \qquad
        T_b = \frac{w_b h + h_r}{h + h_r}

    Both are affine in ``neb_fesc`` (:math:`T = 1 - C(1 - f_{\rm esc})` for a
    fesc-independent :math:`C`), so composing this array with a further
    per-age young/old blend (``dust/two_component.py``'s
    ``1 - y(a)(1 - lyc_transmission)``) commutes: substituting an
    age-blended escape fraction into :math:`T_a`/:math:`T_b` gives the
    identical result as blending :math:`T_a`/:math:`T_b` themselves, so this
    fix is exact for that consumer too with no changes there. The SAME
    affine property also lets a caller recover the pure LyC-band SELECTION
    weight (independent of ``neb_fesc``) as ``1 - lyman_edge_transmission(wave,
    0.0, edge_aa)``: at ``neb_fesc=0`` this function returns exactly
    :math:`1 - C`, the fraction of each node's quadrature weight that falls
    below the edge.
    """
    wave = jnp.asarray(wave)
    fesc = jnp.asarray(neb_fesc)
    below = wave < edge_aa

    below_next = jnp.concatenate([below[1:], jnp.array([True])])
    below_prev = jnp.concatenate([jnp.array([True]), below[:-1]])
    is_a = below & (~below_next)  # last node below the edge
    is_b = (~below) & below_prev  # first node at/above the edge

    wave_prev = jnp.concatenate([wave[:1], wave[:-1]])
    wave_next = jnp.concatenate([wave[1:], wave[-1:]])
    h_left = wave - wave_prev
    h_right = wave_next - wave

    lam_a = jnp.sum(jnp.where(is_a, wave, 0.0))
    h_l = jnp.sum(jnp.where(is_a, h_left, 0.0))
    h = jnp.sum(jnp.where(is_a, h_right, 0.0))
    h_r = jnp.sum(jnp.where(is_b, h_right, 0.0))

    h_safe = jnp.where(h > 0.0, h, 1.0)
    s = (edge_aa - lam_a) / h_safe
    w_a = 2.0 * (fesc * (s - s * s / 2.0) + (0.5 - s + s * s / 2.0))
    w_b = 2.0 * (fesc * s * s / 2.0 + 0.5 - s * s / 2.0)
    left_denom = jnp.where((h_l + h) > 0.0, h_l + h, 1.0)
    right_denom = jnp.where((h + h_r) > 0.0, h + h_r, 1.0)
    t_a = (fesc * h_l + w_a * h) / left_denom
    t_b = (w_b * h + h_r) / right_denom

    base = jnp.where(below, fesc, jnp.ones_like(wave))
    return jnp.where(is_a, t_a, jnp.where(is_b, t_b, base))
