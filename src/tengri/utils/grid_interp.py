# SPDX-License-Identifier: BSD-3-Clause
"""Generic template preintegration through photometric filters.

Provides a universal function to collapse the wavelength dimension of any
template grid (SSP, CLOUDY, DL07, SKIRTOR, etc.) into filter-integrated
photometry. At runtime, N-dimensional triweight interpolation in the
grid parameter space + physical scaling gives photometry without ever
touching the full wavelength grid.

The triweight kernel (Hearin et al. 2023) gives C²-continuous gradients,
critical for gradient-based inference (VI, MAP, NUTS).

References
----------
- Zacharegkas et al. 2025, arXiv:2506.19919 (photometric precomputation)
- Hearin et al. 2023, Open J. Astrophysics, 6, 1 (triweight kernel / DSPS)
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from tengri.utils.filter_convention import FilterConvention, filter_weight_np as _filter_weight_np
from tengri.utils.interpolation import compute_grid_weights, edges_for_grid

__all__ = [
    "PreintegratedGrid",
    "PreintegratedLines",
    "interp_nd_pchip",
    "interp_nd_triweight",
    "preintegrate_grid",
    "preintegrate_lines",
    "slice_fixed_axes",
]

# numpy >= 2.0 uses trapezoid; older versions used trapz
_np_trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


@dataclasses.dataclass(frozen=True)
class PreintegratedGrid:
    """Template grid with wavelength dimension collapsed into filter integrals.

    Φ(*grid_dims, band) = ∫ L_ν(*grid_dims, λ) T_b(λ) λ dλ / ∫ T_b(λ) λ dλ

    Grid dimensions are opaque — the caller knows what they mean
    (e.g. (n_met, n_age) for SSP, (n_logU, n_Z) for CLOUDY).

    Parameters
    ----------
    phot : jnp.ndarray
        (*grid_dims, n_filters). Filter-integrated photometry.
    moment : jnp.ndarray or None
        (*grid_dims, n_filters). Taylor moment for dust correction,
        or None if not precomputed.
    axes : tuple[jnp.ndarray, ...]
        One array per grid dimension, giving node coordinates.
    edges : tuple[jnp.ndarray, ...]
        Precomputed bin edges for triweight interpolation, one per axis.
    effective_wavelengths : jnp.ndarray
        (n_filters,) observed-frame effective wavelengths [Ångström].
    effective_wavelengths_rest : jnp.ndarray
        (n_filters,) rest-frame effective wavelengths [Ångström].
    flux_scale : float
        Geometric scaling factor (1+z) / (4π d_L²) [cm⁻²].
    n_filters : int
        Number of filters.
    """

    phot: jnp.ndarray
    moment: jnp.ndarray | None
    axes: tuple[jnp.ndarray, ...]
    edges: tuple[jnp.ndarray, ...]
    effective_wavelengths: jnp.ndarray
    effective_wavelengths_rest: jnp.ndarray
    flux_scale: float
    n_filters: int


@dataclasses.dataclass(frozen=True)
class PreintegratedLines:
    """Emission lines preintegrated through filters via point-sampling.

    A line at λ_line contributes T_b(λ_line_obs) × λ_line_obs / ∫ T_b(λ) λ dλ
    to filter b. This is exact — no numerical integration needed.

    Parameters
    ----------
    line_filter_weights : jnp.ndarray
        (n_lines, n_filters). Weight of each line in each filter.
    axes : tuple[jnp.ndarray, ...]
        Grid axes for luminosity interpolation (if grid-based).
    edges : tuple[jnp.ndarray, ...]
        Precomputed bin edges for triweight interpolation.
    """

    line_filter_weights: jnp.ndarray
    axes: tuple[jnp.ndarray, ...]
    edges: tuple[jnp.ndarray, ...]


def _vectorized_interp(
    xp_target: np.ndarray, xp_source: np.ndarray, yp_source: np.ndarray
) -> np.ndarray:
    """Vectorized linear interpolation: all grid points at once.

    Replaces the inner ``for ...: np.interp(...)`` loops with a single
    vectorized NumPy operation. Computes interpolation indices and weights
    once, then applies via fancy indexing.

    Parameters
    ----------
    xp_target : array, shape (n_target,)
        Target x-coordinates (e.g. filter wavelengths).
    xp_source : array, shape (n_source,)
        Source x-coordinates (must be sorted ascending).
    yp_source : array, shape (..., n_source)
        Source y-values. All leading dimensions are preserved;
        interpolation occurs only on the trailing axis.

    Returns
    -------
    array, shape (..., n_target)
        Interpolated values, with out-of-bounds set to 0.
    """
    n_source = len(xp_source)

    # Compute interpolation indices and fractional weights once
    idx = np.searchsorted(xp_source, xp_target) - 1
    idx = np.clip(idx, 0, n_source - 2)

    dx = xp_source[idx + 1] - xp_source[idx]
    # Guard against zero-width bins (shouldn't happen with sorted grids)
    dx = np.where(dx > 0, dx, 1.0)
    frac = (xp_target - xp_source[idx]) / dx

    # Vectorized gather: fancy indexing on the trailing dimension
    result = (1.0 - frac) * yp_source[..., idx] + frac * yp_source[..., idx + 1]

    # Zero out-of-bounds (left and right)
    oob = (xp_target < xp_source[0]) | (xp_target > xp_source[-1])
    result[..., oob] = 0.0

    return result


def preintegrate_grid(
    templates: np.ndarray,
    wave_rest: np.ndarray,
    filter_waves: list[np.ndarray],
    filter_trans: list[np.ndarray],
    redshift: float,
    dl_cm: float,
    axes: tuple[np.ndarray, ...] = (),
    taylor: bool = False,
    energy_normalize: bool = False,
    convention: FilterConvention = FilterConvention.BESSELL,
) -> PreintegratedGrid:
    """Precompute filter-integrated photometry from a template grid.

    Collapses the wavelength dimension of a template grid into filter
    integrals. Supports arbitrary grid dimensionality (e.g. 2D for SSP's
    (n_met, n_age), 3D for CLOUDY's (n_logU, n_Z, n_Q_H), etc.).

    The filter integration formula (photon-counting ``BESSELL`` default,
    weight ``w = 1/λ``; ``ENERGY`` uses ``w = 1/λ²``) matches
    :func:`tengri.observation.photometry.compute_flux_density`:
        Φ = ∫ L_ν(λ_obs) T_b(λ_obs) w(λ_obs) dλ / ∫ T_b(λ_obs) w(λ_obs) dλ

    When ``taylor=True``, also precomputes the first spectral moment:
        Ψ = ∫ L_ν(λ) (λ - λ_eff) T(λ) w(λ) dλ / ∫ T(λ) w(λ) dλ
    where ``λ_eff = ∫ λ T w dλ / ∫ T w dλ`` is the weight's first moment, so
    Ψ ≡ 0 for a flat template.

    This enables first-order Taylor dust correction at runtime, reducing
    the age-dust-metallicity factorization error by ~5×.

    Parameters
    ----------
    templates : ndarray
        Shape (*grid_dims, n_wave). Template luminosity L_ν [erg/s/Hz]
        in rest frame.
    wave_rest : ndarray
        Shape (n_wave,). Rest-frame wavelengths [Ångström].
    filter_waves : list[ndarray]
        Wavelength grid per filter [Ångström], observed frame.
    filter_trans : list[ndarray]
        Transmission curve per filter (relative, 0-1).
    redshift : float
        Source redshift.
    dl_cm : float
        Luminosity distance [cm].
    axes : tuple[ndarray, ...]
        One array per grid dimension. Used for runtime interpolation.
        If empty, runtime interpolation is disabled.
    taylor : bool
        If True, precompute first spectral moment tensor. Default False.
    energy_normalize : bool
        If True, normalize each template to unit bolometric luminosity
        ``∫ L_ν dν = 1`` before filter integration, so runtime
        ``L_absorbed * lookup(...)`` produces correctly scaled photometry.
        Inputs MUST be ``L_ν`` [erg/s/Hz] — historically this branch
        divided by ``∫ L_ν dλ`` (frequency–wavelength mismatch), which
        caused a wavelength-shape-dependent normalisation error. Pass
        ``False`` if templates are already normalised at load time
        (Dale2014/Astrodust/BOSA/THEMIS) — the divide is then an
        unnecessary round-trip. Default False.
    convention : FilterConvention
        Bandpass weight ``w(λ)``: ``BESSELL`` (default, photon-counting
        ``1/λ``, matches DSPS) or ``ENERGY`` (``1/λ²``, matches CIGALE). The
        precomputed LUT bakes the convention in, so it must match the
        convention used by the exact path at evaluation time.

    Returns
    -------
    PreintegratedGrid
        Precomputed grid with phot, moment, axes, edges, etc.
    """
    templates = np.asarray(templates)
    wave_rest = np.asarray(wave_rest)

    # Extract grid shape (everything except the trailing wavelength dimension)
    *grid_dims, n_wave = templates.shape
    n_filters = len(filter_waves)

    # Flatten grid dimensions for vectorized processing
    # Shape: (n_grid_points, n_wave) where n_grid_points = prod(grid_dims)
    n_grid_points = int(np.prod(grid_dims)) if grid_dims else 1
    templates_flat = templates.reshape(n_grid_points, n_wave)

    # Normalize templates if requested (before integration).
    # Inputs are L_ν [erg/s/Hz]; bolometric luminosity is ∫ L_ν dν, NOT
    # ∫ L_ν dλ. Sort by ν ascending (i.e. λ descending) to keep the
    # trapezoid sign positive.
    if energy_normalize:
        c_aa = 2.99792458e18  # speed of light [Å/s]
        nu = c_aa / wave_rest
        sort_idx = np.argsort(nu)
        nu_sorted = nu[sort_idx]
        bol_lum = np.zeros(n_grid_points)
        for i in range(n_grid_points):
            bol_lum[i] = _np_trapezoid(templates_flat[i][sort_idx], nu_sorted)
        bol_lum = np.where(bol_lum > 0, bol_lum, 1.0)
        templates_flat = templates_flat / bol_lum[:, None]

    # Redshift wavelengths to observed frame
    wave_obs = wave_rest * (1.0 + redshift)

    # Compute filter effective wavelengths and integrals under weight w(λ).
    # denom = ∫ T w dλ ;  λ_eff = ∫ λ T w dλ / ∫ T w dλ  (weight first moment,
    # the self-consistent Taylor expansion centre).
    eff_waves_obs = np.zeros(n_filters)
    filter_denoms = np.zeros(n_filters)
    for f_idx, (fw, ft) in enumerate(zip(filter_waves, filter_trans)):
        fw_np = np.asarray(fw, dtype=np.float64)
        ft_np = np.asarray(ft, dtype=np.float64)
        tw_np = ft_np * _filter_weight_np(fw_np, convention)
        filter_denoms[f_idx] = _np_trapezoid(tw_np, fw_np)
        eff_waves_obs[f_idx] = _np_trapezoid(tw_np * fw_np, fw_np) / np.maximum(
            filter_denoms[f_idx], 1e-30
        )

    eff_waves_rest = eff_waves_obs / (1.0 + redshift)

    # Precompute filter-integrated photometry and moments
    phot_flat = np.zeros((n_grid_points, n_filters))
    moment_flat = np.zeros((n_grid_points, n_filters)) if taylor else None

    for f_idx, (fw, ft) in enumerate(zip(filter_waves, filter_trans)):
        fw_np = np.asarray(fw, dtype=np.float64)
        ft_np = np.asarray(ft, dtype=np.float64)
        denom = filter_denoms[f_idx]

        # Interpolate all templates to filter wavelengths
        # Shape: (n_grid_points, n_filter_waves)
        templates_on_filt = _vectorized_interp(fw_np, wave_obs, templates_flat)

        # Integrate: ∫ L_ν T w(λ) dλ   (w = 1/λ Bessell, 1/λ² energy)
        weight = (ft_np * _filter_weight_np(fw_np, convention))[None, :]
        integrand = templates_on_filt * weight
        num = _np_trapezoid(integrand, fw_np, axis=-1)
        phot_flat[:, f_idx] = num / np.maximum(denom, 1e-30)

        # Compute Taylor moment if requested
        if taylor:
            dlam = fw_np[None, :] - eff_waves_obs[f_idx]
            num_moment = _np_trapezoid(templates_on_filt * dlam * weight, fw_np, axis=-1)
            moment_flat[:, f_idx] = num_moment / np.maximum(denom, 1e-30)

    # Reshape back to original grid dimensions
    phot = jnp.array(phot_flat.reshape(*grid_dims, n_filters))
    moment = jnp.array(moment_flat.reshape(*grid_dims, n_filters)) if taylor else None

    # Compute flux scale (geometric factor). Avoid calling the ``@jit``'d
    # ``lnu_to_fnu`` and then ``float()``-casting its result — under a
    # surrounding jit trace ``dl_cm`` and ``redshift`` arrive as tracers
    # and ``float(traced)`` raises ``ConcretizationTypeError``. Inline the
    # math so the expression returns a Python float when inputs are concrete
    # and a JAX scalar when traced; both are valid downstream multipliers.
    import math as _math

    flux_scale = (1.0 + redshift) / (4.0 * _math.pi * dl_cm**2)

    # Convert axes to JAX arrays and precompute edges
    axes_jax = tuple(jnp.asarray(ax) for ax in axes)
    edges_jax = tuple(edges_for_grid(ax) for ax in axes_jax)

    return PreintegratedGrid(
        phot=phot,
        moment=moment,
        axes=axes_jax,
        edges=edges_jax,
        effective_wavelengths=jnp.asarray(eff_waves_obs),
        effective_wavelengths_rest=jnp.asarray(eff_waves_rest),
        flux_scale=flux_scale,
        n_filters=n_filters,
    )


def preintegrate_lines(
    line_wavelengths: np.ndarray,
    filter_waves: list[np.ndarray],
    filter_trans: list[np.ndarray],
    redshift: float,
    axes: tuple[np.ndarray, ...] = (),
    convention: FilterConvention = FilterConvention.BESSELL,
) -> PreintegratedLines:
    """Precompute emission line weights through filters.

    Each emission line at rest wavelength λ_line contributes to a filter
    via point-sampling of the filter transmission at the observed wavelength
    λ_line_obs = λ_line × (1 + z).

    The weight for line i in filter b mirrors the continuum convention
    (:func:`preintegrate_grid`) so line and continuum combine consistently:
        w_{ib} = T_b(λ_line_obs) × w(λ_line_obs) / ∫ T_b(λ) w(λ) dλ
    with ``w = 1/λ`` for ``BESSELL`` (default) and ``w = 1/λ²`` for ``ENERGY``.

    Parameters
    ----------
    line_wavelengths : ndarray
        Shape (n_lines,). Rest-frame wavelengths [Ångström].
    filter_waves : list[ndarray]
        Wavelength grid per filter [Ångström], observed frame.
    filter_trans : list[ndarray]
        Transmission curve per filter (relative, 0-1).
    redshift : float
        Source redshift.
    axes : tuple[ndarray, ...]
        Grid axes for interpolation (if the line fluxes are grid-based).
    convention : FilterConvention
        Bandpass weight; must match the continuum/exact path. Default
        ``BESSELL`` (``1/λ``).

    Returns
    -------
    PreintegratedLines
        Weights (n_lines, n_filters) and grid information.
    """
    line_wavelengths = np.asarray(line_wavelengths)
    n_lines = len(line_wavelengths)
    n_filters = len(filter_waves)

    # Redshift line wavelengths to observed frame
    line_wavelengths_obs = line_wavelengths * (1.0 + redshift)

    # Compute filter denominators (normalization): ∫ T w dλ
    filter_denoms = np.zeros(n_filters)
    for f_idx, (fw, ft) in enumerate(zip(filter_waves, filter_trans)):
        fw_np = np.asarray(fw, dtype=np.float64)
        ft_np = np.asarray(ft, dtype=np.float64)
        filter_denoms[f_idx] = _np_trapezoid(ft_np * _filter_weight_np(fw_np, convention), fw_np)

    # Compute line weights via point-sampling: T(λ_line) w(λ_line) / ∫ T w dλ
    line_filter_weights = np.zeros((n_lines, n_filters))
    for line_idx, lam_obs in enumerate(line_wavelengths_obs):
        lam_arr = np.asarray([lam_obs], dtype=np.float64)
        w_at_line = float(_filter_weight_np(lam_arr, convention)[0])
        for f_idx, (fw, ft) in enumerate(zip(filter_waves, filter_trans)):
            fw_np = np.asarray(fw, dtype=np.float64)
            ft_np = np.asarray(ft, dtype=np.float64)

            # Interpolate transmission at the line's observed wavelength.
            # ``left=right=0`` zeroes lines outside the filter passband (np.interp
            # otherwise returns the edge value, which the 1/λ weight would
            # amplify for blue out-of-band lines).
            T_at_line = np.interp(lam_obs, fw_np, ft_np, left=0.0, right=0.0)
            denom = filter_denoms[f_idx]
            line_filter_weights[line_idx, f_idx] = T_at_line * w_at_line / np.maximum(denom, 1e-30)

    # Convert axes and precompute edges
    axes_jax = tuple(jnp.asarray(ax) for ax in axes)
    edges_jax = tuple(edges_for_grid(ax) for ax in axes_jax)

    return PreintegratedLines(
        line_filter_weights=jnp.asarray(line_filter_weights),
        axes=axes_jax,
        edges=edges_jax,
    )


def _tensor_contract(grid: jnp.ndarray, weights_per_axis: list[jnp.ndarray]) -> jnp.ndarray:
    """Contract grid along leading axes with weight vectors.

    Each weight vector has shape (n_i,) and contracts along axis i.
    After each contraction, the next weight contracts along axis 0 of
    the reduced tensor.

    Parameters
    ----------
    grid : jnp.ndarray
        Shape (*grid_dims, n_trailing). Grid to contract.
    weights_per_axis : list[jnp.ndarray]
        One weight vector per grid dimension. w[i] has shape (grid_dims[i],).

    Returns
    -------
    jnp.ndarray
        Shape (n_trailing,) if grid was (*grid_dims, n_trailing).
    """
    result = grid
    for w in weights_per_axis:
        result = jnp.tensordot(w, result, axes=([0], [0]))
    return result


@jax.jit
def interp_nd_triweight(
    grid: jnp.ndarray,
    axes: tuple[jnp.ndarray, ...],
    edges: tuple[jnp.ndarray, ...],
    point: tuple,
    scatters: tuple | None = None,
) -> jnp.ndarray:
    """N-dimensional smooth interpolation via triweight kernel.

    Interpolates a grid over all dimensions using the triweight kernel
    (Hearin et al. 2023). Returns C²-continuous gradients, critical for
    gradient-based inference.

    The grid can have any shape; the trailing dimension is preserved.
    For example, a shape (*grid_dims, n_filters) returns shape (n_filters).

    Parameters
    ----------
    grid : jnp.ndarray
        Shape (*grid_dims, ...). Grid to interpolate. All leading dimensions
        are interpolation axes; trailing dimensions are preserved.
    axes : tuple[jnp.ndarray, ...]
        One axis per interpolation dimension. axes[i] has shape (grid_dims[i],).
    edges : tuple[jnp.ndarray, ...]
        Precomputed bin edges from edges_for_grid(), one per axis.
    point : tuple
        Query point coordinates. point[i] is a scalar on axis i.
    scatters : tuple or None
        Kernel bandwidths per axis. If None, uses 0.5 * (axis[i+1] - axis[i])
        for each axis (half grid spacing). Recommended for smooth inference.

    Returns
    -------
    jnp.ndarray
        Interpolated value(s). Shape is (...,) where (...) is the trailing
        dimensions of grid (all after the interpolation axes).

    Notes
    -----
    The triweight kernel integrates the CDF between bin edges, giving
    smooth weights that transition across grid nodes. This is superior to
    piecewise-linear interpolation, which has discontinuous first derivatives.
    """
    n_dims = len(axes)

    # Compute scatters if not provided
    if scatters is None:
        scatters_computed = []
        for ax in axes:
            dax = ax[1] - ax[0]  # First spacing (assumes uniform)
            scatters_computed.append(0.5 * dax)
        scatters = tuple(scatters_computed)

    # Compute weights along each axis
    weights_per_axis = []
    for i in range(n_dims):
        w = compute_grid_weights(point[i], axes[i], scatter=scatters[i], edges=edges[i])
        weights_per_axis.append(w)

    # Contract grid with weights
    return _tensor_contract(grid, weights_per_axis)


def _bcast_axis0(vec: jnp.ndarray, ndim: int) -> jnp.ndarray:
    """Reshape a length-k vector to broadcast against an array's leading axis."""
    return vec.reshape((vec.shape[0],) + (1,) * (ndim - 1))


def _pchip_edge_slope(
    h0: jnp.ndarray, h1: jnp.ndarray, m0: jnp.ndarray, m1: jnp.ndarray
) -> jnp.ndarray:
    """Shape-preserving one-sided endpoint slope (Fritsch & Carlson 1980).

    Mirrors SciPy's ``PchipInterpolator`` ``_edge_case`` so the endpoints stay
    monotone and never overshoot. ``h0, h1`` are the two boundary interval
    widths (scalars); ``m0, m1`` are the corresponding secant slopes (arrays).
    """
    d = ((2.0 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
    # Zero the slope at a local extremum; otherwise cap at 3x the boundary
    # secant so the cubic cannot overshoot.
    mask_extremum = jnp.sign(d) != jnp.sign(m0)
    mask_cap = (jnp.sign(m0) != jnp.sign(m1)) & (jnp.abs(d) > 3.0 * jnp.abs(m0))
    d = jnp.where(mask_extremum, 0.0, d)
    d = jnp.where((~mask_extremum) & mask_cap, 3.0 * m0, d)
    return d


def _pchip_slopes(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Monotone Hermite slopes at each node (Fritsch & Carlson 1980).

    Parameters
    ----------
    x : jnp.ndarray, shape (n,)
        Strictly ascending node coordinates.
    y : jnp.ndarray, shape (n, *rest)
        Node values; interpolation is over the leading axis.

    Returns
    -------
    jnp.ndarray, shape (n, *rest)
        Per-node tangents that make the piecewise cubic monotone (no overshoot)
        on monotone data and C¹-continuous everywhere.
    """
    h = jnp.diff(x)  # (n-1,)
    delta = jnp.diff(y, axis=0) / _bcast_axis0(h, y.ndim)  # (n-1, *rest)

    h_prev = h[:-1]  # (n-2,)
    h_next = h[1:]  # (n-2,)
    d_left = delta[:-1]  # (n-2, *rest)
    d_right = delta[1:]  # (n-2, *rest)

    w1 = _bcast_axis0(2.0 * h_next + h_prev, y.ndim)
    w2 = _bcast_axis0(h_next + 2.0 * h_prev, y.ndim)

    # Weighted harmonic mean of the bracketing secants. Guard the divisions so
    # the unused branch of the where() carries no NaN into the VJP.
    d_left_safe = jnp.where(d_left == 0.0, 1.0, d_left)
    d_right_safe = jnp.where(d_right == 0.0, 1.0, d_right)
    harmonic = (w1 + w2) / (w1 / d_left_safe + w2 / d_right_safe)
    interior = jnp.where(d_left * d_right > 0.0, harmonic, 0.0)  # (n-2, *rest)

    first = _pchip_edge_slope(h[0], h[1], delta[0], delta[1])
    last = _pchip_edge_slope(h[-1], h[-2], delta[-1], delta[-2])

    return jnp.concatenate([first[None], interior, last[None]], axis=0)


def _pchip_eval_axis0(x: jnp.ndarray, y: jnp.ndarray, xq) -> jnp.ndarray:
    """Evaluate the monotone-cubic interpolant at scalar ``xq`` over axis 0."""
    n = x.shape[0]
    slopes = _pchip_slopes(x, y)

    xq_c = jnp.clip(xq, x[0], x[-1])
    i = jnp.clip(jnp.searchsorted(x, xq_c) - 1, 0, n - 2)
    xi = x[i]
    h = x[i + 1] - xi
    t = (xq_c - xi) / h
    t2 = t * t
    t3 = t2 * t

    # Cubic Hermite basis functions.
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2

    return h00 * y[i] + h10 * h * slopes[i] + h01 * y[i + 1] + h11 * h * slopes[i + 1]


def interp_nd_pchip(
    grid: jnp.ndarray,
    axes: tuple[jnp.ndarray, ...],
    point: tuple,
) -> jnp.ndarray:
    """N-dimensional node-exact monotone-cubic (PCHIP) interpolation.

    Unlike the triweight *smoother* (:func:`interp_nd_triweight`), this is an
    *interpolant*: it passes exactly through the tabulated nodes while keeping
    C¹-continuous gradients. The per-axis tangents use the Fritsch & Carlson
    (1980) shape-preserving (monotone) rule, so the cubic never overshoots —
    safe even on sparse, nearest-neighbour-filled grids where a natural cubic
    spline would ring. Applied separably as a tensor product, one axis at a
    time.

    Use this for tabulated libraries whose feature position (e.g. an SED peak
    wavelength) shifts sharply across the grid, where the triweight kernel's
    neighbour-averaging would smear the feature. The cost relative to triweight
    is that gradients are only C¹ (not C²).

    Parameters
    ----------
    grid : jnp.ndarray, shape (*grid_dims, *trailing)
        Values to interpolate. The leading ``len(axes)`` dimensions are the
        interpolation axes; all trailing dimensions are preserved.
    axes : tuple[jnp.ndarray, ...]
        One strictly-ascending coordinate vector per interpolation dimension;
        ``axes[i]`` has shape ``(grid_dims[i],)``.
    point : tuple
        Query coordinates, one scalar per axis.

    Returns
    -------
    jnp.ndarray, shape (*trailing,)
        Interpolated values. At an exact node the tabulated value is returned
        to floating-point precision.

    Notes
    -----
    **JIT-compatible**: yes — pure ``jnp`` ops with gathers on traced indices.

    **Gradient-safe**: yes — C¹ in the query coordinates; guarded divisions in
    the slope computation keep the VJP finite at flat segments.

    Queries are clipped to the grid extent (no extrapolation beyond the
    boundary nodes).

    References
    ----------
    .. [1] F. N. Fritsch & R. E. Carlson, "Monotone Piecewise Cubic
       Interpolation," SIAM J. Numer. Anal. 17, 238 (1980).
       DOI: 10.1137/0717021.
    """
    reduced = grid
    for ax, p in zip(axes, point, strict=True):
        reduced = _pchip_eval_axis0(ax, reduced, p)
    return reduced


def slice_fixed_axes(
    preint: PreintegratedGrid | PreintegratedLines,
    fixed: dict[int, float],
) -> PreintegratedGrid | PreintegratedLines:
    """Collapse fixed axes in a preintegrated grid via triweight interpolation.

    When a grid parameter is fixed (not free during inference), its axis
    can be collapsed at init time.  This reduces the grid dimensionality
    and makes runtime interpolation cheaper.

    For example, if a CLOUDY grid has axes (logZ, log_age, logU) and
    logU is Fixed(-3.0), calling ``slice_fixed_axes(preint, {2: -3.0})``
    returns a grid with axes (logZ, log_age) — the logU axis is removed
    by triweight-interpolating to -3.0.

    Works for both PreintegratedGrid and PreintegratedLines.

    Parameters
    ----------
    preint : PreintegratedGrid or PreintegratedLines
        The preintegrated data to slice.
    fixed : dict[int, float]
        Mapping of axis index → fixed value.  Axes are numbered from 0.
        E.g. ``{2: -3.0}`` collapses axis 2 at value -3.0.

    Returns
    -------
    PreintegratedGrid or PreintegratedLines
        New object with reduced dimensionality.  Axes and edges for the
        fixed dimensions are removed.
    """
    if not fixed:
        return preint

    axes = list(preint.axes)
    edges = list(preint.edges)

    # Handle PreintegratedGrid (has phot and moment)
    if isinstance(preint, PreintegratedGrid):
        phot = preint.phot
        moment = preint.moment

        # Process in reverse order so axis indices remain valid after each slice
        for axis_idx in sorted(fixed.keys(), reverse=True):
            value = fixed[axis_idx]
            ax = axes[axis_idx]
            ed = edges[axis_idx]

            # Compute triweight weights at the fixed value
            w = compute_grid_weights(value, ax, scatter=0.5 * float(ax[1] - ax[0]), edges=ed)

            # Contract phot along this axis using einsum-style contraction.
            # tensordot(w, phot, ([0], [axis_idx])) removes axis_idx from phot,
            # preserving the order of all other axes.
            phot = jnp.tensordot(w, phot, axes=([0], [axis_idx]))

            if moment is not None:
                moment = jnp.tensordot(w, moment, axes=([0], [axis_idx]))

            # Remove from axes and edges lists
            axes.pop(axis_idx)
            edges.pop(axis_idx)

        return PreintegratedGrid(
            phot=phot,
            moment=moment,
            axes=tuple(axes),
            edges=tuple(edges),
            effective_wavelengths=preint.effective_wavelengths,
            effective_wavelengths_rest=preint.effective_wavelengths_rest,
            flux_scale=preint.flux_scale,
            n_filters=preint.n_filters,
        )

    # Handle PreintegratedLines (has line_filter_weights)
    elif isinstance(preint, PreintegratedLines):
        line_filter_weights = preint.line_filter_weights

        # Process in reverse order
        for axis_idx in sorted(fixed.keys(), reverse=True):
            value = fixed[axis_idx]
            ax = axes[axis_idx]
            ed = edges[axis_idx]

            # Compute triweight weights at the fixed value
            w = compute_grid_weights(value, ax, scatter=0.5 * float(ax[1] - ax[0]), edges=ed)

            # Contract line_filter_weights along this axis.
            # Shape (*grid_dims, n_lines, n_filters) → (*reduced_grid_dims, n_lines, n_filters)
            # Note: axis_idx indexes the grid dimensions (not n_lines or n_filters)
            line_filter_weights = jnp.tensordot(w, line_filter_weights, axes=([0], [axis_idx]))

            # Remove from axes and edges lists
            axes.pop(axis_idx)
            edges.pop(axis_idx)

        return PreintegratedLines(
            line_filter_weights=line_filter_weights,
            axes=tuple(axes),
            edges=tuple(edges),
        )

    else:
        raise TypeError(f"Unsupported type: {type(preint)}")
