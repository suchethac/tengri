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

from tengri.utils.conversions import lnu_to_fnu
from tengri.utils.interpolation import compute_grid_weights, edges_for_grid

__all__ = [
    "PreintegratedGrid",
    "PreintegratedLines",
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

    Attributes
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

    Attributes
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
) -> PreintegratedGrid:
    """Precompute filter-integrated photometry from a template grid.

    Collapses the wavelength dimension of a template grid into filter
    integrals. Supports arbitrary grid dimensionality (e.g. 2D for SSP's
    (n_met, n_age), 3D for CLOUDY's (n_logU, n_Z, n_Q_H), etc.).

    The filter integration formula:
        Φ = ∫ L_ν(λ_obs) T_b(λ_obs) λ_obs dλ / ∫ T_b(λ_obs) λ_obs dλ

    When ``taylor=True``, also precomputes the first spectral moment:
        Ψ = ∫ L_ν(λ) (λ - λ_eff) T(λ) λ dλ / ∫ T(λ) λ dλ

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
        before integration. Useful for DL07/SKIRTOR templates scaled
        by L_absorbed at runtime. Default False.

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

    # Normalize templates if requested (before integration)
    if energy_normalize:
        # Integrate each template over wavelength to get bolometric luminosity
        bol_lum = np.zeros(n_grid_points)
        for i in range(n_grid_points):
            bol_lum[i] = _np_trapezoid(templates_flat[i], wave_rest)
        # Normalize to unity (avoid division by zero)
        bol_lum = np.where(bol_lum > 0, bol_lum, 1.0)
        templates_flat = templates_flat / bol_lum[:, None]

    # Redshift wavelengths to observed frame
    wave_obs = wave_rest * (1.0 + redshift)

    # Compute filter effective wavelengths and integrals
    eff_waves_obs = np.zeros(n_filters)
    filter_denoms = np.zeros(n_filters)
    for f_idx, (fw, ft) in enumerate(zip(filter_waves, filter_trans)):
        fw_np = np.asarray(fw, dtype=np.float64)
        ft_np = np.asarray(ft, dtype=np.float64)
        filter_denoms[f_idx] = _np_trapezoid(ft_np * fw_np, fw_np)
        eff_waves_obs[f_idx] = _np_trapezoid(ft_np * fw_np**2, fw_np) / np.maximum(
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

        # Integrate: ∫ L_ν T λ dλ
        weight = ft_np[None, :] * fw_np[None, :]
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

    # Compute flux scale (geometric factor)
    flux_scale = float(lnu_to_fnu(1.0, dl_cm, redshift))

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
) -> PreintegratedLines:
    """Precompute emission line weights through filters.

    Each emission line at rest wavelength λ_line contributes to a filter
    via point-sampling of the filter transmission at the observed wavelength
    λ_line_obs = λ_line × (1 + z).

    The weight for line i in filter b is:
        w_{ib} = T_b(λ_line_obs) × λ_line_obs / ∫ T_b(λ) λ dλ

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

    # Compute filter denominators (normalization)
    filter_denoms = np.zeros(n_filters)
    for f_idx, (fw, ft) in enumerate(zip(filter_waves, filter_trans)):
        fw_np = np.asarray(fw, dtype=np.float64)
        ft_np = np.asarray(ft, dtype=np.float64)
        filter_denoms[f_idx] = _np_trapezoid(ft_np * fw_np, fw_np)

    # Compute line weights via point-sampling
    line_filter_weights = np.zeros((n_lines, n_filters))
    for line_idx, lam_obs in enumerate(line_wavelengths_obs):
        for f_idx, (fw, ft) in enumerate(zip(filter_waves, filter_trans)):
            fw_np = np.asarray(fw, dtype=np.float64)
            ft_np = np.asarray(ft, dtype=np.float64)

            # Interpolate transmission at the line's observed wavelength
            T_at_line = np.interp(lam_obs, fw_np, ft_np)
            denom = filter_denoms[f_idx]
            line_filter_weights[line_idx, f_idx] = T_at_line * lam_obs / np.maximum(denom, 1e-30)

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
