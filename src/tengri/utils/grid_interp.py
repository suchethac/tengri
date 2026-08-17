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
- Hearin et al. 2023, MNRAS, 521, 1741 (triweight kernel / DSPS)

"""

from __future__ import annotations

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np

from tengri.utils.filter_convention import FilterConvention, filter_weight_np as _filter_weight_np
from tengri.utils.interpolation import compute_grid_weights, edges_for_grid
from tengri.utils.physics_constants import C_AA
from tengri.utils.scale import log10_flux_scale as _log10_flux_scale

__all__ = [
    "PreintegratedGrid",
    "PreintegratedLines",
    "interp_nd_pchip",
    "interp_nd_triweight",
    "pchip_interp_1d",
    "preintegrate_grid",
    "preintegrate_lines",
    "slice_fixed_axes",
]

# numpy >= 2.0 uses trapezoid; older versions used trapz
_np_trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def _cumtrapz_rows(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Cumulative trapezoid along the last axis, starting at 0.

    Shapes: y (..., m), x (m,) -> (..., m).
    """
    seg = 0.5 * (y[..., 1:] + y[..., :-1]) * np.diff(x)
    head = np.zeros((*y.shape[:-1], 1), dtype=seg.dtype)
    return np.concatenate([head, np.cumsum(seg, axis=-1)], axis=-1)


def _interp_rows(xq: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Linear interpolation of every row of ``y`` (over ``x``) at ``xq``.

    ``np.interp`` takes a single 1-D table, so it cannot do this in one call.
    Shapes: xq (q,), x (m,), y (..., m) -> (..., q).
    """
    idx = np.clip(np.searchsorted(x, xq) - 1, 0, x.size - 2)
    x0, x1 = x[idx], x[idx + 1]
    span = x1 - x0
    t = np.where(span > 0, (xq - x0) / np.where(span > 0, span, 1.0), 0.0)
    y0, y1 = y[..., idx], y[..., idx + 1]
    return y0 + (y1 - y0) * t


def subband_quadrature(
    grid: np.ndarray,
    tw_grid: np.ndarray,
    integrand: np.ndarray,
    denom: float,
    n_subbands: int,
    eff_wave_obs: float,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Sub-band weights and quadrature nodes for one filter (#1122).

    Splits the band into ``K`` sub-bands of equal filter mass and returns, per
    template, the filter integral restricted to each and the template's own
    flux-weighted centroid there:

    .. math::

        \Phi_k = \int_{\lambda_k}^{\lambda_{k+1}} L_\nu T w \,d\lambda
                 \Big/ \int T w \,d\lambda
        \qquad
        \bar\lambda_k = \frac{\int_k \lambda L_\nu T w \,d\lambda}
                             {\int_k L_\nu T w \,d\lambda}

    Both are attenuation-independent, so a multiplicative screen is applied at
    runtime as :math:`\sum_k \Phi_k A(\bar\lambda_k)` — an exact factorization.

    The single implementation, shared by the fixed-z (:func:`preintegrate_grid`)
    and free-z (ztable) precomputes so the two cannot drift.

    Parameters
    ----------
    grid : ndarray, shape (m,)
        Union quadrature grid, observed frame [Angstrom].
    tw_grid : ndarray, shape (m,)
        Transmission x filter weight on ``grid`` [dimensionless].
    integrand : ndarray, shape (..., m)
        ``templates * tw_grid`` — the thing whose band integral is wanted.
    denom : float
        ``int tw dlambda``, the filter normalization.
    n_subbands : int
        Number of sub-bands K.
    eff_wave_obs : float
        Filter effective wavelength, observed frame — the node fallback where a
        template has no flux in a sub-band [Angstrom].

    Returns
    -------
    phi : ndarray, shape (..., K)
        Sub-band filter integrals. Sums over K to the full band integral.
    nodes : ndarray, shape (..., K)
        Quadrature nodes, observed frame [Angstrom].

    Raises
    ------
    AssertionError
        If the partition is not flux-conserving.

    Notes
    -----
    **JIT-compatible**: no — build-time numpy precompute.

    The partition is built from CUMULATIVE integrals interpolated at the edges.
    Selecting grid points inside each sub-band instead silently drops the
    fractional interval at every edge: the sub-integrals stop summing to the
    whole and the quadrature error then GROWS with K. Conservation is asserted.
    """
    K = int(n_subbands)
    cum_w = _cumtrapz_rows(tw_grid, grid)
    edges = np.interp(np.linspace(0.0, cum_w[-1], K + 1), cum_w, grid)

    cum_sw = _cumtrapz_rows(integrand, grid)
    cum_lsw = _cumtrapz_rows(integrand * grid, grid)
    i_k = np.diff(_interp_rows(edges, grid, cum_sw), axis=-1)
    l_k = np.diff(_interp_rows(edges, grid, cum_lsw), axis=-1)

    total = cum_sw[..., -1]
    resid = np.abs(i_k.sum(axis=-1) - total)
    scale = np.maximum(np.abs(total), 1e-300)
    if not np.all(resid / scale < 1e-10):
        raise AssertionError(
            "sub-band partition is not flux-conserving (max relative residual "
            f"{float(np.max(resid / scale)):.3e}). The sub-band integrals must "
            "sum to the full filter integral."
        )

    # The node is the template's OWN flux-weighted centroid in the sub-band —
    # that is what makes the rule track the spectrum and converge as 1/K^2, and
    # it also makes the first moment vanish identically, so a Taylor term on top
    # would add exactly zero. Where a template has no flux the weight is zero, so
    # the node cannot change the result — but it is still fed through the dust
    # law, which goes as 1/lambda, so it must stay finite and positive.
    live = i_k != 0.0
    nodes = np.where(live, l_k / np.where(live, i_k, 1.0), eff_wave_obs)
    return i_k / np.maximum(denom, 1e-30), nodes


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
    subband_phot : jnp.ndarray or None
        (*grid_dims, n_filters, n_subbands). Filter integral restricted to
        each sub-band, or None. Sums over the sub-band axis to ``phot``.
    subband_waves : jnp.ndarray or None
        (*grid_dims, n_filters, n_subbands). Observed-frame quadrature node
        of each sub-band — the template's own flux-weighted centroid there,
        so it tracks the spectrum. [Ångström]
    subband_waves_rest : jnp.ndarray or None
        (*grid_dims, n_filters, n_subbands). Same nodes, rest frame. [Ångström]
    axes : tuple[jnp.ndarray, ...]
        One array per grid dimension, giving node coordinates.
    edges : tuple[jnp.ndarray, ...]
        Precomputed bin edges for triweight interpolation, one per axis.
    effective_wavelengths : jnp.ndarray
        (n_filters,) observed-frame effective wavelengths [Ångström].
    effective_wavelengths_rest : jnp.ndarray
        (n_filters,) rest-frame effective wavelengths [Ångström].
    log10_flux_scale : float
        ``log10`` of the geometric scaling factor (1+z) / (4π d_L²)
        [dex re cm⁻²]. Stored as a log because the linear factor is ~1e-57 and
        exactly ``0.0`` in float32 at every distance, so a stored linear scale is
        zeroed by the cast alone (#1859). Apply it with
        :func:`tengri.utils.scale.apply_log10_scale`.
    n_filters : int
        Number of filters.
    """

    phot: jnp.ndarray
    moment: jnp.ndarray | None
    axes: tuple[jnp.ndarray, ...]
    edges: tuple[jnp.ndarray, ...]
    effective_wavelengths: jnp.ndarray
    effective_wavelengths_rest: jnp.ndarray
    log10_flux_scale: float
    n_filters: int
    subband_phot: jnp.ndarray | None = None
    subband_waves: jnp.ndarray | None = None
    subband_waves_rest: jnp.ndarray | None = None


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
    n_subbands: int = 0,
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

    When ``n_subbands=K > 0``, the same integral is also computed **restricted
    to each of K sub-bands** carrying equal filter mass, together with the
    template's own flux-weighted centroid in each:

    .. math::

        \\Phi_{k} = \\int_{\\lambda_k}^{\\lambda_{k+1}} L_\\nu T w \\,d\\lambda
                    \\Big/ \\int T w \\,d\\lambda
        \\qquad
        \\bar\\lambda_{k} = \\frac{\\int_{\\lambda_k}^{\\lambda_{k+1}}
            \\lambda L_\\nu T w \\,d\\lambda}
            {\\int_{\\lambda_k}^{\\lambda_{k+1}} L_\\nu T w \\,d\\lambda}

    Both are attenuation-independent, so a *multiplicative* screen A(λ) is
    then applied at runtime as a K-point quadrature, exactly factorized:

    .. math::

        \\int L_\\nu A T w \\,d\\lambda \\Big/ \\int T w \\,d\\lambda
        \\;\\approx\\; \\sum_k \\Phi_k \\, A(\\bar\\lambda_k)

    This supersedes ``taylor``. The Taylor form evaluates A at one point per
    filter and *extrapolates* linearly away from it, which diverges where the
    attenuation curve steepens (GALEX FUV: +45 % at z=0.05 rising to +215 % at
    z=1). The quadrature *evaluates* A instead of extrapolating it and converges
    as 1/K²: K=3 holds ≲1.2 % in FUV, K=5 ≲0.5 %. See #1122.

    Only *multiplicative* transformations need this. Additive emitters (dust IR,
    radio, X-ray, AGN) factorize exactly through the rank-1/rank-K band response
    (#1107, #1117) and need no sub-bands.

    When ``taylor=True``, precomputes instead the first spectral moment:
        Ψ = ∫ L_ν(λ) (λ - λ_eff) T(λ) w(λ) dλ / ∫ T(λ) w(λ) dλ
    where ``λ_eff = ∫ λ T w dλ / ∫ T w dλ`` is the weight's first moment, so
    Ψ ≡ 0 for a flat template.

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
        Superseded by ``n_subbands``; see the equations above.
    n_subbands : int
        Number of equal-filter-mass sub-bands K for the multiplicative
        quadrature. ``0`` (default) disables it. K=3 is the recommended
        working point: it is *cheaper* than the Taylor moment at runtime
        (one tensor and an ``exp``, versus two tensors and a ``pow``) and
        ~40× more accurate in the rest-UV.
    energy_normalize : bool
        If True, normalize each template to unit bolometric luminosity
        ``∫ L_ν dν = 1`` before filter integration, so runtime
        ``L_absorbed * lookup(...)`` produces correctly scaled photometry.
        Inputs MUST be ``L_ν`` [erg/s/Hz] — historically this branch
        divided by ``∫ L_ν dλ`` (frequency–wavelength mismatch), which
        caused a wavelength-shape-dependent normalization error. Pass
        ``False`` if templates are already normalized at load time
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
        nu = C_AA / wave_rest
        sort_idx = np.argsort(nu)
        nu_sorted = nu[sort_idx]
        bol_lum = np.zeros(n_grid_points)
        for i in range(n_grid_points):
            bol_lum[i] = _np_trapezoid(templates_flat[i][sort_idx], nu_sorted)
        bol_lum = np.where(bol_lum > 0, bol_lum, 1.0)
        templates_flat = templates_flat / bol_lum[:, None]

    # Redshift wavelengths to observed frame
    wave_obs = wave_rest * (1.0 + redshift)

    # Precompute filter-integrated photometry and moments on the union
    # quadrature grid — template nodes + filter nodes, with the smooth
    # transmission interpolated, never the template point-sampled at the
    # filter table's nodes (#960). Matches lnu_filter_integral so the LUT
    # and exact paths agree. λ_eff = ∫ λ T w dλ / ∫ T w dλ (weight first
    # moment, the self-consistent Taylor expansion center) is evaluated on
    # the SAME union grid so the moment Ψ vanishes for a flat template.
    eff_waves_obs = np.zeros(n_filters)
    phot_flat = np.zeros((n_grid_points, n_filters))
    moment_flat = np.zeros((n_grid_points, n_filters)) if taylor else None
    K = int(n_subbands)
    sub_phot = np.zeros((n_grid_points, n_filters, K)) if K > 0 else None
    sub_waves = np.zeros((n_grid_points, n_filters, K)) if K > 0 else None

    for f_idx, (fw, ft) in enumerate(zip(filter_waves, filter_trans)):
        fw_np = np.asarray(fw, dtype=np.float64)
        ft_np = np.asarray(ft, dtype=np.float64)
        grid = np.sort(np.concatenate([wave_obs, fw_np]))
        # Transmission is identically zero outside the filter table's
        # support (left=0/right=0 below), so union-grid segments with both
        # endpoints beyond an edge integrate to exactly zero. Keep one node
        # past each edge (the straddle segments) and drop the rest — same
        # clip as precompute_photometry_ztable.
        lo = max(np.searchsorted(grid, fw_np[0]) - 1, 0)
        hi = min(np.searchsorted(grid, fw_np[-1], side="right") + 1, grid.size)
        grid = grid[lo:hi]
        trans_on_grid = np.interp(grid, fw_np, ft_np, left=0.0, right=0.0)
        tw_grid = trans_on_grid * _filter_weight_np(grid, convention)
        denom = _np_trapezoid(tw_grid, grid)
        eff_waves_obs[f_idx] = _np_trapezoid(tw_grid * grid, grid) / np.maximum(denom, 1e-30)

        # Interpolate all templates onto the union grid
        # Shape: (n_grid_points, len(grid))
        templates_on_grid = _vectorized_interp(grid, wave_obs, templates_flat)

        # Integrate: ∫ L_ν T w(λ) dλ   (w = 1/λ Bessell, 1/λ² energy)
        weight = tw_grid[None, :]
        integrand = templates_on_grid * weight
        num = _np_trapezoid(integrand, grid, axis=-1)
        phot_flat[:, f_idx] = num / np.maximum(denom, 1e-30)

        # Compute Taylor moment if requested
        if taylor:
            dlam = grid[None, :] - eff_waves_obs[f_idx]
            num_moment = _np_trapezoid(templates_on_grid * dlam * weight, grid, axis=-1)
            moment_flat[:, f_idx] = num_moment / np.maximum(denom, 1e-30)

        # Sub-band quadrature nodes and weights (#1122) — the single
        # implementation, shared with the free-z ztable precompute so the two
        # cannot drift.
        if K > 0:
            sub_phot[:, f_idx, :], sub_waves[:, f_idx, :] = subband_quadrature(
                grid, tw_grid, integrand, denom, K, float(eff_waves_obs[f_idx])
            )

    eff_waves_rest = eff_waves_obs / (1.0 + redshift)

    # Reshape back to original grid dimensions
    phot = jnp.array(phot_flat.reshape(*grid_dims, n_filters))
    moment = jnp.array(moment_flat.reshape(*grid_dims, n_filters)) if taylor else None
    if K > 0:
        sub_phot_j = jnp.array(sub_phot.reshape(*grid_dims, n_filters, K))
        sub_waves_j = jnp.array(sub_waves.reshape(*grid_dims, n_filters, K))
        sub_waves_rest_j = sub_waves_j / (1.0 + redshift)
    else:
        sub_phot_j = sub_waves_j = sub_waves_rest_j = None

    # Geometric factor, as a log10 offset. Not ``lnu_to_fnu`` + ``float()``:
    # under a surrounding jit trace ``dl_cm`` and ``redshift`` arrive as tracers
    # and ``float(traced)`` raises ``ConcretizationTypeError``. The shared helper
    # returns a Python-float-compatible scalar when inputs are concrete and a JAX
    # scalar when traced, and — unlike the linear form this replaced — survives
    # float32, where (1+z)/(4 pi d_L^2) is exactly 0.0 at every distance (#1859).
    log10_flux_scale = _log10_flux_scale(redshift, dl_cm)

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
        log10_flux_scale=log10_flux_scale,
        n_filters=n_filters,
        subband_phot=sub_phot_j,
        subband_waves=sub_waves_j,
        subband_waves_rest=sub_waves_rest_j,
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


@functools.partial(jax.jit, static_argnames=("index_space_interp",))
def interp_nd_triweight(
    grid: jnp.ndarray,
    axes: tuple[jnp.ndarray, ...],
    edges: tuple[jnp.ndarray, ...],
    point: tuple,
    scatters: tuple | None = None,
    index_space_interp: bool | None = None,
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
    index_space_interp : bool or None
        Whether to use index-space interpolation for non-uniform axes.
        Passed through to :func:`compute_grid_weights`. See its documentation
        for details. Default None (see compute_grid_weights).

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

    **Non-uniform axis handling (I6, #1851)**: For explicitly non-uniform grids
    (e.g. Fritz tau, Nenkova tau), pass ``index_space_interp=True`` to use the
    corrected index-space path. This eliminates the nearest-neighbor degeneracy
    and produces smooth gradients throughout the grid range. The interpolant is
    C2 within intervals and C0 at nodes where adjacent spacings differ.
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
        w = compute_grid_weights(
            point[i], axes[i], scatter=scatters[i], edges=edges[i],
            index_space_interp=index_space_interp
        )
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

    # Weighted harmonic mean of the bracketing secants, used only where the two
    # secants share a sign (monotonic). Gate the division INPUTS on that SAME
    # condition (double-where), not merely on d==0: for non-monotonic data the
    # secants have opposite signs, so w1/d_left + w2/d_right can pass through 0,
    # sending harmonic->inf. The outer where() discards that value, but its VJP
    # then forms 0*inf = NaN (poisoning the gradient of any param the grid
    # values depend on — e.g. the SKIRTOR dust-template pchip in
    # skirtor_disc_dust_ratio, #892). Replacing d_left/d_right with 1.0 wherever
    # the harmonic is unused keeps the denominator = w1+w2 > 0, so the reverse
    # pass stays finite. Forward is unchanged: interior is 0 in exactly the same
    # places (d_left*d_right <= 0).
    mono = d_left * d_right > 0.0
    d_left_safe = jnp.where(mono, d_left, 1.0)
    d_right_safe = jnp.where(mono, d_right, 1.0)
    harmonic = (w1 + w2) / (w1 / d_left_safe + w2 / d_right_safe)
    interior = jnp.where(mono, harmonic, 0.0)  # (n-2, *rest)

    first = _pchip_edge_slope(h[0], h[1], delta[0], delta[1])
    last = _pchip_edge_slope(h[-1], h[-2], delta[-1], delta[-2])

    return jnp.concatenate([first[None], interior, last[None]], axis=0)


def _pchip_eval_axis0(x: jnp.ndarray, y: jnp.ndarray, xq, *, extrapolate: bool = False):
    """Evaluate the monotone-cubic interpolant at ``xq`` over axis 0.

    ``extrapolate=False`` (the default, and what the tensor-product path uses)
    clamps queries to ``[x[0], x[-1]]``, so outside the table the value is held
    at the boundary node. ``extrapolate=True`` lets the edge cubic continue past
    the end nodes, which is what the ProSpect ``spline`` SFH relies on.
    """
    n = x.shape[0]
    slopes = _pchip_slopes(x, y)

    xq_c = xq if extrapolate else jnp.clip(xq, x[0], x[-1])
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


def _linear_eval_axis0(x: jnp.ndarray, y: jnp.ndarray, xq) -> jnp.ndarray:
    """Evaluate the piecewise-linear interpolant at scalar ``xq`` over axis 0.

    C0, not C1 — which is the point. Where the tabulated function has a genuine
    kink and a knot sits on it, a cubic's two-sided slope estimate straddles the
    corner and degrades to O(h); linear interpolation has no slope to get wrong.
    """
    n = x.shape[0]
    xq_c = jnp.clip(xq, x[0], x[-1])
    i = jnp.clip(jnp.searchsorted(x, xq_c) - 1, 0, n - 2)
    h = x[i + 1] - x[i]
    t = (xq_c - x[i]) / jnp.where(h > 0.0, h, 1.0)  # guard a degenerate cell
    return (1.0 - t) * y[i] + t * y[i + 1]


#: Per-axis evaluators selectable via ``interp_nd_pchip(..., kinds=...)``.
_EVAL_AXIS0 = {"pchip": _pchip_eval_axis0, "linear": _linear_eval_axis0}


def pchip_interp_1d(
    x: jnp.ndarray,
    y: jnp.ndarray,
    xq: jnp.ndarray,
    *,
    extrapolate: bool = False,
) -> jnp.ndarray:
    """Monotone piecewise-cubic (PCHIP) interpolation of a 1-D table.

    The single 1-D PCHIP entry point for the package. Shape-preserving tangents
    follow Fritsch & Carlson (1980) [1]_ with SciPy's boundary rule, so the
    result matches :class:`scipy.interpolate.PchipInterpolator` to round-off,
    passes exactly through every node, and never overshoots on monotone data.

    Parameters
    ----------
    x : array_like, shape (n,)
        Strictly ascending node coordinates [any consistent unit].
    y : array_like, shape (n,)
        Node values [any unit].
    xq : array_like, shape (m,)
        Query coordinates, same unit as ``x``. Need not be sorted.
    extrapolate : bool, optional
        If ``False`` (default), queries outside ``[x[0], x[-1]]`` are clamped to
        the boundary node value. If ``True``, the edge cubic continues past the
        end nodes.

    Returns
    -------
    ndarray, shape (m,)
        Interpolated values [same unit as ``y``].

    Notes
    -----
    **JIT/grad/vmap compatible**: yes. Interior tangents are the *weighted*
    harmonic mean of the two bracketing secants,

    .. math::

        \\frac{w_0 + w_1}{w_0/\\delta_0 + w_1/\\delta_1}, \\quad
        w_0 = 2h_1 + h_0, \\quad w_1 = h_1 + 2h_0,

    where :math:`h_i` are the interval widths and :math:`\\delta_i` the secant
    slopes. The weights matter only on a **non-uniform** grid — dropping them
    reduces to the unweighted harmonic mean, which is a silent error wherever
    the nodes are not equally spaced.

    The division inputs are gated on the same monotonicity condition as the
    output (the "double ``where``" pattern). Gating only the output lets the
    denominator pass through zero on non-monotonic data, and the discarded
    ``inf`` then forms ``0 * inf = NaN`` in the reverse pass.

    References
    ----------
    .. [1] F. N. Fritsch and R. E. Carlson, "Monotone Piecewise Cubic
       Interpolation," SIAM J. Numer. Anal., 17(2), 238-246 (1980).
       https://doi.org/10.1137/0717021

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> x = jnp.array([0.0, 1.0, 2.0, 3.0])
    >>> y = jnp.array([0.0, 1.0, 3.0, 6.0])
    >>> pchip_interp_1d(x, y, jnp.array([0.0, 1.5, 3.0]))
    Array([0.        , 1.86666667, 6.        ], dtype=float64)
    """
    return _pchip_eval_axis0(x, y, xq, extrapolate=extrapolate)


def interp_nd_pchip(
    grid: jnp.ndarray,
    axes: tuple[jnp.ndarray, ...],
    point: tuple,
    kinds: tuple[str, ...] | None = None,
) -> jnp.ndarray:
    """N-dimensional node-exact monotone-cubic (PCHIP) interpolation.

    Unlike the triweight *smoother* (:func:`interp_nd_triweight`), this is an
    *interpolant*: it passes exactly through the tabulated nodes while keeping
    C¹-continuous gradients. The per-axis tangents use the Fritsch & Carlson
    (1980) shape-preserving (monotone) rule, so the cubic never overshoots —
    safe even on sparse, nearest-neighbor-filled grids where a natural cubic
    spline would ring. Applied separably as a tensor product, one axis at a
    time.

    Use this for tabulated libraries whose feature position (e.g. an SED peak
    wavelength) shifts sharply across the grid, where the triweight kernel's
    neighbor-averaging would smear the feature. The cost relative to triweight
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
    kinds : tuple of {'pchip', 'linear'}, optional
        Interpolation kind per axis, in the same order as ``axes``. Defaults to
        PCHIP on every axis. Select ``'linear'`` for an axis whose tabulated
        function has C0 kinks at (some of) its knots: PCHIP's two-sided tangent
        estimate straddles such a corner and the neighboring cells inherit an
        O(1) slope error, so the local interpolation error decays only as O(h).
        A linear interpolant with knots on the kinks converges normally. Applies
        only where the kink locations are known and gridded — on a smooth axis,
        or a kinked axis whose knots miss the kinks, PCHIP is the better choice.

    Returns
    -------
    jnp.ndarray, shape (*trailing,)
        Interpolated values. At an exact node the tabulated value is returned
        to floating-point precision.

    Raises
    ------
    ValueError
        If ``kinds`` has the wrong length or names an unknown kind.

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
    if kinds is None:
        kinds = ("pchip",) * len(axes)
    elif len(kinds) != len(axes):
        raise ValueError(f"kinds has {len(kinds)} entries but there are {len(axes)} axes")
    unknown = set(kinds) - set(_EVAL_AXIS0)
    if unknown:
        raise ValueError(f"unknown interpolation kind(s) {sorted(unknown)}; expected pchip/linear")

    reduced = grid
    for ax, p, kind in zip(axes, point, kinds, strict=True):
        reduced = _EVAL_AXIS0[kind](ax, reduced, p)
    return reduced


def resample_template(
    wave_out: jnp.ndarray,
    wave_in: jnp.ndarray,
    flux_in: jnp.ndarray,
    *,
    left: float = 0.0,
    right: float = 0.0,
    log_flux: bool = True,
) -> jnp.ndarray:
    """Resample a tabulated template onto a wavelength grid, in log space.

    Interpolates linearly in :math:`\\log \\lambda` and (by default) in
    :math:`\\log F`, i.e. a power law between adjacent nodes:

    .. math::

        F(\\lambda) = F_i \\left(\\frac{\\lambda}{\\lambda_i}\\right)^{s_i},
        \\qquad
        s_i = \\frac{\\ln(F_{i+1}/F_i)}{\\ln(\\lambda_{i+1}/\\lambda_i)}

    where :math:`\\lambda` is wavelength [Angstrom], :math:`F` the tabulated
    quantity (any units; returned unchanged), and :math:`s_i` the local
    log-log slope [dimensionless].

    Template libraries are stored on coarse, log-spaced wavelength grids and
    evaluated on a much finer model grid — SKIRTOR v3 is 136 points
    (:math:`R \\sim 7`) against model grids oversampling it ~150x. SED tails
    (Rayleigh-Jeans, modified blackbody, synchrotron) are power laws, which
    this form reproduces exactly at any sampling density. Interpolating
    linearly in linear :math:`\\lambda` instead lays a straight chord across a
    convex curve, so it overestimates one-sidedly; that bias reaches 3.3 % in
    the 70 and 160 micron bands for SKIRTOR v3.

    Parameters
    ----------
    wave_out : array_like, shape (n_out,)
        Query wavelengths [Angstrom]. Need not be sorted.
    wave_in : array_like, shape (n_in,)
        Native template wavelengths [Angstrom], strictly ascending.
    flux_in : array_like, shape (n_in,)
        Tabulated values at ``wave_in`` (any units).
    left, right : float, optional
        Values returned below ``wave_in[0]`` and above ``wave_in[-1]``.
        Default 0.0 for both, matching an SED template that has no support
        outside its tabulated range. Pass 1.0 for a multiplicative factor.
    log_flux : bool, optional
        Interpolate in log flux (default True). Set False for signed
        quantities, where the log is undefined; log wavelength is still used.

    Returns
    -------
    ndarray, shape (n_out,)
        Resampled values, in the units of ``flux_in``.

    Notes
    -----
    **JIT-compatible**: yes — pure ``jnp`` ops with gathers on traced indices.

    **Gradient-safe**: yes. Intervals with a non-positive endpoint fall back to
    linear-in-flux, gated with the double-``where`` pattern so the discarded
    ``log`` branch never forms ``0 * inf`` in the VJP.

    Exact at the native nodes to floating-point precision, and never overshoots
    the two bracketing node values.
    """
    n = wave_in.shape[0]
    lx = jnp.log(wave_in)
    lxq = jnp.log(wave_out)

    i = jnp.clip(jnp.searchsorted(lx, lxq) - 1, 0, n - 2)
    x0, x1 = lx[i], lx[i + 1]
    y0, y1 = flux_in[i], flux_in[i + 1]

    h = x1 - x0
    t = jnp.clip((lxq - x0) / jnp.where(h > 0.0, h, 1.0), 0.0, 1.0)

    linear = y0 + t * (y1 - y0)
    if log_flux:
        # Gate the log INPUTS on the same condition as the output (double-where):
        # feeding a non-positive endpoint to log() yields -inf, and the outer
        # where() would then form 0 * inf = NaN in the reverse pass.
        both_pos = (y0 > 0.0) & (y1 > 0.0)
        y0_safe = jnp.where(both_pos, y0, 1.0)
        y1_safe = jnp.where(both_pos, y1, 1.0)
        ly0 = jnp.log(y0_safe)
        out = jnp.where(both_pos, jnp.exp(ly0 + t * (jnp.log(y1_safe) - ly0)), linear)
    else:
        out = linear

    out = jnp.where(wave_out < wave_in[0], left, out)
    return jnp.where(wave_out > wave_in[-1], right, out)


def loglog_integral(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Integrate ``y`` over ``x`` treating each segment as a power law.

    The integral of the interpolant :func:`resample_template` builds, so a
    template normalized with this function still integrates to the same value
    after resampling. Over one segment, with
    :math:`s = \\ln(y_1/y_0) / \\ln(x_1/x_0)`:

    .. math::

        \\int_{x_0}^{x_1} y_0 (x/x_0)^{s}\\, dx
        = x_0 y_0 \\, a \\, \\frac{e^{u} - 1}{u},
        \\qquad a = \\ln\\frac{x_1}{x_0}, \\quad u = a + \\ln\\frac{y_1}{y_0}

    Written with ``expm1(u)/u`` rather than the textbook
    :math:`(x_1 y_1 - x_0 y_0)/(s+1)` because that form is 0/0 at
    :math:`s = -1` — a real case, since :math:`\\nu F_\\nu` flat is exactly
    :math:`s = -1`.

    Parameters
    ----------
    x : array_like, shape (n,)
        Monotonic, strictly positive abscissa (wavelength [Angstrom] or
        frequency [Hz]). Descending ``x`` returns a negative integral, matching
        ``jnp.trapezoid``, so existing ``-trapezoid(lnu, nu)`` call sites keep
        their sign convention.
    y : array_like, shape (n,)
        Values at ``x`` (any units).

    Returns
    -------
    ndarray, shape ()
        The integral, in units of ``x`` times ``y``.

    Notes
    -----
    **JIT-compatible**: yes.

    **Gradient-safe**: yes — segments with a non-positive endpoint fall back to
    the trapezoid rule under a double-``where``, so the discarded ``log``
    branch cannot form ``0 * inf`` in the VJP.
    """
    x0, x1 = x[:-1], x[1:]
    y0, y1 = y[:-1], y[1:]

    positive = (y0 > 0.0) & (y1 > 0.0)
    y0_safe = jnp.where(positive, y0, 1.0)
    y1_safe = jnp.where(positive, y1, 1.0)

    a = jnp.log(x1 / x0)
    u = a + jnp.log(y1_safe) - jnp.log(y0_safe)
    # expm1(u)/u, continued to its finite limit 1 + u/2 as u -> 0.
    small = jnp.abs(u) < 1e-7
    u_safe = jnp.where(small, 1.0, u)
    ratio = jnp.where(small, 1.0 + 0.5 * u, jnp.expm1(u_safe) / u_safe)

    power_law = x0 * y0_safe * a * ratio
    trapezoid = 0.5 * (y0 + y1) * (x1 - x0)
    return jnp.sum(jnp.where(positive, power_law, trapezoid))


def slice_fixed_axes(
    preint: PreintegratedGrid | PreintegratedLines,
    fixed: dict[int, float],
    index_space_interp: bool | None = None,
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
    index_space_interp : bool or None
        Whether to use index-space interpolation for non-uniform axes.
        Passed through to :func:`compute_grid_weights`. Default None.

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
        sub_phot = preint.subband_phot
        sub_waves = preint.subband_waves
        sub_waves_rest = preint.subband_waves_rest
        # A quadrature node is a RATIO, λ_k = ∫λSw / ∫Sw, and the weighted mean
        # of ratios is not the ratio of weighted means. Interpolate the node's
        # numerator (λ_k·Φ_k) and denominator (Φ_k) separately and divide at the
        # end — that reproduces the centroid of the interpolated template exactly,
        # where averaging the nodes directly would bias them.
        sub_num = None if sub_waves is None else sub_waves * sub_phot
        sub_num_rest = None if sub_waves_rest is None else sub_waves_rest * sub_phot

        # Process in reverse order so axis indices remain valid after each slice
        for axis_idx in sorted(fixed.keys(), reverse=True):
            value = fixed[axis_idx]
            ax = axes[axis_idx]
            ed = edges[axis_idx]

            # Compute triweight weights at the fixed value
            w = compute_grid_weights(
                value, ax, scatter=0.5 * float(ax[1] - ax[0]), edges=ed,
                index_space_interp=index_space_interp
            )

            # Contract phot along this axis using einsum-style contraction.
            # tensordot(w, phot, ([0], [axis_idx])) removes axis_idx from phot,
            # preserving the order of all other axes.
            phot = jnp.tensordot(w, phot, axes=([0], [axis_idx]))

            if moment is not None:
                moment = jnp.tensordot(w, moment, axes=([0], [axis_idx]))

            if sub_phot is not None:
                sub_phot = jnp.tensordot(w, sub_phot, axes=([0], [axis_idx]))
                sub_num = jnp.tensordot(w, sub_num, axes=([0], [axis_idx]))
                sub_num_rest = jnp.tensordot(w, sub_num_rest, axes=([0], [axis_idx]))

            # Remove from axes and edges lists
            axes.pop(axis_idx)
            edges.pop(axis_idx)

        if sub_phot is not None:
            # Where a template has no flux in a sub-band the node is multiplied by
            # zero, so its value cannot change the result — but it is still fed
            # through the dust law, which goes as 1/λ. A zero node there yields
            # inf/NaN that survives into the GRADIENT even though the forward value
            # is finite. Fall back to the filter's effective wavelength: finite,
            # positive, and physically sane.
            live = sub_phot != 0.0
            safe = jnp.where(live, sub_phot, 1.0)
            sub_waves = jnp.where(live, sub_num / safe, preint.effective_wavelengths[:, None])
            sub_waves_rest = jnp.where(
                live, sub_num_rest / safe, preint.effective_wavelengths_rest[:, None]
            )

        return PreintegratedGrid(
            phot=phot,
            moment=moment,
            axes=tuple(axes),
            edges=tuple(edges),
            effective_wavelengths=preint.effective_wavelengths,
            effective_wavelengths_rest=preint.effective_wavelengths_rest,
            log10_flux_scale=preint.log10_flux_scale,
            n_filters=preint.n_filters,
            subband_phot=sub_phot,
            subband_waves=sub_waves,
            subband_waves_rest=sub_waves_rest,
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
            w = compute_grid_weights(
                value, ax, scatter=0.5 * float(ax[1] - ax[0]), edges=ed,
                index_space_interp=index_space_interp
            )

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
