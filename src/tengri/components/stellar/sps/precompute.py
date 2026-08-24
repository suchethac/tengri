# SPDX-License-Identifier: BSD-3-Clause
"""Pre-computation of SSP observables at fixed or tabulated redshift.

The key insight: at inference time, redshift is either known (fixed) or
varies slowly. Everything that depends on z but NOT on model parameters
(SFH weights, dust, Z) can be pre-computed, eliminating the expensive
wavelength dimension from the gradient tape.

Four levels of pre-computation:

1. Photometric (fixed z): Pre-integrate SSP through filters.
   Speedup: 30-50x (Zacharegkas+2025 Section 3).

2. Spectroscopic (fixed z): Pre-rebin SSP to observed pixel wavelengths.
   Reduces wavelength dimension from ~7000 to ~1000-3000.

3. Photometric z-table (free z): Pre-integrate SSP at each z in a grid.
   At inference: interpolate to current z. Same speedup, z is free.
   Follows DSPS precompute_ssp_obsmags_on_z_table approach.

4. Combined: Both simultaneously.

Usage:
    # Fixed redshift (once per galaxy)
    precomp = precompute_photometry(ssp_data, filters, redshift)

    # Free redshift (once per survey / filter set)
    ztable = precompute_photometry_ztable(ssp_data, filters)
    # At each MCMC step: interpolate to current z
    flux = fast_photometry_ztable(weights, ztable, z, dust_params)
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from tengri.utils.filter_convention import FilterConvention, filter_weight_np as _filter_weight_np
from tengri.utils.grid_interp import preintegrate_grid, subband_quadrature
from tengri.utils.physics_constants import TEN_PC_CM
from tengri.utils.scale import (
    apply_log10_scale,
    log10_flux_scale as _log10_flux_scale,
    log10_weighted_sum as _log10_weighted_sum,
)

# SSP precompute grid axes: (lgmet, lg_age_gyr). The age axis is never user-fixed
# (ages are determined by the SFH). Only metallicity may be Fixed.
AXIS_PARAMS: tuple[str, ...] = ("met_logzsol",)

# numpy >= 2.0 uses trapezoid; older versions used trapz
_np_trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def _vectorized_interp(
    xp_target: np.ndarray, xp_source: np.ndarray, yp_source: np.ndarray
) -> np.ndarray:
    """Vectorized linear interpolation: all (met, age) SSPs at once.

    Replaces the inner ``for m in n_met: for a in n_age: np.interp(...)``
    double loop with a single vectorized NumPy operation.  Computes
    interpolation indices and weights once, then applies via fancy indexing.

    Parameters
    ----------
    xp_target : array, shape (n_target,)
        Target x-coordinates (e.g. filter wavelengths).
    xp_source : array, shape (n_source,)
        Source x-coordinates (must be sorted ascending).
    yp_source : array, shape (n_met, n_age, n_source)
        Source y-values for all metallicities and ages.

    Returns
    -------
    array, shape (n_met, n_age, n_target)
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

    # Vectorized gather: (n_met, n_age, n_target) via fancy indexing
    result = (1.0 - frac) * yp_source[:, :, idx] + frac * yp_source[:, :, idx + 1]

    # Zero out-of-bounds (left and right)
    oob = (xp_target < xp_source[0]) | (xp_target > xp_source[-1])
    result[:, :, oob] = 0.0

    return result


class PhotometricPrecomputation(NamedTuple):
    """Pre-computed SSP broadband fluxes for fast photometry.

    Attributes
    ----------
    ssp_phot : array, shape (n_met, n_age, n_filters)
        SSP broadband flux per metallicity, age, and filter [erg/s/Hz/Msun].
        Φ_{ijb} = ∫ SSP(Z_i, t_j, λ) T_b(λ) λ dλ / ∫ T_b(λ) λ dλ
    ssp_phot_moment : array or None, shape (n_met, n_age, n_filters)
        First spectral moment of the SSP within each filter [Angstrom].
        Ψ_{ijb} = ∫ SSP(Z_i, t_j, λ) (λ - λ_eff) T_b(λ) λ dλ / ∫ T_b(λ) λ dλ
        Used for the Taylor-corrected dust approximation:
        f_b ≈ A(λ_eff)·Φ + A'(λ_eff)·Ψ, which captures the SSP–dust
        covariance to first order and reduces the factorization error by ~5×.
        None when Taylor correction is disabled.
    effective_wavelengths : array, shape (n_filters,)
        Effective wavelength of each filter [Angstrom], observed frame.
        Used for evaluating dust at a single wavelength per band.
    effective_wavelengths_rest : array, shape (n_filters,)
        Effective wavelength in rest frame [Angstrom].
    log10_flux_scale : float
        ``log10`` of the geometric factor (1+z) / (4π dL²) [dex re cm⁻²]. Stored
        as a log: the linear factor is ``0.0`` in float32 at every distance, so a
        stored linear scale is zeroed by the cast alone (#1859).
    redshift : float
        Source redshift [dimensionless].
    n_filters : int
        Number of filters [dimensionless].
    ssp_subband_phot : array or None, shape (n_met, n_age, n_filters, n_subbands)
        Filter integral restricted to each sub-band. Sums over the last axis
        to ``ssp_phot``. None unless ``n_subbands > 0``. [erg/s/Hz]
    ssp_subband_waves_rest : array or None, shape (n_met, n_age, n_filters, n_subbands)
        Rest-frame quadrature node of each sub-band: the template's own
        flux-weighted centroid there. None unless ``n_subbands > 0``. [Angstrom]
    ssp_subband_phot_igm : array or None, shape (n_met, n_age, n_filters, n_subbands)
        ``ssp_subband_phot`` with the IGM transmission at each node folded in
        (#1135): Φ_{majk} · T_IGM(λ*_{majk} · (1+z), z) [erg/s/Hz], where λ* is
        the sub-band's quadrature node. Injected by ``SEDModel.build`` when a
        mean-IGM model is present and precomputable; None otherwise. Folded here
        on the metallicity axis, before the SSP contraction, because the
        runtime node is a met-weighted average whose weights move with the free
        ``met_logzsol``.
        The REST-frame band lives in :class:`RestBandPrecomputation`, built once by
        :func:`precompute_restband_photometry` and carried on the stellar component's
        state: one builder for the fixed-z and free-z paths alike (#1148).

    Notes
    -----
    **JIT-compatible**: no; this is a data container produced by
    :func:`precompute_photometry` at startup; data itself is immutable
    and suitable for use in JAX operations.

    """

    ssp_phot: jnp.ndarray
    ssp_phot_moment: "jnp.ndarray | None"
    effective_wavelengths: jnp.ndarray
    effective_wavelengths_rest: jnp.ndarray
    log10_flux_scale: float
    redshift: float
    n_filters: int
    ssp_subband_phot: "jnp.ndarray | None" = None
    ssp_subband_waves_rest: "jnp.ndarray | None" = None
    ssp_subband_phot_igm: "jnp.ndarray | None" = None


class SpectroscopicPrecomputation(NamedTuple):
    """Pre-rebinned SSP templates for fast spectroscopy.

    Attributes
    ----------
    ssp_on_pixels : array, shape (n_met, n_age, n_pix)
        SSP flux interpolated to observed spectral pixel wavelengths
        in rest frame [erg/s/Hz/Msun].
    wave_rest_pixels : array, shape (n_pix,)
        Rest-frame wavelengths of spectral pixels [Angstrom].
    wave_obs_pixels : array, shape (n_pix,)
        Observed-frame wavelengths [Angstrom].
    log10_flux_scale : float
        ``log10`` of the geometric scaling factor (1+z) / (4π dL²)
        [dex re cm⁻²]. Stored as a log for the reason given in
        :class:`PhotometricPrecomputation` (#1859).
    redshift : float
        Source redshift [dimensionless].

    Notes
    -----
    **JIT-compatible**: no; this is a data container produced by
    :func:`precompute_spectroscopy` at startup; data itself is immutable
    and suitable for use in JAX operations.

    """

    ssp_on_pixels: jnp.ndarray
    wave_rest_pixels: jnp.ndarray
    wave_obs_pixels: jnp.ndarray
    log10_flux_scale: float
    redshift: float


def precompute_photometry(
    ssp_data,
    filter_waves,
    filter_trans,
    redshift,
    dl_cm,
    taylor_correction: bool = True,
    n_subbands: int = 0,
    fixed: dict[int, float] | None = None,
) -> PhotometricPrecomputation:
    """Pre-compute SSP broadband fluxes for all filters.

    Eliminates the wavelength integral from the inference loop.  After
    precomputation, galaxy photometry is a weighted sum over the SSP grid.

    Following Zacharegkas+2025 §3, we precompute the exact filter-integrated
    SSP tensor Φ_{ijb} and evaluate dust at a single effective wavelength
    per band.  When ``taylor_correction=True`` (default), we also precompute
    the first spectral moment Ψ_{ijb} = <SSP·(λ-λ_eff)>_b which enables
    a first-order Taylor correction that captures the SSP–dust covariance
    and reduces the factorization error by ~5× (from ~1.3% to ~0.26% for
    SDSS g-band worst case with Charlot–Fall dust).

    Parameters
    ----------
    ssp_data : SSPData
        SSP templates with ssp_wave [Angstrom], ssp_flux [Lsun/Hz/Msun],
        ssp_lg_age_gyr, ssp_lgmet.
    filter_waves : list of array
        Wavelength grid per filter (observed frame [Angstrom]).
    filter_trans : list of array
        Transmission curve per filter (dimensionless, in [0, 1]).
    redshift : float
        Source redshift (dimensionless).
    dl_cm : float
        Luminosity distance [cm].
    taylor_correction : bool
        Precompute the spectral moment tensor Ψ for first-order Taylor
        dust correction (default True).  Adds one tensor of the same shape
        as Φ; inference cost is negligible (one extra dust derivative per
        filter, computed via finite differences).

        Superseded by ``n_subbands``: the Taylor form *extrapolates* the
        attenuation linearly away from λ_eff, which diverges where the
        curve steepens (GALEX FUV +45 % at z=0.05, +215 % at z=1). See #1122.
    n_subbands : int
        Number of sub-bands K for the multiplicative dust quadrature
        (default 0 = off). Builds Φ_k and the per-template quadrature nodes,
        so the screen is *evaluated* at K points per band rather than
        extrapolated from one. Converges as 1/K²; K=3 is the working point
        and is cheaper at runtime than the Taylor moment it replaces.
    fixed : dict[int, float], optional
        Mapping of axis index → fixed value. Axes are numbered from 0:

        - 0: lgmet (metallicity in log10(Z/Zsun))

        If provided, these axes are collapsed at init time via triweight
        interpolation. Default None.

    Returns
    -------
    PhotometricPrecomputation
        Pre-computed SSP photometry data for the fused photometry kernel.

    Notes
    -----
    **JIT-compatible**: no; this is a data precomputation function that
    runs once at startup, not inside the inference loop. Uses numpy and HDF5 I/O.
    **Gradient-safe**: not applicable (CPU preprocessing).

    """
    from tengri.utils.grid_interp import slice_fixed_axes

    # Delegate to preintegrate_grid, which handles all wavelength integration
    # for arbitrary grid dimensionality (2D for normal SSP, 4D for alpha-enhanced).
    # Grid axes: (n_met, n_age, n_wave) → axes are (lgmet, lg_age_gyr)
    preint = preintegrate_grid(
        templates=np.asarray(ssp_data.ssp_flux),
        wave_rest=np.asarray(ssp_data.ssp_wave),
        filter_waves=[np.asarray(fw) for fw in filter_waves],
        filter_trans=[np.asarray(ft) for ft in filter_trans],
        redshift=redshift,
        dl_cm=dl_cm,
        axes=(
            np.asarray(ssp_data.ssp_lgmet),
            np.asarray(ssp_data.ssp_lg_age_gyr),
        ),
        taylor=taylor_correction,
        n_subbands=n_subbands,
    )

    # Collapse fixed axes if provided
    if fixed:
        preint = slice_fixed_axes(preint, fixed)

    return PhotometricPrecomputation(
        ssp_phot=preint.phot,
        ssp_phot_moment=preint.moment,
        effective_wavelengths=preint.effective_wavelengths,
        effective_wavelengths_rest=preint.effective_wavelengths_rest,
        log10_flux_scale=preint.log10_flux_scale,
        redshift=float(redshift),
        n_filters=preint.n_filters,
        ssp_subband_phot=preint.subband_phot,
        ssp_subband_waves_rest=preint.subband_waves_rest,
    )


class RestBandPrecomputation(NamedTuple):
    r"""SSP templates preintegrated through each filter placed in the REST frame.

    Attributes
    ----------
    ssp_restband_phot : array, shape (n_met, n_age, n_filters)
        Filter integral at z=0 [erg/s/Hz/Msun].
    ssp_restband_subband_phot : array or None, shape (n_met, n_age, n_filters, n_subbands)
        Sub-band quadrature weights for the rest band. None unless ``n_subbands > 0``.
    ssp_restband_subband_waves : array or None, shape (n_met, n_age, n_filters, n_subbands)
        Quadrature nodes of the rest band [Angstrom].
    restband_eff_waves : array, shape (n_filters,)
        The wavelength the rest band samples: the filter's own pivot [Angstrom].

    Notes
    -----
    **JIT-compatible**: no, build-time data container.
    """

    ssp_restband_phot: jnp.ndarray
    ssp_restband_subband_phot: "jnp.ndarray | None"
    ssp_restband_subband_waves: "jnp.ndarray | None"
    restband_eff_waves: jnp.ndarray


def precompute_restband_photometry(
    ssp_data,
    filter_waves,
    filter_trans,
    n_subbands: int = 0,
    fixed: dict[int, float] | None = None,
) -> RestBandPrecomputation:
    r"""Preintegrate the SSP grid through each filter placed in the **rest** frame.

    ``phot_rest_fnu`` (and so ``Observables.mag_absolute``) is the SED reprojected
    at :math:`z=0`, :math:`d_L=10\,{\rm pc}`: *the galaxy as it is*. The filter
    therefore sits in the rest frame and samples the rest SED at its **own** pivot
    wavelength:

    .. math::

        \Phi^{\rm rest}_{ijb} =
        \frac{\int S_{ij}(\lambda)\, R_b(\lambda)\, w(\lambda)\,{\rm d}\lambda}
             {\int R_b(\lambda)\, w(\lambda)\,{\rm d}\lambda}

    with :math:`S_{ij}` the SSP [erg/s/Hz/Msun], :math:`R_b` the filter response and
    :math:`w` the convention weight (ADR-0017).

    This is a **different quantity** from :func:`precompute_photometry`'s
    ``ssp_phot``, which places the filter in the *observed* frame and so samples
    rest :math:`\lambda_{\rm eff}/(1+z)`. Conflating them is #1148: the LUT path
    reused the observed-band integral for ``phot_rest_fnu`` and disagreed with the
    exact path by 769 % in ``des_g`` at z=0.5, growing to orders of magnitude in the
    blue: an *absolute* magnitude that depended on the source's redshift.

    **Redshift does not appear.** At z=0 the filter always samples the same rest
    wavelengths, so this is one build-time constant that serves fixed-z *and* free-z
    models: no z-table, no interpolation, no runtime cost.

    Parameters
    ----------
    ssp_data : SSPData
        SSP templates.
    filter_waves : list of array
        Wavelength grid per filter [Angstrom].
    filter_trans : list of array
        Transmission curve per filter [dimensionless].
    n_subbands : int
        Sub-band quadrature order K for the dust screen across the rest band
        (#1122). 0 disables it, leaving the screen evaluated at the pivot.
    fixed : dict[int, float], optional
        Axis index → fixed value, collapsed at build time (axis 0 = lgmet).

    Returns
    -------
    RestBandPrecomputation

    Notes
    -----
    **JIT-compatible**: no, runs once at build time.
    """
    from tengri.utils.grid_interp import slice_fixed_axes

    preint = preintegrate_grid(
        templates=np.asarray(ssp_data.ssp_flux),
        wave_rest=np.asarray(ssp_data.ssp_wave),
        filter_waves=[np.asarray(fw) for fw in filter_waves],
        filter_trans=[np.asarray(ft) for ft in filter_trans],
        redshift=0.0,
        dl_cm=TEN_PC_CM,
        axes=(
            np.asarray(ssp_data.ssp_lgmet),
            np.asarray(ssp_data.ssp_lg_age_gyr),
        ),
        taylor=False,  # the rest band uses the quadrature, never the Taylor form
        n_subbands=n_subbands,
    )
    if fixed:
        preint = slice_fixed_axes(preint, fixed)

    return RestBandPrecomputation(
        ssp_restband_phot=preint.phot,
        ssp_restband_subband_phot=preint.subband_phot,
        ssp_restband_subband_waves=preint.subband_waves_rest,
        # At z=0 the rest and observed effective wavelengths coincide; this is the
        # filter's own pivot, and it is what the rest band samples.
        restband_eff_waves=preint.effective_wavelengths_rest,
    )


def precompute_spectroscopy(
    ssp_data, wave_obs_pixels, redshift, dl_cm
) -> SpectroscopicPrecomputation:
    """Pre-rebin SSP templates to observed spectral pixel wavelengths.

    Reduces the wavelength dimension from ~7000 (native SSP) to
    ~1000-3000 (observed pixels), cutting memory and compute.

    Parameters
    ----------
    ssp_data : SSPData
        SSP templates with ssp_flux [Lsun/Hz/Msun] and ssp_wave [Angstrom].
    wave_obs_pixels : array, shape (n_pix,)
        Observed-frame wavelengths of spectral pixels [Angstrom].
    redshift : float
        Source redshift (dimensionless).
    dl_cm : float
        Luminosity distance [cm].

    Returns
    -------
    SpectroscopicPrecomputation
        Pre-rebinned SSP data for :func:`fast_spectrum`.

    Notes
    -----
    **JIT-compatible**: no, data precomputation function with numpy I/O.
    **Gradient-safe**: not applicable (CPU preprocessing).

    """
    wave_rest_pixels = wave_obs_pixels / (1.0 + redshift)

    # Interpolate all SSP spectra to the pixel rest-frame wavelengths
    # Vectorized: compute weights once, apply to all (met, age) simultaneously
    import numpy as np

    ssp_flux_np = np.asarray(ssp_data.ssp_flux)
    wave_rest_np = np.asarray(wave_rest_pixels)
    wave_ssp_np = np.asarray(ssp_data.ssp_wave)
    ssp_on_pixels_np = _vectorized_interp(wave_rest_np, wave_ssp_np, ssp_flux_np)
    ssp_on_pixels = jnp.array(ssp_on_pixels_np)

    # The (1+z)/(4π d_L²) geometric scale as a log10 offset, rather than calling
    # the ``@jit``'d ``lnu_to_fnu`` and ``float()``-casting; the latter raises
    # ConcretizationTypeError when this precompute runs inside a jit trace
    # (e.g. when the user calls ``predict_*`` for the first time inside a
    # ``jax.jit`` wrapper without warming the chain). See companion fix in
    # ``utils/grid_interp.py``. Log, not linear: the factor is ~1e-57 and stores
    # as exactly 0.0 in float32 (#1859).
    log10_flux_scale = _log10_flux_scale(redshift, dl_cm)
    # ``redshift`` may itself be a tracer when this path runs inside jit;
    # keep the conversion defensive in that case too.
    try:
        log10_flux_scale_out = float(log10_flux_scale)
        redshift_out = float(redshift)
    except (TypeError, jax.errors.ConcretizationTypeError):
        log10_flux_scale_out = log10_flux_scale
        redshift_out = redshift

    return SpectroscopicPrecomputation(
        ssp_on_pixels=ssp_on_pixels,
        wave_rest_pixels=wave_rest_pixels,
        wave_obs_pixels=wave_obs_pixels,
        log10_flux_scale=log10_flux_scale_out,
        redshift=redshift_out,
    )


class SpectroscopicZTable(NamedTuple):
    """Free-z spectroscopy marker carrying the observed pixel grid.

    Unlike the photometric ztable, the SSP flux is **not** pre-tabulated
    across redshift. A spectrum resolves absorption features that sweep
    across a fixed observed pixel as ``z`` changes, so interpolating
    ``ssp_on_pixels`` between z-grid nodes would be inaccurate near any
    spectral feature (filter integration hides this in the photometric
    case). Instead, the stellar component re-interpolates the SSP cube to
    ``wave_obs / (1 + z)`` at runtime: exact and fully differentiable in z.

    Attributes
    ----------
    wave_obs_pixels : array, shape (n_pix,)
        Observed-frame pixel wavelengths [Angstrom] (z-independent).
    z_min, z_max : float
        Redshift bounds the table was built for (advisory; runtime z is
        not clamped to them).

    Notes
    -----
    **JIT-compatible**: no, startup marker; leaves are immutable and safe
    inside JAX operations.

    """

    wave_obs_pixels: jnp.ndarray
    z_min: float
    z_max: float


def precompute_spectroscopy_ztable(
    ssp_data,
    wave_obs_pixels,
    z_grid=None,
    z_min=0.001,
    z_max=3.0,
    n_z=100,
) -> SpectroscopicZTable:
    """Build the free-z spectroscopy marker (observed pixel grid only).

    The free-z spectrum path interpolates the SSP cube to the rest-frame
    pixel grid ``wave_obs / (1 + z)`` at runtime rather than pre-tabulating
    across redshift (see :class:`SpectroscopicZTable` for why). This builder
    therefore just records the observed pixel grid and the z bounds.

    Parameters
    ----------
    ssp_data : SSPData
        SSP templates (unused here; kept for signature symmetry with the
        photometric ztable builder).
    wave_obs_pixels : array, shape (n_pix,)
        Observed-frame pixel wavelengths [Angstrom].
    z_grid : array, optional
        Custom redshift grid; only its min/max are recorded.
    z_min, z_max : float
        Redshift bounds [dimensionless].
    n_z : int
        Unused (kept for signature symmetry).

    Returns
    -------
    SpectroscopicZTable
        Free-z marker carrying the observed pixel grid.

    Notes
    -----
    **JIT-compatible**: no, startup marker.
    **Gradient-safe**: not applicable (CPU preprocessing).

    """
    del ssp_data, n_z
    if z_grid is not None:
        z_grid = jnp.asarray(z_grid)
        z_min, z_max = float(z_grid[0]), float(z_grid[-1])
    return SpectroscopicZTable(
        wave_obs_pixels=jnp.asarray(wave_obs_pixels),
        z_min=float(z_min),
        z_max=float(z_max),
    )


class PhotometricZTable(NamedTuple):
    """Pre-computed SSP broadband fluxes on a redshift grid.

    For free-redshift inference: interpolate this table to the current z
    instead of computing the full wavelength integral each step.

    Attributes
    ----------
    ssp_phot_table : array, shape (n_z, n_met, n_age, n_filters)
        SSP broadband flux at each redshift, metallicity, age, and filter
        [erg/s/Hz/Msun].
    eff_waves_rest_table : array, shape (n_z, n_filters)
        Rest-frame effective wavelengths at each redshift [Angstrom].
    log10_flux_scale_table : array, shape (n_z,)
        Geometric factor (1+z)/(4π dL²) at each redshift [dimensionless].
    z_grid : array, shape (n_z,)
        Redshift grid [dimensionless].
    n_filters : int
        Number of filters [dimensionless].
    igm_trans_table : array, shape (n_z, n_filters)
        IGM transmission at effective observed wavelengths per redshift
        [dimensionless]. All ones when IGM is not applied.

    Notes
    -----
    **JIT-compatible**: no; this is a data container produced by
    :func:`precompute_photometry_ztable` at startup; data itself is immutable
    and suitable for use in JAX operations.

    """

    ssp_phot_table: jnp.ndarray
    eff_waves_rest_table: jnp.ndarray
    log10_flux_scale_table: jnp.ndarray
    z_grid: jnp.ndarray
    n_filters: int
    igm_trans_table: jnp.ndarray
    # Taylor moment Ψ on the z grid, shape
    # ``(n_z, n_met, n_age, n_filters)``. ``None`` when
    # ``taylor_correction=False`` was passed to
    # :func:`precompute_photometry_ztable`. Used by the free-z dust LUT
    # path to apply ``A·Φ + A'·Ψ`` at the source's redshift.
    ssp_phot_moment_table: jnp.ndarray | None = None
    #: (n_z, n_met, n_age, n_filters, n_subbands) sub-band filter integrals (#1122).
    ssp_subband_phot_table: jnp.ndarray | None = None
    #: (n_z, n_met, n_age, n_filters, n_subbands) rest-frame quadrature nodes [A].
    subband_waves_rest_table: jnp.ndarray | None = None
    #: (n_z, n_met, n_age, n_filters, n_subbands) sub-band filter integrals with the
    #: IGM transmission at each node folded in (#1135). Injected by ``SEDModel.build``
    #: from the *cached* node table, so it is not part of the on-disk z-table (and the
    #: cache key needs no IGM term). ``None`` when the IGM is absent or reads free
    #: parameters (patchy reionization, DLAs).
    ssp_subband_phot_igm_table: jnp.ndarray | None = None


# Bump when the quadrature or table layout changes: invalidates every
# cached z-table built by an older algorithm.
_ZTABLE_CACHE_VERSION = 1


def _ztable_cache_dir():
    """Resolve the on-disk z-table cache directory, or None when disabled.

    ``TENGRI_DISABLE_PRECOMP_CACHE=1`` opts out;
    ``TENGRI_PRECOMP_CACHE_DIR`` overrides the location
    (default ``~/.cache/tengri_precomp``).
    """
    import os
    from pathlib import Path

    if os.environ.get("TENGRI_DISABLE_PRECOMP_CACHE"):
        return None
    custom = os.environ.get("TENGRI_PRECOMP_CACHE_DIR")
    return Path(custom) if custom else Path.home() / ".cache" / "tengri_precomp"


def _ztable_cache_key(
    ssp_data,
    filter_waves,
    filter_trans,
    z_grid,
    apply_igm,
    taylor_correction,
    convention,
    n_subbands=0,
) -> str:
    """Content hash over everything the table depends on."""
    import hashlib

    h = hashlib.sha256()
    h.update(f"v{_ZTABLE_CACHE_VERSION}".encode())
    for arr in (ssp_data.ssp_wave, ssp_data.ssp_flux):
        a = np.ascontiguousarray(np.asarray(arr))
        h.update(repr((a.shape, a.dtype.str)).encode())
        h.update(a)
    for fw, ft in zip(filter_waves, filter_trans):
        for arr in (fw, ft):
            a = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
            h.update(a)
    h.update(np.ascontiguousarray(np.asarray(z_grid, dtype=np.float64)))
    # n_subbands changes the table's CONTENT, so it must change the hash. Without
    # it a cached K=0 table is reused for a K=5 model and the quadrature silently
    # no-ops -- persistently, across processes (#1122).
    # The PAYLOAD SCHEMA is part of the key, not just the payload's inputs. #1859
    # renamed ``flux_scale_table`` -> ``log10_flux_scale_table`` and changed what
    # the numbers mean; without a version bump a warm cache would either KeyError
    # or, worse, hand a linear table to a consumer expecting logs. Bump this
    # whenever the npz field set or the meaning of a field changes.
    h.update(
        repr(
            (
                bool(apply_igm),
                bool(taylor_correction),
                str(convention),
                int(n_subbands),
                "schema=2",
            )
        ).encode()
    )
    return h.hexdigest()


def precompute_photometry_ztable(
    ssp_data,
    filter_waves,
    filter_trans,
    z_grid=None,
    z_min=0.001,
    z_max=3.0,
    n_z=100,
    apply_igm=False,
    taylor_correction: bool = False,
    n_subbands: int = 0,
    convention: FilterConvention = FilterConvention.BESSELL,
) -> PhotometricZTable:
    """Pre-compute SSP broadband fluxes on a redshift grid, disk-cached.

    Thin caching wrapper around :func:`_compute_photometry_ztable` (same
    signature: see it for parameter semantics). The table depends only on
    the SSP grid, the filter set, the z grid, and the quadrature flags, so
    it is content-hashed and persisted under ``~/.cache/tengri_precomp``:
    the first build of a given (SSP, filters, z-grid) combination pays the
    quadrature; every later build: any process, any model: loads the npz
    in well under a second. ``TENGRI_DISABLE_PRECOMP_CACHE=1`` opts out,
    ``TENGRI_PRECOMP_CACHE_DIR`` relocates the cache.

    Notes
    -----
    **JIT-compatible**: no, data precomputation with file I/O.
    """
    if z_grid is None:
        z_grid = jnp.linspace(z_min, z_max, n_z)
    else:
        z_grid = jnp.asarray(z_grid)

    cache_dir = _ztable_cache_dir()
    cache_path = None
    if cache_dir is not None:
        key = _ztable_cache_key(
            ssp_data,
            filter_waves,
            filter_trans,
            z_grid,
            apply_igm,
            taylor_correction,
            convention,
            n_subbands,
        )
        cache_path = cache_dir / f"ztable_{key}.npz"
        if cache_path.is_file():
            with np.load(cache_path, allow_pickle=False) as d:
                return PhotometricZTable(
                    ssp_phot_table=jnp.array(d["ssp_phot_table"]),
                    eff_waves_rest_table=jnp.array(d["eff_waves_rest_table"]),
                    log10_flux_scale_table=jnp.array(d["log10_flux_scale_table"]),
                    z_grid=jnp.array(d["z_grid"]),
                    n_filters=int(d["n_filters"]),
                    igm_trans_table=jnp.array(d["igm_trans_table"]),
                    ssp_phot_moment_table=(
                        jnp.array(d["ssp_phot_moment_table"])
                        if "ssp_phot_moment_table" in d.files
                        else None
                    ),
                    ssp_subband_phot_table=(
                        jnp.array(d["ssp_subband_phot_table"])
                        if "ssp_subband_phot_table" in d.files
                        else None
                    ),
                    subband_waves_rest_table=(
                        jnp.array(d["subband_waves_rest_table"])
                        if "subband_waves_rest_table" in d.files
                        else None
                    ),
                )

    table = _compute_photometry_ztable(
        ssp_data,
        filter_waves,
        filter_trans,
        z_grid=z_grid,
        apply_igm=apply_igm,
        taylor_correction=taylor_correction,
        n_subbands=n_subbands,
        convention=convention,
    )

    if cache_path is not None:
        import os
        import tempfile

        cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "ssp_phot_table": np.asarray(table.ssp_phot_table),
            "eff_waves_rest_table": np.asarray(table.eff_waves_rest_table),
            "log10_flux_scale_table": np.asarray(table.log10_flux_scale_table),
            "z_grid": np.asarray(table.z_grid),
            "n_filters": np.asarray(table.n_filters),
            "igm_trans_table": np.asarray(table.igm_trans_table),
        }
        if table.ssp_phot_moment_table is not None:
            payload["ssp_phot_moment_table"] = np.asarray(table.ssp_phot_moment_table)
        if table.ssp_subband_phot_table is not None:
            payload["ssp_subband_phot_table"] = np.asarray(table.ssp_subband_phot_table)
            payload["subband_waves_rest_table"] = np.asarray(table.subband_waves_rest_table)
        # Atomic publish: concurrent builds of the same key race benignly.
        fd, tmp = tempfile.mkstemp(dir=cache_dir, suffix=".npz.tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                np.savez(f, **payload)
            os.replace(tmp, cache_path)
        except OSError:
            import contextlib

            with contextlib.suppress(OSError):
                os.unlink(tmp)

    return table


def _compute_photometry_ztable(
    ssp_data,
    filter_waves,
    filter_trans,
    z_grid=None,
    z_min=0.001,
    z_max=3.0,
    n_z=100,
    apply_igm=False,
    taylor_correction: bool = False,
    n_subbands: int = 0,
    convention: FilterConvention = FilterConvention.BESSELL,
) -> PhotometricZTable:
    """Pre-compute SSP broadband fluxes on a redshift grid.

    Evaluates the full wavelength integral at each z in the grid.
    At inference time, interpolate to the current z: same speedup
    as fixed-z precomputation, but z is now a free parameter.

    Parameters
    ----------
    ssp_data : SSPData
        SSP templates with ssp_flux [Lsun/Hz/Msun], ssp_wave [Angstrom].
    filter_waves : list of array
        Wavelength grid per filter (observed frame [Angstrom]).
    filter_trans : list of array
        Transmission curve per filter (dimensionless, in [0, 1]).
    z_grid : array, optional
        Custom redshift grid (dimensionless). If None, uses
        linspace(z_min, z_max, n_z).
    z_min : float
        Minimum redshift (dimensionless, default 0.001).
    z_max : float
        Maximum redshift (dimensionless, default 3.0).
    n_z : int
        Number of redshift grid points (default 100).
    apply_igm : bool
        If True, precompute IGM transmission (Inoue+2014) at the
        effective observed wavelengths for each z in the grid.
        Default False (igm_trans_table will be all ones).

    Returns
    -------
    PhotometricZTable
        Pre-computed table for :func:`fast_photometry_ztable` and
        :func:`interpolate_ztable`.

    Notes
    -----
    **JIT-compatible**: no, data precomputation function with nested loops.
    **Gradient-safe**: not applicable (CPU preprocessing).

    """
    from tengri.cosmology import luminosity_distance

    if z_grid is None:
        z_grid = jnp.linspace(z_min, z_max, n_z)
    else:
        z_grid = jnp.asarray(z_grid)

    n_z_pts = len(z_grid)
    n_filters = len(filter_waves)
    n_met = ssp_data.ssp_flux.shape[0]
    n_age = ssp_data.ssp_flux.shape[1]

    import numpy as np

    ssp_phot_all = np.zeros((n_z_pts, n_met, n_age, n_filters))
    eff_waves_rest_all = np.zeros((n_z_pts, n_filters))
    log10_flux_scale_all = np.zeros(n_z_pts)
    igm_trans_all = np.ones((n_z_pts, n_filters))
    # Taylor moment Ψ on the z grid (only if requested).
    ssp_phot_moment_all = (
        np.zeros((n_z_pts, n_met, n_age, n_filters)) if taylor_correction else None
    )
    # n_z_pts, NOT n_z: the caller may pass an explicit ``z_grid``, in which case
    # ``n_z`` is a stale default and the tables come out the wrong length.
    K = int(n_subbands)
    ssp_subband_all = (
        np.zeros((n_z_pts, n_met, n_age, n_filters, K), dtype=np.float64) if K > 0 else None
    )
    subband_waves_all = (
        np.zeros((n_z_pts, n_met, n_age, n_filters, K), dtype=np.float64) if K > 0 else None
    )

    ssp_flux_np = np.asarray(ssp_data.ssp_flux)
    wave_ssp_np = np.asarray(ssp_data.ssp_wave)

    # Filter effective wavelengths (z-independent pivot for the IGM lookup)
    # under the bandpass weight w(λ) (ADR-0017): λ_eff = ∫ λ T w dλ / ∫ T w dλ,
    # evaluated on the filter's own nodes. The integral denominators; and the
    # per-(z, filter) λ_eff the Taylor moment expands around: are computed on
    # the union quadrature grid inside the z loop, matching
    # lnu_filter_integral (#960) so the LUT and exact paths agree.
    eff_waves_obs = []  # observed-frame effective wavelengths (z-independent)
    for fw, ft in zip(filter_waves, filter_trans):
        fw_np, ft_np = np.asarray(fw), np.asarray(ft)
        tw_np = ft_np * _filter_weight_np(fw_np, convention)
        eff_waves_obs.append(
            _np_trapezoid(tw_np * fw_np, fw_np) / max(_np_trapezoid(tw_np, fw_np), 1e-30)
        )
    eff_waves_obs = np.array(eff_waves_obs)

    for zi, z_val in enumerate(z_grid):
        z_val = float(z_val)
        wave_obs = wave_ssp_np * (1.0 + z_val)

        # Effective wavelengths: observed-frame are filter properties (z-independent)
        eff_waves_rest_all[zi] = eff_waves_obs / (1.0 + z_val)

        # IGM transmission at effective observed wavelengths
        if apply_igm:
            from tengri.components.igm import igm_transmission

            igm_trans_all[zi] = np.asarray(igm_transmission(jnp.asarray(eff_waves_obs), z_val))

        # Pre-integrate SSP through each filter (vectorized over met × age)
        # on the union quadrature grid; SED nodes + filter nodes, with the
        # smooth transmission interpolated, never the SED point-sampled at
        # the filter table's nodes (#960). Matches lnu_filter_integral so
        # the LUT and exact paths agree.
        for f_idx, (fw, ft) in enumerate(zip(filter_waves, filter_trans)):
            fw_np, ft_np = np.asarray(fw), np.asarray(ft)
            grid = np.sort(np.concatenate([wave_obs, fw_np]))
            # Transmission is identically zero outside the filter table's
            # support (left=0/right=0 below), so union-grid segments with
            # both endpoints beyond an edge integrate to exactly zero.
            # Keep one node past each edge (the straddle segments) and drop
            # the rest: the SSP grid spans 91 Å–100 µm while a band
            # transmits over a narrow window, so this cuts the quadrature
            # ~10–50× at identical integral values.
            lo = max(np.searchsorted(grid, fw_np[0]) - 1, 0)
            hi = min(np.searchsorted(grid, fw_np[-1], side="right") + 1, grid.size)
            grid = grid[lo:hi]
            trans_on_grid = np.interp(grid, fw_np, ft_np, left=0.0, right=0.0)
            tw_np = trans_on_grid * _filter_weight_np(grid, convention)
            denom = _np_trapezoid(tw_np, grid)
            # λ_eff on the SAME union grid: the Taylor moment must expand
            # around the weight's first moment under its own quadrature, so
            # that Ψ ≡ 0 for a flat template.
            eff_waves_rest_all[zi, f_idx] = (
                _np_trapezoid(tw_np * grid, grid) / max(denom, 1e-30) / (1.0 + z_val)
            )

            ssp_on_grid = _vectorized_interp(grid, wave_obs, ssp_flux_np)
            integrand = ssp_on_grid * tw_np[None, None, :]
            num = _np_trapezoid(integrand, grid, axis=-1)
            ssp_phot_all[zi, :, :, f_idx] = num / max(denom, 1e-30)

            # Taylor moment Ψ at this z and filter.
            # Ψ_{ijb} = ∫ SSP(λ) (λ - λ_eff_rest) T_b(λ_obs) w(λ_obs) dλ_obs / ∫ T_b w dλ_obs
            # Note: λ_eff_rest is the rest-frame effective wavelength of this filter at this z.
            if taylor_correction:
                lambda_rest_per_obs = grid / (1.0 + z_val)
                lambda_minus_eff = lambda_rest_per_obs - eff_waves_rest_all[zi, f_idx]
                moment_integrand = (
                    ssp_on_grid * tw_np[None, None, :] * lambda_minus_eff[None, None, :]
                )
                moment_num = _np_trapezoid(moment_integrand, grid, axis=-1)
                ssp_phot_moment_all[zi, :, :, f_idx] = moment_num / max(denom, 1e-30)

            # Sub-band quadrature (#1122): same helper the fixed-z precompute
            # uses, so the two paths cannot drift. Nodes come back observed-frame;
            # store them rest-frame, which is where the dust law is evaluated.
            if K > 0:
                phi_k, nodes_obs = subband_quadrature(
                    grid, tw_np, integrand, denom, K, float(eff_waves_obs[f_idx])
                )
                ssp_subband_all[zi, :, :, f_idx, :] = phi_k
                subband_waves_all[zi, :, :, f_idx, :] = nodes_obs / (1.0 + z_val)

        # Geometric flux scale, stored as a log10 offset. The build is eager
        # float64 and the linear value is correct here: it is the *storage* that
        # loses it: ~1e-57 casts to exactly 0.0 in a float32 array, so a linear
        # table is zeroed for every z above ~0 (#1859).
        dl_cm = float(luminosity_distance(z_val))
        log10_flux_scale_all[zi] = float(_log10_flux_scale(z_val, dl_cm))

    return PhotometricZTable(
        ssp_phot_table=jnp.array(ssp_phot_all),
        eff_waves_rest_table=jnp.array(eff_waves_rest_all),
        log10_flux_scale_table=jnp.array(log10_flux_scale_all),
        z_grid=z_grid,
        n_filters=n_filters,
        igm_trans_table=jnp.array(igm_trans_all),
        ssp_phot_moment_table=(
            jnp.array(ssp_phot_moment_all) if ssp_phot_moment_all is not None else None
        ),
        ssp_subband_phot_table=(
            jnp.array(ssp_subband_all) if ssp_subband_all is not None else None
        ),
        subband_waves_rest_table=(
            jnp.array(subband_waves_all) if subband_waves_all is not None else None
        ),
    )


# ── Fast inference-time functions (these run inside the MCMC loop)


@jax.jit
def fast_photometry(
    weights: jnp.ndarray,
    ssp_phot_at_z: jnp.ndarray,
    dust_at_eff: jnp.ndarray,
    log10_flux_scale: float,
) -> jnp.ndarray:
    """Compute photometry from pre-computed SSP broadband fluxes.

    This is what runs at each MCMC step. No wavelength integrals;
    just a weighted sum over the age dimension.

    c_gal(band) = 10**log10_flux_scale * sum_i weights_i * dust_i(band) * c_SSP_i(band)

    Parameters
    ----------
    weights : array, shape (n_age,)
        Normalized SFH weights [Msun].
    ssp_phot_at_z : array, shape (n_age, n_filters)
        Pre-computed SSP broadband fluxes [Lsun/Hz/Msun] at the target metallicity.
        Already interpolated in Z from the full grid.
    dust_at_eff : array, shape (n_age, n_filters)
        Dust transmission (dimensionless, in [0, 1]) evaluated at
        effective wavelengths per age/band.
    log10_flux_scale : float
        ``log10`` of the geometric factor (1+z) / (4 π d_L²) [dex re cm⁻²], from
        :func:`tengri.utils.scale.log10_flux_scale`.

    Returns
    -------
    array, shape (n_filters,)
        Observed flux density per filter [erg/s/cm²/Hz].

    Notes
    -----
    **JIT-compatible**: yes, uses ``jnp.einsum`` for fast weighted sum.
    **Gradient-safe**: yes.

    The scale arrives as a ``log10`` offset because the linear factor is ~1e-57
    and exactly ``0.0`` in float32 at every distance; so a float32 fit returned
    zero flux in every band, finite and silent (#1859). The applied product
    (~1e-29) is representable; only the factor was not.
    """
    # Weighted sum: weights [Msun] * ssp [Lsun/Hz/Msun] * dust -> Lsun/Hz
    from tengri.components.stellar.sps.dsps_wrapper import LSUN_ERG_PER_S

    flux_lsun = jnp.einsum("i,if,if->f", weights, dust_at_eff, ssp_phot_at_z)
    return apply_log10_scale(flux_lsun * LSUN_ERG_PER_S, log10_flux_scale)


@jax.jit
def fast_spectrum(
    weights: jnp.ndarray,
    ssp_on_pixels_at_z: jnp.ndarray,
    dust_at_pixels: jnp.ndarray,
    log10_flux_scale: float,
) -> jnp.ndarray:
    """Compute spectrum from pre-rebinned SSP templates.

    Parameters
    ----------
    weights : array, shape (n_age,)
        Normalized SFH weights [Msun].
    ssp_on_pixels_at_z : array, shape (n_age, n_pix)
        Pre-rebinned SSP flux [Lsun/Hz/Msun] at target metallicity.
    dust_at_pixels : array, shape (n_age, n_pix)
        Dust transmission (dimensionless, in [0, 1]) at each pixel
        wavelength per age.
    log10_flux_scale : float
        ``log10`` of the geometric factor (1+z) / (4 π d_L²) [dex re cm⁻²], from
        :func:`tengri.utils.scale.log10_flux_scale`.

    Returns
    -------
    array, shape (n_pix,)
        Model flux [erg/s/cm²/Angstrom or erg/s/cm²/Hz] at each spectral pixel.

    Notes
    -----
    **JIT-compatible**: yes, uses ``jnp.einsum`` for fast weighted sum.
    **Gradient-safe**: yes.

    The scale arrives as a ``log10`` offset for the reason given in
    :func:`fast_photometry`: the linear factor is ``0.0`` in float32 (#1859).
    """
    flux = jnp.einsum("i,ip,ip->p", weights, dust_at_pixels, ssp_on_pixels_at_z)
    return apply_log10_scale(flux, log10_flux_scale)


@jax.jit
def interpolate_ssp_phot_metallicity(
    ssp_phot: jnp.ndarray, ssp_lgmet: jnp.ndarray, log_z: float
) -> jnp.ndarray:
    """Interpolate pre-computed SSP photometry to target metallicity.

    Parameters
    ----------
    ssp_phot : array, shape (n_met, n_age, n_filters)
        Pre-computed SSP broadband fluxes [Lsun/Hz/Msun].
    ssp_lgmet : array, shape (n_met,)
        Metallicity grid [log10(Z/Zsun)], sorted ascending.
    log_z : float
        Target log10(Z/Zsun).

    Returns
    -------
    array, shape (n_age, n_filters)
        Interpolated SSP photometry [Lsun/Hz/Msun].

    Notes
    -----
    **JIT-compatible**: yes, all operations use ``jnp`` primitives.
    **Gradient-safe**: yes, linear interpolation is differentiable.

    """
    log_z_clamped = jnp.clip(log_z, ssp_lgmet[0], ssp_lgmet[-1])
    idx = jnp.clip(jnp.searchsorted(ssp_lgmet, log_z_clamped) - 1, 0, len(ssp_lgmet) - 2)
    frac = (log_z_clamped - ssp_lgmet[idx]) / (ssp_lgmet[idx + 1] - ssp_lgmet[idx])
    return (1.0 - frac) * ssp_phot[idx] + frac * ssp_phot[idx + 1]


@jax.jit
def interpolate_ztable(ztable_ssp_phot, ztable_eff_rest, ztable_log10_flux_scale, z_grid, z):
    """Interpolate z-table to a specific redshift (JIT-compatible).

    Linear interpolation along the z dimension of the precomputed table.

    Parameters
    ----------
    ztable_ssp_phot : array, shape (n_z, n_met, n_age, n_filters)
        Precomputed SSP photometry [Lsun/Hz/Msun] on z grid.
    ztable_eff_rest : array, shape (n_z, n_filters)
        Rest-frame effective wavelengths [Angstrom] on z grid.
    ztable_log10_flux_scale : array, shape (n_z,)
        ``log10`` of the geometric flux scale (1+z)/(4π d_L²) [dex re cm⁻²] on
        the z grid.
    z_grid : array, shape (n_z,)
        Redshift grid (dimensionless), sorted ascending.
    z : float
        Target redshift (dimensionless).

    Returns
    -------
    ssp_phot : array, shape (n_met, n_age, n_filters)
        Interpolated SSP photometry [Lsun/Hz/Msun].
    eff_waves_rest : array, shape (n_filters,)
        Interpolated rest-frame effective wavelengths [Angstrom].
    log10_flux_scale : float
        ``log10`` of the interpolated geometric factor [dex re cm⁻²].

    Notes
    -----
    **JIT-compatible**: yes, all operations use ``jnp`` primitives.
    **Gradient-safe**: yes, linear interpolation is differentiable.

    The flux scale is stored and interpolated in ``log10`` because the linear
    table is ~1e-57 and casts to exactly ``0.0`` in float32: the table builds
    correctly in float64 and is zeroed on storage (#1859).

    **Still the arithmetic interpolation, not the geometric one.**
    :func:`~tengri.utils.scale.log10_weighted_sum` reproduces
    ``(1-f)·s_i + f·s_{i+1}`` exactly; a plain ``lerp`` of the logs would be a
    different function (the geometric mean at the midpoint) and would silently
    move float64 results.
    """
    z_clamped = jnp.clip(z, z_grid[0], z_grid[-1])
    idx = jnp.clip(jnp.searchsorted(z_grid, z_clamped) - 1, 0, len(z_grid) - 2)
    frac = (z_clamped - z_grid[idx]) / (z_grid[idx + 1] - z_grid[idx])

    ssp_phot = (1.0 - frac) * ztable_ssp_phot[idx] + frac * ztable_ssp_phot[idx + 1]
    eff_rest = (1.0 - frac) * ztable_eff_rest[idx] + frac * ztable_eff_rest[idx + 1]
    log10_flux_scale = _log10_weighted_sum(
        jnp.stack([ztable_log10_flux_scale[idx], ztable_log10_flux_scale[idx + 1]]),
        jnp.stack([1.0 - frac, frac]),
    )

    return ssp_phot, eff_rest, log10_flux_scale


@jax.jit
def interpolate_ztable_smooth(
    ztable_ssp_phot: jnp.ndarray,
    ztable_eff_rest: jnp.ndarray,
    ztable_log10_flux_scale: jnp.ndarray,
    z_grid: jnp.ndarray,
    z: float,
    scatter: float,
) -> tuple:
    """Interpolate z-table using the triweight kernel (C²-continuous gradients).

    Preferred over :func:`interpolate_ztable` for free-z inference with
    gradient-based methods (VI, MAP).  Piecewise-linear interpolation has
    discontinuous first derivatives at grid nodes, which manifests as kinks
    in the log-likelihood landscape that slow gradient-based optimizers.
    The triweight kernel integrates the CDF between bin edges (Hearin+2023),
    spreading weight smoothly across neighboring nodes and giving C²-continuous
    ``d(flux)/dz`` gradients throughout.

    Parameters
    ----------
    ztable_ssp_phot : array, shape (n_z, n_met, n_age, n_filters)
        Precomputed SSP photometry [Lsun/Hz/Msun] on z grid.
    ztable_eff_rest : array, shape (n_z, n_filters)
        Rest-frame effective wavelengths [Angstrom] on z grid.
    ztable_log10_flux_scale : array, shape (n_z,)
        ``log10`` of the geometric flux scale (1+z)/(4π d_L²) [dex re cm⁻²] on
        the z grid.
    z_grid : array, shape (n_z,)
        Redshift grid (dimensionless, sorted ascending).
    z : float
        Target redshift (dimensionless).
    scatter : float
        Triweight kernel bandwidth (same units as z_grid).
        Recommended: ``0.5 * dz`` where ``dz`` is the grid spacing;
        gives < 0.05% interpolation error with C²-continuous gradients.
        Larger values spread more weight to neighbors (smoother gradients,
        lower accuracy); smaller values concentrate on the nearest node.

    Returns
    -------
    ssp_phot : array, shape (n_met, n_age, n_filters)
        Interpolated SSP photometry [Lsun/Hz/Msun].
    eff_waves_rest : array, shape (n_filters,)
        Interpolated rest-frame effective wavelengths [Angstrom].
    log10_flux_scale : float
        ``log10`` of the interpolated geometric factor [dex re cm⁻²].

    Notes
    -----
    **JIT-compatible**: yes, uses triweight kernel via :func:`compute_grid_weights`.
    **Gradient-safe**: yes; C²-continuous gradients.

    The kernel-weighted average of the flux scale is taken in ``log10`` for the
    reason given in :func:`interpolate_ztable`, and by the same primitive, so it
    remains the arithmetic weighted sum ``Σ w_z s_z`` exactly (#1859).
    """
    from tengri.utils.interpolation import compute_grid_weights, edges_for_grid

    edges = edges_for_grid(z_grid)
    w = compute_grid_weights(z, z_grid, scatter=scatter, edges=edges)
    ssp_phot = jnp.einsum("z,zmaf->maf", w, ztable_ssp_phot)
    eff_rest = jnp.einsum("z,zf->f", w, ztable_eff_rest)
    log10_flux_scale = _log10_weighted_sum(ztable_log10_flux_scale, w)
    return ssp_phot, eff_rest, log10_flux_scale


@jax.jit
def interpolate_igm_ztable(igm_trans_table, z_grid, z):
    """Interpolate precomputed IGM transmission to a specific redshift.

    Parameters
    ----------
    igm_trans_table : array, shape (n_z, n_filters)
        Precomputed IGM transmission (dimensionless, in [0, 1]) on z grid.
    z_grid : array, shape (n_z,)
        Redshift grid (dimensionless), sorted ascending.
    z : float
        Target redshift (dimensionless).

    Returns
    -------
    igm_trans : array, shape (n_filters,)
        Interpolated IGM transmission (dimensionless).

    Notes
    -----
    **JIT-compatible**: yes, all operations use ``jnp`` primitives.
    **Gradient-safe**: yes, linear interpolation is differentiable.

    """
    z_clamped = jnp.clip(z, z_grid[0], z_grid[-1])
    idx = jnp.clip(jnp.searchsorted(z_grid, z_clamped) - 1, 0, len(z_grid) - 2)
    frac = (z_clamped - z_grid[idx]) / (z_grid[idx + 1] - z_grid[idx])
    return (1.0 - frac) * igm_trans_table[idx] + frac * igm_trans_table[idx + 1]


# ───────────────────────────────────────────────────────────────────
# Protocol-shaped entry points (new in restructure)
# ───────────────────────────────────────────────────────────────────


def precompute(
    filter_waves: list,
    filter_trans: list,
    redshift: float,
    parameters=None,
    *,
    ssp_data,
    dl_cm: float,
    taylor_correction: bool = True,
):
    """Protocol-shaped wrapper around :func:`precompute_photometry`.

    Auto-collapses the metallicity axis if ``met_logzsol`` is Fixed in
    ``parameters``.

    Parameters
    ----------
    filter_waves, filter_trans : list
        Filter curves (observed frame [Angstrom]).
    redshift : float
        Source redshift (dimensionless).
    parameters : Parameters | None
        Parameter spec, used to detect Fixed-axis parameters.
    ssp_data : dict
        SSP data: wavelength [Angstrom], metallicity grid [log10(Z/Zsun)],
        age grid [log10(Gyr)], SSP spectra [Lsun/Hz/Msun].
    dl_cm : float
        Luminosity distance [cm].
    taylor_correction : bool, optional
        If True, compute and include SSP first-moment correction for dust
        attenuation. Default True.

    Returns
    -------
    PhotometricPrecomputation
        Precomputed SSP photometry, optionally collapsed for Fixed parameters.

    Notes
    -----
    **JIT-compatible**: no, data precomputation with conditional collapsing.
    **Gradient-safe**: not applicable (CPU preprocessing).

    """
    fixed: dict[int, float] | None = None
    if parameters is not None and parameters.is_fixed("met_logzsol"):
        # Convert met_logzsol (log10(Z/Zsun)) to absolute log10(Z) via LOG10_ZSUN.
        from tengri.parameters.translate import LOG10_ZSUN

        met_logz_abs = float(parameters.fixed_value("met_logzsol")) + LOG10_ZSUN
        fixed = {0: met_logz_abs}
    return precompute_photometry(
        ssp_data=ssp_data,
        filter_waves=filter_waves,
        filter_trans=filter_trans,
        redshift=redshift,
        dl_cm=dl_cm,
        taylor_correction=taylor_correction,
        fixed=fixed,
    )


def build_lookup(preint, **kwargs):
    """Return SSP photometry lookup for the fused kernels.

    SSP photometry is consumed directly by the fused kernels; no JIT
    lookup is returned here. This is a Protocol placeholder.

    Parameters
    ----------
    preint : PhotometricPrecomputation
        Precomputed SSP photometry with ssp_phot, effective wavelengths,
        and flux scaling factors.
    **kwargs
        Ignored; accepted for Protocol consistency.

    Returns
    -------
    None
        SSP photometry is used directly in the fused kernels without a
        separate JIT lookup function.

    Notes
    -----
    **JIT-compatible**: not applicable (placeholder function).
    **Gradient-safe**: not applicable (placeholder function).

    This function implements the Protocol interface but performs no operation.
    The precomputed photometry is passed directly to fast_photometry() or
    similar kernel functions.

    """
    return None
