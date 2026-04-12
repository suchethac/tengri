"""Kubota & Done (2018) 3-zone disc preintegration through photometric filters.

Precomputes filter-integrated lookup tables for the three K&D disc zones:

1. **Planck table** (Zone 1, outer disc): B_nu(T) integrated through each
   filter as a function of temperature.
2. **nthcomp table** (Zone 2, warm Comptonization): nthcomp spectral shape
   integrated through each filter as a function of (gamma, kTe, kTbb).
3. **Corona table** (Zone 3, hot corona): cutoff power law integrated through
   each filter as a function of (Gamma, kT_hot).

At runtime the radial integration operates on filter-level quantities
(n_filters numbers per ring) instead of wavelength-level (17,000 per ring),
giving ~40x speedup for the K&D component.

References
----------
- Kubota & Done 2018, MNRAS, 480, 1247
- docs/dev/kd18-precomputation.md (design document)
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from tengri.models.agn._phys import (
    C_LIGHT as _C_LIGHT,
    H_PLANCK as _H_PLANCK,
    K_BOLTZ as _K_BOLTZ,
)
from tengri.utils.physics_constants import KEV_TO_ERG as _KEV_TO_ERG

# numpy >= 2.0 uses trapezoid; older versions used trapz
_np_trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


# ───────────────────────────────────────────────────────────────────
# Data classes
# ───────────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class KDPreintegratedData:
    """Preintegrated K&D 2018 disc components for photometric fast-path.

    All tables store filter-integrated spectral shapes per grid point.
    At runtime, ring contributions are looked up and summed — no
    wavelength-level computation needed.

    Attributes
    ----------
    planck_table : jnp.ndarray
        Shape (n_T, n_filters). Filter-integrated Planck B_nu(T) for each
        temperature grid point and filter.
    planck_T_grid : jnp.ndarray
        Shape (n_T,). Temperature grid [K] (log-spaced).
    nthcomp_table : jnp.ndarray or None
        Shape (n_gamma, n_kTe, n_kTbb, n_filters). Filter-integrated
        nthcomp spectral shape. None if templates not available.
    nthcomp_gamma_grid : jnp.ndarray or None
        Shape (n_gamma,).
    nthcomp_kTe_grid : jnp.ndarray or None
        Shape (n_kTe,).
    nthcomp_kTbb_grid : jnp.ndarray or None
        Shape (n_kTbb,).
    corona_table : jnp.ndarray
        Shape (n_Gamma, n_kT, n_filters). Filter-integrated cutoff
        power-law shape for the hot corona.
    corona_Gamma_grid : jnp.ndarray
        Shape (n_Gamma,). Hard X-ray photon index grid.
    corona_kT_grid : jnp.ndarray
        Shape (n_kT,). Hot corona temperature grid [keV].
    effective_bandwidths_hz : jnp.ndarray
        Shape (n_filters,). Effective frequency bandwidths for L_bol estimation.
    n_filters : int
        Number of photometric filters.
    """

    planck_table: jnp.ndarray
    planck_T_grid: jnp.ndarray
    nthcomp_table: jnp.ndarray | None
    nthcomp_gamma_grid: jnp.ndarray | None
    nthcomp_kTe_grid: jnp.ndarray | None
    nthcomp_kTbb_grid: jnp.ndarray | None
    corona_table: jnp.ndarray
    corona_Gamma_grid: jnp.ndarray
    corona_kT_grid: jnp.ndarray
    effective_bandwidths_hz: jnp.ndarray
    n_filters: int


# ───────────────────────────────────────────────────────────────────
# Planck filter table (Zone 1: outer disc)
# ───────────────────────────────────────────────────────────────────


def _build_planck_filter_table(
    T_grid: np.ndarray,
    filter_waves: list[np.ndarray],
    filter_trans: list[np.ndarray],
    redshift: float,
) -> np.ndarray:
    """Precompute Planck B_nu(T) integrated through filters.

    For each T in T_grid and each filter b:
        planck_table[i, b] = int[B_nu(T_i, lam) T_b(lam) lam dlam]
                             / int[T_b(lam) lam dlam]

    where B_nu is evaluated at observed-frame wavelengths shifted to
    rest frame: nu = c / (lam_obs / (1+z)).

    Parameters
    ----------
    T_grid : ndarray, shape (n_T,)
        Temperature grid [K].
    filter_waves : list[ndarray]
        Wavelength grid per filter [Angstrom], observed frame.
    filter_trans : list[ndarray]
        Transmission per filter.
    redshift : float
        Source redshift.

    Returns
    -------
    ndarray, shape (n_T, n_filters)
        Filter-integrated Planck function.
    """
    n_T = len(T_grid)
    n_filters = len(filter_waves)
    table = np.zeros((n_T, n_filters), dtype=np.float64)

    for f_idx, (fw, ft) in enumerate(zip(filter_waves, filter_trans)):
        fw_np = np.asarray(fw, dtype=np.float64)
        ft_np = np.asarray(ft, dtype=np.float64)

        # Rest-frame frequency at each filter wavelength point
        wave_rest = fw_np / (1.0 + redshift)
        nu = _C_LIGHT / (wave_rest * 1e-8)  # Angstrom -> cm -> Hz

        # Filter denominator: int[T(lam) lam dlam]
        denom = _np_trapezoid(ft_np * fw_np, fw_np)
        if denom <= 0:
            continue

        for i_T, T in enumerate(T_grid):
            if T <= 0:
                continue
            # Planck B_nu(T) at each frequency
            x = np.clip(_H_PLANCK * nu / (_K_BOLTZ * T), 1e-10, 500.0)
            b_nu = 2.0 * _H_PLANCK * nu**3 / _C_LIGHT**2 / np.expm1(x)
            # Integrate: int[B_nu T_b lam dlam] / int[T_b lam dlam]
            table[i_T, f_idx] = _np_trapezoid(b_nu * ft_np * fw_np, fw_np) / denom

    return table


# ───────────────────────────────────────────────────────────────────
# nthcomp filter table (Zone 2: warm Comptonization)
# ───────────────────────────────────────────────────────────────────


def _build_nthcomp_filter_table(
    filter_waves: list[np.ndarray],
    filter_trans: list[np.ndarray],
    redshift: float,
    template_path: Path | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Precompute nthcomp spectral shape integrated through filters.

    Reads the precomputed nthcomp template table (gamma, kTe, kTbb, nu)
    and collapses the nu dimension into filter integrals.

    Parameters
    ----------
    filter_waves : list[ndarray]
        Wavelength grid per filter [Angstrom], observed frame.
    filter_trans : list[ndarray]
        Transmission per filter.
    redshift : float
        Source redshift.
    template_path : Path, optional
        Path to nthcomp_templates.h5. Defaults to data/nthcomp_templates.h5.

    Returns
    -------
    tuple (table, gamma_grid, kTe_grid, kTbb_grid)
        table: (n_gamma, n_kTe, n_kTbb, n_filters) or None
        Remaining are 1D grid arrays or None if templates unavailable.
    """
    import h5py

    if template_path is None:
        template_path = Path(__file__).parents[4] / "data" / "nthcomp_templates.h5"

    if not template_path.exists():
        return None, None, None, None

    try:
        with h5py.File(template_path, "r") as f:
            gamma_grid = f["gamma_grid"][:].astype(np.float64)
            kTe_grid = f["kte_grid"][:].astype(np.float64)
            kTbb_grid = f["ktbb_grid"][:].astype(np.float64)
            nu_grid = f["nu_grid"][:].astype(np.float64)
            # shape: (n_gamma, n_kTe, n_kTbb, n_nu)
            raw_table = f["table"][:].astype(np.float64)
    except Exception:
        return None, None, None, None

    n_gamma, n_kTe, n_kTbb, n_nu = raw_table.shape
    n_filters = len(filter_waves)

    # For each filter, compute the integral of the nthcomp shape through it.
    # The nthcomp shape at each (gamma, kTe, kTbb) is stored on the nu_grid.
    # We need to interpolate it onto filter frequencies, then integrate.
    table = np.zeros((n_gamma, n_kTe, n_kTbb, n_filters), dtype=np.float64)

    for f_idx, (fw, ft) in enumerate(zip(filter_waves, filter_trans)):
        fw_np = np.asarray(fw, dtype=np.float64)
        ft_np = np.asarray(ft, dtype=np.float64)

        # Rest-frame frequency at each filter wavelength point
        wave_rest = fw_np / (1.0 + redshift)
        nu_filter = _C_LIGHT / (wave_rest * 1e-8)  # Hz

        # Filter denominator
        denom = _np_trapezoid(ft_np * fw_np, fw_np)
        if denom <= 0:
            continue

        # Flatten the 3D grid for vectorized interpolation
        # raw_table shape: (n_gamma, n_kTe, n_kTbb, n_nu)
        flat_table = raw_table.reshape(-1, n_nu)  # (N, n_nu)

        # Interpolate each grid point's spectrum onto filter frequencies
        # nu_grid is sorted ascending; nu_filter may be descending (wave ascending)
        # We need nu_grid sorted ascending for np.interp
        sort_idx = np.argsort(nu_grid)
        nu_sorted = nu_grid[sort_idx]
        flat_sorted = flat_table[:, sort_idx]

        # Vectorized interpolation: all grid points at once
        from tengri.core.preintegrate import _vectorized_interp

        # _vectorized_interp expects (xp_target, xp_source, yp_source)
        # where yp_source shape is (..., n_source)
        spectra_on_filter = _vectorized_interp(nu_filter, nu_sorted, flat_sorted)
        # shape: (N, n_filter_points)

        # Integrate: int[shape * T_b * lam dlam] / int[T_b * lam dlam]
        integrand = spectra_on_filter * ft_np[None, :] * fw_np[None, :]
        numerator = _np_trapezoid(integrand, fw_np, axis=-1)
        table_flat = numerator / denom

        table[:, :, :, f_idx] = table_flat.reshape(n_gamma, n_kTe, n_kTbb)

    return table, gamma_grid, kTe_grid, kTbb_grid


# ───────────────────────────────────────────────────────────────────
# Corona filter table (Zone 3: hot corona)
# ───────────────────────────────────────────────────────────────────


def _build_corona_filter_table(
    Gamma_grid: np.ndarray,
    kT_grid_keV: np.ndarray,
    filter_waves: list[np.ndarray],
    filter_trans: list[np.ndarray],
    redshift: float,
) -> np.ndarray:
    """Precompute cutoff power-law corona shape integrated through filters.

    The corona spectrum is: shape(nu) = nu^(1 - Gamma) * exp(-h*nu / kT)
    normalized so int[shape dnu] = 1 on the fixed RELAGN grid.

    The normalization uses the same fixed [1e-4, 1e4] keV grid as
    ``_hot_corona_lnu`` in ``disc.py``, matching the RELAGN reference
    implementation. This makes the result grid-independent.

    Parameters
    ----------
    Gamma_grid : ndarray, shape (n_Gamma,)
        Hard X-ray photon index grid.
    kT_grid_keV : ndarray, shape (n_kT,)
        Hot corona temperature grid [keV].
    filter_waves : list[ndarray]
        Wavelength grid per filter [Angstrom], observed frame.
    filter_trans : list[ndarray]
        Transmission per filter.
    redshift : float
        Source redshift.

    Returns
    -------
    ndarray, shape (n_Gamma, n_kT, n_filters)
        Filter-integrated normalized corona shape.
    """
    n_Gamma = len(Gamma_grid)
    n_kT = len(kT_grid_keV)
    n_filters = len(filter_waves)
    table = np.zeros((n_Gamma, n_kT, n_filters), dtype=np.float64)

    # Fixed normalization grid matching _hot_corona_lnu and RELAGN:
    # [1e-4, 1e4] keV = [2.418e13, 2.418e21] Hz, 2000 log-spaced points.
    nu_wide = np.geomspace(2.418e13, 2.418e21, 2000)

    for i_G, Gamma in enumerate(Gamma_grid):
        for i_T, kT_keV in enumerate(kT_grid_keV):
            kT_erg = kT_keV * _KEV_TO_ERG
            # Cutoff power-law shape
            x = np.clip(_H_PLANCK * nu_wide / kT_erg, 0.0, 500.0)
            shape = nu_wide ** (1.0 - Gamma) * np.exp(-x)

            # Normalize to unit bolometric: int[shape dnu] = 1
            integral = _np_trapezoid(shape, nu_wide)
            if integral <= 0:
                continue
            shape_normed = shape / integral

            # Integrate through each filter
            for f_idx, (fw, ft) in enumerate(zip(filter_waves, filter_trans)):
                fw_np = np.asarray(fw, dtype=np.float64)
                ft_np = np.asarray(ft, dtype=np.float64)

                wave_rest = fw_np / (1.0 + redshift)
                nu_filter = _C_LIGHT / (wave_rest * 1e-8)

                denom = _np_trapezoid(ft_np * fw_np, fw_np)
                if denom <= 0:
                    continue

                # Interpolate normalized shape onto filter frequencies
                sort_idx = np.argsort(nu_filter)
                nu_filt_sorted = nu_filter[sort_idx]
                shape_on_filt = np.interp(
                    nu_filt_sorted, nu_wide, shape_normed, left=0.0, right=0.0
                )
                # Unsort back to filter wavelength order
                unsort_idx = np.argsort(sort_idx)
                shape_on_filt = shape_on_filt[unsort_idx]

                num = _np_trapezoid(shape_on_filt * ft_np * fw_np, fw_np)
                table[i_G, i_T, f_idx] = num / denom

    return table


# ───────────────────────────────────────────────────────────────────
# Effective bandwidths for L_bol estimation
# ───────────────────────────────────────────────────────────────────


def _compute_effective_bandwidths_hz(
    filter_waves: list[np.ndarray],
    filter_trans: list[np.ndarray],
    redshift: float,
) -> np.ndarray:
    """Compute effective frequency bandwidth per filter for L_bol estimation.

    Returns Voronoi-style frequency widths so that sum(f_nu * bw) ~ L_bol.

    Parameters
    ----------
    filter_waves : list[ndarray]
        Wavelength grid per filter [Angstrom], observed frame.
    filter_trans : list[ndarray]
        Transmission per filter.
    redshift : float
        Source redshift.

    Returns
    -------
    ndarray, shape (n_filters,)
        Effective frequency bandwidths [Hz].
    """
    # Compute effective frequencies per filter
    n_filters = len(filter_waves)
    eff_nu = np.zeros(n_filters)
    for f_idx, (fw, ft) in enumerate(zip(filter_waves, filter_trans)):
        fw_np = np.asarray(fw, dtype=np.float64)
        ft_np = np.asarray(ft, dtype=np.float64)
        wave_rest = fw_np / (1.0 + redshift)
        nu = _C_LIGHT / (wave_rest * 1e-8)
        # Transmission-weighted mean frequency
        denom = _np_trapezoid(ft_np * fw_np, fw_np)
        if denom > 0:
            eff_nu[f_idx] = _np_trapezoid(ft_np * fw_np * nu, fw_np) / denom
        else:
            eff_nu[f_idx] = _C_LIGHT / (np.mean(fw_np) * 1e-8)

    # Sort by frequency for Voronoi partitioning
    sort_idx = np.argsort(eff_nu)
    eff_nu_sorted = eff_nu[sort_idx]

    # Voronoi bandwidths: midpoints between adjacent filters
    bw_sorted = np.zeros(n_filters)
    for i in range(n_filters):
        lo = eff_nu_sorted[i - 1] if i > 0 else eff_nu_sorted[i] * 0.5
        hi = eff_nu_sorted[i + 1] if i < n_filters - 1 else eff_nu_sorted[i] * 2.0
        bw_sorted[i] = 0.5 * (hi - lo)

    # Unsort back to original filter order
    bw = np.zeros(n_filters)
    bw[sort_idx] = bw_sorted

    return bw


# ───────────────────────────────────────────────────────────────────
# Public API: build everything
# ───────────────────────────────────────────────────────────────────


def preintegrate_kd_components(
    filter_waves: list[np.ndarray],
    filter_trans: list[np.ndarray],
    redshift: float,
    n_T: int = 200,
    T_min: float = 100.0,
    T_max: float = 3e7,
    n_Gamma: int = 20,
    n_kT_corona: int = 15,
) -> KDPreintegratedData:
    """Precompute all K&D disc filter tables at model init time.

    Builds lookup tables for the three K&D zones by integrating their
    spectral shapes through the photometric filters. This is called
    once at ``Model.__init__`` when K&D AGN is enabled with fixed
    redshift and photometric filters.

    Parameters
    ----------
    filter_waves : list[ndarray]
        Wavelength grid per filter [Angstrom], observed frame.
    filter_trans : list[ndarray]
        Transmission per filter (0-1).
    redshift : float
        Source redshift (fixed).
    n_T : int
        Number of temperature grid points for the Planck table.
    T_min, T_max : float
        Temperature range [K] for the Planck table. Must span from
        cool outer disc (~1000 K) to hot inner disc (~3e7 K).
    n_Gamma : int
        Number of photon index grid points for the corona table.
    n_kT_corona : int
        Number of temperature grid points for the corona table.

    Returns
    -------
    KDPreintegratedData
        All precomputed tables ready for runtime lookup.
    """
    n_filters = len(filter_waves)

    # --- Planck table ---
    T_grid = np.geomspace(T_min, T_max, n_T)
    planck_table = _build_planck_filter_table(T_grid, filter_waves, filter_trans, redshift)

    # --- nthcomp table ---
    nthcomp_table, gamma_grid, kTe_grid, kTbb_grid = _build_nthcomp_filter_table(
        filter_waves, filter_trans, redshift
    )

    # --- Corona table ---
    Gamma_grid = np.linspace(1.4, 3.0, n_Gamma)
    kT_grid_keV = np.geomspace(10.0, 500.0, n_kT_corona)  # 10-500 keV
    corona_table = _build_corona_filter_table(
        Gamma_grid,
        kT_grid_keV,
        filter_waves,
        filter_trans,
        redshift,
    )

    # --- Effective bandwidths ---
    bw_hz = _compute_effective_bandwidths_hz(filter_waves, filter_trans, redshift)

    return KDPreintegratedData(
        planck_table=jnp.asarray(planck_table),
        planck_T_grid=jnp.asarray(T_grid),
        nthcomp_table=jnp.asarray(nthcomp_table) if nthcomp_table is not None else None,
        nthcomp_gamma_grid=jnp.asarray(gamma_grid) if gamma_grid is not None else None,
        nthcomp_kTe_grid=jnp.asarray(kTe_grid) if kTe_grid is not None else None,
        nthcomp_kTbb_grid=jnp.asarray(kTbb_grid) if kTbb_grid is not None else None,
        corona_table=jnp.asarray(corona_table),
        corona_Gamma_grid=jnp.asarray(Gamma_grid),
        corona_kT_grid=jnp.asarray(kT_grid_keV),
        effective_bandwidths_hz=jnp.asarray(bw_hz),
        n_filters=n_filters,
    )


# ───────────────────────────────────────────────────────────────────
# Runtime: preintegrated K&D disc (filter-level radial integration)
# ───────────────────────────────────────────────────────────────────


def _lookup_planck_filter(
    T: jnp.ndarray,
    T_grid: jnp.ndarray,
    planck_table: jnp.ndarray,
) -> jnp.ndarray:
    """Look up filter-integrated Planck B_nu at temperature T.

    Linear interpolation in log(T) space for smooth gradients.

    Parameters
    ----------
    T : scalar
        Temperature [K].
    T_grid : (n_T,)
        Temperature grid [K] (log-spaced).
    planck_table : (n_T, n_filters)
        Precomputed Planck filter table.

    Returns
    -------
    (n_filters,) — filter-integrated B_nu at temperature T.
    """
    log_T = jnp.log10(jnp.maximum(T, 1.0))
    log_T_grid = jnp.log10(T_grid)

    # Clamped linear interpolation in log(T)
    idx_hi = jnp.searchsorted(log_T_grid, log_T, side="right")
    idx_lo = jnp.clip(idx_hi - 1, 0, len(T_grid) - 2)
    idx_hi = jnp.clip(idx_hi, 1, len(T_grid) - 1)

    span = log_T_grid[idx_hi] - log_T_grid[idx_lo]
    frac = jnp.where(span > 0, (log_T - log_T_grid[idx_lo]) / span, 0.0)
    frac = jnp.clip(frac, 0.0, 1.0)

    return planck_table[idx_lo] * (1.0 - frac) + planck_table[idx_hi] * frac


def _lookup_nthcomp_filter(
    gamma: jnp.ndarray,
    kTe: jnp.ndarray,
    kTbb: jnp.ndarray,
    gamma_grid: jnp.ndarray,
    kTe_grid: jnp.ndarray,
    kTbb_grid: jnp.ndarray,
    nthcomp_table: jnp.ndarray,
) -> jnp.ndarray:
    """Look up filter-integrated nthcomp shape via trilinear interpolation.

    Parameters
    ----------
    gamma, kTe, kTbb : scalar
        Query point (photon index, electron T [keV], seed T [keV]).
    *_grid : 1D arrays
        Grid axes.
    nthcomp_table : (n_gamma, n_kTe, n_kTbb, n_filters)
        Precomputed table.

    Returns
    -------
    (n_filters,) — filter-integrated nthcomp shape.
    """

    def _interp_axis(val, grid):
        n = grid.shape[0]
        idx_hi = jnp.searchsorted(grid, val, side="right")
        idx_lo = jnp.clip(idx_hi - 1, 0, n - 2)
        idx_hi_c = jnp.clip(idx_hi, 1, n - 1)
        span = grid[idx_hi_c] - grid[idx_lo]
        frac = jnp.where(span > 0, (val - grid[idx_lo]) / span, 0.0)
        return idx_lo, jnp.clip(frac, 0.0, 1.0)

    ig, fg = _interp_axis(gamma, gamma_grid)
    it, ft = _interp_axis(kTe, kTe_grid)
    ib, fb = _interp_axis(kTbb, kTbb_grid)

    # Trilinear interpolation over 8 corners
    def _c(dg, dt, db):
        return nthcomp_table[ig + dg, it + dt, ib + db]

    s00 = _c(0, 0, 0) * (1 - fg) + _c(1, 0, 0) * fg
    s10 = _c(0, 1, 0) * (1 - fg) + _c(1, 1, 0) * fg
    s01 = _c(0, 0, 1) * (1 - fg) + _c(1, 0, 1) * fg
    s11 = _c(0, 1, 1) * (1 - fg) + _c(1, 1, 1) * fg
    s0 = s00 * (1 - ft) + s10 * ft
    s1 = s01 * (1 - ft) + s11 * ft
    return s0 * (1 - fb) + s1 * fb


def _lookup_corona_filter(
    Gamma: jnp.ndarray,
    kT_keV: jnp.ndarray,
    Gamma_grid: jnp.ndarray,
    kT_grid: jnp.ndarray,
    corona_table: jnp.ndarray,
) -> jnp.ndarray:
    """Look up filter-integrated corona shape via bilinear interpolation.

    Parameters
    ----------
    Gamma : scalar
        Hard X-ray photon index.
    kT_keV : scalar
        Hot corona temperature [keV].
    Gamma_grid, kT_grid : 1D arrays
        Grid axes.
    corona_table : (n_Gamma, n_kT, n_filters)
        Precomputed table.

    Returns
    -------
    (n_filters,) — filter-integrated normalized corona shape.
    """

    def _interp_axis(val, grid):
        n = grid.shape[0]
        idx_hi = jnp.searchsorted(grid, val, side="right")
        idx_lo = jnp.clip(idx_hi - 1, 0, n - 2)
        idx_hi_c = jnp.clip(idx_hi, 1, n - 1)
        span = grid[idx_hi_c] - grid[idx_lo]
        frac = jnp.where(span > 0, (val - grid[idx_lo]) / span, 0.0)
        return idx_lo, jnp.clip(frac, 0.0, 1.0)

    iG, fG = _interp_axis(Gamma, Gamma_grid)
    iT, fT = _interp_axis(kT_keV, kT_grid)

    # Bilinear interpolation
    c00 = corona_table[iG, iT]
    c10 = corona_table[iG + 1, iT]
    c01 = corona_table[iG, iT + 1]
    c11 = corona_table[iG + 1, iT + 1]

    s0 = c00 * (1 - fG) + c10 * fG
    s1 = c01 * (1 - fG) + c11 * fG
    return s0 * (1 - fT) + s1 * fT


def kubota_done_disc_preintegrated(
    kd_data: KDPreintegratedData,
    agn_log_lbol: float,
    agn_frac: float = 1.0,
    agn_log_mbh: float = 8.0,
    agn_log_ledd: float = -1.0,
    agn_a_spin: float = 0.0,
    agn_cos_inc: float = 0.5,
    agn_f_hard: float = 0.02,
    agn_gamma_warm: float = 2.5,
    agn_kt_warm: float = 0.2,
    agn_gamma_hard: float = 1.8,
    agn_kt_hot: float = 100.0,
    agn_r_warm_ratio: float = 2.0,
    n_radii: int = 50,
    agn_self_consistent_gamma: bool = False,
    **_kwargs,
) -> jnp.ndarray:
    """Preintegrated K&D (2018) 3-zone disc — filter-level radial integration.

    Same physics as ``kubota_done_disc()`` but operates on filter-level
    quantities (n_filters per ring) instead of wavelength-level (17k per ring).
    Zone radii, temperatures, and normalization are identical.

    Parameters
    ----------
    kd_data : KDPreintegratedData
        Precomputed filter tables from ``preintegrate_kd_components()``.
    agn_log_lbol : float
        log10(L_bol / Lsun).
    [remaining params identical to kubota_done_disc]

    Returns
    -------
    array, shape (n_filters,)
        Filter-integrated L_nu per filter [erg/s/Hz], NOT flux-scaled.
        Caller must apply flux_scale = (1+z)/(4 pi dL^2).
    """
    from tengri.models.agn.disc import (
        _eddington_luminosity,
        _gravitational_radius,
        _isco_radius,
        _l_seed_geometric,
        _r_hot_bisect,
        _self_gravity_radius,
        beloborodov_gamma_hot,
    )
    from tengri.utils.physics_constants import (
        G_GRAV as _G_GRAV,
        K_BOLTZ_KEV as _K_BOLTZ_KEV,
        M_SUN as _MSUN_G,
        SIGMA_SB as _SIGMA_SB,
    )

    # --- Black hole parameters (identical to kubota_done_disc) ---
    r_g = _gravitational_radius(agn_log_mbh)
    r_isco_rg = _isco_radius(agn_a_spin)
    r_isco_cm = r_isco_rg * r_g

    eta = 1.0 - jnp.sqrt(1.0 - 2.0 / (3.0 * r_isco_rg))
    l_edd = _eddington_luminosity(agn_log_mbh)
    l_edd_ratio = jnp.clip(10.0**agn_log_ledd, 1e-10, 1.0)
    l_bol_erg = l_edd_ratio * l_edd
    mdot = l_bol_erg / (eta * _C_LIGHT**2)

    t_in = (
        3.0
        * _G_GRAV
        * 10.0**agn_log_mbh
        * _MSUN_G
        * mdot
        / (8.0 * jnp.pi * _SIGMA_SB * r_isco_cm**3)
    ) ** 0.25

    # --- Zone radii (identical) ---
    f_hard_safe = jnp.clip(agn_f_hard, 1e-6, 0.5)
    l_hot_target = f_hard_safe * l_edd
    r_hot_cm = _r_hot_bisect(r_isco_cm, t_in, l_hot_target)

    r_warm_ratio_safe = jnp.clip(agn_r_warm_ratio, 1.1, 10.0)
    r_warm_cm = r_hot_cm * r_warm_ratio_safe

    r_sg_rg = _self_gravity_radius(agn_log_mbh, l_edd_ratio)
    r_out_cm = jnp.maximum(r_sg_rg, r_isco_rg * 10.0) * r_g

    r_hot_cm = jnp.clip(r_hot_cm, r_isco_cm * 1.01, r_out_cm * 0.5)
    r_warm_cm = jnp.clip(r_warm_cm, r_hot_cm * 1.01, r_out_cm * 0.9)

    # ===============================================================
    # Zone 1: Outer standard disc — filter-level Planck lookup
    # ===============================================================
    log_r_warm = jnp.log10(r_warm_cm)
    log_r_out = jnp.log10(r_out_cm)
    log_r_outer = jnp.linspace(log_r_warm, log_r_out, n_radii)
    r_outer = 10.0**log_r_outer

    r_ratio_outer = r_outer / r_isco_cm
    torque_outer = jnp.maximum(1.0 - jnp.sqrt(1.0 / r_ratio_outer), 1e-30) ** 0.25
    t_outer = t_in * r_ratio_outer ** (-0.75) * torque_outer

    d_log_r_outer = log_r_outer[1] - log_r_outer[0]
    dr_outer = r_outer * jnp.log(10.0) * d_log_r_outer

    def _outer_ring_phot(r_cm, t_ring, dr_ring):
        b_filt = _lookup_planck_filter(t_ring, kd_data.planck_T_grid, kd_data.planck_table)
        area = jnp.pi * 2.0 * jnp.pi * r_cm * dr_ring
        return b_filt * area * jnp.maximum(agn_cos_inc, 0.01)

    outer_phot = jnp.sum(
        jax.vmap(_outer_ring_phot)(r_outer, t_outer, dr_outer), axis=0
    )  # (n_filters,)

    # Bolometric from outer disc: sigma*T^4 * dA * cos(i)
    # This matches the spectral integral of pi*B_nu*dA*cos(i) over all nu.
    outer_bol = jnp.sum(
        _SIGMA_SB * t_outer**4 * 2.0 * jnp.pi * r_outer * dr_outer * jnp.maximum(agn_cos_inc, 0.01)
    )

    # ===============================================================
    # Zone 2: Warm Comptonization — filter-level nthcomp lookup
    # ===============================================================
    log_r_hot = jnp.log10(r_hot_cm)
    log_r_warm_grid = jnp.linspace(log_r_hot, log_r_warm, n_radii)
    r_warm_grid = 10.0**log_r_warm_grid

    r_ratio_warm = r_warm_grid / r_isco_cm
    torque_warm = jnp.maximum(1.0 - jnp.sqrt(1.0 / r_ratio_warm), 1e-30) ** 0.25
    t_warm = t_in * r_ratio_warm ** (-0.75) * torque_warm

    d_log_r_warm = log_r_warm_grid[1] - log_r_warm_grid[0]
    dr_warm = r_warm_grid * jnp.log(10.0) * d_log_r_warm

    has_nthcomp = kd_data.nthcomp_table is not None

    if has_nthcomp:

        def _warm_ring_phot(r_cm, t_ring, dr_ring):
            # Bolometric power from this ring via Stefan-Boltzmann (exact).
            # This matches the full-wavelength path which computes
            # p_plain = abs(trapezoid(B_nu, nu)) ≈ sigma*T^4/pi
            # then l_total = p_plain * pi * 2*pi*r*dr * cos(i)
            #              = sigma*T^4 * 2*pi*r*dr * cos(i)
            area = jnp.pi * 2.0 * jnp.pi * r_cm * dr_ring
            l_total = _SIGMA_SB * t_ring**4 / jnp.pi * area * jnp.maximum(agn_cos_inc, 0.01)

            kTbb_keV = _K_BOLTZ_KEV * t_ring
            nthcomp_filt = _lookup_nthcomp_filter(
                agn_gamma_warm,
                agn_kt_warm,
                kTbb_keV,
                kd_data.nthcomp_gamma_grid,
                kd_data.nthcomp_kTe_grid,
                kd_data.nthcomp_kTbb_grid,
                kd_data.nthcomp_table,
            )
            return nthcomp_filt * l_total
    else:
        # Fallback: use Planck lookup (no Comptonization enhancement)
        def _warm_ring_phot(r_cm, t_ring, dr_ring):
            b_filt = _lookup_planck_filter(t_ring, kd_data.planck_T_grid, kd_data.planck_table)
            area = jnp.pi * 2.0 * jnp.pi * r_cm * dr_ring
            return b_filt * area * jnp.maximum(agn_cos_inc, 0.01)

    warm_phot = jnp.sum(
        jax.vmap(_warm_ring_phot)(r_warm_grid, t_warm, dr_warm), axis=0
    )  # (n_filters,)

    # Bolometric from warm zone: same energy conservation (nthcomp preserves bolometric)
    warm_bol = jnp.sum(
        _SIGMA_SB
        * t_warm**4
        * 2.0
        * jnp.pi
        * r_warm_grid
        * dr_warm
        * jnp.maximum(agn_cos_inc, 0.01)
    )

    # ===============================================================
    # Zone 3: Hot corona — single filter lookup
    # ===============================================================
    l_hot_erg = jnp.minimum(f_hard_safe * l_edd, l_bol_erg * 0.5)

    # Self-consistent Gamma (same as full-wavelength path)
    l_seed_geom = _l_seed_geometric(r_isco_cm, r_hot_cm, r_out_cm, t_in)
    gamma_hard_sc = beloborodov_gamma_hot(l_hot_erg, l_seed_geom)
    gamma_hard_eff = jnp.where(agn_self_consistent_gamma, gamma_hard_sc, agn_gamma_hard)

    corona_filt = _lookup_corona_filter(
        gamma_hard_eff,
        agn_kt_hot,
        kd_data.corona_Gamma_grid,
        kd_data.corona_kT_grid,
        kd_data.corona_table,
    )
    hot_phot = l_hot_erg * corona_filt  # (n_filters,)

    # ===============================================================
    # Combine and normalize to L_bol * agn_frac
    # ===============================================================
    # The full-wavelength code normalizes: scale = L_bol_requested / int[L_nu dnu].
    # We can't compute int[L_nu dnu] from filter photometry alone (filters miss
    # most of the bolometric SED for UV/X-ray-bright AGN). Instead, compute the
    # bolometric from the Stefan-Boltzmann integral of each zone's radial grid:
    #   L_bol = sum_rings(sigma * T^4 * dA) for disc zones + L_hot for corona.
    # This equals the spectral integral by energy conservation (same as the
    # full-wavelength code's trapezoid integral, to numerical precision).
    l_bol_unnorm = outer_bol + warm_bol + l_hot_erg
    l_bol_requested = 10.0**agn_log_lbol * 3.828e33 * agn_frac  # LSUN_ERG
    scale = l_bol_requested / jnp.maximum(l_bol_unnorm, 1e-100)

    total_phot = outer_phot + warm_phot + hot_phot
    return total_phot * scale
