# SPDX-License-Identifier: BSD-3-Clause
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

from tengri.components.agn._params import (
    DEFAULT_AGN_COS_INC,
    DEFAULT_AGN_LOG_MBH,
    DEFAULT_AGN_LUM_RATIO,
)
from tengri.components.agn._phys import (
    C_LIGHT as _C_LIGHT,
    H_PLANCK as _H_PLANCK,
    K_BOLTZ as _K_BOLTZ,
    ring_area as _ring_area,
)
from tengri.utils.physics_constants import KEV_TO_ERG as _KEV_TO_ERG, L_SUN

# numpy >= 2.0 uses trapezoid; older versions used trapz
_np_trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


# ───────────────────────────────────────────────────────────────────
# Data classes
# ───────────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class KDPreintegratedData:
    """Preintegrated K&D 2018 disc components for photometric fast-path.

    All tables store filter-integrated spectral shapes per grid point.
    At runtime, ring contributions are looked up and summed: no
    wavelength-level computation needed.

    Attributes
    ----------
    planck_table : jnp.ndarray
        Shape (n_T, n_filters). Filter-integrated Planck B_nu(T) [erg/s/cm^3]
        for each temperature grid point and filter.
    planck_T_grid : jnp.ndarray
        Shape (n_T,). Temperature grid [K] (log-spaced).
    nthcomp_table : jnp.ndarray or None
        Shape (n_gamma, n_kTe, n_kTbb, n_filters). Filter-integrated
        nthcomp spectral shape [erg/s/cm^3]. None if templates not available.
    nthcomp_gamma_grid : jnp.ndarray or None
        Shape (n_gamma,). Photon index grid for warm Comptonization.
    nthcomp_kTe_grid : jnp.ndarray or None
        Shape (n_kTe,). Electron temperature grid [keV] for warm zone.
    nthcomp_kTbb_grid : jnp.ndarray or None
        Shape (n_kTbb,). Seed blackbody temperature grid [keV].
    corona_table : jnp.ndarray
        Shape (n_Gamma, n_kT, n_kTbb, n_filters). Filter-integrated
        thermal-Comptonization shape [erg/s/cm^3] for the hot corona, with
        both the electron-temperature cutoff and the seed-photon rollover.
    corona_Gamma_grid : jnp.ndarray
        Shape (n_Gamma,). Hard X-ray photon index grid [dimensionless].
    corona_kT_grid : jnp.ndarray
        Shape (n_kT,). Hot corona electron-temperature grid [keV].
    corona_kTbb_grid : jnp.ndarray
        Shape (n_kTbb,). Seed-photon temperature grid [keV] for the
        low-energy rollover (K&D 2018, Section 2.2).
    effective_bandwidths_hz : jnp.ndarray
        Shape (n_filters,). Effective frequency bandwidths [Hz] for L_bol
        estimation via sum(f_nu * bw).
    n_filters : int
        Number of photometric filters [dimensionless].

    Notes
    -----
    This is an immutable dataclass (``frozen=True``) designed for efficient
    lookup during K&D AGN photometric inference. All arrays are JAX arrays
    and thus compatible with JIT compilation and autodiff.

    **Precomputation cost**: One-time at model initialization. The cost
    scales as O(n_T × n_filters) for Planck, O(n_gamma × n_kTe × n_kTbb
    × n_filters) for nthcomp, and O(n_Gamma × n_kT × n_filters) for corona.
    Typical values: n_T=200, n_filters=10–50, n_gamma≈20, n_kTe≈20, n_kTbb≈20.

    **Filter dimensions**: All filter-indexed arrays are ordered by filter
    index, not by wavelength or frequency. At runtime, these are indexed
    directly by filter ID, enabling fast lookup without spectral integration.
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
    corona_kTbb_grid: jnp.ndarray
    effective_bandwidths_hz: jnp.ndarray
    n_filters: int


# K&D uses a custom dataclass (KDPreintegratedData) with three non-uniform tables
# (Planck per-T, nthcomp per-(gamma,kTe,kTbb), corona per-(Gamma,kT_hot)). These
# correspond to internal K&D physics parameters, not user-facing priors. Auto-
# collapse on user-Fixed parameters is not yet wired for K&D: tracked in
# docs/dev/optimization-architecture.md. The empty AXIS_PARAMS signals this.
AXIS_PARAMS: tuple[str, ...] = ()


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
        from tengri._data_setup import find_data

        # Honors $TENGRI_DATA_DIR (#1431); None when the grid is absent, which
        # the caller already treats as "templates unavailable".
        found = find_data("nthcomp_templates.h5")
        if found is None:
            return None, None, None, None
        template_path = found

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
    except (OSError, KeyError, ValueError):
        # OSError: file read error
        # KeyError: dataset missing from HDF5 file
        # ValueError: dtype conversion failed
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
        from tengri.utils.grid_interp import _vectorized_interp

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
    kTbb_seed_grid_keV: np.ndarray,
    filter_waves: list[np.ndarray],
    filter_trans: list[np.ndarray],
    redshift: float,
) -> np.ndarray:
    r"""Precompute the thermal-Comptonization corona shape integrated through filters.

    The corona spectrum is bounded by a high-energy cutoff at the electron
    temperature and a low-energy rollover at the seed-photon energy:

    .. math::

        {\rm shape}(\nu) = \nu^{\,1-\Gamma}
                           \, \exp(-h\nu / kT_e)
                           \, \exp(-\nu_{\rm seed} / \nu)

    normalized so :math:`\int {\rm shape}\, d\nu = 1` on the fixed RELAGN grid.
    This matches ``_hot_corona_lnu`` in ``disc.py`` term for term (including the
    seed-photon rollover, Kubota & Done 2018 Section 2.2), keeping the
    preintegrated photometry consistent with the full-wavelength path.

    The normalization uses the same fixed [1e-4, 1e4] keV grid as
    ``_hot_corona_lnu``, making the result independent of the caller's grid.

    Parameters
    ----------
    Gamma_grid : ndarray, shape (n_Gamma,)
        Hard X-ray photon index grid.
    kT_grid_keV : ndarray, shape (n_kT,)
        Hot corona electron-temperature grid [keV].
    kTbb_seed_grid_keV : ndarray, shape (n_kTbb,)
        Seed-photon temperature grid [keV] setting the low-energy rollover
        frequency :math:`\nu_{\rm seed} = kT_{\rm seed} / h`.
    filter_waves : list[ndarray]
        Wavelength grid per filter [Angstrom], observed frame.
    filter_trans : list[ndarray]
        Transmission per filter.
    redshift : float
        Source redshift.

    Returns
    -------
    ndarray, shape (n_Gamma, n_kT, n_kTbb, n_filters)
        Filter-integrated normalized corona shape.
    """
    n_Gamma = len(Gamma_grid)
    n_kT = len(kT_grid_keV)
    n_kTbb = len(kTbb_seed_grid_keV)
    n_filters = len(filter_waves)
    table = np.zeros((n_Gamma, n_kT, n_kTbb, n_filters), dtype=np.float64)

    # Fixed normalization grid matching _hot_corona_lnu and RELAGN:
    # [1e-4, 1e4] keV = [2.418e13, 2.418e21] Hz, 2000 log-spaced points.
    nu_wide = np.geomspace(2.418e13, 2.418e21, 2000)

    # Precompute per-filter rest-frame frequencies and bandpass denominators.
    filt_cache = []
    for fw, ft in zip(filter_waves, filter_trans):
        fw_np = np.asarray(fw, dtype=np.float64)
        ft_np = np.asarray(ft, dtype=np.float64)
        wave_rest = fw_np / (1.0 + redshift)
        nu_filter = _C_LIGHT / (wave_rest * 1e-8)
        denom = _np_trapezoid(ft_np * fw_np, fw_np)
        sort_idx = np.argsort(nu_filter)
        unsort_idx = np.argsort(sort_idx)
        filt_cache.append((fw_np, ft_np, nu_filter[sort_idx], unsort_idx, denom))

    for i_G, Gamma in enumerate(Gamma_grid):
        for i_T, kT_keV in enumerate(kT_grid_keV):
            kT_erg = kT_keV * _KEV_TO_ERG
            x = np.clip(_H_PLANCK * nu_wide / kT_erg, 0.0, 500.0)
            base = nu_wide ** (1.0 - Gamma) * np.exp(-x)
            for i_S, kTbb_keV in enumerate(kTbb_seed_grid_keV):
                # Low-energy seed-photon rollover at nu_seed = kT_seed / h.
                nu_seed = kTbb_keV * _KEV_TO_ERG / _H_PLANCK
                seed_roll = np.exp(-np.clip(nu_seed / nu_wide, 0.0, 700.0))
                shape = base * seed_roll

                integral = _np_trapezoid(shape, nu_wide)
                if integral <= 0:
                    continue
                shape_normed = shape / integral

                for f_idx, (fw_np, ft_np, nu_sorted, unsort_idx, denom) in enumerate(filt_cache):
                    if denom <= 0:
                        continue
                    shape_on_filt = np.interp(
                        nu_sorted, nu_wide, shape_normed, left=0.0, right=0.0
                    )[unsort_idx]
                    num = _np_trapezoid(shape_on_filt * ft_np * fw_np, fw_np)
                    table[i_G, i_T, i_S, f_idx] = num / denom

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
    n_kTbb_corona: int = 16,
) -> KDPreintegratedData:
    """Precompute all K&D disc filter tables at model init time.

    Builds lookup tables for the three K&D zones by integrating their
    spectral shapes through the photometric filters. This is called
    once at ``SEDModel.__init__`` when K&D AGN is enabled with fixed
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

    Notes
    -----
    **JIT-compatible**: no, this function performs build-time integration
    via NumPy. The returned tables are JIT-compatible and can be used in
    traced functions.

    This function is called once at ``SEDModel.__init__`` when K&D AGN
    is enabled with photometric filters. The precomputed tables avoid
    redundant filter integration during inference, providing ~40× speedup
    for the K&D component by reducing per-ring computation from ~17,000
    wavelength points to a handful of filter indices.

    **Temperature grid**: The Planck table uses a logarithmic temperature
    grid spanning [T_min, T_max] to capture the optically thick disc
    spectrum across all AGN luminosities and black hole masses. Typical
    outer-disc temperatures range from ~1000 K (cool outer regions) to
    ~3×10^7 K (hot inner regions).

    **nthcomp availability**: If precomputed nthcomp templates
    (data/nthcomp_templates.h5) are unavailable, a simplified modified-
    blackbody proxy is used at runtime (see disc.py:1083-1094). This
    has ~5–10% shape error but allows offline computation without the
    external RELAGN dependency.
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
    # Seed-photon temperature axis: kT_NT(R_hot)*exp(y_warm) spans roughly
    # 5e-4 to 0.1 keV (NIR to EUV) across the M_BH / Eddington / Gamma_warm space.
    kTbb_seed_grid_keV = np.geomspace(5.0e-4, 0.12, n_kTbb_corona)
    corona_table = _build_corona_filter_table(
        Gamma_grid,
        kT_grid_keV,
        kTbb_seed_grid_keV,
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
        corona_kTbb_grid=jnp.asarray(kTbb_seed_grid_keV),
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
    (n_filters,): filter-integrated B_nu at temperature T.
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
    (n_filters,): filter-integrated nthcomp shape.
    """

    def _interp_axis(val, grid):
        """Compute clamped linear interpolation indices and fractions on grid."""
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
        """Index into nthcomp filter table at trilinear corner offset."""
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
    kTbb_seed_keV: jnp.ndarray,
    Gamma_grid: jnp.ndarray,
    kT_grid: jnp.ndarray,
    kTbb_grid: jnp.ndarray,
    corona_table: jnp.ndarray,
) -> jnp.ndarray:
    """Look up filter-integrated corona shape via trilinear interpolation.

    Parameters
    ----------
    Gamma : scalar
        Hard X-ray photon index.
    kT_keV : scalar
        Hot corona electron temperature [keV].
    kTbb_seed_keV : scalar
        Seed-photon temperature [keV] (sets the low-energy rollover).
    Gamma_grid, kT_grid, kTbb_grid : 1D arrays
        Grid axes.
    corona_table : (n_Gamma, n_kT, n_kTbb, n_filters)
        Precomputed table.

    Returns
    -------
    (n_filters,): filter-integrated normalized corona shape.
    """

    def _interp_axis(val, grid):
        """Compute clamped linear interpolation indices and fractions on grid."""
        n = grid.shape[0]
        idx_hi = jnp.searchsorted(grid, val, side="right")
        idx_lo = jnp.clip(idx_hi - 1, 0, n - 2)
        idx_hi_c = jnp.clip(idx_hi, 1, n - 1)
        span = grid[idx_hi_c] - grid[idx_lo]
        frac = jnp.where(span > 0, (val - grid[idx_lo]) / span, 0.0)
        return idx_lo, jnp.clip(frac, 0.0, 1.0)

    iG, fG = _interp_axis(Gamma, Gamma_grid)
    iT, fT = _interp_axis(kT_keV, kT_grid)
    iS, fS = _interp_axis(kTbb_seed_keV, kTbb_grid)

    def _at(dS):
        """Bilinear (Gamma, kT) interpolation at seed-axis offset dS."""
        c00 = corona_table[iG, iT, iS + dS]
        c10 = corona_table[iG + 1, iT, iS + dS]
        c01 = corona_table[iG, iT + 1, iS + dS]
        c11 = corona_table[iG + 1, iT + 1, iS + dS]
        s0 = c00 * (1 - fG) + c10 * fG
        s1 = c01 * (1 - fG) + c11 * fG
        return s0 * (1 - fT) + s1 * fT

    return _at(0) * (1 - fS) + _at(1) * fS


# ───────────────────────────────────────────────────────────────────
# Private helpers for kubota_done_disc_preintegrated
# ───────────────────────────────────────────────────────────────────


def _compute_bh_and_radii(
    agn_log_mbh: float,
    agn_a_spin: float,
    agn_log_lbol: float,
    agn_f_hard: float,
    agn_r_warm_ratio: float,
    l_edd: jnp.ndarray,
    r_isco_cm: jnp.ndarray,
    _G_GRAV: float,
    _MSUN_G: float,
    _SIGMA_SB: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, float]:
    """Compute black hole parameters and zone radii for K&D disc.

    Computes accretion rate, inner disc temperature, and zone boundaries
    (hot, warm, outer). Returns all quantities needed for radial integration
    in subsequent zones.

    Parameters
    ----------
    agn_log_mbh : float
        log10(M_bh / Msun).
    agn_a_spin : float
        Dimensionless spin parameter [0, 1].
    agn_log_lbol : float
        Bolometric luminosity log10(L_bol / L_sun). The Eddington ratio is
        derived from it (#846).
    agn_f_hard : float
        Fractional hard X-ray luminosity [0, 0.5].
    agn_r_warm_ratio : float
        Warm-to-hot radius ratio [1.1, 10].
    l_edd : ndarray
        Eddington luminosity [erg/s].
    r_isco_cm : ndarray
        ISCO radius [cm].
    _G_GRAV : float
        Gravitational constant [cm^3 g^-1 s^-2].
    _MSUN_G : float
        Solar mass [g].
    _SIGMA_SB : float
        Stefan-Boltzmann constant [erg cm^-2 K^-4 s^-1].

    Returns
    -------
    tuple
        (r_hot_cm, r_warm_cm, r_out_cm, t_in, eta): hot zone radius, warm
        zone radius, outer disc radius, inner disc temperature, accretion
        efficiency [all in CGS units].
    """
    from tengri.components.agn.disc import (
        _gravitational_radius,
        _isco_radius,
        _r_hot_bisect,
        _self_gravity_radius,
    )

    r_g = _gravitational_radius(agn_log_mbh)
    r_isco_rg = _isco_radius(agn_a_spin)

    eta = 1.0 - jnp.sqrt(1.0 - 2.0 / (3.0 * r_isco_rg))
    # E fix (#846): derive the Eddington ratio from L_bol (mirrors the runtime
    # kubota_done_disc; keeps this preintegration path bit-consistent with it).
    l_bol_erg = 10.0**agn_log_lbol * L_SUN
    l_edd_ratio = jnp.clip(l_bol_erg / l_edd, 1e-10, 1.0)
    mdot = l_bol_erg / (eta * _C_LIGHT**2)

    t_in = (
        3.0
        * _G_GRAV
        * 10.0**agn_log_mbh
        * _MSUN_G
        * mdot
        / (8.0 * jnp.pi * _SIGMA_SB * r_isco_cm**3)
    ) ** 0.25

    # Zone radii
    f_hard_safe = jnp.clip(agn_f_hard, 1e-6, 0.5)
    l_hot_target = f_hard_safe * l_edd
    r_hot_cm = _r_hot_bisect(r_isco_cm, t_in, l_hot_target)

    r_warm_ratio_safe = jnp.clip(agn_r_warm_ratio, 1.1, 10.0)
    r_warm_cm = r_hot_cm * r_warm_ratio_safe

    r_sg_rg = _self_gravity_radius(agn_log_mbh, l_edd_ratio)
    r_out_cm = jnp.maximum(r_sg_rg, r_isco_rg * 10.0) * r_g

    r_hot_cm = jnp.clip(r_hot_cm, r_isco_cm * 1.01, r_out_cm * 0.5)
    r_warm_cm = jnp.clip(r_warm_cm, r_hot_cm * 1.01, r_out_cm * 0.9)

    return r_hot_cm, r_warm_cm, r_out_cm, t_in, eta


def _integrate_outer_zone(
    r_warm_cm: jnp.ndarray,
    r_out_cm: jnp.ndarray,
    r_isco_cm: jnp.ndarray,
    t_in: jnp.ndarray,
    n_radii: int,
    kd_data: KDPreintegratedData,
    agn_cos_inc: float,
    _SIGMA_SB: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Integrate outer standard disc via filter-level Planck lookup.

    Computes photometric and bolometric contributions from the optically
    thick outer disc (temperature-integrated, from r_warm to r_out).

    Parameters
    ----------
    r_warm_cm : ndarray, scalar
        Warm-zone boundary radius [cm].
    r_out_cm : ndarray, scalar
        Outer disc boundary radius [cm].
    r_isco_cm : ndarray, scalar
        ISCO radius [cm].
    t_in : ndarray, scalar
        Inner disc temperature [K].
    n_radii : int
        Number of radial rings [dimensionless].
    kd_data : KDPreintegratedData
        Precomputed Planck filter table.
    agn_cos_inc : float
        cos(inclination angle).
    _SIGMA_SB : float
        Stefan-Boltzmann constant [erg cm^-2 K^-4 s^-1].

    Returns
    -------
    tuple
        (outer_phot, outer_bol): filter-integrated photometry (n_filters,)
        and bolometric luminosity [erg/s] [scalar].
    """
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
        """Compute filter-integrated Planck photometry for outer-disc annulus."""
        b_filt = _lookup_planck_filter(t_ring, kd_data.planck_T_grid, kd_data.planck_table)
        return b_filt * _ring_area(r_cm, dr_ring, agn_cos_inc)

    outer_phot = jnp.sum(
        jax.vmap(_outer_ring_phot)(r_outer, t_outer, dr_outer), axis=0
    )  # (n_filters,)

    outer_bol = jnp.sum(
        _SIGMA_SB * t_outer**4 * 2.0 * jnp.pi * r_outer * dr_outer * jnp.maximum(agn_cos_inc, 0.01)
    )

    return outer_phot, outer_bol


def _integrate_warm_zone(
    r_hot_cm: jnp.ndarray,
    r_warm_cm: jnp.ndarray,
    r_isco_cm: jnp.ndarray,
    t_in: jnp.ndarray,
    n_radii: int,
    kd_data: KDPreintegratedData,
    agn_cos_inc: float,
    agn_gamma_warm: float,
    agn_kt_warm: float,
    _SIGMA_SB: float,
    _K_BOLTZ_KEV: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Integrate warm Comptonization zone via filter-level nthcomp/Planck lookup.

    Computes photometric and bolometric contributions from the warm
    Comptonization zone. Uses nthcomp templates if available; falls back
    to Planck for unavailable templates.

    Parameters
    ----------
    r_hot_cm : ndarray, scalar
        Hot-zone boundary radius [cm].
    r_warm_cm : ndarray, scalar
        Warm-zone boundary radius [cm].
    r_isco_cm : ndarray, scalar
        ISCO radius [cm].
    t_in : ndarray, scalar
        Inner disc temperature [K].
    n_radii : int
        Number of radial rings [dimensionless].
    kd_data : KDPreintegratedData
        Precomputed nthcomp and Planck filter tables.
    agn_cos_inc : float
        cos(inclination angle).
    agn_gamma_warm : float
        Photon index for warm Comptonization [dimensionless].
    agn_kt_warm : float
        Electron temperature for warm zone [keV].
    _SIGMA_SB : float
        Stefan-Boltzmann constant [erg cm^-2 K^-4 s^-1].
    _K_BOLTZ_KEV : float
        Boltzmann constant [keV/K].

    Returns
    -------
    tuple
        (warm_phot, warm_bol): filter-integrated photometry (n_filters,)
        and bolometric luminosity [erg/s] [scalar].
    """
    log_r_hot = jnp.log10(r_hot_cm)
    log_r_warm_grid = jnp.linspace(log_r_hot, jnp.log10(r_warm_cm), n_radii)
    r_warm_grid = 10.0**log_r_warm_grid

    r_ratio_warm = r_warm_grid / r_isco_cm
    torque_warm = jnp.maximum(1.0 - jnp.sqrt(1.0 / r_ratio_warm), 1e-30) ** 0.25
    t_warm = t_in * r_ratio_warm ** (-0.75) * torque_warm

    d_log_r_warm = log_r_warm_grid[1] - log_r_warm_grid[0]
    dr_warm = r_warm_grid * jnp.log(10.0) * d_log_r_warm

    has_nthcomp = kd_data.nthcomp_table is not None

    if has_nthcomp:

        def _warm_ring_phot(r_cm, t_ring, dr_ring):
            """Compute per-filter flux for one warm-zone annulus via nthcomp lookup."""
            l_total = _SIGMA_SB * t_ring**4 / jnp.pi * _ring_area(r_cm, dr_ring, agn_cos_inc)

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

        def _warm_ring_phot(r_cm, t_ring, dr_ring):
            """Compute per-filter flux for one warm-zone annulus via Planck fallback."""
            b_filt = _lookup_planck_filter(t_ring, kd_data.planck_T_grid, kd_data.planck_table)
            return b_filt * _ring_area(r_cm, dr_ring, agn_cos_inc)

    warm_phot = jnp.sum(
        jax.vmap(_warm_ring_phot)(r_warm_grid, t_warm, dr_warm), axis=0
    )  # (n_filters,)

    warm_bol = jnp.sum(
        _SIGMA_SB
        * t_warm**4
        * 2.0
        * jnp.pi
        * r_warm_grid
        * dr_warm
        * jnp.maximum(agn_cos_inc, 0.01)
    )

    return warm_phot, warm_bol


def kubota_done_disc_preintegrated(
    kd_data: KDPreintegratedData,
    agn_log_lbol: float,
    agn_lum_ratio: float = DEFAULT_AGN_LUM_RATIO,
    agn_log_mbh: float = DEFAULT_AGN_LOG_MBH,
    agn_log_ledd: float = -1.0,
    agn_a_spin: float = 0.0,
    agn_cos_inc: float = DEFAULT_AGN_COS_INC,
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
    """Preintegrated K&D (2018) 3-zone disc: filter-level radial integration.

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

    Notes
    -----
    **JIT-compatible**: yes, uses ``jnp`` primitives and ``jax.vmap``.

    This is the photometric fast-path variant of ``kubota_done_disc()``. It
    replaces wavelength-level integration with filter-level lookup tables,
    providing ~40× speedup at the cost of no direct wavelength access.

    **Physics equivalence**: Zone radii, temperature profiles, radiative
    efficiency, and normalization are identical to the spectroscopic path.
    The only difference is the integration domain: filters instead of
    wavelengths. Bolometric luminosity is still computed analytically from
    radial integration (σT^4 × dA) to remain grid-independent.

    **Precomputed tables**: Requires ``kd_data`` from
    ``preintegrate_kd_components()``, built once at model initialization.
    If nthcomp templates are unavailable, falls back to Planck lookup
    (i.e., no Comptonization). See disc.py:1083-1094 for details.

    References
    ----------
    .. [1] A. Kubota and C. Done, "A physical model of the broad-band continuum
       of AGN and its implications for the UV/X relation and optical variability,"
       MNRAS, 480, 1247 (2018). arXiv:1804.00171.
       https://doi.org/10.1093/mnras/sty1890
    """
    from tengri.components.agn.disc import (
        _eddington_luminosity,
        _gravitational_radius,
        _isco_radius,
        _l_seed_geometric,
        beloborodov_gamma_hot,
    )
    from tengri.utils.physics_constants import (
        G_GRAV as _G_GRAV,
        K_BOLTZ_KEV as _K_BOLTZ_KEV,
        M_SUN as _MSUN_G,
        SIGMA_SB as _SIGMA_SB,
    )

    # --- Black hole parameters and zone radii ---
    r_g = _gravitational_radius(agn_log_mbh)
    r_isco_rg = _isco_radius(agn_a_spin)
    r_isco_cm = r_isco_rg * r_g
    l_edd = _eddington_luminosity(agn_log_mbh)
    # E fix (#846): L_bol is the knob; Eddington ratio derived (see runtime path).
    l_bol_erg = 10.0**agn_log_lbol * L_SUN

    r_hot_cm, r_warm_cm, r_out_cm, t_in, _eta = _compute_bh_and_radii(
        agn_log_mbh,
        agn_a_spin,
        agn_log_lbol,
        agn_f_hard,
        agn_r_warm_ratio,
        l_edd,
        r_isco_cm,
        _G_GRAV,
        _MSUN_G,
        _SIGMA_SB,
    )

    # ── Zone 1: Outer standard disc ──
    outer_phot, outer_bol = _integrate_outer_zone(
        r_warm_cm,
        r_out_cm,
        r_isco_cm,
        t_in,
        n_radii,
        kd_data,
        agn_cos_inc,
        _SIGMA_SB,
    )

    # ── Zone 2: Warm Comptonization ──
    warm_phot, warm_bol = _integrate_warm_zone(
        r_hot_cm,
        r_warm_cm,
        r_isco_cm,
        t_in,
        n_radii,
        kd_data,
        agn_cos_inc,
        agn_gamma_warm,
        agn_kt_warm,
        _SIGMA_SB,
        _K_BOLTZ_KEV,
    )

    # ── Zone 3: Hot corona ──
    f_hard_safe = jnp.clip(agn_f_hard, 1e-6, 0.5)
    l_hot_erg = jnp.minimum(f_hard_safe * l_edd, l_bol_erg * 0.5)

    # Self-consistent Gamma (same as full-wavelength path)
    l_seed_geom = _l_seed_geometric(r_isco_cm, r_hot_cm, r_out_cm, t_in)
    gamma_hard_sc = beloborodov_gamma_hot(l_hot_erg, l_seed_geom)
    gamma_hard_eff = jnp.where(agn_self_consistent_gamma, gamma_hard_sc, agn_gamma_hard)

    # Hot-flow seed-photon temperature (K&D 2018, Section 2.2), identical to the
    # full-wavelength path: kT_seed = k T_NT(R_hot) * exp(y_warm), with y_warm
    # recovered from Gamma_warm via Gamma = sqrt(9/4 + 4/y) - 1/2.
    r_ratio_hot = r_hot_cm / r_isco_cm
    torque_hot = jnp.maximum(1.0 - jnp.sqrt(1.0 / r_ratio_hot), 1e-30) ** 0.25
    t_nt_rhot = t_in * r_ratio_hot ** (-0.75) * torque_hot
    y_warm_denom = jnp.maximum((agn_gamma_warm + 0.5) ** 2 - 2.25, 1e-3)
    y_warm = jnp.clip(4.0 / y_warm_denom, 0.0, 10.0)
    kT_seed_keV = _K_BOLTZ_KEV * t_nt_rhot * jnp.exp(y_warm)

    corona_filt = _lookup_corona_filter(
        gamma_hard_eff,
        agn_kt_hot,
        kT_seed_keV,
        kd_data.corona_Gamma_grid,
        kd_data.corona_kT_grid,
        kd_data.corona_kTbb_grid,
        kd_data.corona_table,
    )
    hot_phot = l_hot_erg * corona_filt  # (n_filters,)

    # ── Combine and normalize to L_bol * agn_lum_ratio ─────────────────
    # The full-wavelength code normalizes: scale = L_bol_requested / int[L_nu dnu].
    # We can't compute int[L_nu dnu] from filter photometry alone (filters miss
    # most of the bolometric SED for UV/X-ray-bright AGN). Instead, compute the
    # bolometric from the Stefan-Boltzmann integral of each zone's radial grid:
    #   L_bol = sum_rings(sigma * T^4 * dA) for disc zones + L_hot for corona.
    # This equals the spectral integral by energy conservation (same as the
    # full-wavelength code's trapezoid integral, to numerical precision).
    l_bol_unnorm = outer_bol + warm_bol + l_hot_erg
    l_bol_requested = 10.0**agn_log_lbol * L_SUN * agn_lum_ratio
    scale = l_bol_requested / jnp.maximum(l_bol_unnorm, 1e-100)

    total_phot = outer_phot + warm_phot + hot_phot
    return total_phot * scale


# ───────────────────────────────────────────────────────────────────
# Protocol-shaped entry points (new in restructure)
# ───────────────────────────────────────────────────────────────────


def precompute(filter_waves: list, filter_trans: list, redshift: float, parameters=None, **kwargs):
    """Protocol-shaped entry point for K&D 3-zone disc precompute.

    Delegates to :func:`preintegrate_kd_components`, which builds Planck,
    nthcomp, and corona filter lookup tables. The K&D grid axes are internal
    physics coordinates (temperature, photon index, etc.), not user-facing
    priors, so auto-collapse-on-Fixed is a no-op (``AXIS_PARAMS = ()``).

    Parameters
    ----------
    filter_waves : list[ndarray]
        Wavelength grid per filter [Angstrom], observed frame.
    filter_trans : list[ndarray]
        Transmission per filter (0–1).
    redshift : float
        Source redshift (fixed at init time). [dimensionless]
    parameters : Parameters | None, optional
        Unused: K&D grid axes are internal physics coords, not user priors.
    **kwargs
        Forwarded to :func:`preintegrate_kd_components` (e.g. ``n_T``,
        ``T_min``, ``T_max``, ``n_Gamma``, ``n_kT_corona``).

    Returns
    -------
    KDPreintegratedData
        All precomputed filter-integrated tables for the 3-zone K&D disc.

    References
    ----------
    .. [1] A. Kubota and C. Done, "A physical model of the broad-band continuum
           of AGN and its implications for the UV/X relation and optical variability,"
           MNRAS, 480, 1247 (2018). arXiv:1804.00171.
           https://doi.org/10.1093/mnras/sty1890

    Notes
    -----
    **JIT-compatible**: no, this is a build-time function using NumPy.
    The returned tables are JIT-compatible.
    """
    return preintegrate_kd_components(filter_waves, filter_trans, redshift, **kwargs)


def build_lookup(preint, **kwargs):
    """K&D runtime uses the preintegrated dataclass directly via fused kernels.

    Parameters
    ----------
    preint : KDPreintegratedData
        Preintegrated K&D data.
    **kwargs
        Ignored; accepted for Protocol consistency.

    Returns
    -------
    None
        K&D runtime lookup is performed directly in the fused kernels.

    Notes
    -----
    **JIT-compatible**: not applicable: K&D integration happens at
    model initialization time, not at inference time.
    """
    return None
