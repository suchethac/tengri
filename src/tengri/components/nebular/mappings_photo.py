# SPDX-License-Identifier: BSD-3-Clause
"""MAPPINGS V photoionization grid backends (Flury et al. 2024).

Provides stellar (Starburst99, BPASS) and AGN (OPTXAGNF) backends using
MAPPINGS V v5.2.1 grids from Flury et al. 2024 (arXiv:2412.06763),
available at Zenodo 10.5281/zenodo.14140949.

Grid characteristics
--------------------

- Nicholls+2017 empirical stellar abundance patterns (ζ_O = 0.05–2)
- Jenkins+2009/2014 empirical dust depletion (F★ = 0.43)
- CHIANTI v10 atomic data
- Two density structures: isobaric (cpr) and isochoric (cdn)

Physical pipeline (stellar)
---------------------------
1. SSP spectrum → Q_H (ionizing photon rate, photons/s/Msun)
2. For each young age bin i with weight w_i:
   a. Interpolate logHB_per_logq(ζ_O, log_age, logU, logn) from grid
   b. Interpolate line_ratio(ζ_O, log_age, logU, logn, line) from grid
   c. L_line_i = ratio × 10^{logHB_per_logq} × Q_H_i × w_i  [erg/s]
3. Sum over age bins, convert to Lsun

Physical pipeline (AGN)
-----------------------
As above but grid axes are (ζ_O, log_MBH, log_Edd, logU, logn).
Q_H is supplied by the caller (e.g. from an AGN disc model).

Metallicity convention
----------------------
The Flury grids use ζ_O (solar-relative oxygen abundance, column "z").
Internally we store ζ_O and interpolate in ζ_O space (not absolute log Z)
because the grid is uniform in ζ_O, making interpolation exact. The
high-level API accepts neb_logZ_gas (absolute log10 Z) and converts on-the-fly.

Build the HDF5 grid file once:
    python scripts/build_flury2024_grids.py

References
----------

- Flury et al. 2024, arXiv:2412.06763
- Sutherland & Dopita 2017 (MAPPINGS V)
- Nicholls et al. 2017 (empirical abundance scaling)
- Jenkins 2009, 2014 (dust depletion)

"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import NamedTuple

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from tengri._data_setup import package_or_env_data_path
from tengri.components.nebular._constants import _LOG10_ZSUN, _LSUN_ERG
from tengri.components.nebular._shared import (
    _interp_index_weight,
    _qh_bilinear,
    compute_qh,
    render_nebular_lines,
    sanitize_qh_table,
)
from tengri.utils.grid_interp import (
    PreintegratedGrid,
    PreintegratedLines,
    preintegrate_lines,
    slice_fixed_axes,
)
from tengri.utils.interpolation import compute_grid_weights, edges_for_grid

# ── Ionizing spectrum warnings ────────────────────────────────────


class IonizingSpectrumInconsistencyError(Exception):
    """Raised when the nebular ionizing source is inconsistent with the DSPS SSPs.

    The stellar continuum and the nebular line predictions would be driven by
    different stellar population models. Suppress with
    ``ionizing_source_warning='warn'`` or ``'suppress'`` if you have verified
    this is acceptable for your science case.
    """


class IonizingSpectrumInconsistencyWarning(UserWarning):
    """Warning variant of IonizingSpectrumInconsistencyError."""


def _emit_mappings_agn_ionizing_warning(mode: str) -> None:
    """Emit warning about MappingsPhotoAGNBackend Q_H source requirement."""
    msg = (
        "MappingsPhotoAGNBackend: Q_H must be supplied by the AGN disc model: "
        "it is not self-consistently derived from SSPs. Ensure you are passing "
        "log_l_ion_erg from an AGN disc model (e.g. kubota_done_full). "
        "The ionizing shape is a power law; for a composite starburst+AGN region, "
        "use a mixed-source model. "
        "To suppress: pass ionizing_source_warning='suppress'."
    )
    if mode == "raise":
        raise IonizingSpectrumInconsistencyError(msg)
    warnings.warn(msg, IonizingSpectrumInconsistencyWarning, stacklevel=3)


def _emit_mappings_stellar_ionizing_warning(model: str, mode: str) -> None:
    """Emit error/warning about MappingsPhotoStellarBackend ionizing source mismatch."""
    msg = (
        f"MappingsPhotoStellarBackend (model='{model}'): the ionizing radiation field "
        "used to compute nebular line predictions is from a "
        f"{'Starburst99' if model == 'sb99' else 'BPASS v2.2'} grid embedded in "
        "MAPPINGS V: this is NOT derived from your DSPS SSPs. The stellar continuum "
        "and the nebular lines are driven by DIFFERENT stellar population models. "
        "This inconsistency can bias predicted line ratios at ages < 20 Myr and for "
        "non-solar metallicity. For self-consistent nebular emission, use "
        "CloudyGridBackend or CueBackend instead. "
        "To suppress: pass ionizing_source_warning='warn' or 'suppress'."
    )
    if mode == "raise":
        raise IonizingSpectrumInconsistencyError(msg)
    if mode == "warn":
        warnings.warn(msg, IonizingSpectrumInconsistencyWarning, stacklevel=3)
    # mode == "suppress": silent


# Age cut for nebular emission (same as CloudyGridBackend)
_MAX_NEB_LOG_AGE_YR = 8.0  # log10(100 Myr in yr)

_DEFAULT_GRID_PATH = package_or_env_data_path("flury2024_grids.h5")


# ── NamedTuples for pre-loaded grid data ──────────────────────────


class MappingsStellarGridData(NamedTuple):
    """Pre-loaded stellar (sb99/bpass) MAPPINGS V grid.

    All arrays are JAX arrays for JIT-compatibility. Metallicity axis is ζ_O
    (solar-relative), interpolated continuously. SFH axis is discrete (inst
    or cont) and selected by index at runtime.
    """

    # Line wavelengths (vacuum Angstrom)
    line_wavelengths: jnp.ndarray  # (N_lines,)

    # Grid axes
    zo_axis: jnp.ndarray  # (N_z,)   ζ_O solar-relative metallicity
    logU_axis: jnp.ndarray  # (N_u,)   log10(U)
    log_age_yr_axis: jnp.ndarray  # (N_a,)   log10(age/yr), inst ages
    logn_axis: jnp.ndarray  # (N_n,)   log10(n_H / cm^-3)

    # SFH axis (discrete string labels; matched by index)
    sfh_labels: list  # (N_s,)  e.g. ["cont", "inst"]
    sfh_idx_inst: int  # index for "inst" SFH
    sfh_idx_cont: int  # index for "cont" SFH

    # Grid values: shape (N_z, N_a, N_s, N_u, N_n)
    logHB_per_logq: jnp.ndarray

    # Line flux ratios relative to Hβ: shape (N_z, N_a, N_s, N_u, N_n, N_lines)
    line_ratios: jnp.ndarray


class MappingsAGNGridData(NamedTuple):
    """Pre-loaded AGN (OPTXAGNF) MAPPINGS V grid.

    Axes: (ζ_O, log_MBH, log_Edd, logU, logn).
    """

    line_wavelengths: jnp.ndarray  # (N_lines,)

    zo_axis: jnp.ndarray  # (N_z,)
    logU_axis: jnp.ndarray  # (N_u,)
    logmbh_axis: jnp.ndarray  # (N_m,)  log10(M_BH / Msun)
    logedd_axis: jnp.ndarray  # (N_e,)  log10(L / L_Edd)
    logn_axis: jnp.ndarray  # (N_n,)

    logHB_per_lum: jnp.ndarray  # (N_z, N_m, N_e, N_u, N_n)  log10(L_Hβ/L_ion)
    line_ratios: jnp.ndarray  # (N_z, N_m, N_e, N_u, N_n, N_lines)


# ── HDF5 loaders ──────────────────────────────────────────────────


def _load_stellar_grid(filepath: str | Path, model: str, density: str) -> MappingsStellarGridData:
    """Load a stellar MAPPINGS V grid from the HDF5 file.

    Parameters
    ----------
    filepath : str or Path
        Path to flury2024_grids.h5 (built by build_flury2024_grids.py).
    model : str
        Stellar model: "sb99" or "bpass".
    density : str
        Density structure: "cpr" (isobaric) or "cdn" (isochoric).

    """
    with h5py.File(filepath, "r") as f:
        grp = f[f"{model}/{density}"]

        zo_axis = jnp.array(grp["z_axis"][:])  # ζ_O
        logU_axis = jnp.array(grp["logU_axis"][:])
        log_age_yr_axis = jnp.array(grp["log_age_yr_axis"][:])
        logn_axis = jnp.array(grp["logn_axis"][:])

        sfh_raw = grp["sfh_labels"][:]
        sfh_labels = [s.decode() if isinstance(s, bytes) else s for s in sfh_raw]

        # Find discrete SFH indices
        try:
            sfh_idx_inst = sfh_labels.index("inst")
        except ValueError:
            sfh_idx_inst = 0
        try:
            sfh_idx_cont = sfh_labels.index("cont")
        except ValueError:
            sfh_idx_cont = sfh_idx_inst

        logHB_per_logq = jnp.array(grp["logHB_per_logq"][:])  # (N_z,N_a,N_s,N_u,N_n)
        line_ratios = jnp.array(grp["line_ratios"][:])  # (N_z,N_a,N_s,N_u,N_n,N_lines)

        # Wavelengths stored in the top-level group per model/density sub-group
        line_wavelengths = jnp.array(grp["line_wavelengths_aa"][:])

    return MappingsStellarGridData(
        line_wavelengths=line_wavelengths,
        zo_axis=zo_axis,
        logU_axis=logU_axis,
        log_age_yr_axis=log_age_yr_axis,
        logn_axis=logn_axis,
        sfh_labels=sfh_labels,
        sfh_idx_inst=sfh_idx_inst,
        sfh_idx_cont=sfh_idx_cont,
        logHB_per_logq=logHB_per_logq,
        line_ratios=line_ratios,
    )


def _load_agn_grid(filepath: str | Path, density: str) -> MappingsAGNGridData:
    """Load the AGN OPTXAGNF MAPPINGS V grid.

    Parameters
    ----------
    filepath : str or Path
        Path to flury2024_grids.h5.
    density : str
        "cpr" or "cdn".

    """
    with h5py.File(filepath, "r") as f:
        grp = f[f"agn_oxaf/{density}"]

        line_wavelengths = jnp.array(grp["line_wavelengths_aa"][:])
        zo_axis = jnp.array(grp["z_axis"][:])
        logU_axis = jnp.array(grp["logU_axis"][:])
        logmbh_axis = jnp.array(grp["logmbh_axis"][:])
        logedd_axis = jnp.array(grp["logedd_axis"][:])
        logn_axis = jnp.array(grp["logn_axis"][:])
        logHB_per_lum = jnp.array(grp["logHB_per_lum"][:])  # (N_z,N_m,N_e,N_u,N_n)
        line_ratios = jnp.array(grp["line_ratios"][:])  # (N_z,N_m,N_e,N_u,N_n,N_lines)

    return MappingsAGNGridData(
        line_wavelengths=line_wavelengths,
        zo_axis=zo_axis,
        logU_axis=logU_axis,
        logmbh_axis=logmbh_axis,
        logedd_axis=logedd_axis,
        logn_axis=logn_axis,
        logHB_per_lum=logHB_per_lum,
        line_ratios=line_ratios,
    )


# ── Q_H computation (identical to cloudy_grid.py) ─────────────────

_compute_qh_grid = jax.vmap(jax.vmap(compute_qh, in_axes=(None, 0)), in_axes=(None, 0))


# ── Metallicity conversion helpers ────────────────────────────────


def _log_z_abs_to_zo(log_z_abs: float) -> float:
    """Convert absolute log10(Z) → ζ_O (solar-relative).

    ζ_O = Z / Z_sun = 10^(log10(Z) − log10(Z_sun))
    """
    return 10.0 ** (log_z_abs - _LOG10_ZSUN)


# ── Stellar grid interpolation: 4-D (ζ_O, log_age, logU, logn) + sfh slice


def _interp_stellar_grid(
    data: jnp.ndarray,
    grid: MappingsStellarGridData,
    zo_val: float,
    log_age_yr_val: float,
    logU_val: float,
    logn_val: float,
    sfh_idx: int,
) -> jnp.ndarray:
    """4-D interpolation over (ζ_O, log_age, logU, logn) for a fixed SFH slice.

    Parameters
    ----------
    data : (N_z, N_a, N_s, N_u, N_n, ...) array
    sfh_idx : int
        Index along the discrete sfh axis (0=cont, 1=inst, or as stored).

    """
    # Slice out the discrete SFH dimension → (N_z, N_a, N_u, N_n, ...)
    sliced = data[:, :, sfh_idx, :, :]

    iz, wz = _interp_index_weight(zo_val, grid.zo_axis)
    ia, wa = _interp_index_weight(log_age_yr_val, grid.log_age_yr_axis)
    iu, wu = _interp_index_weight(logU_val, grid.logU_axis)
    in_, wn = _interp_index_weight(logn_val, grid.logn_axis)

    def _get(iz_, ia_, iu_, in_):
        """Retrieve value from sliced 4D grid."""
        return sliced[iz_, ia_, iu_, in_]

    # 4-D linear interpolation (16 corners)
    def _lerp4(iz_, ia_, iu_, in_):
        """Perform 4D linear interpolation over logn, logU, log_age, and log_Z."""
        # Accumulate over n
        c0 = _get(iz_, ia_, iu_, in_) * (1 - wn) + _get(iz_, ia_, iu_, in_ + 1) * wn
        c1 = _get(iz_, ia_, iu_ + 1, in_) * (1 - wn) + _get(iz_, ia_, iu_ + 1, in_ + 1) * wn
        cu = c0 * (1 - wu) + c1 * wu
        return cu

    ca0 = _lerp4(iz, ia, iu, in_) * (1 - wa) + _lerp4(iz, ia + 1, iu, in_) * wa
    ca1 = _lerp4(iz + 1, ia, iu, in_) * (1 - wa) + _lerp4(iz + 1, ia + 1, iu, in_) * wa

    return ca0 * (1 - wz) + ca1 * wz


# ── AGN grid interpolation: 4-D (ζ_O, logMBH, logEdd, logU, logn) ─


def _interp_agn_grid(
    data: jnp.ndarray,
    grid: MappingsAGNGridData,
    zo_val: float,
    logmbh_val: float,
    logedd_val: float,
    logU_val: float,
    logn_val: float,
) -> jnp.ndarray:
    """5-D linear interpolation for AGN grid (ζ_O, logMBH, logEdd, logU, logn).

    Parameters
    ----------
    data : (N_z, N_m, N_e, N_u, N_n, ...) array

    """
    iz, wz = _interp_index_weight(zo_val, grid.zo_axis)
    im, wm = _interp_index_weight(logmbh_val, grid.logmbh_axis)
    ie, we = _interp_index_weight(logedd_val, grid.logedd_axis)
    iu, wu = _interp_index_weight(logU_val, grid.logU_axis)
    in_, wn = _interp_index_weight(logn_val, grid.logn_axis)

    def _get(iz_, im_, ie_, iu_, in_):
        """Retrieve value from 5D AGN grid."""
        return data[iz_, im_, ie_, iu_, in_]

    def _lerp_n(iz_, im_, ie_, iu_):
        """Interpolate over logn axis."""
        return _get(iz_, im_, ie_, iu_, in_) * (1 - wn) + _get(iz_, im_, ie_, iu_, in_ + 1) * wn

    def _lerp_un(iz_, im_, ie_):
        """Interpolate over logU and logn axes."""
        return _lerp_n(iz_, im_, ie_, iu) * (1 - wu) + _lerp_n(iz_, im_, ie_, iu + 1) * wu

    def _lerp_eun(iz_, im_):
        """Interpolate over logEdd, logU, and logn axes."""
        return _lerp_un(iz_, im_, ie) * (1 - we) + _lerp_un(iz_, im_, ie + 1) * we

    def _lerp_meun(iz_):
        """Interpolate over logMBH, logEdd, logU, and logn axes."""
        return _lerp_eun(iz_, im) * (1 - wm) + _lerp_eun(iz_, im + 1) * wm

    return _lerp_meun(iz) * (1 - wz) + _lerp_meun(iz + 1) * wz


# ── MappingsPhotoStellarBackend ───────────────────────────────────


class MappingsPhotoStellarBackend:
    """MAPPINGS V stellar photoionization backend (Flury et al. 2024).

    Predicts nebular emission line luminosities by interpolating the
    Starburst99 or BPASS MAPPINGS V grids over (ζ_O, log_age, logU, logn),
    weighted by the SSP ionizing photon rate Q_H.

    Parameters
    ----------
    grid_path : str or Path
        Path to flury2024_grids.h5 (built by scripts/build_flury2024_grids.py).
    model : str
        Stellar model: "sb99" (Starburst99) or "bpass" (BPASS v2.2).
    density : str
        Density structure: "cpr" (isobaric, recommended) or "cdn" (isochoric).
    ssp_data : optional
        SSP data object with attributes ssp_wave, ssp_flux, ssp_lgmet,
        ssp_lg_age_gyr. If provided, Q_H table is precomputed at init.
    sfh_mode : str
        "inst" (instantaneous burst) or "cont" (continuous SF). Default "inst".

    Notes
    -----
    Uses constant logn across all age bins. If your application requires
    density to vary with age, use the direct interpolation API instead.

    This backend has ``has_continuum = False`` (MAPPINGS V provides only line
    emission, not nebular continuum).  For applications that need continuum,
    wrap with ``tengri.components.nebular._shared.NebularContinuumFallback``::

        from tengri.components.nebular._shared import NebularContinuumFallback

        backend = MappingsPhotoStellarBackend(...)
        with_cont = NebularContinuumFallback(backend, fallback_mode="warn")

    Example
    -------
    >>> backend = MappingsPhotoStellarBackend("data/flury2024_grids.h5", "sb99", "cpr")
    >>> wave, lum = backend.predict_nebular_line_luminosities(
    ...     ssp_weights, ssp_log_ages_yr, log_z=-2.0, neb_logU=-3.0
    ... )

    """

    name = "mappings_photo_stellar"
    has_free_params = True

    #: erg/s per [Lsun] for this backend's line catalog (#1559). IAU 2015, the
    #: convention the MAPPINGS tabulation is built on. See CueBackend for the
    #: one backend that deviates.
    lsun_erg: float = _LSUN_ERG

    def __init__(
        self,
        grid_path: str | Path = _DEFAULT_GRID_PATH,
        model: str = "sb99",
        density: str = "cpr",
        ssp_data=None,
        sfh_mode: str = "inst",
        ionizing_source_warning: str = "raise",
    ) -> None:
        if model not in ("sb99", "bpass"):
            raise ValueError(f"model must be 'sb99' or 'bpass', got {model!r}")
        if density not in ("cpr", "cdn"):
            raise ValueError(f"density must be 'cpr' or 'cdn', got {density!r}")
        if ionizing_source_warning not in ("raise", "warn", "suppress"):
            raise ValueError("ionizing_source_warning must be 'raise', 'warn', or 'suppress'")
        _emit_mappings_stellar_ionizing_warning(model, ionizing_source_warning)

        self.has_continuum = False
        self.model = model
        self.density = density
        self.sfh_mode = sfh_mode
        self.grid = _load_stellar_grid(grid_path, model, density)

        # Select discrete SFH index
        try:
            self._sfh_idx = self.grid.sfh_labels.index(sfh_mode)
        except ValueError:
            self._sfh_idx = self.grid.sfh_idx_inst

        self._qh_table = None
        self._qh_log_met = None
        self._qh_log_age = None
        self._young_idx = None

        # Photometry preintegration storage (mirrors CB19Backend/CloudyGridBackend duck-type)
        self._preint_continuum: PreintegratedGrid | None = None
        self._preint_lines: PreintegratedLines | None = None
        self._line_lum_collapsed: jnp.ndarray | None = None
        self._has_preint_photometry: bool = False

        if ssp_data is not None:
            self._precompute_qh(ssp_data)

    @property
    def line_names(self) -> list[str]:
        """PyNeb-format line names from the grid (e.g. 'O3_5007A')."""
        # Re-read from file lazily if needed; store on first access
        if not hasattr(self, "_line_names"):
            grid_path = _DEFAULT_GRID_PATH
            with h5py.File(grid_path, "r") as f:
                raw = f[f"{self.model}/{self.density}/line_names"][:]
            self._line_names = [s.decode() if isinstance(s, bytes) else s for s in raw]
        return self._line_names

    def _precompute_qh(self, ssp_data) -> None:
        """Precompute Q_H(metallicity, age) table from SSP spectra."""
        ssp_wave = ssp_data.ssp_wave
        ssp_flux = ssp_data.ssp_flux  # (n_met, n_age, n_wave)
        qh_raw = _compute_qh_grid(ssp_wave, ssp_flux)
        # Replace Inf/NaN with 0: old SSP files with empty far-UV bins
        # produce non-finite Q_H values that would poison the interpolator.
        self._qh_table = sanitize_qh_table(qh_raw, backend_name="MappingsPhotoBackend")
        self._qh_log_met = ssp_data.ssp_lgmet
        self._qh_log_age = ssp_data.ssp_lg_age_gyr + 9.0  # log(age/yr)

        ssp_log_ages = np.array(self._qh_log_age)
        self._young_idx = np.where(ssp_log_ages <= _MAX_NEB_LOG_AGE_YR)[0]

    def _get_qh_at(self, log_z: float, log_age_yr: float) -> float:
        """Bilinear interpolation of Q_H table at (log_z, log_age_yr)."""
        return _qh_bilinear(
            self._qh_table,
            self._qh_log_met,
            self._qh_log_age,
            log_z,
            log_age_yr,
            missing=0.0,
        )

    def predict_nebular_line_luminosities(
        self,
        ssp_weights: jnp.ndarray,
        ssp_log_ages_yr: jnp.ndarray,
        log_z: float,
        neb_logU: float = -3.0,
        neb_logZ_gas: float | None = None,
        neb_logn: float = 2.0,
        neb_fesc: float = 0.0,
        neb_fesc_lya: float = 0.0,
        **_kwargs,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute emission line luminosities from MAPPINGS V stellar grid.

        L_line = Σ_i w_i × Q_H(Z★, age_i) × ratio_i × 10^{logHB_per_logq_i} / L_sun_erg

        Parameters
        ----------
        ssp_weights : array, (n_age,)
            CSP mass weights (Msun per SSP age bin).
        ssp_log_ages_yr : array, (n_age,)
            log10(age/yr) of SSP age bins.
        log_z : float
            Stellar metallicity log10(Z) absolute (for Q_H lookup).
        neb_logU : float
            Ionization parameter log10(U).
        neb_logZ_gas : float or None
            Gas metallicity log10(Z) absolute. None → ties to stellar Z.
        neb_logn : float
            log10(n_H / cm^-3). Default 2.0 (typical HII region density).
        neb_fesc : float
            Ionizing photon escape fraction [0, 1].
        neb_fesc_lya : float
            Ly-alpha specific escape fraction [0, 1].

        Returns
        -------
        wavelengths : array, shape (n_lines,)
            Line wavelengths in vacuum [Angstrom].
        luminosities : array, shape (n_lines,)
            Line luminosities [Lsun].

        Notes
        -----
        **JIT-compatible**: yes, all grid interpolations use ``jnp`` primitives.

        **SFH modes**: "inst" (instantaneous), "cont" (continuous): determines
        which MAPPINGS grid row is used for logHB_per_logq.

        **Metallicity**: Input neb_logZ_gas is absolute log10(Z); internally
        converted to solar-relative ζ_O for grid interpolation via
        _log_z_abs_to_zo.

        References
        ----------
        .. [1] Flury et al. 2024, "MAPPINGS V photoionization grids for nebular
            emission prediction", arXiv:2412.06763
        .. [2] R. S. Sutherland & M. A. Dopita 2017, "Effects of Preionization
            in Radiative Shocks. I. Self-consistent Models," ApJS, 229, 34.
            https://doi.org/10.3847/1538-4365/aa6541

        """
        if neb_logZ_gas is None:
            neb_logZ_gas = log_z

        grid = self.grid
        sfh_idx = self._sfh_idx

        # Convert absolute log10(Z) → ζ_O for grid interpolation
        zo_val = _log_z_abs_to_zo(neb_logZ_gas)

        young_idx = self._young_idx
        if young_idx is None:
            young_ages = ssp_log_ages_yr
            young_weights = ssp_weights
        else:
            young_ages = ssp_log_ages_yr[young_idx]
            young_weights = ssp_weights[young_idx]

        def _contrib_one_age(log_age_i, weight_i):
            """Compute weighted line luminosity contribution for one SSP age bin."""
            qh_i = self._get_qh_at(log_z, log_age_i)

            logHB_pq_i = _interp_stellar_grid(
                grid.logHB_per_logq,
                grid,
                zo_val,
                log_age_i,
                neb_logU,
                neb_logn,
                sfh_idx,
            )
            ratios_i = _interp_stellar_grid(
                grid.line_ratios,
                grid,
                zo_val,
                log_age_i,
                neb_logU,
                neb_logn,
                sfh_idx,
            )

            # L_line = ratio × 10^{logHB_per_logq} × Q_H  [erg/s]
            l_hb_per_qh = 10.0**logHB_pq_i  # erg/photon
            return weight_i * qh_i * l_hb_per_qh * ratios_i * (1.0 - neb_fesc)

        all_contribs = jax.vmap(_contrib_one_age)(young_ages, young_weights)
        total_line_lum = jnp.sum(all_contribs, axis=0)  # (n_lines,)

        # Differential Ly-alpha escape (same pattern as CloudyGridBackend)
        lya_idx = jnp.argmin(jnp.abs(grid.line_wavelengths - 1215.67))
        lya_scale = (1.0 - neb_fesc_lya) / jnp.maximum(1.0 - neb_fesc, 1e-10)
        total_line_lum = total_line_lum.at[lya_idx].multiply(lya_scale)

        return grid.line_wavelengths, total_line_lum

    def preintegrate_for_photometry(
        self,
        filter_waves: list,
        filter_trans: list,
        redshift: float,
        dl_cm: float,
        fixed: dict[int, float] | None = None,
        *,
        neb_logn: float = 2.0,
    ) -> None:
        """Preintegrate MAPPINGS V lines through filters; expose CLOUDY-shaped surface.

        MAPPINGS V stellar grid has four continuous interpolation axes for the
        SED fitting use case: (ζ_O, log_age, logU, logn). The hybrid kernel's
        nebular preint branch (``_kernels/hybrid.py``) only knows how to
        interpolate the CLOUDY-shaped 3-axis surface ``(log_met_abs, log_age_yr, log_U)``.
        We bridge by:

        1. Converting absolute log10(Z) → ζ_O for the grid, then collapsing the
           logn axis to the caller-supplied default (HII region by default).
        2. Relabeling axis 0 from ζ_O (solar-relative) to absolute log10(Z) so
           the kernel's ``_gas_z`` lands on correct coordinates.
        3. Filling ``_preint_continuum.phot`` with zeros: MAPPINGS V provides
           only line emission, no nebular continuum.

        After the call the backend exposes the same surface as
        :class:`~tengri.components.nebular.cloudy_grid.CloudyGridBackend`:
        ``_has_preint_photometry``, ``_preint_continuum``, ``_preint_lines``,
        ``_line_lum_collapsed``, ``_qh_table``, ``_qh_log_met``,
        ``_qh_log_age``, ``_young_idx``, ``grid.line_wavelengths``.

        Parameters
        ----------
        filter_waves : list[ndarray]
            Per-filter observed-frame wavelength grids [Angstrom].
        filter_trans : list[ndarray]
            Per-filter transmission curves (0-1).
        redshift : float
            Source redshift [dimensionless].
        dl_cm : float
            Luminosity distance [cm]. Currently unused (MAPPINGS V provides only
            line emission and lines are projected via filter weights), kept for
            signature compatibility with CloudyGridBackend.
        fixed : dict[int, float], optional
            CLOUDY-shape axis index → value mapping. ``0`` = absolute
            log10(Z), ``1`` = log10(age/yr), ``2`` = log10(U).
        neb_logn : float, keyword-only
            Default density to collapse the logn axis on. MAPPINGS V grid range
            [0.5, 3.5]. Default 2.0 (typical HII region) [log10(cm^-3)].

        Notes
        -----
        **JIT-compatible**: no, build-time NumPy / one-time triweight
        collapses. The resulting attributes are JAX arrays usable inside
        the JIT'd kernel body.
        """
        del dl_cm  # MAPPINGS V: no continuum, lines via point-sampling: no F_nu scaling needed
        grid = self.grid
        sfh_idx = self._sfh_idx

        # 1. Collapse logn axis at the default value.
        # Start with shape (N_z, N_a, N_s, N_u, N_n, N_lines)
        logHB_per_logq_data = jnp.asarray(grid.logHB_per_logq)
        line_ratios_data = jnp.asarray(grid.line_ratios)

        # Slice out the discrete SFH dimension → (N_z, N_a, N_u, N_n, N_lines)
        logHB_per_logq_data = logHB_per_logq_data[:, :, sfh_idx, :, :]
        line_ratios_data = line_ratios_data[:, :, sfh_idx, :, :, :]

        # Collapse logn axis (axis 3) to neb_logn via triweight
        logn_ax = jnp.asarray(grid.logn_axis)
        scatter = 0.5 * float(logn_ax[1] - logn_ax[0])
        w_logn = compute_grid_weights(
            neb_logn, logn_ax, scatter=scatter, edges=edges_for_grid(logn_ax)
        )
        # w_logn is (N_n,); tensordot contracts with axis 3
        logHB_per_logq_data = jnp.tensordot(w_logn, logHB_per_logq_data, axes=([0], [3]))
        line_ratios_data = jnp.tensordot(w_logn, line_ratios_data, axes=([0], [3]))
        # Shapes now: (N_z, N_a, N_u, N_lines) each

        # 2. Build CLOUDY-shape surface axes.
        # Convert ζ_O → absolute log10(Z) for the kernel.
        zo_axis = jnp.asarray(grid.zo_axis)
        log_met_abs_axis = jnp.log10(zo_axis) + _LOG10_ZSUN  # ζ_O to absolute log10(Z)
        log_age_axis = jnp.asarray(grid.log_age_yr_axis)
        log_U_axis = jnp.asarray(grid.logU_axis)
        surface_axes = (log_met_abs_axis, log_age_axis, log_U_axis)
        surface_edges = tuple(edges_for_grid(ax) for ax in surface_axes)

        # 3. Continuum: zeros (MAPPINGS V has no nebular continuum).
        n_filters = len(filter_waves)
        cont_zero = jnp.zeros(
            (
                log_met_abs_axis.shape[0],
                log_age_axis.shape[0],
                log_U_axis.shape[0],
                n_filters,
            ),
            dtype=jnp.float64,
        )
        self._preint_continuum = PreintegratedGrid(
            phot=cont_zero,
            moment=None,
            axes=surface_axes,
            edges=surface_edges,
            effective_wavelengths=jnp.zeros(n_filters),
            effective_wavelengths_rest=jnp.zeros(n_filters),
            log10_flux_scale=0.0,  # unit scale; the caller applies the cosmology
            n_filters=n_filters,
        )

        # 4. Line filter weights via point-sampling (delegates to shared helper).
        self._preint_lines = preintegrate_lines(
            np.asarray(grid.line_wavelengths),
            filter_waves,
            filter_trans,
            redshift,
            axes=surface_axes,
        )

        # 5. Compute line luminosity grid: L_line/Q_H = ratio × 10^logHB_per_logq
        # where logHB_per_logq is log10(L_Hβ / Q_H) [erg/photon], not the Lsun·s
        # version that CB19 uses. The kernel will multiply by Q_H at runtime.
        # logHB_per_logq is log10(L_Hβ / Q_H) [erg·s/photon]. Convert to Lsun·s/photon
        # and take log10 so the kernel's 10**log_lum × Q_H round-trips correctly. This
        # matches the CB19 contract (cf. cloudy_cb19.py:580-583).
        logHB_per_logq_expanded = logHB_per_logq_data[..., jnp.newaxis]  # (N_z, N_a, N_u, 1)
        line_lum_lsun_per_q = line_ratios_data * (10.0**logHB_per_logq_expanded) / _LSUN_ERG
        line_lum = jnp.log10(line_lum_lsun_per_q)

        # 6. Apply caller-provided fixed dict (axes are CLOUDY-shape indices: 0,1,2).
        if fixed:
            self._preint_continuum = slice_fixed_axes(self._preint_continuum, fixed)
            line_axes_remaining = list(surface_axes)
            for axis_idx in sorted(fixed.keys(), reverse=True):
                ax = line_axes_remaining[axis_idx]
                val = fixed[axis_idx]
                scatter = 0.5 * float(ax[1] - ax[0])
                w = compute_grid_weights(val, ax, scatter=scatter, edges=edges_for_grid(ax))
                line_lum = jnp.tensordot(w, line_lum, axes=([0], [axis_idx]))
                line_axes_remaining.pop(axis_idx)
            # Update PreintegratedLines.axes/edges to match the collapsed line_lum
            new_axes = tuple(line_axes_remaining)
            self._preint_lines = PreintegratedLines(
                line_filter_weights=self._preint_lines.line_filter_weights,
                axes=new_axes,
                edges=tuple(edges_for_grid(ax) for ax in new_axes),
            )

        self._line_lum_collapsed = line_lum
        self._has_preint_photometry = True

    def predict_nebular_sed(
        self,
        ssp_weights: jnp.ndarray,
        ssp_wave: jnp.ndarray,
        ssp_log_ages_yr: jnp.ndarray,
        log_z: float,
        neb_logU: float = -3.0,
        neb_logZ_gas: float | None = None,
        neb_logn: float = 2.0,
        neb_fesc: float = 0.0,
        neb_fesc_lya: float = 0.0,
        line_sigma_aa: float = 0.0,
        line_sigma_kms: float = 0.0,
        **_kwargs,
    ) -> jnp.ndarray:
        """Compute nebular emission line SED on the SSP wavelength grid.

        Lines are added as Gaussians (if line_sigma_aa > 0) or delta functions
        (nearest pixel). No nebular continuum: use CloudyGridBackend for that.

        Parameters
        ----------
        ssp_weights : array, shape (n_age,)
            CSP mass weights [Msun per age bin].
        ssp_wave : array, shape (n_wave,)
            SSP wavelength grid [Angstrom].
        ssp_log_ages_yr : array, shape (n_age,)
            log10(age/yr) of SSP age bins.
        log_z : float
            Stellar metallicity [log10(Z)].
        neb_logU : float, optional
            Ionization parameter [log10(U)]. Default: -3.0.
        neb_logZ_gas : float or None, optional
            Gas metallicity [log10(Z)]. None → ties to stellar Z. Default: None.
        neb_logn : float, optional
            Hydrogen density [log10(n_H/cm^-3)]. Default: 2.0.
        neb_fesc : float, optional
            Ionizing photon escape fraction [0, 1]. Default: 0.0.
        neb_fesc_lya : float, optional
            Ly-alpha specific escape fraction [0, 1]. Default: 0.0.
        line_sigma_aa : float, optional
            Gaussian line width [Angstrom]. Default: 0.0 (delta function).

        Returns
        -------
        array, shape (n_wave,)
            Nebular emission line SED [erg/s/Hz] on the SSP wavelength grid.

        Notes
        -----
        **JIT-compatible**: yes, delegates to predict_nebular_line_luminosities
        and place_line_profiles, both JIT-compatible.

        **Continuum**: This backend returns lines only; no nebular continuum.
        For continuum predictions, use CloudyGridBackend or compose with
        NebularContinuumFallback.

        **Line profile**: Gaussian width is constant across all wavelengths.
        For wavelength-dependent broadening, modify place_line_profiles.

        References
        ----------
        .. [1] Flury et al. 2024, "MAPPINGS V photoionization grids for nebular
            emission prediction", arXiv:2412.06763
        .. [2] R. S. Sutherland & M. A. Dopita 2017, "Effects of Preionization
            in Radiative Shocks. I. Self-consistent Models," ApJS, 229, 34.
            https://doi.org/10.3847/1538-4365/aa6541

        """
        line_wave, line_lum = self.predict_nebular_line_luminosities(
            ssp_weights,
            ssp_log_ages_yr,
            log_z,
            neb_logU=neb_logU,
            neb_logZ_gas=neb_logZ_gas,
            neb_logn=neb_logn,
            neb_fesc=neb_fesc,
            neb_fesc_lya=neb_fesc_lya,
        )

        return render_nebular_lines(
            jnp.asarray(line_wave), jnp.asarray(line_lum), ssp_wave, line_sigma_aa, line_sigma_kms
        )


# ── MappingsPhotoAGNBackend ───────────────────────────────────────


class MappingsPhotoAGNBackend:
    """MAPPINGS V AGN photoionization backend (Flury et al. 2024).

    Predicts NLR emission line luminosities by interpolating the OPTXAGNF
    MAPPINGS V grids over (ζ_O, log_MBH, log_Edd, logU, logn).

    Unlike the stellar backend, Q_H is *not* derived from SSP spectra: the
    AGN SED provides the ionizing photons. Call `predict_agn_line_luminosities`
    with an externally computed Q_H (photons/s) from the AGN disc model.

    This backend has ``has_continuum = False``.  For applications that need
    nebular continuum, wrap with
    ``tengri.components.nebular._shared.NebularContinuumFallback``::

        from tengri.components.nebular._shared import NebularContinuumFallback

        backend = MappingsPhotoAGNBackend(...)
        with_cont = NebularContinuumFallback(backend, fallback_mode="warn")

    Parameters
    ----------
    grid_path : str or Path
        Path to flury2024_grids.h5.
    density : str
        Density structure: "cpr" or "cdn".

    Example
    -------
    >>> backend = MappingsPhotoAGNBackend("data/flury2024_grids.h5", density="cpr")
    >>> wave, lum = backend.predict_agn_line_luminosities(
    ...     agn_log_l_ion_erg=45.0,  # log10(L_ion / erg s^-1)
    ...     neb_logZ_gas=-2.0,
    ...     neb_logU=-2.0,
    ...     agn_logmbh=7.0,
    ...     agn_logedd=-0.5,
    ... )

    """

    name = "mappings_photo_agn"
    has_free_params = True

    def __init__(
        self,
        grid_path: str | Path = _DEFAULT_GRID_PATH,
        density: str = "cpr",
        ionizing_source_warning: str = "warn",
    ) -> None:
        if density not in ("cpr", "cdn"):
            raise ValueError(f"density must be 'cpr' or 'cdn', got {density!r}")
        if ionizing_source_warning not in ("raise", "warn", "suppress"):
            raise ValueError("ionizing_source_warning must be 'raise', 'warn', or 'suppress'")
        if ionizing_source_warning != "suppress":
            _emit_mappings_agn_ionizing_warning(ionizing_source_warning)
        self.has_continuum = False
        self.density = density
        self.grid = _load_agn_grid(grid_path, density)

    def predict_agn_line_luminosities(
        self,
        agn_log_l_ion_erg: float,
        neb_logZ_gas: float = _LOG10_ZSUN,
        neb_logU: float = -2.0,
        agn_logmbh: float = 7.0,
        agn_logedd: float = -0.5,
        neb_logn: float = 3.0,
        neb_fesc: float = 0.0,
        **_kwargs,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute AGN NLR emission line luminosities.

        L_line = ratio × 10^{logHB_per_lum} × L_ion × (1 − fesc) / L_sun_erg

        The OPTXAGNF AGN CSV stores log10(L_ion) in erg/s (not Q_H in ph/s),
        so ``logHB_per_lum = log10(L_Hβ / L_ion)`` is dimensionless. Scaling
        by the caller-supplied ionizing luminosity gives absolute NLR luminosities.

        Parameters
        ----------
        agn_log_l_ion_erg : float
            log10 of the AGN ionizing luminosity in erg/s.
        neb_logZ_gas : float
            Gas metallicity log10(Z) absolute.
        neb_logU : float
            Ionization parameter log10(U).
        agn_logmbh : float
            log10(M_BH / Msun).
        agn_logedd : float
            log10(L / L_Edd).
        neb_logn : float
            log10(n_H / cm^-3). Default 3.0 (NLR density).
        neb_fesc : float
            Photon escape fraction [0, 1].

        Returns
        -------
        wavelengths : array, shape (n_lines,)
            Line wavelengths in vacuum [Angstrom].
        luminosities : array, shape (n_lines,)
            Line luminosities [Lsun].

        Notes
        -----
        **JIT-compatible**: yes, all grid interpolations use ``jnp`` primitives.

        **Ionizing source**: Q_H is supplied by the caller and derived from
        the AGN disc model (e.g., Accretion disk spectrum), NOT from SSP spectra.
        Ensure consistency between the ionizing spectrum shape and the choice of
        logU and logedd.

        **Metallicity**: Input neb_logZ_gas is absolute log10(Z); internally
        converted to solar-relative ζ_O for grid interpolation.

        References
        ----------
        .. [1] Flury et al. 2024, "MAPPINGS V photoionization grids for nebular
            emission prediction", arXiv:2412.06763
        .. [2] R. S. Sutherland & M. A. Dopita 2017, "Effects of Preionization
            in Radiative Shocks. I. Self-consistent Models," ApJS, 229, 34.
            https://doi.org/10.3847/1538-4365/aa6541

        """
        grid = self.grid
        zo_val = _log_z_abs_to_zo(neb_logZ_gas)

        logHB_pl = _interp_agn_grid(
            grid.logHB_per_lum,
            grid,
            zo_val,
            agn_logmbh,
            agn_logedd,
            neb_logU,
            neb_logn,
        )
        ratios = _interp_agn_grid(
            grid.line_ratios,
            grid,
            zo_val,
            agn_logmbh,
            agn_logedd,
            neb_logU,
            neb_logn,
        )

        # L_Hβ = 10^{logHB_per_lum} × L_ion_erg
        # L_line = ratio × L_Hβ / L_sun_erg
        l_hb_frac = 10.0**logHB_pl  # L_Hβ / L_ion (dimensionless)
        l_ion_erg = 10.0**agn_log_l_ion_erg
        line_lum = ratios * l_hb_frac * l_ion_erg * (1.0 - neb_fesc)

        return grid.line_wavelengths, line_lum
