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

from tengri.models.nebular._constants import _C_CGS, _LOG10_ZSUN, _LSUN_ERG
from tengri.models.nebular._shared import _interp_index_weight, compute_qh

# ---------------------------------------------------------------------------
# Ionizing spectrum warnings
# ---------------------------------------------------------------------------


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
        "MappingsPhotoAGNBackend: Q_H must be supplied by the AGN disc model — "
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
        "MAPPINGS V — this is NOT derived from your DSPS SSPs. The stellar continuum "
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

_DEFAULT_GRID_PATH = Path(__file__).resolve().parents[4] / "data" / "flury2024_grids.h5"


# ---------------------------------------------------------------------------
# NamedTuples for pre-loaded grid data
# ---------------------------------------------------------------------------


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

    # Grid values — shape (N_z, N_a, N_s, N_u, N_n)
    logHB_per_logq: jnp.ndarray

    # Line flux ratios relative to Hβ — shape (N_z, N_a, N_s, N_u, N_n, N_lines)
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


# ---------------------------------------------------------------------------
# HDF5 loaders
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Q_H computation (identical to cloudy_grid.py)
# ---------------------------------------------------------------------------

_compute_qh_grid = jax.vmap(jax.vmap(compute_qh, in_axes=(None, 0)), in_axes=(None, 0))


# ---------------------------------------------------------------------------
# Metallicity conversion helpers
# ---------------------------------------------------------------------------


def _log_z_abs_to_zo(log_z_abs: float) -> float:
    """Convert absolute log10(Z) → ζ_O (solar-relative).

    ζ_O = Z / Z_sun = 10^(log10(Z) − log10(Z_sun))
    """
    return 10.0 ** (log_z_abs - _LOG10_ZSUN)


# ---------------------------------------------------------------------------
# Stellar grid interpolation: 4-D (ζ_O, log_age, logU, logn) + sfh slice
# ---------------------------------------------------------------------------


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
        return sliced[iz_, ia_, iu_, in_]

    # 4-D linear interpolation (16 corners)
    def _lerp4(iz_, ia_, iu_, in_):
        # Accumulate over n
        c0 = _get(iz_, ia_, iu_, in_) * (1 - wn) + _get(iz_, ia_, iu_, in_ + 1) * wn
        c1 = _get(iz_, ia_, iu_ + 1, in_) * (1 - wn) + _get(iz_, ia_, iu_ + 1, in_ + 1) * wn
        cu = c0 * (1 - wu) + c1 * wu
        return cu

    ca0 = _lerp4(iz, ia, iu, in_) * (1 - wa) + _lerp4(iz, ia + 1, iu, in_) * wa
    ca1 = _lerp4(iz + 1, ia, iu, in_) * (1 - wa) + _lerp4(iz + 1, ia + 1, iu, in_) * wa

    return ca0 * (1 - wz) + ca1 * wz


# ---------------------------------------------------------------------------
# AGN grid interpolation: 4-D (ζ_O, logMBH, logEdd, logU, logn)
# ---------------------------------------------------------------------------


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
        return data[iz_, im_, ie_, iu_, in_]

    def _lerp_n(iz_, im_, ie_, iu_):
        return _get(iz_, im_, ie_, iu_, in_) * (1 - wn) + _get(iz_, im_, ie_, iu_, in_ + 1) * wn

    def _lerp_un(iz_, im_, ie_):
        return _lerp_n(iz_, im_, ie_, iu) * (1 - wu) + _lerp_n(iz_, im_, ie_, iu + 1) * wu

    def _lerp_eun(iz_, im_):
        return _lerp_un(iz_, im_, ie) * (1 - we) + _lerp_un(iz_, im_, ie + 1) * we

    def _lerp_meun(iz_):
        return _lerp_eun(iz_, im) * (1 - wm) + _lerp_eun(iz_, im + 1) * wm

    return _lerp_meun(iz) * (1 - wz) + _lerp_meun(iz + 1) * wz


# ---------------------------------------------------------------------------
# MappingsPhotoStellarBackend
# ---------------------------------------------------------------------------


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

    Example
    -------
    >>> backend = MappingsPhotoStellarBackend("data/flury2024_grids.h5", "sb99", "cpr")
    >>> wave, lum = backend.predict_nebular_line_luminosities(
    ...     ssp_weights, ssp_log_ages_yr, log_z=-2.0, neb_logU=-3.0
    ... )
    """

    name = "mappings_photo_stellar"
    has_free_params = True

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
        self._qh_table = _compute_qh_grid(ssp_wave, ssp_flux)
        self._qh_log_met = ssp_data.ssp_lgmet
        self._qh_log_age = ssp_data.ssp_lg_age_gyr + 9.0  # log(age/yr)

        ssp_log_ages = np.array(self._qh_log_age)
        self._young_idx = np.where(ssp_log_ages <= _MAX_NEB_LOG_AGE_YR)[0]

    def _get_qh_at(self, log_z: float, log_age_yr: float) -> float:
        """Bilinear interpolation of Q_H table at (log_z, log_age_yr)."""
        if self._qh_table is None:
            return 0.0
        iz, wz = _interp_index_weight(log_z, self._qh_log_met)
        ia, wa = _interp_index_weight(log_age_yr, self._qh_log_age)
        q00 = self._qh_table[iz, ia]
        q01 = self._qh_table[iz, ia + 1]
        q10 = self._qh_table[iz + 1, ia]
        q11 = self._qh_table[iz + 1, ia + 1]
        q0 = q00 * (1 - wa) + q01 * wa
        q1 = q10 * (1 - wa) + q11 * wa
        return q0 * (1 - wz) + q1 * wz

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
        wavelengths : array, (n_lines,)   vacuum Angstrom
        luminosities : array, (n_lines,)  Lsun
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
            # Convert to Lsun: divide by _LSUN_ERG
            l_hb_per_qh = 10.0**logHB_pq_i  # erg/photon
            return weight_i * qh_i * l_hb_per_qh * ratios_i * (1.0 - neb_fesc) / _LSUN_ERG

        all_contribs = jax.vmap(_contrib_one_age)(young_ages, young_weights)
        total_line_lum = jnp.sum(all_contribs, axis=0)  # (n_lines,)

        # Differential Ly-alpha escape (same pattern as CloudyGridBackend)
        lya_idx = jnp.argmin(jnp.abs(grid.line_wavelengths - 1215.67))
        lya_scale = (1.0 - neb_fesc_lya) / jnp.maximum(1.0 - neb_fesc, 1e-10)
        total_line_lum = total_line_lum.at[lya_idx].multiply(lya_scale)

        return grid.line_wavelengths, total_line_lum

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
        **_kwargs,
    ) -> jnp.ndarray:
        """Compute nebular emission line SED on the SSP wavelength grid.

        Lines are added as Gaussians (if line_sigma_aa > 0) or delta functions
        (nearest pixel). No nebular continuum — use CloudyGridBackend for that.

        Returns
        -------
        array, (n_wave,)  Lsun/Hz on the SSP wavelength grid.
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

        neb_sed = jnp.zeros_like(ssp_wave)
        n_wave = ssp_wave.shape[0]

        if line_sigma_aa > 0:
            for j in range(len(line_wave)):
                lw = line_wave[j]
                ll = line_lum[j]
                sigma_nu = line_sigma_aa * _C_CGS / (lw * 1e-8) ** 2
                profile = jnp.exp(-0.5 * ((ssp_wave - lw) / line_sigma_aa) ** 2)
                profile = profile / (jnp.sqrt(2 * jnp.pi) * sigma_nu)
                neb_sed = neb_sed + ll * profile
        else:
            for j in range(len(line_wave)):
                idx = jnp.argmin(jnp.abs(ssp_wave - line_wave[j]))
                idx = jnp.clip(idx, 1, n_wave - 2)
                dwave = jnp.abs(ssp_wave[idx + 1] - ssp_wave[idx - 1]) / 2.0
                dnu = _C_CGS / (ssp_wave[idx] * 1e-8) ** 2 * dwave * 1e-8
                neb_sed = neb_sed.at[idx].add(line_lum[j] / dnu)

        return neb_sed


# ---------------------------------------------------------------------------
# MappingsPhotoAGNBackend
# ---------------------------------------------------------------------------


class MappingsPhotoAGNBackend:
    """MAPPINGS V AGN photoionization backend (Flury et al. 2024).

    Predicts NLR emission line luminosities by interpolating the OPTXAGNF
    MAPPINGS V grids over (ζ_O, log_MBH, log_Edd, logU, logn).

    Unlike the stellar backend, Q_H is *not* derived from SSP spectra — the
    AGN SED provides the ionizing photons. Call `predict_agn_line_luminosities`
    with an externally computed Q_H (photons/s) from the AGN disc model.

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
        wavelengths : array, (n_lines,)   vacuum Angstrom
        luminosities : array, (n_lines,)  Lsun
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
        line_lum = ratios * l_hb_frac * l_ion_erg * (1.0 - neb_fesc) / _LSUN_ERG

        return grid.line_wavelengths, line_lum
