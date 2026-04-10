"""CLOUDY grid nebular emission backend.

Loads precomputed CLOUDY photoionization grids (from FSPS/Byler+2017 or
Synthesizer) and computes nebular emission as a function of ionization
parameter (logU) and gas metallicity (logZ_gas).

The physical pipeline:
1. SSP spectrum → integrate below 912 A → Q_H (ionizing photon rate)
2. Q_H × grid(logU, logZ, age) → line luminosities + nebular continuum
3. Apply dust (diffuse only, no birth cloud) to nebular emission
4. Add to stellar SED

References
----------
- Byler et al. 2017, ApJ, 840, 44
- diffhtwo (ArgonneCPAC) for JAX grid interpolation patterns
"""

from typing import NamedTuple

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from tengri.models.nebular._constants import _C_CGS, _LOG10_ZSUN, _LSUN_ERG
from tengri.models.nebular._shared import _interp_index_weight, compute_qh
from tengri.utils.interpolation import compute_grid_weights, edges_for_grid


class CloudyGridData(NamedTuple):
    """Pre-loaded CLOUDY grid data."""

    # Lines
    line_wavelengths: jnp.ndarray  # (n_lines,) rest-frame Angstrom
    line_luminosity: jnp.ndarray  # (n_met, n_age, n_logU, n_lines) Lsun/Q_H
    line_log_met: jnp.ndarray  # (n_met_lines,) log10(Z)
    line_log_age: jnp.ndarray  # (n_age_lines,) log10(age/yr)
    line_log_U: jnp.ndarray  # (n_logU,) log10(U)

    # Continuum
    cont_wavelength: jnp.ndarray  # (n_wave_cont,) Angstrom
    cont_luminosity: jnp.ndarray  # (n_met, n_age, n_logU, n_wave) Lsun_Hz/Q_H
    cont_log_met: jnp.ndarray  # (n_met_cont,) log10(Z)
    cont_log_age: jnp.ndarray  # (n_age_cont,) log10(age/yr)
    cont_log_U: jnp.ndarray  # (n_logU,) shared with lines


def load_cloudy_grid(filepath: str) -> CloudyGridData:
    """Load a tengri-format CLOUDY grid HDF5 file.

    Following FSPS convention, stores luminosities in log10 space
    for interpolation accuracy. A floor of 10^{-95} prevents log(0).

    Metallicity axes are converted from log10(Z/Zsun) (FSPS convention
    in the HDF5 file) to absolute log10(Z) at load time, matching the
    SSP metallicity grid convention used by DSPS.

    Parameters
    ----------
    filepath : str
        Path to cloudy_grid_*.h5 file (from convert_fsps_cloudy_grid.py).
    """
    _LOG_FLOOR = 1e-95  # FSPS convention to avoid log(0)

    with h5py.File(filepath, "r") as f:
        line_lum_raw = np.array(f["lines/luminosity"][:])
        cont_lum_raw = np.array(f["continuum/luminosity"][:])

        # Store in log10 space (FSPS convention for interpolation accuracy)
        line_lum_log = np.log10(line_lum_raw + _LOG_FLOOR)
        cont_lum_log = np.log10(cont_lum_raw + _LOG_FLOOR)

        # Convert metallicity from log10(Z/Zsun) → absolute log10(Z)
        line_log_met_abs = np.array(f["lines/axes/log_met"][:]) + _LOG10_ZSUN
        cont_log_met_abs = np.array(f["continuum/axes/log_met"][:]) + _LOG10_ZSUN

        return CloudyGridData(
            line_wavelengths=jnp.array(f["lines/wavelength"][:]),
            line_luminosity=jnp.array(line_lum_log),  # log10 space!
            line_log_met=jnp.array(line_log_met_abs),  # absolute log10(Z)
            line_log_age=jnp.array(f["lines/axes/log_age_yr"][:]),
            line_log_U=jnp.array(f["lines/axes/log_U"][:]),
            cont_wavelength=jnp.array(f["continuum/wavelength"][:]),
            cont_luminosity=jnp.array(cont_lum_log),  # log10 space!
            cont_log_met=jnp.array(cont_log_met_abs),  # absolute log10(Z)
            cont_log_age=jnp.array(f["continuum/axes/log_age_yr"][:]),
            cont_log_U=jnp.array(f["continuum/axes/log_U"][:]),
        )


# ---------------------------------------------------------------------------
# Q_H computation (ionizing photon rate)
# ---------------------------------------------------------------------------

# Vectorized over metallicity and age dimensions
_compute_qh_grid = jax.vmap(
    jax.vmap(compute_qh, in_axes=(None, 0)),
    in_axes=(None, 0),
)


# ---------------------------------------------------------------------------
# Grid interpolation (trilinear in logZ, logAge, logU)
# ---------------------------------------------------------------------------


def _trilinear_interp(
    data: jnp.ndarray,
    grid_z: jnp.ndarray,
    grid_age: jnp.ndarray,
    grid_u: jnp.ndarray,
    z_val: float,
    age_val: float,
    u_val: float,
) -> jnp.ndarray:
    """Trilinear interpolation on a 3D grid (+ trailing dimensions).

    Parameters
    ----------
    data : array, shape (n_z, n_age, n_u, ...)
        Grid data with 3 leading axes and arbitrary trailing shape.
    grid_z, grid_age, grid_u : array
        Grid axis values.
    z_val, age_val, u_val : float
        Query point.

    Returns
    -------
    array, shape (...)
        Interpolated value.
    """
    iz, wz = _interp_index_weight(z_val, grid_z)
    ia, wa = _interp_index_weight(age_val, grid_age)
    iu, wu = _interp_index_weight(u_val, grid_u)

    # 8 corners of the cube
    c000 = data[iz, ia, iu]
    c001 = data[iz, ia, iu + 1]
    c010 = data[iz, ia + 1, iu]
    c011 = data[iz, ia + 1, iu + 1]
    c100 = data[iz + 1, ia, iu]
    c101 = data[iz + 1, ia, iu + 1]
    c110 = data[iz + 1, ia + 1, iu]
    c111 = data[iz + 1, ia + 1, iu + 1]

    # Interpolate along U
    c00 = c000 * (1 - wu) + c001 * wu
    c01 = c010 * (1 - wu) + c011 * wu
    c10 = c100 * (1 - wu) + c101 * wu
    c11 = c110 * (1 - wu) + c111 * wu

    # Interpolate along age
    c0 = c00 * (1 - wa) + c01 * wa
    c1 = c10 * (1 - wa) + c11 * wa

    # Interpolate along Z
    return c0 * (1 - wz) + c1 * wz


def _trilinear_interp_smooth(
    data: jnp.ndarray,
    grid_z: jnp.ndarray,
    grid_age: jnp.ndarray,
    grid_u: jnp.ndarray,
    z_val: float,
    age_val: float,
    u_val: float,
    scatter: float = 0.2,
    edges_z: jnp.ndarray | None = None,
    edges_age: jnp.ndarray | None = None,
    edges_u: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Smooth triweight-kernel interpolation on a 3-D CLOUDY grid.

    Replaces :func:`_trilinear_interp` for ``grid_interp="triweight"``.
    Returns C²-continuous gradients through grid nodes; no kinks.

    Uses :func:`_shared.compute_grid_weights` on each axis independently,
    then contracts all three weight vectors against the full grid array
    via ``tensordot`` — equivalent to the outer-product weighted sum

        result = Σ_{z,a,u} wz[z] · wa[a] · wu[u] · data[z, a, u, ...]

    The trailing dimensions of ``data`` (e.g. n_lines or n_wave) pass through
    unchanged.

    Parameters
    ----------
    data : array, shape (n_z, n_age, n_u, ...)
    grid_z, grid_age, grid_u : array
        Sorted axis values.
    z_val, age_val, u_val : float
        Query point.
    scatter : float
        Triweight kernel bandwidth (same units as each axis).  Default 0.2.
    edges_z, edges_age, edges_u : array or None
        Precomputed bin edges from :func:`edges_for_grid`.  When ``None``,
        edges are computed on the fly.
    """
    wz = compute_grid_weights(z_val, grid_z, scatter, edges=edges_z)
    wa = compute_grid_weights(age_val, grid_age, scatter, edges=edges_age)
    wu = compute_grid_weights(u_val, grid_u, scatter, edges=edges_u)
    result = jnp.tensordot(wz, data, axes=([0], [0]))  # (n_age, n_u, ...)
    result = jnp.tensordot(wa, result, axes=([0], [0]))  # (n_u, ...)
    result = jnp.tensordot(wu, result, axes=([0], [0]))  # (...)
    return result


# ---------------------------------------------------------------------------
# Main backend class
# ---------------------------------------------------------------------------


class CloudyGridBackend:
    """CLOUDY grid-based nebular emission backend.

    Loads a precomputed CLOUDY grid and computes nebular emission
    (lines + continuum) at arbitrary (logU, logZ_gas) via grid interpolation.
    Q_H is computed on-the-fly from the SSP spectrum.

    Parameters
    ----------
    grid_path : str
        Path to tengri-format CLOUDY HDF5 grid.
    ssp_data : SSPData
        SSP templates (for Q_H computation).
    grid_interp : {"linear", "triweight"}
        Interpolation mode for the CLOUDY grid axes (logZ_gas, log_age, logU).

        ``"linear"`` (default) — piecewise-linear trilinear interpolation.
        Fast; exact at grid nodes; kinks in the gradient at node boundaries.

        ``"triweight"`` — smooth triweight-kernel interpolation (Hearin et al.
        2023 Eq. 10).  C²-continuous gradients through every node; all three
        axes use the same kernel bandwidth ``grid_scatter``.  Slightly slower
        than linear (~3× tensordot cost vs 8-corner lookup) but fully
        differentiable everywhere.
    grid_scatter : float
        Triweight kernel bandwidth in the natural units of each axis (dex).
        Only used when ``grid_interp="triweight"``.  Default 0.2.
    """

    def __init__(
        self,
        grid_path: str,
        ssp_data=None,
        grid_interp: str = "linear",
        grid_scatter: float = 0.2,
    ) -> None:
        if grid_interp not in ("linear", "triweight"):
            raise ValueError(f"grid_interp must be 'linear' or 'triweight', got {grid_interp!r}")
        self.name = "cloudy_grid"
        self.has_free_params = True
        self.has_continuum = True
        self._grid_interp = grid_interp
        self._grid_scatter = grid_scatter
        self.grid = load_cloudy_grid(grid_path)

        # Max age for nebular emission: 100 Myr (conservative).
        # CLOUDY grid stops at ~20 Myr, but Q_H is non-negligible up to
        # ~100 Myr from post-AGB/HB stars. Beyond 100 Myr, Q_H drops
        # >6 orders of magnitude below peak — safe to ignore.
        self._max_neb_log_age = 8.0  # log10(100 Myr in yr)

        # Precompute triweight bin edges (static grid, avoids rebuilding in JIT)
        if grid_interp == "triweight":
            self._edges_z_line = edges_for_grid(self.grid.line_log_met)
            self._edges_age_line = edges_for_grid(self.grid.line_log_age)
            self._edges_u_line = edges_for_grid(self.grid.line_log_U)
            self._edges_z_cont = edges_for_grid(self.grid.cont_log_met)
            self._edges_age_cont = edges_for_grid(self.grid.cont_log_age)
            self._edges_u_cont = edges_for_grid(self.grid.cont_log_U)

        # Precompute Q_H table and young-age index from SSP if provided
        self._qh_table = None
        self._young_idx = None  # indices of SSP age bins with nebular emission
        if ssp_data is not None:
            self._precompute_qh(ssp_data)

        # Photometry preintegration storage
        self._preint_continuum = None
        self._preint_lines = None
        self._has_preint_photometry = False

    def preintegrate_for_photometry(
        self,
        filter_waves: list,
        filter_trans: list,
        redshift: float,
        dl_cm: float,
    ) -> None:
        """Preintegrate CLOUDY continuum + lines through photometric filters.

        After calling this, the backend can compute nebular photometry
        via fast grid interpolation instead of full-wavelength evaluation.

        The continuum grid is converted from log10(Lsun_Hz/Q_H) to linear
        Lsun_Hz/Q_H, then preintegrated through filters. The line wavelengths
        are point-sampled through filters for exact line contributions.

        Results stored in:
        - self._preint_continuum: PreintegratedGrid (n_met, n_age, n_logU, n_filters)
        - self._preint_lines: PreintegratedLines (n_lines, n_filters)

        Parameters
        ----------
        filter_waves : list
            List of filter wavelength arrays (Angstrom).
        filter_trans : list
            List of filter transmission curves.
        redshift : float
            Redshift for redshifting observed-frame wavelengths.
        dl_cm : float
            Luminosity distance (cm).
        """
        from tengri.core.preintegrate import preintegrate_grid, preintegrate_lines

        # Convert continuum from log10 to linear Lsun_Hz/Q_H.
        # The CLOUDY grid uses a floor of log10 = -95 for zero luminosity.
        # Clip to zero below -90 to avoid 1e-95 polluting the filter integral.
        cont_log = np.asarray(self.grid.cont_luminosity)
        cont_linear = np.where(cont_log > -90.0, 10.0**cont_log, 0.0)

        # Preintegrate continuum through filters
        self._preint_continuum = preintegrate_grid(
            cont_linear,
            np.asarray(self.grid.cont_wavelength),
            filter_waves,
            filter_trans,
            redshift,
            dl_cm,
            axes=(
                np.asarray(self.grid.cont_log_met),
                np.asarray(self.grid.cont_log_age),
                np.asarray(self.grid.cont_log_U),
            ),
        )

        # Preintegrate lines through filters
        self._preint_lines = preintegrate_lines(
            np.asarray(self.grid.line_wavelengths),
            filter_waves,
            filter_trans,
            redshift,
            axes=(
                np.asarray(self.grid.line_log_met),
                np.asarray(self.grid.line_log_age),
                np.asarray(self.grid.line_log_U),
            ),
        )

        # Set flag
        self._has_preint_photometry = True

    def _precompute_qh(self, ssp_data) -> None:
        """Precompute Q_H(metallicity, age) table from SSP spectra.

        This avoids recomputing the ionizing integral at every inference step.
        """
        ssp_wave = ssp_data.ssp_wave
        ssp_flux = ssp_data.ssp_flux  # (n_met, n_age, n_wave)

        # Compute Q_H for each (met, age) — vectorized
        self._qh_table = _compute_qh_grid(ssp_wave, ssp_flux)
        # Store as JAX arrays so dynamic indexing works inside jax.grad/vmap
        self._qh_log_met = jnp.asarray(ssp_data.ssp_lgmet)
        self._qh_log_age = jnp.asarray(ssp_data.ssp_lg_age_gyr + 9.0)  # log(age/yr)

        # Precompute indices of young SSP age bins (only these produce
        # ionizing photons and contribute to nebular emission)
        ssp_log_ages = np.array(self._qh_log_age)
        young_mask = ssp_log_ages <= self._max_neb_log_age
        self._young_idx = np.where(young_mask)[0]
        self._n_young = len(self._young_idx)

    def _get_qh_at(
        self,
        log_z: float,
        log_age_yr: float,
    ) -> float:
        """Get Q_H at a specific (logZ, logAge) via interpolation."""
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

    def _make_interp_fn(
        self,
        data: jnp.ndarray,
        grid_z: jnp.ndarray,
        grid_age: jnp.ndarray,
        grid_u: jnp.ndarray,
        edges_z: jnp.ndarray | None = None,
        edges_age: jnp.ndarray | None = None,
        edges_u: jnp.ndarray | None = None,
    ):
        """Build an interpolation closure for the configured grid_interp mode."""
        if self._grid_interp == "triweight":
            s = self._grid_scatter

            def _interp(z, a, u):
                return _trilinear_interp_smooth(
                    data,
                    grid_z,
                    grid_age,
                    grid_u,
                    z,
                    a,
                    u,
                    s,
                    edges_z=edges_z,
                    edges_age=edges_age,
                    edges_u=edges_u,
                )
        else:

            def _interp(z, a, u):
                return _trilinear_interp(data, grid_z, grid_age, grid_u, z, a, u)

        return _interp

    def predict_nebular_line_luminosities(
        self,
        ssp_weights: jnp.ndarray,
        ssp_log_ages_yr: jnp.ndarray,
        log_z: float,
        neb_logU: float = -3.0,
        neb_logZ_gas: float | None = None,
        neb_fesc: float = 0.0,
        neb_fesc_lya: float = 0.0,
        **_kwargs,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute emission line luminosities (vectorized over age bins).

        L_line = sum_i [w_i * Q_H(Z, age_i) * grid(Z_gas, age_i, logU) * (1-f_esc)]

        Ly-alpha (1215.67 A) is treated separately: its luminosity is scaled
        by (1-neb_fesc_lya)/(1-neb_fesc) relative to other lines, reflecting
        resonant scattering that suppresses Ly-alpha escape independently.

        Parameters
        ----------
        ssp_weights : array, shape (n_age,)
            CSP mass weights (Msun per age bin).
        ssp_log_ages_yr : array, shape (n_age,)
            log10(age/yr) of SSP age bins.
        log_z : float
            Stellar metallicity log10(Z) (absolute).
        neb_logU : float
            Ionization parameter log10(U). Default -3.0.
        neb_logZ_gas : float or None
            Gas metallicity log10(Z). None = tie to stellar Z.
        neb_fesc : float
            Escape fraction [0, 1].
        neb_fesc_lya : float
            Ly-alpha escape fraction [0, 1]. Default 0.0.

        Returns
        -------
        wavelengths : array, shape (n_lines,)
        luminosities : array, shape (n_lines,)
        """
        if neb_logZ_gas is None:
            neb_logZ_gas = log_z

        grid = self.grid

        # Only young SSP age bins contribute (age < ~20 Myr)
        # Slice to young bins only — 93 → ~10 bins, ~10x less work
        young_idx = self._young_idx
        young_ages = ssp_log_ages_yr[young_idx]
        young_weights = ssp_weights[young_idx]

        _interp_lines = self._make_interp_fn(
            grid.line_luminosity,
            grid.line_log_met,
            grid.line_log_age,
            grid.line_log_U,
            edges_z=getattr(self, "_edges_z_line", None),
            edges_age=getattr(self, "_edges_age_line", None),
            edges_u=getattr(self, "_edges_u_line", None),
        )

        def _line_contrib_one_age(log_age_i, weight_i):
            qh_i = self._get_qh_at(log_z, log_age_i)
            log_lum_per_qh = _interp_lines(neb_logZ_gas, log_age_i, neb_logU)
            return weight_i * qh_i * (10.0**log_lum_per_qh) * (1.0 - neb_fesc)

        # vmap over young age bins only, then sum
        all_contribs = jax.vmap(_line_contrib_one_age)(
            young_ages, young_weights
        )  # (n_young, n_lines)

        total_line_lum = jnp.sum(all_contribs, axis=0)  # (n_lines,)

        # Apply differential Ly-alpha escape fraction
        # Ly-alpha at 1215.67 A: scale by (1-fesc_lya)/(1-fesc) to replace
        # the generic fesc with the Ly-alpha-specific one
        lya_idx = jnp.argmin(jnp.abs(grid.line_wavelengths - 1215.67))
        lya_scale = (1.0 - neb_fesc_lya) / jnp.maximum(1.0 - neb_fesc, 1e-10)
        total_line_lum = total_line_lum.at[lya_idx].multiply(lya_scale)

        return grid.line_wavelengths, total_line_lum

    def predict_nebular_continuum(
        self,
        ssp_weights: jnp.ndarray,
        ssp_log_ages_yr: jnp.ndarray,
        log_z: float,
        neb_logU: float = -3.0,
        neb_logZ_gas: float | None = None,
        neb_fesc: float = 0.0,
        **_kwargs,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute nebular continuum SED (vectorized over age bins).

        Returns
        -------
        wavelength : array, shape (n_wave_cont,)
        luminosity : array, shape (n_wave_cont,)
            Nebular continuum L_nu (Lsun/Hz, converted to erg/s/Hz at SED assembly).
        """
        if neb_logZ_gas is None:
            neb_logZ_gas = log_z

        grid = self.grid

        # Only young age bins
        young_idx = self._young_idx
        young_ages = ssp_log_ages_yr[young_idx]
        young_weights = ssp_weights[young_idx]

        _interp_cont = self._make_interp_fn(
            grid.cont_luminosity,
            grid.cont_log_met,
            grid.cont_log_age,
            grid.cont_log_U,
            edges_z=getattr(self, "_edges_z_cont", None),
            edges_age=getattr(self, "_edges_age_cont", None),
            edges_u=getattr(self, "_edges_u_cont", None),
        )

        def _cont_contrib_one_age(log_age_i, weight_i):
            qh_i = self._get_qh_at(log_z, log_age_i)
            log_cont_per_qh = _interp_cont(neb_logZ_gas, log_age_i, neb_logU)
            return weight_i * qh_i * (10.0**log_cont_per_qh) * (1.0 - neb_fesc)

        all_contribs = jax.vmap(_cont_contrib_one_age)(
            young_ages, young_weights
        )  # (n_young, n_wave_cont)

        total_cont = jnp.sum(all_contribs, axis=0)
        return grid.cont_wavelength, total_cont

    def predict_nebular_sed(
        self,
        ssp_weights: jnp.ndarray,
        ssp_wave: jnp.ndarray,
        ssp_log_ages_yr: jnp.ndarray,
        log_z: float,
        neb_logU: float = -3.0,
        neb_logZ_gas: float | None = None,
        neb_fesc: float = 0.0,
        neb_fesc_lya: float = 0.0,
        line_sigma_aa: float = 0.0,
        **_kwargs,
    ) -> jnp.ndarray:
        """Compute total nebular emission on the SSP wavelength grid.

        Combines emission lines (as delta functions or Gaussians) with
        nebular continuum, interpolated onto the SSP wavelength grid.

        Parameters
        ----------
        ssp_weights : array, shape (n_age,)
            CSP mass weights.
        ssp_wave : array, shape (n_wave,)
            SSP wavelength grid (Angstrom).
        ssp_log_ages_yr : array, shape (n_age,)
            log10(age/yr) of SSP bins.
        log_z : float
            Stellar metallicity log10(Z) absolute.
        neb_logU : float
            Ionization parameter.
        neb_logZ_gas : float or None
            Gas metallicity log10(Z) (absolute). None = tie to stellar.
        neb_fesc : float
            Ionizing photon escape fraction.
        neb_fesc_lya : float
            Ly-alpha escape fraction [0, 1]. Default 0.0.
        line_sigma_aa : float
            Gaussian width for emission lines (Angstrom). 0 = delta function
            (add to nearest pixel).

        Returns
        -------
        array, shape (n_wave,)
            Nebular SED in erg/s/Hz on the SSP wavelength grid.
        """
        # Get line luminosities
        line_wave, line_lum = self.predict_nebular_line_luminosities(
            ssp_weights,
            ssp_log_ages_yr,
            log_z,
            neb_logU=neb_logU,
            neb_logZ_gas=neb_logZ_gas,
            neb_fesc=neb_fesc,
            neb_fesc_lya=neb_fesc_lya,
        )

        # Get continuum
        cont_wave, cont_lum = self.predict_nebular_continuum(
            ssp_weights,
            ssp_log_ages_yr,
            log_z,
            neb_logU=neb_logU,
            neb_logZ_gas=neb_logZ_gas,
            neb_fesc=neb_fesc,
        )

        # Interpolate continuum onto SSP wavelength grid
        neb_sed = jnp.interp(ssp_wave, cont_wave, cont_lum, left=0.0, right=0.0)

        # Add emission lines
        if line_sigma_aa > 0:
            # Gaussian profiles: spread L_line (Lsun) into L_nu (Lsun/Hz).
            # sigma_nu = sigma_aa[Å→cm] * c[cm/s] / lambda[cm]^2  (wavelength→frequency width).
            # profile / (sqrt(2π) * sigma_nu) normalises to ∫profile dnu = 1 (units: 1/Hz).
            for j in range(len(line_wave)):
                lw = line_wave[j]
                ll = line_lum[j]
                sigma_nu = line_sigma_aa * 1e-8 * _C_CGS / (lw * 1e-8) ** 2
                profile = jnp.exp(-0.5 * ((ssp_wave - lw) / line_sigma_aa) ** 2)
                profile = profile / (jnp.sqrt(2 * jnp.pi) * sigma_nu)
                neb_sed = neb_sed + ll * profile
        else:
            # Delta functions: add to nearest pixel
            # Convert line luminosity to flux density: L_line / delta_nu
            n_wave = ssp_wave.shape[0]
            for j in range(len(line_wave)):
                idx = jnp.argmin(jnp.abs(ssp_wave - line_wave[j]))
                # Clamp to [1, n_wave-2] so both idx-1 and idx+1 are valid
                idx = jnp.clip(idx, 1, n_wave - 2)
                # Approximate delta_nu from pixel width
                dwave = jnp.abs(ssp_wave[idx + 1] - ssp_wave[idx - 1]) / 2.0
                dnu = _C_CGS / (ssp_wave[idx] * 1e-8) ** 2 * dwave * 1e-8
                neb_sed = neb_sed.at[idx].add(line_lum[j] / dnu)

        # Convert from internal Lsun/Hz to erg/s/Hz
        return neb_sed * _LSUN_ERG
