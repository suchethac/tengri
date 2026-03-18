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


# Physical constants
_H_PLANCK = 6.62607015e-27   # erg s
_C_CGS = 2.99792458e10       # cm/s
_LSUN_ERG = 3.828e33         # erg/s
_LYMAN_LIMIT = 911.8         # Angstrom


class CloudyGridData(NamedTuple):
    """Pre-loaded CLOUDY grid data."""

    # Lines
    line_wavelengths: jnp.ndarray    # (n_lines,) rest-frame Angstrom
    line_luminosity: jnp.ndarray     # (n_met, n_age, n_logU, n_lines) Lsun/Q_H
    line_log_met: jnp.ndarray        # (n_met_lines,) log10(Z)
    line_log_age: jnp.ndarray        # (n_age_lines,) log10(age/yr)
    line_log_U: jnp.ndarray          # (n_logU,) log10(U)

    # Continuum
    cont_wavelength: jnp.ndarray     # (n_wave_cont,) Angstrom
    cont_luminosity: jnp.ndarray     # (n_met, n_age, n_logU, n_wave) Lsun_Hz/Q_H
    cont_log_met: jnp.ndarray        # (n_met_cont,) log10(Z)
    cont_log_age: jnp.ndarray        # (n_age_cont,) log10(age/yr)
    cont_log_U: jnp.ndarray          # (n_logU,) shared with lines


def load_cloudy_grid(filepath: str) -> CloudyGridData:
    """Load a diffsed-format CLOUDY grid HDF5 file.

    Parameters
    ----------
    filepath : str
        Path to cloudy_grid_*.h5 file (from convert_fsps_cloudy_grid.py).
    """
    with h5py.File(filepath, "r") as f:
        return CloudyGridData(
            line_wavelengths=jnp.array(f["lines/wavelength"][:]),
            line_luminosity=jnp.array(f["lines/luminosity"][:]),
            line_log_met=jnp.array(f["lines/axes/log_met"][:]),
            line_log_age=jnp.array(f["lines/axes/log_age_yr"][:]),
            line_log_U=jnp.array(f["lines/axes/log_U"][:]),
            cont_wavelength=jnp.array(f["continuum/wavelength"][:]),
            cont_luminosity=jnp.array(f["continuum/luminosity"][:]),
            cont_log_met=jnp.array(f["continuum/axes/log_met"][:]),
            cont_log_age=jnp.array(f["continuum/axes/log_age_yr"][:]),
            cont_log_U=jnp.array(f["continuum/axes/log_U"][:]),
        )


# ---------------------------------------------------------------------------
# Q_H computation (ionizing photon rate)
# ---------------------------------------------------------------------------

@jax.jit
def compute_qh(
    ssp_wave: jnp.ndarray,
    ssp_flux: jnp.ndarray,
) -> float:
    """Compute ionizing photon rate Q_H from an SSP spectrum.

    Q_H = integral_{0}^{912A} [L_nu / (h * nu)] d_nu

    Parameters
    ----------
    ssp_wave : array, shape (n_wave,)
        SSP wavelength grid in Angstrom (increasing).
    ssp_flux : array, shape (n_wave,)
        SSP flux in Lsun/Hz/Msun.

    Returns
    -------
    float
        Q_H in photons/s/Msun.
    """
    # Convert wavelength to frequency
    nu = _C_CGS / (ssp_wave * 1e-8)  # Hz

    # L_nu in erg/s/Hz/Msun
    l_nu = ssp_flux * _LSUN_ERG

    # Photon rate density: L_nu / (h * nu)
    photon_rate = l_nu / (_H_PLANCK * nu)

    # Mask to ionizing wavelengths only (below Lyman limit)
    mask = ssp_wave < _LYMAN_LIMIT

    # Integrate over frequency (nu decreases as wave increases)
    # Use negative sign because we integrate in wave space (increasing)
    integrand = jnp.where(mask, photon_rate, 0.0)
    qh = -jnp.trapezoid(integrand, nu)

    return jnp.maximum(qh, 0.0)


# Vectorized over metallicity and age dimensions
_compute_qh_grid = jax.vmap(
    jax.vmap(compute_qh, in_axes=(None, 0)),
    in_axes=(None, 0),
)


# ---------------------------------------------------------------------------
# Grid interpolation (trilinear in logZ, logAge, logU)
# ---------------------------------------------------------------------------

def _interp_index_weight(
    x: float,
    grid: jnp.ndarray,
) -> tuple[int, float]:
    """Find bracketing index and interpolation weight for 1D grid.

    Returns (i, w) such that value = grid[i]*(1-w) + grid[i+1]*w.
    Clips to grid bounds.
    """
    # Clip to grid range
    x_clipped = jnp.clip(x, grid[0], grid[-1])

    # Find index
    idx = jnp.searchsorted(grid, x_clipped, side="right") - 1
    idx = jnp.clip(idx, 0, len(grid) - 2)

    # Interpolation weight
    dx = grid[idx + 1] - grid[idx]
    w = jnp.where(dx > 0, (x_clipped - grid[idx]) / dx, 0.0)

    return idx, w


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


# ---------------------------------------------------------------------------
# Main backend class
# ---------------------------------------------------------------------------

class CloudyGridBackend:
    """CLOUDY grid-based nebular emission backend.

    Loads a precomputed CLOUDY grid and computes nebular emission
    (lines + continuum) at arbitrary (logU, logZ_gas) via trilinear
    interpolation. Q_H is computed on-the-fly from the SSP spectrum.

    Parameters
    ----------
    grid_path : str
        Path to diffsed-format CLOUDY HDF5 grid.
    ssp_data : SSPData
        SSP templates (for Q_H computation).
    """

    def __init__(self, grid_path: str, ssp_data=None) -> None:
        self.name = "cloudy_grid"
        self.has_free_params = True
        self.grid = load_cloudy_grid(grid_path)

        # Precompute Q_H table from SSP if provided
        self._qh_table = None
        if ssp_data is not None:
            self._precompute_qh(ssp_data)

    def _precompute_qh(self, ssp_data) -> None:
        """Precompute Q_H(metallicity, age) table from SSP spectra.

        This avoids recomputing the ionizing integral at every inference step.
        """
        ssp_wave = ssp_data.ssp_wave
        ssp_flux = ssp_data.ssp_flux  # (n_met, n_age, n_wave)

        # Compute Q_H for each (met, age) — vectorized
        self._qh_table = _compute_qh_grid(ssp_wave, ssp_flux)
        self._qh_log_met = ssp_data.ssp_lgmet
        self._qh_log_age = ssp_data.ssp_lg_age_gyr + 9.0  # log(age/yr)

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

    def predict_nebular_line_luminosities(
        self,
        ssp_weights: jnp.ndarray,
        ssp_log_ages_yr: jnp.ndarray,
        log_z: float,
        neb_logU: float = -3.0,
        neb_logZ_gas: float = None,
        neb_fesc: float = 0.0,
        **_kwargs,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute emission line luminosities.

        For each SSP age bin with non-zero weight:
          L_line = weight * Q_H(Z, age) * grid(Z_gas, age, logU)

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
            Escape fraction [0, 1]. Fraction of ionizing photons
            that escape without producing nebular emission.

        Returns
        -------
        wavelengths : array, shape (n_lines,)
            Emission line wavelengths (rest-frame Angstrom).
        luminosities : array, shape (n_lines,)
            Line luminosities (Lsun).
        """
        if neb_logZ_gas is None:
            neb_logZ_gas = log_z

        grid = self.grid
        n_ages = len(ssp_weights)

        # Sum over age bins: L_line = sum_i [w_i * Q_H(Z, age_i) * grid(Z_gas, age_i, logU)]
        total_line_lum = jnp.zeros(len(grid.line_wavelengths))

        for i in range(n_ages):
            log_age_i = ssp_log_ages_yr[i]

            # Only young stars produce ionizing photons (age < ~20 Myr)
            # Grid only covers ages up to ~20 Myr anyway
            age_in_grid = (
                (log_age_i >= grid.line_log_age[0])
                & (log_age_i <= grid.line_log_age[-1])
            )

            # Q_H at this (Z, age)
            qh_i = self._get_qh_at(log_z, log_age_i)

            # Grid luminosity per Q_H at (Z_gas, age, logU)
            lum_per_qh = _trilinear_interp(
                grid.line_luminosity,
                grid.line_log_met,
                grid.line_log_age,
                grid.line_log_U,
                neb_logZ_gas,
                log_age_i,
                neb_logU,
            )

            # Contribution: weight * Q_H * grid_value * (1 - f_esc)
            contrib = ssp_weights[i] * qh_i * lum_per_qh * (1.0 - neb_fesc)
            total_line_lum = total_line_lum + jnp.where(age_in_grid, contrib, 0.0)

        return grid.line_wavelengths, total_line_lum

    def predict_nebular_continuum(
        self,
        ssp_weights: jnp.ndarray,
        ssp_log_ages_yr: jnp.ndarray,
        log_z: float,
        neb_logU: float = -3.0,
        neb_logZ_gas: float = None,
        neb_fesc: float = 0.0,
        **_kwargs,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute nebular continuum SED.

        Returns
        -------
        wavelength : array, shape (n_wave_cont,)
            Nebular continuum wavelength grid (Angstrom).
        luminosity : array, shape (n_wave_cont,)
            Nebular continuum L_nu (Lsun/Hz).
        """
        if neb_logZ_gas is None:
            neb_logZ_gas = log_z

        grid = self.grid
        n_ages = len(ssp_weights)

        total_cont = jnp.zeros(len(grid.cont_wavelength))

        for i in range(n_ages):
            log_age_i = ssp_log_ages_yr[i]

            age_in_grid = (
                (log_age_i >= grid.cont_log_age[0])
                & (log_age_i <= grid.cont_log_age[-1])
            )

            qh_i = self._get_qh_at(log_z, log_age_i)

            cont_per_qh = _trilinear_interp(
                grid.cont_luminosity,
                grid.cont_log_met,
                grid.cont_log_age,
                grid.cont_log_U,
                neb_logZ_gas,
                log_age_i,
                neb_logU,
            )

            contrib = ssp_weights[i] * qh_i * cont_per_qh * (1.0 - neb_fesc)
            total_cont = total_cont + jnp.where(age_in_grid, contrib, 0.0)

        return grid.cont_wavelength, total_cont

    def predict_nebular_sed(
        self,
        ssp_weights: jnp.ndarray,
        ssp_wave: jnp.ndarray,
        ssp_log_ages_yr: jnp.ndarray,
        log_z: float,
        neb_logU: float = -3.0,
        neb_logZ_gas: float = None,
        neb_fesc: float = 0.0,
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
            Gas metallicity. None = tie to stellar.
        neb_fesc : float
            Ionizing photon escape fraction.
        line_sigma_aa : float
            Gaussian width for emission lines (Angstrom). 0 = delta function
            (add to nearest pixel).

        Returns
        -------
        array, shape (n_wave,)
            Nebular SED in Lsun/Hz on the SSP wavelength grid.
        """
        # Get line luminosities
        line_wave, line_lum = self.predict_nebular_line_luminosities(
            ssp_weights, ssp_log_ages_yr, log_z,
            neb_logU=neb_logU, neb_logZ_gas=neb_logZ_gas, neb_fesc=neb_fesc,
        )

        # Get continuum
        cont_wave, cont_lum = self.predict_nebular_continuum(
            ssp_weights, ssp_log_ages_yr, log_z,
            neb_logU=neb_logU, neb_logZ_gas=neb_logZ_gas, neb_fesc=neb_fesc,
        )

        # Interpolate continuum onto SSP wavelength grid
        neb_sed = jnp.interp(ssp_wave, cont_wave, cont_lum, left=0.0, right=0.0)

        # Add emission lines
        if line_sigma_aa > 0:
            # Gaussian profiles
            for j in range(len(line_wave)):
                lw = line_wave[j]
                ll = line_lum[j]
                # Convert line luminosity (Lsun) to Lsun/Hz via Gaussian
                # sigma_nu = sigma_lambda * c / lambda^2
                sigma_nu = line_sigma_aa * _C_CGS / (lw * 1e-8) ** 2
                profile = jnp.exp(-0.5 * ((ssp_wave - lw) / line_sigma_aa) ** 2)
                profile = profile / (jnp.sqrt(2 * jnp.pi) * sigma_nu)
                neb_sed = neb_sed + ll * _LSUN_ERG * profile
        else:
            # Delta functions: add to nearest pixel
            # Convert line luminosity to flux density: L_line / delta_nu
            for j in range(len(line_wave)):
                idx = jnp.argmin(jnp.abs(ssp_wave - line_wave[j]))
                # Approximate delta_nu from pixel width
                dwave = jnp.abs(ssp_wave[idx + 1] - ssp_wave[idx - 1]) / 2.0
                dnu = _C_CGS / (ssp_wave[idx] * 1e-8) ** 2 * dwave * 1e-8
                line_flux_density = line_lum[j] / dnu  # Lsun/Hz
                neb_sed = neb_sed.at[idx].add(line_flux_density)

        return neb_sed
