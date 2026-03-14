"""DSPS (Differentiable Stellar Population Synthesis) integration.

Wraps the DSPS CSP integral and SSP template loading. DSPS provides
the differentiable mapping from SFH weights → composite stellar
population spectrum, which is the core of the forward model.

References
----------
- Hearin et al. 2023 (arXiv:2112.08423): DSPS
- SSP templates: https://halos.as.arizona.edu/suchethacooray/ssp-spectra/
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple


class SSPData(NamedTuple):
    """Container for SSP template data.

    Attributes
    ----------
    ssp_wave : array, shape (n_wave,)
        Rest-frame wavelength grid (Angstrom).
    ssp_flux : array, shape (n_met, n_age, n_wave)
        SSP luminosity per unit mass (Lsun/Hz/Msun or similar).
    ssp_lg_age_gyr : array, shape (n_age,)
        Log10(age/Gyr) of SSP templates.
    ssp_lgmet : array, shape (n_met,)
        Log10(Z/Zsun) metallicity grid.
    """
    ssp_wave: jnp.ndarray
    ssp_flux: jnp.ndarray
    ssp_lg_age_gyr: jnp.ndarray
    ssp_lgmet: jnp.ndarray


def load_ssp_data(filepath: str) -> SSPData:
    """Load SSP templates from DSPS-compatible HDF5 file.

    Parameters
    ----------
    filepath : str
        Path to HDF5 file with fields: ssp_wave, ssp_flux,
        ssp_lg_age_gyr, ssp_lgmet.

    Returns
    -------
    SSPData
        Loaded SSP template data.
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py required for SSP loading: pip install h5py")

    with h5py.File(filepath, "r") as f:
        return SSPData(
            ssp_wave=jnp.array(f["ssp_wave"][:]),
            ssp_flux=jnp.array(f["ssp_flux"][:]),
            ssp_lg_age_gyr=jnp.array(f["ssp_lg_age_gyr"][:]),
            ssp_lgmet=jnp.array(f["ssp_lgmet"][:]),
        )


def load_ssp_data_dsps(filepath: str) -> SSPData:
    """Load SSP templates using DSPS native loader.

    Falls back to load_ssp_data() if DSPS is not installed.

    Parameters
    ----------
    filepath : str
        Path to HDF5 file.

    Returns
    -------
    SSPData
        Loaded SSP template data.
    """
    try:
        from dsps import load_ssp_templates
        ssp_data = load_ssp_templates(fn=filepath)
        return SSPData(
            ssp_wave=jnp.array(ssp_data.ssp_wave),
            ssp_flux=jnp.array(ssp_data.ssp_flux),
            ssp_lg_age_gyr=jnp.array(ssp_data.ssp_lg_age_gyr),
            ssp_lgmet=jnp.array(ssp_data.ssp_lgmet),
        )
    except ImportError:
        return load_ssp_data(filepath)


@jax.jit
def compute_csp_weights(sfr_on_ssp_ages: jnp.ndarray,
                        ssp_ages_yr: jnp.ndarray) -> jnp.ndarray:
    """Compute SFH weights (mass formed per SSP age bin).

    Returns the stellar mass formed in each age bin (Msun), NOT
    normalized to sum=1. This way the CSP SED = sum(w_i * SSP_i)
    is in Lsun/Hz (same as DSPS), not Lsun/Hz/Msun.

    The total stellar mass formed is sum(weights).

    Parameters
    ----------
    sfr_on_ssp_ages : array, shape (n_age,)
        Star formation rate at each SSP age (Msun/yr).
    ssp_ages_yr : array, shape (n_age,)
        SSP ages in years.

    Returns
    -------
    array, shape (n_age,)
        Mass formed per age bin (Msun). Sum = total mass formed.
    """
    # Trapezoidal half-widths for each age bin
    dt = jnp.concatenate([
        jnp.array([ssp_ages_yr[1] - ssp_ages_yr[0]]),
        0.5 * (ssp_ages_yr[2:] - ssp_ages_yr[:-2]),
        jnp.array([ssp_ages_yr[-1] - ssp_ages_yr[-2]]),
    ])
    return sfr_on_ssp_ages * dt


LSUN_ERG_PER_S = 3.828e33  # erg/s (IAU 2015)


@jax.jit
def compute_csp_sed(weights: jnp.ndarray,
                    ssp_flux_at_met: jnp.ndarray,
                    dust_attenuation: jnp.ndarray) -> jnp.ndarray:
    """Compute composite stellar population SED.

    SED = Lsun * sum_i (weight_i * dust_i * ssp_flux_i)

    where weights are in Msun (mass formed per bin) and SSP flux
    is in Lsun/Hz/Msun. The result is in erg/s/Hz.

    Parameters
    ----------
    weights : array, shape (n_age,)
        Mass formed per age bin (Msun) from compute_csp_weights.
    ssp_flux_at_met : array, shape (n_age, n_wave)
        SSP spectra at fixed metallicity (Lsun/Hz/Msun).
    dust_attenuation : array, shape (n_age, n_wave)
        Multiplicative dust transmission per age and wavelength.

    Returns
    -------
    array, shape (n_wave,)
        Composite SED in erg/s/Hz (rest-frame luminosity density).
    """
    # weights [Msun] * ssp [Lsun/Hz/Msun] * dust [dimensionless] -> Lsun/Hz
    sed_lsun = jnp.einsum("i,iw,iw->w", weights, dust_attenuation, ssp_flux_at_met)
    return sed_lsun * LSUN_ERG_PER_S  # -> erg/s/Hz


@jax.jit
def interpolate_metallicity(ssp_flux: jnp.ndarray,
                            ssp_lgmet: jnp.ndarray,
                            log_z: float) -> jnp.ndarray:
    """Interpolate SSP flux to a target metallicity.

    Linear interpolation in log(Z/Zsun) space between the two
    nearest metallicity grid points.

    Parameters
    ----------
    ssp_flux : array, shape (n_met, n_age, n_wave)
        Full SSP flux grid.
    ssp_lgmet : array, shape (n_met,)
        Log10(Z/Zsun) grid.
    log_z : float
        Target log10(Z/Zsun).

    Returns
    -------
    array, shape (n_age, n_wave)
        Interpolated SSP flux.
    """
    # Clamp to grid bounds
    log_z_clamped = jnp.clip(log_z, ssp_lgmet[0], ssp_lgmet[-1])

    # Find bracketing indices
    idx = jnp.searchsorted(ssp_lgmet, log_z_clamped) - 1
    idx = jnp.clip(idx, 0, len(ssp_lgmet) - 2)

    # Linear interpolation weight
    frac = (log_z_clamped - ssp_lgmet[idx]) / (ssp_lgmet[idx + 1] - ssp_lgmet[idx])

    return (1.0 - frac) * ssp_flux[idx] + frac * ssp_flux[idx + 1]
