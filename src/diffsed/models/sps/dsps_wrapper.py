"""DSPS (Differentiable Stellar Population Synthesis) integration.

Wraps the DSPS CSP integral and SSP template loading. DSPS provides
the differentiable mapping from SFH weights → composite stellar
population spectrum, which is the core of the forward model.

References
----------
- Hearin et al. 2023 (arXiv:2112.08423): DSPS
- SSP templates: https://halos.as.arizona.edu/suchethacooray/ssp-spectra/
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp


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
    ssp_mass_remaining : array, shape (n_met, n_age), optional
        Fraction of formed mass still in living stars + remnants
        at each age and metallicity. Computed from stellar evolution
        tracks; depends on IMF and isochrone library. None if not
        available (surviving mass cannot be computed).
    """

    ssp_wave: jnp.ndarray
    ssp_flux: jnp.ndarray
    ssp_lg_age_gyr: jnp.ndarray
    ssp_lgmet: jnp.ndarray
    ssp_mass_remaining: jnp.ndarray | None = None


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
        raise ImportError("h5py required for SSP loading: pip install h5py") from None

    with h5py.File(filepath, "r") as f:
        mass_remaining = None
        if "ssp_mass_remaining" in f:
            mass_remaining = jnp.array(f["ssp_mass_remaining"][:])
        return SSPData(
            ssp_wave=jnp.array(f["ssp_wave"][:]),
            ssp_flux=jnp.array(f["ssp_flux"][:]),
            ssp_lg_age_gyr=jnp.array(f["ssp_lg_age_gyr"][:]),
            ssp_lgmet=jnp.array(f["ssp_lgmet"][:]),
            ssp_mass_remaining=mass_remaining,
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
def compute_csp_weights(sfr_on_ssp_ages: jnp.ndarray, ssp_ages_yr: jnp.ndarray) -> jnp.ndarray:
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
    dt = jnp.concatenate(
        [
            jnp.array([ssp_ages_yr[1] - ssp_ages_yr[0]]),
            0.5 * (ssp_ages_yr[2:] - ssp_ages_yr[:-2]),
            jnp.array([ssp_ages_yr[-1] - ssp_ages_yr[-2]]),
        ]
    )
    return sfr_on_ssp_ages * dt


# ---------------------------------------------------------------------------
# Alpha-element enhancement
# ---------------------------------------------------------------------------

# Coefficient converting [alpha/Fe] to total metallicity offset.
# Alpha elements (O, Mg, Si, Ca, Ti) dominate the metal mass budget,
# so [Z/H]_eff ≈ [Fe/H] + A * [alpha/Fe] with A ~ 0.75.
# Reference: Thomas, Maraston & Bender 2003; Vazdekis et al. 2015.
_ALPHA_TO_Z_COEFF = 0.75


@jax.jit
def effective_metallicity(log_z_fe: float, alpha_fe: float = 0.0) -> float:
    """Convert [Fe/H] + [alpha/Fe] to effective total metallicity.

    Approximates the effect of alpha-element enhancement on the SED
    as a shift in the total metallicity used for SSP interpolation:

        [Z/H]_eff = [Fe/H] + 0.75 * [alpha/Fe]

    This is the standard approach when SSP templates are computed at
    fixed abundance ratios and cannot be changed at runtime.

    Parameters
    ----------
    log_z_fe : float
        Iron abundance [Fe/H] (or equivalently, log10(Z) when
        [alpha/Fe] = 0, i.e. the existing ``log_z`` parameter).
    alpha_fe : float, optional
        Alpha-element enhancement [alpha/Fe] in dex.
        Default is 0.0 (solar abundance ratios).

    Returns
    -------
    float
        Effective total metallicity log10(Z_eff) in the same
        units as ``log_z_fe``.

    References
    ----------
    Thomas, Maraston & Bender 2003, MNRAS 339, 897
    Vazdekis et al. 2015, MNRAS 449, 1177
    """
    return log_z_fe + _ALPHA_TO_Z_COEFF * alpha_fe


LSUN_ERG_PER_S = 3.828e33  # erg/s (IAU 2015)


@jax.jit
def compute_csp_sed(
    weights: jnp.ndarray, ssp_flux_at_met: jnp.ndarray, dust_attenuation: jnp.ndarray
) -> jnp.ndarray:
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
def interpolate_metallicity(
    ssp_flux: jnp.ndarray, ssp_lgmet: jnp.ndarray, log_z: float
) -> jnp.ndarray:
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


@jax.jit
def interpolate_metallicity_evolving(
    ssp_flux: jnp.ndarray,
    ssp_lgmet: jnp.ndarray,
    log_z_per_age: jnp.ndarray,
) -> jnp.ndarray:
    """Interpolate SSP flux with a different metallicity per age bin.

    Each SSP age bin is interpolated at its own metallicity, enabling
    time-evolving metallicity models (e.g., chemical enrichment).

    Parameters
    ----------
    ssp_flux : array, shape (n_met, n_age, n_wave)
        Full SSP flux grid.
    ssp_lgmet : array, shape (n_met,)
        Log10(Z/Zsun) grid.
    log_z_per_age : array, shape (n_age,)
        Target log10(Z/Zsun) at each age bin.

    Returns
    -------
    array, shape (n_age, n_wave)
        Interpolated SSP flux with per-age metallicity.
    """
    def _interp_one_age(log_z_i, ssp_flux_at_age_i):
        """Interpolate a single age bin at its metallicity.

        Parameters
        ----------
        log_z_i : scalar
            Target log10(Z/Zsun) for this age bin.
        ssp_flux_at_age_i : array, shape (n_met, n_wave)
            SSP flux at all metallicities for this age bin.

        Returns
        -------
        array, shape (n_wave,)
            Interpolated flux.
        """
        log_z_c = jnp.clip(log_z_i, ssp_lgmet[0], ssp_lgmet[-1])
        idx = jnp.clip(
            jnp.searchsorted(ssp_lgmet, log_z_c) - 1,
            0,
            len(ssp_lgmet) - 2,
        )
        frac = (log_z_c - ssp_lgmet[idx]) / (ssp_lgmet[idx + 1] - ssp_lgmet[idx])
        return (1.0 - frac) * ssp_flux_at_age_i[idx] + frac * ssp_flux_at_age_i[idx + 1]

    # ssp_flux is (n_met, n_age, n_wave); transpose to (n_age, n_met, n_wave)
    # so vmap over the leading (age) axis pairs each age with its metallicity
    ssp_flux_by_age = jnp.transpose(ssp_flux, (1, 0, 2))  # (n_age, n_met, n_wave)
    return jax.vmap(_interp_one_age)(log_z_per_age, ssp_flux_by_age)


@jax.jit
def interpolate_mass_remaining_evolving(
    ssp_mass_remaining: jnp.ndarray,
    ssp_lgmet: jnp.ndarray,
    log_z_per_age: jnp.ndarray,
) -> jnp.ndarray:
    """Interpolate mass-remaining with a different metallicity per age bin.

    Parameters
    ----------
    ssp_mass_remaining : array, shape (n_met, n_age)
        Fraction of formed mass surviving at each age and metallicity.
    ssp_lgmet : array, shape (n_met,)
        Log10(Z/Zsun) grid.
    log_z_per_age : array, shape (n_age,)
        Target log10(Z/Zsun) at each age bin.

    Returns
    -------
    array, shape (n_age,)
        Interpolated mass-remaining fraction per age bin.
    """
    def _interp_one_age(log_z_i, mr_at_age_i):
        log_z_c = jnp.clip(log_z_i, ssp_lgmet[0], ssp_lgmet[-1])
        idx = jnp.clip(
            jnp.searchsorted(ssp_lgmet, log_z_c) - 1,
            0,
            len(ssp_lgmet) - 2,
        )
        frac = (log_z_c - ssp_lgmet[idx]) / (ssp_lgmet[idx + 1] - ssp_lgmet[idx])
        return (1.0 - frac) * mr_at_age_i[idx] + frac * mr_at_age_i[idx + 1]

    # ssp_mass_remaining is (n_met, n_age); transpose to (n_age, n_met)
    mr_by_age = jnp.transpose(ssp_mass_remaining, (1, 0))  # (n_age, n_met)
    return jax.vmap(_interp_one_age)(log_z_per_age, mr_by_age)


@jax.jit
def compute_log_z_evolving(
    ssp_lg_age_gyr: jnp.ndarray,
    log_z_initial: float,
    log_z_final: float,
    t_universe_gyr: float,
) -> jnp.ndarray:
    """Compute per-age-bin metallicity from a linear-in-log ramp.

    The metallicity evolves linearly in log(Z/Zsun) space:

        log_z(t_lookback) = log_z_final + (log_z_initial - log_z_final)
                            * t_lookback / t_universe

    where t_lookback=0 is today (log_z_final) and t_lookback=t_universe
    is the oldest stars (log_z_initial). SSP ages are lookback times.

    Parameters
    ----------
    ssp_lg_age_gyr : array, shape (n_age,)
        Log10(age/Gyr) of SSP templates.
    log_z_initial : float
        Metallicity of the oldest stars (at t_lookback = t_universe),
        in log10(Z/Zsun) internally (absolute log10(Z)).
    log_z_final : float
        Metallicity at present day (t_lookback = 0), in log10(Z).
    t_universe_gyr : float
        Age of the universe at the observed redshift (Gyr).

    Returns
    -------
    array, shape (n_age,)
        log10(Z) at each SSP age bin.
    """
    age_gyr = 10.0 ** ssp_lg_age_gyr
    # Clamp lookback time to [0, t_universe] so extrapolation is safe
    t_frac = jnp.clip(age_gyr / t_universe_gyr, 0.0, 1.0)
    return log_z_final + (log_z_initial - log_z_final) * t_frac


@jax.jit
def interpolate_mass_remaining(
    ssp_mass_remaining: jnp.ndarray, ssp_lgmet: jnp.ndarray, log_z: float
) -> jnp.ndarray:
    """Interpolate mass-remaining fraction to a target metallicity.

    Parameters
    ----------
    ssp_mass_remaining : array, shape (n_met, n_age)
        Fraction of formed mass surviving at each age and metallicity.
    ssp_lgmet : array, shape (n_met,)
        Log10(Z/Zsun) grid.
    log_z : float
        Target log10(Z/Zsun).

    Returns
    -------
    array, shape (n_age,)
        Interpolated mass-remaining fraction.
    """
    log_z_clamped = jnp.clip(log_z, ssp_lgmet[0], ssp_lgmet[-1])
    idx = jnp.searchsorted(ssp_lgmet, log_z_clamped) - 1
    idx = jnp.clip(idx, 0, len(ssp_lgmet) - 2)
    frac = (log_z_clamped - ssp_lgmet[idx]) / (ssp_lgmet[idx + 1] - ssp_lgmet[idx])
    return (1.0 - frac) * ssp_mass_remaining[idx] + frac * ssp_mass_remaining[idx + 1]


@jax.jit
def compute_surviving_mass(weights: jnp.ndarray, mass_remaining_at_met: jnp.ndarray) -> float:
    """Compute surviving stellar mass from CSP weights and mass-remaining.

    Parameters
    ----------
    weights : array, shape (n_age,)
        Mass formed per age bin (Msun) from compute_csp_weights.
    mass_remaining_at_met : array, shape (n_age,)
        Fraction of formed mass surviving at each age (from
        interpolate_mass_remaining).

    Returns
    -------
    float
        Total surviving stellar mass (Msun).
    """
    return jnp.sum(weights * mass_remaining_at_met)
