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
    ssp_flux: jnp.ndarray  # (n_met, n_age, n_wave) or future (n_met, n_alpha, n_age, n_wave)
    ssp_lg_age_gyr: jnp.ndarray
    ssp_lgmet: jnp.ndarray
    ssp_mass_remaining: jnp.ndarray | None = None
    # Future: ssp_alpha_fe grid for alpha-enhanced templates (Vazdekis+2015, MIST)
    # When available, ssp_flux becomes (n_met, n_alpha, n_age, n_wave) and
    # interpolation adds a third dimension. The current met_alpha_fe parameter
    # uses effective_metallicity() as an approximation for 2D grids.
    ssp_alpha_fe: jnp.ndarray | None = None


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

        alpha_fe = None
        if "ssp_alpha_fe" in f:
            alpha_fe = jnp.array(f["ssp_alpha_fe"][:])

        return SSPData(
            ssp_wave=jnp.array(f["ssp_wave"][:]),
            ssp_flux=jnp.array(f["ssp_flux"][:]),
            ssp_lg_age_gyr=jnp.array(f["ssp_lg_age_gyr"][:]),
            ssp_lgmet=jnp.array(f["ssp_lgmet"][:]),
            ssp_mass_remaining=mass_remaining,
            ssp_alpha_fe=alpha_fe,
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


def has_alpha_grid(ssp_data: SSPData) -> bool:
    """Check if SSP data includes an [alpha/Fe] grid dimension.

    When True, ssp_flux has shape (n_met, n_alpha, n_age, n_wave) and
    proper bilinear (Z, [α/Fe]) interpolation should be used instead of
    the effective_metallicity approximation.

    Parameters
    ----------
    ssp_data : SSPData
        Loaded SSP template data.

    Returns
    -------
    bool
        True if ssp_alpha_fe is present and ssp_flux is 4D.
    """
    return (
        ssp_data.ssp_alpha_fe is not None
        and ssp_data.ssp_flux.ndim == 4
    )


@jax.jit
def interpolate_met_alpha(
    ssp_flux: jnp.ndarray,
    ssp_lgmet: jnp.ndarray,
    ssp_alpha_fe: jnp.ndarray,
    log_z: float,
    alpha_fe: float,
) -> jnp.ndarray:
    """Bilinear interpolation in (metallicity, [α/Fe]) for 4D SSP grids.

    This is the correct approach when alpha-enhanced SSP templates are
    available (e.g., sMILES, BPASS v2.3, α-MC).  It replaces the
    ``effective_metallicity()`` approximation, which is only valid when
    α-enhanced templates are NOT available.

    Parameters
    ----------
    ssp_flux : array, shape (n_met, n_alpha, n_age, n_wave)
        SSP flux on the full (Z, [α/Fe]) grid.
    ssp_lgmet : array, shape (n_met,)
        Log10(Z/Zsun) metallicity grid.  This is **total metallicity**
        [M/H], NOT [Fe/H], following sMILES/α-MC convention.
    ssp_alpha_fe : array, shape (n_alpha,)
        [α/Fe] grid values (e.g., [-0.2, 0.0, +0.2, +0.4, +0.6]).
    log_z : float
        Target total metallicity [M/H] = log10(Z/Zsun).
    alpha_fe : float
        Target [α/Fe] in dex.

    Returns
    -------
    array, shape (n_age, n_wave)
        Interpolated SSP flux at the target (Z, [α/Fe]).
    """
    # Metallicity index and fraction
    lz = jnp.clip(log_z, ssp_lgmet[0], ssp_lgmet[-1])
    iz = jnp.clip(jnp.searchsorted(ssp_lgmet, lz) - 1, 0, len(ssp_lgmet) - 2)
    fz = (lz - ssp_lgmet[iz]) / (ssp_lgmet[iz + 1] - ssp_lgmet[iz])

    # Alpha index and fraction
    afe = jnp.clip(alpha_fe, ssp_alpha_fe[0], ssp_alpha_fe[-1])
    ia = jnp.clip(jnp.searchsorted(ssp_alpha_fe, afe) - 1, 0, len(ssp_alpha_fe) - 2)
    fa = (afe - ssp_alpha_fe[ia]) / (ssp_alpha_fe[ia + 1] - ssp_alpha_fe[ia])

    # Bilinear: four corners → (n_age, n_wave)
    return (
        (1.0 - fz) * (1.0 - fa) * ssp_flux[iz, ia]
        + fz * (1.0 - fa) * ssp_flux[iz + 1, ia]
        + (1.0 - fz) * fa * ssp_flux[iz, ia + 1]
        + fz * fa * ssp_flux[iz + 1, ia + 1]
    )


@jax.jit
def interpolate_met_alpha_evolving(
    ssp_flux: jnp.ndarray,
    ssp_lgmet: jnp.ndarray,
    ssp_alpha_fe: jnp.ndarray,
    log_z_per_age: jnp.ndarray,
    alpha_fe_per_age: jnp.ndarray,
) -> jnp.ndarray:
    """Per-age bilinear interpolation in (Z, [α/Fe]) for time-evolving abundances.

    Each SSP age bin can have a different metallicity AND a different
    [α/Fe], enabling physically motivated chemical evolution where old
    stars are α-enhanced and young stars are solar-scaled.

    Parameters
    ----------
    ssp_flux : array, shape (n_met, n_alpha, n_age, n_wave)
        SSP flux on the full (Z, [α/Fe]) grid.
    ssp_lgmet : array, shape (n_met,)
        Log10(Z/Zsun) metallicity grid ([M/H], total metallicity).
    ssp_alpha_fe : array, shape (n_alpha,)
        [α/Fe] grid values.
    log_z_per_age : array, shape (n_age,)
        Target [M/H] at each SSP age bin.
    alpha_fe_per_age : array, shape (n_age,)
        Target [α/Fe] at each SSP age bin.

    Returns
    -------
    array, shape (n_age, n_wave)
        Interpolated SSP flux with per-age (Z, [α/Fe]).
    """

    def _interp_one_age(lz_i, afe_i, flux_at_age_i):
        # flux_at_age_i: (n_met, n_alpha, n_wave)
        lz = jnp.clip(lz_i, ssp_lgmet[0], ssp_lgmet[-1])
        iz = jnp.clip(jnp.searchsorted(ssp_lgmet, lz) - 1, 0, len(ssp_lgmet) - 2)
        fz = (lz - ssp_lgmet[iz]) / (ssp_lgmet[iz + 1] - ssp_lgmet[iz])

        afe = jnp.clip(afe_i, ssp_alpha_fe[0], ssp_alpha_fe[-1])
        ia = jnp.clip(jnp.searchsorted(ssp_alpha_fe, afe) - 1, 0, len(ssp_alpha_fe) - 2)
        fa = (afe - ssp_alpha_fe[ia]) / (ssp_alpha_fe[ia + 1] - ssp_alpha_fe[ia])

        return (
            (1.0 - fz) * (1.0 - fa) * flux_at_age_i[iz, ia]
            + fz * (1.0 - fa) * flux_at_age_i[iz + 1, ia]
            + (1.0 - fz) * fa * flux_at_age_i[iz, ia + 1]
            + fz * fa * flux_at_age_i[iz + 1, ia + 1]
        )

    # Transpose: (n_met, n_alpha, n_age, n_wave) → (n_age, n_met, n_alpha, n_wave)
    flux_by_age = jnp.transpose(ssp_flux, (2, 0, 1, 3))
    return jax.vmap(_interp_one_age)(log_z_per_age, alpha_fe_per_age, flux_by_age)


@jax.jit
def compute_alpha_fe_evolving(
    ssp_lg_age_gyr: jnp.ndarray,
    alpha_fe_old: float,
    alpha_fe_young: float,
    t_universe_gyr: float,
) -> jnp.ndarray:
    """Compute per-age [α/Fe] from a linear ramp in lookback time.

    Old stars (large lookback time) have high [α/Fe] (formed before
    Type Ia SNe enriched Fe).  Young stars have lower [α/Fe] (solar
    or sub-solar).  This is the standard chemical evolution prediction.

    The ramp is linear in lookback time::

        [α/Fe](t_lookback) = α_young + (α_old - α_young) * t_lookback / t_universe

    Parameters
    ----------
    ssp_lg_age_gyr : array, shape (n_age,)
        Log10(age/Gyr) of SSP templates (= lookback time for SSP bins).
    alpha_fe_old : float
        [α/Fe] of the oldest stars (at t_lookback = t_universe).
        Typically +0.3 to +0.5 for massive ellipticals.
    alpha_fe_young : float
        [α/Fe] at present day (t_lookback ≈ 0).
        Typically ~0.0 (solar) for disk galaxies.
    t_universe_gyr : float
        Age of the universe at the observed redshift (Gyr).

    Returns
    -------
    array, shape (n_age,)
        [α/Fe] at each SSP age bin.
    """
    age_gyr = 10.0**ssp_lg_age_gyr
    t_frac = jnp.clip(age_gyr / t_universe_gyr, 0.0, 1.0)
    return alpha_fe_young + (alpha_fe_old - alpha_fe_young) * t_frac


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


# ---------------------------------------------------------------------------
# Smooth metallicity interpolation (triweight kernel, DSPS-compatible)
# ---------------------------------------------------------------------------

_LGMET_LO = -4.0
_LGMET_HI = 0.5


@jax.jit
def _tw_cuml_kern(x, m, h):
    """Triweight kernel CDF (same as DSPS _tw_cuml_kern).

    Cumulative distribution of the triweight kernel with support |z| < 3.
    Returns 0 for z < -3, 1 for z > 3, smooth polynomial between.
    """
    z = (x - m) / h
    val = -5.0 * z**7 / 69984.0 + 7.0 * z**5 / 2592.0 - 35.0 * z**3 / 864.0 + 35.0 * z / 96.0 + 0.5
    val = jnp.where(z < -3.0, 0.0, val)
    val = jnp.where(z > 3.0, 1.0, val)
    return val


@jax.jit
def _get_lgmet_bin_edges(grid, lo=_LGMET_LO, hi=_LGMET_HI):
    """Bin edges from midpoints, matching DSPS convention.

    Uses half-spacing on each side, with outer edges clamped.
    """
    edges = jnp.concatenate([jnp.array([lo]), 0.5 * (grid[:-1] + grid[1:]), jnp.array([hi])])
    return edges


@jax.jit
def compute_lgmet_weights(log_z, ssp_lgmet, lgmet_scatter=0.1):
    """Metallicity weights via triweight CDF integration (DSPS-compatible).

    Integrates the triweight kernel CDF between bin edges, exactly
    matching the DSPS ``triweighted_histogram`` approach. The kernel
    has support at |z| < 3σ, giving smooth multi-bin weights.

    Parameters
    ----------
    log_z : float
        Target log10(Z/Zsun).
    ssp_lgmet : array (n_met,)
        SSP metallicity grid.
    lgmet_scatter : float
        Kernel bandwidth in dex (DSPS default: 0.1).

    Returns
    -------
    array (n_met,) — normalized weights summing to 1.
    """
    edges = _get_lgmet_bin_edges(ssp_lgmet)
    # CDF difference: probability mass in each bin
    # Note: CDF(lo) - CDF(hi) gives the mass between lo and hi
    # because _tw_cuml_kern returns CDF of the flipped kernel.
    # DSPS convention: _tw_cuml_kern(x, lo, sig) - _tw_cuml_kern(x, hi, sig)
    # where x is the galaxy metallicity, lo/hi are bin edges.
    cdf_lo = _tw_cuml_kern(log_z, edges[:-1], lgmet_scatter)
    cdf_hi = _tw_cuml_kern(log_z, edges[1:], lgmet_scatter)
    raw = cdf_lo - cdf_hi

    total = jnp.sum(raw)
    nearest = jnp.argmin(jnp.abs(ssp_lgmet - log_z))
    fallback = jnp.zeros_like(raw).at[nearest].set(1.0)
    return jnp.where(total > 0, raw / total, fallback)


@jax.jit
def interpolate_metallicity_smooth(ssp_flux, ssp_lgmet, log_z, lgmet_scatter=0.1):
    """Interpolate SSP flux using triweight kernel over metallicity.

    C2-continuous gradients. Matches DSPS approach (Hearin+2023).

    Parameters
    ----------
    ssp_flux : array (n_met, n_age, n_wave)
    ssp_lgmet : array (n_met,)
    log_z : float — target log10(Z/Zsun)
    lgmet_scatter : float — kernel bandwidth in dex

    Returns
    -------
    array (n_age, n_wave)
    """
    w = compute_lgmet_weights(log_z, ssp_lgmet, lgmet_scatter)
    return jnp.einsum("m,maw->aw", w, ssp_flux)


@jax.jit
def interpolate_metallicity_smooth_evolving(ssp_flux, ssp_lgmet, log_z_per_age, lgmet_scatter=0.1):
    """Triweight metallicity interpolation with per-age Z.

    Parameters
    ----------
    ssp_flux : array (n_met, n_age, n_wave)
    ssp_lgmet : array (n_met,)
    log_z_per_age : array (n_age,) — per-bin log10(Z/Zsun)
    lgmet_scatter : float

    Returns
    -------
    array (n_age, n_wave)
    """

    def _one_age(log_z_i, flux_at_age_i):
        w = compute_lgmet_weights(log_z_i, ssp_lgmet, lgmet_scatter)
        return jnp.einsum("m,mw->w", w, flux_at_age_i)

    flux_by_age = jnp.transpose(ssp_flux, (1, 0, 2))
    return jax.vmap(_one_age)(log_z_per_age, flux_by_age)


@jax.jit
def interpolate_mass_remaining_smooth(ssp_mass_remaining, ssp_lgmet, log_z, lgmet_scatter=0.1):
    """Smooth mass-remaining interpolation using triweight kernel."""
    w = compute_lgmet_weights(log_z, ssp_lgmet, lgmet_scatter)
    return jnp.einsum("m,ma->a", w, ssp_mass_remaining)


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
    age_gyr = 10.0**ssp_lg_age_gyr
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
