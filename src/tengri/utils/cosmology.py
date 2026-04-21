"""Cosmology utilities backed by DSPS.

Thin wrappers around dsps.cosmology.flat_wcdm (Hearin+ JAX-based flat w0-wa-CDM).
All functions accept either a CosmoParams object or convenience h0/om0 kwargs.
All distances are returned in cm unless otherwise noted (e.g., _mpc suffix).

This module replaces the previous local quadrature implementation with DSPS,
which uses higher-order numerical integration and is fully JIT-compatible.
"""

from __future__ import annotations

import jax.numpy as jnp
from dsps.cosmology import PLANCK15, WMAP5, CosmoParams
from dsps.cosmology.flat_wcdm import (
    age_at_z as _dsps_age_at_z,
    age_at_z0 as _dsps_age_at_z0,
    angular_diameter_distance_to_z,
    comoving_distance_to_z,
    differential_comoving_volume,
    distance_modulus_to_z,
    lookback_to_z,
    luminosity_distance_to_z,
)

from tengri.utils.physics_constants import MPC_CM

# Planck 2018 cosmology (tengri default)
PLANCK18 = CosmoParams(Om0=0.315, w0=-1.0, wa=0.0, h=0.674)
DEFAULT_COSMO = PLANCK18

# Backward-compat scalar defaults (for positional arg parsing)
DEFAULT_H0 = 67.4  # km/s/Mpc
DEFAULT_OM0 = 0.315

__all__ = [
    "DEFAULT_COSMO",
    "DEFAULT_H0",
    "DEFAULT_OM0",
    "PLANCK15",
    "PLANCK18",
    "WMAP5",
    "CosmoParams",
    "age_at_z",
    "age_at_z0",
    "angular_diameter_distance",
    "angular_diameter_distance_mpc",
    "arcsec_per_kpc",
    "comoving_distance",
    "comoving_distance_mpc",
    "comoving_volume_element",
    "distance_modulus",
    "kpc_per_arcsec",
    "lookback_time",
    "luminosity_distance",
    "luminosity_distance_mpc",
    "z_at_cosmic_time",
    "z_at_lookback_time",
]


def _resolve_cosmo(
    cosmo: CosmoParams | None = None,
    h0: float | None = None,
    om0: float | None = None,
) -> CosmoParams:
    """Convert flexible cosmology inputs to CosmoParams.

    Priority: cosmo object > h0/om0 kwargs > PLANCK18 defaults.

    Parameters
    ----------
    cosmo : CosmoParams, optional
        Full DSPS cosmology parameter set.
    h0 : float, optional
        Hubble constant in km/s/Mpc. Converted to h = H0/100.
    om0 : float, optional
        Matter density parameter Ω_m.

    Returns
    -------
    CosmoParams

    Raises
    ------
    ValueError
        If both ``cosmo`` and ``h0``/``om0`` are provided.
    """
    if cosmo is not None and (h0 is not None or om0 is not None):
        raise ValueError("Pass either cosmo or h0/om0, not both")
    if cosmo is not None:
        return cosmo
    if h0 is not None or om0 is not None:
        return CosmoParams(
            Om0=om0 if om0 is not None else DEFAULT_OM0,
            w0=-1.0,
            wa=0.0,
            h=(h0 / 100.0) if h0 is not None else DEFAULT_COSMO.h,
        )
    return DEFAULT_COSMO


def luminosity_distance(
    z: float,
    h0: float | None = None,
    om0: float | None = None,
    *,
    cosmo: CosmoParams | None = None,
) -> float:
    """Luminosity distance in cm.

    Accepts positional h0/om0 or a keyword-only ``cosmo`` object.
    Priority: positional h0/om0 > keyword cosmo > defaults.

    Parameters
    ----------
    z : float
        Redshift.
    h0 : float, optional
        Hubble constant in km/s/Mpc. If provided, used for CosmoParams.
    om0 : float, optional
        Matter density parameter Ω_m. If provided, used for CosmoParams.
    cosmo : CosmoParams, optional
        Full cosmology parameter set (keyword-only). Overrides h0/om0 if both
        provided.

    Returns
    -------
    float
        Luminosity distance in cm.
    """
    # Handle positional args taking priority
    if h0 is not None or om0 is not None:
        cosmo = None  # Ignore cosmo if positional args provided
    c = _resolve_cosmo(cosmo=cosmo, h0=h0, om0=om0)
    dl_mpc = luminosity_distance_to_z(z, c.Om0, c.w0, c.wa, c.h)
    return dl_mpc * MPC_CM


def luminosity_distance_mpc(
    z: float,
    h0: float | None = None,
    om0: float | None = None,
    *,
    cosmo: CosmoParams | None = None,
) -> float:
    """Luminosity distance in Mpc.

    Parameters
    ----------
    z : float
        Redshift.
    h0 : float, optional
        Hubble constant in km/s/Mpc.
    om0 : float, optional
        Matter density parameter Ω_m.
    cosmo : CosmoParams, optional
        Full cosmology parameter set (keyword-only).

    Returns
    -------
    float
        Luminosity distance in Mpc.
    """
    if h0 is not None or om0 is not None:
        cosmo = None
    c = _resolve_cosmo(cosmo=cosmo, h0=h0, om0=om0)
    return luminosity_distance_to_z(z, c.Om0, c.w0, c.wa, c.h)


def comoving_distance(
    z: float,
    h0: float | None = None,
    om0: float | None = None,
    *,
    cosmo: CosmoParams | None = None,
) -> float:
    """Comoving distance in cm.

    Parameters
    ----------
    z : float
        Redshift.
    h0 : float, optional
        Hubble constant in km/s/Mpc.
    om0 : float, optional
        Matter density parameter Ω_m.
    cosmo : CosmoParams, optional
        Full cosmology parameter set (keyword-only).

    Returns
    -------
    float
        Comoving distance in cm.
    """
    if h0 is not None or om0 is not None:
        cosmo = None
    c = _resolve_cosmo(cosmo=cosmo, h0=h0, om0=om0)
    dc_mpc = comoving_distance_to_z(z, c.Om0, c.w0, c.wa, c.h)
    return dc_mpc * MPC_CM


def comoving_distance_mpc(
    z: float,
    h0: float | None = None,
    om0: float | None = None,
    *,
    cosmo: CosmoParams | None = None,
) -> float:
    """Comoving distance in Mpc.

    Parameters
    ----------
    z : float
        Redshift.
    h0 : float, optional
        Hubble constant in km/s/Mpc.
    om0 : float, optional
        Matter density parameter Ω_m.
    cosmo : CosmoParams, optional
        Full cosmology parameter set (keyword-only).

    Returns
    -------
    float
        Comoving distance in Mpc.
    """
    if h0 is not None or om0 is not None:
        cosmo = None
    c = _resolve_cosmo(cosmo=cosmo, h0=h0, om0=om0)
    return comoving_distance_to_z(z, c.Om0, c.w0, c.wa, c.h)


def angular_diameter_distance(
    z: float,
    h0: float | None = None,
    om0: float | None = None,
    *,
    cosmo: CosmoParams | None = None,
) -> float:
    """Angular diameter distance in cm.

    Parameters
    ----------
    z : float
        Redshift.
    h0 : float, optional
        Hubble constant in km/s/Mpc.
    om0 : float, optional
        Matter density parameter Ω_m.
    cosmo : CosmoParams, optional
        Full cosmology parameter set (keyword-only).

    Returns
    -------
    float
        Angular diameter distance in cm.
    """
    if h0 is not None or om0 is not None:
        cosmo = None
    c = _resolve_cosmo(cosmo=cosmo, h0=h0, om0=om0)
    da_mpc = angular_diameter_distance_to_z(z, c.Om0, c.w0, c.wa, c.h)
    return da_mpc * MPC_CM


def angular_diameter_distance_mpc(
    z: float,
    h0: float | None = None,
    om0: float | None = None,
    *,
    cosmo: CosmoParams | None = None,
) -> float:
    """Angular diameter distance in Mpc.

    Parameters
    ----------
    z : float
        Redshift.
    h0 : float, optional
        Hubble constant in km/s/Mpc.
    om0 : float, optional
        Matter density parameter Ω_m.
    cosmo : CosmoParams, optional
        Full cosmology parameter set (keyword-only).

    Returns
    -------
    float
        Angular diameter distance in Mpc.
    """
    if h0 is not None or om0 is not None:
        cosmo = None
    c = _resolve_cosmo(cosmo=cosmo, h0=h0, om0=om0)
    return angular_diameter_distance_to_z(z, c.Om0, c.w0, c.wa, c.h)


def distance_modulus(
    z: float,
    h0: float | None = None,
    om0: float | None = None,
    *,
    cosmo: CosmoParams | None = None,
) -> float:
    """Distance modulus in magnitudes.

    Calculated as μ = 5 log10(d_L_pc) - 5 where d_L_pc is luminosity distance
    in parsecs.

    Parameters
    ----------
    z : float
        Redshift.
    h0 : float, optional
        Hubble constant in km/s/Mpc.
    om0 : float, optional
        Matter density parameter Ω_m.
    cosmo : CosmoParams, optional
        Full cosmology parameter set (keyword-only).

    Returns
    -------
    float
        Distance modulus in magnitudes.
    """
    if h0 is not None or om0 is not None:
        cosmo = None
    c = _resolve_cosmo(cosmo=cosmo, h0=h0, om0=om0)
    return distance_modulus_to_z(z, c.Om0, c.w0, c.wa, c.h)


def lookback_time(
    z: float,
    h0: float | None = None,
    om0: float | None = None,
    *,
    cosmo: CosmoParams | None = None,
) -> float:
    """Lookback time to redshift z in Gyr.

    Lookback time is the time since the universe had redshift z. At z=0,
    lookback time is 0 (by definition).

    Parameters
    ----------
    z : float
        Redshift.
    h0 : float, optional
        Hubble constant in km/s/Mpc.
    om0 : float, optional
        Matter density parameter Ω_m.
    cosmo : CosmoParams, optional
        Full cosmology parameter set (keyword-only).

    Returns
    -------
    float
        Lookback time in Gyr.
    """
    if h0 is not None or om0 is not None:
        cosmo = None
    c = _resolve_cosmo(cosmo=cosmo, h0=h0, om0=om0)
    return lookback_to_z(z, c.Om0, c.w0, c.wa, c.h)


def age_at_z(
    z: float,
    h0: float | None = None,
    om0: float | None = None,
    *,
    cosmo: CosmoParams | None = None,
) -> float:
    """Age of universe at redshift z in Gyr.

    Parameters
    ----------
    z : float
        Redshift (scalar or array).
    h0 : float, optional
        Hubble constant in km/s/Mpc.
    om0 : float, optional
        Matter density parameter Ω_m.
    cosmo : CosmoParams, optional
        Full cosmology parameter set (keyword-only).

    Returns
    -------
    float or array
        Age of universe in Gyr. Returns scalar if input is scalar, array if
        input is array.
    """
    if h0 is not None or om0 is not None:
        cosmo = None
    c = _resolve_cosmo(cosmo=cosmo, h0=h0, om0=om0)
    # DSPS age_at_z always returns array (minimum shape (1,))
    result = _dsps_age_at_z(z, c.Om0, c.w0, c.wa, c.h)
    # Return scalar if input was scalar
    return result[0] if jnp.ndim(z) == 0 else result


def age_at_z0(
    h0: float | None = None,
    om0: float | None = None,
    *,
    cosmo: CosmoParams | None = None,
) -> float:
    """Age of universe at z=0 (present day) in Gyr.

    Parameters
    ----------
    h0 : float, optional
        Hubble constant in km/s/Mpc.
    om0 : float, optional
        Matter density parameter Ω_m.
    cosmo : CosmoParams, optional
        Full cosmology parameter set (keyword-only).

    Returns
    -------
    float
        Age of universe in Gyr.
    """
    if h0 is not None or om0 is not None:
        cosmo = None
    c = _resolve_cosmo(cosmo=cosmo, h0=h0, om0=om0)
    return _dsps_age_at_z0(c.Om0, c.w0, c.wa, c.h)


def comoving_volume_element(
    z: float,
    h0: float | None = None,
    om0: float | None = None,
    *,
    cosmo: CosmoParams | None = None,
) -> float:
    """Differential comoving volume element at redshift z in Mpc³/sr.

    This is the comoving volume element per unit solid angle per unit
    redshift: dV_c / (dz dΩ).

    Parameters
    ----------
    z : float
        Redshift (scalar or array).
    h0 : float, optional
        Hubble constant in km/s/Mpc.
    om0 : float, optional
        Matter density parameter Ω_m.
    cosmo : CosmoParams, optional
        Full cosmology parameter set (keyword-only).

    Returns
    -------
    float or array
        Comoving volume element in Mpc³/sr. Returns scalar if input is scalar.
    """
    if h0 is not None or om0 is not None:
        cosmo = None
    c = _resolve_cosmo(cosmo=cosmo, h0=h0, om0=om0)
    # differential_comoving_volume requires array input; ensure z is array
    z_arr = jnp.atleast_1d(z)
    result = differential_comoving_volume(z_arr, c.Om0, c.w0, c.wa, c.h)
    # Return scalar if input was scalar
    return result[0] if jnp.ndim(z) == 0 else result


def arcsec_per_kpc(
    z: float,
    h0: float | None = None,
    om0: float | None = None,
    *,
    cosmo: CosmoParams | None = None,
) -> float:
    """Angular scale at redshift z in arcsec/kpc.

    Conversion factor from physical kpc to arcseconds using the angular
    diameter distance: arcsec/kpc = 206265 / (d_A_Mpc × 1000).

    Parameters
    ----------
    z : float
        Redshift.
    h0 : float, optional
        Hubble constant in km/s/Mpc.
    om0 : float, optional
        Matter density parameter Ω_m.
    cosmo : CosmoParams, optional
        Full cosmology parameter set (keyword-only).

    Returns
    -------
    float
        Angular scale in arcsec/kpc.
    """
    if h0 is not None or om0 is not None:
        cosmo = None
    c = _resolve_cosmo(cosmo=cosmo, h0=h0, om0=om0)
    da_mpc = angular_diameter_distance_to_z(z, c.Om0, c.w0, c.wa, c.h)
    # 206265 arcsec/radian, 1000 kpc/Mpc
    return 206265.0 / (da_mpc * 1000.0)


def kpc_per_arcsec(
    z: float,
    h0: float | None = None,
    om0: float | None = None,
    *,
    cosmo: CosmoParams | None = None,
) -> float:
    """Physical scale at redshift z in kpc/arcsec.

    Inverse of arcsec_per_kpc. Physical kpc per arcsecond on the sky.

    Parameters
    ----------
    z : float
        Redshift.
    h0 : float, optional
        Hubble constant in km/s/Mpc.
    om0 : float, optional
        Matter density parameter Ω_m.
    cosmo : CosmoParams, optional
        Full cosmology parameter set (keyword-only).

    Returns
    -------
    float
        Physical scale in kpc/arcsec.
    """
    if h0 is not None or om0 is not None:
        cosmo = None
    c = _resolve_cosmo(cosmo=cosmo, h0=h0, om0=om0)
    da_mpc = angular_diameter_distance_to_z(z, c.Om0, c.w0, c.wa, c.h)
    # Inverse of arcsec_per_kpc
    return (da_mpc * 1000.0) / 206265.0


def z_at_cosmic_time(
    t_gyr: float,
    h0: float | None = None,
    om0: float | None = None,
    *,
    cosmo: CosmoParams | None = None,
    z_max: float = 30.0,
    n_grid: int = 512,
) -> float:
    """Redshift at a given cosmic time (age of universe).

    Numerically inverts age_at_z(z) = t using a pre-built lookup table
    with linear interpolation. Useful for converting SFH time grids to
    redshift grids for CSFR reconstruction.

    Parameters
    ----------
    t_gyr : float or array
        Cosmic time (age of universe) in Gyr. Must be between 0 and
        age_at_z0. Values outside this range are clipped.
    h0 : float, optional
        Hubble constant in km/s/Mpc.
    om0 : float, optional
        Matter density parameter Ω_m.
    cosmo : CosmoParams, optional
        Full cosmology parameter set (keyword-only).
    z_max : float
        Maximum redshift for the lookup table (default: 30).
    n_grid : int
        Number of points in the lookup table (default: 512).

    Returns
    -------
    float or array
        Redshift corresponding to the given cosmic time. Returns z_max
        for t < age(z_max), and 0 for t >= age(0).
    """
    if h0 is not None or om0 is not None:
        cosmo = None
    c = _resolve_cosmo(cosmo=cosmo, h0=h0, om0=om0)

    # Build lookup table: z_grid → t_grid (decreasing in t)
    z_grid = jnp.linspace(0.0, z_max, n_grid)
    t_grid = _dsps_age_at_z(z_grid, c.Om0, c.w0, c.wa, c.h)

    # t_grid is decreasing (age decreases with z). Flip for interp.
    t_flip = t_grid[::-1]  # now increasing
    z_flip = z_grid[::-1]  # corresponding z (now decreasing)

    return jnp.interp(t_gyr, t_flip, z_flip)


def z_at_lookback_time(
    t_lookback_gyr: float,
    h0: float | None = None,
    om0: float | None = None,
    *,
    cosmo: CosmoParams | None = None,
    z_max: float = 30.0,
    n_grid: int = 512,
) -> float:
    """Redshift at a given lookback time.

    Numerically inverts lookback_time(z) = t using a pre-built lookup table
    with linear interpolation.

    Parameters
    ----------
    t_lookback_gyr : float or array
        Lookback time in Gyr. 0 = now, age_at_z0 = Big Bang.
    h0 : float, optional
        Hubble constant in km/s/Mpc.
    om0 : float, optional
        Matter density parameter Ω_m.
    cosmo : CosmoParams, optional
        Full cosmology parameter set (keyword-only).
    z_max : float
        Maximum redshift for the lookup table (default: 30).
    n_grid : int
        Number of points in the lookup table (default: 512).

    Returns
    -------
    float or array
        Redshift corresponding to the given lookback time. Returns 0
        for t_lookback=0, z_max for t_lookback >= lookback(z_max).
    """
    if h0 is not None or om0 is not None:
        cosmo = None
    c = _resolve_cosmo(cosmo=cosmo, h0=h0, om0=om0)

    # Build lookup table: z_grid → t_lookback_grid (increasing)
    z_grid = jnp.linspace(0.0, z_max, n_grid)
    t0 = _dsps_age_at_z0(c.Om0, c.w0, c.wa, c.h)
    t_age = _dsps_age_at_z(z_grid, c.Om0, c.w0, c.wa, c.h)
    t_lookback_grid = t0 - t_age  # increasing with z

    return jnp.interp(t_lookback_gyr, t_lookback_grid, z_grid)
