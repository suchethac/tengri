# SPDX-License-Identifier: BSD-3-Clause
"""Cosmology utilities backed by DSPS.

Thin wrappers around dsps.cosmology.flat_wcdm (Hearin+ JAX-based flat w0-wa-CDM).
All functions accept either a CosmoParams object or convenience h0/om0 kwargs.
All distances are returned in cm unless otherwise noted (e.g., _mpc suffix).

This module replaces the previous local quadrature implementation with DSPS,
which uses higher-order numerical integration and is fully JIT-compatible.
dsps and blackjax are imported inside functions (lazy imports) to defer the
allocation of device buffers at import time, allowing a bare `import tengri`
to succeed on float64-less backends (jax-mps); the first use of a dsps or
blackjax path still fails loudly, which is the acceptable failure mode.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from tengri.utils.physics_constants import MPC_CM


class CosmoParams(NamedTuple):
    """Field-for-field mirror of dsps.cosmology.flat_wcdm.CosmoParams.

    A flat w0-wa CDM cosmology defined by four parameters.
    dsps functions receive these fields positionally in this order.

    Attributes
    ----------
    Om0 : float
        Matter density fraction Ω_m at z=0.
    w0 : float
        Dark energy equation of state at z=0.
    wa : float
        Dark energy equation of state parameter (1+w = (1+w0) * a^wa).
    h : float
        Hubble parameter h = H0 / (100 km/s/Mpc).
    """

    Om0: float
    w0: float
    wa: float
    h: float


# Planck 2018 cosmology (tengri default).
# Values from Planck Collaboration 2020, A&A 641, A6 (TT,TE,EE+lowE+lensing):
#   H0 = 67.66 km/s/Mpc → h = 0.6766
#   Om0 = 0.30966
# These also match astropy.cosmology.Planck18: see #401 for the drift fix.
PLANCK18 = CosmoParams(Om0=0.30966, w0=-1.0, wa=0.0, h=0.6766)
DEFAULT_COSMO = PLANCK18

# Backward-compat scalar defaults (for positional arg parsing).
DEFAULT_H0 = 67.66  # km/s/Mpc: matches PLANCK18.h × 100
DEFAULT_OM0 = 0.30966

__all__ = [
    "DEFAULT_COSMO",
    "DEFAULT_H0",
    "DEFAULT_OM0",
    "PLANCK18",
    "CosmoParams",
    "age_at_z",
    "age_at_z0",
    "age_at_z0_host",
    "angular_diameter_distance",
    "angular_diameter_distance_mpc",
    "arcsec_per_kpc",
    "comoving_distance",
    "comoving_distance_mpc",
    "comoving_volume_element",
    "cosmo_from_astropy",
    "distance_modulus",
    "kpc_per_arcsec",
    "lookback_time",
    "luminosity_distance",
    "luminosity_distance_mpc",
    "z_at_cosmic_time",
    "z_at_lookback_time",
]


def __getattr__(name: str):
    """Lazy load PLANCK15 and WMAP5 from dsps.cosmology on access (PEP 562).

    These are vendored here as CosmoParams objects when accessed, so a bare
    import tengri does not pull in dsps.cosmology.

    Parameters
    ----------
    name : str
        The attribute name.

    Returns
    -------
    CosmoParams or other
        For PLANCK15 and WMAP5, returns a CosmoParams object with the
        corresponding values from dsps.cosmology.

    Raises
    ------
    AttributeError
        If the name is not PLANCK15 or WMAP5.
    """
    if name == "PLANCK15":
        from dsps.cosmology import PLANCK15 as _dsps_planck15

        return CosmoParams(*_dsps_planck15)
    if name == "WMAP5":
        from dsps.cosmology import WMAP5 as _dsps_wmap5

        return CosmoParams(*_dsps_wmap5)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


#: Names served lazily by :func:`__getattr__`; they are not in ``__all__`` because they
#: do not exist at import time (#2276), so :func:`__dir__` lists them for discovery.
_LAZY_NAMES: tuple[str, ...] = ("PLANCK15", "WMAP5")


def __dir__() -> list[str]:
    return sorted([*globals(), *_LAZY_NAMES])


def cosmo_from_astropy(astropy_cosmo) -> CosmoParams:
    """Convert an :mod:`astropy.cosmology` object to DSPS :class:`CosmoParams`.

    Provides build-time ergonomics for users who think in astropy terms.
    The returned :class:`CosmoParams` is the JIT-safe form tengri stores
    internally; astropy objects are heavy and not JIT-compatible, so this
    helper is the boundary between the two worlds.

    Supports flat cosmologies only: :class:`astropy.cosmology.FlatLambdaCDM`
    (``w0=-1``, ``wa=0``) and :class:`astropy.cosmology.Flatw0waCDM`.
    Non-flat cosmologies raise :class:`ValueError`; DSPS's underlying
    ``flat_wcdm`` engine has no support for them.

    Parameters
    ----------
    astropy_cosmo : astropy.cosmology.FLRW
        Any astropy cosmology object with attributes ``Om0`` (Ω_m at z=0),
        ``H0`` (Hubble constant with units), and optionally ``w0`` and
        ``wa`` for the w₀wₐCDM family.

    Returns
    -------
    CosmoParams
        Flat w₀wₐCDM dataclass with ``Om0``, ``h``, ``w0``, ``wa``.

    Raises
    ------
    ValueError
        If the cosmology is non-flat (Ω_de + Ω_m ≠ 1).

    Examples
    --------
    >>> from astropy.cosmology import Planck18
    >>> from tengri.cosmology import cosmo_from_astropy
    >>> cp = cosmo_from_astropy(Planck18)
    >>> abs(cp.h - 0.6766) < 1e-4
    True

    Custom dark-energy equation of state via :class:`Flatw0waCDM`:

    >>> from astropy.cosmology import Flatw0waCDM  # doctest: +SKIP
    >>> de = Flatw0waCDM(H0=70, Om0=0.3, w0=-0.95, wa=-0.05)  # doctest: +SKIP
    >>> cp = cosmo_from_astropy(de)  # doctest: +SKIP
    >>> cp.w0, cp.wa  # doctest: +SKIP
    (-0.95, -0.05)

    Notes
    -----
    Optional import: ``astropy`` is not a hard dependency of tengri.
    This function only needs astropy installed on the caller's side
    when it's actually invoked. The forward-model JIT path never
    imports astropy.
    """
    # Flat-only guard: DSPS doesn't support non-flat cosmologies.
    # Use astropy's curvature density Ok0 (0 for any flat cosmology,
    # regardless of how Ode0 splits between dark energy / neutrinos /
    # photons in Planck18, WMAP9, etc.).
    ok0 = float(getattr(astropy_cosmo, "Ok0", 0.0))
    if abs(ok0) > 1e-6:
        raise ValueError(
            f"DSPS supports flat cosmologies only; got Ok0 = {ok0:.6f}. "
            f"Use astropy.cosmology.FlatLambdaCDM or FlatwCDM / Flatw0waCDM."
        )
    om0 = float(astropy_cosmo.Om0)
    h = float(astropy_cosmo.H0.value) / 100.0
    w0 = float(getattr(astropy_cosmo, "w0", -1.0))
    wa = float(getattr(astropy_cosmo, "wa", 0.0))
    return CosmoParams(Om0=om0, w0=w0, wa=wa, h=h)


def _resolve_cosmo(
    cosmo: CosmoParams | None = None,
    h0: float | None = None,
    om0: float | None = None,
    w0: float | None = None,
    wa: float | None = None,
) -> CosmoParams:
    """Convert flexible cosmology inputs to CosmoParams.

    Priority: cosmo object > scalar kwargs > PLANCK18 defaults.

    Parameters
    ----------
    cosmo : CosmoParams or object with Om0, w0, wa, h attributes, optional
        Cosmology parameter set. Accepts any object with Om0, w0, wa, h
        attributes (e.g., a dsps.cosmology.flat_wcdm.CosmoParams).
    h0 : float, optional
        Hubble constant in km/s/Mpc. Converted to h = H0/100.
    om0 : float, optional
        Matter density parameter Ω_m.
    w0 : float, optional
        Dark-energy equation-of-state at z=0. Default -1.0 (ΛCDM).
    wa : float, optional
        Dark-energy equation-of-state evolution. Default 0.0 (ΛCDM).

    Returns
    -------
    CosmoParams

    Raises
    ------
    ValueError
        If both ``cosmo`` and any scalar kwarg are provided.
    """
    scalar_kwargs = (h0, om0, w0, wa)
    if cosmo is not None and any(v is not None for v in scalar_kwargs):
        raise ValueError("Pass either cosmo or scalar kwargs (h0/om0/w0/wa), not both")
    if cosmo is not None:
        return cosmo
    if any(v is not None for v in scalar_kwargs):
        return CosmoParams(
            Om0=om0 if om0 is not None else DEFAULT_OM0,
            w0=w0 if w0 is not None else -1.0,
            wa=wa if wa is not None else 0.0,
            h=(h0 / 100.0) if h0 is not None else DEFAULT_COSMO.h,
        )
    return DEFAULT_COSMO


def _cosmo_from_args(
    h0: float | None,
    om0: float | None,
    cosmo: CosmoParams | None,
) -> CosmoParams:
    """Resolve the ``(h0, om0, cosmo)`` convention shared by every wrapper below.

    Positional ``h0``/``om0`` take priority: when either is given, a
    keyword ``cosmo`` is ignored (never an error). With neither, falls
    back to the module defaults via :func:`_resolve_cosmo`.
    """
    if h0 is not None or om0 is not None:
        cosmo = None
    return _resolve_cosmo(cosmo=cosmo, h0=h0, om0=om0)


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

    At z=0, returns 10 pc (the standard optical absolute-magnitude distance
    convention) to enable finite L_ν → F_ν conversion in observation models.

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
        Luminosity distance in cm. At z=0, returns 10 pc (3.086e19 cm).
    """
    from dsps.cosmology.flat_wcdm import luminosity_distance_to_z

    c = _cosmo_from_args(h0, om0, cosmo)
    dl_mpc = luminosity_distance_to_z(z, c.Om0, c.w0, c.wa, c.h)
    # At z=0, use 10 pc (1e-5 Mpc) for the optical absolute-magnitude convention
    dl_mpc = jnp.where(z <= 0.0, 1e-5, dl_mpc)
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
    from dsps.cosmology.flat_wcdm import luminosity_distance_to_z

    c = _cosmo_from_args(h0, om0, cosmo)
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
    from dsps.cosmology.flat_wcdm import comoving_distance_to_z

    c = _cosmo_from_args(h0, om0, cosmo)
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
    from dsps.cosmology.flat_wcdm import comoving_distance_to_z

    c = _cosmo_from_args(h0, om0, cosmo)
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
    from dsps.cosmology.flat_wcdm import angular_diameter_distance_to_z

    c = _cosmo_from_args(h0, om0, cosmo)
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
    from dsps.cosmology.flat_wcdm import angular_diameter_distance_to_z

    c = _cosmo_from_args(h0, om0, cosmo)
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
    from dsps.cosmology.flat_wcdm import distance_modulus_to_z

    c = _cosmo_from_args(h0, om0, cosmo)
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
    from dsps.cosmology.flat_wcdm import lookback_to_z

    c = _cosmo_from_args(h0, om0, cosmo)
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
    from dsps.cosmology.flat_wcdm import age_at_z as _dsps_age_at_z

    c = _cosmo_from_args(h0, om0, cosmo)
    # DSPS age_at_z always returns array (minimum shape (1,))
    result = _dsps_age_at_z(z, c.Om0, c.w0, c.wa, c.h)
    # Return scalar if input was scalar
    return result[0] if jnp.ndim(z) == 0 else result


def age_at_z0_host(cosmo: CosmoParams = DEFAULT_COSMO) -> float:
    """Age of universe at z=0 (present day) in Gyr, computed on host.

    Pure-numpy implementation of dsps.cosmology.flat_wcdm.age_at_z0,
    designed to be called at module import time without triggering
    DSPS's float64 device buffer allocation on float64-less backends
    (jax-mps, MLX). Returns numerically identical results to DSPS.

    Uses 512 log-spaced redshift nodes and trapezoidal integration,
    exactly matching DSPS's integration strategy.

    Parameters
    ----------
    cosmo : CosmoParams, optional
        Cosmology parameters (default: PLANCK18).

    Returns
    -------
    float
        Age of universe in Gyr.

    Notes
    -----
    This function avoids all JAX and DSPS imports, making it safe
    to call at module scope for deferred DSPS initialization.
    """
    import numpy as np

    # DSPS constants: MPC in cm, YEAR in seconds
    MPC = 3.08567758149e24
    YEAR = 31556925.2

    # 512 log-spaced redshift nodes: logspace(0, 3, 512) - 1.0 = [0, ..., 999]
    z_nodes = np.logspace(0.0, 3.0, 512) - 1.0

    # Cosmological parameters
    Om0, w0, wa, h = cosmo.Om0, cosmo.w0, cosmo.wa, cosmo.h

    # Dark energy density at z=0 (matter-dominated assumption for flat LCDM)
    Ode0 = 1.0 - Om0

    # Compute E(z) = H(z) / H0 at each node
    # E(z) = sqrt(Om0*(1+z)^3 + Ode0*rho_de_z(z))
    # where rho_de_z(z) = a^(-3*(1+w0+wa)) * exp(-3*wa*(1-a)), a = 1/(1+z)
    a_nodes = 1.0 / (1.0 + z_nodes)
    rho_de_z = a_nodes ** (-3.0 * (1.0 + w0 + wa)) * np.exp(-3.0 * wa * (1.0 - a_nodes))
    E_z = np.sqrt(Om0 * (1.0 + z_nodes) ** 3.0 + Ode0 * rho_de_z)

    # Integrand: dt/dz = 1 / (H(z) * (1+z)) [in units where H0=1]
    integrand = 1.0 / (E_z * (1.0 + z_nodes))

    # Trapezoidal rule integration: integral 1/(E(z)*(1+z)) dz
    # Result is proportional to age / H0
    integrated = np.trapezoid(integrand, z_nodes)

    # Multiply by Hubble time (in Gyr) to get age in Gyr
    # hubble_time = 1 / H0 = 1 / (100*h km/s/Mpc) in Gyr units
    # The constant 1e-16 converts the MPC/YEAR ratio to Gyr:
    # 1e-16 * MPC / YEAR / h = Hubble time in Gyr
    hubble_time_gyr = 1e-16 * MPC / YEAR / h
    age_gyr = integrated * hubble_time_gyr

    return float(age_gyr)


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
    from dsps.cosmology.flat_wcdm import age_at_z0 as _dsps_age_at_z0

    c = _cosmo_from_args(h0, om0, cosmo)
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
    from dsps.cosmology.flat_wcdm import differential_comoving_volume

    c = _cosmo_from_args(h0, om0, cosmo)
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
    from dsps.cosmology.flat_wcdm import angular_diameter_distance_to_z

    c = _cosmo_from_args(h0, om0, cosmo)
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
    from dsps.cosmology.flat_wcdm import angular_diameter_distance_to_z

    c = _cosmo_from_args(h0, om0, cosmo)
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
    from dsps.cosmology.flat_wcdm import age_at_z as _dsps_age_at_z

    c = _cosmo_from_args(h0, om0, cosmo)

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
    from dsps.cosmology.flat_wcdm import age_at_z, age_at_z0

    c = _cosmo_from_args(h0, om0, cosmo)

    # Build lookup table: z_grid → t_lookback_grid (increasing)
    z_grid = jnp.linspace(0.0, z_max, n_grid)
    t0 = age_at_z0(c.Om0, c.w0, c.wa, c.h)
    t_age = age_at_z(z_grid, c.Om0, c.w0, c.wa, c.h)
    t_lookback_grid = t0 - t_age  # increasing with z

    return jnp.interp(t_lookback_gyr, t_lookback_grid, z_grid)
