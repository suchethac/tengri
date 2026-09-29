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

from tengri._completion import curated_dir
from tengri._x64_hold import hold_x64_preference
from tengri.utils.physics_constants import (
    C_CGS,
    C_KM_S,
    G_GRAV,
    JULIAN_YEAR_S,
    K_BOLTZ_EV,
    KOMATSU_NU_SHAPE_INDEX,
    KOMATSU_NU_SHAPE_SCALE,
    MPC_CM,
    NEUTRINO_FERMI_DIRAC_CORRECTION,
    NU_TO_CMB_TEMP_RATIO,
    SIGMA_SB,
)


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
    Tcmb0 : float, optional
        CMB temperature at z=0 in Kelvin. Default: 2.7255 K (astropy Planck18).
    Neff : float, optional
        Effective number of relativistic species. Default: 3.046.
    m_nu_eV : tuple[float, float, float], optional
        Neutrino masses in eV (one per species). Default: (0, 0, 0.06) eV.
    """

    Om0: float
    w0: float
    wa: float
    h: float
    Tcmb0: float = 2.7255
    Neff: float = 3.046
    m_nu_eV: tuple[float, float, float] = (0.0, 0.0, 0.06)


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

# ============================================================================
# Fixed-node Gauss-Legendre quadrature for JAX-traceable integration
# ============================================================================


def _get_gl_nodes_weights_np(n: int = 512) -> tuple:
    """Compute Gauss-Legendre quadrature nodes and weights as plain numpy.

    Parameters
    ----------
    n : int
        Number of quadrature nodes (default: 512).

    Returns
    -------
    tuple of (nodes, weights)
        nodes: ndarray, shape (n,), quadrature points in [-1, 1]
        weights: ndarray, shape (n,), quadrature weights

    Notes
    -----
    Plain ``np.ndarray`` (float64), never a JAX array: a module-scope
    ``jnp.asarray(..., dtype=jnp.float64)`` is a *device* array materialized
    by ``import tengri`` itself, which crashes on float64-less backends
    (jax-mps) and is exactly the import-time allocation #2271/#1880 forbid.
    Converted to a JAX array only at use time (:func:`_comoving_distance_mpc_jax`,
    :func:`_age_gyr_jax`), without forcing a dtype, so it canonicalizes to
    whichever precision the caller's ``jax_enable_x64`` state has at that
    point rather than the state at import time.
    """
    from numpy.polynomial.legendre import leggauss

    return leggauss(n)


# Precompute Gauss-Legendre quadrature at module load time as plain numpy
# host arrays (never a JAX device array — see _get_gl_nodes_weights_np).
# 512 nodes for <1e-4 relative error at z ≤ 10.
_GL_NODES_512_NP, _GL_WEIGHTS_512_NP = _get_gl_nodes_weights_np(512)


# ============================================================================
# Radiation / massive-neutrino density (astropy.cosmology.Planck18 physics)
# ============================================================================


def _radiation_fields(cosmo) -> tuple[float, float, tuple[float, float, float]]:
    """Extract (Tcmb0 [K], Neff, m_nu_eV [eV]) from a cosmo-like object.

    Falls back to :class:`CosmoParams`'s declared field defaults for
    4-field objects that predate radiation support (e.g.
    ``dsps.cosmology.flat_wcdm.CosmoParams``, an ``(Om0, w0, wa, h)``
    tuple with no radiation attributes at all) — :func:`_resolve_cosmo`
    accepts any object with ``Om0``/``w0``/``wa``/``h`` attributes, so
    every reader of the radiation fields must tolerate their absence too.

    Parameters
    ----------
    cosmo : CosmoParams or cosmo-like object
        Cosmology parameter set.

    Returns
    -------
    tuple of (Tcmb0, Neff, m_nu_eV)
    """
    defaults = CosmoParams._field_defaults
    tcmb0 = getattr(cosmo, "Tcmb0", defaults["Tcmb0"])
    neff = getattr(cosmo, "Neff", defaults["Neff"])
    m_nu_eV = getattr(cosmo, "m_nu_eV", defaults["m_nu_eV"])
    return tcmb0, neff, m_nu_eV


def _nu_relative_density(z, neff: float, m_nu_eV: tuple[float, float, float], tcmb0: float):
    r"""Neutrino energy density relative to the photon energy density.

    Implements the Komatsu et al. (2011) fitting formula used by
    ``astropy.cosmology`` [1]_:

    .. math::

        f_\nu(z) = 0.2271\, N_{\rm eff}/n_\nu \sum_i
            \left[1 + \left(0.3173\, y_i(z)\right)^{1.83}\right]^{1/1.83}

        y_i(z) = \frac{m_{\nu,i} c^2}{k_B T_{\nu 0} (1+z)}

    where :math:`n_\nu` is the number of neutrino species (``len(m_nu_eV)``,
    3 for tengri's fixed 3-species convention), the sum runs over all
    species (a massless species has :math:`y_i \equiv 0`, contributing
    exactly 1 to the sum — algebraically identical to astropy's separate
    "massless" bucket), and :math:`T_{\nu 0} = 0.7138\, T_{\rm cmb0}`.

    Parameters
    ----------
    z : float or array_like
        Redshift (scalar or array; any shape).
    neff : float
        Effective number of relativistic neutrino species N_eff.
    m_nu_eV : tuple of float
        Neutrino rest masses [eV], one per species.
    tcmb0 : float
        CMB temperature at z=0 [K].

    Returns
    -------
    float or ndarray
        :math:`f_\nu(z)`, same shape as ``z``.

    Notes
    -----
    JIT/grad-safe: pure JAX operations on a fixed-length (3-species) array,
    no data-dependent branching or shapes. Safe for ``Tcmb0 = 0`` (radiation
    off): the neutrino temperature is then also 0, so ``m_nu / (k_B T_nu)``
    is guarded against 0/0 and returns ``y_i = 0`` (masses become
    irrelevant, matching astropy's "Tcmb0=0 turns off photons and
    neutrinos").

    References
    ----------
    .. [1] Komatsu, E., et al. (2011), "Seven-Year Wilkinson Microwave
           Anisotropy Probe (WMAP) Observations: Cosmological
           Interpretation", ApJS, 192, 18, eq. 26,
           https://doi.org/10.1088/0067-0049/192/2/18
    """
    z = jnp.asarray(z)
    n_nu = len(m_nu_eV)
    neff_per_nu = neff / n_nu
    tnu0 = NU_TO_CMB_TEMP_RATIO * tcmb0
    kt_nu0_ev = K_BOLTZ_EV * tnu0
    kt_nu0_ev_safe = jnp.where(kt_nu0_ev > 0.0, kt_nu0_ev, 1.0)
    m_nu = jnp.asarray(m_nu_eV)
    nu_y = jnp.where(kt_nu0_ev > 0.0, m_nu / kt_nu0_ev_safe, 0.0)
    curr_nu_y = nu_y / (1.0 + z)[..., None]
    rel_mass_per = (1.0 + (KOMATSU_NU_SHAPE_SCALE * curr_nu_y) ** KOMATSU_NU_SHAPE_INDEX) ** (
        1.0 / KOMATSU_NU_SHAPE_INDEX
    )
    rel_mass = jnp.sum(rel_mass_per, axis=-1)
    return NEUTRINO_FERMI_DIRAC_CORRECTION * neff_per_nu * rel_mass


def _photon_omega0(tcmb0: float, h: float) -> float:
    r"""Photon density fraction Ω_γ at z=0 from the CMB temperature.

    .. math::

        \Omega_\gamma = \frac{a_B T_{\rm cmb0}^4}{\rho_{\rm crit,0}},
        \qquad a_B = \frac{4\sigma_{\rm SB}}{c^3}, \qquad
        \rho_{\rm crit,0} = \frac{3 H_0^2}{8\pi G}

    Parameters
    ----------
    tcmb0 : float
        CMB temperature at z=0 [K].
    h : float
        Hubble parameter h = H0 / (100 km/s/Mpc).

    Returns
    -------
    float
        Ω_γ (dimensionless).

    Notes
    -----
    Matches ``astropy.cosmology.FLRW.Ogamma0`` (radiation constant over
    critical density) to ~5e-5 relative precision, the limit of
    :data:`~tengri.utils.physics_constants.G_GRAV`'s 4-significant-figure
    rounding — negligible against Ω_γ's ~0.2% share of E(z)² at z=10.
    JIT/grad-safe: pure JAX/Python arithmetic, no branching.
    """
    a_b_c2 = 4.0 * SIGMA_SB / C_CGS**3.0
    h0_cgs = h * 100.0 * 1.0e5 / MPC_CM
    rho_crit0 = 3.0 * h0_cgs**2.0 / (8.0 * jnp.pi * G_GRAV)
    return a_b_c2 * tcmb0**4.0 / rho_crit0


def _flat_density_params(cosmo) -> tuple[float, float]:
    """(Ω_γ0, Ω_de0) at z=0 for a flat cosmology.

    Ω_de0 is fixed by flatness: Ω_de0 = 1 - Ω_m0 - Ω_γ0 - Ω_ν0.

    Parameters
    ----------
    cosmo : CosmoParams or cosmo-like object
        Must expose ``Om0`` and ``h``; radiation fields are optional
        (see :func:`_radiation_fields`).

    Returns
    -------
    tuple of (ogamma0, ode0)
    """
    tcmb0, neff, m_nu_eV = _radiation_fields(cosmo)
    ogamma0 = _photon_omega0(tcmb0, cosmo.h)
    onu0 = ogamma0 * _nu_relative_density(0.0, neff, m_nu_eV, tcmb0)
    ode0 = 1.0 - cosmo.Om0 - ogamma0 - onu0
    return ogamma0, ode0


def _rho_de_z(z, w0: float, wa: float):
    r"""CPL dark-energy density relative to its z=0 value.

    .. math::

        \rho_{\rm de}(a)/\rho_{\rm de,0} = a^{-3(1+w_0+w_a)}
            \exp\left[-3 w_a (1-a)\right], \qquad a = 1/(1+z)

    Matches ``dsps.cosmology.flat_wcdm._rho_de_z`` exactly (Chevallier &
    Polarski 2001; Linder 2003), restoring the w0/wa generality of the
    DSPS-backed path this module replaces for the radiation-inclusive
    distance/age integrals below. For ΛCDM (w0=-1, wa=0) this is
    identically 1.

    Parameters
    ----------
    z : float or array_like
        Redshift.
    w0 : float
        Dark-energy equation-of-state parameter at z=0.
    wa : float
        Dark-energy equation-of-state evolution parameter.

    Returns
    -------
    float or ndarray

    Notes
    -----
    JIT/grad-safe: pure JAX arithmetic.
    """
    a = 1.0 / (1.0 + z)
    return a ** (-3.0 * (1.0 + w0 + wa)) * jnp.exp(-3.0 * wa * (1.0 - a))


def _e_of_z(z, cosmo, ogamma0: float, ode0: float):
    r"""Normalized Hubble parameter E(z) = H(z)/H0, flat cosmology.

    Implements astropy's ``FlatLambdaCDM.efunc`` formula, generalized to
    CPL dark energy to match ``dsps.cosmology.flat_wcdm``:

    .. math::

        E(z) = \sqrt{\Omega_m(1+z)^3 + \Omega_\gamma(1+z)^4
                    \left[1 + f_\nu(z)\right]
                    + \Omega_{\rm de}\,\rho_{\rm de}(z)/\rho_{\rm de,0}}

    Parameters
    ----------
    z : float or array_like
        Redshift (scalar or array).
    cosmo : CosmoParams or cosmo-like object
        Exposes ``Om0``, ``w0``, ``wa``; radiation fields optional.
    ogamma0 : float
        Photon density fraction Ω_γ at z=0 (from :func:`_flat_density_params`).
    ode0 : float
        Dark energy density fraction Ω_de at z=0 (from
        :func:`_flat_density_params`).

    Returns
    -------
    float or ndarray
        E(z) (dimensionless).

    Notes
    -----
    Radiation includes photons and massive neutrinos, following astropy's
    ``Planck18`` parameterization (Planck Collaboration 2020, A&A 641, A6).
    JIT/grad-safe: pure JAX operations, no Python-level branching.
    """
    tcmb0, neff, m_nu_eV = _radiation_fields(cosmo)
    zp1 = 1.0 + z
    or_z = ogamma0 * (1.0 + _nu_relative_density(z, neff, m_nu_eV, tcmb0))
    de_z = ode0 * _rho_de_z(z, cosmo.w0, cosmo.wa)
    return jnp.sqrt(cosmo.Om0 * zp1**3.0 + or_z * zp1**4.0 + de_z)


def _comoving_distance_mpc_jax(z: float, cosmo, ogamma0: float, ode0: float) -> float:
    r"""Comoving distance via Gauss-Legendre quadrature in JAX.

    Integrates:

    .. math::

        \chi(z) = \frac{c}{H_0} \int_0^z \frac{dz'}{E(z')}

    using fixed 512-node Gauss-Legendre quadrature for <1e-4 relative error.

    Transformation: u ∈ [-1, 1] → z' ∈ [0, z] via z' = z(1+u)/2.

    Parameters
    ----------
    z : float
        Redshift.
    cosmo : CosmoParams or cosmo-like object
        See :func:`_e_of_z`.
    ogamma0 : float
        Photon density fraction Ω_γ at z=0.
    ode0 : float
        Dark energy density fraction Ω_de at z=0.

    Returns
    -------
    float
        Comoving distance in Mpc.

    Notes
    -----
    JIT/grad-safe: pure JAX operations, no Python loops.
    """
    # Map Gauss-Legendre nodes from [-1, 1] to [0, z]. Converted to a JAX
    # array here (not at module scope) so the dtype follows the caller's
    # current jax_enable_x64 state rather than the state at import time.
    gl_nodes = jnp.asarray(_GL_NODES_512_NP)
    gl_weights = jnp.asarray(_GL_WEIGHTS_512_NP)
    # z' = z * (1 + u) / 2, so dz'/du = z/2
    z_prime = z * (1.0 + gl_nodes) / 2.0
    e_z_prime = _e_of_z(z_prime, cosmo, ogamma0, ode0)
    integrand = 1.0 / e_z_prime
    # Gauss-Legendre quadrature: integral = sum(w_i * f(u_i)) * (dz/2)
    integral = jnp.sum(gl_weights * integrand) * (z / 2.0)
    c_over_h0_mpc = C_KM_S / (100.0 * cosmo.h)
    return integral * c_over_h0_mpc


def _age_gyr_jax(z: float, cosmo, ogamma0: float, ode0: float) -> float:
    r"""Age of universe at redshift z via Gauss-Legendre quadrature in JAX.

    Integrates:

    .. math::
        t(z) = \frac{1}{H_0} \int_z^\infty \frac{dz'}{(1+z')E(z')}

    using the substitution :math:`a = 1/(1+z')`, :math:`a \in [0, a_z]`
    with :math:`a_z = 1/(1+z)`:

    .. math::
        t(z) = \frac{1}{H_0} \int_0^{a_z} \frac{da}{a\, E(z'(a))}

    on fixed 512-node Gauss-Legendre quadrature. The naive z'-linear
    substitution used through z_max=1000 concentrates nodes on the flat,
    negligible tail instead of the rapidly-varying region near z'~z,
    biasing age(z) low by a near-constant ~4e-4 Gyr across the whole z
    range (~0.09% at z=10, above the 1e-4 relative target); the
    a-substitution matches astropy to <1e-6 relative at every z tested.

    Parameters
    ----------
    z : float
        Redshift.
    cosmo : CosmoParams or cosmo-like object
        See :func:`_e_of_z`.
    ogamma0 : float
        Photon density fraction Ω_γ at z=0.
    ode0 : float
        Dark energy density fraction Ω_de at z=0.

    Returns
    -------
    float
        Age of universe in Gyr.

    Notes
    -----
    The GL nodes never land exactly on the a=0 endpoint (open interval),
    so the 1/a factor is always evaluated away from the a→0 limit where
    the integrand itself vanishes (E(z') ~ a^{-3/2} there); no epsilon
    guard is needed. JIT/grad-safe: pure JAX operations, no Python loops.
    """
    # Converted to a JAX array here (not at module scope), same rationale
    # as _comoving_distance_mpc_jax above.
    gl_nodes = jnp.asarray(_GL_NODES_512_NP)
    gl_weights = jnp.asarray(_GL_WEIGHTS_512_NP)
    a_z = 1.0 / (1.0 + z)
    a_nodes = a_z * (1.0 + gl_nodes) / 2.0
    z_prime = 1.0 / a_nodes - 1.0
    integrand = 1.0 / (a_nodes * _e_of_z(z_prime, cosmo, ogamma0, ode0))
    integral = jnp.sum(gl_weights * integrand) * (a_z / 2.0)
    hubble_time_gyr = MPC_CM / (100.0 * cosmo.h * 1.0e5 * JULIAN_YEAR_S * 1.0e9)
    return integral * hubble_time_gyr


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
        with hold_x64_preference():
            from dsps.cosmology import PLANCK15 as _dsps_planck15

        return CosmoParams(*_dsps_planck15)
    if name == "WMAP5":
        with hold_x64_preference():
            from dsps.cosmology import WMAP5 as _dsps_wmap5

        return CosmoParams(*_dsps_wmap5)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


#: Names served lazily by :func:`__getattr__`; they are not in ``__all__`` because they
#: do not exist at import time (#2276), so :func:`curated_dir` lists them for discovery.
_LAZY_NAMES: tuple[str, ...] = ("PLANCK15", "WMAP5")

_CURATED_DIR = tuple([*__all__, *_LAZY_NAMES])
__dir__ = curated_dir(_CURATED_DIR)


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
        Flat w₀wₐCDM dataclass with ``Om0``, ``h``, ``w0``, ``wa``, and the
        radiation fields ``Tcmb0``, ``Neff``, ``m_nu_eV`` carried over from
        the astropy object (``Tcmb0=0`` turns off both photons and
        neutrinos, matching astropy's own convention).

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

    defaults = CosmoParams._field_defaults
    tcmb0_attr = getattr(astropy_cosmo, "Tcmb0", None)
    tcmb0 = float(tcmb0_attr.value) if tcmb0_attr is not None else defaults["Tcmb0"]
    neff = float(getattr(astropy_cosmo, "Neff", defaults["Neff"]))
    m_nu_attr = getattr(astropy_cosmo, "m_nu", None)
    if m_nu_attr is None:
        m_nu_eV: tuple[float, float, float] = (0.0, 0.0, 0.0)
    else:
        import astropy.units as u

        m_nu_vals = [float(v) for v in m_nu_attr.to_value(u.eV)]
        m_nu_vals = [*m_nu_vals, 0.0, 0.0, 0.0][:3]
        m_nu_eV = (m_nu_vals[0], m_nu_vals[1], m_nu_vals[2])

    return CosmoParams(Om0=om0, w0=w0, wa=wa, h=h, Tcmb0=tcmb0, Neff=neff, m_nu_eV=m_nu_eV)


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

    Notes
    -----
    Pure JAX implementation using 512-node Gauss-Legendre quadrature.
    Includes radiation (photons + massive neutrinos) per
    astropy.cosmology.Planck18, matched to <1e-4 relative for D_L and
    age(z) at z ≤ 10 (#2517). JIT/grad-safe.
    """
    c = _cosmo_from_args(h0, om0, cosmo)
    ogamma0, ode0 = _flat_density_params(c)
    dc_mpc = _comoving_distance_mpc_jax(z, c, ogamma0, ode0)
    # Luminosity distance: D_L = (1+z) * D_c
    dl_mpc = (1.0 + z) * dc_mpc
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

    Notes
    -----
    Pure JAX implementation using 512-node Gauss-Legendre quadrature.
    Includes radiation (photons + massive neutrinos) per
    astropy.cosmology.Planck18, matched to <1e-4 relative for D_L and
    age(z) at z ≤ 10 (#2517). JIT/grad-safe.
    """
    c = _cosmo_from_args(h0, om0, cosmo)
    ogamma0, ode0 = _flat_density_params(c)
    dc_mpc = _comoving_distance_mpc_jax(z, c, ogamma0, ode0)
    # Luminosity distance: D_L = (1+z) * D_c
    dl_mpc = (1.0 + z) * dc_mpc
    # At z=0, return 1e-5 Mpc (10 pc) for optical absolute magnitude convention
    return jnp.where(z <= 0.0, 1e-5, dl_mpc)


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

    Notes
    -----
    Pure JAX implementation using 512-node Gauss-Legendre quadrature.
    Includes radiation (photons + massive neutrinos) per
    astropy.cosmology.Planck18 (#2517). JIT/grad-safe.
    """
    c = _cosmo_from_args(h0, om0, cosmo)
    ogamma0, ode0 = _flat_density_params(c)
    dc_mpc = _comoving_distance_mpc_jax(z, c, ogamma0, ode0)
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

    Notes
    -----
    Pure JAX implementation using 512-node Gauss-Legendre quadrature.
    Includes radiation (photons + massive neutrinos) per
    astropy.cosmology.Planck18 (#2517). JIT/grad-safe.
    """
    c = _cosmo_from_args(h0, om0, cosmo)
    ogamma0, ode0 = _flat_density_params(c)
    return _comoving_distance_mpc_jax(z, c, ogamma0, ode0)


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

    Notes
    -----
    Flat-FRW distance-duality relation D_A = D_C / (1+z) (Hogg 1999,
    astro-ph/9905116, eq. 18), routed through the same radiation-inclusive
    :func:`comoving_distance` as :func:`luminosity_distance` (#2517) rather
    than DSPS's radiation-free ``angular_diameter_distance_to_z`` — keeping
    D_A = D_L / (1+z)^2 exact regardless of Tcmb0/Neff/m_nu_eV.
    JIT/grad-safe: pure JAX operations.
    """
    c = _cosmo_from_args(h0, om0, cosmo)
    ogamma0, ode0 = _flat_density_params(c)
    dc_mpc = _comoving_distance_mpc_jax(z, c, ogamma0, ode0)
    return (dc_mpc / (1.0 + z)) * MPC_CM


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

    Notes
    -----
    See :func:`angular_diameter_distance` — routes through the
    radiation-inclusive comoving distance rather than DSPS (#2517).
    JIT/grad-safe: pure JAX operations.
    """
    c = _cosmo_from_args(h0, om0, cosmo)
    ogamma0, ode0 = _flat_density_params(c)
    dc_mpc = _comoving_distance_mpc_jax(z, c, ogamma0, ode0)
    return dc_mpc / (1.0 + z)


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

    Notes
    -----
    Derived from :func:`luminosity_distance_mpc` (radiation-inclusive,
    #2517) rather than DSPS's ``distance_modulus_to_z``.
    JIT/grad-safe: pure JAX operations.
    """
    dl_mpc = luminosity_distance_mpc(z, h0, om0, cosmo=cosmo)
    dl_pc = dl_mpc * 1.0e6
    return 5.0 * jnp.log10(dl_pc) - 5.0


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

    Notes
    -----
    t_lookback(z) = age_at_z0() - age_at_z(z), both radiation-inclusive
    (#2517) rather than DSPS's ``lookback_to_z``.
    JIT/grad-safe: pure JAX operations.
    """
    return age_at_z0(h0, om0, cosmo=cosmo) - age_at_z(z, h0, om0, cosmo=cosmo)


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

    Notes
    -----
    Pure JAX implementation using 512-node Gauss-Legendre quadrature.
    Includes radiation (photons + massive neutrinos) per
    astropy.cosmology.Planck18, matched to <1e-4 relative for z ≤ 10
    (#2517). JIT/grad-safe.
    """
    c = _cosmo_from_args(h0, om0, cosmo)
    ogamma0, ode0 = _flat_density_params(c)
    # Use vectorize to handle array inputs
    z_arr = jnp.atleast_1d(z)
    result = jnp.vectorize(lambda z_val: _age_gyr_jax(z_val, c, ogamma0, ode0))(z_arr)
    # Return scalar if input was scalar
    return result[0] if jnp.ndim(z) == 0 else result


def age_at_z0_host(cosmo: CosmoParams = DEFAULT_COSMO) -> float:
    r"""Age of universe at z=0 (present day) in Gyr, computed on host.

    Pure-numpy implementation of the same radiation-inclusive physics as
    :func:`age_at_z0` (#2517: matter + CPL dark energy + photons + massive
    neutrinos), designed to be called at module import time without
    executing any JAX operation or triggering DSPS's float64 device
    buffer allocation on float64-less backends (jax-mps, MLX).

    Uses the same 512-node Gauss-Legendre, :math:`a=1/(1+z')` quadrature as
    :func:`_age_gyr_jax` (the node/weight tables are already plain numpy
    host arrays, :data:`_GL_NODES_512_NP` / :data:`_GL_WEIGHTS_512_NP`, so
    reusing them here costs nothing extra at import time), rather than
    DSPS's own log-spaced trapezoidal scheme: DSPS's own quadrature is not
    a target to match bit-for-bit here (#2517 already breaks that
    equivalence to add the missing radiation physics), and the
    log-spaced/trapezoidal scheme was measurably coarser (~8e-5 relative
    against astropy vs this scheme's ~1e-7), enough to round to a
    different 3rd decimal for the ``_AGE_UNIV_GYR`` age-of-universe cap
    (``components/stellar/sfh/registry.py``).

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
    to call at module scope for deferred DSPS initialization. It
    duplicates (in plain numpy) the same E(z) formula as
    :func:`_e_of_z`/:func:`_nu_relative_density`/:func:`_photon_omega0`
    rather than calling them, precisely so that no ``jax.numpy`` op is
    dispatched at import time.
    """
    import numpy as np

    # Cosmological parameters
    Om0, w0, wa, h = cosmo.Om0, cosmo.w0, cosmo.wa, cosmo.h
    tcmb0, neff, m_nu_eV = _radiation_fields(cosmo)

    # Photon density Ω_γ at z=0 (numpy port of _photon_omega0).
    a_b_c2 = 4.0 * SIGMA_SB / C_CGS**3.0
    h0_cgs = h * 100.0 * 1.0e5 / MPC_CM
    rho_crit0 = 3.0 * h0_cgs**2.0 / (8.0 * np.pi * G_GRAV)
    ogamma0 = a_b_c2 * tcmb0**4.0 / rho_crit0

    # Neutrino relative density f_nu(z) (numpy port of _nu_relative_density).
    n_nu = len(m_nu_eV)
    neff_per_nu = neff / n_nu
    tnu0 = NU_TO_CMB_TEMP_RATIO * tcmb0
    kt_nu0_ev = K_BOLTZ_EV * tnu0
    kt_nu0_ev_safe = kt_nu0_ev if kt_nu0_ev > 0.0 else 1.0
    m_nu = np.asarray(m_nu_eV)
    nu_y = (m_nu / kt_nu0_ev_safe) if kt_nu0_ev > 0.0 else np.zeros_like(m_nu)

    def _nu_relative_density_np(z):
        curr_nu_y = nu_y / (1.0 + z)[..., None]
        rel_mass_per = (1.0 + (KOMATSU_NU_SHAPE_SCALE * curr_nu_y) ** KOMATSU_NU_SHAPE_INDEX) ** (
            1.0 / KOMATSU_NU_SHAPE_INDEX
        )
        return NEUTRINO_FERMI_DIRAC_CORRECTION * neff_per_nu * rel_mass_per.sum(-1)

    onu0 = ogamma0 * float(_nu_relative_density_np(np.array([0.0]))[0])
    Ode0 = 1.0 - Om0 - ogamma0 - onu0

    def _e_of_z_np(z):
        zp1 = 1.0 + z
        or_z = ogamma0 * (1.0 + _nu_relative_density_np(z))
        de_z = (
            Ode0 * (1.0 / zp1) ** (-3.0 * (1.0 + w0 + wa)) * np.exp(-3.0 * wa * (1.0 - 1.0 / zp1))
        )
        return np.sqrt(Om0 * zp1**3.0 + or_z * zp1**4.0 + de_z)

    # age(0) = (1/H0) * integral_0^1 da / (a * E(z'(a))), a = 1/(1+z')
    a_nodes = (1.0 + _GL_NODES_512_NP) / 2.0
    z_prime = 1.0 / a_nodes - 1.0
    integrand = 1.0 / (a_nodes * _e_of_z_np(z_prime))
    integrated = np.sum(_GL_WEIGHTS_512_NP * integrand) * 0.5

    # Multiply by Hubble time (in Gyr) to get age in Gyr: 1/H0 = 1 Mpc /
    # (100*h km/s), same MPC_CM / JULIAN_YEAR_S conversion as _age_gyr_jax
    # (not DSPS's own MPC/YEAR constants, so the two host/JAX age functions
    # stay numerically consistent rather than each pinned to a different
    # unit convention).
    hubble_time_gyr = MPC_CM / (100.0 * h * 1.0e5 * JULIAN_YEAR_S * 1.0e9)
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

    Notes
    -----
    Pure JAX implementation using 512-node Gauss-Legendre quadrature.
    Includes radiation (photons + massive neutrinos) per
    astropy.cosmology.Planck18 (#2517). JIT/grad-safe.
    """
    c = _cosmo_from_args(h0, om0, cosmo)
    ogamma0, ode0 = _flat_density_params(c)
    return _age_gyr_jax(0.0, c, ogamma0, ode0)


def comoving_volume_element(
    z: float,
    h0: float | None = None,
    om0: float | None = None,
    *,
    cosmo: CosmoParams | None = None,
) -> float:
    r"""Differential comoving volume element at redshift z in Mpc³/sr.

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

    Notes
    -----
    Flat-FRW comoving volume element, dV_c/dz/dΩ = D_H D_C(z)² / E(z)
    (Hogg 1999, astro-ph/9905116, eq. 28-29 at Ω_k=0), evaluated with the
    same radiation-inclusive D_C(z)/E(z) as :func:`luminosity_distance`
    (#2517) rather than DSPS's ``differential_comoving_volume``.
    JIT/grad-safe: pure JAX operations.
    """
    c = _cosmo_from_args(h0, om0, cosmo)
    ogamma0, ode0 = _flat_density_params(c)
    z_arr = jnp.atleast_1d(z)
    dc_mpc = jnp.vectorize(lambda z_val: _comoving_distance_mpc_jax(z_val, c, ogamma0, ode0))(
        z_arr
    )
    ez = _e_of_z(z_arr, c, ogamma0, ode0)
    d_h_mpc = C_KM_S / (100.0 * c.h)
    result = d_h_mpc * dc_mpc**2.0 / ez
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

    Notes
    -----
    Derived from :func:`angular_diameter_distance_mpc` (radiation-inclusive,
    #2517) rather than DSPS's ``angular_diameter_distance_to_z``.
    JIT/grad-safe: pure JAX operations.
    """
    da_mpc = angular_diameter_distance_mpc(z, h0, om0, cosmo=cosmo)
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

    Notes
    -----
    Derived from :func:`angular_diameter_distance_mpc` (radiation-inclusive,
    #2517) rather than DSPS's ``angular_diameter_distance_to_z``.
    JIT/grad-safe: pure JAX operations.
    """
    da_mpc = angular_diameter_distance_mpc(z, h0, om0, cosmo=cosmo)
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

    Notes
    -----
    The lookup table is built from tengri's own radiation-inclusive
    :func:`age_at_z` (#2517) rather than DSPS's ``age_at_z``, so the
    inversion stays consistent with the age(z) it is inverting.
    """
    # Build lookup table: z_grid → t_grid (decreasing in t)
    z_grid = jnp.linspace(0.0, z_max, n_grid)
    t_grid = age_at_z(z_grid, h0, om0, cosmo=cosmo)

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

    Notes
    -----
    The lookup table is built from tengri's own radiation-inclusive
    :func:`age_at_z` / :func:`age_at_z0` (#2517) rather than DSPS's, so the
    inversion stays consistent with :func:`lookback_time`.
    """
    # Build lookup table: z_grid → t_lookback_grid (increasing)
    z_grid = jnp.linspace(0.0, z_max, n_grid)
    t0 = age_at_z0(h0, om0, cosmo=cosmo)
    t_age = age_at_z(z_grid, h0, om0, cosmo=cosmo)
    t_lookback_grid = t0 - t_age  # increasing with z

    return jnp.interp(t_lookback_gyr, t_lookback_grid, z_grid)
