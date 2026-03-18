"""Accretion disc models for AGN emission.

Two models are provided:

1. **Simple power-law + UV cutoff** — minimal AGN disc with 3 parameters.
2. **Multi-color disc (Shakura-Sunyaev)** — physically-motivated standard thin
   disc following Kubota & Done (2018), simplified to the key parameters.

Both return specific luminosity L_nu in Lsun/Hz as a function of rest-frame
wavelength. All functions are pure JAX and JIT-compilable.

Physical constants are in CGS. Wavelength inputs are in Angstrom.

References
----------
- Shakura & Sunyaev 1973, A&A, 24, 337
- Kubota & Done 2018, MNRAS, 480, 1247
- Nandra & Pounds 1994, MNRAS, 268, 405 (power-law slopes)
"""

import jax
import jax.numpy as jnp

# ===================================================================
# Physical constants (CGS)
# ===================================================================

_H_PLANCK = 6.62607015e-27      # Planck constant [erg s]
_K_BOLTZ = 1.380649e-16         # Boltzmann constant [erg K^-1]
_C_LIGHT = 2.99792458e10        # Speed of light [cm s^-1]
_SIGMA_SB = 5.670374419e-5      # Stefan-Boltzmann [erg cm^-2 s^-1 K^-4]
_LSUN_ERG = 3.828e33            # Solar luminosity [erg s^-1]
_MSUN_G = 1.989e33              # Solar mass [g]
_G_GRAV = 6.674e-8              # Gravitational constant [cm^3 g^-1 s^-2]
_SIGMA_T = 6.6524e-25           # Thomson cross section [cm^2]
_M_PROTON = 1.6726e-24          # Proton mass [g]
_ANGSTROM_CM = 1e-8             # Angstrom -> cm


# ===================================================================
# Planck function (numerically stable)
# ===================================================================

def _planck_lnu(
    nu: jnp.ndarray,
    temperature: float,
) -> jnp.ndarray:
    """Planck function B_nu(T) in erg s^-1 cm^-2 Hz^-1 sr^-1.

    Uses log-space exponent to avoid overflow at low T or high nu.
    Returns 0 where temperature <= 0 (JIT-safe).

    Parameters
    ----------
    nu : array
        Frequency [Hz].
    temperature : float
        Temperature [K].

    Returns
    -------
    array
        B_nu(T) [erg s^-1 cm^-2 Hz^-1 sr^-1].
    """
    # Clamp temperature to avoid division by zero
    t_safe = jnp.maximum(temperature, 1.0)
    x = _H_PLANCK * nu / (_K_BOLTZ * t_safe)
    # Clip exponent to avoid overflow in exp
    x_clip = jnp.clip(x, 0.0, 500.0)
    prefactor = 2.0 * _H_PLANCK * nu**3 / _C_LIGHT**2
    return prefactor / (jnp.exp(x_clip) - 1.0)


def _wavelength_to_nu(wavelength_angstrom: jnp.ndarray) -> jnp.ndarray:
    """Convert wavelength (Angstrom) to frequency (Hz)."""
    return _C_LIGHT / (wavelength_angstrom * _ANGSTROM_CM)


# ===================================================================
# Model 1: Simple power-law disc + UV cutoff
# ===================================================================

def powerlaw_disc(
    wavelength: jnp.ndarray,
    agn_log_lbol: float,
    agn_frac: float = 1.0,
    agn_alpha: float = -1.0,
    agn_T_max: float = 1e5,
    **_kwargs,
) -> jnp.ndarray:
    """Simple power-law accretion disc with exponential UV cutoff.

    L_nu = L_bol * f_disc * C * nu^alpha * exp(-h*nu / k*T_max)

    where C is the normalization constant ensuring integral = L_bol * f_disc.
    In practice we normalize numerically.

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Rest-frame wavelength [Angstrom].
    agn_log_lbol : float
        log10(L_bol / Lsun). Total AGN bolometric luminosity.
    agn_frac : float
        Fraction of L_bol emitted by the disc (0 to 1). Default 1.0.
    agn_alpha : float
        Spectral slope. Typical range: -1.5 to -0.5.
        Default -1.0 (flat in nu*L_nu).
    agn_T_max : float
        Maximum disc temperature [K]. Sets UV cutoff.
        Typical range: 1e4 to 1e6. Default 1e5.

    Returns
    -------
    array, shape (n_wave,)
        Specific luminosity L_nu [Lsun Hz^-1].
    """
    l_bol_erg = 10.0**agn_log_lbol * _LSUN_ERG
    nu = _wavelength_to_nu(wavelength)

    # Unnormalized spectral shape
    x = _H_PLANCK * nu / (_K_BOLTZ * jnp.maximum(agn_T_max, 1.0))
    x_clip = jnp.clip(x, 0.0, 500.0)
    shape = nu**agn_alpha * jnp.exp(-x_clip)

    # Normalize: integrate shape * dnu over the grid via trapezoid
    # Sort by increasing nu for integration
    nu_sorted = jnp.sort(nu)
    shape_sorted = nu_sorted**agn_alpha * jnp.exp(
        -jnp.clip(_H_PLANCK * nu_sorted / (_K_BOLTZ * jnp.maximum(agn_T_max, 1.0)), 0.0, 500.0)
    )
    integral = jnp.trapezoid(shape_sorted, nu_sorted)
    integral_safe = jnp.maximum(jnp.abs(integral), 1e-100)

    l_nu_erg = l_bol_erg * agn_frac * shape / integral_safe
    return l_nu_erg / _LSUN_ERG


# ===================================================================
# Model 2: Multi-color disc (Shakura-Sunyaev thin disc)
# ===================================================================

def _isco_radius(a_spin: float) -> float:
    """Innermost stable circular orbit in units of R_g = GM/c^2.

    Bardeen, Press & Teukolsky (1972) formula for prograde orbits.

    Parameters
    ----------
    a_spin : float
        Dimensionless spin parameter (0 to 0.998).

    Returns
    -------
    float
        r_isco / R_g.
    """
    # Clamp spin to physical range
    a = jnp.clip(a_spin, 0.0, 0.998)
    z1 = 1.0 + (1.0 - a**2) ** (1.0 / 3.0) * (
        (1.0 + a) ** (1.0 / 3.0) + (1.0 - a) ** (1.0 / 3.0)
    )
    z2 = jnp.sqrt(3.0 * a**2 + z1**2)
    return 3.0 + z2 - jnp.sqrt((3.0 - z1) * (3.0 + z1 + 2.0 * z2))


def _eddington_luminosity(log_mbh: float) -> float:
    """Eddington luminosity [erg s^-1].

    L_Edd = 4 * pi * G * M_BH * m_p * c / sigma_T

    Parameters
    ----------
    log_mbh : float
        log10(M_BH / Msun).

    Returns
    -------
    float
        L_Edd [erg s^-1].
    """
    m_bh_g = 10.0**log_mbh * _MSUN_G
    return 4.0 * jnp.pi * _G_GRAV * m_bh_g * _M_PROTON * _C_LIGHT / _SIGMA_T


def _gravitational_radius(log_mbh: float) -> float:
    """Gravitational radius R_g = GM/c^2 [cm].

    Parameters
    ----------
    log_mbh : float
        log10(M_BH / Msun).
    """
    return _G_GRAV * 10.0**log_mbh * _MSUN_G / _C_LIGHT**2


def multicolor_disc(
    wavelength: jnp.ndarray,
    agn_log_lbol: float,
    agn_frac: float = 1.0,
    agn_log_mbh: float = 8.0,
    agn_log_ledd: float = -1.0,
    agn_a_spin: float = 0.0,
    agn_cos_inc: float = 0.5,
    n_radii: int = 50,
    **_kwargs,
) -> jnp.ndarray:
    """Shakura-Sunyaev multi-color disc (standard thin disc).

    The disc SED is a sum of blackbodies at different radii:
        L_nu = integral_{r_isco}^{r_out} B_nu(T(r)) * 4*pi^2*r*dr * cos(i)

    where T(r) = T_in * (r/r_in)^{-3/4} * (1 - sqrt(r_in/r))^{1/4}
    and T_in is set by the accretion rate.

    This is the simplified Kubota & Done (2018) disc, using only the
    outer standard disc zone (no warm Comptonization or hot corona).

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Rest-frame wavelength [Angstrom].
    agn_log_lbol : float
        log10(L_bol / Lsun). Total AGN bolometric luminosity.
    agn_frac : float
        Fraction of L_bol emitted by the disc. Default 1.0.
    agn_log_mbh : float
        log10(M_BH / Msun). Determines temperature profile.
        Typical range: 6 to 10. Default 8.0.
    agn_log_ledd : float
        log10(L / L_Edd). Eddington ratio.
        Typical range: -3 to 0. Default -1.0.
    agn_a_spin : float
        Dimensionless spin (0 to 0.998). Default 0.0 (Schwarzschild).
    agn_cos_inc : float
        Cosine of inclination angle. Default 0.5 (60 deg).
    n_radii : int
        Number of radial integration points. Default 50.

    Returns
    -------
    array, shape (n_wave,)
        Specific luminosity L_nu [Lsun Hz^-1].
    """
    nu = _wavelength_to_nu(wavelength)

    r_g = _gravitational_radius(agn_log_mbh)
    r_isco = _isco_radius(agn_a_spin)
    r_in = r_isco * r_g  # [cm]

    # Outer disc radius: 1000 * r_isco (self-gravity limit)
    r_out = 1000.0 * r_isco * r_g

    # Accretion rate from L = eta * Mdot * c^2
    # eta ~ 1/(4*r_isco) for Novikov-Thorne, simplified to 0.1
    eta = 0.1
    l_edd = _eddington_luminosity(agn_log_mbh)
    l_bol_erg = jnp.minimum(10.0**agn_log_ledd, 1.0) * l_edd
    mdot = l_bol_erg / (eta * _C_LIGHT**2)  # [g s^-1]

    # Inner temperature: T_in = (3 * G * M * Mdot / (8*pi*sigma_SB * r_in^3))^(1/4)
    t_in = (
        3.0 * _G_GRAV * 10.0**agn_log_mbh * _MSUN_G * mdot
        / (8.0 * jnp.pi * _SIGMA_SB * r_in**3)
    ) ** 0.25

    # Radial grid (logarithmic spacing)
    log_r_min = jnp.log10(r_in)
    log_r_max = jnp.log10(r_out)
    log_r_grid = jnp.linspace(log_r_min, log_r_max, n_radii)
    r_grid = 10.0**log_r_grid  # [cm]

    # Temperature profile: T(r) = T_in * (r/r_in)^{-3/4} * (1 - sqrt(r_in/r))^{1/4}
    r_ratio = r_grid / r_in
    torque_correction = jnp.maximum(1.0 - jnp.sqrt(1.0 / r_ratio), 1e-30) ** 0.25
    t_profile = t_in * r_ratio ** (-0.75) * torque_correction  # [K]

    # Integrate: L_nu = sum_i [ B_nu(T_i) * 4 * pi^2 * r_i * dr_i * cos(i) ]
    # dr from logarithmic spacing: dr = r * d(ln r) = r * ln(10) * d(log r)
    d_log_r = log_r_grid[1] - log_r_grid[0]
    dr = r_grid * jnp.log(10.0) * d_log_r  # [cm]

    # B_nu at each (radius, wavelength): shape (n_radii, n_wave)
    # Use vmap over radii
    def _ring_lnu(r_cm, t_ring, dr_ring):
        b_nu = _planck_lnu(nu, t_ring)  # (n_wave,)
        area = 4.0 * jnp.pi**2 * r_cm * dr_ring  # [cm^2]
        return b_nu * area * jnp.maximum(agn_cos_inc, 0.01)

    ring_contributions = jax.vmap(_ring_lnu)(r_grid, t_profile, dr)  # (n_radii, n_wave)
    l_nu_intrinsic = jnp.sum(ring_contributions, axis=0)  # (n_wave,) [erg s^-1 Hz^-1]

    # Renormalize to requested L_bol * agn_frac
    l_bol_requested = 10.0**agn_log_lbol * _LSUN_ERG * agn_frac
    l_nu_total = jnp.trapezoid(l_nu_intrinsic, _wavelength_to_nu(wavelength))
    l_nu_total_safe = jnp.maximum(jnp.abs(l_nu_total), 1e-100)
    scale = l_bol_requested / l_nu_total_safe

    return l_nu_intrinsic * scale / _LSUN_ERG
