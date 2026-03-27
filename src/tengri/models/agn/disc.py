"""Accretion disc models for AGN emission.

Three models are provided:

1. **Simple power-law + UV cutoff** — minimal AGN disc with 3 parameters.
2. **Multi-color disc (Shakura-Sunyaev)** — physically-motivated standard thin
   disc following Kubota & Done (2018), simplified to the key parameters.
   Implements the outer standard disc zone only.
3. **Kubota & Done 3-zone disc** — full K&D (2018) model with outer standard
   disc, warm Comptonization (soft X-ray excess), and hot corona (hard X-ray
   power law). Three radially-stratified zones with self-consistent radii.

All return specific luminosity L_nu in Lsun/Hz as a function of rest-frame
wavelength. All functions are pure JAX and JIT-compilable.

Physical constants are in CGS. Wavelength inputs are in Angstrom.

References
----------
- Shakura & Sunyaev 1973, A&A, 24, 337
- Kubota & Done 2018, MNRAS, 480, 1247
- Nandra & Pounds 1994, MNRAS, 268, 405 (power-law slopes)
- Done et al. 2012, MNRAS, 420, 1848 (QSOSED)
"""

import jax
import jax.numpy as jnp

# ===================================================================
# Physical constants (CGS)
# ===================================================================

_H_PLANCK = 6.62607015e-27  # Planck constant [erg s]
_K_BOLTZ = 1.380649e-16  # Boltzmann constant [erg K^-1]
_C_LIGHT = 2.99792458e10  # Speed of light [cm s^-1]
_SIGMA_SB = 5.670374419e-5  # Stefan-Boltzmann [erg cm^-2 s^-1 K^-4]
_LSUN_ERG = 3.828e33  # Solar luminosity [erg s^-1]
_MSUN_G = 1.989e33  # Solar mass [g]
_G_GRAV = 6.674e-8  # Gravitational constant [cm^3 g^-1 s^-2]
_SIGMA_T = 6.6524e-25  # Thomson cross section [cm^2]
_M_PROTON = 1.6726e-24  # Proton mass [g]
_ANGSTROM_CM = 1e-8  # Angstrom -> cm


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
    # Sort by increasing nu for integration (reuse shape via indices)
    sort_idx = jnp.argsort(nu)
    integral = jnp.trapezoid(shape[sort_idx], nu[sort_idx])
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
    z1 = 1.0 + (1.0 - a**2) ** (1.0 / 3.0) * ((1.0 + a) ** (1.0 / 3.0) + (1.0 - a) ** (1.0 / 3.0))
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

    # Radiative efficiency from Novikov-Thorne: eta = 1 - sqrt(1 - 2/(3*r_isco))
    # For a=0 (Schwarzschild): r_isco=6, eta=0.057
    # For a=0.998 (maximal spin): r_isco~1.24, eta~0.32
    eta = 1.0 - jnp.sqrt(1.0 - 2.0 / (3.0 * r_isco))
    l_edd = _eddington_luminosity(agn_log_mbh)
    l_bol_erg = jnp.minimum(10.0**agn_log_ledd, 1.0) * l_edd
    mdot = l_bol_erg / (eta * _C_LIGHT**2)  # [g s^-1]

    # Inner temperature: T_in = (3 * G * M * Mdot / (8*pi*sigma_SB * r_in^3))^(1/4)
    t_in = (
        3.0 * _G_GRAV * 10.0**agn_log_mbh * _MSUN_G * mdot / (8.0 * jnp.pi * _SIGMA_SB * r_in**3)
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
        area = 2.0 * jnp.pi * r_cm * dr_ring  # [cm^2]
        return b_nu * area * jnp.maximum(agn_cos_inc, 0.01)

    ring_contributions = jax.vmap(_ring_lnu)(r_grid, t_profile, dr)  # (n_radii, n_wave)
    l_nu_intrinsic = jnp.sum(ring_contributions, axis=0)  # (n_wave,) [erg s^-1 Hz^-1]

    # Renormalize to requested L_bol * agn_frac
    l_bol_requested = 10.0**agn_log_lbol * _LSUN_ERG * agn_frac
    l_nu_total = jnp.trapezoid(l_nu_intrinsic, _wavelength_to_nu(wavelength))
    l_nu_total_safe = jnp.maximum(jnp.abs(l_nu_total), 1e-100)
    scale = l_bol_requested / l_nu_total_safe

    return l_nu_intrinsic * scale / _LSUN_ERG


# ===================================================================
# Model 3: Kubota & Done (2018) 3-zone disc
# ===================================================================

# keV -> erg conversion
_KEV_TO_ERG = 1.602176634e-9


def _warm_comptonization_lnu(
    nu: jnp.ndarray,
    temperature: float,
    nu_warm: float,
    gamma_warm: float,
) -> jnp.ndarray:
    """Modified blackbody for the warm Comptonization zone.

    The warm zone produces a soft X-ray excess via optically thick,
    warm electron scattering. The SED is a Comptonized blackbody:

        L_nu ~ B_nu(T_disc(r)) * (nu / nu_warm)^(Gamma_warm - 1)

    for nu > nu_warm, otherwise pure blackbody.

    Parameters
    ----------
    nu : array
        Frequency [Hz].
    temperature : float
        Local disc temperature [K].
    nu_warm : float
        Warm electron characteristic frequency [Hz],
        derived from kT_warm.
    gamma_warm : float
        Warm Comptonization photon index (~2.5).

    Returns
    -------
    array
        Modified B_nu [erg s^-1 cm^-2 Hz^-1 sr^-1].
    """
    b_nu = _planck_lnu(nu, temperature)
    # Comptonization enhancement: smooth transition at nu_warm
    ratio = nu / jnp.maximum(nu_warm, 1.0)
    # Only enhance above nu_warm; below it's a standard blackbody
    enhancement = jnp.where(
        ratio > 1.0,
        ratio ** (gamma_warm - 1.0),
        1.0,
    )
    return b_nu * enhancement


def _hot_corona_lnu(
    nu: jnp.ndarray,
    l_hot_erg: float,
    gamma_hard: float,
    kt_hot_erg: float,
) -> jnp.ndarray:
    """Hot corona emission: power law with exponential cutoff.

    The optically thin, hot corona produces hard X-ray emission:

        L_nu ~ nu^(1 - Gamma_hard) * exp(-h*nu / kT_hot)

    Normalized so that the frequency-integrated luminosity = l_hot_erg.

    Parameters
    ----------
    nu : array
        Frequency [Hz].
    l_hot_erg : float
        Total hot corona luminosity [erg s^-1].
    gamma_hard : float
        Hard X-ray photon index (~1.8).
    kt_hot_erg : float
        Hot corona temperature [erg] (= kT_hot in erg).

    Returns
    -------
    array
        L_nu [erg s^-1 Hz^-1].
    """
    # Power law with exponential cutoff
    x = _H_PLANCK * nu / jnp.maximum(kt_hot_erg, 1e-30)
    x_clip = jnp.clip(x, 0.0, 500.0)
    shape = nu ** (1.0 - gamma_hard) * jnp.exp(-x_clip)

    # Normalize: integrate shape over frequency
    sort_idx = jnp.argsort(nu)
    integral = jnp.trapezoid(shape[sort_idx], nu[sort_idx])
    integral_safe = jnp.maximum(jnp.abs(integral), 1e-100)

    return l_hot_erg * shape / integral_safe


def kubota_done_disc(
    wavelength: jnp.ndarray,
    agn_log_lbol: float,
    agn_frac: float = 1.0,
    agn_log_mbh: float = 8.0,
    agn_log_ledd: float = -1.0,
    agn_a_spin: float = 0.0,
    agn_cos_inc: float = 0.5,
    agn_f_hard: float = 0.02,
    agn_gamma_warm: float = 2.5,
    agn_kt_warm: float = 0.2,
    agn_gamma_hard: float = 1.8,
    agn_kt_hot: float = 100.0,
    agn_r_warm_ratio: float = 2.0,
    n_radii: int = 50,
    **_kwargs,
) -> jnp.ndarray:
    """Kubota & Done (2018) 3-zone accretion disc.

    Three radially stratified zones:

    1. **Outer standard disc** (r > R_warm): Shakura-Sunyaev blackbody.
       Produces the optical/UV big blue bump.
    2. **Warm Comptonization** (R_hot < r < R_warm): Optically thick,
       warm electrons. Produces the soft X-ray excess. SED is a modified
       blackbody: B_nu(T(r)) * (nu/nu_warm)^(Gamma_warm - 1).
    3. **Hot corona** (R_ISCO < r < R_hot): Optically thin, hot electrons.
       Produces hard X-ray power law with exponential cutoff.

    Zone radii are determined from an approximate QSOSED prescription:
    - R_hot = R_ISCO * (1 + f_hard * L_bol/L_Edd)^(1/3)
    - R_warm = R_hot * r_warm_ratio

    The total SED is normalized so all 3 zones sum to
    L_bol * agn_frac.

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Rest-frame wavelength [Angstrom].
    agn_log_lbol : float
        log10(L_bol / Lsun). Total AGN bolometric luminosity.
    agn_frac : float
        Fraction of L_bol emitted by the disc (all 3 zones). Default 1.0.
    agn_log_mbh : float
        log10(M_BH / Msun). Default 8.0.
    agn_log_ledd : float
        log10(L / L_Edd). Eddington ratio. Default -1.0.
    agn_a_spin : float
        Dimensionless BH spin (0 to 0.998). Default 0.0.
    agn_cos_inc : float
        Cosine of inclination. Default 0.5.
    agn_f_hard : float
        Fraction of L_Edd emitted by the hot corona. Default 0.02.
    agn_gamma_warm : float
        Warm Comptonization photon index. Default 2.5.
    agn_kt_warm : float
        Warm electron temperature [keV]. Default 0.2.
    agn_gamma_hard : float
        Hard X-ray photon index. Default 1.8.
    agn_kt_hot : float
        Hot corona temperature [keV]. Default 100.0.
    agn_r_warm_ratio : float
        R_warm / R_hot ratio. Default 2.0.
    n_radii : int
        Number of radial integration points per zone. Default 50.

    Returns
    -------
    array, shape (n_wave,)
        Specific luminosity L_nu [Lsun Hz^-1].

    References
    ----------
    - Kubota & Done 2018, MNRAS, 480, 1247
    - Done et al. 2012, MNRAS, 420, 1848 (QSOSED)
    """
    nu = _wavelength_to_nu(wavelength)

    # --- Black hole parameters ---
    r_g = _gravitational_radius(agn_log_mbh)
    r_isco_rg = _isco_radius(agn_a_spin)
    r_isco_cm = r_isco_rg * r_g

    # Radiative efficiency (Novikov-Thorne)
    eta = 1.0 - jnp.sqrt(1.0 - 2.0 / (3.0 * r_isco_rg))

    l_edd = _eddington_luminosity(agn_log_mbh)
    l_edd_ratio = jnp.clip(10.0**agn_log_ledd, 1e-10, 1.0)
    l_bol_erg = l_edd_ratio * l_edd
    mdot = l_bol_erg / (eta * _C_LIGHT**2)

    # Inner temperature
    t_in = (
        3.0
        * _G_GRAV
        * 10.0**agn_log_mbh
        * _MSUN_G
        * mdot
        / (8.0 * jnp.pi * _SIGMA_SB * r_isco_cm**3)
    ) ** 0.25

    # --- Zone radii (QSOSED approximation) ---
    # R_hot: approximate from QSOSED
    f_hard_safe = jnp.clip(agn_f_hard, 1e-6, 0.5)
    r_hot_rg = r_isco_rg * (1.0 + f_hard_safe * l_edd_ratio) ** (1.0 / 3.0)
    r_hot_cm = r_hot_rg * r_g

    # R_warm: parameterized as multiple of R_hot
    r_warm_ratio_safe = jnp.clip(agn_r_warm_ratio, 1.1, 10.0)
    r_warm_cm = r_hot_cm * r_warm_ratio_safe

    # Outer disc radius: self-gravity limit
    r_out_cm = 1000.0 * r_isco_rg * r_g

    # Ensure proper ordering: R_ISCO < R_hot < R_warm < R_out
    r_hot_cm = jnp.clip(r_hot_cm, r_isco_cm * 1.01, r_out_cm * 0.5)
    r_warm_cm = jnp.clip(r_warm_cm, r_hot_cm * 1.01, r_out_cm * 0.9)

    # --- Warm Comptonization characteristic frequency ---
    kt_warm_erg = agn_kt_warm * _KEV_TO_ERG
    nu_warm = kt_warm_erg / _H_PLANCK

    # --- Hot corona temperature ---
    kt_hot_erg = agn_kt_hot * _KEV_TO_ERG

    # --- Corona luminosity ---
    # L_hot = f_hard * L_Edd (capped at L_bol)
    l_hot_erg = jnp.minimum(f_hard_safe * l_edd, l_bol_erg * 0.5)

    # ===============================================================
    # Zone 1: Outer standard disc (r > R_warm)
    # ===============================================================
    log_r_warm = jnp.log10(r_warm_cm)
    log_r_out = jnp.log10(r_out_cm)
    log_r_outer = jnp.linspace(log_r_warm, log_r_out, n_radii)
    r_outer = 10.0**log_r_outer

    r_ratio_outer = r_outer / r_isco_cm
    torque_outer = jnp.maximum(1.0 - jnp.sqrt(1.0 / r_ratio_outer), 1e-30) ** 0.25
    t_outer = t_in * r_ratio_outer ** (-0.75) * torque_outer

    d_log_r_outer = log_r_outer[1] - log_r_outer[0]
    dr_outer = r_outer * jnp.log(10.0) * d_log_r_outer

    def _outer_ring(r_cm, t_ring, dr_ring):
        b_nu = _planck_lnu(nu, t_ring)
        area = 2.0 * jnp.pi * r_cm * dr_ring
        return b_nu * area * jnp.maximum(agn_cos_inc, 0.01)

    l_nu_outer = jnp.sum(jax.vmap(_outer_ring)(r_outer, t_outer, dr_outer), axis=0)

    # ===============================================================
    # Zone 2: Warm Comptonization (R_hot < r < R_warm)
    # ===============================================================
    log_r_hot = jnp.log10(r_hot_cm)
    log_r_warm_grid = jnp.linspace(log_r_hot, log_r_warm, n_radii)
    r_warm_grid = 10.0**log_r_warm_grid

    r_ratio_warm = r_warm_grid / r_isco_cm
    torque_warm = jnp.maximum(1.0 - jnp.sqrt(1.0 / r_ratio_warm), 1e-30) ** 0.25
    t_warm = t_in * r_ratio_warm ** (-0.75) * torque_warm

    d_log_r_warm = log_r_warm_grid[1] - log_r_warm_grid[0]
    dr_warm = r_warm_grid * jnp.log(10.0) * d_log_r_warm

    def _warm_ring(r_cm, t_ring, dr_ring):
        b_nu_mod = _warm_comptonization_lnu(nu, t_ring, nu_warm, agn_gamma_warm)
        area = 2.0 * jnp.pi * r_cm * dr_ring
        return b_nu_mod * area * jnp.maximum(agn_cos_inc, 0.01)

    l_nu_warm = jnp.sum(jax.vmap(_warm_ring)(r_warm_grid, t_warm, dr_warm), axis=0)

    # ===============================================================
    # Zone 3: Hot corona (R_ISCO < r < R_hot)
    # ===============================================================
    l_nu_hot = _hot_corona_lnu(nu, l_hot_erg, agn_gamma_hard, kt_hot_erg)

    # ===============================================================
    # Combine and normalize
    # ===============================================================
    l_nu_total = l_nu_outer + l_nu_warm + l_nu_hot

    # Normalize all 3 zones to L_bol * agn_frac
    l_bol_requested = 10.0**agn_log_lbol * _LSUN_ERG * agn_frac
    sort_idx = jnp.argsort(nu)
    l_nu_integral = jnp.trapezoid(l_nu_total[sort_idx], nu[sort_idx])
    l_nu_integral_safe = jnp.maximum(jnp.abs(l_nu_integral), 1e-100)
    scale = l_bol_requested / l_nu_integral_safe

    return l_nu_total * scale / _LSUN_ERG
