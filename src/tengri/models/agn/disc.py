"""Accretion disc models for AGN emission.

Four models are provided:

1. **Simple power-law + UV cutoff** — minimal AGN disc with 3 parameters.
2. **Multi-color disc (Shakura-Sunyaev)** — physically-motivated standard thin
   disc following Kubota & Done (2018), simplified to the key parameters.
   Implements the outer standard disc zone only.
3. **Kubota & Done 3-zone disc** — full K&D (2018) model with outer standard
   disc, warm Comptonization (soft X-ray excess), and hot corona (hard X-ray
   power law). Three radially-stratified zones with self-consistent radii.
4. **ADAF + truncated disc** — for low-luminosity AGN (L/L_Edd < 0.01).
   The inner disc transitions to an advection-dominated accretion flow
   (optically thin, radiatively inefficient). Based on Mahadevan (1997)
   and Nemmen+2014.

All return specific luminosity L_nu in erg/s/Hz as a function of rest-frame
wavelength. All functions are pure JAX and JIT-compilable.

Physical constants are in CGS. Wavelength inputs are in Angstrom.

References
----------
- Shakura & Sunyaev 1973, A&A, 24, 337
- Kubota & Done 2018, MNRAS, 480, 1247
- Nandra & Pounds 1994, MNRAS, 268, 405 (power-law slopes)
- Done et al. 2012, MNRAS, 420, 1848 (QSOSED)
- Mahadevan 1997, ApJ, 477, 585 (ADAF spectra)
- Nemmen et al. 2014, MNRAS, 438, 2804 (ADAF modelling)
- Lopez et al. 2024 (ADAF + truncated disc for LLAGN)
- Beloborodov 1999, ApJ, 510, L123 (self-consistent Gamma_hot)
"""

import warnings

import jax
import jax.numpy as jnp

from tengri.models.agn._nthcomp import (
    _TABLE_AVAILABLE as _NTHCOMP_AVAILABLE,
    nthcomp_lnu_interp as _nthcomp_lnu_interp,
)
from tengri.models.agn._phys import (
    C_LIGHT as _C_LIGHT,
    H_PLANCK as _H_PLANCK,
    K_BOLTZ as _K_BOLTZ,
    LSUN_ERG as _LSUN_ERG,
    planck_lnu as _planck_lnu,
    wavelength_to_nu as _wavelength_to_nu,
)
from tengri.utils.physics_constants import (
    G_GRAV as _G_GRAV,
    K_BOLTZ_KEV as _K_BOLTZ_KEV,
    KEV_TO_ERG as _KEV_TO_ERG,
    M_PROTON as _M_PROTON,
    M_SUN as _MSUN_G,
    SIGMA_SB as _SIGMA_SB,
    SIGMA_T as _SIGMA_T,
)

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
        Specific luminosity L_nu [erg s^-1 Hz^-1].
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
    return l_nu_erg


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

    Notes
    -----
    To ensure finite gradients at the Schwarzschild limit (a=0), we clamp
    the argument to the final square root to a small positive value (1e-20).
    The BPT72 formula has a gradient singularity at a=0 where (3-z1)→0,
    which makes sqrt((3-z1)*(3+z1+2*z2)) undefined in AD. The physical
    limit is correct (r_isco=6 for a=0), but the gradient path must be
    stabilized for JAX autodiff to work.
    """
    # Clamp spin to physical range
    a = jnp.clip(a_spin, 0.0, 0.998)
    z1 = 1.0 + (1.0 - a**2) ** (1.0 / 3.0) * ((1.0 + a) ** (1.0 / 3.0) + (1.0 - a) ** (1.0 / 3.0))
    z2 = jnp.sqrt(3.0 * a**2 + z1**2)
    # Clamp sqrt argument to avoid zero-to-zero gradient singularity at a=0
    sqrt_arg = jnp.maximum((3.0 - z1) * (3.0 + z1 + 2.0 * z2), 1e-20)
    return 3.0 + z2 - jnp.sqrt(sqrt_arg)


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


def _nt_l_diss_analytic(x_hot: float, r_isco_cm: float, t_in: float) -> float:
    """Analytic NT emissivity integral over the hot corona zone (K&D 2018 Eq. 2).

    Integrates F_NT = σ T_NT^4(R) over the disc annuli from R_ISCO to R_hot.
    With T_NT(R) = T_in * (R/R_isco)^{-3/4} * (1 - sqrt(R_isco/R))^{1/4},

        L_diss = 2 * ∫_{R_isco}^{R_hot} σ T_NT^4 * 2πR dR
               = L_0 * h(x_hot)

    where L_0 = 4π R_isco^2 σ T_in^4 and the analytic form is:

        h(x) = 1/10 - 1/(2x^2) + 2/(5 x^{5/2})     (x = R_hot / R_isco ≥ 1)

    h(1) = 0 (empty corona), h(∞) → 0.1 (entire NT disc luminosity).

    Parameters
    ----------
    x_hot : float
        R_hot / R_ISCO ≥ 1.
    r_isco_cm : float
        ISCO radius [cm].
    t_in : float
        Inner disc temperature T_in [K].

    Returns
    -------
    float
        L_diss [erg s^-1].
    """
    l0 = 4.0 * jnp.pi * r_isco_cm**2 * _SIGMA_SB * t_in**4
    h = 0.1 - 0.5 * x_hot ** (-2.0) + 0.4 * x_hot ** (-2.5)
    return l0 * jnp.maximum(h, 0.0)


def _r_hot_bisect(
    r_isco_cm: float,
    t_in: float,
    l_hot_target: float,
    n_iter: int = 40,
) -> float:
    """Bisection solve for R_hot from K&D 2018 Eq. 2 (JAX-compatible).

    Finds R_hot such that L_diss,hot(R_ISCO, R_hot) = l_hot_target by
    bisection in log(x_hot) space, where x_hot = R_hot / R_ISCO.

    The NT emissivity integral has an analytic closed form:

        L_diss(x_hot) = L_0 * [1/10 - 1/(2x^2) + 2/(5 x^{5/2})]

    This is strictly monotone in x_hot, so bisection is guaranteed to
    converge. After n_iter=40 steps the bracket width is < 2^{-40} ≈ 1e-12
    of its initial log-width (log x in [0, 9.2]), sufficient for machine
    precision.

    Parameters
    ----------
    r_isco_cm : float
        ISCO radius [cm].
    t_in : float
        Inner disc temperature [K].
    l_hot_target : float
        Target L_diss,hot [erg s^-1]. Clipped to < L_max = L_0/10.
    n_iter : int
        Number of bisection iterations. Default 40.

    Returns
    -------
    float
        R_hot [cm].
    """
    # Maximum possible L_diss (entire disc, x→∞): h→0.1, so L_max = L0 * 0.1
    l0 = 4.0 * jnp.pi * r_isco_cm**2 * _SIGMA_SB * t_in**4
    # Clip target to (0, L_max); if l_hot_target >= L_max, r_hot → ∞ (use upper bound)
    l_target = jnp.clip(l_hot_target, 1e-100, l0 * 0.099)

    def _h(log_x):
        x = jnp.exp(log_x)
        return l0 * (0.1 - 0.5 * x ** (-2.0) + 0.4 * x ** (-2.5))

    # Bisect in log(x_hot) in [log(1.001), log(1e4)]
    lo = jnp.log(1.001)
    hi = jnp.log(1.0e4)

    def _step(state, _):
        lo_i, hi_i = state
        mid = (lo_i + hi_i) * 0.5
        l_mid = _h(mid)
        go_right = l_mid < l_target
        return (jnp.where(go_right, mid, lo_i), jnp.where(go_right, hi_i, mid)), None

    (lo_f, hi_f), _ = jax.lax.scan(_step, (lo, hi), None, length=n_iter)
    x_hot = jnp.exp((lo_f + hi_f) * 0.5)
    return x_hot * r_isco_cm


def _l_seed_geometric(
    r_isco_cm: float,
    r_hot_cm: float,
    r_out_cm: float,
    t_in: float,
    n_radii: int = 100,
) -> float:
    """Geometric seed photon luminosity intercepted by the hot corona (K&D 2018 Eq. 3).

    Integrates over disc radii R > R_hot with the geometric covering fraction
    of the hot flow as seen from the disc:

        L_seed = 2 * ∫_{R_hot}^{R_out} F_NT(R) * [Θ(R) / π] * 2πR dR

    where (K&D 2018 Eq. 4, assuming H = R_hot):

        Θ(R) = θ_0 - (1/2)sin(2θ_0),   sin θ_0 = H/R = R_hot / R

    The factor Θ(R)/π is the solid-angle fraction of the spherical hot flow
    (height H = R_hot) subtended at disc radius R. This formula assumes the
    hot corona is quasi-spherical with scale height equal to its truncation
    radius, as in the K&D 2018 geometry.

    Parameters
    ----------
    r_isco_cm : float
        ISCO radius [cm].
    r_hot_cm : float
        Hot corona radius [cm].
    r_out_cm : float
        Outer disc radius [cm].
    t_in : float
        Inner disc temperature [K].
    n_radii : int
        Number of logarithmically spaced radial integration points.

    Returns
    -------
    float
        L_seed [erg s^-1].
    """
    log_r_min = jnp.log10(r_hot_cm)
    log_r_max = jnp.log10(r_out_cm)
    log_r = jnp.linspace(log_r_min, log_r_max, n_radii)
    r = 10.0**log_r  # [cm]
    d_log_r = (log_r_max - log_r_min) / (n_radii - 1)
    dr = r * jnp.log(10.0) * d_log_r  # [cm]

    # NT temperature and emissivity
    r_ratio = r / r_isco_cm
    torque = jnp.maximum(1.0 - jnp.sqrt(1.0 / r_ratio), 1e-30) ** 0.25
    t_r = t_in * r_ratio ** (-0.75) * torque  # [K]
    f_nt = _SIGMA_SB * t_r**4  # [erg s^-1 cm^-2]

    # Geometric covering factor Θ(R)/π (K&D 2018 Eq. 4), H = R_hot
    # Clamp sin_th0 to avoid infinite gradients at arcsin boundaries (±1).
    # The interior points are r >> r_hot, so sin_th0 << 1. Only the first
    # point (r ≈ r_hot) can approach 1. We use (0.001, 0.999) to avoid
    # gradient singularities while maintaining physical correctness.
    sin_th0 = jnp.clip(r_hot_cm / r, 0.001, 0.999)
    th0 = jnp.arcsin(sin_th0)  # θ_0 in [0, π/2]
    theta_r = th0 - 0.5 * jnp.sin(2.0 * th0)  # Θ(R) = θ_0 - (1/2)sin(2θ_0)
    covering = theta_r / jnp.pi  # Θ(R)/π ∈ [0, 0.5]

    # L_seed = 2 * ∫ F_NT * (Θ/π) * 2πR dR
    integrand = f_nt * covering * 2.0 * jnp.pi * r  # [erg s^-1 cm^-1]
    l_seed = 2.0 * jnp.sum(integrand * dr)
    return jnp.maximum(l_seed, 1e-100)


def _self_gravity_radius(log_mbh: float, l_edd_ratio: float, alpha_visc: float = 0.1) -> float:
    """Self-gravity (Toomre instability) radius in units of R_g.

    The disc becomes gravitationally unstable beyond this radius; for
    r > r_sg the disc fragments into clumps rather than accreting.
    This is the physically motivated outer boundary for the thin disc.

    Laor & Netzer (1989), Eq. 10:
        r_sg = 2150 * (alpha/0.1)^{2/9} * lambda_Edd^{4/9}
               * (M_BH / 10^8 M_sun)^{-2/9}   [R_g]

    where lambda_Edd = L_bol / L_Edd is the Eddington ratio and
    alpha is the Shakura-Sunyaev viscosity parameter (default 0.1).

    Reference: Laor, A. & Netzer, H. (1989), MNRAS 238, 897.
    Also used in qsosed (Quera-Bofarull) as `gravity_radius`.

    Parameters
    ----------
    log_mbh : float
        log10(M_BH / Msun).
    l_edd_ratio : float
        Eddington ratio lambda_Edd = L_bol / L_Edd (0 to 1).
    alpha_visc : float
        Shakura-Sunyaev viscosity parameter. Default 0.1.

    Returns
    -------
    float
        r_sg in units of R_g.
    """
    m9 = 10.0**log_mbh / 1.0e8  # M_BH / 10^8 M_sun
    m9_safe = jnp.maximum(m9, 1e-6)
    lambda_safe = jnp.clip(l_edd_ratio, 1e-10, 1.0)
    alpha_safe = jnp.maximum(alpha_visc, 1e-4)
    return (
        2150.0
        * (alpha_safe / 0.1) ** (2.0 / 9.0)
        * lambda_safe ** (4.0 / 9.0)
        * m9_safe ** (-2.0 / 9.0)
    )


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
        L_nu = integral_{r_isco}^{r_out} B_nu(T(r)) * 2*pi^2*r*dr * cos(i)

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
        Specific luminosity L_nu [erg s^-1 Hz^-1].
    """
    nu = _wavelength_to_nu(wavelength)

    r_g = _gravitational_radius(agn_log_mbh)
    r_isco = _isco_radius(agn_a_spin)
    r_in = r_isco * r_g  # [cm]

    # Radiative efficiency from Novikov-Thorne: eta = 1 - sqrt(1 - 2/(3*r_isco))
    # For a=0 (Schwarzschild): r_isco=6, eta=0.057
    # For a=0.998 (maximal spin): r_isco~1.24, eta~0.32
    eta = 1.0 - jnp.sqrt(1.0 - 2.0 / (3.0 * r_isco))
    l_edd = _eddington_luminosity(agn_log_mbh)
    l_bol_erg = jnp.minimum(10.0**agn_log_ledd, 1.0) * l_edd

    # Outer disc radius: Laor & Netzer (1989) self-gravity (Toomre) radius.
    # Beyond r_sg the disc fragments rather than accretes; this is the
    # physically motivated outer boundary used by qsosed (Quera-Bofarull).
    # r_sg ~ 2150 * (alpha/0.1)^{2/9} * lambda_Edd^{4/9} * (M/1e8)^{-2/9} R_g.
    l_edd_ratio = jnp.minimum(10.0**agn_log_ledd, 1.0)
    r_sg_rg = _self_gravity_radius(agn_log_mbh, l_edd_ratio)
    r_out = jnp.maximum(r_sg_rg, r_isco * 10.0) * r_g  # at least 10 r_isco
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
        # dL_nu = pi * B_nu * dA * cos(i): the pi comes from integrating B_nu cos(theta)
        # over the hemisphere (Rybicki & Lightman 1979, Eq. 1.6).  dA = 2*pi*r*dr.
        area = jnp.pi * 2.0 * jnp.pi * r_cm * dr_ring  # [cm^2 sr]
        return b_nu * area * jnp.maximum(agn_cos_inc, 0.01)

    ring_contributions = jax.vmap(_ring_lnu)(r_grid, t_profile, dr)  # (n_radii, n_wave)
    l_nu_intrinsic = jnp.sum(ring_contributions, axis=0)  # (n_wave,) [erg s^-1 Hz^-1]

    # Renormalize to requested L_bol * agn_frac
    l_bol_requested = 10.0**agn_log_lbol * _LSUN_ERG * agn_frac
    # Sort by ascending frequency before integrating (nu descends when wave ascends).
    # Using jnp.abs() on a descending-x trapezoid is brittle — sort explicitly.
    _nu = _wavelength_to_nu(wavelength)
    _sort_idx = jnp.argsort(_nu)
    l_nu_total = jnp.trapezoid(l_nu_intrinsic[_sort_idx], _nu[_sort_idx])
    l_nu_total_safe = jnp.maximum(jnp.abs(l_nu_total), 1e-100)
    scale = l_bol_requested / l_nu_total_safe

    return l_nu_intrinsic * scale


# ===================================================================
# Model 3: Kubota & Done (2018) 3-zone disc
# ===================================================================


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
    # Seed frequency from the LOCAL disc blackbody temperature at this ring radius.
    # The warm Comptonization zone up-scatters photons from the disc temperature
    # to the warm electron temperature kT_warm (K&D 2018, MNRAS 480, 1247, Eq. 3).
    nu_seed = _K_BOLTZ * temperature / _H_PLANCK
    # Power-law enhancement between nu_seed and nu_warm (soft X-ray cutoff)
    ratio = nu / jnp.maximum(nu_seed, 1.0)
    # Cap the enhancement at (nu_warm/nu_seed)^(Gamma-1) to avoid divergence above cutoff
    max_enh = (nu_warm / jnp.maximum(nu_seed, 1.0)) ** (gamma_warm - 1.0)
    enhancement = jnp.where(
        ratio > 1.0,
        jnp.minimum(ratio ** (gamma_warm - 1.0), max_enh),
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
    # Note: trapezoid() works with unsorted x values; we avoid explicit
    # sorting/indexing to prevent NaN gradients in JAX autodiff.
    integral = jnp.trapezoid(shape, nu)
    integral_safe = jnp.maximum(jnp.abs(integral), 1e-100)

    return l_hot_erg * shape / integral_safe


def beloborodov_gamma_hot(
    l_diss_hot: float,
    l_seed: float,
) -> float:
    """Self-consistent hard X-ray photon index (Beloborodov 1999).

    Derives the spectral index of the hot corona from the ratio of
    dissipated luminosity to seed photon luminosity.

    Kubota & Done (2018, MNRAS 480 1247) Eq. 6 rewrites the Beloborodov
    (1999, ApJ 510 L123) Compton-amplification result as:

        Gamma_hot = (7/3) * (L_diss / L_seed)^{-0.1}

    The exponent -0.1 is the K&D 2018 formulation of Beloborodov (1999).

    Parameters
    ----------
    l_diss_hot : float
        Luminosity dissipated in the hot corona [any units].
    l_seed : float
        Soft photon luminosity intercepted by the corona [same units].

    Returns
    -------
    float
        Hard X-ray photon index, clipped to [1.4, 3.0].

    References
    ----------
    - Beloborodov 1999, ApJ, 510, L123
    - Kubota & Done 2018, MNRAS, 480, 1247, Eq. 6
    """
    ratio = jnp.clip(l_diss_hot / jnp.maximum(l_seed, 1e-30), 1e-3, 1e3)
    gamma = (7.0 / 3.0) * ratio ** (-0.1)  # K&D 2018 Eq. 6
    return jnp.clip(gamma, 1.4, 3.0)


def compute_l2500(
    wavelength: jnp.ndarray,
    l_nu: jnp.ndarray,
) -> float:
    """Extract monochromatic luminosity at rest-frame 2500 Angstrom.

    Linearly interpolates L_nu onto 2500 A. Useful for computing
    alpha_ox and other AGN diagnostics.

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Rest-frame wavelength [Angstrom], need not be sorted.
    l_nu : array, shape (n_wave,)
        Specific luminosity [erg s^-1 Hz^-1].

    Returns
    -------
    float
        L_nu at 2500 A [erg s^-1 Hz^-1].
    """
    sort_idx = jnp.argsort(wavelength)
    return jnp.interp(2500.0, wavelength[sort_idx], l_nu[sort_idx])


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
    agn_self_consistent_gamma: bool = False,
    **_kwargs,
) -> jnp.ndarray:
    """Kubota & Done (2018) 3-zone accretion disc.

    Three radially stratified zones:

    1. **Outer standard disc** (r > R_warm): Shakura-Sunyaev blackbody.
       Produces the optical/UV big blue bump.
    2. **Warm Comptonization** (R_hot < r < R_warm): Optically thick,
       warm electrons. Produces the soft X-ray excess. SED is computed via
       the nthcomp Kompaneets solver (K&D 2018 Eq. 2.2) when templates are
       available (run ``scripts/build_nthcomp_templates.py``), otherwise
       uses a simplified modified-blackbody proxy.
    3. **Hot corona** (R_ISCO < r < R_hot): Optically thin, hot electrons.
       Produces hard X-ray power law with exponential cutoff.

    Zone radii are determined self-consistently from the NT emissivity:

    - R_hot: bisection solve of K&D 2018 Eq. 2,
      L_diss,hot = 2 ∫_{R_ISCO}^{R_hot} σ T_NT^4 · 2πR dR = f_hard · L_Edd.
      Uses 40-step JAX-compatible bisection on the analytic NT integral
      h(x) = 1/10 - 1/(2x^2) + 2/(5x^{5/2}) for exact machine-precision result.
    - R_warm = R_hot * r_warm_ratio (default factor 2, per K&D 2018 §4.3).
    - R_out: Laor & Netzer (1989) self-gravity radius (K&D paper explicitly
      sets r_out = r_sg).

    L_seed (seed photons for the hot corona) is the geometric integral of
    K&D 2018 Eq. 3: 2 ∫_{R_hot}^{R_out} F_NT · Θ(R)/π · 2πR dR,
    where Θ(R) = θ_0 - sin(2θ_0)/2, sin θ_0 = R_hot/R (corona scale height
    H = R_hot). This drives the self-consistent Beloborodov Γ_hot.

    The total SED is normalized so all 3 zones sum to L_bol * agn_frac.

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
    agn_self_consistent_gamma : bool
        If True, derive ``agn_gamma_hard`` self-consistently from the
        Beloborodov (1999) relation using the corona energy balance
        instead of using the input value. Default False.

    Returns
    -------
    array, shape (n_wave,)
        Specific luminosity L_nu [erg s^-1 Hz^-1].

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

    # --- Zone radii ---
    # R_hot: exact energy-balance solve from K&D 2018 Eq. 2.
    # Find R_hot such that L_diss,hot(R_ISCO, R_hot) = f_hard * L_Edd,
    # where L_diss,hot = 2 ∫_{R_ISCO}^{R_hot} σ T_NT^4 * 2πR dR.
    # Uses JAX-compatible bisection on the analytic NT integral (40 iterations,
    # log-x bracket → machine-precision convergence). Replaces the previous
    # closed-form approximation r_hot ≈ r_isco*(1 + f_hard*λ)^{1/3} which had
    # ~10% error; the bisection is exact and adds negligible compile overhead.
    f_hard_safe = jnp.clip(agn_f_hard, 1e-6, 0.5)
    l_hot_target = f_hard_safe * l_edd
    r_hot_cm = _r_hot_bisect(r_isco_cm, t_in, l_hot_target)

    # R_warm: parameterized as multiple of R_hot (qsosed hardwires factor 2)
    r_warm_ratio_safe = jnp.clip(agn_r_warm_ratio, 1.1, 10.0)
    r_warm_cm = r_hot_cm * r_warm_ratio_safe

    # Outer disc radius: Laor & Netzer (1989) self-gravity (Toomre) radius.
    # qsosed uses this formula (as gravity_radius); replaces the previous
    # fixed 1000*r_isco approximation which was off by factors of a few for
    # extreme M_BH or lambda_Edd.
    r_sg_rg = _self_gravity_radius(agn_log_mbh, l_edd_ratio)
    r_out_cm = jnp.maximum(r_sg_rg, r_isco_rg * 10.0) * r_g

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
        # dL_nu = pi * B_nu * dA * cos(i) (Rybicki & Lightman 1979, Eq. 1.6)
        area = jnp.pi * 2.0 * jnp.pi * r_cm * dr_ring
        return b_nu * area * jnp.maximum(agn_cos_inc, 0.01)

    l_nu_outer = jnp.sum(jax.vmap(_outer_ring)(r_outer, t_outer, dr_outer), axis=0)

    # ===============================================================
    # Zone 2: Warm Comptonization (R_hot < r < R_warm)
    # ===============================================================
    if not _NTHCOMP_AVAILABLE:
        warnings.warn(
            "nthcomp templates not found — warm Comptonization zone uses the "
            "simplified modified-blackbody proxy (BUG-04 workaround). "
            "Run `python scripts/build_nthcomp_templates.py` for the full "
            "K&D (2018) Kompaneets treatment.",
            stacklevel=3,
        )

    log_r_hot = jnp.log10(r_hot_cm)
    log_r_warm_grid = jnp.linspace(log_r_hot, log_r_warm, n_radii)
    r_warm_grid = 10.0**log_r_warm_grid

    r_ratio_warm = r_warm_grid / r_isco_cm
    torque_warm = jnp.maximum(1.0 - jnp.sqrt(1.0 / r_ratio_warm), 1e-30) ** 0.25
    t_warm = t_in * r_ratio_warm ** (-0.75) * torque_warm

    d_log_r_warm = log_r_warm_grid[1] - log_r_warm_grid[0]
    dr_warm = r_warm_grid * jnp.log(10.0) * d_log_r_warm

    if _NTHCOMP_AVAILABLE:
        # Full nthcomp path (K&D 2018 Section 2.2): solve the Kompaneets equation
        # per annulus via trilinear table interpolation.  Each annulus emits the
        # same bolometric power as the NT blackbody (energy conservation), but the
        # spectral shape is the actual Comptonized spectrum rather than a modified
        # blackbody.
        # Reference: agnsed.py _do_warm_annuli (scotthgn/pyAGNSED)
        # Note: trapezoid() works with unsorted x values; we avoid explicit
        # sorting/indexing to prevent NaN gradients in JAX autodiff (gather ops
        # on float32 data from interpolation can produce NaN during backprop).
        def _warm_ring(r_cm, t_ring, dr_ring):
            b_nu_plain = _planck_lnu(nu, t_ring)
            p_plain = jnp.abs(jnp.trapezoid(b_nu_plain, nu))
            area = jnp.pi * 2.0 * jnp.pi * r_cm * dr_ring
            l_total = p_plain * area * jnp.maximum(agn_cos_inc, 0.01)
            kTbb_keV = _K_BOLTZ_KEV * t_ring
            shape = _nthcomp_lnu_interp(nu, agn_gamma_warm, agn_kt_warm, kTbb_keV)
            p_shape = jnp.abs(jnp.trapezoid(shape, nu))
            shape_norm = shape / jnp.maximum(p_shape, 1e-100)
            return shape_norm * l_total
    else:
        # Simplified fallback: modified blackbody proxy (QSOSED/Synthesizer style).
        # Accurate for optical/UV photometric fitting (Paper I).
        # Run scripts/build_nthcomp_templates.py for the full K&D (2018) treatment.
        def _warm_ring(r_cm, t_ring, dr_ring):
            b_nu_plain = _planck_lnu(nu, t_ring)
            b_nu_mod = _warm_comptonization_lnu(nu, t_ring, nu_warm, agn_gamma_warm)
            p_plain = jnp.abs(jnp.trapezoid(b_nu_plain, nu))
            p_comp = jnp.abs(jnp.trapezoid(b_nu_mod, nu))
            renorm = p_plain / jnp.maximum(p_comp, 1e-100)
            # dL_nu = pi * B_nu_mod * dA * cos(i) (Rybicki & Lightman 1979, Eq. 1.6)
            area = jnp.pi * 2.0 * jnp.pi * r_cm * dr_ring
            return b_nu_mod * renorm * area * jnp.maximum(agn_cos_inc, 0.01)

    l_nu_warm = jnp.sum(jax.vmap(_warm_ring)(r_warm_grid, t_warm, dr_warm), axis=0)

    # ===============================================================
    # Zone 3: Hot corona (R_ISCO < r < R_hot)
    # ===============================================================
    # Self-consistent Gamma_hot (Beloborodov 1999 / K&D 2018 Eq. 6):
    #   Gamma_hot = (7/3) * (L_diss / L_seed)^{-0.1}
    # L_seed: geometric integral of F_NT * Θ(R)/π over disc radii R > R_hot
    # (K&D 2018 Eq. 3), where Θ(R) = θ_0 - sin(2θ_0)/2, sin θ_0 = R_hot/R.
    # This replaces the previous approximation of using the warm zone bolometric
    # luminosity as a proxy; the geometric integral correctly accounts for
    # the covering fraction that decreases as 1/R^2 far from the corona,
    # concentrating the seed contribution near R_hot where Θ/π → 0.5.
    l_seed_geom = _l_seed_geometric(r_isco_cm, r_hot_cm, r_out_cm, t_in)
    gamma_hard_sc = beloborodov_gamma_hot(l_hot_erg, l_seed_geom)

    # Use self-consistent value or the user-supplied value
    gamma_hard_eff = jnp.where(agn_self_consistent_gamma, gamma_hard_sc, agn_gamma_hard)
    l_nu_hot = _hot_corona_lnu(nu, l_hot_erg, gamma_hard_eff, kt_hot_erg)

    # ===============================================================
    # Combine and normalize
    # ===============================================================
    l_nu_total = l_nu_outer + l_nu_warm + l_nu_hot

    # Normalize all 3 zones to L_bol * agn_frac
    l_bol_requested = 10.0**agn_log_lbol * _LSUN_ERG * agn_frac
    l_nu_integral = jnp.trapezoid(l_nu_total, nu)
    l_nu_integral_safe = jnp.maximum(jnp.abs(l_nu_integral), 1e-100)
    scale = l_bol_requested / l_nu_integral_safe

    return l_nu_total * scale


# ===================================================================
# Model 4: ADAF + truncated disc (low-luminosity AGN)
# ===================================================================


def adaf_disc(
    wavelength: jnp.ndarray,
    agn_log_lbol: float,
    agn_frac: float = 0.1,
    agn_log_mbh: float = 8.0,
    agn_log_ledd: float = -3.0,
    agn_r_tr: float = 100.0,
    agn_adaf_beta: float = 0.5,
    agn_adaf_delta: float = 0.01,
    agn_cos_inc: float = 0.5,
    n_radii: int = 50,
    **_kwargs,
) -> jnp.ndarray:
    """ADAF + truncated disc model for low-luminosity AGN.

    At low accretion rates (L/L_Edd < 0.01), the inner disc transitions
    to an advection-dominated accretion flow. The outer disc remains
    as a standard Shakura-Sunyaev disc truncated at ``r_tr``.

    The ADAF SED has three components:

    1. **Synchrotron** (radio-mm): L_nu ~ nu^(1/3) * exp(-nu/nu_c)
    2. **Bremsstrahlung** (X-ray): L_nu ~ nu^(-0.5) * exp(-h*nu/kT_e)
    3. **Inverse Compton** (hard X-ray): L_nu ~ nu^(-(p-1)/2)

    The truncated outer disc contributes UV/optical emission from
    ``r_tr`` outward.

    Based on Mahadevan 1997, ApJ 477, 585 and Nemmen+2014.

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Rest-frame wavelength [Angstrom].
    agn_log_lbol : float
        log10(L_bol / Lsun). Total AGN bolometric luminosity.
    agn_frac : float
        Fraction of L_bol emitted by this component (0 to 1). Default 0.1.
    agn_log_mbh : float
        log10(M_BH / Msun). Determines gravitational radius and ADAF
        synchrotron peak frequency. Typical range: 6 to 10. Default 8.0.
    agn_log_ledd : float
        log10(L / L_Edd). Eddington ratio. For ADAF regime, should be
        < -2. Default -3.0 (sub-Eddington).
    agn_r_tr : float
        Truncation radius in gravitational radii (R_g = GM/c^2). Inner
        edge of the thin disc / outer edge of ADAF. Typical: 10-1000 R_g.
        Default 100.0.
    agn_adaf_beta : float
        Ratio of magnetic to total pressure in ADAF (0 to 1). Controls
        the synchrotron emission strength. Default 0.5.
    agn_adaf_delta : float
        Fraction of viscous energy directly heating electrons (0 to 1).
        Controls the electron temperature. Default 0.01.
    agn_cos_inc : float
        Cosine of inclination angle. Default 0.5 (60 deg).
    n_radii : int
        Number of radial integration points for the outer disc. Default 50.

    Returns
    -------
    array, shape (n_wave,)
        Specific luminosity L_nu [erg s^-1 Hz^-1].

    Notes
    -----
    The total ADAF luminosity scales as ~L_bol * r_ISCO / r_tr because
    most gravitational energy is advected into the black hole rather than
    radiated. Larger truncation radii therefore produce weaker ADAF
    emission and stronger outer disc emission.

    The synchrotron peak frequency scales as ~10^12 * (M/10^8)^(-1/2) Hz,
    placing it in the sub-mm/mm regime for typical SMBH masses.

    References
    ----------
    - Mahadevan 1997, ApJ, 477, 585
    - Nemmen et al. 2014, MNRAS, 438, 2804
    - Lopez et al. 2024
    """
    nu = _wavelength_to_nu(wavelength)
    l_bol_erg = 10.0**agn_log_lbol * _LSUN_ERG

    # --- Black hole parameters ---
    r_g = _gravitational_radius(agn_log_mbh)
    r_isco_rg = _isco_radius(0.0)  # Schwarzschild for LLAGN
    r_tr_safe = jnp.maximum(agn_r_tr, r_isco_rg + 1.0)  # r_tr > r_isco

    # Eddington ratio and accretion rate
    l_edd = _eddington_luminosity(agn_log_mbh)
    l_edd_ratio = jnp.clip(10.0**agn_log_ledd, 1e-10, 1.0)

    # Radiative efficiency (Novikov-Thorne for Schwarzschild)
    eta = 1.0 - jnp.sqrt(1.0 - 2.0 / (3.0 * r_isco_rg))
    mdot = l_edd_ratio * l_edd / (eta * _C_LIGHT**2)

    # ===============================================================
    # ADAF component (r < r_tr)
    # ===============================================================
    # ADAF radiative luminosity fraction: L_adaf ~ L_bol * (r_isco / r_tr)
    # Most energy is advected, not radiated
    adaf_efficiency = r_isco_rg / r_tr_safe
    l_adaf_erg = l_bol_erg * adaf_efficiency

    # --- ADAF electron temperature ---
    # T_e scales with (delta/m_dot)^0.5 from the electron energy balance.
    # Mahadevan (1997, ApJ 477 585) Eq. 4-9 give T_e from the coupled
    # electron-proton energy equations; the key scaling is
    #   T_e ∝ (δ / ṁ)^0.5 × virial_constant
    # where ṁ = L/L_Edd.  At m_dot = delta = 1 this recovers ~5×10^9 K.
    m_dot_dimensionless = jnp.clip(l_edd_ratio, 1e-10, 1.0)
    t_e = 5e9 * jnp.sqrt(jnp.maximum(agn_adaf_delta, 1e-6) / m_dot_dimensionless)
    t_e = jnp.clip(t_e, 1e8, 5e11)  # physical range [K]

    # --- Synchrotron component ---
    # Peak frequency (Mahadevan 1997 Eq. 24):
    #   nu_peak ∝ M_BH^{-1/2} * m_dot^{1/2}
    # The M_BH and m_dot scalings follow directly from nu_c = (3/2)*nu_cyclotron*(T_e/m_e c^2)^2
    # combined with B ∝ m_dot^{1/2} M_BH^{-1/2} in the ADAF equipartition field.
    nu_peak_sync = 1e12 * (10.0**agn_log_mbh / 1e8) ** (-0.5) * m_dot_dimensionless**0.5
    # Synchrotron spectrum: two-regime shape (Mahadevan 1997 Eq. 24).
    # Below the self-absorption frequency nu_sa ~ nu_peak/3: nu^{5/2} (Rayleigh-Jeans).
    # Above nu_sa: nu^{1/3} (optically-thin synchrotron).
    # Join continuously at nu_sa.
    nu_sa = nu_peak_sync / 3.0  # self-absorption break (Mahadevan 1997)
    nu_ratio_sync = nu / jnp.maximum(nu_peak_sync, 1.0)
    nu_ratio_sa = nu / jnp.maximum(nu_sa, 1.0)
    sync_thin = nu_ratio_sync ** (1.0 / 3.0) * jnp.exp(-jnp.clip(nu_ratio_sync / 3.0, 0.0, 500.0))
    # Continuous join at nu_sa: nu_sa^{1/3} = nu_sa^{5/2}/nu_sa^{5/2} * nu_sa^{1/3}
    sync_thick = nu_ratio_sa**2.5 * (nu_sa / jnp.maximum(nu_peak_sync, 1.0)) ** (1.0 / 3.0)
    sync_shape = jnp.where(nu < nu_sa, sync_thick, sync_thin)

    # --- Bremsstrahlung component ---
    # Non-relativistic bremsstrahlung is spectrally flat (nu^0) below the exponential
    # cutoff at kT_e/h (Rybicki & Lightman Ch. 5; Mahadevan 1997 Eq. 3).
    nu_brem = _K_BOLTZ * t_e / _H_PLANCK
    nu_ratio_brem = nu / jnp.maximum(nu_brem, 1.0)
    brem_shape = jnp.exp(-jnp.clip(nu_ratio_brem, 0.0, 500.0))  # flat spectrum (nu^0)

    # --- Inverse Compton component ---
    # Power-law with index -(p-1)/2 where p ~ 2.5 for ADAF
    # IC is sub-dominant in sub-mm but contributes at hard X-ray
    p_ic = 2.5
    ic_shape = nu ** (-(p_ic - 1.0) / 2.0) * jnp.exp(
        -jnp.clip(nu / jnp.maximum(nu_brem, 1.0), 0.0, 500.0)
    )

    # Relative weights (Mahadevan 1997): synchrotron dominates at low nu,
    # bremsstrahlung at intermediate, IC at high
    # beta controls synchrotron strength (magnetic field)
    beta_safe = jnp.clip(agn_adaf_beta, 0.01, 0.99)
    sync_weight = beta_safe
    brem_weight = 1.0 - beta_safe
    ic_weight = 0.1 * beta_safe  # IC is typically sub-dominant

    adaf_shape = sync_weight * sync_shape + brem_weight * brem_shape + ic_weight * ic_shape

    # Normalize ADAF to l_adaf_erg
    sort_idx = jnp.argsort(nu)
    adaf_integral = jnp.trapezoid(adaf_shape[sort_idx], nu[sort_idx])
    adaf_integral_safe = jnp.maximum(jnp.abs(adaf_integral), 1e-100)
    l_nu_adaf = l_adaf_erg * adaf_shape / adaf_integral_safe

    # ===============================================================
    # Truncated outer disc (r > r_tr)
    # ===============================================================
    r_in_cm = r_tr_safe * r_g  # inner edge at truncation radius (= r_tr)
    r_out_cm = 1000.0 * r_isco_rg * r_g  # self-gravity limit

    # Inner temperature normalization at r_tr (the truncated disc inner edge).
    # For a truncated disc the zero-torque inner BC is at r_tr, not r_isco,
    # because no material exists below r_tr (Manmoto+2000; Meyer+2000).
    t_in = (
        3.0
        * _G_GRAV
        * 10.0**agn_log_mbh
        * _MSUN_G
        * mdot
        / (8.0 * jnp.pi * _SIGMA_SB * r_in_cm**3)
    ) ** 0.25

    # Radial grid (logarithmic spacing from r_tr to r_out)
    log_r_min = jnp.log10(r_in_cm)
    log_r_max = jnp.log10(r_out_cm)
    log_r_grid = jnp.linspace(log_r_min, log_r_max, n_radii)
    r_grid = 10.0**log_r_grid

    # Temperature profile: T(r) = T_in * (r/r_tr)^{-3/4} * (1 - sqrt(r_tr/r))^{1/4}
    # Zero-torque at r_tr (disc inner edge), consistent with Novikov-Thorne
    # applied to a disc truncated at r_tr rather than at r_isco.
    r_ratio = r_grid / r_in_cm
    torque_correction = jnp.maximum(1.0 - jnp.sqrt(1.0 / r_ratio), 1e-30) ** 0.25
    t_profile = t_in * r_ratio ** (-0.75) * torque_correction

    d_log_r = log_r_grid[1] - log_r_grid[0]
    dr = r_grid * jnp.log(10.0) * d_log_r

    def _ring_lnu(r_cm, t_ring, dr_ring):
        b_nu = _planck_lnu(nu, t_ring)
        # dL_nu = pi * B_nu * dA * cos(i) (Rybicki & Lightman 1979, Eq. 1.6)
        area = jnp.pi * 2.0 * jnp.pi * r_cm * dr_ring
        return b_nu * area * jnp.maximum(agn_cos_inc, 0.01)

    ring_contributions = jax.vmap(_ring_lnu)(r_grid, t_profile, dr)
    l_nu_disc = jnp.sum(ring_contributions, axis=0)

    # ===============================================================
    # Combine and normalize
    # ===============================================================
    # Disc luminosity: L_bol * (1 - r_isco/r_tr) [remaining after ADAF]
    l_disc_erg = l_bol_erg * (1.0 - adaf_efficiency)
    disc_integral = jnp.trapezoid(l_nu_disc[sort_idx], nu[sort_idx])
    disc_integral_safe = jnp.maximum(jnp.abs(disc_integral), 1e-100)
    l_nu_disc_norm = l_disc_erg * l_nu_disc / disc_integral_safe

    l_nu_total = l_nu_adaf + l_nu_disc_norm

    # Apply overall AGN fraction scaling
    l_bol_requested = l_bol_erg * agn_frac
    total_integral = jnp.trapezoid(l_nu_total[sort_idx], nu[sort_idx])
    total_integral_safe = jnp.maximum(jnp.abs(total_integral), 1e-100)
    scale = l_bol_requested / total_integral_safe

    return l_nu_total * scale
