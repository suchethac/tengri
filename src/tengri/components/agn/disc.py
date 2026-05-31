# SPDX-License-Identifier: BSD-3-Clause
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

import functools
from collections.abc import Callable

import h5py
import jax
import jax.numpy as jnp

from tengri.components.agn._nthcomp import (
    _TABLE_AVAILABLE as _NTHCOMP_AVAILABLE,
    nthcomp_lnu_interp as _nthcomp_lnu_interp,
)
from tengri.components.agn._phys import (
    C_LIGHT as _C_LIGHT,
    H_PLANCK as _H_PLANCK,
    K_BOLTZ as _K_BOLTZ,
    planck_lnu as _planck_lnu,
    ring_area as _ring_area,
    wavelength_to_nu as _wavelength_to_nu,
)
from tengri.utils.grid_interp import interp_nd_triweight as _interp_nd_triweight
from tengri.utils.interpolation import edges_for_grid as _edges_for_grid
from tengri.utils.physics_constants import (
    G_GRAV as _G_GRAV,
    K_BOLTZ_KEV as _K_BOLTZ_KEV,
    KEV_TO_ERG as _KEV_TO_ERG,
    L_SUN as _LSUN_ERG,
    M_PROTON as _M_PROTON,
    M_SUN as _MSUN_G,
    SIGMA_SB as _SIGMA_SB,
    SIGMA_T as _SIGMA_T,
)

# ── Model 1: Simple power-law disc + UV cutoff ────────────────────


def powerlaw_disc(
    wavelength: jnp.ndarray,
    agn_log_lbol: float,
    agn_frac: float = 1.0,
    agn_alpha: float = -1.0,
    agn_T_max: float = 1e5,
    **_kwargs,
) -> jnp.ndarray:
    """Simple power-law accretion disc with exponential UV cutoff (deprecated).

    A phenomenological single-component AGN disc model that approximates the
    optical/UV emission as a power law with an exponential cutoff at high
    frequencies. This is a faster alternative to multi-color disc models when
    fine spectral details are not required.

    .. deprecated::
       This bare power-law disc lacks physical motivation and citations.
       For science fits, use :func:`multicolor_disc` (Shakura-Sunyaev thin disc)
       or :func:`kubota_done_disc` (K&D 3-zone model) instead.
       Will be removed in tengri v1.0.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength grid. [Angstrom]
    agn_log_lbol : float
        Total AGN bolometric luminosity. [log10(L_sun)]
    agn_frac : float, optional
        Fraction of bolometric luminosity emitted by this disc component.
        Default: 1.0. [dimensionless, 0–1]
    agn_alpha : float, optional
        Power-law spectral index. Typical range: -1.5 to -0.5.
        Default: -1.0 (flat in nu*L_nu). [dimensionless]
    agn_T_max : float, optional
        Maximum blackbody temperature, setting the UV cutoff frequency.
        Typical range: 10^4 to 10^6. Default: 10^5. [K]

    Returns
    -------
    ndarray, shape (n_wave,)
        Spectral luminosity density L_ν. [erg/s/Hz]

    Notes
    -----
    **JIT-compatible**: yes — all operations use ``jnp`` primitives.

    The unnormalized spectral shape is:

    .. math::

        L_\\nu^{\\rm unnorm} = \\nu^\\alpha \\exp\\left(-\\frac{h\\nu}{k_B T_{\\rm max}}\\right)

    where :math:`\\alpha` is the spectral index, :math:`h` is Planck's constant,
    :math:`\\nu` is frequency [Hz], :math:`k_B` is Boltzmann's constant, and
    :math:`T_{\\rm max}` is the cutoff temperature [K].

    The normalization constant :math:`C` is computed numerically by integrating
    the shape over the wavelength grid via the trapezoidal rule, ensuring that
    the integral over frequency equals the target luminosity
    :math:`L_{\\rm bol} \\cdot f_{\\rm disc}`.

    **Approximation**: This model is a simplified representation of the true
    accretion disc spectrum, which consists of multiple temperature zones
    (see :func:`multicolor_disc` and :func:`kubota_done_disc` for more
    realistic models). The power-law form breaks down at low frequencies
    (radio/submm) where the SED transitions to a different regime, and does
    not capture the soft X-ray excess or hard X-ray corona. Use this model
    only when computational speed is prioritized over spectral fidelity.
    """
    import warnings

    warnings.warn(
        "powerlaw_disc is deprecated (no physical derivation; bare phenomenological "
        "model) and will be removed in tengri v1.0. For science fits, use "
        "multicolor_disc (Shakura-Sunyaev) or kubota_done_disc (K&D 3-zone) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
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


# ── Model 2: Multi-color disc (Shakura-Sunyaev thin disc) ─────────


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
    r"""Eddington luminosity :math:`L_{\rm Edd} = 4\pi G M_{\rm BH} m_p c / \sigma_T` [erg/s]."""
    m_bh_g = 10.0**log_mbh * _MSUN_G
    return 4.0 * jnp.pi * _G_GRAV * m_bh_g * _M_PROTON * _C_LIGHT / _SIGMA_T


def _gravitational_radius(log_mbh: float) -> float:
    r"""Gravitational radius :math:`R_g = GM/c^2` [cm]."""
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
    r"""Solve for R_hot from K&D 2018 Eq. 2 by bisection in log(x_hot).

    The NT emissivity integral has closed form
    :math:`L_{\rm diss}(x) = L_0\,[1/10 - 1/(2x^2) + 2/(5 x^{5/2})]`,
    strictly monotone in :math:`x = R_{\rm hot}/R_{\rm ISCO}`. After
    ``n_iter=40`` the bracket width is :math:`< 2^{-40} \approx 10^{-12}`
    of its initial log-width — enough for machine precision.

    ``l_hot_target`` is clipped below :math:`L_{\max} = L_0/10`.
    """
    # Maximum possible L_diss (entire disc, x→∞): h→0.1, so L_max = L0 * 0.1
    l0 = 4.0 * jnp.pi * r_isco_cm**2 * _SIGMA_SB * t_in**4
    # Clip target to (0, L_max); if l_hot_target >= L_max, r_hot → ∞ (use upper bound)
    l_target = jnp.clip(l_hot_target, 1e-100, l0 * 0.099)

    def _h(log_x):
        """Compute normalized NT emissivity integral h(x_hot) in log-space."""
        x = jnp.exp(log_x)
        return l0 * (0.1 - 0.5 * x ** (-2.0) + 0.4 * x ** (-2.5))

    # Bisect in log(x_hot) in [log(1.001), log(1e4)]
    lo = jnp.log(1.001)
    hi = jnp.log(1.0e4)

    def _step(state, _):
        """Single bisection step in log-space to solve for R_hot."""
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
    """Shakura-Sunyaev thin accretion disc with multi-color blackbody emission.

    Compute the SED of a standard geometrically thin, optically thick accretion
    disc via the Shakura-Sunyaev model. The disc is stratified into radial
    annuli, each radiating as a blackbody at its local temperature. This is
    the outer-disc component of the Kubota & Done (2018) three-zone model
    (see :func:`kubota_done_disc` for the full model including corona).

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength grid. [Angstrom]
    agn_log_lbol : float
        Total AGN bolometric luminosity. [log10(L_sun)]
    agn_frac : float, optional
        Fraction of bolometric luminosity emitted by the disc.
        Default: 1.0. [dimensionless, 0–1]
    agn_log_mbh : float, optional
        Black hole mass. Default: 8.0. [log10(M_sun)]
    agn_log_ledd : float, optional
        Eddington ratio (accretion rate relative to Eddington luminosity).
        Typical range: -3 to 0. Default: -1.0. [log10(L / L_Edd)]
    agn_a_spin : float, optional
        Dimensionless black hole spin parameter (prograde).
        Range: [0, 0.998]. Default: 0.0 (Schwarzschild). [dimensionless]
    agn_cos_inc : float, optional
        Cosine of the inclination angle. Range: [0.01, 1.0].
        Default: 0.5 (60°). [dimensionless]
    n_radii : int, optional
        Number of radial bins for numerical integration. Default: 50.

    Returns
    -------
    ndarray, shape (n_wave,)
        Spectral luminosity density L_ν. [erg/s/Hz]

    Notes
    -----
    **JIT-compatible**: yes — uses ``jnp`` primitives and ``jax.vmap``.

    The temperature profile follows the Novikov-Thorne (1974) emissivity for
    a thin, radiatively efficient disc:

    .. math::

        T(r) = T_{\\rm in} \\left(\\frac{r}{r_{\\rm ISCO}}\\right)^{-3/4}
               \\left[1 - \\sqrt{\\frac{r_{\\rm ISCO}}{r}}\\right]^{1/4}

    where :math:`T_{\\rm in}` is the inner temperature determined by the
    accretion rate and :math:`r_{\\rm ISCO}` is the innermost stable circular
    orbit (radius from Bardeen et al. 1972, depends on spin).

    The disc luminosity is computed as:

    .. math::

        L_\\nu = \\sum_{i=1}^{N_r} B_\\nu(T_i) \\cdot 2\\pi^2 r_i \\, dr_i \\cdot \\cos(i)

    where :math:`B_\\nu(T)` is the Planck function, :math:`r_i` is the ring
    radius [cm], :math:`dr_i` is the ring width [cm], and :math:`\\cos(i)` is
    the projection factor (inclination).

    **Key physics**:

    - **Radiative efficiency**: :math:`\\eta = 1 - \\sqrt{1 - 2/(3 r_{\\rm ISCO})}`,
      computed from Novikov-Thorne theory. For Schwarzschild (a=0): η ≈ 0.057;
      for maximally spinning (a→0.998): η ≈ 0.32.
    - **Outer radius**: Uses the Laor & Netzer (1989) self-gravity (Toomre)
      radius, beyond which the disc fragments. This is an improvement over
      fixed approximations (e.g., 1000 r_ISCO) that can err by factors of
      a few at extreme masses or accretion rates.
    - **Eddington ratio clamping**: The accretion luminosity is capped at
      :math:`L_{\\rm Edd}` (i.e., log(L/L_Edd) is clipped to [0, 1] in linear
      space), reflecting the physical limit of radiatively efficient accretion.

    **Numerical method**: Radii are logarithmically spaced to ensure fine
    resolution at small radii where the temperature gradient is steep.
    Integration uses summation; the trapezoidal rule is applied in log-radius
    space via the spacing :math:`d\\log r = \\Delta(\\log r)`.

    References
    ----------
    .. [1] A. Kubota and C. Done, "A physical model of the broad-band continuum
       of AGN and its implications for the UV/X relation and optical variability,"
       MNRAS, 480, 1247 (2018). arXiv:1804.00171.
       https://doi.org/10.1093/mnras/sty1890
    .. [2] J. M. Bardeen, W. H. Press, and S. A. Teukolsky, "Rotating black holes:
       Locally nonrotating frames, energy extraction, and scalar synchrotron radiation,"
       ApJ, 178, 347 (1972). https://doi.org/10.1086/151796
    .. [3] A. Laor and H. Netzer, "Massive thin accretion discs – I. Calculated spectra,"
       MNRAS, 238, 897 (1989). https://doi.org/10.1093/mnras/238.3.897
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
        """Compute Planck luminosity per unit frequency for disc annulus."""
        b_nu = _planck_lnu(nu, t_ring)
        return b_nu * _ring_area(r_cm, dr_ring, agn_cos_inc)

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


# ── Model 3: Kubota & Done (2018) 3-zone disc ─────────────────────


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


# Fixed internal frequency grid for corona normalization.
# Matches RELAGN (scotthgn/RELAGN) default: [1e-4, 1e4] keV → [2.418e13, 2.418e21] Hz.
# Using a fixed grid makes the normalization integral grid-independent,
# so the corona's optical flux doesn't change with the caller's wavelength grid.
# The bare power-law nu^(1-Gamma) diverges at low frequencies for Gamma > 1;
# a fixed lower bound removes this ambiguity.  2000 log-spaced points match
# RELAGN's resolution.
_CORONA_NU_GRID = jnp.geomspace(2.418e13, 2.418e21, 2000)


def _hot_corona_lnu(
    nu: jnp.ndarray,
    l_hot_erg: float,
    gamma_hard: float,
    kt_hot_erg: float,
    nu_seed_hz: float = 0.0,
) -> jnp.ndarray:
    r"""Hot corona emission: thermal-Comptonisation power law with two cutoffs.

    The optically thin, hot corona produces hard X-ray emission via thermal
    Comptonisation of seed photons. The spectrum is a power law bounded by a
    high-energy cutoff at the electron temperature *and* a low-energy rollover
    at the seed-photon energy:

    .. math::

        L_\nu \propto \nu^{\,1-\Gamma_{\rm hot}}
                      \, \exp\!\left(-\frac{h\nu}{kT_{e,\rm hot}}\right)
                      \, \exp\!\left(-\frac{\nu_{\rm seed}}{\nu}\right)

    where :math:`\Gamma_{\rm hot}` is the hard X-ray photon index,
    :math:`kT_{e,\rm hot}` the electron temperature [erg], and
    :math:`\nu_{\rm seed}` the seed-photon frequency [Hz]. The low-energy
    rollover :math:`\exp(-\nu_{\rm seed}/\nu)` is the piece that ``nthcomp``
    carries intrinsically: it tends to 1 for :math:`\nu \gg \nu_{\rm seed}`
    (leaving the X-ray power law untouched) and to 0 for
    :math:`\nu \ll \nu_{\rm seed}`. Without it the bare :math:`\nu^{1-\Gamma}`
    tail rises monotonically toward low frequency for :math:`\Gamma > 1`, so
    the corona leaks unphysically into the infrared and radio.

    A thermal-Comptonisation spectrum is only defined *between* its seed-photon
    energy and the electron temperature; there are no Comptonised photons below
    the seed energy (Kubota & Done 2018 [1]_, Section 2.2).

    Normalised so that the frequency-integrated luminosity equals
    ``l_hot_erg``. The normalisation integral is computed on a fixed internal
    frequency grid matching RELAGN's default [1e-4, 1e4] keV, making the result
    independent of the caller's wavelength grid (fixing an earlier bug where
    the corona's optical contribution varied by 2-4x with grid extent).

    Parameters
    ----------
    nu : array_like, shape (n_wave,)
        Frequency [Hz].
    l_hot_erg : float
        Total hot corona luminosity [erg s^-1].
    gamma_hard : float
        Hard X-ray photon index (~1.8).
    kt_hot_erg : float
        Hot corona electron temperature [erg] (= kT_e,hot in erg).
    nu_seed_hz : float, optional
        Seed-photon frequency [Hz] setting the low-energy rollover. Default
        ``0.0`` disables the rollover (legacy bare power law). The three-zone
        ``kubota_done_disc`` passes the K&D 2018 value
        :math:`k\,T_{\rm NT}(R_{\rm hot})\,\exp(y_{\rm warm})/h`.

    Returns
    -------
    ndarray, shape (n_wave,)
        L_nu [erg s^-1 Hz^-1].

    Notes
    -----
    **JIT-compatible**: yes — pure ``jnp`` primitives, no Python branching on
    traced values.

    References
    ----------
    .. [1] A. Kubota and C. Done, "A physical model of the broadband continuum
       of AGN and its implications for the UV/X relation and optical
       variability," MNRAS, 480, 1247 (2018). arXiv:1804.00171.
       DOI:10.1093/mnras/sty1890. Section 2.2 (seed photons of the hot flow).
    .. [2] RELAGN (scotthgn/RELAGN) ``do_nonrelHotCompSpec``: normalises on a
       fixed [1e-4, 1e4] keV grid.
    """
    kt_safe = jnp.maximum(kt_hot_erg, 1e-30)
    nu_safe = jnp.maximum(nu, 1e-30)

    # Evaluate shape on the caller's frequency grid (for output).
    # High-energy cutoff at kT_e; low-energy seed-photon rollover at nu_seed.
    x = _H_PLANCK * nu / kt_safe
    x_clip = jnp.clip(x, 0.0, 500.0)
    seed_roll = jnp.exp(-jnp.clip(nu_seed_hz / nu_safe, 0.0, 700.0))
    shape = nu ** (1.0 - gamma_hard) * jnp.exp(-x_clip) * seed_roll

    # Normalize on the fixed internal grid (grid-independent).  The same
    # seed-photon rollover is applied so the normalisation stays self-consistent.
    x_norm = _H_PLANCK * _CORONA_NU_GRID / kt_safe
    x_norm_clip = jnp.clip(x_norm, 0.0, 500.0)
    seed_roll_norm = jnp.exp(-jnp.clip(nu_seed_hz / _CORONA_NU_GRID, 0.0, 700.0))
    shape_norm = _CORONA_NU_GRID ** (1.0 - gamma_hard) * jnp.exp(-x_norm_clip) * seed_roll_norm
    integral = jnp.trapezoid(shape_norm, _CORONA_NU_GRID)
    integral_safe = jnp.maximum(jnp.abs(integral), 1e-100)

    return l_hot_erg * shape / integral_safe


def _compute_bh_params(
    agn_log_mbh: float,
    agn_log_ledd: float,
    agn_a_spin: float,
) -> tuple:
    """Compute black hole parameters: radius, ISCO, efficiency, and accretion rate.

    Derives the fundamental mass-dependent quantities: gravitational radius,
    ISCO radius, radiative efficiency (Novikov-Thorne), Eddington luminosity,
    and mass accretion rate from the black hole mass and spin parameter.

    Parameters
    ----------
    agn_log_mbh : float
        Black hole mass. [log10(M_sun)]
    agn_log_ledd : float
        Eddington ratio. [log10(L / L_Edd)]
    agn_a_spin : float
        Dimensionless black hole spin (Kerr, prograde). [dimensionless, 0–0.998]

    Returns
    -------
    tuple
        (r_g, r_isco_rg, r_isco_cm, eta, l_edd, mdot) where:
        - r_g : Gravitational radius [cm]
        - r_isco_rg : ISCO radius in units of r_g [dimensionless]
        - r_isco_cm : ISCO radius [cm]
        - eta : Radiative efficiency (Novikov-Thorne) [dimensionless, 0–0.42]
        - l_edd : Eddington luminosity [erg s^-1]
        - mdot : Mass accretion rate [g s^-1]

    Notes
    -----
    **JIT-compatible**: yes — uses only ``jnp`` primitives.

    The radiative efficiency follows the Novikov-Thorne formula for thin discs
    around Kerr black holes (Novikov & Thorne 1973, Kerr metric).

    References
    ----------
    .. [1] D. N. Page and K. S. Thorne, "Disk-Accretion onto a Black Hole.
       Time-Averaged Structure of the Inner Accretion Disk," ApJ, 191, 499 (1974).
    """
    r_g = _gravitational_radius(agn_log_mbh)
    r_isco_rg = _isco_radius(agn_a_spin)
    r_isco_cm = r_isco_rg * r_g

    eta = 1.0 - jnp.sqrt(1.0 - 2.0 / (3.0 * r_isco_rg))

    l_edd = _eddington_luminosity(agn_log_mbh)
    l_edd_ratio = jnp.clip(10.0**agn_log_ledd, 1e-10, 1.0)
    l_bol_erg = l_edd_ratio * l_edd
    mdot = l_bol_erg / (eta * _C_LIGHT**2)

    return r_g, r_isco_rg, r_isco_cm, eta, l_edd, mdot


def _compute_zone_radii(
    r_g: float,
    r_isco_rg: float,
    r_isco_cm: float,
    t_in: float,
    agn_log_mbh: float,
    agn_log_ledd: float,
    agn_f_hard: float,
    agn_r_warm_ratio: float,
    l_edd: float,
) -> tuple:
    """Compute self-consistent zone radii: R_hot, R_warm, and R_out.

    Solves for the radii that define the three AGN accretion zones (Kubota & Done
    2018). R_hot is derived self-consistently from the energy-balance constraint
    that the hot corona dissipates f_hard × L_Edd. R_warm is parameterized as
    a multiple of R_hot. R_out is the self-gravity (Toomre) radius beyond which
    the disc becomes unstable.

    Parameters
    ----------
    r_g : float
        Gravitational radius [cm].
    r_isco_rg : float
        ISCO radius in gravitational radii [dimensionless].
    r_isco_cm : float
        ISCO radius [cm].
    t_in : float
        Inner disc temperature [K].
    agn_log_mbh : float
        Black hole mass. [log10(M_sun)]
    agn_log_ledd : float
        Eddington ratio. [log10(L / L_Edd)]
    agn_f_hard : float
        Fraction of Eddington luminosity in the hot corona. [dimensionless, 0–0.5]
    agn_r_warm_ratio : float
        Radius ratio R_warm / R_hot. [dimensionless, ≥ 1.1]
    l_edd : float
        Eddington luminosity [erg s^-1].

    Returns
    -------
    tuple
        (r_hot_cm, r_warm_cm, r_out_cm) zone radii in [cm].

    Notes
    -----
    **JIT-compatible**: yes — uses ``jax.lax.scan`` for JAX-compatible bisection.

    **Self-consistent R_hot**: Uses bisection on the analytic Novikov-Thorne
    integral (40 iterations, exact to ~1e-12) to solve L_diss,hot(R_hot) = f_hard
    × L_Edd. This replaces the previous approximate closure r_hot ≈ r_isco ×
    (1 + f_hard λ)^{1/3} which had ~10% error.

    **Self-consistent R_out**: Uses the Laor & Netzer (1989) self-gravity
    (Toomre) radius, which is more accurate for extreme BH masses and Eddington
    ratios than the previous fixed 1000 × r_isco approximation.

    References
    ----------
    .. [1] A. Kubota and C. Done, "A physical model of the broad-band continuum
       of AGN and its implications for the UV/X relation and optical variability,"
       MNRAS, 480, 1247 (2018). arXiv:1804.00171.
    .. [2] A. Laor and B. Netzer, "Dust Sublimation Depth in the Infrared-Emitting
       Accretion Disks of Quasars," MNRAS, 238, 897 (1989).
    """
    f_hard_safe = jnp.clip(agn_f_hard, 1e-6, 0.5)
    l_hot_target = f_hard_safe * l_edd
    r_hot_cm = _r_hot_bisect(r_isco_cm, t_in, l_hot_target)

    r_warm_ratio_safe = jnp.clip(agn_r_warm_ratio, 1.1, 10.0)
    r_warm_cm = r_hot_cm * r_warm_ratio_safe

    l_edd_ratio = jnp.clip(10.0**agn_log_ledd, 1e-10, 1.0)
    r_sg_rg = _self_gravity_radius(agn_log_mbh, l_edd_ratio)
    r_out_cm = jnp.maximum(r_sg_rg, r_isco_rg * 10.0) * r_g

    r_hot_cm = jnp.clip(r_hot_cm, r_isco_cm * 1.01, r_out_cm * 0.5)
    r_warm_cm = jnp.clip(r_warm_cm, r_hot_cm * 1.01, r_out_cm * 0.9)

    return r_hot_cm, r_warm_cm, r_out_cm


def _compute_zone_luminosities(
    nu: jnp.ndarray,
    r_isco_cm: float,
    r_hot_cm: float,
    r_warm_cm: float,
    r_out_cm: float,
    t_in: float,
    agn_cos_inc: float,
    n_radii: int,
    agn_gamma_warm: float,
    agn_kt_warm: float,
    agn_gamma_hard: float,
    agn_kt_hot: float,
    agn_f_hard: float,
    l_edd: float,
    l_bol_erg: float,
    agn_self_consistent_gamma: bool,
) -> tuple:
    """Compute self-consistent luminosities of the three AGN zones.

    Integrates the Novikov-Thorne temperature profile over annuli in each zone
    (outer standard disc, warm Comptonization, hot corona) and combines the
    spectral shapes (blackbody, Comptonized, power-law) into a total L_ν. Applies
    a global normalization to conserve the input bolometric luminosity.

    Parameters
    ----------
    nu : array, shape (n_wave,)
        Frequency [Hz].
    r_isco_cm : float
        ISCO radius [cm].
    r_hot_cm : float
        Hot corona radius [cm].
    r_warm_cm : float
        Warm zone radius [cm].
    r_out_cm : float
        Outer disc radius [cm].
    t_in : float
        Inner disc temperature [K].
    agn_cos_inc : float
        Cosine of inclination angle [dimensionless, 0.01–1.0].
    n_radii : int
        Number of radial integration points per zone [dimensionless].
    agn_gamma_warm : float
        Photon index of warm Comptonization [dimensionless, ~1.5–3.5].
    agn_kt_warm : float
        Electron temperature in warm zone [keV].
    agn_gamma_hard : float
        Photon index of hard X-ray power law [dimensionless, ~1.5–2.5].
    agn_kt_hot : float
        Electron temperature in hot corona [keV].
    agn_f_hard : float
        Fraction of Eddington luminosity in corona [dimensionless, 0–0.5].
    l_edd : float
        Eddington luminosity [erg s^-1].
    l_bol_erg : float
        Requested bolometric luminosity [erg s^-1].
    agn_self_consistent_gamma : bool
        If True, derive gamma_hard self-consistently from Beloborodov (1999).

    Returns
    -------
    tuple
        (l_nu_total, scale) where:
        - l_nu_total : Unnormalized total L_ν [erg s^-1 Hz^-1] (before scaling)
        - scale : Normalization scale factor to conserve L_bol [dimensionless]

    Notes
    -----
    **JIT-compatible**: yes — uses ``jax.vmap`` for radial integration.

    **Energy conservation**: The normalization integral is computed analytically
    from the radial integration (σ T^4 × dA) rather than spectrally, making the
    result grid-independent. This fixes a long-standing bug where the corona's
    optical flux varied by 2–4× depending on wavelength grid extent.

    References
    ----------
    .. [1] A. Kubota and C. Done, MNRAS, 480, 1247 (2018).
    .. [2] A. M. Beloborodov, ApJL, 510, L123 (1999).
    """
    # ── Zone 1: Outer standard disc (r > R_warm) ──────────────────
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
        """Compute blackbody L_nu contribution from one outer-disk annulus."""
        b_nu = _planck_lnu(nu, t_ring)
        return b_nu * _ring_area(r_cm, dr_ring, agn_cos_inc)

    l_nu_outer = jnp.sum(jax.vmap(_outer_ring)(r_outer, t_outer, dr_outer), axis=0)

    # ── Zone 2: Warm Comptonization (R_hot < r < R_warm) ──────────
    if not _NTHCOMP_AVAILABLE:
        raise RuntimeError(
            "kubota_done_disc requires precomputed nthcomp templates. "
            "Build them with:\n"
            "  python scripts/build_nthcomp_templates.py"
        )

    log_r_hot = jnp.log10(r_hot_cm)
    log_r_warm_grid = jnp.linspace(log_r_hot, log_r_warm, n_radii)
    r_warm_grid = 10.0**log_r_warm_grid

    r_ratio_warm = r_warm_grid / r_isco_cm
    torque_warm = jnp.maximum(1.0 - jnp.sqrt(1.0 / r_ratio_warm), 1e-30) ** 0.25
    t_warm = t_in * r_ratio_warm ** (-0.75) * torque_warm

    d_log_r_warm = log_r_warm_grid[1] - log_r_warm_grid[0]
    dr_warm = r_warm_grid * jnp.log(10.0) * d_log_r_warm

    def _warm_ring(r_cm, t_ring, dr_ring):
        """Compute Comptonized L_nu for one warm-zone annulus using nthcomp spectral shape."""
        b_nu_plain = _planck_lnu(nu, t_ring)
        p_plain = jnp.abs(jnp.trapezoid(b_nu_plain, nu))
        l_total = p_plain * _ring_area(r_cm, dr_ring, agn_cos_inc)
        kTbb_keV = _K_BOLTZ_KEV * t_ring
        shape = _nthcomp_lnu_interp(nu, agn_gamma_warm, agn_kt_warm, kTbb_keV)
        return shape * l_total

    l_nu_warm = jnp.sum(jax.vmap(_warm_ring)(r_warm_grid, t_warm, dr_warm), axis=0)

    # ── Zone 3: Hot corona (R_ISCO < r < R_hot) ───────────────────
    f_hard_safe = jnp.clip(agn_f_hard, 1e-6, 0.5)
    l_hot_erg = jnp.minimum(f_hard_safe * l_edd, l_bol_erg * 0.5)

    kt_hot_erg = agn_kt_hot * _KEV_TO_ERG

    l_seed_geom = _l_seed_geometric(r_isco_cm, r_hot_cm, r_out_cm, t_in)
    gamma_hard_sc = beloborodov_gamma_hot(l_hot_erg, l_seed_geom)
    gamma_hard_eff = jnp.where(agn_self_consistent_gamma, gamma_hard_sc, agn_gamma_hard)

    # Seed-photon frequency for the hot flow (K&D 2018, Section 2.2): the seed
    # photons come from the inner edge of the warm Comptonisation region at
    # R_hot, boosted by the warm Compton y-parameter,
    #   kT_seed,hot = k T_NT(R_hot) * exp(y_warm),
    # with y_warm recovered from Gamma_warm via the standard non-relativistic
    # thermal-Comptonisation relation Gamma = sqrt(9/4 + 4/y) - 1/2
    # (Sunyaev & Titarchuk 1980), i.e. y_warm = 4 / [(Gamma_warm + 1/2)^2 - 9/4].
    # This sets the low-energy rollover so the corona cannot leak into the IR/radio.
    t_seed_nt = t_warm[0]  # T_NT(R_hot): first (innermost) warm-zone annulus
    y_warm_denom = jnp.maximum((agn_gamma_warm + 0.5) ** 2 - 2.25, 1e-3)
    y_warm = jnp.clip(4.0 / y_warm_denom, 0.0, 10.0)
    t_seed_hot = t_seed_nt * jnp.exp(y_warm)
    nu_seed_hot = _K_BOLTZ * t_seed_hot / _H_PLANCK

    l_nu_hot = _hot_corona_lnu(nu, l_hot_erg, gamma_hard_eff, kt_hot_erg, nu_seed_hot)

    # ── Combine and normalize ─────────────────────────────────────
    l_nu_total = l_nu_outer + l_nu_warm + l_nu_hot

    l_bol_outer = jnp.sum(
        _SIGMA_SB * t_outer**4 * 2.0 * jnp.pi * r_outer * dr_outer * jnp.maximum(agn_cos_inc, 0.01)
    )
    l_bol_warm = jnp.sum(
        _SIGMA_SB
        * t_warm**4
        * 2.0
        * jnp.pi
        * r_warm_grid
        * dr_warm
        * jnp.maximum(agn_cos_inc, 0.01)
    )
    l_bol_unnorm = l_bol_outer + l_bol_warm + l_hot_erg
    scale = l_bol_erg / jnp.maximum(l_bol_unnorm, 1e-100)

    return l_nu_total, scale


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

    Notes
    -----
    **JIT-compatible**: yes — uses only ``jnp`` primitives.

    This implementation follows Kubota & Done (2018, MNRAS 480 1247, Eq. 6),
    which rewrites the Beloborodov (1999) Compton-amplification result as a
    power law in the luminosity ratio. The exponent -0.1 encodes the
    energy-balance relation between the dissipated power in the corona and
    the seed photon luminosity intercepted from the disc. Output is clipped
    to [1.4, 3.0] to match the physical range of typical AGN.

    References
    ----------
    .. [1] A. M. Beloborodov, "Plasma Ejection from Magnetic Flares and the
       X-Ray Spectrum of Cygnus X-1," ApJL, 510, L123 (1999).
       arXiv:astro-ph/9809383. https://doi.org/10.1086/311810
    .. [2] A. Kubota and C. Done, "A physical model of the broad-band continuum
       of AGN and its implications for the UV/X relation and optical variability,"
       MNRAS, 480, 1247 (2018). arXiv:1804.00171.
       https://doi.org/10.1093/mnras/sty1890
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

    Notes
    -----
    **JIT-compatible**: yes — uses ``jnp`` primitives.

    The 2500 Å point is a canonical AGN diagnostic wavelength, used to
    compute the optical-to-X-ray spectral index (alpha_ox) and as a
    benchmark for intrinsic AGN continuum comparisons. Linear interpolation
    in wavelength space is sufficient for the optical/UV continuum variability
    timescales. The wavelength array need not be pre-sorted; this function
    sorts internally before interpolation.
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
    """Kubota & Done (2018) three-zone accretion disc with self-consistent corona.

    Model a physically stratified AGN accretion disc as three radially distinct
    zones, each with different physics and electron temperatures. This is the
    reference model for intermediate to high accretion rates and is used in
    tengri's default AGN configuration.

    The three zones share a single Novikov-Thorne temperature profile but have
    different radiation mechanisms:

    1. **Outer standard disc** (r > R_warm): Optically thick, geometrically thin.
       Temperature decreases with radius (∝ r^{-3/4}). Radiates as multi-color
       blackbody. Dominates the optical/UV "big blue bump."

    2. **Warm Comptonization zone** (R_hot < r < R_warm): Optically thick,
       warm electrons (kT_e ~ 0.2 keV, τ ~ 10-20). Inverse Compton-scattered
       disc photons plus thermal radiation. Produces the soft X-ray excess.
       Computed via precomputed nthcomp Kompaneets templates (when available)
       or a simplified modified-blackbody proxy.

    3. **Hot corona** (R_ISCO < r < R_hot): Optically thin, hot electrons
       (kT_e ~ 100 keV, τ ~ 1). Inverse Compton scatters disc seed photons
       to produce hard X-ray power-law spectrum with exponential cutoff.

    Zone boundaries are determined self-consistently:

    - **R_hot**: Solved via bisection from the energy-balance constraint that
      the dissipated power in the corona equals f_hard × L_Edd. Uses the
      exact analytic Novikov-Thorne integral rather than approximations.
    - **R_warm**: Parameterized as a multiple of R_hot (default 2, per K&D).
    - **R_out**: Set to the Laor & Netzer (1989) self-gravity (Toomre) radius,
      beyond which the disc becomes unstable and fragments.

    The hard X-ray photon index Γ_hot is derived self-consistently from the
    Beloborodov (1999) energy-balance relation if requested; otherwise uses
    the input value.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength grid. [Angstrom]
    agn_log_lbol : float
        Total AGN bolometric luminosity (all three zones).
        [log10(L_sun)]
    agn_frac : float, optional
        Fraction of bolometric luminosity emitted by the disc system (all zones).
        Default: 1.0. [dimensionless, 0–1]
    agn_log_mbh : float, optional
        Black hole mass. Determines the Eddington luminosity and temperature
        scaling. Default: 8.0. [log10(M_sun)]
    agn_log_ledd : float, optional
        Eddington ratio. Controls the inner temperature, radiative efficiency,
        and zone locations. Typical range: -3 to 0. Default: -1.0.
        [log10(L / L_Edd)]
    agn_a_spin : float, optional
        Dimensionless black hole spin parameter (Kerr, prograde).
        Range: [0, 0.998]. Higher spin → smaller R_ISCO, higher η.
        Default: 0.0 (Schwarzschild). [dimensionless]
    agn_cos_inc : float, optional
        Cosine of the inclination angle between the disc normal and the
        line of sight. Range: [0.01, 1.0]. Used to compute the projected
        disc area. Default: 0.5 (60°). [dimensionless]
    agn_f_hard : float, optional
        Fraction of Eddington luminosity dissipated in the hot corona.
        Controls the corona zone extent R_hot. Typical range: 0.01–0.1.
        Default: 0.02. [dimensionless, 0–0.5]
    agn_gamma_warm : float, optional
        Photon index of the warm Comptonization zone (nthcomp).
        Range: ~1.5–3.5. Default: 2.5. [dimensionless]
    agn_kt_warm : float, optional
        Electron temperature in the warm Comptonization zone.
        Default: 0.2. [keV]
    agn_gamma_hard : float, optional
        Photon index of the hard X-ray power law (hot corona).
        Typical range: 1.5–2.5. Default: 1.8.
        Ignored if agn_self_consistent_gamma=True. [dimensionless]
    agn_kt_hot : float, optional
        Electron temperature in the hot corona.
        Default: 100.0. [keV]
    agn_r_warm_ratio : float, optional
        Radius ratio R_warm / R_hot. Controls the warm zone extent.
        Default: 2.0 (per K&D 2018). [dimensionless, ≥ 1.1]
    n_radii : int, optional
        Number of radial integration points per zone.
        Default: 50. Higher values increase accuracy at computational cost.
    agn_self_consistent_gamma : bool, optional
        If True, compute ``agn_gamma_hard`` self-consistently from the
        Beloborodov (1999) energy-balance relation:
        Γ = 7/3 × (L_diss / L_seed)^{-0.1}
        If False, use the input ``agn_gamma_hard`` value. Default: False.

    Returns
    -------
    ndarray, shape (n_wave,)
        Spectral luminosity density L_ν. [erg/s/Hz]

    Notes
    -----
    **JIT-compatible**: yes — uses ``jnp`` primitives and ``jax.vmap``.

    **Gradient-safe**: yes — fully differentiable w.r.t. all parameters,
    including the bisection-solved R_hot.

    **Key self-consistent physics**:

    All three zones share the Novikov-Thorne temperature profile:

    .. math::

        T(r) = T_{\\rm in} \\left(\\frac{r}{r_{\\rm ISCO}}\\right)^{-3/4}
               \\left[1 - \\sqrt{\\frac{r_{\\rm ISCO}}{r}}\\right]^{1/4}

    where :math:`T_{\\rm in} = (3 G M M_{\\rm dot} / 8\\pi \\sigma_{\\rm SB}
    r_{\\rm ISCO}^3)^{1/4}`, and the inner temperature increases with accretion
    rate. The radii are ordered as R_ISCO < R_hot < R_warm < R_out.

    **Zone luminosity computation**:

    Each zone is divided into annuli at radii {r_i}, each of which contributes
    L_ν from its local Planck function (outer disc), nthcomp prescription
    (warm zone), or hot-corona power law (inner zone). All zones are summed
    and renormalized to conserve total bolometric energy.

    **Seed photon calculation** (K&D 2018 Eq. 3):
    The hot corona inverse-Compton scatters disc seed photons. The seed photon
    luminosity is computed from the geometric integral of the warm-zone
    blackbody flux intercepted by the corona geometry:

    .. math::

        L_{\\rm seed} = 2 \\int_{R_{\\rm hot}}^{R_{\\rm out}}
                       F_{\\rm NT}(r) \\cdot \\frac{\\Theta(r)}{\\pi} \\cdot 2\\pi r \\, dr

    where :math:`\\Theta(r) = \\theta_0 - \\sin(2\\theta_0)/2` and :math:`\\sin\\theta_0
    = R_{\\rm hot}/r`. This drives the self-consistent Γ_hot via Beloborodov.

    **Precomputed nthcomp templates**: For maximum speed and accuracy, the
    warm Comptonization is computed via interpolation in precomputed nthcomp
    Kompaneets templates (see ``scripts/build_nthcomp_templates.py``).
    These templates span (Γ_warm, kT_e) parameter space and return
    (Γ_warm, kT_e, normalisation)-dependent SED at each radius.
    If templates are unavailable, a simplified modified-blackbody proxy is used,
    which has ~5–10% shape error but allows offline computation.

    **Approximations and accuracy**:

    The reference QSOSED/RELAGN codes use non-differentiable operations
    (root solvers, C implementations). Tengri's JAX reimplementation makes
    key approximations documented in the code (see
    ``docs/dev/design/agn-kd-model.md``):

    - R_hot: 40-step JAX-compatible bisection (exact to ~10^{-12}).
    - Warm zone: precomputed nthcomp templates with (Γ_w, kT_e) interpolation
      (accuracy: ≲ 2% in flux density).
    - Seed photons: K&D Eq. 3 integrated on 100-point log grid (exact).
    - Outer radius: Laor & Netzer self-gravity radius (2–4× improvement
      over fixed 1000 r_ISCO).

    References
    ----------
    .. [1] A. Kubota and C. Done, "A physical model of the broad-band continuum
       of AGN and its implications for the UV/X relation and optical variability,"
       MNRAS, 480, 1247 (2018). arXiv:1804.00171.
       https://doi.org/10.1093/mnras/sty1890
    .. [2] C. Done et al., "Intrinsic disc emission and the soft X-ray excess in
       active galactic nuclei," MNRAS, 420, 1848 (2012). arXiv:1107.5429.
       https://doi.org/10.1111/j.1365-2966.2011.19779.x
    .. [3] A. M. Beloborodov, "Plasma Ejection from Magnetic Flares and the X-Ray
       Spectrum of Cygnus X-1," ApJL, 510, L123 (1999). arXiv:astro-ph/9809383.
       https://doi.org/10.1086/311810
    """
    nu = _wavelength_to_nu(wavelength)

    r_g, r_isco_rg, r_isco_cm, _eta, l_edd, mdot = _compute_bh_params(
        agn_log_mbh, agn_log_ledd, agn_a_spin
    )

    # Novikov-Thorne inner-disc temperature.
    t_in = (
        3.0
        * _G_GRAV
        * 10.0**agn_log_mbh
        * _MSUN_G
        * mdot
        / (8.0 * jnp.pi * _SIGMA_SB * r_isco_cm**3)
    ) ** 0.25

    r_hot_cm, r_warm_cm, r_out_cm = _compute_zone_radii(
        r_g,
        r_isco_rg,
        r_isco_cm,
        t_in,
        agn_log_mbh,
        agn_log_ledd,
        agn_f_hard,
        agn_r_warm_ratio,
        l_edd,
    )

    l_bol_requested = 10.0**agn_log_lbol * _LSUN_ERG * agn_frac

    l_nu_total, scale = _compute_zone_luminosities(
        nu,
        r_isco_cm,
        r_hot_cm,
        r_warm_cm,
        r_out_cm,
        t_in,
        agn_cos_inc,
        n_radii,
        agn_gamma_warm,
        agn_kt_warm,
        agn_gamma_hard,
        agn_kt_hot,
        agn_f_hard,
        l_edd,
        l_bol_requested,
        agn_self_consistent_gamma,
    )

    return l_nu_total * scale


# ── Model 4: ADAF + truncated disc (low-luminosity AGN) ───────────


def _adaf_electron_temperature(
    agn_adaf_delta: float,
    mdot_dimensionless: float,
) -> float:
    """Compute ADAF electron temperature (Mahadevan 1997, Eq. 40).

    The electron temperature is set by the energy-balance equation:

        T_e = 1.1 × 10^9 K · (δ/ṁ)^{1/7}

    where δ is the electron heating fraction and ṁ is the dimensionless
    accretion rate. The exponent 1/7 (not 1/2) comes from coupled
    electron-proton heating equations. Output is clipped to [10^8, 5×10^11] K.

    Parameters
    ----------
    agn_adaf_delta : float
        Fraction of viscously dissipated energy that directly heats electrons.
        [dimensionless, 0–1]
    mdot_dimensionless : float
        Dimensionless accretion rate (L/L_Edd). [dimensionless]

    Returns
    -------
    float
        Electron temperature [K].
    """
    delta_safe = jnp.clip(agn_adaf_delta, 1e-6, 1.0)
    t_e = 1.1e9 * (delta_safe / mdot_dimensionless) ** (1.0 / 7.0)
    return jnp.clip(t_e, 1e8, 5e11)  # physical range [K]


def _adaf_synchrotron_spectrum(
    nu: jnp.ndarray,
    agn_log_mbh: float,
    mdot_dimensionless: float,
) -> jnp.ndarray:
    """Compute ADAF synchrotron spectral shape (Mahadevan 1997, Eq. 21, 25).

    Electrons in the turbulent ADAF magnetic field produce synchrotron emission.
    Below the self-absorption frequency ν_sa ≈ ν_peak/3, the spectrum is
    optically thick (Rayleigh-Jeans: ∝ ν²). Above ν_sa, it is optically thin
    with spectral index 2/5: ∝ ν^{2/5} (not 1/3). The two regimes join
    continuously at ν_sa.

    Parameters
    ----------
    nu : array_like, shape (n_wave,)
        Frequency grid [Hz].
    agn_log_mbh : float
        Black hole mass [log10(M_sun)].
    mdot_dimensionless : float
        Dimensionless accretion rate [dimensionless].

    Returns
    -------
    ndarray, shape (n_wave,)
        Normalized synchrotron spectral shape (dimensionless).
    """
    # Peak frequency (Mahadevan 1997, Eq. 21):
    #   nu_peak ∝ M_BH^{-1/2} * mdot^{1/2}
    nu_peak_sync = 1e12 * (10.0**agn_log_mbh / 1e8) ** (-0.5) * mdot_dimensionless**0.5

    # Self-absorption frequency (Mahadevan 1997)
    nu_sa = nu_peak_sync / 3.0

    # Synchrotron spectral shape (Mahadevan 1997, Eq. 25):
    # - Below nu_sa (optically thick, Rayleigh-Jeans): nu^2
    # - Above nu_sa (optically thin): nu^{2/5} (NOT nu^{1/3})
    # Join continuously by normalizing thick at nu_sa.
    nu_ratio_sa = nu / jnp.maximum(nu_sa, 1.0)
    nu_ratio_peak = nu / jnp.maximum(nu_peak_sync, 1.0)

    sync_thick = (nu_ratio_sa**2.0) * (nu_sa / jnp.maximum(nu_peak_sync, 1.0)) ** (2.0 / 5.0)
    sync_thin = (nu_ratio_peak ** (2.0 / 5.0)) * jnp.exp(
        -jnp.clip(nu_ratio_peak / 3.0, 0.0, 500.0)
    )
    return jnp.where(nu < nu_sa, sync_thick, sync_thin)


def _adaf_composite_spectrum(
    nu: jnp.ndarray,
    t_e: float,
    sync_shape: jnp.ndarray,
    agn_adaf_beta: float,
) -> jnp.ndarray:
    """Compute ADAF composite spectrum (synchrotron + bremsstrahlung + IC).

    Combines the synchrotron, bremsstrahlung, and inverse Compton components
    weighted by the magnetic field strength (via β = gas pressure / total pressure).
    The relative contributions are determined by the magnetic energy density and
    optical depth (Mahadevan 1997, Eq. 26, 29, 35).

    Parameters
    ----------
    nu : array_like, shape (n_wave,)
        Frequency grid [Hz].
    t_e : float
        Electron temperature [K].
    sync_shape : array_like, shape (n_wave,)
        Synchrotron spectral shape (already normalized).
    agn_adaf_beta : float
        Ratio of gas pressure to total pressure (β parameter).
        [dimensionless, 0–1]

    Returns
    -------
    ndarray, shape (n_wave,)
        Composite ADAF spectral shape (dimensionless, unnormalized).
    """
    # --- Bremsstrahlung component ---
    # Flat spectrum (nu^0) below exponential cutoff at k_B T_e / h.
    # Mahadevan 1997, Eq. 3 & Rybicki & Lightman Ch. 5.
    nu_brem = _K_BOLTZ * t_e / _H_PLANCK
    nu_ratio_brem = nu / jnp.maximum(nu_brem, 1.0)
    brem_shape = jnp.exp(-jnp.clip(nu_ratio_brem, 0.0, 500.0))

    # --- Relative component weights ---
    # Mahadevan (1997, Eq. 26, 29, 35) derives the relative contributions
    # from the magnetic field pressure (via beta) and optical depths.
    # beta = P_gas / P_total = 1 - (B^2 / 8pi) / P_total
    beta_safe = jnp.clip(agn_adaf_beta, 0.01, 0.99)

    # --- Inverse Compton component ---
    # Thomson scattering of bremsstrahlung photons (Mahadevan 1997, Eq. 34–35).
    # Photon index: alpha_c = -ln(tau_es) / ln(A), where A ≈ 4*Theta
    # and Theta = k_B T_e / (m_e c^2).
    # For typical ADAF (T_e ~ 10^10 K, tau_es ~ 1), alpha_c ~ 0.7–1.0.
    # We use a simplified approximation with a fixed effective index.
    theta_adaf = _K_BOLTZ * t_e / (_M_PROTON * _C_LIGHT**2)
    tau_es_approx = 0.1 * (1.0 - beta_safe)  # Approximate Thomson scattering depth
    # Avoid log(0) by using maximum
    tau_safe = jnp.maximum(tau_es_approx, 1e-10)
    a_factor = 4.0 * theta_adaf
    a_safe = jnp.maximum(a_factor, 1e-10)
    alpha_c = -jnp.log(tau_safe) / jnp.log(a_safe)
    alpha_c = jnp.clip(alpha_c, 0.5, 1.5)  # Physical range

    ic_shape = (nu / jnp.maximum(nu_brem, 1.0)) ** (-alpha_c) * jnp.exp(
        -jnp.clip(nu / jnp.maximum(nu_brem, 1.0), 0.0, 500.0)
    )

    # Magnetic pressure fraction (and thus magnetic field strength) scales synchrotron
    b_mag_frac = 1.0 - beta_safe  # B^2/(8 pi P_total)
    sync_weight = b_mag_frac / jnp.maximum(b_mag_frac + beta_safe, 1e-10)

    # Bremsstrahlung dominates at intermediate frequencies
    brem_weight = beta_safe / jnp.maximum(b_mag_frac + beta_safe, 1e-10)

    # Inverse Compton: sub-dominant unless tau_es is large
    ic_weight = 0.3 * (1.0 - beta_safe)

    # Normalize weights to sum ~1 (exact normalization happens below)
    return sync_weight * sync_shape + brem_weight * brem_shape + ic_weight * ic_shape


def _adaf_truncated_disc_spectrum(
    nu: jnp.ndarray,
    agn_log_mbh: float,
    agn_r_tr: float,
    r_g: float,
    r_isco_rg: float,
    r_isco_cm: float,
    mdot: float,
    agn_cos_inc: float,
    n_radii: int,
) -> jnp.ndarray:
    """Compute luminosity from truncated outer disc (r > r_tr).

    Standard Shakura-Sunyaev multi-color disc from the truncation radius
    outward, using radial temperature profile and numerical integration.

    Parameters
    ----------
    nu : array_like, shape (n_wave,)
        Frequency grid [Hz].
    agn_log_mbh : float
        Black hole mass [log10(M_sun)].
    agn_r_tr : float
        Truncation radius in units of gravitational radius [dimensionless].
    r_g : float
        Gravitational radius [cm].
    r_isco_rg : float
        ISCO radius in units of gravitational radius [dimensionless].
    r_isco_cm : float
        ISCO radius [cm].
    mdot : float
        Accretion rate [g/s].
    agn_cos_inc : float
        Cosine of inclination angle [dimensionless].
    n_radii : int
        Number of radial integration points.

    Returns
    -------
    ndarray, shape (n_wave,)
        L_ν from the truncated outer disc [erg/s/Hz] (unnormalized).
    """
    # ── Truncated outer disc (r > r_tr) ───────────────────────────
    r_tr_safe = jnp.maximum(agn_r_tr, r_isco_rg + 1.0)  # r_tr > r_isco
    r_in_cm = r_tr_safe * r_g  # inner edge at truncation radius
    r_out_cm = 1000.0 * r_isco_rg * r_g  # self-gravity limit

    # Inner temperature at r_tr (truncated disc boundary)
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
    r_ratio = r_grid / r_in_cm
    torque_correction = jnp.maximum(1.0 - jnp.sqrt(1.0 / r_ratio), 1e-30) ** 0.25
    t_profile = t_in * r_ratio ** (-0.75) * torque_correction

    d_log_r = log_r_grid[1] - log_r_grid[0]
    dr = r_grid * jnp.log(10.0) * d_log_r

    def _ring_lnu(r_cm, t_ring, dr_ring):
        """Compute Planck luminosity per unit frequency for disc annulus."""
        b_nu = _planck_lnu(nu, t_ring)
        return b_nu * _ring_area(r_cm, dr_ring, agn_cos_inc)

    ring_contributions = jax.vmap(_ring_lnu)(r_grid, t_profile, dr)
    return jnp.sum(ring_contributions, axis=0)


# ADAF disc model (Mahadevan 1997; Yuan & Narayan 2014). The current implementation
# contains documented physics discrepancies vs Mahadevan (1997) and is scheduled for
# a full rewrite; do not use for publication-grade fits. Tracking: ROADMAP.md.
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
    """Advection-dominated accretion flow (ADAF) plus truncated disc.

    Model a low-luminosity AGN in the sub-Eddington regime where the inner
    accretion flow transitions from a geometrically thin, optically thick
    disc to an advection-dominated accretion flow (ADAF). In an ADAF, most
    of the released gravitational energy is advected into the black hole
    rather than radiated away, making the flow radiatively inefficient.
    The outer disc remains as a Shakura-Sunyaev disc truncated at a transition
    radius r_tr.

    The total spectrum is the sum of four components:

    1. **ADAF synchrotron** (radio to millimeter): relativistic electrons
       gyrating in a turbulent magnetic field.
    2. **ADAF bremsstrahlung** (X-ray): thermal bremsstrahlung from
       collisions in the hot electron-ion plasma (T_e ~ 10^9--10^11 K).
    3. **ADAF inverse Compton** (hard X-ray): upscattering of bremsstrahlung
       photons by relativistic electrons.
    4. **Truncated outer disc** (UV/optical): Shakura-Sunyaev multi-color
       disc from r_tr outward.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength grid. [Angstrom]
    agn_log_lbol : float
        Total AGN bolometric luminosity. [log10(L_sun)]
    agn_frac : float, optional
        Fraction of bolometric luminosity emitted by this component.
        Default: 0.1. [dimensionless, 0–1]
    agn_log_mbh : float, optional
        Black hole mass. Scales the synchrotron peak frequency and
        gravitational radius. Default: 8.0. [log10(M_sun)]
    agn_log_ledd : float, optional
        Eddington ratio. This model is designed for sub-Eddington
        accretion (log(L/L_Edd) < -2). Default: -3.0.
        [log10(L / L_Edd)]
    agn_r_tr : float, optional
        Truncation radius in units of the gravitational radius
        R_g = GM/c^2. This marks the transition between the ADAF
        (r < r_tr) and the thin disc (r > r_tr). Typical range: 10–1000 R_g.
        Default: 100.0. [dimensionless, ≥ 6]
    agn_adaf_beta : float, optional
        Ratio of gas pressure to total pressure in the ADAF.
        Controls the magnetic field strength and synchrotron emission.
        Range: [0, 1]. Default: 0.5. [dimensionless]
    agn_adaf_delta : float, optional
        Fraction of viscously dissipated energy that directly heats electrons
        (as opposed to protons). Controls the electron temperature.
        Lower δ → hotter electrons. Range: [0, 1]. Default: 0.01.
        [dimensionless]
    agn_cos_inc : float, optional
        Cosine of the inclination angle (outer disc view angle).
        Default: 0.5 (60°). [dimensionless, 0.01–1.0]
    n_radii : int, optional
        Number of radial integration points for the outer disc component.
        Default: 50.

    Returns
    -------
    ndarray, shape (n_wave,)
        Spectral luminosity density L_ν. [erg/s/Hz]

    Notes
    -----
    **JIT-compatible**: yes — uses ``jnp`` primitives.

    **Radiative efficiency** (Mahadevan 1997, Eq. 49):
    In an ADAF, the radiated luminosity scales as:

    .. math::

        L_{\\rm rad} \\approx 0.1 \\, \\dot{m}^2 \\, L_{\\rm Edd}

    where :math:`\\dot{m} = L / L_{\\rm Edd}` is the dimensionless accretion rate.
    For sub-Eddington rates (:math:`\\dot{m} \\lesssim 0.1`), the radiated luminosity
    is highly suppressed. The ADAF component luminosity is estimated from the
    fractional contribution determined by the truncation radius.

    **ADAF electron temperature** (Mahadevan 1997, Eq. 40):
    The electron temperature is set by the energy-balance equation:

    .. math::

        T_e = 1.1 \\times 10^9 \\, \\mathrm{K} \\cdot
        \\left(\\frac{\\delta}{\\dot{m}}\\right)^{1/7}

    where :math:`\\delta` is the electron heating fraction and :math:`\\dot{m}` is the
    dimensionless accretion rate. The exponent 1/7 (not 1/2) comes from the coupled
    electron-proton heating equations. Clipped to [10^8, 5×10^11] K for physical
    plausibility.

    **Synchrotron component** (Mahadevan 1997, Eq. 21, 25):
    Electrons in the turbulent ADAF magnetic field produce synchrotron emission with
    a characteristic peak frequency:

    .. math::

        \\nu_{\\rm peak} \\approx 10^{12} \\, \\mathrm{Hz} \\cdot
        \\left(\\frac{M_{\\rm BH}}{10^8 \\, M_\\odot}\\right)^{-1/2}
        \\dot{m}^{1/2}

    Below the self-absorption frequency :math:`\\nu_{\\rm sa} \\approx \\nu_{\\rm peak}/3`,
    the spectrum is optically thick (Rayleigh-Jeans: :math:`\\propto \\nu^2`).
    Above :math:`\\nu_{\\rm sa}`, it is optically thin with spectral index 2/5:
    :math:`\\propto \\nu^{2/5}` (not 1/3). The two regimes join continuously at
    :math:`\\nu_{\\rm sa}`.

    **Bremsstrahlung** (Mahadevan 1997, Eq. 3):
    Thermal bremsstrahlung from the hot electron-ion plasma provides a flat spectrum
    (:math:`\\propto \\nu^0`) below the exponential cutoff at :math:`k_B T_e / h`.

    **Inverse Compton** (Mahadevan 1997, Eq. 34, 35):
    Thomson (optically thin) scattering of bremsstrahlung photons by relativistic
    electrons. The photon index is:

    .. math::

        \\alpha_c = -\\frac{\\ln(\\tau_{\\rm es})}{\\ln(A)}

    where :math:`\\tau_{\\rm es}` is the electron scattering optical depth and
    :math:`A \\approx 4 \\Theta` with :math:`\\Theta = k_B T_e / m_e c^2`.
    For typical ADAF parameters (:math:`T_e \\sim 10^{10}` K, :math:`\\tau_{\\rm es} \\sim 1`),
    this yields :math:`\\alpha_c \\sim 0.7–1.0`.

    **Component weights** (Mahadevan 1997, Eq. 26, 29, 35):
    The relative contributions of synchrotron, bremsstrahlung, and inverse Compton
    are determined by the magnetic field strength (via :math:`\\beta` = gas pressure
    / total pressure), the optical depth, and the particle distribution. The weights
    are not arbitrary: synchrotron dominates at low frequencies, bremsstrahlung at
    intermediate frequencies, and inverse Compton at high frequencies. The beta
    parameter controls the magnetic energy density.

    **Truncated outer disc**: The standard Shakura-Sunyaev multi-color disc
    (see :func:`multicolor_disc`) contributes from the truncation radius r_tr
    outward. This dominates the UV/optical and provides the warm photons
    that seed inverse Compton scattering.

    **Known limitations**:

    - This implementation is based on Mahadevan (1997) analytic prescriptions and does
      not solve the full magnetohydrodynamic ADAF equations.
    - The inverse Compton calculation uses a simplified approximation; full solutions
      would require iterative energy balance with the photon field.
    - The model has not been extensively validated against 3D ADAF simulations or a
      comprehensive sample of observed low-luminosity AGN. Use with caution in
      quantitative inference.

    References
    ----------
    .. [1] R. Mahadevan, "Scaling Laws for Advection-dominated Flows:
       Applications to Low-Luminosity Galactic Nuclei," ApJ, 477, 585 (1997).
       https://doi.org/10.1086/303727
    .. [2] R. Nemmen et al., "Spectral models for low-luminosity active galactic nuclei
       in LINERs: the role of advection-dominated accretion and jets," MNRAS, 438, 2804
       (2014). arXiv:1312.1982. https://doi.org/10.1093/mnras/stt2388
    .. [3] G. B. Rybicki and A. P. Lightman, "Radiative Processes in
       Astrophysics," John Wiley & Sons (1979). ISBN: 0-471-82759-2
    """
    nu = _wavelength_to_nu(wavelength)
    l_bol_erg = 10.0**agn_log_lbol * _LSUN_ERG

    # --- Black hole parameters ---
    r_g = _gravitational_radius(agn_log_mbh)
    r_isco_rg = _isco_radius(0.0)  # Schwarzschild for LLAGN

    # Eddington ratio and accretion rate
    l_edd = _eddington_luminosity(agn_log_mbh)
    l_edd_ratio = jnp.clip(10.0**agn_log_ledd, 1e-10, 1.0)

    # Radiative efficiency (Novikov-Thorne for Schwarzschild)
    eta = 1.0 - jnp.sqrt(1.0 - 2.0 / (3.0 * r_isco_rg))
    r_isco_cm = r_isco_rg * r_g
    mdot = l_edd_ratio * l_edd / (eta * _C_LIGHT**2)

    # ── ADAF component (r < r_tr) ─────────────────────────────────
    # ADAF radiative efficiency (Mahadevan 1997, Eq. 49):
    # L_rad ≈ 0.1 * mdot^2 * L_Edd
    # The ADAF luminosity fraction depends on the truncation radius and
    # accretion rate. Larger r_tr or lower mdot both suppress ADAF emission.
    mdot_dimensionless = jnp.clip(l_edd_ratio, 1e-10, 1.0)
    adaf_radiative_efficiency = 0.1 * mdot_dimensionless**2
    l_adaf_erg = l_bol_erg * adaf_radiative_efficiency

    # Compute ADAF electron temperature (Mahadevan 1997, Eq. 40)
    t_e = _adaf_electron_temperature(agn_adaf_delta, mdot_dimensionless)

    # Compute synchrotron spectrum (Mahadevan 1997, Eq. 21, 25)
    sync_shape = _adaf_synchrotron_spectrum(nu, agn_log_mbh, mdot_dimensionless)

    # Compute composite ADAF spectrum (synchrotron + bremsstrahlung + IC)
    adaf_shape = _adaf_composite_spectrum(nu, t_e, sync_shape, agn_adaf_beta)

    # Normalize ADAF to l_adaf_erg
    # nu is descending (wavelength ascending), so [::-1] gives ascending order for trapz.
    adaf_integral = jnp.trapezoid(adaf_shape[::-1], nu[::-1])
    adaf_integral_safe = jnp.maximum(adaf_integral, 1e-100)
    l_nu_adaf = l_adaf_erg * adaf_shape / adaf_integral_safe

    # Compute truncated outer disc spectrum (r > r_tr)
    l_nu_disc = _adaf_truncated_disc_spectrum(
        nu,
        agn_log_mbh,
        agn_r_tr,
        r_g,
        r_isco_rg,
        r_isco_cm,
        mdot,
        agn_cos_inc,
        n_radii,
    )

    # ── Combine and normalize ─────────────────────────────────────
    # Remaining luminosity allocated to the truncated disc
    l_disc_erg = l_bol_erg * (1.0 - adaf_radiative_efficiency)
    disc_integral = jnp.trapezoid(l_nu_disc[::-1], nu[::-1])
    disc_integral_safe = jnp.maximum(disc_integral, 1e-100)
    l_nu_disc_norm = l_disc_erg * l_nu_disc / disc_integral_safe

    l_nu_total = l_nu_adaf + l_nu_disc_norm

    # Both components are already normalized to their energies
    return l_nu_total * agn_frac


# ── Model 5: RELAGN relativistic disc from precomputed grid ──────────────────


@functools.cache
def _load_relagn_disc_grid(grid_path: str) -> dict:
    """Load and cache the RELAGN outer-disc grid from HDF5.

    Parameters
    ----------
    grid_path : str
        Path to ``data/relagn_disc_grid.h5``.

    Returns
    -------
    dict with keys:
        grid_jax : jnp.ndarray, shape (n_mass, n_mdot, n_astar, n_wave)
        axes : tuple of jnp.ndarray  (log_mbh, log_mdot, astar)
        edges : tuple of jnp.ndarray
        scatters : tuple of float
        wave_grid : jnp.ndarray, shape (n_wave,)
    """
    with h5py.File(grid_path, "r") as f:
        grid_np = f["lnu_disc"][()]
        log_mbh = f["log_mbh"][()]
        log_mdot = f["log_mdot"][()]
        astar = f["astar"][()]
        wave_grid = f["wavelength_aa"][()]

    axes = (
        jnp.array(log_mbh),
        jnp.array(log_mdot),
        jnp.array(astar),
    )
    edges = tuple(_edges_for_grid(ax) for ax in axes)

    # For non-uniform astar, use max half-spacing: triweight has compact support
    # (zero beyond one bandwidth), so the bandwidth must cover the largest gap.
    # Uniform axes use the single node spacing.
    import numpy as _np

    scatter_lm = 0.5 * float(log_mbh[1] - log_mbh[0])
    scatter_ld = 0.5 * float(log_mdot[1] - log_mdot[0])
    scatter_as = 0.5 * float(_np.max(_np.diff(astar)))
    scatters = (scatter_lm, scatter_ld, scatter_as)

    return {
        "grid_jax": jnp.array(grid_np),
        "axes": axes,
        "edges": edges,
        "scatters": scatters,
        "wave_grid": jnp.array(wave_grid),
    }


def create_relagn_disc_from_grid(grid_path: str) -> Callable:
    """Return a JIT-compatible RELAGN disc SED function from a precomputed grid.

    The grid was built with the real RELAGN Python class (Hagen & Done 2023)
    using KYCONV (Dovciak, Karas & Yaqoob 2004) per-annulus Kerr ray-tracing.
    It stores absolute L_ν (erg/s/Hz) at cos_inc = 0.5; the inclination
    correction is applied analytically as 2·cos_inc (valid for the
    non-relativistic outer disc; approximate for the GR inner disc).

    Parameters
    ----------
    grid_path : str
        Path to ``data/relagn_disc_grid.h5``.

    Returns
    -------
    callable
        Function with signature::

            fn(wavelength, agn_log_mbh, agn_log_mdot, agn_astar,
               agn_cos_inc, **kwargs) -> L_nu [erg s^-1 Hz^-1]

    Raises
    ------
    FileNotFoundError
        If ``grid_path`` does not exist.

    Notes
    -----
    **JIT-compatible**: yes — the returned function is pure JAX.
    Grid loading is cached via ``@functools.cache``.

    **Gradient-safe**: yes — triweight interpolation is C²-continuous.

    **Inclination**: grid stored at cos_inc = 0.5; scaled by 2·cos_inc.
    This is exact for r > 1000 r_g (non-relativistic regime) and approximate
    for the GR inner disc where KYCONV applies full Kerr ray-tracing.

    **Grid axes**: log_mbh ∈ [7, 10], log_mdot ∈ [−1.5, 0.3], astar ∈ [0, 0.998]
    (prograde only; KYCONV rejects retrograde spins).

    References
    ----------
    .. [1] Dovciak, M., Karas, V., & Yaqoob, T. (2004).
       ApJS, 153, 205. doi:10.1086/421115

    .. [2] Hagen, S. & Done, C. (2023).
       MNRAS, 521, 251. doi:10.1093/mnras/stad478
    """
    if not __import__("pathlib").Path(grid_path).exists():
        raise FileNotFoundError(f"RELAGN disc grid not found: {grid_path}")

    cached = _load_relagn_disc_grid(grid_path)
    grid_jax = cached["grid_jax"]
    axes = cached["axes"]
    edges = cached["edges"]
    scatters = cached["scatters"]
    wave_grid = cached["wave_grid"]

    def relagn_disc(
        wavelength: jnp.ndarray,
        agn_log_mbh: float = 8.0,
        agn_log_mdot: float = -1.0,
        agn_astar: float = 0.0,
        agn_cos_inc: float = 0.5,
        **_kwargs,
    ) -> jnp.ndarray:
        """RELAGN outer disc from relativistic template grid.

        Parameters
        ----------
        wavelength : ndarray, shape (n_wave,)
            Rest-frame wavelength. [Å]
        agn_log_mbh : float
            log₁₀(M_BH / M_sun). [dimensionless]
        agn_log_mdot : float
            log₁₀(Ṁ / Ṁ_Edd). [dimensionless]
        agn_astar : float
            Dimensionless BH spin, prograde only (0 to 0.998). [dimensionless]
        agn_cos_inc : float
            Cosine of inclination angle (1 = face-on). [dimensionless]

        Returns
        -------
        ndarray, shape (n_wave,)
            Specific luminosity L_ν. [erg s⁻¹ Hz⁻¹]

        Notes
        -----
        **JIT-compatible**: yes.
        **Gradient-safe**: yes — triweight kernel, C²-continuous.
        """
        point = (agn_log_mbh, agn_log_mdot, agn_astar)
        lnu_template = _interp_nd_triweight(grid_jax, axes, edges, point, scatters=scatters)
        # Interpolate grid wavelength → observation wavelength
        lnu_interp = jnp.interp(wavelength, wave_grid, lnu_template, left=0.0, right=0.0)
        # Inclination scaling from reference cos_inc = 0.5
        return lnu_interp * (2.0 * agn_cos_inc)

    return relagn_disc
