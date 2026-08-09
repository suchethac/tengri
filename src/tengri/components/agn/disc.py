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
- Nemmen et al. 2014, MNRAS, 438, 2804 (ADAF modeling)
- Lopez et al. 2024 (ADAF + truncated disc for LLAGN)
- Beloborodov 1999, ApJ, 510, L123 (self-consistent Gamma_hot)

"""

import functools
import math
from collections.abc import Callable

import h5py
import jax
import jax.numpy as jnp

from tengri.components.agn._nthcomp import (
    _TABLE_AVAILABLE as _NTHCOMP_AVAILABLE,
    nthcomp_lnu_interp as _nthcomp_lnu_interp,
)
from tengri.components.agn._params import (
    DEFAULT_AGN_COS_INC,
    DEFAULT_AGN_LOG_LBOL,
    DEFAULT_AGN_LOG_MBH,
    DEFAULT_AGN_LUM_RATIO,
)
from tengri.components.agn._phys import (
    C_LIGHT as _C_LIGHT,
    H_PLANCK as _H_PLANCK,
    K_BOLTZ as _K_BOLTZ,
    bolometric_integral_nu as _bolometric_integral_nu,
    planck_lnu as _planck_lnu,
    ring_area as _ring_area,
    wavelength_to_nu as _wavelength_to_nu,
)
from tengri.utils.grid_interp import interp_nd_triweight as _interp_nd_triweight, resample_template
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
from tengri.utils.scale import pow10 as _pow10, representable_floor as _representable_floor

# log10 of the cgs constants that make the Shakura-Sunyaev disc's bolometric /
# Eddington / accretion-rate intermediates overflow float32 (#1206). At a
# realistic AGN luminosity the LINEAR forms — L_bol ~1e44, L_Edd ~1e46, the
# ``t_in**4`` numerator ~1e58 erg/s — all exceed float32 max (3.4e38), yet the
# RESULTS (mdot ~1e24 g/s, t_in ~1e5 K, lambda_Edd ~1e-2) are representable.
# The float32 branch of ``multicolor_disc`` forms every such quantity as a log10
# sum and materializes it only at the representable result via ``pow10``.
_LOG10_LSUN_ERG: float = math.log10(_LSUN_ERG)
_LOG10_C_LIGHT: float = math.log10(_C_LIGHT)
_LOG10_MSUN_G: float = math.log10(_MSUN_G)
_LOG10_3G: float = math.log10(3.0 * _G_GRAV)
_LOG10_8PI_SIGMA_SB: float = math.log10(8.0 * math.pi * _SIGMA_SB)
# log10(L_Edd) at M_BH = 1 M_sun: L_Edd = 4*pi*G*M*m_p*c/sigma_T, linear in M_BH.
_LOG10_L_EDD_1MSUN: float = math.log10(
    4.0 * math.pi * _G_GRAV * _MSUN_G * _M_PROTON * _C_LIGHT / _SIGMA_T
)
# Constants for the Kubota & Done float32 path (#1206): the three-zone model
# carries absolute cgs luminosities (l_edd ~1e46, l0 ~1e42, energy integrals
# ~1e44) that overflow float32. The float32 branch works those in L_sun units
# (÷ L_sun) via pre-divided constants so no ~1e44 intermediate ever forms.
# L_Edd(1 M_sun) in L_sun; l_edd_lsun = _L_EDD_1MSUN_LSUN * 10**log_mbh.
_L_EDD_1MSUN_LSUN: float = float(
    4.0 * math.pi * _G_GRAV * _MSUN_G * _M_PROTON * _C_LIGHT / _SIGMA_T
) / float(_LSUN_ERG)
# 4*pi*sigma_SB / L_sun; l0_lsun = _4PI_SIGMA_SB_OVER_LSUN * r_isco_cm**2 * t_in**4.
_4PI_SIGMA_SB_OVER_LSUN: float = float(4.0 * math.pi * _SIGMA_SB) / float(_LSUN_ERG)
# 2*pi*sigma_SB / L_sun; per-annulus (sigma T^4 * 2*pi*r*dr) energy in L_sun.
_2PI_SIGMA_SB_OVER_LSUN: float = float(2.0 * math.pi * _SIGMA_SB) / float(_LSUN_ERG)
# L_sun / c^2; mdot [g/s] = _LSUN_OVER_C2 * 10**log_lbol / eta (avoids l_bol_erg).
_LSUN_OVER_C2: float = float(_LSUN_ERG) / float(_C_LIGHT) ** 2

# ── Model 1: Simple power-law disc + UV cutoff ────────────────────


def powerlaw_disc(
    wavelength: jnp.ndarray,
    agn_log_lbol: float,
    agn_lum_ratio: float = DEFAULT_AGN_LUM_RATIO,
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
    agn_lum_ratio : float, optional
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

    l_nu_erg = l_bol_erg * agn_lum_ratio * shape / integral_safe
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
    float32: bool = False,
) -> float:
    r"""Solve for R_hot from K&D 2018 Eq. 2 by bisection in log(x_hot).

    The NT emissivity integral has closed form
    :math:`L_{\rm diss}(x) = L_0\,[1/10 - 1/(2x^2) + 2/(5 x^{5/2})]`,
    strictly monotone in :math:`x = R_{\rm hot}/R_{\rm ISCO}`. After
    ``n_iter=40`` the bracket width is :math:`< 2^{-40} \approx 10^{-12}`
    of its initial log-width — enough for machine precision.

    ``l_hot_target`` is clipped below :math:`L_{\max} = L_0/10`.
    """
    # Maximum possible L_diss (entire disc, x→∞): h→0.1, so L_max = L0 * 0.1.
    # Float32 (#1206): ``l0`` is ~1e42 erg/s (overflow); the bisection needs only
    # the RATIO l_hot_target / l0, so compute both in L_sun units (``l_hot_target``
    # arrives in L_sun on the float32 path). The pre-divided 4*pi*sigma/L_sun
    # constant folds first so no ~1e42 intermediate forms.
    if float32:
        l0 = _4PI_SIGMA_SB_OVER_LSUN * r_isco_cm**2 * t_in**4
    else:
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
    float32: bool = False,
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
    # Float32 (#1206): the seed integral (integrand * dr) reaches ~1e43 erg/s and
    # overflows; return L_seed in L_sun units by folding 1/L_sun into the surface
    # flux. Downstream ratios (Beloborodov) are unit-invariant.
    if float32:
        f_nt = (_SIGMA_SB / _LSUN_ERG) * t_r**4  # [L_sun s^-1... i.e. erg/s/cm^2 / L_sun]
    else:
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

               * (M_BH / 10^9 M_sun)^{-2/9}   [R_g]

    where lambda_Edd = L_bol / L_Edd is the Eddington ratio and
    alpha is the Shakura-Sunyaev viscosity parameter (default 0.1).

    The mass normalization is 10^9 M_sun, matching the canonical qsosed
    implementation (Quera-Bofarull, ``Sed.gravity_radius``):
    ``r_sg = 2150 * mass^{-2/9} * mdot^{4/9} * alpha^{2/9}`` with
    ``mass = M_BH / 10^9 M_sun``. A prior version normalized by 10^8 M_sun,
    which made r_sg a factor 10^{2/9} ~ 1.67 too small at every mass and
    truncated the coolest outer annuli (deficient near-IR disc tail vs the
    AGNfitter-rX KD18 reference).

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
    m9 = 10.0**log_mbh / 1.0e9  # M_BH / 10^9 M_sun (qsosed convention)
    m9_safe = jnp.maximum(m9, 1e-6)
    lambda_safe = jnp.clip(l_edd_ratio, 1e-10, 1.0)
    alpha_safe = jnp.maximum(alpha_visc, 1e-4)
    return (
        2150.0
        * (alpha_safe / 0.1) ** (2.0 / 9.0)
        * lambda_safe ** (4.0 / 9.0)
        * m9_safe ** (-2.0 / 9.0)
    )


# ── EUV / soft-X-ray power-law tail for the thin disc ─────────────────
# A bare Shakura-Sunyaev disc Wien-cuts off in the EUV (< ~150 A), so its
# emission below 100 A is negligible. Empirical AGN disc templates (e.g.
# CIGALE's SKIRTOR piecewise power law) instead carry a rising power-law
# tail into the EUV / soft X-ray. ``multicolor_disc`` can optionally blend
# such a tail onto the Wien core (the ``euv_tail`` argument).
_EUV_TAIL_LAMBDA_BREAK_AA = 912.0  # [A] Lyman limit — onset of the EUV tail
_EUV_TAIL_LAMBDA_CUT_AA = 30.0  # [A] short-wavelength floor (~0.41 keV)
_EUV_TAIL_DEFAULT_SLOPE = 1.0  # L_nu ~ nu^slope (CIGALE-skirtor-like rise)
_EUV_TAIL_FRAC = 0.02  # tail bolometric budget as a fraction of L_disc


def _apply_euv_tail(wavelength, nu, l_nu_wien, euv_tail):
    r"""Blend an EUV / soft-X-ray power-law tail onto a Wien-cutoff thin disc.

    Parameters
    ----------
    wavelength : ndarray, shape (n_wave,)
        Rest-frame wavelength grid. [Angstrom]
    nu : ndarray, shape (n_wave,)
        Matching frequency grid. [Hz]
    l_nu_wien : ndarray, shape (n_wave,)
        The Wien-limited multi-color blackbody disc spectrum. [erg/s/Hz]
    euv_tail : None or str or float
        EUV behavior over :math:`\lambda \in [30, 912]` A:

        * ``None`` / ``"wien"`` — no tail; pure Wien cutoff (bare thin disc).
        * ``"powerlaw"`` / ``"both"`` — blend a power-law tail at the default
          slope ``_EUV_TAIL_DEFAULT_SLOPE``. ``"both"`` is a synonym; the Wien
          core is always preserved (the tail only fills where it exceeds Wien).
        * float — user-defined slope :math:`s` with :math:`L_\nu \propto \nu^s`.

    Returns
    -------
    ndarray, shape (n_wave,)
        Disc spectrum with the EUV tail blended in (pre-normalization). [erg/s/Hz]

    Notes
    -----
    **JIT-compatible**: yes — ``euv_tail`` is resolved to a static slope and a
    static on/off flag at trace time (it is not a traced array), so the
    branch is a Python-level decision and the math is pure ``jnp``.

    The tail has the **shape** :math:`L_\nu \propto \nu^s` over
    :math:`\lambda \in [30, 912]` A but its **amplitude** is fixed by
    normalizing its bolometric content to a small fraction
    (``_EUV_TAIL_FRAC``, default 2 %) of the disc's pre-tail bolometric
    luminosity — a bounded "soft-excess"-like budget. (A bare power law
    anchored to the disc peak would diverge in energy and swamp the optical.)
    It is blended via ``maximum`` so the UV/optical/Wien-peak region is
    untouched: the tail only contributes where the Wien spectrum has already
    fallen below it. The caller renormalizes the blended spectrum back to
    :math:`L_{\rm bol}`, so total energy is conserved and the optical is
    reduced only by the ~2 % moved into the EUV.
    """
    if euv_tail is None or euv_tail == "wien":
        return l_nu_wien
    if euv_tail in ("powerlaw", "both"):
        slope = _EUV_TAIL_DEFAULT_SLOPE
    else:
        slope = float(euv_tail)

    nu_break = _wavelength_to_nu(jnp.asarray(_EUV_TAIL_LAMBDA_BREAK_AA))
    in_euv = (wavelength <= _EUV_TAIL_LAMBDA_BREAK_AA) & (wavelength >= _EUV_TAIL_LAMBDA_CUT_AA)
    # Power-law SHAPE on the EUV band only (zero elsewhere).
    shape = jnp.where(in_euv, (nu / nu_break) ** slope, 0.0)

    # Normalize the tail to a fixed fraction of the disc bolometric so the
    # EUV budget is bounded regardless of slope. Integrate over frequency
    # (sort ascending; nu descends as wavelength ascends).
    sort_idx = jnp.argsort(nu)
    shape_bol = jnp.maximum(jnp.abs(jnp.trapezoid(shape[sort_idx], nu[sort_idx])), 1e-100)
    if l_nu_wien.dtype == jnp.float32:
        # Float32 (#1206): the disc bolometric integral ``trapz(l_nu_wien, nu)`` ~
        # 1e28 erg/s/Hz over ~1e15 Hz is ~1e43 erg/s — past float32 max (3.4e38).
        # Peak-factor it (integrate the O(1) residual, carry the peak) and group
        # the small factors (``frac · disc_bol/shape_bol`` ~ few·1e-4) before the
        # ~1e28 peak so no out-of-range product materializes.
        # stop_gradient: factorization constant, multiplied back below (#1436).
        _peak = jax.lax.stop_gradient(jnp.max(jnp.abs(l_nu_wien)))
        _peak = jnp.where(_peak > 0.0, _peak, 1.0)
        _disc_bol_hat = jnp.abs(jnp.trapezoid(l_nu_wien[sort_idx] / _peak, nu[sort_idx]))
        tail = shape * ((_EUV_TAIL_FRAC * _disc_bol_hat / shape_bol) * _peak)
    else:
        disc_bol = jnp.abs(jnp.trapezoid(l_nu_wien[sort_idx], nu[sort_idx]))
        tail = shape * (_EUV_TAIL_FRAC * disc_bol / shape_bol)
    return jnp.maximum(l_nu_wien, tail)


def multicolor_disc(
    wavelength: jnp.ndarray,
    agn_log_lbol: float,
    agn_lum_ratio: float = DEFAULT_AGN_LUM_RATIO,
    agn_log_mbh: float = DEFAULT_AGN_LOG_MBH,
    agn_log_ledd: float = -1.0,
    agn_a_spin: float = 0.0,
    agn_cos_inc: float = DEFAULT_AGN_COS_INC,
    n_radii: int = 50,
    euv_tail: str | float | None = "powerlaw",
    agn_log_lbol_shape: float | None = None,
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
    agn_lum_ratio : float, optional
        Fraction of bolometric luminosity emitted by the disc.
        Default: 1.0. [dimensionless, 0–1]
    agn_log_mbh : float, optional
        Black hole mass. Default: 8.0. [log10(M_sun)]
    agn_log_ledd : float, optional
        **DEPRECATED / IGNORED (#846).** The Eddington ratio is now DERIVED from
        ``agn_log_lbol`` and ``agn_log_mbh`` (lambda_Edd = L_bol / L_Edd), so the
        disc shape is self-consistent with the requested L_bol. This parameter
        is retained for backward compatibility but has no effect; setting or
        freeing it emits a build-time warning. Default: -1.0.
    agn_a_spin : float, optional
        Dimensionless black hole spin parameter (prograde).
        Range: [0, 0.998]. Default: 0.0 (Schwarzschild). [dimensionless]
    agn_cos_inc : float, optional
        Cosine of the inclination angle. Range: [0.01, 1.0].
        Default: 0.5 (60°). [dimensionless]
    n_radii : int, optional
        Number of radial bins for numerical integration. Default: 50.
    euv_tail : {"powerlaw", "both", "wien"}, float, or None, optional
        EUV / soft-X-ray behavior below the Lyman limit (912 A).

        * ``"powerlaw"`` (default) / ``"both"`` — blend a CIGALE-like power-law
          tail onto the Wien core so the disc carries flux below ~100 A.
        * ``"wien"`` / ``None`` — pure Shakura-Sunyaev Wien cutoff (the bare
          thin disc; emission below ~150 A is negligible and the EUV / soft
          X-ray is supplied by the corona instead).
        * float — user-defined slope :math:`s` with :math:`L_\\nu \\propto \\nu^s`.

        Default: ``"powerlaw"``. The tail only fills the EUV where the Wien
        spectrum has already dropped below it, so the UV/optical is unchanged
        to sub-percent after the bolometric renormalization. See
        :func:`_apply_euv_tail`.

    Returns
    -------
    ndarray, shape (n_wave,)
        Spectral luminosity density L_ν. [erg/s/Hz]

    Notes
    -----
    **JIT-compatible**: yes — uses ``jnp`` primitives and ``jax.vmap``.
    ``euv_tail`` is a static (trace-time) selector, not a traced argument.

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
    # E fix (#846): agn_log_lbol is THE luminosity knob; the Eddington ratio is
    # DERIVED from it (lambda_Edd = L_bol / L_Edd, matching RELAGN's
    # mdot = L_bol/L_edd, relagn.py:288,330), so the disc shape (T_in, r_out) is
    # self-consistent with the requested L_bol. Previously the shape was built
    # from agn_log_ledd and then rescaled to agn_log_lbol — a decoupling that
    # left T_in / r_out corresponding to the wrong luminosity. agn_log_ledd is
    # now ignored here (a build-time warning fires if a user sets/frees it).
    #
    # Shape vs normalization luminosity (#1206). The disc SHAPE (T_in, r_out,
    # lambda_Edd) is set by ``_log_lbol_shape``; the output MAGNITUDE by
    # ``agn_log_lbol`` (the renorm target below). They coincide by default
    # (``agn_log_lbol_shape=None``), which is the float64 path. Only float32
    # separates them — the AGN component evaluates the SHAPE at the true L_bol
    # but normalizes MAGNITUDE to a low reference so the runner's ~1e40 L_lambda
    # arithmetic stays in float32 range; the true magnitude is re-applied
    # downstream. See the float32 branch below.
    _log_lbol_shape = agn_log_lbol if agn_log_lbol_shape is None else agn_log_lbol_shape

    # Outer disc radius: Laor & Netzer (1989) self-gravity (Toomre) radius.
    # Beyond r_sg the disc fragments rather than accretes; this is the
    # physically motivated outer boundary used by qsosed (Quera-Bofarull).
    # r_sg ~ 2150 * (alpha/0.1)^{2/9} * lambda_Edd^{4/9} * (M/1e8)^{-2/9} R_g.
    if wavelength.dtype == jnp.float32:
        # Log-space so the ~1e44 L_bol, ~1e46 L_Edd and ~1e58 erg/s ``t_in**4``
        # numerator never materialize (float32 max 3.4e38). The RESULTS —
        # lambda_Edd ~1e-2, mdot ~1e24 g/s, t_in ~1e5 K — are all representable.
        _log_l_bol_erg = _log_lbol_shape + _LOG10_LSUN_ERG
        _log_l_edd = _LOG10_L_EDD_1MSUN + agn_log_mbh
        l_edd_ratio = jnp.clip(_pow10(_log_l_bol_erg - _log_l_edd), 1e-10, 1.0)
        _log_mdot = _log_l_bol_erg - jnp.log10(eta) - 2.0 * _LOG10_C_LIGHT
        mdot = _pow10(_log_mdot)  # [g s^-1]
        _log_t_in4 = (
            _LOG10_3G
            + agn_log_mbh
            + _LOG10_MSUN_G
            + _log_mdot
            - _LOG10_8PI_SIGMA_SB
            - 3.0 * jnp.log10(r_in)
        )
        t_in = _pow10(0.25 * _log_t_in4)  # [K]
    else:
        l_bol_erg = 10.0**_log_lbol_shape * _LSUN_ERG
        l_edd = _eddington_luminosity(agn_log_mbh)
        l_edd_ratio = jnp.clip(l_bol_erg / l_edd, 1e-10, 1.0)  # derived: lambda_Edd
        mdot = l_bol_erg / (eta * _C_LIGHT**2)  # [g s^-1]
        # Inner temperature: T_in = (3 * G * M * Mdot / (8*pi*sigma_SB * r_in^3))^(1/4)
        t_in = (
            3.0
            * _G_GRAV
            * 10.0**agn_log_mbh
            * _MSUN_G
            * mdot
            / (8.0 * jnp.pi * _SIGMA_SB * r_in**3)
        ) ** 0.25

    r_sg_rg = _self_gravity_radius(agn_log_mbh, l_edd_ratio)
    r_out = jnp.maximum(r_sg_rg, r_isco * 10.0) * r_g  # at least 10 r_isco

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

    # Optionally blend an EUV / soft-X-ray power-law tail onto the Wien core
    # before renormalizing, so the tail's energy is taken out of L_bol rather
    # than added on top (energy-conserving). Default "powerlaw" gives the disc
    # a CIGALE-like rise below ~100 A; "wien" recovers the bare thin disc.
    l_nu_intrinsic = _apply_euv_tail(wavelength, nu, l_nu_intrinsic, euv_tail)

    # Renormalize to requested L_bol * agn_lum_ratio (the MAGNITUDE is set by
    # ``agn_log_lbol`` — the reference on the float32 path — NOT the shape
    # luminosity above).
    # Sort by ascending frequency before integrating (nu descends when wave ascends).
    # Using jnp.abs() on a descending-x trapezoid is brittle — sort explicitly.
    _nu = _wavelength_to_nu(wavelength)
    _sort_idx = jnp.argsort(_nu)
    if wavelength.dtype == jnp.float32:
        # Log-space renorm: ``l_bol_requested`` ~1e44 and the integral
        # ``l_nu_total`` (l_nu_intrinsic ~1e28 over ~1e15 Hz → ~1e43) both
        # overflow float32; only their ratio (``scale`` ~1e-33 when normalizing
        # a true-shape disc to a 1e10 erg/s reference) is needed. Peak-factor the
        # integrand so the trapezoid stays in range, and carry the peak in log10.
        _log_l_bol_req = agn_log_lbol + _LOG10_LSUN_ERG + jnp.log10(agn_lum_ratio)
        # stop_gradient: factorization constant, added back as log10(_peak) (#1436).
        _peak = jax.lax.stop_gradient(jnp.max(jnp.abs(l_nu_intrinsic)))
        _peak = jnp.where(_peak > 0.0, _peak, 1.0)
        _hat_total = jnp.trapezoid(l_nu_intrinsic[_sort_idx] / _peak, _nu[_sort_idx])
        # ``representable_floor``, not the bare ``1e-100`` (#1492): float32's
        # smallest subnormal is 1.4e-45, so the literal IS 0.0 there and this
        # branch — the float32 one — was the guard providing nothing. A zero
        # integral would take log10 to -inf and the scale to inf. Returns
        # ``1e-100`` unchanged under x64, so float64 is bit-identical.
        _log_l_nu_total = jnp.log10(_peak) + jnp.log10(
            jnp.maximum(jnp.abs(_hat_total), _representable_floor(1e-100))
        )
        scale = _pow10(_log_l_bol_req - _log_l_nu_total)
    else:
        l_bol_requested = 10.0**agn_log_lbol * _LSUN_ERG * agn_lum_ratio
        l_nu_total = jnp.trapezoid(l_nu_intrinsic[_sort_idx], _nu[_sort_idx])
        l_nu_total_safe = jnp.maximum(jnp.abs(l_nu_total), _representable_floor(1e-100))
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
    r"""Hot corona emission: thermal-Comptonization power law with two cutoffs.

    The optically thin, hot corona produces hard X-ray emission via thermal
    Comptonization of seed photons. The spectrum is a power law bounded by a
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

    A thermal-Comptonization spectrum is only defined *between* its seed-photon
    energy and the electron temperature; there are no Comptonized photons below
    the seed energy (Kubota & Done 2018 [1]_, Section 2.2).

    Normalized so that the frequency-integrated luminosity equals
    ``l_hot_erg``. The normalization integral is computed on a fixed internal
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
    .. [2] RELAGN (scotthgn/RELAGN) ``do_nonrelHotCompSpec``: normalizes on a
       fixed [1e-4, 1e4] keV grid.
    """
    kt_safe = jnp.maximum(kt_hot_erg, 1e-30)

    def _comp_shape(freq):
        """Band-limited Comptonization shape: seed rollover x power law x cutoff.

        Computed once and reused for both the output grid and the normalization
        grid so the two can never drift apart.
        """
        freq_safe = jnp.maximum(freq, 1e-30)
        x_hi = jnp.clip(_H_PLANCK * freq / kt_safe, 0.0, 500.0)  # electron-temperature cutoff
        x_lo = jnp.clip(nu_seed_hz / freq_safe, 0.0, 700.0)  # seed-photon rollover
        return freq ** (1.0 - gamma_hard) * jnp.exp(-x_hi) * jnp.exp(-x_lo)

    shape = _comp_shape(nu)
    # Normalize on the fixed internal grid (grid-independent).
    shape_norm = _comp_shape(_CORONA_NU_GRID)
    integral = jnp.trapezoid(shape_norm, _CORONA_NU_GRID)
    integral_safe = jnp.maximum(jnp.abs(integral), 1e-100)

    return l_hot_erg * shape / integral_safe


def _compute_bh_params(
    agn_log_mbh: float,
    agn_log_lbol: float,
    agn_a_spin: float,
    float32: bool = False,
) -> tuple:
    """Compute black hole parameters: radius, ISCO, efficiency, and accretion rate.

    Derives the fundamental mass-dependent quantities: gravitational radius,
    ISCO radius, radiative efficiency (Novikov-Thorne), Eddington luminosity,
    and mass accretion rate from the black hole mass and spin parameter.

    Parameters
    ----------
    agn_log_mbh : float
        Black hole mass. [log10(M_sun)]
    agn_log_lbol : float
        Bolometric luminosity. [log10(L_bol / L_sun)]
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
    # E fix (#846): derive the accretion rate from the requested L_bol
    # (agn_log_lbol) instead of the now-derived Eddington ratio, so the zone
    # structure (T_in, radii) is self-consistent with L_bol. lambda_Edd is
    # recovered downstream as L_bol / L_Edd (see _compute_zone_radii).
    if float32:
        # Float32 (#1206): the ~1e44 erg/s l_bol_erg intermediate overflows;
        # mdot ~1e24 g/s is representable. Fold L_sun/c^2 (a pre-divided
        # constant ~4e12) so only ``10**log_lbol`` (~1e11) is materialized.
        mdot = _LSUN_OVER_C2 * 10.0**agn_log_lbol / eta
        # l_edd (~1e46 erg/s) stays out-of-range here; downstream float32
        # branches recompute it in L_sun units from agn_log_mbh.
    else:
        l_bol_erg = 10.0**agn_log_lbol * _LSUN_ERG
        mdot = l_bol_erg / (eta * _C_LIGHT**2)

    return r_g, r_isco_rg, r_isco_cm, eta, l_edd, mdot


def _compute_zone_radii(
    r_g: float,
    r_isco_rg: float,
    r_isco_cm: float,
    t_in: float,
    agn_log_mbh: float,
    agn_log_lbol: float,
    agn_f_hard: float,
    agn_r_warm_ratio: float,
    l_edd: float,
    float32: bool = False,
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
    agn_log_lbol : float
        Bolometric luminosity. [log10(L_bol / L_sun)]
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
    # Float32 (#1206): l_edd ~1e46 erg/s overflows, but the zone structure needs
    # only the ratio l_hot_target/l0 (in the bisection) and lambda_Edd = L_bol /
    # L_Edd. Work L_Edd in L_sun (linear in M_BH) so both stay representable.
    if float32:
        l_edd_lsun = _L_EDD_1MSUN_LSUN * 10.0**agn_log_mbh
        l_hot_target = f_hard_safe * l_edd_lsun  # L_sun
        r_hot_cm = _r_hot_bisect(r_isco_cm, t_in, l_hot_target, float32=True)
        l_edd_ratio = jnp.clip(10.0**agn_log_lbol / l_edd_lsun, 1e-10, 1.0)
    else:
        l_hot_target = f_hard_safe * l_edd
        r_hot_cm = _r_hot_bisect(r_isco_cm, t_in, l_hot_target)
        # E fix (#846): lambda_Edd = L_bol / L_Edd, derived from the requested
        # agn_log_lbol (not the now-derived agn_log_ledd).
        l_edd_ratio = jnp.clip(10.0**agn_log_lbol * _LSUN_ERG / l_edd, 1e-10, 1.0)

    r_warm_ratio_safe = jnp.clip(agn_r_warm_ratio, 1.1, 10.0)
    r_warm_cm = r_hot_cm * r_warm_ratio_safe
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
    float32: bool = False,
    agn_log_mbh: float = DEFAULT_AGN_LOG_MBH,
    agn_log_lbol_shape: float = 0.0,
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
        kTbb_keV = _K_BOLTZ_KEV * t_ring
        shape = _nthcomp_lnu_interp(nu, agn_gamma_warm, agn_kt_warm, kTbb_keV)
        if float32:
            # Float32 (#1206): the ring bolometric ``p_plain * ring_area`` ~1e42
            # erg/s overflows, though the ring L_nu (~1e27) is representable.
            # Fold the tiny normalized ``shape`` in first so no ~1e42 forms.
            return (shape * p_plain) * _ring_area(r_cm, dr_ring, agn_cos_inc)
        l_total = p_plain * _ring_area(r_cm, dr_ring, agn_cos_inc)
        return shape * l_total

    l_nu_warm = jnp.sum(jax.vmap(_warm_ring)(r_warm_grid, t_warm, dr_warm), axis=0)

    # ── Zone 3: Hot corona (R_ISCO < r < R_hot) ───────────────────
    f_hard_safe = jnp.clip(agn_f_hard, 1e-6, 0.5)
    # Float32 (#1206): l_hot_erg ~5e43 and l_seed ~1e44 erg/s overflow. Work both
    # in L_sun units (l_edd from M_BH, L_bol from the SHAPE luminosity — the
    # corona fraction lambda_Edd must track the TRUE L_bol, not the reference the
    # magnitude normalizes to). Beloborodov uses only their ratio, so units cancel.
    if float32:
        _l_edd_lsun = _L_EDD_1MSUN_LSUN * 10.0**agn_log_mbh
        l_hot_erg = jnp.minimum(f_hard_safe * _l_edd_lsun, 10.0**agn_log_lbol_shape * 0.5)
        l_seed_geom = _l_seed_geometric(r_isco_cm, r_hot_cm, r_out_cm, t_in, float32=True)
    else:
        l_hot_erg = jnp.minimum(f_hard_safe * l_edd, l_bol_erg * 0.5)
        l_seed_geom = _l_seed_geometric(r_isco_cm, r_hot_cm, r_out_cm, t_in)

    kt_hot_erg = agn_kt_hot * _KEV_TO_ERG

    gamma_hard_sc = beloborodov_gamma_hot(l_hot_erg, l_seed_geom)
    gamma_hard_eff = jnp.where(agn_self_consistent_gamma, gamma_hard_sc, agn_gamma_hard)

    # Seed-photon frequency for the hot flow (K&D 2018, Section 2.2): the seed
    # photons come from the inner edge of the warm Comptonization region at
    # R_hot, boosted by the warm Compton y-parameter,
    #   kT_seed,hot = k T_NT(R_hot) * exp(y_warm),
    # with y_warm recovered from Gamma_warm via the standard non-relativistic
    # thermal-Comptonization relation Gamma = sqrt(9/4 + 4/y) - 1/2
    # (Sunyaev & Titarchuk 1980), i.e. y_warm = 4 / [(Gamma_warm + 1/2)^2 - 9/4].
    # This sets the low-energy rollover so the corona cannot leak into the IR/radio.
    t_seed_nt = t_warm[0]  # T_NT(R_hot): first (innermost) warm-zone annulus
    y_warm_denom = jnp.maximum((agn_gamma_warm + 0.5) ** 2 - 2.25, 1e-3)
    y_warm = jnp.clip(4.0 / y_warm_denom, 0.0, 10.0)
    t_seed_hot = t_seed_nt * jnp.exp(y_warm)
    nu_seed_hot = _K_BOLTZ * t_seed_hot / _H_PLANCK

    # The corona L_nu scales linearly with l_hot_erg. On the float32 path
    # l_hot_erg is in L_sun, so the corona comes out in L_sun/Hz — convert back to
    # erg/s/Hz by folding L_sun in (the ~1e28 product forms without the ~5e43
    # intermediate).
    l_nu_hot = _hot_corona_lnu(nu, l_hot_erg, gamma_hard_eff, kt_hot_erg, nu_seed_hot)
    if float32:
        l_nu_hot = l_nu_hot * _LSUN_ERG

    # ── Combine and normalize ─────────────────────────────────────
    l_nu_total = l_nu_outer + l_nu_warm + l_nu_hot

    # Zone bolometric integrals (sigma T^4 * 2*pi*r*dr) reach ~1e44 erg/s and
    # overflow float32; work them in L_sun (pre-divided 2*pi*sigma/L_sun folds
    # first). l_hot_erg is already L_sun on this path, and ``l_bol_erg`` is passed
    # in L_sun too (the reference normalization), so ``scale`` is a clean ratio.
    if float32:
        l_bol_outer = jnp.sum(
            _2PI_SIGMA_SB_OVER_LSUN
            * t_outer**4
            * r_outer
            * dr_outer
            * jnp.maximum(agn_cos_inc, 0.01)
        )
        l_bol_warm = jnp.sum(
            _2PI_SIGMA_SB_OVER_LSUN
            * t_warm**4
            * r_warm_grid
            * dr_warm
            * jnp.maximum(agn_cos_inc, 0.01)
        )
    else:
        l_bol_outer = jnp.sum(
            _SIGMA_SB
            * t_outer**4
            * 2.0
            * jnp.pi
            * r_outer
            * dr_outer
            * jnp.maximum(agn_cos_inc, 0.01)
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
    agn_lum_ratio: float = DEFAULT_AGN_LUM_RATIO,
    agn_log_mbh: float = DEFAULT_AGN_LOG_MBH,
    agn_log_ledd: float = -1.0,
    agn_a_spin: float = 0.0,
    agn_cos_inc: float = DEFAULT_AGN_COS_INC,
    agn_f_hard: float = 0.02,
    agn_gamma_warm: float = 2.5,
    agn_kt_warm: float = 0.2,
    agn_gamma_hard: float = 1.8,
    agn_kt_hot: float = 100.0,
    agn_r_warm_ratio: float = 2.0,
    n_radii: int = 50,
    agn_self_consistent_gamma: bool = False,
    agn_log_lbol_shape: float | None = None,
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
    agn_lum_ratio : float, optional
        Fraction of bolometric luminosity emitted by the disc system (all zones).
        Default: 1.0. [dimensionless, 0–1]
    agn_log_mbh : float, optional
        Black hole mass. Determines the Eddington luminosity and temperature
        scaling. Default: 8.0. [log10(M_sun)]
    agn_log_ledd : float, optional
        **DEPRECATED / IGNORED (#846).** The Eddington ratio — and hence the
        inner temperature, accretion rate, and zone radii — is now DERIVED from
        ``agn_log_lbol`` and ``agn_log_mbh`` (lambda_Edd = L_bol / L_Edd), so the
        3-zone structure is self-consistent with the requested L_bol. Retained
        for backward compatibility but has no effect; setting or freeing it
        emits a build-time warning. Default: -1.0.
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
    (Γ_warm, kT_e, normalization)-dependent SED at each radius.
    If templates are unavailable, a simplified modified-blackbody proxy is used,
    which has ~5–10% shape error but allows offline computation.

    **Approximations and accuracy**:

    The reference QSOSED/RELAGN codes use non-differentiable operations
    (root solvers, C implementations). Tengri's JAX reimplementation makes
    key approximations documented in the code (see
    ``docs/dev/archive/design/agn_kd_model.md``):

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
    _f32 = wavelength.dtype == jnp.float32
    # Shape luminosity (temperature, zone structure, corona fraction) vs the
    # normalization luminosity (output magnitude). They coincide by default (the
    # float64 path). On float32 the AGN component passes the TRUE L_bol for the
    # SHAPE while normalizing MAGNITUDE to a low reference, so the runner's ~1e40
    # L_lambda arithmetic stays in float32 range; the true scale is re-applied
    # downstream (#1206).
    _lbol_shape = agn_log_lbol if agn_log_lbol_shape is None else agn_log_lbol_shape

    r_g, r_isco_rg, r_isco_cm, _eta, l_edd, mdot = _compute_bh_params(
        agn_log_mbh, _lbol_shape, agn_a_spin, float32=_f32
    )

    # Novikov-Thorne inner-disc temperature.
    if _f32:
        # Log-space: the ``3 G M mdot`` numerator ~1e58 erg/s overflows float32;
        # t_in ~1e5 K is representable.
        _log_t_in4 = (
            _LOG10_3G
            + agn_log_mbh
            + _LOG10_MSUN_G
            + jnp.log10(mdot)
            - _LOG10_8PI_SIGMA_SB
            - 3.0 * jnp.log10(r_isco_cm)
        )
        t_in = _pow10(0.25 * _log_t_in4)
    else:
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
        _lbol_shape,
        agn_f_hard,
        agn_r_warm_ratio,
        l_edd,
        float32=_f32,
    )

    # Normalization magnitude from agn_log_lbol (the reference on the float32
    # path); pass it in L_sun there so the zone helper's ``scale`` is a clean ratio.
    if _f32:
        l_bol_requested = 10.0**agn_log_lbol * agn_lum_ratio  # L_sun
    else:
        l_bol_requested = 10.0**agn_log_lbol * _LSUN_ERG * agn_lum_ratio

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
        float32=_f32,
        agn_log_mbh=agn_log_mbh,
        agn_log_lbol_shape=_lbol_shape,
    )

    return l_nu_total * scale


# ── Model 4: ADAF — DEPRECATED delegator to the faithful adaf_spectrum ────────


def adaf_disc(
    wavelength,
    agn_log_lbol,
    agn_lum_ratio=DEFAULT_AGN_LUM_RATIO,
    agn_log_mbh=DEFAULT_AGN_LOG_MBH,
    agn_adaf_beta=0.5,
    agn_adaf_delta=0.1,
    agn_adaf_alpha=0.3,
    **_legacy_kwargs,
):
    """DEPRECATED thin alias for the faithful Mahadevan 1997 ADAF (#898).

    The old ``adaf_disc`` (which misapplied Eq. 49 and bundled an ad-hoc truncated
    outer disc) was removed; this delegates to
    :func:`~tengri.components.agn.adaf.adaf_spectrum`. The retired arguments
    ``agn_log_ledd`` / ``agn_r_tr`` / ``agn_cos_inc`` are accepted for backward
    compatibility but **ignored** (mdot is now derived from ``agn_log_lbol``).
    New code should call ``adaf_spectrum`` directly. Slated for removal once its
    remaining callers/tests migrate (see #898 follow-up).

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength [Angstrom].
    agn_log_lbol : float
        log10 of bolometric luminosity [Lsun].
    agn_lum_ratio, agn_log_mbh, agn_adaf_beta, agn_adaf_delta, agn_adaf_alpha
        Forwarded to :func:`adaf_spectrum` (see there).

    Returns
    -------
    ndarray, shape (n_wave,)
        L_nu [erg/s/Hz], the faithful Mahadevan 1997 ADAF.
    """
    from tengri.components.agn.adaf import adaf_spectrum

    return adaf_spectrum(
        wavelength,
        agn_log_lbol=agn_log_lbol,
        agn_lum_ratio=agn_lum_ratio,
        agn_log_mbh=agn_log_mbh,
        agn_adaf_alpha=agn_adaf_alpha,
        agn_adaf_beta=agn_adaf_beta,
        agn_adaf_delta=agn_adaf_delta,
    )


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
        # Kept on this branch (main's #1578 dropped it): the float32
        # renormalization below is this branch's #1206 work and reads it. Read
        # off the declaration rather than the literal 11.0 it used to repeat,
        # which is #1578's own rule.
        agn_log_lbol: float = DEFAULT_AGN_LOG_LBOL,
        agn_log_mbh: float = DEFAULT_AGN_LOG_MBH,
        agn_log_mdot: float = -1.0,
        agn_astar: float = 0.0,
        agn_cos_inc: float = DEFAULT_AGN_COS_INC,
        **_kwargs,
    ) -> jnp.ndarray:
        """RELAGN outer disc from relativistic template grid.

        Parameters
        ----------
        wavelength : ndarray, shape (n_wave,)
            Rest-frame wavelength. [Å]
        agn_log_lbol : float
            log₁₀(L_bol / L_sun) — the normalization the template is scaled to.
            [dimensionless]
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

        **Normalization (behavior change, #1206).** The template *shape* comes from
        (M_BH, Ṁ, a\\*); its **normalization** is set by ``agn_log_lbol``, matching
        every other disc in the composable menu (``multicolor``, ``kubota_done``,
        ``slone_netzer``, …). Previously the grid's own absolute normalization was
        used and ``agn_log_lbol`` had no effect — which both surprised users who set
        it and made the disc unusable in float32, since the grid's absolute
        ``λL_λ(5100 Å) ≈ 2.6e44`` erg/s exceeds the float32 maximum (3.4e38).

        As with ``multicolor_disc``, the bolometric renormalization divides out any
        wavelength-independent prefactor, so ``agn_cos_inc`` (a pure ``2 cos i``
        scaling of this grid) no longer changes the disc's normalization; viewing
        anisotropy enters downstream through the runner's inclination handling.
        """
        point = (agn_log_mbh, agn_log_mdot, agn_astar)
        lnu_template = _interp_nd_triweight(grid_jax, axes, edges, point, scatters=scatters)
        # Interpolate grid wavelength → observation wavelength
        lnu_interp = resample_template(wavelength, wave_grid, lnu_template, left=0.0, right=0.0)
        # Inclination scaling from reference cos_inc = 0.5
        lnu_interp = lnu_interp * (2.0 * agn_cos_inc)

        # Renormalize to the requested bolometric luminosity (#1206).
        nu = _wavelength_to_nu(wavelength)
        l_scale = 10.0**agn_log_lbol * _LSUN_ERG
        if wavelength.dtype == jnp.float32:
            # Float32: the template's own bolometric integral is ~1e45 erg/s and
            # overflows, which would flush the disc to zero. Peak-factor and
            # regroup — ``(l_scale / hat_int) * (lnu / peak)`` is algebraically
            # identical to ``l_scale * lnu / (peak * hat_int)``.
            # stop_gradient: factorization constant; peak * hat_int == bolint(lnu) (#1436).
            peak = jax.lax.stop_gradient(jnp.max(jnp.abs(lnu_interp)))
            peak = jnp.where(peak > 0.0, peak, 1.0)
            hat_int = _bolometric_integral_nu(lnu_interp / peak, nu, floor=1e-30)
            return (l_scale / hat_int) * (lnu_interp / peak)
        integral_safe = _bolometric_integral_nu(lnu_interp, nu, floor=1e-100)
        return l_scale * lnu_interp / integral_safe

    return relagn_disc
