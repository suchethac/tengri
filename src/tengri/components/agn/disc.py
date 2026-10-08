# SPDX-License-Identifier: BSD-3-Clause
"""Accretion disc models for AGN emission.

Four models are provided:

1. **Simple power-law + UV cutoff**: minimal AGN disc with 3 parameters.
2. **Multi-color disc (Shakura-Sunyaev)**: physically-motivated standard thin
   disc following Kubota & Done (2018), simplified to the key parameters.
   Implements the outer standard disc zone only.
3. **Kubota & Done 3-zone disc**: full K&D (2018) model with outer standard
   disc, warm Comptonization (soft X-ray excess), and hot corona (hard X-ray
   power law). Three radially-stratified zones with self-consistent radii.
4. **ADAF + truncated disc**, for low-luminosity AGN (L/L_Edd < 0.01).
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
import numpy as np

from tengri.components.agn._nt_emissivity import (
    X_MAX as _NT_X_MAX,
    isco_radius as _isco_radius,
    nt_dh_dlogx as _nt_dh_dlogx,
    nt_h as _nt_h,
    nt_h_ceiling as _nt_h_ceiling,
    nt_rt as _nt_rt,
)
from tengri.components.agn._nthcomp import (
    _TABLE_AVAILABLE as _NTHCOMP_AVAILABLE,
    nthcomp_lnu_interp as _nthcomp_lnu_interp,
    nthcomp_norm_grid as _nthcomp_norm_grid,
)
from tengri.components.agn._params import (
    DEFAULT_AGN_COS_INC,
    DEFAULT_AGN_LOG_LBOL,
    DEFAULT_AGN_LOG_MBH,
    DEFAULT_AGN_LUM_RATIO,
)
from tengri.components.agn._phys import (
    C_LIGHT as _C_LIGHT,
    COS_INC_ISOTROPIC_REFERENCE as _COS_INC_ISOTROPIC_REFERENCE,
    H_PLANCK as _H_PLANCK,
    K_BOLTZ as _K_BOLTZ,
    TWO_FACES as _TWO_FACES,
    planck_lnu as _planck_lnu,
    ring_area as _ring_area,
    wavelength_to_nu as _wavelength_to_nu,
)
from tengri.components.agn._template_grid import scale_to_lbol_native
from tengri.utils.grid_interp import interp_nd_triweight as _interp_nd_triweight, resample_template
from tengri.utils.host_array import device_table, host_array
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
from tengri.utils.scale import (
    apply_log10_scale,
    log10_add,
    log10_weighted_sum,
    pow10 as _pow10,
    representable_denominator as _representable_denominator,
    representable_exponent,
    representable_floor as _representable_floor,
)

#: Scale of the nthcomp template shape [dimensionless]. The shape integrates to 1 over frequency,
#: so it is ~1e-17 per Hz while the ring factor it multiplies is ~1e47: dividing it by this unit
#: makes it O(1), and the unit is restored as an exponent on the summed spectrum (#2767).
_NTHCOMP_UNIT: float = 1.0e-17
_LOG10_NTHCOMP_UNIT: float = math.log10(_NTHCOMP_UNIT)

# log10 of the cgs constants that make the Shakura-Sunyaev disc's bolometric /
# Eddington / accretion-rate intermediates overflow float32 (#1206). At a
# realistic AGN luminosity the LINEAR forms: L_bol ~1e44, L_Edd ~1e46, the
# ``t_in**4`` numerator ~1e58 erg/s: all exceed float32 max (3.4e38), yet the
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
# ENERGY BUDGET. ``agn_log_lbol`` is the accretion power ``L_acc = int (D_nu + H_nu) dnu``: the
# radiation of the disc and warm zones, ``D_nu`` (BOTH faces, all directions), plus that of the
# corona, ``H_nu`` (isotropic). Every annulus emits ``dL_nu(i) = 4 pi B_nu(T) 2 pi r dr cos i``
# (``_ring_area``), the luminosity density an observer at inclination i assigns assuming
# isotropy; its frequency integral is ``4 sigma T^4 2 pi r dr cos i = 2 cos i dD`` with
# ``dD = 2 sigma T^4 2 pi r dr`` the two-face power of the ring. The bolometric normalizers sum
# ``D`` through ``_TWO_FACES * sigma T^4 2 pi r dr`` (no cos i), so the returned spectrum is
# ``2 cos i D_nu + H_nu``: its mean over cos i in [0, 1] is ``L_acc`` and its power at
# cos i = 0.5 is ``L_acc``. The hot corona enters ``kubota_done``'s normalizer as its FULL power
# ``l_hot_erg`` (K&D 2018 Eq. 2's two-face dissipation, isotropic), and ``l_seed`` is not
# summed: it only sets Gamma_hot.
# int B_nu(T) dnu = (2 pi^4 k^4 / 15 h^3 c^2) T^4 = sigma_Planck T^4 / pi, built from the very
# h, k, c that ``planck_lnu`` uses so the identity is exact for it (it differs from the tabulated
# sigma_SB by the constants' rounding). [erg s^-1 cm^-2 Hz^0 sr^-1 K^-4]. Used wherever a ring
# or disc bolometric content is needed: analytic, never a quadrature on the caller's grid (#2572).
_BNU_BOL_PER_T4: float = float(
    2.0 * math.pi**4 * _K_BOLTZ**4 / (15.0 * _H_PLANCK**3 * _C_LIGHT**2)
)
# 4*pi*sigma_SB / L_sun; l0_lsun = _4PI_SIGMA_SB_OVER_LSUN * r_isco_cm**2 * t_in**4.
_4PI_SIGMA_SB_OVER_LSUN: float = float(4.0 * math.pi * _SIGMA_SB) / float(_LSUN_ERG)
# 2*pi*sigma_SB / L_sun; per-annulus (sigma T^4 * 2*pi*r*dr) energy in L_sun.
_2PI_SIGMA_SB_OVER_LSUN: float = float(2.0 * math.pi * _SIGMA_SB) / float(_LSUN_ERG)
# L_sun / c^2; mdot [g/s] = _LSUN_OVER_C2 * 10**log_lbol / eta (avoids l_bol_erg).
_LSUN_OVER_C2: float = float(_LSUN_ERG) / float(_C_LIGHT) ** 2
# R_g = G*M_bh/c^2 = _GRAV_RADIUS_PER_MSUN * 10**log_mbh [cm] (#2210): folding
# G*M_sun/c^2 into one ~1.477e5 constant keeps the product inside float32 range
# across the whole declared agn_log_mbh prior, unlike forming M_bh in grams
# (``10**log_mbh * M_sun``) as a standalone ~1e39-1e43 intermediate first.
_GRAV_RADIUS_PER_MSUN: float = _G_GRAV * _MSUN_G / _C_LIGHT**2

# ── Model 1: Simple power-law disc + UV cutoff ────────────────────


#: Internal normalization band of ``powerlaw_disc`` [Angstrom] (descending = ascending nu).
_PL_BAND_AA = np.geomspace(1.0e8, 10.0, 2049)


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
    **JIT-compatible**: yes, all operations use ``jnp`` primitives.

    The unnormalized spectral shape is:

    .. math::

        L_\\nu^{\\rm unnorm} = \\nu^\\alpha \\exp\\left(-\\frac{h\\nu}{k_B T_{\\rm max}}\\right)

    where :math:`\\alpha` is the spectral index, :math:`h` is Planck's constant,
    :math:`\\nu` is frequency [Hz], :math:`k_B` is Boltzmann's constant, and
    :math:`T_{\\rm max}` is the cutoff temperature [K].

    The normalization constant :math:`C` is computed numerically by integrating
    the shape over the fixed band 10 A - 1e8 A (2049 log-spaced internal nodes; the
    caller's wavelength grid plays no part), so that the integral over frequency of the
    shape over that band equals the target luminosity
    :math:`L_{\\rm bol} \\cdot f_{\\rm disc}` and the SED is the same on any grid.

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
    x_clip = jnp.clip(x, 0.0, representable_exponent(500.0, base=math.e))
    shape = nu**agn_alpha * jnp.exp(-x_clip)

    # Normalize over the FIXED band [_PL_LAMBDA_HI, _PL_LAMBDA_LO] (log-spaced internal nodes), not
    # over the caller's grid: the same SED must come out whichever grid it is evaluated on (#2572).
    nu_int = _wavelength_to_nu(jnp.asarray(_PL_BAND_AA, dtype=nu.dtype))
    x_int = jnp.clip(
        _H_PLANCK * nu_int / (_K_BOLTZ * jnp.maximum(agn_T_max, 1.0)),
        0.0,
        representable_exponent(500.0, base=math.e),
    )
    shape_int = nu_int**agn_alpha * jnp.exp(-x_int)
    integral = jnp.trapezoid(shape_int, nu_int)
    integral_safe = jnp.maximum(jnp.abs(integral), 1e-100)

    l_nu_erg = l_bol_erg * agn_lum_ratio * shape / integral_safe
    return l_nu_erg


# ── Model 2: Multi-color disc (Shakura-Sunyaev thin disc) ─────────


def _log10_eddington_luminosity(log_mbh: float) -> float:
    r"""log10 Eddington luminosity, :math:`\log_{10} L_{\rm Edd}` [log10(erg/s)].

    Carried in log space (#2210): the linear form is ~1.26e44 erg/s at the
    bottom of the declared ``agn_log_mbh`` prior, past float32's 3.403e38
    ceiling. Callers form only the Eddington ratio or other log-domain
    combinations; the linear :math:`L_{\rm Edd}` is never materialized.
    """
    return _LOG10_L_EDD_1MSUN + log_mbh


def _gravitational_radius(log_mbh: float) -> float:
    r"""Gravitational radius :math:`R_g = GM/c^2` [cm].

    Regrouped (#2210) so the precomputed :math:`GM_\odot/c^2`
    (``_GRAV_RADIUS_PER_MSUN``, ~1.477e5, in float32 range) multiplies
    :math:`10^{\log_{10} M_{\rm BH}}`, rather than forming :math:`M_{\rm BH}`
    in grams as a standalone ~1e39-1e43 intermediate first.
    """
    return _GRAV_RADIUS_PER_MSUN * _pow10(log_mbh)


def _nt_l_diss_analytic(x_hot: float, r_isco_cm: float, t_in: float, a_spin: float = 0.0) -> float:
    """NT dissipation of the hot flow, K&D 2018 Eq. 2, with the Page-Thorne emissivity.

    Integrates ``F_NT = sigma T_NT^4`` over the disc annuli from R_ISCO to R_hot, both
    faces::

        L_diss = 2 * int_{R_isco}^{R_hot} sigma T_NT^4 * 2 pi R dR = L_0 * h(x_hot)

    with ``L_0 = 4 pi R_isco^2 sigma T_in^4``, ``x = R / R_isco`` and
    ``sigma T_NT^4 = sigma T_in^4 x^-3 Rt(x; a)``, where ``Rt`` is the Page & Thorne
    (1974, ApJ 191, 499) factor of ``tengri.components.agn._nt_emissivity`` (zero at the
    ISCO, -> 1 far out)::

        h(x) = int_1^x x'^-2 Rt(x'; a) dx'

    evaluated by fixed-node Gauss-Legendre quadrature in ``ln x`` (no closed form for
    a != 0). ``h(1) = 0`` (empty corona); for the Newtonian zero-torque profile
    ``Rt = 1 - x^-1/2`` it would be ``1/3 - 1/x + 2/(3 x^{3/2})``. The trailing ``x`` of the
    integrand ``x^-3 Rt x`` is the ``R dR`` area element.

    Parameters
    ----------
    x_hot : float
        R_hot / R_ISCO >= 1.
    r_isco_cm : float
        ISCO radius [cm].
    t_in : float
        Reference temperature ``T_in = (3 G M Mdot / 8 pi sigma R_isco^3)^(1/4)`` [K].
    a_spin : float, optional
        Dimensionless BH spin (default 0).

    Returns
    -------
    float
        L_diss [erg s^-1].
    """
    l0 = 4.0 * jnp.pi * r_isco_cm**2 * _SIGMA_SB * t_in**4
    h_hot = _nt_h(jnp.log(x_hot), a_spin)  # a dissipated power: >= 0 up to round-off
    return l0 * jnp.where(h_hot > 0.0, h_hot, 0.0)


def _nt_l0(r_isco_cm: float, t_in: float, float32: bool = False) -> float:
    """``L_0 = 4 pi R_isco^2 sigma T_in^4`` [erg/s; L_sun on the float32 path (#1206)].

    The NT disc's total dissipation is ``L_0 * h(inf)``; ``h(inf) = 1/3`` for the
    Newtonian profile and 0.233 (a=0) .. 0.1 (a=0.998) for Page-Thorne.
    """
    if float32:
        # Formed from logarithms: the reverse pass of ``r^2 T^4`` multiplies an incoming cotangent
        # by ``T^4`` (~3e20) before the small constant can bring it back, and that product leaves
        # float32 (#2767). The derivative through the exponent is ``L_0 * d(log)``, which does not.
        return _pow10(
            math.log10(_4PI_SIGMA_SB_OVER_LSUN)
            + 2.0 * jnp.log10(r_isco_cm)
            + 4.0 * jnp.log10(t_in)
        )
    return 4.0 * jnp.pi * r_isco_cm**2 * _SIGMA_SB * t_in**4


def _r_hot_bisect(
    r_isco_cm: float,
    t_in: float,
    l_hot_target: float,
    n_iter: int = 40,
    float32: bool = False,
    a_spin: float = 0.0,
) -> float:
    r"""Solve for R_hot from K&D 2018 Eq. 2 by bisection in log(x_hot).

    :math:`L_{\rm diss}(x) = L_0\,h(x)` with the Page-Thorne dissipation integral
    :math:`h` (see :func:`_nt_l_diss_analytic`), strictly monotone in
    :math:`x = R_{\rm hot}/R_{\rm ISCO}`. After ``n_iter=40`` the bracket width is
    :math:`< 2^{-40} \approx 10^{-12}` of its initial log-width: enough for machine
    precision.

    ``l_hot_target`` is clipped to :math:`0.99\,L_0 h(10^4)` (see
    ``nt_h_ceiling`` in ``tengri.components.agn._nt_emissivity``).

    Differentiable: see :func:`_solve_log_x_hot` (implicit-function JVP, #2572).
    """
    # Float32 (#1206): ``l0`` is ~1e42 erg/s (overflow); the bisection needs only
    # the RATIO l_hot_target / l0, so compute both in L_sun units (``l_hot_target``
    # arrives in L_sun on the float32 path). The pre-divided 4*pi*sigma/L_sun
    # constant folds first so no ~1e42 intermediate forms.
    l0 = _nt_l0(r_isco_cm, t_in, float32)
    x_hot = jnp.exp(_solve_log_x_hot(l_hot_target, l0, a_spin, n_iter))
    return x_hot * r_isco_cm


_X_LO = 1.001
_R_HOT_POLISH_STEPS = 3


def _bisect_log_x(l_hot_target, l0, a_spin, n_iter):
    """Bisect ``l0 * h(x) = l_target`` in log(x) over [log 1.001, log 1e4] (``lax.scan``).

    ``l_target = clip(l_hot_target, 1e-100, l0 * h_ceiling(a))``.
    """
    l_target = jnp.clip(l_hot_target, 1e-100, l0 * _nt_h_ceiling(a_spin))

    def _step(state, _):
        """Single bisection step in log-space to solve for R_hot."""
        lo_i, hi_i = state
        mid = (lo_i + hi_i) * 0.5
        go_right = l0 * _nt_h(mid, a_spin) < l_target
        return (jnp.where(go_right, mid, lo_i), jnp.where(go_right, hi_i, mid)), None

    (lo_f, hi_f), _ = jax.lax.scan(
        _step, (jnp.log(_X_LO), jnp.log(_NT_X_MAX)), None, length=n_iter
    )
    # The bracket is 9.2 / 2^n_iter wide (8e-12 at 40), and the SED is steep enough in R_hot
    # that this quantizes the root into steps of ~1e-6 in the parameters. Newton steps on
    # F(u) = l0 h(u) - l_target, kept inside the bracket (which holds the root), converge
    # to the residual of the quadrature itself; a flat or pinned end leaves the midpoint.
    u = (lo_f + hi_f) * 0.5
    for _ in range(_R_HOT_POLISH_STEPS):
        slope = l0 * _nt_dh_dlogx(u, a_spin)
        step = (l0 * _nt_h(u, a_spin) - l_target) / jnp.where(slope > 0.0, slope, 1.0)
        u = jnp.clip(u - jnp.where(slope > 0.0, step, 0.0), lo_f, hi_f)
    return u


@functools.partial(jax.custom_jvp, nondiff_argnums=(3,))
def _solve_log_x_hot(l_hot_target, l0, a_spin, n_iter):
    """Root ``log(x_hot)`` of ``F = l0*h(x; a) - l_target``, differentiable by the IFT (#2572).

    The forward value is the bisection above. Differentiating *through* the ``scan``
    returns exactly 0 (the bracket ends are constants and ``where`` only selects between
    them), so ``R_hot`` carried no sensitivity to ``M_BH``, ``L_bol``, ``f_hard`` or spin
    wherever it is not clipped. The rule below is the implicit-function derivative at the
    converged root::

        d(log x) = (dl_target - h dl0 - l0 (dh/da) da) / (l0 dh/d log x),
        dh/d log x = Rt(x) / x   (exact integrand),  dh/da by autodiff of the quadrature.

    It is exactly 0 where the target's clip (to ``(1e-100, l0 h_ceiling)``) is active or
    the root is pinned to a bracket end, where ``R_hot`` really is independent of the
    target: evaluating the IFT there would leave ~1e-14 of bisection residual amplified
    by ``r_isco`` (~1e14 cm). (The weak ``a`` dependence of the saturation root itself is
    neglected there.)
    """
    return _bisect_log_x(l_hot_target, l0, a_spin, n_iter)


@_solve_log_x_hot.defjvp
def _solve_log_x_hot_jvp(n_iter, primals, tangents):
    """Implicit-function-theorem JVP of :func:`_solve_log_x_hot`."""
    l_hot_target, l0, a_spin = primals
    d_target, d_l0, d_a = tangents
    u = _bisect_log_x(l_hot_target, l0, a_spin, n_iter)
    h_u, dh_da = jax.jvp(lambda aa: _nt_h(u, aa), (a_spin,), (jnp.asarray(d_a, dtype=u.dtype),))
    dF_du = l0 * _nt_dh_dlogx(u, a_spin)
    interior = (
        (l_hot_target > l0 * _nt_h(jnp.log(_X_LO), a_spin))
        & (l_hot_target < l0 * _nt_h_ceiling(a_spin))
        & (l_hot_target > 1e-100)
    )
    neg_dF_dtheta = d_target - h_u * d_l0 - l0 * dh_da  # -dF/dtheta for F = l0 h - l_target
    safe = jnp.where(interior, dF_du, 1.0)
    tangent = jnp.where(interior, neg_dF_dtheta / safe, 0.0)
    return u, tangent


def _l_seed_geometric(
    r_isco_cm: float,
    r_hot_cm: float,
    r_out_cm: float,
    t_in: float,
    n_radii: int = 100,
    float32: bool = False,
    a_spin: float = 0.0,
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
    rt = jnp.maximum(_nt_rt(r_ratio, a_spin), 1e-30) ** 0.25  # Page-Thorne (1974) factor
    t_r = t_in * r_ratio ** (-0.75) * rt  # [K]
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

    Laor & Netzer (1989):
        r_sg = 2150 * alpha^{2/9} * lambda_Edd^{4/9}
               * (M_BH / 10^9 M_sun)^{-2/9}   [R_g]

    where lambda_Edd = Mdot / Mdot_Edd (= L_bol / L_Edd) is the Eddington ratio and
    alpha is the Shakura-Sunyaev viscosity parameter (default 0.1). ``alpha`` enters as
    ``alpha^{2/9}`` (not ``(alpha/0.1)^{2/9}``), so at alpha = 0.1 the form is
    ``2150 (M/1e8)^{-2/9} lambda^{4/9}``. This is the expression evaluated by
    ``Sed.gravity_radius`` in qsosed (Quera-Bofarull) and ``calc_rsg`` in the
    QSOSED/RELQSO Fortran (Hagen & Done), and the outer radius of the K&D 2018 disc
    ("rout ... set to equal the self-gravity rsg (Laor & Netzer 1989)").

    Reference: Laor, A. & Netzer, H. (1989), MNRAS 238, 897.

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
    # M_BH / 10^9 M_sun, as one exponential of a difference: the reverse pass multiplies by this
    # value itself, not by ``10**log_mbh`` ahead of the division by 1e9, which leaves float32
    # (#2767).
    m9 = _pow10(log_mbh - 9.0)
    m9_safe = jnp.maximum(m9, 1e-6)
    lambda_safe = jnp.clip(l_edd_ratio, 1e-10, 1.0)
    alpha_safe = jnp.maximum(alpha_visc, 1e-4)
    return (
        2150.0 * alpha_safe ** (2.0 / 9.0) * lambda_safe ** (4.0 / 9.0) * m9_safe ** (-2.0 / 9.0)
    )


# ── EUV / soft-X-ray power-law tail for the thin disc ─────────────────
# A bare Shakura-Sunyaev disc Wien-cuts off in the EUV (< ~150 A), so its
# emission below 100 A is negligible. Empirical AGN disc templates (e.g.
# CIGALE's SKIRTOR piecewise power law) instead carry a rising power-law
# tail into the EUV / soft X-ray. ``multicolor_disc`` can optionally blend
# such a tail onto the Wien core (the ``euv_tail`` argument).
_EUV_TAIL_LAMBDA_BREAK_AA = 912.0  # [A] Lyman limit: onset of the EUV tail
_EUV_TAIL_LAMBDA_CUT_AA = 30.0  # [A] short-wavelength floor (~0.41 keV)
_EUV_TAIL_DEFAULT_SLOPE = 1.0  # L_nu ~ nu^slope (CIGALE-skirtor-like rise)
_EUV_TAIL_FRAC = 0.02  # tail bolometric budget as a fraction of L_disc
_C_AA_PER_S = float(_C_LIGHT) * 1.0e8  # c [A/s]


def _euv_tail_slope(euv_tail):
    """Static slope of the EUV tail, or ``None`` when the tail is off."""
    if euv_tail is None or euv_tail == "wien":
        return None
    if euv_tail in ("powerlaw", "both"):
        return _EUV_TAIL_DEFAULT_SLOPE
    return float(euv_tail)


def _euv_tail_lnu(wavelength, nu, slope, amplitude):
    """EUV tail ``amplitude * (nu/nu_break)^slope`` on 30-912 A, 0 elsewhere. [erg/s/Hz]"""
    nu_break = _wavelength_to_nu(jnp.asarray(_EUV_TAIL_LAMBDA_BREAK_AA))
    in_euv = (wavelength <= _EUV_TAIL_LAMBDA_BREAK_AA) & (wavelength >= _EUV_TAIL_LAMBDA_CUT_AA)
    return jnp.where(in_euv, (nu / nu_break) ** slope, 0.0) * amplitude


def _euv_tail_shape_integral(slope):
    r"""``int (nu/nu_break)^s dnu`` over the 30-912 A band, closed form [Hz].

    :math:`\nu_b/(s+1)\,[(\nu_c/\nu_b)^{s+1} - 1]` (``nu_b ln(nu_c/nu_b)`` for ``s = -1``).
    """
    nu_b = _C_AA_PER_S / _EUV_TAIL_LAMBDA_BREAK_AA  # plain floats: static, safe under jit/grad
    nu_c = _C_AA_PER_S / _EUV_TAIL_LAMBDA_CUT_AA
    if abs(slope + 1.0) < 1e-12:
        return nu_b * math.log(nu_c / nu_b)
    return nu_b / (slope + 1.0) * ((nu_c / nu_b) ** (slope + 1.0) - 1.0)


#: Internal nodes for the EUV-band excess integral (log-spaced, 912 A -> 30 A = ascending nu).
_EUV_BAND_AA = np.geomspace(_EUV_TAIL_LAMBDA_BREAK_AA, _EUV_TAIL_LAMBDA_CUT_AA, 1025)


def _apply_euv_tail(wavelength, nu, l_nu_wien, euv_tail, disc_bol_lsun):
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

        * ``None`` / ``"wien"``: no tail; pure Wien cutoff (bare thin disc).
        * ``"powerlaw"`` / ``"both"``: blend a power-law tail at the default
          slope ``_EUV_TAIL_DEFAULT_SLOPE``. ``"both"`` is a synonym; the Wien
          core is always preserved (the tail only fills where it exceeds Wien).
        * float: user-defined slope :math:`s` with :math:`L_\nu \propto \nu^s`.
    disc_bol_lsun : float
        Bolometric luminosity of the Wien disc, analytic (sum of the ring Stefan-Boltzmann
        powers), independent of the caller's grid. [L_sun]

    Returns
    -------
    ndarray, shape (n_wave,)
        Disc spectrum with the EUV tail blended in (pre-normalization). [erg/s/Hz]

    Notes
    -----
    **JIT-compatible**: yes, ``euv_tail`` is resolved to a static slope and a
    static on/off flag at trace time (it is not a traced array), so the
    branch is a Python-level decision and the math is pure ``jnp``.

    The tail has the **shape** :math:`L_\nu \propto \nu^s` over
    :math:`\lambda \in [30, 912]` A but its **amplitude** is fixed by
    normalizing its bolometric content to a small fraction
    (``_EUV_TAIL_FRAC``, default 2 %) of the disc's pre-tail bolometric
    luminosity, a bounded "soft-excess"-like budget. (A bare power law
    anchored to the disc peak would diverge in energy and swamp the optical.)
    The shape integral and the disc bolometric are closed-form, so the amplitude does not
    depend on the wavelength grid the SED is evaluated on (#2572). It is blended via
    ``maximum`` so the UV/optical/Wien-peak region is untouched: the tail only contributes
    where the Wien spectrum has already fallen below it. The caller renormalizes the blended
    spectrum back to :math:`L_{\rm bol}`, so total energy is conserved and the optical is
    reduced only by the ~2 % moved into the EUV.
    """
    slope = _euv_tail_slope(euv_tail)
    if slope is None:
        return l_nu_wien
    # amplitude in erg/s/Hz: FRAC * L_disc / int shape dnu. Folding L_sun in last keeps the
    # ~1e43 erg/s disc bolometric (past float32's 3.4e38) from ever forming (#1206).
    if l_nu_wien.dtype == jnp.float32:
        # Float32 (#1206/#1436): peak-factor the amplitude exactly as the grid-quadrature form did,
        # so the reverse-mode cotangent products stay in range: ``disc_bol_hat`` is the O(1e-3..1)
        # bolometric in units of the peak (stop_gradient constant, multiplied back).
        _peak = jax.lax.stop_gradient(jnp.max(jnp.abs(l_nu_wien)))
        _peak = jnp.where(_peak > 0.0, _peak, 1.0)
        _disc_bol_hat = disc_bol_lsun * (_LSUN_ERG / _peak)
        amplitude = (_EUV_TAIL_FRAC * _disc_bol_hat / _euv_tail_shape_integral(slope)) * _peak
    else:
        amplitude = (_EUV_TAIL_FRAC * disc_bol_lsun / _euv_tail_shape_integral(slope)) * _LSUN_ERG
    tail = _euv_tail_lnu(wavelength, nu, slope, amplitude)
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
        Accretion power of the disc, integrated over all directions (both
        faces); independent of the inclination. [log10(L_sun)]
    agn_lum_ratio : float, optional
        Fraction of the accretion power emitted by the disc.
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
        Cosine of the inclination angle. Range: [0, 1]. The returned spectrum
        scales as :math:`2\\cos i`; the shape does not depend on it.
        Default: 0.866 (30°). [dimensionless]
    n_radii : int, optional
        Number of radial bins for numerical integration. Default: 50.
    euv_tail : {"powerlaw", "both", "wien"}, float, or None, optional
        EUV / soft-X-ray behavior below the Lyman limit (912 A).

        * ``"powerlaw"`` (default) / ``"both"``: blend a CIGALE-like power-law
          tail onto the Wien core so the disc carries flux below ~100 A.
        * ``"wien"`` / ``None``: pure Shakura-Sunyaev Wien cutoff (the bare
          thin disc; emission below ~150 A is negligible and the EUV / soft
          X-ray is supplied by the corona instead).
        * float: user-defined slope :math:`s` with :math:`L_\\nu \\propto \\nu^s`.

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
    **JIT-compatible**: yes, uses ``jnp`` primitives and ``jax.vmap``.
    ``euv_tail`` is a static (trace-time) selector, not a traced argument.

    The temperature profile is the Shakura-Sunyaev thin disc with a zero-torque inner
    boundary at the ISCO (a Newtonian-flux form with the Bardeen ISCO radius), *not* the
    relativistic Page & Thorne (1974) emissivity: that is used by the Kubota & Done family
    (:func:`kubota_done_disc`, see ``tengri.components.agn._nt_emissivity``), whose
    reference model defines it. Its total dissipation is ``1.46 eta Mdot c^2`` at a=0
    (1.0 for Page-Thorne), so ``L_bol`` here is a normalization, not an energy balance:

    .. math::

        T(r) = T_{\\rm in} \\left(\\frac{r}{r_{\\rm ISCO}}\\right)^{-3/4}
               \\left[1 - \\sqrt{\\frac{r_{\\rm ISCO}}{r}}\\right]^{1/4}

    where :math:`T_{\\rm in}` is the inner temperature determined by the
    accretion rate and :math:`r_{\\rm ISCO}` is the innermost stable circular
    orbit (radius from Bardeen et al. 1972, depends on spin).

    The disc luminosity is the luminosity density an observer at inclination
    :math:`i` assigns assuming isotropy:

    .. math::

        L_\\nu(i) = \\sum_{j=1}^{N_r} 4\\pi B_\\nu(T_j)\\, 2\\pi r_j \\, dr_j \\cos i
                  = 2\\cos i\\; D_\\nu ,

    where :math:`B_\\nu(T)` is the Planck function, :math:`r_j` is the ring
    radius [cm], :math:`dr_j` is the ring width [cm] and :math:`D_\\nu` is the
    angle-integrated (two-face) spectral luminosity. ``agn_log_lbol`` fixes
    :math:`\\int D_\\nu\\,d\\nu`: the mean of :math:`\\int L_\\nu(i)\\,d\\nu` over
    :math:`\\cos i \\in [0, 1]` is :math:`L_{\\rm bol}`, and at :math:`\\cos i = 0.5` the
    line-of-sight power equals it.

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
    # from agn_log_ledd and then rescaled to agn_log_lbol, a decoupling that
    # left T_in / r_out corresponding to the wrong luminosity. agn_log_ledd is
    # now ignored here (a build-time warning fires if a user sets/frees it).
    #
    # Shape vs normalization luminosity (#1206). The disc SHAPE (T_in, r_out,
    # lambda_Edd) is set by ``_log_lbol_shape``; the output MAGNITUDE by
    # ``agn_log_lbol`` (the renorm target below). They coincide by default
    # (``agn_log_lbol_shape=None``), which is the float64 path. Only float32
    # separates them, the AGN component evaluates the SHAPE at the true L_bol
    # but normalizes MAGNITUDE to a low reference so the runner's ~1e40 L_lambda
    # arithmetic stays in float32 range; the true magnitude is re-applied
    # downstream. See the float32 branch below.
    _log_lbol_shape = agn_log_lbol if agn_log_lbol_shape is None else agn_log_lbol_shape

    # Outer disc radius: Laor & Netzer (1989) self-gravity (Toomre) radius.
    # Beyond r_sg the disc fragments rather than accretes; this is the
    # physically motivated outer boundary used by qsosed (Quera-Bofarull).
    # r_sg = 2150 * alpha^{2/9} * lambda_Edd^{4/9} * (M/1e9)^{-2/9} R_g.
    if wavelength.dtype == jnp.float32:
        # Log-space so the ~1e44 L_bol, ~1e46 L_Edd and ~1e58 erg/s ``t_in**4``
        # numerator never materialize (float32 max 3.4e38). The RESULTS
        # (lambda_Edd ~1e-2, mdot ~1e24 g/s, t_in ~1e5 K) are all representable.
        _log_l_bol_erg = _log_lbol_shape + _LOG10_LSUN_ERG
        _log_l_edd = _log10_eddington_luminosity(agn_log_mbh)
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
        _log_l_edd = _log10_eddington_luminosity(agn_log_mbh)
        # derived: lambda_Edd, formed in log space so L_Edd (~1e46 erg/s) never
        # stands alone (#2210).
        l_edd_ratio = jnp.clip(_pow10(_log_lbol_shape + _LOG10_LSUN_ERG - _log_l_edd), 1e-10, 1.0)
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

    # Integrate: L_nu = sum_i [ B_nu(T_i) * 8 * pi^2 * r_i * dr_i * cos(i) ] = 2 cos(i) D_nu
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
    l_nu_wien = jnp.sum(ring_contributions, axis=0)  # (n_wave,) [erg s^-1 Hz^-1]

    # Bolometric content of the Wien disc, closed form: each ring radiates
    # int B_nu dnu * area = (sigma_Planck T^4 / pi) * area. Folded in L_sun so the ~1e44 erg/s
    # total is representable in float32 (#1206); no quadrature on the caller's grid (#2572).
    def _disc_bol_lsun(cos_inc):
        """Closed-form power of the Wien disc seen at ``cos_inc`` [L_sun]."""
        ring_area_all = jax.vmap(lambda rr, drr: _ring_area(rr, drr, cos_inc))(r_grid, dr)
        # Group the factors so no float32 intermediate leaves the normal range: (c * T^4) ~ 1e19
        # and (area / L_sun) ~ 1e0..1e3; the bare ``c / L_sun`` ~ 5e-39 is subnormal, flushed to 0
        # by XLA.
        return jnp.sum((_BNU_BOL_PER_T4 * t_profile**4) * (ring_area_all / float(_LSUN_ERG)))

    disc_bol_lsun = _disc_bol_lsun(agn_cos_inc)

    # Optionally blend an EUV / soft-X-ray power-law tail onto the Wien core
    # before renormalizing, so the tail's energy is taken out of L_bol rather
    # than added on top (energy-conserving). Default "powerlaw" gives the disc
    # a CIGALE-like rise below ~100 A; "wien" recovers the bare thin disc.
    l_nu_intrinsic = _apply_euv_tail(wavelength, nu, l_nu_wien, euv_tail, disc_bol_lsun)

    # Bolometric content of the BLENDED spectrum, in units of ``unit`` [erg/s]: the Wien disc plus
    # what the tail adds where it exceeds Wien, int max(0, tail - wien) dnu, on fixed internal
    # nodes across the 30-912 A band (the tail is zero outside it). Independent of the caller's
    # grid (#2572). ``unit`` lets the float32 path carry O(1)-O(1e12) numbers (below) instead of
    # ~1e43 erg/s, and keeps the reverse-mode cotangent of the band quadrature in range.
    _slope = _euv_tail_slope(euv_tail)

    def _blended_bol(unit, cos_inc):
        """Power of the blended disc seen at ``cos_inc``, in units of ``unit`` [erg/s]."""
        disc_u = _disc_bol_lsun(cos_inc) * (_LSUN_ERG / unit)
        if _slope is None:
            return disc_u
        _nu_b = _wavelength_to_nu(jnp.asarray(_EUV_BAND_AA, dtype=nu.dtype))
        _tail_u = _euv_tail_lnu(
            device_table(_EUV_BAND_AA),
            _nu_b,
            _slope,
            _EUV_TAIL_FRAC * disc_u / _euv_tail_shape_integral(_slope),
        )
        _wien_u = jnp.sum(
            jax.vmap(
                lambda rr, tt, drr: _planck_lnu(_nu_b, tt) * (_ring_area(rr, drr, cos_inc) / unit)
            )(r_grid, t_profile, dr),
            axis=0,
        )
        _excess_u = _tail_u - _wien_u  # the tail's excess over the Wien form; >= 0 by definition
        return disc_u + jnp.trapezoid(jnp.where(_excess_u > 0.0, _excess_u, 0.0), _nu_b)

    # Renormalize to requested L_bol * agn_lum_ratio (the MAGNITUDE is set by
    # ``agn_log_lbol`` (the reference on the float32 path) NOT the shape
    # luminosity above). ``agn_log_lbol`` is the angle-integrated accretion power, so the
    # normalization is the blended power at ``_COS_INC_ISOTROPIC_REFERENCE`` (where a
    # ``2 cos i`` disc radiates its angle-integrated power), and the spectrum, which is
    # homogeneous of degree one in cos i, keeps its own ``2 cos i``.
    # The bolometric normalization below is closed-form / on fixed internal nodes (#2572).
    if wavelength.dtype == jnp.float32:
        # Log-space renorm: ``l_bol_requested`` ~1e44 and the integral
        # ``l_nu_total`` (l_nu_intrinsic ~1e28 over ~1e15 Hz → ~1e43) both
        # overflow float32; only their ratio (~1e-33 when normalizing a
        # true-shape disc to a 1e10 erg/s reference) is needed. Normalize the
        # integrand so the trapezoid stays in range, and take the exponent
        # against the normalized integral, so neither side is ever formed.
        _log_l_bol_req = agn_log_lbol + _LOG10_LSUN_ERG + jnp.log10(agn_lum_ratio)
        # stop_gradient: factorization constant, canceled by the peak the
        # returned ``_l_hat`` is divided by (#1436).
        _peak = jax.lax.stop_gradient(jnp.max(jnp.abs(l_nu_intrinsic)))
        _peak = jnp.where(_peak > 0.0, _peak, 1.0)
        # The RETURNED array is the peak-normalized one, and the peak is left
        # out of the exponent to match (#1439). Algebraically
        # ``(l/p) * 10**(req - log h) == l * 10**(req - log p - log h)``, but the
        # two differ in *reverse* mode: renormalizing makes ``sum(g * arr)`` a
        # cotangent of the returned array, and JAX forms exactly that inner
        # product when transposing ``arr * scale``. With the raw ``l`` (~1e28)
        # and the cotangent the AGN reference offset hands back (~10**34.6), that
        # sum is ~1e64 -> ``inf`` in float32, while its partner
        # ``d scale/d l ~ 1e-64`` flushes to 0, and ``inf * 0`` is the NaN #1439
        # reported. On the O(1) ``_l_hat`` both factors are in range. Forward
        # mode never forms the product and was correct either way, which is why
        # this presented as a mode asymmetry.
        #
        # Normalized to unit L1, not to unit peak: ``sum(g * arr)`` is bounded by
        # ``max|g| * sum|arr|``, so unit L1 caps it at the incoming cotangent
        # itself, while unit peak leaves a factor of n_wave (~3e3, 3.5 decades)
        # on top. That factor is the difference between working and overflowing
        # at the top of the declared ``agn_log_lbol`` prior: at log L_bol = 12 the
        # AGN reference offset hands back ~10**35.6, and 3e3 * 10**35.6 is past
        # float32's 3.4e38 ceiling. Both divisors are stop_gradient constants, so
        # they cancel analytically and leave the single correct derivative path.
        _over_peak = l_nu_intrinsic / _peak
        _norm = jax.lax.stop_gradient(jnp.sum(jnp.abs(_over_peak)))
        _norm = jnp.where(_norm > 0.0, _norm, 1.0)
        _l_hat = _over_peak / _norm
        # The bolometric of ``_l_hat`` is the closed-form total over the same two constants it
        # was divided by (kept in log space: ~1e43 erg/s is past float32).
        # Closed-form bolometric of ``_l_hat``, in its own units (peak * norm): grid-independent,
        # and every factor stays O(1)..O(1e12), so neither the forward nor the reverse-mode
        # products leave float32 range (#1439). ``representable_floor``, not the bare 1e-100
        # (#1492): float32's smallest subnormal is 1.4e-45, so the literal IS 0.0 there.
        _log_hat_total = jnp.log10(
            jnp.maximum(
                _blended_bol(_peak * _norm, _COS_INC_ISOTROPIC_REFERENCE),
                _representable_floor(1e-100),
            )
        )
        # The result's own peak, so it is in range whenever the output is.
        _scale_hat = _pow10(_log_l_bol_req - _log_hat_total)
        # optimization_barrier: without it XLA is free to re-associate
        # ``(l/p) * s`` back into ``l * (s/p)``, which reinstates the ~1e64
        # inner product this factorization exists to avoid (the same reason the
        # stellar component's barriers are load-bearing, #1436).
        return jax.lax.optimization_barrier(_l_hat) * _scale_hat

    l_bol_requested = 10.0**agn_log_lbol * _LSUN_ERG * agn_lum_ratio
    l_nu_total = _blended_bol(1.0, _COS_INC_ISOTROPIC_REFERENCE)
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
_CORONA_NU_GRID = host_array(np.geomspace(2.418e13, 2.418e21, 2000))


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
    independent of the caller's wavelength grid (the corona's optical contribution
    does not vary with grid extent).

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
    shape : ndarray, shape (n_wave,)
        The Comptonization shape relative to its peak on the internal grid, so
        ``max <= ~1`` [dimensionless].
    log10_amplitude : float
        ``log10(l_hot_erg / integral)`` [dex], so that
        :math:`L_\nu` [erg s^-1 Hz^-1] ``= shape * 10**log10_amplitude``. The product is
        never formed here: its factors sit at opposite ends of the float32 range, and the
        caller applies the exponent once, to the normalized shape, alongside its own
        normalization (#2767).

    Notes
    -----
    **JIT-compatible**: yes, pure ``jnp`` primitives, no Python branching on
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

    def _comp_log_shape(freq):
        """Log of the band-limited Comptonization shape: rollover x power law x cutoff.

        Kept in log space and exponentiated only after a shift by the grid peak
        (see below). ``exp(-x_hi)`` and ``exp(-x_lo)`` each underflow to 0 in float32
        before their product could be formed, so the product is never formed that way.
        """
        freq_safe = jnp.maximum(freq, 1e-30)
        x_hi = jnp.clip(  # electron-temperature cutoff
            _H_PLANCK * freq / kt_safe, 0.0, representable_exponent(500.0, base=math.e)
        )
        x_lo = jnp.clip(  # seed-photon rollover
            nu_seed_hz / freq_safe, 0.0, representable_exponent(700.0, base=math.e)
        )
        return (1.0 - gamma_hard) * jnp.log(freq_safe) - x_hi - x_lo

    grid_nu = device_table(_CORONA_NU_GRID)
    # Normalize on the fixed internal grid (grid-independent). The shape and the
    # integral are both taken relative to the grid peak in log space: shape/integral is
    # unchanged by the shift, the integral is O(1) per node, and the shift's own
    # derivative is zero, so it is stopped. Without the shift the reverse pass
    # multiplies cotangents by 1/peak, which overflows float32 (#2767).
    log_norm = _comp_log_shape(grid_nu)
    log_peak = jax.lax.stop_gradient(jnp.max(log_norm))
    integral = jnp.trapezoid(jnp.exp(log_norm - log_peak), grid_nu)
    integral_safe = jnp.maximum(jnp.abs(integral), _representable_denominator(1e-100))

    shape_scaled = jnp.exp(_comp_log_shape(nu) - log_peak)
    log10_amplitude = jnp.log10(jnp.maximum(l_hot_erg, _representable_floor(1e-100))) - jnp.log10(
        integral_safe
    )
    return shape_scaled, log10_amplitude


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
        (r_g, r_isco_rg, r_isco_cm, eta, log10_l_edd, mdot) where:

        - r_g : Gravitational radius [cm]
        - r_isco_rg : ISCO radius in units of r_g [dimensionless]
        - r_isco_cm : ISCO radius [cm]
        - eta : Radiative efficiency (Novikov-Thorne) [dimensionless, 0–0.42]
        - log10_l_edd : log10 Eddington luminosity [log10(erg s^-1)]
        - mdot : Mass accretion rate [g s^-1]

    Notes
    -----
    **JIT-compatible**: yes, uses only ``jnp`` primitives.

    The radiative efficiency follows the Novikov-Thorne formula for thin discs
    around Kerr black holes (Novikov & Thorne 1973, Kerr metric).

    ``log10_l_edd`` is returned in log space, not as the linear Eddington
    luminosity (#2210): the linear form is ~1.26e44 erg/s at the bottom of the
    declared ``agn_log_mbh`` prior, past float32's 3.403e38 ceiling everywhere
    in that prior, whether or not ``float32`` is set here -- the caller decides
    what to form from it.

    References
    ----------
    .. [1] D. N. Page and K. S. Thorne, "Disk-Accretion onto a Black Hole.
       Time-Averaged Structure of Accretion Disk," ApJ, 191, 499 (1974).
    """
    r_g = _gravitational_radius(agn_log_mbh)
    r_isco_rg = _isco_radius(agn_a_spin)
    r_isco_cm = r_isco_rg * r_g

    eta = 1.0 - jnp.sqrt(1.0 - 2.0 / (3.0 * r_isco_rg))

    log10_l_edd = _log10_eddington_luminosity(agn_log_mbh)
    # E fix (#846): derive the accretion rate from the requested L_bol
    # (agn_log_lbol) instead of the now-derived Eddington ratio, so the zone
    # structure (T_in, radii) is self-consistent with L_bol. lambda_Edd is
    # recovered downstream as L_bol / L_Edd (see _compute_zone_radii).
    if float32:
        # Float32 (#1206): the ~1e44 erg/s l_bol_erg intermediate overflows;
        # mdot ~1e24 g/s is representable. Fold L_sun/c^2 (a pre-divided
        # constant ~4e12) so only ``10**log_lbol`` (~1e11) is materialized.
        mdot = _LSUN_OVER_C2 * 10.0**agn_log_lbol / eta
    else:
        l_bol_erg = 10.0**agn_log_lbol * _LSUN_ERG
        mdot = l_bol_erg / (eta * _C_LIGHT**2)

    return r_g, r_isco_rg, r_isco_cm, eta, log10_l_edd, mdot


def _hot_flow_luminosity(
    agn_f_hard: float,
    log10_l_edd: float,
    l0: float,
    a_spin: float = 0.0,
    float32: bool = False,
    agn_log_mbh: float = DEFAULT_AGN_LOG_MBH,
    x_hot_max: float | None = None,
) -> float:
    """Hot-flow dissipation ``L_hot = f_hard L_Edd``, limited by what the disc can supply (#2572).

    K&D 2018 (MNRAS 480, 1247) Sec 4.3: "we fix Ldiss,hot = 0.02 LEdd, which defines
    rhot" (Eq. 2: ``Ldiss,hot = 2 int_{R_isco}^{R_hot} sigma T_NT^4 2 pi R dR``). The
    public QSOSED/RELQSO source (``relqso.f``, Hagen & Done) does the same: the
    "disc truncation radius (Rhot) is set by the requirement that the dissipated
    luminosity in the corona Lx_diss=0.02Ledd", and when the integral never reaches it
    ("WARNING!!! Ldiss never reaches 0.02Ledd => No upper limit for r_hot") the whole
    flow is hot, ``rh = rout``, with no disc or warm zone. There is **no** ``L_bol / 2``
    cap in either, so the corona radiates the full ``f_hard L_Edd`` (a ``min(f_hard L_Edd,
    L_bol / 2)`` cap would weaken it by 21% at ``log lambda_Edd = -1.5``).

    tengri cannot drop zones (static shapes), so the unreachable case is represented by
    saturating: ``L_hot`` is limited to what the annuli inside the largest admissible
    ``R_hot`` dissipate, ``L0 h(x_hot_max)`` with ``x_hot_max = 0.5 R_out / R_isco`` (the zone
    clip of :func:`_compute_zone_radii`), and never above ``L0 h_ceiling(a)`` (99% of the
    disc's total dissipation ``L0 h(inf)``, the bisection's own ceiling). The corona then
    radiates exactly the power the hot flow dissipates inside ``R_hot``:
    ``L_diss,hot(R_hot) = L_hot`` holds by construction, saturated or not. Where the disc can
    supply ``f_hard L_Edd`` -- the paper's grid is ``mdot = 0.03 - 1`` -- ``L_hot = f_hard
    L_Edd`` exactly.

    Parameters
    ----------
    x_hot_max : float, optional
        ``0.5 R_out / R_isco``, the largest ``R_hot / R_isco`` the zone clip allows
        [dimensionless]. Omitted, only the bisection's ceiling applies.

    Both ``R_hot`` (the zone radii) and the corona normalization (the SED) use THIS
    value. ``l0`` is ``4 pi R_isco^2 sigma T_in^4`` (:func:`_nt_l0`).

    Returns erg/s, or L_sun on the float32 path (#1206: ~1e44 erg/s overflows).
    """
    f_hard_safe = jnp.clip(agn_f_hard, 1e-6, 0.5)
    if float32:
        l_hot = f_hard_safe * (_L_EDD_1MSUN_LSUN * 10.0**agn_log_mbh)
    else:
        l_hot = f_hard_safe * _pow10(log10_l_edd)
    ceiling = l0 * _nt_h_ceiling(a_spin)
    if x_hot_max is not None:
        ceiling = jnp.minimum(ceiling, l0 * _nt_h(jnp.log(x_hot_max), a_spin))
    return jnp.minimum(l_hot, ceiling)


def _hot_zone_x_max(r_isco_rg: float, r_sg_rg: float) -> float:
    """Largest admissible ``R_hot / R_isco``: the zone clip ``0.5 R_out`` in units of ``R_isco``.

    ``R_out = max(r_sg, 10 R_isco)`` (both in ``R_g``), so the ratio needs no physical constants.
    """
    return 0.5 * jnp.maximum(r_sg_rg, r_isco_rg * 10.0) / r_isco_rg


def _compute_zone_radii(
    r_g: float,
    r_isco_rg: float,
    r_isco_cm: float,
    t_in: float,
    agn_log_mbh: float,
    agn_log_lbol: float,
    agn_f_hard: float,
    agn_r_warm_ratio: float,
    log10_l_edd: float,
    float32: bool = False,
    agn_a_spin: float = 0.0,
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
    log10_l_edd : float
        log10 Eddington luminosity. [log10(erg s^-1)]

    Returns
    -------
    tuple
        (r_hot_cm, r_warm_cm, r_out_cm) zone radii in [cm].

    Notes
    -----
    **JIT-compatible**: yes, uses ``jax.lax.scan`` for JAX-compatible bisection.

    **Self-consistent R_hot**: Uses bisection on the Page-Thorne (relativistic
    Novikov-Thorne) dissipation integral (40 iterations, exact to ~1e-12) to solve
    L_diss,hot(R_hot) = f_hard × L_Edd (K&D 2018 Eq. 2), spin included.

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
    # lambda_Edd = L_bol / L_Edd (E fix, #846), from the requested agn_log_lbol rather than the
    # now-derived agn_log_ledd. It is one ``pow10`` of a difference of logarithms in both
    # precisions (#2210): neither L_Edd (~1e46 erg/s) nor L_bol is materialized, and in reverse
    # mode the cotangent is multiplied by the ratio itself, not by ``L_bol`` before a division by
    # ``L_Edd`` (the former form overflows float32 for a cotangent of ~1e32, #2767).
    l_edd_ratio = jnp.clip(_pow10(agn_log_lbol + _LOG10_LSUN_ERG - log10_l_edd), 1e-10, 1.0)
    r_sg_rg = _self_gravity_radius(agn_log_mbh, l_edd_ratio)
    r_out_cm = jnp.maximum(r_sg_rg, r_isco_rg * 10.0) * r_g

    # R_hot is solved from the SAME L_hot the corona radiates (#2572), limited to what the
    # annuli inside the zone clip (0.5 R_out) dissipate.
    l_hot_target = _hot_flow_luminosity(
        agn_f_hard,
        log10_l_edd,
        _nt_l0(r_isco_cm, t_in, float32),
        agn_a_spin,
        float32=float32,
        agn_log_mbh=agn_log_mbh,
        x_hot_max=_hot_zone_x_max(r_isco_rg, r_sg_rg),
    )
    if float32:
        r_hot_cm = _r_hot_bisect(r_isco_cm, t_in, l_hot_target, float32=True, a_spin=agn_a_spin)
    else:
        r_hot_cm = _r_hot_bisect(r_isco_cm, t_in, l_hot_target, a_spin=agn_a_spin)

    r_warm_ratio_safe = jnp.clip(agn_r_warm_ratio, 1.1, 10.0)
    r_warm_cm = r_hot_cm * r_warm_ratio_safe

    r_hot_cm = jnp.clip(r_hot_cm, r_isco_cm * 1.01, r_out_cm * 0.5)
    r_warm_cm = jnp.clip(r_warm_cm, r_hot_cm * 1.01, r_out_cm * 0.9)

    return r_hot_cm, r_warm_cm, r_out_cm


def _ring_area_relative(r_cm, dr_cm, cos_inc, r_ref_cm, dr_ref_cm):
    """Ring area in units of ``ring_area(r_ref, dr_ref, 1)``, from O(1) ratios [dimensionless].

    ``ring_area`` is bilinear in ``r`` and ``dr``, so ``ring_area(r, dr, c) = ring_area(r_ref,
    dr_ref, 1) * ring_area(r/r_ref, dr/dr_ref, c) / ring_area(1, 1, 1)``. The ~6e31 cm^2 areas, and
    the cotangents that would multiply them in reverse mode, never form in float32 (#2767).
    """
    return _ring_area(r_cm / r_ref_cm, dr_cm / dr_ref_cm, cos_inc) / _ring_area(1.0, 1.0, 1.0)


def _log10_ring_area_unit(r_ref_cm, dr_ref_cm):
    """``log10(ring_area(r_ref, dr_ref, 1))`` [dex], the unit of :func:`_ring_area_relative`."""
    return math.log10(_ring_area(1.0, 1.0, 1.0)) + jnp.log10(r_ref_cm) + jnp.log10(dr_ref_cm)


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
    log10_l_edd: float,
    l_bol_erg: float,
    agn_self_consistent_gamma: bool,
    float32: bool = False,
    agn_log_mbh: float = DEFAULT_AGN_LOG_MBH,
    agn_log_lbol_shape: float = 0.0,
    agn_a_spin: float = 0.0,
    nthcomp_table=None,
) -> tuple:
    """Compute self-consistent luminosities of the three AGN zones, disc and corona apart.

    Integrates the Novikov-Thorne temperature profile over annuli in each zone
    (outer standard disc, warm Comptonization, hot corona) and combines the
    spectral shapes (blackbody, Comptonized, power-law) into a total L_ν. Applies
    a global normalization to conserve the input accretion power.

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
        Cosine of inclination angle [dimensionless, 0–1].
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
    log10_l_edd : float
        log10 Eddington luminosity. [log10(erg s^-1)]
    l_bol_erg : float
        Requested bolometric luminosity [erg s^-1].
    agn_self_consistent_gamma : bool
        If True, derive gamma_hard self-consistently from Beloborodov (1999).

    Returns
    -------
    tuple
        (l_nu_total, scale, l_nu_disc, l_nu_hot, corona_fraction, log10_scale) where:

        - l_nu_total : Total L_ν(i) = 2 cos i D_ν + H_ν, already scaled to ``l_bol_erg``
          [erg s^-1 Hz^-1]
        - scale : Normalization factor with ``scale * (D + H) = l_bol_erg``, the accretion
          power; independent of ``agn_cos_inc``. Informative only: the spectra are scaled
          through ``log10_scale``, never by this linear factor [dimensionless]
        - l_nu_disc : Disc and warm-zone part of the line-of-sight spectrum, ``2 cos i D_ν``,
          scaled [erg s^-1 Hz^-1]
        - l_nu_hot : Corona ``H_ν``, isotropic, scaled [erg s^-1 Hz^-1]
        - corona_fraction : ``H / (D + H)``, the share of the accretion power the corona
          carries, in closed form from the radial integration (no spectral grid)
          [dimensionless]
        - log10_scale : ``log10(scale)``, formed as a difference of logarithms so the
          normalization can be applied to an unnormalized spectrum without leaving
          the float32 range in either autodiff mode [dex]

    Notes
    -----
    **JIT-compatible**: yes, uses ``jax.vmap`` for radial integration.

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
    rt_outer = jnp.maximum(_nt_rt(r_ratio_outer, agn_a_spin), 1e-30) ** 0.25
    t_outer = t_in * r_ratio_outer ** (-0.75) * rt_outer

    d_log_r_outer = log_r_outer[1] - log_r_outer[0]
    dr_outer = r_outer * jnp.log(10.0) * d_log_r_outer

    # The Planck prefactor 2 h nu (nu/c)^2 reaches ~4e11 at 0.1 A and a ring area reaches
    # ~6e31 cm^2, so their product (~2e43) leaves float32 even though ``exp(-h nu / k T)``
    # drives the product's true value to zero there; XLA forms ``(prefactor * area) * exp(-x)``
    # and gets ``inf * 0 = NaN`` (#2767). The ring areas are therefore carried relative to
    # their largest value, a pure factorization constant held under ``stop_gradient``, and
    # that scale is applied once, in log10, to the ring sum.
    r_outer_ref = jax.lax.stop_gradient(jnp.max(r_outer))
    dr_outer_ref = jax.lax.stop_gradient(jnp.max(dr_outer))
    log10_ring_area_ref = _log10_ring_area_unit(r_outer_ref, dr_outer_ref)

    def _outer_ring(r_cm, t_ring, dr_ring):
        """Blackbody L_nu of one outer-disk annulus, in units of the largest ring area."""
        b_nu = _planck_lnu(nu, t_ring)
        return b_nu * _ring_area_relative(r_cm, dr_ring, agn_cos_inc, r_outer_ref, dr_outer_ref)

    l_nu_outer = apply_log10_scale(
        jnp.sum(jax.vmap(_outer_ring)(r_outer, t_outer, dr_outer), axis=0), log10_ring_area_ref
    )

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
    rt_warm = jnp.maximum(_nt_rt(r_ratio_warm, agn_a_spin), 1e-30) ** 0.25
    t_warm = t_in * r_ratio_warm ** (-0.75) * rt_warm

    d_log_r_warm = log_r_warm_grid[1] - log_r_warm_grid[0]
    dr_warm = r_warm_grid * jnp.log(10.0) * d_log_r_warm

    # Ring areas are carried relative to the largest one, as in the outer zone, and the scale is
    # applied once to the ring sum (#2767): ``p_plain * area`` is ~1e47 and the template shape is
    # ~1e-17 per Hz, so neither product nor cotangent is formed at its own magnitude.
    r_warm_ref = jax.lax.stop_gradient(jnp.max(r_warm_grid))
    dr_warm_ref = jax.lax.stop_gradient(jnp.max(dr_warm))
    log10_warm_area_ref = _log10_ring_area_unit(r_warm_ref, dr_warm_ref)

    nu_norm = device_table(_nthcomp_norm_grid())

    def _warm_ring(r_cm, t_ring, dr_ring):
        """Comptonized L_nu of one warm-zone annulus, in units of the largest ring area."""
        # Ring blackbody power per unit area: int B_nu dnu = (sigma_Planck/pi) T^4, closed form.
        # (It was trapz(B_nu, nu) on the CALLER's grid, which moved the SED by 4e-3 between
        # grids of different extent and density: #2572.)
        p_plain = _BNU_BOL_PER_T4 * t_ring**4
        kTbb_keV = _K_BOLTZ_KEV * t_ring
        shape = _nthcomp_lnu_interp(
            nu, agn_gamma_warm, agn_kt_warm, kTbb_keV, _template=nthcomp_table, unit=_NTHCOMP_UNIT
        )
        # The template is normalized by its own integral over its band, so the ring carries
        # exactly ``p_plain`` (the template's own normalization is ~0.5 % off, #2733).
        shape_band = _nthcomp_lnu_interp(
            nu_norm,
            agn_gamma_warm,
            agn_kt_warm,
            kTbb_keV,
            _template=nthcomp_table,
            unit=_NTHCOMP_UNIT,
        )
        # The ring sum is multiplied by ``_NTHCOMP_UNIT`` at the end, so the shape carries 1/unit.
        shape = shape / (jnp.trapezoid(shape_band, nu_norm) * _NTHCOMP_UNIT)
        # The normalized ``shape`` (per ``_NTHCOMP_UNIT``) is folded in before the area, so the
        # ring's bolometric power is never formed on its own.
        return (shape * p_plain) * _ring_area_relative(
            r_cm, dr_ring, agn_cos_inc, r_warm_ref, dr_warm_ref
        )

    l_nu_warm = apply_log10_scale(
        jnp.sum(jax.vmap(_warm_ring)(r_warm_grid, t_warm, dr_warm), axis=0),
        log10_warm_area_ref + _LOG10_NTHCOMP_UNIT,
    )

    # ── Zone 3: Hot corona (R_ISCO < r < R_hot) ───────────────────
    # Float32 (#1206): l_hot_erg ~5e43 and l_seed ~1e44 erg/s overflow. Work both
    # in L_sun units (l_edd from M_BH, L_bol from the SHAPE luminosity: the
    # corona fraction lambda_Edd must track the TRUE L_bol, not the reference the
    # magnitude normalizes to). Beloborodov uses only their ratio, so units cancel.
    l_hot_erg = _hot_flow_luminosity(
        agn_f_hard,
        log10_l_edd,
        _nt_l0(r_isco_cm, t_in, float32),
        agn_a_spin,
        float32=float32,
        agn_log_mbh=agn_log_mbh,
        x_hot_max=0.5 * r_out_cm / r_isco_cm,
    )
    if float32:
        l_seed_geom = _l_seed_geometric(
            r_isco_cm, r_hot_cm, r_out_cm, t_in, float32=True, a_spin=agn_a_spin
        )
    else:
        l_seed_geom = _l_seed_geometric(r_isco_cm, r_hot_cm, r_out_cm, t_in, a_spin=agn_a_spin)

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
    # ``k/h`` is folded first (2.1e10 Hz/K): the reverse pass divides a cotangent by this factor
    # and ``1/h`` alone (1.5e26) takes a ~4e12 cotangent past the float32 maximum (#2767).
    nu_seed_hot = (_K_BOLTZ / _H_PLANCK) * t_seed_hot

    # The corona L_nu scales linearly with l_hot_erg. On the float32 path
    # l_hot_erg is in L_sun, so the corona comes out in L_sun/Hz; its exponent gains
    # log10(L_sun) to give erg/s/Hz, and it is applied only at the end, to the normalized shape.
    hot_shape, log10_hot_amplitude = _hot_corona_lnu(
        nu, l_hot_erg, gamma_hard_eff, kt_hot_erg, nu_seed_hot
    )
    if float32:
        log10_hot_amplitude = log10_hot_amplitude + _LOG10_LSUN_ERG

    # ── Normalize ─────────────────────────────────────────────────
    l_nu_disc = l_nu_outer + l_nu_warm

    # Zone bolometric integrals (sigma T^4 * 2*pi*r*dr) reach ~1e44 erg/s and
    # overflow float32; work them in L_sun (pre-divided 2*pi*sigma/L_sun folds
    # first). l_hot_erg is already L_sun on this path, and ``l_bol_erg`` is passed
    # in L_sun too (the reference normalization), so ``scale`` is a clean ratio.
    # The accretion power of the disc and warm zones, D, is both faces of every annulus
    # (``_TWO_FACES``); it carries no cos i, so ``scale`` does not depend on the inclination.
    #
    # The sum is carried as a log10 exponent: every ring term ``c T^4 r dr`` is formed as a sum of
    # logarithms and the rings are combined in a base-10 logsumexp, so no product of the large
    # (``T^4``, ``r``, ``dr``) and small (``c``) factors is ever formed, and the reverse pass
    # carries O(1) weights instead of the ~1e28-scale unnormalized spectra (#2767).
    log10_c = math.log10(_2PI_SIGMA_SB_OVER_LSUN if float32 else 2.0 * math.pi * _SIGMA_SB)
    r_both = jnp.concatenate([r_outer, r_warm_grid])
    t_both = jnp.concatenate([t_outer, t_warm])
    dr_both = jnp.concatenate([dr_outer, dr_warm])
    log10_rings = (
        log10_c
        + 4.0 * jnp.log10(jnp.maximum(t_both, _representable_floor(1e-30)))
        + jnp.log10(r_both)
        + jnp.log10(dr_both)
    )
    log10_disc_power = math.log10(_TWO_FACES) + log10_weighted_sum(log10_rings, 1.0)
    log10_hot_power = jnp.log10(jnp.maximum(l_hot_erg, _representable_floor(1e-100)))
    log10_unnorm = jnp.maximum(
        log10_add(log10_disc_power, log10_hot_power),
        jnp.log10(_representable_denominator(1e-100)),
    )
    log10_scale = jnp.log10(l_bol_erg) - log10_unnorm
    scale = _pow10(log10_scale)

    # Applied once, to spectra whose own peak is factored out, so no unnormalized ~1e28 spectrum
    # (or the cotangent that would multiply it in reverse mode) is formed in float32 (#2767).
    l_nu_disc_norm = apply_log10_scale(l_nu_disc, log10_scale)
    l_nu_hot_norm = apply_log10_scale(hot_shape, log10_hot_amplitude + log10_scale)

    return (
        l_nu_disc_norm + l_nu_hot_norm,
        scale,
        l_nu_disc_norm,
        l_nu_hot_norm,
        _pow10(log10_hot_power - log10_unnorm),
        log10_scale,
    )


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
    **JIT-compatible**: yes, uses only ``jnp`` primitives.

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
    ratio = jnp.clip(
        l_diss_hot / jnp.maximum(l_seed, _representable_denominator(1e-30)), 1e-3, 1e3
    )
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
    **JIT-compatible**: yes, uses ``jnp`` primitives.

    The 2500 Å point is a canonical AGN diagnostic wavelength, used to
    compute the optical-to-X-ray spectral index (alpha_ox) and as a
    benchmark for intrinsic AGN continuum comparisons. Linear interpolation
    in wavelength space is sufficient for the optical/UV continuum variability
    timescales. The wavelength array need not be pre-sorted; this function
    sorts internally before interpolation.
    """
    sort_idx = jnp.argsort(wavelength)
    return jnp.interp(2500.0, wavelength[sort_idx], l_nu[sort_idx])


def kubota_done_disc_split(
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
    _template=None,
    **_kwargs,
) -> tuple:
    r"""Kubota & Done (2018) disc and corona, returned apart (for the SKIRTOR tie).

    The same model as :func:`kubota_done_disc` (see there for the physics and the
    parameters), with the two parts of its line-of-sight spectrum
    :math:`L_\nu(i) = 2\cos i\, D_\nu + H_\nu` separated: the optically thick disc and warm
    Comptonization zones :math:`2\cos i\, D_\nu` and the isotropic corona :math:`H_\nu`, and
    the corona's share of the accretion power in closed form.

    Returns
    -------
    l_nu_total : ndarray, shape (n_wave,)
        :math:`L_\nu(i)`, identical to :func:`kubota_done_disc` [erg/s/Hz].
    l_nu_disc : ndarray, shape (n_wave,)
        The disc and warm-zone part :math:`2\cos i\, D_\nu` [erg/s/Hz].
    l_nu_hot : ndarray, shape (n_wave,)
        The corona :math:`H_\nu` [erg/s/Hz].
    corona_fraction : float
        :math:`P_H / (P_D + P_H)`, with :math:`P_H = \int H_\nu\,d\nu` and
        :math:`P_D = \int D_\nu\,d\nu` the angle-integrated powers of the corona and of the
        two-face disc. It follows from the radial dissipation integrals (the Page-Thorne
        profile for :math:`D`, the hot-flow dissipation :math:`f_{\rm hard} L_{\rm Edd}` for
        :math:`H`) and is independent of the wavelength grid and of ``agn_cos_inc``
        [dimensionless].

    Notes
    -----
    **JIT-compatible**: yes. ``l_nu_disc + l_nu_hot`` equals ``l_nu_total`` exactly.
    """
    return kubota_done_disc(
        wavelength,
        agn_log_lbol=agn_log_lbol,
        agn_lum_ratio=agn_lum_ratio,
        agn_log_mbh=agn_log_mbh,
        agn_log_ledd=agn_log_ledd,
        agn_a_spin=agn_a_spin,
        agn_cos_inc=agn_cos_inc,
        agn_f_hard=agn_f_hard,
        agn_gamma_warm=agn_gamma_warm,
        agn_kt_warm=agn_kt_warm,
        agn_gamma_hard=agn_gamma_hard,
        agn_kt_hot=agn_kt_hot,
        agn_r_warm_ratio=agn_r_warm_ratio,
        n_radii=n_radii,
        agn_self_consistent_gamma=agn_self_consistent_gamma,
        agn_log_lbol_shape=agn_log_lbol_shape,
        _template=_template,
        _return_parts=True,
    )


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
    _template=None,
    _return_parts: bool = False,
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
      Page-Thorne dissipation integral (Eq. 2), spin included.
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
        Accretion power of all three zones, integrated over all directions
        (both faces of the disc); independent of the inclination. The
        line-of-sight luminosity is :math:`\\int L_\\nu(i)\\,d\\nu`, a derived
        quantity. [log10(L_sun)]
    agn_lum_ratio : float, optional
        Fraction of the accretion power emitted by the disc system (all zones).
        Default: 1.0. [dimensionless, 0–1]
    agn_log_mbh : float, optional
        Black hole mass. Determines the Eddington luminosity and temperature
        scaling. Default: 8.0. [log10(M_sun)]
    agn_log_ledd : float, optional
        **DEPRECATED / IGNORED (#846).** The Eddington ratio, and hence the
        inner temperature, accretion rate, and zone radii: is now DERIVED from
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
        line of sight. Range: [0, 1]. The disc and warm zones radiate
        :math:`\\propto\\cos i` (Kubota & Done 2018, Sect. 2.1) and the returned
        spectrum is :math:`L_\\nu(i) = 2\\cos i\\,D_\\nu + H_\\nu`, with :math:`D_\\nu` the
        angle-integrated spectrum of the disc and warm zones and :math:`H_\\nu` that of
        the isotropic corona (Sect. 2.2). The zone radii, temperatures and
        :math:`\\dot m` do not depend on it. Default: 0.866 (30°). [dimensionless]
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
    **JIT-compatible**: yes, uses ``jnp`` primitives and ``jax.vmap``.

    **Gradient-safe**: yes, fully differentiable w.r.t. all parameters,
    including the bisection-solved R_hot.

    **Key self-consistent physics**:

    All three zones share the relativistic Page & Thorne (1974) thin-disc temperature
    profile (as K&D 2018 and QSOSED/RELQSO; ``tengri.components.agn._nt_emissivity``):

    .. math::

        T(r) = T_{\\rm in} \\left(\\frac{r}{r_{\\rm ISCO}}\\right)^{-3/4}
               R_t(r; a)^{1/4}

    with :math:`R_t = C/B \\to 0` at the ISCO and :math:`\\to 1` far out (the Newtonian
    zero-torque form :math:`1-\\sqrt{r_{\\rm ISCO}/r}` dissipates 1.46 :math:`L_{\\rm bol}`
    at a=0 instead of 1.02).

    where :math:`T_{\\rm in} = (3 G M M_{\\rm dot} / 8\\pi \\sigma_{\\rm SB}
    r_{\\rm ISCO}^3)^{1/4}`, and the inner temperature increases with accretion
    rate. The radii are ordered as R_ISCO < R_hot < R_warm < R_out.

    **Zone luminosity computation**:

    Each zone is divided into annuli at radii {r_i}, each of which contributes
    L_ν from its local Planck function (outer disc), nthcomp prescription
    (warm zone), or hot-corona power law (inner zone). All zones are summed
    and renormalized so that :math:`\\int (D_\\nu + H_\\nu)\\,d\\nu` equals the accretion power.

    **Inclination and energy** (K&D 2018, Sects. 2.1 and 2.2): the disc and warm zones are
    optically thick and radiate :math:`\\propto\\cos i`; the corona is optically thin and
    isotropic. With :math:`D_\\nu` the angle-integrated (two-face) spectrum of the disc and
    warm zones and :math:`H_\\nu` that of the corona, the model returns

    .. math::

        L_\\nu(i) = 2\\cos i\\, D_\\nu + H_\\nu , \\qquad
        \\int (D_\\nu + H_\\nu)\\,d\\nu = L_{\\rm acc} = 10^{\\mathtt{agn\\_log\\_lbol}} L_\\odot ,

    the luminosity density an observer at inclination :math:`i` assigns assuming isotropy
    (AGNSED's :math:`\\cos i / 0.5`). The mean of :math:`\\int L_\\nu(i)\\,d\\nu` over
    :math:`\\cos i \\in [0, 1]` is :math:`L_{\\rm acc}`, and at :math:`\\cos i = 0.5` the
    line-of-sight power equals it; face-on the disc is twice as luminous. The corona's
    power :math:`H` is the hot-flow dissipation :math:`f_{\\rm hard} L_{\\rm Edd}` (the
    Page-Thorne dissipation inside :math:`R_{\\rm hot}`), and :math:`D` is the dissipation
    of the thin disc outside it, so the split follows K&D's energy budget. The
    shape quantities (:math:`\\dot m`, :math:`T(r)`, :math:`R_{\\rm hot}`,
    :math:`R_{\\rm warm}`) follow from :math:`L_{\\rm acc}` and do not depend on
    :math:`i`.

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

    r_g, r_isco_rg, r_isco_cm, _eta, log10_l_edd, mdot = _compute_bh_params(
        agn_log_mbh, _lbol_shape, agn_a_spin, float32=_f32
    )

    # Reference inner-disc temperature T_in (the profile is T_in x^-3/4 Rt^1/4).
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
        log10_l_edd,
        float32=_f32,
        agn_a_spin=agn_a_spin,
    )

    # Normalization magnitude from agn_log_lbol (the reference on the float32
    # path); pass it in L_sun there so the zone helper's ``scale`` is a clean ratio.
    if _f32:
        l_bol_requested = 10.0**agn_log_lbol * agn_lum_ratio  # L_sun
    else:
        l_bol_requested = 10.0**agn_log_lbol * _LSUN_ERG * agn_lum_ratio

    l_nu_total, _scale, l_nu_disc, l_nu_hot, corona_fraction, _log10_scale = (
        _compute_zone_luminosities(
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
            log10_l_edd,
            l_bol_requested,
            agn_self_consistent_gamma,
            float32=_f32,
            agn_log_mbh=agn_log_mbh,
            agn_log_lbol_shape=_lbol_shape,
            agn_a_spin=agn_a_spin,
            nthcomp_table=_template,
        )
    )

    if _return_parts:
        return l_nu_total, l_nu_disc, l_nu_hot, corona_fraction
    return l_nu_total


# ── Model 4: ADAF; DEPRECATED delegator to the faithful adaf_spectrum ────────


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
    It stores the outer disc alone (no warm zone, no corona) as absolute L_ν
    (erg/s/Hz) at cos_inc = 0.5; the inclination correction is applied
    analytically as 2·cos_inc (valid for the non-relativistic outer disc;
    approximate for the GR inner disc).

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
    **JIT-compatible**: yes, the returned function is pure JAX.
    Grid loading is cached via ``@functools.cache``.

    **Gradient-safe**: yes, triweight interpolation is C²-continuous.

    **Inclination**: grid stored at cos_inc = 0.5, where the line-of-sight power
    equals the angle-integrated power; scaled by 2·cos_inc.
    This is exact for r > 1000 r_g (non-relativistic regime) and approximate
    for the GR inner disc where KYCONV applies full Kerr ray-tracing.

    **Grid axes**: log_mbh ∈ [7, 10], log_mdot ∈ [−1.5, 0.3], astar ∈ [0, 0.998]
    (prograde only; KYCONV rejects retrograde spins).

    References
    ----------
    .. [1] Dovciak, M., Karas, V., & Yaqoob, T. (2004).
       ApJS, 153, 205. doi:10.1086/421115

    .. [2] Hagen, S. & Done, C. (2023).
       MNRAS, 525, 3455-3467. doi:10.1093/mnras/stad2499
    """
    if not __import__("pathlib").Path(grid_path).exists():
        raise FileNotFoundError(f"RELAGN disc grid not found: {grid_path}")

    return functools.partial(relagn_disc_from_grid, _load_relagn_disc_grid(grid_path))


def load_relagn_default_grid() -> dict:
    """Load the packaged RELAGN disc grid (discovery + cache).

    This is the ``template_loader`` the RELAGN disc block registers, so the
    forward model can hoist the ~27 MB library out of the JIT trace and hand
    it to the block as an argument.

    Returns
    -------
    dict
        Template arrays; see :func:`_load_relagn_disc_grid`.

    Raises
    ------
    FileNotFoundError
        If ``data/relagn_disc_grid.h5`` is not present.

    Notes
    -----
    ``_find_relagn_grid`` is imported inside the body on purpose:
    :mod:`tengri.components.agn.unified` imports *this* module, so a
    module-level import would close the cycle.
    """
    from tengri.components.agn.unified import _find_relagn_grid

    return _load_relagn_disc_grid(_find_relagn_grid())


def relagn_disc_from_grid(
    grid: dict,
    wavelength: jnp.ndarray,
    # Kept on this branch (main's #1578 dropped it): the bolometric
    # renormalization below is this branch's #1206 work and reads it. Read off
    # the declaration rather than repeating the literal 11.0, which is #1578's
    # own rule.
    agn_log_lbol: float = DEFAULT_AGN_LOG_LBOL,
    agn_log_mbh: float = DEFAULT_AGN_LOG_MBH,
    agn_log_mdot: float = -1.0,
    agn_astar: float = 0.0,
    agn_cos_inc: float = DEFAULT_AGN_COS_INC,
    **_kwargs,
) -> jnp.ndarray:
    r"""RELAGN outer disc from the relativistic template grid.

    Parameters
    ----------
    grid : dict
        Template arrays from :func:`_load_relagn_disc_grid`. Taken as an
        **argument** rather than closed over, so the forward model can thread
        the ~27 MB library through ``jax.jit`` as a ``Parameter`` instead of
        baking it into the graph as ``Constant`` ops (#1383).
    wavelength : ndarray, shape (n_wave,)
        Rest-frame wavelength. [Å]
    agn_log_lbol : float
        :math:`\log_{10}(L_{\rm acc}/L_\odot)`, the accretion power the template
        is normalized to, integrated over all directions. [dimensionless]
    agn_log_mbh : float
        :math:`\log_{10}(M_{\rm BH}/M_\odot)`. [dimensionless]
    agn_log_mdot : float
        :math:`\log_{10}(\dot M / \dot M_{\rm Edd})`. [dimensionless]
    agn_astar : float
        Dimensionless BH spin, prograde only (0 to 0.998). [dimensionless]
    agn_cos_inc : float
        Cosine of inclination (1 = face-on). [dimensionless]

    Returns
    -------
    ndarray, shape (n_wave,)
        Disc :math:`L_\nu`. [erg/s/Hz]

    Notes
    -----
    **JIT-compatible**: yes.
    **Gradient-safe**: yes, triweight kernel, C²-continuous.

    **Normalization (behavior change, #1206).** The template *shape* comes from
    (M_BH, Ṁ, a\*); its **normalization** is set by ``agn_log_lbol``, matching
    every other disc in the composable menu (``multicolor``, ``kubota_done``,
    ``slone_netzer``, …). Previously the grid's own absolute normalization was
    used and ``agn_log_lbol`` had no effect, which both surprised users who set
    it and made the disc unusable in float32, since the grid's absolute
    ``λL_λ(5100 Å) ≈ 2.6e44`` erg/s exceeds the float32 maximum (3.4e38).

    The normalization integral is taken over the template's own native
    wavelength grid, before resampling, so the output does not depend on the
    caller's wavelength sampling and a grid that stops short of the template
    carries only the part of ``L_bol`` that falls inside it.

    **Inclination**: the grid holds the disc alone at cos_inc = 0.5, where the
    line-of-sight power is the angle-integrated power. The template is normalized
    to ``agn_log_lbol`` at that reference and the spectrum is then scaled by
    :math:`2\cos i`: the returned :math:`L_\nu(i) = 2\cos i\,D_\nu` with
    :math:`\int D_\nu\,d\nu = L_{\rm bol}`, as for ``multicolor_disc``. The mean
    of its power over :math:`\cos i \in [0, 1]` is :math:`L_{\rm bol}`.
    """
    point = (agn_log_mbh, agn_log_mdot, agn_astar)
    lnu_template = _interp_nd_triweight(
        jnp.asarray(grid["grid_jax"]),
        tuple(jnp.asarray(a) for a in grid["axes"]),
        tuple(jnp.asarray(e) for e in grid["edges"]),
        point,
        scatters=grid["scatters"],
        index_space_interp=True,
    )
    # The grid holds the disc alone at the reference cos_inc = 0.5, where the line-of-sight
    # power is the angle-integrated power: the template is normalized AT the reference and the
    # inclination scaling 2 cos i is applied to the spectrum on top.
    wave_native = jnp.asarray(grid["wave_grid"])
    # Interpolate grid wavelength -> observation wavelength
    lnu_interp = resample_template(
        wavelength, wave_native, lnu_template * (2.0 * agn_cos_inc), left=0.0, right=0.0
    )

    # Renormalize to the requested bolometric luminosity (#1206), with the
    # integral taken on the template's native grid (before resampling) so the
    # disc does not depend on the caller's wavelength sampling or range. The
    # float32 peak-factoring (the template's integral is ~1e45 erg/s and would
    # overflow) lives in the helper.
    return scale_to_lbol_native(
        lnu_template, wave_native, lnu_interp, 10.0**agn_log_lbol * _LSUN_ERG
    )
