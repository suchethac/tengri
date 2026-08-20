# SPDX-License-Identifier: BSD-3-Clause
"""Intergalactic medium absorption (Inoue et al. 2014).

Computes the mean IGM transmission T_IGM(lambda_obs, z_source) accounting for:

- Lyman-series line absorption from the Ly-alpha forest (LAF)
- Lyman-series line absorption from Damped Ly-alpha systems (DLA)
- Lyman-continuum absorption from the LAF
- Lyman-continuum absorption from DLA systems

All functions are pure JAX (JIT-compilable, differentiable).

Reference: Inoue, A. K., Shimizu, I., Iwata, I., & Tanaka, M. 2014,
           MNRAS, 442, 1805

Coefficient tables from eazy-py (Brammer et al.):
    https://github.com/gbrammer/eazy-py/blob/master/eazy/data/
"""

import jax
import jax.numpy as jnp

from tengri.components.igm._params import DEFAULT_DLA_LOG_N_HI
from tengri.components.igm.dla import _A_LYA, _F_LYA, _NU_LYA, _WL_LYA
from tengri.cosmology import PLANCK18
from tengri.utils.physics_constants import C_CGS

# ── Lyman series wavelengths (Angstrom) for lines j=2 (Ly-alpha) to j=40
_N_LINES = 39

# Rest-frame wavelengths of Lyman series lines (Angstrom), vacuum.
# Values match Inoue+2014 Table 2 / eazy-py LAFcoeff.txt exactly. The first
# line is vacuum Lyman-alpha = 1215.67 Å (NOT 1216.0 — a rounded value put the
# forest edge ~0.33 Å rest / ~2.6 Å observed at z=7 redward of every other
# code; see tests/regression/paper/test_igm_inoue.py).
_LAMBDA_LYMAN = jnp.array(
    [
        1215.67,
        1025.720,
        972.537,
        949.743,
        937.803,
        930.748,
        926.226,
        923.150,
        920.963,
        919.352,
        918.129,
        917.181,
        916.429,
        915.824,
        915.329,
        914.919,
        914.576,
        914.286,
        914.039,
        913.826,
        913.641,
        913.480,
        913.339,
        913.215,
        913.104,
        913.006,
        912.918,
        912.839,
        912.768,
        912.703,
        912.645,
        912.592,
        912.543,
        912.499,
        912.458,
        912.420,
        912.385,
        912.353,
        912.324,
    ]
)

# Lyman limit wavelength (Inoue et al. 2014 uses 911.8 Å; matches eazy-py).
_LAMBDA_LIMIT = 911.8  # Angstrom

# ── LAF coefficients: A_j^LAF for 3 regimes (Inoue+2014 Eq. 21) ───
# Shape: (39, 3) — [A_j1, A_j2, A_j3]
# From eazy-py LAFcoeff.txt
_A_LAF = jnp.array(
    [
        [1.690e-02, 2.354e-03, 1.026e-04],
        [4.692e-03, 6.536e-04, 2.849e-05],
        [2.239e-03, 3.119e-04, 1.360e-05],
        [1.319e-03, 1.837e-04, 8.010e-06],
        [8.707e-04, 1.213e-04, 5.287e-06],
        [6.178e-04, 8.606e-05, 3.752e-06],
        [4.609e-04, 6.421e-05, 2.799e-06],
        [3.569e-04, 4.971e-05, 2.167e-06],
        [2.843e-04, 3.960e-05, 1.726e-06],
        [2.318e-04, 3.229e-05, 1.408e-06],
        [1.923e-04, 2.679e-05, 1.168e-06],
        [1.622e-04, 2.261e-05, 9.854e-07],
        [1.385e-04, 1.930e-05, 8.414e-07],
        [1.196e-04, 1.666e-05, 7.264e-07],
        [1.043e-04, 1.453e-05, 6.334e-07],
        [9.174e-05, 1.278e-05, 5.571e-07],
        [8.128e-05, 1.132e-05, 4.936e-07],
        [7.251e-05, 1.010e-05, 4.403e-07],
        [6.505e-05, 9.062e-06, 3.950e-07],
        [5.868e-05, 8.174e-06, 3.563e-07],
        [5.319e-05, 7.409e-06, 3.230e-07],
        [4.843e-05, 6.746e-06, 2.941e-07],
        [4.427e-05, 6.167e-06, 2.689e-07],
        [4.063e-05, 5.660e-06, 2.467e-07],
        [3.738e-05, 5.207e-06, 2.270e-07],
        [3.454e-05, 4.811e-06, 2.097e-07],
        [3.199e-05, 4.456e-06, 1.943e-07],
        [2.971e-05, 4.139e-06, 1.804e-07],
        [2.766e-05, 3.853e-06, 1.680e-07],
        [2.582e-05, 3.596e-06, 1.568e-07],
        [2.415e-05, 3.364e-06, 1.466e-07],
        [2.263e-05, 3.153e-06, 1.374e-07],
        [2.126e-05, 2.961e-06, 1.291e-07],
        [2.000e-05, 2.786e-06, 1.214e-07],
        [1.885e-05, 2.627e-06, 1.145e-07],
        [1.780e-05, 2.479e-06, 1.080e-07],
        [1.682e-05, 2.343e-06, 1.021e-07],
        [1.593e-05, 2.219e-06, 9.673e-08],
        [1.510e-05, 2.103e-06, 9.169e-08],
    ]
)

# ── DLA coefficients: A_j^DLA for 2 regimes (Inoue+2014 Eq. 22) ───
# Shape: (39, 2) — [A_j^DLA1 (power 2, lambda_obs < 3 lambda_j),
#                   A_j^DLA2 (power 3, lambda_obs >= 3 lambda_j)].
# Values are Inoue+2014 (MNRAS 442, 1805) Table 2 columns A_DLA_J_1 /
# A_DLA_J_2 verbatim (the same table eazy-py, BAGPIPES and Synthesizer
# ship). A prior transcription stored the second column ~2.87-3.65x too
# high (a near-copy of the first) and the first column ~1.1-1.27x too high
# in the line tail, over-absorbing the z >= 2 Lyman continuum. Regression:
# tests/components/igm/test_inoue14_dla_coefficients.py.
_A_DLA = jnp.array(
    [
        [1.617e-04, 5.390e-05],
        [1.545e-04, 5.151e-05],
        [1.498e-04, 4.992e-05],
        [1.460e-04, 4.868e-05],
        [1.429e-04, 4.763e-05],
        [1.402e-04, 4.672e-05],
        [1.377e-04, 4.590e-05],
        [1.355e-04, 4.516e-05],
        [1.335e-04, 4.448e-05],
        [1.316e-04, 4.385e-05],
        [1.298e-04, 4.326e-05],
        [1.281e-04, 4.271e-05],
        [1.265e-04, 4.218e-05],
        [1.250e-04, 4.168e-05],
        [1.236e-04, 4.120e-05],
        [1.222e-04, 4.075e-05],
        [1.209e-04, 4.031e-05],
        [1.197e-04, 3.989e-05],
        [1.185e-04, 3.949e-05],
        [1.173e-04, 3.910e-05],
        [1.162e-04, 3.872e-05],
        [1.151e-04, 3.836e-05],
        [1.140e-04, 3.800e-05],
        [1.130e-04, 3.766e-05],
        [1.120e-04, 3.732e-05],
        [1.110e-04, 3.700e-05],
        [1.101e-04, 3.668e-05],
        [1.091e-04, 3.637e-05],
        [1.082e-04, 3.607e-05],
        [1.073e-04, 3.578e-05],
        [1.065e-04, 3.549e-05],
        [1.056e-04, 3.521e-05],
        [1.048e-04, 3.493e-05],
        [1.040e-04, 3.466e-05],
        [1.032e-04, 3.440e-05],
        [1.024e-04, 3.414e-05],
        [1.017e-04, 3.389e-05],
        [1.009e-04, 3.364e-05],
        [1.002e-04, 3.339e-05],
    ]
)


# ── Lyman series optical depth (LAF) ──────────────────────────────


def _tau_ls_laf(
    wave_obs: jnp.ndarray,
    z_source: float,
) -> jnp.ndarray:
    """Lyman-series Lyman-alpha forest optical depth (Inoue et al. 2014, Eq. 21).

    Vectorized over all 39 Lyman transitions (no Python loop).
    """
    # Broadcast: wave_obs (n_wave,) vs _LAMBDA_LYMAN (39,)
    # Shapes: lam_j (39, 1), wave (1, n_wave) -> (39, n_wave)
    lam_j = _LAMBDA_LYMAN[:, None]  # (39, 1)
    wave = wave_obs[None, :]  # (1, n_wave)

    lam_max = lam_j * (1.0 + z_source)
    active = (wave > lam_j) & (wave <= lam_max)

    lam_break1 = 2.2 * lam_j
    lam_break2 = 5.7 * lam_j

    ratio = wave / lam_j  # (39, n_wave)
    t1 = _A_LAF[:, 0:1] * ratio**1.2
    t2 = _A_LAF[:, 1:2] * ratio**3.7
    t3 = _A_LAF[:, 2:3] * ratio**5.5

    t_j = jnp.where(wave < lam_break1, t1, jnp.where(wave < lam_break2, t2, t3))
    return jnp.sum(jnp.where(active, t_j, 0.0), axis=0)


# ── Lyman series optical depth (DLA) ──────────────────────────────


def _tau_ls_dla(
    wave_obs: jnp.ndarray,
    z_source: float,
) -> jnp.ndarray:
    """Lyman-series damped Lyman-alpha optical depth (Inoue et al. 2014, Eq. 22).

    Vectorized over all 39 Lyman transitions (no Python loop).
    """
    lam_j = _LAMBDA_LYMAN[:, None]  # (39, 1)
    wave = wave_obs[None, :]  # (1, n_wave)

    lam_max = lam_j * (1.0 + z_source)
    active = (wave > lam_j) & (wave <= lam_max)

    lam_break = 3.0 * lam_j
    ratio = wave / lam_j

    t1 = _A_DLA[:, 0:1] * ratio**2.0
    t2 = _A_DLA[:, 1:2] * ratio**3.0

    t_j = jnp.where(wave < lam_break, t1, t2)
    return jnp.sum(jnp.where(active, t_j, 0.0), axis=0)


# ── Lyman continuum optical depth (LAF) ───────────────────────────


def _tau_lc_laf(
    wave_obs: jnp.ndarray,
    z_source: float,
) -> jnp.ndarray:
    """Lyman-continuum Lyman-alpha forest optical depth (Inoue et al. 2014, Eqs. 25–27).

    Implements the same piecewise structure as eazy-py (Brammer et al.),
    with observed wavelength regimes ``wave_obs ≷ lamL*(1+z1,2)`` and three
    source-redshift regimes (z_S < 1.2, 1.2 ≤ z_S < 4.7, z_S ≥ 4.7). The
    active mask ``wave_obs < lamL*(1+z_source)`` naturally extends opacity
    below the rest-frame Lyman limit (912 Å), where the previous
    implementation incorrectly returned τ = 0 (closes #494).
    """
    lam_L = _LAMBDA_LIMIT
    z1 = 1.2
    z2 = 4.7
    one_plus_zs = 1.0 + z_source

    # In-range: photon was absorbed at some absorber redshift between 0 and z_source.
    in_range = wave_obs < lam_L * one_plus_zs

    # Clamp the wavelength ratio to its physical maximum (1 + z_source) before
    # raising to fractional powers. Outside ``in_range`` the result is masked to
    # zero, but JAX evaluates all branches, so the clamp keeps gradients finite
    # and prevents large-r overflow at long observed wavelengths.
    r = jnp.minimum(wave_obs / lam_L, one_plus_zs)

    # ── z_S < 1.2: single observed-wavelength regime ──
    t_low = 0.3248 * (r**1.2 - one_plus_zs ** (-0.9) * r**2.1)

    # ── 1.2 ≤ z_S < 4.7: two sub-regimes split at wave_obs = lamL*(1+z1) ──
    above_z1 = wave_obs >= lam_L * (1.0 + z1)
    t_mid_above = 2.545e-2 * (one_plus_zs**1.6 * r**2.1 - r**3.7)
    t_mid_below = 2.545e-2 * one_plus_zs**1.6 * r**2.1 + 0.3248 * r**1.2 - 0.2496 * r**2.1
    t_mid = jnp.where(above_z1, t_mid_above, t_mid_below)

    # ── z_S ≥ 4.7: three sub-regimes split at lamL*(1+z1) and lamL*(1+z2) ──
    above_z2 = wave_obs > lam_L * (1.0 + z2)
    between_z1z2 = above_z1 & ~above_z2  # lamL*(1+z1) ≤ wave_obs ≤ lamL*(1+z2)
    t_hi_top = 5.221e-4 * (one_plus_zs**3.4 * r**2.1 - r**5.5)
    t_hi_mid = 5.221e-4 * one_plus_zs**3.4 * r**2.1 + 0.2182 * r**2.1 - 2.545e-2 * r**3.7
    t_hi_bot = 5.221e-4 * one_plus_zs**3.4 * r**2.1 + 0.3248 * r**1.2 - 3.140e-2 * r**2.1
    t_high = jnp.where(
        above_z2,
        t_hi_top,
        jnp.where(between_z1z2, t_hi_mid, t_hi_bot),
    )

    tau = jnp.where(
        z_source < z1,
        t_low,
        jnp.where(z_source < z2, t_mid, t_high),
    )
    # Gate on z_source > 0 — the analytic fit is defined as an integral
    # over (0, z_S] and is only ≈ 0 (not exactly 0) at z_S = 0. Physically
    # the path length vanishes at z=0, so transmission must be 1.
    in_range = in_range & (z_source > 0.0)
    return jnp.where(in_range, jnp.clip(tau, min=0.0), 0.0)


# ── Lyman continuum optical depth (DLA) ───────────────────────────


def _tau_lc_dla(
    wave_obs: jnp.ndarray,
    z_source: float,
) -> jnp.ndarray:
    """Lyman-continuum damped Lyman-alpha optical depth (Inoue et al. 2014, Eqs. 28–29).

    Implements the same approach as eazy-py (Brammer et al.). At z_S ≥ 2 the
    formula splits at the observed wavelength ``lamL*(1+z1_DLA)``; below 912
    Å rest the opacity is non-zero (closes #494).
    """
    lam_L = _LAMBDA_LIMIT
    z1 = 2.0
    one_plus_zs = 1.0 + z_source

    in_range = wave_obs < lam_L * one_plus_zs
    r = jnp.minimum(wave_obs / lam_L, one_plus_zs)
    # Floor r away from zero so r**-0.3 stays finite under jnp.where (gradient safety).
    r_safe = jnp.maximum(r, 1e-3)

    # ── z_S < 2 — single observed-wavelength regime ──
    t_low = (
        0.2113 * one_plus_zs**2.0 - 0.07661 * one_plus_zs**2.3 * r_safe ** (-0.3) - 0.1347 * r**2.0
    )

    # ── z_S ≥ 2 — two sub-regimes split at wave_obs = lamL*(1+z1) ──
    above_z1 = wave_obs >= lam_L * (1.0 + z1)
    t_hi_above = (
        0.04696 * one_plus_zs**3.0
        - 0.01779 * one_plus_zs**3.3 * r_safe ** (-0.3)
        - 0.02916 * r**3.0
    )
    t_hi_below = (
        0.6340
        + 0.04696 * one_plus_zs**3.0
        - 0.01779 * one_plus_zs**3.3 * r_safe ** (-0.3)
        - 0.1347 * r**2.0
        - 0.2905 * r_safe ** (-0.3)
    )
    t_high = jnp.where(above_z1, t_hi_above, t_hi_below)

    tau = jnp.where(z_source < z1, t_low, t_high)
    # Gate on z_source > 0 — the analytic fit is defined as an integral
    # over (0, z_S] and is only ≈ 0 (not exactly 0) at z_S = 0. Physically
    # the path length vanishes at z=0, so transmission must be 1.
    in_range = in_range & (z_source > 0.0)
    return jnp.where(in_range, jnp.clip(tau, min=0.0), 0.0)


# ── CGM damping wing absorption (Asada et al. 2025 / Totani et al. 2006) ──


def _cgm_damping_wing_tau(
    wave_obs: jnp.ndarray,
    z_source: float,
    z_mid: float | None = None,
    dz: float | None = None,
    log_nhi: float | None = None,
) -> jnp.ndarray:
    r"""CGM damping wing optical depth from neutral hydrogen (Asada et al. 2025).

    At z > 5, neutral hydrogen in the circumgalactic medium produces a redward
    Lyα damping wing on top of the Inoue+2014 mean IGM. The cross-section is the
    Totani et al. (2006) Eq. 4 frequency-dependent form, and the column-density
    evolution defaults to the Asada+2025 paper sigmoid (Eq. 2). Closes #502.

    Parameters
    ----------
    wave_obs : array_like, shape (n_wave,)
        Observed-frame wavelength. [Å]
    z_source : float
        Redshift of the source. [dimensionless]
    z_mid, dz, log_nhi : float, optional
        Legacy sigmoid knobs
        :math:`N_{\rm HI}(z) = 10^{\rm log\_nhi} / (1 + e^{-(z-z_{\rm mid})/dz})`.
        If any of the three is supplied the legacy form is used; otherwise the
        Asada+2025 paper sigmoid is applied with the published coefficients.

    Returns
    -------
    ndarray, shape (n_wave,)
        Damping wing optical depth. [dimensionless, ≥ 0]

    Notes
    -----
    The Asada+2025 (Eq. 2) column-density evolution is

    .. math::

        \log_{10} N_{\rm HI}(z) = \frac{3.592}{1 + e^{-1.841(z - 6)}} + 18.001

    and the Lyα cross-section (Totani et al. 2006, Eq. 4) is

    .. math::

        \sigma_\alpha(\nu) = \frac{3 \lambda_\alpha^2 f_{12} \Lambda}{8\pi}
            \frac{\Lambda (\nu/\nu_\alpha)^4}
                 {4\pi^2 (\nu - \nu_\alpha)^2 + \Lambda^2 (\nu/\nu_\alpha)^6/4}

    with :math:`\Lambda = A_{21,\,\rm Ly\alpha}` (the Einstein A coefficient) and
    :math:`f_{12} = 0.4162` (Morton 2003). Constants come from
    :mod:`tengri.components.igm.dla`. The previous implementation used a flat
    Lorentzian with a numerical constant that was ~10⁹ too small.

    **Upstream**: Asada et al. (2025), ApJL 983, L2 — column-density evolution;
    Totani et al. (2006), PASJ 58, 485 — Lyα cross-section.
    """
    # Column-density evolution N_HI(z) — paper sigmoid by default; legacy form
    # if the user supplies any of the (z_mid, dz, log_nhi) knobs.
    if z_mid is not None or dz is not None or log_nhi is not None:
        z_mid_eff = 7.0 if z_mid is None else z_mid
        dz_eff = 0.5 if dz is None else dz
        log_nhi_eff = 21.0 if log_nhi is None else log_nhi
        n_hi = (10.0**log_nhi_eff) / (1.0 + jnp.exp(-(z_source - z_mid_eff) / dz_eff))
    else:
        # Asada+2025 Eq. 2 coefficients (paper sigmoid).
        log_nhi_z = 3.592 / (1.0 + jnp.exp(-1.841 * (z_source - 6.0))) + 18.001
        n_hi = 10.0**log_nhi_z

    # The Totani+06 cross-section is in the CGM rest frame (≈ source rest frame).
    # An observed-frame photon at wave_obs absorbs in the CGM at rest wavelength
    # wave_obs / (1+z_source); convert to frequency and take Δν off Lyα.
    lya_obs = _WL_LYA * (1.0 + z_source)
    wave_rest = wave_obs / (1.0 + z_source)
    nu_rest = C_CGS / (wave_rest * 1e-8)
    delta_nu = nu_rest - _NU_LYA
    nu_ratio = nu_rest / _NU_LYA  # = ν / ν_α

    # Totani+06 Eq. 4. The (ν/ν_α)^4 factor is what curves the cross-section
    # away from a flat Lorentzian in the far wing.
    lam_cm = _WL_LYA * 1e-8
    prefactor = 3.0 * lam_cm**2 * _F_LYA * _A_LYA / (8.0 * jnp.pi)
    numerator = _A_LYA * nu_ratio**4
    denominator = 4.0 * jnp.pi**2 * delta_nu**2 + (_A_LYA**2) * nu_ratio**6 / 4.0
    sigma_dw = prefactor * numerator / denominator

    # Damping wing is redward of Lyα-at-source and only matters at z > 5
    # (below this the CGM is essentially ionized).
    tau = n_hi * sigma_dw
    tau = jnp.where(wave_obs > lya_obs, tau, 0.0)
    tau = jnp.where(z_source > 5.0, tau, 0.0)
    return jnp.clip(tau, min=0.0)


# ── Public API ────────────────────────────────────────────────────


def igm_transmission(
    wave_obs: jnp.ndarray,
    z_source: float,
    add_cgm: bool = False,
    cgm_z_mid: float | None = None,
    cgm_dz: float | None = None,
    cgm_log_nhi: float | None = None,
) -> jnp.ndarray:
    r"""Compute mean IGM transmission including Lyman-series and continuum absorption.

    Implements the Inoue et al. (2014) prescription for the mean intergalactic medium
    (IGM) absorption from Lyman-series lines and continuum across all 39 Lyman transitions
    (Lyman-alpha through n=40), accounting for both the Lyman-alpha forest (LAF) and
    damped Lyman-alpha systems (DLA). Optionally includes circumgalactic medium (CGM)
    damping wing absorption at z > 5.

    Parameters
    ----------
    wave_obs : array_like, shape (n_wave,)
        Observed-frame wavelength. [Å]
    z_source : float
        Redshift of the source galaxy. [dimensionless]
    add_cgm : bool, optional
        If True, add CGM damping wing absorption (Asada et al. 2025) at z > 5.
        Default: False (experimental feature, not yet fully validated against observations).
    cgm_z_mid : float, optional
        Redshift midpoint of the sigmoid column density evolution. [dimensionless]
        Default: 7.0.
    cgm_dz : float, optional
        Redshift width of the sigmoid transition. [dimensionless] Default: 0.5.
    cgm_log_nhi : float, optional
        log10(N_HI / cm^-2) at the plateau of the sigmoid evolution. Canonical Asada+2025 value (21.0)
        produces τ ≈ 0.15 (15% absorption) redward of Lyα at z=7; log_nhi ≤ 19 is invisible. [dimensionless]
        Default: 21.0.

    Returns
    -------
    ndarray, shape (n_wave,)
        Transmission factor T(λ) ∈ [0, 1]. [dimensionless]
        Multiply the rest-frame SED (in erg/s/Hz) by T to obtain the observed flux absorbed by the IGM.

    Notes
    -----
    **JIT-compatible**: yes — all operations are ``jnp`` primitives, fully vectorized over wavelength.

    **Gradient-safe**: yes — differentiable everywhere via :math:`\exp(-\tau)` (no discontinuities).

    The total IGM optical depth is:

    .. math::

        \tau_{\rm IGM}(\lambda_{\rm obs}, z_s) = \tau_{\rm LS}^{\rm LAF} + \tau_{\rm LS}^{\rm DLA}
        + \tau_{\rm LC}^{\rm LAF} + \tau_{\rm LC}^{\rm DLA}

    where LS = Lyman-series line absorption, LC = Lyman-continuum absorption, LAF = Lyman-alpha forest,
    and DLA = damped Lyman-alpha systems. Each component is computed via piecewise power-law fits
    to photoionization simulations (Inoue et al. 2014, Tables 2–3).

    The **Lyman-series line absorption** (LAF) is:

    .. math::

        \tau_j^{\rm LAF}(\lambda_{\rm obs}) = \begin{cases}
        A_{j,1}^{\rm LAF} \, (\lambda_{\rm obs}/\lambda_j)^{1.2} & \lambda_{\rm obs} < 2.2\,\lambda_j \\
        A_{j,2}^{\rm LAF} \, (\lambda_{\rm obs}/\lambda_j)^{3.7} & 2.2\,\lambda_j \leq \lambda_{\rm obs} < 5.7\,\lambda_j \\
        A_{j,3}^{\rm LAF} \, (\lambda_{\rm obs}/\lambda_j)^{5.5} & \lambda_{\rm obs} \geq 5.7\,\lambda_j
        \end{cases}

    summed over all 39 Lyman lines (j = 2, Ly-alpha, to j = 40). The DLA line absorption follows
    a similar form with two regimes (Inoue et al. 2014, Eq. 22).

    The **transmission** is:

    .. math::

        T_{\rm IGM}(\lambda_{\rm obs}, z_s) = \exp[-\tau_{\rm IGM}(\lambda_{\rm obs}, z_s)]

    **High-redshift extension (z > 5)**: When ``add_cgm=True``, CGM damping wing absorption
    (Asada et al. 2025) is added, modeling neutral hydrogen in the circumgalactic medium.
    This affects wavelengths redward of Lyman-alpha at the source redshift and is important
    for z > 5 galaxies.

    **Approximations**:

    - **Mean transmission**: This is the mean IGM absorption averaged over cosmic variance.
        Individual sightlines have additional scatter from the Lyman-alpha forest (not included).
    - **Piecewise power laws**: The Inoue et al. (2014) model uses analytic fits to simulations
        rather than full radiative transfer. Accuracy is ~5–10% relative to simulations.
    - **Validity**: Calibrated for :math:`z \lesssim 6` (Inoue et al. 2014). At :math:`z > 6`,
        reionization introduces significant variance; the patchy reionization model is available separately.

    **Upstream**: Implements Inoue et al. (2014) [1]_ mean IGM transmission model, with
    CGM damping wing extension from Asada et al. (2025) [2]_. Coefficients extracted from
    eazy-py (Brammer et al.).

    References
    ----------
    .. [1] A. K. Inoue, I. Shimizu, I. Iwata, and M. Tanaka, "An updated analytic model for
       attenuation by the intergalactic medium," MNRAS, 442, 1805 (2014).
       https://doi.org/10.1093/mnras/stu936

    .. [2] Y. Asada et al., "Improving Photometric Redshifts of Epoch of Reionization Galaxies:
       A New Empirical Transmission Curve with Neutral Hydrogen Damping Wing Ly-alpha Absorption,"
       ApJL, 983, L2 (2025).
       arXiv:2410.21543. https://doi.org/10.3847/2041-8213/adc388
    """
    tau_total = (
        _tau_ls_laf(wave_obs, z_source)
        + _tau_ls_dla(wave_obs, z_source)
        + _tau_lc_laf(wave_obs, z_source)
        + _tau_lc_dla(wave_obs, z_source)
    )

    # CGM damping wing (Asada+2025): additional absorption at z > 5
    tau_cgm = jnp.where(
        add_cgm,
        _cgm_damping_wing_tau(wave_obs, z_source, cgm_z_mid, cgm_dz, cgm_log_nhi),
        0.0,
    )
    tau_total = tau_total + tau_cgm

    return jnp.exp(-jnp.clip(tau_total, min=0.0))


# ── Patchy reionization damping wing (Mason+2018, Keating+2025) ───


def _damping_wing_tau(
    wave_obs: jnp.ndarray,
    z: float,
    x_HI: float,
    R_bubble: float,
) -> jnp.ndarray:
    """Damping wing optical depth from a partially neutral IGM.

    Follows Miralda-Escude (1998, ApJ 501, 15) Eq. 9 for the
    integrated damping wing profile from a uniform neutral IGM
    extending from the bubble edge to high redshift.

    The damping wing extends *redward* of Ly-alpha at the source
    redshift (unlike the Lyman series forest which is blueward).
    An ionized bubble of radius R_bubble around the source carves
    out a proximity zone where tau = 0.

    Parameters
    ----------
    wave_obs : array, shape (n_wave,)
        Observed-frame wavelength [Angstrom].
    z : float
        Source redshift.
    x_HI : float
        Volume-averaged neutral fraction (0 = ionized, 1 = neutral).
    R_bubble : float
        Ionized bubble radius [proper Mpc].

    Returns
    -------
    array, shape (n_wave,)
        Damping wing optical depth (>= 0).

    Notes
    -----
    The Gunn-Peterson optical depth is tau_GP = 6.45e5 * ((1+z)/7)^1.5.
    The integrated damping wing from a uniform neutral medium gives
    (Miralda-Escude 1998):

        tau_DW(x) ~ tau_GP * x_HI * Lambda_alpha / (pi * x^2)

    where x = (nu - nu_alpha)/nu_alpha is the dimensionless frequency
    offset and Lambda_alpha = Gamma_alpha / (4*pi*nu_alpha) is the
    dimensionless damping constant.  The bubble provides a lower
    integration limit that suppresses absorption close to Ly-alpha.
    """
    # Gunn-Peterson optical depth at Ly-alpha (Miralda-Escude 1998 Eq. 1)
    tau_GP = 6.45e5 * ((1.0 + z) / 7.0) ** 1.5

    # Observed Ly-alpha at source redshift
    lya_obs = _WL_LYA * (1.0 + z)

    # Dimensionless wavelength offset: x = lambda_obs/lya_obs - 1
    # x > 0 is redward of Lya (damping wing side)
    x_wave = wave_obs / lya_obs - 1.0

    # Damping constant: Lambda = Gamma_alpha / (4*pi*nu_alpha)
    # nu_alpha = c / (lambda_alpha * 1e-8)
    lambda_alpha = _WL_LYA * 1e-8  # cm
    nu_alpha = C_CGS / lambda_alpha
    lambda_damp = _A_LYA / (4.0 * jnp.pi * nu_alpha)

    # Bubble edge in dimensionless frequency offset:
    # The bubble of radius R_bubble [pMpc] corresponds to a velocity
    # offset v_bubble = R_bubble * H(z), hence a wavelength offset
    # x_bubble = v_bubble / c.
    # H(z) = H_0 * sqrt(Omega_m * (1+z)^3) for matter-dominated era
    # Use canonical PLANCK18 cosmology — sourced from tengri.cosmology
    # (Planck 2020, A&A 641, A6: h = 0.6766, Om0 = 0.30966).
    h_z_kms_per_mpc = 100.0 * PLANCK18.h * jnp.sqrt(PLANCK18.Om0 * (1.0 + z) ** 3)
    v_bubble = R_bubble * h_z_kms_per_mpc  # km/s
    x_bubble = v_bubble / 2.998e5  # dimensionless

    # Integrated damping wing tau from Miralda-Escude (1998) Eq. 9:
    # For a source behind a semi-infinite slab of neutral gas extending from
    # the bubble edge (at offset x_bubble) to high redshift, the integrated
    # damping wing is obtained from shifting the integration bounds by the
    # observed frequency offset x_obs:
    #   tau(x_obs) = tau_GP * x_HI * Lambda / (pi * x_eff)
    # where x_eff = x_obs + x_bubble is the effective offset accounting for
    # the bubble size. Each parcel at velocity v > v_bubble sees the photon
    # at cumulative offset x_obs + x_bubble (the bubble suppresses the
    # near-line absorption at x < x_bubble).
    # This represents tau ∝ 1/x in the far wing, not 1/x^2 (which would be
    # a thin shell at the bubble edge; the extended medium gives 1/x).
    x_eff = x_wave + x_bubble  # Effective offset including bubble size
    x_eff_safe = jnp.maximum(x_eff, 1e-10)  # Avoid division by zero
    tau_wing = tau_GP * x_HI * lambda_damp / (jnp.pi * x_eff_safe)

    # Smooth bubble mask: suppress for x < x_bubble (inside bubble)
    bubble_mask = jax.nn.sigmoid((x_wave - x_bubble) / jnp.maximum(x_bubble * 0.1, 1e-8))
    tau_wing = tau_wing * bubble_mask

    # Only apply redward of Ly-alpha at source (damping wing side)
    tau_wing = jnp.where(wave_obs > lya_obs, tau_wing, 0.0)

    return jnp.clip(tau_wing, min=0.0)


def igm_transmission_patchy(
    wave_obs: jnp.ndarray,
    z: float,
    x_HI: float = 0.0,
    R_bubble: float = 1.0,
    **kwargs,
) -> jnp.ndarray:
    """IGM transmission with patchy reionization damping wing.

    Extends Inoue+2014 with a neutral IGM damping wing component
    for z > 5.5 where reionization is incomplete. When ``x_HI = 0``,
    this reduces exactly to standard ``igm_transmission``.

    The damping wing optical depth follows Miralda-Escude (1998):

        tau_DW(lambda) = tau_GP * x_HI * (gamma / 4pi)
                         / (delta_nu^2 + (gamma/4pi)^2)

    where delta_nu is the frequency offset from Ly-alpha at the source
    redshift, and the Gunn-Peterson optical depth is:

        tau_GP = 6.45e5 * ((1+z)/7)^1.5

    An ionized bubble of radius ``R_bubble`` around the source creates
    a proximity zone that suppresses the damping wing close to Ly-alpha.

    Parameters
    ----------
    wave_obs : array, shape (n_wave,)
        Observed-frame wavelength [Angstrom].
    z : float
        Source redshift.
    x_HI : float
        Volume-averaged neutral hydrogen fraction. 0 = fully ionized
        (standard Inoue+2014), 1 = fully neutral. At z~6: x_HI ~ 0.1-0.5.
        At z~8: x_HI ~ 0.5-0.9. Default 0.0.
    R_bubble : float
        Radius of ionized bubble around the source [proper Mpc].
        Typical: 0.5-5 pMpc at z~7. Larger bubbles reduce the damping
        wing absorption. Default 1.0.
    **kwargs
        Additional keyword arguments passed to ``igm_transmission``
        (e.g., add_cgm, cgm_z_mid, etc.).

    Returns
    -------
    array, shape (n_wave,)
        Transmission factor T in [0, 1]. Multiply rest-frame SED by
        T to get absorbed spectrum.

    Notes
    -----
    At z < 5.5, the neutral fraction is effectively zero and this
    function returns the same result as ``igm_transmission``.

    The damping wing extends *redward* of Ly-alpha (unlike the Lyman
    series forest which absorbs blueward). This means it affects the
    UV continuum longward of Ly-alpha, which is not captured by the
    standard Inoue+2014 model.

    References
    ----------

    - Miralda-Escude 1998, ApJ, 501, 15
    - Mason et al. 2018, ApJ, 856, 2
    - Keating et al. 2025

    """
    # Standard Inoue+2014 transmission
    t_inoue = igm_transmission(wave_obs, z, **kwargs)

    # Damping wing from partially neutral IGM
    tau_wing = _damping_wing_tau(wave_obs, z, x_HI, R_bubble)
    t_wing = jnp.exp(-tau_wing)

    return t_inoue * t_wing


# ── Madau+1995 IGM (alternative, simpler model) ───────────────────


# Lyman-series rest-frame wavelengths and coefficients (Madau 1995, Table 1).
# 17 lines from Ly-alpha (1216 Å) to Ly-limit.
_MADAU_LYW = jnp.array(
    [
        1215.67,
        1025.72,
        972.537,
        949.743,
        937.803,
        930.748,
        926.226,
        923.150,
        920.963,
        919.352,
        918.129,
        917.181,
        916.429,
        915.824,
        915.329,
        914.919,
        914.576,
    ]
)
_MADAU_COEFF = jnp.array(
    [
        3.6e-3,
        1.7e-3,
        1.1846e-3,
        9.410e-4,
        7.960e-4,
        6.967e-4,
        6.236e-4,
        5.665e-4,
        5.200e-4,
        4.817e-4,
        4.487e-4,
        4.200e-4,
        3.947e-4,
        3.720e-4,
        3.520e-4,
        3.334e-4,
        3.1644e-4,
    ]
)
_MADAU_LYLIM = 911.75  # Lyman limit [Angstrom]


def igm_transmission_madau(
    wave_obs: jnp.ndarray,
    z: float,
    igm_factor: float = 1.0,
) -> jnp.ndarray:
    r"""Mean IGM transmission using the Madau (1995) model.

    Computes line-of-sight Lyman-series absorption from 17 lines plus
    continuum absorption below the Lyman limit. This is the simpler of the
    two IGM models available in tengri; the default is :func:`igm_transmission`
    (Inoue+2014) which includes 39 lines and DLA contributions.

    Parameters
    ----------
    wave_obs : array_like, shape (n_wave,)
        Observed-frame wavelengths [Angstrom].
    z : float
        Source redshift.
    igm_factor : float, optional
        Multiplicative fudge factor for IGM strength. Default 1.0 (mean IGM).

    Returns
    -------
    T_igm : jnp.ndarray, shape (n_wave,)
        Mean IGM transmission [dimensionless], in [0, 1]. Values outside the modeled
        wavelength range are set to 1.0 (no attenuation).

    Notes
    -----
    **JIT-compatible**: yes — pure ``jnp`` operations with ``jax.lax.scan``
    for the line summation.

    **Line opacity formula**: The optical depth from each Lyman-series line is:

    .. math::

        \tau_j(\lambda) = A_j \left(\frac{\lambda}{\lambda_j}\right)^{3.46}

    where :math:`A_j` are the line coefficients from Madau (1995) Table 1 [1]_,
    :math:`\lambda_j` are the rest-frame Lyman-series wavelengths [Angstrom], and
    the formula applies for :math:`\lambda_j \leq \lambda_\mathrm{obs} \leq \lambda_j(1+z)`.

    ``wave_obs`` is **observed-frame** wavelength (consistent with tengri's
    IGM convention). The Madau+1995 model takes observed-frame wavelengths
    directly.

    Implements Prospector ``add_igm`` in ``fake_fsps.py`` (Johnson+2021 [2]_).
    The Inoue+2014 model (:func:`igm_transmission`) supersedes this for science
    use; Madau+1995 is provided for comparison and backward compatibility.

    References
    ----------
    .. [1] P. Madau, "Radiative Transfer in a Clumpy Universe: The Colors of
       High-Redshift Galaxies," ApJ, 441, 18 (1995).
       https://doi.org/10.1086/175332
    .. [2] B. D. Johnson et al., "Stellar Population Inference," ApJS, 254,
       22 (2021). arXiv:2012.01426. https://doi.org/10.3847/1538-4295/abef67

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> wave_obs = jnp.linspace(1000.0, 10000.0, 200)
    >>> T = igm_transmission_madau(wave_obs, z=3.0)
    >>> T.shape
    (200,)
    >>> float(T[0]) < 1.0  # attenuated at Ly-alpha
    True
    """
    wave_obs = jnp.asarray(wave_obs, dtype=jnp.float64)

    # ── Lyman-series line absorption ─────────────────────────────
    # For each line j: tau_j(lam) = A_j * (lam/lyw_j)^3.46
    # applied over lyw_j <= lam <= lyw_j * (1 + z)
    def _add_line_tau(carry, line_data):
        """Accumulate optical depth from one Lyman-series line."""
        lyw_j, coeff_j = line_data[0], line_data[1]
        lmax = lyw_j * (1.0 + z)
        in_range = (wave_obs >= lyw_j) & (wave_obs <= lmax)
        tau_j = coeff_j * (wave_obs / lyw_j) ** 3.46
        return carry + jnp.where(in_range, tau_j, 0.0), None

    tau_line, _ = jax.lax.scan(
        _add_line_tau,
        jnp.zeros_like(wave_obs),
        jnp.stack([_MADAU_LYW, _MADAU_COEFF], axis=1),
    )

    # ── Lyman-continuum absorption ────────────────────────────────
    # Madau+1995 Eq. 16: continuum opacity below Lyman limit
    xc = wave_obs / _MADAU_LYLIM
    xem = 1.0 + z

    # Clamp xc to [1.0, xem]
    xc_lo = jnp.clip(xc, 1.0, None)
    xc_hi = jnp.clip(xc_lo, None, xem)

    tau_cont = (
        0.25 * xc_hi**3 * (jnp.exp(0.46 * jnp.log(xem)) - jnp.exp(0.46 * jnp.log(xc_hi)))
        + 9.4 * xc_hi**1.5 * (jnp.exp(0.18 * jnp.log(xem)) - jnp.exp(0.18 * jnp.log(xc_hi)))
        - 0.7 * xc_hi**3 * (xc_hi ** (-1.32) - xem ** (-1.32))
        - 0.023 * (xem**1.68 - xc_hi**1.68)
    )
    # Continuum only applies blueward of Lyman limit in the observed frame
    tau_cont = jnp.where(wave_obs < _MADAU_LYLIM * (1.0 + z), tau_cont, 0.0)

    tau_total = (tau_line + tau_cont) * igm_factor
    return jnp.exp(-jnp.clip(tau_total, 0.0, None))


# ── Registry ──────────────────────────────────────────────────────────
#
# IGM transmission backends. Canonical name keys (publication-correct);
# legacy aliases are resolved by ``_IGM_ALIASES`` so both the dict-grammar
# validator path and the SEDModel dispatch can read from one source of
# truth (per ADR-0005 / ADR-0008). Each value is the pure-JAX transmission
# function and shares the public signature ``(wave_obs, z, **kwargs)``.

from tengri.components.igm.meiksin06 import igm_transmission_meiksin06


def igm_transmission_asada25(wave_obs: jnp.ndarray, z: float, **kwargs: object) -> jnp.ndarray:
    r"""Inoue+2014 mean IGM plus the Asada+2025 proximate-CGM damping wing.

    A registry model (not a flag) so future CGM prescriptions slot in as
    additional registry entries. Uses the published Asada+2025 (ApJL 983, L2)
    H I column-density sigmoid (``cgm_*`` left at their paper defaults). Fixes
    the z > 7 photometric-redshift bias from assuming a sharp Lyman break.

    Notes
    -----
    **JIT-compatible**: yes — delegates to :func:`igm_transmission`.
    """
    del kwargs
    return igm_transmission(wave_obs, z, add_cgm=True)


IGM_TRANSMISSION_MODELS: dict[str, object] = {
    "inoue14": igm_transmission,
    "madau": igm_transmission_madau,
    # Added by #446 (CIGALE-matching IGM) but #343's refactor missed wiring
    # this into the canonical registry; the dict-grammar validator and
    # builder factory both accepted ``"meiksin06"`` while
    # ``IGM_TRANSMISSION_MODELS`` and ``resolve_igm_model`` did not — exactly
    # the kind of drift the parity contract test was added to catch.
    "meiksin06": igm_transmission_meiksin06,
    # Inoue+2014 + Asada+2025 CGM damping wing, as its own model so new CGM
    # prescriptions register flatly rather than accreting boolean flags.
    "asada25": igm_transmission_asada25,
}

#: Back-compat aliases that route to canonical registry keys. The bare
#: ``"inoue"`` was the internal default in tengri pre-2026-05 while the
#: dict-grammar API consistently used ``"inoue14"``; both now resolve to
#: the same Inoue+2014 function.
_IGM_ALIASES: dict[str, str] = {
    "inoue": "inoue14",
}


def resolve_igm_model(name: str) -> object:
    """Return the IGM transmission function for ``name``, resolving aliases.

    Parameters
    ----------
    name : str
        Registry key (e.g. ``"inoue14"``, ``"madau"``) or a recognized
        alias (e.g. ``"inoue"``).

    Returns
    -------
    Callable
        Pure-JAX transmission function ``(wave_obs, z, **kwargs) -> T_igm``.

    Raises
    ------
    ValueError
        If ``name`` is neither a registry key nor a known alias.
    """
    resolved = _IGM_ALIASES.get(name, name)
    if resolved not in IGM_TRANSMISSION_MODELS:
        available = sorted(IGM_TRANSMISSION_MODELS.keys() | _IGM_ALIASES.keys())
        raise ValueError(f"Unknown IGM model {name!r}. Available: {available}")
    return IGM_TRANSMISSION_MODELS[resolved]


def igm_absorption(
    wave_obs: jnp.ndarray,
    z: float,
    *,
    igm_x_HI: float = 0.0,
    igm_bubble_mpc: float = 10.0,
    igm_patchy: bool = False,
    igm_model: str = "inoue",
    use_dla: bool = False,
    dla_z: float = 0.0,
    dla_log_n_hi: float = DEFAULT_DLA_LOG_N_HI,
    dla_temp: float = 1e4,
    dla_b_turb: float = 0.0,
) -> jnp.ndarray:
    r"""Total observed-frame absorption — the single flat dispatch.

    Composes the observed-frame transmission from a *mean-IGM model* and
    optional *modifiers*, and is the ONE call every observed-frame consumer
    uses (the exact ``predict_obs_sed`` path and the :class:`IGMSEDComponent`
    photometry/spectroscopy projection), so all paths stay consistent (#932):

    * **mean IGM**: resolved once from the registry
      (:func:`resolve_igm_model`: ``inoue``/``inoue14``, ``madau``,
      ``meiksin06``, ``asada25`` = Inoue + Asada+2025 CGM damping wing), or
      replaced by the patchy-reionization model when ``igm_patchy`` is set.
    * **DLA**: an optional multiplicative damped-Lyman-α absorber layered on
      top of the mean IGM (``use_dla``), so photometry and spectroscopy both
      see it rather than only ``predict_obs_sed``.

    New CGM prescriptions are added as new *registry models* (like
    ``asada25``), not flags — keeping the per-model dispatch flat.

    Parameters
    ----------
    wave_obs : ndarray, shape (n_wave,)
        Observed-frame wavelength [Angstrom].
    z : float
        Source redshift [dimensionless].
    igm_x_HI, igm_bubble_mpc : float, optional
        Patchy-reionization neutral fraction (0-1) and bubble radius [proper
        Mpc]; only used when ``igm_patchy=True``.
    igm_patchy : bool, optional
        Use the patchy reionization damping-wing model instead of the mean
        IGM. Default ``False``.
    igm_model : str, optional
        Registry key or alias of the mean-IGM model. Default ``"inoue"``.
    use_dla : bool, optional
        Multiply by a damped-Lyman-α absorber. Default ``False``.
    dla_z, dla_log_n_hi, dla_temp, dla_b_turb : float, optional
        DLA absorber redshift (0 → source ``z``), log10 H I column density
        [cm^-2], temperature [K], and turbulent Doppler velocity [km/s].

    Returns
    -------
    ndarray, shape (n_wave,)
        Transmission fraction [dimensionless, 0-1].

    Notes
    -----
    **JIT-compatible**: yes. ``igm_patchy`` / ``use_dla`` are static structural
    flags, so their branches resolve at trace time. The absorber knobs
    (``igm_x_HI``, ``igm_bubble_mpc``, ``dla_*``) are runtime values that may be
    traced free parameters — they must never gate a Python branch. At
    ``igm_x_HI = 0`` the patchy path reduces bit-for-bit to the mean-IGM model.
    """
    if igm_model in ("none", None):
        # Mean IGM disabled (e.g. DLA-only, or a low-z fit with only a
        # foreground absorber): start from unit transmission.
        transmission = jnp.ones_like(wave_obs)
    elif igm_patchy:
        # ``igm_patchy`` is a static structural flag, so this branch resolves at
        # trace time. ``igm_x_HI`` is a (possibly free) param and therefore a
        # tracer under jit — it must NOT gate the branch. Guarding on it with
        # ``and igm_x_HI > 0.0`` raised TracerBoolConversionError on every path
        # (#1149). The guard was also redundant: the damping-wing optical depth
        # scales linearly with ``x_HI``, so ``igm_transmission_patchy`` reduces
        # bit-for-bit to the mean model at ``x_HI = 0`` (``exp(-0) = 1``).
        transmission = igm_transmission_patchy(wave_obs, z, x_HI=igm_x_HI, R_bubble=igm_bubble_mpc)
    else:
        transmission = resolve_igm_model(igm_model)(wave_obs, z)
    if use_dla:
        from tengri.components.igm.dla import dla_transmission_obs

        z_dla = jnp.where(dla_z > 0.0, dla_z, z)
        transmission = transmission * dla_transmission_obs(
            wave_obs,
            z_dla=z_dla,
            log_n_hi=dla_log_n_hi,
            temp=dla_temp,
            b_turb_kms=dla_b_turb,
        )
    return transmission
