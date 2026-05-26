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

from tengri.cosmology import PLANCK18
from tengri.utils.physics_constants import C_CGS

# ── Lyman series wavelengths (Angstrom) for lines j=2 (Ly-alpha) to j=40
_N_LINES = 39

# Rest-frame wavelengths of Lyman series lines (Angstrom)
_LAMBDA_LYMAN = jnp.array(
    [
        1216.0,
        1025.720,
        972.537,
        949.743,
        937.804,
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

# Lyman limit wavelength
_LAMBDA_LIMIT = 912.0  # Angstrom

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
# Shape: (39, 2) — [A_j1, A_j2]
# From eazy-py DLAcoeff.txt
_A_DLA = jnp.array(
    [
        [1.617e-04, 1.545e-04],
        [1.545e-04, 1.477e-04],
        [1.498e-04, 1.432e-04],
        [1.460e-04, 1.395e-04],
        [1.429e-04, 1.366e-04],
        [1.402e-04, 1.340e-04],
        [1.381e-04, 1.320e-04],
        [1.363e-04, 1.302e-04],
        [1.348e-04, 1.289e-04],
        [1.338e-04, 1.279e-04],
        [1.327e-04, 1.269e-04],
        [1.319e-04, 1.261e-04],
        [1.313e-04, 1.255e-04],
        [1.307e-04, 1.249e-04],
        [1.303e-04, 1.245e-04],
        [1.299e-04, 1.242e-04],
        [1.296e-04, 1.239e-04],
        [1.293e-04, 1.236e-04],
        [1.291e-04, 1.234e-04],
        [1.289e-04, 1.232e-04],
        [1.287e-04, 1.230e-04],
        [1.286e-04, 1.229e-04],
        [1.284e-04, 1.228e-04],
        [1.283e-04, 1.227e-04],
        [1.282e-04, 1.225e-04],
        [1.281e-04, 1.224e-04],
        [1.281e-04, 1.224e-04],
        [1.280e-04, 1.223e-04],
        [1.279e-04, 1.223e-04],
        [1.279e-04, 1.222e-04],
        [1.278e-04, 1.222e-04],
        [1.278e-04, 1.221e-04],
        [1.277e-04, 1.221e-04],
        [1.277e-04, 1.220e-04],
        [1.277e-04, 1.220e-04],
        [1.276e-04, 1.220e-04],
        [1.276e-04, 1.219e-04],
        [1.276e-04, 1.219e-04],
        [1.275e-04, 1.219e-04],
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
    """Lyman-continuum Lyman-alpha forest optical depth (Inoue et al. 2014, Eqs. 25–27)."""
    # Absorbers at redshift z_abs contribute for wave_obs = 911.8*(1+z_abs)
    # So wave_obs must be > 911.8 (rest Lyman limit) and < 911.8*(1+z_source)
    active = (wave_obs > _LAMBDA_LIMIT) & (wave_obs <= _LAMBDA_LIMIT * (1.0 + z_source))

    z_obs = wave_obs / _LAMBDA_LIMIT - 1.0
    # Clamp z_obs >= 0 so fractional exponents (1.2, 3.7, 5.5) never receive a
    # negative base in the inactive region (wave_obs < lambda_limit). JAX evaluates
    # all branches regardless of the active mask, so without this clamp the power
    # expressions produce NaN for short-wavelength photons.
    z_obs_safe = jnp.maximum(z_obs, 0.0)

    # Three source-redshift regimes
    # Regime z_S < 1.2
    t_low = (
        0.325 * ((1.0 + z_obs_safe) ** 1.2 - jnp.clip(1.0 + z_source, max=2.2) ** 1.2)
        - 9.4e-2 * ((1.0 + z_obs_safe) ** 3.7 - jnp.clip(1.0 + z_source, max=2.2) ** 3.7)
        + 0.01478 * ((1.0 + z_obs_safe) ** 5.5 - jnp.clip(1.0 + z_source, max=2.2) ** 5.5)
    )

    # Regime 1.2 <= z_S < 4.7
    t_mid = (
        2.55e-2 * ((1.0 + z_obs_safe) ** 1.2 - (1.0 + z_source) ** 1.2)
        - 0.325 * ((1.0 + z_obs_safe) ** 1.2 - jnp.clip(1.0 + z_source, max=2.2) ** 1.2)
        - 1.15e-2 * ((1.0 + z_obs_safe) ** 3.7 - jnp.clip(1.0 + z_source, max=5.7) ** 3.7)
        + 9.4e-2 * ((1.0 + z_obs_safe) ** 3.7 - jnp.clip(1.0 + z_source, max=2.2) ** 3.7)
        - 7.83e-4 * ((1.0 + z_obs_safe) ** 5.5 - jnp.clip(1.0 + z_source, max=5.7) ** 5.5)
        + 0.01478 * ((1.0 + z_obs_safe) ** 5.5 - jnp.clip(1.0 + z_source, max=2.2) ** 5.5)
    )

    # Regime z_S >= 4.7
    t_high = (
        5.22e-4 * ((1.0 + z_obs_safe) ** 1.2 - (1.0 + z_source) ** 1.2)
        + 2.55e-2 * ((1.0 + z_obs_safe) ** 1.2 - (1.0 + z_source) ** 1.2)
        - 0.325 * ((1.0 + z_obs_safe) ** 1.2 - jnp.clip(1.0 + z_source, max=2.2) ** 1.2)
        - 1.328e-3 * ((1.0 + z_obs_safe) ** 3.7 - (1.0 + z_source) ** 3.7)
        - 1.15e-2 * ((1.0 + z_obs_safe) ** 3.7 - jnp.clip(1.0 + z_source, max=5.7) ** 3.7)
        + 9.4e-2 * ((1.0 + z_obs_safe) ** 3.7 - jnp.clip(1.0 + z_source, max=2.2) ** 3.7)
        - 5.15e-5 * ((1.0 + z_obs_safe) ** 5.5 - (1.0 + z_source) ** 5.5)
        - 7.83e-4 * ((1.0 + z_obs_safe) ** 5.5 - jnp.clip(1.0 + z_source, max=5.7) ** 5.5)
        + 0.01478 * ((1.0 + z_obs_safe) ** 5.5 - jnp.clip(1.0 + z_source, max=2.2) ** 5.5)
    )

    tau = jnp.where(
        z_source < 1.2,
        t_low,
        jnp.where(z_source < 4.7, t_mid, t_high),
    )

    return jnp.where(active, jnp.clip(tau, min=0.0), 0.0)


# ── Lyman continuum optical depth (DLA) ───────────────────────────


def _tau_lc_dla(
    wave_obs: jnp.ndarray,
    z_source: float,
) -> jnp.ndarray:
    """Lyman-continuum damped Lyman-alpha optical depth (Inoue et al. 2014, Eqs. 28–29)."""
    active = (wave_obs > _LAMBDA_LIMIT) & (wave_obs <= _LAMBDA_LIMIT * (1.0 + z_source))
    z_obs = wave_obs / _LAMBDA_LIMIT - 1.0

    # Two source-redshift regimes
    t_low = (
        0.2113 * (1.0 + z_source) ** 2.0
        - 7.661e-2 * (1.0 + z_source) ** 2.5 * (1.0 + z_obs) ** (-0.5)
        - 0.1347 * (1.0 + z_obs) ** 2.0
    )

    t_high = (
        4.696e-2 * (1.0 + z_source) ** 3.0
        - 1.779e-2 * (1.0 + z_source) ** 3.5 * (1.0 + z_obs) ** (-0.5)
        - 2.916e-2 * (1.0 + z_obs) ** 3.0
    )

    tau = jnp.where(z_source < 2.0, t_low, t_high)
    return jnp.where(active, jnp.clip(tau, min=0.0), 0.0)


# ── CGM damping wing absorption (Asada et al. 2025) ───────────────

# Physical constants for damping wing calculation
_LAMBDA_LYA = 1215.67  # Angstrom (Ly-alpha rest wavelength)
_NU_LYA = C_CGS / (_LAMBDA_LYA * 1e-8)  # Hz
_GAMMA_LYA = 6.265e8  # s^-1 (Ly-alpha natural line width)
_SIGMA_0 = 5.9e-14  # cm^2 Hz (Ly-alpha cross-section constant: pi*e^2/(m_e*c)*f_12)


def _cgm_damping_wing_tau(
    wave_obs: jnp.ndarray,
    z_source: float,
    z_mid: float = 7.0,
    dz: float = 0.5,
    log_nhi: float = 21.0,
) -> jnp.ndarray:
    r"""CGM damping wing optical depth from neutral hydrogen (Asada et al. 2025).

    At z > 5, neutral hydrogen in the circumgalactic medium produces Lyman-alpha
    damping wing absorption not captured by the Inoue et al. (2014) model. The damping
    wing profile is the Lorentzian far-wing of the Lyman-alpha cross-section.

    Parameters
    ----------
    wave_obs : array_like, shape (n_wave,)
        Observed-frame wavelength. [Å]
    z_source : float
        Redshift of the source. [dimensionless]
    z_mid : float, optional
        Redshift midpoint of the sigmoid column density evolution. [dimensionless] Default: 7.0.
    dz : float, optional
        Redshift width of the sigmoid. [dimensionless] Default: 0.5.
    log_nhi : float, optional
        log10(N_HI / cm^-2) at the plateau. Canonical Asada+2025 value: 21.0 (τ ≈ 0.15 at z=7).
        log_nhi ≤ 19 is effectively invisible. [dimensionless] Default: 21.0.

    Returns
    -------
    ndarray, shape (n_wave,)
        Damping wing optical depth. [dimensionless, ≥ 0]

    Notes
    -----
    The column density evolves as:

    .. math::

        N_{\rm HI}(z) = \frac{N_{\rm HI,0}}{1 + \exp[-(z - z_{\rm mid})/\Delta z]}

    and the damping wing cross-section is the Lorentzian far-wing:

    .. math::

        \sigma_{\rm DW}(\Delta\nu) = \sigma_0 \frac{\Gamma_{\rm Ly\alpha}/(4\pi)}{(\Delta\nu)^2 + [\Gamma_{\rm Ly\alpha}/(4\pi)]^2}

    with :math:`\sigma_0 = 5.9 \times 10^{-14}` cm²·Hz and
    :math:`\Gamma_{\rm Ly\alpha} = 6.265 \times 10^8` s⁻¹.

    **Upstream**: Asada et al. (2025) damping wing model for the epoch of reionization.
    """
    # Sigmoid column density evolution: N_HI rises steeply above z_mid
    n_hi = (10.0**log_nhi) / (1.0 + jnp.exp(-(z_source - z_mid) / dz))

    # Observed Ly-alpha wavelength at source redshift
    lya_obs = _LAMBDA_LYA * (1.0 + z_source)

    # Frequency offset from Ly-alpha at the source
    # nu_obs = c / (wave_obs * 1e-8), nu_lya_obs = c / (lya_obs * 1e-8)
    # Delta_nu = nu_obs - nu_lya_obs (positive = blueward of Ly-alpha)
    nu_obs = C_CGS / (wave_obs * 1e-8)
    nu_lya_obs = C_CGS / (lya_obs * 1e-8)
    delta_nu = nu_obs - nu_lya_obs

    # Damping wing cross-section (Lorentzian far-wing approximation)
    # sigma_DW = sigma_0 * (gamma / (4*pi)) / (delta_nu^2 + (gamma/(4*pi))^2)
    # In the far wing (|delta_nu| >> gamma/4pi), this simplifies to
    # sigma_DW ~ sigma_0 * gamma / (4*pi*delta_nu^2)
    # We use the full Lorentzian for numerical stability near line center.
    gamma_4pi = _GAMMA_LYA / (4.0 * jnp.pi)
    sigma_dw = _SIGMA_0 * gamma_4pi / (delta_nu**2 + gamma_4pi**2)

    # Optical depth: only apply redward of Ly-alpha at source (damping wing)
    # and only for wavelengths near Ly-alpha (within ~200 A observed)
    tau = n_hi * sigma_dw

    # Only absorb redward of Ly-alpha at source redshift (wave_obs > lya_obs)
    # The damping wing is the red wing absorption from the CGM
    tau = jnp.where(wave_obs > lya_obs, tau, 0.0)

    # Only apply at z > 5 (below this, CGM is ionized and negligible)
    tau = jnp.where(z_source > 5.0, tau, 0.0)

    return jnp.clip(tau, min=0.0)


# ── Public API ────────────────────────────────────────────────────


def igm_transmission(
    wave_obs: jnp.ndarray,
    z_source: float,
    add_cgm: bool = False,
    cgm_z_mid: float = 7.0,
    cgm_dz: float = 0.5,
    cgm_log_nhi: float = 21.0,
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
    lya_obs = _LAMBDA_LYA * (1.0 + z)

    # Dimensionless wavelength offset: x = lambda_obs/lya_obs - 1
    # x > 0 is redward of Lya (damping wing side)
    x_wave = wave_obs / lya_obs - 1.0

    # Damping constant: Lambda = Gamma_alpha / (4*pi*nu_alpha)
    # nu_alpha = c / (lambda_alpha * 1e-8)
    lambda_alpha = _LAMBDA_LYA * 1e-8  # cm
    nu_alpha = C_CGS / lambda_alpha
    lambda_damp = _GAMMA_LYA / (4.0 * jnp.pi * nu_alpha)

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
    # tau(x) = tau_GP * x_HI * Lambda / pi * [1/x_bubble - 1/x]
    # for x > x_bubble (outside the bubble on the red side).
    #
    # The 1/x_bubble term gives the total column from the bubble edge,
    # and 1/x corrects for the integration starting point.
    # Use soft clipping for differentiability.
    x_safe = jnp.maximum(x_wave, x_bubble + 1e-10)
    tau_wing = (
        tau_GP * x_HI * lambda_damp / jnp.pi * (1.0 / jnp.maximum(x_bubble, 1e-10) - 1.0 / x_safe)
    )

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

    Ported from Prospector ``add_igm`` in ``fake_fsps.py`` (Johnson+2021 [2]_).
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
