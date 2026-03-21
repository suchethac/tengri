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

import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Lyman series wavelengths (Angstrom) for lines j=2 (Ly-alpha) to j=40
# ---------------------------------------------------------------------------
_N_LINES = 39

# Rest-frame wavelengths of Lyman series lines (Angstrom)
_LAMBDA_LYMAN = jnp.array(
    [
        1215.670,
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
_LAMBDA_LIMIT = 911.8  # Angstrom

# ---------------------------------------------------------------------------
# LAF coefficients: A_j^LAF for 3 regimes (Inoue+2014 Eq. 21)
# Shape: (39, 3) — [A_j1, A_j2, A_j3]
# From eazy-py LAFcoeff.txt
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# DLA coefficients: A_j^DLA for 2 regimes (Inoue+2014 Eq. 22)
# Shape: (39, 2) — [A_j1, A_j2]
# From eazy-py DLAcoeff.txt
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Lyman series optical depth (LAF)
# ---------------------------------------------------------------------------


def _tau_ls_laf(
    wave_obs: jnp.ndarray,
    z_source: float,
) -> jnp.ndarray:
    """Lyman-series LAF optical depth (Inoue+2014 Eq. 21).

    Three piecewise power-law regimes per line, with exponents 1.2, 3.7, 5.5.
    Vectorized over all 39 Lyman lines (no Python loop).
    """
    # Broadcast: wave_obs (n_wave,) vs _LAMBDA_LYMAN (39,)
    # Shapes: lam_j (39, 1), wave (1, n_wave) -> (39, n_wave)
    lam_j = _LAMBDA_LYMAN[:, None]  # (39, 1)
    wave = wave_obs[None, :]  # (1, n_wave)

    lam_max = lam_j * (1.0 + z_source)
    active = (wave > lam_j) & (wave < lam_max)

    lam_break1 = 2.2 * lam_j
    lam_break2 = 5.7 * lam_j

    ratio = wave / lam_j  # (39, n_wave)
    t1 = _A_LAF[:, 0:1] * ratio**1.2
    t2 = _A_LAF[:, 1:2] * ratio**3.7
    t3 = _A_LAF[:, 2:3] * ratio**5.5

    t_j = jnp.where(wave < lam_break1, t1, jnp.where(wave < lam_break2, t2, t3))
    return jnp.sum(jnp.where(active, t_j, 0.0), axis=0)


# ---------------------------------------------------------------------------
# Lyman series optical depth (DLA)
# ---------------------------------------------------------------------------


def _tau_ls_dla(
    wave_obs: jnp.ndarray,
    z_source: float,
) -> jnp.ndarray:
    """Lyman-series DLA optical depth (Inoue+2014 Eq. 22).

    Two piecewise regimes per line, with exponents 2.0, 3.0.
    Vectorized over all 39 Lyman lines (no Python loop).
    """
    lam_j = _LAMBDA_LYMAN[:, None]  # (39, 1)
    wave = wave_obs[None, :]  # (1, n_wave)

    lam_max = lam_j * (1.0 + z_source)
    active = (wave > lam_j) & (wave < lam_max)

    lam_break = 3.0 * lam_j
    ratio = wave / lam_j

    t1 = _A_DLA[:, 0:1] * ratio**2.0
    t2 = _A_DLA[:, 1:2] * ratio**3.0

    t_j = jnp.where(wave < lam_break, t1, t2)
    return jnp.sum(jnp.where(active, t_j, 0.0), axis=0)


# ---------------------------------------------------------------------------
# Lyman continuum optical depth (LAF)
# ---------------------------------------------------------------------------


def _tau_lc_laf(
    wave_obs: jnp.ndarray,
    z_source: float,
) -> jnp.ndarray:
    """Lyman-continuum LAF optical depth (Inoue+2014 Eqs. 25-27)."""
    # Absorbers at redshift z_abs contribute for wave_obs = 911.8*(1+z_abs)
    # So wave_obs must be > 911.8 (rest Lyman limit) and < 911.8*(1+z_source)
    active = (wave_obs > _LAMBDA_LIMIT) & (wave_obs < _LAMBDA_LIMIT * (1.0 + z_source))

    z_obs = wave_obs / _LAMBDA_LIMIT - 1.0

    # Three source-redshift regimes
    # Regime z_S < 1.2
    t_low = (
        0.325 * ((1.0 + z_obs) ** 1.2 - jnp.clip(1.0 + z_source, a_max=2.2) ** 1.2)
        - 9.4e-2 * ((1.0 + z_obs) ** 3.7 - jnp.clip(1.0 + z_source, a_max=2.2) ** 3.7)
        + 0.01478 * ((1.0 + z_obs) ** 5.5 - jnp.clip(1.0 + z_source, a_max=2.2) ** 5.5)
    )

    # Regime 1.2 <= z_S < 4.7
    t_mid = (
        2.55e-2 * ((1.0 + z_obs) ** 1.2 - (1.0 + z_source) ** 1.2)
        - 0.325 * ((1.0 + z_obs) ** 1.2 - jnp.clip(1.0 + z_source, a_max=2.2) ** 1.2)
        - 1.15e-2 * ((1.0 + z_obs) ** 3.7 - jnp.clip(1.0 + z_source, a_max=5.7) ** 3.7)
        + 9.4e-2 * ((1.0 + z_obs) ** 3.7 - jnp.clip(1.0 + z_source, a_max=2.2) ** 3.7)
        - 7.83e-4 * ((1.0 + z_obs) ** 5.5 - jnp.clip(1.0 + z_source, a_max=5.7) ** 5.5)
        + 0.01478 * ((1.0 + z_obs) ** 5.5 - jnp.clip(1.0 + z_source, a_max=2.2) ** 5.5)
    )

    # Regime z_S >= 4.7
    t_high = (
        5.22e-4 * ((1.0 + z_obs) ** 1.2 - (1.0 + z_source) ** 1.2)
        + 2.55e-2 * ((1.0 + z_obs) ** 1.2 - (1.0 + z_source) ** 1.2)
        - 0.325 * ((1.0 + z_obs) ** 1.2 - jnp.clip(1.0 + z_source, a_max=2.2) ** 1.2)
        - 1.328e-3 * ((1.0 + z_obs) ** 3.7 - (1.0 + z_source) ** 3.7)
        - 1.15e-2 * ((1.0 + z_obs) ** 3.7 - jnp.clip(1.0 + z_source, a_max=5.7) ** 3.7)
        + 9.4e-2 * ((1.0 + z_obs) ** 3.7 - jnp.clip(1.0 + z_source, a_max=2.2) ** 3.7)
        - 5.15e-5 * ((1.0 + z_obs) ** 5.5 - (1.0 + z_source) ** 5.5)
        - 7.83e-4 * ((1.0 + z_obs) ** 5.5 - jnp.clip(1.0 + z_source, a_max=5.7) ** 5.5)
        + 0.01478 * ((1.0 + z_obs) ** 5.5 - jnp.clip(1.0 + z_source, a_max=2.2) ** 5.5)
    )

    tau = jnp.where(
        z_source < 1.2,
        t_low,
        jnp.where(z_source < 4.7, t_mid, t_high),
    )

    return jnp.where(active, jnp.clip(tau, a_min=0.0), 0.0)


# ---------------------------------------------------------------------------
# Lyman continuum optical depth (DLA)
# ---------------------------------------------------------------------------


def _tau_lc_dla(
    wave_obs: jnp.ndarray,
    z_source: float,
) -> jnp.ndarray:
    """Lyman-continuum DLA optical depth (Inoue+2014 Eqs. 28-29)."""
    active = (wave_obs > _LAMBDA_LIMIT) & (wave_obs < _LAMBDA_LIMIT * (1.0 + z_source))
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
    return jnp.where(active, jnp.clip(tau, a_min=0.0), 0.0)


# ---------------------------------------------------------------------------
# CGM damping wing absorption (Asada et al. 2025)
# ---------------------------------------------------------------------------

# Physical constants for damping wing calculation
_C_CGS_IGM = 2.99792458e10    # cm/s
_LAMBDA_LYA = 1215.67         # Angstrom (Ly-alpha rest wavelength)
_NU_LYA = _C_CGS_IGM / (_LAMBDA_LYA * 1e-8)  # Hz
_GAMMA_LYA = 6.265e8          # s^-1 (Ly-alpha natural line width)
_SIGMA_0 = 5.9e-14            # cm^2 Hz (Ly-alpha cross-section constant: pi*e^2/(m_e*c)*f_12)


def _cgm_damping_wing_tau(
    wave_obs: jnp.ndarray,
    z_source: float,
    z_mid: float = 7.0,
    dz: float = 0.5,
    log_nhi: float = 20.0,
) -> jnp.ndarray:
    """CGM damping wing optical depth (Asada et al. 2025).

    At z > 6, neutral hydrogen in the circumgalactic medium causes
    Ly-alpha damping wing absorption that the Inoue+2014 model does
    not capture. The damping wing profile is the Lorentzian far-wing
    of the Ly-alpha cross-section.

    tau_DW(lambda) = N_HI(z) * sigma_DW(lambda, z)

    where N_HI follows a sigmoid evolution with redshift and sigma_DW
    is the damping wing cross-section near Ly-alpha.

    Parameters
    ----------
    wave_obs : array, shape (n_wave,)
        Observed-frame wavelength in Angstrom.
    z_source : float
        Source redshift.
    z_mid : float
        Midpoint redshift of the sigmoid N_HI evolution. Default 7.0.
    dz : float
        Width of the sigmoid transition. Default 0.5.
    log_nhi : float
        log10(N_HI / cm^-2) at the plateau. Default 20.0.

    Returns
    -------
    array, shape (n_wave,)
        Damping wing optical depth (>= 0).
    """
    # Sigmoid column density evolution: N_HI rises steeply above z_mid
    n_hi = (10.0 ** log_nhi) / (1.0 + jnp.exp(-(z_source - z_mid) / dz))

    # Observed Ly-alpha wavelength at source redshift
    lya_obs = _LAMBDA_LYA * (1.0 + z_source)

    # Frequency offset from Ly-alpha at the source
    # nu_obs = c / (wave_obs * 1e-8), nu_lya_obs = c / (lya_obs * 1e-8)
    # Delta_nu = nu_obs - nu_lya_obs (positive = blueward of Ly-alpha)
    nu_obs = _C_CGS_IGM / (wave_obs * 1e-8)
    nu_lya_obs = _C_CGS_IGM / (lya_obs * 1e-8)
    delta_nu = nu_obs - nu_lya_obs

    # Damping wing cross-section (Lorentzian far-wing approximation)
    # sigma_DW = sigma_0 * (gamma / (4*pi)) / (delta_nu^2 + (gamma/(4*pi))^2)
    # In the far wing (|delta_nu| >> gamma/4pi), this simplifies to
    # sigma_DW ~ sigma_0 * gamma / (4*pi*delta_nu^2)
    # We use the full Lorentzian for numerical stability near line center.
    gamma_4pi = _GAMMA_LYA / (4.0 * jnp.pi)
    sigma_dw = _SIGMA_0 * gamma_4pi / (delta_nu ** 2 + gamma_4pi ** 2)

    # Optical depth: only apply redward of Ly-alpha at source (damping wing)
    # and only for wavelengths near Ly-alpha (within ~200 A observed)
    tau = n_hi * sigma_dw

    # Only absorb redward of Ly-alpha at source redshift (wave_obs > lya_obs)
    # The damping wing is the red wing absorption from the CGM
    tau = jnp.where(wave_obs > lya_obs, tau, 0.0)

    # Only apply at z > 5 (below this, CGM is ionized and negligible)
    tau = jnp.where(z_source > 5.0, tau, 0.0)

    return jnp.clip(tau, a_min=0.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def igm_transmission(
    wave_obs: jnp.ndarray,
    z_source: float,
    add_cgm: bool = True,
    cgm_z_mid: float = 7.0,
    cgm_dz: float = 0.5,
    cgm_log_nhi: float = 20.0,
) -> jnp.ndarray:
    """Compute mean IGM transmission T_IGM(lambda_obs, z_source).

    Implements the Inoue et al. (2014) prescription for the mean
    intergalactic medium absorption from the Ly-alpha forest and
    damped Ly-alpha systems. Optionally adds CGM damping wing
    absorption at z > 5 following Asada et al. (2025).

    Parameters
    ----------
    wave_obs : array, shape (n_wave,)
        Observed-frame wavelength in Angstrom.
    z_source : float
        Source redshift.
    add_cgm : bool
        If True (default), add CGM damping wing absorption at z > 5.
    cgm_z_mid : float
        Midpoint redshift of the sigmoid N_HI evolution. Default 7.0.
    cgm_dz : float
        Width of the sigmoid transition. Default 0.5.
    cgm_log_nhi : float
        log10(N_HI / cm^-2) at the plateau. Default 20.0.

    Returns
    -------
    array, shape (n_wave,)
        Transmission factor T in [0, 1]. Multiply rest-frame SED by
        T to get absorbed spectrum.
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

    return jnp.exp(-jnp.clip(tau_total, a_min=0.0))
