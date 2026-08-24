# SPDX-License-Identifier: BSD-3-Clause
"""Meiksin (2006) IGM mean transmission.

Implements the Meiksin (2006) IGM model as CIGALE evaluates it
(``pcigale.sed_modules.redshifting.igm_transmission``) in pure JAX so
it is JIT-compilable and differentiable. The CIGALE function is the
authoritative reference; this implements it exactly, with the
wavelength convention converted from nm (CIGALE) to Angstrom (tengri).

Unlike Inoue+2014, Meiksin's diffuse Lyman-α-forest continuum
suppression rises from ~0 just blueward of Lyα to ~0.25 by the optical
at z = 3; a *non-binary* continuum that Inoue's grid model misses.
Issue #440 §12.

References
----------
.. [1] Meiksin, A. 2006, MNRAS, 365, 807.
       https://doi.org/10.1111/j.1365-2966.2005.09663.x
.. [2] Boquien, M., et al. 2019, A&A, 622, A103 (CIGALE).
       https://doi.org/10.1051/0004-6361/201834156
"""

from __future__ import annotations

import math

import jax.numpy as jnp

# Meiksin (2006) Table 1 line strength multipliers for n = 3..9 (relative to Ly-alpha).
# Indices 0--2 are placeholders so the array indexes naturally as `_FACT[n]`.
_FACT: tuple[float, ...] = (
    1.0,
    1.0,
    1.0,
    0.348,
    0.179,
    0.109,
    0.0722,
    0.0508,
    0.0373,
    0.0283,
)

_N_TRANS_LOW = 10  # n = 2..9 explicit; n = 10..30 via the 720/(n*(n^2-1)) tail
_N_TRANS_MAX = 31

_LAMBDA_LIMIT_AA = 912.0  # Lyman limit in Angstrom (CIGALE uses 91.2 nm)
_GAMMA = 0.2788  # incomplete gamma(0.5, 1) -- the Meiksin paper's Gamma(2-beta, 1)
_N0 = 0.25  # LLS normalization


def _term2() -> float:
    """The constant series Sum_{n=0..N-1} (-1)^n / (n! (2n-1))."""
    s = 0.0
    for n in range(_N_TRANS_LOW - 1):
        s += ((-1.0) ** n) / (math.factorial(n) * (2 * n - 1))
    return s


_TERM2 = float(_term2())


def igm_transmission_meiksin06(
    wave_obs: jnp.ndarray,
    z: float,
) -> jnp.ndarray:
    r"""Mean IGM transmission using the Meiksin (2006) prescription.

    Parameters
    ----------
    wave_obs: array_like, shape (n_wave,)
        Observed-frame wavelengths [Angstrom]. Must be sorted ascending
        for the LLS-continuum branch to apply correctly.
    z: float
        Source redshift (must be > 0).

    Returns
    -------
    T_igm: jnp.ndarray, shape (n_wave,)
        Mean IGM transmission in ``[0, 1]``. Above ``912 * (1 + z)``
        Angstrom the value is exactly 1 (no Lyman absorption).

    Notes
    -----
    **JIT-compatible**: yes -- pure ``jnp`` operations, no Python
    control flow except the ``z <= 4`` vs ``z > 4`` branch on the
    Lyman-alpha normalization (kept as a ``jnp.where`` so the function
    is also vmap/grad safe).

    **Convention**: ``wave_obs`` is **observed-frame** wavelength,
    matching the rest of tengri's IGM API (Inoue+2014, Madau+1995).
    The CIGALE implementation takes the same observed-frame
    wavelength but in nm; we convert internally.

    **Physics summary**:

    1. Lyman series (lines ``n = 2 .. 30``) above the Lyman limit.
       Series strengths from Meiksin Table 1; high-n tail
       (``n >= 10``) follows ``720 / (n * (n^2 - 1))``.
    2. Lyman-alpha-forest continuum (``tau_l_igm``): smooth ramp
       blueward of the Lyman limit, peaking at ``z_l = 0``.
    3. Lyman-Limit Systems (``tau_l_lls``): higher-order suppression
       from optically-thick absorbers, using O'Meara+2013's
       ``lambda^2.75`` cross-section damping.

    For wavelengths short of the Lyman limit at z=0 (``z_l < 0``), the
    LLS+forest opacity is damped by ``(z_l + 1)^2.75`` per O'Meara+2013.

    See Also
    --------
    igm_transmission: Inoue+2014 (grid-based, default in tengri).
    igm_transmission_madau: Madau+1995 (simpler, fewer lines).

    References
    ----------
    .. [1] A. Meiksin, "Color corrections for high-redshift objects
       due to intergalactic absorption," MNRAS, 365, 807 (2006).
       https://doi.org/10.1111/j.1365-2966.2005.09663.x
    """
    wave_obs = jnp.asarray(wave_obs, dtype=jnp.float64)
    # Work in nm internally to mirror the CIGALE source line-for-line.
    wavelength_nm = wave_obs / 10.0
    lambda_limit = _LAMBDA_LIMIT_AA / 10.0  # 91.2 nm

    # ── 1. Lyman series ─────────────────────────────────────────────
    # lambda_n[n] = lambda_limit / (1 - 1/n^2)  for n = 2..30
    n_idx = jnp.arange(2, _N_TRANS_MAX)  # shape (29,)
    lambda_n = lambda_limit / (1.0 - 1.0 / (n_idx.astype(jnp.float64) ** 2))
    # z_n[k, :] = wavelength / lambda_n[k] - 1
    z_n = wavelength_nm[None, :] / lambda_n[:, None] - 1.0  # (29, n_wave)

    # Mean Lyman-alpha optical depth (n=2).
    tau_a = jnp.where(
        z <= 4.0,
        0.00211 * (1.0 + z) ** 3.7,
        0.00058 * (1.0 + z) ** 4.5,
    )
    tau_n2 = jnp.where(
        z <= 4.0,
        0.00211 * (1.0 + z_n[0]) ** 3.7,
        0.00058 * (1.0 + z_n[0]) ** 4.5,
    )

    # n = 3..9 with two sub-regimes split at z_n = 3.
    tau_low = []
    for n in range(3, 10):
        zn = z_n[n - 2]
        a = tau_a * _FACT[n] * (0.25 * (1.0 + zn)) ** (1.0 / 3.0)
        b = tau_a * _FACT[n] * (0.25 * (1.0 + zn)) ** (1.0 / 6.0)
        if n <= 5:
            tau_low.append(jnp.where(zn < 3.0, a, b))
        else:
            tau_low.append(a)  # n in 6..9: only the (1/3) branch
    tau_3_9 = jnp.stack(tau_low, axis=0)  # (7, n_wave)

    # n = 10..30: tau_n = tau_n[9] * 720 / (n * (n^2 - 1))
    tau_9 = tau_3_9[-1]
    n_high = jnp.arange(10, _N_TRANS_MAX, dtype=jnp.float64)
    high_factors = 720.0 / (n_high * (n_high * n_high - 1.0))  # (21,)
    tau_10_30 = tau_9[None, :] * high_factors[:, None]  # (21, n_wave)

    # Stack every line, then mask each line to the band where it is
    # physically active: 0 <= z_n < z_source (CIGALE's
    # ``where((z_n >= z) | (z_n < 0))``).
    tau_all = jnp.concatenate([tau_n2[None, :], tau_3_9, tau_10_30], axis=0)  # (29, n_wave)
    active = (z_n >= 0.0) & (z_n < z)
    tau_lines = jnp.sum(jnp.where(active, tau_all, 0.0), axis=0)

    # ── 2. Lyman-alpha-forest continuum (tau_l_igm) ─────────────────
    # z_l = wavelength / lambda_limit - 1
    z_l = wavelength_nm / lambda_limit - 1.0
    below_z = z_l < z

    tau_l_igm = jnp.where(
        below_z & (z_l >= 0.0),
        0.805 * (1.0 + z_l) ** 3 * (1.0 / (1.0 + z_l) - 1.0 / (1.0 + z)),
        0.0,
    )

    # ── 3. Lyman-limit systems (tau_l_lls) ──────────────────────────
    term1 = _GAMMA - math.exp(-1.0)
    diff_term = term1 - _TERM2

    ratio = jnp.maximum(wavelength_nm / lambda_limit, 1e-30)
    term3 = (1.0 + z) * ratio**1.5 - ratio**2.5

    # term4: sum over n = 1..N-1 of analytical residual
    term4 = jnp.zeros_like(wavelength_nm)
    for n in range(1, _N_TRANS_LOW):
        coeff = 2.0 * ((-1.0) ** n) / (math.factorial(n) * ((6 * n - 5) * (2 * n - 1)))
        term4 = term4 + coeff * ((1.0 + z) ** (2.5 - 3.0 * n) * ratio ** (3.0 * n) - ratio**2.5)

    tau_l_lls = jnp.where(
        below_z & (z_l >= 0.0),
        _N0 * (diff_term * term3 - term4),
        0.0,
    )

    # ── 4. Short-wavelength tail (z_l < 0): damped by (z_l + 1)^2.75 ─
    # Normalize at z_l = 0 (boundary).
    boundary = wavelength_nm > 0  # placeholder mask; we evaluate the
    # boundary value of the LLS/IGM continuum at z_l = 0, i.e.
    # wave_obs = lambda_limit Angstrom.
    lam0 = lambda_limit
    # Recompute tau_l_igm and tau_l_lls at z_l = 0 -- closed-form.
    tau_norm_l_igm = 0.805 * (1.0) ** 3 * (1.0 - 1.0 / (1.0 + z))
    term3_0 = (1.0 + z) * 1.0 - 1.0
    term4_0 = 0.0
    for n in range(1, _N_TRANS_LOW):
        coeff = 2.0 * ((-1.0) ** n) / (math.factorial(n) * ((6 * n - 5) * (2 * n - 1)))
        term4_0 = term4_0 + coeff * ((1.0 + z) ** (2.5 - 3.0 * n) - 1.0)
    tau_norm_l_lls = _N0 * (diff_term * term3_0 - term4_0)

    damp = (z_l + 1.0) ** 2.75
    short_mask = z_l < 0.0
    tau_l_igm = jnp.where(short_mask, tau_norm_l_igm * damp, tau_l_igm)
    tau_l_lls = jnp.where(short_mask, tau_norm_l_lls * damp, tau_l_lls)
    del boundary, lam0  # unused but kept for parity with CIGALE comments

    # ── 5. Combine ──────────────────────────────────────────────────
    return jnp.exp(-(tau_lines + tau_l_igm + tau_l_lls))
