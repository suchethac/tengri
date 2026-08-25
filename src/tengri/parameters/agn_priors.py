# SPDX-License-Identifier: BSD-3-Clause
"""Informative AGN prior penalty terms for the log-prior.

These functions provide opt-in soft (Gaussian) penalty terms that can be added
to a model's log-prior to encode domain knowledge about AGN SED properties.
Unlike per-parameter priors (Uniform, Gaussian, etc.), these are composite
penalties linking multiple AGN components.

Each function returns a scalar log-prior contribution (≤ 0) suitable for
summation into the total log-prior via::

    log_prior_total = log_prior_params + agn_prior_energy_balance(...)
        + agn_prior_agn_fraction_floor(...) + agn_prior_midir_uv_tie(...)

Implements the same AGN prior penalty functions as AGNfitter (Calistro
Rivera et al. 2016, ApJ 833, 98, arXiv:1606.05648) and AGNfitter-rX
(Martínez-Ramírez / Zhuang et al. 2024, arXiv:2405.12111); validated
against the corresponding branches of ``functions/PRIORS_AGNfitter.py``.

All functions are JAX-compatible and JIT/grad-safe.
"""

from __future__ import annotations

import jax.numpy as jnp


def agn_prior_energy_balance(
    l_gal_att: jnp.ndarray,
    l_sb_emit: jnp.ndarray,
    tolerance: str = "flexible",
) -> jnp.ndarray:
    """Soft Gaussian penalty enforcing galaxy absorbed ≈ starburst IR emission.

    Encodes the physical constraint that dust absorbed by the galaxy should be
    re-emitted in the infrared by star formation (e.g., starburst component).
    This prevents unphysical models where the attenuated luminosity exceeds the
    reprocessed luminosity.

    Parameters
    ----------
    l_gal_att : float
        Log10 of galaxy attenuated luminosity [erg/s].
    l_sb_emit : float
        Log10 of starburst IR emission luminosity [erg/s].
    tolerance : str, optional
        Prior type: "flexible" or "restrictive".

        - "flexible": soft Gaussian around equality (σ=0.1).
        - "restrictive": narrower Gaussian around equality (σ=0.1,
          returns -inf if emission < absorption).

        Default: "flexible".

    Returns
    -------
    float
        Log-prior penalty contribution (≤ 0). Returns -inf only in
        restrictive mode when absorption exceeds emission.

    Notes
    -----
    **JIT-compatible**: yes.
    **Grad-compatible**: yes (smooth Gaussian).

    The penalty is applied in log-space as::

        log(L_sb_emit / L_gal_att) ~ N(0, σ²)

    where σ = 0.1 (AGNfitter convention). Both modes use a Gaussian
    centered at equality; "flexible" mode has no hard constraint
    (returns penalty even when unphysical), while "restrictive" mode
    returns -inf for emission < absorption.

    References
    ----------
    .. [1] G. Calistro Rivera et al., "AGNfitter: A Bayesian MCMC Approach
       to Fitting Spectral Energy Distributions of AGNs," ApJ 833, 98
       (2016). arXiv:1606.05648. DOI:10.3847/1538-4357/833/1/98. The
       σ = 0.1 constant and the ``flexible`` / ``restrictive`` branch
       structure follow ``prior_energy_balance`` in
       ``functions/PRIORS_AGNfitter.py``.
    .. [2] L. N. Martínez-Ramírez et al., "AGNFITTER-RX: Modeling the
       radio-to-X-ray spectral energy distributions of AGNs," A&A, 688, A46
       (2024). arXiv:2405.12111. DOI:10.1051/0004-6361/202449329; the
       extended AGNfitter-rX branch in which this prior is retained.
    """
    # Ensure JAX arrays
    l_sb_emit = jnp.asarray(l_sb_emit)
    l_gal_att = jnp.asarray(l_gal_att)
    frac_ratio = l_sb_emit - l_gal_att  # log10(L_sb_emit / L_gal_att)

    # Gaussian penalty σ=0.1 (from AGNfitter PRIORS_AGNfitter.py line 104)
    mu = 0.0
    sigma = 0.1
    gaussian_term = -0.5 * ((frac_ratio - mu) / sigma) ** 2

    if tolerance == "flexible":
        # Always apply Gaussian penalty; no hard cutoff
        return gaussian_term
    elif tolerance == "restrictive":
        # Apply Gaussian but return -inf if emission < absorption
        is_physical = frac_ratio >= 0.0
        return jnp.where(is_physical, gaussian_term, -jnp.inf)
    else:
        raise ValueError(f"tolerance must be 'flexible' or 'restrictive', got {tolerance}")


def agn_prior_agn_fraction_floor(
    l_agn: jnp.ndarray,
    l_galaxy: jnp.ndarray,
    floor: float = 0.01,
) -> jnp.ndarray:
    """Soft penalty enforcing a minimum AGN fraction (mid-IR window).

    Encodes empirical constraints on the AGN/galaxy luminosity ratio in the
    mid-infrared, penalizing models where the AGN contributes below a typical
    observational floor.

    Parameters
    ----------
    l_agn : float
        Log10 of AGN luminosity [erg/s] (e.g., hot torus or accretion disk).
    l_galaxy : float
        Log10 of galaxy (stellar+dust) luminosity [erg/s].
    floor : float, optional
        Minimum AGN fraction (linear, not log). Default: 0.01 (1%).

    Returns
    -------
    float
        Log-prior penalty contribution (≤ 0). Smooth Gaussian with σ = 0.5
        (AGNfitter-style clipped Gaussian, with a floor).

    Notes
    -----
    **JIT-compatible**: yes.
    **Grad-compatible**: yes (smooth Gaussian).

    The AGN fraction is computed as::

        f_agn = L_agn / (L_agn + L_galaxy)

    The penalty is applied as a Gaussian with σ = 0.5 in log-space,
    penalizing deviations of the AGN fraction from the floor value.

    References
    ----------
    .. [1] G. Calistro Rivera et al., "AGNfitter: A Bayesian MCMC Approach
       to Fitting Spectral Energy Distributions of AGNs," ApJ 833, 98
       (2016). arXiv:1606.05648. The σ = 0.5 value and the floor-Gaussian
       construction follow ``prior_AGNfraction`` in
       ``functions/PRIORS_AGNfitter.py``.
    """
    # Compute f_agn = L_agn / (L_agn + L_galaxy) in log space
    # Using logaddexp for numerical stability
    log_total = jnp.logaddexp(l_agn * jnp.log(10.0), l_galaxy * jnp.log(10.0)) / jnp.log(10.0)
    log_f_agn = l_agn - log_total

    # Gaussian penalty with σ = 0.5, centered at floor (from AGNfitter line 416)
    mu = jnp.log10(jnp.asarray(floor))
    sigma = 0.5
    penalty = -0.5 * ((log_f_agn - mu) / sigma) ** 2

    return penalty


def agn_prior_midir_uv_tie(
    l_mir_torus: jnp.ndarray,
    l_uv_disc: jnp.ndarray,
) -> jnp.ndarray:
    """Soft Gaussian penalty linking mid-IR (torus) and UV (accretion disc).

    Encodes physical correlations between the accretion disc (UV) and dusty
    torus (mid-IR) luminosities. These components are coupled via accretion
    rate and inclination; this prior penalizes deviations from observed
    mid-IR–UV luminosity ratios.

    Parameters
    ----------
    l_mir_torus : float
        Log10 of torus mid-IR emission at ~6 microns [erg/s].
    l_uv_disc : float
        Log10 of accretion disc UV emission [erg/s].

    Returns
    -------
    float
        Log-prior penalty contribution (≤ 0). Gaussian penalty with σ = 0.6
        (from AGNfitter line 354, combining mid-IR–Xray σ=0.5 and α_OX σ=0.1).

    Notes
    -----
    **JIT-compatible**: yes.
    **Grad-compatible**: yes (smooth Gaussian).

    The penalty is applied in log-space as::

        log(L_mir / L_uv) ~ N(0, σ²)

    This enforces a typical ratio of mid-IR to UV luminosities observed in
    AGN SEDs. The σ = 0.6 accounts for both intrinsic scatter and
    observational uncertainty.

    References
    ----------
    .. [1] G. Calistro Rivera et al., "AGNfitter: A Bayesian MCMC Approach
       to Fitting Spectral Energy Distributions of AGNs," ApJ 833, 98
       (2016). arXiv:1606.05648. The σ = 0.6 combined-scatter value
       follows ``prior_midIR_UV`` in ``functions/PRIORS_AGNfitter.py``,
       which absorbs the mid-IR–X-ray σ ≈ 0.5 and the α_OX σ ≈ 0.1.
    .. [2] D. W. Just et al., "The X-Ray Properties of the Most Luminous
       Quasars from the Sloan Digital Sky Survey," ApJ 665, 1004 (2007).
       arXiv:0706.4514. Source of the α_OX–L_2500 correlation used in
       the X-ray leg of this prior.
    """
    # Log-ratio of luminosities (ensure JAX arrays)
    l_mir_torus = jnp.asarray(l_mir_torus)
    l_uv_disc = jnp.asarray(l_uv_disc)
    ratio = l_mir_torus - l_uv_disc

    # Gaussian penalty (μ=0, σ=0.6 from AGNfitter line 354)
    mu = 0.0
    sigma = 0.6
    penalty = -0.5 * ((ratio - mu) / sigma) ** 2

    return penalty
