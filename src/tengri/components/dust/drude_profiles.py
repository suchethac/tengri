# SPDX-License-Identifier: BSD-3-Clause
"""PAH Drude-profile decomposition (Smith+2007).

This module provides a public interface for Polycyclic Aromatic Hydrocarbon
(PAH) feature modeling using Drude profiles, following Smith et al. (2007,
ApJ 656 770).  It also exposes a least-squares fitter that recovers per-feature
strengths from an observed mid-IR SED.

References
----------
Smith et al. 2007, ApJ 656, 770: PAHFIT; Table 2 feature list.

"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax.typing import ArrayLike

# ── Feature table ─────────────────────────────────────────────────


class PAHFeature(NamedTuple):
    """Parameters of a single PAH Drude profile.

    Attributes
    ----------
    wave_um: float
        Central wavelength in microns.
    gamma: float
        Fractional FWHM (Δλ/λ₀).  Absolute FWHM = gamma * wave_um.
    strength: float
        Relative strength (Smith+2007 PAHFIT SINGS median; 7.60 μm = 1.0).

    Notes
    -----
    Used to define the 18-feature PAH template from Smith et al. (2007).
    Each feature can be evaluated as a Drude profile via
    :func:`drude_profile` or combined via :func:`pah_template`.

    """

    wave_um: float
    gamma: float
    strength: float


# 18-entry table: 3.3 μm C-H stretch + 17 features from Smith+2007 Table 2.
# Format: PAHFeature(center_um, fractional_gamma, relative_strength).
# The 3.3 μm feature lies outside IRS coverage in Smith+2007: parameters
# from Tokunaga+1991 and Li & Draine 2001.
SMITH2007_PAH_FEATURES: tuple[PAHFeature, ...] = (
    PAHFeature(3.30, 0.0152, 0.06),  # 3.3 μm C-H stretch
    PAHFeature(5.27, 0.034, 0.04),  # Smith+2007 Table 2
    PAHFeature(5.70, 0.035, 0.03),
    PAHFeature(6.22, 0.030, 0.30),
    PAHFeature(6.69, 0.070, 0.02),
    PAHFeature(7.42, 0.126, 0.15),
    PAHFeature(7.60, 0.044, 1.00),  # strongest feature (reference)
    PAHFeature(7.85, 0.053, 0.45),
    PAHFeature(8.33, 0.050, 0.05),
    PAHFeature(8.61, 0.039, 0.33),
    PAHFeature(10.68, 0.020, 0.02),
    PAHFeature(11.23, 0.012, 0.20),
    PAHFeature(11.33, 0.032, 0.45),
    PAHFeature(11.99, 0.045, 0.05),
    PAHFeature(12.62, 0.042, 0.15),
    PAHFeature(12.69, 0.013, 0.05),
    PAHFeature(13.48, 0.040, 0.03),
    PAHFeature(14.04, 0.016, 0.02),
)

# Number of features: convenience constant.
N_PAH_FEATURES: int = len(SMITH2007_PAH_FEATURES)

# Pre-packed JAX arrays (module-level, JIT-compatible).
_CENTERS_UM = jnp.array([f.wave_um for f in SMITH2007_PAH_FEATURES])
_GAMMAS = jnp.array([f.gamma for f in SMITH2007_PAH_FEATURES])
_FWHMS_UM = _CENTERS_UM * _GAMMAS  # absolute FWHM in microns
_DEFAULT_STRENGTHS = jnp.array([f.strength for f in SMITH2007_PAH_FEATURES])


# ── Core profile functions ────────────────────────────────────────


def drude_profile(
    wave_um: ArrayLike,
    wave0_um: float,
    gamma: float,
) -> jnp.ndarray:
    """Single Drude profile for one PAH feature.

    The Drude profile is::

        D(λ) = γ² / ((λ/λ₀ - λ₀/λ)² + γ²)

    where γ = Δλ/λ₀ is the fractional FWHM.  Peak value is unity at λ = λ₀;
    FWHM in wavelength space ≈ γ * λ₀ (exact for small γ).

    Parameters
    ----------
    wave_um: array_like, shape (n_wave,)
        Wavelength grid [μm].
    wave0_um: float
        Central wavelength [μm].
    gamma: float
        Fractional FWHM Δλ/λ₀ [dimensionless].

    Returns
    -------
    jnp.ndarray, shape (n_wave,)
        Dimensionless Drude profile normalized to peak = 1.

    Notes
    -----
    **JIT-compatible**: yes.

    **Gradient-safe**: yes.

    """
    lam = jnp.asarray(wave_um)
    # gamma is the fractional FWHM; g_over_l0 = gamma is already Δλ/λ₀.
    # Smith+2007 Eq. 1: D(λ) = γ² / ((λ/λ₀ - λ₀/λ)² + γ²).
    x = lam / wave0_um - wave0_um / lam
    return gamma**2 / (x**2 + gamma**2)


def compute_pah_template(
    wave_um: ArrayLike,
    strengths: ArrayLike | None = None,
) -> jnp.ndarray:
    """Sum of all 18 PAH Drude profiles weighted by feature strengths.

    Returns the combined profile in *wavelength-space* (L_lambda convention).
    To convert to L_nu: multiply by ``λ² / c``.

    Parameters
    ----------
    wave_um: array_like, shape (n_wave,)
        Wavelength grid [μm].
    strengths: array_like, shape (18,) or None
        Per-feature amplitude weights [dimensionless].  If ``None``, the
        Smith+2007 SINGS median strengths (``SMITH2007_PAH_FEATURES``) are used.

    Returns
    -------
    jnp.ndarray, shape (n_wave,)
        Dimensionless continuum-normalized PAH shape (non-negative everywhere).

    Notes
    -----
    **JIT-compatible**: yes.

    **Gradient-safe**: yes.

    """
    lam = jnp.asarray(wave_um)
    s = _DEFAULT_STRENGTHS if strengths is None else jnp.asarray(strengths)

    # Vectorized broadcast: (n_wave, 1) × (1, n_feat)
    lam2d = lam[:, None]
    lam0 = _CENTERS_UM[None, :]
    g_over_l0 = (_FWHMS_UM / _CENTERS_UM)[None, :]  # (1, n_feat)
    x = lam2d / lam0 - lam0 / lam2d
    profiles = s[None, :] * g_over_l0**2 / (x**2 + g_over_l0**2)
    return jnp.sum(profiles, axis=1)


# ── Decomposition ─────────────────────────────────────────────────


def decompose_pah(
    wave_um: ArrayLike,
    sed: ArrayLike,
    *,
    continuum: ArrayLike | None = None,
) -> dict[str, jnp.ndarray]:
    """Project an observed mid-IR SED onto the PAH Drude basis.

    Solves the unconstrained linear least-squares problem::

        min_s  ‖ (sed - continuum) - A @ s ‖²

    where ``A`` is the ``(n_wave, 18)`` design matrix of peak-normalized
    Drude profiles.  The solution is computed via the normal equations and
    is fully JAX-differentiable: pass the returned ``strengths`` directly
    into the tengri fitter or use them as a warm-start for ``pah_template``.

    For non-negative strengths, use the returned ``strengths`` as free
    parameters in a fitter with ``Uniform(0, ...)`` priors.

    Parameters
    ----------
    wave_um: array_like, shape (n_wave,)
        Wavelength grid [μm].
    sed: array_like, shape (n_wave,)
        Observed (or model) SED [arbitrary units].
    continuum: array_like, shape (n_wave,) or None
        Underlying continuum to subtract before fitting [same units as ``sed``].
        If ``None``, the raw ``sed`` is used.

    Returns
    -------
    dict with keys:
        ``"strengths"`` : ndarray, shape (18,), fitted feature amplitudes
            [dimensionless].
        ``"fitted_pah"``: ndarray, shape (n_wave,), reconstructed PAH profile
            [same units as sed].
        ``"residual"``  : ndarray, shape (n_wave,),
            (sed - continuum) - fitted_pah [same units as sed].

    References
    ----------
    .. [1] D. M. Smith et al., "The Mid-Infrared Emission of Ultraluminous
           Infrared Galaxies," ApJ, 656, 770 (2007). arXiv:astro-ph/0701042.
           https://doi.org/10.1086/510378

    Notes
    -----
    **JIT-compatible**: yes.

    **Gradient-safe**: yes. Use for warm-starting or sensitivity analysis.

    """
    lam = jnp.asarray(wave_um)
    y = jnp.asarray(sed)
    if continuum is not None:
        y = y - jnp.asarray(continuum)

    # Build design matrix A: shape (n_wave, n_feat)
    lam2d = lam[:, None]
    lam0 = _CENTERS_UM[None, :]
    gamma = _GAMMAS[None, :]
    x = lam2d / lam0 - lam0 / lam2d
    A = gamma**2 / (x**2 + gamma**2)  # (n_wave, n_feat), peak=1

    # Normal equations: s = (AᵀA)⁻¹ Aᵀ y
    s = jnp.linalg.lstsq(A, y, rcond=None)[0]

    fitted_pah = A @ s
    residual = y - fitted_pah

    return {
        "strengths": s,
        "fitted_pah": fitted_pah,
        "residual": residual,
    }
