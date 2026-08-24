# SPDX-License-Identifier: BSD-3-Clause
"""Pure JAX, JIT-compatible spectral unit conversion utilities.

All functions take and return bare jnp arrays (no unit objects) in CGS units:

- Luminosity: erg/s
- Flux: erg/s/cm²/Hz or erg/s/cm²/Å
- Wavelength: Ångström (Å)
- Frequency: Hz
- Optical depth / Attenuation: dimensionless

The conversion formulae use fundamental physical constants from
`tengri.utils.physics_constants` (CODATA 2018, IAU 2015).

References
----------

- Spectral density formula: L_λ = L_ν × c / λ²
  (e.g., Rybicki & Lightman 1979, Radiative Processes in Astrophysics)
- Cosmological flux-luminosity: f_ν = L_ν × (1+z) / (4π d_L²)
  (e.g., Hogg et al. 1999, AJ, 118, 1407)
- Morton (1991) vacuum-air conversion: ApJS, 77, 119
- Edlén (1953) air-vacuum conversion: JOSA, 43(5), 339

"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import jit

from tengri.utils.physics_constants import (
    C_AA,
    JY_CGS,
    L_SUN,
    MAGGIES_ZP_CGS,
)
from tengri.utils.scale import apply_log10_scale, log10_flux_scale, log10_four_pi_dl2

__all__ = [
    "air_to_vacuum",
    "attenuation_to_tau",
    "erg_per_s_to_lsun",
    "flambda_to_fnu",
    "fnu_to_flambda",
    "fnu_to_jy",
    "fnu_to_lnu",
    "fnu_to_maggies",
    "fnu_to_njy",
    "fnu_to_ujy",
    "jy_to_fnu",
    "llambda_to_lnu",
    "lnu_to_fnu",
    "lnu_to_llambda",
    "log_z_abs_to_logzsol",
    "logzsol_to_log_z_abs",
    "lsun_to_erg_per_s",
    "maggies_to_fnu",
    "njy_to_fnu",
    "tau_to_attenuation",
    "ujy_to_fnu",
    "vacuum_to_air",
]


# ── Spectral Density Conversions (Luminosity) ─────────────────────


@jit
def lnu_to_llambda(
    lnu: jnp.ndarray,
    wavelength_aa: jnp.ndarray,
) -> jnp.ndarray:
    """Convert spectral luminosity from per-frequency to per-wavelength basis.

    L_λ = L_ν × c / λ²

    Parameters
    ----------
    lnu : jnp.ndarray
        Spectral luminosity density L_ν [erg/s/Hz].
    wavelength_aa : jnp.ndarray
        Wavelength [Ångström]. Shape must be broadcastable with lnu.

    Returns
    -------
    jnp.ndarray
        Spectral luminosity density L_λ [erg/s/Å].

    Notes
    -----
    Formula: L_λ = L_ν × |dν/dλ| = L_ν × c / λ²

    Reference: Rybicki & Lightman (1979), Radiative Processes in
    Astrophysics, equation 1.59.
    """
    wavelength_aa = jnp.asarray(wavelength_aa)
    return lnu * C_AA / (wavelength_aa**2)


@jit
def llambda_to_lnu(
    llambda: jnp.ndarray,
    wavelength_aa: jnp.ndarray,
) -> jnp.ndarray:
    """Convert spectral luminosity from per-wavelength to per-frequency basis.

    L_ν = L_λ × λ² / c

    Parameters
    ----------
    llambda : jnp.ndarray
        Spectral luminosity density L_λ [erg/s/Å].
    wavelength_aa : jnp.ndarray
        Wavelength [Ångström]. Shape must be broadcastable with llambda.

    Returns
    -------
    jnp.ndarray
        Spectral luminosity density L_ν [erg/s/Hz].

    Notes
    -----
    Inverse of lnu_to_llambda().
    """
    wavelength_aa = jnp.asarray(wavelength_aa)
    return llambda * (wavelength_aa**2) / C_AA


@jit
def fnu_to_flambda(
    fnu: jnp.ndarray,
    wavelength_aa: jnp.ndarray,
) -> jnp.ndarray:
    """Convert spectral flux from per-frequency to per-wavelength basis.

    f_λ = f_ν × c / λ²

    Parameters
    ----------
    fnu : jnp.ndarray
        Spectral flux density f_ν [erg/s/cm²/Hz].
    wavelength_aa : jnp.ndarray
        Wavelength [Ångström]. Shape must be broadcastable with fnu.

    Returns
    -------
    jnp.ndarray
        Spectral flux density f_λ [erg/s/cm²/Å].

    Notes
    -----
    Same formula as lnu_to_llambda() since both are spectral densities.
    """
    wavelength_aa = jnp.asarray(wavelength_aa)
    return fnu * C_AA / (wavelength_aa**2)


@jit
def flambda_to_fnu(
    flambda: jnp.ndarray,
    wavelength_aa: jnp.ndarray,
) -> jnp.ndarray:
    """Convert spectral flux from per-wavelength to per-frequency basis.

    f_ν = f_λ × λ² / c

    Parameters
    ----------
    flambda : jnp.ndarray
        Spectral flux density f_λ [erg/s/cm²/Å].
    wavelength_aa : jnp.ndarray
        Wavelength [Ångström]. Shape must be broadcastable with flambda.

    Returns
    -------
    jnp.ndarray
        Spectral flux density f_ν [erg/s/cm²/Hz].

    Notes
    -----
    Inverse of fnu_to_flambda().
    """
    wavelength_aa = jnp.asarray(wavelength_aa)
    return flambda * (wavelength_aa**2) / C_AA


# ── Flux Density Unit Conversions ─────────────────────────────────


@jit
def fnu_to_jy(fnu_cgs: jnp.ndarray) -> jnp.ndarray:
    """Convert flux from CGS to Jansky.

    Parameters
    ----------
    fnu_cgs : jnp.ndarray
        Spectral flux density [erg/s/cm²/Hz].

    Returns
    -------
    jnp.ndarray
        Spectral flux density [Jy].

    Notes
    -----
    1 Jy ≡ 10⁻²³ erg/s/cm²/Hz.
    """
    return fnu_cgs / JY_CGS


@jit
def jy_to_fnu(fnu_jy: jnp.ndarray) -> jnp.ndarray:
    """Convert flux from Jansky to CGS.

    Parameters
    ----------
    fnu_jy : jnp.ndarray
        Spectral flux density [Jy].

    Returns
    -------
    jnp.ndarray
        Spectral flux density [erg/s/cm²/Hz].

    Notes
    -----
    Inverse of fnu_to_jy().
    """
    return fnu_jy * JY_CGS


@jit
def fnu_to_ujy(fnu_cgs: jnp.ndarray) -> jnp.ndarray:
    """Convert flux from CGS to microjansky.

    Parameters
    ----------
    fnu_cgs : jnp.ndarray
        Spectral flux density [erg/s/cm²/Hz].

    Returns
    -------
    jnp.ndarray
        Spectral flux density [μJy].

    Notes
    -----
    1 μJy ≡ 10⁻³ Jy ≡ 10⁻²⁶ erg/s/cm²/Hz.
    """
    return fnu_cgs / JY_CGS / 1e-6


@jit
def ujy_to_fnu(fnu_ujy: jnp.ndarray) -> jnp.ndarray:
    """Convert flux from microjansky to CGS.

    Parameters
    ----------
    fnu_ujy : jnp.ndarray
        Spectral flux density [μJy].

    Returns
    -------
    jnp.ndarray
        Spectral flux density [erg/s/cm²/Hz].

    Notes
    -----
    Inverse of fnu_to_ujy().
    """
    return fnu_ujy * JY_CGS * 1e-6


@jit
def fnu_to_njy(fnu_cgs: jnp.ndarray) -> jnp.ndarray:
    """Convert flux from CGS to nanojansky.

    Parameters
    ----------
    fnu_cgs : jnp.ndarray
        Spectral flux density [erg/s/cm²/Hz].

    Returns
    -------
    jnp.ndarray
        Spectral flux density [nJy].

    Notes
    -----
    1 nJy ≡ 10⁻⁹ Jy ≡ 10⁻³² erg/s/cm²/Hz.
    """
    return fnu_cgs / JY_CGS / 1e-9


@jit
def njy_to_fnu(fnu_njy: jnp.ndarray) -> jnp.ndarray:
    """Convert flux from nanojansky to CGS.

    Parameters
    ----------
    fnu_njy : jnp.ndarray
        Spectral flux density [nJy].

    Returns
    -------
    jnp.ndarray
        Spectral flux density [erg/s/cm²/Hz].

    Notes
    -----
    Inverse of fnu_to_njy().
    """
    return fnu_njy * JY_CGS * 1e-9


@jit
def fnu_to_maggies(fnu_cgs: jnp.ndarray) -> jnp.ndarray:
    """Convert flux from CGS to maggies (AB magnitude system).

    Parameters
    ----------
    fnu_cgs : jnp.ndarray
        Spectral flux density [erg/s/cm²/Hz].

    Returns
    -------
    jnp.ndarray
        Spectral flux in maggies (dimensionless; 1 maggie ≡ 3631 Jy).

    Notes
    -----
    The AB zeropoint is defined as f_ν(m_AB=0) = 3631 Jy.
    m_AB = -2.5 × log10(f_ν [erg/s/cm²/Hz]) - 48.6.

    Reference: Oke & Gunn (1983), ApJ, 266, 713.
    """
    return fnu_cgs / MAGGIES_ZP_CGS


@jit
def maggies_to_fnu(maggies: jnp.ndarray) -> jnp.ndarray:
    """Convert flux from maggies to CGS.

    Parameters
    ----------
    maggies : jnp.ndarray
        Spectral flux in maggies (dimensionless).

    Returns
    -------
    jnp.ndarray
        Spectral flux density [erg/s/cm²/Hz].

    Notes
    -----
    Inverse of fnu_to_maggies().
    """
    return maggies * MAGGIES_ZP_CGS


# ── Luminosity Conversions ────────────────────────────────────────


@jit
def erg_per_s_to_lsun(luminosity_erg: jnp.ndarray) -> jnp.ndarray:
    """Convert luminosity from erg/s to solar luminosities.

    Parameters
    ----------
    luminosity_erg : jnp.ndarray
        Luminosity [erg/s].

    Returns
    -------
    jnp.ndarray
        Luminosity [L_⊙].

    Notes
    -----
    1 L_⊙ ≡ 3.828e33 erg/s (IAU 2015 nominal value).
    """
    return luminosity_erg / L_SUN


@jit
def lsun_to_erg_per_s(luminosity_lsun: jnp.ndarray) -> jnp.ndarray:
    """Convert luminosity from solar luminosities to erg/s.

    Parameters
    ----------
    luminosity_lsun : jnp.ndarray
        Luminosity [L_⊙].

    Returns
    -------
    jnp.ndarray
        Luminosity [erg/s].

    Notes
    -----
    Inverse of erg_per_s_to_lsun().
    """
    return luminosity_lsun * L_SUN


# ── Cosmological Flux-Luminosity Conversions ──────────────────────


@jit
def lnu_to_fnu(
    lnu: jnp.ndarray,
    dl_cm: jnp.ndarray,
    redshift: jnp.ndarray,
) -> jnp.ndarray:
    """Convert spectral luminosity to spectral flux (k-corrected for redshift).

    f_ν = L_ν × (1+z) / (4π d_L²)

    Parameters
    ----------
    lnu : jnp.ndarray
        Spectral luminosity density L_ν [erg/s/Hz] in the rest frame.
    dl_cm : jnp.ndarray
        Luminosity distance d_L [cm].
    redshift : jnp.ndarray
        Redshift z. Shape must be broadcastable with lnu and dl_cm.

    Returns
    -------
    jnp.ndarray
        Spectral flux density f_ν [erg/s/cm²/Hz] in the observed frame.

    Notes
    -----
    The factor (1+z) accounts for the redshifting of photon energies.
    The 1/(4π d_L²) is the inverse-square dilution over luminosity distance.

    Reference: Hogg et al. (1999), AJ, 118, 1407.

    Notes
    -----
    The flux-scale factor is computed via log-offset arithmetic (apply_log10_scale)
    to avoid float32 underflow. See issue #1186.
    """
    redshift = jnp.asarray(redshift)
    dl_cm = jnp.asarray(dl_cm)
    log10_factor = log10_flux_scale(redshift, dl_cm)
    return apply_log10_scale(lnu, log10_factor)


@jit
def fnu_to_lnu(
    fnu: jnp.ndarray,
    dl_cm: jnp.ndarray,
    redshift: jnp.ndarray,
) -> jnp.ndarray:
    """Convert spectral flux to spectral luminosity (inverse k-correction).

    L_ν = f_ν × 4π d_L² / (1+z)

    Parameters
    ----------
    fnu : jnp.ndarray
        Spectral flux density f_ν [erg/s/cm²/Hz] in the observed frame.
    dl_cm : jnp.ndarray
        Luminosity distance d_L [cm].
    redshift : jnp.ndarray
        Redshift z. Shape must be broadcastable with fnu and dl_cm.

    Returns
    -------
    jnp.ndarray
        Spectral luminosity density L_ν [erg/s/Hz] in the rest frame.

    Notes
    -----
    Inverse of lnu_to_fnu(). The flux-scale factor is computed via log-offset
    arithmetic to avoid float32 overflow. See issue #1186.
    """
    redshift = jnp.asarray(redshift)
    dl_cm = jnp.asarray(dl_cm)
    log10_inv = log10_four_pi_dl2(dl_cm) - jnp.log10(1.0 + redshift)
    return apply_log10_scale(fnu, log10_inv)


# ── Optical Depth & Attenuation ───────────────────────────────────


# ── Metallicity convention ────────────────────────────────────────


@jit
def logzsol_to_log_z_abs(logzsol: jnp.ndarray) -> jnp.ndarray:
    r"""Convert metallicity from solar-relative to absolute.

    .. math::

        \log_{10} Z = \log_{10}(Z/Z_\odot) + \log_{10} Z_\odot

    :math:`\log_{10} Z_\odot` is ``LOG10_ZSUN`` = -1.8477 (Asplund 2009,
    :math:`Z_\odot` = 0.0142, matching MIST).

    Parameters
    ----------
    logzsol : array_like
        Metallicity relative to solar [dex, log10(Z/Zsun)].

    Returns
    -------
    ndarray
        Absolute metallicity [dex, log10(Z)]: the SSP grid's own convention.

    Notes
    -----
    **JIT-compatible**: yes.

    This is the direction ``param_map`` applies on the way *in*. The inverse,
    :func:`log_z_abs_to_logzsol`, is what a user-facing readout needs: without
    it a summary of ``met_logzsol`` comes back 1.85 dex below the value that
    was set.
    """
    from tengri.utils.physics_constants import LOG10_ZSUN

    return logzsol + LOG10_ZSUN


@jit
def log_z_abs_to_logzsol(log_z_abs: jnp.ndarray) -> jnp.ndarray:
    r"""Convert metallicity from absolute to solar-relative.

    .. math::

        \log_{10}(Z/Z_\odot) = \log_{10} Z - \log_{10} Z_\odot

    Parameters
    ----------
    log_z_abs : array_like
        Absolute metallicity [dex, log10(Z)], as the SSP grid stores it.

    Returns
    -------
    ndarray
        Metallicity relative to solar [dex, log10(Z/Zsun)]; the convention
        every user-facing metallicity uses, ``met_logzsol`` included.

    Notes
    -----
    **JIT-compatible**: yes.

    Apply this at a publish boundary, not inside the physics: the SSP grid,
    ``state.derived["log_metallicity_history"]`` and the interpolators all work
    in absolute log10(Z) and must keep doing so.
    """
    from tengri.utils.physics_constants import LOG10_ZSUN

    return log_z_abs - LOG10_ZSUN


@jit
def tau_to_attenuation(tau: jnp.ndarray) -> jnp.ndarray:
    """Convert optical depth τ to attenuation in magnitudes.

    A_λ = 2.5 × log10(e) × τ ≈ 1.0857 × τ

    Parameters
    ----------
    tau : jnp.ndarray
        Optical depth τ (dimensionless). Typically in range [0, 10].

    Returns
    -------
    jnp.ndarray
        Attenuation A_λ [magnitudes].

    Notes
    -----
    The conversion factor 2.5 × log10(e) ≈ 1.08574 relates optical depth
    (which describes transmission as e^(-τ)) to magnitudes
    (which describe transmission as 10^(-A/2.5)).

    Formula: A_λ = -2.5 × log10(e^(-τ)) = 2.5 × log10(e) × τ.
    """
    return 2.5 * jnp.log10(jnp.e) * tau


@jit
def attenuation_to_tau(a_mag: jnp.ndarray) -> jnp.ndarray:
    """Convert attenuation in magnitudes to optical depth τ.

    τ = A_λ / (2.5 × log10(e)) ≈ A_λ / 1.0857

    Parameters
    ----------
    a_mag : jnp.ndarray
        Attenuation A_λ [magnitudes].

    Returns
    -------
    jnp.ndarray
        Optical depth τ (dimensionless).

    Notes
    -----
    Inverse of tau_to_attenuation().
    """
    return a_mag / (2.5 * jnp.log10(jnp.e))


# ── Wavelength Conversions (numpy, not JIT) ───────────────────────


def vacuum_to_air(wavelength_aa: np.ndarray) -> np.ndarray:
    """Convert vacuum wavelengths to air wavelengths.

    Uses the Morton (1991) formula with 4 terms.

    Parameters
    ----------
    wavelength_aa : np.ndarray
        Vacuum wavelengths [Ångström].

    Returns
    -------
    np.ndarray
        Air wavelengths [Ångström].

    Notes
    -----
    The refractive index of air is:
        n = 1 + 2.735182e-4 + 131.4182 / λ² + 2.76249e8 / λ⁴
    where λ is in Ångström.

    This formula is valid for 2000 Å ≤ λ ≤ 100 μm.

    Reference: Morton, D. C. (1991), ApJS, 77, 119.
    """
    wavelength_aa = np.asarray(wavelength_aa, dtype=np.float64)
    inv_lambda_sq = 1.0 / (wavelength_aa**2)
    n = 1.0 + 2.735182e-4 + 131.4182 * inv_lambda_sq + 2.76249e8 * inv_lambda_sq**2
    return wavelength_aa / n


def air_to_vacuum(wavelength_aa: np.ndarray) -> np.ndarray:
    """Convert air wavelengths to vacuum wavelengths.

    Uses the Edlén (1953) dispersion formula for standard air.

    Parameters
    ----------
    wavelength_aa : np.ndarray
        Air wavelengths [Ångström].

    Returns
    -------
    np.ndarray
        Vacuum wavelengths [Ångström].

    Notes
    -----
    The refractive index of air is computed from the wavenumber σ = 1e4 / λ_air:
        n = 1 + 6.4328e-5 + 2.94981e-2 / (146 - σ²) + 2.5540e-4 / (41 - σ²)

    and λ_vac = n · λ_air. This is the Edlén (1953) formula for standard air
    (15 °C, 760 mmHg): the same conversion used by IRAF/SDSS. The Edlén
    (1966) revision differs negligibly for optical spectroscopy.

    Reference: Edlén, B. (1953), "The Dispersion of Standard Air",
    J. Opt. Soc. Am., 43(5), 339.
    """
    wavelength_aa = np.asarray(wavelength_aa, dtype=np.float64)

    # Wavenumber in cm^-1 for air
    sigma = 1e4 / wavelength_aa

    # Refractive index (Edlén 1953)
    n = 1.0 + 6.4328e-5 + 2.94981e-2 / (146.0 - sigma**2) + 2.5540e-4 / (41.0 - sigma**2)

    return wavelength_aa * n
