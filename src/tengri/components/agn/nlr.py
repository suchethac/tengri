"""Narrow Line Region (NLR) emission model.

The NLR is photoionized gas illuminated by the AGN accretion disc.
It produces nebular-like emission: a power-law continuum plus
forbidden-line emission at key wavelengths.

The NLR emission is isotropic (not masked by the torus) because it
extends on kpc scales beyond the torus opening angle.

For computational efficiency this module uses analytic line profiles
rather than full CLOUDY grids. Each emission line is a Gaussian with
FWHM ~ 500 km/s (narrow lines) placed at the laboratory wavelength.

All functions are pure JAX and JIT-compilable.

References
----------
- Groves et al. 2004, ApJS, 153, 75 (NLR photoionization models)
- Feltre et al. 2016, MNRAS, 456, 3354 (NLR emission-line diagnostics)
- Vijarnwannaluk et al. 2022 (NLR covering fractions)
"""

import jax.numpy as jnp

from tengri.components.agn._phys import gaussian_line_profile as _gaussian_line_profile

# ── Physical constants ────────────────────────────────────────────
from tengri.utils.physics_constants import (
    AA_TO_CM as _ANGSTROM_CM,
    C_CGS as _C_LIGHT,
)

# ── NLR emission-line template ────────────────────────────────────

# Key narrow emission lines: (rest wavelength [Angstrom], relative strength)
# Relative strengths are approximate, calibrated to typical Seyfert 2 spectra.
# Normalized so that the sum of line luminosities ~ 1.
_NLR_LINES = jnp.array(
    [
        # [OII] 3727 doublet (blended)
        [3727.0, 0.15],
        # [NeIII] 3869
        [3869.0, 0.04],
        # H-beta 4861 (narrow)
        [4861.0, 0.10],
        # [OIII] 4959 (ratio 5007/4959 = 2.98; Storey & Zeippen 2000)
        [4959.0, 0.117],
        # [OIII] 5007 (strongest NLR line)
        [5007.0, 0.35],
        # [OI] 6300
        [6300.0, 0.03],
        # [NII] 6548
        [6548.0, 0.05],
        # H-alpha 6563 (narrow)
        [6563.0, 0.20],
        # [NII] 6583
        [6583.0, 0.15],
        # [SII] 6716
        [6716.0, 0.06],
        # [SII] 6731
        [6731.0, 0.06],
    ]
)

_NLR_LINE_WAVELENGTHS = _NLR_LINES[:, 0]
_NLR_LINE_STRENGTHS = _NLR_LINES[:, 1]
# Normalize so sum = 1
_NLR_LINE_STRENGTHS_NORM = _NLR_LINE_STRENGTHS / jnp.sum(_NLR_LINE_STRENGTHS)

# Default NLR line FWHM [km/s]
_NLR_FWHM_KMS = 500.0

# Fraction of intercepted luminosity converted to line emission
# (the rest heats dust / continuum). Typical: 5-15%.
_NLR_LINE_EFFICIENCY = 0.10

# Fraction of intercepted luminosity re-emitted as continuum
_NLR_CONTINUUM_EFFICIENCY = 0.02

# Continuum power-law slope (F_nu ~ nu^alpha, typical NLR continuum)
_NLR_CONTINUUM_ALPHA = -1.5


def compute_nlr_sed(
    wavelength: jnp.ndarray,
    l_disc_bol_erg: float,
    covering_fraction: float = 0.1,
    fwhm_kms: float = _NLR_FWHM_KMS,
    line_efficiency: float = _NLR_LINE_EFFICIENCY,
) -> jnp.ndarray:
    """NLR emission spectrum: line emission + power-law continuum.

    The NLR receives ``covering_fraction * L_disc`` and re-emits
    a fraction as emission lines and a small continuum component.

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Rest-frame wavelength [Angstrom].
    l_disc_bol_erg : float
        Bolometric disc luminosity [erg s^-1].
    covering_fraction : float
        NLR covering fraction (0 to 1). Default 0.1.
    fwhm_kms : float
        Line FWHM [km/s]. Default 500.
    line_efficiency : float
        Fraction of intercepted luminosity converted to line emission.
        Default 0.10.

    Returns
    -------
    array, shape (n_wave,)
        NLR L_nu [erg s^-1 Hz^-1].
    """
    l_intercepted = covering_fraction * l_disc_bol_erg

    # --- Line emission ---
    l_lines_total = line_efficiency * l_intercepted

    # Sum Gaussian profiles for each line
    def _single_line(line_data):
        """Compute luminosity-weighted Gaussian profile for one NLR emission line."""
        lam_c = line_data[0]
        strength = line_data[1]
        profile = _gaussian_line_profile(wavelength, lam_c, fwhm_kms)
        return strength * l_lines_total * profile

    # vmap over lines
    from jax import vmap

    line_spectra = vmap(_single_line)(_NLR_LINES)
    # Renormalize strengths
    strength_sum = jnp.sum(_NLR_LINE_STRENGTHS)
    l_nu_lines = jnp.sum(line_spectra, axis=0) / jnp.maximum(strength_sum, 1e-30)

    # --- Continuum emission ---
    l_cont_total = _NLR_CONTINUUM_EFFICIENCY * l_intercepted
    nu = _C_LIGHT / (wavelength * _ANGSTROM_CM)
    cont_shape = nu**_NLR_CONTINUUM_ALPHA

    # Normalize continuum
    sort_idx = jnp.argsort(nu)
    integral = jnp.trapezoid(cont_shape[sort_idx], nu[sort_idx])
    integral_safe = jnp.maximum(jnp.abs(integral), 1e-100)
    l_nu_cont = l_cont_total * cont_shape / integral_safe

    return l_nu_lines + l_nu_cont


nlr_emission = compute_nlr_sed
