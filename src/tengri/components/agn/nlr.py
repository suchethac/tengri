# SPDX-License-Identifier: BSD-3-Clause
"""Narrow Line Region (NLR) emission model.

The NLR is photoionized gas illuminated by the AGN accretion disc.
It produces forbidden-line emission at key wavelengths.

The NLR emission is isotropic (not masked by the torus) because it
extends on kpc scales beyond the torus opening angle.

For computational efficiency this module uses analytic line profiles
rather than full CLOUDY grids. Each emission line is a Gaussian with
FWHM ~ 500 km/s (narrow lines) placed at the rest-frame wavelength.
Line ratios are calibrated to the Richardson et al. (2014) AGN NLR
template (Table 3, column 'a42'), which provides observationally-derived
emission-line diagnostics for moderate AGN luminosity at intermediate
inclination angle.

All functions are pure JAX and JIT-compilable.

NLR module map (#897) — these are **distinct**, not duplicates
---------------------------------------------------------------

* ``nlr.py`` (this module) — the single-source **analytic** NLR physics
  kernel (``compute_nlr_sed``, Richardson+2014 line ratios). Consumed by the
  composable block ``blocks/nlr_analytic.py`` (the canonical, grammar-reachable
  path via ``agn={'nlr': {'type': 'analytic'}}``).
* ``nlr_cloudy.py`` — grid-backed (Feltre / Synthesizer CLOUDY) NLR **adapters**.
  A *different* physics source (photoionization grids, not analytic line
  ratios), consumed by ``blocks/nlr_synthesizer*.py``. Not a duplicate.
* (removed) ``nlr_model.py`` — a one-file ``SEDModelComponent`` that
  wrapped this kernel but was a grammar-unreachable orphan with drifted param
  names (``agn_nlr_cov_frac`` vs the canonical ``agn_nlr_cf``); deleted in #897
  since ``blocks/nlr_analytic`` already delivers this physics bit-identically.

References
----------

- Richardson et al. 2014, ApJ, 786, 87 (NLR emission-line template)
- Feltre et al. 2016, MNRAS, 456, 3354 (NLR emission-line diagnostics)

"""

import jax.numpy as jnp

from tengri.components.agn._phys import gaussian_line_profile as _gaussian_line_profile

# ── Physical constants ────────────────────────────────────────────

# ── NLR emission-line template ────────────────────────────────────

# Default NLR line FWHM [km/s]
_NLR_FWHM_KMS = 500.0

# Default fraction of intercepted luminosity converted to line emission.
# This is promoted to a free parameter `agn_nlr_line_efficiency`
# with Uniform(0.01, 0.30) prior in _params.py.
_NLR_LINE_EFFICIENCY_DEFAULT = 0.10


def compute_nlr_sed(
    wavelength: jnp.ndarray,
    l_disc_bol_erg: float,
    covering_fraction: float = 0.1,
    fwhm_kms: float = _NLR_FWHM_KMS,
    line_efficiency: float = _NLR_LINE_EFFICIENCY_DEFAULT,
    **_kwargs,
) -> jnp.ndarray:
    """NLR emission spectrum using Richardson+2014 AGN template.

    Thin wrapper delegating to :func:`compute_nlr_sed_richardson2014`.
    Kept for backward compatibility; new code should call
    `compute_nlr_sed_richardson2014` directly.

    The NLR receives ``covering_fraction * L_disc`` and re-emits
    a fraction as emission lines (line-only; no power-law continuum).

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

    Notes
    -----
    **JIT-compatible**: yes — uses ``jnp`` primitives and ``jax.vmap``.
    """
    return compute_nlr_sed_richardson2014(
        wavelength=wavelength,
        l_disc_bol_erg=l_disc_bol_erg,
        covering_fraction=covering_fraction,
        fwhm_kms=fwhm_kms,
        line_efficiency=line_efficiency,
    )


# ── Richardson+2014 NLR template ──────────────────────────────────

# Emission line wavelengths and normalized fluxes from Richardson+2014 Table 3 'a42'.
# Lines sorted by wavelength [Angstrom], fluxes normalized to Hbeta=1.
# Source: FSPS emline_wavelengths at indices
# [38, 40, 41, 43, 45, 50, 51, 52, 59, 61, 62, 64, 68, 69, 70, 72, 73, 74, 75, 76, 77, 78, 80]
_RICHARDSON_WAVES = jnp.array(
    [
        3727.1180,  # [O II] 3726
        3799.0277,  # Ba-8 3798
        3836.5280,  # Ba-7 3835
        3869.9172,  # [Ne III] 3869
        3890.2127,  # Ba-6 3889
        4102.9514,  # Ba-delta 4101.76A
        4341.7476,  # Ba-gamma 4341
        4364.2938,  # [O III] 4363
        4862.7629,  # Ba-beta 4861
        4960.3702,  # [O III] 4959
        5008.3137,  # [O III] 5007
        5201.7880,  # [N I] 5200
        5756.2941,  # [N II] 5755
        5877.3583,  # He I 5875.64A
        6302.1385,  # [O I] 6300
        6365.6364,  # [O I] 6363
        6549.9587,  # [N II] 6548
        6564.7229,  # Ba-alpha 6563 (H-alpha)
        6585.3687,  # [N II] 6584
        6680.0956,  # He I 6678.15A
        6718.3965,  # [S II] 6716
        6732.7805,  # [S II] 6731
        7137.8656,  # [Ar III] 7135
    ]
)

#: Fixed doublet intensity ratios (Storey & Zeippen 2000).
#:
#: Both members of each pair decay from the *same* upper level, so the ratio is
#: set by the two transition probabilities alone — independent of density,
#: temperature, ionization parameter and abundance. It is a constraint, not a
#: measurement, and a template that carries the two lines as independent
#: numbers is free to violate it.
#:
#: It did. Richardson+2014 Table 3 'a42' tabulates [N II] 6548 = 0.79 against
#: 6584 = 2.13, i.e. a ratio of 2.70 — 9% off the atomic value, in a quantity
#: with no physical freedom (#1752). The weak member of each doublet is now
#: derived from the strong one. The strong lines keep their tabulated values,
#: so [O III] 5007 and [N II] 6583 — the lines BPT diagnostics are built on —
#: are untouched, and only the tied partners move.
_OIII_5007_4959_RATIO = 2.98
_NII_6583_6548_RATIO = 2.96

_RICHARDSON_FLUXES = jnp.array(
    [
        2.96,
        0.06,
        0.1,
        1.0,
        0.2,
        0.25,
        0.48,
        0.13,
        1.0,
        8.53 / _OIII_5007_4959_RATIO,  # [O III] 4959 — tied to 5007 (was 2.87)
        8.53,
        0.07,
        0.02,
        0.1,
        0.33,
        0.09,
        2.13 / _NII_6583_6548_RATIO,  # [N II] 6548 — tied to 6584 (was 0.79)
        2.86,
        2.13,
        0.03,
        0.77,
        0.65,
        0.19,
    ]
)


def compute_nlr_sed_richardson2014(
    wavelength: jnp.ndarray,
    l_disc_bol_erg: float,
    covering_fraction: float = 0.1,
    fwhm_kms: float = _NLR_FWHM_KMS,
    line_efficiency: float = _NLR_LINE_EFFICIENCY_DEFAULT,
    **_kwargs,
) -> jnp.ndarray:
    """AGN NLR spectrum using Richardson+2014 Table 3 'a42' line template.

    The narrow-line region (NLR) is photoionized gas illuminated by the AGN
    accretion disc. This function synthesizes the NLR emission spectrum using
    the emission-line template from Richardson et al. (2014), which provides
    AGN-specific line ratios derived from the 'a42' column of Table 3
    (moderate AGN luminosity, intermediate inclination angle).

    The NLR receives ``covering_fraction * L_disc`` and converts a fraction
    into line emission only (no power-law continuum). Each line is modeled
    as a Gaussian profile at the rest-frame wavelength.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength [Angstrom].
    l_disc_bol_erg : float
        Bolometric disc luminosity [erg s^-1].
    covering_fraction : float, optional
        NLR covering fraction (0 to 1). Default 0.1.
    fwhm_kms : float, optional
        Line FWHM [km/s]. Default 500.
    line_efficiency : float, optional
        Fraction of intercepted luminosity converted to line emission.
        Default 0.10.

    Returns
    -------
    l_nu : ndarray, shape (n_wave,)
        NLR L_nu [erg s^-1 Hz^-1].

    Notes
    -----
    **JIT-compatible**: yes — uses ``jnp`` primitives and ``jax.vmap``.

    The Richardson+2014 'a42' template uses 23 emission lines normalized to
    H-beta = 1. The strongest line is [O III] 5007 at 8.53× H-beta,
    consistent with typical Seyfert 2 AGN narrow-line ratios. Line profiles
    are Gaussian with fixed FWHM (narrow lines, ~500 km/s).

    Implements the same AGN NLR line table as Prospector (Johnson et al. 2021
    [2]_); validated against its output.

    References
    ----------
    .. [1] J. C. Richardson, et al., "Optical Spectroscopy of Post-Starburst
       Galaxies," ApJ, 2014, 786, 87. Table 3, column 'a42'.
       https://doi.org/10.1088/0004-637X/786/2/87
    .. [2] B. D. Johnson, et al., "Prospector: Inferring the Star Formation
       Histories of Galaxies from Observed Spectral Energy Distributions,"
       ApJS, 254, 22, 2021. https://doi.org/10.3847/1538-4365/abef67
    """
    l_intercepted = covering_fraction * l_disc_bol_erg

    # --- Line emission ---
    l_lines_total = line_efficiency * l_intercepted

    # Compute the total flux in the template (for normalization)
    flux_sum = jnp.sum(_RICHARDSON_FLUXES)

    # Luminosity per unit flux (normalized to Hbeta)
    l_per_flux = l_lines_total / jnp.maximum(flux_sum, 1e-30)

    # Sum Gaussian profiles for each line
    def _single_line(line_data):
        """Compute luminosity-weighted Gaussian profile for one emission line."""
        wave_c = line_data[0]
        flux_ratio = line_data[1]
        profile = _gaussian_line_profile(wavelength, wave_c, fwhm_kms)
        return flux_ratio * l_per_flux * profile

    # vmap over lines
    from jax import vmap

    line_data = jnp.stack([_RICHARDSON_WAVES, _RICHARDSON_FLUXES], axis=1)
    line_spectra = vmap(_single_line)(line_data)
    l_nu_lines = jnp.sum(line_spectra, axis=0)

    return l_nu_lines
