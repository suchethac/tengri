# SPDX-License-Identifier: BSD-3-Clause
"""Narrow Line Region (NLR) emission model.

The NLR is photoionized gas illuminated by the AGN accretion disc.
It produces forbidden-line emission at key wavelengths.

The NLR emission is isotropic (not masked by the torus) because it
extends on kpc scales beyond the torus opening angle.

For computational efficiency this module uses analytic line profiles
rather than full CLOUDY grids. Each emission line is a Gaussian with
FWHM ~ 500 km/s (narrow lines) placed at the rest-frame wavelength.

Line ratios come from Richardson et al. (2014) Table 3, column 'a42' —
*dereddened emission-line strengths for the AGN locus, relative to Hbeta*.
Richardson et al. selected SDSS galaxies whose spectra are not dominated by
star formation, then co-added them into fifteen high-S/N composite spectra
(``a00`` ... ``a42``) forming a sequence in **NLR ionization level**. The
column label is a position on that ionization sequence — it is not a
luminosity, an inclination, or a photoionization-model grid point. The
numbers are *measurements* off stacked observed spectra, which is what makes
them empirical rather than theoretical (see the doublet note below).

All functions are pure JAX and JIT-compilable.

NLR module map (#897) — these are **distinct**, not duplicates
---------------------------------------------------------------

* ``nlr.py`` (this module) — the single-source **analytic** NLR physics
  kernel (``compute_nlr_sed``, Richardson+2014 line ratios). Consumed by the
  composable block ``blocks/nlr_analytic.py`` (the canonical, grammar-reachable
  path via ``agn={'nlr': {'type': 'analytic'}}``).
* ``nlr_cloudy.py``: grid-backed (Feltre / Synthesizer CLOUDY) NLR **adapters**.
  A *different* physics source (photoionization grids, not analytic line
  ratios), consumed by ``blocks/nlr_synthesizer*.py``. Not a duplicate.
* (removed) ``nlr_model.py`` — a one-file ``SEDModelComponent`` that
  wrapped this kernel but was a grammar-unreachable orphan with drifted param
  names (``agn_nlr_cov_frac`` vs the canonical ``agn_nlr_cf``); deleted in #897
  since ``blocks/nlr_analytic`` already delivers this physics bit-identically.

References
----------

- Richardson et al. 2014, MNRAS, 437, 2376 (NLR emission-line template)
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

#: DO NOT "correct" the forbidden doublets in this table to their atomic ratios.
#:
#: Both members of [O III] 4959/5007, [N II] 6548/6584 and [O I] 6300/6363 decay
#: from the same upper level, so in a *photoionization model* their intensity
#: ratio is fixed by the transition probabilities alone and carries no freedom.
#: That is a constraint on ``components/nebular/shock.py`` and on the Cloudy-grid
#: backends, and it is enforced there.
#:
#: It is **not** a constraint here. Richardson+2014 Table 3 is dereddened line
#: strengths *measured* off stacked SDSS composites, so each entry carries the
#: deblending and S/N systematics of the stack. As tabulated, a42 gives
#: 5007/4959 = 2.97 (atomic 2.98), 6584/6548 = 2.70 (atomic ~2.94) and
#: 6300/6363 = 3.67 (atomic ~3.00). The [N II] and [O I] deviations are real
#: properties of the measurement — [N II] 6548 is a weak line on the H-alpha
#: wing, and 6363 is quoted to one significant figure — not defects to repair.
#:
#: #1752 rewrote 6548 to 2.13/2.96 and 4959 to 8.53/2.98 on the atomic argument.
#: That was wrong twice over: it imposed model physics on a measurement, and it
#: broke the parity this table exists for — the values below are the published
#: a42 column, agreeing value-for-value with the same table as carried by
#: Prospector (``AGNSpecModel.init_aline_info``), which is the claim the
#: docstring makes. Reverted; the guard is now a parity test against the
#: published values, not an atomic-ratio assertion.
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
        2.87,
        8.53,
        0.07,
        0.02,
        0.1,
        0.33,
        0.09,
        0.79,
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
    AGN-specific line ratios derived from the 'a42' column of Table 3 — one
    of fifteen composite SDSS spectra forming a sequence in NLR ionization
    level.

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

    **Empirical, not theoretical.** Table 3 lists dereddened strengths measured
    off stacked observed spectra, so the forbidden doublets do not reproduce
    their atomic branching ratios exactly ([N II] 6584/6548 = 2.70 against an
    atomic ~2.94). Those deviations are inherited from the measurement and are
    deliberately preserved — see the note above ``_RICHARDSON_FLUXES``. For a
    template whose doublets are tied by construction, use a photoionization
    backend (``agn={'nlr': {'type': 'synthesizer'}}``) instead.

    Implements the same AGN NLR line table as Prospector (Johnson et al. 2021
    [2]_); validated against its output.

    References
    ----------
    .. [1] C. T. Richardson, J. T. Allen, J. A. Baldwin, P. C. Hewett, and
       G. J. Ferland, "Interpreting the ionization sequence in AGN
       emission-line spectra," MNRAS, 2014, 437, 3, 2376-2403. Table 3,
       column 'a42'. https://doi.org/10.1093/mnras/stt2056
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
