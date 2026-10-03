# SPDX-License-Identifier: BSD-3-Clause
"""SKIRTOR disc spectrum models.

Three piecewise power-law disc spectrum models for AGN torus emission,
implementing the same models as CIGALE's ``skirtor2016.py`` module
(Boquien et al. 2019); validated against its output.

All functions return dimensionless normalized disc spectra (unit area over
the nanometer axis, from the closed-form integral of the broken power law),
zero below the first and from the last breakpoint, ready for luminosity
scaling and convolution with dust extinction. Wavelength inputs and breakpoints
are in **nanometer** (CIGALE's native SED convention); convert from
Angstrom at the call site if needed.

References
----------

- Stalevski et al. 2012, MNRAS, 420, 2756 (SKIRTOR radiative transfer)
- Stalevski et al. 2016, MNRAS, 458, 2288 (updated SKIRTOR grid)
- Boquien et al. 2019, A&A, 622, A103 (CIGALE code)
- Schartmann et al. 2005, A&A, 437, 861 (alternative torus RT)
- Lopez et al. 2024 (ADAF-disc transition)

"""

from __future__ import annotations

import jax.numpy as jnp

from tengri.utils.scale import LN10 as _LN10, pow10 as _pow10

# |x| below which (1 - e^-x)/x is evaluated by its series (the quotient cancels).
_SERIES_X = 1e-4


def piecewise_powerlaw_disk(
    wavelength: jnp.ndarray,
    limits: jnp.ndarray,
    coefs: jnp.ndarray,
) -> jnp.ndarray:
    r"""Continuous broken power-law disc spectrum on ``[limits[0], limits[-1])``, unit area.

    The spectrum follows :math:`\lambda^{\alpha_k}` between ``limits[k]`` and
    ``limits[k+1]``, is continuous at the breakpoints, and is zero outside
    ``[limits[0], limits[-1])``, the half-open interval of CIGALE's slices. It is
    normalized by the closed-form integral of the power-law segments, so its level
    at a given wavelength does not depend on the sampling of ``wavelength``. CIGALE's
    ``skirtor2016`` ``disk()`` [1]_ builds the same function and divides by the
    trapezoid area on its own axis.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Wavelength grid in nanometer (CIGALE convention).
    limits : array_like, shape (n_segment + 1,)
        Wavelength breakpoints in the same unit as ``wavelength`` (nm).
        Must be strictly increasing.
        Defines n_segment wavelength intervals.
    coefs : array_like, shape (n_segment,)
        Power-law indices :math:`\alpha_k` for each segment.

    Returns
    -------
    spectrum : ndarray, shape (n_wave,)
        Spectrum per nm, integrating to 1 over ``[limits[0], limits[-1])`` and
        zero elsewhere. [nm^-1]

    Notes
    -----
    **JIT-compatible**: yes, uses ``jnp`` primitives.

    With :math:`a_k` = ``limits[k]``, :math:`b_k` = ``limits[k+1]`` and the
    continuity constants :math:`c_0 = 1`,
    :math:`c_k = c_{k-1}\,a_k^{\alpha_{k-1}-\alpha_k}`, the spectrum is
    :math:`s(\lambda) = c_k\lambda^{\alpha_k}/N` in segment :math:`k` and

    .. math::

        N = \sum_k c_k\,b_k^{\alpha_k+1}\,\ln\frac{b_k}{a_k}\,
            \frac{1 - e^{-x_k}}{x_k},
        \qquad x_k = (\alpha_k + 1)\ln\frac{b_k}{a_k},

    whose :math:`\alpha_k = -1` limit is :math:`c_k\ln(b_k/a_k)`. Every term is
    formed as a base-10 logarithm and exponentiated only after the normalization
    is subtracted, so no factor leaves the float32 window (a steep segment at
    :math:`\lambda \sim 10^6`--:math:`10^7` nm has :math:`\lambda^{-4} \sim
    10^{-40}` against a continuity constant :math:`\sim 10^{40}`).

    References
    ----------
    .. [1] M. Boquien et al., "CIGALE: a python Code Investigating GALaxy
       Emission," A&A, 622, A103 (2019). arXiv:1811.03094.
       https://doi.org/10.1051/0004-6361/201834156
    """
    segment_indices = jnp.clip(
        jnp.searchsorted(limits, wavelength, side="right") - 1, 0, len(coefs) - 1
    )
    in_range = (wavelength >= limits[0]) & (wavelength < limits[-1])

    log_limits = jnp.log10(limits)
    # log10(c_k) = log10(c_{k-1}) + (alpha_{k-1} - alpha_k) * log10(a_k)
    log_norm_steps = (coefs[:-1] - coefs[1:]) * log_limits[1 : len(coefs)]
    log_norms = jnp.concatenate(
        [jnp.zeros((1,), dtype=log_norm_steps.dtype), jnp.cumsum(log_norm_steps)]
    )

    # Closed-form segment integrals, in log10. ``g = (1 - e^-x)/x`` has the
    # series branch for |x| below the cancellation floor (alpha = -1 exactly).
    ln_ratio = (log_limits[1:] - log_limits[:-1]) * _LN10
    x = (coefs + 1.0) * ln_ratio
    small = jnp.abs(x) < _SERIES_X
    x_safe = jnp.where(small, 1.0, x)
    g = jnp.where(small, 1.0 - 0.5 * x + x * x / 6.0, -jnp.expm1(-x_safe) / x_safe)
    log_seg = log_norms + (coefs + 1.0) * log_limits[1:] + jnp.log10(ln_ratio * g)
    log_peak = jnp.max(log_seg)
    log_total = log_peak + jnp.log10(jnp.sum(_pow10(log_seg - log_peak)))

    safe_wave = jnp.where(in_range, wavelength, 1.0)
    log_spectrum = (
        coefs[segment_indices] * jnp.log10(safe_wave) + log_norms[segment_indices] - log_total
    )
    return jnp.where(in_range, _pow10(log_spectrum), 0.0)


def skirtor_disk_spectrum(
    wavelength: jnp.ndarray,
    delta: float = 0.0,
) -> jnp.ndarray:
    """SKIRTOR clumpy torus disc spectrum (normalized to unit area).

    Piecewise power-law disc spectrum from the SKIRTOR 2016 models
    (Stalevski et al. 2012). The delta parameter modulates the steep
    mid-infrared slope to allow fitting variations in clumpiness.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Wavelength grid in nanometer (CIGALE convention).
    delta : float
        Slope modulation parameter. Range: [-1.0, 1.0]. Default: 0.0.
        Higher delta → shallower mid-IR falloff. This parameter shifts the
        slope coefficient between 100 and 5000 nm from -1.5 by :math:`+\\delta`.

    Returns
    -------
    spectrum : ndarray, shape (n_wave,)
        Dimensionless normalized spectrum.

    Notes
    -----
    **JIT-compatible**: yes, uses ``jnp`` primitives.

    **Reference**: Implements CIGALE ``skirtor2016.py`` (Boquien et al. 2019
    [2]_); validated against its output.

    References
    ----------
    .. [1] M. Stalevski et al., "3D radiative transfer modeling of the dusty
       torus around AGN: the influence of clumping," MNRAS, 420, 2756 (2012).
       arXiv:1109.1286. https://doi.org/10.1111/j.1365-2966.2011.19775.x
    .. [2] M. Boquien et al., "CIGALE: a python Code Investigating GALaxy
       Emission," A&A, 622, A103 (2019). arXiv:1811.03094.
       https://doi.org/10.1051/0004-6361/201834156
    """
    limits = jnp.array([8.0, 10.0, 100.0, 5000.0, 1e6])
    coefs = jnp.array([0.2, -1.0, -1.5 + delta, -4.0])
    return piecewise_powerlaw_disk(wavelength, limits, coefs)


def schartmann2005_disk_spectrum(
    wavelength: jnp.ndarray,
    delta: float = 0.0,
) -> jnp.ndarray:
    """Schartmann et al. (2005) torus disc spectrum (normalized to unit area).

    Alternative piecewise power-law disc spectrum based on radiative transfer
    models. Features a shallower near-IR slope (1.0 instead of 0.2) and
    smoother mid-IR transition compared to SKIRTOR.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Wavelength grid in nanometer (CIGALE convention).
    delta : float
        Slope modulation parameter. Range: [-1.0, 1.0]. Default: 0.0.
        Higher delta → shallower mid-IR falloff. This parameter shifts the
        slope coefficient between 125 and 1e4 nm from -1.5 by :math:`+\\delta`.

    Returns
    -------
    spectrum : ndarray, shape (n_wave,)
        Dimensionless normalized spectrum.

    Notes
    -----
    **JIT-compatible**: yes, uses ``jnp`` primitives.

    **Reference**: Implements CIGALE ``skirtor2016.py`` (Boquien et al. 2019
    [2]_); validated against its output.

    References
    ----------
    .. [1] M. Schartmann et al., "Towards a physical model of dust tori in
       active galactic nuclei. Radiative transfer calculations for a hydrostatic torus
       model," A&A, 437, 861 (2005).
       https://doi.org/10.1051/0004-6361:20042363
    .. [2] M. Boquien et al., "CIGALE: a python Code Investigating GALaxy
       Emission," A&A, 622, A103 (2019). arXiv:1811.03094.
       https://doi.org/10.1051/0004-6361/201834156
    """
    limits = jnp.array([8.0, 50.0, 125.0, 10000.0, 1e6])
    coefs = jnp.array([1.0, -0.2, -1.5 + delta, -4.0])
    return piecewise_powerlaw_disk(wavelength, limits, coefs)


def adaf_disk_spectrum(
    wavelength: jnp.ndarray,
    delta: float = 0.0,
) -> jnp.ndarray:
    """ADAF + truncated disc blend spectrum (normalized to unit area).

    Blends ADAF (advection-dominated accretion flow) and thin disc spectra
    via a delta parameter, allowing a smooth transition between radiatively
    inefficient and efficient accretion regimes.

    At delta=0, returns the ADAF spectrum. At delta=1, returns the thin disc
    spectrum. Intermediate values are smooth blends.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Wavelength grid in nanometer (CIGALE convention).
    delta : float
        Blend parameter. Range: [0.0, 1.0]. Default: 0.0.
        delta=0 → pure ADAF. delta=1 → pure thin disc.
        The spectrum is: (1 - delta) * ADAF + delta * DISC.

    Returns
    -------
    spectrum : ndarray, shape (n_wave,)
        Dimensionless normalized spectrum.

    Notes
    -----
    **JIT-compatible**: yes.

    The ADAF component covers 8--100000 A with shallower slopes and a
    multi-zone structure. The thin disc component is steeper and truncated
    at longer wavelengths.

    **Reference**: Implements CIGALE ``skirtor2016.py`` ``adaf_disk()``
    (Lopez et al. 2024 [1]_, Boquien et al. 2019 [2]_); validated against
    its output.

    References
    ----------
    .. [1] I. E. Lopez et al., "Modeling the X-ray emission of AGN in CIGALE
       and application to eROSITA," A&A, 691, A163 (2024). arXiv:2407.16182.
       https://doi.org/10.1051/0004-6361/202449801
    .. [2] M. Boquien et al., "CIGALE: a python Code Investigating GALaxy
       Emission," A&A, 622, A103 (2019). arXiv:1811.03094.
       https://doi.org/10.1051/0004-6361/201834156
    """
    # ADAF delta is the ADAF->thin-disc blend weight, defined on [0, 1] (CIGALE
    # skirtor2016 disk_type=2). Clip it before it enters the disc breakpoints so
    # an out-of-range value (the shared ``agn_delta`` prior spans [-1, 1]) cannot
    # produce a negative/non-monotonic limit and a non-finite spectrum.
    delta_c = jnp.clip(delta, 0.0, 1.0)

    # ADAF spectrum
    limits_adaf = jnp.array([8.0, 75.0, 300.0, 1100.0, 2700.0, 20000.0, 100000.0, 1e6])
    coefs_adaf = jnp.array([0.5, 0.15, 0.45, -0.05, -0.55, -1.5, -4.0])

    # Thin disc spectrum (delta-modulated)
    limits_disc = jnp.array(
        [8.0, 50.0, 2000.0 - (delta_c * 1875.0), 5000.0 - (delta_c * 2000.0), 10000.0, 1e6]
    )
    coefs_disc = jnp.array(
        [
            9.0 - (8.0 * delta_c),
            4.2 - (4.4 * delta_c),
            0.7 - (2.2 * delta_c),
            -6.5 + (5.0 * delta_c),
            -4.0,
        ]
    )

    # Compute both spectra
    spec_adaf = piecewise_powerlaw_disk(wavelength, limits_adaf, coefs_adaf)
    spec_disc = piecewise_powerlaw_disk(wavelength, limits_disc, coefs_disc)

    # Blend: (1 - delta) * ADAF + delta * DISC
    # Note: delta parameter here is a blend weight, not slope modulation
    blend_weight = jnp.clip(delta, 0.0, 1.0)
    blended = (1.0 - blend_weight) * spec_adaf + blend_weight * spec_disc

    return blended
