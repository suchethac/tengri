# SPDX-License-Identifier: BSD-3-Clause
"""SKIRTOR disc spectrum models.

Three piecewise power-law disc spectrum models for AGN torus emission,
implementing the same models as CIGALE's ``skirtor2016.py`` module
(Boquien et al. 2019); validated against its output.

All functions return dimensionless normalized disc spectra (integrated to
unit area under a linear wavelength grid), ready for luminosity scaling
and convolution with dust extinction. Wavelength inputs and breakpoints
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

import jax
import jax.numpy as jnp

from tengri.utils.scale import pow10 as _pow10


def piecewise_powerlaw_disk(
    wavelength: jnp.ndarray,
    limits: jnp.ndarray,
    coefs: jnp.ndarray,
) -> jnp.ndarray:
    """Generic piecewise power-law disc spectrum (normalized to unit area).

    Constructs a piecewise power-law spectrum with breakpoints at specified
    wavelengths and power-law indices in each segment. The spectrum is
    normalized such that the integral over wavelength equals 1.0 (for
    compatibility with CIGALE's normalization).

    Parameters
    ----------
    wavelength: array_like, shape (n_wave,)
        Wavelength grid in nanometer (CIGALE convention).
    limits: array_like, shape (n_segment + 1,)
        Wavelength breakpoints in the same unit as ``wavelength`` (nm).
        Must be strictly increasing.
        Defines n_segment wavelength intervals.
    coefs: array_like, shape (n_segment,)
        Power-law indices for each segment. The spectrum in segment i
        follows :math:`\\lambda^{\\alpha_i}` where :math:`\\alpha_i` is coefs[i].

    Returns
    -------
    spectrum: ndarray, shape (n_wave,)
        Dimensionless normalized spectrum (integral over wavelength = 1.0).

    Notes
    -----
    **JIT-compatible**: yes, uses ``jnp`` primitives.

    The construction proceeds as follows:

    1. For each wavelength segment between limits[i] and limits[i+1],
       the spectrum is a power law :math:`\\lambda^{\\alpha_i}`.
    2. Continuity is enforced at breakpoints via normalization factors:
       :math:`norm_i = norm_{i-1} \\times limits[i]^{\\alpha_{i-1} - \\alpha_i}`.
    3. The unnormalized spectrum is computed piecewise.
    4. Integration via trapezoidal rule (linear grid) yields the normalization
       constant C such that :math:`\\int_0^\\infty C \\times spectrum \\, d\\lambda = 1`.

    The result is frame-independent and can be scaled by any luminosity.
    """
    # Find which segment each wavelength belongs to
    segment_indices = jnp.searchsorted(limits, wavelength, side="right") - 1
    segment_indices = jnp.clip(segment_indices, 0, len(coefs) - 1)

    if wavelength.dtype == jnp.float32:
        # Float32 (#1206): the two factors of ``wavelength**coef * norm`` blow past
        # the float32 window in OPPOSITE directions even though their product is
        # O(1). With a steep segment (coef = -4) at λ ~1e6-1e7 nm,
        # ``wavelength**coef`` ~1e-36..1e-40 flushes to 0 while the matching
        # continuity ``norm`` ~1e40 overflows to inf: so ``0 * inf = nan`` over the
        # whole long-wavelength tail. Build the same spectrum as a single log10 sum
        # (continuity norms become a cumulative SUM of ``coef_step * log10(limit)``)
        # and materialize only the representable result. Exact in float64, which is
        # why the linear form below is kept verbatim for it.
        log_limits = jnp.log10(limits)
        # log10(norm_i) = log10(norm_{i-1}) + (coef_{i-1} - coef_i) * log10(limit_i)
        log_norm_steps = (coefs[:-1] - coefs[1:]) * log_limits[1 : len(coefs)]
        log_norms = jnp.concatenate(
            [jnp.zeros((1,), dtype=log_norm_steps.dtype), jnp.cumsum(log_norm_steps)]
        )
        log_spectrum = coefs[segment_indices] * jnp.log10(wavelength) + log_norms[segment_indices]
        # Peak-factor before exponentiating: the absolute level is arbitrary (the
        # unit-area normalization divides it straight out), so subtracting the peak
        # keeps every value in range and leaves the normalized result unchanged.
        log_spectrum = log_spectrum - jnp.max(log_spectrum)
        spectrum = _pow10(log_spectrum)
        integral = jnp.trapezoid(spectrum, wavelength)
        integral_safe = jnp.maximum(jnp.abs(integral), 1e-30)
        return spectrum / integral_safe

    # Compute normalization factors at each breakpoint for continuity
    norms = jnp.ones(len(coefs))

    def _update_norm(carry, idx):
        """Update norm using JAX-compatible where instead of if."""
        norms_arr = carry
        new_norm = norms_arr[idx - 1] * limits[idx] ** (coefs[idx - 1] - coefs[idx])
        # Use where to avoid conditional tracing
        norms_arr = norms_arr.at[idx].set(jnp.where(idx > 0, new_norm, norms_arr[idx]))
        return norms_arr, None

    norms, _ = jax.lax.scan(_update_norm, norms, jnp.arange(1, len(coefs)))

    # Construct piecewise power-law spectrum
    coef_at_wave = coefs[segment_indices]
    norm_at_wave = norms[segment_indices]
    spectrum = (wavelength**coef_at_wave) * norm_at_wave

    # Normalize to unit area
    integral = jnp.trapezoid(spectrum, wavelength)
    integral_safe = jnp.maximum(jnp.abs(integral), 1e-100)

    return spectrum / integral_safe


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
    wavelength: array_like, shape (n_wave,)
        Wavelength grid in nanometer (CIGALE convention).
    delta: float
        Slope modulation parameter. Range: [-1.0, 1.0]. Default: 0.0.
        Higher delta → steeper mid-IR falloff. This parameter shifts the
        slope coefficient at 100-5000 A from -1.5 by :math:`-\\delta`.

    Returns
    -------
    spectrum: ndarray, shape (n_wave,)
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
    wavelength: array_like, shape (n_wave,)
        Wavelength grid in nanometer (CIGALE convention).
    delta: float
        Slope modulation parameter. Range: [-1.0, 1.0]. Default: 0.0.
        Higher delta → steeper mid-IR falloff.

    Returns
    -------
    spectrum: ndarray, shape (n_wave,)
        Dimensionless normalized spectrum.

    Notes
    -----
    **JIT-compatible**: yes, uses ``jnp`` primitives.

    **Reference**: Implements CIGALE ``skirtor2016.py`` (Boquien et al. 2019
    [2]_); validated against its output.

    References
    ----------
    .. [1] M. Schartmann et al., "Three-dimensional radiative transfer models
       of clumpy tori in Seyfert galaxies," A&A, 437, 861 (2005).
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
    wavelength: array_like, shape (n_wave,)
        Wavelength grid in nanometer (CIGALE convention).
    delta: float
        Blend parameter. Range: [0.0, 1.0]. Default: 0.0.
        delta=0 → pure ADAF. delta=1 → pure thin disc.
        The spectrum is: (1 - delta) * ADAF + delta * DISC.

    Returns
    -------
    spectrum: ndarray, shape (n_wave,)
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
