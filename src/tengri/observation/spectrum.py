# SPDX-License-Identifier: BSD-3-Clause
"""Pixel-level spectroscopic forward model.

Fits every spectral pixel directly, with an optional multiplicative
calibration polynomial to absorb flux-calibration uncertainties
(following Prospector / Johnson+2021).

Includes emission-line placement with instrument-resolution blending,
relevant for R < 1000 spectroscopy where close lines merge.

Also provides wavelength-dependent Line Spread Function (LSF) convolution
for instruments with variable spectral resolution (e.g., JWST NIRSpec PRISM).
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from tengri.units import lnu_to_fnu

# ── SSP library spectral resolutions (velocity dispersion in km/s)
SSP_LIBRARY_RESOLUTIONS: dict[str, float] = {
    "miles": 70.0,  # R ~ 2500 at 5000 A, sigma ~ 70 km/s
    "c3k": 15.0,  # R ~ 10000, sigma ~ 15 km/s
    "fsps_default": 70.0,  # MILES-based (default FSPS)
}


# ── Speed of light ────────────────────────────────────────────────
_C_KM_S = 299792.458  # km/s
_FWHM_TO_SIGMA = 2.354820045030949  # 2*sqrt(2*ln(2))


# ── Instrument resolution profiles ────────────────────────────────


def nirspec_prism_resolution(wave_um: jnp.ndarray) -> jnp.ndarray:
    """JWST NIRSpec PRISM R(lambda), ranges from approximately 30 to 300.

    Approximate from NIRSpec documentation. R increases roughly linearly
    from 0.6 to 5.3 microns.

    Parameters
    ----------
    wave_um: array, shape (n_wave,)
        Observed wavelength [micron].

    Returns
    -------
    ndarray, shape (n_wave,)
        Spectral resolution R = lambda / delta_lambda (dimensionless).

    Notes
    -----
    Not JIT-compatible (uses Python-side clipping for readability).

    """
    return jnp.clip(30.0 + 55.0 * (wave_um - 0.6), 30.0, 330.0)


def nirspec_g140m_resolution(wave_um: jnp.ndarray) -> jnp.ndarray:
    """JWST NIRSpec G140M grating, roughly constant R ≈ 1000.

    Parameters
    ----------
    wave_um: array, shape (n_wave,)
        Observed wavelength [micron].

    Returns
    -------
    ndarray, shape (n_wave,)
        Spectral resolution R ≈ 1000 (dimensionless).

    Notes
    -----
    Not JIT-compatible (uses Python ones_like for constant array).

    """
    return 1000.0 * jnp.ones_like(wave_um)


# ── Line Spread Function (LSF) convolution ────────────────────────


def _resolution_to_sigma_kms(resolution: jnp.ndarray) -> jnp.ndarray:
    """Convert spectral resolution R to velocity dispersion sigma.

    sigma = c / (FWHM_TO_SIGMA * R)

    Parameters
    ----------
    resolution: array or scalar
        Spectral resolution R = lambda / delta_lambda (dimensionless).

    Returns
    -------
    array or scalar
        Velocity dispersion [km/s].

    Notes
    -----
    Private helper. Not JIT-compatible (may be called with traced values).

    """
    return _C_KM_S / (_FWHM_TO_SIGMA * resolution)


def _is_log_uniform(wave) -> bool:
    r"""Whether ``wave`` is uniform in :math:`\ln\lambda`, so one pixel scale serves.

    ``True`` for a tracer: a traced grid has no values to inspect at trace time.
    The gap is the one :func:`_require_log_uniform_grid` documents, and narrow for
    the same reason, a spectroscopic wavelength grid is normally a fixed
    instrument array closed over by the jitted function, not an argument traced
    through it. Answering ``True`` keeps the single-FFT path, which is what such a
    grid got before #1791.

    Parameters
    ----------
    wave: array_like, shape (n_pix,)
        Wavelength grid [Angstrom].

    Returns
    -------
    bool
        ``True`` if one ``d ln lambda`` describes the whole grid.

    Notes
    -----
    Private helper. Build-time (NumPy), not JIT-compatible. Called with a
    concrete grid it runs once at *trace* time, so it costs nothing per
    evaluation.
    """
    if isinstance(wave, jax.core.Tracer):
        return True
    w = np.asarray(wave, dtype=np.float64)
    if w.size < 3 or not np.all(np.isfinite(w)) or np.any(w <= 0.0):
        return True  # not a grid this helper can speak about; let the caller fail
    dln = np.diff(np.log(w))
    mean = float(np.mean(dln))
    if mean == 0.0:
        return True
    return bool(float(np.ptp(dln) / abs(mean)) <= _LOG_UNIFORM_RTOL)


@jax.jit
def _apply_lsf_constant_r(
    spectrum: jnp.ndarray,
    wave_obs: jnp.ndarray,
    sigma_eff_kms: float,
) -> jnp.ndarray:
    """FFT convolution in log-wavelength space for constant R.

    This is equivalent to velocity_broaden but with the effective
    (library-subtracted) sigma.

    Parameters
    ----------
    spectrum: array, shape (n_pix,)
        Input spectral flux.
    wave_obs: array, shape (n_pix,)
        Observed wavelength grid [Angstrom]. Must be uniform in ``ln(lambda)``,
        not merely evenly spaced, the pixel scale is read once from the first
        pair, so a linear grid under-broadens by ``wave[0]/lambda`` (#1742).
        ``apply_lsf`` dispatches here only for a grid that satisfies this, and
        sends the rest to :func:`_apply_lsf_variable_r` (#1791), so the
        precondition binds only direct callers of this helper.
    sigma_eff_kms: float
        Effective velocity dispersion [km/s] (after library subtraction).

    Returns
    -------
    ndarray, shape (n_pix,)
        Smoothed spectrum (same units as input).

    Notes
    -----
    JIT-compatible: yes. Private helper for apply_lsf.

    """
    sigma_v = sigma_eff_kms / _C_KM_S
    dlnwave = jnp.log(wave_obs[1] / wave_obs[0])
    sigma_pix = sigma_v / dlnwave

    n = spectrum.shape[0]
    freq = jnp.fft.rfftfreq(n)
    kernel_ft = jnp.exp(-2.0 * jnp.pi**2 * sigma_pix**2 * freq**2)

    flux_ft = jnp.fft.rfft(spectrum)
    return jnp.fft.irfft(flux_ft * kernel_ft, n=n)


@partial(jax.jit, static_argnums=(3,))
def _apply_lsf_variable_r(
    spectrum: jnp.ndarray,
    wave_obs: jnp.ndarray,
    sigma_eff_kms: jnp.ndarray,
    n_bins: int = 16,
) -> jnp.ndarray:
    """Piecewise-constant LSF convolution for variable R.

    Splits the wavelength range into ``n_bins`` segments. Within each
    segment the mean effective sigma is used for an FFT convolution.
    The segments are blended with smooth (raised-cosine) overlap to
    avoid discontinuities. Accurate to approximately 1% for typical instrument
    profiles and fully differentiable.

    Parameters
    ----------
    spectrum: array, shape (n_pix,)
        Input spectral flux.
    wave_obs: array, shape (n_pix,)
        Observed wavelength grid [Angstrom]. Any strictly increasing grid: each
        bin takes its pixel scale from the local ``d ln lambda``, so a grid that
        is not log-uniform is handled here rather than refused (#1791).
    sigma_eff_kms: array, shape (n_pix,)
        Effective velocity dispersion at each pixel [km/s].
    n_bins: int, optional
        Number of piecewise-constant segments. More bins gives better
        accuracy but requires more FFTs. Typical: 10–20. Default 16.

    Returns
    -------
    ndarray, shape (n_pix,)
        Smoothed spectrum (same units as input).

    Notes
    -----
    JIT-compatible: yes, `n_bins` is a static argument.
    Gradient-safe: yes. Private helper for apply_lsf.

    """
    n_pix = spectrum.shape[0]
    # Local pixel scale, not the blue-end value for the whole array. Each bin
    # already convolves with its own sigma; giving it its own d(ln lambda) as
    # well is what lets this path serve a grid that is not log-uniform, where a
    # single global scale under-broadened by wave[0]/lambda (#1791). On a
    # log-uniform grid jnp.gradient returns that same constant, so nothing moves.
    dlnwave_local = jnp.gradient(jnp.log(wave_obs))
    freq = jnp.fft.rfftfreq(n_pix)
    flux_ft = jnp.fft.rfft(spectrum)

    # Pixel indices for bin edges (uniform split)
    bin_edges = jnp.linspace(0, n_pix, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1] - bin_edges[0]

    # Pixel index array
    pix_idx = jnp.arange(n_pix, dtype=jnp.float64)

    def _convolve_bin(carry, bin_center):
        r"""Convolve with the mean sigma for one bin, weighted by overlap."""
        # Pixel index of bin center
        center = bin_center
        half_w = bin_width * 0.75  # overlap region for blending

        # Smooth weight: raised cosine (1 at center, 0 outside)
        dist = jnp.abs(pix_idx - center) / half_w
        weight = jnp.where(dist < 1.0, 0.5 * (1.0 + jnp.cos(jnp.pi * dist)), 0.0)

        # Mean sigma in this bin (weighted by the bin window)
        bin_mask = jnp.where(
            jnp.abs(pix_idx - center) < bin_width,
            1.0,
            0.0,
        )
        # Both clamps written out rather than hoisted into a shared name: XLA
        # common-subexpression-eliminates them, and tools/check_zero_hiding_clamps.py
        # matches the division syntactically, so hoisting would retire a site from
        # that audit while the clamp is still there, shrinking the inventory
        # silently is the one thing that guard exists to prevent.
        n_in_bin = jnp.sum(bin_mask)
        sigma_mean = jnp.sum(sigma_eff_kms * bin_mask) / jnp.maximum(n_in_bin, 1.0)
        dlnwave_mean = jnp.sum(dlnwave_local * bin_mask) / jnp.maximum(n_in_bin, 1.0)

        # FFT convolution with this sigma, at this bin's own pixel scale
        sigma_pix = (sigma_mean / _C_KM_S) / dlnwave_mean
        kernel_ft = jnp.exp(-2.0 * jnp.pi**2 * sigma_pix**2 * freq**2)
        convolved = jnp.fft.irfft(flux_ft * kernel_ft, n=n_pix)

        return carry + weight * convolved, None

    # Accumulate weighted contributions from all bins
    result = jnp.zeros(n_pix)
    result, _ = jax.lax.scan(_convolve_bin, result, bin_centers)

    # Normalize by total weight at each pixel
    def _weight_bin(carry, bin_center):
        r"""Accumulate raised-cosine overlap weights for all bins at each pixel."""
        center = bin_center
        half_w = bin_width * 0.75
        dist = jnp.abs(pix_idx - center) / half_w
        weight = jnp.where(dist < 1.0, 0.5 * (1.0 + jnp.cos(jnp.pi * dist)), 0.0)
        return carry + weight, None

    total_weight, _ = jax.lax.scan(_weight_bin, jnp.zeros(n_pix), bin_centers)
    total_weight = jnp.maximum(total_weight, 1e-30)

    return result / total_weight


def apply_lsf(
    spectrum: jnp.ndarray,
    wave_obs: jnp.ndarray,
    resolution: jnp.ndarray | float,
    sigma_lib_kms: float = 0.0,
    n_bins: int = 16,
    sigma_v_kms: float = 0.0,
) -> jnp.ndarray:
    r"""Apply wavelength-dependent Line Spread Function with library resolution subtraction.

    Convolves the input spectrum with a Gaussian kernel that combines instrument
    line-spread function (LSF) and stellar velocity dispersion, accounting for
    the pre-existing broadening in the SSP library. Uses FFT convolution for
    speed and differentiability.

    The effective kernel width at each pixel is computed via quadrature subtraction:

    .. math::

        \\sigma_\\mathrm{eff}(\\lambda) =
            \\sqrt{\\sigma_\\mathrm{inst}(\\lambda)^2 - \\sigma_\\mathrm{lib}^2}

    where :math:`\\sigma_\\mathrm{inst}(\\lambda) = c / (2.3548 \\times R(\\lambda))`
    is the instrument's velocity dispersion [km/s] from spectral resolution
    :math:`R(\\lambda) = \\lambda / \\Delta\\lambda`.

    **Special case**: If :math:`\\sigma_\\mathrm{inst} < \\sigma_\\mathrm{lib}` at
    some wavelengths, no broadening is applied (cannot sharpen an already-broadened
    spectrum). This happens when the SSP library resolution is better than the
    instrument's LSF.

    **Implementation**: For constant R (scalar input), uses a single FFT convolution
    in log-wavelength space (fast, O(N log N)). For variable R (array input),
    uses a piecewise-constant approximation with ``n_bins`` segments and smooth
    raised-cosine blending at boundaries (~10–20 FFTs, accurate to ~1%).

    Parameters
    ----------
    spectrum: array, shape (n_pix,)
        Input spectral flux at observed wavelengths [erg/s/cm²/Hz or arbitrary units].
    wave_obs: array, shape (n_pix,)
        Observed-frame wavelength grid [Ångstrom]. Any strictly increasing grid
        is accepted. One not uniform in log-wavelength, a linearly-spaced grid,
        for instance, is routed through the piecewise path, which carries a
        per-bin pixel scale, so the requested width is delivered either way
        (#1791). A log-uniform grid with scalar ``R`` keeps the single-FFT path.
    resolution: array, shape (n_pix,) or float
        Spectral resolution :math:`R(\\lambda) = \\lambda / \\Delta\\lambda`.

        - Scalar: constant resolution across wavelength (fast path)
        - Array: per-pixel wavelength-dependent resolution (e.g., JWST NIRSpec PRISM)

    sigma_lib_kms: float, optional
        SSP library velocity dispersion [km/s]. Subtracted in quadrature
        from instrument LSF. Default 0.0 (no subtraction). Common values:

        - MILES-based (FSPS default): 70 km/s
        - C3K: 15 km/s
        - IRTF: 20 km/s

        Use ``SSP_LIBRARY_RESOLUTIONS[library_name]`` for pre-defined values.
    n_bins: int, optional
        Number of piecewise-constant segments for variable-R approximation.
        Ignored for scalar R. Higher values are more accurate but slower.
        Typical: 10–20. Default 16.
    sigma_v_kms: float, optional
        Intrinsic galaxy velocity dispersion :math:`\\sigma_v` [km/s] added
        in quadrature to :math:`\\sigma_{\\rm eff}`. This is the broadening
        from stellar dynamics, distinct from instrument LSF
        (``resolution``) and from the SSP-library template resolution
        (``sigma_lib_kms``). Default 0.0 (no extra broadening).

    Returns
    -------
    spectrum_smoothed: array, shape (n_pix,)
        Spectrum convolved with the effective LSF kernel [same units as input].

    Notes
    -----
    **JIT-compatible**: no, the dispatch logic (constant vs. variable R)
    uses Python-side branching. Wrap the result in :func:`jax.jit` only
    if R is known at trace time.

    **Gradient-safe**: yes, all operations inside the conditionally-selected
    path are differentiable.

    **Log-wavelength convolution**: Convolution is performed in log-wavelength
    space, which correctly represents velocity-space broadening. A grid that is
    not uniform in ``ln(lambda)`` takes the piecewise path, where each bin uses
    its own local ``d ln lambda``, ``n_bins`` FFTs instead of one, and nothing is
    resampled, so flux conservation and the zero-width identity stay exact to
    machine precision. Before #1791 such a grid was convolved with the pixel scale
    read from its blue end, under-broadening by ``wave[0]/lambda``, 0.60 at
    5000 A on a 3000-10000 A grid, and biasing any fitted ``sigma_v_kms`` high by
    the reciprocal. Measured recovery of a requested 200 km/s on
    ``linspace(3000, 10000)``: 0.991 at the default ``n_bins=16``.

    **Boundary handling**: FFT convolution wraps at the edges (circular convolution).
    For small spectra (N < 1000 pixels) or incomplete coverage, consider padding
    before calling this function.

    See Also
    --------
    velocity_broaden: Convolve with velocity dispersion only (no library subtraction).
    nirspec_prism_resolution: JWST NIRSpec PRISM variable-R function.
    nirspec_g140m_resolution: JWST NIRSpec G140M constant-R function.

    Examples
    --------
    **Constant spectral resolution (R = 100):**

    >>> spectrum_smoothed = apply_lsf(spectrum, wave_obs, resolution=100.0)

    **JWST NIRSpec PRISM (variable resolution, accounting for MILES library):**

    >>> wave_um = wave_obs / 1e4  # convert Angstrom to micron
    >>> R_prism = nirspec_prism_resolution(wave_um)
    >>> spectrum_smoothed = apply_lsf(
    ...     spectrum,
    ...     wave_obs,
    ...     resolution=R_prism,
    ...     sigma_lib_kms=70.0,  # MILES-based SSP library
    ... )

    **Custom wavelength-dependent resolution:**

    >>> R_custom = 100.0 + 50.0 * (wave_obs / 5000.0)  # increases with wavelength
    >>> spectrum_smoothed = apply_lsf(spectrum, wave_obs, resolution=R_custom)

    """
    # Clamp non-negative (priors enforce this; clamp keeps trace-safe path
    # for callers that pass sigma_v_kms in via the params dict, the prior
    # guards against negatives, so this is purely defensive).
    sigma_v_kms = jnp.maximum(jnp.asarray(sigma_v_kms), 0.0)

    resolution = jnp.asarray(resolution)

    # Compute instrument sigma at each pixel
    sigma_inst_kms = _C_KM_S / (_FWHM_TO_SIGMA * resolution)

    # Subtract library resolution and add intrinsic stellar velocity
    # dispersion in quadrature:
    #   σ_total² = σ_inst² − σ_lib² + σ_v²
    # σ_lib accounts for the broadening already baked into the SSP
    # templates; σ_v is the intrinsic galaxy velocity dispersion.
    sigma_lib2 = sigma_lib_kms**2
    sigma_v2 = sigma_v_kms**2
    sigma_eff_kms = jnp.sqrt(jnp.maximum(sigma_inst_kms**2 - sigma_lib2, 0.0) + sigma_v2)

    # The single-FFT path reads one pixel scale, ``log(wave[1]/wave[0])``, which
    # describes the whole array only on a grid uniform in ln(lambda). On any other
    # grid it under-broadens by ``wave[0]/lambda`` (#1791). Rather than refuse such
    # grids, #1742's remedy for ``velocity_broaden``, which would take
    # spectroscopy with it, since tengri's own forward model runs on linear
    # observed grids, send them through the piecewise path, which carries a
    # per-bin pixel scale. Nothing is resampled, so the FFT normalization still
    # conserves flux exactly and a zero-width kernel is still the identity.
    if resolution.ndim == 0 and _is_log_uniform(wave_obs):
        # Scalar R on a log-uniform grid: one FFT, and the scale is exact.
        return _apply_lsf_constant_r(spectrum, wave_obs, sigma_eff_kms)

    # Per-pixel R, or a grid whose pixel scale varies: piecewise-constant in both.
    sigma_per_pixel = jnp.broadcast_to(jnp.atleast_1d(sigma_eff_kms), spectrum.shape)
    return _apply_lsf_variable_r(spectrum, wave_obs, sigma_per_pixel, n_bins)


def project_spectrum(
    sed_rest: jnp.ndarray,
    wave_rest: jnp.ndarray,
    wave_obs: jnp.ndarray,
    redshift: float,
    dl_cm: float,
    *,
    resolution: jnp.ndarray | float | None = None,
    sigma_lib_kms: float = 0.0,
    n_bins: int = 16,
    sigma_v_kms: float = 0.0,
    cal_coeffs: jnp.ndarray | None = None,
    cal_wave_range: tuple[float, float] | None = None,
    conserving: bool = False,
    resolution_matrix: object | None = None,
) -> jnp.ndarray:
    r"""Project a panchromatic model SED onto an observed-frame spectrum grid.

    Consolidates the spectrum projection pipeline: compute observed-frame fluxes
    at pixel wavelengths via interpolation, then optionally apply wavelength-dependent
    Line Spread Function (LSF) convolution accounting for instrument resolution, then
    optionally apply multiplicative flux-calibration polynomial.

    **Absorption (IGM/DLA/Milky Way) is composed by callers, not here.**
    Flux calibration is now applied here; IGM/DLA/MW absorption is still
    composed by callers.

    Parameters
    ----------
    sed_rest: array, shape (n_wave,)
        Rest-frame spectral luminosity density [erg/s/Hz] on the
        rest-frame wavelength grid.
    wave_rest: array, shape (n_wave,)
        Rest-frame wavelength grid [Angstrom].
    wave_obs: array, shape (n_pix,)
        Observed-frame wavelength at each spectral pixel [Angstrom].
    redshift: float
        Source redshift z.
    dl_cm: float
        Luminosity distance [cm].
    resolution: float, array, or None
        Spectral resolution :math:`R(\lambda) = \lambda / \Delta\lambda`.
        If ``None``, LSF is skipped. If scalar, constant resolution; if array,
        per-pixel wavelength-dependent resolution (e.g., JWST NIRSpec PRISM).
    sigma_lib_kms: float, optional
        SSP library velocity dispersion [km/s], subtracted in quadrature from
        instrument LSF. Default 0.0 (no subtraction). Common values: MILES 70 km/s,
        C3K 15 km/s.
    n_bins: int, optional
        Number of piecewise-constant segments for variable-R LSF approximation.
        Ignored when resolution is scalar. Default 16.
    sigma_v_kms: float, optional
        Intrinsic galaxy velocity dispersion [km/s] added in quadrature to LSF.
        Default 0.0 (no extra broadening).
    cal_coeffs: array, shape (order,), or None, optional
        Chebyshev calibration polynomial coefficients ``[a_1, ..., a_N]``,
        where ``N`` is the calibration order. If ``None`` (default), no
        calibration is applied. Empty array ``[]`` gives unity calibration
        (no-op).
    cal_wave_range: tuple[float, float], optional
        ``(wave_min, wave_max)`` wavelength range for normalizing the
        Chebyshev polynomial to [-1, 1]. Used only when ``cal_coeffs`` is not
        ``None``. If omitted, defaults to ``(wave_obs.min(), wave_obs.max())``.
    conserving: bool, optional
        Resample the model onto the pixel grid with a flux-conserving bin
        integral (:func:`compute_spectrum_conserving`) instead of point
        interpolation. Default ``False`` (point sampling, unbiased only when
        the model grid is much finer than the pixels). Set for low-resolution
        spectroscopy where point sampling aliases; see #1166.
    resolution_matrix: BandedMatrix or None, optional
        Banded instrument resolution operator (DESI/PFS spectro-perfectionism;
        Bolton & Schlegel 2010). When supplied, the flux-conserving-resampled
        model is projected through ``R @ model`` at pixel resolution and this
        **replaces** the Gaussian ``apply_lsf``, the matrix already encodes the
        true LSF (the Redrock/FastSpecFit convention). Default ``None`` (Gaussian
        LSF from ``resolution``). See :func:`~tengri.observation.banded.banded_matvec`.
        #1163.

    Returns
    -------
    ndarray, shape (n_pix,)
        Observed spectral flux density [erg/s/cm²/Hz] at each pixel, optionally
        broadened by the instrument LSF and scaled by the calibration polynomial.

    Notes
    -----
    **JIT-compatible**: yes when `resolution`'s None-ness and
    `cal_coeffs`'s None-ness are fixed at trace time. Both None checks are
    Python-level structural branches.

    **What this does**: Projects the panchromatic model-grid SED onto an
    instrument wavelength grid, the result is a *spectrum* (observed-frame F_nu
    on `wave_obs`), distinct from the model-grid SED itself.

    **Composition pattern**: Called by observers/projectors that (1) may apply
    IGM/DLA attenuation BEFORE calling this function, (2) flux calibration is
    applied here (when ``cal_coeffs`` is provided), and (3) may apply Milky Way
    reddening BEFORE calling this function. Calibration order (after LSF) is
    non-negotiable: the polynomial models wavelength-dependent instrumental
    flux-calibration error on the observed, already-smoothed spectrum.

    **Calibration convention**:

    .. math::

        C(\lambda) = 1 + \sum_{n=1}^{N} c_n \, T_n(x), \qquad
        x = \frac{2\lambda - \lambda_{\min} - \lambda_{\max}}
                 {\lambda_{\max} - \lambda_{\min}}

    where :math:`T_n` are Chebyshev polynomials of the first kind, :math:`x` maps
    ``cal_wave_range`` onto :math:`[-1, 1]` (dimensionless), :math:`c_n` are the
    coefficients ``cal_c1..cal_cN`` (dimensionless), and :math:`C(\lambda)` is a
    dimensionless multiplicative correction to the observed spectrum.

    The constant term is **fixed** at :math:`c_0 = 1`: a free constant is
    degenerate with the model's overall normalization (stellar mass), so the
    coefficients describe only the *wavelength-dependent* part of the calibration
    error. This is a deliberate difference from Prospector [1]_, whose
    ``PolyOptCal`` instead solves for every coefficient including the constant by
    least squares against the data; tengri samples ``cal_c1..cal_cN`` under
    explicit priors, or marginalizes them analytically (see
    :func:`~tengri.observation.calibration.marginalize_calibration`). The
    multiplicative Chebyshev form and its application *after* instrumental
    smoothing follow Prospector.

    See :func:`~tengri.observation.calibration.calibration_polynomial`.

    References
    ----------
    .. [1] Johnson, B. D., Leja, J., Conroy, C., & Speagle, J. S. (2021).
           "Stellar Population Inference with Prospector."
           ApJS, 254, 22. arXiv:2012.01426.

    See Also
    --------
    compute_spectrum: Compute observed spectrum (no LSF).
    apply_lsf: Apply LSF convolution separately.
    velocity_broaden: Broaden by velocity dispersion only.
    apply_calibration: Apply calibration polynomial to a spectrum.

    """
    from tengri.observation.calibration import apply_calibration

    # ``conserving`` is a static structural flag (resolved from the ``resample``
    # mode before the trace), so this branch resolves at trace time (#1166).
    resampler = compute_spectrum_conserving if conserving else compute_spectrum
    flux = resampler(sed_rest, wave_rest, wave_obs, redshift, dl_cm)
    if resolution_matrix is not None:
        # The banded resolution matrix (DESI/PFS spectro-perfectionism; Bolton &
        # Schlegel 2010) encodes the true instrument LSF at pixel resolution and
        # is applied to the model *after* resampling onto the pixel grid, it
        # REPLACES the Gaussian ``apply_lsf`` (Redrock/FastSpecFit convention).
        # ``resolution_matrix`` is static structural config, so this branch
        # resolves at trace time. #1163.
        from tengri.observation.banded import banded_matvec

        flux = banded_matvec(resolution_matrix.offsets, resolution_matrix.data, flux)
    elif resolution is not None:
        flux = apply_lsf(
            flux,
            wave_obs,
            resolution,
            sigma_lib_kms=sigma_lib_kms,
            n_bins=n_bins,
            sigma_v_kms=sigma_v_kms,
        )
    if cal_coeffs is not None:
        wmin, wmax = (
            cal_wave_range if cal_wave_range is not None else (wave_obs.min(), wave_obs.max())
        )
        flux = apply_calibration(flux, wave_obs, cal_coeffs, wmin, wmax)
    return flux


@jax.jit
def compute_spectrum(
    sed_rest: jnp.ndarray,
    wave_rest: jnp.ndarray,
    wave_obs: jnp.ndarray,
    redshift: float,
    dl_cm: float,
) -> jnp.ndarray:
    """Compute observed spectrum at arbitrary pixel wavelengths.

    Maps observed wavelengths back to rest-frame coordinates (accounting for
    redshift), evaluates the rest-frame SED at those wavelengths via interpolation,
    and scales to observed flux using luminosity distance and (1+z) redshift factor.

    Parameters
    ----------
    sed_rest: array, shape (n_wave,)
        Rest-frame spectral luminosity density [erg/s/Hz] on the
        rest-frame wavelength grid.
    wave_rest: array, shape (n_wave,)
        Rest-frame wavelength grid [Angstrom].
    wave_obs: array, shape (n_pix,)
        Observed-frame wavelength at each spectral pixel [Angstrom].
    redshift: float
        Source redshift z.
    dl_cm: float
        Luminosity distance [cm].

    Returns
    -------
    ndarray, shape (n_pix,)
        Model spectral flux density [erg/s/cm²/Hz] at each pixel.

    Notes
    -----
    JIT-compatible: yes, all operations are ``jnp`` primitives.
    Gradient-safe: yes, differentiable w.r.t. redshift and dl_cm.

    Uses linear interpolation (``jnp.interp``) to evaluate the rest-frame
    SED at rest-frame wavelengths corresponding to observed pixel wavelengths.
    SED is clamped to zero outside the wavelength domain.

    References
    ----------
    Standard cosmological flux conversion: observer-frame flux density is
    derived from rest-frame spectral luminosity density via (1+z) dimming
    and inverse-square-law scaling with luminosity distance.

    """
    # Map observed wavelengths to rest-frame
    wave_rest_query = wave_obs / (1.0 + redshift)

    # Interpolate rest-frame SED
    sed_at_pixels = jnp.interp(wave_rest_query, wave_rest, sed_rest, left=0.0, right=0.0)

    # Apply the (1+z)/(4π d_L²) dimming to the pixel SED directly. A standalone
    # ``flux_scale = lnu_to_fnu(1.0, ...)`` is ~1e-58 and underflows float32 to
    # zero (peak 1.0 absorbs none of the -58 decades); applied to sed_at_pixels
    # (~1e30) apply_log10_scale folds the offset into the array peak and the
    # result stays in range. Identical in float64 (#1206).
    return lnu_to_fnu(sed_at_pixels, dl_cm, redshift)


def _flux_conserving_resample(
    wave_rest: jnp.ndarray, sed_rest: jnp.ndarray, wave_query: jnp.ndarray
) -> jnp.ndarray:
    r"""Bin-integrated (flux-conserving) resample of ``sed_rest`` onto ``wave_query``.

       Each output value is the *mean flux density over that pixel's wavelength bin*,
    the integral of the model over the bin divided by the bin width, rather
       than a point sample at the pixel center (Carnall 2017, SpectRes, eq. 3):

       .. math::

           \tilde{f}_j = \frac{1}{\Delta\lambda_j}
                         \int_{\lambda_j^-}^{\lambda_j^+} f(\lambda)\, d\lambda

       where the bin edges :math:`\lambda_j^\pm` are the midpoints between adjacent
       ``wave_query`` centers. Point sampling is unbiased only when the model grid is
       much finer than the pixel spacing; when a pixel spans one or more model bins
       (low-resolution spectroscopy, e.g. NIRSpec PRISM) it aliases the sub-pixel
       structure, biasing the integrated continuum. The bin integral does not.

       Implemented as a difference of the cumulative trapezoidal integral evaluated
       at the bin edges, so it is O(n_wave + n_pix), JIT-compatible, and
       differentiable w.r.t. ``sed_rest`` (the edges are static; only the SED varies).
       Outside the model grid the cumulative integral is flat, so out-of-range bins
       contribute zero, matching ``compute_spectrum``'s ``left=0, right=0`` clamp.

       Parameters
       ----------
       wave_rest: array, shape (n_wave,)
           Rest-frame model wavelength grid [Angstrom], strictly increasing.
       sed_rest: array, shape (n_wave,)
           Rest-frame flux density on ``wave_rest`` [erg/s/Hz].
       wave_query: array, shape (n_pix,)
           Rest-frame pixel-center wavelengths to resample onto [Angstrom].

       Returns
       -------
       ndarray, shape (n_pix,)
           Bin-averaged flux density at each pixel [erg/s/Hz].

       Notes
       -----
       **JIT-compatible**: yes. **Gradient-safe**: yes (linear in ``sed_rest``).

       References
       ----------
       .. [1] Carnall, A. C. 2017, "SpectRes: A Fast Spectral Resampling Tool in
              Python", arXiv:1705.05165.
    """
    mid = 0.5 * (wave_query[1:] + wave_query[:-1])
    lo = wave_query[:1] - 0.5 * (wave_query[1:2] - wave_query[:1])
    hi = wave_query[-1:] + 0.5 * (wave_query[-1:] - wave_query[-2:-1])
    edges = jnp.concatenate([lo, mid, hi])  # (n_pix + 1,)

    # Cumulative trapezoidal integral of the model, clamped flat outside the grid.
    cum = jnp.concatenate(
        [jnp.zeros(1), jnp.cumsum(0.5 * (sed_rest[1:] + sed_rest[:-1]) * jnp.diff(wave_rest))]
    )
    flux_at_edges = jnp.interp(edges, wave_rest, cum)
    return (flux_at_edges[1:] - flux_at_edges[:-1]) / (edges[1:] - edges[:-1])


@jax.jit
def compute_spectrum_conserving(
    sed_rest: jnp.ndarray,
    wave_rest: jnp.ndarray,
    wave_obs: jnp.ndarray,
    redshift: float,
    dl_cm: float,
) -> jnp.ndarray:
    """Flux-conserving twin of :func:`compute_spectrum`.

    Identical cosmological scaling, but resamples the rest-frame SED onto the
    observed pixels with a bin integral (:func:`_flux_conserving_resample`)
    instead of point interpolation. Use for low-resolution spectroscopy
    (``Spectroscopy(resample="conserving")`` or ``"auto"``), where point sampling
    aliases the sub-pixel structure; see :func:`_flux_conserving_resample`.

    Parameters and returns match :func:`compute_spectrum`.

    Notes
    -----
    **JIT-compatible**: yes. **Gradient-safe**: yes.
    """
    wave_rest_query = wave_obs / (1.0 + redshift)
    sed_at_pixels = _flux_conserving_resample(wave_rest, sed_rest, wave_rest_query)
    # Dimming applied to the pixel SED directly, not as a standalone flux_scale
    # (~1e-58, which underflows float32 to zero). See _resample_to_spectrum
    # above and #1206.
    return lnu_to_fnu(sed_at_pixels, dl_cm, redshift)


#: Fractional spread in ``d(ln lambda)`` tolerated before a grid is called
#: non-log-uniform. A genuine ``logspace`` grid lands ~1e-14 here; a linear grid
#: over 4500-5500 A lands ~0.2, so there is four orders of magnitude of daylight
#: between the two and the threshold is not a tuning knob.
_LOG_UNIFORM_RTOL = 1e-6


def _require_log_uniform_grid(wave, caller: str) -> None:
    """Raise unless ``wave`` is uniform in ``ln(lambda)`` (#1742).

    A no-op when ``wave`` is a tracer: a traced grid has no values to inspect at
    trace time. That is a real gap rather than a safe default, it is narrow
    because a spectroscopic wavelength grid is normally a fixed instrument array
    closed over by the jitted function, not an argument traced through it.

    Parameters
    ----------
    wave: array_like, shape (n_pix,)
        Wavelength grid to check [Angstrom].
    caller: str
        Function name, quoted in the error so the message names the API the user
        actually called.
    """
    if isinstance(wave, jax.core.Tracer):
        return
    w = np.asarray(wave, dtype=np.float64)
    if w.size < 3 or not np.all(np.isfinite(w)) or np.any(w <= 0.0):
        return  # not a grid this check can speak about; let the caller fail
    dln = np.diff(np.log(w))
    mean = float(np.mean(dln))
    if mean == 0.0:
        return
    spread = float(np.ptp(dln) / abs(mean))
    if spread <= _LOG_UNIFORM_RTOL:
        return
    # Report the size of the error, not just its existence: the under-broadening
    # is a constant factor wave[0]/lambda, so quote it at the array center.
    lam_mid = float(w[w.size // 2])
    factor = float(w[0]) / lam_mid
    raise ValueError(
        f"{caller} requires a wavelength grid uniform in ln(lambda), but this "
        f"grid's d(ln lambda) varies by a fraction {spread:.3g} across the array "
        f"(tolerance {_LOG_UNIFORM_RTOL:g}), a linearly-spaced grid does this. "
        f"The convolution is a constant Gaussian in ln(lambda), so one FFT is "
        f"correct only on a log grid; on this one the broadening would come out "
        f"low by about {factor:.4g}x at {lam_mid:.1f} A (issue #1742), with no "
        f"other symptom. Resample onto a log grid first, e.g. "
        f"wave_log = jnp.logspace(jnp.log10(wave[0]), jnp.log10(wave[-1]), "
        f"wave.size), interpolate the flux onto it, broaden, and interpolate back."
    )


def velocity_broaden(
    flux: jnp.ndarray,
    wave: jnp.ndarray,
    sigma_km_s: float,
) -> jnp.ndarray:
    """Broaden a spectrum by stellar velocity dispersion.

    Convolves with a Gaussian in log-wavelength space (equivalent to
    velocity space: Δv/c = Δln(λ)). Uses FFT convolution for speed.

    Parameters
    ----------
    flux: array, shape (n_pix,)
        Input spectral flux.
    wave: array, shape (n_pix,)
        Wavelength grid [Angstrom]. Must be uniformly spaced **in
        log-wavelength**, e.g. ``jnp.logspace(...)``, not ``jnp.linspace(...)``.
        A linearly-spaced grid is rejected; see Notes.
    sigma_km_s: float
        Velocity dispersion [km/s]. Typical range: 50–300 km/s.

    Returns
    -------
    ndarray, shape (n_pix,)
        Broadened spectrum (same units as input).

    Raises
    ------
    ValueError
        If ``wave`` is not uniform in ``ln(lambda)`` and is concrete at trace
        time. Resample onto a log grid first.

    Notes
    -----
    JIT-compatible: yes. Gradient-safe: yes.

    The convolution is a *constant* Gaussian in ``ln(lambda)``, because
    ``Delta v / c = Delta ln(lambda)``. That is what makes one FFT correct for
    the whole array, and it is exact only when the grid is uniform in
    ``ln(lambda)``.

    **On a linear grid the result is wrong by a constant factor**
    ``wave[0] / lambda_line`` (issue #1742). ``d(ln lambda) = d(lambda)/lambda``
    varies as ``1/lambda`` there, so a width read off the first pixel pair sets
    the kernel by the *bluest* pixel while the feature sits elsewhere. Measured
    on ``linspace(4500, 5500, 4096)`` with a line at 5000 A: 100 km/s recovered
    as 90.2, 500 as 450.1, a constant 0.900 = 4500/5000. The error scales with
    the wavelength *range*, not the pixel count, so refining the grid does not
    help: across 3000–10000 A a line at 9000 A would be broadened to 0.33 of the
    requested width, and a fitted velocity dispersion inherits that smoothly,
    with nothing looking broken.

    This previously passed silently, and the Parameters section said "uniformly
    spaced", instructing users to do the thing that breaks it. It now raises
    instead, on the reasoning recorded in :mod:`tengri.forward.approx_policy`:
    a silently-defaulting read is worse than a loud failure.

    The check is skipped when ``wave`` is a tracer, since a traced grid cannot
    be inspected at trace time. Under ``jax.jit`` with a concrete (closed-over)
    grid, the usual case for a fixed instrument grid, it still fires.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from tengri.observation.spectrum import velocity_broaden
    >>> wave = jnp.logspace(jnp.log10(4500.0), jnp.log10(5500.0), 256)
    >>> flux = jnp.ones_like(wave)
    >>> out = velocity_broaden(flux, wave, 150.0)
    >>> out.shape == wave.shape
    True

    """
    _require_log_uniform_grid(wave, "velocity_broaden")
    return _velocity_broaden_impl(flux, wave, sigma_km_s)


@jax.jit
def _velocity_broaden_impl(
    flux: jnp.ndarray,
    wave: jnp.ndarray,
    sigma_km_s: float,
) -> jnp.ndarray:
    """FFT convolution for :func:`velocity_broaden`, with the check already done.

    Split out so the grid check runs on concrete values: the public function was
    itself ``@jax.jit``, which makes ``wave`` a tracer inside it, and a guard that
    can never see its argument is not a guard (#1742).
    """
    sigma_v = sigma_km_s / _C_KM_S  # fractional velocity dispersion

    # Pixel scale in log-wavelength. Constant across the array precisely because
    # the grid is uniform in ln(lambda), checked by the caller, not assumed.
    dlnwave = jnp.log(wave[1] / wave[0])

    # Gaussian kernel width in pixels
    sigma_pix = sigma_v / dlnwave

    # Build Gaussian kernel in Fourier space (faster than real-space)
    n = len(flux)
    freq = jnp.fft.rfftfreq(n)
    kernel_ft = jnp.exp(-2.0 * jnp.pi**2 * sigma_pix**2 * freq**2)

    # FFT convolution
    flux_ft = jnp.fft.rfft(flux)
    broadened = jnp.fft.irfft(flux_ft * kernel_ft, n=n)

    return broadened


# ── Speed of light in Angstrom/s (for frequency conversions) ──────
from tengri.utils.physics_constants import C_AA as _C_AA_PER_S


@jax.jit
def blend_emission_lines(
    line_wavelengths: jnp.ndarray,
    line_luminosities: jnp.ndarray,
    spectral_resolution: float,
    wave_out: jnp.ndarray,
    redshift: float = 0.0,
) -> jnp.ndarray:
    """Place emission lines onto a wavelength grid, blending by instrument resolution.

    Each line is represented as a Gaussian whose width is set by the
    instrument's spectral resolution R = lambda / delta_lambda. Lines
    closer than delta_lambda are effectively blended. The output is in
    L_sun/Hz, ready to be added to a continuum SED.

    Vectorized over all lines simultaneously using ``jax.vmap`` for
    efficient GPU/TPU execution.

    Parameters
    ----------
    line_wavelengths: array, shape (n_lines,)
        Rest-frame line wavelengths [Angstrom].
    line_luminosities: array, shape (n_lines,)
        Line luminosities [L_sun]. Total integrated luminosity per line.
    spectral_resolution: float
        Instrument spectral resolution R = lambda / delta_lambda (dimensionless).
        Typical values: R ~ 100 (photometry), R ~ 1000 (low-res spectroscopy),
        R ~ 5000 (medium-res).
    wave_out: array, shape (n_pix,)
        Output wavelength grid [Angstrom] in observed frame.
    redshift: float, optional
        Source redshift. Default 0.0.

    Returns
    -------
    ndarray, shape (n_pix,)
        Emission-line spectrum [L_sun/Hz] on the output grid.
        Add to a continuum SED (also in L_sun/Hz) before applying
        cosmological dimming.

    Notes
    -----
    JIT-compatible: yes, vmapped over lines. Gradient-safe: yes.

    The Gaussian FWHM at each line is FWHM = lambda_obs / R, giving
    sigma = lambda_obs / (2.3548 * R). The profile is normalized to
    integrate to 1 in wavelength space. Luminosity is converted from
    L_sun (wavelength-integrated) to L_sun/Hz (spectral density).

    """

    def _single_line(lam_rest, lum):
        r"""Compute Gaussian profile for one line.

        Parameters
        ----------
        lam_rest: scalar
            Rest-frame wavelength (Angstrom).
        lum: scalar
            Line luminosity (Lsun).

        Returns
        -------
        array, shape (n_pix,)
            Contribution to the spectrum (Lsun/Hz).

        """
        lam_obs = lam_rest * (1.0 + redshift)
        sigma_aa = lam_obs / (_FWHM_TO_SIGMA * spectral_resolution)

        # Gaussian profile normalized in wavelength space: integral = 1
        profile = jnp.exp(-0.5 * ((wave_out - lam_obs) / sigma_aa) ** 2) / (
            jnp.sqrt(2.0 * jnp.pi) * sigma_aa
        )

        # Convert Lsun (integrated over wavelength) to Lsun/Hz:
        # delta_nu = c / lam_obs^2 * sigma_aa  (characteristic freq width)
        # profile_nu = lum * profile_lambda / delta_nu
        # But more directly: profile is normalized in lambda, so
        # L_lambda = lum * profile  [Lsun/AA]
        # L_nu = L_lambda * lambda^2 / c  [Lsun/Hz]
        # At each pixel: L_nu = lum * profile * wave_out^2 / c
        return lum * profile * wave_out**2 / _C_AA_PER_S

    # Vectorize over all lines and sum
    all_profiles = jax.vmap(_single_line)(line_wavelengths, line_luminosities)
    return jnp.sum(all_profiles, axis=0)
