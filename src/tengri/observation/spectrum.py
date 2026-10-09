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

import math
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from tengri.components.lyc import LYMAN_LIMIT_AA, edge_interp, ionizing_mask
from tengri.units import lnu_to_fnu

# ── SSP library spectral resolutions (velocity dispersion in km/s) ──
# Flat fallback used only when the loaded SSP grid has no per-wavelength
# resolution curve (SSPData.ssp_resolution_kms, #2518); see
# tengri.components.stellar.sps.dsps_wrapper for the curve itself, read
# from FSPS's own per-node tables.
SSP_LIBRARY_RESOLUTIONS: dict[str, float] = {
    "miles": 70.0,  # mid-range approximation; true resolution runs ~92-43 km/s
    "c3k_a": 42.4,  # R = 3000 over the shipped grid's densest 2750-9100 A core
    "c3k": 42.4,  # alias of c3k_a
    "fsps_default": 70.0,  # MILES-based (default FSPS)
}


def first_invalid_wavelength(
    w: np.ndarray,
) -> tuple[int, str] | None:
    """Return the first offending wavelength index and the violation rule.

    Checks for non-finite, non-positive, and non-increasing violations in order.

    Parameters
    ----------
    w : ndarray, shape (n,)
        Wavelength grid [Angstrom].

    Returns
    -------
    tuple[int, str] | None
        (first_offending_index, rule_violated) or None if valid. Rules are
        "non-finite" (NaN/inf), "non-positive" (<=0), "non-increasing" (not
        strictly monotonic).
    """
    if not np.all(np.isfinite(w)):
        idx = int(np.where(~np.isfinite(w))[0][0])
        return idx, "non-finite"
    if np.any(w <= 0.0):
        idx = int(np.where(w <= 0.0)[0][0])
        return idx, "non-positive"
    if len(w) > 1 and not np.all(np.diff(w) > 0.0):
        idx = int(np.where(np.diff(w) <= 0.0)[0][0])
        return idx, "non-increasing"
    return None


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
    wave_um : array, shape (n_wave,)
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
    wave_um : array, shape (n_wave,)
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
    resolution : array or scalar
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
    wave : array_like, shape (n_pix,)
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
    if w.size < 3 or first_invalid_wavelength(w) is not None:
        return True  # not a grid this helper can speak about; let the caller fail
    dln = np.diff(np.log(w))
    mean = float(np.mean(dln))
    if mean == 0.0:
        return True
    return bool(float(np.ptp(dln) / abs(mean)) <= _LOG_UNIFORM_RTOL)


# ── Padded, fast-length, windowed Gaussian convolution (#2832) ─────

#: Kernel half-width, in Gaussian sigmas, kept as padding beyond each spectrum edge.
#: Truncating a Gaussian at 5 sigma drops about 6e-7 of its weight.
_LSF_KERNEL_HALF_WIDTH_SIGMAS = 5.0

#: Minimum padding [pixels] per side (#2832). The kernel is sampled in Fourier space,
#: so for a sub-pixel sigma its spatial tails (the ringing of the Nyquist cut) decay
#: only as h**-2 and are not covered by 5 sigma. Measured against a converged
#: reference (sigma_pix 0.28 and 0.66): h = 8 gives 3e-4, 64 gives 4e-6, 128 gives
#: 9e-7, 256 gives 1.5e-7. At the production grid (n = 7909, 2000 km/s) the 5-sigma
#: bound is 265 pixels, so the floor adds nothing there.
_LSF_RINGING_FLOOR_PIXELS = 256

#: Bound [km/s] on any component of the Gaussian sigma that is traced at trace time.
#: A traced sigma has no value to read, so the padding assumes it is at most this
#: (#2832). A concrete sigma is measured, so this bound only applies to traced ones.
_LSF_MAX_SIGMA_KMS = 2000.0


class _BinLayout(NamedTuple):
    """Static (NumPy) layout of the piecewise bins of :func:`_apply_lsf_variable_r`.

    ``windowed`` selects short per-bin transforms over each bin's own window;
    otherwise every bin is transformed at the full padded length. ``sym`` is the
    symmetric margin added to the spectrum on each side.
    """

    windowed: bool
    centers: np.ndarray
    half_w: float
    bin_width: float
    core_start: np.ndarray | None
    core_len: int
    seg_len: int
    sym: int


def _next_fast_fft_len(n: int) -> int:
    """Smallest integer ``>= n`` whose only prime factors are 2, 3 and 5.

    Parameters
    ----------
    n : int
        Lower bound on the transform length.

    Returns
    -------
    int
        The smallest 5-smooth integer that is at least ``n``.

    Notes
    -----
    Private helper. Pure Python and static, evaluated at trace time.
    """
    m = max(int(n), 1)
    while True:
        k = m
        for p in (2, 3, 5):
            while k % p == 0:
                k //= p
        if k == 1:
            return m
        m += 1


def _is_traced(x) -> bool:
    """Whether ``x`` is a tracer, so its value is unknown at trace time."""
    return isinstance(x, jax.core.Tracer)


def _lsf_sigma_bound_kms(resolution=None, sigma_lib_kms=None, sigma_v_kms=None) -> float:
    """Upper bound [km/s] on the Gaussian sigma the convolution uses (#2832).

    The kernel's sigma is ``sqrt(max(sigma_inst**2 - sigma_lib**2, 0) + sigma_v**2)``
    with ``sigma_inst = c / (2.3548 R)``, the same quadrature as :func:`apply_lsf`.
    When every input is concrete, the bound is the exact maximum of that sigma over
    pixels. When an input is traced, it is assumed to be at most
    :data:`_LSF_MAX_SIGMA_KMS`; the concrete components are measured and combined in
    quadrature with it. ``sigma_lib`` only subtracts, so the traced branch drops it,
    which keeps the bound an upper one.

    Parameters
    ----------
    resolution : float, array or None
        Spectral resolution R, or None when there is no instrument term.
    sigma_lib_kms : float, array or None
        Library dispersion [km/s], or None.
    sigma_v_kms : float, array or None
        Intrinsic velocity dispersion [km/s], or None.

    Returns
    -------
    float
        Sigma bound [km/s].

    Notes
    -----
    Private helper, evaluated at trace time from the raw inputs of a public call.
    """
    inputs = (resolution, sigma_lib_kms, sigma_v_kms)
    if not any(_is_traced(x) for x in inputs):
        sigma_inst = (
            np.zeros(1)
            if resolution is None
            else _C_KM_S / (_FWHM_TO_SIGMA * np.atleast_1d(np.asarray(resolution, np.float64)))
        )
        lib = (
            np.zeros(1)
            if sigma_lib_kms is None
            else np.atleast_1d(np.asarray(sigma_lib_kms, np.float64))
        )
        v = (
            np.zeros(1)
            if sigma_v_kms is None
            else np.atleast_1d(np.maximum(np.asarray(sigma_v_kms, np.float64), 0.0))
        )
        eff = np.sqrt(np.maximum(sigma_inst**2 - lib**2, 0.0) + v**2)
        return float(np.max(eff))
    squares = 0.0
    if resolution is not None:
        if _is_traced(resolution):
            squares += _LSF_MAX_SIGMA_KMS**2
        else:
            r = np.atleast_1d(np.asarray(resolution, np.float64))
            squares += float(np.max(_C_KM_S / (_FWHM_TO_SIGMA * r))) ** 2
    if sigma_v_kms is not None:
        if _is_traced(sigma_v_kms):
            squares += _LSF_MAX_SIGMA_KMS**2
        else:
            v = np.atleast_1d(np.maximum(np.asarray(sigma_v_kms, np.float64), 0.0))
            squares += float(np.max(v)) ** 2
    return math.sqrt(squares)


def _variable_min_dln(w: np.ndarray, n: int, n_bins: int) -> float:
    """Smallest bin-mean ``d ln lambda`` over the piecewise bins (#2832).

    Each bin's kernel reads ``d ln lambda`` averaged over the pixels within one bin
    width of its center (:func:`_bin_sigma_and_dln`). The smallest such mean sets the
    widest kernel in pixels, so it is the scale the padding must cover. A single
    near-duplicate pixel barely moves a bin mean, so it does not shrink the padding.
    """
    dln = np.gradient(np.log(w))
    width = n / n_bins
    pix = np.arange(n, dtype=np.float64)
    centers = (np.arange(n_bins) + 0.5) * width
    mask = np.abs(pix[None, :] - centers[:, None]) < width
    counts = np.maximum(mask.sum(axis=1), 1.0)
    means = (mask * dln[None, :]).sum(axis=1) / counts
    return float(np.min(means))


def _lsf_pad_pixels(wave, n: int, sigma_kms: float, n_bins: int | None = None) -> int:
    """Kernel padding [pixels] per side, from the sigma the kernel uses (#2832).

    Parameters
    ----------
    wave : array_like, shape (n,), or tracer
        Wavelength grid [Angstrom].
    n : int
        Number of pixels.
    sigma_kms : float
        Upper bound on the Gaussian sigma [km/s], from :func:`_lsf_sigma_bound_kms`.
    n_bins : int or None, optional
        Piecewise bin count of the variable path. ``None`` is the constant and
        velocity path, which reads the first pair ``d ln lambda`` as its pixel scale.

    Returns
    -------
    int
        Padding per side [pixels]: ``ceil(5 sigma / (c d ln lambda))``, floored at
        :data:`_LSF_RINGING_FLOOR_PIXELS`.

    Notes
    -----
    The padding is 5 sigma of the kernel in pixels, at the pixel scale the kernel
    actually uses: the first pair for the constant path, the smallest bin-mean
    ``d ln lambda`` for the variable path (:func:`_variable_min_dln`). A traced grid
    has no pixel scale to read, so the exact choice ``n - 1`` is used. Symmetric
    padding accepts any width, so the padding is never capped below the kernel.
    """
    n = int(n)
    if n <= 1:
        return 0
    if _is_traced(wave):
        return n - 1
    w = np.asarray(wave, dtype=np.float64)
    if n_bins is None:
        dln = float(np.log(w[1] / w[0]))
    else:
        dln = _variable_min_dln(w, n, n_bins)
    if not dln > 0.0:
        return n - 1
    half = math.ceil(_LSF_KERNEL_HALF_WIDTH_SIGMAS * float(sigma_kms) / (_C_KM_S * dln))
    return int(max(half, _LSF_RINGING_FLOOR_PIXELS))


def _resolve_pad(pad_pixels, wave, n, sigma_bound, n_bins=None) -> int:
    """The explicit ``pad_pixels`` override if given, else the padding from the grid."""
    if pad_pixels is not None:
        return int(pad_pixels)
    return _lsf_pad_pixels(wave, n, sigma_bound, n_bins)


def _gaussian_fft_convolve(spectrum: jnp.ndarray, sigma_pix, pad: int) -> jnp.ndarray:
    """Gaussian convolution on a symmetric margin, transformed at a fast length.

    Parameters
    ----------
    spectrum : array, shape (n,)
        Input spectrum.
    sigma_pix : float or array
        Gaussian sigma [pixels].
    pad : int
        Symmetric margin per side [pixels] (static).

    Returns
    -------
    ndarray, shape (n,)
        Convolved spectrum, cropped to the original ``n`` pixels.

    Notes
    -----
    The zero tail between the margin and the fast transform length is harmless:
    the margin holds the kernel reach, so no core pixel sees the wrap.
    """
    n = spectrum.shape[0]
    length = _next_fast_fft_len(n + 2 * pad)
    padded = jnp.pad(spectrum, pad, mode="symmetric")
    freq = jnp.fft.rfftfreq(length)
    kernel_ft = jnp.exp(-2.0 * jnp.pi**2 * sigma_pix**2 * freq**2)
    conv = jnp.fft.irfft(jnp.fft.rfft(padded, n=length) * kernel_ft, n=length)
    return conv[pad : pad + n]


@partial(jax.jit, static_argnames=("pad",))
def _apply_lsf_constant_r(
    spectrum: jnp.ndarray,
    wave_obs: jnp.ndarray,
    sigma_eff_kms: float,
    *,
    pad: int,
) -> jnp.ndarray:
    """FFT convolution in log-wavelength space for constant R.

    This is equivalent to velocity_broaden but with the effective
    (library-subtracted) sigma.

    Parameters
    ----------
    spectrum : array, shape (n_pix,)
        Input spectral flux.
    wave_obs : array, shape (n_pix,)
        Observed wavelength grid [Angstrom]. Must be uniform in ``ln(lambda)``,
        not merely evenly spaced, the pixel scale is read once from the first
        pair, so a linear grid under-broadens by ``wave[0]/lambda`` (#1742).
        ``apply_lsf`` dispatches here only for a grid that satisfies this, and
        sends the rest to :func:`_apply_lsf_variable_r` (#1791), so the
        precondition binds only direct callers of this helper.
    sigma_eff_kms : float
        Effective velocity dispersion [km/s] (after library subtraction).
    pad : int
        Symmetric margin per side [pixels] (static), from :func:`_lsf_pad_pixels`.

    Returns
    -------
    ndarray, shape (n_pix,)
        Smoothed spectrum (same units as input).

    Notes
    -----
    JIT-compatible: yes. Private helper for apply_lsf. Gradient-safe: yes.
    """
    sigma_v = sigma_eff_kms / _C_KM_S
    dlnwave = jnp.log(wave_obs[1] / wave_obs[0])
    sigma_pix = sigma_v / dlnwave
    return _gaussian_fft_convolve(spectrum, sigma_pix, pad)


def _raised_cos_weight(pos, center, half_w):
    """Raised-cosine bin weight: 1 at ``center``, zero beyond ``half_w``."""
    dist = jnp.abs(pos - center) / half_w
    return jnp.where(dist < 1.0, 0.5 * (1.0 + jnp.cos(jnp.pi * dist)), 0.0)


def _bin_sigma_and_dln(sigma_eff_kms, dlnwave_local, pix_idx, center, bin_width):
    """Mean sigma and mean d ln lambda over the pixels within one bin width of ``center``.

    The two clamps are written out at each use rather than hoisted into a shared
    name: the zero-hiding audit (``tools/check_zero_hiding_clamps.py``) matches the
    division syntactically, and a hoisted clamp would drop out of its inventory.
    """
    bin_mask = jnp.where(jnp.abs(pix_idx - center) < bin_width, 1.0, 0.0)
    n_in_bin = jnp.sum(bin_mask)
    sigma_mean = jnp.sum(sigma_eff_kms * bin_mask) / jnp.maximum(n_in_bin, 1.0)
    dlnwave_mean = jnp.sum(dlnwave_local * bin_mask) / jnp.maximum(n_in_bin, 1.0)
    return sigma_mean, dlnwave_mean


def _variable_r_layout(n_pix: int, n_bins: int, pad: int, bin_ids) -> _BinLayout:
    """Static bin geometry for the piecewise path, from ``n_pix`` and ``n_bins`` only.

    Bins are the uniform pixel split of the spectrum. A bin's raised-cosine weight is
    nonzero only within ``0.75`` of a bin width of its center, so the windowed path
    transforms just that core plus ``pad`` pixels of margin, at a fast length.
    """
    width = n_pix / n_bins
    half_w = 0.75 * width
    all_centers = (np.arange(n_bins) + 0.5) * width
    ids = np.arange(n_bins) if bin_ids is None else np.asarray(bin_ids, dtype=int)
    centers = all_centers[ids]
    core_len = math.ceil(1.5 * width) + 2
    seg_win = _next_fast_fft_len(core_len + 2 * pad)
    seg_full = _next_fast_fft_len(n_pix + 2 * pad)
    if centers.size == 0 or seg_win >= seg_full:
        return _BinLayout(False, centers, half_w, width, None, n_pix, seg_full, pad)
    core_start = np.floor(centers - half_w).astype(int) - 1
    seg_start = core_start - pad
    sym = max(0, -int(seg_start.min()), int((seg_start + seg_win).max()) - n_pix)
    return _BinLayout(True, centers, half_w, width, core_start, core_len, seg_win, sym)


def _full_length_bins(spectrum, sigma_eff_kms, dlnwave_local, pix_idx, layout, pad):
    """Sum of the selected bins, each transformed over the whole padded spectrum."""
    n_pix = spectrum.shape[0]
    length = layout.seg_len
    flux_ft = jnp.fft.rfft(jnp.pad(spectrum, pad, mode="symmetric"), n=length)
    freq = jnp.fft.rfftfreq(length)

    def _convolve_bin(acc, center):
        sigma_mean, dlnwave_mean = _bin_sigma_and_dln(
            sigma_eff_kms, dlnwave_local, pix_idx, center, layout.bin_width
        )
        sigma_pix = (sigma_mean / _C_KM_S) / dlnwave_mean
        kernel_ft = jnp.exp(-2.0 * jnp.pi**2 * sigma_pix**2 * freq**2)
        conv = jnp.fft.irfft(flux_ft * kernel_ft, n=length)[pad : pad + n_pix]
        return acc + _raised_cos_weight(pix_idx, center, layout.half_w) * conv, None

    acc, _ = jax.lax.scan(_convolve_bin, jnp.zeros(n_pix), jnp.asarray(layout.centers))
    return acc


def _windowed_bins(spectrum, sigma_eff_kms, dlnwave_local, pix_idx, layout, pad):
    """Sum of the selected bins, each transformed over its own window and margin."""
    n_pix = spectrum.shape[0]
    sym = layout.sym
    seg_len = layout.seg_len
    core_len = layout.core_len
    padded = jnp.pad(spectrum, sym, mode="symmetric")
    freq = jnp.fft.rfftfreq(seg_len)
    core_offsets = jnp.arange(core_len, dtype=jnp.float64)
    xs = (
        jnp.asarray(layout.centers),
        jnp.asarray(layout.core_start - pad + sym),
        jnp.asarray(layout.core_start + sym),
        jnp.asarray(layout.core_start.astype(np.float64)),
    )

    def _window_bin(acc, x):
        center, seg_pos, core_pos, core_pix0 = x
        seg = jax.lax.dynamic_slice(padded, (seg_pos,), (seg_len,))
        sigma_mean, dlnwave_mean = _bin_sigma_and_dln(
            sigma_eff_kms, dlnwave_local, pix_idx, center, layout.bin_width
        )
        sigma_pix = (sigma_mean / _C_KM_S) / dlnwave_mean
        kernel_ft = jnp.exp(-2.0 * jnp.pi**2 * sigma_pix**2 * freq**2)
        conv = jnp.fft.irfft(jnp.fft.rfft(seg) * kernel_ft, n=seg_len)[pad : pad + core_len]
        contrib = _raised_cos_weight(core_pix0 + core_offsets, center, layout.half_w) * conv
        current = jax.lax.dynamic_slice(acc, (core_pos,), (core_len,))
        return jax.lax.dynamic_update_slice(acc, current + contrib, (core_pos,)), None

    acc, _ = jax.lax.scan(_window_bin, jnp.zeros(n_pix + 2 * sym), xs)
    return acc[sym : sym + n_pix]


@partial(jax.jit, static_argnames=("n_bins", "pad", "bin_ids"))
def _apply_lsf_variable_r(
    spectrum: jnp.ndarray,
    wave_obs: jnp.ndarray,
    sigma_eff_kms: jnp.ndarray,
    n_bins: int = 16,
    *,
    pad: int,
    bin_ids: tuple[int, ...] | None = None,
) -> jnp.ndarray:
    """Piecewise-constant LSF convolution for variable R.

    Splits the wavelength range into ``n_bins`` segments. Within each
    segment the mean effective sigma is used for an FFT convolution.
    The segments are blended with smooth (raised-cosine) overlap to
    avoid discontinuities. Accurate to approximately 1% for typical instrument
    profiles and fully differentiable.

    Parameters
    ----------
    spectrum : array, shape (n_pix,)
        Input spectral flux.
    wave_obs : array, shape (n_pix,)
        Observed wavelength grid [Angstrom]. Any strictly increasing grid: each
        bin takes its pixel scale from the local ``d ln lambda``, so a grid that
        is not log-uniform is handled here rather than refused (#1791).
    sigma_eff_kms : array, shape (n_pix,)
        Effective velocity dispersion at each pixel [km/s].
    n_bins : int, optional
        Number of piecewise-constant segments. More bins gives better
        accuracy but requires more FFTs. Typical: 10–20. Default 16.
    pad : int
        Symmetric margin per side [pixels] (static), from :func:`_lsf_pad_pixels`.
    bin_ids : tuple of int or None, optional
        Static subset of bin indices to sum (static). ``None`` sums all bins. A
        subset is exact on every pixel that all of its bins touch, and is used by
        :func:`broaden_velocity_only_window`; elsewhere the result is not meaningful.

    Returns
    -------
    ndarray, shape (n_pix,)
        Smoothed spectrum (same units as input).

    Notes
    -----
    JIT-compatible: yes, ``n_bins``, ``pad`` and ``bin_ids`` are static. Gradient-safe:
    yes. Private helper for apply_lsf.

    Each bin's transform is short: only its window (the pixels where its weight is
    nonzero) plus ``pad`` pixels of margin on each side, at a fast length. When that
    saves nothing, every bin is transformed over the whole padded spectrum instead.
    """
    n_pix = spectrum.shape[0]
    layout = _variable_r_layout(n_pix, n_bins, pad, bin_ids)
    # Local pixel scale, not the blue-end value for the whole array. Each bin
    # already convolves with its own sigma; giving it its own d(ln lambda) as
    # well is what lets this path serve a grid that is not log-uniform, where a
    # single global scale under-broadened by wave[0]/lambda (#1791). On a
    # log-uniform grid jnp.gradient returns that same constant, so nothing moves.
    dlnwave_local = jnp.gradient(jnp.log(wave_obs))
    pix_idx = jnp.arange(n_pix, dtype=jnp.float64)
    if layout.windowed:
        result = _windowed_bins(spectrum, sigma_eff_kms, dlnwave_local, pix_idx, layout, pad)
    else:
        result = _full_length_bins(spectrum, sigma_eff_kms, dlnwave_local, pix_idx, layout, pad)

    # Normalize by total weight at each pixel
    centers = jnp.asarray(layout.centers)
    total_weight = jnp.sum(
        _raised_cos_weight(pix_idx[None, :], centers[:, None], layout.half_w), axis=0
    )
    total_weight = jnp.maximum(total_weight, 1e-30)

    return result / total_weight


def _is_concrete_nonpositive(sigma) -> bool:
    """Whether ``sigma`` is a concrete scalar that is ``<= 0``, known at trace time (#2832).

    Python and NumPy scalars and 0-d NumPy arrays are concrete. A tracer or a jax
    array is not, even when its value happens to be known, so the static skip never
    depends on a runtime value.
    """
    if isinstance(sigma, (jax.core.Tracer, jax.Array)):
        return False
    if isinstance(sigma, (int, float, np.generic, np.ndarray)):
        arr = np.asarray(sigma)
        return arr.ndim == 0 and bool(arr <= 0)
    return False


def _bins_touching(n_pix: int, n_bins: int, lo: int, hi: int) -> tuple[int, ...]:
    """Indices of the bins whose weight window meets the pixel range ``[lo, hi)``."""
    width = n_pix / n_bins
    half_w = 0.75 * width
    centers = (np.arange(n_bins) + 0.5) * width
    keep = (centers + half_w >= lo) & (centers - half_w <= hi)
    return tuple(int(k) for k in np.flatnonzero(keep))


def broaden_velocity_only(
    flux: jnp.ndarray,
    wave: jnp.ndarray,
    sigma_v_kms: jnp.ndarray | float,
    n_bins: int = 16,
    *,
    pad_pixels: int | None = None,
) -> jnp.ndarray:
    r"""Convolve ``flux`` with the galaxy's own velocity dispersion alone (#2589).

    A pure :math:`\sigma_v` Gaussian in log-wavelength space, with no
    instrument or library term. Unlike :func:`velocity_broaden`, which
    requires ``wave`` uniform in :math:`\ln\lambda` (one global pixel
    scale), this dispatches to the piecewise machinery of
    :func:`apply_lsf`'s variable-resolution path
    (:func:`_apply_lsf_variable_r`), which reads its pixel scale locally
    (``jnp.gradient(jnp.log(wave))``) and so works on any strictly
    increasing grid -- in particular the tengri rest-frame model grid,
    which is log-uniform over the SSP library's native range but not
    over its sparser long-wavelength filler. On a grid that *is*
    log-uniform with a constant :math:`\sigma_v`, every piecewise bin
    shares one sigma and the raised-cosine blend weights sum to 1, so
    the result equals the single-FFT :func:`velocity_broaden` to
    floating-point precision.

    Used to broaden the stellar piece of the spectrum *before* IGM
    transmission is multiplied in
    (:func:`~tengri.observation.observation.project_spectrum_kernel_split`),
    and (since #2506) to give the banded resolution-matrix path its own
    galaxy-kinematics term ahead of ``R @ model``.

    Parameters
    ----------
    flux : ndarray, shape (n,)
        Spectrum to broaden [erg/s/Hz or erg/s/cm^2/Hz].
    wave : ndarray, shape (n,)
        Wavelength grid [Angstrom]. Any strictly increasing grid.
    sigma_v_kms : float or ndarray
        Velocity dispersion [km/s]. Non-positive is the identity. A concrete
        non-positive scalar (Python, NumPy, or 0-d NumPy) returns ``flux`` unchanged
        before any array work and emits no FFT (#2832). A traced or jax-array sigma
        keeps the ``jnp.where`` guard, so it stays jit/grad-safe.
    n_bins : int, default 16
        Piecewise-constant segment count (see :func:`_apply_lsf_variable_r`).
    pad_pixels : int or None, optional
        Static padding [pixels] per side. ``None`` derives it from the grid and the
        sigma: a concrete sigma is measured, and a traced sigma is assumed to be at
        most ``_LSF_MAX_SIGMA_KMS`` (2000 km/s). A traced grid pads by ``n - 1``.
        An explicit value is used as given.

    Returns
    -------
    ndarray, shape (n,)
        Broadened spectrum (same units as input).

    Notes
    -----
    **JIT-compatible**: yes. **Gradient-safe**: yes. The convolution is padded by a
    margin set from the grid (:func:`_lsf_pad_pixels`), so the kernel never wraps
    from one spectrum edge onto the other (#2712, #2832).
    """
    if _is_concrete_nonpositive(sigma_v_kms):
        return flux
    sigma_v = jnp.maximum(jnp.asarray(sigma_v_kms, dtype=jnp.asarray(flux).dtype), 0.0)
    pad = _resolve_pad(
        pad_pixels, wave, flux.shape[0], _lsf_sigma_bound_kms(sigma_v_kms=sigma_v_kms), n_bins
    )
    broadened = _apply_lsf_variable_r(
        flux, wave, jnp.broadcast_to(sigma_v, flux.shape), n_bins, pad=pad
    )
    return jnp.where(sigma_v > 0.0, broadened, flux)


def broaden_velocity_only_window(
    flux: jnp.ndarray,
    wave: jnp.ndarray,
    sigma_v_kms: jnp.ndarray | float,
    n_bins: int,
    lo: int,
    hi: int,
) -> jnp.ndarray:
    r""":func:`broaden_velocity_only` exact on the pixel range ``[lo, hi)`` only (#2832).

    Pixels in ``[lo, hi)`` equal the full-grid :func:`broaden_velocity_only` result to
    floating-point precision: every bin that touches them is summed, with the bin
    geometry of the full grid. Pixels outside the range are returned unbroadened. Use
    it when only that range is read downstream, as the rest-grid pass of the IGM
    branch does for the observed window.

    Parameters
    ----------
    flux : ndarray, shape (n,)
        Spectrum to broaden [erg/s/Hz or erg/s/cm^2/Hz].
    wave : ndarray, shape (n,)
        Wavelength grid [Angstrom], the full grid of ``flux``.
    sigma_v_kms : float or ndarray
        Velocity dispersion [km/s], handled as in :func:`broaden_velocity_only`.
    n_bins : int
        Piecewise-constant segment count of the full grid.
    lo, hi : int
        Pixel range ``[lo, hi)`` that must be exact, static Python ints.

    Returns
    -------
    ndarray, shape (n,)
        Broadened spectrum on ``[lo, hi)``, the input elsewhere.

    Notes
    -----
    **JIT-compatible**: yes, for a concrete ``wave``, ``lo`` and ``hi``. **Gradient-safe**:
    yes. The bin subset is static, so only the bins that touch ``[lo, hi)`` are
    transformed.
    """
    if _is_concrete_nonpositive(sigma_v_kms):
        return flux
    n_pix = flux.shape[0]
    lo, hi = max(int(lo), 0), min(int(hi), n_pix)
    if lo >= hi:
        return flux
    sigma_v = jnp.maximum(jnp.asarray(sigma_v_kms, dtype=jnp.asarray(flux).dtype), 0.0)
    pad = _lsf_pad_pixels(wave, n_pix, _lsf_sigma_bound_kms(sigma_v_kms=sigma_v_kms), n_bins)
    broadened = _apply_lsf_variable_r(
        flux,
        wave,
        jnp.broadcast_to(sigma_v, flux.shape),
        n_bins,
        pad=pad,
        bin_ids=_bins_touching(n_pix, n_bins, lo, hi),
    )
    windowed = jnp.concatenate([flux[:lo], broadened[lo:hi], flux[hi:]])
    return jnp.where(sigma_v > 0.0, windowed, flux)


def rest_grid_observed_window(wave_rest, wave_obs, redshift) -> tuple[int, int] | None:
    r"""Rest-grid pixel range that the observed grid reads, or ``None`` if not static (#2832).

    The observed window is :math:`[\lambda_{\rm obs,min}, \lambda_{\rm obs,max}]
    / (1 + z)` widened by one largest observed pixel. The result is padded by
    :func:`_lsf_pad_pixels` rest pixels on each side, so interpolation and flux
    integration at the window edges also read exact values.

    Parameters
    ----------
    wave_rest : array_like, shape (n_wave,)
        Rest-frame wavelength grid [Angstrom].
    wave_obs : array_like, shape (n_pix,)
        Observed-frame wavelength grid [Angstrom].
    redshift : float
        Source redshift z.

    Returns
    -------
    tuple[int, int] or None
        ``(lo, hi)`` with ``0 <= lo <= hi <= n_wave``, or ``None`` when any argument is
        a tracer, in which case the window cannot be read at trace time.

    Notes
    -----
    Private helper. Build-time (NumPy). Its result is a static slice, so the caller
    must treat a ``None`` as "use the full grid".
    """
    if any(isinstance(x, jax.core.Tracer) for x in (wave_rest, wave_obs, redshift)):
        return None
    wr = np.asarray(wave_rest, dtype=np.float64)
    wo = np.asarray(wave_obs, dtype=np.float64)
    z = float(np.asarray(redshift, dtype=np.float64))
    n_wave = wr.shape[0]
    step = float(np.max(np.diff(wo))) if wo.size > 1 else 0.0
    lam_lo = (float(wo.min()) - step) / (1.0 + z)
    lam_hi = (float(wo.max()) + step) / (1.0 + z)
    pad = _lsf_pad_pixels(wr, n_wave, _LSF_MAX_SIGMA_KMS)
    lo = int(np.searchsorted(wr, lam_lo, side="right")) - 1 - pad
    hi = int(np.searchsorted(wr, lam_hi, side="left")) + 1 + pad
    return max(lo, 0), min(hi, n_wave)


def resolve_sigma_lib_kms(
    wave_obs: jnp.ndarray,
    redshift: jnp.ndarray | float,
    sigma_lib_kms: jnp.ndarray | float,
    sigma_lib_curve: tuple[jnp.ndarray, jnp.ndarray] | None = None,
) -> jnp.ndarray | float:
    r"""Resolve ``sigma_lib_kms`` for :func:`apply_lsf`/:func:`project_spectrum` (#2518).

    When ``sigma_lib_curve`` is given, interpolates the loaded SSP library's
    own per-wavelength resolution curve (:attr:`SSPData.ssp_resolution_kms`,
    tabulated on the library's rest-frame wavelength grid) onto the
    observed pixel grid mapped into the rest frame,
    :math:`\lambda_{\rm rest} = \lambda_{\rm obs} / (1 + z)`, and returns
    that per-pixel array in place of ``sigma_lib_kms``. Otherwise returns
    ``sigma_lib_kms`` unchanged -- the flat per-library scalar fallback
    (:attr:`~tengri.observation.spectroscopy.Spectroscopy.sigma_lib_kms`,
    used when the loaded library has no documented per-wavelength curve).

    Parameters
    ----------
    wave_obs : array, shape (n_pix,)
        Observed-frame wavelength grid [Angstrom].
    redshift : float or array
        Source redshift z.
    sigma_lib_kms : float or array, shape (n_pix,)
        Flat fallback library velocity dispersion [km/s], returned as-is
        when ``sigma_lib_curve`` is ``None``.
    sigma_lib_curve : tuple[array, array] or None, optional
        ``(ssp_wave_rest, ssp_resolution_kms)`` [Angstrom], [km/s] --
        :attr:`SSPData.ssp_wave` and :attr:`SSPData.ssp_resolution_kms` from
        the loaded SSP grid. ``None`` (default) skips the curve; that is a
        structural (pre-trace) choice, which SSP library is loaded is
        resolved once at model-build time, not per call on a traced value,
        so branching on ``sigma_lib_curve is None`` stays JIT/grad-safe.

    Returns
    -------
    ndarray, shape (n_pix,), or float
        Per-pixel σ_lib [km/s] when ``sigma_lib_curve`` is given, else
        ``sigma_lib_kms`` unchanged.

    Notes
    -----
    **JIT-compatible**: yes. **Gradient-safe**: yes with respect to
    ``redshift`` (``jnp.interp`` is differentiable); ``sigma_lib_curve``'s
    two arrays are ordinary JIT inputs, not differentiated through.

    See Also
    --------
    apply_lsf : Consumes the resolved ``sigma_lib_kms``.
    """
    if sigma_lib_curve is None:
        return sigma_lib_kms
    curve_wave_rest, curve_sigma_kms = sigma_lib_curve
    wave_rest = jnp.asarray(wave_obs) / (1.0 + jnp.asarray(redshift))
    return jnp.interp(wave_rest, curve_wave_rest, curve_sigma_kms)


def apply_lsf(
    spectrum: jnp.ndarray,
    wave_obs: jnp.ndarray,
    resolution: jnp.ndarray | float,
    sigma_lib_kms: jnp.ndarray | float = 0.0,
    n_bins: int = 16,
    sigma_v_kms: float = 0.0,
    *,
    pad_pixels: int | None = None,
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
    spectrum : array, shape (n_pix,)
        Input spectral flux at observed wavelengths [erg/s/cm²/Hz or arbitrary units].
    wave_obs : array, shape (n_pix,)
        Observed-frame wavelength grid [Ångstrom]. Any strictly increasing grid
        is accepted. One not uniform in log-wavelength, a linearly-spaced grid,
        for instance, is routed through the piecewise path, which carries a
        per-bin pixel scale, so the requested width is delivered either way
        (#1791). A log-uniform grid with scalar ``R`` keeps the single-FFT path.
    resolution : array, shape (n_pix,) or float
        Spectral resolution :math:`R(\\lambda) = \\lambda / \\Delta\\lambda`.

        - Scalar: constant resolution across wavelength (fast path)
        - Array: per-pixel wavelength-dependent resolution (e.g., JWST NIRSpec PRISM)

    sigma_lib_kms : float or array, shape (n_pix,), optional
        SSP library velocity dispersion [km/s]. Subtracted in quadrature
        from instrument LSF. Default 0.0 (no subtraction).

        - Scalar (float): constant library resolution across spectrum.
          Common values: MILES-based (FSPS default) 70 km/s, C3K 15 km/s, IRTF 20 km/s.
          Use ``SSP_LIBRARY_RESOLUTIONS[library_name]`` for pre-defined values.
        - Array (n_pix,): per-wavelength library resolution derived from the
          loaded SSP template's LSF. Example: MILES has FWHM ≈ 2.51 Å (constant
          in wavelength), giving σ_v(λ) ∝ 1/λ, from ~91 km/s at 3525 Å to ~43 km/s
          at 7500 Å. Retrieved from ``SSPData.ssp_resolution_kms`` after load.
    n_bins : int, optional
        Number of piecewise-constant segments for variable-R approximation.
        Ignored for scalar R. Higher values are more accurate but slower.
        Typical: 10–20. Default 16.
    sigma_v_kms : float, optional
        Intrinsic galaxy velocity dispersion :math:`\\sigma_v` [km/s] added
        in quadrature to :math:`\\sigma_{\\rm eff}`. This is the broadening
        from stellar dynamics, distinct from instrument LSF
        (``resolution``) and from the SSP-library template resolution
        (``sigma_lib_kms``). Default 0.0 (no extra broadening).
    pad_pixels : int or None, optional
        Static padding [pixels] per side (keyword-only). ``None`` derives it from the
        sigma of this call: the exact maximum over pixels when every input is
        concrete, and otherwise each traced input is assumed to be at most
        ``_LSF_MAX_SIGMA_KMS`` (2000 km/s) and combined in quadrature with the
        concrete ones. A traced grid pads by ``n - 1``. An explicit value is used as
        given, so it must cover 5 sigma of the kernel to be exact.

    Returns
    -------
    spectrum_smoothed : array, shape (n_pix,)
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

    **Boundary handling**: the convolution runs on a reflecting (symmetric) margin of
    :func:`_lsf_pad_pixels` pixels per side (5 sigma of the kernel at the pixel scale
    the kernel uses, and at least 256 pixels for sub-pixel kernels), so the kernel
    does not wrap from one spectrum edge onto the other (#2712, #2832).

    See Also
    --------
    velocity_broaden : Convolve with velocity dispersion only (no library subtraction).
    nirspec_prism_resolution : JWST NIRSpec PRISM variable-R function.
    nirspec_g140m_resolution : JWST NIRSpec G140M constant-R function.

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
    sigma_bound = _lsf_sigma_bound_kms(resolution, sigma_lib_kms, sigma_v_kms)
    sigma_v_kms = jnp.maximum(jnp.asarray(sigma_v_kms), 0.0)

    resolution = jnp.asarray(resolution)
    sigma_lib_kms = jnp.asarray(sigma_lib_kms)

    # Compute instrument sigma at each pixel
    sigma_inst_kms = _C_KM_S / (_FWHM_TO_SIGMA * resolution)

    # Subtract library resolution and add intrinsic stellar velocity
    # dispersion in quadrature:
    #   σ_total² = σ_inst² − σ_lib² + σ_v²
    # σ_lib accounts for the broadening already baked into the SSP
    # templates; σ_v is the intrinsic galaxy velocity dispersion.
    sigma_lib2 = sigma_lib_kms**2
    sigma_v2 = sigma_v_kms**2

    # Compute the effective broadening kernel, clamping to zero when
    # the library resolution exceeds the instrument resolution at some pixels.
    deficit = sigma_inst_kms**2 - sigma_lib2
    sigma_eff_kms = jnp.sqrt(jnp.maximum(deficit, 0.0) + sigma_v2)

    # The single-FFT path reads one pixel scale, ``log(wave[1]/wave[0])``, which
    # describes the whole array only on a grid uniform in ln(lambda). On any other
    # grid it under-broadens by ``wave[0]/lambda`` (#1791). Rather than refuse such
    # grids, #1742's remedy for ``velocity_broaden``, which would take
    # spectroscopy with it, since tengri's own forward model runs on linear
    # observed grids, send them through the piecewise path, which carries a
    # per-bin pixel scale. Nothing is resampled, so the FFT normalization still
    # conserves flux exactly and a zero-width kernel is still the identity.
    if resolution.ndim == 0 and sigma_lib_kms.ndim == 0 and _is_log_uniform(wave_obs):
        # Scalar R and scalar σ_lib on a log-uniform grid: one FFT, and the scale is exact.
        pad = _resolve_pad(pad_pixels, wave_obs, spectrum.shape[0], sigma_bound)
        return _apply_lsf_constant_r(spectrum, wave_obs, sigma_eff_kms, pad=pad)

    # Per-pixel R, or per-pixel σ_lib, or a grid whose pixel scale varies:
    # piecewise-constant in all three.
    sigma_per_pixel = jnp.broadcast_to(jnp.atleast_1d(sigma_eff_kms), spectrum.shape)
    pad = _resolve_pad(pad_pixels, wave_obs, spectrum.shape[0], sigma_bound, n_bins)
    return _apply_lsf_variable_r(spectrum, wave_obs, sigma_per_pixel, n_bins, pad=pad)


def project_spectrum(
    sed_rest: jnp.ndarray,
    wave_rest: jnp.ndarray,
    wave_obs: jnp.ndarray,
    redshift: float,
    dl_cm: float,
    *,
    resolution: jnp.ndarray | float | None = None,
    sigma_lib_kms: jnp.ndarray | float = 0.0,
    n_bins: int = 16,
    sigma_v_kms: float = 0.0,
    cal_coeffs: jnp.ndarray | None = None,
    cal_wave_range: tuple[float, float] | None = None,
    conserving: bool = False,
    resolution_matrix: object | None = None,
    has_lyc_edge: bool = False,
    lsf_pad_pixels: int | None = None,
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
    sed_rest : array, shape (n_wave,)
        Rest-frame spectral luminosity density [erg/s/Hz] on the
        rest-frame wavelength grid.
    wave_rest : array, shape (n_wave,)
        Rest-frame wavelength grid [Angstrom].
    wave_obs : array, shape (n_pix,)
        Observed-frame wavelength at each spectral pixel [Angstrom].
    redshift : float
        Source redshift z.
    dl_cm : float
        Luminosity distance [cm].
    resolution : float, array, or None
        Spectral resolution :math:`R(\lambda) = \lambda / \Delta\lambda`.
        If ``None``, LSF is skipped. If scalar, constant resolution; if array,
        per-pixel wavelength-dependent resolution (e.g., JWST NIRSpec PRISM).
    sigma_lib_kms : float or array, shape (n_pix,), optional
        SSP library velocity dispersion [km/s], subtracted in quadrature from
        instrument LSF. Default 0.0 (no subtraction).

        - Scalar (float): constant library resolution (e.g., MILES 70 km/s, C3K 15 km/s).
        - Array (n_pix,): per-wavelength library resolution from ``SSPData.ssp_resolution_kms``.
    n_bins : int, optional
        Number of piecewise-constant segments for variable-R LSF approximation.
        Ignored when resolution is scalar. Default 16.
    sigma_v_kms : float, optional
        Intrinsic galaxy velocity dispersion [km/s] added in quadrature to LSF.
        Default 0.0 (no extra broadening).
    cal_coeffs : array, shape (order,), or None, optional
        Chebyshev calibration polynomial coefficients ``[a_1, ..., a_N]``,
        where ``N`` is the calibration order. If ``None`` (default), no
        calibration is applied. Empty array ``[]`` gives unity calibration
        (no-op).
    cal_wave_range : tuple[float, float], optional
        ``(wave_min, wave_max)`` wavelength range for normalizing the
        Chebyshev polynomial to [-1, 1]. Used only when ``cal_coeffs`` is not
        ``None``. If omitted, defaults to ``(wave_obs.min(), wave_obs.max())``.
    conserving : bool, optional
        Resample the model onto the pixel grid with a flux-conserving bin
        integral (:func:`compute_spectrum_conserving`) instead of point
        interpolation. Default ``False`` (point sampling, unbiased only when
        the model grid is much finer than the pixels). Set for low-resolution
        spectroscopy where point sampling aliases; see #1166.
    resolution_matrix : BandedMatrix or None, optional
        Banded instrument resolution operator (DESI/PFS spectro-perfectionism;
        Bolton & Schlegel 2010). When supplied, ``sigma_v_kms`` is applied to the
        resampled model first (the matrix is the *instrument* LSF only and does
        not carry the galaxy's own kinematic broadening; #2506), then the result
        is projected through ``R @ model`` at pixel resolution. This **replaces**
        the Gaussian ``apply_lsf``, the matrix already encodes the true instrument
        LSF (the Redrock/FastSpecFit convention); ``sigma_lib_kms`` is **not**
        subtracted on this path (see #2506 follow-up). Default ``None`` (Gaussian
        LSF from ``resolution``). See :func:`~tengri.observation.banded.banded_matvec`.
        #1163.
    has_lyc_edge : bool, optional
        ``sed_rest`` carries a Lyman-continuum mask (a photoionized nebular
        backend published ``lyc_transmission``), so it is a step at the Lyman
        edge, not a ramp across the model cell straddling it. The resampler
        then reads the cell with the step model of :mod:`tengri.components.lyc`,
        as photometry does (#2447). Static. Default ``False``.

    lsf_pad_pixels : int or None, optional
        Static padding [pixels] per side for the LSF (see :func:`apply_lsf`). ``None``
        derives it from the grid and the sigma. Used on the observed grid; the
        conserving path's LSF on the model grid is not affected.

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
    compute_spectrum : Compute observed spectrum (no LSF).
    apply_lsf : Apply LSF convolution separately.
    velocity_broaden : Broaden by velocity dispersion only.
    apply_calibration : Apply calibration polynomial to a spectrum.

    """
    from tengri.observation.calibration import apply_calibration

    # ``conserving`` is a static structural flag (resolved from the ``resample``
    # mode before the trace), so this branch resolves at trace time (#1166).
    if conserving and resolution is not None and resolution_matrix is None:
        # The detector integrates the light that has already passed through the
        # line-spread function: broaden on the model grid, then take the pixel
        # integral (#2530).
        flux = compute_spectrum_conserving_lsf(
            sed_rest,
            wave_rest,
            wave_obs,
            redshift,
            dl_cm,
            resolution=resolution,
            sigma_lib_kms=sigma_lib_kms,
            n_bins=n_bins,
            sigma_v_kms=sigma_v_kms,
        )
        resolution = None
    else:
        resampler = compute_spectrum_conserving if conserving else compute_spectrum
        flux = resampler(sed_rest, wave_rest, wave_obs, redshift, dl_cm, has_lyc_edge=has_lyc_edge)
    if resolution_matrix is not None:
        # The banded matrix (DESI/PFS spectro-perfectionism; Bolton & Schlegel 2010)
        # is the *instrument* LSF and replaces the Gaussian apply_lsf (#1163). The
        # galaxy's own kinematic broadening is not in it, so sigma_v is applied to
        # the resampled model first (#2506: it was silently dropped here, leaving
        # d spectrum / d sigma_v = 0). The piecewise path is used because DESI
        # pixels are linear in lambda (#1742/#1791); at sigma_v = 0 the kernel is
        # the identity, so existing fits are unchanged.
        from tengri.observation.banded import banded_matvec

        flux = broaden_velocity_only(
            flux, wave_obs, sigma_v_kms, n_bins, pad_pixels=lsf_pad_pixels
        )
        flux = banded_matvec(resolution_matrix.offsets, resolution_matrix.data, flux)
    elif resolution is not None:
        flux = apply_lsf(
            flux,
            wave_obs,
            resolution,
            sigma_lib_kms=sigma_lib_kms,
            n_bins=n_bins,
            sigma_v_kms=sigma_v_kms,
            pad_pixels=lsf_pad_pixels,
        )
    if cal_coeffs is not None:
        wmin, wmax = (
            cal_wave_range if cal_wave_range is not None else (wave_obs.min(), wave_obs.max())
        )
        flux = apply_calibration(flux, wave_obs, cal_coeffs, wmin, wmax)
    return flux


@partial(jax.jit, static_argnames=("has_lyc_edge",))
def compute_spectrum(
    sed_rest: jnp.ndarray,
    wave_rest: jnp.ndarray,
    wave_obs: jnp.ndarray,
    redshift: float,
    dl_cm: float,
    has_lyc_edge: bool = False,
) -> jnp.ndarray:
    """Compute observed spectrum at arbitrary pixel wavelengths.

    Maps observed wavelengths back to rest-frame coordinates (accounting for
    redshift), evaluates the rest-frame SED at those wavelengths via interpolation,
    and scales to observed flux using luminosity distance and (1+z) redshift factor.

    Parameters
    ----------
    sed_rest : array, shape (n_wave,)
        Rest-frame spectral luminosity density [erg/s/Hz] on the
        rest-frame wavelength grid.
    wave_rest : array, shape (n_wave,)
        Rest-frame wavelength grid [Angstrom].
    wave_obs : array, shape (n_pix,)
        Observed-frame wavelength at each spectral pixel [Angstrom].
    redshift : float
        Source redshift z.
    dl_cm : float
        Luminosity distance [cm].
    has_lyc_edge : bool, optional
        Interpolate the model cell straddling the Lyman edge with the step
        model (:func:`tengri.components.lyc.edge_interp`) instead of a linear
        ramp. Static. Default ``False``.

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
    if has_lyc_edge:
        sed_at_pixels = edge_interp(wave_rest_query, wave_rest, sed_rest)
    else:
        sed_at_pixels = jnp.interp(wave_rest_query, wave_rest, sed_rest, left=0.0, right=0.0)

    # Apply the (1+z)/(4π d_L²) dimming to the pixel SED directly. A standalone
    # ``flux_scale = lnu_to_fnu(1.0, ...)`` is ~1e-58 and underflows float32 to
    # zero (peak 1.0 absorbs none of the -58 decades); applied to sed_at_pixels
    # (~1e30) apply_log10_scale folds the offset into the array peak and the
    # result stays in range. Identical in float64 (#1206).
    return lnu_to_fnu(sed_at_pixels, dl_cm, redshift)


def _flux_conserving_resample(
    wave_rest: jnp.ndarray,
    sed_rest: jnp.ndarray,
    wave_query: jnp.ndarray,
    edge_aa: float | None = None,
) -> jnp.ndarray:
    r"""Bin-integrated (flux-conserving) resample of ``sed_rest`` onto ``wave_query``.

    Each output value is the *mean flux density over that pixel's wavelength bin*
    (the integral of the model over the bin divided by the bin width) rather
    than a point sample at the pixel center (Carnall 2017, SpectRes, eq. 3):

    .. math::

        \tilde{f}_j = \frac{1}{\Delta\lambda_j}
                      \int_{\lambda_j^-}^{\lambda_j^+} f(\lambda)\, d\lambda

    where the bin edges :math:`\lambda_j^\pm` are the midpoints between adjacent
    ``wave_query`` centers. Point sampling is unbiased only when the model grid is
    much finer than the pixel spacing; when a pixel spans one or more model bins
    (low-resolution spectroscopy, e.g. NIRSpec PRISM) it aliases the sub-pixel
    structure, biasing the integrated continuum. The bin integral does not.

    Implemented as a difference of the exact cumulative integral of the
    piecewise-linear model evaluated at the bin edges, so it is O(n_wave + n_pix),
    JIT-compatible, and differentiable w.r.t. ``sed_rest`` and, through the
    edges, the redshift.
    Outside the model grid the cumulative integral is flat, so out-of-range bins
    contribute zero, matching ``compute_spectrum``'s ``left=0, right=0`` clamp.

    Parameters
    ----------
    wave_rest : array, shape (n_wave,)
        Rest-frame model wavelength grid [Angstrom], strictly increasing.
    sed_rest : array, shape (n_wave,)
        Rest-frame flux density on ``wave_rest`` [erg/s/Hz].
    wave_query : array, shape (n_pix,)
        Rest-frame pixel-center wavelengths to resample onto [Angstrom].
    edge_aa : float or None, optional
        Rest-frame wavelength of a step in the model [Angstrom] (the Lyman
        edge of an SED carrying a Lyman-continuum mask). The model cell
        straddling it is integrated as the step of
        :func:`tengri.components.lyc.edge_interp`, not a linear ramp.
        ``None`` (default): piecewise linear everywhere.

    Returns
    -------
    ndarray, shape (n_pix,)
        Bin-averaged flux density at each pixel [erg/s/Hz].

    Notes
    -----
    **JIT-compatible**: yes. **Gradient-safe**: yes (linear in ``sed_rest``).

    With ``edge_aa`` inside the cell :math:`[\lambda_a, \lambda_b]` of width
    :math:`h`, with node values :math:`y_a, y_b` and :math:`d = \lambda_e -
    \lambda_a`, the cumulative integral at :math:`\lambda_a + t` gains the
    difference between the step and the ramp,

    .. math::

        \Delta(t) = y_a \min(t, d) + y_b \max(t - d, 0)
                     - y_a t - \frac{(y_b - y_a)\, t^2}{2h},
        \qquad 0 \le t \le h,

    held at :math:`\Delta(h)` above the cell and zero below it, so the pixel
    mean is exact for the step model (#2447).

    References
    ----------
    .. [1] Carnall, A. C. 2017, "SpectRes: A Fast Spectral Resampling Tool in
           Python", arXiv:1705.05165.
    """
    mid = 0.5 * (wave_query[1:] + wave_query[:-1])
    lo = wave_query[:1] - 0.5 * (wave_query[1:2] - wave_query[:1])
    hi = wave_query[-1:] + 0.5 * (wave_query[-1:] - wave_query[-2:-1])
    edges = jnp.concatenate([lo, mid, hi])  # (n_pix + 1,)

    # Exact cumulative integral of the piecewise-linear model, clamped flat
    # outside the grid. Inside interval i the integral from its left node is
    # f_i t + (f_{i+1} - f_i) t^2 / (2 h_i), so its slope is the interpolated
    # model value and the pixel flux is continuous in the edge position with a
    # continuous first derivative (a linear interpolation of the node integrals
    # would leave a slope that jumps each time an edge crosses a node, a step
    # in the redshift gradient; #2530).
    h = jnp.diff(wave_rest)
    node_cum = jnp.concatenate(
        [jnp.zeros(1), jnp.cumsum(0.5 * (sed_rest[1:] + sed_rest[:-1]) * h)]
    )
    e = jnp.clip(edges, wave_rest[0], wave_rest[-1])
    i = jnp.clip(jnp.searchsorted(wave_rest, e, side="right") - 1, 0, wave_rest.shape[0] - 2)
    t = e - wave_rest[i]
    f0 = sed_rest[i]
    flux_at_edges = node_cum[i] + f0 * t + 0.5 * (sed_rest[i + 1] - f0) / h[i] * t * t
    if edge_aa is not None:
        flux_at_edges = flux_at_edges + _step_minus_ramp(wave_rest, sed_rest, e, edge_aa)
    return (flux_at_edges[1:] - flux_at_edges[:-1]) / (edges[1:] - edges[:-1])


def _step_minus_ramp(wave_rest, sed_rest, x, edge_aa):
    """Cumulative-integral difference, step model minus linear ramp, at ``x``."""
    n_ion = jnp.sum(ionizing_mask(wave_rest, edge_aa).astype(jnp.int32))
    has_bracket = (n_ion > 0) & (n_ion < wave_rest.shape[0])
    ia = jnp.clip(n_ion - 1, 0, wave_rest.shape[0] - 2)
    lam_a, h = wave_rest[ia], wave_rest[ia + 1] - wave_rest[ia]
    y_a, y_b = sed_rest[ia], sed_rest[ia + 1]
    d = edge_aa - lam_a
    # Position of ``x`` within the cell: 0 below it, h above it.
    t = jnp.where(x < lam_a, 0.0, jnp.where(x > lam_a + h, h, x - lam_a))
    # y_a min(t, d) + y_b (t - d)_+ - [y_a t + (y_b - y_a) t^2 / 2h]: only the jump
    # y_b - y_a survives, times the step's ramp minus the linear ramp's.
    past_edge = jnp.where(t > d, t - d, 0.0)
    delta = (y_b - y_a) * (past_edge - 0.5 * t * t / h)
    return jnp.where(has_bracket, delta, 0.0)


@partial(jax.jit, static_argnames=("has_lyc_edge",))
def compute_spectrum_conserving(
    sed_rest: jnp.ndarray,
    wave_rest: jnp.ndarray,
    wave_obs: jnp.ndarray,
    redshift: float,
    dl_cm: float,
    has_lyc_edge: bool = False,
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
    edge_aa = LYMAN_LIMIT_AA if has_lyc_edge else None
    sed_at_pixels = _flux_conserving_resample(wave_rest, sed_rest, wave_rest_query, edge_aa)
    # Dimming applied to the pixel SED directly, not as a standalone flux_scale
    # (~1e-58, which underflows float32 to zero). See _resample_to_spectrum
    # above and #1206.
    return lnu_to_fnu(sed_at_pixels, dl_cm, redshift)


def compute_spectrum_conserving_lsf(
    sed_rest: jnp.ndarray,
    wave_rest: jnp.ndarray,
    wave_obs: jnp.ndarray,
    redshift: float,
    dl_cm: float,
    *,
    resolution: jnp.ndarray | float,
    sigma_lib_kms: jnp.ndarray | float = 0.0,
    n_bins: int = 16,
    sigma_v_kms: jnp.ndarray | float = 0.0,
) -> jnp.ndarray:
    r"""Line-spread function on the model grid, then the pixel integral (#2530).

    A detector pixel records the mean over its edges of the light that has
    passed through the instrument's line-spread function, so the Gaussian
    kernel :math:`\sqrt{\sigma_{\rm inst}^2 - \sigma_{\rm lib}^2 + \sigma_v^2}`
    (see :func:`apply_lsf`) acts on the model before the bin integral of
    :func:`compute_spectrum_conserving`. The kernel is a Gaussian in
    :math:`\ln\lambda`, so it is applied on the rest-frame model grid with
    the resolution and library width read at the observed wavelength
    :math:`(1+z)\lambda`.

    Parameters
    ----------
    sed_rest, wave_rest, wave_obs, redshift, dl_cm
        As in :func:`compute_spectrum`.
    resolution : float or array, shape (n_pix,)
        Resolving power :math:`R(\lambda)` at the pixels; interpolated onto the
        model grid when an array.
    sigma_lib_kms : float or array, shape (n_pix,), optional
        Library velocity dispersion [km/s] (scalar, or per pixel).
    n_bins : int, optional
        Piecewise-constant segments of the variable-width kernel.
    sigma_v_kms : float, optional
        Intrinsic velocity dispersion [km/s], added in quadrature.

    Returns
    -------
    ndarray, shape (n_pix,)
        Observed flux density [erg/s/cm^2/Hz] per pixel.

    Notes
    -----
    **JIT-compatible**: yes. **Gradient-safe**: yes.
    """
    wave_model_obs = wave_rest * (1.0 + redshift)

    def _onto_model(x):
        x = jnp.asarray(x)
        return x if x.ndim == 0 else jnp.interp(wave_model_obs, wave_obs, x)

    # An array resolution takes the piecewise path (rest grids are not log-uniform).
    sed_broad = apply_lsf(
        sed_rest,
        wave_rest,
        _onto_model(resolution),
        sigma_lib_kms=_onto_model(sigma_lib_kms),
        n_bins=n_bins,
        sigma_v_kms=sigma_v_kms,
    )
    return compute_spectrum_conserving(sed_broad, wave_rest, wave_obs, redshift, dl_cm)


#: Fractional spread in ``d(ln lambda)`` tolerated before a grid is called
#: non-log-uniform. A genuine ``logspace`` grid lands ~1e-14 here; a linear grid
#: over 4500-5500 A lands ~0.2, so there is four orders of magnitude of daylight
#: between the two and the threshold is not a tuning knob.
_LOG_UNIFORM_RTOL = 1e-6


def _require_log_uniform_grid(wave, caller: str) -> None:
    """Raise unless ``wave`` is uniform in ``ln(lambda)`` (#1742).

    A no-op when ``wave`` is a tracer: a traced grid has no values to inspect at
    trace time. That is a real gap rather than a safe default; it is narrow
    because a spectroscopic wavelength grid is normally a fixed instrument array
    closed over by the jitted function, not an argument traced through it.

    Parameters
    ----------
    wave : array_like, shape (n_pix,)
        Wavelength grid to check [Angstrom].
    caller : str
        Function name, quoted in the error so the message names the API the user
        actually called.
    """
    if isinstance(wave, jax.core.Tracer):
        return
    w = np.asarray(wave, dtype=np.float64)
    if w.size < 3 or first_invalid_wavelength(w) is not None:
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
    flux : array, shape (n_pix,)
        Input spectral flux.
    wave : array, shape (n_pix,)
        Wavelength grid [Angstrom]. Must be uniformly spaced **in
        log-wavelength**, e.g. ``jnp.logspace(...)``, not ``jnp.linspace(...)``.
        A linearly-spaced grid is rejected; see Notes.
    sigma_km_s : float
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
    grid (the usual case for a fixed instrument grid) it still fires.

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
    sigma_bound = _lsf_sigma_bound_kms(sigma_v_kms=sigma_km_s)
    pad = _lsf_pad_pixels(wave, flux.shape[0], sigma_bound)
    return _velocity_broaden_impl(flux, wave, sigma_km_s, pad=pad)


@partial(jax.jit, static_argnames=("pad",))
def _velocity_broaden_impl(
    flux: jnp.ndarray,
    wave: jnp.ndarray,
    sigma_km_s: float,
    *,
    pad: int,
) -> jnp.ndarray:
    """FFT convolution for :func:`velocity_broaden`, with the check already done.

    Split out so the grid check runs on concrete values: the public function was
    itself ``@jax.jit``, which makes ``wave`` a tracer inside it, and a guard that
    can never see its argument is not a guard (#1742). ``pad`` is the static kernel
    margin from :func:`_lsf_pad_pixels` (#2832).
    """
    sigma_v = sigma_km_s / _C_KM_S  # fractional velocity dispersion

    # Pixel scale in log-wavelength. Constant across the array precisely because
    # the grid is uniform in ln(lambda), checked by the caller, not assumed.
    dlnwave = jnp.log(wave[1] / wave[0])

    # Gaussian kernel width in pixels
    sigma_pix = sigma_v / dlnwave

    return _gaussian_fft_convolve(flux, sigma_pix, pad)


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
    line_wavelengths : array, shape (n_lines,)
        Rest-frame line wavelengths [Angstrom].
    line_luminosities : array, shape (n_lines,)
        Line luminosities [L_sun]. Total integrated luminosity per line.
    spectral_resolution : float
        Instrument spectral resolution R = lambda / delta_lambda (dimensionless).
        Typical values: R ~ 100 (photometry), R ~ 1000 (low-res spectroscopy),
        R ~ 5000 (medium-res).
    wave_out : array, shape (n_pix,)
        Output wavelength grid [Angstrom] in observed frame.
    redshift : float, optional
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
        lam_rest : scalar
            Rest-frame wavelength (Angstrom).
        lum : scalar
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
