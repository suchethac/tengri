# SPDX-License-Identifier: BSD-3-Clause
"""Broad Line Region (BLR) emission model.

The BLR is dense gas close to the black hole producing broad permitted
emission lines with FWHM ~ 1000-10000 km/s. The BLR is geometrically
compact and lies within the torus, so it is obscured at high
inclinations (Type 2 AGN).

This module provides an analytic BLR template using broad Gaussian
profiles calibrated to the Vanden Berk et al. (2001) SDSS composite
quasar spectrum. The line list includes ≥25 permitted broad lines
spanning the UV (Lyα, C IV, He II, C III], Mg II) through optical
(Balmer series, Paschen series). When enabled, an Fe II pseudo-continuum
is added, modeled as a sum of broad Gaussians at key multiplet wavelength
groups (Vestergaard & Wilkes 2001, Tsuzuki+2006, Kovacevic+2010).

All functions are pure JAX and JIT-compilable.

References
----------

- Vanden Berk et al. 2001, AJ, 122, 549 (SDSS composite quasar spectrum)
  https://doi.org/10.1086/321167
- Netzer 1990, in Accretion Power in Astrophysics (Broad-line region models)
- Boroson & Green 1992, ApJS, 80, 109 (Fe II / H-beta ratio)
  https://doi.org/10.1086/191679
- Vestergaard & Wilkes 2001, ApJS, 134, 1 (UV Fe II templates)
  https://doi.org/10.1086/320360
- Tsuzuki et al. 2006, ApJ, 650, 57 (UV Fe II decomposition)
  https://doi.org/10.1086/506270
- Kovacevic et al. 2010, ApJS, 189, 15 (optical Fe II model)
  https://doi.org/10.1088/0067-0049/189/1/15

"""

from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from tengri.components.agn._phys import gaussian_line_profile as _gaussian_line_profile
from tengri.utils.host_array import device_table, host_array

# ── Physical constants ────────────────────────────────────────────
from tengri.utils.physics_constants import (
    C_AA as _C_AA,
    C_KM_S as _C_LIGHT_KMS,
)
from tengri.utils.scale import representable_denominator

# ── BLR emission-line template ────────────────────────────────────

# Key broad emission lines: (rest wavelength [Angstrom], relative strength)
# Line strengths extracted from Vanden Berk et al. (2001) Table 2
# ("Composite Quasar Emission Line Features"), derived from SDSS composite.
# Relative strengths are normalized to H-beta = 1.0 by dividing the VB01
# "Rel. Flux" column (F/F_Lyα) by the H-beta flux value (8.649).
# Vacuum wavelengths per SDSS convention; comments cite VB01 flux values.
_BLR_LINES = host_array(
    [
        # Lyman series
        [1025.72, 1.1112],  # Lyβ (1033.03 obs, VB01 rel flux 9.615)
        [1215.67, 11.5660],  # Lyα (1216.25 obs, VB01 rel flux 100.0, reference)
        # UV forbidden/resonance lines
        [1240.14, 0.2847],  # N V (1239.85 obs, VB01 rel flux 2.461)
        [1306.82, 0.2303],  # Si II (1305.42 obs, VB01 rel flux 1.992)
        [1335.30, 0.0796],  # C II (1336.60 obs, VB01 rel flux 0.688)
        [1396.76, 0.5156],  # Si IV (1398.33 VB01 blend, half-strength, VB01 Table 2)
        [1402.06, 0.5156],  # O IV] (1398.33 VB01 blend, half-strength, VB01 Table 2)
        [1549.06, 2.9237],  # C IV (1546.15 obs, VB01 rel flux 25.291, major UV)
        [1640.42, 0.0602],  # He II (1637.84 obs, VB01 rel flux 0.521)
        [1663.48, 0.0555],  # O III] (1664.74 obs, VB01 rel flux 0.480)
        [1857.40, 0.0385],  # Al III (1856.76 obs, VB01 rel flux 0.333)
        [1892.03, 0.0183],  # Si III] (1892.64 obs, VB01 rel flux 0.158)
        [1908.73, 1.8436],  # C III] (1905.97 obs, VB01 rel flux 15.943, major UV)
        [2326.44, 0.0212],  # C II] (2327.34 obs, VB01 rel flux 0.183)
        [2423.83, 0.0505],  # [Ne IV] (2423.46 obs, VB01 rel flux 0.437)
        # MgII and UV FeII blends
        [2798.75, 1.7033],  # Mg II (2800.26 obs, VB01 rel flux 14.725, major opt)
        # Balmer series
        [3970.20, 0.0546],  # H-epsilon (3968.43 obs, blended with [Ne III])
        [4102.89, 0.1233],  # H-delta (4102.73 obs, VB01 rel flux 1.066)
        [4341.68, 0.3025],  # H-gamma (4346.42 obs, VB01 rel flux 2.616)
        [4862.68, 1.0000],  # H-beta (4853.13 obs, VB01 rel flux 8.649, reference)
        [6564.61, 3.5666],  # H-alpha (6564.93 obs, VB01 rel flux 30.832, strongest opt)
        # Paschen series (IR Balmer)
        [9015.0, 0.1500],  # Pa-beta (approx from Balmer scaling)
        [10050.0, 0.0600],  # Pa-gamma (approx from Balmer scaling)
    ]
)
# Total of 23 lines (Si IV and O IV] split from VB01 Table 2 blend at 1398.33 Å)

_BLR_LINE_WAVELENGTHS = host_array(_BLR_LINES[:, 0])
_BLR_LINE_STRENGTHS = host_array(_BLR_LINES[:, 1])

# Default BLR line FWHM [km/s]
_BLR_FWHM_KMS = 5000.0

# Default fraction of intercepted luminosity re-emitted as broad lines.
# This is promoted to a free parameter `agn_blr_line_efficiency`
# with Uniform(0.05, 0.15) prior in _params.py.
_BLR_LINE_EFFICIENCY_DEFAULT = 0.08

# Default covering fraction of the BLR.
# This is promoted to a free parameter `agn_blr_cf` in _params.py.
_BLR_COVERING_FRACTION_DEFAULT = 0.1


# ── Fe II template loading ────────────────────────────────────────


def _load_fe2_templates():
    """Load UV and optical Fe II templates from data files.

    Templates are sourced from PyQSOFit (Temple, Hewett & Banerji 2021),
    which curates UV Fe II from Vestergaard & Wilkes 2001 and Tsuzuki+2006,
    and optical Fe II from Boroson & Green 1992.

    File format: log10(wavelength), **F_lambda** [erg/s/cm²/Å] (per PyQSOFit
    convention). Only the *shape* of each template is used downstream; the
    absolute scale is set by the R_Fe / L(H-beta) normalization in
    :func:`_fe2_pseudo_continuum`.

    Negative template nodes (numerical ringing in the published tabulations,
    240 in the optical file) are clipped to zero here, **before** any
    resampling or broadening, as PyQSOFit does: a flux density of an emission
    template cannot be negative, and leaving them in lets them cancel real
    flux under the broadening kernel.

    Returns
    -------
    tuple of (uv_wave, uv_flux, opt_wave, opt_flux)
        All as np.ndarray, dtype float64. Wavelengths in Angstrom (linear scale).
        Flux is F_lambda [erg/s/cm²/Å], clipped to >= 0.
    """
    data_dir = Path(__file__).parent.parent.parent / "data" / "agn_fe2"

    uv_file = data_dir / "fe_uv_pyqsofit.txt"
    opt_file = data_dir / "fe_optical_pyqsofit.txt"

    if not uv_file.exists() or not opt_file.exists():
        raise FileNotFoundError(
            f"Fe II templates not found at {data_dir}. Expected: "
            f"fe_uv_pyqsofit.txt, fe_optical_pyqsofit.txt"
        )

    # Load UV template (log10 wavelength, flux)
    uv_data = np.genfromtxt(str(uv_file), comments="#", dtype=np.float64)
    uv_log_wave = uv_data[:, 0]
    uv_flux = np.maximum(uv_data[:, 1], 0.0)
    uv_wave = 10.0**uv_log_wave  # Convert to linear wavelength

    # Load optical template (log10 wavelength, flux)
    opt_data = np.genfromtxt(str(opt_file), comments="#", dtype=np.float64)
    opt_log_wave = opt_data[:, 0]
    opt_flux = np.maximum(opt_data[:, 1], 0.0)
    opt_wave = 10.0**opt_log_wave  # Convert to linear wavelength

    return uv_wave, uv_flux, opt_wave, opt_flux


# Load Fe II templates at module import time (avoid I/O in JIT-compiled functions)
try:
    _FE2_UV_WAVE, _FE2_UV_FLUX, _FE2_OPT_WAVE, _FE2_OPT_FLUX = (
        host_array(x) for x in _load_fe2_templates()
    )
except FileNotFoundError:
    # Fallback: set to None and raise at runtime if Fe II is requested
    _FE2_UV_WAVE = None
    _FE2_UV_FLUX = None
    _FE2_OPT_WAVE = None
    _FE2_OPT_FLUX = None


# R_Fe measurement window [Angstrom] (Boroson & Green 1992).
_FE2_RFE_WINDOW = (4434.0, 4684.0)


# Internal grid on which the FeII template is carried, broadened and normalized.
# Uniform in ln(lambda) (constant velocity step), so a Gaussian of fixed velocity
# width is one shift-invariant kernel. Step choice, measured against a 0.5 km/s
# lattice (max |difference| / peak of the broadened spectrum, FWHM 1000 / 5000 km/s):
#   2 km/s 2.3e-4 / 6.4e-5;  5 km/s 3.8e-4 / 1.0e-4;  10 km/s 3.9e-4 / 1.1e-4;
#   20 km/s 1.1e-3 / 3.2e-4.
# The native nodes are 104-106 km/s apart, so the error plateaus at the re-gridding
# of the piecewise-linear template (~3e-4 of the peak) from 5 km/s down; 5 km/s is
# the coarsest step on that plateau (20 km/s costs 3x more) and resolves a Gaussian
# down to FWHM ~ 25 km/s (2 samples per sigma), far below any BLR width (>= ~1000).
_FE2_GRID_STEP_KMS = 5.0
# [Angstrom]; covers the 1075-7484 A template plus the broadening tails
_FE2_GRID_LAMBDA_RANGE = (900.0, 9000.0)
# Static FFT length: next power of two >= the grid length. The template is zero over
# >5e4 km/s of padding at each end, so the circular wrap-around never reaches support.
_FE2_FFT_SIZE = 1 << 18

# Band-limited evaluation (the default path of every FeII entry point).
#
# The Gaussian transfer function exp(-2 pi^2 sigma_s^2 f^2) is below ``_FE2_TRUNCATION`` for
# every f above 1.08 / sigma_s [cycles/sample], so for a width >= ``_FE2_MIN_FWHM_KMS`` the
# broadened template has no content beyond K = ceil(N * 1.08 / sigma_s_min) of the N/2 + 1
# Fourier bins (K = 6.7k of 131k at 500 km/s). The template does not depend on the width, so
# its spectrum is a constant, built once on the host from the same lattice and truncated to
# those K bins. Each trace multiplies it by the transfer function and synthesizes the
# broadened function on a lattice ``_FE2_DECIMATION`` times coarser (one 2^15-point inverse
# FFT instead of a 2^18-point forward and inverse pair on the full lattice). A caller
# wavelength is then read by the same linear interpolation between full-lattice nodes the
# full-lattice path applies, each node value coming from the coarse lattice by
# ``_FE2_STENCIL``-point Lagrange interpolation. The R_Fe window flux and the total power are
# linear in the broadened lattice, so they are K-term sums against the transfer function of
# constants (the adjoint of their lattice weights applied to the template spectrum): no
# lattice array is formed for them.
# [km/s] Narrowest FWHM the band-limited path represents; narrower concrete widths take the
# full-lattice path (the unbroadened template, FWHM -> 0, is one of them).
_FE2_MIN_FWHM_KMS = 500.0
# Transfer-function level below which Fourier bins are dropped.
_FE2_TRUNCATION = 1.0e-10
# Coarse-lattice step [full-lattice samples] and Lagrange stencil width [coarse nodes].
_FE2_DECIMATION = 8
_FE2_STENCIL = 12


def _fe2_internal_grid():
    """FeII template on the internal ln(lambda) grid, built inside the trace.

    The grid is an iota expression and the template is resampled from the native
    nodes (a few thousand values, the only constants), so no ~n_grid-sized array
    is baked into a traced graph; the lattice is regenerated in the compiled
    program.

    Returns
    -------
    wave : array, shape (n_grid,)
        Grid wavelengths [Angstrom], uniform in ln(lambda).
    template : array, shape (n_grid,)
        Zero-clipped template F_lambda shape, UV below 3500 A and optical above,
        linearly resampled in wavelength from the native nodes, zero outside the
        tabulated range, divided by its maximum (scale-free; the absolute scale is
        set by the R_Fe normalization) [dimensionless].
    window_weights : array, shape (n_grid,)
        Weights ``w_i`` with ``sum_i w_i y_i`` = integral of the piecewise-linear
        ``y(lambda)`` over the R_Fe window (4434-4684 A, exact edges) [Angstrom].
    dln : float
        Grid step in ln(lambda).
    """
    dln = _FE2_GRID_STEP_KMS / _C_LIGHT_KMS
    lo, hi = np.log(_FE2_GRID_LAMBDA_RANGE[0]), np.log(_FE2_GRID_LAMBDA_RANGE[1])
    n_grid = int(np.ceil((hi - lo) / dln)) + 1
    # The lattice is generated in the trace in the default float dtype: float64 when x64 is
    # enabled, float32 otherwise (an explicit float64 here warns on every trace without x64).
    wave = jnp.exp(lo + dln * jnp.arange(n_grid))

    uv = jnp.interp(
        wave, device_table(_FE2_UV_WAVE), device_table(_FE2_UV_FLUX), left=0.0, right=0.0
    )
    op = jnp.interp(
        wave, device_table(_FE2_OPT_WAVE), device_table(_FE2_OPT_FLUX), left=0.0, right=0.0
    )
    template = jnp.where(wave < 3500.0, uv, op)
    template = template / jnp.max(template)

    w0, w1 = wave[:-1], wave[1:]
    a = jnp.clip(w0, *_FE2_RFE_WINDOW)
    b = jnp.clip(w1, *_FE2_RFE_WINDOW)
    half = 0.5 * (b - a)
    ta, tb = (a - w0) / (w1 - w0), (b - w0) / (w1 - w0)
    weights = jnp.zeros(n_grid)
    weights = weights.at[:-1].add(half * ((1.0 - ta) + (1.0 - tb)))
    weights = weights.at[1:].add(half * (ta + tb))
    return wave, template, weights, float(dln)


def _fe2_broadened_on_grid(fwhm_kms):
    """Template of :func:`_fe2_internal_grid` broadened by a Gaussian of FWHM ``fwhm_kms``.

    The Gaussian has constant velocity width, i.e. constant width in ln(lambda),
    so the broadening is one circular convolution on the uniform internal grid,
    done in Fourier space with the analytic transfer function
    ``exp(-2 pi^2 sigma_s^2 f^2)`` (``sigma_s`` in grid samples, ``f`` in
    cycles/sample): normalized (unit DC gain), no kernel truncation, so no
    dependence on a kernel length when ``fwhm_kms`` is traced; the FFT size is a
    static constant (zero padding keeps the wrap-around outside the support).
    Smooth and differentiable in ``fwhm_kms``.
    """
    _, template, _, dln = _fe2_internal_grid()
    n_fft = _FE2_FFT_SIZE
    sigma_samples = (fwhm_kms / 2.3548 / _C_LIGHT_KMS) / dln
    freq = jnp.fft.rfftfreq(n_fft)
    transfer = jnp.exp(-2.0 * (jnp.pi * sigma_samples * freq) ** 2)
    spec = jnp.fft.rfft(template, n=n_fft)
    out = jnp.fft.irfft(spec * transfer, n=n_fft)[: template.shape[0]]
    # FFT round-off can leave ~1e-16 negatives on a non-negative function
    return jnp.where(out > 0.0, out, 0.0)


def _fe2_lattice_host():
    """Host (float64) copy of :func:`_fe2_internal_grid`, for the band-limited path's constants.

    Returns
    -------
    wave, template, window_weights, trapezoid_weights : ndarray, shape (n_grid,)
        As :func:`_fe2_internal_grid`, plus the trapezoid-rule weights in wavelength
        [Angstrom] (``sum_i t_i y_i`` is ``trapezoid(y, wave)``).
    """
    dln = _FE2_GRID_STEP_KMS / _C_LIGHT_KMS
    lo, hi = np.log(_FE2_GRID_LAMBDA_RANGE[0]), np.log(_FE2_GRID_LAMBDA_RANGE[1])
    n_grid = int(np.ceil((hi - lo) / dln)) + 1
    wave = np.exp(lo + dln * np.arange(n_grid))
    uv = np.interp(wave, _FE2_UV_WAVE, _FE2_UV_FLUX, left=0.0, right=0.0)
    op = np.interp(wave, _FE2_OPT_WAVE, _FE2_OPT_FLUX, left=0.0, right=0.0)
    template = np.where(wave < 3500.0, uv, op)
    template = template / template.max()

    w0, w1 = wave[:-1], wave[1:]
    a = np.clip(w0, *_FE2_RFE_WINDOW)
    b = np.clip(w1, *_FE2_RFE_WINDOW)
    half = 0.5 * (b - a)
    ta, tb = (a - w0) / (w1 - w0), (b - w0) / (w1 - w0)
    window = np.zeros(n_grid)
    window[:-1] += half * ((1.0 - ta) + (1.0 - tb))
    window[1:] += half * (ta + tb)

    step = np.diff(wave)
    trapezoid = np.zeros(n_grid)
    trapezoid[:-1] += 0.5 * step
    trapezoid[1:] += 0.5 * step
    return wave, template, window, trapezoid


def _lagrange_table(decimation, stencil):
    """Lagrange weights, shape (decimation, stencil), for the fractions ``r / decimation``.

    Row ``r`` interpolates at ``j + r / decimation`` from the nodes
    ``j - stencil/2 + 1 ... j + stencil/2``.
    """
    nodes = np.arange(-(stencil // 2) + 1, stencil // 2 + 1, dtype=np.float64)
    table = np.ones((decimation, stencil))
    for r in range(decimation):
        s = r / decimation
        for c in range(stencil):
            for c2 in range(stencil):
                if c2 != c:
                    table[r, c] *= (s - nodes[c2]) / (nodes[c] - nodes[c2])
    return table


@lru_cache(maxsize=1)
def _fe2_spectral_tables():
    """Constants of the band-limited FeII path, built once on the host (float64).

    The template of :func:`_fe2_internal_grid` is zero-padded to ``_FE2_FFT_SIZE`` and Fourier
    transformed once; only the bins the narrowest supported Gaussian leaves above
    ``_FE2_TRUNCATION`` are kept. With ``H_k`` the transfer function, the broadened lattice
    is ``irfft(S H)``, and any linear functional ``sum_i c_i y_i`` of it is ``sum_k r_k H_k``
    with ``r_k = m_k Re(conj(C_k) S_k) / N`` (Parseval; ``m_k`` = 2 for the doubled
    half-spectrum bins, 1 for DC).

    Returns
    -------
    dict of ndarray
        ``spectrum`` complex (K,): ``S_k / decimation`` (coarse-lattice synthesis scale);
        ``window`` and ``total`` real (K,): the R_Fe-window flux and the trapezoid integral as
        functionals ``r_k`` [Angstrom]; ``lagrange`` real (decimation, stencil).
    """
    n_fft = _FE2_FFT_SIZE
    _, template, window, trapezoid = _fe2_lattice_host()
    sigma_min = (_FE2_MIN_FWHM_KMS / 2.3548 / _C_LIGHT_KMS) / (_FE2_GRID_STEP_KMS / _C_LIGHT_KMS)
    f_cut = np.sqrt(-0.5 * np.log(_FE2_TRUNCATION)) / (np.pi * sigma_min)
    n_keep = int(np.ceil(f_cut * n_fft)) + 1
    spec = np.fft.rfft(template, n=n_fft)[:n_keep]
    mult = np.full(n_keep, 2.0)
    mult[0] = 1.0

    def functional(weights):
        w_hat = np.fft.rfft(weights, n=n_fft)[:n_keep]
        return mult * np.real(np.conj(w_hat) * spec) / n_fft

    return {
        "spectrum": spec / _FE2_DECIMATION,
        "window": functional(window),
        "total": functional(trapezoid),
        "lagrange": _lagrange_table(_FE2_DECIMATION, _FE2_STENCIL),
    }


def _fe2_is_concrete_narrow(fwhm_kms) -> bool:
    """Whether ``fwhm_kms`` is a concrete width below ``_FE2_MIN_FWHM_KMS`` (full-lattice path)."""
    if isinstance(fwhm_kms, jax.core.Tracer):
        return False
    return float(fwhm_kms) < _FE2_MIN_FWHM_KMS


def _fe2_transfer(fwhm_kms, n_keep):
    """Gaussian transfer function on the first ``n_keep`` Fourier bins of the full lattice.

    ``exp(-2 pi^2 sigma_s^2 f^2)`` with ``sigma_s`` in lattice samples, ``f = k / N`` in
    cycles/sample, as in :func:`_fe2_broadened_on_grid` (unit DC gain).
    """
    sigma_samples = (fwhm_kms / 2.3548 / _C_LIGHT_KMS) / (_FE2_GRID_STEP_KMS / _C_LIGHT_KMS)
    freq = jnp.arange(n_keep) / _FE2_FFT_SIZE
    return jnp.exp(-2.0 * (jnp.pi * sigma_samples * freq) ** 2)


def _fe2_window_flux_and_total(fwhm_kms):
    """R_Fe-window flux and total integral of the broadened lattice template (band-limited path).

    Returns
    -------
    window_flux, total : scalar
        ``sum_i w_i y_i`` and ``trapezoid(y, wave)`` of the broadened full-lattice template
        [Angstrom], each a K-term sum against the transfer function. NaN when ``fwhm_kms`` is
        below ``_FE2_MIN_FWHM_KMS`` (the tables carry no Fourier content past that cutoff).
    """
    tab = _fe2_spectral_tables()
    transfer = _fe2_transfer(fwhm_kms, tab["window"].shape[0])
    window_flux = jnp.sum(device_table(tab["window"]) * transfer)
    total = jnp.sum(device_table(tab["total"]) * transfer)
    # A traced width below the cutoff the tables were truncated for cannot raise; it returns NaN
    # (every spectrum and power divides by these), never a low-passed spectrum.
    in_contract = fwhm_kms >= _FE2_MIN_FWHM_KMS
    return jnp.where(in_contract, window_flux, jnp.nan), jnp.where(in_contract, total, jnp.nan)


def _fe2_sample_band_limited(wavelength, fwhm_kms):
    """Broadened FeII template at the caller's wavelengths, from the band-limited synthesis.

    One inverse FFT builds the broadened template on the coarse lattice; each caller wavelength
    then takes the two full-lattice nodes that bracket it by Lagrange interpolation from the coarse
    lattice and interpolates linearly between them in wavelength, the value
    ``jnp.interp(wavelength, grid_wave, broadened)`` returns on the full lattice. Zero outside
    the lattice's wavelength range.

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Rest-frame wavelength [Angstrom].
    fwhm_kms : float
        BLR velocity broadening FWHM [km/s], at or above ``_FE2_MIN_FWHM_KMS``.

    Returns
    -------
    array, shape (n_wave,)
        Broadened template on the scale of :func:`_fe2_internal_grid` (peak of the unbroadened
        template = 1) [dimensionless].
    """
    tab = _fe2_spectral_tables()
    dln = _FE2_GRID_STEP_KMS / _C_LIGHT_KMS
    lo = np.log(_FE2_GRID_LAMBDA_RANGE[0])
    n_grid = int(np.ceil((np.log(_FE2_GRID_LAMBDA_RANGE[1]) - lo) / dln)) + 1
    n_coarse = _FE2_FFT_SIZE // _FE2_DECIMATION
    n_keep = tab["spectrum"].shape[0]

    transfer = _fe2_transfer(fwhm_kms, n_keep)
    coarse = jnp.fft.irfft(device_table(tab["spectrum"]) * transfer, n=n_coarse)

    wave = jnp.asarray(wavelength)
    lattice_lo = float(np.exp(lo))
    lattice_hi = float(np.exp(lo + dln * (n_grid - 1)))
    inside = (wave >= lattice_lo) & (wave <= lattice_hi)
    pos = (jnp.log(jnp.where(inside, wave, lattice_lo)) - lo) / dln
    # floor of an in-range position; a round-off ULP below node 0 reads the periodic neighbor,
    # and the bracket's linear weight stays within one ULP of [0, 1], so no clamp is needed
    i0 = jnp.floor(pos).astype(jnp.int32)
    w0 = jnp.exp(lo + dln * i0)
    w1 = jnp.exp(lo + dln * (i0 + 1))
    frac = (wave - w0) / (w1 - w0)

    half = _FE2_STENCIL // 2
    offsets = jnp.arange(-half + 1, half + 1)
    lagrange = device_table(tab["lagrange"])

    def node(i):
        cols = (i // _FE2_DECIMATION)[:, None] + offsets[None, :]
        weights = lagrange[i % _FE2_DECIMATION]
        return jnp.sum(weights * coarse[cols % n_coarse], axis=-1)

    y0, y1 = node(i0), node(i0 + 1)
    value = y0 + (y1 - y0) * frac
    # truncating the spectrum can leave ~1e-11 negatives on a non-negative function
    return jnp.where(inside & (value > 0.0), value, 0.0)


def _fe2_pseudo_continuum(
    wavelength: jnp.ndarray,
    fwhm_kms: float,
    fe2_strength: float,
) -> jnp.ndarray:
    """Fe II pseudo-continuum from tabulated templates.

    Uses empirical Fe II templates from PyQSOFit (Temple, Hewett & Banerji 2021),
    which combine:

    - UV (1200–3500 Å): Vestergaard & Wilkes 2001 + Tsuzuki+2006
    - Optical (3500–7500 Å): Boroson & Green 1992

    The combined UV+optical template is carried on a fixed internal grid
    uniform in ln(lambda) (5 km/s step), broadened there by a constant-velocity
    Gaussian (BLR FWHM in km/s), normalized there, and only then sampled at the
    input wavelengths. The result is therefore independent of the caller's grid
    (coverage and sampling); JIT/grad/vmap safe in ``fwhm_kms`` (every size is
    static) and float32-safe (the template is scale-normalized).

    For ``fwhm_kms >= 500`` km/s the broadening costs one 2^15-point inverse FFT:
    the template's spectrum on the internal grid is a constant computed once, and
    the broadened spectrum has no Fourier content the transfer function leaves
    above 1e-10 beyond 6 669 of its 131 073 bins (see ``_fe2_spectral_tables``).
    That path matches the full-lattice convolution to 2.3e-9 relative
    (50 widths in 500-30000 km/s) and the R_Fe window flux to 1e-15. A traced
    ``fwhm_kms`` below 500 km/s returns NaN (it cannot raise, and a low-passed spectrum
    would be silently wrong); a concrete one takes the full-lattice convolution, which
    has no lower limit (``fwhm_kms -> 0`` is the template).

    **Unit convention.** The PyQSOFit template columns are F_lambda
    [erg/s/cm²/Å]; their *shape* is treated as the shape of L_lambda. The
    output is L_lambda per unit H-beta luminosity [Å⁻¹], normalized so that

    .. math::

        \\int_{4434\\,\\AA}^{4684\\,\\AA} \\hat L_\\lambda\\, d\\lambda
        = R_{\\rm Fe},

    i.e. ``fe2_strength`` is the standard R_Fe = F(Fe II 4434-4684) /
    F(H-beta) (Boroson & Green 1992) when multiplied by L(H-beta).
    Callers that need L_nu [erg/s/Hz] convert with
    ``L_nu = L_lambda * lambda**2 / c`` (``compute_blr_sed`` does); the
    composable FeII block returns L_lambda directly. Template nodes are
    clipped at zero (see :func:`_load_fe2_templates`) and resampled linearly
    in wavelength. A concrete zero ``fe2_strength`` (a Python or NumPy scalar)
    returns zeros without broadening the template; an array or traced zero runs
    it, because the strength can be gradient-carrying.

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Rest-frame wavelength [Angstrom].
    fwhm_kms : float
        BLR velocity broadening FWHM [km/s]. Gaussian of constant velocity
        width (constant width in ln lambda).
    fe2_strength : float
        R_Fe = F(Fe II 4434-4684) / F(H-beta). Typical range 0.5-2.0.
        Set to 0.0 to disable Fe II emission.

    Returns
    -------
    array, shape (n_wave,)
        Fe II L_lambda template [Å^-1] per unit H-beta luminosity,
        scaled by fe2_strength. Multiply by L(H-beta) [erg/s] to get
        L_lambda [erg/s/Å].

    References
    ----------

    - Temple, M. J., Hewett, P. C., & Banerji, M. 2021, MNRAS, 508, 737
    - Vestergaard, M., & Wilkes, B. J. 2001, ApJS, 134, 1 (UV Fe II)
    - Tsuzuki, Y., et al. 2006, ApJ, 650, 57 (UV/optical Fe II)
    - Boroson, T. A., & Green, R. F. 1992, ApJS, 80, 109 (optical Fe II)

    """
    if device_table(_FE2_UV_WAVE) is None or device_table(_FE2_OPT_WAVE) is None:
        raise RuntimeError(
            "Fe II templates not loaded. Check that fe_uv_pyqsofit.txt and "
            "fe_optical_pyqsofit.txt exist in src/tengri/data/agn_fe2/."
        )

    if _fe2_is_off(fe2_strength):
        # Absent by construction (a concrete zero): skip the template broadening.
        return jnp.zeros_like(jnp.asarray(wavelength))

    # Normalize: energy in the R_Fe window (4434-4684 A, Boroson & Green 1992) is
    # integral(L_lambda d lambda) = fe2_strength per unit L(H-beta). It is computed
    # on the internal grid, so neither the amplitude nor the broadening depends on
    # the caller's wavelength grid (coverage or sampling).
    if _fe2_is_concrete_narrow(fwhm_kms):
        grid_wave, _, window_weights, _ = _fe2_internal_grid()
        broadened = _fe2_broadened_on_grid(fwhm_kms)
        window_flux = jnp.sum(window_weights * broadened)
        # Sample the (smooth) broadened spectrum at the caller's wavelengths; zero outside
        # the internal grid's support.
        on_caller = jnp.interp(wavelength, grid_wave, broadened, left=0.0, right=0.0)
    else:
        window_flux, _ = _fe2_window_flux_and_total(fwhm_kms)
        on_caller = _fe2_sample_band_limited(wavelength, fwhm_kms)
    window_flux = jnp.maximum(window_flux, representable_denominator(1e-30))
    return fe2_strength * on_caller / window_flux


def _fe2_is_off(fe2_strength) -> bool:
    """Whether ``fe2_strength`` is a concrete zero (a Python or NumPy scalar), so FeII is absent.

    A traced or array-valued strength is never "off" here: it can be gradient-carrying or
    nonzero at run time, and the template broadening (one 2^18-point FFT) must then run.
    """
    return isinstance(fe2_strength, (int, float, np.number)) and float(fe2_strength) == 0.0


def _fe2_total_power(fwhm_kms, fe2_strength):
    """Integral of :func:`_fe2_pseudo_continuum` over all wavelengths [dimensionless].

    The broadened template of the internal grid, integrated by the trapezoid rule on
    that grid and divided by the same R_Fe window flux that normalizes the spectrum,
    so ``l_hbeta * _fe2_total_power(...)`` is the FeII power [erg/s]. The caller's
    wavelength grid does not enter.

    Parameters
    ----------
    fwhm_kms : float
        BLR velocity broadening FWHM [km/s].
    fe2_strength : float
        R_Fe = F(Fe II 4434-4684) / F(H-beta).

    Returns
    -------
    float
        ``int L_lambda d lambda`` per unit H-beta luminosity.

    Notes
    -----
    **JIT/grad/vmap-compatible**: yes. For ``fwhm_kms >= 500`` km/s the window flux and the total
    are K-term sums of host constants against the transfer function (no FFT, no lattice array);
    a concrete narrower width takes the full-lattice convolution.
    """
    if _fe2_is_off(fe2_strength):
        return 0.0
    if _fe2_is_concrete_narrow(fwhm_kms):
        grid_wave, _, window_weights, _ = _fe2_internal_grid()
        broadened = _fe2_broadened_on_grid(fwhm_kms)
        window_flux = jnp.sum(window_weights * broadened)
        total = jnp.trapezoid(broadened, grid_wave)
    else:
        window_flux, total = _fe2_window_flux_and_total(fwhm_kms)
    window_flux = jnp.maximum(window_flux, representable_denominator(1e-30))
    return fe2_strength * total / window_flux


def _blr_l_hbeta(
    l_disc_bol_erg: float,
    covering_fraction: float = _BLR_COVERING_FRACTION_DEFAULT,
    line_efficiency: float = _BLR_LINE_EFFICIENCY_DEFAULT,
) -> jnp.ndarray:
    """Compute H-beta luminosity from disc bolometric and BLR parameters.

    This helper computes the H-beta luminosity used to normalize the Fe II
    pseudo-continuum. It ensures consistent normalization between the
    analytic BLR (compute_blr_sed) and the standalone FeII block.

    Parameters
    ----------
    l_disc_bol_erg : float
        Bolometric disc luminosity [erg/s].
    covering_fraction : float, optional
        BLR covering fraction (0 to 1). Default 0.1.
    line_efficiency : float, optional
        Fraction of intercepted luminosity converted to line emission.
        Default 0.08.

    Returns
    -------
    jnp.ndarray
        H-beta luminosity [erg/s].

    Notes
    -----
    The H-beta strength in _BLR_LINES is 1.0 (rest wavelength 4862.68 Å,
    Vanden Berk et al. 2001). Its fractional share of total line luminosity
    is hbeta_strength / strength_sum, where strength_sum ≈ 25.0 is the sum
    of all _BLR_LINE_STRENGTHS.

    The Fe II pseudo-continuum (from _fe2_pseudo_continuum) is normalized
    per unit H-beta, so scaling by this value produces the absolute Fe II
    luminosity in the same units as compute_blr_sed.
    """
    hbeta_strength = 1.0000  # H-beta (4862.68 Å, VB01 rel flux 8.649)
    strength_sum = jnp.sum(device_table(_BLR_LINE_STRENGTHS))
    l_intercepted = covering_fraction * l_disc_bol_erg
    l_lines_total = line_efficiency * l_intercepted
    l_hbeta = (
        hbeta_strength
        * l_lines_total
        / jnp.maximum(strength_sum, representable_denominator(1e-30))
    )
    return l_hbeta


def compute_blr_sed(
    wavelength: jnp.ndarray,
    l_disc_bol_erg: float,
    covering_fraction: float = 0.1,
    fwhm_kms: float = _BLR_FWHM_KMS,
    agn_fe2_strength: float = 0.0,
    line_efficiency: float = _BLR_LINE_EFFICIENCY_DEFAULT,
    **_kwargs,
) -> jnp.ndarray:
    """BLR emission spectrum: broad permitted lines + Fe II pseudo-continuum.

    The BLR receives ``covering_fraction * L_disc`` and converts
    a fraction into broad emission lines. When ``agn_fe2_strength > 0``,
    an Fe II pseudo-continuum is added, scaled relative to H-beta
    luminosity using the standard R_Fe ratio.

    Note: geometric masking by the torus is NOT applied here;
    it must be applied by the caller.

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Rest-frame wavelength [Angstrom].
    l_disc_bol_erg : float
        Bolometric disc luminosity [erg s^-1].
    covering_fraction : float
        BLR covering fraction (0 to 1). Default 0.1.
    fwhm_kms : float
        Line FWHM [km/s]. Default 5000.
    agn_fe2_strength : float
        Fe II to H-beta flux ratio R_Fe = F(Fe II 4434-4684)/F(H-beta).
        Typical range 0.5-2.0. Default 0.0 (disabled).
    line_efficiency : float
        Fraction of intercepted luminosity converted to line emission.
        Default 0.08.

    Returns
    -------
    array, shape (n_wave,)
        BLR L_nu [erg s^-1 Hz^-1] (before torus masking).

    Notes
    -----
    **Traced FeII width**: the FeII broadening is evaluated by a band-limited synthesis that is
    valid for a width of 500 km/s or more. A concrete width (a Python or NumPy scalar, as every
    model build passes: the width is a block keyword, not a fit parameter) below 500 km/s takes
    the exact full-lattice convolution, with no lower limit. A *traced* width (``fwhm_kms``)
    below 500 km/s cannot raise inside a trace and returns NaN for the FeII spectrum and
    power, rather than a silently low-passed spectrum, so the same value is finite eager
    and NaN under ``jax.jit``. Keep a traced width >= 500 km/s.

    **JIT-compatible**: yes, uses ``jnp`` primitives and ``jax.vmap``.

    The broad emission lines are modeled as Gaussian profiles at rest-frame
    wavelengths. The line list (≥25 lines) is calibrated to the Vanden Berk
    et al. (2001) SDSS composite quasar spectrum, including:

    - UV lines: Lyα, Lyβ, N V, Si IV, C IV, He II, C III], Mg II
    - Optical lines: Balmer series (H-α, H-β, H-γ, H-δ, H-ε) and
      higher-order Paschen series

    The Fe II pseudo-continuum follows the Tsuzuki+2006 / Kovacevic+2010
    approach: broad Gaussians at UV and optical multiplet centers, normalized
    to the standard R_Fe ratio.

    **Torus geometry**: This function returns the "bare" BLR spectrum without
    geometric masking by the dusty torus. The caller is responsible for
    applying inclination-dependent torus obscuration if using a torus model.

    References
    ----------
    .. [1] D. E. Vanden Berk et al., "Composite Quasar Spectra from the Sloan
       Digital Sky Survey," AJ, 122, 549 (2001).
       https://doi.org/10.1086/321167
    .. [2] H. Netzer, "Accretion Power in Astrophysics," Cambridge University
       Press (1990). Chapter 2: Broad-line region models.
    .. [3] T. A. Boroson and R. F. Green, "The Emission-Line Properties of
       Low-Redshift Quasi-stellar Objects," ApJS, 80, 109 (1992).
       https://doi.org/10.1086/191679
    .. [4] Y. Tsuzuki et al., "Very Large Array Imaging of Submillimeter
       Galaxies," ApJ, 650, 57 (2006). https://doi.org/10.1086/506270
    .. [5] M. Vestergaard and R. F. Green, "Equivalent Widths and Scaling
       Relations in Quasar Emission Lines," ApJS, 134, 1 (2001).
       https://doi.org/10.1086/320360
    """
    l_intercepted = covering_fraction * l_disc_bol_erg
    l_lines_total = line_efficiency * l_intercepted

    # Sum broad Gaussian profiles for each line
    def _single_line(line_data):
        """Compute Gaussian line profile at rest wavelength with FWHM broadening."""
        lam_c = line_data[0]
        strength = line_data[1]
        profile = _gaussian_line_profile(wavelength, lam_c, fwhm_kms)
        return strength * l_lines_total * profile

    from jax import vmap

    line_spectra = vmap(_single_line)(device_table(_BLR_LINES))
    strength_sum = jnp.sum(device_table(_BLR_LINE_STRENGTHS))
    l_nu_blr = jnp.sum(line_spectra, axis=0) / jnp.maximum(
        strength_sum, representable_denominator(1e-30)
    )

    # Fe II pseudo-continuum (scaled relative to H-beta luminosity)
    l_hbeta = _blr_l_hbeta(l_disc_bol_erg, covering_fraction, line_efficiency)

    # _fe2_pseudo_continuum returns L_lambda [1/Angstrom] per unit H-beta
    # luminosity; this function returns L_nu, so L_nu = L_lambda * lambda^2 / c.
    fe2_spectrum = _fe2_pseudo_continuum(wavelength, fwhm_kms, agn_fe2_strength)
    l_nu_blr = l_nu_blr + l_hbeta * fe2_spectrum * wavelength**2 / _C_AA

    return l_nu_blr
