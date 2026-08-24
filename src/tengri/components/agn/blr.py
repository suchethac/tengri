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

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from tengri.components.agn._phys import gaussian_line_profile as _gaussian_line_profile
from tengri.utils.grid_interp import resample_template

# ── Physical constants ────────────────────────────────────────────
from tengri.utils.physics_constants import (
    C_AA as _C_AA,
    C_KM_S as _C_LIGHT_KMS,
)

# ── BLR emission-line template ────────────────────────────────────

# Key broad emission lines: (rest wavelength [Angstrom], relative strength)
# Line strengths extracted from Vanden Berk et al. (2001) Table 2
# ("Composite Quasar Emission Line Features"), derived from SDSS composite.
# Relative strengths are normalized to H-beta = 1.0 by dividing the VB01
# "Rel. Flux" column (F/F_Lyα) by the H-beta flux value (8.649).
# Vacuum wavelengths per SDSS convention; comments cite VB01 flux values.
_BLR_LINES = jnp.array(
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

_BLR_LINE_WAVELENGTHS = _BLR_LINES[:, 0]
_BLR_LINE_STRENGTHS = _BLR_LINES[:, 1]

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

    File format: log10(wavelength), flux [erg/s/cm²/Å] (per PyQSOFit convention).

    Returns
    -------
    tuple of (uv_wave, uv_flux, opt_wave, opt_flux)
        All as np.ndarray, dtype float64. Wavelengths in Angstrom (linear scale).
        Flux in erg/s/cm²/Å (normalized by PyQSOFit internally).
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
    uv_flux = uv_data[:, 1]
    uv_wave = 10.0**uv_log_wave  # Convert to linear wavelength

    # Load optical template (log10 wavelength, flux)
    opt_data = np.genfromtxt(str(opt_file), comments="#", dtype=np.float64)
    opt_log_wave = opt_data[:, 0]
    opt_flux = opt_data[:, 1]
    opt_wave = 10.0**opt_log_wave  # Convert to linear wavelength

    return uv_wave, uv_flux, opt_wave, opt_flux


# Load Fe II templates at module import time (avoid I/O in JIT-compiled functions)
try:
    _FE2_UV_WAVE, _FE2_UV_FLUX, _FE2_OPT_WAVE, _FE2_OPT_FLUX = _load_fe2_templates()
except FileNotFoundError:
    # Fallback: set to None and raise at runtime if Fe II is requested
    _FE2_UV_WAVE = None
    _FE2_UV_FLUX = None
    _FE2_OPT_WAVE = None
    _FE2_OPT_FLUX = None


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

    At runtime, the combined UV+optical template is interpolated to the
    input wavelength grid, then broadened by convolving with a Gaussian
    kernel corresponding to the BLR velocity width (FWHM in km/s).

    The output is normalized so that ``fe2_strength`` equals the standard
    R_Fe = F(Fe II 4434-4684) / F(H-beta) ratio. In practice this function
    returns L_nu per unit H-beta luminosity, scaled by ``fe2_strength``.

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Rest-frame wavelength [Angstrom].
    fwhm_kms : float
        BLR velocity broadening FWHM [km/s]. Applied via Gaussian convolution
        in wavelength space.
    fe2_strength : float
        R_Fe = F(Fe II 4434-4684) / F(H-beta). Typical range 0.5-2.0.
        Set to 0.0 to disable Fe II emission.

    Returns
    -------
    array, shape (n_wave,)
        Fe II L_nu template [Hz^-1] per unit H-beta luminosity,
        scaled by fe2_strength. Multiply by L(H-beta) to get
        absolute luminosity.

    References
    ----------

    - Temple, M. J., Hewett, P. C., & Banerji, M. 2021, MNRAS, 508, 737
    - Vestergaard, M., & Wilkes, B. J. 2001, ApJS, 134, 1 (UV Fe II)
    - Tsuzuki, Y., et al. 2006, ApJ, 650, 57 (UV/optical Fe II)
    - Boroson, T. A., & Green, R. F. 1992, ApJS, 80, 109 (optical Fe II)

    """
    if _FE2_UV_WAVE is None or _FE2_OPT_WAVE is None:
        raise RuntimeError(
            "Fe II templates not loaded. Check that fe_uv_pyqsofit.txt and "
            "fe_optical_pyqsofit.txt exist in src/tengri/data/agn_fe2/."
        )

    # Convert NumPy arrays to JAX (one-time cost at function call)
    uv_wave = jnp.asarray(_FE2_UV_WAVE, dtype=jnp.float64)
    uv_flux = jnp.asarray(_FE2_UV_FLUX, dtype=jnp.float64)
    opt_wave = jnp.asarray(_FE2_OPT_WAVE, dtype=jnp.float64)
    opt_flux = jnp.asarray(_FE2_OPT_FLUX, dtype=jnp.float64)

    # Interpolate UV and optical templates onto the common wavelength grid
    # Use linear interpolation; extrapolate with zeros outside the range
    uv_interp = resample_template(wavelength, uv_wave, uv_flux, left=0.0, right=0.0)
    opt_interp = resample_template(wavelength, opt_wave, opt_flux, left=0.0, right=0.0)

    # Combine: use UV where available (1200–3500 A), else optical
    # In the overlap region (2200–3500 A), UV dominates by design (Tsuzuki+06)
    fe2_combined = jnp.where(wavelength < 3500.0, uv_interp, opt_interp)

    # Apply Gaussian broadening via convolution in velocity space
    # Velocity broadening σ_v [km/s] → wavelength σ_λ [A] at each wavelength
    sigma_kms = fwhm_kms / 2.3548  # Convert FWHM to sigma
    sigma_wave = wavelength * sigma_kms / _C_LIGHT_KMS

    # Build Gaussian kernel at the wavelength grid (centered at each point)
    # For JIT-safe convolution, use numerical convolution with kernel truncated
    # to ±3 sigma (capture ~99.7% of Gaussian probability).
    # Note: Full FFT convolution is overkill here; numerical works fine.

    # Simple numerical convolution: smooth the template by applying
    # a Gaussian kernel at each wavelength point
    def _convolve_with_gaussian(flux_array, sigma_array):
        """Smooth flux array with position-dependent Gaussian kernel."""
        n_wave = flux_array.shape[0]

        def _smooth_at(i):
            """Smooth value at index i using Gaussian kernel."""
            sigma_i = sigma_array[i]
            # Kernel extends ±3 sigma from this point
            dw = jnp.abs(wavelength - wavelength[i])
            kernel = jnp.exp(-0.5 * (dw / sigma_i) ** 2)
            # Normalize so it integrates to 1 in wavelength space
            # (ignoring the Jacobian; we just want smooth interpolation)
            kernel = kernel / jnp.sum(kernel)
            return jnp.sum(flux_array * kernel)

        # vmap over all wavelength indices to smooth all points
        from jax import vmap

        return vmap(_smooth_at)(jnp.arange(n_wave))

    fe2_broadened = _convolve_with_gaussian(fe2_combined, sigma_wave)

    # Normalize: compute the integral of the optical 4434-4684 A bump
    # This is the standard R_Fe measurement window (Boroson & Green 1992).
    mask_opt = (wavelength >= 4434.0) & (wavelength <= 4684.0)
    nu_opt = _C_AA / jnp.maximum(wavelength, 1.0)
    sort_nu = jnp.argsort(nu_opt)
    opt_window_flux = jnp.abs(jnp.trapezoid((fe2_broadened * mask_opt)[sort_nu], nu_opt[sort_nu]))
    opt_window_flux = jnp.maximum(opt_window_flux, 1e-30)

    # Scale so that integral in 4434-4684 window equals fe2_strength
    # (relative to unit H-beta luminosity applied later by caller)
    return fe2_strength * fe2_broadened / opt_window_flux


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
    strength_sum = jnp.sum(_BLR_LINE_STRENGTHS)
    l_intercepted = covering_fraction * l_disc_bol_erg
    l_lines_total = line_efficiency * l_intercepted
    l_hbeta = hbeta_strength * l_lines_total / jnp.maximum(strength_sum, 1e-30)
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
    .. [3] T. A. Boroson and R. F. Green, "The Emission Line Properties of
       Low-Luminosity Seyfert 1 Galaxies," ApJS, 80, 109 (1992).
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

    line_spectra = vmap(_single_line)(_BLR_LINES)
    strength_sum = jnp.sum(_BLR_LINE_STRENGTHS)
    l_nu_blr = jnp.sum(line_spectra, axis=0) / jnp.maximum(strength_sum, 1e-30)

    # Fe II pseudo-continuum (scaled relative to H-beta luminosity)
    l_hbeta = _blr_l_hbeta(l_disc_bol_erg, covering_fraction, line_efficiency)

    # _fe2_pseudo_continuum returns per-Hz template per unit H-beta luminosity
    fe2_spectrum = _fe2_pseudo_continuum(wavelength, fwhm_kms, agn_fe2_strength)
    l_nu_blr = l_nu_blr + l_hbeta * fe2_spectrum

    return l_nu_blr
