# SPDX-License-Identifier: BSD-3-Clause
"""Rest-frame SED spectral diagnostics for galaxy analysis.

This module provides pure-JAX functions for computing rest-frame spectral
diagnostics (UV slopes, break indices, colors, etc.) from SED arrays.

All functions are JIT-compatible and fully differentiable.

Available Diagnostics
---------------------

- **uv_slope_beta**: Calzetti et al. (1994) UV spectral slope
- **dn4000**: Balogh et al. (1999) 4000-Å break index
- **irx**: Infrared excess from dust and FUV luminosity
- **rest_frame_luminosity**: Synthetic photometry through a filter
- **rest_frame_color**: Magnitude difference through two filters

References
----------

- Calzetti, D., Kinney, A. L., Storchi-Bergmann, T., 1994, ApJ, 429, 582
- Balogh, M. L., Morris, S. L., Yee, H. K. C., Carlberg, R. G., Ellingson, E.,
  1999, ApJ, 527, 54

"""

import jax
import jax.numpy as jnp

from tengri.utils.filter_convention import FilterConvention, filter_weight as _filter_weight
from tengri.utils.physics_constants import C_AA
from tengri.utils.scale import representable_denominator

# ── UV Slope (Calzetti et al. 1994) ────────────────────────────────


def uv_slope_beta(wavelength_aa: jnp.ndarray, l_nu: jnp.ndarray) -> float:
    r"""UV spectral slope β from Calzetti et al. (1994) 10-window method.

    Computes the UV slope by fitting log(F_lambda) vs log(lambda) in 10 clean
    spectral windows that avoid the 2175-Å dust bump and strong spectral
    features.

    Parameters
    ----------
    wavelength_aa: array_like, shape (n_wave,)
        Rest-frame wavelength grid in Ångstrom. [Å]
    l_nu: array_like, shape (n_wave,)
        Luminosity density L_ν. [erg s⁻¹ Hz⁻¹] or [Lsun Hz⁻¹]

    Returns
    -------
    float
        Spectral slope β where F_λ ∝ λ^β. [dimensionless]

    Notes
    -----
    **JIT-compatible**: yes, all operations are ``jnp`` primitives.

    The method uses 10 wavelength windows (Calzetti+1994 Table 2) that avoid
    the 2175-Å absorption feature and Ly-alpha:

    ===== ==================
    No.   Range (Ångstrom)
    ===== ==================
    1     1268–1284
    2     1309–1316
    3     1342–1371
    4     1407–1515
    5     1562–1583
    6     1611–1711
    7     1760–1833
    8     1866–1890
    9     1930–1950
    10    2400–2580
    ===== ==================

    Within each window, we extract F_λ values and perform a linear regression
    of log(F_λ) vs log(λ) to measure the slope β. The final result is the
    weighted mean across windows.

    References
    ----------
    .. [1] Calzetti, D., Kinney, A. L., Storchi-Bergmann, T., 1994,
           ApJ, 429, 582.
           https://doi.org/10.1086/174330
    """
    window_lo = jnp.array(
        [
            1268.0,
            1309.0,
            1342.0,
            1407.0,
            1562.0,
            1611.0,
            1760.0,
            1866.0,
            1930.0,
            2400.0,
        ]
    )
    window_hi = jnp.array(
        [
            1284.0,
            1316.0,
            1371.0,
            1515.0,
            1583.0,
            1711.0,
            1833.0,
            1890.0,
            1950.0,
            2580.0,
        ]
    )

    f_lambda = l_nu * (C_AA / (wavelength_aa**2))

    log_w = jnp.log(wavelength_aa)
    log_f = jnp.log(jnp.maximum(f_lambda, 1e-40))

    def _window_slope(lo_hi):
        """Compute slope and weight for a wavelength window."""
        lo, hi = lo_hi
        mask = ((wavelength_aa >= lo) & (wavelength_aa <= hi)).astype(jnp.float64)
        n = jnp.sum(mask)
        mean_x = jnp.sum(mask * log_w) / jnp.maximum(n, 1.0)
        mean_y = jnp.sum(mask * log_f) / jnp.maximum(n, 1.0)
        cov_xy = jnp.sum(mask * (log_w - mean_x) * (log_f - mean_y))
        var_x = jnp.sum(mask * (log_w - mean_x) ** 2)
        slope = jnp.where(var_x > 0.0, cov_xy / var_x, 0.0)
        weight = jnp.where(n >= 2.0, 1.0, 0.0)
        return slope, weight

    slopes, weights = jax.vmap(_window_slope)(jnp.stack([window_lo, window_hi], axis=1))
    total_weight = jnp.sum(weights)
    beta = jnp.where(total_weight > 0.0, jnp.sum(slopes * weights) / total_weight, 0.0)

    return beta


# ── Dn4000 (Balogh et al. 1999) ────────────────────────────────────


def dn4000(wavelength_aa: jnp.ndarray, l_nu: jnp.ndarray) -> float:
    r"""Dn4000 narrow-band break index (Balogh et al. 1999).

    Computes the ratio of average rest-frame flux density in a red band
    (4000–4100 Å) to a blue band (3850–3950 Å).

    Parameters
    ----------
    wavelength_aa: array_like, shape (n_wave,)
        Rest-frame wavelength grid in Ångstrom. [Å]
    l_nu: array_like, shape (n_wave,)
        Luminosity density L_ν. [erg s⁻¹ Hz⁻¹] or [Lsun Hz⁻¹]

    Returns
    -------
    float
        Dn4000 index: ratio of red to blue band flux. [dimensionless]

    Notes
    -----
    **JIT-compatible**: yes, all operations are ``jnp`` primitives.

    Dn4000 is a diagnostic of age: young star-forming galaxies with hot OB
    stars have Dn4000 ≈ 0.4–1.0 (weak 4000-Å break), while old quiescent
    galaxies have Dn4000 > 1.5 (strong break).

    References
    ----------
    .. [1] Balogh, M. L., Morris, S. L., Yee, H. K. C., Carlberg, R. G.,
           Ellingson, E., 1999, ApJ, 527, 54.
           https://doi.org/10.1086/308056
    """
    blue_mask = (wavelength_aa >= 3850.0) & (wavelength_aa <= 3950.0)
    red_mask = (wavelength_aa >= 4000.0) & (wavelength_aa <= 4100.0)

    n_blue = jnp.sum(blue_mask)
    n_red = jnp.sum(red_mask)

    f_blue = jnp.sum(jnp.where(blue_mask, l_nu, 0.0)) / jnp.maximum(n_blue, 1.0)
    f_red = jnp.sum(jnp.where(red_mask, l_nu, 0.0)) / jnp.maximum(n_red, 1.0)

    return jnp.where(
        f_blue > 0.0,
        f_red / f_blue,
        0.0,
    )


# ── Infrared Excess (IRX) ──────────────────────────────────────────


def irx(l_dust: float, l_fuv: float) -> float:
    r"""Infrared excess IRX as a function of dust and FUV luminosity.

    IRX quantifies the ratio of dust emission (IR) to non-attenuated UV
    emission (FUV).

    Parameters
    ----------
    l_dust: float
        Total dust emission luminosity. [Lsun] or [erg s⁻¹]
    l_fuv: float
        Far-UV (rest-frame 1500 Å) luminosity. [Lsun] or [erg s⁻¹]

    Returns
    -------
    float
        log₁₀(L_dust / L_FUV). [dimensionless]

    Notes
    -----
    **JIT-compatible**: yes, all operations are ``jnp`` primitives.

    The infrared excess is:

    .. math::

        \text{IRX} = \log_{10}(L_{\rm dust} / L_{\rm FUV})

    High IRX (> 0.5) indicates significant dust attenuation. IRX ≈ 0
    corresponds to unobscured star formation. IRX < 0 is rare and indicates
    highly ionized ISM with little dust.

    Typical IRX–UV slope (β) relations (Meurer+1999):

    - Starbursts: IRX ≈ β_UV + 2.0 − 2.5
    - Quiescent: IRX ≈ β_UV + 0.7

    See also ``uv_slope_beta``.
    """
    # Clamp to avoid log(0)
    l_dust_safe = jnp.maximum(l_dust, 1e-40)
    l_fuv_safe = jnp.maximum(l_fuv, 1e-40)
    return jnp.log10(l_dust_safe / l_fuv_safe)


# ── Rest-frame Photometry ──────────────────────────────────────────


# ── Equivalent Width ──────────────────────────────────────────────


def equivalent_width(
    wavelength_aa: jnp.ndarray,
    l_nu: jnp.ndarray,
    line_center_aa: float,
    window_aa: float = 20.0,
    continuum_width_aa: float = 50.0,
) -> float:
    r"""Rest-frame equivalent width of a spectral feature.

    Estimates the continuum from symmetric sidebands flanking the line
    and integrates (F - F_cont) / F_cont over the line window.

    Parameters
    ----------
    wavelength_aa: array_like, shape (n_wave,)
        Rest-frame wavelength grid [Å].
    l_nu: array_like, shape (n_wave,)
        Luminosity density L_ν [erg s⁻¹ Hz⁻¹].
    line_center_aa: float
        Central wavelength of the feature [Å].
    window_aa: float
        Half-width of the line integration window [Å].
        Default: 20.0.
    continuum_width_aa: float
        Width of each sideband used for continuum estimation [Å].
        Sidebands are placed at ``[line_center ± window ± continuum_width]``.
        Default: 50.0.

    Returns
    -------
    float
        Equivalent width [Å].  Positive for emission, negative for
        absorption (astronomical convention).

    Notes
    -----
    **JIT-compatible**: yes, all operations are ``jnp`` primitives.

    The equivalent width is:

    .. math::

        \mathrm{EW} = \int_{\lambda_0 - w}^{\lambda_0 + w}
        \frac{F(\lambda) - F_{\mathrm{cont}}}{F_{\mathrm{cont}}} \, d\lambda

    where :math:`F_{\mathrm{cont}}` is estimated as the mean flux in two
    symmetric sidebands flanking the line window.

    References
    ----------
    .. [1] Vollmann, K. & Eversberg, T., 2006, AN, 327, 862.
           Standard definition of spectroscopic equivalent width.
    """
    f_lambda = l_nu * (C_AA / (wavelength_aa**2))

    line_mask = (wavelength_aa >= line_center_aa - window_aa) & (
        wavelength_aa <= line_center_aa + window_aa
    )

    blue_lo = line_center_aa - window_aa - continuum_width_aa
    blue_hi = line_center_aa - window_aa
    red_lo = line_center_aa + window_aa
    red_hi = line_center_aa + window_aa + continuum_width_aa

    blue_mask = (wavelength_aa >= blue_lo) & (wavelength_aa <= blue_hi)
    red_mask = (wavelength_aa >= red_lo) & (wavelength_aa <= red_hi)

    n_blue = jnp.sum(blue_mask)
    n_red = jnp.sum(red_mask)
    f_cont = (
        jnp.sum(jnp.where(blue_mask, f_lambda, 0.0)) + jnp.sum(jnp.where(red_mask, f_lambda, 0.0))
    ) / jnp.maximum(n_blue + n_red, 1.0)

    integrand = jnp.where(
        line_mask,
        # Derivative-sized floor: f_cont is a denominator, so its VJP needs
        # 1/floor**2 representable (#1860). 1e-40 squares to 0.0 in float32.
        (f_lambda - f_cont) / jnp.maximum(f_cont, representable_denominator(1e-40)),
        0.0,
    )

    return jnp.trapezoid(integrand, wavelength_aa)


# ── Rest-frame Photometry ──────────────────────────────────────────


def rest_frame_luminosity(
    wavelength_aa: jnp.ndarray,
    l_nu: jnp.ndarray,
    filter_wave_aa: jnp.ndarray,
    filter_trans: jnp.ndarray,
    convention: FilterConvention = FilterConvention.BESSELL,
) -> float:
    r"""Synthetic rest-frame luminosity through a filter.

    Computes rest-frame photometry by convolving L_ν with a filter response,
    using the same bandpass weight as the observed-frame kernel so rest- and
    observed-frame photometry are consistent (matching DSPS ``calc_rest_flux``).

    Parameters
    ----------
    wavelength_aa: array_like, shape (n_wave,)
        Rest-frame wavelength grid in Ångstrom. [Å]
    l_nu: array_like, shape (n_wave,)
        Luminosity density L_ν. [erg s⁻¹ Hz⁻¹] or [Lsun Hz⁻¹]
    filter_wave_aa: array_like, shape (n_filter,)
        Filter wavelength grid in Ångstrom. [Å]
    filter_trans: array_like, shape (n_filter,)
        Filter transmission (normalized to peak = 1). [dimensionless]
    convention: FilterConvention, optional
        Bandpass weight. ``BESSELL`` (default) is photon-counting
        (:math:`w=1/\lambda`, matches DSPS ``calc_rest_flux``); ``ENERGY``
        is :math:`w=1/\lambda^2` (CIGALE).

    Returns
    -------
    float
        Synthetic luminosity. [Lsun] or [erg s⁻¹]

    Notes
    -----
    **JIT-compatible**: yes, all operations are ``jnp`` primitives.

    The photometric luminosity is:

    .. math::

        L = \frac{\int T(\lambda) L_\nu(\lambda) w(\lambda) d\lambda}
                 {\int T(\lambda) w(\lambda) d\lambda}

    where T(λ) is the filter transmission and :math:`w(\lambda)` the bandpass
    weight (:math:`1/\lambda` for ``BESSELL``, :math:`1/\lambda^2` for
    ``ENERGY``).

    The filter is interpolated to the wavelength grid of the SED using
    linear interpolation with zero-padding outside the filter range.
    """
    # Interpolate filter to SED wavelength grid
    filter_trans_interp = jnp.interp(
        wavelength_aa,
        filter_wave_aa,
        filter_trans,
        left=0.0,
        right=0.0,
    )

    # Weighted average with bandpass weight w(lambda)
    weight = filter_trans_interp * _filter_weight(wavelength_aa, convention)
    numerator = jnp.trapezoid(weight * l_nu, wavelength_aa)
    denominator = jnp.trapezoid(weight, wavelength_aa)

    return jnp.where(denominator > 0.0, numerator / denominator, 0.0)


# ── Rest-frame Color ───────────────────────────────────────────────


def rest_frame_color(
    wavelength_aa: jnp.ndarray,
    l_nu: jnp.ndarray,
    filter1_wave_aa: jnp.ndarray,
    filter1_trans: jnp.ndarray,
    filter2_wave_aa: jnp.ndarray,
    filter2_trans: jnp.ndarray,
) -> float:
    r"""Rest-frame color from two filters (magnitude difference).

    Computes rest-frame photometric color as m1 - m2.

    Parameters
    ----------
    wavelength_aa: array_like, shape (n_wave,)
        Rest-frame wavelength grid in Ångstrom. [Å]
    l_nu: array_like, shape (n_wave,)
        Luminosity density L_ν. [erg s⁻¹ Hz⁻¹] or [Lsun Hz⁻¹]
    filter1_wave_aa: array_like, shape (n_f1,)
        First filter wavelength grid in Ångstrom. [Å]
    filter1_trans: array_like, shape (n_f1,)
        First filter transmission. [dimensionless]
    filter2_wave_aa: array_like, shape (n_f2,)
        Second filter wavelength grid in Ångstrom. [Å]
    filter2_trans: array_like, shape (n_f2,)
        Second filter transmission. [dimensionless]

    Returns
    -------
    float
        Color m1 - m2 (in AB magnitudes). [mag]

    Notes
    -----
    **JIT-compatible**: yes, all operations are ``jnp`` primitives.

    The color is computed from the flux ratio:

    .. math::

        c = m_1 - m_2 = -2.5 \log_{10}(f_1 / f_2)

    where f1, f2 are synthetic fluxes through the two filters.

    If either flux is zero or negative (due to interpolation artifacts),
    the color is set to 0.
    """
    l1 = rest_frame_luminosity(wavelength_aa, l_nu, filter1_wave_aa, filter1_trans)
    l2 = rest_frame_luminosity(wavelength_aa, l_nu, filter2_wave_aa, filter2_trans)

    l1_safe = jnp.maximum(l1, 1e-40)
    l2_safe = jnp.maximum(l2, 1e-40)

    color = -2.5 * jnp.log10(l1_safe / l2_safe)

    return jnp.where(
        (l1 > 0.0) & (l2 > 0.0),
        color,
        0.0,
    )
