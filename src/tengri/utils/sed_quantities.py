# SPDX-License-Identifier: BSD-3-Clause
"""Pure JAX functions for computing derived physical quantities from SEDs.

This module provides the computational primitives used by both the lazy
``Prediction`` object (for single-galaxy exploration) and the JIT-compatible
``predict_sfh_quantities`` / ``predict_sed_quantities`` methods (for
population-level batch computation via ``jax.vmap``).

All functions are:

- **Pure**: no side effects, no mutation, no caching
- **JIT-compatible**: can be wrapped in ``jax.jit``
- **Differentiable**: gradients flow through all computations
- **Static-shape**: use ``jnp.where`` masks (not dynamic slicing)
  so array shapes are known at trace time

Physical conventions
--------------------

- SED units: erg/s/Hz (rest-frame luminosity L_ν)
- Wavelength: Angstrom (ascending order in ``ssp_wave``)
- Frequency: Hz (``ν = c / λ``, descending when λ is ascending)
- Luminosity integrals: ``L = -∫ L_ν dν`` (negative sign because ν
  is descending when integrated via ``jnp.trapezoid`` with ascending λ)
- Line luminosities: Lsun
- Time: years (ages), converted to Gyr for output where noted
- Mass: Msun

References
----------

- Balogh et al. 1999, ApJ, 527, 54: Dn4000 definition
- Wang et al. 2024, ApJ; modified Balmer break
- Bell 2003, ApJ, 586, 794: FIR-radio correlation
- Murphy et al. 2011, ApJ, 737, 67: radio-SFR calibration
- Lehmer et al. 2010, ApJ, 724, 559: XRB scaling relations
- Lehmer et al. 2016, ApJ, 825, 7: updated XRB scaling
- Duras et al. 2020, A&A, 636, A73: AGN bolometric corrections
- Condon 1992, ARA&A, 30, 575: thermal radio emission

"""

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

from tengri.utils.magnitudes import fnu_to_ab_mag, lnu_to_absolute_ab_mag
from tengri.utils.physics_constants import C_AA, L_SUN, PC_CM
from tengri.utils.scale import LN10, log10_magnitude, pow10, representable_denominator

# Re-export for convenience
__all__ = [
    "C_AA",
    "L_SUN",
    "PC_CM",
]

# ── Key emission line wavelengths (rest-frame, Angstrom) ──────────

KEY_LINES = {
    "lya": (1215.67,),
    "civ_1549": (1548.19, 1550.78),
    "oii": (3727.12, 3730.12),
    "hbeta": (4862.76,),
    "oiii_4959": (4960.30,),
    "oiii_5007": (5008.31,),
    "nii_6548": (6549.96,),
    "halpha": (6564.72,),
    "nii_6584": (6585.37,),
    "sii_6717": (6718.40,),
    "sii_6731": (6732.78,),
}
"""Key emission lines for survey diagnostics.

Each value is a tuple of rest-frame wavelengths (Angstrom). For doublets
(e.g., [OII] 3726+3729), both components are listed and their luminosities
are summed by :func:`extract_line_luminosity`.
"""


# ── SFH-based quantities (no SED needed) ──────────────────────────


def _FLOOR() -> float:
    """The ``1e-50`` guard floor these quantities use, made representable (#1492).

    ``1e-50`` is below float32's smallest subnormal (1.4e-45), so every floor
    in this module was **exactly 0.0** in float32: ``log(0) = -inf``, and the
    UV-slope regression came back NaN where float64 returned a finite (wrong
    but survivable) number. ``representable_floor`` raises it to the working
    dtype's smallest normal; float64 keeps ``1e-50`` unchanged.
    """
    from tengri.utils.scale import representable_floor

    return representable_floor(1e-50)


def compute_mass_weighted_age(weights: jnp.ndarray, ssp_ages_yr: jnp.ndarray) -> jnp.ndarray:
    """Mass-weighted stellar age.

    Parameters
    ----------
    weights : array, shape (n_age,)
        CSP mass weights (Msun per SSP age bin), from
        :func:`~tengri.components.stellar.sps.dsps_wrapper.compute_csp_weights`.
    ssp_ages_yr : array, shape (n_age,)
        SSP isochrone ages in years (lookback time).

    Returns
    -------
    float
        Mass-weighted age in Gyr:
        ``Σ(w_i × age_i) / Σ(w_i) / 1e9``.
        NaN when ``Σ(w_i) == 0``; with no mass there is no mass-weighted age.

    Notes
    -----
    **JIT-compatible**: yes.

    Degenerate input returns NaN rather than 0.0 (#1404). A clamped denominator
    alone would yield a finite ``0.0`` here, which reads as "every star just
    formed"; a plausible-looking answer for a model with no stellar mass at all.
    """
    # Select the denominator BEFORE dividing, not the quotient after. The outer
    # ``where`` picks NaN on the degenerate branch but does not protect the
    # reverse pass: both branches are differentiated, the quotient's VJP carries
    # -num/den**2, and a 1e-30 denominator squares to exactly 0.0 in float32
    # (below tiny = 1.18e-38), so the discarded branch contributes 0 * inf = NaN
    # to the surviving one (#1860).
    total = jnp.sum(weights)
    ok = total > 1e-20
    safe_total = jnp.where(ok, total, 1.0)
    return jnp.where(ok, jnp.sum(weights * ssp_ages_yr) / safe_total / 1e9, jnp.nan)


def compute_mass_weighted_metallicity(
    weights: jnp.ndarray,
    ssp_ages_yr: jnp.ndarray,
    log_z: float,
    log_z_initial: float | None = None,
    log_z_final: float | None = None,
) -> jnp.ndarray:
    """Mass-weighted metallicity.

    .. warning::
        ``log_z_initial`` and ``log_z_final`` must be Python ``None``
        or concrete floats, not JAX traced values. The ``if None``
        branch is resolved at trace time, so this function is JIT-safe
        only when the evolving-Z flag is a static Python bool.

    For a single metallicity parameter, this is trivially ``log_z``.
    For evolving metallicity, Z varies linearly with lookback time:

    .. math::

        \\log Z(t) = \\log Z_{\\rm final}
            + (\\log Z_{\\rm initial} - \\log Z_{\\rm final}) \\times t/t_{\\max}

    The mass-weighted value is then ``log10(Σ(w_i × Z_i) / Σ(w_i))``,
    where ``Z_i = 10^{log_z_i}`` is the linear metallicity at each bin.

    Parameters
    ----------
    weights : array, shape (n_age,)
        CSP mass weights.
    ssp_ages_yr : array, shape (n_age,)
        SSP ages in years.
    log_z : float
        Single metallicity log10(Z) (used when not evolving).
    log_z_initial : float, optional
        Initial (oldest) metallicity log10(Z). If None, returns ``log_z``.
    log_z_final : float, optional
        Final (present-day) metallicity log10(Z).

    Returns
    -------
    float
        Mass-weighted metallicity in log10(Z).
    """
    if log_z_initial is None or log_z_final is None:
        return log_z

    t_max = jnp.max(ssp_ages_yr)
    t_frac = ssp_ages_yr / jnp.maximum(t_max, 1.0)
    log_z_per_bin = log_z_final + (log_z_initial - log_z_final) * t_frac
    z_linear = 10.0**log_z_per_bin
    total_w = jnp.sum(weights)
    # NaN, not 0.0, when there is no mass to weight by (#1404). Denominator
    # selected before the divide: see compute_mass_weighted_age for why the
    # outer ``where`` alone leaves the reverse pass NaN in float32 (#1860).
    ok = total_w > 1e-20
    safe_total_w = jnp.where(ok, total_w, 1.0)
    mean_z = jnp.where(ok, jnp.sum(weights * z_linear) / safe_total_w, jnp.nan)
    return jnp.log10(jnp.maximum(mean_z, 1e-30))


# ── SED-based quantities ──────────────────────────────────────────

#: log10 of the solar luminosity [dex re erg/s]. Folded into the bolometric
#: reductions so the erg/s value is never materialized (see _trapz_to_lsun).
LOG10_L_SUN: float = float(np.log10(L_SUN))


def derived_luminosity_lsun(
    derived: Mapping[str, Any], key: str, log_key: str, default: float = 0.0
) -> jnp.ndarray:
    r"""Read an erg/s ``state.derived`` key in :math:`L_\odot`, log companion first.

    The cross-component contract publishes its energy-balance luminosities in
    erg/s (``L_ir``, ``L_absorbed``) alongside a ``log10`` companion
    (``log_L_ir``). For a :math:`10^{10}\,M_\odot` galaxy the linear key is
    ~3.6e43 and is ``inf`` in float32, while the companion is ~43.6 dex and
    exact: and the attenuator computes the companion *first*
    (``L_ir = pow10(log_L_ir)``), so reading it is strictly closer to the
    source. Consumers that divided the linear key by :math:`L_\odot` returned
    ``inf`` for a ~9.5e9 :math:`L_\odot` answer that float32 holds easily
    (issue #1837).

    Parameters
    ----------
    derived : Mapping
        ``state.derived``.
    key : str
        Linear key name [erg/s], used only when the companion is absent.
    log_key : str
        ``log10`` companion key name [dex re erg/s].
    default : float, optional
        Value in erg/s when neither key is present. Default 0.0.

    Returns
    -------
    ndarray, shape ()
        The luminosity in :math:`L_\odot`. Exactly ``0.0`` when the companion
        is ``-inf`` (the "this term is exactly zero" sentinel).

    Notes
    -----
    **JIT/grad/vmap-compatible**: yes. The key presence test is a Python-level
    branch on a static dict, not a traced value.
    """
    log_value = derived.get(log_key)
    if log_value is not None:
        return pow10(jnp.asarray(log_value) - LOG10_L_SUN)
    return jnp.asarray(derived.get(key, default)) / L_SUN


def derived_weights_peak_relative(
    derived: Mapping[str, Any], key: str, log_key: str
) -> jnp.ndarray:
    r"""Per-bin weights from an erg/s ``state.derived`` array, rescaled by their peak.

    For weights used only inside :math:`\sum x_i w_i / \sum w_i`, any factor
    common to every bin cancels exactly, so the absolute scale is free to
    discard; and discarding it is what makes the mean computable in float32.
    ``L_age`` peaks at ~3.3e42 erg/s, so 85 of 93 bins are ``inf`` there and
    ``ssp_ages_yr * L_age`` overflows a second time on top (~1e10 x), while the
    weighted mean itself is of order 1 (issue #1837).

    Parameters
    ----------
    derived : Mapping
        ``state.derived``.
    key : str
        Linear per-bin key [erg/s], used only when the companion is absent.
    log_key : str
        ``log10`` companion key [dex re erg/s].

    Returns
    -------
    ndarray, shape (n_bin,)
        Weights in ``[0, 1]``, the brightest bin exactly ``1.0``. All-zero when
        every bin is dark.

    Notes
    -----
    **JIT/grad/vmap-compatible**: yes. ``-inf`` is the "this bin emits nothing"
    sentinel and powers back to exactly ``0.0``; an all-dark array leaves the
    peak non-finite, so the offset falls back to zero and every weight
    underflows to ``0.0`` exactly as the linear path did.
    """
    log_values = derived.get(log_key)
    if log_values is None:
        return jnp.asarray(derived[key])
    log_values = jnp.asarray(log_values)
    peak = jnp.max(log_values)
    peak = jnp.where(jnp.isfinite(peak), peak, 0.0)
    return pow10(log_values - peak)


def _trapz_to_lsun(integrand: jnp.ndarray, nu: jnp.ndarray) -> jnp.ndarray:
    r"""``-∫ integrand dν`` expressed in :math:`L_\odot`, range-safe in float32.

    .. math::

        L = \frac{-\int L_\nu\,d\nu}{L_\odot}

    where :math:`L_\nu` is the integrand [erg/s/Hz] and :math:`\nu` the frequency
    grid [Hz]. Computing that literally forms the erg/s value first (~1e43 for a
    1e10 Msun galaxy), which exceeds the float32 ceiling of 3.4e38 and returns
    ``inf``; even though the :math:`L_\odot` answer (~1e9) is perfectly
    representable. Factoring the integrand by its peak and folding
    :math:`1/L_\odot` into the same exponent keeps every intermediate in range
    (issue #1206).

    Parameters
    ----------
    integrand : array_like, shape (n_wave,)
        Rest-frame :math:`L_\nu` [erg/s/Hz]; may be signed.
    nu : array_like, shape (n_wave,)
        Frequency grid [Hz], descending when wavelength ascends.

    Returns
    -------
    ndarray, shape ()
        The integral in :math:`L_\odot`. Exactly ``0.0`` for an all-zero
        integrand.

    Notes
    -----
    **JIT/grad/vmap-compatible**: yes. Equal to the naive form to ~1e-14 relative
    in float64; finite in float32 whenever the :math:`L_\odot` result is.
    """
    # stop_gradient: pure factorization constant (#1436). The peak divides the
    # integrand and multiplies back through pow10(log10(peak)), so its derivative
    # contributions cancel analytically. Left free they are two autodiff paths that
    # cancel in float64 but not float32, leaving an uncancelled term.
    peak = jax.lax.stop_gradient(jnp.max(jnp.abs(integrand), initial=0.0))
    peak = jnp.where(peak > 0, peak, jnp.ones_like(peak))
    norm = -jnp.trapezoid(integrand / peak, nu)
    return norm * pow10(jnp.log10(peak) - LOG10_L_SUN)


def compute_bolometric_luminosity(sed: jnp.ndarray, wave: jnp.ndarray) -> jnp.ndarray:
    """Bolometric luminosity from the full SED.

    Integrates L_ν over all frequencies:

    .. math::

        L_{\\rm bol} = -\\int L_\\nu \\, d\\nu

    The negative sign arises because ν is descending when λ is ascending.

    Parameters
    ----------
    sed : array, shape (n_wave,)
        Rest-frame SED in erg/s/Hz.
    wave : array, shape (n_wave,)
        Wavelength grid in Angstrom (ascending).

    Returns
    -------
    float
        Bolometric luminosity in Lsun.
    """
    return _trapz_to_lsun(sed, C_AA / wave)


def compute_l_tir(sed: jnp.ndarray, wave: jnp.ndarray) -> jnp.ndarray:
    """Total infrared luminosity (8–1000 μm).

    Standard definition used in IRX-β studies and IR luminosity functions.

    Parameters
    ----------
    sed : array, shape (n_wave,)
        Rest-frame SED in erg/s/Hz.
    wave : array, shape (n_wave,)
        Wavelength grid in Angstrom.

    Returns
    -------
    float
        L_TIR in Lsun. Zero if no flux in the 8–1000 μm range.
    """
    mask = (wave >= 8.0e4) & (wave <= 1.0e7)  # 8-1000 μm in Angstrom
    sed_ir = jnp.where(mask, sed, 0.0)
    return jnp.maximum(_trapz_to_lsun(sed_ir, C_AA / wave), 0.0)


def compute_l_dust_absorbed(
    sed_intrinsic: jnp.ndarray, sed_attenuated: jnp.ndarray, wave: jnp.ndarray
) -> jnp.ndarray:
    """Dust-absorbed luminosity.

    The energy removed from the stellar SED by dust attenuation:

    .. math::

        L_{\\rm abs} = \\int (L_{\\nu,\\rm intrinsic}
                       - L_{\\nu,\\rm attenuated}) \\, d\\nu

    This should equal L_TIR when ``dust_eta_balance = 1.0`` (strict
    energy balance).

    Parameters
    ----------
    sed_intrinsic : array, shape (n_wave,)
        Unattenuated stellar SED in erg/s/Hz.
    sed_attenuated : array, shape (n_wave,)
        Dust-attenuated stellar SED in erg/s/Hz.
    wave : array, shape (n_wave,)
        Wavelength grid in Angstrom.

    Returns
    -------
    float
        Dust-absorbed luminosity in Lsun.
    """
    absorbed = sed_intrinsic - sed_attenuated
    return jnp.maximum(_trapz_to_lsun(absorbed, C_AA / wave), 0.0)


def _mean_flux_in_band(sed, wave, lam_lo, lam_hi):
    """Mean flux density in a wavelength band.

    Helper for spectral indices (Dn4000, Balmer break, FUV/NUV).
    Uses ``jnp.where`` masks to keep shapes static for JIT.

    Parameters
    ----------
    sed : array, shape (n_wave,)
        SED in erg/s/Hz.
    wave : array, shape (n_wave,)
        Wavelength in Angstrom (ascending).
    lam_lo, lam_hi : float
        Band edges in Angstrom.

    Returns
    -------
    float
        Mean L_ν in the band (erg/s/Hz).
    """
    mask = (wave >= lam_lo) & (wave <= lam_hi)
    w = mask.astype(sed.dtype)
    # Trapezoid-weighted mean: ∫(sed * dλ) / ∫(dλ) within the band
    sed_masked = jnp.where(mask, sed, 0.0)
    num = jnp.trapezoid(sed_masked, wave)
    den = jnp.trapezoid(w, wave)
    # Return NaN if the band has no wavelength coverage (den ≈ 0). Denominator
    # selected before the divide: see compute_mass_weighted_age (#1860).
    ok = den > 1e-20
    return jnp.where(ok, num / jnp.where(ok, den, 1.0), jnp.nan)


def compute_dn4000(sed: jnp.ndarray, wave: jnp.ndarray) -> jnp.ndarray:
    """Narrow 4000 Å break index (Balogh et al. 1999).

    Defined as the ratio of mean f_ν in the red (4000–4100 Å) to blue
    (3850–3950 Å) windows:

    .. math::

        D_n(4000) = \\frac{\\langle f_\\nu \\rangle_{4000-4100}}
                          {\\langle f_\\nu \\rangle_{3850-3950}}

    Values range ~1.0 (young starbursts) to ~2.5 (old passive galaxies).

    Parameters
    ----------
    sed : array, shape (n_wave,)
        SED in erg/s/Hz.
    wave : array, shape (n_wave,)
        Wavelength in Angstrom.

    Returns
    -------
    float
        Dn4000 (dimensionless).
    """
    red = _mean_flux_in_band(sed, wave, 4000.0, 4100.0)
    blue = _mean_flux_in_band(sed, wave, 3850.0, 3950.0)
    # Denominator floor sized for its derivative, not its value: 1e-30 squares
    # to exactly 0.0 in float32 so the quotient's VJP divides by zero (#1860).
    return red / jnp.maximum(blue, representable_denominator(1e-30))


def compute_balmer_break(sed: jnp.ndarray, wave: jnp.ndarray) -> jnp.ndarray:
    """Modified Balmer break (Wang et al. 2024).

    Optimized for high-redshift galaxies where the classical D4000
    (metal absorption) is weak. Uses a bluer window (3620–3720 Å)
    that captures the hydrogen bound-free discontinuity:

    .. math::

        BB = \\frac{\\langle f_\\nu \\rangle_{4000-4100}}
                   {\\langle f_\\nu \\rangle_{3620-3720}}

    Parameters
    ----------
    sed : array, shape (n_wave,)
        SED in erg/s/Hz.
    wave : array, shape (n_wave,)
        Wavelength in Angstrom.

    Returns
    -------
    float
        Balmer break strength (dimensionless).
    """
    red = _mean_flux_in_band(sed, wave, 4000.0, 4100.0)
    blue = _mean_flux_in_band(sed, wave, 3620.0, 3720.0)
    # Derivative-sized denominator floor: see compute_dn4000 (#1860).
    return red / jnp.maximum(blue, representable_denominator(1e-30))


def compute_uv_slope_beta(sed: jnp.ndarray, wave: jnp.ndarray) -> jnp.ndarray:
    """UV spectral slope β from 1250–2600 Å.

    Fit to the power-law form ``f_λ ∝ λ^β``. Since the SED is in
    f_ν units, we fit ``ln(f_ν)`` vs ``ln(λ)`` and subtract 2:

    .. math::

        \\beta = \\frac{d \\ln f_\\nu}{d \\ln \\lambda} - 2

    Uses analytic weighted least-squares with a boolean mask as weights,
    keeping array shapes static for JIT.

    Parameters
    ----------
    sed : array, shape (n_wave,)
        SED in erg/s/Hz.
    wave : array, shape (n_wave,)
        Wavelength in Angstrom.

    Returns
    -------
    float
        UV slope β (typically -2.5 to 0 for star-forming galaxies).
    """
    mask = (wave >= 1250.0) & (wave <= 2600.0)
    w = mask.astype(sed.dtype)
    log_wave = jnp.log(jnp.maximum(wave, 1.0))
    log_fnu = jnp.log(jnp.maximum(sed, _FLOOR()))

    # Weighted linear regression: slope = (Σwxy - ΣwxΣwy/Σw) / (Σwx² - (Σwx)²/Σw)
    sw = jnp.sum(w)
    sx = jnp.sum(w * log_wave)
    sy = jnp.sum(w * log_fnu)
    sxx = jnp.sum(w * log_wave**2)
    sxy = jnp.sum(w * log_wave * log_fnu)

    # Three denominators, all previously floored at 1e-30: derivative-unsafe in
    # float32, where 1e-60 flushes to 0.0 and the quotient VJP divides by zero.
    # ``sw`` is selected before the divide (the trailing ``where`` guards the
    # value, not the reverse pass); ``denom`` can vanish on a genuinely
    # degenerate fit, so it takes a derivative-sized floor instead (#1860).
    ok = sw > 1.0
    safe_sw = jnp.where(ok, sw, 1.0)
    denom = sxx - sx**2 / safe_sw
    slope_fnu = (sxy - sx * sy / safe_sw) / jnp.maximum(denom, representable_denominator(1e-30))
    # Return NaN if no wavelength points in the 1250-2600 Å window
    return jnp.where(ok, slope_fnu - 2.0, jnp.nan)


def compute_fuv_flux(sed: jnp.ndarray, wave: jnp.ndarray) -> jnp.ndarray:
    """Mean flux density in the far-UV (1000–1700 Å).

    Traces star formation on ~10–30 Myr timescales, dominated by
    O- and B-type stars.

    Parameters
    ----------
    sed : array, shape (n_wave,)
        SED in erg/s/Hz.
    wave : array, shape (n_wave,)
        Wavelength in Angstrom.

    Returns
    -------
    float
        Mean f_ν in erg/s/Hz.
    """
    return _mean_flux_in_band(sed, wave, 1000.0, 1700.0)


def compute_nuv_flux(sed: jnp.ndarray, wave: jnp.ndarray) -> jnp.ndarray:
    """Mean flux density in the near-UV (1700–3200 Å).

    Traces star formation on ~30–100 Myr timescales, including
    contributions from early- to mid-B type stars.

    Parameters
    ----------
    sed : array, shape (n_wave,)
        SED in erg/s/Hz.
    wave : array, shape (n_wave,)
        Wavelength in Angstrom.

    Returns
    -------
    float
        Mean f_ν in erg/s/Hz.
    """
    return _mean_flux_in_band(sed, wave, 1700.0, 3200.0)


def compute_m_uv(sed: jnp.ndarray, wave: jnp.ndarray) -> jnp.ndarray:
    """Absolute UV magnitude at rest-frame 1500 Å.

    Standard quantity for UV luminosity functions. Computed as the
    mean f_ν in a 100 Å window around 1500 Å, converted to absolute
    AB magnitude assuming the SED is at 10 pc:

    .. math::

        M_{\\rm UV} = -2.5 \\log_{10}\\left(
            \\frac{\\langle L_\\nu \\rangle}{4\\pi (10\\,{\\rm pc})^2}
        \\right) - 48.6

    Parameters
    ----------
    sed : array, shape (n_wave,)
        SED in erg/s/Hz (rest-frame luminosity).
    wave : array, shape (n_wave,)
        Wavelength in Angstrom.

    Returns
    -------
    float
        M_UV in AB magnitudes.
    """
    l_nu = _mean_flux_in_band(sed, wave, 1450.0, 1550.0)
    return lnu_to_absolute_ab_mag(l_nu)


def compute_uv_luminosity_1600(sed: jnp.ndarray, wave: jnp.ndarray) -> jnp.ndarray:
    """Monochromatic UV luminosity νL_ν at rest-frame 1600 Å.

    Used in IRX-β and ionizing efficiency calculations.

    Parameters
    ----------
    sed : array, shape (n_wave,)
        SED in erg/s/Hz.
    wave : array, shape (n_wave,)
        Wavelength in Angstrom.

    Returns
    -------
    float
        νL_ν at 1600 Å in erg/s.
    """
    l_nu_1600 = jnp.interp(1600.0, wave, sed)
    nu_1600 = C_AA / 1600.0
    return nu_1600 * l_nu_1600


#: ``log10(c / 1600 A)`` [dex re Hz]: the 1600 A pivot frequency, kept in the
#: exponent so ``nu L_nu`` is never materialized (see
#: :func:`compute_log_uv_luminosity_1600`).
LOG10_NU_1600: float = float(np.log10(C_AA / 1600.0))


def compute_log_uv_luminosity_1600(sed: jnp.ndarray, wave: jnp.ndarray) -> jnp.ndarray:
    r"""``log10`` of the monochromatic UV luminosity at rest-frame 1600 A.

    .. math::

        \log_{10}\left(\frac{(\nu L_\nu)_{1600\,\mathrm{A}}}{\mathrm{erg/s}}\right)

    The range-safe companion to :func:`compute_uv_luminosity_1600`. That
    function returns :math:`\nu L_\nu` in erg/s, which is ~5e42 for a
    :math:`10^{10}\,M_\odot` galaxy and therefore **not representable in
    float32** at all: its ``inf`` then propagated into ``irx`` as ``NaN``, even
    though IRX itself is a dex ratio of order unity (issue #1837).

    Parameters
    ----------
    sed : array_like, shape (n_wave,)
        Rest-frame :math:`L_\nu` [erg/s/Hz].
    wave : array_like, shape (n_wave,)
        Rest-frame wavelength grid [Angstrom], ascending.

    Returns
    -------
    ndarray, shape ()
        :math:`\log_{10}(\nu L_\nu)` [dex re erg/s]. ``-inf`` where the
        interpolated :math:`L_\nu` is exactly zero, following the
        :func:`~tengri.utils.scale.log10_magnitude` sentinel contract.

    Notes
    -----
    **JIT/grad/vmap-compatible**: yes. Equal to
    ``log10(compute_uv_luminosity_1600(...))`` to ~1e-15 relative in float64,
    and finite in float32 wherever :math:`L_\nu` itself is.
    """
    l_nu_1600 = jnp.interp(1600.0, wave, sed)
    return log10_magnitude(l_nu_1600) + LOG10_NU_1600


def compute_irx(
    l_tir_lsun: jnp.ndarray,
    l_uv_erg: jnp.ndarray | None = None,
    *,
    log_l_uv_erg: jnp.ndarray | None = None,
) -> jnp.ndarray:
    r"""Infrared excess :math:`\mathrm{IRX} = \log_{10}(L_\mathrm{TIR}/L_\mathrm{UV})`.

    .. math::

        \mathrm{IRX} = \log_{10}\left(\frac{L_\mathrm{TIR}}{L_\mathrm{UV}}\right)
                     = \log_{10} L_\mathrm{TIR}[L_\odot] + \log_{10} L_\odot
                       - \log_{10} L_\mathrm{UV}[\mathrm{erg/s}]

    Parameters
    ----------
    l_tir_lsun : array_like, shape ()
        Total IR luminosity [Lsun].
    l_uv_erg : array_like, shape (), optional
        UV luminosity :math:`\nu L_\nu` [erg/s]. Mutually exclusive with
        ``log_l_uv_erg``. **Not float32-representable** for a normal galaxy
        (~5e42 against a 3.4e38 ceiling); prefer the log form there.
    log_l_uv_erg : array_like, shape (), optional
        :math:`\log_{10}(\nu L_\nu / (\mathrm{erg/s}))` [dex], as returned by
        :func:`compute_log_uv_luminosity_1600`. The float32-safe route.

    Returns
    -------
    ndarray, shape ()
        IRX [dex].

    Raises
    ------
    TypeError
        If neither or both of ``l_uv_erg`` and ``log_l_uv_erg`` are given.

    Notes
    -----
    **JIT/grad/vmap-compatible**: yes.

    Evaluated as a difference of logarithms rather than a ratio. The previous
    form materialized ``l_tir_lsun * L_SUN`` (~7e41 erg/s), which overflows
    float32 on its own; so IRX was ``NaN`` there even when both inputs were
    finite, and even though IRX is a dex ratio of order unity (issue #1837).
    Clamping in the log domain is exactly equivalent to the previous linear
    clamp because ``log10`` is monotone:
    ``log10(max(x, f)) == max(log10(x), log10(f))``. float64 is unchanged to
    ~1e-15 absolute.
    """
    if (l_uv_erg is None) == (log_l_uv_erg is None):
        raise TypeError(
            "compute_irx requires exactly one of l_uv_erg (linear, erg/s) or "
            "log_l_uv_erg (dex). Pass log_l_uv_erg from "
            "compute_log_uv_luminosity_1600 for a float32-safe result."
        )
    log_floor = jnp.log10(jnp.asarray(_FLOOR()))
    log_l_tir_erg = jnp.maximum(log10_magnitude(l_tir_lsun) + LOG10_L_SUN, log_floor)
    if log_l_uv_erg is None:
        log_uv = jnp.maximum(log10_magnitude(l_uv_erg), log_floor)
    else:
        log_uv = jnp.maximum(jnp.asarray(log_l_uv_erg), log_floor)
    return log_l_tir_erg - log_uv


def compute_rest_uv_color(sed: jnp.ndarray, wave: jnp.ndarray) -> jnp.ndarray:
    """Rest-frame U-V color from rectangular band approximations.

    Uses approximate Johnson U (3200–3900 Å) and V (5000–5800 Å)
    bands. Sufficient for UVJ classification; for precision photometry
    use :meth:`~tengri.SEDModel.predict_magnitudes` with loaded
    filter curves.

    Parameters
    ----------
    sed : array, shape (n_wave,)
        SED in erg/s/Hz.
    wave : array, shape (n_wave,)
        Wavelength in Angstrom.

    Returns
    -------
    float
        U - V in AB magnitudes.
    """
    f_u = _mean_flux_in_band(sed, wave, 3200.0, 3900.0)
    f_v = _mean_flux_in_band(sed, wave, 5000.0, 5800.0)
    mag_u = fnu_to_ab_mag(f_u)
    mag_v = fnu_to_ab_mag(f_v)
    return mag_u - mag_v


# ── Luminosity-weighted quantities (need per-bin SED info) ────────


#: Dynamic range demanded of the per-bin sum before a luminosity-weighted mean
#: is considered meaningful. A pure ratio, so it carries no units and survives
#: any rescaling of the bins.
_WEIGHT_SUM_REL_FLOOR = 1e-12


def _emits_enough_to_weight_by(l_per_bin: jnp.ndarray, l_total: jnp.ndarray) -> jnp.ndarray:
    """Is there enough light for ``sum(l * x) / sum(l)`` to mean anything?

    Scale-free by construction: compares the sum against the largest single
    bin, so it tests the *shape* of the distribution rather than its
    magnitude.

    This replaces a bare ``l_total > 1e-20``. That constant was chosen when
    :func:`_per_bin_luminosity_relative` returned erg/s; the helper now divides
    by the peak of ``ssp_flux_at_z`` and drops ``L_sun``, rescaling its output
    by ~3.8e18, so the threshold stopped being anchored to anything the moment
    the units moved under it. It kept passing because the live regime sits ~44
    decades clear of it either way: the constant did not change, its meaning
    did.

    Parameters
    ----------
    l_per_bin : array_like, shape (n_age,)
        Per-age-bin luminosity in any common units.
    l_total : array_like, scalar
        ``sum(l_per_bin)``, passed in because both callers already have it.

    Returns
    -------
    ndarray, shape (), bool
        True when a weighted mean is meaningful.

    Notes
    -----
    **JIT/grad/vmap-safe**: yes. Boolean output, so it carries no gradient; the
    callers pair it with a where-dummy denominator to keep theirs finite.
    """
    scale = jnp.max(jnp.abs(l_per_bin), initial=0.0)
    return l_total > _WEIGHT_SUM_REL_FLOOR * scale


def _per_bin_luminosity_relative(
    weights: jnp.ndarray, ssp_flux_at_z: jnp.ndarray, wave: jnp.ndarray
) -> jnp.ndarray:
    """Per-age-bin bolometric luminosity up to one common positive factor.

    The erg/s spelling this replaces multiplied by ``L_sun`` *inside* the
    vmapped body, forming ~1e41 per bin; above the float32 ceiling. Every
    consumer forms a luminosity-weighted *average* (``sum(l * x) / sum(l)``),
    where any common positive factor cancels exactly, so that value is never
    needed. Dropping the ``L_sun`` conversion and the overall peak is therefore
    behavior-preserving for the ratios and removes the overflow (issue #1206).

    Parameters
    ----------
    weights : array_like, shape (n_age,)
        CSP mass weights [Msun per bin].
    ssp_flux_at_z : array_like, shape (n_age, n_wave)
        Metallicity-interpolated SSP flux [Lsun/Hz/Msun].
    wave : array_like, shape (n_wave,)
        Wavelength grid [Angstrom].

    Returns
    -------
    ndarray, shape (n_age,)
        Per-bin luminosity in arbitrary (shared) units; ratios only.

    Notes
    -----
    **JIT/grad/vmap-compatible**: yes.
    """
    nu = C_AA / wave
    # stop_gradient: factorization constant (#1436). Unlike the other peak-factored
    # reductions this one never multiplies the peak back: but both consumers
    # (luminosity-weighted age and metallicity, the only two, and this helper is
    # private) divide by ``sum(l_per_bin)``, so ``(sum L_i a_i / p) / (sum L_j / p)``
    # is exactly p-independent and the peak's derivative is analytically zero.
    # That is what the "ratios only" contract above buys.
    #
    # The one regime where p would survive is the degenerate branch in those callers
    # (see ``_emits_enough_to_weight_by``). It cannot arise for a real galaxy: the
    # sum runs many decades above the largest bin's rounding: and that branch returns
    # NaN, which has no meaningful derivative either way.
    peak = jax.lax.stop_gradient(jnp.max(jnp.abs(ssp_flux_at_z), initial=0.0))
    peak = jnp.where(peak > 0, peak, jnp.ones_like(peak))

    def _lbol_one_bin(w_i, flux_i):
        return -jnp.trapezoid(w_i * (flux_i / peak), nu)

    return jax.vmap(_lbol_one_bin)(weights, ssp_flux_at_z)


def compute_luminosity_weighted_age(
    weights: jnp.ndarray,
    ssp_flux_at_z: jnp.ndarray,
    ssp_ages_yr: jnp.ndarray,
    wave: jnp.ndarray,
) -> jnp.ndarray:
    """Luminosity-weighted stellar age.

    Weights each SSP bin by its bolometric luminosity rather than its
    mass, giving a diagnostic biased toward the light-dominating
    population.

    Parameters
    ----------
    weights : array, shape (n_age,)
        CSP mass weights.
    ssp_flux_at_z : array, shape (n_age, n_wave)
        Metallicity-interpolated SSP flux.
    ssp_ages_yr : array, shape (n_age,)
        SSP ages in years.
    wave : array, shape (n_wave,)
        Wavelength in Angstrom.

    Returns
    -------
    float
        Luminosity-weighted age in Gyr.
    """
    l_per_bin = _per_bin_luminosity_relative(weights, ssp_flux_at_z, wave)
    l_total = jnp.sum(l_per_bin)
    live = _emits_enough_to_weight_by(l_per_bin, l_total)
    # NaN, not 0.0, when the population emits nothing to weight by (#1404).
    return jnp.where(
        live,
        jnp.sum(l_per_bin * ssp_ages_yr) / jnp.where(live, l_total, 1.0) / 1e9,
        jnp.nan,
    )


def compute_luminosity_weighted_metallicity(
    weights: jnp.ndarray,
    ssp_flux_at_z: jnp.ndarray,
    ssp_ages_yr: jnp.ndarray,
    wave: jnp.ndarray,
    log_z: float,
    log_z_initial: float | None = None,
    log_z_final: float | None = None,
) -> jnp.ndarray:
    """Luminosity-weighted metallicity.

    Parameters
    ----------
    weights : array, shape (n_age,)
        CSP mass weights.
    ssp_flux_at_z : array, shape (n_age, n_wave)
        Metallicity-interpolated SSP flux.
    ssp_ages_yr : array, shape (n_age,)
        SSP ages in years.
    wave : array, shape (n_wave,)
        Wavelength in Angstrom.
    log_z : float
        Single metallicity log10(Z).
    log_z_initial, log_z_final : float, optional
        For evolving metallicity.

    Returns
    -------
    float
        Luminosity-weighted metallicity in log10(Z).
    """
    if log_z_initial is None or log_z_final is None:
        return log_z

    l_per_bin = _per_bin_luminosity_relative(weights, ssp_flux_at_z, wave)
    l_total = jnp.sum(l_per_bin)

    t_max = jnp.max(ssp_ages_yr)
    t_frac = ssp_ages_yr / jnp.maximum(t_max, 1.0)
    log_z_per_bin = log_z_final + (log_z_initial - log_z_final) * t_frac
    z_linear = 10.0**log_z_per_bin

    live = _emits_enough_to_weight_by(l_per_bin, l_total)
    # NaN, not 0.0, when the population emits nothing to weight by (#1404).
    mean_z = jnp.where(
        live, jnp.sum(l_per_bin * z_linear) / jnp.where(live, l_total, 1.0), jnp.nan
    )
    return jnp.log10(jnp.maximum(mean_z, 1e-30))


# ── Emission line extraction ──────────────────────────────────────


def extract_line_luminosity(
    line_waves: jnp.ndarray, line_lums: jnp.ndarray, target_waves: tuple[float, ...]
) -> jnp.ndarray:
    """Extract emission line luminosity by wavelength matching.

    For doublets (multiple target wavelengths), the luminosities
    of all matched components are summed.

    Parameters
    ----------
    line_waves : array, shape (n_lines,)
        Rest-frame line wavelengths from nebular model.
    line_lums : array, shape (n_lines,)
        Line luminosities. **Unit-preserving**: this function indexes and sums,
        so the output carries whatever unit the input did. Its caller
        :func:`~tengri.forward.component_factory.state_to_emission_lines`
        passes ``state.derived["line_lums"]``, which is [erg/s].
    target_waves : tuple of float
        Target wavelength(s) in Angstrom. For doublets, pass both
        components (e.g., ``(3727.12, 3730.12)`` for [OII]).

    Returns
    -------
    float
        Total line luminosity, in the same unit as ``line_lums``. Returns NaN
        if ``line_waves`` is empty (no nebular model).

    Notes
    -----
    This said "Lsun" on both sides until #1559, at which point the only caller
    had been passing erg/s for some time. Nothing computed the wrong answer
    (the function never converts), but the docstring was evidence for the belief
    that the published catalog was in Lsun, which is how three backends came to
    publish it that way.
    """
    if line_waves.shape[0] == 0:
        return jnp.array(jnp.nan)

    def _lookup_one(target):
        """Extract line luminosity by nearest-wavelength matching."""
        idx = jnp.argmin(jnp.abs(line_waves - target))
        return line_lums[idx]

    total = jnp.array(0.0)
    for tw in target_waves:
        total = total + _lookup_one(tw)

    return total


def extract_log_line_luminosity(
    line_waves: jnp.ndarray, log_line_lums: jnp.ndarray, target_waves: tuple[float, ...]
) -> jnp.ndarray:
    r"""``log10`` of :func:`extract_line_luminosity`, without forming the linear value.

    The float32-safe companion. Line luminosities are ~1e40-1e42 erg/s, past
    float32's 3.4e38 ceiling, so the linear extraction is ``inf`` there and a
    ``log10`` taken afterwards inherits it; a log companion computed *after* the
    overflow is a no-op (#1534). This reads the upstream ``log_line_lums`` instead
    and never leaves the log domain.

    Parameters
    ----------
    line_waves : array, shape (n_lines,)
        Rest-frame line wavelengths [Angstrom].
    log_line_lums : array, shape (n_lines,)
        ``log10`` line luminosities [dex re erg/s]. **Unit-preserving in the same
        sense as the linear form**: the output is ``log10`` of whatever unit the
        input is the ``log10`` of.
    target_waves : tuple of float
        Target wavelength(s) [Angstrom]. For doublets, pass both components.

    Returns
    -------
    ndarray, scalar
        ``log10`` of the summed luminosity [dex]. NaN if ``line_waves`` is empty,
        matching the linear form.

    Notes
    -----
    **JIT-compatible**: yes. **Gradient-safe**: yes; ``logsumexp`` is smooth, and a
    component that is exactly zero enters as ``-inf`` and drops out of the sum
    without producing NaN.

    .. math::

        \log_{10} \sum_k L_k = \frac{1}{\ln 10}\,
            \mathrm{logsumexp}_k\!\left(\ln 10 \cdot \log_{10} L_k\right)

    **The sum is the whole difficulty.** Doublets ([OII] 3727+3730, and the
    ``key_lines`` entries that pair components) sum their matched entries, and a
    sum is not a log-domain operation; taking the max, or adding the logs, would
    both be wrong. ``logsumexp`` is exact for it and is the same primitive
    ``_derive_cue_params_from_ssp`` uses for ``total_logqion``.

    For a single-wavelength target the sum has one term and this reduces to the
    stored value exactly, so the common case costs nothing in accuracy.
    """
    if line_waves.shape[0] == 0:
        return jnp.array(jnp.nan)

    def _lookup_one(target):
        idx = jnp.argmin(jnp.abs(line_waves - target))
        return log_line_lums[idx]

    stacked = jnp.stack([_lookup_one(tw) for tw in target_waves])
    return logsumexp(LN10 * stacked) / LN10


# ── Radio quantities (empirical scaling relations) ────────────────


def compute_l_radio_1p4ghz_from_sfr(sfr: jnp.ndarray) -> jnp.ndarray:
    """1.4 GHz radio luminosity from SFR (Murphy et al. 2011).

    .. math::

        L_{1.4\\,{\\rm GHz}} = \\frac{\\rm SFR}{5.52 \\times 10^{-22}}

    Parameters
    ----------
    sfr : float
        Star formation rate in Msun/yr.

    Returns
    -------
    float
        L_1.4GHz in erg/s/Hz.
    """
    return sfr / 5.52e-22


def compute_l_radio_thermal(q_h: jnp.ndarray) -> jnp.ndarray:
    """Thermal (free-free) radio luminosity at 1.4 GHz from Q_H.

    Following Condon (1992), the thermal radio luminosity is:

    .. math::

        L_{\\rm th}(1.4\\,{\\rm GHz}) \\approx 5.5 \\times 10^{-28}
            \\times (T_e / 10^4)^{0.45} \\times Q_H

    assuming T_e = 10^4 K.

    Parameters
    ----------
    q_h : float
        Ionizing photon production rate in photons/s.

    Returns
    -------
    float
        Thermal radio luminosity at 1.4 GHz in erg/s/Hz.
    """
    return 5.5e-28 * q_h


def compute_l_radio_thermal_from_log_qh(log_q_h: jnp.ndarray) -> jnp.ndarray:
    r"""Thermal (free-free) radio luminosity at 1.4 GHz from :math:`\log_{10} Q_H`.

    Log-domain form of :func:`compute_l_radio_thermal`: folds the ~1e56 Q_H into the
    exponent so no float32-overflowing intermediate is materialized (#1206).

    .. math::

        \log_{10} L_{\rm th} = \log_{10} Q_H + \log_{10}(5.5\times10^{-28})

    Parameters
    ----------
    log_q_h : array_like, scalar
        log10 of the ionizing photon rate [dex re photons/s]; -inf for zero flux.

    Returns
    -------
    ndarray, shape ()
        Thermal 1.4 GHz luminosity [erg/s/Hz]; 0.0 when ``log_q_h`` is -inf.

    Notes
    -----
    JIT/grad/vmap-safe. Equals ``compute_l_radio_thermal(10**log_q_h)`` to ~1e-12 in
    float64 and stays finite in float32.
    """
    from tengri.utils.scale import pow10

    return pow10(log_q_h + jnp.log10(5.5e-28))


_TINY = 1e-30  # Floor for safe division (shared with radio/stellar components)


def compute_xi_ion_from_log_qh(
    log_q_h: jnp.ndarray, sed: jnp.ndarray, wave: jnp.ndarray
) -> jnp.ndarray:
    r"""Ionizing photon production efficiency :math:`\xi_{\rm ion}` [Hz/erg] from log10(Q_H).

    Computed in the log domain so the FUV energy density :math:`\nu L_\nu \sim 10^{43}` erg/s
    never materializes in float32 (#1206).

    .. math::

        \xi_{\rm ion} = \frac{Q_H}{\nu_{1500}\, L_{\nu,\rm FUV}}

    Parameters
    ----------
    log_q_h : array_like, scalar
        log10 ionizing photon rate [dex re photons/s].
    sed : array_like, shape (n_wave,)
        Rest-frame :math:`L_\nu` [erg/s/Hz] used to measure the FUV.
    wave : array_like, shape (n_wave,)
        Rest-frame wavelength grid [Angstrom].

    Returns
    -------
    ndarray, shape ()
        :math:`\xi_{\rm ion}` [Hz/erg]; float32-finite.
    """
    from tengri.utils.physics_constants import C_AA
    from tengri.utils.scale import pow10

    fuv = compute_fuv_flux(sed, wave)
    nu_uv = C_AA / 1500.0
    fuv_pos = fuv > 0
    log_nu_l_uv = jnp.where(
        fuv_pos, jnp.log10(jnp.where(fuv_pos, fuv, 1.0)) + jnp.log10(nu_uv), jnp.log10(_TINY)
    )
    log_nu_l_uv = jnp.maximum(log_nu_l_uv, jnp.log10(_TINY))
    return pow10(log_q_h - log_nu_l_uv)


def compute_q_ir(l_tir_lsun: jnp.ndarray, l_1p4ghz: jnp.ndarray) -> jnp.ndarray:
    """FIR-radio correlation parameter q_TIR.

    .. math::

        q_{\\rm TIR} = \\log_{10}\\left(
            \\frac{L_{\\rm TIR}}{3.75 \\times 10^{12}\\,{\\rm W}}
        \\right) - \\log_{10}\\left(
            \\frac{L_{1.4}}{\\rm W\\,Hz^{-1}}
        \\right)

    Typical value: q_TIR ≈ 2.64 (Bell 2003).

    Parameters
    ----------
    l_tir_lsun : float
        Total IR luminosity in Lsun.
    l_1p4ghz : float
        1.4 GHz luminosity in erg/s/Hz.

    Returns
    -------
    float
        q_TIR (dimensionless).
    """
    l_tir_w = l_tir_lsun * L_SUN * 1e-7  # erg/s → W
    l_radio_w = l_1p4ghz * 1e-7  # erg/s/Hz → W/Hz
    return jnp.log10(jnp.maximum(l_tir_w, _FLOOR()) / 3.75e12) - jnp.log10(
        jnp.maximum(l_radio_w, _FLOOR())
    )


# ── X-ray quantities (empirical scaling relations) ────────────────


def compute_l_x_xrb(sfr: jnp.ndarray, stellar_mass: jnp.ndarray) -> jnp.ndarray:
    """X-ray luminosity from X-ray binaries (0.5–8 keV).

    Combines high-mass XRBs (proportional to SFR) and low-mass XRBs
    (proportional to stellar mass) following Lehmer et al. (2010, 2016):

    .. math::

        L_{X,{\\rm XRB}} = 2.6 \\times 10^{39} \\times {\\rm SFR}
                          + 9.05 \\times 10^{28} \\times M_\\star

    Parameters
    ----------
    sfr : float
        Star formation rate in Msun/yr.
    stellar_mass : float
        Stellar mass in Msun.

    Returns
    -------
    float
        L_X in erg/s.
    """
    l_hmxb = 2.6e39 * sfr
    l_lmxb = 9.05e28 * stellar_mass
    return l_hmxb + l_lmxb


#: ``log10`` of the Lehmer+ XRB coefficients, as Python floats. Kept pre-logged
#: because ``2.6e39`` is past float32's ceiling: any expression that materializes it
#: in the working dtype: including ``jnp.log10(2.6e39)``; is ``inf`` before the log
#: is applied. Computed in numpy (float64) at import, read as a scalar thereafter.
_LOG10_HMXB_COEFF: float = float(np.log10(2.6e39))
_LOG10_LMXB_COEFF: float = float(np.log10(9.05e28))


def compute_log_l_x_xrb(sfr: jnp.ndarray, log_stellar_mass: jnp.ndarray) -> jnp.ndarray:
    r"""``log10`` of :func:`compute_l_x_xrb`, without forming the linear value.

    The float32-safe companion. The HMXB coefficient alone is
    :math:`2.6\times10^{39}`, past float32's 3.4e38 ceiling, so *the first term
    overflows before it is even multiplied by the SFR*: the linear form cannot be
    evaluated in float32 at any star formation rate, including zero.

    Parameters
    ----------
    sfr : array_like
        Star formation rate [Msun/yr].
    log_stellar_mass : array_like
        ``log10`` stellar mass [dex re Msun]. Taken in log because that is how the
        stellar component publishes it (``log_mstar``); the linear form is ~1e10
        and representable, but round-tripping through it is pointless.

    Returns
    -------
    ndarray
        ``log10(L_X,XRB / (erg/s))`` [dex].

    Notes
    -----
    **JIT-compatible**: yes. **Gradient-safe**: yes; an exactly-zero SFR enters as
    ``-inf`` and drops out of the sum without producing NaN.

    .. math::

        \log_{10} L_{X,\rm XRB} = \log_{10}\!\left(
            10^{\,39.415 + \log_{10}\rm SFR} + 10^{\,28.957 + \log_{10}M_\star}\right)

    evaluated as a base-10 ``logsumexp``. The two terms are the HMXB (SFR-tracking)
    and LMXB (mass-tracking) populations of Lehmer et al. (2010, 2016); they differ
    by ~10 decades, so the sum is dominated by one or the other and a naive
    ``max`` would be close but not equal; ``logsumexp`` is exact.

    References
    ----------
    .. [1] Lehmer, B. D. et al. "The 2 Ms Chandra Deep Field-North Survey and the
       740 ks Extended Chandra Deep Field-South Survey: Improved Point-Source
       Catalogs." 2010, ApJ, 724, 559. :doi:`10.1088/0004-637X/724/1/559`
    """
    log_sfr = log10_magnitude(jnp.asarray(sfr))
    # The COEFFICIENTS are pre-logged as Python floats. Writing `jnp.log10(2.6e39)`
    # instead puts 2.6e39 into a float32 array first, where it is already `inf`,
    # the log is taken of infinity and the whole companion returns `inf` on inputs
    # that are perfectly representable. Caught by this module's own float32 test.
    log_hmxb = _LOG10_HMXB_COEFF + log_sfr
    log_lmxb = _LOG10_LMXB_COEFF + jnp.asarray(log_stellar_mass)
    stacked = jnp.stack(jnp.broadcast_arrays(log_hmxb, log_lmxb))
    return logsumexp(LN10 * stacked, axis=0) / LN10


def compute_log_l_x_agn(log_l_bol_agn_erg: jnp.ndarray) -> jnp.ndarray:
    r"""``log10`` of :func:`compute_l_x_agn`, without forming the linear value.

    Parameters
    ----------
    log_l_bol_agn_erg : array_like
        ``log10`` AGN bolometric luminosity [dex re erg/s].

    Returns
    -------
    ndarray
        ``log10(L_X,AGN / (erg/s))`` [dex].

    Notes
    -----
    **JIT-compatible**: yes.

    The bolometric correction is *already* a function of the log luminosity:
    ``k_bol = a[1 + (log10(L_bol/Lsun)/b)^c]``; so the linear form takes a log,
    applies the correction, and then divides in linear space. This one stays in
    log throughout:

    .. math::

        \log_{10} L_{X,\rm AGN} = \log_{10} L_{\rm bol} - \log_{10} k_{\rm bol}

    Identical arithmetic on the correction itself, so it tracks
    :func:`compute_l_x_agn` to round-off in float64 and is finite in float32 where
    the linear form is ``inf``.

    References
    ----------
    .. [1] Duras, F. et al. "Universal bolometric corrections for active galactic
       nuclei over seven luminosity decades." 2020, A&A, 636, A73.
       :doi:`10.1051/0004-6361/201936817`
    """
    log_l_bol = jnp.asarray(log_l_bol_agn_erg)
    log_l_sol = log_l_bol - LOG10_L_SUN
    a, b, c = 15.33, 11.48, 16.20
    k_bol = a * (1.0 + (log_l_sol / b) ** c)
    return log_l_bol - jnp.log10(jnp.maximum(k_bol, 1.0))


def compute_l_x_agn(l_bol_agn_erg: jnp.ndarray) -> jnp.ndarray:
    """AGN X-ray luminosity (2–10 keV) from bolometric luminosity.

    Uses the Duras et al. (2020) bolometric correction:

    .. math::

        k_{\\rm bol} = a \\left[1 + \\left(
            \\frac{\\log L_{\\rm bol} / L_\\odot}{b}
        \\right)^c \\right]

    with a=15.33, b=11.48, c=16.20 for the 2–10 keV band.

    Parameters
    ----------
    l_bol_agn_erg : float
        AGN bolometric luminosity in erg/s.

    Returns
    -------
    float
        AGN 2–10 keV luminosity in erg/s.
    """
    log_l_sol = jnp.log10(jnp.maximum(l_bol_agn_erg, _FLOOR()) / L_SUN)
    # Duras+2020 Eq. 6, Table 2 (2-10 keV)
    a, b, c = 15.33, 11.48, 16.20
    k_bol = a * (1.0 + (log_l_sol / b) ** c)
    return l_bol_agn_erg / jnp.maximum(k_bol, 1.0)


# ── Ionizing photon budget ────────────────────────────────────────


def compute_ionizing_efficiency(q_h: jnp.ndarray, l_uv_erg: jnp.ndarray) -> jnp.ndarray:
    """Ionizing photon production efficiency ξ_ion.

    Key parameter for cosmic reionization studies:

    .. math::

        \\xi_{\\rm ion} = Q_H / L_{\\rm UV}

    where L_UV is the monochromatic UV luminosity density at 1500 Å
    in erg/s/Hz.

    Parameters
    ----------
    q_h : float
        Ionizing photon production rate (photons/s).
    l_uv_erg : float
        UV luminosity νL_ν at 1600 Å in erg/s, or L_ν in erg/s/Hz.
        Convention varies: typically expressed as
        ``log10(ξ_ion / Hz erg^-1)``.

    Returns
    -------
    float
        log10(ξ_ion) in Hz/erg.
    """
    return jnp.log10(jnp.maximum(q_h, _FLOOR()) / jnp.maximum(l_uv_erg, _FLOOR()))
