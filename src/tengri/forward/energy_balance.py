# SPDX-License-Identifier: BSD-3-Clause
r"""Canonical dust energy-balance integral (``L_absorbed``).

Single source of truth for the bolometric absorbed luminosity that feeds
dust IR re-emission (#922). Every exact-path computation of ``L_absorbed``
goes through :func:`bolometric_absorbed`; the build-time LUT in
:mod:`tengri.components.dust.energy_balance_precompute` is the precomputed
factorization of the *same* integral and must agree with it.

Physics convention: Lyman-continuum photons (:math:`\lambda < 912` Å) ionize
hydrogen — their energy re-emerges as nebular line and continuum emission,
not as dust heating — so they are excluded from the energy-balance integral,
matching CIGALE [1]_.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _absorbed_integrand(
    sed_intrinsic: jnp.ndarray,
    sed_attenuated: jnp.ndarray,
    wave: jnp.ndarray,
    lyman_cutoff_aa: float | None,
) -> jnp.ndarray:
    """LyC-masked absorbed integrand :math:`L_\\nu^{\\rm intr} - L_\\nu^{\\rm att}`."""
    absorbed_lnu = sed_intrinsic - sed_attenuated
    if lyman_cutoff_aa is not None:
        absorbed_lnu = jnp.where(wave >= lyman_cutoff_aa, absorbed_lnu, 0.0)
    return absorbed_lnu


def _peak_factored_trapezoid(
    integrand: jnp.ndarray, nu: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Integrate ``integrand/peak`` over ``nu``, returning the factored pieces.

    The absorbed luminosity is a product of two individually representable
    factors — an integrand of ~1e28 erg/s/Hz and a frequency span of ~1e15 Hz —
    whose product (~1e43 erg/s) exceeds the float32 ceiling of 3.4e38. Dividing
    the integrand by its own peak makes the reduction O(1e15), so no
    intermediate leaves float32 range; the caller re-applies ``peak``, in log
    space where it must.

    Returns
    -------
    signed_norm : ndarray, shape ()
        ``trapezoid(integrand / peak, nu)`` — signed, follows grid orientation.
    peak : ndarray, shape ()
        The factored-out scale (1.0 when the integrand is zero or non-finite).
    ok : ndarray, shape (), bool
        False when the integrand is all-zero or genuinely non-finite; callers
        map it to the zero/``-inf`` result rather than propagating NaN.
    """
    # stop_gradient: pure factorization constant (#1436). The caller re-applies this
    # peak to signed_norm, so the product is peak-independent and the peak's
    # derivative is analytically zero. Cancels in float64, not in float32.
    peak = jax.lax.stop_gradient(jnp.max(jnp.abs(integrand), initial=0.0))
    usable = jnp.isfinite(peak) & (peak > 0)
    safe_peak = jnp.where(usable, peak, 1.0)
    signed_norm = jnp.trapezoid(integrand / safe_peak, nu)
    return signed_norm, safe_peak, usable & jnp.isfinite(signed_norm)


def bolometric_absorbed_log10(
    sed_intrinsic: jnp.ndarray,
    sed_attenuated: jnp.ndarray,
    nu: jnp.ndarray,
    *,
    wave: jnp.ndarray,
    lyman_cutoff_aa: float | None = 912.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""log10 of the absorbed bolometric luminosity — the float32-safe contract.

    .. math::

        \log_{10} L_{\rm abs} = \log_{10} \left| \int_{\lambda \ge
            \lambda_{\rm LyC}} \left[ L_\nu^{\rm intr}(\lambda) -
            L_\nu^{\rm att}(\lambda) \right] d\nu \right|

    where :math:`L_\nu^{\rm intr}` is the intrinsic (unattenuated) SED
    [erg/s/Hz], :math:`L_\nu^{\rm att}` the dust-attenuated SED [erg/s/Hz],
    and :math:`\lambda_{\rm LyC}` the Lyman-continuum cutoff [Angstrom].

    Same integral, same LyC convention, and the same guard semantics as
    :func:`bolometric_absorbed` — only the output representation differs.
    Magnitude and sign are returned separately because that *is* what a
    signed quantity looks like in log space; callers that only need the
    energy (nearly all of them — the linear form's sign merely tracks grid
    orientation) discard the sign, while callers combining two absorbed
    terms need it to reproduce ``|a + b|`` rather than ``|a| + |b|``.

    Parameters
    ----------
    sed_intrinsic : array_like, shape (n_wave,)
        Intrinsic SED before dust attenuation [erg/s/Hz].
    sed_attenuated : array_like, shape (n_wave,)
        Dust-attenuated SED [erg/s/Hz].
    nu : array_like, shape (n_wave,)
        Frequency grid corresponding to ``wave`` [Hz].
    wave : array_like, shape (n_wave,)
        Wavelength grid [Angstrom]; used only for the Lyman-continuum mask.
    lyman_cutoff_aa : float or None, optional
        Lyman-continuum cutoff [Angstrom]. ``None`` disables the mask.
        Default 912.0.

    Returns
    -------
    log_magnitude : ndarray, shape ()
        :math:`\log_{10}(|L_{\rm abs}| / (\mathrm{erg/s}))` [dex], or
        ``-inf`` when nothing is absorbed (which powers back to exactly 0.0)
        or when the inputs are genuinely non-finite.
    sign : ndarray, shape ()
        Sign of the signed integral, for combining terms via
        :func:`tengri.utils.scale.log10_add`. 0.0 when nothing is absorbed.

    Notes
    -----
    **JIT-compatible**: yes — pure ``jnp``; ``lyman_cutoff_aa`` is a static
    Python value. Safe under ``grad`` and ``vmap``: the zero case takes the
    where-dummy path, so no NaN reaches the backward pass.

    The ``peak`` factored out of the integrand cancels analytically between
    the two log terms, so the gradient is that of the unfactored integral.

    Absorbed luminosities are ~1e43 erg/s — six decades past the float32
    ceiling — so this log form, not :func:`bolometric_absorbed`, is what a
    pure-float32 (JAX-Metal) forward pass must consume (#1206).
    """
    integrand = _absorbed_integrand(sed_intrinsic, sed_attenuated, wave, lyman_cutoff_aa)
    signed_norm, peak, ok = _peak_factored_trapezoid(integrand, nu)
    magnitude = jnp.abs(signed_norm)
    positive = ok & (magnitude > 0)
    safe = jnp.where(positive, magnitude, 1.0)
    log_magnitude = jnp.where(positive, jnp.log10(safe) + jnp.log10(peak), -jnp.inf)
    return log_magnitude, jnp.where(positive, jnp.sign(signed_norm), 0.0)


def bolometric_absorbed(
    sed_intrinsic: jnp.ndarray,
    sed_attenuated: jnp.ndarray,
    nu: jnp.ndarray,
    *,
    wave: jnp.ndarray,
    lyman_cutoff_aa: float | None = 912.0,
) -> jnp.ndarray:
    r"""Signed bolometric luminosity absorbed by dust, LyC-masked.

    .. math::

        L_{\rm abs} = \int_{\lambda \ge \lambda_{\rm LyC}}
            \left[ L_\nu^{\rm intr}(\lambda) - L_\nu^{\rm att}(\lambda) \right]
            d\nu

    where :math:`L_\nu^{\rm intr}` is the intrinsic (unattenuated) SED
    [erg/s/Hz], :math:`L_\nu^{\rm att}` the dust-attenuated SED [erg/s/Hz],
    and :math:`\lambda_{\rm LyC}` the Lyman-continuum cutoff [Angstrom].

    Parameters
    ----------
    sed_intrinsic : array_like, shape (n_wave,)
        Intrinsic SED before dust attenuation [erg/s/Hz].
    sed_attenuated : array_like, shape (n_wave,)
        Dust-attenuated SED [erg/s/Hz].
    nu : array_like, shape (n_wave,)
        Frequency grid corresponding to ``wave`` [Hz]. Passed to
        ``jnp.trapezoid`` as-is — no sorting is applied, so the sign of the
        result follows the grid orientation (descending ``nu`` for ascending
        ``wave`` gives a negative integral for net absorption).
    wave : array_like, shape (n_wave,)
        Wavelength grid [Angstrom]; used only for the Lyman-continuum mask.
    lyman_cutoff_aa : float or None, optional
        Lyman-continuum cutoff [Angstrom]; energy absorbed at
        ``wave < lyman_cutoff_aa`` is excluded (those photons ionize H, they
        do not heat dust). ``None`` disables the mask (integrate the full
        grid). Default 912.0.

    Returns
    -------
    ndarray, shape ()
        Signed absorbed bolometric luminosity [erg/s]. Callers apply
        ``jnp.abs`` (sign robustness against grid orientation) and any
        energy-balance relaxation factor (``dust_eta_balance``) themselves.
        Non-finite *inputs* (e.g. Inf·0 artifacts from extreme-metallicity
        SSP fluxes, BUG-NSS-02 era) are clamped to 0.0 — the guard the
        retired compositional kernel carried; identity for finite inputs.

    Notes
    -----
    **JIT-compatible**: yes — pure ``jnp``; ``lyman_cutoff_aa`` is a static
    Python value, so the mask branch resolves at trace time. Safe under
    ``grad`` and ``vmap``.

    **Not float32-representable.** Absorbed luminosities are ~1e43 erg/s,
    six decades past the float32 ceiling, so this returns ``inf`` under pure
    float32 no matter how the reduction is arranged. Use
    :func:`bolometric_absorbed_log10` there (#1206). The integrand is
    peak-factored so that overflow is confined to that final re-scaling:
    previously the reduction itself overflowed and the non-finite guard
    turned the ``inf`` into **0.0**, silently switching dust IR emission off
    rather than failing loudly.

    The fast-path LUT (:func:`tengri.components.dust.
    energy_balance_precompute.lut_l_absorbed_stellar`) is the precomputed
    factorization of this integral over the SSP grid; the two must agree
    (contract test: ``tests/contract/test_energy_balance_lut.py``).

    Cross-code conventions: CIGALE zeroes its attenuation curves at
    λ ≤ 91.2 nm and Bagpipes masks the ionizing continuum via ``fesc`` —
    both exclude LyC from dust heating, as here. FSPS does *not* mask the
    LyC, so ``L_dust`` comparisons against FSPS/Prospector carry this
    convention difference. The mask also protects the integral from
    attenuation laws whose far-UV extrapolation amplifies (k(λ) < 0 below
    the law's calibrated range, e.g. Calzetti).

    References
    ----------
    .. [1] Boquien, M., et al. 2019, A&A, 622, A103.
           https://doi.org/10.1051/0004-6361/201834156

    """
    integrand = _absorbed_integrand(sed_intrinsic, sed_attenuated, wave, lyman_cutoff_aa)
    signed_norm, peak, ok = _peak_factored_trapezoid(integrand, nu)
    return jnp.where(ok, signed_norm * peak, 0.0)
