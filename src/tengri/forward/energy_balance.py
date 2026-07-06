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

import jax.numpy as jnp


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
        Non-finite integrals (e.g. Inf·0 artifacts from extreme-metallicity
        SSP fluxes, BUG-NSS-02 era) are clamped to 0.0 — the guard the
        retired compositional kernel carried; identity for finite inputs.

    Notes
    -----
    **JIT-compatible**: yes — pure ``jnp``; ``lyman_cutoff_aa`` is a static
    Python value, so the mask branch resolves at trace time. Safe under
    ``grad`` and ``vmap``.

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
    absorbed_lnu = sed_intrinsic - sed_attenuated
    if lyman_cutoff_aa is not None:
        absorbed_lnu = jnp.where(wave >= lyman_cutoff_aa, absorbed_lnu, 0.0)
    signed = jnp.trapezoid(absorbed_lnu, nu)
    return jnp.where(jnp.isfinite(signed), signed, 0.0)
