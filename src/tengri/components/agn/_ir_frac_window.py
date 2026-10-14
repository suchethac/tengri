# SPDX-License-Identifier: BSD-3-Clause
"""CIGALE's ``lambda_fracAGN`` window for ``agn_ir_frac`` (#2662).

CIGALE ``skirtor2016`` sets the AGN dust power from the fraction of the light in a
wavelength window (``pcigale/sed_modules/skirtor2016.py:473-498``)::

    scale     = int_w L_host dlam / AGNlumin        (window)
              = L_host(lam) / AGNlumin(lam)          (lam_min == lam_max, monochromatic)
    AGNlumin  = int_w L_agn dlam / D_unit            (per unit AGN dust luminosity)
    agn_power = scale * f / (1 - f)

so ``f`` is the AGN fraction of the total light in the window. The total-dust case
``lambda_fracAGN="0/0"`` is ``scale = L_host`` and is not handled here.

The functions below return ``log10(agn_power)`` in erg/s. The ratio is formed in log
space because the window luminosities of a host near 1e43 erg/s do not fit float32.

Tier: Tier 2 (scientific helper, ``components/``).

Notes
-----
JIT/grad/vmap compatible. The window integral is evaluated on the caller's rest grid with
the edges included by linear interpolation (``np.interp``) and the trapezoid rule, the same
node set as pcigale (``hstack([lam_min, grid_in_window, lam_max])``). It is written as a
clip of the grid, so no boolean indexing enters the trace and a zero-length segment
contributes nothing.
"""

from __future__ import annotations

import jax.numpy as jnp

from tengri.utils.scale import representable_floor

#: Speed of light [Angstrom/s]; enters once, through the dust luminosity ``D``.
_C_AA_PER_S = 2.99792458e18

#: Floor for a luminosity integral before its logarithm (dtype-safe).
_FLOOR = representable_floor(1.0e-30)


def _log10_safe(value):
    return jnp.log10(jnp.maximum(jnp.asarray(value), _FLOOR))


def _window_q(wave, lnu, lam_min, lam_max):
    """``int_w (L_nu / lam^2) dlam`` over ``[lam_min, lam_max]``; the L_lambda integral / c.

    Parameters
    ----------
    wave : array_like, shape (n_wave,)
        Rest wavelength [Angstrom], ascending.
    lnu : array_like, shape (n_wave,)
        Spectral luminosity [erg/s/Hz] on ``wave``.
    lam_min, lam_max : float
        Window edges [Angstrom].

    Returns
    -------
    ndarray, scalar
        The window integral of ``L_nu / lam^2`` [erg/s/Hz/Angstrom].
    """
    lam_c = jnp.clip(wave, lam_min, lam_max)
    f_c = jnp.interp(lam_c, wave, lnu / wave**2)
    return jnp.trapezoid(f_c, lam_c)


def _mono_ratio(wave, host_lnu, agn_lnu, lam):
    """``L_host(lam) / L_agn(lam)`` by linear interpolation on the rest grid."""
    h = jnp.interp(lam, wave, host_lnu / wave**2)
    w = jnp.interp(lam, wave, agn_lnu / wave**2)
    return h / w


def log10_agn_power_from_window(
    wave, host_lnu, agn_lnu, dust_lnu, frac, lam_min: float, lam_max: float
):
    """``log10`` of CIGALE's ``agn_power`` for a wavelength window (erg/s).

    Parameters
    ----------
    wave : array_like, shape (n_wave,)
        Rest wavelength [Angstrom], ascending.
    host_lnu : array_like, shape (n_wave,)
        Host SED without the AGN (stellar, nebular, dust emission) [erg/s/Hz].
    agn_lnu : array_like, shape (n_wave,)
        AGN SED per unit dust luminosity, or at any finite normalization: only the
        ratio ``dust / agn`` enters [erg/s/Hz].
    dust_lnu : array_like, shape (n_wave,)
        AGN dust (torus plus polar) SED at the same normalization as ``agn_lnu``
        [erg/s/Hz].
    frac : float or array_like
        AGN fraction ``f`` of the light in the window, ``0 < f < 1``.
    lam_min, lam_max : float
        Window edges [Angstrom]. ``lam_min == lam_max != 0`` is monochromatic.

    Returns
    -------
    ndarray, scalar
        ``log10(agn_power / (erg/s))``.

    Notes
    -----
    Implements ``agn_power = f/(1-f) * H * D / W`` with ``H``, ``W`` the host and AGN
    window luminosities and ``D`` the AGN dust luminosity, the pcigale definition at
    ``skirtor2016.py:473-498``. The monochromatic case replaces ``H`` and ``W`` by the
    L_lambda values at the wavelength. JIT/grad/vmap compatible; ``lam_min`` and
    ``lam_max`` are static Python floats.
    """
    frac = jnp.asarray(frac)
    log_odds = _log10_safe(frac) - _log10_safe(1.0 - frac)
    # D = int L_nu dnu = c * int (L_nu / lam^2) dlam over the whole grid.
    log_dust = _log10_safe(_C_AA_PER_S) + _log10_safe(jnp.trapezoid(dust_lnu / wave**2, wave))
    if lam_min == lam_max:
        log_ratio = _log10_safe(_mono_ratio(wave, host_lnu, agn_lnu, lam_min))
    else:
        log_ratio = _log10_safe(_window_q(wave, host_lnu, lam_min, lam_max)) - _log10_safe(
            _window_q(wave, agn_lnu, lam_min, lam_max)
        )
    return log_odds + log_ratio + log_dust


def agn_power_from_window(wave, host_lnu, agn_lnu, dust_lnu, frac, lam_min, lam_max):
    """``agn_power`` [erg/s] for a wavelength window; the linear form of the log helper.

    Parameters
    ----------
    See :func:`log10_agn_power_from_window`.

    Returns
    -------
    ndarray, scalar
        ``agn_power`` [erg/s].

    Notes
    -----
    Exponentiates the log form, so it is only as accurate as float64 allows. The component
    uses the log form directly. JIT/grad/vmap compatible.
    """
    log_power = log10_agn_power_from_window(
        wave, host_lnu, agn_lnu, dust_lnu, frac, lam_min, lam_max
    )
    return jnp.power(10.0, log_power)
