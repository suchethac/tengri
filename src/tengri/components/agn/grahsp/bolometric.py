# SPDX-License-Identifier: BSD-3-Clause
"""GRAHSP bolometric AGN luminosities and AGN fractions.

Implements §2.1.4 of Buchner+ 2024 (arXiv:2405.19297):

- ``lumBolBBB``: trapezoidal integral of the AGN UV-optical components
  (BBB + lines + FeII), restricted to :math:`\\lambda \\geq 91.2\\,\\mathrm{nm}`
  (the Lyman limit). This is the rest-frame, isotropic, intrinsic luminosity
  *before* attenuation.
- ``lumBolTOR``: integral of the torus component over its full wavelength
  support.
- ``ratioTORBBB``: ``lumBolTOR / lumBolBBB``.
- ``fracAGNDale``: AGN luminosity fraction in the 5-20 um range relative to
  the *total* (AGN + galaxy) luminosity in the same band: following
  Dale+ 2014 ApJ 784, 83.
- ``fracAGNTOR``: ``lumBolTOR / L_galaxy_bolometric``.

References
----------
.. [1] Buchner, J. et al. 2024, arXiv:2405.19297, §2.1.4.
.. [2] Dale, D. A. et al. 2014, ApJ, 784, 83.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

__all__ = [
    "LYMAN_LIMIT_NM",
    "agn_fraction_dale",
    "bolometric_luminosity_bbb",
    "bolometric_luminosity_torus",
    "frac_agn_tor",
    "ratio_tor_bbb",
]

LYMAN_LIMIT_NM: float = 91.2
"""Lyman limit at 91.2 nm: lower bound for ``lumBolBBB`` integration."""


def _trapz_above(
    wave_nm: Array, L_lambda: Array, lower_nm: float, upper_nm: float | None = None
) -> Array:
    """Trapezoidal integral of L_lambda over [lower, upper] (nm, inclusive)."""
    mask = wave_nm >= lower_nm
    if upper_nm is not None:
        mask = mask & (wave_nm <= upper_nm)
    # jnp.trapezoid is JAX-native (>=0.4.26); fall back to trapz on older.
    integrand = jnp.where(mask, L_lambda, 0.0)
    try:
        return jnp.trapezoid(integrand, wave_nm)
    except AttributeError:  # pragma: no cover: older jax
        return jnp.trapz(integrand, wave_nm)


def bolometric_luminosity_bbb(
    wave_nm: Array,
    L_lambda_bbb_total: Array,
) -> Array:
    r"""Integrate the AGN BBB-side spectrum above the Lyman limit.

    Parameters
    ----------
    wave_nm : array_like, shape (n_wave,)
        Wavelength grid [nm].
    L_lambda_bbb_total : array_like, shape (n_wave,)
        Sum of BBB + lines + FeII luminosity densities [erg/s/nm].

    Returns
    -------
    L_bol_BBB : float
        Integrated luminosity [erg/s] above 91.2 nm.

    Notes
    -----
    JIT-compatible. The far-UV/X-ray contribution below 91.2 nm is
    deliberately excluded; see paper §2.1.4: that band is rarely observed
    and a model-dependent correction is left to the user.
    """
    return _trapz_above(jnp.asarray(wave_nm), jnp.asarray(L_lambda_bbb_total), LYMAN_LIMIT_NM)


def bolometric_luminosity_torus(
    wave_nm: Array,
    L_lambda_torus: Array,
) -> Array:
    r"""Total torus bolometric luminosity (no wavelength restriction).

    Parameters
    ----------
    wave_nm : array_like, shape (n_wave,)
    L_lambda_torus : array_like, shape (n_wave,)
        Torus L_lambda [erg/s/nm] (continuum + Si feature).

    Returns
    -------
    L_bol_TOR : float
        Integrated luminosity [erg/s] over the entire wavelength range.
    """
    return jnp.trapezoid(jnp.asarray(L_lambda_torus), jnp.asarray(wave_nm))


def agn_fraction_dale(
    wave_nm: Array,
    L_lambda_agn_total: Array,
    L_lambda_gal_total: Array,
    lower_um: float = 5.0,
    upper_um: float = 20.0,
) -> Array:
    r"""AGN luminosity fraction in the 5-20 um band (Dale+ 2014 definition).

    .. math::

       f_{\rm AGN, Dale} = \frac{\int_{5\,\mu m}^{20\,\mu m} L_\lambda^{\rm AGN}\,d\lambda}
                                {\int_{5\,\mu m}^{20\,\mu m}
                                 (L_\lambda^{\rm AGN} + L_\lambda^{\rm gal})\,d\lambda}.

    Parameters
    ----------
    wave_nm : array_like, shape (n_wave,)
    L_lambda_agn_total, L_lambda_gal_total : array_like, shape (n_wave,)
        AGN-side and galaxy-side luminosity densities [erg/s/nm].
    lower_um, upper_um : float, optional
        Integration band [um]. Defaults reproduce Dale+ 2014.

    Returns
    -------
    fraction : float
        Dimensionless. Returns 0 when the denominator is zero.

    Notes
    -----
    JIT-compatible.
    """
    lower_nm = lower_um * 1000.0
    upper_nm = upper_um * 1000.0
    L_agn = _trapz_above(jnp.asarray(wave_nm), jnp.asarray(L_lambda_agn_total), lower_nm, upper_nm)
    L_total = L_agn + _trapz_above(
        jnp.asarray(wave_nm), jnp.asarray(L_lambda_gal_total), lower_nm, upper_nm
    )
    return jnp.where(L_total > 0, L_agn / L_total, 0.0)


def ratio_tor_bbb(l_bol_torus: Array, l_bol_bbb: Array) -> Array:
    r"""Torus-to-BBB bolometric luminosity ratio (upstream ``ratioTORBBB``).

    .. math::

       R_{\rm TOR/BBB} = L_{\rm bol}^{\rm TOR} / L_{\rm bol}^{\rm BBB}.

    Parameters
    ----------
    l_bol_torus, l_bol_bbb : float
        Torus and BBB bolometric luminosities [erg/s] (see
        :func:`bolometric_luminosity_torus`, :func:`bolometric_luminosity_bbb`).

    Returns
    -------
    ratio : float
        Dimensionless; 0 when ``l_bol_bbb`` is non-positive.

    Notes
    -----
    JIT-compatible. Mirrors ``agn.ratioTORBBB`` published by upstream
    ``activatebol`` (Buchner+ 2024 §2.1.4).
    """
    l_bol_bbb = jnp.asarray(l_bol_bbb)
    return jnp.where(l_bol_bbb > 0, jnp.asarray(l_bol_torus) / l_bol_bbb, 0.0)


def frac_agn_tor(l_bol_torus: Array, l_gal_bol: Array) -> Array:
    r"""Torus AGN fraction relative to the galaxy bolometric luminosity.

    .. math::

       f_{\rm AGN, TOR} = \frac{L_{\rm bol}^{\rm TOR}}
                               {L_{\rm bol}^{\rm TOR} + L_{\rm gal}^{\rm bol}}.

    Parameters
    ----------
    l_bol_torus : float
        Torus bolometric luminosity [erg/s].
    l_gal_bol : float
        Galaxy bolometric luminosity [erg/s].

    Returns
    -------
    fraction : float
        Dimensionless in [0, 1]; 0 when the denominator is non-positive.

    Notes
    -----
    JIT-compatible. Mirrors ``agn.fracAGNTOR`` published by upstream
    ``activatebol`` (Buchner+ 2024 §2.1.4).
    """
    denom = jnp.asarray(l_bol_torus) + jnp.asarray(l_gal_bol)
    return jnp.where(denom > 0, jnp.asarray(l_bol_torus) / denom, 0.0)
