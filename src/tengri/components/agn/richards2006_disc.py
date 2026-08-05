# SPDX-License-Identifier: BSD-3-Clause
r"""Richards+2006 mean Type-1 quasar SED template.

Empirical big blue bump (BBB) from 259 Type-1 SDSS quasars
(Richards et al. 2006, ApJ 166, 470). The template covers
~30 Å (soft X-ray) through ~30 cm (radio); tengri loads it
without normalization and rescales at the bolometric anchor.

This is an alternative empirical disc choice alongside
``qsogen`` (Temple+2021, narrower wavelength range, different
power-law parameterization) and the physical multicolor /
Kubota-Done 2018 discs.

References
----------
.. [1] G. T. Richards, et al., "Spectral Energy Distributions and
   Multiwavelength Selection of Type 1 Quasars," ApJ, 166, 470 (2006).
   https://doi.org/10.1086/506525
"""

from __future__ import annotations

from importlib.resources import files

import jax.numpy as jnp
import numpy as np

__all__ = [
    "RICHARDS2006_NU_FNU",
    "RICHARDS2006_WAVE_AA",
    "richards2006",
    "richards2006_disc",
]


def _load_template() -> tuple[np.ndarray, np.ndarray]:
    """Load Richards+2006 (wavelength, nu·F_nu) tabulation at import time."""
    path = files("tengri.data.agn_bbb") / "richards2006.dat"
    with path.open("r") as fh:
        arr = np.loadtxt(fh)
    wave_aa = np.asarray(arr[:, 0], dtype=np.float64)
    nu_fnu = np.asarray(arr[:, 1], dtype=np.float64)
    return wave_aa, nu_fnu


RICHARDS2006_WAVE_AA, RICHARDS2006_NU_FNU = _load_template()
"""Tabulated Richards+2006 template, ascending in wavelength [Å]."""

# Pre-compute L_nu shape: nu·F_nu / nu = F_nu, then proportional to L_nu.
# We treat the shipped column as nu·F_nu (arbitrary scale) and divide by nu
# to get the F_nu shape, since SED-fitting outputs are normalized at the
# bolometric anchor downstream.
from tengri.utils.physics_constants import C_AA as _C_AA_PER_S

_RICHARDS2006_NU_HZ = _C_AA_PER_S / RICHARDS2006_WAVE_AA
_RICHARDS2006_LNU_SHAPE = RICHARDS2006_NU_FNU / _RICHARDS2006_NU_HZ
# Integrate L_nu shape over frequency for bolometric normalization
_idx_sort = jnp.argsort(_RICHARDS2006_NU_HZ)
_RICHARDS2006_BOL_INTEGRAL = float(
    jnp.trapezoid(
        jnp.asarray(_RICHARDS2006_LNU_SHAPE)[_idx_sort],
        jnp.asarray(_RICHARDS2006_NU_HZ)[_idx_sort],
    )
)

# L_sun in erg/s (IAU 2015)
from tengri.components.agn._params import DEFAULT_AGN_LOG_LBOL
from tengri.utils.physics_constants import L_SUN as _L_SUN_ERG_S


def richards2006_disc(
    wavelength: jnp.ndarray,
    log_lbol: float = DEFAULT_AGN_LOG_LBOL,
) -> jnp.ndarray:
    r"""Richards+2006 mean quasar SED at a chosen bolometric luminosity.

    The shipped template is integrated over frequency to give the
    arbitrary-unit bolometric value, then the spectrum is rescaled so that
    :math:`\int L_\nu \, d\nu = 10^{\log L_{\rm bol}} \cdot L_\odot`.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength grid [Å]. Values outside the template's
        coverage (30.5 Å — 3×10⁸ Å) yield 0.
    log_lbol : float, optional
        Bolometric luminosity in :math:`\log_{10}(L/L_\odot)`. Defaults to
        the declared ``agn_log_lbol`` default.

    Returns
    -------
    ndarray, shape (n_wave,)
        Spectral luminosity density :math:`L_\nu` [erg/s/Hz].

    Notes
    -----
    **JIT-compatible**: yes — pure ``jnp.interp`` + linear scale.

    This is an empirical SDSS composite; it carries no free spectral-shape
    parameters. To vary the UV slope, prefer the GRAHSP bending power-law
    (``AGN_MODELS["grahsp"]``). To vary the optical break, prefer QSOgen
    (``AGN_MODELS["qsogen"]``). To anchor to a physical disc, prefer
    ``multicolor_agn`` or ``kubota_done_full``.

    References
    ----------
    .. [1] G. T. Richards et al., ApJ, 166, 470 (2006).
       https://doi.org/10.1086/506525
    """
    # Look up L_nu shape at requested wavelengths (zero outside template range)
    lnu_shape = jnp.interp(
        wavelength,
        jnp.asarray(RICHARDS2006_WAVE_AA),
        jnp.asarray(_RICHARDS2006_LNU_SHAPE),
        left=0.0,
        right=0.0,
    )
    # Rescale to L_bol target (template integral is constant, computed at load)
    target_bol_erg_s = (10.0**log_lbol) * _L_SUN_ERG_S
    norm = target_bol_erg_s / _RICHARDS2006_BOL_INTEGRAL
    return lnu_shape * norm


# Deprecated: richards2006 is no longer registered in AGN_MODELS.
# Use composable AGN blocks instead: agn_disc_block="richards2006_disc".
# This function is retained for backward compatibility if imported directly.
def richards2006(
    wavelength: jnp.ndarray,
    agn_log_lbol: float = DEFAULT_AGN_LOG_LBOL,
    agn_lum_ratio: float = 1.0,
    **_kwargs,
) -> jnp.ndarray:
    """Richards+2006 BBB composite — registered model entry point.

    Thin wrapper around :func:`richards2006_disc` matching the
    AGN_MODELS registry signature::

        fn(wavelength, agn_log_lbol, **kwargs) -> L_nu

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength grid [Å].
    agn_log_lbol : float, optional
        Total AGN bolometric luminosity, :math:`\\log_{10}(L_\\odot)`.
        Defaults to the declared ``agn_log_lbol`` default.
    agn_lum_ratio : float, optional
        Fraction of bolometric luminosity emitted by this component.
        Default: 1.0.
    **_kwargs
        Additional keyword arguments (ignored — Richards+2006 is a fixed
        empirical template with no free shape parameters).

    Returns
    -------
    ndarray, shape (n_wave,)
        Spectral luminosity density [erg/s/Hz].
    """
    sed = richards2006_disc(wavelength, log_lbol=agn_log_lbol)
    return sed * agn_lum_ratio
