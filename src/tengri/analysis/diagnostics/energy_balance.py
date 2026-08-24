# SPDX-License-Identifier: BSD-3-Clause
r"""Dust energy-balance diagnostic.

Self-contained utility for verifying that a model prediction conserves
energy across the dust absorption / re-emission step:

.. math::

    \int_{\rm UV-NIR} \!\!\bigl(L_\nu^{\rm unatten} - L_\nu^{\rm atten}\bigr) \, d\nu
    \;\stackrel{?}{\approx}\;
    \int_{\rm IR} \!\! L_\nu^{\rm dust\,emission} \, d\nu

Useful for sanity-checking a fit's posterior predictions and for
spotting bugs in the dust attenuation / emission split. The check is
agnostic of the underlying dust law and the SED-modeling code that
produced the inputs.

References
----------

- Charlot, S. & Fall, S. M., 2000, ApJ, 539, 718 (energy balance ansatz).
- da Cunha, E. et al., 2008, MNRAS, 388, 1595 (MAGPHYS).

"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

__all__ = ["dust_energy_balance", "integrate_lnu_over_band"]

# Speed of light, Å/s.
from tengri.utils.physics_constants import C_AA as _C_AA_S

# Default integration bands (rest-frame Å).
_UV_NIR_LO = 912.0  # Lyman limit
_UV_NIR_HI = 3.0e4  # ~3 μm
_IR_LO = 3.0e4  # ~3 μm (paired with UV-NIR upper edge)
_IR_HI = 3.0e7  # ~3 mm


def integrate_lnu_over_band(
    wavelength_aa: jnp.ndarray,
    l_nu: jnp.ndarray,
    lambda_lo_aa: float,
    lambda_hi_aa: float,
) -> jnp.ndarray:
    r"""Integrate :math:`L_\nu` over a wavelength band.

    Equivalent to :math:`\int F_\lambda \, d\lambda` over the same band
    (since :math:`L_\nu \, d\nu = F_\lambda \, d\lambda` up to sign).

    Parameters
    ----------
    wavelength_aa: array_like, shape (n_wave,)
        Wavelength grid in ascending order. [Å]
    l_nu: array_like, shape (n_wave,)
        Spectral luminosity density. [erg/s/Hz] or [Lsun/Hz]
    lambda_lo_aa: float
        Lower band edge. [Å]
    lambda_hi_aa: float
        Upper band edge. [Å]

    Returns
    -------
    float
        Integrated luminosity over the band, in the same units as
        :math:`L_\nu \cdot \nu` (so [erg/s] or [Lsun]).

    Notes
    -----
    **JIT-compatible**: yes.

    Bands that fall entirely outside the supplied grid return 0; bands
    that partially overlap return only the overlapping portion. The
    integration is performed in :math:`\lambda` space using
    :math:`\int F_\lambda \, d\lambda = \int (c/\lambda^2)\, L_\nu \, d\lambda`,
    which avoids the irregular spacing introduced by switching to
    :math:`\nu = c/\lambda`.
    """
    wave = jnp.asarray(wavelength_aa)
    l_nu = jnp.asarray(l_nu)
    f_lambda = l_nu * (_C_AA_S / wave**2)  # erg/s/Å (or Lsun/Å)
    in_band = (wave >= lambda_lo_aa) & (wave <= lambda_hi_aa)
    integrand = jnp.where(in_band, f_lambda, 0.0)
    return jnp.trapezoid(integrand, wave)


def dust_energy_balance(
    wavelength_aa: jnp.ndarray,
    l_nu_unattenuated: jnp.ndarray,
    l_nu_attenuated: jnp.ndarray,
    l_nu_dust_emission: jnp.ndarray,
    tol: float = 0.05,
    uv_nir_band_aa: tuple[float, float] = (_UV_NIR_LO, _UV_NIR_HI),
    ir_band_aa: tuple[float, float] = (_IR_LO, _IR_HI),
) -> dict:
    r"""Check whether absorbed and re-emitted dust energy balance.

    Integrates :math:`(L_\nu^{\rm unatten} - L_\nu^{\rm atten})` over
    the UV-NIR band and :math:`L_\nu^{\rm dust\,emission}` over the IR
    band, and reports the ratio.

    Parameters
    ----------
    wavelength_aa: array_like, shape (n_wave,)
        Rest-frame wavelength grid (ascending). [Å]
    l_nu_unattenuated: array_like, shape (n_wave,)
        Stellar (+ AGN) :math:`L_\nu` *before* dust attenuation. [erg/s/Hz]
    l_nu_attenuated: array_like, shape (n_wave,)
        Stellar (+ AGN) :math:`L_\nu` *after* dust attenuation. [erg/s/Hz]
    l_nu_dust_emission: array_like, shape (n_wave,)
        Dust thermal :math:`L_\nu` re-emission. [erg/s/Hz]
    tol: float, optional
        Fractional tolerance on the absorbed/emitted ratio for the
        ``balanced`` flag. Default 0.05 (5%).
    uv_nir_band_aa: tuple of (float, float), optional
        Band over which absorption is integrated. Default (912, 3e4) Å.
    ir_band_aa: tuple of (float, float), optional
        Band over which re-emission is integrated. Default (3e4, 3e7) Å.

    Returns
    -------
    dict
        Keys:

        - ``absorbed``: float: :math:`\int (L_\nu^{\rm unatten} -
          L_\nu^{\rm atten}) d\nu` over the UV-NIR band. [erg/s]
        - ``emitted``: float; :math:`\int L_\nu^{\rm dust} d\nu` over
          the IR band. [erg/s]
        - ``ratio``: float: ``emitted / absorbed`` (``inf`` if
          ``absorbed`` is zero, ``nan`` if both are zero).
        - ``balanced``: bool; ``abs(ratio - 1) <= tol`` and absorbed > 0.

    Notes
    -----
    **JIT-compatible**: yes for the underlying integrals; the dict
    return is constructed at trace time.

    The default UV-NIR/IR band split at 3 μm is the conventional
    boundary used by MAGPHYS and CIGALE. For galaxies with significant
    PAH or torus emission the user should widen the IR band.

    Examples
    --------
    >>> result = dust_energy_balance(wave, l_unatten, l_atten, l_dust)
    >>> if not result["balanced"]:
    ...     print(f"WARN: dust ratio {result['ratio']:.2f}, expected ~1")
    """
    wave = jnp.asarray(wavelength_aa)
    diff = jnp.asarray(l_nu_unattenuated) - jnp.asarray(l_nu_attenuated)
    absorbed = float(integrate_lnu_over_band(wave, diff, *uv_nir_band_aa))
    emitted = float(integrate_lnu_over_band(wave, l_nu_dust_emission, *ir_band_aa))
    if absorbed == 0.0 and emitted == 0.0:
        ratio: float = float("nan")
    elif absorbed == 0.0:
        ratio = float("inf") if emitted > 0 else 0.0
    else:
        ratio = emitted / absorbed
    balanced = bool(np.isfinite(ratio) and absorbed > 0.0 and abs(ratio - 1.0) <= tol)
    return {
        "absorbed": absorbed,
        "emitted": emitted,
        "ratio": ratio,
        "balanced": balanced,
    }
