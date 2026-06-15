# SPDX-License-Identifier: BSD-3-Clause
"""Shared constants for disc block modules."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from tengri.utils.physics_constants import L_SUN

#: Speed of light in Å × Hz, for L_ν → L_λ conversion.
_C_AA_PER_S: float = 2.99792458e18

#: Solar luminosity [erg/s] — IAU 2015 nominal (already in erg/s, see
#: :data:`tengri.utils.physics_constants.L_SUN`).
_L_SUN_ERG: float = L_SUN


def _cigale_disc_lambda(
    wavelength_aa: Array,
    agn_log_lbol: float,
    spectrum_per_nm_fn,
    delta: float,
) -> Array:
    r"""Common L_λ scaffold for CIGALE piecewise-power-law disc blocks.

    Parameters
    ----------
    wavelength_aa : array_like, shape (n_wave,)
        Rest-frame wavelength [Å].
    agn_log_lbol : float
        :math:`\log_{10}(L_{\rm bol}/L_\odot)`.
    spectrum_per_nm_fn : callable
        One of :func:`skirtor_disk_spectrum`,
        :func:`schartmann2005_disk_spectrum`, or
        :func:`adaf_disk_spectrum`. Takes ``(wave_nm, delta)`` and returns
        a dimensionless spectrum normalised so its integral over the
        nm axis equals one.
    delta : float
        CIGALE ``delta`` slope/blend modulator.

    Returns
    -------
    L_lambda : ndarray, shape (n_wave,)
        :math:`L_\lambda` [erg/s/Å].
    """
    wave_aa = jnp.asarray(wavelength_aa)
    wave_nm = wave_aa / 10.0
    # Unit-normalised spectrum on the nm grid (integral over nm = 1).
    s_per_nm = spectrum_per_nm_fn(wave_nm, delta=delta)
    # Convert to a unit-normalised density on the Å grid (÷10).
    s_per_aa = s_per_nm / 10.0
    L_bol_erg = (10.0**agn_log_lbol) * _L_SUN_ERG
    return s_per_aa * L_bol_erg
