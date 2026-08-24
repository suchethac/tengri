# SPDX-License-Identifier: BSD-3-Clause
"""SED result container.

Provides the :class:`SEDResult` NamedTuple returned by
:meth:`~tengri.forward.sed_model.SEDModel.predict_rest_sed` and
:meth:`~tengri.forward.sed_model.SEDModel.predict_obs_sed`.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


class SEDResult(NamedTuple):
    """Rest-frame or observed-frame SED with its wavelength grid.

    Container for SED predictions returned by
    :meth:`~tengri.forward.sed_model.SEDModel.predict_rest_sed` and
    :meth:`~tengri.forward.sed_model.SEDModel.predict_obs_sed`.

    Parameters
    ----------
    wavelength: ndarray, shape (n_wave,)
        Wavelength grid. [Angstrom]
    sed: ndarray, shape (n_wave,)
        Spectral luminosity density for rest-frame SED or flux density for
        observed-frame. [erg/s/Hz] for rest-frame, [erg/s/cm^2/Hz] for
        observed-frame.

    Returns
    -------
    SEDResult
        Named tuple with wavelength and SED arrays.

    Attributes
    ----------
    wavelength: ndarray, shape (n_wave,)
        Wavelength grid. [Angstrom]
    sed: ndarray, shape (n_wave,)
        Spectral luminosity density for rest-frame SED or flux density for
        observed-frame. [erg/s/Hz] for rest-frame, [erg/s/cm^2/Hz] for
        observed-frame.

    Notes
    -----
    This is a NamedTuple (JAX pytree), making it compatible with ``jax.jit``
    and ``jax.vmap``. Fields are JAX arrays.
    """

    wavelength: jnp.ndarray
    sed: jnp.ndarray
