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

    Attributes
    ----------
    wavelength : jnp.ndarray, shape (n_wave,)
        Wavelength grid in Angstrom. Rest-frame for ``predict_rest_sed()``,
        observed-frame for ``predict_obs_sed()``.
    sed : jnp.ndarray, shape (n_wave,)
        Spectral luminosity density in erg/s/Hz.
    """

    wavelength: jnp.ndarray
    sed: jnp.ndarray
