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
        Wavelength grid [Angstrom]. Rest-frame for ``predict_rest_sed()``,
        observed-frame for ``predict_obs_sed()``.
    sed : jnp.ndarray, shape (n_wave,)
        Spectral luminosity density [erg/s/Hz] for rest-frame SED,
        or flux density [erg/s/cm²/Hz] for observed-frame.

    Returns
    -------
    This is a NamedTuple (JAX pytree) returned by
    :meth:`~tengri.forward.sed_model.SEDModel.predict_rest_sed` and
    :meth:`~tengri.forward.sed_model.SEDModel.predict_obs_sed`.

    Notes
    -----
    JAX-compatible container. Fields are JAX arrays compatible with
    ``jax.jit`` and ``jax.vmap``. Returned by prediction methods.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from tengri import SEDResult
    >>> wave = jnp.linspace(1000.0, 30000.0, 500)
    >>> sed = jnp.zeros(500)
    >>> result = SEDResult(wavelength=wave, sed=sed)
    >>> result.sed.shape
    (500,)
    """

    wavelength: jnp.ndarray
    sed: jnp.ndarray
