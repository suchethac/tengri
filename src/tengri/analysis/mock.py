# SPDX-License-Identifier: BSD-3-Clause
"""Mock galaxy generation utilities.

Standalone ``generate_mock`` function for quick mock photometry
generation. For the ``SEDModel``-based API, use ``SEDModel.mock()`` instead.
"""

import jax
import jax.numpy as jnp

# Re-export MockData so callers can import from one place
from tengri.forward.sed_model import MockData

__all__ = ["MockData", "MockDict", "generate_mock"]


class MockDict(dict):
    """A ``dict`` that also exposes its keys as attributes.

    ``generate_mock`` returns this container so that both mapping access
    (``mock["flux_obs"]``) and attribute access (``mock.flux_obs``) work. That
    matches the attribute surface of the :class:`MockData` object returned by
    :meth:`SEDModel.mock`, so notebook and user code written against either
    surface works against both; the recurring "dict vs object" footgun.

    It is a genuine ``dict`` (``isinstance(mock, dict)`` is ``True`` and every
    ``dict`` method is available), so existing key-based consumers are
    unaffected.

    Notes
    -----
    Registered as a JAX pytree that flattens identically to a plain ``dict``
    (sorted keys), so ``jax.tree_util`` operations over a ``generate_mock``
    result are unchanged.
    """

    def __getattr__(self, name: str):
        # __getattr__ only fires when normal attribute lookup fails, so dict
        # methods (keys/get/items/...) are never shadowed.
        try:
            return self[name]
        except KeyError:
            raise AttributeError(
                f"MockDict has no key {name!r} (available: {sorted(self)})"
            ) from None


jax.tree_util.register_pytree_node(
    MockDict,
    lambda d: (tuple(d[k] for k in sorted(d)), tuple(sorted(d))),
    lambda keys, values: MockDict(zip(keys, values)),
)


def generate_mock(model, params, key=None, snr=20.0, noise=None):
    """Generate mock galaxy photometry with optional Gaussian noise.

    Computes noiseless predicted photometry, then optionally realizes noise
    at a specified signal-to-noise ratio. Useful for testing data pipelines,
    validating inference, and parameter recovery studies.

    Parameters
    ----------
    model : object
        Any object with a ``predict_photometry(params)`` method that returns
        an array of flux densities.
    params : dict[str, ndarray]
        Model parameter values (typically sampled or optimized).
    key : jax.Array (PRNGKey), optional
        Random key for noise realization. If ``None``, only noiseless
        photometry is returned (no ``flux_obs`` key in output).
    snr : float, optional
        Signal-to-noise ratio (flux_true / noise_std). Default: 20.0. Ignored
        when ``noise`` is given.
    noise : array_like, optional
        Observed 1-sigma uncertainties [erg/s/cm²/Hz] to draw each band from
        (CIGALE's ``mock_flag`` (Boquien et al. 2019, A&A 622, A103):
        ``flux_obs ~ N(flux_true, |observed error|)``, pcigale
        ``managers/observations.py`` ``generate_mock``). Must have the shape
        of the predicted photometry: ``(n_bands,)`` for a single parameter set,
        ``(n_batch, n_bands)`` for batched parameters; scalars are refused.
        The absolute value is taken. CIGALE draws its mock from the errors
        after its model-error term has been added in quadrature
        (``additionalerror``, 10 % of the flux by default), so to reproduce a
        CIGALE mock pass ``sqrt(err**2 + (0.1 * flux)**2)``; the raw catalog
        errors give less scatter. Default ``None``: ``sigma = flux_true / snr``.

    Returns
    -------
    MockDict
        A ``dict`` subclass (so ``mock["flux_obs"]`` works) that also exposes
        its keys as attributes (so ``mock.flux_obs`` works, matching the
        :class:`MockData` object returned by :meth:`SEDModel.mock`). Keys:

        - ``flux_true`` : noiseless predicted photometry [erg/s/cm²/Hz]
        - ``noise`` : noise standard deviation per band [erg/s/cm²/Hz] (the
          ``noise`` argument when given, else ``flux_true / snr``)
        - ``params`` : the input parameter values
        - ``flux_obs`` : observed (noisy) photometry (only if key is not None)

    Raises
    ------
    ValueError
        If ``noise`` does not have the shape of the predicted photometry.

    Notes
    -----
    **Noise model**: Assumes Gaussian noise with σ = flux_true / SNR
    (appropriate for photon-limited observations), or with the supplied
    per-band ``noise`` (a catalog's observed errors).

    Examples
    --------
    >>> import jax.random
    >>> from tengri.forward import SEDModel
    >>> model = SEDModel(...)
    >>> params = {'redshift': 0.1, ...}
    >>> key = jax.random.PRNGKey(42)
    >>> mock = generate_mock(model, params, key=key, snr=10.0)
    >>> print(f"True flux shape: {mock['flux_true'].shape}")
    >>> print(f"Obs. flux shape: {mock['flux_obs'].shape}")
    """
    flux_true = model.predict_photometry(params)
    if noise is None:
        sigma = flux_true / snr
    else:
        sigma = jnp.abs(jnp.asarray(noise))
        if sigma.shape != flux_true.shape:
            raise ValueError(
                f"noise has shape {sigma.shape}, expected {flux_true.shape} "
                "(one observed 1-sigma per predicted band)."
            )

    result = MockDict(
        flux_true=flux_true,
        noise=sigma,
        params=params,
    )

    if key is not None:
        flux_obs = flux_true + sigma * jax.random.normal(key, shape=flux_true.shape)
        result["flux_obs"] = flux_obs

    return result
