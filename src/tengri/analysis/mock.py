# SPDX-License-Identifier: BSD-3-Clause
"""Mock galaxy generation utilities.

Standalone ``generate_mock`` function for quick mock photometry
generation. For the ``SEDModel``-based API, use ``SEDModel.mock()`` instead.
"""

import jax

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


def generate_mock(model, params, key=None, snr=20.0):
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
        Signal-to-noise ratio (flux_true / noise_std). Default: 20.0.

    Returns
    -------
    MockDict
        A ``dict`` subclass (so ``mock["flux_obs"]`` works) that also exposes
        its keys as attributes (so ``mock.flux_obs`` works, matching the
        :class:`MockData` object returned by :meth:`SEDModel.mock`). Keys:

        - ``flux_true`` : noiseless predicted photometry [erg/s/cm²/Hz]
        - ``noise`` : noise standard deviation per band [erg/s/cm²/Hz]
        - ``params`` : the input parameter values
        - ``flux_obs`` : observed (noisy) photometry (only if key is not None)

    Notes
    -----
    **Noise model**: Assumes Gaussian noise with σ = flux_true / SNR.
    This is appropriate for photon-limited observations.

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
    noise = flux_true / snr

    result = MockDict(
        flux_true=flux_true,
        noise=noise,
        params=params,
    )

    if key is not None:
        flux_obs = flux_true + noise * jax.random.normal(key, shape=flux_true.shape)
        result["flux_obs"] = flux_obs

    return result
