"""Mock galaxy generation utilities.

Standalone ``generate_mock`` function for quick mock photometry
generation. For the ``SEDModel``-based API, use ``SEDModel.mock()`` instead.
"""

import jax

# Re-export MockData so callers can import from one place
from tengri.forward.sed_model import MockData

__all__ = ["MockData", "generate_mock"]


def generate_mock(model, params, key=None, snr=20.0):
    """Generate mock photometry with Gaussian noise.

    Parameters
    ----------
    model : object
        Any object with a ``predict_photometry(params)`` method.
    params : dict
        Model parameter values.
    key : jax.random.PRNGKey, optional
        Random key for noise realization. If ``None``, only noiseless
        photometry is returned (no ``flux_obs`` key).
    snr : float
        Signal-to-noise ratio.

    Returns
    -------
    dict
        Dictionary with keys ``flux_true``, ``noise``, ``params``,
        and (if *key* is not None) ``flux_obs``.
    """
    flux_true = model.predict_photometry(params)
    noise = flux_true / snr

    result = {
        "flux_true": flux_true,
        "noise": noise,
        "params": params,
    }

    if key is not None:
        flux_obs = flux_true + noise * jax.random.normal(key, shape=flux_true.shape)
        result["flux_obs"] = flux_obs

    return result
