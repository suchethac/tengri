# SPDX-License-Identifier: BSD-3-Clause
"""Likelihood protocol: log-probability of data given a forward-model prediction.

Decouples inference (:mod:`tengri.inference`) from the forward model
(:class:`tengri.SEDModel`). An inference backend only needs to know
that a likelihood produces a scalar log-probability given (data,
prediction); it doesn't need to import any physics module.

Nothing in `tengri` consumes this protocol yet — current inference
reaches into ``Fitter``-internal helpers. A future pass will rewire
them to a Likelihood implementation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

import jax.numpy as jnp

__all__ = ["Likelihood"]


@runtime_checkable
class Likelihood(Protocol):
    """Contract for log-probability of data given prediction.

    Concrete implementations:

    - ``GaussianLikelihood``: independent Gaussian errors
      (chi-squared + log-normalization).
    - ``StudentTLikelihood``: heavy-tailed alternative for outlier
      tolerance.
    - ``GPLikelihood``: Gaussian-process correlated noise on
      spectroscopy.
    - ``UpperLimitLikelihood``: half-Gaussian / step-function for
      non-detections.
    - ``CompositeLikelihood``: sum of channel-specific likelihoods
      (phot + spec + lines).

    Required attributes
    -------------------
    name : str
        Stable identifier. Examples: ``"gaussian"``, ``"student_t"``,
        ``"gp"``, ``"composite"``.

    Required methods
    ----------------
    log_prob(prediction, data, noise_params) -> scalar
        Pure JAX. Returns the log-likelihood. Sign convention: a
        higher number means a better fit. Inference backends typically
        minimize the negative.

    declared_parameters() -> list[str]
        Parameter name strings the likelihood owns (noise floors,
        Student-t degrees of freedom, GP kernel hyperparameters).
        Domain prefix: ``noise_``.

    Notes
    -----
    **JIT-compatible:** :meth:`log_prob` is pure JAX. The static data
    arrays (observed F_nu, errors, masks) are held as JAX arrays on
    ``self``; only the *prediction* and any traced *noise_params*
    flow in.
    """

    name: str

    def declared_parameters(self) -> list[str]:
        """Parameter name strings the likelihood owns.

        Returns
        -------
        list[str]
            List of parameter names (empty for most likelihoods;
            non-empty only for adapters that fit nuisance parameters,
            e.g., :class:`ELineFittedLikelihood`).
        """
        ...

    def log_prob(
        self,
        prediction: Mapping[str, jnp.ndarray],
        params: Mapping[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """Log-likelihood scalar.

        Parameters
        ----------
        prediction : mapping of str -> array
            Output of :meth:`tengri.protocols.ObservationModel.predict`.
        params : mapping of str -> array
            Free parameters whose name starts with the likelihood's
            domain prefix (``noise_``).

        Returns
        -------
        ndarray, shape ()
            Scalar log-likelihood. Higher = better fit. Must be
            differentiable w.r.t. every traced input for gradient-based
            inference (MAP, VI, HMC) to work.
        """
        ...
