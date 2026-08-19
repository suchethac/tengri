# SPDX-License-Identifier: BSD-3-Clause
"""Bayesian model averaging (BMA) over per-model log evidences.

BMA combines predictions from multiple models by weighting them according to
their posterior model probabilities. Under a flat prior over models, these
probabilities are proportional to the marginal likelihoods (evidences).

This module provides:
- :func:`bma_weights`: softmax of log evidences (posterior model probabilities)
- :func:`bma_resample`: multinomial pooling of physical-space samples from
  multiple models, weighted by their model probabilities
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import jax
import numpy as np


def bma_weights(
    posteriors: Sequence | Mapping,
) -> np.ndarray | dict:
    """Compute Bayesian model averaging weights from log evidences.

    Under a flat prior over models, the posterior model probabilities are
    proportional to the marginal likelihoods (log evidences). This function
    computes the softmax of log evidences, which are the normalized weights
    for model averaging.

    Parameters
    ----------
    posteriors : Sequence or Mapping
        If Sequence: iterable of posterior-like objects, each with a
        `log_evidence` attribute.
        If Mapping: dict-like with model names as keys and posterior-like
        objects as values.

    Returns
    -------
    weights : np.ndarray or dict
        If input is a Sequence, returns a numpy array of shape (n_models,)
        with weights summing to 1.0.
        If input is a Mapping, returns a dict with the same keys as the input,
        with float values summing to 1.0.

    Raises
    ------
    ValueError
        If any posterior's `log_evidence` is None, NaN, or infinite.
        The error message names the offending model (by index if Sequence,
        by key if Mapping).

    Notes
    -----
    The softmax is computed using the log-sum-exp trick with max-shifting
    for numerical stability:

    .. code-block:: python

        w_i = exp(log_evidence_i - max(log_evidence)) / sum_j(...)

    This prevents overflow/underflow when evidence values are large or
    small in absolute magnitude.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import types
    >>> # Create mock posteriors
    >>> p1 = types.SimpleNamespace(log_evidence=0.0)
    >>> p2 = types.SimpleNamespace(log_evidence=jnp.log(3.0))
    >>> weights = bma_weights([p1, p2])
    >>> weights
    array([0.25, 0.75])
    """
    if isinstance(posteriors, Mapping):
        return _bma_weights_dict(posteriors)
    else:
        return _bma_weights_sequence(posteriors)


def _bma_weights_sequence(posteriors: Sequence) -> np.ndarray:
    """Compute BMA weights from a sequence of posteriors."""
    log_evidences = []
    for i, post in enumerate(posteriors):
        log_z = post.log_evidence
        if log_z is None or not np.isfinite(log_z):
            raise ValueError(
                f"Posterior at index {i} has invalid log_evidence: {log_z}. "
                "Refit with an evidence method: 'nss', 'laplace', or 'hmc_is'."
            )
        log_evidences.append(float(log_z))

    log_evidences = np.array(log_evidences)

    # Max-shifted softmax for numerical stability
    log_max = np.max(log_evidences)
    exp_shifted = np.exp(log_evidences - log_max)
    weights = exp_shifted / np.sum(exp_shifted)

    return weights


def _bma_weights_dict(posteriors: Mapping) -> dict:
    """Compute BMA weights from a mapping of posteriors."""
    keys = list(posteriors.keys())
    log_evidences = []

    for key in keys:
        post = posteriors[key]
        log_z = post.log_evidence
        if log_z is None or not np.isfinite(log_z):
            raise ValueError(
                f"Model '{key}' has invalid log_evidence: {log_z}. "
                "Refit with an evidence method: 'nss', 'laplace', or 'hmc_is'."
            )
        log_evidences.append(float(log_z))

    log_evidences = np.array(log_evidences)

    # Max-shifted softmax for numerical stability
    log_max = np.max(log_evidences)
    exp_shifted = np.exp(log_evidences - log_max)
    weights_array = exp_shifted / np.sum(exp_shifted)

    return {key: float(w) for key, w in zip(keys, weights_array)}


def bma_resample(
    posteriors: Sequence | Mapping,
    n_draws: int,
    key: jax.random.PRNGKey,
) -> dict:
    """Resample from multiple models weighted by Bayesian model averaging.

    Performs multinomial resampling of physical-space samples from posterior
    chains, where each model contributes a number of draws proportional to its
    BMA weight. Only parameters present in **all** models are returned (the
    intersection of sample keys).

    Parameters
    ----------
    posteriors : Sequence or Mapping
        Iterable of posterior-like objects, each with `log_evidence` and
        `samples` attributes.
    n_draws : int
        Total number of draws to resample from the pooled posteriors.
    key : jax.random.PRNGKey
        JAX PRNG key for deterministic randomness.

    Returns
    -------
    resampled : dict
        Dictionary mapping parameter names (the intersection of all sample
        keys) to numpy arrays of shape (n_draws,).

    Raises
    ------
    ValueError
        If any posterior has ``samples=None``, or if the intersection of
        sample keys is empty (no shared parameters).

    Notes
    -----
    **Intersection semantics**: For structurally different models, only
    parameters present in all chains are pooled. This is the operation
    the CANDELS notebook requires for model comparison.

    **Sampling procedure**:

    1. Compute BMA weights via :func:`bma_weights`
    2. Draw model indices via multinomial with probabilities = weights
    3. For each model, resample (with replacement) from its chain
    4. Concatenate samples from all models, preserving the order of models
       by draw count

    Examples
    --------
    >>> import jax
    >>> import numpy as np
    >>> import types
    >>> key = jax.random.PRNGKey(42)
    >>> p1 = types.SimpleNamespace(
    ...     log_evidence=0.0, samples={"param_a": np.array([1.0, 2.0, 3.0])}
    ... )
    >>> p2 = types.SimpleNamespace(
    ...     log_evidence=0.0, samples={"param_a": np.array([10.0, 20.0, 30.0])}
    ... )
    >>> resampled = bma_resample([p1, p2], n_draws=100, key=key)
    >>> resampled["param_a"].shape
    (100,)
    """
    # Convert to sequence if needed
    if isinstance(posteriors, Mapping):
        posteriors_seq = list(posteriors.values())
    else:
        posteriors_seq = list(posteriors)

    # Validate all posteriors have samples
    for i, post in enumerate(posteriors_seq):
        if post.samples is None:
            raise ValueError(
                f"Posterior at index {i} has samples=None. "
                "Refit with a method that produces samples."
            )

    # Compute BMA weights
    weights = bma_weights(posteriors_seq)

    # Find intersection of sample keys
    all_keys = [set(post.samples.keys()) for post in posteriors_seq]
    shared_keys = set.intersection(*all_keys) if all_keys else set()

    if not shared_keys:
        raise ValueError(
            "No intersection of sample keys across models. "
            "Models must share at least one parameter for pooling."
        )

    shared_keys = sorted(shared_keys)  # deterministic ordering

    # Sample model indices via multinomial
    key_multinomial, key_resample = jax.random.split(key)

    # Use JAX for deterministic multinomial sampling
    # jax.random.multinomial returns counts for each category in a single draw
    # We want to replicate this n_draws times, so we sample model indices repeatedly
    model_indices = jax.random.choice(
        key_multinomial, len(posteriors_seq), shape=(n_draws,), p=weights
    )
    model_indices = np.asarray(model_indices)

    # Count how many draws belong to each model
    model_counts = np.bincount(model_indices, minlength=len(posteriors_seq))

    # Resample from each model's chain
    resampled_data = {key: [] for key in shared_keys}

    key_idx = key_resample
    for model_idx, count in enumerate(model_counts):
        if count == 0:
            continue

        post = posteriors_seq[model_idx]
        chain_len = len(post.samples[shared_keys[0]])

        # Split key for this model's resampling
        key_idx, subkey = jax.random.split(key_idx)

        # Sample indices with replacement from this model's chain
        indices = jax.random.choice(subkey, chain_len, shape=(count,), replace=True)
        indices = np.asarray(indices)

        # Collect samples for each shared parameter
        for param_key in shared_keys:
            samples = np.asarray(post.samples[param_key])
            resampled_data[param_key].append(samples[indices])

    # Concatenate samples across models
    result = {}
    for param_key in shared_keys:
        result[param_key] = np.concatenate(resampled_data[param_key])

    return result
