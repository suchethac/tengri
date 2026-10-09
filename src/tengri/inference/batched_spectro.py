# SPDX-License-Identifier: BSD-3-Clause
"""Batched spectroscopic log-likelihood and vmapped MAP across galaxies (#2833).

:mod:`tengri.forward.batched_spectrum` evaluates one galaxy's spectrum from a
:class:`~tengri.observation.batched.SpectroBatch` slice. This module adds the
Gaussian data term, the standardized negative log-posterior and a batched MAP
fit over every galaxy in a catalog.

Standardized space
------------------
The MAP fit runs in the standardized (unbounded) space of
:class:`~tengri.parameters.parameters.Parameters`. Every free parameter
:math:`\\theta` has a latent :math:`\\xi` with :math:`\\xi \\sim \\mathcal{N}(0, 1)`
under the prior, and the physical value is ``dist.unstandardize(xi)``. The
negative log-posterior in that space is

.. math::

    -\\log p(\\xi \\mid d) = -\\log p(d \\mid \\theta(\\xi)) + \\tfrac{1}{2} \\sum_i \\xi_i^2
    + \\text{const.}

where the Jacobian of the pushforward cancels against the prior density, so
the prior term carries no Jacobian correction (see
:func:`~tengri.inference.loss_functions.standardized_neg_log_prior`).

Redshift is not a latent
------------------------
``redshift`` is excluded from the latents. Each galaxy's redshift comes from
its data (``obs.z``), not from the fit. The template model keeps its redshift
free only so that the batched forward pass can read ``obs.z`` through the
parameter path; the MAP never samples it.

v1 scope
--------
Gaussian diagonal noise only. Not covered: covariance matrices, Student-t
noise, calibration marginalization, emission-line channels, spectral indices,
and stochastic (GP-field) SFH. A stochastic model raises
:class:`NotImplementedError`.

JIT note
--------
:func:`batched_log_likelihood` and :func:`batched_neg_log_posterior` return
pure JAX functions, safe under :func:`jax.vmap`, :func:`jax.grad` and
:func:`jax.jit`. :func:`fit_spectra_map_vmap` is a Python driver: it compiles
one Adam scan per bucket, and the bucket spec is the compile signature.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from tengri.forward.batched_spectrum import (
    check_template_model,
    predict_batched_observables,
)
from tengri.inference.likelihoods.gaussian import diag_gaussian_log_prob
from tengri.inference.loss_functions import (
    _unstandardize_parameters,
    standardized_neg_log_prior,
)
from tengri.observation.batched import SpectroBatch, SpectroBatchSpec, require_spec
from tengri.parameters.priors import Gaussian

_REDSHIFT = "redshift"


def _refuse_stochastic(model) -> None:
    """Raise :class:`NotImplementedError` for a GP-field (stochastic) SFH spec."""
    if model.spec.stochastic:
        raise NotImplementedError(
            "batched spectroscopic fitting does not support stochastic (GP-field) SFH yet; "
            "use a parametric SFH"
        )


def _free_latent_names(model) -> list[str]:
    """Free parameter names that the MAP fits, with the redshift removed."""
    return [n for n in model.spec.free_params if n != _REDSHIFT]


def _prepare_template(model) -> Any:
    """Validate the template and build its component chain outside any trace.

    Returns the threaded ``(ssp_data, template_data, ztable_data)`` tuple that
    :func:`~tengri.forward.batched_spectrum.predict_batched_observables` takes.
    """
    _refuse_stochastic(model)
    check_template_model(model)
    if getattr(model, "_cached_component_chain", None) is None:
        model._cached_component_chain = model._build_component_chain()
    return model._resolve_threaded_data(None, None, None)


def batched_log_likelihood(
    model, spec: SpectroBatchSpec, *, conserving: bool
) -> Callable[[dict, SpectroBatch], jax.Array]:
    """Return the one-galaxy Gaussian log-likelihood as a pure JAX function.

    Parameters
    ----------
    model : SEDModel
        Template model with a free redshift (see
        :func:`~tengri.forward.batched_spectrum.check_template_model`).
    spec : SpectroBatchSpec
        Static bucket key of the batch the function will receive.
    conserving : bool
        Flux-conserving pixel integral, passed to the forward pass.

    Returns
    -------
    callable
        ``loglik(params, obs) -> scalar``. ``params`` maps each free physical
        parameter name to a scalar [physical units]; ``obs`` is one galaxy
        (:meth:`SpectroBatch.galaxy`). The function returns
        :math:`-\\tfrac12 \\chi^2` over the real spectral pixels, plus the
        photometric term when ``spec.has_phot``. The normalization constant is
        dropped. Call under :func:`jax.vmap` for a batch.

    Raises
    ------
    NotImplementedError
        If the model's SFH is stochastic.
    ValueError
        If the template fails :func:`check_template_model`, or ``spec`` is not a
        :class:`SpectroBatchSpec`.

    Notes
    -----
    **JIT-compatible**: yes; the returned function is pure JAX. The Gaussian
    term uses ``presence=obs.pix_mask``, so padded and masked pixels contribute
    exactly zero to the value and to every gradient. The template's component
    chain is built here, outside the trace.
    """
    require_spec(spec)
    threaded = _prepare_template(model)

    def loglik(params: dict, obs: SpectroBatch) -> jax.Array:
        out = predict_batched_observables(
            model, params, obs, spec, conserving=conserving, threaded=threaded
        )
        total = diag_gaussian_log_prob(out["spec_fnu"], obs.flux, obs.sigma, presence=obs.pix_mask)
        if spec.has_phot:
            total = total + diag_gaussian_log_prob(
                out["phot_fnu"], obs.phot_flux, obs.phot_err, presence=obs.phot_presence
            )
        return total

    return loglik


def batched_neg_log_posterior(
    model, spec: SpectroBatchSpec, *, conserving: bool
) -> Callable[[dict, SpectroBatch], jax.Array]:
    """Return the one-galaxy negative log-posterior in standardized space.

    Parameters
    ----------
    model : SEDModel
        Template model (see :func:`batched_log_likelihood`).
    spec : SpectroBatchSpec
        Static bucket key of the batch the function will receive.
    conserving : bool
        Flux-conserving pixel integral, passed to the forward pass.

    Returns
    -------
    callable
        ``nlp(params_unbounded, obs) -> scalar``. ``params_unbounded`` maps each
        free parameter other than redshift to its standardized latent; ``obs`` is
        one galaxy. The value is
        :math:`-\\log p(d \\mid \\theta(\\xi)) + \\tfrac12 \\sum_i \\xi_i^2`, up to a
        constant.

    Raises
    ------
    NotImplementedError
        If the model's SFH is stochastic.

    Notes
    -----
    **JIT-compatible**: yes; pure JAX under :func:`jax.vmap` and
    :func:`jax.grad`. The physical values come from
    :func:`~tengri.inference.loss_functions._unstandardize_parameters`, and only
    the free names are passed on, so fixed values never reach the forward pass.
    """
    loglik = batched_log_likelihood(model, spec, conserving=conserving)
    free_names = _free_latent_names(model)
    fixed_values = dict(model.spec.get_fixed_values())

    def nlp(params_unbounded: dict, obs: SpectroBatch) -> jax.Array:
        physical = _unstandardize_parameters(
            params_unbounded, model.spec, free_names, fixed_values, stochastic=False
        )
        phys_free = {n: physical[n] for n in free_names}
        prior = standardized_neg_log_prior(params_unbounded, free_names, stochastic=False)
        return -loglik(phys_free, obs) + prior

    return nlp


def init_unbounded_batch(model, n_galaxies: int, key) -> dict:
    """Draw one set of standardized initial latents per galaxy.

    Parameters
    ----------
    model : SEDModel
        Template model. Its free parameters, other than redshift, are initialized.
    n_galaxies : int
        Batch size ``B``.
    key : jax.Array
        PRNG key. Split into one key per galaxy.

    Returns
    -------
    dict of str to ndarray
        Each free parameter name other than redshift maps to an array of shape
        ``(B,)``.

    Raises
    ------
    NotImplementedError
        If the model's SFH is stochastic.

    Notes
    -----
    Mirrors :meth:`~tengri.inference.fitter.Fitter._initialize_unbounded`: a
    Gaussian prior starts at its standardized mean, and any other prior starts
    at ``0.1 * N(0, 1)``. Each galaxy's keys come from splitting the key over the
    full free list, redshift included, so the non-redshift entries match the
    Fitter's draw for the same per-galaxy key.

    **JIT-compatible**: yes under :func:`jax.vmap`; this is a setup helper.
    """
    _refuse_stochastic(model)
    names = list(model.spec.free_params)
    dists = {n: model.spec.get_distribution(n) for n in names}

    def init_one(galaxy_key):
        keys = jax.random.split(galaxy_key, len(names) + 1)
        out = {}
        for i, name in enumerate(names):
            if name == _REDSHIFT:
                continue
            dist = dists[name]
            if isinstance(dist, Gaussian):
                out[name] = dist.standardize(jnp.array(dist.mu))
            else:
                out[name] = 0.1 * jax.random.normal(keys[i], shape=())
        return out

    return jax.vmap(init_one)(jax.random.split(key, n_galaxies))


def _to_physical(model, params_unbounded_batch: dict) -> dict:
    """Map batched standardized latents to batched physical free parameters."""
    free_names = _free_latent_names(model)
    fixed_values = dict(model.spec.get_fixed_values())

    def one(p):
        physical = _unstandardize_parameters(
            p, model.spec, free_names, fixed_values, stochastic=False
        )
        return {n: physical[n] for n in free_names}

    return jax.vmap(one)(params_unbounded_batch)


def _fit_bucket(model, spec, batch, *, conserving, n_steps, learning_rate, key):
    """Run Adam on one bucket, returning final latents, final nlp and the loss trace."""
    import optax

    nlp = batched_neg_log_posterior(model, spec, conserving=conserving)
    n_gal = int(batch.z.shape[0])
    init = init_unbounded_batch(model, n_gal, key)

    optimizer = optax.adam(learning_rate)
    opt_state = jax.vmap(optimizer.init)(init)
    value_and_grad = jax.vmap(jax.value_and_grad(nlp))

    def step(carry, _):
        params, state = carry
        losses, grads = value_and_grad(params, batch)
        updates, state = jax.vmap(optimizer.update)(grads, state, params)
        params = jax.vmap(optax.apply_updates)(params, updates)
        return (params, state), losses

    (params_final, _), loss_trace = jax.lax.scan(step, (init, opt_state), None, length=n_steps)
    final_nlp = jax.vmap(nlp)(params_final, batch)
    return params_final, final_nlp, loss_trace


def fit_spectra_map_vmap(
    model,
    batches: Sequence[tuple[SpectroBatchSpec, SpectroBatch, np.ndarray]],
    *,
    conserving: bool,
    n_steps: int = 500,
    learning_rate: float = 0.05,
    seed: int = 0,
) -> dict[str, Any]:
    """Batched MAP fit of spectra across galaxies, one Adam scan per bucket.

    Parameters
    ----------
    model : SEDModel
        Template model with a free redshift (see
        :func:`~tengri.forward.batched_spectrum.check_template_model`).
    batches : sequence of (SpectroBatchSpec, SpectroBatch, ndarray)
        Output of :func:`~tengri.observation.batched.build_spectro_batches`.
    conserving : bool
        Flux-conserving pixel integral.
    n_steps : int, optional
        Adam iterations per bucket. Default 500.
    learning_rate : float, optional
        Adam learning rate in standardized space. Default 0.05.
    seed : int, optional
        PRNG seed for the initial latents. Default 0.

    Returns
    -------
    dict
        ``"params"``: dict of str to ndarray, shape ``(N,)``. The MAP physical
        value of each free parameter other than redshift, for every galaxy in
        the original catalog order (``N`` is the total galaxy count).
        ``"nlp"``: ndarray, shape ``(N,)``. The final negative log-posterior of
        each galaxy, in the same order.
        ``"loss_history"``: list of ndarray, one per bucket in ``batches`` order,
        each of shape ``(n_steps, B_bucket)``: the negative log-posterior at
        each step before that step's update.

    Raises
    ------
    NotImplementedError
        If the model's SFH is stochastic.
    ValueError
        If the template fails :func:`check_template_model`.

    Notes
    -----
    **JIT-compatible**: no; this is a Python driver. Each bucket compiles once for
    its :class:`SpectroBatchSpec`. Results are returned in the catalog order by
    permuting with the concatenated bucket index arrays.
    """
    _refuse_stochastic(model)
    check_template_model(model)
    if not batches:
        raise ValueError("batches is empty; build it with build_spectro_batches")

    base_key = jax.random.PRNGKey(seed)
    bucket_keys = jax.random.split(base_key, len(batches))
    buckets = []
    for (spec, batch, index), key in zip(batches, bucket_keys):
        params_u, final_nlp, trace = _fit_bucket(
            model,
            spec,
            batch,
            conserving=conserving,
            n_steps=n_steps,
            learning_rate=learning_rate,
            key=key,
        )
        buckets.append(
            {
                "index": np.asarray(index, dtype=int),
                "params": _to_physical(model, params_u),
                "nlp": final_nlp,
                "trace": trace,
            }
        )

    catalog_index = np.concatenate([b["index"] for b in buckets])
    order = np.argsort(catalog_index, kind="stable")
    names = _free_latent_names(model)
    params = {n: jnp.concatenate([b["params"][n] for b in buckets])[order] for n in names}
    nlp = jnp.concatenate([b["nlp"] for b in buckets])[order]
    return {
        "params": params,
        "nlp": nlp,
        "loss_history": [b["trace"] for b in buckets],
    }
