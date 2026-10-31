# SPDX-License-Identifier: BSD-3-Clause
"""Batched spectroscopic log-likelihood and vmapped MAP across galaxies.

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
Gaussian diagonal noise only. Emission lines are covered with
``eline_mode="fitted"``: each independent line amplitude is a latent with the
prior of the non-batched fitter, and the returned ``params`` carry the
``eline_amp_*`` entries next to the physical parameters.

Calibration marginalization is covered: with ``calibration_marginalize=True``
the spectral term is the closed-form marginal over a Chebyshev calibration
polynomial, the same :func:`~tengri.observation.linear_marginal.marginalize_linear_nuisance`
primitive that the per-galaxy calibration-marginalized likelihood evaluates.

Emission lines marginalized analytically (``eline_mode="marginalized"`` with a flat
prior) are covered, jointly with the calibration polynomial when
``calibration_marginalize=True``. Not covered: Cloudy-prior line marginalization
(``eline_prior_type="cloudy"``), spectral covariance, Student-t and
calibration-floor noise, spectral indices, and stochastic (GP-field) SFH. A
stochastic model raises :class:`NotImplementedError`.

JIT note
--------
:func:`batched_log_likelihood` and :func:`batched_neg_log_posterior` return
pure JAX functions, safe under :func:`jax.vmap`, :func:`jax.grad` and
:func:`jax.jit`. :func:`fit_spectra_map_vmap` is a Python driver: it compiles
one Adam scan per bucket, and the bucket spec is the compile signature.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from jax.scipy.linalg import block_diag

from tengri.config.exceptions import ConfigError
from tengri.forward.batched_spectrum import (
    batched_eline_design,
    batched_line_set,
    check_template_model,
    predict_batched_observables,
    require_model_spec,
)
from tengri.inference.backends.mcmc.catalog import build_catalog_mcmc_engine
from tengri.inference.likelihoods.gaussian import diag_gaussian_log_prob
from tengri.inference.loss_functions import (
    _unstandardize_parameters,
    standardized_neg_log_prior,
)
from tengri.observation.batched import SpectroBatch, SpectroBatchSpec
from tengri.observation.calibration import calibration_basis
from tengri.observation.eline_marginalization import eline_amplitude_names
from tengri.observation.linear_marginal import marginalize_linear_nuisance
from tengri.parameters.priors import Gaussian, Uniform

_REDSHIFT = "redshift"


def _refuse_stochastic(model) -> None:
    """Raise :class:`NotImplementedError` for a GP-field (stochastic) SFH spec."""
    if model.spec.stochastic:
        raise NotImplementedError(
            "batched spectroscopic fitting does not support stochastic (GP-field) SFH yet; "
            "use a parametric SFH"
        )


def _refuse_calibration_order(model) -> None:
    """Raise :class:`ConfigError` if the template also fits an explicit calibration polynomial."""
    spectroscopy = getattr(model.observation, "spectroscopy", None)
    if spectroscopy is not None and spectroscopy.calibration_order > 0:
        raise ConfigError(
            "Cannot enable both explicit calibration fitting (calibration_order > 0) "
            "and analytic calibration marginalization (calibration_marginalize=True). "
            "This would double-count the calibration polynomial degrees of freedom.\n\n"
            "Choose one:\n"
            "  • To fit calibration coefficients as free parameters:\n"
            "    Set calibration_marginalize=False (default) and keep "
            "calibration_order > 0.\n"
            "  • To marginalize calibration analytically:\n"
            "    Set calibration_order=0 on your spectroscopy and "
            "enable calibration_marginalize=True."
        )


def _spectral_marginal_loglik(model_flux, obs: SpectroBatch, n_poly: int, prior_sigma: float):
    """Marginal log-likelihood of one galaxy's spectrum over a Chebyshev calibration.

    The design is ``model_flux[:, None] * T(x)`` on the real grid's extremes, so padded
    pixels (zero weight through ``obs.pix_mask``) carry no information.
    """
    design = model_flux[:, None] * calibration_basis(obs.wave, obs.cal_lo, obs.cal_hi, n_poly)
    return marginalize_linear_nuisance(
        data=obs.flux,
        model0=model_flux,
        design=design,
        sigma=obs.sigma,
        pix_mask=obs.pix_mask,
        prior_precision=jnp.eye(n_poly) / prior_sigma**2,
    ).log_like


def _latent_spec(model):
    """Return the spec whose free parameters are the latents the fit samples.

    With emission lines off this is ``model.spec``. With ``eline_mode`` set, each
    independent ``eline_amp_<name>`` gets the prior ``Uniform(-b, b)`` with
    ``b = 10 * eline_prior_sigma``, the same prior as ``Fitter._init_eline_arrays``.
    """
    spectroscopy = model.observation.spectroscopy
    if spectroscopy.eline_mode in ("off", "marginalized"):
        return model.spec
    names = eline_amplitude_names(spectroscopy.effective_catalog)
    bound = 10.0 * spectroscopy.eline_prior_sigma
    return model.spec.merge_observation_params(**{nm: Uniform(-bound, bound) for nm in names})


def _free_latent_names(model) -> list[str]:
    """Free parameter names that the MAP fits, with the redshift removed."""
    return [n for n in _latent_spec(model).free_params if n != _REDSHIFT]


def _prepare_template(model) -> Any:
    """Validate the template and build its component chain outside any trace.

    Returns the threaded ``(ssp_data, template_data, ztable_data)`` tuple that
    :func:`~tengri.forward.batched_spectrum.predict_batched_observables` takes.
    """
    _refuse_stochastic(model)
    check_template_model(model)
    return model.threaded_template_inputs()


def _marginal_lines_loglik(
    model,
    line_set,
    model0,
    params: dict,
    obs: SpectroBatch,
    *,
    calibration_marginalize: bool,
    cal_n_poly: int,
    cal_prior_sigma: float,
) -> jax.Array:
    """Marginal log-likelihood over the line amplitudes, jointly with the calibration.

    The design is the line matrix of :func:`batched_eline_design`, with the
    calibration columns ``model0 * T(x)`` in front when ``calibration_marginalize``.
    A line with no support on the real grid is removed through ``col_mask``.
    The normalization constant is dropped, as in the per-galaxy line marginal.
    """
    spectroscopy = model.observation.spectroscopy
    full = {**dict(model.spec.get_fixed_values()), **params}
    g_eff = batched_eline_design(line_set, obs, full)
    line_cols = jnp.sum(jnp.abs(g_eff), axis=0) > 0
    line_prec = jnp.eye(g_eff.shape[1]) / spectroscopy.eline_prior_sigma**2
    if calibration_marginalize:
        basis = calibration_basis(obs.wave, obs.cal_lo, obs.cal_hi, cal_n_poly)
        design = jnp.concatenate([model0[:, None] * basis, g_eff], axis=1)
        col_mask = jnp.concatenate([jnp.ones(cal_n_poly, dtype=bool), line_cols])
        precision = block_diag(jnp.eye(cal_n_poly) / cal_prior_sigma**2, line_prec)
    else:
        design, col_mask, precision = g_eff, line_cols, line_prec
    weight = obs.pix_mask
    n_eff = jnp.sum(weight)
    log_sigma_sum = jnp.sum(weight * jnp.log(obs.sigma))
    result = marginalize_linear_nuisance(
        obs.flux,
        model0,
        design,
        sigma=obs.sigma,
        pix_mask=obs.pix_mask,
        prior_precision=precision,
        col_mask=col_mask.astype(design.dtype),
        scale=jnp.exp(log_sigma_sum / n_eff),
    )
    return result.log_like + 0.5 * n_eff * math.log(2.0 * math.pi) + log_sigma_sum


def batched_log_likelihood(
    model,
    spec: SpectroBatchSpec,
    *,
    calibration_marginalize: bool = False,
    cal_n_poly: int = 3,
    cal_prior_sigma: float = 1.0,
) -> Callable[[dict, SpectroBatch], jax.Array]:
    """Return the one-galaxy Gaussian log-likelihood as a pure JAX function.

    Parameters
    ----------
    model : SEDModel
        Template model with a free redshift (see
        :func:`~tengri.forward.batched_spectrum.check_template_model`).
    spec : SpectroBatchSpec
        Static bucket key of the batch the function will receive. Its
        ``conserving`` flag selects the pixel integral.
    calibration_marginalize : bool, optional
        Replace the spectral Gaussian term by the marginal over a Chebyshev
        calibration polynomial of order ``cal_n_poly``, integrated out under a
        Gaussian prior. The template must carry ``calibration_order=0``. Default False.
    cal_n_poly : int, optional
        Number of calibration coefficients, T_1 through T_``cal_n_poly``. Used only
        with ``calibration_marginalize``. Default 3.
    cal_prior_sigma : float, optional
        Standard deviation of the Gaussian prior on each calibration coefficient.
        Used only with ``calibration_marginalize``. Default 1.0.

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
    ConfigError
        If ``calibration_marginalize`` is set and the template's
        ``calibration_order`` is above zero (the calibration would be counted twice).

    Notes
    -----
    **JIT-compatible**: yes; the returned function is pure JAX. The Gaussian
    term uses ``presence=obs.pix_mask``, so padded and masked pixels contribute
    exactly zero to the value and to every gradient. The template's component
    chain is built here, outside the trace.

    With ``calibration_marginalize=False`` the spectral term drops the normalization
    constant :math:`-\\tfrac12 N \\ln 2\\pi - \\sum_i \\ln\\sigma_i` over the real pixels,
    as :func:`~tengri.inference.likelihoods.gaussian.diag_gaussian_log_prob` does. With
    ``calibration_marginalize=True`` the spectral term is a marginal likelihood and
    keeps that constant, as the per-galaxy ``CalibrationMarginalizedLikelihood`` does,
    so the two settings differ by that constant and by nothing else. The photometric
    term is unchanged in both. With ``eline_mode="marginalized"`` the spectral term is
    the marginal over the line amplitudes, and over the calibration polynomial too when
    ``calibration_marginalize=True``, with the normalization constant dropped.
    """
    threaded = _prepare_template(model)
    require_model_spec(model, spec)
    spectroscopy = model.observation.spectroscopy
    marginal_lines = spectroscopy.eline_mode == "marginalized"
    if calibration_marginalize:
        _refuse_calibration_order(model)
        if spectroscopy.eline_mode == "fitted":
            raise NotImplementedError(
                "Combined calibration marginalization + fitted (non-marginalized) "
                "emission-line amplitudes is not currently supported. Use "
                "eline_mode='marginalized' for the joint marginal, or disable "
                "calibration_marginalize."
            )
    line_set = batched_line_set(model) if marginal_lines else None

    def loglik(params: dict, obs: SpectroBatch) -> jax.Array:
        out = predict_batched_observables(model, params, obs, spec, threaded=threaded)
        if marginal_lines:
            total = _marginal_lines_loglik(
                model,
                line_set,
                out["spec_fnu"],
                params,
                obs,
                calibration_marginalize=calibration_marginalize,
                cal_n_poly=cal_n_poly,
                cal_prior_sigma=cal_prior_sigma,
            )
        elif calibration_marginalize:
            total = _spectral_marginal_loglik(out["spec_fnu"], obs, cal_n_poly, cal_prior_sigma)
        else:
            total = diag_gaussian_log_prob(
                out["spec_fnu"], obs.flux, obs.sigma, presence=obs.pix_mask
            )
        if spec.has_phot:
            total = total + diag_gaussian_log_prob(
                out["phot_fnu"], obs.phot_flux, obs.phot_err, presence=obs.phot_presence
            )
        return total

    return loglik


def batched_neg_log_posterior(
    model,
    spec: SpectroBatchSpec,
    *,
    calibration_marginalize: bool = False,
    cal_n_poly: int = 3,
    cal_prior_sigma: float = 1.0,
) -> Callable[[dict, SpectroBatch], jax.Array]:
    """Return the one-galaxy negative log-posterior in standardized space.

    Parameters
    ----------
    model : SEDModel
        Template model (see :func:`batched_log_likelihood`).
    spec : SpectroBatchSpec
        Static bucket key of the batch the function will receive.
    calibration_marginalize, cal_n_poly, cal_prior_sigma
        Forwarded to :func:`batched_log_likelihood`.

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
    ``_unstandardize_parameters`` (``tengri.inference.loss_functions``), and only
    the free names are passed on, so fixed values never reach the forward pass.
    """
    loglik = batched_log_likelihood(
        model,
        spec,
        calibration_marginalize=calibration_marginalize,
        cal_n_poly=cal_n_poly,
        cal_prior_sigma=cal_prior_sigma,
    )
    free_names = _free_latent_names(model)
    fixed_values = dict(model.spec.get_fixed_values())
    latent_spec = _latent_spec(model)

    def nlp(params_unbounded: dict, obs: SpectroBatch) -> jax.Array:
        physical = _unstandardize_parameters(
            params_unbounded, latent_spec, free_names, fixed_values, stochastic=False
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
        Each free parameter name other than redshift, including the
        ``eline_amp_*`` amplitudes when the template fits lines, maps to an array
        of shape ``(B,)``.

    Raises
    ------
    NotImplementedError
        If the model's SFH is stochastic.

    Notes
    -----
    Mirrors ``_initialize_unbounded`` (``tengri.inference.fitter.Fitter``): a
    Gaussian prior starts at its standardized mean, and any other prior starts
    at ``0.1 * N(0, 1)``. Each galaxy's keys come from splitting the key over the
    full free list, redshift included, so the non-redshift entries match the
    Fitter's draw for the same per-galaxy key.

    **JIT-compatible**: yes under :func:`jax.vmap`; this is a setup helper.
    """
    _refuse_stochastic(model)
    latent_spec = _latent_spec(model)
    names = list(latent_spec.free_params)
    dists = {n: latent_spec.get_distribution(n) for n in names}

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
    latent_spec = _latent_spec(model)

    def one(p):
        physical = _unstandardize_parameters(
            p, latent_spec, free_names, fixed_values, stochastic=False
        )
        return {n: physical[n] for n in free_names}

    return jax.vmap(one)(params_unbounded_batch)


def _slice_tree(tree, start: int, stop: int):
    """Return ``tree`` with every leaf sliced along its leading batch axis."""
    return jax.tree_util.tree_map(lambda a: a[start:stop], tree)


def _fit_chunk(nlp, batch, init, *, n_steps, learning_rate):
    """Run Adam on one chunk of galaxies from the given latents."""
    import optax

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


def _fit_bucket(
    model,
    spec,
    batch,
    *,
    n_steps,
    learning_rate,
    key,
    chunk_size,
    calibration_marginalize=False,
    cal_n_poly=3,
    cal_prior_sigma=1.0,
):
    """Run Adam on one bucket, returning final latents, final nlp and the loss trace.

    The initial latents are drawn for the whole bucket and then sliced, so the
    result does not depend on ``chunk_size``. Each chunk is a separate ``vmap``
    of at most ``chunk_size`` galaxies, which bounds the peak memory of the
    gradient.
    """
    nlp = batched_neg_log_posterior(
        model,
        spec,
        calibration_marginalize=calibration_marginalize,
        cal_n_poly=cal_n_poly,
        cal_prior_sigma=cal_prior_sigma,
    )
    n_gal = int(batch.z.shape[0])
    init = init_unbounded_batch(model, n_gal, key)
    step = n_gal if chunk_size is None else int(chunk_size)
    parts = []
    for start in range(0, n_gal, step):
        stop = min(start + step, n_gal)
        parts.append(
            _fit_chunk(
                nlp,
                _slice_tree(batch, start, stop),
                _slice_tree(init, start, stop),
                n_steps=n_steps,
                learning_rate=learning_rate,
            )
        )
    params_final = jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs), *[p[0] for p in parts])
    final_nlp = jnp.concatenate([p[1] for p in parts])
    loss_trace = jnp.concatenate([p[2] for p in parts], axis=1)
    return params_final, final_nlp, loss_trace


def fit_spectra_map_vmap(
    model,
    batches: Sequence[tuple[SpectroBatchSpec, SpectroBatch, np.ndarray]],
    *,
    n_steps: int = 500,
    learning_rate: float = 0.05,
    seed: int = 0,
    chunk_size: int | None = None,
    calibration_marginalize: bool = False,
    cal_n_poly: int = 3,
    cal_prior_sigma: float = 1.0,
) -> dict[str, Any]:
    """Batched MAP fit of spectra across galaxies, one Adam scan per bucket.

    Parameters
    ----------
    model : SEDModel
        Template model with a free redshift (see
        :func:`~tengri.forward.batched_spectrum.check_template_model`).
    batches : sequence of (SpectroBatchSpec, SpectroBatch, ndarray)
        Output of :func:`~tengri.observation.batched.build_spectro_batches` built
        with ``model=model``.
    n_steps : int, optional
        Adam iterations per bucket. Default 500.
    learning_rate : float, optional
        Adam learning rate in standardized space. Default 0.05.
    seed : int, optional
        PRNG seed for the initial latents. Default 0.
    chunk_size : int, optional
        Galaxies per ``vmap`` call within a bucket. ``None`` (default) maps the
        whole bucket at once. Smaller chunks bound the peak memory of the gradient
        on a device with little of it; the result does not depend on this value.
    calibration_marginalize : bool, optional
        Marginalize a Chebyshev calibration polynomial out of the spectral term, as
        :func:`batched_log_likelihood` does. Default False.
    cal_n_poly : int, optional
        Calibration coefficients when ``calibration_marginalize``. Default 3.
    cal_prior_sigma : float, optional
        Prior standard deviation of each calibration coefficient. Default 1.0.

    Returns
    -------
    dict
        ``"params"``: dict of str to ndarray, shape ``(N,)``. The MAP physical
        value of each free parameter other than redshift, for every galaxy in
        the original catalog order (``N`` is the total galaxy count). With
        ``eline_mode="fitted"`` the ``eline_amp_*`` line amplitudes are included.
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
        If the template fails :func:`check_template_model`, ``batches`` is empty,
        or ``chunk_size`` is not a positive integer.
    ConfigError
        If ``calibration_marginalize`` is set and the template's
        ``calibration_order`` is above zero.

    Notes
    -----
    **JIT-compatible**: no; this is a Python driver. Each bucket compiles once for
    its :class:`SpectroBatchSpec` (and once more for a shorter final chunk).
    Results are returned in the catalog order by permuting with the concatenated
    bucket index arrays.
    """
    _refuse_stochastic(model)
    check_template_model(model)
    if calibration_marginalize:
        _refuse_calibration_order(model)
    if not batches:
        raise ValueError("batches is empty; build it with build_spectro_batches")
    if chunk_size is not None and (not isinstance(chunk_size, int) or chunk_size < 1):
        raise ValueError(f"chunk_size must be a positive integer or None, got {chunk_size!r}")

    base_key = jax.random.PRNGKey(seed)
    bucket_keys = jax.random.split(base_key, len(batches))
    buckets = []
    for (spec, batch, index), key in zip(batches, bucket_keys):
        params_u, final_nlp, trace = _fit_bucket(
            model,
            spec,
            batch,
            n_steps=n_steps,
            learning_rate=learning_rate,
            key=key,
            chunk_size=chunk_size,
            calibration_marginalize=calibration_marginalize,
            cal_n_poly=cal_n_poly,
            cal_prior_sigma=cal_prior_sigma,
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


def _bucket_catalog_inputs(
    model, spec, batch, *, calibration_marginalize=False, cal_n_poly=3, cal_prior_sigma=1.0
):
    """Flat log-density and per-galaxy substitution for one bucket's catalog sampler.

    Parameters
    ----------
    model : SEDModel
        Template model (see :func:`batched_neg_log_posterior`).
    spec : SpectroBatchSpec
        Static key of the bucket.
    batch : SpectroBatch
        The bucket's galaxies, with a leading batch axis.

    Returns
    -------
    flat : tuple
        ``(log_post_2arg, unravel_fn, init_flat, template_data_args)`` in the form
        :func:`~tengri.inference.backends.mcmc.catalog.build_catalog_mcmc_engine`
        takes as ``flat_logdensity``. ``log_post_2arg(x, data_args)`` is the
        log-posterior of the standardized latents ``x`` for the galaxy in
        ``data_args["obs"]``.
    substitute : callable
        ``(data, noise, presence, redshift, line_flux_obs, line_flux_err) ->
        {"obs": data}``, the per-galaxy ``data_args`` the engine substitutes.

    Notes
    -----
    Build-time helper, not JIT-compatible. The unravel order is that of
    :func:`jax.flatten_util.ravel_pytree` on one galaxy's latents, which sorts the
    names, so every flat position maps back to the same named latent.
    """
    nlp = batched_neg_log_posterior(
        model,
        spec,
        calibration_marginalize=calibration_marginalize,
        cal_n_poly=cal_n_poly,
        cal_prior_sigma=cal_prior_sigma,
    )
    init_one = {n: v[0] for n, v in init_unbounded_batch(model, 1, jax.random.PRNGKey(0)).items()}
    init_flat, unravel_fn = ravel_pytree(init_one)

    def log_post_2arg(position, data_args):
        return -nlp(unravel_fn(position), data_args["obs"])

    def substitute(data, noise, presence, redshift, line_flux_obs, line_flux_err):
        return {"obs": data}

    return (log_post_2arg, unravel_fn, init_flat, {"obs": batch.galaxy(0)}), substitute


def _flat_rows(latents: dict) -> jax.Array:
    """Stack a dict of ``(n_gal,)`` latents into ``(n_gal, D)`` flat rows."""
    return jax.vmap(lambda d: ravel_pytree(d)[0])(latents)


def _bucket_starts(model, n_gal: int, key, map_params, index: np.ndarray) -> jax.Array:
    """Flat sampler start, ``(n_gal, D)``, for one bucket.

    With ``map_params`` the MAP values of this bucket's galaxies are standardized
    back to latents. A row that is not finite falls back to a prior draw, so a
    failed MAP never starts a chain at NaN.
    """
    prior_rows = _flat_rows(init_unbounded_batch(model, n_gal, key))
    if map_params is None:
        return prior_rows
    latent_spec = _latent_spec(model)
    mapped = {
        n: latent_spec.get_distribution(n).standardize(
            jnp.asarray(np.asarray(map_params[n])[index])
        )
        for n in _free_latent_names(model)
    }
    map_rows = _flat_rows(mapped)
    keep = jnp.all(jnp.isfinite(map_rows), axis=1, keepdims=True)
    return jnp.where(keep, map_rows, prior_rows)


def fit_spectra_catalog_mcmc(
    model,
    batches: Sequence[tuple[SpectroBatchSpec, SpectroBatch, np.ndarray]],
    *,
    sampler: str = "nuts",
    n_warmup: int = 200,
    n_burnin: int = 0,
    n_samples: int = 200,
    seed: int = 0,
    init: str = "map",
    map_steps: int = 200,
    batch_size: int | None = None,
    calibration_marginalize: bool = False,
    cal_n_poly: int = 3,
    cal_prior_sigma: float = 1.0,
    **engine_kwargs,
) -> dict[str, Any]:
    """Sample every galaxy's spectrum posterior with the catalog MCMC engine.

    Parameters
    ----------
    model : SEDModel
        Template model with a free redshift (see
        :func:`~tengri.forward.batched_spectrum.check_template_model`).
    batches : sequence of (SpectroBatchSpec, SpectroBatch, ndarray)
        Output of :func:`~tengri.observation.batched.build_spectro_batches` built
        with ``model=model``.
    sampler : {"nuts", "hmc", "chees"}, optional
        Sampler handed to
        :func:`~tengri.inference.backends.mcmc.catalog.build_catalog_mcmc_engine`.
        Default ``"nuts"``.
    n_warmup : int, optional
        Adaptation steps, run per galaxy. Default 200.
    n_burnin : int, optional
        Post-warmup draws discarded per galaxy. Default 0.
    n_samples : int, optional
        Draws kept per galaxy. Default 200.
    seed : int, optional
        PRNG seed for the MAP, the prior starts and the sampler. Default 0.
    init : {"map", "prior"}, optional
        ``"map"`` warm-starts every galaxy from :func:`fit_spectra_map_vmap`;
        ``"prior"`` draws starts from :func:`init_unbounded_batch`. Default
        ``"map"``.
    map_steps : int, optional
        Adam steps of the MAP warm start. Ignored when ``init="prior"``. Default 200.
    batch_size : int, optional
        Galaxies run together per ``jax.lax.map`` step, under one ``vmap``
        (passed as ``batch_size`` to ``lax.map``). ``None`` (default) runs the
        galaxies one at a time. On a CPU a small value or ``None`` is right: a wide
        ``vmap`` of NUTS trees waits for its deepest tree and costs memory. On a
        GPU a large value is what buys parallel throughput. The draws do not
        depend on this value, up to floating-point reduction order.
    calibration_marginalize : bool, optional
        Marginalize a Chebyshev calibration polynomial out of the spectral term, as
        :func:`batched_log_likelihood` does. Also applies to the MAP warm start.
        Default False.
    cal_n_poly : int, optional
        Calibration coefficients when ``calibration_marginalize``. Default 3.
    cal_prior_sigma : float, optional
        Prior standard deviation of each calibration coefficient. Default 1.0.
    **engine_kwargs
        Further keyword arguments of ``build_catalog_mcmc_engine``, such as
        ``target_accept_rate`` or ``use_dense``.

    Returns
    -------
    dict
        ``"draws"``: dict of str to ndarray, shape ``(N, n_draws)``. The physical
        posterior draws of each free parameter other than redshift, in the
        original catalog order (``N`` is the total galaxy count). With
        ``eline_mode="fitted"`` the ``eline_amp_*`` line amplitudes are included.
        ``"divergences"``: ndarray of int, shape ``(N,)``. Divergent transitions
        per galaxy, in the same order.

    Raises
    ------
    NotImplementedError
        If the model's SFH is stochastic.
    ValueError
        If the template fails :func:`check_template_model`, ``batches`` is empty,
        the model has no free parameter other than redshift, or ``init`` is not
        ``"map"`` or ``"prior"``.

    Notes
    -----
    Build-time Python driver, not JIT-compatible; the per-galaxy sampler is.
    Each bucket compiles once for its :class:`SpectroBatchSpec`. Every galaxy
    adapts its own step size and mass matrix, so a galaxy's draws do not depend on
    the galaxies it was batched with. The sampled space is the standardized latent
    space of :func:`batched_neg_log_posterior`, whose prior is iid N(0, 1), and the
    draws are unstandardized to physical values only on return.
    """
    _refuse_stochastic(model)
    check_template_model(model)
    if init not in ("map", "prior"):
        raise ValueError(f"init must be 'map' or 'prior', got {init!r}")
    if not batches:
        raise ValueError("batches is empty; build it with build_spectro_batches")
    if not _free_latent_names(model):
        raise ValueError("the model has no free parameter other than redshift to sample")

    map_params = None
    if init == "map":
        map_params = fit_spectra_map_vmap(
            model,
            batches,
            n_steps=map_steps,
            seed=seed,
            calibration_marginalize=calibration_marginalize,
            cal_n_poly=cal_n_poly,
            cal_prior_sigma=cal_prior_sigma,
        )["params"]

    bucket_keys = jax.random.split(jax.random.PRNGKey(seed), len(batches))
    per_bucket = []
    for (spec, batch, index), bucket_key in zip(batches, bucket_keys):
        index = np.asarray(index, dtype=int)
        n_gal = int(batch.z.shape[0])
        flat, substitute = _bucket_catalog_inputs(
            model,
            spec,
            batch,
            calibration_marginalize=calibration_marginalize,
            cal_n_poly=cal_n_poly,
            cal_prior_sigma=cal_prior_sigma,
        )
        unravel_fn = flat[1]
        start_key, sample_key = jax.random.split(bucket_key)
        starts = _bucket_starts(model, n_gal, start_key, map_params, index)
        run_one, _ = build_catalog_mcmc_engine(
            None,
            sampler,
            n_warmup=n_warmup,
            n_burnin=n_burnin,
            n_samples=n_samples,
            flat_logdensity=flat,
            substitute=substitute,
            **engine_kwargs,
        )

        def one_galaxy(xs, run_one=run_one):
            x0, gal_key, obs = xs
            return run_one(x0, gal_key, obs, None, None, None, None, None)

        positions, divergent = jax.lax.map(
            one_galaxy,
            (starts, jax.random.split(sample_key, n_gal), batch),
            batch_size=batch_size,
        )
        n_draws = positions.shape[1]
        latents = jax.vmap(unravel_fn)(positions.reshape(-1, positions.shape[-1]))
        physical = _to_physical(model, latents)
        per_bucket.append(
            {
                "index": index,
                "draws": {n: np.asarray(v).reshape(n_gal, n_draws) for n, v in physical.items()},
                "divergences": np.asarray(divergent).sum(axis=1),
            }
        )

    catalog_index = np.concatenate([b["index"] for b in per_bucket])
    order = np.argsort(catalog_index, kind="stable")
    names = _free_latent_names(model)
    draws = {n: np.concatenate([b["draws"][n] for b in per_bucket], axis=0)[order] for n in names}
    divergences = np.concatenate([b["divergences"] for b in per_bucket])[order]
    return {"draws": draws, "divergences": divergences}
