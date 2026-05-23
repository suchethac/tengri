# SPDX-License-Identifier: BSD-3-Clause
"""Loss and log-likelihood builders consumed by the Fitter inference engines.

Three public builders — :func:`build_loss_fn`, :func:`build_loglikelihood_fn`,
and :func:`build_loglikelihood_unbounded_fn` — are thin wrappers over a
single private core, :func:`_build_data_neg_log_likelihood_fn`. The core
routes through ``fitter._user_likelihood`` (the likelihood-adapter cohort
auto-built by :meth:`Fitter._maybe_build_default_likelihood`) and falls
through to one tiny legacy χ² branch only for the case the cohort cannot
yet express (``data_mask + non-photometry``).

Closures returned from each builder pull Fitter state into local
variables at construction time, so they hold no reference to the Fitter
itself and can be reused across galaxies that share the same Model.
"""

from __future__ import annotations

import jax.numpy as jnp

__all__ = [
    "build_loglikelihood_fn",
    "build_loglikelihood_unbounded_fn",
    "build_logprior_fn",
    "build_loss_fn",
]

# ── Shared helpers (called inside traced closures) ───────────────────────


def _unstandardize_parameters(params_unbounded, spec, free_names, fixed_values, stochastic):
    """Convert unbounded ξ → physical params, merge fixed values, attach psd_xi, resolve mirrors.

    The single source of truth for the standardized → physical transform.
    Each distribution's unstandardize() method implements h(ξ) such that
    P(h(ξ)) |dh/dξ| = φ(ξ), giving Jacobian cancellation in the Hamiltonian.
    """
    params = {}
    for name in free_names:
        dist = spec.get_distribution(name)
        params[name] = dist.unstandardize(params_unbounded[name])
    for name, val in fixed_values.items():
        params[name] = val
    if stochastic and "psd_xi" in params_unbounded:
        params["psd_xi"] = params_unbounded["psd_xi"]
    return spec.resolve_mirrors(params)


def _build_prediction(
    model,
    params,
    data_type,
    *,
    has_line_fluxes,
    has_indices,
    index_defs,
    data_args,
    use_components=False,
):
    """Forward-model prediction in one place.

    Returns ``(prediction_dict, predicted, pred_phot, pred_spec)``:

    - ``prediction_dict``: keys ``phot_fnu`` / ``spec_fnu`` / optional
      ``line_fluxes`` / ``indices`` — fed to user / auto-built Likelihood adapters.
    - ``predicted``: concatenated array (legacy χ² fall-through still uses this).
    - ``pred_phot`` / ``pred_spec``: split components or ``None``.
    """
    if data_type == "photometry":
        if use_components:
            predicted = model.predict_photometry_components(params)
        else:
            predicted = model.predict_photometry(params)
        pred_phot, pred_spec = predicted, None
    elif data_type == "spectroscopy":
        if use_components:
            predicted = model.predict_spectrum_components(params)
        else:
            predicted = model.predict_spectrum(params)
        pred_phot, pred_spec = None, predicted
    elif data_type == "joint":
        if use_components:
            pred_phot = model.predict_photometry_components(params)
            pred_spec = model.predict_spectrum_components(params)
        else:
            pred_phot = model.predict_photometry(params)
            pred_spec = model.predict_spectrum(params)
        predicted = jnp.concatenate([pred_phot, pred_spec])
    else:
        raise ValueError(f"Unknown data_type: {data_type}")

    prediction = {}
    if pred_phot is not None:
        prediction["phot_fnu"] = pred_phot
    if pred_spec is not None:
        prediction["spec_fnu"] = pred_spec
    if has_line_fluxes:
        prediction["line_fluxes"] = model.predict_line_fluxes(
            params, target_wavelengths=data_args["line_flux_waves"]
        )
    if has_indices:
        prediction["indices"] = model.predict_spectral_indices(params, index_defs)

    return prediction, predicted, pred_phot, pred_spec


# ── Builders ─────────────────────────────────────────────────────────────


def _build_data_neg_log_likelihood_fn(fitter):
    """Build the data-term function ``neg_log_lik(params, data_args) -> -log p(d|params)``.

    Single source of truth for the data term. All three public builders
    (:func:`build_loss_fn`, :func:`build_loglikelihood_fn`,
    :func:`build_loglikelihood_unbounded_fn`) compose around this so they
    cannot drift in sign / formula / branch coverage.

    Routes through ``fitter._user_likelihood`` when set (the adapter
    cohort built by :meth:`Fitter._maybe_build_default_likelihood`). The
    only configuration that still falls through to the inline χ² is
    ``data_mask + non-photometry`` (censoring across the concatenated
    data array — not yet expressible as a single-channel adapter); any
    other configuration that reaches the fall-through hits an explicit
    ``AssertionError`` so missing auto-build coverage is loud, not
    silent.

    Takes **physical** parameters; the unstandardize transform belongs
    in the wrappers.
    """
    from tengri.observation.noise import (
        censored_neg_log_likelihood,
        get_noise_dof,
        uses_student_t,
    )

    model = fitter.model
    data_type = fitter.data_type
    spec = fitter.spec
    noise_dof = get_noise_dof(spec) if uses_student_t(spec) else None
    use_censored = fitter.data_mask is not None
    has_line_fluxes = "line_flux_waves" in fitter._data_args
    has_indices = "index_obs" in fitter._data_args
    index_defs = None
    if has_indices:
        obs_for_idx = getattr(model, "observation", None)
        if obs_for_idx is not None and obs_for_idx.spectral_indices is not None:
            index_defs = obs_for_idx.spectral_indices.index_defs
    user_likelihood = getattr(fitter, "_user_likelihood", None)
    use_components = bool(getattr(fitter, "use_components", False))

    def neg_log_lik(params, data_args):
        """-log p(d | params). Caller supplies physical params + data_args."""
        data = data_args["data"]
        noise = data_args["noise"]

        prediction, predicted, _pred_phot, _pred_spec = _build_prediction(
            model,
            params,
            data_type,
            has_line_fluxes=has_line_fluxes,
            has_indices=has_indices,
            index_defs=index_defs,
            data_args=data_args,
            use_components=use_components,
        )

        # Auto-built / user-supplied Likelihood adapter handles the data
        # term (and any extras composed via CompositeLikelihood).
        if user_likelihood is not None:
            return -user_likelihood.log_prob(prediction, params)

        # Legacy χ² fall-through. Exactly one case reaches this code path:
        # data_mask + spec/joint (censoring across the
        # concatenated data array, not addressable via a single-channel
        # adapter). All other configurations are covered by auto-build
        # in Fitter._maybe_build_default_likelihood. If you find yourself
        # hitting the AssertionError below, the auto-build cohort is
        # missing coverage for your case — extend it there, don't add a
        # branch here.
        if use_censored:
            mask = data_args["data_mask"]
            f_cal = params.get("noise_frac_cal", 0.0)
            e_lh = censored_neg_log_likelihood(
                data, noise, predicted, mask, f_cal=f_cal, dof=noise_dof
            )
        else:
            raise AssertionError(
                "Legacy χ² fall-through reached for an unexpected configuration. "
                "Auto-build (Fitter._maybe_build_default_likelihood) returned None "
                "but use_censored is False. Either extend the auto-build cohort to "
                "cover this configuration, or raise NotImplementedError at the "
                "auto-build bail-out site rather than falling through here."
            )

        # Line-flux / spectral-index extras for the censored fall-through.
        # The user-likelihood path composes these via CompositeLikelihood;
        # the censored path needs them inlined since it bypasses the cohort.
        if has_line_fluxes:
            model_lf = model.predict_line_fluxes(
                params, target_wavelengths=data_args["line_flux_waves"]
            )
            chi2_lines = jnp.sum(
                ((data_args["line_flux_obs"] - model_lf) / data_args["line_flux_err"]) ** 2
            )
            e_lh = e_lh + 0.5 * chi2_lines
        if has_indices:
            model_idx = model.predict_spectral_indices(params, index_defs)
            chi2_idx = jnp.sum(
                ((data_args["index_obs"] - model_idx) / data_args["index_err"]) ** 2
            )
            e_lh = e_lh + 0.5 * chi2_idx

        return e_lh

    return neg_log_lik


def build_loss_fn(fitter):
    """Build the information Hamiltonian (loss function) from Fitter state.

    Constructs a JAX-differentiable loss function encapsulating the likelihood
    and isotropic Gaussian prior on standardized parameters. The loss function
    takes unbounded parameters and observed data as separate arguments so the
    compiled XLA program can be reused across galaxies with the same model
    structure.

    The Hamiltonian is:

    .. math::

        \\mathcal{H}(\\boldsymbol{\\xi} \\mid \\mathbf{d}) =
        \\frac{1}{2}\\chi^2(\\mathbf{d}, \\mathbf{f}(\\mathbf{h}(\\boldsymbol{\\xi})))
        + \\frac{1}{2}\\boldsymbol{\\xi}^\\top\\boldsymbol{\\xi}

    where :math:`\\mathbf{h}(\\boldsymbol{\\xi})` are the per-parameter
    transforms from standardized to physical space, and :math:`\\chi^2`
    includes contributions from photometry, spectroscopy, emission lines,
    calibration, and optional noise models.

    Parameters
    ----------
    fitter : Fitter
        Fitter instance with model, data, parameters, and configuration.

    Returns
    -------
    callable
        ``loss_fn(params_unbounded, data_args) -> scalar``

        Parameters:

        - ``params_unbounded`` : dict — Standardized parameters ξ (any real values).
        - ``data_args`` : dict — Observed data, noise, and noise models
          from ``fitter._data_args``.

        Returns scalar loss value suitable for optimization/sampling.

    Notes
    -----
    **Standardized parameterization**: The returned loss function works in
    standardized (unbounded) space where all parameters follow N(0,1) prior.
    Physical bounds and distributions are handled via per-parameter transforms.
    This ensures the prior always cancels to the isotropic quadratic term
    :math:`\\frac{1}{2}\\boldsymbol{\\xi}^\\top\\boldsymbol{\\xi}` regardless
    of prior type (Uniform, Gaussian, LogUniform, etc.).

    **Likelihood dispatch**: The data term ``-log p(d|params)`` is computed
    by :func:`_build_data_neg_log_likelihood_fn`, which routes through the
    auto-built :class:`Likelihood` adapter cohort (see
    :meth:`Fitter._maybe_build_default_likelihood`). Each physically
    distinct configuration — diagonal Gaussian, Student-t / variable
    noise, censored photometry, multivariate Gaussian, calibration
    marginalisation, emission-line marginalisation (flat or Cloudy
    prior), explicitly-fitted line amplitudes, and the combined
    calibration + e-line marginalisation — is handled by a dedicated
    adapter, not by branches in this builder.

    **Data as explicit arguments**: Observed data, noise, and noise models are
    passed via ``data_args`` (not captured in closure). This allows multiple
    Fitters on the same Model to share compiled XLA programs.

    **JIT compatibility**: The returned function is fully JAX-compatible and
    safe inside :func:`jax.jit`, :func:`jax.grad`, :func:`jax.value_and_grad`.

    References
    ----------
    .. [1] Standardized parameterization derivation and Jacobian cancellation:
       Section 2.2 and Appendix A of the tengri methods paper.
    """
    free_names = fitter._free_names
    fixed_values = fitter._fixed_values
    spec = fitter.spec
    stochastic = spec.stochastic
    neg_log_lik = _build_data_neg_log_likelihood_fn(fitter)

    def loss_fn(params_unbounded, data_args):
        """Compute loss: -log_lik + ½ξᵀξ prior on standardized params."""
        params = _unstandardize_parameters(
            params_unbounded, spec, free_names, fixed_values, stochastic
        )
        e_lh = neg_log_lik(params, data_args)

        # Prior contributions (IFT Hamiltonian).
        #
        # The information Hamiltonian in standardized space is:
        #   H(ξ|d) = ½χ²(d, h(ξ)) + ½ξᵀξ
        #
        # Each distribution implements unstandardize(ξ) = h(ξ) such that
        # P(h(ξ)) |dh/dξ| = φ(ξ), so the change-of-variables Jacobian
        # cancels the prior density and leaves the isotropic quadratic
        # term. Exact for Uniform, Gaussian, LogUniform, LogNormal,
        # StudentT priors.  Reference: tengri paper §2.2 + Appendix A.
        # Sum across batched per-galaxy axes too: hierarchical fits
        # (PopulationSEDModel) have per-galaxy xi with shape (N,) or
        # higher rank. The prior penalty must reduce to a scalar so
        # the loss is a single number. ``jnp.sum`` is a no-op for
        # rank-0 xi (single-galaxy fits) and reduces (N,) → scalar
        # for hierarchical.
        prior_penalty = 0.0
        for name in free_names:
            prior_penalty = prior_penalty + jnp.sum(params_unbounded[name] ** 2)
        if stochastic and "psd_xi" in params_unbounded:
            prior_penalty = prior_penalty + jnp.sum(params_unbounded["psd_xi"] ** 2)
        return e_lh + 0.5 * prior_penalty

    return loss_fn


def build_logprior_fn(fitter):
    """Build log-prior function in physical parameter space.

    Evaluates the joint prior density over all free parameters in their
    physical (original) space. Automatically dispatches to vectorized
    computation for all-Uniform priors (major speedup) or general loop
    for mixed-type priors.

    Parameters
    ----------
    fitter : Fitter
        Fitter instance with ``spec`` (Parameters) and ``_free_names``.

    Returns
    -------
    callable
        ``logprior_fn(free_params) -> scalar``

        where ``free_params`` is a dict of physical parameter values.
        Returns log-density (or ``-inf`` if outside bounds).

    Notes
    -----
    **Vectorization optimization**: When all free parameters have Uniform
    priors, uses JAX vectorized operations for ~367× speedup (D=8 case).
    For mixed Uniform/Gaussian/LogUniform priors, falls back to scalar loop.

    **Bounds handling**: Uniform priors return ``-inf`` if any parameter
    falls outside bounds; all other distributions use their continuous
    pdf/cdf definitions and can have finite density at boundaries.

    Examples
    --------
    >>> fitter = Fitter(model, data, noise)
    >>> logprior = build_logprior_fn(fitter)
    >>> lp = logprior({"stellar_mass": 11.0, "age_gyr": 1.5})
    """
    from tengri.parameters.priors import Uniform

    spec = fitter.spec
    free_names = fitter._free_names

    # Check if all distributions are Uniform
    all_uniform = all(isinstance(spec.get_distribution(name), Uniform) for name in free_names)

    if all_uniform and len(free_names) > 0:
        # Vectorized uniform prior for significant speedup
        # Extract bounds once at build time
        lower_bounds = jnp.array([fitter._bounds[name][0] for name in free_names])
        upper_bounds = jnp.array([fitter._bounds[name][1] for name in free_names])
        widths = upper_bounds - lower_bounds
        log_widths_sum = jnp.sum(jnp.log(widths))

        def logprior_fn(free_params):
            """Fast vectorized log prior for Uniform case: -log(widths) if in bounds."""
            # Stack parameter values into array
            param_values = jnp.array([free_params[name] for name in free_names])
            # Vectorized bounds check
            in_bounds = jnp.all((param_values >= lower_bounds) & (param_values <= upper_bounds))
            return jnp.where(in_bounds, -log_widths_sum, -jnp.inf)

    else:
        # General case: loop over distributions (mixed Uniform/Gaussian/etc)
        def logprior_fn(free_params):
            """Sum log probabilities from each distribution (Uniform, Gaussian, etc)."""
            lp = 0.0
            for name in free_names:
                dist = spec.get_distribution(name)
                lp = lp + dist.log_prob(free_params[name])
            return lp

    return logprior_fn


def build_loglikelihood_fn(fitter):
    """Build log-likelihood function in physical parameter space.

    Constructs a JAX-differentiable function that computes the log-likelihood
    of observed data given physical parameters. Fixed parameters are
    automatically merged, and observed data are passed separately from the
    closure to enable XLA compilation reuse across galaxies.

    Parameters
    ----------
    fitter : Fitter
        Fitter instance with model, data, parameters, and likelihood config.

    Returns
    -------
    callable
        ``loglikelihood_fn(free_params, data_args) -> scalar``

        where ``free_params`` is dict of physical parameters and ``data_args``
        is the data argument bundle from ``fitter._data_args``.

    Notes
    -----
    Thin wrapper over :func:`_build_data_neg_log_likelihood_fn`: merges
    fixed values, resolves mirrored parameters, then negates the data
    term to return ``+log p(d|params)``. Dispatch over likelihood
    variants (Gaussian, Student-t, censored, multivariate Gaussian,
    calibration / e-line marginalisation, fitted line amplitudes) is
    handled by the auto-built :class:`Likelihood` adapter cohort in
    :meth:`Fitter._maybe_build_default_likelihood`, not by branches
    here.

    **Data as explicit arguments**: Observed data, measurement uncertainties,
    and noise models are passed via ``data_args`` (not captured in closure).
    This allows multiple Fitters using the same Model to share compiled XLA
    programs and run on different galaxies.

    **JIT compatibility**: The returned function is fully JAX-compatible and
    safe inside :func:`jax.jit`, :func:`jax.grad`, :func:`jax.value_and_grad`.

    See Also
    --------
    build_loss_fn : Returns likelihood + isotropic prior (Hamiltonian).
    build_logprior_fn : Returns only prior log-density.
    """
    fixed_values = fitter._fixed_values
    spec = fitter.spec
    neg_log_lik = _build_data_neg_log_likelihood_fn(fitter)

    def loglikelihood_fn(free_params, data_args):
        """Compute log p(d | params) — physical params, no prior."""
        params = dict(free_params)
        for name, val in fixed_values.items():
            params[name] = val
        params = spec.resolve_mirrors(params)
        return -neg_log_lik(params, data_args)

    return loglikelihood_fn


def build_loglikelihood_unbounded_fn(fitter):
    """Build a log-likelihood function in unbounded parameter space.

    For Elliptical Slice Sampling, which handles the N(0,I) prior
    internally.  Returns ``loglik(params_unbounded, data_args)``.

    Parameters
    ----------
    fitter : Fitter

    Returns
    -------
    callable
        ``loglik_unbounded(params_unbounded, data_args) -> scalar``

    Notes
    -----
    Composes directly around :func:`_build_data_neg_log_likelihood_fn`,
    matching the shape of :func:`build_loss_fn` and
    :func:`build_loglikelihood_fn`. All three wrappers share the same
    data-term core so they cannot drift in sign / formula / branch
    coverage.
    """
    free_names = fitter._free_names
    fixed_values = fitter._fixed_values
    spec = fitter.spec
    stochastic = spec.stochastic
    neg_log_lik = _build_data_neg_log_likelihood_fn(fitter)

    def loglik_unbounded(params_unbounded, data_args):
        """Compute log-likelihood after unstandardizing unbounded parameters to physical space."""
        params = _unstandardize_parameters(
            params_unbounded, spec, free_names, fixed_values, stochastic
        )
        return -neg_log_lik(params, data_args)

    return loglik_unbounded
