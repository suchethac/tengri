# SPDX-License-Identifier: BSD-3-Clause
"""Loss and log-likelihood builders consumed by the Fitter inference engines.

Three public builders, :func:`build_loss_fn`, :func:`build_loglikelihood_fn`,
and :func:`build_loglikelihood_unbounded_fn`, are thin wrappers over a
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

from tengri.components.stellar.sfh.young_fraction import (
    YOUNG_FRACTION_NAMES,
    young_fraction_latents,
    young_fraction_prior_for_spec,
)

__all__ = [
    "build_loglikelihood_fn",
    "build_loglikelihood_unbounded_fn",
    "build_logprior_fn",
    "build_loss_fn",
    "standardized_neg_log_prior",
]

# ── Shared helpers (called inside traced closures) ───────────────────────


def standardized_neg_log_prior(
    params_unbounded,
    free_names,
    *,
    stochastic,
    centering=1.0,
    psd_sigma_dex=None,
    young_fraction=None,
):
    r"""Negative log-prior of the standardized latents, up to a constant.

    Adds the young-fraction term of the continuity SFH when the latents carry
    ``sfh_cont_young_frac_k``; see :func:`_standardized_neg_log_prior_core` for
    the iid part and the GP-field term.

    Parameters
    ----------
    young_fraction : YoungFractionPrior or None, optional
        The exact pullback of the ratio prior to the young fractions, from
        :func:`~tengri.components.stellar.sfh.young_fraction.young_fraction_prior_for_spec`.
        Required whenever ``params_unbounded`` carries the young-fraction
        latents; omitting it there raises rather than dropping the term.

    Returns
    -------
    ndarray, scalar
        Negative log-prior [nats], rank 0.

    Raises
    ------
    ConfigError
        If the young-fraction latents are present and ``young_fraction`` is None.

    Notes
    -----
    **JIT/grad/vmap-safe**: pure reductions and a ``jnp.where`` box test.
    """
    base = _standardized_neg_log_prior_core(
        params_unbounded,
        free_names,
        stochastic=stochastic,
        centering=centering,
        psd_sigma_dex=psd_sigma_dex,
    )
    has_young = any(name in params_unbounded for name in YOUNG_FRACTION_NAMES)
    if not has_young:
        return base
    if young_fraction is None:
        from tengri.config.exceptions import ConfigError

        raise ConfigError(
            "the continuity young-fraction latents are present but no young-fraction "
            "prior was passed to standardized_neg_log_prior; without it the Student-t "
            "ratio prior is dropped. Pass young_fraction=young_fraction_prior_for_spec(spec)."
        )
    log_density = young_fraction.log_density_from_latents(young_fraction_latents(params_unbounded))
    return base - log_density


def _standardized_neg_log_prior_core(
    params_unbounded, free_names, *, stochastic, centering=1.0, psd_sigma_dex=None
):
    r"""Negative log-prior of the standardized latents, up to a constant.

    Every prior is standardized, so the prior on the unbounded latents is
    :math:`\mathcal{N}(0, I)` and this is

    .. math::

        -\log p(\xi) = \tfrac{1}{2} \sum_i \xi_i^2 + \text{const.}

    Parameters
    ----------
    params_unbounded : dict of str to array_like
        Latents in unbounded (standardized) space. Entries may be scalars for a
        single galaxy or shape ``(n_gal,)`` for a hierarchical fit.
    free_names : sequence of str
        Names of the free scalar parameters to include.
    stochastic : bool
        Whether the model carries a stochastic-SFH field. When ``True`` and
        ``"psd_xi"`` is present, the field latents are included.
    centering : float, optional
        Field parameterization ``a`` in ``[0, 1]`` [dimensionless]. Default
        ``1.0``, the standardized map, where the field term is
        :math:`\tfrac12 \xi^\top \xi` exactly as before. At ``a < 1`` the field
        latent's prior is :math:`\mathcal{N}(0, \sigma_s^{2-2a} I)` and this
        term is replaced by
        :func:`~tengri.components.stellar.sfh.gp_sfh.drw_latent_log_prior`.
    psd_sigma_dex : array_like, optional
        Physical modulation amplitude :math:`\sigma` [dex]. Required when
        ``centering < 1``, the latent prior depends on it, which is precisely
        what partial centering trades away. Ignored at ``a = 1``.

    Returns
    -------
    ndarray, scalar
        :math:`\tfrac12 \sum \xi^2` [dimensionless]. **Always rank 0**, whatever
        the rank of the inputs.

    Notes
    -----
    **JIT/grad/vmap-safe**: pure reductions, called inside traced closures.

    The reduction to a scalar is the whole reason this is a function rather than
    two inline loops. Hierarchical fits (``PopulationSEDModel``) carry per-galaxy
    parameters of shape ``(n_gal,)`` or higher rank; without ``jnp.sum`` on every
    term the "penalty" keeps that shape, and an objective that is a vector either
    broadcasts into nonsense or fails somewhere unrelated to the cause.
    ``InferenceContext.log_prior_fn`` restated this rule without the reduction and
    returned shape ``(n_gal,)`` against a docstring promising a scalar, harmless
    only because nothing in ``src/`` called it. This is also the single seam a
    change of field parameterization has to touch (#1355).
    """
    penalty = jnp.zeros(())
    for name in free_names:
        penalty = penalty + jnp.sum(params_unbounded[name] ** 2)
    if stochastic and "psd_xi" in params_unbounded:
        if float(centering) != 1.0:
            # The prior travels with the map (#1355). At a < 1 the latent is no
            # longer standardized: its variance is sigma_s^(2-2a), and the
            # -n(1-a) log sigma_s normalizer couples it to a SAMPLED parameter,
            # so omitting it does not shift the posterior by a constant, it
            # changes the sigma marginal, silently, at every a.
            if psd_sigma_dex is None:
                raise ValueError(
                    "centering < 1 needs psd_sigma_dex: the field latent's prior "
                    "is N(0, sigma_s^(2-2a) I), which depends on the sampled "
                    "amplitude. Passing None would silently target the "
                    "standardized prior with a partially-centered map (#1355)."
                )
            from tengri.components.stellar.sfh.gp_sfh import drw_latent_log_prior

            # 0.5 * xi^2 is a NEGATIVE log-prior up to a constant; the helper
            # returns a normalized LOG-prior, so it enters negated.
            return penalty * 0.5 - drw_latent_log_prior(
                params_unbounded["psd_xi"], psd_sigma_dex, centering=centering
            )
        penalty = penalty + jnp.sum(params_unbounded["psd_xi"] ** 2)
    return 0.5 * penalty


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
        # Publish under BOTH names. The sampler's vector keys the GP latents
        # ``psd_xi``, but ``StellarSEDComponent`` reads ``sfh_field_xi`` and
        # silently falls back to ``jnp.zeros(n_grid)`` when it is absent
        # (components/stellar/component.py). Attaching only ``psd_xi`` therefore
        # pinned the GP field to zero for the whole fit: no exception, no
        # warning, just exp(0 - K0/2) = constant and a likelihood with exactly
        # zero gradient w.r.t. the latents. The burstiness degrees of freedom
        # were sampled from their prior and never reached the SED.
        params["psd_xi"] = params_unbounded["psd_xi"]
        params["sfh_field_xi"] = params_unbounded["psd_xi"]
    return spec.resolve_mirrors(params)


from tengri.inference.data_term import (  # noqa: F401 (re-exported for callers)
    DataTermOptions,
    _build_prediction,
    _resolve_measured_line_defs,
    build_data_term,
)

# ── Builders ─────────────────────────────────────────────────────────────


def _build_data_neg_log_likelihood_fn(fitter):
    """Return ``neg_log_lik(params, data_args) -> -log p(d|params)`` for a Fitter-like object.

    Thin adapter over :func:`~tengri.inference.data_term.build_data_term`. The structural
    options are read with defaults, so a bare stand-in works; the term itself is positive
    and is negated here for the minimizers.
    """
    term = build_data_term(
        fitter.model,
        spec=fitter.spec,
        data_args=fitter._data_args,
        options=DataTermOptions.from_fitter(fitter),
        likelihood=getattr(fitter, "_user_likelihood", None),
    )

    def neg_log_lik(params, data_args):
        """-log p(d | params). Caller supplies physical params + data_args."""
        return -term.log_prob(params, data_args)

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

        - ``params_unbounded`` : dict, Standardized parameters ξ (any real values).
        - ``data_args`` : dict, Observed data, noise, and noise models
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
    distinct configuration, diagonal Gaussian, Student-t / variable
    noise, censored photometry, multivariate Gaussian, calibration
    marginalization, emission-line marginalization (flat or Cloudy
    prior), explicitly-fitted line amplitudes, and the combined
    calibration + e-line marginalization, is handled by a dedicated
    adapter, not by branches in this builder.

    **Data as explicit arguments**: Observed data, noise, and noise models are
    passed via ``data_args`` (not captured in closure). This allows multiple
    Fitters on the same Model to share compiled XLA programs.

    **JIT compatibility**: The returned function is fully JAX-compatible and
    safe inside :func:`jax.jit`, :func:`jax.grad`, :func:`jax.value_and_grad`.

    **Extra log-prior hook**: when ``fitter._extra_log_prior`` is not
    ``None`` (set via ``Fitter(..., extra_log_prior=...)``), the returned
    ``loss_fn`` additionally calls ``fitter.model.predict_state(params)`` on
    the resolved physical parameters and SUBTRACTS
    ``fitter._extra_log_prior(params, state)`` from the loss (subtracted,
    not added, because this function returns a quantity to MINIMIZE, unlike
    :func:`build_logprior_fn`'s ``logprior_fn`` which returns a log-prior
    directly and therefore adds the term). This is the SAME hook honored by
    :func:`build_logprior_fn`, wired into both objectives so MAP, VI, and
    MCMC (all of which minimize :attr:`~tengri.inference.context.
    InferenceContext.neg_log_posterior_fn`, this function cached) see it,
    not only nested sampling / evidence. See
    :class:`~tengri.inference.fitter.Fitter`'s ``extra_log_prior``
    parameter for the full contract and a worked example.

    References
    ----------
    .. [1] Standardized parameterization derivation and Jacobian cancellation:
       Section 2.2 and Appendix A of the tengri methods paper.
    """
    free_names = fitter._free_names
    fixed_values = fitter._fixed_values
    spec = fitter.spec
    stochastic = spec.stochastic
    # Static: the field parameterization is a build-time choice, so it is read
    # once here rather than per call (#1355).
    field_centering = float(getattr(spec, "field_centering", 1.0))
    young_fraction = young_fraction_prior_for_spec(spec)
    neg_log_lik = _build_data_neg_log_likelihood_fn(fitter)
    # Extra log-prior hook (#[task-7], RULING R10): the SAME fitter._extra_log_prior
    # honored by build_logprior_fn (physical-space prior, nested sampling /
    # evidence) must also reach this objective, since MAP/VI/MCMC all minimize
    # THIS function via InferenceContext.neg_log_posterior_fn -- a hook that
    # only reached one of the two objectives would be a silent no-op on the
    # primary inference backends. See fitter.Fitter's ``extra_log_prior``
    # docstring for the full contract.
    extra_log_prior = getattr(fitter, "_extra_log_prior", None)
    model = fitter.model

    def loss_fn(params_unbounded, data_args):
        """Compute loss: -log_lik + ½ξᵀξ prior on standardized params [- extra_log_prior]."""
        params = _unstandardize_parameters(
            params_unbounded, spec, free_names, fixed_values, stochastic
        )
        # Per-galaxy runtime redshift override (batched catalog path, #1337 phase 2).
        # When the caller threads a redshift through ``data_args`` it replaces the
        # baked fixed value, so ONE compiled program serves every per-galaxy redshift
        # (via the ``catalog_z_range`` ztable LUT). No single-galaxy fit ever sets
        # ``data_args["redshift"]``, so ``params`` is unchanged there and the emitted
        # program is byte-identical.
        if "redshift" in data_args:
            params = {**params, "redshift": data_args["redshift"]}
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
        # The per-galaxy reduction lives in the helper, see its Notes for why a
        # rank-0 result is load-bearing rather than cosmetic.
        loss = e_lh + standardized_neg_log_prior(
            params_unbounded,
            free_names,
            stochastic=stochastic,
            centering=field_centering,
            # Physical sigma, read from the unstandardized dict: at a < 1 the
            # latent prior depends on it, so it has to be the sampled value and
            # not a constant.
            psd_sigma_dex=(params.get("sfh_field_psd_sigma") if field_centering != 1.0 else None),
            young_fraction=young_fraction,
        )
        if extra_log_prior is not None:
            # ``loss`` is a NEGATIVE log posterior (to minimize); extra_log_prior
            # returns a log-prior TERM (to add to the log posterior), so it is
            # SUBTRACTED here -- the opposite sign convention from
            # build_logprior_fn's logprior_fn, which returns (and therefore
            # ADDS) a log-prior directly. Calls predict_state a second time
            # (not threaded through the SSP-grid fast path that neg_log_lik
            # uses, see test_loss_ssp_threading.py); only paid when this hook
            # is actually set.
            # predict_state handles merging Fixed values internally (#2296).
            # Only pass free parameters.
            free_params_dict = {k: v for k, v in params.items() if k in free_names}
            state = model.predict_state(free_params_dict)
            loss = loss - extra_log_prior(params, state)
        return loss

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
        Fitter instance with ``spec`` (Parameters), ``_free_names``, and
        (optionally) an ``extra_log_prior`` callable set at construction
        (see :class:`~tengri.inference.fitter.Fitter`).

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

    **Extra log-prior hook**: when ``fitter._extra_log_prior`` is not
    ``None``, the returned ``logprior_fn`` additionally (a) merges fixed
    values and resolves mirrored parameters into ``free_params`` to build
    the full physical parameter dict, (b) calls
    ``fitter.model.predict_state(params)`` to get the predicted
    :class:`~tengri.protocols.ForwardState`, and (c) adds
    ``fitter._extra_log_prior(params, state)`` to the per-parameter prior
    density computed above. This is the one seam a composite prior linking
    multiple predicted quantities (e.g. an AGNfitter-style energy-balance
    penalty, :mod:`tengri.agn.priors`) can use without tengri hard-coding any
    specific physics; see :class:`~tengri.inference.fitter.Fitter`'s
    ``extra_log_prior`` parameter for the full contract and a worked example.

    **JIT-compatibility**: the extra term calls ``predict_state``, exactly as
    the likelihood term already does elsewhere in this module, so it is
    JIT/grad-safe under the same conditions (the user's ``extra_log_prior``
    itself must be a pure function of ``(params, state)`` with no
    Python-level branching on traced values).

    Examples
    --------
    >>> fitter = Fitter(model, data, noise)
    >>> logprior = build_logprior_fn(fitter)
    >>> lp = logprior({"stellar_mass": 11.0, "age_gyr": 1.5})
    """
    from tengri.config.exceptions import ConfigError
    from tengri.parameters.priors import Uniform

    spec = fitter.spec
    free_names = fitter._free_names
    extra_log_prior = getattr(fitter, "_extra_log_prior", None)
    if young_fraction_prior_for_spec(spec) is not None:
        raise ConfigError(
            "the young-fraction prior term is not evaluated on the physical-space log-prior "
            "(build_logprior_fn, used by evidence and nested sampling); the Student-t ratio "
            "prior would be dropped. Use coordinates='ratios' on this path."
        )

    # Check if all distributions are Uniform
    all_uniform = all(isinstance(spec.get_distribution(name), Uniform) for name in free_names)

    if all_uniform and len(free_names) > 0:
        # Vectorized uniform prior for significant speedup
        # Extract bounds once at build time
        lower_bounds = jnp.array([fitter._bounds[name][0] for name in free_names])
        upper_bounds = jnp.array([fitter._bounds[name][1] for name in free_names])
        widths = upper_bounds - lower_bounds
        log_widths_sum = jnp.sum(jnp.log(widths))

        def base_logprior_fn(free_params):
            """Fast vectorized log prior for Uniform case: -log(widths) if in bounds."""
            # Stack parameter values into array
            param_values = jnp.array([free_params[name] for name in free_names])
            # Vectorized bounds check
            in_bounds = jnp.all((param_values >= lower_bounds) & (param_values <= upper_bounds))
            return jnp.where(in_bounds, -log_widths_sum, -jnp.inf)

    else:
        # General case: loop over distributions (mixed Uniform/Gaussian/etc)
        def base_logprior_fn(free_params):
            """Sum log probabilities from each distribution (Uniform, Gaussian, etc)."""
            lp = 0.0
            for name in free_names:
                dist = spec.get_distribution(name)
                lp = lp + dist.log_prob(free_params[name])
            return lp

    if extra_log_prior is None:
        return base_logprior_fn

    fixed_values = fitter._fixed_values
    model = fitter.model

    def logprior_fn(free_params):
        """Per-parameter log prior plus the user's ``extra_log_prior`` term."""
        # predict_state handles merging Fixed values internally (#2296).
        # Only pass free parameters.
        state = model.predict_state(free_params)
        # For the extra_log_prior callback, construct the full params dict
        # (mirrors are resolved internally by predict_state)
        params = dict(free_params)
        for name, val in fixed_values.items():
            params[name] = val
        params = spec.resolve_mirrors(params)
        return base_logprior_fn(free_params) + extra_log_prior(params, state)

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
    calibration / e-line marginalization, fitted line amplitudes) is
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
        """Compute log p(d | params), physical params, no prior."""
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
