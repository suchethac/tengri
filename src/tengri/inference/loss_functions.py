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

from tengri.inference.likelihoods.gaussian import standardized_residual

__all__ = [
    "build_loglikelihood_fn",
    "build_loglikelihood_unbounded_fn",
    "build_logprior_fn",
    "build_loss_fn",
    "standardized_neg_log_prior",
]

# ── Shared helpers (called inside traced closures) ───────────────────────


def standardized_neg_log_prior(
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
    has_line_ratios=False,
    measured_line_defs=None,
    jit_inputs=None,
    threaded_impl=None,
):
    """Forward-model prediction in one place.

    Returns ``(prediction_dict, predicted, pred_phot, pred_spec)``:

    - ``prediction_dict``: keys ``phot_fnu`` / ``spec_fnu`` / optional
      ``line_fluxes`` / ``indices``, fed to user / auto-built Likelihood adapters.
    - ``predicted``: concatenated array (legacy χ² fall-through still uses this).
    - ``pred_phot`` / ``pred_spec``: split components or ``None``.

    **JIT data threading.** When ``jit_inputs`` (the ``data_args["_jit_inputs"]``
    bundle of ``ssp_data`` / ``template_data`` / ``fixed_values``) and
    ``threaded_impl`` (``model._get_or_build_predict_observables_jit()``) are
    supplied, the photometry / spectroscopy / joint channels route through the
    threaded orchestrator and the feature channels thread ``predict_state``. This
    keeps the SSP grid and template arrays as XLA ``Parameter`` ops in the outer
    inference trace instead of closure-captured ``Constant`` ops, see
    :func:`_build_data_neg_log_likelihood_fn`. ``use_components`` keeps its own
    component-split path (not threaded here).
    """
    # Single threaded forward for phot/spec/joint: one orchestrator call
    # returns an Observables carrying every configured channel. ``_obs`` is
    # None when threading is unavailable (dummy models, ``use_components``) or
    # when ``jit_inputs`` was not built, then we fall back to the eager
    # ``model.predict_*`` accessors (which closure-capture the SSP grid, so
    # this path bakes it; acceptable for the non-inference / dummy-model uses).
    _obs = None
    if threaded_impl is not None and jit_inputs is not None and not use_components:
        _obs = threaded_impl(
            params,
            jit_inputs["fixed_values"],
            jit_inputs["ssp_data"],
            jit_inputs["template_data"],
        )

    if data_type == "photometry":
        if _obs is not None:
            predicted = _obs.phot_fnu
        elif use_components:
            predicted = model._photometry_via_state(params)
        else:
            predicted = model.predict_photometry(params)
        pred_phot, pred_spec = predicted, None
    elif data_type == "spectroscopy":
        if _obs is not None:
            predicted = _obs.spec_fnu
        elif use_components:
            predicted = model._spectrum_via_state(params)
        else:
            predicted = model.predict_spectrum(params)
        pred_phot, pred_spec = None, predicted
    elif data_type == "joint":
        if _obs is not None:
            pred_phot, pred_spec = _obs.phot_fnu, _obs.spec_fnu
        elif use_components:
            pred_phot = model._photometry_via_state(params)
            pred_spec = model._spectrum_via_state(params)
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

    # Line fluxes, line ratios, and spectral indices all derive from the SAME
    # orchestrator forward, so ``predict_state`` is computed ONCE when a feature
    # channel needs it and threaded into each predictor, a joint fit with lines +
    # ratios + indices runs the full-grid forward once per loss eval, not three
    # times.
    #
    # But ``predict_state`` *is* the full-grid forward, and that is exactly what
    # the emission-line precompute (``approx=FeaturePrecomp()``) exists to skip:
    # the window LUT reconstructs the lines from SED-free SFH weights. Calling
    # predict_state anyway would leave the LUT wrapped around the cost it was
    # meant to avoid, a fast path that is a no-op. So when lines are the only
    # feature channel and the model carries the precompute, the full-grid forward
    # is not built at all.
    # NB: reading ``model.approx.feature_precomp`` here instead looks like the
    # obvious fix for the flag/config disagreement below, and it is a PESSIMIZATION:
    # it forces ``needs_state=False``, and computing the lines separately costs more
    # than sharing one ``predict_state`` with the photometry channel. Measured on a
    # 10-parameter Cue model: 5,021,451 -> 5,859,984 gradient FLOPs (+16.7%).
    fast_lines = bool(getattr(model, "_fast_line_measurement", False))
    needs_state = has_line_ratios or has_indices or (has_line_fluxes and not fast_lines)
    if not needs_state:
        feature_state = None
    elif jit_inputs is not None:
        # Thread the SSP grid + template arrays so the feature forward
        # (line fluxes / ratios / indices) does not bake them into the outer
        # inference trace either.
        feature_state = model.predict_state(
            params,
            fixed_values=jit_inputs["fixed_values"],
            ssp_data=jit_inputs["ssp_data"],
            template_data=jit_inputs["template_data"],
        )
    else:
        feature_state = model.predict_state(params)
    if has_line_fluxes:
        if measured_line_defs is not None:
            # No discrete line catalog (BakedIn) → measure the fluxes off the
            # model spectrum the way a pipeline does (measure_line_fluxes). With
            # FeaturePrecomp the measurement runs against the SSP window LUT.
            prediction["line_fluxes"] = model.measure_line_fluxes(
                params, measured_line_defs, approx=fast_lines, state=feature_state
            )
        else:
            prediction["line_fluxes"] = model.predict_line_fluxes(
                params, target_wavelengths=data_args["line_flux_waves"], state=feature_state
            )
    if has_line_ratios:
        prediction["line_ratios"] = model.predict_line_ratios(
            params, model.observation.line_ratios, state=feature_state
        )
    if has_indices:
        prediction["indices"] = model.predict_spectral_indices(
            params, index_defs, state=feature_state
        )

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
    data array, not yet expressible as a single-channel adapter); any
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
    has_line_ratios = "line_ratio_obs" in fitter._data_args
    has_indices = "index_obs" in fitter._data_args
    index_defs = None
    if has_indices:
        obs_for_idx = getattr(model, "observation", None)
        if obs_for_idx is not None and obs_for_idx.spectral_indices is not None:
            index_defs = obs_for_idx.spectral_indices.index_defs
    # Line-flux channel: backends with no discrete catalog (BakedIn) can't run
    # predict_line_fluxes, measure the fluxes off the spectrum instead. Build the
    # continuum windows ONCE from the observation's concrete line centers (never
    # from traced data_args) so the jitted loss sees a static LineDef set.
    measured_line_defs = None
    if has_line_fluxes and not model._has_line_catalog():
        obs_for_lines = getattr(model, "observation", None)
        lf = getattr(obs_for_lines, "line_fluxes", None) if obs_for_lines is not None else None
        if lf is not None:
            import numpy as _np

            from tengri.observation.line_measurement import default_line_defs

            measured_line_defs = default_line_defs(_np.asarray(lf.wavelengths), tuple(lf.names))
    user_likelihood = getattr(fitter, "_user_likelihood", None)
    use_components = bool(getattr(fitter, "use_components", False))
    # Build-time signature check: the internal adapter cohort accepts
    # ``data_args`` (call-time data threading); user-supplied likelihoods
    # with the two-argument signature keep working on their baked arrays.
    if user_likelihood is not None:
        import inspect

        _likelihood_takes_data_args = (
            "data_args" in inspect.signature(user_likelihood.log_prob).parameters
        )
    else:
        _likelihood_takes_data_args = False

    # Template threading (#250 follow-up, generalized): route every channel
    # through the threaded orchestrator ``_impl`` so the outer JIT trace
    # (HMC/NUTS/VI/MAP loss_fn) sees ssp_data + template_data + fixed_values as
    # outer-level Parameters. Without threading, the outer JIT inlines
    # ``model.predict_photometry`` / ``predict_spectrum`` / ``predict_state`` →
    # ``predict_observables_jit`` and bakes the SSP grid (15×93×5994 floats) into
    # the HLO as a constant, ballooning cold compile to ~40 s. Threaded path:
    # cold ≈ 3-5 s. Originally photometry-only; now covers spectroscopy, joint,
    # and the feature channels too (the SSP bake was identical on those paths,
    # the fast path just never reached them). ``_build_prediction`` reads the
    # threaded arrays out of ``data_args["_jit_inputs"]`` at call time.
    _threaded_impl = (
        model._get_or_build_predict_observables_jit()
        if not use_components and hasattr(model, "_get_or_build_predict_observables_jit")
        else None
    )

    # Eager channel-scale pre-check (#1495): evaluate every likelihood channel
    # once, at a reference parameter draw, against the SAME prediction dict the
    # traced loss builds below, and refuse to hand back a loss whose channels
    # cannot coexist on a representable scale. A units-mismatched channel
    # produces a chi-squared that silently annihilates every other channel via
    # floating-point absorption; this converts that into a loud construction
    # error naming the channel. Runs once per loss build, outside JIT.
    if user_likelihood is not None:
        import jax as _jax

        from tengri.inference.composite_likelihood import (
            CompositeLikelihood,
            _check_channel_scales,
        )

        _ref_params = dict(spec.sample(_jax.random.PRNGKey(0)))
        _ref_prediction, _, _, _ = _build_prediction(
            model,
            _ref_params,
            data_type,
            has_line_fluxes=has_line_fluxes,
            has_indices=has_indices,
            index_defs=index_defs,
            data_args=fitter._data_args,
            use_components=use_components,
            has_line_ratios=has_line_ratios,
            measured_line_defs=measured_line_defs,
        )
        _channels = (
            user_likelihood.likelihoods
            if isinstance(user_likelihood, CompositeLikelihood)
            else (user_likelihood,)
        )
        _check_channel_scales(_channels, _ref_prediction, _ref_params, fitter._data_args)

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
            has_line_ratios=has_line_ratios,
            measured_line_defs=measured_line_defs,
            jit_inputs=data_args.get("_jit_inputs"),
            threaded_impl=_threaded_impl,
        )

        # Auto-built / user-supplied Likelihood adapter handles the data
        # term (and any extras composed via CompositeLikelihood).
        # ``data_args`` is forwarded so adapters read the CURRENT
        # Fitter's data, the compiled loss is shared across Fitters
        # (get_or_build_cached), and adapter-baked arrays would XLA-bake
        # the first galaxy's data into every subsequent fit.
        if user_likelihood is not None:
            if _likelihood_takes_data_args:
                return -user_likelihood.log_prob(prediction, params, data_args=data_args)
            return -user_likelihood.log_prob(prediction, params)

        # Legacy χ² fall-through. Exactly one case reaches this code path:
        # data_mask + spec/joint (censoring across the
        # concatenated data array, not addressable via a single-channel
        # adapter). All other configurations are covered by auto-build
        # in Fitter._maybe_build_default_likelihood. If you find yourself
        # hitting the AssertionError below, the auto-build cohort is
        # missing coverage for your case, extend it there, don't add a
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
                standardized_residual(
                    data_args["line_flux_obs"], model_lf, data_args["line_flux_err"]
                )
                ** 2
            )
            e_lh = e_lh + 0.5 * chi2_lines
        if has_line_ratios:
            model_lr = model.predict_line_ratios(params, model.observation.line_ratios)
            chi2_ratios = jnp.sum(
                standardized_residual(
                    data_args["line_ratio_obs"], model_lr, data_args["line_ratio_err"]
                )
                ** 2
            )
            e_lh = e_lh + 0.5 * chi2_ratios
        if has_indices:
            model_idx = model.predict_spectral_indices(params, index_defs)
            chi2_idx = jnp.sum(
                standardized_residual(data_args["index_obs"], model_idx, data_args["index_err"])
                ** 2
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
    neg_log_lik = _build_data_neg_log_likelihood_fn(fitter)

    def loss_fn(params_unbounded, data_args):
        """Compute loss: -log_lik + ½ξᵀξ prior on standardized params."""
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
        return e_lh + standardized_neg_log_prior(
            params_unbounded,
            free_names,
            stochastic=stochastic,
            centering=field_centering,
            # Physical sigma, read from the unstandardized dict: at a < 1 the
            # latent prior depends on it, so it has to be the sampled value and
            # not a constant.
            psd_sigma_dex=(params.get("sfh_field_psd_sigma") if field_centering != 1.0 else None),
        )

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
