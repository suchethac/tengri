"""Loss and log-likelihood builders extracted from Fitter.

These are module-level functions that take a Fitter instance and return
compiled callables.  They were originally methods on Fitter; extracted here
to keep fitter.py under the 800-line project limit and to make each builder
independently testable.

The builders follow the same alias-then-close pattern: they pull all Fitter
state into local variables at the top so that the returned closures are
self-contained and never hold a reference to the Fitter.
"""

from __future__ import annotations

import jax.numpy as jnp

from tengri.inference.likelihoods.gaussian import diag_gaussian_chi2

__all__ = [
    "build_loglikelihood_fn",
    "build_loglikelihood_unbounded_fn",
    "build_logprior_fn",
    "build_loss_fn",
]

# ── Shared helpers (called inside traced closures) ───────────────────────


def _build_eline_G_eff(params, fixed_values, model, eline_wavelengths, constraint_matrix):
    """Build emission line design matrix with doublet constraints applied."""
    from tengri.observation.eline_marginalization import (
        apply_doublet_constraints,
        build_eline_design_matrix,
    )

    z = params.get("redshift", fixed_values.get("redshift", 0.0))
    sigma_kms = params.get("eline_sigma_kms", 0.0)
    delta_v = params.get("eline_delta_v_kms", 0.0)
    resolution = getattr(model, "_spectral_resolution", None) or 2000.0
    G = build_eline_design_matrix(
        model._wave_obs,
        eline_wavelengths,
        resolution,
        z,
        eline_sigma_kms=sigma_kms,
        eline_delta_v_kms=delta_v,
    )
    return apply_doublet_constraints(G, constraint_matrix)


def _marginalize_elines(
    residual,
    noise,
    G_eff,
    params,
    fixed_values,
    prior_type,
    prior_sigma,
    prior_width_dex,
    independent_wavelengths,
):
    """Analytically marginalize emission line amplitudes (cloudy or flat prior)."""
    from tengri.observation.eline_marginalization import marginalize_emission_lines

    if prior_type == "cloudy":
        from tengri.observation.eline_priors import marginalize_emission_lines_cloudy

        log_z = params.get("met_logzsol", fixed_values.get("met_logzsol", 0.0))
        neb_logU = params.get("neb_logU", fixed_values.get("neb_logU", -3.0))
        return marginalize_emission_lines_cloudy(
            residual,
            noise,
            G_eff,
            log_z=log_z,
            neb_logU=neb_logU,
            line_wavelengths=independent_wavelengths,
            prior_width_dex=prior_width_dex,
        )
    prior_var = jnp.full(G_eff.shape[1], prior_sigma**2)
    return marginalize_emission_lines(residual, noise, G_eff, prior_variance=prior_var)


def _split_joint_data(data, noise, predicted, n_phot):
    """Split concatenated [photometry, spectroscopy] arrays at n_phot."""
    return (
        data[:n_phot],
        data[n_phot:],
        noise[:n_phot],
        noise[n_phot:],
        predicted[:n_phot],
        predicted[n_phot:],
    )


def _calibration_log_likelihood(predicted, data, noise, wave_obs, n_poly, prior_sigma):
    """Log-likelihood with calibration polynomial marginalized out."""
    from tengri.observation.calibration import marginalize_calibration

    log_like, _c_hat, _c_err = marginalize_calibration(
        predicted, data, noise, wave_obs, n_poly=n_poly, prior_sigma=prior_sigma
    )
    return log_like


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
    model, params, data_type, mode, *, has_line_fluxes, has_indices, index_defs, data_args
):
    """Forward-model prediction in one place.

    Returns ``(prediction_dict, predicted, pred_phot, pred_spec)``:

    - ``prediction_dict``: keys ``phot_fnu`` / ``spec_fnu`` / optional
      ``line_fluxes`` / ``indices`` — fed to user / auto-built Likelihood adapters.
    - ``predicted``: concatenated array (legacy χ² fall-through still uses this).
    - ``pred_phot`` / ``pred_spec``: split components or ``None``.
    """
    if data_type == "photometry":
        predicted = model.predict_photometry(params, mode=mode)
        pred_phot, pred_spec = predicted, None
    elif data_type == "spectroscopy":
        predicted = model.predict_spectrum(params, model._wave_obs, mode=mode)
        pred_phot, pred_spec = None, predicted
    elif data_type == "joint":
        pred_phot = model.predict_photometry(params, mode=mode)
        pred_spec = model.predict_spectrum(params, model._wave_obs, mode=mode)
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
        prediction["indices"] = model.predict_spectral_indices(params, index_defs, mode=mode)

    return prediction, predicted, pred_phot, pred_spec


# ── Builders ─────────────────────────────────────────────────────────────


def _build_data_neg_log_likelihood_fn(fitter, mode="_traceable"):
    """Build the data-term function ``neg_log_lik(params, data_args) -> -log p(d|params)``.

    Single source of truth for the data term. Both :func:`build_loss_fn`
    and :func:`build_loglikelihood_fn` compose around this so the two
    cannot drift in sign / formula / branch coverage.

    Routes through ``fitter._user_likelihood`` when set (the Phase II-1
    Likelihood-adapter cohort built by
    :meth:`Fitter._maybe_build_default_likelihood`). Otherwise falls
    through to the legacy χ² switch — which still covers the cases the
    auto-build does not yet handle (cloudy-prior e-line marginalisation,
    explicitly-fitted line amplitudes, joint/spec + variable_noise, and a
    handful of edge bail-outs).

    Takes **physical** parameters; the unstandardize transform belongs
    in the wrappers.
    """
    from tengri.observation.noise import (
        censored_log_likelihood,
        get_noise_dof,
        has_noise_model,
        uses_student_t,
        variable_noise_hamiltonian,
    )

    model = fitter.model
    data_type = fitter.data_type
    fixed_values = fitter._fixed_values
    spec = fitter.spec
    use_variable_noise = has_noise_model(spec)
    noise_dof = get_noise_dof(spec) if uses_student_t(spec) else None
    use_censored = fitter.data_mask is not None
    use_cal_marg = fitter._calibration_marginalize and fitter._has_spectroscopy
    cal_n_poly = fitter._cal_n_poly
    cal_prior_sigma = fitter._cal_prior_sigma
    use_eline_marg = fitter._eline_marginalize
    use_eline_fitted = fitter._eline_fitted
    eline_wavelengths = fitter._eline_wavelengths
    eline_independent_wavelengths = fitter._eline_independent_wavelengths
    eline_constraint_matrix = fitter._eline_constraint_matrix
    eline_prior_type = fitter._eline_prior_type
    eline_prior_sigma = fitter._eline_prior_sigma
    eline_prior_width_dex = fitter._eline_prior_width_dex
    eline_amplitude_names = fitter._eline_amplitude_names
    has_spec_cov = "spec_cov_inv" in fitter._data_args
    has_line_fluxes = "line_flux_waves" in fitter._data_args
    has_indices = "index_obs" in fitter._data_args
    index_defs = None
    if has_indices:
        obs_for_idx = getattr(model, "observation", None)
        if obs_for_idx is not None and obs_for_idx.spectral_indices is not None:
            index_defs = obs_for_idx.spectral_indices.index_defs
    n_phot = 0
    if has_spec_cov and data_type == "joint":
        obs = getattr(model, "observation", None)
        if obs is not None:
            n_phot = obs.n_data_phot
    user_likelihood = getattr(fitter, "_user_likelihood", None)

    def neg_log_lik(params, data_args):
        """-log p(d | params). Caller supplies physical params + data_args."""
        data = data_args["data"]
        noise = data_args["noise"]

        prediction, predicted, pred_phot, _ = _build_prediction(
            model,
            params,
            data_type,
            mode,
            has_line_fluxes=has_line_fluxes,
            has_indices=has_indices,
            index_defs=index_defs,
            data_args=data_args,
        )

        # Auto-built / user-supplied Likelihood adapter handles the data
        # term (and any extras composed via CompositeLikelihood).
        if user_likelihood is not None:
            return -user_likelihood.log_prob(prediction, params)

        # Legacy χ² fall-through — only fires when auto-build returned
        # None (cloudy/fitted e-lines, joint+variable_noise, edge bail-outs).
        # Most-specific branches first.
        if use_eline_marg and use_cal_marg and data_type == "spectroscopy":
            G_eff = _build_eline_G_eff(
                params, fixed_values, model, eline_wavelengths, eline_constraint_matrix
            )
            _, a_hat, _ = _marginalize_elines(
                data - predicted,
                noise,
                G_eff,
                params,
                fixed_values,
                eline_prior_type,
                eline_prior_sigma,
                eline_prior_width_dex,
                eline_independent_wavelengths,
            )
            pred_with_lines = predicted + G_eff @ a_hat
            ll_spec = _calibration_log_likelihood(
                pred_with_lines, data, noise, model._wave_obs, cal_n_poly, cal_prior_sigma
            )
            e_lh = -ll_spec
        elif use_eline_marg and use_cal_marg and data_type == "joint":
            G_eff = _build_eline_G_eff(
                params, fixed_values, model, eline_wavelengths, eline_constraint_matrix
            )
            n_p = pred_phot.shape[0]
            data_phot, data_spec, noise_phot, noise_spec, p_phot, p_spec = _split_joint_data(
                data, noise, predicted, n_p
            )
            _, a_hat, _ = _marginalize_elines(
                data_spec - p_spec,
                noise_spec,
                G_eff,
                params,
                fixed_values,
                eline_prior_type,
                eline_prior_sigma,
                eline_prior_width_dex,
                eline_independent_wavelengths,
            )
            p_spec_with_lines = p_spec + G_eff @ a_hat
            chi2_phot = diag_gaussian_chi2(p_phot, data_phot, noise_phot)
            ll_spec = _calibration_log_likelihood(
                p_spec_with_lines,
                data_spec,
                noise_spec,
                model._wave_obs,
                cal_n_poly,
                cal_prior_sigma,
            )
            e_lh = 0.5 * chi2_phot - ll_spec
        elif use_cal_marg and data_type == "spectroscopy":
            ll_spec = _calibration_log_likelihood(
                predicted, data, noise, model._wave_obs, cal_n_poly, cal_prior_sigma
            )
            e_lh = -ll_spec
        elif use_cal_marg and data_type == "joint":
            n_p = pred_phot.shape[0]
            data_phot, data_spec, noise_phot, noise_spec, p_phot, p_spec = _split_joint_data(
                data, noise, predicted, n_p
            )
            chi2_phot = diag_gaussian_chi2(p_phot, data_phot, noise_phot)
            ll_spec = _calibration_log_likelihood(
                p_spec, data_spec, noise_spec, model._wave_obs, cal_n_poly, cal_prior_sigma
            )
            e_lh = 0.5 * chi2_phot - ll_spec
        elif use_eline_marg and data_type == "spectroscopy":
            G_eff = _build_eline_G_eff(
                params, fixed_values, model, eline_wavelengths, eline_constraint_matrix
            )
            ln_l_eline, _, _ = _marginalize_elines(
                data - predicted,
                noise,
                G_eff,
                params,
                fixed_values,
                eline_prior_type,
                eline_prior_sigma,
                eline_prior_width_dex,
                eline_independent_wavelengths,
            )
            e_lh = -ln_l_eline
        elif use_eline_marg and data_type == "joint":
            G_eff = _build_eline_G_eff(
                params, fixed_values, model, eline_wavelengths, eline_constraint_matrix
            )
            n_p = pred_phot.shape[0]
            data_phot, data_spec, noise_phot, noise_spec, p_phot, p_spec = _split_joint_data(
                data, noise, predicted, n_p
            )
            ln_l_eline, _, _ = _marginalize_elines(
                data_spec - p_spec,
                noise_spec,
                G_eff,
                params,
                fixed_values,
                eline_prior_type,
                eline_prior_sigma,
                eline_prior_width_dex,
                eline_independent_wavelengths,
            )
            chi2_phot = diag_gaussian_chi2(p_phot, data_phot, noise_phot)
            e_lh = 0.5 * chi2_phot - ln_l_eline
        elif use_eline_fitted and data_type == "spectroscopy":
            G_eff = _build_eline_G_eff(
                params, fixed_values, model, eline_wavelengths, eline_constraint_matrix
            )
            a = jnp.array([params[nm] for nm in eline_amplitude_names])
            predicted_with_lines = predicted + G_eff @ a
            chi2 = diag_gaussian_chi2(predicted_with_lines, data, noise)
            e_lh = 0.5 * chi2
        elif use_eline_fitted and data_type == "joint":
            G_eff = _build_eline_G_eff(
                params, fixed_values, model, eline_wavelengths, eline_constraint_matrix
            )
            n_p = pred_phot.shape[0]
            data_phot, data_spec, noise_phot, noise_spec, p_phot, p_spec = _split_joint_data(
                data, noise, predicted, n_p
            )
            a = jnp.array([params[nm] for nm in eline_amplitude_names])
            p_spec_with_lines = p_spec + G_eff @ a
            chi2_phot = diag_gaussian_chi2(p_phot, data_phot, noise_phot)
            chi2_spec = diag_gaussian_chi2(p_spec_with_lines, data_spec, noise_spec)
            e_lh = 0.5 * (chi2_phot + chi2_spec)
        elif use_censored:
            mask = data_args["data_mask"]
            f_cal = params.get("noise_frac_cal", 0.0)
            e_lh = censored_log_likelihood(
                data, noise, predicted, mask, f_cal=f_cal, dof=noise_dof
            )
        elif use_variable_noise:
            f_cal = params.get("noise_frac_cal", 0.0)
            e_lh = variable_noise_hamiltonian(data, noise, predicted, f_cal, dof=noise_dof)
        elif has_spec_cov and data_type == "spectroscopy":
            diff = data - predicted
            chi2 = diff @ data_args["spec_cov_inv"] @ diff
            e_lh = 0.5 * chi2
        elif has_spec_cov and data_type == "joint":
            data_phot = data[:n_phot]
            p_phot = predicted[:n_phot]
            noise_phot = noise[:n_phot]
            diff_spec = data[n_phot:] - predicted[n_phot:]
            chi2_phot = diag_gaussian_chi2(p_phot, data_phot, noise_phot)
            chi2_spec = diff_spec @ data_args["spec_cov_inv"] @ diff_spec
            e_lh = 0.5 * (chi2_phot + chi2_spec)
        else:
            chi2 = diag_gaussian_chi2(predicted, data, noise)
            e_lh = 0.5 * chi2

        # Line-flux / spectral-index constraints (legacy fall-through only;
        # the user-likelihood path composes these via CompositeLikelihood).
        if has_line_fluxes:
            model_lf = model.predict_line_fluxes(
                params, target_wavelengths=data_args["line_flux_waves"]
            )
            chi2_lines = jnp.sum(
                ((data_args["line_flux_obs"] - model_lf) / data_args["line_flux_err"]) ** 2
            )
            e_lh = e_lh + 0.5 * chi2_lines
        if has_indices:
            model_idx = model.predict_spectral_indices(params, index_defs, mode=mode)
            chi2_idx = jnp.sum(
                ((data_args["index_obs"] - model_idx) / data_args["index_err"]) ** 2
            )
            e_lh = e_lh + 0.5 * chi2_idx

        return e_lh

    return neg_log_lik


def build_loss_fn(fitter, mode="_traceable"):
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

    mode : str, optional
        Forward model prediction mode. Default ``"_traceable"`` is safe
        inside JIT scopes (used by NIFTy geoVI). Use ``"auto"`` (~1.5×
        speedup) for non-JIT methods (MAP, Laplace, Pathfinder, NUTS,
        Ray Tracing, NSS). See
        :doc:`docs/dev/jit-optimization-report-2026-04-18.md`.

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

    **Likelihood variants**: Automatically branches based on Fitter config:

    - Photometry: χ²(data, prediction)
    - Spectroscopy with calibration marginalization: Integrated over polynomial
    - Emission lines (marginalized): Integrated over line amplitudes
    - Emission lines (fitted): Amplitudes are explicit parameters
    - Joint photometry + spectroscopy: Separate χ² per component
    - Noise models: Student-t or other likelihoods if configured

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
    from tengri.inference.loss_functions import _build_data_neg_log_likelihood_fn

    free_names = fitter._free_names
    fixed_values = fitter._fixed_values
    spec = fitter.spec
    stochastic = spec.stochastic
    neg_log_lik = _build_data_neg_log_likelihood_fn(fitter, mode=mode)

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
        prior_penalty = 0.0
        for name in free_names:
            prior_penalty = prior_penalty + params_unbounded[name] ** 2
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


def build_loglikelihood_fn(fitter, mode="_traceable"):
    """Build log-likelihood function in physical parameter space.

    Constructs a JAX-differentiable function that computes the log-likelihood
    of observed data given physical parameters. Fixed parameters are
    automatically merged, and observed data are passed separately from the
    closure to enable XLA compilation reuse across galaxies.

    Parameters
    ----------
    fitter : Fitter
        Fitter instance with model, data, parameters, and likelihood config.

    mode : str, optional
        Forward model prediction mode. Default ``"_traceable"`` is safe
        inside JIT (used by NIFTy geoVI). Use ``"auto"`` (~1.5× speedup)
        for non-JIT methods (MAP, Laplace, Pathfinder, NUTS, Ray Tracing, NSS).

    Returns
    -------
    callable
        ``loglikelihood_fn(free_params, data_args) -> scalar``

        where ``free_params`` is dict of physical parameters and ``data_args``
        is the data argument bundle from ``fitter._data_args``.

    Notes
    -----
    **Likelihood variants**: Automatically branches based on data_type and
    Fitter config:

    - **Photometry**: χ² = Σ(d_i - f_i)²/σ_i²
    - **Spectroscopy with calibration**: Marginalizes Chebyshev poly
    - **Emission lines (marginalized)**: Integrates line amplitudes analytically
    - **Emission lines (fitted)**: Line amplitude parameters in χ²
    - **Joint photometry + spectroscopy**: Separate χ² per component
    - **Spectral covariance**: Uses precision matrix instead of diagonal noise
    - **Noise models**: Student-t or other likelihoods if configured

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
    neg_log_lik = _build_data_neg_log_likelihood_fn(fitter, mode=mode)

    def loglikelihood_fn(free_params, data_args):
        """Compute log p(d | params) — physical params, no prior."""
        params = dict(free_params)
        for name, val in fixed_values.items():
            params[name] = val
        params = spec.resolve_mirrors(params)
        return -neg_log_lik(params, data_args)

    return loglikelihood_fn


def build_loglikelihood_unbounded_fn(fitter, mode="_traceable"):
    """Build a log-likelihood function in unbounded parameter space.

    For Elliptical Slice Sampling, which handles the N(0,I) prior
    internally.  Returns ``loglik(params_unbounded, data_args)``.

    Parameters
    ----------
    fitter : Fitter
    mode : str, optional
        Forward model prediction mode. Default "_traceable" is safe inside
        JIT scopes (used by NIFTy VI/geoVI). Use "auto" for better performance
        with MAP, Laplace, Pathfinder, NUTS, Raytrace, NSS (~1.5x speedup).

    Returns
    -------
    callable
        ``loglik_unbounded(params_unbounded, data_args) -> scalar``
    """
    loglik_fn = fitter._get_or_build_loglikelihood_fn(mode=mode)
    free_names = fitter._free_names
    fixed_values = fitter._fixed_values
    spec = fitter.spec

    def loglik_unbounded(params_unbounded, data_args):
        """Compute log-likelihood after unstandardizing unbounded parameters to physical space."""
        params = _unstandardize_parameters(
            params_unbounded, spec, free_names, fixed_values, spec.stochastic
        )
        return loglik_fn(params, data_args)

    return loglik_unbounded
