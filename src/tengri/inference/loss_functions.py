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

from tengri.parameters.priors import Gaussian, LogUniform
from tengri.utils.transforms import to_bounded


def build_loss_fn(fitter, mode="_traceable"):
    """Build a differentiable loss function.

    The loss function takes an unbounded parameter dict **and a
    ``data_args`` dict** and returns a scalar: chi² + prior penalties.
    Observed data are passed as explicit arguments (not captured in
    the closure) so that the compiled XLA program can be reused across
    galaxies sharing the same model structure.

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
        ``loss_fn(params_unbounded, data_args) -> scalar``
    """
    from tengri.observation.noise import (
        get_noise_dof,
        has_noise_model,
        uses_student_t,
        variable_noise_hamiltonian,
    )

    model = fitter.model
    data_type = fitter.data_type
    free_names = fitter._free_names
    bounds = fitter._bounds
    fixed_values = fitter._fixed_values
    spec = fitter.spec
    stochastic = spec.stochastic
    use_variable_noise = has_noise_model(spec)
    noise_dof = get_noise_dof(spec) if uses_student_t(spec) else None
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

    def loss_fn(params_unbounded, data_args):
        data = data_args["data"]
        noise = data_args["noise"]

        # Convert unbounded → physical for free params
        params = {}
        for name in free_names:
            lo, hi = bounds[name]
            params[name] = to_bounded(params_unbounded[name], lo, hi)

        # Merge fixed values
        for name, val in fixed_values.items():
            params[name] = val

        # Add psd_xi if stochastic
        if stochastic and "psd_xi" in params_unbounded:
            params["psd_xi"] = params_unbounded["psd_xi"]

        # Forward model prediction
        if data_type == "photometry":
            predicted = model.predict_photometry(params, mode=mode)
        elif data_type == "spectroscopy":
            predicted = model.predict_spectrum(params, model._wave_obs, mode=mode)
        elif data_type == "joint":
            pred_phot = model.predict_photometry(params, mode=mode)
            pred_spec = model.predict_spectrum(params, model._wave_obs, mode=mode)
            predicted = jnp.concatenate([pred_phot, pred_spec])
        else:
            raise ValueError(f"Unknown data_type: {data_type}")

        # Likelihood energy — ordered most-specific first so combined branches
        # (eline+cal) are not shadowed by the less-specific cal-only branches.
        if use_eline_marg and use_cal_marg and data_type == "spectroscopy":
            # Both: marginalize lines first, then calibration on line-added prediction
            from tengri.observation.calibration import marginalize_calibration
            from tengri.observation.eline_marginalization import (
                apply_doublet_constraints,
                build_eline_design_matrix,
                marginalize_emission_lines,
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
            G_eff = apply_doublet_constraints(G, eline_constraint_matrix)
            if eline_prior_type == "cloudy":
                from tengri.observation.eline_priors import (
                    marginalize_emission_lines_cloudy,
                )

                log_z = params.get("met_logzsol", fixed_values.get("met_logzsol", 0.0))
                neb_logU = params.get("neb_logU", fixed_values.get("neb_logU", -3.0))
                _, a_hat, _ = marginalize_emission_lines_cloudy(
                    data - predicted,
                    noise,
                    G_eff,
                    log_z=log_z,
                    neb_logU=neb_logU,
                    line_wavelengths=eline_independent_wavelengths,
                    prior_width_dex=eline_prior_width_dex,
                )
            else:
                prior_var = jnp.full(G_eff.shape[1], eline_prior_sigma**2)
                _, a_hat, _ = marginalize_emission_lines(
                    data - predicted, noise, G_eff, prior_variance=prior_var
                )
            pred_with_lines = predicted + G_eff @ a_hat
            log_like_spec, _c_hat, _c_err = marginalize_calibration(
                pred_with_lines,
                data,
                noise,
                model._wave_obs,
                n_poly=cal_n_poly,
                prior_sigma=cal_prior_sigma,
            )
            e_lh = -log_like_spec
        elif use_eline_marg and use_cal_marg and data_type == "joint":
            # Joint + both: marginalize lines on spec part, then calibration on spec,
            # standard chi2 for photometry
            from tengri.observation.calibration import marginalize_calibration
            from tengri.observation.eline_marginalization import (
                apply_doublet_constraints,
                build_eline_design_matrix,
                marginalize_emission_lines,
            )

            z = params.get("redshift", fixed_values.get("redshift", 0.0))
            sigma_kms = params.get("eline_sigma_kms", 0.0)
            delta_v = params.get("eline_delta_v_kms", 0.0)
            resolution = getattr(model, "_spectral_resolution", None) or 2000.0
            n_phot = model.predict_photometry(params, mode=mode).shape[0]
            data_phot = data[:n_phot]
            data_spec = data[n_phot:]
            noise_phot = noise[:n_phot]
            noise_spec = noise[n_phot:]
            pred_phot = predicted[:n_phot]
            pred_spec = predicted[n_phot:]
            G = build_eline_design_matrix(
                model._wave_obs,
                eline_wavelengths,
                resolution,
                z,
                eline_sigma_kms=sigma_kms,
                eline_delta_v_kms=delta_v,
            )
            G_eff = apply_doublet_constraints(G, eline_constraint_matrix)
            if eline_prior_type == "cloudy":
                from tengri.observation.eline_priors import (
                    marginalize_emission_lines_cloudy,
                )

                log_z = params.get("met_logzsol", fixed_values.get("met_logzsol", 0.0))
                neb_logU = params.get("neb_logU", fixed_values.get("neb_logU", -3.0))
                _, a_hat, _ = marginalize_emission_lines_cloudy(
                    data_spec - pred_spec,
                    noise_spec,
                    G_eff,
                    log_z=log_z,
                    neb_logU=neb_logU,
                    line_wavelengths=eline_independent_wavelengths,
                    prior_width_dex=eline_prior_width_dex,
                )
            else:
                prior_var = jnp.full(G_eff.shape[1], eline_prior_sigma**2)
                _, a_hat, _ = marginalize_emission_lines(
                    data_spec - pred_spec, noise_spec, G_eff, prior_variance=prior_var
                )
            pred_spec_with_lines = pred_spec + G_eff @ a_hat
            chi2_phot = jnp.sum(((data_phot - pred_phot) / noise_phot) ** 2)
            log_like_spec, _c_hat, _c_err = marginalize_calibration(
                pred_spec_with_lines,
                data_spec,
                noise_spec,
                model._wave_obs,
                n_poly=cal_n_poly,
                prior_sigma=cal_prior_sigma,
            )
            e_lh = 0.5 * chi2_phot - log_like_spec
        elif use_cal_marg and data_type == "spectroscopy":
            # Analytically marginalize over calibration polynomial
            from tengri.observation.calibration import (
                marginalize_calibration,
            )

            log_like_spec, _c_hat, _c_err = marginalize_calibration(
                predicted,
                data,
                noise,
                model._wave_obs,
                n_poly=cal_n_poly,
                prior_sigma=cal_prior_sigma,
            )
            e_lh = -log_like_spec
        elif use_cal_marg and data_type == "joint":
            # Joint: marginalize spectroscopic part, standard chi2 for photometry
            from tengri.observation.calibration import (
                marginalize_calibration,
            )

            n_phot = model.predict_photometry(params, mode=mode).shape[0]
            data_phot = data[:n_phot]
            data_spec = data[n_phot:]
            noise_phot = noise[:n_phot]
            noise_spec = noise[n_phot:]
            pred_phot = predicted[:n_phot]
            pred_spec = predicted[n_phot:]

            chi2_phot = jnp.sum(((data_phot - pred_phot) / noise_phot) ** 2)
            log_like_spec, _c_hat, _c_err = marginalize_calibration(
                pred_spec,
                data_spec,
                noise_spec,
                model._wave_obs,
                n_poly=cal_n_poly,
                prior_sigma=cal_prior_sigma,
            )
            e_lh = 0.5 * chi2_phot - log_like_spec
        elif use_eline_marg and data_type == "spectroscopy":
            # Analytically marginalize emission line amplitudes
            from tengri.observation.eline_marginalization import (
                apply_doublet_constraints,
                build_eline_design_matrix,
                marginalize_emission_lines,
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
            G_eff = apply_doublet_constraints(G, eline_constraint_matrix)
            residual = data - predicted
            if eline_prior_type == "cloudy":
                from tengri.observation.eline_priors import (
                    marginalize_emission_lines_cloudy,
                )

                log_z = params.get("met_logzsol", fixed_values.get("met_logzsol", 0.0))
                neb_logU = params.get("neb_logU", fixed_values.get("neb_logU", -3.0))
                ln_l_eline, _, _ = marginalize_emission_lines_cloudy(
                    residual,
                    noise,
                    G_eff,
                    log_z=log_z,
                    neb_logU=neb_logU,
                    line_wavelengths=eline_independent_wavelengths,
                    prior_width_dex=eline_prior_width_dex,
                )
            else:
                prior_var = jnp.full(G_eff.shape[1], eline_prior_sigma**2)
                ln_l_eline, _, _ = marginalize_emission_lines(
                    residual, noise, G_eff, prior_variance=prior_var
                )
            e_lh = -ln_l_eline
        elif use_eline_marg and data_type == "joint":
            # Joint: marginalize lines on spectroscopic part, standard chi2 for photometry
            from tengri.observation.eline_marginalization import (
                apply_doublet_constraints,
                build_eline_design_matrix,
                marginalize_emission_lines,
            )

            z = params.get("redshift", fixed_values.get("redshift", 0.0))
            sigma_kms = params.get("eline_sigma_kms", 0.0)
            delta_v = params.get("eline_delta_v_kms", 0.0)
            resolution = getattr(model, "_spectral_resolution", None) or 2000.0
            n_phot = model.predict_photometry(params, mode=mode).shape[0]
            data_phot = data[:n_phot]
            data_spec = data[n_phot:]
            noise_phot = noise[:n_phot]
            noise_spec = noise[n_phot:]
            pred_phot = predicted[:n_phot]
            pred_spec = predicted[n_phot:]
            G = build_eline_design_matrix(
                model._wave_obs,
                eline_wavelengths,
                resolution,
                z,
                eline_sigma_kms=sigma_kms,
                eline_delta_v_kms=delta_v,
            )
            G_eff = apply_doublet_constraints(G, eline_constraint_matrix)
            residual_spec = data_spec - pred_spec
            if eline_prior_type == "cloudy":
                from tengri.observation.eline_priors import (
                    marginalize_emission_lines_cloudy,
                )

                log_z = params.get("met_logzsol", fixed_values.get("met_logzsol", 0.0))
                neb_logU = params.get("neb_logU", fixed_values.get("neb_logU", -3.0))
                ln_l_eline, _, _ = marginalize_emission_lines_cloudy(
                    residual_spec,
                    noise_spec,
                    G_eff,
                    log_z=log_z,
                    neb_logU=neb_logU,
                    line_wavelengths=eline_independent_wavelengths,
                    prior_width_dex=eline_prior_width_dex,
                )
            else:
                prior_var = jnp.full(G_eff.shape[1], eline_prior_sigma**2)
                ln_l_eline, _, _ = marginalize_emission_lines(
                    residual_spec, noise_spec, G_eff, prior_variance=prior_var
                )
            chi2_phot = jnp.sum(((data_phot - pred_phot) / noise_phot) ** 2)
            e_lh = 0.5 * chi2_phot - ln_l_eline
        elif use_eline_fitted and data_type == "spectroscopy":
            # Fitted mode: amplitudes are explicit params; add line prediction to continuum
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
            G_eff = apply_doublet_constraints(G, eline_constraint_matrix)
            a = jnp.array([params[nm] for nm in eline_amplitude_names])
            predicted_with_lines = predicted + G_eff @ a
            chi2 = jnp.sum(((data - predicted_with_lines) / noise) ** 2)
            e_lh = 0.5 * chi2
        elif use_eline_fitted and data_type == "joint":
            # Joint fitted: add lines to spectroscopic part, standard chi2 for photometry
            from tengri.observation.eline_marginalization import (
                apply_doublet_constraints,
                build_eline_design_matrix,
            )

            z = params.get("redshift", fixed_values.get("redshift", 0.0))
            sigma_kms = params.get("eline_sigma_kms", 0.0)
            delta_v = params.get("eline_delta_v_kms", 0.0)
            resolution = getattr(model, "_spectral_resolution", None) or 2000.0
            n_phot = model.predict_photometry(params, mode=mode).shape[0]
            data_phot = data[:n_phot]
            data_spec = data[n_phot:]
            noise_phot = noise[:n_phot]
            noise_spec = noise[n_phot:]
            pred_phot = predicted[:n_phot]
            pred_spec = predicted[n_phot:]
            G = build_eline_design_matrix(
                model._wave_obs,
                eline_wavelengths,
                resolution,
                z,
                eline_sigma_kms=sigma_kms,
                eline_delta_v_kms=delta_v,
            )
            G_eff = apply_doublet_constraints(G, eline_constraint_matrix)
            a = jnp.array([params[nm] for nm in eline_amplitude_names])
            pred_spec_with_lines = pred_spec + G_eff @ a
            chi2_phot = jnp.sum(((data_phot - pred_phot) / noise_phot) ** 2)
            chi2_spec = jnp.sum(((data_spec - pred_spec_with_lines) / noise_spec) ** 2)
            e_lh = 0.5 * (chi2_phot + chi2_spec)
        elif use_variable_noise:
            f_cal = params.get("noise_frac_cal", 0.0)
            e_lh = variable_noise_hamiltonian(data, noise, predicted, f_cal, dof=noise_dof)
        else:
            chi2 = jnp.sum(((data - predicted) / noise) ** 2)
            e_lh = 0.5 * chi2

        # Prior contributions (IFT Hamiltonian: H = -log L + 0.5 * ξᵀξ)
        # All unbounded parameters get a standard normal prior because
        # the sigmoid transform maps N(0,1) → Uniform(lo, hi). Without
        # this term, MCMC chains drift to ±∞ in unbounded space for
        # weakly-constrained parameters.
        prior_penalty = 0.0

        # Standard normal prior on ALL unbounded parameters
        for name in free_names:
            prior_penalty += params_unbounded[name] ** 2

        # Standard normal prior on psd_xi
        if stochastic and "psd_xi" in params_unbounded:
            prior_penalty += jnp.sum(params_unbounded["psd_xi"] ** 2)

        # Additional prior contributions for non-Uniform distributions
        # (replace the implicit N(0,1) → Uniform with the actual prior)
        for name in free_names:
            dist = spec.get_distribution(name)
            if isinstance(dist, Gaussian):
                val = params[name]
                prior_penalty -= 2.0 * dist.log_prob(val)
            elif isinstance(dist, LogUniform):
                val = params[name]
                uniform_lp = -jnp.log(dist.hi - dist.lo)
                prior_penalty -= 2.0 * (dist.log_prob(val) - uniform_lp)

        return e_lh + 0.5 * prior_penalty

    return loss_fn


def build_logprior_fn(fitter):
    """Build a log-prior function in physical parameter space.

    Parameters
    ----------
    fitter : Fitter

    Returns
    -------
    callable
        ``logprior_fn(free_params) -> scalar``
    """
    spec = fitter.spec
    free_names = fitter._free_names

    def logprior_fn(free_params):
        lp = 0.0
        for name in free_names:
            dist = spec.get_distribution(name)
            lp = lp + dist.log_prob(free_params[name])
        return lp

    return logprior_fn


def build_loglikelihood_fn(fitter, mode="_traceable"):
    """Build a log-likelihood function in physical parameter space.

    Returns a function: ``(dict of free params, data_args) → scalar``.
    Fixed parameters are automatically merged.  Observed data are
    passed via ``data_args`` (not captured in the closure) for cache
    reuse across galaxies.

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
        ``loglikelihood_fn(free_params, data_args) -> scalar``
    """
    from tengri.observation.noise import (
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

    def loglikelihood_fn(free_params, data_args):
        data = data_args["data"]
        noise = data_args["noise"]

        # Merge free + fixed
        params = dict(free_params)
        for name, val in fixed_values.items():
            params[name] = val

        # Forward model prediction
        if data_type == "photometry":
            predicted = model.predict_photometry(params, mode=mode)
        elif data_type == "spectroscopy":
            predicted = model.predict_spectrum(params, model._wave_obs, mode=mode)
        elif data_type == "joint":
            pred_phot = model.predict_photometry(params, mode=mode)
            pred_spec = model.predict_spectrum(params, model._wave_obs, mode=mode)
            predicted = jnp.concatenate([pred_phot, pred_spec])
        else:
            raise ValueError(f"Unknown data_type: {data_type}")

        # Log-likelihood — ordered most-specific first so combined branches
        # (eline+cal) are not shadowed by the less-specific cal-only branches.
        if use_eline_marg and use_cal_marg and data_type == "spectroscopy":
            # Both: marginalize lines first, then calibration on line-added prediction
            from tengri.observation.calibration import marginalize_calibration
            from tengri.observation.eline_marginalization import (
                apply_doublet_constraints,
                build_eline_design_matrix,
                marginalize_emission_lines,
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
            G_eff = apply_doublet_constraints(G, eline_constraint_matrix)
            if eline_prior_type == "cloudy":
                from tengri.observation.eline_priors import (
                    marginalize_emission_lines_cloudy,
                )

                log_z = params.get("met_logzsol", fixed_values.get("met_logzsol", 0.0))
                neb_logU = params.get("neb_logU", fixed_values.get("neb_logU", -3.0))
                _, a_hat, _ = marginalize_emission_lines_cloudy(
                    data - predicted,
                    noise,
                    G_eff,
                    log_z=log_z,
                    neb_logU=neb_logU,
                    line_wavelengths=eline_independent_wavelengths,
                    prior_width_dex=eline_prior_width_dex,
                )
            else:
                prior_var = jnp.full(G_eff.shape[1], eline_prior_sigma**2)
                _, a_hat, _ = marginalize_emission_lines(
                    data - predicted, noise, G_eff, prior_variance=prior_var
                )
            pred_with_lines = predicted + G_eff @ a_hat
            log_like_spec, _c_hat, _c_err = marginalize_calibration(
                pred_with_lines,
                data,
                noise,
                model._wave_obs,
                n_poly=cal_n_poly,
                prior_sigma=cal_prior_sigma,
            )
            return log_like_spec
        elif use_eline_marg and use_cal_marg and data_type == "joint":
            # Joint + both: marginalize lines on spec part, then calibration on spec,
            # standard chi2 for photometry
            from tengri.observation.calibration import marginalize_calibration
            from tengri.observation.eline_marginalization import (
                apply_doublet_constraints,
                build_eline_design_matrix,
                marginalize_emission_lines,
            )

            z = params.get("redshift", fixed_values.get("redshift", 0.0))
            sigma_kms = params.get("eline_sigma_kms", 0.0)
            delta_v = params.get("eline_delta_v_kms", 0.0)
            resolution = getattr(model, "_spectral_resolution", None) or 2000.0
            n_phot = model.predict_photometry(params, mode=mode).shape[0]
            data_phot = data[:n_phot]
            data_spec = data[n_phot:]
            noise_phot = noise[:n_phot]
            noise_spec = noise[n_phot:]
            pred_phot = predicted[:n_phot]
            pred_spec = predicted[n_phot:]
            G = build_eline_design_matrix(
                model._wave_obs,
                eline_wavelengths,
                resolution,
                z,
                eline_sigma_kms=sigma_kms,
                eline_delta_v_kms=delta_v,
            )
            G_eff = apply_doublet_constraints(G, eline_constraint_matrix)
            if eline_prior_type == "cloudy":
                from tengri.observation.eline_priors import (
                    marginalize_emission_lines_cloudy,
                )

                log_z = params.get("met_logzsol", fixed_values.get("met_logzsol", 0.0))
                neb_logU = params.get("neb_logU", fixed_values.get("neb_logU", -3.0))
                _, a_hat, _ = marginalize_emission_lines_cloudy(
                    data_spec - pred_spec,
                    noise_spec,
                    G_eff,
                    log_z=log_z,
                    neb_logU=neb_logU,
                    line_wavelengths=eline_independent_wavelengths,
                    prior_width_dex=eline_prior_width_dex,
                )
            else:
                prior_var = jnp.full(G_eff.shape[1], eline_prior_sigma**2)
                _, a_hat, _ = marginalize_emission_lines(
                    data_spec - pred_spec, noise_spec, G_eff, prior_variance=prior_var
                )
            pred_spec_with_lines = pred_spec + G_eff @ a_hat
            chi2_phot = jnp.sum(((data_phot - pred_phot) / noise_phot) ** 2)
            log_like_spec, _c_hat, _c_err = marginalize_calibration(
                pred_spec_with_lines,
                data_spec,
                noise_spec,
                model._wave_obs,
                n_poly=cal_n_poly,
                prior_sigma=cal_prior_sigma,
            )
            return -0.5 * chi2_phot + log_like_spec
        elif use_cal_marg and data_type == "spectroscopy":
            from tengri.observation.calibration import (
                marginalize_calibration,
            )

            log_like_spec, _c_hat, _c_err = marginalize_calibration(
                predicted,
                data,
                noise,
                model._wave_obs,
                n_poly=cal_n_poly,
                prior_sigma=cal_prior_sigma,
            )
            return log_like_spec
        elif use_cal_marg and data_type == "joint":
            from tengri.observation.calibration import (
                marginalize_calibration,
            )

            n_phot = model.predict_photometry(params, mode=mode).shape[0]
            data_phot = data[:n_phot]
            data_spec = data[n_phot:]
            noise_phot = noise[:n_phot]
            noise_spec = noise[n_phot:]
            pred_phot = predicted[:n_phot]
            pred_spec = predicted[n_phot:]

            chi2_phot = jnp.sum(((data_phot - pred_phot) / noise_phot) ** 2)
            log_like_spec, _c_hat, _c_err = marginalize_calibration(
                pred_spec,
                data_spec,
                noise_spec,
                model._wave_obs,
                n_poly=cal_n_poly,
                prior_sigma=cal_prior_sigma,
            )
            return -0.5 * chi2_phot + log_like_spec
        elif use_eline_marg and data_type == "spectroscopy":
            # Analytically marginalize emission line amplitudes
            from tengri.observation.eline_marginalization import (
                apply_doublet_constraints,
                build_eline_design_matrix,
                marginalize_emission_lines,
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
            G_eff = apply_doublet_constraints(G, eline_constraint_matrix)
            residual = data - predicted
            if eline_prior_type == "cloudy":
                from tengri.observation.eline_priors import (
                    marginalize_emission_lines_cloudy,
                )

                log_z = params.get("met_logzsol", fixed_values.get("met_logzsol", 0.0))
                neb_logU = params.get("neb_logU", fixed_values.get("neb_logU", -3.0))
                ln_l_eline, _, _ = marginalize_emission_lines_cloudy(
                    residual,
                    noise,
                    G_eff,
                    log_z=log_z,
                    neb_logU=neb_logU,
                    line_wavelengths=eline_independent_wavelengths,
                    prior_width_dex=eline_prior_width_dex,
                )
            else:
                prior_var = jnp.full(G_eff.shape[1], eline_prior_sigma**2)
                ln_l_eline, _, _ = marginalize_emission_lines(
                    residual, noise, G_eff, prior_variance=prior_var
                )
            return ln_l_eline
        elif use_eline_marg and data_type == "joint":
            # Joint: marginalize lines on spectroscopic part, standard chi2 for photometry
            from tengri.observation.eline_marginalization import (
                apply_doublet_constraints,
                build_eline_design_matrix,
                marginalize_emission_lines,
            )

            z = params.get("redshift", fixed_values.get("redshift", 0.0))
            sigma_kms = params.get("eline_sigma_kms", 0.0)
            delta_v = params.get("eline_delta_v_kms", 0.0)
            resolution = getattr(model, "_spectral_resolution", None) or 2000.0
            n_phot = model.predict_photometry(params, mode=mode).shape[0]
            data_phot = data[:n_phot]
            data_spec = data[n_phot:]
            noise_phot = noise[:n_phot]
            noise_spec = noise[n_phot:]
            pred_phot = predicted[:n_phot]
            pred_spec = predicted[n_phot:]
            G = build_eline_design_matrix(
                model._wave_obs,
                eline_wavelengths,
                resolution,
                z,
                eline_sigma_kms=sigma_kms,
                eline_delta_v_kms=delta_v,
            )
            G_eff = apply_doublet_constraints(G, eline_constraint_matrix)
            residual_spec = data_spec - pred_spec
            if eline_prior_type == "cloudy":
                from tengri.observation.eline_priors import (
                    marginalize_emission_lines_cloudy,
                )

                log_z = params.get("met_logzsol", fixed_values.get("met_logzsol", 0.0))
                neb_logU = params.get("neb_logU", fixed_values.get("neb_logU", -3.0))
                ln_l_eline, _, _ = marginalize_emission_lines_cloudy(
                    residual_spec,
                    noise_spec,
                    G_eff,
                    log_z=log_z,
                    neb_logU=neb_logU,
                    line_wavelengths=eline_independent_wavelengths,
                    prior_width_dex=eline_prior_width_dex,
                )
            else:
                prior_var = jnp.full(G_eff.shape[1], eline_prior_sigma**2)
                ln_l_eline, _, _ = marginalize_emission_lines(
                    residual_spec, noise_spec, G_eff, prior_variance=prior_var
                )
            chi2_phot = jnp.sum(((data_phot - pred_phot) / noise_phot) ** 2)
            return -0.5 * chi2_phot + ln_l_eline
        elif use_eline_fitted and data_type == "spectroscopy":
            # Fitted mode: amplitudes are explicit params; add line prediction to continuum
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
            G_eff = apply_doublet_constraints(G, eline_constraint_matrix)
            a = jnp.array([params[nm] for nm in eline_amplitude_names])
            predicted_with_lines = predicted + G_eff @ a
            chi2 = jnp.sum(((data - predicted_with_lines) / noise) ** 2)
            return -0.5 * chi2
        elif use_eline_fitted and data_type == "joint":
            # Joint fitted: add lines to spectroscopic part, standard chi2 for photometry
            from tengri.observation.eline_marginalization import (
                apply_doublet_constraints,
                build_eline_design_matrix,
            )

            z = params.get("redshift", fixed_values.get("redshift", 0.0))
            sigma_kms = params.get("eline_sigma_kms", 0.0)
            delta_v = params.get("eline_delta_v_kms", 0.0)
            resolution = getattr(model, "_spectral_resolution", None) or 2000.0
            n_phot = model.predict_photometry(params, mode=mode).shape[0]
            data_phot = data[:n_phot]
            data_spec = data[n_phot:]
            noise_phot = noise[:n_phot]
            noise_spec = noise[n_phot:]
            pred_phot = predicted[:n_phot]
            pred_spec = predicted[n_phot:]
            G = build_eline_design_matrix(
                model._wave_obs,
                eline_wavelengths,
                resolution,
                z,
                eline_sigma_kms=sigma_kms,
                eline_delta_v_kms=delta_v,
            )
            G_eff = apply_doublet_constraints(G, eline_constraint_matrix)
            a = jnp.array([params[nm] for nm in eline_amplitude_names])
            pred_spec_with_lines = pred_spec + G_eff @ a
            chi2_phot = jnp.sum(((data_phot - pred_phot) / noise_phot) ** 2)
            chi2_spec = jnp.sum(((data_spec - pred_spec_with_lines) / noise_spec) ** 2)
            return -0.5 * (chi2_phot + chi2_spec)
        elif use_variable_noise:
            f_cal = params.get("noise_frac_cal", 0.0)
            return -variable_noise_hamiltonian(data, noise, predicted, f_cal, dof=noise_dof)
        else:
            chi2 = jnp.sum(((data - predicted) / noise) ** 2)
            return -0.5 * chi2

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
    bounds = fitter._bounds
    free_names = fitter._free_names
    fixed_values = fitter._fixed_values
    spec = fitter.spec

    def loglik_unbounded(params_unbounded, data_args):
        params = {}
        for name in free_names:
            lo, hi = bounds[name]
            params[name] = to_bounded(params_unbounded[name], lo, hi)
        for name, val in fixed_values.items():
            params[name] = val
        if spec.stochastic and "psd_xi" in params_unbounded:
            params["psd_xi"] = params_unbounded["psd_xi"]
        return loglik_fn(params, data_args)

    return loglik_unbounded
