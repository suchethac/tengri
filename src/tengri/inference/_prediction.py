# SPDX-License-Identifier: BSD-3-Clause
"""Forward-model prediction dictionary and measured-line windows for the data term.

:func:`_build_prediction` evaluates the forward model once per likelihood call and returns
the channel dictionary the likelihood adapters read. :func:`_resolve_measured_line_defs`
gives the continuum windows used when a model has no discrete line catalog. Both are
shared with :mod:`tengri.inference.data_term` and :mod:`tengri.inference.mass_profile`.
"""

from __future__ import annotations

import jax.numpy as jnp

__all__ = ["_build_prediction", "_resolve_measured_line_defs"]


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
    needs_line_lums=False,
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
    # ``params`` here is the fit machinery's own resolved physical dict (free
    # + Fixed, mirrors merged) -- Fitter's contract for neg_log_lik / the
    # extra_log_prior hook, and NOT a user-supplied params dict (#2296 refuses
    # THOSE; this one is always internally consistent with the spec, so
    # naming a Fixed key here can never disagree with the pinned value).
    # Every public predict_*/measure_* method below refuses a Fixed key of
    # its own regardless, so it gets the free-only subset; the merged
    # ``params`` stays available to the caller (neg_log_lik reads
    # ``params.get("noise_frac_cal", ...)`` and hands ``params`` to the
    # Likelihood adapter directly).
    _spec = getattr(model, "spec", None)
    if _spec is not None and hasattr(_spec, "free_params"):
        _free_names = set(_spec.free_params)
        free_params = {k: v for k, v in params.items() if k in _free_names}
        # Evaluation's fixed values (e.g., runtime redshift from data_args)
        eval_fixed = {k: v for k, v in params.items() if k not in _free_names}
    else:
        free_params = params
        eval_fixed = None

    # Single threaded forward for phot/spec/joint: one orchestrator call
    # returns an Observables carrying every configured channel. ``_obs`` is
    # None when threading is unavailable (dummy models, ``use_components``) or
    # when ``jit_inputs`` was not built, then we fall back to the eager
    # ``model.predict_*`` accessors (which closure-capture the SSP grid, so
    # this path bakes it; acceptable for the non-inference / dummy-model uses).
    _obs = None
    _threaded_lums = None
    if threaded_impl is not None and jit_inputs is not None and not use_components:
        _obs = threaded_impl(
            params,
            jit_inputs["fixed_values"],
            jit_inputs["ssp_data"],
            jit_inputs["template_data"],
            jit_inputs.get("ztable_data"),
        )
        if needs_line_lums:
            _obs, _threaded_lums = _obs

    if data_type == "photometry":
        if _obs is not None:
            predicted = _obs.phot_fnu
        elif use_components:
            predicted = model._photometry_via_state(free_params)
        else:
            predicted = model.predict_photometry(free_params)
        pred_phot, pred_spec = predicted, None
    elif data_type == "spectroscopy":
        if _obs is not None:
            predicted = _obs.spec_fnu
        elif use_components:
            predicted = model._spectrum_via_state(free_params)
        else:
            predicted = model.predict_spectrum(free_params)
        pred_phot, pred_spec = None, predicted
    elif data_type == "joint":
        if _obs is not None:
            pred_phot, pred_spec = _obs.phot_fnu, _obs.spec_fnu
        elif use_components:
            pred_phot = model._photometry_via_state(free_params)
            pred_spec = model._spectrum_via_state(free_params)
        else:
            pred_phot = model.predict_photometry(free_params)
            pred_spec = model.predict_spectrum(free_params)
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
    needs_state = (
        has_line_ratios
        or has_indices
        or (has_line_fluxes and not fast_lines)
        or (needs_line_lums and _threaded_lums is None)
    )
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
        feature_state = model.predict_state(free_params)
    if needs_line_lums and _threaded_lums is not None:
        prediction["log_line_lums_attenuated"] = _threaded_lums
    elif needs_line_lums:
        derived = feature_state.derived
        prediction["log_line_lums_attenuated"] = derived.get(
            "log_line_lums_attenuated", derived["log_line_lums"]
        )
    if has_line_fluxes:
        if measured_line_defs is not None:
            # No discrete line catalog (BakedIn) → measure the fluxes off the
            # model spectrum the way a pipeline does (measure_line_fluxes). With
            # FeaturePrecomp the measurement runs against the SSP window LUT.
            prediction["line_fluxes"] = model.measure_line_fluxes(
                free_params,
                measured_line_defs,
                approx=fast_lines,
                state=feature_state,
                fixed_values=eval_fixed,
            )
        else:
            prediction["line_fluxes"] = model.predict_line_fluxes(
                free_params,
                target_wavelengths=data_args["line_flux_waves"],
                state=feature_state,
                fixed_values=eval_fixed,
            )
        # ``line_flux_scaling`` (#2527): a declared calibration nuisance of the
        # DATA channel (default ``Fixed(1.0)``, bit-identical unless a user
        # frees it), absorbing an aperture/absolute-flux-calibration mismatch
        # between the line-flux measurement and the rest of the SED. Applied
        # HERE, to the array the likelihood compares against
        # ``data_args["line_flux_obs"]`` -- never inside ``predict_line_fluxes``
        # / ``measure_line_fluxes`` themselves, so line ratios, BPT, and every
        # other derived property built from those methods are untouched.
        prediction["line_fluxes"] = prediction["line_fluxes"] * params.get(
            "line_flux_scaling", 1.0
        )
    if has_line_ratios:
        prediction["line_ratios"] = model.predict_line_ratios(
            free_params,
            model.observation.line_ratios,
            state=feature_state,
            fixed_values=eval_fixed,
        )
    if has_indices:
        # Spectral indices accept evaluation-time fixed_values (e.g. runtime z).
        prediction["indices"] = model.predict_spectral_indices(
            free_params, index_defs, state=feature_state, fixed_values=eval_fixed
        )

    return prediction, predicted, pred_phot, pred_spec


def _resolve_measured_line_defs(model, *, has_line_fluxes: bool):
    """Continuum-window line definitions for backends with no discrete catalog.

    BakedIn nebular backends cannot run ``predict_line_fluxes``, so the fluxes
    are measured off the model spectrum instead; these are the windows that
    measurement uses. Built from the observation's concrete line centers (never
    from traced ``data_args``) so a jitted loss sees a static ``LineDef`` set.

    Returns ``None`` when the model has a discrete line catalog, in which case
    the caller uses ``predict_line_fluxes`` directly.

    Shared with :mod:`tengri.inference.mass_profile`, which must resolve the
    same windows the loss does: the profiled quadratic scores its line block
    against the loss's own prediction, so two derivations that could disagree
    would silently score different quantities.
    """
    measured_line_defs = None
    if has_line_fluxes and not model._has_line_catalog():
        obs_for_lines = getattr(model, "observation", None)
        lf = getattr(obs_for_lines, "line_fluxes", None) if obs_for_lines is not None else None
        if lf is not None:
            import numpy as _np

            from tengri.observation.line_measurement import default_line_defs

            measured_line_defs = default_line_defs(_np.asarray(lf.wavelengths), tuple(lf.names))
    return measured_line_defs
