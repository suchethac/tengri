# SPDX-License-Identifier: BSD-3-Clause
"""Likelihood protocol adapters for SED fitting.

Builds a :class:`~tengri.inference.likelihoods.Likelihood` adapter from an
:class:`~tengri.inference.context.InferenceContext`.

The accepts-context signature (added in the Step-D-prime architectural
deepening, 2026-05-18) closes the leak Step D left behind: the prior
implementation accepted a raw ``Fitter`` and reached into 15+ private
attributes (``fitter._calibration_marginalize``, ``fitter._eline_*``,
``fitter._data_args`` …). With ADR-0009 in place, ``InferenceContext`` is
the right seam: data, noise, and likelihood-shape config all live on
context properties. A backend (or future hierarchical likelihood model)
that wants to build a likelihood without a full Fitter only needs an
``InferenceContext`` and the data arrays it already exposes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tengri.inference.context import InferenceContext

__all__ = ["build_base_likelihood", "build_likelihood_extras"]


def build_base_likelihood(context: InferenceContext):
    """Choose the BASE adapter for the main data channel(s).

    Routes through the Phase II-1 likelihood protocol cohort:

    - simple diagonal Gaussian → ``PhotometryLikelihood`` /
      ``SpectroscopyLikelihood``
    - joint phot+spec → ``CompositeLikelihood``
    - Student-t (variable noise) → ``StudentTLikelihood``
    - censored data → ``CensoredLikelihood``
    - spec covariance → ``MultivariateGaussianLikelihood``
    - calibration marginalisation → ``CalibrationMarginalisedLikelihood``
    - flat-prior e-line marginalisation → ``ELineMarginalisedLikelihood``
      (with a per-call design-matrix builder closure)

    Parameters
    ----------
    context : InferenceContext
        Seam exposing data, noise, parameter spec, and likelihood-shape
        configuration. See :class:`InferenceContext` for the contract.

    Returns
    -------
    Likelihood or None
        A Likelihood adapter object, or None if the configuration cannot be
        expressed via the protocol (currently: Cloudy-prior e-line
        marginalisation or e-line fitted amplitudes).
    """
    from tengri.inference.composite_likelihood import CompositeLikelihood
    from tengri.inference.likelihoods import (
        CalibrationELineMarginalisedLikelihood,
        CalibrationMarginalisedLikelihood,
        CensoredLikelihood,
        CloudyELineMarginalisedLikelihood,
        ELineFittedLikelihood,
        ELineMarginalisedLikelihood,
        MultivariateGaussianLikelihood,
        StudentTLikelihood,
    )
    from tengri.inference.photometry_likelihood import PhotometryLikelihood
    from tengri.inference.spectroscopy_likelihood import SpectroscopyLikelihood
    from tengri.observation.noise import (
        get_noise_dof,
        has_noise_model,
        uses_student_t,
    )

    # ── Censored mask (photometry): wraps full data with CensoredLikelihood
    if context.data_mask is not None and context.data_type == "photometry":
        dof = get_noise_dof(context.spec) if uses_student_t(context.spec) else None
        return CensoredLikelihood(
            obs=context.data,
            err=context.noise,
            mask=context.data_mask,
            dof=dof,
            f_cal_param="noise_frac_cal" if has_noise_model(context.spec) else None,
            channel="phot_fnu",
        )
    # Censored mask on spec / joint isn't covered by a single-channel
    # adapter (the mask spans the concatenated data array). Defer to
    # the legacy χ² fall-through, which applies `censored_neg_log_likelihood`
    # uniformly across the concatenated prediction. Without this
    # bail-out, downstream branches would build a plain
    # SpectroscopyLikelihood / Composite that silently ignores the
    # mask — a real bug, fix is the bail-out itself.
    if context.data_mask is not None:
        return None

    # ── Variable-noise / Student-t (no censoring) ──────────────
    # When the parameter spec declares noise_frac_cal / noise_dof,
    # apply the variable-noise (Student-t) adapter per channel.
    # `f_cal_param="noise_frac_cal"` reads the fractional calibration
    # uncertainty from the params dict at log_prob time. The same
    # f_cal applies to both phot and spec channels for joint data —
    # matches the legacy `variable_noise_hamiltonian` semantics.
    if has_noise_model(context.spec) or uses_student_t(context.spec):
        dof = get_noise_dof(context.spec) if uses_student_t(context.spec) else None
        if context.data_type == "photometry":
            return StudentTLikelihood(
                obs=context.data,
                err=context.noise,
                dof=dof,
                f_cal_param="noise_frac_cal",
                channel="phot_fnu",
            )
        if context.data_type == "spectroscopy":
            return StudentTLikelihood(
                obs=context.data,
                err=context.noise,
                dof=dof,
                f_cal_param="noise_frac_cal",
                channel="spec_fnu",
            )
        if context.data_type == "joint":
            n_phot = _n_phot_split(context)
            return CompositeLikelihood(
                StudentTLikelihood(
                    obs=context.data[:n_phot],
                    err=context.noise[:n_phot],
                    dof=dof,
                    f_cal_param="noise_frac_cal",
                    channel="phot_fnu",
                ),
                StudentTLikelihood(
                    obs=context.data[n_phot:],
                    err=context.noise[n_phot:],
                    dof=dof,
                    f_cal_param="noise_frac_cal",
                    channel="spec_fnu",
                ),
            )

    # ── Spec covariance (correlated noise on spectroscopy) ──────
    if "spec_cov_inv" in context.data_args:
        cov_inv = context.data_args["spec_cov_inv"]
        if context.data_type == "spectroscopy":
            return MultivariateGaussianLikelihood(
                obs=context.data, cov_inv=cov_inv, channel="spec_fnu"
            )
        if context.data_type == "joint":
            n_phot = _n_phot_split(context)
            return CompositeLikelihood(
                PhotometryLikelihood(
                    fnu_obs=context.data[:n_phot], fnu_err=context.noise[:n_phot]
                ),
                MultivariateGaussianLikelihood(
                    obs=context.data[n_phot:], cov_inv=cov_inv, channel="spec_fnu"
                ),
            )

    # ── Combined calibration polynomial + eline marginalisation ──
    # Most common galaxy spectroscopy configuration (Prospector-style).
    # Sequential composition: marginalise lines → add MAP amplitudes
    # to the prediction → run cal-marg on the line-augmented model.
    # Supports both flat and Cloudy eline priors via the same adapter.
    # eline_fitted + cal_marg is not yet expressible (would need a
    # mixed marginalised/fitted variant); legacy switch covers it.
    if context.calibration_marginalize and context.eline_fitted:
        raise NotImplementedError(
            "Combined calibration marginalisation + fitted (non-marginalised) "
            "emission-line amplitudes is not currently supported. "
            "Use eline_marginalize=True (with optional eline_prior_type) for "
            "the standard Prospector-style configuration, or disable "
            "calibration_marginalize."
        )
    if context.calibration_marginalize and context.eline_marginalize:
        wavelength = getattr(context.model, "_wave_obs", None)
        if wavelength is None:
            raise ValueError("Calibration marginalisation requires model._wave_obs.")
        builder = _make_eline_design_builder(context)
        if builder is None:
            raise ValueError(
                "Emission-line marginalisation requires _eline_wavelengths and "
                "_eline_constraint_matrix to be set on the fitter."
            )
        if context.data_type == "spectroscopy":
            spec_obs, spec_err = context.data, context.noise
        else:  # joint
            n_phot = _n_phot_split(context)
            spec_obs = context.data[n_phot:]
            spec_err = context.noise[n_phot:]
        cal_eline_lk = CalibrationELineMarginalisedLikelihood(
            fnu_obs=spec_obs,
            fnu_err=spec_err,
            wavelength=wavelength,
            design_matrix_builder=builder,
            n_poly=context.cal_n_poly,
            prior_sigma=context.cal_prior_sigma,
            eline_prior_type=context.eline_prior_type or "flat",
            eline_prior_sigma=context.eline_prior_sigma or 1e10,
            eline_line_wavelengths=context.eline_independent_wavelengths,
            eline_prior_width_dex=context.eline_prior_width_dex,
            channel="spec_fnu",
        )
        if context.data_type == "spectroscopy":
            return cal_eline_lk
        return CompositeLikelihood(
            PhotometryLikelihood(fnu_obs=context.data[:n_phot], fnu_err=context.noise[:n_phot]),
            cal_eline_lk,
        )

    # ── Calibration polynomial only (no elines) ─────────────────
    if context.calibration_marginalize and context.has_spectroscopy:
        wavelength = getattr(context.model, "_wave_obs", None)
        if wavelength is None:
            raise ValueError("Calibration marginalisation requires model._wave_obs.")
        cal_lk = CalibrationMarginalisedLikelihood(
            fnu_obs=context.data
            if context.data_type == "spectroscopy"
            else context.data[_n_phot_split(context) :],
            fnu_err=context.noise
            if context.data_type == "spectroscopy"
            else context.noise[_n_phot_split(context) :],
            wavelength=wavelength,
            n_poly=context.cal_n_poly,
            prior_sigma=context.cal_prior_sigma,
            channel="spec_fnu",
        )
        if context.data_type == "spectroscopy":
            return cal_lk
        n_phot = _n_phot_split(context)
        return CompositeLikelihood(
            PhotometryLikelihood(fnu_obs=context.data[:n_phot], fnu_err=context.noise[:n_phot]),
            cal_lk,
        )

    # ── E-line: marginalised (flat / cloudy prior) OR fitted ────
    if context.eline_marginalize or context.eline_fitted:
        if context.data_type == "spectroscopy":
            spec_obs = context.data
            spec_err = context.noise
        elif context.data_type == "joint":
            n_phot = _n_phot_split(context)
            spec_obs = context.data[n_phot:]
            spec_err = context.noise[n_phot:]
        else:
            raise NotImplementedError(
                "Emission-line marginalisation / fitting requires spectroscopy "
                "or joint data; got data_type='photometry'. Either disable "
                "the eline flag or use spectroscopy."
            )
        builder = _make_eline_design_builder(context)
        if builder is None:
            raise ValueError(
                "Emission-line path requires _eline_wavelengths and "
                "_eline_constraint_matrix to be set on the fitter."
            )
        # Pick the right adapter for the eline mode.
        if context.eline_fitted:
            eline_lk = ELineFittedLikelihood(
                fnu_obs=spec_obs,
                fnu_err=spec_err,
                design_matrix_builder=builder,
                amplitude_names=tuple(context.eline_amplitude_names),
                channel="spec_fnu",
            )
        elif context.eline_prior_type == "cloudy":
            eline_lk = CloudyELineMarginalisedLikelihood(
                fnu_obs=spec_obs,
                fnu_err=spec_err,
                design_matrix_builder=builder,
                line_wavelengths=context.eline_independent_wavelengths,
                prior_width_dex=context.eline_prior_width_dex,
                channel="spec_fnu",
            )
        else:
            eline_lk = ELineMarginalisedLikelihood(
                fnu_obs=spec_obs,
                fnu_err=spec_err,
                design_matrix_builder=builder,
                channel="spec_fnu",
            )
        if context.data_type == "spectroscopy":
            return eline_lk
        return CompositeLikelihood(
            PhotometryLikelihood(
                fnu_obs=context.data[: _n_phot_split(context)],
                fnu_err=context.noise[: _n_phot_split(context)],
            ),
            eline_lk,
        )

    # ── Plain diagonal Gaussian cases ───────────────────────────
    if context.data_type == "photometry":
        return PhotometryLikelihood(fnu_obs=context.data, fnu_err=context.noise)
    if context.data_type == "spectroscopy":
        return SpectroscopyLikelihood(fnu_obs=context.data, fnu_err=context.noise)
    if context.data_type == "joint":
        n_phot = _n_phot_split(context)
        return CompositeLikelihood(
            PhotometryLikelihood(fnu_obs=context.data[:n_phot], fnu_err=context.noise[:n_phot]),
            SpectroscopyLikelihood(fnu_obs=context.data[n_phot:], fnu_err=context.noise[n_phot:]),
        )
    return None


def build_likelihood_extras(context: InferenceContext):
    """Constraint-style likelihoods composed on top of the base.

    Reads the optional ``line_flux_*`` and ``index_*`` data_args
    and emits :class:`GaussianLikelihood` instances pinned to the
    ``"line_fluxes"`` / ``"indices"`` prediction-dict keys (which
    the user-likelihood short-circuit in
    :func:`tengri.inference.loss_functions.build_loss_fn`
    populates by calling ``model.predict_line_fluxes`` /
    ``model.predict_spectral_indices``).

    Parameters
    ----------
    context : InferenceContext
        Seam exposing ``data_args`` (the loss-function closure dict).

    Returns
    -------
    list of Likelihood
        Likelihood adapters for constraint terms (line fluxes, spectral indices).
    """
    from tengri.inference.likelihoods import GaussianLikelihood

    data_args = context.data_args
    extras = []
    if "line_flux_waves" in data_args:
        extras.append(
            GaussianLikelihood(
                obs=data_args["line_flux_obs"],
                err=data_args["line_flux_err"],
                channel="line_fluxes",
                name="line_flux_constraint",
            )
        )
    if "index_obs" in data_args:
        extras.append(
            GaussianLikelihood(
                obs=data_args["index_obs"],
                err=data_args["index_err"],
                channel="indices",
                name="spectral_index_constraint",
            )
        )
    return extras


# ── Helpers (private) ────────────────────────────────────────────────


def _n_phot_split(context: InferenceContext) -> int:
    """Number of photometric data points in joint (phot+spec) data.

    Raises ``ValueError`` if ``model.observation.n_data_phot`` is missing —
    joint data cannot be split without it.
    """
    obs = getattr(context.model, "observation", None)
    n_phot = getattr(obs, "n_data_phot", None)
    if n_phot is None:
        raise ValueError(
            "Joint (phot+spec) data requires model.observation.n_data_phot to be set."
        )
    return n_phot


def _make_eline_design_builder(context: InferenceContext):
    """Build a closure that rebuilds the e-line design matrix per call.

    The returned callable takes the params dict and returns a
    ``(n_pixels, n_lines)`` design matrix with current redshift +
    line-shape parameters baked in. Required because line
    wavelengths shift with z and line widths can be free
    parameters too.

    Returns ``None`` when the underlying e-line state is absent.
    """
    from tengri.inference.loss_functions import _build_eline_G_eff

    if context.eline_wavelengths is None or context.eline_constraint_matrix is None:
        return None

    fixed_values = context.fixed_values
    model = context.model
    wavelengths = context.eline_wavelengths
    constraint_matrix = context.eline_constraint_matrix

    def builder(params):
        return _build_eline_G_eff(params, fixed_values, model, wavelengths, constraint_matrix)

    return builder
