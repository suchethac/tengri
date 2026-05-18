"""Likelihood module — extracted from Fitter for reusability and separation of concerns.

This module defines the :class:`Likelihood` class that encapsulates compiled
likelihood callables for photometry, spectroscopy, emission lines, and spectral
indices. It separates likelihood construction (Fitter responsibility) from
likelihood evaluation (inference engine responsibility).

The Likelihood module enables:
- Custom likelihood subclasses (e.g., RobustLikelihood with Student-t photometry)
- Inference backends to consume a pre-built likelihood without coupling to Fitter internals
- Bit-identical log-probability values across different inference methods

Usage
-----
    from tengri import SEDModel, Fitter
    from tengri.inference.likelihood import Likelihood

    model = SEDModel(...)
    observation = ...
    fitter = Fitter(model, data, noise)

    # Likelihood is auto-built by Fitter, but can also be built independently:
    likelihood = Likelihood.build(
        model=model,
        observation=observation,
        calibration_spec={...}
    )

    # Call the combined likelihood:
    log_p_total = likelihood.log_p_total(params, data)
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tengri.forward.sed_model import SEDModel
    from tengri.observation import Observation

__all__ = ["Likelihood", "RobustLikelihood"]


@dataclasses.dataclass(frozen=True)
class Likelihood:
    """Compiled likelihood callables for a given model, observation, and noise configuration.

    This dataclass holds pre-compiled JAX functions for the data term of the
    Hamiltonian. Each callable takes (params, data_args) and returns a scalar
    log-probability (or its negative).

    Parameters
    ----------
    log_p_phot : callable or None
        Photometric likelihood log_prob(prediction, params). None if no photometry.
    log_p_spec : callable or None
        Spectroscopy likelihood log_prob(prediction, params). None if no spectroscopy.
    log_p_lines : callable or None
        Emission-line likelihood. None if no e-line data/constraints.
    log_p_indices : callable or None
        Spectral indices likelihood. None if no index data/constraints.
    log_p_total : callable
        Combined likelihood log_prob(params, data_args) → scalar.
        Routes through the composed structure of the above to compute
        the total data-term log-probability.

    Attributes
    ----------
    _fitter_state : dict or None
        Cached state from the Fitter that built this likelihood (for debugging).
        Not part of the public API.

    Notes
    -----
    This is a frozen dataclass. Once constructed, it is immutable. Customize
    behaviour by subclassing (e.g., RobustLikelihood) and overriding the
    component callables or log_p_total method.

    Examples
    --------
    Build a likelihood from a Fitter's internals:

        likelihood = Likelihood.build(fitter.model, fitter.observation, {...})
        log_p = likelihood.log_p_total(params, {'data': data, 'noise': noise})

    Subclass for custom behaviour:

        class RobustLikelihood(Likelihood):
            '''Student-t photometric likelihood for outlier-robust fitting.'''

            @classmethod
            def build(cls, model, observation, calibration_spec=None):
                # Build Student-t photometry instead of Gaussian
                # ... (implementation below)
    """

    log_p_phot: Callable | None = None
    log_p_spec: Callable | None = None
    log_p_lines: Callable | None = None
    log_p_indices: Callable | None = None
    log_p_total: Callable | None = None

    _fitter_state: dict | None = dataclasses.field(default=None, init=False, repr=False)

    @classmethod
    def build(
        cls,
        model: SEDModel,
        observation: Observation | None,
        calibration_spec: dict | None = None,
        fitter_state: dict | None = None,
    ) -> Likelihood:
        """Build a Likelihood from model, observation, and calibration config.

        This is the primary constructor. It mirrors the logic of
        :meth:`Fitter._build_base_likelihood` and :meth:`Fitter._build_likelihood_extras`,
        extracting the likelihood-construction code out of Fitter's __init__.

        Parameters
        ----------
        model : SEDModel
            The forward model, including the precomputed components and state.
        observation : Observation or None
            The observation configuration (filters, spectroscopy, noise models).
        calibration_spec : dict, optional
            Calibration configuration with keys:
            - 'marginalize': bool — whether to marginalise over calibration polynomial
            - 'n_poly': int — order of calibration polynomial (default 3)
            - 'prior_sigma': float — prior width on polynomial coefficients
            - 'eline_marginalize': bool
            - 'eline_prior_type': str ('flat', 'cloudy', or None)
            - 'eline_prior_sigma': float
            - 'eline_prior_width_dex': float
            - 'eline_independent_wavelengths': array or None
            - 'data_args': dict with any special data (e.g., line_flux_obs, index_obs)
            - 'data': observed flux array
            - 'noise': noise array
            - 'data_mask': optional mask for censored data
            - 'data_type': 'photometry', 'spectroscopy', or 'joint'
            - 'spec': Parameters object (for noise model config)
        fitter_state : dict, optional
            Internal state from the Fitter that called this (for debugging).
            Not used in the likelihood computation; stored for introspection.

        Returns
        -------
        Likelihood
            A frozen Likelihood instance with log_p_total and component callables.

        Raises
        ------
        ValueError
            If calibration_spec is incomplete or inconsistent.
        NotImplementedError
            If the data configuration is not yet supported.

        Notes
        -----
        For now, this method is called exclusively by :meth:`Fitter.__init__` via
        :meth:`_maybe_build_default_likelihood`. In the future (after the Fitter
        → backend-adapter refactor, Step #1), it will also be called directly by
        inference backend adapters.

        All likelihood construction logic that was in Fitter goes here. This
        includes handling of:
        - Censored photometry (CensoredLikelihood)
        - Variable-noise / Student-t (StudentTLikelihood)
        - Covariance-based spectroscopy (MultivariateGaussianLikelihood)
        - Calibration marginalisation (CalibrationMarginalisedLikelihood)
        - Emission-line marginalisation (ELineMarginalisedLikelihood)
        - Fitted line amplitudes (ELineFittedLikelihood)
        - Composite channel composition (CompositeLikelihood)
        - Line-flux / spectral-index constraints (GaussianLikelihood)
        """
        # Build using the internal _do_build method; subclasses can override
        # just the component-construction logic there.
        return cls._do_build(model, observation, calibration_spec, fitter_state)

    @classmethod
    def _do_build(
        cls,
        model: SEDModel,
        observation: Observation | None,
        calibration_spec: dict | None = None,
        fitter_state: dict | None = None,
    ) -> Likelihood:
        """Internal builder — override in subclasses to customize component callables.

        This method does the actual work and can be overridden by subclasses
        (e.g., RobustLikelihood) to substitute custom component callables
        (e.g., Student-t photometric likelihood instead of Gaussian).

        The default implementation routes through the current Fitter code.
        """
        from tengri.inference.composite_likelihood import CompositeLikelihood

        # Extract calibration spec (with defaults)
        if calibration_spec is None:
            calibration_spec = {}

        # Unpack the spec dict for backward compatibility
        data = calibration_spec.get("data")
        noise = calibration_spec.get("noise")
        data_mask = calibration_spec.get("data_mask")
        data_type = calibration_spec.get("data_type", "photometry")
        spec = calibration_spec.get("spec")
        data_args = calibration_spec.get("data_args", {})
        calibration_marginalize = calibration_spec.get("marginalize", False)
        cal_n_poly = calibration_spec.get("n_poly", 3)
        cal_prior_sigma = calibration_spec.get("prior_sigma", 1.0)
        eline_marginalize = calibration_spec.get("eline_marginalize", False)
        eline_fitted = calibration_spec.get("eline_fitted", False)
        eline_prior_type = calibration_spec.get("eline_prior_type")
        eline_prior_sigma = calibration_spec.get("eline_prior_sigma", 1e10)
        eline_independent_wavelengths = calibration_spec.get("eline_independent_wavelengths")
        eline_prior_width_dex = calibration_spec.get("eline_prior_width_dex")
        eline_wavelengths = calibration_spec.get("_eline_wavelengths")
        eline_constraint_matrix = calibration_spec.get("_eline_constraint_matrix")
        eline_amplitude_names = calibration_spec.get("_eline_amplitude_names", [])
        fixed_values = calibration_spec.get("_fixed_values", {})
        has_spectroscopy = data_type in ("spectroscopy", "joint")

        # Build the base likelihood (handles photometry, spectroscopy, joint data)
        base_likelihood = cls._build_base_likelihood(
            model=model,
            data=data,
            noise=noise,
            data_mask=data_mask,
            data_type=data_type,
            spec=spec,
            data_args=data_args,
            calibration_marginalize=calibration_marginalize,
            cal_n_poly=cal_n_poly,
            cal_prior_sigma=cal_prior_sigma,
            eline_marginalize=eline_marginalize,
            eline_fitted=eline_fitted,
            eline_prior_type=eline_prior_type,
            eline_prior_sigma=eline_prior_sigma,
            eline_independent_wavelengths=eline_independent_wavelengths,
            eline_prior_width_dex=eline_prior_width_dex,
            eline_wavelengths=eline_wavelengths,
            eline_constraint_matrix=eline_constraint_matrix,
            eline_amplitude_names=eline_amplitude_names,
            fixed_values=fixed_values,
            has_spectroscopy=has_spectroscopy,
        )

        # Build extras (line fluxes, spectral indices)
        extras = cls._build_likelihood_extras(data_args)

        # Compose if needed
        if base_likelihood is None:
            combined_likelihood = None
        elif not extras:
            combined_likelihood = base_likelihood
        else:
            combined_likelihood = CompositeLikelihood(base_likelihood, *extras)

        # Create the log_p_total callable
        def log_p_total(params: dict, data_args_dict: dict) -> float:
            """Compute total log-probability of data given params.

            Parameters
            ----------
            params : dict
                Physical parameters (not standardized).
            data_args_dict : dict
                Observed data, noise, and optional noise models.

            Returns
            -------
            float
                Log-probability (or its negative, depending on the likelihood adapter).
            """
            if combined_likelihood is None:
                return 0.0
            # Build the prediction dict from data
            prediction = {"phot_fnu": data_args_dict.get("data")}
            return combined_likelihood.log_prob(prediction, params)

        # Return the Likelihood instance
        likelihood = cls(
            log_p_phot=None,  # Component callables not exposed yet
            log_p_spec=None,
            log_p_lines=None,
            log_p_indices=None,
            log_p_total=log_p_total,
        )

        # Store fitter state for introspection if provided
        if fitter_state is not None:
            object.__setattr__(likelihood, "_fitter_state", fitter_state)

        return likelihood

    @classmethod
    def _build_base_likelihood(
        cls,
        model: SEDModel,
        data: Any,
        noise: Any,
        data_mask: Any | None = None,
        data_type: str = "photometry",
        spec: Any = None,
        data_args: dict | None = None,
        calibration_marginalize: bool = False,
        cal_n_poly: int = 3,
        cal_prior_sigma: float = 1.0,
        eline_marginalize: bool = False,
        eline_fitted: bool = False,
        eline_prior_type: str | None = None,
        eline_prior_sigma: float = 1e10,
        eline_independent_wavelengths: Any | None = None,
        eline_prior_width_dex: Any | None = None,
        eline_wavelengths: Any | None = None,
        eline_constraint_matrix: Any | None = None,
        eline_amplitude_names: list | None = None,
        fixed_values: dict | None = None,
        has_spectroscopy: bool = False,
    ) -> Any:
        """Build the base likelihood (the primary data-term component).

        This mirrors :meth:`Fitter._build_base_likelihood` exactly.
        Override in subclasses to substitute custom likelihood adapters.
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
        from tengri.inference.loss_functions import _build_eline_G_eff
        from tengri.inference.photometry_likelihood import PhotometryLikelihood
        from tengri.inference.spectroscopy_likelihood import SpectroscopyLikelihood
        from tengri.observation.noise import (
            get_noise_dof,
            has_noise_model,
            uses_student_t,
        )

        if eline_amplitude_names is None:
            eline_amplitude_names = []
        if fixed_values is None:
            fixed_values = {}
        if data_args is None:
            data_args = {}

        def _n_phot_split() -> int:
            """Helper to get the photometry split in joint data."""
            obs = getattr(model, "observation", None)
            n_phot = getattr(obs, "n_data_phot", None)
            if n_phot is None:
                raise ValueError(
                    "Joint (phot+spec) data requires model.observation.n_data_phot to be set."
                )
            return n_phot

        def _make_eline_design_builder():
            """Build a closure that rebuilds the e-line design matrix per call."""
            if eline_wavelengths is None or eline_constraint_matrix is None:
                return None

            wavelengths = eline_wavelengths
            constraint_matrix = eline_constraint_matrix

            def builder(params):
                return _build_eline_G_eff(
                    params, fixed_values, model, wavelengths, constraint_matrix
                )

            return builder

        # ── Censored mask (photometry) ──────────────────────────────
        if data_mask is not None and data_type == "photometry":
            dof = (
                get_noise_dof(spec) if spec is not None and uses_student_t(spec)
                else None
            )
            return CensoredLikelihood(
                obs=data,
                err=noise,
                mask=data_mask,
                dof=dof,
                f_cal_param=(
                    "noise_frac_cal" if spec is not None and has_noise_model(spec)
                    else None
                ),
                channel="phot_fnu",
            )
        # Censored on spec/joint defers to legacy χ² (not covered by adapter cohort)
        if data_mask is not None:
            return None

        # ── Variable-noise / Student-t ─────────────────────────────
        if spec is not None and (
            has_noise_model(spec) or uses_student_t(spec)
        ):
            dof = get_noise_dof(spec) if uses_student_t(spec) else None
            if data_type == "photometry":
                return StudentTLikelihood(
                    obs=data,
                    err=noise,
                    dof=dof,
                    f_cal_param="noise_frac_cal",
                    channel="phot_fnu",
                )
            if data_type == "spectroscopy":
                return StudentTLikelihood(
                    obs=data,
                    err=noise,
                    dof=dof,
                    f_cal_param="noise_frac_cal",
                    channel="spec_fnu",
                )
            if data_type == "joint":
                n_phot = _n_phot_split()
                return CompositeLikelihood(
                    StudentTLikelihood(
                        obs=data[:n_phot],
                        err=noise[:n_phot],
                        dof=dof,
                        f_cal_param="noise_frac_cal",
                        channel="phot_fnu",
                    ),
                    StudentTLikelihood(
                        obs=data[n_phot:],
                        err=noise[n_phot:],
                        dof=dof,
                        f_cal_param="noise_frac_cal",
                        channel="spec_fnu",
                    ),
                )

        # ── Spec covariance ────────────────────────────────────────
        if "spec_cov_inv" in data_args:
            cov_inv = data_args.get("spec_cov_inv")
            if data_type == "spectroscopy":
                return MultivariateGaussianLikelihood(
                    obs=data, cov_inv=cov_inv, channel="spec_fnu"
                )
            if data_type == "joint":
                n_phot = _n_phot_split()
                return CompositeLikelihood(
                    PhotometryLikelihood(fnu_obs=data[:n_phot], fnu_err=noise[:n_phot]),
                    MultivariateGaussianLikelihood(
                        obs=data[n_phot:], cov_inv=cov_inv, channel="spec_fnu"
                    ),
                )

        # ── Combined calibration + e-line marginalisation ─────────
        if calibration_marginalize and eline_fitted:
            raise NotImplementedError(
                "Combined calibration marginalisation + fitted (non-marginalised) "
                "emission-line amplitudes is not currently supported."
            )
        if calibration_marginalize and eline_marginalize:
            wavelength = getattr(model, "_wave_obs", None)
            if wavelength is None:
                raise ValueError("Calibration marginalisation requires model._wave_obs.")
            builder = _make_eline_design_builder()
            if builder is None:
                raise ValueError("Emission-line marginalisation requires design builder.")
            if data_type == "spectroscopy":
                spec_obs, spec_err = data, noise
            else:  # joint
                n_phot = _n_phot_split()
                spec_obs = data[n_phot:]
                spec_err = noise[n_phot:]
            cal_eline_lk = CalibrationELineMarginalisedLikelihood(
                fnu_obs=spec_obs,
                fnu_err=spec_err,
                wavelength=wavelength,
                design_matrix_builder=builder,
                n_poly=cal_n_poly,
                prior_sigma=cal_prior_sigma,
                eline_prior_type=eline_prior_type or "flat",
                eline_prior_sigma=eline_prior_sigma or 1e10,
                eline_line_wavelengths=eline_independent_wavelengths,
                eline_prior_width_dex=eline_prior_width_dex,
                channel="spec_fnu",
            )
            if data_type == "spectroscopy":
                return cal_eline_lk
            return CompositeLikelihood(
                PhotometryLikelihood(fnu_obs=data[:n_phot], fnu_err=noise[:n_phot]),
                cal_eline_lk,
            )

        # ── Calibration polynomial only ────────────────────────────
        if calibration_marginalize and has_spectroscopy:
            wavelength = getattr(model, "_wave_obs", None)
            if wavelength is None:
                raise ValueError("Calibration marginalisation requires model._wave_obs.")
            cal_lk = CalibrationMarginalisedLikelihood(
                fnu_obs=data if data_type == "spectroscopy" else data[_n_phot_split() :],
                fnu_err=noise if data_type == "spectroscopy" else noise[_n_phot_split() :],
                wavelength=wavelength,
                n_poly=cal_n_poly,
                prior_sigma=cal_prior_sigma,
                channel="spec_fnu",
            )
            if data_type == "spectroscopy":
                return cal_lk
            n_phot = _n_phot_split()
            return CompositeLikelihood(
                PhotometryLikelihood(fnu_obs=data[:n_phot], fnu_err=noise[:n_phot]),
                cal_lk,
            )

        # ── E-line: marginalised OR fitted ─────────────────────────
        if eline_marginalize or eline_fitted:
            if data_type == "spectroscopy":
                spec_obs = data
                spec_err = noise
            elif data_type == "joint":
                n_phot = _n_phot_split()
                spec_obs = data[n_phot:]
                spec_err = noise[n_phot:]
            else:
                raise NotImplementedError(
                    "Emission-line marginalisation / fitting requires spectroscopy "
                    "or joint data; got data_type='photometry'."
                )
            builder = _make_eline_design_builder()
            if builder is None:
                raise ValueError(
                    "Emission-line path requires _eline_wavelengths and "
                    "_eline_constraint_matrix to be set."
                )
            # Pick the right adapter for the eline mode
            if eline_fitted:
                eline_lk = ELineFittedLikelihood(
                    fnu_obs=spec_obs,
                    fnu_err=spec_err,
                    design_matrix_builder=builder,
                    amplitude_names=tuple(eline_amplitude_names),
                    channel="spec_fnu",
                )
            elif eline_prior_type == "cloudy":
                eline_lk = CloudyELineMarginalisedLikelihood(
                    fnu_obs=spec_obs,
                    fnu_err=spec_err,
                    design_matrix_builder=builder,
                    line_wavelengths=eline_independent_wavelengths,
                    prior_width_dex=eline_prior_width_dex,
                    channel="spec_fnu",
                )
            else:
                eline_lk = ELineMarginalisedLikelihood(
                    fnu_obs=spec_obs,
                    fnu_err=spec_err,
                    design_matrix_builder=builder,
                    channel="spec_fnu",
                )
            if data_type == "spectroscopy":
                return eline_lk
            return CompositeLikelihood(
                PhotometryLikelihood(
                    fnu_obs=data[: _n_phot_split()],
                    fnu_err=noise[: _n_phot_split()],
                ),
                eline_lk,
            )

        # ── Plain diagonal Gaussian cases ──────────────────────────
        if data_type == "photometry":
            return PhotometryLikelihood(fnu_obs=data, fnu_err=noise)
        if data_type == "spectroscopy":
            return SpectroscopyLikelihood(fnu_obs=data, fnu_err=noise)
        if data_type == "joint":
            n_phot = _n_phot_split()
            return CompositeLikelihood(
                PhotometryLikelihood(fnu_obs=data[:n_phot], fnu_err=noise[:n_phot]),
                SpectroscopyLikelihood(fnu_obs=data[n_phot:], fnu_err=noise[n_phot:]),
            )
        return None

    @classmethod
    def _build_likelihood_extras(cls, data_args: dict) -> list:
        """Build constraint-style likelihoods (line fluxes, spectral indices).

        This mirrors :meth:`Fitter._build_likelihood_extras` exactly.
        """
        from tengri.inference.likelihoods import GaussianLikelihood

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


@dataclasses.dataclass(frozen=True)
class RobustLikelihood(Likelihood):
    """Example subclass: Student-t photometry for outlier-robust fitting.

    This is a stub demonstrating how to subclass Likelihood to customize
    the likelihood computation. In this example, we override the base
    likelihood to use Student-t instead of Gaussian for photometry.

    A real RobustLikelihood would:
    1. Override _build_base_likelihood to substitute StudentTLikelihood for phot
    2. Set degrees of freedom based on model/observation config
    3. Store the dof as an instance field for later reference

    Notes
    -----
    This is a minimal stub included in the Likelihood module to demonstrate
    that the extracted Likelihood seam supports inheritance. It is NOT
    fully implemented and is only used for testing that custom subclasses
    thread through Fitter without requiring further edits.
    """

    pass  # Subclass body is empty; full implementation deferred to a follow-up PR
