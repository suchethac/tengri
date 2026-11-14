# SPDX-License-Identifier: BSD-3-Clause
"""One data term: the Gaussian-family log-likelihood shared by every inference path.

:func:`build_data_term` returns a :class:`DataTerm` whose ``log_prob(params, data_args)``
is the positive log-likelihood ``log p(d | params)``. The per-galaxy Fitter composes this
function, and :mod:`tengri.inference.loss_functions` negates it for the minimizers.

The term reads no ``Fitter`` state. Its inputs are the physical model, the parameter spec,
the traced ``data_args`` bundle (see :func:`build_data_args`) and a
:class:`DataTermOptions`. Structure (which keys are present, the likelihood family, the
emission-line mode) is fixed at build time; values ride in ``data_args`` so one compiled
program serves every galaxy with the same structure.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import jax.numpy as jnp
import numpy as np

from tengri.inference._prediction import _build_prediction, _resolve_measured_line_defs
from tengri.inference.likelihoods.gaussian import (
    LOG_2PI,
    inv_noise_std,
    standardized_residual,
    student_t_log_norm_per_point,
)
from tengri.inference.likelihoods.protocol import log_normalization_of

__all__ = [
    "DataTerm",
    "DataTermContext",
    "DataTermOptions",
    "ResolvedDataTerm",
    "build_data_args",
    "build_data_term",
    "resolve_data_term_options",
]


@dataclass(frozen=True)
class DataTermOptions:
    """Structural choices that fix which data term is built.

    Parameters
    ----------
    data_type : str
        ``"photometry"``, ``"spectroscopy"`` or ``"joint"``.
    calibration_marginalize : bool, optional
        Whether the spectral calibration polynomial is marginalized.
    cal_n_poly : int, optional
        Order of the calibration polynomial when it is marginalized.
    cal_prior_sigma : float, optional
        Gaussian prior width on the calibration coefficients.
    eline_marginalize : bool or None, optional
        Emission-line marginalization. ``None`` takes the spectroscopy ``eline_mode``.
    eline_prior_type : str or None, optional
        Line-amplitude prior, ``"flat"`` or ``"model"``. ``None`` takes the config's value.
    use_components : bool, optional
        Route the forward model through its component-split path.

    Notes
    -----
    Frozen and hashable, so two terms built from equal options share a structure.
    """

    data_type: str
    calibration_marginalize: bool = False
    cal_n_poly: int = 3
    cal_prior_sigma: float = 1.0
    eline_marginalize: bool | None = None
    eline_prior_type: str | None = None
    use_components: bool = False

    @classmethod
    def from_fitter(cls, fitter: Any) -> DataTermOptions:
        """Read the options from a Fitter-like object, defaulting any missing attribute.

        Parameters
        ----------
        fitter : object
            A ``Fitter`` or a duck-typed stand-in carrying the same private attributes.

        Returns
        -------
        DataTermOptions
            The options the fitter's data term is built from.
        """
        return cls(
            data_type=getattr(fitter, "data_type", "photometry"),
            calibration_marginalize=bool(getattr(fitter, "_calibration_marginalize", False)),
            cal_n_poly=getattr(fitter, "_cal_n_poly", 3),
            cal_prior_sigma=getattr(fitter, "_cal_prior_sigma", 1.0),
            eline_marginalize=getattr(fitter, "_eline_marginalize", None),
            eline_prior_type=getattr(fitter, "_eline_prior_type", None),
            use_components=bool(getattr(fitter, "use_components", False)),
        )


@dataclass(frozen=True)
class DataTermContext:
    """Fitter-free snapshot of the attributes the likelihood cohort reads.

    :mod:`tengri.inference.likelihood` builders accept any object with these attribute
    names, so the cohort is built from a plain record instead of a live ``Fitter``.

    Parameters
    ----------
    model : SEDModel
        The forward model.
    spec : Parameters
        The working parameter spec, including merged emission-line amplitudes.
    data, noise : array_like
        Observed values and their uncertainties.
    data_type : str
        ``"photometry"``, ``"spectroscopy"`` or ``"joint"``.
    data_mask : array_like or None
        Censoring flags, or ``None`` for an uncensored fit.
    data_args : dict
        The traced data bundle built by :func:`build_data_args`.
    has_spectroscopy : bool
        Whether the fit has a spectroscopic channel.
    fixed_values : dict
        Fixed parameter values, including any per-fit override.
    calibration_marginalize, cal_n_poly, cal_prior_sigma
        Calibration marginalization settings.
    eline_marginalize, eline_fitted, eline_prior_type, eline_prior_sigma,
    eline_prior_width_dex, eline_wavelengths, eline_independent_wavelengths,
    eline_constraint_matrix, eline_amplitude_names
        Emission-line settings, as produced by :func:`resolve_data_term_options`.

    Notes
    -----
    Values are snapshotted at construction. The Fitter builds its cohort once during
    ``__init__``, after every field here is final.
    """

    model: Any
    spec: Any
    data: Any
    noise: Any
    data_type: str
    data_mask: Any
    data_args: Mapping
    has_spectroscopy: bool
    fixed_values: Mapping
    calibration_marginalize: bool
    cal_n_poly: int
    cal_prior_sigma: float
    eline_marginalize: bool
    eline_fitted: bool
    eline_prior_type: str | None
    eline_prior_sigma: float | None
    eline_prior_width_dex: float | None
    eline_wavelengths: Any
    eline_independent_wavelengths: Any
    eline_constraint_matrix: Any
    eline_amplitude_names: list

    @classmethod
    def from_fitter(cls, fitter: Any) -> DataTermContext:
        """Snapshot a Fitter's attributes into a context.

        Parameters
        ----------
        fitter : Fitter
            A Fitter whose data arguments and emission-line state are final.

        Returns
        -------
        DataTermContext
            The record the likelihood cohort is built from.
        """
        return cls(
            model=fitter.model,
            spec=fitter.spec,
            data=fitter.data,
            noise=fitter.noise,
            data_type=fitter.data_type,
            data_mask=fitter.data_mask,
            data_args=fitter._data_args,
            has_spectroscopy=fitter._has_spectroscopy,
            fixed_values=fitter._fixed_values,
            calibration_marginalize=fitter._calibration_marginalize,
            cal_n_poly=fitter._cal_n_poly,
            cal_prior_sigma=fitter._cal_prior_sigma,
            eline_marginalize=fitter._eline_marginalize,
            eline_fitted=fitter._eline_fitted,
            eline_prior_type=fitter._eline_prior_type,
            eline_prior_sigma=fitter._eline_prior_sigma,
            eline_prior_width_dex=fitter._eline_prior_width_dex,
            eline_wavelengths=fitter._eline_wavelengths,
            eline_independent_wavelengths=fitter._eline_independent_wavelengths,
            eline_constraint_matrix=fitter._eline_constraint_matrix,
            eline_amplitude_names=fitter._eline_amplitude_names,
        )


@dataclass(frozen=True, eq=False)
class DataTerm:
    """A built data term: the positive log-likelihood of one structure.

    Parameters
    ----------
    data_args : dict
        The bundle the term was built against. Its keys and shapes fix the structure.
    likelihood : Likelihood or None
        The adapter cohort member that evaluates the channels, or ``None`` when the
        censored fall-through is used.
    options : DataTermOptions
        The resolved structural options.
    cache_key : tuple
        Hashable summary of ``options`` and the ``data_args`` key set and shapes.

    Notes
    -----
    :meth:`log_prob` closes over structure only. It is pure in ``(params, data_args)``,
    so ``jax.jit(term.log_prob)`` compiles once per shape set and a second call with new
    values reuses the program.
    """

    data_args: Mapping
    likelihood: Any
    options: DataTermOptions
    cache_key: tuple
    _log_lik: Callable = field(repr=False)
    _log_norm: Callable | None = field(repr=False, default=None)

    def log_prob(self, params: Mapping, data_args: Mapping):
        """Evaluate ``log p(d | params)``.

        Parameters
        ----------
        params : dict
            Resolved physical parameters, free and fixed, with mirrors merged.
        data_args : dict
            The traced data bundle, with the same structure as ``self.data_args``.

        Returns
        -------
        ndarray, scalar
            Log-likelihood [nats], rank 0.
        """
        return self._log_lik(params, data_args)

    def log_normalization(self, data_args: Mapping):
        """Return the constant :meth:`log_prob` omits, in nats (#3427).

        ``log_prob(params, data_args) + log_normalization(data_args)`` is the exact log
        density of the data. The constant depends on the data and the fixed structure
        only: a free noise parameter's ``log sigma_eff`` is already inside ``log_prob``.

        Parameters
        ----------
        data_args : dict
            The traced data bundle, with the same structure as ``self.data_args``.

        Returns
        -------
        scalar
            The normalization [nats]. Zero when the term has no normalization to add.
        """
        if self._log_norm is None:
            return 0.0
        return self._log_norm(data_args)


@dataclass(frozen=True, eq=False)
class ResolvedDataTerm:
    """Emission-line configuration after resolution against a model.

    Parameters
    ----------
    options : DataTermOptions
        The input options with ``eline_marginalize`` and ``eline_prior_type`` resolved.
    eline_fitted : bool
        Whether line amplitudes are explicit latent parameters.
    eline_wavelengths, eline_independent_wavelengths, eline_names, eline_constraint_matrix
        Catalog arrays, or ``None`` when no line channel is marginalized or fitted.
    eline_prior_sigma, eline_prior_width_dex : float
        Prior scales for the line block.
    eline_amplitude_names : list of str
        Latent names of the fitted line amplitudes.
    eline_amp_priors : dict
        Uniform priors to merge into the spec for the fitted amplitudes.
    """

    options: DataTermOptions
    eline_fitted: bool
    eline_wavelengths: Any
    eline_independent_wavelengths: Any
    eline_names: Any
    eline_constraint_matrix: Any
    eline_prior_sigma: float
    eline_prior_width_dex: float
    eline_amplitude_names: list
    eline_amp_priors: dict


def build_data_args(
    model: Any,
    data: Any,
    noise: Any,
    *,
    data_mask: Any = None,
    presence: Any = None,
    line_flux_data: Any = None,
    runtime_redshift: Any = None,
    params_override: Mapping | None = None,
    line_model_sigma: bool = True,
) -> dict:
    """Build the traced data bundle the data term and the JIT engines consume.

    Parameters
    ----------
    model : SEDModel
        The forward model. Its ``observation`` supplies the spectroscopy, ratio and index
        channels, and its spec supplies the fixed values for threading.
    data : array_like, shape (n_data,)
        Observed values.
    noise : array_like, shape (n_data,)
        Uncertainties of ``data``.
    data_mask : array_like, shape (n_data,), optional
        Censoring flags (0 detected, 1 upper limit, -1 lower limit).
    presence : array_like, shape (n_data,), optional
        Exact 0/1 band presence.
    line_flux_data : LineFluxData, optional
        Per-fit line-flux configuration. ``None`` falls back to the observation's own.
    runtime_redshift : float, optional
        Per-galaxy redshift routed in as a traced input.
    params_override : dict, optional
        Per-fit fixed-value overrides, merged into the threaded fixed values.
    line_model_sigma : bool, optional
        Whether the model's fractional line-flux sigma is packed.

    Returns
    -------
    dict
        Keys ``data``, ``noise``, ``sqrt_noise_inv`` and ``n_data``, plus whichever of
        ``data_mask``, ``presence``, ``redshift``, the spectroscopy covariance, the
        line-flux, line-ratio and index arrays, and ``_jit_inputs`` are configured.

    Notes
    -----
    The key set is structure: a key is present exactly when its channel is configured.
    Values are data and may change between calls without a recompile.
    """
    args = {
        "data": data,
        "noise": noise,
        "sqrt_noise_inv": inv_noise_std(noise),
        "n_data": jnp.int32(len(data)),
    }
    if data_mask is not None:
        args["data_mask"] = data_mask
    if presence is not None:
        args["presence"] = presence
    if runtime_redshift is not None:
        args["redshift"] = jnp.asarray(runtime_redshift)

    obs = getattr(model, "observation", None)
    if obs is not None:
        _add_observation_channels(args, obs, model, line_flux_data, line_model_sigma)
    _add_jit_inputs(args, model, params_override)
    return args


def _add_observation_channels(args, obs, model, line_flux_data, line_model_sigma):
    """Pack the covariance, line-flux, line-ratio and index channels into ``args``."""
    spec_cfg = getattr(obs, "spectroscopy", None)
    if spec_cfg is not None and getattr(spec_cfg, "has_covariance", False):
        if spec_cfg.has_banded_covariance:
            args["spec_chol_data"] = spec_cfg.chol_data
        else:
            args["spec_cov_inv"] = spec_cfg.cov_inv

    line_flux_cfg = (
        line_flux_data if line_flux_data is not None else getattr(obs, "line_fluxes", None)
    )
    if line_flux_cfg is not None:
        args["line_flux_obs"] = line_flux_cfg.fluxes
        args["line_flux_err"] = line_flux_cfg.errors
        args["line_flux_waves"] = line_flux_cfg.wavelengths
        limit_mask = getattr(line_flux_cfg, "limit_mask", None)
        if limit_mask is not None:
            args["line_flux_limit_mask"] = limit_mask
        model_sigma = _line_model_sigma(model, line_flux_cfg, line_model_sigma)
        if model_sigma is not None:
            args["line_flux_model_sigma_frac"] = jnp.asarray(model_sigma)

    line_ratio_cfg = getattr(obs, "line_ratios", None)
    if line_ratio_cfg is not None:
        args["line_ratio_obs"] = line_ratio_cfg.ratios
        args["line_ratio_err"] = line_ratio_cfg.errors

    index_cfg = getattr(obs, "spectral_indices", None)
    if index_cfg is not None:
        args["index_obs"] = index_cfg.values
        args["index_err"] = index_cfg.errors


def _line_model_sigma(model, line_flux_cfg, enabled):
    """Return the model's fractional line sigma, or ``None`` when it is disabled or zero."""
    if not enabled:
        return None
    hook = getattr(model, "line_model_sigma_frac", None)
    if hook is None:
        return None
    sigma = np.asarray(hook(line_flux_cfg.wavelengths), dtype=np.float64)
    if not np.any(sigma > 0.0):
        return None
    return sigma


def _add_jit_inputs(args, model, params_override):
    """Thread the SSP grid, template arrays and fixed values under ``_jit_inputs``."""
    _supports = getattr(model, "_supports_jit_threading", None)
    _threadable = _supports() if callable(_supports) else True
    if not _threadable or not all(
        hasattr(model, attr)
        for attr in ("spec", "ssp_data", "_template_data_for_jit", "_ztable_data_for_jit")
    ):
        return
    jit_fixed_values = dict(model.spec.get_fixed_values())
    if params_override is not None:
        jit_fixed_values.update(params_override)
    args["_jit_inputs"] = {
        "fixed_values": jit_fixed_values,
        "ssp_data": model.ssp_data,
        "template_data": model._template_data_for_jit(),
        "ztable_data": model._ztable_data_for_jit(),
    }


def resolve_data_term_options(model: Any, options: DataTermOptions) -> ResolvedDataTerm:
    """Resolve emission-line defaults from the model's spectroscopy config.

    Parameters
    ----------
    model : SEDModel
        The forward model whose ``observation.spectroscopy`` carries ``eline_mode``,
        ``eline_prior_type`` and the line catalog.
    options : DataTermOptions
        The requested options. ``eline_marginalize`` and ``eline_prior_type`` may be
        ``None`` to take the config's value.

    Returns
    -------
    ResolvedDataTerm
        The resolved options and, when a line channel is marginalized or fitted, the
        catalog arrays and fitted-amplitude priors.

    Notes
    -----
    Marginalization and fitted modes both require spectroscopy: a photometry-only fit
    resolves both to ``False`` whatever the config says.
    """
    obs = getattr(model, "observation", None)
    spec_config = obs.spectroscopy if obs is not None else None
    has_spec = options.data_type in ("spectroscopy", "joint")
    has_mode = spec_config is not None and hasattr(spec_config, "eline_mode")

    eline_marginalize = options.eline_marginalize
    if eline_marginalize is None:
        eline_marginalize = has_mode and spec_config.eline_mode == "marginalized"
    eline_marginalize = bool(eline_marginalize) and has_spec

    eline_fitted = bool(has_mode and spec_config.eline_mode == "fitted") and has_spec

    eline_prior_type = options.eline_prior_type
    if eline_prior_type is None:
        if spec_config is not None and hasattr(spec_config, "eline_prior_type"):
            _raw = spec_config.eline_prior_type
            eline_prior_type = _raw if isinstance(_raw, str) else "flat"
        else:
            eline_prior_type = "flat"

    resolved = ResolvedDataTerm(
        options=replace(
            options,
            eline_marginalize=eline_marginalize,
            eline_prior_type=eline_prior_type,
        ),
        eline_fitted=eline_fitted,
        eline_wavelengths=None,
        eline_independent_wavelengths=None,
        eline_names=None,
        eline_constraint_matrix=None,
        eline_prior_sigma=100.0,
        eline_prior_width_dex=0.3,
        eline_amplitude_names=[],
        eline_amp_priors={},
    )
    if not (eline_marginalize or eline_fitted):
        return resolved
    return _resolve_eline_catalog(resolved, spec_config)


def _resolve_eline_catalog(resolved: ResolvedDataTerm, spec_config: Any) -> ResolvedDataTerm:
    """Fill the catalog arrays, constraint matrix and fitted-amplitude priors."""
    from tengri.observation.line_list import LineList

    if spec_config is not None and spec_config.eline_catalog is not None:
        catalog = spec_config.effective_catalog
    else:
        catalog = LineList.default_13()

    fix_doublets = True
    if spec_config is not None and hasattr(spec_config, "eline_fix_doublets"):
        fix_doublets = spec_config.eline_fix_doublets
    if fix_doublets:
        constraint = catalog.build_constraint_matrix()
    else:
        constraint = jnp.eye(catalog.n_lines)

    prior_sigma = getattr(spec_config, "eline_prior_sigma", 100.0) if spec_config else 100.0
    prior_width = getattr(spec_config, "eline_prior_width_dex", 0.3) if spec_config else 0.3

    amplitude_names: list = []
    amp_priors: dict = {}
    if resolved.eline_fitted:
        from tengri.observation.eline_marginalization import eline_amplitude_names
        from tengri.parameters.priors import Uniform

        amplitude_names = eline_amplitude_names(catalog)
        amp_bound = 10.0 * prior_sigma
        amp_priors = {nm: Uniform(-amp_bound, amp_bound) for nm in amplitude_names}

    return replace(
        resolved,
        eline_wavelengths=catalog.wavelengths,
        eline_independent_wavelengths=catalog.independent_wavelengths,
        eline_names=catalog.names,
        eline_constraint_matrix=constraint,
        eline_prior_sigma=prior_sigma,
        eline_prior_width_dex=prior_width,
        eline_amplitude_names=amplitude_names,
        eline_amp_priors=amp_priors,
    )


def _data_term_cache_key(options: DataTermOptions, data_args: Mapping) -> tuple:
    """Return the hashable structure summary of a data term."""
    shapes = tuple(
        sorted(
            (k, tuple(jnp.shape(v))) for k, v in data_args.items() if not isinstance(v, Mapping)
        )
    )
    return (options, tuple(sorted(data_args.keys())), shapes)


def build_data_term(
    model: Any,
    *,
    spec: Any = None,
    data_args: Mapping,
    options: DataTermOptions,
    likelihood: Any = None,
    fixed_values: Mapping | None = None,
) -> DataTerm:
    """Build the positive data term for one model, structure and likelihood family.

    Parameters
    ----------
    model : SEDModel
        The forward model. Its ``observation`` gives the line-index definitions, and its
        ``_get_or_build_predict_observables_jit`` family gives the threaded forward.
    spec : Parameters, optional
        The parameter spec the physical parameters come from. Defaults to ``model.spec``;
        a Fitter passes its working spec, which can carry merged emission-line amplitudes.
    data_args : dict
        The build-time bundle. Its key set selects the channels; its values are used only
        for the eager channel-scale check.
    options : DataTermOptions
        Structural options, with emission-line fields already resolved.
    likelihood : Likelihood, optional
        The likelihood adapter cohort member. ``None`` uses the censored fall-through,
        which is only valid when ``data_args`` carries ``data_mask``.

    Returns
    -------
    DataTerm
        A term whose ``log_prob(params, data_args)`` is ``log p(d | params)`` [nats].

    Raises
    ------
    AssertionError
        If the fall-through is reached without a ``data_mask``: the auto-built cohort is
        missing coverage for that configuration.

    Notes
    -----
    Takes **physical** parameters. The unstandardize transform belongs to the callers.
    The likelihood adapter receives ``data_args`` at call time, so one compiled program
    serves every galaxy that shares this structure. Marginals go through
    :mod:`tengri.observation.linear_marginal` inside the adapters.

    **JIT compatibility**: the returned ``log_prob`` is pure and safe inside
    :func:`jax.jit`, :func:`jax.grad` and :func:`jax.vmap`.
    """
    from tengri.observation.noise import (
        censored_neg_log_likelihood,
        get_noise_dof,
        uses_student_t,
    )

    spec = model.spec if spec is None else spec
    fixed_values = spec.get_fixed_values() if fixed_values is None else fixed_values
    data_type = options.data_type
    free_names = set(spec.free_params)
    noise_dof = get_noise_dof(spec) if uses_student_t(spec) else None
    use_censored = "data_mask" in data_args
    has_line_fluxes = "line_flux_waves" in data_args
    has_line_ratios = "line_ratio_obs" in data_args
    has_indices = "index_obs" in data_args
    index_defs = None
    if has_indices:
        obs_for_idx = getattr(model, "observation", None)
        if obs_for_idx is not None and obs_for_idx.spectral_indices is not None:
            index_defs = obs_for_idx.spectral_indices.index_defs
    measured_line_defs = _resolve_measured_line_defs(model, has_line_fluxes=has_line_fluxes)
    needs_line_lums = options.eline_prior_type == "model"
    use_components = options.use_components
    # The likelihood adapters take ``data_args`` at call time; a user likelihood with the
    # two-argument signature keeps working on its baked arrays.
    _likelihood_takes_data_args = likelihood is not None and "data_args" in _signature_parameters(
        likelihood.log_prob
    )

    # Template threading: every channel routes through the threaded orchestrator so the
    # outer JIT trace sees the SSP grid and template arrays as Parameters, not Constants.
    _threaded_impl = (
        None
        if use_components or not hasattr(model, "_get_or_build_predict_observables_jit")
        else model._get_or_build_predict_observables_lines_jit()
        if needs_line_lums
        else model._get_or_build_predict_observables_jit()
    )

    if likelihood is not None:
        _check_reference_channel_scales(
            model,
            spec,
            likelihood,
            data_args,
            options,
            fixed_values=fixed_values,
            has_line_fluxes=has_line_fluxes,
            has_indices=has_indices,
            index_defs=index_defs,
            has_line_ratios=has_line_ratios,
            measured_line_defs=measured_line_defs,
            needs_line_lums=needs_line_lums,
        )

    def log_lik(params, call_data_args):
        """log p(d | params), positive, for one structure."""
        prediction, predicted, _pred_phot, _pred_spec = _build_prediction(
            model,
            params,
            data_type,
            has_line_fluxes=has_line_fluxes,
            has_indices=has_indices,
            index_defs=index_defs,
            data_args=call_data_args,
            use_components=use_components,
            has_line_ratios=has_line_ratios,
            measured_line_defs=measured_line_defs,
            jit_inputs=call_data_args.get("_jit_inputs"),
            threaded_impl=_threaded_impl,
            needs_line_lums=needs_line_lums,
        )

        if likelihood is not None:
            if _likelihood_takes_data_args:
                return likelihood.log_prob(prediction, params, data_args=call_data_args)
            return likelihood.log_prob(prediction, params)

        return _censored_fallthrough(
            model,
            params,
            call_data_args,
            predicted,
            free_names=free_names,
            noise_dof=noise_dof,
            has_line_fluxes=has_line_fluxes,
            has_line_ratios=has_line_ratios,
            has_indices=has_indices,
            index_defs=index_defs,
            use_censored=use_censored,
            censored_nll=censored_neg_log_likelihood,
        )

    def log_norm(call_data_args):
        """The constant ``log_lik`` omits, for the same structure (#3427)."""
        if likelihood is not None:
            return log_normalization_of(likelihood, call_data_args)
        return _censored_fallthrough_normalization(
            call_data_args,
            noise_dof=noise_dof,
            has_extra_channels=has_line_fluxes or has_line_ratios or has_indices,
        )

    return DataTerm(
        data_args=data_args,
        likelihood=likelihood,
        options=options,
        cache_key=_data_term_cache_key(options, data_args),
        _log_lik=log_lik,
        _log_norm=log_norm,
    )


def _censored_fallthrough_normalization(data_args, *, noise_dof, has_extra_channels):
    """Normalization of the censored fall-through: its detected photometric bands (#3427).

    The fall-through's censored energy carries ``log sigma_eff`` for the detected bands
    already. Its line-flux, line-ratio and index channels are chi-squared forms with no
    normalization here, so their constants are not included and a warning says so.
    """
    if "data_mask" not in data_args:
        return 0.0
    if has_extra_channels:
        warnings.warn(
            "The censored fall-through has line-flux, line-ratio or index channels; their "
            "normalization constants are not in log_evidence (#3427).",
            stacklevel=3,
        )
    n_detected = jnp.sum(data_args["data_mask"] == 0)
    if noise_dof is not None:
        # The Student-t density carries its own -(1/2) log(nu pi); no separate 2 pi.
        return n_detected * student_t_log_norm_per_point(noise_dof)
    return -0.5 * n_detected * LOG_2PI


def _signature_parameters(fn) -> set:
    """Return the parameter names of ``fn``."""
    import inspect

    return set(inspect.signature(fn).parameters)


def _check_reference_channel_scales(
    model,
    spec,
    likelihood,
    data_args,
    options,
    fixed_values,
    *,
    has_line_fluxes,
    has_indices,
    index_defs,
    has_line_ratios,
    measured_line_defs,
    needs_line_lums,
):
    """Evaluate every likelihood channel once at a reference draw and refuse mismatched scales.

    A units-mismatched channel makes a chi-squared that silently annihilates the others by
    floating-point absorption. This turns that into a construction error naming the channel.
    """
    import jax as _jax

    from tengri.inference.composite_likelihood import (
        CompositeLikelihood,
        _check_channel_scales,
    )

    ref_params = dict(spec.sample(_jax.random.PRNGKey(0)))
    ref_params.update(fixed_values)
    ref_prediction, _, _, _ = _build_prediction(
        model,
        ref_params,
        options.data_type,
        has_line_fluxes=has_line_fluxes,
        has_indices=has_indices,
        index_defs=index_defs,
        data_args=data_args,
        use_components=options.use_components,
        has_line_ratios=has_line_ratios,
        measured_line_defs=measured_line_defs,
        needs_line_lums=needs_line_lums,
    )
    channels = (
        likelihood.likelihoods if isinstance(likelihood, CompositeLikelihood) else (likelihood,)
    )
    _check_channel_scales(channels, ref_prediction, ref_params, data_args)


def _censored_fallthrough(
    model,
    params,
    data_args,
    predicted,
    *,
    free_names,
    noise_dof,
    has_line_fluxes,
    has_line_ratios,
    has_indices,
    index_defs,
    use_censored,
    censored_nll,
):
    """Log-likelihood for ``data_mask`` with no adapter: censored photometry plus line extras."""
    if not use_censored:
        raise AssertionError(
            "Legacy χ² fall-through reached for an unexpected configuration. "
            "Auto-build (Fitter._maybe_build_default_likelihood) returned None "
            "but use_censored is False. Either extend the auto-build cohort to "
            "cover this configuration, or raise NotImplementedError at the "
            "auto-build bail-out site rather than falling through here."
        )
    data = data_args["data"]
    noise = data_args["noise"]
    ll = -censored_nll(
        data,
        noise,
        predicted,
        data_args["data_mask"],
        f_cal=params.get("noise_frac_cal", 0.0),
        dof=noise_dof,
    )
    free_params = {k: v for k, v in params.items() if k in free_names}
    if has_line_fluxes:
        ll = ll + _line_flux_extra(model, params, free_params, data_args, censored_nll)
    if has_line_ratios:
        model_lr = model.predict_line_ratios(free_params, model.observation.line_ratios)
        chi2_ratios = jnp.sum(
            standardized_residual(
                data_args["line_ratio_obs"], model_lr, data_args["line_ratio_err"]
            )
            ** 2
        )
        ll = ll - 0.5 * chi2_ratios
    if has_indices:
        model_idx = model.predict_spectral_indices(free_params, index_defs)
        chi2_idx = jnp.sum(
            standardized_residual(data_args["index_obs"], model_idx, data_args["index_err"]) ** 2
        )
        ll = ll - 0.5 * chi2_idx
    return ll


def _line_flux_extra(model, params, free_params, data_args, censored_nll):
    """Line-flux term of the fall-through: censored when limits or model sigma are packed."""
    model_lf = model.predict_line_fluxes(
        free_params, target_wavelengths=data_args["line_flux_waves"]
    )
    model_lf = model_lf * params.get("line_flux_scaling", 1.0)
    limit_mask = data_args.get("line_flux_limit_mask")
    model_sigma = data_args.get("line_flux_model_sigma_frac")
    if limit_mask is None and model_sigma is None:
        chi2_lines = jnp.sum(
            standardized_residual(data_args["line_flux_obs"], model_lf, data_args["line_flux_err"])
            ** 2
        )
        return -0.5 * chi2_lines
    if limit_mask is None:
        limit_mask = jnp.zeros(data_args["line_flux_obs"].shape[0], dtype=jnp.int32)
    return -censored_nll(
        data_args["line_flux_obs"],
        data_args["line_flux_err"],
        model_lf,
        limit_mask,
        model_sigma_frac=model_sigma,
    )
