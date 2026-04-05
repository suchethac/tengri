"""Inference engine: fit observed data using MAP, NUTS, Ray Tracing, or geoVI.

The Fitter separates inference strategy from the forward model. It builds
a loss function from the Model's predictions and the ParamSpec's priors,
then runs the chosen optimizer/sampler.

Usage:
    from tengri import Model, Fitter

    fitter = Fitter(model, data, noise)
    result_map = fitter.run("map", n_steps=1500)
    result_rts = fitter.run("raytrace", init_from=result_map)
    result_nuts = fitter.run("nuts", init_from=result_map, n_warmup=500)
"""

from __future__ import annotations

import time
import warnings

import jax
import jax.numpy as jnp

from tengri.distributions import Gaussian, LogUniform
from tengri.utils.transforms import to_bounded, to_unbounded

# ---------------------------------------------------------------------------
# Method name unification
# ---------------------------------------------------------------------------

# Maps deprecated/old method strings → new canonical names.
_DEPRECATED_METHOD_ALIASES: dict[str, str] = {
    "geovi": "vi",
    "native_geovi": "vi",
    "mgvi": "vi_linear",
    "native_mgvi": "vi_linear",
    "evi": "vi_linear",
    "native_evi": "vi_linear",
    "fast_geovi": "vi_nifty",
    "nifty_geovi": "vi_nifty",
    "fast_mgvi": "vi_nifty_linear",
    "nifty_mgvi": "vi_nifty_linear",
    "raytrace": "mcmc_raytrace",
    "nuts": "mcmc_nuts",
    "elliptical_slice": "mcmc_ess",
    "nss": "evidence",
    "geovi_nuts": "vi",
}

# Threshold for "mcmc" auto-selection: low-D → NUTS, high-D → Ray Tracing.
_MCMC_AUTO_D_THRESHOLD = 20

# Threshold for "auto" method selection.
_AUTO_D_THRESHOLDS = (15, 50)  # (laplace_max, vi_linear_max)


def _simple_cg(mat_fn, b, x0, maxiter=30, miniter=6):
    """Lightweight CG solve for catalog fitting. JIT-friendly."""
    _eps = 6.0 * jnp.finfo(jnp.float64).eps
    r = mat_fn(x0) - b
    d = r
    gamma = jnp.dot(r, r)
    energy = jnp.dot((r - b) / 2, x0)
    init = (x0, r, d, gamma, energy, jnp.int32(-2), jnp.int32(0))

    def cond(s):
        return s[5] < -1

    def body(s):
        x, r, d, pg, pe, info, i = s
        i = i + 1
        q = mat_fn(d)
        curv = jnp.dot(d, q)
        alpha = pg / curv
        info = jnp.where(curv <= 0.0, jnp.int32(0), info)
        alpha = jnp.where(curv <= 0.0, 0.0, alpha)
        x = x - alpha * d
        r = jnp.where((i % 20 == 0) & (info < -1), mat_fn(x) - b, r - alpha * q)
        gamma = jnp.dot(r, r)
        energy = jnp.dot((r - b) / 2, x)
        ed = pe - energy
        info = jnp.where(ed < -_eps * jnp.abs(energy), jnp.int32(-1), info)
        info = jnp.where((ed < 1e-4) & (i >= miniter) & (info < -1), jnp.int32(0), info)
        info = jnp.where((i >= maxiter) & (info < -1), i, info)
        d = d * jnp.maximum(0.0, gamma / (pg + 1e-30)) + r
        return (x, r, d, gamma, energy, info, i)

    return jax.lax.while_loop(cond, body, init)[0]


class Fitter:
    """Inference engine for tengri models.

    Parameters
    ----------
    model : Model
        Configured forward model.
    data : array
        Observed data (photometry or spectrum).
    noise : array
        1-sigma uncertainties.
    data_type : str or None
        ``"photometry"``, ``"spectroscopy"``, or ``"joint"``.
        If ``None`` (default), inferred from ``model.observation``.
        Explicit values override the inferred type.
    calibration_marginalize : bool
        If ``True``, analytically marginalize over spectroscopic
        calibration polynomial coefficients (Chebyshev) when computing
        the spectroscopic log-likelihood.  Only applies when
        ``data_type`` is ``"spectroscopy"`` or ``"joint"``.
        Follows the Prospector approach (Johnson et al. 2021).
        Default ``False``.
    cal_n_poly : int
        Number of Chebyshev polynomial coefficients for calibration
        marginalization (order 1 through ``cal_n_poly``).  Default 3.
    cal_prior_sigma : float
        Standard deviation of the Gaussian prior on each calibration
        coefficient.  Default 1.0.
    eline_marginalize : bool or None
        Whether to analytically marginalize emission line amplitudes.
        ``None`` (default) auto-detects from the model's ``SpectroscopyConfig``
        (uses ``eline_mode="marginalized"`` setting).
    eline_prior_type : str or None
        Prior type for emission lines: ``"flat"`` or ``"cloudy"``.
        ``None`` auto-detects from ``SpectroscopyConfig.eline_prior_type``.
    """

    def __init__(
        self,
        model,
        data,
        noise,
        data_type=None,
        calibration_marginalize=False,
        cal_n_poly=3,
        cal_prior_sigma=1.0,
        eline_marginalize=None,
        eline_prior_type=None,
    ):
        self.model = model
        self.data = jnp.asarray(data)
        self.noise = jnp.asarray(noise)

        # Infer data_type from Observation if not provided
        if data_type is None:
            obs = getattr(model, "observation", None)
            if obs is not None:
                data_type = obs.data_type
            else:
                data_type = "photometry"  # backward compat default

        self.data_type = data_type
        self.spec = model.spec

        # Calibration marginalization settings
        self._calibration_marginalize = calibration_marginalize
        self._cal_n_poly = cal_n_poly
        self._cal_prior_sigma = cal_prior_sigma
        self._has_spectroscopy = data_type in ("spectroscopy", "joint")

        # Emission line marginalization settings
        _spec_config = getattr(model, "_spectroscopy_config", None)
        # Also try model.observation.spectroscopy if _spectroscopy_config not set
        if _spec_config is None:
            obs = getattr(model, "observation", None)
            if obs is not None:
                _spec_config = getattr(obs, "spectroscopy", None)

        if eline_marginalize is None:
            if _spec_config is not None and hasattr(_spec_config, "eline_mode"):
                eline_marginalize = _spec_config.eline_mode == "marginalized"
            else:
                eline_marginalize = False

        self._eline_marginalize = bool(eline_marginalize) and self._has_spectroscopy

        if eline_prior_type is None:
            if _spec_config is not None and hasattr(_spec_config, "eline_prior_type"):
                eline_prior_type = _spec_config.eline_prior_type
            else:
                eline_prior_type = "flat"
        self._eline_prior_type = eline_prior_type

        # Precompute static arrays for emission line fitting
        if self._eline_marginalize:
            from tengri.models.observation.line_catalog import LineCatalog

            if _spec_config is not None and _spec_config.eline_catalog is not None:
                _catalog = _spec_config.effective_catalog
            else:
                _catalog = LineCatalog.default_13()
            self._eline_wavelengths = _catalog.wavelengths
            self._eline_independent_wavelengths = _catalog.independent_wavelengths
            self._eline_names = _catalog.names
            fix_doublets = True
            if _spec_config is not None and hasattr(_spec_config, "eline_fix_doublets"):
                fix_doublets = _spec_config.eline_fix_doublets
            if fix_doublets:
                self._eline_constraint_matrix = _catalog.build_constraint_matrix()
            else:
                self._eline_constraint_matrix = jnp.eye(_catalog.n_lines)
            self._eline_prior_sigma = (
                getattr(_spec_config, "eline_prior_sigma", 100.0) if _spec_config else 100.0
            )
            self._eline_prior_width_dex = (
                getattr(_spec_config, "eline_prior_width_dex", 0.3) if _spec_config else 0.3
            )
        else:
            self._eline_wavelengths = None
            self._eline_independent_wavelengths = None
            self._eline_names = None
            self._eline_constraint_matrix = None
            self._eline_prior_sigma = 100.0
            self._eline_prior_width_dex = 0.3

        # Consistency check: SpectroscopyConfig.eline_broad vs ParamSpec.eline_broad
        if _spec_config is not None and getattr(_spec_config, "eline_broad", False):
            spec_has_broad = getattr(self.spec, "eline_broad", False)
            if not spec_has_broad:
                import warnings

                warnings.warn(
                    "SpectroscopyConfig has eline_broad=True but ParamSpec was built with "
                    "eline_broad=False. The broad-component velocity dispersion parameter "
                    "'eline_broad_sigma_kms' will not be sampled. "
                    "Pass eline_broad=True to ParamSpec() to fix this.",
                    UserWarning,
                    stacklevel=2,
                )

        # Separate free and fixed parameters
        self._free_names = self.spec.free_params
        self._fixed_values = self.spec.get_fixed_values()

        # Build bounds for free params
        self._bounds = {}
        for name in self._free_names:
            dist = self.spec.get_distribution(name)
            self._bounds[name] = dist.bounds

        # Pre-compute data-dependent arguments passed to JIT'd functions.
        # These are passed as explicit arguments (not closed over) so that
        # engines compiled for one galaxy can be reused for another with
        # the same model + parameter structure.
        noise_inv = 1.0 / self.noise**2
        self._data_args = {
            "data": self.data,
            "noise": self.noise,
            "noise_inv": noise_inv,
            "sqrt_noise_inv": jnp.sqrt(noise_inv),
            "n_data": jnp.int32(len(self.data)),
        }

        # JIT posterior sampler — call compile() to pre-compile, or it
        # compiles lazily on first VI run.
        self._jit_sampler = None

    def _engine_cache_key(self):
        """Return a hashable key identifying the JIT engine shape.

        Two Fitters sharing the same Model will reuse the same compiled
        engine if their cache keys match (same data_type, stochastic
        flag, latent dimension, data length, free parameter names, and
        noise model presence).
        """
        from tengri.core.noise import has_noise_model

        return (
            self.data_type,
            self.spec.stochastic,
            self.spec.n_grid if self.spec.stochastic else 0,
            len(self.data),
            tuple(sorted(self._free_names)),
            has_noise_model(self.spec),
            self._eline_marginalize,
            self._calibration_marginalize,
            self._eline_prior_type,
        )

    def _get_or_build_engine(self, pos_dict):
        """Return the JIT engine, reusing a cached version when possible.

        Engines are cached on the Model object so that multiple Fitters
        created with the same Model (but different data) share the same
        compiled XLA programs.
        """
        if self._jit_sampler is not None:
            return self._jit_sampler

        cache_key = self._engine_cache_key()
        if not hasattr(self.model, "_jit_engine_cache"):
            self.model._jit_engine_cache = {}
        if cache_key in self.model._jit_engine_cache:
            self._jit_sampler = self.model._jit_engine_cache[cache_key]
            return self._jit_sampler

        engine = self._build_jit_engine(pos_dict)
        self.model._jit_engine_cache[cache_key] = engine
        self._jit_sampler = engine
        return engine

    def summary(self) -> str:
        """Return a human-readable summary of the fitting problem.

        Returns
        -------
        str
            Formatted summary showing data shape, free parameters,
            priors, bounds, and available inference methods.
        """
        sep = "─" * 66
        lines: list[str] = [f"Fitter  data_type: {self.data_type}", sep]

        # Data shape
        n_data = self.data.shape[0]
        snr_med = float(jnp.median(jnp.abs(self.data / self.noise)))
        lines.append(f"  Data points: {n_data}")
        lines.append(f"  Median S/N:  {snr_med:.1f}")

        # Dimensionality
        n_free = len(self._free_names)
        n_grid = self.model._n_grid if self.model._has_field else 0
        dim_str = f"{n_free} free"
        if n_grid:
            dim_str += f" + {n_grid} latent (ξ)"
        lines.append(f"  Parameters:  {dim_str}")
        lines.append("")

        # Free parameter table
        hdr = f"  {'Parameter':<32s} {'Prior':<26s} {'Bounds'}"
        lines.append(hdr)
        lines.append("  " + "─" * 64)
        for name in self._free_names:
            dist = self.spec.get_distribution(name)
            lo, hi = dist.bounds
            lines.append(f"  {name:<32s} {dist!r:<26s} [{lo:.4g}, {hi:.4g}]")

        # Available methods
        lines.append("")
        lines.append(
            "  Methods:     map, raytrace, nuts, geovi, mgvi, geovi_nuts, "
            "laplace, pathfinder, elliptical_slice"
        )

        lines.append(sep)
        return "\n".join(lines)

    # -------------------------------------------------------------------
    # Loss function construction
    # -------------------------------------------------------------------

    def _build_loss_fn(self):
        """Build a differentiable loss function.

        The loss function takes an unbounded parameter dict **and a
        ``data_args`` dict** and returns a scalar: chi² + prior penalties.
        Observed data are passed as explicit arguments (not captured in
        the closure) so that the compiled XLA program can be reused across
        galaxies sharing the same model structure.

        Returns
        -------
        callable
            ``loss_fn(params_unbounded, data_args) -> scalar``
        """
        from tengri.core.noise import (
            get_noise_dof,
            has_noise_model,
            uses_student_t,
            variable_noise_hamiltonian,
        )

        model = self.model
        data_type = self.data_type
        free_names = self._free_names
        bounds = self._bounds
        fixed_values = self._fixed_values
        spec = self.spec
        stochastic = spec.stochastic
        use_variable_noise = has_noise_model(spec)
        noise_dof = get_noise_dof(spec) if uses_student_t(spec) else None
        use_cal_marg = self._calibration_marginalize and self._has_spectroscopy
        cal_n_poly = self._cal_n_poly
        cal_prior_sigma = self._cal_prior_sigma
        use_eline_marg = self._eline_marginalize
        eline_wavelengths = self._eline_wavelengths
        eline_independent_wavelengths = self._eline_independent_wavelengths
        eline_constraint_matrix = self._eline_constraint_matrix
        eline_prior_type = self._eline_prior_type
        eline_prior_sigma = self._eline_prior_sigma
        eline_prior_width_dex = self._eline_prior_width_dex

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
                predicted = model.predict_photometry(params)
            elif data_type == "spectroscopy":
                predicted = model.predict_spectrum(params, model._wave_obs)
            elif data_type == "joint":
                pred_phot = model.predict_photometry(params)
                pred_spec = model.predict_spectrum(params, model._wave_obs)
                predicted = jnp.concatenate([pred_phot, pred_spec])
            else:
                raise ValueError(f"Unknown data_type: {data_type}")

            # Likelihood energy — ordered most-specific first so combined branches
            # (eline+cal) are not shadowed by the less-specific cal-only branches.
            if use_eline_marg and use_cal_marg and data_type == "spectroscopy":
                # Both: marginalize lines first, then calibration on line-added prediction
                from tengri.models.observation.calibration import marginalize_calibration
                from tengri.models.observation.eline_marginalization import (
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
                    from tengri.models.observation.eline_priors import (
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
                from tengri.models.observation.calibration import marginalize_calibration
                from tengri.models.observation.eline_marginalization import (
                    apply_doublet_constraints,
                    build_eline_design_matrix,
                    marginalize_emission_lines,
                )

                z = params.get("redshift", fixed_values.get("redshift", 0.0))
                sigma_kms = params.get("eline_sigma_kms", 0.0)
                delta_v = params.get("eline_delta_v_kms", 0.0)
                resolution = getattr(model, "_spectral_resolution", None) or 2000.0
                n_phot = model.predict_photometry(params).shape[0]
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
                    from tengri.models.observation.eline_priors import (
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
                from tengri.models.observation.calibration import (
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
                from tengri.models.observation.calibration import (
                    marginalize_calibration,
                )

                n_phot = model.predict_photometry(params).shape[0]
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
                from tengri.models.observation.eline_marginalization import (
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
                    from tengri.models.observation.eline_priors import (
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
                from tengri.models.observation.eline_marginalization import (
                    apply_doublet_constraints,
                    build_eline_design_matrix,
                    marginalize_emission_lines,
                )

                z = params.get("redshift", fixed_values.get("redshift", 0.0))
                sigma_kms = params.get("eline_sigma_kms", 0.0)
                delta_v = params.get("eline_delta_v_kms", 0.0)
                resolution = getattr(model, "_spectral_resolution", None) or 2000.0
                n_phot = model.predict_photometry(params).shape[0]
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
                    from tengri.models.observation.eline_priors import (
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
            elif use_variable_noise:
                f_cal = params.get("noise_frac_cal", 0.0)
                e_lh = variable_noise_hamiltonian(data, noise, predicted, f_cal, dof=noise_dof)
            else:
                chi2 = jnp.sum(((data - predicted) / noise) ** 2)
                e_lh = 0.5 * chi2

            # Prior contributions
            prior_penalty = 0.0

            # Standard normal prior on psd_xi
            if stochastic and "psd_xi" in params_unbounded:
                prior_penalty += jnp.sum(params_unbounded["psd_xi"] ** 2)

            # Additional prior contributions for non-Uniform distributions
            for name in free_names:
                dist = spec.get_distribution(name)
                if isinstance(dist, Gaussian):
                    val = params[name]
                    prior_penalty -= 2.0 * dist.log_prob(val)
                elif isinstance(dist, LogUniform):
                    val = params[name]
                    # LogUniform correction: log_prob difference from Uniform
                    uniform_lp = -jnp.log(dist.hi - dist.lo)
                    prior_penalty -= 2.0 * (dist.log_prob(val) - uniform_lp)

            return e_lh + 0.5 * prior_penalty

        return loss_fn

    def _get_or_build_loss_fn(self):
        """Return the cached loss function, building it if needed.

        The loss function is cached on the Model object keyed by
        ``_engine_cache_key()`` so that multiple Fitters with the same
        model structure share the same compiled XLA program.
        """
        cache_key = self._engine_cache_key()
        if not hasattr(self.model, "_loss_fn_cache"):
            self.model._loss_fn_cache = {}
        if cache_key in self.model._loss_fn_cache:
            return self.model._loss_fn_cache[cache_key]
        loss_fn = self._build_loss_fn()
        self.model._loss_fn_cache[cache_key] = loss_fn
        return loss_fn

    def _build_logprior_fn(self):
        """Build a log-prior function in physical parameter space.

        Returns a function: dict of free params → scalar log-prior.
        """
        spec = self.spec
        free_names = self._free_names

        def logprior_fn(free_params):
            lp = 0.0
            for name in free_names:
                dist = spec.get_distribution(name)
                lp = lp + dist.log_prob(free_params[name])
            return lp

        return logprior_fn

    def _build_loglikelihood_fn(self):
        """Build a log-likelihood function in physical parameter space.

        Returns a function: ``(dict of free params, data_args) → scalar``.
        Fixed parameters are automatically merged.  Observed data are
        passed via ``data_args`` (not captured in the closure) for cache
        reuse across galaxies.
        """
        from tengri.core.noise import (
            get_noise_dof,
            has_noise_model,
            uses_student_t,
            variable_noise_hamiltonian,
        )

        model = self.model
        data_type = self.data_type
        fixed_values = self._fixed_values
        spec = self.spec
        use_variable_noise = has_noise_model(spec)
        noise_dof = get_noise_dof(spec) if uses_student_t(spec) else None
        use_cal_marg = self._calibration_marginalize and self._has_spectroscopy
        cal_n_poly = self._cal_n_poly
        cal_prior_sigma = self._cal_prior_sigma
        use_eline_marg = self._eline_marginalize
        eline_wavelengths = self._eline_wavelengths
        eline_independent_wavelengths = self._eline_independent_wavelengths
        eline_constraint_matrix = self._eline_constraint_matrix
        eline_prior_type = self._eline_prior_type
        eline_prior_sigma = self._eline_prior_sigma
        eline_prior_width_dex = self._eline_prior_width_dex

        def loglikelihood_fn(free_params, data_args):
            data = data_args["data"]
            noise = data_args["noise"]

            # Merge free + fixed
            params = dict(free_params)
            for name, val in fixed_values.items():
                params[name] = val

            # Forward model prediction
            if data_type == "photometry":
                predicted = model.predict_photometry(params)
            elif data_type == "spectroscopy":
                predicted = model.predict_spectrum(params, model._wave_obs)
            elif data_type == "joint":
                pred_phot = model.predict_photometry(params)
                pred_spec = model.predict_spectrum(params, model._wave_obs)
                predicted = jnp.concatenate([pred_phot, pred_spec])
            else:
                raise ValueError(f"Unknown data_type: {data_type}")

            # Log-likelihood — ordered most-specific first so combined branches
            # (eline+cal) are not shadowed by the less-specific cal-only branches.
            if use_eline_marg and use_cal_marg and data_type == "spectroscopy":
                # Both: marginalize lines first, then calibration on line-added prediction
                from tengri.models.observation.calibration import marginalize_calibration
                from tengri.models.observation.eline_marginalization import (
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
                    from tengri.models.observation.eline_priors import (
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
                from tengri.models.observation.calibration import marginalize_calibration
                from tengri.models.observation.eline_marginalization import (
                    apply_doublet_constraints,
                    build_eline_design_matrix,
                    marginalize_emission_lines,
                )

                z = params.get("redshift", fixed_values.get("redshift", 0.0))
                sigma_kms = params.get("eline_sigma_kms", 0.0)
                delta_v = params.get("eline_delta_v_kms", 0.0)
                resolution = getattr(model, "_spectral_resolution", None) or 2000.0
                n_phot = model.predict_photometry(params).shape[0]
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
                    from tengri.models.observation.eline_priors import (
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
                from tengri.models.observation.calibration import (
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
                from tengri.models.observation.calibration import (
                    marginalize_calibration,
                )

                n_phot = model.predict_photometry(params).shape[0]
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
                from tengri.models.observation.eline_marginalization import (
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
                    from tengri.models.observation.eline_priors import (
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
                from tengri.models.observation.eline_marginalization import (
                    apply_doublet_constraints,
                    build_eline_design_matrix,
                    marginalize_emission_lines,
                )

                z = params.get("redshift", fixed_values.get("redshift", 0.0))
                sigma_kms = params.get("eline_sigma_kms", 0.0)
                delta_v = params.get("eline_delta_v_kms", 0.0)
                resolution = getattr(model, "_spectral_resolution", None) or 2000.0
                n_phot = model.predict_photometry(params).shape[0]
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
                    from tengri.models.observation.eline_priors import (
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
            elif use_variable_noise:
                f_cal = params.get("noise_frac_cal", 0.0)
                return -variable_noise_hamiltonian(data, noise, predicted, f_cal, dof=noise_dof)
            else:
                chi2 = jnp.sum(((data - predicted) / noise) ** 2)
                return -0.5 * chi2

        return loglikelihood_fn

    def _get_or_build_loglikelihood_fn(self):
        """Return the cached log-likelihood function, building if needed."""
        cache_key = self._engine_cache_key()
        if not hasattr(self.model, "_loglik_fn_cache"):
            self.model._loglik_fn_cache = {}
        if cache_key in self.model._loglik_fn_cache:
            return self.model._loglik_fn_cache[cache_key]
        loglik_fn = self._build_loglikelihood_fn()
        self.model._loglik_fn_cache[cache_key] = loglik_fn
        return loglik_fn

    def _initialize_unbounded(self, key):
        """Create initial unbounded parameter dict."""
        params = {}
        keys = jax.random.split(key, len(self._free_names) + 1)

        for i, name in enumerate(self._free_names):
            dist = self.spec.get_distribution(name)
            if isinstance(dist, Gaussian):
                # Initialize at mu in unbounded space
                lo, hi = dist.bounds
                params[name] = to_unbounded(jnp.array(dist.mu), lo, hi)
            else:
                # Initialize near midpoint (u=0) with small perturbation
                params[name] = 0.1 * jax.random.normal(keys[i])

        if self.spec.stochastic:
            params["psd_xi"] = 0.1 * jax.random.normal(keys[-1], shape=(self.spec.n_grid,))

        return params

    def _unbounded_from_posterior(self, posterior):
        """Convert a Posterior's params to unbounded space for init."""
        params = {}
        for name in self._free_names:
            if name in posterior.params:
                lo, hi = self._bounds[name]
                val = jnp.clip(jnp.array(posterior.params[name]), lo + 1e-6, hi - 1e-6)
                params[name] = to_unbounded(val, lo, hi)
            else:
                params[name] = jnp.array(0.0)

        if self.spec.stochastic and "psd_xi" in posterior.params:
            params["psd_xi"] = posterior.params["psd_xi"]
        elif self.spec.stochastic:
            params["psd_xi"] = jnp.zeros(self.spec.n_grid)

        return params

    # -------------------------------------------------------------------
    # Convert unbounded samples to physical space
    # -------------------------------------------------------------------

    def _to_physical(self, params_unbounded):
        """Convert a single unbounded param dict to physical space."""
        params = {}
        for name in self._free_names:
            lo, hi = self._bounds[name]
            params[name] = to_bounded(params_unbounded[name], lo, hi)
        for name, val in self._fixed_values.items():
            params[name] = jnp.array(val)
        if self.spec.stochastic and "psd_xi" in params_unbounded:
            params["psd_xi"] = params_unbounded["psd_xi"]
        return params

    # -------------------------------------------------------------------
    # Posterior sampling
    # -------------------------------------------------------------------

    def _draw_posterior_samples(
        self,
        likelihood,
        pos_dict,
        key,
        n_samples,
        existing_samples,
        *,
        method="jit",
        verbose=True,
    ):
        """Draw posterior samples from the converged geoVI approximation.

        Parameters
        ----------
        method : str
            "jit" (default) — JIT-compiled CG solve, ~0.2ms/sample.
            "blackjax" — BlackJAX NUTS (independent MCMC, not geoVI).
            "nifty" — NIFTy draw_linear_residual (slow, ~540ms/sample).
        """
        if method == "jit":
            return self._draw_jit_samples(
                pos_dict, key, n_samples, existing_samples, verbose=verbose
            )
        if method == "blackjax":
            try:
                return self._draw_blackjax_samples(
                    likelihood, pos_dict, key, n_samples, existing_samples, verbose=verbose
                )
            except ImportError:
                if verbose:
                    print("  blackjax not installed, falling back to JIT sampling")
                return self._draw_jit_samples(
                    pos_dict, key, n_samples, existing_samples, verbose=verbose
                )
        return self._draw_nifty_samples(
            likelihood, pos_dict, key, n_samples, existing_samples, verbose=verbose
        )

    def _build_jit_engine(self, pos_dict):
        """Build JIT-compiled inference engine: optimizer + posterior sampler.

        Returns a dict with compiled functions for the full EVI pipeline.
        All functions operate on flat arrays and use jax.lax.while_loop
        for zero Python overhead.

        The geoVI path uses NIFTy's actual implementations of CG,
        Newton-CG, sample drawing, and nonlinear curving — imported
        directly and called within the JIT boundary. This ensures
        mathematical equivalence with ``jft.optimize_kl``.
        """
        from tengri.core.noise import (
            compute_std_inv,
            get_noise_dof,
            has_noise_model,
            uses_student_t,
            variable_noise_hamiltonian,
            variable_noise_metric_vec,
        )

        # Import NIFTy for the exact geoVI path
        try:
            from nifty8.re.evi import Samples as NiftySamples
            from nifty8.re.optimize_kl import OptimizeVI

            _has_nifty = True
        except ImportError:
            _has_nifty = False

        model = self.model
        data_type = self.data_type
        free_names = self._free_names
        bounds = self._bounds
        fixed_values = self._fixed_values
        stochastic = self.spec.stochastic
        # data/noise are NO LONGER captured here as local variables.
        # Instead they are passed at call-time via the ``data_args`` dict
        # so that the compiled engine can be reused across galaxies.
        use_variable_noise = has_noise_model(self.spec)
        noise_dof = get_noise_dof(self.spec) if uses_student_t(self.spec) else None

        # --- Signal response (physics only) ---
        def signal_response(primals):
            params = {}
            for name in free_names:
                lo, hi = bounds[name]
                params[name] = to_bounded(primals[name], lo, hi)
            for name, val in fixed_values.items():
                params[name] = val
            if stochastic and "psd_xi" in primals:
                params["psd_xi"] = primals["psd_xi"]
            if data_type == "photometry":
                return model.predict_photometry(params)
            elif data_type == "spectroscopy":
                return model.predict_spectrum(params, model._wave_obs)
            elif data_type == "joint":
                p = model.predict_photometry(params)
                s = model.predict_spectrum(params, model._wave_obs)
                return jnp.concatenate([p, s])
            raise ValueError(f"Unknown data_type: {data_type}")

        # --- Signal + noise response for variable noise ---
        if use_variable_noise:

            def signal_noise_response(primals, data_args):
                """Return (predicted, std_inv) tuple for variable noise metric."""
                params = {}
                for name in free_names:
                    lo, hi = bounds[name]
                    params[name] = to_bounded(primals[name], lo, hi)
                for name, val in fixed_values.items():
                    params[name] = val
                if stochastic and "psd_xi" in primals:
                    params["psd_xi"] = primals["psd_xi"]
                if data_type == "photometry":
                    predicted = model.predict_photometry(params)
                elif data_type == "spectroscopy":
                    predicted = model.predict_spectrum(params, model._wave_obs)
                elif data_type == "joint":
                    p = model.predict_photometry(params)
                    s = model.predict_spectrum(params, model._wave_obs)
                    predicted = jnp.concatenate([p, s])
                else:
                    raise ValueError(f"Unknown data_type: {data_type}")
                f_cal = params.get("noise_frac_cal", 0.0)
                noise = data_args["noise"]
                std_inv = compute_std_inv(noise, predicted, f_cal)
                return predicted, std_inv

        # --- Flatten/unflatten (static shapes) ---
        param_keys = sorted(pos_dict.keys())
        slices = []
        idx = 0
        for k in param_keys:
            arr = jnp.atleast_1d(pos_dict[k]).ravel()
            shape = jnp.atleast_1d(pos_dict[k]).shape
            slices.append((idx, idx + arr.shape[0], shape))
            idx += arr.shape[0]
        d_total = idx
        n_data = len(self.data)  # static shape — same for all galaxies with same obs

        def flatten(d):
            return jnp.concatenate([jnp.atleast_1d(d[k]).ravel() for k in param_keys])

        def unflatten(x):
            d = {}
            for i_k, k in enumerate(param_keys):
                start, end, shape = slices[i_k]
                val = jax.lax.dynamic_slice(x, (start,), (end - start,)).reshape(shape)
                if shape == (1,):
                    val = val[0]
                d[k] = val
            return d

        # --- Core primitives ---
        _eps = 6.0 * jnp.finfo(jnp.float64).eps

        if use_variable_noise:

            def metric_vec(xi, v, data_args):
                """GGN metric for VariableCovarianceGaussian likelihood."""
                data = data_args["data"]

                def _snr(primals):
                    return signal_noise_response(primals, data_args)

                return variable_noise_metric_vec(xi, v, _snr, data, unflatten, flatten)

            def hamiltonian(xi, data_args):
                """E_lh + 0.5 ||xi||^2 with variable noise (includes logdet)."""
                data = data_args["data"]
                noise = data_args["noise"]
                pred = signal_response(unflatten(xi))
                primals = unflatten(xi)
                params = {}
                for name in free_names:
                    lo, hi = bounds[name]
                    params[name] = to_bounded(primals[name], lo, hi)
                for name, val in fixed_values.items():
                    params[name] = val
                f_cal = params.get("noise_frac_cal", 0.0)
                return variable_noise_hamiltonian(
                    data, noise, pred, f_cal, dof=noise_dof
                ) + 0.5 * jnp.sum(xi**2)

        else:

            def metric_vec(xi, v, data_args):
                """M(xi) @ v = J^T N^{-1} J v + v."""
                noise_inv = data_args["noise_inv"]
                xi_d, v_d = unflatten(xi), unflatten(v)
                _, Jv = jax.jvp(signal_response, (xi_d,), (v_d,))
                _, vjp_fn = jax.vjp(signal_response, xi_d)
                return flatten(vjp_fn(noise_inv * Jv)[0]) + v

            def hamiltonian(xi, data_args):
                """H(xi) = 0.5 chi2 + 0.5 ||xi||^2."""
                data = data_args["data"]
                noise = data_args["noise"]
                pred = signal_response(unflatten(xi))
                chi2 = jnp.sum(((data - pred) / noise) ** 2)
                return 0.5 * chi2 + 0.5 * jnp.sum(xi**2)

        def H_vg(xi, data_args):
            """Hamiltonian value and gradient w.r.t. xi only."""
            return jax.value_and_grad(lambda x: hamiltonian(x, data_args))(xi)

        _tiny = 6.0 * jnp.finfo(jnp.float64).tiny
        _n_reset = 20

        def cg_solve(mat_fn, b, x0, maxiter=30, miniter=6, absdelta=0.0, resnorm=0.0):
            """CG solve: mat_fn(x) = b.

            Exact port of NIFTy ``_static_cg`` (conjugate_gradient.py:217-388)
            for flat arrays.  Residual-norm (L2) is the primary convergence
            criterion; energy-based absdelta is secondary.  Negative curvature
            on the first CG iteration triggers a steepest-descent fallback.
            """
            r = mat_fn(x0) - b
            d = r
            gamma = jnp.dot(r, r)
            energy = jnp.dot((r - b) / 2, x0)
            init_info = jnp.where(gamma == 0.0, jnp.int32(0), jnp.int32(-2))
            init = (x0, r, d, gamma, energy, init_info, jnp.int32(0))

            def cond(s):
                return s[5] < -1

            def body(s):
                pos, r, d, prev_gamma, prev_energy, info, i = s
                i = i + 1

                q = mat_fn(d)
                curv = jnp.dot(d, q)
                alpha = prev_gamma / curv

                # Negative / zero curvature (NIFTy cg:278-286)
                info = jnp.where(curv <= 0.0, jnp.int32(0), info)
                alpha = jnp.where(curv <= 0.0, 0.0, alpha)
                pos = pos - alpha * d
                # First iter + negative curvature: steepest-descent fallback
                pos = jnp.where(
                    (curv < 0.0) & (i <= 1),
                    prev_energy / (-curv) * (-b),
                    pos,
                )

                # Periodic residual reset (NIFTy cg:287-291)
                r_reset = mat_fn(pos) - b
                r_step = r - q * alpha
                r = jnp.where((i % _n_reset == 0) & (info < -1), r_reset, r_step)

                gamma = jnp.dot(r, r)

                # Tiny gamma (NIFTy cg:295)
                info = jnp.where(
                    (gamma >= 0.0) & (gamma <= _tiny) & (info != -1),
                    jnp.int32(0),
                    info,
                )

                # Residual norm -- PRIMARY (NIFTy cg:296-298, norm_ord=2)
                r_norm = jnp.sqrt(gamma)
                info = jnp.where(
                    (resnorm > 0.0) & (r_norm < resnorm) & (i >= miniter) & (info != -1),
                    jnp.int32(0),
                    info,
                )

                # Energy -- SECONDARY (NIFTy cg:301-313)
                energy = jnp.dot((r - b) / 2, pos)
                energy_diff = prev_energy - energy
                neg_energy_eps = -_eps * jnp.abs(energy)
                info = jnp.where(
                    energy_diff < neg_energy_eps,
                    jnp.where(info < -1, i, info),
                    info,
                )
                info = jnp.where(
                    (absdelta > 0.0) & (energy_diff < absdelta) & (i >= miniter) & (info != -1),
                    jnp.int32(0),
                    info,
                )

                # Maxiter (NIFTy cg:314)
                info = jnp.where((i >= maxiter) & (info != -1), i, info)

                # Update search direction (NIFTy cg:316)
                d = d * jnp.maximum(0.0, gamma / prev_gamma) + r

                return (pos, r, d, gamma, energy, info, i)

            return jax.lax.while_loop(cond, body, init)[0]

        # --- Posterior sampler: draw linear residuals ---
        def draw_residuals(pos_f, subkeys, data_args):
            """Draw n linear residual samples (vmapped)."""
            sqrt_ni = data_args["sqrt_noise_inv"]
            n_d = n_data  # static, captured at engine-build time

            def draw_one(subkey):
                k1, k2 = jax.random.split(subkey)
                eta_pr = jax.random.normal(k1, shape=(d_total,))
                eta_lh = jax.random.normal(k2, shape=(n_d,))
                _, vjp_fn = jax.vjp(signal_response, unflatten(pos_f))
                jt = flatten(vjp_fn(sqrt_ni * eta_lh)[0])
                return cg_solve(
                    lambda v: metric_vec(pos_f, v, data_args),
                    jt + eta_pr,
                    eta_pr,
                    maxiter=30,
                    miniter=6,
                    absdelta=1e-4,
                )

            return jax.vmap(draw_one)(subkeys)

        def _draw_batch_fn(pos_f, k, data_args):
            return draw_residuals(pos_f, k, data_args)

        draw_batch = jax.jit(jax.vmap(_draw_batch_fn, in_axes=(None, 0, None)))

        # --- geoVI: nonlinear coordinate transform primitives ---

        def transformation_flat(pos_f, data_args):
            """t(x) = sqrt(N^{-1}) @ f(x). Maps to whitened data-space."""
            sqrt_ni = data_args["sqrt_noise_inv"]
            return sqrt_ni * signal_response(unflatten(pos_f))

        def left_sqrt_metric_flat(pos_f, v_data, data_args):
            """L^T(pos) @ v = J^T(pos) @ sqrt(N^{-1}) @ v.

            Maps whitened data-space vector to parameter-space.
            Matches NIFTy's ``likelihood.left_sqrt_metric(pos, v)``
            for the Gaussian case.
            """
            sqrt_ni = data_args["sqrt_noise_inv"]
            _, vjp_fn = jax.vjp(signal_response, unflatten(pos_f))
            return flatten(vjp_fn(sqrt_ni * v_data)[0])

        def right_sqrt_metric_flat(pos_f, v_param, data_args):
            """L(pos) @ v = sqrt(N^{-1}) @ J(pos) @ v.

            Maps parameter-space vector to whitened data-space.
            Matches NIFTy's ``likelihood.right_sqrt_metric(pos, v)``
            for the Gaussian case.
            """
            sqrt_ni = data_args["sqrt_noise_inv"]
            _, Jv = jax.jvp(signal_response, (unflatten(pos_f),), (unflatten(v_param),))
            return sqrt_ni * Jv

        def draw_metric_sample(pos_f, subkey, data_args):
            """Draw one sample with covariance M = J^T N^{-1} J + I.

            This is ``draw_linear_residual(..., from_inverse=False)``
            in NIFTy. The metric sample is NOT CG-inverted.
            """
            sqrt_ni = data_args["sqrt_noise_inv"]
            n_d = n_data  # static, captured at engine-build time
            k1, k2 = jax.random.split(subkey)
            eta_pr = jax.random.normal(k1, shape=(d_total,))
            eta_lh = jax.random.normal(k2, shape=(n_d,))
            _, vjp_fn = jax.vjp(signal_response, unflatten(pos_f))
            jt = flatten(vjp_fn(sqrt_ni * eta_lh)[0])
            return jt + eta_pr

        def _newton_cg_flat(
            fun_and_grad,
            hessp,
            x0,
            custom_gradnorm=None,
            maxiter=10,
            miniter=0,
            xtol=1e-5,
            energy_reduction_factor=0.1,
        ):
            """Newton-CG with successive-halving line search.

            Exact port of NIFTy ``_static_newton_cg`` (optimize.py:285-449)
            for flat arrays.  Includes adaptive CG tolerance, steepest-descent
            reset after 5 line-search halvings, and custom gradient norm.
            """
            ncg_xtol = xtol * d_total  # NIFTy: xtol * size(x0)

            def gradnorm(v):
                if custom_gradnorm is not None:
                    return custom_gradnorm(v)
                return jnp.sum(jnp.abs(v))  # L1 norm (NIFTy default)

            energy, g = fun_and_grad(x0)
            init_state = (
                x0,
                energy,
                jnp.array(jnp.inf),
                g,
                jnp.where(maxiter == 0, jnp.int32(0), jnp.int32(-2)),
                jnp.int32(0),
            )

            def ncg_cond(state):
                return state[4] < -1

            def ncg_body(state):
                pos, energy, old_energy, g, status, i = state
                i = i + 1

                # Adaptive CG tolerance (NIFTy optimize.py:351-358)
                cg_abd_fallback = jnp.array(0.0, dtype=energy.dtype)
                cg_absdelta = jnp.where(
                    ~jnp.isinf(old_energy),
                    energy_reduction_factor * (old_energy - energy),
                    cg_abd_fallback,
                )
                cg_absdelta = jnp.array(cg_absdelta, dtype=energy.dtype)

                # CG resnorm (NIFTy optimize.py:359-360, norm_ord=1)
                mag_g = jnp.sum(jnp.abs(g))
                cg_resnorm = jnp.minimum(0.5, jnp.sqrt(mag_g)) * mag_g

                # CG solve (NIFTy: norm_ord=1, _raise_nonposdef=False)
                nat_g = cg_solve(
                    lambda v: hessp(pos, v),
                    g,
                    jnp.zeros_like(pos),
                    maxiter=min(200, 20 * d_total),
                    miniter=min(6, min(200, 20 * d_total)),
                    absdelta=cg_absdelta,
                    resnorm=cg_resnorm,
                )

                # Line search: successive halving (NIFTy optimize.py:452-523)
                # State: (status, iter, new_pos, new_energy, new_g,
                #         dd, grad_scaling, reset, nhev)
                ls_init = (
                    jnp.int32(-2),
                    jnp.int32(0),
                    pos,
                    jnp.array(jnp.inf),
                    g,
                    nat_g,
                    1.0,
                    jnp.bool_(False),
                    jnp.int32(0),
                )

                def ls_cond(ls):
                    return ls[0] < -1

                def ls_body(ls):
                    (
                        ls_st,
                        ls_i,
                        _np,
                        _ne,
                        _ng,
                        dd,
                        gs,
                        reset,
                        nhev,
                    ) = ls
                    new_pos = pos - gs * dd
                    new_e, new_g = fun_and_grad(new_pos)
                    ls_st = jnp.where(new_e <= energy, jnp.int32(0), ls_st)
                    gs = jnp.where(ls_st < -1, gs / 2.0, gs)
                    # Steepest descent reset at iteration 5
                    do_reset = (ls_i == 5) & (ls_st < -1)
                    reset = jnp.where(do_reset, jnp.bool_(True), reset)
                    gs = jnp.where(do_reset, 1.0, gs)
                    gam = jnp.dot(g, g)
                    curv = jnp.dot(g, hessp(pos, g))
                    sd_dd = gam / curv * g
                    dd = jnp.where(do_reset, sd_dd, dd)
                    nhev = nhev + do_reset.astype(jnp.int32)
                    # Abort after 8 iterations
                    do_abort = (ls_i == 8) & (ls_st < -1)
                    ls_st = jnp.where(do_abort, jnp.int32(-1), ls_st)
                    return (
                        ls_st,
                        ls_i + 1,
                        new_pos,
                        new_e,
                        new_g,
                        dd,
                        gs,
                        reset,
                        nhev,
                    )

                ls_result = jax.lax.while_loop(ls_cond, ls_body, ls_init)
                (
                    ls_status,
                    ls_iter,
                    new_pos,
                    new_energy,
                    new_g,
                    dd,
                    gs,
                    _reset,
                    _nhev,
                ) = ls_result

                status = jnp.where(ls_status != 0, jnp.int32(-1), status)

                # Update only if line search succeeded (NIFTy opt:381-385)
                success = status < -1
                old_energy = jnp.where(success, energy, old_energy)
                energy_out = jnp.where(success, new_energy, energy)
                energy_diff = jnp.where(success, old_energy - energy_out, 0.0)
                pos_out = jnp.where(success, new_pos, pos)
                g_out = jnp.where(success, new_g, g)
                gs_out = jnp.where(success, gs, 0.0)

                descent_norm = gs_out * gradnorm(dd)

                # absdelta convergence (NIFTy optimize.py:407-414)
                min_cond = (ls_iter < 2) & (i > miniter)
                status = jnp.where(
                    (energy_diff >= 0.0) & (energy_diff < 1e-3) & min_cond & (status != -1),
                    jnp.int32(0),
                    status,
                )
                # xtol convergence (NIFTy optimize.py:415-417)
                status = jnp.where(
                    (descent_norm <= ncg_xtol) & (i > miniter) & (status != -1),
                    jnp.int32(0),
                    status,
                )
                # maxiter (NIFTy optimize.py:418)
                status = jnp.where((i == maxiter) & (status < -1), i, status)

                return (pos_out, energy_out, old_energy, g_out, status, i)

            result = jax.lax.while_loop(ncg_cond, ncg_body, init_state)
            return result[0], result[1]

        def curve_residual(m, r_linear, metric_key, sign, data_args):
            """Nonlinearly update a linear residual to a geoVI curved residual.

            Exact port of NIFTy ``nonlinearly_update_residual``
            (evi.py:136-217) using ``_newton_cg_flat`` for the inner
            Newton-CG optimization.

            Parameters
            ----------
            m : flat array, expansion point
            r_linear : flat array, linear residual (covariance M^{-1})
            metric_key : PRNG key (same as used for draw_residuals)
            sign : +1.0 or -1.0 (for mirrored samples)
            data_args : dict, data-dependent arguments

            Returns
            -------
            flat array : curved residual (x_opt - m)
            """
            x0 = m + r_linear
            ms = sign * draw_metric_sample(m, metric_key, data_args)
            trafo_at_m = transformation_flat(m, data_args)

            def phi_vg(x):
                trafo_x = transformation_flat(x, data_args)
                delta_trafo = trafo_x - trafo_at_m
                g_x = (x - m) + left_sqrt_metric_flat(m, delta_trafo, data_args)
                r = ms - g_x
                val = 0.5 * jnp.dot(r, r)
                ngrad = r + left_sqrt_metric_flat(
                    x, right_sqrt_metric_flat(m, r, data_args), data_args
                )
                return val, -ngrad

            def phi_metric(x, v):
                tm = (
                    left_sqrt_metric_flat(m, right_sqrt_metric_flat(x, v, data_args), data_args)
                    + v
                )
                return (
                    left_sqrt_metric_flat(x, right_sqrt_metric_flat(m, tm, data_args), data_args)
                    + tm
                )

            # sampnorm (evi.py:178-181)
            def sampnorm(natgrad):
                fpp = right_sqrt_metric_flat(m, natgrad, data_args)
                return jnp.sqrt(jnp.dot(natgrad, natgrad) + jnp.dot(fpp, fpp))

            x_opt, _ = _newton_cg_flat(
                phi_vg,
                phi_metric,
                x0,
                custom_gradnorm=sampnorm,
                maxiter=3,
                miniter=0,
                xtol=1e-3,
                energy_reduction_factor=0.1,
            )
            return x_opt - m

        def draw_nonlinear_residuals(m, subkeys, data_args):
            """Draw geoVI nonlinear residuals: linear draw + curving + mirror.

            Returns (2*n_samples, D) array: curved residuals with mirrored pairs.
            Matches NIFTy's ``nonlinear_resample`` sample mode.
            """
            # First draw linear residuals
            linear_residuals = draw_residuals(m, subkeys, data_args)

            # Curve each residual and its mirror
            def curve_pair(r, subkey):
                r_pos = curve_residual(m, r, subkey, sign=1.0, data_args=data_args)
                r_neg = curve_residual(m, -r, subkey, sign=-1.0, data_args=data_args)
                return r_pos, r_neg

            pos_curved, neg_curved = jax.vmap(curve_pair)(linear_residuals, subkeys)
            return jnp.concatenate([pos_curved, neg_curved], axis=0)

        def update_nonlinear_residuals(m, prev_residuals, subkeys, data_args):
            """Re-curve existing residuals at updated expansion point.

            Takes 2*n_samples residuals (first half positive, second half
            negative mirrors) and re-applies geoVI curving at the new m.
            Matches NIFTy's ``nonlinear_update`` sample mode.
            """
            n_half = prev_residuals.shape[0] // 2
            r_pos = prev_residuals[:n_half]
            r_neg = prev_residuals[n_half:]

            def recurve_pair(r_p, r_n, subkey):
                new_p = curve_residual(m, r_p, subkey, sign=1.0, data_args=data_args)
                new_n = curve_residual(m, r_n, subkey, sign=-1.0, data_args=data_args)
                return new_p, new_n

            new_pos, new_neg = jax.vmap(recurve_pair)(r_pos, r_neg, subkeys)
            return jnp.concatenate([new_pos, new_neg], axis=0)

        # --- EVI optimizer: fully JIT'd optimize_kl ---
        def kl_vg(m, residuals, data_args):
            """KL value and gradient averaged over samples."""

            def single_vg(r):
                return H_vg(m + r, data_args)

            vals, grads = jax.vmap(single_vg)(residuals)
            return jnp.mean(vals), jnp.mean(grads, axis=0)

        def kl_metric(m, residuals, v, data_args):
            """KL metric-vector product averaged over samples."""

            def single_met(r):
                return metric_vec(m + r, v, data_args)

            return jnp.mean(jax.vmap(single_met)(residuals), axis=0)

        def evi_step(m, subkey, n_samples, data_args):
            """One EVI iteration: draw samples + Newton-CG KL minimize.

            Returns (m_new, kl_value).
            """
            # Draw linear residual samples + mirror
            sample_keys = jax.random.split(subkey, n_samples)
            residuals = draw_residuals(m, sample_keys, data_args)
            residuals = jnp.concatenate([residuals, -residuals], axis=0)

            # Newton-CG KL minimization (same path as evi_step_full)
            def _evi_kl_vg(m_cur):
                return kl_vg(m_cur, residuals, data_args)

            def _evi_kl_hessp(m_cur, v):
                return kl_metric(m_cur, residuals, v, data_args)

            m_opt, kl_val = _newton_cg_flat(
                _evi_kl_vg,
                _evi_kl_hessp,
                m,
                maxiter=10,
                miniter=0,
                xtol=1e-5,
                energy_reduction_factor=0.1,
            )
            return m_opt, kl_val

        def run_evi(init_pos, key, data_args, n_iterations, n_samples, kl_rtol):
            """Run EVI with automatic convergence detection.

            ``n_iterations`` is dynamic — uses ``jax.random.fold_in``
            for per-iteration keys so no pre-split is needed.
            """

            # State: (m, prev_kl, iteration, converged)
            def cond_fn(state):
                _m, _prev_kl, i, converged = state
                return (~converged) & (i < n_iterations)

            def body_fn(state):
                m, prev_kl, i, converged = state
                subkey = jax.random.fold_in(key, i)
                m_new, kl_val = evi_step(m, subkey, n_samples, data_args)
                # Relative KL change
                rel_change = jnp.abs(prev_kl - kl_val) / (jnp.abs(prev_kl) + 1e-10)
                # Converge if relative change < rtol and at least 5 iterations done
                converged = (rel_change < kl_rtol) & (i >= 5)
                return (m_new, kl_val, i + 1, converged)

            # First iteration (no convergence check)
            first_key = jax.random.fold_in(key, 0)
            m0, kl0 = evi_step(init_pos, first_key, n_samples, data_args)
            init_state = (m0, kl0, jnp.int32(1), jnp.bool_(False))

            m_final, _kl_final, n_iters, _ = jax.lax.while_loop(cond_fn, body_fn, init_state)
            return m_final, n_iters

        # --- geoVI optimizer: per-mode functions (no lax.switch) ---
        #
        # Each sample mode gets its own evi_step function so that JAX
        # compiles ONLY the code path actually used.  This avoids the
        # 56s compilation cost of tracing all three branches via
        # ``jax.lax.switch``.
        #
        # ``sample_mode`` is a **static** string argument: JAX caches
        # a separate compiled version for each mode.
        SAMPLE_LINEAR = jnp.int32(0)
        SAMPLE_NONLINEAR_RESAMPLE = jnp.int32(1)
        SAMPLE_NONLINEAR_UPDATE = jnp.int32(2)

        def _kl_minimize(m, residuals, constants_mask, data_args):
            """Newton-CG KL minimization with constants mask."""

            def _masked_kl_vg(m_cur, res):
                val, grad = kl_vg(m_cur, res, data_args)
                grad = jnp.where(constants_mask, 0.0, grad)
                return val, grad

            def _masked_kl_metric(m_cur, res, v):
                v_masked = jnp.where(constants_mask, 0.0, v)
                mv = kl_metric(m_cur, res, v_masked, data_args)
                return jnp.where(constants_mask, 0.0, mv)

            def _fun_and_grad(m_cur):
                return _masked_kl_vg(m_cur, residuals)

            def _hessp(m_cur, v):
                return _masked_kl_metric(m_cur, residuals, v)

            return _newton_cg_flat(
                _fun_and_grad,
                _hessp,
                m,
                maxiter=10,
                miniter=0,
                xtol=1e-5,
                energy_reduction_factor=0.1,
            )

        _RESAMPLE_EVERY = 5  # refresh stale samples every N iterations

        def evi_step_full(
            m,
            subkey,
            n_samples,
            sample_mode,
            prev_residuals,
            prev_keys,
            constants_mask,
            pe_mask,
            data_args,
            iteration=0,
        ):
            """One geoVI iteration — ``sample_mode`` must be a static string.

            When used inside ``run_evi_geovi`` (which marks ``sample_mode``
            as static), JAX compiles a separate version per mode.  The
            unused branches are never traced, so ``"linear"`` compiles in
            ~0.03s while ``"nonlinear_resample"`` compiles in ~56s.

            Parameters
            ----------
            sample_mode : str  (STATIC — triggers recompilation per value)
                ``"linear_resample"`` — fresh MGVI samples (standard MGVI)
                ``"linear_sample"`` — reuse keys from prev iter (deterministic MGVI)
                ``"nonlinear_resample"`` — fresh geoVI samples (standard geoVI)
                ``"nonlinear_sample"`` — reuse keys + curve (deterministic geoVI)
                ``"nonlinear_update"`` — re-curve existing residuals at new m
            data_args : dict
                Data-dependent arguments (data, noise, noise_inv, etc.).

            Returns
            -------
            m_new, kl_value, new_residuals, used_keys
            """
            # Key handling: _resample = fresh keys, _sample = reuse prev keys
            if sample_mode.endswith("_resample") or sample_mode == "geovi":
                sample_keys = jax.random.split(subkey, n_samples)
            elif sample_mode == "nonlinear_update":
                sample_keys = prev_keys
            else:  # _sample modes: reuse
                sample_keys = prev_keys

            # Python if — only the used branch is traced by JAX
            if sample_mode == "geovi":
                # Optimal schedule: resample at iter 0 and every
                # _RESAMPLE_EVERY, nonlinear_update in between.
                # Uses jax.lax.cond (traces both branches, executes one).
                do_resample = (iteration == 0) | (iteration % _RESAMPLE_EVERY == 0)

                def _do_resample(_):
                    return draw_nonlinear_residuals(m, sample_keys, data_args)

                def _do_update(_):
                    return update_nonlinear_residuals(m, prev_residuals, prev_keys, data_args)

                residuals = jax.lax.cond(do_resample, _do_resample, _do_update, None)
            elif sample_mode in ("nonlinear_resample", "nonlinear_sample"):
                residuals = draw_nonlinear_residuals(m, sample_keys, data_args)
            elif sample_mode == "nonlinear_update":
                residuals = update_nonlinear_residuals(m, prev_residuals, sample_keys, data_args)
            else:  # linear_resample, linear_sample
                res = draw_residuals(m, sample_keys, data_args)
                residuals = jnp.concatenate([res, -res], axis=0)

            # Apply point estimates mask
            residuals = residuals * pe_mask[None, :]

            # KL minimization
            m_opt, kl_val = _kl_minimize(m, residuals, constants_mask, data_args)
            return m_opt, kl_val, residuals, sample_keys

        def run_evi_geovi(init_pos, key, data_args, n_iterations, n_samples, kl_rtol, sample_mode):
            """Run geoVI with automatic convergence detection.

            ``n_iterations`` is a **dynamic** traced value — changing it
            does NOT trigger recompilation.  Keys are generated on-the-fly
            via ``jax.random.fold_in`` instead of pre-splitting.

            ``sample_mode`` is a **static** string — JAX compiles a
            separate XLA program per mode.  All 5 NIFTy modes supported:

            - ``"linear_resample"`` — fresh MGVI samples each iteration
            - ``"linear_sample"`` — reuse PRNG keys (deterministic MGVI)
            - ``"nonlinear_resample"`` — fresh geoVI samples
            - ``"nonlinear_sample"`` — reuse keys + curve (deterministic geoVI)
            - ``"nonlinear_update"`` — re-curve existing residuals at new m
            """
            # Generate per-iteration keys on-the-fly via fold_in (no
            # pre-split needed, so n_iterations can be dynamic).
            dummy_residuals = jnp.zeros((2 * n_samples, d_total))
            dummy_keys = jax.random.split(jax.random.fold_in(key, 0), n_samples)
            no_constants = jnp.zeros(d_total, dtype=bool)
            all_sampled = jnp.ones(d_total)

            # State: (m, prev_kl, residuals, prev_keys, iter, converged)
            def cond_fn(state):
                _m, _prev_kl, _res, _pk, i, converged = state
                return (~converged) & (i < n_iterations)

            def body_fn(state):
                m, prev_kl, prev_res, prev_k, i, converged = state
                subkey = jax.random.fold_in(key, i)
                m_new, kl_val, new_res, new_k = evi_step_full(
                    m,
                    subkey,
                    n_samples,
                    sample_mode,
                    prev_res,
                    prev_k,
                    no_constants,
                    all_sampled,
                    data_args,
                    iteration=i,
                )
                rel_change = jnp.abs(prev_kl - kl_val) / (jnp.abs(prev_kl) + 1e-10)
                converged = (rel_change < kl_rtol) & (i >= 5)
                return (m_new, kl_val, new_res, new_k, i + 1, converged)

            # First iteration (always resample to establish initial keys)
            first_key = jax.random.fold_in(key, 0)
            m0, kl0, res0, keys0 = evi_step_full(
                init_pos,
                first_key,
                n_samples,
                sample_mode,
                dummy_residuals,
                dummy_keys,
                no_constants,
                all_sampled,
                data_args,
            )
            init_state = (
                m0,
                kl0,
                res0,
                keys0,
                jnp.int32(1),
                jnp.bool_(False),
            )

            result = jax.lax.while_loop(cond_fn, body_fn, init_state)
            return result[0], result[4]  # m_final, n_iters

        # --- Parameter range mapping for mask construction ---
        param_ranges = {}
        for i_k, k in enumerate(param_keys):
            start, end, _shape = slices[i_k]
            param_ranges[k] = (start, end)

        def make_mask(param_names):
            """Create boolean mask: True for named params, False otherwise."""
            mask = jnp.zeros(d_total, dtype=bool)
            for name in param_names:
                if name in param_ranges:
                    start, end = param_ranges[name]
                    mask = mask.at[start:end].set(True)
            return mask

        def make_pe_mask(param_names):
            """Create point-estimate mask: 0.0 for PE params, 1.0 for sampled."""
            mask = jnp.ones(d_total)
            for name in param_names:
                if name in param_ranges:
                    start, end = param_ranges[name]
                    mask = mask.at[start:end].set(0.0)
            return mask

        # --- NIFTy-backed geoVI: exact NIFTy math, minimal Python overhead ---
        # Uses NIFTy's OptimizeVI.update directly (already JIT'd internally)
        # but skips logging, pickling, and callbacks for speed.
        nifty_likelihood = None
        nifty_opt_vi = None
        if _has_nifty:
            try:
                import nifty8.re as jft

                # Build the NIFTy likelihood (same as _run_nifty_vi)
                _nifty_domain = {}
                for name in self._free_names:
                    _nifty_domain[name] = jft.ShapeWithDtype(())
                if self.spec.stochastic:
                    _nifty_domain["psd_xi"] = jft.ShapeWithDtype((self.spec.n_grid,))
                _nifty_model = jft.Model(jax.jit(signal_response), domain=_nifty_domain)
                if not use_variable_noise:
                    nifty_likelihood = jft.Gaussian(self.data, self._data_args["noise_inv"]).amend(
                        _nifty_model
                    )
                # Build OptimizeVI with vmap and JIT
                # (this pre-compiles all the internal functions)
                nifty_opt_vi = OptimizeVI(
                    nifty_likelihood,
                    n_total_iterations=50,  # max, actual controlled by caller
                    kl_jit=True,
                    residual_jit=True,
                    kl_map=jax.vmap,
                    residual_map=jax.vmap,
                )
            except Exception:
                _has_nifty = False
                nifty_likelihood = None
                nifty_opt_vi = None

        def run_nifty_jit(
            init_pos_flat,
            key,
            n_iterations,
            n_samples,
            sample_mode_str,
            draw_linear_kwargs,
            nonlinearly_update_kwargs,
            kl_kwargs,
        ):
            """Run NIFTy's exact optimize_kl with minimal Python overhead.

            Uses NIFTy's OptimizeVI.update (already JIT'd) in a tight
            Python loop — no logging, no pickling, no callbacks.
            Exact same math as ``jft.optimize_kl``.

            Returns (converged_flat, n_iters).
            """
            import nifty8.re as jft

            pos_dict = unflatten(init_pos_flat)
            samples = NiftySamples(pos=jft.Vector(pos_dict), samples=None, keys=None)
            state = nifty_opt_vi.init_state(
                key,
                n_samples=n_samples,
                sample_mode=sample_mode_str,
                draw_linear_kwargs=draw_linear_kwargs,
                nonlinearly_update_kwargs=nonlinearly_update_kwargs,
                kl_kwargs=kl_kwargs,
            )
            for _i in range(n_iterations):
                samples, state = nifty_opt_vi.update(samples, state)
            converged = samples.pos
            pos_d = converged.tree if hasattr(converged, "tree") else dict(converged)
            return flatten(pos_d), samples

        # Compile the core functions with dummy data.
        # data_args is passed as argument (not closed over) to all JIT'd
        # functions so the same compiled XLA program can be reused with
        # different data of the same shape.
        dummy_pos = flatten(pos_dict)
        dummy_keys = jax.random.split(jax.random.PRNGKey(0), 2)
        dummy_data_args = self._data_args

        # Pre-compile posterior sampler
        draw_samples_jit = jax.jit(draw_residuals)
        _ = draw_samples_jit(dummy_pos, dummy_keys, dummy_data_args)

        # Pre-compile native optimizer (for n_iterations=10, n_samples=3)
        # n_iterations is DYNAMIC (while_loop handles it). n_samples is
        # STATIC (determines array shapes in draw_residuals).
        run_evi_jit = jax.jit(run_evi, static_argnames=("n_samples",))
        _ = run_evi_jit(
            dummy_pos,
            jax.random.PRNGKey(0),
            dummy_data_args,
            n_iterations=2,
            n_samples=2,
            kl_rtol=1e-2,
        )

        # Pre-compile native geoVI optimizer.
        # sample_mode is STATIC: JAX compiles a separate XLA program per mode.
        # "linear" compiles in ~0.03s, "nonlinear_*" in ~56s (one-time cost).
        # n_iterations is DYNAMIC: the while_loop handles variable counts
        # without recompilation. n_samples is STATIC because it determines
        # array shapes (residuals, sample keys) that JAX needs at trace time.
        run_evi_geovi_jit = jax.jit(
            run_evi_geovi,
            static_argnames=("n_samples", "sample_mode"),
        )

        return {
            "run_evi": run_evi_jit,
            "run_evi_geovi": run_evi_geovi_jit,
            "run_nifty_jit": run_nifty_jit if _has_nifty else None,
            "nifty_likelihood": nifty_likelihood,
            "draw_samples": draw_samples_jit,
            "draw_nonlinear_samples": jax.jit(draw_nonlinear_residuals),
            "draw_batch": draw_batch,
            "flatten": flatten,
            "unflatten": unflatten,
            "param_keys": param_keys,
            "param_ranges": param_ranges,
            "make_mask": make_mask,
            "make_pe_mask": make_pe_mask,
            "d_total": d_total,
            "SAMPLE_LINEAR": SAMPLE_LINEAR,
            "SAMPLE_NONLINEAR_RESAMPLE": SAMPLE_NONLINEAR_RESAMPLE,
            "SAMPLE_NONLINEAR_UPDATE": SAMPLE_NONLINEAR_UPDATE,
            "evi_step_full": evi_step_full,
            # geoVI-NUTS primitives (coordinate transform + metric)
            "transformation_flat": transformation_flat,
            "left_sqrt_metric_flat": left_sqrt_metric_flat,
            "right_sqrt_metric_flat": right_sqrt_metric_flat,
            "metric_vec": metric_vec,
            "cg_solve": cg_solve,
            "hamiltonian": hamiltonian,
        }

    def compile(
        self,
        *,
        n_iterations=15,
        n_samples=3,
        n_posterior_samples=200,
        modes=("linear_resample", "nonlinear_update"),
        verbose=True,
    ):
        """Pre-compile the JIT inference engine ahead of time.

        Triggers XLA compilation for all specified modes so that
        subsequent ``fitter.run()`` calls have zero compilation delay.
        Compiled programs are cached both in-memory (this session)
        and on disk (``/tmp/tengri_jax_cache``, survives restarts).

        Parameters
        ----------
        n_iterations : int
            Iteration count for the pre-compilation run.  Changing
            ``n_iterations`` at run time does NOT trigger recompilation
            (the iteration count is a dynamic traced value).
        n_samples : int
            Compile for this sample count.  Changing ``n_samples``
            at run time DOES trigger recompilation (array shapes
            depend on it).
        n_posterior_samples : int
            Compile posterior draw for this many samples.
        modes : tuple of str
            Which sample modes to pre-compile. Each mode compiles
            separately. Default covers MGVI + geoVI update (fastest).
            Add ``"nonlinear_resample"`` for full geoVI (~56s extra).
        verbose : bool
            Print compilation progress.

        Example
        -------
        >>> fitter = Fitter(model, data, noise)
        >>> fitter.compile()  # ~3s for default modes
        >>> fitter.compile(
        ...     modes=(  # ~60s for all modes
        ...         "linear_resample",
        ...         "nonlinear_update",
        ...         "nonlinear_resample",
        ...     )
        ... )
        >>> result = fitter.run("native_geovi")  # instant
        """
        dummy_pos = self._initialize_unbounded(jax.random.PRNGKey(0))
        if self._jit_sampler is None:
            self._jit_sampler = self._get_or_build_engine(dummy_pos)

        engine = self._jit_sampler
        flatten = engine["flatten"]
        pos_flat = flatten(dummy_pos)
        data_args = self._data_args

        if verbose:
            print(
                f"Compiling: n_iter={n_iterations}, n_samp={n_samples}, "
                f"n_post={n_posterior_samples}, modes={modes}"
            )

        # Pre-compile each optimization mode
        for mode in modes:
            if verbose:
                print(f"  Compiling {mode}...", end="", flush=True)
            t0 = time.time()
            engine["run_evi_geovi"](
                pos_flat,
                jax.random.PRNGKey(0),
                data_args,
                n_iterations=n_iterations,
                n_samples=n_samples,
                kl_rtol=0.0,
                sample_mode=mode,
            )
            if verbose:
                print(f" {time.time() - t0:.1f}s")

        # Pre-compile MGVI optimizer (old path, used by native_mgvi)
        if verbose:
            print("  Compiling MGVI (old path)...", end="", flush=True)
        t0 = time.time()
        engine["run_evi"](
            pos_flat,
            jax.random.PRNGKey(0),
            data_args,
            n_iterations=n_iterations,
            n_samples=n_samples,
            kl_rtol=1e-2,
        )
        if verbose:
            print(f" {time.time() - t0:.1f}s")

        # Pre-compile posterior draw
        if verbose:
            print(
                f"  Compiling posterior draw ({n_posterior_samples} samples)...",
                end="",
                flush=True,
            )
        t0 = time.time()
        draw_keys = jax.random.split(jax.random.PRNGKey(0), n_posterior_samples)
        engine["draw_samples"](pos_flat, draw_keys, data_args)
        if verbose:
            print(f" {time.time() - t0:.1f}s")

        if verbose:
            print("Compilation complete.")
        return self

    def _draw_jit_samples(self, pos_dict, key, n_samples, existing_samples, *, verbose=True):
        """Draw geoVI linear residual samples via JIT-compiled CG.

        Same math as NIFTy's draw_linear_residual but fully JIT-compiled:
        1. Draw z = J^T sqrt(N^{-1}) eta1 + eta2  (eta_i ~ N(0,I))
        2. Solve M @ residual = z via CG  (M = J^T N^{-1} J + I)
        3. Sample = pos + residual

        ~2000x faster than NIFTy's Python-loop CG.
        """
        if verbose:
            print(f"  Drawing {n_samples} posterior samples (JIT CG)...")

        if self._jit_sampler is None:
            self._jit_sampler = self._get_or_build_engine(pos_dict)

        engine = self._jit_sampler
        flatten, unflatten = engine["flatten"], engine["unflatten"]
        pos_flat = flatten(pos_dict)
        draw_keys = jax.random.split(key, n_samples)
        residuals_flat = engine["draw_samples"](pos_flat, draw_keys, self._data_args)

        for i in range(n_samples):
            res = unflatten(residuals_flat[i])
            combined = {k: pos_dict[k] + res[k] for k in pos_dict}
            existing_samples.append(combined)

        return existing_samples

    def _draw_nonlinear_jit_samples(
        self, pos_dict, key, n_samples, existing_samples, *, verbose=True
    ):
        """Draw geoVI nonlinear posterior samples via JIT engine.

        Unlike ``_draw_jit_samples`` (linear CG only), this applies
        the geoVI coordinate curving to each sample.  Produces
        samples from the nonlinear approximation, capturing
        banana-shaped degeneracies that the linear Gaussian misses.

        Uses ``draw_nonlinear_residuals`` from the JIT engine.
        """
        if verbose:
            print(f"  Drawing {n_samples} nonlinear posterior samples (JIT geoVI)...")

        if self._jit_sampler is None:
            self._jit_sampler = self._get_or_build_engine(pos_dict)

        engine = self._jit_sampler
        flatten, unflatten = engine["flatten"], engine["unflatten"]
        pos_flat = flatten(pos_dict)
        data_args = self._data_args

        # Draw in batches to avoid OOM for large n_samples
        batch_size = min(n_samples, 50)
        draw_keys = jax.random.split(key, n_samples)

        for batch_start in range(0, n_samples, batch_size):
            batch_end = min(batch_start + batch_size, n_samples)
            batch_keys = draw_keys[batch_start:batch_end]
            # draw_nonlinear_samples returns (2*n, D): first n positive, last n mirrors
            residuals_flat = engine["draw_nonlinear_samples"](pos_flat, batch_keys, data_args)
            n_batch = batch_end - batch_start
            # Use only the first n (positive) samples, not the mirrors
            for i in range(n_batch):
                res = unflatten(residuals_flat[i])
                combined = {k: pos_dict[k] + res[k] for k in pos_dict}
                existing_samples.append(combined)

        return existing_samples

    def _draw_blackjax_samples(
        self, likelihood, pos_dict, key, n_samples, existing_samples, *, verbose=True
    ):
        """Draw samples via BlackJAX NUTS (independent MCMC, not geoVI)."""
        import blackjax

        if verbose:
            print(f"  Drawing {n_samples} posterior samples via BlackJAX NUTS...")

        if likelihood is not None:

            @jax.jit
            def logdensity_fn(x):
                lh_val = likelihood(x)
                prior = 0.5 * sum(jnp.sum(v**2) for v in x.values())
                return -lh_val - prior

        else:
            # Build log-density from the loss function (used by _run_native_vi path)
            loss_fn = self._get_or_build_loss_fn()
            _data_args = self._data_args

            @jax.jit
            def logdensity_fn(x):
                return -loss_fn(x, _data_args)

        warmup_key, sample_key = jax.random.split(key)
        n_warmup = min(200, n_samples)
        warmup = blackjax.window_adaptation(blackjax.nuts, logdensity_fn)
        (state, parameters), _ = warmup.run(warmup_key, pos_dict, num_steps=n_warmup)

        if verbose:
            print(f"  Warmup done ({n_warmup} steps). Sampling...")

        kernel = blackjax.nuts(logdensity_fn, **parameters).step

        @jax.jit
        def one_step(state, rng_key):
            state, _ = kernel(rng_key, state)
            return state, state

        keys = jax.random.split(sample_key, n_samples)
        _, states = jax.lax.scan(one_step, state, keys)

        sample_positions = states.position
        for i in range(n_samples):
            sd = jax.tree.map(lambda x, _i=i: x[_i], sample_positions)
            existing_samples.append(sd)

        return existing_samples

    def _draw_nifty_samples(
        self, likelihood, pos_dict, key, n_samples, existing_samples, *, verbose=True
    ):
        """Draw samples via NIFTy's draw_linear_residual (slow, ~540ms/sample)."""
        import nifty8.re as jft

        if verbose:
            print(f"  Drawing {n_samples} posterior samples (NIFTy CG)...")

        converged_pos = jft.Vector(pos_dict)
        draw_keys = jax.random.split(key, n_samples)
        for sub_key in draw_keys:
            try:
                residual, _ = jft.draw_linear_residual(
                    likelihood,
                    converged_pos,
                    sub_key,
                    cg_kwargs={"absdelta": 1e-4, "maxiter": 30},
                )
                sample_tree = residual.tree if hasattr(residual, "tree") else dict(residual)
                pos_tree = (
                    converged_pos.tree if hasattr(converged_pos, "tree") else dict(converged_pos)
                )
                combined = {k: pos_tree[k] + sample_tree[k] for k in pos_tree}
                existing_samples.append(combined)
            except Exception:
                break

        return existing_samples

    # -------------------------------------------------------------------
    # Fully JIT'd EVI optimizer
    # -------------------------------------------------------------------

    def _run_native_vi(self, *, key, **kwargs):
        from tengri.inference.vi import run_native_vi

        return run_native_vi(self, key=key, **kwargs)

    def run(self, method: str = "vi", *, init_from=None, key=None, **kwargs):
        """Run inference.

        Parameters
        ----------
        method : str
            Canonical names (recommended):

            ``"vi"``             — Variational inference (geoVI by default).
            ``"vi_linear"``      — Linear VI / MGVI.
            ``"vi_nifty"``       — NIFTy tight-loop geoVI.
            ``"vi_nifty_linear"``— NIFTy tight-loop MGVI.
            ``"mcmc"``           — MCMC, auto-selects NUTS (D≤20) or Ray Tracing (D>20).
            ``"mcmc_raytrace"``  — Ray Tracing explicitly.
            ``"mcmc_nuts"``      — NUTS via BlackJAX.
            ``"mcmc_ess"``       — Elliptical Slice Sampling.
            ``"map"``            — MAP optimization.
            ``"laplace"``        — Gaussian at MAP.
            ``"pathfinder"``     — L-BFGS path.
            ``"evidence"``       — Nested Slice Sampling (log Z).
            ``"auto"``           — Auto-selects by dimensionality.

            Power-user ``vi_flavor=`` kwarg (only with ``method="vi"``):
            ``vi_flavor="nifty"``      — NIFTy tight loop.
            ``vi_flavor="nifty_full"`` — NIFTy with full logging.
            ``vi_flavor="linear"``     — Linearized geoVI (MGVI).

            Deprecated aliases (still work, emit DeprecationWarning):
            ``"geovi"``, ``"native_geovi"`` → ``"vi"``
            ``"mgvi"``, ``"native_mgvi"``   → ``"vi_linear"``
            ``"fast_geovi"``, ``"nifty_geovi"`` → ``"vi_nifty"``
            ``"fast_mgvi"``, ``"nifty_mgvi"``   → ``"vi_nifty_linear"``
            ``"raytrace"`` → ``"mcmc_raytrace"``
            ``"nuts"``     → ``"mcmc_nuts"``
            ``"elliptical_slice"`` → ``"mcmc_ess"``
            ``"nss"``      → ``"evidence"``

        init_from : Posterior, optional
            Use a previous result as initialization.
        key : PRNGKey, optional
            Random key.
        vi_flavor : str, optional
            Backend variant for ``method="vi"`` only.
            ``"nifty"``, ``"nifty_full"``, or ``"linear"``.
        **kwargs
            Method-specific arguments passed to the underlying sampler.

        Returns
        -------
        Posterior
            Inference results with ``._fitter`` back-reference set.
        """
        if key is None:
            key = jax.random.PRNGKey(42)

        # Pop vi_flavor before forwarding kwargs
        vi_flavor = kwargs.pop("vi_flavor", None)

        # Resolve deprecated aliases
        if method in _DEPRECATED_METHOD_ALIASES:
            canonical = _DEPRECATED_METHOD_ALIASES[method]
            warnings.warn(
                f"Method '{method}' is deprecated. Use '{canonical}' instead. "
                f"Old names will be removed in tengri v1.0.",
                DeprecationWarning,
                stacklevel=2,
            )
            # Special case: geovi_nuts was a hybrid; map to vi with blackjax posterior
            if method == "geovi_nuts":
                kwargs.setdefault("posterior_method", "blackjax")
            method = canonical

        # --- "auto" method: dimensionality-based selection ---
        if method == "auto":
            d = self.spec.n_free
            lo, hi = _AUTO_D_THRESHOLDS
            if d <= lo:
                method = "laplace"
            elif d <= hi:
                method = "vi_linear"
            else:
                method = "vi"

        # --- Dispatch to underlying _run_* methods ---
        if method == "map":
            result = self._run_map(key=key, init_from=init_from, **kwargs)

        elif method in ("vi", "vi_linear"):
            # vi_flavor overrides method for power users
            if vi_flavor == "nifty":
                result = self._run_fast_vi(
                    key=key,
                    init_from=init_from,
                    sample_mode="nonlinear_resample",
                    posterior_method="nonlinear",
                    **kwargs,
                )
            elif vi_flavor == "nifty_full":
                result = self._run_nifty_vi(key=key, init_from=init_from, **kwargs)
            elif vi_flavor == "linear" or method == "vi_linear":
                result = self._run_native_vi(
                    key=key,
                    init_from=init_from,
                    sample_mode="linear",
                    **kwargs,
                )
            else:
                # Default "vi": geoVI (nonlinear, most accurate)
                result = self._run_native_vi(
                    key=key,
                    init_from=init_from,
                    sample_mode="geovi",
                    **kwargs,
                )

        elif method == "vi_nifty":
            result = self._run_fast_vi(
                key=key,
                init_from=init_from,
                sample_mode="nonlinear_resample",
                posterior_method="nonlinear",
                **kwargs,
            )

        elif method == "vi_nifty_linear":
            result = self._run_fast_vi(
                key=key,
                init_from=init_from,
                sample_mode="linear_resample",
                **kwargs,
            )

        elif method == "mcmc":
            # Auto-select: NUTS for low-D (exact gold-standard), RT for high-D
            d = self.spec.n_free
            if d <= _MCMC_AUTO_D_THRESHOLD:
                result = self._run_nuts(key=key, init_from=init_from, **kwargs)
            else:
                result = self._run_raytrace(key=key, init_from=init_from, **kwargs)

        elif method == "mcmc_raytrace":
            result = self._run_raytrace(key=key, init_from=init_from, **kwargs)

        elif method == "mcmc_nuts":
            result = self._run_nuts(key=key, init_from=init_from, **kwargs)

        elif method == "mcmc_ess":
            result = self._run_elliptical_slice(key=key, init_from=init_from, **kwargs)

        elif method == "evidence":
            result = self._run_nss(key=key, init_from=init_from, **kwargs)

        elif method == "laplace":
            result = self._run_laplace(key=key, init_from=init_from, **kwargs)

        elif method == "pathfinder":
            result = self._run_pathfinder(key=key, init_from=init_from, **kwargs)

        else:
            raise ValueError(
                f"Unknown method: '{method}'. "
                f"Canonical names: 'vi', 'vi_linear', 'vi_nifty', 'vi_nifty_linear', "
                f"'mcmc', 'mcmc_raytrace', 'mcmc_nuts', 'mcmc_ess', 'map', 'laplace', "
                f"'pathfinder', 'evidence', 'auto'. "
                f"See Fitter.run() docstring for deprecated aliases."
            )

        # Attach back-reference so Posterior.refine() works
        result._fitter = self
        return result

    def _run_nss(self, *, key, **kwargs):
        from tengri.inference.evidence import run_nss

        return run_nss(self, key=key, **kwargs)

    def _run_fast_vi(self, *, key, **kwargs):
        from tengri.inference.vi import run_fast_vi

        return run_fast_vi(self, key=key, **kwargs)

    def fit_batch(
        self,
        batch,
        *,
        method="native_geovi",
        key=None,
        verbose=True,
        **kwargs,
    ):
        """Fit a batch of galaxies efficiently.

        Creates a Fitter per galaxy, sharing the XLA compilation cache.
        The first galaxy pays compile cost; subsequent galaxies load
        from the persistent XLA cache (milliseconds each).

        Works with any inference method — native_geovi (default) gives
        the best speed. Also usable for hierarchical individual fits.

        Parameters
        ----------
        batch : list of dict
            Each dict has "flux_obs" and "noise" arrays.
        method : str
            Default "native_geovi". Any method from run().
        key : PRNGKey, optional
        verbose : bool
        **kwargs
            Passed to run() (n_iterations, n_samples, n_seeds, etc).

        Returns
        -------
        list of Posterior

        Example
        -------
        >>> galaxies = [{"flux_obs": f, "noise": n} for f, n in zip(fluxes, noises)]
        >>> results = fitter.fit_batch(galaxies)
        >>> # First: ~15s compile. Rest: ~2ms each (native_geovi).
        """
        if key is None:
            key = jax.random.PRNGKey(42)

        if "native" in method and "n_seeds" not in kwargs:
            kwargs["n_seeds"] = 5

        n_gal = len(batch)
        if verbose:
            print(f"fit_batch: {n_gal} galaxies, method={method}")

        results = []
        t0 = time.time()

        for i, gal in enumerate(batch):
            gal_key = jax.random.fold_in(key, i)
            t_gal = time.time()

            fitter_i = Fitter(
                self.model,
                gal["flux_obs"],
                gal["noise"],
                data_type=self.data_type,
            )
            result_i = fitter_i.run(method, key=gal_key, verbose=False, **kwargs)
            results.append(result_i)

            dt = time.time() - t_gal
            if verbose and (i < 3 or (i + 1) % max(1, n_gal // 10) == 0 or i == n_gal - 1):
                chi2 = result_i.diagnostics.get("chi2_dof", "?")
                chi2_str = f"{chi2:.2f}" if isinstance(chi2, float) else str(chi2)
                print(f"  Galaxy {i + 1}/{n_gal}: chi2/dof={chi2_str}, {dt:.1f}s")

        t_total = time.time() - t0
        if verbose:
            print(f"  Done: {n_gal} galaxies in {t_total:.1f}s ({t_total / n_gal:.1f}s/galaxy)")

        return results

    def _run_map(self, *, key, **kwargs):
        from tengri.inference.map_dispatch import run_map

        return run_map(self, key=key, **kwargs)

    def _run_raytrace(self, *, key, **kwargs):
        from tengri.inference.mcmc import run_raytrace

        return run_raytrace(self, key=key, **kwargs)

    def _run_nuts(self, *, key, **kwargs):
        from tengri.inference.mcmc import run_nuts

        return run_nuts(self, key=key, **kwargs)

    def _build_loglikelihood_unbounded_fn(self):
        """Build a log-likelihood function in unbounded parameter space.

        For Elliptical Slice Sampling, which handles the N(0,I) prior
        internally.  Returns ``loglik(params_unbounded, data_args)``.
        """
        loglik_fn = self._get_or_build_loglikelihood_fn()
        bounds = self._bounds
        free_names = self._free_names
        fixed_values = self._fixed_values
        spec = self.spec

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

    def _run_laplace(self, *, key, **kwargs):
        from tengri.inference.map_dispatch import run_laplace

        return run_laplace(self, key=key, **kwargs)

    def _run_pathfinder(self, *, key, **kwargs):
        from tengri.inference.map_dispatch import run_pathfinder

        return run_pathfinder(self, key=key, **kwargs)

    def _run_elliptical_slice(self, *, key, **kwargs):
        from tengri.inference.mcmc import run_elliptical_slice

        return run_elliptical_slice(self, key=key, **kwargs)

    def _run_nifty_vi(self, *, key, **kwargs):
        from tengri.inference.vi import run_nifty_vi

        return run_nifty_vi(self, key=key, **kwargs)
