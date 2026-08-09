# SPDX-License-Identifier: BSD-3-Clause
"""High-level Galaxy facade for one-liner SED fitting.

The Galaxy class bundles SSP data, observation, parameters, and model
into a single object so users don't have to construct each by hand.
"""

from __future__ import annotations

import os
import platform
import sys
import warnings
from typing import Any

import jax
import numpy as np

from tengri._display import _display
from tengri._logo import logo_str
from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
from tengri.config.settings import SEDModelConfig
from tengri.forward.sed_model import SEDModel
from tengri.inference.fitter import Fitter
from tengri.observation.noise_model import NoiseModel
from tengri.observation.observation import Observation
from tengri.observation.photometry_config import Photometry

# To erg/s/cm2/Hz (canonical): 1 Jy = 1e-23; 1 maggie = 3631 Jy (AB zero point).
_FLUX_UNIT_TO_CGS = {
    "erg/s/cm2/Hz": 1.0,
    "erg/s/cm^2/Hz": 1.0,  # alt notation
    "uJy": 1e-29,
    "mJy": 1e-26,
    "Jy": 1e-23,
    "nJy": 1e-32,
    "maggies": 3.631e-20,
}


class Galaxy:
    """User-facing facade around SEDModel + Parameters + Observation + Fitter.

    Bundles SSP data, observation, parameters, and model into a single object
    for simplified one-liner SED fitting. Not a dataclass — it has mutable state
    (result after fit) and methods. Construction is via classmethods, not
    __init__ directly.

    Attributes
    ----------
    ssp : SSPData
        Stellar population synthesis data (grid of stellar templates).
    observation : Observation
        Observation configuration (photometry, spectroscopy, noise).
    parameters : Parameters
        Parameter specification with priors.
    model_config : SEDModelConfig
        Model configuration (dust, nebular, AGN, etc.).
    model : SEDModel or None
        Forward model (lazily constructed on first forward pass or before fit).
    result : Posterior or None
        Posterior/result after running fit().
    preset_name : str or None
        Name of the preset used (e.g. "starforming"), if any.
    """

    def __init__(
        self,
        *,
        ssp,
        observation,
        parameters,
        model_config,
        model=None,
        preset_name=None,
    ):
        """Initialize Galaxy (not intended for direct use; use classmethods).

        Parameters
        ----------
        ssp : SSPData
            Stellar population synthesis data.
        observation : Observation
            Observation configuration.
        parameters : Parameters
            Parameter specification with priors.
        model_config : ModelConfig
            Model configuration.
        model : SEDModel or None
            Forward model (if pre-built).
        preset_name : str or None
            Name of preset used.
        """
        self.ssp = ssp
        self.observation = observation
        self.parameters = parameters
        self.model_config = model_config
        self.model = model
        self.result = None
        self.preset_name = preset_name

        # Live citation accumulator. Populated from the configured components
        # at construction and extended with the inference-backend citation(s)
        # on every call to .fit(). See tengri.citations.Bibliography.
        from tengri.citations.bibliography import Bibliography

        src = f"Galaxy(preset={preset_name})" if preset_name else "Galaxy"
        self.bibliography = Bibliography.from_config(model_config, source=src)

    @classmethod
    def from_arrays(
        cls,
        *,
        filters: list[str],
        flux: list[float] | np.ndarray,
        flux_err: list[float] | np.ndarray,
        redshift: float | None = None,
        ssp_path: str | None = None,
        ssp: Any | None = None,
        preset: str = "starforming",
        flux_unit: str = "erg/s/cm2/Hz",
        model_config: SEDModelConfig | None = None,
    ) -> Galaxy:
        """Build a Galaxy from plain arrays — the common entry point.

        Parameters
        ----------
        filters : list of str
            Filter names (e.g., ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]).
        flux : array_like, shape (n_filters,)
            Observed fluxes [erg/s/cm²/Hz] by default.
        flux_err : array_like, shape (n_filters,)
            Flux uncertainties (same units as flux).
        redshift : float or None
            Redshift. If None, must be specified in preset.
        ssp_path : str or None
            Path to SSP HDF5 file. If None, ssp must be provided.
        ssp : SSPData or None
            Pre-loaded SSP data. If None, ssp_path must be provided.
        preset : str
            Preset name: "starforming", "quiescent", "high_z". Default: "starforming".
        flux_unit : str
            Flux unit. One of "erg/s/cm2/Hz", "uJy", "mJy", "Jy", "nJy", "maggies".
            Default: "erg/s/cm2/Hz".
        model_config : SEDModelConfig or None
            Model configuration. If None, uses preset defaults.

        Returns
        -------
        Galaxy

        Raises
        ------
        ValueError
            If neither ssp_path nor ssp provided, or if flux_unit unknown.
        """
        # Load SSP data
        if ssp is None:
            if ssp_path is None:
                raise ValueError(
                    "Either ssp_path or ssp must be provided. Example: ssp_path='data/ssp_mist.h5'"
                )
            ssp = load_ssp_data(ssp_path)
        if ssp_path is None:
            ssp_path = "(provided object)"

        # Convert fluxes to canonical unit
        flux = np.asarray(flux)
        flux_err = np.asarray(flux_err)

        if flux_unit not in _FLUX_UNIT_TO_CGS:
            raise ValueError(
                f"Unknown flux_unit '{flux_unit}'. Supported: {list(_FLUX_UNIT_TO_CGS.keys())}"
            )
        scale = _FLUX_UNIT_TO_CGS[flux_unit]
        flux_cgs = flux * scale
        flux_err_cgs = flux_err * scale

        # Build Photometry
        photometry = Photometry.from_names(filters)

        # Build Observation (assume Gaussian noise, no spectroscopy for from_arrays)
        noise_model = NoiseModel()
        observation = Observation(
            photometry=photometry,
            spectroscopy=None,
            noise=noise_model,
        )

        # Get preset parameters and config
        from tengri.presets import resolve_preset

        parameters, config = resolve_preset(preset, redshift=redshift, model_config=model_config)

        # Create instance
        galaxy = cls(
            ssp=ssp,
            observation=observation,
            parameters=parameters,
            model_config=config,
            preset_name=preset,
        )
        # Store original flux arrays for later use (e.g., in fit)
        galaxy._flux_obs = flux_cgs
        galaxy._noise = flux_err_cgs

        return galaxy

    @classmethod
    def from_observation(
        cls,
        observation: Observation,
        *,
        ssp: Any | None = None,
        ssp_path: str | None = None,
        preset: str = "starforming",
        redshift: float | None = None,
        model_config: SEDModelConfig | None = None,
    ) -> Galaxy:
        """Build a Galaxy from an existing Observation object.

        Parameters
        ----------
        observation : Observation
            Pre-built Observation (photometry, spectroscopy, noise).
        ssp : SSPData or None
            Pre-loaded SSP data.
        ssp_path : str or None
            Path to SSP HDF5 file.
        preset : str
            Preset name.
        redshift : float or None
            Redshift (overrides any fixed redshift in preset).
        model_config : SEDModelConfig or None
            Model configuration.

        Returns
        -------
        Galaxy

        Raises
        ------
        ValueError
            If neither ssp_path nor ssp provided.
        """
        if ssp is None:
            if ssp_path is None:
                raise ValueError("Either ssp_path or ssp must be provided.")
            ssp = load_ssp_data(ssp_path)

        from tengri.presets import resolve_preset

        parameters, config = resolve_preset(preset, redshift=redshift, model_config=model_config)

        return cls(
            ssp=ssp,
            observation=observation,
            parameters=parameters,
            model_config=config,
            preset_name=preset,
        )

    def build_model(self) -> SEDModel:
        """Construct the SEDModel lazily.

        Returns
        -------
        SEDModel
            The forward model (stored in self.model).

        Notes
        -----
        The model is JIT-compiled on the first forward pass.
        This method is called automatically by fit() if needed.
        """
        if self.model is None:
            self.model = SEDModel(self.parameters, self.ssp, observation=self.observation)
        return self.model

    def predict_via_components(self, params):
        """Forward pass via the SEDComponent orchestrator.

        Convenience wrapper around
        :meth:`tengri.SEDModel.predict_state` that lazy-builds
        the underlying ``SEDModel`` and threads the params dict through
        the component chain. Returns a :class:`tengri.protocols.ForwardState`.

        Use this when you want the orchestrator's published cross-component
        derived quantities (``L_ir``, ``L_agn_bol``, ``log_mstar``,
        ``lnu_age``, …) directly without going through
        :meth:`fit`. For inference, keep using :meth:`fit`.

        Parameters
        ----------
        params : Mapping
            Free parameter values keyed by canonical name.

        Returns
        -------
        ForwardState
            See :meth:`SEDModel.predict_state`.

        See Also
        --------
        :meth:`Galaxy.predict` : Unified entry point with a ``backend``
            switch between the legacy ``Prediction`` lazy view and the
            orchestrator's ``ForwardState`` (this method).

        Examples
        --------
        >>> g = Galaxy.from_arrays(filters=..., flux=..., flux_err=...)  # doctest: +SKIP
        >>> state = g.predict_via_components(params)  # doctest: +SKIP
        >>> state.derived["log_mstar"]  # doctest: +SKIP
        """
        return self.build_model().predict_state(params)

    def predict(self, params, backend: str = "legacy"):
        """Compute a forward-model prediction for ``params``.

        Unified entry point. The ``backend`` argument selects
        between the legacy tier-dispatch path (returns a
        :class:`Prediction` lazy view) and the orchestrator path (returns
        a :class:`ForwardState`).

        Parameters
        ----------
        params : Mapping
            Free parameter values keyed by canonical name.
        backend : {"legacy", "component"}, optional
            ``"legacy"`` (default) returns the lazy :class:`Prediction`
            wrapper from :meth:`SEDModel.predict`, exposing the
            ``.sfh`` / ``.sed`` / ``.lines`` / ``.radio`` / ``.xray`` /
            ``.ionizing`` property groups. ``"component"`` returns a
            :class:`tengri.protocols.ForwardState` from the
            SEDComponent orchestrator with all cross-component
            quantities published in ``state.derived``.

            The default remains ``"legacy"`` until the v1.0 cutover.
            The two backends agree on the rest-frame SED at
            ``rtol ≤ 5e-2`` for the configurations the stellar migration
            covers (stellar + dust + IGM + radio + X-ray); see
            ``docs/dev/archive/phase_ii_2_stellar_migration.md``.

        Returns
        -------
        Prediction or ForwardState
            ``Prediction`` for ``backend="legacy"``,
            :class:`tengri.protocols.ForwardState` for ``backend="component"``.

        Raises
        ------
        ValueError
            If ``backend`` is not ``"legacy"`` or ``"component"``.

        Examples
        --------
        Legacy lazy view:

        >>> g = Galaxy.from_arrays(filters=..., flux=..., flux_err=...)  # doctest: +SKIP
        >>> pred = g.predict(params)  # backend="legacy" (default)
        >>> pred.sfh.stellar_mass  # doctest: +SKIP

        Orchestrator path:

        >>> state = g.predict(params, backend="component")
        >>> state.derived["log_mstar"]  # doctest: +SKIP
        """
        if backend == "legacy":
            return self.build_model().predict(params)
        if backend == "component":
            return self.predict_via_components(params)
        raise ValueError(
            f"backend must be 'legacy' or 'component', got {backend!r}. "
            f"Default is 'legacy' until the orchestrator cutover."
        )

    def fit(
        self,
        method: str | None = None,
        verbose: bool = True,
        *,
        approx="auto",
        backend: str | None = None,
        **kwargs,
    ) -> Galaxy:
        """Run inference and store result.

        .. note::

           Unlike :meth:`tengri.ForwardModel.fit`, which defaults to
           ``"vi"``, this facade defaults to ``"map"`` — a point estimate with
           **no uncertainties**. That difference is deliberate: ``Galaxy`` is
           the beginner-facing shortcut and ``"vi"`` costs ~100 s cold and
           ~20 GB RSS at D=6-7, which is not a reasonable thing to do by
           surprise. It is called out here because it used to be silent: five
           fit surfaces disagreed about the default and nothing said so
           (#1289). If you want a posterior, pass ``method="vi"``.

        Parameters
        ----------
        method : str, optional
            Inference method. Default ``"map"``.
            ``tengri.list_inference_methods()`` is the live list — "map",
            "vi", "vi_linear", "mcmc_nuts", "mcmc_raytrace", "mcmc_hmc",
            "mcmc_dynamic_hmc", "mcmc_adjusted_mclmc", "mcmc_ess", "nss"
            among them.

            This docstring used to list ``"vi_native"``, which has never been
            a registered name and raises ``KeyError``, alongside
            ``"mcmc_ghmc"`` and ``"mcmc_mclmc"``, now ``tier="broken"``
            (#1287).
        backend : str, optional
            Deprecated alias for ``method``. Every other fit surface in the
            package spells this argument ``method``; ``backend`` here was the
            odd one out. Passing it emits a :class:`DeprecationWarning`.
        verbose : bool
            Print progress. Default: True.
        approx : {"auto", None} or precompute config, default "auto"
            Fit-time approximation policy. "auto" routes the fit through the
            precompute LUT selected by data type; None forces the exact
            wave-grid path; an explicit config overrides. Model prediction
            stays exact regardless. See :meth:`ForwardModel.fit`.
        **kwargs
            Additional arguments passed to Fitter.run() (e.g., n_steps, n_iterations,
            n_samples, n_warmup, init_from, prewarm, etc.).

        Returns
        -------
        Galaxy
            Returns self for chaining.

        Raises
        ------
        AttributeError
            If flux data not available (from_arrays required).
        """
        # `backend=` is the legacy spelling of `method=` (#1289). Every other
        # fit surface says `method`; this one said `backend`, so the same
        # concept had two names depending on which entry point you found.
        if backend is not None:
            if method is not None:
                raise TypeError(
                    "Galaxy.fit() got both 'method' and its deprecated alias "
                    f"'backend' (method={method!r}, backend={backend!r}). "
                    "Pass 'method' only."
                )
            warnings.warn(
                "Galaxy.fit(backend=...) is deprecated; use "
                "Galaxy.fit(method=...). Every other fit surface in tengri "
                "spells this argument 'method'.",
                DeprecationWarning,
                stacklevel=2,
            )
            method = backend
        if method is None:
            method = "map"

        # Ensure model is built
        self.build_model()

        if not hasattr(self, "_flux_obs"):
            raise AttributeError(
                "Flux data not available. Use Galaxy.from_arrays() to provide observed fluxes."
            )

        # Instantiate Fitter (the model carries its Observation)
        fitter = Fitter(self.model, self.flux_obs, self.noise, approx=approx)

        # Run inference
        self.result = fitter.run(method, verbose=verbose, **kwargs)

        # Record the method used for later save/load
        self._last_backend = method
        # Record the inference backend citation(s) now that a fit has run.
        # Must be `method`, not the raw `backend` alias -- the latter is None
        # unless the caller used the deprecated spelling, which would silently
        # drop the citation for every ordinary call.
        self.bibliography.add_backend(method)

        return self

    def summary(self) -> Any:
        """Return a summary DataFrame of fitted parameters.

        Returns
        -------
        pandas.DataFrame
            One row with parameter medians and 68% credible intervals
            (for sampling-based backends) or point estimates (for MAP).
            Columns: {param}_median, {param}_lo68, {param}_hi68 (sampling),
            or {param}_value (MAP).

        Raises
        ------
        RuntimeError
            If fit() has not been called yet.
        """
        if self.result is None:
            raise RuntimeError("No fit result available. Call fit() first.")

        import tengri

        return tengri.posteriors_to_dataframe([self.result])

    def plot(self, figsize: tuple[int, int] = (14, 5)) -> Any:
        """Plot SED fit and SFH recovery.

        Parameters
        ----------
        figsize : tuple of int
            Figure size (width, height) in inches.

        Returns
        -------
        matplotlib.figure.Figure
            The figure object.

        Raises
        ------
        RuntimeError
            If fit() has not been called yet.
        """
        if self.result is None:
            raise RuntimeError("No fit result available. Call fit() first.")

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Plotting requires matplotlib: pip install matplotlib") from None

        fig, (ax_sed, ax_sfh) = plt.subplots(1, 2, figsize=figsize)

        # SED fit
        if self.observation.photometry is not None:
            wave_eff = np.array([f.lambda_eff for f in self.observation.photometry.filters])
            ax_sed.errorbar(
                wave_eff,
                np.array(self._flux_obs),
                yerr=np.array(self._noise),
                fmt="o",
                ms=6,
                label="Observed",
                elinewidth=1.5,
                capsize=4,
            )

            # Plot posterior median and 68% band
            n_pred = min(256, len(next(iter(self.result.samples.values()))))
            samples_sub = {k: v[:n_pred] for k, v in self.result.samples.items()}
            pred_samples = jax.vmap(self.model.predict_photometry)(samples_sub)
            pred_median = np.median(np.array(pred_samples), axis=0)
            pred_lo68 = np.percentile(np.array(pred_samples), 16, axis=0)
            pred_hi68 = np.percentile(np.array(pred_samples), 84, axis=0)

            ax_sed.scatter(
                wave_eff,
                pred_median,
                marker="D",
                s=40,
                zorder=5,
                label="Model (median)",
            )
            ax_sed.fill_between(wave_eff, pred_lo68, pred_hi68, alpha=0.3, label="68% CI")
            ax_sed.set_xlabel("Wavelength [Å]")
            ax_sed.set_ylabel("Flux [erg/s/cm²/Hz]")
            ax_sed.set_title("SED Fit")
            ax_sed.legend()

        # SFH recovery
        ax_sfh.set_title("Star Formation History")
        ax_sfh.set_xlabel("Lookback time [Gyr]")
        ax_sfh.set_ylabel("SFR [Msun/yr]")
        ax_sfh.legend()

        fig.suptitle(f"Galaxy Fit ({self.preset_name or 'custom'})")
        fig.tight_layout()

        return fig

    def _infer_citation_keys(self) -> list[str]:
        """Infer citation keys based on model configuration and fit backend.

        Returns
        -------
        list of str
            Citation registry keys applicable to this Galaxy.

        Notes
        -----
        Base citations always include: "tengri", "dsps", "jax".
        Additional keys added based on model_config and inference backend.
        Designed defensively — handles missing config fields gracefully.
        """
        # Base components
        citations = ["tengri", "dsps", "jax"]

        # Add based on config (defensive: use getattr to handle missing fields)
        if self.model_config is not None:
            if getattr(self.model_config, "dust", None) is not None:
                citations.append("calzetti2000")
                citations.append("charlot_fall2000")
            if getattr(self.model_config, "nebular", None) is not None:
                citations.append("cue")
            if getattr(self.model_config, "igm", None) is not None:
                citations.append("inoue2014")

        # Add based on inference backend
        last_backend = getattr(self, "_last_backend", None)
        if last_backend is not None:
            if "vi" in last_backend.lower():
                citations.append("nifty")
            if "nuts" in last_backend.lower() or "mcmc" in last_backend.lower():
                citations.append("blackjax")

        return citations

    def cite(self, fmt: str = "list"):
        """Return citations for the components this Galaxy is configured to use.

        Thin wrapper around :attr:`bibliography`. Prefer the Bibliography
        attribute directly for new code.

        Parameters
        ----------
        fmt : {"list", "short", "bibtex", "report", "bibliography"}

            - "list" (default): list of ``Citation`` records.
            - "short": newline-joined one-line forms.
            - "bibtex": BibTeX blocks separated by blank lines.
            - "report": grouped human-readable report.
            - "bibliography": the :class:`Bibliography` container itself.

        """
        if fmt == "bibliography":
            return self.bibliography
        if fmt == "list":
            return self.bibliography.to_list()
        if fmt == "short":
            return "\n".join(str(c) for c in self.bibliography)
        if fmt == "bibtex":
            return self.bibliography.to_bibtex()
        if fmt == "report":
            return self.bibliography.report()
        raise ValueError(
            f"Unknown fmt '{fmt}'. Use one of "
            "'list' / 'short' / 'bibtex' / 'report' / 'bibliography'."
        )

    def explain(self) -> str:
        """Return a plain-English explanation of what was fit.

        Returns
        -------
        str
            Human-readable paragraph describing the fit.
        """
        n_bands = self.observation.photometry.n_filters if self.observation.photometry else 0

        # Determine redshift handling: check whether it's in free_params vs fixed.
        if "redshift" in self.parameters.free_params:
            z_str = "z (free)"
        else:
            # Fixed: try the parameter-dict attribute; fall back to "fixed".
            z_val = None
            params_dict = getattr(self.parameters, "_params", None) or getattr(
                self.parameters, "params", None
            )
            if isinstance(params_dict, dict):
                z_val = params_dict.get("redshift")
            z_str = f"z={z_val}" if z_val is not None else "z (fixed)"

        backend = getattr(self, "_last_backend", None) or "not yet fit"
        n_params = len(self.parameters.free_params)

        return (
            f"Galaxy fit using {self.preset_name or 'custom'} preset at {z_str}. "
            f"Data: {n_bands} photometric bands. Model: {n_params} free parameters. "
            f"Inference: {backend}."
        )

    def save(self, path: str) -> None:
        """Save this Galaxy's fit to an HDF5 file.

        Requires ``.fit()`` to have been called. Wraps FitResult.save()
        with a Provenance snapshot plus a minimal record of the Galaxy
        configuration (preset name, backend used, flux/err arrays if present).

        Parameters
        ----------
        path : str
            Target HDF5 path.

        Raises
        ------
        RuntimeError
            If .fit() has not been called.
        ImportError
            If h5py is not installed.
        ResultSerializationError
            If any sample entry could not be written. The file is still
            created and holds everything writable; see
            :meth:`tengri.results.FitResult.save`.
        """
        if self.result is None:
            raise RuntimeError("Galaxy has not been fitted. Call .fit(...) first.")

        from tengri.results import FitRecord, FitResult

        # Citation keys for this run — mirror the logic in self._infer_citation_keys()
        citation_keys = self._infer_citation_keys()

        fr = FitResult(
            inner=self.result,
            record=FitRecord.capture(),
            citation_keys=citation_keys,
            backend=getattr(self, "_last_backend", None),
            preset=getattr(self, "preset_name", None),
        )
        fr.save(path)

    @classmethod
    def load_result(cls, path: str):
        """Load a FitResult previously saved by Galaxy.save.

        Returns the FitResult directly (not a reconstructed Galaxy — the
        underlying SEDModel and Observation are not part of the HDF5 schema).

        Parameters
        ----------
        path : str
            Path to HDF5 file.

        Returns
        -------
        FitResult
            The loaded result wrapper with provenance and citations.

        Raises
        ------
        ImportError
            If h5py is not installed.
        KeyError
            If HDF5 schema is invalid.
        """
        from tengri.results import FitResult

        return FitResult.load(path)

    @property
    def flux_obs(self) -> np.ndarray:
        """Observed fluxes in canonical units [erg/s/cm²/Hz]."""
        if not hasattr(self, "_flux_obs"):
            raise AttributeError("Flux data not available.")
        return self._flux_obs

    @property
    def noise(self) -> np.ndarray:
        """Noise (flux uncertainties) in canonical units [erg/s/cm²/Hz]."""
        if not hasattr(self, "_noise"):
            raise AttributeError("Noise data not available.")
        return self._noise


def doctor() -> str:
    """Run an environment health check and return a human-readable report.

    Checks:

    - JAX version and backend
    - 64-bit (x64) enabled
    - XLA compilation cache directory
    - tengri version
    - SSP data discoverability
    - Python version
    - Platform

    All checks are non-fatal; failures are reported as "WARNING: ..." rather
    than raising.

    Returns
    -------
    str
        Health check report (also printed).
    """
    lines = []

    # Logo (respects TENGRI_NO_LOGO env var)
    logo_output = logo_str(compact=False)
    if logo_output:
        lines.append(logo_output)
        lines.append("")

    # Header
    lines.append("=" * 60)
    lines.append("tengri Environment Health Check")
    lines.append("=" * 60)
    lines.append("")

    # Python version
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    lines.append(f"Python: {py_version}")

    # Platform
    lines.append(f"Platform: {platform.system()} {platform.release()}")

    # tengri version
    try:
        import tengri

        lines.append(f"tengri: {tengri.__version__}")
    except (ImportError, AttributeError) as e:
        lines.append(f"WARNING: Could not determine tengri version: {e}")

    lines.append("")

    # JAX version and backend
    try:
        lines.append(f"JAX: {jax.__version__}")
        backend = jax.default_backend()
        lines.append(f"JAX backend: {backend}")
    except Exception as e:
        lines.append(f"WARNING: Could not determine JAX backend: {e}")

    # x64 enabled
    try:
        x64_enabled = jax.config.jax_enable_x64
        status = "OK" if x64_enabled else "DISABLED"
        lines.append(f"JAX x64 (float64): {status}")
        if not x64_enabled:
            lines.append("  → Required for cosmological distance calculations at z>0.01")
    except Exception as e:
        lines.append(f"WARNING: Could not check x64: {e}")

    # XLA cache directory
    try:
        cache_dir = jax.config.jax_compilation_cache_dir
        if cache_dir:
            if os.path.isdir(cache_dir):
                writable = os.access(cache_dir, os.W_OK)
                status = "writable" if writable else "READ-ONLY"
                lines.append(f"XLA cache: {cache_dir} ({status})")
            else:
                lines.append(f"WARNING: XLA cache dir does not exist: {cache_dir}")
        else:
            lines.append("XLA cache: not configured")
    except Exception as e:
        lines.append(f"WARNING: Could not check XLA cache: {e}")

    lines.append("")

    # SSP data availability. Search exactly where the loaders search
    # (data_dirs), and glob what download_ssp actually writes: the old
    # "ssp_*.h5" pattern could not match "fsps_prsc_miles_chabrier.h5", so
    # doctor reported "no SSP data" for a correctly populated install.
    lines.append("SSP Data:")
    from tengri._data_setup import TENGRI_DATA_ENV, find_ssp_files

    found = find_ssp_files()
    if found:
        lines.append(f"  ✓ Found: {found[0]}")
        if len(found) > 1:
            lines.append(f"    ({len(found)} SSP grids visible)")

    if not found:
        lines.append("  WARNING: No SSP data found in common locations.")
        lines.append("    Run tengri.download_ssp() to fetch the default grid")
        lines.append("    (tengri.list_known_ssps() shows alternatives), or point")
        lines.append(f"    ${TENGRI_DATA_ENV} at an existing SSP directory.")

    lines.append("")

    # Summary
    lines.append("=" * 60)
    lines.append("For issues, see: https://github.com/suchethac/tengri")
    lines.append("=" * 60)

    report = "\n".join(lines)
    _display(report)
    return report
