"""Posterior inference results with sampling and diagnostics.

The Posterior object stores parameter samples (or point estimates for MAP),
provides summary statistics, derived quantities, and can convert to ArviZ
format or back to a ParamSpec for mock generation.

Usage:
    result = model.fit(data, noise, method="nuts")
    print(result.summary())
    sfh_draws = [model.predict_sfh(result.resample(key)) for ...]
    idata = result.to_arviz()
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np


@dataclass
class Posterior:
    """Inference results with sampling and diagnostics.

    Attributes
    ----------
    samples : dict or None
        Posterior samples in physical space. Each value has shape (n_samples, ...).
        None for MAP results.
    params : dict
        Best-fit (MAP) or posterior mean parameters.
    method : str
        Inference method name.
    wall_time_s : float
        Total wall-clock time in seconds.
    diagnostics : dict
        Method-specific diagnostics.
    loss_history : array or None
        Loss values over iterations (MAP only).
    _model : Model
        Reference to the model (for derived quantities).
    eline_fluxes : array or None
        Emission line fluxes. Shape (n_samples, n_lines) for MCMC, or
        (n_lines,) for MAP. Flux units match the input data.
    eline_flux_cov : array or None
        Posterior covariance of line fluxes. Shape (n_samples, n_lines, n_lines)
        or (n_lines, n_lines).
    eline_names : tuple or None
        Line identifiers matching eline_fluxes columns.
    eline_wavelengths : array or None
        Rest-frame wavelengths matching eline_fluxes columns (Angstrom).
    """

    samples: dict | None
    params: dict
    method: str
    wall_time_s: float
    diagnostics: dict
    loss_history: jnp.ndarray | None = None
    log_evidence: float | None = None
    _model: object = field(default=None, repr=False)
    _fitter: object = field(default=None, repr=False)
    eline_fluxes: jnp.ndarray | None = field(default=None, repr=False)
    eline_flux_cov: jnp.ndarray | None = field(default=None, repr=False)
    eline_names: tuple | None = field(default=None, repr=False)
    eline_wavelengths: jnp.ndarray | None = field(default=None, repr=False)

    # -------------------------------------------------------------------
    # Derived quantities
    # -------------------------------------------------------------------

    @functools.cached_property
    def derived(self) -> dict:
        """Derived physical quantities (stellar mass, SFR, sSFR).

        For MAP: computed on the single best-fit → dict of scalars.
        For NUTS/geoVI: computed on all samples → dict of arrays.
        """
        if self._model is None:
            raise RuntimeError("No model reference — cannot compute derived quantities")

        if self.samples is None:
            # MAP: single point
            return self._model.predict_derived(self.params)

        # Sampling: compute for each sample
        n_samples = next(iter(self.samples.values())).shape[0]
        derived_lists = {}

        for i in range(n_samples):
            sample_i = {k: v[i] for k, v in self.samples.items()}
            d_i = self._model.predict_derived(sample_i)
            for k, v in d_i.items():
                if k not in derived_lists:
                    derived_lists[k] = []
                derived_lists[k].append(v)

        return {k: jnp.stack(v) for k, v in derived_lists.items()}

    def line_fluxes(self) -> dict[str, tuple[float, float, float]]:
        """Emission line flux posterior summaries.

        Returns median and 68% credible interval for each emission line.

        Returns
        -------
        dict
            ``{line_name: (median, lo_68, hi_68)}`` for each emission line.
            Flux units match the input data. For MAP results, all three
            values are the same (single-point estimate).

        Raises
        ------
        ValueError
            If no emission line fluxes are available. Set
            ``eline_mode="marginalized"`` in ``SpectroscopyConfig`` to enable.

        Examples
        --------
        ::

            fluxes = result.line_fluxes()
            ha_median, ha_lo, ha_hi = fluxes["Halpha"]
        """
        if self.eline_fluxes is None:
            raise ValueError(
                "No emission line fluxes available. "
                "Set eline_mode='marginalized' or 'fitted' in SpectroscopyConfig."
            )
        result = {}
        for i, name in enumerate(self.eline_names):
            if self.eline_fluxes.ndim == 1:
                # MAP: single estimate
                val = float(self.eline_fluxes[i])
                result[name] = (val, val, val)
            else:
                flux_i = self.eline_fluxes[:, i]
                lo, med, hi = jnp.percentile(flux_i, jnp.array([16.0, 50.0, 84.0]))
                result[name] = (float(med), float(lo), float(hi))
        return result

    def bpt_nii(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """BPT-NII ([NII]/Hα vs [OIII]/Hβ) diagram coordinates.

        Returns log10 line ratios for each posterior sample.

        Returns
        -------
        log_nii_ha : array (n_samples,) or scalar
            log10([NII]6584 / Hα)
        log_oiii_hb : array (n_samples,) or scalar
            log10([OIII]5007 / Hβ)

        Raises
        ------
        ValueError
            If emission line fluxes are not available or BPT lines are missing.

        Examples
        --------
        ::

            x, y = result.bpt_nii()
            plt.scatter(x, y, alpha=0.3)
        """
        if self.eline_fluxes is None:
            raise ValueError("No emission line fluxes available.")
        names = list(self.eline_names)
        required = ["NII_6584", "Halpha", "OIII_5007", "Hbeta"]
        missing = [n for n in required if n not in names]
        if missing:
            raise ValueError(f"BPT lines not in catalog: {missing}")

        def _get(name):
            idx = names.index(name)
            if self.eline_fluxes.ndim == 1:
                return self.eline_fluxes[idx]
            return self.eline_fluxes[:, idx]

        nii = _get("NII_6584")
        ha = _get("Halpha")
        oiii = _get("OIII_5007")
        hb = _get("Hbeta")
        # NaN for non-detections (negative amplitudes); clamping to 1e-30 would
        # give log10(1e-30/F) ~ -30 and corrupt BPT diagrams.
        log_nii_ha = jnp.where(
            (nii > 0) & (ha > 0),
            jnp.log10(jnp.maximum(nii, 1e-30) / jnp.maximum(ha, 1e-30)),
            jnp.nan,
        )
        log_oiii_hb = jnp.where(
            (oiii > 0) & (hb > 0),
            jnp.log10(jnp.maximum(oiii, 1e-30) / jnp.maximum(hb, 1e-30)),
            jnp.nan,
        )
        return log_nii_ha, log_oiii_hb

    def balmer_decrement(self) -> tuple[float, float, float]:
        """Observed Hα/Hβ ratio from posterior line fluxes.

        Returns the posterior distribution of the Balmer decrement
        (Hα/Hβ), which is a direct dust attenuation diagnostic.
        The intrinsic Case B ratio is 2.86; higher values indicate dust.

        Returns
        -------
        tuple
            ``(median, lo_68, hi_68)`` of Hα/Hβ. For MAP results,
            all three values are equal.

        Raises
        ------
        ValueError
            If Hα or Hβ fluxes are not available.

        Examples
        --------
        ::

            med, lo, hi = result.balmer_decrement()
            print(f"Ha/Hb = {med:.2f} [{lo:.2f}, {hi:.2f}]")
            # Intrinsic Case B = 2.86; excess indicates dust attenuation
        """
        if self.eline_fluxes is None:
            raise ValueError("No emission line fluxes available.")
        names = list(self.eline_names)
        for n in ["Halpha", "Hbeta"]:
            if n not in names:
                raise ValueError(f"Required line '{n}' not in catalog.")

        def _get(name):
            idx = names.index(name)
            if self.eline_fluxes.ndim == 1:
                return self.eline_fluxes[idx]
            return self.eline_fluxes[:, idx]

        eps = 1e-30
        ratio = jnp.maximum(_get("Halpha"), eps) / jnp.maximum(_get("Hbeta"), eps)
        if ratio.ndim == 0:
            v = float(ratio)
            return (v, v, v)
        lo, med, hi = jnp.percentile(ratio, jnp.array([16.0, 50.0, 84.0]))
        return (float(med), float(lo), float(hi))

    def equivalent_widths(self) -> dict[str, tuple[float, float, float]]:
        """Rest-frame emission line equivalent widths.

        EW(λ) = F_line / f_cont(λ_line), where f_cont is the continuum
        flux density at the line center.

        Returns
        -------
        dict
            ``{line_name: (median_EW, lo_68, hi_68)}`` in Angstrom.

        Notes
        -----
        Requires ``_model`` to compute the continuum prediction.
        Not yet implemented — raises ``NotImplementedError``.
        """
        raise NotImplementedError(
            "equivalent_widths() not yet implemented. "
            "Use line_fluxes() and divide by the continuum model manually."
        )

    # -------------------------------------------------------------------
    # Summary statistics
    # -------------------------------------------------------------------

    def stats(self) -> dict:
        """Median and 68% credible intervals for all parameters.

        Returns
        -------
        dict
            Keys: parameter names.
            Values: dict with "median", "lo_68", "hi_68" (or just "value" for MAP).
        """
        result = {}

        if self.samples is None:
            # MAP: point estimates
            for name, val in self.params.items():
                if name == "psd_xi":
                    continue
                result[name] = {"value": float(jnp.mean(val))}
        else:
            # Sampling: percentiles
            for name, arr in self.samples.items():
                if name == "psd_xi":
                    continue
                if arr.ndim == 1:
                    vals = np.array(arr)
                    result[name] = {
                        "median": float(np.median(vals)),
                        "lo_68": float(np.percentile(vals, 16)),
                        "hi_68": float(np.percentile(vals, 84)),
                    }

        return result

    def summary(self) -> dict:
        """Deprecated. Use stats() instead."""
        import warnings

        warnings.warn(
            "Posterior.summary() is deprecated. Use Posterior.stats() instead. "
            "Will be removed in tengri v1.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.stats()

    def summary_table(self) -> str:
        """Return a formatted string table of parameter summaries.

        For MAP: shows parameter values.
        For sampling: shows median with 68% credible intervals and ESS.

        Returns
        -------
        str
            Formatted table string.
        """
        sep = "─" * 66
        lines: list[str] = []
        n_samples = "MAP" if self.samples is None else next(iter(self.samples.values())).shape[0]
        lines.append(
            f"Posterior  method: {self.method}  "
            f"samples: {n_samples}  "
            f"wall_time: {self.wall_time_s:.1f}s"
        )
        lines.append(sep)

        stats = self.summary()
        if not stats:
            lines.append("  (no parameters)")
            lines.append(sep)
            return "\n".join(lines)

        if self.samples is None:
            # MAP table
            hdr = f"  {'Parameter':<32s} {'Value':>12s}"
            lines.append(hdr)
            lines.append("  " + "─" * 44)
            for name in sorted(stats):
                val = stats[name]["value"]
                lines.append(f"  {name:<32s} {val:>12.4f}")
        else:
            # Sampling table with credible intervals
            # Try to get ESS if available
            ess = self.diagnostics.get("ess_bulk", {})
            hdr = f"  {'Parameter':<28s} {'Median':>9s} {'16%':>9s} {'84%':>9s} {'ESS':>7s}"
            lines.append(hdr)
            lines.append("  " + "─" * 64)
            for name in sorted(stats):
                s = stats[name]
                med = f"{s['median']:.4f}"
                lo = f"{s['lo_68']:.4f}"
                hi = f"{s['hi_68']:.4f}"
                ess_val = ess.get(name)
                ess_str = f"{ess_val:.0f}" if ess_val is not None else "—"
                lines.append(f"  {name:<28s} {med:>9s} {lo:>9s} {hi:>9s} {ess_str:>7s}")

        # Diagnostics summary
        diag = self.diagnostics
        diag_parts: list[str] = []
        if "accept_rate" in diag:  # key from raytrace (fitter.py:3509)
            diag_parts.append(f"accept={diag['accept_rate']:.1%}")
        if "n_divergent" in diag:  # key from NUTS (nuts.py:193)
            diag_parts.append(f"divergences={diag['n_divergent']}")
        if "final_loss" in diag:
            diag_parts.append(f"loss={diag['final_loss']:.2f}")
        if diag_parts:
            lines.append("")
            lines.append(f"  Diagnostics: {', '.join(diag_parts)}")

        if self.log_evidence is not None:
            err = self.diagnostics.get("log_evidence_err")
            if err is not None:
                lines.append(f"  log Z (evidence) = {self.log_evidence:.2f} ± {err:.2f}")
            else:
                lines.append(f"  log Z (evidence) = {self.log_evidence:.2f}")

        lines.append(sep)
        return "\n".join(lines)

    # -------------------------------------------------------------------
    # Autocorrelation and effective sample size
    # -------------------------------------------------------------------

    @staticmethod
    def _autocorrelation_1d(x: np.ndarray, max_lag: int | None = None) -> np.ndarray:
        """Compute normalized autocorrelation function for a 1D chain.

        Uses FFT for efficiency (O(N log N) instead of O(N * max_lag)).
        """
        n = len(x)
        if max_lag is None:
            max_lag = n // 2
        max_lag = min(max_lag, n - 1)

        x = x - np.mean(x)
        var = np.var(x)
        if var < 1e-30:
            return np.zeros(max_lag + 1)

        # FFT-based autocorrelation
        fft_size = 2 ** int(np.ceil(np.log2(2 * n)))
        fft_x = np.fft.rfft(x, n=fft_size)
        acf_full = np.fft.irfft(fft_x * np.conj(fft_x))
        acf = acf_full[: max_lag + 1] / (var * n)
        return acf

    def autocorrelation(self, max_lag: int | None = None) -> dict:
        """Compute autocorrelation function for each scalar parameter.

        Parameters
        ----------
        max_lag : int, optional
            Maximum lag. Default: n_samples // 2.

        Returns
        -------
        dict
            Keys: parameter names.
            Values: 1D array of autocorrelation from lag 0 to max_lag.
        """
        if self.samples is None:
            raise ValueError("Autocorrelation requires samples (not MAP)")

        result = {}
        for name, arr in self.samples.items():
            if name == "psd_xi":
                continue
            if arr.ndim == 1:
                result[name] = self._autocorrelation_1d(np.array(arr), max_lag)
        return result

    def effective_sample_size(self) -> dict:
        """Estimate effective sample size (ESS) for each parameter.

        Uses Sokal's self-consistent window method (Behroozi 2025):
        τ = 1 + 2 Σ ρ(k), truncated at k > 5τ. Takes the max of
        standard and absolute-deviation autocorrelation times for
        a conservative estimate.

        Returns
        -------
        dict
            Keys: parameter names. Values: ESS (float).
        """
        if self.samples is None:
            raise ValueError("ESS requires samples (not MAP)")

        from tengri.diagnostics.autocorrelation import effective_sample_size

        ess_info = effective_sample_size({k: np.asarray(v) for k, v in self.samples.items()})
        return {name: info["ess"] for name, info in ess_info.items()}

    def autocorrelation_time(self) -> dict:
        """Estimate integrated autocorrelation time for each parameter.

        Uses Sokal's self-consistent window method with both standard
        and absolute-deviation modes (Behroozi 2025).

        Returns
        -------
        dict
            Keys: parameter names.
            Values: dict with 'tau_standard', 'tau_absolute', 'tau_max',
            'ess', 'chain_converged'.
        """
        if self.samples is None:
            raise ValueError("Autocorrelation time requires samples (not MAP)")

        from tengri.diagnostics.autocorrelation import effective_sample_size

        return effective_sample_size({k: np.asarray(v) for k, v in self.samples.items()})

    def check_convergence(self, verbose: bool = True) -> dict:
        """Check chain convergence using autocorrelation diagnostics.

        Follows Behroozi (2025): chain is converged when N > 5τ for
        all parameters.

        Parameters
        ----------
        verbose : bool
            Print diagnostics table.

        Returns
        -------
        dict
            Keys: 'all_converged', 'params', 'warnings'.
        """
        if self.samples is None:
            raise ValueError("Convergence check requires samples (not MAP)")

        from tengri.diagnostics.autocorrelation import check_chain_length

        return check_chain_length(
            {k: np.asarray(v) for k, v in self.samples.items()},
            verbose=verbose,
        )

    def diagnostics_summary(self) -> str:
        """Print a diagnostics summary including ESS and R-hat proxy."""
        if self.samples is None:
            return f"MAP result (no samples): method={self.method}"

        ess = self.effective_sample_size()
        summary = self.summary()

        lines = [
            f"Method: {self.method}",
            f"Samples: {next(iter(self.samples.values())).shape[0]}",
            f"Wall time: {self.wall_time_s:.1f}s",
            "",
            f"{'Parameter':<22s} {'Median':>8s} {'68% CI':>18s} {'ESS':>6s}",
            "-" * 58,
        ]
        for name in sorted(summary.keys()):
            s = summary[name]
            if "median" in s:
                e = ess.get(name, float("nan"))
                ci = f"[{s['lo_68']:.3f}, {s['hi_68']:.3f}]"
                lines.append(f"{name:<22s} {s['median']:>8.3f} {ci:>18s} {e:>6.0f}")

        return "\n".join(lines)

    # -------------------------------------------------------------------
    # Resampling
    # -------------------------------------------------------------------

    def resample(self, key, n=1) -> dict:
        """Resample from posterior with replacement.

        Parameters
        ----------
        key : PRNGKey
            Random key.
        n : int
            Number of resamples.

        Returns
        -------
        dict
            If n=1: parameter name → scalar value.
            If n>1: parameter name → array of shape (n, ...).
        """
        if self.samples is None:
            # MAP: just return the point estimate
            if n == 1:
                return dict(self.params)
            return {k: jnp.broadcast_to(v, (n, *v.shape)) for k, v in self.params.items()}

        n_available = next(iter(self.samples.values())).shape[0]
        indices = jax.random.choice(key, n_available, shape=(n,), replace=True)

        if n == 1:
            idx = indices[0]
            return {k: v[idx] for k, v in self.samples.items()}

        return {k: v[indices] for k, v in self.samples.items()}

    # -------------------------------------------------------------------
    # Conversion
    # -------------------------------------------------------------------

    def to_param_spec(self):
        """Convert posterior to an empirical ParamSpec.

        For MAP: all parameters become Fixed at their best-fit values.
        For sampling: fit clipped Gaussian to each marginal.

        Returns
        -------
        ParamSpec
        """
        from tengri.core.parameters import ParamSpec
        from tengri.distributions import Fixed, Gaussian

        kwargs = {}

        if self.samples is None:
            # MAP: all Fixed
            for name, val in self.params.items():
                if name == "psd_xi":
                    continue
                kwargs[name] = Fixed(float(jnp.mean(val)))
        else:
            # Sampling: fit Gaussian to each marginal
            for name, arr in self.samples.items():
                if name == "psd_xi":
                    continue
                if arr.ndim == 1:
                    vals = np.array(arr)
                    kwargs[name] = Gaussian(
                        mu=float(np.median(vals)),
                        sigma=float(np.std(vals)),
                        lo=float(np.min(vals)),
                        hi=float(np.max(vals)),
                    )

        # Copy settings from original spec
        if self._model is not None:
            kwargs["stochastic"] = self._model.spec.stochastic
            kwargs["n_grid"] = self._model.spec.n_grid

        return ParamSpec(**kwargs)

    def to_arviz(self):
        """Convert to ArviZ InferenceData for diagnostics.

        Returns
        -------
        az.InferenceData
        """
        try:
            import arviz as az
        except ImportError:
            raise ImportError("arviz required: pip install arviz") from None

        if self.samples is None:
            raise ValueError("Cannot convert MAP result to ArviZ (no samples)")

        posterior = {}
        for name, arr in self.samples.items():
            if name == "psd_xi":
                continue  # skip high-dimensional latent
            arr_np = np.array(arr)
            if arr_np.ndim == 1:
                # Add chain dimension: (1, n_samples)
                posterior[name] = arr_np[np.newaxis, :]

        return az.from_dict(posterior=posterior)

    # -------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------

    def plot_corner(
        self, params=None, truths=None, figsize=None, color="C0", fig=None, axes=None, label=None
    ):
        """Plot corner (triangle) plot of posterior distributions.

        Parameters
        ----------
        params : list of str, optional
            Parameter names to include. Defaults to all scalar physical params.
        truths : dict, optional
            True values to mark with dashed lines.
        figsize : tuple, optional
            Figure size.
        color : str
            Color for this posterior's contours and histograms.
        fig, axes : matplotlib Figure, ndarray of Axes, optional
            If provided, overlay on existing corner plot (for comparing posteriors).
        label : str, optional
            Legend label for this posterior.

        Returns
        -------
        fig : matplotlib Figure
        """
        import matplotlib.pyplot as plt

        if self.samples is None:
            raise ValueError("Corner plot requires posterior samples (not MAP)")

        if params is None:
            # Exclude psd_xi (high-dimensional) and fixed params (constant values)
            params = []
            for k in sorted(self.samples.keys()):
                if k == "psd_xi":
                    continue
                arr = self.samples[k]
                if arr.ndim != 1:
                    continue
                # Skip if all values are identical (fixed parameter)
                if float(jnp.std(arr)) < 1e-10:
                    continue
                params.append(k)

        # Add derived quantities if model is available
        derived = {}
        derived_truths = {}
        if self._model is not None:
            try:
                d = self.derived
                for k in ["stellar_mass", "sfr_100myr", "sfr_10myr"]:
                    if k in d:
                        derived[k] = np.array(d[k])
            except Exception:
                pass
            # Compute truth derived quantities for truth lines
            if truths is not None:
                try:
                    d_true = self._model.predict_derived(truths)
                    for k in derived:
                        if k in d_true:
                            derived_truths[k] = float(d_true[k])
                except Exception:
                    pass

        n = len(params) + len(derived)
        if fig is None or axes is None:
            if figsize is None:
                figsize = (min(2.0 * n, 14), min(2.0 * n, 14))
            fig, axes = plt.subplots(n, n, figsize=figsize)
        if n == 1:
            axes = np.array([[axes]])
        # If axes is a flat list (e.g., from fig.axes), reshape to 2D
        if isinstance(axes, list):
            axes = np.array(axes).reshape(n, n)

        all_names = list(params) + list(derived.keys())
        all_data = {}
        for name in params:
            all_data[name] = np.array(self.samples[name])
        all_data.update(derived)

        # Labels
        label_map = {
            "sfh_alpha": r"$\alpha$",
            "sfh_beta": r"$\beta$",
            "sfh_tau_peak_gyr": r"$\tau_{\rm peak}$ (Gyr)",
            "sfh_peak_sfr": r"SFR$_{\rm peak}$",
            "psd_sigma": r"$\sigma_{\rm burst}$",
            "psd_tau_myr": r"$\tau_{\rm burst}$ (Myr)",
            "met_logzsol": r"log Z",
            "dust_tau_bc": r"$\tau_{\rm bc}$",
            "dust_tau_diff": r"$\tau_{\rm diff}$",
            "stellar_mass": r"$\log\,M_*$",
            "sfr_100myr": r"$\log\,$SFR$_{100}$",
            "sfr_10myr": r"$\log\,$SFR$_{10}$",
        }

        # Merge derived truths into truths dict for truth-line plotting
        all_truths = dict(truths) if truths else {}
        all_truths.update(derived_truths)

        for i, name_i in enumerate(all_names):
            xi = all_data[name_i]
            _log_derived = ("stellar_mass", "sfr_100myr", "sfr_10myr")
            if name_i in _log_derived:
                xi = np.log10(np.maximum(xi, 1e-30))

            for j, name_j in enumerate(all_names):
                ax = axes[i, j]
                xj = all_data[name_j]
                if name_j in _log_derived:
                    xj = np.log10(np.maximum(xj, 1e-30))

                if j > i:
                    ax.set_visible(False)
                    continue

                if i == j:
                    # Diagonal: 1D histogram + KDE + quantile title
                    # (following corner.py conventions)
                    n_bins = min(20, max(5, len(xi) // 3))
                    ax.hist(
                        xi,
                        bins=n_bins,
                        color=color,
                        alpha=0.2,
                        density=True,
                        edgecolor="white",
                        lw=0.5,
                    )
                    try:
                        from scipy.stats import gaussian_kde

                        kde = gaussian_kde(xi)
                        x_grid = np.linspace(np.min(xi), np.max(xi), 200)
                        lbl = label if (i == 0 and label) else None
                        ax.plot(x_grid, kde(x_grid), color=color, lw=1.5, label=lbl)
                    except (ImportError, np.linalg.LinAlgError):
                        pass  # fall back to histogram only
                    # Quantile lines + title (corner.py style)
                    q16, q50, q84 = np.percentile(xi, [16, 50, 84])
                    ax.axvline(q50, color=color, ls="-", lw=1.0, alpha=0.6)
                    ax.axvline(q16, color=color, ls=":", lw=0.7, alpha=0.4)
                    ax.axvline(q84, color=color, ls=":", lw=0.7, alpha=0.4)
                    if all_truths and name_i in all_truths:
                        tv = all_truths[name_i]
                        if name_i in ("stellar_mass", "sfr_100myr", "sfr_10myr"):
                            tv = np.log10(max(float(tv), 1e-30))
                        ax.axvline(tv, color="#4682b4", ls="--", lw=1.5)
                else:
                    # Off-diagonal: 2D histogram contours (corner.py style)
                    # hist2d is more robust than KDE at low ESS
                    try:
                        from scipy.ndimage import gaussian_filter

                        n_bins_2d = min(30, max(10, int(np.sqrt(len(xj)))))
                        H, xe, ye = np.histogram2d(
                            xj,
                            xi,
                            bins=n_bins_2d,
                            range=[[np.min(xj), np.max(xj)], [np.min(xi), np.max(xi)]],
                        )
                        H = gaussian_filter(H, sigma=1.0)
                        # Contour levels at 68% and 95% credible regions
                        H_sorted = np.sort(H.ravel())[::-1]
                        H_cumsum = np.cumsum(H_sorted) / np.sum(H_sorted)
                        level_68 = H_sorted[np.searchsorted(H_cumsum, 0.68)]
                        level_95 = H_sorted[np.searchsorted(H_cumsum, 0.95)]
                        xc = 0.5 * (xe[:-1] + xe[1:])
                        yc = 0.5 * (ye[:-1] + ye[1:])
                        X, Y = np.meshgrid(xc, yc)
                        # Guard: contourf needs strictly increasing levels
                        levels = sorted(set([level_95, level_68, H.max()]))
                        if len(levels) < 2 or H.max() <= 0:
                            raise ValueError("degenerate histogram")
                        cs = ax.contourf(
                            X,
                            Y,
                            H.T,
                            levels=levels,
                            colors=[color],
                            alpha=np.linspace(0.1, 0.3, len(levels) - 1),
                        )
                        # Remove white edges between contour bands.
                        for c in getattr(cs, "collections", []):
                            c.set_edgecolor("face")
                            c.set_rasterized(True)
                        ax.contour(
                            X,
                            Y,
                            H.T,
                            levels=levels[:-1],
                            colors=[color],
                            linewidths=0.8,
                            alpha=0.7,
                        )
                    except (ImportError, np.linalg.LinAlgError, ValueError):
                        # Fallback to scatter if hist2d fails
                        ax.scatter(xj, xi, s=8, alpha=0.4, color=color, edgecolors="none")
                    if all_truths and name_j in all_truths and name_i in all_truths:
                        tj = float(all_truths[name_j])
                        ti = float(all_truths[name_i])
                        if name_j in _log_derived:
                            tj = np.log10(max(tj, 1e-30))
                        if name_i in _log_derived:
                            ti = np.log10(max(ti, 1e-30))
                        ax.axvline(tj, color="#4682b4", ls="--", lw=0.8, alpha=0.5)
                        ax.axhline(ti, color="#4682b4", ls="--", lw=0.8, alpha=0.5)

                # Labels on edges only
                if i == n - 1:
                    ax.set_xlabel(label_map.get(name_j, name_j), fontsize=11)
                    ax.tick_params(axis="x", labelsize=9, rotation=45)
                else:
                    ax.set_xticklabels([])
                if j == 0 and i > 0:
                    ax.set_ylabel(label_map.get(name_i, name_i), fontsize=11)
                    ax.tick_params(axis="y", labelsize=9)
                else:
                    ax.set_yticklabels([])

                ax.tick_params(labelsize=9)

        plt.tight_layout()
        return fig

    def plot_sed(self, n_draws=200, wave_range=(1000, 30000), ax=None):
        """Plot posterior predictive SED with credible interval.

        Draws ``n_draws`` parameter samples from the posterior, computes
        the rest-frame SED for each, and shades the 16th–84th percentile
        band around the median.

        Parameters
        ----------
        n_draws : int
            Number of posterior draws to use for the band. Ignored for
            MAP results (plots single SED).
        wave_range : (float, float)
            Wavelength range in Angstrom to display.
        ax : matplotlib Axes, optional
            Axes to plot on. Creates new figure if None.

        Returns
        -------
        fig : matplotlib Figure
        """
        import matplotlib.pyplot as plt

        if self._model is None:
            raise ValueError("plot_sed requires a model reference (produced by model.fit())")

        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        else:
            fig = ax.get_figure()

        model = self._model

        if self.samples is not None and n_draws > 1:
            key = jax.random.PRNGKey(0)
            n_samples = next(iter(self.samples.values())).shape[0]
            idx = jax.random.choice(
                key, n_samples, shape=(min(n_draws, n_samples),), replace=False
            )
            seds = []
            for i in np.array(idx):
                p = {
                    k: float(v[i]) if v.ndim == 1 else np.array(v[i])
                    for k, v in self.samples.items()
                }
                try:
                    sed = np.array(model.predict_rest_sed(p).sed)
                    seds.append(sed)
                except Exception:
                    pass
            if seds:
                seds = np.stack(seds, axis=0)
                wave = np.array(model.wavelengths)
                mask = (wave >= wave_range[0]) & (wave <= wave_range[1])
                wave_m = wave[mask]
                lo = np.percentile(seds[:, mask], 16, axis=0)
                med = np.percentile(seds[:, mask], 50, axis=0)
                hi = np.percentile(seds[:, mask], 84, axis=0)
                norm = float(med[np.argmin(np.abs(wave_m - 5500))]) or 1.0
                ax.fill_between(
                    wave_m,
                    lo * wave_m / norm,
                    hi * wave_m / norm,
                    alpha=0.3,
                    color="C0",
                    label="16–84%",
                )
                ax.plot(wave_m, med * wave_m / norm, color="C0", lw=1.8, label="median")
        else:
            wave = np.array(model.wavelengths)
            mask = (wave >= wave_range[0]) & (wave <= wave_range[1])
            sed = np.array(model.predict_rest_sed(self.params).sed)
            norm = float(sed[np.argmin(np.abs(wave - 5500))]) or 1.0
            ax.plot(
                wave[mask], sed[mask] * wave[mask] / norm, color="C0", lw=1.8, label="best-fit"
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Wavelength [$\AA$]", fontsize=11)
        ax.set_ylabel(r"$\lambda F_\lambda$ (normalized at 5500 Å)", fontsize=11)
        ax.legend(fontsize=10, frameon=False)
        ax.set_title("Posterior Predictive SED", fontsize=11)
        fig.tight_layout()
        return fig

    def plot_sfh(self, n_draws=200, ax=None):
        """Plot posterior SFH with credible interval.

        Parameters
        ----------
        n_draws : int
            Number of posterior draws. Ignored for MAP (plots single SFH).
        ax : matplotlib Axes, optional
            Axes to plot on. Creates new figure if None.

        Returns
        -------
        fig : matplotlib Figure
        """
        import matplotlib.pyplot as plt

        if self._model is None:
            raise ValueError("plot_sfh requires a model reference (produced by model.fit())")

        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 4))
        else:
            fig = ax.get_figure()

        model = self._model

        if self.samples is not None and n_draws > 1:
            key = jax.random.PRNGKey(1)
            n_samples = next(iter(self.samples.values())).shape[0]
            idx = jax.random.choice(
                key, n_samples, shape=(min(n_draws, n_samples),), replace=False
            )
            sfhs_mean, sfhs_full, t_gyr = [], [], None
            for i in np.array(idx):
                p = {
                    k: float(v[i]) if v.ndim == 1 else np.array(v[i])
                    for k, v in self.samples.items()
                }
                try:
                    sfh_dict = model.predict_sfh(p)
                    if t_gyr is None:
                        t_gyr = np.array(sfh_dict["t_gyr"])
                    sfhs_mean.append(np.array(sfh_dict["sfr_mean"]))
                    sfhs_full.append(np.array(sfh_dict["sfr_full"]))
                except Exception:
                    pass
            if sfhs_mean and t_gyr is not None:
                sfhs_mean = np.stack(sfhs_mean, axis=0)
                sfhs_full = np.stack(sfhs_full, axis=0)
                lo = np.percentile(sfhs_full, 16, axis=0)
                med = np.percentile(sfhs_full, 50, axis=0)
                hi = np.percentile(sfhs_full, 84, axis=0)
                ax.fill_between(t_gyr, lo, hi, alpha=0.3, color="C0", label="16–84%")
                ax.plot(t_gyr, med, color="C0", lw=1.8, label="median (stochastic)")
                med_mean = np.percentile(sfhs_mean, 50, axis=0)
                ax.plot(t_gyr, med_mean, color="0.5", lw=1.2, ls="--", label="median (smooth)")
        else:
            sfh_dict = model.predict_sfh(self.params)
            t_gyr = np.array(sfh_dict["t_gyr"])
            ax.plot(t_gyr, np.array(sfh_dict["sfr_full"]), color="C0", lw=1.8, label="best-fit")
            ax.plot(
                t_gyr,
                np.array(sfh_dict["sfr_mean"]),
                color="0.5",
                lw=1.2,
                ls="--",
                label="smooth component",
            )

        ax.set_xlabel("Lookback time [Gyr]", fontsize=11)
        ax.set_ylabel(r"SFR [$M_\odot$ yr$^{-1}$]", fontsize=11)
        ax.legend(fontsize=10, frameon=False)
        ax.set_title("Star Formation History", fontsize=11)
        fig.tight_layout()
        return fig

    # -------------------------------------------------------------------
    # Display
    # -------------------------------------------------------------------

    def __repr__(self) -> str:
        n = "None" if self.samples is None else next(iter(self.samples.values())).shape[0]
        return (
            f"Posterior(method='{self.method}', n_samples={n}, wall_time={self.wall_time_s:.1f}s)"
        )

    # -------------------------------------------------------------------
    # Method chaining
    # -------------------------------------------------------------------

    def refine(self, method: str, **kwargs):
        """Re-run inference from this result using a different method.

        Requires that this Posterior was produced by ``model.fit()`` or
        ``fitter.run()`` — both set the ``._fitter`` back-reference.

        Parameters
        ----------
        method : str
            Any canonical method name accepted by ``Fitter.run()``.
            E.g. ``"mcmc_raytrace"``, ``"mcmc_nuts"``, ``"vi"``.
        **kwargs
            Passed to ``Fitter.run()`` (e.g. ``n_steps``, ``n_warmup``).

        Returns
        -------
        Posterior
            New result warm-started from this posterior.

        Raises
        ------
        RuntimeError
            If ``._fitter`` is not set (Posterior created outside model.fit/fitter.run).

        Examples
        --------
        >>> result_vi = model.fit(flux, noise)
        >>> result_exact = result_vi.refine("mcmc_raytrace", n_steps=1000)
        """
        if self._fitter is None:
            raise RuntimeError(
                "Posterior.refine() requires a back-reference to its Fitter. "
                "Use model.fit() or fitter.run() to produce this Posterior. "
                "Posteriors loaded from disk or created manually lack this reference."
            )
        return self._fitter.run(method, init_from=self, **kwargs)

    def validate(self, n_steps: int = 200, **kwargs):
        """Run a short MCMC check and return a validation summary.

        Runs ``n_steps`` of Ray Tracing (or NUTS for D≤20) from this
        posterior's MAP estimate, then computes the marginal overlap
        between this posterior and the MCMC check posterior for each
        parameter.

        Parameters
        ----------
        n_steps : int
            Number of MCMC steps. Default 200 (quick sanity check).
        **kwargs
            Forwarded to the MCMC run.

        Returns
        -------
        dict
            Keys: ``"mcmc_result"`` (Posterior), ``"overlap"`` (dict of
            float per parameter, 1.0 = perfect overlap), ``"passed"``
            (bool, True when all overlaps > 0.5).

        Raises
        ------
        RuntimeError
            If ``._fitter`` is not set.
        """
        if self._fitter is None:
            raise RuntimeError(
                "Posterior.validate() requires a back-reference to its Fitter. "
                "Use model.fit() or fitter.run() to produce this Posterior."
            )
        d = self._fitter.spec.n_free
        mcmc_method = "mcmc_nuts" if d <= 20 else "mcmc_raytrace"
        mcmc_result = self._fitter.run(mcmc_method, init_from=self, n_steps=n_steps, **kwargs)

        # Compute per-parameter marginal overlap (histogram intersection)
        overlap: dict[str, float] = {}
        if self.samples is not None and mcmc_result.samples is not None:
            import numpy as np

            for name in self.samples:
                if name == "psd_xi":
                    continue
                vi_arr = np.array(self.samples[name])
                mc_arr = np.array(mcmc_result.samples[name])
                if vi_arr.ndim != 1:
                    continue
                lo = min(vi_arr.min(), mc_arr.min())
                hi = max(vi_arr.max(), mc_arr.max())
                if hi <= lo:
                    overlap[name] = 1.0
                    continue
                bins = np.linspace(lo, hi, 30)
                h_vi, _ = np.histogram(vi_arr, bins=bins, density=True)
                h_mc, _ = np.histogram(mc_arr, bins=bins, density=True)
                bin_w = bins[1] - bins[0]
                overlap[name] = float(np.sum(np.minimum(h_vi, h_mc)) * bin_w)

        passed = all(v > 0.5 for v in overlap.values()) if overlap else True
        return {"mcmc_result": mcmc_result, "overlap": overlap, "passed": passed}
