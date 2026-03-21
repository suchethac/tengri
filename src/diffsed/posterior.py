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
    """

    samples: dict | None
    params: dict
    method: str
    wall_time_s: float
    diagnostics: dict
    loss_history: jnp.ndarray | None = None
    _model: object = field(default=None, repr=False)

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

    # -------------------------------------------------------------------
    # Summary statistics
    # -------------------------------------------------------------------

    def summary(self) -> dict:
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
        if "acceptance_rate" in diag:
            diag_parts.append(f"accept={diag['acceptance_rate']:.1%}")
        if "n_divergences" in diag:
            diag_parts.append(f"divergences={diag['n_divergences']}")
        if "final_loss" in diag:
            diag_parts.append(f"loss={diag['final_loss']:.2f}")
        if diag_parts:
            lines.append("")
            lines.append(f"  Diagnostics: {', '.join(diag_parts)}")

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

        Uses the initial positive sequence estimator (Geyer 1992):
        truncate the autocorrelation sum at the first negative pair.

        Returns
        -------
        dict
            Keys: parameter names. Values: ESS (float).
        """
        if self.samples is None:
            raise ValueError("ESS requires samples (not MAP)")

        acfs = self.autocorrelation()
        result = {}

        for name, acf in acfs.items():
            n = next(iter(self.samples.values())).shape[0]
            # Initial positive sequence: sum pairs of consecutive ACF values
            # and stop when the pair sum goes negative
            tau = 1.0  # starts at lag 0 (acf[0] = 1)
            for i in range(1, len(acf) - 1, 2):
                pair_sum = acf[i] + acf[i + 1] if i + 1 < len(acf) else acf[i]
                if pair_sum < 0:
                    break
                tau += 2.0 * pair_sum
            result[name] = n / tau

        return result

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
        from diffsed.distributions import Fixed, Gaussian
        from diffsed.param_spec import ParamSpec

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

    # -------------------------------------------------------------------
    # Display
    # -------------------------------------------------------------------

    def __repr__(self) -> str:
        n = "None" if self.samples is None else next(iter(self.samples.values())).shape[0]
        return (
            f"Posterior(method='{self.method}', n_samples={n}, wall_time={self.wall_time_s:.1f}s)"
        )
