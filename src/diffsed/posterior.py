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
from typing import Optional

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

    samples: Optional[dict]
    params: dict
    method: str
    wall_time_s: float
    diagnostics: dict
    loss_history: Optional[jnp.ndarray] = None
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
            return {k: jnp.broadcast_to(v, (n,) + v.shape)
                    for k, v in self.params.items()}

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
        from diffsed.param_spec import ParamSpec
        from diffsed.distributions import Fixed, Gaussian

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
            raise ImportError("arviz required: pip install arviz")

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

    def plot_corner(self, params=None, truths=None, figsize=None,
                    color="C0", fig=None, axes=None, label=None):
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
        if self._model is not None:
            try:
                d = self.derived
                for k in ["stellar_mass", "sfr_100myr"]:
                    if k in d:
                        derived[k] = np.array(d[k])
            except Exception:
                pass

        n = len(params) + len(derived)
        if fig is None or axes is None:
            if figsize is None:
                figsize = (min(2.0 * n, 14), min(2.0 * n, 14))
            fig, axes = plt.subplots(n, n, figsize=figsize)
        if n == 1:
            axes = np.array([[axes]])

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
            "stellar_mass": r"log M$_*$",
            "sfr_100myr": r"SFR$_{100}$",
        }

        for i, name_i in enumerate(all_names):
            xi = all_data[name_i]
            if name_i == "stellar_mass":
                xi = np.log10(np.maximum(xi, 1.0))

            for j, name_j in enumerate(all_names):
                ax = axes[i, j]
                xj = all_data[name_j]
                if name_j == "stellar_mass":
                    xj = np.log10(np.maximum(xj, 1.0))

                if j > i:
                    ax.set_visible(False)
                    continue

                if i == j:
                    # Diagonal: 1D KDE + histogram
                    n_bins = min(20, max(5, len(xi) // 3))
                    ax.hist(xi, bins=n_bins, color=color, alpha=0.2,
                            density=True, edgecolor="white", lw=0.5)
                    try:
                        from scipy.stats import gaussian_kde
                        kde = gaussian_kde(xi)
                        x_grid = np.linspace(np.min(xi), np.max(xi), 200)
                        lbl = label if (i == 0 and label) else None
                        ax.plot(x_grid, kde(x_grid), color=color, lw=1.5,
                                label=lbl)
                    except (ImportError, np.linalg.LinAlgError):
                        pass  # fall back to histogram only
                    if truths and name_i in truths:
                        tv = truths[name_i]
                        if name_i == "stellar_mass":
                            tv = np.log10(max(tv, 1.0))
                        ax.axvline(tv, color="k", ls="--", lw=1.5)
                else:
                    # Off-diagonal: 2D KDE contours
                    try:
                        from scipy.stats import gaussian_kde
                        xy = np.vstack([xj, xi])
                        kde = gaussian_kde(xy)
                        x_grid = np.linspace(np.min(xj), np.max(xj), 80)
                        y_grid = np.linspace(np.min(xi), np.max(xi), 80)
                        X, Y = np.meshgrid(x_grid, y_grid)
                        Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
                        # Contour levels at 68% and 95% credible regions
                        Z_sorted = np.sort(Z.ravel())[::-1]
                        Z_cumsum = np.cumsum(Z_sorted) / np.sum(Z_sorted)
                        level_68 = Z_sorted[np.searchsorted(Z_cumsum, 0.68)]
                        level_95 = Z_sorted[np.searchsorted(Z_cumsum, 0.95)]
                        ax.contourf(X, Y, Z, levels=[level_95, level_68, Z.max()],
                                    colors=[color], alpha=[0.1, 0.3])
                        ax.contour(X, Y, Z, levels=[level_95, level_68],
                                   colors=[color], linewidths=0.8, alpha=0.7)
                    except (ImportError, np.linalg.LinAlgError):
                        # Fallback to scatter if KDE fails
                        ax.scatter(xj, xi, s=8, alpha=0.4, color=color,
                                   edgecolors="none")
                    if truths:
                        if name_j in truths and name_i in truths:
                            tj = truths[name_j]
                            ti = truths[name_i]
                            if name_j == "stellar_mass":
                                tj = np.log10(max(tj, 1.0))
                            if name_i == "stellar_mass":
                                ti = np.log10(max(ti, 1.0))
                            ax.axvline(tj, color="k", ls="--", lw=0.8, alpha=0.5)
                            ax.axhline(ti, color="k", ls="--", lw=0.8, alpha=0.5)

                # Labels on edges only
                if i == n - 1:
                    ax.set_xlabel(label_map.get(name_j, name_j), fontsize=11)
                    ax.tick_params(axis='x', labelsize=9, rotation=45)
                else:
                    ax.set_xticklabels([])
                if j == 0 and i > 0:
                    ax.set_ylabel(label_map.get(name_i, name_i), fontsize=11)
                    ax.tick_params(axis='y', labelsize=9)
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
            f"Posterior(method='{self.method}', "
            f"n_samples={n}, "
            f"wall_time={self.wall_time_s:.1f}s)"
        )
