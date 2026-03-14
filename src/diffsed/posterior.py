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
    # Display
    # -------------------------------------------------------------------

    def __repr__(self) -> str:
        n = "None" if self.samples is None else next(iter(self.samples.values())).shape[0]
        return (
            f"Posterior(method='{self.method}', "
            f"n_samples={n}, "
            f"wall_time={self.wall_time_s:.1f}s)"
        )
