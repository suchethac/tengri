# SPDX-License-Identifier: BSD-3-Clause
"""Simulation-Based Inference (SBI) infrastructure for tengri.

Provides training data generation from tengri's forward model, HDF5 I/O
for external neural posterior training, and a wrapper for pre-trained
neural posteriors.

This module does NOT implement SBI training (use ``sbi``, ``nflows``, or
``zuko`` for that). It bridges tengri's differentiable forward model with
external SBI frameworks by generating (theta, x) pairs from the prior.

Usage
-----
Generate training data::

    from tengri import SEDModel, Parameters, Uniform
    from tengri.inference.sbi import generate_sbi_training_data, save_sbi_training_data

    spec = Parameters(...)
    model = SEDModel(spec, ssp, filters=filters)
    data = generate_sbi_training_data(model, spec, n_samples=100_000)
    save_sbi_training_data(data, "sbi_training.h5")

Load a pre-trained posterior::

    from tengri.inference.sbi import SBIPosterior

    posterior = SBIPosterior.from_file("trained_posterior.pkl")
    samples = posterior.sample(observation, n_samples=10_000)

References
----------

- Cranmer, Brehmer & Louppe 2020, PNAS, 117, 30055
- Alsing et al. 2019, MNRAS, 488, 4440 (DELFI for galaxy SEDs)

"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from tengri.parameters.parameters import Parameters


def generate_sbi_training_data(
    model: Any,
    spec: Parameters,
    n_samples: int = 100_000,
    key: jnp.ndarray | None = None,
    obs_type: str = "photometry",
    noise_model: str = "gaussian",
    snr_range: tuple[float, float] = (5.0, 50.0),
) -> dict[str, Any]:
    """Generate training data for simulation-based inference.

    Samples parameters from the prior (Parameters), runs the forward
    model, adds realistic noise, and returns (theta, x) pairs suitable
    for training a neural posterior estimator.

    Parameters
    ----------
    model: Model
        Configured tengri Model with filters/wavelength grid.
    spec: Parameters
        Parameter specification with priors.
    n_samples: int
        Number of training simulations. Default 100,000.
    key: PRNGKey or None
        Random key. If None, uses ``jax.random.key(42)``.
    obs_type: str
        ``"photometry"`` or ``"spectroscopy"``.
    noise_model: str
        ``"gaussian"`` (default) or ``"heteroscedastic"``.
    snr_range: tuple of float
        Range of S/N ratios to sample uniformly.

    Returns
    -------
    dict
        ``'theta'``: array (n_samples, n_params) -- parameter values.
        ``'x'``: array (n_samples, n_obs) -- simulated observables.
        ``'param_names'``: list of str -- parameter names.
        ``'x_type'``: str -- observation type.

    Raises
    ------
    ValueError
        If ``obs_type`` is not ``"photometry"`` or ``"spectroscopy"``.
        If ``noise_model`` is not ``"gaussian"`` or ``"heteroscedastic"``.
        If ``n_samples`` < 1.
    """
    if obs_type not in ("photometry", "spectroscopy"):
        raise ValueError(f"obs_type must be 'photometry' or 'spectroscopy', got '{obs_type}'")
    if noise_model not in ("gaussian", "heteroscedastic"):
        raise ValueError(
            f"noise_model must be 'gaussian' or 'heteroscedastic', got '{noise_model}'"
        )
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")

    if key is None:
        key = jax.random.key(42)

    # Split keys for sampling, forward model noise, and SNR
    key_prior, key_noise, key_snr = jax.random.split(key, 3)

    # Sample parameters from prior
    theta_dict = spec.sample_batch(key_prior, n_samples)

    # Extract parameter names and stack into array
    param_names = sorted(theta_dict.keys())
    theta_arrays = []
    for name in param_names:
        arr = theta_dict[name]
        if arr.ndim == 1:
            theta_arrays.append(arr[:, None])
        else:
            # Multi-dimensional params like sfh_field_xi
            theta_arrays.append(arr)
    theta = jnp.concatenate(theta_arrays, axis=1)

    # Run forward model (vectorized)
    if obs_type == "photometry":
        predict_fn = model.predict_photometry
    else:
        predict_fn = model.predict_spectrum

    # Use vmap for batched prediction; fall back to a loop for
    # non-traceable callables (e.g., mocks in tests).
    try:
        x_clean = jax.vmap(predict_fn)(theta_dict)
    except (TypeError, ValueError, RuntimeError, AttributeError):
        # TypeError: predict_fn isn't vmappable (non-traceable callable)
        # ValueError: theta_dict structure incompatible with vmap
        # RuntimeError: JAX compilation error
        # AttributeError: predict_fn doesn't behave as expected
        # Fallback: loop over samples individually
        results = []
        for i in range(n_samples):
            single = {k: v[i] for k, v in theta_dict.items()}
            results.append(predict_fn(single))
        x_clean = jnp.stack(results)

    # Sample per-observation SNR
    snr_min, snr_max = snr_range
    snr = jax.random.uniform(key_snr, shape=x_clean.shape, minval=snr_min, maxval=snr_max)

    # Add noise
    if noise_model == "gaussian":
        sigma = jnp.abs(x_clean) / snr
        noise = jax.random.normal(key_noise, shape=x_clean.shape) * sigma
        x = x_clean + noise
    else:
        # Heteroscedastic: SNR varies per band, scale with sqrt(flux)
        sigma = jnp.abs(x_clean) / snr
        # Scale sigma by sqrt of relative flux to mimic photon noise
        rel_flux = jnp.abs(x_clean) / (jnp.mean(jnp.abs(x_clean), axis=1, keepdims=True) + 1e-30)
        sigma = sigma * jnp.sqrt(jnp.maximum(rel_flux, 0.1))
        noise = jax.random.normal(key_noise, shape=x_clean.shape) * sigma
        x = x_clean + noise

    return {
        "theta": theta,
        "x": x,
        "param_names": param_names,
        "x_type": obs_type,
    }


def save_sbi_training_data(data: dict[str, Any], path: str) -> None:
    """Save SBI training data to HDF5.

    Parameters
    ----------
    data: dict
        Output from :func:`generate_sbi_training_data`.
    path: str
        Output file path (should end in ``.h5``).

    Raises
    ------
    ImportError
        If ``h5py`` is not installed.
    """
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "h5py is required to save SBI training data. Install with: pip install h5py"
        ) from exc

    filepath = Path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(str(filepath), "w") as f:
        f.create_dataset("theta", data=data["theta"])
        f.create_dataset("x", data=data["x"])
        f.attrs["param_names"] = data["param_names"]
        f.attrs["x_type"] = data["x_type"]


def load_sbi_training_data(path: str) -> dict[str, Any]:
    """Load SBI training data from HDF5.

    Parameters
    ----------
    path: str
        Path to HDF5 file saved by :func:`save_sbi_training_data`.

    Returns
    -------
    dict
        Same structure as :func:`generate_sbi_training_data` output.

    Raises
    ------
    ImportError
        If ``h5py`` is not installed.
    FileNotFoundError
        If the file does not exist.
    """
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "h5py is required to load SBI training data. Install with: pip install h5py"
        ) from exc

    filepath = Path(path)
    if not filepath.is_file():
        raise FileNotFoundError(f"SBI training data not found: {path}")

    with h5py.File(str(filepath), "r") as f:
        theta = jnp.array(f["theta"][:])
        x = jnp.array(f["x"][:])
        param_names = list(f.attrs["param_names"])
        x_type = str(f.attrs["x_type"])

    return {
        "theta": theta,
        "x": x,
        "param_names": param_names,
        "x_type": x_type,
    }


class SBIPosterior:
    """Wrapper for a pre-trained neural posterior estimator.

    Loads a trained model (from ``sbi``, ``nflows``, or ``zuko``) and
    provides ``sample()`` and ``log_prob()`` methods compatible with
    tengri's Posterior interface.

    The neural posterior is expected to be a callable object with:

    - ``sample(n, x=observation)``: draw n posterior samples
    - ``log_prob(theta, x=observation)``: evaluate log-posterior

    This class handles serialization and provides a uniform interface
    regardless of the underlying SBI framework.

    Parameters
    ----------
    posterior: object
        A trained neural posterior with ``sample`` and ``log_prob`` methods.
    param_names: list of str
        Parameter names corresponding to columns of theta.
    metadata: dict or None
        Optional metadata (training config, architecture, etc.).
    """

    def __init__(
        self,
        posterior: Any,
        param_names: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self._posterior = posterior
        self.param_names = param_names or []
        self.metadata = metadata or {}

    @classmethod
    def from_file(cls, path: str) -> SBIPosterior:
        """Load a pre-trained neural posterior from a pickle file.

        Parameters
        ----------
        path: str
            Path to the saved posterior (``.pkl``).

        Returns
        -------
        SBIPosterior
            Loaded posterior wrapper.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        """
        filepath = Path(path)
        if not filepath.is_file():
            raise FileNotFoundError(f"Pre-trained posterior not found: {path}")

        # B301: a pre-trained posterior the caller points at explicitly.
        # Unpickling executes arbitrary code, so only load files you produced
        # or otherwise trust; there is no way to validate one before it runs.
        with open(filepath, "rb") as f:
            state = pickle.load(f)  # nosec B301

        if isinstance(state, dict):
            return cls(
                posterior=state["posterior"],
                param_names=state.get("param_names"),
                metadata=state.get("metadata"),
            )
        # Assume the whole object is the posterior
        return cls(posterior=state)

    def save(self, path: str) -> None:
        """Save the posterior wrapper to a pickle file.

        Parameters
        ----------
        path: str
            Output path (``.pkl``).
        """
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "posterior": self._posterior,
            "param_names": self.param_names,
            "metadata": self.metadata,
        }
        with open(filepath, "wb") as f:
            pickle.dump(state, f)

    def sample(
        self,
        observation: jnp.ndarray,
        n_samples: int = 10_000,
    ) -> jnp.ndarray:
        """Sample from the neural posterior given an observation.

        Parameters
        ----------
        observation: array, shape (n_obs,)
            Observed data vector (photometry or spectrum).
        n_samples: int
            Number of posterior samples to draw.

        Returns
        -------
        array, shape (n_samples, n_params)
            Posterior samples.
        """
        if not hasattr(self._posterior, "sample"):
            raise AttributeError(
                "The loaded posterior does not have a 'sample' method. "
                "Ensure it is a trained neural posterior from sbi/nflows/zuko."
            )
        return self._posterior.sample(n_samples, x=observation)

    def log_prob(
        self,
        theta: jnp.ndarray,
        observation: jnp.ndarray,
    ) -> jnp.ndarray:
        """Evaluate log-posterior density.

        Parameters
        ----------
        theta: array, shape (n_samples, n_params) or (n_params,)
            Parameter values at which to evaluate.
        observation: array, shape (n_obs,)
            Observed data vector.

        Returns
        -------
        array, shape (n_samples,) or scalar
            Log-posterior density values.
        """
        if not hasattr(self._posterior, "log_prob"):
            raise AttributeError(
                "The loaded posterior does not have a 'log_prob' method. "
                "Ensure it is a trained neural posterior from sbi/nflows/zuko."
            )
        return self._posterior.log_prob(theta, x=observation)

    def summary(self) -> str:
        """Return a human-readable summary of the posterior."""
        lines = ["SBIPosterior"]
        lines.append(f"  Parameters: {len(self.param_names)}")
        if self.param_names:
            lines.append(f"  Names: {', '.join(self.param_names)}")
        if self.metadata:
            for k, v in self.metadata.items():
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)
