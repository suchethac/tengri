"""Shared inference utilities: loss functions, prior, result containers.

These are used by all inference backends (Adam, NUTS, geoVI).
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp


class InferenceResult(NamedTuple):
    """Container for inference results from any backend.

    Attributes
    ----------
    params : dict
        Best-fit or posterior mean parameters.
    samples : dict or None
        Posterior samples (None for MAP). Each value has shape (n_samples, ...).
    loss_history : array or None
        Loss/energy values over iterations.
    wall_time_s : float
        Total wall-clock time in seconds.
    method : str
        Inference method name.
    diagnostics : dict
        Method-specific diagnostics (R-hat, ESS, KL divergence, etc.).
    """

    params: dict
    samples: dict | None
    loss_history: jnp.ndarray | None
    wall_time_s: float
    method: str
    diagnostics: dict


class PriorConfig(NamedTuple):
    """Prior bounds and distributions for model parameters.

    Each entry is (lower_bound, upper_bound) for bounded parameters,
    or (mean, std) for Gaussian priors on unbounded parameters.

    Attributes
    ----------
    psd_sigma : tuple
        (lo, hi) bounds for PSD amplitude.
    psd_tau_yr : tuple
        (lo, hi) bounds for PSD timescale in years.
    alpha : tuple
        (lo, hi) bounds for falling power-law index.
    beta : tuple
        (lo, hi) bounds for rising power-law index.
    tau_sfh : tuple
        (lo, hi) bounds for SFH turnover in years.
    sfr_norm : tuple
        (lo, hi) bounds for SFR normalization.
    log_z_abs : tuple
        (lo, hi) bounds for log metallicity (absolute).
    tau_bc : tuple
        (lo, hi) bounds for birth cloud dust.
    tau_diff : tuple
        (lo, hi) bounds for diffuse dust.
    dust_slope : tuple
        (lo, hi) bounds for dust power-law slope.
    """

    psd_sigma: tuple = (0.1, 5.0)
    psd_tau_yr: tuple = (1e6, 500e6)
    alpha: tuple = (0.1, 5.0)
    beta: tuple = (0.1, 3.0)
    tau_sfh: tuple = (0.1e9, 12e9)
    sfr_norm: tuple = (0.01, 200.0)
    log_z_abs: tuple = (-2.0, 0.2)
    tau_bc: tuple = (0.0, 4.0)
    tau_diff: tuple = (0.0, 3.0)
    dust_slope: tuple = (-1.5, 0.0)


DEFAULT_PRIOR = PriorConfig()


def build_loss_fn(forward_model, data, noise, prior_config=None, data_type="photometry"):
    """Build a differentiable loss function for inference.

    loss(params) = -log_likelihood(params) - log_prior(params)

    Parameters
    ----------
    forward_model : ForwardModel
        Configured forward model instance.
    data : array
        Observed data (flux densities or spectrum).
    noise : array
        1-sigma uncertainties (same shape as data).
    prior_config : PriorConfig, optional
        Prior bounds. Defaults to DEFAULT_PRIOR.
    data_type : str
        "photometry", "spectroscopy", or "joint".

    Returns
    -------
    callable
        loss(params_unbounded) -> scalar loss value.
    """
    if prior_config is None:
        prior_config = DEFAULT_PRIOR

    from diffsed.utils.transforms import to_bounded

    # Build the bounded parameter transform
    bounds = {
        "psd_sigma": prior_config.psd_sigma,
        "psd_tau_yr": prior_config.psd_tau_yr,
        "alpha": prior_config.alpha,
        "beta": prior_config.beta,
        "tau_sfh": prior_config.tau_sfh,
        "sfr_norm": prior_config.sfr_norm,
        "log_z_abs": prior_config.log_z_abs,
        "tau_bc": prior_config.tau_bc,
        "tau_diff": prior_config.tau_diff,
        "dust_slope": prior_config.dust_slope,
    }

    def loss(params_unbounded):
        # Transform unbounded -> bounded for physical params
        params = {"xi": params_unbounded["xi"]}
        for key, (lo, hi) in bounds.items():
            params[key] = to_bounded(params_unbounded[key], lo, hi)

        # Forward model prediction
        if data_type == "photometry":
            predicted = forward_model.predict_photometry(params)
        elif data_type == "spectroscopy":
            predicted = forward_model.predict_spectrum(params, forward_model._wave_obs)
        elif data_type == "joint":
            pred_phot = forward_model.predict_photometry(params)
            pred_spec = forward_model.predict_spectrum(params, forward_model._wave_obs)
            predicted = jnp.concatenate([pred_phot, pred_spec])
        else:
            raise ValueError(f"Unknown data_type: {data_type}")

        # Gaussian log-likelihood: -0.5 * sum((d - m)^2 / sigma^2)
        chi2 = jnp.sum(((data - predicted) / noise) ** 2)

        # Standard normal prior on xi: -0.5 * xi^T xi
        xi_prior = jnp.sum(params_unbounded["xi"] ** 2)

        # Flat prior on unbounded physical params (bounded transform
        # handles the Jacobian implicitly via the sigmoid)
        return 0.5 * chi2 + 0.5 * xi_prior

    return loss


def initialize_params(key, n_grid=256, prior_config=None):
    """Initialize parameters in unbounded space.

    Parameters
    ----------
    key : PRNGKey
        Random key.
    n_grid : int
        GP grid size.
    prior_config : PriorConfig, optional
        Used to center the initial guess.

    Returns
    -------
    dict
        Initial parameter dictionary in unbounded space.
    """
    if prior_config is None:
        prior_config = DEFAULT_PRIOR

    keys = jax.random.split(key, 11)

    return {
        "xi": 0.1 * jax.random.normal(keys[0], shape=(n_grid,)),
        "psd_sigma": jax.random.normal(keys[1]) * 0.5,
        "psd_tau_yr": jax.random.normal(keys[2]) * 0.5,
        "alpha": jax.random.normal(keys[3]) * 0.3,
        "beta": jax.random.normal(keys[4]) * 0.3,
        "tau_sfh": jax.random.normal(keys[5]) * 0.3,
        "sfr_norm": jax.random.normal(keys[6]) * 0.3,
        "log_z_abs": jax.random.normal(keys[7]) * 0.3,
        "tau_bc": jax.random.normal(keys[8]) * 0.3,
        "tau_diff": jax.random.normal(keys[9]) * 0.3,
        "dust_slope": jax.random.normal(keys[10]) * 0.3,
    }


def unbounded_to_physical(params_unbounded, prior_config=None):
    """Convert unbounded parameter dict to physical values.

    Parameters
    ----------
    params_unbounded : dict
        Parameters in unbounded space.
    prior_config : PriorConfig, optional
        Bounds for the sigmoid transform.

    Returns
    -------
    dict
        Parameters in physical (bounded) space.
    """
    if prior_config is None:
        prior_config = DEFAULT_PRIOR

    from diffsed.utils.transforms import to_bounded

    bounds = {
        "psd_sigma": prior_config.psd_sigma,
        "psd_tau_yr": prior_config.psd_tau_yr,
        "alpha": prior_config.alpha,
        "beta": prior_config.beta,
        "tau_sfh": prior_config.tau_sfh,
        "sfr_norm": prior_config.sfr_norm,
        "log_z_abs": prior_config.log_z_abs,
        "tau_bc": prior_config.tau_bc,
        "tau_diff": prior_config.tau_diff,
        "dust_slope": prior_config.dust_slope,
    }

    params = {"xi": params_unbounded["xi"]}
    for key, (lo, hi) in bounds.items():
        params[key] = to_bounded(params_unbounded[key], lo, hi)
    return params
