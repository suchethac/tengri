"""Geometric Variational Inference (geoVI) via NIFTy.re.

The primary inference method for diffsed. geoVI finds a coordinate
transformation where the posterior is approximately Gaussian, then
draws samples in that transformed space. Much faster than MCMC for
high-dimensional problems (256-dim GP latent space).

The key advantage: geoVI uses the Fisher information metric to
adapt the sampling coordinates, making it naturally efficient for
the correlated field model where the posterior geometry is complex.

References:
- Frank et al. 2021, Entropy 23(7):853 (arXiv:2105.10470)
- Edenhofer et al. 2024 (arXiv:2402.16683) — NIFTy.re

Usage:
    from diffsed.inference.geovi import fit_geovi
    result = fit_geovi(model, data, noise, n_iterations=10, n_samples=6)
"""

import time

import jax
import jax.numpy as jnp

from diffsed.inference.common import (
    InferenceResult,
    PriorConfig,
    DEFAULT_PRIOR,
    initialize_params,
    unbounded_to_physical,
)


def fit_geovi(
    forward_model,
    data,
    noise,
    prior_config=None,
    data_type="photometry",
    n_iterations=10,
    n_samples=6,
    key=None,
    init_params=None,
    sample_mode="nonlinear_resample",
    verbose=True,
):
    """Fit a galaxy via geoVI (NIFTy.re).

    Parameters
    ----------
    forward_model : ForwardModel
        Configured forward model.
    data : array
        Observed data.
    noise : array
        1-sigma uncertainties.
    prior_config : PriorConfig, optional
        Prior bounds.
    data_type : str
        "photometry", "spectroscopy", or "joint".
    n_iterations : int
        Number of KL minimization iterations.
    n_samples : int
        Number of posterior samples per iteration.
        Schedule: starts with fewer, increases. Final iteration uses n_samples.
    key : PRNGKey, optional
        Random key.
    init_params : dict, optional
        Initial parameters (unbounded).
    sample_mode : str
        "nonlinear_resample" (geoVI) or "linear_resample" (MGVI).
    verbose : bool
        Print progress.

    Returns
    -------
    InferenceResult
        Posterior samples, KL divergence history, timing.
    """
    try:
        import nifty.re as jft
    except ImportError:
        raise ImportError(
            "NIFTy.re required for geoVI: pip install nifty8[re]"
        )

    if prior_config is None:
        prior_config = DEFAULT_PRIOR
    if key is None:
        key = jax.random.PRNGKey(42)

    from diffsed.utils.transforms import to_bounded

    # --- Build NIFTy.re model and likelihood ---

    # Define the bounds for sigmoid transforms
    bounds = {
        "sigma_ps": prior_config.sigma_ps,
        "tau_ps": prior_config.tau_ps,
        "alpha": prior_config.alpha,
        "beta": prior_config.beta,
        "tau_sfh": prior_config.tau_sfh,
        "sfr_norm": prior_config.sfr_norm,
        "log_z": prior_config.log_z,
        "tau_v1": prior_config.tau_v1,
        "tau_v2": prior_config.tau_v2,
        "dust_n": prior_config.dust_n,
    }

    def signal_response(primals):
        """Map standardized latent variables to predicted observables.

        This is the nonlinear forward model that NIFTy.re will differentiate
        through to compute the Fisher metric for geoVI.
        """
        # Transform unbounded -> physical
        params = {"xi": primals["xi"]}
        for param_key, (lo, hi) in bounds.items():
            params[param_key] = to_bounded(primals[param_key], lo, hi)

        # Forward model
        if data_type == "photometry":
            return forward_model.predict_photometry(params)
        elif data_type == "spectroscopy":
            return forward_model.predict_spectrum(
                params, forward_model._wave_obs
            )
        elif data_type == "joint":
            pred_phot = forward_model.predict_photometry(params)
            pred_spec = forward_model.predict_spectrum(
                params, forward_model._wave_obs
            )
            return jnp.concatenate([pred_phot, pred_spec])
        else:
            raise ValueError(f"Unknown data_type: {data_type}")

    # Build NIFTy.re model
    n_grid = forward_model.config.n_grid
    domain = {
        "xi": jft.ShapeWithDtype((n_grid,)),
    }
    for param_key in bounds:
        domain[param_key] = jft.ShapeWithDtype(())

    model = jft.Model(signal_response, domain=domain)

    # Gaussian likelihood: N^-1 = diag(1/noise^2)
    noise_cov_inv = 1.0 / noise ** 2
    likelihood = jft.Gaussian(data, noise_cov_inv).amend(model)

    # --- Initialize ---
    if init_params is None:
        key, init_key = jax.random.split(key)
        init_params = initialize_params(init_key, n_grid, prior_config)

    # --- Run KL optimization (geoVI or MGVI) ---
    if verbose:
        n_data = len(data)
        n_params = n_grid + len(bounds)
        mode = "geoVI" if sample_mode == "nonlinear_resample" else "MGVI"
        print(f"{mode}: {n_params} params, {n_data} data points, "
              f"{n_iterations} iterations")

    t0 = time.time()

    # Sample schedule: increase samples over iterations
    n_total_samples = n_samples
    delta = max(1, n_total_samples - 1)

    key, opt_key = jax.random.split(key)
    samples, state = jft.optimize_kl(
        likelihood,
        init_params,
        n_total_iterations=n_iterations,
        n_samples=lambda i: max(1, 1 + int(i * delta / max(n_iterations - 1, 1))),
        key=opt_key,
        sample_mode=sample_mode,
        odir=None,  # no disk output
        name="diffsed",
    )

    wall_time = time.time() - t0

    # --- Extract posterior samples ---
    # Convert NIFTy Samples to dict of arrays
    sample_list = list(samples)
    n_posterior = len(sample_list)

    physical_samples = {}
    for s in sample_list:
        phys = unbounded_to_physical(dict(s), prior_config)
        for k, v in phys.items():
            if k not in physical_samples:
                physical_samples[k] = []
            physical_samples[k].append(v)

    physical_samples = {
        k: jnp.stack(v) for k, v in physical_samples.items()
    }

    # Posterior mean
    best_params = {k: jnp.mean(v, axis=0) for k, v in physical_samples.items()}

    if verbose:
        mode = "geoVI" if sample_mode == "nonlinear_resample" else "MGVI"
        print(f"  {mode} complete in {wall_time:.1f}s, "
              f"{n_posterior} posterior samples")

    return InferenceResult(
        params=best_params,
        samples=physical_samples,
        loss_history=None,
        wall_time_s=wall_time,
        method=f"{'geoVI' if sample_mode == 'nonlinear_resample' else 'MGVI'} (NIFTy.re)",
        diagnostics={
            "n_iterations": n_iterations,
            "n_samples": n_posterior,
            "sample_mode": sample_mode,
        },
    )
