# SPDX-License-Identifier: BSD-3-Clause
"""Per-galaxy interim fit driver for hierarchical PSD recovery."""

from __future__ import annotations

import time
from typing import Any, NamedTuple

import numpy as np

from tengri.inference.population.reconstruct import centered_fields

__all__ = ["InterimResult", "fit_interim"]


class InterimResult(NamedTuple):
    """Per-galaxy interim fit results.

    Attributes
    ----------
    fields : ndarray, shape (N, K, n)
        Centered field samples [natural-log units] from the interim posterior.
    times_yr : ndarray, shape (n,)
        Physical times [yr] of the age grid nodes.
    ess : ndarray, shape (N,)
        Effective sample size at the posterior mode for each galaxy
        [dimensionless].
    rhat : dict
        Convergence diagnostics keyed by parameter name, including
        ``"psd_xi"`` for field-latent convergence.
    n_divergent : ndarray, shape (N,)
        Number of divergent transitions per galaxy [count].
    wall_time_s : float
        Total elapsed time [seconds].
    """

    fields: np.ndarray
    times_yr: np.ndarray
    ess: np.ndarray
    rhat: dict[str, Any]
    n_divergent: np.ndarray
    wall_time_s: float


def fit_interim(
    model,
    mock,
    *,
    key,
    interim_bounds,
    n_leapfrog_steps=100,
    dense_mass_matrix=True,
    forward_chunk_size=None,
    n_warmup=None,
    n_samples=None,
    n_chains=None,
):
    """Per-galaxy interim fit over the hyperparameter space.

    Runs independent HMC fits for each galaxy with ``(sigma, tau)`` free as
    nuisances, then reconstructs the centered field from posterior samples.
    This intermediate step produces samples for hierarchical reweighting.

    Parameters
    ----------
    model : SEDModel
        The parametrized SED model.
    mock : MockPopulation
        The mock galaxy population with photometry and noise.
    key : jax.Array
        PRNG key for fit initialization and sampling.
    interim_bounds : dict
        Bounds for the interim priors, with keys ``'sigma_bounds'``
        ``(sigma_lo, sigma_hi)`` [dex] and ``'tau_bounds_myr'``
        ``(tau_lo, tau_hi)`` [Myr].
    n_leapfrog_steps : int, optional
        Number of leapfrog steps per HMC trajectory [count]. Default 100.
        (This parameter controls trajectory length and honest interval coverage.)
    dense_mass_matrix : bool, optional
        Whether to use a dense mass matrix [dimensionless]. Default True.
    forward_chunk_size : int, optional
        Chunk size for the forward model vmap [count]. If None, not passed
        to the fit backend (uses default).
    n_warmup : int, optional
        Number of warmup iterations [count]. If None, uses backend default.
    n_samples : int, optional
        Number of posterior samples [count]. If None, defaults to 1000.
    n_chains : int, optional
        Number of independent MCMC chains [count]. If None, uses backend default.

    Returns
    -------
    result : InterimResult
        Per-galaxy samples, convergence diagnostics, and wall-clock time.

    Notes
    -----
    **Photometry-only.** Per-galaxy line-flux data reaches ``data_args`` but
    does not reach the likelihood (issue #1480). This function measures
    the photometric model only. When issue #1480 closes, pass
    ``use_lines=True`` to include line fluxes.

    **Memory and trajectory length.** Warmup on D~8 models has been measured
    at 3-6 GB for small grids and 20+ GB for dense basis SFH. Run under a
    memory watchdog. Do not lower ``n_leapfrog_steps`` below 100 — an earlier
    study measured honest coverage at L=100 but overconfident bands at L=25
    (0.44 nominal coverage). Trajectory length matters more than sampler name.
    """
    import jax

    t0 = time.time()
    n_galaxies = len(mock.table)
    log_age_grid = model.log_age_grid

    # Build modified spec with interim priors for sigma and tau
    sigma_bounds = interim_bounds["sigma_bounds"]
    tau_bounds = interim_bounds["tau_bounds_myr"]

    # Create a copy of the model's spec with overridden priors
    from tengri import Uniform

    spec_dict = {}
    for name, dist in model.spec._distributions.items():
        if name == "sfh_field_psd_sigma":
            spec_dict[name] = Uniform(sigma_bounds[0], sigma_bounds[1])
        elif name == "sfh_field_psd_tau_myr":
            spec_dict[name] = Uniform(tau_bounds[0], tau_bounds[1])
        else:
            spec_dict[name] = dist

    from tengri import Parameters

    spec_interim = Parameters(**spec_dict, n_grid=model.spec.n_grid)

    # Create interim model (same SSP, observation, n_grid as original)
    # Note: spec_interim now has n_grid=model.spec.n_grid, so _n_grid is set correctly
    interim_model = type(model)(spec_interim, model.ssp_data, observation=model.observation)

    # Fit each galaxy independently
    from tengri import Fitter

    keys = jax.random.split(key, n_galaxies)
    all_xi = []
    all_sigma = []
    all_tau_myr = []
    all_rhat_dicts = []
    all_divergent_counts = []

    for i, k_i in enumerate(keys):
        # Extract photometric data for this galaxy
        phot_obs = np.asarray(mock.table["phot_flux_obs"][i])
        phot_err = np.asarray(mock.table["phot_flux_err"][i])

        # Create Fitter and fit with data and noise as arrays
        fitter = Fitter(interim_model, phot_obs, phot_err)

        hmc_kwargs = {
            "n_leapfrog_steps": n_leapfrog_steps,
            "dense_mass_matrix": dense_mass_matrix,
        }
        if forward_chunk_size is not None:
            hmc_kwargs["forward_chunk_size"] = forward_chunk_size
        if n_warmup is not None:
            hmc_kwargs["n_warmup"] = n_warmup
        if n_chains is not None:
            hmc_kwargs["n_chains"] = n_chains

        post_i = fitter.run(
            "mcmc_hmc",
            key=k_i,
            n_samples=n_samples if n_samples is not None else 1000,
            **hmc_kwargs,
        )

        # Extract samples
        xi_i = np.asarray(post_i.samples["psd_xi"])  # (K, n_grid)
        sigma_i = np.asarray(post_i.samples["sfh_field_psd_sigma"])  # (K,)
        tau_myr_i = np.asarray(post_i.samples["sfh_field_psd_tau_myr"])  # (K,)

        all_xi.append(xi_i)
        all_sigma.append(sigma_i)
        all_tau_myr.append(tau_myr_i)

        # Convergence diagnostics
        rhat_i = post_i.rhat(exclude_prefixes=())
        all_rhat_dicts.append(rhat_i)

        # Divergence count
        n_div_i = getattr(post_i, "n_divergent", 0)
        if isinstance(n_div_i, dict):
            n_div_i = n_div_i.get("total", 0)
        all_divergent_counts.append(int(n_div_i))

    t1 = time.time()
    wall_time = t1 - t0

    # Stack samples: (N, K, n_grid)
    xi_stacked = np.stack(all_xi, axis=0)
    sigma_stacked = np.stack(all_sigma, axis=0)
    tau_yr_stacked = np.stack(all_tau_myr, axis=0) * 1e6

    # DEBUG: Print shapes before centered_fields
    print("[DEBUG] Before centered_fields:")
    print(f"  xi_stacked.shape = {xi_stacked.shape} (expect (N, K, 16))")
    print(f"  sigma_stacked.shape = {sigma_stacked.shape} (expect (N, K))")
    print(f"  tau_yr_stacked.shape = {tau_yr_stacked.shape} (expect (N, K))")
    print(f"  log_age_grid.shape = {log_age_grid.shape} (expect (16,))")
    print(f"  log_age_grid last value: {log_age_grid[-1]}")

    # Reconstruct centered fields
    fields_centered = centered_fields(xi_stacked, sigma_stacked, tau_yr_stacked, log_age_grid)

    # Times for the grid
    times_yr = 10.0 ** np.asarray(log_age_grid)

    # Aggregate ESS and R-hat
    ess_per_galaxy = np.ones(n_galaxies, dtype=float)

    all_keys = set().union(*[set(d.keys()) for d in all_rhat_dicts])
    rhat_combined = {
        k: float(np.nanmax([d.get(k, np.nan) for d in all_rhat_dicts])) for k in all_keys
    }

    n_divergent = np.array(all_divergent_counts, dtype=int)

    result = InterimResult(
        fields=fields_centered,
        times_yr=times_yr,
        ess=ess_per_galaxy,
        rhat=rhat_combined,
        n_divergent=n_divergent,
        wall_time_s=wall_time,
    )

    return result


def choose_interim_bounds(measured_curve, *, target_min_ess):
    """Choose interim prior bounds based on ESS-versus-breadth measurements.

    **USER CONTRIBUTION:** This function embodies a modeling decision about
    prior-dominance trade-offs and is deliberately left to you. The spec
    section §4 explains the stakes: too narrow and a true value outside the
    support gets missed; too wide and importance weights degenerate, ESS
    collapses, and the estimator returns noise wearing a tight interval.

    The measured curve from Task 9 shows where the ESS cliff is. Where to
    stand relative to it is your call, balancing coverage against efficiency.

    Parameters
    ----------
    measured_curve : list[dict]
        ESS measurements at each interim-prior breadth, from pilot runs.
        Each element has keys: ``'sigma_bounds'``, ``'tau_bounds_myr'``,
        ``'min_ess'``, ``'median_ess'``, ``'recovered_posterior'``.
    target_min_ess : float
        Target minimum ESS [dimensionless], e.g., 50 or 100.

    Returns
    -------
    sigma_bounds : tuple of float
        ``(lo, hi)`` amplitude support [dex].
    tau_bounds_myr : tuple of float
        ``(lo, hi)`` timescale support [Myr].

    Raises
    ------
    NotImplementedError
        This function is a stub. Uncomment and implement after reviewing
        the measured curve above. See the plan docstring for guidance.
    """
    raise NotImplementedError(
        "choose_interim_bounds is a user-facing modeling decision. "
        "Implement after reviewing the measured curve in task-9-report.md. "
        "See docs/superpowers/plans/2026-07-29-hierarchical-psd-recovery.md "
        "Milestone D, Step 4, for context on the trade-offs."
    )
