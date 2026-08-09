# SPDX-License-Identifier: BSD-3-Clause
"""Per-galaxy interim fit driver for hierarchical PSD recovery."""

from __future__ import annotations

import gc
import logging
import time
from typing import Any, NamedTuple

import numpy as np

from tengri.inference.population.reconstruct import centered_fields

__all__ = ["InterimResult", "fit_interim"]

logger = logging.getLogger(__name__)


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
    # Defaulted fields must follow every non-default one in a NamedTuple.
    rhat_median: dict[str, Any] | None = None
    rhat_frac_above_1p01: dict[str, Any] | None = None


#: Injected-truth keys this guard knows how to check, mapped to the
#: ``interim_bounds`` entry that must contain them.
_TRUTH_TO_BOUNDS = {
    "sfh_field_psd_sigma": ("sigma_bounds", "sigma", ""),
    "sfh_field_psd_tau_myr": ("tau_bounds_myr", "tau", " Myr"),
}


def _assert_truth_within_interim_bounds(mock, interim_bounds):
    """Refuse a mock whose injected truth lies outside the interim support.

    ``make_population`` validates each truth against the *model's* prior, but
    ``fit_interim`` then overrides the shared PSD priors with
    ``interim_bounds`` — so a truth that was reachable before the override can
    be unreachable after it, and nothing re-checked (issue #1575).

    The failure this prevents is expensive and misdirects: the optimizer walks
    to a boundary that sits at infinity in unbounded space, every MAP restart
    returns a non-finite loss, and the resulting message advises tuning
    ``learning_rate``/``n_restarts`` — none of which can reach a mode outside
    the support. On the N=8 PSD pilot that arrived 50 minutes into the run.

    Parameters
    ----------
    mock : MockPopulation
        Population carrying ``truth_params``. Real data has no injected truth,
        so an absent or unrelated ``truth_params`` is skipped rather than
        rejected.
    interim_bounds : dict
        ``{"sigma_bounds": (lo, hi), "tau_bounds_myr": (lo, hi)}``, the bounds
        the fit will actually use.

    Raises
    ------
    ValueError
        If any galaxy's injected truth falls outside the matching bounds.
        Bounds are inclusive: a truth exactly on an edge is inside the support.
    """
    for galaxy, truths in enumerate(getattr(mock, "truth_params", None) or []):
        for name, (bounds_key, label, unit) in _TRUTH_TO_BOUNDS.items():
            if name not in truths or bounds_key not in interim_bounds:
                continue
            value = float(np.asarray(truths[name]))
            lo, hi = (float(b) for b in interim_bounds[bounds_key])
            if lo <= value <= hi:
                continue
            raise ValueError(
                f"Injected truth {label}={value:g}{unit} (galaxy {galaxy}) is OUTSIDE "
                f"the interim_bounds[{bounds_key!r}] = ({lo:g}, {hi:g}) this fit will "
                f"use. The truth is unreachable, so every MAP restart diverges to a "
                f"non-finite loss and no optimizer setting recovers it. Either move "
                f"the injected truth inside these bounds or widen interim_bounds to "
                f"contain it. Note make_population validated this truth against the "
                f"MODEL's prior, which interim_bounds overrides (issue #1575)."
            )


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
    thin=8,
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
    thin : int, optional
        Keep every ``thin``-th posterior draw before the population step.
        Default 8. The estimator's (n_nodes, N, K) table is what limits the
        population size; measured ESS is ~600 of 4000 draws, so thinning costs
        little and buys a linear reduction in that table.
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

    # Before anything expensive: is the truth even reachable under the bounds
    # this fit will use? Checked here rather than in make_population because
    # interim_bounds overrides the priors make_population validated against.
    _assert_truth_within_interim_bounds(mock, interim_bounds)

    n_galaxies = len(mock.table)
    log_age_grid = model.log_age_grid

    # Build modified spec with interim priors for sigma and tau
    sigma_bounds = interim_bounds["sigma_bounds"]
    tau_bounds = interim_bounds["tau_bounds_myr"]

    # Create a copy of the model's spec with overridden priors
    from tengri import Uniform

    spec_dict = {}
    for name in model.spec.free_params:
        dist = model.spec.get_distribution(name)
        if name == "sfh_field_psd_sigma":
            spec_dict[name] = Uniform(sigma_bounds[0], sigma_bounds[1])
        elif name == "sfh_field_psd_tau_myr":
            spec_dict[name] = Uniform(tau_bounds[0], tau_bounds[1])
        else:
            spec_dict[name] = dist

    from tengri import Parameters

    # ``field_centering`` rides along explicitly: this spec is rebuilt from the
    # free-parameter distributions alone, so any structural setting not named
    # here is silently reset to its default — and an interim fit that quietly
    # reverts to the non-centered map is exactly the null result #1355's A/B
    # would misread as "the knob does nothing" (#1355).
    spec_interim = Parameters(
        **spec_dict,
        n_grid=model.spec.n_grid,
        field_centering=getattr(model.spec, "field_centering", 1.0),
    )

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

        # Release the per-galaxy Posterior and Fitter before the next iteration.
        # A Posterior holds a reference to its model, and a Fitter's data_args
        # carry the threaded SSP grid (~62 MB). Holding N of those alive is how
        # two sweeps were OOM-killed at N=16 with no traceback — the kernel
        # SIGKILLs without letting Python report anything.
        del post_i, fitter
        gc.collect()

    t1 = time.time()
    wall_time = t1 - t0

    # Stack samples: (N, K, n_grid)
    xi_stacked = np.stack(all_xi, axis=0)
    sigma_stacked = np.stack(all_sigma, axis=0)
    tau_yr_stacked = np.stack(all_tau_myr, axis=0) * 1e6

    # Thin the chains before reconstructing fields.
    #
    # The estimator materializes a (n_nodes, N, K) table: at 60x60 nodes, N=12
    # and K=4000 that is 3600*12*4000*8 = 1.4 GB, and the importance weights are
    # a second copy. That is what OOM-kills the sweep, not the per-galaxy state.
    #
    # Thinning is nearly free here because the draws are autocorrelated: measured
    # ESS at the posterior mode is ~600 out of 4000, so keeping every 8th sample
    # discards mostly redundant information while cutting the table 8x. Monte
    # Carlo error scales with ESS, not with the raw sample count.
    if thin > 1:
        n_draws_before = xi_stacked.shape[1]
        xi_stacked = xi_stacked[:, ::thin, :]
        sigma_stacked = sigma_stacked[:, ::thin]
        tau_yr_stacked = tau_yr_stacked[:, ::thin]
        # Reported because it changes the returned shape: callers asserting on
        # `fields` see n_samples // thin draws, not n_samples (#1575).
        logger.info(
            "thinned by %d: %d -> %d draws per galaxy",
            thin,
            n_draws_before,
            xi_stacked.shape[1],
        )

    # Reconstruct centered fields
    fields_centered = centered_fields(
        xi_stacked,
        sigma_stacked,
        tau_yr_stacked,
        log_age_grid,
        centering=getattr(model.spec, "field_centering", 1.0),
    )

    # Times for the grid
    times_yr = 10.0 ** np.asarray(log_age_grid)

    # Aggregate ESS and R-hat
    ess_per_galaxy = np.ones(n_galaxies, dtype=float)

    # Aggregate per-galaxy R-hat three ways, not one.
    #
    # The MAX over galaxies is structurally unable to improve with N on a nested
    # galaxy set: make_population seeds galaxies with jax.random.split(key, N),
    # whose first N outputs are a prefix of the longer stream (verified: the
    # galaxy-0 and galaxy-3 keys are identical at N=4 and N=256), so galaxy
    # i is the same galaxy at every population size, and if the worst chain is
    # among the first few the max cannot move. Measured: tau's max R-hat was
    # 1.6394 at BOTH N=4 and N=8, to four decimals. That is not evidence about
    # pooling; it is the same bad galaxy reported twice.
    #
    # The median and the fraction above 1.01 can both respond to N, so they are
    # what to read when asking whether adding galaxies helps.
    all_keys = set().union(*[set(d.keys()) for d in all_rhat_dicts])

    def _vals(k):
        v = np.array([d.get(k, np.nan) for d in all_rhat_dicts], dtype=float)
        return v[np.isfinite(v)]

    rhat_combined = {
        k: float(np.nanmax(_vals(k))) if _vals(k).size else float("nan") for k in all_keys
    }
    rhat_median = {
        k: float(np.median(_vals(k))) if _vals(k).size else float("nan") for k in all_keys
    }
    rhat_frac_bad = {
        k: float(np.mean(_vals(k) > 1.01)) if _vals(k).size else float("nan") for k in all_keys
    }

    n_divergent = np.array(all_divergent_counts, dtype=int)

    result = InterimResult(
        fields=fields_centered,
        times_yr=times_yr,
        ess=ess_per_galaxy,
        rhat=rhat_combined,
        rhat_median=rhat_median,
        rhat_frac_above_1p01=rhat_frac_bad,
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
