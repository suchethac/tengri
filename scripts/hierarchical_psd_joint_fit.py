# SPDX-License-Identifier: BSD-3-Clause
"""Joint hierarchical fit of the shared PSD block, as a control on the two-step.

The two-step estimator fits each galaxy under a wide interim prior and then
reweights the draws onto a ``(sigma, tau)`` grid. Its measured failure is a
per-galaxy tilt of +0.098 nats toward the grid corner, traced to the importance
correction dividing by the grid-averaged pushforward ``p_0`` while the draws
actually came from a per-galaxy Laplace Gaussian.

A joint fit has no interim prior, no importance weights and no ``p_0``, so that
mechanism cannot occur. It is therefore the control that separates "the
estimator is biased" from "the observable does not identify tau".

Everything else is held fixed to the bank: same ``build_model``, same truth,
same ``PRNGKey(0)`` population stream, so the first N galaxies here ARE the
first N galaxies of ``psd_bank_fixed``.

Run::

  PYTHONPATH=<worktree>/src:. JAX_PLATFORMS=cpu \\
    python scripts/hierarchical_psd_joint_fit.py --n 4 --method mcmc_nuts
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import time

import jax
import numpy as np
from scripts.hierarchical_psd_recovery_run import (
    INTERIM_SIGMA_BOUNDS,
    INTERIM_TAU_BOUNDS_MYR,
    SNR_LINE,
    SNR_PHOT,
    TRUTH_SIGMA,
    TRUTH_TAU_MYR,
    build_model,
)

from tengri import Parameters, Uniform, load_ssp_data
from tengri.analysis.population_mocks import make_population

SSP_PATH = "data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
_SHARED = ("sfh_field_psd_sigma", "sfh_field_psd_tau_myr")


def _peak_rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def _retuned_model(model):
    """Same model with the shared PSD block on the interim prior bounds.

    The joint fit is compared against a grid spanning the interim bounds, so
    the shared prior must span the same range or the comparison confounds a
    prior difference with an estimator difference.
    """
    spec_dict = {}
    for name in model.spec.free_params:
        if name == "sfh_field_psd_sigma":
            spec_dict[name] = Uniform(*INTERIM_SIGMA_BOUNDS)
        elif name == "sfh_field_psd_tau_myr":
            spec_dict[name] = Uniform(*INTERIM_TAU_BOUNDS_MYR)
        else:
            spec_dict[name] = model.spec.get_distribution(name)
    spec = Parameters(**spec_dict, n_grid=model.spec.n_grid)
    return type(model)(spec, model.ssp_data, observation=model.observation)


def build_joint(n_galaxies):
    """Build the joint hierarchical forward model over N mock galaxies.

    Returns
    -------
    forward : ForwardModel
        Hierarchical forward wrapping a ``PopulationSEDModel``.
    mock : MockPopulation
        The generated population, for truth bookkeeping.
    """
    from tengri.forward.forward_model import ForwardModel
    from tengri.forward.population_sed_model import PopulationSEDModel

    ssp = load_ssp_data(SSP_PATH)
    model = _retuned_model(build_model(ssp))

    # PRNGKey(0) and the same n_galaxies as the bank: make_population consumes a
    # prefix-stable split, so galaxy i here is galaxy i there.
    mock = make_population(
        model,
        n_galaxies=n_galaxies,
        sigma_true=TRUTH_SIGMA,
        tau_true_myr=TRUTH_TAU_MYR,
        key=jax.random.PRNGKey(0),
        snr_phot=SNR_PHOT,
        snr_line=SNR_LINE,
    )

    galaxies = [
        {
            "flux_obs": np.asarray(mock.table[i]["phot_flux_obs"]),
            "noise": np.asarray(mock.table[i]["phot_flux_err"]),
        }
        for i in range(n_galaxies)
    ]
    pop = PopulationSEDModel(
        sed=model,
        galaxies=galaxies,
        shared=_SHARED,
        priors={
            "sfh_field_psd_sigma": INTERIM_SIGMA_BOUNDS,
            "sfh_field_psd_tau_myr": INTERIM_TAU_BOUNDS_MYR,
        },
    )
    forward = ForwardModel.build(population=pop, observation=model.observation)
    return forward, mock


def summarize(samples, name, truth):
    """Print and return the 68% interval of one shared parameter."""
    arr = np.asarray(samples).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"name": name, "truth": truth, "n_finite": 0}
    lo, med, hi = np.percentile(arr, [16.0, 50.0, 84.0])
    covers = bool(lo <= truth <= hi)
    print(
        f"  {name:24s} {lo:9.3f} - {hi:9.3f}  (median {med:8.3f}, "
        f"truth {truth:7.2f})  {'COVERS' if covers else 'misses'}"
    )
    return {
        "name": name,
        "truth": truth,
        "lower": float(lo),
        "median": float(med),
        "upper": float(hi),
        "covers": covers,
        "n_finite": int(arr.size),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4, help="population size [galaxies]")
    ap.add_argument("--method", default="mcmc_nuts")
    ap.add_argument("--n-warmup", type=int, default=500)
    ap.add_argument("--n-samples", type=int, default=500)
    ap.add_argument("--n-chains", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--probe", action="store_true", help="build and evaluate the loss only")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import tengri

    print("tengri:", tengri.__file__, flush=True)

    t0 = time.time()
    forward, _mock = build_joint(args.n)
    print(f"built joint forward over N={args.n} in {time.time() - t0:.1f}s", flush=True)

    from tengri.inference.fitter import Fitter

    fitter = Fitter(forward)
    print(f"  data {tuple(fitter.data.shape)}  noise {tuple(fitter.noise.shape)}")
    free = list(fitter.spec.free_params)
    print(f"  D = {len(free)} free params; shared present: {[s for s in _SHARED if s in free]}")

    if args.probe:
        # Evaluate the joint objective once. A hierarchical forward that
        # constructs but cannot produce a finite scalar is the failure this
        # catches before an hour of sampling does.
        import jax.numpy as jnp

        from tengri.inference.context import InferenceContext

        ctx = InferenceContext.from_target(fitter)
        key = jax.random.PRNGKey(args.seed)
        init = ctx.initial_params(key)
        shapes = {k: tuple(np.shape(v)) for k, v in init.items()}
        print(f"  init shapes: {shapes}")
        print(f"  total latent dimension D = {sum(int(np.prod(s)) for s in shapes.values())}")
        # data_args is a dict passed as ONE positional arg, not splatted:
        # ``*ctx.data_args`` unpacks it into its string keys and JAX then
        # reports "cannot interpret <class 'str'> as an abstract array".
        val = ctx.neg_log_posterior_fn(init, ctx.data_args)
        print(f"  neg_log_posterior at init = {float(val):.4f} finite={bool(jnp.isfinite(val))}")
        return

    t0 = time.time()
    # dense_mass_matrix=False is not a tuning preference: a dense metric at
    # D = 2 + N*(8 + n_grid) is a D^2 matrix whose NUTS warmup is documented in
    # CLAUDE.md to peak past 20 GB by D ~ 8, and a 15 GB total-RSS guard
    # SIGKILLs python on this machine.
    run_kw = dict(
        key=jax.random.PRNGKey(args.seed),
        n_warmup=args.n_warmup,
        n_samples=args.n_samples,
        n_chains=args.n_chains,
    )
    if args.method.startswith("mcmc_"):
        run_kw["dense_mass_matrix"] = False
    result = fitter.run(args.method, **run_kw)
    wall = time.time() - t0
    print(f"\n{args.method} on N={args.n}: {wall:.1f}s, peak RSS {_peak_rss_gb():.2f} GB")

    rows = []
    print("\nshared block, 68% credible intervals:")
    for name, truth in zip(_SHARED, (TRUTH_SIGMA, TRUTH_TAU_MYR)):
        draws = result.samples.get(name)
        if draws is None:
            print(f"  {name}: ABSENT from posterior samples")
            continue
        rows.append(summarize(draws, name, truth))

    try:
        rhat = result.rhat
        print(f"\nR-hat: {rhat}")
    except Exception as exc:
        print(f"\nR-hat unavailable: {exc}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(
                {
                    "n": args.n,
                    "method": args.method,
                    "wall_s": wall,
                    "peak_rss_gb": _peak_rss_gb(),
                    "n_warmup": args.n_warmup,
                    "n_samples": args.n_samples,
                    "n_chains": args.n_chains,
                    "shared": rows,
                },
                fh,
                indent=2,
            )
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
