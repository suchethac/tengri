#!/usr/bin/env python
# SPDX-License-Identifier: BSD-3-Clause
"""Is the autocorrelation ESS measuring anything on an SMC particle population?

Every table in ``bench/`` carries a ``min ESS`` column, every reader knows what
it means there, and for tempered SMC it means something else. This script is the
measurement that establishes it, on a target whose posterior is known in closed
form so "the population is healthy" is checked rather than assumed.

Three numbers, on the **same particles**:

* the autocorrelation ESS in the order the sampler returns them,
* the autocorrelation ESS after a within-population permutation -- which changes
  nothing at all about the sample and everything about a time-series estimator,
* the ancestor ESS, ``N^2 / sum(c_i^2)`` over resampling multiplicities.

A permutation that moves the first number is proof that the first number is
reading particle *order*. Two things put order in it: systematic resampling
leaves copies of a surviving particle adjacent, and the returned draws are the
populations concatenated, so a disagreement between populations enters as a step
in the middle of the series rather than as noise.

Usage::

    JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/diagnose_smc_particle_ess.py
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from tengri.analysis.diagnostics.autocorrelation import effective_sample_size, rhat
from tengri.inference.backends.mcmc._shared import _SMC_MAX_TEMPERATURES, _smc_scan
from tengri.inference.backends.mcmc.smc import SMC_TARGET_ACCEPT_RATE


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dim", type=int, default=4)
    ap.add_argument("--particles", type=int, default=512)
    ap.add_argument("--populations", type=int, default=2)
    ap.add_argument("--scale", type=float, default=0.3)
    ap.add_argument("--shift", type=float, default=1.5)
    ap.add_argument("--mcmc-steps", type=int, default=2)
    ap.add_argument("--leapfrog", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    mean = jnp.full((args.dim,), args.shift)

    def logprior(position, data_args):
        """The standardized N(0, I) prior; ``data_args`` unused, as in the backend."""
        del data_args
        return -0.5 * jnp.sum(position**2)

    def loglik(position, data_args):
        """A Gaussian data term offset from the prior, so tempering has work to do."""
        del data_args
        return -0.5 * jnp.sum(((position - mean) / args.scale) ** 2)

    particles, _log_z, n_temp, _lam, _ss, _ndiv, _acc, anc = _smc_scan(
        jnp.eye(args.dim),
        jax.random.split(jax.random.PRNGKey(args.seed), args.populations),
        (),
        logprior,
        loglik,
        args.particles,
        args.mcmc_steps,
        args.leapfrog,
        0.5,
        0.3,
        0.5,
        SMC_TARGET_ACCEPT_RATE,
        _SMC_MAX_TEMPERATURES,
        None,
    )
    draws = np.asarray(particles).reshape(-1, args.dim)
    var = args.scale**2 / (1.0 + args.scale**2)

    print(
        f"rungs {[int(v) for v in n_temp]}   ancestor ESS {float(jnp.min(anc)):.1f} "
        f"of {args.particles}"
    )
    print(
        f"posterior mean  got {draws.mean(0).round(4)}  "
        f"analytic {args.shift * var / args.scale**2:.4f}"
    )
    print(f"posterior sd    got {draws.std(0).round(4)}  analytic {np.sqrt(var):.4f}")
    named = {f"x{i}": draws[:, i] for i in range(args.dim)}
    print(f"max split R-hat {max(rhat(named).values()):.4f}")

    as_returned = effective_sample_size(named)
    rng = np.random.default_rng(args.seed)
    perm = np.concatenate(
        [i * args.particles + rng.permutation(args.particles) for i in range(args.populations)]
    )
    shuffled = effective_sample_size({k: v[perm] for k, v in named.items()})

    print(f"\n{'param':<8}{'ESS as returned':>18}{'ESS after shuffle':>20}{'ratio':>8}")
    for i in range(args.dim):
        k = f"x{i}"
        a, b = as_returned[k]["ess"], shuffled[k]["ess"]
        print(f"{k:<8}{a:>18.1f}{b:>20.1f}{b / max(a, 1e-9):>8.2f}")
    print(f"\nunique particle fraction {len(np.unique(draws, axis=0)) / draws.shape[0]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
