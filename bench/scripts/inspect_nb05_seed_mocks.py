#!/usr/bin/env python
# SPDX-License-Identifier: BSD-3-Clause
"""Is nb05's seed 0/1 hostility in the sampler or in the injected truth?

Two of `05_fitting_photometry`'s four measured seeds are hostile to every
sampler tried: NUTS returns split-R-hat 1.4e13 and 2.4e13, ChEES+precond
returns 1.239 and 1.059, and no configuration converges. The other two (2 and
7) are clean for ChEES+precond (R-hat 1.002 and 1.000 at min ESS 324.8 and
201.4).

Same model, same code, same settings -- only the mock differs. So before
concluding anything about the sampler, look at what was injected. A truth
sitting on a prior boundary is not a sampler problem: the posterior piles up
against a wall the unbounded transform sends to infinity, and every sampler
sees the same wall.

This costs one model build and four prior draws. No fit, no sampler, no GPU.

Usage::

    JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/inspect_nb05_seed_mocks.py
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import warnings

warnings.filterwarnings("ignore")

import jax
import numpy as np
from benchmark_notebook_sampler import NOTEBOOKS

import tengri
from tengri import generate_mock

#: How close to a prior edge counts as "against the wall", as a fraction of the
#: declared range. The latent space is a bounded->unbounded transform, so a
#: truth at 2% of the range is already several units out in the latent
#: coordinate the sampler actually moves in.
EDGE_FRAC = 0.05


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", default="05", choices=sorted(NOTEBOOKS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 7])
    args = parser.parse_args()

    cfg = NOTEBOOKS[args.notebook]
    ssp = tengri.load_ssp("fsps_prsc_miles_chabrier", download=True)
    sed = cfg["build"](ssp)
    names = list(sed.spec.free_params)

    print(f"notebook {args.notebook}: D = {len(names)}, SNR = {cfg['snr']}")
    print("each cell is the injected truth as a FRACTION of the parameter's declared range;")
    print(f"** marks within {100 * EDGE_FRAC:.0f}% of an edge\n")

    header = f"{'parameter':<32}" + "".join(f"{f'seed {s}':>12}" for s in args.seeds)
    print(header)
    print("-" * len(header))

    truths = {}
    for seed in args.seeds:
        key_truth, key_mock, _ = jax.random.split(jax.random.PRNGKey(seed), 3)
        truth = sed.spec.sample(key_truth)
        # Draw the mock too: it is what the seed actually produces, and a
        # pathological *noise* realization would not show up in the truth alone.
        mock = generate_mock(sed, truth, key=key_mock, snr=cfg["snr"])
        truths[seed] = (truth, mock)

    edge_counts = dict.fromkeys(args.seeds, 0)
    for name in names:
        dist = sed.spec.get_distribution(name)
        lo, hi = dist.bounds
        cells = ""
        for seed in args.seeds:
            value = float(np.asarray(truths[seed][0][name]))
            frac = (value - lo) / (hi - lo)
            at_edge = frac < EDGE_FRAC or frac > 1.0 - EDGE_FRAC
            edge_counts[seed] += at_edge
            cells += f"{f'{frac:.3f}' + ('**' if at_edge else '  '):>12}"
        print(f"{name:<32}{cells}")

    print("-" * len(header))
    print(f"{'parameters within 5% of an edge':<32}" + "".join(
        f"{edge_counts[s]:>12}" for s in args.seeds
    ))

    # A mock whose fluxes span many decades, or which carries a near-zero band,
    # is a different hazard from a truth at a prior edge: the likelihood's
    # curvature is then set by one band and the metric is badly conditioned
    # wherever the sampler is.
    spans, faint = "", ""
    for seed in args.seeds:
        flux = np.asarray(truths[seed][1]["flux_obs"])
        noise = np.asarray(truths[seed][1]["noise"])
        spans += f"{float(np.max(flux) / max(float(np.min(np.abs(flux))), 1e-30)):>12.3g}"
        faint += f"{int(np.sum(np.abs(flux) < noise)):>12}"
    print(f"\n{'flux dynamic range (max/min)':<32}{spans}")
    print(f"{'bands below 1-sigma of noise':<32}{faint}")


if __name__ == "__main__":
    main()
