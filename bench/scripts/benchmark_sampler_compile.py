#!/usr/bin/env python
# SPDX-License-Identifier: BSD-3-Clause
"""Compile cost and program size of one sampler's scan, per sampler.

``bench/reports/2026-08-30_mclmc_tuning.md`` measured that **75% of a cold NUTS
fit is XLA** -- 189.4 s cold against 46.8 s warm -- and that MCLMC's
fixed-length scan compiled **14x cheaper**, 10.4 s against 142.6 s, because
NUTS compiles a ragged tree-doubling ``while`` loop and MCLMC compiles a
straight-line step. That is a property of the *program*, not of the fit, so it
can be measured without sampling at all: lower each sampler's jitted scan,
count the StableHLO lines, and time the XLA compile.

Doing it that way is deliberate. A cold-minus-warm wall clock is contaminated
by the persistent JAX cache (``2026-08-31_catalog_preconditioning.md``
Caveat 6: two cells came out negative because "compile" was a cache load), and
this box is shared. A lowered program's line count is deterministic and load
independent.

The comparison is between programs that do the same job -- warmup plus the full
sampling chain in one jit -- so the numbers are directly comparable across
samplers even though the samplers are not.

Usage::

    JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/benchmark_sampler_compile.py
    JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/benchmark_sampler_compile.py --notebook 05
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import jax
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tengri  # noqa: E402
from benchmark_notebook_sampler import NOTEBOOKS  # noqa: E402
from tengri import Fitter  # noqa: E402
from tengri.analysis.mock import generate_mock  # noqa: E402
from tengri.inference._sample_utils import _maybe_map_init  # noqa: E402
from tengri.inference.backends.mcmc import _shared  # noqa: E402

#: Draws per program. Small and identical across samplers: the scan length is a
#: loop bound in the lowered program, not an unrolled body, so it moves the
#: line count by a constant handful and moves the *ratio* not at all.
N_CHAIN = 200

#: Warmup steps per program, likewise identical across samplers.
N_WARMUP = 200


def _lines_and_compile(fn, *args):
    """StableHLO line count and XLA compile seconds for one jitted callable."""
    lowered = fn.lower(*args)
    text = lowered.as_text()
    t0 = time.perf_counter()
    lowered.compile()
    return len(text.splitlines()), time.perf_counter() - t0, text


def main() -> None:
    """Lower each sampler's full scan on one fixture and report size and cost."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", choices=sorted(NOTEBOOKS), default="ctl-dpl")
    parser.add_argument("--json", default=None, help="write the rows here")
    args = parser.parse_args()

    cfg = NOTEBOOKS[args.notebook]
    ssp = tengri.load_ssp(cfg.get("ssp", "fsps_prsc_miles_chabrier"), download=True)
    sed = cfg["build"](ssp)
    k_truth, k_mock, k_fit = jax.random.split(jax.random.PRNGKey(cfg["seed"]), 3)
    mock = generate_mock(sed, sed.spec.sample(k_truth), key=k_mock, snr=cfg["snr"])
    fitter = Fitter(
        sed,
        np.asarray(mock["flux_obs"]),
        np.asarray(mock["noise"]),
        data_type="photometry",
    )
    init_params, _ = _maybe_map_init(fitter, k_fit, None, False)
    log_p2, _unravel, init_flat, data_args = _shared._get_flat_logdensity(fitter, init_params)
    n_dim = int(init_flat.shape[0])
    keys = jax.random.split(jax.random.PRNGKey(0), N_CHAIN)
    wkey = jax.random.PRNGKey(1)

    cases = {
        "nuts (max_doublings=10)": (
            _shared._nuts_full_scan,
            (init_flat, wkey, keys, log_p2, data_args, N_WARMUP, 10, False, 0.8, False),
        ),
        "hmc L=10 diag": (
            _shared._hmc_full_scan,
            (init_flat, wkey, keys, log_p2, data_args, N_WARMUP, 10, False, 0.85),
        ),
        "hmc L=10 lowrank": (
            _shared._hmc_low_rank_full_scan,
            (init_flat, wkey, keys, log_p2, data_args, N_WARMUP, 10, 10, 0.85),
        ),
        "barker": (
            _shared._first_order_full_scan,
            (init_flat, wkey, keys, log_p2, data_args, N_WARMUP, "barker", 0.574),
        ),
        "mala": (
            _shared._first_order_full_scan,
            (init_flat, wkey, keys, log_p2, data_args, N_WARMUP, "mala", 0.574),
        ),
    }

    print(f"fixture {args.notebook}: D = {n_dim}, {N_WARMUP} warmup + {N_CHAIN} draws")
    header = f"{'sampler':<26}{'HLO lines':>11}{'while loops':>13}{'compile s':>11}"
    print(header)
    print("-" * len(header))
    rows = []
    for label, (fn, call_args) in cases.items():
        try:
            n_lines, t_compile, text = _lines_and_compile(fn, *call_args)
        except Exception as exc:  # noqa: BLE001 - a bench reports, it does not raise
            print(f"{label:<26}  FAILED: {type(exc).__name__}: {str(exc)[:120]}")
            continue
        # The structural column. A ``while`` in the lowered program is a
        # data-dependent trip count: the thing that makes a vmapped batch run
        # at the speed of its slowest lane, and the thing XLA spends its
        # compile budget on.
        n_while = text.count("stablehlo.while")
        rows.append(
            {
                "sampler": label,
                "hlo_lines": n_lines,
                "while_loops": n_while,
                "compile_s": t_compile,
                "n_dim": n_dim,
            }
        )
        print(f"{label:<26}{n_lines:>11}{n_while:>13}{t_compile:>11.2f}")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"notebook": args.notebook, "rows": rows}, fh, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
