#!/usr/bin/env python
# SPDX-License-Identifier: BSD-3-Clause
"""Program size and XLA compile cost per sampler, on a target with no model in it.

Companion to ``benchmark_sampler_compile.py``, which measures the same thing on
a real tengri fixture. This one lowers the identical scans against a plain
anisotropic Gaussian, so the numbers are a property of the **sampler's control
flow** and not of the forward model -- which compiles its own loops
(CLAUDE.md's ``age_kernel`` note counts 6-14 of them) and would otherwise
dominate the comparison.

**Why it exists, and the mistake it corrects.** The first version of the
contract test beside it asserted that Barker's lowered program contains *zero*
``stablehlo.while``. That is false and the assertion caught it: ``lax.scan``
itself lowers to a ``stablehlo.while`` with a constant trip count, so the
count separates nothing. "Branch-free" is a real property, but the number of
``while`` ops is a **proxy** for it and a bad one. What is not a proxy is the
size of the program XLA has to compile and how long it takes to do it -- which
is the quantity ``bench/reports/2026-08-30_mclmc_tuning.md`` actually measured
(10.4 s against 142.6 s, 14x) and the quantity the speed claim rests on.

Usage::

    JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/benchmark_sampler_program_size.py
"""

from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp

from tengri.inference.backends.mcmc import _shared

#: Dimension of the toy target. The lowered control flow does not depend on it.
N_DIM = 4

#: Scan lengths. Loop *bounds* in the lowered program, not unrolled bodies.
N_WARMUP = 20
N_CHAIN = 20


def _target(position, data_args):
    """Anisotropic Gaussian log-density, in the ``(position, data_args)`` shape."""
    return -0.5 * jnp.sum((position / data_args) ** 2)


def main() -> None:
    """Lower each sampler's full scan and report size, while count and compile."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", default=None, help="write the rows here")
    args = parser.parse_args()

    scales = jnp.asarray([10.0**i for i in range(N_DIM)])
    init = jnp.zeros(N_DIM)
    wkey = jax.random.PRNGKey(0)
    keys = jax.random.split(jax.random.PRNGKey(1), N_CHAIN)

    cases = {
        "nuts (max_doublings=10)": (
            _shared._nuts_full_scan,
            (init, wkey, keys, _target, scales, N_WARMUP, 10, False, 0.8, False),
        ),
        "hmc L=10 diag": (
            _shared._hmc_full_scan,
            (init, wkey, keys, _target, scales, N_WARMUP, 10, False, 0.85),
        ),
        # Warmup only; the sampling half is ``_hmc_chain_scan``, shared with
        # ``mcmc_hmc``, so it is already priced by that row.
        "hmc L=10 lowrank (warmup)": (
            _shared._hmc_low_rank_warmup_only,
            (init, wkey, _target, scales, N_WARMUP, 10, 2, 0.85),
        ),
        "barker": (
            _shared._first_order_full_scan,
            (init, wkey, keys, _target, scales, N_WARMUP, "barker", 0.574),
        ),
        "mala": (
            _shared._first_order_full_scan,
            (init, wkey, keys, _target, scales, N_WARMUP, "mala", 0.574),
        ),
    }

    print(f"toy anisotropic Gaussian, D = {N_DIM}, cond = {float(scales[-1] / scales[0]) ** 2:g}")
    header = f"{'sampler':<26}{'HLO lines':>11}{'while ops':>11}{'compile s':>11}"
    print(header)
    print("-" * len(header))
    rows = []
    for label, (fn, call_args) in cases.items():
        lowered = fn.lower(*call_args)
        text = lowered.as_text()
        t0 = time.perf_counter()
        lowered.compile()
        compile_s = time.perf_counter() - t0
        row = {
            "sampler": label,
            "hlo_lines": len(text.splitlines()),
            "while_ops": text.count("stablehlo.while"),
            "compile_s": compile_s,
        }
        rows.append(row)
        print(f"{label:<26}{row['hlo_lines']:>11}{row['while_ops']:>11}{compile_s:>11.2f}")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"n_dim": N_DIM, "rows": rows}, fh, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
