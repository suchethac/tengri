#!/usr/bin/env python
# SPDX-License-Identifier: BSD-3-Clause
"""Ten-second smoke of the samplers added by the blackjax survey.

Not a benchmark. This runs each new backend once on the smallest fixture the
harness has, with a draw budget too small to say anything about mixing, and
asserts only that the program builds, runs, returns finite draws, and reports
the diagnostics its report will read. It exists because
``tests/inference/test_backend_conformance.py`` is auto-marked ``slow`` and so
never runs in the default suite -- CLAUDE.md's own warning that "adding a
backend and seeing green tells you nothing".

Usage::

    JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/smoke_new_backends.py
"""

from __future__ import annotations

import os
import sys

import jax
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tengri  # noqa: E402
from benchmark_notebook_sampler import NOTEBOOKS  # noqa: E402
from tengri import Data, ForwardModel  # noqa: E402
from tengri.analysis.mock import generate_mock  # noqa: E402

CASES = [
    ("mcmc_barker", dict(n_warmup=60, n_samples=200, precondition=None)),
    ("mcmc_barker", dict(n_warmup=60, n_samples=200, precondition=True)),
    ("mcmc_mala", dict(n_warmup=60, n_samples=200, precondition=None)),
    ("mcmc_hmc_lowrank", dict(n_warmup=120, n_samples=60, n_leapfrog_steps=5)),
    (
        "mcmc_hmc_lowrank",
        dict(n_warmup=120, n_samples=60, n_leapfrog_steps=5, precondition=True),
    ),
]


def main() -> None:
    """Run each case once and print its shape, R-hat, min ESS and diagnostics."""
    cfg = NOTEBOOKS["ctl-dpl"]
    ssp = tengri.load_ssp(cfg.get("ssp", "fsps_prsc_miles_chabrier"), download=True)
    sed = cfg["build"](ssp)
    k_truth, k_mock, k_fit = jax.random.split(jax.random.PRNGKey(cfg["seed"]), 3)
    mock = generate_mock(sed, sed.spec.sample(k_truth), key=k_mock, snr=cfg["snr"])
    data = Data(photometry=(np.asarray(mock["flux_obs"]), np.asarray(mock["noise"])))

    failures = 0
    for method, kwargs in CASES:
        forward = ForwardModel.build(sed=cfg["build"](ssp))
        seed_fit = forward.fit(
            data, method="map", key=k_fit, n_restarts=2, n_steps=200, verbose=False
        )
        try:
            post = forward.fit(
                data,
                key=k_fit,
                init_from=seed_fit,
                method=method,
                n_chains=2,
                verbose=False,
                **kwargs,
            )
        except Exception as exc:  # noqa: BLE001 - a smoke reports, it does not raise
            failures += 1
            print(f"{method} {kwargs}: FAILED -- {type(exc).__name__}: {exc}")
            continue
        arrays = {k: np.asarray(v) for k, v in post.samples.items()}
        finite = all(np.all(np.isfinite(v)) for v in arrays.values())
        n_draws = next(iter(arrays.values())).shape[0]
        rhat = max(post.rhat().values())
        diag = post.diagnostics or {}
        print(
            f"{method:<18} precond={kwargs.get('precondition')!s:<5} "
            f"draws={n_draws:<6} finite={finite} rhat={rhat:.4f} "
            f"wall={post.wall_time_s:.1f}s "
            f"step={diag.get('step_size'):.4g} "
            f"acc={diag.get('acceptance_rate')} div={diag.get('n_divergent')} "
            f"grad/draw={diag.get('n_gradients_per_draw') or diag.get('n_leapfrog_steps')}"
        )
        if not finite:
            failures += 1

    print(f"\n{len(CASES) - failures}/{len(CASES)} cases ran and returned finite draws")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
