#!/usr/bin/env python
# SPDX-License-Identifier: BSD-3-Clause
"""Rank samplers on the quickstart's model by seconds per effective sample.

Motivated by a migration that looked obvious and was not: nb06 and nb07 went
6.3x and 3.4x moving from NUTS to fixed-length HMC, and the same swap makes the
quickstart *worse*. See
``bench/reports/2026-08-17_quickstart_nuts_vs_hmc.md``.

Two columns exist because both of the obvious ones lie here:

* **Wall time** rewards a sampler for drawing correlated samples quickly. HMC
  at L=20 is 8.8x faster than NUTS and returns 18.8 effective samples.
* **Mean ESS across parameters** hides the failure mode, which is a single
  weakly-identified direction dragging while the rest look healthy. The
  worst-mixing parameter is therefore named per row.

Wall time is also the one number this machine cannot measure reliably under
load; ESS, R-hat and the divergence count are deterministic given the seed.

Usage::

    JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/benchmark_quickstart_sampler.py
    JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/benchmark_quickstart_sampler.py --quick

    # today's quickstart rather than the one the 2026-08-17 table measured
    JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/benchmark_quickstart_sampler.py \\
        --fixture 00now

THIS FILE NO LONGER OWNS A MODEL (#2096)
========================================

It used to spell one out, and every problem below followed from that.

**It could not build.** The committed ``build_model`` passed the retired
``dust=`` peer group (split into ``dust_attenuation`` / ``dust_emission``) and
named ``law_bc`` without ``law_diff``, which #1989 made an error. Every row
raised ``ValueError`` before a single sample was drawn, so nobody could run this
script to notice anything else about it -- which is the mechanism #2096
describes: an unrunnable harness cannot report its own drift.

**It also could not reproduce its own table.** ``build_model`` had no ``met``
group, so it built **D = 6**, while
``bench/reports/2026-08-17_quickstart_nuts_vs_hmc.md`` states **D = 7** and names
``met_logzsol`` as the worst-mixing parameter of its ``hmc L=160`` row. A
parameter cannot be the worst-mixing one in a table produced by a model that
does not have it. The published numbers are a correct measurement of a D = 7
model with free metallicity; the *committed builder* is what drifted away from
them, at some point after the report was written. **The report's numbers stand
and are not restated here** -- what changed is that the fixture this file runs
is once again the one with free metallicity, so the table is reachable again.

**And it duplicated a model that had already drifted twice.** Two independent
repairs of ``benchmark_notebook_sampler.py`` in two working trees chose two
different dust laws and produced two contradictory nb05 baselines for one seed.
So the model lives in exactly one place now:
:data:`benchmark_notebook_sampler.NOTEBOOKS`, imported below. This file supplies
the sampler ladder and the table; it does not describe a galaxy.

WHICH FIXTURE
=============

``--fixture`` selects from the shared registry. The three that are quickstarts:

============  ================================================================
``00``        the **pre-#2044** quickstart -- tsnorm SFH, two-component
              Calzetti, free metallicity, D=7, 12 bands. The default, because
              this is the model ``2026-08-17_quickstart_nuts_vs_hmc.md``
              measured and comparability with it is the reason this script
              exists.
``00pre``     the same fixture in the pre-#1989 dust spelling
              (``law_bc="calzetti"`` + ``law_diff="power_law"``), which is what
              this file's own ``build_model`` literally contained.
``00now``     ``notebooks/00_quickstart.py`` **as shipped today** -- dpl SFH,
              ONE Calzetti screen, nebular baked into the wNE grid, D=6. #2044
              moved the quickstart here on 2026-08-23. No published row measures
              it; a run of this fixture is a new measurement, not a comparison.
============  ================================================================

``00`` and ``00pre`` are *deliberately* historical and
``tools/check_harness_parity.py`` knows it; ``00now`` is the one held to
today's notebook. See that file for how the distinction is enforced and what to
do when it fails.
"""

from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import warnings

warnings.filterwarnings("ignore")

import sys

import jax
import numpy as np

import tengri
from tengri import Data, ForwardModel, generate_mock
from tengri.analysis.diagnostics.autocorrelation import effective_sample_size

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_notebook_sampler import NOTEBOOKS

#: The quickstart fixtures this script offers. The registry holds others
#: (nb01, nb05, the two controls); those belong to
#: ``benchmark_notebook_sampler.py``, which has the sampler families for them.
FIXTURES = ("00", "00pre", "00now")

#: The notebook's own convergence claim, and so the bar a replacement must clear.
MAX_RHAT = 1.01
MAX_DIVERGENCES = 0


def build_model(ssp, fixture: str = "00"):
    """The quickstart's model and its ``ForwardModel``, from the shared registry.

    Kept under this name because three reports cite
    ``benchmark_quickstart_sampler.build_model`` when describing what they
    measured. It no longer *contains* a model: it looks one up, so a fix applied
    to the registry reaches every consumer instead of one working tree.
    """
    return (sed := NOTEBOOKS[fixture]["build"](ssp)), ForwardModel.build(sed=sed)


def configurations(quick: bool) -> dict[str, dict]:
    """Sampler recipes to compare, keyed by label."""
    draws = 150 if quick else 600
    nuts_draws = 150 if quick else 250
    warmup = 300 if quick else 1000
    nuts_warmup = 300 if quick else 1500
    leapfrogs = (20, 40) if quick else (20, 40, 80, 160)

    configs = {
        "nuts (shipped)": dict(
            method="mcmc_nuts",
            n_warmup=nuts_warmup,
            n_samples=nuts_draws,
            n_burnin=0,
            dense_mass_matrix=False,
            target_accept_rate=0.9,
        )
    }
    for leapfrog in leapfrogs:
        configs[f"hmc L={leapfrog}"] = dict(
            method="mcmc_hmc",
            n_warmup=warmup,
            n_samples=draws,
            n_leapfrog_steps=leapfrog,
            dense_mass_matrix=True,
            target_accept_rate=0.9,
        )
    return configs


def score(posterior, wall: float) -> dict:
    """Diagnostics that decide adoption, plus the parameter that mixes worst."""
    rhats = posterior.rhat()
    ess = effective_sample_size({k: np.asarray(v) for k, v in posterior.samples.items()})
    finite = [(k, v["ess"]) for k, v in ess.items() if np.isfinite(v["ess"])]
    worst_name, worst_ess = (
        min(finite, key=lambda pair: pair[1]) if finite else ("?", float("nan"))
    )
    return {
        "wall": wall,
        "rhat": max(float(v) for v in rhats.values()) if rhats else float("nan"),
        "divergences": int(posterior.diagnostics.get("n_divergent", 0) or 0),
        "min_ess": worst_ess,
        "worst": worst_name,
        "sec_per_ess": wall / max(worst_ess, 1e-9),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="shorter chains for a smoke run")
    parser.add_argument(
        "--fixture",
        default="00",
        choices=FIXTURES,
        help="which quickstart model (default: 00, the pre-#2044 one the published table used)",
    )
    args = parser.parse_args()

    cfg = NOTEBOOKS[args.fixture]
    ssp = tengri.load_ssp(cfg.get("ssp", "fsps_prsc_miles_chabrier"), download=True)
    sed, _ = build_model(ssp, args.fixture)

    # The fixture owns the mock, not this file: the seed, SNR and chain count
    # that produced the published table are recorded beside the model rather
    # than restated here, so the two cannot drift apart.
    key_truth, key_mock, key_fit = jax.random.split(jax.random.PRNGKey(cfg["seed"]), 3)
    mock = generate_mock(sed, sed.spec.sample(key_truth), key=key_mock, snr=cfg["snr"])
    data = Data(photometry=(np.asarray(mock["flux_obs"]), np.asarray(mock["noise"])))
    n_chains = cfg["n_chains"]

    print(f"fixture {args.fixture}: {cfg['note']}\n")
    print(
        f"D = {len(sed.spec.free_params)} free parameters, "
        f"{sed.observation.photometry.n_filters} bands, {n_chains} chains, "
        f"seed {cfg['seed']}, SNR {cfg['snr']:g}"
    )
    print(f"adoption bar: max split R-hat < {MAX_RHAT} and {MAX_DIVERGENCES} divergences\n")
    header = (
        f"{'config':<20}{'wall s':>9}{'maxRhat':>10}{'div':>5}"
        f"{'minESS':>9}{'s/ESS':>9}  worst-mixing parameter"
    )
    print(header)
    print("-" * len(header), flush=True)

    rows = {}
    for label, kwargs in configurations(args.quick).items():
        # A fresh model per row: caches are keyed on the tuning settings, but a
        # fresh build also keeps the MAP seed identical across rows.
        _, forward = build_model(ssp, args.fixture)
        seed = forward.fit(
            data, method="map", key=key_fit, n_restarts=8, n_steps=800, verbose=False
        )
        started = time.perf_counter()
        posterior = forward.fit(
            data, key=key_fit, init_from=seed, n_chains=n_chains, verbose=False, **kwargs
        )
        rows[label] = score(posterior, time.perf_counter() - started)
        row = rows[label]
        print(
            f"{label:<20}{row['wall']:>9.1f}{row['rhat']:>10.4f}{row['divergences']:>5}"
            f"{row['min_ess']:>9.1f}{row['sec_per_ess']:>9.3f}  {row['worst']}",
            flush=True,
        )

    print("\nverdict (ranked on seconds per effective sample):")
    baseline = rows.get("nuts (shipped)")
    for label, row in sorted(rows.items(), key=lambda kv: kv[1]["sec_per_ess"]):
        clears = row["rhat"] < MAX_RHAT and row["divergences"] <= MAX_DIVERGENCES
        versus = (
            f"{baseline['sec_per_ess'] / max(row['sec_per_ess'], 1e-9):5.2f}x vs nuts"
            if baseline
            else ""
        )
        print(f"  {label:<20} {'clears bar' if clears else 'MISSES bar':<11} {versus}")


if __name__ == "__main__":
    main()
