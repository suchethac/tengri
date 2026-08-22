#!/usr/bin/env python
# SPDX-License-Identifier: BSD-3-Clause
"""Rank samplers on ``00_quickstart``'s model by seconds per effective sample.

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
"""

from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import warnings

warnings.filterwarnings("ignore")

import jax
import numpy as np

import tengri
from tengri import (
    FIXED,
    FREE,
    Data,
    Fixed,
    ForwardModel,
    Observation,
    Photometry,
    SEDModel,
    Uniform,
    WavePrecomp,
    builders,
    generate_mock,
)
from tengri.analysis.diagnostics.autocorrelation import effective_sample_size

#: The quickstart's own bands: GALEX + SDSS + 2MASS + WISE, UV through near-IR.
FILTERS = (
    "galex_fuv",
    "galex_nuv",
    "sdss_u",
    "sdss_g",
    "sdss_r",
    "sdss_i",
    "sdss_z",
    "2mass_j",
    "2mass_h",
    "2mass_ks",
    "wise_w1",
    "wise_w2",
)

#: The notebook's own convergence claim, and so the bar a replacement must clear.
MAX_RHAT = 1.01
MAX_DIVERGENCES = 0


def build_model(ssp):
    """The quickstart's model: tsnorm SFH, two-component Calzetti, fixed z."""
    sed = SEDModel.build(
        ssp_data=ssp,
        observation=Observation(photometry=Photometry.from_names(list(FILTERS))),
        approx=WavePrecomp(),
        sfh=builders.sfh.tsnorm(all_params=FREE),
        dust=builders.dust.two_component(
            all_params=FIXED, law_bc="calzetti", tau_bc=Uniform(0.0, 1.0)
        ),
        neb=builders.neb.none(),
        redshift=Fixed(0.05),
    )
    return sed, ForwardModel.build(sed=sed)


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
    args = parser.parse_args()

    ssp = tengri.load_ssp("fsps_prsc_miles_chabrier", download=True)
    sed, _ = build_model(ssp)

    key_truth, key_mock, key_fit = jax.random.split(jax.random.PRNGKey(9), 3)
    mock = generate_mock(sed, sed.spec.sample(key_truth), key=key_mock, snr=30.0)
    data = Data(photometry=(np.asarray(mock["flux_obs"]), np.asarray(mock["noise"])))

    print(f"D = {len(sed.spec.free_params)} free parameters, {len(FILTERS)} bands, 4 chains")
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
        _, forward = build_model(ssp)
        seed = forward.fit(
            data, method="map", key=key_fit, n_restarts=8, n_steps=800, verbose=False
        )
        started = time.perf_counter()
        posterior = forward.fit(
            data, key=key_fit, init_from=seed, n_chains=4, verbose=False, **kwargs
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
