#!/usr/bin/env python
# SPDX-License-Identifier: BSD-3-Clause
"""Rank samplers on a published notebook's own model, by seconds per effective sample.

Generalizes ``benchmark_quickstart_sampler.py`` to the other two notebooks that
never followed nb06/nb07 from NUTS to fixed-length HMC. That earlier migration
bought 6.3x and 3.4x; ``bench/reports/2026-08-17_quickstart_nuts_vs_hmc.md``
measured the same swap on ``00_quickstart`` and found it **worse**, and closed
with the warning that ``01_why_jax`` and ``05_fitting_photometry`` were *not*
measured and must not be assumed to follow. This script measures them.

Two columns exist because both of the obvious ones lie:

* **Wall time** rewards a sampler for drawing correlated samples quickly. On
  the quickstart, HMC at L=20 was 8.8x faster than NUTS and returned 18.8
  effective samples.
* **Mean ESS across parameters** hides the failure mode, which is a single
  weakly-identified direction dragging while the rest look healthy. The
  worst-mixing parameter is therefore named per row.

Wall time is the one number this machine cannot measure reliably under load;
ESS, R-hat and the divergence count are deterministic given the seed.

Usage::

    JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/benchmark_notebook_sampler.py --notebook 05
    JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/benchmark_notebook_sampler.py --notebook 01 --quick
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
    recipes,
)
from tengri.analysis.diagnostics.autocorrelation import effective_sample_size

#: Each notebook's own convergence claim, and so the bar a replacement clears.
MAX_RHAT = 1.01
MAX_DIVERGENCES = 0

_NB05_FILTERS = (
    "galex_fuv", "galex_nuv", "sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z",
    "2mass_j", "2mass_h", "2mass_ks", "wise_w1", "wise_w2", "wise_w3", "wise_w4",
)
_NB01_FILTERS = ("sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z", "wise_w1")


def _build_nb05(ssp):
    """``05_fitting_photometry``: the quickstart model + logzsol + tau_diff. D=8."""
    return SEDModel.build(
        ssp_data=ssp,
        observation=Observation(photometry=Photometry.from_names(list(_NB05_FILTERS))),
        approx=WavePrecomp(),
        sfh=builders.sfh.tsnorm(defaults=FREE),
        dust=builders.dust.two_component(
            defaults=FIXED,
            law_bc="calzetti",
            tau_bc=Uniform(0.0, 1.0),
            tau_diff=Uniform(0.0, 1.0),
            emission=builders.dust.emission.modified_blackbody(defaults=FIXED),
        ),
        neb=builders.neb.none(),
        met={"logzsol": Uniform(-1.5, 0.3)},
        redshift=Fixed(0.05),
    )


def _build_nb01(ssp):
    """``01_why_jax``: the minimal mock-recovery recipe, six bands. D=5."""
    return SEDModel.build(
        ssp_data=ssp,
        observation=Observation(photometry=Photometry.from_names(list(_NB01_FILTERS))),
        **recipes.mock_recovery_minimal(),
    )


#: Per-notebook setup. ``shipped`` mirrors the notebook's committed fit call, so
#: the baseline row is what a reader actually runs -- not a tuned stand-in.
NOTEBOOKS = {
    "01": dict(
        build=_build_nb01,
        seed=1,
        snr=20.0,
        n_chains=4,
        shipped=dict(method="mcmc_nuts", n_warmup=100, n_samples=100),
        note=(
            "The committed fit is labelled in the notebook as a timing "
            "demonstration, NOT a converged posterior -- it produces one bar in "
            "a chart against an emcee literature baseline. Read the R-hat column "
            "accordingly; 100 warmup is not meant to clear any bar."
        ),
    ),
    "05": dict(
        build=_build_nb05,
        seed=7,
        snr=20.0,
        n_chains=2,
        shipped=dict(method="mcmc_nuts", n_warmup=600, n_samples=600),
        note=(
            "D=8 with dense_mass_matrix=True is the configuration CLAUDE.md "
            "records peaking at 20+ GB in NUTS warmup. HMC rows here therefore "
            "run dense_mass_matrix=False unless --dense is passed."
        ),
    ),
}


def configurations(nb: str, quick: bool, dense: bool) -> dict[str, dict]:
    """Sampler recipes to compare, keyed by label."""
    cfg = NOTEBOOKS[nb]
    shipped = dict(cfg["shipped"])
    if quick:
        shipped["n_warmup"] = min(shipped["n_warmup"], 300)
        shipped["n_samples"] = min(shipped["n_samples"], 150)

    configs = {"nuts (shipped)": shipped}

    draws = 150 if quick else max(600, shipped["n_samples"])
    warmup = 300 if quick else 1000
    leapfrogs = (20, 40) if quick else (20, 40, 80, 160)
    for leapfrog in leapfrogs:
        configs[f"hmc L={leapfrog}"] = dict(
            method="mcmc_hmc",
            n_warmup=warmup,
            n_samples=draws,
            n_leapfrog_steps=leapfrog,
            dense_mass_matrix=dense,
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
    parser.add_argument("--notebook", choices=sorted(NOTEBOOKS), required=True)
    parser.add_argument("--quick", action="store_true", help="shorter chains for a smoke run")
    parser.add_argument(
        "--dense", action="store_true", help="dense mass matrix on the HMC rows (memory-hungry)"
    )
    args = parser.parse_args()

    cfg = NOTEBOOKS[args.notebook]
    ssp = tengri.load_ssp("fsps_prsc_miles_chabrier", download=True)
    sed = cfg["build"](ssp)

    key_truth, key_mock, key_fit = jax.random.split(jax.random.PRNGKey(cfg["seed"]), 3)
    mock = generate_mock(sed, sed.spec.sample(key_truth), key=key_mock, snr=cfg["snr"])
    data = Data(photometry=(np.asarray(mock["flux_obs"]), np.asarray(mock["noise"])))

    print(f"notebook {args.notebook}: D = {len(sed.spec.free_params)} free parameters, "
          f"{cfg['n_chains']} chains")
    print(f"adoption bar: max split R-hat < {MAX_RHAT} and {MAX_DIVERGENCES} divergences")
    print(f"note: {cfg['note']}\n")

    header = (
        f"{'config':<20}{'wall s':>9}{'maxRhat':>10}{'div':>5}"
        f"{'minESS':>9}{'s/ESS':>9}  worst-mixing parameter"
    )
    print(header)
    print("-" * len(header), flush=True)

    rows = {}
    for label, kwargs in configurations(args.notebook, args.quick, args.dense).items():
        # A fresh model per row: adaptation caches are keyed on tuning settings
        # (#1853), and a fresh build also keeps the MAP seed identical per row.
        forward = ForwardModel.build(sed=cfg["build"](ssp))
        seed = forward.fit(
            data, method="map", key=key_fit, n_restarts=8, n_steps=800, verbose=False
        )
        started = time.perf_counter()
        posterior = forward.fit(
            data, key=key_fit, init_from=seed, n_chains=cfg["n_chains"], verbose=False, **kwargs
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
