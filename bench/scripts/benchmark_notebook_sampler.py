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
    JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/benchmark_notebook_sampler.py \
        --notebook 01 --quick

    # six seeds per row, one fit per subprocess (the 2026-08-21 campaign protocol)
    JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/benchmark_notebook_sampler.py \
        --notebook 05 --only "nuts (shipped),mclmc" --seeds 6

``--notebook 00`` reproduces ``benchmark_quickstart_sampler.py``'s model, seed
and SNR so all three published configurations are reachable from one harness.

**The ``div`` column is not defined for every sampler.** MCLMC is unadjusted —
there is no Metropolis step that could reject, so there is no divergence to
count — and it prints ``n/a`` with the energy-error variance per dimension
(EEVPD) in its own column instead. A zero in ``div`` would be a claim about a
mechanism the sampler does not have; ``bench/reports/2026-08-17_nb01_nb05_nuts_vs_hmc.md``
already warns that zero divergences is not evidence of convergence for a
fixed-trajectory sampler, and that warning applies with more force here.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
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
from tengri.config.exceptions import DeadFitError

#: Each notebook's own convergence claim, and so the bar a replacement clears.
MAX_RHAT = 1.01
MAX_DIVERGENCES = 0

_NB05_FILTERS = (
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
    "wise_w3",
    "wise_w4",
)
_NB01_FILTERS = ("sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z", "wise_w1")
_NB00_FILTERS = (
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

#: The healthy control. ``jwst_nonparametric_fits`` -- a non-tsnorm, 9-D
#: ``continuity`` SFH against 19 JWST bands at z = 1.5 -- re-measured across six
#: seeds in #2014, where NUTS on a diagonal metric returns a median min ESS of
#: 119 in 85 s. Every other row in this file is a tsnorm posterior, and all three
#: of those are degenerate: Finding 15 of
#: ``bench/reports/2026-08-20_cuda_device_matrix.md`` puts their min ESS at 1-4
#: out of 600 draws. A sampler comparison run only on degenerate fixtures cannot
#: separate "this sampler is slow" from "this posterior is hard", because NUTS
#: answers bad geometry by doubling its trajectory -- up to 2^10 leapfrog steps
#: per draw where a healthy posterior needs 2^3-2^5 -- so the wall clock and the
#: non-convergence are one phenomenon, not two.
_CTL_BROAD = (
    "jwst_f090w",
    "jwst_f115w",
    "jwst_f150w",
    "jwst_f200w",
    "jwst_f277w",
    "jwst_f356w",
    "jwst_f444w",
)
_CTL_MEDIUM = (
    "jwst_f140m",
    "jwst_f162m",
    "jwst_f182m",
    "jwst_f210m",
    "jwst_f250m",
    "jwst_f300m",
    "jwst_f335m",
    "jwst_f360m",
    "jwst_f410m",
    "jwst_f430m",
    "jwst_f460m",
    "jwst_f480m",
)
_CTL_Z_GAL = 1.5


def _build_ctl(ssp):
    """The non-tsnorm control: 9-D ``continuity`` SFH, 19 JWST bands, z = 1.5.

    Mirrors ``notebooks/jwst_nonparametric_fits.py`` exactly, including the
    ``tau_bc = 0.0`` pin that #2014 measured as the difference between 176 and 30
    effective samples at the same seed. Its ``ssp_data`` differs from the three
    tsnorm rows' (``prsc_miles_chabrier_wNE``, nebular baked in) because the page
    turns nebular emission on.
    """
    from tengri.cosmology import age_at_z

    t_univ = float(age_at_z(_CTL_Z_GAL))
    bin_edges = np.concatenate([[0.0, 0.03], np.logspace(np.log10(0.1), np.log10(t_univ), 6)])
    return SEDModel.build(
        ssp_data=ssp,
        observation=Observation(photometry=Photometry.from_names(list(_CTL_BROAD + _CTL_MEDIUM))),
        redshift=Fixed(_CTL_Z_GAL),
        sfh={"type": "continuity", "all_params": FREE, "bin_edges_gyr": bin_edges},
        met={"logzsol": Uniform(-1.5, 0.3), "all_params": FIXED},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": FIXED,
            "tau_bc": 0.0,
            "tau_diff": Uniform(0.0, 2.0),
        },
        neb={"type": "ssp"},
        approx=WavePrecomp(),
    )


def _build_nb00(ssp):
    """``00_quickstart``: tsnorm SFH, two-component Calzetti, fixed z. D=7, 12 bands.

    Reconstructed to match ``bench/reports/2026-08-17_quickstart_nuts_vs_hmc.md``,
    which states **D = 7** and names ``met_logzsol`` as the worst-mixing
    parameter of its HMC L=160 row. ``benchmark_quickstart_sampler.build_model``
    as committed builds **D = 6** — it has no ``met`` group — so it can no longer
    reproduce its own published table; the free metallicity is restored here on
    ``05_fitting_photometry``'s range, that notebook being described as this
    model plus ``met_logzsol`` and ``dust_tau_diff``.
    """
    return SEDModel.build(
        ssp_data=ssp,
        observation=Observation(photometry=Photometry.from_names(list(_NB00_FILTERS))),
        approx=WavePrecomp(),
        sfh=builders.sfh.tsnorm(all_params=FREE),
        dust_attenuation=builders.dust.two_component(
            all_params=FIXED, law_bc="calzetti", law_diff="power_law", tau_bc=Uniform(0.0, 1.0)
        ),
        neb=builders.neb.none(),
        met={"logzsol": Uniform(-1.5, 0.3)},
        redshift=Fixed(0.05),
    )


def _build_nb05(ssp):
    """``05_fitting_photometry``: the quickstart model + logzsol + tau_diff. D=8.

    ``dust=builders.dust.two_component(..., emission=...)`` and a lone
    ``law_bc=`` were how this read when the 2026-08-17 report was written; both
    spellings have since been retired (dust split into the peer groups
    ``dust_attenuation`` / ``dust_emission``, and a per-screen law now requires
    naming both screens). The script raised ``ValueError`` at model build for
    every row, so **neither this file nor
    ``bench/scripts/benchmark_quickstart_sampler.py`` could be rerun at all**
    until this was repaired.

    ``law_diff="power_law"`` restores what the retired spelling resolved to —
    ``TwoComponentDustConfig.law_diff``'s own default, i.e. Charlot & Fall's
    Calzetti birth cloud over a power-law diffuse screen. Writing
    ``law="calzetti"`` instead (both screens Calzetti) looks like the same
    repair and is a **different model**: measured on this notebook at seed 7,
    it moves the shipped NUTS row from the published R-hat 1.0033 / 0
    divergences / ESS 144 to R-hat 1.14 / 166 divergences / ESS 3.0.
    """
    return SEDModel.build(
        ssp_data=ssp,
        observation=Observation(photometry=Photometry.from_names(list(_NB05_FILTERS))),
        approx=WavePrecomp(),
        sfh=builders.sfh.tsnorm(all_params=FREE),
        dust_attenuation=builders.dust.two_component(
            all_params=FIXED,
            law_bc="calzetti",
            law_diff="power_law",
            tau_bc=Uniform(0.0, 1.0),
            tau_diff=Uniform(0.0, 1.0),
        ),
        dust_emission=builders.dust.emission.modified_blackbody(all_params=FIXED),
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
    "ctl": dict(
        build=_build_ctl,
        ssp="prsc_miles_chabrier_wNE",
        seed=4,
        snr=20.0,
        n_chains=2,
        shipped=dict(
            method="mcmc_nuts",
            n_warmup=1000,
            n_samples=400,
            dense_mass_matrix=False,
        ),
        note=(
            "The HEALTHY control, and the only non-tsnorm row here. Read every "
            "other row against it: a sampler comparison run only on degenerate "
            "posteriors measures the fixture, not the sampler."
        ),
    ),
    "00": dict(
        build=_build_nb00,
        seed=9,
        snr=30.0,
        n_chains=4,
        dense_hmc=True,
        shipped=dict(
            method="mcmc_nuts",
            n_warmup=1500,
            n_samples=250,
            n_burnin=0,
            dense_mass_matrix=False,
            target_accept_rate=0.9,
        ),
        note=(
            "The quickstart's own model and seed, reproduced from "
            "benchmark_quickstart_sampler.py so all three published "
            "configurations are reachable from one harness."
        ),
    ),
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
            dense_mass_matrix=dense or cfg.get("dense_hmc", False),
            target_accept_rate=0.9,
        )

    # MCLMC draws are single integrator steps, not trajectories, so its
    # n_samples is an order of magnitude larger than NUTS's *by construction*
    # and not by generosity: successive draws sit ~L/step_size ~ 40-50 steps
    # apart. Two budgets, so the report can show the dependence rather than
    # assert a number. `allow_unvalidated` because the backend is quarantined
    # until a campaign like this one clears it.
    for label, (mclmc_warmup, mclmc_draws) in {
        "mclmc": (5000, 20000),
        "mclmc 2x": (5000, 40000),
    }.items():
        configs[label] = dict(
            method="mcmc_mclmc",
            n_warmup=300 if quick else mclmc_warmup,
            n_samples=1000 if quick else mclmc_draws,
            allow_unvalidated=True,
        )
    return configs


def _gradients_per_draw(diag: dict) -> float | None:
    """Gradient evaluations one draw cost, or None when the sampler cannot say.

    The load-independent half of the comparison, and the one that survives a
    shared box. NUTS reports ``tree_depth_mean``, and a tree of mean depth d is
    2**d leapfrog steps, each one gradient: that is how it answers bad geometry,
    by spending more of them, up to 2**10. MCLMC cannot answer geometry at all —
    the McLachlan integrator is two gradients per step and one step per draw,
    always — so this column is the whole structural difference between them in
    one number.
    """
    if "tree_depth_mean" in diag:
        return float(2.0 ** diag["tree_depth_mean"])
    if "energy_var_per_dim" in diag:  # MCLMC: one isokinetic McLachlan step
        return 2.0
    return None


def score(posterior, wall: float) -> dict:
    """Diagnostics that decide adoption, plus the parameter that mixes worst.

    ``divergences`` is ``None``, not ``0``, for a sampler whose diagnostics
    carry no ``n_divergent`` key. An unadjusted sampler has no accept step, so
    the count does not exist; reporting a zero would read as "no divergences
    were found" when the truth is that none could be. Those runs report
    ``eevpd`` instead — the achieved energy-error variance per dimension,
    against the target the tuner aimed at.
    """
    diag = posterior.diagnostics or {}
    rhats = posterior.rhat()
    grad_per_draw = _gradients_per_draw(diag)
    ess = effective_sample_size({k: np.asarray(v) for k, v in posterior.samples.items()})
    finite = [(k, v["ess"]) for k, v in ess.items() if np.isfinite(v["ess"])]
    worst_name, worst_ess = (
        min(finite, key=lambda pair: pair[1]) if finite else ("?", float("nan"))
    )
    return {
        "wall": wall,
        "rhat": max(float(v) for v in rhats.values()) if rhats else float("nan"),
        "divergences": (None if "n_divergent" not in diag else int(diag["n_divergent"] or 0)),
        "eevpd": diag.get("energy_var_per_dim"),
        "eevpd_target": diag.get("energy_var_per_dim_target"),
        "nonfinite_steps": diag.get("n_nonfinite_steps"),
        "min_ess": worst_ess,
        "worst": worst_name,
        "sec_per_ess": wall / max(worst_ess, 1e-9),
        "grad_per_draw": grad_per_draw,
        "grad_per_ess": (
            None
            if grad_per_draw is None
            else grad_per_draw
            * diag.get("n_samples", 0)
            * diag.get("n_chains", 1)
            / max(worst_ess, 1e-9)
        ),
    }


def clears_bar(row: dict) -> bool:
    """The notebooks' own adoption bar, with the divergence clause made honest.

    ``divergences is None`` means the sampler cannot report divergences, so the
    clause is vacuous rather than satisfied; the energy diagnostics are what
    substitute for it and they are printed beside the row, not folded into a
    pass/fail.
    """
    if row.get("dead_fit"):
        return False
    if not (row["rhat"] < MAX_RHAT):
        return False
    return row["divergences"] is None or row["divergences"] <= MAX_DIVERGENCES


def run_one(nb: str, label: str, kwargs: dict, seed: int) -> dict:
    """Build the notebook's mock at ``seed``, MAP-seed it, run one fit, score it."""
    cfg = NOTEBOOKS[nb]
    ssp = tengri.load_ssp(cfg.get("ssp", "fsps_prsc_miles_chabrier"), download=True)
    sed = cfg["build"](ssp)
    key_truth, key_mock, key_fit = jax.random.split(jax.random.PRNGKey(seed), 3)
    mock = generate_mock(sed, sed.spec.sample(key_truth), key=key_mock, snr=cfg["snr"])
    data = Data(photometry=(np.asarray(mock["flux_obs"]), np.asarray(mock["noise"])))

    # A fresh model per row: adaptation caches are keyed on tuning settings
    # (#1853), and a fresh build also keeps the MAP seed identical per row.
    forward = ForwardModel.build(sed=cfg["build"](ssp))
    map_seed = forward.fit(
        data, method="map", key=key_fit, n_restarts=8, n_steps=800, verbose=False
    )
    started = time.perf_counter()
    try:
        posterior = forward.fit(
            data,
            key=key_fit,
            init_from=map_seed,
            n_chains=cfg["n_chains"],
            verbose=False,
            **kwargs,
        )
    except DeadFitError as exc:
        # A refusal is an outcome, not a missing value (#2090). Before that PR
        # these seeds came back as a frozen posterior plus a warning, which
        # scored as a row rather than as a failure; recording it as one keeps
        # the seed in the denominator.
        return {
            "wall": time.perf_counter() - started,
            "rhat": float("nan"),
            "divergences": None,
            "dead_fit": str(exc)[:300],
            "eevpd": None,
            "eevpd_target": None,
            "nonfinite_steps": None,
            "min_ess": 0.0,
            "worst": "REFUSED (DeadFitError)",
            "sec_per_ess": float("inf"),
            "grad_per_draw": None,
            "grad_per_ess": None,
        }
    return score(posterior, time.perf_counter() - started)


def format_row(label: str, row: dict) -> str:
    """One table line, with ``n/a`` where a column does not apply to the sampler."""
    if row.get("dead_fit"):
        return f"{label:<20}{row['wall']:>9.1f}{'REFUSED (DeadFitError)':>44}"
    div = "n/a" if row["divergences"] is None else str(row["divergences"])
    gpd = "" if row.get("grad_per_draw") is None else f"{row['grad_per_draw']:>8.1f}"
    gpe = "" if row.get("grad_per_ess") is None else f"{row['grad_per_ess']:>10.0f}"
    eevpd = "" if row.get("eevpd") is None else f"  EEVPD {row['eevpd']:.2e}"
    return (
        f"{label:<20}{row['wall']:>9.1f}{row['rhat']:>10.4f}{div:>5}"
        f"{row['min_ess']:>9.1f}{row['sec_per_ess']:>9.3f}{gpd}{gpe}  {row['worst']}{eevpd}"
    )


def _sweep(args, configs: dict[str, dict]) -> None:
    """Driver: one fit per subprocess, ``args.seeds`` seeds per row.

    A subprocess per fit is not fastidiousness. Adaptation and MAP caches live
    on the Model and are content-keyed, so two rows in one process can share an
    entry that the second row's settings should have invalidated (#1853), and a
    seed sweep is precisely the shape that trips it. A fresh interpreter is the
    only guarantee that the row measured is the row requested.
    """
    seeds = [NOTEBOOKS[args.notebook]["seed"] + i for i in range(args.seeds)]
    results: dict[str, list[dict]] = {}
    for label in configs:
        results[label] = []
        for seed in seeds:
            proc = subprocess.run(
                [
                    sys.executable,
                    os.path.abspath(__file__),
                    "--notebook",
                    args.notebook,
                    "--only",
                    label,
                    "--seed",
                    str(seed),
                    "--emit-json",
                    *(["--quick"] if args.quick else []),
                    *(["--dense"] if args.dense else []),
                ],
                capture_output=True,
                text=True,
                env={**os.environ, "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS", "cpu")},
            )
            if proc.returncode != 0:
                print(f"{label:<20} seed {seed}: FAILED\n{proc.stderr[-2000:]}", flush=True)
                continue
            row = json.loads(proc.stdout.strip().splitlines()[-1])
            results[label].append(row)
            print(f"  seed {seed}: " + format_row(label, row), flush=True)

    print("\nper-row summary over seeds (worst seed decides the bar):")
    header = (
        f"{'config':<20}{'seeds':>6}{'maxRhat':>10}{'div':>5}"
        f"{'minESS':>9}{'medWall':>9}  worst param"
    )
    print(header)
    print("-" * len(header))
    for label, rows in results.items():
        if not rows:
            print(f"{label:<20}  no successful seeds")
            continue
        dead = [r for r in rows if r.get("dead_fit")]
        live = [r for r in rows if not r.get("dead_fit")]
        if not live:
            print(f"{label:<20}{len(rows):>6}  every seed REFUSED (DeadFitError)")
            continue
        worst = max(live, key=lambda r: r["rhat"])
        div_vals = [r["divergences"] for r in live if r["divergences"] is not None]
        div = "n/a" if not div_vals else str(max(div_vals))
        min_ess = min(r["min_ess"] for r in live)
        med_wall = float(np.median([r["wall"] for r in rows]))
        eevpd = [r["eevpd"] for r in rows if r.get("eevpd") is not None]
        tail = f"  max EEVPD {max(eevpd):.2e}" if eevpd else ""
        if dead:
            tail += f"  [{len(dead)}/{len(rows)} seeds REFUSED]"
        print(
            f"{label:<20}{len(rows):>6}{worst['rhat']:>10.4f}{div:>5}"
            f"{min_ess:>9.1f}{med_wall:>9.1f}  {worst['worst']}{tail}"
        )
        passes = sum(clears_bar(r) for r in rows)
        print(f"{'':20}  clears bar on {passes}/{len(rows)} seeds")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"notebook": args.notebook, "seeds": seeds, "rows": results}, fh, indent=2)
        print(f"\nwrote {args.json}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", choices=sorted(NOTEBOOKS), required=True)
    parser.add_argument("--quick", action="store_true", help="shorter chains for a smoke run")
    parser.add_argument(
        "--dense", action="store_true", help="dense mass matrix on the HMC rows (memory-hungry)"
    )
    parser.add_argument("--seed", type=int, default=None, help="override the notebook's own seed")
    parser.add_argument("--only", default=None, help="comma-separated config labels to run")
    parser.add_argument(
        "--seeds",
        type=int,
        default=0,
        help="run each row across this many consecutive seeds, one fit per subprocess",
    )
    parser.add_argument("--json", default=None, help="write the seed sweep's rows here")
    parser.add_argument(
        "--emit-json", action="store_true", help="internal: print one JSON row and exit"
    )
    args = parser.parse_args()

    cfg = NOTEBOOKS[args.notebook]
    configs = configurations(args.notebook, args.quick, args.dense)
    if args.only:
        wanted = [label.strip() for label in args.only.split(",")]
        missing = [label for label in wanted if label not in configs]
        if missing:
            parser.error(f"unknown config(s) {missing}; available: {sorted(configs)}")
        configs = {label: configs[label] for label in wanted}

    if args.emit_json:
        label, kwargs = next(iter(configs.items()))
        row = run_one(args.notebook, label, kwargs, args.seed or cfg["seed"])
        print(json.dumps(row))
        return

    print(f"notebook {args.notebook}: {cfg['n_chains']} chains")
    print(f"adoption bar: max split R-hat < {MAX_RHAT} and {MAX_DIVERGENCES} divergences")
    print(f"note: {cfg['note']}\n")

    if args.seeds:
        _sweep(args, configs)
        return

    header = (
        f"{'config':<20}{'wall s':>9}{'maxRhat':>10}{'div':>5}"
        f"{'minESS':>9}{'s/ESS':>9}{'grad/draw':>8}{'grad/ESS':>10}  worst-mixing parameter"
    )
    print(header)
    print("-" * len(header), flush=True)

    rows = {}
    for label, kwargs in configs.items():
        rows[label] = run_one(args.notebook, label, kwargs, args.seed or cfg["seed"])
        print(format_row(label, rows[label]), flush=True)

    print("\nverdict (ranked on seconds per effective sample):")
    baseline = rows.get("nuts (shipped)")
    for label, row in sorted(rows.items(), key=lambda kv: kv[1]["sec_per_ess"]):
        versus = (
            f"{baseline['sec_per_ess'] / max(row['sec_per_ess'], 1e-9):5.2f}x vs nuts"
            if baseline
            else ""
        )
        verdict = "clears bar" if clears_bar(row) else "MISSES bar"
        print(f"  {label:<20} {verdict:<11} {versus}")


if __name__ == "__main__":
    main()
