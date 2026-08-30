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
import json
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
_NB00_FILTERS = (
    "galex_fuv", "galex_nuv", "sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z",
    "2mass_j", "2mass_h", "2mass_ks", "wise_w1", "wise_w2",
)


def _build_nb00(ssp):
    """``00_quickstart``: tsnorm SFH, two-component Calzetti, fixed z. D=7, 12 bands.

    Mirrors ``benchmark_quickstart_sampler.py``'s ``build_model`` so the rows here
    stay comparable with ``bench/reports/2026-08-17_quickstart_nuts_vs_hmc.md``,
    which is the whole point of measuring this configuration again.
    """
    return SEDModel.build(
        ssp_data=ssp,
        observation=Observation(photometry=Photometry.from_names(list(_NB00_FILTERS))),
        approx=WavePrecomp(),
        sfh=builders.sfh.tsnorm(all_params=FREE),
        dust_attenuation=builders.dust.two_component(
            all_params=FIXED, law="calzetti", tau_bc=Uniform(0.0, 1.0)
        ),
        neb=builders.neb.none(),
        # Free metallicity is what makes this D=7 rather than D=6, and the
        # 2026-08-17 report's L=160 row names ``met_logzsol`` as its
        # worst-mixing parameter -- so it was free there too.
        met={"logzsol": Uniform(-2.0, 0.2)},
        redshift=Fixed(0.05),
    )


def _build_nb05(ssp):
    """``05_fitting_photometry``: the quickstart model + logzsol + tau_diff. D=8."""
    return SEDModel.build(
        ssp_data=ssp,
        observation=Observation(photometry=Photometry.from_names(list(_NB05_FILTERS))),
        approx=WavePrecomp(),
        sfh=builders.sfh.tsnorm(all_params=FREE),
        dust_attenuation=builders.dust.two_component(
            all_params=FIXED,
            law="calzetti",
            tau_bc=Uniform(0.0, 1.0),
            tau_diff=Uniform(0.0, 1.0),
        ),
        dust_emission=builders.dust.emission.modified_blackbody(all_params=FIXED),
        neb=builders.neb.none(),
        met={"logzsol": Uniform(-1.5, 0.3)},
        redshift=Fixed(0.05),
    )


def _build_ctl(ssp):
    """Non-tsnorm control: the same 14 bands as nb05 over a **DPL** SFH. D=7.

    Notebooks 00, 01 and 05 all run a ``tsnorm`` SFH, and
    ``bench/reports/2026-08-20_cuda_device_matrix.md`` Finding 15 measured that
    family's ``skew``/``trunc``/``width_gyr`` as strongly degenerate: ESS_min
    stays 1.7-4.3 *even with 260 spectral pixels*, so 52x the data moves it only
    1.7 to 4.3 and it is not a data-volume problem.

    A sampler that misses the bar on all three notebooks is therefore
    uninterpretable on its own: "this sampler is wrong" and "this posterior is
    degenerate for everything" predict the same table. This row separates them.
    Dual power law is the family ``recipes.star_forming_photometry`` ships and
    the one the 2026-05-22 backend validation used, so a failure here is about
    the sampler.
    """
    return SEDModel.build(
        ssp_data=ssp,
        observation=Observation(photometry=Photometry.from_names(list(_NB05_FILTERS))),
        approx=WavePrecomp(),
        sfh=builders.sfh.dpl(all_params=FREE),
        dust_attenuation=builders.dust.two_component(
            all_params=FIXED,
            law="calzetti",
            tau_bc=Uniform(0.0, 1.0),
            tau_diff=Uniform(0.0, 1.0),
        ),
        dust_emission=builders.dust.emission.modified_blackbody(all_params=FIXED),
        neb=builders.neb.none(),
        met={"logzsol": Uniform(-1.5, 0.3)},
        redshift=Fixed(0.05),
    )


def _build_nb01(ssp):
    """``01_why_jax``: the minimal mock-recovery recipe, six bands. D=7."""
    return SEDModel.build(
        ssp_data=ssp,
        observation=Observation(photometry=Photometry.from_names(list(_NB01_FILTERS))),
        **recipes.mock_recovery_minimal(),
    )


#: Per-notebook setup. ``shipped`` mirrors the notebook's committed fit call, so
#: the baseline row is what a reader actually runs -- not a tuned stand-in.
NOTEBOOKS = {
    "00": dict(
        build=_build_nb00,
        # PRNGKey(9) at SNR 30, not this file's usual (1, 20): these are
        # ``benchmark_quickstart_sampler.py``'s own values, and the whole reason
        # to carry nb00 here is that its rows stay comparable with
        # 2026-08-17_quickstart_nuts_vs_hmc.md. Changing them silently would
        # produce a different mock and a baseline that looks like a regression.
        seed=9,
        snr=30.0,
        n_chains=4,
        shipped=dict(
            method="mcmc_nuts",
            n_warmup=1500,
            n_samples=250,
            n_burnin=0,
            dense_mass_matrix=False,
            target_accept_rate=0.9,
        ),
        note=(
            "Mirrors benchmark_quickstart_sampler.py's own NUTS row so the table "
            "stays comparable with 2026-08-17_quickstart_nuts_vs_hmc.md."
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
    "ctl": dict(
        build=_build_ctl,
        # nb05's seed, SNR and chain count exactly: this row is a CONTROL for
        # the SFH family, so everything else must be held fixed or it controls
        # for nothing.
        seed=7,
        snr=20.0,
        n_chains=2,
        shipped=dict(method="mcmc_nuts", n_warmup=600, n_samples=600),
        note=(
            "NOT a notebook. The non-tsnorm control: nb05's mock, bands, seed, "
            "SNR and chain count over a DPL SFH instead of tsnorm. Exists so a "
            "sampler failure on 00/01/05 can be told apart from the tsnorm "
            "family's own degeneracy (2026-08-20_cuda_device_matrix.md, "
            "Finding 15)."
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


#: Sampler families a row can belong to. ``--methods`` selects a subset.
FAMILIES = ("nuts", "hmc", "ghmc", "chees")


def configurations(nb: str, quick: bool, dense: bool, families=FAMILIES) -> dict[str, dict]:
    """Sampler recipes to compare, keyed by label."""
    cfg = NOTEBOOKS[nb]
    shipped = dict(cfg["shipped"])
    if quick:
        shipped["n_warmup"] = min(shipped["n_warmup"], 300)
        shipped["n_samples"] = min(shipped["n_samples"], 150)

    configs = {}
    if "nuts" in families:
        configs["nuts (shipped)"] = shipped

    draws = 150 if quick else max(600, shipped["n_samples"])
    warmup = 300 if quick else 1000
    if "hmc" in families:
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
    if "ghmc" in families:
        # GHMC is one leapfrog per step, so a step is ~L times cheaper than an
        # HMC row's and the draw budget is scaled up to match: comparing a
        # 600-draw GHMC against a 600-draw L=160 HMC would be comparing samplers
        # given 160x different gradient budgets. Warmup is the MEADS ensemble's,
        # priced at n_warmup * n_ensemble gradients.
        #
        # ``allow_unvalidated`` is required while mcmc_ghmc is tier="broken" --
        # which is exactly the claim these rows exist to settle. Remove it if
        # and only if the tier moves.
        for ensemble in ((32,) if quick else (32, 64)):
            configs[f"ghmc meads E={ensemble}"] = dict(
                method="mcmc_ghmc",
                n_warmup=warmup,
                n_burnin=200 if quick else 500,
                n_samples=draws * 4,
                n_ensemble=ensemble,
                allow_unvalidated=True,
            )
    if "chees" in families:
        # Two rows, and the pair IS the experiment. ChEES with
        # `mass_matrix_estimation=None` has no metric of its own -- the geometry
        # comes from tengri's analytic `J^T N^-1 J + I` or from nowhere -- so
        # "did preconditioning help?" cannot be answered by one row.
        #
        # The draw budget matches the HMC rows rather than GHMC's x4: a ChEES
        # step is a full L-leapfrog HMC proposal, not GHMC's single leapfrog, so
        # equal draws is already roughly equal gradient budget.
        # Three arms, and the third is not redundant with the second.
        # ``precondition=True`` resolves to DEFAULT_WHITENING_STRENGTH = 0.5, so
        # a metric of condition 1e6 is whitened only to 1e3. ``1.0`` is the full
        # whitening that actually drives the condition number to 1 at the
        # expansion point -- and the one #1442 warns amplifies a *misspecified*
        # metric without bound. Which of those two effects dominates on these
        # posteriors is a measurement, not a preference.
        for label, precondition in (
            ("chees", None),
            ("chees+precond", True),
            ("chees+full", 1.0),
        ):
            configs[label] = dict(
                method="mcmc_chees",
                n_warmup=warmup,
                n_burnin=200 if quick else 500,
                n_samples=draws,
                n_ensemble=32,
                precondition=precondition,
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
        "unique_frac": _unique_draw_fraction(posterior),
        # Leapfrog steps per proposal actually in effect. Deliberately NOT named
        # "learned": mcmc_hmc reports its hand-set L under the same diagnostics
        # key, and the whole point of the ChEES rows is that theirs was not set.
        # Which it is comes from the config label; this column says what it was.
        # None for samplers with no trajectory at all (NUTS reports a tree depth,
        # not a length).
        "n_leapfrog": posterior.diagnostics.get("n_leapfrog_steps"),
        "step_size": posterior.diagnostics.get("step_size"),
        # NUTS only. The gradient cost of a NUTS draw is ~2**tree_depth - 1
        # leapfrog steps, so this is the column that says whether a slow fit is
        # a slow sampler or a posterior forcing the tree deeper. A healthy
        # geometry sits at depth 3-5 (7-31 leapfrogs); saturation at
        # max_num_doublings means every draw paid the cap.
        "tree_depth_mean": posterior.diagnostics.get("tree_depth_mean"),
        "tree_depth_max": posterior.diagnostics.get("tree_depth_max"),
        "frac_max_depth": posterior.diagnostics.get("frac_max_depth"),
        "leapfrogs_per_draw": (
            2.0 ** posterior.diagnostics["tree_depth_mean"] - 1.0
            if posterior.diagnostics.get("tree_depth_mean") is not None
            else posterior.diagnostics.get("n_leapfrog_steps")
        ),
    }


def _unique_draw_fraction(posterior) -> float:
    """Fraction of the joint draws that are distinct positions.

    Zero divergences is not evidence of health. ``mcmc_nuts`` returned a
    *completely frozen* chain on 3.1% of galaxies with zero divergences reported
    (#1999): every proposal rejected, R-hat near 1.0 because within- and
    between-chain variance are both zero, and nothing in the divergence column
    to see. Split R-hat cannot detect that and neither can ESS on its own, so
    the count of distinct rows is carried as its own column. Healthy chains sit
    near 1.0; anything far below it is a chain that stopped moving.
    """
    keys = sorted(posterior.samples)
    if not keys:
        return float("nan")
    matrix = np.column_stack([np.asarray(posterior.samples[k]).ravel() for k in keys])
    return float(len(np.unique(matrix, axis=0)) / matrix.shape[0])


def _fmt_rhat(value: float) -> str:
    """Four decimals near the bar, scientific once a chain has actually diverged.

    A fixed ``.4f`` was fine while every row was near 1.0. It is not: a NUTS row
    on nb05's seed-0 mock reported R-hat 1.4e13, which printed 19 characters wide
    and ran into the neighboring column, so the whole line became unreadable
    exactly when it had the most to say.
    """
    return f"{value:>10.4f}" if abs(value) < 1e4 else f"{value:>10.3e}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", choices=sorted(NOTEBOOKS), required=True)
    parser.add_argument("--quick", action="store_true", help="shorter chains for a smoke run")
    parser.add_argument(
        "--dense", action="store_true", help="dense mass matrix on the HMC rows (memory-hungry)"
    )
    parser.add_argument(
        "--methods",
        default=",".join(FAMILIES),
        help=f"comma-separated subset of {FAMILIES}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "override the notebook's own seed. The campaign protocol is six seeds "
            "per row, ONE FIT PER SUBPROCESS -- a shared process reuses the "
            "adaptation cache and the compile cache across seeds, so the second "
            "seed onward is not an independent measurement."
        ),
    )
    parser.add_argument("--json", default=None, help="append one JSON row per config to this file")
    args = parser.parse_args()

    families = tuple(m.strip() for m in args.methods.split(",") if m.strip())
    unknown = set(families) - set(FAMILIES)
    if unknown:
        parser.error(f"unknown --methods entries {sorted(unknown)}; choose from {FAMILIES}")

    cfg = NOTEBOOKS[args.notebook]
    seed = cfg["seed"] if args.seed is None else args.seed
    ssp = tengri.load_ssp("fsps_prsc_miles_chabrier", download=True)
    sed = cfg["build"](ssp)

    key_truth, key_mock, key_fit = jax.random.split(jax.random.PRNGKey(seed), 3)
    mock = generate_mock(sed, sed.spec.sample(key_truth), key=key_mock, snr=cfg["snr"])
    data = Data(photometry=(np.asarray(mock["flux_obs"]), np.asarray(mock["noise"])))

    print(f"notebook {args.notebook}: D = {len(sed.spec.free_params)} free parameters, "
          f"{cfg['n_chains']} chains, seed {seed}")
    print(f"adoption bar: max split R-hat < {MAX_RHAT} and {MAX_DIVERGENCES} divergences")
    print(f"note: {cfg['note']}\n")

    header = (
        f"{'config':<20}{'wall s':>9}{'maxRhat':>10}{'div':>5}"
        f"{'minESS':>9}{'s/ESS':>9}{'uniq':>7}  worst-mixing parameter"
    )
    print(header)
    print("-" * len(header), flush=True)

    rows = {}
    for label, kwargs in configurations(args.notebook, args.quick, args.dense, families).items():
        # A fresh model per row: adaptation caches are keyed on tuning settings
        # (#1853), and a fresh build also keeps the MAP seed identical per row.
        forward = ForwardModel.build(sed=cfg["build"](ssp))
        map_seed = forward.fit(
            data, method="map", key=key_fit, n_restarts=8, n_steps=800, verbose=False
        )
        started = time.perf_counter()
        posterior = forward.fit(
            data,
            key=key_fit,
            init_from=map_seed,
            n_chains=cfg["n_chains"],
            verbose=False,
            **kwargs,
        )
        rows[label] = score(posterior, time.perf_counter() - started)
        row = rows[label]
        print(
            f"{label:<20}{row['wall']:>9.1f}{_fmt_rhat(row['rhat'])}{row['divergences']:>5}"
            f"{row['min_ess']:>9.1f}{row['sec_per_ess']:>9.3f}{row['unique_frac']:>7.3f}"
            f"  {row['worst']}",
            flush=True,
        )
        if args.json:
            with open(args.json, "a") as fh:
                fh.write(
                    json.dumps(
                        {"notebook": args.notebook, "seed": seed, "config": label, **row}
                    )
                    + "\n"
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
