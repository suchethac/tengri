# SPDX-License-Identifier: BSD-3-Clause
"""Does MCLMC's over-target energy error move the posterior?

R-hat and ESS are statements about mixing, not about which distribution was
mixed to, and this backend has already been shown to report R-hat 1.0007 on a
run whose energy error was 170,000x its target. The only honest test is to put
the marginals beside an exact sampler's on the same data and the same seed.

Per parameter:
  * mean and sd, with the difference expressed as a z-score on the combined
    Monte Carlo standard error, so "agrees within MC error" is a number;
  * the 16/50/84 quantiles either side;
  * a two-sample Kolmogorov-Smirnov statistic on the stored draws.

MCSE uses each run's own ESS (sd / sqrt(ESS)), which is what makes the z-score
fair between a 1200-draw NUTS run and an 80000-draw MCLMC one.
"""

from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import warnings

warnings.filterwarnings("ignore")

import jax
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark_notebook_sampler as B

import tengri
from tengri import Data, ForwardModel, generate_mock
from tengri.analysis.diagnostics.autocorrelation import effective_sample_size

NB = sys.argv[1] if len(sys.argv) > 1 else "05"
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 8
OUT = sys.argv[3] if len(sys.argv) > 3 else None

cfg = B.NOTEBOOKS[NB]
ssp = tengri.load_ssp(cfg.get("ssp", "fsps_prsc_miles_chabrier"), download=True)
sed0 = cfg["build"](ssp)
kt, km, kf = jax.random.split(jax.random.PRNGKey(SEED), 3)
truth_draw = sed0.spec.sample(kt)
mock = generate_mock(sed0, truth_draw, key=km, snr=cfg["snr"])
data = Data(photometry=(np.asarray(mock["flux_obs"]), np.asarray(mock["noise"])))
truth = {k: float(np.asarray(v)) for k, v in truth_draw.items()}


def run(**kw):
    forward = ForwardModel.build(sed=cfg["build"](ssp))
    seed = forward.fit(data, method="map", key=kf, n_restarts=8, n_steps=800, verbose=False)
    t = time.perf_counter()
    p = forward.fit(data, key=kf, init_from=seed, n_chains=cfg["n_chains"], verbose=False, **kw)
    ess = effective_sample_size({k: np.asarray(v) for k, v in p.samples.items()})
    return p, ess, time.perf_counter() - t


nuts, nuts_ess, nuts_wall = run(**cfg["shipped"])
mclmc, mclmc_ess, mclmc_wall = run(
    method="mcmc_mclmc", n_warmup=5000, n_samples=40000, allow_unvalidated=True
)

md = mclmc.diagnostics
ratio = md["energy_var_per_dim"] / md["energy_var_per_dim_target"]
print(f"\nnotebook {NB}, seed {SEED}, {cfg['n_chains']} chains")
print(
    f"NUTS   {cfg['shipped']['n_warmup']}/{cfg['shipped']['n_samples']}: "
    f"{nuts_wall:.0f} s, {nuts.diagnostics.get('n_divergent')} divergences, "
    f"maxRhat {max(float(v) for v in nuts.rhat().values()):.4f}"
)
print(
    f"MCLMC  5000/40000: {mclmc_wall:.0f} s, EEVPD {md['energy_var_per_dim']:.3e} "
    f"({ratio:.0f}x target), max|dE| {md['max_abs_energy_change']:.3g}, "
    f"step {md['step_size']:.4f}, maxRhat "
    f"{max(float(v) for v in mclmc.rhat().values()):.4f}"
)

free = sorted(
    k
    for k in nuts.samples
    if k in nuts_ess
    and k in mclmc_ess
    and np.isfinite(nuts_ess[k]["ess"])
    and np.isfinite(mclmc_ess[k]["ess"])
)

hdr = (
    f"{'parameter':<28}{'NUTS mean':>11}{'MCLMC mean':>11}{'z(mean)':>9}"
    f"{'NUTS sd':>10}{'MCLMC sd':>10}{'sd ratio':>9}{'KS':>7}{'truth':>10}"
)
print("\n" + hdr)
print("-" * len(hdr))
rows = []
for k in free:
    a, b = np.asarray(nuts.samples[k]), np.asarray(mclmc.samples[k])
    mcse_a = a.std() / np.sqrt(max(nuts_ess[k]["ess"], 1e-9))
    mcse_b = b.std() / np.sqrt(max(mclmc_ess[k]["ess"], 1e-9))
    z = float((b.mean() - a.mean()) / np.sqrt(mcse_a**2 + mcse_b**2))
    ks = float(stats.ks_2samp(a, b).statistic)
    rows.append(
        dict(
            parameter=k,
            nuts_mean=float(a.mean()),
            mclmc_mean=float(b.mean()),
            z_mean=z,
            nuts_sd=float(a.std()),
            mclmc_sd=float(b.std()),
            sd_ratio=float(b.std() / a.std()),
            ks=ks,
            truth=truth.get(k),
            nuts_q=[float(q) for q in np.percentile(a, [16, 50, 84])],
            mclmc_q=[float(q) for q in np.percentile(b, [16, 50, 84])],
        )
    )
    print(
        f"{k:<28}{a.mean():>11.4f}{b.mean():>11.4f}{z:>9.2f}"
        f"{a.std():>10.4f}{b.std():>10.4f}{b.std() / a.std():>9.3f}"
        f"{ks:>7.3f}{truth.get(k, float('nan')):>10.4f}"
    )

worst_z = max(abs(r["z_mean"]) for r in rows)
worst_ks = max(r["ks"] for r in rows)
print(f"\nworst |z(mean)| = {worst_z:.2f}   worst KS = {worst_ks:.3f}")
print("\nquantiles (16 / 50 / 84):")
for r in rows:
    qa, qb = r["nuts_q"], r["mclmc_q"]
    print(
        f"  {r['parameter']:<28} NUTS {qa[0]:8.4f} {qa[1]:8.4f} {qa[2]:8.4f}   "
        f"MCLMC {qb[0]:8.4f} {qb[1]:8.4f} {qb[2]:8.4f}"
    )

if OUT:
    with open(OUT, "w") as fh:
        json.dump(
            {
                "notebook": NB,
                "seed": SEED,
                "n_chains": cfg["n_chains"],
                "nuts": {
                    "config": cfg["shipped"],
                    "wall_s": nuts_wall,
                    "divergences": nuts.diagnostics.get("n_divergent"),
                    "max_split_rhat": max(float(v) for v in nuts.rhat().values()),
                },
                "mclmc": {
                    "config": {"n_warmup": 5000, "n_samples": 40000},
                    "wall_s": mclmc_wall,
                    "max_split_rhat": max(float(v) for v in mclmc.rhat().values()),
                    "energy_var_per_dim": md["energy_var_per_dim"],
                    "energy_var_per_dim_target": md["energy_var_per_dim_target"],
                    "energy_var_ratio": ratio,
                    "max_abs_energy_change": md["max_abs_energy_change"],
                    "step_size": md["step_size"],
                },
                "worst_abs_z_mean": worst_z,
                "worst_ks": worst_ks,
                "per_parameter": rows,
            },
            fh,
            indent=2,
        )
    print(f"\nwrote {OUT}")
