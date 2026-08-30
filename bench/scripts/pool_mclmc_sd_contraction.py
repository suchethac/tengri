# SPDX-License-Identifier: BSD-3-Clause
"""Pool the MCLMC/NUTS sd ratios across seeds: is the contraction real?

Six of eight sd ratios below 1 on one seed is p = 0.29 under a sign test --
suggestive, not a claim. Pooling the parameter-seed pairs turns it into one.

Two statistics, because they answer different questions:
  * sign test over all pairs -- is the DIRECTION consistent?
  * mean log sd ratio with its own standard error -- what is the MAGNITUDE, and
    is it distinguishable from zero? Logs because a ratio's null is 1, not 0,
    and log makes the two directions symmetric.

Seed 12 is excluded and the exclusion is the point: NUTS returned 1200
divergences out of 1200 draws there and every parameter sd is ~1e-4, i.e. the
reference sampler froze. Comparing against a frozen chain measures the
reference's failure, not MCLMC's bias.
"""

from __future__ import annotations

import glob
import json
import math
import os

import numpy as np
from scipy import stats

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
PATTERN = "2026-08-30_mclmc_posterior_agreement_nb05_seed*.json"
paths = sorted(glob.glob(os.path.join(RES, PATTERN)))

pairs, excluded = [], []
for path in paths:
    with open(path) as fh:
        d = json.load(fh)
    seed = d["seed"]
    # A frozen reference is not a reference. NUTS sd ~1e-4 against MCLMC's ~1e-1
    # is a dead chain, and its sd ratio carries no information about bias.
    nuts_sds = [r["nuts_sd"] for r in d["per_parameter"]]
    mclmc_sds = [r["mclmc_sd"] for r in d["per_parameter"]]
    if np.median(np.array(mclmc_sds) / np.array(nuts_sds)) > 5.0:
        excluded.append((seed, d["nuts"]["divergences"], float(np.median(nuts_sds))))
        continue
    for r in d["per_parameter"]:
        pairs.append((seed, r["parameter"], r["sd_ratio"], r["z_mean"], r["ks"]))

print(f"pooled from {len(paths)} seed files at {RES}")
for seed, div, med_sd in excluded:
    print(
        f"  EXCLUDED seed {seed}: NUTS returned {div} divergences and a median "
        f"parameter sd of {med_sd:.2e} -- the reference chain is frozen"
    )

ratios = np.array([p[2] for p in pairs])
logs = np.log(ratios)
n = len(ratios)
below = int((ratios < 1.0).sum())
sign_p = stats.binomtest(below, n, 0.5).pvalue
mean_log = float(logs.mean())
se_log = float(logs.std(ddof=1) / math.sqrt(n))
t_stat = mean_log / se_log
t_p = float(2 * stats.t.sf(abs(t_stat), df=n - 1))

print(f"\n{n} parameter-seed pairs from seeds {sorted({p[0] for p in pairs})}")
print(f"sd ratios below 1: {below}/{n}   sign test p = {sign_p:.4f}")
print(f"mean log sd ratio = {mean_log:+.5f} +/- {se_log:.5f} (t = {t_stat:+.2f}, p = {t_p:.4f})")
print(
    f"  -> MCLMC marginals are {(math.exp(mean_log) - 1) * 100:+.2f}% "
    f"[{(math.exp(mean_log - 1.96 * se_log) - 1) * 100:+.2f}%, "
    f"{(math.exp(mean_log + 1.96 * se_log) - 1) * 100:+.2f}%] wide vs NUTS"
)

zs = np.array([p[3] for p in pairs])
kss = np.array([p[4] for p in pairs])
print(f"\nworst |z(mean)| over all pairs = {np.abs(zs).max():.2f}")
print(f"worst KS over all pairs        = {kss.max():.3f}")
print(f"mean z(mean)                   = {zs.mean():+.3f} (a shift, if any, is in the mean)")

print("\nper-parameter sd ratio across seeds:")
names = sorted({p[1] for p in pairs})
for name in names:
    vals = [p[2] for p in pairs if p[1] == name]
    print(f"  {name:<28} " + "  ".join(f"{v:.3f}" for v in vals) + f"   mean {np.mean(vals):.3f}")

out = os.path.join(RES, "2026-08-30_mclmc_sd_contraction_pooled.json")
with open(out, "w") as fh:
    json.dump(
        {
            "question": "is MCLMC's marginal contraction vs NUTS real, and how large?",
            "notebook": "05",
            "seeds_used": sorted({p[0] for p in pairs}),
            "seeds_excluded": [
                {
                    "seed": s,
                    "nuts_divergences": d,
                    "nuts_median_sd": m,
                    "reason": "frozen reference chain",
                }
                for s, d, m in excluded
            ],
            "n_pairs": n,
            "sd_ratios_below_one": below,
            "sign_test_p": float(sign_p),
            "mean_log_sd_ratio": mean_log,
            "se_log_sd_ratio": se_log,
            "t_stat": float(t_stat),
            "t_p": t_p,
            "percent_width_change": (math.exp(mean_log) - 1) * 100,
            "percent_ci95": [
                (math.exp(mean_log - 1.96 * se_log) - 1) * 100,
                (math.exp(mean_log + 1.96 * se_log) - 1) * 100,
            ],
            "worst_abs_z_mean": float(np.abs(zs).max()),
            "worst_ks": float(kss.max()),
            "pairs": [
                {"seed": s, "parameter": p, "sd_ratio": r, "z_mean": z, "ks": k}
                for s, p, r, z, k in pairs
            ],
        },
        fh,
        indent=2,
    )
print(f"\nwrote {out}")
