# SPDX-License-Identifier: BSD-3-Clause
"""Does REPAIRING the collapsed fits remove the railing? (Not dropping them.)

Section 4i showed that EXCLUDING the collapsed galaxies removes the railing and
recovers sigma. Exclusion is not a valid estimator -- it selects on an inferred
quantity and biases the population. The honest version is to refit them.

At N=64, where the railing first appears (all-Laplace gives tau 434.6-491.2),
nine galaxies are flagged collapsed: 10 12 13 19 35 39 42 50 61. Each is
replaced by an mcmc_nuts refit of the SAME galaxy; every healthy galaxy keeps
its original Laplace fit. Only the broken fits change, so the comparison
isolates the repair rather than confounding it with a change of method.

Three arms:
  all-laplace  the bank as measured                     -> expect RAILED
  repaired     nine collapsed fits replaced by NUTS     -> the test
  healthy-only exclusion, for reference (biased, 4i)
"""

import json
import os

import numpy as np

from tengri.inference.population.diagnostics import credible_interval
from tengri.inference.population.estimator import SharedGrid, shared_log_posterior
from tengri.inference.population.reconstruct import centered_fields

with open("psd_bank_fixed/bank_meta.json") as fh:
    meta = json.load(fh)
LOG_AGE = np.asarray(meta["log_age_grid"])
TIMES = np.asarray(10.0**LOG_AGE)
SIG, TAU = meta["truth_sigma"], meta["truth_tau_myr"] * 1e6

grid = SharedGrid.uniform(
    tau_prior="uniform",
    sigma_bounds=tuple(meta["interim_sigma_bounds"]),
    tau_bounds_yr=(
        meta["interim_tau_bounds_myr"][0] * 1e6,
        meta["interim_tau_bounds_myr"][1] * 1e6,
    ),
    n_sigma=60,
    n_tau=60,
)
gs, gt = np.asarray(grid.sigma), np.asarray(grid.tau_yr)

N = 64
# repair sources, in priority order; healthy galaxies fall through to the bank
SOURCES = [
    "psd_bank_repair64b",
    "psd_bank_repair64",
    "psd_bank_nuts",
    "psd_bank_repair",
]


def load(i, allow_repair):
    if allow_repair:
        for d in SOURCES:
            p = f"{d}/gal_{i:04d}.npz"
            if os.path.exists(p):
                return np.load(p), d
    return np.load(f"psd_bank_fixed/gal_{i:04d}.npz"), "psd_bank_fixed"


def fields_of(d):
    xi = np.asarray(d["xi"])
    xi = xi.reshape(-1, xi.shape[-1])
    sig = np.asarray(d["sigma"]).reshape(-1)
    tau = np.asarray(d["tau_myr"]).reshape(-1) * 1e6
    k = min(xi.shape[0], sig.size)
    return np.asarray(centered_fields(xi[:k], sig[:k], tau[:k], LOG_AGE))


collapsed = []
for i in range(N):
    d, _ = load(i, allow_repair=False)
    ev = np.linalg.eigvalsh(np.cov(d["xi"], rowvar=False))
    if ev.sum() < 6.0 or ev.min() < 1e-2:
        collapsed.append(i)
print(f"N={N}: {len(collapsed)} collapsed -> {collapsed}")

missing = [i for i in collapsed if load(i, True)[1] == "psd_bank_fixed"]
if missing:
    print(f"NOT YET REPAIRED: {missing} -- rerun when their fits land\n")

K = 400
arms = {}
for name, repair, keep in (
    ("all-laplace", False, list(range(N))),
    ("repaired", True, list(range(N))),
    ("healthy-only", False, [i for i in range(N) if i not in collapsed]),
):
    f = []
    for i in keep:
        d, src = load(i, repair)
        m = fields_of(d)
        f.append(m[:K] if m.shape[0] >= K else m[np.arange(K) % m.shape[0]])
    arms[name] = np.array(f)

print(f"{'arm':>14} {'n':>4}  {'sigma 68% (0.75)':>21}  {'tau 68% Myr (150)':>23}")
for name, arr in arms.items():
    lp, ess = shared_log_posterior(arr, TIMES, grid, node_chunk=16)
    ci = credible_interval(np.asarray(lp), grid)
    zz = np.asarray(lp).reshape(gs.size, gt.size)
    a, b = np.unravel_index(np.argmax(zz), zz.shape)
    ok_s = "ok" if ci["sigma_lower"] <= SIG <= ci["sigma_upper"] else "MISS"
    ok_t = "ok" if ci["tau_lower_yr"] <= TAU <= ci["tau_upper_yr"] else "MISS"
    rail = "  RAILED" if ci["tau_lower_yr"] > 400e6 else ""
    lo_s, hi_s = ci["sigma_lower"], ci["sigma_upper"]
    lo_t, hi_t = ci["tau_lower_yr"] / 1e6, ci["tau_upper_yr"] / 1e6
    print(
        f"{name:>14} {arr.shape[0]:4d}  {lo_s:8.3f}-{hi_s:<8.3f} {ok_s:>4} "
        f"{lo_t:8.1f}-{hi_t:<8.1f} {ok_t:>4}{rail}"
        f"   mode ({gs[a]:.3f}, {gt[b] / 1e6:.0f})",
        flush=True,
    )
