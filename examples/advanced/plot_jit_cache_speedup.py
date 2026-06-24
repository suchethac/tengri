r"""
The persistent JIT cache turns the second-run cold start into a no-op
=====================================================================

Every ``import tengri`` enables an on-disk JAX compile cache at
``~/.cache/tengri_jax_cache`` (override with ``TENGRI_JAX_CACHE_DIR``,
opt out with ``TENGRI_DISABLE_JAX_CACHE=1``). The cache survives Python
restarts: notebook re-runs, slurm tasks, and benchmark workers all skip
the expensive first compile.

Within one Python process JAX already memoizes every JIT, so a second call
to the same function is free regardless of the on-disk cache. The cache
only matters when a *new process* is spawned. To measure honestly we
launch four fresh subprocesses against an empty cache directory:

* Run 1: cold cache. JAX traces and compiles every kernel touched by
  ``predict_photometry`` and writes them to disk.
* Runs 2–4: warm cache. JAX reads compiled artefacts off disk.

The bar chart reports total subprocess wall time. The cold→warm drop is
roughly an order of magnitude. The same effect lands on ``predict_rest_sed``,
``predict_spectrum`` and any inference loop the user spawns.

References: docs/inference/compilation_cache.md (full configuration);
``tengri.clear_cache()`` to wipe the cache after a JAX upgrade.
"""

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time

import matplotlib.pyplot as plt
import numpy as np

from tengri.analysis.plotting import setup_style

setup_style()

# Resolve the absolute SSP path in the parent so the child subprocess can
# load it directly without relying on cwd-walking.
from pathlib import Path as _Path

_DATA = next(p / "data" for p in _Path.cwd().resolve().parents if (p / "data").exists())
_SSP_PATH = str(next(_DATA.glob("ssp_*.h5")))

CHILD_SOURCE = textwrap.dedent(
    """
    import json, os, time

    t_total = time.perf_counter()
    t0 = time.perf_counter()
    import tengri
    t_import = time.perf_counter() - t0

    t0 = time.perf_counter()
    ssp = tengri.load_ssp_data(os.environ["TENGRI_DEMO_SSP"])
    obs = tengri.Observation(
        photometry=tengri.Photometry.from_names(
            ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z",
             "2mass_j", "2mass_h", "2mass_ks"]
        )
    )
    sed_model = tengri.SEDModel.build(
        ssp, observation=obs,
        sfh={"type": "tsnorm", "*": tengri.FREE},
        dust={"type": "two_component", "*": tengri.FIXED,
              "tau_diff": tengri.Uniform(0.0, 1.5),
              "tau_bc": tengri.Uniform(0.0, 1.5)},
        redshift=tengri.Fixed(0.05),
        approx=tengri.WavePrecomp(n_z=50),
    )
    t_build = time.perf_counter() - t0

    import jax
    truth = dict(sed_model.spec.sample(jax.random.PRNGKey(0)))
    mock = sed_model.mock(truth, snr=20.0, key=jax.random.PRNGKey(1))

    # MAP fit. Heavier than a single predict_photometry — compiles
    # forward + gradient + optimizer step. This is what fills the cache.
    forward = tengri.ForwardModel.build(sed=sed_model, observation=obs)
    t0 = time.perf_counter()
    forward.fit(
        mock.flux_obs, mock.noise,
        method="map", optimizer="adam", n_steps=50, verbose=False,
    )
    t_map = time.perf_counter() - t0

    t_wall = time.perf_counter() - t_total
    print(json.dumps(dict(
        wall=t_wall, import_=t_import, build=t_build, map=t_map,
    )))
    """
).strip()


def run_subprocess(cache_dir):
    env = {
        **os.environ,
        "TENGRI_JAX_CACHE_DIR": str(cache_dir),
        "TENGRI_DEMO_SSP": _SSP_PATH,
        "JAX_PLATFORMS": "cpu",
        "TF_CPP_MIN_LOG_LEVEL": "3",
    }
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-c", CHILD_SOURCE],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    parent_wall = time.perf_counter() - t0
    last_line = proc.stdout.strip().splitlines()[-1]
    timings = json.loads(last_line)
    timings["parent_wall"] = parent_wall
    return timings


N_RUNS = 4
with tempfile.TemporaryDirectory(prefix="tengri_demo_cache_") as cache_dir:
    results = [run_subprocess(cache_dir) for _ in range(N_RUNS)]

walls = np.array([r["parent_wall"] for r in results])
imports = np.array([r["import_"] for r in results])
builds = np.array([r["build"] for r in results])
maps = np.array([r["map"] for r in results])

# ─── Figure: subprocess wall time, stacked by phase ────────────────────────
fig, ax = plt.subplots(figsize=(6.6, 4.0))
labels = ["Run 1\n(cold cache)"] + [f"Run {i + 2}\n(warm cache)" for i in range(N_RUNS - 1)]
x = np.arange(N_RUNS)

# Stack: import, build (model construction), MAP fit (forward + grad
# + optimizer compile, the dominant JIT cost). Anything not accounted
# for goes into a thin "process overhead" sliver.
remainder = walls - (imports + builds + maps)
remainder = np.clip(remainder, 0.0, None)

ax.bar(x, imports, color="C0", label="import tengri")
ax.bar(x, builds, bottom=imports, color="C3", label="SEDModel.build")
ax.bar(
    x,
    maps,
    bottom=imports + builds,
    color="C1",
    label="MAP fit (forward + grad + optimizer compile)",
)
ax.bar(x, remainder, bottom=imports + builds + maps, color="0.7", label="process overhead")

# Total-wall-time annotation above each bar.
for xi, total in zip(x, walls):
    ax.text(xi, total + 0.15, f"{total:.2f} s", ha="center", fontsize=9, fontweight="bold")

speedup = walls[0] / walls[1:].mean()
ax.text(
    0.98,
    0.95,
    f"warm-cache speedup: {speedup:.1f}x",
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=10,
    color="0.2",
    bbox=dict(facecolor="white", edgecolor="0.7", boxstyle="round,pad=0.4"),
)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Subprocess wall time  [s]")
ax.set_ylim(0, walls.max() * 1.25)
ax.legend(frameon=False, fontsize=8.5, loc="upper right", bbox_to_anchor=(1.0, 0.88))

fig.savefig("plot_jit_cache_speedup.png", dpi=150, bbox_inches="tight")
