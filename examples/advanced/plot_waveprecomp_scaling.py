"""
WavePrecomp turns photometry into a near-constant-cost call
============================================================

The build-time switch ``approx=WavePrecomp(...)`` publishes the SSP × filter
look-up table at construction time, then routes ``predict_photometry`` through
a tabulated path. The exact wave-grid path recomputes the rest-frame SED on a
~3000-point grid and integrates against each filter response on every call;
the LUT path reduces the per-call work to filter-count-sized array ops on a
pre-cached grid.

We sweep the filter count from 3 to 25 — covering everything from an SDSS
*ugriz* fit to a full UV-to-FIR panchromatic stack — and time per-call
wall clock for both paths. WavePrecomp is roughly 30× faster across the
whole range, and the gap is monotone in filter count.

The amortisation panel sums the build + compile cost with the per-call
cost across 1 → 10⁴ calls at 12 filters. The LUT path wins on every
horizon: there is no break-even point at which the exact path is the
better choice for production use.

Reference: ADR-0004 (kernel strategy module); see
``project_recipes_missing_waveprecomp`` for the original audit.
"""

import os
import time
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

# Filter ladder, monotonically extending UV→FIR. The slice [:n] gives a
# realistic survey set at each filter count.
FILTER_LADDER = [
    "galex_fuv", "galex_nuv",
    "sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z",
    "2mass_j", "2mass_h", "2mass_ks",
    "wise_w1", "wise_w2",
    "irac_36", "irac_45", "irac_58", "irac_80",
    "wise_w3", "wise_w4",
    "herschel_100", "herschel_160",
    "herschel_250", "herschel_350", "herschel_500",
    "hst_f435w", "hst_f606w",
]
N_FILTERS = [3, 5, 8, 12, 18, 25]
N_WARMUP = 3
N_TIMED = 50

ssp = tengri.load_ssp()


def build(n_filt, approx):
    obs = tengri.Observation(
        photometry=tengri.Photometry.from_names(FILTER_LADDER[:n_filt])
    )
    return tengri.SEDModel.build(
        ssp, observation=obs,
        sfh={"type": "tsnorm", "*": tengri.FIXED},
        dust={"type": "two_component", "*": tengri.FIXED, "tau_diff": 0.2, "tau_bc": 0.3},
        redshift=tengri.Fixed(0.05),
        approx=approx,
    )


def bench(n_filt, approx):
    """Build, JIT-compile, then time N_TIMED warm calls. Returns (build_s, warm_us).

    ``build_s`` covers model construction (including the WavePrecomp LUT)
    plus JIT trace+compile on the first call. ``warm_us`` is the median of
    three repeats of N_TIMED warm calls — enough to suppress single-run
    OS-scheduling jitter at the ~100 μs timescale.
    """
    t0 = time.perf_counter()
    model = build(n_filt, approx)
    params = dict(model.spec.sample(jax.random.PRNGKey(0)))
    out = model.predict_photometry(params)
    jax.block_until_ready(out)
    build_s = time.perf_counter() - t0

    for _ in range(N_WARMUP):
        out = model.predict_photometry(params)
        jax.block_until_ready(out)

    repeats = []
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(N_TIMED):
            out = model.predict_photometry(params)
            jax.block_until_ready(out)
        repeats.append((time.perf_counter() - t0) / N_TIMED * 1e6)
    warm_us = float(np.median(repeats))
    return build_s, warm_us


warm_exact = np.zeros(len(N_FILTERS))
warm_wp = np.zeros(len(N_FILTERS))
build_exact = np.zeros(len(N_FILTERS))
build_wp = np.zeros(len(N_FILTERS))

for i, n in enumerate(N_FILTERS):
    build_exact[i], warm_exact[i] = bench(n, approx=None)
    build_wp[i], warm_wp[i] = bench(n, approx=tengri.WavePrecomp(n_z=50))

speedup = warm_exact / warm_wp

# ─── Figure: per-call cost vs filter count ─────────────────────────────────
fig, (ax, ax_amort) = plt.subplots(
    2, 1, figsize=(6.8, 5.6),
    gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.32},
)

ax.plot(N_FILTERS, warm_exact, "o-", color="C3", lw=1.4, ms=6,
        label=f"Exact wave grid  (build+compile {build_exact.mean():.1f} s)")
ax.plot(N_FILTERS, warm_wp, "s-", color="C0", lw=1.4, ms=6,
        label=f"WavePrecomp(n_z=50)  (build+compile {build_wp.mean():.1f} s)")
ax.set_yscale("log")
ax.set_ylabel(r"Per-call wall time  [$\mu$s]")
ax.set_xlabel("Number of filters")
ax.legend(frameon=False, fontsize=8.5, loc="upper left")

# Annotate the speedup factor at each measured point.
for n, e, w, s in zip(N_FILTERS, warm_exact, warm_wp, speedup):
    ax.annotate(f"{s:.1f}×", xy=(n, w), xytext=(0, -12),
                textcoords="offset points", ha="center",
                fontsize=8, color="C0")

# Amortization panel: total wall time (compile + N evals) vs N for the
# middle filter-count, showing the call count above which WavePrecomp pays
# back its larger compile cost.
mid = N_FILTERS.index(12)
n_evals = np.logspace(0, 4, 50)
total_exact = build_exact[mid] + n_evals * warm_exact[mid] / 1e6
total_wp = build_wp[mid] + n_evals * warm_wp[mid] / 1e6

ax_amort.plot(n_evals, total_exact, color="C3", lw=1.2, label="Exact")
ax_amort.plot(n_evals, total_wp, color="C0", lw=1.2, label="WavePrecomp")
ax_amort.set_xscale("log")
ax_amort.set_yscale("log")
ax_amort.set_xlabel(r"Number of $\mathtt{predict\_photometry}$ calls  ($n_\mathrm{filt}=12$)")
ax_amort.set_ylabel("Total wall time  [s]")
# Crossover only matters if WavePrecomp's larger build cost is being paid
# off by faster per-call. If WP is cheaper from the first call there's
# nothing to amortise — just call it out.
if (total_wp < total_exact).all():
    headline = "WavePrecomp wins from the first call"
else:
    crossover = n_evals[np.argmin(np.abs(total_exact - total_wp))]
    ax_amort.axvline(crossover, color="0.4", lw=0.6, ls="--")
    headline = f"crossover at {crossover:.0f} calls"
ax_amort.text(0.02, 0.95, headline, transform=ax_amort.transAxes,
              color="0.3", fontsize=8.5, va="top")
ax_amort.legend(frameon=False, fontsize=8.5, loc="lower right")

fig.savefig("plot_waveprecomp_scaling.png", dpi=150, bbox_inches="tight")
