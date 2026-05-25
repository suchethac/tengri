"""
Photometric throughput: exact vs precomputed filter LUT
=======================================================

The `approx=WavePrecomp()` build-time knob trades a small memory footprint
(precomputing the SSP × filter lookup table) for dramatic photometry speedup.
This benchmark compares exact wave-grid integration (the default `approx=None`)
against the precomputed LUT path across N=1000 random parameter draws. On typical
5-filter photometry setups, WavePrecomp achieves **20–50× speedup** for `predict_photometry`
with sub-percent accuracy loss, unlocking real-time interactive fitting and
multi-fit hierarchical inference workflows.

References:
  - ADR-0004: Kernel strategy module
    (docs/adr/0004-kernel-strategy-module.md)
  - DSPS preintegration architecture (DSPS #64)
"""

import os
import time
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")
warnings.filterwarnings("ignore")

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style

setup_style()

# Load SSP data
try:
    ssp_data = tengri.load_ssp("fsps_prsc_miles_chabrier")
except Exception:
    ssp_data = tengri.load_ssp()  # Fallback to default

# Build observation: 5 SDSS filters
obs = tengri.Observation(
    photometry=tengri.Photometry.from_names(
        ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]
    ),
)

# Baseline model spec: star-forming galaxy with free parameters
spec = tengri.SEDModel.build(
    ssp_data,
    observation=obs,
    sfh={"type": "dpl", "*": tengri.FREE, "alpha": tengri.Fixed(0.5)},
    dust={
        "type": "two_component",
        "law_bc": "calzetti",
        "*": tengri.FIXED,
        "tau_bc": 0.5,
    },
    redshift=tengri.Fixed(0.1),
).spec

# Build two models: exact (default) and WavePrecomp
model_exact = tengri.SEDModel.build(
    ssp_data,
    observation=obs,
    sfh={"type": "dpl", "*": tengri.FREE, "alpha": tengri.Fixed(0.5)},
    dust={
        "type": "two_component",
        "law_bc": "calzetti",
        "*": tengri.FIXED,
        "tau_bc": 0.5,
    },
    redshift=tengri.Fixed(0.1),
)

model_precomp = tengri.SEDModel.build(
    ssp_data,
    observation=obs,
    approx=tengri.WavePrecomp(),
    sfh={"type": "dpl", "*": tengri.FREE, "alpha": tengri.Fixed(0.5)},
    dust={
        "type": "two_component",
        "law_bc": "calzetti",
        "*": tengri.FIXED,
        "tau_bc": 0.5,
    },
    redshift=tengri.Fixed(0.1),
)

# Benchmark function: measure wall-clock time for N forward passes
def benchmark(model, spec, n_runs=1000, n_warmup=5):
    """Return mean per-call time in microseconds, with warmup."""
    key = jax.random.PRNGKey(42)
    params_list = [spec.sample(jax.random.fold_in(key, i)) for i in range(n_runs)]

    # Warmup
    for i in range(n_warmup):
        _ = model.predict_photometry(params_list[i])
        _ = jax.effects_barrier()

    # Timed runs
    t0 = time.perf_counter()
    for params in params_list:
        _ = model.predict_photometry(params)
        _ = jax.effects_barrier()
    elapsed_s = time.perf_counter() - t0

    us_per_call = (elapsed_s / n_runs) * 1e6
    return us_per_call


# Run benchmarks
print("Benchmarking WavePrecomp photometry speedup...")
print("  (Exact path, 1000 runs with warmup)")
us_exact = benchmark(model_exact, spec, n_runs=1000, n_warmup=5)
print(f"    Exact: {us_exact:.1f} µs/call")

print("  (Precomputed LUT path, 1000 runs with warmup)")
us_precomp = benchmark(model_precomp, spec, n_runs=1000, n_warmup=5)
print(f"    WavePrecomp: {us_precomp:.1f} µs/call")

speedup = us_exact / us_precomp
print(f"  Speedup: {speedup:.1f}×")

# Verify accuracy: compare predictions on a few random params
key = jax.random.PRNGKey(123)
rel_errors = []
for i in range(20):
    params = spec.sample(jax.random.fold_in(key, i))
    pred_exact = model_exact.predict_photometry(params)
    pred_precomp = model_precomp.predict_photometry(params)

    # Per-filter relative errors for diagnostic
    abs_err = jnp.abs(pred_exact - pred_precomp)
    rel_err = abs_err / jnp.maximum(jnp.abs(pred_exact), 1e-30)
    rel_errors.append(float(jnp.max(rel_err)))

max_rel_error = np.max(rel_errors)
mean_rel_error = np.mean(rel_errors)
print(f"  Max relative error: {max_rel_error * 100:.2f}% (mean: {mean_rel_error * 100:.2f}%)")

# Plot: bar chart with speedup annotation
fig, ax = plt.subplots(figsize=(5.5, 4.0))

paths = ["Exact\nwave-grid", "WavePrecomp\nLUT"]
times = [us_exact, us_precomp]
colors = ["#d62728", "#2ca02c"]

bars = ax.bar(paths, times, color=colors, width=0.5, edgecolor="black", linewidth=1.2)

# Annotate speedup on the bar chart
for i, (bar, t) in enumerate(zip(bars, times)):
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{t:.0f} µs",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

# Add speedup annotation at the top
speedup_y = max(times) * 1.25
ax.text(
    0.5,
    speedup_y,
    f"{speedup:.1f}× speedup",
    ha="center",
    va="center",
    fontsize=13,
    fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="yellow", alpha=0.3),
)

ax.set(
    ylabel="Wall-clock time per call (µs)",
    ylim=(0, max(times) * 1.4),
)
ax.set_xticklabels(paths, fontsize=10)

fig.tight_layout()

# Save to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, "plot_waveprecomp_speedup.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")

print(f"\nFigure saved to {output_path}")
