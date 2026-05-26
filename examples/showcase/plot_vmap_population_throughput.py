"""
JAX vmap turns one SED into a population fit — 10,000 galaxies in seconds

JAX's `vmap` (vectorized map) transforms a single-sample function into a
batched operation, enabling population-scale throughput with automatic
parallelization. This example builds a galaxy SED model, then measures
end-to-end wall-clock time to predict photometry for N = 1, 10, 100, 1,000,
and 10,000 galaxies simultaneously. Scaling is sublinear due to XLA's
Just-In-Time compilation and parallelism — typical batches peak at 10–100
galaxies per millisecond on CPU. A JIT warm-up run excludes compilation
overhead, revealing true amortized throughput.

Reference: Bradbury et al. 2018, JAX: composable transformations of Python+NumPy
programs (https://arxiv.org/abs/1811.02361); JAX documentation on vmap
(https://jax.readthedocs.io/en/latest/api_reference/jax.numpy.vmap.html).
"""

import time
import warnings

import jax
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

# Build a minimal star-forming galaxy model
BANDS = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]
obs = tengri.Observation(photometry=tengri.Photometry.from_names(BANDS))

# Load SSP library and build model with free SFH and dust
ssp_data = tengri.load_ssp()
model = tengri.SEDModel.build(
    ssp_data,
    observation=obs,
    sfh={"type": "dpl", "*": tengri.FREE},
    dust={
        "type": "two_component",
        "law_bc": "calzetti",
        "*": tengri.FIXED,
        "tau_bc": 0.5,
    },
    redshift=tengri.Fixed(0.05),
)

# Define batch sizes to test
batch_sizes = np.array([1, 10, 100, 1000, 10000])
wall_times_ms = np.zeros(len(batch_sizes))
throughputs_per_sec = np.zeros(len(batch_sizes))

# Key for random parameter sampling
key = jax.random.PRNGKey(42)

# Test each batch size
for i, n_batch in enumerate(batch_sizes):
    # Sample random parameters for the batch
    keys = jax.random.split(key, n_batch)
    batch_params = jax.vmap(model.spec.sample)(keys)

    # Compile vmapped photometry prediction
    batch_predict = jax.jit(jax.vmap(model.predict_photometry))

    # Warm-up run to compile the vmapped kernel (excludes from timing)
    _ = batch_predict(batch_params)
    batch_predict(batch_params)[0].block_until_ready()

    # Measure wall-clock time over multiple runs
    n_runs = 50 if n_batch <= 1000 else 10
    t0 = time.perf_counter()
    for _ in range(n_runs):
        photometry = batch_predict(batch_params)
        # Ensure computation is complete on device
        jax.tree.map(lambda x: x.block_until_ready(), photometry)
    t_total = time.perf_counter() - t0

    # Average per run and convert to milliseconds
    wall_times_ms[i] = (t_total / n_runs) * 1e3

    # Throughput in galaxies per second
    throughputs_per_sec[i] = n_batch / (wall_times_ms[i] / 1e3)

    print(
        f"N={n_batch:>5d}: {wall_times_ms[i]:7.2f} ms "
        f"({throughputs_per_sec[i]:7.0f} galaxies/sec, "
        f"{wall_times_ms[i] / n_batch * 1e6:6.1f} µs/galaxy)"
    )

# Plot scaling: wall time and throughput on log-log
fig, (ax_time, ax_throughput) = plt.subplots(1, 2, figsize=(10.0, 3.8))

# Left: wall-clock time
ax_time.scatter(batch_sizes, wall_times_ms, s=60, color="C0", zorder=3)
ax_time.plot(batch_sizes, wall_times_ms, color="C0", lw=1.2, zorder=2)

# Overlay linear reference (proportional to N)
linear_ref = (wall_times_ms[0] / batch_sizes[0]) * batch_sizes
ax_time.plot(batch_sizes, linear_ref, ls="--", color="0.5", lw=1.0, label="Linear scaling")

ax_time.set_xlabel("Batch size (number of galaxies)")
ax_time.set_ylabel("Wall time [ms]")
ax_time.set_xscale("log")
ax_time.set_yscale("log")
ax_time.legend(frameon=False, fontsize=9)
ax_time.grid(True, alpha=0.3, which="both")

# Right: throughput (galaxies per second)
ax_throughput.scatter(batch_sizes, throughputs_per_sec, s=60, color="C1", zorder=3)
ax_throughput.plot(batch_sizes, throughputs_per_sec, color="C1", lw=1.2, zorder=2)

ax_throughput.set_xlabel("Batch size (number of galaxies)")
ax_throughput.set_ylabel("Throughput [galaxies/s]")
ax_throughput.set_xscale("log")
ax_throughput.set_yscale("log")
ax_throughput.grid(True, alpha=0.3, which="both")

fig.tight_layout()
plt.savefig("plot_vmap_population_throughput.png", dpi=150, bbox_inches="tight")

# Print summary
print(
    f"\nSummary:\n"
    f"  Peak throughput: {throughputs_per_sec.max():.0f} galaxies/sec "
    f"at N={batch_sizes[throughputs_per_sec.argmax()]}\n"
    f"  Time for 10,000 galaxies: {wall_times_ms[-1]:.1f} ms\n"
    f"  Scaling efficiency: {(wall_times_ms[0] / batch_sizes[0]) / (wall_times_ms[-1] / batch_sizes[-1]):.1f}× "
    f"(linear=1.0, sublinear>1.0 via XLA amortization)"
)
