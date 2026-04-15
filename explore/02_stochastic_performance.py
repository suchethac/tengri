# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Stochastic SFH Fitting
#
# Full correlated-field SFH with GP latent (D = 135).
# Three inference methods compared on the same mock galaxy:
#
# 1. **geoVI (NIFTy)** — variational, fast default (~12s)
# 2. **native_geovi (JIT)** — compile once, then ~12s cached
# 3. **Ray Tracing** — exact MCMC (Behroozi 2025)
#
# All produce posterior samples, SFH recovery, and corner plots.

# %%
import os
import sys
import time
import warnings

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    _here = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _here = os.getcwd()
proj_root = os.path.abspath(os.path.join(_here, ".."))
os.chdir(proj_root)
sys.path.insert(0, os.path.join(proj_root, "notebooks"))

from tengri import (  # noqa: E402
    Fitter,
    Fixed,
    Model,
    Observation,
    ParamSpec,
    Uniform,
    load_ssp_data,
)
from tengri.observation import SpectroscopyConfig  # noqa: E402

from _plot_style import (  # noqa: E402
    COLORS,
    plot_corner_comparison,
    plot_sfh,
    safe_corner,
    setup_style,
)

setup_style()

FIGDIR = os.path.join("explore", "figures")
os.makedirs(FIGDIR, exist_ok=True)

# %% [markdown]
# ## 1. Setup

# %%
ssp_data = load_ssp_data(
    "data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5",
)
WAVE_OBS = jnp.linspace(3800.0, 9200.0, 200)
obs = Observation(spectroscopy=SpectroscopyConfig(wave_obs=WAVE_OBS))

# %% [markdown]
# ## 2. Stochastic model (D = 135)

# %%
N_GRID = 128

spec = ParamSpec(
    sfh_dpl_alpha=Uniform(0.5, 3.0),
    sfh_dpl_beta=Uniform(0.3, 2.0),
    sfh_dpl_tau_gyr=Uniform(1.0, 8.0),
    sfh_dpl_log_peak_sfr=Uniform(0.0, 1.5),
    met_logzsol=Uniform(-2.0, 0.2),
    dust_tau_bc=Uniform(0.0, 2.0),
    dust_tau_diff=Uniform(0.0, 1.0),
    dust_slope=Fixed(-0.7),
    sfh_field_psd_sigma=Fixed(1.5),
    sfh_field_psd_tau_myr=Fixed(50.0),
    redshift=Fixed(0.1),
    stochastic=True,
    n_grid=N_GRID,
)
model = Model(spec, ssp_data, observation=obs)
n_phys = len(spec.free_params)
D = n_phys + N_GRID
print(f"{n_phys} physical + {N_GRID} GP latent = D={D}")

# %%
key = jax.random.PRNGKey(123)
true_params = spec.sample(key)
mock = model.mock_spectrum(true_params, WAVE_OBS, snr=30.0, key=key)

# %% [markdown]
# ## 3. geoVI — NIFTy fast path (default)
#
# Calls JIT'd primitives from Python. No heavy XLA compile.
# 2000 posterior samples. Best for single-galaxy fitting.

# %%
fitter = Fitter(model, mock.flux_obs, mock.noise)

t0 = time.perf_counter()
result_geovi = fitter.run(
    "geovi", n_iterations=15, n_posterior_samples=2000, verbose=True,
)
t_geovi = time.perf_counter() - t0
n_samples_geovi = len(next(iter(result_geovi.samples.values())))
print(f"\ngeoVI: {t_geovi:.1f}s, {n_samples_geovi} samples")

# %% [markdown]
# ### 3a. geoVI results

# %%
result_geovi.summary_table()

# %%
fig, ax = plt.subplots(figsize=(7, 4))
plot_sfh(model, result_geovi, true_params=true_params, ax=ax,
         method="geoVI", label="geoVI")
ax.set_title(f"geoVI — {t_geovi:.0f}s, {n_samples_geovi} samples")
fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "02_sfh_geovi.png"),
            dpi=150, bbox_inches="tight")
plt.show()

# %%
fig = safe_corner(result_geovi, truths=true_params)
if fig is not None:
    fig.savefig(os.path.join(FIGDIR, "02_corner_geovi.png"),
                dpi=150, bbox_inches="tight")
    plt.show()

# %% [markdown]
# ## 4. native_geovi — fully JIT (for batch/vmap)
#
# Entire optimizer compiled into one XLA program.
# MAP-initialized (1 seed). 2000 posterior samples.
# First call: ~60s (compile). Cached calls: ~12s.

# %%
# First call: XLA compilation
t0 = time.perf_counter()
result_native_1 = fitter.run(
    "native_geovi", n_iterations=15, n_seeds=1, verbose=True,
)
t_native_compile = time.perf_counter() - t0
print(f"\nnative_geovi (1st, compile): {t_native_compile:.1f}s")

# %%
# Second call: XLA cached
t0 = time.perf_counter()
result_native_2 = fitter.run(
    "native_geovi", n_iterations=15, n_seeds=1,
    key=jax.random.PRNGKey(999), verbose=True,
)
t_native_cached = time.perf_counter() - t0
n_samples_native = len(next(iter(result_native_2.samples.values())))
print(f"\nnative_geovi (cached): {t_native_cached:.1f}s, "
      f"{n_samples_native} samples")
print(f"Speedup: {t_native_compile / t_native_cached:.1f}x")

# %% [markdown]
# ### 4a. native_geovi results

# %%
fig, ax = plt.subplots(figsize=(7, 4))
plot_sfh(model, result_native_2, true_params=true_params, ax=ax,
         method="geoVI", label="native geoVI",
         color=COLORS["rt"])
ax.set_title(f"native geoVI — {t_native_cached:.0f}s cached, "
             f"{n_samples_native} samples")
fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "02_sfh_native.png"),
            dpi=150, bbox_inches="tight")
plt.show()

# %%
fig = safe_corner(result_native_2, truths=true_params)
if fig is not None:
    fig.savefig(os.path.join(FIGDIR, "02_corner_native.png"),
                dpi=150, bbox_inches="tight")
    plt.show()

# %% [markdown]
# ## 5. Ray Tracing — exact MCMC (Behroozi 2025)
#
# Stochastic-gradient resilient MCMC. Provides exact posterior
# samples (no variational approximation). Slower mixing for
# spectroscopy (tight likelihood → small step_size needed).

# %%
# MAP initialization for RT
t0 = time.perf_counter()
result_map = fitter.run("map", n_steps=1000, verbose=False)
t_map = time.perf_counter() - t0

# Ray Tracing
# step_size=0.002 for spectroscopy (40x smaller than photometry)
t0 = time.perf_counter()
result_rt = fitter.run(
    "raytrace",
    init_from=result_map,
    step_size=0.002,
    n_leapfrog_steps=50,
    n_burnin=200,
    n_steps=2000,
    verbose=True,
)
t_rt = time.perf_counter() - t0
n_samples_rt = len(next(iter(result_rt.samples.values())))
accept = result_rt.diagnostics.get("accept_rate", 0)
print(f"\nRay Tracing: {t_rt:.1f}s, {n_samples_rt} samples, "
      f"accept={accept:.1%}")

# %% [markdown]
# ### 5a. RT convergence

# %%
print("tengri check_convergence (RT):")
result_rt.check_convergence(verbose=True)

# %%
fig, ax = plt.subplots(figsize=(7, 4))
plot_sfh(model, result_rt, true_params=true_params, ax=ax,
         method="RT", label="Ray Tracing")
ax.set_title(f"Ray Tracing — {t_rt:.0f}s, {n_samples_rt} samples")
fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "02_sfh_rt.png"),
            dpi=150, bbox_inches="tight")
plt.show()

# %%
fig = safe_corner(result_rt, truths=true_params)
if fig is not None:
    fig.savefig(os.path.join(FIGDIR, "02_corner_rt.png"),
                dpi=150, bbox_inches="tight")
    plt.show()

# %% [markdown]
# ## 6. Three-way comparison

# %% [markdown]
# ### 6a. SFH recovery

# %%
fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))

for ax, result, label, color, t, ns in [
    (axes[0], result_geovi, "geoVI", COLORS["geovi"],
     t_geovi, n_samples_geovi),
    (axes[1], result_native_2, "native geoVI", COLORS["rt"],
     t_native_cached, n_samples_native),
    (axes[2], result_rt, "Ray Tracing", COLORS["nuts"],
     t_rt, n_samples_rt),
]:
    plot_sfh(model, result, true_params=true_params, ax=ax,
             color=color, label=label)
    ax.set_title(f"{label} — {t:.0f}s")

fig.suptitle(f"Stochastic SFH recovery (D={D})", y=1.02)
fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "02_sfh_comparison.png"),
            dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 6b. Corner overlay (geoVI vs native vs RT)

# %%
fig = plot_corner_comparison(
    [result_geovi, result_native_2, result_rt],
    labels=["geoVI", "native geoVI", "Ray Tracing"],
    colors=[COLORS["geovi"], COLORS["rt"], COLORS["nuts"]],
    truths=true_params,
)
if fig is not None:
    fig.savefig(os.path.join(FIGDIR, "02_corner_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.show()

# %% [markdown]
# ### 6c. Spectrum fits

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

for ax, result, label, color, t in [
    (axes[0], result_geovi, "geoVI", COLORS["geovi"], t_geovi),
    (axes[1], result_native_2, "native geoVI", COLORS["rt"],
     t_native_cached),
    (axes[2], result_rt, "Ray Tracing", COLORS["nuts"], t_rt),
]:
    s = result.samples
    nd = min(50, len(list(s.values())[0]))
    for j in range(nd):
        draw = {k: v[j] for k, v in s.items()}
        pred = model.predict_spectrum(draw, WAVE_OBS)
        ax.plot(np.array(WAVE_OBS), np.array(pred),
                color=color, alpha=0.06, lw=0.5)
    ax.errorbar(np.array(WAVE_OBS), np.array(mock.flux_obs),
                yerr=np.array(mock.noise), fmt=".", ms=2,
                color=COLORS["data"], alpha=0.4, zorder=5)
    ax.plot(np.array(WAVE_OBS), np.array(mock.flux_true),
            color=COLORS["truth"], lw=1.2, label="Truth")
    ax.set_xlabel("Observed wavelength [Å]")
    ax.set_ylabel("Flux density")
    ax.set_title(f"{label} — {t:.1f}s")
    ax.legend(fontsize=8)

fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "02_spectrum_fit.png"), dpi=150)
plt.show()

# %% [markdown]
# ## 7. Batch: 10 galaxies — native_geovi
#
# Engine cached on Model from §4. Each new Fitter reuses it.

# %%
N_BATCH = 10
batch_keys = jax.random.split(jax.random.PRNGKey(777), N_BATCH)

batch_mocks = []
batch_truths = []
for bk in batch_keys:
    true_i = spec.sample(bk)
    mock_i = model.mock_spectrum(true_i, WAVE_OBS, snr=30.0, key=bk)
    batch_mocks.append(mock_i)
    batch_truths.append(true_i)

# %%
times_batch = []
results_batch = []

for i, (mock_i, bk) in enumerate(zip(batch_mocks, batch_keys)):
    f_i = Fitter(model, mock_i.flux_obs, mock_i.noise)
    t0 = time.perf_counter()
    r_i = f_i.run("native_geovi", n_iterations=15, n_seeds=1,
                   key=bk, verbose=False)
    dt = time.perf_counter() - t0
    times_batch.append(dt)
    results_batch.append(r_i)
    print(f"  Galaxy {i+1:>2d}: {dt:.1f}s")

# %%
print("\n" + "=" * 50)
print(f"Batch {N_BATCH} galaxies — native_geovi (MAP init)")
print("-" * 50)
print(f"Compile (1st call, §4):        {t_native_compile:.1f}s")
print(f"Per galaxy (cached):           {np.mean(times_batch):.1f}s")
print(f"Total ({N_BATCH} galaxies):           "
      f"{np.sum(times_batch):.0f}s")
amort = (t_native_compile + np.sum(times_batch)) / (N_BATCH + 1)
print(f"Amortized (incl compile):      {amort:.1f}s")
print("=" * 50)

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.bar(range(1, N_BATCH + 1), times_batch, color=COLORS["rt"])
ax1.axhline(t_native_cached, color="grey", ls=":", lw=1,
            label=f"Same-fitter cached: {t_native_cached:.0f}s")
ax1.set_xlabel("Galaxy")
ax1.set_ylabel("Wall time [s]")
ax1.set_title("Per-galaxy (engine cached on Model)")
ax1.legend(fontsize=8)

cum = np.cumsum(times_batch)
amort_curve = (t_native_compile + cum) / np.arange(1, N_BATCH + 1)
ax2.plot(range(1, N_BATCH + 1), amort_curve, "o-", color=COLORS["rt"],
         label="Amortized (incl compile)")
ax2.axhline(np.mean(times_batch), color="grey", ls=":", lw=1,
            label=f"Marginal: {np.mean(times_batch):.1f}s")
ax2.set_xlabel("Galaxies fitted")
ax2.set_ylabel("Cost per galaxy [s]")
ax2.set_title("Compile cost amortizes to zero")
ax2.legend(fontsize=8)
ax2.set_ylim(0, None)

fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "02_batch_timing.png"), dpi=150)
plt.show()

# %% [markdown]
# ### 7a. Batch SFH gallery

# %%
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
show_idx = [0, 3, 6, 9]

for ax, idx in zip(axes.flatten(), show_idx):
    plot_sfh(model, results_batch[idx],
             true_params=batch_truths[idx], ax=ax,
             method="geoVI", label="native geoVI",
             color=COLORS["rt"])
    ax.set_title(f"Galaxy {idx+1} — {times_batch[idx]:.1f}s")

fig.suptitle(f"Batch SFH recovery — native geoVI (D={D})", y=1.02)
fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "02_batch_sfh_gallery.png"),
            dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 8. Timing summary

# %%
print("=" * 60)
print(f"{'Method':<30s} {'Time':>8s} {'Samples':>8s}")
print("-" * 60)
print(f"{'geoVI (NIFTy)':<30s} {t_geovi:>7.1f}s "
      f"{n_samples_geovi:>8d}")
print(f"{'native_geovi (compile)':<30s} "
      f"{t_native_compile:>7.1f}s {n_samples_native:>8d}")
print(f"{'native_geovi (cached)':<30s} "
      f"{t_native_cached:>7.1f}s {n_samples_native:>8d}")
print(f"{'Ray Tracing':<30s} {t_rt:>7.1f}s "
      f"{n_samples_rt:>8d}")
print("-" * 60)
print(f"{'Batch 10 (mean/galaxy)':<30s} "
      f"{np.mean(times_batch):>7.1f}s")
print(f"{'Batch 10 (total)':<30s} "
      f"{np.sum(times_batch):>7.0f}s")
print("=" * 60)
