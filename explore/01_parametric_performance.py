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
# # Parametric SFH Fitting
#
# Standard parametric SFH (truncated skew-normal, D = 8).
#
# **Two inference methods compared:**
# 1. `NUTS` — gold-standard HMC, exact posterior
# 2. `geoVI` (NIFTy) — variational, fast default
#
# Shows: SFH recovery, corner plots, spectrum fits,
# convergence diagnostics, and batch timing for 10 galaxies.

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
    SEDModel,
    Observation,
    Parameters,
    Uniform,
    load_ssp_data,
)
from tengri import Spectroscopy  # noqa: E402

from _plot_style import (  # noqa: E402
    COLORS,
    plot_corner_comparison,
    plot_sfh,
    safe_corner,
    setup_style,
)

setup_style()

sys.path.insert(0, _here)
from acor import autocorrelation_fft, batch_acor, split_rhat  # noqa: E402

FIGDIR = os.path.join("explore", "figures")
os.makedirs(FIGDIR, exist_ok=True)

# %% [markdown]
# ## 1. Setup

# %%
ssp_data = load_ssp_data(
    "data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5",
)
WAVE_OBS = jnp.linspace(3800.0, 9200.0, 200)
obs = Observation(spectroscopy=Spectroscopy(wave_obs=WAVE_OBS))

# %% [markdown]
# ## 2. Parametric model (D = 8)

# %%
spec = Parameters(
    sfh_tsnorm_log_peak_sfr=Uniform(-1.0, 2.5),
    sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),
    sfh_tsnorm_width_gyr=Uniform(0.3, 5.0),
    sfh_tsnorm_skew=Uniform(-3.0, 3.0),
    sfh_tsnorm_trunc=Uniform(1.0, 10.0),
    met_logzsol=Uniform(-2.0, 0.2),
    dust_tau_bc=Uniform(0.0, 2.0),
    dust_tau_diff=Uniform(0.0, 1.5),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.1),
    mean_sfh_type="tsnorm",
)
model = SEDModel(spec, ssp_data, observation=obs)
D = len(spec.free_params)
print(f"D = {D} free parameters: {spec.free_params}")

# %%
key = jax.random.PRNGKey(42)
true_params = {**spec.sample(key)}
true_params["sfh_tsnorm_log_peak_sfr"] = jnp.array(2.0)  # SFR ~ 100 Msun/yr
true_params["sfh_tsnorm_peak_lbt_gyr"] = jnp.array(3.0)
true_params["sfh_tsnorm_width_gyr"] = jnp.array(3.0)
true_params["sfh_tsnorm_skew"] = jnp.array(0.3)
true_params["sfh_tsnorm_trunc"] = jnp.array(2.0)
mock = model.mock_spectrum(true_params, WAVE_OBS, snr=30.0, key=key)

# %% [markdown]
# ## 3. NUTS (gold-standard HMC)
#
# Exact posterior via Hamiltonian Monte Carlo. Best for low-D parametric
# models where you want the ground-truth posterior.

# %%
fitter = Fitter(model, mock.flux_obs, mock.noise)

t0 = time.perf_counter()
fitter.compile(verbose=False)
t_compile = time.perf_counter() - t0

t0 = time.perf_counter()
result_map = fitter.run("map", n_steps=500, verbose=False)
t_map = time.perf_counter() - t0

t0 = time.perf_counter()
result_nuts = fitter.run(
    "nuts",
    init_from=result_map,
    n_warmup=500,
    n_samples=500,
    target_accept_rate=0.85,
    verbose=False,
)
t_nuts = time.perf_counter() - t0

print(f"Compile: {t_compile:.1f}s  |  MAP: {t_map:.1f}s  |  "
      f"NUTS: {t_nuts:.1f}s")

# %% [markdown]
# ### 3a. NUTS convergence

# %%
print("tengri check_convergence (NUTS):")
result_nuts.check_convergence(verbose=True)

# %%
# External FFT cross-check + split R-hat
samples_nuts = result_nuts.samples
scalar = {
    k: np.asarray(v) for k, v in samples_nuts.items()
    if np.asarray(v).ndim == 1 and np.var(np.asarray(v)) > 1e-30
}
names = sorted(scalar.keys())
arr2d = np.column_stack([scalar[k] for k in names])
taus, converged = batch_acor(arr2d, c=5.0)
rhats = split_rhat(arr2d)
n_samp = arr2d.shape[0]

print(f"\n{'Parameter':<30s} {'τ(FFT)':>7s} {'ESS':>6s} "
      f"{'R-hat':>7s} {'OK?':>5s}")
print("-" * 57)
for i, name in enumerate(names):
    ess = n_samp / (2.0 * taus[i]) if taus[i] > 0 else n_samp
    ok = converged[i] and rhats[i] < 1.05
    print(f"{name:<30s} {taus[i]:>7.1f} {ess:>6.0f} "
          f"{rhats[i]:>7.4f} {'OK' if ok else 'WARN':>5s}")

# %% [markdown]
# ### 3b. NUTS SFH recovery

# %%
fig, ax = plt.subplots(figsize=(7, 4))
plot_sfh(model, result_nuts, true_params=true_params, ax=ax,
         method="NUTS", label="NUTS")
ax.set_title(f"SFH recovery — NUTS (D={D}, {t_nuts:.0f}s)")
fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "01_sfh_nuts.png"),
            dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 3c. NUTS corner plot

# %%
fig = safe_corner(result_nuts, truths=true_params)
if fig is not None:
    fig.savefig(os.path.join(FIGDIR, "01_corner_nuts.png"),
                dpi=150, bbox_inches="tight")
    plt.show()

# %% [markdown]
# ## 4. geoVI — NIFTy fast path (default)
#
# Geometric variational inference. Much faster than NUTS, produces
# approximate posteriors. Default method for production fitting.

# %%
t0 = time.perf_counter()
result_geovi = fitter.run(
    "geovi", n_iterations=10, n_posterior_samples=500, verbose=True,
)
t_geovi = time.perf_counter() - t0
print(f"\ngeoVI (NIFTy): {t_geovi:.1f}s")

# %% [markdown]
# ### 4a. geoVI results

# %%
result_geovi.summary_table()

# %% [markdown]
# ### 4b. geoVI SFH recovery

# %%
fig, ax = plt.subplots(figsize=(7, 4))
plot_sfh(model, result_geovi, true_params=true_params, ax=ax,
         method="geoVI", label="geoVI")
ax.set_title(f"SFH recovery — geoVI (D={D}, {t_geovi:.0f}s)")
fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "01_sfh_geovi.png"),
            dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 4c. geoVI corner plot

# %%
fig = safe_corner(result_geovi, truths=true_params)
if fig is not None:
    fig.savefig(os.path.join(FIGDIR, "01_corner_geovi.png"),
                dpi=150, bbox_inches="tight")
    plt.show()

# %% [markdown]
# ## 5. Comparison — NUTS vs geoVI

# %% [markdown]
# ### 5a. SFH side-by-side

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
plot_sfh(model, result_nuts, true_params=true_params, ax=ax1,
         method="NUTS", label="NUTS")
ax1.set_title(f"NUTS — {t_nuts:.0f}s")
plot_sfh(model, result_geovi, true_params=true_params, ax=ax2,
         method="geoVI", label="geoVI")
ax2.set_title(f"geoVI — {t_geovi:.0f}s")
fig.suptitle(f"Parametric SFH recovery (D={D})", y=1.02)
fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "01_sfh_comparison.png"),
            dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 5b. Corner overlay

# %%
fig = plot_corner_comparison(
    [result_nuts, result_geovi],
    labels=["NUTS", "geoVI"],
    colors=[COLORS["nuts"], COLORS["geovi"]],
    truths=true_params,
)
if fig is not None:
    fig.savefig(os.path.join(FIGDIR, "01_corner_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.show()

# %% [markdown]
# ### 5c. Spectrum fits

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, result, label, color, t in [
    (axes[0], result_nuts, "NUTS", COLORS["nuts"], t_nuts),
    (axes[1], result_geovi, "geoVI", COLORS["geovi"], t_geovi),
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
    ax.set_title(f"{label} (D={D}) — {t:.0f}s")
    ax.legend(fontsize=8)

fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "01_spectrum_fit.png"), dpi=150)
plt.show()

# %% [markdown]
# ## 6. Batch: 10 galaxies

# %%
N_BATCH = 10
batch_keys = jax.random.split(jax.random.PRNGKey(999), N_BATCH)

times_nuts = []
times_geovi = []

for i, bk in enumerate(batch_keys):
    true_i = spec.sample(bk)
    mock_i = model.mock_spectrum(true_i, WAVE_OBS, snr=30.0, key=bk)
    f_i = Fitter(model, mock_i.flux_obs, mock_i.noise)

    # NUTS
    t0 = time.perf_counter()
    r_map = f_i.run("map", n_steps=500, verbose=False)
    r_nuts = f_i.run("nuts", init_from=r_map, n_warmup=500,
                      n_samples=500, target_accept_rate=0.85,
                      verbose=False)
    dt_nuts = time.perf_counter() - t0
    times_nuts.append(dt_nuts)

    # geoVI
    t0 = time.perf_counter()
    r_geovi = f_i.run("geovi", n_iterations=10,
                       n_posterior_samples=500, verbose=False)
    dt_geovi = time.perf_counter() - t0
    times_geovi.append(dt_geovi)

    print(f"  Galaxy {i+1:>2d}: NUTS={dt_nuts:.1f}s  "
          f"geoVI={dt_geovi:.1f}s")

# %%
print("\n" + "=" * 55)
print(f"{'Method':<20s} {'Mean/gal':>10s} {'Total':>10s}")
print("-" * 55)
print(f"{'NUTS':<20s} "
      f"{np.mean(times_nuts):>9.1f}s "
      f"{np.sum(times_nuts):>9.0f}s")
print(f"{'geoVI (NIFTy)':<20s} "
      f"{np.mean(times_geovi):>9.1f}s "
      f"{np.sum(times_geovi):>9.0f}s")
print("=" * 55)

# %%
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(1, N_BATCH + 1)
w = 0.35
ax.bar(x - w / 2, times_nuts, w, label="NUTS", color=COLORS["nuts"])
ax.bar(x + w / 2, times_geovi, w, label="geoVI",
       color=COLORS["geovi"])
ax.set_xlabel("Galaxy")
ax.set_ylabel("Wall time [s]")
ax.set_title(f"Batch {N_BATCH} galaxies — NUTS vs geoVI")
ax.legend(fontsize=8)
fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "01_batch_comparison.png"), dpi=150)
plt.show()

# %% [markdown]
# ## 7. Timing summary

# %%
print("=" * 50)
print(f"{'Scenario':<30s} {'Time':>10s}")
print("-" * 50)
print(f"{'NUTS (single galaxy)':<30s} {t_nuts:>9.1f}s")
print(f"{'geoVI (single galaxy)':<30s} {t_geovi:>9.1f}s")
print(f"{'NUTS batch (mean/gal)':<30s} "
      f"{np.mean(times_nuts):>9.1f}s")
print(f"{'geoVI batch (mean/gal)':<30s} "
      f"{np.mean(times_geovi):>9.1f}s")
print(f"{'Compile (one-time)':<30s} {t_compile:>9.1f}s")
print("=" * 50)
