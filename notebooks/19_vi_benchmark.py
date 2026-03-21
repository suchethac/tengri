# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # VI Methods Benchmark: Speed, Accuracy, and Compilation
#
# Comprehensive comparison of all variational inference methods in diffsed.
# For each method we measure:
# - **Compile time** (one-time JIT/XLA cost)
# - **Run time** (per-galaxy execution after compilation)
# - **Hamiltonian** H at convergence (lower = better fit)
# - **Chi2/dof** (should be ~1 for a good fit)
# - **Posterior predictive** (predicted data should bracket observed)

# %%
import logging
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)
logging.getLogger("nifty8").setLevel(logging.ERROR)

from diffsed import Fitter, Model, ParamSpec, Uniform
from diffsed.models.observation.filters import load_filter_set
from diffsed.models.sps.dsps_wrapper import load_ssp_data

# %% [markdown]
# ## Setup

# %%
ssp = load_ssp_data("../data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")
filters = load_filter_set(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])

spec = ParamSpec(
    mean_sfh_type="tsnorm",
    sfh_tsnorm_log_peak_sfr=Uniform(-1.0, 2.5),
    sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),
    sfh_tsnorm_width_gyr=Uniform(0.2, 5.0),
    sfh_tsnorm_skew=Uniform(-1.0, 1.0),
    sfh_tsnorm_trunc=Uniform(1.0, 10.0),
    met_logzsol=Uniform(-1.5, 0.2),
    dust_tau_bc=Uniform(0.0, 3.0),
    dust_tau_diff=Uniform(0.0, 2.0),
    dust_slope=-0.7,
    redshift=0.1,
)

model = Model(spec, ssp, filters=filters)
key = jax.random.PRNGKey(42)
true_params = spec.sample(key)
mock = model.mock(true_params, snr=20.0, key=key)
fitter = Fitter(model, mock.flux_obs, mock.noise, data_type="photometry")

D = len(spec.free_params)
N = len(mock.flux_obs)
print(f"D = {D} free parameters, N = {N} data points")

# %% [markdown]
# ## Benchmark All Methods
#
# For each method:
# 1. **Cold run** (includes compilation): measures total wall time on first call
# 2. **Warm run** (cached): measures pure execution time
# 3. Compute H and chi2/dof from the result

# %%
N_ITER = 15
N_SAMP = 3
N_POST = 200

methods = {
    # --- Default (geovi with optimal resample+update schedule) ---
    "geovi (default)": {"method": "geovi"},
    # --- Fast: NIFTy tight loop ---
    "fast_mgvi": {"method": "fast_mgvi"},
    "fast_evi": {"method": "fast_evi"},
    "fast_geovi": {"method": "fast_geovi"},
    # --- Full NIFTy ---
    "nifty_geovi": {"method": "nifty_geovi"},
    # --- Native JIT: linear ---
    "native_mgvi": {"method": "native_mgvi", "n_seeds": 1},
    # --- Native JIT: geoVI ---
    "native_geovi": {"method": "native_geovi", "n_seeds": 1},
    # --- MAP (point estimate, for reference) ---
    "map": {"method": "map", "n_steps": 500},
}

results = {}
timings = {}

for label, kwargs in methods.items():
    method = kwargs.pop("method")
    common = {}
    if method not in ("map",):
        common = {
            "n_iterations": N_ITER,
            "n_samples": N_SAMP,
            "n_posterior_samples": N_POST,
        }

    # Run twice: first = cold (may include compilation), second = warm (cached)
    try:
        t0 = time.time()
        r = fitter.run(method, verbose=False, key=jax.random.PRNGKey(42), **common, **kwargs)
        t_cold = time.time() - t0

        t0 = time.time()
        r = fitter.run(method, verbose=False, key=jax.random.PRNGKey(42), **common, **kwargs)
        t_warm = time.time() - t0
    except Exception as e:
        print(f"{label:30s}  FAILED: {e}")
        kwargs["method"] = method
        continue

    t_compile = max(0, t_cold - t_warm)

    # Compute chi2/dof
    chi2_dof = None
    try:
        if hasattr(r, "params") and isinstance(r.params, dict) and r.params:
            pred = model.predict_photometry(r.params)
            chi2 = float(jnp.sum(((mock.flux_obs - pred) / mock.noise) ** 2))
            chi2_dof = chi2 / N
    except Exception:
        pass

    results[label] = r
    timings[label] = {
        "cold": t_cold,
        "warm": t_warm,
        "compile": t_compile,
        "chi2_dof": chi2_dof,
    }

    chi2_str = f"chi2/dof={chi2_dof:.2f}" if chi2_dof is not None else "chi2/dof=—"
    print(
        f"{label:30s}  compile={t_compile:6.1f}s  run={t_warm:6.1f}s  total={t_cold:6.1f}s  {chi2_str}"
    )

    # Restore kwargs for next iteration
    kwargs["method"] = method

# %% [markdown]
# ## Timing Comparison

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

labels = list(timings.keys())
compile_times = [timings[l]["compile"] for l in labels]
run_times = [timings[l]["warm"] for l in labels]

y = np.arange(len(labels))
ax1.barh(y, compile_times, height=0.4, label="Compile (one-time)", color="C1", alpha=0.8)
ax1.barh(y + 0.4, run_times, height=0.4, label="Run (per galaxy)", color="C0", alpha=0.8)
ax1.set_yticks(y + 0.2)
ax1.set_yticklabels(labels, fontsize=9)
ax1.set_xlabel("Time (seconds)")
ax1.set_title("Compile vs Run Time")
ax1.legend(fontsize=9)
ax1.invert_yaxis()

# Log scale version
ax2.barh(
    y, [max(0.001, c) for c in compile_times], height=0.4, label="Compile", color="C1", alpha=0.8
)
ax2.barh(
    y + 0.4, [max(0.001, r) for r in run_times], height=0.4, label="Run", color="C0", alpha=0.8
)
ax2.set_yticks(y + 0.2)
ax2.set_yticklabels(labels, fontsize=9)
ax2.set_xlabel("Time (seconds, log scale)")
ax2.set_xscale("log")
ax2.set_title("Compile vs Run Time (log scale)")
ax2.legend(fontsize=9)
ax2.invert_yaxis()

fig.tight_layout()
plt.show()

# %% [markdown]
# ## Accuracy Comparison: Chi2/dof

# %%
fig, ax = plt.subplots(figsize=(8, 4))
labels_with_chi2 = [l for l in labels if timings[l]["chi2_dof"] is not None]
chi2s = [timings[l]["chi2_dof"] for l in labels_with_chi2]
colors = ["C2" if c < 3 else "C1" if c < 10 else "C3" for c in chi2s]

bars = ax.barh(labels_with_chi2, chi2s, color=colors, alpha=0.8)
ax.axvline(1.0, color="k", ls="--", alpha=0.5, label="Ideal (chi2/dof=1)")
ax.set_xlabel("Chi2 / dof")
ax.set_title(f"Fit Quality ({N_ITER} iterations, {N_SAMP} samples)")
ax.legend()

for bar, c in zip(bars, chi2s):
    ax.text(
        bar.get_width() + 0.1,
        bar.get_y() + bar.get_height() / 2,
        f"{c:.2f}",
        va="center",
        fontsize=9,
    )

ax.invert_yaxis()
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Posterior Predictive Check

# %%
band_names = ["u", "g", "r", "i", "z"]
band_wave = np.array([3551, 4686, 6166, 7480, 8932])

vi_methods = [
    l for l in labels if l != "map" and isinstance(results[l].samples, dict) and results[l].samples
]
n_vi = len(vi_methods)

fig, axes = plt.subplots(1, n_vi, figsize=(3.5 * n_vi, 4), sharey=True)
if n_vi == 1:
    axes = [axes]

for ax, label in zip(axes, vi_methods):
    result = results[label]
    first_key = next(iter(result.samples))
    n_samp = min(100, len(result.samples[first_key]))

    predictions = []
    for i in range(n_samp):
        sample = {k: v[i] for k, v in result.samples.items()}
        pred = model.predict_photometry(sample)
        predictions.append(np.array(pred))

    if not predictions:
        continue

    predictions = np.stack(predictions)
    pred_med = np.median(predictions, axis=0)
    pred_lo = np.percentile(predictions, 16, axis=0)
    pred_hi = np.percentile(predictions, 84, axis=0)

    ax.fill_between(band_wave, pred_lo, pred_hi, alpha=0.3, color="C0", label="68% CI")
    ax.plot(band_wave, pred_med, "o-", color="C0", ms=4, label="Median pred")
    ax.errorbar(
        band_wave,
        np.array(mock.flux_obs),
        yerr=np.array(mock.noise),
        fmt="s",
        color="k",
        ms=5,
        label="Observed",
    )

    chi2 = timings[label]["chi2_dof"]
    chi2_str = f"$\\chi^2$/dof={chi2:.1f}" if chi2 else ""
    t_str = f"run={timings[label]['warm']:.1f}s"
    ax.set_title(f"{label}\n{chi2_str}  {t_str}", fontsize=10)
    ax.set_xlabel("Wavelength (A)")
    if ax == axes[0]:
        ax.set_ylabel("Flux density")
    ax.legend(fontsize=7)

fig.suptitle("Posterior Predictive Check", fontsize=13, y=1.02)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Parameter Recovery

# %%
free_names = list(spec.free_params)
n_params = len(free_names)
vi_methods_for_params = [
    l for l in labels if l != "map" and isinstance(results[l].samples, dict) and results[l].samples
]

fig, axes = plt.subplots(n_params, 1, figsize=(10, 2.2 * n_params), sharex=True)

for i, pname in enumerate(free_names):
    ax = axes[i]
    true_val = float(true_params[pname])

    for j, label in enumerate(vi_methods_for_params):
        result = results[label]
        if pname not in result.samples:
            continue
        vals = np.array(result.samples[pname])
        med = np.median(vals)
        lo, hi = np.percentile(vals, [16, 84])
        ax.errorbar(
            j,
            med,
            yerr=[[med - lo], [hi - med]],
            fmt="o",
            color=f"C{j}",
            ms=5,
            capsize=3,
            label=label if i == 0 else "",
        )

    ax.axhline(true_val, color="k", ls="--", alpha=0.5)
    ax.set_ylabel(pname, fontsize=8)
    ax.set_xticks(range(len(vi_methods_for_params)))
    ax.set_xticklabels(vi_methods_for_params, fontsize=7, rotation=20)

if n_params > 0:
    axes[0].legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.6))

fig.suptitle("Parameter Recovery (median +/- 68% CI, dashed = truth)", fontsize=12, y=1.01)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Summary Table

# %%
print(f"{'Method':<30s} {'Compile':>8s} {'Run':>8s} {'Total':>8s} {'Chi2/dof':>9s}")
print("-" * 68)
for label in labels:
    t = timings[label]
    chi2 = f"{t['chi2_dof']:.2f}" if t["chi2_dof"] else "—"
    print(f"{label:<30s} {t['compile']:7.1f}s {t['warm']:7.1f}s {t['cold']:7.1f}s {chi2:>9s}")

print()
print("Notes:")
print(f"  - {N_ITER} iterations, {N_SAMP} samples/iter, {N_POST} posterior samples")
print(f"  - D = {D} free params, N = {N} data points")
print("  - 'geovi (default)' uses optimal resample+update schedule")
print("  - Compile time is one-time; run time is per-galaxy (cached)")
print("  - native_mgvi compile is near-zero; native_geovi ~56s one-time")
