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
# # Computational Benchmarks (Paper §4.4)
#
# Wall-clock performance of all five inference methods on both the
# smooth (D~7) and stochastic (D~137) configurations.
#
# **Paper figures generated:**
# - **Fig 8**: Wall-clock inference time (5 methods × 2 configs)
# - **Fig 10**: Sampler comparison in posterior space (appendix)

# %%
import time, os
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
import numpy as np

from tengri import (
    SEDModel, ParamSpec, Uniform, Fixed, Fitter,
    load_ssp_data, load_filter_set,
)
import sys; sys.path.insert(0, ".")
from _plot_style import (setup_style, COLORS, SAMPLER_STYLE,
                          plot_sfh_comparison, plot_corner_comparison,
                          convergence_table)
setup_style()

FIG_DIR = "notebook_figures"
os.makedirs(FIG_DIR, exist_ok=True)
def savefig(fig, name):
    fig.savefig(os.path.join(FIG_DIR, f"A04_{name}.png"),
                bbox_inches="tight", dpi=72)

ssp_data = load_ssp_data(
    "../data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")
filters = load_filter_set(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])

# %% [markdown]
# ## Smooth Configuration (D ~ 7)

# %%
spec_smooth = ParamSpec(
    sfh_alpha=Uniform(0.5, 3.0), sfh_beta=Uniform(0.5, 3.0),
    sfh_tau_peak_gyr=Uniform(0.5, 13.0), sfh_peak_sfr=Uniform(0.1, 100.0),
    psd_sigma=Fixed(0.0), psd_tau_myr=Fixed(50.0),
    met_logzsol=Uniform(-2.0, 0.5),
    dust_tau_bc=Uniform(0.0, 2.0), dust_tau_diff=Uniform(0.0, 2.0),
    dust_slope=Fixed(-0.7), redshift=Fixed(0.1), stochastic=False,
)
model_sm = SEDModel(spec_smooth, ssp_data, filters=filters)
true_sm = dict(sfh_alpha=1.0, sfh_beta=1.5, sfh_tau_peak_gyr=8.0,
               sfh_peak_sfr=30.0, met_logzsol=-0.3,
               dust_tau_bc=0.5, dust_tau_diff=0.3)
mock_sm = model_sm.mock(true_sm, snr=20.0, key=jax.random.PRNGKey(10))
fitter_sm = Fitter(model_sm, mock_sm.flux_obs, mock_sm.noise, data_type="photometry")

timings_sm = {}
results_sm = {}

for method, kwargs in [
    ("map", {"n_steps": 500}),
    ("raytrace", {"n_burnin": 200, "n_steps": 2000}),
    ("nuts", {"n_warmup": 1000, "n_samples": 1000}),
    ("geovi", {"n_iterations": 25, "n_samples": 6}),
]:
    t0 = time.perf_counter()
    if method in ("raytrace", "nuts", "geovi"):
        init = results_sm.get("map")
        res = fitter_sm.run(method, init_from=init, **kwargs)
    else:
        res = fitter_sm.run(method, **kwargs)
    dt = time.perf_counter() - t0
    timings_sm[method] = dt
    results_sm[method] = res
    print(f"  {method}: {dt:.1f} s")

# %% [markdown]
# ## Stochastic Configuration (D ~ 137)

# %%
spec_stoch = ParamSpec(
    sfh_alpha=Uniform(0.5, 3.0), sfh_beta=Uniform(0.5, 3.0),
    sfh_tau_peak_gyr=Uniform(0.5, 13.0), sfh_peak_sfr=Uniform(0.1, 100.0),
    psd_sigma=Uniform(0.1, 4.0), psd_tau_myr=Uniform(1.0, 300.0),
    met_logzsol=Uniform(-2.0, 0.5),
    dust_tau_bc=Uniform(0.0, 2.0), dust_tau_diff=Uniform(0.0, 2.0),
    dust_slope=Fixed(-0.7), redshift=Fixed(0.1),
    stochastic=True, n_grid=128,
)
model_st = SEDModel(spec_stoch, ssp_data, filters=filters)
key = jax.random.PRNGKey(20)
true_st = spec_stoch.sample(key)
true_st.update(sfh_alpha=1.0, sfh_beta=1.5, sfh_tau_peak_gyr=8.0,
               sfh_peak_sfr=30.0, psd_sigma=2.0, psd_tau_myr=20.0,
               met_logzsol=-0.3, dust_tau_bc=0.5, dust_tau_diff=0.3)
mock_st = model_st.mock(true_st, snr=20.0, key=key)
fitter_st = Fitter(model_st, mock_st.flux_obs, mock_st.noise, data_type="photometry")

timings_st = {}
results_st = {}

for method, kwargs in [
    ("map", {"n_steps": 1000}),
    ("raytrace", {"n_burnin": 200, "n_steps": 2000,
                  "step_size": 0.005, "n_leapfrog_steps": 200}),
    ("geovi", {"n_iterations": 25, "n_samples": 6}),
]:
    t0 = time.perf_counter()
    if method in ("raytrace", "geovi"):
        init = results_st.get("map")
        res = fitter_st.run(method, init_from=init, **kwargs)
    else:
        res = fitter_st.run(method, **kwargs)
    dt = time.perf_counter() - t0
    timings_st[method] = dt
    results_st[method] = res
    print(f"  {method}: {dt:.1f} s")

# %% [markdown]
# ## Paper Figure 8: Wall-Clock Timing

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

method_colors = {"map": COLORS["map"], "raytrace": COLORS["rt"],
                 "nuts": COLORS["nuts"], "geovi": COLORS["geovi"]}

# Smooth
ax = axes[0]
methods = list(timings_sm.keys())
times = [timings_sm[m] for m in methods]
bars = ax.barh(methods, times,
               color=[method_colors.get(m, "0.5") for m in methods])
ax.set_xlabel("Wall-clock time [s]"); ax.set_title(f"Smooth ($D={spec_smooth.n_free}$)")
for bar, t in zip(bars, times):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f"{t:.1f}s", va="center", fontsize=9)

# Stochastic
ax = axes[1]
methods = list(timings_st.keys())
times = [timings_st[m] for m in methods]
bars = ax.barh(methods, times,
               color=[method_colors.get(m, "0.5") for m in methods])
ax.set_xlabel("Wall-clock time [s]"); ax.set_title(f"Stochastic ($D\\approx{spec_stoch.n_free}$)")
for bar, t in zip(bars, times):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f"{t:.1f}s", va="center", fontsize=9)

fig.suptitle("Paper Figure 8: Inference Wall-Clock Time", fontsize=14, y=1.02)
fig.tight_layout(); savefig(fig, "paper_fig08_benchmarks"); plt.show()

# %% [markdown]
# ## Convergence Summary

# %%
print("=== Smooth configuration ===")
convergence_table({k: v for k, v in results_sm.items() if k != "map"})

print("\n=== Stochastic configuration ===")
convergence_table({k: v for k, v in results_st.items() if k != "map"})

# %% [markdown]
# ## Paper Figure 10: Sampler Comparison (Appendix)

# %%
fig = plot_corner_comparison(
    [results_sm["raytrace"], results_sm["geovi"], results_sm["nuts"]],
    ["Ray Tracing", "geoVI", "NUTS"],
    colors=[COLORS["rt"], COLORS["geovi"], COLORS["nuts"]],
    truths=true_sm,
)
if fig is not None:
    fig.suptitle("Paper Figure 10: Sampler Comparison (Smooth Config)",
                 fontsize=12, y=1.02)
    savefig(fig, "paper_fig10_sampler_comparison")
plt.show()
