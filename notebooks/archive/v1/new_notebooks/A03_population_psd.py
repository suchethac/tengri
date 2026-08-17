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
# # Population-Level PSD Recovery (Paper §4.3)
#
# Tests 5–7: hierarchical inference sharing PSD hyperparameters across
# galaxy ensembles.
#
# **Paper figure generated:**
# - **Fig 7**: Population-level PSD recovery with N-scaling and
#   population distinction

# %%
import os, time
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
import numpy as np

from tengri import (
    SEDModel, ParamSpec, Uniform, Fixed,
    load_ssp_data, load_filter_set,
)
from tengri.inference.hierarchical import HierarchicalFitter

import sys; sys.path.insert(0, ".")
from _plot_style import setup_style, COLORS
setup_style()

FIG_DIR = "notebook_figures"
os.makedirs(FIG_DIR, exist_ok=True)
def savefig(fig, name):
    fig.savefig(os.path.join(FIG_DIR, f"A03_{name}.png"),
                bbox_inches="tight", dpi=72)

ssp_data = load_ssp_data(
    "../data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")
filters = load_filter_set(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])

spec = ParamSpec(
    sfh_alpha=Uniform(0.5, 3.0), sfh_beta=Uniform(0.5, 3.0),
    sfh_tau_peak_gyr=Uniform(0.5, 13.0), sfh_peak_sfr=Uniform(0.1, 100.0),
    psd_sigma=Uniform(0.1, 4.0), psd_tau_myr=Uniform(1.0, 300.0),
    met_logzsol=Uniform(-2.0, 0.5),
    dust_tau_bc=Uniform(0.0, 2.0), dust_tau_diff=Uniform(0.0, 2.0),
    dust_slope=Fixed(-0.7), redshift=Fixed(0.1),
    stochastic=True, n_grid=128,
)
model = SEDModel(spec, ssp_data, filters=filters)


def generate_population(N, sigma, tau_myr, seed):
    """Generate N mock galaxies with shared PSD parameters."""
    key = jax.random.PRNGKey(seed)
    mocks = []
    for i in range(N):
        key, subkey = jax.random.split(key)
        tp = spec.sample(subkey)
        tp.update(psd_sigma=sigma, psd_tau_myr=tau_myr,
                  sfh_alpha=1.0, sfh_beta=1.5, sfh_tau_peak_gyr=8.0,
                  sfh_peak_sfr=30.0, met_logzsol=-0.3,
                  dust_tau_bc=0.5, dust_tau_diff=0.3)
        mocks.append(model.mock(tp, snr=20.0, key=subkey))
    return mocks


# %% [markdown]
# ## Test 5: DRW PSD Recovery (N=500)

# %%
TRUE_SIGMA = 2.0
TRUE_TAU = 20.0
N_HIER = 500

pop = generate_population(N_HIER, TRUE_SIGMA, TRUE_TAU, seed=500)
print(f"Generated {N_HIER} galaxies (σ={TRUE_SIGMA}, τ={TRUE_TAU} Myr)")

hfitter = HierarchicalFitter(model, pop,
                              shared_params=["psd_sigma", "psd_tau_myr"])
t0 = time.perf_counter()
result = hfitter.run("geovi", n_iterations=25, n_samples=6)
t_hier = time.perf_counter() - t0
print(f"Hierarchical geoVI: {t_hier:.0f} s")

# %% [markdown]
# ## Test 7: N-Scaling

# %%
N_values = [50, 100, 200, 500]
sigma_widths, tau_widths = [], []

for N in N_values:
    hf = HierarchicalFitter(model, pop[:N],
                            shared_params=["psd_sigma", "psd_tau_myr"])
    res = hf.run("geovi", n_iterations=20, n_samples=6)
    sigma_widths.append(np.std(res.shared_samples["psd_sigma"]))
    tau_widths.append(np.std(res.shared_samples["psd_tau_myr"]))
    print(f"  N={N:3d}: Δσ={sigma_widths[-1]:.3f}, Δτ={tau_widths[-1]:.1f} Myr")

# %% [markdown]
# ## Test 6: Population Distinction

# %%
pop_moderate = generate_population(200, sigma=1.0, tau_myr=50.0, seed=600)
pop_bursty = generate_population(200, sigma=2.5, tau_myr=10.0, seed=700)

hf_mod = HierarchicalFitter(model, pop_moderate,
                            shared_params=["psd_sigma", "psd_tau_myr"])
res_mod = hf_mod.run("geovi", n_iterations=20, n_samples=6)

hf_bur = HierarchicalFitter(model, pop_bursty,
                            shared_params=["psd_sigma", "psd_tau_myr"])
res_bur = hf_bur.run("geovi", n_iterations=20, n_samples=6)

# %% [markdown]
# ## Paper Figure 7: Combined Panel

# %%
fig = plt.figure(figsize=(16, 5))
gs = fig.add_gridspec(1, 3, wspace=0.3)

# Panel 1: Hierarchical recovery (N=500)
ax = fig.add_subplot(gs[0])
ax.scatter(result.shared_samples["psd_sigma"],
           result.shared_samples["psd_tau_myr"],
           s=3, alpha=0.3, color=COLORS["rt"])
ax.plot(TRUE_SIGMA, TRUE_TAU, "x", ms=14, mew=3, color=COLORS["truth"],
        zorder=10, label="Truth")
ax.set_xlabel(r"$\sigma_{\rm PS}$"); ax.set_ylabel(r"$\tau_{\rm PS}$ [Myr]")
ax.set_title(f"DRW Recovery ($N={N_HIER}$)"); ax.legend()

# Panel 2: N-scaling
ax = fig.add_subplot(gs[1])
ax.loglog(N_values, sigma_widths, "o-", color=COLORS["rt"], lw=2, ms=8,
          label=r"$\sigma_{\rm PS}$")
ax.loglog(N_values, tau_widths, "s-", color=COLORS["geovi"], lw=2, ms=8,
          label=r"$\tau_{\rm PS}$")
ref_s = sigma_widths[0] * np.sqrt(N_values[0]) / np.sqrt(N_values)
ax.loglog(N_values, ref_s, "--", color="0.5", label=r"$\propto 1/\sqrt{N}$")
ax.set_xlabel("$N$ galaxies"); ax.set_ylabel("Posterior width")
ax.set_title(r"$\sqrt{N}$ Shrinkage"); ax.legend(fontsize=8)

# Panel 3: Population distinction
ax = fig.add_subplot(gs[2])
ax.scatter(res_mod.shared_samples["psd_sigma"],
           res_mod.shared_samples["psd_tau_myr"],
           s=3, alpha=0.3, color="C0", label="Moderate (σ=1, τ=50)")
ax.scatter(res_bur.shared_samples["psd_sigma"],
           res_bur.shared_samples["psd_tau_myr"],
           s=3, alpha=0.3, color="C3", label="Bursty (σ=2.5, τ=10)")
ax.plot(1.0, 50.0, "x", ms=12, mew=3, color="C0")
ax.plot(2.5, 10.0, "x", ms=12, mew=3, color="C3")
ax.set_xlabel(r"$\sigma_{\rm PS}$"); ax.set_ylabel(r"$\tau_{\rm PS}$ [Myr]")
ax.set_title("Population Distinction"); ax.legend(fontsize=8)

fig.suptitle("Paper Figure 7: Population-Level PSD Recovery",
             fontsize=14, y=1.02)
savefig(fig, "paper_fig07_population_psd")
plt.show()
