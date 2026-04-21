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
# # Individual SFH Recovery (Paper §4.2)
#
# Tests 1–3 from the paper: individual-galaxy SFH recovery across four
# PSD regimes and two data types (photometry and spectroscopy).
#
# **Paper figures generated:**
# - **Fig 5**: Individual SFH recovery (4 regimes × 2 data types)
# - **Fig 6**: Joint PSD posterior (σ_PS, τ_PS) for individual galaxies

# %%
import os, time
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np

from tengri import (
    SEDModel, ParamSpec, Uniform, Fixed, Fitter,
    load_ssp_data, load_filter_set,
)
import sys; sys.path.insert(0, ".")
from _plot_style import setup_style, COLORS, plot_sfh, plot_corner_comparison
setup_style()

FIG_DIR = "notebook_figures"
os.makedirs(FIG_DIR, exist_ok=True)
def savefig(fig, name):
    fig.savefig(os.path.join(FIG_DIR, f"A02_{name}.png"),
                bbox_inches="tight", dpi=72)

ssp_data = load_ssp_data(
    "../data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")
filters = load_filter_set(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])

PSD_REGIMES = {
    "Smooth":        {"sigma": 0.5, "tau_myr": 200, "color": "#1b9e77"},
    "Moderate":      {"sigma": 1.0, "tau_myr": 50,  "color": "#d95f02"},
    "Bursty":        {"sigma": 2.0, "tau_myr": 20,  "color": "#7570b3"},
    "Highly bursty": {"sigma": 3.0, "tau_myr": 5,   "color": "#e7298a"},
}


def add_sfh_inset(ax, t_gyr, sfr, inset_range_myr=200, **kw):
    ax_in = inset_axes(ax, width="35%", height="40%", loc="upper right",
                       borderpad=1.5)
    t_myr = np.asarray(t_gyr) * 1e3
    mask = t_myr <= inset_range_myr
    if mask.sum() > 2:
        ax_in.plot(t_myr[mask], np.asarray(sfr)[mask], **kw)
    ax_in.set_xlim(0, inset_range_myr)
    ax_in.set_xlabel("Myr", fontsize=6); ax_in.tick_params(labelsize=5)
    ax_in.axvspan(0, 10, alpha=0.08, color="blue", zorder=0)
    ax_in.axvspan(0, 100, alpha=0.04, color="purple", zorder=0)
    ax_in.set_title("200 Myr", fontsize=6, pad=1)
    return ax_in


# %% [markdown]
# ## Paper Figure 5: SFH Recovery (4 regimes × 2 data types)

# %%
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

fig, axes = plt.subplots(4, 2, figsize=(14, 16), sharex=True)

for row, (regime_name, r) in enumerate(PSD_REGIMES.items()):
    key = jax.random.PRNGKey(row * 100 + 7)
    tp = spec.sample(key)
    tp.update(sfh_alpha=1.0, sfh_beta=1.5, sfh_tau_peak_gyr=8.0,
              sfh_peak_sfr=30.0, psd_sigma=r["sigma"], psd_tau_myr=r["tau_myr"],
              met_logzsol=-0.3, dust_tau_bc=0.5, dust_tau_diff=0.3)

    sfh_true = model.predict_sfh(tp)

    for col, dtype in enumerate(["photometry", "spectroscopy"]):
        ax = axes[row, col]

        if dtype == "photometry":
            mock = model.mock(tp, snr=20.0, key=key, data_type="photometry")
        else:
            mock = model.mock(tp, snr=20.0, key=key, data_type="spectroscopy",
                              R=100, wave_range=(1000, 8000))

        fitter = Fitter(model, mock.flux_obs, mock.noise, data_type=dtype)
        map_res = fitter.run("map", n_steps=500)
        rt_res = fitter.run("raytrace", init_from=map_res,
                            n_burnin=100, n_steps=1000,
                            step_size=0.005, n_leapfrog_steps=100)

        plot_sfh(model, rt_res, true_params=tp, ax=ax, method="RT",
                 color=r["color"], label="RT posterior")

        # Inset
        add_sfh_inset(ax, sfh_true["t_gyr"], sfh_true["sfr_full"],
                      color=COLORS["truth"], lw=1.5)

        if row == 0:
            ax.set_title(dtype.capitalize(), fontsize=12, fontweight="bold")
        if col == 0:
            ax.set_ylabel(rf"{regime_name}" + "\n" +
                          r"SFR [M$_\odot$/yr]", fontsize=9)

    print(f"  {regime_name} done")

for ax in axes[-1]:
    ax.set_xlabel("Lookback time [Gyr]")

fig.suptitle("Paper Figure 5: Individual SFH Recovery", fontsize=14, y=1.01)
fig.tight_layout()
savefig(fig, "paper_fig05_sfh_recovery")
plt.show()

# %% [markdown]
# ## Paper Figure 6: Joint PSD Posterior (Individual)

# %%
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

for i, (regime_name, r) in enumerate(PSD_REGIMES.items()):
    ax = axes.flat[i]
    key = jax.random.PRNGKey(i * 100 + 7)
    tp = spec.sample(key)
    tp.update(psd_sigma=r["sigma"], psd_tau_myr=r["tau_myr"],
              sfh_alpha=1.0, sfh_beta=1.5, sfh_tau_peak_gyr=8.0,
              sfh_peak_sfr=30.0, met_logzsol=-0.3,
              dust_tau_bc=0.5, dust_tau_diff=0.3)

    # Photometry fit
    mock_p = model.mock(tp, snr=20.0, key=key, data_type="photometry")
    fitter_p = Fitter(model, mock_p.flux_obs, mock_p.noise, data_type="photometry")
    map_p = fitter_p.run("map", n_steps=500)
    rt_p = fitter_p.run("raytrace", init_from=map_p, n_burnin=100,
                        n_steps=1000, step_size=0.005, n_leapfrog_steps=100)

    # Spectroscopy fit
    mock_s = model.mock(tp, snr=20.0, key=key, data_type="spectroscopy",
                        R=100, wave_range=(1000, 8000))
    fitter_s = Fitter(model, mock_s.flux_obs, mock_s.noise, data_type="spectroscopy")
    map_s = fitter_s.run("map", n_steps=500)
    rt_s = fitter_s.run("raytrace", init_from=map_s, n_burnin=100,
                        n_steps=1000, step_size=0.005, n_leapfrog_steps=100)

    # Plot joint posteriors
    ax.scatter(rt_p.samples["psd_sigma"], rt_p.samples["psd_tau_myr"],
               s=2, alpha=0.2, color=COLORS["rt"], label="Photometry")
    ax.scatter(rt_s.samples["psd_sigma"], rt_s.samples["psd_tau_myr"],
               s=2, alpha=0.2, color="#d62728", label="Spectroscopy")
    ax.axvline(r["sigma"], color=COLORS["truth"], lw=1.5, ls="--")
    ax.axhline(r["tau_myr"], color=COLORS["truth"], lw=1.5, ls="--")
    ax.plot(r["sigma"], r["tau_myr"], "x", ms=10, mew=3,
            color=COLORS["truth"], zorder=10)
    ax.set_title(regime_name, fontsize=10)
    ax.set_xlabel(r"$\sigma_{\rm PS}$"); ax.set_ylabel(r"$\tau_{\rm PS}$ [Myr]")
    if i == 0:
        ax.legend(fontsize=8)

fig.suptitle("Paper Figure 6: Joint PSD Posterior (Individual Galaxies)",
             fontsize=12, y=1.01)
fig.tight_layout()
savefig(fig, "paper_fig06_psd_posterior")
plt.show()
