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
# # Mock Program: Generating the Paper's Test Suite
#
# This notebook generates all mock data used in the paper (§3.1).
# The mock program spans four PSD regimes, three redshifts, and
# two data types (photometry + spectroscopy).
#
# **Outputs saved to `../data/mocks/` for use by A02–A04.**
#
# **Paper figures generated here:**
# - **Fig 2**: PSD → SFH (four regimes)
# - **Fig 3**: Recovery test design matrix (schematic)

# %%
import os, json
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
import numpy as np

from tengri import (
    SEDModel, ParamSpec, Uniform, Fixed,
    load_ssp_data, load_filter_set,
)
from tengri.sfh.psd_models import psd_drw, drw_variance
from tengri.sfh.gp_sfh import compute_sqrt_power_drw, generate_gp_fourier
from tengri.sfh.mean_sfh import double_powerlaw
from tengri.utils.grid import make_log_age_grid, grid_spacing, log_age_to_age_yr

import sys; sys.path.insert(0, ".")
from _plot_style import setup_style, COLORS
setup_style()

FIG_DIR = "notebook_figures"
MOCK_DIR = "../data/mocks"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(MOCK_DIR, exist_ok=True)
def savefig(fig, name):
    fig.savefig(os.path.join(FIG_DIR, f"A01_{name}.png"),
                bbox_inches="tight", dpi=72)

ssp_data = load_ssp_data(
    "../data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")

# %% [markdown]
# ## PSD Regimes

# %%
PSD_REGIMES = {
    "smooth":        {"sigma": 0.5, "tau_myr": 200, "color": "#1b9e77"},
    "moderate":      {"sigma": 1.0, "tau_myr": 50,  "color": "#d95f02"},
    "bursty":        {"sigma": 2.0, "tau_myr": 20,  "color": "#7570b3"},
    "highly_bursty": {"sigma": 3.0, "tau_myr": 5,   "color": "#e7298a"},
}

REDSHIFTS = [0.1, 2.0, 6.0]
FILTER_SETS = {
    0.1: ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"],
    2.0: ["hst_f435w", "hst_f606w", "hst_f775w", "hst_f814w",
           "hst_f850lp", "hst_f105w", "hst_f125w", "hst_f160w"],
    6.0: ["jwst_f090w", "jwst_f115w", "jwst_f150w", "jwst_f200w",
           "jwst_f277w", "jwst_f356w", "jwst_f410m", "jwst_f444w"],
}
N_MOCK = 100

# %% [markdown]
# ## Paper Figure 2: PSD → SFH

# %%
N_GRID = 128
log_ages = make_log_age_grid(N_GRID)
d_log = grid_spacing(log_ages)
ages_yr = log_age_to_age_yr(log_ages)
ages_gyr = ages_yr / 1e9

mean_sfr = double_powerlaw(ages_yr, 1.5, 1.0, 5e9, 5.0)

omega = jnp.logspace(-4, 1, 500)
key = jax.random.PRNGKey(42)

fig = plt.figure(figsize=(16, 8))
gs = fig.add_gridspec(2, 4, hspace=0.35, wspace=0.3)

# Top-left: PSD
ax_psd = fig.add_subplot(gs[0, 0])
for name, r in PSD_REGIMES.items():
    P = psd_drw(omega, r["sigma"], r["tau_myr"])
    ax_psd.loglog(omega, P, lw=2, color=r["color"],
                  label=rf"$\sigma$={r['sigma']}, $\tau$={r['tau_myr']}")
ax_psd.set_xlabel(r"$\omega$ [rad/Myr]"); ax_psd.set_ylabel(r"$P(\omega)$")
ax_psd.set_title("DRW Power Spectrum"); ax_psd.legend(fontsize=6)

# Top-right: ACF
ax_acf = fig.add_subplot(gs[0, 1])
from tengri.sfh.psd_models import drw_acf
dt = jnp.linspace(0, 500, 300)
for name, r in PSD_REGIMES.items():
    acf = drw_acf(dt, r["sigma"], r["tau_myr"])
    ax_acf.plot(dt, acf / acf[0], lw=2, color=r["color"])
ax_acf.set_xlabel(r"$\Delta t$ [Myr]"); ax_acf.set_ylabel("Normalised ACF")
ax_acf.set_title("Autocorrelation")

# Bottom row: 4 SFH panels
for i, (name, r) in enumerate(PSD_REGIMES.items()):
    ax = fig.add_subplot(gs[1, i])
    sqrt_p = compute_sqrt_power_drw(N_GRID, d_log, r["sigma"], r["tau_myr"] * 1e6)
    gp = generate_gp_fourier(key, sqrt_p, N_GRID)
    var = drw_variance(r["sigma"])
    sfr = mean_sfr * jnp.exp(gp - var / 2.0)

    ax.plot(ages_gyr, sfr, lw=1.5, color=r["color"])
    ax.plot(ages_gyr, mean_sfr, lw=0.8, ls="--", color="gray")
    ax.set_xlabel("Lookback [Gyr]")
    if i == 0:
        ax.set_ylabel(r"SFR [M$_\odot$/yr]")
    label = name.replace("_", " ").title()
    ax.set_title(rf"{label}: $\sigma$={r['sigma']}, $\tau$={r['tau_myr']} Myr",
                 fontsize=9)
    ax.set_xlim(0, 13.5); ax.set_ylim(bottom=0)

fig.suptitle("Paper Figure 2: From PSD to SFH", fontsize=14, y=1.01)
savefig(fig, "paper_fig02_psd_to_sfh"); plt.show()

# %% [markdown]
# ## Generate Mock Populations

# %%
for z in REDSHIFTS:
    try:
        filters = load_filter_set(FILTER_SETS[z])
    except Exception as e:
        print(f"Skipping z={z}: {e}")
        continue

    spec = ParamSpec(
        sfh_alpha=Uniform(0.5, 3.0), sfh_beta=Uniform(0.5, 3.0),
        sfh_tau_peak_gyr=Uniform(0.5, 13.0), sfh_peak_sfr=Uniform(0.1, 100.0),
        psd_sigma=Uniform(0.1, 4.0), psd_tau_myr=Uniform(1.0, 300.0),
        met_logzsol=Uniform(-2.0, 0.0),
        dust_tau_bc=Uniform(0.0, 2.0), dust_tau_diff=Uniform(0.0, 1.0),
        dust_slope=Fixed(-0.7), redshift=Fixed(z),
        stochastic=True, n_grid=128,
    )
    model = SEDModel(spec, ssp_data, filters=filters)

    for regime_name, r in PSD_REGIMES.items():
        key = jax.random.PRNGKey(int(z * 1000) + hash(regime_name) % 10000)
        mocks = []

        for i in range(N_MOCK):
            key, subkey = jax.random.split(key)
            tp = spec.sample(subkey)
            tp.update(psd_sigma=r["sigma"], psd_tau_myr=r["tau_myr"])
            mock_i = model.mock(tp, snr=20.0, key=subkey)
            mocks.append({"true_params": tp, "flux_obs": mock_i.flux_obs,
                          "noise": mock_i.noise})

        outpath = os.path.join(MOCK_DIR, f"z{z}_{regime_name}.npz")
        # Save as compressed numpy archive
        np.savez_compressed(outpath,
                            flux_obs=np.array([m["flux_obs"] for m in mocks]),
                            noise=np.array([m["noise"] for m in mocks]))
        print(f"Saved {outpath}: {N_MOCK} galaxies at z={z}, {regime_name}")

print("\nMock generation complete.")
