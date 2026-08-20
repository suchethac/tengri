"""
Continuity Prior vs PSD-Governed Prior: Stochastic Structure at Fixed Mean
===========================================================================

At fixed mean SFH and stellar mass, continuity (Leja+2019) and field (PSD-governed)
priors yield strikingly different stochastic realizations: continuity produces smooth
log-normal transitions; field produces controlled burstiness governed by σ_field.

Twenty samples from each prior, overlaid translucently with median shown in bold.
Visual width difference reveals each prior's implicit stochastic assumptions.
"""

import warnings

import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np

from tengri import FREE, Fixed, SEDModel, load_ssp
from tengri.plot import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

ssp = load_ssp("fsps_prsc_miles_chabrier")

model_cont = SEDModel.build(
    ssp_data=ssp, sfh={"type": "continuity", "all_params": FREE}, redshift=Fixed(0.0)
)
model_field = SEDModel.build(
    ssp_data=ssp,
    sfh=[
        {"type": "tsnorm", "all_params": FREE},
        {"type": "field", "psd_sigma": 0.6, "psd_tau_myr": 100},
    ],
    redshift=Fixed(0.0),
)


def sample_sfh(model, n_samples=20):
    key = jr.PRNGKey(42)
    samples, age_gyr = [], None
    for i in range(n_samples):
        key = jr.fold_in(key, i)
        sfh = model.predict_sfh(model.spec.sample(key))
        sfr = np.array(sfh["sfr_mean"])
        samples.append(sfr)
        if age_gyr is None:
            age_gyr = np.array(sfh["t_gyr"])
    return np.array(samples), age_gyr


sfr_cont, age_gyr = sample_sfh(model_cont)
sfr_field, _ = sample_sfh(model_field)
sfr_mean = np.median(sfr_cont, axis=0)

fig, (ax_cont, ax_field) = plt.subplots(1, 2, figsize=(12.0, 5.0), sharex=True, sharey=True)

for ax, sfr_samples, color, label in [
    (ax_cont, sfr_cont, "#3388aa", "Continuity (Leja+19)"),
    (ax_field, sfr_field, "#cc4477", "Field PSD (Caplar & Tacchella)"),
]:
    for sfr in sfr_samples:
        ax.plot(age_gyr, np.clip(sfr, 1e-2, 1e1), lw=0.6, alpha=0.25, color=color)
    ax.plot(age_gyr, sfr_mean, "darkred", lw=2.2, label="Mean SFH", zorder=5)
    ax.set(
        xlabel=r"Lookback Time [Gyr]",
        xscale="log",
        yscale="log",
        xlim=(0.01, 13.8),
        ylim=(1e-2, 2e0),
    )
    if ax is ax_cont:
        ax.set_ylabel(r"SFR [M$_\odot$ yr$^{-1}$]")
    ax.text(
        0.97,
        0.97,
        label,
        transform=ax.transAxes,
        fontsize=11,
        weight="bold",
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(
            boxstyle="round", facecolor=("wheat" if ax is ax_cont else "lightblue"), alpha=0.6
        ),
    )
    ax.grid(True, alpha=0.3, which="both")

fig.legend(["Mean SFH"], loc="upper center", bbox_to_anchor=(0.5, 1.02), fontsize=10)
fig.tight_layout()
plt.savefig("plot_continuity_vs_bursty_psd.png", dpi=150, bbox_inches="tight")
