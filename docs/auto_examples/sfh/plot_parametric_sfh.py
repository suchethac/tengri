"""
Catalog of parametric star-formation-history models
====================================================

Compare all parametric SFH models available in tengri. Each is evaluated on
a lookback-time grid with representative parameters, showing the range of
morphologies from smooth exponentials to sharp truncations. No SSP data required.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

t_lookback = jnp.linspace(1e5, 14e9, 1000)
t_gyr = np.array(t_lookback) / 1e9

# --- Evaluate each SFH model with representative parameters ---
models = {
    "tsnorm (truncated skew-normal)": tengri.tsnorm(
        t_lookback, log_total_mass=10.0, peak_lbt=6e9, width=2e9, skew=1.0, trunc=3.0
    ),
    "snorm (skew-normal)": tengri.snorm(
        t_lookback, log_total_mass=10.0, peak_lbt=6e9, width=2e9, skew=1.5
    ),
    "norm (Gaussian)": tengri.norm(t_lookback, log_total_mass=10.0, peak_lbt=6e9, width=2e9),
    "lnorm (log-normal)": tengri.lnorm(
        t_lookback, log_total_mass=10.0, peak=6e9, width=0.3, age=13.6e9
    ),
    "dpl (double power law)": tengri.dpl(
        t_lookback, alpha=2.0, beta=1.0, tau=5e9, age=13.6e9, log_total_mass=10.0
    ),
    "exponential": tengri.exponential(t_lookback, log_total_mass=10.0, tau=3e9, start=1e9),
    "delayed exponential": tengri.delayed_exponential(
        t_lookback, log_total_mass=10.0, tau=3e9, start=1e9
    ),
    "constant": tengri.constant(t_lookback, log_total_mass=10.63, start=2e9, end=10e9),
}

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]

# --- Plot ---
fig, ax = plt.subplots(figsize=(9, 5))
for (name, sfr), color in zip(models.items(), colors):
    ax.plot(t_gyr, np.array(sfr), label=name, color=color, lw=1.5)

ax.set_xlabel("Lookback time [Gyr]")
ax.set_ylabel("SFR [M$_\\odot$/yr]")
ax.set_xlim(0, 14)
ax.set_ylim(0, None)
ax.legend(fontsize=10, frameon=False, ncol=2, loc="upper right")
fig.tight_layout()
plt.savefig("plot_parametric_sfh.png", dpi=150, bbox_inches="tight")
