"""
Quenching pathways: fast vs slow termination of star formation
==============================================================

Compare three star-formation histories representing distinct quenching scenarios:
(1) Constantly star-forming (no quenching), (2) Slowly quenched exponential decay
(tau=4 Gyr, peak 6 Gyr ago), and (3) Rapidly quenched post-starburst (truncated
skew-normal, peak 2 Gyr ago, width 0.3 Gyr). The resulting rest-frame SEDs exhibit
markedly different colors, equivalent widths (Hα), and spectral slopes, highlighting
how quenching timescale imprints on observable photometry and spectroscopy.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")
warnings.filterwarnings("ignore", message=".*deprecated.*")

ssp = tengri.load_ssp()

# --- Define three quenching scenarios ---
scenarios = [
    {
        "name": "Star-forming (no quench)",
        "sfh": {
            "type": "const",
            "all_params": tengri.FIXED,
            "log_total_mass": 10.64,
            # start_gyr is the lookback to SF ONSET and end_gyr to SF CESSATION,
            # so start_gyr is the LARGER number: forming stars from 13.8 Gyr ago
            # until 0.1 Gyr ago. Written the other way round this window is empty
            # and the curve is identically zero -- which is what this example
            # plotted until #1277 made the ordering an error rather than a silent
            # zero-mass galaxy.
            "start_gyr": 13.8,
            "end_gyr": 0.1,
        },
        "color": "#1f77b4",
    },
    {
        "name": "Slow quench (τ=4 Gyr)",
        "sfh": {
            "type": "dexp",
            "all_params": tengri.FIXED,
            "tau_gyr": 4.0,
            "log_total_mass": 10.0,
        },
        "color": "#ff7f0e",
    },
    {
        "name": "Fast post-starburst (t_sb=2 Gyr)",
        "sfh": {
            "type": "tsnorm",
            "all_params": tengri.FIXED,
            "log_total_mass": 10.0,
            "peak_lbt_gyr": 2.0,
            "width_gyr": 0.3,
            "skew": 1.0,
            "trunc": 3.0,
        },
        "color": "#d62728",
    },
]

# --- Build models for each scenario ---
models = {}
params_dict = {}
for scenario in scenarios:
    model = tengri.SEDModel.build(
        ssp,
        sfh=scenario["sfh"],
        dust={
            "type": "two_component",
            "all_params": tengri.FIXED,
            "tau_diff": 0.15,
            "tau_bc": 0.3,
        },
        redshift=tengri.Fixed(0.05),
    )
    models[scenario["name"]] = model
    params_dict[scenario["name"]] = dict(model.spec.sample(jax.random.PRNGKey(42)))

# --- Compute SFH and SED predictions ---
sfh_data = {}
sed_data = {}

t_lookback = jnp.linspace(0, 13.8, 500)

for scenario in scenarios:
    name = scenario["name"]
    model = models[name]
    params = params_dict[name]

    # SFH predictions
    sfh_out = model.predict_sfh(params)
    sfh_data[name] = {
        "t_gyr": np.asarray(sfh_out["t_gyr"]),
        "sfr_mean": np.asarray(sfh_out["sfr_mean"]),
    }

    # Rest-frame SED predictions
    sed_out = model.predict(params)
    wave = np.asarray(model.wavelengths)
    nu = 2.998e18 / wave  # c in A/s -> frequency in Hz
    nu_l_nu = nu * np.asarray(sed_out.rest_sed())
    sed_data[name] = {"wave": wave, "nu_l_nu": nu_l_nu}

# --- Create two-panel figure ---
fig, axes = plt.subplots(2, 1, figsize=(7, 7))

# --- Top panel: SFH comparison ---
ax_sfh = axes[0]
for scenario in scenarios:
    name = scenario["name"]
    sfh = sfh_data[name]
    ax_sfh.plot(sfh["t_gyr"], sfh["sfr_mean"], label=name, color=scenario["color"], lw=2.0)

ax_sfh.set_xlabel("Lookback time [Gyr]")
ax_sfh.set_ylabel(r"SFR [$M_\odot$ yr$^{-1}$]")
ax_sfh.set_xlim(0, 13.8)
ax_sfh.set_ylim(0, None)
ax_sfh.legend(fontsize=9, frameon=False, loc="upper right")
ax_sfh.grid(alpha=0.3, linestyle=":", linewidth=0.5)

# --- Bottom panel: SED comparison (log-log) ---
ax_sed = axes[1]
for scenario in scenarios:
    name = scenario["name"]
    sed = sed_data[name]
    ax_sed.loglog(sed["wave"], sed["nu_l_nu"], label=name, color=scenario["color"], lw=1.5)

# Annotate key diagnostic features
ax_sed.axvline(6564.61, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)  # H-alpha
ax_sed.axvline(4000, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)  # D4000 break
ax_sed.text(6564.61, 1.2e46, "Hα", fontsize=8, ha="center", color="gray", alpha=0.7)
ax_sed.text(4000, 1.2e46, "D4000", fontsize=8, ha="center", color="gray", alpha=0.7)

ax_sed.set_xlim(100, 1e5)
ax_sed.set_xlabel(r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]")
ax_sed.set_ylabel(r"$\nu L_\nu$  [erg s$^{-1}$]")
ax_sed.legend(fontsize=9, frameon=False, loc="lower left")
ax_sed.grid(alpha=0.3, linestyle=":", linewidth=0.5, which="both")

fig.tight_layout()
plt.savefig("plot_quenching_pathway_compare.png", dpi=150, bbox_inches="tight")
