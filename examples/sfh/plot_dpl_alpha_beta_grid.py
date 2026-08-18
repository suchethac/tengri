"""
Double power-law SFH parameter space: early growth α vs late quenching β
========================================================================

A 3×3 grid showing how the rising slope α (columns) and falling slope β (rows)
together control the full SFH morphology. Early-time α determines assembly
speed; late-time β sets the post-peak decay. The optical SED responds across
each cell. Bottom panels show representative 1D sweeps: α alone (left, at fixed β)
and β alone (right, at fixed α), illustrating how each parameter independently
shapes the full UV-to-IR SED.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.plot import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

ssp = tengri.load_ssp()

# Grid: alpha (rising slope, columns) and beta (falling slope, rows)
alphas = [0.5, 1.5, 3.0]
betas = [0.5, 1.5, 3.0]

fig, axes = plt.subplots(3, 3, figsize=(12, 10))

baseline = dict(
    tau_gyr=3.0,
    log_total_mass=10.0,
)

# Pre-compute y-axis limits across all panels for consistent scaling
y_min = float("inf")
y_max = float("-inf")
sed_cache = {}

for i, beta in enumerate(betas):
    for j, alpha in enumerate(alphas):
        model = tengri.SEDModel.build(
            ssp,
            sfh={
                "type": "dpl",
                "all_params": tengri.FIXED,
                "alpha": alpha,
                "beta": beta,
                "tau_gyr": baseline["tau_gyr"],
                "log_total_mass": 10.0,
            },
            dust={
                "type": "two_component",
                "all_params": tengri.FIXED,
                "tau_diff": 0.2,
                "tau_bc": 0.3,
            },
            redshift=tengri.Fixed(0.1),
        )

        baseline_params = dict(model.spec.sample(jax.random.PRNGKey(0)))
        params_eval = {**baseline_params}

        out = model.predict(params_eval)
        wave = np.asarray(model.wavelengths)
        sed = np.asarray(out.rest_sed())

        # Optical region
        mask = (wave > 4000) & (wave < 8000)
        sed_cache[(i, j)] = (wave[mask], sed[mask])

        # Track limits for shared y-axis
        y_min = min(y_min, sed[mask].min())
        y_max = max(y_max, sed[mask].max())

# Now plot with shared y-limits and labels
for i, beta in enumerate(betas):
    for j, alpha in enumerate(alphas):
        ax = axes[i, j]
        wave_opt, sed_opt = sed_cache[(i, j)]

        ax.plot(
            wave_opt,
            sed_opt,
            "C0-",
            lw=2.0,
        )

        # Add panel label
        ax.text(
            0.98,
            0.98,
            rf"$\alpha={alpha:.1f}$, $\beta={beta:.1f}$",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )

        ax.set_ylim(y_min * 0.9, y_max * 1.1)
        ax.set_xlabel(r"Wavelength [$\AA$]", fontsize=9)
        ax.set_ylabel(r"$L_\nu$ [erg/s/Hz]", fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=8)

fig.tight_layout()

# --- Add bottom panels: α and β 1D sweeps in SED space ---
C_AA_PER_S = 2.998e18

# Panel α sweep (fixed β)
model_alpha = tengri.SEDModel.build(
    ssp,
    sfh={
        "type": "dpl",
        "all_params": tengri.FIXED,
        "alpha": tengri.Uniform(0.3, 6.0),
        "beta": 2.5,
        "tau_gyr": 1.5,
        "log_total_mass": 10.0,
    },
    dust={"type": "two_component", "all_params": tengri.FIXED, "tau_diff": 0.2, "tau_bc": 0.3},
    redshift=tengri.Fixed(0.1),
)
baseline_alpha = dict(model_alpha.spec.sample(jax.random.PRNGKey(0)))

alpha_values = np.linspace(0.5, 6.0, 7)
norm_alpha = mpl.colors.Normalize(vmin=alpha_values.min(), vmax=alpha_values.max())
cmap = plt.get_cmap("viridis")

fig_bottom = plt.figure(figsize=(14, 4.2))
ax_alpha = fig_bottom.add_subplot(121)
ax_beta = fig_bottom.add_subplot(122)

for alpha in alpha_values:
    params = {**baseline_alpha, "sfh_dpl_alpha": jnp.float64(alpha)}
    out = model_alpha.predict(params)
    wave = np.asarray(model_alpha.wavelengths)
    nu = C_AA_PER_S / wave
    nu_l_nu = nu * np.asarray(out.rest_sed())
    ax_alpha.loglog(wave, nu_l_nu, color=cmap(norm_alpha(alpha)), lw=1.4)

ax_alpha.set_xlim(800, 3e4)
ax_alpha.set_ylim(1e40, 5e43)
ax_alpha.set_xlabel(r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]")
ax_alpha.set_ylabel(r"$\nu L_\nu$  [erg s$^{-1}$]")
cbar_alpha = fig_bottom.colorbar(
    plt.cm.ScalarMappable(norm=norm_alpha, cmap=cmap), ax=ax_alpha, pad=0.01
)
cbar_alpha.set_label(r"DPL rising slope $\alpha$")

# Panel β sweep (fixed α)
model_beta = tengri.SEDModel.build(
    ssp,
    sfh={
        "type": "dpl",
        "all_params": tengri.FIXED,
        "beta": tengri.Uniform(0.3, 10.0),
        "alpha": 1.5,
        "tau_gyr": 3.0,
        "log_total_mass": 10.0,
    },
    dust={"type": "two_component", "all_params": tengri.FIXED, "tau_diff": 0.2, "tau_bc": 0.3},
    redshift=tengri.Fixed(0.1),
)
baseline_beta = dict(model_beta.spec.sample(jax.random.PRNGKey(0)))

beta_values = np.linspace(0.3, 10.0, 7)
norm_beta = mpl.colors.Normalize(vmin=beta_values.min(), vmax=beta_values.max())

for beta in beta_values:
    params = {**baseline_beta, "sfh_dpl_beta": jnp.float64(beta)}
    out = model_beta.predict(params)
    wave = np.asarray(model_beta.wavelengths)
    nu = C_AA_PER_S / wave
    nu_l_nu = nu * np.asarray(out.rest_sed())
    ax_beta.loglog(wave, nu_l_nu, color=cmap(norm_beta(beta)), lw=1.4)

ax_beta.set_xlim(800, 3e4)
ax_beta.set_ylim(1e40, 5e43)
ax_beta.set_xlabel(r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]")
ax_beta.set_ylabel(r"$\nu L_\nu$  [erg s$^{-1}$]")
cbar_beta = fig_bottom.colorbar(
    plt.cm.ScalarMappable(norm=norm_beta, cmap=cmap), ax=ax_beta, pad=0.01
)
cbar_beta.set_label(r"DPL falling slope $\beta$")

fig_bottom.tight_layout()
plt.savefig("plot_dpl_alpha_beta_grid.png", dpi=150, bbox_inches="tight")
