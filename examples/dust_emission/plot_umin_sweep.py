r"""
U_min: DL07 and THEMIS agree on the FIR peak
==============================================

Minimum radiation field intensity U_min controls diffuse dust heating.
Higher U_min implies hotter dust and FIR peak shifted blueward toward
shorter wavelengths. This figure overlays DL07 and THEMIS dust emission
templates at matched U_min values to demonstrate that both libraries
agree on the FIR peak location (~100 μm).
"""

import warnings

import h5py
import jax
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri import FIXED, SEDModel, data_path, load_ssp, recipes
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")
warnings.filterwarnings("ignore", message=".*deprecated.*")

# Wavelength grid and constants for THEMIS data
c_aa_per_s = 2.99792458e18

# Load THEMIS templates for reference
with h5py.File(data_path("themis_templates.h5"), "r") as f:
    wave_themis_aa = np.asarray(f["wavelength_aa"][:])
    qhac_grid = np.asarray(f["qhac_grid"][:])
    umin_grid_themis = np.asarray(f["umin_grid"][:])
    single_u_themis = np.asarray(f["single_u"][:])

wave_themis_um = wave_themis_aa * 1.0e-4
nu_themis = c_aa_per_s / wave_themis_aa
i_qhac = int(np.argmin(np.abs(qhac_grid - 0.17)))

# Select U_min values that exist in THEMIS grid
u_min_values = [0.5, 2.0, 5.0, 15.0]

# Build DL07 model with U_min as a free parameter
model_dl07 = SEDModel.build(
    ssp_data=load_ssp(),
    sfh={"type": "dpl", "*": FIXED},
    dust={
        "type": "two_component",
        "*": FIXED,
        "emission": {
            "type": "draine_li2007",
            "*": FIXED,
            "dust_gamma_dl": 0.01,
            "dust_qpah": 2.5,
            "dust_umin": tengri.Uniform(0.1, 30.0),
        },
    },
    redshift=tengri.Fixed(0.0),
)
baseline_dl07 = dict(model_dl07.spec.sample(jax.random.PRNGKey(0)))

# Compute DL07 SEDs at selected U_min values
dl07_seds = {}
wave_dl07_aa = None
for u_min in u_min_values:
    params = {**baseline_dl07, "dust_umin": float(u_min)}
    pred = model_dl07.predict_rest_sed(params)
    sed = np.asarray(pred.sed)
    wave = np.asarray(pred.wavelength)
    # Convert L_nu [erg/s/Hz] to nu*L_nu [erg/s] so the FIR peak is at the actual peak
    nu_local = c_aa_per_s / wave
    dl07_seds[u_min] = nu_local * sed
    if wave_dl07_aa is None:
        wave_dl07_aa = wave

wave_dl07_um = wave_dl07_aa * 1.0e-4

# Set up figure with colormap
fig, ax = plt.subplots(figsize=(9.5, 6.0))
cmap = plt.get_cmap("plasma")

# Plot DL07 (solid lines) and THEMIS (dashed lines) for each U_min
for k, u_min in enumerate(u_min_values):
    color = cmap(k / max(1, len(u_min_values) - 1))
    label = rf"$U_{{\rm min}} = {u_min:.1f}$"

    # DL07 line (solid)
    sed_dl07 = dl07_seds[u_min]
    # Normalize to peak in the FIR window (>5 μm) so the stellar continuum
    # doesn't dominate the peak finder.
    fir_mask = wave_dl07_um > 5.0
    sed_dl07_norm = sed_dl07 / np.max(sed_dl07[fir_mask])
    ax.plot(
        wave_dl07_um,
        sed_dl07_norm,
        color=color,
        lw=1.5,
        linestyle="-",
        label=label if k == 0 else None,
    )

    # THEMIS line (dashed) — find closest U_min in THEMIS grid
    iu_themis = int(np.argmin(np.abs(umin_grid_themis - u_min)))
    L_nu_themis = single_u_themis[i_qhac, iu_themis]
    sed_themis = nu_themis * L_nu_themis
    # Normalize to peak for shape comparison
    sed_themis_norm = sed_themis / np.max(sed_themis)
    ax.plot(
        wave_themis_um,
        sed_themis_norm,
        color=color,
        lw=1.5,
        linestyle="--",
    )

# Add custom legend entries for linestyle convention
from matplotlib.lines import Line2D

legend_lines = [
    Line2D([0], [0], color="k", linestyle="-", lw=1.5, label="DL07"),
    Line2D([0], [0], color="k", linestyle="--", lw=1.5, label="THEMIS"),
]
ax.set(
    xscale="log",
    yscale="log",
    xlabel=r"$\lambda\ [\mu\mathrm{m}]$",
    ylabel=r"$\nu L_\nu / \mathrm{peak}$ (normalised)",
    xlim=(5.0, 1.0e3),
    ylim=(1e-3, 2.0),
    title=r"$U_{\rm min}$: DL07 and THEMIS agree on the FIR peak ($\sim 100\ \mu\mathrm{m}$)",
)

# Combine U_min and linestyle legends
legend1 = ax.legend(
    handles=legend_lines,
    loc="upper right",
    frameon=False,
    fontsize=9,
)
ax.add_artist(legend1)

# Add U_min colorbar proxy
norm = plt.Normalize(vmin=u_min_values[0], vmax=u_min_values[-1])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.02)
cbar.set_label(r"$U_{\rm min}$", fontsize=10)

fig.tight_layout()
plt.savefig("plot_umin_sweep.png", dpi=150, bbox_inches="tight")
