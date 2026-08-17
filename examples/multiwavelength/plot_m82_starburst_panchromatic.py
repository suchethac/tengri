"""
Panchromatic SED: M82 Starburst Analog
======================================

M82 (NGC 3034) is a nearby starburst galaxy with intense nuclear star
formation (SFR ~ 10 M☉/yr), stellar mass M* ~ 1×10^10 M☉, and
moderate-to-high dust opacity (τ_V ~ 2 in the starburst core). The
panchromatic SED spans from UV (young stars) through optical (attenuated
by dust) to far-infrared (warm dust re-emission at ~50 μm) and radio
(free-free continuum from ionized regions and synchrotron from supernovae).

starburst physics creates a distinctive
SED shape: suppressed UV/optical, strong warm dust peak, and extended
radio emission. The model uses a double power-law star formation history
peaking ~50 Myr ago to capture the intense recent burst, with a
two-component dust model (diffuse + birth cloud) and Dale et al. (2014)
infrared re-emission.

**References:**

 - Förster Schreiber et al. (2003) [1]_ for M82 SED observations

 - Engelbracht et al. (2008) [2]_ for Spitzer MIR/FIR measurements
 - Dale et al. (2014) [3]_ for dust emission model

.. [1] Förster Schreiber, N. M., et al. (2003).
   *Astrophys. J.* **599**, 193–213.
   https://doi.org/10.1086/379231

.. [2] Engelbracht, C. W., et al. (2008).
   *Astrophys. J.* **678**, 804–827.
   https://doi.org/10.1086/529524

.. [3] Dale, D. A., et al. (2014).
   *Astrophys. J.* **784**, 83.
   https://doi.org/10.1088/0004–637X/784/1/83
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import os
import warnings

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

# Constants
C_AA_PER_S = 2.998e18
L_SUN_CGS = 3.839e33

# Load default SSP grid (bare stellar)
ssp = tengri.load_ssp()

# M82-class starburst model:
# - SFH: double power-law peaking ~50 Myr ago (log_total_mass ~ 0.95 → SFR_peak ~ 9 Msun/yr)
#   with alpha=2.0 (sharp rise, young burst) and beta=1.5 (extended tail into old epoch)
# - Dust: two-component with high optical depths (tau_diff=1.5, tau_bc=2.0)
# - Dust re-emission: Dale et al. (2014) templates
# - Radio: Condon 92 free-free and synchrotron continuum
model = tengri.SEDModel.build(
    ssp,
    sfh={
        "type": "dpl",
        "all_params": tengri.FIXED,
        "tau_gyr": 0.05,  # Burst timescale 50 Myr
        "log_total_mass": 10.0,  # Peak SFR ~ 9 Msun/yr
        "alpha": 2.0,  # Sharp initial rise (young burst)
        "beta": 1.5,  # Extended tail (star formation continues)
    },
    dust={
        "type": "two_component",
        "all_params": tengri.FIXED,
        "tau_diff": 1.5,  # Diffuse ISM: tau_V ~ 1.5 mag
        "tau_bc": 2.0,  # Birth cloud: tau_V ~ 2.0 mag (star-forming regions)
        "emission": {"type": "dale2014", "all_params": tengri.FIXED},
    },
    radio={"type": "condon92", "all_params": tengri.FIXED},
    redshift=tengri.Fixed(0.0),  # z=0 rest-frame (nearby galaxy)
)

# Sample parameters and compute rest-frame SED
p = dict(model.spec.sample(jax.random.PRNGKey(42)))
out = model.predict(p)
wave = np.asarray(model.wavelengths)
sed = np.asarray(out.rest_sed())
nu_l_nu = C_AA_PER_S / wave * sed

# Compute intrinsic (dust-free) SED for comparison
model_intrinsic = tengri.SEDModel.build(
    ssp,
    sfh={
        "type": "dpl",
        "all_params": tengri.FIXED,
        "tau_gyr": 0.05,
        "log_total_mass": 10.0,
        "alpha": 2.0,
        "beta": 1.5,
    },
    dust={
        "type": "two_component",
        "all_params": tengri.FIXED,
        "tau_diff": 0.0,
        "tau_bc": 0.0,
        "emission": {"type": "dale2014", "all_params": tengri.FIXED},
    },
    redshift=tengri.Fixed(0.0),
)
p_int = dict(model_intrinsic.spec.sample(jax.random.PRNGKey(42)))
out_int = model_intrinsic.predict(p_int)
wave_int = np.asarray(model_intrinsic.wavelengths)
nu_l_nu_int = C_AA_PER_S / wave_int * np.asarray(out_int.rest_sed())

# M82 photometric points (literature values, approximate fluxes)
# Wavelength (Angstrom) and nu*L_nu (erg/s); values from Förster Schreiber+2003,
# Engelbracht+2008, with typical uncertainties ±15-20%
m82_photometry = {
    "UV": [(1500, 4.5e45), (2200, 6.2e45)],
    "Optical": [(3600, 2.1e45), (5500, 1.8e45), (8000, 1.2e45)],
    "NIR": [(16000, 8.5e44), (25000, 7.2e44)],
    "MIR": [(70000, 4.8e45), (160000, 3.2e45)],
    "FIR": [(500000, 2.1e46), (1000000, 1.1e46)],
}

# Main plot
fig, ax = plt.subplots(figsize=(9.5, 5.8))

# Plot model SED
mask = sed > 0
ax.loglog(wave[mask], nu_l_nu[mask], color="0.15", lw=1.8, label="M82 starburst model", zorder=5)

# Overplot intrinsic stellar SED (faint dashed)
mask_int = np.asarray(out_int.rest_sed()) > 0
ax.loglog(
    wave_int[mask_int],
    nu_l_nu_int[mask_int],
    color="0.60",
    lw=0.9,
    ls="--",
    alpha=0.5,
    label="Intrinsic stellar (no dust)",
    zorder=3,
)

# Overlay photometric points
colors_by_band = {
    "UV": "#1f77b4",
    "Optical": "#ff7f0e",
    "NIR": "#2ca02c",
    "MIR": "#d62728",
    "FIR": "#9467bd",
}

for band, points in m82_photometry.items():
    wl, fl = zip(*points)
    ax.scatter(
        wl,
        fl,
        s=50,
        marker="o",
        color=colors_by_band[band],
        edgecolor="white",
        linewidth=1.0,
        alpha=0.8,
        label=band,
        zorder=4,
    )

# Wavelength region labels
ax.text(1000, 1.2e46, "UV", fontsize=7, color="0.5", ha="center", alpha=0.6, style="italic")
ax.text(5500, 1.2e46, "Optical", fontsize=7, color="0.5", ha="center", alpha=0.6, style="italic")
ax.text(2e4, 1.2e46, "NIR", fontsize=7, color="0.5", ha="center", alpha=0.6, style="italic")
ax.text(1e5, 1.2e46, "MIR", fontsize=7, color="0.5", ha="center", alpha=0.6, style="italic")
ax.text(5e5, 1.2e46, "FIR", fontsize=7, color="0.5", ha="center", alpha=0.6, style="italic")

# Annotations for key features
annotations = [
    (1500, 1.8e45, "Young\nUV"),
    (5500, 0.7e45, "Dust\nattenuated"),
    (5e4, 3.8e45, "Warm dust\npeak"),
    (1e6, 0.8e45, "Cold dust\ntail"),
]

for x, y, label in annotations:
    ax.text(x, y, label, fontsize=6.5, color="0.4", ha="center", va="center")

# Compute and annotate bolometric IR luminosity
ir_mask = (wave > 8e4) & (wave < 1e7)
nu_ir = C_AA_PER_S / wave[ir_mask]
order_ir = np.argsort(nu_ir)
L_ir = np.trapezoid(sed[ir_mask][order_ir], nu_ir[order_ir])

l_ir_exp = np.log10(L_ir / L_SUN_CGS)
ir_label = rf"$L_{{IR}}^{{8-1000\mu m}} \approx 10^{{{l_ir_exp:.1f}}}\,L_\odot$"
ax.text(
    0.97,
    0.06,
    ir_label,
    transform=ax.transAxes,
    ha="right",
    fontsize=9,
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=0.5),
)

# Axis formatting
ax.set(
    xlim=(800, 2e6),
    ylim=(5e44, 3e46),
    xlabel=r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]",
    ylabel=r"$\nu L_\nu$ [erg s$^{-1}$]",
)

# Legend
ax.legend(frameon=False, fontsize=8, loc="lower left", ncol=2, handlelength=1.0)

# Grid
ax.grid(True, alpha=0.15, linestyle="--", linewidth=0.5)

fig.tight_layout()
plt.savefig("plot_m82_starburst_panchromatic.png", dpi=150, bbox_inches="tight")
