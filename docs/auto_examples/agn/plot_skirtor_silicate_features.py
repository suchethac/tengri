"""
SKIRTOR Torus: Silicate features from face-on to edge-on
=========================================================

The 9.7 μm and 18 μm silicate bands are strong diagnostics of AGN torus
orientation. When viewing the torus face-on (high cos_inc), dust emission
dominates and silicates appear in emission. Edge-on views (low cos_inc)
show silicates in absorption against the hot dust continuum.

This example builds four SKIRTOR AGN models at different inclinations and
plots the 3–30 μm rest-frame SED, annotating silicate band centers.

Reference: Stalevski et al. (2012, 2016); Hao et al. (2007).
"""

import warnings

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")
warnings.filterwarnings("ignore", message=".*deprecated.*")

# ============================================================================
# Load SSP and build baseline AGN model with SKIRTOR torus
# ============================================================================

ssp = tengri.load_ssp()

# Baseline model: star-forming galaxy + SKIRTOR AGN, all fixed except cos_inc
model = tengri.SEDModel.build(
    ssp,
    sfh={
        "type": "dpl",
        "*": tengri.FIXED,
        "tau_gyr": 3.0,
        "log_peak_sfr": 0.5,
        "alpha": 2.0,
        "beta": 2.5,
    },
    dust={"type": "two_component", "*": tengri.FIXED, "tau_diff": 0.1, "tau_bc": 0.1},
    agn={
        "type": "composable",
        "disc": {"type": "multicolor", "*": tengri.FIXED},
        "torus": {"type": "skirtor", "*": tengri.FIXED, "tau_skirtor": 7.0},
        "lines": {"type": "nlr", "*": tengri.FIXED},
        "*": tengri.FIXED,
        "log_lbol": 12.5,  # log10(L_bol / L_sun)
    },
    redshift=tengri.Fixed(0.05),
)

# Sample baseline parameters
baseline = dict(model.spec.sample(jax.random.PRNGKey(0)))

# ============================================================================
# Inclination angles: cos_inc = 1.0 (face-on) → 0.0 (edge-on)
# ============================================================================

cos_inc_values = np.array([1.0, 0.75, 0.5, 0.0])
labels = [
    r"$\cos \theta_{\rm torus} = 1.0$ (face-on)",
    r"$\cos \theta_{\rm torus} = 0.75$",
    r"$\cos \theta_{\rm torus} = 0.5$",
    r"$\cos \theta_{\rm torus} = 0.0$ (edge-on)",
]

# Colormap from face-on (yellow) to edge-on (blue)
colors_list = ["gold", "orange", "red", "darkblue"]

# ============================================================================
# Plot: 3–30 μm rest-frame rest-frame SED with silicate band annotations
# ============================================================================

fig, ax = plt.subplots(figsize=(8.5, 5.5))

for cos_inc, color, label in zip(cos_inc_values, colors_list, labels):
    params = {**baseline, "agn_cos_inc": jnp.float64(cos_inc)}
    out = model.predict_rest_sed(params)

    wave_rest = np.asarray(out.wavelength)
    sed_rest = np.asarray(out.sed)  # erg/s/Hz

    # Select 3–30 μm rest-frame window
    mask = (wave_rest >= 3e4) & (wave_rest <= 3e5)  # Angstrom
    wave_um = wave_rest[mask] / 1e4
    sed_window = sed_rest[mask]

    ax.loglog(wave_um, sed_window, color=color, lw=2.0, label=label)

# Mark silicate band centers
silicate_9p7 = 9.7  # μm (S_w feature in emission)
silicate_18 = 18.0  # μm (L_w feature)

y_min, y_max = ax.get_ylim()
ax.axvline(silicate_9p7, color="gray", linestyle="--", alpha=0.5, linewidth=1.2)
ax.axvline(silicate_18, color="gray", linestyle="--", alpha=0.5, linewidth=1.2)
ax.text(silicate_9p7, y_max * 0.95, "9.7 μm", fontsize=9, ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
ax.text(silicate_18, y_max * 0.90, "18 μm", fontsize=9, ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

ax.set_xlim(3, 30)
ax.set_xlabel(r"Rest-frame wavelength $\lambda$ [$\mu$m]", fontsize=11)
ax.set_ylabel(r"$L_\nu$ [erg s$^{-1}$ Hz$^{-1}$]", fontsize=11)
ax.set_title("SKIRTOR torus: silicate features across inclinations", fontsize=12, pad=12)
ax.legend(fontsize=10, frameon=True, loc="upper right", framealpha=0.95)
ax.grid(True, alpha=0.3, which="both")

fig.tight_layout()
plt.savefig("plot_skirtor_silicate_features.png", dpi=150, bbox_inches="tight")
plt.show()
