"""
UV-to-radio SED of a dusty starburst ULIRG
===========================================

A heavily obscured starburst archetype (Arp 220-class ULIRG) with
high optical depth and extreme far-infrared dominance. The SED shows:
stellar intrinsic (suppressed by dust), stellar attenuated, dust
re-emission dominating at 60 μm, and radio extension. Demonstrates
how dust attenuation redirects all UV/optical photons into the infrared,
completely transforming the SED from young, luminous starbursts.

References: Sanders & Mirabel 1996 (ULIRG definition), Soifer et al. 1987
(Arp 220 SED properties).
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import jax
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

C_AA_PER_S = 2.998e18

# Dusty starburst parameters: M* ~ 5e10 Msun, SFR ~ 100 Msun/yr, tau_V > 5
# anchored to match Arp 220's L_IR ~ 10^12 L_sun
ssp = tengri.load_ssp()
model = tengri.SEDModel.build(
    ssp,
    sfh={
        "type": "const",
        "*": tengri.FIXED,
        "log_total_mass": 10.0,  # 100 Msun/yr over 100 Myr → 1e10 Msun
        "start_gyr": 0.1,
        "end_gyr": 0.0,  # 100 Myr ongoing burst
    },
    dust={
        "type": "two_component",
        "*": tengri.FIXED,
        "tau_diff": 3.0,  # diffuse dust: tau_V = 3 mag
        "tau_bc": 2.5,  # birth cloud opacity: tau_V = 2.5 mag
        "emission": {"type": "dale2014", "*": tengri.FIXED},
    },
    radio={"type": "condon92", "*": tengri.FIXED},
    redshift=tengri.Fixed(0.018),  # Arp 220 redshift
)

# Sample parameters and compute rest-frame SED
p = dict(model.spec.sample(jax.random.PRNGKey(0)))
out = model.predict_rest_sed(p)
wave = np.asarray(out.wavelength)
nu_l_nu = C_AA_PER_S / wave * np.asarray(out.sed)

# Compute additional components for illustration: intrinsic stellar SED (no dust)
model_intrinsic = tengri.SEDModel.build(
    ssp,
    sfh={
        "type": "const",
        "*": tengri.FIXED,
        "log_total_mass": 10.0,
        "start_gyr": 0.1,
        "end_gyr": 0.0,
    },
    dust={
        "type": "two_component",
        "*": tengri.FIXED,
        "tau_diff": 0.0,
        "tau_bc": 0.0,
        "emission": {"type": "dale2014", "*": tengri.FIXED},
    },
    redshift=tengri.Fixed(0.018),
)
p_intrinsic = dict(model_intrinsic.spec.sample(jax.random.PRNGKey(0)))
out_intrinsic = model_intrinsic.predict_rest_sed(p_intrinsic)
wave_int = np.asarray(out_intrinsic.wavelength)
nu_l_nu_intrinsic = C_AA_PER_S / wave_int * np.asarray(out_intrinsic.sed)

# Attenuated but dust-free version (to isolate stellar attenuation)
model_attenuated = tengri.SEDModel.build(
    ssp,
    sfh={
        "type": "const",
        "*": tengri.FIXED,
        "log_total_mass": 10.0,
        "start_gyr": 0.1,
        "end_gyr": 0.0,
    },
    dust={
        "type": "two_component",
        "*": tengri.FIXED,
        "tau_diff": 3.0,
        "tau_bc": 2.5,
        "emission": {"type": "dale2014", "*": tengri.FIXED},
    },
    radio={"type": "condon92", "*": tengri.FIXED},
    redshift=tengri.Fixed(0.018),
)
p_att = dict(model_attenuated.spec.sample(jax.random.PRNGKey(0)))
out_att = model_attenuated.predict_rest_sed(p_att)
wave_att = np.asarray(out_att.wavelength)
nu_l_nu_att = C_AA_PER_S / wave_att * np.asarray(out_att.sed)

# Main plot: one panel showing composite SED with annotations
fig, ax = plt.subplots(figsize=(7.6, 4.6))

# Plot the main panchromatic SED (stellar attenuated + dust re-emission + radio)
mask = np.asarray(out.sed) > 0
ax.loglog(
    wave[mask], nu_l_nu[mask], color="0.15", lw=1.4, label="Full SED (attenuated + dust + radio)"
)

# Overplot intrinsic stellar for reference (fainter grey)
mask_int = np.asarray(out_intrinsic.sed) > 0
ax.loglog(
    wave_int[mask_int],
    nu_l_nu_intrinsic[mask_int],
    color="0.65",
    lw=0.8,
    ls="--",
    alpha=0.5,
    label="Stellar intrinsic (dust-free)",
)

# Annotations for key features
annotations = [
    (1500, 5e45, "UV\n(suppressed)"),
    (5000, 3e45, "Optical\n(deeply attenuated)"),
    (1e5, 8e45, "Warm\ndust"),
    (6e5, 1.2e46, "FIR peak\n(60 μm)"),
    (3e7, 2e45, "Free-free\nknee"),
    (2e8, 5e44, "Synchrotron"),
]

for x, y, label in annotations:
    ax.text(x, y, label, fontsize=6.5, color="0.4", ha="center", va="center")

# Styling
ax.set(
    xlim=(1e3, 3e9),
    ylim=(1e44, 2e46),
    xlabel=r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]",
    ylabel=r"$\nu L_\nu$  [erg s$^{-1}$]",
)

# Bolometric IR luminosity annotation
ir_mask = (wave > 8e4) & (wave < 1e7)
nu_ir = C_AA_PER_S / wave[ir_mask]
order_ir = np.argsort(nu_ir)
L_ir = np.trapezoid(np.asarray(out.sed)[ir_mask][order_ir], nu_ir[order_ir])
l_sun_cgs = 3.839e33

ax.text(
    0.97,
    0.05,
    rf"$L_{{\rm IR}}^{{(8\text{{–}}1000\,\mu\mathrm{{m}})}} \approx 10^{{{np.log10(L_ir / l_sun_cgs):.1f}}}\,L_\odot$",
    transform=ax.transAxes,
    ha="right",
    fontsize=9,
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=0.5),
)

ax.legend(frameon=False, fontsize=7.5, loc="lower left")

fig.tight_layout()
plt.savefig("plot_panchromatic_dusty_starburst.png", dpi=150, bbox_inches="tight")
