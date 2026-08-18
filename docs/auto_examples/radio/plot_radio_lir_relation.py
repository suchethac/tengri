"""
FIR-radio correlation: L_IR × q_IR sets radio loudness scale
=============================================================

The FIR-radio correlation links far-infrared luminosity (dust-reprocessed
star-formation energy) to 1.4 GHz synchrotron emission. The dimensionless
q_IR parameter relates the two via L_IR ∝ L_1.4GHz^(10^q_IR/2.5). Brighter
starbursts emit stronger radio across all frequencies. We sweep L_IR over
10^10–10^13 L_sun at fixed q_IR = 2.64 (canonical; Bell 2003).

Reference: Bell 2003, ApJ 586, 794.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from tengri.plot import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

from tengri.radio import radio_star_forming

wave = jnp.logspace(7, 11, 600)  # Angstrom: 1 mm = 1e7 Å, 10 m = 1e11 Å

log_L_ir = np.array([10.0, 11.0, 12.0, 13.0])
L_ir_values = 10.0**log_L_ir
norm = mpl.colors.Normalize(vmin=log_L_ir.min(), vmax=log_L_ir.max())
cmap = plt.get_cmap("viridis")

q_ir = 2.64

fig, ax = plt.subplots(figsize=(6.5, 4.2))

all_lnu = []
for L_ir, log_lir in zip(L_ir_values, log_L_ir):
    L_nu = radio_star_forming(wave, L_ir=L_ir, q_ir=q_ir, alpha_sf=0.8)
    all_lnu.append(L_nu)
    nu_ghz = (3e18 / np.array(wave)) / 1e9
    ax.loglog(nu_ghz, np.array(L_nu), color=cmap(norm(log_lir)), lw=1.4)

ax.set_xlabel(r"Frequency $\nu$ [GHz]")
ax.set_ylabel(r"$L_\nu$ [erg s$^{-1}$ Hz$^{-1}$]")
ax.invert_xaxis()
ax.set_xlim(200, 0.1)
# Tighten y-limits based on data range
ymax = max([np.max(arr) for arr in all_lnu])
ymin = min([np.min(arr) for arr in all_lnu])
ax.set_ylim(ymin / 3.0, ymax * 3.0)

cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.01)
cbar.set_label(r"log$_{10}$ L$_{\rm IR}$ [L$_\odot$]")

fig.tight_layout()
plt.savefig("plot_radio_lir_relation.png", dpi=150, bbox_inches="tight")
