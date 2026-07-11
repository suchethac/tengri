"""
SDSS Filter Transmission Curves
================================

Plot the ugriz filter transmission curves from the SDSS photometric system.
Filters are loaded from the SVO Filter Profile Service via tengri's filter registry.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style

setup_style()

filter_names = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]
_waves, _trans, curves = tengri.load_filter_set(filter_names)

# Band colors matching standard SDSS convention
band_colors = {
    "sdss_u": "#4B0082",
    "sdss_g": "#00AA00",
    "sdss_r": "#CC0000",
    "sdss_i": "#DD6600",
    "sdss_z": "#880000",
}

fig, ax = plt.subplots(figsize=(8, 4))

for fc, name in zip(curves, filter_names):
    wave_ang = np.array(fc.wave)
    trans = np.array(fc.trans)
    color = band_colors[name]
    ax.fill_between(wave_ang, 0, trans, alpha=0.25, color=color)
    ax.plot(wave_ang, trans, lw=1.5, color=color, label=name.replace("sdss_", ""))

ax.set_xlabel(r"Wavelength [$\AA$]")
ax.set_ylabel("Transmission")
ax.set_xlim(2800, 11500)
ax.set_ylim(0, None)
ax.legend(frameon=False, ncol=5, loc="upper right")
fig.tight_layout()

plt.savefig("plot_filter_curves.png", dpi=150, bbox_inches="tight")
