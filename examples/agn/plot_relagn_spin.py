"""
RELAGN Spin Sweep
=================

Demonstrate the effect of BH spin on the relativistic outer-disc SED using
the RELAGN model (Hagen & Done 2023) with KYCONV Kerr-metric ray-tracing
(Dovciak, Karas & Yaqoob 2004).

Higher spin → smaller ISCO → hotter inner disc → bluer UV peak.
Requires the precomputed grid ``data/relagn_disc_grid.h5``.
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from tengri.agn import get_agn_model
from tengri.analysis.plotting import setup_style

setup_style()

# %%
# Wavelength grid: 100 Å (UV) to 3 µm (NIR)
wavelength = jnp.logspace(np.log10(100), np.log10(3e4), 800)
wave_um = np.array(wavelength) / 1e4

# Fixed BH properties
log_mbh = 8.5    # log10(M / Msun)
log_mdot = -0.5  # log10(Mdot / Mdot_Edd)

# Spin nodes to plot — prograde range [0, 0.998]
astar_values = [0.0, 0.3, 0.6, 0.9, 0.998]
colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(astar_values)))

# %%
# Load RELAGN model from registry
try:
    relagn_fn = get_agn_model("relagn")
except KeyError as exc:
    raise SystemExit("relagn model not registered — rebuild grid first.") from exc

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax_sed, ax_peak = axes

peak_wave = []
peak_lnu = []

for astar, color in zip(astar_values, colors):
    l_nu = np.array(relagn_fn(
        wavelength,
        agn_log_mbh=log_mbh,
        agn_log_mdot=log_mdot,
        agn_astar=float(astar),
        agn_cos_inc=0.5,
        agn_torus_frac=0.0,  # disc only for clarity
    ))
    mask = l_nu > 0
    if not mask.any():
        continue

    ax_sed.loglog(
        wave_um[mask], l_nu[mask], lw=1.8, color=color,
        label=rf"$a_* = {astar:.3f}$",
    )
    # UV peak (shortest wavelength at maximum L_nu)
    uv_mask = (np.array(wavelength) < 3000) & mask
    if uv_mask.any():
        i_pk = np.argmax(l_nu[uv_mask])
        peak_wave.append(wave_um[uv_mask][i_pk])
        peak_lnu.append(l_nu[uv_mask][i_pk])
    else:
        peak_wave.append(np.nan)
        peak_lnu.append(np.nan)

ax_sed.set_xlabel(r"Wavelength [$\mu$m]")
ax_sed.set_ylabel(r"$L_\nu$ [erg s$^{-1}$ Hz$^{-1}$]")
ax_sed.set_title(
    rf"RELAGN disc: $\log M = {log_mbh}$, $\log \dot{{m}} = {log_mdot}$"
)
ax_sed.set_xlim(0.01, 3)
ax_sed.legend(frameon=False, fontsize=9)

# %%
# Panel 2: UV peak shift vs spin
ax_peak.plot(
    astar_values[:len(peak_wave)],
    [pw * 1e4 for pw in peak_wave],  # back to Angstrom
    "o-", color="royalblue", lw=2, ms=7,
)
ax_peak.set_xlabel(r"BH spin $a_*$")
ax_peak.set_ylabel(r"UV peak wavelength [$\AA$]")
ax_peak.set_title("Disc UV peak shifts blueward with spin")
ax_peak.invert_yaxis()

fig.tight_layout()
# Higher spin concentrates emission at smaller ISCO radii.
# KYCONV Kerr-metric ray-tracing (Dovciak+2004) computes the photon orbit
# correction per annulus up to r = 1000 Rg; beyond that RELAGN uses the
# non-relativistic formula.

plt.savefig("relagn_spin_sweep.png", dpi=150, bbox_inches="tight")
plt.show()
