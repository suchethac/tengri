"""
UV bump shape controlled by attenuation curve slope
====================================================

The 2175 Å UV bump sits atop a power-law continuum. Varying the slope
parameter δ (delta) in the Kriek & Conroy attenuation law steepens or
flattens the UV continuum, which changes the bump's prominence relative
to the surrounding curve. We zoom on rest-frame 1500–3500 Å to isolate
the bump region and show how δ ∈ [−1, +0.5] reshapes the attenuation curve.

The attenuation curve k(λ) is plotted directly (not a galaxy SED) to
reveal the dust opacity function shape — this isolates the physical effect
of slope on the bump without SED modeling confounds.

Reference: Kriek & Conroy 2013, ApJL, 775, L16 (Eq. 5).
"""

import warnings

import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from tengri.analysis.plotting import setup_style
from tengri.dust import resolve_dust_law

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

# Attenuation curve grid, zoomed on the bump region (1500–3500 Å)
wave = jnp.linspace(1500.0, 3500.0, 1000)

dust_fn = resolve_dust_law("kriek_conroy")

# Sweep delta (slope) from steeper (negative) to flatter (positive)
delta_values = np.linspace(-1.0, 0.5, 7)
norm = mpl.colors.Normalize(vmin=delta_values.min(), vmax=delta_values.max())
cmap = plt.get_cmap("viridis")

fig, ax = plt.subplots(figsize=(6.5, 4.2))
for delta in delta_values:
    k_lambda = dust_fn(wave, dust_bump_strength=1.0, dust_delta=delta)
    ax.plot(
        wave / 1e4,
        k_lambda,
        lw=1.4,
        color=cmap(norm(delta)),
    )

# Mark the 2175 Å bump for reference
ax.axvline(0.2175, ls=":", color="red", lw=1.0, alpha=0.6)
ax.text(0.2175, ax.get_ylim()[1] * 0.95, "2175 Å", fontsize=8, ha="center", color="red", alpha=0.7)

ax.set_xlim(0.15, 0.35)
ax.set_xlabel(r"Wavelength [$\mu$m]")
ax.set_ylabel(r"$k(\lambda)$ (normalized at 5500 $\mathrm{\AA}$)")

cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.01)
cbar.set_label(r"Slope parameter $\delta$")

fig.tight_layout()
plt.savefig("plot_uv_bump_strength_sweep.png", dpi=150, bbox_inches="tight")
