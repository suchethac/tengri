"""
AGN SEDModel Hierarchy
===================

Compare all AGN model tiers in tengri: from simple power-law disc + single
torus (3 parameters) through the full unified NLR/BLR model (12+ parameters).
Each tier adds physical complexity. No SSP data required.
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from tengri.agn import AGN_MODELS, get_agn_model
from tengri.analysis.plotting import setup_style

setup_style()

# %%
# Wavelength grid: 100 Angstrom (UV) to 500 micron (MIR)
wavelength = jnp.logspace(np.log10(100), np.log10(5e5), 1000)
wave_um = np.array(wavelength) / 1e4  # micron for plotting

# Fixed bolometric luminosity: L_bol = 10^45 erg/s ~ 10^{11.6} Lsun
log_lbol = 45.0

# %%
# Evaluate each registered AGN model
# ------------------------------------
# The registry contains: simple, standard, kubota_done, skirtor,
# unified_nlr_blr, qsogen. Some may need external data; skip gracefully.

model_order = ["simple", "standard", "kubota_done", "skirtor", "unified_nlr_blr", "qsogen"]
colors = plt.cm.Set1(np.linspace(0.0, 0.8, len(model_order)))
param_counts = {
    "simple": 3,
    "standard": 6,
    "kubota_done": 8,
    "skirtor": 7,
    "unified_nlr_blr": "12+",
    "qsogen": 7,
}

fig, ax = plt.subplots(figsize=(10, 6))

for name, color in zip(model_order, colors):
    if name not in AGN_MODELS:
        continue
    try:
        model_fn = get_agn_model(name)
        l_nu = model_fn(wavelength, agn_log_lbol=log_lbol)
        l_nu_np = np.array(l_nu)
        # Skip if all zeros or NaN
        if np.all(l_nu_np <= 0) or np.any(np.isnan(l_nu_np)):
            continue
        n_params = param_counts.get(name, "?")
        ax.loglog(
            wave_um, l_nu_np, lw=1.8, color=color,
            label=f"{name} ({n_params} params)",
        )
    except Exception:
        # SEDModel may require external data (e.g. SKIRTOR grids)
        continue

ax.set_xlabel(r"Wavelength [$\mu$m]")
ax.set_ylabel(r"$L_\nu$ [$L_\odot$ Hz$^{-1}$]")
ax.set_title(
    rf"AGN SEDModel Hierarchy at $\log L_{{\mathrm{{bol}}}} = {log_lbol:.0f}$"
)
ax.set_xlim(0.01, 50)
ax.legend(frameon=False, fontsize=10)
fig.tight_layout()

# %%
# Each tier adds physical complexity:
#
# - **simple**: power-law disc + single-T torus (fast, 3 free params)
# - **standard**: multi-color Shakura-Sunyaev disc + two-T torus (6 params)
# - **kubota_done**: Kubota & Done (2018) disc with BH spin + clumpy torus (8+ params)
# - **skirtor**: power-law disc + SKIRTOR clumpy torus (Stalevski+2012, 7 params)
# - **unified_nlr_blr**: kubota_done + NLR/BLR line regions (12+ params)
# - **qsogen**: Temple, Hewett & Banerji (2021) empirical quasar SED (7 params)

plt.savefig("agn_hierarchy.png", dpi=150, bbox_inches="tight")
plt.show()
