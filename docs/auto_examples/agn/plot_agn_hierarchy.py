"""
AGN SEDModel Hierarchy
===================

Compare all AGN model tiers in tengri: from simple power-law disc + single
torus (3 parameters) through the full unified NLR/BLR model (12+ parameters).
Each tier adds physical complexity. No SSP data required.

.. sphx-glr-precomputed-img:

.. image:: images/sphx_glr_plot_agn_hierarchy_001.png
   :alt: plot_agn_hierarchy
   :class: sphx-glr-single-img

"""

# TODO: refactor to SEDModel.build API (currently uses low-level internal API)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from tengri.agn import AGN_MODELS, resolve_agn_model
from tengri.analysis.plotting import setup_style

setup_style()

# %%
# Wavelength grid: 100 Angstrom (UV) to 500 micron (MIR)
wavelength = jnp.logspace(np.log10(100), np.log10(5e5), 1000)
wave_um = np.array(wavelength) / 1e4  # micron for plotting

# Fixed bolometric luminosity: L_bol = 10^12 Lsun ~ 10^45.6 erg/s (typical QSO)
log_lbol = 12.0

# %%
# Evaluate each registered AGN model
# ------------------------------------
# Models shown in order of increasing physical complexity.
# "standard" is omitted because at the same log_Lbol it is degenerate with
# "kubota_done" (same disc defaults, same torus parameterization).
# "relagn" is self-normalizing from BH physics (M, Mdot, spin): parameters
# chosen so that log_Lbol ≈ 12.1 Lsun matches the reference.

model_order = ["simple", "kubota_done", "skirtor", "unified_nlr_blr", "qsogen", "relagn"]
colors = plt.cm.Set1(np.linspace(0.0, 0.9, len(model_order)))
param_counts = {
    "simple": 3,
    "kubota_done": 8,
    "skirtor": 7,
    "unified_nlr_blr": "12+",
    "qsogen": 7,
    "relagn": 8,
}

# Model-specific kwargs: relagn is self-normalizing (no agn_log_lbol).
# log_mbh=8.5, log_mdot=-1.0, a=0.5 gives log_Lbol ≈ 12.1 Lsun.
model_kwargs = {
    "relagn": dict(agn_log_mbh=8.5, agn_log_mdot=-1.0, agn_astar=0.5),
}

fig, ax = plt.subplots(figsize=(10, 6))

for name, color in zip(model_order, colors):
    if name not in AGN_MODELS:
        continue
    try:
        model_fn = resolve_agn_model(name)
        kwargs = model_kwargs.get(name, {"agn_log_lbol": log_lbol})
        l_nu = model_fn(wavelength, **kwargs)
        l_nu_np = np.array(l_nu)
        if np.all(l_nu_np <= 0) or np.any(np.isnan(l_nu_np)):
            continue
        n_params = param_counts.get(name, "?")
        ax.loglog(
            wave_um,
            l_nu_np,
            lw=1.8,
            color=color,
            label=f"{name} ({n_params} params)",
        )
    except Exception:
        continue

ax.set_xlabel(r"Wavelength [$\mu$m]")
ax.set_ylabel(r"$L_\nu$ [erg s$^{-1}$ Hz$^{-1}$]")
ax.set_title(rf"AGN SEDModel Hierarchy at $\log L_{{\mathrm{{bol}}}} = {log_lbol:.0f}$")
ax.set_xlim(0.01, 50)
ax.set_ylim(1e27, 1e32)
ax.legend(frameon=False, fontsize=10)
fig.tight_layout()
# Each tier adds physical complexity:
#
# - **simple**: power-law disc + single-T torus (fast, 3 free params)
# - **kubota_done**: Kubota & Done (2018) 3-zone disc + two-T torus (8 params)
# - **skirtor**: power-law disc + SKIRTOR Monte Carlo torus (Stalevski+2012, 7 params)
# - **unified_nlr_blr**: kubota_done disc + NLR/BLR line regions (12+ params)
# - **qsogen**: Temple, Hewett & Banerji (2021) empirical quasar SED (7 params)
# - **relagn**: RELAGN outer disc (Hagen & Done 2023) + KYCONV Kerr ray-tracing;
#   self-normalizing from BH physics — shown at log_Lbol ≈ 12.1 (8 params)

plt.savefig("plot_agn_hierarchy.png", dpi=150, bbox_inches="tight")
