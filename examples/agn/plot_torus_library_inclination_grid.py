"""
AGN torus libraries across viewing angle: silicate feature and geometry
=========================================================================

Different radiative-transfer and empirical torus libraries encode the
Type-1↔Type-2 unified-model transition differently. SKIRTOR uses a 3D
clumpy model with a smooth density distribution and produces symmetric
silicate absorption/emission features. CAT3D-WIND employs a wind-like
clumpy geometry. Nenkova et al. (CLUMPY) offers a simpler analytical
approach. This grid shows how each library's silicate 9.7 μm feature and
overall IR reprocessing vary with inclination at fixed L_bol and (where
applicable) opening angle, revealing library-specific anisotropies and
feature depths.

Note: not all libraries expose identical parameter sets; fixed values are
used for those with no equivalent degree of freedom.

References
----------
.. [1] Stalevski, M., Fritz, J., Baes, M., et al. 2012, MNRAS, 420, 2756
   — SKIRTOR radiative-transfer torus.
.. [2] Hönig, S. F. & Kishimoto, M. 2017, ApJ, 838, L20 — CAT3D-WIND.
.. [3] Nenkova, M., Sirocky, M. M., Nikutta, R., et al. 2008, ApJ, 685, 147
   — CLUMPY models.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import jax
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")
warnings.filterwarnings("ignore", message=".*deprecated.*")

C_AA_PER_S = 2.998e18
LOG_LBOL = 12.0

SFH = {"type": "const", "*": tengri.FIXED, "log_total_mass": -10.0}
DUST = {"type": "two_component", "*": tengri.FIXED, "tau_diff": 0.0, "tau_bc": 0.0}

ssp = tengri.load_ssp()

inclination_deg = np.array([0.0, 30.0, 60.0, 80.0, 90.0])
cos_inc_values = np.cos(np.radians(inclination_deg))
norm = mpl.colors.Normalize(vmin=inclination_deg.min(), vmax=inclination_deg.max())
cmap = plt.get_cmap("viridis")

torus_configs = [
    ("SKIRTOR", {"type": "skirtor", "*": tengri.FIXED}),
    ("CAT3D-WIND", {"type": "cat3d_wind", "*": tengri.FIXED}),
    ("CLUMPY (Nenkova)", {"type": "nenkova", "*": tengri.FIXED}),
]

fig, axes = plt.subplots(1, len(torus_configs), figsize=(14, 4.5), sharey=True)

for ax_idx, (lib_name, torus_block) in enumerate(torus_configs):
    ax = axes[ax_idx]

    for cos_inc in cos_inc_values:
        model = tengri.SEDModel.build(
            ssp,
            sfh=SFH,
            dust=DUST,
            agn={
                "disc": {"type": "multicolor", "*": tengri.FIXED},
                "torus": torus_block,
                "*": tengri.FIXED,
                "log_lbol": LOG_LBOL,
                "frac": 1.0,
                "cos_inc": cos_inc,
            },
            redshift=tengri.Fixed(0.0),
        )
        params = dict(model.spec.sample(jax.random.PRNGKey(0)))
        out = model.predict(params)
        wave = np.asarray(model.wavelengths)
        nu_lnu = (C_AA_PER_S / wave) * np.asarray(out.rest_sed())
        inc_deg = np.degrees(np.arccos(cos_inc))
        ax.loglog(wave, nu_lnu, color=cmap(norm(inc_deg)), lw=1.2)

    ax.set(
        xlim=(100, 1e6),
        xlabel=r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]",
    )
    ax.set_title(lib_name, fontsize=11, fontweight="bold")
    ax.grid(True, which="minor", alpha=0.2, linestyle=":")

axes[0].set_ylabel(r"$\nu L_\nu$ [erg s$^{-1}$]")

cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cbar = fig.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
    cax=cbar_ax,
    label=r"Inclination $i$ [deg]",
)

fig.tight_layout(rect=[0, 0, 0.90, 1])
plt.savefig("plot_torus_library_inclination_grid.png", dpi=150, bbox_inches="tight")
