"""
QSOgen emission lines: Baldwin effect across AGN luminosity
===========================================================

The QSOgen model (Temple+2021) includes empirical UV/optical emission-line
forest and broad Balmer continuum. The relative strength of these line
features with respect to the continuum obeys the **Baldwin effect**:
luminous quasars show weaker equivalent-width emission lines (the line
flux grows sublinearly with continuum). This sweep shows the Baldwin
effect in the QSOgen template across six decades of bolometric luminosity
(log L_bol = 9 to 13 L_sun), revealing the Ly-alpha + C IV feature cluster
around 1000–1600 Å and optical hydrogen Balmer lines (Hα, Hβ).

References
----------
.. [1] Temple, M. J., Hewett, P. C., & Banerji, M. 2021, MNRAS, 508, 737
   — QSOgen empirical template with Baldwin effect parametrization.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import jax
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.plot import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")
warnings.filterwarnings("ignore", message=".*deprecated.*")

C_AA_PER_S = 2.998e18

SFH = {"type": "const", "all_params": tengri.FIXED, "log_total_mass": -10.0}
DUST = {
    "law": "power_law",
    "type": "two_component",
    "all_params": tengri.FIXED,
    "tau_diff": 0.0,
    "tau_bc": 0.0,
}

ssp = tengri.load_ssp()

log_lbol_values = np.array([10.0, 11.0, 12.0, 13.0])
norm = mpl.colors.Normalize(vmin=log_lbol_values.min(), vmax=log_lbol_values.max())
cmap = plt.get_cmap("viridis")

fig, ax = plt.subplots(figsize=(6.5, 4.2))
for lbol in log_lbol_values:
    model = tengri.SEDModel.build(
        ssp,
        sfh=SFH,
        dust=DUST,
        agn={
            "disc": {"type": "qsogen", "all_params": tengri.FIXED},
            "nlr": {"type": "none", "all_params": tengri.FIXED},
            "blr": {"type": "qsogen", "all_params": tengri.FIXED},
            "all_params": tengri.FIXED,
            "lum_ratio": 1.0,
            "log_lbol": lbol,
        },
        redshift=tengri.Fixed(0.0),
    )
    params = dict(model.spec.sample(jax.random.PRNGKey(0)))
    out = model.predict(params)
    wave = np.asarray(model.wavelengths)
    nu_lnu = (C_AA_PER_S / wave) * np.asarray(out.rest_sed())
    ax.loglog(wave, nu_lnu, color=cmap(norm(lbol)), lw=1.4)

cbar = fig.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
    ax=ax,
    pad=0.01,
    label=r"$\log L_{\rm bol}$ [$L_\odot$]",
)

ax.set(
    xlim=(100, 1e6),
    xlabel=r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]",
    ylabel=r"$\nu L_\nu$  [erg s$^{-1}$]",
)
ax.grid(True, which="minor", alpha=0.2, linestyle=":")

fig.tight_layout()
plt.savefig("plot_agn_qsogen_emline_sweep.png", dpi=150, bbox_inches="tight")
