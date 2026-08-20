"""
Cumulative buildup of the GRAHSP AGN recipe, one sub-block at a time
=====================================================================

The ``agn.disc``, ``agn.lines``, ``agn.feii``, ``agn.torus``,
``agn.atten`` sub-blocks of ``SEDModel.build`` are composable: turning
one on at a time and overlaying the all-on reference (dashed gray)
shows which features each sub-block contributes. Five panels at fixed
log L_bol = 12.0, all built via the public nested-dict grammar:

1. disc only (GRAHSP broken power-law)
2. + GRAHSP narrow + broad lines
3. + GRAHSP Fe II forest
4. + GRAHSP log-Gaussian torus
5. + GRAHSP bi-attenuation curve  (= the reference recipe)

Reference: Buchner et al. 2024 (GRAHSP recipe).
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import jax
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.plot import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

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

BLOCK_PROGRESSION = [
    ("disc only", {"disc": {"type": "grahsp_sbpl", "all_params": tengri.FIXED}}),
    (
        "+ lines",
        {
            "disc": {"type": "grahsp_sbpl", "all_params": tengri.FIXED},
            "nlr": {"type": "grahsp", "all_params": tengri.FIXED},
            "blr": {"type": "grahsp", "all_params": tengri.FIXED},
        },
    ),
    (
        "+ Fe II",
        {
            "disc": {"type": "grahsp_sbpl", "all_params": tengri.FIXED},
            "nlr": {"type": "grahsp", "all_params": tengri.FIXED},
            "blr": {"type": "grahsp", "all_params": tengri.FIXED},
            "feii": {"type": "grahsp", "all_params": tengri.FIXED},
        },
    ),
    (
        "+ torus",
        {
            "disc": {"type": "grahsp_sbpl", "all_params": tengri.FIXED},
            "nlr": {"type": "grahsp", "all_params": tengri.FIXED},
            "blr": {"type": "grahsp", "all_params": tengri.FIXED},
            "feii": {"type": "grahsp", "all_params": tengri.FIXED},
            "torus": {"type": "grahsp", "all_params": tengri.FIXED},
        },
    ),
    (
        "+ attenuation (full)",
        {
            "disc": {"type": "grahsp_sbpl", "all_params": tengri.FIXED},
            "nlr": {"type": "grahsp", "all_params": tengri.FIXED},
            "blr": {"type": "grahsp", "all_params": tengri.FIXED},
            "feii": {"type": "grahsp", "all_params": tengri.FIXED},
            "torus": {"type": "grahsp", "all_params": tengri.FIXED},
            "atten": {"type": "grahsp_biatten", "all_params": tengri.FIXED},
        },
    ),
]


def predict_nu_lnu(blocks):
    agn = {"all_params": tengri.FIXED, "log_lbol": 12.0, "lum_ratio": 1.0, **blocks}
    model = tengri.SEDModel.build(ssp, sfh=SFH, dust=DUST, agn=agn, redshift=tengri.Fixed(0.0))
    p = dict(model.spec.sample(jax.random.PRNGKey(0)))
    out = model.predict(p)
    wave_um = np.asarray(model.wavelengths) * 1.0e-4
    nu_lnu = C_AA_PER_S / np.asarray(model.wavelengths) * np.asarray(out.rest_sed())
    return wave_um, np.where(nu_lnu > 0, nu_lnu, np.nan)


wave_um, full_sed = predict_nu_lnu(BLOCK_PROGRESSION[-1][1])

fig, axes = plt.subplots(1, 5, figsize=(15.0, 3.6), sharey=True)
colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(BLOCK_PROGRESSION)))
for ax, (label, blocks), color in zip(axes, BLOCK_PROGRESSION, colors):
    _, panel_sed = predict_nu_lnu(blocks)
    ax.loglog(wave_um, full_sed, lw=1.0, color="0.6", ls="--", label="full recipe")
    ax.loglog(wave_um, panel_sed, lw=1.8, color=color, label=label)
    ax.set_xlim(5.0e-3, 1.0e2)
    ax.set_ylim(1.0e42, 1.0e47)
    ax.set_xlabel(r"$\lambda$  [$\mu$m]")
    ax.text(0.04, 0.95, label, transform=ax.transAxes, va="top", fontsize=9)
    ax.legend(loc="lower center", fontsize=8, frameon=False)

axes[0].set_ylabel(r"$\nu L_\nu$  [erg s$^{-1}$]")
fig.tight_layout()
plt.savefig("plot_composable_block_toggles.png", dpi=150, bbox_inches="tight")
