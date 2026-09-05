"""
AGN disc continuum: every registered model at fixed L_bol
=========================================================

All thirteen accretion-disc backbones registered under ``agn.disc.type``,
at fixed bolometric luminosity ``log L_bol = 12.5`` (in log L_sun),
evaluated in isolation with the host suppressed and no torus/lines/dust.
The differences between the curves are entirely how each model partitions
the disc power across wavelength: pure blackbody vs warm Comptonization,
relativistic vs Newtonian potential, radiatively efficient thin disc vs
inefficient ADAF, empirical composite vs first-principles continuum.

Swap any one into a full model with ``agn={'disc': {'type': <name>}}``.
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

DISC_MODELS = [
    ("multicolor", "multicolor thin disc (Shakura–Sunyaev)"),
    ("kubota_done", "Kubota & Done 2018 (3-zone)"),
    ("relagn", "RELAGN (relativistic Kerr)"),
    ("qsogen", "QSOGEN (Temple+2021)"),
    ("richards2006", "Richards+2006 SDSS composite"),
    ("slone_netzer", "Slone & Netzer 2012"),
    ("schartmann2005", "Schartmann 2005 (X-CIGALE)"),
    ("schartmann2005_skirtor_atten", "Schartmann 2005 + SKIRTOR atten."),
    ("skirtor", "SKIRTOR empirical (Stalevski+2016)"),
    ("grahsp_sbpl", "GRAHSP bending power-law"),
    ("powerlaw", "power-law + UV cutoff"),
    ("adaf", "ADAF (Mahadevan 1997)"),
    ("adaf_lopez2024", "ADAF–thin blend (X-CIGALE)"),
]
# Qualitative palette — 13 unordered models need distinguishable hues, not a
# sequential colormap.
COLORS = plt.cm.tab20(np.linspace(0, 1, 20))[: len(DISC_MODELS)]

C_AA_PER_S = 2.998e18
SFH = {"type": "const", "all_params": tengri.Fixed(tengri.DEFAULT), "log_total_mass": -10.0}
DUST = {
    "law": "power_law",
    "type": "two_component",
    "all_params": tengri.Fixed(tengri.DEFAULT),
    "tau_diff": 0.0,
    "tau_bc": 0.0,
}

# `powerlaw` is a bare phenomenological disc that tengri deprecates for science
# fits (use multicolor or kubota_done for that). It is on this panel to show
# what a power-law gives up against a physically derived disc. The
# DeprecationWarning is expected here only.
warnings.filterwarnings("ignore", message=".*powerlaw_disc is deprecated.*")

ssp = tengri.load_ssp()
fig, ax = plt.subplots(figsize=(7.2, 4.6))

for (disc, label), color in zip(DISC_MODELS, COLORS):
    model = tengri.SEDModel.build(
        ssp,
        sfh=SFH,
        dust_attenuation=DUST,
        agn={
            "disc": {"type": disc, "all_params": tengri.Fixed(tengri.DEFAULT)},
            "all_params": tengri.Fixed(tengri.DEFAULT),
            "log_lbol": 12.5,
            "lum_ratio": 1.0,
        },
        redshift=tengri.Fixed(0.05),
    )
    p = dict(model.spec.sample(jax.random.PRNGKey(0)))
    out = model.predict(p)
    wave = np.asarray(model.wavelengths)
    nu_l_nu = C_AA_PER_S / wave * np.asarray(out.rest_sed())
    ax.loglog(wave, nu_l_nu, color=color, lw=1.4, label=label)

ax.set(
    xlim=(20, 3e5),
    ylim=(1e41, 5e47),
    xlabel=r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]",
    ylabel=r"$\nu L_\nu$  [erg s$^{-1}$]",
)
ax.axvspan(1, 100, color="0.93", alpha=0.6, lw=0)
ax.text(30, 2e47, "X-ray", color="0.4", fontsize=8, va="top")
ax.text(2000, 2e47, "UV/optical BBB", color="0.4", fontsize=8, va="top")
ax.text(2e5, 2e47, "NIR cutoff", color="0.4", fontsize=8, va="top", ha="right")
ax.legend(frameon=False, fontsize=7, loc="lower center", ncol=2)

fig.tight_layout()
plt.savefig("plot_agn_disc_compare.png", dpi=150, bbox_inches="tight")
