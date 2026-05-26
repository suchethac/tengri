"""
AGN disc continuum: model comparison at fixed L_bol
=====================================================

Six accretion-disc backbones at fixed bolometric luminosity
``log L_bol = 12.5`` (in log L_sun), evaluated in isolation with the
host suppressed and no torus/lines/dust. The differences between the
curves are entirely how each model partitions the disc power across
wavelength: pure blackbody vs warm Comptonization, relativistic vs
Newtonian potential, empirical-fit vs first-principles continuum.

Models compared (the six production disc selectors under
``agn.disc.type``):
- ``multicolor``   — Shakura–Sunyaev α-disc (Kubota & Done 2018)
- ``kubota_done``  — same family, full warm-Compton treatment
- ``qsogen``       — Temple+2021 empirical type-1 SED
- ``grahsp_sbpl``  — Lussier+2023 GRAHSP broken power-law
- ``powerlaw``     — generic power-law disc
- ``adaf``         — radiatively inefficient accretion flow (Mahadevan 1997)
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import jax
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

DISC_MODELS = [
    ("qsogen", "QSOGEN (Temple+2021)"),
    ("multicolor", "multicolor disc (K&D 2018)"),
    ("kubota_done", "Kubota & Done 2018 (full)"),
    ("grahsp_sbpl", "GRAHSP broken power-law"),
    ("powerlaw", "power-law disc"),
    ("adaf", "ADAF (Mahadevan 1997)"),
]
COLORS = plt.cm.viridis(np.linspace(0.05, 0.92, len(DISC_MODELS)))

C_AA_PER_S = 2.998e18
SFH = {"type": "const", "*": tengri.FIXED, "log_sfr": -10.0}
DUST = {"type": "two_component", "*": tengri.FIXED, "tau_diff": 0.0, "tau_bc": 0.0}

ssp = tengri.load_ssp()
fig, ax = plt.subplots(figsize=(7.2, 4.6))

for (disc, label), color in zip(DISC_MODELS, COLORS):
    model = tengri.SEDModel.build(
        ssp,
        sfh=SFH,
        dust=DUST,
        agn={
            "disc": {"type": disc, "*": tengri.FIXED},
            "*": tengri.FIXED,
            "log_lbol": 12.5,
            "frac": 1.0,
        },
        redshift=tengri.Fixed(0.05),
    )
    p = dict(model.spec.sample(jax.random.PRNGKey(0)))
    out = model.predict_rest_sed(p)
    wave = np.asarray(out.wavelength)
    nu_l_nu = C_AA_PER_S / wave * np.asarray(out.sed)
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
ax.legend(frameon=False, fontsize=8, loc="lower center")

fig.tight_layout()
plt.savefig("plot_agn_disc_compare.png", dpi=150, bbox_inches="tight")
