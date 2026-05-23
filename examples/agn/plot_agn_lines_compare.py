"""
AGN emission-line backbones compared
======================================

Four production line backbones layered on top of the same disc +
torus at fixed ``log L_bol = 12.5``. The line backbone controls
which optical/UV emission features the model produces — narrow-line
region forbidden lines, broad-line permitted lines, or pre-canned
empirical line lists.

Line backbones compared (three production selectors under
``agn.lines.type``; the GRAHSP catalog needs an extra data bundle
not shipped with the gallery):
- ``nlr``     — Narrow-line region (Feltre+2016 photoionization grid)
- ``blr``     — Broad-line region (Cracco+2016 photoionization grid)
- ``qsogen``  — Temple+2021 empirical type-1 lines
"""

import warnings

import jax
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

LINES_MODELS = [
    ("nlr",    "NLR (Feltre+2016)"),
    ("blr",    "BLR (Cracco+2016)"),
    ("qsogen", "QSOGEN empirical"),
]
COLORS = plt.cm.viridis(np.linspace(0.05, 0.9, len(LINES_MODELS)))

C_AA_PER_S = 2.998e18
ssp = tengri.load_ssp()
fig, ax = plt.subplots(figsize=(7.4, 4.6))

for (line_kind, label), color in zip(LINES_MODELS, COLORS):
    model = tengri.SEDModel.build(
        ssp,
        sfh={"type": "const", "*": tengri.FIXED, "log_sfr": -10.0},
        dust={"type": "two_component", "*": tengri.FIXED,
              "tau_diff": 0.0, "tau_bc": 0.0},
        agn={
            "disc":  {"type": "multicolor", "*": tengri.FIXED},
            "torus": {"type": "skirtor",    "*": tengri.FIXED},
            "lines": {"type": line_kind,    "*": tengri.FIXED},
            "*": tengri.FIXED, "log_lbol": 12.5, "frac": 1.0,
        },
        redshift=tengri.Fixed(0.05),
    )
    p = dict(model.spec.sample(jax.random.PRNGKey(0)))
    out = model.predict_rest_sed(p)
    wave = np.asarray(out.wavelength)
    nu_l_nu = C_AA_PER_S / wave * np.asarray(out.sed)
    ax.semilogy(wave, nu_l_nu, color=color, lw=1.0, label=label, alpha=0.85)

# Zoom on the rest-frame optical/UV where the lines actually live.
ax.set_xlim(1000, 7500)
ax.set_ylim(1e44, 5e46)
ax.set_xlabel(r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]")
ax.set_ylabel(r"$\nu L_\nu$  [erg s$^{-1}$]")

# Mark the canonical AGN lines.
LINE_MARKS = [
    (1216, r"Ly$\alpha$"),
    (1549, "C IV"),
    (1909, "C III]"),
    (2798, "Mg II"),
    (4861, r"H$\beta$"),
    (5007, "[O III]"),
    (6563, r"H$\alpha$"),
]
for lam, name in LINE_MARKS:
    ax.axvline(lam, color="0.85", lw=0.4, alpha=0.5)
    ax.text(lam, 4e46, name, fontsize=7, color="0.5",
            ha="center", va="bottom", rotation=90)

ax.legend(frameon=False, fontsize=8, loc="lower right")

fig.tight_layout()
fig.savefig("plot_agn_lines_compare.png", dpi=150, bbox_inches="tight")
