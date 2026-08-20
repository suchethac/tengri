"""
Building up an AGN SED: disc, then torus, then lines
=====================================================

Four AGN configurations of increasing physical complexity at the same
bolometric luminosity (log L_bol = 12.5 in L_sun units) — bare
multicolor disc, +SKIRTOR torus, +NLR narrow-line forest, and an
empirical QSOgen template that bundles all of the above. The reader
sees which spectral feature each block introduces (mid-IR torus bump,
optical narrow lines, broad UV continuum) and which are essentially
universal across the modeling choice.

Reference: Kubota & Done 2018, MNRAS, 480, 1247 (multicolor disc);
Stalevski et al. 2016, MNRAS, 458, 2288 (SKIRTOR);
Temple, Hewett & Banerji 2021, MNRAS, 508, 737 (QSOgen).
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
LOG_LBOL = 12.5
SFH = {"type": "const", "all_params": tengri.FIXED, "log_total_mass": -10.0}
DUST = {
    "law": "power_law",
    "type": "two_component",
    "all_params": tengri.FIXED,
    "tau_diff": 0.0,
    "tau_bc": 0.0,
}

ssp = tengri.load_ssp()

TIERS = [
    (
        "bare multicolor disc",
        {"disc": {"type": "multicolor", "all_params": tengri.FIXED}},
    ),
    (
        "disc + SKIRTOR torus",
        {
            "disc": {"type": "multicolor", "all_params": tengri.FIXED},
            "torus": {"type": "skirtor", "all_params": tengri.FIXED},
        },
    ),
    (
        "disc + torus + NLR lines",
        {
            "disc": {"type": "multicolor", "all_params": tengri.FIXED},
            "torus": {"type": "skirtor", "all_params": tengri.FIXED},
            "nlr": {"type": "analytic", "all_params": tengri.FIXED},
            "blr": {"type": "none", "all_params": tengri.FIXED},
        },
    ),
    (
        "empirical QSOgen template",
        {"disc": {"type": "qsogen", "all_params": tengri.FIXED}},
    ),
]
colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(TIERS)))

fig, ax = plt.subplots(figsize=(7.5, 4.8))
for (label, blocks), color in zip(TIERS, colors):
    agn = {"all_params": tengri.FIXED, "log_lbol": LOG_LBOL, "lum_ratio": 1.0, **blocks}
    model = tengri.SEDModel.build(ssp, sfh=SFH, dust=DUST, agn=agn, redshift=tengri.Fixed(0.0))
    p = dict(model.spec.sample(jax.random.PRNGKey(0)))
    out = model.predict(p)
    wave = np.asarray(model.wavelengths)
    nu_l_nu = C_AA_PER_S / wave * np.asarray(out.rest_sed())
    ax.loglog(wave, nu_l_nu, color=color, lw=1.5, label=label)

ax.set(
    xlim=(100, 5.0e5),
    ylim=(1.0e42, 5.0e47),
    xlabel=r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]",
    ylabel=r"$\nu L_\nu$  [erg s$^{-1}$]",
)
ax.legend(frameon=False, fontsize=9, loc="lower center")
fig.tight_layout()
plt.savefig("plot_agn_hierarchy.png", dpi=150, bbox_inches="tight")
