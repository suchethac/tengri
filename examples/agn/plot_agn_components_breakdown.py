"""
AGN composite SED: per-block decomposition
============================================

A single ``log L_bol = 12.5`` composable AGN built up component by
component — disc alone, +torus, +narrow lines, +broad lines — so the
reader can see what each block contributes to the total spectrum.
The bottom panel shows the same decomposition stacked so the layers
add up to the full SED.

This is the diagnostic figure for "where does the AGN signal in my
data come from?" — broad-line decompositions need ``blr``, NLR
fitters need ``nlr``, NIR/MIR colour fitters need ``torus`` (and
disc choice barely matters longward of 1 μm), etc.
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

C_AA_PER_S = 2.998e18

ssp = tengri.load_ssp()
COMMON = dict(
    sfh={"type": "const", "*": tengri.FIXED, "log_sfr": -10.0},
    dust={"type": "two_component", "*": tengri.FIXED, "tau_diff": 0.0, "tau_bc": 0.0},
    redshift=tengri.Fixed(0.05),
)
BASE_AGN = dict(disc={"type": "multicolor", "*": tengri.FIXED})


def _agn(extra_blocks=()):
    agn = {"*": tengri.FIXED, "log_lbol": 12.5, "frac": 1.0, **BASE_AGN}
    for key, value in extra_blocks:
        agn[key] = value
    model = tengri.SEDModel.build(ssp, agn=agn, **COMMON)
    p = dict(model.spec.sample(jax.random.PRNGKey(0)))
    out = model.predict_rest_sed(p)
    return np.asarray(out.wavelength), np.asarray(out.sed)


configs = [
    ("disc only", (), "#8b4513"),
    ("+ torus (SKIRTOR)", (("torus", {"type": "skirtor", "*": tengri.FIXED}),), "#cc7733"),
    (
        "+ NLR",
        (
            ("torus", {"type": "skirtor", "*": tengri.FIXED}),
            ("lines", {"type": "nlr", "*": tengri.FIXED}),
        ),
        "#5588cc",
    ),
    (
        "+ BLR",
        (
            ("torus", {"type": "skirtor", "*": tengri.FIXED}),
            ("lines", {"type": "blr", "*": tengri.FIXED}),
        ),
        "#4477aa",
    ),
    (
        "+ FeII",
        (
            ("torus", {"type": "skirtor", "*": tengri.FIXED}),
            ("lines", {"type": "blr", "*": tengri.FIXED}),
            ("feii", {"type": "grahsp", "*": tengri.FIXED}),
        ),
        "#dd6699",
    ),
]

waves, seds = {}, {}
for label, blocks, _ in configs:
    w, s = _agn(blocks)
    waves[label] = w
    seds[label] = s

wave = waves["disc only"]
nu = C_AA_PER_S / wave

fig, (ax_top, ax_bot) = plt.subplots(
    2,
    1,
    figsize=(7.4, 6.8),
    sharex=True,
    gridspec_kw={"hspace": 0.05, "height_ratios": [3, 2]},
)

for label, _, color in configs:
    ax_top.loglog(wave, nu * seds[label], color=color, lw=1.5, label=label)
ax_top.set(ylabel=r"$\nu L_\nu$  [erg s$^{-1}$]", ylim=(1e42, 5e46))
ax_top.legend(frameon=False, fontsize=8, loc="lower center")

disc_only = seds["disc only"]
torus_contrib = seds["+ torus (SKIRTOR)"] - disc_only
nlr_contrib = seds["+ NLR"] - seds["+ torus (SKIRTOR)"]
blr_contrib = seds["+ BLR"] - seds["+ torus (SKIRTOR)"]
feii_contrib = seds["+ FeII"] - seds["+ BLR"]

ax_bot.loglog(wave, nu * disc_only, color="#8b4513", lw=1.2, label="disc")
ax_bot.loglog(
    wave,
    nu * np.where(torus_contrib > 0, torus_contrib, np.nan),
    color="#cc7733",
    lw=1.2,
    label="torus",
)
ax_bot.loglog(
    wave,
    nu * np.where(nlr_contrib > 0, nlr_contrib, np.nan),
    color="#5588cc",
    lw=1.2,
    label="NLR lines",
)
ax_bot.loglog(
    wave,
    nu * np.where(blr_contrib > 0, blr_contrib, np.nan),
    color="#4477aa",
    lw=1.2,
    ls=":",
    label="BLR lines",
)
ax_bot.loglog(
    wave,
    nu * np.where(feii_contrib > 0, feii_contrib, np.nan),
    color="#dd6699",
    lw=1.2,
    label="FeII",
)
ax_bot.set(
    xlabel=r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]",
    ylabel=r"$\nu L_\nu^{\rm block}$  [erg s$^{-1}$]",
    xlim=(80, 2e6),
    ylim=(1e42, 5e46),
)
ax_bot.legend(frameon=False, fontsize=8, loc="lower right")

plt.savefig("plot_agn_components_breakdown.png", dpi=150, bbox_inches="tight")
