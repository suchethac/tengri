"""
AGN dusty torus: library comparison at fixed L_bol
=====================================================

All ten dusty-torus libraries registered under ``agn.torus.type``,
reprocessing the same accretion-disc continuum at fixed
``log L_bol = 12.5`` (in log L_sun) and standard inclination. The disc
is held at ``multicolor`` (Kubota & Done 2018) so the differences in the
curves are entirely how each torus library geometrically distributes hot
grains and re-emits the absorbed UV in the MIR — clumpy radiative
transfer (SKIRTOR, CLUMPY, CAT3D-WIND) vs smooth-dust grids (Fritz,
Silva) vs phenomenological graybodies.

``fritz`` and ``skirtor`` are the two CIGALE production tori — smooth
(Fritz+2006) versus clumpy (Stalevski+2016) — so contrasting them on
this panel isolates the smooth-vs-clumpy silicate-feature behavior
near 9.7 and 18 μm.
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

TORI = [
    ("skirtor", "SKIRTOR (Stalevski+2016)"),
    ("skirtor_agnfitter", "SKIRTOR_mean_3p (AGNfitter-rX)"),
    ("cat3d_wind", "CAT3D-WIND (Hönig & Kishimoto 2017)"),
    ("nenkova", "CLUMPY (Nenkova+2008)"),
    ("fritz", "Fritz+2006 smooth"),
    ("silva04", "Silva+2004 smooth"),
    ("grahsp", "GRAHSP IR torus + Si"),
    ("qsogen", "QSOGEN hot-dust blackbody"),
    ("simple", "single-T graybody"),
    ("two_temperature", "two-T graybody"),
]
# Qualitative palette — 10 unordered libraries need distinguishable hues.
COLORS = plt.cm.tab10(np.linspace(0, 1, 10))[: len(TORI)]

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
fig, ax = plt.subplots(figsize=(7.2, 4.6))

# `simple` and `two_temperature` are phenomenological graybodies, and tengri
# deprecates them for science fits (use SKIRTOR or Silva+04 for that). They are
# on this panel to show what a graybody gives up against a real radiative-transfer
# torus. The DeprecationWarning is expected here only.
warnings.filterwarnings("ignore", message=".*is a toy AGN torus model.*")

for (torus, label), color in zip(TORI, COLORS):
    model = tengri.SEDModel.build(
        ssp,
        sfh=SFH,
        dust_attenuation=DUST,
        agn={
            "disc": {"type": "multicolor", "all_params": tengri.FIXED},
            "torus": {"type": torus, "all_params": tengri.FIXED},
            "all_params": tengri.FIXED,
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
    xlim=(1e3, 3e6),
    ylim=(1e42, 5e46),
    xlabel=r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]",
    ylabel=r"$\nu L_\nu$  [erg s$^{-1}$]",
)
for um, name in [(3.0, "L"), (10.0, "N"), (24.0, "MIPS-24"), (70.0, "FIR")]:
    ax.axvline(um * 1.0e4, color="0.85", lw=0.4, alpha=0.6)
    ax.text(um * 1.0e4, 5e46 * 0.5, f"{name}", fontsize=7, color="0.5", ha="center", va="top")
ax.legend(frameon=False, fontsize=7, loc="lower left", ncol=2)

fig.tight_layout()
plt.savefig("plot_agn_torus_compare.png", dpi=150, bbox_inches="tight")
