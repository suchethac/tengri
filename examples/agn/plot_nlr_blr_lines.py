"""
Narrow vs broad line region: a velocity-width contrast in two windows
======================================================================

Identical AGN configuration (multicolor disc + SKIRTOR torus at log
L_bol = 12.5), one with the narrow-line region (FWHM ~ a few hundred
km/s, characteristic Type-2 spectrum) and the other with the broad-line
region (FWHM ~ thousands of km/s, Type-1). Side-by-side zooms on the
UV (Ly-alpha, C IV) and the optical (Hbeta, [O III], Hα) make the
velocity-width contrast unmistakable while controlling for continuum.

Companion to ``plot_agn_lines_compare.py``, which sweeps line backbones
across the full UV-optical range; this script focuses on the kinematic
fingerprint at fixed library.

Reference: Osterbrock & Ferland 2006, *Astrophysics of Gaseous Nebulae
and Active Galactic Nuclei* (line classification by FWHM).
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
SFH = {"type": "const", "all_params": tengri.FIXED, "log_total_mass": -10.0}
DUST = {"type": "two_component", "all_params": tengri.FIXED, "tau_diff": 0.0, "tau_bc": 0.0}

ssp = tengri.load_ssp()


# Mapping of deprecated lines.type to new nlr/blr types
_LINES_EXPANSION = {
    "nlr": {"nlr_type": "analytic", "blr_type": "none"},
    "blr": {"nlr_type": "none", "blr_type": "analytic"},
}


def _build(lines_type):
    mapping = _LINES_EXPANSION.get(lines_type)
    if not mapping:
        raise ValueError(f"Unknown lines type: {lines_type}")

    model = tengri.SEDModel.build(
        ssp,
        sfh=SFH,
        dust=DUST,
        agn={
            "all_params": tengri.FIXED,
            "log_lbol": 12.5,
            "lum_ratio": 1.0,
            "disc": {"type": "multicolor", "all_params": tengri.FIXED},
            "torus": {"type": "skirtor", "all_params": tengri.FIXED},
            "nlr": {"type": mapping["nlr_type"], "all_params": tengri.FIXED},
            "blr": {"type": mapping["blr_type"], "all_params": tengri.FIXED},
        },
        redshift=tengri.Fixed(0.0),
    )
    p = dict(model.spec.sample(jax.random.PRNGKey(0)))
    out = model.predict(p)
    wave = np.asarray(model.wavelengths)
    return wave, C_AA_PER_S / wave * np.asarray(out.rest_sed())


wave_nlr, nl_nlr = _build("nlr")
wave_blr, nl_blr = _build("blr")

UV_LINES = [(1216, r"Ly$\alpha$"), (1549, "C IV"), (1909, "C III]"), (2798, "Mg II")]
OPT_LINES = [(4861, r"H$\beta$"), (5007, "[O III]"), (6563, r"H$\alpha$"), (6731, "[S II]")]

fig, (ax_uv, ax_opt) = plt.subplots(1, 2, figsize=(11.0, 4.6))
for ax, xlim, marks in [(ax_uv, (1000, 3000), UV_LINES), (ax_opt, (4500, 7000), OPT_LINES)]:
    ax.semilogy(wave_nlr, nl_nlr, color="#1f77b4", lw=1.4, label="NLR (Type-2)")
    ax.semilogy(wave_blr, nl_blr, color="#d62728", lw=1.4, label="BLR (Type-1)")
    ax.set_xlim(*xlim)
    ax.set_ylim(1.0e44, 5.0e46)
    ax.set_xlabel(r"Rest-frame $\lambda$ [$\mathrm{\AA}$]")
    for wl, name in marks:
        ax.axvline(wl, color="0.7", ls=":", lw=0.6)
        ax.text(wl, 2.0e44, name, fontsize=8, color="0.4", rotation=90, va="bottom")

ax_uv.set_ylabel(r"$\nu L_\nu$  [erg s$^{-1}$]")
ax_opt.legend(frameon=False, fontsize=9, loc="upper right")

fig.tight_layout()
plt.savefig("plot_nlr_blr_lines.png", dpi=150, bbox_inches="tight")
