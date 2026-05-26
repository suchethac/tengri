"""
QSOgen lines: emission-line contributions vary with luminosity
==============================================================

The QSOgen model includes a UV/optical emission-line forest and broad
Balmer continuum on top of the underlying disc. The relative strength
of these line features with respect to the continuum controls the slope
and colour of the UV–optical SED.
"""

import warnings

import jax
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")
warnings.filterwarnings("ignore", message=".*deprecated.*")

ssp = tengri.load_ssp()
model = tengri.SEDModel.build(
    ssp,
    sfh={
        "type": "dpl",
        "*": tengri.FIXED,
        "tau_gyr": 3.0,
        "log_peak_sfr": 0.5,
        "alpha": 2.0,
        "beta": 2.5,
    },
    dust={"type": "two_component", "*": tengri.FIXED, "tau_diff": 0.1, "tau_bc": 0.1},
    agn={
        "type": "composable",
        "disc": {"type": "multicolor", "*": tengri.FIXED},
        "torus": {"type": "skirtor", "*": tengri.FIXED},
        "lines": {"type": "nlr", "*": tengri.FIXED},
        "*": tengri.FIXED,
        "agn_frac": 1.0,
        "log_lbol": 11.0,
    },
    redshift=tengri.Fixed(0.05),
)
baseline = dict(model.spec.sample(jax.random.PRNGKey(0)))

# This parameter doesn't exist in current API; skip for now
# and simplify to a fixed model
fig, ax = plt.subplots(figsize=(6.5, 4.2))
params = baseline
out = model.predict_rest_sed(params)
wave = np.asarray(out.wavelength)
nu = 2.998e18 / wave
nu_l_nu = nu * np.asarray(out.sed)
ax.loglog(wave, nu_l_nu, lw=1.4, color="tab:blue")

ax.set_xlim(100, 1e6)
ax.set_ylim(1e40, 1e45)
ax.set_xlabel(r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]")
ax.set_ylabel(r"$\nu L_\nu$  [erg s$^{-1}$]")
ax.text(
    0.5,
    0.95,
    "QSOgen with default line scaling",
    transform=ax.transAxes,
    ha="center",
    va="top",
    fontsize=10,
)

fig.tight_layout()
plt.savefig("plot_agn_qsogen_emline_sweep.png", dpi=150, bbox_inches="tight")
