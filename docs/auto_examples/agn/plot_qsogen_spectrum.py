"""
QSOgen empirical quasar SED across four decades of bolometric luminosity
=========================================================================

The Temple, Hewett & Banerji (2021) QSOgen empirical template, used as
the ``agn.disc.type="qsogen"`` selector. We sweep log L_bol from 10.0
to 13.5 (in L_sun units) at fixed redshift to show that the template's
spectral *shape* is approximately self-similar across the quasar
luminosity function — the only knob that moves features (the
Baldwin-effect drop in C IV/Ly-alpha equivalent width) is the
bolometric normalisation.

Reference: Temple, Hewett & Banerji 2021, MNRAS, 508, 737.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

C_AA_PER_S = 2.998e18
SFH = {"type": "const", "*": tengri.FIXED, "log_total_mass": -10.0}
DUST = {"type": "two_component", "*": tengri.FIXED, "tau_diff": 0.0, "tau_bc": 0.0}

ssp = tengri.load_ssp()
model = tengri.SEDModel.build(
    ssp,
    sfh=SFH,
    dust=DUST,
    agn={
        "*": tengri.FIXED,
        "log_lbol": tengri.Uniform(10.0, 13.5),
        "frac": 1.0,
        "disc": {"type": "qsogen", "*": tengri.FIXED},
    },
    redshift=tengri.Fixed(0.0),
)
baseline = dict(model.spec.sample(jax.random.PRNGKey(0)))

log_lbol_values = np.linspace(10.0, 13.5, 8)
norm = mpl.colors.Normalize(vmin=log_lbol_values.min(), vmax=log_lbol_values.max())
cmap = plt.get_cmap("plasma")

fig, ax = plt.subplots(figsize=(7.0, 4.6))
for lbol in log_lbol_values:
    params = {**baseline, "agn_log_lbol": jnp.float64(lbol)}
    out = model.predict_rest_sed(params)
    wave_um = np.asarray(out.wavelength) * 1.0e-4
    nu_l_nu = C_AA_PER_S / np.asarray(out.wavelength) * np.asarray(out.sed)
    ax.loglog(wave_um, nu_l_nu, color=cmap(norm(lbol)), lw=1.4)

ax.set(
    xlim=(0.01, 10.0),
    ylim=(1.0e41, 1.0e47),
    xlabel=r"Rest-frame wavelength [$\mu$m]",
    ylabel=r"$\nu L_\nu$  [erg s$^{-1}$]",
)
cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.01)
cbar.set_label(r"$\log_{10}(L_{\rm bol} / L_\odot)$")

fig.tight_layout()
plt.savefig("plot_qsogen_spectrum.png", dpi=150, bbox_inches="tight")
