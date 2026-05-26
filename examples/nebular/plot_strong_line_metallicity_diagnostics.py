"""
Strong-line gas-phase metallicity diagnostics
==============================================

Four widely-used optical strong-line metallicity diagnostics
evaluated across the Cue ``logZ_gas`` prior. Each one carries a
different systematic — Pettini & Pagel 2004 O3N2 saturates at high
Z, the R23 ratio is double-valued, N2 (Marino+2013) is monotonic
but small dynamic range, and the [Ne III]/[O II] diagnostic is
weakly Z-dependent.

This is the diagnostic family every observer chooses between when
converting line ratios to a 12 + log(O/H) on a sample.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

SSP = tengri.load_ssp("fsps_prsc_miles_chabrier")
model = tengri.SEDModel.build(
    SSP,
    sfh={"type": "dpl", "*": tengri.FIXED, "tau_gyr": 0.05,
         "log_total_mass": 10.0, "alpha": 4.0, "beta": 2.0},
    dust={"type": "two_component", "*": tengri.FIXED,
          "tau_diff": 0.0, "tau_bc": 0.0},
    neb={"type": "cue", "*": tengri.FIXED,
         "logZ_gas": tengri.Uniform(-2.0, 0.5)},
    redshift=tengri.Fixed(0.0),
)
baseline = dict(model.spec.sample(jax.random.PRNGKey(0)))

z_grid = np.linspace(-1.5, 0.4, 18)
o3, n2_ha, hb, oii, ne3, sii = [np.empty_like(z_grid) for _ in range(6)]

for i, z in enumerate(z_grid):
    p = {**baseline, "neb_logZ_gas": jnp.float64(z)}
    L = model.predict_emission_lines(p)
    o3[i]    = float(L.oiii_5007)
    n2_ha[i] = float(L.nii_6584 / L.halpha)
    hb[i]    = float(L.hbeta)
    oii[i]   = float(L.oii)
    ne3[i]   = float(L.civ_1549)  # use CIV as alt high-ion probe
    sii[i]   = float(L.sii_6717 + L.sii_6731)

# Convert log Z/Zsun to 12 + log(O/H), assuming 8.69 at solar
twelve_oh = 8.69 + z_grid

fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.4), sharex=True,
                          gridspec_kw={"hspace": 0.08, "wspace": 0.22})
ax_o3hb, ax_n2, ax_o32, ax_n2o2 = axes.ravel()

ax_o3hb.plot(twelve_oh, np.log10(o3 / hb), color="C0", lw=1.6)
ax_o3hb.set_ylabel(r"$\log\,[\mathrm{O\,III}]\,5007 / \mathrm{H}\beta$")
ax_o3hb.text(0.05, 0.93, "Kewley & Dopita 2002", transform=ax_o3hb.transAxes,
             fontsize=8, color="0.4")

ax_n2.plot(twelve_oh, np.log10(n2_ha), color="C3", lw=1.6)
ax_n2.set_ylabel(r"$\log\,[\mathrm{N\,II}]\,6584 / \mathrm{H}\alpha$")
ax_n2.text(0.05, 0.93, "Pettini & Pagel 2004 N2", transform=ax_n2.transAxes,
           fontsize=8, color="0.4")

ax_o32.plot(twelve_oh, np.log10(o3 / oii), color="C2", lw=1.6)
ax_o32.set_ylabel(r"$\log\,O_{32}$ ($[\mathrm{O\,III}]/[\mathrm{O\,II}]$)")
ax_o32.set_xlabel(r"$12 + \log(\mathrm{O/H})$")

ax_n2o2.plot(twelve_oh, np.log10(np.maximum(sii / oii, 1e-3)),
             color="C1", lw=1.6)
ax_n2o2.set_ylabel(r"$\log\,[\mathrm{S\,II}]/[\mathrm{O\,II}]$")
ax_n2o2.set_xlabel(r"$12 + \log(\mathrm{O/H})$")
ax_n2o2.text(0.05, 0.93, "low-ion proxy", transform=ax_n2o2.transAxes,
             fontsize=8, color="0.4")

plt.savefig("plot_strong_line_metallicity_diagnostics.png", dpi=150,
            bbox_inches="tight")
