"""
X-ray SED response to host SFR (HMXB scaling)
==============================================

Tengri's ``simple`` X-ray model attaches a population of high- and
low-mass X-ray binaries to the host galaxy, with HMXB luminosity
scaling linearly with the current SFR (Mineo+2012 relation) and LMXB
with the stellar mass (Lehmer+2010). We sweep host SFR and watch the
0.5-100 keV X-ray continuum rise.

(The AGN-driven X-ray corona contribution as a function of
``agn_alpha_ox``, ``agn_log_lbol`` is deferred until the composable
AGN block routing under ``SEDModel.build`` is fixed — issue #258.)
"""

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

ssp = tengri.load_ssp()
model = tengri.SEDModel.build(
    ssp,
    sfh={"type": "const", "*": tengri.FIXED,
         "log_sfr": tengri.Uniform(-2.0, 3.0),
         "start_gyr": 13.0, "end_gyr": 0.0},
    dust={"type": "two_component", "*": tengri.FIXED,
          "tau_diff": 0.3, "tau_bc": 0.5},
    xray={"type": "simple", "*": tengri.FIXED},
    redshift=tengri.Fixed(0.05),
)
baseline = dict(model.spec.sample(jax.random.PRNGKey(0)))

log_sfr_grid = np.linspace(-1.0, 2.5, 7)
norm = mpl.colors.Normalize(vmin=log_sfr_grid.min(), vmax=log_sfr_grid.max())
cmap = plt.get_cmap("viridis")

fig, ax = plt.subplots(figsize=(7.0, 4.6))
for log_sfr in log_sfr_grid:
    out = model.predict_rest_sed({**baseline,
                                  "sfh_const_log_sfr": jnp.float64(log_sfr)})
    wave = np.asarray(out.wavelength)
    nu_l_nu = C_AA_PER_S / wave * np.asarray(out.sed)
    ax.loglog(wave, nu_l_nu, color=cmap(norm(log_sfr)), lw=1.4)

ax.set_xlim(0.5, 1e3)  # X-ray: 100 keV → 1 Å, 100 eV → 1000 Å
ax.set_ylim(1e36, 1e44)
ax.set_xlabel(r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]")
ax.set_ylabel(r"$\nu L_\nu$  [erg s$^{-1}$]")

# Mark common X-ray energies
for kev, name in [(0.5, "0.5 keV"), (2.0, "2 keV"), (10.0, "10 keV")]:
    # lambda [Å] = 12398.4 / E[eV]
    lam = 12398.4 / (kev * 1000.0)
    ax.axvline(lam, color="0.7", lw=0.5, ls=":")
    ax.text(lam * 1.05, 1e43, name, fontsize=7, color="0.45", rotation=90,
            va="top")

cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.01)
cbar.set_label(r"$\log\,\mathrm{SFR}\,/\,M_\odot\,\mathrm{yr}^{-1}$")

fig.tight_layout()
fig.savefig("plot_xray_vs_sfr.png", dpi=150, bbox_inches="tight")
