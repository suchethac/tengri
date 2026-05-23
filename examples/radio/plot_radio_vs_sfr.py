"""
Radio SED response to host SFR
================================

The Condon 1992 radio model couples the SF-driven synchrotron + free-free
continuum to the host SFR via the FIR-radio correlation. We sweep the
host SFR from 0.1 to 100 M_sun / yr and watch the GHz luminosity rise
linearly with SFR.

(Originally this script would have demonstrated the AGN-driven jet
contribution as ``agn_log_lbol`` varies — but the composable AGN block
under ``SEDModel.build`` does not currently activate the AGN SED, see
issue #258. Until that is fixed, the AGN + radio combined plot is
deferred.)
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
          "tau_diff": 0.3, "tau_bc": 0.5,
          "emission": {"type": "dale2014", "*": tengri.FIXED}},
    radio={"type": "condon92", "*": tengri.FIXED},
    redshift=tengri.Fixed(0.05),
)
baseline = dict(model.spec.sample(jax.random.PRNGKey(0)))

log_sfr_grid = np.linspace(-1.0, 2.0, 7)  # 0.1 to 100 Msun/yr
norm = mpl.colors.Normalize(vmin=log_sfr_grid.min(), vmax=log_sfr_grid.max())
cmap = plt.get_cmap("viridis")

fig, ax = plt.subplots(figsize=(7.0, 4.6))
for log_sfr in log_sfr_grid:
    out = model.predict_rest_sed({**baseline,
                                  "sfh_const_log_sfr": jnp.float64(log_sfr)})
    wave = np.asarray(out.wavelength)
    nu_l_nu = C_AA_PER_S / wave * np.asarray(out.sed)
    ax.loglog(wave, nu_l_nu, color=cmap(norm(log_sfr)), lw=1.4)

ax.set_xlim(1e6, 3e9)  # 100 μm → 30 m
ax.set_ylim(1e36, 1e44)
ax.set_xlabel(r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]")
ax.set_ylabel(r"$\nu L_\nu$  [erg s$^{-1}$]")
for nu_ghz, name in [(1.4, "1.4 GHz"), (3.0, "3 GHz"), (0.150, "150 MHz")]:
    lam = 2.998e10 / (nu_ghz * 1e9) * 1e10
    ax.axvline(lam, color="0.7", lw=0.5, ls=":")
    ax.text(lam * 1.02, 5e43, name, fontsize=7, color="0.45")

cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.01)
cbar.set_label(r"$\log\,\mathrm{SFR}\,/\,M_\odot\,\mathrm{yr}^{-1}$")

fig.tight_layout()
fig.savefig("plot_radio_vs_sfr.png", dpi=150, bbox_inches="tight")
