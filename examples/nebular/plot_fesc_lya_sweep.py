"""
Lyα escape fraction controls Lyman-alpha strength
=================================================

The Lyα-specific escape fraction ``f_esc_lya`` sets what fraction of Lyα
photons can escape the ISM without scattering. Higher ``f_esc_lya`` suppresses
the Lyα emission line while leaving other nebular lines unchanged.
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
warnings.filterwarnings("ignore", message=".*deprecated.*")

ssp = tengri.load_ssp("fsps_prsc_miles_chabrier")
model = tengri.SEDModel.build(
    ssp,
    sfh={
        "type": "dpl",
        "*": tengri.FIXED,
        "alpha": 3.0,
        "beta": 2.0,
        "tau_gyr": 0.3,
        "log_peak_sfr": 1.5,
    },
    dust={"type": "two_component", "*": tengri.FIXED, "tau_diff": 0.05, "tau_bc": 0.1},
    neb={"type": "cue", "*": tengri.FIXED, "neb_fesc_lya": tengri.Uniform(0.0, 1.0)},
    redshift=tengri.Fixed(0.05),
)
baseline = dict(model.spec.sample(jax.random.PRNGKey(0)))

fesc_lya_values = np.linspace(0.0, 1.0, 7)
norm = mpl.colors.Normalize(vmin=fesc_lya_values.min(), vmax=fesc_lya_values.max())
cmap = plt.get_cmap("viridis")

fig, ax = plt.subplots(figsize=(6.5, 4.2))
ymax = 0.0
ymin = np.inf
for fesc_lya in fesc_lya_values:
    params = {**baseline, "neb_fesc_lya": jnp.float64(fesc_lya)}
    out = model.predict_rest_sed(params)
    wave = np.asarray(out.wavelength)
    nu = 2.998e18 / wave
    nu_l_nu = nu * np.asarray(out.sed)
    vis = (wave > 1100) & (wave < 1300)
    ymax = max(ymax, float(np.max(nu_l_nu[vis])))
    pos = nu_l_nu[vis][nu_l_nu[vis] > 0]
    if pos.size:
        ymin = min(ymin, float(np.min(pos)))
    ax.semilogy(wave, nu_l_nu, color=cmap(norm(fesc_lya)), lw=1.4)

ax.set_xlim(1100, 1300)
ax.set_ylim(0.3 * ymin, 3.0 * ymax)
ax.set_xlabel(r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]")
ax.set_ylabel(r"$\nu L_\nu$  [erg s$^{-1}$]")
ax.axvline(1215.67, color="0.55", lw=0.5, ls=":")
ax.text(1215.67, ymax * 0.4, r"Ly$\alpha$",
        color="0.4", fontsize=8, rotation=90, va="center", ha="right")

cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.01)
cbar.set_label(r"$f_{\mathrm{esc,Ly}\alpha}$")

fig.tight_layout()
fig.savefig("plot_fesc_lya_sweep.png", dpi=150, bbox_inches="tight")
