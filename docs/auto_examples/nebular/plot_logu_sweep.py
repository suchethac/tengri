"""
Ionization parameter controls optical line strength
===================================================

Higher ionisation parameter ``log U`` drives stronger [OIII] and [NII]
emission, steering the galaxy toward the Seyfert region on the BPT diagram.
We vary ``log U`` across the typical range for star-forming galaxies.
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
        "alpha": 1.0,
        "beta": 2.5,
        "tau_gyr": 0.3,
        "log_peak_sfr": 1.5,
    },
    dust={"type": "two_component", "*": tengri.FIXED, "tau_diff": 0.0, "tau_bc": 0.0},
    neb={"type": "cue", "*": tengri.FIXED, "neb_logU": tengri.Uniform(-4.0, -1.5)},
    redshift=tengri.Fixed(0.05),
)
baseline = dict(model.spec.sample(jax.random.PRNGKey(0)))

logu_values = np.linspace(-4.0, -1.5, 7)
norm = mpl.colors.Normalize(vmin=logu_values.min(), vmax=logu_values.max())
cmap = plt.get_cmap("viridis")

fig, ax = plt.subplots(figsize=(6.5, 4.2))
for logu in logu_values:
    params = {**baseline, "neb_logU": jnp.float64(logu)}
    out = model.predict_rest_sed(params)
    wave = np.asarray(out.wavelength)
    nu = 2.998e18 / wave
    nu_l_nu = nu * np.asarray(out.sed)
    ax.semilogy(wave, nu_l_nu, color=cmap(norm(logu)), lw=1.4)

ax.set_xlim(4000, 7500)
ax.set_xlabel(r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]")
ax.set_ylabel(r"$\nu L_\nu$  [erg s$^{-1}$]")

cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.01)
cbar.set_label(r"$\log U$")

fig.tight_layout()
fig.savefig("plot_logu_sweep.png", dpi=150, bbox_inches="tight")
