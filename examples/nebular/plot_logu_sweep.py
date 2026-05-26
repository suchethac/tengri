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
    neb={"type": "cue", "*": tengri.FIXED, "logU": tengri.Uniform(-4.0, -1.5)},
    redshift=tengri.Fixed(0.05),
)
baseline = dict(model.spec.sample(jax.random.PRNGKey(0)))

logu_values = np.linspace(-4.0, -1.5, 7)
norm = mpl.colors.Normalize(vmin=logu_values.min(), vmax=logu_values.max())
cmap = plt.get_cmap("plasma")

fig, ax = plt.subplots(figsize=(7.0, 4.2))
# Plot line luminosities directly — emission lines respond to logU strongly
# but they are sub-percent perturbations on the dominant stellar continuum,
# so subtracting the continuum (via predict_emission_lines) is the only way
# to see logU's effect cleanly.
oiii_lum, halpha_lum, nii_lum = [], [], []
for logu in logu_values:
    params = {**baseline, "neb_logU": jnp.float64(logu)}
    lines = model.predict_emission_lines(params)
    oiii_lum.append(float(lines.oiii_5007))
    halpha_lum.append(float(lines.halpha))
    nii_lum.append(float(lines.nii_6584))

oiii_lum = np.asarray(oiii_lum)
halpha_lum = np.asarray(halpha_lum)
nii_lum = np.asarray(nii_lum)

ax.semilogy(logu_values, oiii_lum / halpha_lum, "o-", lw=2.0,
            color="#2a9d8f", label=r"[OIII]$\lambda$5007 / H$\alpha$")
ax.semilogy(logu_values, nii_lum / halpha_lum, "s-", lw=2.0,
            color="#e76f51", label=r"[NII]$\lambda$6584 / H$\alpha$")

ax.set_xlabel(r"$\log U$")
ax.set_ylabel(r"Line ratio")
ax.legend(loc="best", frameon=False, fontsize=10)
ax.set_title(r"logU controls optical emission-line diagnostics", fontsize=12)

fig.tight_layout()
plt.savefig("plot_logu_sweep.png", dpi=150, bbox_inches="tight")
