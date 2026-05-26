"""
Stellar-population age moves a galaxy on the BPT diagram
=========================================================

Young massive stars produce harder ionising continua and drive the
nebular emission toward higher [O III]/Hbeta. We sweep the SFH
timescale ``tau_gyr`` from 0.1 to 2 Gyr on a single dual power-law
model and plot the resulting line ratios against the Kewley+2001 /
Kauffmann+2003 demarcation curves. The locus migrates from the
star-forming wing into the composite region as the population ages —
SFH timescale is the upstream knob behind the BPT ionisation sequence.

Distinct from ``plot_bpt_cue_grid.py`` (log U × log Z_gas grid at
fixed age) and ``plot_cue_logu_line_ratios.py`` (1-D log U sweep).

Reference: Kewley et al. 2001, ApJ, 556, 121;
Kauffmann et al. 2003, MNRAS, 346, 1055.
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
warnings.filterwarnings("ignore", message=".*deprecated.*")

log_nii_ha_grid = np.linspace(-1.5, 0.3, 200)
log_oiii_hb_kewley = 0.61 / (log_nii_ha_grid - 0.47) + 1.19
log_oiii_hb_kauff = 0.61 / (log_nii_ha_grid - 0.05) + 1.3

ssp = tengri.load_ssp("fsps_prsc_miles_chabrier")
model = tengri.SEDModel.build(
    ssp,
    sfh={
        "type": "dpl",
        "*": tengri.FIXED,
        "alpha": 1.0,
        "beta": 2.5,
        "tau_gyr": tengri.Uniform(0.1, 2.0),
        "log_total_mass": 10.0,
    },
    dust={"type": "two_component", "*": tengri.FIXED, "tau_diff": 0.0, "tau_bc": 0.0},
    neb={"type": "cue", "*": tengri.FIXED},
    redshift=tengri.Fixed(0.0),
)
baseline = dict(model.spec.sample(jax.random.PRNGKey(0)))

fig, ax = plt.subplots(figsize=(8, 6.5))

mask_k = log_nii_ha_grid < 0.47
ax.plot(
    log_nii_ha_grid[mask_k],
    log_oiii_hb_kewley[mask_k],
    "k-",
    lw=1.5,
    label="Kewley+2001",
)
mask_kauff = log_nii_ha_grid < 0.05
ax.plot(
    log_nii_ha_grid[mask_kauff],
    log_oiii_hb_kauff[mask_kauff],
    "k--",
    lw=1.2,
    label="Kauffmann+2003",
)

ages = np.linspace(0.1, 2.0, 7)
colors = plt.cm.viridis(np.linspace(0, 1, len(ages)))

for age, color in zip(ages, colors):
    params = {**baseline, "sfh_dpl_tau_gyr": jnp.float64(age)}
    lines = model.predict_emission_lines(params)
    ha = float(lines.halpha)
    hb = float(lines.hbeta)
    nii = float(lines.nii_6584)
    oiii = float(lines.oiii_5007)
    if ha > 0 and hb > 0 and oiii > 0 and nii > 0:
        log_n2_ha = np.log10(nii / ha)
        log_o3_hb = np.log10(oiii / hb)
        ax.scatter(log_n2_ha, log_o3_hb, s=80, c=[color], edgecolors="k", lw=0.5, zorder=5)

ax.text(-1.3, -0.5, "SF", fontsize=10, color="#1f77b4", ha="center")
ax.text(0.1, 0.8, "Composite", fontsize=10, color="#ff7f0e", ha="center")
ax.text(0.35, 1.2, "Seyfert", fontsize=10, color="#d62728", ha="center")

ax.set_xlabel(r"$\log$ [NII] / H$\alpha$")
ax.set_ylabel(r"$\log$ [OIII] / H$\beta$")
ax.set_xlim(-1.6, 0.7)
ax.set_ylim(-1.2, 1.5)
ax.legend(fontsize=10, frameon=False, loc="lower right")

sm = plt.cm.ScalarMappable(
    cmap=plt.cm.viridis, norm=plt.Normalize(vmin=ages.min(), vmax=ages.max())
)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, label="Age [Gyr]")

fig.tight_layout()
plt.savefig("plot_neb_bpt_logu_grid.png", dpi=150, bbox_inches="tight")
