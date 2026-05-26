"""
Optical line-ratio diagnostics along the metallicity gradient
=============================================================

Three classical strong-line metallicity diagnostics computed as a function
of gas-phase metallicity (``logZ_gas``). The plot spans 12 + log(O/H) from
~7 to ~9 and illustrates key observational features: the saturation of
[O III]/H-beta at high metallicity (Kewley & Dopita 2002), the monotonic
but small dynamic range of [N II]/H-alpha (Marino et al. 2013), and the
famous double-valued R23 ratio which peaks near 12 + log(O/H) ≈ 8.3
(Pagel et al. 1979).

why observers cannot uniquely invert a single line
ratio to metallicity without breaking the R23 degeneracy or adopting a
secondary diagnostic.
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

# Load bare-stellar SSP (required by Cue nebular backend)
SSP = tengri.load_ssp("fsps_prsc_miles_chabrier")

# Build model with free logZ_gas parameter spanning the diagnostic range
model = tengri.SEDModel.build(
    SSP,
    sfh={
        "type": "dpl",
        "*": tengri.FIXED,
        "tau_gyr": 0.05,
        "log_total_mass": 10.0,
        "alpha": 4.0,
        "beta": 2.0,
    },
    dust={"type": "two_component", "*": tengri.FIXED, "tau_diff": 0.0, "tau_bc": 0.0},
    neb={"type": "cue", "*": tengri.FIXED, "logZ_gas": tengri.Uniform(-2.0, 0.5)},
    redshift=tengri.Fixed(0.0),
)

# Sample baseline parameters and then sweep logZ_gas
baseline = dict(model.spec.sample(jax.random.PRNGKey(0)))

z_grid = np.linspace(-1.5, 0.4, 24)
log_o3hb = np.empty_like(z_grid)
log_n2ha = np.empty_like(z_grid)
log_r23 = np.empty_like(z_grid)

for i, z in enumerate(z_grid):
    p = {**baseline, "neb_logZ_gas": jnp.float64(z)}
    L = model.predict_emission_lines(p)

    log_o3hb[i] = float(np.log10(L.oiii_5007 / L.hbeta))
    log_n2ha[i] = float(np.log10(L.nii_6584 / L.halpha))

    # R23 = ([O II] + [O III]) / H-beta
    # Note: oii is the doublet [OII] 3726+3729 summed
    r23 = float((L.oii + L.oiii_4959 + L.oiii_5007) / L.hbeta)
    log_r23[i] = float(np.log10(r23))

# Convert log(Z/Zsun) to 12 + log(O/H); assume 8.69 at solar metallicity
twelve_oh = 8.69 + z_grid

# Single-panel plot showing all three diagnostics
fig, ax = plt.subplots(figsize=(8.0, 5.0))

ax.plot(twelve_oh, log_o3hb, color="C0", lw=2.0, label=r"$[\mathrm{O\,III}]/\mathrm{H}\beta$")
ax.plot(twelve_oh, log_n2ha, color="C3", lw=2.0, label=r"$[\mathrm{N\,II}]/\mathrm{H}\alpha$")
ax.plot(
    twelve_oh,
    log_r23,
    color="C2",
    lw=2.0,
    label=r"$R_{23}$ ([$\mathrm{O\,II}$]+[$\mathrm{O\,III}$])/H$\beta$)",
)

ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
ax.set_xlabel(r"$12 + \log(\mathrm{O/H})$", fontsize=11)
ax.set_ylabel(r"$\log\,(\mathrm{line\,ratio})$", fontsize=11)
ax.grid(True, alpha=0.3, linestyle=":")

plt.savefig("plot_line_ratios_metallicity_evolution.png", dpi=150, bbox_inches="tight")
