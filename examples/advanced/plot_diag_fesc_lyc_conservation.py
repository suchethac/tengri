"""
Lyman continuum escape fraction conservation in Cue nebular model
==================================================================

When ionizing photons escape (f_esc > 0), fewer LyC photons ionize the ISM
within the galaxy, suppressing all nebular line emission proportionally:
L(Hα) ∝ (1 − f_esc) × Q_H, where Q_H is the intrinsic ionizing photon rate.
"""

import warnings

import jax
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

# Load bare-stellar SSP (Cue requirement)
ssp = tengri.load_ssp("fsps_prsc_miles_chabrier")

# Build young SF model with fixed SFH, metallicity, ionization
model = tengri.SEDModel.build(
    ssp,
    sfh={
        "type": "dpl",
        "all_params": tengri.FIXED,
        "alpha": 4.0,
        "beta": 2.0,
        "tau_gyr": 1.5,
        "log_total_mass": 10.0,
    },
    dust={"type": "two_component", "all_params": tengri.FIXED, "tau_diff": 0.0, "tau_bc": 0.0},
    neb={"type": "cue", "all_params": tengri.FIXED, "logU": -3.0, "logZ_gas": -1.0, "fesc": 0.0},
    redshift=tengri.Fixed(0.0),
)

baseline = dict(model.spec.sample(jax.random.PRNGKey(42)))

# Sweep f_esc from 0 to 0.95 in 10 steps
fesc_vals = np.linspace(0.0, 0.95, 10)
halpha_lums = []

for fesc in fesc_vals:
    params = {**baseline, "neb_fesc": float(fesc)}
    lines = model.predict(params).lines
    halpha_lums.append(float(lines.halpha))

halpha_lums = np.array(halpha_lums)

# Normalize to fesc=0 to check the (1 - fesc) scaling
halpha_norm = halpha_lums / halpha_lums[0]
theoretical = 1.0 - fesc_vals

# Compute max deviation from theoretical
max_deviation = np.max(np.abs(halpha_norm - theoretical))

# Plot
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10.5, 4.0))

# LEFT: L(Hα) vs f_esc with (1 - f_esc) overlay
ax_left.plot(fesc_vals, halpha_norm, "o-", lw=1.5, ms=5, label="tengri Cue")
ax_left.plot(fesc_vals, theoretical, "s--", lw=1.5, ms=5, label=r"$(1 - f_{\rm esc})$")
ax_left.set_xlabel(r"$f_{\rm esc}$ [Lyman continuum escape fraction]")
ax_left.set_ylabel(r"$L(H\alpha) / L(H\alpha)|_{f_{\rm esc}=0}$")
ax_left.legend(frameon=False, fontsize=9)
ax_left.set_xlim(-0.05, 1.0)
ax_left.grid(True, alpha=0.3)

# RIGHT: Ratio (tengri / theoretical) vs f_esc
ratio = halpha_norm / theoretical
ax_right.plot(fesc_vals, ratio, "o-", lw=1.5, ms=5, color="C2")
ax_right.axhline(1.0, color="k", linestyle="--", lw=1.0, alpha=0.5)
ax_right.set_xlabel(r"$f_{\rm esc}$")
ax_right.set_ylabel(r"tengri / Theory")
ax_right.set_xlim(-0.05, 1.0)
ax_right.grid(True, alpha=0.3)

fig.tight_layout()
plt.savefig("plot_diag_fesc_lyc_conservation.png", dpi=150, bbox_inches="tight")
print(f"Max deviation from y=x: {max_deviation:.2%}")
