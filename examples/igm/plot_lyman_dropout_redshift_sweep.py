"""
Lyman Dropout Redshift Sweep: IGM Absorption Evolution

As redshift increases, the Lyman edge (rest-frame 912 Å) shifts to longer
observed wavelengths. This example shows how the Lyman dropout sweeps across
the optical and near-infrared bands at :math:`z = 3, 4, 5, 6, 7`, progressively
absorbing shorter-wavelength photometry and dropping it out of optical surveys.

A young star-forming galaxy (age 10 Myr, modest dust :math:`\\tau_V = 0.1`) is
modeled once in the rest frame with a brief starburst, then IGM transmission
(Inoue et al. 2014) is applied to the observed-frame SED. The sharp absorption
feature — the Lyman discontinuity at observed 912(1+z) Å — marches redward and
deepens with z, enabling photometric dropout selection at high redshift.

References:
- Inoue et al. 2014, MNRAS, 442, 1805 — IGM model
- Madau 1995, ApJ, 441, 18 — Lyman-edge physics
- Steidel et al. 1996, AJ, 112, 352 — Dropout-selection origins
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import jax
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

# Redshifts to show
REDSHIFTS = [3.0, 4.0, 5.0, 6.0, 7.0]
COLORS = plt.cm.viridis(np.linspace(0.2, 0.95, len(REDSHIFTS)))

# Conversion constant: c in Angstrom/s
C_AA_PER_S = 2.998e18

# Galaxy model: double power law SFH tuned to young, bursty star formation
# with modest dust attenuation
SFH_CONFIG = {
    "type": "dpl",
    "*": tengri.FIXED,
    "tau_gyr": 0.01,  # Very short timescale (10 Myr) — young burst
    "log_total_mass": 10.0,  # Moderate star formation rate
    "alpha": 2.0,  # Rising phase exponent
    "beta": 2.5,  # Declining phase exponent
}

DUST_CONFIG = {
    "type": "two_component",
    "*": tengri.FIXED,
    "tau_diff": 0.1,  # Diffuse dust
    "tau_bc": 0.0,  # Minimal birth cloud dust
}

# Build and plot
fig, ax = plt.subplots(figsize=(7.4, 4.8))

for z, color in zip(REDSHIFTS, COLORS):
    # Build model with IGM block
    model = tengri.SEDModel.build(
        tengri.load_ssp(),
        sfh=SFH_CONFIG,
        dust=DUST_CONFIG,
        igm={"type": "inoue14"},
        redshift=tengri.Fixed(z),
    )

    # Sample parameters and predict
    p = dict(model.spec.sample(jax.random.PRNGKey(0)))
    out = model.predict_rest_sed(p)

    # Convert to observed frame: wave_obs = wave_rest * (1 + z)
    wave_rest = np.asarray(out.wavelength)
    wave_obs = wave_rest * (1.0 + z)

    # Convert SED (erg/s/Hz) to nu*F_nu for log-log display
    nu = C_AA_PER_S / wave_obs
    nu_f_nu = nu * np.asarray(out.sed)

    ax.loglog(wave_obs, nu_f_nu, color=color, lw=1.5, label=f"$z={z:g}$")

# Mark Lyman-edge positions at each redshift
for z, color in zip(REDSHIFTS, COLORS):
    lyman_edge_obs = 912.0 * (1.0 + z)
    ax.axvline(lyman_edge_obs, color=color, lw=0.5, ls="--", alpha=0.5)

# Annotations and formatting
ax.set_xlim(500, 2e4)
ax.set_ylim(1e29, 5e32)
ax.set_xlabel(r"Observed wavelength $\lambda$ [$\mathrm{\AA}$]", fontsize=12)
ax.set_ylabel(r"$\nu L_\nu$ [erg/s]", fontsize=12)
ax.legend(frameon=False, fontsize=10, loc="upper right")
ax.text(
    912 * 7.5,
    1.5e32,
    r"← Lyman edge (912 Å$\times(1+z)$)",
    fontsize=9,
    color="0.4",
)

fig.tight_layout()
plt.savefig("plot_lyman_dropout_redshift_sweep.png", dpi=150, bbox_inches="tight")
