"""
Birth-cloud attenuation age dependence: Charlot & Fall 2000
===========================================================

The Charlot & Fall 2000 two-component dust model splits attenuation into:
- ``τ_bc`` (birth-cloud): attenuates only young stellar ages (< 10 Myr)
- ``τ_diff`` (diffuse ISM): attenuates all stellar light

The age dependence shows up clearly when we build four single-age
populations and compare their SEDs. Young populations (1, 10 Myr) show
strong UV suppression from ``τ_bc``, while older populations (100, 1000 Myr)
are unaffected by birth-cloud dust — only diffuse attenuation remains.

Reference: Charlot & Fall 2000, ApJ, 539, 718.
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

C_AA_PER_S = 2.998e18

ssp = tengri.load_ssp()

# Fixed dust parameters: strong birth-cloud + modest diffuse
TAU_BC = 1.0
TAU_DIFF = 0.3

# Ages (in Myr) and their corresponding peak lookback times
# Use tsnorm with narrow width to approximate single-age populations
# peak_lbt_gyr ≈ log10(age_gyr) for young populations
ages_myr = np.array([1.0, 10.0, 100.0, 1000.0])
peak_lbt_values = np.array([6.0, 7.0, 8.0, 9.0])  # roughly log10(age_gyr), 1 Myr to 1 Gyr

colors = plt.cm.cool(np.linspace(0.2, 0.9, len(ages_myr)))

fig, ax = plt.subplots(figsize=(6.5, 4.2))

for i, (age_myr, peak_lbt_gyr) in enumerate(zip(ages_myr, peak_lbt_values)):
    # Single-age population via tsnorm (truncated star-formation normalized to peak)
    # peak_lbt_gyr sets the peak of the SFH; narrow width approximates single age
    model = tengri.SEDModel.build(
        ssp,
        sfh={
            "type": "tsnorm",
            "*": tengri.FIXED,
            "peak_lbt_gyr": float(peak_lbt_gyr),
            "width_gyr": 0.1,  # narrow width to approximate single age
            "log_total_mass": 10.0,  # normalized scale
        },
        dust={
            "type": "two_component",
            "*": tengri.FIXED,
            "tau_bc": TAU_BC,
            "tau_diff": TAU_DIFF,
            "slope": -0.7,
        },
        redshift=tengri.Fixed(0.0),  # z=0 for rest-frame SED
    )

    p = dict(model.spec.sample(jax.random.PRNGKey(0)))
    out = model.predict_rest_sed(p)

    wave = np.asarray(out.wavelength)
    nu_lnu = C_AA_PER_S / wave * np.asarray(out.sed)

    label = f"{age_myr:.0f} Myr" if age_myr < 100 else f"{age_myr / 1e3:.1f} Gyr"
    ax.loglog(wave, nu_lnu, color=colors[i], lw=2.0, label=label)

ax.set_xlim(1000, 30000)
ax.set_xlabel(r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]")
ax.set_ylabel(r"$\nu L_\nu$  [erg s$^{-1}$]")

ax.legend(fontsize=9, frameon=True, loc="upper right")

# Annotation box explaining the attenuation model
textstr = (
    r"Charlot \& Fall 2000: "
    + f"$\\tau_{{\\rm bc}}={TAU_BC:.1f}$, "
    + f"$\\tau_{{\\rm diff}}={TAU_DIFF:.1f}$"
)
ax.text(
    0.05,
    0.95,
    textstr,
    transform=ax.transAxes,
    fontsize=9,
    verticalalignment="top",
    bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="0.7", lw=0.5),
)

fig.tight_layout()
plt.savefig("plot_birth_cloud_vs_diffuse_age.png", dpi=150, bbox_inches="tight")
