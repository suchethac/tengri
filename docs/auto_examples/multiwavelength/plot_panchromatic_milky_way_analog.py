"""
Panchromatic SED: Milky Way Analog

A nearby Milky Way-mass galaxy (M*~5×10^10 M☉, SFR~2 M☉/yr) across
the full electromagnetic spectrum from X-ray (10 Å) to radio (10^9 Å).

Star formation history follows a double power-law with recent sustained
activity. Dust attenuates UV/optical and re-emits in the infrared.
Radio continuum from star-forming regions and X-rays from stellar binaries
are included.

Reading order: stellar continuum (gray) → dust-attenuated stellar
(faded) → dust emission (warm FIR + cold submm) → radio (cm-wavelength)
→ X-ray (from accreting binaries).

**References:**

 - Kennicutt & Evans (2012) [1]_ for MW analog SFR calibration
 - Bruzual & Charlot (2003) [2]_ for stellar population synthesis
 - Dale et al. (2014) [3]_ for dust emission model

.. [1] Kennicutt, R. C., & Evans, N. J. (2012).
   *Annu. Rev. Astron. Astrophys.* **50**, 531–608.
   https://doi.org/10.1146/annurev-astro-081811–146504

.. [2] Bruzual, G., & Charlot, S. (2003).
   *Mon. Not. R. Astron. Soc.* **344**, 1000–1028.
   https://doi.org/10.1046/j.1365–8711.2003.06897.x

.. [3] Dale, D. A., et al. (2014).
   *Astrophys. J.* **784**, 83.
   https://doi.org/10.1088/0004–637X/784/1/83
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import os
import warnings

import matplotlib

matplotlib.use("Agg")
import jax
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")


# Load stellar population synthesis grid
SSP = tengri.load_ssp("fsps_prsc_miles_chabrier")

# MW analog: moderate-mass, sustained star formation with a recent uptick
# Double power law: τ_gyr = 13 Gyr (age of universe proxy), mild cusp
# log_total_mass ≈ 0 → peak SFR ~ 1 Msun/yr, log10 scaling
HOST = dict(
    sfh={
        "type": "dpl",
        "all_params": tengri.FIXED,
        "tau_gyr": 9.0,  # Main growth timescale
        "log_total_mass": 10.0,  # Peak SFR ~ 2.2 Msun/yr (Kennicutt & Evans 2012)
        "alpha": 0.8,  # Early rise
        "beta": 0.5,  # Late decline
    },
    redshift=tengri.Fixed(0.05),  # z=0.05 for cosmic variance context
)

# Dust: two-component model (diffuse + birth cloud) + Dale et al. IR emission
# τ values typical for MW analogs (Calzetti 2000 calibration)
DUST_ON = {
    "type": "two_component",
    "all_params": tengri.FIXED,
    "tau_diff": 0.35,  # Diffuse ISM optical depth
    "tau_bc": 0.45,  # Birth cloud optical depth
    "emission": {"type": "dale2014", "all_params": tengri.FIXED},  # FIR + submm reprocessing
}

# Dust-free reference
DUST_OFF = {
    "type": "two_component",
    "all_params": tengri.FIXED,
    "tau_diff": 0.0,
    "tau_bc": 0.0,
}


def _nuLnu(**blocks):
    """Build model, sample, predict, and return nu*L_nu in rest frame."""
    model = tengri.SEDModel.build(SSP, **HOST, **blocks)
    p = dict(model.spec.sample(jax.random.PRNGKey(0)))
    out = model.predict(p)
    wave = np.asarray(model.wavelengths)
    return wave, 2.998e18 / wave * np.asarray(out.rest_sed())


# Build cumulative traces: each adds one physics layer
# Wavelength label positions depend on y scale; set after first plot
RUNS = [
    ("Stellar continuum", "#666666", dict(dust=DUST_OFF)),
    ("Attenuated by dust", "#999999", dict(dust=DUST_ON)),
    (
        "+ Dust emission",
        "#dd7733",
        dict(dust=DUST_ON, neb={"type": "cue", "all_params": tengri.FIXED}),
    ),
    (
        "+ Radio (SF regions)",
        "#3366cc",
        dict(
            dust=DUST_ON,
            neb={"type": "cue", "all_params": tengri.FIXED},
            radio={"type": "condon92", "all_params": tengri.FIXED},
        ),
    ),
    (
        "+ X-ray (XRBs)",
        "#9933cc",
        dict(
            dust=DUST_ON,
            neb={"type": "cue", "all_params": tengri.FIXED},
            radio={"type": "condon92", "all_params": tengri.FIXED},
            xray={"type": "simple", "all_params": tengri.FIXED},
        ),
    ),
]

fig, ax = plt.subplots(figsize=(9.0, 5.5))

# Plot each cumulative trace
for label, color, blocks in RUNS:
    wave, nuL = _nuLnu(**blocks)
    ax.loglog(wave, nuL, color=color, lw=1.5, label=label, zorder=3)

# Wavelength region labels (positioned to guide reading)
ax.text(100, 8e42, "X-ray", fontsize=7, color="0.5", ha="center", alpha=0.6, style="italic")
ax.text(1000, 8e42, "UV", fontsize=7, color="0.5", ha="center", alpha=0.6, style="italic")
ax.text(5500, 8e42, "Optical", fontsize=7, color="0.5", ha="center", alpha=0.6, style="italic")
ax.text(2e4, 8e42, "NIR", fontsize=7, color="0.5", ha="center", alpha=0.6, style="italic")
ax.text(1e5, 8e42, "MIR", fontsize=7, color="0.5", ha="center", alpha=0.6, style="italic")
ax.text(5e6, 8e42, "FIR", fontsize=7, color="0.5", ha="center", alpha=0.6, style="italic")
ax.text(1e9, 8e42, "Radio", fontsize=7, color="0.5", ha="center", alpha=0.6, style="italic")

# Axis formatting
ax.set(
    xlim=(10, 1e9),
    ylim=(1e41, 1e44),
    xlabel=r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]",
    ylabel=r"$\nu L_\nu$ [erg s$^{-1}$]",
)

# Legend: two columns, no frame
ax.legend(frameon=False, fontsize=8.5, loc="lower left", ncol=2, handlelength=1.2)

# Grid for clarity (light)
ax.grid(True, alpha=0.15, linestyle="--", linewidth=0.5)

fig.tight_layout()
plt.savefig("plot_panchromatic_milky_way_analog.png", dpi=150, bbox_inches="tight")
