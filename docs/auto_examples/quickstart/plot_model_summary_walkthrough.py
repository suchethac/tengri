"""
Understanding model structure through parameter provenance tags
===============================================================
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.plot import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

# Load bare-stellar SSP via the public loader (Cue requires a bare SSP).
ssp = tengri.load_ssp("fsps_prsc_miles_chabrier")

# Build a model with deliberately mixed provenance sources.
# - sfh: tsnorm with FREE wildcards except one fixed param
# - dust: two-component with FIXED wildcards but user override on tau_bc
# - neb: all defaults (no wildcard)
# - redshift: explicit user value
model = tengri.SEDModel.build(
    ssp,
    sfh={
        "type": "tsnorm",
        "all_params": tengri.FREE,
        "logzsol": tengri.Fixed(-0.1),  # [user] override on a FREE wildcard
    },
    dust_attenuation={
        "type": "two_component",
        "law": "calzetti",
        "all_params": tengri.FIXED,
        "tau_bc": 0.5,  # [user] override on a FIXED wildcard
    },
    neb={"type": "cue"},  # All defaults
    redshift=tengri.Fixed(0.05),  # [user]
)

# Capture the summary() output as a string
summary_out = model.spec.summary_str()

# Create a sample parameter dict for the SED prediction
key = jax.random.PRNGKey(0)
params = dict(model.spec.sample(key))
# Override a few to show variability
params["sfh_tsnorm_log_total_mass"] = jnp.float64(11.5)
params["sfh_tsnorm_peak_lbt_gyr"] = jnp.float64(3.0)
params["sfh_tsnorm_skew"] = jnp.float64(-0.2)

# Predict rest-frame SED
sed = model.predict(params)
wave = np.asarray(model.wavelengths)
nu = 2.998e18 / wave  # Å/s -> Hz
nu_l_nu = nu * np.asarray(sed.rest_sed())

# Create a 2-panel figure:
# LEFT: text rendering of summary (monospace, annotated)
# RIGHT: the predicted SED
fig, (ax_text, ax_sed) = plt.subplots(
    1, 2, figsize=(13.5, 5.0), gridspec_kw={"width_ratios": [1.3, 1]}
)

# LEFT panel: render the summary text in monospace
ax_text.axis("off")
ax_text.set_xlim(0, 1)
ax_text.set_ylim(0, 1)

# Render the summary in monospace font, aligned top-left
y_pos = 0.98
lines = summary_out.split("\n")
for line in lines:
    ax_text.text(
        0.02,
        y_pos,
        line,
        fontfamily="monospace",
        fontsize=8,
        verticalalignment="top",
        transform=ax_text.transAxes,
    )
    y_pos -= 0.035

# RIGHT panel: rest-frame SED
ax_sed.loglog(wave, nu_l_nu, lw=1.8, color="C0")
ax_sed.set_xlim(800, 3e4)
ax_sed.set_ylim(1e40, 5e43)
ax_sed.set_xlabel(r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]")
ax_sed.set_ylabel(r"$\nu L_\nu$  [erg s$^{-1}$]")
ax_sed.grid(True, alpha=0.2, which="both")

fig.tight_layout()
plt.savefig("plot_model_summary_walkthrough.png", dpi=150, bbox_inches="tight")
