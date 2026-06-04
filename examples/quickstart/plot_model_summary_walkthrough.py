"""
Understanding model structure through parameter provenance tags
==============================================================

The ``model.spec.summary()`` method displays each parameter's source
through provenance tags: ``[user]`` for explicit overrides, ``[* FREE]``
and ``[* FIXED]`` for wildcard expansions, and ``[default]`` for
registry defaults. We build a model with mixed constraints, display the
annotated summary as a figure caption, and show the predicted SED.
"""

import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

# Load bare-stellar SSP from canonical data directory (Cue requires bare SSP)
ssp_path = Path("/Users/suchethacooray/Projects/tengri/data/fsps_prsc_miles_chabrier.h5")
ssp = tengri.load_ssp_data(str(ssp_path))

# Build a model with deliberately mixed provenance sources.
# - sfh: tsnorm with FREE wildcards except one fixed param
# - dust: two-component with FIXED wildcards but user override on tau_bc
# - neb: all defaults (no wildcard)
# - redshift: explicit user value
model = tengri.SEDModel.build(
    ssp,
    sfh={
        "type": "tsnorm",
        "*": tengri.FREE,
        "logzsol": tengri.Fixed(-0.1),  # [user] override on a FREE wildcard
    },
    dust={
        "type": "two_component",
        "law_bc": "calzetti",
        "*": tengri.FIXED,
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
params["sfh_tsnorm_peak_lbt_gyr"] = jnp.float64(3.0)
params["sfh_tsnorm_skew"] = jnp.float64(-0.2)

# Predict rest-frame SED
sed = model.predict_rest_sed(params)
wave = np.asarray(sed.wavelength)
nu = 2.998e18 / wave  # Å/s -> Hz
nu_l_nu = nu * np.asarray(sed.sed)

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
output_path = Path(__file__).parent / "plot_model_summary_walkthrough.png"
plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
plt.close(fig)
