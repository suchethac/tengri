"""
Balmer Decrement Tests Dust Attenuation on Emission Lines
==========================================================

The Balmer decrement measures dust attenuation via hydrogen recombination line
ratios: H-alpha / H-beta is sensitive to extinction (Calzetti et al. 2000).
Without dust, the intrinsic ratio is ~2.78–2.86 (Case B). Here we sweep
dust optical depth (τ_diff ∈ [0, 2]) and measure how the predicted H-alpha
and H-beta change. We derive A_V = 1.086 × τ_diff and compare against the
Calzetti+2000 expectation.

``predict(params).lines`` folds in the diffuse dust attenuation, so the
predicted decrement already carries it. The curve here climbs less steeply
than the pure Calzetti law used as the reference: birth-cloud attenuation and
the dust normalization both differ from that idealization.

Reference: Calzetti et al. 2000, ApJ, 533, 682 (Balmer decrement and dust
attenuation law).
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
warnings.filterwarnings("ignore", message=".*deprecated.*")

# Load bare-stellar SSP (required for Cue)
ssp = tengri.load_ssp("fsps_prsc_miles_chabrier")

# Build model with tau_diff as the only free parameter
# Young star-forming galaxy with Cue nebular backend
model = tengri.SEDModel.build(
    ssp,
    sfh={
        "type": "dpl",
        "all_params": tengri.FIXED,
        "alpha": 1.5,
        "beta": 2.0,
        "tau_gyr": 0.1,
        "log_total_mass": 10.0,
    },
    dust={
        "type": "two_component",
        "all_params": tengri.FIXED,
        "tau_bc": tengri.FIXED,  # Birth cloud dust fixed
        "tau_diff": tengri.Uniform(0.0, 2.0),  # Sweep diffuse dust
        "slope": tengri.Fixed(-0.7),
    },
    neb={
        "type": "cue",
        "all_params": tengri.FIXED,
        "logZ_gas": -0.2,
        "logU": -3.0,
    },
    redshift=tengri.Fixed(0.05),
)

# Sample tau_diff values
tau_diff_values = np.linspace(0.0, 2.0, 21)
av_values = 1.086 * tau_diff_values  # Derived A_V

# Store line ratios
ha_hb_ratio = []

# Get baseline parameters
baseline_params = dict(model.spec.sample(jax.random.PRNGKey(0)))

for tau_diff in tau_diff_values:
    # Only vary dust optical depth
    params = {**baseline_params, "dust_tau_diff": np.float64(tau_diff), "dust_tau_bc": 0.1}
    lines = model.predict(params).lines

    if lines is not None:
        ha = float(lines.halpha)
        hb = float(lines.hbeta)

        if ha > 0 and hb > 0:
            ratio = ha / hb
            ha_hb_ratio.append(ratio)
        else:
            ha_hb_ratio.append(np.nan)
    else:
        ha_hb_ratio.append(np.nan)

# Create figure
fig, ax = plt.subplots(figsize=(8, 5))

ha_hb_ratio = np.array(ha_hb_ratio)
valid_mask = ~np.isnan(ha_hb_ratio)
valid_av = av_values[valid_mask]
valid_ratios = ha_hb_ratio[valid_mask]

# Plot tengri prediction
ax.plot(
    valid_av,
    valid_ratios,
    "o-",
    lw=2.0,
    ms=5,
    color="C0",
    label="tengri H-alpha / H-beta",
)

# Overlay Calzetti+2000 expectation: intrinsic ratio modified by dust
# Calzetti attenuation at Hα (6563 Å) and Hβ (4861 Å)
# A_λ / A_V = 1.0 at 5500 Å (reference); use effective Calzetti curve
intrinsic_ratio = 2.85  # Case B at solar metallicity
# Calzetti curve: A_λ / A_V increases shortward of 5500 Å
# Approximate for Balmer lines: Hβ is shorter → more attenuated
# Empirically: Hβ attenuation ~ 1.3 × A_V, Hα ~ 0.97 × A_V (rough)
calzetti_av_grid = np.linspace(0, 2.0, 100)
calzetti_ha_atten = 0.97 * calzetti_av_grid
calzetti_hb_atten = 1.3 * calzetti_av_grid
calzetti_ratio = intrinsic_ratio * 10 ** (0.4 * (calzetti_hb_atten - calzetti_ha_atten))

ax.plot(
    calzetti_av_grid,
    calzetti_ratio,
    "--",
    lw=2.0,
    color="C1",
    label="Calzetti+2000 expectation",
)

ax.set_xlabel(r"Dust Attenuation $A_V$ [mag]", fontsize=12)
ax.set_ylabel(r"H-alpha / H-beta", fontsize=12)
ax.set_ylim([2.5, 5.0])
ax.legend(frameon=False, loc="upper left", fontsize=10)
ax.grid(True, alpha=0.3)

# Add note
ax.text(
    0.98,
    0.05,
    "Line predictions include diffuse dust attenuation",
    transform=ax.transAxes,
    fontsize=9,
    va="bottom",
    ha="right",
    style="italic",
    color="0.4",
)

fig.tight_layout()
plt.savefig("plot_usecase_balmer_decrement_av.png", dpi=150, bbox_inches="tight")
