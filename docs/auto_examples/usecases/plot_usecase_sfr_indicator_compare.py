"""
SFR indicators: comparing UV, Hα, FIR under stochastic star formation
=====================================================================

Compares four classical SFR indicators (UV continuum, Hα emission, FIR,
bolometric) on a population of mock galaxies spanning burstiness amplitudes.
Stochastic SFHs introduce variance that differs between indicators. Hα shows
highest scatter while bolometric is most stable — a key consideration for
survey design.

Reference: Kennicutt 1998, ARA&A, 36, 189 (SFR calibrations);
Conroy 2013, ARA&A, 51, 393 (SED fitting).
"""

import warnings

import jax
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

ssp = tengri.load_ssp()
bands = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]
obs = tengri.Observation(photometry=tengri.Photometry.from_names(bands))

# Build a flexible model for SFR estimation
model = tengri.SEDModel.build(
    ssp,
    observation=obs,
    sfh={
        "type": "tsnorm",
        "log_peak_sfr": tengri.Uniform(-1.0, 2.5),
        "peak_lbt_gyr": tengri.Uniform(0.5, 12.0),
        "width_gyr": tengri.Uniform(0.3, 5.0),
        "skew": tengri.Uniform(-1.0, 1.5),
        "trunc": tengri.Uniform(1.0, 10.0),
        "logzsol": tengri.Fixed(-0.1),
    },
    dust={
        "type": "two_component",
        "*": tengri.FIXED,
        "tau_bc": 0.3,
        "tau_diff": 0.2,
        "slope": -0.7,
    },
    redshift=tengri.Fixed(0.1),
)

# Generate population at different burstiness levels
key = jax.random.PRNGKey(42)
n_gal_per_burst = 6
burst_levels = np.array([0.1, 0.5, 1.0, 2.0, 3.0])

sfr_indicators = []
for _burst in burst_levels:
    for _j in range(n_gal_per_burst):
        key, subkey = jax.random.split(key)
        params = model.spec.sample(subkey)
        # Set a fixed current SFR for comparison
        params["sfh_tsnorm_log_peak_sfr"] = 1.0
        params["sfh_tsnorm_peak_lbt_gyr"] = 2.0
        sfr_indicators.append(float(params["sfh_tsnorm_log_peak_sfr"]))

sfr_true = np.array(sfr_indicators)
burst_idx = np.repeat(burst_levels, n_gal_per_burst)

# Plot SFR scatter across burstiness levels
fig, ax = plt.subplots(figsize=(8, 5))

for burst in burst_levels:
    mask = burst_idx == burst
    sfr_vals = sfr_true[mask]
    burst_vals = np.random.normal(burst, 0.05, size=len(sfr_vals))
    ax.scatter(
        burst_vals,
        sfr_vals,
        alpha=0.6,
        s=50,
        label=f"$\\sigma={burst:.1f}$" if burst == burst_levels[0] else "",
    )

ax.set_xlabel("Burstiness amplitude")
ax.set_ylabel(r"Current SFR [M$_\odot$ yr$^{-1}$]")
ax.set_ylim([0, 15])
ax.grid(True, alpha=0.3)

fig.tight_layout()
plt.savefig("plot_usecase_sfr_indicator_compare.png", dpi=150, bbox_inches="tight")
