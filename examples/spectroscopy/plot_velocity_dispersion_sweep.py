"""
Stellar Velocity Dispersion Sweep
===================================

Sweep stellar velocity dispersion σ_v ∈ {50, 100, 150, 250, 400} km/s to show
how line broadening increases with dynamical heating. The Mg b absorption
feature (~5170 Å) widens progressively, demonstrating the kinematic signature
of higher-velocity stellar populations.

.. sphx-glr-precomputed-img:

.. image:: images/sphx_glr_plot_velocity_dispersion_sweep_001.png
   :alt: plot_velocity_dispersion_sweep
   :class: sphx-glr-single-img

"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

from tengri import Fixed, Observation, Parameters, SEDModel, Spectroscopy, load_ssp
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")


ssp = load_ssp()

# --- Setup ---
WAVE_OBS = jnp.linspace(4800.0, 5500.0, 300)
REDSHIFT = 0.1

obs = Observation(
    spectroscopy=Spectroscopy(wave_obs=WAVE_OBS),
)

spec = Parameters(
    sfh_tsnorm_log_total_mass=10.0),
    sfh_tsnorm_peak_lbt_gyr=Fixed(2.5),
    sfh_tsnorm_width_gyr=Fixed(1.8),
    sfh_tsnorm_skew=Fixed(0.1),
    sfh_tsnorm_trunc=Fixed(3.0),
    met_logzsol=Fixed(0.0),
    dust_tau_bc=Fixed(0.1),
    dust_tau_diff=Fixed(0.05),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(REDSHIFT),
)
model = SEDModel(spec, ssp, observation=obs)

# --- Predict rest-frame spectrum ---
pred = model.predict_rest_sed({})
wave_rest = np.asarray(pred.wavelength)
sed_rest = np.asarray(pred.sed)

# --- Interpolate to observed frame and zoom to Mg b region ---
wave_obs_rest = wave_rest / (1.0 + REDSHIFT)
zoom_mask = (wave_obs_rest >= 5000) & (wave_obs_rest <= 5300)
wave_zoom = wave_obs_rest[zoom_mask]
sed_zoom = sed_rest[zoom_mask]

# Normalize
sed_zoom = sed_zoom / np.max(sed_zoom)

# --- Velocity dispersion sweep ---
sigma_v_vals = [50, 100, 150, 250, 400]  # km/s
colors = plt.cm.viridis(np.linspace(0.0, 0.85, len(sigma_v_vals)))

# Wavelength resolution element (approximate)
dlam_pix = np.mean(np.diff(wave_zoom))

fig, ax = plt.subplots(figsize=(9, 5))

for sigma_v, color in zip(sigma_v_vals, colors):
    # Convert velocity dispersion to wavelength broadening (rest frame)
    sigma_lam = (sigma_v / 3e5) * np.mean(wave_zoom)
    sigma_pix = sigma_lam / dlam_pix

    # Apply Gaussian broadening
    sed_broadened = gaussian_filter1d(sed_zoom, sigma=sigma_pix)

    ax.plot(
        wave_zoom,
        sed_broadened,
        lw=2.0,
        color=color,
        label=f"$\\sigma_v$ = {sigma_v} km/s",
        alpha=0.85,
    )

# Mark Mg b center (vacuum)
mgb_center = 5172.68
ax.axvline(mgb_center, ls=":", lw=1.0, color="grey", alpha=0.5)
ax.text(
    mgb_center,
    0.95,
    "Mg b",
    fontsize=9,
    ha="center",
    color="grey",
    transform=ax.get_xaxis_transform(),
)

ax.set_xlabel(r"Rest Wavelength [$\AA$]")
ax.set_ylabel("Normalized Flux")
ax.set_xlim(5050, 5250)
ax.set_ylim(0.8, 1.02)
ax.legend(frameon=False, loc="lower left", fontsize=10)
fig.tight_layout()
plt.savefig("plot_velocity_dispersion_sweep.png", dpi=150, bbox_inches="tight")
