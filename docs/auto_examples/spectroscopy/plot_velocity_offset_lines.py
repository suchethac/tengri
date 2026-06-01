"""
Emission-line velocity dispersion: narrow [NII] to broad Hα
===========================================================

Sweep emission-line velocity dispersion σ_v from 50 km/s (narrow-line region)
to 5000 km/s (broad-line region) to show how Hα broadens while the [NII]
doublet becomes buried in the Hα wing. Demonstrates the kinematic signature
distinguishing AGN BLR from NLR.

.. sphx-glr-precomputed-img:

.. image:: images/sphx_glr_plot_velocity_offset_lines_001.png
   :alt: plot_velocity_offset_lines
   :class: sphx-glr-single-img

"""

import warnings

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Ensure non-interactive backend (required for headless execution)
mpl.use("Agg")

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")


# ─── Setup ───────────────────────────────────────────────────────────
SIGMA_GRID = np.array([50.0, 200.0, 500.0, 1500.0, 5000.0])
Z = 0.05  # Avoid predict_spectrum NaN at z=0
WAVE_OBS = jnp.linspace(6450.0 * (1 + Z), 6700.0 * (1 + Z), 800)
WAVE_REST = np.asarray(WAVE_OBS) / (1 + Z)

# Load SSP and build model with Cue nebular component (bare-stellar SSP)
ssp = tengri.load_ssp("fsps_prsc_miles_chabrier")
spec = tengri.Spectroscopy(wave_obs=WAVE_OBS, resolution=3000.0)
obs = tengri.Observation(spectroscopy=spec)

model = tengri.SEDModel.build(
    ssp,
    observation=obs,
    sfh={
        "type": "dpl",
        "*": tengri.FIXED,
        "alpha": 3.0,
        "beta": 0.3,
        "tau_gyr": 0.03,
        "log_total_mass": 8.48,
    },
    dust={"type": "two_component", "*": tengri.FIXED, "tau_bc": 0.0, "tau_diff": 0.05},
    neb={"type": "cue", "*": tengri.FIXED, "logU": tengri.Fixed(-2.5), "fesc": tengri.Fixed(0.0)},
    redshift=tengri.Fixed(Z),
)

baseline = dict(model.spec.sample(jax.random.PRNGKey(0)))

# ─── Sweep emission-line sigma_v ──────────────────────────────────
norm = mpl.colors.Normalize(vmin=SIGMA_GRID.min(), vmax=SIGMA_GRID.max())
cmap = plt.get_cmap("viridis")

fig, ax = plt.subplots(figsize=(7.0, 4.6))
for sigma in SIGMA_GRID:
    params = {**baseline, "sigma_v_kms": jnp.float64(sigma)}
    spec_out = model.predict_spectrum(params, wave_obs=WAVE_OBS)
    flux = np.asarray(spec_out)

    # Continuum normalization at 6600 Å
    cont_mask = (WAVE_REST >= 6600) & (WAVE_REST <= 6650)
    f_cont = np.median(flux[cont_mask])
    ax.plot(WAVE_REST, flux / f_cont, color=cmap(norm(sigma)), lw=1.2)

# Mark emission lines
ax.axvline(6549.86, color="0.55", lw=0.4, ls=":")  # [NII] 6548
ax.axvline(6564.61, color="0.55", lw=0.4, ls=":")  # Hα 6565
ax.axvline(6585.28, color="0.55", lw=0.4, ls=":")  # [NII] 6584
ax.text(6564.61, 1.20, "Hα", fontsize=8, color="0.4", ha="center")
ax.text(6549.86, 1.27, "[NII]", fontsize=7, color="0.4", ha="center")

ax.set(
    xlim=(6480, 6680),
    yscale="log",
    xlabel=r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]",
    ylabel=r"$F_\lambda\,/\,F_{\rm cont}$ (normalised at 6600-6650 Å)",
)
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.01)
cb.set_label(r"$\sigma_v$  [km s$^{-1}$]")

fig.tight_layout()
plt.show()
