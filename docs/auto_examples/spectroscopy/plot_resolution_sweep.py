"""
Instrumental resolution controls Hα + [N II] line blending
===========================================================

Spectral resolution R determines whether the Hα + [N II] emission-line
complex appears as a single blended feature (low R) or resolves into three
distinct lines (high R). Varying R from 100 to 10000 reveals the transition
from kinematically degenerate at R~100 (SDSS/DESI-like) to fully resolved
at R~5000 (JWST-like).
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import warnings

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.plot import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

ssp = tengri.load_ssp("fsps_prsc_miles_chabrier")

REDSHIFT = 0.05

spec = tengri.Spectroscopy(
    wave_obs=jnp.linspace(6400.0 * (1 + REDSHIFT), 6700.0 * (1 + REDSHIFT), 800)
)
obs = tengri.Observation(spectroscopy=spec)

model = tengri.SEDModel.build(
    ssp,
    observation=obs,
    sfh={
        "type": "dpl",
        "all_params": tengri.FIXED,
        "tau_gyr": 0.05,
        "log_total_mass": 10.0,
        "alpha": 2.5,
        "beta": 1.8,
    },
    dust={"type": "two_component", "all_params": tengri.FIXED, "tau_bc": 0.1, "tau_diff": 0.05},
    neb={"type": "cue", "all_params": tengri.FIXED, "logU": -2.0},
    redshift=tengri.Fixed(REDSHIFT),
)

baseline = dict(model.spec.sample(jax.random.PRNGKey(0)))

resolution_vals = np.array([100.0, 300.0, 1000.0, 3000.0, 5000.0, 10000.0])
norm = mpl.colors.Normalize(vmin=resolution_vals.min(), vmax=resolution_vals.max())
cmap = plt.get_cmap("viridis")

fig, ax = plt.subplots(figsize=(7.5, 4.8))

wave_rest = np.linspace(6400.0, 6700.0, 600)
for R in resolution_vals:
    spec_r = tengri.Spectroscopy(wave_obs=wave_rest * (1 + REDSHIFT), resolution=float(R))
    obs_r = tengri.Observation(spectroscopy=spec_r)

    model_r = tengri.SEDModel.build(
        ssp,
        observation=obs_r,
        sfh={
            "type": "dpl",
            "all_params": tengri.FIXED,
            "tau_gyr": 0.05,
            "log_total_mass": 10.0,
            "alpha": 2.5,
            "beta": 1.8,
        },
        dust={
            "type": "two_component",
            "all_params": tengri.FIXED,
            "tau_bc": 0.1,
            "tau_diff": 0.05,
        },
        neb={"type": "cue", "all_params": tengri.FIXED, "logU": -2.0},
        redshift=tengri.Fixed(REDSHIFT),
    )

    flux = np.asarray(model_r.predict_spectrum(baseline, wave_obs=wave_rest * (1 + REDSHIFT)))
    cont_mask = (wave_rest >= 6400.0) & (wave_rest <= 6450.0)
    f_cont = np.median(flux[cont_mask])
    ax.plot(wave_rest, flux / f_cont, color=cmap(norm(R)), lw=1.4, label=f"R = {R:.0f}")

ax.axvline(6548.05, color="0.5", lw=0.5, ls=":", alpha=0.7)
ax.axvline(6562.80, color="0.5", lw=0.5, ls=":", alpha=0.7)
ax.axvline(6583.46, color="0.5", lw=0.5, ls=":", alpha=0.7)

ax.set_xlim(6480.0, 6620.0)
ax.set_ylim(0.85, 1.18)
ax.set_xlabel(r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]")
ax.set_ylabel(r"Normalized $F_\lambda$")

cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.01)
cbar.set_label(r"Resolution $R$ ($\lambda / \Delta\lambda$)")

fig.tight_layout()
plt.savefig("plot_resolution_sweep.png", dpi=150, bbox_inches="tight")
