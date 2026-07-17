"""
Rest-to-observer-frame transformation of panchromatic SEDs with redshift
=========================================================================

Same galaxy rest-frame panchromatic SED (UV through radio) observed at
increasing redshifts. Cosmological redshift transforms rest-frame wavelengths
and dims luminosity, shifting spectral features to infrared bands at high
redshift where ground-based surveys probe star formation epochs.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

ssp = tengri.load_ssp()

# Redshifts to sweep
redshifts = np.array([0.2, 0.8, 1.5, 3.0])
cmap = plt.get_cmap("viridis")
norm = plt.Normalize(vmin=redshifts.min(), vmax=redshifts.max())

fig, ax = plt.subplots(figsize=(12, 5.2))

key = jax.random.PRNGKey(0)

for z in redshifts:
    model = tengri.SEDModel.build(
        ssp,
        sfh={
            "type": "dpl",
            "all_params": tengri.FIXED,
            "alpha": 2.0,
            "beta": 2.5,
            "tau_gyr": 1.5,
            "log_total_mass": 10.0,
        },
        dust={
            "type": "two_component",
            "all_params": tengri.FIXED,
            "tau_bc": 0.5,
            "tau_diff": 0.3,
            "emission": {"type": "dale2014", "all_params": tengri.FIXED},
        },
        redshift=tengri.Fixed(z),
    )

    baseline = dict(model.spec.sample(key))
    params = {**baseline}

    # Predict rest-frame SED
    out = model.predict(params)
    wave_rest = np.asarray(model.wavelengths)
    sed_rest = np.asarray(out.rest_sed())

    # Shift to observed frame
    wave_obs_um = (wave_rest / 1e4) * (1 + z)
    l_nu_obs = sed_rest / (1 + z)

    # Radio component (rest-frame)
    wave_radio_rest = jnp.logspace(7, 11, 150)
    l_ir_erg = 3e11 * 3.839e33
    l_radio_rest = np.array(
        tengri.radio.radio_star_forming(wave_radio_rest, L_ir=l_ir_erg, alpha_sf=0.8)
    )
    wave_radio_obs_um = (wave_radio_rest / 1e4) * (1 + z)
    l_radio_obs = l_radio_rest / (1 + z)

    # Plot stellar + dust SED
    mask = sed_rest > 0
    ax.loglog(
        wave_obs_um[mask],
        l_nu_obs[mask],
        lw=2.0,
        color=cmap(norm(z)),
    )

    # Plot radio component
    mask_r = l_radio_rest > 0
    ax.loglog(
        wave_radio_obs_um[mask_r],
        l_radio_obs[mask_r],
        lw=2.0,
        color=cmap(norm(z)),
    )

ax.set_xlim(0.05, 1e6)
ax.set_ylim(1e19, 1e35)
ax.set_xlabel(r"Observed-frame wavelength $\lambda$ [$\mu$m]")
ax.set_ylabel(r"$L_\nu$ [erg s$^{-1}$ Hz$^{-1}$]")

cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.01)
cbar.set_label(r"Redshift $z$")

fig.tight_layout()
plt.savefig("plot_panchromatic_redshift_sweep.png", dpi=150, bbox_inches="tight")
