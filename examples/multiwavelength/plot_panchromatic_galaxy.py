"""
UV-to-radio SED of a star-forming galaxy
=========================================
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
        "tau_diff": 0.3,
        "tau_bc": 0.5,
        "emission": {"type": "dale2014", "all_params": tengri.FIXED},
    },
    redshift=tengri.Fixed(0.05),
)

baseline = dict(model.spec.sample(jax.random.PRNGKey(0)))
params = {**baseline}

# Predict stellar + dust SED (rest-frame, UV through IR)
out = model.predict(params)
wave_sed = np.asarray(model.wavelengths)
sed = np.asarray(out.rest_sed())

# Wavelength grid: UV through radio
wave_full = jnp.logspace(jnp.log10(500.0), jnp.log10(1e10), 3000)
wave_full_um = np.array(wave_full) / 1e4

# Radio synchrotron (separate calculation)
# Assume typical L_IR ~ 3e11 L_sun for this star-forming galaxy
L_ir_erg = 3e11 * 3.839e33
l_radio = np.array(tengri.radio.radio_star_forming(wave_full, L_ir=L_ir_erg, alpha_sf=0.8))

# Extend stellar SED to radio with radio component
wave_sed_um = np.array(wave_sed) / 1e4
nu_sed = 2.998e18 / np.array(wave_sed)
nu_l_nu_sed = nu_sed * np.array(sed)

nu_full = 2.998e18 / wave_full
nu_l_nu_radio = nu_full * l_radio

fig, ax = plt.subplots(figsize=(10, 5.2))

# Stellar + dust SED
mask_sed = sed > 0
ax.loglog(wave_sed_um[mask_sed], nu_l_nu_sed[mask_sed], color="C0", lw=2.0, label="Stellar + dust")

# Radio synchrotron
mask_radio = l_radio > 0
ax.loglog(wave_full_um[mask_radio], nu_l_nu_radio[mask_radio], color="C2", lw=2.0, label="Radio")

# Regime labels
for x_um, _lbl in [(0.1, "UV"), (0.5, "Optical"), (10.0, "IR"), (300.0, "FIR"), (1e4, "Radio")]:
    ax.axvline(x_um, color="0.8", lw=0.6, ls=":", alpha=0.5)

ax.set_xlim(0.08, 3e4)
# Tighten y-limits based on data range (stellar/dust dominates; radio is faint context)
ymax = max(np.max(nu_l_nu_sed[mask_sed]), np.max(nu_l_nu_radio[mask_radio]))
ymin = min(np.min(nu_l_nu_sed[mask_sed]), np.min(nu_l_nu_radio[mask_radio]))
ax.set_ylim(ymin / 2.0, ymax * 2.0)
ax.set_xlabel(r"Rest-frame wavelength $\lambda$ [$\mu$m]")
ax.set_ylabel(r"$\nu L_\nu$ [erg s$^{-1}$]")
ax.legend(frameon=False, fontsize=9)

fig.tight_layout()
plt.savefig("plot_panchromatic_galaxy.png", dpi=150, bbox_inches="tight")
