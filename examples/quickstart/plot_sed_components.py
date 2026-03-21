"""
SED Components
==============

Predict a galaxy SED and visualize its components: the intrinsic
stellar emission and the dust-attenuated total. Uses the lazy
``model.predict()`` API and direct SED computation to show the
effect of dust attenuation on the spectrum.
"""

import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from tengri import (
    Fixed,
    Model,
    Observation,
    ParamSpec,
    Photometry,
    Uniform,
    load_ssp_data,
)

# --- Load SSP data ---
SSP_PATH = "data/ssp_fsps_v3.2.h5"
try:
    ssp = load_ssp_data(SSP_PATH)
except FileNotFoundError:
    print(f"SSP data not found at {SSP_PATH}. Skipping.")
    sys.exit(0)

# --- Define a dusty galaxy model ---
spec = ParamSpec(
    sfh_tsnorm_log_peak_sfr=Fixed(1.2),
    sfh_tsnorm_peak_lbt_gyr=Fixed(5.0),
    sfh_tsnorm_width_gyr=Fixed(2.0),
    sfh_tsnorm_skew=Fixed(0.5),
    sfh_tsnorm_trunc=Fixed(3.0),
    met_logzsol=Fixed(-0.3),
    dust_tau_bc=Fixed(1.0),
    dust_tau_diff=Fixed(0.5),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.0),
    mean_sfh_type="tsnorm",
)

obs = Observation(photometry=Photometry.from_names(["sdss_r"]))
model = Model(spec, ssp, observation=obs)
params = spec.sample(jax.random.PRNGKey(0))

# --- Compute SEDs: with and without dust ---
sed_total = model.predict_sed(params)

spec_nodust = ParamSpec(
    **{k: Fixed(float(params[k])) for k in params},
    dust_tau_bc=Fixed(0.0),
    dust_tau_diff=Fixed(0.0),
    mean_sfh_type="tsnorm",
)
model_nodust = Model(spec_nodust, ssp, observation=obs)
sed_intrinsic = model_nodust.predict_sed(spec_nodust.sample(jax.random.PRNGKey(0)))

wave = np.array(ssp.ssp_wave)
sed_total_np = np.array(sed_total)
sed_intr_np = np.array(sed_intrinsic)

# --- Plot ---
fig, ax = plt.subplots(figsize=(9, 4.5))
mask = (wave > 900) & (wave < 30000)

ax.plot(wave[mask] / 1e4, sed_intr_np[mask], color="C0", lw=1.2,
        alpha=0.8, label="Intrinsic (no dust)")
ax.plot(wave[mask] / 1e4, sed_total_np[mask], color="C3", lw=1.2,
        label="Attenuated (total)")
ax.fill_between(wave[mask] / 1e4, sed_total_np[mask], sed_intr_np[mask],
                alpha=0.15, color="C3", label="Dust absorbed")

ax.set_xlabel(r"Wavelength [$\mu$m]")
ax.set_ylabel(r"$L_\nu$ [erg/s/Hz]")
ax.set_title("SED Components: Intrinsic vs Dust-Attenuated")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(0.09, 3.0)
ax.legend(fontsize=9, frameon=False, loc="upper right")
fig.tight_layout()
plt.savefig("plot_sed_components.png", dpi=150, bbox_inches="tight")
plt.show()
