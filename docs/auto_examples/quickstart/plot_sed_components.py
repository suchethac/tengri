"""
Dust attenuation across the SED: intrinsic, attenuated, and absorbed
====================================================================
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

ssp = tengri.load_ssp()
obs = tengri.Observation(
    photometry=tengri.Photometry.from_names(["sdss_r"]),
)
model = tengri.SEDModel.build(
    ssp,
    observation=obs,
    sfh={
        "type": "tsnorm",
        "all_params": tengri.FIXED,
        "log_total_mass": 10.0,
        "peak_lbt_gyr": 5.0,
        "width_gyr": 2.0,
        "skew": 0.5,
        "trunc": 3.0,
    },
    dust_attenuation={
        "type": "two_component",
        "law": "calzetti",
        "all_params": tengri.FIXED,
        "tau_bc": 1.0,
        "tau_diff": 0.5,
        "slope": -0.7,
    },
    redshift=tengri.Fixed(0.0),
)

params = dict(model.spec.sample(jax.random.PRNGKey(0)))
sed_total = np.array(model.predict(params).rest_sed())
sed_intrinsic = np.array(
    model.predict(
        {**params, "dust_tau_bc": jnp.array(0.0), "dust_tau_diff": jnp.array(0.0)}
    ).rest_sed()
)

wave_um = np.array(model.ssp_data.ssp_wave) / 1e4
mask = (wave_um > 0.09) & (wave_um < 3.0)

fig, ax = plt.subplots(figsize=(7.0, 4.2))
ax.plot(wave_um[mask], sed_intrinsic[mask], color="C0", lw=1.4, label="Intrinsic (no dust)")
ax.plot(wave_um[mask], sed_total[mask], color="C3", lw=1.4, label="Attenuated (total)")
ax.fill_between(
    wave_um[mask],
    sed_total[mask],
    sed_intrinsic[mask],
    alpha=0.2,
    color="C3",
    label="Dust absorbed",
)

ax.set(
    xlabel=r"Wavelength [$\mu$m]",
    ylabel=r"$\nu L_\nu$ [erg s$^{-1}$]",
    xscale="log",
    yscale="log",
    xlim=(0.09, 3.0),
)
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
plt.savefig("plot_sed_components.png", dpi=150, bbox_inches="tight")
