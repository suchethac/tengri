"""
SED Components
==============

Predict a galaxy SED at fixed parameters, then re-predict with dust
optical depths set to zero. Overplot to see how much UV-optical light
the dust absorbed.

.. sphx-glr-precomputed-img:

.. image:: images/sphx_glr_plot_sed_components_001.png
   :alt: plot_sed_components
   :class: sphx-glr-single-img

"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from tengri import (
    FIXED,
    Fixed,
    Observation,
    Photometry,
    SEDModel,
    data_path,
    load_ssp,
)
from tengri.analysis.plotting import setup_style

setup_style()

# --- Build a fully-fixed dusty galaxy at z = 0 ---
obs = Observation(
    photometry=Photometry.from_names(["sdss_r"], cache_dir=str(data_path("filters"))),
)
model = SEDModel.from_groups(
    ssp_data=load_ssp(),
    observation=obs,
    sfh={
        "type": "tsnorm",
        "*": FIXED,
        "log_peak_sfr": 1.2,
        "peak_lbt_gyr": 5.0,
        "width_gyr": 2.0,
        "skew": 0.5,
        "trunc": 3.0,
    },
    dust={
        "type": "two_component",
        "law_bc": "calzetti",
        "*": FIXED,
        "tau_bc": 1.0,
        "tau_diff": 0.5,
        "slope": -0.7,
    },
    redshift=Fixed(0.0),
)

# --- Predict twice: with dust, and with tau_bc=tau_diff=0 ---
params = model.spec.sample(jax.random.PRNGKey(0))
sed_total = np.array(model.predict_rest_sed(params).sed)
sed_intrinsic = np.array(
    model.predict_rest_sed(
        {**params, "dust_tau_bc": jnp.array(0.0), "dust_tau_diff": jnp.array(0.0)}
    ).sed
)

# --- Plot ---
wave_um = np.array(model.ssp_data.ssp_wave) / 1e4
mask = (wave_um > 0.09) & (wave_um < 3.0)

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(
    wave_um[mask], sed_intrinsic[mask], color="C0", lw=1.2, alpha=0.8, label="Intrinsic (no dust)"
)
ax.plot(wave_um[mask], sed_total[mask], color="C3", lw=1.2, label="Attenuated (total)")
ax.fill_between(
    wave_um[mask],
    sed_total[mask],
    sed_intrinsic[mask],
    alpha=0.15,
    color="C3",
    label="Dust absorbed",
)

ax.set(
    xlabel=r"Wavelength [$\mu$m]",
    ylabel=r"$L_\nu$ [erg/s/Hz]",
    title="SED Components: Intrinsic vs Dust-Attenuated",
    xscale="log",
    yscale="log",
    xlim=(0.09, 3.0),
    ylim=(1e20, 1e29),
)
ax.legend(fontsize=10, frameon=False, loc="upper right")
fig.tight_layout()
plt.savefig("plot_sed_components.png", dpi=150, bbox_inches="tight")
plt.show()
