"""
JWST NIRCam color-color diagnostics for high-z galaxy classification
====================================================================
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

ssp = tengri.load_ssp()

# Build JWST filter observation
try:
    obs = tengri.Observation(
        photometry=tengri.Photometry.from_names(["jwst_f150w", "jwst_f277w", "jwst_f444w"])
    )
except Exception:
    # Fallback if filter names differ
    obs = None

# Generate three galaxy classes
key = jax.random.PRNGKey(123)
colors_all = {"sf": [], "passive": [], "dusty": []}
z_all = {"sf": [], "passive": [], "dusty": []}

# Star-forming (z=1-7, young, extended SFH)
for _i in range(50):
    z = np.random.uniform(1.0, 7.0)
    model = tengri.SEDModel.build(
        ssp,
        observation=obs,
        sfh={
            "type": "tsnorm",
            "log_total_mass": 10.0,
            "peak_lbt_gyr": tengri.Uniform(0.2, 2.0),
            "width_gyr": tengri.Uniform(0.5, 3.0),
            "skew": tengri.Uniform(-1.0, 1.0),
            "trunc": tengri.Uniform(1.5, 5.0),
            "logzsol": tengri.Uniform(-1.0, 0.1),
        },
        dust={
            "type": "two_component",
            "tau_bc": tengri.Uniform(0.0, 0.8),
            "tau_diff": tengri.Uniform(0.0, 0.5),
            "slope": tengri.Fixed(-0.7),
        },
        redshift=tengri.Fixed(z),
    )
    key, subkey = jax.random.split(key)
    params = model.spec.sample(subkey)
    phot = np.asarray(model.predict_photometry(params))
    if len(phot) == 3:
        f0, f1, f2 = phot[0], phot[1], phot[2]
        color1 = -2.5 * np.log10(max(f0 / f1, 1e-3))
        color2 = -2.5 * np.log10(max(f1 / f2, 1e-3))
        colors_all["sf"].append((color1, color2))
        z_all["sf"].append(z)

# Passive (z=1-3, old, narrow SFH)
for _i in range(50):
    z = np.random.uniform(1.0, 3.0)
    model = tengri.SEDModel.build(
        ssp,
        observation=obs,
        sfh={
            "type": "tsnorm",
            "log_total_mass": 10.0,
            "peak_lbt_gyr": tengri.Uniform(7.0, 11.0),
            "width_gyr": tengri.Uniform(0.5, 1.5),
            "skew": tengri.Uniform(-1.5, 0.0),
            "trunc": tengri.Uniform(1.5, 3.0),
            "logzsol": tengri.Uniform(-0.2, 0.3),
        },
        dust={
            "type": "two_component",
            "all_params": tengri.FIXED,
            "tau_bc": 0.05,
            "tau_diff": 0.02,
            "slope": -0.7,
        },
        redshift=tengri.Fixed(z),
    )
    key, subkey = jax.random.split(key)
    params = model.spec.sample(subkey)
    phot = np.asarray(model.predict_photometry(params))
    if len(phot) == 3:
        f0, f1, f2 = phot[0], phot[1], phot[2]
        color1 = -2.5 * np.log10(max(f0 / f1, 1e-3))
        color2 = -2.5 * np.log10(max(f1 / f2, 1e-3))
        colors_all["passive"].append((color1, color2))
        z_all["passive"].append(z)

# Dusty/AGN (z=2-4, high dust, high SFR)
for _i in range(50):
    z = np.random.uniform(2.0, 4.0)
    model = tengri.SEDModel.build(
        ssp,
        observation=obs,
        sfh={
            "type": "tsnorm",
            "log_total_mass": 10.0,
            "peak_lbt_gyr": tengri.Uniform(0.5, 3.0),
            "width_gyr": tengri.Uniform(1.0, 4.0),
            "skew": tengri.Uniform(-0.5, 1.0),
            "trunc": tengri.Uniform(2.0, 6.0),
            "logzsol": tengri.Uniform(-0.5, 0.2),
        },
        dust={
            "type": "two_component",
            "tau_bc": tengri.Uniform(0.8, 2.0),
            "tau_diff": tengri.Uniform(0.5, 1.5),
            "slope": tengri.Fixed(-0.7),
        },
        redshift=tengri.Fixed(z),
    )
    key, subkey = jax.random.split(key)
    params = model.spec.sample(subkey)
    phot = np.asarray(model.predict_photometry(params))
    if len(phot) == 3:
        f0, f1, f2 = phot[0], phot[1], phot[2]
        color1 = -2.5 * np.log10(max(f0 / f1, 1e-3))
        color2 = -2.5 * np.log10(max(f1 / f2, 1e-3))
        colors_all["dusty"].append((color1, color2))
        z_all["dusty"].append(z)

# Plot
fig, ax = plt.subplots(figsize=(8, 7))

# Plot each class
for key_name, marker, color, label in [
    ("sf", "o", "#1f77b4", "Star-forming"),
    ("passive", "s", "#d62728", "Passive"),
    ("dusty", "^", "#ff7f0e", "Dusty/AGN"),
]:
    if colors_all[key_name]:
        colors_arr = np.array(colors_all[key_name])
        ax.scatter(
            colors_arr[:, 0],
            colors_arr[:, 1],
            marker=marker,
            s=60,
            alpha=0.7,
            edgecolors="k",
            lw=0.5,
            color=color,
            label=label,
        )

ax.set_xlabel(r"F150W - F277W [mag]")
ax.set_ylabel(r"F277W - F444W [mag]")
ax.legend(frameon=False, loc="upper left")
ax.set_xlim([-0.5, 3.0])
ax.set_ylim([-0.5, 2.0])

fig.tight_layout()
plt.savefig("plot_usecase_jwst_color_color.png", dpi=150, bbox_inches="tight")
