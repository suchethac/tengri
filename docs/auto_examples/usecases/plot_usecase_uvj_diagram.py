"""
UVJ diagram: rest-frame colors separate star-forming from quiescent
===================================================================

Rest-frame U−V vs V−J colors separate star-forming from quiescent galaxies.
The Williams+2009 quiescent wedge marks the boundary between dusty
star-forming and passive systems.

Reference: Williams et al. 2009, ApJ, 691, 1879; Wuyts et al. 2007, ApJ, 655.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import jax
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.plot import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")


def _color(flux: np.ndarray) -> tuple[float, float]:
    """Return (U-V, V-J) AB-magnitude colors from f_nu."""
    f_u, f_v, f_j = float(flux[0]), float(flux[1]), float(flux[2])
    if f_u <= 0 or f_v <= 0 or f_j <= 0:
        return np.nan, np.nan
    uv = -2.5 * np.log10(f_u / f_v)
    vj = -2.5 * np.log10(f_v / f_j)
    return uv, vj


def _sample_population(model, spec, n: int, seed: int) -> np.ndarray:
    """Draw n galaxy parameter samples and compute UVJ colors."""
    out = np.full((n, 2), np.nan)
    for i in range(n):
        params = spec.sample(jax.random.fold_in(jax.random.PRNGKey(seed), i))
        flux = np.asarray(model.predict_photometry(params))
        out[i] = _color(flux)
    return out


ssp = tengri.load_ssp()

# Rest-frame UVJ: Johnson U/V + 2MASS J. z=0.01 keeps colors rest-frame
# (1% shift) while avoiding D_L → 0 singularity at z=0.
obs = tengri.Observation(
    photometry=tengri.Photometry.from_names(["johnson_u", "johnson_v", "2mass_j"]),
)

# Star-forming population: ongoing SF, modest dust, broad SFH
sf_model = tengri.SEDModel.build(
    ssp,
    observation=obs,
    sfh={
        "type": "tsnorm",
        "all_params": tengri.FIXED,
        "log_total_mass": 10.0,
        "peak_lbt_gyr": tengri.Uniform(0.5, 4.0),
        "width_gyr": tengri.Uniform(1.0, 4.0),
        "skew": tengri.Uniform(-0.5, 1.0),
        "trunc": tengri.Uniform(2.0, 6.0),
        "logzsol": tengri.Uniform(-0.5, 0.2),
    },
    dust={
        "law": "power_law",
        "type": "two_component",
        "all_params": tengri.FIXED,
        "tau_bc": tengri.Uniform(0.1, 1.5),
        "tau_diff": tengri.Uniform(0.1, 1.0),
        "slope": -0.7,
    },
    redshift=tengri.Fixed(0.01),
)

# Passive population: old assembly, narrow burst, minimal dust
passive_model = tengri.SEDModel.build(
    ssp,
    observation=obs,
    sfh={
        "type": "tsnorm",
        "all_params": tengri.FIXED,
        "log_total_mass": 10.0,
        "peak_lbt_gyr": tengri.Uniform(7.0, 11.0),
        "width_gyr": tengri.Uniform(0.5, 1.5),
        "skew": tengri.Uniform(-1.5, 0.0),
        "trunc": tengri.Uniform(1.5, 3.0),
        "logzsol": tengri.Uniform(-0.2, 0.3),
    },
    dust={
        "law": "power_law",
        "type": "two_component",
        "all_params": tengri.FIXED,
        "tau_bc": tengri.Uniform(0.0, 0.15),
        "tau_diff": tengri.Uniform(0.0, 0.1),
        "slope": -0.7,
    },
    redshift=tengri.Fixed(0.01),
)

sf_uvj = _sample_population(sf_model, sf_model.spec, n=120, seed=0)
passive_uvj = _sample_population(passive_model, passive_model.spec, n=60, seed=1)

# --- Plot ---
fig, ax = plt.subplots(figsize=(8, 7))

# Williams+2009 quiescent wedge (z<1): (U-V)>1.3, (V-J)<1.6, (U-V)>0.88*(V-J)+0.49
vj_grid = np.linspace(-0.5, 1.6, 100)
uv_diag = 0.88 * vj_grid + 0.49
ax.plot(
    vj_grid,
    np.maximum(uv_diag, 1.3),
    color="0.25",
    lw=1.6,
    ls="--",
    label="Williams+2009 quiescent box",
)
ax.plot([1.6, 1.6], [0.88 * 1.6 + 0.49, 2.5], color="0.25", lw=1.6, ls="--")

ax.scatter(
    sf_uvj[:, 1],
    sf_uvj[:, 0],
    s=42,
    alpha=0.65,
    edgecolor="none",
    color="#1f77b4",
    label=f"Star-forming (n={(~np.isnan(sf_uvj[:, 0])).sum()})",
)
ax.scatter(
    passive_uvj[:, 1],
    passive_uvj[:, 0],
    s=58,
    alpha=0.85,
    marker="s",
    edgecolor="0.15",
    linewidth=0.6,
    color="#d62728",
    label=f"Passive (n={(~np.isnan(passive_uvj[:, 0])).sum()})",
)

ax.set_xlim(-0.5, 2.5)
ax.set_ylim(-0.2, 3.0)
ax.set_xlabel(r"$V - J$ [mag, rest-frame]")
ax.set_ylabel(r"$U - V$ [mag, rest-frame]")
ax.legend(frameon=False, loc="upper left")

# Annotate the regions
ax.text(0.6, 2.7, "Quiescent", fontsize=11, color="#a02020", ha="left")
ax.text(2.0, 0.4, "Dusty SF", fontsize=11, color="#666666", ha="center")
ax.text(0.0, 0.4, "Unobscured SF", fontsize=11, color="#1a4f8b", ha="left")

fig.tight_layout()
plt.savefig("plot_usecase_uvj_diagram.png", dpi=150, bbox_inches="tight")
