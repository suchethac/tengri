"""
Intrinsic UV continuum slope β vs single-burst age
====================================================

Dust-free UV continuum slope of an SSP swept from 10 Myr to 1 Gyr.
β is fit in the Calzetti+1994 windows (1268–2580 Å) to ``F_λ ∝ λ^β``.

The fast drop from β ≈ −2.5 at 10 Myr to ≈ −0.5 by 100 Myr is the
intrinsic clock used to interpret observed UV slopes: a galaxy with
β ≈ −1 either has a ~100 Myr stellar population (no dust) or a
younger one with moderate reddening (the classic IRX–β degeneracy
that ``plot_usecase_uv_slope_beta`` recovers).
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

C_AA_PER_S = 2.998e18

# Calzetti+1994 windows for the UV slope fit.
WINDOWS = np.array(
    [
        [1268, 1284],
        [1309, 1316],
        [1342, 1371],
        [1407, 1515],
        [1562, 1583],
        [1677, 1740],
        [1760, 1833],
        [1866, 1890],
        [1930, 1950],
        [2400, 2580],
    ]
)


def _beta_uv(wave, l_nu):
    f_lam = l_nu * C_AA_PER_S / wave**2
    mask = np.zeros_like(wave, dtype=bool)
    for lo, hi in WINDOWS:
        mask |= (wave >= lo) & (wave <= hi)
    slope, _ = np.polyfit(np.log10(wave[mask]), np.log10(f_lam[mask]), 1)
    return float(slope)


model = tengri.SEDModel.build(
    tengri.load_ssp(),
    sfh={
        "type": "tsnorm",
        "all_params": tengri.FIXED,
        "peak_lbt_gyr": tengri.Uniform(0.01, 13.0),
        "width_gyr": 0.05,
        "log_total_mass": 10.0,
        "skew": 0.0,
        "trunc": 13.0,
    },
    dust={"type": "two_component", "all_params": tengri.FIXED, "tau_diff": 0.0, "tau_bc": 0.0},
    redshift=tengri.Fixed(0.01),
)
baseline = dict(model.spec.sample(jax.random.PRNGKey(0)))

ages = np.geomspace(0.01, 1.0, 24)
beta = np.empty_like(ages)
for i, age in enumerate(ages):
    p = {**baseline, "sfh_tsnorm_peak_lbt_gyr": jnp.float64(age)}
    out = model.predict(p)
    beta[i] = _beta_uv(np.asarray(model.wavelengths), np.asarray(out.rest_sed()))

fig, ax = plt.subplots(figsize=(6.4, 4.4))
ax.plot(ages * 1.0e3, beta, color="C0", lw=1.6)
ax.axhline(-2.5, color="0.55", lw=0.5, ls=":")
ax.text(11, -2.48, "young-starburst floor β ≈ -2.5", color="0.4", fontsize=8, va="bottom")
ax.set(
    xscale="log",
    xlim=(10, 1000),
    xlabel="Stellar burst age  [Myr]",
    ylabel=r"UV continuum slope  $\beta$  (Calzetti+1994 windows)",
)

fig.tight_layout()
plt.savefig("plot_uv_slope_age.png", dpi=150, bbox_inches="tight")
