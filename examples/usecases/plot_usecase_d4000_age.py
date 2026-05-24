"""
The 4000 Å break as a stellar age proxy
=========================================

The 4000 Å break ``D_n(4000)`` — Bruzual 1983, Balogh+1999 — measures
the discontinuity around 4000 Å produced by the line-blanketing of
ionised metals in the atmospheres of old stars. It rises monotonically
with the mass-weighted age of the stellar population and is one of
the most widely used age indicators in SDSS-style optical-only data
(Kauffmann+2003).

We build single-burst SSPs at a grid of ages from 30 Myr to 13 Gyr,
measure D_n(4000) in the canonical Balogh+1999 windows (3850–3950 Å
numerator and 4000–4100 Å denominator on F_ν), and overlay the
observational SDSS quenched-galaxy locus.

References:
- Bruzual 1983, ApJ, 273, 105
- Balogh, Morris, Yee, Carlberg & Ellingson 1999, ApJ, 527, 54
- Kauffmann et al. 2003, MNRAS, 341, 33
"""

import warnings

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

# Balogh+1999 windows on F_ν (Å)
BLUE_LO, BLUE_HI = 3850.0, 3950.0
RED_LO, RED_HI = 4000.0, 4100.0


def _d4000(wave_aa: np.ndarray, l_nu: np.ndarray) -> float:
    """Narrow Balogh+1999 D_n(4000) on F_ν."""
    blue = (wave_aa >= BLUE_LO) & (wave_aa <= BLUE_HI)
    red = (wave_aa >= RED_LO) & (wave_aa <= RED_HI)
    return float(np.mean(l_nu[red]) / np.mean(l_nu[blue]))


ssp = tengri.load_ssp()
model = tengri.SEDModel.build(
    ssp,
    # Narrow truncated-skew-normal burst (50 Myr 1σ), sliding in lookback
    # time → approximates a single-age stellar population for the D_n(4000)
    # measurement.
    sfh={
        "type": "tsnorm",
        "*": tengri.FIXED,
        "peak_lbt_gyr": tengri.Uniform(0.05, 13.0),
        "width_gyr": 0.10,
        "skew": 0.0,
        "trunc": 13.5,
        "log_peak_sfr": 1.0,
    },
    dust={"type": "two_component", "*": tengri.FIXED, "tau_diff": 0.0, "tau_bc": 0.0},
    redshift=tengri.Fixed(0.0),
)
baseline = dict(model.spec.sample(jax.random.PRNGKey(0)))

age_gyr = np.geomspace(0.05, 11.0, 24)
d4000_grid = np.empty_like(age_gyr)

for i, age in enumerate(age_gyr):
    out = model.predict_rest_sed(
        {
            **baseline,
            "sfh_tsnorm_peak_lbt_gyr": jnp.float64(age),
        }
    )
    d4000_grid[i] = _d4000(np.asarray(out.wavelength), np.asarray(out.sed))

# Schematic SDSS quenched-galaxy locus (Kauffmann+2003 figure 6, hand-traced).
sdss_age = np.array([0.1, 0.3, 1.0, 3.0, 6.0, 10.0])
sdss_d4000 = np.array([1.10, 1.20, 1.45, 1.75, 1.92, 2.00])

fig, ax = plt.subplots(figsize=(6.5, 4.4))
ax.plot(age_gyr, d4000_grid, color="C0", lw=1.6, label="tengri SSP (single burst)")
ax.plot(
    sdss_age,
    sdss_d4000,
    "s",
    color="0.4",
    ms=6,
    mfc="white",
    mew=1.0,
    label="SDSS quenched locus (schematic)",
)
ax.axhspan(1.5, 1.6, color="0.92", alpha=0.6, lw=0)
ax.text(11.5, 1.55, "green-valley cut\n(Kauffmann+2003)", fontsize=8, color="0.35", ha="right")
ax.set_xscale("log")
ax.set_xlim(0.02, 14.0)
ax.set_ylim(1.05, 2.25)
ax.set_xlabel(r"Stellar population age [Gyr]")
ax.set_ylabel(r"$D_n(4000)$  [F$_\nu$, Balogh+1999]")
ax.legend(frameon=False, fontsize=9, loc="lower right")

fig.tight_layout()
fig.savefig("plot_usecase_d4000_age.png", dpi=150, bbox_inches="tight")
