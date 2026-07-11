"""
SSP grid edge behavior: clipping, extrapolation, NaN
=====================================================

Stellar population synthesis grids cover finite (age, metallicity) ranges.
This diagnostic probes what happens at boundaries: clip, extrapolate, or error?
We fix the SFH and vary stellar metallicity across the SSP grid boundary—inside,
at the edge, and beyond. The resulting SEDs reveal the interpolation behavior;
any NaN or error surfaces immediately in the plot.

Reference: SSP boundary handling is governed by DSPS interpolation (Hearin+ 2023).
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

ssp = tengri.load_ssp("fsps_prsc_miles_chabrier")
met_min = ssp.ssp_lgmet.min()
met_max = ssp.ssp_lgmet.max()

# Model with DPL SFH, fixed dust, Cue nebular
model = tengri.SEDModel.build(
    ssp,
    sfh={
        "type": "dpl",
        "*": tengri.FIXED,
        "alpha": 2.0,
        "beta": 2.0,
        "tau_gyr": 2.0,
        "log_total_mass": 10.0,
    },
    dust={"type": "two_component", "*": tengri.FIXED, "tau_diff": 0.15, "tau_bc": 0.25},
    neb={"type": "cue", "*": tengri.FIXED},
    redshift=tengri.Fixed(0.0),
)

# Baseline parameters
baseline = dict(model.spec.sample(jax.random.PRNGKey(42)))

# Probe metallicity grid: inside, at boundary, outside
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
met_values = [
    met_min + 0.3,  # Well inside
    met_max - 0.1,  # Near max
    met_max,  # Exactly at max
    met_max + 0.2,  # Outside grid
]
titles = [
    f"Met = {met_values[0]:.3f}\n(inside grid)",
    f"Met = {met_values[1]:.3f}\n(near boundary)",
    f"Met = {met_values[2]:.3f}\n(at boundary)",
    f"Met = {met_values[3]:.3f}\n(outside grid)",
]

for ax, met, _title in zip(axes.flat, met_values, titles):
    params = {**baseline, "met_logzsol": jnp.float64(met)}

    try:
        out = model.predict_rest_sed(params)
        wave = np.asarray(out.wavelength)
        sed = np.asarray(out.sed)
        ok = np.isfinite(sed)

        if not ok.any():
            ax.text(
                0.5,
                0.5,
                "All NaN",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
                color="red",
                weight="bold",
            )
        else:
            wave, sed = wave[ok], sed[ok]
            nu = 2.998e18 / wave
            ax.loglog(wave, nu * sed, color="C0", lw=2)
            ax.set_xlabel(r"$\lambda$ [$\mathrm{\AA}$]")
            ax.set_ylabel(r"$\nu L_\nu$ [erg s$^{-1}$]")
            ax.grid(True, alpha=0.3)

    except Exception as e:
        msg = type(e).__name__
        ax.text(
            0.5,
            0.5,
            f"Error:\n{msg}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
            color="red",
        )

plt.tight_layout()
plt.savefig("plot_diag_ssp_grid_edge_behavior.png", dpi=150, bbox_inches="tight")
