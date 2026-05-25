"""
Double power-law SFH parameter space: early growth α vs late quenching β
=====================================================================

A 3×3 grid showing how the rising slope α (columns) and falling slope β (rows)
together control the full SFH morphology. Early-time α determines assembly
speed; late-time β sets the post-peak decay. The optical SED responds across
each cell, revealing how parameter space maps to stellar age.
"""

import warnings

import jax
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

ssp = tengri.load_ssp()

# Grid: alpha (rising slope, columns) and beta (falling slope, rows)
alphas = [0.5, 1.5, 3.0]
betas = [0.5, 1.5, 3.0]

fig, axes = plt.subplots(3, 3, figsize=(12, 10))

baseline = dict(
    tau_gyr=3.0,
    log_peak_sfr=1.0,
)

for i, beta in enumerate(betas):
    for j, alpha in enumerate(alphas):
        ax = axes[i, j]

        model = tengri.SEDModel.build(
            ssp,
            sfh={
                "type": "dpl",
                "*": tengri.FIXED,
                "alpha": alpha,
                "beta": beta,
                "tau_gyr": baseline["tau_gyr"],
                "log_peak_sfr": baseline["log_peak_sfr"],
            },
            dust={"type": "two_component", "*": tengri.FIXED, "tau_diff": 0.2, "tau_bc": 0.3},
            redshift=tengri.Fixed(0.1),
        )

        baseline_params = dict(model.spec.sample(jax.random.PRNGKey(0)))
        params_eval = {**baseline_params}

        out = model.predict_rest_sed(params_eval)
        wave = np.asarray(out.wavelength)
        sed = np.asarray(out.sed)

        # Optical region
        mask = (wave > 4000) & (wave < 8000)

        ax.plot(
            wave[mask],
            sed[mask],
            "C0-",
            lw=2.0,
        )
        ax.set_xlabel(r"Wavelength [$\AA$]", fontsize=9)
        ax.set_ylabel(r"$L_\nu$ [erg/s/Hz]", fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=8)

fig.tight_layout()
plt.savefig("plot_dpl_alpha_beta_grid.png", dpi=150, bbox_inches="tight")
