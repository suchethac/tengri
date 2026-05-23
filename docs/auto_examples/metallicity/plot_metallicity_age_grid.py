r"""
Stellar Continuum: Metallicity × Age Grid
==========================================

3×4 panel grid showing how the stellar continuum responds to metallicity
at three ages. Demonstrates the classic age-metallicity degeneracy: a
metal-rich young population can mimic a metal-poor old population in
the optical continuum.

.. sphx-glr-precomputed-img:

.. image:: images/sphx_glr_plot_metallicity_age_grid_001.png
   :alt: plot_metallicity_age_grid
   :class: sphx-glr-single-img

"""

import jax
import matplotlib.pyplot as plt
import numpy as np

from tengri import Fixed, SEDModel, load_ssp, recipes
from tengri.analysis.plotting import setup_style

setup_style()

ssp = load_ssp()
logz_grid = [-1.0, -0.3, 0.0, 0.3]
age_grid = [0.1, 1.0, 5.0]
colors = plt.cm.viridis(np.linspace(0.0, 0.85, len(age_grid)))

fig, axes = plt.subplots(len(age_grid), len(logz_grid), figsize=(14, 10))
fig.suptitle("Stellar Continuum: Metallicity × Age Degeneracy", fontsize=13, y=0.995)

for i, age in enumerate(age_grid):
    for j, logz in enumerate(logz_grid):
        ax = axes[i, j]
        # Peak the SFH at this age so it dominates the light; zero dust for clean continuum.
        recipe = recipes.dust_demo()
        recipe["sfh"].update(peak_lbt_gyr=age, width_gyr=0.3, skew=0.0, trunc=max(3.0, age + 2.0))
        recipe["dust"].update(tau_bc=0.0, tau_diff=0.0)
        recipe["redshift"] = Fixed(0.0)
        model = SEDModel.build(ssp_data=ssp, **recipe)

        params = model.spec.sample(jax.random.PRNGKey(0))
        params["met_logzsol"] = float(logz)
        pred = model.predict_rest_sed(params)
        wave = np.array(pred.wavelength)
        sed = np.array(pred.sed)

        # Normalize at 5500 Å
        i_norm = int(np.argmin(np.abs(wave - 5500.0)))
        sed_norm = sed / sed[i_norm] if sed[i_norm] > 0 else sed

        wave_um = wave / 1e4
        mask = (wave_um > 0.3) & (wave_um < 2.0) & (sed_norm > 0)
        ax.loglog(wave_um[mask], sed_norm[mask], color=colors[i], lw=2.0)
        ax.set(xlim=(0.3, 2.0), ylim=(0.1, 10))
        ax.tick_params(labelsize=8)
        if j == 0:
            ax.set_ylabel(f"Age = {age:.1f} Gyr\n" + r"$\lambda F_\lambda$", fontsize=9)
        if i == 0:
            ax.set_title(rf"$\log Z/Z_\odot$ = {logz:.1f}", fontsize=10, fontweight="bold")
        if i == len(age_grid) - 1:
            ax.set_xlabel(r"Wavelength [$\mu$m]", fontsize=9)
        else:
            ax.set_xticklabels([])

fig.tight_layout()
plt.savefig("plot_metallicity_age_grid.png", dpi=150, bbox_inches="tight")
plt.show()
