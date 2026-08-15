"""
Recipe introspection and SED morphology comparison
===================================================

Call ``tengri.list_recipes()`` to see the shipped menu with SSP requirements
(bare-stellar, wNE, or any) and ``tengri.describe_recipe(name)`` to fetch
a recipe's docstring. Three models showcase the morphological diversity:
star-forming (DPL+Cue nebular, free z to 6), quiescent at z=0.05 (dexp,
lower dust ceiling), and AGN-panchromatic (full composite, z to 6).
All require bare-stellar SSP (Cue backend).
"""

import io
import warnings

import jax
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri import recipes
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")
warnings.filterwarnings("ignore", message=".*wNE.*")

C_AA_PER_S = 2.998e18

# Load bare-stellar SSP (required for Cue nebular backend in all recipes)
ssp = tengri.load_ssp("fsps_prsc_miles_chabrier")

# Define the three recipes to showcase
RECIPE_CONFIGS = [
    ("star_forming_photometry", recipes.star_forming_photometry, "#3377cc"),
    ("quiescent_z0", recipes.quiescent_z0, "#cc3333"),
    ("agn_panchromatic", recipes.agn_panchromatic, "#9933aa"),
]

# Compute SEDs
fig, (ax_menu, ax_seds) = plt.subplots(
    2, 1, figsize=(9.5, 7.0), gridspec_kw={"height_ratios": [1, 1.6]}
)

# Left panel: render the recipe menu
recipe_table = tengri.list_recipes()
table_str = io.StringIO()
table_str.write(str(recipe_table))
menu_text = table_str.getvalue()

ax_menu.axis("off")
ax_menu.text(
    0.05,
    0.95,
    menu_text,
    transform=ax_menu.transAxes,
    fontfamily="monospace",
    fontsize=7,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
)

# Right panel: overlay rest-frame SEDs
for name, recipe_fn, color in RECIPE_CONFIGS:
    try:
        model = tengri.SEDModel.build(ssp_data=ssp, **recipe_fn())
    except Exception:
        continue

    p = dict(model.spec.sample(jax.random.PRNGKey(0)))
    out = model.predict(p)
    wave = np.asarray(model.wavelengths)
    nu_l_nu = C_AA_PER_S / wave * np.asarray(out.rest_sed())
    ax_seds.loglog(wave, nu_l_nu, color=color, lw=1.4, label=name)

ax_seds.set(
    xlim=(700, 5e6),
    ylim=(1e38, 5e45),
    xlabel=r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]",
    ylabel=r"$\nu L_\nu$  [erg s$^{-1}$]",
)
ax_seds.legend(frameon=False, fontsize=9, loc="lower right")
ax_seds.grid(True, alpha=0.2, which="both")

fig.tight_layout()
plt.savefig("plot_recipe_introspection_tour.png", dpi=150, bbox_inches="tight")
