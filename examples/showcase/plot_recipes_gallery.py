"""
Recipes for common science cases
=================================
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

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

# Load bare-stellar SSP (required by Cue nebular backend in 4 of 5 recipes)
BARE = tengri.load_ssp("fsps_prsc_miles_chabrier")

# Recipe configurations with display colors
RECIPES = [
    ("star_forming_photometry", recipes.star_forming_photometry, "#1f77b4"),
    ("quiescent_z0", recipes.quiescent_z0, "#d62728"),
    ("agn_panchromatic", recipes.agn_panchromatic, "#9467bd"),
    ("stochastic_sfh_jwst", recipes.stochastic_sfh_jwst, "#2ca02c"),
    ("mock_recovery_minimal", recipes.mock_recovery_minimal, "#ff7f0e"),
]

fig, ax = plt.subplots(figsize=(8, 5.2))

plotted = 0
first_failure: Exception | None = None

for name, recipe_fn, color in RECIPES:
    try:
        # Build model from recipe
        model = tengri.SEDModel.build(ssp_data=BARE, **recipe_fn())
    except Exception as e:
        # One recipe needing an SSP flavor BARE does not carry is a real skip;
        # all of them failing is a broken build path — see the guard below.
        if first_failure is None:
            first_failure = e
        continue

    # Sample at prior medians for canonical SED shape
    params = dict(model.spec.sample(jax.random.PRNGKey(0)))
    out = model.predict(params)

    # Convert to νL_ν for perceptual SED display
    wave = np.asarray(model.wavelengths)
    nu_l_nu = C_AA_PER_S / wave * np.asarray(out.rest_sed())

    ax.loglog(wave, nu_l_nu, color=color, lw=1.6, label=name)
    plotted += 1

if plotted == 0:
    raise RuntimeError(
        f"none of the {len(RECIPES)} recipes built, so this gallery is empty. "
        f"First failure: {type(first_failure).__name__}: {first_failure}"
    ) from first_failure

ax.set(
    xlim=(500, 1e7),
    ylim=(1e37, 1e46),
    xlabel=r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]",
    ylabel=r"$\nu L_\nu$ [erg s$^{-1}$]",
)
ax.legend(frameon=False, fontsize=8.5, loc="lower right")

fig.tight_layout()
plt.savefig("plot_recipes_gallery.png", dpi=150, bbox_inches="tight")
