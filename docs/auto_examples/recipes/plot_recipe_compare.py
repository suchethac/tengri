"""
What each shipped tengri recipe produces
==========================================

``tengri.recipes`` ships several curated starting-point model configs
that map common astronomer use-cases onto the nested-dict ``SEDModel.build``
grammar. This card overlays the rest-frame SED of every shipped recipe
so users can pick by eye:

- ``star_forming_photometry`` — broadband photometric fit of a typical SF galaxy
- ``quiescent_z0``           — old red passively-evolving system
- ``agn_panchromatic``       — composable AGN + host
- ``stochastic_sfh_jwst``    — IFT correlated-field SFH for high-z JWST data
- ``mock_recovery_minimal``  — smallest model usable for parameter-recovery tests
- ``dust_demo``              — emphasis on dust attenuation + emission

Each is built with no overrides and evaluated at default parameter values.
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

BARE = tengri.load_ssp("fsps_prsc_miles_chabrier")
SSPS = {
    "star_forming_photometry": BARE,
    "quiescent_z0": BARE,
    "agn_panchromatic": BARE,
    "stochastic_sfh_jwst": BARE,
    "mock_recovery_minimal": BARE,
    "dust_demo": BARE,
}
RECIPE_FNS = [
    ("star_forming_photometry", recipes.star_forming_photometry, "#3377cc"),
    ("quiescent_z0", recipes.quiescent_z0, "#cc3333"),
    ("agn_panchromatic", recipes.agn_panchromatic, "#9933aa"),
    ("stochastic_sfh_jwst", recipes.stochastic_sfh_jwst, "#33aa55"),
    ("mock_recovery_minimal", recipes.mock_recovery_minimal, "#aa8833"),
    ("dust_demo", recipes.dust_demo, "#666666"),
]

fig, ax = plt.subplots(figsize=(7.4, 4.8))

plotted = 0
first_failure: Exception | None = None

for name, recipe_fn, color in RECIPE_FNS:
    try:
        model = tengri.SEDModel.build(ssp_data=SSPS[name], **recipe_fn())
    except Exception as e:
        # A recipe that needs an SSP flavor this box does not have is a real
        # skip. Every recipe failing is not — see the guard below.
        if first_failure is None:
            first_failure = e
        continue
    p = dict(model.spec.sample(jax.random.PRNGKey(0)))
    out = model.predict(p)
    wave = np.asarray(model.wavelengths)
    nu_l_nu = C_AA_PER_S / wave * np.asarray(out.rest_sed())
    ax.loglog(wave, nu_l_nu, color=color, lw=1.4, label=name)
    plotted += 1

# Without this, a build break in every recipe renders an empty axes and the
# gallery runner reports the example as passing -- the exact hole the runner
# exists to close (#1145). Swallowing per-recipe is fine; swallowing all of
# them means the figure shows nothing it claims to compare.
if plotted == 0:
    raise RuntimeError(
        f"none of the {len(RECIPE_FNS)} recipes built, so this comparison is "
        f"empty. First failure: {type(first_failure).__name__}: {first_failure}"
    ) from first_failure

ax.set(
    xlim=(700, 5e6),
    ylim=(1e38, 5e45),
    xlabel=r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]",
    ylabel=r"$\nu L_\nu$  [erg s$^{-1}$]",
)
ax.legend(frameon=False, fontsize=8, loc="lower right")

fig.tight_layout()
plt.savefig("plot_recipe_compare.png", dpi=150, bbox_inches="tight")
