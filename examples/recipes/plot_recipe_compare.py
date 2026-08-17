"""
What each shipped tengri recipe produces
==========================================

Six curated recipes span galaxy populations: star-forming at 0–6 (bare-stellar
SSP), quiescent at z ≈ 0.05 (bare-stellar, τ_diff-free to trace dust),
AGN panchromatic (bare-stellar, full AGN composite with disc+torus+radio+xray),
stochastic JWST high-z with burstiness (bare-stellar, DPL+field at 0.5–12),
mock-recovery minimal (any SSP, 4–5 free params for benchmarking), and
dust-demo (wNE only — baked nebular emission visualized). All use WavePrecomp()
except photoz (ztable does not cover z > 12). Use ``load_ssp("*.wNE")`` only
for dust_demo; others silently under-predict if fed wNE.
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

for name, recipe_fn, color in RECIPE_FNS:
    try:
        model = tengri.SEDModel.build(ssp_data=SSPS[name], **recipe_fn())
    except Exception:
        continue
    p = dict(model.spec.sample(jax.random.PRNGKey(0)))
    out = model.predict(p)
    wave = np.asarray(model.wavelengths)
    nu_l_nu = C_AA_PER_S / wave * np.asarray(out.rest_sed())
    ax.loglog(wave, nu_l_nu, color=color, lw=1.4, label=name)

ax.set(
    xlim=(700, 5e6),
    ylim=(1e38, 5e45),
    xlabel=r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]",
    ylabel=r"$\nu L_\nu$  [erg s$^{-1}$]",
)
ax.legend(frameon=False, fontsize=8, loc="lower right")

fig.tight_layout()
plt.savefig("plot_recipe_compare.png", dpi=150, bbox_inches="tight")
