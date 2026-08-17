"""
AGN host-galaxy decomposition: disentangling Seyfert contributions
===================================================================

A Seyfert galaxy SED is decomposed photometrically by varying the AGN
contribution fraction ``agn_lum_ratio`` from 0 (pure host) to 1.0 (pure AGN)
to 0.5 (composite). how to isolate the AGN contribution
from the host galaxy using a single model and varying a structural
parameter — useful for diagnosing photometric AGN contamination.

Three traces shown:

- **Host only** (``agn_lum_ratio=0``): Star-forming galaxy, SFH + dust
- **AGN only** (``agn_lum_ratio=1.0``): Bolometric luminosity fixed
- **Composite** (``agn_lum_ratio=0.5``): 50–50 mix, physical decomposition

For real data, this three-trace pattern can be extended to a suite of
``agn_lum_ratio`` values to fit a composite Seyfert II SED photometrically.
The analytic scaling is ``L_total = (1 - frac) * L_host + frac * L_AGN``.

References
----------
.. [1] Ciesla et al. 2015, A&A, 576, A10 — Host-AGN decomposition via SED fitting
.. [2] Stalevski et al. 2017, MNRAS, 470, 3876 — IR torus models in composite SEDs
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import jax
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

C_AA_PER_S = 2.998e18

ssp = tengri.load_ssp()

# Shared model components: star-forming host + dust
COMMON = dict(
    sfh={
        "type": "dpl",
        "all_params": tengri.FIXED,
        "tau_gyr": 2.0,
        "log_total_mass": 10.0,
        "alpha": 1.5,
        "beta": 1.8,
    },
    dust={
        "type": "two_component",
        "all_params": tengri.FIXED,
        "law_bc": "calzetti",
        "tau_diff": 0.2,
        "tau_bc": 0.5,
        "emission": {"type": "dale2014", "all_params": tengri.FIXED},
    },
    redshift=tengri.Fixed(0.05),
)

# Base AGN configuration: Seyfert II with log_lbol = 11.5 (L_bol ~ 3e11 L_sun)
BASE_AGN = dict(
    disc={"type": "multicolor", "all_params": tengri.FIXED},
    torus={"type": "skirtor", "all_params": tengri.FIXED},
    log_lbol=11.5,
)


def _build_model(agn_lum_ratio):
    """Build a model with specified AGN fraction."""
    agn_config = {**BASE_AGN, "all_params": tengri.FIXED, "lum_ratio": agn_lum_ratio}
    model = tengri.SEDModel.build(ssp, agn=agn_config, **COMMON)
    p = dict(model.spec.sample(jax.random.PRNGKey(0)))
    out = model.predict(p)
    return np.asarray(model.wavelengths), np.asarray(out.rest_sed())


# Compute three traces: host-only, AGN-only, composite
traces = {}
for label, frac in [("Host only", 0.0), ("AGN only", 1.0), ("Composite (50-50)", 0.5)]:
    wave, sed = _build_model(frac)
    traces[label] = (wave, sed)

# Set up figure
fig, ax = plt.subplots(figsize=(7.4, 4.8))

# Plot the three traces
colors = {"Host only": "#2ca02c", "AGN only": "#d62728", "Composite (50-50)": "#1f77b4"}
wave_ref = traces["Host only"][0]
nu = C_AA_PER_S / wave_ref

for label in ["Host only", "AGN only", "Composite (50-50)"]:
    wave, sed = traces[label]
    ax.loglog(wave, nu * sed, color=colors[label], lw=1.8, label=label)

# Formatting
ax.set(
    xlabel=r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]",
    ylabel=r"$\nu L_\nu$  [erg s$^{-1}$]",
    xlim=(1000, 1e6),
    ylim=(1e42, 3e45),
)
ax.legend(frameon=False, fontsize=9, loc="lower left")
ax.grid(True, which="minor", alpha=0.2, linestyle=":")

plt.savefig("plot_agn_host_decomposition.png", dpi=150, bbox_inches="tight")
