"""
SKIRTOR torus: AGNfitter-averaged vs. X-CIGALE full grid
=========================================================

SKIRTOR (Stalevski et al. 2016) is a clumpy radiative transfer torus model
with a three-dimensional parameter space (half-opening angle, inclination,
optical depth). Two different implementations exist in tengri:

1. **X-CIGALE faithful** (component type "skirtor"): Uses the full
   pre-computed SKIRTOR grid with all ~25,000 templates, achieving
   maximum fidelity to the original grid-based library.

2. **AGNfitter-averaged** (component type "skirtor_agnfitter"): Uses the
   AGNfitter-rX averaged sub-grid with only ~27 templates, obtained by
   marginalizing over clumpiness (p, q) and radial index before storing.

The averaged variant is computationally faster but produces slightly
different SED shapes, especially in the mid-infrared. At fixed geometry
and optical depth, the full grid often peaks redder (cooler dust) than
the averaged grid, reflecting averaging over clumpy configurations.

This example compares both implementations at matched parameters to
visualize the difference, demonstrating the trade-off between fidelity
and speed in torus library selection.

References
----------
.. [1] M. Stalevski et al., "3D radiative transfer modeling of the dusty
   torus around AGN — the influence of clumping," MNRAS 420, 2756 (2012).
   arXiv:1109.1286.
.. [2] M. Stalevski et al., "The dust covering factor in AGN — combining the
   IR torus emission with polar dust component," MNRAS 458, 2288 (2016).
   arXiv:1602.01954.
.. [3] L. N. Martínez-Ramírez et al., "AGNfitter-rx: Modeling the
   radio-to-X-ray spectral energy distributions of AGNs," A&A 688, A46
   (2024). arXiv:2405.12111.
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
warnings.filterwarnings("ignore", message=".*deprecated.*")

ssp = tengri.load_ssp()

# Minimal host
SFH = {"type": "const", "all_params": tengri.FIXED, "log_total_mass": -10.0}
DUST = {"type": "two_component", "all_params": tengri.FIXED, "tau_diff": 0.0, "tau_bc": 0.0}

# Matched AGN parameters
log_lbol = 12.0
agn_lum_ratio = 0.6

c_aa_s = 2.998e18

fig, ax = plt.subplots(figsize=(7.5, 4.8))

# Note: Attempting to build with "skirtor_agnfitter" may fail if the
# component is not registered. In that case, the example guards with a
# try-except and still compiles successfully; the reader can manually
# test the variant when the component is available.

torus_models = []
first_failure: Exception | None = None
try:
    model_cigale = tengri.SEDModel.build(
        ssp,
        sfh=SFH,
        dust=DUST,
        agn={
            "type": "composable",
            "disc": {"type": "powerlaw", "all_params": tengri.FIXED},
            "torus": {"type": "skirtor", "all_params": tengri.FIXED},
            "all_params": tengri.FIXED,
            "log_lbol": log_lbol,
            "lum_ratio": agn_lum_ratio,
            "oa_skirtor": 30.0,  # half-opening angle [deg]
            "incl_skirtor": 45.0,  # inclination [deg]
            "tv_skirtor": 6.0,  # optical depth at 9.7 micron
        },
        redshift=tengri.Fixed(0.05),
    )
    torus_models.append(("skirtor (X-CIGALE full grid)", model_cigale, "C0"))
except Exception as e:
    first_failure = e
    print(f"Warning: X-CIGALE model unavailable ({e})")

try:
    model_agnfitter = tengri.SEDModel.build(
        ssp,
        sfh=SFH,
        dust=DUST,
        agn={
            "type": "composable",
            "disc": {"type": "powerlaw", "all_params": tengri.FIXED},
            "torus": {"type": "skirtor_agnfitter", "all_params": tengri.FIXED},
            "all_params": tengri.FIXED,
            "log_lbol": log_lbol,
            "lum_ratio": agn_lum_ratio,
            "oa_skirtor": 30.0,
            "incl_skirtor": 45.0,
            "tv_skirtor": 6.0,
        },
        redshift=tengri.Fixed(0.05),
    )
    torus_models.append(("skirtor_agnfitter (averaged grid)", model_agnfitter, "C1"))
except Exception as e:
    if first_failure is None:
        first_failure = e
    print(f"Warning: AGNfitter-averaged model unavailable ({e})")

# The whole point of this example is the *comparison*. One side missing is worth
# a warning; both missing leaves a figure with annotations and no SEDs, which
# the runner would otherwise score as a pass.
if not torus_models:
    raise RuntimeError(
        f"neither torus model built, so there is nothing to compare. First "
        f"failure: {type(first_failure).__name__}: {first_failure}"
    ) from first_failure

# Plot each model
for label, model, color in torus_models:
    p = dict(model.spec.sample(jax.random.PRNGKey(0)))
    out = model.predict(p)

    wave = np.asarray(model.wavelengths)
    nu_l_nu = c_aa_s / wave * np.asarray(out.rest_sed())

    ax.loglog(wave, nu_l_nu, color=color, lw=2.0, label=label, alpha=0.8)

ax.set_xlim(100, 3e5)
ax.set_ylim(1e41, 1e48)
ax.set_xlabel(r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]")
ax.set_ylabel(r"$\nu L_\nu$  [erg s$^{-1}$]")

# Annotations
ax.axvspan(1e4, 3e4, color="brown", alpha=0.08, lw=0)
ax.text(1.5e4, 5e47, "Mid-IR torus", color="brown", fontsize=8, weight="bold")

ax.legend(frameon=False, fontsize=9, loc="lower left")
ax.grid(True, which="major", alpha=0.2)

fig.tight_layout()
plt.savefig("plot_skirtor_agnfitter_vs_cigale.png", dpi=150, bbox_inches="tight")
