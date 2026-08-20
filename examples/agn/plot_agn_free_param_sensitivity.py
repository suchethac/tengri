"""
AGN parameters are free-able now — and every one moves the SED
==============================================================

Until recently the ``agn_*`` parameters were declared with *fixed* defaults and
no prior range, so the build grammar's ``FREE`` controls
(``agn={'all_params': FREE}``, ``recipes.agn_panchromatic()``) silently resolved every
AGN parameter to a constant — a fit would freeze the entire AGN sector with no
error. The registry now gives each parameter a physically-motivated
``Uniform``/``LogUniform`` prior (Nenkova+2008, Kubota & Done 2018,
Stalevski+2016 grid extents), so ``FREE`` actually frees them.

A group-level ``agn={'all_params': FREE}`` is **block-scoped**: it frees only the
parameters the active disc / torus / lines blocks actually consume, not the
full declared superset — so you never get unconstrained no-op nuisance
dimensions for parameters belonging to inactive blocks.

This example builds one composable AGN (multicolor disc + Nenkova clumpy torus
+ NLR lines), prints the now-non-empty free-parameter set, and sweeps three of
those parameters across their priors. Each panel shows the parameter visibly
reshaping the SED — the visual form of the "no silent no-ops" contract checked
by ``tests/contract/test_agn_block_consumes.py``.

References: Nenkova et al. 2008 (CLUMPY torus); Kubota & Done 2018 (multicolor
disc); Feltre et al. 2016 (NLR grid).
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import jax
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.plot import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

C_AA_PER_S = 2.998e18

SFH = {"type": "const", "all_params": tengri.FIXED, "log_total_mass": -10.0}
DUST = {
    "law": "power_law",
    "type": "two_component",
    "all_params": tengri.FIXED,
    "tau_diff": 0.0,
    "tau_bc": 0.0,
}
# Bare type selectors: no per-block ``'all_params'``, so the *top-level* AGN
# wildcard governs every block's parameters. (A sub-block's own ``'all_params'``
# would override
# the top-level one for that block — a useful nuance, but it would hide the
# point here.)
BLOCKS = {
    "disc": {"type": "multicolor"},
    "torus": {"type": "nenkova"},
    "nlr": {"type": "analytic"},
    "blr": {"type": "none"},
}

ssp = tengri.load_ssp()

# %%
# The headline: ``'all_params': FREE`` now frees the AGN parameters the active
# blocks consume. Before the registry fix this set was empty.
agn_free = {"all_params": tengri.FREE, "log_lbol": 12.0, "lum_ratio": 1.0, **BLOCKS}
model_free = tengri.SEDModel.build(
    ssp, sfh=SFH, dust_attenuation=DUST, agn=agn_free, redshift=tengri.Fixed(0.0)
)
free_agn = [p for p in model_free.spec.free_params if p.startswith("agn_")]
print(f"AGN parameters freed by agn={{'all_params': FREE}} (block-scoped): {sorted(free_agn)}")

# %%
# Sweep three consumed parameters across their priors. We build the model once
# with the AGN sector held fixed at its defaults, then override one parameter at
# a time in the prediction dict — a clean, deterministic parameter sweep.
agn_fixed = {"all_params": tengri.FIXED, "log_lbol": 12.0, "lum_ratio": 1.0, **BLOCKS}
model = tengri.SEDModel.build(ssp, sfh=SFH, dust_attenuation=DUST, agn=agn_fixed, redshift=tengri.Fixed(0.0))
base = dict(model.spec.sample(jax.random.PRNGKey(0)))

SWEEPS = [
    ("agn_log_mbh", np.linspace(6.5, 9.5, 5), r"$\log_{10}(M_{\rm BH}/M_\odot)$"),
    ("agn_log_ledd", np.linspace(-1.8, 0.3, 5), r"$\log_{10}(L/L_{\rm Edd})$"),
    ("agn_tau", np.linspace(10.0, 140.0, 5), r"Nenkova torus $\tau_{9.7}$"),
]

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharex=True, sharey=True)
for ax, (param, values, label) in zip(axes, SWEEPS):
    cmap = plt.cm.viridis(np.linspace(0.15, 0.9, len(values)))
    for v, color in zip(values, cmap):
        out = model.predict({**base, param: float(v)})
        wave_um = np.asarray(model.wavelengths) * 1.0e-4
        nu_lnu = C_AA_PER_S / np.asarray(model.wavelengths) * np.asarray(out.rest_sed())
        ax.loglog(
            wave_um,
            np.where(nu_lnu > 0, nu_lnu, np.nan),
            lw=1.5,
            color=color,
            label=f"{v:.1f}",
        )
    ax.set(
        xlim=(5.0e-3, 1.0e3),
        ylim=(1.0e43, 1.0e47),
        xlabel=r"Rest-frame wavelength [$\mu$m]",
        title=label,
    )
    ax.legend(loc="lower center", fontsize=8, frameon=False, ncol=2)

axes[0].set_ylabel(r"$\nu L_\nu$  [erg s$^{-1}$]")
fig.suptitle("Each freed AGN parameter visibly reshapes the SED (no silent no-ops)")
fig.tight_layout()
plt.savefig("plot_agn_free_param_sensitivity.png", dpi=150, bbox_inches="tight")
