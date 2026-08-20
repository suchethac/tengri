"""
Fe II pseudo-continuum strength evolution
==========================================

Iron pseudo-continuum (Fe II) emission in AGN produces characteristic humps
in the near-UV and optical bands. The strength and shape are governed by
the Fe II equivalent width and ionization state, parameterized in tengri
by the ``agn_fe2_strength`` parameter relative to H-beta (Balmer lines).

At strength = 0, Fe II is suppressed entirely. As strength increases to
~0.5–2.0 (realistic quasar values), the continuum rises in the
2200–3000 Å and 4400–4700 Å humps. This example sweeps ``agn_fe2_strength``
from 0 to 1.5 to show the progressive Fe II contribution to the overall
AGN continuum in isolation (no host galaxy).

The Fe II strength is one of the composable AGN blocks; it is evaluated
in the "lines" stage after the accretion disc continuum but modulates
the observed SED via overlapping broad emission-line pseudo-continuum.

References
----------
.. [1] I. M. McHardy et al., "An origin of the X-ray and UV/optical
   correlations in active galactic nuclei," Nature 444, 730–732 (2006).
.. [2] M. Vestergaard & B. M. Peterson, "Determining black-hole masses
   in active galactic nuclei," ApJ 641, 689–709 (2006).
   arXiv:astro-ph/0601042.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.plot import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")
warnings.filterwarnings("ignore", message=".*deprecated.*")

ssp = tengri.load_ssp()

# Minimal host: suppress stellar emission, focus on AGN continuum
SFH = {"type": "const", "all_params": tengri.FIXED, "log_total_mass": -10.0}
DUST = {
    "law": "power_law",
    "type": "two_component",
    "all_params": tengri.FIXED,
    "tau_diff": 0.0,
    "tau_bc": 0.0,
}

# Fe II strength sweep: 0 (none) → 1.5 (strong)
fe2_strength_values = np.linspace(0.0, 1.5, 6)

# Build model with composable AGN: multicolor disc + BLR (with Fe II)
model = tengri.SEDModel.build(
    ssp,
    sfh=SFH,
    dust=DUST,
    agn={
        "type": "composable",
        "disc": {"type": "multicolor", "all_params": tengri.FIXED},
        "blr": {"type": "analytic", "all_params": tengri.FIXED, "agn_blr_cf": 0.1},
        "all_params": tengri.FIXED,
        "log_lbol": 12.0,
        "log_ledd": -1.0,
        "lum_ratio": 1.0,
    },
    redshift=tengri.Fixed(0.05),
)

baseline = dict(model.spec.sample(jax.random.PRNGKey(0)))

# Colormap for Fe II strength
norm = mpl.colors.Normalize(vmin=fe2_strength_values.min(), vmax=fe2_strength_values.max())
cmap = plt.get_cmap("plasma")

# The Fe II pseudo-continuum is a few-percent additive feature riding on the
# bright accretion-disc continuum, so the sweep is invisible on a full-SED log
# axis. We therefore show two panels: the full AGN SED for context (left) and
# the Fe II *excess* over the strength = 0 baseline (right), which isolates the
# two iron humps and makes the sweep legible.
fig, (ax, ax_excess) = plt.subplots(1, 2, figsize=(11.0, 4.5))
c_aa_s = 2.998e18

seds = []
for fe2_strength in fe2_strength_values:
    params = {**baseline, "agn_fe2_strength": jnp.float64(fe2_strength)}
    out = model.predict(params)

    wave = np.asarray(model.wavelengths)
    nu_l_nu = c_aa_s / wave * np.asarray(out.rest_sed())
    seds.append(nu_l_nu)

    color = cmap(norm(fe2_strength))
    ax.loglog(wave, nu_l_nu, color=color, lw=1.5, label=f"Fe II strength = {fe2_strength:.2f}")

base_sed = seds[0]  # strength = 0 reference

# Left: full AGN SED (multicolor disc + BLR), zoomed to where the data lives.
ax.set_xlim(500, 1e5)
ax.set_ylim(1e43, 3e45)
ax.set_xlabel(r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]")
ax.set_ylabel(r"$\nu L_\nu$  [erg s$^{-1}$]")
ax.axvspan(2200, 3000, color="red", alpha=0.1, lw=0)
ax.axvspan(4400, 4700, color="red", alpha=0.1, lw=0)
ax.set_title("Full AGN SED", fontsize=10)
ax.legend(frameon=False, fontsize=8, loc="lower left")

# Right: Fe II excess over the strength = 0 baseline — the two iron humps.
for nu_l_nu, fe2_strength in zip(seds, fe2_strength_values):
    if fe2_strength == 0.0:
        continue
    ax_excess.plot(
        wave,
        nu_l_nu - base_sed,
        color=cmap(norm(fe2_strength)),
        lw=1.5,
        label=f"{fe2_strength:.2f}",
    )
ax_excess.set_xscale("log")
ax_excess.set_xlim(1500, 8000)
ax_excess.axvspan(2200, 3000, color="red", alpha=0.1, lw=0)
ax_excess.axvspan(4400, 4700, color="red", alpha=0.1, lw=0)
ax_excess.set_xlabel(r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]")
ax_excess.set_ylabel(r"$\nu\,\Delta L_\nu$ over Fe II $=0$  [erg s$^{-1}$]")
ax_excess.set_title("Fe II excess", fontsize=10)
ax_excess.annotate("UV\nFe II", (2600, 0), color="red", fontsize=8, ha="center", weight="bold")
ax_excess.annotate(
    "optical\nFe II", (4550, 0), color="red", fontsize=8, ha="center", weight="bold"
)
ax_excess.legend(title="Fe II strength", frameon=False, fontsize=8, loc="upper right")

fig.tight_layout()
plt.show()
