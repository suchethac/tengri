"""
Black-hole spin hardens the UV slope through ISCO migration
============================================================

In a relativistic accretion-disc model the inner boundary sits at the
innermost stable circular orbit (ISCO). Higher spin shrinks the ISCO,
raises the inner-disc temperature, and shifts disc power blueward — the
UV spectral slope alpha (L_nu ~ nu^alpha across 912 to 3000 Å)
hardens monotonically with spin. We sweep a_spin from 0 to 0.998 on the
Kubota & Done (2018) disc backbone, the public-API entry point for
spin-sensitive disc physics in tengri, and report alpha alongside the
SEDs.

Reference: Kubota & Done 2018, MNRAS, 480, 1247 (warm-Compton disc with
relativistic ISCO); Hagen & Done 2023, MNRAS, 525, 3455 (RELAGN
formulation).
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.plot import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

C_AA_PER_S = 2.998e18
SPIN_VALUES = (0.0, 0.3, 0.6, 0.9, 0.998)
SFH = {"type": "const", "all_params": tengri.FIXED, "log_total_mass": -10.0}
DUST = {
    "law": "power_law",
    "type": "two_component",
    "all_params": tengri.FIXED,
    "tau_diff": 0.0,
    "tau_bc": 0.0,
}

ssp = tengri.load_ssp()
model = tengri.SEDModel.build(
    ssp,
    sfh=SFH,
    dust=DUST,
    agn={
        "all_params": tengri.FIXED,
        "log_lbol": 12.5,
        "lum_ratio": 1.0,
        "log_mbh": 8.5,
        "a_spin": tengri.Uniform(0.0, 0.998),
        "disc": {"type": "kubota_done", "all_params": tengri.FIXED},
    },
    redshift=tengri.Fixed(0.0),
)
baseline = dict(model.spec.sample(jax.random.PRNGKey(0)))

fig, (ax_sed, ax_slope) = plt.subplots(1, 2, figsize=(11.0, 4.6))
colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(SPIN_VALUES)))

uv_slopes = []
for spin, color in zip(SPIN_VALUES, colors):
    params = {**baseline, "agn_a_spin": jnp.float64(spin)}
    out = model.predict(params)
    wave = np.asarray(model.wavelengths)
    l_nu = np.asarray(out.rest_sed())
    wave_um = wave * 1.0e-4
    nu = C_AA_PER_S / wave
    mask = (l_nu > 0) & (wave > 50.0)
    ax_sed.loglog(wave_um[mask], l_nu[mask], color=color, lw=1.6, label=rf"$a_* = {spin:.3f}$")

    uv_mask = (wave >= 912.0) & (wave <= 3000.0) & (l_nu > 0)
    if uv_mask.sum() > 5:
        slope = np.polyfit(np.log10(nu[uv_mask]), np.log10(l_nu[uv_mask]), 1)[0]
    else:
        slope = np.nan
    uv_slopes.append(slope)

ax_sed.set(
    xlim=(0.01, 3.0),
    xlabel=r"Rest-frame wavelength [$\mu$m]",
    ylabel=r"$L_\nu$  [erg s$^{-1}$ Hz$^{-1}$]",
)
ax_sed.legend(frameon=False, fontsize=9, loc="lower center")

ax_slope.plot(SPIN_VALUES, uv_slopes, "o-", color="royalblue", lw=1.5, ms=6)
ax_slope.set(
    xlabel=r"BH spin $a_*$",
    ylabel=r"UV slope $\alpha$  ($L_\nu \propto \nu^\alpha$, 912 to 3000 $\mathrm{\AA}$)",
)

fig.tight_layout()
plt.savefig("plot_relagn_spin.png", dpi=150, bbox_inches="tight")
