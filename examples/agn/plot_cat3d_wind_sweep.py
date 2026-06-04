"""
CAT3D-Wind clumpy torus: Wind mass fraction and orientation
===========================================================

The CAT3D-Wind clumpy torus model (Hönig & Kishimoto 2017) combines a
mid-plane clumpy-disc structure with a polar outflow (wind). The torus
morphology is parameterized by three observables: inclination angle
``agn_cos_inc`` (1 = face-on, 0 = edge-on), clump distribution power-law
index ``agn_a_cat3d``, and wind mass fraction ``agn_fwd_cat3d``.

This example demonstrates how the wind fraction modulates the torus infrared
reprocessing. Higher wind fractions (fwd → 1) produce more polar-directed
obscuration, which affects the observed UV-to-IR SED depending on viewing
angle. The clumpy structure also produces flatter SEDs in the mid-infrared
(5–30 µm) compared to smooth toruses like SKIRTOR.

References
----------
.. [1] S. F. Hönig & M. Kishimoto, "The dusty heart of nearby active
   galaxies. II. From clumpy torus models to a unified model," ApJL 838,
   L20 (2017). arXiv:1702.08691.
.. [2] L. N. Martínez-Ramírez et al., "AGNfitter-rx: Modeling the
   radio-to-X-ray spectral energy distributions of AGNs," A&A 688, A46
   (2024). arXiv:2405.12111.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import jax
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")
warnings.filterwarnings("ignore", message=".*deprecated.*")

ssp = tengri.load_ssp()

# Minimal host
SFH = {"type": "const", "*": tengri.FIXED, "log_total_mass": -10.0}
DUST = {"type": "two_component", "*": tengri.FIXED, "tau_diff": 0.0, "tau_bc": 0.0}

# Wind fraction sweep: 0 (pure disc) → 1 (pure wind)
fwd_values = np.linspace(0.0, 1.0, 6)

# Inclination angles (face-on, edge-on)
cos_inc_values = [0.8, 0.2]

fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.5), sharey=True)

c_aa_s = 2.998e18

for ax_idx, cos_inc in enumerate(cos_inc_values):
    ax = axes[ax_idx]

    # Colormap for wind fraction
    norm = mpl.colors.Normalize(vmin=fwd_values.min(), vmax=fwd_values.max())
    cmap = plt.get_cmap("cool")

    for fwd in fwd_values:
        model = tengri.SEDModel.build(
            ssp,
            sfh=SFH,
            dust=DUST,
            agn={
                "type": "composable",
                "disc": {"type": "powerlaw", "*": tengri.FIXED},
                # The CAT3D-Wind grid parameters (radial cloud index a_cat3d and
                # wind mass fraction fwd_cat3d) live in the torus sub-block;
                # cos_inc is the shared AGN viewing angle at the agn top level.
                "torus": {
                    "type": "cat3d_wind",
                    "*": tengri.FIXED,
                    "a_cat3d": -1.5,
                    "fwd_cat3d": fwd,
                },
                "*": tengri.FIXED,
                "log_lbol": 11.5,
                "frac": 0.5,
                "cos_inc": cos_inc,
            },
            redshift=tengri.Fixed(0.05),
        )
        p = dict(model.spec.sample(jax.random.PRNGKey(0)))
        out = model.predict_rest_sed(p)

        wave = np.asarray(out.wavelength)
        nu_l_nu = c_aa_s / wave * np.asarray(out.sed)

        color = cmap(norm(fwd))
        label = f"fwd = {fwd:.2f}" if ax_idx == 1 else None
        ax.loglog(wave, nu_l_nu, color=color, lw=1.4, label=label)

    ax.set_xlim(100, 1e5)
    ax.set_ylim(1e40, 1e47)
    ax.set_xlabel(r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]")
    if ax_idx == 0:
        ax.set_ylabel(r"$\nu L_\nu$  [erg s$^{-1}$]")

    # Title with inclination angle
    inc_deg = np.degrees(np.arccos(cos_inc))
    ax.set_title(f"Inclination = {inc_deg:.1f}° (cos i = {cos_inc:.2f})", fontsize=10)
    ax.grid(True, which="major", alpha=0.2)

# Legend
axes[1].legend(frameon=False, fontsize=8, loc="lower left")

fig.suptitle("CAT3D-Wind: Wind fraction and viewing angle effects", fontsize=11, weight="bold")
fig.tight_layout()
plt.show()
