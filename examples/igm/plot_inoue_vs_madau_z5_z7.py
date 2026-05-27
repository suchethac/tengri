"""
Inoue+2014 vs Madau 1995 across the Lyman break at z=5 and z=7
==============================================================

The two standard IGM transmission prescriptions diverge most
visibly across the Lyman-alpha forest and the Lyman limit. Madau
(1995) is the original analytic Lyman-series effective optical
depth; Inoue+2014 added Lyman-continuum and damped-Lyα systems
in a more careful integral over the H I distribution.

Top: an LBG-like spectrum (young, low dust) multiplied by each
transmission curve. Bottom: the bare transmission. Vertical
markers are observed-frame Lyα 1216 Å and Lyman limit 912 Å.

Reference: Inoue, Shimizu, Iwata, Tanaka 2014 MNRAS 442 1805.
"""

import warnings

import jax
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style
from tengri.igm import igm_transmission, igm_transmission_madau

setup_style()
warnings.filterwarnings("ignore")

REDSHIFTS = (5.0, 7.0)
LAMB_LYA, LAMB_LL = 1216.0, 912.0
wave_rest = np.linspace(700.0, 1500.0, 1024)

ssp = tengri.load_ssp("fsps_prsc_miles_chabrier")
model = tengri.SEDModel.build(
    ssp,
    sfh={"type": "dpl", "*": tengri.FIXED, "tau_gyr": 0.05,
         "log_peak_sfr": 0.8, "alpha": 2.8, "beta": 1.5},
    dust={"type": "two_component", "*": tengri.FIXED,
          "tau_diff": 0.01, "tau_bc": 0.0},
    redshift=tengri.Fixed(5.0),
)
out = model.predict_rest_sed(dict(model.spec.sample(jax.random.PRNGKey(0))))
L_nu_rest = np.interp(wave_rest, np.asarray(out.wavelength),
                      np.asarray(out.sed), left=0.0, right=0.0)
L_nu_norm = L_nu_rest / L_nu_rest[np.argmin(np.abs(wave_rest - 1400.0))]

fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.2), sharey="row")
for col, z in enumerate(REDSHIFTS):
    wave_obs = wave_rest * (1.0 + z)
    t_mad = np.asarray(igm_transmission_madau(wave_obs, z=z))
    t_ino = np.asarray(igm_transmission(wave_obs, z_source=z, add_cgm=False))
    lya_obs = LAMB_LYA * (1.0 + z)
    ll_obs = LAMB_LL * (1.0 + z)

    ax = axes[0, col]
    ax.plot(wave_obs, L_nu_norm, color="0.5", lw=1.0, ls=":", label="intrinsic")
    ax.plot(wave_obs, L_nu_norm * t_mad, color="C0", lw=1.5, ls="--", label="Madau 1995")
    ax.plot(wave_obs, L_nu_norm * t_ino, color="C3", lw=1.5, label="Inoue+2014")
    ax.axvline(lya_obs, color="0.4", ls=":", lw=0.8)
    ax.axvline(ll_obs, color="0.4", ls=":", lw=0.8)
    ax.text(0.97, 0.92, f"$z = {z:.0f}$", transform=ax.transAxes,
            ha="right", va="top", fontsize=10)
    ax.set(ylabel=r"$L_\nu / L_\nu(1400\,\mathrm{\AA})$", ylim=(0, 1.2))
    if col == 0:
        ax.legend(frameon=False, fontsize=8, loc="upper left")

    ax = axes[1, col]
    ax.plot(wave_obs, t_mad, color="C0", lw=1.5, ls="--", label="Madau 1995")
    ax.plot(wave_obs, t_ino, color="C3", lw=1.5, label="Inoue+2014")
    ax.axvline(lya_obs, color="0.4", ls=":", lw=0.8)
    ax.axvline(ll_obs, color="0.4", ls=":", lw=0.8)
    ax.text(lya_obs, 1.05, r" Ly$\alpha$", fontsize=8, color="0.4")
    ax.text(ll_obs, 1.05, " LL", fontsize=8, color="0.4")
    ax.set(ylabel=r"$e^{-\tau_{\rm IGM}}$",
           xlabel=r"observed $\lambda$ [$\mathrm{\AA}$]",
           ylim=(-0.02, 1.18))

fig.tight_layout()
plt.savefig("plot_inoue_vs_madau_z5_z7.png", dpi=150, bbox_inches="tight")
