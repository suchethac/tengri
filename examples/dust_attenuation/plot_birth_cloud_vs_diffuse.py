"""
Birth-cloud vs diffuse-ISM dust: which knob does what?
========================================================

The Charlot & Fall 2000 two-component dust model splits attenuation
into a birth-cloud component (``τ_bc``) that only the youngest stellar
ages see, and a diffuse-ISM component (``τ_diff``) that attenuates all
stellar light. The two are degenerate for an old population (every
star is "old" by the BC clock, so τ_bc has no effect) but separate
cleanly for a young one.

Two panels for a young (peak 0.3 Gyr) burst:
- ``τ_bc`` sweep at fixed ``τ_diff = 0.2``   — far-UV strongly suppressed
- ``τ_diff`` sweep at fixed ``τ_bc = 0.5``  — full SED suppressed
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

C_AA_PER_S = 2.998e18

ssp = tengri.load_ssp()
SFH = {
    "type": "dpl",
    "*": tengri.FIXED,
    "tau_gyr": 0.3,
    "log_peak_sfr": 1.5,
    "alpha": 4.0,
    "beta": 2.0,
}


def _model(tau_diff, tau_bc):
    return tengri.SEDModel.build(
        ssp,
        sfh=SFH,
        dust={
            "type": "two_component",
            "*": tengri.FIXED,
            "tau_diff": tau_diff,
            "tau_bc": tau_bc,
            "slope": -0.7,
        },
        redshift=tengri.Fixed(0.05),
    )


fig, (ax_bc, ax_diff) = plt.subplots(
    1, 2, figsize=(11.5, 4.4), sharey=True, gridspec_kw={"wspace": 0.05}
)

tau_bc_vals = np.linspace(0.0, 2.0, 7)
tau_diff_vals = np.linspace(0.0, 1.5, 7)
norm_bc = mpl.colors.Normalize(vmin=tau_bc_vals.min(), vmax=tau_bc_vals.max())
norm_diff = mpl.colors.Normalize(vmin=tau_diff_vals.min(), vmax=tau_diff_vals.max())

for tau in tau_bc_vals:
    model = _model(tau_diff=0.2, tau_bc=float(tau))
    p = dict(model.spec.sample(jax.random.PRNGKey(0)))
    out = model.predict_rest_sed(p)
    wave = np.asarray(out.wavelength)
    ax_bc.loglog(
        wave, C_AA_PER_S / wave * np.asarray(out.sed), color=plt.cm.viridis(norm_bc(tau)), lw=1.4
    )

for tau in tau_diff_vals:
    model = _model(tau_diff=float(tau), tau_bc=0.5)
    p = dict(model.spec.sample(jax.random.PRNGKey(0)))
    out = model.predict_rest_sed(p)
    wave = np.asarray(out.wavelength)
    ax_diff.loglog(
        wave, C_AA_PER_S / wave * np.asarray(out.sed), color=plt.cm.viridis(norm_diff(tau)), lw=1.4
    )

for ax in (ax_bc, ax_diff):
    ax.set(
        xlim=(800, 3e4),
        ylim=(1e41, 5e44),
        xlabel=r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]",
    )

ax_bc.set_ylabel(r"$\nu L_\nu$  [erg s$^{-1}$]")
ax_bc.text(
    0.05,
    0.93,
    r"sweep $\tau_{\rm bc}$, fixed $\tau_{\rm diff}=0.2$",
    transform=ax_bc.transAxes,
    fontsize=9,
    color="0.15",
    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", lw=0.5),
)
ax_diff.text(
    0.05,
    0.93,
    r"sweep $\tau_{\rm diff}$, fixed $\tau_{\rm bc}=0.5$",
    transform=ax_diff.transAxes,
    fontsize=9,
    color="0.15",
    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", lw=0.5),
)

cb_bc = fig.colorbar(plt.cm.ScalarMappable(norm=norm_bc, cmap="viridis"), ax=ax_bc, pad=0.01)
cb_bc.set_label(r"$\tau_{\rm bc}$  [mag]")
cb_diff = fig.colorbar(plt.cm.ScalarMappable(norm=norm_diff, cmap="viridis"), ax=ax_diff, pad=0.01)
cb_diff.set_label(r"$\tau_{\rm diff}$  [mag]")

plt.savefig("plot_birth_cloud_vs_diffuse.png", dpi=150, bbox_inches="tight")
