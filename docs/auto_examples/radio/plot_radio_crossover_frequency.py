r"""
Synchrotron / free-free balance vs synchrotron slope α_sf
=========================================================

A star-forming galaxy's GHz continuum is set by two components:
non-thermal synchrotron from supernova remnants (steep, L_ν ∝
ν^{-α_sf}) and thermal free-free from H II regions (flat, L_ν ∝
ν^{-0.1}). Their ratio at fixed frequency depends sensitively on
the synchrotron spectral index α_sf — flatter spectra leave more
of the GHz luminosity to free-free, steeper spectra are
synchrotron-dominated until the (sub-mm) crossover.

We hold L_IR = 10^{11} L_sun and the Bell+2003 IR-radio q
parameter fixed, sweep α_sf, and show how the thermal fraction
``f_thermal(ν) = L_ff / (L_ff + L_sync)`` rises with frequency
— faster for flatter synchrotron.

References:

- Condon 1992 ARA&A 30 575.
- Murphy et al. 2011 ApJ 737 67 (free-free calibration).

"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import warnings

import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from tengri.plot import setup_style
from tengri.radio import compute_radio_components

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

C_AA_S = 2.998e18
L_SUN = 3.828e33

wave = jnp.logspace(7, 11, 400)
nu_ghz = (C_AA_S / np.asarray(wave)) / 1e9

alpha_sf_vals = np.array([0.5, 0.7, 0.8, 1.0, 1.2])
norm = mpl.colors.Normalize(alpha_sf_vals.min(), alpha_sf_vals.max())
cmap = plt.get_cmap("viridis")

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

for alpha in alpha_sf_vals:
    c = compute_radio_components(
        wavelength=wave,
        L_ir=1e11 * L_SUN,
        L_agn_bol=0.0,
        q_ir=2.64,
        alpha_sf=alpha,
        sfr_mode="bell2003",
        include_freefree=True,
        T_e=1e4,
        alpha_ff=-0.1,
    )
    L_sync = np.asarray(c["synchrotron"])
    L_ff = np.asarray(c["freefree"])
    L_tot = L_sync + L_ff
    f_th = L_ff / np.where(L_tot > 0, L_tot, np.nan)

    col = cmap(norm(alpha))
    ax.loglog(nu_ghz, L_sync, color=col, lw=0.7, ls="--", alpha=0.55)
    ax.loglog(nu_ghz, L_ff, color=col, lw=0.7, ls=":", alpha=0.55)
    ax.loglog(nu_ghz, L_tot, color=col, lw=1.7, label=rf"$\alpha_{{\rm sf}} = {alpha:.1f}$")
    ax2.semilogx(nu_ghz, f_th, color=col, lw=1.6)

ax.set(
    xlim=(0.1, 300),
    ylim=(5e27, 5e30),
    xlabel=r"$\nu$ [GHz]",
    ylabel=r"$L_\nu$ [erg s$^{-1}$ Hz$^{-1}$]",
)
ax.legend(frameon=False, fontsize=8, loc="upper right")

ax2.axhline(0.5, color="0.7", ls="--", lw=0.7)
ax2.set(
    xlim=(0.1, 300),
    ylim=(0, 1),
    xlabel=r"$\nu$ [GHz]",
    ylabel=r"thermal fraction  $L_{\rm ff} / L_{\rm total}$",
)
ax2.text(0.4, 0.55, "f$_{th}$=0.5", fontsize=8, color="0.5")

fig.tight_layout()
plt.savefig("plot_radio_crossover_frequency.png", dpi=150, bbox_inches="tight")
