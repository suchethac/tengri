"""
Panchromatic SED of a low-metallicity dwarf irregular galaxy
=============================================================

A low-mass, low-metallicity dwarf irregular (M*~10^8 M☉, Z~0.1 Z☉)
with high specific star formation rate. The SED highlights: strong UV
continuum from young stars, dominant optical emission lines (Hα 6563 Å,
[OIII] 5007 Å, Hβ) on a faint continuum, minimal dust attenuation, and
negligible far-infrared. Equivalent width of Hα is extreme (~100s Å).
Metal-poor stellar populations and active star formation drive the
starburst signature visible from UV through optical.

**References:**

 - Östlin et al. (2014) [1]_ for dwarf starburst SED archetypes
 - Chabrier (2003) [2]_ for young, metal-poor IMF
 - Cue (2009) [3]_ for nebular emission at low metallicity

.. [1] Östlin, G., et al. (2014).
   *Astrophys. J.* **797**, 11.
   https://doi.org/10.1088/0004–637X/797/1/11

.. [2] Chabrier, G. (2003).
   *Publ. Astron. Soc. Pac.* **115**, 763–795.
   https://doi.org/10.1086/376392

.. [3] Cue, C. (2009).
   *Astrophys. J. Suppl. Ser.* **183**, 1–39.
   https://doi.org/10.1088/0067–0049/183/1/1
"""

import warnings

import jax
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

# Load bare-stellar SSP (required for Cue nebular backend)
SSP = tengri.load_ssp("fsps_prsc_miles_chabrier")

# Dwarf irregular: extremely young starburst, log-flat SFH, low metallicity
model = tengri.SEDModel.build(
    SSP,
    sfh={
        "type": "const",
        "all_params": tengri.FIXED,
        "log_total_mass": 7.20,  # SFR ~ 0.3 Msun/yr (high sSFR for 1e8 Msun)
        "start_gyr": 0.05,  # Recent burst: last 50 Myr
        "end_gyr": 0.0,
    },
    dust={
        "type": "two_component",
        "all_params": tengri.FIXED,
        "tau_diff": 0.05,  # Minimal dust: tau_V ~ 0.05 (almost no attenuation)
        "tau_bc": 0.05,
        "emission": {"type": "dale2014", "all_params": tengri.FIXED},
    },
    neb={
        "type": "cue",
        "all_params": tengri.FIXED,  # Fixed Cue line emission (low-Z)
    },
    redshift=tengri.Fixed(0.001),  # Nearby for high signal-to-noise
)

# Sample and predict rest-frame SED
p = dict(model.spec.sample(jax.random.PRNGKey(0)))
out = model.predict(p)
wave = np.asarray(model.wavelengths)
nu_l_nu = 2.998e18 / wave * np.asarray(out.rest_sed())

fig, ax = plt.subplots(figsize=(8.0, 5.0))

# Plot full panchromatic SED (mask zeros for clean loglog)
mask = np.asarray(out.rest_sed()) > 0
ax.loglog(
    wave[mask],
    nu_l_nu[mask],
    color="C0",
    lw=1.6,
    label="Dwarf starburst (UV-rich, low dust)",
)

# Annotate emission lines with vertical markers (y placed at SED value + headroom)
for x_lam, lbl in [
    (1216, r"Ly$\alpha$"),
    (4861, r"H$\beta$"),
    (5007, "[OIII]5007"),
    (6563, r"H$\alpha$"),
]:
    j = int(np.argmin(np.abs(wave - x_lam)))
    ax.axvline(x_lam, color="0.7", ls=":", lw=0.6)
    ax.text(x_lam, nu_l_nu[j] * 2.5, " " + lbl, fontsize=8, color="0.4")

# Axis labels and limits
ax.set(
    xlim=(500, 1e7),
    xlabel=r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]",
    ylabel=r"$\nu L_\nu$ [erg s$^{-1}$]",
)
nu_l_nu_pos = nu_l_nu[mask]
ax.set_ylim(nu_l_nu_pos.max() / 1e4, nu_l_nu_pos.max() * 3)

# Properties box (lower right)
props_text = (
    "$M_*$ ≈ $10^8$ M$_\\odot$ | SFR ≈ 0.3 M$_\\odot$/yr\nZ ≈ 0.1 Z$_\\odot$ | sSFR ≈ 3 Gyr$^{-1}$"
)
ax.text(
    0.97,
    0.05,
    props_text,
    transform=ax.transAxes,
    ha="right",
    fontsize=8,
    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.7", lw=0.5),
)

ax.legend(frameon=False, fontsize=8.5, loc="upper left")

fig.tight_layout()
plt.savefig("plot_dwarf_irregular_sed.png", dpi=150, bbox_inches="tight")
