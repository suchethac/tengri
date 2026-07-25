"""
Big Blue Bump: multicolor disc temperature evolution with black-hole mass
==========================================================================

The Shakura-Sunyaev thin disc model shows how the big blue bump (BBB) peak
shifts to longer wavelengths as black-hole mass increases. At fixed Eddington
ratio ``log(L_bol / L_Edd) = -1.0``, the disc temperature scales as
:math:`T_{\\rm in} \\propto (\\dot{m} / m_\\odot)^{1/4}`, where the inner
temperature determines the location of peak νLν. Higher mass → lower accretion
rate → cooler disc → redder peak.

Sweeping ``agn_log_mbh`` from 6 to 9.5 (6 to ~3e9 solar masses) at fixed
Eddington ratio samples the range from stellar-mass-like accretion physics to
supermassive black holes.
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
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")
warnings.filterwarnings("ignore", message=".*deprecated.*")

# Minimum SSP for composable AGN (bare-stellar required by Cue nebular emulator).
ssp = tengri.load_ssp()

# Build model with multicolor disc only, all components fixed.
model = tengri.SEDModel.build(
    ssp,
    sfh={
        "type": "dpl",
        "all_params": tengri.FIXED,
        "tau_gyr": 3.0,
        "log_total_mass": 10.0,
        "alpha": 2.0,
        "beta": 2.5,
    },
    dust={"type": "two_component", "all_params": tengri.FIXED, "tau_diff": 0.1, "tau_bc": 0.1},
    agn={
        "type": "composable",
        "disc": {"type": "multicolor", "all_params": tengri.FIXED},
        "all_params": tengri.FIXED,
        "lum_ratio": 1.0,
        "log_ledd": -1.0,
    },
    redshift=tengri.Fixed(0.05),
)

# Sample baseline parameters, then override log_mbh and compute agn_log_lbol
# for each MBH mass.
baseline = dict(model.spec.sample(jax.random.PRNGKey(0)))

# Black hole masses: log10(M / M_sun) from 6 to 9.5.
# Eddington luminosity: L_Edd = 3.2e4 * M / M_sun * L_sun.
# At fixed L / L_Edd = 10^{-1.0}, log_lbol = log_mbh + log(3.2e4) + log_ledd.
# log(3.2e4) ≈ 4.505, so log_lbol = log_mbh + 4.505 + (-1.0) ≈ log_mbh + 3.505.
log_mbh_values = np.linspace(6.0, 9.5, 8)
log_ledd = -1.0
log_lbol_offset = np.log10(3.2e4) + log_ledd  # ≈ 3.505

norm = mpl.colors.Normalize(vmin=log_mbh_values.min(), vmax=log_mbh_values.max())
cmap = plt.get_cmap("viridis")

fig, ax = plt.subplots(figsize=(6.5, 4.2))
for log_mbh in log_mbh_values:
    log_lbol = log_mbh + log_lbol_offset
    params = {
        **baseline,
        "agn_log_mbh": jnp.float64(log_mbh),
        "agn_log_lbol": jnp.float64(log_lbol),
    }
    out = model.predict(params)
    wave = np.asarray(model.wavelengths)
    nu = 2.998e18 / wave  # frequency in Hz
    nu_l_nu = nu * np.asarray(out.rest_sed())
    ax.loglog(wave, nu_l_nu, color=cmap(norm(log_mbh)), lw=1.4)

ax.set_xlim(100, 1e4)
ax.set_ylim(1e40, 1e48)
ax.set_xlabel(r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]")
ax.set_ylabel(r"$\nu L_\nu$  [erg s$^{-1}$]")

cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.01)
cbar.set_label(r"$\log M_{\mathrm{BH}} / M_\odot$")

fig.tight_layout()
plt.savefig("plot_agn_bbb_mbh_sweep.png", dpi=150, bbox_inches="tight")
