"""
GRAHSP Fig. 1 reproduction: panchromatic AGN + host overview
=============================================================

Reproduction of Fig. 1 of Buchner et al. (2024, GRAHSP): how the individual
model components sum to the total emission (black). The AGN side is the
GRAHSP bending power-law disk/BBB (blue), iron + emission-line forest (red),
and the dusty torus (yellow dashed), normalised so the disk has
:math:`L_{5100\\,\\mathrm{\\AA}}^{\\rm AGN}=10^{44}\\,\\mathrm{erg\\,s^{-1}}
=10^{37}\\,\\mathrm{W}` (blue square); the torus is anchored at 12 µm (yellow
diamond). The host is a stellar population (purple) and its reprocessed dust
emission (green).

.. note::

   This build uses a ``wNE`` SSP, where nebular line emission is **baked into
   the stellar templates**. The paper's separate gray "nebular emission" curve
   therefore is not shown as an independent component here — it is included
   within the purple stellar curve. A bare-stellar SSP + Cue nebular backend
   would separate it (see ``docs`` on SSP variants).
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import warnings

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri import FIXED, Fixed, SEDModel
from tengri.analysis.plotting import setup_style
from tengri.components.agn.grahsp.model import GRAHSPParams, evaluate_grahsp_agn
from tengri.components.agn.grahsp.templates import load_grahsp_templates

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

C_NM_HZ = 2.99792458e17  # c in nm/s
ERG_TO_W = 1e-7
SSP_PATH = "data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
L5100_ERG = 1.0e44  # disk normalisation (paper: = 1e37 W)

ssp = tengri.load_ssp(SSP_PATH)
# Host galaxy only (stellar + dust energy balance is independent of the AGN);
# the AGN is overlaid below from a matched GRAHSP evaluation so the disk is
# pinned to the paper's L5100 normalisation.
model = SEDModel.build(
    ssp_data=ssp,
    sfh={"type": "dpl", "*": FIXED, "log_total_mass": 12.3},
    dust={
        "type": "two_component",
        "law_bc": "calzetti",
        "*": FIXED,
        "tau_bc": 1.0,
        "tau_diff": 0.3,
        "emission": {"type": "dale2014", "*": FIXED},
    },
    redshift=Fixed(0.01),
)
params = model.spec.get_fixed_values()
st = model.predict_state(params)
wave_nm = np.asarray(st.wave)
wave_um = wave_nm / 1e4


def nu_Lnu_W(lnu):
    # L_nu [erg/s/Hz] -> nu*L_nu = lambda*L_lambda [W]
    return np.asarray(lnu) * (C_NM_HZ / wave_nm) * ERG_TO_W


# --- Galaxy components (rest frame, from the forward state) ---
stellar = nu_Lnu_W(st.derived["sed_dust_attenuated"])  # attenuated stellar (incl. baked nebular)
dust_ir = nu_Lnu_W(st.derived["sed_dust_ir"])  # galaxy dust emission

# --- AGN sub-components, split via a matched GRAHSP evaluation ---
templates = load_grahsp_templates()
agn_params = GRAHSPParams(l5100=L5100_ERG, a_feii=5.0, fcov=0.4, si=-1.0)
sed_agn = evaluate_grahsp_agn(jnp.asarray(wave_nm * 0.1), agn_params, templates)  # nm grid
# evaluate returns L_lambda [erg/s/nm]; convert to L_nu then to nu*L_nu.
to_lnu = (wave_nm**2) / C_NM_HZ  # L_lambda[/nm] -> L_nu : * lambda^2 / c (lambda in nm)
disk = nu_Lnu_W(np.asarray(sed_agn.bbb) * to_lnu)
torus = nu_Lnu_W(np.asarray(sed_agn.torus + sed_agn.si) * to_lnu)
agn_lines = nu_Lnu_W(
    np.asarray(sed_agn.broad_lines + sed_agn.narrow_lines + sed_agn.feii) * to_lnu
)

total = stellar + dust_ir + disk + torus + agn_lines

fig, ax = plt.subplots(figsize=(11.0, 6.2))
ax.plot(wave_um, disk, color="#1f9fe0", lw=2.0, label="AGN disk", zorder=4)
ax.plot(wave_um, torus, color="#e0a020", lw=2.2, ls="--", label="AGN torus", zorder=4)
ax.plot(wave_um, agn_lines, color="#ff6b5a", lw=0.7, label="AGN lines", zorder=3)
ax.plot(wave_um, stellar, color="#7b2fbe", lw=1.8, label="Stellar attenuated", zorder=4)
ax.plot(wave_um, dust_ir, color="#3a8a3a", lw=1.8, label="Dust", zorder=3)
ax.plot(wave_um, total, color="k", lw=1.8, label="Total", zorder=5)

# Normalisation markers.
i5100 = int(np.argmin(np.abs(wave_um - 0.510)))
ax.plot([0.510], [disk[i5100]], "s", color="#1f9fe0", ms=9, zorder=6)
i12 = int(np.argmin(np.abs(wave_um - 12.0)))
ax.plot([12.0], [torus[i12]], "D", color="#e0a020", ms=9, zorder=6)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(0.05, 100.0)
ax.set_ylim(1e34, 3e38)
ax.set_xlabel(r"Wavelength [$\mu$m]")
ax.set_ylabel(r"Luminosity $\lambda L_\lambda$ [W]")
ax.legend(loc="upper right", frameon=True, fontsize=9, ncol=2, title="Components")

secax = ax.secondary_xaxis(
    "top", functions=(lambda x: C_NM_HZ / 1e3 / x, lambda nu: C_NM_HZ / 1e3 / nu)
)
secax.set_xlabel("Frequency [Hz]")

fig.tight_layout()
plt.show()
