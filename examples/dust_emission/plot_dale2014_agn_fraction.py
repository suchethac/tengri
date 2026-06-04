"""
Dale 2014 dust IR: AGN fraction (CIGALE-faithful additive mixing)
=================================================================

The Dale et al. (2014) IR template family can be combined with a pure-AGN
("quasar") template to represent dust heated by an obscured AGN in addition to
the star-forming ISM. tengri reproduces CIGALE's convention, where the AGN is a
*separate power source* added on top of the stellar-heated dust:

.. math::

    L_\\mathrm{dust} = L_\\mathrm{absorbed}, \\quad
    L_\\mathrm{AGN} = L_\\mathrm{dust}\\,\\frac{f_\\mathrm{AGN}}{1 - f_\\mathrm{AGN}}, \\quad
    \\mathrm{SED} = L_\\mathrm{dust}\\,\\mathrm{SF}(\\alpha) + L_\\mathrm{AGN}\\,\\mathrm{QSO}

so the total emitted IR is :math:`L_\\mathrm{dust}/(1-f_\\mathrm{AGN})` — it
*grows* with ``f_AGN`` because the AGN injects energy the stars did not.

This uses the ``dale2014_cigale`` emission type, whose SF templates are
regenerated from CIGALE's database and which also ships the quasar template
(``scripts/regenerate_dale2014_from_cigale.py``). The default ``dale2014`` is
the Wyoming-source star-forming-only release.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import warnings

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

C_AA_PER_S = 2.998e18

ssp = tengri.load_ssp()

model = tengri.SEDModel.build(
    ssp,
    sfh={"type": "const", "*": tengri.FIXED, "log_total_mass": 11.0},
    dust={
        "type": "two_component",
        "*": tengri.FIXED,
        "tau_diff": 1.0,
        "tau_bc": 0.3,
        "emission": {"type": "dale2014_cigale", "*": tengri.FIXED},
    },
    redshift=tengri.Fixed(0.05),
)
p0 = dict(model.spec.sample(jax.random.PRNGKey(0)))

frac_agn_values = [0.0, 0.2, 0.5, 0.8]
colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(frac_agn_values)))

fig, ax = plt.subplots(figsize=(7.2, 4.6))

for f, c in zip(frac_agn_values, colors):
    out = model.predict_rest_sed({**p0, "dust_frac_agn": jnp.float64(f)})
    wave = np.asarray(out.wavelength)
    nu_l_nu = C_AA_PER_S / wave * np.asarray(out.sed)
    ax.loglog(wave, nu_l_nu, color=c, lw=2.0, label=rf"$f_{{\rm AGN}}={f:.1f}$")

ax.axvline(8e4, color="0.85", lw=0.5, linestyle=":")
ax.text(8e4, 3e44, "8 μm", fontsize=7.5, color="0.6", ha="center", va="bottom")
ax.axvline(60e4, color="0.85", lw=0.5, linestyle=":")
ax.text(60e4, 3e44, "60 μm", fontsize=7.5, color="0.6", ha="center", va="bottom")

ax.set(
    xlim=(1e4, 1e7),
    xlabel=r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]",
    ylabel=r"$\nu L_\nu$  [erg s$^{-1}$]",
    title="Dale 2014: AGN-heated dust adds MIR power (CIGALE additive mixing)",
)
ax.legend(frameon=False, fontsize=9, loc="upper right")

fig.tight_layout()
plt.savefig("plot_dale2014_agn_fraction.png", dpi=150, bbox_inches="tight")
