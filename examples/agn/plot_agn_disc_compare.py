"""
AGN model comparison — disc + torus families
===============================================

tengri ships several alternative AGN engines. At a fixed bolometric
luminosity ``log L_bol = 12.5`` (in log L_sun) and standard inclination,
we evaluate each backbone in isolation (zero stellar SFR, no dust on
the AGN, ``agn_frac = 1``) and overlay the rest-frame ``νL_ν``.

Note: this script uses the legacy ``tengri.Parameters(agn_model=...)``
flat-kwarg form because the composable AGN routing under
``SEDModel.build(agn={'type': 'composable', ...})`` does not currently
activate the AGN SED — see issue #258. Switch to the nested-dict form
when that's fixed.

Models compared (top-level ``agn_model`` selectors):
- ``qsogen``           — Temple+2021 empirical type-1 QSO SED
- ``multicolor_agn``   — Shakura–Sunyaev disc + 2-T torus (K&D 2018)
- ``kubota_done``      — K&D 2018 simplified Comptonization
- ``kubota_done_full`` — K&D 2018 full 3-zone disc + Compton hump
- ``relagn``           — RELAGN relativistic disc + 2-T torus (Hagen & Done 2023)
- ``skirtor``          — power-law disc + SKIRTOR clumpy torus (Stalevski+2016)
- ``silva04``          — Silva+2004 power-law disc + smooth torus
"""

import warnings

import jax
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")
warnings.filterwarnings("ignore", message=".*deprecated.*")
warnings.filterwarnings("ignore", message=".*toy.*")

MODELS = [
    ("qsogen",           "QSOGEN (Temple+2021)"),
    ("multicolor_agn",   "multicolor disc + 2-T torus (K&D 2018)"),
    ("kubota_done",      "K&D 2018 simplified"),
    ("kubota_done_full", "K&D 2018 full"),
    ("relagn",           "RELAGN (Hagen & Done 2023)"),
    ("skirtor",          "SKIRTOR (Stalevski+2016)"),
    ("silva04",          "Silva+2004"),
]
COLORS = plt.cm.viridis(np.linspace(0.05, 0.92, len(MODELS)))

ssp = tengri.load_ssp()
C_AA_PER_S = 2.998e18

fig, ax = plt.subplots(figsize=(7.4, 4.8))

for (model_name, label), color in zip(MODELS, COLORS):
    spec = tengri.Parameters(
        agn_model=model_name,
        agn_frac=tengri.Fixed(1.0),
        agn_log_lbol=tengri.Fixed(12.5),
        mean_sfh_type="const",
        sfh_const_log_sfr=tengri.Fixed(-10.0),
        dust_tau_diff=tengri.Fixed(0.0),
        dust_tau_bc=tengri.Fixed(0.0),
        redshift=tengri.Fixed(0.05),
    )
    model = tengri.SEDModel(spec, ssp)
    p = dict(model.spec.sample(jax.random.PRNGKey(0)))
    out = model.predict_rest_sed(p)
    wave = np.asarray(out.wavelength)
    nu_l_nu = C_AA_PER_S / wave * np.asarray(out.sed)
    ax.loglog(wave, nu_l_nu, color=color, lw=1.4, label=label)

ax.set_xlim(20, 3e6)
ax.set_ylim(1e42, 5e47)
ax.set_xlabel(r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]")
ax.set_ylabel(r"$\nu L_\nu$  [erg s$^{-1}$]")
ax.axvspan(1, 100, color="0.93", alpha=0.6, lw=0)
ax.text(30, 2e47, "X-ray", color="0.4", fontsize=8, va="top")
ax.text(2000, 2e47, "UV/optical BBB", color="0.4", fontsize=8, va="top")
ax.text(2e5, 2e47, "MIR torus", color="0.4", fontsize=8, va="top", ha="right")
ax.legend(frameon=False, fontsize=7.5, loc="lower right")

fig.tight_layout()
fig.savefig("plot_agn_disc_compare.png", dpi=150, bbox_inches="tight")
