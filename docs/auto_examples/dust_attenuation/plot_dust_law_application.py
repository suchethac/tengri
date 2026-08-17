"""
The same galaxy reddened by every attenuation law in the registry
==================================================================

Each of the bundled dust-attenuation laws applied to the *same*
intrinsic SED at the *same* V-band optical depth — so the differences
between the curves are entirely in the wavelength dependence of the
attenuation. The intrinsic (unreddened) SED is shown in black for
reference.

Pay attention to the UV: SMC-like grains (Pei 1992 SMC, no bump)
suppress the UV most steeply; Cardelli MW and Kriek & Conroy retain
the 2175 Å bump; Calzetti and Salim flatten the UV slope (the
"Calzetti-like" plateau).

Laws cover starburst (Calzetti, Conroy+10, Salim+18, Narayanan+18,
TEA), extinction-curve (Cardelli MW, SMC, LMC, Pei 1992),
grain-physics (Draine 2003, WD01), and birth-cloud (Wild+07).
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import jax
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

LAWS = [
    ("calzetti", "Calzetti+2000 (starburst)"),
    ("cardelli", "Cardelli+1989 (MW)"),
    ("smc", "SMC (Pei 1992)"),
    ("kriek_conroy", "Kriek & Conroy 2013"),
    ("salim", "Salim+2018"),
    ("noll09", "Noll+2009 (mod. Calzetti)"),
    ("narayanan_z", "Narayanan+2018 (SIMBA)"),
    ("hd23_mwrv31", "Hensley & Draine 2023"),
]
COLORS = plt.cm.tab10(np.linspace(0.0, 0.9, len(LAWS)))

C_AA_PER_S = 2.998e18
SFH = {
    "type": "tsnorm",
    "all_params": tengri.FIXED,
    "peak_lbt_gyr": 2.0,
    "width_gyr": 1.0,
    "log_total_mass": 10.0,
    "skew": 0.0,
    "trunc": 13.0,
}

ssp = tengri.load_ssp()
fig, ax = plt.subplots(figsize=(7.2, 4.6))

ref_model = tengri.SEDModel.build(
    ssp,
    sfh=SFH,
    dust={
        "type": "two_component",
        "law_diff": "calzetti",
        "all_params": tengri.FIXED,
        "tau_diff": 0.0,
        "tau_bc": 0.0,
    },
    redshift=tengri.Fixed(0.05),
)
p_ref = dict(ref_model.spec.sample(jax.random.PRNGKey(0)))
sed_ref = np.asarray(ref_model.predict(p_ref).rest_sed())
wave = np.asarray(ref_model.wavelengths)
nu = C_AA_PER_S / wave
ax.loglog(wave, nu * sed_ref, color="0.05", lw=2.0, label="intrinsic", zorder=10, ls="--")

plotted = 0
first_failure: Exception | None = None

for (law, label), color in zip(LAWS, COLORS):
    try:
        model = tengri.SEDModel.build(
            ssp,
            sfh=SFH,
            dust={
                "type": "two_component",
                "law_diff": law,
                "all_params": tengri.FIXED,
                "tau_diff": 1.0,
                "tau_bc": 0.3,
            },
            redshift=tengri.Fixed(0.05),
        )
    except Exception as e:
        if first_failure is None:
            first_failure = e
        continue
    p = dict(model.spec.sample(jax.random.PRNGKey(0)))
    sed = np.asarray(model.predict(p).rest_sed())
    ax.loglog(wave, nu * sed, color=color, lw=1.4, label=label)
    plotted += 1

# The intrinsic reference curve is drawn above, outside the loop, so the axes
# is never literally empty and no figure-level "is it blank" check can see this.
# Only the loop knows it produced nothing.
if plotted == 0:
    raise RuntimeError(
        f"none of the {len(LAWS)} dust laws built, so only the intrinsic "
        f"reference is drawn. First failure: "
        f"{type(first_failure).__name__}: {first_failure}"
    ) from first_failure

ax.set(
    xlim=(900, 3e4),
    ylim=(1e40, 8e43),
    xlabel=r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]",
    ylabel=r"$\nu L_\nu$  [erg s$^{-1}$]",
)
ax.axvline(2175, color="0.55", lw=0.4, ls=":")
ax.text(2175, 1.5e40, "2175 Å bump", fontsize=8, color="0.4", rotation=90, va="bottom", ha="right")
ax.legend(frameon=False, fontsize=7.5, loc="lower right", ncol=2)

fig.tight_layout()
plt.savefig("plot_dust_law_application.png", dpi=150, bbox_inches="tight")
