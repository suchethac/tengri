"""
Radio blocks: which q_IR calibration, and which AGN synchrotron shape
=====================================================================

The radio group is two independent choices — a star-forming block tied to the
FIR-radio correlation, and an AGN block — so this compares them one at a time
on the same galaxy.

(Left) The three q_IR calibrations, with AGN radio off. They disagree most
where the correlation is least constrained: ~50% at 150 MHz, the LOFAR band
that ``mccheyne2022`` was fit in, converging to a few percent by 10 GHz. The
mass- and redshift-dependent calibrations sit below the fixed-q one for this
galaxy, which is the whole point of preferring them.

(Right) The AGN blocks, on a radio-loud AGN. ``dpl`` bends — it is a broken
double power-law with an exponential aging cutoff — while ``powerlaw`` keeps
one slope, so they separate toward high frequency rather than in normalization.
``none`` is star-forming synchrotron alone.

``loudness`` is ``log10(L_5GHz / L_B)``, so 0 is the radio-quiet boundary and
the 2.0 used here is a radio-loud source; its prior runs to 4. Passing 100
"to be safe" is not off the end of a scale, it is 1e100.

Reference: Bell+2003; Delvecchio+2021; McCheyne+2022 (q_IR calibrations).
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import jax
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.plot import setup_style
from tengri.units import C_AA

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")
warnings.filterwarnings("ignore", message=".*experimental.*")

SFH = {"type": "const", "all_params": tengri.FIXED, "log_total_mass": 10.5}
DUST = {
    "law": "power_law",
    "type": "two_component",
    "all_params": tengri.FIXED,
    "tau_diff": 1.0,
    "tau_bc": 1.5,
    "emission": {"type": "dale2014", "all_params": tengri.FIXED},
}
AGN = {
    "type": "composable",
    "all_params": tengri.FIXED,
    "disc": {"type": "qsogen", "all_params": tengri.FIXED},
    "torus": {"type": "skirtor", "all_params": tengri.FIXED},
    "log_lbol": 12.0,  # log10(L_bol / L_sun) at API level, never erg/s
}

ssp = tengri.load_ssp("fsps_prsc_miles_chabrier", download=True)


def spectrum(radio, *, with_agn):
    """L_nu on a frequency axis for one radio configuration."""
    model = tengri.SEDModel.build(
        ssp_data=ssp,
        sfh=SFH,
        dust=DUST,
        radio=radio,
        redshift=tengri.Fixed(0.05),
        **({"agn": AGN} if with_agn else {}),
    )
    # Nothing free, so the draw is the declared values: same galaxy each time.
    assert not model.spec.free_params, model.spec.free_params
    pred = model.predict(model.spec.sample(jax.random.PRNGKey(0)))
    nu_ghz = (C_AA / np.asarray(pred.wave_rest)) / 1e9
    return nu_ghz, np.asarray(pred.rest_sed())


fig, (ax_sf, ax_agn) = plt.subplots(1, 2, figsize=(11.0, 4.2), sharey=True)

sf_blocks = [r["name"] for r in tengri.list_radio_blocks() if r["category"] == "sf"]
sf_blocks = [b for b in sf_blocks if b != "none"]
for name, color in zip(sf_blocks, plt.get_cmap("viridis")(np.linspace(0.1, 0.85, len(sf_blocks)))):
    nu, sed = spectrum(
        {
            "sf": {"type": name, "all_params": tengri.FIXED},
            "agn": {"type": "none"},
            "all_params": tengri.FIXED,
        },
        with_agn=False,
    )
    band = (nu > 0.05) & (nu < 100.0)
    ax_sf.loglog(nu[band], sed[band], color=color, lw=1.4, label=name)

agn_blocks = [r["name"] for r in tengri.list_radio_blocks() if r["category"] == "agn"]
for name, color in zip(
    agn_blocks, plt.get_cmap("viridis")(np.linspace(0.1, 0.85, len(agn_blocks)))
):
    nu, sed = spectrum(
        {
            "sf": {"type": "bell2003", "all_params": tengri.FIXED},
            "agn": {"type": name, "all_params": tengri.FIXED, "loudness": 2.0},
            "all_params": tengri.FIXED,
        },
        with_agn=True,
    )
    band = (nu > 0.05) & (nu < 100.0)
    ax_agn.loglog(nu[band], sed[band], color=color, lw=1.4, label=name)

for ax, label in ((ax_sf, "star-forming block"), (ax_agn, "AGN block")):
    ax.axvline(0.15, color="0.8", lw=0.4, ls=":")
    ax.axvline(1.4, color="0.8", lw=0.4, ls=":")
    ax.set_xlabel(r"Rest-frame frequency $\nu$ [GHz]")
    ax.legend(frameon=False, fontsize=8, title=label, title_fontsize=8)
ax_sf.set_ylabel(r"$L_\nu$ [erg s$^{-1}$ Hz$^{-1}$]")

fig.tight_layout()
fig.savefig("plot_radio_model_family_compare.png", dpi=150, bbox_inches="tight")
