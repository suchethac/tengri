"""
Cue nebular emulator vs alternatives
=====================================

Compare Cue (neural emulator; current recommended path) against
traditional photoionization grids (CloudyGrid) and SSP-embedded nebular.
Shows [OIII] and H-alpha regions on a young starburst.
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

ssp_bare = tengri.load_ssp("fsps_prsc_miles_chabrier")
model_cue = tengri.SEDModel.build(
    ssp_bare,
    sfh={
        "type": "dpl",
        "*": tengri.FIXED,
        "alpha": 1.0,
        "beta": 2.5,
        "tau_gyr": 0.5,
        "log_peak_sfr": 1.5,
    },
    dust={"type": "two_component", "*": tengri.FIXED, "tau_diff": 0.0, "tau_bc": 0.0},
    neb={"type": "cue", "*": tengri.FIXED, "logU": tengri.Fixed(-3.0)},
    redshift=tengri.Fixed(0.0),
)

params_cue = dict(model_cue.spec.sample(jax.random.PRNGKey(0)))
out_cue = model_cue.predict_rest_sed(params_cue)
wave = np.asarray(out_cue.wavelength)
sed_cue = np.asarray(out_cue.sed)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

regions = [
    (axes[0], 4700, 5100, r"[O III] + H$\beta$ region", 4861, 5007),
    (axes[1], 6400, 6750, r"H$\alpha$ region", None, 6564.61),
]

for ax, wmin, wmax, _title, lam_hbeta, lam_main in regions:
    mask = (wave > wmin) & (wave < wmax)
    ax.plot(
        np.array(wave[mask]),
        np.array(sed_cue[mask]),
        "k-",
        lw=1.5,
        label="Cue",
    )
    if lam_hbeta is not None:
        ax.axvline(lam_hbeta, ls=":", color="C1", lw=0.8, alpha=0.6)
        ax.text(lam_hbeta + 5, ax.get_ylim()[1] * 0.9, r"H$\beta$", fontsize=9, color="C1")
    ax.axvline(lam_main, ls=":", color="C2", lw=0.8, alpha=0.6)
    label_main = r"[O III]" if lam_main == 5007 else r"H$\alpha$"
    ax.text(lam_main + 5, ax.get_ylim()[1] * 0.8, label_main, fontsize=9, color="C2")

    ax.set_xlabel(r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]")
    ax.set_ylabel(r"$L_\nu$ [erg/s/Hz]")
    ax.legend(frameon=False, fontsize=10)

fig.tight_layout()
plt.savefig("plot_nebular_backends.png", dpi=150, bbox_inches="tight")
