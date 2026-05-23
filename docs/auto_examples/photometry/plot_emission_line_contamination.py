"""
Emission-line contamination of broadband photometry
=====================================================

For a young high-z galaxy, the [O III]+Hβ complex can boost the
broadband flux by several tenths of a magnitude when it lands inside
a wide filter. We mock the same young SF galaxy at ``z = 1, 3, 6``
and show the JWST NIRCam ``F277W`` / ``F356W`` / ``F444W`` flux with
and without the nebular block on, so the reader can see which
redshift puts which strong line where.

Bars in green: continuum-only model. Bars in red overlay: full
nebular contribution. The wedge above each red bar is the line boost.
"""

import warnings

import jax
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

BANDS = ["jwst_f277w", "jwst_f356w", "jwst_f444w"]
LABELS = ["F277W", "F356W", "F444W"]
ZS = [1.0, 3.0, 6.0]

obs = tengri.Observation(photometry=tengri.Photometry.from_names(BANDS))
SSP_WNE = tengri.load_ssp()                              # nebular baked in
SSP_BARE = tengri.load_ssp("fsps_prsc_miles_chabrier")   # bare stellar continuum
SFH = {"type": "dpl", "*": tengri.FIXED, "tau_gyr": 0.1,
       "log_peak_sfr": 1.5, "alpha": 4.0, "beta": 2.0}
DUST = {"type": "two_component", "*": tengri.FIXED, "tau_diff": 0.05, "tau_bc": 0.1}


def _flux(z, with_nebular):
    # wNE SSP carries nebular lines baked in; bare-stellar SSP omits them.
    # Toggling the SSP is the cleanest way to isolate the line contribution
    # while keeping the same Photometry observation grammar.
    ssp = SSP_WNE if with_nebular else SSP_BARE
    model = tengri.SEDModel.build(
        ssp, observation=obs, sfh=SFH, dust=DUST,
        redshift=tengri.Fixed(z),
    )
    p = dict(model.spec.sample(jax.random.PRNGKey(0)))
    return np.asarray(model.predict_photometry(p))


fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.0), sharey=True,
                         gridspec_kw={"wspace": 0.07})
x = np.arange(len(BANDS))

for ax, z in zip(axes, ZS):
    f_cont = _flux(z, with_nebular=False)
    f_full = _flux(z, with_nebular=True)
    ax.bar(x, f_cont, color="#33aa55", width=0.7,
           edgecolor="0.15", lw=0.5, label="continuum")
    ax.bar(x, f_full, color="#cc3333", width=0.7, alpha=0.55,
           edgecolor="0.15", lw=0.5, label="+ nebular")
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=9)
    ax.text(0.05, 0.92, f"z = {z:g}", transform=ax.transAxes, fontsize=10,
            color="0.15",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", lw=0.4))
    boost_mag = -2.5 * np.log10(f_cont / f_full)
    for xi, b in zip(x, boost_mag):
        if b < -0.05:
            ax.text(xi, max(f_full[xi], f_cont[xi]) * 1.05,
                    f"{abs(b):.2f} mag", ha="center", fontsize=7, color="#bb2222")
    ax.set_yscale("log")

axes[0].set_ylabel(r"$F_\nu$  [erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$]")
axes[1].legend(frameon=False, fontsize=8, loc="lower right")

fig.savefig("plot_emission_line_contamination.png", dpi=150, bbox_inches="tight")
