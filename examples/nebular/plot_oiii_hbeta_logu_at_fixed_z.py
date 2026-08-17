r"""
[O III]/Hβ vs ionization parameter at fixed gas metallicity
===========================================================

The optical [O III] 5007 / Hβ ratio is set primarily by the
ionization parameter ``log U``: more energetic Lyman continuum
photons per H atom ionize more O+ to O++, while Hβ recombination
depends mostly on the ionizing photon rate (``Q_H``) and is roughly
insensitive to ``log U``. The ratio therefore rises monotonically
with ``log U`` at fixed gas metallicity.

We sweep ``log U`` from -3 to -1 at three sub-solar metallicities
covered by the Cue emulator (Li+2025), with dust-on-lines and LyC
escape switched off so the trend isolates the photoionization
response cleanly.

Reference: Kewley & Dopita 2002 ApJS 142 35; Li et al. 2025, ApJ, 986, 9
(Cue neural emulator; arXiv:2405.04598).
"""

import warnings

import jax
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore")

ssp = tengri.load_ssp("fsps_prsc_miles_chabrier")
logz_values = np.array([-1.5, -1.0, -0.5])
logu_grid = np.linspace(-3.0, -1.0, 11)

fig, ax = plt.subplots(figsize=(6.8, 4.3))
markers = ["o", "s", "^"]
cmap = plt.get_cmap("viridis")

for k, logz in enumerate(logz_values):
    model = tengri.SEDModel.build(
        ssp,
        sfh={
            "type": "dpl",
            "all_params": tengri.FIXED,
            "alpha": 1.0,
            "beta": 2.5,
            "tau_gyr": 0.03,
            "log_total_mass": 10.0,
        },
        dust={"type": "two_component", "all_params": tengri.FIXED, "tau_diff": 0.0, "tau_bc": 0.0},
        neb={
            "type": "cue",
            "all_params": tengri.FIXED,
            "logZ_gas": tengri.Fixed(logz),
            "logU": tengri.Uniform(-3.0, -1.0),
            "fesc": tengri.Fixed(0.0),
        },
        redshift=tengri.Fixed(0.05),
    )
    baseline = dict(model.spec.sample(jax.random.PRNGKey(0)))
    ratio = []
    for u in logu_grid:
        p = {**baseline, "neb_logU": np.float64(u)}
        lines = model.predict(p).lines
        o3 = float(lines.oiii_5007)
        hb = float(lines.hbeta)
        ratio.append(o3 / hb if hb > 0 else np.nan)
    ratio = np.array(ratio)
    good = np.isfinite(ratio)
    ax.plot(
        logu_grid[good],
        ratio[good],
        marker=markers[k],
        ms=5,
        lw=1.5,
        color=cmap(0.15 + 0.7 * k / (len(logz_values) - 1)),
        label=rf"$\log Z_{{\rm gas}}/Z_\odot = {logz:+.1f}$",
    )

ax.set(
    xlabel=r"Ionization parameter $\log U$",
    ylabel=r"$[\mathrm{O\,III}]\,5007 \,/\, \mathrm{H}\beta$",
    xlim=(-3.05, -0.95),
)
ax.legend(frameon=False, fontsize=9, loc="upper left")
fig.tight_layout()
plt.savefig("plot_oiii_hbeta_logu_at_fixed_z.png", dpi=150, bbox_inches="tight")
