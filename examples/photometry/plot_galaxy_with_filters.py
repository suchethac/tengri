"""
Galaxy SED with photometric filter coverage
==============================================

What each photometric system measures depends on where its filters
sit relative to the rest-frame spectral features. We overlay six
common filter sets (GALEX *NUV*, SDSS *ugriz*, 2MASS *JHK*, WISE
*W1/W2/W3*, Euclid *YJH*, JWST NIRCam wide bands) on top of an
observed-frame star-forming galaxy SED at ``z = 0.5``.

The figure is meant as a quick reference for: which filter samples
the Balmer break, where MIR PAH features land, which JWST band picks
up rest-frame ``5500 Å`` at moderate z, etc.
"""

import warnings

import jax
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style
from tengri.observation.filters import load_filter

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

C_AA_PER_S = 2.998e18

SFH = {
    "type": "tsnorm",
    "*": tengri.FIXED,
    "peak_lbt_gyr": 2.0,
    "width_gyr": 1.5,
    "log_peak_sfr": 1.3,
    "skew": 0.2,
    "trunc": 13.0,
}
DUST = {
    "type": "two_component",
    "*": tengri.FIXED,
    "tau_diff": 0.4,
    "tau_bc": 0.6,
    "emission": {"type": "dale2014", "*": tengri.FIXED},
}

model = tengri.SEDModel.build(
    tengri.load_ssp(),
    sfh=SFH,
    dust=DUST,
    redshift=tengri.Fixed(0.5),
)
p = dict(model.spec.sample(jax.random.PRNGKey(0)))
out = model.predict_rest_sed(p)
wave_rest = np.asarray(out.wavelength)
wave_obs = wave_rest * 1.5
nu_l_nu = C_AA_PER_S / wave_obs * np.asarray(out.sed)

FILTERS_BY_GROUP = [
    ("GALEX", ["galex_fuv", "galex_nuv"], "#4477aa"),
    ("SDSS", ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"], "#66ccaa"),
    ("2MASS", ["2mass_j", "2mass_h", "2mass_ks"], "#aa9933"),
    ("Euclid", ["euclid_vis", "euclid_y", "euclid_j", "euclid_h"], "#cc4488"),
    (
        "JWST",
        ["jwst_f150w", "jwst_f200w", "jwst_f277w", "jwst_f356w", "jwst_f444w", "jwst_f770w"],
        "#aa3333",
    ),
    ("WISE", ["wise_w1", "wise_w2", "wise_w3"], "#882288"),
]

fig, (ax_sed, ax_filt) = plt.subplots(
    2,
    1,
    figsize=(9.0, 5.4),
    sharex=True,
    gridspec_kw={"height_ratios": [3, 1], "hspace": 0.04},
)

vis = (wave_obs > 1.0e3) & (wave_obs < 2e6)
ax_sed.loglog(wave_obs[vis], nu_l_nu[vis], color="0.15", lw=1.2)
ax_sed.set(ylabel=r"$\nu L_\nu$  [erg s$^{-1}$]", ylim=(1e40, 5e44))

for label, names, color in FILTERS_BY_GROUP:
    for name in names:
        try:
            f = load_filter(name)
        except Exception:
            continue
        ax_filt.fill_between(
            np.asarray(f.wave), 0, np.asarray(f.trans), color=color, alpha=0.4, lw=0
        )
    # one transparent rectangle for the legend handle
    ax_filt.fill_between([], [], color=color, alpha=0.6, label=label)

ax_filt.set(
    ylim=(0, 0.7), xlabel=r"Observed wavelength $\lambda$ [$\mathrm{\AA}$]", ylabel="transmission"
)
ax_filt.legend(frameon=False, fontsize=8, loc="upper right", ncol=2)
ax_sed.set_xlim(1.0e3, 1.5e6)

fig.savefig("plot_galaxy_with_filters.png", dpi=150, bbox_inches="tight")
