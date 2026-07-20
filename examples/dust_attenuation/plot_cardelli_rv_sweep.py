"""
Cardelli MW attenuation: sweeping R_V

The Cardelli+1989 Milky Way attenuation curve is a family
parameterized by ``R_V = A_V / E(B-V)``. Smaller R_V (≲ 3) gives a
steeper UV rise and stronger 2175 Å bump (denser lines of sight,
small grains dominate); larger R_V (≳ 4.5) flattens the UV slope
(processed grains, larger sizes).

Sweep over the canonical range R_V ∈ [2.0, 5.5] showing the family
on the same intrinsic SED at τ_V = 1.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import jax
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

C_AA_PER_S = 2.998e18
ssp = tengri.load_ssp()
SFH = {
    "type": "tsnorm",
    "all_params": tengri.FIXED,
    "peak_lbt_gyr": 0.05,
    "width_gyr": 0.05,
    "log_total_mass": 10.0,
    "skew": 0.0,
    "trunc": 13.0,
}


def _model(rv):
    return tengri.SEDModel.build(
        ssp,
        sfh=SFH,
        dust={
            "type": "two_component",
            "all_params": tengri.FIXED,
            "tau_diff": 1.0,
            "tau_bc": 0.0,
            "law_diff": "cardelli",
            "Rv": rv,
        },
        redshift=tengri.Fixed(0.05),
    )


rv_grid = np.linspace(2.0, 5.5, 8)
norm = mpl.colors.Normalize(vmin=rv_grid.min(), vmax=rv_grid.max())
cmap = plt.get_cmap("viridis")

fig, ax = plt.subplots(figsize=(6.6, 4.4))
for rv in rv_grid:
    model = _model(float(rv))
    p = dict(model.spec.sample(jax.random.PRNGKey(0)))
    out = model.predict(p)
    wave = np.asarray(model.wavelengths)
    nu_l_nu = C_AA_PER_S / wave * np.asarray(out.rest_sed())
    ax.loglog(wave, nu_l_nu, color=cmap(norm(rv)), lw=1.4)

ax.axvline(2175, color="0.55", lw=0.4, ls=":")
ax.text(2175, 8e43, "2175 Å bump", fontsize=8, color="0.4", ha="right", rotation=90)
ax.set(
    xlim=(900, 1e4),
    ylim=(1e41, 1e44),
    xlabel=r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]",
    ylabel=r"$\nu L_\nu$  [erg s$^{-1}$]",
)
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.01)
cb.set_label(r"$R_V = A_V / E(B-V)$")

fig.tight_layout()
plt.savefig("plot_cardelli_rv_sweep.png", dpi=150, bbox_inches="tight")
