"""
AGN UV-to-X-ray SED: delta alpha_OX deviation shifts X-ray relative to disc
===========================================================================

In the X-CIGALE AGN model (Yang et al. 2020), the X-ray corona is tied to the
accretion-disc UV continuum through the α_OX–L_2500 relation. A deviation
delta_alpha_ox lets the intrinsic X-ray-to-UV ratio float around that
empirical relation: positive values brighten the corona, negative values
suppress it. Because the disc anchors the normalization, all SEDs pivot at
the EUV/soft-X-ray join; only the hard-X-ray side fans out.

We sweep delta_alpha_ox over [−0.3, +0.3] at fixed L_bol, showing the
UV-to-X-ray SED (rest-frame 100 Å – 100 keV) and the spectral pivot.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import warnings

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

_C_AA = 2.998e18
_KEV_AA = 12.398
_LOG_1P24 = np.log10(_KEV_AA / 10.0)

ssp = tengri.load_ssp()
model = tengri.SEDModel.build(
    ssp,
    sfh={"type": "const", "*": tengri.FIXED, "log_total_mass": 11.0, "start_gyr": 13.0},
    dust={"type": "two_component", "*": tengri.FIXED, "tau_diff": 0.0, "tau_bc": 0.0},
    agn={
        "disc": {"type": "qsogen", "*": tengri.FIXED},
        "torus": {"type": "none", "*": tengri.FIXED},
        "*": tengri.FIXED,
        "log_lbol": 14.0,
        "frac": 1.0,
    },
    xray={
        "type": "simple",
        "*": tengri.FIXED,
        "delta_alpha_ox": tengri.Uniform(-0.3, 0.3),
    },
    redshift=tengri.Fixed(0.0),
)
baseline = dict(model.spec.sample(jax.random.PRNGKey(0)))

delta_values = np.array([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3])
cmap = plt.get_cmap("rainbow_r")
norm = mpl.colors.Normalize(vmin=-0.3, vmax=0.3)

fig, ax = plt.subplots(figsize=(7.6, 5.2))
for delta in delta_values:
    out = model.predict_rest_sed({**baseline, "xray_delta_alpha_ox": jnp.float64(delta)})
    wave = np.asarray(out.wavelength)
    log_nm = np.log10(wave / 10.0)
    l_nu = np.asarray(out.sed)
    ax.plot(
        log_nm,
        np.log10(np.clip(l_nu, 1e-30, None)),
        color=cmap(norm(delta)),
        lw=1.8,
    )

ax.annotate(
    "", xy=(-0.85, 28.0), xytext=(0.4, 28.0), arrowprops=dict(arrowstyle="->", lw=1.4, color="k")
)
ax.text(-0.25, 28.15, "observable\nX-ray", ha="center", va="bottom", fontsize=9)

ax.set(
    xlim=(-0.9, 3.0),
    ylim=(24.5, 30.5),
    xlabel=r"$\log\lambda$ (nm, rest-frame)",
    ylabel=r"$\log L_\nu$ (cgs)",
)
ax_top = ax.secondary_xaxis("top", functions=(lambda x: _LOG_1P24 - x, lambda lk: _LOG_1P24 - lk))
ax_top.set_xlabel(r"$\log E$ (keV, rest-frame)")

cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.01)
cbar.set_label(r"$\Delta\alpha_{\rm OX}$ [dex]")

fig.tight_layout()
plt.savefig("plot_xray_delta_alpha_ox_sed.png", dpi=150, bbox_inches="tight")
