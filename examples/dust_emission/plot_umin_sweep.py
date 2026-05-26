r"""
U_min: DL07 and THEMIS agree on the FIR peak
==============================================

Minimum radiation field intensity :math:`U_{\min}` controls diffuse dust
heating. Higher :math:`U_{\min}` implies hotter dust and an FIR peak
shifted blueward. This figure builds two `SEDModel`s — one with the
Draine & Li 2007 dust-emission backend, one with THEMIS (Jones et al.
2017) — and overlays their rest-frame predictions at matched
:math:`U_{\min}` values to show the two libraries agree on the peak
location (~100 μm).
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

C_AA_PER_S = 2.998e18

u_min_values = [0.5, 2.0, 5.0, 15.0]


def build_model(emission_type: str) -> tengri.SEDModel:
    """Build a stellar+dust model whose dust-emission backend is selectable."""
    return tengri.SEDModel.build(
        ssp_data=tengri.load_ssp(),
        sfh={"type": "dpl", "*": tengri.FIXED, "log_peak_sfr": 1.0,
             "tau_gyr": 1.0, "alpha": 2.0, "beta": 2.5},
        dust={
            "type": "two_component",
            "*": tengri.FIXED,
            "tau_diff": 1.0,
            "tau_bc": 0.5,
            "emission": {
                "type": emission_type,
                "*": tengri.FIXED,
                "umin": tengri.Uniform(0.1, 30.0),
            },
        },
        redshift=tengri.Fixed(0.0),
    )


model_dl07 = build_model("draine_li2007")
model_themis = build_model("themis")

baseline_dl07 = dict(model_dl07.spec.sample(jax.random.PRNGKey(0)))
baseline_themis = dict(model_themis.spec.sample(jax.random.PRNGKey(0)))


def nu_l_nu(model: tengri.SEDModel, baseline: dict, u_min: float):
    params = {**baseline, "dust_umin": float(u_min)}
    pred = model.predict_rest_sed(params)
    wave = np.asarray(pred.wavelength)
    sed = np.asarray(pred.sed)
    return wave, (C_AA_PER_S / wave) * sed


fig, ax = plt.subplots(figsize=(9.5, 6.0))
cmap = plt.get_cmap("plasma")

for k, u_min in enumerate(u_min_values):
    color = cmap(k / max(1, len(u_min_values) - 1))

    # DL07 (solid)
    wave_dl07_aa, nl_dl07 = nu_l_nu(model_dl07, baseline_dl07, u_min)
    wave_dl07_um = wave_dl07_aa * 1.0e-4
    fir_mask = wave_dl07_um > 5.0
    ax.plot(
        wave_dl07_um,
        nl_dl07 / np.max(nl_dl07[fir_mask]),
        color=color, lw=1.6, linestyle="-",
    )

    # THEMIS (dashed)
    wave_th_aa, nl_th = nu_l_nu(model_themis, baseline_themis, u_min)
    wave_th_um = wave_th_aa * 1.0e-4
    fir_mask = wave_th_um > 5.0
    ax.plot(
        wave_th_um,
        nl_th / np.max(nl_th[fir_mask]),
        color=color, lw=1.6, linestyle="--",
    )

# Linestyle legend (DL07 vs THEMIS)
from matplotlib.lines import Line2D
ls_legend = [
    Line2D([0], [0], color="k", linestyle="-", lw=1.6, label="DL07"),
    Line2D([0], [0], color="k", linestyle="--", lw=1.6, label="THEMIS"),
]
ax.legend(handles=ls_legend, loc="upper right", frameon=False, fontsize=9)

# U_min colorbar
norm = plt.Normalize(vmin=u_min_values[0], vmax=u_min_values[-1])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.02)
cbar.set_label(r"$U_{\rm min}$", fontsize=10)

ax.set(
    xscale="log",
    yscale="log",
    xlabel=r"$\lambda\ [\mu\mathrm{m}]$",
    ylabel=r"$\nu L_\nu / \mathrm{peak}$ (normalised)",
    xlim=(5.0, 1.0e3),
    ylim=(1e-3, 2.0),
    title=r"$U_{\rm min}$: DL07 and THEMIS agree on the FIR peak ($\sim 100\ \mu\mathrm{m}$)",
)

fig.tight_layout()
plt.savefig("plot_umin_sweep.png", dpi=150, bbox_inches="tight")
