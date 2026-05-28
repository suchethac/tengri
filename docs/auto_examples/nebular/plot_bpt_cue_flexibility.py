"""
Cue nebular knobs affect BPT positions individually
====================================================

The Cue neural emulator responds to 12+ parameters. We show how each knob
(ionization, metallicity, density, abundances, ionizing slope) moves a
galaxy on the BPT-N plane ``log [OIII]/Hβ`` vs ``log [NII]/Hα``. Each
panel sweeps one parameter while holding fiducial values fixed. Kewley+2001
and Kauffmann+2003 demarcations shown for reference.
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
warnings.filterwarnings("ignore", message=".*deprecated.*")

ssp = tengri.load_ssp("fsps_prsc_miles_chabrier")

sweeps = {
    r"$\log U$": ("neb_logU", np.linspace(-4.0, -1.8, 7)),
    r"$\log Z_{\rm gas}$": ("neb_logZ_gas", np.linspace(-1.5, 0.4, 7)),
    r"$\log n_{\rm H}$": ("neb_n_h", np.linspace(1.0, 1e3, 7)),
}

nh_grid = np.linspace(-2.0, 0.45, 200)
log_oiii_hb_kewley = 0.61 / (nh_grid - 0.47) + 1.19
log_oiii_hb_kauff = 0.61 / (nh_grid - 0.05) + 1.3

fig, axes = plt.subplots(1, 3, figsize=(13, 4))

for ax, (_label, (param_name, param_values)) in zip(axes, sweeps.items()):
    log_n2_ha_points = []
    log_o3_hb_points = []

    for pval in param_values:
        spec_dict = {
            "sfh": {
                "type": "dpl",
                "*": tengri.FIXED,
                "alpha": 1.0,
                "beta": 2.5,
                "tau_gyr": 0.05,
                "log_total_mass": 10.0,
            },
            "dust": {"type": "two_component", "*": tengri.FIXED, "tau_diff": 0.0, "tau_bc": 0.0},
            "neb": {
                "type": "cue",
                "*": tengri.FIXED,
                param_name: tengri.Fixed(pval),
            },
            "redshift": tengri.Fixed(0.05),
        }
        model = tengri.SEDModel.build(ssp, **spec_dict)
        params = dict(model.spec.sample(jax.random.PRNGKey(0)))
        out = model.predict_rest_sed(params)
        lines = out.emission_lines

        if lines is not None:
            lines_dict = dict(lines)
            ha = lines_dict.get(6562.79, 1e-20)
            hb = lines_dict.get(4860.2, 1e-20)
            nii = lines_dict.get(6583.34, 1e-20)
            oiii = lines_dict.get(5008.24, 1e-20)
            if ha > 0 and hb > 0 and oiii > 0 and nii > 0:
                log_n2_ha_points.append(np.log10(nii / ha))
                log_o3_hb_points.append(np.log10(oiii / hb))

    if log_n2_ha_points:
        cmap = plt.get_cmap("viridis")
        norm = plt.Normalize(vmin=param_values.min(), vmax=param_values.max())
        colors = cmap(norm(param_values[: len(log_n2_ha_points)]))
        ax.scatter(log_n2_ha_points, log_o3_hb_points, c=colors, s=50, edgecolors="k", lw=0.5)

    mask_k = nh_grid < 0.47
    ax.plot(nh_grid[mask_k], log_oiii_hb_kewley[mask_k], "k-", lw=1.0, alpha=0.5)
    mask_kauff = nh_grid < 0.05
    ax.plot(nh_grid[mask_kauff], log_oiii_hb_kauff[mask_kauff], "k--", lw=0.8, alpha=0.5)

    ax.set_xlabel(r"$\log$ [NII] / H$\alpha$")
    ax.set_ylabel(r"$\log$ [OIII] / H$\beta$")
    ax.set_xlim(-2.0, 0.6)
    ax.set_ylim(-1.2, 1.5)
    ax.grid(True, alpha=0.2)

fig.tight_layout()
plt.savefig("plot_bpt_cue_flexibility.png", dpi=150, bbox_inches="tight")
