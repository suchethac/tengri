"""
Photo-z degeneracy in color–color space: low-z dusty vs high-z quiescent
===========================================================================

Two galaxies with very different star formation histories and dust can
collide in color–color space, making photo-z ambiguous. Here, a young
dusty star-forming galaxy at z≈0.5 and an old quiescent galaxy at z≈2
follow nearly identical (u-g, g-r) tracks and intersect at a single point.
This shows why intermediate-wavelength photometry is essential for robust
photo-z classification.
"""

import warnings

import jax
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.plot import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")


obs = tengri.Observation(photometry=tengri.Photometry.from_names(["sdss_u", "sdss_g", "sdss_r"]))


def _build_template(peak_lbt, width, tau_diff):
    """Build a model at a given SFH and dust."""
    return tengri.SEDModel.build(
        tengri.load_ssp("fsps_prsc_miles_chabrier"),
        observation=obs,
        sfh={
            "type": "tsnorm",
            "all_params": tengri.Fixed(tengri.DEFAULT),
            "peak_lbt_gyr": peak_lbt,
            "width_gyr": width,
        },
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "all_params": tengri.Fixed(tengri.DEFAULT),
            "tau_diff": tau_diff,
            "tau_bc": 0.3,
        },
        redshift=tengri.Uniform(0.01, 3.5),
    )


model_dust = _build_template(2.0, 2.0, 0.8)
model_old = _build_template(10.5, 1.0, 0.05)


def _colors(model, params, z_grid):
    """Compute u-g and g-r colors across redshifts."""
    ug, gr = [], []
    for z in z_grid:
        p = {**params, "redshift": float(z)}
        flux = np.asarray(model.predict_photometry(p))
        ug.append(-2.5 * np.log10(flux[0] / flux[1]))
        gr.append(-2.5 * np.log10(flux[1] / flux[2]))
    return np.array(ug), np.array(gr)


z_grid = np.linspace(0.01, 3.2, 120)
p_dust = dict(model_dust.spec.sample(jax.random.PRNGKey(0)))
p_old = dict(model_old.spec.sample(jax.random.PRNGKey(1)))

ug_dust, gr_dust = _colors(model_dust, p_dust, z_grid)
ug_old, gr_old = _colors(model_old, p_old, z_grid)

fig, ax = plt.subplots(figsize=(6.5, 5.4))
ax.plot(ug_dust, gr_dust, color="#d62728", lw=2.0, label="Dusty young (z∼0.5)", zorder=3)
ax.plot(ug_old, gr_old, color="#1f77b4", lw=2.0, label="Old quiescent (z∼2)", zorder=3)

for z_m, color in [(0.5, "#d62728"), (2.0, "#1f77b4")]:
    idx = int(np.argmin(np.abs(z_grid - z_m)))
    ug, gr = (ug_dust, gr_dust) if color == "#d62728" else (ug_old, gr_old)
    j = min(idx + 5, len(z_grid) - 1)
    ax.arrow(
        ug[idx],
        gr[idx],
        ug[j] - ug[idx],
        gr[j] - gr[idx],
        head_width=0.08,
        head_length=0.05,
        fc=color,
        ec=color,
        alpha=0.6,
        zorder=2,
    )

# Mark collision point
dist = np.sqrt((ug_dust - ug_old) ** 2 + (gr_dust - gr_old) ** 2)
i_int = np.argmin(dist)
ax.scatter(
    ug_dust[i_int],
    gr_dust[i_int],
    s=200,
    marker="*",
    color="gold",
    ec="k",
    lw=1.2,
    zorder=10,
    label=f"Collision at z≈{z_grid[i_int]:.2f}",
)

ax.set(
    xlabel=r"$u - g$ [AB mag]",
    ylabel=r"$g - r$ [AB mag]",
    xlim=(0.3, 2.5),
    ylim=(-0.15, 1.4),
)
ax.legend(frameon=False, fontsize=8, loc="lower right")
ax.grid(alpha=0.2)

fig.tight_layout()
plt.savefig("plot_photoz_color_degeneracy_grid.png", dpi=150, bbox_inches="tight")
