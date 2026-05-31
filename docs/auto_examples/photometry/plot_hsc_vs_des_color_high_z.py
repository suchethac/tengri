"""
HSC vs DES filter i-band differences at high redshift
======================================================

The Subaru HSC and Blanco DECam i-bands have different red-edge cutoffs
(HSC i-2 at ~850 nm, DECam i at ~870 nm). This 20 nm difference produces
measurable colour offsets when a sharp spectral feature sweeps through
the i-band — particularly the Lyman break at z~3.5–4.5. We show (r − i)
colours for an LBG template across both filter sets to highlight the
divergence in the high-redshift regime.
"""

import jax
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style

setup_style()


def _flux(model, params):
    """Photometric flux prediction."""
    return np.asarray(model.predict_photometry(params))


# Baseline LBG model: young, dusty, star-forming
ssp = tengri.load_ssp("fsps_prsc_miles_chabrier")

obs_hsc = tengri.Observation(photometry=tengri.Photometry.from_names(["hsc_r", "hsc_i"]))
obs_des = tengri.Observation(photometry=tengri.Photometry.from_names(["des_r", "des_i"]))

# Two LBG-like populations
models = {}
for pop_name, peak_lbt_gyr, tau_bc in [
    ("Dusty LBG", 1.2, 0.6),
    ("Less dusty", 1.2, 0.2),
]:
    base_spec = {
        "sfh": {
            "type": "tsnorm",
            "*": tengri.FIXED,
            "peak_lbt_gyr": peak_lbt_gyr,
            "width_gyr": 1.0,
            "log_total_mass": 10.0,
            "skew": 0.0,
            "trunc": 13.0,
        },
        "dust": {
            "type": "two_component",
            "*": tengri.FIXED,
            "tau_bc": tau_bc,
            "tau_diff": 0.1,
            "slope": -0.7,
        },
        "redshift": tengri.Uniform(2.5, 5.0),
    }
    models[pop_name] = {
        "hsc": tengri.SEDModel.build(ssp, observation=obs_hsc, **base_spec),
        "des": tengri.SEDModel.build(ssp, observation=obs_des, **base_spec),
    }

z_grid = np.linspace(2.8, 4.8, 120)
fig, ax = plt.subplots(figsize=(6.4, 4.5))

colors_plot = {"Dusty LBG": "#CC3333", "Less dusty": "#3377CC"}

for pop_name, pop_models in models.items():
    baseline_hsc = dict(pop_models["hsc"].spec.sample(jax.random.PRNGKey(42)))
    baseline_des = dict(pop_models["des"].spec.sample(jax.random.PRNGKey(42)))

    ri_hsc = np.empty_like(z_grid)
    ri_des = np.empty_like(z_grid)

    for i, z in enumerate(z_grid):
        params_hsc = {**baseline_hsc, "redshift": float(z)}
        params_des = {**baseline_des, "redshift": float(z)}

        flux_hsc = _flux(pop_models["hsc"], params_hsc)
        flux_des = _flux(pop_models["des"], params_des)

        ri_hsc[i] = -2.5 * np.log10(flux_hsc[0] / flux_hsc[1])
        ri_des[i] = -2.5 * np.log10(flux_des[0] / flux_des[1])

    # HSC: solid line, DES: dashed
    ax.plot(z_grid, ri_hsc, color=colors_plot[pop_name], lw=2.0, label=f"{pop_name} (HSC)")
    ax.plot(
        z_grid,
        ri_des,
        color=colors_plot[pop_name],
        lw=2.0,
        linestyle="--",
        label=f"{pop_name} (DECam)",
    )

# Shade the Lyman break window (z ≈ 3.5–4.5)
ax.axvspan(3.5, 4.5, alpha=0.08, color="gray", label="Lyman break zone")

ax.set(xlabel=r"Redshift", ylabel=r"$r - i$  [AB mag]")
ax.legend(frameon=False, fontsize=8.5, loc="best")
ax.grid(False)

fig.tight_layout()
plt.savefig("plot_hsc_vs_des_color_high_z.png", dpi=150, bbox_inches="tight")
