"""
Workflow: Inference Method Comparison
======================================

Compares three inference methods on identical mock data: MAP (point estimate),
geoVI/VI (variational approximation), and NUTS (gold-standard MCMC).
Shows how each method differs in capturing posterior shape and uncertainty.
MAP underestimates uncertainty; VI approximates the shape; NUTS is the reference.
This workflow demonstrates method choice tradeoffs for practitioners.
"""

from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np

from tengri import (
    Fitter,
    Fixed,
    Observation,
    Parameters,
    Photometry,
    SEDModel,
    Uniform,
    load_ssp_data,
    safe_corner,
    setup_style,
)

setup_style()

jax.config.update("jax_enable_x64", True)


# --- SSP data ---
def _find_ssp():
    """Locate SSP data from project root or docs/ (sphinx-gallery) cwd."""
    name = "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
    for p in [
        Path("data") / name,
        Path("../data") / name,
        Path("../../data") / name,
        Path("../../../data") / name,
    ]:
        if p.exists():
            return str(p)
    return None


SSP_PATH = _find_ssp()

# Locate filter cache
_FILTER_DIR = next(
    (
        str(d)
        for d in [
            Path("data/filters"),
            Path("../data/filters"),
            Path("../../data/filters"),
            Path("../../../data/filters"),
        ]
        if d.exists()
    ),
    "data/filters",
)
if SSP_PATH is None:
    raise FileNotFoundError("SSP data not found — skipping example")

ssp = load_ssp_data(SSP_PATH)

bands = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]
obs = Observation(photometry=Photometry.from_names(bands, cache_dir=_FILTER_DIR))

# --- Model ---
spec = Parameters(
    sfh_tsnorm_log_peak_sfr=Uniform(-1.0, 2.5),
    sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),
    sfh_tsnorm_width_gyr=Uniform(0.3, 5.0),
    sfh_tsnorm_skew=Uniform(-1.0, 1.5),
    sfh_tsnorm_trunc=Uniform(1.0, 10.0),
    met_logzsol=Uniform(-2.0, 0.2),
    dust_tau_bc=Uniform(0.0, 2.0),
    dust_tau_diff=Uniform(0.0, 1.5),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.1),
    mean_sfh_type="tsnorm",
)

model = SEDModel(spec, ssp, observation=obs)

# --- Generate mock photometry ---
key = jax.random.PRNGKey(42)
true_params = spec.sample(key)
true_params["sfh_tsnorm_peak_lbt_gyr"] = 2.5
true_params["sfh_tsnorm_width_gyr"] = 1.5
true_params["sfh_tsnorm_log_peak_sfr"] = 0.9
true_params["sfh_tsnorm_skew"] = 0.2
true_params["dust_tau_bc"] = 0.6
true_params["dust_tau_diff"] = 0.4
mock = model.mock(true_params, snr=20.0, key=key)

# --- Fit 1: MAP ---
fitter = Fitter(model, data=mock.flux_obs, noise=mock.noise)
post_map = fitter.run("map", optimizer="adam", n_steps=400, verbose=False)

# --- Fit 2: geoVI/VI ---
fitter.compile(verbose=False)
post_vi = fitter.run(
    "vi",
    n_iterations=15,
    n_samples=4,
    n_posterior_samples=2000,
    verbose=False,
)

# --- Fit 3: NUTS ---
post_nuts = fitter.run(
    "mcmc_nuts",
    n_steps=200,
    n_walkers=2,
    n_warmup=100,
    verbose=False,
)

# --- Plot 1: SFH comparison (MAP vs VI vs NUTS) ---
sfh_true = model.predict_sfh(true_params)
sfh_map = model.predict_sfh(post_map.params)
sfh_vi = model.predict_sfh(post_vi.params)
sfh_nuts = model.predict_sfh(post_nuts.params)

fig_sfh, ax_sfh = plt.subplots(figsize=(9, 5))

t_gyr_true = np.array(sfh_true["t_gyr"])
mask = t_gyr_true < 5.0

ax_sfh.plot(
    t_gyr_true[mask],
    np.array(sfh_true["sfr_mean"])[mask],
    "k-",
    lw=2.5,
    label="Truth",
)
ax_sfh.plot(
    t_gyr_true[mask],
    np.array(sfh_map["sfr_mean"])[mask],
    "--",
    color="C3",
    lw=1.8,
    label="MAP (point estimate)",
)
ax_sfh.plot(
    t_gyr_true[mask],
    np.array(sfh_vi["sfr_mean"])[mask],
    "--",
    color="C1",
    lw=1.8,
    label="geoVI (variational)",
)
ax_sfh.plot(
    t_gyr_true[mask],
    np.array(sfh_nuts["sfr_mean"])[mask],
    "--",
    color="C0",
    lw=1.8,
    label="NUTS (gold standard)",
)

ax_sfh.set_xlabel("Lookback time [Gyr]", fontsize=12)
ax_sfh.set_ylabel("SFR [Msun/yr]", fontsize=12)
ax_sfh.set_title("Method Comparison: SFH Recovery", fontsize=12, fontweight="bold")
ax_sfh.legend(fontsize=10, frameon=False, loc="upper right", lw=2.0)
ax_sfh.set_ylim(bottom=0)

fig_sfh.tight_layout()
plt.savefig("plot_workflow_method_comparison_sfh.png", dpi=150, bbox_inches="tight")

# --- Plot 2: Corner plot (VI with NUTS sample points overlaid) ---
fig_corner = safe_corner(post_vi, truths=true_params)
if fig_corner is not None:
    fig_corner.suptitle(
        "geoVI posterior (2D) with truth and MAP (dashed)",
        y=1.02,
        fontsize=12,
        fontweight="bold",
    )

    # Overlay MAP as vertical/horizontal lines on 1D/2D axes
    map_vals = [float(post_map.params[p]) for p in spec.free_params]
    n = len(spec.free_params)
    n_axes = int(np.ceil(np.sqrt(len(fig_corner.axes))))
    if n_axes > 0:
        axes = np.array(fig_corner.axes).reshape(n_axes, n_axes)
    else:
        axes = np.array(fig_corner.axes)

    for i in range(n):
        if i < n_axes:
            label_str = "MAP" if i == 0 else ""
            axes[i, i].axvline(map_vals[i], color="C3", ls=":", lw=1.5, label=label_str)
            for j in range(i):
                if j < n_axes:
                    axes[i, j].scatter(
                        [map_vals[j]], [map_vals[i]], color="C3", s=30, marker="x", zorder=5
                    )
    if n_axes > 0 and len(fig_corner.axes) > n_axes * n_axes - n_axes:
        axes[0, 0].legend(fontsize=9)

    fig_corner.tight_layout()
    plt.savefig("plot_workflow_method_comparison_corner.png", dpi=150, bbox_inches="tight")

plt.show()
