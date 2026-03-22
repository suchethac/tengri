"""
MAP vs geoVI Posterior Comparison
=================================

Compares point-estimate (MAP) and variational (native_geovi) inference
on mock 5-band photometry. Overlays posteriors as a corner plot.
"""

from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

from tengri import (
    Fitter,
    Fixed,
    Model,
    ParamSpec,
    Uniform,
    load_filter_set,
    load_ssp_data,
    safe_corner,
    setup_style,
)

setup_style()


# --- Data ---
def _find_ssp():
    """Locate SSP data from project root or docs/ (sphinx-gallery) cwd."""
    name = "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
    for p in [Path("data") / name, Path("../data") / name,
              Path("../../data") / name, Path("../../../data") / name]:
        if p.exists():
            return str(p)
    return None


SSP_PATH = _find_ssp()

# Locate filter cache
_FILTER_DIR = next(
    (str(d) for d in [Path("data/filters"), Path("../data/filters"),
     Path("../../data/filters"), Path("../../../data/filters")] if d.exists()),
    "data/filters",
)
if SSP_PATH is None:
    raise FileNotFoundError("SSP data not found — skipping example")

ssp = load_ssp_data(SSP_PATH)
filters = load_filter_set(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"], cache_dir=_FILTER_DIR)

# --- Model ---
spec = ParamSpec(
    sfh_tsnorm_log_peak_sfr=Uniform(-1.0, 2.5),
    sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),
    sfh_tsnorm_width_gyr=Uniform(0.3, 5.0),
    sfh_tsnorm_skew=Uniform(-3.0, 3.0),
    sfh_tsnorm_trunc=Uniform(1.0, 10.0),
    met_logzsol=Uniform(-2.0, 0.2),
    dust_tau_bc=Uniform(0.0, 2.0),
    dust_tau_diff=Uniform(0.0, 1.5),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.1),
    mean_sfh_type="tsnorm",
)
model = Model(spec, ssp, filters=filters)

# --- Mock photometry ---
key = jax.random.PRNGKey(42)
true_params = spec.sample(key)
mock = model.mock(true_params, snr=20.0, key=key)

# --- Fit: MAP ---
fitter = Fitter(model, mock.flux_obs, mock.noise, data_type="photometry")
result_map = fitter.run("map", n_steps=300, verbose=False)

# --- Fit: native_geovi (quick settings) ---
fitter.compile(verbose=False)
result_geovi = fitter.run(
    "native_geovi",
    n_iterations=10,
    n_samples=4,
    n_seeds=3,
    n_posterior_samples=2000,
    verbose=False,
)

# --- Figure: corner comparison ---
fig = safe_corner(result_geovi, truths=true_params)
if fig is not None:
    # Mark MAP point
    map_vals = [float(result_map.params[p]) for p in spec.free_params]
    n = len(spec.free_params)
    axes = np.array(fig.axes).reshape(n, n)
    for i in range(n):
        axes[i, i].axvline(map_vals[i], color="C3", ls="--", lw=1.2, label="MAP")
        for j in range(i):
            axes[i, j].axhline(map_vals[i], color="C3", ls="--", lw=0.8)
            axes[i, j].axvline(map_vals[j], color="C3", ls="--", lw=0.8)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("MAP (dashed red) vs native_geovi posteriors", y=1.02)

outdir = Path(__file__).resolve().parent.parent / "figures" if "__file__" in dir() else Path(".")
outdir.mkdir(parents=True, exist_ok=True)
plt.savefig(str(outdir / "method_comparison.png"), dpi=150, bbox_inches="tight")
plt.show()
