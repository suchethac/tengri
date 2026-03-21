"""
Convergence Diagnostics
=======================

Runs a quick fit and displays convergence diagnostics: ESS per parameter,
summary table, and trace plots for a subset of parameters.
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

# --- Model + mock ---
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
key = jax.random.PRNGKey(7)
true_params = spec.sample(key)
mock = model.mock(true_params, snr=20.0, key=key)

# --- Fit with native_geovi ---
fitter = Fitter(model, mock.flux_obs, mock.noise, data_type="photometry")
fitter.run("map", n_steps=300, verbose=False)
fitter.compile(verbose=False)
posterior = fitter.run(
    "native_geovi",
    n_iterations=10,
    n_samples=4,
    n_seeds=3,
    n_posterior_samples=3000,
    verbose=False,
)

# --- Diagnostics ---
print(posterior.summary_table())
ess = posterior.effective_sample_size()

# --- Figure: ESS bar chart + trace plots ---
names = spec.free_params
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Left: ESS per parameter
ess_vals = [ess.get(n, 0.0) for n in names]
colors = ["#2ecc71" if e > 100 else "#e74c3c" for e in ess_vals]
short_names = [n.replace("sfh_tsnorm_", "").replace("_", " ") for n in names]
axes[0].barh(short_names, ess_vals, color=colors)
axes[0].axvline(100, color="k", ls="--", lw=0.8, label="ESS = 100 threshold")
axes[0].set_xlabel("Effective sample size")
axes[0].set_title("ESS per parameter")
axes[0].legend(fontsize=7)

# Right: trace plot for first 3 params
for i, name in enumerate(names[:3]):
    samples = np.array(posterior.samples[name])
    axes[1].plot(samples, alpha=0.6, lw=0.4, label=short_names[i])
axes[1].set_xlabel("Sample index")
axes[1].set_ylabel("Parameter value")
axes[1].set_title("Trace plots (first 3 parameters)")
axes[1].legend(fontsize=7)

fig.tight_layout()
outdir = Path(__file__).resolve().parent.parent / "figures" if "__file__" in dir() else Path(".")
outdir.mkdir(parents=True, exist_ok=True)
plt.savefig(str(outdir / "convergence.png"), dpi=150, bbox_inches="tight")
plt.show()
