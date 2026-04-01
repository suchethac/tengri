"""
Corner Plot with Truth Overlay
==============================

Fits mock photometry and displays a corner plot with injected truth
values marked. Uses tengri's safe_corner utility.
"""

from pathlib import Path

import jax
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

from tengri import (
    Fitter,
    Fixed,
    Model,
    Observation,
    ParamSpec,
    Photometry,
    Uniform,
    load_ssp_data,
    safe_corner,
    setup_style,
)

setup_style()


# --- Data ---
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

if SSP_PATH is None:
    raise FileNotFoundError("SSP data not found — skipping example")

ssp = load_ssp_data(SSP_PATH)
obs = Observation(
    photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
)

# --- Model + mock ---
spec = ParamSpec(
    sfh_tsnorm_log_peak_sfr=Uniform(-1.0, 2.5),
    sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),
    sfh_tsnorm_width_gyr=Uniform(0.3, 5.0),
    met_logzsol=Uniform(-2.0, 0.2),
    dust_tau_diff=Uniform(0.0, 1.5),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.1),
    mean_sfh_type="tsnorm",
)
model = Model(spec, ssp, observation=obs)
key = jax.random.PRNGKey(99)
true_params = spec.sample(key)
# Override to ensure star-forming galaxy
true_params["sfh_tsnorm_peak_lbt_gyr"] = 3.0
true_params["sfh_tsnorm_width_gyr"] = 2.0
true_params["sfh_tsnorm_log_peak_sfr"] = 1.0
mock = model.mock(true_params, snr=25.0, key=key)

# --- Fit ---
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

# --- Corner plot ---
fig = safe_corner(posterior, truths=true_params)
if fig is not None:
    fig.suptitle("Posterior corner plot (truth = blue lines)", y=1.02)

outdir = Path(__file__).resolve().parent.parent / "figures" if "__file__" in dir() else Path(".")
outdir.mkdir(parents=True, exist_ok=True)
plt.savefig(str(outdir / "corner.png"), dpi=150, bbox_inches="tight")
plt.show()
