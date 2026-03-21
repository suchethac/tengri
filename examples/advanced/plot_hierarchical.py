"""
Hierarchical PSD Inference
===========================

Sets up a small population of mock galaxies sharing the same burstiness
PSD parameters (sigma, tau), runs HierarchicalFitter briefly, and
displays the shared PSD posterior vs truth.
"""

import os
import sys
import time

import jax
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

from tengri import (
    Fixed,
    HierarchicalFitter,
    Model,
    ParamSpec,
    Uniform,
    load_filter_set,
    load_ssp_data,
    setup_style,
)

setup_style()

# --- Data ---
SSP_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "data",
    "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5",
)
if not os.path.exists(SSP_PATH):
    sys.exit("SSP data not found — skipping")

ssp = load_ssp_data(SSP_PATH)
filters = load_filter_set(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])

# --- True shared PSD ---
TRUE_SIGMA = 2.0
TRUE_TAU = 20.0
N_GAL = 4


def model_factory(psd_sigma=1.0, psd_tau_myr=50.0):
    """Create a Model with fixed PSD — called by HierarchicalFitter."""
    spec = ParamSpec(
        sfh_tsnorm_log_peak_sfr=Uniform(-1.0, 2.5),
        sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),
        sfh_tsnorm_width_gyr=Uniform(0.3, 5.0),
        sfh_tsnorm_skew=Uniform(-3.0, 3.0),
        sfh_tsnorm_trunc=Uniform(1.0, 10.0),
        sfh_field_psd_sigma=Fixed(psd_sigma),
        sfh_field_psd_tau_myr=Fixed(psd_tau_myr),
        met_logzsol=Uniform(-2.0, 0.2),
        dust_tau_bc=Uniform(0.0, 2.0),
        dust_tau_diff=Uniform(0.0, 1.5),
        dust_slope=Fixed(-0.7),
        redshift=Fixed(0.1),
        mean_sfh_type=["tsnorm", "field"],
        n_grid=128,
    )
    return Model(spec, ssp, filters=filters)


# --- Generate mock galaxies ---
key = jax.random.PRNGKey(42)
model_gen = model_factory(psd_sigma=TRUE_SIGMA, psd_tau_myr=TRUE_TAU)
galaxies = []
for i in range(N_GAL):
    k = jax.random.fold_in(key, i)
    params = model_gen.spec.sample(k)
    mock = model_gen.mock(params, snr=20.0, key=jax.random.fold_in(k, 1))
    galaxies.append({"flux_obs": mock.flux_obs, "noise": mock.noise})
print(f"Generated {N_GAL} mock galaxies with sigma={TRUE_SIGMA}, tau={TRUE_TAU} Myr")

# --- Hierarchical fit (quick) ---
hfitter = HierarchicalFitter(
    model_factory,
    galaxies,
    psd_sigma_prior=(0.1, 4.0),
    psd_tau_prior=(1.0, 300.0),
    data_type="photometry",
)

t0 = time.perf_counter()
result = hfitter.run(
    "evi",
    n_iterations=20,
    n_samples=4,
    n_posterior_samples=500,
    n_seeds=5,
    verbose=False,
    key=jax.random.PRNGKey(0),
)
elapsed = time.perf_counter() - t0
print(f"Hierarchical fit: {elapsed:.1f}s")

# --- Figure: shared PSD posterior ---
sig_samples = np.array(result.shared_samples["psd_sigma"])
tau_samples = np.array(result.shared_samples["psd_tau_myr"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.hist(sig_samples, bins=30, density=True, alpha=0.7, color="steelblue")
ax1.axvline(TRUE_SIGMA, color="crimson", ls="--", lw=2, label=f"Truth = {TRUE_SIGMA}")
ax1.set_xlabel(r"$\sigma_{\rm PS}$")
ax1.set_ylabel("Density")
ax1.set_title("Shared PSD amplitude")
ax1.legend()

ax2.hist(tau_samples, bins=30, density=True, alpha=0.7, color="steelblue")
ax2.axvline(TRUE_TAU, color="crimson", ls="--", lw=2, label=f"Truth = {TRUE_TAU} Myr")
ax2.set_xlabel(r"$\tau_{\rm PS}$ [Myr]")
ax2.set_ylabel("Density")
ax2.set_title("Shared PSD timescale")
ax2.legend()

fig.suptitle(f"Hierarchical PSD recovery ({N_GAL} galaxies, {elapsed:.0f}s)")
fig.tight_layout()

outdir = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(outdir, exist_ok=True)
plt.savefig(os.path.join(outdir, "hierarchical.png"), dpi=150, bbox_inches="tight")
plt.show()
