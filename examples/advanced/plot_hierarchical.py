"""
Hierarchical PSD inference across a small population
====================================================

Generate a population of mock galaxies whose star-formation histories share
the same Fourier-field power-spectrum (sigma, tau). Then fit them jointly
with PopulationFitter and recover the shared PSD hyperparameters via
hierarchical Bayesian shrinkage.

The stochastic-SFH path still uses the flat-Parameters constructor (the
"expert escape hatch"), because the nested-dict ``SEDModel.build`` grammar
does not yet expose composed ``[tsnorm, field]`` SFHs as of 2026-05.

Reference: Leja et al. 2019 (rapid field inference with correlated priors).
"""

import time
import warnings

import jax
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri import Fixed, Parameters, SEDModel, Uniform
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

ssp = tengri.load_ssp()
obs = tengri.Observation(
    photometry=tengri.Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
)

TRUE_SIGMA = 2.0
TRUE_TAU = 20.0
N_GAL = 4


def model_factory(psd_sigma: float = 1.0, psd_tau_myr: float = 50.0) -> SEDModel:
    """tsnorm mean SFH modulated by a Fourier `field` with fixed (sigma, tau)."""
    spec = Parameters(
        mean_sfh_type=["tsnorm", "field"],
        sfh_tsnorm_log_peak_sfr=Uniform(-1.0, 2.5),
        sfh_tsnorm_peak_lbt_gyr=Uniform(1.0, 8.0),
        sfh_tsnorm_width_gyr=Uniform(0.5, 3.0),
        sfh_tsnorm_skew=Fixed(0.0),
        sfh_tsnorm_trunc=Fixed(3.0),
        sfh_field_psd_sigma=Fixed(psd_sigma),
        sfh_field_psd_tau_myr=Fixed(psd_tau_myr),
        met_logzsol=Fixed(0.0),
        dust_tau_bc=Fixed(0.3),
        dust_tau_diff=Fixed(0.2),
        dust_slope=Fixed(-0.7),
        redshift=Fixed(0.1),
        n_grid=128,
    )
    return SEDModel(spec, ssp, observation=obs)


key = jax.random.PRNGKey(42)
model_gen = model_factory(psd_sigma=TRUE_SIGMA, psd_tau_myr=TRUE_TAU)
galaxies = []
for i in range(N_GAL):
    k = jax.random.fold_in(key, i)
    params = model_gen.spec.sample(k)
    # SEDModel.mock() currently traces through n_grid under stochastic SFH; we
    # build the mock photometry manually from predict_observables for now.
    flux_truth = np.asarray(model_gen.predict_observables(params).phot_fnu)
    noise = np.abs(flux_truth) / 20.0
    perturb = np.asarray(jax.random.normal(jax.random.fold_in(k, 1), flux_truth.shape))
    galaxies.append({"flux_obs": flux_truth + noise * perturb, "noise": noise})
print(f"Generated {N_GAL} mock galaxies with sigma={TRUE_SIGMA}, tau={TRUE_TAU} Myr")

hfitter = tengri.PopulationFitter(
    model_factory,
    galaxies,
    psd_sigma_prior=(0.1, 4.0),
    psd_tau_prior=(1.0, 300.0),
)

t0 = time.perf_counter()
result = hfitter.run(
    "vi_linear",
    n_iterations=20,
    n_samples=4,
    n_posterior_samples=500,
    verbose=False,
    key=jax.random.PRNGKey(0),
)
elapsed = time.perf_counter() - t0
print(f"Hierarchical fit: {elapsed:.1f}s")

keys = list(result.shared_samples.keys())
sig_key = next(k for k in keys if "psd" in k and ("sigma" in k or "_u" in k or "amp" in k))
tau_key = next(k for k in keys if "psd" in k and "tau" in k)
sig_samples = np.array(result.shared_samples[sig_key])
tau_samples = np.array(result.shared_samples[tau_key])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.hist(sig_samples, bins=30, density=True, alpha=0.7, color="steelblue")
ax1.axvline(TRUE_SIGMA, color="crimson", ls="--", lw=2, label=f"Truth = {TRUE_SIGMA}")
ax1.set_xlabel(r"$\sigma_{\rm PSD}$")
ax1.set_ylabel("Density")
ax1.legend(frameon=False)

ax2.hist(tau_samples, bins=30, density=True, alpha=0.7, color="steelblue")
ax2.axvline(TRUE_TAU, color="crimson", ls="--", lw=2, label=f"Truth = {TRUE_TAU} Myr")
ax2.set_xlabel(r"$\tau_{\rm PSD}$ [Myr]")
ax2.set_ylabel("Density")
ax2.legend(frameon=False)

fig.tight_layout()
plt.savefig("plot_hierarchical.png", dpi=150, bbox_inches="tight")
