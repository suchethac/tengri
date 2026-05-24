"""
Recovering shared PSD hyperparameters from a small galaxy population
=====================================================================

Four mock galaxies whose star-formation histories share the same
Fourier-field power-spectrum (sigma, tau) — a hierarchical inference
problem. We mock the photometry, fit jointly with
``tengri.PopulationFitter``, and check that the shared (sigma, tau)
posteriors bracket the input values. Composed ``[tsnorm, field]`` SFHs
are now reachable through the nested-dict grammar (``sfh.type`` accepts
a list), so the model builds via ``SEDModel.build`` like any other.

Reference: Leja et al. 2019, ApJ, 876, 3 (rapid field inference with
correlated priors).
"""

import time
import warnings

import jax
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

TRUE_SIGMA = 2.0
TRUE_TAU = 20.0
N_GAL = 4

ssp = tengri.load_ssp()
obs = tengri.Observation(
    photometry=tengri.Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
)


def model_factory(psd_sigma: float = 1.0, psd_tau_myr: float = 50.0) -> tengri.SEDModel:
    """tsnorm mean SFH modulated by a Fourier field with fixed (sigma, tau)."""
    return tengri.SEDModel.build(
        ssp,
        observation=obs,
        sfh={
            "type": ["tsnorm", "field"],
            "*": tengri.FIXED,
            "tsnorm_log_peak_sfr": tengri.Uniform(-1.0, 2.5),
            "tsnorm_peak_lbt_gyr": tengri.Uniform(1.0, 8.0),
            "tsnorm_width_gyr": tengri.Uniform(0.5, 3.0),
            "tsnorm_skew": 0.0,
            "tsnorm_trunc": 3.0,
            "field_psd_sigma": psd_sigma,
            "field_psd_tau_myr": psd_tau_myr,
        },
        dust={
            "type": "two_component",
            "*": tengri.FIXED,
            "tau_bc": 0.3,
            "tau_diff": 0.2,
            "slope": -0.7,
        },
        met_logzsol=tengri.Fixed(0.0),
        redshift=tengri.Fixed(0.1),
        n_grid=128,
    )


key = jax.random.PRNGKey(42)
model_gen = model_factory(psd_sigma=TRUE_SIGMA, psd_tau_myr=TRUE_TAU)
galaxies = []
for i in range(N_GAL):
    k = jax.random.fold_in(key, i)
    params = model_gen.spec.sample(k)
    flux_truth = np.asarray(model_gen.predict_observables(params).phot_fnu)
    noise = np.abs(flux_truth) / 20.0
    perturb = np.asarray(jax.random.normal(jax.random.fold_in(k, 1), flux_truth.shape))
    galaxies.append({"flux_obs": flux_truth + noise * perturb, "noise": noise})

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
print(f"Hierarchical fit: {time.perf_counter() - t0:.1f}s")

keys = list(result.shared_samples.keys())
sig_key = next(k for k in keys if "psd" in k and ("sigma" in k or "_u" in k or "amp" in k))
tau_key = next(k for k in keys if "psd" in k and "tau" in k)
sig_samples = np.asarray(result.shared_samples[sig_key])
tau_samples = np.asarray(result.shared_samples[tau_key])

fig, (ax_sig, ax_tau) = plt.subplots(1, 2, figsize=(10.0, 3.8))
ax_sig.hist(sig_samples, bins=30, density=True, alpha=0.7, color="steelblue")
ax_sig.axvline(TRUE_SIGMA, color="crimson", ls="--", lw=1.5, label=f"Truth = {TRUE_SIGMA}")
ax_sig.set_xlabel(r"$\sigma_{\rm PSD}$")
ax_sig.set_ylabel("Density")
ax_sig.legend(frameon=False, fontsize=9)

ax_tau.hist(tau_samples, bins=30, density=True, alpha=0.7, color="steelblue")
ax_tau.axvline(TRUE_TAU, color="crimson", ls="--", lw=1.5, label=f"Truth = {TRUE_TAU} Myr")
ax_tau.set_xlabel(r"$\tau_{\rm PSD}$  [Myr]")
ax_tau.set_ylabel("Density")
ax_tau.legend(frameon=False, fontsize=9)

fig.tight_layout()
plt.savefig("plot_hierarchical.png", dpi=150, bbox_inches="tight")
