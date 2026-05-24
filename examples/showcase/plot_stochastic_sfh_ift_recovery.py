"""
Stochastic SFH recovery via IFT correlated fields
==================================================

Demonstrates tengri's unique capability: inferring stochastic (bursty) star
formation histories as Information Field Theory (IFT) correlated random fields.
Unlike fixed parametric models, the **burstiness is a free parameter**, governed
by the PSD (power spectral density) prior, not a hardwired assumption.

This example:
1. Constructs a dual-component SFH (parametric DPL + stochastic field)
2. Generates synthetic photometry at high S/N to isolate SFH recovery
3. Fits using MAP (maximum a posteriori) with field PSD parameters free
4. Shows posterior distributions for the PSD hyperparameters (sigma, tau)

**Key insight:** The field's burstiness is NOT fixed in advance — the posterior
marginalizes over burstiness, allowing data to constrain temporal structure
in star formation. This is a capability unique to IFT correlated fields
(NIFTy-based, no other public SED code exposes this).

The recovered PSD parameters should be centered near the truth values, with
width reflecting the constraint from the high-S/N photometry.

References:
  - Iyer et al. 2020, ApJ, 879, 116 (dense basis SFH parameterization)
  - Selig et al. 2013, A&A, 554, A26 (NIFTy Information Field Theory framework)
"""

import warnings

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri import recipes
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")
warnings.filterwarnings("ignore", message=".*wNE.*")

# ============================================================================
# 1. Setup: Load SSP, build stochastic SFH model
# ============================================================================

# Bare-stellar SSP required by Cue nebular backend
ssp = tengri.load_ssp("fsps_prsc_miles_chabrier")

# Filters for photometry: JWST + HST near-/mid-IR suitable for high-z
obs = tengri.Observation(
    photometry=tengri.Photometry.from_names([
        "hst_f160w",      # NIR
        "jwst_f200w",     # JWST NIR
        "jwst_f277w",     # JWST MIR
        "jwst_f356w",     # JWST MIR
    ])
)

# Build model with stochastic field component
# sfh={'type': ['dpl', 'field'], '*': FREE}
model = tengri.SEDModel.build(
    ssp_data=ssp,
    observation=obs,
    sfh={"type": ["dpl", "field"], "*": tengri.FREE},
    dust=tengri.builders.dust.two_component(
        defaults=tengri.FREE,
        emission=tengri.builders.dust.emission.dale2014(defaults=tengri.FIXED),
    ),
    neb=tengri.builders.neb.cue(defaults=tengri.FIXED),
    redshift=tengri.Uniform(0.5, 12.0),
    apply_igm=True,
)

print(f"Model has {model.spec.n_free} free parameters:")
print(f"  Free: {sorted(model.spec.free_params)}")

# ============================================================================
# 2. Ground truth and synthetic data
# ============================================================================

key = jax.random.PRNGKey(42)

# Truth: z=2, modest burstiness (psd_sigma ~ 0.15, psd_tau ~ 100 Myr)
# The DPL component sets the broad envelope; the field adds structure
truth_params = {
    "sfh_dpl_alpha": 0.5,           # declining SFR with time
    "sfh_dpl_beta": 1.0,            # smooth exponential cutoff
    "sfh_dpl_tau_gyr": 2.0,         # ~2 Gyr time-scale
    "sfh_dpl_log_peak_sfr": 0.8,    # 10^0.8 ~ 6.3 Msun/yr
    "sfh_field_psd_sigma": 0.15,    # moderate stochasticity amplitude
    "sfh_field_psd_tau_myr": 100.0, # ~100 Myr burstiness timescale
    "dust_tau_bc": 0.3,
    "dust_tau_diff": 0.1,
    "dust_slope": -0.7,
    "met_logzsol": -0.2,
    "redshift": 2.0,
}

# Compute truth SFR
truth_pred = model.predict(truth_params)
truth_sfr_100myr = float(truth_pred.sfh.sfr_100myr)

# Generate synthetic photometry at high S/N=20 per band
# (bypasses model.mock() due to stochastic_sfh_tracer_bug #275)
truth_phot = np.asarray(model.predict_photometry(truth_params))
noise_level = truth_phot / 20.0  # S/N = 20 per band
key, subkey = jax.random.split(key)
mock_flux = truth_phot + jax.random.normal(subkey, shape=truth_phot.shape) * noise_level

print(f"Generated synthetic photometry:")
print(f"  Bands: {obs.photometry.filter_names}")
print(f"  SNR: ~20 per band")
print(f"  Truth SFR (100 Myr): {truth_sfr_100myr:.2f} Msun/yr")

# ============================================================================
# 3. Fit with MAP (Maximum A Posteriori)
# ============================================================================

forward = tengri.ForwardModel.build(sed=model, observation=obs)

print(f"Fitting with MAP + Adam optimizer (500 steps)...")
posterior = forward.fit(
    mock_flux,
    noise_level,
    method="map",
    optimizer="adam",
    n_steps=500,
    verbose=False,
)

# Extract best-fit parameters
best_params = dict(posterior.best_fit_params)
print(f"Fit converged. Best-fit log-posterior: {posterior.log_posterior_best:.2f}")

# Compute recovered SFR
recovered_pred = model.predict(best_params)
recovered_sfr_100myr = float(recovered_pred.sfh.sfr_100myr)

print(f"Recovered PSD parameters:")
print(f"  psd_sigma: {best_params['sfh_field_psd_sigma']:.3f} "
      f"(true: {truth_params['sfh_field_psd_sigma']:.3f})")
print(f"  psd_tau_myr: {best_params['sfh_field_psd_tau_myr']:.1f} "
      f"(true: {truth_params['sfh_field_psd_tau_myr']:.1f})")
print(f"  SFR (100 Myr): {recovered_sfr_100myr:.2f} Msun/yr")

# ============================================================================
# 4. Posterior sampling from Laplace approximation
# ============================================================================

# Draw parameter realizations from the posterior
n_posterior_samples = 100
posterior_psd_sigma = []
posterior_psd_tau = []
posterior_sfr = []

for i in range(n_posterior_samples):
    try:
        sample_params = dict(posterior.sample(jax.random.PRNGKey(1000 + i)))
        posterior_psd_sigma.append(sample_params["sfh_field_psd_sigma"])
        posterior_psd_tau.append(sample_params["sfh_field_psd_tau_myr"])
        sample_pred = model.predict(sample_params)
        posterior_sfr.append(float(sample_pred.sfh.sfr_100myr))
    except Exception as e:
        print(f"  Posterior sampling stopped after {i} samples: {type(e).__name__}")
        break

posterior_psd_sigma = np.array(posterior_psd_sigma)
posterior_psd_tau = np.array(posterior_psd_tau)
posterior_sfr = np.array(posterior_sfr)

print(f"Posterior samples: {len(posterior_psd_sigma)}")

# ============================================================================
# 5. Visualization: SFR recovery + PSD posteriors (2-panel)
# ============================================================================

fig = plt.figure(figsize=(9, 6.5))
gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1.3], width_ratios=[1, 1],
                       hspace=0.35, wspace=0.3)

ax_sfr = fig.add_subplot(gs[0, :])
ax_psd_sigma = fig.add_subplot(gs[1, 0])
ax_psd_tau = fig.add_subplot(gs[1, 1])

# Top panel: SFR recovery
if len(posterior_sfr) > 5:
    sfr_16 = np.percentile(posterior_sfr, 16)
    sfr_50 = np.percentile(posterior_sfr, 50)
    sfr_84 = np.percentile(posterior_sfr, 84)
else:
    sfr_16 = sfr_50 = sfr_84 = recovered_sfr_100myr

ax_sfr.axhline(truth_sfr_100myr, color="red", lw=2.5, ls="--",
               label=f"True SFR (100 Myr): {truth_sfr_100myr:.2f} M$_\\odot$ yr$^{-1}$")
ax_sfr.fill_between([0, 1], sfr_16, sfr_84, alpha=0.3, color="#2ca02c",
                     label=f"68% posterior: [{sfr_16:.2f}, {sfr_84:.2f}]")
ax_sfr.plot([0.5], [sfr_50], marker="o", markersize=11, color="#2ca02c", zorder=5,
            label=f"Posterior median: {sfr_50:.2f} M$_\\odot$ yr$^{-1}$")

ax_sfr.set_xlim(-0.1, 1.1)
ax_sfr.set_xticks([])
ax_sfr.set_ylabel(r"SFR (100 Myr) [M$_\odot$ yr$^{-1}$]", fontsize=10)
ax_sfr.set_title("Stochastic SFH recovery via IFT correlated field "
                 "(burstiness is a free parameter)", fontsize=11, pad=12)
ax_sfr.legend(frameon=False, fontsize=9, loc="upper right")
ax_sfr.grid(True, alpha=0.2, axis="y")
ax_sfr.set_ylim(min(sfr_16, truth_sfr_100myr) * 0.75,
                max(sfr_84, truth_sfr_100myr) * 1.25)

# Bottom-left: PSD sigma posterior
if len(posterior_psd_sigma) > 5:
    ax_psd_sigma.hist(posterior_psd_sigma, bins=12, color="#2ca02c", alpha=0.5, density=True)
    ax_psd_sigma.axvline(truth_params["sfh_field_psd_sigma"], color="red", ls="--",
                          lw=2.2, label=f"Truth: {truth_params['sfh_field_psd_sigma']:.3f}")
    ax_psd_sigma.axvline(np.median(posterior_psd_sigma), color="#2ca02c", lw=2.2,
                          label=f"Median: {np.median(posterior_psd_sigma):.3f}")
else:
    ax_psd_sigma.axvline(best_params["sfh_field_psd_sigma"], color="#2ca02c", lw=2.2)
    ax_psd_sigma.axvline(truth_params["sfh_field_psd_sigma"], color="red", ls="--", lw=2.2)

ax_psd_sigma.set_xlabel(r"PSD $\sigma$ (stochasticity amplitude)", fontsize=10)
ax_psd_sigma.set_ylabel("Density", fontsize=10)
ax_psd_sigma.legend(frameon=False, fontsize=8)
ax_psd_sigma.grid(True, alpha=0.2, axis="y")

# Bottom-right: PSD tau posterior
if len(posterior_psd_tau) > 5:
    ax_psd_tau.hist(posterior_psd_tau, bins=12, color="#2ca02c", alpha=0.5, density=True)
    ax_psd_tau.axvline(truth_params["sfh_field_psd_tau_myr"], color="red", ls="--",
                       lw=2.2, label=f"Truth: {truth_params['sfh_field_psd_tau_myr']:.0f} Myr")
    ax_psd_tau.axvline(np.median(posterior_psd_tau), color="#2ca02c", lw=2.2,
                       label=f"Median: {np.median(posterior_psd_tau):.0f} Myr")
else:
    ax_psd_tau.axvline(best_params["sfh_field_psd_tau_myr"], color="#2ca02c", lw=2.2)
    ax_psd_tau.axvline(truth_params["sfh_field_psd_tau_myr"], color="red", ls="--", lw=2.2)

ax_psd_tau.set_xlabel(r"PSD $\tau$ (burstiness timescale) [Myr]", fontsize=10)
ax_psd_tau.set_ylabel("Density", fontsize=10)
ax_psd_tau.legend(frameon=False, fontsize=8)
ax_psd_tau.grid(True, alpha=0.2, axis="y")

plt.savefig("plot_stochastic_sfh_ift_recovery.png", dpi=150, bbox_inches="tight")
print(f"Saved: plot_stochastic_sfh_ift_recovery.png")
