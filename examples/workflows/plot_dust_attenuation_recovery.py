"""
Recovering dust attenuation from optical-NIR photometry
========================================================

Dust attenuates stellar radiation in the UV-optical and reddens the
continuum in the NIR. This example simulates optical and NIR photometry
from a dusty star-forming galaxy, adds realistic noise, and recovers the
dust attenuation parameters using MAP optimization. The figure compares
the true and recovered SEDs with a photometric residual panel.

Reference: Conroy 2013, ARA&A, 51, 393 (SED fitting overview);
Calzetti et al. 2000, ApJ, 533, 682 (attenuation law).
"""

import warnings

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

# Use bare-stellar SSP (nebular disabled in model below)
ssp = tengri.load_ssp("fsps_prsc_miles_chabrier")

# Use SDSS + 2MASS to span optical-NIR where dust emission affects stellar colors
obs = tengri.Observation(
    photometry=tengri.Photometry.from_names([
        "sdss_g", "sdss_r", "sdss_i",
        "2mass_j", "2mass_h", "2mass_ks",
    ])
)

# Build a dust-focused model: free attenuation optical depths, fixed SFH
model = tengri.SEDModel.build(
    ssp,
    observation=obs,
    sfh={"type": "tsnorm", "*": tengri.FIXED,
         "log_peak_sfr": 1.0, "peak_lbt_gyr": 0.5, "width_gyr": 0.3,
         "skew": 0.1, "trunc": 1.0},
    dust={
        "type": "two_component",
        "law_bc": "calzetti",
        "*": tengri.FREE,  # tau_bc and tau_diff free
        "slope": -0.7,
    },
    neb={"type": "none", "*": tengri.FIXED},
    redshift=tengri.Fixed(0.1),
)

# Generate truth with modest dust optical depths
key = jax.random.PRNGKey(42)
truth = dict(model.spec.sample(key))
truth.update(
    dust_tau_bc=0.4,    # Birth cloud optical depth (stellar continuum attenuation)
    dust_tau_diff=0.2,  # Diffuse ISM optical depth
)

# Mock photometry at S/N ~ 20 (optical) to 10 (NIR)
snr = np.array([20.0, 20.0, 18.0, 15.0, 13.0, 12.0])  # Per-band S/N
mock = model.mock(truth, snr=snr, key=key)

# Fit with MAP (fast, suitable for gallery)
forward = tengri.ForwardModel.build(sed=model, observation=obs)
posterior = forward.fit(
    mock.flux_obs,
    mock.noise,
    method="map",
    optimizer="adam",
    n_steps=300,
    verbose=False,
)
fit_params = posterior.params

# Extract photometry and residuals
flux_truth = np.asarray(mock.flux_true)
flux_fit = np.asarray(model.predict_photometry(fit_params))
flux_obs = np.asarray(mock.flux_obs)
noise = np.asarray(mock.noise)
wave_eff = np.array([float(jnp.mean(w)) for w in obs.photometry.filter_waves])

# Compute rest-frame SED (at z=0.1, FIR is mostly unshifted relative to rest-frame)
sed_truth = model.predict_rest_sed(truth)
sed_fit = model.predict_rest_sed(fit_params)
wave_rest = np.asarray(sed_truth.wavelength)
z = 0.1
wave_obs = wave_rest * (1.0 + z)

# Scale SED to match photometry at a reference band (PACS green, 100 μm)
idx_ref = np.argmin(np.abs(wave_obs - wave_eff[1]))
scale = flux_truth[1] / np.asarray(sed_truth.sed)[idx_ref]
fnu_truth = scale * np.asarray(sed_truth.sed)
fnu_fit = scale * np.asarray(sed_fit.sed)

# Two-panel layout: SED + residuals
fig, (ax_sed, ax_res) = plt.subplots(
    2, 1,
    figsize=(7.0, 5.2),
    sharex=True,
    gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
)

# Top: SED with data overlay (optical to NIR, where dust emission dominates)
vis = (wave_obs > 3000) & (wave_obs < 2.5e4)
ax_sed.plot(wave_obs[vis], fnu_truth[vis], color="0.55", lw=0.8,
            label="Truth SED")
ax_sed.plot(wave_obs[vis], fnu_fit[vis], color="C3", lw=0.8, alpha=0.8,
            label="MAP SED")
ax_sed.errorbar(
    wave_eff,
    flux_obs,
    yerr=noise,
    fmt="o",
    color="k",
    ms=5,
    capsize=2,
    label="Mock optical+NIR",
)
ax_sed.plot(wave_eff, flux_truth, "s", color="0.2", ms=6, mfc="none",
            mew=1.2, label="Truth bands")
ax_sed.plot(wave_eff, flux_fit, "^", color="C3", ms=6, mfc="none",
            mew=1.2, label="MAP bands")
ax_sed.set_yscale("log")
ax_sed.set_ylabel(r"$F_\nu$  [erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$]")
ax_sed.legend(frameon=False, fontsize=8, ncol=2, loc="lower left")

# Bottom: residuals in units of noise
residual = (flux_fit - flux_obs) / noise
ax_res.axhline(0.0, color="0.5", lw=0.6)
ax_res.axhspan(-1.0, 1.0, color="0.85", alpha=0.6, lw=0)
ax_res.plot(wave_eff, residual, "o", color="C3", ms=5)
ax_res.set_ylim(-3.5, 3.5)
ax_res.set_ylabel(r"$(F_{\rm fit} - F_{\rm obs})/\sigma$")
ax_res.set_xlabel(r"Observed wavelength $\lambda$ [$\mathrm{\AA}$]")

# Save figure to the same directory as this script
import pathlib
script_dir = pathlib.Path(__file__).parent
plt.savefig(script_dir / "plot_dust_attenuation_recovery.png", dpi=150, bbox_inches="tight")

# Validation: check recovery within ~1σ
tau_bc_truth = float(truth["dust_tau_bc"])
tau_bc_fit = float(fit_params["dust_tau_bc"])
tau_diff_truth = float(truth["dust_tau_diff"])
tau_diff_fit = float(fit_params["dust_tau_diff"])
print(f"Dust tau_bc:  truth={tau_bc_truth:.3f}, fit={tau_bc_fit:.3f}")
print(f"Dust tau_diff: truth={tau_diff_truth:.3f}, fit={tau_diff_fit:.3f}")
print(f"Photometric residual σ: {np.std(residual):.2f}")
