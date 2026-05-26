"""
Fitting a stochastic SFH with a smooth parametric prior leaves a UV residual
============================================================================

A common SED-fitting failure mode: pick a smooth parametric SFH (delayed
exponential, tau-model, lognormal) for a galaxy whose true star-formation
history has short-timescale bursts. The continuum-anchored bands (optical,
NIR) absorb the mass and the fit looks plausible — but the UV bands, where
young O/B stars dominate, carry the residual of the recent burst.

We mock a galaxy with a *dpl + field* stochastic SFH (mean double-power-law
modulated by a damped-random-walk Gaussian process), then fit it with a
*dexp* (delayed exponential) — the same physics minus the burstiness. The
top panel shows the rest-frame SED of truth and fit; the bottom panel shows
broadband residuals divided by the photometric noise. UV bands sit
systematically off, NIR bands recover.

The misspecification penalty is the structure in the residuals, not the
chi-squared total — which is why purely numerical convergence checks miss
this failure mode (Leja et al. 2019; Carnall et al. 2019; Iyer et al. 2019).
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

BANDS = [
    "galex_fuv", "galex_nuv",
    "sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z",
    "2mass_j", "2mass_h", "2mass_ks",
]
Z = 0.05
SNR = 30.0  # generous: the trap should be visible above shot noise.

obs = tengri.Observation(photometry=tengri.Photometry.from_names(BANDS))
ssp = tengri.load_ssp()

# ─── Truth: dpl mean + field modulator (stochastic, bursty) ──────────────────
truth_model = tengri.SEDModel.build(
    ssp,
    observation=obs,
    sfh={
        "type": ["dpl", "field"],
        "*": tengri.FIXED,
        # Smooth backbone — peaks ~2 Gyr ago, falls toward present.
        "alpha": 3.0, "beta": 2.0, "tau_gyr": 2.0, "log_peak_sfr": 1.0,
        # Stochastic burstiness — DRW with sigma ~ 0.8 dex on an 80 Myr
        # timescale gives recent-burst structure clearly visible in UV.
        "psd_sigma": 0.8, "psd_tau_myr": 80.0,
    },
    dust={"type": "two_component", "*": tengri.FIXED, "tau_diff": 0.25, "tau_bc": 0.3},
    redshift=tengri.Fixed(Z),
)

# Seed picked so the GP realisation has a recent (<~100 Myr) excursion —
# without a recent feature there is no trap to diagnose.
key_truth = jax.random.PRNGKey(7)
truth = dict(truth_model.spec.sample(key_truth))
mock = truth_model.mock(truth, snr=SNR, key=jax.random.PRNGKey(0))

# ─── Wrong model: smooth delayed exponential, no field ───────────────────────
fit_model = tengri.SEDModel.build(
    ssp,
    observation=obs,
    sfh={
        "type": "dexp",
        "*": tengri.FREE,
        "start_gyr": tengri.Fixed(0.0),
    },
    dust={
        "type": "two_component", "*": tengri.FIXED,
        "tau_diff": tengri.Uniform(0.0, 1.5),
        "tau_bc": tengri.Uniform(0.0, 1.5),
    },
    redshift=tengri.Fixed(Z),
)

forward = tengri.ForwardModel.build(sed=fit_model, observation=obs)
posterior = forward.fit(
    mock.flux_obs, mock.noise,
    method="map", optimizer="adam", n_steps=400, verbose=False,
)
fit_params = posterior.params

# ─── Photometry + rest-frame SED for plotting ────────────────────────────────
flux_obs = np.asarray(mock.flux_obs)
noise = np.asarray(mock.noise)
flux_fit = np.asarray(fit_model.predict_photometry(fit_params))
wave_eff = np.array([float(jnp.mean(w)) for w in obs.photometry.filter_waves])

sed_truth = truth_model.predict_rest_sed(truth)
sed_fit = fit_model.predict_rest_sed(fit_params)
wave_rest = np.asarray(sed_truth.wavelength)
wave_obs = wave_rest * (1.0 + Z)


def _scale_to_r(sed_arr):
    idx = np.argmin(np.abs(wave_obs - wave_eff[BANDS.index("sdss_r")]))
    return flux_obs[BANDS.index("sdss_r")] / sed_arr[idx]


fnu_truth = _scale_to_r(np.asarray(sed_truth.sed)) * np.asarray(sed_truth.sed)
fnu_fit = _scale_to_r(np.asarray(sed_fit.sed)) * np.asarray(sed_fit.sed)

# ─── Figure: SED + residuals ────────────────────────────────────────────────
fig, (ax_sed, ax_res) = plt.subplots(
    2, 1, figsize=(7.0, 5.4), sharex=True,
    gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
)

vis = (wave_obs > 1.3e3) & (wave_obs < 2.5e4)
ax_sed.plot(wave_obs[vis], fnu_truth[vis], color="0.4", lw=0.9,
            label="Truth (dpl + stochastic field)")
ax_sed.plot(wave_obs[vis], fnu_fit[vis], color="C3", lw=0.9, alpha=0.85,
            label="MAP fit (dexp, smooth)")
ax_sed.errorbar(wave_eff, flux_obs, yerr=noise, fmt="o", color="k",
                ms=4.5, capsize=2, label=f"Mock (S/N = {SNR:.0f})")
ax_sed.set_xscale("log")
ax_sed.set_yscale("log")
ax_sed.set_ylabel(r"$F_\nu$  [erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$]")
ax_sed.legend(frameon=False, fontsize=8, loc="lower right")

residual = (flux_fit - flux_obs) / noise
band_colour = ["C0" if w < 4e3 else "0.3" for w in wave_eff]
ax_res.axhspan(-1, 1, color="0.85", alpha=0.5, zorder=0)
ax_res.axhline(0.0, color="0.4", lw=0.6)
for w, r, c in zip(wave_eff, residual, band_colour):
    ax_res.plot(w, r, "o", color=c, ms=6)
ax_res.set_xscale("log")
ax_res.set_xlabel(r"Observed wavelength  [$\mathrm{\AA}$]")
ax_res.set_ylabel(r"$(F_\mathrm{fit} - F_\mathrm{obs}) / \sigma$")
ylim = max(4.0, 1.2 * float(np.abs(residual).max()))
ax_res.set_ylim(-ylim, ylim)
ax_res.axvspan(1e3, 4e3, color="C0", alpha=0.08, zorder=-1)
ax_res.text(2e3, ylim * 0.75, "UV: model misspecification",
            color="C0", fontsize=8, ha="center", va="top")

fig.savefig("plot_wrong_model_trap.png", dpi=150, bbox_inches="tight")
