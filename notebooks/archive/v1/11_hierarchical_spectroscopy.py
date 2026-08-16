# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Hierarchical PSD Recovery from Spectroscopy
#
# Tutorial 05 demonstrated hierarchical PSD recovery from **photometry**.
# Here we use **spectroscopy** (SDSS-like, 3800--9200 A, 200 pixels),
# which provides $\sim$40x more data points per galaxy and constrains
# burstiness through spectral features (D4000, Balmer lines, UV slope).
#
# We show:
# 1. Individual galaxy spectroscopic fits with posterior contours
# 2. Hierarchical EVI (JIT) recovery of shared PSD
# 3. Comparison: photometry-only vs spectroscopy
# 4. Scaling with number of galaxies
#
# > **Key result:** spectroscopy breaks the $\sigma$--$\tau$ degeneracy
# > that photometry alone cannot resolve.

# %%
import warnings

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)
warnings.filterwarnings("ignore")

import os
import sys

sys.path.insert(0, ".")
from _plot_style import setup_style

setup_style()
os.makedirs("notebook_figures", exist_ok=True)

from tengri import (
    Fixed,
    Fitter,
    HierarchicalFitter,
    SEDModel,
    ParamSpec,
    Uniform,
    load_filter_set,
    load_ssp_data,
)

key = jax.random.PRNGKey(42)

# %% [markdown]
# ## Setup: Data and SEDModel
#
# We use a stochastic SFH model (DPL mean + GP field) with 128 GP grid
# points, giving D$\sim$137 free parameters per galaxy. The spectroscopic
# wavelength grid covers 3800--9200 A at 200 pixels ($\sim$27 A/pixel).

# %%
ssp = load_ssp_data("../data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")
filters = load_filter_set(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])

# Spectroscopic wavelength grid (SDSS-like)
WAVE_OBS = jnp.linspace(3800.0, 9200.0, 200)
N_PIX = len(WAVE_OBS)

# Truth
TRUE_SIGMA = 1.5
TRUE_TAU = 50.0  # Myr


def model_factory(psd_sigma=1.0, psd_tau_myr=50.0):
    """SEDModel factory for hierarchical fitting.

    HierarchicalFitter calls this with shared PSD values.
    Returns a SEDModel with those PSD params FIXED.
    """
    spec = ParamSpec(
        sfh_dpl_alpha=Uniform(0.5, 3.0),
        sfh_dpl_beta=Uniform(0.5, 3.0),
        sfh_dpl_tau_gyr=Uniform(0.5, 13.0),
        sfh_dpl_log_peak_sfr=Uniform(-1.0, 2.0),
        sfh_field_psd_sigma=Fixed(psd_sigma),
        sfh_field_psd_tau_myr=Fixed(psd_tau_myr),
        met_logzsol=Uniform(-2.0, 0.5),
        dust_tau_bc=Uniform(0.0, 2.0),
        dust_tau_diff=Uniform(0.0, 2.0),
        dust_slope=Fixed(-0.7),
        redshift=Fixed(0.1),
        mean_sfh_type=["dpl", "field"],
        n_grid=128,
    )
    m = SEDModel(spec, ssp, filters=filters)
    m.precompute_spectroscopy(WAVE_OBS)
    return m


model = model_factory(psd_sigma=TRUE_SIGMA, psd_tau_myr=TRUE_TAU)
print(f"Wavelength: {float(WAVE_OBS[0]):.0f}--{float(WAVE_OBS[-1]):.0f} A, {N_PIX} pixels")
print(f"Per-galaxy: D = {model.spec.n_free} physical + 128 GP = ~137 free params")
print(f"True PSD: sigma={TRUE_SIGMA}, tau={TRUE_TAU} Myr")

# %% [markdown]
# ## Generate Mock Population

# %%
N_GAL = 10
SNR = 30.0

galaxies_spec = []
galaxies_phot = []
mock_params_list = []
mock_spectra_true = []

for i in range(N_GAL):
    k = jax.random.fold_in(key, i)
    params = model.spec.sample(k)
    # Spectroscopic mock
    mock_s = model.mock_spectrum(params, WAVE_OBS, snr=SNR, key=jax.random.fold_in(k, 1))
    galaxies_spec.append({"flux_obs": mock_s.flux_obs, "noise": mock_s.noise})
    mock_spectra_true.append(mock_s.flux_true)
    # Photometric mock (same galaxy)
    mock_p = model.mock(params, snr=20.0, key=jax.random.fold_in(k, 2))
    galaxies_phot.append({"flux_obs": mock_p.flux_obs, "noise": mock_p.noise})
    mock_params_list.append(params)

print(f"Generated {N_GAL} mock galaxies with SNR={SNR} (spec) / 20 (phot)")

# %%
# --- Show 4 example spectra ---
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
for ax, idx in zip(axes.ravel(), range(4)):
    wave = np.array(WAVE_OBS)
    flux_obs = np.array(galaxies_spec[idx]["flux_obs"])
    flux_true = np.array(mock_spectra_true[idx])
    noise = np.array(galaxies_spec[idx]["noise"])

    ax.fill_between(wave, flux_obs - noise, flux_obs + noise, alpha=0.3, color="C0")
    ax.plot(wave, flux_obs, "C0-", lw=0.5, alpha=0.7, label="Observed")
    ax.plot(wave, flux_true, "k-", lw=1.5, label="Truth")
    ax.set_xlabel(r"$\lambda_{\rm obs}$ [\AA]")
    ax.set_ylabel(r"$f_\nu$ [erg/s/cm$^2$/Hz]")
    ax.set_title(f"Galaxy {idx}", fontsize=11)
    if idx == 0:
        ax.legend(fontsize=9)

fig.suptitle(f"Mock spectra (N={N_GAL}, SNR={SNR})", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("notebook_figures/11_hierspec_fig01_spectra.png", dpi=100, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Individual Galaxy Fits
#
# Before hierarchical inference, fit each galaxy individually with EVI.
# This shows the per-galaxy posterior quality and serves as initialization.

# %%
# --- Individual fits ---
# Each galaxy creates a new Fitter (data is bound at construction).
# The XLA persistent cache (/tmp/tengri_jax_cache) means the
# compiled program is loaded from disk after the first galaxy,
# so galaxy 2+ only pay ~1s for JAX tracing, not ~15s for XLA compile.
#
# For true catalog-scale fitting (>100 galaxies), use the native
# engine directly via fitter._build_jit_engine() to avoid per-galaxy
# Fitter construction overhead.
import time as _time

individual_results = []

for i in range(min(4, N_GAL)):
    _t0 = _time.perf_counter()
    k = jax.random.fold_in(key, 1000 + i)
    fitter_i = Fitter(model, galaxies_spec[i]["flux_obs"],
                       galaxies_spec[i]["noise"], data_type="spectroscopy")
    result_i = fitter_i.run("native_geovi", n_iterations=30, n_samples=3,
                             n_posterior_samples=500, n_seeds=3,
                             key=k, verbose=False)
    _dt = _time.perf_counter() - _t0
    individual_results.append(result_i)
    chi2 = result_i.diagnostics.get("chi2_dof", "?")
    print(f"Galaxy {i}: chi2/dof={chi2}, wall={_dt:.1f}s")

# %%
# --- Posterior contours for first 2 galaxies ---
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

param_pairs = [
    ("sfh_dpl_alpha", "sfh_dpl_tau_gyr"),
    ("dust_tau_bc", "met_logzsol"),
    ("sfh_dpl_log_peak_sfr", "dust_tau_diff"),
]
param_labels = {
    "sfh_dpl_alpha": r"$\alpha_{\rm SFH}$",
    "sfh_dpl_tau_gyr": r"$\tau_{\rm peak}$ [Gyr]",
    "dust_tau_bc": r"$\tau_{\rm BC}$",
    "met_logzsol": r"$\log(Z/Z_\odot)$",
    "sfh_dpl_log_peak_sfr": r"$\log$ SFR$_{\rm peak}$",
    "dust_tau_diff": r"$\tau_{\rm diff}$",
}

for row, gal_idx in enumerate([0, 1]):
    result = individual_results[gal_idx]
    samples = result.samples
    true_p = mock_params_list[gal_idx]

    for col, (px, py) in enumerate(param_pairs):
        ax = axes[row, col]
        sx = np.array(samples[px])
        sy = np.array(samples[py])
        ax.scatter(sx, sy, c="C0", s=2, alpha=0.15, rasterized=True)
        ax.axvline(float(true_p[px]), color="C3", ls="--", lw=1.5)
        ax.axhline(float(true_p[py]), color="C3", ls="--", lw=1.5)
        ax.set_xlabel(param_labels.get(px, px), fontsize=10)
        ax.set_ylabel(param_labels.get(py, py), fontsize=10)
        if col == 0:
            ax.set_title(f"Galaxy {gal_idx}", fontsize=11)

fig.suptitle("Individual EVI posteriors (spec, SNR=30)", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("notebook_figures/11_hierspec_fig02_individual.png", dpi=100, bbox_inches="tight")
plt.show()

# %%
# --- Spectral fits: observed vs posterior predictive ---
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

for ax, gal_idx in zip(axes.ravel(), range(min(4, len(individual_results)))):
    result = individual_results[gal_idx]
    wave = np.array(WAVE_OBS)
    flux_obs = np.array(galaxies_spec[gal_idx]["flux_obs"])
    noise = np.array(galaxies_spec[gal_idx]["noise"])
    flux_true = np.array(mock_spectra_true[gal_idx])

    # Posterior predictive: draw spectra from posterior samples
    n_draws = min(50, len(next(iter(result.samples.values()))))
    spec_draws = []
    for j in range(n_draws):
        draw = {k: v[j] for k, v in result.samples.items()}
        spec_draw = model.predict_spectrum(draw, WAVE_OBS)
        spec_draws.append(np.array(spec_draw))
    spec_draws = np.array(spec_draws)
    lo, hi = np.percentile(spec_draws, [16, 84], axis=0)

    ax.fill_between(wave, lo, hi, alpha=0.3, color="C0", label="68% CI")
    ax.plot(wave, flux_obs, "k.", ms=1, alpha=0.4, label="Data")
    ax.plot(wave, flux_true, "C3-", lw=1.5, label="Truth")
    ax.plot(wave, np.median(spec_draws, axis=0), "C0-", lw=1, label="Median")

    residual = (flux_obs - np.median(spec_draws, axis=0)) / noise
    chi2_dof = np.sum(residual**2) / len(wave)
    ax.set_title(f"Galaxy {gal_idx} ($\\chi^2/\\nu={chi2_dof:.2f}$)", fontsize=11)
    ax.set_xlabel(r"$\lambda$ [\AA]")
    ax.set_ylabel(r"$f_\nu$")
    if gal_idx == 0:
        ax.legend(fontsize=8)

fig.suptitle("Posterior predictive spectra", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("notebook_figures/11_hierspec_fig03_fits.png", dpi=100, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Hierarchical EVI: Spectroscopy vs Photometry
#
# The key test: does spectroscopy provide tighter PSD constraints than
# photometry? We run hierarchical EVI on both data types and compare.

# %%
# --- Hierarchical EVI with spectroscopy ---
hfitter_spec = HierarchicalFitter(
    model_factory, galaxies_spec,
    psd_sigma_prior=(0.1, 4.0),
    psd_tau_prior=(1.0, 300.0),
    data_type="spectroscopy",
)

key, subkey = jax.random.split(key)
result_spec = hfitter_spec.run(
    "evi", key=subkey,
    n_iterations=50, n_samples=6,
    n_posterior_samples=500, n_seeds=10,
    kl_rtol=0.0,
)
print(f"\nSpectroscopy: {result_spec.wall_time_s:.1f}s")
print(result_spec.summary())

# %%
# --- Hierarchical EVI with photometry (same galaxies) ---
hfitter_phot = HierarchicalFitter(
    model_factory, galaxies_phot,
    psd_sigma_prior=(0.1, 4.0),
    psd_tau_prior=(1.0, 300.0),
    data_type="photometry",
)

key, subkey = jax.random.split(key)
result_phot = hfitter_phot.run(
    "evi", key=subkey,
    n_iterations=50, n_samples=6,
    n_posterior_samples=500, n_seeds=10,
    kl_rtol=0.0,
)
print(f"\nPhotometry: {result_phot.wall_time_s:.1f}s")
print(result_phot.summary())

# %%
# --- Compare PSD posteriors: spec vs phot ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

sig_spec = np.array(result_spec.shared_samples["psd_sigma"])
tau_spec = np.array(result_spec.shared_samples["psd_tau_myr"])
sig_phot = np.array(result_phot.shared_samples["psd_sigma"])
tau_phot = np.array(result_phot.shared_samples["psd_tau_myr"])

ax1.hist(sig_spec, bins=30, density=True, alpha=0.6, color="C0",
         edgecolor="k", linewidth=0.5, label=f"Spectroscopy (N={N_GAL})")
ax1.hist(sig_phot, bins=30, density=True, alpha=0.6, color="C1",
         edgecolor="k", linewidth=0.5, label=f"Photometry (N={N_GAL})")
ax1.axvline(TRUE_SIGMA, color="C3", ls="--", lw=2, label=f"Truth = {TRUE_SIGMA}")
ax1.set_xlabel(r"$\sigma_{\rm PSD}$", fontsize=12)
ax1.set_ylabel("Density")
ax1.set_title(r"$\sigma_{\rm PSD}$: spectroscopy vs photometry")
ax1.legend(fontsize=10)

ax2.hist(tau_spec, bins=30, density=True, alpha=0.6, color="C0",
         edgecolor="k", linewidth=0.5, label="Spectroscopy")
ax2.hist(tau_phot, bins=30, density=True, alpha=0.6, color="C1",
         edgecolor="k", linewidth=0.5, label="Photometry")
ax2.axvline(TRUE_TAU, color="C3", ls="--", lw=2, label=f"Truth = {TRUE_TAU} Myr")
ax2.set_xlabel(r"$\tau_{\rm PSD}$ [Myr]", fontsize=12)
ax2.set_ylabel("Density")
ax2.set_title(r"$\tau_{\rm PSD}$: spectroscopy vs photometry")
ax2.legend(fontsize=10)

plt.tight_layout()
plt.savefig("notebook_figures/11_hierspec_fig04_spec_vs_phot.png", dpi=100, bbox_inches="tight")
plt.show()

# Quantitative comparison
for label, sig_s, tau_s in [("Spec", sig_spec, tau_spec), ("Phot", sig_phot, tau_phot)]:
    sig_lo, sig_hi = np.percentile(sig_s, [16, 84])
    tau_lo, tau_hi = np.percentile(tau_s, [16, 84])
    print(f"{label}: sigma={np.median(sig_s):.2f} [{sig_lo:.2f}, {sig_hi:.2f}], "
          f"tau={np.median(tau_s):.1f} [{tau_lo:.1f}, {tau_hi:.1f}] Myr")

# %% [markdown]
# ## Individual Galaxy SFH Recovery from Hierarchical Fit

# %%
# --- SFH recovery for 4 galaxies from hierarchical spectroscopic fit ---
if result_spec.individual_samples is not None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    for ax, idx in zip(axes.ravel(), range(min(4, N_GAL))):
        true_p = mock_params_list[idx]
        sfh_true = model.predict_sfh(true_p)
        t_gyr = np.array(sfh_true["t_gyr"])

        # Posterior SFH draws
        ind_samples = result_spec.individual_samples[idx]
        n_draws = min(50, len(next(iter(ind_samples.values()))))
        sfr_draws = []
        for j in range(n_draws):
            draw = {k: v[j] for k, v in ind_samples.items()}
            # Add shared PSD (not needed for SFH prediction since xi is provided)
            draw["sfh_field_psd_sigma"] = TRUE_SIGMA
            draw["sfh_field_psd_tau_myr"] = TRUE_TAU
            try:
                sfh_draw = model.predict_sfh(draw)
                sfr_draws.append(np.array(sfh_draw["sfr_mean"]))
            except Exception:
                pass

        if sfr_draws:
            sfr_draws = np.array(sfr_draws)
            lo, hi = np.percentile(sfr_draws, [16, 84], axis=0)
            ax.fill_between(t_gyr, lo, hi, alpha=0.3, color="C0", label="68% CI")
            ax.plot(t_gyr, np.median(sfr_draws, axis=0), "C0--", lw=1.5, label="Median")

        ax.plot(t_gyr, np.array(sfh_true["sfr_mean"]), "k-", lw=2, label="Truth")
        ax.set_xlabel("Lookback time [Gyr]")
        ax.set_ylabel(r"SFR [$M_\odot$/yr]")
        ax.set_yscale("log")
        ax.set_title(f"Galaxy {idx}", fontsize=11)
        if idx == 0:
            ax.legend(fontsize=9)

    fig.suptitle("SFH recovery from hierarchical spectroscopic fit", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig("notebook_figures/11_hierspec_fig05_sfh.png", dpi=100, bbox_inches="tight")
    plt.show()
else:
    print("No individual samples available (memory optimization)")

# %% [markdown]
# ## Scaling: Posterior Width vs Number of Galaxies
#
# The shared PSD posterior should shrink as $\sim 1/\sqrt{N}$.

# %%
# --- Scaling experiment ---
N_SIZES = [3, 5, 8, 10]
sigma_widths = []
sigma_medians = []

print(f"{'N':>5s}  {'sigma_med':>9s}  {'sigma_CI':>14s}  {'tau_med':>9s}  {'tau_CI':>14s}  {'Time':>6s}")
print("-" * 70)

for n_sub in N_SIZES:
    gals_sub = galaxies_spec[:n_sub]
    hfitter_sub = HierarchicalFitter(
        model_factory, gals_sub,
        psd_sigma_prior=(0.1, 4.0),
        psd_tau_prior=(1.0, 300.0),
        data_type="spectroscopy",
    )
    key, subkey = jax.random.split(key)
    result_sub = hfitter_sub.run(
        "evi", key=subkey,
        n_iterations=50, n_samples=6,
        n_posterior_samples=300, n_seeds=5,
        kl_rtol=0.0, verbose=False,
    )

    sig_s = np.array(result_sub.shared_samples["psd_sigma"])
    tau_s = np.array(result_sub.shared_samples["psd_tau_myr"])
    sig_lo, sig_hi = np.percentile(sig_s, [16, 84])
    tau_lo, tau_hi = np.percentile(tau_s, [16, 84])
    sig_w = sig_hi - sig_lo
    sigma_widths.append(sig_w)
    sigma_medians.append(np.median(sig_s))

    print(f"  {n_sub:>3d}  {np.median(sig_s):>9.2f}  "
          f"[{sig_lo:.2f}, {sig_hi:.2f}]  "
          f"{np.median(tau_s):>9.1f}  [{tau_lo:.1f}, {tau_hi:.1f}]  "
          f"{result_sub.wall_time_s:>5.1f}s")

# %%
# --- Plot: sigma posterior width vs N ---
fig, ax = plt.subplots(figsize=(7, 5))

ns = np.array(N_SIZES, dtype=float)
sw = np.array(sigma_widths)

ax.plot(ns, sw, "C0o-", ms=10, lw=2, label=r"$\sigma$ 68% CI width")
if sw[0] > 0:
    ref = sw[0] * np.sqrt(ns[0]) / np.sqrt(ns)
    ax.plot(ns, ref, "k--", lw=1.5, alpha=0.5, label=r"$\propto 1/\sqrt{N}$")

ax.set_xlabel("Number of galaxies $N$", fontsize=12)
ax.set_ylabel(r"$\sigma_{\rm PSD}$ 68% CI width", fontsize=12)
ax.set_title("Spectroscopic hierarchical: posterior shrinkage", fontsize=13)
ax.legend(fontsize=11)
ax.set_xticks(N_SIZES)
plt.tight_layout()
plt.savefig("notebook_figures/11_hierspec_fig06_scaling.png", dpi=100, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Summary
#
# | Data type | $\sigma$ constraint | $\tau$ constraint | Wall time |
# |-----------|--------------------|--------------------|-----------|
# | Photometry (5 bands) | Wide | Very wide | Fast |
# | Spectroscopy (200 pix) | Tighter | Still challenging | Slower |
#
# **Key findings:**
# 1. Individual spectroscopic fits recover physical params well (chi2/dof ~ 1)
# 2. Spectroscopy provides more constraining power on PSD than photometry
# 3. PSD timescale tau remains the hardest parameter to constrain
# 4. Posterior width decreases with N (hierarchical pooling works)
#
# **Next:** Tutorial 12 explores nebular emission and how emission lines
# provide additional constraints on recent star formation.
