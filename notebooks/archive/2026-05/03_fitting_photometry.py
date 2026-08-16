# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Fitting Photometry: From Real Data to Posterior
#
# **What you'll learn:**
# - Real-data workflow: load → precompute filters → inference → diagnostics
# - NUTS on photometry (fast + exact via photometric precomputation)
# - Parameter recovery and posterior credible regions
# - How spectroscopy breaks age-dust-metallicity degeneracies
# - Batch fitting with vmap
#
# **Prerequisites:** [`00_quickstart.py`](00_quickstart.py) (NUTS basics).
# **Next:** [`04_fitting_spectra.py`](04_fitting_spectra.py), [`05_joint_photometry_spectroscopy.py`](05_joint_photometry_spectroscopy.py).
#
# ---
#
# You have broadband photometry from a galaxy survey (real FITS or mock), redshift, and noise.
# Infer stellar mass, SFH, dust, metallicity—and see the age-dust-metallicity degeneracy at work.
# Understand how spectroscopy breaks it and why precomputation makes photometric fits blazingly fast.

# %%
import os
import sys
import time
import warnings

try:
    _nb_dir = os.path.dirname(os.path.abspath(__file__))
    _repo_root = os.path.abspath(os.path.join(_nb_dir, ".."))
except NameError:
    _nb_dir = os.getcwd()
    _repo_root = os.path.abspath(os.path.join(_nb_dir, ".."))

_src = os.path.join(_repo_root, "src")
if os.path.isdir(os.path.join(_src, "tengri")):
    sys.path.insert(0, _src)
sys.path.insert(0, _repo_root)
sys.path.insert(0, _nb_dir)

import importlib.util

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)
warnings.filterwarnings("ignore", category=FutureWarning)

# Headless / notebook CI: limit XLA memory growth
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.45")

# Gate expensive fits (batch of 100 galaxies, filter sweep). Default off.
RUN_EXPENSIVE = False

from tengri import (
    Fitter,
    Fixed,
    SEDModel,
    Observation,
    Parameters,
    Photometry,
    Spectroscopy,
    Uniform,
    load_ssp_data,
)

# Locate ``notebooks/_plot_style.py`` and ``data/`` root (nbclient cwd is often wrong).

_repo_data_root = None
_spec_tengri = importlib.util.find_spec("tengri")
if _spec_tengri is not None and _spec_tengri.origin:
    _walk = os.path.dirname(os.path.abspath(_spec_tengri.origin))
    for _step in range(12):
        _candidate = os.path.join(_walk, "notebooks", "_plot_style.py")
        if os.path.isfile(_candidate):
            sys.path.insert(0, os.path.dirname(_candidate))
            _repo_data_root = os.path.dirname(os.path.dirname(os.path.abspath(_candidate)))
            break
        _parent_walk = os.path.dirname(_walk)
        if _parent_walk == _walk:
            break
        _walk = _parent_walk

if _repo_data_root is None:
    _np_here = os.path.abspath(os.getcwd())
    while True:
        if os.path.isfile(os.path.join(_np_here, "_plot_style.py")):
            sys.path.insert(0, _np_here)
            _repo_data_root = os.path.dirname(_np_here)
            break
        _ppt = os.path.join(_np_here, "notebooks", "_plot_style.py")
        if os.path.isfile(_ppt):
            _nbsd = os.path.dirname(_ppt)
            sys.path.insert(0, _nbsd)
            _repo_data_root = os.path.dirname(_nbsd)
            break
        _parent_here = os.path.dirname(_np_here)
        if _parent_here == _np_here:
            break
        _np_here = _parent_here

if _repo_data_root is not None and os.path.isdir(os.path.join(_repo_data_root, "data")):
    os.chdir(_repo_data_root)
elif os.path.isdir(os.path.join(_repo_root, "data")):
    os.chdir(_repo_root)
elif os.path.isdir("data"):
    pass
elif os.path.isdir(os.path.join("..", "data")):
    os.chdir("..")

FIGDIR = os.path.join("notebooks", "figures", "fitting_photometry")
os.makedirs(FIGDIR, exist_ok=True)

from _plot_style import COLORS, convergence_table, plot_sfh, safe_corner, setup_style

setup_style()

# %% [markdown]
# ## Section 1: Synthesizing and Fitting Real-like Data
#
# We simulate a "real" galaxy by generating mock photometry and spectroscopy from a known truth,
# then fit the photometry with NUTS. This workflow mirrors the steps you'd take on actual FITS data:
# load wavelengths, noise, redshift, and inversion vectors. Replace the mock generation
# with your data loader (see "Loading Your Own Data" section below).

# %%
ssp_data = load_ssp_data("data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")

# Real-like data: spectroscopy grid, photometry filter set, redshift
WAVE_OBS = jnp.linspace(3800.0, 9200.0, 200)
REDSHIFT = 0.05  # Spectroscopic redshift (known)

# Observation: photometry + spectroscopy
obs = Observation(
    photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]),
    spectroscopy=Spectroscopy(wave_obs=WAVE_OBS),
)

# %% [markdown]
# ### Calibration and Noise Model
#
# Real spectra have systematics:
# - **Inverse variance (ivar):** `noise = sqrt(1/ivar)` after masking bad pixels (ivar ≤ 0).
# - **Telluric masking:** B-band (6860–6960 Å), A-band (7580–7700 Å), water (9300–9700 Å).
# - **Calibration floor:** ~5–10% systematic uncertainty. Add in quadrature: `noise_eff = sqrt(noise² + (0.05*flux)²)`.
# - **Flux calibration:** SDSS gives ~1% calibration uncertainty (sky subtraction, standard star).
#
# For this demo, we simulate noise from SNR specifications; production code should load real ivar.

# %%
# Define model (same structure as quickstart)
spec = Parameters(
    sfh_db_log_total_mass=Uniform(8, 12),
    sfh_db_log_sfr_inst=Uniform(-2, 3),
    sfh_db_tx_frac_0=Uniform(0.05, 0.95),
    sfh_db_tx_frac_1=Uniform(0.05, 0.95),
    sfh_db_tx_frac_2=Uniform(0.05, 0.95),
    met_logzsol=Uniform(-2.0, 0.2),
    dust_tau_bc=Uniform(0.0, 2.0),
    dust_tau_diff=Uniform(0.0, 1.5),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(REDSHIFT),
    mean_sfh_type="dense_basis",
)

model = SEDModel(spec, ssp_data, observation=obs)

# Generate "real" galaxy (in practice, load from FITS)
key = jax.random.PRNGKey(42)
true_params = spec.sample(key)
# Override to a typical star-forming galaxy (still forming stars now)
true_params = {**true_params}
true_params["sfh_db_log_total_mass"] = jnp.array(10.5)
true_params["sfh_db_log_sfr_inst"] = jnp.array(0.8)
true_params["sfh_db_tx_frac_0"] = jnp.array(0.25)
true_params["sfh_db_tx_frac_1"] = jnp.array(0.35)
true_params["sfh_db_tx_frac_2"] = jnp.array(0.4)

# Mock data (SNR=30 for spectroscopy, SNR=20 for photometry)
mock_spec = model.mock_spectrum(true_params, WAVE_OBS, snr=30.0, key=key)
mock_phot = model.mock(true_params, snr=20.0, key=jax.random.fold_in(key, 1))

print(f"Mock galaxy at z={REDSHIFT}:")
print(f"  Spectrum: {len(WAVE_OBS)} pixels, SNR=30 (obs-frame 3800–9200 Å)")
print(f"  Photometry: {obs.photometry.n_filters} bands, SNR=20")

# %%
# --- FIGURE 1: Mock spectrum and photometry ---
fig, (ax_spec, ax_phot) = plt.subplots(1, 2, figsize=(14, 4))

# Left: spectrum
w = np.array(WAVE_OBS)
ax_spec.errorbar(
    w,
    np.array(mock_spec.flux_obs),
    yerr=np.array(mock_spec.noise),
    fmt=".",
    ms=2,
    color=COLORS["data"],
    alpha=0.4,
    label="Observed (SNR=30)",
)
ax_spec.plot(w, np.array(mock_spec.flux_true), color=COLORS["truth"], lw=1.2, label="Truth")
ax_spec.set_xlabel("Observed wavelength [Å]")
ax_spec.set_ylabel("Flux density")
ax_spec.set_title(f"Galaxy Spectrum at z = {REDSHIFT}")
ax_spec.legend(fontsize=10)
ax_spec.grid(True, alpha=0.3)

# Right: photometry
band_names = list(obs.photometry.names)
band_idx = np.arange(len(band_names))
ax_phot.errorbar(
    band_idx,
    np.array(mock_phot.flux_obs),
    yerr=np.array(mock_phot.noise),
    fmt="o",
    ms=7,
    color=COLORS["data"],
    alpha=0.7,
    label="Observed (SNR=20)",
)
ax_phot.plot(
    band_idx,
    np.array(mock_phot.flux_true),
    "s",
    ms=9,
    color=COLORS["truth"],
    alpha=0.8,
    label="Truth",
)
ax_phot.set_xticks(band_idx)
ax_phot.set_xticklabels(band_names, rotation=45, ha="right", fontsize=10)
ax_phot.set_ylabel(r"$f_\nu$ [erg/s/cm$^2$/Hz]")
ax_phot.set_title("Mock Photometry (5 SDSS bands)")
ax_phot.legend(fontsize=10)
ax_phot.grid(True, alpha=0.3, axis="y")

fig.suptitle("Mock Data: Spectrum + Photometry", fontsize=12)
fig.tight_layout()
# plt.savefig(os.path.join(FIGDIR, "fig01_mock_data.png", dpi=300, bbox_inches="tight"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Section 2: Photometric Fitting with NUTS
#
# We fit the photometry alone (5 SDSS bands) using NUTS, a fast exact sampler.
# Then we'll show how adding spectroscopy breaks the age-dust-metallicity degeneracy.

# %%
# Photometry-only observation
obs_phot = Observation(
    photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
)
model_phot = SEDModel(spec, ssp_data, observation=obs_phot)

# Fit with NUTS
fitter_phot = Fitter(model_phot, mock_phot.flux_obs, mock_phot.noise)

# Warm up with MAP
_ = fitter_phot.run("map", n_steps=300, verbose=False)

t0 = time.perf_counter()
result_phot = fitter_phot.run(
    "mcmc_nuts",
    n_warmup=500,
    n_samples=1000,
    verbose=False,
)
t_fit_phot = time.perf_counter() - t0

print(f"Photometry-only fit: {t_fit_phot:.1f}s (NUTS)")
_ = convergence_table({"NUTS": result_phot}, verbose=True)

# %% [markdown]
# ### Handling Photometric Redshift Uncertainty
#
# Real photometric surveys have redshift uncertainties (photo-z errors), typically σ_z ~ 0.02–0.05.
# Instead of fixing redshift as `Fixed(0.1)`, you can marginalize over a photo-z prior.
# This weakens parameter constraints but makes the fit robust to photo-z errors.
#
# The posterior then marginalizes over z, folding redshift uncertainty into stellar mass and SFH posteriors.

# %%
# Example: replace Fixed(0.1) with a Gaussian photo-z prior σ_z = 0.02
from tengri import Gaussian

# Re-define the spec with a photo-z prior (instead of Fixed)
spec_with_photoz = Parameters(
    sfh_db_log_total_mass=Uniform(8, 12),
    sfh_db_log_sfr_inst=Uniform(-2, 3),
    sfh_db_tx_frac_0=Uniform(0.05, 0.95),
    sfh_db_tx_frac_1=Uniform(0.05, 0.95),
    sfh_db_tx_frac_2=Uniform(0.05, 0.95),
    met_logzsol=Uniform(-2.0, 0.2),
    dust_tau_bc=Uniform(0.0, 2.0),
    dust_tau_diff=Uniform(0.0, 1.5),
    dust_slope=Fixed(-0.7),
    redshift=Gaussian(REDSHIFT, 0.02, lo=0.001, hi=6.0),  # Photo-z prior: μ=0.1, σ=0.02
    mean_sfh_type="dense_basis",
)

# Fit with the photo-z prior (optional; commented out to save time)
# Note: This fit will be slower (extra free parameter) and posteriors wider.
# model_photoz = SEDModel(spec_with_photoz, ssp_data, observation=obs_phot)
# fitter_photoz = Fitter(model_photoz, mock_phot.flux_obs, mock_phot.noise)
# result_photoz = fitter_photoz.run("mcmc_nuts", n_warmup=500, n_samples=1000, verbose=False)

print("Photo-z prior: redshift marginalized with σ_z=0.02")
print("Posterior would be wider than fixed-z fit, but robust to photo-z errors.")

# %%
# Figure 2: Photometric corner plot
result_phot.plot_corner(
    figsize=(12, 12),
    truths={p: float(true_params[p]) for p in spec.free_params},
)
fig = plt.gcf()
fig.suptitle("Photometry-Only Posterior (5 SDSS bands, NUTS)", fontsize=12, y=0.995)
# plt.savefig(os.path.join(FIGDIR, "fig02_corner_phot.png", dpi=300, bbox_inches="tight"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Section 3: Joint Photometry + Spectroscopy
#
# Even low-resolution spectroscopy dramatically breaks the age-dust degeneracy.
# Compare photometry-only vs spectroscopy-only vs joint posteriors.

# %%
# Joint fit
obs_joint = Observation(
    photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]),
    spectroscopy=Spectroscopy(wave_obs=WAVE_OBS, resolution=100),
)

data_joint = obs_joint.pack_data(phot=mock_phot.flux_obs, spec=mock_spec.flux_obs)
noise_joint = obs_joint.pack_data(phot=mock_phot.noise, spec=mock_spec.noise)

model_joint = SEDModel(spec, ssp_data, observation=obs_joint)
fitter_joint = Fitter(model_joint, data_joint, noise_joint)

_ = fitter_joint.run("map", n_steps=300, verbose=False)

t0 = time.perf_counter()
result_joint = fitter_joint.run(
    "mcmc_nuts",
    n_warmup=500,
    n_samples=1000,
    verbose=False,
)
t_fit_joint = time.perf_counter() - t0

print(f"Joint fit (phot+spec): {t_fit_joint:.1f}s (NUTS)")

# Spectroscopy-only fit for comparison
obs_spec = Observation(spectroscopy=Spectroscopy(wave_obs=WAVE_OBS))
model_spec = SEDModel(spec, ssp_data, observation=obs_spec)
fitter_spec = Fitter(model_spec, mock_spec.flux_obs, mock_spec.noise)
_ = fitter_spec.run("map", n_steps=300, verbose=False)

t0 = time.perf_counter()
result_spec = fitter_spec.run(
    "mcmc_nuts",
    n_warmup=500,
    n_samples=1000,
    verbose=False,
)
t_fit_spec = time.perf_counter() - t0

print(f"Spectroscopy-only fit: {t_fit_spec:.1f}s (NUTS)")

# %%
# Figure 3: Phot-only vs joint vs spec-only 1D marginals
truths_dict = {p: float(true_params[p]) for p in spec.free_params}

fig = plt.figure(figsize=(15, 15))
gs = fig.add_gridspec(len(spec.free_params), 3, hspace=0.4, wspace=0.3)

params_list = spec.free_params
for i, pname in enumerate(params_list):
    # Photometry-only
    ax_p = fig.add_subplot(gs[i, 0])
    ax_p.hist(
        np.array(result_phot.samples[pname]),
        bins=30,
        alpha=0.6,
        color=COLORS.get("data", "C0"),
        density=True,
    )
    tv = truths_dict[pname]
    ax_p.axvline(tv, color="red", lw=1.5, label="Truth")
    ax_p.set_ylabel("Density")
    ax_p.set_title(f"{pname.replace('sfh_db_', '')} (phot-only)")
    ax_p.legend(fontsize=10)

    # Spectroscopy-only
    ax_s = fig.add_subplot(gs[i, 1])
    ax_s.hist(
        np.array(result_spec.samples[pname]),
        bins=30,
        alpha=0.6,
        color=COLORS.get("mcmc_nuts", "C2"),
        density=True,
    )
    ax_s.axvline(tv, color="red", lw=1.5, label="Truth")
    ax_s.set_ylabel("Density")
    ax_s.set_title(f"{pname.replace('sfh_db_', '')} (spec-only)")
    ax_s.legend(fontsize=10)

    # Joint
    ax_j = fig.add_subplot(gs[i, 2])
    ax_j.hist(
        np.array(result_joint.samples[pname]),
        bins=30,
        alpha=0.6,
        color=COLORS.get("vi", "C1"),
        density=True,
    )
    ax_j.axvline(tv, color="red", lw=1.5, label="Truth")
    ax_j.set_ylabel("Density")
    ax_j.set_title(f"{pname.replace('sfh_db_', '')} (joint)")
    ax_j.legend(fontsize=10)

fig.suptitle("Degeneracies: Photometry-Only vs Spectroscopy vs Joint (NUTS)", fontsize=14)
# plt.savefig(os.path.join(FIGDIR, "fig03_posteriors_comparison.png", dpi=300, bbox_inches="tight"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Section 4: Batch Fitting
#
# tengri's vmap architecture scales to large catalogs. Fit 24 galaxies in parallel with NUTS.
# **Gated** behind ``RUN_EXPENSIVE`` — default off; set to True for full run.

# %%
if RUN_EXPENSIVE:
    N_GAL = 24
    keys = jax.random.split(jax.random.PRNGKey(0), N_GAL)
    true_params_batch = jax.vmap(spec.sample)(keys)

    mocks_batch = []
    for i in range(N_GAL):
        p_i = {k: v[i] for k, v in true_params_batch.items()}
        m = model_phot.mock(p_i, snr=15.0, key=jax.random.fold_in(jax.random.PRNGKey(0), i))
        mocks_batch.append(m)

    print(f"Generated {N_GAL} mocks with SNR=15")

    fitter_batch = Fitter(model_phot, mocks_batch[0].flux_obs, mocks_batch[0].noise)
    galaxy_list = [{"flux_obs": m.flux_obs, "noise": m.noise} for m in mocks_batch]

    t0 = time.perf_counter()
    results_batch = fitter_batch.fit_batch(
        galaxy_list,
        method="mcmc_nuts",
        n_warmup=300,
        n_samples=600,
        verbose=False,
    )
    t_batch = time.perf_counter() - t0
    print(f"Batch fit {N_GAL} galaxies in {t_batch:.1f}s ({t_batch / N_GAL:.2f}s per galaxy)")
else:
    print(
        "Batch fit skipped (RUN_EXPENSIVE=False). "
        "Set RUN_EXPENSIVE=True to fit 24 galaxies (~60–90 s on CPU)."
    )
    N_GAL = 0
    true_params_batch = None
    mocks_batch = None
    results_batch = None

# %%
# Figure 4: Stellar mass recovery (gated by RUN_EXPENSIVE)
if not RUN_EXPENSIVE or results_batch is None:
    print("Figure 4 skipped (batch fit not run).")
else:
    fig4, ax = plt.subplots(figsize=(8, 8))

    true_mstar = []
    recovered_mstar_med = []
    recovered_mstar_lo = []
    recovered_mstar_hi = []
    dust_tau_bc_vals = []

    for i, res in enumerate(results_batch):
        p_i = {k: v[i] for k, v in true_params_batch.items()}
        true_mstar.append(float(p_i["sfh_db_log_sfr_inst"]))
        recovered_mstar_med.append(float(np.median(res.samples["sfh_db_log_sfr_inst"])))
        recovered_mstar_lo.append(float(np.percentile(res.samples["sfh_db_log_sfr_inst"], 16)))
        recovered_mstar_hi.append(float(np.percentile(res.samples["sfh_db_log_sfr_inst"], 84)))
        dust_tau_bc_vals.append(float(np.median(res.samples["dust_tau_bc"])))

    true_mstar = np.array(true_mstar)
    recovered_mstar_med = np.array(recovered_mstar_med)
    recovered_mstar_lo = np.array(recovered_mstar_lo)
    recovered_mstar_hi = np.array(recovered_mstar_hi)
    dust_tau_bc_vals = np.array(dust_tau_bc_vals)

    sc = ax.scatter(
        true_mstar,
        recovered_mstar_med,
        c=dust_tau_bc_vals,
        s=40,
        alpha=0.7,
        cmap="viridis",
        edgecolors="k",
        linewidth=0.3,
    )
    ax.errorbar(
        true_mstar,
        recovered_mstar_med,
        yerr=[
            recovered_mstar_med - recovered_mstar_lo,
            recovered_mstar_hi - recovered_mstar_med,
        ],
        fmt="none",
        ecolor="grey",
        alpha=0.3,
        elinewidth=0.5,
    )

    lim = [
        min(true_mstar.min(), recovered_mstar_lo.min()),
        max(true_mstar.max(), recovered_mstar_hi.max()),
    ]
    ax.plot(lim, lim, "k--", lw=1, label="1:1")
    ax.set_xlabel("True log SFR_inst")
    ax.set_ylabel("Recovered log SFR_inst (median ± 68%)")
    ax.set_title(f"Batch Fitting: {N_GAL} Galaxies (NUTS)")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(r"$\tau_{BC}$")
    ax.legend()
    fig4.tight_layout()
    plt.show()

# %% [markdown]
# ## Section 5: Filter Sensitivity
#
# Posterior width depends strongly on filter coverage.
# Compare SDSS-only, SDSS+2MASS, and SDSS+2MASS+WISE on the same mock galaxy.

# %%
# Three filter configurations
configs = {
    "SDSS (5 bands)": ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"],
    "SDSS + 2MASS (8 bands)": [
        "sdss_u",
        "sdss_g",
        "sdss_r",
        "sdss_i",
        "sdss_z",
        "twomass_j",
        "twomass_h",
        "twomass_ks",
    ],
    "SDSS + 2MASS + WISE (12 bands)": [
        "sdss_u",
        "sdss_g",
        "sdss_r",
        "sdss_i",
        "sdss_z",
        "twomass_j",
        "twomass_h",
        "twomass_ks",
        "wise_w1",
        "wise_w2",
        "wise_w3",
        "wise_w4",
    ],
}

posteriors_by_config = {}

if RUN_EXPENSIVE:
    for config_name, filter_names in configs.items():
        obs_cfg = Observation(photometry=Photometry.from_names(filter_names))
        model_cfg = SEDModel(spec, ssp_data, observation=obs_cfg)

        key_cfg = jax.random.PRNGKey(99)
        true_params_cfg = spec.sample(key_cfg)
        mock_cfg = model_cfg.mock(true_params_cfg, snr=15.0, key=key_cfg)

        fitter_cfg = Fitter(model_cfg, mock_cfg.flux_obs, mock_cfg.noise)
        _ = fitter_cfg.run("map", n_steps=200, verbose=False)
        result_cfg = fitter_cfg.run(
            "mcmc_nuts",
            n_warmup=300,
            n_samples=600,
            verbose=False,
        )
        posteriors_by_config[config_name] = result_cfg
else:
    print(
        "Filter sweep skipped (RUN_EXPENSIVE=False). "
        "Set RUN_EXPENSIVE=True to fit three filter configurations (~45 s)."
    )

# %%
# Figure 5: Posterior width by filter config (gated by RUN_EXPENSIVE)
if not RUN_EXPENSIVE or not posteriors_by_config:
    print("Figure 5 skipped (filter sweep not run).")
else:
    fig5, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, (config_name, result) in zip(axes, posteriors_by_config.items()):
        params_to_plot = [
            "sfh_db_log_sfr_inst",
            "dust_tau_bc",
            "met_logzsol",
        ]
        labels_plot = ["log SFR_inst", r"$\tau_{BC}$", "log Z/Z☉"]

        widths = []
        for pname in params_to_plot:
            lo = float(np.percentile(result.samples[pname], 16))
            hi = float(np.percentile(result.samples[pname], 84))
            widths.append(hi - lo)

        ax.bar(labels_plot, widths, color=COLORS.get("mcmc_nuts", "C2"), alpha=0.7, edgecolor="k")
        ax.set_ylabel("68% CI Width")
        ax.set_title(config_name)
        ax.grid(axis="y", alpha=0.3)

    fig5.suptitle("Posterior Width vs Filter Configuration", fontsize=13)
    fig5.tight_layout()
    plt.show()

# %% [markdown]
# ## Loading Your Own Data
#
# Replace the mock generation with your FITS loader:
#
# ```python
# from astropy.io import fits
#
# hdu = fits.open("my_galaxy.fits")
# flux_obs = hdu[1].data["flux"]
# ivar = hdu[1].data["ivar"]
# wave_obs = hdu[1].data["wavelength"]  # observed-frame Angstrom
#
# # Mask bad pixels, apply inverse-variance
# good = (ivar > 0) & np.isfinite(ivar) & np.isfinite(flux_obs)
# noise = np.sqrt(1.0 / ivar[good])
# flux_obs, wave_obs = flux_obs[good], wave_obs[good]
#
# # Calibration floor (5% systematic)
# noise_effective = np.sqrt(noise**2 + (0.05 * flux_obs)**2)
#
# # Mask tellurics: B-band (6860–6960 Å), A-band (7580–7700 Å), water (9300–9700 Å)
# mask_tellur = (
#     ((wave_obs > 6860) & (wave_obs < 6960)) |
#     ((wave_obs > 7580) & (wave_obs < 7700)) |
#     ((wave_obs > 9300) & (wave_obs < 9700))
# )
# good_wave = ~mask_tellur
# flux_obs, wave_obs, noise_effective = flux_obs[good_wave], wave_obs[good_wave], noise_effective[good_wave]
#
# # Create photometry + spectroscopy observation with known redshift
# redshift_spec = hdu[0].header["Z"]
# obs = Observation(
#     photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]),
#     spectroscopy=Spectroscopy(wave_obs=wave_obs, resolution=100),
# )
#
# # Fit with tengri
# model = SEDModel(spec, ssp_data, observation=obs)
# fitter = Fitter(model, flux_obs, noise_effective, data_type="mixed")
# result = fitter.run("mcmc_nuts", n_warmup=500, n_samples=1000)
# ```

# %% [markdown]
# ## Summary & Next Steps
#
# **What we've shown:**
# - Real-data workflow: load mock spectrum+photometry, fit with NUTS, recover posterior.
# - Age-dust-metallicity degeneracy in photometry; spectroscopy breaks it.
# - Batch fitting of 24 galaxies in parallel (NUTS).
# - Filter sensitivity: more bands (especially NIR/FIR) → narrower posteriors.
#
# **Key takeaways:**
# - Always fit with MAP initialization; check `convergence_table()` diagnostics.
# - Joint phot+spec is always worth it: degeneracy nearly vanishes.
# - Batch fitting scales to your catalog size with vmap.
# - Real data needs calibration floor + telluric masking; edge cases fail silently.
#
# **Next:** [`04_fitting_spectra.py`](04_fitting_spectra.py) for spectroscopy-only workflows.
# [`05_joint_photometry_spectroscopy.py`](05_joint_photometry_spectroscopy.py) for
# detailed joint analysis. [`11_population.py`](11_population.py) for hierarchical inference.
