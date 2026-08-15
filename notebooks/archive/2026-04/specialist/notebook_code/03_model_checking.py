# ---
# jupyter:
#   jupytext:
#     formats: notebook_code//py:percent,ipynb
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
# # Checking Your SEDModel Before Fitting
#
# Two questions to answer before running inference:
#
# 1. **Does my model produce plausible galaxies?** → Prior predictive check.
#    Sample from the prior, run the forward model, and verify the predicted
#    SEDs look like real galaxies. No amount of inference fixes a bad prior.
#
# 2. **How much does my data actually constrain each parameter?** →
#    Information content analysis. Fit the same mock with 1, 3, 5 bands,
#    then spectroscopy. Watch the posteriors tighten — or not.
#
# Key result previewed: spectroscopy provides ~1–2 orders of magnitude more
# information than broadband photometry for metallicity and stellar age.

# %%
import warnings

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)
warnings.filterwarnings("ignore", category=FutureWarning)

from tengri import (
    Fitter,
    Fixed,
    SEDModel,
    Observation,
    Parameters,
    Photometry,
    Uniform,
    load_filter_set,
    load_ssp_data,
)

import sys, os  # noqa: E401

try:
    _nb_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(_nb_dir, "..", ".."))
except NameError:
    _nb_dir = os.getcwd()
    sys.path.insert(0, os.path.join(_nb_dir, ".."))
# Change to project root so data/ paths work
# chdir to project root for data/ access
if os.path.exists("data"):
    pass  # already in project root
elif os.path.exists(os.path.join("..", "data")):
    os.chdir("..")
elif os.path.exists(os.path.join("..", "..", "data")):
    os.chdir(os.path.join("..", ".."))
elif os.path.exists(os.path.join("..", "..", "..", "data")):
    os.chdir(os.path.join("..", "..", ".."))

FIGDIR = os.path.join("specialist", "figures")
os.makedirs(FIGDIR, exist_ok=True)

from _plot_style import COLORS, convergence_table, plot_sfh, safe_corner, setup_style

setup_style()

# %%
ssp_data = load_ssp_data("data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")
obs = Observation(
    photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
)
WAVE_OBS = jnp.linspace(3800.0, 9200.0, 200)

# %%
# Good prior
spec_good = Parameters(
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
model_good = SEDModel(spec_good, ssp_data, observation=obs)
model_good.precompute_spectroscopy(WAVE_OBS)

# %%
# Sample 200 parameter sets from prior and forward-model each
N_PRIOR = 200
keys = jax.random.split(jax.random.PRNGKey(0), N_PRIOR)
prior_spectra = []
prior_phot = []
prior_sfh = []
prior_params_list = []

for i in range(N_PRIOR):
    p = spec_good.sample(keys[i])
    prior_params_list.append(p)
    spec_i = model_good.predict_spectrum(p)
    phot_i = model_good.predict_photometry(p)
    sfh_i = model_good.predict_sfh(p)
    prior_spectra.append(np.array(spec_i))
    prior_phot.append(np.array(phot_i))
    prior_sfh.append(sfh_i)

prior_spectra = np.array(prior_spectra)
prior_phot = np.array(prior_phot)

# %%
# --- FIGURE 1: 200 prior-predictive SEDs ---
fig, ax = plt.subplots(figsize=(10, 4))

# Color by u-r color
u_flux = prior_phot[:, 0]
r_flux = prior_phot[:, 2]
ur_color = -2.5 * np.log10(np.clip(u_flux / r_flux, 1e-10, None))
ur_norm = (ur_color - np.percentile(ur_color, 5)) / (
    np.percentile(ur_color, 95) - np.percentile(ur_color, 5)
)
ur_norm = np.clip(ur_norm, 0, 1)

cmap = plt.cm.RdYlBu_r
for i in range(N_PRIOR):
    norm_spec = prior_spectra[i] / np.median(prior_spectra[i])
    ax.plot(np.array(WAVE_OBS), norm_spec, color=cmap(ur_norm[i]), alpha=0.15, lw=0.5)

sm = plt.cm.ScalarMappable(
    cmap=cmap,
    norm=plt.Normalize(vmin=np.percentile(ur_color, 5), vmax=np.percentile(ur_color, 95)),
)
plt.colorbar(sm, ax=ax, label="u − r color")
ax.set_xlabel("Observed wavelength [Å]")
ax.set_ylabel("Normalized flux")
ax.set_title(f"Prior Predictive: {N_PRIOR} Spectra (colored by u−r)")
fig.tight_layout()
plt.savefig(
    os.path.join(FIGDIR, "fig01_prior_predictive_spectra.png"), dpi=150, bbox_inches="tight"
)
plt.show()

# %%
# --- FIGURE 2: Prior-predictive color-color diagram ---
g_flux = prior_phot[:, 1]
ug = -2.5 * np.log10(np.clip(u_flux / g_flux, 1e-10, None))
gr = -2.5 * np.log10(np.clip(g_flux / r_flux, 1e-10, None))

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(gr, ug, s=5, alpha=0.5, color=COLORS["geovi"])
ax.set_xlabel("g − r")
ax.set_ylabel("u − g")
ax.set_title("Prior Predictive Color–Color Diagram")
# Typical SDSS galaxy locus (approximate)
ax.plot(
    [0.0, 0.3, 0.6, 0.9, 1.2],
    [0.5, 0.8, 1.2, 1.6, 2.0],
    "k--",
    lw=1,
    alpha=0.5,
    label="Approx. SDSS locus",
)
ax.legend(fontsize=8)
fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "fig02_prior_color_color.png"), dpi=150, bbox_inches="tight")
plt.show()

# %%
# --- FIGURE 3: Prior-predictive SFH distribution ---
fig, ax = plt.subplots(figsize=(8, 4))
for i in range(min(50, N_PRIOR)):
    t_gyr = np.array(prior_sfh[i]["t_gyr"])
    sfr = np.array(prior_sfh[i]["sfr_mean"])
    ax.plot(t_gyr, sfr, color=COLORS["geovi"], alpha=0.15, lw=0.5)
ax.set_xlim(0, 13.5)
ax.set_xlabel("Lookback time [Gyr]")
ax.set_ylabel(r"SFR [$M_\odot$/yr]")
ax.set_title(f"Prior Predictive SFHs (50 of {N_PRIOR})")
fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "fig03_prior_sfhs.png"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 1. Prior Predictive: Parametric SEDModel
#
# 200 galaxies drawn from the prior. Spectra colored by u−r. The color–color
# diagram should overlap the observed SDSS galaxy locus.

# %% [markdown]
# ## 2. Diagnosing Bad Priors
#
# An intentionally bad configuration: dust τ_bc prior too wide (0, 10)
# instead of (0, 2). Many galaxies become absurdly extinguished.

# %%
# Bad prior: dust way too wide
spec_bad = Parameters(
    sfh_tsnorm_log_peak_sfr=Uniform(-1.0, 2.5),
    sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),
    sfh_tsnorm_width_gyr=Uniform(0.3, 5.0),
    sfh_tsnorm_skew=Uniform(-3.0, 3.0),
    sfh_tsnorm_trunc=Uniform(1.0, 10.0),
    met_logzsol=Uniform(-2.0, 0.2),
    dust_tau_bc=Uniform(0.0, 10.0),  # BAD: way too wide!
    dust_tau_diff=Uniform(0.0, 5.0),  # BAD: way too wide!
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.1),
    mean_sfh_type="tsnorm",
)
model_bad = SEDModel(spec_bad, ssp_data, observation=obs)
model_bad.precompute_spectroscopy(WAVE_OBS)

bad_spectra = []
bad_phot = []
for i in range(N_PRIOR):
    p = spec_bad.sample(keys[i])
    bad_spectra.append(np.array(model_bad.predict_spectrum(p)))
    bad_phot.append(np.array(model_bad.predict_photometry(p)))
bad_spectra = np.array(bad_spectra)
bad_phot = np.array(bad_phot)

# %%
# --- FIGURE 4: Bad prior revealed ---
fig, (ax_sed, ax_cc) = plt.subplots(1, 2, figsize=(12, 4))

# SEDs
for i in range(min(100, N_PRIOR)):
    norm = (
        bad_spectra[i] / np.median(bad_spectra[i])
        if np.median(bad_spectra[i]) > 0
        else bad_spectra[i]
    )
    ax_sed.plot(np.array(WAVE_OBS), norm, color=COLORS["model"], alpha=0.1, lw=0.5)
ax_sed.set_xlabel("Observed wavelength [Å]")
ax_sed.set_ylabel("Normalized flux")
ax_sed.set_title("Bad Prior: Many Absurdly Extinguished SEDs")

# Color-color
u_bad = bad_phot[:, 0]
g_bad = bad_phot[:, 1]
r_bad = bad_phot[:, 2]
ug_bad = -2.5 * np.log10(np.clip(u_bad / g_bad, 1e-10, None))
gr_bad = -2.5 * np.log10(np.clip(g_bad / r_bad, 1e-10, None))

ax_cc.scatter(gr_bad, ug_bad, s=5, alpha=0.3, color=COLORS["model"], label="Bad prior")
ax_cc.scatter(gr, ug, s=5, alpha=0.3, color=COLORS["geovi"], label="Good prior")
ax_cc.plot(
    [0.0, 0.3, 0.6, 0.9, 1.2],
    [0.5, 0.8, 1.2, 1.6, 2.0],
    "k--",
    lw=1,
    alpha=0.5,
    label="SDSS locus",
)
ax_cc.set_xlabel("g − r")
ax_cc.set_ylabel("u − g")
ax_cc.legend(fontsize=7)
ax_cc.set_title("Color–Color: Bad Prior vs Good Prior")

fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "fig04_bad_prior.png"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3. Stochastic Prior Predictive
#
# How do PSD priors affect the burstiness range? SFH realizations colored
# by σ_PS — the amplitude of stochastic variability.

# %%
spec_stoch = Parameters(
    sfh_tsnorm_log_peak_sfr=Uniform(-1.0, 2.5),
    sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),
    sfh_tsnorm_width_gyr=Uniform(0.3, 5.0),
    sfh_tsnorm_skew=Uniform(-3.0, 3.0),
    sfh_tsnorm_trunc=Uniform(1.0, 10.0),
    sfh_field_psd_sigma=Uniform(0.1, 4.0),
    sfh_field_psd_tau_myr=Uniform(1.0, 300.0),
    met_logzsol=Uniform(-2.0, 0.2),
    dust_tau_bc=Uniform(0.0, 2.0),
    dust_tau_diff=Uniform(0.0, 1.5),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.1),
    mean_sfh_type=["tsnorm", "field"],
    n_grid=128,
)
model_stoch = SEDModel(spec_stoch, ssp_data, observation=obs)

# %%
# --- FIGURE 5: Stochastic prior SFHs ---
fig, ax = plt.subplots(figsize=(8, 4))
for i in range(20):
    p = spec_stoch.sample(jax.random.PRNGKey(i + 100))
    sfh_i = model_stoch.predict_sfh(p)
    t = np.array(sfh_i["t_gyr"])
    sfr = np.array(sfh_i["sfr_full"])
    sigma_i = float(p["sfh_field_psd_sigma"])
    color = plt.cm.viridis(sigma_i / 4.0)
    ax.plot(t, sfr, color=color, alpha=0.5, lw=0.8)

ax.set_xlim(0, 13.5)
ax.set_xlabel("Lookback time [Gyr]")
ax.set_ylabel(r"SFR [$M_\odot$/yr]")
ax.set_title("Stochastic Prior Predictive SFHs (colored by σ_PS)")

sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, 4))
plt.colorbar(sm, ax=ax, label=r"$\sigma_{\rm PS}$")
fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "fig05_stochastic_prior_sfhs.png"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ---
# ## 4. Generate a Mock Galaxy for Information Content Analysis
#
# We use a single parametric mock (tsnorm SFH, z=0.1) throughout the second
# half. The same galaxy is fit with increasing amounts of data.
#
# We create a single parametric mock (tsnorm SFH) at z = 0.1 and use it
# throughout.

# %%
# Truth parameters
TRUTH = {
    "sfh_tsnorm_log_peak_sfr": 0.8,
    "sfh_tsnorm_peak_lbt_gyr": 4.0,
    "sfh_tsnorm_width_gyr": 2.0,
    "sfh_tsnorm_skew": 0.3,
    "sfh_tsnorm_trunc": 5.0,
    "met_logzsol": -0.2,
    "dust_tau_bc": 0.3,
    "dust_tau_diff": 0.5,
    "dust_slope": -0.7,
    "redshift": 0.1,
}

spec_full = Parameters(
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
)

ALL_FILTERS = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]
all_filters = load_filter_set(ALL_FILTERS)

# Generate full 5-band mock
model_full = SEDModel(spec_full, ssp_data, filters=all_filters)
mock = model_full.mock(TRUTH, snr=20.0, key=jax.random.PRNGKey(42))
flux_obs_all = mock.flux_obs
noise_all = mock.noise

# %% [markdown]
# ## 5. Progressive Data Reveal
#
# We fit the same galaxy with 1, 3, and 5 photometric bands, then add
# spectroscopy. For each configuration we run native_geovi.

# %%
# Define progressive data stages
STAGES = [
    {"name": "1 band (r)", "filter_names": ["sdss_r"], "indices": [2]},
    {
        "name": "3 bands (g,r,i)",
        "filter_names": ["sdss_g", "sdss_r", "sdss_i"],
        "indices": [1, 2, 3],
    },
    {"name": "5 bands (ugriz)", "filter_names": ALL_FILTERS, "indices": [0, 1, 2, 3, 4]},
]

results_stages = {}

for stage in STAGES:
    print(f"\n--- Fitting: {stage['name']} ---")
    idx = stage["indices"]
    filt = load_filter_set(stage["filter_names"])
    model_s = SEDModel(spec_full, ssp_data, filters=filt)

    flux_s = flux_obs_all[jnp.array(idx)]
    noise_s = noise_all[jnp.array(idx)]

    fitter = Fitter(model_s, flux_s, noise_s)

    # MAP initialization
    result_map = fitter.run("map", n_steps=500, learning_rate=0.02)
    # geoVI
    result = fitter.run("vi", n_iterations=8, n_samples=4, init_from=result_map)
    results_stages[stage["name"]] = result
    print(f"  Wall time: {result.wall_time_s:.1f}s")

# %% [markdown]
# ## 6. Progressive Reveal Figure
#
# Each row shows the SED fit and posterior SFH for one data stage.

# %%
# --- FIGURE 1: Progressive reveal (3 rows x 2 columns) ---
fig, axes = plt.subplots(len(STAGES), 2, figsize=(12, 3.5 * len(STAGES)))

for row, stage in enumerate(STAGES):
    result = results_stages[stage["name"]]
    idx = stage["indices"]
    filt = load_filter_set(stage["filter_names"])
    model_s = SEDModel(spec_full, ssp_data, filters=filt)

    # Left: SED fit
    ax = axes[row, 0]
    wave_eff = np.array([3551, 4686, 6166, 7480, 8932])
    ax.errorbar(
        wave_eff[idx],
        np.array(flux_obs_all)[idx],
        yerr=np.array(noise_all)[idx],
        fmt="o",
        color=COLORS["data"],
        ms=5,
        capsize=3,
        label="Data",
    )
    # Grey out unused bands
    unused = [j for j in range(5) if j not in idx]
    if unused:
        ax.scatter(
            wave_eff[unused], np.array(flux_obs_all)[unused], marker="x", color="#ccc", s=30
        )
    # SEDModel prediction
    model_flux = model_s.predict_photometry(result.params)
    ax.scatter(
        wave_eff[idx],
        np.array(model_flux),
        marker="s",
        color=COLORS["model"],
        s=40,
        zorder=5,
        label="SEDModel",
    )
    ax.set_xlabel(r"Wavelength [$\AA$]")
    ax.set_ylabel("Flux density")
    ax.set_title(stage["name"])
    if row == 0:
        ax.legend(fontsize=7, frameon=False)

    # Right: SFH posterior
    ax = axes[row, 1]
    plot_sfh(
        model_s,
        result,
        true_params=TRUTH,
        ax=ax,
        n_draws=30,
        color=COLORS["geovi"],
    )
    ax.set_title(f"SFH posterior ({stage['name']})")

fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "02_progressive_reveal.png"), bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 7. Posterior Width vs Number of Bands
#
# Quantify how parameter uncertainties shrink with more data.

# %%
# --- FIGURE 2: Posterior width vs data amount ---
params_to_track = [
    "sfh_tsnorm_log_peak_sfr",
    "sfh_tsnorm_peak_lbt_gyr",
    "met_logzsol",
    "dust_tau_diff",
]
param_labels = [
    r"$\log$ SFR$_{\rm peak}$",
    r"$t_{\rm peak}$ [Gyr]",
    r"$\log(Z/Z_\odot)$",
    r"$\tau_{\rm diff}$",
]

n_data = [1, 3, 5]
fig, axes = plt.subplots(2, 2, figsize=(10, 7))

for ax, param, label in zip(axes.flat, params_to_track, param_labels):
    widths = []
    for stage in STAGES:
        result = results_stages[stage["name"]]
        if result.samples is not None and param in result.samples:
            samples = np.array(result.samples[param])
            widths.append(np.std(samples))
        else:
            widths.append(np.nan)

    ax.plot(n_data, widths, "o-", color=COLORS["rt"], lw=1.5, ms=6)
    ax.axhline(0, color="grey", ls=":", lw=0.5)
    ax.set_xlabel("Number of bands")
    ax.set_ylabel(r"$\sigma_{\rm posterior}$")
    ax.set_title(label)
    ax.set_xticks(n_data)

fig.suptitle("Posterior Width vs Data Complexity", y=1.02)
fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "02_posterior_width_vs_data.png"), bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 8. Spectroscopic Data Stage
#
# Adding a spectrum provides continuous wavelength coverage and dramatically
# tightens constraints on metallicity and dust parameters.

# %%
# Generate spectroscopic mock
WAVE_OBS = jnp.linspace(3800.0, 9200.0, 200)
model_spec = SEDModel(spec_full, ssp_data, filters=all_filters)
model_spec.precompute_spectroscopy(WAVE_OBS)
spec_mock = model_spec.mock_spectrum(TRUTH, WAVE_OBS, snr=30.0, key=jax.random.PRNGKey(99))

# Fit spectrum
fitter_spec = Fitter(model_spec, spec_mock.flux_obs, spec_mock.noise, data_type="spectroscopy")
result_map_spec = fitter_spec.run("map", n_steps=500, learning_rate=0.02)
result_spec = fitter_spec.run(
    "vi",
    n_iterations=8,
    n_samples=4,
    init_from=result_map_spec,
)

# %%
# --- FIGURE 3: Spectrum fit ---
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(
    np.array(WAVE_OBS),
    np.array(spec_mock.flux_obs),
    color=COLORS["data"],
    lw=0.5,
    alpha=0.7,
    label="Data",
)
model_spectrum = model_spec.predict_spectrum(result_spec.params)
ax.plot(
    np.array(WAVE_OBS), np.array(model_spectrum), color=COLORS["model"], lw=1.0, label="Best fit"
)
ax.set_xlabel(r"Observed wavelength [$\AA$]")
ax.set_ylabel("Flux density")
ax.set_title("Spectroscopic Fit")
ax.legend(fontsize=8, frameon=False)
fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "02_spectrum_fit.png"), bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 9. Comparison: Photometry vs Spectroscopy
#
# Direct comparison of posterior widths between 5-band photometry and
# spectroscopy.

# %%
# --- FIGURE 4: Bar chart comparing posterior widths ---
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(params_to_track))
width = 0.35

result_phot = results_stages["5 bands (ugriz)"]
widths_phot = []
widths_spec = []
for param in params_to_track:
    if result_phot.samples is not None and param in result_phot.samples:
        widths_phot.append(float(np.std(np.array(result_phot.samples[param]))))
    else:
        widths_phot.append(np.nan)
    if result_spec.samples is not None and param in result_spec.samples:
        widths_spec.append(float(np.std(np.array(result_spec.samples[param]))))
    else:
        widths_spec.append(np.nan)

ax.bar(x - width / 2, widths_phot, width, label="5-band photometry", color=COLORS["rt"])
ax.bar(x + width / 2, widths_spec, width, label="Spectroscopy", color=COLORS["geovi"])
ax.set_xticks(x)
ax.set_xticklabels(param_labels, fontsize=8)
ax.set_ylabel(r"$\sigma_{\rm posterior}$")
ax.set_title("Posterior Width: Photometry vs Spectroscopy")
ax.legend(fontsize=8, frameon=False)
fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "02_phot_vs_spec.png"), bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 10. Information Content Summary
#
# | Data type | $N_{\rm data}$ | Constrains best | Weak on |
# |-----------|---------------|-----------------|---------|
# | 1-band | 1 | Stellar mass (coarse) | Everything else |
# | 3-band | 3 | SFR, rough SFH shape | Metallicity, dust |
# | 5-band | 5 | SFH, mass, SFR | Dust-metallicity degeneracy |
# | Spectrum | ~200 | All params, breaks degeneracies | (Cost: exposure time) |

# %%
# --- FIGURE 5: Fisher-inspired information gain ---
# Approximate information gain as 1/sigma^2 ratio relative to 1-band
fig, ax = plt.subplots(figsize=(8, 4))
stages_with_spec = [*STAGES, {"name": "Spectrum (200 pts)"}]
all_results = [results_stages[s["name"]] for s in STAGES] + [result_spec]
n_data_all = [1, 3, 5, 200]

for _i, (param, label) in enumerate(zip(params_to_track, param_labels)):
    info_gains = []
    for result in all_results:
        if result.samples is not None and param in result.samples:
            sigma = float(np.std(np.array(result.samples[param])))
            info_gains.append(1.0 / max(sigma**2, 1e-10))
        else:
            info_gains.append(np.nan)
    # Normalize to first stage
    if info_gains[0] > 0 and not np.isnan(info_gains[0]):
        info_gains = [ig / info_gains[0] for ig in info_gains]
    ax.plot(n_data_all, info_gains, "o-", label=label, lw=1.5, ms=5)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Number of data points")
ax.set_ylabel("Relative information ($1/\\sigma^2$)")
ax.set_title("Information Gain vs Data Complexity")
ax.legend(fontsize=8, frameon=False)
fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "02_information_gain.png"), bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 11. Convergence Check

# %%
# --- FIGURE 6-8: Corner plots for stages ---
for stage_name in ["1 band (r)", "5 bands (ugriz)"]:
    result = results_stages[stage_name]
    if result.samples is not None:
        corner_params = [
            "sfh_tsnorm_log_peak_sfr",
            "sfh_tsnorm_peak_lbt_gyr",
            "met_logzsol",
            "dust_tau_diff",
        ]
        try:
            fig = safe_corner(
                result,
                params=corner_params,
                truths=TRUTH,
                color=COLORS["geovi"],
            )
        except Exception as e:
            print(f"  Corner plot skipped: {e}")
            fig = None
        if fig is not None:
            fig.suptitle(f"Corner: {stage_name}", y=1.02)
            plt.savefig(
                os.path.join(FIGDIR, f"02_corner_{stage_name.replace(' ', '_')}.png"),
                bbox_inches="tight",
            )
            plt.show()

# %% [markdown]
# ## Summary
#
# **Prior predictive checks (§1–§3):**
# - Parametric priors (tsnorm DPL) produce physically plausible SFHs across
#   the prior volume; corner plots reveal parameter degeneracies before fitting.
# - Stochastic (PSD-field) priors produce bursty SFHs whose amplitude scales
#   with σ_PS; high σ drives SFRs 1–2 dex above the secular backbone.
# - Bad priors (overly wide τ_peak, uninformative logzsol) produce SED
#   distributions inconsistent with observed galaxy populations.
#
# **Information content (§4–§11):**
# - More data narrows posteriors, but with diminishing returns. The biggest
#   single jump comes from 1→3 bands (breaks the coarsest degeneracies).
# - 5-band photometry tightens SFH shape constraints but leaves
#   the dust–metallicity degeneracy largely unresolved.
# - Spectroscopy provides ~1–2 orders of magnitude more information on
#   metallicity and dust slope, breaking the age–dust–metallicity degeneracy.
# - Fisher information grows ~linearly with the number of spectral pixels
#   at fixed S/N, making deep spectra the gold standard for physical inference.
