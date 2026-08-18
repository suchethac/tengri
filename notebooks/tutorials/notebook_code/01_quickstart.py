# ---
# jupyter:
#   jupytext:
#     formats: notebook_code//py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Differentiable SED Fitting in 10 Seconds
#
# tengri combines Information Field Theory correlated-field priors with
# differentiable stellar population synthesis and JIT-compiled variational
# inference. This notebook fits a galaxy spectrum two ways: first a smooth
# parametric SFH (7 free parameters), then a bursty stochastic SFH
# (137 free parameters). Both finish in seconds on a laptop CPU. No GPU required.

# %%
import time
import warnings

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)
warnings.filterwarnings("ignore", category=FutureWarning)

from tengri import (
    Fixed,
    ForwardModel,
    SEDModel,
    Observation,
    Parameters,
    Photometry,
    Uniform,
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

FIGDIR = os.path.join("tutorials", "figures")
os.makedirs(FIGDIR, exist_ok=True)

from _plot_style import (
    COLORS,
    SPECTRAL_FEATURES,
    convergence_table,
    plot_corner_comparison,
    plot_sfh,
    safe_corner,
    setup_style,
)

setup_style()

# %%
# Load SSP templates and SDSS filters
ssp_data = load_ssp_data(
    "data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
)
obs = Observation(photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]))
print(
    f"SSP templates: {ssp_data.ssp_flux.shape[0]} metallicities × {ssp_data.ssp_flux.shape[1]} ages "
    f"× {ssp_data.ssp_flux.shape[-1]} wavelengths"
)

# %% [markdown]
# ## Part A: A Smooth Galaxy Spectrum
#
# We start with the simplest useful model: a truncated skew-normal SFH
# (Bellstedt+2020) with 7 free parameters. This is comparable to what
# Prospector or BAGPIPES would fit — but fully differentiable and much faster.

# %%
# Define the parameter specification
spec_param = Parameters(
    sfh_tsnorm_log_total_mass=Uniform(7.0, 12.5),
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
print(f"Free parameters ({spec_param.n_free}):")
for name in spec_param.free_params:
    print(f"  {name}")

# %%
# Create the model with spectroscopic precomputation
WAVE_OBS = jnp.linspace(3800.0, 9200.0, 200)  # SDSS-like, 200 pixels
model_param = SEDModel(spec_param, ssp_data, observation=obs)
model_param.precompute_spectroscopy(WAVE_OBS)
print(f"SEDModel created: {spec_param.n_free} free parameters, {len(WAVE_OBS)} spectral pixels")

# %%
# The forward model is fast
params_test = spec_param.sample(jax.random.PRNGKey(99))

# Raw (first call, includes tracing)
t0 = time.perf_counter()
_ = model_param.predict_spectrum(params_test)
t_raw = (time.perf_counter() - t0) * 1e3

# JIT-compiled
jit_predict = jax.jit(model_param.predict_spectrum)
_ = jit_predict(params_test)  # compile
t0 = time.perf_counter()
for _ in range(1000):
    _ = jit_predict(params_test)
    _.block_until_ready()
t_jit = (time.perf_counter() - t0) / 1000 * 1e6

print(f"Forward model: {t_raw:.1f} ms (raw)  →  {t_jit:.0f} µs (JIT-compiled)")

# %%
# Generate a mock galaxy spectrum
key = jax.random.PRNGKey(42)
true_params_param = spec_param.sample(key)
# Override tsnorm to a typical star-forming galaxy (still forming stars now)
true_params_param = {**true_params_param}
true_params_param["sfh_tsnorm_log_total_mass"] = jnp.array(10.5)
true_params_param["sfh_tsnorm_peak_lbt_gyr"] = jnp.array(3.0)
true_params_param["sfh_tsnorm_width_gyr"] = jnp.array(3.0)
true_params_param["sfh_tsnorm_skew"] = jnp.array(0.3)
true_params_param["sfh_tsnorm_trunc"] = jnp.array(2.0)
mock_param = model_param.mock_spectrum(true_params_param, WAVE_OBS, snr=30.0, key=key)

print("True parameters:")
for name in spec_param.free_params:
    print(f"  {name:30s} = {float(true_params_param[name]):.4f}")

# %%
# --- FIGURE 1: The Mock Spectrum ---
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.errorbar(
    np.array(WAVE_OBS),
    np.array(mock_param.flux_obs),
    yerr=np.array(mock_param.noise),
    fmt=".",
    ms=3,
    color=COLORS["data"],
    alpha=0.6,
    label="Observed (SNR = 30)",
    zorder=2,
)
ax.plot(
    np.array(WAVE_OBS),
    np.array(mock_param.flux_true),
    color=COLORS["truth"],
    lw=1.5,
    label="Truth (noiseless)",
    zorder=3,
)
# Annotate key spectral features
for feat_name, feat_wave in SPECTRAL_FEATURES.items():
    wave_obs_feat = feat_wave * (1 + 0.1)  # z = 0.1
    if 3800 < wave_obs_feat < 9200:
        ax.axvline(wave_obs_feat, color="gray", ls=":", lw=0.5, alpha=0.5)
        ax.text(
            wave_obs_feat, ax.get_ylim()[1] * 0.95, feat_name,
            fontsize=6, ha="center", va="top", rotation=90, color="gray",
        )
ax.set_xlabel("Observed wavelength [Å]")
ax.set_ylabel("Flux density")
ax.legend(fontsize=8, loc="upper right")
ax.set_title("Mock Rest-Frame Spectrum at z = 0.1 (SNR = 30)")
fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "fig01_mock_spectrum_param.png"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### Fitting with native_geovi
#
# native_geovi is tengri's default inference method: geometric Variational
# Inference (geoVI; Frank et al. 2021) reimplemented as a fully JIT-compiled
# JAX program. It constructs a coordinate transform that flattens the posterior
# geometry — straightening curved degeneracies like the age–dust banana — then
# draws samples. The JIT engine eliminates all Python overhead, achieving a
# 500× speedup over standard implementations.

# %%
# MAP initialization + native_geovi inference
forward_param = ForwardModel.build(sed=model_param)

t0 = time.perf_counter()
result_map_param = forward_param.fit(
    mock_param.flux_obs, mock_param.noise, method="map", n_steps=500, verbose=False
)
t_map = time.perf_counter() - t0

# XLA compilation (one-time cost, cached on disk for future sessions)
t0_compile = time.perf_counter()
forward_param.prewarm(method="vi")
t_compile = time.perf_counter() - t0_compile

# Inference runtime (this is what you pay per galaxy after compilation)
t0 = time.perf_counter()
result_geovi_param = forward_param.fit(
    mock_param.flux_obs, mock_param.noise,
    method="vi",
    n_iterations=15,
    n_samples=6,
    n_seeds=5,
    n_posterior_samples=10000,
    verbose=False,
)
t_geovi = time.perf_counter() - t0

print(f"XLA compile: {t_compile:.1f}s (one-time, cached)")
print(f"MAP init:    {t_map:.1f}s")
print(f"native_geovi: {t_geovi:.1f}s  ← runtime per galaxy")

# %%
# --- FIGURE 2: Spectral Fit ---
spec_samples = []
n_draws = 50
sample_keys = jax.random.split(jax.random.PRNGKey(0), n_draws)
for i in range(n_draws):
    idx = i % len(result_geovi_param.samples[spec_param.free_params[0]])
    draw_params = {k: v[idx] for k, v in result_geovi_param.samples.items()}
    spec_draw = model_param.predict_spectrum(draw_params)
    spec_samples.append(np.array(spec_draw))
spec_samples = np.array(spec_samples)
spec_median = np.median(spec_samples, axis=0)

fig, (ax_fit, ax_res) = plt.subplots(
    2, 1, figsize=(10, 5), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
)
wave_np = np.array(WAVE_OBS)
obs_np = np.array(mock_param.flux_obs)
noise_np = np.array(mock_param.noise)
true_np = np.array(mock_param.flux_true)

# Top: spectral fit
ax_fit.errorbar(
    wave_np, obs_np, yerr=noise_np, fmt=".", ms=2, color=COLORS["data"],
    alpha=0.4, zorder=1,
)
for s in spec_samples[:50]:
    ax_fit.plot(wave_np, s, color=COLORS["vi"], alpha=0.03, lw=0.5, zorder=2)
ax_fit.plot(wave_np, spec_median, color=COLORS["vi"], lw=1.5, label="native_geovi median", zorder=3)
ax_fit.plot(wave_np, true_np, color=COLORS["truth"], lw=1, ls="--", label="Truth", zorder=4)
ax_fit.legend(fontsize=8)
ax_fit.set_ylabel("Flux density")

# Bottom: residuals
residuals = (obs_np - spec_median) / noise_np
ax_res.scatter(wave_np, residuals, s=2, c=COLORS["data"], alpha=0.5)
ax_res.axhline(0, color="k", lw=0.5)
ax_res.axhspan(-1, 1, alpha=0.1, color="gray")
ax_res.axhspan(-2, 2, alpha=0.05, color="gray")
ax_res.set_ylabel(r"$(f_{\rm obs} - f_{\rm model}) / \sigma$")
ax_res.set_xlabel("Observed wavelength [Å]")
ax_res.set_ylim(-4, 4)

chi2 = np.sum(residuals**2) / len(residuals)
ax_fit.set_title(f"Spectral Fit — native_geovi (reduced $\\chi^2$ = {chi2:.2f})")
fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "fig02_spectral_fit_param.png"), dpi=150, bbox_inches="tight")
plt.show()

# %%
# --- FIGURE 3: SFH Recovery ---
fig, ax = plt.subplots(figsize=(8, 4))
plot_sfh(model_param, result_geovi_param, true_params=true_params_param, ax=ax,
         color=COLORS["vi"], label="vi", method="geoVI")
ax.set_title("SFH Recovery — Parametric (D = 7)")
# 200 Myr inset — truth + posterior SFH draws
sfh_true_param = model_param.predict_sfh(true_params_param)
t_gyr_p = np.array(sfh_true_param["t_gyr"])
sfr_key_p = "sfr_full" if model_param.spec.stochastic else "sfr_mean"
sfr_true_p = np.array(sfh_true_param[sfr_key_p])
inset = ax.inset_axes([0.58, 0.58, 0.38, 0.38])
mask_200 = t_gyr_p < 0.2
if hasattr(t_gyr_p, "__len__") and np.any(mask_200):
    t_inset = t_gyr_p[mask_200] * 1e3  # Gyr → Myr
    # Posterior SFH draws
    if result_geovi_param.samples is not None:
        n_samp = len(next(iter(result_geovi_param.samples.values())))
        sfh_draws = []
        for i in range(n_samp):
            s_i = {k: result_geovi_param.samples[k][i] for k in result_geovi_param.samples}
            sfh_draws.append(np.array(model_param.predict_sfh(s_i)[sfr_key_p])[mask_200])
        sfh_arr = np.array(sfh_draws)
        lo, hi = np.percentile(sfh_arr, [16, 84], axis=0)
        median = np.median(sfh_arr, axis=0)
        inset.fill_between(t_inset, lo, hi, color=COLORS["vi"], alpha=0.3, lw=0)
        inset.plot(t_inset, median, color=COLORS["vi"], lw=1.2, label="Posterior")
    else:
        sfh_fit = model_param.predict_sfh(result_geovi_param.params)
        inset.plot(t_inset, np.array(sfh_fit[sfr_key_p])[mask_200], color=COLORS["vi"], lw=1.2, ls="--", label="MAP")
    inset.plot(t_inset, sfr_true_p[mask_200], color=COLORS["truth"], lw=1.5, label="Truth")
    inset.set_xlabel("Lookback [Myr]", fontsize=6)
    inset.set_ylabel("SFR", fontsize=6)
    inset.tick_params(labelsize=5)
    inset.set_xlim(0, 200)
    inset.legend(fontsize=5, loc="upper right")
fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "fig03_sfh_param.png"), dpi=150, bbox_inches="tight")
plt.show()

# %%
# --- FIGURE 4: Corner Plot ---
fig = safe_corner(result_geovi_param, truths=true_params_param)
if fig is not None:
    fig.suptitle("Parametric Posterior — native_geovi", y=1.02)
    plt.savefig(os.path.join(FIGDIR, "fig04_corner_param.png"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### Validation: Does geoVI Match Exact MCMC?
#
# geoVI is approximate — it's variational, not exact. To verify it's reliable,
# we compare with NUTS (No-U-Turn Sampler, Hoffman & Gelman 2014), the
# gold-standard exact sampler for low-dimensional problems. For D = 7,
# they should agree.

# %%
# Run NUTS from MAP initialization
t0 = time.perf_counter()
result_nuts_param = forward_param.fit(
    mock_param.flux_obs, mock_param.noise,
    method="mcmc_nuts",
    n_warmup=500,
    n_samples=1000,
    init_from=result_map_param,
    verbose=False,
)
t_nuts = time.perf_counter() - t0
print(f"NUTS: {t_nuts:.1f}s")

# Convergence diagnostics
ct = convergence_table(
    {"vi": result_geovi_param, "NUTS": result_nuts_param}
)

# %% [markdown]
# ### Parameter recovery

# %%
print(f"{'Parameter':<32s} {'True':>8s} {'Median':>8s} {'16%':>8s} {'84%':>8s} {'Status':>6s}")
print("-" * 76)
for name in spec_param.free_params:
    truth = float(true_params_param[name])
    lo, med, hi = np.percentile(result_geovi_param.samples[name], [16, 50, 84])
    covered = "\u2713" if lo <= truth <= hi else "MISS"
    print(f"  {name:<30s} {truth:8.3f} {med:8.3f} {lo:8.3f} {hi:8.3f} {covered:>6s}")

# %%
# --- FIGURE 5: native_geovi vs NUTS ---
fig = plot_corner_comparison(
    [result_geovi_param, result_nuts_param],
    labels=["vi", "NUTS"],
    colors=[COLORS["vi"], COLORS["mcmc_nuts"]],
    truths=true_params_param,
)
if fig is not None:
    fig.suptitle(
        f"native_geovi ({t_geovi:.1f}s) vs NUTS ({t_nuts:.1f}s) — D = 7",
        y=1.02,
    )
    plt.savefig(os.path.join(FIGDIR, "fig05_geovi_vs_nuts.png"), dpi=150, bbox_inches="tight")
plt.show()

# %%
# Speed comparison
print("\n  Method         | Wall Clock | Effective Samples | ESS/sec")
print("  " + "-" * 60)
for name, res, t in [
    ("vi", result_geovi_param, t_geovi),
    ("NUTS", result_nuts_param, t_nuts),
]:
    n_samp = len(next(iter(res.samples.values()))) if res.samples else 0
    print(f"  {name:<16s} | {t:>8.1f} s | {n_samp:>17d} | {n_samp / t:>7.0f}")

# %% [markdown]
# **Part A Takeaway**: For a 7-parameter model, native_geovi and NUTS give the
# same answer. native_geovi is an order of magnitude faster. For low-dimensional
# problems where you want exact MCMC guarantees, NUTS works — but it doesn't
# scale. Now let's see what happens when dimensionality explodes.

# %% [markdown]
# ## Part B: A Bursty Galaxy — 137 Parameters
#
# Real galaxies don't form stars smoothly. Star formation fluctuates on
# timescales from ~1 Myr (molecular cloud collapse) to ~1 Gyr (mergers,
# quenching). tengri models this burstiness as a Gaussian process controlled
# by a power spectral density (PSD). The PSD has two physical parameters:
# σ_PS (amplitude of fluctuations) and τ_PS (coherence timescale). The GP
# field adds 128 correlated latent dimensions. Total: 9 physical + 128 GP = 137
# free parameters.

# %% [markdown]
# This high dimensionality is exactly where standard MCMC methods like NUTS
# break down — the curse of dimensionality means chains mix too slowly. But
# because our entire model is differentiable, native_geovi exploits gradient
# information to navigate this space efficiently.

# %%
# Define the stochastic parameter specification
spec_stoch = Parameters(
    sfh_tsnorm_log_total_mass=Uniform(7.0, 12.5),
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
print(f"Stochastic model: {spec_stoch.n_free} free parameters")
print(f"  Physical: {len([p for p in spec_stoch.free_params if 'xi' not in p])}")
print(f"  GP latent: {len([p for p in spec_stoch.free_params if 'xi' in p])}")

# %%
# Create stochastic model with spectroscopic precomputation
model_stoch = SEDModel(spec_stoch, ssp_data, observation=obs)
model_stoch.precompute_spectroscopy(WAVE_OBS)

# %%
# Generate a bursty mock galaxy — supernova-feedback regime
key = jax.random.PRNGKey(123)
true_params_stoch = spec_stoch.sample(key)
# Override to a typical star-forming galaxy with dramatic burstiness
true_params_stoch = {**true_params_stoch}
true_params_stoch["sfh_tsnorm_log_total_mass"] = jnp.array(10.5)
true_params_stoch["sfh_tsnorm_peak_lbt_gyr"] = jnp.array(3.0)
true_params_stoch["sfh_tsnorm_width_gyr"] = jnp.array(3.0)
true_params_stoch["sfh_tsnorm_skew"] = jnp.array(0.3)
true_params_stoch["sfh_tsnorm_trunc"] = jnp.array(2.0)
true_params_stoch["sfh_field_psd_sigma"] = jnp.array(2.0)
true_params_stoch["sfh_field_psd_tau_myr"] = jnp.array(20.0)

mock_stoch = model_stoch.mock_spectrum(
    true_params_stoch, WAVE_OBS, snr=30.0, key=jax.random.fold_in(key, 1)
)
print("True PSD parameters:")
print(f"  σ_PS = {float(true_params_stoch['sfh_field_psd_sigma']):.1f}")
print(f"  τ_PS = {float(true_params_stoch['sfh_field_psd_tau_myr']):.0f} Myr")

# %%
# --- FIGURE 6: The Bursty Truth ---
sfh_true = model_stoch.predict_sfh(true_params_stoch)
t_gyr = np.array(sfh_true["t_gyr"])
sfr_full = np.array(sfh_true["sfr_full"])
sfr_mean = np.array(sfh_true["sfr_mean"])

fig, (ax_sfh, ax_spec) = plt.subplots(1, 2, figsize=(12, 4))

# Left: True SFH
ax_sfh.plot(t_gyr, sfr_full, color=COLORS["truth"], lw=1.5, label="Full SFH (with GP)")
ax_sfh.plot(t_gyr, sfr_mean, color=COLORS["sfh_mean"], lw=1, ls="--", label="Mean SFH (secular)")
ax_sfh.set_xlabel("Lookback time [Gyr]")
ax_sfh.set_ylabel(r"SFR [$M_\odot\,{\rm yr}^{-1}$]")
ax_sfh.set_xlim(0, 13.5)
ax_sfh.legend(fontsize=8)
ax_sfh.set_title("True Bursty SFH (σ = 2.0, τ = 20 Myr)")
# 200 Myr inset
inset = ax_sfh.inset_axes([0.55, 0.55, 0.4, 0.4])
mask_200 = t_gyr < 0.2
inset.plot(t_gyr[mask_200] * 1e3, sfr_full[mask_200], color=COLORS["truth"], lw=1)
inset.plot(t_gyr[mask_200] * 1e3, sfr_mean[mask_200], color=COLORS["sfh_mean"], lw=0.8, ls="--")
inset.set_xlabel("Lookback [Myr]", fontsize=6)
inset.set_ylabel("SFR", fontsize=6)
inset.tick_params(labelsize=5)
inset.set_xlim(0, 200)

# Right: Mock spectrum
ax_spec.errorbar(
    np.array(WAVE_OBS), np.array(mock_stoch.flux_obs), yerr=np.array(mock_stoch.noise),
    fmt=".", ms=2, color=COLORS["data"], alpha=0.5,
)
ax_spec.plot(np.array(WAVE_OBS), np.array(mock_stoch.flux_true), color=COLORS["truth"], lw=1)
ax_spec.set_xlabel("Observed wavelength [Å]")
ax_spec.set_ylabel("Flux density")
ax_spec.set_title("Mock Spectrum (SNR = 30)")

fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "fig06_bursty_truth.png"), dpi=150, bbox_inches="tight")
plt.show()

# %%
# MAP + native_geovi on the stochastic model
forward_stoch = ForwardModel.build(sed=model_stoch)

t0 = time.perf_counter()
result_map_stoch = forward_stoch.fit(
    mock_stoch.flux_obs, mock_stoch.noise, method="map", n_steps=1000, verbose=False
)
t_map_s = time.perf_counter() - t0

# XLA compilation (one-time cost, cached on disk)
t0_compile_s = time.perf_counter()
forward_stoch.prewarm(method="vi")
t_compile_s = time.perf_counter() - t0_compile_s

# Inference runtime
t0 = time.perf_counter()
result_geovi_stoch = forward_stoch.fit(
    mock_stoch.flux_obs, mock_stoch.noise,
    method="vi",
    n_iterations=20,
    n_samples=6,
    n_seeds=5,
    n_posterior_samples=10000,
    verbose=False,
)
t_geovi_s = time.perf_counter() - t0

print(f"\n{'=' * 55}")
print(f"  137-dimensional posterior in {t_geovi_s:.1f}s runtime")
print(f"{'=' * 55}")
print(f"  XLA compile: {t_compile_s:.1f}s (one-time, cached)")
print(f"  MAP init:    {t_map_s:.1f}s")
print(f"  native_geovi: {t_geovi_s:.1f}s  ← runtime per galaxy")

# %%
# --- FIGURE 7: Stochastic SFH Recovery (THE MONEY FIGURE) ---
fig, ax = plt.subplots(figsize=(8, 4))
plot_sfh(
    model_stoch, result_geovi_stoch, true_params=true_params_stoch,
    ax=ax, color=COLORS["vi"], label="vi", method="geoVI",
    show_mean_sfh=True,
)
ax.set_title(
    f"Stochastic SFH Recovery — 137 parameters, {t_geovi_s:.1f}s",
    fontweight="bold",
)
# 200 Myr inset — truth + posterior SFH draws
sfh_true_s = model_stoch.predict_sfh(true_params_stoch)
t_gyr_s = np.array(sfh_true_s["t_gyr"])
sfr_key_s = "sfr_full"
sfr_true_s = np.array(sfh_true_s[sfr_key_s])
inset = ax.inset_axes([0.58, 0.58, 0.38, 0.38])
mask_200 = t_gyr_s < 0.2
if hasattr(t_gyr_s, "__len__") and np.any(mask_200):
    t_inset = t_gyr_s[mask_200] * 1e3
    if result_geovi_stoch.samples is not None:
        n_samp = len(next(iter(result_geovi_stoch.samples.values())))
        sfh_draws = []
        for i in range(n_samp):
            s_i = {k: result_geovi_stoch.samples[k][i] for k in result_geovi_stoch.samples}
            sfh_draws.append(np.array(model_stoch.predict_sfh(s_i)[sfr_key_s])[mask_200])
        sfh_arr = np.array(sfh_draws)
        lo, hi = np.percentile(sfh_arr, [16, 84], axis=0)
        median = np.median(sfh_arr, axis=0)
        inset.fill_between(t_inset, lo, hi, color=COLORS["vi"], alpha=0.3, lw=0)
        inset.plot(t_inset, median, color=COLORS["vi"], lw=1.2, label="Posterior")
    else:
        sfh_fit = model_stoch.predict_sfh(result_geovi_stoch.params)
        inset.plot(t_inset, np.array(sfh_fit[sfr_key_s])[mask_200], color=COLORS["vi"], lw=1.2, ls="--", label="MAP")
    inset.plot(t_inset, sfr_true_s[mask_200], color=COLORS["truth"], lw=1.5, label="Truth")
    inset.set_xlabel("Lookback [Myr]", fontsize=6)
    inset.set_ylabel("SFR", fontsize=6)
    inset.tick_params(labelsize=5)
    inset.set_xlim(0, 200)
    inset.legend(fontsize=5, loc="upper right")
fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "fig07_sfh_stochastic_money.png"), dpi=150, bbox_inches="tight")
plt.show()

# %%
# --- FIGURE 8: Spectral Fit (stochastic) ---
spec_samples_s = []
for i in range(50):
    idx = i % len(result_geovi_stoch.samples[spec_stoch.free_params[0]])
    draw_params = {k: v[idx] for k, v in result_geovi_stoch.samples.items()}
    spec_draw = model_stoch.predict_spectrum(draw_params)
    spec_samples_s.append(np.array(spec_draw))
spec_samples_s = np.array(spec_samples_s)
spec_median_s = np.median(spec_samples_s, axis=0)

fig, (ax_fit, ax_res) = plt.subplots(
    2, 1, figsize=(10, 5), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
)
obs_s = np.array(mock_stoch.flux_obs)
noise_s = np.array(mock_stoch.noise)
true_s = np.array(mock_stoch.flux_true)

ax_fit.errorbar(wave_np, obs_s, yerr=noise_s, fmt=".", ms=2, color=COLORS["data"], alpha=0.4)
for s in spec_samples_s[:50]:
    ax_fit.plot(wave_np, s, color=COLORS["vi"], alpha=0.03, lw=0.5)
ax_fit.plot(wave_np, spec_median_s, color=COLORS["vi"], lw=1.5, label="native_geovi median")
ax_fit.plot(wave_np, true_s, color=COLORS["truth"], lw=1, ls="--", label="Truth")
ax_fit.legend(fontsize=8)
ax_fit.set_ylabel("Flux density")

residuals_s = (obs_s - spec_median_s) / noise_s
ax_res.scatter(wave_np, residuals_s, s=2, c=COLORS["data"], alpha=0.5)
ax_res.axhline(0, color="k", lw=0.5)
ax_res.axhspan(-1, 1, alpha=0.1, color="gray")
ax_res.axhspan(-2, 2, alpha=0.05, color="gray")
ax_res.set_ylabel(r"$(f_{\rm obs} - f_{\rm model}) / \sigma$")
ax_res.set_xlabel("Observed wavelength [Å]")
ax_res.set_ylim(-4, 4)

chi2_s = np.sum(residuals_s**2) / len(residuals_s)
ax_fit.set_title(f"Spectral Fit — Stochastic D = 137 (reduced $\\chi^2$ = {chi2_s:.2f})")
fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "fig08_spectral_fit_stochastic.png"), dpi=150, bbox_inches="tight")
plt.show()

# %%
# --- FIGURE 9: Physical Parameter Corner ---
phys_params = [p for p in spec_stoch.free_params if "xi" not in p]
fig = safe_corner(result_geovi_stoch, truths=true_params_stoch, params=phys_params)
if fig is not None:
    fig.suptitle("Physical Parameters — Stochastic (D = 137)", y=1.02)
    plt.savefig(os.path.join(FIGDIR, "fig09_corner_stochastic.png"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### The Ray Tracing Sampler: Exact MCMC at High D
#
# What if you want exact, unbiased posteriors at D = 137? The Ray Tracing
# Sampler (Behroozi 2025) is a physics-inspired MCMC method that propagates
# "light rays" through parameter space, using Snell's law to bend trajectories
# toward high-likelihood regions. Unlike NUTS, it's ~250× more tolerant of
# gradient noise and works efficiently at D = 137. It's the primary exact
# MCMC method in tengri.

# %%
# Ray Tracing on the stochastic model
t0 = time.perf_counter()
result_rt_stoch = forward_stoch.fit(
    mock_stoch.flux_obs, mock_stoch.noise,
    method="mcmc_raytrace",
    init_from=result_map_stoch,
    n_burnin=200,
    n_steps=2000,
    step_size=0.05,
    n_leapfrog_steps=50,
    verbose=False,
)
t_rt_s = time.perf_counter() - t0

acc = result_rt_stoch.diagnostics.get("acceptance_rate", float("nan"))
print(f"Ray Tracing: {t_rt_s:.1f}s, acceptance = {acc:.1%}")

# %%
# Convergence diagnostics for D=137 methods
ct_stoch = convergence_table(
    {"vi": result_geovi_stoch, "Ray Tracing": result_rt_stoch}
)

# %%
# --- FIGURE 10: All methods on one plot ---
fig, ax = plt.subplots(figsize=(10, 5))

# Truth first — bold, on top
sfh_true_cmp = model_stoch.predict_sfh(true_params_stoch)
t_gyr_cmp = np.array(sfh_true_cmp["t_gyr"])
ax.plot(t_gyr_cmp, sfh_true_cmp["sfr_full"], color=COLORS["truth"],
        lw=3.0, zorder=10, label="Truth")
ax.plot(t_gyr_cmp, sfh_true_cmp["sfr_mean"], color=COLORS["truth"],
        lw=1.5, ls=":", alpha=0.4, zorder=10)

# Overlay each method's posterior SFH as 68% CI band
for result, color, label in [
    (result_geovi_stoch, COLORS["vi"], f"native_geovi ({t_geovi_s:.1f}s)"),
    (result_rt_stoch, COLORS["rt"], f"Ray Tracing ({t_rt_s:.1f}s)"),
]:
    if result.samples is not None:
        n_samp = len(next(iter(result.samples.values())))
        sfh_draws = []
        for i in range(n_samp):
            s_i = {k: result.samples[k][i] for k in result.samples}
            sfh_draws.append(np.array(model_stoch.predict_sfh(s_i)["sfr_full"]))
        sfh_arr = np.array(sfh_draws)
        lo, hi = np.percentile(sfh_arr, [16, 84], axis=0)
        median = np.median(sfh_arr, axis=0)
        ax.fill_between(t_gyr_cmp, lo, hi, color=color, alpha=0.2, lw=0, label=f"{label} (68% CI)")
        ax.plot(t_gyr_cmp, median, color=color, lw=1.5, zorder=5)

ax.set_xlabel(r"$\mathrm{Lookback\ time\ /\ Gyr}$")
ax.set_ylabel(r"$\mathrm{SFR\ /\ M_\odot\ yr^{-1}}$")
ax.set_xlim(0, 13.5)
ax.set_ylim(bottom=0.)
ax.legend(loc="upper right")
ax.set_title("Stochastic SFH Recovery — native_geovi vs Ray Tracing (D = 137)")

# 200 Myr inset with both posteriors
inset = ax.inset_axes([0.55, 0.55, 0.4, 0.4])
mask_cmp = t_gyr_cmp < 0.2
if np.any(mask_cmp):
    t_inset = t_gyr_cmp[mask_cmp] * 1e3
    for result, color, label in [
        (result_geovi_stoch, COLORS["vi"], "geoVI"),
        (result_rt_stoch, COLORS["rt"], "RT"),
    ]:
        if result.samples is not None:
            n_samp = len(next(iter(result.samples.values())))
            draws = []
            for i in range(n_samp):
                s_i = {k: result.samples[k][i] for k in result.samples}
                draws.append(np.array(model_stoch.predict_sfh(s_i)["sfr_full"])[mask_cmp])
            arr = np.array(draws)
            lo, hi = np.percentile(arr, [16, 84], axis=0)
            med = np.median(arr, axis=0)
            inset.fill_between(t_inset, lo, hi, color=color, alpha=0.25, lw=0)
            inset.plot(t_inset, med, color=color, lw=1.2, label=label)
    inset.plot(t_inset, np.array(sfh_true_cmp["sfr_full"])[mask_cmp],
               color=COLORS["truth"], lw=2.0, label="Truth")
    inset.set_xlabel("Lookback [Myr]", fontsize=6)
    inset.set_ylabel("SFR", fontsize=6)
    inset.tick_params(labelsize=5)
    inset.set_xlim(0, 200)
    inset.legend(fontsize=5, loc="upper right")

fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "fig10_geovi_vs_rt.png"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Summary

# %%
# Summary timing table
print("\n  Summary")
print("  " + "=" * 75)
print(f"  {'SEDModel':<20s} {'D':>4s}  {'Method':<16s} {'Compile':>8s} {'Runtime':>8s}  Notes")
print("  " + "-" * 75)
print(f"  {'Parametric':<20s} {'7':>4s}  {'native_geovi':<16s} {t_compile:>7.1f}s {t_geovi:>7.1f}s  Default")
print(f"  {'Parametric':<20s} {'7':>4s}  {'NUTS':<16s} {'':>8s} {t_nuts:>7.1f}s  Exact, gold standard")
print(f"  {'Stochastic':<20s} {'137':>4s}  {'native_geovi':<16s} {t_compile_s:>7.1f}s {t_geovi_s:>7.1f}s  Default")
print(f"  {'Stochastic':<20s} {'137':>4s}  {'Ray Tracing':<16s} {'':>8s} {t_rt_s:>7.1f}s  Exact (Behroozi 2025)")
print("  " + "=" * 75)
print("\n  Compile is one-time (cached on disk). Runtime is per galaxy.")
print(f"  Headline: 137D posterior in {t_geovi_s:.0f}s runtime with native_geovi.")

# %% [markdown]
# ## What You Just Did
#
# 1. Fit a smooth 7D SFH and verified native_geovi matches NUTS.
# 2. Fit a bursty 137D SFH — something standard samplers can't handle — in seconds.
# 3. Recovered star formation burst features from a single galaxy spectrum.
# 4. Validated with the exact Ray Tracing Sampler (Behroozi 2025).

# %% [markdown]
# ## What's Next
#
# - **tutorials/02** — Learn the SEDModel, ForwardModel, Posterior classes in detail.
# - **tutorials/03** — Understand the IFT/PSD/GP machinery.
# - **tutorials/04** — See how SFH becomes an observable SED.
# - **tutorials/05** — Check your model before fitting (prior predictive).
# - **demonstrations/** — Catalog-scale photometry, hierarchical population
#   inference, real data fitting, and more.
