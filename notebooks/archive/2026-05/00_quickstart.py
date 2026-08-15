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
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Quickstart
#
# **What you'll learn:**
# - Build a `SEDModel` with free parameters and priors
# - Generate mock photometry from a known galaxy
# - Run NUTS (exact, JIT-compiled inference) to recover parameters
# - Validate parameter recovery via corner plots and convergence diagnostics
#
# **Prerequisites:** None (standalone entry point).
# **Next:** [`02_sed_anatomy.py`](02_sed_anatomy.py) for component decomposition.
#
# ---
#
# tengri is a JAX SED fitting framework: one unified forward model, every major inference backend, fully differentiable and JIT-compiled.
#
# A **7-parameter galaxy SED fit** from photometric mock data. The same differentiable JAX code handles:
# - **Forward model:** stellar continuum (DSPS SSP), nebular emission, dust attenuation, infrared re-radiation.
# - **Inference:** NUTS (No-U-Turn Sampler), JIT-compiled, exact sampling.
# - **Diagnostics:** corner plot, SFH recovery, posterior predictive checks.
#
# **Physics:** Star formation history as truncated skew-normal (7 free parameters); dust as two-component attenuation (birth cloud + ISM);
# metallicity from SSP grid; nebular continuum from ionizing spectrum.
#
# **Why differentiable:** Every inference method shares the same scalar objective—the information Hamiltonian
# $\mathcal{H} = \frac{1}{2}\chi^2 + \frac{1}{2}\xi^\top\xi$—ensuring consistency.
# The forward model is pure JAX: fully differentiable, JIT-compiled to GPU/CPU, composable with any optimizer.

# %% [markdown]
# **Spine location:** `notebooks/00_quickstart.py` (not `notebook_code/`).

# %%
import os
import sys
import time
import warnings

# Must be set before JAX initializes its XLA backend (first computation, not import).
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.45")

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

import jax
import jax.numpy as jnp
import matplotlib

# Use non-interactive backend when run as a plain script (not in Jupyter).
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
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
    Spectroscopy,
    Uniform,
    load_ssp_data,
)

# Locate ``notebooks/_plot_style.py`` and ``data/`` root (nbclient cwd is often wrong).
import importlib.util

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

FIGDIR = os.path.join("notebooks", "figures", "quickstart")
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
# tengri logo banner — prints the hex-spiral mark (default size).
import tengri as tg

tg.print_logo()
print(f"tengri {tg.__version__}")

# %%
# Load SSP templates and multi-wavelength photometry
ssp_data = load_ssp_data("data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")
from tengri.observation import Photometry

# Try to load candidate filters (UV to radio); fall back to 2MASS if unavailable
_candidate_filters = [
    "galex_fuv",
    "galex_nuv",
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
    "herschel_pacs70",
    "herschel_pacs160",
]
phot_bands_list = []
for band in _candidate_filters:
    try:
        Photometry.from_names([band])
        phot_bands_list.append(band)
    except Exception:
        pass
if not phot_bands_list:
    phot_bands_list = ["twomass_j", "twomass_h", "twomass_ks"]

phot_obs = Photometry.from_names(phot_bands_list, cache_dir="data/filters")
obs = Observation(photometry=phot_obs)
print(
    f"SSP: {ssp_data.ssp_flux.shape[0]} Z × {ssp_data.ssp_flux.shape[1]} ages × {ssp_data.ssp_flux.shape[-1]} λ"
)
print(f"Photometry ({phot_obs.n_filters} bands): {', '.join(phot_obs.names)}")

# %% [markdown]
# ## Part 0: One SED from X-ray to radio
#
# The fits below use only a **small slice** in wavelength. First, plot the **full
# panchromatic** prediction (stellar + nebular in the SSP, dust attenuation and IR
# re-radiation, radio and X-ray scalings) on a single log–log axis. The shaded band
# marks the optical window used in Part A.

# %%
warnings.filterwarnings(
    "ignore",
    message=".*BakedInBackend.*",
    category=UserWarning,
)
_z_q = 0.1
spec_pan = Parameters(
    mean_sfh_type="dense_basis",
    sfh_db_log_total_mass=Fixed(10.0),
    sfh_db_log_sfr_inst=Fixed(0.5),
    sfh_db_tx_frac_0=Fixed(0.2),
    sfh_db_tx_frac_1=Fixed(0.4),
    sfh_db_tx_frac_2=Fixed(0.4),
    met_logzsol=Fixed(0.0),
    dust_tau_bc=Fixed(0.8),
    dust_tau_diff=Fixed(0.4),
    dust_slope=Fixed(-0.7),
    dust_emission="draine_li2007",
    dust_T=Fixed(35.0),
    dust_qpah=Fixed(2.5),
    nebular_ssp=True,
    radio=True,
    xray=True,
    radio_q_ir=Fixed(2.64),
    redshift=Fixed(_z_q),
)
model_pan = SEDModel(spec_pan, ssp_data, observation=None)
params_pan = spec_pan.sample(jax.random.PRNGKey(101))
wave_pan = jnp.logspace(0.8, 7.15, 900)
sed_pan = model_pan.predict_spectrum(params_pan, wave_pan)
wave_pan_np = np.array(wave_pan)
sed_pan_np = np.array(sed_pan)
valid = np.isfinite(sed_pan_np) & (sed_pan_np > 0)

fig0, ax0 = plt.subplots(figsize=(12, 4.2))
ax0.loglog(wave_pan_np[valid], sed_pan_np[valid], color=COLORS.get("model", "C0"), lw=1.2)
ax0.axvspan(3800.0, 9200.0, alpha=0.25, color="0.5", label="Part A spectrum window (obs. Å)")
ax0.set_xlabel(r"Observed wavelength [$\mathrm{\AA}$]")
ax0.set_ylabel(r"$f_\nu$ [erg/s/cm$^2$/Hz]")
ax0.set_title("Panchromatic forward model (same SSP family as the fits below)")
ax0.set_xlim(float(wave_pan_np.min()), float(wave_pan_np.max()))
y_valid = sed_pan_np[valid]
ax0.set_ylim(float(y_valid.min()) * 0.5, float(y_valid.max()) * 2.0)
ax0.grid(True, alpha=0.3)
ax0.legend(loc="upper right", fontsize=10)
fig0.tight_layout()
# plt.savefig(os.path.join(FIGDIR, "fig00_panchromatic.png", dpi=300, bbox_inches="tight"), dpi=150, bbox_inches="tight")
plt.show()
del (
    model_pan,
    sed_pan,
    sed_pan_np,
    wave_pan,
    wave_pan_np,
)  # free SSP device memory before inference

# %%
# Define 7-parameter model (dense basis SFH + stellar mass, metallicity, dust)
spec_param = Parameters(
    sfh_db_log_total_mass=Uniform(8, 12),
    sfh_db_log_sfr_inst=Uniform(-2, 3),
    sfh_db_tx_frac_0=Uniform(0.05, 0.95),
    sfh_db_tx_frac_1=Uniform(0.05, 0.95),
    sfh_db_tx_frac_2=Uniform(0.05, 0.95),
    met_logzsol=Uniform(-2.0, 0.2),
    dust_tau_bc=Uniform(0.0, 2.0),
    dust_tau_diff=Uniform(0.0, 1.5),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.1),
    mean_sfh_type="dense_basis",
)
print(f"Free parameters ({spec_param.n_free}): {', '.join(spec_param.free_params)}")

# Create model with photometric precomputation
model_param = SEDModel(spec_param, ssp_data, observation=obs)

# Benchmark: forward model is fast
params_test = spec_param.sample(jax.random.PRNGKey(99))
t0 = time.perf_counter()
_ = model_param.predict_photometry(params_test)
t_raw = (time.perf_counter() - t0) * 1e3
jit_predict = jax.jit(model_param.predict_photometry)
_ = jit_predict(params_test)
t0 = time.perf_counter()
for _ in range(100):
    _ = jit_predict(params_test)
    _.block_until_ready()
t_jit = (time.perf_counter() - t0) / 100 * 1e6
print(f"Forward model: {t_raw:.1f} ms (raw) → {t_jit:.0f} µs (JIT)")

# %%
# Generate mock photometry: monotonically increasing SFH with high SFR (30 Msun/yr)
key = jax.random.PRNGKey(42)
true_params_param = spec_param.sample(key)
# Override to monotonically rising SFH: early low → now high SFR
true_params_param = {**true_params_param}
true_params_param["sfh_db_log_total_mass"] = jnp.array(10.8)
true_params_param["sfh_db_log_sfr_inst"] = jnp.array(1.48)  # log(30) ≈ 1.48 Msun/yr
true_params_param["sfh_db_tx_frac_0"] = jnp.array(0.1)  # Early epoch (low weight)
true_params_param["sfh_db_tx_frac_1"] = jnp.array(0.25)  # Middle epoch
true_params_param["sfh_db_tx_frac_2"] = jnp.array(
    0.65
)  # Recent epoch (high weight, rising profile)
true_params_param["met_logzsol"] = jnp.array(-0.1)  # Solar-ish metallicity
true_params_param["dust_tau_bc"] = jnp.array(0.5)
true_params_param["dust_tau_diff"] = jnp.array(0.3)
mock_param = model_param.mock(true_params_param, snr=50.0, key=key)

print("True parameters (monotonically rising SFH, SFR_inst = 30 Msun/yr):")
for name in spec_param.free_params:
    print(f"  {name:30s} = {float(true_params_param[name]):.4f}")

# %%
# --- FIGURE 1: Mock Multi-Wavelength Photometry ---
fig, ax = plt.subplots(figsize=(12, 4))
band_names = list(phot_obs.names)
band_idx = np.arange(len(band_names))
flux_true = np.array(mock_param.flux_true)
flux_obs = np.array(mock_param.flux_obs)
noise = np.array(mock_param.noise)

ax.errorbar(
    band_idx,
    flux_obs,
    yerr=noise,
    fmt="o",
    ms=7,
    color=COLORS["data"],
    alpha=0.7,
    label="Observed (SNR = 50)",
    zorder=2,
)
ax.plot(
    band_idx,
    flux_true,
    "s",
    ms=9,
    color=COLORS["truth"],
    alpha=0.8,
    label="Truth (noiseless)",
    zorder=3,
)

# Shade filter families based on wavelength (approximate)
n_bands = len(band_names)
if n_bands >= 2:
    ax.axvspan(-0.5, 1.5, alpha=0.08, color="purple", label="UV")
if n_bands >= 7:
    ax.axvspan(2.5, 6.5, alpha=0.08, color="cyan", label="Optical")
if n_bands >= 9:
    ax.axvspan(6.5, 9.5, alpha=0.08, color="red", label="NIR")
if n_bands >= 10:
    ax.axvspan(9.5, n_bands + 0.5, alpha=0.08, color="orange", label="MIR/FIR")

ax.set_xticks(band_idx)
ax.set_xticklabels(band_names, rotation=45, ha="right", fontsize=10)
ax.set_ylabel(r"$f_\nu$ [erg/s/cm$^2$/Hz]", fontsize=10)
ax.set_title(
    "Mock SED: Multi-Wavelength Photometry (Monotonically Rising SFH, SFR = 30 $M_\\odot$/yr)"
)
ax.legend(fontsize=10, loc="upper left", ncol=2)
ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
plt.show()

# %%
# Run NUTS (No-U-Turn Sampler) inference
os.environ["TENGRI_NO_BACKGROUND_COMPILE"] = "1"
fitter_param = Fitter(model_param, mock_param.flux_obs, mock_param.noise)
t0 = time.perf_counter()
result_mcmc = fitter_param.run("mcmc_nuts", n_warmup=500, n_samples=1000, verbose=False)
t_mcmc = time.perf_counter() - t0
print(f"NUTS: {t_mcmc:.1f}s (fast, exact, JIT-compiled)")

# %%
# --- FIGURE 2: NUTS Photometric Fit ---
phot_samples = []
n_draws = 50

# Draw from NUTS posterior
for i in range(n_draws):
    idx = i % len(result_mcmc.samples[spec_param.free_params[0]])
    draw_params = {k: v[idx] for k, v in result_mcmc.samples.items()}
    phot_draw = model_param.predict_photometry(draw_params)
    phot_samples.append(np.array(phot_draw))

phot_median = np.median(np.array(phot_samples), axis=0)

fig, ax = plt.subplots(figsize=(12, 5))
band_idx = np.arange(len(band_names))
obs_np = np.array(mock_param.flux_obs)
noise_np = np.array(mock_param.noise)
true_np = np.array(mock_param.flux_true)

# Data
ax.errorbar(
    band_idx,
    obs_np,
    yerr=noise_np,
    fmt="o",
    ms=8,
    color=COLORS["data"],
    alpha=0.7,
    label="Observed",
    zorder=3,
)

# Posterior samples
for s in phot_samples[:30]:
    ax.plot(band_idx, s, "-", color=COLORS["mcmc_nuts"], alpha=0.02, lw=0.8, zorder=1)

# Median
ax.plot(
    band_idx,
    phot_median,
    "D-",
    color=COLORS["mcmc_nuts"],
    ms=7,
    lw=2.5,
    label=f"NUTS median ({t_mcmc:.1f}s)",
    zorder=4,
)

# Truth
ax.plot(band_idx, true_np, "s", color=COLORS["truth"], ms=9, alpha=0.8, label="Truth", zorder=5)

ax.set_xticks(band_idx)
ax.set_xticklabels(band_names, rotation=45, ha="right", fontsize=10)
ax.set_ylabel(r"$f_\nu$ [erg/s/cm$^2$/Hz]")
ax.set_title("NUTS: Photometric Fit (Monotonic Rising SFH, SFR = 30 $M_\\odot$/yr)")
ax.legend(fontsize=10, loc="upper left")
ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
plt.show()

# %%
# --- FIGURE 3: SFH Recovery ---
fig, ax = plt.subplots(figsize=(10, 4))
plot_sfh(
    model_param,
    result_mcmc,
    true_params=true_params_param,
    ax=ax,
    color=COLORS["mcmc_nuts"],
    label="NUTS",
    method="NUTS",
)
ax.set_title("SFH Recovery: Monotonically Rising Profile (D = 7, NUTS)")
fig.tight_layout()
# plt.savefig(os.path.join(FIGDIR, "fig03_sfh_param.png", dpi=300, bbox_inches="tight"), dpi=150, bbox_inches="tight")
plt.show()

# %%
# --- FIGURE 4: Corner Plot ---
fig = plot_corner_comparison(
    [result_mcmc],
    labels=["NUTS"],
    colors=[COLORS["mcmc_nuts"]],
    truths=true_params_param,
)
if fig is not None:
    fig.suptitle("NUTS: Parametric Posterior (Monotonic Rising SFH, D = 7)", y=1.02)
plt.show()

# %%
# Convergence diagnostics and parameter recovery
ct = convergence_table({"NUTS": result_mcmc})
print("\nParameter Recovery:")
print(f"{'Parameter':<30s} {'True':>8s} {'Median':>8s} {'16–84%':>12s} {'Status':>6s}")
print("-" * 70)
for name in spec_param.free_params:
    truth = float(true_params_param[name])
    lo, med, hi = np.percentile(result_mcmc.samples[name], [16, 50, 84])
    covered = "✓" if lo <= truth <= hi else "MISS"
    print(f"  {name:<28s} {truth:8.3f} {med:8.3f} [{lo:6.3f}, {hi:6.3f}] {covered:>6s}")

n_mcmc = len(next(iter(result_mcmc.samples.values())))
print(f"\nNUTS: {t_mcmc:.1f}s, {n_mcmc} samples, {n_mcmc / t_mcmc:.0f} ESS/s")

# %%
print("\n✓ Quickstart complete: NUTS inference on mock photometry (7 parameters, <5s)")

# %% [markdown]
# ## What You Learned
#
# - End-to-end SED fitting: mock generation → inference → diagnostics
# - NUTS scales to 7 parameters on photometry with <5s runtime
# - Tight parameter recovery validates model architecture
#
# **Next:** [`02_sed_anatomy.py`](02_sed_anatomy.py) for panchromatic decomposition, then real-data workflows in 03–05.
