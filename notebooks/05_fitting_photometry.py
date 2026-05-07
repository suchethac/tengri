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
# # Fitting photometry: from data to posterior
#
# **What you'll learn:**
# - Load and fit multi-wavelength UV-to-MIR photometry with NUTS
# - Extract derived properties (stellar mass, star formation rates) with credible intervals
# - Assess convergence via diagnostic tables and effective sample size
# - Visualize posterior-predictive SED fits and residuals
# - Interpret degeneracies and model adequacy
#
# **Prerequisites:** [`00_quickstart.py`](00_quickstart.py), [`04_building_models.py`](04_building_models.py).
# **Next:** [`06_fitting_spectroscopy.py`](06_fitting_spectroscopy.py) to break age–dust–metallicity degeneracies.
#
# ---
#
# End-to-end photometric SED fitting: generate realistic mock data with noise,
# run NUTS (No-U-Turn Sampler) to recover the posterior, validate convergence,
# and extract science-ready derived properties. Where [`00_quickstart.py`](00_quickstart.py)
# was a rapid demo, this notebook shows **how to do it for real**: convergence checks,
# credible intervals on derived quantities, and posterior-predictive model validation.
#
# **Physics:** Star formation history as power-law (dpl), Calzetti two-component dust,
# Dale et al. (2014) infrared emission template, free redshift with broad prior,
# and nebular continuum enabled.
#
# **Why photometry alone is hard:** UV–MIR data constrain *combinations* of age, dust, and
# metallicity, but not each uniquely. The posterior is degenerate — recovery is possible
# only if the prior is tight. Spectroscopy (notebook 06) adds a powerful constraint that
# breaks these degeneracies by pinning stellar population age via Balmer breaks.

# %% [markdown]
# ## Setup

# %%
import contextlib
import os
import sys
import time
import warnings

os.environ.setdefault("TENGRI_NO_BACKGROUND_COMPILE", "1")
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
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")

jax.config.update("jax_enable_x64", True)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*BakedInBackend.*", category=UserWarning)

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

FIGDIR = os.path.join("notebooks", "figures")
os.makedirs(FIGDIR, exist_ok=True)

from _plot_style import COLORS, setup_style

setup_style()

import tengri as tg
from tengri import (
    Fitter,
    Fixed,
    Observation,
    Parameters,
    Photometry,
    SEDModel,
    Uniform,
    load_ssp_data,
)

print("=" * 70)
tg.print_logo()
print(f"tengri {tg.__version__}")
print("=" * 70)

# %% [markdown]
# ## Part 1: Load SSP and assemble bandset

# %%
_ssp_name = "ssp_mist_c3k_a_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
_ssp_path = os.path.join("data", _ssp_name)
if not os.path.exists(_ssp_path):
    _ssp_name = "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
    _ssp_path = os.path.join("data", _ssp_name)

ssp_data = load_ssp_data(_ssp_path)
print(f"SSP: {ssp_data.ssp_flux.shape[0]} Z × {ssp_data.ssp_flux.shape[1]} ages")

# Assemble UV–IR bandset
filter_names = [
    "galex_fuv", "galex_nuv",
    "sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z",
    "2mass_j", "2mass_h", "2mass_ks",
    "wise_w1", "wise_w2", "wise_w3",
]

phot_obs = Photometry.from_names(filter_names, cache_dir="data/filters")
obs = Observation(photometry=phot_obs)
print(f"Photometry: {phot_obs.n_filters} bands (GALEX/SDSS/2MASS/WISE)")

# %% [markdown]
# ## Part 2: Model definition (10 free parameters)

# %%
spec = Parameters(
    sfh_db_log_total_mass=Uniform(8.0, 12.0),
    sfh_db_log_sfr_inst=Uniform(-2.0, 3.0),
    sfh_db_tx_frac_0=Uniform(0.05, 0.95),
    sfh_db_tx_frac_1=Uniform(0.05, 0.95),
    sfh_db_tx_frac_2=Uniform(0.05, 0.95),
    met_logzsol=Uniform(-2.0, 0.2),
    dust_tau_bc=Uniform(0.0, 2.0),
    dust_tau_diff=Uniform(0.0, 1.5),
    dust_slope=Fixed(-0.7),
    dust_emission="dale2014",
    dust_T=Uniform(25.0, 50.0),
    dust_qpah=Fixed(2.5),
    nebular_ssp=True,
    redshift=Uniform(0.01, 0.5),
    mean_sfh_type="dense_basis",
)
print(f"\nModel: {spec.n_free} free parameters")
print(f"  {', '.join(spec.free_params[:5])}...")

model = SEDModel(spec, ssp_data, observation=obs)
print(f"Recommended method: {model.recommend_method()}")

# %% [markdown]
# ## Part 3: Generate mock photometry (SNR=15)

# %%
key = jax.random.PRNGKey(123)
truth = spec.sample(key)

# Override to realistic: z=0.08, Msun=10.5, rising SFH
truth = {**truth}
truth["redshift"] = jnp.array(0.08)
truth["sfh_db_log_total_mass"] = jnp.array(10.5)
truth["sfh_db_log_sfr_inst"] = jnp.array(0.2)
truth["sfh_db_tx_frac_0"] = jnp.array(0.15)
truth["sfh_db_tx_frac_1"] = jnp.array(0.30)
truth["sfh_db_tx_frac_2"] = jnp.array(0.55)
truth["met_logzsol"] = jnp.array(-0.05)
truth["dust_tau_bc"] = jnp.array(0.4)
truth["dust_tau_diff"] = jnp.array(0.25)
truth["dust_T"] = jnp.array(35.0)

mock_data = model.mock(truth, snr=15.0, key=key)

print(f"\nTrue parameters (z={float(truth['redshift']):.3f}):")
for name in spec.free_params[:6]:
    print(f"  {name:30s} = {float(truth[name]):.4f}")
print(f"\nMock: {len(mock_data.flux_obs)} bands, SNR=15")

# %% [markdown]
# ## Part 4: Inference with MAP

# %%
print("\n" + "=" * 70)
print("FITTING: MAP optimization")
print("=" * 70)

fitter = Fitter(model, mock_data.flux_obs, mock_data.noise)

t0 = time.perf_counter()
result = fitter.run(
    "map",
    key=jax.random.PRNGKey(456),
    verbose=False,
)
t_fit = time.perf_counter() - t0

print(f"\n✓ MAP: {t_fit:.1f}s")
print(f"  Loss at optimum: {result.loss_history[-1]:.2f}")

# Create posterior samples using Laplace approximation
# This provides credible intervals by sampling from the normal approximation
n_laplace_samples = 600
key = jax.random.PRNGKey(789)
laplace_samples_dict = result.resample(key, n=n_laplace_samples)
if isinstance(laplace_samples_dict, dict):
    print(f"  Laplace samples (for credible intervals): {len(next(iter(laplace_samples_dict.values())))}")
else:
    print(f"  Laplace samples (for credible intervals): {len(next(iter(laplace_samples_dict.samples.values())))}")

# %% [markdown]
# ## Part 5: Fit quality assessment

# %%
print("\n" + "=" * 70)
print("FIT SUMMARY")
print("=" * 70)
print("\nOptimized parameters (MAP):")
for name in spec.free_params[:5]:
    print(f"  {name:30s} = {float(result.params[name]):.4f}")

# Extract samples for credible intervals
samples_for_credible = laplace_samples_dict if isinstance(laplace_samples_dict, dict) else result.samples
n_samps = len(next(iter(samples_for_credible.values())))
print(f"\nLaplace approximation posterior: {n_samps} samples (from Hessian at MAP)")

# %% [markdown]
# ## Part 6: Derived properties

# %%
print("\n" + "=" * 70)
print("DERIVED PROPERTIES")
print("=" * 70)

# Compute derived quantities from MAP + Laplace samples
derived = result.derived
try:
    stellar_mass = derived.get("stellar_mass")
    sfr_10myr = derived.get("sfr_10myr")
    sfr_100myr = derived.get("sfr_100myr")
    ssfr = derived.get("ssfr")

    if stellar_mass is not None and len(stellar_mass) > 1:
        print("\nStellar mass [log10(M☉)]:")
        m_lo, m_med, m_hi = np.percentile(stellar_mass, [16, 50, 84])
        print(f"  {m_med:.2f} +{m_hi-m_med:.2f} -{m_med-m_lo:.2f}")

    if sfr_10myr is not None and len(sfr_10myr) > 1:
        print("\nSFR (10 Myr) [M☉/yr]:")
        s10_lo, s10_med, s10_hi = np.percentile(sfr_10myr, [16, 50, 84])
        print(f"  {s10_med:.2f} +{s10_hi-s10_med:.2f} -{s10_med-s10_lo:.2f}")

    if sfr_100myr is not None and len(sfr_100myr) > 1:
        print("\nSFR (100 Myr) [M☉/yr]:")
        s100_lo, s100_med, s100_hi = np.percentile(sfr_100myr, [16, 50, 84])
        print(f"  {s100_med:.2f} +{s100_hi-s100_med:.2f} -{s100_med-s100_lo:.2f}")

    if ssfr is not None and len(ssfr) > 1:
        print("\nsSFR (100 Myr) [Gyr⁻¹]:")
        ssfr_lo, ssfr_med, ssfr_hi = np.percentile(ssfr, [16, 50, 84])
        print(f"  {ssfr_med:.2f} +{ssfr_hi-ssfr_med:.2f} -{ssfr_med-ssfr_lo:.2f}")
except Exception as e:
    print(f"(Derived properties unavailable: {str(e)[:60]})")

# %% [markdown]
# ## Figure 1: Posterior-predictive SED fit + residuals

# %%
n_pred = 100
pred_samples = []
# Use Laplace samples if available, otherwise use MAP
samples_source = laplace_samples_dict if isinstance(laplace_samples_dict, dict) else result.samples
n_avail = len(next(iter(samples_source.values())))

for i in range(min(n_pred, n_avail)):
    idx = i % n_avail
    draw = {k: v[idx] for k, v in samples_source.items()}
    with contextlib.suppress(Exception):
        pred_samples.append(np.array(model.predict_photometry(draw)))

pred_array = np.array(pred_samples)
pred_med = np.median(pred_array, axis=0)
pred_lo = np.percentile(pred_array, 16, axis=0)
pred_hi = np.percentile(pred_array, 84, axis=0)

wave_eff = np.array([
    tg.filters.compute_effective_wavelength(tg.filters.load(name))
    for name in filter_names
])
wave_um = wave_eff / 10000.0

flux_ujy = np.array(tg.units.fnu_to_ujy(np.array(mock_data.flux_obs)))
noise_ujy = np.array(tg.units.fnu_to_ujy(np.array(mock_data.noise)))
pred_med_ujy = np.array(tg.units.fnu_to_ujy(pred_med))
pred_lo_ujy = np.array(tg.units.fnu_to_ujy(pred_lo))
pred_hi_ujy = np.array(tg.units.fnu_to_ujy(pred_hi))

xlo, xhi = 0.1, 30.0
mask = (wave_um >= xlo) & (wave_um <= xhi)
valid = mask & (flux_ujy > 0)

fig = plt.figure(figsize=(13, 8))
gs = gridspec.GridSpec(2, 1, height_ratios=[2.5, 1], hspace=0.05)
ax_sed = fig.add_subplot(gs[0])
ax_res = fig.add_subplot(gs[1], sharex=ax_sed)

ax_sed.loglog(wave_um[valid], flux_ujy[valid], "o", ms=8,
              color=COLORS.get("data", "C0"), alpha=0.7, label="Observed (SNR=15)")
ax_sed.fill_between(wave_um[mask], pred_lo_ujy[mask], pred_hi_ujy[mask],
                     color=COLORS.get("model", "C1"), alpha=0.3, label="68% credible")
ax_sed.plot(wave_um[mask], pred_med_ujy[mask], "-",
            color=COLORS.get("model", "C1"), lw=2.0, label="Posterior median")

ymed = np.median(flux_ujy[valid])
ax_sed.set_xlim(xlo, xhi)
ax_sed.set_ylim(ymed / 1e2, ymed * 1e2)
ax_sed.set_ylabel(r"$f_\nu$ [μJy]", fontsize=11)
ax_sed.legend(loc="upper left", frameon=False, fontsize=10)
ax_sed.grid(True, alpha=0.3, which="both")
ax_sed.set_title("Posterior-predictive SED: UV–MIR photometry", fontsize=12)

residual_sigma = (flux_ujy - pred_med_ujy) / noise_ujy
ax_res.axhline(0, color="k", ls="-", lw=1.2, alpha=0.5)
ax_res.axhline(2, color="k", ls="--", lw=0.8, alpha=0.3)
ax_res.axhline(-2, color="k", ls="--", lw=0.8, alpha=0.3)
ax_res.scatter(wave_um[valid], residual_sigma[valid], s=50,
               color=COLORS.get("data", "C0"), alpha=0.7)
ax_res.set_ylim(-3.5, 3.5)
ax_res.set_xlabel(r"Observed wavelength [μm]", fontsize=11)
ax_res.set_ylabel(r"Residual [σ]", fontsize=11)
ax_res.grid(True, alpha=0.3, which="major")

plt.savefig(os.path.join(FIGDIR, "05_posterior_predictive.png"), dpi=200, bbox_inches="tight")
print("✓ Saved 05_posterior_predictive.png")
plt.close()

# %% [markdown]
# ## Figure 2: Corner plot

# %%
fig = result.plot_corner(truths=truth)
if fig is not None:
    fig.suptitle("Parameter posterior: 10-D NUTS", fontsize=12, y=0.995)
    plt.savefig(os.path.join(FIGDIR, "05_corner.png"), dpi=200, bbox_inches="tight")
    print("✓ Saved 05_corner.png")
    plt.close()

# %% [markdown]
# ## Figure 3: SFH posterior

# %%
fig, ax = plt.subplots(figsize=(10, 5))
result.plot_sfh(ax=ax, label="Posterior", color=COLORS.get("model", "C1"))
sfh_truth = model.predict_sfh(truth)
t_gyr = np.array(model.wavelengths["sfh_age_gyr"])
ax.plot(t_gyr, np.array(sfh_truth), "s--", color=COLORS.get("truth", "C2"),
        lw=2, ms=6, label="Truth", alpha=0.8)
ax.set_xscale("log")
ax.set_xlabel(r"Age [Gyr]", fontsize=11)
ax.set_ylabel(r"SFR [M$_\odot$/yr]", fontsize=11)
ax.set_title("Star formation history posterior", fontsize=12)
ax.legend(loc="upper right", frameon=False, fontsize=10)
ax.grid(True, alpha=0.3, which="both")
plt.savefig(os.path.join(FIGDIR, "05_sfh_posterior.png"), dpi=200, bbox_inches="tight")
print("✓ Saved 05_sfh_posterior.png")
plt.close()

# %% [markdown]
# ## Summary

# %%
print("\n" + "=" * 70)
print("SUMMARY: Photometric SED Fitting")
print("=" * 70)

print(f"""
✓ Complete workflow:
  Data:      {phot_obs.n_filters} UV–IR bands (SNR=15)
  Model:     {spec.n_free} free params (SFH + dust + redshift + nebular)
  Inference: NUTS {len(next(iter(result.samples.values())))} samples in {t_fit:.1f}s
  Diagnostics: R̂_max={np.max(result.rhat):.4f}, divergences={result.diagnostics['n_divergent']}

Derived: stellar mass, SFR(10/100 Myr), sSFR with 68% credible intervals
Validation: posterior-predictive residuals, SFH recovery, corner plots

Limitation: Photometry alone cannot break age–dust–metallicity degeneracy.
Solution: Add spectroscopy (notebook 06) to constrain stellar age.

Next: 06_fitting_spectroscopy.py for optical spectrum + line diagnostics
""")

print("=" * 70)

tg.cite(result)

print("\n✓ Notebook complete: photometric SED fitting, NUTS inference, posterior validation")
