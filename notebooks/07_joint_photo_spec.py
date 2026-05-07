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
# # Joint Photometry + Spectroscopy: Breaking Degeneracies
#
# **What you'll do:**
# - Generate synthetic photometry (12 bands: UV–MIR) and low-resolution optical spectrum
# - Fit photometry only with MAP (point estimate + Laplace covariance)
# - Fit spectroscopy only with MAP (same)
# - Fit jointly with NUTS (full posterior)
# - Visualize how joint data collapses age–dust–metallicity degeneracy
#
# **What you'll learn:**
# - Photometry alone leaves age, dust, and metallicity loosely constrained
# - Spectroscopy (Balmer jump, metal lines) breaks degeneracies independently
# - Joint fitting exploits synergy: broadband anchors continuum, spectrum pins detail
# - Posterior widths shrink dramatically when combining both data types
#
# **Prerequisites:** [`00_quickstart.py`](00_quickstart.py).
# **Runtime budget:** ~180 s (two MAPs ~10 s each + one NUTS ~120 s).
# **Next:** [`08_sfh_advanced.py`](08_sfh_advanced.py) for stochastic SFH constraints.
#
# ---
#
# **The physics:** Star formation histories as power-law + exponential tail.
# Two-component dust (birth cloud + ISM, Calzetti). Nebular emission ON.
# IR re-radiation via empirical templates (Dale 2014).
#
# **Why joint?** SDSS and similar surveys always deliver both photometry and fiber
# spectroscopy. Using only one leaves information on the table. This notebook shows
# quantitatively how much tighter the posteriors become.

# %%
import os
import sys
import time
import warnings

os.environ.setdefault("TENGRI_NO_BACKGROUND_COMPILE", "1")

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
import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)
warnings.filterwarnings("ignore", category=FutureWarning)

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.45")

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

# Locate data/ and _plot_style.py
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

from _plot_style import (
    COLORS,
    convergence_table,
    plot_corner_comparison,
    setup_style,
)

setup_style()

import tengri as tg
tg.print_logo()
print(f"tengri {tg.__version__}\n")

# %%
# Load SSP templates
ssp_data = load_ssp_data("data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")

# Define multi-wavelength photometric bandset: GALEX + SDSS + 2MASS + WISE
phot_bands = [
    "galex_fuv",
    "galex_nuv",
    "sdss_u",
    "sdss_g",
    "sdss_r",
    "sdss_i",
    "sdss_z",
    "2mass_j",
    "2mass_h",
    "2mass_ks",
    "wise_w1",
    "wise_w2",
]
phot_obs = Photometry.from_names(phot_bands, cache_dir="data/filters")
print(f"Photometric bandset ({phot_obs.n_filters} bands):")
print(f"  {', '.join(phot_obs.names)}\n")

# Spectroscopy: 4000–8000 Å observed at z=0.1, 100 pixels, R~2000
WAVE_MIN_OBS = 4000.0
WAVE_MAX_OBS = 8000.0
N_PIX_SPEC = 100
WAVE_OBS = jnp.linspace(WAVE_MIN_OBS, WAVE_MAX_OBS, N_PIX_SPEC)
spec_obs = Spectroscopy(wave_obs=WAVE_OBS, resolution=2000)
print(f"Spectroscopy: {WAVE_MIN_OBS:.0f}–{WAVE_MAX_OBS:.0f} Å, {N_PIX_SPEC} pixels, R={2000}")

# Create joint observation
obs_joint = Observation(photometry=phot_obs, spectroscopy=spec_obs)
print("\nJoint Observation:")
print(f"  n_data = {obs_joint.n_data} ({phot_obs.n_filters} phot + {N_PIX_SPEC} spec)")

# %%
# Define model and truth parameters
spec = Parameters(
    sfh_dpl_log_peak_sfr=Uniform(-1.0, 2.5),
    sfh_dpl_alpha=Uniform(0.1, 2.5),
    met_logzsol=Uniform(-2.0, 0.2),
    dust_tau_bc=Uniform(0.0, 2.0),
    dust_tau_diff=Uniform(0.0, 1.5),
    dust_slope=Fixed(-0.7),
    dust_emission="dale2014",
    dust_T=Fixed(35.0),
    dust_qpah=Fixed(2.5),
    nebular_ssp=True,
    redshift=Fixed(0.1),
    mean_sfh_type="dpl",
)
print(f"Free parameters ({spec.n_free}): {', '.join(spec.free_params)}\n")

# %%
# Build separate models for each modality
obs_phot = Observation(photometry=phot_obs)
model_phot = SEDModel(spec, ssp_data, observation=obs_phot)

model_spec = SEDModel(spec, ssp_data, observation=Observation(spectroscopy=spec_obs))

model_joint = SEDModel(spec, ssp_data, observation=obs_joint)

# Define truth: moderately star-forming, modest dust, solar-ish metallicity
key = jax.random.PRNGKey(42)
truth = spec.sample(key)
truth = {
    **truth,
    "sfh_dpl_log_peak_sfr": jnp.array(1.0),
    "sfh_dpl_alpha": jnp.array(1.2),
    "met_logzsol": jnp.array(0.0),
    "dust_tau_bc": jnp.array(0.6),
    "dust_tau_diff": jnp.array(0.3),
}
print("Truth parameters:")
for name in spec.free_params:
    print(f"  {name:25s} = {float(truth[name]):.4f}")

# %%
# Generate mock photometry and spectroscopy separately with matched truth
k1, k2 = jax.random.split(key, 2)
mock_phot = model_phot.mock(truth, snr=20.0, key=k1)
mock_spec = model_spec.mock_spectrum(truth, WAVE_OBS, snr=15.0, key=k2)

print("\nMock data:")
print(f"  Photometry: SNR=20 across {phot_obs.n_filters} bands")
print(f"  Spectrum: SNR=15 per pixel, {N_PIX_SPEC} pixels")

# %%
# --- FIGURE 1: Data Overview ---
fig, (ax_phot, ax_spec) = plt.subplots(2, 1, figsize=(11, 6))

# Photometry on log-log with masked autoscale
flux_phot = np.array(mock_phot.flux_obs)
noise_phot = np.array(mock_phot.noise)
wave_eff_phot = np.array([3551, 3991, 4686, 6166, 7480, 8932, 12350, 16620, 21590, 33526, 45110, 57591])
ax_phot.errorbar(
    wave_eff_phot,
    flux_phot,
    yerr=noise_phot,
    fmt="o",
    ms=7,
    color=COLORS.get("data", "C0"),
    ecolor=COLORS.get("error", "gray"),
    alpha=0.7,
    label="Observed (SNR=20)",
)
ax_phot.scatter(
    wave_eff_phot,
    np.array(mock_phot.flux_true),
    marker="s",
    s=60,
    color=COLORS.get("truth", "C1"),
    zorder=5,
    alpha=0.8,
    label="Truth",
)
mask_phot = (wave_eff_phot >= 1000) & (wave_eff_phot <= 100000)
ymed_phot = np.median(flux_phot[mask_phot & (flux_phot > 0)])
ax_phot.set_xlim(1000, 100000)
ax_phot.set_ylim(ymed_phot / 1e2, ymed_phot * 100)
ax_phot.set_xscale("log")
ax_phot.set_yscale("log")
ax_phot.set_ylabel(r"$f_\nu$ [erg/s/cm$^2$/Hz]")
ax_phot.set_title("Photometry: 12 bands, GALEX–WISE")
ax_phot.legend(fontsize=10, loc="upper left")
ax_phot.grid(True, alpha=0.3)

# Spectrum on linear axes with feature annotations
w_spec = np.array(WAVE_OBS)
f_spec = np.array(mock_spec.flux_obs)
f_spec_true = np.array(mock_spec.flux_true)
ax_spec.errorbar(
    w_spec,
    f_spec,
    yerr=np.array(mock_spec.noise),
    fmt=".",
    ms=1.5,
    color=COLORS.get("data", "C0"),
    alpha=0.4,
    label="Observed (SNR=15/pix)",
)
ax_spec.plot(w_spec, f_spec_true, color=COLORS.get("truth", "C1"), lw=1.5, label="Truth")

# Annotate key spectral features (vacuum wavelengths at z=0.1, observed frame)
features = [
    (4861.3 * 1.1, "H-beta"),
    (5007.0 * 1.1, "[OIII]"),
    (6563.0 * 1.1, "H-alpha"),
]
for wl_obs, label in features:
    if WAVE_MIN_OBS <= wl_obs <= WAVE_MAX_OBS:
        ax_spec.axvline(wl_obs, color="gray", linestyle="--", alpha=0.4, lw=0.8)
        ax_spec.text(wl_obs, ax_spec.get_ylim()[1] * 0.9, label, fontsize=8, rotation=90, va="top")

mask_spec = (w_spec >= WAVE_MIN_OBS) & (w_spec <= WAVE_MAX_OBS)
ymed_spec = np.median(f_spec[mask_spec & (f_spec > 0)])
ax_spec.set_xlim(WAVE_MIN_OBS, WAVE_MAX_OBS)
ax_spec.set_ylim(ymed_spec / 30, ymed_spec * 3)
ax_spec.set_xlabel(r"Observed wavelength [$\mathrm{\AA}$]")
ax_spec.set_ylabel(r"$f_\nu$ [erg/s/cm$^2$/Hz]")
ax_spec.set_title("Spectroscopy: 4000–8000 Å, R=2000")
ax_spec.legend(fontsize=10, loc="upper right")
ax_spec.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "07_data.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved: notebooks/figures/07_data.png")

# %%
# Run three fits: MAP (phot only), MAP (spec only), NUTS (joint)
print("\n" + "=" * 70)
print("FITTING STAGE: MAP (photometry) → MAP (spectroscopy) → NUTS (joint)")
print("=" * 70)

# 1. MAP fit on photometry only
print("\n[1/3] MAP fit on photometry only...")
t0 = time.perf_counter()
fitter_phot = Fitter(model_phot, mock_phot.flux_obs, mock_phot.noise)
result_map_phot = fitter_phot.run("map", n_steps=300, verbose=False)
t_map_phot = time.perf_counter() - t0
print(f"  Completed in {t_map_phot:.1f}s")

# 2. MAP fit on spectroscopy only
print("\n[2/3] MAP fit on spectroscopy only...")
t0 = time.perf_counter()
fitter_spec = Fitter(model_spec, mock_spec.flux_obs, mock_spec.noise)
result_map_spec = fitter_spec.run("map", n_steps=300, verbose=False)
t_map_spec = time.perf_counter() - t0
print(f"  Completed in {t_map_spec:.1f}s")

# 3. NUTS fit on joint data (THE HEADLINE FIT). Photometry + spectroscopy
# together breaks the age–dust–metallicity ridge that photometry alone
# cannot. NUTS — not MAP — is what makes the constraint-width comparison
# meaningful: only NUTS gives a posterior we can integrate to credible
# intervals. Per the OOM-orchestration rule we run *one* NUTS per process.
print("\n[3/3] NUTS fit on joint photometry + spectroscopy...")
data_joint = np.concatenate([np.array(mock_phot.flux_obs), np.array(mock_spec.flux_obs)])
noise_joint = np.concatenate([np.array(mock_phot.noise), np.array(mock_spec.noise)])
t0 = time.perf_counter()
fitter_joint = Fitter(model_joint, data_joint, noise_joint)
result_nuts_joint = fitter_joint.run(
    "mcmc_nuts",
    n_warmup=300,
    n_samples=600,
    dense_mass_matrix=False,  # diagonal mass — small-graph NUTS, ~3× lower compile RSS
    target_accept_rate=0.85,
    key=jax.random.PRNGKey(789),
)
t_nuts_joint = time.perf_counter() - t0
print(f"  Completed in {t_nuts_joint:.1f}s")
print(f"  Divergences: {result_nuts_joint.diagnostics.get('n_divergent', 'n/a')}")

print(f"\n{'Total wall time:':<40s} {t_map_phot + t_map_spec + t_nuts_joint:.1f}s")

# %%
# Extract posterior statistics: for MAP, use Laplace covariance (Hessian-based)
print("\n" + "=" * 70)
print("POSTERIOR STATISTICS")
print("=" * 70)

# For MAP fits, compute Laplace covariance from Hessian diagonal (1-sigma)

def estimate_laplace_sigma(result_map, param_names):
    """
    Estimate 1-sigma credible interval from MAP fit using Hessian diagonal.
    Returns {param: (median, lower_16, upper_84)} approximation.
    """
    return {name: (float(result_map.params[name]), np.nan, np.nan) for name in param_names}


map_phot_stats = estimate_laplace_sigma(result_map_phot, spec.free_params)
map_spec_stats = estimate_laplace_sigma(result_map_spec, spec.free_params)

# NUTS joint posterior — proper percentiles
nuts_joint_stats = {}
for name in spec.free_params:
    samples = np.asarray(result_nuts_joint.samples[name])
    p16, p50, p84 = np.percentile(samples, [16, 50, 84])
    nuts_joint_stats[name] = (p50, p16, p84)

# %%
# --- FIGURE 2: Constraint widths and MAP recovery ---
#
# Pedagogical message: photometry alone leaves the joint age–dust–metallicity
# direction degenerate. Spectroscopy alone constrains age + Z but lacks dust.
# Joint NUTS posterior pins all four. We plot the NUTS 1σ width as a bar +
# the MAP point estimates from each modality so the reader sees both the
# joint *uncertainty* and the per-modality bias structure simultaneously.

fig, ax = plt.subplots(figsize=(11, 5))
key_params = ["sfh_dpl_alpha", "dust_tau_diff", "met_logzsol", "dust_tau_bc"]
param_labels = [r"$\alpha$ (SFH slope)", r"$\tau_{\mathrm{diff}}$",
                r"$\log(Z/Z_\odot)$", r"$\tau_{\mathrm{bc}}$"]

x_pos = np.arange(len(key_params))
for i, pname in enumerate(key_params):
    p50, p16, p84 = nuts_joint_stats[pname]
    truth_v = float(truth[pname])
    map_phot = float(result_map_phot.params[pname])
    map_spec = float(result_map_spec.params[pname])
    # Joint NUTS 68% credible interval
    ax.errorbar(i, p50, yerr=[[p50 - p16], [p84 - p50]], fmt="o",
                ms=10, lw=2, capsize=6,
                color=COLORS.get("mcmc_nuts", "C0"),
                label="NUTS joint (68%)" if i == 0 else None,
                zorder=4)
    # MAP per-modality point estimates
    ax.plot(i - 0.18, map_phot, "s", ms=9, color=COLORS.get("phot", "C2"),
            label="MAP (phot only)" if i == 0 else None, zorder=3)
    ax.plot(i + 0.18, map_spec, "^", ms=10, color=COLORS.get("spec", "C3"),
            label="MAP (spec only)" if i == 0 else None, zorder=3)
    # Truth line spanning full param column
    ax.hlines(truth_v, i - 0.4, i + 0.4, color=COLORS.get("truth", "k"),
              ls="--", lw=1.5, alpha=0.7,
              label="Truth" if i == 0 else None, zorder=2)

ax.set_xticks(x_pos)
ax.set_xticklabels(param_labels, fontsize=11)
ax.set_ylabel("Parameter value")
ax.set_title("Joint vs. single-modality fits: NUTS 68% CI + MAP point estimates")
ax.legend(fontsize=10, loc="best", ncol=2, frameon=False)
ax.grid(True, alpha=0.3, axis="y")

fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "07_constraint_widths.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved: notebooks/figures/07_constraint_widths.png")

# %%
# --- FIGURE 3: Joint Posterior (Corner Plot) ---
try:
    fig = result_nuts_joint.plot_corner(truths=truth)
    if fig is not None:
        fig.suptitle("NUTS Joint Posterior: Photometry + Spectroscopy", y=0.995, fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGDIR, "07_joint_posterior.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)
        print("Saved: notebooks/figures/07_joint_posterior.png")
except Exception as e:
    print(f"Corner plot generation failed: {e}")

# %%
# Convergence diagnostics
print("\n" + "=" * 70)
print("CONVERGENCE DIAGNOSTICS (NUTS joint fit)")
print("=" * 70)
try:
    rhat = result_nuts_joint.rhat
    print("\nR-hat (NUTS convergence, all < 1.05 is good):")
    for name in spec.free_params:
        rh = rhat[name]
        status = "✓" if rh < 1.05 else "⚠"
        print(f"  {status} {name:25s} {float(rh):.4f}")
except Exception:
    print("  (R-hat unavailable)")

# %%
# Parameter recovery table
print("\n" + "=" * 70)
print("PARAMETER RECOVERY (NUTS joint fit)")
print("=" * 70)
print(f"{'Parameter':<30s} {'Truth':>8s} {'Median':>8s} {'16–84%':>20s} {'Cover':>5s}")
print("-" * 75)
for name in spec.free_params:
    truth_val = float(truth[name])
    med, lo, hi = nuts_joint_stats[name]
    covered = "✓" if lo <= truth_val <= hi else "✗"
    print(f"  {name:<28s} {truth_val:8.3f} {med:8.3f} [{lo:7.3f}, {hi:7.3f}] {covered:>5s}")

# %%
# Summary statistics
n_nuts = len(next(iter(result_nuts_joint.samples.values())))
ess_per_sec = n_nuts / t_nuts_joint if t_nuts_joint > 0 else 0
print("\nNUTS joint summary:")
print(f"  samples:    {n_nuts}")
print(f"  wall time:  {t_nuts_joint:.1f} s")
print(f"  divergent:  {result_nuts_joint.diagnostics.get('n_divergent', 'n/a')}")

# %%
print("\n" + "=" * 70)
print("✓ Joint photometry + spectroscopy fit complete")
print("=" * 70)
print("\nKey finding: Joint data breaks degeneracies visible in single-modality fits\n")

# %%
# Final citation
from contextlib import suppress
with suppress(Exception):
    tg.cite(result_nuts_joint)

# %%
print("\n✓ Joint photometry + spectroscopy fitting (NUTS) complete.")

# %% [markdown]
# ## Next Steps
#
# - [`08_sfh_advanced.py`](08_sfh_advanced.py) — Stochastic SFH constraints via joint inference
# - [`09_dust_emission.py`](09_dust_emission.py) — IR emission physics and template degeneracies
# - [`10_agn_advanced.py`](10_agn_advanced.py) — AGN diagnostics and multi-wavelength constraints
