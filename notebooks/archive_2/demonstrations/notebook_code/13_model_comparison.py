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
# # SEDModel Comparison with Nested Slice Sampling
#
# tengri's other inference methods (geoVI, Ray Tracing, NUTS) target the
# **posterior** — they tell you *what* the parameters are given a model. But
# they cannot tell you *which model is better*. For that you need the
# **Bayesian evidence** (marginal likelihood):
#
# $$\mathcal{Z} = \int \mathcal{L}(\boldsymbol{\theta})\,\pi(\boldsymbol{\theta})\,d\boldsymbol{\theta}$$
#
# This notebook introduces **Nested Slice Sampling** (NSS; Yallup, Kroupa &
# Handley 2026), tengri's evidence computation method. It uses Hit-and-Run
# Slice Sampling (HRSS) as the inner kernel to explore the constrained prior,
# accumulating the evidence integral as it compresses the prior volume.
#
# **Scope:** NSS is restricted to smooth (non-stochastic) parametric models
# where D ≲ 30. For the high-dimensional stochastic GP model (D ~ 137),
# use geoVI or Ray Tracing.

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
    Fitter,
    Fixed,
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
if os.path.exists("data"):
    pass  # already in project root
elif os.path.exists(os.path.join("..", "data")):
    os.chdir("..")
elif os.path.exists(os.path.join("..", "..", "data")):
    os.chdir(os.path.join("..", ".."))
elif os.path.exists(os.path.join("..", "..", "..", "data")):
    os.chdir(os.path.join("..", "..", ".."))

FIGDIR = os.path.join("demonstrations", "figures")
os.makedirs(FIGDIR, exist_ok=True)

from _plot_style import (
    COLORS,
    convergence_table,
    plot_corner_comparison,
    plot_sfh,
    setup_style,
)

setup_style()

# %%
ssp_data = load_ssp_data("data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")
obs = Observation(
    photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
)

# %% [markdown]
# ## Two Competing Models
#
# We set up two parametric SFH models with different complexity:
#
# | SEDModel | SFH type | Free params | Physical motivation |
# |-------|----------|-------------|-------------------|
# | **A** (simple) | Double power law | 5 | Smooth rise-and-fall SFH |
# | **B** (complex) | Truncated skew-normal | 7 | Asymmetric SFH with truncation |
#
# We generate mock data from SEDModel A, then ask: **does the extra complexity
# of SEDModel B help, or does the simpler model suffice?** The Bayes factor
# answers this question.

# %%
# --- SEDModel A: Double power law (simple, 5 free params) ---
spec_A = Parameters(
    sfh_dpl_alpha=Uniform(0.5, 3.0),
    sfh_dpl_beta=Uniform(0.5, 3.0),
    sfh_dpl_tau_gyr=Uniform(0.5, 10.0),
    met_logzsol=Uniform(-2.0, 0.2),
    dust_tau_bc=Uniform(0.0, 2.0),
    dust_tau_diff=Fixed(0.5),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.1),
    stochastic=False,
)
model_A = SEDModel(spec_A, ssp_data, observation=obs)

# --- SEDModel B: Truncated skew-normal (complex, 7 free params) ---
spec_B = Parameters(
    sfh_tsnorm_log_peak_sfr=Uniform(-1.0, 2.5),
    sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),
    sfh_tsnorm_width_gyr=Uniform(0.3, 5.0),
    sfh_tsnorm_skew=Uniform(-3.0, 3.0),
    sfh_tsnorm_trunc=Uniform(1.0, 10.0),
    met_logzsol=Uniform(-2.0, 0.2),
    dust_tau_bc=Uniform(0.0, 2.0),
    dust_tau_diff=Fixed(0.5),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.1),
    mean_sfh_type="tsnorm",
    stochastic=False,
)
model_B = SEDModel(spec_B, ssp_data, observation=obs)

print(f"SEDModel A (double power law):      D = {spec_A.n_free}")
print(f"SEDModel B (truncated skew-normal): D = {spec_B.n_free}")

# %%
# Generate mock data from SEDModel A (the truth)
key = jax.random.PRNGKey(42)
true_params_A = spec_A.sample(key)

# Override to a typical star-forming galaxy
true_params_A = {**true_params_A}
true_params_A["sfh_dpl_alpha"] = jnp.array(1.5)
true_params_A["sfh_dpl_beta"] = jnp.array(1.0)
true_params_A["sfh_dpl_tau_gyr"] = jnp.array(5.0)
true_params_A["met_logzsol"] = jnp.array(-0.5)
true_params_A["dust_tau_bc"] = jnp.array(0.3)

mock = model_A.mock(true_params_A, snr=20.0, key=jax.random.PRNGKey(1))
print(f"Mock photometry: {len(mock.flux_obs)} bands, SNR ≈ 20")

# %% [markdown]
# ## Running NSS
#
# NSS is invoked via `fitter.run("evidence")`. Key parameters:
#
# | Parameter | Default | Description |
# |-----------|---------|-------------|
# | `n_live` | 500 | Number of live points — more = more precise evidence |
# | `num_delete` | 50 | Points replaced per iteration — controls speed vs accuracy |
# | `num_inner_steps` | D | HRSS walk length — longer = less correlated samples |
# | `log_evidence_tol` | -3.0 | Stop when remaining evidence < e⁻³ of total |
#
# The first call includes JIT compilation overhead (~10–30s). Subsequent
# calls with the same model structure are much faster.

# %%
# Fit both models to the same data
fitter_A = Fitter(model_A, mock.flux_obs, mock.noise)
fitter_B = Fitter(model_B, mock.flux_obs, mock.noise)

print("Running NSS on SEDModel A (double power law)...")
result_A = fitter_A.run(
    "evidence",
    n_live=500,
    num_delete=50,
    key=jax.random.PRNGKey(0),
)

# %%
print("\nRunning NSS on SEDModel B (truncated skew-normal)...")
result_B = fitter_B.run(
    "evidence",
    n_live=500,
    num_delete=50,
    key=jax.random.PRNGKey(0),
)

# %% [markdown]
# ## The Bayes Factor
#
# The Bayes factor compares two models:
#
# $$K = \frac{\mathcal{Z}_A}{\mathcal{Z}_B} = \exp(\log\mathcal{Z}_A - \log\mathcal{Z}_B)$$
#
# On the Jeffreys scale:
#
# | $|\ln K|$ | Evidence |
# |-----------|----------|
# | < 1 | Inconclusive |
# | 1–2.5 | Moderate |
# | 2.5–5 | Strong |
# | > 5 | Decisive |

# %%
logZ_A = result_A.log_evidence
logZ_B = result_B.log_evidence
ln_K = logZ_A - logZ_B

print(f"  SEDModel A (DPL, D={spec_A.n_free}):    log Z = {logZ_A:.2f}")
print(f"  SEDModel B (tsnorm, D={spec_B.n_free}):  log Z = {logZ_B:.2f}")
print(f"  ln K (A vs B) = {ln_K:.2f}")
print()

if abs(ln_K) < 1:
    verdict = "Inconclusive — cannot distinguish"
elif abs(ln_K) < 2.5:
    preferred = "A" if ln_K > 0 else "B"
    verdict = f"Moderate preference for SEDModel {preferred}"
elif abs(ln_K) < 5:
    preferred = "A" if ln_K > 0 else "B"
    verdict = f"Strong preference for SEDModel {preferred}"
else:
    preferred = "A" if ln_K > 0 else "B"
    verdict = f"Decisive preference for SEDModel {preferred}"

print(f"  Verdict: {verdict}")
print()
print("  (Data was generated from SEDModel A — the Bayes factor should")
print("   prefer A, penalizing B's extra parameters via Occam's razor.)")

# %%
# --- FIGURE 1: Evidence comparison ---
fig, ax = plt.subplots(figsize=(7, 4))

bar_colors = [COLORS["rt"], COLORS["geovi"]]
labels = [f"A: DPL (D={spec_A.n_free})", f"B: tsnorm (D={spec_B.n_free})"]
logZs = [logZ_A, logZ_B]
bars = ax.barh([0, 1], logZs, color=bar_colors, alpha=0.85, height=0.5)

for bar, lz in zip(bars, logZs):
    x_pos = lz - 0.3 if lz < 0 else lz + 0.1
    ax.text(
        x_pos,
        bar.get_y() + bar.get_height() / 2,
        f"{lz:.1f}",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="white" if lz < 0 else "k",
    )

ax.set_yticks([0, 1])
ax.set_yticklabels(labels, fontsize=11)
ax.set_xlabel("log Z (evidence)", fontsize=12)
ax.set_title(f"Bayes factor: ln K = {ln_K:.1f}  →  {verdict}", fontsize=11)
ax.axvline(0, color="k", lw=0.5, ls=":")
fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "fig01_evidence_comparison.png"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Posterior Samples from NSS
#
# NSS produces posterior samples as a byproduct — they come from importance
# resampling of the dead and live points weighted by prior volume.

# %%
print("SEDModel A posterior:")
print(result_A.summary_table())
print()
print("SEDModel B posterior:")
print(result_B.summary_table())

# %%
# --- FIGURE 2: Corner plot for SEDModel A ---
fig_corner = result_A.plot_corner(truths=true_params_A, color=COLORS["rt"])
if fig_corner is not None:
    fig_corner.suptitle("SEDModel A (DPL): NSS Posterior", y=1.02, fontsize=12)
    plt.savefig(os.path.join(FIGDIR, "fig02_corner_model_A.png"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## NSS vs geoVI: Same Posteriors, Only NSS Gives Evidence
#
# For the same model, NSS and geoVI should recover similar posteriors. The
# key difference: NSS also computes log Z, enabling model comparison.

# %%
# Run geoVI on SEDModel A for comparison
print("Running geoVI on SEDModel A for posterior comparison...")
result_map_A = fitter_A.run(
    "map",
    n_steps=500,
    verbose=False,
)

t0_compile_geovi = time.perf_counter()
fitter_A.compile(verbose=False)
t_compile_geovi = time.perf_counter() - t0_compile_geovi

t0_run_geovi = time.perf_counter()
result_geovi_A = fitter_A.run(
    "vi",
    n_iterations=15,
    n_samples=6,
    n_seeds=3,
    n_posterior_samples=2000,
    init_from=result_map_A,
    verbose=False,
)
t_run_geovi = time.perf_counter() - t0_run_geovi

print(f"  geoVI compile: {t_compile_geovi:.1f}s | runtime: {t_run_geovi:.1f}s")
print(f"  NSS wall time: {result_A.wall_time_s:.1f}s")

# %%
# --- FIGURE 3: NSS vs geoVI corner comparison ---
fig = plot_corner_comparison(
    [result_A, result_geovi_A],
    labels=["NSS", "vi"],
    colors=[COLORS["rt"], COLORS["geovi"]],
    truths=true_params_A,
)
if fig is not None:
    fig.suptitle(
        f"SEDModel A: NSS (log Z = {logZ_A:.1f}) vs geoVI (no evidence)",
        y=1.02,
        fontsize=11,
    )
    plt.savefig(os.path.join(FIGDIR, "fig03_nss_vs_geovi.png"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Diagnostics
#
# NSS provides several diagnostics:
#
# - **ESS** (effective sample size): how many independent posterior samples
# - **n_iterations**: number of NS contraction steps
# - **n_dead**: total dead points (evidence integral resolution)

# %%
for name, result in [("SEDModel A", result_A), ("SEDModel B", result_B)]:
    d = result.diagnostics
    print(f"  {name}:")
    print(f"    log Z       = {d['log_evidence']:.2f}")
    print(f"    ESS         = {d['ess']:.0f}")
    print(f"    iterations  = {d['n_iterations']}")
    print(f"    dead points = {d['n_dead']}")
    print(f"    wall time   = {result.wall_time_s:.1f}s")
    print()

# %% [markdown]
# ## SEDModel A: Posterior Validation

# %%
# --- FIGURE 4: Posterior predictive photometry for SEDModel A ---
phot_config = obs.photometry
wave_eff = np.array([float(jnp.mean(fc.wave)) for fc in phot_config.filters])

n_draws = 50
posterior_phot_A = []
for i in range(n_draws):
    idx = i % len(next(iter(result_A.samples.values())))
    s_i = {k: v[idx] for k, v in result_A.samples.items()}
    posterior_phot_A.append(np.array(model_A.predict_photometry(s_i)))
posterior_phot_A = np.array(posterior_phot_A)

phot_true_A = np.array(model_A.predict_photometry(true_params_A))

fig, (ax_top, ax_bot) = plt.subplots(
    2,
    1,
    figsize=(8, 5),
    height_ratios=[3, 1],
    sharex=True,
)
fig.subplots_adjust(hspace=0.05)

for draw in posterior_phot_A:
    ax_top.plot(wave_eff, draw, "-", color=COLORS["rt"], alpha=0.08, lw=0.8)
median_pred_A = np.median(posterior_phot_A, axis=0)
ax_top.plot(wave_eff, median_pred_A, "s", ms=5, color=COLORS["rt"], zorder=4, label="NSS (median)")
ax_top.errorbar(
    wave_eff,
    np.array(mock.flux_obs),
    yerr=np.array(mock.noise),
    fmt="o",
    ms=7,
    color=COLORS["data"],
    capsize=3,
    zorder=5,
    label="Observed",
)
ax_top.scatter(
    wave_eff,
    phot_true_A,
    marker="D",
    s=40,
    facecolors="none",
    edgecolors=COLORS["truth"],
    linewidths=1.2,
    zorder=6,
    label="Truth",
)
ax_top.set_ylabel(r"$f_\nu$")
ax_top.legend(loc="upper right", fontsize=9)
ax_top.set_title("SEDModel A (DPL): NSS Posterior Predictive Photometry")

residuals_A = (np.array(mock.flux_obs) - median_pred_A) / np.array(mock.noise)
ax_bot.axhline(0, color="0.5", ls="--", lw=0.8)
ax_bot.axhspan(-1, 1, alpha=0.05, color="0.5")
band_colors = [COLORS.get(b, COLORS["data"]) for b in ["u", "g", "r", "i", "z"]]
ax_bot.bar(wave_eff, residuals_A, width=200, color=band_colors, alpha=0.7)
ax_bot.set_xlabel(r"Wavelength ($\AA$)")
ax_bot.set_ylabel(r"$(d - f)/\sigma$")
ax_bot.set_ylim(-4, 4)
plt.setp(ax_top.get_xticklabels(), visible=False)

plt.savefig(
    os.path.join(FIGDIR, "fig04_posterior_predictive_A.png"),
    dpi=150,
    bbox_inches="tight",
)
plt.show()

# %%
# --- FIGURE 5: SFH recovery for SEDModel A ---
fig, ax = plt.subplots(figsize=(8, 4))
plot_sfh(
    model_A,
    result_A,
    true_params=true_params_A,
    ax=ax,
    color=COLORS["rt"],
    label="NSS",
    method="RT",
)
ax.set_title("SEDModel A (DPL): SFH Recovery from NSS")
fig.tight_layout()
plt.savefig(
    os.path.join(FIGDIR, "fig05_sfh_recovery_A.png"),
    dpi=150,
    bbox_inches="tight",
)
plt.show()

# %%
# --- Parameter recovery table for SEDModel A ---
print(f"\n{'Parameter':<25s} {'Truth':>10s} {'Median':>10s} {'68% CI':>20s}")
print("-" * 67)
for pname in spec_A.free_params:
    if pname in result_A.samples and pname in true_params_A:
        chain = np.array(result_A.samples[pname])
        truth_val = float(true_params_A[pname])
        med = float(np.median(chain))
        lo, hi = float(np.percentile(chain, 16)), float(np.percentile(chain, 84))
        print(f"{pname:<25s} {truth_val:10.3f} {med:10.3f} [{lo:8.3f}, {hi:8.3f}]")

# %% [markdown]
# ## When to Use NSS
#
# | Use case | Method |
# |----------|--------|
# | **SEDModel comparison** (which SFH, which dust law?) | NSS |
# | **Parameter estimation** (smooth, D ≤ 30) | geoVI or NUTS (faster) |
# | **Parameter estimation** (stochastic, D ~ 137) | geoVI or Ray Tracing |
# | **Quick evidence from MAP** (Gaussian approx.) | Laplace |
#
# NSS is **not** a replacement for geoVI or Ray Tracing for parameter
# estimation — it's slower for that purpose. Its unique value is the
# evidence integral for principled model selection.
#
# ### Computational Notes
#
# - First call includes JIT compilation (~10–30s). Subsequent calls are faster.
# - `n_live=500` with `num_delete=50` is a good default.
# - For higher precision on log Z, increase `n_live` (evidence error ∝ 1/√n_live).
# - NSS uses covariance-adaptive directions (empirical cov of live points),
#   so it handles correlations automatically.
# - Exact implementation of the Yallup, Kroupa & Handley (2026)
#   algorithm from the handley-lab/blackjax fork, running entirely locally
#   with no extra dependencies.
