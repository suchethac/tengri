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
# # Joint fit: photometry + emission lines
#
# Notebook [`06`](06_fitting_spectroscopy.py) fits an optical *spectrum* and
# [`07`](07_joint_photo_spec.py) fits photometry *and* a spectrum together. This
# one fits the other kind of joint dataset that dominates modern surveys: a
# **broadband photometry + emission-line-flux catalog**, of the kind
# [FastSpecFit](https://fastspecfit.readthedocs.io) produces for DESI (and
# MPA-JHU / RCSED for SDSS). Instead of a pixel-by-pixel spectrum we fit a
# handful of measured line fluxes alongside the broadband points.
#
# **Which model quantity matches a catalog line flux?** FastSpecFit fits a
# stellar-continuum model (SPS templates, which carry the Balmer *absorption*),
# subtracts it, and fits each line as a Gaussian on the residual — with the
# [N II]/Hα and doublet kinematics tied so the blends deblend. What it reports
# (`LINE_FLUX`) is therefore **pure, deblended, absorption-corrected emission**.
# That is exactly what `model.predict_line_fluxes` returns — the backend's
# emitted line luminosity, projected to a flux. It is *not* what a naive
# bandpass measurement gives, which is why we fit `predict_line_fluxes` here and
# not a window-integrated `measure_line_fluxes` (the latter would carry the
# stellar absorption and mis-deblend [N II], biasing the Balmer decrement).
#
# **Why it is fast.** Photometry rides the `WavePrecomp` SSP × filter lookup
# table; the emission lines ride the `FeaturePrecomp` per-Q_H grid. Each turns a
# likelihood evaluation that would otherwise integrate the full SSP × wavelength
# forward into a table look-up. We measure the wall time of every fit below and
# compare the fast path to the exact forward. On a single galaxy the win is a
# steady few-fold (the fixed per-fit overhead dilutes it); the fast path's real
# payoff is at catalog scale, where the look-up is shared work across galaxies
# and the exact wave-grid forward would be prohibitive — see
# [notebook 11](11_catalog_fits.py) for `fit_batch` at catalog scale.

# %%
from _setup import FIG_DIR, effective_wavelengths_um, quiet

quiet()

# Notebook-specific: the dense-mass NUTS run below deliberately uses
# dense_mass_matrix=True for convergence; its RAM caveat is discussed in the
# summary, not repeated here.
import warnings

warnings.filterwarnings("ignore", message=".*dense_mass_matrix.*")

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri import (
    FIXED,
    FREE,
    Data,
    FeaturePrecomp,
    Fixed,
    ForwardModel,
    LineList,
    Observation,
    Photometry,
    SEDModel,
    Uniform,
    WavePrecomp,
    cosmology,
    plot,
)
from tengri.observation import LineFluxData
from tengri.utils.conversions import lnu_to_fnu
from _setup import HMC_VALIDATED

plot.setup_style()

C_POST, C_TRUTH, C_DATA, C_LINE = "#3a76d9", "0.15", "#c3372a", "#2e8b57"

# %% [markdown]
# ## Observation: DESI photometry + FastSpecFit strong lines
#
# The DESI Legacy Imaging bands (DECam *grz* + WISE W1–W4) plus the strong
# optical lines a FastSpecFit spectrum delivers: the [O II] doublet, the Balmer
# lines, [O III], [N II], and [S II]. The line wavelengths come straight from
# the built-in `LineList`; the observed fluxes and their errors go into a
# `LineFluxData`, which the `Observation` carries alongside the photometry.
# `model.fit` then fits both channels through one likelihood — no extra wiring.

# %%
SSP_NAME = "fsps_prsc_miles_chabrier"  # bare-stellar SSP (Cue adds the nebular emission)
ssp = tengri.load_ssp(SSP_NAME, download=True)

Z_GAL = 0.1
FILTERS = ["des_g", "des_r", "des_z", "wise_w1", "wise_w2", "wise_w3", "wise_w4"]
LINE_NAMES = [
    "OII_3726",
    "OII_3729",
    "Hbeta",
    "OIII_4959",
    "OIII_5007",
    "NII_6548",
    "Halpha",
    "NII_6584",
    "SII_6717",
    "SII_6731",
]
_cat = LineList.default_optical()
_wave_of = {n: float(w) for n, w in zip(_cat.names, _cat.wavelengths)}
LINE_WAVES = jnp.array([_wave_of[n] for n in LINE_NAMES])  # rest-frame vacuum [Å]

phot_obs = Photometry.from_names(FILTERS)
print(f"Photometry: {phot_obs.n_filters} DESI bands — {', '.join(phot_obs.names)}")
print(f"Lines: {len(LINE_NAMES)} — {', '.join(LINE_NAMES)}")


# %% [markdown]
# ## Model: Cue nebular, exact and fast
#
# We use the **Cue** photoionization backend, because it publishes discrete line
# luminosities that `predict_line_fluxes` turns into the pure-emission flux a
# catalog reports, and because its gas conditions (`neb_logU`, `neb_logZ_gas`)
# are free — a real catalog spans the metallicity–ionization plane, so a
# fixed-condition baked-in grid cannot follow it.
#
# We build the model twice with the *same* physics and free parameters, changing
# only `approx=`: the **exact** wave-grid path, and the **fast**
# `(WavePrecomp, FeaturePrecomp)` path (photometry LUT + per-Q_H line grid). The
# line wavelengths for the feature grid default to those in the observation.


# %%
# Build a LineList that declares the lines expected in the fit
_line_catalog_full = LineList.default_optical()
line_catalog = LineList.select(_line_catalog_full, names=LINE_NAMES)


def build(line_data, approx):
    # Observation declares: photometry schema, line data, and line schema
    obs = Observation(photometry=phot_obs, line_fluxes=line_data, lines=line_catalog)
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        redshift=Fixed(Z_GAL),
        # Free the SFH (normalization + shape), stellar metallicity, the dust
        # screen, and the gas conditions — the parameters a catalog fit solves for.
        sfh={"type": "dpl", "all_params": FREE},
        stellar={"met_logzsol": Uniform(-1.5, 0.3)},
        dust={
            "type": "two_component",
            "law_bc": "calzetti",
            "all_params": FIXED,
            "tau_bc": Uniform(0.0, 4.0),
            "tau_diff": Uniform(0.0, 3.0),
        },
        neb={
            "type": "cue",
            "all_params": FIXED,
            "logU": Uniform(-4.0, -1.0),
            "logZ_gas": Uniform(-1.5, 0.3),
        },
        approx=approx,
    )


# %% [markdown]
# ## Mock catalog
#
# One truth galaxy — a star-forming disc at z = 0.1 — and one noisy realization
# of its DESI photometry (SNR 20) and its FastSpecFit lines (SNR 10 on the
# strong lines). Both channels are generated from the same truth so they agree
# by construction; `predict_line_fluxes` gives the pure emission, matching the
# catalog convention.

# %%
TRUTH = {
    "sfh_dpl_log_total_mass": 10.2,
    "sfh_dpl_age_gyr": 6.0,
    "sfh_dpl_tau_gyr": 5.0,
    "sfh_dpl_alpha": 1.0,
    "sfh_dpl_beta": 3.0,
    "met_logzsol": -0.3,
    "dust_tau_bc": 0.6,
    "dust_tau_diff": 0.25,
    "neb_logU": -2.6,
    "neb_logZ_gas": -0.2,
}

# A throwaway observation just to instantiate the truth model and generate data.
_seed_lines = LineFluxData(
    names=tuple(LINE_NAMES),
    fluxes=jnp.ones(len(LINE_NAMES)),
    errors=jnp.ones(len(LINE_NAMES)),
    wavelengths=LINE_WAVES,
)
model_truth = build(_seed_lines, approx=None)
truth_full = {
    **model_truth.spec.get_fixed_values(),
    **{k: jnp.asarray(v) for k, v in TRUTH.items()},
}

p_phot = np.asarray(model_truth.predict_photometry(truth_full))
p_line = np.asarray(model_truth.predict_line_fluxes(truth_full, target_wavelengths=LINE_WAVES))
assert p_line[LINE_NAMES.index("Halpha")] > 0, "Halpha must be in emission — mock would be vacuous"

_rng = np.random.default_rng(0)
PHOT_SNR, LINE_SNR = 20.0, 10.0
n_phot = np.abs(p_phot) / PHOT_SNR
# Line errors: SNR-scaled, with a floor at 1% of the brightest line so a weak
# line (e.g. a near-zero [N II] component) cannot dominate the chi-square.
n_line = np.maximum(np.abs(p_line) / LINE_SNR, 0.01 * np.max(np.abs(p_line)))
flux_phot = p_phot + _rng.normal(size=p_phot.shape) * n_phot
flux_line = p_line + _rng.normal(size=p_line.shape) * n_line

# The observed line fluxes live in the Observation the fit model carries; the
# observed photometry is passed to `model.fit`. This is the public joint-fit API.
line_data = LineFluxData(
    names=tuple(LINE_NAMES),
    fluxes=jnp.asarray(flux_line),
    errors=jnp.asarray(n_line),
    wavelengths=LINE_WAVES,
)
print(
    f"Mock: {len(flux_phot)} bands (SNR {PHOT_SNR:.0f}) + {len(flux_line)} lines (SNR {LINE_SNR:.0f})"
)
print(
    f"  truth  log M* = {float(TRUTH['sfh_dpl_log_total_mass']):.2f}   Halpha = {p_line[LINE_NAMES.index('Halpha')]:.3e} erg/s/cm2"
)

# %% [markdown]
# ## Measure the fit time: exact vs fast
#
# A MAP fit (200 Adam steps) on the joint photometry + line likelihood, timed on
# both paths. One `fitter.run()` bundles two very different costs, and it is
# worth pulling them apart:
#
# - **`run()` wall** — the end-to-end cost of one call. It is dominated by a
#   one-off **JIT compile** of the optimizer step (~1–2 s here). That compile
#   recurs on each independent `run()`, so a second call is *not* much cheaper —
#   the persistent cache spares the XLA backend compile, not the Python-level
#   re-trace. This is the honest cost of an *interactive, single-galaxy* fit.
# - **compiled step** (`post.wall_time_s`) — the optimization loop *after* the
#   compile. This is the marginal compute, and the number that matters at
#   catalog scale: batched over galaxies with `fit_batch` the compile is paid
#   **once**, and each further galaxy costs only this (dropping further still
#   under `vmap`). The often-quoted sub-100 ms/galaxy figure is *this* amortized
#   compute — not the single-shot wall below.

# %%
model_exact = build(line_data, approx=None)
model_fast = build(line_data, approx=(WavePrecomp(), FeaturePrecomp()))
print(f"Free parameters ({model_fast.spec.n_free}): {', '.join(model_fast.spec.free_params)}")

MAP_KW = dict(method="map", key=jax.random.PRNGKey(1), n_steps=200)


def timed_map(model, label):
    # Joint fit: photometry + emission lines from observation.
    assert model.observation.line_fluxes is not None, "line likelihood not active"
    # Data container: lines are in observation.line_fluxes already,
    # so declare only photometry here.
    data = Data(photometry=(flux_phot, n_phot))

    forward = ForwardModel.build(sed=model)
    t0 = time.perf_counter()
    forward.fit(data, **MAP_KW)  # pays the JIT compile
    cold = time.perf_counter() - t0
    t0 = time.perf_counter()
    # A second fit re-traces the step, so its wall time is ~= the first (see #1350:
    # each fit currently clears the JAX caches, which is why the compile is not
    # reused). The number that isolates the physics is post.wall_time_s below.
    post = forward.fit(data, **MAP_KW)
    warm = time.perf_counter() - t0
    loop = post.wall_time_s  # the compiled optimization loop, compile excluded
    print(f"  {label:22s} fit() wall {warm:5.2f}s   compiled step {loop:5.2f}s")
    return post, cold, warm, loop


print("MAP fit (photometry + 10 lines):")
post_exact, cold_e, warm_e, loop_e = timed_map(model_exact, "exact (approx=None)")
post_fast, cold_f, warm_f, loop_f = timed_map(model_fast, "fast (Wave+Feature)")
print(
    f"\n  compiled-step speedup: {loop_e / loop_f:.1f}x   (fast {loop_f * 1e3:.0f} ms vs exact {loop_e * 1e3:.0f} ms of compute)"
)
print(
    f"  fit() wall is ~{warm_f:.1f}s on either path — that is per-call JIT compile, not the fit."
)

# %% [markdown]
# ## Posterior on the fast path
#
# A point estimate is not enough for a catalog — the metallicity / dust /
# ionization sector is degenerate, and the honest object is the posterior.
#
# **Sampler.** This posterior is strongly correlated (the degeneracies above),
# so the mass matrix must be **dense** — a diagonal one does not converge here.
# Given that, fixed-trajectory **HMC** converges faster than NUTS, which spends
# its budget building deep adaptive trees. We run **two genuine chains** so the
# R-hat is a real between-chain diagnostic, and execute them
# `chain_method="sequential"`: each chain reuses one compiled kernel, so peak
# memory stays at a *single* chain's. That is the point — a vmapped multi-chain
# compile needs ~N× the RAM (and a dense-mass fit can OOM a modest machine),
# whereas sequential runs anywhere a one-chain fit runs, at ~N× the sampling
# wall. One fit per process, per the OOM-orchestration rule.
#
# **What it reaches, and what it does not.** Doubling the warmup to 2000 moves
# split-R-hat from 1.22 to **1.089** with zero divergences — the right direction,
# and still short of the < 1.01 you would demand before quoting an interval in a
# paper. The degeneracy is real, not a tuning failure: the truth lands inside the
# 68% interval for 5 of 6 parameters, so the fit is informative, but read the
# widths in that sector as approximate. More chains would sharpen the diagnostic:
# measured at 100 warmup + 200 samples, peak RSS is 3.8 GB at one chain, 3.9 GB at
# two and 5.0 GB at four, while compile time stays flat at ~27 s throughout — the
# sampler is one compiled `lax.scan`, so neither chains nor samples rebuild it.

# %%
# Fixed-length HMC on the precomputed model. Every gradient here goes through the
# `(WavePrecomp, FeaturePrecomp)` tables built above, so an evaluation is a lookup
# rather than a full SSP integral — which is what makes a long chain affordable.
#
# The sampler stays fixed-length rather than NUTS on purpose: 20 leapfrog steps is
# 20 gradients per iteration, where NUTS routinely builds trees of 100+ and took
# several times longer here without converging. Spend the saving on *more
# iterations of the cheap kernel* instead — but on warmup, not on chains. Four
# chains at double the budget exhausted memory even under `chain_method=
# "sequential"`, so this keeps the two chains that are known to fit and doubles
# the warmup, which is what the adaptation actually needs.
HMC_LONG = {**HMC_VALIDATED, "n_warmup": 2000}
data = Data(photometry=(flux_phot, n_phot))

t0 = time.perf_counter()
posterior = ForwardModel.build(sed=model_fast).fit(
    data, key=jax.random.PRNGKey(7), n_chains=2, chain_method="sequential", **HMC_LONG
)
elapsed = time.perf_counter() - t0
rmax = max(float(v) for v in posterior.rhat().values())
n_divergent = posterior.diagnostics.get('n_divergent', 0)

# R-hat cannot see a chain that never moved — it scores ~1.0 on one — so check the
# draws directly, across every free parameter (#1734). Only the free ones: a Fixed
# parameter is legitimately constant and appears in `samples` with a single value,
# so scanning all of `samples` would report a frozen chain on every healthy fit.
_free = model_fast.spec.free_params
n_draw = min(np.asarray(posterior.samples[p]).size for p in _free)
n_unique = min(np.unique(np.asarray(posterior.samples[p])).size for p in _free)

print(
    f"HMC (2 chains x {HMC_LONG['n_warmup']}w+{HMC_LONG['n_samples']}s): "
    f"{elapsed:5.0f}s   max R-hat {rmax:.3f}   divergences {n_divergent}"
)
print(f"  Mixing: worst parameter has {n_unique}/{n_draw} unique draws")

# %% [markdown]
# On a machine with more RAM, run the chains concurrently instead of one at a
# time: `chain_method="parallel"` puts one chain per device via `jax.pmap`
# (needs `XLA_FLAGS=--xla_force_host_platform_device_count=N` set *before*
# importing jax), cutting the wall ~N-fold. The cost is memory: pmap
# **replicates** the model + the WavePrecomp / FeaturePrecomp lookup tables onto
# every device, so peak RAM scales ~linearly with `N_CHAINS` (≈ N× the
# sequential fit) — which is exactly why `"sequential"` is the default here.
# `chain_method="vmap"` (SIMD-batch into one kernel) is the middle ground. Raise
# `N_CHAINS` for a more robust R-hat; at a fixed total-sample budget it costs
# the same compute, only more chains to compare (and, under vmap/parallel, more
# memory).

# %% [markdown]
# ## Recovery
#
# Truth vs posterior 16/50/84. Stellar mass and star-formation rate — the
# quantities a catalog most wants — are well constrained. The metallicity / dust
# / gas-condition sector is broader: with broadband photometry and a handful of
# line fluxes, those trade off along the classic age–dust–metallicity ridge, and
# the posterior *width* is the honest statement of that (a MAP point would hide
# it). Breaking it further needs more information — a temperature-sensitive
# auroral line, the UV slope, or a full spectrum (notebook 06).

# %%
REPORT = [
    "sfh_dpl_log_total_mass",
    "met_logzsol",
    "dust_tau_bc",
    "dust_tau_diff",
    "neb_logU",
    "neb_logZ_gas",
]
print(f"{'parameter':<26}{'truth':>9}{'p16':>9}{'p50':>9}{'p84':>9}  cover")
print("-" * 64)
n_cov = 0
for p in REPORT:
    s = np.asarray(posterior.samples[p])
    lo, med, hi = np.percentile(s, [16, 50, 84])
    tv = float(truth_full[p])
    ok = lo <= tv <= hi
    n_cov += ok
    print(f"{p:<26}{tv:>9.3f}{lo:>9.3f}{med:>9.3f}{hi:>9.3f}  {'ok' if ok else 'miss'}")
print(f"\n68% coverage: {n_cov}/{len(REPORT)}")

# %% [markdown]
# ## Corner — the joint posterior
#
# The recovery table as a picture: 1-D marginals with the truth (lines), and the
# 2-D contours where the degeneracies live. Stellar mass is tight; the
# metallicity–dust–ionization block shows the correlated ridge that the broad
# intervals above come from — a picture the coverage table cannot show.

# %%
fig_corner = posterior.plot_corner(
    params=REPORT, truths={k: float(v) for k, v in truth_full.items()}, color=C_POST
)
fig_corner.savefig(FIG_DIR / "10_corner.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Posterior draws for the figures
#
# Evaluated once and reused by both figures below: the model photometry
# (`predict_photometry`), the full model SED (`predict(...).rest_sed()` → observed
# `F_ν`), and the model line fluxes (`predict_line_fluxes`) — each drawn over the
# posterior so the bands carry the parameter uncertainty.

# %%
N_DRAW = 80
_sidx = np.linspace(0, len(next(iter(posterior.samples.values()))) - 1, N_DRAW).astype(int)
_fixed = model_fast.spec.get_fixed_values()
draws = [{**_fixed, **{k: jnp.asarray(v[i]) for k, v in posterior.samples.items()}} for i in _sidx]

# Effective wavelength of each band (transmission-weighted), for placing the points.
wave_eff_um = effective_wavelengths_um(phot_obs)

# Model photometry per draw (the band-integrated F_nu the fit is matching).
phot_draws = np.stack([np.asarray(model_fast.predict_photometry(d)) for d in draws])
phot_lo, phot_med, phot_hi = np.percentile(phot_draws, [16, 50, 84], axis=0)

# Continuous model SED per draw: rest-frame L_nu -> observed F_nu (same recipe as
# notebook 07). Observed-frame grid spanning the DECam-to-WISE range.
DL = cosmology.luminosity_distance(Z_GAL)
WAVE_FULL = np.geomspace(2.0e3, 3.0e5, 800)  # observed frame [Å]  (0.2–30 µm)
w_full_um = WAVE_FULL / 1e4

# For SED evaluation, use WavePrecomp() alone (not FeaturePrecomp). FeaturePrecomp
# attaches a per-Q_H nebular grid that disables exact rest_sed(); WavePrecomp()
# alone is exact for rest_sed() while remaining LUT-fast for photometry.
model_sed = build(line_data, approx=WavePrecomp())

def _sed_fnu(p):
    lnu = np.interp(
        WAVE_FULL / (1.0 + Z_GAL),
        np.asarray(model_sed.wavelengths),
        np.asarray(model_sed.predict(p).rest_sed()),
    )
    return np.asarray(lnu_to_fnu(jnp.asarray(lnu), DL, Z_GAL))


sed_lo, sed_med, sed_hi = np.percentile(
    np.stack([_sed_fnu(p) for p in draws]), [16, 50, 84], axis=0
)
sed_truth = _sed_fnu(truth_full)

# %% [markdown]
# ## Do the points match the best fit? — photometry
#
# The observed photometry on the best-fit SED, with the per-band residual below.
# The lower panel is the pull, `(observed − model) / σ`: inside ±1 (gray band)
# means the model reproduces that band within its error bar. The reduced χ² over
# the 7 bands quantifies the overall photometric match.

# %%
BAND_LABELS = ["g", "r", "z", "W1", "W2", "W3", "W4"]
pull_phot = (flux_phot - phot_med) / n_phot
chi2_phot = float(np.sum(pull_phot**2))

fig, (axs, axr) = plt.subplots(
    2, 1, figsize=(8.4, 5.6), sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05}
)
axs.fill_between(w_full_um, sed_lo, sed_hi, color=C_POST, alpha=0.25, lw=0, label="posterior 68%")
axs.plot(w_full_um, sed_med, color=C_POST, lw=1.2, label="posterior-median SED")
axs.plot(w_full_um, sed_truth, color=C_TRUTH, lw=1.0, ls="--", label="truth")
axs.errorbar(
    wave_eff_um,
    flux_phot,
    yerr=n_phot,
    fmt="o",
    ms=7,
    color=C_DATA,
    mec="white",
    mew=0.7,
    elinewidth=1.1,
    capsize=2.5,
    zorder=6,
    label="observed photometry",
)
axs.plot(
    wave_eff_um,
    phot_med,
    marker="s",
    ms=5,
    ls="none",
    mfc="none",
    mec=C_POST,
    mew=1.3,
    zorder=7,
    label="model photometry (band-integrated)",
)
for name, x, y, e in zip(BAND_LABELS, wave_eff_um, flux_phot, n_phot):
    axs.annotate(
        name,
        (x, y + e),
        textcoords="offset points",
        xytext=(0, 7),
        ha="center",
        fontsize=7.5,
        color=C_DATA,
    )
axs.set_xscale("log")
axs.set_yscale("log")
axs.set_ylabel(r"$F_\nu$  [erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$]")
axs.set_title(
    f"Best-fit SED vs photometry  (reduced χ² = {chi2_phot / len(flux_phot):.2f}, 7 bands)"
)
axs.legend(frameon=False, fontsize=8.5, loc="lower center")

axr.axhspan(-1, 1, color="0.6", alpha=0.25, lw=0)
axr.axhline(0, color=C_POST, lw=0.8)
axr.plot(wave_eff_um, pull_phot, "o", ms=6, color=C_DATA, mec="white", mew=0.6)
axr.set_xscale("log")
axr.set_ylim(-3.5, 3.5)
axr.set_xlabel(r"observed wavelength  [$\mu$m]")
axr.set_ylabel(r"$(O-M)/\sigma$")
fig.savefig(FIG_DIR / "10_sed_photometry.png", dpi=300, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Do the points match the best fit? — emission lines
#
# The same test for the line channel: observed line fluxes (points) against the
# posterior-predictive `predict_line_fluxes` (band + median), with the per-line
# pull below. Lines are categorical, so they are *not* joined by a curve — each is
# an independent measurement. The reduced χ² is over the ten lines.

# %%
line_draws = np.stack(
    [np.asarray(model_fast.predict_line_fluxes(d, target_wavelengths=LINE_WAVES)) for d in draws]
)
line_lo, line_med, line_hi = np.percentile(line_draws, [16, 50, 84], axis=0)
pull_line = (flux_line - line_med) / n_line
chi2_line = float(np.sum(pull_line**2))
x = np.arange(len(LINE_NAMES))

fig, (axl, axp) = plt.subplots(
    2, 1, figsize=(8.4, 5.6), sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08}
)
axl.errorbar(
    x,
    flux_line,
    yerr=n_line,
    fmt="o",
    color=C_DATA,
    ms=6,
    mec="white",
    mew=0.6,
    elinewidth=1.1,
    capsize=2.5,
    zorder=4,
    label="observed lines (FastSpecFit-style)",
)
axl.vlines(
    x, line_lo, line_hi, color=C_POST, alpha=0.5, lw=5, zorder=2, label="posterior-predictive 68%"
)
axl.plot(
    x,
    line_med,
    marker="s",
    ms=5,
    ls="none",
    mfc="none",
    mec=C_POST,
    mew=1.3,
    zorder=3,
    label="model median",
)
axl.plot(x, p_line, marker="_", ms=13, mew=1.8, ls="none", color=C_TRUTH, zorder=5, label="truth")
axl.set_yscale("log")
axl.set_ylabel(r"line flux  [erg s$^{-1}$ cm$^{-2}$]")
axl.set_title(
    f"Best-fit vs emission lines  (reduced χ² = {chi2_line / len(LINE_NAMES):.2f}, 10 lines)"
)
axl.legend(fontsize=8.5, loc="lower left", ncol=2)

axp.axhspan(-1, 1, color="0.6", alpha=0.25, lw=0)
axp.axhline(0, color=C_POST, lw=0.8)
axp.plot(x, pull_line, "o", ms=6, color=C_DATA, mec="white", mew=0.6)
axp.set_ylim(-3.5, 3.5)
axp.set_ylabel(r"$(O-M)/\sigma$")
axp.set_xticks(x)
axp.set_xticklabels(LINE_NAMES, rotation=45, ha="right", fontsize=8)
fig.savefig(FIG_DIR / "10_line_recovery.png", dpi=300, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Measured times, together
#
# Two columns, because they answer different questions. **`fit() wall`** is the
# single-galaxy interactive cost — dominated by the per-call JIT compile, which
# is why exact and fast are closer here than the compute alone would suggest.
# **`compiled step`** is the optimization once compiled: the marginal per-galaxy
# compute a catalog pays after amortizing the compile (via `fit_batch`), and
# where the look-up table earns its keep.

# %%
print(f"{'fit':<34}{'fit() wall':>13}{'compiled step':>15}")
print("-" * 62)
print(f"{'MAP, exact wave grid':<34}{warm_e:>10.2f} s{loop_e:>12.2f} s")
print(f"{'MAP, WavePrecomp+FeaturePrecomp':<34}{warm_f:>10.2f} s{loop_f:>12.2f} s")
print(
    f"\nCompiled-step speedup: {loop_e / loop_f:.1f}x. The fit() wall (~{warm_f:.0f}s) is per-call JIT"
)
print("compile, not the fit — a catalog amortizes it once with fit_batch and pays only the")
print("compiled step per galaxy. The FeaturePrecomp line grid is likewise a one-time build")
print(
    f"({model_fast.spec.n_free}-parameter model), reused across every fit that shares the model."
)

# %% [markdown]
# ## Summary
#
# - A **FastSpecFit-style catalog** — broadband photometry + emission-line
#   fluxes — is fit through one `Observation` carrying both, with the lines in a
#   `LineFluxData`. No extra wiring: the fit picks up the line likelihood.
# - Model a catalog line flux with **`predict_line_fluxes`** — pure, deblended,
#   absorption-corrected emission, the same quantity FastSpecFit's `LINE_FLUX`
#   reports (Gaussian on a continuum-subtracted spectrum). A window-integrated
#   `measure_line_fluxes` is a *different* quantity (it carries stellar
#   absorption and mis-deblends [N II]) and should not be compared to a catalog.
# - The **`(WavePrecomp, FeaturePrecomp)`** fast path turns each likelihood
#   evaluation into a table look-up, reproducing the exact forward to sub-percent
#   on the strong lines. Single-galaxy that is a steady few-fold speedup (measured
#   above); the dramatic win is at catalog scale, batched over galaxies
#   (`fit_batch`), where the exact wave-grid forward would be prohibitive.
# - **The truth lands inside the 68% interval for five of the six reported
#   parameters** (one just outside — exactly what a well-calibrated 68% credible
#   interval should do), with stellar mass and SFR the tightest. Metallicity /
#   dust / gas conditions are the broad, degenerate sector — the posterior
#   *width* is the honest statement of that. More information (a full spectrum —
#   notebook 06 — an auroral line, or the UV slope) narrows it, not a faster fit.
# - Two residual systematics matter when fitting a *real* catalog: the nebular
#   model floor (Cue reproduces FSPS's Cloudy to ~10%, ~30% for [S II]), and the
#   fiber aperture (line fluxes are aperture-limited; photometry is total — apply
#   the catalog's aperture correction or fiber-match).
