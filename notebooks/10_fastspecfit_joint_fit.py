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
# Broadband photometry + emission-line-flux catalog fit (e.g., FastSpecFit for DESI).
# Uses `predict_line_fluxes` for pure, deblended, absorption-corrected emission —
# the same quantity FastSpecFit's `LINE_FLUX` reports. Window integrals carry stellar
# absorption and mis-deblend [N II], so do not compare them to this model.

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
    FeaturePrecomp,
    Fitter,
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

plot.setup_style()

C_POST, C_TRUTH, C_DATA, C_LINE = "#3a76d9", "0.15", "#c3372a", "#2e8b57"

# %% [markdown]
# ## Observation: DESI photometry + FastSpecFit strong lines
#
# DESI Legacy Imaging (DECam *grz* + WISE W1–W4) plus ten strong optical lines:
# [O II], Balmer lines, [O III], [N II], [S II]. FastSpecFit fits a stellar-continuum
# model (SPS templates with Balmer absorption), subtracts it, and fits each line as
# a Gaussian on the residual — with [N II]/Hα doublet kinematics tied so the blends
# deblend properly. Its output `LINE_FLUX` is pure, deblended, absorption-corrected
# emission. That is exactly what `predict_line_fluxes` returns: the backend's emitted
# line luminosity projected to a flux.

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
# ## Model: Cue nebular photoionization
#
# Cue backend because it publishes discrete line luminosities matching FastSpecFit's
# reporting convention. Gas conditions (`neb_logU`, `neb_logZ_gas`) are free — a real
# catalog spans the metallicity–ionization plane. Three build paths: exact wave-grid,
# photometry LUT alone (`WavePrecomp`), and fast pair adding nebular grid (`FeaturePrecomp`).


# %%
def build(line_data, approx):
    obs = Observation(photometry=phot_obs, line_fluxes=line_data)
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        redshift=Fixed(Z_GAL),
        # Free the SFH (normalization + shape), stellar metallicity, the dust
        # screen, and the gas conditions — the parameters a catalog fit solves for.
        sfh={"type": "dpl", "all_params": FREE},
        met={"logzsol": Uniform(-1.5, 0.3)},
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
# One truth galaxy: star-forming disc at z = 0.1 with generated DESI photometry
# (S/N 20) and FastSpecFit lines (S/N 10 on the strong lines). Line errors have
# a floor at 1% of the brightest line so a weak line (e.g., near-zero [N II]
# component) does not dominate the fit. Both channels come from the same truth,
# so they agree by construction. `predict_line_fluxes` gives the pure emission,
# matching the catalog convention.

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
# ## Measure the fit time: exact vs WavePrecomp vs fast
#
# MAP fit (200 Adam steps) on photometry + line likelihood, timed on all three paths.
# `WavePrecomp` is the photometry lookup — SSP × filter table replacing full integration
# with table look-up. `FeaturePrecomp` adds a per-Q_H nebular grid. The line channel
# already keeps nebular work off the per-gradient path, so on a line-flux fit the
# fast path has little left to remove. Read **compiled step** (`post.wall_time_s`):
# the optimization loop after JIT compile. That is the only column where `approx=`
# matters; the wall time is per-call compile, not the fit.

# %%
model_exact = build(line_data, approx=None)
model_wave = build(line_data, approx=WavePrecomp())
model_fast = build(line_data, approx=(WavePrecomp(), FeaturePrecomp()))
print(f"Free parameters ({model_fast.spec.n_free}): {', '.join(model_fast.spec.free_params)}")

MAP_KW = dict(method="map", key=jax.random.PRNGKey(1), n_steps=200)
N_REPS = 3

# The fourth entry is model_exact a second time: the A/A control. It gets its own
# ForwardModel so it is built exactly like the arms it calibrates.
ARMS = [
    ("exact (approx=None)", model_exact),
    ("WavePrecomp only", model_wave),
    ("fast (Wave+Feature)", model_fast),
    ("exact again (A/A)", model_exact),
]
for _label, _m in ARMS:
    # The line likelihood is active because the Observation carries line_fluxes;
    # the data passed here is photometry, and the observation says so, so there
    # is no channel to declare.
    assert _m.observation.line_fluxes is not None, "line likelihood not active"
built = [(label, ForwardModel.build(sed=model)) for label, model in ARMS]

# Round-robin AND rotated. Interleaving alone is not enough: whichever arm runs
# first in a pass pays the first-touch costs, and if the order never changes that
# penalty lands on the same arm every rep and survives the min. Rotating by one
# each pass moves every arm through a different slot, so no arm is structurally
# first.
# Before timing anything: what does the FIT actually run? `Fitter(approx="auto")`
# — the default, and what `fit()` uses — RE-RESOLVES the build-time `approx=`. So
# three models built three ways need not be three configurations at fit time.
# Print it rather than assume it. This is the same class of error as timing
# process position, one level down: an arm can be mislabelled as well as mistimed.
print("resolved fit-time precompute (what approx= actually buys a FIT):")
for _label, _fwd in built:
    _st = Fitter(_fwd, flux_phot, n_phot).model.approx
    _tags = [
        _n
        for _n, _on in (
            ("WavePrecomp", _st.wave_precomp),
            ("SpectrumPrecomp", _st.spectrum_precomp),
            ("FeaturePrecomp", _st.feature_precomp),
        )
        if _on
    ]
    print(f"  {_label:<22} -> {', '.join(_tags) or 'exact (no LUT)'}")

loops: dict[str, list[float]] = {label: [] for label, _ in built}
walls: dict[str, list[float]] = {label: [] for label, _ in built}
posts: dict[str, object] = {}
# Measure each arm N_REPS times, round-robin, with the start position rotating
# each pass. This isolates first-touch costs (page faults, lazy imports, SSP grid
# cache warmup) from the compute we care about. A/A control (exact built twice)
# says what machine noise floor to expect. Verdict rule: a speedup counts only if
# its excess over 1.0 is ≥2x the control's excess. min() discards first-touch.
print(f"MAP fit (photometry + 10 lines), {N_REPS} rotated reps, min reported:")
for rep in range(N_REPS):
    cut = rep % len(built)
    for label, fwd in built[cut:] + built[:cut]:
        t0 = time.perf_counter()
        post = fwd.fit(flux_phot, n_phot, **MAP_KW)
        walls[label].append(time.perf_counter() - t0)
        loops[label].append(post.wall_time_s)
        posts[label] = post

loop = {label: min(v) for label, v in loops.items()}
wall = {label: min(v) for label, v in walls.items()}
for label, _ in built:
    spread = max(loops[label]) / min(loops[label])
    print(
        f"  {label:22s} fit() wall {wall[label]:5.2f}s   "
        f"compiled step {loop[label]:5.2f}s   (rep spread {spread:4.1f}x)"
    )

L_E, L_W, L_F, L_AA = (label for label, _ in built)
post_exact, post_wave, post_fast = posts[L_E], posts[L_W], posts[L_F]
loop_e, loop_w, loop_f, loop_aa = loop[L_E], loop[L_W], loop[L_F], loop[L_AA]
warm_e, warm_w, warm_f = wall[L_E], wall[L_W], wall[L_F]

r_aa = max(loop_e, loop_aa) / min(loop_e, loop_aa)
r_fast = loop_e / loop_f

RESOLVE_MARGIN = 2.0
resolved = (r_fast - 1.0) > RESOLVE_MARGIN * (r_aa - 1.0)
print(f"\n  A/A control (same model, twice): {r_aa:5.2f}x  <- the noise floor")
print(
    f"  compiled-step ratio exact -> fast: {r_fast:5.2f}x   "
    f"(fast {loop_f * 1e3:.0f} ms vs exact {loop_e * 1e3:.0f} ms)"
)
print("  attributed, one knob at a time:")
print(f"    WavePrecomp        {loop_e:6.3f}s -> {loop_w:6.3f}s   {loop_e / loop_w:6.2f}x")
print(f"    + FeaturePrecomp   {loop_w:6.3f}s -> {loop_f:6.3f}s   {loop_w / loop_f:6.2f}x")
if not resolved:
    print(
        f"  -> NOT resolved: on this fit, the two opt-ins buy nothing measurable."
        f" An observation with a line channel already keeps nebular work off the"
        f" per-gradient path, so exact starts fast and the lookups have little left to remove."
    )
else:
    print(
        f"  -> resolved: excess {r_fast - 1.0:.2f} is more than {RESOLVE_MARGIN:.0f}x"
        f" the control excess {r_aa - 1.0:.2f}."
    )
print(f"  fit() wall (~{warm_f:.1f}s) is per-call JIT compile, not the fit.")

# %% [markdown]
# ## Posterior on the fast path
#
# Posterior on photometry alone (no line channel); strongly correlated, so a dense
# mass matrix. Two chains, sequential execution: each reuses one compiled kernel,
# so peak memory equals one chain. One fit per notebook process (dense matrix can
# hit 20+ GB on D ≈ 8).

# %%
N_WARMUP, N_SAMPLES, N_CHAINS, N_LEAPFROG = 500, 300, 2, 100
t0 = time.perf_counter()
posterior = ForwardModel.build(sed=model_fast).fit(
    flux_phot,
    n_phot,
    method="mcmc_hmc",
    key=jax.random.PRNGKey(7),
    n_warmup=N_WARMUP,
    n_samples=N_SAMPLES,
    n_chains=N_CHAINS,
    n_leapfrog_steps=N_LEAPFROG,
    dense_mass_matrix=True,
    target_accept_rate=0.85,
    chain_method="sequential",  # peak memory = one chain (runs on cheap hardware)
)
rmax = max(float(v) for v in posterior.rhat().values())
print(
    f"HMC ({N_CHAINS} chains x {N_WARMUP}w+{N_SAMPLES}s, L={N_LEAPFROG}, sequential): "
    f"{time.perf_counter() - t0:5.0f}s   max R-hat {rmax:.3f}   "
    f"divergences {posterior.diagnostics.get('n_divergent', 'n/a')}"
)

# %% [markdown]
# Parallel chains with `jax.pmap` run ~N-fold faster but replicate model +
# LUTs onto every device, so peak RAM scales linearly with chains. `vmap` is
# the middle ground. Sequential is safest for dense mass matrix on modest machines.

# %% [markdown]
# ## Recovery
#
# Truth vs posterior 16/50/84 percentiles. Stellar mass and SFR are well constrained.
# Metallicity / dust / gas conditions trade off along the age–dust–metallicity ridge;
# posterior width is the honest statement of that degeneracy. With broadband photometry
# and a handful of line fluxes, those three parameters cannot be broken further. Full
# spectrum, temperature-sensitive auroral line, or UV slope would tighten it.

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
# 1-D marginals show each parameter's distribution with truth (lines). 2-D contours
# show the covariance structure where degeneracies live. Stellar mass is tight;
# the metallicity–dust–ionization block shows the correlated ridge from the broad
# posterior intervals above.

# %%
fig_corner = posterior.plot_corner(
    params=REPORT, truths={k: float(v) for k, v in truth_full.items()}, color=C_POST
)
fig_corner.savefig(FIG_DIR / "10_corner.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Posterior draws for figures
#
# Model photometry, full SED, and line fluxes drawn over the posterior distribution
# to propagate parameter uncertainty into the figure bands. Each draw resamples the
# posterior to capture the 16/50/84 percentile range.

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


def _sed_fnu(p):
    lnu = np.interp(
        WAVE_FULL / (1.0 + Z_GAL),
        np.asarray(model_fast.wavelengths),
        np.asarray(model_fast.predict(p).rest_sed()),
    )
    return np.asarray(lnu_to_fnu(jnp.asarray(lnu), DL, Z_GAL))


sed_lo, sed_med, sed_hi = np.percentile(
    np.stack([_sed_fnu(p) for p in draws]), [16, 50, 84], axis=0
)
sed_truth = _sed_fnu(truth_full)

# %% [markdown]
# ## Do the points match the best fit? — photometry
#
# Observed photometry on the posterior-median SED. Lower panel is the pull `(O−M)/σ`:
# inside ±1 (gray band) means the model reproduces that band within error. Reduced χ²
# quantifies the overall photometric match.

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
# Observed line fluxes against posterior-predictive predictions. Lines are categorical,
# so no curve joins them — each is an independent measurement. Lower panel: pull over
# the ten lines. Reduced χ² quantifies the line match.

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
# **`fit() wall`**: end-to-end cost of one call, including per-call JIT compile.
# This is the honest interactive cost for a single galaxy.
# **`compiled step`** (`post.wall_time_s`): optimization loop after compile. This is the
# marginal per-galaxy compute that a catalog pays after amortizing the compile via `fit_batch`.
# It is the only metric where an `approx=` choice can show up. Rep spread
# (slowest/fastest) reveals first-touch costs; the A/A arm (exact built twice) stays near
# 1.0x. Ratios below are quoted against that control — a ratio without a noise floor
# is not a measurement.

# %%
print(f"{'fit':<34}{'fit() wall':>13}{'compiled step':>15}")
print("-" * 62)
print(f"{'MAP, exact wave grid':<34}{warm_e:>10.2f} s{loop_e:>12.2f} s")
print(f"{'MAP, WavePrecomp only':<34}{warm_w:>10.2f} s{loop_w:>12.2f} s")
print(f"{'MAP, WavePrecomp+FeaturePrecomp':<34}{warm_f:>10.2f} s{loop_f:>12.2f} s")
print(f"{'A/A control (exact, again)':<34}{'':>10}  {loop_aa:>12.2f} s")
print(f"\nNoise floor (A/A, same model twice): {r_aa:.2f}x. Compiled-step ratio:")
print(f"{loop_e / loop_f:.2f}x overall — {loop_e / loop_w:.2f}x from WavePrecomp, a further")
print(
    f"{loop_w / loop_f:.2f}x from FeaturePrecomp. "
    + (
        "None of these resolve against the control."
        if not resolved
        else "The overall ratio resolves against the control."
    )
)
print(f"The fit() wall (~{warm_f:.1f}s) is per-call JIT")
print("compile, not the fit — a catalog amortizes it once with fit_batch and pays only the")
print("compiled step per galaxy. The FeaturePrecomp nebular grid is likewise a one-time build")
print(
    f"({model_fast.spec.n_free}-parameter model), reused across every fit that shares the model."
)

# %% [markdown]
# ## Summary
#
# - Fit FastSpecFit photometry + line fluxes through one `Observation` carrying both
#   in `LineFluxData`. The fit automatically activates the line likelihood.
# - Use `predict_line_fluxes` for pure, deblended, absorption-corrected emission
#   (Gaussian on continuum-subtracted spectrum) matching `FastSpecFit.LINE_FLUX`.
#   Window integrals are different: they carry stellar absorption and mis-deblend [N II].
# - Build-time `WavePrecomp` and `FeaturePrecomp` are lookup tables; their win at catalog
#   scale (batched `fit_batch`) is real and separate — the build is shared across galaxies.
#   A line-channel observation already keeps nebular work off the per-gradient path, so
#   on this fit the fast path buys little.
# - Stellar mass and SFR are well constrained. Metallicity / dust / gas conditions
#   degenerate along the age–dust–metallicity ridge; the posterior width is the honest
#   statement of that. Tighter constraints need a full spectrum, auroral line, or UV slope.
# - Real catalogs: account for Cue's systematics (reproduces Cloudy to ~10%, ~30% for [S II]),
#   fiber-aperture effects (line fluxes aperture-limited; photometry total), and dense mass
#   matrix RAM cost (one fit per notebook process).
