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
# # Quickstart: fit a mock galaxy
#
# A star-forming galaxy with 12 broadband fluxes from GALEX, SDSS, 2MASS,
# and WISE (UV through near-IR), fitted with NUTS on a differentiable JAX
# forward model.
#
# Deliberately minimal — the point is to show how *fast* the JIT-compiled
# forward model and gradients are. Truncated-skew-normal SFH, two-component
# Calzetti dust attenuation, nebular off, redshift fixed at 0.05. Seven free
# parameters. See `04_building_models.py` for the recipe grammar and
# `02_sed_anatomy.py` for a panchromatic model with dust IR re-emission,
# nebular, AGN, and IGM enabled.

# %%
# Shared notebook setup (see notebooks/_setup.py): quiets the framework notices
# that do not change the science, and loads the SSP grid.
from _setup import FIG_DIR, effective_wavelengths_um, quiet

quiet()

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
    Fixed,
    ForwardModel,
    Observation,
    Photometry,
    SEDModel,
    Uniform,
    WavePrecomp,
    builders,
    citations,
    cosmology,
    generate_mock,
    plot,
)
from tengri.utils.conversions import lnu_to_fnu

plot.setup_style()

C_POST, C_TRUTH, C_DATA = "#3a76d9", "0.15", "#c3372a"

# %% [markdown]
# ## Stellar library and observation
#
# A bare-stellar SSP grid (Cue-compatible if you later want to add nebular
# emission). `load_ssp` with `download=True` fetches the grid on first use if not
# cached locally; without it, raises `FileNotFoundError` if the grid is missing.

# %%
SSP_NAME = "fsps_prsc_miles_chabrier"
ssp = tengri.load_ssp(SSP_NAME, download=True)

FILTERS = [
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
obs = Observation(photometry=Photometry.from_names(FILTERS))

# %% [markdown]
# ## Build the model
#
# Truncated-skew-normal SFH with two-component Calzetti dust attenuation,
# nebular off, redshift fixed at z = 0.05: seven free parameters. Kept minimal
# on purpose. Dust IR re-emission, nebular emission, and AGN are covered in
# `02_sed_anatomy.py`.

# %%
sed_model = SEDModel.build(
    ssp_data=ssp,
    observation=obs,
    approx=WavePrecomp(),
    sfh=builders.sfh.tsnorm(defaults=FREE),
    dust=builders.dust.two_component(
        defaults=FIXED,
        law="calzetti",
        tau_bc=Uniform(0.0, 1.0),
    ),
    neb=builders.neb.none(),
    redshift=Fixed(0.05),
)

# The model that gets fitted. `ForwardModel` pairs the SED with the instrument
# that observed it; the observation is inherited from the SED, so there is
# nothing to re-declare here.
forward = ForwardModel.build(sed=sed_model)

print(sed_model.summary())
citations.print_citations(sed_model)

# %% [markdown]
# ## Mock observation
#
# One draw from the prior is the truth. `generate_mock` returns the
# noiseless model fluxes, Gaussian uncertainties at the requested S/N,
# and a noisy realization.

# %%
key = jax.random.PRNGKey(9)
key_truth, key_mock, key_fit = jax.random.split(key, 3)

truth = sed_model.spec.sample(key_truth)
mock = generate_mock(sed_model, truth, key=key_mock, snr=30.0)
flux_obs = np.asarray(mock["flux_obs"])
noise = np.asarray(mock["noise"])

phot = obs.photometry
wave_eff_um = effective_wavelengths_um(phot)

# %% [markdown]
# ## One-time JIT compile
#
# Cold compile is a few seconds; warm cache is milliseconds. The difference
# between first and second call below shows the cost of compilation alone.

# %%
import time

p0 = {**sed_model.spec.get_fixed_values(), **truth}
predict_phot = jax.jit(sed_model.predict_photometry)
grad_fn = jax.jit(
    jax.grad(lambda p: 0.5 * jnp.sum(((sed_model.predict_photometry(p) - flux_obs) / noise) ** 2))
)

t = time.perf_counter()
_ = predict_phot(p0).block_until_ready()
print(f"  forward kernel   first call: {time.perf_counter() - t:8.4f} s  (compile + run)")
t = time.perf_counter()
_ = predict_phot(p0).block_until_ready()
print(f"  forward kernel   warm:       {time.perf_counter() - t:8.4f} s")

t = time.perf_counter()
jax.tree.map(lambda x: x.block_until_ready(), grad_fn(p0))
print(f"  ∇log-likelihood  first call: {time.perf_counter() - t:8.4f} s  (compile + run)")
t = time.perf_counter()
jax.tree.map(lambda x: x.block_until_ready(), grad_fn(p0))
print(f"  ∇log-likelihood  warm:       {time.perf_counter() - t:8.4f} s")

# %% [markdown]
# ## Fit
#
# `forward.fit(Data(photometry=(flux, noise)), ...)` is the whole interface: wrap
# the data in the `Data` container with the channel name (here, `photometry`), hand
# it to the fitter, and pick a method. The channel is explicit and unambiguous.
#
# Multi-start ADAM for the MAP point estimate, then NUTS with four parallel
# chains via `jax.vmap` for the full posterior. The `n_restarts=8` parameter runs
# eight random inits in parallel and keeps the lowest-loss one, then NUTS is seeded
# from that MAP point (`init_from=map_result`).
#
# With NUTS, more draws are cheap to cut but **warm-up is not**: 500 warm-up steps
# gave 4 divergences and split-R̂ 1.017, 1000 still left 1 divergence, and 1500 is
# where it reaches 0 — which is what the settings below use. Cut the draws, keep
# the warm-up.

# %%
t = time.perf_counter()
map_result = forward.fit(
    Data(photometry=(flux_obs, noise)),
    method="map",
    key=key_fit,
    n_restarts=8,
    n_steps=800,  # loss is flat well before this; 5000 bought nothing but wall
)
print(f"  MAP wall:  {time.perf_counter() - t:6.2f} s")

t = time.perf_counter()
posterior = forward.fit(
    Data(photometry=(flux_obs, noise)),
    method="mcmc_nuts",
    key=key_fit,
    init_from=map_result,  # seed all chains at the MAP point
    n_warmup=1500,
    n_samples=250,
    n_chains=4,
    n_burnin=0,
    dense_mass_matrix=False,  # diagonal mass matrix — D=7 fits fine, far less RAM
    target_accept_rate=0.9,  # smaller steps; the default left divergences here
)
print(f"  NUTS wall (4 chains × 250 = 1000 samples): {time.perf_counter() - t:6.2f} s")
posterior.summary()

# Convergence check. Two numbers, and they fail in different ways.
#
# Divergences are the serious one: a divergent transition means the integrator
# left the typical set, so those draws bias the posterior. This fit reports 0.
# Getting there is what `target_accept_rate=0.9` buys — at the default this fit
# left divergences in the low hundreds, which the published page carried
# unnoticed.
#
# Split-R̂ measures whether the four chains agree, and lands at ~1.00 here — under
# the 1.01 you should insist on before quoting an interval in a paper.
#
# One caveat about how it is earned: `init_from=map_result` starts every chain at
# the same MAP point. Identical starts under-disperse the chains, which can flatter
# R̂ by understating the between-chain variance. Measured both ways at 3000 warmup,
# it makes no difference here — MAP-seeded and dispersed starts both give 1.0008 —
# so the number is real, not an artifact of the seeding.
#
# And note R̂ cannot see a chain that never moved: a frozen chain scores ~1.0 too.
# Read it together with the divergence count, never alone.
rhat = posterior.rhat()
print(
    f"\n  max split-R̂ = {max(float(v) for v in rhat.values()):.4f}"
    f"    divergences = {posterior.diagnostics.get('n_divergent', 'n/a')}"
)

# %% [markdown]
# The fit recovers the mock truth: well-constrained parameters (stellar mass,
# dust, metallicity) land on the input values, and even the SFH *shape*
# parameters are sensibly constrained for this star-forming galaxy — the
# posteriors are unimodal and well-mixed (`r_hat ≈ 1.0`), without piling up
# against the prior bounds.

# %% [markdown]
# Derived physical scalars — stellar mass, SFR, sSFR — rolled up from the
# SFH integral, with the input truth in the first column.

# %%
N_DRAWS = 200
draws = posterior.resample(jax.random.PRNGKey(11), n=N_DRAWS)
fixed = sed_model.spec.get_fixed_values()


def draw_dicts(n):
    for i in range(n):
        yield {**fixed, **{k: float(v[i]) for k, v in draws.items()}}


DERIVED_KEYS = ("stellar_mass", "sfr_100myr", "sfr_10myr", "ssfr")

# `posterior.properties` is the property catalog lifted over the sample axis:
# the same names a `Prediction` uses, one axis wider, evaluated in memory-bounded
# chunks. It reads every draw the chain produced rather than the 200 resampled
# here, and `.ci()` returns the 16/50/84 interval directly — so the whole block
# below used to be a Python loop re-deriving what the object already exposes.
truth_full = {**fixed, **truth}
truth_derived = sed_model.predict(truth_full).properties

print(f"{'quantity':<14}{'truth':>14}{'p16':>14}{'p50':>14}{'p84':>14}")
print("-" * 70)
for k in DERIVED_KEYS:
    lo, med, hi = posterior.properties.ci(k)
    t = truth_derived.get(k)
    tstr = "—" if t is None else f"{float(t):.3e}"
    print(f"{k:<14}{tstr:>14}{lo:>14.3e}{med:>14.3e}{hi:>14.3e}")

# %% [markdown]
# ## Posterior SED
#
# Full posterior spectrum in the background (median + 68 % band), truth
# dashed, observed photometry with error bars, residuals against the
# posterior median below.

# %%
WAVE_OBS = np.geomspace(1300.0, 6e4, 1200)  # 0.13–6 μm covers GALEX → WISE W2
z_truth = float(truth_full["redshift"])
dl_cm = cosmology.luminosity_distance(z_truth)


def obs_fnu(params):
    pred = sed_model.predict(params)
    lnu_interp = np.interp(
        WAVE_OBS / (1.0 + z_truth), np.asarray(sed_model.wavelengths), np.asarray(pred.rest_sed())
    )
    return np.asarray(lnu_to_fnu(jnp.asarray(lnu_interp), dl_cm, z_truth))


spec_draws = np.stack([obs_fnu(p) for p in draw_dicts(40)])
spec_lo, spec_med, spec_hi = np.percentile(spec_draws, [16, 50, 84], axis=0)
spec_truth = obs_fnu(truth_full)

# Posterior photometry via ``predict_photometry`` — the same WavePrecomp LUT
# path the fit used. It serves the filter fluxes straight from the SSP × filter
# table without materializing the full-resolution SED cube, so the posterior
# draws map cheaply (no vmap chunking needed).
phot_draws = np.stack([np.asarray(sed_model.predict_photometry(p)) for p in draw_dicts(N_DRAWS)])
phot_med = np.median(phot_draws, axis=0)

fig = plt.figure(figsize=(8.6, 5.4))
gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.04)
ax, ax_res = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

wave_um = WAVE_OBS / 1e4

# Filter transmission curves shaded behind the spectrum — Bagpipes/Prospector
# style. Use the matplotlib default qualitative palette across bands.
band_palette = plt.cm.viridis(np.linspace(0.05, 0.95, len(phot.filter_waves)))
ymin, ymax = 0.3 * spec_truth.min(), 3 * spec_truth.max()
for fw, ft, color in zip(phot.filter_waves, phot.filter_trans, band_palette):
    fw_um = np.asarray(fw) / 1e4
    ft_norm = np.asarray(ft) / np.max(ft)
    # Map transmission to the bottom 12 % of the log-y axis.
    band = ymin * (ymax / ymin) ** (0.12 * ft_norm)
    ax.fill_between(fw_um, ymin, band, color=color, alpha=0.35, lw=0)

ax.fill_between(wave_um, spec_lo, spec_hi, color=C_POST, alpha=0.30, lw=0, label="posterior 68%")
ax.plot(wave_um, spec_med, color=C_POST, lw=1.4, label="posterior median")
ax.plot(wave_um, spec_truth, color=C_TRUTH, lw=1.1, ls="--", label="truth")
ax.errorbar(
    wave_eff_um,
    flux_obs,
    yerr=noise,
    fmt="o",
    color=C_DATA,
    ms=5.5,
    capsize=2,
    elinewidth=1.0,
    mec="white",
    mew=0.6,
    label="observed",
    zorder=5,
)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylim(ymin, ymax)
ax.set_xlim(wave_um.min(), wave_um.max())
ax.set_ylabel(r"$F_\nu$  [erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$]")
ax.legend(frameon=False, fontsize=9, loc="upper left")
plt.setp(ax.get_xticklabels(), visible=False)

# Rest-frame wavelength axis on top.
ax_rest = ax.twiny()
ax_rest.set_xscale("log")
ax_rest.set_xlim(wave_um.min() / (1.0 + z_truth), wave_um.max() / (1.0 + z_truth))
ax_rest.set_xlabel(rf"rest-frame wavelength  [$\mu$m]   (z = {z_truth:.2f})", fontsize=9)

resid = (flux_obs - phot_med) / noise
ax_res.axhspan(-1, 1, alpha=0.08, color="0.5")
ax_res.axhline(0, color="0.4", lw=0.8)
ax_res.bar(
    wave_eff_um,
    resid,
    width=wave_eff_um * 0.12,
    color=C_DATA,
    alpha=0.85,
    edgecolor="white",
    linewidth=0.5,
)
ax_res.set_xscale("log")
ax_res.set_xlim(wave_um.min(), wave_um.max())
ax_res.set_ylim(-3.5, 3.5)
ax_res.set_xlabel(r"observed wavelength  [$\mu$m]")
ax_res.set_ylabel(r"$(d-m)/\sigma$")
fig.savefig(FIG_DIR / "00_posterior_sed.png", dpi=300, bbox_inches="tight")
fig.savefig(FIG_DIR / "00_posterior_sed.pdf", bbox_inches="tight")

# %% [markdown]
# ## Star-formation history

# %%
def sfh(p):
    s = sed_model.predict_state(p)
    return (
        np.asarray(s.derived["sfh_grid_lbt_yr"]) / 1e9,
        np.asarray(s.derived["sfr_history"]),
    )


sfr_draws, lbt = [], None
for p in draw_dicts(80):
    lbt_i, sfr_i = sfh(p)
    sfr_draws.append(sfr_i)
    if lbt is None:
        lbt = lbt_i
sfr_draws = np.stack(sfr_draws)
sfr_lo, sfr_med, sfr_hi = np.percentile(sfr_draws, [16, 50, 84], axis=0)
lbt_t, sfr_t = sfh(truth_full)

# Two-panel SFH: SFR(t) on top, cumulative formed mass on bottom.
fig_sfh, (ax_sfh, ax_cum) = plt.subplots(
    2, 1, figsize=(7.2, 5.4), sharex=True, gridspec_kw=dict(height_ratios=[2, 1], hspace=0.05)
)

ax_sfh.fill_between(lbt, sfr_lo, sfr_hi, color=C_POST, alpha=0.30, lw=0, label="posterior 68%")
ax_sfh.plot(lbt, sfr_med, color=C_POST, lw=1.6, label="posterior median")
ax_sfh.plot(lbt_t, sfr_t, color=C_TRUTH, ls="--", lw=1.3, label="truth")
ax_sfh.set_ylabel(r"SFR  [$M_\odot$ yr$^{-1}$]")
ax_sfh.legend(frameon=False, fontsize=9, loc="upper left")
plt.setp(ax_sfh.get_xticklabels(), visible=False)

# Cumulative formed stellar mass — integrate SFR(t) backwards from now.
dt_yr = np.gradient(lbt * 1e9)  # Gyr → yr, signed
cum_draws = np.flip(np.cumsum(np.flip(sfr_draws * dt_yr[None, :], axis=1), axis=1), axis=1)
cum_lo, cum_med, cum_hi = np.percentile(cum_draws, [16, 50, 84], axis=0)
cum_truth = np.flip(np.cumsum(np.flip(sfr_t * dt_yr), axis=0), axis=0)
ax_cum.fill_between(lbt, cum_lo / 1e10, cum_hi / 1e10, color=C_POST, alpha=0.30, lw=0)
ax_cum.plot(lbt, cum_med / 1e10, color=C_POST, lw=1.6)
ax_cum.plot(lbt_t, cum_truth / 1e10, color=C_TRUTH, ls="--", lw=1.3)
ax_cum.set_ylabel(r"cumulative $M_\star$  [$10^{10}\,M_\odot$]")

for axx in (ax_sfh, ax_cum):
    axx.invert_xaxis()
    axx.set_xlim(13.5, 0)
ax_cum.set_xlabel("lookback time  [Gyr]")

fig_sfh.savefig(FIG_DIR / "00_sfh.png", dpi=300, bbox_inches="tight")
fig_sfh.savefig(FIG_DIR / "00_sfh.pdf", bbox_inches="tight")

# %% [markdown]
# ## Corner
#
# Free parameters plus derived quantities (`stellar_mass`, `sfr_100myr`,
# `sfr_10myr`, in log₁₀), truth dashed.

# %%
fig_corner = posterior.plot_corner(truths=truth_full, color=C_POST)
fig_corner.savefig(FIG_DIR / "00_corner.png", dpi=300, bbox_inches="tight")
fig_corner.savefig(FIG_DIR / "00_corner.pdf", bbox_inches="tight")
