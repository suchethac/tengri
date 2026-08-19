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
# # Nonparametric SFH fits to JWST broad and medium bands
#
# > ⚠️ **Experimental.** A research demonstration using experimental APIs that may change between releases.
#
# A Prospector-style setup: the `continuity` star-formation history
# (piecewise-constant SFR in age bins, Student-t priors on the log ratios of
# adjacent bins, following Leja et al. 2019) fit to JWST NIRCam photometry at
# $z = 1.5$, first for one galaxy and then for a small catalog. The physics is
# standard; the point of this page is operational. The ratio posterior is
# correlated, and the sampler settings that look like the fix for that are the
# ones that break it.
#
# The filter set is 7 broad bands (F090W to F444W) plus 12 medium bands (F140M
# to F480M). At $z = 1.5$ the medium bands sample the rest-frame optical
# continuum whose shape the ratio bins respond to, and F162M lands on
# H$\alpha$, so the fit uses a wNE stellar library with nebular emission baked
# into the templates.

# %%
import time
import warnings

# _setup must be imported before jax: it sets TF_CPP_MIN_LOG_LEVEL, which XLA
# only reads at import.
from _plot_style import setup_style
from _setup import FIG_DIR, effective_wavelengths_um, quiet

quiet()
setup_style()

# Correct notices, correct to ignore here: the wNE library warns that nebular
# emission is already in the templates and must be paired with the baked-in
# backend, which is the pairing used below; two_component reports the dust
# parameters it holds fixed, and holding them fixed is the point.
warnings.filterwarnings("ignore", message=r"(?s).*is a wNE .*")
warnings.filterwarnings("ignore", message=r"(?s).*run with that physics held constant.*")

import jax
import matplotlib.pyplot as plt
import numpy as np
from blackjax.diagnostics import effective_sample_size

import tengri
from tengri import (
    FIXED,
    FREE,
    Catalog,
    Fixed,
    ForwardModel,
    Observation,
    Photometry,
    SEDModel,
    Uniform,
    WavePrecomp,
)
from tengri.cosmology import age_at_z

# %% [markdown]
# ## Model
#
# Nine free parameters: total mass plus six bin ratios from the SFH, stellar
# metallicity, and the diffuse dust optical depth. `WavePrecomp` routes
# photometry through a precomputed SSP-by-filter table, which is what makes the
# samplers below affordable.
#
# One thing to know before reading any SFH below. The `continuity` bin edges are
# a fixed ladder out to 13.7 Gyr and do not follow the redshift, while the
# composite-population kernel zeroes star formation older than the universe at
# the fit redshift. At $z = 1.5$ that leaves the two oldest bins entirely
# outside the observable range: they take no likelihood, sample their Student-t
# prior, and drag on the sampler for nothing. `sfh_cont_log_total_mass` is the
# normalization over the whole ladder rather than the mass that survives the
# mask, so read it as the fitted parameter it is and take physical masses from
# the derived `log_mstar_surviving`. Passing `sfh={'bin_edges_gyr': ...}` looks
# like the fix and is accepted, but the value never reaches the forward pass, so
# the layout below is the default one either way.

# %%
Z_GAL = 1.5
SNR = 20.0
N_WARMUP = 400
N_SAMPLES = 400

BROAD = [
    "jwst_f090w",
    "jwst_f115w",
    "jwst_f150w",
    "jwst_f200w",
    "jwst_f277w",
    "jwst_f356w",
    "jwst_f444w",
]
MEDIUM = [
    "jwst_f140m",
    "jwst_f162m",
    "jwst_f182m",
    "jwst_f210m",
    "jwst_f250m",
    "jwst_f300m",
    "jwst_f335m",
    "jwst_f360m",
    "jwst_f410m",
    "jwst_f430m",
    "jwst_f460m",
    "jwst_f480m",
]

ssp_data = tengri.load_ssp("prsc_miles_chabrier_wNE")
phot = Photometry.from_names(BROAD + MEDIUM)

T_UNIV = float(age_at_z(Z_GAL))
print(f"age of universe at z = {Z_GAL}: {T_UNIV:.2f} Gyr")

t0 = time.perf_counter()
model = SEDModel.build(
    ssp_data=ssp_data,
    observation=Observation(photometry=phot),
    redshift=Fixed(Z_GAL),
    sfh={"type": "continuity", "all_params": FREE},
    met={"logzsol": Uniform(-1.5, 0.3)},
    dust={
        "type": "two_component",
        "law_bc": "calzetti",
        "all_params": FIXED,
        "tau_diff": Uniform(0.0, 2.0),
    },
    neb={"type": "ssp"},
    approx=WavePrecomp(),
)
build_wall = time.perf_counter() - t0

MASS_KEY = next(k for k in model.spec.free_params if k.endswith("log_total_mass"))
RATIO_KEYS = [p for p in model.spec.free_params if "ratio" in p]
FIXED_VALUES = dict(model.spec.get_fixed_values())
print(f"build + WavePrecomp table: {build_wall:.1f} s")
print(f"free parameters ({model.spec.n_free}): {', '.join(model.spec.free_params)}")

# %% [markdown]
# ## Mock galaxy
#
# Truth: log total mass $= 10.3$, constant SFR across the bins (all ratios zero),
# slightly subsolar metallicity, moderate diffuse dust. Fluxes are perturbed at
# S/N 20 per band. A positive ratio means the SFR rises towards the present.

# %%
rng = np.random.default_rng(4)

truth = dict(FIXED_VALUES)
truth[MASS_KEY] = 10.3
truth["met_logzsol"] = -0.3
truth["dust_tau_diff"] = 0.3
for name in RATIO_KEYS:
    truth[name] = 0.0

flux_true = np.asarray(model.predict_photometry(truth))
noise = np.abs(flux_true) / SNR
flux_obs = flux_true + rng.normal(size=flux_true.shape) * noise

wave_um = effective_wavelengths_um(phot)
fig, ax = plt.subplots(figsize=(7, 4))
sel_b = np.arange(len(BROAD))
sel_m = np.arange(len(BROAD), len(BROAD) + len(MEDIUM))
ax.errorbar(
    wave_um[sel_b],
    flux_obs[sel_b],
    yerr=noise[sel_b],
    fmt="s",
    ms=6,
    color="#3a76d9",
    label="broad (7)",
)
ax.errorbar(
    wave_um[sel_m],
    flux_obs[sel_m],
    yerr=noise[sel_m],
    fmt="o",
    ms=5,
    color="#c2571a",
    label="medium (12)",
)
ax.plot(wave_um, flux_true, "x", ms=5, color="0.3", label="truth")
ax.axvline(6564.61e-4 * (1 + Z_GAL), ls=":", color="0.6")
ax.text(6564.61e-4 * (1 + Z_GAL), ax.get_ylim()[1] * 0.9, r" H$\alpha$", color="0.4", fontsize=9)
ax.set(
    xscale="log",
    yscale="log",
    xlabel="observed wavelength [micron]",
    ylabel=r"$F_\nu$ [erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$]",
    title=f"Mock NIRCam photometry, z = {Z_GAL}, S/N {SNR:.0f}",
)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(FIG_DIR / "jwst_nonparametric_mock.png", dpi=110)
plt.show()

# %% [markdown]
# ## Individual mode
#
# MAP for the point estimate, NUTS for the posterior, with the mass matrix left
# at its default. The bin ratios are correlated, which invites two settings that
# both make things worse. Same data, same seed, 400 samples each:
#
# | sampler | wall | min ESS | s per effective sample | divergences |
# | --- | --- | --- | --- | --- |
# | `mcmc_nuts`, diagonal mass (default) | 174 s | 165 | 1.05 | 7 |
# | `mcmc_nuts`, `dense_mass_matrix=True` | 57 s | 10 | 5.78 | 77 |
# | `mcmc_hmc`, 20 leapfrog steps | 10 s | 4 | 2.32 | 0 |
# | `mcmc_hmc`, 150 leapfrog steps | 55 s | 35 | 1.58 | 0 |
#
# The two quickest rows are quick because they are not moving. Dense adaptation
# has to estimate 45 covariance entries from the same warmup that fixes 9
# diagonal ones, and the noisy metric destabilizes the integrator, so the wall
# time falls because divergent trajectories stop early. Short fixed-length
# trajectories never diverge and barely explore; the cure is length, not warmup,
# since HMC at the settings the method selection page validates (1000 warmup, a
# dense metric, 20 steps) still returns min ESS 4. Wall time alone ranks these
# in the wrong order, so read seconds per effective sample.
#
# The rows come from separate processes, one fit each, because NUTS warmup can
# peak well above the resident model. Repeating them moves the effective sample
# sizes by tens of percent, and the wall times with whatever else the machine is
# doing, but never by enough to reorder the rows.

# %%
forward = ForwardModel.build(sed=model)

t0 = time.perf_counter()
map_post = forward.fit(flux_obs, noise, method="map", key=jax.random.PRNGKey(1), n_steps=300)
map_wall = time.perf_counter() - t0
map_logm = float(map_post.params[MASS_KEY])
print(f"MAP: {map_wall:.1f} s, log total mass = {map_logm:.2f} (truth {truth[MASS_KEY]:.2f})")

t0 = time.perf_counter()
posterior = forward.fit(
    flux_obs,
    noise,
    method="mcmc_nuts",
    key=jax.random.PRNGKey(2),
    n_warmup=N_WARMUP,
    n_samples=N_SAMPLES,
    verbose=False,
)
nuts_wall = time.perf_counter() - t0


def min_ess(post):
    """Smallest effective sample size over the free parameters, and its name."""
    ess = {
        p: float(effective_sample_size(np.asarray(post.samples[p]).reshape(1, -1)))
        for p in model.spec.free_params
    }
    name = min(ess, key=ess.get)
    return ess[name], name


ess_val, ess_name = min_ess(posterior)
lo, med, hi = np.percentile(np.asarray(posterior.samples[MASS_KEY]), [16, 50, 84])
print(
    f"NUTS: {nuts_wall:.1f} s, min ESS {ess_val:.0f} ({ess_name}), {nuts_wall / ess_val:.2f} s/ESS"
)
print(
    f"log total mass = {med:.2f} [+{hi - med:.2f} -{med - lo:.2f}] (truth {truth[MASS_KEY]:.2f})"
)
print(f"divergences: {posterior.diagnostics.get('n_divergent', 0)}")

# %% [markdown]
# ## Catalog mode
#
# Eight galaxies at the same redshift, spread over a dex in mass, with their own
# dust, metallicity and SFH shape: two rising, two declining, two constant, and
# two with a burst in the youngest bin. `Catalog.fit` runs one vectorized
# program over all of them, and `forward_chunk_size` is left at its `"auto"`
# default, which sizes the batch from a memory budget; forcing `K = N` on a
# model this heavy can exceed available RAM.
#
# This is the one place the individual recipe does not carry over. Batched NUTS
# on this model spent over fifteen minutes in XLA compilation without producing
# a sample, because vectorizing a trajectory whose length is decided per step
# builds a far larger graph than a fixed-length one. HMC has no such branch and
# compiles in seconds, so catalog mode runs the 150-step trajectory from the
# table above. Read the per-galaxy effective sample sizes below as screening
# quality: use the catalog pass to rank and flag, then refit what matters with
# the individual recipe.

# %%
N_GAL = 8
SHAPES = {
    "rising": np.full(6, 0.4),
    "declining": np.full(6, -0.4),
    "constant": np.zeros(6),
    "burst": np.array([1.3, -0.3, 0.0, 0.0, 0.0, 0.0]),
}
SHAPE_ORDER = ["rising", "declining", "constant", "burst"] * 2

logm_true = rng.uniform(9.5, 10.8, N_GAL)
truths = []
band_names = list(phot.names)
table = {b: np.zeros(N_GAL) for b in band_names}
table.update({f"{b}_err": np.zeros(N_GAL) for b in band_names})
for i in range(N_GAL):
    tr = dict(truth)
    tr[MASS_KEY] = logm_true[i]
    tr["met_logzsol"] = rng.uniform(-0.8, 0.0)
    tr["dust_tau_diff"] = rng.uniform(0.1, 0.8)
    for k, r in zip(RATIO_KEYS, SHAPES[SHAPE_ORDER[i]] + rng.normal(0.0, 0.1, 6)):
        tr[k] = float(r)
    truths.append(tr)
    f_true = np.asarray(model.predict_photometry(tr))
    n = np.abs(f_true) / SNR
    f = f_true + rng.normal(size=f_true.shape) * n
    for j, b in enumerate(band_names):
        table[b][i] = f[j]
        table[f"{b}_err"][i] = n[j]

cat = Catalog(forward, table, flux_unit="cgs_fnu")

t0 = time.perf_counter()
catalog_post = cat.fit(
    key=jax.random.PRNGKey(6),
    method="mcmc_hmc",
    n_warmup=N_WARMUP,
    n_samples=N_SAMPLES,
    n_leapfrog_steps=150,
    target_accept_rate=0.9,
    verbose=False,
)
cat_wall = time.perf_counter() - t0

logm_med, logm_lo, logm_hi, ess_per_gal = [], [], [], []
for p in catalog_post:
    s = np.asarray(p.samples[MASS_KEY])
    q16, q50, q84 = np.percentile(s, [16, 50, 84])
    logm_med.append(q50)
    logm_lo.append(q16)
    logm_hi.append(q84)
    ess_per_gal.append(min_ess(p)[0])
logm_med, logm_lo, logm_hi = map(np.array, (logm_med, logm_lo, logm_hi))
resid = logm_med - logm_true
print(
    f"catalog HMC: {cat_wall:.1f} s = {cat_wall / N_GAL:.1f} s per galaxy "
    f"({N_GAL} galaxies, one program)"
)
print(f"log total mass recovery: bias {resid.mean():+.2f} dex, scatter {resid.std():.2f} dex")
print(f"min ESS per galaxy: {', '.join(f'{e:.0f}' for e in ess_per_gal)}")
print(f"divergences: {catalog_post.diagnostics.get('n_divergent_total', 'n/a')}")

# %% [markdown]
# ## What the catalog fits look like
#
# Three of the eight, one per SFH shape: the photometry with the posterior model
# over it, the recovered history against the input, and the mass posterior. The
# histories are drawn from the posterior, so the width of the band is the
# constraint the bands actually place on each age bin, and they stop at the age
# of the universe because that is where the kernel stops counting.

# %%
N_DRAWS = 40
SHOW = [0, 1, 3]  # rising, declining, burst
ORDER = np.argsort(wave_um)  # band lists are broad-then-medium, not by wavelength


def draw_products(post, key):
    """Model photometry and SFH for N_DRAWS posterior draws."""
    draws = post.resample(key, n=N_DRAWS)
    fluxes, sfrs, lbt = [], [], None
    for i in range(N_DRAWS):
        p = {**FIXED_VALUES, **{k: float(v[i]) for k, v in draws.items()}}
        fluxes.append(np.asarray(model.predict_photometry(p)))
        s = model.predict_state(p)
        sfrs.append(np.asarray(s.derived["sfr_history"]))
        if lbt is None:
            lbt = np.asarray(s.derived["sfh_grid_lbt_yr"]) / 1e9
    return np.array(fluxes), np.array(sfrs), lbt


fig, axes = plt.subplots(len(SHOW), 3, figsize=(11.5, 3.1 * len(SHOW)))
for row, gal in enumerate(SHOW):
    post = catalog_post[gal]
    fluxes, sfrs, lbt = draw_products(post, jax.random.PRNGKey(20 + gal))
    truth_state = model.predict_state(truths[gal])
    sfr_truth = np.asarray(truth_state.derived["sfr_history"])
    obs = np.array([table[b][gal] for b in band_names])
    err = np.array([table[f"{b}_err"][gal] for b in band_names])

    ax = axes[row, 0]
    ax.errorbar(
        wave_um[ORDER],
        obs[ORDER],
        yerr=err[ORDER],
        fmt="o",
        ms=4,
        color="0.25",
        label="data",
        zorder=3,
    )
    ax.fill_between(
        wave_um[ORDER],
        np.percentile(fluxes, 16, axis=0)[ORDER],
        np.percentile(fluxes, 84, axis=0)[ORDER],
        color="#3a76d9",
        alpha=0.45,
        label="posterior",
    )
    ax.set(xscale="log", yscale="log", ylabel=r"$F_\nu$")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(["1", "2", "3", "4", "5"])
    ax.set_title(f"galaxy {gal}: {SHAPE_ORDER[gal]}", fontsize=10)
    if row == 0:
        ax.legend(frameon=False, fontsize=8)

    ax = axes[row, 1]
    vis = lbt <= T_UNIV  # older than the universe at z: masked by the CSP kernel
    ax.fill_between(
        lbt[vis],
        np.percentile(sfrs, 16, axis=0)[vis],
        np.percentile(sfrs, 84, axis=0)[vis],
        color="#2f7d3f",
        alpha=0.4,
        label="posterior",
    )
    ax.plot(lbt[vis], np.median(sfrs, axis=0)[vis], color="#2f7d3f", lw=1.5)
    ax.plot(lbt[vis], sfr_truth[vis], color="0.1", lw=1.5, ls="--", label="truth")
    ax.set(xlim=(0, T_UNIV), yscale="log", ylabel=r"SFR [M$_\odot$/yr]")
    if row == 0:
        ax.legend(frameon=False, fontsize=8)

    ax = axes[row, 2]
    ax.hist(np.asarray(post.samples[MASS_KEY]).ravel(), bins=25, color="#3a76d9", alpha=0.7)
    ax.axvline(logm_true[gal], color="0.1", lw=1.5)
    ax.set(ylabel="samples")

axes[-1, 0].set_xlabel("observed wavelength [micron]")
axes[-1, 1].set_xlabel("lookback time [Gyr]")
axes[-1, 2].set_xlabel("log total mass (line: truth)")
fig.tight_layout()
fig.savefig(FIG_DIR / "jwst_nonparametric_examples.png", dpi=110)
plt.show()

# %% [markdown]
# ## Recovery

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

ax1.hist(np.asarray(posterior.samples[MASS_KEY]).ravel(), bins=30, color="#3a76d9", alpha=0.7)
ax1.axvline(truth[MASS_KEY], color="0.1", lw=1.5, label="truth")
ax1.axvline(map_logm, color="#c2571a", ls="--", lw=1.5, label="MAP")
ax1.set(xlabel="log total mass", ylabel="posterior samples", title="Individual: NUTS")
ax1.legend(frameon=False)

ax2.errorbar(
    logm_true,
    logm_med,
    yerr=[logm_med - logm_lo, logm_hi - logm_med],
    fmt="o",
    ms=6,
    color="#2f7d3f",
)
lims = [9.3, 11.0]
ax2.plot(lims, lims, ls=":", color="0.5")
ax2.set(
    xlim=lims,
    ylim=lims,
    xlabel="true log total mass",
    ylabel="recovered log total mass (median, 68%)",
    title=f"Catalog: batched HMC, {N_GAL} galaxies",
)

fig.tight_layout()
fig.savefig(FIG_DIR / "jwst_nonparametric_recovery.png", dpi=110)
plt.show()

# %% [markdown]
# ## Cost summary

# %%
print(f"{'stage':<44}{'wall':>10}")
print("-" * 54)
print(f"{'model build + WavePrecomp (one-time)':<44}{build_wall:>8.1f} s")
print(f"{'MAP, 9 parameters':<44}{map_wall:>8.1f} s")
print(f"{'NUTS posterior, single galaxy':<44}{nuts_wall:>8.1f} s")
print(f"{'catalog HMC (150 leapfrog), per galaxy':<44}{cat_wall / N_GAL:>8.1f} s")

# %% [markdown]
# Rules of thumb from this configuration, on a laptop CPU with a warm compile
# cache. A MAP takes a second or two. A nonparametric posterior that actually
# mixes costs a few minutes for a single galaxy, which is slower than the
# parametric SFHs in the tutorials and is the price of the correlated ratio
# geometry, not a sign of a misconfigured fit. Catalog mode costs tens of
# seconds per galaxy and buys that back by taking fewer effective samples each.
#
# A fit far outside those ranges usually means a wrong setting. Check that the
# model was built with `WavePrecomp` first. Then check what the sampler did
# rather than how long it took: a nonparametric fit that returns in seconds has
# almost certainly not moved. The "Choosing an inference method" page has the
# sampler decision table.
