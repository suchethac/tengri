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
# A star-forming galaxy with 14 broadband fluxes from GALEX, SDSS, 2MASS,
# and WISE (UV through mid-IR), fitted with NUTS on a differentiable JAX
# forward model.
#
# Truncated-skew-normal SFH, two-component Calzetti dust with modified-
# blackbody IR re-emission, nebular off, redshift fixed at 0.05. Seven free
# parameters. See `04_building_models.py` for the recipe grammar and
# `02_sed_anatomy.py` for a panchromatic model with nebular, AGN, and
# IGM enabled.

# %%
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri import (
    FIXED,
    FREE,
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
    load_ssp_data,
    plot,
)
from tengri.utils.conversions import lnu_to_fnu

plot.setup_style()
FIG_DIR = Path("_figs")
FIG_DIR.mkdir(exist_ok=True)

C_POST, C_TRUTH, C_DATA = "#3a76d9", "0.15", "#c3372a"

# %% [markdown]
# ## Stellar library and observation
#
# A bare-stellar SSP grid (Cue-compatible if you later want to add nebular
# emission). `download_ssp` fetches on first use.

# %%
SSP_NAME = "fsps_prsc_miles_chabrier"
ssp_path = Path("../data") / f"{SSP_NAME}.h5"
if not ssp_path.exists():
    ssp_path = Path(tengri.download_ssp(SSP_NAME))
ssp = load_ssp_data(str(ssp_path))

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
    "wise_w3",
    "wise_w4",
]
obs = Observation(photometry=Photometry.from_names(FILTERS))

# %% [markdown]
# ## Build the model
#
# A truncated skew-normal SFH, two-component Calzetti dust with a
# modified-blackbody IR re-emission (so the WISE mid-IR bands carry signal),
# nebular emission off, redshift fixed at z = 0.05. Seven free
# parameters. `model.summary()` prints the assembled pipeline;
# `citations.print_citations` pulls the bibliography straight from the
# registry — enough for a methods section.

# %%
sed_model = SEDModel.build(
    ssp_data=ssp,
    observation=obs,
    approx=WavePrecomp(),
    sfh=builders.sfh.tsnorm(defaults=FREE),
    dust=builders.dust.two_component(
        defaults=FIXED,
        law_bc="calzetti",
        tau_bc=Uniform(0.0, 1.0),
        emission=builders.dust.emission.modified_blackbody(defaults=FIXED),
    ),
    neb=builders.neb.none(),
    redshift=Fixed(0.05),
)
forward = ForwardModel.build(sed=sed_model, observation=obs)

print(sed_model.summary())
citations.print_citations(sed_model)

# %% [markdown]
# ## Mock observation
#
# One draw from the prior is the truth. `generate_mock` returns the
# noiseless model fluxes, Gaussian uncertainties at the requested SNR,
# and a noisy realisation.

# %%
key = jax.random.PRNGKey(7)
key_truth, key_mock, key_fit = jax.random.split(key, 3)

truth = sed_model.spec.sample(key_truth)
mock = generate_mock(sed_model, truth, key=key_mock, snr=30.0)
flux_obs = np.asarray(mock["flux_obs"])
noise = np.asarray(mock["noise"])

phot = obs.photometry
wave_eff_um = (
    np.array(
        [
            np.trapezoid(w * t, w) / np.trapezoid(t, w)
            for w, t in zip(phot.filter_waves, phot.filter_trans)
        ]
    )
    / 1e4
)

# %% [markdown]
# ## One-time JIT compile
#
# First touch of the photometric forward kernel and its gradient triggers
# XLA compilation against the precomputed SSP × filter LUT (the
# `WavePrecomp` knob set above). Cold cache is a few seconds; warm cache
# (`~/.cache/tengri_jax_cache`) is milliseconds. Subsequent calls are
# pure numeric throughput — no Python in the hot path.

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
# MAP (ADAM) for a point estimate, then NUTS for the full posterior.
# Each NUTS leapfrog step is one gradient evaluation above, and the
# warmup + sampling loop is jit-fused via `lax.scan` inside BlackJAX —
# Hamiltonian step throughput sets the wall.
#
# *Note*: the canonical entry is `forward.fit(...)` (issue #211), but it
# currently bypasses the WavePrecomp LUT (issue #281, ~16× slowdown on
# this model). Fitting through the SEDModel directly is the documented
# workaround until #281 lands.

# %%
from tengri.inference.fitter import Fitter

t = time.perf_counter()
map_result = Fitter(sed_model, flux_obs, noise, data_type="photometry").run(
    method="map", key=key_fit, n_steps=200
)
print(f"  MAP wall: {time.perf_counter() - t:6.2f} s")

# %% [markdown]
# ### Pre-warm + cached warmup
#
# A tiny throwaway NUTS call compiles the chain kernel and runs window
# adaptation once. The result is cached on the model, so the *real* NUTS
# call below reuses the step size and mass matrix — no second warmup.

# %%
t = time.perf_counter()
_ = Fitter(sed_model, flux_obs, noise, data_type="photometry").run(
    method="mcmc_nuts", key=key_fit, n_warmup=400, n_samples=20, n_burnin=0, verbose=False
)
print(f"  pre-warm wall: {time.perf_counter() - t:6.2f} s")

# %% [markdown]
# ### NUTS — four chains in parallel via vmap
#
# `n_chains=4` runs four independent chains over jittered initial
# positions, sharing the cached adaptation. They scan in parallel via
# `jax.vmap` and XLA SIMD — wall ≈ one chain's worth on CPU, 4× more
# samples for ~the same cost. 1 600 samples lands in a couple of seconds.

# %%
t = time.perf_counter()
posterior = Fitter(sed_model, flux_obs, noise, data_type="photometry").run(
    method="mcmc_nuts",
    key=key_fit,
    n_warmup=400,
    n_samples=400,
    n_chains=4,
    n_burnin=0,
)
print(f"  NUTS wall (4 chains × 400 samples): {time.perf_counter() - t:6.2f} s")
posterior.summary()

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
samples = {k: [] for k in DERIVED_KEYS}
for p in draw_dicts(N_DRAWS):
    d = sed_model.predict_derived(p)
    for k in DERIVED_KEYS:
        v = d.get(k)
        samples[k].append(float("nan") if v is None else float(v))

truth_full = {**fixed, **truth}
truth_derived = sed_model.predict_derived(truth_full)
print(f"{'quantity':<14}{'truth':>14}{'p16':>14}{'p50':>14}{'p84':>14}")
print("-" * 70)
for k in DERIVED_KEYS:
    lo, med, hi = np.percentile(samples[k], [16, 50, 84])
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
WAVE_OBS = np.geomspace(1300.0, 3e5, 1200)  # 0.13–30 μm covers GALEX → WISE W4
z_truth = float(truth_full["redshift"])
dl_cm = cosmology.luminosity_distance(z_truth)


def obs_fnu(params):
    rest = sed_model.predict_rest_sed(params, wave=WAVE_OBS / (1.0 + z_truth))
    return np.asarray(lnu_to_fnu(jnp.asarray(rest.sed), dl_cm, z_truth))


spec_draws = np.stack([obs_fnu(p) for p in draw_dicts(60)])
spec_lo, spec_med, spec_hi = np.percentile(spec_draws, [16, 50, 84], axis=0)
spec_truth = obs_fnu(truth_full)

phot_draws = np.asarray(jax.vmap(lambda p: forward.predict(p)["phot_fnu"])(draws))
phot_med = np.median(phot_draws, axis=0)

fig = plt.figure(figsize=(7.4, 5.6))
gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
ax, ax_res = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

wave_um = WAVE_OBS / 1e4
ax.fill_between(wave_um, spec_lo, spec_hi, color=C_POST, alpha=0.25, lw=0, label="posterior 68%")
ax.plot(wave_um, spec_med, color=C_POST, lw=1.2, label="posterior median")
ax.plot(wave_um, spec_truth, color=C_TRUTH, lw=1.0, ls="--", label="truth")
ax.errorbar(
    wave_eff_um,
    flux_obs,
    yerr=noise,
    fmt="o",
    color=C_DATA,
    ms=5,
    capsize=2,
    elinewidth=1.0,
    label="observed",
    zorder=5,
)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylabel(r"$F_\nu$  [erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$]")
ax.set_ylim(0.3 * spec_truth.min(), 3 * spec_truth.max())
ax.legend(frameon=False, fontsize=9, loc="lower right")
plt.setp(ax.get_xticklabels(), visible=False)

resid = (flux_obs - phot_med) / noise
ax_res.axhspan(-1, 1, alpha=0.08, color="0.5")
ax_res.axhline(0, color="0.4", lw=0.8)
ax_res.bar(wave_eff_um, resid, width=wave_eff_um * 0.12, color=C_DATA, alpha=0.8)
ax_res.set_xscale("log")
ax_res.set_ylim(-3.5, 3.5)
ax_res.set_xlabel(r"observed wavelength  [$\mu$m]")
ax_res.set_ylabel(r"$(d-m)/\sigma$")
fig.savefig(FIG_DIR / "00_posterior_sed.png", dpi=300, bbox_inches="tight")

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
for p in draw_dicts(120):
    lbt_i, sfr_i = sfh(p)
    sfr_draws.append(sfr_i)
    if lbt is None:
        lbt = lbt_i
sfr_draws = np.stack(sfr_draws)
sfr_lo, sfr_med, sfr_hi = np.percentile(sfr_draws, [16, 50, 84], axis=0)
lbt_t, sfr_t = sfh(truth_full)

fig_sfh, ax_sfh = plt.subplots(figsize=(7.2, 3.6))
ax_sfh.fill_between(lbt, sfr_lo, sfr_hi, color=C_POST, alpha=0.25, lw=0, label="68% band")
ax_sfh.plot(lbt, sfr_med, color=C_POST, lw=1.4, label="posterior median")
ax_sfh.plot(lbt_t, sfr_t, color=C_TRUTH, ls="--", lw=1.2, label="truth")
ax_sfh.invert_xaxis()
ax_sfh.set_xlim(13.5, 0)
ax_sfh.set_xlabel("lookback time  [Gyr]")
ax_sfh.set_ylabel(r"SFR  [$M_\odot$ yr$^{-1}$]")
ax_sfh.legend(frameon=False, fontsize=9)
fig_sfh.savefig(FIG_DIR / "00_sfh.png", dpi=300, bbox_inches="tight")

# %% [markdown]
# ## Corner
#
# Free parameters plus derived quantities (`stellar_mass`, `sfr_100myr`,
# `sfr_10myr`, in log₁₀), truth dashed.

# %%
fig_corner = posterior.plot_corner(truths=truth_full, color=C_POST)
fig_corner.savefig(FIG_DIR / "00_corner.png", dpi=300, bbox_inches="tight")
