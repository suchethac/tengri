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
# # Quickstart — fit a galaxy in one screen
#
# A star-forming galaxy SED, simulated and fitted back, on a laptop, in a
# couple of minutes. The forward model is pure JAX, so the likelihood and
# its gradient are computed together; NUTS samples directly from the
# posterior without hand-tuned proposals.
#
# Two ideas worth pausing on as the model is built:
#
# - `model.summary()` — every assembled piece (SSP, filters, components,
#   precomputation) in one block. Glance at this before any fit.
# - `tengri.citations` — the bibliography of every physics ingredient,
#   collected automatically. Most of a methods section, generated.
#
# Next: [`01_why_jax.py`](01_why_jax.py) on differentiable inference,
# [`02_sed_anatomy.py`](02_sed_anatomy.py) on the panchromatic anatomy,
# [`03_discovering_the_menu.py`](03_discovering_the_menu.py) on the menu.

# %%
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri import (
    Fitter,
    Observation,
    Photometry,
    SEDModel,
    citations,
    generate_mock,
    load_ssp_data,
    plot,
    recipes,
)

plot.setup_style()
FIG_DIR = Path("_figs")
FIG_DIR.mkdir(exist_ok=True)

# %% [markdown]
# ## SSP grid
#
# The Cue nebular emulator used by `recipes.star_forming_photometry()`
# requires a **bare-stellar** SSP (no baked-in nebular contribution).
# `list_known_ssps()` lists the catalogue; `download_ssp()` fetches on
# demand.

# %%
SSP_NAME = "fsps_prsc_miles_chabrier"
ssp_path = Path("../data") / f"{SSP_NAME}.h5"
if not ssp_path.exists():
    ssp_path = Path(tengri.download_ssp(SSP_NAME))
ssp = load_ssp_data(str(ssp_path))

# %% [markdown]
# ## Photometry
#
# Six SDSS+WISE bands, observed-frame. Wrap the filter set in `Photometry`,
# pass it to `Observation` (the projection target of the model).

# %%
filters = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z", "wise_w1"]
obs = Observation(photometry=Photometry.from_names(filters))

# %% [markdown]
# ## Build the model
#
# `recipes.mock_recovery_minimal()` is the cleanest stable starting
# point: a truncated-skew-normal SFH, single-knob dust attenuation,
# nebular emission off, redshift fixed at 0.05. Five free parameters.
# Notebook [`02_sed_anatomy.py`](02_sed_anatomy.py) layers the full
# panchromatic physics (nebular, AGN, dust IR, IGM) onto a kitchen-sink
# model.

# %%
model = SEDModel.build(
    ssp_data=ssp,
    observation=obs,
    **recipes.mock_recovery_minimal(),
)
print(model.summary())

# %% [markdown]
# ## Provenance
#
# Every physical ingredient in the model carries its citations through
# the registry. `collect_citations` walks the pipeline; the resulting
# `Bibliography` exports to a printable list or to BibTeX.

# %%
citations.print_citations(model)
# bib_text = citations.citations_bibtex(model)  # → .bib for a manuscript

# %% [markdown]
# ## Mock galaxy
#
# Sample one set of parameters from the prior; this is the "truth" we will
# try to recover. `generate_mock` returns true fluxes, Gaussian
# uncertainties at the requested SNR, and a noisy realisation.

# %%
key = jax.random.PRNGKey(7)
key_truth, key_mock, key_fit = jax.random.split(key, 3)

truth = model.spec.sample(key_truth)
mock = generate_mock(model, truth, key=key_mock, snr=30.0)
flux_obs = mock["flux_obs"]
noise = mock["noise"]

# %% [markdown]
# ## Fit
#
# NUTS samples the posterior using the model gradient. `Fitter` selects a
# χ² photometric likelihood by inspecting `obs`. `tengri.lean()` is the
# default cache mode and drops the JIT engine after the fit.

# %%
fitter = Fitter(model, flux_obs, noise, data_type="photometry")
posterior = fitter.run(
    method="mcmc_nuts",
    key=key_fit,
    n_warmup=400,
    n_samples=400,
)
posterior.summary()

# %% [markdown]
# ## Hero figure — posterior SED and SFH
#
# Top: the panchromatic SED with the posterior median, 68 % credible band,
# truth (dashed), and the six observed photometry points. Bottom: the
# inferred SFH against the truth.

# %%
draw_batch = posterior.resample(jax.random.PRNGKey(11), n=200)
phot_draws = np.asarray(
    jax.vmap(lambda p: model.predict_observables(p).phot_fnu)(draw_batch)
)
phot = obs.photometry
wave_eff = np.array([
    np.trapezoid(w * t, w) / np.trapezoid(t, w)
    for w, t in zip(phot.filter_waves, phot.filter_trans)
]) / 1e4  # μm, transmission-weighted mean per band

fig = plt.figure(figsize=(7.2, 6.4))
gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.35)
ax_sed = fig.add_subplot(gs[0])
ax_sfh = fig.add_subplot(gs[1])

# SED panel
flux_lo, flux_med, flux_hi = np.percentile(phot_draws, [16, 50, 84], axis=0)
ax_sed.fill_between(wave_eff, flux_lo, flux_hi, color="#3a76d9", alpha=0.25, label="68% band")
ax_sed.plot(wave_eff, flux_med, color="#3a76d9", lw=1.4, label="posterior median")
ax_sed.plot(wave_eff, np.asarray(mock["flux_true"]), color="0.2", lw=1.0, ls="--", label="truth")
ax_sed.errorbar(wave_eff, np.asarray(flux_obs), yerr=np.asarray(noise),
                fmt="o", color="#c3372a", ms=4, capsize=2, label="observed")
ax_sed.set_xscale("log"); ax_sed.set_yscale("log")
ax_sed.set_xlabel(r"observed wavelength $\lambda$  [$\mu$m]")
ax_sed.set_ylabel(r"$F_\nu$  [erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$]")
ax_sed.legend(frameon=False, fontsize=9, loc="best")

# SFH panel — SFH history lives in the orchestrator state, not the
# Prediction wrapper (which only exposes scalar derived quantities).
def _sfh(p_dict):
    s = model.predict_state(p_dict)
    return (np.asarray(s.derived["sfh_grid_lbt_yr"]) / 1e9,
            np.asarray(s.derived["sfr_history"]))

sfr_draws, lbt = [], None
fixed = model.spec.get_fixed_values()
for i in range(80):
    p = {k: float(v[i]) for k, v in draw_batch.items()}
    p = {**fixed, **p}
    lbt_i, sfr_i = _sfh(p)
    sfr_draws.append(sfr_i)
    if lbt is None:
        lbt = lbt_i
sfr_draws = np.stack(sfr_draws)
sfr_lo, sfr_med, sfr_hi = np.percentile(sfr_draws, [16, 50, 84], axis=0)
ax_sfh.fill_between(lbt, sfr_lo, sfr_hi, color="#3a76d9", alpha=0.25)
ax_sfh.plot(lbt, sfr_med, color="#3a76d9", lw=1.4, label="posterior median")
lbt_t, sfr_t = _sfh({**fixed, **truth})
ax_sfh.plot(lbt_t, sfr_t, color="0.2", ls="--", lw=1.0, label="truth")
ax_sfh.set_yscale("log")
ax_sfh.invert_xaxis()
ax_sfh.set_xlabel("lookback time [Gyr]")
ax_sfh.set_ylabel(r"SFR  [$M_\odot$/yr]")
ax_sfh.legend(frameon=False, fontsize=9)
fig.savefig(FIG_DIR / "00_posterior_sed_sfh.png", dpi=300, bbox_inches="tight")

# %% [markdown]
# ## Posterior corner
#
# The physically meaningful free parameters — stellar mass, current SFR,
# birth-cloud τ, and redshift — are well constrained from six bands plus
# IR coverage. Wider tails appear on the SFH shape parameters, which is
# expected.

# %%
import corner
sampled = list(posterior.samples.keys())
prefer = ["sfh_tsnorm_log_peak_sfr", "sfh_tsnorm_peak_lbt_gyr",
          "sfh_tsnorm_width_gyr", "dust_tau_bc",
          "sfh_dpl_log_peak_sfr", "sfh_dpl_alpha", "redshift"]
# Keep only params with non-trivial dynamic range (a stuck chain or
# tightly-pinned param breaks corner.corner otherwise).
corner_params = []
for p in prefer:
    if p not in sampled:
        continue
    s = np.asarray(posterior.samples[p])
    if s.std() > 1e-6 * (np.abs(s.mean()) + 1e-9):
        corner_params.append(p)
corner_params = corner_params[:5]
samples_arr = np.column_stack([np.asarray(posterior.samples[k]) for k in corner_params])
truths_arr = [float(truth[k]) if k in truth else None for k in corner_params]
fig_corner = corner.corner(
    samples_arr,
    labels=[k.replace("sfh_dpl_", "").replace("_", " ") for k in corner_params],
    truths=truths_arr,
    color="#3a76d9",
    truth_color="0.2",
    show_titles=True,
    title_fmt=".2f",
    quantiles=[0.16, 0.5, 0.84],
)
fig_corner.savefig(FIG_DIR / "00_corner.png", dpi=300, bbox_inches="tight")
