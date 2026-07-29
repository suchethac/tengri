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
# # Recovering a bursty star-formation history from emission-line fluxes + photometry
#
# > ⚠️ **Experimental.** A research demonstration that explores experimental
# > features and may use APIs that change between releases; it sits outside the
# > supported tutorial sequence.
#
# **The question.** tengri models a galaxy's star-formation history (SFH) as a
# smooth backbone times a stochastic **Gaussian-process "field"** — coherent
# bursts drawn from a damped-random-walk power spectrum. Does that field SFH
# actually *recover* when we inject a known one and fit it back? And how do the
# cheap approximate samplers behave as we grow the SFH dimension?
#
# **The model.** A **rising, still-star-forming double power law** (Carnall et al.
# 2018) — *not* a quiescent galaxy — modulated by the GP field:
#
# $$
# \mathrm{SFR}(t) = \mathrm{SFR}_{\mathrm{DPL}}(t)\,\bigl[\log M_\star\bigr]
#                   \times \exp\!\bigl(\mathrm{GP}(t) - \tfrac12 K_0\bigr),
# \qquad
# P(\omega) = \frac{\sigma_{\mathrm{PSD}}^2\,\tau_{\mathrm{PSD}}}
#                  {1 + (\tau_{\mathrm{PSD}}\,\omega)^2}.
# $$
#
# The field lives on a log-age grid of `n_grid` points, so **the SFH dimension is
# `n_grid`**: the latent vector `sfh_field_xi ∈ ℝ^{n_grid}` is what inference must
# pin down, on top of ~9 physical parameters.
#
# **The observable: emission-line fluxes + broadband photometry.** A single
# galaxy's broadband SED alone cannot recover a *recent, bursty* SFH — the young
# light is degenerate with dust and drowned by the older population (Wang et al.
# 2025). The **Balmer and forbidden lines** trace the last few Myr of star
# formation directly, so we fit **measured line fluxes** (Hα, Hβ, [OIII], [SII],
# [OII]) jointly with 10 broadband bands. Nebular emission is **baked into the
# SSP** (a wNE library), so the lines are *measured off the model spectrum* the
# way a pipeline measures data — no separate emission model.
#
# **Staged, simplest-first.** We start small and only escalate once each rung
# holds:
#
# 1. **Stage 0** — one galaxy at a *small* dimension (`n_grid=16`, D=25): a **MAP**
#    fit (does the mode land on the truth?) then a **dense-mass HMC** posterior
#    (long trajectory; full NUTS confirms the same recovery, ~8× slower). The bar
#    is binary — the injected SFH sits inside the credible bands, so the field
#    prior is *not broken*.
# 2. **Stage 1** — a few catalog galaxies at a modestly higher dimension.
# 3. **Stage 2** — the full 16-galaxy catalog × {NUTS, geoVI, raytracer} × the
#    `n_grid` ladder — a companion script,
#    `scripts/stochastic_sfh_dimension_scaling.py` (one fit per process, OOM-guarded).
#
# **Speed knob.** `approx=WavePrecomp()` puts the photometry on a lookup table.
# The lines are measured on the **exact** field forward — `FeaturePrecomp`'s fast
# line path assumes a non-field SFH and does not apply to the GP field.

# %%
import contextlib
import os
import sys
import time
import warnings

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
import numpy as np


@contextlib.contextmanager
def silence():
    """Mute NIFTy's per-iteration geoVI solver log (writes straight to fd 1/2)."""
    dn = os.open(os.devnull, os.O_WRONLY)
    o1, o2 = os.dup(1), os.dup(2)
    try:
        os.dup2(dn, 1)
        os.dup2(dn, 2)
        yield
    finally:
        os.dup2(o1, 1)
        os.dup2(o2, 2)
        os.close(dn)
        os.close(o1)
        os.close(o2)


# NOTE: this file uses the canonical forward.fit() surface for inference.
# age_gyr=12 at z=0.1 forms ~1% of mass just before the Big Bang (truncated); benign here.
warnings.filterwarnings("ignore", message=r".*before the Big Bang.*")

sys.path.insert(0, ".")
from _plot_style import setup_style

setup_style()
os.makedirs("figures", exist_ok=True)

from pathlib import Path

import tengri
from tengri import (
    FREE,
    Fixed,
    ForwardModel,
    NoiseModel,
    Observation,
    Photometry,
    SEDModel,
    WavePrecomp,
    builders,
    load_ssp_data,
)
from tengri.analysis.plotting import plot_sfh
from tengri.observation import LineFluxData
from tengri.observation.line_measurement import default_line_defs

C_POST, C_TRUTH, C_DATA = "#3a76d9", "0.15", "#c3372a"

# ── Experiment knobs ──────────────────────────────────────────────────────
Z_SPEC = 0.1  # fixed redshift for the mock catalog
N_GRID = 16  # Stage-0 SFH dimension (D = n_grid + 9 physical)
N_CAT = 218  # size of the prior-drawn mock catalog
N_FIT = 16  # the first-N slice we materialize data for / fit
PHOT_SNR = 20.0  # per-band photometric SNR
LINE_SNR = 10.0  # per-line flux SNR

# %% [markdown]
# ## Section 1: Library, observation (photometry + line fluxes), model
#
# A **wNE** SSP grid bakes nebular emission in, so the Balmer/forbidden lines that
# trace recent star formation are modeled for free. The observation is **10
# broadband bands** (GALEX UV → 2MASS NIR) plus **measured emission-line fluxes**.
# `approx=WavePrecomp()` puts the photometry on a lookup table; the lines ride on
# the exact field forward.

# %%
SSP_NAME = "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0"
ssp_path = Path("../data") / f"{SSP_NAME}.h5"
if not ssp_path.exists():
    ssp_path = Path(tengri.download_ssp(SSP_NAME))
ssp_data = load_ssp_data(str(ssp_path))
print(
    f"SSP: {ssp_data.ssp_flux.shape[0]} Z x {ssp_data.ssp_flux.shape[1]} ages "
    f"x {ssp_data.ssp_flux.shape[-1]} lambda  (nebular baked in)"
)

# Photometry: GALEX UV (dust!) + SDSS optical + 2MASS NIR.
PHOT_BANDS = [
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
]
phot = Photometry.from_names(PHOT_BANDS)

# Strong star-forming optical lines. We drop Hγ / [NII], which sit atop stellar
# Balmer absorption and measure near zero for these SFHs (their SNR-scaled errors
# would then dominate the χ² with pure noise).
LINE_NAMES = [
    "Halpha",
    "Hbeta",
    "OIII_5007",
    "OIII_4959",
    "SII_6717",
    "SII_6731",
    "OII_3726",
    "OII_3729",
]
# A template LineFluxData fixes the line identities/wavelengths the model is fit
# against; the observed values are overwritten per galaxy below.
line_template = LineFluxData.from_dict({nm: (1e-16, 1e-17) for nm in LINE_NAMES})
line_defs = default_line_defs(np.asarray(line_template.wavelengths), tuple(line_template.names))

noise_model = NoiseModel(calibration_floor=0.01, student_t_dof=None)


def build(line_fluxes, n_grid=N_GRID):
    """Build the joint phot + line-flux stochastic SFH model.

    Rising DPL backbone + GP field (all free), fixed metallicity, two-component
    Calzetti dust, baked-in (wNE) nebular, fixed z. Photometry on the WavePrecomp
    LUT; lines measured on the exact field forward.
    """
    return SEDModel.build(
        ssp_data=ssp_data,
        observation=Observation(photometry=phot, line_fluxes=line_fluxes, noise=noise_model),
        sfh={"type": ["dpl", "field"], "*": FREE},
        stellar={"met_logzsol": Fixed(-0.3)},
        dust=builders.dust.two_component(defaults=FREE, law_bc="calzetti"),
        neb=builders.neb.ssp(),
        redshift=Fixed(Z_SPEC),
        apply_igm=False,
        n_grid=n_grid,
        approx=WavePrecomp(),
    )


model = build(line_template)
spec = model.spec
fixed_values = spec.get_fixed_values()
print(f"Observation: {phot.n_filters} bands + {len(LINE_NAMES)} line fluxes")
print(
    f"SFH dimension D = {spec.n_free} physical + {spec.n_grid} field = {spec.n_free + spec.n_grid}"
)
print(f"Free physical parameters: {spec.free_params}")

# %% [markdown]
# ## Section 2: A star-forming truth template
#
# The backbone is a **rising** DPL still forming stars at the present day (age =
# 12 Gyr, τ = 13 Gyr just past the present, so we sit on the rising β branch at
# z = 0.1). The GP field (σ_PSD = 0.3 dex, τ_PSD = 200 Myr — the molecular-cloud
# decorrelation time of Tacchella, Forbes & Caplar 2020) rides on top. We anchor
# the *mean* rate to a target present-day SFR by rescaling the mass-normalized
# DPL, which leaves the rising shape intact.

# %%
DPL_TEMPLATE = {
    "sfh_dpl_alpha": 2.0,
    "sfh_dpl_beta": 1.5,
    "sfh_dpl_age_gyr": 12.0,
    "sfh_dpl_tau_gyr": 13.0,
}
MET_FIXED = -0.3

# Star-forming population prior for the mock truths. Physically-typical *modest*
# dust and *moderate* burstiness: the wide fitting prior can draw heavily-dusty
# galaxies whose phot+line SEDs are genuinely age/dust/SFR-degenerate, so a
# recovery test injects a realistic population and fits it with the model's wider
# prior. (The dimension sweep uses the same population.)
DUST_BC_RANGE = (0.1, 0.5)
DUST_DIFF_RANGE = (0.05, 0.35)
SIGMA_RANGE = (0.1, 0.5)
TAU_MYR_RANGE = (50.0, 400.0)
SFR0_RANGE = (3.0, 40.0)


def _present_index(t_gyr):
    """Present epoch = smallest lookback time (convention-robust, tengri #549)."""
    return int(np.argmin(np.asarray(t_gyr)))


def anchor_mass_to_sfr(base_truth, target_sfr0):
    """Return log_total_mass that sets the mean (backbone) present-day SFR to target.

    ``sfr_mean`` is the field-independent DPL backbone, so one rescale hits the
    target exactly while leaving the rising shape and the field bursts intact.
    """
    sfh = model.predict_sfh({**fixed_values, **base_truth})
    p = _present_index(sfh["t_gyr"])
    sfr0 = float(np.asarray(sfh["sfr_mean"])[p])
    if not np.isfinite(sfr0) or sfr0 <= 0.0:
        raise ValueError(f"backbone present-day SFR = {sfr0}; cannot anchor a rising DPL.")
    return float(base_truth["sfh_dpl_log_total_mass"]) + float(np.log10(target_sfr0 / sfr0))


# %% [markdown]
# ## Section 3: A mock catalog drawn from the population prior
#
# We draw `N_CAT` star-forming galaxies: each gets a **fresh field realization**
# (`sfh_field_xi ~ N(0, I)`) and population-prior draws of burstiness (σ_PSD,
# τ_PSD), dust, and mean SFR, with the DPL backbone **fixed star-forming**. The
# data are the *same* observable the fit sees — photometry at SNR ≈ 20 and line
# fluxes measured off the model spectrum at SNR ≈ 10. We materialize (and save)
# the first `N_FIT` line-emitting galaxies as the shared input for the dimension
# sweep.


# %%
def draw_truth(seed):
    """Draw one star-forming mock truth from the population prior."""
    k_xi, k_bc, k_diff, k_sig, k_tau, k_sfr = jax.random.split(jax.random.PRNGKey(seed), 6)
    drawn = spec.sample(k_xi)  # fresh field xi ~ N(0, I)

    def _u(k, lo, hi):
        return float(jax.random.uniform(k, minval=lo, maxval=hi))

    truth = {
        **drawn,
        **{k: jnp.array(v) for k, v in DPL_TEMPLATE.items()},  # fixed SF backbone
        "sfh_dpl_log_total_mass": jnp.array(11.0),
        "met_logzsol": jnp.array(MET_FIXED),
        "dust_tau_bc": jnp.array(_u(k_bc, *DUST_BC_RANGE)),
        "dust_tau_diff": jnp.array(_u(k_diff, *DUST_DIFF_RANGE)),
        "sfh_field_psd_sigma": jnp.array(_u(k_sig, *SIGMA_RANGE)),
        "sfh_field_psd_tau_myr": jnp.array(_u(k_tau, *TAU_MYR_RANGE)),
    }
    target_sfr0 = float(10.0 ** _u(k_sfr, np.log10(SFR0_RANGE[0]), np.log10(SFR0_RANGE[1])))
    truth["sfh_dpl_log_total_mass"] = jnp.array(anchor_mass_to_sfr(truth, target_sfr0))
    return truth


def synthesize(truth, seed):
    """Photometry (SNR 20) + measured line fluxes (SNR 10) for one truth."""
    truth_full = {**fixed_values, **truth}
    mp = model.mock(truth_full, snr=PHOT_SNR, key=jax.random.PRNGKey(seed + 10_000))
    flux_phot, noise_phot = np.asarray(mp.flux_obs), np.asarray(mp.noise)
    lf_true = np.asarray(model.measure_line_fluxes(truth_full, line_defs, fast=False))
    lf_err = np.abs(lf_true) / LINE_SNR
    rng = np.random.default_rng(seed + 20_000)
    lf_obs = lf_true + lf_err * rng.standard_normal(lf_true.shape)
    return flux_phot, noise_phot, lf_obs, lf_err, lf_true


prior_truths = [draw_truth(s) for s in range(N_CAT)]
print(f"Drew {len(prior_truths)} star-forming mock truths from the prior.")

# Keep the first N_FIT prior-drawn galaxies whose Hα is in *emission*. A down-burst
# in the field can suppress recent SF into net Balmer absorption; a star-forming,
# line-emitting catalog filters those rare cases out (the #1152 near-zero-line trap).
catalog, truths, skipped = [], [], 0
for i in range(N_CAT):
    fp, npn, lo, le, lt = synthesize(prior_truths[i], i)
    if lt[LINE_NAMES.index("Halpha")] <= 0:
        skipped += 1
        continue
    catalog.append(dict(flux_phot=fp, noise_phot=npn, line_obs=lo, line_err=le, line_true=lt))
    truths.append(prior_truths[i])
    if len(catalog) == N_FIT:
        break
print(
    f"Kept {len(catalog)} line-emitting galaxies "
    f"(skipped {skipped} with Hα in net absorption) — phot + {len(LINE_NAMES)} lines."
)

# Persist the fitted slice (truths + data) so the sweep script can reuse it.
np.savez(
    "figures/stochastic_catalog.npz",
    line_names=np.array(LINE_NAMES),
    line_waves=np.asarray(line_template.wavelengths),
    **{f"g{i}_{k}": np.asarray(v) for i, g in enumerate(catalog) for k, v in g.items()},
    **{
        f"g{i}_truth_{k}": np.asarray(v)
        for i in range(len(catalog))
        for k, v in {kk: truths[i][kk] for kk in spec.free_params}.items()
    },
)
print("Saved figures/stochastic_catalog.npz")

# %% [markdown]
# ## Section 4: Stage 0 — does it recover? (one galaxy, n_grid=16)
#
# **Does the field SFH recover *at all*?** We answer with a **controlled fiducial**
# — a modest-dust (τ_diff = 0.2), moderate-burstiness (σ_PSD = 0.3) star-forming
# galaxy — where per-galaxy recovery is clean and *validated* (the same truth
# recovers with both a full NUTS run and the HMC below). Recovery is **not uniform
# across the population**: heavy dust or strong bursts let the free field + dust
# soak up an old-vs-young backbone ambiguity — a young+dusty galaxy mimics an
# old+rising one at χ² ≈ 1 — so the detailed SFH *shape* degrades even while the
# mean SFR and burst amplitude stay robust. That per-galaxy fragility, and how it
# scales with dust, burstiness, and dimension D, is what the Section 6 sweep and
# hierarchical inference address.
#
# **MAP** first (multi-start ADAM — does the mode land on the truth?), then a
# **dense-mass HMC** posterior. With only ~18 data points constraining a 25-D
# model, the honest posterior is **wide and correlated** (mass ↔ SFR, τ_BC ↔ recent
# SFR), so exploring it needs a **long trajectory**: `n_leapfrog_steps=100`
# reproduces the full-NUTS result (the injected SFH inside the bands) at a fraction
# of the cost — adaptive NUTS confirms the same recovery but builds very deep trees
# and runs ~8× slower — while a *short* trajectory (L ≈ 25) gets stuck near the mode
# and returns deceptively tight, overconfident bands. The bar is binary: the
# injected SFH inside the credible bands, σ_PSD not railed.

# %%
# Controlled fiducial (see the note above): modest dust, moderate burstiness, mean
# SFR anchored to 20 M⊙/yr, validated to recover cleanly. The prior catalog above
# spans harder cases; the Section 6 sweep quantifies how recovery degrades.
fiducial = {
    **spec.sample(jax.random.PRNGKey(2026)),  # a field realization
    **{k: jnp.array(v) for k, v in DPL_TEMPLATE.items()},
    "met_logzsol": jnp.array(MET_FIXED),
    "dust_tau_bc": jnp.array(0.3),
    "dust_tau_diff": jnp.array(0.2),
    "sfh_field_psd_sigma": jnp.array(0.3),
    "sfh_field_psd_tau_myr": jnp.array(200.0),
    "sfh_dpl_log_total_mass": jnp.array(11.0),
}
fiducial["sfh_dpl_log_total_mass"] = jnp.array(anchor_mass_to_sfr(fiducial, 20.0))
truth = fiducial
truth_full = {**fixed_values, **truth}

fp, npn, lo, le, lt = synthesize(truth, 2026)
assert lt[LINE_NAMES.index("Halpha")] > 0, "fiducial Halpha not in emission"
g = dict(flux_phot=fp, noise_phot=npn, line_obs=lo, line_err=le, line_true=lt)
print(
    f"Fiducial galaxy: sigma_PSD={float(truth['sfh_field_psd_sigma']):.2f}, "
    f"tau_diff={float(truth['dust_tau_diff']):.2f}, mean SFR anchored to 20 Msun/yr"
)

lfd = LineFluxData(
    names=tuple(LINE_NAMES),
    fluxes=jnp.asarray(g["line_obs"]),
    errors=jnp.asarray(g["line_err"]),
    wavelengths=line_template.wavelengths,
)
model_fit = build(lfd, n_grid=N_GRID)
# Inference runs through ForwardModel, the canonical fit surface; the
# observation is inherited from the SED model.
forward = ForwardModel.build(sed=model_fit)

# MAP budget: 10 000 steps, not 2 000. A budget scan on this exact problem gives
# chi2/N = 3.63 at 2 000 steps and 1.21 at 10 000, against 1.41 at the truth --
# so the shorter budget stops well short of the mode and reads as a bad fit.
# 30 000 steps and 16 restarts change nothing (1.22), so 10 000 is the plateau.
t0 = time.perf_counter()
result_map = forward.fit(
    g["flux_phot"],
    g["noise_phot"],
    method="map",
    n_steps=10_000,
    n_restarts=6,
    key=jax.random.PRNGKey(0),
    verbose=False,
)
t_map = time.perf_counter() - t0

t0 = time.perf_counter()
result_hmc = forward.fit(
    g["flux_phot"],
    g["noise_phot"],
    method="mcmc_hmc",
    init_from=result_map,
    # Budget: HMC cost is (n_warmup + n_samples) x n_leapfrog_steps x n_chains
    # objective evaluations, ~2.7 ms each here -> 800+600 at L=100 is ~20 min on a
    # laptop CPU, plus ~6 for the MAP above. The original 400/300 was too short to
    # trust at D=25; 1500/1000 measured 39 min, more than a demonstration warrants.
    # L=100 is deliberate — the default L=10 under-explores this posterior.
    #
    # Convergence here is limited by GEOMETRY, not budget: the non-centering
    # gp_x = cholesky(cov(sigma, tau)) @ xi rotates with psd_tau, so the curvature
    # is position-dependent and one global dense mass matrix cannot represent it.
    # Expect R-hat ~1.1 and a nonzero divergence count however long this runs
    # (#1301). Read the recovery values; do not quote the uncertainties.
    n_warmup=800,
    n_samples=600,
    n_leapfrog_steps=100,
    dense_mass_matrix=True,
    key=jax.random.PRNGKey(1),
    verbose=False,
)
t_hmc = time.perf_counter() - t0
samples = result_hmc.samples
n_total = int(next(iter(samples.values())).shape[0])
print(f"MAP: {t_map:.1f}s")
print(f"HMC: {t_hmc:.1f}s, {n_total} samples")

# Fit quality at MAP (photometry + lines).
map_full = {**fixed_values, **result_map.params}
best_phot = np.asarray(model_fit.predict_photometry(map_full))
best_lf = np.asarray(model_fit.measure_line_fluxes(map_full, line_defs, fast=False))
resid = np.concatenate(
    [
        (g["flux_phot"] - best_phot) / g["noise_phot"],
        (g["line_obs"] - best_lf) / g["line_err"],
    ]
)
chi2_n = float(np.sum(resid**2) / resid.size)
print(f"chi2/N (MAP, phot+lines) = {chi2_n:.2f}")

np.savez(
    "figures/stochastic_samples.npz",
    **{k: np.asarray(v) for k, v in samples.items() if v.ndim == 1},
)

# %% [markdown]
# ### Parameter recovery

# %%
print(f"{'parameter':26s} {'truth':>9s} {'p50':>9s} {'68% CI':>20s}  in?")
print("-" * 74)
for name in spec.free_params:
    if name in samples and samples[name].ndim == 1:
        a = np.asarray(samples[name])
        lo, med, hi = np.percentile(a, [16, 50, 84])
        tr = float(truth[name])
        inside = "IN" if lo <= tr <= hi else "OUT"
        print(f"{name:26s} {tr:9.3f} {med:9.3f}  [{lo:8.3f},{hi:8.3f}]  {inside}")

# %% [markdown]
# ## Section 5: SFH recovery — linear and log-time (the money figure)
#
# Posterior SFH draws through `predict_sfh`. The thick black line is the **mean**
# SFH (DPL backbone); the thin red line is the **actual** SFH with the GP field on
# top. The **log-time** panel is what a linear axis hides: it gives the recent,
# bursty history the resolution to be seen and judged.


# %%
def draw_params(i):
    d = {k: (float(v[i]) if v.ndim == 1 else np.asarray(v[i])) for k, v in samples.items()}
    return {**fixed_values, **d}


sfh_draws = np.array(
    [np.asarray(model_fit.predict_sfh(draw_params(i))["sfr_full"]) for i in range(n_total)]
)
t_gyr = np.asarray(model_fit.predict_sfh(draw_params(0))["t_gyr"])
median_sfh = np.median(sfh_draws, axis=0)
lo68, hi68 = np.percentile(sfh_draws, [16, 84], axis=0)
lo95, hi95 = np.percentile(sfh_draws, [2.5, 97.5], axis=0)
sfh_true = model_fit.predict_sfh(truth_full)
t_true = np.asarray(sfh_true["t_gyr"])
sfr_true = np.asarray(sfh_true["sfr_full"])
sfr_mean_true = np.asarray(sfh_true["sfr_mean"])

# Coverage diagnostics — on the NATIVE log-age grid, not the curves plotted above.
#
# `predict_sfh` defaults to a uniform LINEAR-time resampling for plotting. Scoring
# on it silently reweights the answer: the step is age_max/n_linear = 13.8 Myr, so
# of the 37 samples inside the recent 0.5 Gyr only 2 land below 15 Myr, while 5 of
# the 16 log-age nodes do. Every megayear then counts equally and 15-500 Myr swamps
# the young bins. Coverage over NODES is coverage over the model's actual degrees
# of freedom; coverage over resampled points counts interpolated duplicates.
sfh_nod = model_fit.predict_sfh(truth_full, grid="native")
t_nod = np.asarray(sfh_nod["t_gyr"])
sfr_true_nod = np.asarray(sfh_nod["sfr_full"])
draws_nod = np.array(
    [
        np.asarray(model_fit.predict_sfh(draw_params(i), grid="native")["sfr_full"])
        for i in range(n_total)
    ]
)
lo68n, hi68n = np.percentile(draws_nod, [16, 84], axis=0)
lo95n, hi95n = np.percentile(draws_nod, [2.5, 97.5], axis=0)
cov68 = float(np.mean((sfr_true_nod >= lo68n) & (sfr_true_nod <= hi68n)))
cov95 = float(np.mean((sfr_true_nod >= lo95n) & (sfr_true_nod <= hi95n)))
print(
    f"SFH coverage on the {t_nod.size} native log-age nodes: "
    f"68% band {cov68:.2f},  95% band {cov95:.2f}"
)

# Which nodes did the DATA actually constrain? Shrinkage needs no truth, so this
# is the diagnostic that carries over to real galaxies, where coverage does not
# exist. A node whose posterior is as wide as its prior is a prior draw, and
# quoting it as a measured SFR is exactly the failure #1271 made unavoidable.
prior_draws = np.array(
    [
        np.asarray(
            model_fit.predict_sfh(
                {**fixed_values, **model_fit.spec.sample(jax.random.PRNGKey(9_000 + k))},
                grid="native",
            )["sfr_full"]
        )
        for k in range(200)
    ]
)
_pw = np.diff(
    np.percentile(np.log10(np.clip(prior_draws, 1e-12, None)), [16, 84], axis=0), axis=0
)[0]
_qw = np.diff(np.percentile(np.log10(np.clip(draws_nod, 1e-12, None)), [16, 84], axis=0), axis=0)[
    0
]
shrinkage = 1.0 - _qw / np.clip(_pw, 1e-9, None)
print("shrinkage per node (1 = data-driven, 0 = prior draw):")
for _t, _s in zip(t_nod, shrinkage):
    print(f"   {_t:8.4f} Gyr   {_s:5.2f}{'   <-- prior draw' if _s < 0.3 else ''}")

# %%
fig, (axl, axlog) = plt.subplots(1, 2, figsize=(13, 5))


def _bands(a, xs):
    a.fill_between(xs, lo95, hi95, color=C_POST, alpha=0.12, lw=0, label="95% CI")
    a.fill_between(xs, lo68, hi68, color=C_POST, alpha=0.28, lw=0, label="68% CI")
    a.plot(xs, median_sfh, color=C_POST, lw=1.8, label="posterior median")
    a.plot(xs, sfr_mean_true, color="k", lw=3.0, alpha=0.85, label="mean SFH (DPL)", zorder=9)
    a.plot(xs, sfr_true, color=C_DATA, lw=1.0, label="truth (field on top)", zorder=10)
    a.set_ylim(bottom=0)
    a.set_ylabel(r"SFR [$M_\odot\,\mathrm{yr}^{-1}$]")


_bands(axl, t_gyr)
axl.set_xlim(0, 13.5)
axl.set_xlabel("lookback time [Gyr]")
axl.set_title("Linear time")
axl.legend(fontsize=9, loc="upper right")

_bands(axlog, t_gyr)
axlog.set_xscale("log")
axlog.set_xlim(1e-3, 13.5)  # 1 Myr → 13.5 Gyr; present at left, resolves recent bursts
axlog.set_xlabel("lookback time [Gyr] (log)")
axlog.set_title("Log time — the recent, bursty history")
axlog.text(
    0.03,
    0.03,
    f"$n_{{\\rm grid}}={N_GRID}$ (D={spec.n_free + spec.n_grid})\n"
    f"$\\sigma_{{\\rm PSD}}^{{\\rm true}}=0.3$\nHMC L=100: {t_hmc:.0f}s\n"
    f"95% cov: {cov95:.2f}",
    transform=axlog.transAxes,
    fontsize=9,
    va="bottom",
    ha="left",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.5),
)

fig.suptitle("Stochastic field-SFH recovery (HMC L=100; phot + line fluxes)", y=1.02)
fig.tight_layout()
fig.savefig("figures/sfh_recovery_logt.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# The same posterior via the reusable library helper on a **log** axis — one call,
# `plot_sfh(..., xscale="log")`:

# %%
ax = plot_sfh(model_fit, result_hmc, true_params=truth_full, method="HMC", xscale="log")
ax.figure.savefig("figures/sfh_recovery_plot_sfh_log.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Section 6: Scaling up — Stage 1, methods, and dimension
#
# Stage 0 shows the field SFH recovers at `n_grid=16`. The harder questions —
# does recovery hold as the dimension grows, and can the cheap approximate
# samplers (**geoVI**, the **raytracer**) match NUTS — are a sweep that must run
# **one fit per process** (NUTS warmup at high `n_grid` is memory-heavy). That is
# `scripts/stochastic_sfh_dimension_scaling.py`:
#
# ```bash
# LIMIT_GB=20 scripts/run_with_oom_monitor.sh -- \
#     .venv/bin/python scripts/stochastic_sfh_dimension_scaling.py \
#         --n-grid 16 32 64 128 --n-fit 4 --methods mcmc_hmc vi_nonlinear_fast mcmc_raytrace
# ```
#
# It reuses `figures/stochastic_catalog.npz`, fits each galaxy across the `n_grid`
# ladder and each backend, calls `tengri.clear_cache()` between fits, and writes a
# table of SFH-band coverage, σ_PSD bias, and wall time versus dimension D per
# method. The expectation from prior work: NUTS recovers and calibrates; geoVI is
# fast but its bands narrow (under-cover) as the burstiness rises; the raytracer
# is the intended high-D sampler (`step_size ≈ 0.05` near D ~ 137).

# %% [markdown]
# ## Summary
#
# | Quantity | Stage-0 result (n_grid=16, D=25) |
# |---|---|
# | **Fit quality** | reproduces the data within the noise; χ²/N at the MAP ≈ 0.3–1.0 across realizations, against ≈ 1.4 at the truth |
# | **SFH shape** | recovered where the data carry information — see the per-node **shrinkage** printed above, not the band alone |
# | **Burst amplitude** σ_PSD | **not recovered from photometry alone.** σ̂ compresses toward the prior mean (0.26 → 0.46 as σ_true goes 0.2 → 0.6). Adding 8 optical line fluxes tracks the truth (0.17 → 0.61) |
# | **Emission lines** | cut young-bin (< 15 Myr) SFH error by ~60 % at every σ (13/15, 13/15, 12/14 realizations; sign test p = 0.007 / 0.007 / 0.013) |
# | **Full spectrum vs lines** | *identical* below 15 Myr (24/44, p = 0.65); better at 15 Myr – 1 Gyr (36/44, p = 2.5e-5) — that gain is the **continuum**, not the lines |
# | **Convergence** | R̂ up to 1.16 and ≤ 864 divergences / 8000 draws at this budget. Quote recovery *values*; the *uncertainties* are not yet reliable |
#
# **Takeaway.** The stochastic field SFH is recovered where the data constrain it,
# and the honest question is *which bins those are*. Report on the model's own
# log-age grid (`predict_sfh(..., grid="native")`) — the default linear resampling
# puts 2 of 1000 samples below 15 Myr and silently reweights any residual computed
# on it. For **real** galaxies, where no truth exists, replace coverage with the
# per-node **shrinkage** printed above: a node whose posterior is as wide as its
# prior is a prior draw, and plotting it as a measured SFR is the legitimate form
# of the failure that #1271 made unavoidable. Prefer quoting physically-defined
# integrals (SFR over 0–10 and 0–100 Myr, M★, mass-weighted age) over per-node SFR.
#
# **Provenance.** Every claim above was re-measured after #1271, which fixed two
# halves of a silent failure: the GP field latents reached neither the likelihood
# nor the returned `Posterior.params`, so all pre-fix recovery numbers described
# the *prior*. An earlier version of this summary reported "truth inside the 95 %
# band at all lookback times" and "σ_PSD recovered" — both were artifacts of that
# bug and are retracted.
