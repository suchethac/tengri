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
# # Recovering a bursty star-formation history from joint UV–NIR photometry + an optical spectrum
#
# > ⚠️ **Experimental.** This notebook is a research demonstration. It
# > explores experimental features and may use APIs that change between
# > releases; it sits outside the supported tutorial sequence.
#
# A star-forming galaxy observed two ways at once: **10 broadband fluxes**
# (GALEX FUV/NUV + SDSS *ugriz* + 2MASS *JHKs*) and an **800-pixel optical
# spectrum** (rest 3000–8636 Å, R≈2000). Fitted jointly with a **stochastic**
# SFH model and geometric variational inference (geoVI) on a differentiable
# JAX forward model.
#
# The mean SFH is a **mass-normalized double power law** (Carnall et al. 2018)
# — here a **rising** history still forming stars at the present day —
# modulated by a **Gaussian-process "field"** whose power spectral density is
# a damped random walk:
#
# $$
# \mathrm{SFR}(t) = \mathrm{SFR}_{\mathrm{DPL}}(t)\,\bigl[\log M_\star\bigr]
#                   \times \exp\!\bigl(\mathrm{GP}(t)\bigr),
# \qquad
# P(\omega) = \frac{\sigma_{\mathrm{PSD}}^2\,\tau_{\mathrm{PSD}}}
#                  {1 + (\tau_{\mathrm{PSD}}\,\omega)^2}.
# $$
#
# **Why joint?** An optical spectrum alone leaves a strong **dust ↔ SFR
# degeneracy** ("more dust + more star formation" mimics "less dust + less").
# The **GALEX UV** points pin the attenuation slope and break it — so the
# stellar mass, dust, and the SFH are all recovered, not just the SFH shape.
#
# Nebular emission is **baked into the SSP** (a wNE library — cheap, no Cue
# emulator), dust is Calzetti **attenuation only**, no IGM, redshift fixed at
# z = 0.1. The SSP × wavelength integrals are precomputed via
# `approx=SpectrumPrecomp()`, so each forward evaluation is a cached weighted
# sum. The GP field adds 128 latent dimensions on top of the physical ones.
#
# **Outline:**
#
# 1. Library, joint observation (photometry + spectroscopy), precompute
# 2. Build the stochastic model
# 3. A star-forming mock: matched photometry + spectrum
# 4. MAP + geoVI inference on the joint data
# 5. SFH recovery (the money figure) + last-200-Myr zoom
# 6. Fit quality — photometry and spectrum
# 7. Posteriors for the non-SFH parameters (dust corner) + recovery table

# %%
import contextlib
import time
import warnings

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import os
import sys

import matplotlib.pyplot as plt
import numpy as np


@contextlib.contextmanager
def silence():
    """Mute NIFTy's per-iteration geoVI solver log.

    NIFTy writes its OPTIMIZE_KL / SN / MCG progress straight to the stderr
    file descriptor (not via Python logging or the backend's ``verbose``
    flag), so we redirect fds 1/2 to ``/dev/null`` for the duration of the fit.
    """
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


warnings.filterwarnings("ignore", message=r".*Fitter\(sed_model.*deprecated.*")

sys.path.insert(0, ".")
from _plot_style import COLORS, setup_style  # noqa: E402

setup_style()
os.makedirs("figures", exist_ok=True)

from pathlib import Path  # noqa: E402

import tengri  # noqa: E402
from tengri import (  # noqa: E402
    FREE,
    Fitter,
    Fixed,
    NoiseModel,
    Observation,
    Photometry,
    SEDModel,
    Spectroscopy,
    SpectrumPrecomp,
    builders,
    load_ssp_data,
)

C_POST, C_TRUTH, C_DATA = "#3a76d9", "0.15", "#c3372a"

# %% [markdown]
# ## Section 1: Library, joint observation, precompute
#
# A **wNE** SSP grid bakes nebular emission in, so the Balmer/forbidden lines
# that trace recent star formation are modeled for free. The observation is
# **joint**: 10 broadband filters (GALEX UV → 2MASS NIR) plus an 800-pixel
# R≈2000 optical spectrum. `approx=SpectrumPrecomp()` precomputes the SSP ×
# wavelength integrals once.

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

Z_SPEC = 0.1

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

# Spectroscopy: rest 3000-8636 A at z=0.1, 800 pixels, R=2000.
N_PIX = 800
wave_rest = jnp.logspace(np.log10(3000.0), np.log10(8636.0), N_PIX)
wave_obs = wave_rest * (1.0 + Z_SPEC)
spec_config = Spectroscopy(
    wave_obs=wave_obs,
    resolution=2000.0,
    sigma_lib_kms=70.0,
    lsf_n_bins=16,
    calibration_order=0,
    eline_mode="off",  # lines come from the baked-in SSP
)
noise_model = NoiseModel(calibration_floor=0.01, student_t_dof=None)

obs_joint = Observation(photometry=phot, spectroscopy=spec_config, noise=noise_model)
print(f"Joint observation: {phot.n_filters} photometry bands + {N_PIX} spectral pixels")

# %% [markdown]
# ## Section 2: The stochastic model
#
# `sfh={"type": ["dpl", "field"]}` — a mass-normalized DPL backbone plus the GP
# field. Metallicity is held fixed (an independent prior); the 128-D field
# would otherwise soak up the age-sensitive features and reopen the age/Z/dust
# degeneracy. We build a joint model for the fit, plus photometry-only and
# spectroscopy-only models to synthesize the matched mock.


# %%
def build(observation):
    return SEDModel.build(
        ssp_data=ssp_data,
        observation=observation,
        sfh={"type": ["dpl", "field"], "*": FREE},
        stellar={"met_logzsol": Fixed(-0.3)},
        dust=builders.dust.two_component(defaults=FREE, law_bc="calzetti"),
        neb=builders.neb.ssp(),
        redshift=Fixed(Z_SPEC),
        apply_igm=False,
        n_grid=128,
        approx=SpectrumPrecomp(),
    )


model = build(obs_joint)
model_phot = build(Observation(photometry=phot, noise=noise_model))
model_spec = build(Observation(spectroscopy=spec_config, noise=noise_model))

spec = model.spec
fixed_values = spec.get_fixed_values()
print(f"Free physical parameters: {spec.n_free}")
print(f"GP latent dimensions:     {spec.n_grid}")
print(f"Free parameters: {spec.free_params}")

# %% [markdown]
# ## Section 3: A star-forming mock — matched photometry + spectrum
#
# Truth: a **rising**, still star-forming galaxy. The DPL backbone peaks near
# the present epoch ($\tau = 13$ Gyr, so we sit on the rising $\beta$ branch at
# $z=0.1$); its **mean** rate is anchored to **20 M⊙/yr at lookback 0**, with a
# bursty GP field ($\sigma_{\mathrm{PSD}}=0.3$ dex, $\tau_{\mathrm{PSD}}=200$ Myr
# — the molecular-cloud decorrelation time of Tacchella, Forbes & Caplar 2020)
# modulating that mean; near-solar metallicity, modest dust. The same truth
# synthesizes both the photometry and the spectrum.

# %%
truth = spec.sample(jax.random.PRNGKey(2026))
truth = {
    **truth,
    "sfh_dpl_alpha": jnp.array(2.0),
    "sfh_dpl_beta": jnp.array(1.5),
    # Carnall+2018 DPL in cosmic-time-since-formation (tengri #549): SFR peaks at
    # T = tau, present day is T = age_gyr. Galaxy forms early (age = 12 Gyr, ~0.5
    # Gyr after the Big Bang at z=0.1) and the peak (tau = 13 Gyr) lies just past
    # the present, so the mean SFH is still rising at lookback 0.
    "sfh_dpl_age_gyr": jnp.array(12.0),
    "sfh_dpl_tau_gyr": jnp.array(13.0),
    "sfh_dpl_log_total_mass": jnp.array(11.0),
    "sfh_field_psd_sigma": jnp.array(0.3),
    "sfh_field_psd_tau_myr": jnp.array(200.0),
    "met_logzsol": jnp.array(-0.3),
    "dust_tau_bc": jnp.array(0.3),
    "dust_tau_diff": jnp.array(0.2),
}
truth_full = {**fixed_values, **truth}

# Anchor the *mean* SFH to a target present-day rate. The DPL is mass-normalized,
# so the backbone SFR at lookback 0 scales linearly with total mass — one rescale
# of log_total_mass hits the target exactly while leaving the rising shape intact.
# We measure the backbone with the GP field switched off (psd_sigma -> 0 makes the
# multiplicative field flat), because the field is a zero-mean log-perturbation
# that adds bursts *on top* of the mean. Anchoring the mean — not a single
# field-perturbed draw — is what keeps the demo galaxy genuinely star-forming.
TARGET_SFR0 = 20.0  # Msun/yr at lookback 0 (the mean/backbone rate)
# Index by the TIME AXIS, not a hard-coded array position: the present epoch is
# always the smallest lookback time (t_gyr.argmin()), regardless of whether
# predict_sfh returns cosmic-time or lookback order (the #549 convention fix
# flipped that ordering). This keeps the anchor correct across conventions.
_backbone = {**truth_full, "sfh_field_psd_sigma": jnp.array(1e-4)}
_bb_sfh = model.predict_sfh(_backbone)
_present = int(np.argmin(np.asarray(_bb_sfh["t_gyr"])))
_sfr0 = float(np.asarray(_bb_sfh["sfr_full"])[_present])
if not np.isfinite(_sfr0) or _sfr0 <= 0.0:
    raise ValueError(
        f"Backbone present-day SFR is {_sfr0} — cannot anchor. The DPL is "
        "cosmic-time-since-formation (#549): present-day SFR is nonzero only if "
        "sfh_dpl_age_gyr > 0 and the turnover (tau_gyr) places SF at lookback 0."
    )
truth["sfh_dpl_log_total_mass"] = truth["sfh_dpl_log_total_mass"] + jnp.log10(TARGET_SFR0 / _sfr0)
truth_full = {**fixed_values, **truth}

mock_phot = model_phot.mock(truth_full, snr=20.0, key=jax.random.PRNGKey(1))
mock_spec = model_spec.mock_spectrum(truth_full, wave_obs, snr=20.0, key=jax.random.PRNGKey(2))

flux_phot = np.asarray(mock_phot.flux_obs)
noise_phot = np.asarray(mock_phot.noise)
flux_spec = np.asarray(mock_spec.flux_obs)
noise_spec = np.asarray(mock_spec.noise)
wave_obs_np = np.asarray(wave_obs)

# Joint data vector: photometry first, then spectrum (matches the joint model).
data_joint = np.concatenate([flux_phot, flux_spec])
noise_joint = np.concatenate([noise_phot, noise_spec])
n_phot = phot.n_filters

# Photometric effective wavelengths (for plotting).
wave_eff = np.array(
    [
        np.trapezoid(w * t, w) / np.trapezoid(t, w)
        for w, t in zip(phot.filter_waves, phot.filter_trans)
    ]
)

sfh_true = model.predict_sfh(truth_full)
_p = int(np.argmin(np.asarray(sfh_true["t_gyr"])))  # present = min lookback
print(f"truth recent SFR: {float(np.asarray(sfh_true['sfr_full'])[_p]):.2f} Msun/yr")
print(f"joint data: {data_joint.shape[0]} points ({n_phot} phot + {N_PIX} spec)")

# %%
fig, (axp, axs) = plt.subplots(1, 2, figsize=(13, 4.3), gridspec_kw=dict(width_ratios=[1, 1.6]))
# Photometry (UV -> NIR), log-log.
axp.errorbar(
    wave_eff / 1e4,
    flux_phot,
    yerr=noise_phot,
    fmt="o",
    ms=6,
    color=C_DATA,
    capsize=2,
    label="photometry (SNR≈20)",
)
axp.scatter(
    wave_eff / 1e4,
    np.asarray(mock_phot.flux_true),
    marker="D",
    s=28,
    facecolors="none",
    edgecolors="k",
    linewidths=1.0,
    label="truth",
    zorder=5,
)
axp.set_xscale("log")
axp.set_yscale("log")
axp.set_xlabel(r"observed wavelength [$\mu$m]")
axp.set_ylabel(r"$f_\nu$ [erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$]")
axp.set_title("Broadband (GALEX → 2MASS)")
axp.legend(fontsize=8)
# Spectrum.
axs.errorbar(
    wave_obs_np,
    flux_spec,
    yerr=noise_spec,
    fmt=".",
    ms=2,
    color=C_DATA,
    alpha=0.6,
    elinewidth=0.5,
    label="spectrum (SNR≈20)",
)
axs.set_xlabel(r"observed wavelength [$\mathrm{\AA}$]")
axs.set_title(f"Optical spectrum ({N_PIX} pix, R=2000)")
axs.legend(fontsize=8)
fig.tight_layout()
fig.savefig("figures/sfh_spec_input.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Section 4: MAP + geoVI on the joint data
#
# MAP (ADAM) finds the mode; geoVI (`method="vi"`, NIFTy geometric VI — a
# non-Gaussian, coordinate-transformed approximation) does the full posterior.
# The fit runs inside `silence()` so NIFTy's per-iteration solver log doesn't
# flood the output.

# %%
fitter = Fitter(model, data_joint, noise_joint)

with silence():
    t0 = time.perf_counter()
    result_map = fitter.run(method="map", n_steps=600, key=jax.random.PRNGKey(0), verbose=False)
    t_map = time.perf_counter() - t0

    t0 = time.perf_counter()
    # vi_nonlinear_fast: identical geoVI math to method="vi", but the fast path
    # skips NIFTy's logging/pickling and uses the JIT-engine posterior sampler
    # (~35% faster). 20 iterations is where this posterior has converged —
    # 35 gave the same recovery at ~2x the wall-time.
    result_vi = fitter.run(
        method="vi_nonlinear_fast",
        init_from=result_map,
        n_iterations=20,
        n_samples=3,
        n_posterior_samples=1500,
        key=jax.random.PRNGKey(1),
        verbose=False,
    )
    t_vi = time.perf_counter() - t0

samples = result_vi.samples
n_total = next(iter(samples.values())).shape[0]
print(f"MAP:   {t_map:.1f}s")
print(f"geoVI: {t_vi:.1f}s, {n_total} posterior samples")

np.savez(
    "figures/sfh_spec_samples.npz",
    **{k: np.asarray(v) for k, v in samples.items() if v.ndim == 1},
)

# %% [markdown]
# ## Section 5: SFH recovery (the money figure)
#
# Posterior SFH draws through `predict_sfh`. The thick black line is the **mean**
# SFH (DPL backbone); the thin red line is the **actual** SFH with the GP field
# riding on top — coherent ~200 Myr bursts (Tacchella+2020 molecular-cloud
# decorrelation time). Posterior median + 68/95 % bands in blue. Right panel
# zooms the last 200 Myr.


# %%
def draw_params(i):
    drawn = {k: (float(v[i]) if v.ndim == 1 else np.asarray(v[i])) for k, v in samples.items()}
    return {**fixed_values, **drawn}


sfh_draws = np.array(
    [np.asarray(model.predict_sfh(draw_params(i))["sfr_full"]) for i in range(n_total)]
)
t_gyr = np.asarray(model.predict_sfh(draw_params(0))["t_gyr"])
median_sfh = np.median(sfh_draws, axis=0)
lo_68, hi_68 = np.percentile(sfh_draws, [16, 84], axis=0)
lo_95, hi_95 = np.percentile(sfh_draws, [2.5, 97.5], axis=0)
t_true = np.asarray(sfh_true["t_gyr"])
sfr_true = np.asarray(sfh_true["sfr_full"])
sfr_mean_true = np.asarray(sfh_true["sfr_mean"])

# %%
fig, (ax, ax_zoom) = plt.subplots(
    1, 2, figsize=(12, 5), gridspec_kw=dict(width_ratios=[2.6, 1], wspace=0.05)
)


def _draw(a):
    a.fill_between(t_gyr, lo_95, hi_95, color=C_POST, alpha=0.12, lw=0, label="95% CI")
    a.fill_between(t_gyr, lo_68, hi_68, color=C_POST, alpha=0.28, lw=0, label="68% CI")
    a.plot(t_gyr, median_sfh, color=C_POST, lw=1.8, label="posterior median")
    # Thick mean SFH (DPL backbone) with the thin actual (field-on) truth riding on it.
    a.plot(t_true, sfr_mean_true, color="k", lw=3.0, alpha=0.85, label="mean SFH (DPL)", zorder=9)
    a.plot(t_true, sfr_true, color=C_DATA, lw=1.0, label="truth (field on top)", zorder=10)
    a.set_ylim(bottom=0)


_draw(ax)
ax.set_xlim(0, 13.5)
ax.set_xlabel("lookback time [Gyr]")
ax.set_ylabel(r"SFR [$M_\odot\,\mathrm{yr}^{-1}$]")
ax.set_title("Stochastic SFH recovery (geoVI, joint phot + spec)")
ax.legend(fontsize=9, loc="upper right")
txt = "\n".join(
    [
        r"rising DPL,  $\log M_\star^{\rm true}=11.0$",
        r"$\sigma_{\rm PSD}^{\rm true} = 0.3$ dex",
        r"$\tau_{\rm PSD}^{\rm true} = 200$ Myr (Tacchella+20)",
        f"geoVI: {t_vi:.0f}s",
    ]
)
ax.text(
    0.55,
    0.97,
    txt,
    transform=ax.transAxes,
    fontsize=9,
    va="top",
    ha="left",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.5),
)

_draw(ax_zoom)
ax_zoom.set_xlim(0, 0.2)
zmask = t_gyr <= 0.2
ax_zoom.set_ylim(0, 1.25 * float(np.max(hi_95[zmask])))
ax_zoom.set_xlabel("lookback time [Gyr]")
ax_zoom.set_title("last 200 Myr", fontsize=11)
ax_zoom.yaxis.tick_right()
ax_zoom.set_xticks([0.0, 0.1, 0.2])
fig.tight_layout()
fig.savefig("figures/sfh_spec_recovery.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Section 6: Fit quality — photometry and spectrum
#
# Best-fit (MAP) model over both modalities, with standardized residuals.

# %%
map_full = {**fixed_values, **result_map.params}
best_phot = np.asarray(model.predict_photometry(map_full))
best_spec = np.asarray(model.predict_spectrum(map_full, wave_obs=wave_obs))
resid_phot = (flux_phot - best_phot) / noise_phot
resid_spec = (flux_spec - best_spec) / noise_spec
chi2_n = float((np.sum(resid_phot**2) + np.sum(resid_spec**2)) / data_joint.shape[0])

fig = plt.figure(figsize=(13, 5.5))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.6], height_ratios=[3, 1], hspace=0.05, wspace=0.22)
axp, axs = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
axpr, axsr = fig.add_subplot(gs[1, 0], sharex=axp), fig.add_subplot(gs[1, 1], sharex=axs)

axp.errorbar(
    wave_eff / 1e4,
    flux_phot,
    yerr=noise_phot,
    fmt="o",
    ms=5,
    color=C_DATA,
    capsize=2,
    label="data",
)
axp.plot(wave_eff / 1e4, best_phot, "s", ms=6, color=C_POST, label="best fit", zorder=5)
axp.set_xscale("log")
axp.set_yscale("log")
axp.set_ylabel(r"$f_\nu$")
axp.set_title("Photometry")
axp.legend(fontsize=8)
plt.setp(axp.get_xticklabels(), visible=False)
axpr.axhspan(-1, 1, alpha=0.08, color="0.5")
axpr.axhline(0, color="0.4", lw=0.8)
axpr.plot(wave_eff / 1e4, resid_phot, "o", ms=4, color=C_DATA)
axpr.set_xscale("log")
axpr.set_ylim(-3.5, 3.5)
axpr.set_xlabel(r"obs. wavelength [$\mu$m]")
axpr.set_ylabel(r"$(d-m)/\sigma$")

axs.plot(wave_obs_np, flux_spec, ".", ms=2, color=C_DATA, alpha=0.6, label="data")
axs.plot(wave_obs_np, best_spec, "-", color=C_POST, lw=1.0, label="best fit")
axs.set_title(rf"Spectrum  ($\chi^2/N = {chi2_n:.2f}$, joint)")
axs.legend(fontsize=8)
plt.setp(axs.get_xticklabels(), visible=False)
axsr.axhspan(-1, 1, alpha=0.08, color="0.5")
axsr.axhline(0, color="0.4", lw=0.8)
axsr.plot(wave_obs_np, resid_spec, ".", ms=2, color=C_DATA, alpha=0.6)
axsr.set_ylim(-3.5, 3.5)
axsr.set_xlabel(r"obs. wavelength [$\mathrm{\AA}$]")
fig.savefig("figures/sfh_spec_fit.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"chi2 / N (joint best fit) = {chi2_n:.2f}")

# %% [markdown]
# ## Section 7: Posteriors for the non-SFH parameters
#
# The SFH is summarized by the recovery figure (Section 5); here the joint
# posterior over the **non-SFH** parameters — the two dust optical depths —
# with truth markers in red. Adding the GALEX UV tightens the diffuse-screen
# optical depth, but the **birth-cloud** component ($\tau_{\rm BC}$) remains
# only weakly identified and biased high: with a strongly rising SFH there is
# abundant young, dust-embedded light, and $\tau_{\rm BC}$ trades against the
# recent SFR. This residual dust ↔ SFR degeneracy is exactly the kind of
# nuisance a population-level (hierarchical) prior is meant to absorb.

# %%
# Lightweight manual corner over the parameter *samples* (no model calls /
# KDE) — bounded memory. Free-param samples are saved to
# figures/sfh_spec_samples.npz, so this can be regenerated for any subset.
phys_params = ["dust_tau_bc", "dust_tau_diff"]
labels = [p.replace("dust_", "") for p in phys_params]
S = {p: np.asarray(samples[p]) for p in phys_params}
npar = len(phys_params)
fig_c, axc = plt.subplots(npar, npar, figsize=(2.2 * npar, 2.2 * npar))
for i in range(npar):
    xi, ti = S[phys_params[i]], float(truth[phys_params[i]])
    for j in range(npar):
        ax2 = axc[i, j]
        if j > i:
            ax2.set_visible(False)
            continue
        if i == j:
            ax2.hist(xi, bins=30, color=C_POST, alpha=0.8, edgecolor="none")
            ax2.axvline(ti, color=C_DATA, lw=1.4)
            ax2.set_yticks([])
        else:
            xj, tj = S[phys_params[j]], float(truth[phys_params[j]])
            ax2.hist2d(xj, xi, bins=30, cmap="Blues", cmin=1)
            ax2.plot(tj, ti, "*", ms=13, color=C_DATA, mec="k", mew=0.4)
            ax2.set_ylabel(labels[i], fontsize=10)
        if i == npar - 1:
            ax2.set_xlabel(labels[j], fontsize=10)
        ax2.tick_params(labelsize=7)
fig_c.suptitle(
    "Posteriors — dust optical depths (truth in red; "
    r"$\tau_{\rm BC}$ stays degenerate with recent SFR)",
    y=1.0,
    fontsize=11,
)
fig_c.tight_layout()
fig_c.savefig("figures/sfh_spec_corner.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
print(f"{'parameter':24s} {'truth':>9s} {'p50':>9s} {'offset':>8s} {'68% CI':>20s}")
print("-" * 74)
for name in spec.free_params:
    if name in samples and samples[name].ndim == 1:
        a = np.asarray(samples[name])
        lo, med, hi = np.percentile(a, [16, 50, 84])
        tr = float(truth[name])
        offset = f"{100 * (med - tr) / tr:+.0f}%" if tr != 0 else "—"
        print(f"{name:24s} {tr:9.3f} {med:9.3f} {offset:>8s}  [{lo:8.3f}, {hi:8.3f}]")

# %% [markdown]
# ## Summary
#
# | Quantity | Result |
# |---|---|
# | **Stellar mass** (`log_total_mass`) | Recovered tightly — set by the continuum normalization |
# | **Dust** ($\tau_{\rm diff}$) | Diffuse screen recovered; the GALEX UV anchors it |
# | **Dust** ($\tau_{\rm BC}$) | Biased high & broad — birth-cloud dust stays degenerate with the recent SFR even with UV |
# | **SFH** | Posterior 68/95 % bands contain the rising truth across all lookback times, incl. the last-200-Myr zoom |
# | **Burst amplitude** $\sigma_{\rm PSD}$ | Recovered — a modest, separable burstiness is constrained |
# | **Burst timescale** $\tau_{\rm PSD}$ | Prior-dominated — the data don't constrain it per-galaxy (motivates hierarchical inference) |
# | **Speed** | MAP ~15 s, geoVI ~10 min on a laptop CPU; warm forward ~15 ms |
#
# **Key takeaway.** A joint UV–NIR + optical-spectrum fit recovers the stellar
# mass and the full rising SFH of a star-forming galaxy — the truth sits inside
# calibrated 68/95 % bands at every lookback time, including the last-200-Myr
# zoom. The GALEX UV anchors the diffuse dust screen, but two nuisances stay
# per-galaxy degenerate: the **birth-cloud** optical depth $\tau_{\rm BC}$
# (trades against recent SFR) and the burst **timescale** $\tau_{\rm PSD}$
# (prior-dominated). Both are exactly what **hierarchical population inference**
# is designed to pin down.
#
# *Note on geoVI:* geometric VI is **not** a Gaussian/Laplace approximation —
# it fits a standard normal in a nonlinearly transformed coordinate, so the
# posterior is non-Gaussian in parameter space. For a fully independent
# cross-check, re-run with `method="mcmc_raytrace"`.
