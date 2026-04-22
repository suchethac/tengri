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
# # Emission Line Measurements
#
# _15_emission_line_measurements_
#
# Post-inference emission line analysis in tengri, following FastSpecFit
# (Moustakas et al. 2023, ascl:2308.005) conventions but fully differentiable.
#
# **What this notebook covers:**
#
# 1. **`LineList`** — the ~40-line DESI catalog with `is_strong` / `plot_group` metadata
# 2. **`build_line_mask`** — Boolean pixel mask for continuum fitting windows
# 3. **`compute_line_fluxes`** — Integrated Gaussian fluxes from amplitude posteriors
# 4. **`compute_equivalent_widths`** — Rest-frame EW with $(1+z)$ correction
# 5. **`compute_line_moments`** — Flux-weighted centroid and dispersion (differentiable)
# 6. **Differentiability demo** — `jax.grad` and `jax.vmap` through all three functions
# 7. **BPT diagram** — from injected line fluxes
#
# **Key improvement over FastSpecFit:** the soft Gaussian kernel in `compute_line_moments`
# is differentiable through $\sigma_\mathrm{kms}$, enabling gradient-based inference
# through line widths. FastSpecFit's hard $\pm n_\sigma$ window cannot be JIT-compiled.

# %%
import os
import sys
import warnings

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

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

jax.config.update("jax_enable_x64", True)
warnings.filterwarnings("ignore", category=FutureWarning)

from _plot_style import setup_style, COLORS

setup_style()

# %% [markdown]
# ## 1  LineList — DESI catalog with FastSpecFit metadata
#
# The default 40-line catalog now carries `is_strong` (reliably detected in DESI)
# and `plot_group` (QA panel grouping), both directly from FastSpecFit `emlines.ecsv`.

# %%
from tengri.observation.line_list import LineList

cat = LineList.default_optical()

print(f"Total lines: {cat.n_lines}")
print(f"Independent amplitudes (after doublet constraints): {cat.n_independent}")
print(f"is_strong lines: {sum(cat.is_strong)}")
print()

# Show strong lines and their plot groups
strong = [(n, f"{w:.2f}", g)
          for n, w, s, g in zip(cat.names, cat.wavelengths, cat.is_strong, cat.plot_group)
          if s]
print(f"{'Name':<16} {'λ_rest (Å)':>12}  plot_group")
print("-" * 55)
for name, wave, group in strong:
    print(f"{name:<16} {wave:>12}  {group}")

# %%
# Visualise: all lines coloured by is_strong
fig, ax = plt.subplots(figsize=(10, 2.5))

for i, (name, wave, is_s) in enumerate(zip(cat.names, cat.wavelengths, cat.is_strong)):
    color = COLORS["model"] if is_s else "0.65"
    lw = 1.8 if is_s else 0.8
    ax.axvline(float(wave), color=color, lw=lw, alpha=0.85)

# Legend proxies
from matplotlib.lines import Line2D
ax.legend(
    handles=[
        Line2D([0], [0], color=COLORS["model"], lw=1.8, label="is_strong (DESI-reliable)"),
        Line2D([0], [0], color="0.65", lw=0.8, label="other"),
    ],
    loc="upper right",
    fontsize=10,
)

# Annotate a few key lines
for name, wave in [
    (r"Ly$\alpha$", 1215.67),
    ("[OII]", 3727.5),
    (r"H$\beta$", 4862.68),
    ("[OIII]", 5008.24),
    (r"H$\alpha$", 6564.61),
    ("[NII]", 6585.28),
    ("[SII]", 6725),
]:
    ax.text(wave, 0.92, name, transform=ax.get_xaxis_transform(),
            fontsize=7, ha="center", va="top", color="0.3", rotation=90)

ax.set_xlabel(r"Rest-frame wavelength ($\AA$)")
ax.set_yticks([])
ax.set_title("Default ~40-line catalog  (red = is_strong)", fontsize=12)
ax.set_xlim(1100, 9700)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2  build_line_mask — pre-fit continuum window mask
#
# `build_line_mask` produces a boolean array `True` at pixels contaminated by
# emission lines.  Feed the **complement** (`~mask`) to your continuum estimator.
# This is a NumPy-only preprocessing step — it runs once before fitting.

# %%
from tengri.observation.line_mask import build_line_mask

# Simulate a DESI-like observed spectrum at z = 0.15
z = 0.15
wave_obs = np.linspace(3600., 9800., 3000)   # Å, observed frame

# Use the full strong-line set
strong_rest = np.array([float(w) for w, s in zip(cat.wavelengths, cat.is_strong) if s])

mask = build_line_mask(
    wave_obs,
    strong_rest,
    redshift=z,
    line_sigmas_kms=150.0,   # fiducial velocity dispersion
    n_sigma=2.5,
    min_sigma_kms=50.0,
)

print(f"Masked pixels : {mask.sum()} / {len(mask)}  ({mask.mean():.1%})")

# %%
# Visualise mask on a mock continuum + lines spectrum
rng = np.random.default_rng(42)
continuum = 3e-17 * (wave_obs / 5500.) ** (-0.5)
noise_spec = rng.normal(0., 2e-18, size=len(wave_obs))
mock_spec = continuum + noise_spec

# Inject a handful of Gaussian emission lines
_C_KMS = 2.99792458e5
for rest_wave, amp in [(3727.0, 8e-17), (4862.68, 6e-17), (5008.24, 1.5e-16),
                       (6564.61, 3e-16), (6585.28, 1e-16)]:
    obs_w = rest_wave * (1 + z)
    sigma_ang = 150. / _C_KMS * obs_w
    mock_spec = mock_spec + amp * np.exp(-0.5 * ((wave_obs - obs_w) / sigma_ang) ** 2)

fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True,
                         gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05})

axes[0].plot(wave_obs, mock_spec, color="0.4", lw=0.5, label="mock spectrum")
axes[0].fill_between(wave_obs, 0, mock_spec, where=mask,
                     color=COLORS["model"], alpha=0.25, label="masked (emission)")
axes[0].fill_between(wave_obs, 0, mock_spec, where=~mask,
                     color=COLORS["rt"], alpha=0.12, label="continuum window")
axes[0].set_ylabel(r"$f_\lambda$ (erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$)")
axes[0].legend(fontsize=9, loc="upper right")
axes[0].set_title(f"build_line_mask  (z = {z},  σ = 150 km/s,  n_σ = 2.5)", fontsize=12)

axes[1].fill_between(wave_obs, 0, mask.astype(float), color=COLORS["model"], alpha=0.6)
axes[1].set_yticks([0, 1])
axes[1].set_yticklabels(["cont.", "masked"])
axes[1].set_xlabel(r"Observed wavelength ($\AA$)")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3  compute_line_fluxes — Gaussian flux from amplitude posteriors
#
# Given per-line Gaussian **amplitudes** (what the SED model fits), convert to
# integrated fluxes using $F_i = A_i \sqrt{2\pi}\,\sigma_{\rm ang}$, where
# $\sigma_{\rm ang}$ combines intrinsic velocity broadening and the LSF in quadrature.

# %%
from tengri.analysis.diagnostics.lines import (
    compute_line_fluxes,
    compute_equivalent_widths,
    compute_line_moments,
)

# Select the BPT + SII lines for illustration
bpt_names = ["OII_3726", "OII_3729", "Hgamma", "Hbeta",
             "OIII_4959", "OIII_5007", "NII_6548", "Halpha", "NII_6584",
             "SII_6717", "SII_6731"]
bpt = cat.select(names=bpt_names)

print(f"Selected {bpt.n_lines} lines for flux measurement")

# Mock posterior median amplitudes (erg/s/cm²/Å) — star-forming galaxy ratios
_amps = {
    "OII_3726":  8.0e-18, "OII_3729":  9.0e-18,
    "Hgamma":    3.5e-18, "Hbeta":     1.0e-17,
    "OIII_4959": 5.0e-18, "OIII_5007": 1.5e-17,
    "NII_6548":  6.0e-18, "Halpha":    3.0e-17, "NII_6584": 1.8e-17,
    "SII_6717":  5.0e-18, "SII_6731":  4.0e-18,
}
amplitudes = jnp.array([_amps[n] for n in bpt.names])
rest_waves = bpt.wavelengths

sigma_kms = 80.0          # intrinsic line width [km/s]
R_spec = 2500.0           # DESI-like spectral resolution

fluxes = compute_line_fluxes(amplitudes, rest_waves, z, sigma_kms, R_spec)

print(f"\n{'Line':<14} {'Amplitude':>14}  {'Flux':>16}")
print("-" * 48)
for name, amp, flux in zip(bpt.names, amplitudes, fluxes):
    print(f"{name:<14} {float(amp):.2e}  {float(flux):.2e}")

# %%
# Visualise: reconstructed line profiles over a mock continuum
fig, ax = plt.subplots(figsize=(10, 4))

wave_fine = np.linspace(3500., 7200., 8000)
continuum_fine = 3e-17 * (wave_fine / 5500.) ** (-0.5)

# Build model spectrum from amplitudes
model_spec = continuum_fine.copy()
for name, amp, rest_w in zip(bpt.names, amplitudes, bpt.wavelengths):
    obs_w = float(rest_w) * (1 + z)
    sig_ang = sigma_kms / _C_KMS * obs_w
    model_spec = model_spec + float(amp) * np.exp(
        -0.5 * ((wave_fine - obs_w) / sig_ang) ** 2
    )

ax.plot(wave_fine, continuum_fine, color="0.6", lw=1.0, ls="--", label="continuum")
ax.plot(wave_fine, model_spec, color=COLORS["model"], lw=1.0, label="continuum + lines")

# Mark each line
for name, rest_w, is_s in zip(bpt.names, bpt.wavelengths, bpt.is_strong):
    obs_w = float(rest_w) * (1 + z)
    if 3500 < obs_w < 7200:
        ax.axvline(obs_w, color="0.75", lw=0.6, ls=":")
        ax.text(obs_w, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 5e-17,
                name, fontsize=6, ha="center", va="bottom", color="0.4", rotation=80)

ax.set_xlabel(r"Observed wavelength ($\AA$)")
ax.set_ylabel(r"$f_\lambda$ (erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$)")
ax.set_title(f"Mock star-forming galaxy  (z = {z},  σ = {sigma_kms} km/s)", fontsize=12)
ax.legend(fontsize=10)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4  compute_equivalent_widths — rest-frame EW
#
# EW is the most common observational line strength metric. The $(1+z)$ factor
# converts the observed-frame flux ratio to a rest-frame interval width.
# Positive EW = emission; zero when continuum $\leq 0$.

# %%
# Mock continuum sampled at each line center (e.g., from interpolating the posterior SED)
continuum_at_lines = jnp.array([
    float(3e-17 * (float(w) * (1 + z) / 5500.) ** (-0.5))
    for w in bpt.wavelengths
])

ews = compute_equivalent_widths(fluxes, continuum_at_lines, z)

print(f"\n{'Line':<14} {'Flux (erg/s/cm²)':>18}  {'Cont (erg/s/cm²/Å)':>20}  {'EW_rest (Å)':>12}")
print("-" * 70)
for name, flux, cont, ew in zip(bpt.names, fluxes, continuum_at_lines, ews):
    print(f"{name:<14} {float(flux):.2e}  {float(cont):.2e}  {float(ew):>12.1f}")

# %%
# Bar chart of EWs — sorted by rest-frame wavelength
fig, ax = plt.subplots(figsize=(9, 3.5))

x = np.arange(bpt.n_lines)
colors = [COLORS["model"] if s else COLORS["rt"] for s in bpt.is_strong]
bars = ax.bar(x, [float(e) for e in ews], color=colors, alpha=0.85, width=0.7)

ax.set_xticks(x)
ax.set_xticklabels(bpt.names, rotation=40, ha="right", fontsize=9)
ax.set_ylabel(r"Rest-frame EW$_0$ ($\AA$)")
ax.set_title("Emission line equivalent widths", fontsize=12)
ax.axhline(0, color="0.4", lw=0.7)

from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(color=COLORS["model"], alpha=0.85, label="is_strong"),
    Patch(color=COLORS["rt"], alpha=0.85, label="other"),
], fontsize=9)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5  compute_line_moments — velocity centroid and dispersion
#
# FastSpecFit uses a hard $\pm n_\sigma$ pixel window to compute these moments,
# which requires dynamic boolean indexing — incompatible with JAX JIT.
#
# Tengri's soft Gaussian kernel keeps the full pixel array at a fixed shape:
# $$w_k = \exp\!\left(-\frac{v_k^2}{2\,\sigma_w^2}\right)$$
# making the moments differentiable through $\sigma_\mathrm{kms}$ for use
# inside a loss function.

# %%
# Inject a Gaussian emission line for Hα with a known velocity offset
sigma_true_kms = 95.0        # true intrinsic width [km/s]
v_offset_true_kms = +45.0    # galaxy has a +45 km/s redshift relative to line center

halpha_obs = 6564.61 * (1 + z)
halpha_obs_shifted = halpha_obs * (1 + v_offset_true_kms / _C_KMS)

wave_ha = np.linspace(halpha_obs - 60., halpha_obs + 60., 600)
sigma_ang = sigma_true_kms / _C_KMS * halpha_obs_shifted
amp_true = 2.5e-16

rng2 = np.random.default_rng(7)
flux_ha = (
    amp_true * np.exp(-0.5 * ((wave_ha - halpha_obs_shifted) / sigma_ang) ** 2)
    + rng2.normal(0., 3e-18, size=len(wave_ha))
)
ivar_ha = np.full_like(flux_ha, 1. / (3e-18) ** 2)

# Continuum-subtract (here: perfectly known mock continuum)
cont_ha = np.full_like(wave_ha, 3.5e-18)
residual_ha = flux_ha - cont_ha

# Measure moments with a wide soft window (σ_window >> σ_signal to minimise bias)
sigma_window_kms = 800.0
v_cent, sigma_int = compute_line_moments(
    jnp.array(wave_ha), jnp.array(residual_ha), jnp.array(ivar_ha),
    line_obs_wave=halpha_obs,
    sigma_kms=sigma_window_kms,
)

print(f"True   v_offset = {v_offset_true_kms:+.1f} km/s,  σ_int = {sigma_true_kms:.1f} km/s")
print(f"Measured centroid  = {float(v_cent):+.1f} km/s")
print(f"Measured σ_int     = {float(sigma_int):.1f} km/s")
print(f"  (kernel bias: σ_meas = σ_true × σ_w / √(σ_true² + σ_w²)"
      f" = {sigma_true_kms * sigma_window_kms / np.sqrt(sigma_true_kms**2 + sigma_window_kms**2):.1f} km/s)")

# %%
# Plot the moment measurement
v_axis = _C_KMS * (wave_ha - halpha_obs) / halpha_obs  # km/s relative to line center

fig, ax = plt.subplots(figsize=(7, 3.5))

ax.plot(v_axis, residual_ha / amp_true, color="0.5", lw=0.6, label="residual (S/N~83)")
ax.plot(v_axis, np.exp(-0.5 * ((v_axis - v_offset_true_kms) / sigma_true_kms) ** 2),
        color=COLORS["truth"], lw=1.5, ls="--", label=f"true profile (σ = {sigma_true_kms} km/s)")

# Mark centroid and ±σ window
ax.axvline(float(v_cent), color=COLORS["model"], lw=1.5,
           label=f"centroid = {float(v_cent):+.1f} km/s")
for sign in [-1, +1]:
    ax.axvline(float(v_cent) + sign * float(sigma_int),
               color=COLORS["model"], lw=0.8, ls=":")
ax.fill_betweenx([0, 1], float(v_cent) - float(sigma_int),
                 float(v_cent) + float(sigma_int),
                 color=COLORS["model"], alpha=0.10, label=f"σ_int = {float(sigma_int):.1f} km/s")

ax.set_xlabel(r"$v$ relative to $\lambda_0$ (km s$^{-1}$)")
ax.set_ylabel(r"$f$ / peak amplitude")
ax.set_title(r"compute_line_moments  —  H$\alpha$ velocity moments", fontsize=12)
ax.legend(fontsize=9, loc="upper left")
ax.set_xlim(-500, 500)
ax.set_ylim(-0.15, 1.2)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6  Differentiability — jax.grad and jax.vmap
#
# All three measurement functions are JIT-compatible and differentiable, so they
# can sit **inside** a loss function.  FastSpecFit cannot do this.

# %%
# 6a: jax.grad through compute_line_fluxes
def total_ha_flux(amps):
    """Sum of fluxes — toy objective that could appear in a likelihood."""
    return jnp.sum(compute_line_fluxes(amps, rest_waves, z, sigma_kms, R_spec))

grad_fn = jax.jit(jax.grad(total_ha_flux))
grads = grad_fn(amplitudes)

print("∂(total_flux)/∂(amplitude_i)  — should equal sqrt(2π)·σ_ang for each line:")
for name, g in zip(bpt.names, grads):
    print(f"  {name:<14}  {float(g):.4e}")

# %%
# 6b: jax.vmap compute_line_moments over multiple lines simultaneously
halpha_idx = list(bpt.names).index("Halpha")
nii6548_idx = list(bpt.names).index("NII_6548")
nii6584_idx = list(bpt.names).index("NII_6584")

# Build a mock residual spectrum covering the Hα+NII panel
wave_panel = np.linspace(6530. * (1 + z), 6610. * (1 + z), 800)
flux_panel = np.zeros_like(wave_panel)

for name, rest_w, amp_i in [
    ("Halpha",  6564.61, 2.5e-16),
    ("NII_6548", 6549.86, 5e-17),
    ("NII_6584", 6585.28, 1.5e-16),
]:
    obs_w = rest_w * (1 + z)
    sig_i = 95. / _C_KMS * obs_w
    flux_panel += amp_i * np.exp(-0.5 * ((wave_panel - obs_w) / sig_i) ** 2)

flux_panel += rng2.normal(0., 3e-18, size=len(wave_panel))
ivar_panel = np.full_like(flux_panel, 1. / (3e-18) ** 2)

# vmap over the three line centers
panel_centers = jnp.array([w * (1 + z) for w in [6564.61, 6549.86, 6585.28]])
panel_sigma = 800.0  # wide window

vmapped_moments = jax.vmap(
    lambda lam: compute_line_moments(
        jnp.array(wave_panel), jnp.array(flux_panel), jnp.array(ivar_panel),
        line_obs_wave=lam, sigma_kms=panel_sigma,
    )
)
v_cents, sigma_ints = vmapped_moments(panel_centers)

print("\nvmap over Hα+NII panel:")
for name, vc, si in zip(["Halpha", "NII_6548", "NII_6584"], v_cents, sigma_ints):
    print(f"  {name:<12}  v_cent = {float(vc):+6.1f} km/s,  σ_int = {float(si):.1f} km/s")

# %%
# 6c: Demonstrate grad through compute_line_moments
def moment_loss(wave, flux, ivar, lam0, sigma_w):
    """Toy loss: push v_centroid toward zero (used in velocity-field fitting)."""
    v_cent, sigma_int = compute_line_moments(wave, flux, ivar, lam0, sigma_w)
    return v_cent ** 2

grad_sigma = jax.jit(jax.grad(moment_loss, argnums=4))
dL_dsigma = grad_sigma(
    jnp.array(wave_panel), jnp.array(flux_panel), jnp.array(ivar_panel),
    panel_centers[0], panel_sigma
)
print(f"\n∂L/∂σ_window = {float(dL_dsigma):.4e}  (finite → differentiable through σ_kms ✓)")

# %% [markdown]
# ## 7  BPT diagram — star-forming vs AGN classification
#
# Construct the Baldwin-Phillips-Terlevich (BPT) diagnostic ratios directly
# from the measured fluxes.  The [NII]/Hα and [OIII]/Hβ ratios are the most
# robust, requiring only four strong lines.

# %%
# Build a grid of mock galaxies with varying ionisation parameter
# (star-forming sequence: log U increases → higher OIII/Hbeta, lower NII/Halpha)
n_gal = 80
rng3 = np.random.default_rng(1)

# Star-forming sequence parameterised by ionisation parameter log_U
log_U = np.linspace(-3.8, -2.0, n_gal)

# Approximate BPT track (Kewley+2001 / Kauffmann+2003 empirical)
log_nii_ha_sf = -0.3 + 0.15 * (log_U + 3.0) + rng3.normal(0, 0.08, n_gal)
log_oiii_hb_sf = 0.6 * (log_U + 3.0) - 0.5 + rng3.normal(0, 0.10, n_gal)

# AGN sequence (high OIII/Hβ at all NII/Hα)
log_nii_ha_agn = np.linspace(-0.8, 0.4, 20)
log_oiii_hb_agn = 0.73 / (log_nii_ha_agn - 0.32) + 1.30 + rng3.normal(0, 0.08, 20)

# Kewley+2001 maximum starburst demarcation (theoretical upper envelope)
x_kewley = np.linspace(-2.0, 0.35, 200)
y_kewley = 0.61 / (x_kewley - 0.47) + 1.19

# Kauffmann+2003 empirical SF/AGN dividing line
x_kauff = np.linspace(-2.0, 0.0, 200)
y_kauff = 0.61 / (x_kauff - 0.05) + 1.30

# Where do our injected amplitudes sit?
flux_ha_val = float(fluxes[list(bpt.names).index("Halpha")])
flux_hb_val = float(fluxes[list(bpt.names).index("Hbeta")])
flux_oiii_val = float(fluxes[list(bpt.names).index("OIII_5007")])
flux_nii_val = float(fluxes[list(bpt.names).index("NII_6584")])

log_nii_ha_inj = np.log10(flux_nii_val / flux_ha_val)
log_oiii_hb_inj = np.log10(flux_oiii_val / flux_hb_val)

print(f"Injected galaxy BPT position:")
print(f"  log([NII]/Hα) = {log_nii_ha_inj:.3f}")
print(f"  log([OIII]/Hβ) = {log_oiii_hb_inj:.3f}")

# %%
fig, ax = plt.subplots(figsize=(6, 5.5))

ax.scatter(log_nii_ha_sf, log_oiii_hb_sf, s=18, color=COLORS["rt"],
           alpha=0.7, label="Mock SF galaxies", zorder=3)
ax.scatter(log_nii_ha_agn, log_oiii_hb_agn, s=18, color=COLORS["model"],
           alpha=0.7, label="Mock AGN", zorder=3, marker="s")

# Demarcation lines
ax.plot(x_kewley, y_kewley, "k--", lw=1.2, label="Kewley+2001 (max SB)")
ax.plot(x_kauff, y_kauff, "k:", lw=1.2, label="Kauffmann+2003 (empirical)")

# Our injected galaxy
ax.scatter([log_nii_ha_inj], [log_oiii_hb_inj], s=120, color=COLORS["truth"],
           zorder=6, marker="*", label="This notebook's galaxy")

# Shaded regions
ax.fill_between(x_kauff, y_kauff, -2.0, alpha=0.05, color=COLORS["rt"])
ax.fill_between(x_kewley, y_kewley, 1.5, alpha=0.05, color=COLORS["model"])

ax.text(-1.6, -1.0, "Star-forming", fontsize=10, color=COLORS["rt"], style="italic")
ax.text(-0.0, 0.9, "AGN", fontsize=10, color=COLORS["model"], style="italic")

ax.set_xlabel(r"log([N II] $\lambda$6584 / H$\alpha$)")
ax.set_ylabel(r"log([O III] $\lambda$5007 / H$\beta$)")
ax.set_title("BPT diagram from compute_line_fluxes", fontsize=12)
ax.legend(fontsize=8.5, loc="lower left")
ax.set_xlim(-2.2, 0.7)
ax.set_ylim(-1.4, 1.5)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8  Doublet constraints — build_constraint_matrix
#
# FastSpecFit enforces fixed flux ratios for physically constrained doublets
# (e.g., [OIII] 5007/4959 = 2.98 from transition probabilities).  Tengri
# encodes these as `DoubletConstraint` objects and exposes them through
# `build_constraint_matrix()`, which maps $n_\mathrm{independent}$ free
# amplitudes to all $n_\mathrm{lines}$ line amplitudes via
# $\mathbf{a} = C\,\mathbf{a}_\mathrm{ind}$.

# %%
# Show doublet constraints in the full catalog
print("Doublet constraints in default_optical():")
print(f"{'Primary':<14} {'Secondary':<14} {'Ratio':>8}  (primary/secondary)")
print("-" * 45)
for dc in cat.doublets:
    pname = cat.names[dc.primary_idx]
    sname = cat.names[dc.secondary_idx]
    print(f"{pname:<14} {sname:<14} {dc.ratio:>8.3f}")

print(f"\nFull catalog: {cat.n_lines} lines, {len(cat.doublets)} constrained doublets")
print(f"→  {cat.n_independent} independent amplitude parameters")

# %%
# Demonstrate the constraint matrix on the BPT subset
C = bpt.build_constraint_matrix()
print(f"\nConstraint matrix shape: {C.shape}  ({bpt.n_lines} lines × {bpt.n_independent} free amps)")

# Verify OIII ratio is encoded correctly
i_5007 = list(bpt.names).index("OIII_5007")
i_4959 = list(bpt.names).index("OIII_4959")
i_nii84 = list(bpt.names).index("NII_6584")
i_nii48 = list(bpt.names).index("NII_6548")

# Find which column drives each primary
col_oiii = int(jnp.argmax(jnp.abs(C[i_5007])))
col_nii = int(jnp.argmax(jnp.abs(C[i_nii84])))

print(f"\nOIII 5007 row:  C[5007, col] = {float(C[i_5007, col_oiii]):.4f}  (primary = 1.0)")
print(f"OIII 4959 row:  C[4959, col] = {float(C[i_4959, col_oiii]):.4f}  (= 1/2.98 = {1/2.98:.4f})")
print(f"NII  6584 row:  C[6584, col] = {float(C[i_nii84, col_nii]):.4f}  (primary = 1.0)")
print(f"NII  6548 row:  C[6548, col] = {float(C[i_nii48, col_nii]):.4f}  (= 1/2.94 = {1/2.94:.4f})")

# %%
# Visualise the constraint matrix as a heatmap
fig, ax = plt.subplots(figsize=(10, 4))

C_np = np.array(C)
im = ax.imshow(C_np.T, aspect="auto", cmap="Blues", vmin=0, vmax=1)

ax.set_yticks(range(bpt.n_independent))
# Label independent columns with their primary line names
constrained = {dc.secondary_idx for dc in bpt.doublets}
ind_names = [bpt.names[i] for i in range(bpt.n_lines) if i not in constrained]
ax.set_yticklabels(ind_names, fontsize=8)
ax.set_xticks(range(bpt.n_lines))
ax.set_xticklabels(bpt.names, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Independent amplitude")
ax.set_xlabel("Line")
ax.set_title("Constraint matrix  C  (flux = C @ a_independent)", fontsize=12)
plt.colorbar(im, ax=ax, label="coefficient")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 9  plot_group QA panels — FastSpecFit-style diagnostic layout
#
# FastSpecFit organises its QA spectral panels by `plotgroup`: lines in the same
# group are shown together.  `LineList.plot_group` carries this metadata directly
# from `emlines.ecsv`.  Here we replicate the panel layout.

# %%
# Group lines by plot_group and sort groups by median wavelength
from collections import defaultdict

groups = defaultdict(list)
for name, wave, group in zip(cat.names, cat.wavelengths, cat.plot_group):
    groups[group].append((name, float(wave)))

# Sort groups by the bluest line in each group
sorted_groups = sorted(groups.items(), key=lambda kv: min(w for _, w in kv[1]))

print(f"{'plot_group':<40}  lines")
print("-" * 70)
for gname, members in sorted_groups:
    names_str = ", ".join(f"{n}({w:.0f}Å)" for n, w in members)
    print(f"{gname:<40}  {names_str}")

# %%
# Mock a multi-panel QA plot matching FastSpecFit's panel layout
# Show the 8 most important groups (those containing is_strong lines)
strong_set = set(n for n, s in zip(cat.names, cat.is_strong) if s)
important_groups = [
    (g, members) for g, members in sorted_groups
    if any(n in strong_set for n, _ in members)
][:8]

fig = plt.figure(figsize=(14, 7))
fig.suptitle(
    "FastSpecFit-style QA panel layout  (plot_group)  —  z = 0.15",
    fontsize=13, y=1.01
)

n_panels = len(important_groups)
ncols = 4
nrows = (n_panels + ncols - 1) // ncols

for idx, (gname, members) in enumerate(important_groups):
    ax = fig.add_subplot(nrows, ncols, idx + 1)

    # Window centred on the group's median observed wavelength
    obs_waves = [w * (1 + z) for _, w in members]
    w_cen = np.mean(obs_waves)
    half_win = max(60., 2.5 * (max(obs_waves) - min(obs_waves)) + 40.)
    w_lo, w_hi = w_cen - half_win, w_cen + half_win

    wave_win = np.linspace(w_lo, w_hi, 400)
    cont_win = 3e-17 * (wave_win / 5500.) ** (-0.5)
    spec_win = cont_win.copy()

    # Inject lines belonging to this group
    amp_map = {
        "Halpha": 2.5e-16, "NII_6584": 1.5e-16, "NII_6548": 5e-17,
        "Hbeta": 1.0e-16, "OIII_5007": 1.5e-16, "OIII_4959": 5e-17,
        "OII_3726": 8e-17, "OII_3729": 9e-17,
        "SII_6717": 5e-17, "SII_6731": 4e-17,
        "Hgamma": 3.5e-17, "Lya": 3e-16, "NV_1240": 5e-17,
        "CIV_1549": 8e-17, "MgII_2796": 1e-16, "MgII_2803": 1e-16,
    }
    for lname, rest_w in members:
        obs_w = rest_w * (1 + z)
        amp_i = amp_map.get(lname, 2e-17)
        sig_i = 95. / _C_KMS * obs_w
        spec_win += amp_i * np.exp(-0.5 * ((wave_win - obs_w) / sig_i) ** 2)

    spec_win += rng.normal(0., 2e-18, size=len(wave_win))

    ax.plot(wave_win, spec_win / 1e-17, color="0.4", lw=0.6)
    ax.plot(wave_win, cont_win / 1e-17, color=COLORS["rt"], lw=0.8, ls="--")

    # Mark each line in the group
    for lname, rest_w in members:
        obs_w = rest_w * (1 + z)
        color_l = COLORS["model"] if lname in strong_set else "0.6"
        ax.axvline(obs_w, color=color_l, lw=0.9, ls=":")
        ax.text(obs_w, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 30,
                lname, fontsize=5.5, ha="center", va="bottom",
                color=color_l, rotation=75)

    ax.set_xlim(w_lo, w_hi)
    ax.set_title(gname, fontsize=7.5, pad=2)
    ax.set_xlabel(r"$\lambda_\mathrm{obs}$ ($\AA$)", fontsize=7)
    ax.set_ylabel(r"$f$ (10$^{-17}$)", fontsize=7)
    ax.tick_params(labelsize=6)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 10  Broad + narrow component (is_broad_candidate)
#
# FastSpecFit fits a broad AGN component alongside the narrow line for lines
# flagged as `isbroad`.  Tengri carries this as `is_broad_candidate`, which
# marks H Balmer lines + UV broad lines (CIV, MgII, CIII]).
# A broader velocity component ($\sigma \sim 2000$ km/s) sits beneath the
# narrow core.

# %%
broad_lines = [(n, float(w)) for n, w, b in zip(cat.names, cat.wavelengths, cat.is_broad_candidate) if b]
print("is_broad_candidate lines:")
for name, wave in broad_lines:
    print(f"  {name:<14}  {wave:.2f} Å")

# %%
# Demonstrate narrow + broad decomposition for Hα
sigma_narrow = 100.   # km/s — galaxy narrow line
sigma_broad = 2500.   # km/s — AGN broad component
amp_narrow = 2.0e-16
amp_broad = 8.0e-17   # ~ 30% contribution

halpha_rest = 6564.61
halpha_obs_val = halpha_rest * (1 + z)

wave_ha2 = np.linspace(halpha_obs_val - 250., halpha_obs_val + 250., 1000)
v_ha2 = _C_KMS * (wave_ha2 - halpha_obs_val) / halpha_obs_val

sig_narrow_ang = sigma_narrow / _C_KMS * halpha_obs_val
sig_broad_ang = sigma_broad / _C_KMS * halpha_obs_val

narrow_comp = amp_narrow * np.exp(-0.5 * (wave_ha2 - halpha_obs_val) ** 2 / sig_narrow_ang ** 2)
broad_comp = amp_broad * np.exp(-0.5 * (wave_ha2 - halpha_obs_val) ** 2 / sig_broad_ang ** 2)
total = narrow_comp + broad_comp + rng.normal(0, 2e-18, len(wave_ha2))

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(v_ha2, total / 1e-16, color="0.4", lw=0.7, label="total (+ noise)")
ax.fill_between(v_ha2, 0, narrow_comp / 1e-16, alpha=0.4,
                color=COLORS["rt"], label=f"narrow  σ = {sigma_narrow} km/s")
ax.fill_between(v_ha2, 0, broad_comp / 1e-16, alpha=0.3,
                color=COLORS["model"], label=f"broad   σ = {sigma_broad} km/s")
ax.axvline(0, color="0.7", lw=0.7, ls=":")
ax.set_xlabel(r"$v$ (km s$^{-1}$)  relative to H$\alpha$")
ax.set_ylabel(r"$f_\lambda$ (10$^{-16}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$)")
ax.set_title(r"H$\alpha$ narrow + broad decomposition  (is_broad_candidate)", fontsize=12)
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()

# Compute flux for each component separately
flux_narrow = compute_line_fluxes(
    jnp.array([amp_narrow]), jnp.array([halpha_rest]), z, sigma_narrow, R_spec
)
flux_broad = compute_line_fluxes(
    jnp.array([amp_broad]), jnp.array([halpha_rest]), z, sigma_broad, R_spec
)
print(f"\nNarrow Hα flux:  {float(flux_narrow[0]):.2e} erg/s/cm²")
print(f"Broad  Hα flux:  {float(flux_broad[0]):.2e} erg/s/cm²")
print(f"Broad fraction:  {float(flux_broad[0]) / (float(flux_narrow[0]) + float(flux_broad[0])):.1%}")

# %% [markdown]
# ## 11  Analytical emission line marginalization
#
# This is tengri's most powerful emission line capability — absent from FastSpecFit.
# Rather than fitting line amplitudes by NNLS (like FastSpecFit), tengri
# **analytically marginalizes** them out of the likelihood under a Gaussian prior.
# The result is an exact Bayesian integral in O(n_lines³) matrix operations.
#
# The model is: `data = continuum + G @ a + noise`, where `G` is the
# `(n_pix, n_lines)` design matrix of Gaussian profiles.
# Marginalizing over `a` gives:
# ```
# ln L_marg = -½ χ²_marg - ½ prior_penalty + ½ log|Σ_a|
# ```
# where `χ²_marg = χ²_cont - â^T (G^T N⁻¹ G) â` is the improvement from lines,
# and `Σ_a = (G^T N⁻¹ G + Λ⁻¹)⁻¹` is the posterior amplitude covariance.

# %%
from tengri.observation.eline_marginalization import (
    apply_doublet_constraints,
    build_eline_design_matrix,
    build_line_design_matrix,
    expand_constrained_amplitudes,
    marginalize_emission_lines,
    predict_with_marginalized_lines,
)

# Build a mock DESI-like spectrum at z=0.15 with injected lines
z_marg = 0.15
R_marg = 2000.0
n_pix_m = 2000
wave_marg = np.linspace(5500., 9000., n_pix_m)  # observed frame

# Continuum model: simple power law
cont_marg = 3e-17 * (wave_marg / 7000.) ** (-0.5)

# True amplitudes (Halpha + NII doublet + Hbeta + OIII doublet)
line_waves_rest_m = jnp.array([4862.68, 4960.30, 5008.24, 6564.61, 6549.86, 6585.28])
true_amps = jnp.array([5e-18, 1.5e-18, 4.5e-18, 1.5e-17, 1.8e-18, 5.3e-18])
sigma_line_m = 80.0  # km/s

# Build design matrix
G = build_eline_design_matrix(
    jnp.array(wave_marg), line_waves_rest_m,
    spectral_resolution=R_marg, redshift=z_marg,
    eline_sigma_kms=sigma_line_m,
)
print(f"Design matrix G: {G.shape}  ({n_pix_m} pixels × {len(line_waves_rest_m)} lines)")

# Inject lines and add noise
true_spectrum = jnp.array(cont_marg) + G @ true_amps
noise_rms_m = 5e-19
rng_m = np.random.default_rng(17)
obs_spectrum = true_spectrum + jnp.array(rng_m.normal(0., noise_rms_m, n_pix_m))
noise_m = jnp.full(n_pix_m, noise_rms_m)

# %%
# Marginalize: continuum is known, marginalize over line amplitudes
residual_m = obs_spectrum - jnp.array(cont_marg)
ln_l_flat, a_hat_flat, a_cov_flat = marginalize_emission_lines(
    residual_m, noise_m, G, prior_variance=None  # flat prior
)
print(f"Marginalized log-likelihood (flat prior): {float(ln_l_flat):.2f}")
print(f"\nRecovered amplitudes vs truth:")
line_shortnames = ["Hβ", "[OIII]4959", "[OIII]5007", "Hα", "[NII]6548", "[NII]6584"]
for name, a_t, a_r in zip(line_shortnames, true_amps, a_hat_flat):
    print(f"  {name:<12}  truth={float(a_t):.2e}  recovered={float(a_r):.2e}")

# %%
# Now apply doublet constraints: reduce from 6 free amps to 4 independent
# using the LineList constraint matrix for Hβ, OIII 5007, Hα, NII 6584
from tengri.observation.line_list import LineList

cat_m = LineList.default_optical()
sub_m = cat_m.select(names=["Hbeta", "OIII_4959", "OIII_5007", "Halpha", "NII_6548", "NII_6584"])
C_m = sub_m.build_constraint_matrix()
print(f"\nConstraint matrix: {C_m.shape}  ({sub_m.n_lines} lines → {sub_m.n_independent} free amps)")

# Apply doublet constraint: G_eff = G @ C
G_eff = apply_doublet_constraints(G, C_m)
print(f"Constrained design matrix: {G_eff.shape}")

ln_l_constrained, a_hat_ind, a_cov_ind = marginalize_emission_lines(
    residual_m, noise_m, G_eff, prior_variance=None
)
print(f"Marginalized log-likelihood (doublet constraints): {float(ln_l_constrained):.2f}")

# Expand back to full line set
a_hat_full, a_cov_full = expand_constrained_amplitudes(a_hat_ind, a_cov_ind, C_m)
a_err_full = jnp.sqrt(jnp.diag(a_cov_full))
print(f"\nFull-set amplitudes (with propagated uncertainty):")
for name, a_t, a_r, a_e in zip(sub_m.names, true_amps, a_hat_full, a_err_full):
    snr = float(a_r) / float(a_e) if float(a_e) > 0 else 0
    print(f"  {name:<14}  truth={float(a_t):.2e}  recov={float(a_r):.2e} ± {float(a_e):.2e}  S/N={snr:.1f}")

# %%
# Reconstruct the full spectrum and compare residuals
model_full_m = predict_with_marginalized_lines(jnp.array(cont_marg), G, a_hat_flat)

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=False)

# Plot Hα window in observed frame
ha_obs_m = 6564.61 * (1 + z_marg)
w_lo_m, w_hi_m = ha_obs_m - 200., ha_obs_m + 200.
wm = (wave_marg >= w_lo_m) & (wave_marg <= w_hi_m)

axes[0].plot(wave_marg[wm], np.array(obs_spectrum)[wm] / 1e-17,
             color="0.4", lw=0.8, label="observed")
axes[0].plot(wave_marg[wm], np.array(model_full_m)[wm] / 1e-17,
             color=COLORS["model"], lw=1.5, label="continuum + lines (marginalized)")
axes[0].plot(wave_marg[wm], cont_marg[wm] / 1e-17,
             color=COLORS["rt"], lw=1.0, ls="--", label="continuum only")
axes[0].set_ylabel(r"$f$ (10$^{-17}$)", fontsize=10)
axes[0].set_title(r"H$\alpha$ region after marginalization", fontsize=11)
axes[0].legend(fontsize=8.5)

# Hβ + OIII window
hb_obs_m = 4862.68 * (1 + z_marg)
w_lo_b, w_hi_b = hb_obs_m - 300., hb_obs_m + 400.
wb = (wave_marg >= w_lo_b) & (wave_marg <= w_hi_b)

axes[1].plot(wave_marg[wb], np.array(obs_spectrum)[wb] / 1e-17,
             color="0.4", lw=0.8, label="observed")
axes[1].plot(wave_marg[wb], np.array(model_full_m)[wb] / 1e-17,
             color=COLORS["model"], lw=1.5, label="continuum + lines (marginalized)")
axes[1].plot(wave_marg[wb], cont_marg[wb] / 1e-17,
             color=COLORS["rt"], lw=1.0, ls="--", label="continuum only")
axes[1].set_xlabel(r"$\lambda_\mathrm{obs}$ ($\AA$)", fontsize=10)
axes[1].set_ylabel(r"$f$ (10$^{-17}$)", fontsize=10)
axes[1].set_title(r"H$\beta$ + [OIII] region after marginalization", fontsize=11)
axes[1].legend(fontsize=8.5)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 12  Spectral indices — Dn4000, HdA (differentiable)
#
# Tengri measures stellar population diagnostics like the 4000 Å break (Dn4000)
# and Balmer absorption indices (HdA, HgA) directly on forward-model spectra.
# Unlike table look-ups in FastSpecFit, tengri's `measure_index_jax` uses soft
# sigmoid windows — differentiable through the model SED — so index gradients
# flow into the posterior.
#
# Dn4000 > 1.5 indicates an old stellar population (post-starburst or passive).
# HdA < 0 Å indicates emission (dominant star formation); HdA ~ 3-8 Å indicates
# a post-starburst "E+A" population where Balmer absorption is strong.

# %%
from tengri.observation.spectral_indices import (
    STANDARD_INDICES,
    SpectralIndexData,
    measure_index_jax,
)

print("Available standard spectral indices:")
for name, idx in STANDARD_INDICES.items():
    wrange = f"{idx.wave_min:.0f}–{idx.wave_max:.0f} Å"
    print(f"  {name:<10}  {idx.index_type:<6}  {wrange}")

# %%
# Build a simple mock rest-frame spectrum to measure indices on
wave_rest_idx = jnp.linspace(3700., 5500., 3000)

# Young SF galaxy: blue continuum, Hδ in emission → HdA < 0
# Simulate power-law continuum + Hδ emission line
flux_sf = 2e-17 * (wave_rest_idx / 4500.) ** (-0.8)
ha_rest_idx = 4102.89
sigma_hd = 70. / _C_KMS * ha_rest_idx
flux_sf = flux_sf + 3e-18 * jnp.exp(-0.5 * ((wave_rest_idx - ha_rest_idx) / sigma_hd) ** 2) / (
    jnp.sqrt(2 * jnp.pi) * sigma_hd
) * sigma_hd  # approximate emission contribution

# Old passive galaxy: red continuum, strong 4000 Å break, deep Balmer absorption
flux_old = 2e-17 * (wave_rest_idx / 4500.) ** 0.5
# Simulate the 4000 Å break: flux drops blueward of 4000 Å
flux_old = flux_old * jnp.where(wave_rest_idx < 4000., 0.55, 1.0)

# %%
dn4000_def = STANDARD_INDICES["Dn4000"]
hda_def = STANDARD_INDICES["HdA"]

dn4000_sf = float(measure_index_jax(wave_rest_idx, flux_sf, dn4000_def))
hda_sf = float(measure_index_jax(wave_rest_idx, flux_sf, hda_def))

dn4000_old = float(measure_index_jax(wave_rest_idx, flux_old, dn4000_def))
hda_old = float(measure_index_jax(wave_rest_idx, flux_old, hda_def))

print("Spectral index measurements:")
print(f"  {'Galaxy':<18} {'Dn4000':>8}  {'HdA (Å)':>10}")
print(f"  {'Star-forming':<18} {dn4000_sf:>8.3f}  {hda_sf:>10.3f}")
print(f"  {'Passive/old':<18} {dn4000_old:>8.3f}  {hda_old:>10.3f}")
print()
print(f"  Dn4000 > 1.5 → old stellar population  (passive = {dn4000_old:.2f})")
print(f"  HdA < 0 → net Hδ emission  (SF = {hda_sf:.2f} Å)")

# %%
# Show the index measurement windows on the spectra
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, flux, label, color in zip(
    axes,
    [flux_sf, flux_old],
    ["Star-forming (blue, Hδ emission)", "Passive (4000 Å break, no emission)"],
    [COLORS["rt"], COLORS["model"]],
):
    ax.plot(np.array(wave_rest_idx), np.array(flux) / 1e-17, color=color, lw=1.0, label=label)

    # Shade Dn4000 windows
    for lo, hi in dn4000_def.continuum:
        ax.axvspan(lo, hi, alpha=0.12, color="gold", label="Dn4000 windows" if lo == dn4000_def.continuum[0][0] else "")

    # Shade HdA windows
    for lo, hi in hda_def.continuum:
        ax.axvspan(lo, hi, alpha=0.10, color="C2", label="HdA continuum" if lo == hda_def.continuum[0][0] else "")
    lo_f, hi_f = hda_def.feature
    ax.axvspan(lo_f, hi_f, alpha=0.18, color="C1", label="HdA feature")

    ax.set_xlabel(r"$\lambda_\mathrm{rest}$ ($\AA$)", fontsize=10)
    ax.set_ylabel(r"$f$ (10$^{-17}$)", fontsize=10)
    ax.set_title(label, fontsize=10)
    ax.set_xlim(3700, 5500)
    ax.legend(fontsize=7.5, loc="upper right")

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Differentiability of spectral indices
#
# Because `measure_index_jax` uses soft sigmoid edges (`jax.nn.sigmoid`),
# index values are differentiable through the model SED.  This means Dn4000
# and HdA gradients flow directly into the posterior — a capability FastSpecFit
# entirely lacks.

# %%
# Gradient of Dn4000 w.r.t. flux[i] — nonzero only in 3850–4100 Å windows
grad_dn4000 = jax.grad(lambda f: measure_index_jax(wave_rest_idx, f, dn4000_def))(flux_sf)
grad_hda = jax.grad(lambda f: measure_index_jax(wave_rest_idx, f, hda_def))(flux_sf)

fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
axes[0].plot(np.array(wave_rest_idx), np.array(grad_dn4000) / float(jnp.max(jnp.abs(grad_dn4000))),
             color="gold", lw=0.9)
axes[0].set_ylabel("∂Dn4000/∂f  (norm.)", fontsize=9)
axes[0].set_title("Gradient of Dn4000 through SED (soft sigmoid windows)", fontsize=10)
axes[0].axhline(0, color="0.7", lw=0.5)

axes[1].plot(np.array(wave_rest_idx), np.array(grad_hda) / float(jnp.max(jnp.abs(grad_hda))),
             color=COLORS["model"], lw=0.9)
axes[1].set_ylabel("∂HdA/∂f  (norm.)", fontsize=9)
axes[1].set_title("Gradient of HdA through SED", fontsize=10)
axes[1].axhline(0, color="0.7", lw=0.5)
axes[1].set_xlabel(r"$\lambda_\mathrm{rest}$ ($\AA$)", fontsize=10)

plt.tight_layout()
plt.show()
print(f"Dn4000 gradient is nonzero in {int(jnp.sum(jnp.abs(grad_dn4000) > 1e-30))} pixels")
print(f"HdA gradient is nonzero in {int(jnp.sum(jnp.abs(grad_hda) > 1e-30))} pixels")

# %% [markdown]
# ## 13  LineFluxData — photometric fluxes with upper limits
#
# When users have measured emission line fluxes from IFU data, narrow-band
# imaging, or external pipelines, `LineFluxData` provides a declarative
# container for fitting.  Upper limits use a survival-function likelihood
# (`ln L = ln(erfc(...))`) rather than discarding the non-detection.
#
# This is different from FastSpecFit's pixel-level spectral fitting:
# here we compare integrated fluxes directly, enabling joint fitting with
# broadband photometry when spectra are unavailable.

# %%
from tengri.observation.line_flux_data import LineFluxData

# Construct from a measurement dict — mimics output from a line-finding pipeline
lfd_detected = LineFluxData.from_dict({
    "Halpha":    (1.35e-16, 0.12e-16),
    "Hbeta":     (4.8e-17,  0.6e-17),
    "OIII_5007": (9.2e-17,  0.9e-17),
    "NII_6584":  (3.1e-17,  0.5e-17),
    "SII_6717":  (1.8e-17,  0.4e-17),
    "SII_6731":  (1.5e-17,  0.4e-17),
})
print(lfd_detected.summary())
print(f"  Line names:  {lfd_detected.names}")
print(f"  Fluxes (1e-17 erg/s/cm²): {[f'{float(f)/1e-17:.2f}' for f in lfd_detected.fluxes]}")

# %%
# Upper limits: OI_6300 is below detection threshold (report 3-sigma limit)
lfd_with_ul = LineFluxData.from_dict({
    "Halpha":    (1.35e-16, 0.12e-16),
    "Hbeta":     (4.8e-17,  0.6e-17),
    "OIII_5007": (9.2e-17,  0.9e-17),
    "OI_6300":   (3.0e-18,  1.0e-18),   # 3σ upper limit
})
# Mark OI as upper limit
is_ul = jnp.array([False, False, False, True])
lfd_ul = LineFluxData(
    names=lfd_with_ul.names,
    fluxes=lfd_with_ul.fluxes,
    errors=lfd_with_ul.errors,
    wavelengths=lfd_with_ul.wavelengths,
    is_upper_limit=is_ul,
)

# Compare log-likelihoods for a model that predicts OI at different levels
model_fluxes_detected = lfd_ul.fluxes  # exact match
model_fluxes_high_oi = lfd_ul.fluxes.at[3].set(8e-18)  # OI 2x above limit

ll_match = float(lfd_ul.log_likelihood(model_fluxes_detected))
ll_high_oi = float(lfd_ul.log_likelihood(model_fluxes_high_oi))
print(f"Log-likelihood (OI at limit level):  {ll_match:.2f}")
print(f"Log-likelihood (OI 2× above limit): {ll_high_oi:.2f}")
print(f"Δ ln L = {ll_high_oi - ll_match:.2f}  (negative → model above limit is penalised)")

# %%
# Visualise the survival-function likelihood for [OI] upper limit
oi_model_range = np.logspace(np.log10(0.5e-18), np.log10(2e-17), 200)
ll_scan = []
for oi_val in oi_model_range:
    mf = lfd_ul.fluxes.at[3].set(oi_val)
    ll_scan.append(float(lfd_ul.log_likelihood(mf)))

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.semilogx(oi_model_range / 1e-18, ll_scan, color=COLORS["model"], lw=1.5)
ax.axvline(float(lfd_ul.fluxes[3]) / 1e-18, color="0.4", lw=1.0, ls="--",
           label=r"3$\sigma$ limit = " + f"{float(lfd_ul.fluxes[3])/1e-18:.1f}" + r" $\times 10^{-18}$")
ax.set_xlabel(r"[OI] 6300 model flux  (10$^{-18}$ erg/s/cm²)", fontsize=10)
ax.set_ylabel("ln L", fontsize=10)
ax.set_title("[OI] 6300 — upper limit survival-function likelihood", fontsize=11)
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 14  Named line groups — eline_catalog
#
# `eline_catalog.py` provides `LINE_GROUPS` — named collections of emission lines
# used throughout tengri for different science cases.  Notably, tengri extends
# beyond FastSpecFit's optical coverage to UV diagnostics from Flury+2024
# (arXiv:2412.06763) for ionizing photon escape fraction measurements.

# %%
from tengri.observation.eline_catalog import (
    EMISSION_LINES,
    LINE_GROUPS,
    get_line_names,
    get_line_wavelengths,
)

print("Available named line groups:")
for gname, lines in LINE_GROUPS.items():
    print(f"  {gname:<28}  {len(lines)} lines: {', '.join(lines[:4])}{'...' if len(lines) > 4 else ''}")

# %%
# Flury+2024 UV diagnostics (arXiv:2412.06763) — ionizing photon escape
uv_diag_groups = [k for k in LINE_GROUPS if k.startswith("flury")]
print("Flury+2024 UV diagnostics:")
for g in uv_diag_groups:
    lines = get_line_names(g)
    waves = get_line_wavelengths(g)
    print(f"  {g:<28}  {[f'{n} ({float(w):.0f} Å)' for n, w in zip(lines, waves)]}")

# %%
# Compare line coverage: FastSpecFit optical vs tengri (optical + UV)
optical_names = set(get_line_names("optical_narrow"))
uv_names = set(get_line_names("uv_narrow"))
all_names = set(EMISSION_LINES.keys())

print(f"\nLine catalog coverage:")
print(f"  FastSpecFit-equivalent optical lines:  {len(optical_names)}")
print(f"  UV lines (Flury+2024):                 {len(uv_names)}")
print(f"  Total unique lines in tengri catalog:  {len(all_names)}")
print()
print("UV lines and rest-frame wavelengths:")
for name in sorted(uv_names, key=lambda n: EMISSION_LINES[n][0]):
    wave, ltype, _ = EMISSION_LINES[name]
    print(f"  {name:<12}  {wave:.2f} Å  ({ltype})")

# %%
# Visualise the line catalog coverage
all_waves = sorted(EMISSION_LINES[n][0] for n in EMISSION_LINES)
optical_waves = [EMISSION_LINES[n][0] for n in optical_names]
uv_waves_list = [EMISSION_LINES[n][0] for n in uv_names]

fig, ax = plt.subplots(figsize=(12, 2.5))
ax.vlines(uv_waves_list, 0, 1, color=COLORS["model"], lw=1.2, label="UV (Flury+2024)")
ax.vlines(optical_waves, 0, 1, color=COLORS["rt"], lw=1.2, label="Optical (FastSpecFit-equiv.)")
ax.set_xlabel(r"Rest-frame wavelength ($\AA$)", fontsize=10)
ax.set_yticks([])
ax.set_xlim(1100, 7500)
ax.legend(fontsize=9, loc="upper right")
ax.set_title("Tengri emission line catalog coverage", fontsize=11)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary
#
# | Feature | FastSpecFit | Tengri | Differentiable? |
# |---|---|---|---|
# | Line catalog metadata | `emlines.ecsv` (isstrong, plotgroup) | `LineList.is_strong`, `plot_group` | — |
# | Named line groups | none | `LINE_GROUPS` (optical + UV + BPT + Flury) | — |
# | Continuum pixel mask | `LineMasker.linepix_and_contpix()` | `build_line_mask()` | No (NumPy) |
# | Doublet flux constraints | `build_linemodels()` fixed ratios | `build_constraint_matrix()` + `apply_doublet_constraints()` | — |
# | Gaussian line profiles | `build_emline_model()` | `build_eline_design_matrix()` | **Yes** |
# | Integrated line fluxes | `populate_emtable()` FLUX | `compute_line_fluxes()` | **Yes** |
# | Rest-frame EW | `populate_emtable()` EW | `compute_equivalent_widths()` | **Yes** |
# | Velocity moments | `populate_emtable()` SIGMAINT | `compute_line_moments()` | **Yes** (soft kernel) |
# | Line amplitude fitting | NNLS post-fit | `marginalize_emission_lines()` (exact Bayes) | **Yes** |
# | Amplitude covariance | none | `a_cov = (G^T N⁻¹ G + Λ⁻¹)⁻¹` | **Yes** |
# | Photometric line fluxes | none | `LineFluxData` (Gaussian + survival function) | **Yes** |
# | Spectral indices | none | `SpectralIndexData` + `measure_index_jax` | **Yes** |
# | Broad + narrow decomp. | separate broad/narrow templates | `is_broad_candidate` + `build_broad_design_matrix()` | **Yes** |
# | QA panel layout | `plotgroup` column | `LineList.plot_group` | — |
# | UV line diagnostics | none | Flury+2024 groups in `eline_catalog` | — |
#
# The critical design difference: all tengri measurement functions can sit **inside**
# the posterior loss (gradients flow through amplitudes, σ, R, and the SED itself),
# while FastSpecFit's pipeline is entirely post-fit.

# %%
print("All cells complete.")
