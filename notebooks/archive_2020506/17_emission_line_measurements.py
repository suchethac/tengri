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
# _Post-fit spectral analysis: differentiable line diagnostics in tengri_
#
# **Prerequisites:** [`04_fitting_spectra.py`](04_fitting_spectra.py); `examples/nebular/` for backend choices.
# **Continue with:** [`12_diagnostics.py`](12_diagnostics.py), [`13_extending_tengri.py`](13_extending_tengri.py).
#
# ## What you'll learn
#
# - **Extract emission line fluxes and equivalent widths** from fit spectra (API: `compute_line_fluxes`, `compute_equivalent_widths`)
# - **Measure velocity moments** (centroid, dispersion) with soft Gaussian kernels — JIT-compatible (API: `compute_line_moments`)
# - **Build BPT diagrams** using measured line ratios ([OIII]/Hβ vs [NII]/Hα)
# - **Compute gradients through line measurements**: unlike FastSpecFit/template-fitting codes, every line flux is **differentiable w.r.t. physical parameters**. One jax.grad call shows which parameters most affect Hα flux.
#
# **Note:** This notebook uses smooth SED-based SFH (Paper I). See `12_diagnostics` for BPT interpretation across the galaxy population.

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
import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)
warnings.filterwarnings("ignore", category=FutureWarning)

from _plot_style import setup_style, COLORS

setup_style()

# %% [markdown]
# ## 1  Load SSP and fit a spectrum
#
# Start with a fit spectrum; here we mock one by injecting emission lines into a continuum.

# %%
from tengri.observation.line_list import LineList

# Load the default optical line catalog (~40 lines with DESI metadata)
cat = LineList.default_optical()

# Select BPT + SII lines for demonstration
bpt_names = [
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
bpt = cat.select(names=bpt_names)
print(f"Selected {bpt.n_lines} lines for analysis")

# Mock posterior median amplitudes [erg/s/cm²/Å] — star-forming galaxy ratios
_amps = {
    "OII_3726": 8.0e-18,
    "OII_3729": 9.0e-18,
    "Hbeta": 1.0e-17,
    "OIII_4959": 5.0e-18,
    "OIII_5007": 1.5e-17,
    "NII_6548": 6.0e-18,
    "Halpha": 3.0e-17,
    "NII_6584": 1.8e-17,
    "SII_6717": 5.0e-18,
    "SII_6731": 4.0e-18,
}
amplitudes = jnp.array([_amps[n] for n in bpt.names])
rest_waves = bpt.wavelengths

z = 0.15
sigma_kms = 80.0  # intrinsic line width [km/s]
R_spec = 2500.0  # DESI spectral resolution

# %% [markdown]
# ## 2  Extract line fluxes and equivalent widths
#
# Convert Gaussian **amplitudes** (what the SED model fits) to integrated fluxes and EWs.
# These are pure-JAX, fully differentiable functions.

# %%
from tengri.analysis.diagnostics.lines import (
    compute_line_fluxes,
    compute_equivalent_widths,
    compute_line_moments,
)

# Compute integrated line fluxes
fluxes = compute_line_fluxes(amplitudes, rest_waves, z, sigma_kms, R_spec)

# Mock continuum at each line center [erg/s/cm²/Å]
continuum_at_lines = jnp.array(
    [float(3e-17 * (float(w) * (1 + z) / 5500.0) ** (-0.5)) for w in bpt.wavelengths]
)
ews = compute_equivalent_widths(fluxes, continuum_at_lines, z)

print(f"\n{'Line':<14} {'Flux':>14}  {'EW_rest':>10}")
print("-" * 42)
for name, flux, ew in zip(bpt.names, fluxes, ews):
    print(f"{name:<14} {float(flux):.2e}  {float(ew):>10.1f} Å")

# %%
# Visualize: bar chart of EWs by line
fig, ax = plt.subplots(figsize=(9, 3.5))
x = np.arange(bpt.n_lines)
colors = [COLORS["model"] if s else COLORS["rt"] for s in bpt.is_strong]
ax.bar(x, [float(e) for e in ews], color=colors, alpha=0.85, width=0.7)
ax.set_xticks(x)
ax.set_xticklabels(bpt.names, rotation=40, ha="right", fontsize=10)
ax.set_ylabel(r"Rest-frame EW$_0$ ($\AA$)")
ax.set_title("Emission line equivalent widths", fontsize=12)
ax.axhline(0, color="0.4", lw=0.7)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3  Velocity moments via soft Gaussian kernels
#
# Unlike FastSpecFit's hard pixel windows, tengri uses soft Gaussian kernels
# inside `compute_line_moments` — differentiable through σ_kms, so gradients
# flow into the posterior during fitting.

# %%
_C_KMS = 2.99792458e5

# Inject mock Hα residual (observed frame) with a known velocity offset
sigma_true_kms = 95.0
v_offset_true_kms = +45.0

halpha_rest = 6564.61
halpha_obs = halpha_rest * (1 + z)
halpha_obs_shifted = halpha_obs * (1 + v_offset_true_kms / _C_KMS)

wave_ha = np.linspace(halpha_obs - 60.0, halpha_obs + 60.0, 600)
sigma_ang = sigma_true_kms / _C_KMS * halpha_obs_shifted
amp_true = 2.5e-16

rng = np.random.default_rng(7)
flux_ha = amp_true * np.exp(-0.5 * ((wave_ha - halpha_obs_shifted) / sigma_ang) ** 2) + rng.normal(
    0.0, 3e-18, size=len(wave_ha)
)
ivar_ha = np.full_like(flux_ha, 1.0 / (3e-18) ** 2)

residual_ha = flux_ha - 3.5e-18

# Measure moments
sigma_window_kms = 800.0
v_cent, sigma_int = compute_line_moments(
    jnp.array(wave_ha),
    jnp.array(residual_ha),
    jnp.array(ivar_ha),
    line_obs_wave=halpha_obs,
    sigma_kms=sigma_window_kms,
)

print(f"True:      v_offset = {v_offset_true_kms:+.1f} km/s,  σ_int = {sigma_true_kms:.1f} km/s")
print(f"Measured:  v_cent = {float(v_cent):+.1f} km/s,  σ_int = {float(sigma_int):.1f} km/s")

# %%
# Plot the measurement
v_axis = _C_KMS * (wave_ha - halpha_obs) / halpha_obs
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(v_axis, residual_ha / amp_true, color="0.5", lw=0.6, label="residual (S/N ≈ 83)")
ax.plot(
    v_axis,
    np.exp(-0.5 * ((v_axis - v_offset_true_kms) / sigma_true_kms) ** 2),
    color=COLORS["truth"],
    lw=1.5,
    ls="--",
    label=f"true profile (σ = {sigma_true_kms} km/s)",
)
ax.axvline(
    float(v_cent), color=COLORS["model"], lw=1.5, label=f"centroid = {float(v_cent):+.1f} km/s"
)
ax.fill_betweenx(
    [0, 1],
    float(v_cent) - float(sigma_int),
    float(v_cent) + float(sigma_int),
    color=COLORS["model"],
    alpha=0.10,
    label=f"σ_int = {float(sigma_int):.1f} km/s",
)
ax.set_xlabel(r"$v$ relative to $\lambda_0$ (km s$^{-1}$)")
ax.set_ylabel(r"$f$ / peak amplitude")
ax.set_title(r"Soft-kernel velocity moments — H$\alpha$", fontsize=12)
ax.legend(fontsize=10)
ax.set_xlim(-500, 500)
ax.set_ylim(-0.15, 1.2)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4  BPT Diagram
#
# Compute [OIII]/Hβ and [NII]/Hα line ratios to place galaxies on the BPT diagnostic grid.

# %%
# Extract the required lines for BPT
idx_hb = list(bpt.names).index("Hbeta")
idx_oiii = list(bpt.names).index("OIII_5007")
idx_ha = list(bpt.names).index("Halpha")
idx_nii = list(bpt.names).index("NII_6584")

oiii_hb = float(fluxes[idx_oiii]) / float(fluxes[idx_hb])
nii_ha = float(fluxes[idx_nii]) / float(fluxes[idx_ha])

print(f"[OIII]/Hβ  = {oiii_hb:.2f}")
print(f"[NII]/Hα   = {nii_ha:.2f}")

# Plot BPT with Kewley+2001 (star-forming) and Kauffmann+2003 (LINER) boundaries
fig, ax = plt.subplots(figsize=(7, 6))

# Kewley+2001 demarcation (upper boundary: HII regions vs Seyfert 2)
log_nii_ha_kew = np.linspace(-2, 0.5, 100)
log_oiii_hb_kew = 0.61 / (log_nii_ha_kew - 0.05) + 1.3

# Kauffmann+2003 demarcation (SF vs LINER)
log_nii_ha_kau = np.linspace(-2, 0.2, 100)
log_oiii_hb_kau = 0.61 / (log_nii_ha_kau - 0.47) + 1.19

ax.semilogy(
    10.0**log_nii_ha_kew, 10.0**log_oiii_hb_kew, "k--", lw=1.5, label="Kewley+01 (HII/Seyfert2)"
)
ax.semilogy(
    10.0**log_nii_ha_kau, 10.0**log_oiii_hb_kau, "k-", lw=1.0, label="Kauffmann+03 (SF/LINER)"
)

ax.plot(nii_ha, oiii_hb, "o", markersize=10, color=COLORS["model"], label="This galaxy", zorder=5)

ax.set_xlabel(r"[NII] 6584 / H$\alpha$ [6563]")
ax.set_ylabel(r"[OIII] 5007 / H$\beta$ [4861]")
ax.set_xlim(0.01, 3)
ax.set_ylim(0.05, 50)
ax.set_title("BPT Diagnostic Diagram", fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5  THE JAX PAYOFF: Differentiable line fluxes
#
# Every line measurement is a pure-JAX function. You can compute gradients
# w.r.t. physical parameters inside a loss function — something FastSpecFit cannot do.


# %%
# Example: gradient of Hα flux w.r.t. all amplitudes
def total_ha_flux(amps):
    """Sum of all measured line fluxes — a toy objective for fitting."""
    return jnp.sum(compute_line_fluxes(amps, rest_waves, z, sigma_kms, R_spec))


grad_fn = jax.jit(jax.grad(total_ha_flux))
grads = grad_fn(amplitudes)

print("\n∂(total_flux)/∂(amplitude_i)  — gradient of line flux w.r.t. each line:")
print(f"{'Line':<14} {'∂F/∂A':>14}")
print("-" * 32)
for name, g in zip(bpt.names, grads):
    print(f"{name:<14} {float(g):.4e}")

# Visualize: which parameters affect [OIII] flux the most?
fig, ax = plt.subplots(figsize=(9, 3))
x = np.arange(len(bpt.names))
ax.bar(x, [float(g) for g in grads], color=COLORS["model"], alpha=0.8, width=0.6)
ax.set_xticks(x)
ax.set_xticklabels(bpt.names, rotation=40, ha="right", fontsize=10)
ax.set_ylabel(r"∂F_line / ∂A (dimensionless)")
ax.set_title("Gradient: how does total line flux change with each amplitude?", fontsize=12)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary: Differentiable vs. Post-Fit Pipelines
#
# | Feature | FastSpecFit | Tengri | Differentiable? |
# |---|---|---|---|
# | Line catalog | `emlines.ecsv` | `LineList.default_optical()` | — |
# | Integrated fluxes | NNLS fit inside pixel loop | `compute_line_fluxes()` | **Yes** |
# | Equivalent widths | post-fit lookup | `compute_equivalent_widths()` | **Yes** |
# | Velocity moments | hard pixel windows (not JIT) | soft Gaussian kernel | **Yes** |
# | Line amplitude fitting | NNLS (post-fit) | Exact Bayes (inside loss) | **Yes** |
# | BPT diagnostics | computed from NNLS fluxes | computed from gradients | **Yes** |
#
# **Why it matters:** In tengri, line measurements sit **inside the posterior loss**.
# Gradients of [OIII]/Hβ, Hα EW, or velocity dispersion flow directly back into the SED fit,
# enabling physically motivated line-based priors and velocity-field fitting.

# %%
print("All cells complete.")
