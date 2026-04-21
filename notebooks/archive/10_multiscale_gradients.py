# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Multiscale Gradient Sensitivity
#
# Which spectral features, at which resolution, constrain each physical
# parameter and derived quantity?  This notebook answers that question
# with a **wavelet-like decomposition** of gradient sensitivity.
#
# We compute the Jacobian $\partial f / \partial \theta$ of the full
# **stochastic SFH model** at 64 logarithmically spaced spectral scales
# (from 2 Å individual lines to 3000 Å broadband), producing a 2D
# scalogram for each parameter: wavelength × scale × gradient magnitude.
#
# **By the end you will understand:**
# 1. How gradient sensitivity varies across wavelength and spectral scale
# 2. Which parameters and derived quantities need spectroscopy vs photometry
# 3. Where spectral features (D4000, Balmer, Mg b, H$\alpha$) appear in
#    the scalogram and what they constrain
# 4. How to read a multiscale gradient map for survey design

# %%
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
import numpy as np

from tengri import (
    SEDModel, ParamSpec, Uniform, Fixed,
    load_ssp_data, load_filter_set,
)

import sys; sys.path.insert(0, ".")
from _plot_style import setup_style, COLORS, SDSS_WAVE_EFF, SPECTRAL_FEATURES
setup_style()
import os; os.makedirs("notebook_figures", exist_ok=True)

# C3K SSP templates — higher spectral resolution for feature sensitivity
# Download: https://halos.as.arizona.edu/suchethacooray/dsps_ssp/
ssp_data = load_ssp_data("../data/ssp_mist_c3k_a_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")
filters = load_filter_set(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])

print(f"SSP grid: {len(ssp_data.ssp_lgmet)} metallicities × "
      f"{len(ssp_data.ssp_lg_age_gyr)} ages × "
      f"{len(ssp_data.ssp_wave)} wavelengths")
print(f"Wavelength range: {float(ssp_data.ssp_wave[0]):.0f}–"
      f"{float(ssp_data.ssp_wave[-1]):.0f} Å")

# %% [markdown]
# ## Stochastic SEDModel Setup
#
# We use the full stochastic SFH model — including PSD parameters
# ($\sigma$, $\tau$) and the GP latent field — so the gradients
# capture the sensitivity of observables to burstiness parameters,
# not just smooth SFH shape.

# %%
# Stochastic model with free PSD parameters
spec = ParamSpec(
    mean_sfh_type=["dpl", "field"],
    sfh_dpl_alpha=Uniform(0.5, 3.0),
    sfh_dpl_beta=Uniform(0.5, 3.0),
    sfh_dpl_tau_gyr=Uniform(0.5, 13.0),
    sfh_dpl_log_peak_sfr=Uniform(-1.0, 2.0),
    sfh_field_psd_sigma=Uniform(0.1, 4.0),
    sfh_field_psd_tau_myr=Uniform(1.0, 300.0),
    met_logzsol=Uniform(-2.0, 0.5),
    dust_tau_bc=Uniform(0.0, 2.0),
    dust_tau_diff=Uniform(0.0, 2.0),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.1),
    n_grid=128,
)
model = SEDModel(spec, ssp_data, filters=filters)

# Fiducial moderately bursty galaxy
key = jax.random.PRNGKey(42)
true_params = spec.sample(key)
true_params = {**true_params,
    "sfh_dpl_alpha": 1.5, "sfh_dpl_beta": 1.5,
    "sfh_dpl_tau_gyr": 5.0, "sfh_dpl_log_peak_sfr": 1.301,  # log10(20)
    "sfh_field_psd_sigma": 1.5, "sfh_field_psd_tau_myr": 50.0,
    "met_logzsol": -0.3,
    "dust_tau_bc": 0.5, "dust_tau_diff": 0.3,
}

# Physical parameter names (exclude sfh_field_xi latent vector)
phys_names = [n for n in spec.free_params if n != "sfh_field_xi"]
phys_idx = [spec.free_params.index(n) for n in phys_names]

# Parameter vector for Jacobian (physical params only — freeze sfh_field_xi)
param_vec = jnp.array([true_params[k] for k in phys_names])

# Prior ranges for normalization
prior_ranges = np.array([
    spec.get_distribution(n).bounds[1] - spec.get_distribution(n).bounds[0]
    for n in phys_names
])

print(f"Physical parameters ({len(phys_names)}): {phys_names}")
print(f"GP latent dims: 128 (frozen at fiducial for gradient computation)")

# %% [markdown]
# ## Step 1: Broadband Photometric Gradients
#
# Baseline: SDSS $ugriz$ photometry.  The Jacobian
# $\partial f_b / \partial \theta_k$ shows which bands constrain which
# physical parameters.

# %%
def phot_from_phys(vec):
    """Photometry as a function of physical parameters (sfh_field_xi frozen)."""
    p = dict(true_params)
    for i, name in enumerate(phys_names):
        p[name] = vec[i]
    return model.predict_photometry(p)

jac_phot = jax.jacobian(phot_from_phys)(param_vec)
jac_phot_np = np.array(jac_phot)

# Normalize: |df/dθ| × prior range
sens_phot = np.abs(jac_phot_np) * prior_ranges[None, :]
sens_phot_norm = sens_phot / sens_phot.max(axis=1, keepdims=True)

band_names = ["u", "g", "r", "i", "z"]

fig, ax = plt.subplots(figsize=(9, 5))
im = ax.imshow(sens_phot_norm.T, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
ax.set_xticks(range(len(band_names)))
ax.set_xticklabels([f"SDSS {b}" for b in band_names])
ax.set_yticks(range(len(phys_names)))
ax.set_yticklabels([n.replace("_", " ") for n in phys_names])
ax.set_xlabel("Photometric Band")
ax.set_ylabel("Physical Parameter")
ax.set_title(r"Broadband Sensitivity: $|\partial f_\nu / \partial \theta|$ (normalized per band)")
plt.colorbar(im, ax=ax, label="Relative sensitivity")
for i in range(sens_phot_norm.shape[1]):
    for j in range(sens_phot_norm.shape[0]):
        val = sens_phot_norm[j, i]
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=7, color="white" if val > 0.5 else "black")
plt.tight_layout()
plt.savefig("notebook_figures/10_multiscale_gradients_fig01.png", dpi=72, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Step 1b: Broadband Sensitivity to Derived Quantities
#
# Which photometric bands constrain stellar mass $M_*$, SFR$_{100}$,
# and SFR$_{10}$?  We compute $\partial Q / \partial f_b$ via the
# chain rule: $\partial Q / \partial f_b = \sum_k (\partial Q / \partial \theta_k)
# \cdot (\partial \theta_k / \partial f_b)$.

# %%
# Derived quantity Jacobian: dQ/dθ
def derived_from_phys(vec):
    p = dict(true_params)
    for i, name in enumerate(phys_names):
        p[name] = vec[i]
    d = model.predict_derived(p)
    return jnp.array([
        jnp.log10(jnp.clip(d["stellar_mass"], 1e-30, None)),
        jnp.log10(jnp.clip(d["sfr_100myr"], 1e-30, None)),
        jnp.log10(jnp.clip(d["sfr_10myr"], 1e-30, None)),
    ])

jac_derived = np.array(jax.jacobian(derived_from_phys)(param_vec))
derived_names = [r"$\log M_*$", r"$\log$ SFR$_{100}$", r"$\log$ SFR$_{10}$"]

# Broadband sensitivity to derived quantities via chain rule:
# S_Q(band) = Σ_k |df_band/dθ_k| × |dQ/dθ_k|
sens_derived_phot = np.zeros((len(band_names), len(derived_names)))
for q_idx in range(len(derived_names)):
    dq = np.abs(jac_derived[q_idx, :])  # (n_params,)
    for b_idx in range(len(band_names)):
        df = np.abs(jac_phot_np[b_idx, :])  # (n_params,)
        sens_derived_phot[b_idx, q_idx] = np.sum(df * dq)

# Normalize per derived quantity
for q in range(len(derived_names)):
    mx = sens_derived_phot[:, q].max()
    if mx > 0:
        sens_derived_phot[:, q] /= mx

fig, ax = plt.subplots(figsize=(7, 3.5))
im = ax.imshow(sens_derived_phot.T, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
ax.set_xticks(range(len(band_names)))
ax.set_xticklabels([f"SDSS {b}" for b in band_names])
ax.set_yticks(range(len(derived_names)))
ax.set_yticklabels(derived_names)
ax.set_xlabel("Photometric Band")
ax.set_ylabel("Derived Quantity")
ax.set_title("Which bands constrain stellar mass and SFR?")
plt.colorbar(im, ax=ax, label="Relative information content")
for i in range(sens_derived_phot.shape[1]):
    for j in range(sens_derived_phot.shape[0]):
        val = sens_derived_phot[j, i]
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=9, color="white" if val > 0.5 else "black")
plt.tight_layout()
plt.savefig("notebook_figures/10_multiscale_gradients_fig01b.png", dpi=72, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Step 2: Per-Pixel Spectral Gradient
#
# We compute the full per-pixel Jacobian $\partial f_\nu(\lambda) /
# \partial \theta_k$ across the rest-frame spectrum.  This is the
# foundation for the multiscale analysis — the sliding window
# scalograms are computed from this per-pixel gradient.

# %%
# Wavelength grid: full UV-to-NIR coverage
z = 0.1
wave_rest = np.array(ssp_data.ssp_wave)
wave_obs_full = wave_rest * (1 + z)

# Select UV-to-NIR range with good SSP sampling (avoid sparse far-IR)
mask = (wave_obs_full >= 1100) & (wave_obs_full <= 11000)
wave_obs_raw = wave_obs_full[mask]

# Interpolate to uniform 1 Å grid — the native SSP has variable spacing
# (0.7 Å optical, 1.2 Å red, 13 Å NIR) which would bias the sliding window
wave_obs = np.arange(float(wave_obs_raw[0]), float(wave_obs_raw[-1]), 1.0)
print(f"Spectral range: {wave_obs[0]:.0f}–{wave_obs[-1]:.0f} Å "
      f"({len(wave_obs)} pixels on uniform 1 Å grid)")

# Per-pixel spectrum from physical parameters (sfh_field_xi frozen)
wave_obs_jnp = jnp.array(wave_obs)
wave_obs_raw_jnp = jnp.array(wave_obs_raw)

def spectrum_from_phys(vec):
    p = dict(true_params)
    for i, name in enumerate(phys_names):
        p[name] = vec[i]
    sed = model.predict_sed(p)
    sed_raw = sed[mask]
    # Interpolate to uniform grid
    return jnp.interp(wave_obs_jnp, wave_obs_raw_jnp, sed_raw)

# Use vmap + grad for efficient per-pixel Jacobian computation.
# For n_pix >> n_params, compute grad of each pixel individually via vmap.
n_pix = len(wave_obs)
n_params = len(phys_names)

def _grad_pixel_k(vec, pixel_idx):
    """Gradient of pixel pixel_idx w.r.t. parameters."""
    return jax.grad(lambda v: spectrum_from_phys(v)[pixel_idx])(vec)

# vmap over pixel indices — computes all n_pix gradients in parallel
print(f"Computing per-pixel Jacobian ({n_pix} pixels × {n_params} params)...")
print("Using vmap + grad (much faster than jax.jacobian for n_pix >> n_params)")
pixel_indices = jnp.arange(n_pix)

# Process in chunks to avoid OOM
chunk_size = 500
jac_chunks = []
for start in range(0, n_pix, chunk_size):
    end = min(start + chunk_size, n_pix)
    chunk_idx = jnp.arange(start, end)
    chunk_jac = jax.vmap(lambda idx: _grad_pixel_k(param_vec, idx))(chunk_idx)
    jac_chunks.append(np.array(chunk_jac))
    if (start // chunk_size) % 5 == 0:
        print(f"  pixels {start}–{end} of {n_pix}")

jac_pixel = np.concatenate(jac_chunks, axis=0)
print(f"Jacobian shape: {jac_pixel.shape}  (n_pix × n_params)")

# %% [markdown]
# ## Step 3: Binned Scalograms
#
# For each scale $\Delta\lambda$, we bin the per-pixel gradient into
# non-overlapping bins and compute the mean gradient power per bin.
# This is fast and produces clean scalograms.
#
# 64 logarithmically spaced bin widths from 2 Å to 3000 Å.

# %%
bin_widths = np.geomspace(2, 3000, 64).astype(int)
bin_widths = np.unique(bin_widths)
n_scales = len(bin_widths)
wave_np = np.array(wave_obs)

print(f"Scales: {n_scales} from {bin_widths[0]} to {bin_widths[-1]} Å")

# Pre-compute gradient power per pixel per parameter
grad_sq_all = {}
for k, name in enumerate(phys_names):
    grad_sq_all[name] = (jac_pixel[:, k] * prior_ranges[k]) ** 2

# Compute binned scalograms for physical parameters
def fast_bin_mean(wave, values, bin_width):
    """Fast binning using numpy histogram (vectorized, no Python loop)."""
    edges = np.arange(float(wave[0]), float(wave[-1]) + bin_width, bin_width)
    sums, _ = np.histogram(wave, bins=edges, weights=values)
    counts, _ = np.histogram(wave, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = np.divide(sums, counts, out=np.zeros_like(sums, dtype=float),
                      where=counts > 0)
    return centers, means

print("Computing binned scalograms (vectorized)...")
scalograms = {}

for name in phys_names:
    grad_sq = grad_sq_all[name]
    rows = [fast_bin_mean(wave_np, grad_sq, bw) for bw in bin_widths]
    scalograms[name] = rows
    print(f"  {name}: done")

# Derived quantity scalograms (Fisher-weighted)
dq_names_short = ["log_Mstar", "log_SFR100", "log_SFR10"]
for q_idx, (q_label, q_short) in enumerate(zip(derived_names, dq_names_short)):
    dq_dtheta = np.abs(jac_derived[q_idx, :])
    rows = []
    for bw in bin_widths:
        # For each bin, combine parameter gradients weighted by dQ/dθ
        edges = np.arange(float(wave_np[0]), float(wave_np[-1]) + bw, bw)
        centers = 0.5 * (edges[:-1] + edges[1:])
        binned = np.zeros(len(centers))
        for k, name in enumerate(phys_names):
            _, param_binned = fast_bin_mean(wave_np, grad_sq_all[name], bw)
            binned += np.sqrt(np.maximum(param_binned, 0)) * dq_dtheta[k]
        rows.append((centers, binned))
    scalograms[q_short] = rows
    print(f"  {q_label}: done")

print("All scalograms computed.")

# %%
# Interpolate to common grid for 2D visualization
wave_common = np.linspace(float(wave_np[0]) + 100, float(wave_np[-1]) - 100, 500)

scalogram_2d = {}
for prop_name, rows in scalograms.items():
    arr = np.zeros((n_scales, len(wave_common)))
    for s, (centers, binned) in enumerate(rows):
        arr[s, :] = np.interp(wave_common, centers, binned, left=0, right=0)
    mx = arr.max()
    if mx > 0:
        arr /= mx
    scalogram_2d[prop_name] = arr

# %% [markdown]
# ## Step 4: Individual Scalogram Figures
#
# One figure per property.  Spectral features are drawn as labeled
# vertical lines — no SDSS band markers.

# %%
# Comprehensive spectral features (rest-frame → observed at z=0.1)
features_to_label = {
    r"Ly$\alpha$": 1216.0,
    "C IV": 1549.0,
    "C III]": 1909.0,
    "Mg II": 2798.0,
    "[O II]": 3727.0,
    "Ca K": 3934.0,
    "Ca H": 3969.0,
    "D4000": 4000.0,
    r"H$\delta$": 4102.0,
    r"H$\gamma$": 4340.0,
    r"H$\beta$": 4861.0,
    "[O III]": 5007.0,
    "Mg b": 5175.0,
    "Fe 5270": 5270.0,
    "Na D": 5893.0,
    "[O I]": 6300.0,
    "[N II]": 6548.0,
    r"H$\alpha$": 6563.0,
    "[S II]": 6717.0,
    "Ca II": 8542.0,
    "Pa$\\delta$": 10049.0,
}

tick_vals = [2, 5, 10, 20, 50, 100, 200, 500, 1000, 3000]

def plot_scalogram(scalo, title, filename, cmap="magma"):
    """Plot a single scalogram with labeled spectral features."""
    fig, ax = plt.subplots(figsize=(12, 5))

    im = ax.pcolormesh(wave_common, np.log10(bin_widths),
                        scalo, cmap=cmap, vmin=0, vmax=1, rasterized=True)

    # Labeled spectral features
    for feat_name, lam_rest in features_to_label.items():
        lam_obs = lam_rest * (1 + z)
        if float(wave_common[0]) <= lam_obs <= float(wave_common[-1]):
            ax.axvline(lam_obs, color="white", ls="-", lw=0.8, alpha=0.7)
            ax.text(lam_obs + 30, np.log10(bin_widths[-1]) - 0.1,
                    feat_name, fontsize=8, color="white", rotation=90,
                    va="top", ha="left", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.4))

    ax.set_yticks(np.log10(tick_vals))
    ax.set_yticklabels([str(t) for t in tick_vals])
    ax.set_xlabel(r"Observed wavelength [$\AA$]", fontsize=13)
    ax.set_ylabel(r"Window width $\Delta\lambda$ [$\AA$]", fontsize=13)
    ax.set_title(title, fontsize=14)

    plt.colorbar(im, ax=ax, label="Normalized gradient power", pad=0.02)
    plt.tight_layout()
    plt.savefig(f"notebook_figures/{filename}", dpi=72, bbox_inches="tight")
    plt.show()

# %%
# --- One figure per physical parameter ---
for name in phys_names:
    label = name.replace("_", " ")
    plot_scalogram(
        scalogram_2d[name],
        f"Gradient Sensitivity: {label}",
        f"10_multiscale_{name}.png",
    )

# %%
# --- One figure per derived quantity ---
for q_short, q_label in zip(dq_names_short, derived_names):
    plot_scalogram(
        scalogram_2d[q_short],
        f"Information Content: {q_label}",
        f"10_multiscale_{q_short}.png",
        cmap="inferno",
    )

# %% [markdown]
# ## Interpreting the Scalograms
#
# **Physical parameters:**
# - **Dust** (`dust_tau_bc`, `dust_tau_diff`): Strong gradients at all
#   scales in the UV/blue — dust reddening is a broadband effect.
#   Birth-cloud dust may show feature-scale structure near H$\alpha$.
# - **Metallicity** (`met_logzsol`): Concentrated near D4000, Mg b, and
#   metal absorption lines — requires $R \gtrsim 50$ to resolve.
# - **SFH shape** (`sfh_dpl_alpha`, `sfh_dpl_beta`, `sfh_dpl_tau_gyr`):
#   Intermediate scales — the continuum shape changes with SFH.
# - **PSD parameters** (`sfh_field_psd_sigma`, `sfh_field_psd_tau_myr`): Burstiness affects
#   recent SFH most, so gradients concentrate in the UV and near
#   Balmer features.
#
# **Derived quantities:**
# - **$M_*$**: Dominated by NIR/red continuum at all scales — mass is
#   an integrated quantity carried by old stars.
# - **SFR$_{100}$**: UV continuum at intermediate scales + Balmer break.
# - **SFR$_{10}$**: H$\alpha$ and UV at fine scales — only detectable
#   with spectroscopy or narrow-band imaging.

# %%
# --- Summary table: which scales carry information ---
print("Scale dependence summary:")
print(f"{'Property':<25} {'Broadband':>12} {'Intermediate':>12} {'Features':>12}")
print(f"{'':25s} {'(>500 Å)':>12} {'(50-500 Å)':>12} {'(<50 Å)':>12}")
print("-" * 65)

all_props = phys_names + dq_names_short
all_labels = phys_names + [n.replace("$", "").replace("\\log ", "log ") for n in derived_names]

for prop, label in zip(all_props, all_labels):
    s = scalogram_2d[prop]
    bb_mask = bin_widths >= 500
    mid_mask = (bin_widths >= 50) & (bin_widths < 500)
    feat_mask = bin_widths < 50
    bb = np.mean(s[bb_mask, :])
    mid = np.mean(s[mid_mask, :])
    feat = np.mean(s[feat_mask, :])
    total = bb + mid + feat
    if total > 0:
        print(f"  {label:<25} {bb/total:>11.0%} {mid/total:>12.0%} {feat/total:>12.0%}")

# %% [markdown]
# ## What You've Learned
#
# 1. **Gradient sensitivity varies with spectral scale** — not all
#    information lives at the same resolution
# 2. **Dust and SFR normalization** are constrained by broadband SED
#    shape (photometry suffices)
# 3. **Metallicity** requires feature-resolved observations ($R > 50$)
#    to capture D4000 and metal absorption
# 4. **PSD burstiness parameters** are best constrained by UV continuum
#    and Balmer features at intermediate resolution
# 5. **$M_*$** is carried by red/NIR light at all scales; **SFR$_{10}$**
#    requires fine-scale H$\alpha$ or UV features
#
# **See also:**
# - **NB02**: the broadband Jacobian (bottom row of these scalograms)
# - **NB06**: progressive data reveal showing the posterior perspective
# - **NB08**: window functions connecting observables to SFH timescales
