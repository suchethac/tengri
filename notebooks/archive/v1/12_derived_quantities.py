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
# # Derived Physical Quantities
#
# tengri computes a comprehensive set of derived quantities from the
# forward model — from stellar masses and SFRs to UV slopes, emission
# line diagnostics, and radio/X-ray scaling relations.
#
# **Two usage modes:**
#
# 1. **Lazy exploration** — `model.predict(params)` returns a `Prediction`
#    object where quantities are computed on demand and cached.
# 2. **Batch computation** — `jax.vmap(model.predict_sfh_quantities)` for
#    vectorized computation over posterior chains or mock catalogs.
#
# This notebook demonstrates both modes with diagnostic plots.

# %% [markdown]
# ## Setup

# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

from tengri import SEDModel, ParamSpec, Uniform
from tengri.sps.dsps_wrapper import load_ssp_data

# %%
ssp = load_ssp_data("../data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")

spec = ParamSpec(
    mean_sfh_type="tsnorm",
    sfh_tsnorm_log_peak_sfr=Uniform(-1.0, 2.5),
    sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),
    sfh_tsnorm_width_gyr=Uniform(0.2, 5.0),
    sfh_tsnorm_skew=Uniform(-1.0, 1.0),
    sfh_tsnorm_trunc=Uniform(1.0, 10.0),
    met_logzsol=Uniform(-2.0, 0.2),
    dust_tau_bc=Uniform(0.0, 2.0),
    dust_tau_diff=Uniform(0.0, 2.0),
    redshift=0.5,
)

model = SEDModel(spec, ssp)

# %% [markdown]
# ## Mode 1: Lazy Single-Galaxy Exploration
#
# `model.predict(params)` returns a `Prediction` object with six
# property groups: `.sfh`, `.sed`, `.lines`, `.radio`, `.xray`, `.ionizing`.
# Each group computes on first access and caches results.

# %%
params = spec.sample(jax.random.PRNGKey(42))
pred = model.predict(params)

# SFH quantities (only SFH computation triggered here)
print("=== SFH Quantities ===")
print(f"  stellar_mass          = {pred.sfh.stellar_mass:.3e} Msun")
print(f"  stellar_mass_surviving= {pred.sfh.stellar_mass_surviving:.3e} Msun")
print(f"  sfr_100myr            = {pred.sfh.sfr_100myr:.3f} Msun/yr")
print(f"  sfr_10myr             = {pred.sfh.sfr_10myr:.3f} Msun/yr")
print(f"  ssfr                  = {pred.sfh.ssfr:.3e} yr^-1")
print(f"  mass_weighted_age     = {pred.sfh.mass_weighted_age_gyr:.2f} Gyr")
print(f"  mass_weighted_Z       = {pred.sfh.mass_weighted_metallicity:.3f} (log10 Z/Zsun)")

# %%
# SED quantities (full SED computation triggered on first access)
print("=== SED Quantities ===")
print(f"  l_bol                 = {pred.sed.l_bol:.3e} Lsun")
print(f"  l_tir                 = {pred.sed.l_tir:.3e} Lsun")
print(f"  l_dust_absorbed       = {pred.sed.l_dust_absorbed:.3e} Lsun")
print(f"  irx                   = {pred.sed.irx:.3f}")
print(f"  uv_slope_beta         = {pred.sed.uv_slope_beta:.3f}")
print(f"  dn4000                = {pred.sed.dn4000:.3f}")
print(f"  balmer_break          = {pred.sed.balmer_break:.3f}")
print(f"  m_uv                  = {pred.sed.m_uv:.2f} mag")
print(f"  fuv_flux              = {pred.sed.fuv_flux:.3e} erg/s/Hz")
print(f"  nuv_flux              = {pred.sed.nuv_flux:.3e} erg/s/Hz")
print(f"  rest U-V              = {pred.sed.rest_uv_color:.3f} mag")
print(f"  lum_weighted_age      = {pred.sed.luminosity_weighted_age_gyr:.2f} Gyr")

# %%
# Radio and X-ray (empirical scaling relations)
print("=== Radio & X-ray ===")
print(f"  l_1.4GHz              = {pred.radio.l_1p4ghz:.3e} erg/s/Hz")
print(f"  l_x_xrb              = {pred.xray.l_x_xrb:.3e} erg/s")

# %%
# Emission lines (NaN without free nebular model)
print("=== Emission Lines ===")
print(f"  halpha                = {pred.lines.halpha}")
print("  (NaN is expected with baked-in nebular SSPs)")

# %% [markdown]
# ## Mode 2: Batch Computation with `jax.vmap`
#
# For computing derived quantities over many parameter sets — posterior
# chains, mock catalogs, or parameter sweeps — use the JIT-compatible
# group methods with `jax.vmap`. This is **much faster** than looping
# over `model.predict()` because JAX compiles and vectorizes the
# computation into a single fused kernel.
#
# The workflow is:
# 1. Sample or stack parameters into a dict of arrays with leading batch dim
# 2. `jax.vmap(model.predict_sfh_quantities)(params_batch)` → `SFHQuantities`
#    with shape `(n_galaxies,)` for each field
# 3. Same for `predict_sed_quantities` → `SEDQuantities`

# %%
# Step 1: Generate a mock population of 500 galaxies
n_galaxies = 500
keys = jax.random.split(jax.random.PRNGKey(0), n_galaxies)
samples = [spec.sample(k) for k in keys]

# Stack into a dict of arrays (leading batch dimension)
params_batch = {k: jnp.stack([s[k] for s in samples]) for k in samples[0]}
print("Batch param shapes:", {k: v.shape for k, v in list(params_batch.items())[:3]}, "...")

# %%
# Step 2: vmap over SFH quantities — fully vectorized, JIT-compiled
sfh_fn = jax.vmap(model.predict_sfh_quantities)
sfh_batch = sfh_fn(params_batch)

print(f"stellar_mass shape: {sfh_batch.stellar_mass.shape}")
print(
    f"stellar_mass range: [{float(sfh_batch.stellar_mass.min()):.2e}, "
    f"{float(sfh_batch.stellar_mass.max()):.2e}] Msun"
)
print(
    f"mass_weighted_age range: [{float(sfh_batch.mass_weighted_age_gyr.min()):.1f}, "
    f"{float(sfh_batch.mass_weighted_age_gyr.max()):.1f}] Gyr"
)

# %%
# Step 3: vmap over SED quantities (runs the full forward model for each galaxy)
sed_fn = jax.vmap(model.predict_sed_quantities)
sed_batch = sed_fn(params_batch)

print(
    f"l_bol range: [{float(sed_batch.l_bol.min()):.2e}, {float(sed_batch.l_bol.max()):.2e}] Lsun"
)
print(f"dn4000 range: [{float(sed_batch.dn4000.min()):.3f}, {float(sed_batch.dn4000.max()):.3f}]")
print(f"m_uv range: [{float(sed_batch.m_uv.min()):.1f}, {float(sed_batch.m_uv.max()):.1f}] mag")

# %%
# Extract numpy arrays for plotting
masses = np.array(sfh_batch.stellar_mass)
ages_mw = np.array(sfh_batch.mass_weighted_age_gyr)
sfrs = np.array(sfh_batch.sfr_100myr)
dn4000s = np.array(sed_batch.dn4000)
betas = np.array(sed_batch.uv_slope_beta)
m_uvs = np.array(sed_batch.m_uv)
uv_colors = np.array(sed_batch.rest_uv_color)
ages_lw = np.array(sed_batch.luminosity_weighted_age_gyr)

# %% [markdown]
# ## Diagnostic Plots

# %%
fig, axes = plt.subplots(2, 3, figsize=(14, 9))

# 1. Star-forming main sequence
ax = axes[0, 0]
ax.scatter(np.log10(masses), np.log10(np.maximum(sfrs, 1e-5)), s=5, alpha=0.5)
ax.set_xlabel(r"$\log_{10}(M_\star / M_\odot)$")
ax.set_ylabel(r"$\log_{10}(\mathrm{SFR} / M_\odot\,\mathrm{yr}^{-1})$")
ax.set_title("Star-forming Main Sequence")

# 2. Dn4000 vs mass-weighted age
ax = axes[0, 1]
ax.scatter(ages_mw, dn4000s, s=5, alpha=0.5)
ax.set_xlabel("Mass-weighted age (Gyr)")
ax.set_ylabel(r"$D_n(4000)$")
ax.set_title(r"$D_n(4000)$ vs Age")

# 3. M_UV distribution (luminosity function proxy)
ax = axes[0, 2]
ax.hist(m_uvs, bins=30, edgecolor="k", alpha=0.7)
ax.set_xlabel(r"$M_{\mathrm{UV}}$ (AB mag)")
ax.set_ylabel("Count")
ax.set_title("UV Luminosity Distribution")
ax.invert_xaxis()

# 4. Mass-weighted vs luminosity-weighted age
ax = axes[1, 0]
ax.scatter(ages_mw, ages_lw, s=5, alpha=0.5, c=np.log10(masses), cmap="viridis")
ax.plot([0, 13], [0, 13], "k--", alpha=0.3, label="1:1")
ax.set_xlabel("Mass-weighted age (Gyr)")
ax.set_ylabel("Luminosity-weighted age (Gyr)")
ax.set_title("Mass vs Light-weighted Age")
ax.legend(fontsize=8)

# 5. UV slope distribution
ax = axes[1, 1]
finite_betas = betas[np.isfinite(betas) & (np.abs(betas) < 10)]
ax.hist(finite_betas, bins=30, edgecolor="k", alpha=0.7)
ax.set_xlabel(r"UV slope $\beta$")
ax.set_ylabel("Count")
ax.set_title(r"UV Slope $\beta$ Distribution")

# 6. Rest-frame U-V color distribution
ax = axes[1, 2]
ax.hist(uv_colors, bins=30, edgecolor="k", alpha=0.7, color="C1")
ax.set_xlabel("Rest-frame U-V (mag)")
ax.set_ylabel("Count")
ax.set_title("Rest-frame U-V Color")

plt.tight_layout()
plt.savefig("figures/12_derived_quantities.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Summary
#
# | Group | Property | Triggers | Description |
# |-------|----------|----------|-------------|
# | `sfh` | `stellar_mass` | SFH | Total formed mass |
# | `sfh` | `stellar_mass_surviving` | SFH | Living + remnants |
# | `sfh` | `sfr_100myr`, `sfr_10myr` | SFH | Recent SFR averages |
# | `sfh` | `ssfr` | SFH | Specific SFR |
# | `sfh` | `mass_weighted_age_gyr` | SFH | Mass-weighted age |
# | `sfh` | `mass_weighted_metallicity` | SFH | Mass-weighted Z |
# | `sfh` | `luminosity_weighted_age_gyr` | SED | L-weighted age |
# | `sfh` | `luminosity_weighted_metallicity` | SED | L-weighted Z |
# | `sed` | `l_bol` | SED | Bolometric luminosity |
# | `sed` | `l_tir` | SED | Total IR 8-1000 um |
# | `sed` | `l_dust_absorbed` | SED | Dust-absorbed luminosity |
# | `sed` | `irx` | SED | IR excess |
# | `sed` | `uv_slope_beta` | SED | UV spectral slope |
# | `sed` | `dn4000` | SED | 4000A break |
# | `sed` | `balmer_break` | SED | Balmer break (Wang+2024) |
# | `sed` | `m_uv` | SED | Absolute UV magnitude |
# | `sed` | `fuv_flux`, `nuv_flux` | SED | FUV/NUV flux densities |
# | `sed` | `rest_uv_color` | SED | Rest-frame U-V |
# | `lines` | `halpha`, `oii`, etc. | Lines | Emission line luminosities |
# | `lines` | `bpt_nii`, `o3hb`, etc. | Lines | Diagnostic ratios |
# | `radio` | `l_1p4ghz` | SFH | 1.4 GHz luminosity |
# | `xray` | `l_x_xrb` | SFH | XRB luminosity |
# | `ionizing` | `q_h`, `xi_ion` | Lines+SED | Ionizing photon budget |
