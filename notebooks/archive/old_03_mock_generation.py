# ---
# jupyter:
#   jupytext:
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
# # Tutorial 3: Generating Mock Galaxies
#
# Before fitting real data, we validate the pipeline on synthetic observations with known ground truth. This notebook generates mock galaxies at different redshifts, burstiness levels, and noise levels -- the test bed for Tutorials 4-5.
#
# **What you will learn:**
#
# | Section | Topic | Key concept |
# |---------|-------|-------------|
# | 1 | A single mock galaxy | `model.mock()` with known truth |
# | 2 | Effect of signal-to-noise ratio | How noise degrades parameter recovery |
# | 3 | A zoo of galaxy types | Archetypes spanning young/old, dusty/clean, bursty/smooth |
# | 4 | A mock population | Ensemble statistics and color-color diagrams |
# | 5 | Batch performance | Wall-time benchmarks for `mock_batch()` |
# | 6 | Mock spectroscopy | 200 pixels vs 5 bands: breaking degeneracies |
#
# **Prerequisite:** [Tutorial 2: From SFH to Observable SED](02_forward_model.ipynb) covers the forward model components (SSP, dust, photometry). Here we use those components as a black box to generate test data.
#
# > **Note:** This tutorial uses the high-level `SEDModel`/`ParamSpec` API with descriptive parameter names (`sfh_alpha`, `psd_sigma`, `met_logzsol`, etc.). See the [Quickstart](00_quickstart.ipynb) for a minimal example and [Tutorial 4](04_fitting.ipynb) for fitting.

# %%
# %matplotlib inline

import sys
sys.path.insert(0, "../src")

from tengri.utils.devices import setup_jax
setup_jax()

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 11,
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "legend.framealpha": 0.9,
})

os.makedirs("figures", exist_ok=True)

# ---- New high-level API imports ----
from tengri import SEDModel, ParamSpec, Uniform, Gaussian, Fixed, load_ssp_data, load_filter_set
from tengri.observation.photometry import ab_mag_from_flux
from tengri.utils.cosmology import luminosity_distance

# ---- Load SSP templates ----
SSP_PATH = "../data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
ssp_data = load_ssp_data(SSP_PATH)

# ---- Load SDSS filters from SVO ----
band_names = ["u", "g", "r", "i", "z"]
filters = load_filter_set(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
filter_curves = filters[2]  # list of FilterCurve namedtuples
band_centers_arr = jnp.array([float(jnp.mean(fc.wave)) for fc in filter_curves])

filter_colors = {"u": "#7b3294", "g": "#008837", "r": "#d73027",
                 "i": "#fc8d59", "z": "#4d4d4d"}

print(f"SSP templates: {ssp_data.ssp_wave.shape[0]:,} wavelengths, "
      f"{ssp_data.ssp_lg_age_gyr.shape[0]} ages, "
      f"{ssp_data.ssp_lgmet.shape[0]} metallicities")
print(f"Filters: {len(band_names)} SDSS bands ({', '.join(band_names)})")
print("Setup complete.")

# %% [markdown]
# ---
# ## 1. A Single Mock Galaxy at z = 0.1
#
# We start with a single galaxy at $z = 0.1$ with moderate burstiness ($\sigma_{\rm PSD} = 1.5$, $\tau_{\rm PSD} = 30$ Myr). The `model.mock()` method runs the full forward model to produce noiseless photometry, then adds Gaussian noise at a specified signal-to-noise ratio. The true parameters are known exactly -- this is our ground truth for testing parameter recovery in Tutorials 4-5.
#
# The left panel shows the star formation history in **linear lookback time** (Gyr). In the double power-law convention used by `tengri`, `sfh_alpha` controls the declining phase after peak SFR (cosmic time), and `sfh_beta` controls the rising phase. The dashed line is the smooth mean SFH; the solid line includes stochastic GP fluctuations from the DRW power spectrum.

# %%
# ---- Define the parameter specification ----
spec = ParamSpec(
    sfh_alpha=1.5,            # DPL falling slope (cosmic time)
    sfh_beta=0.8,             # DPL rising slope (cosmic time)
    sfh_tau_peak_gyr=2.0,     # DPL turnover time (Gyr)
    sfh_peak_sfr=5.0,         # Peak SFR (M_sun/yr)
    psd_sigma=1.5,            # PSD amplitude (dex)
    psd_tau_myr=30.0,         # PSD damping timescale (Myr)
    met_logzsol=-0.5,         # log10(Z/Z_sun)
    dust_tau_bc=1.0,          # Birth cloud optical depth
    dust_tau_diff=0.3,        # Diffuse ISM optical depth
    dust_slope=-0.7,          # Dust power-law index
    redshift=0.1,
    stochastic=True,
    n_grid=256,
)

# ---- Build SEDModel ----
model = SEDModel(spec, ssp_data, filters=filters)

# ---- Sample parameters (draws psd_xi from N(0,I)) ----
params = spec.sample(jax.random.PRNGKey(42))

# ---- Generate mock photometry at SNR=20 ----
mock = model.mock(params, snr=20.0, key=jax.random.PRNGKey(0))

# ---- Reconstruct the SFH for plotting ----
sfh = model.predict_sfh(params)

# ---- Compute the full observed-frame SED for the background ----
z = 0.1
sed_rest = model.predict_sed(params)
wave_rest = ssp_data.ssp_wave
wave_obs = wave_rest * (1.0 + z)
dl_cm = luminosity_distance(z)
flux_scale = (1.0 + z) / (4.0 * jnp.pi * dl_cm**2)
sed_obs = flux_scale * sed_rest

print(f"True photometry (5 bands, noiseless):")
mags = ab_mag_from_flux(mock.flux_true)
for i, b in enumerate(band_names):
    print(f"  {b}: {float(mock.flux_true[i]):.3e} erg/s/cm2/Hz  "
          f"(AB = {float(mags[i]):.2f})")

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

# ---- Left panel: Star Formation History ----
ax = axes[0]
ax.plot(sfh["t_gyr"], sfh["sfr_full"], color="#2166ac", lw=2.5, label="Full SFH (mean + GP)")
ax.plot(sfh["t_gyr"], sfh["sfr_mean"], "k--", lw=1.5, alpha=0.3, label="Mean SFH")
ax.set_xlabel("Lookback time [Gyr]", fontsize=12)
ax.set_ylabel(r"SFR [M$_{\odot}$/yr]", fontsize=12)
ax.set_title("True Star Formation History", fontsize=12)
ax.set_xlim(0, 13.5)
ax.set_yscale("log")
ax.set_ylim(0.01, 200)
ax.legend(fontsize=9)
ax.text(
    0.02, 0.95,
    f"$\\sigma_\\mathrm{{PSD}}={float(params['psd_sigma']):.1f}$, "
    f"$\\tau_\\mathrm{{PSD}}={float(params['psd_tau_myr']):.0f}$ Myr",
    transform=ax.transAxes, fontsize=9, va="top",
)

# ---- Right panel: Photometry with SED background ----
ax = axes[1]
mask_obs = (wave_obs > 2500) & (wave_obs < 12000)
ax.plot(wave_obs[mask_obs], sed_obs[mask_obs],
        color="#999999", lw=1.2, alpha=0.5, label="SED", zorder=1)
ax.errorbar(
    band_centers_arr, mock.flux_obs, yerr=mock.noise,
    fmt="o", ms=8, capsize=4, color="#d73027", lw=2,
    label="Observed (SNR=20)", zorder=5,
)
ax.plot(band_centers_arr, mock.flux_true,
        "ks", ms=10, zorder=10, label="Truth (noiseless)")
ax.set_xlabel(r"Observed wavelength [$\AA$]", fontsize=12)
ax.set_ylabel(r"$f_\nu$ [erg/s/cm$^2$/Hz]", fontsize=12)
ax.set_title(f"Mock photometry at $z = {z}$", fontsize=12)
ax.set_xticks(band_centers_arr)
ax.set_xticklabels(band_names, fontsize=11)
ax.legend(fontsize=8, loc="upper left")

plt.tight_layout()
plt.savefig("figures/03_single_mock.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"Redshift: z = {z}")
print(f"At SNR=20, the 5-band photometry has ~5% uncertainties per band.")
print(f"The SED (gray) shows what a spectrograph would measure.")

# %% [markdown]
# ---
# ## 2. Effect of Signal-to-Noise Ratio
#
# SNR determines how well physical parameters can be recovered from the photometry. SNR $= 5$ is typical for faint high-$z$ galaxies in JWST surveys; SNR $= 50$ is achievable for bright SDSS galaxies. At low SNR, the error bars are large and degenerate parameters (dust, metallicity) become unconstrained -- the SED shape is buried in noise. At high SNR, the photometric points tightly pin down the SED shape, enabling precise recovery of even weakly constrained parameters.
#
# Below we show the same galaxy observed at five different SNR levels. The gray background SED provides context for where the filter measurements fall relative to the full spectral energy distribution.

# %%
fig, ax = plt.subplots(figsize=(10, 5))

# Background SED
ax.plot(wave_obs[mask_obs], sed_obs[mask_obs],
        color="#999999", lw=1.2, alpha=0.4, zorder=0, label="SED")

# Mock photometry at different SNR values
snr_configs = [
    (5,   "#d73027", -100, "o"),
    (10,  "#fc8d59",  -50, "s"),
    (20,  "#91bfdb",    0, "D"),
    (50,  "#4575b4",   50, "^"),
    (100, "#2d004b",  100, "v"),
]

for snr, color, offset, marker in snr_configs:
    m = model.mock(
        params,
        snr=float(snr),
        key=jax.random.PRNGKey(snr),
    )
    ax.errorbar(
        band_centers_arr + offset, m.flux_obs, yerr=m.noise,
        fmt=marker, ms=6, capsize=3, lw=1.5,
        color=color, label=f"SNR = {snr}", zorder=5,
    )

# Truth
ax.plot(band_centers_arr, mock.flux_true,
        "ks", ms=10, zorder=10, label="Truth")

ax.set_xlabel(r"Observed wavelength [$\AA$]", fontsize=12)
ax.set_ylabel(r"$f_\nu$ [erg/s/cm$^2$/Hz]", fontsize=12)
ax.set_title("Same galaxy at different signal-to-noise ratios", fontsize=12)
ax.set_xticks(band_centers_arr)
ax.set_xticklabels(band_names, fontsize=11)
ax.legend(fontsize=8, ncol=2, loc="upper left")

plt.tight_layout()
plt.savefig("figures/03_snr_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

print("SNR=5: the u-band scatter is enormous -- dust and age are")
print("nearly impossible to disentangle with 5-band photometry alone.")
print("SNR=100: the SED shape is tightly constrained.")
print("Tutorials 4-5 will quantify how SNR affects parameter recovery.")

# %% [markdown]
# ---
# ## 3. A Zoo of Galaxy Types
#
# Different galaxy types occupy different regions of parameter space. We create four archetypes spanning the key axes of variation: young/old, dusty/clean, and bursty/smooth. This tests that the forward model produces physically reasonable SEDs across the full range of galaxy properties.
#
# The four archetypes are:
#
# | Galaxy type | Key properties | Physical interpretation |
# |-------------|---------------|------------------------|
# | **Young starburst** | $\sigma_{\rm PSD}=2.5$, $\tau_{\rm PSD}=10$ Myr, low dust | Highly bursty, UV-bright, like local dwarf starbursts |
# | **Cosmic noon SF** | $\sigma_{\rm PSD}=1.5$, $\tau_{\rm PSD}=50$ Myr, moderate dust | Typical $z \sim 2$ star-forming galaxy on the main sequence |
# | **Dusty star-forming** | $\tau_{\rm BC}=4.0$, $\tau_{\rm diff}=1.5$ | Heavily obscured, UV-suppressed, like submillimeter galaxies |
# | **Quiescent** | $\sigma_{\rm PSD}=0.5$, $\alpha=3.0$, low dust | Smooth, declining SFH with a strong 4000 $\text{\AA}$ break |

# %%
# ---- Define galaxy archetypes as ParamSpecs ----
galaxy_zoo = {
    "Young starburst": {
        "spec_kwargs": dict(
            sfh_alpha=0.5, sfh_beta=2.0, sfh_tau_peak_gyr=0.5, sfh_peak_sfr=20.0,
            psd_sigma=2.5, psd_tau_myr=10.0,
            met_logzsol=-1.0, dust_tau_bc=0.5, dust_tau_diff=0.1, dust_slope=-0.7,
            redshift=0.1, stochastic=True, n_grid=256,
        ),
        "color": "#2166ac", "ls": "-",
    },
    "Cosmic noon SF": {
        "spec_kwargs": dict(
            sfh_alpha=1.5, sfh_beta=0.8, sfh_tau_peak_gyr=2.0, sfh_peak_sfr=10.0,
            psd_sigma=1.5, psd_tau_myr=50.0,
            met_logzsol=-0.5, dust_tau_bc=1.5, dust_tau_diff=0.5, dust_slope=-0.7,
            redshift=0.1, stochastic=True, n_grid=256,
        ),
        "color": "#67a9cf", "ls": "-",
    },
    "Dusty star-forming": {
        "spec_kwargs": dict(
            sfh_alpha=1.0, sfh_beta=1.0, sfh_tau_peak_gyr=3.0, sfh_peak_sfr=15.0,
            psd_sigma=1.0, psd_tau_myr=30.0,
            met_logzsol=-0.3, dust_tau_bc=4.0, dust_tau_diff=1.5, dust_slope=-0.7,
            redshift=0.1, stochastic=True, n_grid=256,
        ),
        "color": "#ef8a62", "ls": "-",
    },
    "Quiescent": {
        "spec_kwargs": dict(
            sfh_alpha=3.0, sfh_beta=1.0, sfh_tau_peak_gyr=5.0, sfh_peak_sfr=2.0,
            psd_sigma=0.5, psd_tau_myr=200.0,
            met_logzsol=-0.3, dust_tau_bc=0.2, dust_tau_diff=0.1, dust_slope=-0.7,
            redshift=0.1, stochastic=True, n_grid=256,
        ),
        "color": "#b2182b", "ls": "-",
    },
}

print("Galaxy archetypes defined:")
for name, gal in galaxy_zoo.items():
    kw = gal["spec_kwargs"]
    print(f"  {name}: psd_sigma={kw['psd_sigma']}, "
          f"psd_tau_myr={kw['psd_tau_myr']:.0f}, "
          f"dust_tau_bc={kw['dust_tau_bc']}, dust_tau_diff={kw['dust_tau_diff']}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

wave_rest = ssp_data.ssp_wave
mask_sed = (wave_rest > 1000) & (wave_rest < 30000)
idx_1um = int(jnp.argmin(jnp.abs(wave_rest - 10000)))

for i, (name, gal) in enumerate(galaxy_zoo.items()):
    # Build a SEDModel for each archetype
    spec_i = ParamSpec(**gal["spec_kwargs"])
    model_i = SEDModel(spec_i, ssp_data, filters=filters)
    params_i = spec_i.sample(jax.random.PRNGKey(i * 7))

    # Compute SFH via predict_sfh
    sfh_i = model_i.predict_sfh(params_i)
    axes[0].plot(sfh_i["t_gyr"], sfh_i["sfr_full"], lw=2.5, color=gal["color"], label=name)

    # Compute SED (rest-frame, normalized at 1 micron)
    sed_i = model_i.predict_sed(params_i)
    sed_norm = sed_i / sed_i[idx_1um]
    axes[1].plot(wave_rest[mask_sed], sed_norm[mask_sed],
                 lw=2.5, color=gal["color"], label=name)

# ---- Left: SFHs ----
ax = axes[0]
ax.set_xlabel("Lookback time [Gyr]", fontsize=12)
ax.set_ylabel(r"SFR [M$_{\odot}$/yr]", fontsize=12)
ax.set_title("Star Formation Histories", fontsize=12)
ax.set_xlim(0, 13.5)
ax.set_yscale("log")
ax.set_ylim(1e-3, 500)
ax.legend(fontsize=8, loc="lower left")

# ---- Right: SEDs ----
ax = axes[1]
ax.set_xlabel(r"Rest-frame wavelength [$\AA$]", fontsize=12)
ax.set_ylabel(r"Normalized $L_\nu$ (at 1 $\mu$m)", fontsize=12)
ax.set_title("Rest-frame SEDs", fontsize=12)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(1000, 30000)
ax.legend(fontsize=8, loc="lower right")

plt.tight_layout()
plt.savefig("figures/03_galaxy_zoo.png", dpi=150, bbox_inches="tight")
plt.show()

print("Young starburst: UV-bright, bursty SFH, steep blue UV slope.")
print("Dusty SF: UV is heavily suppressed by dust_tau_bc=4.0 birth-cloud dust.")
print("Quiescent: red SED dominated by old stars, smooth declining SFH.")
print("Cosmic noon SF: intermediate in all respects.")

# %% [markdown]
# ---
# ## 4. A Mock Population
#
# For population-level tests, we draw parameters from broad priors using `spec.sample_batch()` and generate an ensemble of 20 galaxies with `model.mock_batch()`. The resulting color-color diagram shows the expected diversity: blue star-forming galaxies in the lower-left, red quiescent galaxies in the upper-right, with burstier galaxies ($\sigma_{\rm PSD} > 2$) scattering further from the mean locus. The SFH spaghetti plot on the right illustrates the enormous diversity in star formation histories that can arise from the GP + PSD model.

# %%
n_gal = 200

# ---- Define a ParamSpec with broad priors ----
pop_spec = ParamSpec(
    sfh_alpha=Uniform(0.3, 3.0),
    sfh_beta=Uniform(0.3, 2.0),
    sfh_tau_peak_gyr=Uniform(0.3, 10.0),
    sfh_peak_sfr=Uniform(0.3, 30.0),
    psd_sigma=Uniform(0.3, 3.0),
    psd_tau_myr=Uniform(3.0, 300.0),
    met_logzsol=Uniform(-2.5, -1.3),
    dust_tau_bc=Uniform(0.0, 3.0),
    dust_tau_diff=Uniform(0.0, 1.0),
    dust_slope=-0.7,
    redshift=0.1,
    stochastic=True,
    n_grid=256,
)

pop_model = SEDModel(pop_spec, ssp_data, filters=filters)

# ---- Sample a batch of parameters ----
pop_params = pop_spec.sample_batch(jax.random.PRNGKey(42), n=n_gal)

# ---- Generate mock photometry for the whole batch ----
pop_mocks = pop_model.mock_batch(pop_params, snr=20.0, key=jax.random.PRNGKey(99))

# ---- Compute SFHs for plotting ----
pop_sfhs = []
for i in range(n_gal):
    p_i = {k: v[i] for k, v in pop_params.items()}
    sfh_i = pop_model.predict_sfh(p_i)
    pop_sfhs.append(sfh_i)

# Compute u-g and g-r colors (AB magnitudes)
pop_fluxes = pop_mocks.flux_true  # shape (n_gal, n_bands)
ug = -2.5 * jnp.log10(pop_fluxes[:, 0] / pop_fluxes[:, 1])
gr = -2.5 * jnp.log10(pop_fluxes[:, 1] / pop_fluxes[:, 2])
sigma_ps_vals = pop_params["psd_sigma"]

print(f"Generated {n_gal} galaxies with sample_batch + mock_batch.")
print(f"  psd_sigma range: {float(sigma_ps_vals.min()):.2f} -- "
      f"{float(sigma_ps_vals.max()):.2f}")
print(f"  u-g color range: {float(ug.min()):.2f} -- {float(ug.max()):.2f} mag")
print(f"  g-r color range: {float(gr.min()):.2f} -- {float(gr.max()):.2f} mag")

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

# ---- Left: color-color diagram ----
ax = axes[0]
sc = ax.scatter(
    gr, ug, c=sigma_ps_vals, cmap="RdYlBu_r",
    s=60, edgecolors="k", linewidths=0.5, zorder=5,
    vmin=0.3, vmax=3.0,
)
plt.colorbar(sc, ax=ax, label=r"$\sigma_{\rm PSD}$", shrink=0.85)
ax.set_xlabel("g - r [mag]", fontsize=12)
ax.set_ylabel("u - g [mag]", fontsize=12)
ax.set_title(f"{n_gal} mock galaxies: color-color", fontsize=12)

# ---- Right: SFH spaghetti ----
ax = axes[1]
for i in range(n_gal):
    alpha_val = 0.3 + 0.4 * (float(sigma_ps_vals[i]) - 0.3) / 2.7
    ax.plot(pop_sfhs[i]["t_gyr"], pop_sfhs[i]["sfr_full"],
            alpha=alpha_val, lw=1.0, color="#2166ac")
ax.set_xlabel("Lookback time [Gyr]", fontsize=12)
ax.set_ylabel(r"SFR [M$_{\odot}$/yr]", fontsize=12)
ax.set_title("SFH diversity in the population", fontsize=12)
ax.set_xlim(0, 13.5)
ax.set_yscale("log")
ax.set_ylim(1e-3, 1e4)

plt.tight_layout()
plt.savefig("figures/03_mock_population.png", dpi=150, bbox_inches="tight")
plt.show()

print("Burstier galaxies (red dots) scatter further in color-color space.")
print("The SFH spaghetti plot shows the enormous diversity the GP model can produce.")

# %% [markdown]
# ---
# ## 5. Batch Performance Benchmarks
#
# The `mock_batch()` method generates multiple mock galaxies in a single call. Below we benchmark wall time as a function of batch size. The first call includes JIT compilation overhead; subsequent calls show the amortized cost per galaxy.

# %%
import time

batch_sizes = [1, 10, 50, 100, 500]
timings = []

for n in batch_sizes:
    batch_params = pop_spec.sample_batch(jax.random.PRNGKey(0), n=n)

    # Warmup / JIT compilation (first call)
    _ = pop_model.mock_batch(batch_params, snr=20.0, key=jax.random.PRNGKey(0))

    # Timed run
    t0 = time.perf_counter()
    _ = pop_model.mock_batch(batch_params, snr=20.0, key=jax.random.PRNGKey(1))
    dt = time.perf_counter() - t0
    timings.append(dt)
    print(f"  n={n:>4d}:  {dt:.3f} s  ({dt/n*1000:.1f} ms/galaxy)")

print(f"\nBatch sizes: {batch_sizes}")
print(f"Wall times:  {['%.3f s' % t for t in timings]}")
print(f"Per-galaxy:  {['%.1f ms' % (t/n*1000) for t, n in zip(timings, batch_sizes)]}")

# %% [markdown]
# ---
# ## 6. Mock Spectroscopy
#
# Spectroscopy provides $\sim 200$ independent measurements per galaxy (vs 5 for broadband photometry), resolving individual spectral features like the 4000 $\text{\AA}$ break, Balmer absorption/emission lines, and forbidden-line emission. This dramatically improves parameter recovery -- especially for breaking the age-dust-metallicity degeneracy that plagues photometric SED fitting.
#
# We generate a mock spectrum covering 4000--9000 $\text{\AA}$ in the observed frame at $z = 0.1$ with SNR $= 15$ per pixel using `model.predict_spectrum()`. Key diagnostic features are labeled: the 4000 $\text{\AA}$ break (age/metallicity), H$\beta$ and H$\alpha$ (SFR and dust via the Balmer decrement), and [O III] $\lambda 5007$ (ionization and metallicity).

# %%
# ---- Generate mock spectrum ----
wave_spec = jnp.linspace(4000, 9000, 250)  # observed frame
spec_true = model.predict_spectrum(params, wave_spec)

# Add Gaussian noise at SNR=15 per pixel
snr_spec = 15.0
spec_noise = spec_true / snr_spec
key_spec = jax.random.PRNGKey(7)
spec_obs = spec_true + spec_noise * jax.random.normal(
    key_spec, shape=spec_true.shape
)

print(f"Spectrum: {len(wave_spec)} pixels, "
      f"{float(wave_spec[0]):.0f}--{float(wave_spec[-1]):.0f} Angstrom")
print(f"SNR per pixel: {snr_spec}")
print(f"Rest-frame coverage: {float(wave_spec[0])/(1+z):.0f}--"
      f"{float(wave_spec[-1])/(1+z):.0f} Angstrom")

# %%
fig, ax = plt.subplots(figsize=(10, 4.5))

# Noisy observed spectrum
ax.plot(wave_spec, spec_obs, color="#bbbbbb", lw=0.6, alpha=0.8,
        label=f"Observed (SNR={snr_spec:.0f})", zorder=1)

# Truth
ax.plot(wave_spec, spec_true, color="#d73027", lw=2.5,
        label="Truth", zorder=3)

# Noise envelope (1-sigma)
ax.fill_between(
    wave_spec, spec_true - spec_noise, spec_true + spec_noise,
    color="#d73027", alpha=0.12, zorder=2,
)

# ---- Label key spectral features ----
z = 0.1
features = [
    (4000 * (1 + z), r"4000$\,\AA$ break"),
    (4861 * (1 + z), r"H$\beta$"),
    (5007 * (1 + z), r"[OIII]"),
    (6563 * (1 + z), r"H$\alpha$"),
]
ymax = float(spec_true.max())
for wl, name in features:
    if float(wave_spec[0]) < wl < float(wave_spec[-1]):
        ax.axvline(wl, color="#666666", ls=":", alpha=0.4, lw=1)
        ax.text(wl + 30, ymax * 0.93, name,
                fontsize=8, color="#444444", ha="left", va="top")

ax.set_xlabel(r"Observed wavelength [$\AA$]", fontsize=12)
ax.set_ylabel(r"$f_\nu$ [erg/s/cm$^2$/Hz]", fontsize=12)
ax.set_title(f"Mock spectrum at $z = {z}$ (250 pixels)", fontsize=12)
ax.legend(fontsize=9, loc="lower right")
ax.set_xlim(float(wave_spec[0]), float(wave_spec[-1]))

plt.tight_layout()
plt.savefig("figures/03_mock_spectrum.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"250 spectral pixels vs 5 photometric bands: a 50x increase")
print(f"in independent measurements. The 4000 Angstrom break, Balmer")
print(f"lines, and [OIII] each constrain different physical properties.")

# %% [markdown]
# ---
# ## Summary
#
# This tutorial demonstrated the mock generation capabilities of `tengri` using the `SEDModel`/`ParamSpec` API. The key methods and their use cases are:
#
# | Mock type | Use case | Key method |
# |-----------|----------|------------|
# | Single galaxy | Parameter recovery tests | `model.mock(params, snr, key)` |
# | Different SNR | Forecast data requirements | Vary the `snr` argument |
# | Galaxy zoo | Test across the full parameter space | Define archetype `ParamSpec`s |
# | Population | Ensemble statistics, color-color diagrams | `spec.sample_batch()` + `model.mock_batch()` |
# | Batch benchmark | Performance profiling | `model.mock_batch()` at varying sizes |
# | Spectroscopy | Test spectral fitting, break degeneracies | `model.predict_spectrum(params, wave_obs)` |
#
# **Key takeaways:**
#
# 1. `model.mock()` produces synthetic photometry with known ground truth -- essential for validating any fitting pipeline.
# 2. SNR strongly controls parameter recovery: at SNR $< 10$, degenerate parameters (dust, metallicity, age) are poorly constrained by 5-band photometry alone.
# 3. The GP + PSD model generates physically diverse SFHs spanning young starbursts to quiescent ellipticals.
# 4. `sample_batch()` and `mock_batch()` make it easy to generate ensembles of mock galaxies for population-level tests.
# 5. Spectroscopy provides $\sim 50 \times$ more information than broadband photometry, resolving individual features that break the age-dust-metallicity degeneracy.
#
# **Next:** [Tutorial 4: Fitting Mock Galaxies](04_fitting.ipynb) -- recovering parameters from the mocks generated here.
#
# **Previous:** [Tutorial 2: From SFH to Observable SED](02_forward_model.ipynb)
