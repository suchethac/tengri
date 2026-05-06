# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Tengri Quickstart
#
# *Differentiable SED fitting from the UV to the mid-IR, with HMC posteriors
# in seconds.*
#
# **What you'll do.** Build a galaxy with a truncated-skew-normal (`tsnorm`)
# star formation history, attach a two-component dust attenuation law plus
# Dale & Helou (2014) IR emission with full energy balance, and recover
# the posterior with HMC (NUTS) on mock 14-band UV-to-mid-IR photometry.
# Along the way you will see posteriors over the parameters, the SFH(t)
# curve, derived properties (M⋆, SFR, sSFR, mass-weighted age), and the
# SED itself.
#
# **Why tengri.** Every parameter has an analytic gradient. That makes
# Hamiltonian Monte Carlo — usually a luxury for SED fitting — the
# *default* sampler. The forward model and the inference share the same
# JIT-compiled function: a single likelihood evaluation costs ~1 ms.
#
# **Runtime budget.** End-to-end ≈ 30–40 s on a laptop CPU on a fresh
# JIT cache, dominated by compiling the HMC step + energy-balance
# forward model. The *second* run uses tengri's persistent on-disk
# JIT cache and finishes in ~10 s. Filter curves are downloaded from
# the SVO Filter Profile Service on first run and cached locally.
# Two implementation choices keep this fast: (i) plain HMC instead of
# NUTS — NUTS adds a dense-mass-matrix vmap during warmup that
# multiplies compile time roughly 7×; (ii) the hybrid photometry
# kernel that summarises the SSP grid onto a 200-point coarse
# wavelength grid for the energy-balance integral (rather than baking
# the full 11149-point grid into the JIT). See
# `docs/dev/quickstart_oom_diagnosis.md` for the full story.

# %% [markdown]
# ## 1. Setup
#
# `import tengri` enables 64-bit JAX automatically (cosmological distances
# need it) and turns on a persistent on-disk JIT cache so the second time
# you run a model the compile cost is ~100 ms instead of ~10 s.

# %%
# ruff: noqa: I001  -- imports are deliberately grouped to teach the API
import os

# Workaround for a known jetsam-OOM on macOS when fitting models that
# include `dust_emission="dale2014"`: tengri's Fitter spawns a
# background thread that pre-JITs every inference backend (NUTS, MAP,
# geoVI, raytrace, …). On big forward models, the geoVI compile alone
# can push the kernel over macOS' memory-pressure threshold. Setting
# this env var *before* `import tengri` keeps Fitter to lazy,
# on-demand compilation of just the requested backend. Safe to leave
# on for any single-backend run. See
# `docs/dev/quickstart_oom_diagnosis.md`.
os.environ.setdefault("TENGRI_NO_BACKGROUND_COMPILE", "1")

# Standard library + scientific Python.
from pathlib import Path  # robust path resolution for the SSP file
import jax                # JAX provides the PRNG keys driving stochastic ops
import numpy as np        # plotting + post-hoc summaries (NumPy, not JAX)
import matplotlib.pyplot as plt  # all figures use raw matplotlib

# ─── Tengri top-level classes (the five things every analysis uses) ───
# All are re-exported at the package root (see `tengri/__init__.py`).
#
#   Parameters   prior specification + structural choices. Pass priors
#                as kwargs (`sfh_tsnorm_log_peak_sfr=Uniform(-1, 2.5)`)
#                plus structural flags (`mean_sfh_type=...`,
#                `dust_emission=...`). Anything bound to a `Fixed(value)`
#                is treated as fixed; the rest become `spec.free_params`.
#
#   Photometry   immutable filter-set container. Build with
#                `Photometry.from_names([...])` to fetch transmission
#                curves from the SVO Filter Profile Service (cached on
#                first download).
#
#   Observation  bundles photometry / spectroscopy / emission-line
#                measurements. Passed to `SEDModel(...)` so the model
#                knows what observables to compute.
#
#   SEDModel     the JIT-compiled forward model. Holds the SSP grid,
#                the parameter spec, and the observation. Methods
#                (`predict_rest_sed`, `predict_photometry`,
#                `predict_sfh`, `predict_sfh_quantities`, `mock`) are
#                JIT-friendly and `jax.vmap`-friendly.
#
#   Fitter       inference engine. `Fitter(model, data, noise).run(
#                "mcmc_nuts", ...)` returns a `Posterior` (no need to
#                import that class — it's the return value).
from tengri import (
    Fitter,
    Observation,
    Parameters,
    Photometry,
    SEDModel,
)

# ─── Tengri prior distributions ───────────────────────────────────
# Each is a thin frozen dataclass with `.bounds`, `.mean`, `.sample(key)`
# and a `.transform(unit)` method that maps a [0, 1] uniform draw into
# physical-parameter space. Used as values in the `Parameters(...)`
# kwargs. There are more (`LogUniform`, `Gaussian`, `LogNormal`,
# `StudentT`); these two are all this quickstart needs.
from tengri import (
    Fixed,    # `Fixed(value)` — pin a parameter, removed from free_params
    Uniform,  # `Uniform(lo, hi)` — flat prior on the closed interval
)

# ─── Tengri top-level helpers ─────────────────────────────────────
#
#   load_ssp_data   loads a DSPS-format HDF5 SSP grid (BC03 / FSPS /
#                   MIST / BPASS — anything in DSPS layout). Returns
#                   an `SSPData` immutable container fed to `SEDModel`.
#
#   generate_mock   `model.predict_photometry(truth)` + Gaussian noise
#                   at a chosen SNR. Returns `{"flux_true", "flux_obs",
#                   "noise", "params"}` — exactly the four things you
#                   need to wire into a `Fitter`.
from tengri import (
    generate_mock,
    load_ssp_data,
)

# ─── Tengri themed namespaces ─────────────────────────────────────
#
#   tengri.cosmology   distance / age / volume helpers. Defaults to
#                      Planck 2018 (Ω_m = 0.315, h = 0.674). Used here
#                      to convert luminosity → flux.
#                      Key calls used below:
#                        cosmology.luminosity_distance(z)        [cm]
#                        cosmology.luminosity_distance_mpc(z)    [Mpc]
#
#   tengri.units       flux ↔ luminosity ↔ magnitude conversions —
#                      every helper handles the (1+z) factor explicitly.
#                      Key call used below:
#                        units.lnu_to_fnu(lnu, dl_cm, z)
#                          → f_ν = L_ν · (1+z) / (4π d_L²)
#                            (Hogg+1999, AJ 118, 1407)
#                      Other useful entries: fnu_to_ab_mag, fnu_to_jy,
#                      flambda_to_fnu, air_to_vacuum.
#
#   tengri.plot        matplotlib helpers + tengri visual style.
#                      `plot.setup_style()` sets serif font, tight
#                      layout, 150 dpi. Higher-level plotters
#                      (plot_sed_fit, plot_sfh, safe_corner) and
#                      visual constants (COLORS, SDSS_WAVE_EFF,
#                      SPECTRAL_FEATURES) live here.
#
#   tengri.filters     interactive filter discovery. `list_filters()`
#                      returns the SVO short-name registry;
#                      `suggest(z, coverage="visible_to_nir")` proposes
#                      bands covering a rest-frame range at a given
#                      redshift; `describe("sdss_r")` prints λ_eff
#                      and FWHM. Useful when assembling new bandsets.
#
#   tengri.core        forward-looking *protocol* layer (Phase II-1
#                      scaffold, May 2026): `SEDComponent`,
#                      `PipelineState`, `ObservationModel`, `Likelihood`.
#                      Nothing in user-facing tengri *consumes* these
#                      yet — they exist so future Phase II refactors
#                      can migrate one physics module at a time onto a
#                      Protocol-based contract. You won't need them in
#                      this quickstart, but it's worth knowing the
#                      namespace exists.
from tengri import cosmology, plot, units

plot.setup_style()  # serif font, tight layout, 150 dpi

# %% [markdown]
# ## 1.5. Discover what's available
#
# Before building a model, see what physics is on offer.  Every count
# below is read live from the registry — adding a new model via
# `@register_agn_model` updates these numbers immediately.
#
# ```python
# import tengri
# tengri.summary()                            # one-line counts of every menu
# tengri.help()                               # curated cheatsheet (4 sections)
# tengri.help("dust")                         # topical help: 21 attenuation + 7 emission
#
# tengri.list_agn_models()                    # 12 AGN models with citations
# tengri.list_dust_laws(status="production")  # filter by status
# tengri.list_inference_methods(tier="primary")
# tengri.list_filters()                       # 242 filter curves bundled
#
# tengri.describe("skirtor")                  # full metadata for any name
# tengri.search("torus")                      # cross-menu fuzzy search
# tengri.doctor()                             # env / install / SSP health check
# ```
#
# In Jupyter every `list_*()` call returns a real HTML table; in the REPL
# they print as column-aligned ASCII tables.  Each row carries
# `name`, `status`, `citation`, and `short_doc` so a new collaborator can
# learn the menu without leaving the notebook.
#
# Equivalent CLI (no REPL needed):
# ```
# python -m tengri summary
# python -m tengri doctor
# python -m tengri search torus
# python -m tengri describe skirtor
# ```

# %%
import tengri  # noqa: E402

tengri.summary()

# %% [markdown]
# Run `tengri.help()` for the full cheatsheet, or `tengri.help("agn")`
# (or `"dust"`, `"sfh"`, `"nebular"`, `"components"`, `"inference"`,
# `"filters"`) for a topical view of one menu.

# %% [markdown]
# **Stellar population synthesis grid.** Tengri ships with two
# pre-tabulated SSP grids in `data/`. We use the MIST + C3K + Chabrier
# IMF grid — a standard choice for nearby galaxies. The SSP grid is
# precomputed (mass, metallicity, age, wavelength); everything downstream
# is differentiable.

# %%
_ssp_name = "ssp_mist_c3k_a_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
_repo_root = next(
    p for p in [Path.cwd(), *Path.cwd().parents] if (p / "data" / _ssp_name).exists()
)
ssp = load_ssp_data(str(_repo_root / "data" / _ssp_name))

# %% [markdown]
# ## 2. Filters and redshift
#
# Fourteen broad bands spanning the far-UV (GALEX FUV, NUV) through the
# optical (SDSS *u, g, r, i, z*), near-IR (2MASS *J, H, Ks*) and mid-IR
# (WISE *W1–W4*). This is the canonical UV-to-mid-IR set used for
# energy-balance SED fitting of nearby galaxies (e.g. GSWLC, S4G).
# The mid-IR bands are essential: they constrain the dust-emission
# luminosity that closes the energy balance with the UV–optical
# attenuation.
#
# > **Convention.** Photometry is observed-frame; the SED model lives in
# > rest-frame. Tengri handles the redshift internally — you give it
# > observed fluxes, it gives you posteriors on rest-frame physical
# > parameters.

# %%
filter_names = [
    "galex_fuv", "galex_nuv",
    "sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z",
    "2mass_j", "2mass_h", "2mass_ks",
    "wise_w1", "wise_w2", "wise_w3", "wise_w4",
]
photometry = Photometry.from_names(filter_names)
observation = Observation(photometry=photometry)

print(f"Loaded {photometry.n_filters} filters: {', '.join(photometry.names)}")

# %% [markdown]
# ### Figure 1 — Filter throughputs.
# The 14 bandpasses on a log-wavelength axis, colored by facility. This
# is the wavelength real estate the data live in.

# %%
def _facility(name: str) -> str:
    for prefix in ("galex", "sdss", "2mass", "wise"):
        if name.startswith(prefix):
            return prefix
    return "other"


facility_color = {
    "galex": "#7e57c2",
    "sdss": "#1f77b4",
    "2mass": "#2ca02c",
    "wise": "#d62728",
}

fig, ax = plt.subplots(figsize=(6.5, 3.0))
for fc, name in zip(photometry.filters, photometry.names):
    wave_um = np.asarray(fc.wave) / 1e4
    trans = np.asarray(fc.trans) / np.max(fc.trans)
    color = facility_color[_facility(name)]
    ax.plot(wave_um, trans, color=color, alpha=0.85, lw=1.0)

ax.set_xscale("log")
ax.set_xlabel(r"Observed wavelength [$\mu$m]")
ax.set_ylabel("Normalised transmission")
ax.set_xlim(0.13, 30)
ax.set_ylim(0, 1.05)
for fac, color in facility_color.items():
    ax.plot([], [], color=color, label=fac.upper())
ax.legend(frameon=False, ncol=4, loc="upper center", fontsize=9)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 3. The tengri object model
#
# Five classes do all the heavy lifting. Skim this section once; you'll
# refer back to it as you build your own analyses.
#
# - **`Parameters`** — the *prior specification*. It owns three things:
#   (a) the **structural choices** of your model (which SFH family,
#   which dust geometry, whether nebular emission is on, whether IGM is
#   applied, etc.); (b) the **priors** on every free parameter; and
#   (c) the list of **fixed** parameters. The free vs fixed split is
#   inferred from the prior — a `Fixed(value)` is fixed, anything else
#   is free. Construct with kwargs whose names are the canonical
#   parameter strings (e.g. `sfh_tsnorm_log_peak_sfr`,
#   `dust_tau_diff`). Inspect with `spec.free_params`.
#
# - **`Photometry`** and **`Observation`** — the *observation
#   configuration*. `Photometry.from_names([...])` resolves filter short
#   names against the SVO registry and downloads/caches the transmission
#   curves. `Observation(photometry=..., spectroscopy=...)` bundles
#   photometry, spectroscopy, and emission-line measurements into one
#   immutable container that's passed to the model.
#
# - **`SEDModel`** — the *forward model*. Built from `(spec, ssp,
#   observation=...)`. Once constructed, every method on it
#   (`predict_obs_sed`, `predict_photometry`, `predict_sfh`,
#   `predict_sfh_quantities`) is JIT-compiled and `jax.vmap`-friendly.
#   The first call compiles; every subsequent call is ~1 ms.
#
# - **`Fitter`** — the *inference engine*. Built from `Fitter(model,
#   data, noise)`. Method `run("mcmc_nuts", ...)` returns a `Posterior`.
#   Other supported methods include `"map"` (point estimate), `"vi"`
#   (geoVI variational), `"mcmc_raytrace"` (Behroozi 2025 large-D
#   sampler), `"nss"` (nested slice sampling for evidence). Use
#   `auto` to let tengri pick.
#
# - **`Posterior`** — the *result*. Carries `samples` (dict of
#   per-parameter arrays), `params` (posterior-mean dict),
#   `wall_time_s`, `diagnostics`, and computed-on-demand `derived`
#   properties. Method `summary_table()` prints a quick textual report;
#   `plot_corner(truths=...)`, `plot_sed(...)`, `plot_sfh(...)` give
#   matplotlib figures.
#
# Four thin themed namespaces sit alongside these classes for the
# everyday utilities you reach for outside the model itself:
#
# - **`tengri.cosmology`** — `luminosity_distance(z)` (cm),
#   `luminosity_distance_mpc(z)`, `lookback_time(z)`, `age_at_z(z)`,
#   `distance_modulus(z)`. Defaults to Planck 2018; pass `cosmo=...`
#   to override.
# - **`tengri.units`** — flux/luminosity/magnitude conversions:
#   `lnu_to_fnu`, `fnu_to_ab_mag`, `fnu_to_jy`, `flambda_to_fnu`,
#   `air_to_vacuum`, …
# - **`tengri.plot`** — `setup_style()`, `plot_sed_fit`, `plot_sfh`,
#   `safe_corner`, plus visual constants (`COLORS`, `SDSS_WAVE_EFF`).
# - **`tengri.filters`** — interactive filter discovery:
#   `list_filters()`, `suggest(z, coverage=...)`, `describe(name)`.
#
# A fifth namespace, **`tengri.core`**, hosts forward-looking Protocol
# definitions (`SEDComponent`, `PipelineState`, `ObservationModel`,
# `Likelihood`) used by an in-progress refactor. User code does not
# need to interact with it today.

# %% [markdown]
# ## 4. Build the model
#
# Choices for this quickstart, mapped to `Parameters` kwargs:
#
# | Component | Choice | `Parameters` kwarg(s) |
# |---|---|---|
# | **SFH family** | `tsnorm` (truncated skew-normal) | `mean_sfh_type="tsnorm"` plus 5 priors |
# | **Stellar metallicity** | single-Z | `met_logzsol` |
# | **Dust geometry** | two-component (Charlot & Fall 2000) | `dust_model="two_component"` |
# | **Attenuation law** | Calzetti (2000) | `dust_law_bc="calzetti"` |
# | **IR dust emission** | Dale & Helou (2014) + energy balance | `dust_emission="dale2014"` |
# | **Nebular emission** | baked-in (FSPS-style) | (default — no kwargs) |
# | **IGM** | off (negligible at z = 0.05) | `apply_igm=False` |
# | **Redshift** | fixed | `redshift=Fixed(0.05)` |
#
# **Why `tsnorm`?** A truncated skew-normal SFH is parametric (peak
# height, peak time, width, skewness, truncation) but flexible enough
# to capture both rising and declining histories. Five parameters
# control a smooth, physically-motivated curve — easy to fit, easy to
# interpret.
#
# **What does "energy balance" mean?** Dust grains absorb UV–optical
# photons and re-radiate the energy in the infrared. With
# `dust_emission="dale2014"`, tengri uses the Dale & Helou 2014
# semi-empirical IR template family for the spectral shape and
# normalises it so that *L*<sub>IR</sub> equals
# the total absorbed luminosity computed from the attenuation curves
# and the intrinsic stellar SED. UV and IR fluxes are then constrained
# jointly by a single set of dust parameters — the standard
# CIGALE / MAGPHYS approach. Available IR backends:
# `modified_blackbody`, `casey2012`, `dale2014`, `draine_li2007`,
# `dale2014`, `themis`.

# %%
spec = Parameters(
    mean_sfh_type="tsnorm",
    # tsnorm SFH (5 free params)
    sfh_tsnorm_log_peak_sfr=Uniform(-1.0, 2.5),
    sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),
    sfh_tsnorm_width_gyr=Uniform(0.3, 5.0),
    sfh_tsnorm_skew=Uniform(-1.0, 1.0),
    sfh_tsnorm_trunc=Uniform(1.0, 10.0),
    # Stellar metallicity
    met_logzsol=Uniform(-1.5, 0.3),
    # Dust attenuation: two-component (BC + diffuse) with Calzetti law
    dust_model="two_component",
    dust_law_bc="calzetti",
    dust_tau_bc=Uniform(0.0, 2.0),
    dust_tau_diff=Uniform(0.0, 1.5),
    dust_slope=Fixed(-0.7),
    # IR dust emission with energy balance (Draine & Li 2014)
    dust_emission="dale2014",
    # No IGM at z = 0.05 — saves compute and is physically negligible
    apply_igm=False,
    # Redshift fixed (spec-z case; see notebook 03 for free-z fitting)
    redshift=Fixed(0.05),
)

print(f"Free parameters ({len(spec.free_params)}):")
for p in spec.free_params:
    print(f"  - {p}")

# %% [markdown]
# Building the **`SEDModel`** triggers a one-time JIT compilation of the
# forward pass and (if a fixed redshift is set) precomputes the filter
# projection matrices. The first call is slow; every subsequent call
# is ~1 ms.

# %%
model = SEDModel(spec, ssp, observation=observation)

# Luminosity distance for converting model luminosity to observed flux.
# tengri.cosmology defaults to Planck 2018 (Ω_m = 0.315, h = 0.674);
# pass `cosmo=...` to override.
z = 0.05
dl_cm = float(cosmology.luminosity_distance(z))
dl_mpc = float(cosmology.luminosity_distance_mpc(z))
print(f"d_L(z = {z}) = {dl_mpc:.1f} Mpc  ({dl_cm:.3e} cm)")

# %% [markdown]
# ## 5. Forward model — predict an SED
#
# Pick a "truth" galaxy: a nearby star-forming system at z = 0.05 with
# a peak SFR of ≈ 30 M☉/yr four Gyr ago (a perfectly normal main-sequence
# galaxy in the local volume), modest dust, and roughly solar metallicity.

# %%
truth = {
    # SFH — peak SFR = 30 Msun/yr → log10(30) ≈ 1.48
    "sfh_tsnorm_log_peak_sfr": np.log10(30.0),
    "sfh_tsnorm_peak_lbt_gyr": 4.0,
    "sfh_tsnorm_width_gyr": 3.0,
    "sfh_tsnorm_skew": 0.3,
    "sfh_tsnorm_trunc": 5.0,
    # Stars + dust
    "met_logzsol": -0.1,
    "dust_tau_bc": 0.5,
    "dust_tau_diff": 0.3,
    "dust_slope": -0.7,
    "redshift": 0.05,
}

sed_truth = model.predict_rest_sed(truth)       # rest-frame L_nu, rest-frame λ
phot_truth = model.predict_photometry(truth)    # observed flux density [erg/s/cm²/Hz]
sfh_truth = model.predict_sfh(truth)            # SFR(t) on a uniform grid

# %% [markdown]
# **Unit footnote.** `predict_rest_sed` returns rest-frame
# *L*<sub>ν</sub> in erg s⁻¹ Hz⁻¹. `predict_photometry` returns observed
# *f*<sub>ν</sub> in erg s⁻¹ cm⁻² Hz⁻¹. To overlay them we use the
# canonical conversion **`tengri.units.lnu_to_fnu`** —
# *f*<sub>ν</sub> = *L*<sub>ν</sub> (1+*z*) ⁄ (4π *d*<sub>L</sub>²) — and
# shift wavelengths by (1+*z*) into the observed frame.

# %%
wave_obs_um = np.asarray(sed_truth.wavelength) * (1.0 + z) / 1e4
sed_truth_fnu = np.asarray(units.lnu_to_fnu(sed_truth.sed, dl_cm, z))

# Effective wavelengths of the 14 filters (for placing photometric points)
lam_eff_um = np.array([
    np.trapezoid(np.asarray(fc.trans) * np.asarray(fc.wave), np.asarray(fc.wave))
    / np.trapezoid(np.asarray(fc.trans), np.asarray(fc.wave))
    for fc in photometry.filters
]) / 1e4

# %% [markdown]
# ### Figure 2 — Truth galaxy SED + SFH.
# Top: observed-frame *f*<sub>ν</sub> spectrum. The strong rise at
# 5–500 μm is the dust IR re-emission balancing the UV–optical
# attenuation. Bottom: the truncated-skew-normal SFR(*t*) — peak at 4 Gyr
# lookback, current SFR ≈ 30 M☉/yr.

# %%
fig, (ax_sed, ax_sfh) = plt.subplots(2, 1, figsize=(6.5, 5.6))

# --- SED panel ---
ax_sed.plot(wave_obs_um, sed_truth_fnu, color="black", lw=1.2, label="True SED")
ax_sed.errorbar(lam_eff_um, np.asarray(phot_truth),
                fmt="o", ms=4, color="C3", mfc="white", mec="C3",
                label="True photometry")
ax_sed.set_xscale("log")
ax_sed.set_yscale("log")
ax_sed.set_xlim(0.13, 30)
ax_sed.set_xlabel(r"Observed wavelength [$\mu$m]")
ax_sed.set_ylabel(r"$f_\nu$  [erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$]")
ax_sed.legend(frameon=False, loc="lower center", fontsize=9)
ax_sed.set_title(f"z = {truth['redshift']},  peak SFR = 30 M$_\\odot$/yr")

# --- SFH panel ---
t_lookback = np.asarray(sfh_truth["t_gyr"])
sfr_truth_curve = np.asarray(sfh_truth["sfr_mean"])
ax_sfh.plot(t_lookback, sfr_truth_curve, color="black", lw=1.4)
ax_sfh.fill_between(t_lookback, 0, sfr_truth_curve, color="0.85", alpha=0.6)
ax_sfh.set_xlim(0, 13.5)
ax_sfh.set_ylim(bottom=0)
ax_sfh.set_xlabel("Lookback time [Gyr]")
ax_sfh.set_ylabel(r"SFR  [M$_\odot$ yr$^{-1}$]")
ax_sfh.set_title("Star formation history (tsnorm)")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Mock photometry
#
# Add Gaussian noise at SNR = 20 in every band — appropriate for a
# bright, well-detected nearby galaxy. **`generate_mock`** returns the
# noiseless flux, the noise σ, and the noisy realisation.

# %%
mock = generate_mock(model, truth, key=jax.random.PRNGKey(0), snr=20.0)
flux_obs, noise = mock["flux_obs"], mock["noise"]

# %% [markdown]
# ### Figure 3 — The data we will fit.
# True SED (line) plus the 14 noisy photometric points the fitter will
# see. Both are in flux density units, on the same axis.

# %%
fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.plot(wave_obs_um, sed_truth_fnu, color="black", lw=1.0, alpha=0.85,
        label="True SED")
ax.errorbar(lam_eff_um, np.asarray(flux_obs), yerr=np.asarray(noise),
            fmt="o", ms=5, capsize=2, lw=1.0,
            ecolor="0.4", mfc="white", mec="black",
            label="Mock photometry (SNR ≈ 20)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(0.13, 30)
ax.set_xlabel(r"Observed wavelength [$\mu$m]")
ax.set_ylabel(r"$f_\nu$  [erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$]")
ax.legend(frameon=False, loc="lower center")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Fit with HMC (NUTS)
#
# A **`Fitter`** binds the model to the data and the noise. Its `run`
# method dispatches to the requested inference backend — here NUTS via
# BlackJAX. The defaults adapt the step size and a dense mass matrix
# during 800 warmup iterations, then draw 1500 posterior samples in a
# single chain.
#
# > **Method choice.** For posterior-quality work in ≲ 20 dimensions
# > `mcmc_nuts` is the default — it samples the exact posterior up to
# > MCMC error. For a quick point estimate try `"map"`. For higher
# > dimensions (e.g. stochastic SFH ≳ 50 D) switch to `"mcmc_raytrace"`
# > or `"vi"`. Pass `init_from=map_result` to start NUTS from a MAP
# > estimate (saves wall time on hard problems).

# %%
fitter = Fitter(model, data=flux_obs, noise=noise)

result = fitter.run(
    "mcmc_hmc",                # plain HMC: smaller JIT graph than NUTS
    n_warmup=800,
    n_samples=1500,
    target_accept_rate=0.92,
    key=jax.random.PRNGKey(1),
)

print(f"Wall time   : {result.wall_time_s:.1f} s")
print(f"Step size   : {result.diagnostics['step_size']:.3f}")
print(f"Divergences : {result.diagnostics['n_divergent']}  (target 0)")

# Equivalent shorter form: ``result.summary()`` prints the same table
# without needing an explicit ``print(...)`` call. Both work — pick
# whichever reads best in your notebook.
result.summary()

# %% [markdown]
# ## 8. Parameter posterior
#
# **`Posterior.plot_corner(truths=...)`** marks the truth values and
# (for stellar mass / SFR) overlays the derived quantities computed from
# every posterior sample. Look for tilted ridges — the classic
# age–dust–metallicity degeneracy is the most common culprit when
# constraints look loose.
#
# ### Figure 4 — Corner plot.

# %%
fig = result.plot_corner(truths=truth)
plt.show()

# %% [markdown]
# ## 9. Posterior over the SFH itself
#
# We resample 200 posterior parameter draws and evaluate `model.predict_sfh`
# on each, stacking the resulting SFR(t) curves. The 16/50/84 percentiles
# at each lookback time give the credible bands.
#
# Conceptually: instead of plotting "the best-fit SFH", we plot **the
# distribution of SFHs consistent with the data** — the right thing to
# compare with archaeological reconstructions or simulation catalogues.

# %%
n_draws = 200
posterior_samples = result.resample(jax.random.PRNGKey(7), n=n_draws)


def _draw(i):
    out = {}
    for k, v in posterior_samples.items():
        out[k] = v[i] if hasattr(v, "shape") and v.shape[:1] == (n_draws,) else v
    return out


sfh0 = model.predict_sfh(_draw(0))
t_gyr = np.asarray(sfh0["t_gyr"])
sfr_stack = np.empty((n_draws, t_gyr.size))
sfr_stack[0] = np.asarray(sfh0["sfr_mean"])
for i in range(1, n_draws):
    sfr_stack[i] = np.asarray(model.predict_sfh(_draw(i))["sfr_mean"])

p16, p50, p84 = np.percentile(sfr_stack, [16, 50, 84], axis=0)
p025, p975 = np.percentile(sfr_stack, [2.5, 97.5], axis=0)

# %% [markdown]
# ### Figure 5 — SFH posterior.
# Median (solid) plus 68% (dark) and 95% (light) credible bands; truth
# in dashed red.

# %%
fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.fill_between(t_gyr, p025, p975, color="C0", alpha=0.18, label="95% CI")
ax.fill_between(t_gyr, p16, p84, color="C0", alpha=0.40, label="68% CI")
ax.plot(t_gyr, p50, color="C0", lw=1.4, label="Median")
ax.plot(t_lookback, sfr_truth_curve, color="C3", lw=1.2, ls="--", label="Truth")
ax.set_xlim(0, 13.5)
ax.set_ylim(bottom=0)
ax.set_xlabel("Lookback time [Gyr]")
ax.set_ylabel(r"SFR  [M$_\odot$ yr$^{-1}$]")
ax.legend(frameon=False)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 10. Posterior over derived quantities
#
# **`result.derived`** is a cached property that runs the forward model
# on every posterior sample to produce arrays of astronomer-relevant
# quantities: total stellar mass, SFR averaged over 10 / 100 Myr, and
# sSFR. Mass-weighted age sits on the
# JIT-compatible `predict_sfh_quantities` pytree — we vmap that over
# posterior draws for ~20× speedup vs. a Python loop. These are *all
# first-class outputs of the same JIT-compiled forward pass*; there is
# no separate post-processing stage.

# %%
derived = result.derived
mstar_log = np.log10(np.asarray(derived["stellar_mass"]))
sfr10_log = np.log10(np.asarray(derived["sfr_10myr"]))
ssfr_log = np.log10(np.asarray(derived["ssfr"]))

# Mass-weighted age via vmap (faster + clearer than a Python loop)
sfh_q_fn = jax.jit(jax.vmap(model.predict_sfh_quantities))
sfh_q = sfh_q_fn(posterior_samples)
mw_age = np.asarray(sfh_q.mass_weighted_age_gyr)

# Truths for overlays
truth_mstar_log = np.log10(np.trapezoid(sfr_truth_curve * 1e9, t_lookback))
truth_sfr10 = sfr_truth_curve[t_lookback < 0.02].mean() if (
    t_lookback < 0.02
).any() else sfr_truth_curve[0]
truth_sfr10_log = float(np.log10(max(truth_sfr10, 1e-3)))
truth_ssfr_log = truth_sfr10_log - truth_mstar_log
truth_mw_age = float(
    np.trapezoid(t_lookback * sfr_truth_curve, t_lookback)
    / np.trapezoid(sfr_truth_curve, t_lookback)
)

# %% [markdown]
# ### Figure 6 — Derived-property posteriors.

# %%
fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.6))
quantities = [
    (mstar_log, r"$\log_{10} M_\star\,[M_\odot]$", truth_mstar_log),
    (sfr10_log, r"$\log_{10}$ SFR$_{10\,{\rm Myr}}$  [M$_\odot$ yr$^{-1}$]",
     truth_sfr10_log),
    (ssfr_log,  r"$\log_{10}$ sSFR  [yr$^{-1}$]", truth_ssfr_log),
    (mw_age,    r"Mass-weighted age  [Gyr]", truth_mw_age),
]
for ax, (arr, label, truth_val) in zip(axes.ravel(), quantities):
    ax.hist(arr, bins=30, color="C0", alpha=0.7, density=True)
    lo, med, hi = np.percentile(arr, [16, 50, 84])
    for q in (lo, med, hi):
        ax.axvline(q, color="C0", ls=":", lw=0.9)
    ax.axvline(truth_val, color="C3", ls="--", lw=1.2, label="Truth")
    ax.set_xlabel(label)
    ax.set_yticks([])
    ax.set_title(f"{med:.2f}  ({lo:.2f}, {hi:.2f})", fontsize=9)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 11. Posterior-predictive SED + photometry
#
# A final sanity check. We push every posterior draw through both
# `predict_obs_sed` (continuous SED) and `predict_photometry`
# (band-integrated fluxes), then compare to the data. If the posterior
# is well-calibrated, residuals should sit within ±2σ in every band.

# %%
sed0 = model.predict_rest_sed(_draw(0))
wave_obs_aa = np.asarray(sed0.wavelength) * (1.0 + z)
sed_stack = np.empty((n_draws, wave_obs_aa.size))
sed_stack[0] = np.asarray(sed0.sed)
for i in range(1, n_draws):
    sed_stack[i] = np.asarray(model.predict_rest_sed(_draw(i)).sed)
sed_stack_fnu = np.asarray(units.lnu_to_fnu(sed_stack, dl_cm, z))
sed16, sed50, sed84 = np.percentile(sed_stack_fnu, [16, 50, 84], axis=0)
sed025, sed975 = np.percentile(sed_stack_fnu, [2.5, 97.5], axis=0)

phot_stack = np.empty((n_draws, photometry.n_filters))
for i in range(n_draws):
    phot_stack[i] = np.asarray(model.predict_photometry(_draw(i)))
phot16, phot50, phot84 = np.percentile(phot_stack, [16, 50, 84], axis=0)

# %% [markdown]
# ### Figure 7 — Posterior-predictive SED + photometry with residuals.
# **Top:** continuous SED posterior (median + 68% / 95% bands) over the
# data. Photometry: observed (black) and band-integrated posterior
# median (blue squares with 68% bars). **Bottom:** residuals in σ units.

# %%
wave_obs_um_full = wave_obs_aa / 1e4

fig, (ax_top, ax_res) = plt.subplots(
    2, 1, figsize=(6.5, 4.6), sharex=True,
    gridspec_kw={"height_ratios": [3, 1]},
)

ax_top.fill_between(wave_obs_um_full, sed025, sed975,
                    color="C0", alpha=0.15, label="SED 95% CI", zorder=1)
ax_top.fill_between(wave_obs_um_full, sed16, sed84,
                    color="C0", alpha=0.30, label="SED 68% CI", zorder=2)
ax_top.plot(wave_obs_um_full, sed50, color="C0", lw=1.0,
            label="SED median", zorder=3)

ax_top.errorbar(lam_eff_um, np.asarray(flux_obs), yerr=np.asarray(noise),
                fmt="o", ms=5, color="black", ecolor="0.4",
                mfc="white", capsize=2, label="Data", zorder=5)

phot_err_lo = phot50 - phot16
phot_err_hi = phot84 - phot50
ax_top.errorbar(lam_eff_um, phot50,
                yerr=[phot_err_lo, phot_err_hi],
                fmt="s", ms=4, color="C0", ecolor="C0",
                mfc="C0", mec="C0", capsize=0,
                label="Photometry posterior (median, 68%)", zorder=4)

ax_top.set_xscale("log")
ax_top.set_yscale("log")
ax_top.set_xlim(0.13, 30)
ax_top.set_ylabel(r"$f_\nu$  [erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$]")
ax_top.legend(frameon=False, loc="lower center", fontsize=8, ncol=2)

residual_sigma = (np.asarray(flux_obs) - phot50) / np.asarray(noise)
ax_res.axhline(0, color="0.6", lw=0.6)
ax_res.axhline(2, color="0.6", lw=0.6, ls=":")
ax_res.axhline(-2, color="0.6", lw=0.6, ls=":")
ax_res.scatter(lam_eff_um, residual_sigma, s=22, color="black")
ax_res.set_xscale("log")
ax_res.set_xlabel(r"Observed wavelength [$\mu$m]")
ax_res.set_ylabel(r"$(d - m)/\sigma$")
ax_res.set_ylim(-4, 4)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Where to go next
#
# - **Stochastic SFH** — turn on the GP `sfh_field_psd_*` parameters to
#   model bursty SFHs (notebook `14_stochastic_sfh`).
# - **Joint photometry + spectroscopy** — break the age–dust–Z degeneracy
#   with a few hundred spectral pixels (`05_joint_photometry_spectroscopy`).
# - **AGN** — Kubota & Done disc + SKIRTOR torus + Cue NLR
#   (`10_agn_advanced`).
# - **Galaxy populations** — hierarchical inference across thousands of
#   galaxies (`11_population`).
# - **Variational inference** — fast approximate posteriors at high
#   dimensionality (`15_vi_inference`).
# - **GPU / batch fitting** — the same pipeline runs unchanged on a
#   GPU; see the inference scaling guide.
#
# Every figure on this page was generated by the code above. Copy it,
# change the truth dict, fit your own galaxy.
