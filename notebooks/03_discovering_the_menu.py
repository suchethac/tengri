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
# # Discovering the menu: tengri's introspection API
#
# *Learn what physics tengri offers, without writing a single model or fit.*
#
# **What you'll do.** Walk through every discovery entrypoint — the `list_*()`,
# `describe()`, `search()`, and `doctor()` functions — to learn what's available
# in the registry. You'll filter tables by status or citation, probe the
# metadata for a single model, cross-menu search for a concept, and check
# whether your installation is healthy.
#
# **Why this matters.** Before you build a model, you need to know what
# physics is on offer. Tengri exposes this live — you discover as you go,
# in the REPL or notebook, without leaving Python. Everything returns
# real Python lists and dicts — no weird DSLs or CLIs.
#
# **Runtime budget.** ~2 seconds total. Pure introspection, no compilation,
# no fitting, no downloads. Filter tables, print metadata, search the
# registry. Everything runs on the CPU regardless of GPU availability.

# %% [markdown]
# ## 1. Global overview
#
# Start with a bird's-eye view of everything in tengri.

# %%
import numpy as np

import tengri

# The one-liner summary: counts of every menu.
tengri.summary()

# %% [markdown]
# **`tengri.summary()`** is what you run first after `import tengri`.
# It prints live counts from every registry — add a new model
# via `@register_agn_model` and this number updates automatically.

# %% [markdown]
# ## 2. The full cheatsheet
#
# **`tengri.help()`** is a 5-section narrative. Read it once to see the
# entire surface area of the library.

# %%
tengri.help()

# %% [markdown]
# **`tengri.help("dust")`** narrows to one topic — dust attenuation +
# emission in this case. Try other topics too:
# `"agn"`, `"sfh"`, `"nebular"`, `"components"`, `"inference"`, `"filters"`,
# `"plot"`, `"citations"`.

# %%
tengri.help("dust")

# %% [markdown]
# ## 3. Listing menus
#
# Each menu is a `_RegistryTable` — a real Python list that prints as a
# nice table. You can filter it, iterate it, or slice it like any list.

# %% [markdown]
# ### 3.1. Star formation histories

# %%
sfh_table = tengri.list_sfh_models()
print(f"Total SFH models: {len(sfh_table)}\n")
print(sfh_table)

# %% [markdown]
# Narrow the table with `.filter()` — case-insensitive, chainable.

# %%
tengri.list_sfh_models().filter(citation__contains="Leja")

# %% [markdown]
# ### 3.2. Dust attenuation laws
#
# Twenty-one curves for UV/optical absorption and scattering: Calzetti,
# Cardelli, Charlot-Fall, Kriek & Conroy, Noll, Salim, SMC, LMC, and others.

# %%
tengri.list_dust_laws()

# %% [markdown]
# Filter by production status (exclude experimental models for conservative fits).

# %%
tengri.list_dust_laws().filter(status="production")

# %% [markdown]
# Filter by author — here Calzetti's law.

# %%
tengri.list_dust_laws().filter(citation__contains="Calzetti")

# %% [markdown]
# ### 3.3. Dust emission templates
#
# Seven template families for IR re-radiation: DL07, DL14, Dale2014, THEMIS,
# MBB, Astrodust, BOSA. Energy-balance fitting pairs these with attenuation.

# %%
tengri.list_dust_emission_models()

# %% [markdown]
# ### 3.4. Nebular backends
#
# Four ways to compute emission lines: baked into the SSP grid (default),
# or via neural emulator (CUE), cloud-physics grids (Cloudy, CB19).

# %%
tengri.list_nebular_backends()

# %% [markdown]
# ### 3.5. AGN models
#
# Twelve AGN configurations: Kubota & Done disc (full + simplified variants),
# SKIRTOR torus geometry, multicolor continuum, ADAF accretion, plus QSOgen.

# %%
tengri.list_agn_models()

# %% [markdown]
# Filter to production-ready models.

# %%
tengri.list_agn_models().filter(status="production")

# %% [markdown]
# ### 3.6. Inference methods
#
# Primary methods for new users: HMC/NUTS (exact posterior), MAP (point
# estimate), geoVI (fast approximation). Experimental: raytrace, NSS.

# %%
tengri.list_inference_methods(tier="primary")

# %% [markdown]
# Show all methods, including experimental.

# %%
tengri.list_inference_methods()

# %% [markdown]
# ### 3.7. Physics components
#
# The core SED pipeline: stellar (SFH + SSP), dust (attenuation + emission),
# AGN, nebular (emission lines), radio, IGM, X-ray.

# %%
tengri.list_components()

# %% [markdown]
# ### 3.8. Plotting helpers
#
# Matplotlib wrappers for common figures: SED fits, SFH curves, corner plots,
# corner comparisons, spectrum fits, diagnostics tables.

# %%
tengri.list_plots()

# %% [markdown]
# ### 3.9. Photometric filters
#
# 242+ filter curves from the SVO Filter Profile Service: SDSS, GALEX, 2MASS,
# WISE, HST, Euclid, JWST, and more.

# %%
n_filters = len(tengri.list_filters())
print(f"Total filters: {n_filters}\n")

# %% [markdown]
# Filter by survey using astronomer-friendly aliases.

# %%
tengri.list_filters(survey="SDSS")

# %% [markdown]
# **Smart survey aliases.** `survey="SDSS"` matches `SLOAN_SDSS_*` rows
# even though "SDSS" is the *instrument* in SVO's filename schema, not the
# survey. Works for `"DES"`, `"VISTA"`, `"HSC"`, `"UKIDSS"`, `"PS1"` too.

# %%
tengri.list_filters(survey="GALEX")

# %% [markdown]
# #### Chaining filters — a real example
#
# Chain `.filter()` calls to narrow down. Here we find JWST NIRCam filters
# with "F150" in the band name.

# %%
jwst_nircam_f150 = (
    tengri.list_filters()
    .filter(survey="JWST")
    .filter(instrument="NIRCam", band__contains="F150")
)
print(f"JWST NIRCam F150 filters: {len(jwst_nircam_f150)}\n")
print(jwst_nircam_f150)

# %% [markdown]
# The `.names()` method extracts just the string list — feeds directly
# into `Photometry.from_names()` for zero-friction filter discovery.

# %%
filter_names = jwst_nircam_f150.names()
print(f"Filter names for Photometry():\n{filter_names}")

# %% [markdown]
# ## 4. API discovery surface
#
# Tengri exposes more discovery helpers beyond the menus above.
#
# ### 4.1. Filter suggestion by redshift

# %%
# Suggest a bandset for a low-redshift galaxy (z=0.05) and a high-z target (z=2.5).
low_z_filters = tengri.filters.suggest(redshift=0.05, coverage="visible_to_nir")
print(f"Suggested filters for z=0.05:\n  {low_z_filters}\n")

high_z_filters = tengri.filters.suggest(redshift=2.5, coverage="visible_to_nir")
print(f"Suggested filters for z=2.5:\n  {high_z_filters}\n")

# %% [markdown]
# ### 4.2. Filter properties and metadata

# %%
# Inspect a single filter in detail.
info = tengri.filters.filter_info("jwst_f200w")
print(f"Filter info for {info['name']}:")
print(f"  Facility: {info['facility']}")
print(f"  λ_eff: {info['lambda_eff_str']}")
print(f"  FWHM: {info['fwhm_str']}\n")

# %% [markdown]
# Compute effective wavelength and FWHM for any filter.

# %%
# Load a few filters and compute their properties
test_filters = ["sdss_u", "sdss_g", "2mass_j"]
for fname in test_filters:
    fc = tengri.filters.load_filter(fname)
    wave_np, trans_np = (np.asarray(fc.wave), np.asarray(fc.trans))
    lam_eff = tengri.filters.compute_effective_wavelength(wave_np, trans_np)
    fwhm = tengri.filters.compute_fwhm(wave_np, trans_np)
    print(f"{fname:15s}  λ_eff={lam_eff:8.1f} Å  FWHM={fwhm:7.1f} Å")

# %% [markdown]
# ### 4.3. Introspection API

# %%
# List available example scripts
try:
    tengri.examples()
except NotImplementedError:
    print("❋ api preview — examples() not yet implemented\n")

# %% [markdown]
# Explain the role of major classes.

# %%
# Show architectural role of SEDModel and Posterior
try:
    tengri.explain(tengri.SEDModel)
except NotImplementedError:
    print("❋ api preview — explain() not yet implemented\n")

# %% [markdown]
# Print the fun easter egg.

# %%
tengri.print_logo()

# %% [markdown]
# ### 4.4. Parameter suggestion

# %%
# Ask tengri to suggest default parameters for a given configuration
try:
    suggested = tengri.suggest_parameters(
        mean_sfh_type="dpl",
        dust_law_bc="calzetti",
        dust_emission="dale2014",
    )
    print("Suggested parameter set:")
    print(suggested)
except (NotImplementedError, TypeError):
    print("❋ api preview — suggest_parameters() under development\n")

# %% [markdown]
# ### 4.5. Bibliography registry

# %%
# Cite a single author/model by key
try:
    tengri.cite("calzetti")
except (KeyError, TypeError):
    print("❋ No direct cite('calzetti') in this version; see cite_components()\n")

# %% [markdown]
# ### 4.6. List all registries at once

# %%
all_tables = tengri.list_all()
total = sum(len(table) for table in all_tables.values())
print(f"Total catalog size: {total} entries")
for kind, table in all_tables.items():
    print(f"  {kind:25s}: {len(table):3d}")
    if len(table) > 0:
        print(f"    {table[0] if isinstance(table[0], dict) else table[0].name}")
    print()

# %% [markdown]
# ### 4.7. Filter operators reference
#
# The `.filter()` method supports several operators for fine-grained queries.

# %%
# Example: exact match
exact = tengri.list_filters().filter(instrument="MIRI").names()[:3]
print(f"Exact match (instrument='MIRI'): {exact}\n")

# Example: contains
contains = (
    tengri.list_filters()
    .filter(band__contains="F150")
    .names()[:3]
)
print(f"Contains (band__contains='F150'): {contains}\n")

# Example: startswith
startswith = (
    tengri.list_filters()
    .filter(survey__startswith="JWST")
    .names()[:3]
)
print(f"Startswith (survey__startswith='JWST'): {startswith}\n")

# Example: in tuple (multiple values)
in_tuple = (
    tengri.list_filters()
    .filter(instrument__in=("NIRCam", "MIRI"))
    .names()[:3]
)
print(f"In tuple (instrument__in=...): {in_tuple}\n")

# %% [markdown]
# ## 5. Single-entry lookup
#
# Use **`describe()`** to get full metadata for any name — model, method,
# component, or filter.

# %%
tengri.describe("calzetti")

# %% [markdown]
# Describe an AGN model.

# %%
tengri.describe("skirtor")

# %% [markdown]
# Describe an inference method.

# %%
tengri.describe("mcmc_nuts")

# %% [markdown]
# Describe a dust emission template.

# %%
tengri.describe("dale2014")

# %% [markdown]
# ## 5. Cross-menu search
#
# **`search()`** walks every registry and returns every entry whose name,
# short_doc, citation, or status (case-insensitive) contains your query.
# Hits from different menus get a `kind` column so you can tell them apart.

# %%
tengri.search("torus")

# %% [markdown]
# Search for "dirichlet" SFH.

# %%
tengri.search("dirichlet")

# %% [markdown]
# Search by author — everything Leja-cited.

# %%
tengri.search("Leja")

# %% [markdown]
# ## 9. Environment health
#
# **`doctor()`** checks whether your install is healthy: Python version,
# JAX configuration, SSP data files, GPU availability.

# %%
tengri.doctor()

# %% [markdown]
# ## 8. Hero figure: Photometric filter coverage
#
# Build a multi-wavelength filter set and visualize their transmission curves.
# Color by effective wavelength to reveal the spectral coverage, and annotate
# the Balmer break at z=0.1.

# %%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# Set publication-style defaults
tengri.plot.setup_style()

# 1. Discover filters via the chained API — use available registry names
# Build a representative set: SDSS optical + 2MASS near-IR
sdss_names = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]
ir_names = ["2mass_j", "2mass_h", "2mass_ks"]
all_filter_names = sdss_names + ir_names
print(f"Loading filters: {all_filter_names}\n")

# 2. Load each filter and collect metadata
filters_data = []
for name in all_filter_names:
    try:
        fc = tengri.filters.load_filter(name)
        wave_np, trans_np = np.asarray(fc.wave), np.asarray(fc.trans)
        lam_eff = tengri.filters.compute_effective_wavelength(wave_np, trans_np)
        filters_data.append({
            'name': name,
            'wave': wave_np,
            'trans': trans_np,
            'lam_eff': lam_eff
        })
    except KeyError as e:
        print(f"⚠ Skipping {name}: {e}")

print(f"Loaded {len(filters_data)} filters\n")

# 3. Create the figure with a shared wavelength axis
fig, ax = plt.subplots(figsize=(12, 6))

# Color map: effective wavelength → perceptual color
lam_effs = np.array([f['lam_eff'] for f in filters_data])
norm = Normalize(vmin=lam_effs.min(), vmax=lam_effs.max())
cmap = plt.get_cmap("viridis")

# Plot each filter's transmission curve
for fdata in filters_data:
    color = cmap(norm(fdata['lam_eff']))
    ax.plot(fdata['wave'] / 1e4, fdata['trans'], linewidth=2.0, color=color,
            label=fdata['name'].upper())

# 4. Annotate the Balmer break for a nearby galaxy (z=0.1)
# Balmer edge: 3647 Å (rest) → 4012 Å (z=0.1)
z = 0.1
balmer_obs = 3647 * (1 + z) / 1e4

ax.axvline(balmer_obs, color="red", linestyle="--", alpha=0.5, linewidth=1.5)
ax.text(balmer_obs, 0.95, f"Balmer break (z={z})", rotation=90, va="top", ha="right",
        fontsize=10, color="red", alpha=0.8, fontweight="bold")

# 5. Labels and formatting
ax.set_xlabel("Wavelength (μm)", fontsize=12)
ax.set_ylabel("Transmission (normalized)", fontsize=12)
ax.set_title("Multi-Wavelength Filter Coverage: SDSS + 2MASS", fontsize=13, fontweight="bold")
ax.set_xlim(0.3, 2.5)
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3, linestyle=":")
ax.legend(fontsize=10, frameon=False, loc="upper right", ncol=2)

# 6. Add colorbar for effective wavelength
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, label="λ_eff (Å)", pad=0.02)
cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x/1e3:.1f}k"))

plt.tight_layout()
import os
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/03_filter_coverage.png", dpi=200, bbox_inches="tight")
print("\n✓ Saved hero figure to figures/03_filter_coverage.png")
plt.show()

# %% [markdown]
# **What this figure shows.** SDSS optical bands (3000–10000 Å) combined with
# 2MASS near-IR filters. The Balmer break near 4000 Å (rest-frame) is a strong
# stellar mass tracer. These eight filters give you good wavelength coverage
# for constraining age, dust, and metallicity in local galaxies.

# %% [markdown]
# ## 7. Putting it together: discovery API to Observation
#
# Demonstrate the full workflow — from introspection to building an
# `Observation` ready for a model.
#
# **Note:** `list_filters().names()` returns SVO-style long names
# (e.g. `"SLOAN_SDSS_g"`), but `Photometry.from_names()` also accepts
# shorthand aliases (e.g. `"sdss_g"`). Both work.

# %%
# 1. Discover filters via the registry
sdss_filters = tengri.list_filters(survey="SDSS")
print(f"Found {len(sdss_filters)} SDSS filters\n")
print(sdss_filters)

# %% [markdown]
# 2. Manually build a working filter set using shorthand names
# (what `Photometry.from_names()` expects).

# %%
# Use shorthand names for photometry
filter_names = [
    "sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z",
    "2mass_j", "2mass_h", "2mass_ks",
]
print(f"Filter shorthand names:\n  {filter_names}\n")

# 3. Create Photometry from discovered filters
from tengri import Observation, Photometry

photometry = Photometry.from_names(filter_names)
print(f"Loaded {photometry.n_filters} filters from shorthand names")
print(f"  Filter names: {photometry.names}\n")

# 4. Bundle into an Observation (the object SEDModel expects)
obs = Observation(photometry=photometry)
print(f"✓ Observation ready with {obs.photometry.n_filters} photometric bands")

# %% [markdown]
# That entire workflow — from `list_filters()` all the way to an `Observation`
# you can pass to `SEDModel()` — is three method calls. Discovery and model
# building are tightly coupled, not separate.

# %% [markdown]
# ## 10. CLI mirror
#
# The same introspection functions are available from the command line:
#
# ```bash
# python -m tengri summary
# python -m tengri doctor
# python -m tengri search torus
# python -m tengri describe skirtor
# python -m tengri list agn
# ```
#
# Run in a terminal when you want quick answers without opening a notebook.

# %% [markdown]
# ## Next steps
#
# You now know how to **discover** what tengri offers.  Next, build your
# first model:
#
# - **Notebook 04: Building models** — construct a `Parameters`, `SEDModel`,
#   and `Fitter` from the components you just explored.
# - **Notebook 05: Joint photometry + spectroscopy** — break the age–dust
#   degeneracy with spectral data.
# - **Notebook 14: Stochastic SFH** — add bursty star formation via the GP.
#
# Every name you discovered here becomes a kwarg to `Parameters()`:
#
# ```python
# spec = tengri.Parameters(
#     mean_sfh_type="dpl",                  # from list_sfh_models()
#     dust_law_bc="calzetti",               # from list_dust_laws()
#     dust_emission="dale2014",             # from list_dust_emission_models()
#     agn_model="skirtor",                  # from list_agn_models()
#     nebular_backend="cue",                # from list_nebular_backends()
# )
# ```
#
# Happy exploring!
