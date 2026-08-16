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
# # Discovering the menu — every variant, from inside Python
#
# Tengri is plugin-aware. The library knows what physics it ships, which
# inference backends are installed and compatible with a given model,
# what filters are on disk, and how to fetch missing SSP grids. Asking
# Python is faster than reading docs.

# %%
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

# Keep the rendered tutorial clean: silence framework notices that do not
# change the science shown here (baked-in nebular, the WavePrecomp blue-band
# approximation, and
# recipe/parameter-provenance notices). Genuine deprecations in user-facing
# calls are fixed in the code, not hidden.
#
# WildcardPartialFreeWarning is deliberately NOT silenced. This notebook is
# about discovering what the menu offers, and that warning is the discovery:
# it names the parameters an `all_params: FREE` wildcard could not free
# because they carry only a Fixed default. Hiding it here would leave a
# reader to meet it for the first time alone, in their own code.
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")
warnings.filterwarnings("ignore", message=".*WavePrecomp.*")
warnings.filterwarnings("ignore", message=".*was marked FIXED.*")
warnings.filterwarnings("ignore", message=".*Composable AGN.*")
warnings.filterwarnings("ignore", message=".*before the Big Bang.*")
warnings.filterwarnings("ignore", category=RuntimeWarning)

import inspect
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np

from _setup import FIG_DIR
import tengri
from tengri import (
    FIXED,
    Observation,
    Photometry,
    SEDModel,
    builders,
    citations,
    plot,
    recipes,
)

plot.setup_style()

# %% [markdown]
# ## Bird's eye

# %%
tengri.summary()

# %% [markdown]
# ## SSP catalogs
#
# Bare-stellar grids (e.g. `fsps_prsc_miles_chabrier`) pair with the Cue
# nebular emulator; "wNE" grids carry baked-in nebular emission and pair
# with the `ssp` nebular backend. `download_ssp` fetches missing grids on
# demand.

# %%
tengri.list_known_ssps()

# %%
ssp = tengri.load_ssp("fsps_prsc_miles_chabrier", download=True)

# %% [markdown]
# ## Star-formation history variants
#
# Each entry of `builders.sfh.available()` is a real callable whose
# signature lists its parameters with default priors. Notebook
# autocompletion works on it.

# %%
builders.sfh.available()

# %%
print(inspect.signature(builders.sfh.dpl))

# %% [markdown]
# Two SFHs at the same mass, side by side:

# %%
obs = Observation(
    photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "wise_w1"])
)


def _build_sfh(sfh_factory):
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh=sfh_factory(defaults=FIXED),
        dust={"type": "two_component", "law_bc": "calzetti", "all_params": FIXED},
        neb={"type": "none"},
        redshift=0.05,
    )


fig, ax = plt.subplots(figsize=(7, 3.6))
for name, factory, color in [
    ("DPL", builders.sfh.dpl, "#3a76d9"),
    ("delayed-exp", builders.sfh.dexp, "#c8377d"),
    ("tsnorm burst", builders.sfh.tsnorm, "#d97a3a"),
]:
    m = _build_sfh(factory)
    p = m.spec.sample(jax.random.PRNGKey(0))
    p = {**m.spec.get_fixed_values(), **p}
    s = m.predict_state(p)
    lbt_gyr = np.asarray(s.derived["sfh_grid_lbt_yr"]) / 1e9
    sfr = np.asarray(s.derived["sfr_history"])
    ax.plot(lbt_gyr, sfr, color=color, lw=1.4, label=name)
ax.invert_xaxis()
ax.set_xlabel("lookback time [Gyr]")
ax.set_ylabel(r"SFR  [$M_\odot$/yr]")
ax.set_yscale("log")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(FIG_DIR / "03_sfh_variants.png", dpi=300, bbox_inches="tight")

# %% [markdown]
# ## Dust laws and IR templates
#
# Attenuation laws — Calzetti, Cardelli (MW/LMC/SMC), Prevot SMC, Li 2008,
# Witt & Gordon 2000, Conroy/Charlot–Fall variants — and IR re-emission
# templates (Dale 2014, Draine–Li, THEMIS, Astrodust, BOSA, Casey 2012,
# modified blackbody) all live under `tengri.components.dust`.

# %%
tengri.list_dust_laws()

# %%
tengri.list_dust_emission_models()

# %% [markdown]
# ## Nebular backends
#
# - **`cue`** — neural emulator on Cloudy 17.03 (Li+2024). Fast, smooth,
#   requires bare-stellar SSP.
# - **`ssp`** — baked-in nebular contribution from a wNE SSP grid.
#   Cheapest, locked to grid choices.
# - **`cloudy`** — direct Cloudy evaluation. Accurate, slow.
# - **`none`** — disable.

# %%
tengri.list_nebular_backends()

# %% [markdown]
# ## AGN composables
#
# AGN is built from six orthogonal stages — disc, nlr, blr, feii, torus,
# attenuation — each with its own registry of swappable blocks. The
# composable blocks are the recommended surface for mixing and matching AGN
# components. `recipes.agn_panchromatic()` is one stable composition.

# %%
tengri.list_agn_models()

# %% [markdown]
# ### Composable AGN blocks — discovery and mixing
#
# The composable grammar groups blocks by pipeline stage. Use
# `list_agn_blocks()` to see all available options, grouped by category.
# Each block has a citation, status, and description accessible via
# `describe_agn_block()`.

# %%
tengri.list_agn_blocks()

# %%
# Inspect a specific block
tengri.describe_agn_block("skirtor", category="torus")

# %% [markdown]
# ## Recipes — five curated starting points

# %%
print(list(recipes.__all__))

# %%
recipes.star_forming_photometry()

# %% [markdown]
# ## Inference backends
#
# The `status` column reports whether each backend's dependencies are
# importable. Compatibility against a specific model is rechecked at
# `Fitter.run` time.

# %%
tengri.list_inference_methods()

# %% [markdown]
# ## Parameter and physics provenance
#
# Two introspection surfaces close the loop:
# `model.spec.summary()` tags every parameter with where its value came
# from (`[user]` / `[all_params FREE]` / `[all_params FIXED]` / `[default]`).
# `citations.collect_citations(model)` returns the bibliography of every
# physics ingredient.

# %%
example = SEDModel.build(ssp_data=ssp, observation=obs, **recipes.star_forming_photometry())
example.spec.summary()

# %%
bib = citations.collect_citations(example)
citations.print_citations(example)

# %% [markdown]
# Export to BibTeX in one call:

# %%
bibtex = citations.citations_bibtex(example)
print(bibtex[:600], "…")

# %% [markdown]
# ## Cache and memory
#
# - **`tengri.lean`** — context manager / default mode. Drops the
#   per-Fitter JIT engine after each `run` to release graph memory.
# - **`tengri.persistent`** — opposite mode. Keep the engine for
#   repeated same-shape fits (population sweep, MCMC restart).
# - **`tengri.gc`** — one-shot collect: clears shared caches and asks
#   XLA to release the device.
# - **`tengri.clear_shared_caches()`** — full reset for clean
#   benchmarking.
#
# The persistent JAX compile cache (`~/.cache/tengri_jax_cache`,
# auto-enabled by `import tengri`) is independent of these — it survives
# process restarts.

# %% [markdown]
# ## Cross-namespace search
#
# `tengri.search("term")` queries every registry at once.

# %%
tengri.search("Calzetti")

# %% [markdown]
# ## Sub-namespaces — where to look for what
#
# - **`tengri.cosmology`** — Planck 2018 distance and time integrals
#   (`luminosity_distance_mpc`, `age_at_z`, …).
# - **`tengri.units`** — F_ν ↔ L_ν ↔ AB-mag conversions, vacuum-air, Jy.
# - **`tengri.plot`** — `plot_sed_fit`, `plot_sfh`, `plot_corner_comparison`,
#   `setup_style`, `diagnostics_table`. Re-exports of
#   `tengri.analysis.plotting`.
# - **`tengri.observation`** — `Photometry`, `Spectroscopy`, `Observation`,
#   `NoiseModel`, `LineList`, filter loaders.
# - **`tengri.inference`** — `Catalog` (many galaxies, one call),
#   `VIConfig`, `InferenceContext`. Single-galaxy fits go through
#   `ForwardModel.fit`; `Fitter` is an internal engine, not a surface to
#   call directly.
# - **`tengri.results`** — `Posterior`, `CatalogPosterior`,
#   `PopulationPosterior`, `FitResult`, `MockData`, `Provenance`,
#   `generate_mock`.
# - **`tengri.config`** — `SEDModelConfig`, `SFHConfig`, `DustConfig`,
#   `NebularConfig`, `AGNConfig`.
# - **`tengri.protocols`** — Protocol shapes (`SEDComponent`, `Likelihood`,
#   `ObservationModel`, `DerivedKey`, `ForwardState`).
# - **`tengri.builders`** — config-dict factories with introspectable
#   signatures (`builders.sfh.*`, `builders.dust.*`, `builders.neb.*`,
#   `builders.agn.*`, `builders.igm.*`).
# - **`tengri.citations`** — `Bibliography`, `Citation`,
#   `collect_citations`, `citations_report`, `citations_bibtex`,
#   `print_citations`, `print_bibtex`, `paper_citation`.
#
# From here: try `tengri.help()` for topic-indexed pointers, or
# `dir(tengri.components.<area>)` for any physics sub-namespace.
