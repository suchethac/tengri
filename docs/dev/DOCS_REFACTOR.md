# Docs & Notebooks Refactor Plan

*Written 2026-04-05 — retrospective on what we should have done differently, and the forward plan.*

---

## The Core Problem

The existing documentation has grown to **~66 active notebooks** spread across 8 separate
folder tracks (`demonstrations/`, `reference/`, `fitting/`, `models/`, `quickstart/`,
`specialist/`, `theory/`, `tutorials/`), plus 29 Sphinx gallery scripts in `examples/` — all
with overlapping content and no clear division of labor between them.

The underlying cause: documentation was written in successive waves as the codebase evolved,
with each wave creating a new folder rather than replacing old content. The result:

- **Layered redundancy**: The `demonstrations/` track (15 notebooks) and `fitting/` track (7
  notebooks) cover nearly the same fitting workflows. The `reference/` track (19 notebooks,
  including the best gallery content — `15_model_gallery_attenuation`, `16_`, `17_`, `18_`,
  `19_`) was superseded by `models/` (8 notebooks) but never deleted. `tutorials/` (5
  notebooks) is yet another quickstart attempt running in parallel with `quickstart/` (3
  notebooks).
- **No division of labor** between Sphinx gallery scripts and notebooks — both do parameter
  sweeps with separate, independent code.
- **Wrong emphasis**: The fitting track (22+ notebooks across `demonstrations/` + `fitting/`)
  vastly outnumbers the physics/gallery track, when the average astronomer needs to understand
  the forward model before they care about convergence diagnostics.
- **Intimidating entry points**: `theory/01_sfh_prior.ipynb` (811 lines on IFT) is referenced
  before the reader has seen a single SED. The mathematical machinery precedes the intuition.
- **The best content is hidden**: `reference/15–19` (model gallery notebooks) are the clearest
  parameter sweep notebooks in the repo, but they're marked as superseded and excluded from
  the Sphinx build.

**What fixes the sprawl structurally** is not another subfolder — it is a **single numbered spine**
of notebooks at the repo root (`notebooks/00_*.py` … `notebooks/12_*.py`). Everything in the eight
tracks should either feed that spine, merge into it, or be deleted. The big-picture section below
is that spine; the rest of this document is how to build and maintain it.

---

## Big picture: thirteen root notebooks (reader spine)

*The table below is a **working guide** for planning — filenames and “state” notes can drift;
confirm against the tree under `notebooks/*.py`. Jupytext source is `.py`; synced `.ipynb` may
exist locally.*

These thirteen files are the **intended reader journey**: one ordered path from “first fit”
through forward-model intuition, observation and inference, real data, population fitting, and
extension. They sit at `notebooks/` root (not inside `quickstart/`, `models/`, etc.) so the
numbering is obvious in the file browser and in the docs nav ([Docs Site Restructure](#docs-site-restructure)).

### How this spine relates to the eight tracks

| Role | Folders | Plan |
|------|---------|------|
| **Spine** | `notebooks/00_*.py` … `12_*.py` | Primary maintenance target; docs site links here first |
| **Ingredient library** | `models/notebook_code/`, `quickstart/notebook_code/`, `reference/notebook_code/`, … | Lift sections and patterns *into* the spine; retire duplicate notebooks per [consolidation](#notebook-section-hierarchy-and-storylines) elsewhere in this doc |
| **Gallery** | `examples/` | One-parameter thumbnails; notebooks tell multi-parameter *stories* ([How to Reconcile Gallery Scripts and Notebooks](#how-to-reconcile-gallery-scripts-and-notebooks)) |

### Narrative arc (one line)

**00** end-to-end fit → **01** read a SED → **02–06** physics modules in logical order (SFH, dust,
nebular, AGN, multi-λ) → **07–10** fit photometry, fit spectra, degeneracies, real data → **11**
population / hierarchical → **12** customize.

### Inventory (content + known hygiene)

| # | Notebook | What it covers | State / hygiene (planning notes) |
|---|----------|----------------|----------------------------------|
| 00 | `00_quickstart.py` | Five-cell SED fit — mock, fit, posterior readout | Prefer `Model.from_config()`; keep minimal; align savefig policy with bootstrap (`FIGDIR`) |
| 01 | `01_sed_anatomy.py` | Full SED decomposition, 2×2 component build-up, redshift shift | May use `_predict_sed_component()`; prefer public API or document as intentional; audit `savefig` |
| 02 | `02_sfh_gallery.py` | Parametric SFH shapes, stochastic GP, recovery comparison | Direct SFH calls OK for pedagogy; structure is a template for other galleries |
| 03 | `03_dust_gallery.py` | Attenuation sweeps, IR emission, energy balance | Often uses `Model.from_config()` for full-SED demos vs low-level dust-only plots — keep both roles clear in prose |
| 04 | `04_nebular_gallery.py` | Lines, logU, metallicity, escape fraction, broadening, marginalization | `ParamSpec.build_preset()` / `with_fixed_prior()` style is the target API |
| 05 | `05_agn_gallery.py` | Disc/torus, inclination, M_BH/spin, torus covering, X-ray | Avoid private imports (e.g. `_isco_radius`); re-export or use public accessors |
| 06 | `06_multiwavelength_gallery.py` | IGM, FIR–radio, jets, X-ray, XRB/corona | Standardize `FIGDIR` + commented vs active `savefig` |
| 07 | `07_fitting_photometry.py` | Single fit, joint phot+spec, batch, filter selection | **Broken path:** `FIGDIR` used but not defined in bootstrap — must match [bootstrap block](#authoritative-notebook-templates-follow-these-sources); update deprecated method names in prose (`vi` not `native_geovi`) |
| 08 | `08_fitting_spectra.py` | Spectrum basics, noise, calibration marginalization, eline modes | Generally OK; keep aligned with `SpectroscopyConfig` validation |
| 09 | `09_degeneracies.py` | Fisher, constraint matrix, posterior correlations | `savefig` to CWD vs `FIGDIR` — pick one policy; pytest-conditional skips need a short “CI vs local” note in markdown |
| 10 | `10_real_data.py` | SDSS → fit → BPT, derived quantities | Ensure `_plot_style` imports (`convergence_table`, `plot_sfh`, `safe_corner`) are visible in the notebook or linked helper |
| 11 | `11_population.py` | Population PSD intuition, Block Gibbs, hierarchical fit | Uses `PopulationFitter` (canonical); `HierarchicalFitter` is a deprecated alias — keep docs prose on `PopulationFitter`; figure paths under `FIGDIR` |
| 12 | `12_extending_tengri.py` | Custom priors, PSD, dust, mean SFH | Internal GP hooks may be intentional for “extension” — flag in opening markdown |

### Plan hooks (what to do with this table)

1. **Bootstrap parity** — Every root notebook gets the same `FIGDIR = …` + `os.makedirs` pattern as
   [quickstart template](#authoritative-notebook-templates-follow-these-sources); fix **07** first.
2. **Figure policy** — Either all root notebooks write under `notebooks/figures/<chapter>/` or none
   save by default (commented `savefig`); avoid mixed “silent CWD” vs `FIGDIR`.
3. **Private API** — Spine notebooks should not depend on underscore imports without a one-line
   markdown justification (pedagogical mirror of internal …).
4. **Consolidation** — When retiring a track notebook, map its unique section to a **section
   heading** in the spine row above (see [Notebook Section Hierarchy and Storylines](#notebook-section-hierarchy-and-storylines)).

---

## Authoritative notebook templates (follow these sources)

New or refactored notebooks should **match the existing jupytext percent-format `.py` files**
in the repo, not invent a parallel style. The paths below are the **canonical references**.

### Jupytext header and kernel (all tracks)

Every `notebook_code/*.py` file opens with the same YAML front matter and uses `# %%` /
`# %% [markdown]` cell markers. Example (trim version numbers if they drift):

```yaml
# ---
# jupyter:
#   jupytext:
#     formats: notebook_code//py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
```

### Bootstrap block: project root, `data/`, `_plot_style`

Copy the **same** `sys.path` / `chdir` ladder and `FIGDIR` pattern as the reference notebooks.
This 15-line bootstrap block is **deliberate project style** — it ensures `load_ssp_data("data/...")`
resolves correctly regardless of whether the notebook is run from the repo root, from
`notebooks/`, or from the track subdirectory. Do not remove or simplify it — every notebook in
the repo uses this exact pattern.

Then import `setup_style()` (and other helpers) from `_plot_style` next to the track’s
`notebook_code/` folder.

**Quickstart example** (`notebooks/quickstart/notebook_code/01_quickstart.py`):

```python
import sys, os  # noqa: E401

try:
    _nb_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(_nb_dir, "..", ".."))
except NameError:
    _nb_dir = os.getcwd()
    sys.path.insert(0, os.path.join(_nb_dir, ".."))
if os.path.exists("data"):
    pass
elif os.path.exists(os.path.join("..", "data")):
    os.chdir("..")
elif os.path.exists(os.path.join("..", "..", "data")):
    os.chdir(os.path.join("..", ".."))
elif os.path.exists(os.path.join("..", "..", "..", "data")):
    os.chdir(os.path.join("..", "..", ".."))

FIGDIR = os.path.join("quickstart", "figures")
os.makedirs(FIGDIR, exist_ok=True)

from _plot_style import (
    COLORS,
    SPECTRAL_FEATURES,
    convergence_table,
    plot_corner_comparison,
    plot_sfh,
    safe_corner,
    setup_style,
)

setup_style()
```

**Standard opening code cell** (after markdown title): enable x64 JAX, ignore `FutureWarning`,
then imports — again mirroring `01_quickstart.py`:

```python
import warnings

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)
warnings.filterwarnings("ignore", category=FutureWarning)
```

### Quickstart narrative and API style

**Gold standard:** `notebooks/quickstart/notebook_code/01_quickstart.py`

- Opens with a **short problem statement** in markdown, then **Part A / Part B** (or similar)
  structure.
- Uses **`Parameters`**, **`Model`**, **`Observation`**, **`Photometry`**, **`Fitter`** with
  explicit priors and `mean_sfh_type=...` when not using `Model.from_config()`.
- Uses **`_plot_style`** for convergence, corners, SFH plots — do not duplicate those helpers
  ad hoc.

New “getting started” notebooks should read like a **light edit** of this file, not a greenfield
script.

### Models gallery track — match **very closely**

These files define the **sectioning, depth, and plotting rhythm** for physics galleries. When
adding `02_sfh_gallery.ipynb`-style content, **lift structure from these paths** (markdown
outline + numbered sections + figure-per-concept), not only from `reference/15–19`.

| Canonical `.py` | Topic |
|-----------------|-------|
| `notebooks/models/notebook_code/01_sfh_models.py` | SFH families, tables in markdown, parametric / nonparametric / stochastic / composition |
| `notebooks/models/notebook_code/02_dust_attenuation.py` | Dust attenuation: title, equation block, **numbered “Sections:” list** in the opener, then one major section per `# %% [markdown]` |
| `notebooks/models/notebook_code/03_dust_emission.py` | Dust emission models (same structural pattern as 02) |
| `notebooks/models/notebook_code/04_agn.py` | AGN components and sweeps |
| `notebooks/models/notebook_code/06_nebular.py` | Nebular backends and line physics |
| `notebooks/models/notebook_code/07_multiwavelength.py` | IGM, radio, X-ray bridge |
| `notebooks/models/notebook_code/08_radio.py` | Radio correlation and spectral index (pair with 07 for multi-λ) |

**Models opener pattern** (from `02_dust_attenuation.py` — replicate for new gallery chapters):

```markdown
# %% [markdown]
# # Model Gallery: …
#
# … short motivation + math …
#
# **Sections:**
#
# 1. …
# 2. …
# …
```

**Models import cell pattern:** same JAX/bootstrap as quickstart, then **domain imports**
(`tengri.models.dust...`, etc.), then `_plot_style`, then `FIGDIR` under `models/figures` (or
the appropriate subfolder for that track).

### Theory: forward model — **no figure writes to disk**

**File:** `notebooks/theory/notebook_code/02_forward_model.py`

- This notebook walks the differentiable pipeline (SSP → CSP → dust → photometry) with rich
  figures.
- **Policy:** do **not** persist figures under `theory/figures/` during normal runs. All
  `plt.savefig(...)` calls in this file are **commented out**; rely on `plt.show()` (or the
  Jupyter inline backend) for display.
- If you add a new figure cell, follow the same rule: **omit savefig** or keep it commented
  with a note `# plt.savefig(...)` for optional local exports.

### Source-file index for agents (add to conventions list)

When this doc refers to “copy from models”, use these **exact** paths:

- `notebooks/quickstart/notebook_code/01_quickstart.py` — quickstart / fit narrative
- `notebooks/models/notebook_code/01_sfh_models.py` … `08_radio.py` — model galleries
- `notebooks/theory/notebook_code/02_forward_model.py` — forward-model theory (savefig off)

Sync to `.ipynb` after editing `.py`: from `notebooks/`, run `jupytext --sync` on the paired
files (see project `CLAUDE.md`).

---

## What We Should Have Done First (Pre-Work Checklist)

Before writing a single notebook, these foundations should have been in place:

### 1. A Shared `sweep_parameter()` Utility

```python
# src/tengri/plotting.py — should have existed from day one
def sweep_parameter(
    model,
    param_name: str,
    values,
    *,
    ax=None,
    cmap="viridis",
    label_fmt="{:.2f}",
    log_scale: bool = False,
    components: bool = False,
) -> tuple[Figure, Axes]:
    """
    Plot how one parameter shifts the model SED, all else fixed at defaults.
    Colormapped from low→high value. This is the single most useful function
    in the documentation toolkit.
    """
```

This function is the engine behind every gallery plot. Without it, sweep notebooks are 300-line
`for` loops that are painful to read and impossible to reuse. With it, a parameter gallery is 5
lines per panel.

### 2. A Canonical Visual Language Guide

Establish once, enforce everywhere:

| Convention | Rule |
|------------|------|
| SED x-axis | Rest-frame wavelength in Å, log scale, 912–1e7 Å |
| SED y-axis | λ F_λ normalized at 5500 Å (or absolute L_ν in Lsun/Hz) |
| Colormap for sweeps | `viridis` (low→high parameter value), with colorbar |
| SFH x-axis | Lookback time in Gyr, 0 at right ("now"), age of universe at left |
| SFH y-axis | SFR in M☉/yr |
| Reference SED | Always show fiducial model in gray, swept variants in color |

Without this guide, each notebook author makes independent choices. The 15 different color schemes
currently in the notebooks are the cost of not having it.

### 3. A Model Introspection API

```python
model.param_ranges()   # → dict of {param: (lo, hi, unit, description)}
model.sweep(param, n=8)  # → list of (value, SED) pairs
```

This would let galleries be auto-generated from the model registry. Currently, parameter ranges
for sweeps are hard-coded independently in each notebook, creating drift between documentation
and implementation.

### 4. The Narrative Arc (Written, Not Assumed)

The **default outline** is already the thirteen root notebooks in order — see
[Big picture: thirteen root notebooks](#big-picture-thirteen-root-notebooks-reader-spine). That
spine is the “one-page” journey; extend it only when a user story does not fit any row in the
inventory table.

Sanity-check question (optional, for fitting-heavy edits):

> "An average observer just got JWST photometry + a medium-resolution NIRSpec spectrum of a z=2
> galaxy. What does she need to know, in what order, to trust her posterior?"

If the answer diverges from the spine, either add a **specialist** notebook (clearly labeled) or
merge the missing material into **08–10** — do not start a ninth parallel track.

---

## How to Reconcile Gallery Scripts and Notebooks

This is the key architectural question. Currently `examples/` (Sphinx gallery) and the model
notebooks both do parameter sweeps with entirely separate code, producing overlapping figures.
The answer is **clear non-overlapping jobs**:

### Sphinx Gallery (`examples/`) = Visual Parameter Reference

One script, one parameter, one figure. These are the visual API reference — the browsable
thumbnail index on the docs website. When a reader wants to know what `dust_tau_bc` does, they
click its thumbnail. Each script is 50–80 lines: imports → model at defaults → sweep one
parameter → one clean plot. No prose beyond the docstring title.

**What to add**: A script for every meaningful user-facing parameter. Currently the gallery has
29 scripts covering broad topics (all dust curves in one file, all AGN models in one file).
The refactored gallery has one script per parameter:

```
examples/
├── quickstart/
│   └── plot_first_fit.py             ← keep (mock → fit → posterior in one page)
│
├── sfh/
│   ├── plot_delayed_tau_sweep.py     ← τ from 0.5→10 Gyr
│   ├── plot_dpl_alpha_sweep.py       ← rising slope α
│   ├── plot_dpl_beta_sweep.py        ← falling slope β
│   ├── plot_lnorm_peak_sweep.py      ← log-normal peak SFR
│   ├── plot_psd_sigma_sweep.py       ← burstiness amplitude
│   └── plot_psd_tau_sweep.py         ← burstiness timescale
│
├── dust/
│   ├── plot_tau_bc_sweep.py          ← birth cloud optical depth
│   ├── plot_tau_diff_sweep.py        ← diffuse ISM optical depth
│   ├── plot_dust_slope_sweep.py      ← attenuation curve slope
│   ├── plot_bump_strength_sweep.py   ← 2175 Å UV bump (MW vs SMC)
│   ├── plot_rv_sweep.py              ← R_V (Cardelli family)
│   ├── plot_dust_temp_sweep.py       ← FIR modified blackbody T
│   ├── plot_dust_beta_sweep.py       ← emissivity index β_IR
│   ├── plot_umin_sweep.py            ← Draine & Li radiation field
│   └── plot_qpah_sweep.py            ← PAH mass fraction
│
├── nebular/
│   ├── plot_logu_sweep.py            ← ionization parameter
│   ├── plot_logz_gas_sweep.py        ← gas metallicity → line ratios
│   ├── plot_fesc_sweep.py            ← ionizing photon escape
│   ├── plot_dig_frac_sweep.py        ← DIG mixing on BPT
│   └── plot_line_sigma_sweep.py      ← emission line widths
│
├── agn/
│   ├── plot_inclination_sweep.py     ← type 1 → type 2 transition
│   ├── plot_mbh_sweep.py             ← black hole mass
│   ├── plot_ledd_sweep.py            ← Eddington ratio
│   ├── plot_spin_sweep.py            ← BH spin → radiative efficiency
│   ├── plot_torus_covering_sweep.py  ← torus covering factor
│   ├── plot_agn_alpha_ox_sweep.py    ← UV-to-X-ray slope
│   └── plot_xray_gamma_sweep.py      ← X-ray photon index
│
├── radio/
│   ├── plot_qir_sweep.py             ← FIR-radio correlation
│   ├── plot_alpha_sf_sweep.py        ← synchrotron spectral index
│   └── plot_radio_loudness_sweep.py  ← radio-quiet vs loud AGN
│
├── igm/
│   └── plot_redshift_igm_sweep.py    ← Lyman break sweeping z=0→6
│
└── metallicity/
    ├── plot_logzsol_sweep.py         ← stellar metallicity
    └── plot_alpha_fe_sweep.py        ← alpha-element enhancement
```

Each script follows this template:

```python
"""
Birth Cloud Optical Depth (τ_BC)
=================================
How does the birth cloud dust optical depth reshape the SED of
a young star-forming galaxy? Higher τ_BC reddens the UV and
suppresses nebular emission from HII regions.
"""
from tengri.plotting import sweep_parameter
from tengri import Model  # ... minimal setup

fig, ax = sweep_parameter(
    model,
    "dust_tau_bc",
    values=[0.0, 0.5, 1.0, 2.0, 3.0, 4.0],
    label_fmt="τ_BC = {:.1f}",
    cmap=SWEEP_CMAPS["dust"],
)
ax.set_title("Birth cloud optical depth τ_BC")
```

### Notebooks = Narrative Physics Intuition

Notebooks explain *why* each parameter matters, connect physics to the SED shape, and build
intuition across related parameters. They do **not** reproduce single-parameter sweeps — they
use `sweep_parameter()` for multi-panel comparisons and link to gallery pages for individual
parameters.

The key rule: **if it's a single parameter sweep, it belongs in the gallery. If it's a story
about how multiple parameters interact or what they mean physically, it belongs in a notebook.**

---

## The Quickstart: One Notebook, One Journey

The existing notebook `notebooks/quickstart/notebook_code/01_quickstart.py` is the canonical
reference. Its title is **"Differentiable SED Fitting in 10 Seconds"** and it is the entry point
for new users. **Do not replace this notebook** — refactor and sharpen it using the actual source.

The notebook demonstrates two parallel journeys:

- **Part A** — Smooth parametric SFH (D = 7): complete workflow, MAP init → geoVI → NUTS validation
- **Part B** — Bursty stochastic SFH (D = 137): same pattern, NUTS replaced by Ray Tracing

### Actual imports and bootstrap (from source)

```python
import time, warnings
import jax, jax.numpy as jnp, matplotlib.pyplot as plt, numpy as np

jax.config.update("jax_enable_x64", True)
warnings.filterwarnings("ignore", category=FutureWarning)

from tengri import (
    Fitter, Fixed, Model, Observation, Parameters, Photometry, Uniform, load_ssp_data,
)

import sys, os  # noqa: E401
# ...15-line chdir bootstrap (see bootstrap block section above)...

FIGDIR = os.path.join("quickstart", "figures")
os.makedirs(FIGDIR, exist_ok=True)

from _plot_style import (
    COLORS, SPECTRAL_FEATURES, convergence_table, plot_corner_comparison,
    plot_sfh, safe_corner, setup_style,
)
setup_style()
```

### Core API pattern (Part A — D = 7)

```python
spec_param = Parameters(
    sfh_tsnorm_log_peak_sfr=Uniform(-1.0, 2.5),
    sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),
    sfh_tsnorm_width_gyr=Uniform(0.3, 5.0),
    sfh_tsnorm_skew=Uniform(-3.0, 3.0),
    sfh_tsnorm_trunc=Uniform(1.0, 10.0),
    met_logzsol=Uniform(-2.0, 0.2),
    dust_tau_bc=Uniform(0.0, 2.0),
    dust_tau_diff=Uniform(0.0, 1.5),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.1),
    mean_sfh_type="tsnorm",
)
obs = Observation(
    photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
)
model_param = Model(spec_param, ssp_data, observation=obs)
model_param.precompute_spectroscopy(WAVE_OBS)   # WAVE_OBS = jnp.linspace(3800, 9200, 200)

mock_param = model_param.mock_spectrum(true_params_param, WAVE_OBS, snr=30.0, key=key)

fitter_param = Fitter(model_param, mock_param.flux_obs, mock_param.noise)
result_map_param  = fitter_param.run("map", n_steps=500, verbose=False)
fitter_param.compile(verbose=False)
result_geovi_param = fitter_param.run(
    "vi", n_iterations=15, n_samples=6, n_seeds=5, n_posterior_samples=10000, verbose=False,
)
result_nuts_param  = fitter_param.run(
    "mcmc_nuts", n_warmup=500, n_samples=1000, init_from=result_map_param, verbose=False,
)
convergence_table({"vi": result_geovi_param, "NUTS": result_nuts_param})
```

### Stochastic extension (Part B — D = 137)

```python
spec_stoch = Parameters(
    sfh_tsnorm_log_peak_sfr=Uniform(-1.0, 2.5),
    sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),
    sfh_tsnorm_width_gyr=Uniform(0.3, 5.0),
    sfh_tsnorm_skew=Uniform(-3.0, 3.0),
    sfh_tsnorm_trunc=Uniform(1.0, 10.0),
    sfh_field_psd_sigma=Uniform(0.1, 4.0),
    sfh_field_psd_tau_myr=Uniform(1.0, 300.0),
    met_logzsol=Uniform(-2.0, 0.2),
    dust_tau_bc=Uniform(0.0, 2.0),
    dust_tau_diff=Uniform(0.0, 1.5),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.1),
    mean_sfh_type=["tsnorm", "field"],
    n_grid=128,
)
# Model/Fitter construction identical to Part A
result_rt_stoch = fitter_stoch.run(
    "mcmc_raytrace",
    init_from=result_map_stoch,
    n_burnin=200, n_steps=2000, step_size=0.05, n_leapfrog_steps=50, verbose=False,
)
convergence_table({"vi": result_geovi_stoch, "Ray Tracing": result_rt_stoch})
```

### Figures produced (10 total)

All `plt.savefig(...)` calls must be **commented out** in refactored versions
(e.g. `# plt.savefig(os.path.join(FIGDIR, "fig01_mock_spectrum_param.png"), dpi=150, ...)`).

| Fig | Name in source | Content |
|-----|---------------|---------|
| 1 | `fig01_mock_spectrum_param.png` | Mock spectrum + spectral feature annotations |
| 2 | `fig02_spectral_fit_param.png` | Spectral fit + residual panel, reduced χ² |
| 3 | `fig03_sfh_param.png` | SFH recovery with 200 Myr inset |
| 4 | `fig04_corner_param.png` | Corner plot — parametric posterior |
| 5 | `fig05_geovi_vs_nuts.png` | geoVI vs NUTS corner comparison (D = 7 validation) |
| 6 | `fig06_bursty_truth.png` | True bursty SFH + mock spectrum (side-by-side) |
| 7 | `fig07_sfh_stochastic_money.png` | Stochastic SFH recovery — the money figure |
| 8 | `fig08_spectral_fit_stochastic.png` | Stochastic spectral fit + residuals |
| 9 | `fig09_corner_stochastic.png` | Physical parameter corner (xi latents excluded) |
| 10 | `fig10_geovi_vs_rt.png` | All methods overlay (geoVI + RT) with 200 Myr inset |

### `_plot_style` helpers used

```python
plot_sfh(model, result, true_params=..., ax=ax, color=COLORS["geovi"], label="vi", method="geoVI", show_mean_sfh=True)
plot_corner_comparison([result_geovi, result_nuts], labels=["vi", "NUTS"], colors=[...], truths=...)
safe_corner(result, truths=..., params=[p for p in spec.free_params if "xi" not in p])
convergence_table({"vi": ..., "NUTS": ..., "Ray Tracing": ...})
```

### Summary section (end of notebook)

Timing table comparing all four method/dimensionality combinations, then a "What You Just Did"
and "What's Next" markdown with links to tutorials/02–05 and demonstrations/.

---

#### `01_sed_anatomy.ipynb` — "Anatomy of a Galaxy SED"

**Source material**: `notebooks/theory/notebook_code/02_forward_model.py` (strip prose, keep
component decomp code), `notebooks/models/notebook_code/07_multiwavelength.py` (radio/X-ray SED code)
**Opening sentence**: "This notebook is a map. Every feature in a galaxy SED has a physical
cause — here you'll see each one labelled."
**First figure appears by cell 2.**

**Cell 1 — imports + fiducial model** (15 lines):
```python
# Fiducial galaxy: log M* ~ 10.5, moderate star formation, modest dust, no AGN
# Parameters chosen to show ALL features: UV, optical, NIR, MIR, FIR, radio
spec = ParamSpec(
    sfh_tsnorm_log_peak_sfr=Fixed(1.2),       # SFR ~ 15 Msun/yr
    sfh_tsnorm_peak_lbt_gyr=Fixed(4.0),        # peak 4 Gyr ago
    sfh_tsnorm_width_gyr=Fixed(2.5),
    sfh_tsnorm_skew=Fixed(0.2),
    sfh_tsnorm_trunc=Fixed(6.0),
    met_logzsol=Fixed(0.0),                    # solar metallicity
    dust_tau_bc=Fixed(1.0),                    # modest birth cloud dust
    dust_tau_diff=Fixed(0.3),
    dust_T=Fixed(35.0),                        # cool dust → FIR peak at ~85 μm
    dust_qpah=Fixed(2.5),                      # typical PAH fraction
    neb_logU=Fixed(-3.0),
    radio_q_ir=Fixed(2.64),
    redshift=Fixed(0.05),
    mean_sfh_type="tsnorm",
    nebular=True,
    dust_emission="draine_li2007",
    radio=True,
)
```

**Cell 2 — Figure 1: Full X-ray to radio SED with component labels** (Figure 2a style):
Plot λF_λ from 1 Å (X-ray) to 10 m (radio) on a log-log scale.
Show components as filled areas with different colors:
- Stellar continuum: blue
- Nebular continuum + lines: green
- Dust emission (MIR/FIR): orange
- Radio synchrotron: purple
- Total: black solid line

Annotate with `ax.annotate()` arrows pointing to:
- "Lyman break (912 Å)" at 912 Å
- "UV continuum slope" between 1216–3000 Å
- "4000 Å break" at 4000 Å
- "NIR stellar bump (1.6 μm)" at 16000 Å
- "PAH features" at 33000, 77000 Å
- "FIR peak" at the peak of the dust emission component
- "Radio synchrotron" in the radio range

Figure layout: `fig, ax = plt.subplots(1, 1, figsize=(14, 5))`.
x-axis: `ax.set_xlim(1e0, 1e11)` (Å), log scale.
y-axis: λF_λ normalized to 1 at 5500 Å.

**Cell 3 — Figure 2: Component decomposition four-panel grid**:
Figure layout: `fig, axes = plt.subplots(2, 2, figsize=(12, 8))`.
- Panel (0,0): Stellar only (no dust, no nebular)
- Panel (0,1): + nebular emission (same model, nebular=True)
- Panel (1,0): + dust attenuation (tau_bc=1, tau_diff=0.3)
- Panel (1,1): + dust emission (full model)
Each panel shows the full SED wavelength range (912–1e7 Å optical + MIR/FIR).
Gray line = previous step's SED, color line = current step's SED.
Title of each panel: "Step N: [what was added]".

**Cell 4 — Figure 3: Redshift sequence**:
Show the same fiducial SED observed at z = 0.1, 0.5, 1.0, 2.0, 4.0.
Overplot SDSS/HST/JWST filter transmission curves as gray shaded bands.
Color: use plasma colormap indexed by redshift.
x-axis: **observed-frame** wavelength 1000–50000 Å.
Axis label: "Observed wavelength (Å)".
Show how the Lyman break moves from UV into optical into NIR.
Figure layout: `fig, ax = plt.subplots(1, 1, figsize=(12, 5))`.

**Total target**: ≤ 10 markdown cells, 4 code cells, ≤ 200 lines.

---

#### `01_sfh_models.ipynb` — "Model Gallery: Star Formation Histories"

**Source file**: `notebooks/models/notebook_code/01_sfh_models.py`

This is a **physics-level gallery notebook** — it calls raw SFH functions directly.
No `Parameters`, `Model`, or `Fitter` are used. Every section demonstrates a family of SFH
shapes by calling individual functions and plotting SFR vs lookback time.

**Imports**:

```python
from tengri import (
    AGEMAX_YR, compute_sqrt_power_drw, constant_sfh, delayed_exponential_sfh,
    delayed_tau, dpl, drw_acf, drw_variance, exponential_sfh, generate_gp_fourier,
    gp_from_xi, lnorm, make_log_age_grid, norm, psd_drw, snorm, triweight_burst, tsnorm,
)
from tengri.models.sfh.chemical_evolution import closed_box_metallicity
from tengri.models.sfh.nonparametric import continuity_sfh, dirichlet_sfh
from tengri.models.sfh.psd_models import psd_extended_regulator, psd_matern
from tengri.models.sfh.registry import compute_field_gp, resolve_sfh
from tengri.utils.grid import grid_spacing
```

`_plot_style` imports: `COLORS`, `add_sfh_inset`, `setup_style`.

**FIGDIR**: `os.path.join("models", "figures")`

**Shared constants**:
```python
XLAB_LBT_GYR = r"$\mathrm{Lookback\ time\ /\ Gyr}$"
t_yr = np.linspace(10**6.0, 10**10.14, 2000)
t_gyr = t_yr / 1e9
```

**GP grid** (used in Sections 3 and 4):
```python
N_GRID = 256
log_ages = make_log_age_grid(N_GRID)
d_log_age = grid_spacing(log_ages)
```

**Local helper** (defined once, used throughout):
```python
def add_multi_sfh_inset(ax, t_gyr, y_series, colors, lws, linestyles, ylabel):
    """Creates a 200 Myr inset using inset_axes showing recent SFH detail."""
```

**Opening markdown**: table of all 14 SFH models with name, function, key parameters, and a
rule-of-thumb selection guide.

---

**Section 1 — Parametric SFH shapes**

8 figures, all named with `18_` prefix:

| Figure | Description |
|--------|-------------|
| `18_parametric_sfh_gallery.png` | Side-by-side gallery of all parametric families |
| `18_delayed_tau_vary.png` | delayed_tau sweep over τ |
| `18_dexp_vary.png` | delayed_exponential_sfh parameter sweep |
| `18_dpl_vary.png` | DPL α sweep |
| `18_tsnorm_vary.png` | tsnorm skew sweep |
| `18_gaussian_family.png` | snorm / norm / lnorm comparison |
| `18_const_exp.png` | constant_sfh vs exponential_sfh |
| `18_triweight_burst.png` | triweight_burst width and location sweep |

All `plt.savefig(...)` calls are commented out:
```python
# plt.savefig(os.path.join(FIGDIR, "18_parametric_sfh_gallery.png"), ...)
```

---

**Section 2 — Non-parametric SFH**

2 figures:

| Figure | Description |
|--------|-------------|
| `18_continuity_sfh.png` | 7-bin continuity model (Leja+2019) — SFR in each bin + prior draws |
| `18_dirichlet_sfh.png` | Dirichlet stick-breaking model (Leja+2017) — normalised mass fractions |

Uses `continuity_sfh` and `dirichlet_sfh` from `tengri.models.sfh.nonparametric`.

---

**Section 3 — Stochastic / GP SFH**

4 figures:

| Figure | Description |
|--------|-------------|
| `18_gp_sfh_demo.png` | 5 random DRW GP realisations on the shared time grid |
| `18_drw_psd_vary.png` | DRW PSD parameter grid: σ × τ sweep |
| `18_extended_regulator.png` | Extended Regulator PSD (Tacchella+2020) |
| `18_matern_psd.png` | Matern PSD comparison |

Uses `generate_gp_fourier`, `gp_from_xi`, `psd_drw`, `psd_extended_regulator`, `psd_matern`,
`compute_sqrt_power_drw`, `drw_acf`, `drw_variance` from tengri core.

---

**Section 4 — Composition**

3 figures demonstrating additive and multiplicative SFH composition:

| Figure | Description |
|--------|-------------|
| `18_composition_additive.png` | DPL + constant_sfh additive mix |
| `18_composition_burst.png` | tsnorm + triweight_burst burst-mixture |
| `18_composition_field.png` | tsnorm × exp(GP) field modulator (mean × stochastic) |

Uses `resolve_sfh` and `compute_field_gp` from `tengri.models.sfh.registry`.

---

#### `02_dust_attenuation.ipynb` — "Model Gallery: Dust Attenuation Curves"

**Source file**: `notebooks/models/notebook_code/02_dust_attenuation.py`

A comprehensive visual reference for all 14 attenuation curves in tengri. Most sections are
pure physics-level (no `Parameters`/`Model`/`Fitter`) — they call `get_dust_law(name)` directly.
The final section (Age-Dust Degeneracy) switches to the high-level API to show the astrophysical
motivation for multi-wavelength fitting.

**Inspired by**: the [dust_attenuation package documentation](https://dust-attenuation.readthedocs.io/en/latest/).

**Imports**:

```python
from tengri.models.dust.attenuation import (
    DUST_LAWS,
    get_dust_law,
    li08,
    precompute_dust_age_weights,
    two_component_dust,
    wg00_cloudy,
    wg00_dusty,
    wg00_shell,
)
# Final section only:
from tengri import Fixed, Model, Observation, Parameters, Photometry, Uniform, load_ssp_data
```

`_plot_style` imports: `COLORS`, `setup_style`.

**FIGDIR**: `os.path.join("models", "figures")`

**Wavelength grids**: `jnp.linspace(100.0, 25000.0, 2000)` for overview; per-section grids elsewhere.

**Curve groups** (for color/linestyle consistency):
- Empirical averages (solid): `calzetti`, `leitherer02`
- Modified Calzetti (dashed): `kriek_conroy`, `noll09`, `salim_sbl18`, `salim`
- Extinction MW/SMC/LMC (dotted): `cardelli`, `smc`, `lmc`
- Physics-motivated (dash-dot): `tea`, `conroy2010`, `narayanan_z`
- Parametric (solid thin): `power_law`, `li08`

---

**Section 1 — Overview**

All 14 curves at default parameters on a single figure with family grouping by linestyle.

| Figure | Description |
|--------|-------------|
| `15_all_attenuation_curves.png` | All 14 k(λ) curves, 900–8000 Å, normalized at 5500 Å |

---

**Section 2 — Empirical Average Curves**

Calzetti (C00) vs Leitherer (L02). Highlights the L02 far-UV extension region (970–1200 Å) with
a shaded span.

| Figure | Description |
|--------|-------------|
| `15_calzetti_vs_leitherer.png` | C00 vs L02 with UV extension highlighted |

---

**Section 3 — Modified Calzetti Family**

KC13, N09, SBL18 at δ = −0.3, E_b = 1.5. Layout: main panel (1000–10000 Å) + UV zoom panel
(1000–4000 Å). Documents the ordering difference: KC13 applies slope then adds bump; N09 adds
bump then applies slope.

| Figure | Description |
|--------|-------------|
| `15_modified_calzetti_family.png` | 2-panel: main + UV zoom with 2175 Å region highlighted |

---

**Section 4 — MW / SMC / LMC Extinction Curves**

Cardelli CCM89 (R_V=3.1), SMC (Pei 1992), LMC (Pei 1992). Annotated with bump strength labels.

| Figure | Description |
|--------|-------------|
| `15_mw_smc_lmc.png` | Three extinction curves with 2175 Å bump region shaded |

---

**Section 5 — Physics-Motivated Curves**

3-panel figure:
- TEA (Haskell+2024): δ sweep showing the correlated E_b(δ) = 2.5 exp(3.5δ) relation
- Conroy+2010: MW+power-law blend with Cardelli and power-law references overlaid
- Narayanan+2018: z = [0, 1, 3, 6] redshift evolution

| Figure | Description |
|--------|-------------|
| `15_physics_motivated_curves.png` | 3-panel: TEA / Conroy2010 / Narayanan z-evolution |

---

**Section 6 — Parameter Exploration**

Systematic parameter sweeps for each major curve.

| Figure | Description |
|--------|-------------|
| `15_calzetti_tau_sweep.png` | Transmission T(λ) = exp(−τ_V·k) for τ_V in [0.1, 0.3, 0.5, 1, 2, 4] |
| `15_kc13_delta_sweep.png` | KC13: slope δ sweep at E_b = 0 |
| `15_kc13_bump_sweep.png` | KC13: bump E_b sweep at δ = 0, x-axis 1000–6000 Å |
| `15_cardelli_rv_sweep.png` | Cardelli: R_V in [2.0, 2.5, 3.1, 4.0, 5.0, 5.5] |
| `15_uv_zoom_comparison.png` | UV zoom (1000–4000 Å): Calzetti / SMC / LMC / MW |
| `15_li08_parameter_sweeps.png` | Li+2008: 2×2 grid sweeping c1, c2, c3, c4 independently |
| `15_li08_presets.png` | Li+2008: literature presets (MW-like, SMC-like, Calzetti-like) |

The Li+2008 section includes a correction note: the published L08 parametrization (Eq. 1) uses
four dimensionless coefficients c1–c4, not separate FUV/UV/optical power laws. The notebook
documents this explicitly.

---

**Section 7 — Dust Geometries (WG00)**

Shell (foreground screen), cloudy (homogeneous mix), dusty (clumpy medium) from Witt & Gordon
(2000). Shows that clumpy geometry greys the effective attenuation.

| Figure | Description |
|--------|-------------|
| `15_wg00_geometries.png` | 3-panel (τ_V = 1, 2, 4): shell/cloudy/dusty transmission curves |
| `15_greying_effect.png` | At τ_V=3: left=transmission, right=effective k_eff showing greying |

---

**Section 8 — Two-Component Dust Model**

Charlot & Fall (2000) model with birth cloud + diffuse ISM separation.

| Figure | Description |
|--------|-------------|
| `15_sigmoid_transition.png` | Birth cloud weight w(age) sigmoid for t_birth in [1, 5, 10, 30, 100] Myr |
| `15_two_component_transmission.png` | Transmission by stellar age (1 Myr–5 Gyr) at τ_bc=1.0, τ_diff=0.5 |
| `15_two_component_vary.png` | 2-panel: left = vary τ_bc, right = vary τ_diff (solid=1 Myr, dotted=1 Gyr) |

---

**Section 9 — Summary Table**

Markdown table of all 14 curves with reference, free parameters, UV bump, and recommended use case.

---

**Age-Dust Degeneracy** (uses high-level API)

Computes r−i color on a grid of (τ_diff, peak lookback time) using `Parameters`, `Fixed`, `Model`,
`load_ssp_data`. Shows iso-color contours to illustrate that old low-dust and young dusty galaxies
are degenerate in broadband colors.

```python
ssp_data = load_ssp_data("data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")
obs_ri = Observation(photometry=Photometry.from_names(["sdss_r", "sdss_i"]))
# ... loop over (tau_diff, peak_lbt_gyr) grid ...
```

| Figure | Description |
|--------|-------------|
| `03_age_dust_degeneracy.png` | Contourf: iso-color contours in τ_diff vs peak_lbt_gyr space |

All `fig.savefig(...)` and `plt.savefig(...)` calls are commented out:
```python
# fig.savefig(os.path.join(FIGDIR, "15_all_attenuation_curves.png"), ...)
```

---

#### `03_dust_emission.ipynb` — "Dust Emission Model Gallery"

**Source file**: `notebooks/models/notebook_code/03_dust_emission.py`

A visual reference for all 10 dust emission models in tengri. Covers analytic models
(modified blackbody, Casey 2012, MAGPHYS), physically-motivated template libraries
(DL07, DL14, Dale+2014, Astrodust, BOSA, THEMIS), the energy-balance-split model,
and CMB-correction effects. All models are pure JAX (JIT-compatible, fully differentiable).

**Imports**:

```python
from tengri.models.dust.emission import (
    DUST_EMISSION_MODELS,
    _draine_li2007_analytic_fallback, _draine_li2014_analytic_fallback,
    _dale2014_analytic_fallback, _astrodust_analytic_fallback,
    _bosa_analytic_fallback, _themis_analytic_fallback,
    _pah_template, _PAH_CENTER_UM, _drude_profile, _PAH_FWHM_UM, _PAH_STRENGTH,
    _modified_blackbody_component,
    cmb_corrected_temperature, cmb_contrast_factor,
    energy_balance_split, get_emission_model,
    magphys_dc08, modified_blackbody, casey2012, planck_bnu,
)
```

**Shared setup**:

```python
# Wavelength grid: 1–1000 µm in Angstrom
wave_aa = jnp.logspace(np.log10(1e4), np.log10(1e7), 2000)
wave_um = wave_aa * 1e-4  # for plotting in microns

L_ABS = 1e10  # Lsun — fiducial total absorbed luminosity

FIGDIR = os.path.join("models", "figures")
```

**Color/label dicts** (used across all model-comparison figures):

```python
MODEL_COLORS = {
    "modified_blackbody": "#1f77b4",
    "casey2012": "#ff7f0e",
    "magphys": "#2ca02c",
    "draine_li2007": "#d62728",
    "draine_li2014": "#9467bd",
    "dale2014": "#8c564b",
    "astrodust": "#e377c2",
    "bosa": "#17becf",
    "themis": "#bcbd22",
    "energy_balance_split": "#7f7f7f",
}
MODEL_LABELS = {
    "modified_blackbody": "Modified BB",
    "casey2012": "Casey (2012)",
    "magphys": "MAGPHYS (dC+08)",
    "draine_li2007": "DL07",
    "draine_li2014": "DL14",
    "dale2014": "Dale+2014",
    "astrodust": "Astrodust+PAH",
    "bosa": "BOSA (B&S21)",
    "themis": "THEMIS (J+17)",
    "energy_balance_split": "Energy Balance Split",
}
```

**Local helper**:

```python
def _set_reasonable_log_ylim(ax, pad_log=0.12):
    """Tighten log-scale y limits from line data within the axis x range."""
    ...
```

---

**Section 1 — Overview: All Models at T=35 K**

Single figure comparing every emission model at `L_ABS = 1e10 Lsun` and T=35 K (where applicable).
Template-based models use their analytic fallbacks for guaranteed availability.
`warnings.filterwarnings("ignore", message=".*Template file.*not found.*")` suppresses fallback warnings.

| Figure | Description |
|--------|-------------|
| `16_overview_all_models.png` | All 10 models on one axes, `figsize=(9, 5.5)`, x-axis in µm (log), y-axis Lν/Lsun/Hz (log) |

---

**Section 2a — Modified Blackbody**

Two-panel sweep figure: left = dust temperature T sweep [20, 25, 30, 35, 40, 45 K], right = emissivity index β sweep [1.2, 1.5, 1.8, 2.0, 2.2].

| Figure | Description |
|--------|-------------|
| `16_modified_blackbody.png` | `fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))` |

---

**Section 2b — Casey (2012)**

Two-panel: left = MBB vs Casey 2012 comparison at same T/β, right = mid-IR power-law index `α_mir` sweep [1.0, 1.5, 2.0, 2.5, 3.0].

| Figure | Description |
|--------|-------------|
| `16_casey2012.png` | `fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))` |

---

**Section 2c — MAGPHYS 4-Component**

Four-panel figure: top-left = 4 components (MBB hot/warm/cold + PAH), top-right = PAH Drude profiles zoomed 2–15 µm, bottom-left = ξ_PAH sweep [0.02, 0.04, 0.06, 0.1, 0.15], bottom-right = T_warm + T_cold sweep.

Uses `_pah_template`, `_drude_profile`, `_PAH_CENTER_UM`, `_PAH_FWHM_UM`, `_PAH_STRENGTH`, `_modified_blackbody_component` internals.

| Figure | Description |
|--------|-------------|
| `16_magphys_components.png` | `fig, axes = plt.subplots(2, 2, figsize=(12, 9))` |

---

**Section 3a — Draine & Li (2007)**

Three-panel sweep: left = qPAH [0.5, 1.5, 2.5, 3.5, 4.5 %], centre = U_min [0.1, 0.3, 1.0, 3.0, 10.0], right = γ (fraction in PDR) [0.001, 0.01, 0.05, 0.1, 0.3].
Falls back to `_draine_li2007_analytic_fallback` if `data/dl07_templates.npz` is absent.

| Figure | Description |
|--------|-------------|
| `16_draine_li2007.png` | `fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))` |

---

**Section 3b — Draine & Li (2014)**

α_dl14 sweep [1.5, 2.0, 2.5, 3.0] compared against DL07 reference.
Falls back to `_draine_li2014_analytic_fallback` if `data/dl14_templates.npz` is absent.

| Figure | Description |
|--------|-------------|
| `16_draine_li2014.png` | Single panel, `figsize=(8, 4.5)` |

---

**Section 3c — Dale+2014**

α sweep [1.0, 1.5, 2.0, 2.5, 3.0] showing the ISM heating intensity distribution.

| Figure | Description |
|--------|-------------|
| `16_dale2014.png` | Single panel, `figsize=(8, 4.5)` |

---

**Section 3d — Astrodust+PAH**

Astrodust model compared to DL07 at same parameters; highlights the updated dust-grain model.

| Figure | Description |
|--------|-------------|
| `16_astrodust.png` | Single panel, `figsize=(8, 4.5)` |

---

**Section 3e — BOSA**

sSFR sweep [0.01, 0.1, 1.0, 10.0, 100.0 Gyr⁻¹] showing the star-formation-activity-dependent SED shape.

| Figure | Description |
|--------|-------------|
| `16_bosa.png` | Single panel, `figsize=(8, 4.5)` |

---

**Section 3f — THEMIS**

Two-panel: left = THEMIS vs DL07 comparison, right = q_hac (hydrocarbon abundance) sweep [0.1, 0.2, 0.3, 0.4, 0.5].

| Figure | Description |
|--------|-------------|
| `16_themis.png` | `fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))` |

---

**Section 4 — Energy Balance Split**

Two-panel: left = warm+cold decomposition (η_warm, η_cold fractions), right = AGN IR contribution via `agn_ir_frac` and `eta_balance` departure from strict energy balance.

| Figure | Description |
|--------|-------------|
| `16_energy_balance.png` | `fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))` |

---

**Section 5 — CMB Corrections**

Two-panel: left = effective temperature T_eff vs intrinsic T_dust at z=[0, 1, 2, 4, 6] using `cmb_corrected_temperature`, right = FIR SEDs showing CMB suppression of the Rayleigh-Jeans tail via `cmb_contrast_factor`.

| Figure | Description |
|--------|-------------|
| `16_cmb_corrections.png` | `fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))` |

---

**Section 6 — Summary Table**

Printed text summary + rendered matplotlib table of all 10 models listing: model name, free parameters, template required, CMB correction support, and recommended use case.

| Figure | Description |
|--------|-------------|
| `16_summary_table.png` | `fig, ax = plt.subplots(figsize=(14, 5))`, matplotlib `table()` with styled header and alternating row colors |

---

**Notes** (markdown cell at end of notebook):

- Template-based models (DL07, DL14, Dale+2014, Astrodust, BOSA, THEMIS) auto-load tabulated grids
  from `data/` on first call. If templates are not found, crude analytic fallbacks are used with a
  warning. The fallbacks shown in this notebook are **not suitable for science**.
- All models enforce energy balance: ∫ Lν dν = L_absorbed.
- CMB corrections (da Cunha+2013) are applied automatically when `redshift > 0`. This affects the
  effective dust temperature and suppresses the Rayleigh-Jeans tail.
- The `energy_balance_split` model extends simple energy balance with warm/cold decomposition
  and optional AGN IR contribution. `eta_balance` allows departures from strict energy balance
  (spatial offsets between UV and FIR emission regions).

All `fig.savefig(...)` calls are commented out:
```python
# fig.savefig(os.path.join(FIGDIR, "16_overview_all_models.png"), dpi=150, bbox_inches="tight")
```

---

#### `04_nebular_gallery.ipynb` — "Nebular Emission: Ionization, Metallicity, and Line Diagnostics"

**Source material**:
- `notebooks/models/notebook_code/06_nebular.py` — best source; CLOUDY grid sweeps, Cue sweeps
- `notebooks/specialist/notebook_code/05_emission_line_marginalization.py` — marginalization section

**Opening sentence**: "When hot young stars ionize their surrounding gas, the gas glows — and
those emission lines encode the chemistry, density, and ionization state of the interstellar medium."

**Part 1 — What nebular emission looks like** (~40 lines):

Figure 1: Stellar vs stellar+nebular SED.
Layout: `fig, axes = plt.subplots(1, 2, figsize=(14, 4))`.
Left: Broadband SED (1000–10000 Å). Gray = stellar only. Green = stellar + nebular.
      Annotation arrows: "Lyα 1216 Å", "Hβ 4861 Å", "Hα 6563 Å", "[OIII] 5007 Å".
Right: Zoom to 4700–7000 Å at R~1000 resolution showing the optical forest clearly.
Use a young galaxy: `peak_lbt_gyr=0.5`, `width_gyr=0.3`, `tau_bc=0.2`, `logU=-3.0`.

**Part 2 — Ionization parameter logU** (~50 lines):

Figure 2: logU sweep + BPT impact.
Layout: `fig, axes = plt.subplots(1, 2, figsize=(12, 4))`.
Left: SED 4500–7500 Å for logU in [-4.0, -3.0, -2.0, -1.0]. Colormap: `Greens` (dark=high U).
      Shows [OIII]/Hβ ratio changing dramatically.
Right: BPT diagram ([OIII]/Hβ vs [NII]/Hα). Mark each logU as a colored dot.
       Show Kauffmann+2003 star-forming sequence as a gray dashed line.
       Caption: "logU is the ionization parameter — the ratio of ionizing photon density to gas
       density. High logU → strong [OIII] → upper BPT."

Figure 3: Gas metallicity sweep.
Layout: `fig, ax = plt.subplots(1, 1, figsize=(8, 4))`.
Sweep `neb_logZ_gas` in [-1.5, -0.7, 0.0, 0.3] (Z/Zsun). Colormap: `YlGn`.
x-axis: 4500–7500 Å. Shows [NII]/Hα changing with metallicity (primary metallicity diagnostic).

Figure 4: Escape fraction + DIG mixing.
Layout: `fig, axes = plt.subplots(1, 2, figsize=(12, 4))`.
Left: fesc sweep [0.0, 0.2, 0.5, 0.8, 1.0]. Colormap: `Purples`. Line flux decreases.
Right: DIG fraction sweep [0.0, 0.2, 0.5, 0.8]. On BPT diagram, show how DIG moves
       galaxies toward the LINER/composite region.

**Part 3 — Line widths** (~30 lines):

Figure 5: Velocity dispersion sweep at spectrum resolution.
Layout: `fig, ax = plt.subplots(1, 1, figsize=(10, 4))`.
Sweep `eline_sigma_kms` in [50, 100, 200, 500, 1000] km/s. Colormap: `Blues`.
Zoom to 6450–6650 Å to show Hα+[NII] complex broadening and blending.
Caption: "At σ > 300 km/s (AGN-scale), Hα and [NII] blend and individual lines are unresolved."

**Part 4 — Marginalization** (~30 lines):

No figure for this — just a code demonstration with text output showing log-likelihoods.
Show: `result_fixed = model.fit(..., eline_mode="fixed")` vs
      `result_marginalized = model.fit(..., eline_mode="marginalized")`.
Print `result_fixed.summary_table()` vs `result_marginalized.summary_table()`.
Markdown: "Marginalization analytically integrates out line amplitudes — faster and numerically
more stable than fitting them as free parameters."

**Total target**: ≤ 15 markdown cells, ≤ 7 code cells, ≤ 250 lines.

---

#### `05_agn_gallery.ipynb` — "Active Galactic Nuclei: Disc, Torus, and Unified Model"

**Source material**: `notebooks/models/notebook_code/04_agn.py` (best source)

**Opening sentence**: "An active galactic nucleus can outshine the entire stellar population
of its host galaxy — here's how to recognize one in a SED and what its parameters mean."

**Figure 0 (opener)**: Star-forming galaxy SED vs Type 1 AGN SED at z=1.
Layout: `fig, ax = plt.subplots(1, 1, figsize=(12, 5))`.
Blue = star-forming galaxy (no AGN). Orange = same galaxy + AGN (`agn_log_lbol=45.5`,
`cos_inc=0.7`). x-axis: 1000–1e6 Å. Annotation: "AGN power-law disc", "Torus emission",
"Big blue bump", "Soft X-ray excess".

**Part 1 — Inclination: type 1 vs type 2** (~40 lines):

Figure 1: inclination sweep (the single most physically intuitive AGN plot).
Layout: `fig, axes = plt.subplots(1, 2, figsize=(14, 5))`.
Left: SED 1000 Å – 100 μm for `cos_inc` in [0.95, 0.7, 0.5, 0.3, 0.1]. Colormap: `RdYlBu`.
     Label: cos(i)=0.95 (face-on, type 1) to cos(i)=0.1 (edge-on, type 2).
     Show how UV disc disappears and MIR torus emission takes over.
Right: Schematic cartoon (matplotlib patches) showing disc + torus cross-section with
     sightlines for each inclination. This makes the geometry physically intuitive.

**Part 2 — Black hole and accretion** (~60 lines):

Figure 2: M_BH + L/L_Edd combined.
Layout: `fig, axes = plt.subplots(1, 2, figsize=(12, 4))`.
Left: `agn_log_mbh` sweep [6.5, 7.0, 7.5, 8.0, 8.5]. Colormap: `Purples`.
Right: `agn_log_ledd` sweep [-2.0, -1.0, -0.5, 0.0]. Colormap: `Oranges`.
Caption: "M_BH sets the disc temperature (hotter disc = bluer big blue bump). L/L_Edd sets
the luminosity. Together they determine the UV-optical SED shape."

Figure 3: Black hole spin.
Layout: `fig, ax = plt.subplots(1, 1, figsize=(8, 4))`.
Sweep `agn_spin` in [0.0, 0.5, 0.9, 0.998]. Colormap: `plasma`.
x-axis: 100–10000 Å. Caption: "Higher spin = smaller ISCO radius = higher radiative efficiency
η = bluer, more luminous disc. Maximally spinning BH converts ~42% of rest mass to radiation."

**Part 3 — Torus parameters** (~40 lines):

Figure 4: Torus covering factor + optical depth.
Layout: `fig, axes = plt.subplots(1, 2, figsize=(12, 4))`.
Left: `agn_torus_frac` sweep [0.1, 0.3, 0.5, 0.7, 0.9]. Colormap: `OrRd`.
     x-axis: 1–100 μm. Caption: "Higher covering factor = more MIR emission."
Right: `agn_tau_skirtor` sweep [3, 7, 11, 15]. Same x-axis.

**Part 4 — X-ray** (~30 lines):

Figure 5: X-ray spectral parameters.
Layout: `fig, axes = plt.subplots(1, 2, figsize=(12, 4))`.
Left: `xray_gamma_agn` sweep [1.5, 1.8, 2.1, 2.5]. x-axis: 0.1–100 keV (Å converted).
Right: `xray_alpha_ox` sweep [-1.8, -1.4, -1.0]. x-axis: 1000 Å – 10 keV.
Caption: "α_ox is the slope between 2500 Å and 2 keV — it determines the X-ray loudness of
an AGN relative to its UV disc."

**Total target**: ≤ 15 markdown cells, ≤ 8 code cells, ≤ 280 lines.

---

#### `06_multiwavelength_gallery.ipynb` — "Multi-Wavelength Coverage: IGM, Radio, and X-ray"

**Source material**:
- `notebooks/models/notebook_code/05_igm.py` (30 lines, trivial)
- `notebooks/models/notebook_code/07_multiwavelength.py`
- `examples/advanced/plot_radio_xray.py`

**Opening sentence**: "Beyond the UV-optical, three additional windows tell you things you can't
learn any other way: the IGM stamps high-z photometry, radio traces star formation and AGN jets,
and X-ray pins down accretion."

**Part 1 — IGM** (~40 lines):

Figure 1: IGM Lyman break as a function of redshift.
Layout: `fig, axes = plt.subplots(1, 2, figsize=(14, 4))`.
Left: SED in observed-frame Å for z in [0.1, 0.5, 1.0, 2.0, 4.0, 6.0]. Colormap: `plasma`.
     Overplot gray shaded bands for SDSS u/g/r/i/z and JWST F090W/F150W/F200W filters.
     Caption: "The Lyman break at 912 Å rest-frame shifts into SDSS-r at z~3 and JWST-F090W at z~9."
Right: Lyman-break color (u−g, g−r, etc.) vs redshift showing the dropout criterion.

**Part 2 — Radio** (~70 lines):

Figure 2: FIR-radio correlation — q_IR sweep.
Layout: `fig, axes = plt.subplots(1, 2, figsize=(12, 4))`.
Left: Full SED 1 μm – 1 m for `radio_q_ir` in [2.0, 2.4, 2.64, 3.0]. Colormap: `Blues`.
     Caption: "q_IR is the log ratio of FIR to 1.4 GHz radio luminosity. Lower q_IR = more
     radio-loud relative to FIR. The FIR-radio correlation (q_IR ≈ 2.64) holds for star-forming
     galaxies because both trace the same young stellar population."
Right: Zoom to 1 cm – 1 m showing the radio regime only. Overplot observed 1.4 GHz data point.

Figure 3: Radio spectral index + AGN contribution.
Layout: `fig, axes = plt.subplots(1, 2, figsize=(12, 4))`.
Left: `radio_alpha_sf` sweep [0.5, 0.7, 0.8, 1.0] for SF-dominated radio. Colormap: `Purples`.
Right: Radio-loudness sweep [0, 1, 2, 3] (log scale, AGN jet contribution). Colormap: `Reds`.
      Show how AGN radio dominates at high `radio_loudness`.

**Part 3 — Already covered in 05_agn** (skip X-ray here, just a one-sentence cross-reference):
"X-ray AGN parameters are shown in the AGN gallery notebook (05_agn_gallery)."

**Total target**: ≤ 10 markdown cells, ≤ 5 code cells, ≤ 180 lines.

---

#### `07_fitting_photometry.ipynb` — "Photometric SED Fitting: Single Galaxy to Catalog Scale"

**Source material**: `notebooks/quickstart/notebook_code/01_quickstart.py` (batch pattern),
`notebooks/specialist/notebook_code/07_advanced_spectroscopy.py` (joint fit pattern)

**Opening sentence**: "Fitting photometry means finding the range of SFHs, dust levels, and
metallicities consistent with your flux measurements — here's the complete workflow."

**Section 1 — Single galaxy fit** (~50 lines):
Use the exact same mock as `00_quickstart.ipynb` (same PRNGKey, same model) so the reader
recognizes it. But now show the FULL posterior workflow:
Figure 1: Corner plot of posterior (`result.corner()`).
Figure 2: SED overlay with 68% and 95% posterior bands.
Figure 3: SFH recovery panel.
Layout for Figures 2+3: `fig, axes = plt.subplots(1, 2, figsize=(14, 5))`.

**Section 2 — Joint photometry + spectroscopy fit** (~50 lines):
Source: `notebooks/specialist/notebook_code/07_advanced_spectroscopy.py`. Show how adding a spectrum tightens the posterior.
Figure 4: Side-by-side corner plot or table: phot-only vs joint posteriors.
Layout: `fig, axes = plt.subplots(1, 2, figsize=(14, 5))`. Left=phot, right=joint.
Emphasize: "Adding even low-resolution spectroscopy dramatically breaks the age-dust degeneracy."

**Section 3 — Batch fitting: 100 galaxies** (~40 lines):
Source: `notebooks/quickstart/notebook_code/01_quickstart.py` batch/vmap section.
```python
# Generate 100 mock galaxies
mocks = model.mock_batch(jax.random.PRNGKey(0), n=100, snr=15.0)
# Fit all in one call
results = model.fit_batch(mocks.flux_obs, mocks.noise, method="vi")
```
Figure 5: Scatter plot of true vs recovered stellar mass for 100 galaxies.
Layout: `fig, ax = plt.subplots(1, 1, figsize=(6, 6))`. x=true log M*, y=recovered log M*.
Gray diagonal = perfect recovery. Color = dust optical depth.
Caption: "tengri fits 100 galaxies in parallel via JAX vmap. The compile cost is paid once."

**Section 4 — Choosing your filter set** (~30 lines):
Filter set comparison (draw from `notebooks/specialist/notebook_code/03_model_checking.py`):
Figure 6: Posterior width for 3 filter configurations — SDSS only vs SDSS+2MASS vs SDSS+2MASS+WISE.
Layout: `fig, axes = plt.subplots(1, 3, figsize=(15, 4))`.
Each panel: posterior PDF for stellar mass (marginalized). Show how it tightens with more bands.

**Total target**: ≤ 15 markdown cells, ≤ 8 code cells, ≤ 300 lines.

---

#### `08_fitting_spectra.ipynb` — "Spectroscopic Fitting: Resolution, Noise, and Calibration"

**Source material**: `notebooks/specialist/notebook_code/07_advanced_spectroscopy.py`

**Opening sentence**: "A spectrum gives you more than photometry: individual line ratios constrain
metallicity and ionization, and the continuum shape constrains dust reddening and age simultaneously."

**Section 1 — Spectrum fit basics** (~60 lines):
Generate a 200-pixel mock spectrum (R~2000, 4000–9000 Å).
```python
spec_config = SpectroscopyConfig(
    wave_obs=jnp.linspace(4000, 9000, 200),
    eline_mode="marginalized",
)
obs = Observation(spectroscopy=spec_config)
result = model.fit(mock_spec.flux_obs, mock_spec.noise)
```
Figure 1: Spectrum fit. Layout: `fig, axes = plt.subplots(2, 1, figsize=(14, 7))`.
Top: Observed spectrum (black errorbars) + posterior median (color) + 68% band (fill).
Bottom: Residuals (data − model) / noise. Should be white noise within ±2σ.
Annotation arrows on key features: Hβ, [OIII], Hα, [NII], [SII].

**Section 2 — What spectra constrain** (~40 lines):
Figure 2: Posterior comparison (spec vs phot).
Layout: `fig, axes = plt.subplots(1, 4, figsize=(16, 4))`.
4 panels = 4 parameters: met_logzsol, dust_tau_bc, SFR, stellar_mass.
Each panel: phot posterior (gray fill) vs spec posterior (color fill).
Caption under met_logzsol: "Metallicity is unconstrained by photometry alone (age-metallicity
degeneracy) but the 4000 Å break and Mg lines in the spectrum pin it down."

**Section 3 — Noise models** (~50 lines):
Source: `notebooks/specialist/notebook_code/07_advanced_spectroscopy.py`.

Figure 3: Gaussian vs Student-t noise model on a spectrum with outlier pixels.
Layout: `fig, axes = plt.subplots(1, 2, figsize=(14, 5))`.
Left: Gaussian fit. Outlier pixels pull the fit. Right: Student-t fit (ν=3). Outliers ignored.
```python
spec_gaussian = SpectroscopyConfig(wave_obs=wave, eline_mode="marginalized", noise_dof=0.0)
spec_studentt = SpectroscopyConfig(wave_obs=wave, eline_mode="marginalized", noise_dof=3.0)
```

Figure 4: Calibration polynomial marginalization.
Layout: `fig, axes = plt.subplots(1, 2, figsize=(12, 4))`.
Left: Spectrum with a multiplicative calibration error (5% slope across the band).
Right: Posterior on calibration polynomial coefficients vs true values.
```python
spec_cal = SpectroscopyConfig(wave_obs=wave, eline_mode="marginalized", n_cal_poly=4)
```

**Total target**: ≤ 15 markdown cells, ≤ 8 code cells, ≤ 280 lines.

---

#### `09_degeneracies.ipynb` — "Parameter Degeneracies and Information Content"

**Source material**: `notebooks/specialist/notebook_code/03_model_checking.py` (keep Fisher matrix content)

**Opening sentence**: "Not all parameters are equally constrained by your data — here's how to
map the degeneracies before you fit, so you're not surprised by the posterior."

**Remove**: filter set comparison (move to `07_fitting_photometry`).
**Keep**: Fisher information matrix analysis and age-dust-metallicity degeneracy visualization.

Figure 1: Fisher information ellipses for SDSS-only fit.
Figure 2: 2D marginalized posteriors showing age-dust-metallicity degeneracy triangle.
Figure 3: Visual table — "what constrains what": rows = parameters, columns = data types
(UV phot, optical phot, NIR phot, R~200 spectrum, R~2000 spectrum). Cell = color coding from
red (unconstrained) to blue (well-constrained).

**Total target**: ≤ 10 markdown cells, ≤ 5 code cells, ≤ 150 lines.

---

#### `10_real_data.ipynb` — "End-to-End Fitting: SDSS Data to Physical Posteriors"

**Source material**: `notebooks/specialist/notebook_code/01_real_data.py`,
`notebooks/specialist/notebook_code/02_derived_quantities.py`,
`notebooks/specialist/notebook_code/03_model_checking.py`

**Opening sentence**: "Everything in the previous notebooks worked with mock data — here we
process an actual SDSS galaxy from raw FITS to physical posteriors."

**Use**: SDSS galaxy SDSS J114816.64+525150.3 (a z=0.1 Seyfert 2 with measured properties),
or any well-characterized SDSS object with both spectroscopy and photometry available.

Section 1 — Load and preprocess (units, masking, noise estimation).
Section 2 — Fit: `result = model.fit(phot=(flux, noise), spectrum=(spec_flux, spec_noise))`.
Section 3 — Posterior predictive check: `result.plot_posterior_predictive()`.
Section 4 — Derived quantities: `result.stellar_mass()`, `result.sfr()`, `result.line_fluxes()`.
Section 5 — BPT diagram: `result.bpt_nii()`.

**Total target**: ≤ 12 markdown cells, ≤ 8 code cells, ≤ 250 lines.

---

#### `11_population.ipynb` — "Hierarchical Inference: Shared Priors Across Galaxy Populations"

**Source material**: `notebooks/specialist/notebook_code/06_simulation_sfh.py` (shorten to ~400 lines)

**Opening sentence**: "When you fit many galaxies with a shared prior, each galaxy's posterior
improves — the population teaches each individual object what's physically plausible."

Keep: block Gibbs structure, PSD recovery figure, population posterior plots.
Cut: all derivations of the block Gibbs update equations (move to a callout box).
Cut: 100-galaxy demo (use 20 galaxies, same scientific point, 5× less compute).

Figure 1: Intuition diagram (matplotlib.patches only, no data needed).
Show 20 small panels = 20 SFH posteriors. Left half: independent fits (wide posteriors).
Right half: hierarchical fits (narrowed posteriors). Arrow between halves: "shared PSD prior".

**Total target**: ≤ 12 markdown cells, ≤ 7 code cells, ≤ 400 lines.

---

#### `12_extending_tengri.ipynb` — "Custom Components: Priors, SFH Models, and Dust Laws" *(minor polish only)*

Source: `notebooks/specialist/notebook_code/04_extending_tengri.py`. Already well-structured.
Only change: move SSP loading boilerplate into `_plot_style.py` helper so the notebook
opens immediately with the custom component design, not 30 lines of setup.

---

## The Reusable Sweep Infrastructure (Build First)

The central investment that pays dividends across all 6 gallery notebooks:

```python
# src/tengri/plotting.py — new additions

def sweep_parameter(
    model,
    param_name: str,
    values,
    *,
    ax=None,
    cmap="viridis",
    label_fmt: str = "{:.2f}",
    unit: str = "",
    log_scale: bool = False,
    components: bool = False,
    reference_idx: int = None,
) -> tuple[Figure, Axes]:
    """
    Sweep one parameter across `values`, plot resulting SEDs colormapped from
    low to high value. Reference model (index `reference_idx`) shown in gray.
    """

def parameter_gallery(
    model,
    param_sweep_specs: list[dict],
    *,
    ncols: int = 3,
    figsize_per_panel=(4, 3),
) -> Figure:
    """
    Multi-panel gallery: one panel per parameter in `param_sweep_specs`.
    Each spec: {"param": "dust_tau_bc", "values": [0, 1, 2, 3], "label": "τ_BC"}.
    """

def sfh_sed_comparison(
    model,
    param_name: str,
    values,
    *,
    cmap="plasma",
) -> Figure:
    """
    Two-panel: SFH realizations (left) + corresponding SEDs (right).
    Essential for SFH gallery and the burstiness story.
    """
```

**Why build this first**: With `sweep_parameter()` in place, each gallery notebook is
~50 lines of code to produce publication-quality figures. Without it, each notebook is
300 lines of boilerplate that obscures the physics.

---

## Visual Language Specification

Establish once, import from `plotting.py` everywhere:

```python
# In plotting.py:
SED_XLIM = (912, 1e7)    # Å, rest-frame
SED_XSCALE = "log"
SED_YLABEL = r"$\lambda F_\lambda$ (normalized at 5500 Å)"
SED_XLABEL = r"Rest-frame wavelength (Å)"

SFH_XLABEL = "Lookback time (Gyr)"
SFH_YLABEL = r"SFR (M$_\odot$ yr$^{-1}$)"

SWEEP_CMAPS = {
    "dust":      "YlOrRd",    # yellow→red for reddening
    "agn":       "PuRd",      # purple→red for AGN dominance
    "sfh":       "Blues",     # light→dark for SFH variation
    "nebular":   "Greens",    # for ionization
    "radio":     "cool",      # blue→purple for radio
    "redshift":  "plasma",    # for redshift sweeps
}

REFERENCE_STYLE = dict(color="0.75", lw=1.5, zorder=0, label="reference")
```

---

## What to Delete (Not Archive)

These notebooks add words without adding insight. Their content exists elsewhere in better form:

| Delete | Content absorbed into |
|--------|----------------------|
| `notebooks/theory/notebook_code/02_forward_model.py` | `01_sed_anatomy.ipynb` (same content, visualized first) |
| `notebooks/quickstart/notebook_code/02_tengri_capabilities.py` | The landing page and `01_sed_anatomy.ipynb` |
| `notebooks/specialist/notebook_code/06_simulation_sfh.py` | `02_sfh_gallery.ipynb` (forward modeling is just running the gallery) |
| *(fitting/ notebooks — retired)* | Content merged into specialist notebooks above |


Reduce from 28 notebooks to 12. Every cut is a reader's time saved.

---

## Docs Site Restructure

Current navigation is confusing because it mirrors the notebook categories, not the reader's
mental model.

**This navigation maps 1:1 to the thirteen root notebooks** listed under
[Big picture: thirteen root notebooks](#big-picture-thirteen-root-notebooks-reader-spine). Tutorials,
`models/`, `fitting/`, etc. should not appear as top-level peers once the spine is stable — only
as “source material” links for maintainers.

**Proposed top-level navigation**:

```
Getting Started
  ├── Install in 30 seconds
  └── SED Fitting in Five Code Cells         → 00_quickstart

The Physics (gallery track)
  ├── Anatomy of a Galaxy SED                → 01_sed_anatomy
  ├── Star Formation Histories               → 02_sfh_gallery
  ├── Dust Attenuation and Infrared Emission → 03_dust_gallery
  ├── Nebular Emission                       → 04_nebular_gallery
  ├── Active Galactic Nuclei                 → 05_agn_gallery
  └── Multi-Wavelength Coverage              → 06_multiwavelength_gallery

Fitting Your Data
  ├── Photometric SED Fitting                → 07_fitting_photometry
  ├── Spectroscopic Fitting                  → 08_fitting_spectra
  ├── Parameter Degeneracies                 → 09_degeneracies
  └── End-to-End Fitting with Real Data      → 10_real_data

Advanced
  ├── Hierarchical Population Inference      → 11_population
  └── Custom Components                      → 12_extending_tengri

API Reference (auto-generated)
```

Remove: `observation/` docs section (merge into fitting notebooks as needed),
`performance/` section (move benchmark tables to API reference),
`theory/` section (absorbed into gallery notebooks).

---

## Tone and Language

Every notebook should open with a one-sentence statement of what the reader will understand after
reading it, written for an astronomer who has never used tengri:

**Good opening**: "In this notebook you'll see how dust attenuation progressively reddens the
stellar SED, and how the FIR emission mirrors that absorbed energy — the most direct connection
between UV and submillimeter observations."

**Bad opening** (current style): "In this notebook we demonstrate the dust attenuation and
emission models available in tengri, including the Charlot & Fall two-component model,
Calzetti, Kriek-Conroy, SMC, LMC, and MW extinction curves, as well as the Draine & Li 2007,
Dale et al. 2014..."

Rules:
- Name the physics law or effect first, the software parameter second
- Avoid internal API names in narrative prose ("pass `dust_tau_bc=2.0`" in a callout box, not
  inline in the explanation)
- Each parameter sweep figure caption completes the sentence: "Increasing τ_BC from 0 to 4 ..."
- Use "you" as the subject: "When you increase the ionization parameter..."
- Never write "as mentioned above" — reorganize so each point lands once

---

## Summary: The Pre-Work That Would Have Changed Everything

In priority order, the investments that would have paid the biggest dividends:

1. **`sweep_parameter()` utility** — would have made every gallery notebook 6x shorter
2. **The visual language spec** — would have eliminated the per-notebook style chaos
3. **The narrative arc document** — the spine table in [Big picture](#big-picture-thirteen-root-notebooks-reader-spine); would have prevented the 5-category fragmentation
4. **`sfh_sed_comparison()` two-panel function** — the SFH story requires it and it doesn't exist
5. **Model introspection API** (`model.param_ranges()`) — would have enabled auto-generated galleries
   instead of hard-coded sweep ranges that drift from the actual priors

The notebooks that exist are not wrong — the physics is correct and the coverage is comprehensive.
What they lack is a **shared vocabulary of images**: a common set of plot types that carry the
same meaning in every context, built from the same infrastructure, with the same visual language.
That shared vocabulary is what makes documentation feel like a coherent textbook rather than a
collection of analysis scripts.

---

## Notebook Section Hierarchy and Storylines

*Prerequisite:* read [Big picture: thirteen root notebooks](#big-picture-thirteen-root-notebooks-reader-spine)
for the one-table summary of what each file is for; this section is the **implementation brief**
(section titles, sources, issues, fix requirements).

*For each of the 13 target notebooks, this section gives: the **narrative driving question**,
actual **sections as implemented** (from the 2026-04-05 audit), **canonical source files** to
pull code from during revision, and **known issues** to fix before publication. The narrative
arc and section structure are correct; what needed updating were the source references (the old
`ref/XX` / `fit/XX` aliases pointed to non-existent files) and the missing ground-truth state.*

*Canonical source file paths (all exist; use these):*

| Alias (retired) | Real path |
|-----------------|-----------|
| `ref/18` (SFH) | `notebooks/models/notebook_code/01_sfh_models.py` |
| `ref/15` (attenuation) | `notebooks/models/notebook_code/02_dust_attenuation.py` |
| `ref/16` (dust emission) | `notebooks/models/notebook_code/03_dust_emission.py` |
| `notebooks/models/notebook_code/04_agn.py` (AGN) | `notebooks/models/notebook_code/04_agn.py` |
| — (IGM) | `notebooks/models/notebook_code/05_igm.py` |
| `notebooks/models/notebook_code/06_nebular.py` (nebular) | `notebooks/models/notebook_code/06_nebular.py` |
| — (multiwavelength) | `notebooks/models/notebook_code/07_multiwavelength.py` |
| — (radio) | `notebooks/models/notebook_code/08_radio.py` |
| `notebooks/quickstart/notebook_code/01_quickstart.py` | `notebooks/quickstart/notebook_code/01_quickstart.py` |
| `qs/03` | `notebooks/quickstart/notebook_code/03_bursty_sfh_recovery.py` |
| `notebooks/specialist/notebook_code/01_real_data.py` | `notebooks/specialist/notebook_code/01_real_data.py` |
| `notebooks/specialist/notebook_code/02_derived_quantities.py` | `notebooks/specialist/notebook_code/02_derived_quantities.py` |
| `notebooks/specialist/notebook_code/03_model_checking.py` | `notebooks/specialist/notebook_code/03_model_checking.py` |
| `notebooks/specialist/notebook_code/04_extending_tengri.py` | `notebooks/specialist/notebook_code/04_extending_tengri.py` |
| `notebooks/specialist/notebook_code/05_emission_line_marginalization.py` (eline) | `notebooks/specialist/notebook_code/05_emission_line_marginalization.py` |
| `notebooks/specialist/notebook_code/07_advanced_spectroscopy.py` (spectroscopy) | `notebooks/specialist/notebook_code/07_advanced_spectroscopy.py` |
| `th/02` | `notebooks/theory/notebook_code/02_forward_model.py` |

*`notebooks/specialist/notebook_code/07_advanced_spectroscopy.py`–`notebooks/specialist/notebook_code/06_simulation_sfh.py` (`notebooks/fitting/`) do **not** exist. Content attributed to those aliases
must come from the specialist notebooks above. `ref/15`–`notebooks/models/notebook_code/06_nebular.py` (`notebooks/reference/`) also
do not exist — they were a prior agent's aspirational placeholders.*

---

### `00_quickstart.ipynb` — "SED Fitting in Five Code Cells"

**Driving question**: *I just installed tengri. What does the simplest possible end-to-end
fit look like?*

**Narrative arc**: The reader arrives knowing nothing. They leave having run a full fit and
read a posterior, having typed fewer than 20 lines of code. Every sentence earns its place by
moving the reader one step closer to a fit. No theory. No options. No decisions.

**Actual file**: `notebooks/00_quickstart.py` — 144 lines

**Sections as implemented** (matches narrative plan):
1. "A galaxy in five lines" — `Parameters()` spec, SDSS filters + spectroscopy, `tsnorm` SFH, 8 free params
2. "Generating a mock observation" — `model.mock(key, snr=20.0)`, errorbars over true SED
3. "One line to fit" — `result = model.fit(mock.flux_obs, mock.noise)`, VI auto-selected
4. "What did you learn?" — `result.plot_sed()` + `result.plot_sfh()` side by side
5. "Reading the posterior numbers" — `result.summary_table()`

**Canonical source** for revision: `notebooks/quickstart/notebook_code/01_quickstart.py`

**Known issues**:
- `notebooks/00_quickstart.py` uses old `Model.from_config()` API with only 3 free params — update
  to `Parameters(sfh_tsnorm_log_peak_sfr=Uniform(...), ...)` + `Model(spec_param, ssp_data, observation=obs)`
  following the canonical source exactly
- Missing jupytext `formats: notebook_code//py:percent,ipynb` header at top of file — add it
- Bootstrap uses `from tengri import Model` directly; no `_plot_style.setup_style()` call —
  add `from _plot_style import COLORS, setup_style; setup_style()` for visual consistency
- All `plt.savefig(...)` / `fig.savefig(...)` calls must be **commented out** (not deleted)

#### Section 1 — "A galaxy in five lines"
- **Content**: Build a `Parameters()` spec with SDSS filters + 200-pixel spectroscopy window
  (`WAVE_OBS = jnp.linspace(3800.0, 9200.0, 200)`), `tsnorm` SFH, 8 free parameters.
  Construct model via `Model(spec_param, ssp_data, observation=obs)`.
- **Source**: `notebooks/quickstart/notebook_code/01_quickstart.py` Part A setup (lines ~1–120).
- **Story thread**: "→ Now you have a model. Give it noise and it will give you a galaxy."

#### Section 2 — "Generating a mock observation"
- **Content**: `model.mock(key, snr=20.0)` → plot photometry errorbars over true SED.
- **Source**: `notebooks/quickstart/notebook_code/01_quickstart.py` (mock + plot pattern).
- **Figure**: 1 panel, `figsize=(7, 4)`. True SED = gray line, photometry = black errorbars.
- **Story thread**: "→ This is your data. Now ask: what galaxy could have produced it?"

#### Section 3 — "One line to fit"
- **Content**: `result = model.fit(mock.flux_obs, mock.noise)`. Markdown cell: `vi` auto-selected,
  JAX compiles on first call (~30 s, then fast).
- **Source**: `notebooks/quickstart/notebook_code/01_quickstart.py` "Fitting with vi" section.
- **Story thread**: "→ The fit is done. Now read what tengri learned about your galaxy."

#### Section 4 — "What did you learn?"
- **Content**: `plt.subplots(1, 2, figsize=(12, 4))` — left: SED + posterior band using
  `model.predict_photometry(result.params)`; right: SFH recovery with `plot_sfh(result, model)`.
  Use `safe_corner(result)` from `_plot_style` for parameter corner below.
- **Source**: `notebooks/quickstart/notebook_code/01_quickstart.py` "Parameter recovery" and
  fig04/fig05 sections. `plot_sfh` and `safe_corner` are imported from `_plot_style`.
- **Story thread**: "→ The numbers behind those plots are in the summary table below."

#### Section 5 — "Reading the posterior numbers"
- **Content**: `result.summary_table()`. Columns: median, 16th/84th percentile, ESS.
- **Source**: `notebooks/quickstart/notebook_code/01_quickstart.py` "Parameter recovery" section.
- **Story thread**: "→ To understand *why* these parameters matter physically, start with
  [01_sed_anatomy]."

---

### `01_sed_anatomy.ipynb` — "Anatomy of a Galaxy SED"

**Driving question**: *What is every feature in a galaxy SED, and what physics causes it?*

**Narrative arc**: A galaxy SED is a physical record. This notebook teaches the reader to read
it like a map, feature by feature. It builds the visual vocabulary the rest of the series uses.

**Actual file**: `notebooks/01_sed_anatomy.py` — 305 lines

**Sections as implemented**:
1. "The full story: X-ray to radio in one plot" — full component SED with annotated feature labels
2. "Building up the SED: one component at a time" — 2×2 cumulative component grid
3. "How redshift shifts the SED into your filter set" — SED at z = 0.1, 0.5, 1.0, 2.0, 4.0

**Canonical sources** for revision:
- Component decomp code → `notebooks/theory/notebook_code/02_forward_model.py` "Complete Pipeline"
- Redshift sweep + filter overlay → `notebooks/models/notebook_code/05_igm.py` (IGM section)

**Known issues**:
- 3 `plt.savefig(...)` calls writing to the current working directory (no FIGDIR defined) —
  comment out all three
- Uses private function `_predict_sed_component()` from `tengri.core.sed_pipeline` — replace
  with public `model.predict()` decomposition or document as intentional internal use

#### Section 1 — "The full story: X-ray to radio in one plot"
- **Content**: Full multiwavelength SED of a fiducial galaxy (all components enabled) with
  annotated feature labels. This is the opener — the most striking figure in the notebook.
- **Source**: `notebooks/theory/notebook_code/02_forward_model.py` "Complete Pipeline" section
  (component decomp code); `notebooks/models/notebook_code/06_nebular.py` for obs-model setup.
- **Figure**: `plt.subplots(1, 1, figsize=(14, 5))`. Log x-axis 1 Å–10 m. Annotate with
  arrows for Lyman break, 4000 Å break, NIR bump at 1.6 μm, PAH features, FIR peak, radio.
- **Story thread**: "→ That SED is built from four additive components — let's disassemble it
  one layer at a time."

#### Section 2 — "Building up the SED: one component at a time"
- **Content**: 2×2 grid. Panel (0,0): stars only. Panel (0,1): + nebular. Panel (1,0): + dust
  attenuation. Panel (1,1): + dust emission (full model). Gray = previous step, color = new.
- **Source**: `notebooks/theory/notebook_code/02_forward_model.py` "SSP Building Blocks"
  through "Complete Pipeline" sections.
- **Figure**: `plt.subplots(2, 2, figsize=(12, 8))`.
- **Story thread**: "→ The final step — dust emission — completes the energy balance. Now see
  how the same galaxy looks to a telescope at different cosmic distances."

#### Section 3 — "How redshift shifts the SED into your filter set"
- **Content**: Same galaxy SED at z = 0.1, 0.5, 1.0, 2.0, 4.0. Observed-frame wavelengths.
  Overplot SDSS/JWST filter transmission curves as gray shaded bands.
- **Source**: `notebooks/models/notebook_code/05_igm.py` (redshift sweep pattern);
  `notebooks/theory/notebook_code/02_forward_model.py` "SED → Photometry" for filter overlay.
- **Figure**: `plt.subplots(1, 1, figsize=(12, 5))`. Colormap `plasma` indexed by redshift.
  x-axis: observed-frame 1000–50000 Å.
- **Story thread**: "→ JAX gives us exact gradients — here is what the forward model knows
  about each parameter at each wavelength."

#### Section 4 — "What the data sees: Jacobian gradients"
- **Content**: n_params panels stacked vertically (one per free parameter). Each panel shows
  ∂spectrum/∂θ_i as a function of wavelength with `SPECTRAL_FEATURES` vertical guides.
  Colors from `COLORS["seq"]`. Setup: `spectrum_from_array` closure wraps the model forward
  pass; Jacobian computed exactly with `jax.jacobian(spectrum_from_array)(param_array)`.
  Prints `Jacobian shape: (n_wave, n_params)` to stdout.
- **Source**: `notebooks/theory/notebook_code/02_forward_model.py` §2 "Jacobian Computation"
  (lines ~403–457). Uses `SPECTRAL_FEATURES` from `_plot_style`.
- **Figure**: `plt.subplots(n_params, 1, figsize=(10, 2.2 * n_params), sharex=True)`.
  Figure saved as `# plt.savefig(..., "07_gradient_seds.png")`.
- **Story thread**: "→ Showing all parameters at once as a heatmap reveals the degeneracy
  structure."

#### Section 5 — "Degeneracy structure: sensitivity heatmap and scalogram"
- **Content**: Two figures. Figure 1: sensitivity heatmap — `imshow` of
  `|∂m/∂θ|` normalized per-parameter, x = wavelength, y = parameter name. Colorbar labeled
  "Normalized sensitivity". `SPECTRAL_FEATURES` vertical guides. Figure 2: multiscale
  scalogram for selected parameters (e.g. `psd_sigma`, `psd_tau_yr`, `met_logzsol`).
  Gaussian kernel convolution at `scales = np.linspace(10, 500, 30)` Å; heatmap shows
  integrated sensitivity at each scale.
- **Source**: `notebooks/theory/notebook_code/02_forward_model.py` §3 "Sensitivity Heatmap"
  and §4 "Wavelet-like Scalogram" (lines ~461–555).
- **Figures**: Heatmap: `plt.subplots(figsize=(10, 4))`. Scalogram:
  `plt.subplots(len(params_to_show), 1, figsize=(10, 3 * len(params_to_show)), sharex=True)`.
  Figures saved as `# plt.savefig(..., "07_sensitivity_heatmap.png")` and
  `# plt.savefig(..., "07_scalogram.png")`.
- **Story thread**: "→ The SFH is the most important ingredient controlling all of these
  features simultaneously. The next notebook shows you the full space of SFH shapes."

---

### `02_sfh_gallery.ipynb` — "Star Formation Histories: Parametric and Stochastic Models"

**Driving question**: *How does the way a galaxy assembled its stellar mass change what we
observe, and how do we model that?*

**Narrative arc**: The SFH is the central unknown in galaxy SED fitting. This notebook builds
intuition from smooth to bursty, from parametric to stochastic. By the end the reader
understands why a smooth DPL model fails to recover a bursty galaxy.

**Actual file**: `notebooks/02_sfh_gallery.py` — 263 lines

**Sections as implemented**:
1. "Part 1: Parametric SFH Models" — delayed-τ, log-normal, DPL, tsnorm; per-family parameter sweeps
2. "Part 2: Stochastic SFH (GP Field)" — σ/τ grid, 5 realizations per cell, SED spread
3. "Part 3: SFH Recovery from Mock Data" — DPL vs stochastic model fit comparison on bursty truth
4. "Summary" — markdown recap + pointer to next notebook

**Canonical source** for revision: `notebooks/models/notebook_code/01_sfh_models.py`
(parametric sweeps in Section 1; stochastic GP grid in Section 3)

**Known issues**:
- Imports `setup_style` from `tengri.plotting` — should be `from notebooks._plot_style import setup_style`
- No savefig calls — clean
- No deprecated API usage

#### Section 1 — "The four shapes a galaxy's life can take"
- **Content**: 2×4 grid. Top row: SFH(t) shapes for delayed-τ, log-normal, DPL, tsnorm.
  Bottom row: corresponding SEDs for each family. Plus per-family parameter sweep panels.
- **Source**: `notebooks/models/notebook_code/01_sfh_models.py` Section 1 (parametric SFH).
  Copy the `for` loop over SFH families; strip the per-model deep-dives to one sweep each.
- **Figure**: `plt.subplots(2, 4, figsize=(16, 6))`.
- **Story thread**: "→ Each family has one or two parameters that shape its character. Here is
  what those knobs do."

#### Section 2 — "Bursts and quiet periods: the stochastic SFH"
- **Content**: σ/τ grid — 3 σ values × 3 τ values. Each cell: SFH panel (5 thin realizations
  + mean) + SED panel. Key variables: `psd_sigma`, `psd_tau_yr` (internal years).
- **Source**: `notebooks/models/notebook_code/01_sfh_models.py` Section 3 (stochastic GP SFH).
  Copy the σ/τ grid loop directly.
- **Figure**: `plt.subplots(3, 6, figsize=(24, 10))`.
- **Story thread**: "→ The stochastic model can express all of this — but can it *recover* a
  bursty SFH from real data? That is the most important question."

#### Section 3 — "Why the wrong model gives the wrong answer"
- **Content**: 3 panels. Left: true bursty SFH. Center: DPL model recovery (smooth, biased).
  Right: stochastic field recovery (correct shape).
- **Source**: `notebooks/quickstart/notebook_code/03_bursty_sfh_recovery.py` "The Wrong Model
  Trap" section. This is the single most important scientific figure in the gallery track.
- **Figure**: `plt.subplots(1, 3, figsize=(15, 4))`.
- **Story thread**: "→ With SFH in hand, we need dust — the other major unknown. The next
  notebook shows what dust does to a galaxy SED."

---

### `03_dust_gallery.ipynb` — "Dust Attenuation and Infrared Emission"

**Driving question**: *UV photons disappear and infrared photons appear — where does that
energy go, and how do we measure it?*

**Narrative arc**: Dust attenuation and emission are two sides of one physical coin: energy
absorbed in the UV must be re-radiated in the IR. The notebook first shows attenuation
(the UV disappearing), then shows emission (the IR appearing), then closes by confirming
energy balance.

**Actual file**: `notebooks/03_dust_gallery.py` — 443 lines

**Sections as implemented**:
1. "Part 1: Dust Attenuation" — two-component model sweeps (τ_BC, τ_diff, dust_slope, UV bump, R_V)
2. "Part 2: Infrared Dust Emission" — MBB T/β sweeps; DL07 U_min and q_PAH sweeps
3. "Energy Balance" — absorbed UV vs re-emitted FIR verification

**Canonical sources** for revision:
- Two-component attenuation sweeps → `notebooks/models/notebook_code/02_dust_attenuation.py`
  (Section 8: Two-Component Dust Model — τ_BC, τ_diff, young vs old star transmission)
- Attenuation curve shapes → `notebooks/models/notebook_code/02_dust_attenuation.py`
  (Sections 6b–6d: Kriek-Conroy δ, UV bump Eb, Cardelli R_V)
- MBB and template sweeps → `notebooks/models/notebook_code/03_dust_emission.py`
  (Section 2: analytic MBB; Section 3: DL07 templates)
- Energy balance → `notebooks/models/notebook_code/03_dust_emission.py` Section 4

**Important note on implementation approach**: This notebook uses `Model.from_config()` with a
single fiducial galaxy and modifies parameters via `model.update_params()` for each panel. This
is the high-level approach. The canonical source (`02_dust_attenuation.py`, `03_dust_emission.py`)
uses low-level physics functions directly (`two_component_dust()`, `modified_blackbody()`, etc.).
Both approaches are valid; the high-level approach is more user-friendly for a gallery notebook.
Consider retaining the high-level style but pulling the swept parameter ranges and sweep logic
from the canonical low-level notebooks.

**Known issues**:
- No savefig calls — clean
- `Model.from_config()` is the correct public API — no changes needed here

#### Section 1 — "The two-component picture: birth clouds and the diffuse ISM"
- **Content**: 3 panels. τ_BC sweep (birth cloud attenuation of nebular lines). τ_diff sweep
  (old-star reddening). Combined: how the two act on different stellar ages.
- **Source**: `notebooks/models/notebook_code/02_dust_attenuation.py` Section 8 (two-component
  model; subsections 8b "Young vs Old Star Transmission", 8c "Varying τ_bc and τ_diff
  independently"). Best two-component sweep code in the repo.
- **Figure**: `plt.subplots(1, 3, figsize=(15, 4))`. x-axis: 1000–10000 Å.
  Colormaps: `YlOrRd` (τ_BC), `YlOrBr` (τ_diff).
- **Story thread**: "→ The amount of reddening also depends on the *shape* of the attenuation
  curve — steeper curves darken the UV more than the optical."

#### Section 2 — "The shape of attenuation: from Calzetti to MW"
- **Content**: 3 panels. `dust_slope` sweep (Kriek-Conroy δ). UV bump strength sweep (Eb,
  2175 Å feature). R_V sweep (Cardelli MW family).
- **Source**: `notebooks/models/notebook_code/02_dust_attenuation.py` Sections 6b–6d.
- **Figure**: `plt.subplots(1, 3, figsize=(15, 4))`. Panel 2: zoom to 1500–3500 Å.
  Colormaps: `Blues`, `Purples`, `Greens`.
- **Story thread**: "→ All the absorbed energy has to go somewhere. At 20–100 K, that means
  the infrared."

#### Section 3 — "Where does the energy go: FIR emission"
- **Content**: 2 panels. Dust temperature sweep (20–80 K). Emissivity index β sweep (1.0–2.2).
  T controls peak wavelength; β controls Rayleigh-Jeans slope.
- **Source**: `notebooks/models/notebook_code/03_dust_emission.py` Section 2 (analytic MBB
  models — `T_dust` and `beta` sweeps).
- **Figure**: `plt.subplots(1, 2, figsize=(12, 4))`. x-axis: 10 μm–1 mm (FIR).
  Colormaps: `hot_r` (temperature), `YlOrBr` (β).
- **Story thread**: "→ For precise FIR modeling, the Draine & Li template library captures
  the PAH features that a simple modified blackbody misses."

#### Section 4 — "PAH features and the detailed dust SED"
- **Content**: 3 panels. U_min sweep (radiation field intensity). q_PAH sweep (PAH mass
  fraction), zoomed to 3–20 μm to show features. UV-to-radio SED for low vs high q_PAH.
- **Source**: `notebooks/models/notebook_code/03_dust_emission.py` Section 3 (DL07 templates;
  U_min and q_PAH sweeps). Add `ax.set_xlim(3e4, 2e5)` (in Å) for the q_PAH zoom panel.
- **Figure**: `plt.subplots(1, 3, figsize=(15, 4))`. Colormaps: `Oranges`, `RdPu`.
- **Story thread**: "→ Energy balance means the area absorbed in the UV exactly equals the
  area emitted in the FIR — a consistency check worth verifying."

#### Section 5 — "Energy balance: absorbed = re-emitted"
- **Content**: 1 panel. Attenuation SED (gray) + dust emission SED (orange). Shade the
  attenuation notch and emission peak in matching colors. Text annotations confirming
  integrated energies match.
- **Source**: `notebooks/models/notebook_code/03_dust_emission.py` Section 4 (energy balance).
- **Figure**: `plt.subplots(1, 1, figsize=(10, 4))`.
- **Story thread**: "→ Once the stellar SED is attenuated and re-emitted by dust, young
  stars can also ionize the surrounding gas — producing the emission lines we see next."

---

### `04_nebular_gallery.ipynb` — "Nebular Emission, IGM, and Observation Models"

**Driving question**: *When hot young stars ionize the surrounding gas, what does the spectrum
look like — and what physical diagnostics can we extract from it?*

**Narrative arc**: Opens with a backend decision table (BakedIn / CloudyGrid / Cue / Shocks),
then systematically covers the logU → line ratio physics, Cue's unique N/O and C/O parameters,
shock emission from MAPPINGS V, DIG mixing, and the Q_H link between stellar populations and
nebular luminosity. All figures follow the canonical source notebook exactly.

**Source**: `notebooks/models/notebook_code/06_nebular.py`

**Key imports**:
```python
from tengri import Fixed, Model, Parameters, load_ssp_data
from tengri.models.nebular.shock import shock_line_ratios, _SHOCK_V, _R_OIII, _R_NII, _R_SII, _R_OII, _R_OI, _R_HA
from tengri.models.nebular import shock_emission_sed, mix_dig_emission
from tengri.models.nebular.cloudy_grid import compute_qh
from _plot_style import COLORS, setup_style
```

#### Opening markdown — Backend Decision Table
- **Content**: Markdown table with columns: Backend | Free params | Use when | Limitation.
  Three rows: BakedIn (0 free params, photometry), CloudyGrid (3: logU, Z, n_H, spectroscopy),
  Cue (12, abundance ratios or non-stellar ionizing sources). Rule: "Use BakedIn for photometry.
  CloudyGrid for standard spectroscopy. Cue when you need [N/O], [C/O], or AGN/shock shapes."
- **Source**: `notebooks/models/notebook_code/06_nebular.py` opening markdown (lines 98–107).

#### Section 1.1 — "CLOUDY Grid: line ratios vs log(U)"
- **Content**: 6-panel 2×3 grid. [OIII]5007/Hβ (panel 1), [NII]6583/Hα (panel 2),
  [OII]3727/Hβ (panel 3), [SII]/Hα (panel 4), Balmer decrement (panel 5), BPT diagram with
  viridis colorbar for logU (panel 6). Uses Kewley+2001/Dopita+2013 scaling relations at solar
  metallicity. `logU_grid = np.linspace(-4.0, -1.5, 50)`.
- **Source**: `notebooks/models/notebook_code/06_nebular.py` §1.1 (lines 110–199).
- **Figure**: `plt.subplots(2, 3, figsize=(11, 5.5))`. Suptitle: "CLOUDY-like Line Ratios vs
  Ionization Parameter". Figure saved as `# plt.savefig(..., "19_cloudy_line_ratios.png")`.
- **Story thread**: "→ The three unique capabilities of Cue over the CLOUDY grid are [N/O],
  [C/O], and ionizing spectrum shape."

#### Section 1.2 — "Cue Neural Emulator: N/O, C/O, and Ionizing Spectrum"
- **Content**: 3-panel figure, conditional on `CUE_WEIGHTS_PATH.exists()`. Panel 1: [NII]6583/Hα
  vs `gas_logno` sweep (logno in [-1.5, 0.5], 18 points). Panel 2: [CIII]1909/Hβ vs `gas_logco`
  sweep (logco in [-1.0, 0.5]). Panel 3: BPT scatter — 3 ionizing spectrum configs (Stellar
  O-star, AGN-like flat, Soft cool star) as `o`, `s`, `^` markers. If weights absent, prints
  capability summary to stdout.
- **Source**: `notebooks/models/notebook_code/06_nebular.py` §1.2 (lines 201–321).
  Uses `load_cue_weights`, `predict_all_lines`, `prepare_nn_params_from_dict`.
- **Figure**: `plt.subplots(1, 3, figsize=(13, 4))`. Suptitle: "Cue Neural Emulator: Unique
  Capabilities vs CLOUDY Grid". Figure saved as `# plt.savefig(..., "19_cue_parameter_sweeps.png")`.
- **Story thread**: "→ Shocks from supernovae and AGN outflows produce a different ionization
  signature — here is the MAPPINGS V shock model."

#### Section 1.3 — "Shock Emission: line ratios vs velocity"
- **Content**: 2-panel figure. Panel A: tabulated line ratios vs velocity (from `_SHOCK_V`,
  `_R_OIII`, `_R_NII`, `_R_SII`, `_R_OII`, `_R_OI`) on log-y scale. Panel B: BPT showing shock
  track (Allen+2008) vs HII track (vary logU), with Kauffmann+2003 demarcation dashed.
- **Source**: `notebooks/models/notebook_code/06_nebular.py` §1.3 (lines 323–396). Uses the
  tabulated arrays imported directly from `tengri.models.nebular.shock`.
- **Figure**: `plt.subplots(1, 2, figsize=(9, 3.5))`. Figure saved as
  `# plt.savefig(..., "19_shock_emission.png")`.
- **Story thread**: "→ The DIG — diffuse ionized gas — also alters line ratios by lowering
  the effective ionization parameter."

#### Section 1.4 — "DIG Mixing: effect on [NII]/Hα"
- **Content**: 2-panel figure. Panel A: [NII]6583/Hα vs f_DIG (0–0.7) for three metallicities
  (low Z, solar Z, high Z). Panel B: BPT showing 4 f_DIG values (0.0, 0.2, 0.4, 0.6) as
  separate markers, with Kauffmann+2003 line.
- **Source**: `notebooks/models/notebook_code/06_nebular.py` §1.4 (lines 398–458). Uses
  hand-computed linear mixing (nii_ha_hii=0.3, nii_ha_dig=0.8).
- **Figure**: `plt.subplots(1, 2, figsize=(9, 3.5))`. Figure saved as
  `# plt.savefig(..., "19_dig_mixing.png")`.
- **Story thread**: "→ The physical link between stars and nebular emission is Q_H — the rate
  of hydrogen-ionizing photons."

#### Section 2 — "Q_H: The Link Between Stars and Nebular Emission"
- **Content**: Code cell only (no figure). Load SSP with baked-in nebular
  (`ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5`). Call `compute_qh(wave, ssp_flux[3,5])`
  for young SSP (~10 Myr) and `compute_qh(wave, ssp_flux[3,-5])` for old SSP (~10 Gyr).
  Print Q_H values and ratio. Guarded by `if SSP_WNE_PATH.exists()`.
- **Source**: `notebooks/models/notebook_code/06_nebular.py` §2.1 (lines 460–525).
  `compute_qh` imported from `tengri.models.nebular.cloudy_grid`.
- **Story thread**: "→ Shocks across a continuous velocity grid have distinct diagnostic ratios
  — now shown at higher resolution using `shock_line_ratios()`."

#### Section 3 — "Shock Emission Lines (MAPPINGS V)"
- **Content**: 4-panel 2×2 grid with shared x-axis. Each panel shows one diagnostic ratio vs
  shock velocity (100–1000 km/s, 200 points via `np.linspace`): [NII]6583/Hα, [SII]6716+31/Hα,
  [OI]6300/Hα, [OIII]5007/Hβ. Each panel has an HII region reference dashed line
  (Kewley+2006 typical values). Uses `shock_line_ratios(v)` from
  `tengri.models.nebular.shock`.
- **Source**: `notebooks/models/notebook_code/06_nebular.py` §3 (lines 527–597).
- **Figure**: `plt.subplots(2, 2, figsize=(10, 7), sharex=True)`. Suptitle: "Shock Diagnostic
  Line Ratios (Allen+2008, Solar, n=1 cm⁻³)". Figure saved as
  `# plt.savefig(..., "05_shock_line_ratios.png")`.
- **Story thread**: "→ In a real galaxy the spectrum is a mixture of HII region and shock
  emission — here is how that composite looks."

#### Section 4 — "Shock-HII Mixing"
- **Content**: 1-panel figure showing composite SED = (1 − f_shock) × L_HII + f_shock × L_shock
  for f_shock ∈ [0.0, 0.3, 0.7, 1.0]. Colormap `RdYlBu_r`. x-axis: 3500–7500 Å. Key optical
  diagnostic lines annotated with vertical dotted lines (Hβ, [OIII], [OI], [NII], Hα, [SII]).
  v_shock = 300 km/s. Requires SSP data; guarded by `if wave is not None`.
  Uses `shock_emission_sed(wave_grid, shock_v, l_halpha_ref, line_sigma_aa=3.0)`.
- **Source**: `notebooks/models/notebook_code/06_nebular.py` §4 (lines 599–673).
- **Figure**: `plt.subplots(figsize=(10, 5))`. Figure saved as
  `# plt.savefig(..., "05_shock_hii_mixing.png")`.
- **Story thread**: "→ Diffuse ionized gas (DIG) is pervasive in local galaxies — its
  mixing fraction is now demonstrated via the `mix_dig_emission()` API."

#### Section 5 — "Diffuse Ionized Gas (DIG) Mixing"
- **Content**: 3-panel 1×3 figure showing [NII]6583/Hα, [SII]6716+31/Hα, [OI]6300/Hα
  vs f_DIG (0.0–0.6, 7 points) with `o-` markers. Each panel has a pure-HII dashed reference
  line. DIG ratios from Tacchella+2022 Fig 3. Followed by API code block (markdown):
  `mix_dig_emission(backend, ..., neb_dig_frac=0.4, neb_dig_delta_logU=-1.0)`.
- **Source**: `notebooks/models/notebook_code/06_nebular.py` §5 (lines 675–796).
  Uses `mix_dig_emission` from `tengri.models.nebular`.
- **Figure**: `plt.subplots(1, 3, figsize=(12, 4), sharey=False)`. Suptitle: "DIG Mixing:
  Enhanced Low-Ionization Lines (ΔlogU = −1 dex, Tacchella+2022)". Figure saved as
  `# plt.savefig(..., "05_dig_mixing.png")`.
- **Story thread**: "→ With the full nebular toolkit in hand — CLOUDY, Cue, shocks, DIG —
  the next notebook adds the most dramatic SED contributor: an AGN."

---

### `05_agn_gallery.ipynb` — "Active Galactic Nuclei: Disc, Torus, and Unified Model"

**Driving question**: *How does an AGN change a galaxy's SED, and how can you tell whether
the nucleus or the stars are dominating?*

**Narrative arc**: Opens with a comparison that makes the AGN's impact viscerally clear
(the blue bump, the power law, the torus). Then walks through inclination (which determines
AGN type), black hole physics (mass and spin), and ends with the multiwavelength extension
into X-ray.

#### Section 0 — "The first glimpse: a galaxy with and without an AGN"
- **Content**: 1 panel. Blue = star-forming galaxy (no AGN). Orange = same galaxy + AGN
  (`agn_log_lbol=45.5`, `cos_inc=0.7`). x-axis: 1000–1e6 Å (UV through MIR).
  Annotation arrows: "big blue bump", "power-law continuum", "MIR torus hump".
- **Source**: `notebooks/models/notebook_code/04_agn.py` §1 (Full AGN SED overview — all 4 components:
  disc, torus, BLR, NLR). Setup code for the fiducial AGN at the opening of that section.
- **Figure**: `plt.subplots(1, 1, figsize=(12, 5))`.
- **Story thread**: "→ Whether you see the blue bump or just the torus depends on which
  direction you're looking at the AGN — that's the inclination."

#### Section 1 — "Inclination: the view from above vs through the dust"
- **Content**: 2 panels. Left: SED sweeping `cos_inc` from 0.95 (face-on, type 1) to 0.1
  (edge-on, type 2). Show how the UV disc disappears and MIR torus takes over. Right: schematic
  cartoon (matplotlib patches) of disc + torus cross-section with labeled sightlines.
- **Source**: `notebooks/models/notebook_code/04_agn.py` §7a (Type 1 vs Type 2 Comparison — cos_inc
  sweep and geometry cartoon code).
- **Figure**: `plt.subplots(1, 2, figsize=(14, 5))`. Colormap: `RdYlBu`.
- **Story thread**: "→ The disc temperature and luminosity are set by the black hole mass and
  accretion rate — here is what those knobs do."

#### Section 2 — "The black hole: mass and accretion rate"
- **Content**: 2 panels. Left: `agn_log_mbh` sweep [6.5, 7.0, 7.5, 8.0, 8.5] — hotter disc
  = bluer big blue bump. Right: `agn_log_ledd` sweep [-2, -1, -0.5, 0] — more luminous disc.
- **Source**: `notebooks/models/notebook_code/04_agn.py` §2b (Multicolor Disc — M_BH and Eddington
  ratio sweeps). Direct copy with cosmetic changes.
- **Figure**: `plt.subplots(1, 2, figsize=(12, 4))`. Colormaps: `Purples`, `Oranges`.
- **Story thread**: "→ The maximum disc temperature also depends on how fast the black hole
  spins — a maximally spinning BH converts 42% of rest mass to radiation."

#### Section 3 — "Black hole spin: radiative efficiency and disc temperature"
- **Content**: 1 panel. Sweep `agn_spin` in [0.0, 0.5, 0.9, 0.998]. Higher spin = smaller
  ISCO = hotter, bluer, more luminous disc.
- **Source**: `notebooks/models/notebook_code/04_agn.py` §2b spin sweep section (spin → ISCO → η
  → luminosity chain).
- **Figure**: `plt.subplots(1, 1, figsize=(8, 4))`. Colormap: `plasma`.
- **Story thread**: "→ A fraction of the disc emission is absorbed by the surrounding torus
  and re-emitted at MIR wavelengths — the covering factor controls how much."

#### Section 4 — "The torus: covering factor and optical depth"
- **Content**: 2 panels. Left: `agn_torus_frac` sweep [0.1, 0.3, 0.5, 0.7, 0.9], x-axis
  1–100 μm. Right: `agn_tau_skirtor` sweep [3, 7, 11, 15], same range.
- **Source**: `notebooks/models/notebook_code/04_agn.py` §4c (SKIRTOR Clumpy Torus parameter sweeps).
  Use `skirtor_analytic` function, not the toy models.
- **Figure**: `plt.subplots(1, 2, figsize=(12, 4))`. Colormap: `OrRd`.
- **Story thread**: "→ AGN also produce hard X-ray emission — and the ratio of X-ray to UV
  luminosity is itself a diagnostic of the accretion state."

#### Section 5 — "Into the X-ray: α_OX and the power-law"
- **Content**: 2 panels. Left: `xray_gamma_agn` sweep [1.5, 1.8, 2.1, 2.5], x-axis 0.1–100
  keV. Right: `xray_alpha_ox` sweep [-1.8, -1.4, -1.0], x-axis 1000 Å – 10 keV (bridging
  UV and X-ray).
- **Source**: `notebooks/models/notebook_code/04_agn.py` §2a (Power-Law Disc) for X-ray slope setup.
  New sweep code for α_ox using `xray_alpha_ox` parameter.
- **Figure**: `plt.subplots(1, 2, figsize=(12, 4))`.
- **Story thread**: "→ Beyond UV/optical/IR/X-ray, radio emission traces both star formation
  and AGN jets — the next notebook shows what that regime adds."

---

### `06_multiwavelength_gallery.ipynb` — "Multi-Wavelength Coverage: IGM, Radio, and X-ray"

**Driving question**: *What can radio and X-ray observations tell us that optical photometry
cannot — and how do we tell star-formation-driven emission from AGN?*

**Narrative arc**: Four self-contained sections each cover one wavelength regime beyond the
classical UV–optical SED. IGM sets the high-z context. Radio ties star formation to synchrotron
via the FIR-radio correlation, then shows when AGN jets take over. X-ray closes the loop: the
XRB component traces SFR independently of dust, until AGN L_bol rises and dominates. Together
the four sections make the case that multi-wavelength coverage is the primary lever for breaking
the age–dust–SFR degeneracy.

#### Section 1 — "How the universe absorbs high-z galaxies: the Lyman break"
- **Content**: 2 panels. Left: same galaxy SED in observed-frame for z = [0.1, 0.5, 1, 2, 4, 6],
  with SDSS and JWST filters as gray shaded bands. Right: dropout color criterion (e.g.,
  u−g vs g−r) vs redshift, showing the photometric redshift selection.
- **Source**: `notebooks/models/notebook_code/06_nebular.py` §2 (IGM Absorption — redshift sweep
  with filter overlay). Filter band overlay pattern from `notebooks/models/notebook_code/05_igm.py`.
- **Figure**: `plt.subplots(1, 2, figsize=(14, 4))`. Colormap: `plasma` (z).
- **Story thread**: "→ Star-forming galaxies also emit radio synchrotron, and the ratio of
  that radio emission to FIR luminosity is one of the most robust SFR tracers we have."

#### Section 2 — "Radio synchrotron: the FIR-radio correlation"
- **Content**: 2 panels. Left: full SED from 1 μm to 1 m for three FIRRC calibrations
  (Bell+2003, Delvecchio+2021, McCheyne+2022). Right: zoom to radio regime (1 cm – 1 m).
- **Source**: `notebooks/models/notebook_code/08_radio.py` §1 (FIRRC Calibration Comparison,
  figsize=(11, 4.5)). Imports: `radio_sfr_bell2003`, `radio_sfr_delvecchio2021`,
  `radio_sfr_mccheyne2022`, `radio_total`. Saves to `08_radio_firrc_calibrations.png`.
- **Figure**: `plt.subplots(1, 2, figsize=(11, 4.5))`. Colormap: `Blues`.
- **Story thread**: "→ AGN jets can dominate the radio — the next panel shows how to
  distinguish star-formation-powered radio from AGN jets."

#### Section 3 — "Radio spectral index and AGN radio jets"
- **Content**: 2 panels each for spectral index sweep and AGN radio (simple vs double power-law).
  Component decomposition panel () shows SF + AGN + free-free breakdown.
- **Source**: `notebooks/models/notebook_code/08_radio.py` §2 (Synchrotron Spectral Index,
  figsize=(7, 4)), §4 (AGN Radio Models, figsize=(11, 4.5)), §5 (Component Decomposition).
  Saves to `08_radio_spectral_index.png`, `08_radio_agn.png`, `08_radio_components.png`.
  Also see `notebooks/models/notebook_code/07_multiwavelength.py` §3 for the AGN radio section.
- **Figure**: `plt.subplots(1, 2, figsize=(12, 4))`. Colormaps: `Purples`, `Reds`.
- **Story thread**: "→ The same star-forming galaxies that produce synchrotron radio emission
  also host X-ray binaries — a faint but measurable X-ray signal that scales cleanly with SFR."

#### Section 4 — "X-ray emission: star-forming galaxies and the AGN transition"
- **Content**: 3 panels.
  - **Left**: XRB component for a pure star-forming galaxy — full SED from UV to hard X-ray
    (0.5–10 keV). Sweep `sfr` = [1, 5, 20, 100] M☉/yr to show the linear `L_X–SFR` scaling
    from Mineo et al. (2012) / Lehmer et al. (2016). Label the 2–10 keV band with a gray shaded
    region.
  - **Center**: SFR recovery from X-ray luminosity alone: scatter plot of input SFR vs
    inferred L_X (0.5–8 keV), with the Lehmer+2016 calibration line overlaid.
  - **Right**: AGN transition — same galaxy but with `agn_log_l_bol` sweeping
    [43, 44, 44.5, 45, 46] erg/s. Show when the AGN X-ray dominates over the XRB floor
    (crossover marked with a dashed vertical line at L_AGN_Xray ≈ L_XRB).
- **Key equations**:
  ```
  L_X(SFR) = α * SFR  [Mineo+2012; α ≈ 2.6×10^39 erg/s per M☉/yr]
  L_X(AGN) ∝ L_bol^0.7  [2–10 keV bolometric correction, Hopkins+2007]
  ```
- **Source**:
  - XRB SFR scaling: `src/tengri/models/xray.py` — `xray_lx_sfr()` function (L_X from SFR).
    Check the normalization constant against Mineo+2012 Table 1 / Lehmer+2016 Eq 1.
  - AGN X-ray: `notebooks/models/notebook_code/04_agn.py` §2a (Power-Law Disc — includes
    AGN L_bol → L_X conversion and xray_alpha_ox parameter).
  - Reference output: `notebooks/models/notebook_code/07_multiwavelength.py` — multiwavelength
    overview already produces a combined X-ray panel; use its SED plot template.
- **Figure**: `plt.subplots(1, 3, figsize=(18, 5))`. Left/center colormap: `YlOrRd` (SFR).
  Right colormap: `RdPu` (AGN L_bol). Mark the XRB floor as a dashed horizontal line in the
  right panel.
- **Story thread**: "→ With the full forward model understood — stars, dust, gas, AGN, radio,
  and X-ray — we can now fit real data. The next notebook shows the complete fitting workflow
  for broadband photometry."

---

### `07_fitting_photometry.ipynb` — "Photometric SED Fitting: Single Galaxy to Catalog Scale"

**Driving question**: *I have flux measurements in 8 bands. How do I turn them into a
posterior over stellar mass, SFR, and dust?*

**Narrative arc**: Starts with the same mock from the quickstart (familiar anchor), then
gradually extends: show the full posterior, add a spectrum to break degeneracies, scale to 100
galaxies with vmap, and end with a practical filter-choice guide.

#### Section 1 — "The full posterior for one galaxy"
- **Content**: Use the same mock galaxy and model as `00_quickstart` (same PRNGKey, same
  parameters). But now show `result.corner()` + SED posterior band + SFH recovery side-by-side.
- **Source**: `notebooks/quickstart/notebook_code/01_quickstart.py` "Parameter recovery" section for corner + SFH panel layout.
  `notebooks/quickstart/notebook_code/01_quickstart.py` cells 1–15 for the photometry setup and batch structure.
- **Figure 1**: `result.corner()` (built-in posterior corner). Figure 2: `plt.subplots(1, 2,
  figsize=(14, 5))` — SED posterior band (left) + SFH recovery (right).
- **Story thread**: "→ Adding even a single spectrum dramatically tightens this posterior,
  because the continuum shape independently constrains dust and age."

#### Section 2 — "Photometry alone vs joint photometry + spectrum"
- **Content**: Side-by-side comparison of posteriors from phot-only vs joint fit. Show 4
  key parameters: stellar mass, SFR, dust τ, metallicity. For each: two PDFs overlaid.
- **Source**: `notebooks/specialist/notebook_code/07_advanced_spectroscopy.py` "Comparing Constraints: Photometry vs Spectroscopy vs Joint" section.
  Also `notebooks/specialist/notebook_code/07_advanced_spectroscopy.py` "Photometry vs Spectroscopy: Head-to-Head" for the CI width comparison table.
- **Figure**: `plt.subplots(1, 4, figsize=(16, 4))`.
- **Story thread**: "→ The same forward model that runs once for one galaxy can run in
  parallel on 100 via JAX vmap — here's what catalog-scale fitting looks like."

#### Section 3 — "One hundred galaxies at once: batch fitting with vmap"
- **Content**: Generate 100 mock galaxies with `model.mock_batch(key, n=100, snr=15.0)`.
  Fit with `model.fit_batch()`. Plot true vs recovered stellar mass scatter.
- **Source**: `notebooks/quickstart/notebook_code/01_quickstart.py` "C. Batch Fitting via fit_batch" section — near direct copy.
- **Figure**: `plt.subplots(1, 1, figsize=(6, 6))`. x = true log M*, y = recovered.
  Color by dust optical depth. Diagonal = perfect recovery.
- **Story thread**: "→ The quality of your posterior depends on which filters you use —
  here is a practical guide to filter set selection."

#### Section 4 — "Which filters matter: a practical guide"
- **Content**: 3 panels. For three filter configurations (SDSS-only, +2MASS, +WISE): show
  the 1D posterior PDF for stellar mass. Demonstrate how each additional band tightens the
  constraint.
- **Source**: `notebooks/specialist/notebook_code/03_model_checking.py` "4. Breaking the Degeneracy with More Data" + "6. Practical Guidance"
  sections. Reuse the filter comparison code directly.
- **Figure**: `plt.subplots(1, 3, figsize=(15, 4))`.
- **Story thread**: "→ When you also have a spectrum, the analysis changes — emission lines
  become the dominant constraint on metallicity and ionization."

---

### `08_fitting_spectra.ipynb` — "Spectroscopic Fitting: Resolution, Noise, and Calibration"

**Driving question**: *I have a medium-resolution spectrum. What extra information does it
give me beyond photometry, and how do I get tengri to use it?*

**Narrative arc**: Three things make spectra different from photometry: you see individual
lines (giving metallicity), you see the continuum shape at fine resolution (breaking age-dust
degeneracy), and you have to deal with wavelength-dependent calibration errors and bad pixels.

#### Section 1 — "A 200-pixel mock spectrum and its fit"
- **Content**: Generate a R~2000 mock spectrum (4000–9000 Å, 200 pixels). Fit with
  `eline_mode="marginalized"`. Show the fit with residuals in a 2-row panel.
- **Source**: `notebooks/specialist/notebook_code/07_advanced_spectroscopy.py` cells 1–7 (setup + Figure 1: Mock spectrum with annotated features +
  Figure 2: Spectral fit + residuals). Direct copy of the spectrum fit display pattern.
- **Figure**: `plt.subplots(2, 1, figsize=(14, 7))`. Top: spectrum + model. Bottom: residuals.
  Annotation arrows on Hβ, [OIII], Hα, [NII], [SII].
- **Story thread**: "→ The value of a spectrum is that it constrains parameters photometry
  cannot touch — particularly stellar metallicity."

#### Section 2 — "What a spectrum constrains that photometry cannot"
- **Content**: 4 panels. For each of met_logzsol, dust_tau_bc, SFR, stellar_mass: phot
  posterior (gray fill) vs spec posterior (color fill). Show the dramatic improvement for
  metallicity.
- **Source**: `notebooks/specialist/notebook_code/07_advanced_spectroscopy.py` "Photometry vs Spectroscopy: Head-to-Head" section (cells 20–24).
  The CI width comparison table is also useful here.
- **Figure**: `plt.subplots(1, 4, figsize=(16, 4))`.
- **Story thread**: "→ Real spectra have outlier pixels and systematic calibration errors
  that a Gaussian likelihood cannot handle — here's how to deal with both."

#### Section 3 — "Outlier pixels: Student-t likelihood"
- **Content**: 2 panels. Left: Gaussian fit to spectrum with intentional outlier pixels
  (the fit is pulled). Right: Student-t fit (ν=3) — outliers are ignored.
  `SpectroscopyConfig(noise_dof=3.0)`.
- **Source**: `notebooks/specialist/notebook_code/07_advanced_spectroscopy.py` "2. Student-t Likelihood" and "3. Fitting with Noise Model" sections.
  Direct copy. The intentional outlier injection code is in "1. The Calibration Floor Concept".
- **Figure**: `plt.subplots(1, 2, figsize=(14, 5))`.
- **Story thread**: "→ Calibration errors are even more insidious because they look like
  real signal — tengri can analytically marginalize them away."

#### Section 4 — "Flux calibration errors: analytic marginalization"
- **Content**: 2 panels. Left: mock spectrum with a 5% multiplicative slope calibration error.
  Right: posterior on Chebyshev calibration polynomial coefficients vs true values.
  `SpectroscopyConfig(n_cal_poly=4)`.
- **Source**: `notebooks/specialist/notebook_code/07_advanced_spectroscopy.py` "6. Analytic Calibration Marginalization" and
  "Calibration marginalization in practice" sections.
- **Figure**: `plt.subplots(1, 2, figsize=(12, 4))`.
- **Story thread**: "→ Even the best fit hides hidden constraints and degeneracies — the
  next notebook maps them explicitly."

---

### `09_degeneracies.ipynb` — "Parameter Degeneracies and Information Content"

**Driving question**: *Which parameters are actually constrained by my data, and which just
echo my prior?*

**Narrative arc**: Short and honest. Shows the age-dust-metallicity triangle as a 2D posterior
(what degeneracies look like). Provides a visual lookup table for "what constrains what" so
readers can plan their observations before fitting.

#### Section 1 — "The triangle: age, dust, and metallicity are coupled"
- **Content**: 2D marginalized posteriors showing the age-dust-metallicity degeneracy. Corner
  plot for phot-only fit highlighting the τ_BC–met_logzsol correlation and the age–τ_diff
  anti-correlation.
- **Source**: `notebooks/specialist/notebook_code/03_model_checking.py` "3. The Degeneracy in Action" section. The corner plot + degeneracy
  annotation is the primary content of this notebook section.
- **Figure**: `result.corner(params=["met_logzsol", "dust_tau_bc", "dust_tau_diff",
  "sfh_tsnorm_peak_lbt_gyr"])`.
- **Story thread**: "→ The Fisher information matrix quantifies these degeneracies before
  you run a single fit."

#### Section 2 — "Predicting degeneracies with the Fisher matrix"
- **Content**: Fisher information ellipses for SDSS-only fit. Show the principal axes of the
  Fisher matrix as ellipses in parameter space.
- **Source**: `notebooks/specialist/notebook_code/03_model_checking.py` "2. Generate and Fit a Mock Galaxy" + related cells for Fisher
  computation. The `fitter.fisher_matrix()` call pattern.
- **Figure**: `plt.subplots(1, 1, figsize=(8, 6))`.
- **Story thread**: "→ The practical question is which observations best break which
  degeneracy — here is a one-page guide."

#### Section 3 — "A practical guide: what constrains what"
- **Content**: A visual table (matplotlib imshow or colored text) — rows = parameters,
  columns = data types (UV phot, optical phot, NIR phot, R~200 spec, R~2000 spec). Color-coded
  from red (unconstrained) to blue (well-constrained).
- **Source**: New figure. Values from `notebooks/specialist/notebook_code/03_model_checking.py` "6. Practical Guidance" section and
  `notebooks/specialist/notebook_code/07_advanced_spectroscopy.py` "Feature Accessibility vs Redshift" section.
- **Figure**: `plt.subplots(1, 1, figsize=(10, 6))` with `ax.imshow()`.

---

### `10_real_data.ipynb` — "End-to-End Fitting: SDSS Data to Physical Posteriors"

**Driving question**: *I have a real SDSS galaxy. What steps take me from raw FITS files
to physical posteriors I can trust?*

**Narrative arc**: The reader has been working with clean mock data. Real data has unit
conversions, masked pixels, noise estimation, and potential model failures. This notebook
traces all of those steps for one real galaxy, then shows how to validate the result.

#### Section 1 — "Loading real data: units, masks, and noise"
- **Content**: Load SDSS photometry for a z~0.1 galaxy (or simulated "realistic" data).
  Walk through: Maggies → Jy, masking bad photometry, estimating noise floor.
- **Source**: `notebooks/specialist/notebook_code/01_real_data.py` "Loading and Preparing Data" section — near-direct copy.
- **Story thread**: "→ The model needs a prior that matches what we know about this galaxy
  before we look at the data."

#### Section 2 — "Setting up the model: prior choices matter"
- **Content**: Show a prior predictive check before fitting. Does the prior produce sensible
  SEDs? Does it cover the observed flux range?
- **Source**: `notebooks/specialist/notebook_code/03_model_checking.py` "1. Prior Predictive: Parametric Model" + "2. Diagnosing Bad Priors".
  The prior predictive sweep is essential for real data.
- **Figure**: `model.prior_predictive(n=200).plot()` — shows 200 prior draws over the data.
- **Story thread**: "→ With a validated prior, the fit is one call away."

#### Section 3 — "The fit: joint photometry + spectroscopy"
- **Content**: `result = model.fit(phot=(flux, noise), spectrum=(spec_flux, spec_noise))`.
  Show timing. Show convergence check.
- **Source**: `notebooks/specialist/notebook_code/01_real_data.py` "Fitting" section. `notebooks/quickstart/notebook_code/01_quickstart.py` "Fitting with vi" for timing.
- **Story thread**: "→ The posterior predictive check verifies that the model can reproduce
  the observed data — if it fails, the model is wrong."

#### Section 4 — "Does the model fit the data? Posterior predictive check"
- **Content**: `result.plot_posterior_predictive()`. Show 200 posterior draws over the data.
  Highlight any systematic residuals.
- **Source**: `notebooks/specialist/notebook_code/03_model_checking.py` "9. Comparison: Photometry vs Spectroscopy" and "10. Information
  Content Summary" sections.
- **Figure**: `plt.subplots(1, 2, figsize=(14, 5))` — photometry (left), spectrum (right).
- **Story thread**: "→ Now translate the posterior into physical quantities: stellar mass,
  SFR, emission line fluxes."

#### Section 5 — "Physical quantities from the posterior"
- **Content**: `result.stellar_mass()`, `result.sfr()`, `result.line_fluxes()`. Print a table.
  Plot BPT diagram with the galaxy's position.
- **Source**: `notebooks/specialist/notebook_code/02_derived_quantities.py` (derived quantities — entire notebook), `notebooks/specialist/notebook_code/01_real_data.py` "Caveats" section.
  The BPT plot code from `notebooks/specialist/notebook_code/02_derived_quantities.py`.
- **Figure**: `plt.subplots(1, 1, figsize=(6, 5))` — BPT diagram.

---

### `11_population.ipynb` — "Hierarchical Inference: Shared Priors Across Galaxy Populations"

**Driving question**: *When you have 20 galaxies all born from the same parent population,
how do you let them teach each other?*

**Narrative arc**: Starts with the intuition (20 separate posteriors, each wide). Shows that
sharing a prior shrinks them. Demonstrates that the population recovers the true PSD
hyperparameters. Ends with a practical note on when hierarchical fitting is worth the cost.

#### Section 1 — "The intuition: posteriors shrink when galaxies share a prior"
- **Content**: Matplotlib-patches-only figure. Left half: 20 small SFH panels with wide
  posteriors (independent fits). Right half: same 20 panels with narrowed posteriors
  (hierarchical). Arrow between halves: "shared PSD prior".
- **Source**: New figure (pure matplotlib, no model calls). Style from `notebooks/specialist/notebook_code/06_simulation_sfh.py` "1. Individual
  Fits: Weak Constraints" section for layout inspiration.
- **Figure**: `plt.subplots(4, 5, figsize=(15, 10))`.
- **Story thread**: "→ The shared prior is parameterized by PSD hyperparameters σ and τ —
  here is what a Gibbs update looks like in practice."

#### Section 2 — "Running a hierarchical fit: 20 galaxies"
- **Content**: Generate 20 mock galaxies with the same underlying PSD. Run
  `HierarchicalFitter`. Show convergence of hyperparameters across Gibbs iterations.
- **Source**: `notebooks/specialist/notebook_code/06_simulation_sfh.py` "2. Hierarchical Inference" section — trim to the minimal working
  example (use 20 galaxies, not 100). Keep the Gibbs iteration convergence plot.
- **Figure**: `plt.subplots(1, 2, figsize=(12, 4))` — hyperparameter traces (left),
  final posterior on σ and τ (right).
- **Story thread**: "→ The key test: does the population recover the true PSD parameters?"

#### Section 3 — "Does it work? PSD hyperparameter recovery"
- **Content**: Corner plot of σ and τ hyperparameters. Truth marked as dashed lines.
  Side-by-side: individual fits (wide, biased) vs hierarchical fits (narrow, unbiased).
- **Source**: `notebooks/specialist/notebook_code/06_simulation_sfh.py` "3. Photometric Hierarchical" and "4. √N Scaling" sections.
- **Figure**: `plt.subplots(1, 2, figsize=(12, 4))`.
- **Story thread**: "→ Hierarchical fitting is the right tool when your galaxies share a
  common origin. For datasets where that doesn't hold, you need more flexible models."

#### Section 4 — "When to use it: populations vs fields"
- **Content**: Brief markdown section + one figure. Show `notebooks/specialist/notebook_code/06_simulation_sfh.py` "5. Distinguishing
  Populations" — two populations with different PSD parameters, and whether hierarchical
  fitting can tell them apart.
- **Source**: `notebooks/specialist/notebook_code/06_simulation_sfh.py` "5. Distinguishing Populations" section.
- **Figure**: `plt.subplots(1, 1, figsize=(8, 5))`.

---

### `12_extending_tengri.ipynb` — "Custom Components: Priors, SFH Models, and Dust Laws"

**Driving question**: *The built-in models don't cover my science case. How do I add a
new prior, a new SFH model, or a new dust law?*

**Narrative arc**: Short and practical. Shows the three extension points in order of
complexity: custom prior (5 lines), custom PSD (20 lines), custom dust law (30 lines).
Each extension is a complete working example.

#### Section 1 — "Custom priors in 5 lines"
- **Content**: Define `TruncatedCauchy` distribution inheriting from `tengri.distributions`.
  Show how it slots into `ParamSpec` as a prior for any parameter.
- **Source**: `notebooks/specialist/notebook_code/04_extending_tengri.py` "Custom Prior: TruncatedCauchy" section — direct copy, minor prose
  polish.
- **Story thread**: "→ For more complex customizations, you can define a new PSD shape that
  captures domain knowledge about how your galaxies form stars."

#### Section 2 — "Custom PSD: a new burstiness kernel"
- **Content**: Implement a Matérn-3/2 PSD as a drop-in replacement for DRW. Register it with
  the SFH registry. Show that the resulting SFHs have different correlation structure.
- **Source**: `notebooks/specialist/notebook_code/04_extending_tengri.py` "Custom PSD: Matérn" section — direct copy.
- **Figure**: `plt.subplots(1, 2, figsize=(12, 4))` — power spectrum (left), sample SFHs (right).
- **Story thread**: "→ For completely different physics — a new dust law with a custom
  analytic form — the extension follows the same protocol."

#### Section 3 — "Custom dust law: plug-in attenuation curve"
- **Content**: Implement a simple power-law attenuation curve as a custom dust module.
  Show it produces sensible SEDs and that the gradient flows through it (JAX-compatible).
- **Source**: `notebooks/specialist/notebook_code/04_extending_tengri.py` "Custom Dust: Calzetti vs Charlot & Fall" section and "Summary" section.
- **Figure**: `plt.subplots(1, 2, figsize=(10, 4))` — custom attenuation curve (left),
  SED with custom dust (right).

---

### Implementation Notes for Agents

**The golden rule**: every section in every notebook should be derivable by copying the
indicated source code, trimming prose, and updating the opening sentence to match the
narrative title. Agents should NOT rewrite physics from scratch — they should curate.

**Ordering constraint**: implement notebooks in this order to avoid forward-reference issues:
1. `_plot_style.py` additions (add `sweep_parameter`, `SWEEP_CMAPS`, `SED_XLIM`)
2. `00_quickstart.ipynb` (uses the simplest API, easiest to verify)
3. `01_sed_anatomy.ipynb` (establishes the visual language for everything that follows)
4. `02_sfh_gallery.ipynb` through `06_multiwavelength_gallery.ipynb` in order
5. `07_fitting_photometry.ipynb` through `10_real_data.ipynb` in order
6. `11_population.ipynb` and `12_extending_tengri.ipynb` last

**Tone guard**: before submitting any notebook, check the opening markdown cell of each
section against the tone rules in "Tone and Language" above. If it starts with "In this
section we demonstrate..." — rewrite it. The subject should be the physics, not the software.
