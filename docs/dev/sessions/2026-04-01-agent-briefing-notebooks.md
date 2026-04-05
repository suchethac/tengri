# Agent Briefing: Tengri Notebook & Docs Refactor

Read this file first. It tells you everything you need to know, where to find it, and what to do.

---

## 1. What You Are Doing

You are refactoring the tengri project's Jupyter notebooks, Sphinx documentation, and Sphinx gallery examples. The goal is to replace 39 scattered notebooks across three legacy directories with 26 focused notebooks in 5 purpose-built tracks, fix the broken Sphinx gallery, and update the docs site structure.

**You are NOT writing the paper. You are NOT changing any source code in `src/`. You are only working in:**
- `notebooks/` — Jupyter notebooks (jupytext percent-format `.py` files)
- `docs/` — Sphinx documentation (Markdown + RST)
- `examples/` — Sphinx gallery scripts (`plot_*.py`)
- `implementation_plans/` — where your instructions live

---

## 2. All Relevant File Locations

### The main implementation plan (READ THIS FULLY FIRST)
```
/Users/suchethacooray/Projects/tengri/implementation_plans/notebooks_docs_refactor.md
```
This is the comprehensive plan. It contains every notebook specification, section by section, with source files listed. Read it entirely before starting any work.

### Project rules and conventions (READ BEFORE WRITING ANY CODE)
```
/Users/suchethacooray/Projects/tengri/CLAUDE.md
```
Contains: build commands, code style, naming conventions, critical gotchas (deprecated API names, JAX rules, parameter naming). **You must follow the gotchas section exactly or your notebooks will produce silent bugs.**

### Context documents you may need to reference
```
/Users/suchethacooray/Projects/tengri/docs/internal/AGN_MODEL_COMPARISON.md
  → Comparison of tengri AGN vs CIGALE, Prospector, Bagpipes. Use for models/04_agn.py.

/Users/suchethacooray/Projects/tengri/docs/internal/MODEL_DESCRIPTIONS.md
  → Complete physics formulas for all models. Use when you need parameter names or equations.

/Users/suchethacooray/Projects/tengri/docs/internal/design_philosophy.md
  → Design decisions. Context only.
```

### Source notebooks to copy from (the old ones)
```
/Users/suchethacooray/Projects/tengri/notebooks/tutorials/notebook_code/
  → 01_quickstart.py, 02_the_api.py, 03_the_model.py, 04_the_forward_model.py, 05_prior_predictive.py

/Users/suchethacooray/Projects/tengri/notebooks/demonstrations/notebook_code/
  → 01_spectroscopic_fitting.py through 15_hierarchical_spectroscopy.py

/Users/suchethacooray/Projects/tengri/notebooks/reference/notebook_code/
  → 01_psd_physics.py through 19_model_gallery_nebular.py
```

### Shared plot style utilities (import in every notebook)
```
/Users/suchethacooray/Projects/tengri/notebooks/_plot_style.py
  → Provides: setup_style(), COLORS, convergence_table(), plot_sfh()
  → Every notebook uses this. See any existing notebook for the import pattern.
```

### SSP data file (used in all notebooks)
```
/Users/suchethacooray/Projects/tengri/data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5
```

### Current Sphinx gallery scripts (to fix)
```
/Users/suchethacooray/Projects/tengri/examples/
  → 21 existing plot_*.py scripts, organized in subdirectories
```

### Current docs site
```
/Users/suchethacooray/Projects/tengri/docs/
  → index.md, getting_started/, the_model/, inference/, worked_examples/, etc.
```

### Archive destination
```
/Users/suchethacooray/Projects/tengri/notebooks/archive/
  → Move deprecated notebooks here. Do not delete them.
```

---

## 3. New Folder Structure to Create

Create these directories (they do not exist yet):

```
notebooks/quickstart/notebook_code/
notebooks/quickstart/figures/
notebooks/fitting/notebook_code/
notebooks/fitting/figures/
notebooks/theory/notebook_code/
notebooks/theory/figures/
notebooks/models/notebook_code/
notebooks/models/figures/
notebooks/specialist/notebook_code/
notebooks/specialist/figures/
```

---

## 4. Implementation Phases (do in order)

### Phase 1 — Archive 12 old notebooks
Move these to `notebooks/archive/`:

| File to move | Full path |
|---|---|
| `02_the_api.py` | `notebooks/tutorials/notebook_code/02_the_api.py` |
| `05_inference_methods.py` | `notebooks/demonstrations/notebook_code/05_inference_methods.py` |
| `15_hierarchical_spectroscopy.py` | `notebooks/demonstrations/notebook_code/15_hierarchical_spectroscopy.py` |
| `01_psd_physics.py` | `notebooks/reference/notebook_code/01_psd_physics.py` |
| `02_data_information_content.py` | `notebooks/reference/notebook_code/02_data_information_content.py` |
| `03_dust_models.py` | `notebooks/reference/notebook_code/03_dust_models.py` |
| `04_agn_and_igm.py` | `notebooks/reference/notebook_code/04_agn_and_igm.py` |
| `05_nebular_emission.py` | `notebooks/reference/notebook_code/05_nebular_emission.py` |
| `07_gradient_sensitivity.py` | `notebooks/reference/notebook_code/07_gradient_sensitivity.py` |
| `08_ray_tracing_sampler.py` | `notebooks/reference/notebook_code/08_ray_tracing_sampler.py` |
| `11_advanced_agn.py` | `notebooks/reference/notebook_code/11_advanced_agn.py` |
| `12_nebular_backends.py` | `notebooks/reference/notebook_code/12_nebular_backends.py` |

### Phase 2 — Copy 13 notebooks unchanged
Copy these files to their new locations. Change only the jupytext header path if needed.

| Source | Destination |
|---|---|
| `tutorials/notebook_code/01_quickstart.py` | `quickstart/notebook_code/01_quickstart.py` |
| `demonstrations/notebook_code/01_spectroscopic_fitting.py` | `fitting/notebook_code/01_fitting_spectra.py` |
| `demonstrations/notebook_code/10_degeneracies.py` | `fitting/notebook_code/04_degeneracies.py` |
| `demonstrations/notebook_code/11_joint_fitting.py` | `fitting/notebook_code/03_joint_fitting.py` |
| `demonstrations/notebook_code/13_model_comparison.py` | `fitting/notebook_code/05_model_comparison.py` |
| `demonstrations/notebook_code/06_derived_quantities.py` | `specialist/notebook_code/02_derived_quantities.py` |
| `demonstrations/notebook_code/07_extending_tengri.py` | `specialist/notebook_code/04_extending_tengri.py` |
| `demonstrations/notebook_code/08_fitting_real_data.py` | `specialist/notebook_code/01_real_data.py` |
| `demonstrations/notebook_code/14_emission_line_marginalization.py` | `specialist/notebook_code/05_emission_line_marginalization.py` |
| `reference/notebook_code/09_simulation_sfh.py` | `specialist/notebook_code/06_simulation_sfh.py` |
| `reference/notebook_code/10_multiwavelength.py` | `models/notebook_code/07_multiwavelength.py` |
| `reference/notebook_code/16_model_gallery_dust_emission.py` | `models/notebook_code/03_dust_emission.py` |
| `reference/notebook_code/18_model_gallery_sfh.py` | `models/notebook_code/01_sfh_models.py` |

### Phase 3 — Expansion notebooks (copy base + add sections)
For each: copy the base file, then insert new sections as described in the main plan.

1. `fitting/notebook_code/02_fitting_photometry.py`
   - Base: `demonstrations/notebook_code/02_photometric_catalogs.py`
   - Add at end: photometry precomputation section, fused kernels section, fit_batch section

2. `fitting/notebook_code/06_advanced_inference.py`
   - Base: `demonstrations/notebook_code/12_advanced_inference.py`
   - Insert after NUTS section: full content of `reference/notebook_code/08_ray_tracing_sampler.py`

3. `fitting/notebook_code/07_hierarchical_psd.py`
   - Base: `demonstrations/notebook_code/04_hierarchical_inference.py`
   - Append at end: full content of `demonstrations/notebook_code/15_hierarchical_spectroscopy.py`

4. `models/notebook_code/01_sfh_models.py`
   - Base: `reference/notebook_code/18_model_gallery_sfh.py` (already copied in Phase 2)
   - Add at top: "when to use which model" table (see main plan Section 4 for table contents)

5. `models/notebook_code/02_dust_attenuation.py`
   - Base: `reference/notebook_code/15_model_gallery_attenuation.py`
   - Add after two-component section: age-dust degeneracy figure from `reference/notebook_code/03_dust_models.py` Section 5

6. `specialist/notebook_code/03_model_checking.py`
   - Base: `tutorials/notebook_code/05_prior_predictive.py`
   - Append at end: full content of `reference/notebook_code/02_data_information_content.py`

### Phase 4 — Rewrite/new notebooks
These require writing new content. The main plan specifies every section in detail.

1. `quickstart/notebook_code/02_tengri_capabilities.py`
   - NEW. Six science-facing figures (SFH posterior, corner plot, convergence table, scaling chart, Jacobian heatmap, gradient timing)
   - Source code pattern from: `tutorials/02_the_api.py` (setup boilerplate only), `demonstrations/06_derived_quantities.py` lines 130-157 (SFH figure), `reference/07_gradient_sensitivity.py` lines 40-90 (Jacobian)

2. `quickstart/notebook_code/03_bursty_sfh_recovery.py`
   - Base: `demonstrations/notebook_code/03_bursty_sfh_recovery.py`
   - Append: full content of `demonstrations/notebook_code/09_high_redshift_jwst.py` as new section "Why It Matters at JWST"

3. `theory/notebook_code/01_sfh_prior.py`
   - Base: `tutorials/notebook_code/03_the_model.py`
   - Insert after "Burstiness Plane" section: Green's functions section from `reference/notebook_code/01_psd_physics.py` (final section of that file)

4. `theory/notebook_code/02_forward_model.py`
   - Base: `tutorials/notebook_code/04_the_forward_model.py`
   - Replace the brief Jacobian section with the full content of `reference/notebook_code/07_gradient_sensitivity.py`

5. `models/notebook_code/04_agn.py`
   - Base: `reference/notebook_code/17_model_gallery_agn.py`
   - Add sections from `reference/notebook_code/11_advanced_agn.py`: BH spin panel, Type 1/2 masking section
   - Add comparison table at top (from `docs/internal/AGN_MODEL_COMPARISON.md`)
   - Add AGN+galaxy composite figure from archived `reference/notebook_code/04_agn_and_igm.py`

6. `models/notebook_code/05_igm.py`
   - NEW. 3 figures only. Source: IGM sections from archived `reference/04_agn_and_igm.py` (sections 5-6) + `reference/19_model_gallery_nebular.py` IGM section

7. `models/notebook_code/06_nebular.py`
   - Base: `reference/notebook_code/19_model_gallery_nebular.py`
   - Add: Q_H section from `reference/12_nebular_backends.py`
   - Add: full Cue parameter effects section (new code — 3 sub-figures for N/O, C/O, ionizing shape)
   - Add: Cue vs CloudyGrid comparison from `reference/12_nebular_backends.py`
   - Add: shock BPT figure from `reference/05_nebular_emission.py`
   - Add: DIG contamination figure from `reference/05_nebular_emission.py`

8. `specialist/notebook_code/07_advanced_spectroscopy.py`
   - NEW. Concatenate: `reference/06_noise_models.py` + `reference/13_spectroscopic_tools.py` + `reference/14_alpha_enhancement.py`

### Phase 5 — Fix Sphinx gallery (21 existing scripts)
Read all `.py` files in `examples/`. Find and replace these deprecated names:

| Old (broken) | New (correct) |
|---|---|
| `ForwardModel` | `Model` |
| `fit_catalog` | `fit_batch` |
| `tau_v1` | `tau_bc` |
| `tau_v2` | `tau_diff` |
| `dust_n` | `dust_slope` |
| `sigma_ps` | `psd_sigma` |
| `tau_ps` | `psd_tau_yr` |
| `log_z` | `log_z_abs` |
| `geovi_nifty` or `geovi_full` | `nifty_geovi` |
| `mgvi_nifty` or `mgvi_full` | `nifty_mgvi` |
| `charlot_fall` import | `two_component_dust(law_bc="power_law")` |
| `sfh_alpha` | `sfh_dpl_alpha` (check full prefix with spec.free_params) |

After fixing, run `cd /Users/suchethacooray/Projects/tengri && source .venv/bin/activate && python examples/quickstart/plot_first_fit.py` to verify one script runs cleanly before fixing all.

### Phase 6 — Write 8 new gallery scripts
Create these new files. Each is 60–80 lines, produces one figure, starts with a module docstring. See main plan Section 6.4 for each script's purpose and source.

| File to create | Figure |
|---|---|
| `examples/sfh/plot_bursty_recovery.py` | 4-panel SFH recovery across burstiness regimes |
| `examples/sfh/plot_wrong_model_trap.py` | M★ posterior: stochastic vs parametric on same galaxy |
| `examples/inference/plot_hierarchical_convergence.py` | Posterior width vs N galaxies (1/√N scaling) |
| `examples/agn/plot_agn_type12.py` | Type 1 vs Type 2 SED from geometric masking |
| `examples/dust/plot_dust_emission_models.py` | All 10 dust emission models overlaid |
| `examples/nebular/plot_bpt_diagnostics.py` | BPT diagram: HII + shocks + demarcation lines |
| `examples/spectroscopy/plot_spectral_features.py` | D4000, Hδ, Mg b vs age for 3 metallicities |
| `examples/advanced/plot_fisher_degeneracy.py` | Parameter uncertainty vs filter set (bar chart) |

### Phase 7 — Add README.rst to each gallery subdirectory
Create one file per directory with a 2–3 sentence intro. Sphinx Gallery uses these as section headers.

Files to create:
- `examples/quickstart/README.rst`
- `examples/sfh/README.rst`
- `examples/inference/README.rst`
- `examples/agn/README.rst`
- `examples/dust/README.rst`
- `examples/nebular/README.rst`
- `examples/photometry/README.rst`
- `examples/spectroscopy/README.rst`
- `examples/advanced/README.rst`

Format for each README.rst:
```rst
Section Title
=============

One or two sentences describing what this section of the gallery shows.
```

### Phase 8 — Update Sphinx docs
Edit these files:

**`docs/index.md`**: Add an architecture flowchart (ASCII art, after the intro paragraph) and a "Start here" routing table (see main plan Section 5.3).

**`docs/inference/index.md`**: Add the inference method decision table at the very top (before any other content). See main plan Section 5.3 for the full table.

**`docs/performance/benchmarks.md`**: Add real timing numbers: 140 μs forward (D=7), 356 μs (D=137), 56/63 μs gradient, 21.6× photometry precomputation speedup, native_geovi 56s compile + 0.3s/0.8s run.

**`docs/getting_started/index.md`**: Update notebook links to point to new `quickstart/` track paths.

**`docs/worked_examples/index.md`**: Reorganize by the 5 new tracks (quickstart, fitting, theory, models, specialist).

**Create `docs/getting_started/concepts.md`**: One-page "how tengri works" without code. Three paragraphs: (1) the SFH is a latent field, (2) the forward model maps it to observables, (3) inference inverts this. Add ASCII diagram of the pipeline.

**Remove these three deprecated stub files**:
- `docs/tutorials/index.md`
- `docs/demonstrations/index.md`
- `docs/reference/index.md`

---

## 5. Rules You Must Follow

### Jupytext format
Every `.py` notebook file must start with this header:
```python
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
Copy this header from any existing notebook.

### JAX setup line (REQUIRED at top of every notebook)
```python
jax.config.update("jax_enable_x64", True)
```
This MUST come before any JAX operation. Put it immediately after imports.

### Parameter naming (will cause silent errors if wrong)
- Free params use full prefixes: `sfh_dpl_alpha` NOT `sfh_alpha`
- Always verify with `spec.free_params` and `spec.sample(key).keys()`
- PSD timescale: user-facing = `psd_tau_myr` (Myr); internal = `psd_tau_yr` (years)
- Photometry precomputation requires `redshift=Fixed(...)` in ParamSpec
- `fit_batch` not `fit_catalog`
- vmap batch path requires `method="native_geovi"`, not `"geovi"`

### Importing plot utilities
Every notebook imports from `_plot_style.py` using this pattern (copy from any existing notebook):
```python
import sys, os
try:
    _nb_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(_nb_dir, "..", ".."))
except NameError:
    _nb_dir = os.getcwd()
    sys.path.insert(0, os.path.join(_nb_dir, ".."))
# chdir to project root
if os.path.exists("data"): pass
elif os.path.exists(os.path.join("..", "data")): os.chdir("..")
elif os.path.exists(os.path.join("..", "..", "data")): os.chdir(os.path.join("..", ".."))
elif os.path.exists(os.path.join("..", "..", "..", "data")): os.chdir(os.path.join("..", "..", ".."))
from _plot_style import COLORS, convergence_table, plot_sfh, setup_style
setup_style()
```

### Linting (run before finishing)
```bash
cd /Users/suchethacooray/Projects/tengri
source .venv/bin/activate
ruff check notebooks/ examples/ --fix
ruff format notebooks/ examples/
```
Zero violations required before any commit.

### Syncing notebooks to .ipynb
After writing or editing any `.py` notebook:
```bash
cd /Users/suchethacooray/Projects/tengri/notebooks
jupytext --sync quickstart/notebook_code/*.py
jupytext --sync fitting/notebook_code/*.py
# etc. for each track
```

---

## 6. Standard ParamSpec Templates

Use these as the base for most notebooks. Copy exactly.

**Smooth galaxy (D=7):**
```python
spec = ParamSpec(
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
```

**Stochastic galaxy (D~137):**
```python
spec_stochastic = ParamSpec(
    sfh_dpl_alpha=Uniform(0.5, 5.0),
    sfh_dpl_beta=Uniform(0.5, 5.0),
    sfh_dpl_log_peak_sfr=Uniform(-1.0, 2.5),
    sfh_field_psd_sigma=Uniform(0.01, 1.5),
    sfh_field_psd_tau_myr=Uniform(10.0, 500.0),
    met_logzsol=Uniform(-2.0, 0.2),
    dust_tau_bc=Uniform(0.0, 2.0),
    redshift=Fixed(0.1),
    mean_sfh_type="dpl",
    stochastic_sfh=True,
)
```

---

## 7. Verification Steps

After completing all phases:

```bash
# 1. Full test suite must still pass (notebooks should not break tests)
cd /Users/suchethacooray/Projects/tengri
source .venv/bin/activate
pytest tests/ -q

# 2. Lint check
ruff check notebooks/ examples/ && ruff format --check notebooks/ examples/

# 3. Verify one notebook executes
cd notebooks && python quickstart/notebook_code/01_quickstart.py

# 4. Verify one gallery script executes
cd /Users/suchethacooray/Projects/tengri
python examples/quickstart/plot_first_fit.py

# 5. Build docs (check for errors)
cd docs && make html 2>&1 | grep -i error | head -20
```

---

## 8. Where Memory Is Stored (for reference)

The planning agent stored context here. You can read these for background but they are not required for implementation:

```
/Users/suchethacooray/.claude/projects/-Users-suchethacooray-Projects-tengri/memory/MEMORY.md
  → Index of all memory files

/Users/suchethacooray/.claude/projects/-Users-suchethacooray-Projects-tengri/memory/project_paper_series_vision.md
  → Paper series vision (3-paper arc, community tone)

/Users/suchethacooray/.claude/projects/-Users-suchethacooray-Projects-tengri/memory/project_perf_optimization.md
  → Performance optimization context

/Users/suchethacooray/.claude/plans/squishy-snacking-galaxy.md
  → Earlier draft of the plan (superseded by notebooks_docs_refactor.md)
```

**The single most important file for implementation is:**
```
/Users/suchethacooray/Projects/tengri/implementation_plans/notebooks_docs_refactor.md
```
Read it in full. It contains every notebook specification with section-by-section descriptions.

---

*End of agent briefing.*
