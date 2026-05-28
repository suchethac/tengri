# API Redesign Plan: Unified, Intuitive, Flexible

**Date:** 2026-04-02  
**Principle:** All changes are additive. Existing code keeps working. New API sits on top.

---

## The Problem

Tengri's current API is powerful but has three friction points that slow down new users and create unnecessary cognitive load for everyone:

**1. Method string proliferation.** `fitter.run()` accepts 15+ method strings. Eight are variants of the same underlying algorithm (geoVI): `geovi`, `native_geovi`, `mgvi`, `native_mgvi`, `evi`, `native_evi`, `fast_geovi`, `fast_mgvi`, `nifty_geovi`, `nifty_mgvi`, `geovi_nuts`. These are internal backend choices (NIFTy tight loop vs JIT-compiled vs full logging), not distinct scientific methods. Users from BAGPIPES/Prospector think "variational inference," not "NIFTy fast path."

**2. Construction verbosity.** Fitting a galaxy currently requires four separate objects: `ParamSpec` (with long prefix names like `sfh_tsnorm_log_total_mass`), `SSPData`, `Observation` (containing `Photometry` and/or `SpectroscopyConfig`), and `Fitter`. A newcomer has to read four different docstrings before writing one line of science.

**3. No first-class chaining.** The common workflow — MAP initialization → variational inference → MCMC validation — requires manually threading `init_from=result` between three separate `fitter.run()` calls. There's no way to express "fit, then validate" as a single pipeline.

---

## Proposed Changes

### Change 1: Unified method taxonomy in `Fitter.run()`

Collapse the 15+ strings into a clean three-tier system. All old names remain as deprecated aliases (emit a `DeprecationWarning`, keep working until v1.0).

**New canonical names:**

| New name | What it does | Old aliases |
|----------|-------------|-------------|
| `"vi"` | Variational inference (geoVI by default) | `"geovi"`, `"native_geovi"` |
| `"vi_linear"` | Linear VI / MGVI | `"mgvi"`, `"native_mgvi"`, `"evi"` |
| `"vi_nifty"` | NIFTy tight loop geoVI | `"fast_geovi"`, `"nifty_geovi"` |
| `"vi_nifty_linear"` | NIFTy tight loop MGVI | `"fast_mgvi"`, `"nifty_mgvi"` |
| `"mcmc"` | MCMC — auto-selects sampler by D | — |
| `"mcmc_raytrace"` | Ray Tracing explicitly | `"raytrace"` |
| `"mcmc_nuts"` | NUTS via BlackJAX | `"nuts"` |
| `"mcmc_ess"` | Elliptical Slice Sampling | `"elliptical_slice"` |
| `"map"` | MAP optimization | `"map"` (unchanged) |
| `"laplace"` | Gaussian at MAP | `"laplace"` (unchanged) |
| `"pathfinder"` | L-BFGS path | `"pathfinder"` (unchanged) |
| `"evidence"` | Nested Slice Sampling | `"nss"` |

**New `method="auto"` default**: auto-selects based on dimensionality.
- D ≤ 15 → `"laplace"` (instant)
- 15 < D ≤ 50 → `"vi_linear"` (fast)
- D > 50 → `"vi"` (full geoVI, default)
- With `validate=True` → appends `"mcmc"` step

**New `vi_flavor=` kwarg** for the `"vi"` method when users need to control the backend (power users only, not in the quick-start docs):
```python
fitter.run("vi")                          # JIT-compiled geoVI — default
fitter.run("vi", vi_flavor="nifty")       # NIFTy tight loop
fitter.run("vi", vi_flavor="nifty_full")  # NIFTy with full logging
fitter.run("vi", vi_flavor="linear")      # geoVI linearized → MGVI
```

**Implementation:**
- `Fitter.run()` in `src/tengri/inference/fitter.py` — add new elif branches for new names, keep all old names as deprecated-alias branches that warn then call the same underlying method.
- Add a `_DEPRECATED_METHOD_ALIASES` dict at top of file for clean maintenance.
- The `"mcmc"` auto-selector should check `self.model.spec.n_free` and pick raytrace vs nuts.

---

### Change 2: `model.fit()` — attach inference to the model

The most common workflow should be the shortest path. `Model.fit()` creates a `Fitter` internally and runs inference. It returns a `Posterior` identical to what `fitter.run()` returns.

**Proposed API:**
```python
# Single galaxy, photometry
result = model.fit(flux_obs, noise)                # method="vi" default
result = model.fit(flux_obs, noise, method="mcmc")

# Single galaxy, spectroscopy
result = model.fit(spectrum, noise, data_type="spectroscopy")

# Single galaxy, joint
result = model.fit(
    photometry=(flux_phot, noise_phot),
    spectrum=(flux_spec, noise_spec),
)

# With initialization chain
result = model.fit(flux_obs, noise, init="map")    # MAP first, then vi
```

**Implementation:**
- Add `Model.fit()` method to `src/tengri/core/model.py`.
- It constructs a `Fitter(self, flux_obs, noise, data_type=data_type)` and calls `fitter.run(method, init_from=map_result if init=="map" else None)`.
- If `data_type` is not provided, infer it: 1D array → spectroscopy, 1D array matching n_filters → photometry.
- Expose `model.fitter_` after fit so users who want fine control can still access the underlying Fitter.

---

### Change 3: `Posterior.refine()` — method chaining

`Posterior` should hold a reference to the Fitter that produced it, so users can refine with a better method in one line.

**Proposed API:**
```python
result_vi = model.fit(flux_obs, noise)
result_exact = result_vi.refine("mcmc", n_steps=1000)   # uses result_vi as init

# Full pipeline in one expression
result = (
    model.fit(flux_obs, noise)
         .refine("mcmc", n_steps=500)
)

# Validate convergence, get both
result = model.fit(flux_obs, noise)
result_check = result.validate()   # runs short raytrace/NUTS, compares posteriors
```

**Implementation:**
- Add `Posterior._fitter` attribute (set by `model.fit()` and `fitter.run()`).
- Add `Posterior.refine(method, **kwargs)` → calls `self._fitter.run(method, init_from=self, **kwargs)`.
- Add `Posterior.validate(n_steps=200)` → runs a short MCMC check and returns a `ValidationResult` with KL divergence between VI and MCMC posteriors.
- `_fitter` is optional — if not present, `refine()` raises a helpful error.

---

### Change 4: `Model.from_config()` — grouped, readable construction

Replace the four-step construction (ParamSpec + SSPData + Observation + Model) with a single dict-based factory. The underlying objects are identical — this is purely syntactic sugar.

**Proposed API:**
```python
model = tengri.Model.from_config(
    ssp="data/ssp_prsc_miles_chabrier_wNE.h5",   # path or SSPData object
    sfh="tsnorm",                                  # SFH family name
    dust="charlot_fall",                           # attenuation law
    nebular="baked_in",                            # nebular backend
    agn=None,                                      # or "simple", "unified_nlr_blr", etc.
    redshift=0.1,                                  # Fixed value or "free"
    filters=["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"],
    wave_obs=None,                                 # set for spectroscopy
    priors={
        # SHORT names — prefix stripped because sfh="tsnorm" is known
        "log_total_mass": Uniform(-1.0, 2.5),
        "peak_lbt_gyr": Uniform(0.5, 12.0),
        "width_gyr": Uniform(0.3, 5.0),
        "skew": Uniform(-3.0, 3.0),
        "trunc": Uniform(1.0, 10.0),
        # Standard short names for everything else
        "logzsol": Uniform(-2.0, 0.2),
        "tau_bc": Uniform(0.0, 2.0),
        "tau_diff": Uniform(0.0, 1.5),
    }
)
```

**Key design decisions:**
- When `sfh="tsnorm"` is provided, the `priors` dict accepts short names (`log_total_mass`) instead of the full prefix (`sfh_tsnorm_log_total_mass`). The factory expands them internally.
- When `sfh="dpl+field"` is provided, short names cover both the DPL (`alpha`, `beta`, `log_total_mass`) and the stochastic field (`psd_sigma`, `psd_tau_myr`).
- `agn="simple"` automatically injects `agn_frac` as a free parameter with a default prior; can be overridden in `priors`.
- `redshift="free"` makes redshift a free parameter; `redshift=0.1` fixes it.
- Returns a standard `Model` object — no new class needed.

**Short name mapping table** (stored in `param_translate.py` or a new `param_aliases.py`):

```python
# sfh_type → {short_name: full_name}
_SFH_SHORT_NAMES = {
    "tsnorm": {
        "log_total_mass": "sfh_tsnorm_log_total_mass",
        "peak_lbt_gyr": "sfh_tsnorm_peak_lbt_gyr",
        "width_gyr": "sfh_tsnorm_width_gyr",
        "skew": "sfh_tsnorm_skew",
        "trunc": "sfh_tsnorm_trunc",
    },
    "dpl": {
        "alpha": "sfh_dpl_alpha",
        "beta": "sfh_dpl_beta",
        "log_total_mass": "sfh_dpl_log_total_mass",
        "tau_gyr": "sfh_dpl_tau_gyr",
    },
    # "field" additions apply to any sfh that includes "+field"
    "field": {
        "psd_sigma": "sfh_field_psd_sigma",
        "psd_tau_myr": "sfh_field_psd_tau_myr",
    },
}
# Universal short names (always valid)
_UNIVERSAL_SHORT_NAMES = {
    "logzsol": "met_logzsol",
    "tau_bc": "dust_tau_bc",
    "tau_diff": "dust_tau_diff",
    "dust_slope": "dust_slope",
    "agn_frac": "agn_frac",
    "neb_logU": "neb_logU",
}
```

**Implementation:**
- Add `Model.from_config(ssp, sfh, dust, nebular, agn, redshift, filters, wave_obs, priors)` classmethod in `src/tengri/core/model.py`.
- It resolves short names → full names using the alias table.
- It builds `Photometry.from_names(filters)` if filters provided.
- It builds `SpectroscopyConfig(wave_obs=wave_obs)` if wave_obs provided.
- It builds the `Observation` object.
- It constructs and returns a standard `Model(spec, ssp_data, observation=obs)`.

---

### Change 5: `fit_catalog()` with DataFrame / astropy Table input

The current `fitter.fit_batch()` takes a list of dicts with numpy arrays. This is awkward for users who have their data in a DataFrame or astropy Table.

**Proposed API:**
```python
import pandas as pd
from astropy.table import Table

# pandas DataFrame
results = model.fit_catalog(
    catalog_df,
    flux_cols=["flux_u", "flux_g", "flux_r", "flux_i", "flux_z"],
    err_cols=["flux_err_u", "flux_err_g", "flux_err_r", "flux_err_i", "flux_err_z"],
    redshift_col="z_spec",    # per-row varying redshift; overrides model's fixed redshift
    method="vi",
    n_workers=1,              # for now: sequential; future: multiprocessing pool
    verbose=True,
)

# astropy Table — same interface
cat = Table.read("sdss_catalog.fits")
results = model.fit_catalog(cat, flux_cols=..., err_cols=..., redshift_col="z")

# Returns: list of Posterior objects, same length as input catalog
# With summary extraction:
summary_df = tengri.posteriors_to_dataframe(results, params=["met_logzsol", "dust_tau_bc"])
```

**Implementation:**
- Add `Model.fit_catalog()` in `src/tengri/core/model.py`.
- Accepts `pd.DataFrame`, `astropy.Table`, or list of dicts.
- If `redshift_col` is provided, creates a per-row model with `spec.with_params(redshift=Fixed(row_z))` for each galaxy (leverages existing `ParamSpec.with_params()`).
- Internally loops and calls `self.fit(row_flux, row_noise, method=method)`.
- Add `tengri.posteriors_to_dataframe(results, params)` utility in `src/tengri/__init__.py`.

---

### Change 6: `model.prior_predictive()` — first-class model checking

Prior predictive checks are currently manual (sample params, call `model.predict_photometry()` in a loop). This should be a one-liner.

**Proposed API:**
```python
ppc = model.prior_predictive(n=500, seed=42)
# Returns a PriorPredictive object (thin wrapper around a dict of arrays)

ppc.flux           # shape (n, n_filters) — predicted photometry
ppc.sfh            # shape (n, n_ages) — SFH draws
ppc.params         # dict of shape-(n,) arrays — drawn parameters

# Plotting (delegates to existing plot utilities)
ppc.plot_seds(n_show=50, color_by="sfh_field_psd_sigma")
ppc.plot_colors("sdss_g-sdss_r", "sdss_r-sdss_i", sdss_locus=True)
ppc.check_finite()   # warn if any draws produce NaN/Inf flux
```

**Implementation:**
- Add `Model.prior_predictive(n, seed)` in `src/tengri/core/model.py`.
- Uses `spec.sample_batch(key, n)` + `jax.vmap(self.predict_photometry)`.
- Returns a `PriorPredictive` namedtuple or dataclass with `.flux`, `.sfh`, `.params`.
- Add convenience `PriorPredictive.plot_seds()` and `PriorPredictive.plot_colors()` that call existing `_plot_style.py` utilities.

---

### Change 7: Unified hierarchical interface via `model.fit_population()`

`HierarchicalFitter` is a completely separate class with a different interface. It should be accessible through the same `model.fit_*` namespace.

**Proposed API:**
```python
# Current (still works)
hf = HierarchicalFitter(model_factory, observations, data_list)
result = hf.run("vi")

# New: same fit_* namespace
pop_result = model.fit_population(
    observations_list,     # list of (flux, noise) tuples or list of Posterior objects
    method="vi",
    population_prior={     # hyperpriors on shared PSD parameters
        "psd_sigma": Gaussian(0.5, 0.5),
        "psd_tau_myr": LogUniform(10, 500),
    },
)

# HierarchicalResult interface matches Posterior interface where possible
pop_result.summary()          # population-level parameters
pop_result.individual[0]      # Posterior for galaxy 0 (marginal)
pop_result.plot_population()  # sigma-tau scatter with credible contours
```

**Implementation:**
- Add `Model.fit_population()` in `src/tengri/core/model.py` as a thin wrapper around `HierarchicalFitter`.
- `HierarchicalResult` gets a `.individual` property returning per-galaxy posteriors.
- `HierarchicalResult` gets `.summary()` and `.plot_population()` matching `Posterior`'s interface pattern.

---

## What NOT to do

- **Don't remove `ParamSpec`** — it's the right abstraction for power users and programmatic construction.
- **Don't remove `Fitter`** — it's needed for compile-then-run workflows and multi-seed fitting.
- **Don't make the short names the only way** — `sfh_tsnorm_log_total_mass` should always work.
- **Don't merge `Photometry` / `SpectroscopyConfig` / `Observation`** — the current split is clean at the type level; the verbosity comes from construction, not the type design.
- **Don't add DataFrames as a required dependency** — make pandas/astropy optional imports in `fit_catalog()`.

---

## Implementation order

Each change is independent and non-breaking. Suggested order by impact/effort ratio:

| # | Change | Effort | Impact |
|---|--------|--------|--------|
| 1 | Method unification (new names + deprecation warnings) | Low | High |
| 2 | `Model.fit()` convenience wrapper | Low | High |
| 3 | `Posterior.refine()` chaining | Low | Medium |
| 6 | `model.prior_predictive()` | Low | Medium |
| 4 | `Model.from_config()` with short names | Medium | High |
| 5 | `fit_catalog()` with DataFrame input | Medium | Medium |
| 7 | Unified hierarchical `fit_population()` | Medium | Medium |

Start with 1 and 2 — they eliminate the two biggest friction points with ~50 lines of code each.

---

## Migration path for existing code

After implementing all changes, the new recommended workflow is:

```python
# Before (current — still works)
from tengri import Model, ParamSpec, Uniform, Fitter, load_ssp_data
from tengri import Observation, Photometry

ssp = load_ssp_data("data/ssp.h5")
obs = Observation(photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r"]))
spec = ParamSpec(
    sfh_tsnorm_log_total_mass=Uniform(-1, 2.5),
    sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12),
    sfh_tsnorm_width_gyr=Uniform(0.3, 5),
    met_logzsol=Uniform(-2, 0.2),
    dust_tau_bc=Uniform(0, 2),
    redshift=0.1,
)
model = Model(spec, ssp, observation=obs)
fitter = Fitter(model, flux_obs, noise)
result = fitter.run("geovi")

# After (new — concise, chained)
import tengri

model = tengri.Model.from_config(
    ssp="data/ssp.h5",
    sfh="tsnorm",
    filters=["sdss_u", "sdss_g", "sdss_r"],
    redshift=0.1,
    priors=dict(
        log_total_mass=tengri.Uniform(-1, 2.5),
        peak_lbt_gyr=tengri.Uniform(0.5, 12),
        width_gyr=tengri.Uniform(0.3, 5),
        logzsol=tengri.Uniform(-2, 0.2),
        tau_bc=tengri.Uniform(0, 2),
    )
)
result = model.fit(flux_obs, noise)         # vi by default
result_exact = result.refine("mcmc")        # optional validation
```

Lines of user code: **~15 → ~10**. Concepts to learn: **4 (ParamSpec, Model, Observation, Fitter) → 1 (Model)** for the common case.

---

## Files to modify (source code)

| File | Change |
|------|--------|
| `src/tengri/core/model.py` | Add `from_config()`, `fit()`, `prior_predictive()`, `fit_catalog()`, `fit_population()` |
| `src/tengri/inference/fitter.py` | Add new method aliases, `"auto"` method, `vi_flavor=` kwarg, deprecation warnings |
| `src/tengri/inference/posterior.py` | Add `_fitter` attribute, `refine()`, `validate()` |
| `src/tengri/core/param_translate.py` | Add `_SFH_SHORT_NAMES` and `_UNIVERSAL_SHORT_NAMES` dicts |
| `src/tengri/__init__.py` | Export `posteriors_to_dataframe`, `Model.from_config` in public API |
| `tests/unit/test_api_convenience.py` | New test file covering all new convenience methods |

---

## Documentation changes required

Every layer of documentation needs updating after the API changes land. An agent implementing this plan must update all of the following — the code change is not done until docs, notebooks, and papers are consistent.

### Sphinx docs site (`docs/`)

| File | What to change |
|------|---------------|
| `docs/index.md` | The quickstart code block (after "Quick start" heading) uses `ParamSpec` + `Model` + `Fitter` + `fitter.run("geovi")` — replace with the new `Model.from_config()` + `model.fit()` pattern |
| `docs/getting_started/concepts.md` | Currently explains the 4-object construction (ParamSpec → SSPData → Observation → Model → Fitter). Rewrite to show the 1-object entry point (`Model.from_config` → `model.fit()`), then explain the underlying objects for users who want fine control |
| `docs/inference/index.md` | The "Which method should I use?" decision table uses old names (`geovi`, `native_geovi`, `raytrace`, `nuts`). Replace with new canonical names (`vi`, `mcmc`, `mcmc_raytrace`, `mcmc_nuts`). Add `vi_flavor=` note for power users. Keep old names as a "Deprecated aliases" footnote. |
| `docs/advanced/convergence.md` | Uses `fitter.run("raytrace")`, `fitter.run("native_geovi")` in all code snippets — update to `fitter.run("mcmc_raytrace")` and `fitter.run("vi")`. Also add a new section showing `result.refine()` and `result.validate()` for the convergence-checking workflow. |
| `docs/advanced/batch_fitting.md` | Uses `fitter.fit_batch(galaxies, method="native_geovi")` — update to `model.fit_catalog(df, ...)` pattern as primary example. Keep `fit_batch` as secondary "low-level" example. Replace all `native_geovi` with `vi`. |
| `docs/advanced/extending.md` | Uses `ParamSpec(sfh_custom_my_param=...)` — keep as-is since extension still requires ParamSpec, but add a note that `Model.from_config()` supports `extra_params={}` for adding custom params without full ParamSpec construction |
| `docs/advanced/hierarchical.md` | Add a new section showing `model.fit_population()` as the high-level entry point for hierarchical fitting, alongside the existing `HierarchicalFitter` docs |
| `docs/performance/benchmarks.md` | The "Inference methods" table uses old names — update to new canonical names in timing comparisons |

### Notebooks (all 26 new notebooks in `notebooks/`)

All notebooks use `fitter.run("geovi")` or `fitter.run("native_geovi")` or similar. After the method unification, every call needs to be updated. The notebooks also need to demonstrate the new convenience methods, not just the low-level ones.

**Systematic find-and-replace** across all notebook `.py` files:
- `fitter.run("native_geovi"` → `fitter.run("vi"`
- `fitter.run("geovi"` → `fitter.run("vi"`
- `fitter.run("native_mgvi"` → `fitter.run("vi_linear"`
- `fitter.run("mgvi"` → `fitter.run("vi_linear"`
- `fitter.run("raytrace"` → `fitter.run("mcmc_raytrace"`
- `fitter.run("elliptical_slice"` → `fitter.run("mcmc_ess"`
- `fitter.run("nss"` → `fitter.run("evidence"`

**Notebooks that need more than find-and-replace** (require showing new API patterns):

| Notebook | What to add |
|----------|------------|
| `quickstart/01_quickstart.py` | Add a cell at the top showing `model.fit()` one-liner before showing the verbose Fitter path. This is the first thing new users see. |
| `quickstart/02_tengri_capabilities.py` | The "Scaling to a catalog" section should show `model.fit_catalog(df)` alongside the vmap timing section |
| `fitting/02_fitting_photometry.py` | Replace the `Fitter(model, flux, noise)` + `fitter.run("native_geovi")` pattern with `model.fit(flux, noise)` as the primary demonstration; keep Fitter path as "under the hood" section |
| `fitting/06_advanced_inference.py` | This is the methods comparison notebook — it MUST be updated to use the new canonical method names. Also add a section demonstrating `result.refine("mcmc_raytrace")` chaining. |
| `specialist/03_model_checking.py` | Add `model.prior_predictive(n=500)` as the first cell, replacing the manual sample-loop prior predictive |

After updating, resync all `.py` to `.ipynb` with `jupytext --sync` and copy to `docs/_notebooks/` via `make notebooks`.

### HANDOFF.md

The API example block at the top of HANDOFF.md (lines ~50-80 under "High-Level API") should show the new `Model.from_config()` + `model.fit()` pattern as the primary example, with the old `ParamSpec` + `Fitter` + `fitter.run()` pattern moved to a "Low-level API" subsection below.

### CLAUDE.md (project instructions)

The "High-level API (preferred)" section in `CLAUDE.md` currently shows `Model`, `ParamSpec`, `Fitter`, `Posterior`. Add `Model.from_config()` and `model.fit()` as the recommended entry points. Update the gotchas section with the new deprecated aliases map so future agents don't use old method names.

---

## Paper changes required

**IMPORTANT FOR PAPER-WRITING AGENT:** The following sections in both papers need revision after the API is implemented. The paper agent should read the final implemented API (check `src/tengri/core/model.py` and `src/tengri/inference/fitter.py`) before revising — do not revise based on this plan alone, since implementation details may change.

### Paper I: Code introduction (*(private paper draft)*)

This paper IS about the API. Every section may be affected. Priority sections:

| Section file | What to revise |
|---|---|
| `6-usage.tex` | **Primary target.** This is the usage / API walkthrough section. The entire section should be rewritten around the new simplified API. Lead with `Model.from_config()` + `model.fit()`. Show method chaining via `result.refine()`. Demonstrate `fit_catalog()` for batch use. The old multi-step construction can be shown as "advanced usage" at the end of the section. |
| `4-inference.tex` | The inference section describes the available methods. Replace the enumeration of 10+ method strings with the new 3-tier taxonomy (optimization → VI → MCMC) and the `"auto"` selector. The underlying algorithms (geoVI, Ray Tracing, NUTS) are unchanged — only the user-facing names. |
| `2-design.tex` | The design section justifies architectural choices. Add a paragraph on the "thin high-level / thick low-level" design philosophy: `Model.from_config()` and `model.fit()` are intentionally thin wrappers that expose the full low-level API for power users. The composability of ParamSpec + Model + Fitter remains available. |
| `8-conclusion.tex` | If the conclusion contains any code snippets or method name lists, update them. Specifically check for mentions of "ten inference backends" — after unification, this should say "multiple inference methods accessible through a unified API" or give the new canonical count. |
| `999-appendix.tex` | Likely no changes needed — the appendix covers mathematical details of Ray Tracing and geoVI, which are unchanged. However, if the appendix refers to `fitter.run("raytrace")` or other old method strings, update them. |

### Paper II: Stochastic SFH (*(private paper draft)*)

This paper focuses on the science (IFT SFH, PSD recovery, hierarchical inference), not the API. However, two places reference the API:

| Section file | What to revise |
|---|---|
| `1-introduction.tex` line 24 | "ten inference backends" — update to reflect the new unified count/taxonomy. Suggested replacement: "multiple inference backends unified under a single API" or "eight inference methods spanning MAP optimization to exact MCMC". |
| `999-appendix.tex` line 7 | "The full catalog of ten inference backends available in tengri is documented in Paper I (Section 4)" — update the number if the canonical count changes after unification. Also update any in-text references to `fitter.run()` method strings if present. |

**What NOT to change in Paper II:**
- The mathematical descriptions of geoVI, Ray Tracing, and MGVI (Appendix A) — these describe algorithms, not API
- Any mention of `ParamSpec`, `Model`, or `Fitter` in the methods section — Paper II's audience already knows the old API
- Scientific results, figures, or tables — none of these depend on the API surface

---

## Summary checklist for the implementing agent

After implementing the API changes, go through this list in order:

**Code:**
- [ ] `Fitter.run()` accepts new canonical names + deprecated aliases with warnings
- [ ] `Model.fit()` works and returns a `Posterior`
- [ ] `Posterior.refine()` chains to a new inference run
- [ ] `Model.from_config()` builds a model from a grouped dict
- [ ] `Model.fit_catalog()` accepts DataFrame and list-of-dicts
- [ ] `model.prior_predictive()` returns a `PriorPredictive` object
- [ ] `model.fit_population()` wraps `HierarchicalFitter`
- [ ] All new methods have docstrings with examples
- [ ] `tests/unit/test_api_convenience.py` covers all new methods
- [ ] `ruff check src/ tests/` passes with zero violations

**Notebooks:**
- [ ] All 26 notebooks updated: old method strings replaced with new canonical names
- [ ] `quickstart/01_quickstart.py` shows `model.fit()` as primary entry point
- [ ] `fitting/06_advanced_inference.py` uses new canonical names throughout
- [ ] `specialist/03_model_checking.py` uses `model.prior_predictive()`
- [ ] All notebooks re-synced to `.ipynb` with `jupytext --sync`
- [ ] `make notebooks` run to copy `.ipynb` to `docs/_notebooks/`

**Docs:**
- [ ] `docs/index.md` quickstart block updated
- [ ] `docs/getting_started/concepts.md` updated
- [ ] `docs/inference/index.md` decision table updated with new names
- [ ] `docs/advanced/convergence.md` code snippets updated
- [ ] `docs/advanced/batch_fitting.md` shows `fit_catalog()`
- [ ] `docs/advanced/hierarchical.md` shows `fit_population()`

**Papers (paper-writing agent):**
- [ ] Paper I `6-usage.tex` — rewrite around new simplified API
- [ ] Paper I `4-inference.tex` — update method taxonomy description
- [ ] Paper I `2-design.tex` — add thin/thick API design note
- [ ] Paper II `1-introduction.tex` line 24 — update "ten inference backends"
- [ ] Paper II `999-appendix.tex` line 7 — update backend count reference

**Repo meta:**
- [ ] `HANDOFF.md` API example updated
- [ ] `CLAUDE.md` high-level API section updated with new method names and `Model.from_config()`
