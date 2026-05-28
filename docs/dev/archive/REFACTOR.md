# Tengri Refactor: Lessons and Ground-Up Design

> What we did, what went wrong, what we would do differently if starting from scratch.
> Companion to [`docs/dev/20260404-refactor.md`](20260404-refactor.md) (the executed 7-phase plan). That doc is the *what*.
> This doc is the *why* and the *before* — what a clean-sheet design would look like.
>
> **Last updated**: 2026-04-05 — synthesized from updated module inventory in `20260404-refactor.md`.

---

## Root Causes (in severity order)

### 1. No design contract before code

Class naming conventions were never agreed on. The result:

- `SpectroscopyConfig` is a data container (should be `Spectroscopy`)
- `NoiseConfig` is a statistical model (should be `NoiseModel`)
- `ParamSpec` is a parameter registry (should be `Parameters`)
- `HierarchicalFitter` is an orchestrator (should be `PopulationFitter`)
- `LineCatalog` sounds like a DB table (should be `LineList`)

Parameter naming grew into 3 layers with no explicit contract:

- Short user-facing: `log_total_mass`
- Full prefixed public: `sfh_tsnorm_log_total_mass`
- Internal: `log_total_mass` (same spelling, type-dependent meaning — silent bug factory)

Module boundaries were never defined. `fitter.py` became a 4305-line god file because nobody
decided what "inference" owns vs. what "model" owns.

### 2. Settings and parameters were never separated at the type level

Settings = *which sub-model is active* (dust law, nebular backend) — not in gradient tape.
Parameters = *scalars with priors that get fitted* — in gradient tape.

These are fundamentally different types. They were conflated in `ParamSpec` and `Model.__init__`,
making both enormous. `DustConfig`, `NebularConfig`, `AGNConfig` were designed as frozen
dataclasses much later and retrofitted.

In JAX-based code this distinction is especially critical: settings are Python-level dispatch
resolved *before* JIT; parameters are JAX arrays passed *into* JIT. Conflating them causes
`if nebular:` branches inside compiled functions, forcing recompilation per configuration change
or tracing incorrectly.

### 3. Physics categories accumulated without defaults or deprecation discipline

- SFH models: 8+ (dpl, tsnorm, snorm, norm, lnorm, exp, dexp, const, field, field+burst)
- Dust attenuation: 6 laws
- Dust emission: 5+ models
- AGN: 4 models
- Inference: 14+ named methods for 5 distinct algorithms

No guidance on which to use. No recommended defaults. Each was added as "another option"
rather than replacing or superseding a prior one. The correct default rule is the inverse:
**every new model supersedes a prior one unless explicitly documented otherwise**.

For SFH models: `field` subsumes `dpl` in expressive power but `dpl` was never deprecated
because it was faster. Fine — but that decision should have been documented at the time `field`
was added: "dpl remains as the fast parametric option; field is the flexible default; the
others are legacy." Instead all eight are equal options with no guidance.

### 4. The "tier 2" SED path was never built (later fixed in Phase 5)

- **Tier 1**: Fused photometry (fixed-z + fixed filters) → ~140 µs, 20× speedup
- **Tier 2**: Compositional rest-frame SED (all physics, free-z, JIT'd) → ~300–500 µs
- **Tier 3**: Exact fallback (tabulated SFH, evolving Z, Python dispatch) → ~500–1000 µs

All non-tier-1 work fell through to slow tier 3 because the tier-2 architecture was designed
but never implemented. Performance tiers need to be designed and at least stubbed before
the pipeline hardens.

### 5. Gradient correctness was never systematically tested

All gradient tests checked `isfinite()` only. They pass with wrong signs or missing terms.
The CLOUDY marginalization bug (wrong `ln_L`, biasing MAP/VI gradients) passed all tests.
`cloudy_grid_line_priors()` had zero coverage.

`isfinite()` is not a gradient test. For differentiable scientific code, the only meaningful
test is agreement with finite differences: does `jax.grad` agree with `(f(x+ε) - f(x-ε)) / 2ε`
to within `eps=1e-5`? This is what catches sign errors, missing terms, and wrong normalizations.

### 6. External dependency traps

NIFTy had 99.8% Python overhead (partial wrapping, type checking, dict dispatch). Required
full reimplementation in JAX primitives. The lesson: isolate external library APIs behind
internal abstractions from day one; do not let NIFTy types leak into inference logic.

### 7. Documentation outran implementation

`docs/models/` had full write-ups for: chemical evolution Z(t), shock emission (MAPPINGS),
ADAF disc, MAGPHYS dust, THEMIS dust, patchy IGM reionization, alpha-enhanced SSPs — all
with zero or near-zero code. Creates false scope expectations and makes the project feel
unmanageable.

---

## What Should Have Been Done First

These are not bureaucratic gates. Each prevents a specific class of failure that compounds
over time. Skipping any one of them produces a known category of debt.

### Step 0A — Write three failing API scripts before any class definitions

The scripts define every class name, method name, and parameter name before implementation.
Commit them. They are CI — the API cannot regress.

```python
# grad_student.py — photometric fit, parametric SFH
model = SEDModel.from_config(
    ssp="fsps_default", sfh="dpl", dust="charlot_fall",
    filters=["sdss_u", "sdss_g", "sdss_r"], redshift=0.1,
)
result = model.fit(flux, noise)
result.plot()

# expert.py — stochastic SFH, spectroscopy, MCMC validation
settings = ModelConfig(
    sfh=FieldSFHConfig(n_grid=50),
    nebular=NebularConfig(backend="cloudy", grid_path="..."),
)
params = Parameters()
params.add("sfh_field_psd_sigma", LogUniform(0.1, 3.0))
result = SEDModel(ssp=grid, settings=settings, params=params, obs=obs).fit()
result.refine("mcmc_raytrace", n_steps=1000)

# population.py — hierarchical PSD hyperparameters
pop = SEDModel.fit_population(
    obs_list, shared=["sfh_field_psd_sigma", "sfh_field_psd_tau_myr"]
)
pop.plot_population()
```

The failure mode prevented: API sprawl. If you commit to `model.fit(flux, noise)` as the
one-liner, you can never accidentally require the simple case to construct a `Fitter` manually.

### Step 0B — Write the suffix convention table before any class definitions

| Suffix        | Meaning                                        | Examples                            |
|---------------|------------------------------------------------|-------------------------------------|
| (none)        | Concrete class with behavior                   | `SEDModel`, `Fitter`, `Posterior`   |
| `Config`      | Frozen settings, NOT in gradient tape          | `DustConfig`, `NebularConfig`       |
| `Backend`     | Swappable physics implementation               | `CloudyBackend`, `CueBackend`       |
| `Model`       | Statistical model                              | `NoiseModel`, `CalibrationModel`    |
| `List`        | Registry / catalog                             | `LineList`, `FilterList`            |
| `-er` suffix  | Active orchestrator                            | `Fitter`, `PopulationFitter`        |

Forbidden: `Spec` (vague), `Config` on data containers, `Result` (use `Posterior`).

These suffixes communicate the *role* of a class — what you need when reading unfamiliar
code under time pressure.

### Step 0C — Write the unit contract table before any parameter code

Every parameter must state: user-facing unit / internal unit / conversion. Populate this
table before writing `param_translate.py`.

| Parameter            | User-facing   | Internal     | Conversion     |
|----------------------|---------------|--------------|----------------|
| `sfh_field_psd_tau`  | Myr           | yr           | × 10⁶          |
| `sfh_*_tau_peak`     | Gyr           | yr           | × 10⁹          |
| `met_logzsol`        | log(Z/Z☉)     | log(Z) abs   | + LOG10_ZSUN   |
| `dust_tau_bc`        | optical depth | optical depth | (none)         |

In JAX, unit errors do not throw exceptions — they produce wrong numbers at plausible
magnitudes. A unit contract table forces enumeration of every conversion at the
public/internal boundary before any silent bugs can accumulate.

### Step 0D — Draw the module boundary tree with explicit import rules

```
tengri/
├── core/
│   ├── model.py          < 300 lines: thin orchestrator, no physics
│   ├── parameters.py     parameter registry + prior definitions only
│   ├── settings.py       DustConfig, NebularConfig, AGNConfig (frozen dataclasses)
│   ├── param_map.py      name translation (one layer: public → internal)
│   ├── fused_kernels.py  JIT kernel builders
│   ├── sed_pipeline.py   SED computation engine
│   └── noise.py          NoiseModel
├── inference/
│   ├── fitter.py         < 150 lines: dispatch table only
│   ├── map.py            MAP + optimizer
│   ├── vi.py             geoVI + MGVI (JIT path + NIFTy path)
│   ├── mcmc.py           NUTS + Ray Tracing + ESS
│   ├── evidence.py       Nested Slice Sampling
│   ├── population.py     PopulationFitter
│   └── posterior.py      Posterior + PopulationPosterior
└── models/               physics (unchanged structure, but 1 default per category)
```

**Explicit dependency rules** (write these in each `__init__.py` as comments, enforce in CI):

- `core/` never imports from `inference/`
- `inference/` never imports from `models/` directly — only through `core/`
- `models/` only imports from `utils/`
- `utils/` imports only from stdlib and third-party (jax, numpy, etc.)

Without these rules written down, every "just this once" cross-boundary import is locally
justified and globally corrosive.

### Step 0E — Define the parameter namespace before any Parameters code

```
{sfh_type}_{param}       sfh_dpl_alpha, sfh_field_psd_sigma
met_{param}              met_logzsol
dust_{param}             dust_tau_bc, dust_tau_diff, dust_slope
neb_{param}              neb_logU, neb_logZ_gas, neb_fesc
agn_{submodel}_{param}   agn_disc_log_mbh, agn_torus_frac
eline_{param}            eline_broad
noise_{param}            noise_f_cal, noise_dof
```

No short names in the public API except those explicitly documented in the alias table.
No internal names that shadow public names. Enforce with a regex CI check:
`^(sfh_|met_|dust_|neb_|agn_|eline_|noise_)` — anything not matching is legacy or a bug.

### Step 0F — Write finite-difference gradient tests before each physics function

For every function that will go inside a JIT boundary, write the FD test first.
The test defines the numerical contract. The docstring describes the physics. The code
satisfies both.

```python
# Template for any new differentiable function
def test_my_function_gradient():
    x = jnp.array([...])
    eps = 1e-5
    grad_auto = jax.grad(my_function)(x)
    grad_fd = (my_function(x + eps) - my_function(x - eps)) / (2 * eps)
    np.testing.assert_allclose(grad_auto, grad_fd, rtol=1e-4)
```

The failure mode prevented: the CLOUDY marginalization bug, where wrong `ln_L` normalization
biased MAP/VI gradients for months while passing all `isfinite()` checks.

---

---

## Current State (as of 2026-04-05)

This section captures honest status against the original plan, based on the module inventory
in `20260404-refactor.md`. Not everything marked "complete" in the original plan is actually
at target line counts.

### Execution status

| Phase | Description | Plan status | Reality |
|-------|-------------|-------------|---------|
| 1 | Naming contract | ✅ Complete | Old names still work with `DeprecationWarning` — will be removed v1.0 |
| 2A | Split `fitter.py` | ✅ Complete | `fitter.py` 4305 → 1064 lines. Target was < 150. Inline runners and loss builders still present |
| 2B | Split `model.py` | ⚠️ Partial | 2538 → 2115 lines. Extracted `convenience.py`, `display.py`, `sed_components.py`. `__init__` and `from_config` still contain physics-adjacent setup logic |
| 3 | Settings/Parameters split | ⚠️ Partial | `settings.py` exists (244 lines). `param_spec.py` still 1760 lines — still accepts model selection flags that belong in `ModelConfig` |
| 4 | Canonical method names | ✅ Complete | Now 13 canonical strings (was 9 planned — organic growth added `laplace`, `pathfinder`, `vi_native`, `vi_native_linear`) |
| 5 | Tier 2 SED path | ✅ Complete | `build_fused_rest_sed` + `sed_components.py` (433 lines). Tier dispatch working |
| 6A/B/C | Science fixes + gradient tests | ✅ Complete | CLOUDY 2D interp, ln_L normalization, FD test suite all done |
| 7 | Prune stub docs | ✅ Complete | `docs/models/` removed |

### What remains (unfinished refactor work)

**`fitter.py` (~1064 lines, original target < 150 — not met):**
- **Delegated already:** loss/prior/loglik builders in `loss_functions.py` (Fitter only caches them); MAP / laplace / pathfinder in `map_dispatch.py`; MCMC in `mcmc.py`; native VI in `vi.py`; NIFTy VI via `vi.run_nifty_vi`; JIT engine in `jit_engine.py`.
- **Still here:** large `__init__`, `_data_args`, `compile()`, `run()` dispatch, `fit_batch()`, and VI posterior-sample helpers (`_draw_jit_*`, `_draw_blackjax_samples`, etc.).
- **Next (optional):** move compile + draw helpers behind `vi`/`fitter_support` modules; move init/unbounded helpers to a tiny `init_params` or `common` — realistic goal is shrinking toward **< 600 lines**, not < 150.

**`model.py` (2115 lines, target < 500):**
- `__init__` contains extensive setup: tier selection, kernel building, dust age weight
  precomputation, CSP matrix setup, parameter map construction
- `from_config()` is a large factory method branching over all physics options
- `predict_photometry()`, `predict_spectrum()`, `predict_sed()` contain tier dispatch + fallback logic
- Fix: extract `__init__` setup into a `_ModelBuilder` or separate `_init_*.py` modules

**`param_spec.py` (1760 lines, target: split into `parameters.py` + settings moved out):**
- `ParamSpec.__init__` still accepts `mean_sfh_type`, `nebular`, `dust_emission`, `radio`, `igm` flags
- These flags belong in `ModelConfig` / `SFHConfig` / `NebularConfig` / `MultiwavelengthConfig`
- Fix: deprecate model selection flags in `ParamSpec.__init__` once migration noise is manageable; remove in v1.0 (deferred — see `20260404-refactor.md` Phase 3 migration path)

### Organic growth since the original plan

The inference layer gained exactly what the physics layer gained before it: new options added
alongside old ones without deprecation discipline.

| New file | Lines | Notes |
|----------|-------|-------|
| `inference/sbi.py` | 396 | SBI infrastructure — not in original plan |
| `inference/laplace.py` | 151 | Laplace approximation |
| `inference/pathfinder.py` | 134 | BlackJAX Pathfinder |
| `inference/geovi_nuts.py` | — | geoVI-NUTS hybrid |
| `inference/standardized.py` | — | Standardized parameter space |
| `inference/jit_engine.py` | — | JIT compilation engine |
| `core/precompute_templates.py` | 279 | Template precomputation |
| `core/display.py` | 257 | Introspection helpers |

The canonical method count grew from 9 planned to 13. `vi_native` emerged as 500× faster than
the NIFTy default — but has not been validated across all model configurations. The default
remains `vi` (NIFTy) until validation is complete. This is the correct decision, but it means
the faster path exists without users knowing it.

---

## Rules for New Features Going Forward

Before adding any new model or inference method:

1. **Write the docstring first.** If you cannot describe it in 3 lines, the abstraction is wrong.

2. **Add the parameter to the namespace table** in `docs/parameter_reference.md`. If the name
   clashes or doesn't fit the prefix scheme, rename before writing code.

3. **Write a finite-difference gradient test.** If the transform is not differentiable, it
   cannot go inside a JIT boundary.

4. **Designate one default per physics category.** If adding a new SFH model, document which
   prior model it supersedes and why both still exist (if they do).

5. **If the code does not exist, do not document it in `docs/`.** Put it in `ROADMAP.md`.

6. **Isolate external libraries behind internal abstractions from day one.** Never let
   external types (NIFTy dicts, bagpipes arrays) leak into internal function signatures.
