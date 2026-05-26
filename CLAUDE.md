# Claude Code Instructions for tengri

## Project overview

Differentiable SED fitting code in JAX. Models galaxy star formation histories as IFT correlated fields with PSD-governed burstiness priors. Uses DSPS for differentiable stellar population synthesis.

**Name:** `tengri`.
**Paper draft:** *(private paper draft)*
**Paper I:** Methods + mock recovery. **Paper II:** Real data.

## Build/test commands

```bash
# Lint and format (ALWAYS run before committing)
.venv/bin/ruff check src/ tests/
.venv/bin/ruff format --check src/ tests/
.venv/bin/ruff check --fix src/ tests/    # auto-fix
.venv/bin/ruff format src/ tests/          # auto-format

# Tests (~2224 tests, ~295s)
.venv/bin/pytest tests/ -q

# Benchmarks (run after changes to forward model or inference)
JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/benchmark_forward_model.py
JAX_PLATFORMS=cpu .venv/bin/python scripts/test_vi_memory_hybrid.py

# Notebook sync (jupytext percent-format)
cd notebooks && jupytext --sync *.py

# Cross-validation (NOT run by default)
.venv/bin/pytest -m crossval tests/crossval/
```

## JAX persistent compilation cache

`import tengri` auto-enables a persistent on-disk JAX compile cache at
`~/.cache/tengri_jax_cache` so notebook restarts, slurm tasks, and benchmark
worker subprocesses all skip the expensive first compile (geoVI ~75 s, MGVI
~10 s). Configure via env:

```bash
export TENGRI_JAX_CACHE_DIR=/scratch/$USER/jax_cache  # custom location
export TENGRI_DISABLE_JAX_CACHE=1                     # opt out
```

After upgrading JAX (`pip install -U jax`), wipe stale entries:

```python
import tengri; tengri.clear_cache()
```

Default `min_compile_time_secs=0.05` persists per-filter
`compute_flux_density` kernels and other ~150–250 ms component
precompute compiles. Threshold history: 5.0 (≤ 2026-05-04, skipped
the orchestrator chain), 0.5 (≤ 2026-05-22, missed per-filter
micro-compiles), 0.05 (current). See
`docs/inference/compilation_cache.md` for full details.

## Naming contract (MANDATORY)

**Read `docs/dev/NAMING_CONTRACT.md` before writing any new code, renames, or refactors.**

Canonical names: `SEDModel`, `Parameters`, `Spectroscopy`, `NoiseModel`, `LineList`, `PopulationFitter`.
Deprecated aliases (never use in new code): `Model`, `ParamSpec`, `SpectroscopyConfig`, `NoiseConfig`, `LineCatalog`, `HierarchicalFitter`.

## Code style

- **Ruff** linting + formatting, config in `pyproject.toml` — zero violations required
- Pure JAX functions (no side effects, JIT-compatible)
- Numpydoc docstrings, snake_case, line length 99
- Immutable arrays (`.at[].set()`)
- Units: **years** (time), **Angstrom** (wavelength), **Msun/yr** (SFR), **erg/s/Hz** (SED luminosity L_nu)
- 64-bit precision: `jax.config.update("jax_enable_x64", True)`
- Greek letters (sigma, xi, theta) allowed in docstrings/comments
- **Voice rules** (defensive code, narration, single-use helpers): see [`docs/dev/style-and-voice.md`](docs/dev/style-and-voice.md)

## Documentation (MANDATORY)

**Read `docs/dev/docstring-standard.md` before writing any new function, class, or method.**

**Tier rules (quick reference):**
- **Tier 1 — Public API** (symbols in `__init__.py`): full numpydoc — Parameters, Returns, Raises, Notes (JIT flag), References, Examples.
- **Tier 2 — Scientific functions** (`components/`, `forward/`, `observation/`): Parameters, Returns, Notes (JIT flag + equations + approximation flags), References.
- **Tier 3 — Utilities** (`utils/`, `config/`, `analysis/`): Parameters, Returns minimum.
- **Tier 4 — Private helpers** (`_` prefix): one-sentence summary; Parameters if non-obvious.

**Non-negotiable rules — every violation is a bug:**
- Units ALWAYS in brackets in parameter descriptions: `[erg/s/Hz]`, `[yr]`, `[Msun/yr]`.
- Array shapes ALWAYS annotated: `array_like, shape (n_wave,)` for inputs, `ndarray, shape (n_wave,)` for outputs.
- `.. math::` directive MANDATORY for any function implementing a physical formula. Define every variable with units after the equation.
- Approximations MUST be flagged: *"Approximation of Eq. X in Author+Year — valid for A < B."* Undocumented approximations are a correctness failure.
- Citations MANDATORY when any formula or algorithm comes from a paper. Use `.. [N]` in References with exact title, journal, arXiv ID, and DOI. Never write citations from memory — verify against authoritative sources.
- Upstream code MUST be credited in Notes: *"Ported from Prospector (Johnson et al. 2021 [N]_)"*.
- JIT/grad/vmap compatibility MUST be stated in Notes for all `components/` and `forward/` functions.
- VERIFY equations against the original paper before writing — do not rely on memory or other code.

## Package structure

**For "where do I look to edit X?" — see [`docs/dev/where-things-live.md`](docs/dev/where-things-live.md).**

Layout: `parameters/ -> components/ -> forward/ -> observation/ -> inference/ -> analysis/ -> config/ + utils/`. Public API re-exported at `src/tengri/__init__.py`.

Key directories:
- `core/` — Protocols (SEDComponent, ObservationModel, Likelihood) — Part II scaffold; nothing consumes yet
- `parameters/` — Parameters class, priors, param translation
- `components/` — SED physics: sfh/, sps/, dust/, nebular/, agn/, igm/, radio/, xray/
- `forward/` — SEDModel, pipeline, _kernels/ (JIT strategies, private), precompute/ (protocol+registry)
- `observation/` — photometry, spectroscopy, filters, noise, emission lines
- `inference/` — fitter, posterior, all inference methods (vi, mcmc, nss, map, etc.)
- `analysis/` — diagnostics, plotting, simulate, mock
- `config/` — DustConfig/NebularConfig/SFHConfig/ModelConfig, exceptions, display, deprecation
- `utils/` — cosmology, conversions, interpolation, physics_constants
- `cosmology/` — re-exports from `utils/cosmology` (canonical user-facing path; added 2026-05)
- `units/` — re-exports F_nu/L_nu/AB-magnitude conversions (canonical user-facing path; added 2026-05)
- `plot/` — re-exports plotting helpers from `analysis/plotting` (canonical user-facing path; added 2026-05)
- `_deprecated.py` — `deprecated_alias()` / `deprecated_attribute()` shims (added 2026-05)

Phase 4 sub-namespaces (additive re-export modules; added 2026-05):
- `components/agn/{disc_api,torus_api,lines,compose}.py`
- `components/dust/{attenuation_models,emission_models,pah}.py`

## Model construction API (CURRENT)

User-facing model construction has two surfaces. The **recommended path** is
the nested-dict builder shipped in 2026-05 (`parameters/groups.py`):

```python
from tengri import SEDModel, FREE, FIXED, Fixed, Uniform, recipes

# Preferred: from a recipe
model = SEDModel.build(ssp_data=ssp, observation=obs,
                              **recipes.star_forming_photometry())

# Or hand-rolled with the nested-dict grammar
model = SEDModel.build(
    ssp_data=ssp, observation=obs,
    sfh={'type': 'dpl', '*': FREE, 'beta': Uniform(1, 3)},
    dust={'type': 'two_component', 'law_bc': 'calzetti', '*': FIXED,
          'tau_bc': 0.5, 'emission': {'type': 'dale2014', '*': FIXED}},
    neb={'type': 'cue', '*': FIXED},
    redshift=Fixed(0.05),
)
model.spec.summary()    # provenance-tagged: [user] / [* FREE] / [* FIXED] / [default]
groups = model.spec.to_groups()    # round-trip for inspection/editing

# Or with builder factories (autocomplete-friendly; SFH only as of Phase II-3.3)
from tengri import builders
model = SEDModel.build(
    ssp_data=ssp, observation=obs,
    sfh=builders.sfh.dpl(_=FREE, beta=Uniform(1, 3)),  # ← IDE sees alpha, beta, tau_gyr, log_total_mass
    dust={'type': 'two_component', 'law_bc': 'calzetti', '*': FIXED},
    neb={'type': 'cue', '*': FIXED},
)
```

- Grammar: each group dict accepts `'type'` (structural choice), `'*'`
  wildcard (`FREE`/`FIXED`; default `FIXED`), and per-parameter short-form
  overrides (e.g. `'beta'` inside the sfh group resolves to `sfh_dpl_beta`).
- Sub-blocks: `dust.emission`, plus the five AGN composable selectors —
  `agn.disc`, `agn.torus`, `agn.lines`, `agn.feii`, `agn.atten`. Each nests as
  a dict with its own `'type'`, `'*'`, and per-param keys.
- Sentinels (`FREE`, `FIXED`) are singletons exported from `tengri`.
- Recipes: `tengri.recipes.*` — five curated starting points
  (`star_forming_photometry`, `quiescent_z0`, `agn_panchromatic`,
  `stochastic_sfh_jwst`, `mock_recovery_minimal`). Each docstring states its
  SSP requirement (bare-stellar vs any).
- The flat-kwarg `Parameters(...)` form is the **expert escape hatch** — still
  works, still used internally, but not the recommended user-facing path.

See `docs/dev/api_migration_v0.x.md` for the full grammar reference and
`notebooks/04_building_models.py` for a worked example covering recipe usage,
variant swapping, and round-trip editing. Design plan:
`~/.claude/plans/i-feel-like-its-serene-emerson.md`.

## Key conventions

- **Physical constants**: Import from `utils/physics_constants.py` — do NOT define local constant literals. Exception: `L_SUN_CUE = 3.839e33` in `cue.py` is intentional (Cue training convention, not IAU 2015).
- **Metallicity**: SSP grid is `log10(Z)` absolute, not `log10(Z/Zsun)`. Offset: `LOG10_ZSUN = -1.848`. User-facing `neb_logZ_gas` is Z/Zsun (param_map adds LOG10_ZSUN).
- **ParamSpec free params** use full prefixes: `sfh_dpl_alpha`, `sfh_field_psd_sigma` — NOT shorthand. Check with `spec.free_params`.
- **PSD timescale**: high-level API is **Myr** (`psd_tau_myr`); internal is **years** (`psd_tau_yr`).
- **`agn_log_lbol`**: always `log10(L_bol / L_sun)` at API level. AGN functions convert to erg/s internally.
- **All SED components return erg/s/Hz** (standardized 2026-04-08).
- **Emission line wavelengths**: vacuum throughout (e.g. H-alpha = 6564.61 A). Do NOT use air wavelengths.
- **Nebular constants**: `components/nebular/_constants.py` re-exports from `physics_constants` — don't break these re-exports.
- **AGN shared physics**: `_planck_lnu` in `components/agn/_phys.py` — do NOT duplicate the Planck function.
- **Notebooks**: edit `.py` files (jupytext percent format), never `.ipynb`.

## Adding a new physics block

**Default path — `SEDModelComponent` (one file):**

For any model that has free parameters, a wavelength-dependent emission or
transformation function, and (optionally) a pre-computed library or trained
emulator — closed-form attenuation laws, dust IR libraries, AGN torus
libraries, nebular emulators — subclass `SEDModelComponent`:

```python
class MyModel(SEDModelComponent):
    name = "my_model"               # registry key
    parameter_prefix = "my_"

    T    = Uniform(20.0, 80.0, "temperature",    units="K")
    beta = Uniform( 1.0,  3.0, "emissivity index", units="")

    inputs  = {"L_absorbed": "erg/s"}
    outputs = {"L_ir": "erg/s"}

    def load(self, wave):           # optional: load atlas/weights → self.data
        return None

    def predict(self, p, sed_in, wave, *, L_absorbed):
        sed = my_emission_formula(wave, L_absorbed, p["T"], p["beta"])
        return sed_in + sed, {"L_ir": trapz_freq(sed, wave)}
```

`__init_subclass__` auto-discovers class-level `Distribution` priors,
registers `(name, cls)` so `SEDModel.build(dust={'type': 'my_model'})`
finds it, auto-fills `inputs()`/`outputs()` from the dicts, and provides
sensible `apply()`/`precompute()`. Astronomer writes physics only.

The contract:
* `p` — parameter dict, prefix stripped (`p["T"]`, not `p["my_T"]`)
* `sed_in` — rest-frame L_ν from upstream (erg/s/Hz); zeros if first
* `wave` — rest-frame grid in Å (or filter effective wavelengths under WavePrecomp)
* `**inputs` — keyword args auto-supplied from `state.derived`
* Return `(sed_out, published_dict)` — new SED + dict matching `outputs` keys

**Reference:**
- [`docs/dev/sed-model-components.md`](docs/dev/sed-model-components.md) — full how-to + three worked examples (closed-form, library, NN emulator)
- [`docs/dev/forward-model-architecture.md`](docs/dev/forward-model-architecture.md) — architectural context
- [`docs/adr/0011-sed-model-component-base.md`](docs/adr/0011-sed-model-component-base.md) — the design decision
- [`src/tengri/components/dust/calzetti_model.py`](src/tengri/components/dust/calzetti_model.py) — canonical small port (analytic closed-form)
- [`src/tengri/components/agn/skirtor_model.py`](src/tengri/components/agn/skirtor_model.py) — canonical library port

**Advanced fallback — the bare `SEDComponent` Protocol:**

Reserve this for models that don't fit the `predict(p, sed_in, wave, **inputs)` shape — typically Stellar (SFH + SSP + age weights + nine derived publishes) and IGM (observer-frame transformation). The bare-Protocol pattern is at `src/tengri/protocols/component.py`; the canonical reference is `src/tengri/components/radio/component.py`. Cross-component contract (`inputs/outputs/optional_inputs`) is documented in ADR-0009.

## Adding a new inference backend (InferenceContext shape — ADR-0010)

Every inference backend (MAP, MCMC, VI, NSS, …) receives an
`InferenceContext`, not a `Fitter`. Adding a new sampler is two files:

1. **The runner** — copy `src/tengri/inference/backends/map_dispatch.py` as
   the canonical reference. Signature:

   ```python
   def run_my_sampler(context, *, key, init_from=None, ...):
       from tengri.inference.context import InferenceContext
       from tengri.inference.posterior import Posterior

       context = InferenceContext.from_target(context)
       init_params = context.initial_params(key, init_from=init_from)
       nlp_fn = context.neg_log_posterior_fn   # JIT-cached, minimisation objective
       data_args = context.data_args
       ...
       return Posterior(..., _model=context.model)
   ```

2. **The registration** — one entry in `src/tengri/inference/_registration.py`:

   ```python
   register_backend(
       "my_sampler",
       tier="experimental",
       short_doc="One-line description",
       requires=("optional_dep",),  # if any
       legacy_fitter=False,
   )(run_my_sampler)
   ```

The parametrised conformance suite
(`tests/unit/inference/test_backend_conformance.py`) picks up the new
entry automatically — no test-file edits required.

**JIT rule** (non-negotiable): `InferenceContext` must never be hashed
into a JIT key or passed through `jax.jit` / `jax.vmap` / `jax.lax.scan`
as a traced argument. Pull primitives (`neg_log_posterior_fn`, `data_args`) out of
context *before* entering JAX transforms. The context's
`__jax_array__` guard raises on accidental tracing.

**Source of truth:** `docs/adr/0010-inference-backend-protocol.md`.

## Critical gotchas

- `jax.random.fold_in(key, hash(string))` overflows uint32. Use `abs(hash(x)) % (2**31)`
- Never create `Model`/`ParamSpec` inside a JAX gradient tape (traced values fail in `__init__`)
- JAX Metal (Apple GPU) causes test failures. Use `JAX_PLATFORMS=cpu` for reliable results
- Ray Tracing: step_size=0.05 for D~137; sharp viability cliff at ~0.06 (acceptance drops to 0%)
- NIFTy geoVI: use 4-12 samples per KL iteration, not 80
- `VIConfig.n_samples=3` doubles to 6 effective samples via `mirror_samples=True` — when tuning, think in effective samples
- `"vi"` (NIFTy) and `"vi_native"` (pure-JAX) target the same objective but are NOT posterior-equivalent. Native is ~19× faster warm on 7-D and ~25× on 137-D stochastic (2.8s vs 71s), but PSD timescale `sfh_field_psd_tau_myr` differs by an order of magnitude between paths (82 vs 6 Myr). Validate per-problem before swapping. See `bench/reports/2026-04-17_native_vs_nifty.md`
- Use `.shape[0]` instead of `len()` on JAX arrays to avoid `ConcretizationTypeError` under JIT
- Use tolerance comparison (`abs(x - default) < 1e-6`) not `==` for float equality on traced values
- IGM `igm_transmission(wave_obs, z)` takes **observed-frame** wavelengths (not rest-frame)
- Dust emission templates auto-load from `data/`; analytic fallbacks are NOT suitable for science
- AGN torus in `torus.py` are **toy models** — use SKIRTOR for science
- `agn_torus_frac`: do NOT auto-derive from `cos(theta_torus)` in forward pass (gradient discontinuity)
- Inference internals use `mode="_traceable"` (safe inside JIT). User-facing defaults to `mode="auto"`
- **Build-time `approx=WavePrecomp(...)` is the speed knob** (Phase 3d, 2026-05-20). Opting in publishes the SSP × filter LUT and routes `predict_photometry` through `observation.predict_via_precomp`. Default `approx=None` uses the exact wave-grid path. The dict / bool / string forms (e.g. `approx={'wave_precomp': True}`, `approx=True`, `approx='wave_precomp'`) were removed — `TypeError` at construction. Override ztable sampling via `WavePrecomp(n_z=200, z_min=0.0, z_max=3.0)`.
- **One NUTS fit per notebook process.** Each warmup peaks at 3–6 GB on small models (D ≤ 7 photometry) but can hit 20+ GB on D ≈ 8 with `mean_sfh_type="dense_basis"` — observed 22.78 GB peak on nb00 with default `dense_mass=True`. Multi-fit notebooks (and any single fit on D ≥ 8) need `dense_mass=False` or `mcmc_hmc`. See `docs/dev/notebook_orchestration_oom.md`
- **Subagent rejection ≠ child kill.** A rejected subagent's `python notebook.py` keeps running. After rejecting, run `ps -axo pid,rss,comm | grep python` and `kill -9` zombies

## Testing

**Read `tests/TESTING.md` before writing any test.** It defines the physics-first taxonomy (conservation / bounds / limit / regression_paper / regression_bug / gradient / crossval / contract) and the anti-patterns reviewers reject.

CI guard: `python tools/check_test_markers.py` — every test under `tests/physics/`, `tests/regression/`, `tests/components/`, `tests/contract/` must declare a taxonomy marker.

Test organization:
- `tests/unit/` — fast, no SSP data needed (legacy tree; rehoming in progress)
- `tests/integration/` — needs `data/ssp_*.h5`, skips if missing
- `tests/crossval/` — against bagpipes/FSPS, excluded from default runs
- `tests/physics/`, `tests/regression/`, `tests/contract/` — new structured trees, marker-enforced

Bug fix rule: every fix MUST cite the original paper equation and include a regression test.

Use `chex` for array shape/finite/tree-allclose assertions in tests — see `docs/dev/testing-with-chex.md` for the conventions and conversion recipes.

## Issue / PR labels (apply when opening any new issue or PR)

Every new issue and PR MUST be labelled. Pick at minimum **one area:** label; add **type:**, **cross-cutting**, and GH-default (`bug`/`enhancement`/`documentation`) labels as they apply. Multi-area issues get multiple `area:*` labels.

**Physics areas** (pick if the issue touches that physics subsystem):
`area:agn` · `area:sfh` · `area:dust` · `area:nebular` · `area:stellar` · `area:xray` · `area:radio` · `area:igm`

**Code areas** (pick if the issue touches that internal/public seam):
- `area:api` — `SEDModel.build`, builders, registries, public surface (`__init__.py` re-exports)
- `area:forward` — forward model pipeline, `PipelineState`, kernels, `ForwardModel`
- `area:observation` — photometry, spectroscopy, filters, noise, apertures
- `area:inference` — VI/MCMC/NUTS/MAP backends, `InferenceContext`
- `area:population` — `PopulationSEDModel`, hierarchical PSD fits
- `area:spatial` — `SpatialSEDModel`, fiber, aperture, multi-component spatial
- `area:examples` — sphinx-gallery scripts under `docs/examples/`
- `area:notebooks` — jupytext fundamentals/quickstart notebooks
- `area:docs` — user/dev docs (not gallery, not ADR)
- `area:adr` — architecture decision records under `docs/adr/`
- `area:ci` — GitHub Actions workflows
- `area:perf` — compile time, runtime, benchmarks
- `area:tests` — test infra, taxonomy markers, flakes, rehoming

**Type extensions** (on top of `bug`/`enhancement`):
- `type:refactor` — internal restructure with no behaviour change
- `type:audit` — parity audit vs CIGALE/Prospector/AGNfitter/Synthesizer (finding discrepancies)
- `type:parity` — wire up a specific known upstream model or library variant

**Cross-cutting failure classes** (apply liberally — these are searchable patterns):
- `silent-failure` — silent NaN/Inf returns, dropped kwargs, ignored config (recurring footgun)
- `jit-safety` — tracer leaks, `ConcretizationTypeError`, safe-gradient patterns
- `oom` — memory blowups (warmup, mass matrix, compile)
- `breaking-change` — public API rename or behavioural break

**Examples** (from existing issues):
- "Cue neb_fesc has no effect on LyC" → `area:nebular`, `bug`
- "X-ray missing N_H photoelectric absorption" → `area:xray`, `type:parity`
- "audit: tengri vs CIGALE — stellar normalization 30% low" → `type:audit`, `area:stellar`, `area:dust`, `bug`
- "Derive _VALID_SFH_TYPES from SFH_REGISTRY" → `area:sfh`, `area:api`, `type:refactor`
- "NUTS warmup peaks at 20+ GB on D≈8" → `area:inference`, `oom`, `bug`
- "Tracer leak in Dale 2014 dust IR under jit" → `area:dust`, `jit-safety`, `bug`

Do **not** add: per-component labels (`area:cue`, `area:skirtor`) — too granular; `priority:*` — decays; `area:registry` — covered by title + `type:refactor`.

## qmd search (MANDATORY before reading files)

Search qmd first using `collections: ["tengri"]` before reading any file. Fall back to Read/Glob/Grep only if qmd returns insufficient results.

## References

- `docs/dev/agents.md` — AI agent documentation
- `docs/dev/history/handoff-2026-04.md` — frozen project-status snapshot (pre Phase II-3 closure)
- `docs/dev/design_philosophy.md` — architecture decisions
- `docs/dev/NAMING_CONTRACT.md` — naming conventions (read before any rename/refactor)
- `docs/dev/REFACTOR.md` — refactor plan
- `docs/dev/api_migration_v0.x.md` — public-API migration table (Phase 1→6 + Part II scaffold)
- `docs/known_bugs.md` — bug tracking (all currently fixed)
- `docs/dev/notebook_orchestration_oom.md` — operational rules for OOM-safe notebook authoring (multi-fit, subagent zombies, watchdog)
- `tools/check_param_prefixes.py` — CI guard for free-parameter prefix rule (NAMING_CONTRACT §3.2)
