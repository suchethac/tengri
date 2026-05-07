# Claude Code Instructions for tengri

## Project overview

Differentiable SED fitting code in JAX. Models galaxy star formation histories as IFT correlated fields with PSD-governed burstiness priors. Uses DSPS for differentiable stellar population synthesis.

**Code name:** `tengri` (working name, final TBD).
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
JAX_PLATFORMS=cpu .venv/bin/python scripts/benchmark_forward_model.py
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

Default `min_compile_time_secs=5.0` keeps small SSP/dust kernels out of
the cache. See `docs/inference/compilation_cache.md` for full details.

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

## Adding a new physics block (Phase II-2 component shape)

The forward model is mid-migration to a `SEDComponent`-based pipeline (`src/tengri/core/component.py`). New physics blocks **must** follow the Protocol shape rather than adding branches to `forward/sed_model.py` or `forward/pipeline.py`.

**Canonical adapter to copy:** `src/tengri/components/radio/component.py`. Mirror its layout exactly.

**Required structure** (see `core/component.py` for the Protocol):

```python
@dataclass(frozen=True)
class MySEDComponentConfig(SEDComponentConfig):
    name: str = "my"
    # static knobs only — no parameters that the user fits

@dataclass(frozen=True)
class MySEDComponent:
    config: MySEDComponentConfig = field(default_factory=MySEDComponentConfig)
    name: str = "my"
    parameter_prefix: str = "my_"     # CI-enforced via tools/check_param_prefixes.py

    def declared_parameters(self) -> list[ParamDeclaration]: ...
    def precompute(self, ssp_data=None, wave_grid=None) -> SEDComponentState: ...
    def apply(self, state: PipelineState, params) -> PipelineState: ...
```

**Parameters.** `declared_parameters()` mirrors the entries already in `parameters/_param_defs.py` — do **not** duplicate priors. The `_param_defs.py` registry stays the single source of truth until the migration completes.

**Cross-component coupling.** Read upstream quantities from `state.derived` with a documented fallback (e.g. `state.derived.get("L_ir", 0.0)`); publish your own as `state.derived["L_<name>"]`. Never reach into another component's state directly.

**Source of truth:** `docs/dev/20260404-refactor.md`. The active plan, entropy budget, and PR order live there.

**Do not pre-build** `Parameters.from_components(...)` — it is deferred until ≥5 components have landed.

## Critical gotchas

- `jax.random.fold_in(key, hash(string))` overflows uint32. Use `abs(hash(x)) % (2**31)`
- Never create `Model`/`ParamSpec` inside a JAX gradient tape (traced values fail in `__init__`)
- JAX Metal (Apple GPU) causes test failures. Use `JAX_PLATFORMS=cpu` for reliable results
- Ray Tracing: step_size=0.05 for D~137; sharp viability cliff at ~0.06 (acceptance drops to 0%)
- NIFTy geoVI: use 4-12 samples per KL iteration, not 80
- `VIConfig.n_samples=3` doubles to 6 effective samples via `mirror_samples=True` — when tuning, think in effective samples
- `"vi"` (NIFTy) and `"vi_native"` (pure-JAX) target the same objective but are NOT posterior-equivalent. Native is ~19× faster warm on 7-D and ~25× on 137-D stochastic (2.8s vs 71s), but PSD timescale `sfh_field_psd_tau_myr` differs by an order of magnitude between paths (82 vs 6 Myr). Validate per-problem before swapping. See `docs/dev/benchmarks/2026-04-17_native_vs_nifty.md`
- Use `.shape[0]` instead of `len()` on JAX arrays to avoid `ConcretizationTypeError` under JIT
- Use tolerance comparison (`abs(x - default) < 1e-6`) not `==` for float equality on traced values
- IGM `igm_transmission(wave_obs, z)` takes **observed-frame** wavelengths (not rest-frame)
- Dust emission templates auto-load from `data/`; analytic fallbacks are NOT suitable for science
- AGN torus in `torus.py` are **toy models** — use SKIRTOR for science
- `agn_torus_frac`: do NOT auto-derive from `cos(theta_torus)` in forward pass (gradient discontinuity)
- Inference internals use `mode="_traceable"` (safe inside JIT). User-facing defaults to `mode="auto"`
- **One NUTS fit per notebook process.** Each warmup peaks at 3–6 GB (dense mass-matrix vmap compile). Multi-fit notebooks need `dense_mass=False` or `mcmc_hmc`. See `docs/dev/notebook_orchestration_oom.md`
- **Subagent rejection ≠ child kill.** A rejected subagent's `python notebook.py` keeps running. After rejecting, run `ps -axo pid,rss,comm | grep python` and `kill -9` zombies

## Testing

Every code change MUST include tests. Test organization:
- `tests/unit/` — fast, no SSP data needed
- `tests/integration/` — needs `data/ssp_*.h5`, skips if missing
- `tests/crossval/` — against bagpipes/FSPS, excluded from default runs

Bug fix rule: every fix MUST cite the original paper equation and include a regression test.

## qmd search (MANDATORY before reading files)

Search qmd first using `collections: ["tengri"]` before reading any file. Fall back to Read/Glob/Grep only if qmd returns insufficient results.

## References

- `AGENTS.md` — AI agent documentation
- `HANDOFF.md` — project status and next steps
- `docs/dev/design_philosophy.md` — architecture decisions
- `docs/dev/NAMING_CONTRACT.md` — naming conventions (read before any rename/refactor)
- `docs/dev/REFACTOR.md` — refactor plan
- `docs/dev/api_migration_v0.x.md` — public-API migration table (Phase 1→6 + Part II scaffold)
- `docs/known_bugs.md` — bug tracking (all currently fixed)
- `docs/dev/notebook_orchestration_oom.md` — operational rules for OOM-safe notebook authoring (multi-fit, subagent zombies, watchdog)
- `tools/check_param_prefixes.py` — CI guard for free-parameter prefix rule (NAMING_CONTRACT §3.2)
