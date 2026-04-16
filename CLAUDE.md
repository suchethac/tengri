# Claude Code Instructions for tengri

## Project overview

Differentiable SED fitting code in JAX. Models galaxy star formation histories as IFT correlated fields with PSD-governed burstiness priors. Uses DSPS for differentiable stellar population synthesis.

**Code name:** `tengri` (working name, final TBD).
**Paper draft:** `~/writing-workspace/projects/tengri/`
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

## Package structure

Layout: `parameters/ -> components/ -> forward/ -> observation/ -> inference/ -> analysis/ -> runtime/ + utils/`. Public API re-exported at `src/tengri/__init__.py`.

Key directories:
- `parameters/` — Parameters class, priors, param translation
- `components/` — SED physics: sfh/, sps/, dust/, nebular/, agn/, igm/, radio/, xray/
- `forward/` — SEDModel, pipeline, fused kernels, precompute protocol+registry
- `observation/` — photometry, spectroscopy, filters, noise, emission lines, mock
- `inference/` — fitter, posterior, all inference methods (vi, mcmc, nss, map, etc.)
- `analysis/` — diagnostics, plotting, simulate
- `runtime/` — settings, exceptions, display, deprecation
- `utils/` — cosmology, conversions, interpolation, physics_constants

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

## Critical gotchas

- `jax.random.fold_in(key, hash(string))` overflows uint32. Use `abs(hash(x)) % (2**31)`
- Never create `Model`/`ParamSpec` inside a JAX gradient tape (traced values fail in `__init__`)
- JAX Metal (Apple GPU) causes test failures. Use `JAX_PLATFORMS=cpu` for reliable results
- Ray Tracing: step_size=0.05 for D~137; sharp viability cliff at ~0.06 (acceptance drops to 0%)
- NIFTy geoVI: use 4-12 samples per KL iteration, not 80
- Use `.shape[0]` instead of `len()` on JAX arrays to avoid `ConcretizationTypeError` under JIT
- Use tolerance comparison (`abs(x - default) < 1e-6`) not `==` for float equality on traced values
- IGM `igm_transmission(wave_obs, z)` takes **observed-frame** wavelengths (not rest-frame)
- Dust emission templates auto-load from `data/`; analytic fallbacks are NOT suitable for science
- AGN torus in `torus.py` are **toy models** — use SKIRTOR for science
- `agn_torus_frac`: do NOT auto-derive from `cos(theta_torus)` in forward pass (gradient discontinuity)
- Inference internals use `mode="_traceable"` (safe inside JIT). User-facing defaults to `mode="auto"`

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
- `docs/known_bugs.md` — bug tracking (all currently fixed)
