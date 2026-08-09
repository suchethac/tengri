# AI Agent Guide for tengri

> This document helps AI agents (Claude, GPT, Copilot, Cursor, etc.) understand and work with the `tengri` codebase effectively. If you are an AI assistant helping a user with this package, read this first.
>
> **Forward-looking architecture (the `ForwardModel` outer shell,
> `SEDModel` / `SpatialModel` / `SpatialSEDModel` sub-models,
> multi-population, the joint spec-phot motivation) lives in
> [`docs/dev/archive/forward-model-architecture.md`](docs/dev/archive/forward-model-architecture.md).**
> The architecture overview directly below this banner is the *current*
> shape of the codebase, which is mid-migration toward that target.

## What this package does

`tengri` is a **differentiable galaxy SED (Spectral Energy Distribution) fitting code** written in JAX. It models star formation histories using Information Field Theory — treating the SFH as a continuous correlated field governed by a power spectral density (PSD).

**In plain terms:** Given galaxy observations (photometry or spectra), infer the galaxy's star formation history, dust properties, and metallicity using Bayesian inference with gradient-based samplers.

## Architecture overview

```
Parameters (latent xi + physical params)
    │
    ├─► PSD model (psd_models.py)          → amplitude operator sqrt(P)
    ├─► GP generation (gp_sfh.py)          → x(t) = IFFT(sqrt(P) * xi)
    ├─► Mean SFH (mean_sfh.py)             → SFR_mean(t) = double power law
    │       │
    │       ▼
    │   Full SFR(t) = SFR_mean * exp(x - K(0)/2)
    │       │
    ├─► SPS integral (dsps_wrapper.py)     → intrinsic SED [erg/s/Hz]
    ├─► Dust attenuation (attenuation.py)  → attenuated SED [erg/s/Hz]
    ├─► Nebular emission (nebular/)        → +nebular continuum + lines [erg/s/Hz]
    ├─► AGN (agn/)                         → +AGN disc/torus/BLR/NLR [erg/s/Hz]
    ├─► Radio (radio.py)                   → +synchrotron/free-free [erg/s/Hz]
    ├─► X-ray (xray.py)                    → +XRB/corona [erg/s/Hz]
    ├─► Dust emission (dust/emission.py)   → +IR emission [erg/s/Hz]
    │       │
    │       ▼
    └─► Observables
        ├─► Photometry (photometry.py)     → flux per filter band [erg/s/cm²/Hz]
        └─► Spectroscopy (spectroscopy.py) → flux per wavelength pixel [erg/s/cm²/Å]
```

**All SED components return erg/s/Hz (CGS).** The CSP assembly multiplies SSP templates (Lsun/Hz/Msun) by `LSUN_ERG_PER_S` to produce erg/s/Hz. All multiwavelength components (AGN, radio, X-ray, nebular) were standardized to erg/s/Hz in the 2026-04-08 unit refactor.

**All operations are pure JAX functions.** The entire pipeline is JIT-compilable, differentiable, and vmap-able.

## Component migration (Phase II-2) — active refactor

> **Read this before touching `forward/` or adding any physics block.**

The forward model is mid-migration from the monolithic `SEDModel` class onto the `SEDComponent` Protocol scaffold in `src/tengri/core/component.py`. Phase II-1 has shipped: the Protocol, `PipelineState`, and two reference adapters (`components/radio/component.py`, `components/igm/component.py`). The current effort (Phase II-2) migrates the remaining 6 physics blocks one PR at a time.

**Source of truth:** `docs/dev/20260404-refactor.md`. Read this before producing any plan that touches the forward path.

**Canonical adapter to copy:** `src/tengri/components/radio/component.py`. Mirror its method order, docstring sections, and `state.derived` publish/read idiom. Do not deviate without a one-line `Notes` justification.

**Things you must not reinvent:**
- The parameter registry. Parameters are already domain-organized in `src/tengri/parameters/_param_defs.py` (`_DUST_EMISSION_PARAMS`, `_AGN_PARAMS`, `_RADIO_PARAMS`, …). A component's `declared_parameters()` references these; it does not duplicate them.
- The precompute system. `src/tengri/forward/precompute/registry.py` is the existing registry (~30 entries). New components consume it, they do not replace it.
- Protocols. The only types added in this migration are concrete `SEDComponent` subclasses. No new Protocols, no new abstractions.

**Entropy budget enforced per PR:**
1. Net-negative LOC: each PR deletes ≥ as many lines from `sed_model.py` + `pipeline.py` as it adds elsewhere.
2. Component file ≤ 400 LOC (docstrings included). Split the *physics*, not the *file*.
3. No feature flag. The monolith branch for the migrated block is deleted in the same PR that introduces the component.
4. No placeholders, no `TODO`, no commented-out code, no stub methods.
5. Structural conformance to the previous merged component.
6. CHANGELOG entry must list the monolith lines deleted; an empty deletion list means the PR is not entropy-reducing.

**The `Parameters.from_components(...)` builder is deferred** until ≥5 components have landed. Until then, `_param_defs.py` continues feeding `Parameters` unchanged. Do not pre-build the component-driven parameter constructor.

## Key files to read

**Package layout rewritten 2026-04-15, further refined 2026-04-21** (`core/` dissolved; `models/` → `components/`;
`models/observation/` promoted to top-level `observation/`; `diagnostics/` +
`plotting.py` + `simulate.py` moved under `analysis/`; `distributions.py` →
`parameters/priors.py`; `runtime/` → `config/` for settings/exceptions/display/deprecation;
`mock.py` moved from `observation/` → `analysis/`; ``_kernels/`` was deleted
entirely in Phase 6 (PR #135) — the orchestrator path through
``predict_observables_jit`` replaced it. ``KernelStrategy`` and
``NoCompatibleKernelError`` are now ImportError stubs.
Public API (`from tengri import ...`) unchanged.

| File | Purpose | When to read |
|------|---------|--------------|
| `src/tengri/forward/sed_model.py` | High-level SEDModel class (2957L, split deferred) | Understanding the forward model |
| `src/tengri/forward/orchestrator.py` | The SEDComponent chain runner — replaced the kernel adapter family in Phase 6 (PR #135, 2026-05-20). Calls each component's `apply(state, params, ssp_data=...)` in sequence. | Core forward model |
| `src/tengri/forward/precompute/` | Precompute Protocol + registry + algorithm (`protocol.py`, `registry.py`, `grid.py`, `templates.py`) | Extending precompute |
| `src/tengri/parameters/parameters.py` | Parameters class (canonical, was ParamSpec) | Parameter handling |
| `src/tengri/parameters/priors.py` | Uniform / Gaussian / LogUniform / Fixed | Prior construction |
| `src/tengri/components/sfh/gp_sfh.py` | GP generation from PSD | Core IFT machinery |
| `src/tengri/components/sfh/psd_models.py` | PSD definitions (DRW, Matern) | Understanding the burstiness prior |
| `src/tengri/components/dust/attenuation.py` | Two-component dust attenuation (charlot_fall.py removed) | Dust attenuation |
| `src/tengri/components/dust/emission.py` | Dust IR emission (DL07/DL14/Dale/Casey/…) — 2459L, split deferred | Dust IR models |
| `src/tengri/components/dust/dust_emission_precompute.py` | Protocol adapter for template-based dust emission | Dust IR precompute |
| `src/tengri/components/sps/dsps_wrapper.py` | DSPS CSP integral | SPS integration |
| `src/tengri/components/sps/precompute.py` | SSP precompute Protocol (reference implementation) | SPS precompute |
| `src/tengri/components/agn/disc.py` | AGN disc models (K&D 3-zone, powerlaw, multicolor) | AGN |
| `src/tengri/components/agn/unified.py` | Unified AGN (disc + torus + BLR/NLR + polar dust) | AGN assembly |
| `src/tengri/components/agn/skirtor_precompute.py` | Protocol adapter for SKIRTOR torus | AGN precompute |
| `src/tengri/components/agn/kd_precompute.py` | K&D preintegration (Protocol stub; full wiring deferred) | AGN K&D |
| `src/tengri/components/nebular/cue.py` | Cue NN nebular emulator | Nebular emission |
| `src/tengri/components/nebular/cloudy_precompute.py` | CLOUDY Protocol marker (auto-collapse inside CloudyGridBackend) | Nebular precompute |
| `src/tengri/components/radio/radio.py` | Radio emission (synchrotron, free-free, AGN) | Radio |
| `src/tengri/components/xray/xray.py` | X-ray emission (XRB, AGN corona) | X-ray |
| `src/tengri/observation/` | Photometry, spectroscopy, filters, line_list, noise | Observation layer |
| `src/tengri/analysis/` | Diagnostics, plotting, simulation, mock data generation | Post-fit analysis |
| `src/tengri/config/` | DustConfig/NebularConfig/SFHConfig/ModelConfig, exceptions, display, deprecation | Cross-cutting plumbing |
| `tests/conftest.py` | Test fixtures, grid setup | Understanding test patterns |

## Parameter dictionary convention

The ``SEDModel`` class uses **public parameter names** (via ``Parameters``).
For a DPL + GP field model:

```python
params = {
    # Latent GP variables (standardized: xi ~ N(0, I))
    "sfh_field_xi": jnp.ndarray,          # shape (n_grid,)

    # PSD parameters (DRW)
    "sfh_field_psd_sigma": float,          # PSD amplitude (0.01-3.0)
    "sfh_field_psd_tau_myr": float,        # damping timescale in Myr (10-500)

    # Mean SFH (double power law)
    "sfh_dpl_alpha": float,                # falling slope (0.1-5.0)
    "sfh_dpl_beta": float,                 # rising slope (0.1-3.0)
    "sfh_dpl_tau_gyr": float,              # turnover time in Gyr (0.1-12)
    "sfh_dpl_log_total_mass": float,         # log10 peak SFR (Msun/yr)

    # Metallicity
    "met_logzsol": float,                  # log10(Z/Zsun) (-2.0 to 0.2)

    # Dust (two-component attenuation)
    "dust_tau_bc": float,                  # birth cloud optical depth (0-4)
    "dust_tau_diff": float,                # diffuse ISM optical depth (0-3)
    "dust_slope": float,                   # power-law index (typically -0.7)

    # Redshift
    "redshift": float,
}
```

## Writing documentation (MANDATORY for all new code)

**Read `docs/dev/docstring-standard.md` for the full template and rules.** Below is the
quick reference. Every function or class you add must have a docstring at the appropriate
tier level.

### Tier system

| Tier | Who | Mandatory sections |
|------|-----|--------------------|
| 1 — Public API | `__init__.py` exports (SEDModel, Fitter, Parameters, Posterior, …) | Summary, Parameters, Returns, Raises, Notes (JIT flag), References, Examples |
| 2 — Scientific functions | `components/`, `forward/`, `observation/` | Summary, Parameters, Returns, Notes (JIT flag + equations + approximation flags), References |
| 3 — Utilities | `utils/`, `config/`, `analysis/` | Summary, Parameters, Returns |
| 4 — Private helpers | `_`-prefixed | One-sentence summary; Parameters if non-obvious |

### Equations

Use RST `.. math::` directive — it renders with MathJax in the Sphinx docs:

```rst
Notes
-----
The damped random walk power spectral density is:

.. math::

    P(\omega) = \frac{\sigma^2 \,\tau}{1 + (\tau\,\omega)^2}

where :math:`\sigma` is the PSD amplitude [dimensionless], :math:`\tau` is the
damping timescale [yr], and :math:`\omega` is angular frequency [rad/yr].
This is Eq. 3 of Author+2026 [1]_.
```

**Before writing any equation: verify against the original paper.** If the implementation
is an approximation, flag it explicitly:
```
**Approximation**: valid for :math:`\tau \gtrsim 10\,\Delta t`. Breaks down at small tau.
```

### Citations

Use numbered References with exact paper titles, arXiv ID, and DOI:

```rst
References
----------
.. [1] S. Charlot and S. M. Fall, "A Simple Model for the Absorption of Starlight by
   Dust in Galaxies," ApJ, 539, 718 (2000).
   https://doi.org/10.1086/309250
.. [2] B. D. Johnson et al., "Prospector: Stellar Population Inference from Spectra
   and SEDs," ApJS, 254, 22 (2021). arXiv:2012.01426.
   https://doi.org/10.3847/1538-4365/abef67
```

Never write citations from memory — verify exact titles and identifiers against authoritative sources.

### Upstream code credit

If your code implements a model defined in another package, say so in Notes:

```
**Upstream**: Follows the Prospector ``transforms.py`` implementation
(Johnson et al. 2021 [2]_), implemented in JAX.
```

Include the upstream code's paper in References.

### JIT/grad compatibility

For every function in `components/` or `forward/`, state in Notes:

```
**JIT-compatible**: yes — all operations use ``jnp`` primitives.
**Gradient-safe**: yes.
```

Or, if not JIT-safe:

```
This function is not compatible with :func:`jax.jit` because [reason].
```

### Units and shapes

- Units always in brackets: `[erg/s/Hz]`, `[yr]`, `[Msun/yr]`, `[Angstrom]`
- Input arrays: `array_like, shape (n_wave,)`
- Output arrays: `ndarray, shape (n_wave,)`

---

## Common tasks an agent might need to do

### Generate a mock galaxy SED

```python
from tengri import Model, ParamSpec, Uniform, load_ssp_data, load_filter_set

ssp = load_ssp_data("path/to/ssp_templates.h5")
filters = load_filter_set(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
spec = ParamSpec(
    mean_sfh_type=["dpl", "field"],
    sfh_dpl_alpha=Uniform(0.5, 3.0),
    sfh_dpl_beta=Uniform(0.3, 2.0),
    sfh_dpl_tau_gyr=Uniform(0.5, 10.0),
    sfh_dpl_log_total_mass=10.0, 2),
    sfh_field_psd_sigma=Uniform(0.01, 1.0),
    sfh_field_psd_tau_myr=Uniform(10, 500),
    redshift=0.1,
)
model = Model(spec, ssp, filters=filters)
params = spec.sample(jax.random.PRNGKey(0))
mock = model.mock(params, snr=20.0, key=jax.random.PRNGKey(1))
```

### Compute gradients

```python
# Gradient of any scalar loss w.r.t. any parameter
loss_fn = lambda p: -gaussian_log_likelihood(model.predict_photometry(p), data, noise)
grads = jax.grad(loss_fn)(params)
```

### Add a new dust model

1. Create `src/tengri/components/dust/my_model.py`
2. Implement a function with signature: `(wavelength, age_grid, **params) -> attenuation_factor`
3. The function must be pure JAX (jnp operations only, no side effects)
4. Add tests in `tests/unit/test_dust.py`
5. Register in the dust model registry if replacing the default

### Add a new PSD model

1. Add function to `src/tengri/components/sfh/psd_models.py`
2. Signature: `(omega, **params) -> P(omega)` where omega is angular frequency
3. Must be JIT-compatible and have well-defined gradients
4. Add corresponding `compute_sqrt_power_*` function in `gp_sfh.py`
5. Add tests verifying the integral equals the expected variance

### Add a new template-based precompute-enabled component

See the Precompute Protocol at `src/tengri/forward/precompute/protocol.py`.

1. Add the forward-evaluation function at `src/tengri/components/<comp>/<comp>.py`
   (pure JAX).
2. Create `src/tengri/components/<comp>/<comp>_precompute.py` exposing the Protocol
   surface: `AXIS_PARAMS: tuple[str, ...]` (or `dict[str, tuple[str, ...]]` for
   multi-variant), `precompute(filter_waves, filter_trans, redshift, parameters,
   **kwargs)`, `build_lookup(preint, **kwargs)`. Auto-collapse via
   `slice_fixed_axes` whenever any `AXIS_PARAMS` entry is Fixed in `parameters`.
3. Register in `src/tengri/forward/precompute/registry.py` — one new entry in
   `_REGISTRY`. That's the full extension surface; `SEDModel` does not need
   editing.
4. Add tests in `tests/unit/test_precompute_protocol.py` (the parametrized tests
   auto-cover your new component via the registry).

## Dependencies and their roles

| Package | Import | Role |
|---------|--------|------|
| `jax` | Core | Autodiff, JIT compilation, GPU support |
| `dsps` | `from dsps import load_ssp_templates` | Differentiable SPS, SSP template loading |
| `nifty8.re` (optional) | `import nifty.re as jft` | geoVI/MGVI variational inference |
| `blackjax` (optional) | `import blackjax` | NUTS/HMC sampling |
| `optax` (optional) | `import optax` | Gradient-based optimization (MAP) |

## Linting and formatting

**Ruff** is the project linter and formatter. Configuration lives in `pyproject.toml` under `[tool.ruff]`.

```bash
ruff check src/ tests/              # lint — MUST pass with zero errors
ruff format --check src/ tests/     # format — MUST pass
ruff check --fix src/ tests/        # auto-fix safe violations
ruff format src/ tests/             # auto-format all files
```

**Before writing or modifying any Python code**, ensure your changes pass both `ruff check` and `ruff format --check`. Run these after every code change. Key rules enforced:

- **F**: unused imports/variables (keep imports clean)
- **E/W**: pycodestyle basics (99-char line limit)
- **I**: import sorting (stdlib → third-party → first-party `tengri`)
- **UP**: Python 3.10+ syntax (use `X | None` not `Optional[X]`)
- **B**: bugbear patterns (`raise ... from None` in except, no loop-var capture in closures)
- **SIM**: simplifiable constructs
- **RUF**: Ruff-specific (sorted `__all__`, no unused unpacked vars)

**Allowed exceptions** (configured in pyproject.toml):
- `E402` ignored: `jax.config.update()` must run before JAX imports
- `E741` ignored: single-letter variables (`l`, `I`) common in scientific code
- Greek letters (σ, ξ, θ) allowed in docstrings/comments
- `__init__.py` files: `F401` (unused imports) ignored for re-exports
- `tests/`: `F841` (unused variables) ignored for fixtures
- `notebooks/`, `analysis/`: relaxed rules for exploratory code

## Convergence Diagnostics (mandatory)

Every inference result must be checked for convergence before trusting posteriors.
Use `convergence_check()` or `convergence_table()` from `notebooks/_plot_style.py`.

**Industry-standard thresholds** (Vehtari et al. 2021):
- **ESS**: > 100 per parameter, > 400 total for reliable summaries
- **Divergences** (NUTS): 0 ideal; > 5% = posterior unreliable
- **RT acceptance**: 30–70%; > 90% = chain barely moving
- **NUTS acceptance**: ~80%

**Known issues**: `dust_tau_bc`, `dust_tau_diff`, `met_logzsol` have low ESS due to
age-dust-metallicity degeneracy — a physical limitation, not a sampler bug.

**RT tuning for stochastic models (D~137)**: `step_size=0.05, n_leapfrog_steps=50`.
Sharp viability cliff at step_size~0.06; compensate with more leapfrog steps.

## Testing

```bash
pytest tests/ -v                          # all tests
pytest tests/unit/test_psd_models.py -v   # specific module
pytest tests/ --cov=src/tengri           # with coverage
```

All tests use `jax.config.update("jax_enable_x64", True)` for numerical precision.

## Code conventions

1. **Pure functions**: All model components are stateless pure JAX functions. No global state.
2. **Immutability**: Never mutate arrays. Use `jnp.ndarray.at[].set()` for updates.
3. **Units**: Times in **years** internally. Wavelengths in **Angstrom**. SFR in **Msun/yr**. SED luminosity (L_ν) in **erg/s/Hz** throughout — all component functions (AGN, radio, X-ray, nebular, CSP) return erg/s/Hz. The `agn_log_lbol` parameter is the one exception: it is log10(L_bol / L_sun) at the API boundary, with conversion to erg/s inside each function.
4. **Grid**: 256-point uniform grid in log10(age/yr) from 6.0 to 10.14.
5. **Naming**: `snake_case` everywhere. PSD params use `sigma_ps`, `tau_ps` (not sigma_PS).
6. **Docstrings**: Numpydoc format with Parameters/Returns sections.
7. **Type hints**: Use `X | None` (PEP 604), not `Optional[X]`. Ruff enforces this (UP007/RUF013).

## Notebooks

Notebooks are **jupytext percent-format `.py` files** in `notebooks/`. These are the source of truth.

### How to edit a notebook

1. Open the `.py` file with Read/Edit tools — it's plain Python with `# %%` cell markers
2. Make your changes directly
3. Run `jupytext --sync notebooks/*.py` to regenerate `.ipynb` if needed

### Cell format

```python
# %% [markdown]
# # Section Title
#
# Some explanation with $\LaTeX$ math.

# %%
import jax
import jax.numpy as jnp
result = jnp.array([1, 2, 3])

# %% [markdown]
# Another markdown cell.

# %%
# Another code cell
print(result)
```

### DO NOT

- Edit `.ipynb` files directly (they are gitignored and generated)
- Use the old `_build_nb*.py` / `_nb_helper.py` system (deleted)
- Create new notebooks as `.ipynb` — always create as `.py` in percent format

## What IS implemented (current state as of 2026-04-08)

All major components are implemented and tested:

- **SFH models**: double power law, tsnorm, continuity, dirichlet, GP field (IFT correlated field with DRW/Matern PSD)
- **Stellar populations**: DSPS CSP integral, MILES/C3K SSP templates, alpha-enhancement
- **Dust attenuation**: Charlot & Fall two-component (`two_component_dust`), Calzetti, Reddy, SMC/LMC (Pei 1992), Narayanan+2018 mass-dependent, WG00 geometries
- **Dust emission**: Draine & Li 2007 (tabulated), Dale+2014 (tabulated), Casey 2012 MBB+power law
- **Nebular emission**: BakedIn (H-line scaling), CLOUDY grid interpolation, Cue emulator (NN), MAPPINGS shock+precursor, Feltre+2016 (stub)
- **AGN**: K&D 3-zone disc (powerlaw, multicolor, kubota_done_3zone, ADAF), SKIRTOR torus (skirtor2016_torus, skirtor_analytic), BLR/NLR (unified_nlr_blr with polar dust), QSOgen, nthcomp warm Comptonization (HDF5 templates)
- **Radio**: synchrotron + free-free + AGN compact (Condon+92, Krolik & Chen+91)
- **X-ray**: XRB (HMXB+LMXB), AGN corona (α_ox, nthcomp power law)
- **Observation layer**: photometry (broadband, precomputed), spectroscopy (wavelength grid, emission line fitting/marginalization), calibration polynomial marginalization
- **Inference**: MAP (Adam/AdamW/SGD), Ray Tracing, NUTS, geoVI, MGVI, ESS, NSS, Laplace, Pathfinder, vi_native, vi_native_linear — 13 canonical methods
- **Hierarchical**: HierarchicalFitter with NIFTy CorrelatedFieldMaker (shared PSD)
- **Docs**: Sphinx + Furo site with Sphinx Gallery examples, GitHub Pages

## What is NOT yet implemented

- `eline_mode="fitted"` full posterior — line amplitudes as free params works for MAP/VI but broad component model still has rough edges
- Feltre+2016 NLR backend (`NotImplementedError` stub in `nlr.py`)
- GPU benchmarks (all timing numbers are CPU)
- ADAF model equations verified against Mahadevan (1997) — flagged for rewrite

### Partially implemented — scheduled follow-up (from 2026-04-15 restructure)

- ~~**Precompute Protocol rewiring in SEDModel.**~~ Done, and the entry point is
  gone. `docs/known_bugs.md` has recorded IMP-06 as FIXED since 2026-04-15 (the
  hardcoded switch collapsed to ~25 lines); this bullet went stale then and kept
  describing a "200-line hardcoded switch" that no longer existed. The method it
  named, `_precompute_dust_ir_photometry`, was subsequently deleted outright as
  dead code (#1403) — nothing had called it. Dust-IR precompute now lives in
  `components/dust/dust_emission_precompute.py` (template models) and
  `dust_analytic_precompute.py` (analytic), with models mapped to their module
  by `forward/precompute/registry.py`.
- **K&D 3-zone disc Protocol wiring.** `components/agn/kd_precompute.py` exposes
  `AXIS_PARAMS = ()` and `precompute()` raises `NotImplementedError`. Actual
  K&D preintegration runs via the original `kubota_done_disc_preintegrated`
  call path from `forward/sed_model.py`. Full Protocol wiring requires either
  extending the Protocol to accept a custom `KDPreintegratedData` dataclass or
  refactoring K&D to use the generic `PreintegratedGrid` shape. Tracked as
  `IMP-07`.
- **Auto-collapse-on-Fixed gaps pre-restructure.** Before the refactor, only
  SPS + CLOUDY auto-collapsed on Fixed parameters; DL07/SKIRTOR did not. The
  Protocol adapters now add this uniformly, but the auto-collapse runs only
  when callers use the new `precompute()` entry points. Legacy callers of
  `precompute_dl07_photometry` / `precompute_skirtor_photometry` still skip
  auto-collapse until SEDModel switches to the registry.
- **Taylor correction in template adapters.** `preintegrate_grid(taylor=True)`
  is implemented but opt-in. SSP photometry uses it by default (`ssp_phot_moment`
  in `components/sps/precompute.py`). Template adapters (DL07/Dale/DL14/…) and
  SKIRTOR still use zeroth-order only. Enabling Taylor across all template
  adapters should be a future benchmark study.
- **Large-file splits.** `forward/_kernels/` was deleted entirely in Phase 6
  (PR #135) — the orchestrator path through ``predict_observables_jit``
  replaces the kernel adapter family. `forward/sed_model.py` was ~6000L
  pre-Phase-6 and is now ~5050L; the planned split by fusion strategy
  and by lifecycle (class / factory / fit / predict / summary / precompute)
  is deferred. `analysis/plotting/all.py` (1156L) and
  `components/dust/emission.py` (2459L) splits likewise deferred.
