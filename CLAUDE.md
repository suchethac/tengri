# Claude Code Instructions for tengri

## Project overview

Differentiable SED fitting code in JAX. Models galaxy star formation histories as IFT correlated fields with PSD-governed burstiness priors. Builds on DSPS for the cosmology, metallicity weights, surviving-mass fractions and the SSP grid format; the composite-stellar-population integral on the default path is tengri's own CIC kernel (`sfh={'age_kernel': 'cic'}`), with DSPS's histogram kernel available as `'dsps'` for cross-code parity (#1727).

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

# Tests — the default run is the PR-gating fast tier, in parallel (~7.3k tests)
.venv/bin/pytest tests/ -q

# The heavy trees (tests/inference, tests/integration) are auto-marked `slow`
# in tests/conftest.py and deselected by default: 15% of the tests, but the
# overwhelming majority of the wall clock (two test_user_scenarios tests alone
# run 24 min and 18 min). They run in CI as a schedule/label-gated job.
.venv/bin/pytest tests/ -q -m slow            # the heavy trees only
.venv/bin/pytest tests/ -q -m 'not crossval'  # everything (fast + slow)

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
export TENGRI_JAX_CACHE_MAX_GB=8                      # size cap; 0 = unbounded
```

**The cache is capped at 8 GiB** (`jax_compilation_cache_max_size`, so JAX
evicts). Before #1507 nothing passed a cap — `import tengri` left it at JAX's
`-1` (unlimited) — and combined with `min_compile_time_secs=0.05` below, which
deliberately persists every per-filter micro-kernel, the directory was measured
at **141 GB** on a 48 GB machine. `tengri.clear_cache()` now returns the bytes
it reclaimed.

A sibling cache at `~/.cache/tengri_precomp` persists the WavePrecomp
photometry z-table (the dominant numpy cost of a free-redshift
`SEDModel.build` — the JAX cache cannot help it). Content-hashed on
(SSP grid, filters, z grid, quadrature flags); the first build of a
combination pays the quadrature, later builds load the npz in
milliseconds. `TENGRI_PRECOMP_CACHE_DIR` / `TENGRI_DISABLE_PRECOMP_CACHE`
mirror the JAX-cache knobs. The pytest suite disables it globally in
`tests/conftest.py` (hermeticity); its contract tests opt back in.

After upgrading JAX (`pip install -U jax`), wipe stale entries:

```python
import tengri; tengri.clear_cache()
```

Default `min_compile_time_secs=0.05` persists per-filter
`compute_flux_density` kernels and other ~150–250 ms component
precompute compiles. Threshold history: 5.0 (≤ 2026-05-04, skipped
the orchestrator chain), 0.5 (≤ 2026-05-22, missed per-filter
micro-compiles), 0.05 (current). See
`docs/performance/compilation.md` for full details.

## Naming contract (MANDATORY)

**Read `docs/dev/NAMING_CONTRACT.md` before writing any new code, renames, or refactors.**

Canonical names: `SEDModel`, `Parameters`, `Spectroscopy`, `NoiseModel`, `LineList`, `PopulationFitter`.
Deprecated aliases (never use in new code): `Model`, `ParamSpec`, `SpectroscopyConfig`, `NoiseConfig`, `LineCatalog`, `HierarchicalFitter`.

**Spelling (NAMING_CONTRACT §10):** all identifiers and prose use **American English** (`color`, `normalize`, `catalog`, `center`, `finalize`, `marginalized` — never `colour`, `normalise`, `catalogue`, …). Exception: external data-contract keys keep their upstream spelling (e.g. the Synthesizer HDF5 keys `ionisation_parameter`, `log10_specific_ionising_luminosity`).

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
- Reference codes MUST be credited in Notes: *"Implements the same model as Prospector (Johnson et al. 2021 [N]_); validated against it."* Never describe tengri code as ported/copied/adapted from another codebase: implementations are independent. External template/SSP **data** files are "repackaged", with attribution.
- JIT/grad/vmap compatibility MUST be stated in Notes for all `components/` and `forward/` functions.
- VERIFY equations against the original paper before writing — do not rely on memory or other code.

**Where docs go.** `docs/` serves two audiences and `docs/conf.py` publishes only one of
them: the `exclude_patterns` list there keeps the whole contributor-only side out of the
built site. Check that list before adding a page — being excluded is the default for
anything a user is not meant to read.

- Agent-authored design plans and specs → `docs/internal/plans/` and
  `docs/internal/specs/`. The brainstorming skill defaults to writing under a
  docs/superpowers/ directory instead; this line is the project preference that overrides
  that default, and the old tree was moved so a single `internal` exclude covers every
  contributor-only page.
- Contributor handbook (naming, style, architecture narratives) → `docs/dev/`.
- Anything a user should read → the published tree, plus a toctree entry in
  `docs/index.md`. A page that is neither excluded nor in a toctree builds as an orphan
  and emits a warning.

## Package structure

**For "where do I look to edit X?" — see [`docs/dev/where-things-live.md`](docs/dev/where-things-live.md).**

Layout: `parameters/ -> components/ -> forward/ -> observation/ -> inference/ -> analysis/ -> config/ + utils/`. Public API re-exported at `src/tengri/__init__.py`.

Key directories:
- `protocols/` — Protocols (`SEDComponent`, `ObservationModel`, `Likelihood`, `DerivedState`) + `SEDModelComponent` contract; consumed by the forward pipeline (ADR-0009/0011/0019)
- `parameters/` — Parameters class, priors, param translation
- `components/` — SED physics: sfh/, sps/, dust/, nebular/, agn/, igm/, radio/, xray/
- `forward/` — SEDModel, pipeline, component_factory (the `_REGISTRY` dispatch seam), precompute/ (protocol+registry)
- `observation/` — photometry, spectroscopy, filters, noise, emission lines
- `inference/` — fitter, posterior, all inference methods (vi, mcmc, nss, map, etc.)
- `analysis/` — diagnostics, plotting, simulate, mock
- `config/` — config dataclasses (internal lowering artifacts — build models via `SEDModel.build` grammar; top-level `tengri.DustConfig` etc. emit DeprecationWarning since 2026-07), exceptions, display, deprecation
- `utils/` — cosmology, conversions, interpolation, physics_constants
- `cosmology/` — re-exports from `utils/cosmology` (canonical user-facing path; added 2026-05)
- `units/` — re-exports F_nu/L_nu/AB-magnitude conversions (canonical user-facing path; added 2026-05)
- `plot/` — re-exports plotting helpers from `analysis/plotting` (canonical user-facing path; added 2026-05)
- `_deprecated.py` — `deprecated_alias()` / `deprecated_attribute()` shims (added 2026-05)

## Model construction API (CURRENT)

**Canonical narrative: [`docs/dev/model-construction.md`](docs/dev/model-construction.md)** — the build path, the single `_REGISTRY` dispatch, and the one add-a-model recipe in one place.

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
    sfh={'type': 'dpl', 'all_params': FREE, 'beta': Uniform(1, 3)},
    dust={'type': 'two_component', 'law_bc': 'calzetti', 'all_params': FIXED,
          'tau_bc': 0.5, 'emission': {'type': 'dale2014', 'all_params': FIXED}},
    neb={'type': 'cue', 'all_params': FIXED},
    redshift=Fixed(0.05),
)
model.spec.summary()    # provenance-tagged: [user] / [all_params FREE] / [all_params FIXED] / [default]
groups = model.spec.to_groups()    # round-trip for inspection/editing

# Or with builder factories (autocomplete-friendly; SFH only as of Phase II-3.3)
from tengri import builders
model = SEDModel.build(
    ssp_data=ssp, observation=obs,
    sfh=builders.sfh.dpl(_=FREE, beta=Uniform(1, 3)),  # ← IDE sees alpha, beta, tau_gyr, log_total_mass
    dust={'type': 'two_component', 'law_bc': 'calzetti', 'all_params': FIXED},
    neb={'type': 'cue', 'all_params': FIXED},
)
```

- Grammar: each group dict accepts `'type'` (structural choice), `'all_params'`
  wildcard (`FREE`/`FIXED`; default `FIXED`; the `'*'` synonym is still accepted
  but slated for deprecation), and per-parameter short-form overrides (e.g.
  `'beta'` inside the sfh group resolves to `sfh_dpl_beta`).
- **`met` is the metallicity group**, parallel to `sfh`: `met={'type': 'table'}`,
  `met={'type': 'ramp', 'logzsol_0': ...}`, `met={'logzsol': Uniform(...)}`.
  The `stellar={'met_mode': ...}` spelling of #311 is **gone** (#1720) — it was
  the one group naming its structural key something other than `'type'`, and the
  only group whose name did not match what it configured, which is how #1677
  shipped advice the grammar refused. Passing `stellar=` raises with the
  translation. `tengri.list_metallicity_modes()` is the live menu.
  **The group census is derived**, not restated: `valid_groups` comes from
  `_GROUP_STRUCTURAL_KEYS`, and `SEDModel.build`'s signature is the one
  remaining hand-maintained copy — add a group to both.
- Sub-blocks: `dust.emission`, plus the six AGN composable selectors —
  `agn.disc`, `agn.torus`, `agn.nlr`, `agn.blr`, `agn.feii`, `agn.atten`
  (the deprecated `agn.lines` alias expands to an nlr/blr pair). Each nests
  as a dict with its own `'type'`, `'all_params'`, and per-param keys.
- Composable shock (#851): the top-level `shock={...}` group adds MAPPINGS V
  shock emission as a **separate additive** component that composes with any
  photoionized `neb` backend (both on at once). `shock={'norm': 'frac' |
  'lhalpha', 'abundance': ..., 'component': ..., 'frac'/'log_lhalpha'/
  'velocity'/...: prior}`. `'frac'` (default) scales the galaxy Hα (bit-exact
  with the legacy `shock_emission`); `'lhalpha'` sets an absolute
  `shock_log_lhalpha` (decoupled from the SFR — for AGN NLR/outflow shocks).
  `shock={'type':'none'}` disables. Like radio, `'all_params':FREE` is a no-op for the
  Fixed-default shock bucket — use explicit priors (`shock={'frac':
  Uniform(0,1)}`). Canonical component: `ShockNebular` (`_REGISTRY['shock']`).
  `'mappings'` is the shock group's default `type` and selects `ShockNebular`;
  the older standalone `mappings` component is gone, not merely superseded.
- AGN cross-block normalisation policy: `agn={'type': 'composable', ...,
  'norm': 'cigale_joint' | 'independent'}` (#556). `'cigale_joint'` (default)
  ties disc/torus/polar to CIGALE's single `agn_power` reference (energy-
  conserving; active only for `torus='skirtor'` + `agn_fracAGN>0`).
  `'independent'` keeps each component on its own luminosity scale (disc on
  `agn_log_lbol`, torus on `agn_power`, polar via the legacy face-on proxy) —
  the GRAHSP/AGNfitter-style bookkeeping. See `AGNSEDComponentConfig.agn_norm`.
- Sentinels (`FREE`, `FIXED`) are singletons exported from `tengri`.
- Recipes: `tengri.recipes.*` — ten curated starting points. Five general
  (`star_forming_photometry`, `quiescent_z0`, `stochastic_sfh_jwst`,
  `high_z`, `photoz`), three AGN (`agn_panchromatic`, `composable_agn`,
  `unified_agn`), and two for forward-only work (`mock_recovery_minimal`,
  `dust_demo`). Each docstring states its SSP requirement — three values,
  not two: bare-stellar, wNE (with-nebular-emission), or any.
  `tengri.list_recipes()` is the live list — do not re-enumerate them
  from memory.
- The flat-kwarg `Parameters(...)` form is the **expert escape hatch** — still
  works, still used internally, but not the recommended user-facing path.

See `docs/dev/api_migration_v0.x.md` for the full grammar reference and
`notebooks/04_building_models.py` for a worked example covering recipe usage,
variant swapping, and round-trip editing. Design plan:
`~/.claude/plans/i-feel-like-its-serene-emerson.md`.

## Prediction API (MANDATORY — read before writing ANY prediction code)

**Canonical: `docs/dev/NAMING_CONTRACT.md` §4b.** Binding on all code, docs,
notebooks, examples and agents. Violations are bugs, not style.

**Two surfaces, nothing else public:**

```python
pred = model.predict(params)          # rich + cached; ONE forward pass. Exploration.
model.predict_photometry(params)      # lean, JIT/vmap-safe. The inference hot path.
model.predict_properties(params, names=(...))   # the ONE jit/vmap surface for derived quantities
```

**Observables are uniform callables with defaults** (no `_at`/`_for`/`_on` coinages):

```python
pred.rest_sed()          # L_nu [erg/s/Hz], rest axis    | axis: pred.wave_rest
pred.rest_sed(wave)      # resampled onto YOUR rest-frame grid [Angstrom]
pred.obs_sed()           # L_nu [erg/s/Hz] STILL — obs axis + IGM | axis: pred.wave_obs
pred.obs_sed(wave_obs)   # resampled — OBSERVED-frame grid (its own frame!)
pred.photometry(filters=None, fast=False)   # F_nu [erg/s/cm2/Hz]
pred.spectrum(wave_obs=None)                # F_nu [erg/s/cm2/Hz]
pred.properties["stellar_mass"]      # or the sugar: pred.stellar_mass
```

**UNITS (§4b.3b) — `obs_sed` is NOT a flux.** "Observed" names the *frame*, not a
flux conversion. `rest_sed()` and `obs_sed()` are BOTH L_nu [erg/s/Hz]; they
differ only by the wavelength axis and IGM absorption. The cosmological dimming
`(1+z)/(4*pi*d_L^2)` is applied at the **projection** step
(`observation/redshift_kernel.py`), so only `photometry()` / `magnitudes()` /
`spectrum()` return a flux. Integrating `obs_sed()` as a flux is wrong by ~57
orders of magnitude. (The docstring claimed the opposite for a long time — it was
false. Measure, do not trust the prose.)

**Five rules that have each already caused a shipped bug:**

1. **`model.predict()` takes `params` and NOTHING else.** No `wave=`. Resampling
   lives on the accessor (`pred.rest_sed(wave)`). `model.predict(p, wave=...)`
   raises `TypeError` — and `py_compile` will not catch it.
2. **Never `params.get("redshift", 0.0)`.** A `Fixed` redshift is legitimately
   absent from `params`; the `0.0` default puts the galaxy at 10 pc — a silent
   1e17 flux error. Use `model._get_redshift(params)`. (Not `_get_dl_cm`: it
   discards an explicit override.)
3. **`state.derived[...]` is NOT `posterior.derived`.** The former is
   `ForwardState.derived`, the internal pipeline dict — **not deprecated, leave
   it alone**. Only `Posterior.derived` is deprecated (→ `posterior.properties`).
   Never grep-and-migrate a bare `.derived`.
4. **The SED arrays do not carry their axis** — use `pred.wave_rest` /
   `pred.wave_obs`. Never hand-roll `wave * (1 + z)`.
5. **`pred.rest_sed` without `()` raises.** Deliberately: a bound method coerces
   to a `dtype=object` array and would otherwise plot garbage. A public accessor
   that can be misused must fail loudly, never fail open.

Deprecated (warn + delegate; do not use, do not teach):
`predict_rest_sed`, `predict_obs_sed`, `predict_derived`, `predict_magnitudes`,
`predict_sfh_quantities`, `predict_sed_quantities`, `Posterior.derived`.

## Key conventions

- **Physical constants**: Import from `utils/physics_constants.py` — do NOT define local constant literals. Exception: `L_SUN_CUE = 3.839e33` in `cue.py` is intentional (Cue training convention, not IAU 2015).
- **Metallicity**: SSP grid is `log10(Z)` absolute, not `log10(Z/Zsun)`. Offset: `LOG10_ZSUN = -1.848` (Asplund 2009, **Zsun = 0.0142**, matches MIST). User-facing `met_logzsol` and `neb_logZ_gas` are `log10(Z/Zsun)` (param_map adds LOG10_ZSUN). **Default** `met_logzsol = 0.0` (solar — matches FSPS / Bagpipes). Per-SSP-library Zsun differs: BC03/Padova = 0.0190, PARSEC = 0.0152, BASTI = 0.0200 — see `LOG10_ZSUN_BY_LIBRARY` in `parameters/translate.py`. For bit-exact cross-code comparisons (e.g. CIGALE BC03), reason in **absolute** `log_z_abs = met_logzsol + LOG10_ZSUN` and pin that. See #412 for the audit trace. **Simulation Z(t) histories** enter via `Catalog.from_histories(..., met=, met_gas=, met_unit=)` and are validated in `inference/history_ingest.py`: the lookup clips onto `ssp_lgmet` inside JIT where nothing can raise, so ingest is the only gate — it refuses off-grid nodes by default (`on_out_of_grid='raise'`, unlike the build-time scalar check of #442 which only warns) and flags a metal mass fraction read as `logzsol`, which is *in-grid* and so invisible to any range check (#1677). **Stellar and gas-phase Z are separate knobs**: `met=` is a per-age history selecting SSP templates, `met_gas=` a per-galaxy scalar driving nebular emission. They do **not** track each other — the `if neb_logZ_gas is None: neb_logZ_gas = log_z` inheritance in the four photoionized backends is dead code on the build path, because the grammar always supplies `neb_logZ_gas`'s declared `Fixed(-0.3)` (measured: unset is bit-exact with `-0.3`). A tabulated `met=` beside that default warns rather than enriching the stars but not the gas.
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
- [`docs/dev/model-construction.md`](docs/dev/model-construction.md) — the one narrative (build path + dispatch + this recipe in context)
- [`docs/dev/sed-model-components.md`](docs/dev/sed-model-components.md) — full how-to + three worked examples (closed-form, library, NN emulator)
- [`docs/dev/archive/forward-model-architecture.md`](docs/dev/archive/forward-model-architecture.md) — architectural context
- [`docs/adr/0011-sed-model-component-base.md`](docs/adr/0011-sed-model-component-base.md) — the design decision
- [`src/tengri/components/dust/wg00_model.py`](src/tengri/components/dust/wg00_model.py) — canonical small component (closed-form attenuation)
- [`src/tengri/components/agn/skirtor_model.py`](src/tengri/components/agn/skirtor_model.py) — canonical template-library component

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
(`tests/inference/test_backend_conformance.py`) picks up the new entry
automatically — no test-file edits required. But `tests/inference/` is
auto-marked `slow` (`_SLOW_TREES` in `tests/conftest.py`) and so is
**deselected from the default run and from the PR gate**. Adding a
backend and seeing green tells you nothing; run it explicitly:

```bash
.venv/bin/pytest tests/inference/test_backend_conformance.py -q -m slow
```

**JIT rule** (non-negotiable): `InferenceContext` must never be hashed
into a JIT key or passed through `jax.jit` / `jax.vmap` / `jax.lax.scan`
as a traced argument. Pull primitives (`neg_log_posterior_fn`, `data_args`) out of
context *before* entering JAX transforms. The context's
`__jax_array__` guard raises on accidental tracing.

**Source of truth:** `docs/adr/0010-inference-backend-protocol.md`.

## Adding a hierarchical sampler (the flat seam)

**Read `docs/dev/hierarchical-flat-seam.md` before touching `_hierarchical_flat.py`.**
Every hierarchical sampler operates in ONE standardized space — a flat
unconstrained vector with an iid N(0,1) prior; every declared physical prior
is realized exactly via its distribution's `unstandardize` pushforward
(#1651), and an object without that contract is refused by name. Wiring a sampler is three
edits (a `FLAT_SAMPLERS` entry, one driver branch, the set-pin test in the
same commit) **plus one executed fit** — every wiring in the 2026-08 series
found a runtime-only defect (blackjax API drift, NaN tuning → frozen chain,
prior double-counting). A name may map only to a driver that runs the
algorithm the name promises; stand-ins are silent substitution, the seam's
founding bug. Frozen chains, non-finite tuning, unknown kwargs, broken-tier
access, NSS live sets that cannot span D, and NSS evidence integrals cut off
by the iteration cap all raise loudly by design — do not weaken these guards.

## Critical gotchas

- `jax.random.fold_in(key, hash(string))` overflows uint32. Use `abs(hash(x)) % (2**31)`
- Never create `Model`/`ParamSpec` inside a JAX gradient tape (traced values fail in `__init__`)
- JAX Metal (Apple GPU) causes test failures. Use `JAX_PLATFORMS=cpu` for reliable results
- Ray Tracing: step_size=0.05 for D~137; sharp viability cliff at ~0.06 (acceptance drops to 0%)
- NIFTy geoVI: use 4-12 samples per KL iteration, not 80
- `VIConfig.n_samples=3` doubles to 6 effective samples via `mirror_samples=True` — when tuning, think in effective samples
- `"vi"` (NIFTy) and the pure-JAX `"native_vi_nonlinear"` / `"native_vi_linear"` target the same objective but are NOT posterior-equivalent. Native is ~19× faster warm on 7-D and ~25× on 137-D stochastic (2.8s vs 71s), but PSD timescale `sfh_field_psd_tau_myr` differs by an order of magnitude between paths (82 vs 6 Myr). Both native backends are `tier=broken` (segfault on DPL/dense_basis photometry mocks) — do NOT reach for them, and never teach them in an example. On the batched catalog path they raise `NotImplementedError` for per-galaxy redshift and for presence masks; use `method="mcmc_nuts"` / `"mcmc_hmc"` (both batched and vmappable) or `method="map"` (sequential) instead. There is no `"vi_native"`; that name raises `ParameterError` (a `ValueError`, **not** a `KeyError`) listing every valid method. Asking for a broken-tier backend by its real name raises `BackendError` instead, which names the tier and the working alternatives — two different failures, so catch `TengriError` if you want both. See `bench/reports/2026-04-17_native_vs_nifty.md`
- Use `.shape[0]` instead of `len()` on JAX arrays to avoid `ConcretizationTypeError` under JIT
- Use tolerance comparison (`abs(x - default) < 1e-6`) not `==` for float equality on traced values
- IGM `igm_transmission(wave_obs, z)` takes **observed-frame** wavelengths (not rest-frame)
- Dust emission templates auto-load from `data/`; analytic fallbacks are NOT suitable for science
- AGN torus in `torus.py` are **toy models** — use SKIRTOR for science
- `agn_torus_frac`: do NOT auto-derive from `cos(theta_torus)` in forward pass (gradient discontinuity)
- Inference internals use `mode="_traceable"` (safe inside JIT). User-facing defaults to `mode="auto"`
- **`sfh={'age_kernel': ...}` is NOT a speed knob — and `'dsps'` is the SLOWER one** (2026-07-31, #964). Selects how the SFH is integrated onto the SSP age grid: `'cic'` (default, dense cloud-in-cell integrand) or `'dsps'` (DSPS's histogram kernel). Menu: `tengri.list_age_kernels()`. Measured end-to-end on `predict_photometry` gradients (quiet box, interleaved reps, A/A control): **`'cic'` is 3.5% faster on the exact path and 13% faster under `WavePrecomp()`** — precompute makes DSPS relatively *worse*. The cause is **not** arithmetic: by compiled-HLO cost analysis DSPS uses ~1% **fewer** FLOPs and fewer bytes. It compiles to **2x as many `while` loops** (14 vs 7 exact; 13 vs 6 precomp) and ~40% more fusion regions — sequential, latency-bound, unfusable work. Precompute shrinks the vectorizable part (cic fusions 356→212) but not the loops (dsps whiles 14→13), so DSPS's fixed sequential share grows. Caveat: that is a **CPU wall-clock** effect from op structure, so the ordering is not guaranteed on GPU — re-measure there rather than assuming. **Do not micro-benchmark `compute_dsps_age_weights` to judge any of this — it has no call sites on the model path**; timing it says nothing about `apply()`. `'dsps'` is also not equivalent: it interpolates `log10(M(<t))` in `log10(t)`, annihilating the first SSP node older than the SFH start (3.8% of the mass, +1.2% optical CSP bias vs FSPS/bagpipes — **on the grid that was measured**; the magnitude is one node's share of the mass, so it is SSP-grid dependent: 0.64% on the 93-node ProGeny/MILES grid for the same delayed-tau, 0.13-0.29% for a dpl. Re-measure, do not quote) and shifting the `sfh_*_age_gyr` **gradient by 43%**. Use `'dsps'` only for cross-code parity. `'cic'` + a GP-field SFH raises (the field draw has no dense integrand).
- **Build-time `approx=WavePrecomp(...)` is the speed knob** (2026-05-20). Opting in publishes the SSP × filter LUT and routes `predict_photometry` through `observation.predict_via_precomp`. Default `approx=None` uses the exact wave-grid path — **for forward work only: every fit surface now defaults to the LUT** (2026-08-10). `Fitter`, `PopulationFitter`, and `CatalogFitter` all resolve `approx="auto"` at fit time (photometry → `WavePrecomp`, spectroscopy → `SpectrumPrecomp`); pass `approx=None` to a fitter to force the exact path. Measured on the 2-galaxy D=516 seam fixture: cold HMC 145 s → 19.4 s, cold ESS 750 s → 7.3 s. A fit that takes minutes instead of well under one is a signal something is off this path. The dict / bool / string forms (e.g. `approx={'wave_precomp': True}`, `approx=True`, `approx='wave_precomp'`) were removed — `TypeError` at construction. Override ztable sampling via `WavePrecomp(n_z=200, z_min=0.0, z_max=3.0)`.
- **`FeaturePrecomp` has two jobs, and dust disarms only one** (2026-08-13, #1748 + #1770). Do not reason about it as a single lever — that conflation is #1770, and it cost a measured 4.77x.
  - **Photometry** is served from a per-Q_H grid, which requires zeroing `sed_nebular`; since #1281 that is permitted only when nothing downstream reads the continuum, and `DustSEDComponent` declares it as an input, so **any dusty model disarms this half**. Measured, gradient FLOPs off the compiled HLO with a dust-free control in the same run: with dust `WavePrecomp` 65,438,628 vs the pair 65,438,628 (**1.00x, bit-identical**); without dust 54,827,036 vs 1,789,868 (**30.6x**). Exact FLOP equality is the signature of a config that never reaches the graph. Not a regression to restore: pre-#1281 the shortcut made a dusty model's photometry differ from exact by **0.41%** against a 0.0115% dust-free floor, and a constant forward bias enters the gradient multiplied by SNR (#1671).
  - **A line channel** is served by supplying the line fluxes from the table, so `loss_functions` need not set `needs_state=True` and rebuild the full-grid SED via `predict_state` per likelihood. **Dust does not touch this.** Measured on the #1477 fixture — dusty model, 5 bands + 3 line fluxes, gradient FLOPs of the *fit objective* — `WavePrecomp` 1,933,823 vs the pair **405,825** (**4.77x**), beside a dust-free control identical to the digit either way (251,783).
  - `fast_nebular_can_engage` answers the **photometry** question only. Consult it for the photometry-only top-up; never to decide whether a line-flux fit gets the LUT. #1760 did, on the strength of a guard that measured `jnp.sum(model.predict_photometry(params))` — an objective that cannot observe the line-channel saving — and every dusty line-flux fit silently returned to the pre-#1477 cost.
  - **Measure the objective you are claiming about.** A photometry-surface FLOP count says nothing about the line channel, and vice versa. Guards: `test_bug_1748_feature_precomp_effect.py` (photometry) and `test_bug_1770_line_lut_survives_dust.py` (lines) — run both.
- **One NUTS fit per notebook process.** Each warmup peaks at 3–6 GB on small models (D ≤ 7 photometry) but can hit 20+ GB on D ≈ 8 with `mean_sfh_type="dense_basis"` — observed 22.78 GB peak on nb00 with default `dense_mass_matrix=True`. Multi-fit notebooks (and any single fit on D ≥ 8) need `dense_mass_matrix=False` or `mcmc_hmc`. See `docs/dev/notebook_orchestration_oom.md`
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
- `area:examples` — sphinx-gallery scripts under `examples/` (rendered into `docs/auto_examples/`)
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

- `docs/dev/model-construction.md` — **canonical** model construction narrative (build path + one `_REGISTRY` dispatch + add-a-model recipe)
- `docs/dev/agents.md` — AI agent documentation
- `docs/dev/history/handoff-2026-04.md` — frozen project-status snapshot (pre Phase II-3 closure)
- `docs/dev/design_philosophy.md` — architecture decisions
- `docs/dev/NAMING_CONTRACT.md` — naming conventions (read before any rename/refactor)
- `docs/dev/20260404-refactor.md` — refactor plan
- `docs/dev/api_migration_v0.x.md` — public-API migration table (Phase 1→6 + Part II scaffold)
- `docs/known_bugs.md` — bug tracking (all currently fixed)
- `docs/dev/notebook_orchestration_oom.md` — operational rules for OOM-safe notebook authoring (multi-fit, subagent zombies, watchdog)
- `tools/check_param_prefixes.py` — CI guard for free-parameter prefix rule (NAMING_CONTRACT §3.2)
- `tools/check_param_defaults.py` — CI guard that no signature default falls outside its parameter's declared prior. Such a default is unreachable by any fit and is usually a unit confusion: nine AGN entry points shipped `agn_log_lbol=45.0` (the `log10(erg/s)` magnitude) against a declaration in `log10(L/L_sun)`, so a bare call was ~1e33 too luminous. Read defaults off the declaration with `declared_default(PARAMS, name)` instead of repeating the number (ADR-0011)
- `tools/check_param_ranges.py` — CI guard that a **call-site prior overlaps** its parameter's declared prior; the same rule as `check_param_defaults.py` one step later (that one checks a signature default is *inside* the support). #369 renamed `sfh_*_log_peak_sfr` → `log_total_mass` — `log10(SFR)` became `log10(M*)` — and carried the ranges over unconverted, leaving 150 sites declaring priors like `Uniform(-1.0, 2.5)` on a stellar mass (0.1–316 Msun). Every affected fit converged and nothing raised, so the tests kept passing while sampling a regime no galaxy occupies. **Overlap, not containment** — narrowing or widening a prior is ordinary; a range sharing no point with the declaration is always a units error. Pinned `Fixed()` scalars are deliberately out of scope: `Fixed(0.0)` on a `log_total_mass` is a unit-mass normalization for crossval, not a bug. Runs in `smoke` (imports the registry)
- `tools/check_british_spelling.py` — CI guard for American-English spelling (NAMING_CONTRACT §10); `--fix` to auto-rewrite
- `tools/check_reimplementation_language.py` — CI guard for the credit rule above: fails on "ported from" / "copied from" / "adapted from" and on "port" beside a reference-code name. Code "implements"; data is "repackaged". Files that must quote the banned wording are allowlisted in `EXCLUDE_FILES`
- `tools/check_doc_examples.py` — CI guard that every symbol named in a `src/` docstring or published doc actually exists (`docs/api/*.rst` are autodoc stubs, so docstrings *are* the API reference, and no doctest runner executes them). Runs in the `smoke` job. `docs/dev/` is out of scope by design: design notes and parity audits legitimately name removed or not-yet-built API
- `tools/check_notebook_cells.py` — CI guard that no committed notebook holds a code cell with its newlines deleted (`'import osimport sysimport time'`). 73 such cells sat in five notebooks for months: the JSON stays valid, so every notebook-aware tool loads them happily, `check_notebook_renders.py` covers `docs/spine/` only, and ruff never reaches `notebooks/archive_2`. **Flags on length, not on failing to parse** — a collapsed cell whose first line ends in a comment swallows the rest of the cell and parses cleanly, which was 34 of the 73 and the dangerous half (the ones that fail to parse at least announce themselves). Threshold 120 chars, measured: the longest healthy single-line cell in the tree is 90, the shortest damaged one 148. Fix by rebuilding from the jupytext `.py` mirror
- `tools/check_file_sizes.py` — CI guard ratcheting repository growth. #1817 declined the history rewrite (public since 2026-03-21 with two third-party forks, so renumbering every commit SHA buys clone size and nothing else), which leaves the accumulation rate as the part that compounds. Outside `data/` the ten largest tracked files are all `.ipynb` between 4 and 9 MiB, almost entirely base64 PNG. Existing offenders are listed in `INVENTORY` and may stay but not grow; `data/` gets its own higher limit for the SSP grids. `--list` prints current sizes
- `tools/check_claude_md_paths.py` — CI guard that every repo path named in **this file** exists. The guard above resolves *symbols*; this one resolves *paths*, and nothing did before `area:examples` spent an unknown stretch pointing at a *docs/examples/* that has never existed. **Code markup on a path in this file is an assertion that the path exists** — that is the rule the guard enforces, so write a path you are describing rather than citing (a removed one, a hypothetical) as plain text. A token resolves against the repo root or against `src/tengri/` (this file writes `parameters/groups.py` and 16 others package-relative). Bare basenames (`cue.py`) are references, not paths, and are deliberately not checked
