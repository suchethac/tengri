# Where things live

A one-page guide to the `src/tengri/` layout, written for astronomers opening
the codebase for the first time.

This is **not** an exhaustive package reference — it's a "if I want to edit X,
where do I look?" cheat sheet. Pair it with `style-and-voice.md` (how to write)
and `docstring-standard.md` (how to document).

## The physics is in `components/`

One directory per physics block. Open the one that matches what you want to
edit:

| Want to edit... | Open |
|---|---|
| Star formation history shape, GP-based bursty SFH | `components/stellar/sfh/` |
| Stellar spectra, SSP grids, mass remaining | `components/stellar/sps/` |
| Dust attenuation laws (Calzetti, Charlot+Fall, …) | `components/dust/attenuation.py` |
| Dust IR emission (modified BB, Casey, Dale, DL07, PAH) | `components/dust/emission/` (`analytic/` closed-form components, `templates/` library components) |
| Dust energy-balance integral (`L_absorbed`, LyC-masked) | `forward/energy_balance.py` (canonical `bolometric_absorbed`, #922); fast-path LUT in `components/dust/energy_balance_precompute.py` |
| Nebular continuum + line emission | `components/nebular/` |
| AGN disc / torus / NLR / BLR | `components/agn/` |
| IGM transmission | `components/igm/` |
| Radio synchrotron + free-free + jets | `components/radio/` |
| X-ray (XRBs + AGN corona) | `components/xray/` |

Each component package has:
- `component.py` — the bare `SEDComponent` Protocol adapter (used by stellar, IGM, and other components with rich state)
- `<name>_model.py` — single-file `SEDModelComponent`s (added 2026-05; the default authoring style for new models)
- `_params.py` — the parameters this component owns (priors, descriptions, units)
- `*.py` for the actual physics (e.g. `attenuation.py`, `emission.py`)
- `*_precompute.py` for filter-preintegrated lookup tables (optional)

**North-star file for style:** `components/dust/attenuation.py` reads like
prose from Charlot & Fall (2000) with a Python skeleton attached. Copy its
rhythm.

## Adding a new model

For most new models — closed-form attenuation laws, dust IR libraries, AGN
torus libraries, nebular emulators — write **one file** at
`components/<domain>/<name>_model.py` subclassing `SEDModelComponent`. The
canonical small example is
[`components/dust/wg00_model.py`](../../src/tengri/components/dust/wg00_model.py);
a library example is
[`components/agn/skirtor_model.py`](../../src/tengri/components/agn/skirtor_model.py).

The base class lives at
[`components/sed_model_component.py`](../../src/tengri/components/sed_model_component.py).
For the how-to and the contract, see
[`docs/dev/sed-model-components.md`](sed-model-components.md); for the
build path, the single `_REGISTRY` dispatch, and the per-domain table, see
[`model-construction.md`](model-construction.md) (the canonical narrative —
[`forward-model-architecture.md`](archive/forward-model-architecture.md) is kept as
a historical design doc).

The bare `SEDComponent` Protocol stays as the fallback for models that don't
fit the `predict(p, sed_in, wave, **inputs)` shape — stellar (rich state
machine), IGM (observer-frame transformation). The canonical bare-Protocol
reference is `components/radio/component.py`.

## Defining parameters

Each component owns its own `_params.py` (see the list above and the
`tengri.parameters` package docstring). To add or rename a parameter:

1. Edit the right `_params.py` (the component's, NOT a central registry).
2. The registry in `tengri.parameters.registry` picks it up automatically on
   the next process start.
3. Cross-check with `tools/check_param_prefixes.py` (CI guard).

## Observation, fitting, inference

| Layer | Where |
|---|---|
| Photometry + spectroscopy data containers | `observation/` |
| Filter curves and filter management | `observation/filters/` |
| Noise models (Gaussian, Student-t, GP, calibration marginalization) | `observation/noise.py`, `observation/calibration.py` |
| Emission-line marginalization | `observation/eline_marginalization.py` |
| Instrument LSF (Gaussian) and the DESI/PFS banded resolution matrix | `observation/spectrum.py` (`apply_lsf`), `observation/banded.py` (`banded_matvec`, `resolution_bands_from_desi`, `block_diagonal_bands`) |
| Loading a DESI coadd (per-camera b/r/z grids + their resolution matrices) | `io/desi.py` (`read_desi`, `read_desi_cameras`, `desi_spectroscopy`) |
| The `Fitter` class and JIT-cached engines | `inference/fitter.py`, `inference/jit_engine.py` |
| MCMC / VI / Pathfinder backends | `inference/backends/` |
| Likelihood implementations | `inference/likelihoods/` |
| Posterior containers + diagnostics | `inference/posterior.py` |

## Forward model assembly

| Want to... | Open |
|---|---|
| Reach the forward model from inference | `forward/forward_model.py` (`ForwardModel` — outer shell + `.build()` + `.predict()`). Single- and multi-population; the latter uses the `"<pop>.<prefix>_<param>"` namespace per ADR-0012. |
| Compose populations | `forward/population.py` (`Population(name, sed, spatial=None)` — one SubModel pair per population; names must be distinct and must not contain `.`) |
| Sum predictions across populations | `observation/joint_observation.py` (`JointObservation.predict_summed` — linear-flux sum across populations per channel) |
| Find the SubModel Protocol | `protocols/submodel.py` — runtime-checkable contract (`run`, `declared_parameters`) |
| Build the SED end to end | `forward/sed_model.py` (the `SEDModel` class — the SED chain; satisfies `SubModel` directly via `.run()` + `.declared_parameters()`) |
| Write / edit a spatial profile (Sérsic, exponential, flat slab) | `components/spatial/<name>.py` — subclass `SpatialModelComponent` (auto-discovered free params, default `apply()` writes `state.derived["spatial_profile_2d"]`) |
| Find the SpatialComponent Protocol | `protocols/spatial.py` — runtime-checkable mirror of `SEDComponent` |
| Compose a list of spatial components into one sub-model | `forward/spatial_model.py` — `SpatialModel(components=[...])` satisfies `SubModel` |
| Join a SED chain with a spatial chain for joint fits | `forward/spatial_model.py` — `SpatialSEDModel(sed=..., spatial=...)` runs SED then Spatial |
| Understand how the forward pass is compiled (exact vs `WavePrecomp`/`SpectrumPrecomp`) | `forward/sed_model.py` (`approx=` handling + `predict_observables_jit`) and `forward/orchestrator.py` (`run_components`) |
| Wire a new physics component into the pipeline | `forward/component_factory.py` (`build_components` + the `_resolve_registry_component` seam) |
| Add filter-preintegration to a new component | `forward/precompute/` (registry) + new `*_precompute.py` next to the component |

## User-facing public API

Top-level imports — `from tengri import …`. The full surface is enumerated in
`src/tengri/__init__.py`. The main entry points are:

- `SEDModel`, `Parameters`, `Fitter` — the three you'll touch most often
- `recipes` — pre-built model recipes (`recipes.star_forming_photometry()` etc.)
- `Photometry`, `Spectroscopy`, `Observation`, `NoiseModel` — observational data
- `Posterior`, `MockData`, `FitResult` — inference outputs

The thin re-export namespaces `tengri.cosmology`, `tengri.units`, `tengri.plot`,
`tengri.filters` exist purely so you can write
`from tengri.cosmology import angular_diameter_distance` instead of
`from tengri.utils.cosmology import …`. They are deliberate.

## Navigating the two big spine files

The two largest modules are deliberately long (splitting them was declined —
see #847); navigate them by section instead of scrolling:

**`forward/sed_model.py`** (~6k lines) reads top-to-bottom as:
`WavePrecomp`/`SpectrumPrecomp` approx configs → `SEDModel.__init__` (the
`_init_<domain>()` chain: observation, ssp, sfh, metallicity, dust, igm,
nebular, agn, multiwavelength, instrument, cosmology) → internal param
accessors (`_get_internal_params`) → prediction
methods (`predict_sfh` / `predict_rest_sed` / `predict_obs_sed` /
`predict_photometry` / `predict_spectrum`) → the JIT kernel builder
(`predict_observables_jit`, `_template_data_for_jit`) → component-chain
assembly (`_build_component_chain`) → `SEDModel.build()` / `from_config()`
→ thin `fit*`/`mock*` delegates into `forward/convenience.py`.

**`inference/fitter.py`** (~3k lines): `Fitter.__init__` → method resolution
(`resolve_method`) → the compile cache (`CompileCache` interplay, smart-lean
vs persistent) → `Fitter.run()` (dispatches through the backend registry in
`inference/_registration.py`; every backend receives an `InferenceContext`,
ADR-0010) → posterior-sample drawing helpers.

## Where things definitely are NOT

- **Not in `protocols/`** — `protocols/` holds *interface definitions*
  (`SEDComponent`, `DerivedBundle`, `PipelineState`, etc.), not business logic.
  The real forward-model orchestrator is `forward/sed_model.py`.
  (This directory was called `core/` before May 2026 — the old path still
  works via a one-shot DeprecationWarning shim.)
- **Not in `bench/`, `profiling/`, `preprocessing/`** — these are auxiliary
  packages (benchmarks, profiling helpers, data hygiene). The actual physics
  and inference are in `components/`, `observation/`, `inference/`,
  `forward/`.
- **Not in `presets/`** — that's a small library of canned model + observation
  pairs for tests and demos, not the recipes you'd use day-to-day. For those,
  use `tengri.recipes`.

