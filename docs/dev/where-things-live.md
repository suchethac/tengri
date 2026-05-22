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
| Dust IR emission (modified BB, Casey, Dale, DL07, PAH) | `components/dust/emission.py` |
| Nebular continuum + line emission | `components/nebular/` |
| AGN disc / torus / NLR / BLR | `components/agn/` |
| IGM transmission | `components/igm/` |
| Radio synchrotron + free-free + jets | `components/radio/` |
| X-ray (XRBs + AGN corona) | `components/xray/` |

Each component package has:
- `component.py` — the bare `SEDComponent` Protocol adapter (used by stellar, IGM, and other components with rich state)
- `<name>_model.py` — single-file `SEDModelComponent` ports (added 2026-05; the default authoring style for new models)
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
[`components/dust/calzetti_model.py`](../../src/tengri/components/dust/calzetti_model.py);
a library example is
[`components/agn/skirtor_model.py`](../../src/tengri/components/agn/skirtor_model.py).

The base class lives at
[`components/sed_model_component.py`](../../src/tengri/components/sed_model_component.py).
For the how-to and the contract, see
[`docs/dev/sed-model-components.md`](sed-model-components.md); for the
architectural context (cross-component contract, WavePrecomp, builder
resolution), see [`forward-model-architecture.md`](forward-model-architecture.md).

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
| Noise models (Gaussian, Student-t, GP, calibration marginalisation) | `observation/noise.py`, `observation/calibration.py` |
| Emission-line marginalisation | `observation/eline_marginalization.py` |
| The `Fitter` class and JIT-cached engines | `inference/fitter.py`, `inference/jit_engine.py` |
| MCMC / VI / Pathfinder backends | `inference/backends/` |
| Likelihood implementations | `inference/likelihoods/` |
| Posterior containers + diagnostics | `inference/posterior.py` |

## Forward model assembly

| Want to... | Open |
|---|---|
| Reach the forward model from inference | `forward/forward_model.py` (`ForwardModel` — outer shell + `.build()` + `.predict()`). Single-population only in v0; multi-population lands with ADR-0012. |
| Compose populations | `forward/population.py` (`Population(name, sed, spatial=None)` — one SubModel pair per population) |
| Find the SubModel Protocol | `protocols/submodel.py` — runtime-checkable contract (`run`, `declared_parameters`) |
| Build the SED end to end | `forward/sed_model.py` (the `SEDModel` class — the SED chain; satisfies `SubModel` directly via `.run()` + `.declared_parameters()`) |
| Write / edit a spatial profile (Sérsic, exponential, flat slab) | `components/spatial/<name>.py` — subclass `SpatialModelComponent` (auto-discovered free params, default `apply()` writes `state.derived["spatial_profile_2d"]`) |
| Find the SpatialComponent Protocol | `protocols/spatial.py` — runtime-checkable mirror of `SEDComponent` |
| Understand which kernel (exact / hybrid / fast) is chosen | `forward/_kernels/` |
| Wire a new physics component into the pipeline | `forward/components_assembly.py` |
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

