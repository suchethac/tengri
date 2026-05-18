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
- `component.py` — the `SEDComponent` adapter (the orchestration shape)
- `_params.py` — the parameters this component owns (priors, descriptions, units)
- `*.py` for the actual physics (e.g. `attenuation.py`, `emission.py`)
- `*_precompute.py` for filter-preintegrated lookup tables (optional)

**North-star file for style:** `components/dust/attenuation.py` reads like
prose from Charlot & Fall (2000) with a Python skeleton attached. Copy its
rhythm.

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
| Build the SED end to end | `forward/sed_model.py` (the `SEDModel` class) |
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

- **Not in `core/`** — `core/` holds *protocols and contracts* (interface
  definitions for `SEDComponent`, `DerivedBundle`, etc.), not business logic.
  The real forward-model orchestrator is `forward/sed_model.py`.
- **Not in `bench/`, `profiling/`, `preprocessing/`** — these are auxiliary
  packages (benchmarks, profiling helpers, data hygiene). The actual physics
  and inference are in `components/`, `observation/`, `inference/`,
  `forward/`.
- **Not in `presets/`** — that's a small library of canned model + observation
  pairs for tests and demos, not the recipes you'd use day-to-day. For those,
  use `tengri.recipes`.

## Known mild confusion (and why we tolerate it)

- `components/sfh/` and `components/sps/` exist as deprecation shims for
  `components/stellar/sfh/` and `components/stellar/sps/`. They emit
  `DeprecationWarning` at import time and will be removed in v1.0. New code
  should use the `stellar/` paths.
- `core/` is named `core/` for legacy reasons; "protocols" or "interfaces"
  would be more accurate. Worth renaming if the cost of touching every import
  becomes worth it. (Today, ~12 internal import sites.)
