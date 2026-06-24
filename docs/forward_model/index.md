# Forward model

Tengri's forward model is split into two clearly separated layers.

```
┌─────────────────────────────────────────────────────────┐
│  Inference (Fitter / MAP / NUTS / VI / …)               │
│  Talks only to ForwardModel.predict_observables(params).            │
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│  ForwardModel — the thin outer shell                    │
│  • owns one SubModel and one observation                │
│  • .predict(params) → dict {phot_fnu, spec_fnu, ...}    │
│  • inference never has to choose between predict_*      │
│    methods; the dict tells it which channels exist      │
└────────────────────────┬────────────────────────────────┘
                         ▼
            ┌────────────┴────────────┐
            │      SubModels          │
            ├─────────────────────────┤
            │  SEDModel               │  single galaxy SED chain
            │  PopulationSEDModel          │  N galaxies, shared params (hierarchical)
            │  SpatialModel           │  single galaxy spatial profile
            │  SpatialSEDModel        │  SED + Spatial (joint)
            │  PopulationSpatialSED   │  far-future composition of all of the above
            └─────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│  SEDModel — the SED physics chain (used inside every    │
│  SubModel variant)                                      │
│  star formation history → simple stellar populations    │
│   → nebular and AGN emission → dust attenuation and     │
│   re-emission → IGM absorption → photometry/spectroscopy│
│  Pure JAX. JIT-compilable, gradient-traceable,          │
│  batchable through vmap.                                │
└─────────────────────────────────────────────────────────┘
```

The outer-shell signature stays uniform across all SubModel variants —
construction is always ``ForwardModel.build(<slot>=..., observation=obs)``
and inference is always through the standard ``Fitter`` pipeline.

## The minimum usable fit

```python
import tengri
from tengri import FIXED, Fitter, ForwardModel, Parameters, SEDModel

ssp = tengri.load_ssp_data()
obs = tengri.Observation(photometry=tengri.Photometry.from_names(["sdss_u", "sdss_g", "sdss_r"]))

# 1. Build the SED chain.
sed = SEDModel.build(
    ssp_data=ssp,
    observation=obs,
    sfh={"type": "dpl", "*": FIXED, "log_total_mass": 10.0, 3.0)},
    dust={"type": "two_component", "law_bc": "calzetti", "*": FIXED},
    neb={"type": "none"},
    redshift=tengri.Fixed(0.05),
)

# 2. Wrap it in the outer shell.
forward = ForwardModel.build(sed=sed, observation=obs)

# 3. Hand to inference.
fitter = Fitter(forward, data=photometry_array, noise=noise_array)
posterior = fitter.run("map")
posterior.summary()
```

The `SEDModel.build(...)` call is unchanged from previous releases — same
nested-dict grammar, same SSP/observation arguments, same internal
pipeline. The only addition is `ForwardModel.build(sed=..., observation=...)`
on top, which gives inference a uniform `.predict(params) → dict`
interface.

## Why the split?

The SED chain ("the physics") and the surface that inference consumes
("the prediction dict") have always been two different responsibilities.
Before the split, `SEDModel.predict_photometry`, `SEDModel.predict_spectrum`,
`SEDModel.predict_joint`, etc. encoded "which channels exist" inside the
method *name*. Inference backends therefore had to know whether they were
fitting photometry, spectroscopy, or both, and pick the right method.

After the split:

- Adding a new observation channel is one new key in the prediction
  dict — no inference-side branching.
- Existing user code that calls `sed.predict_photometry(params)`
  directly still works; nothing about `SEDModel` changed.
- The dev-side surface area is smaller (a Protocol with two
  methods — `declared_parameters`, `run` — instead of a handful of
  channel-specific predict methods).

## `ForwardModel.build(...)` reference

```python
ForwardModel.build(
    *,
    sed: SEDModel | None = None,             # single-galaxy SED
    spatial: SpatialModel | None = None,      # add a spatial profile
    population: PopulationSEDModel | None = None,  # hierarchical multi-galaxy
    populations: Iterable[Population] | None = None,  # explicit decomposition
    observation: Observation,
) -> ForwardModel
```

Pick exactly one of `sed=`, `population=`, or `populations=`. Returns
a frozen dataclass. The only API users typically call afterwards is
`forward.predict_observables(params) → dict`. Everything else is via `Fitter` or
posterior helpers.

## Hierarchical population fits

When you have many galaxies that share an underlying parameter — the
canonical case is the PSD hyperparameters ``σ_PSD``, ``τ_PSD`` of the
stochastic-SFH prior — wrap them in a ``PopulationSEDModel`` and pass it
to ``ForwardModel.build(population=...)``:

```python
from tengri import ForwardModel, PopulationSEDModel, SEDModel

template = SEDModel.build(ssp_data=ssp, observation=obs, ...)

pop = PopulationSEDModel(
    sed=template,
    galaxies=[{'flux_obs': ..., 'noise': ...}, ...],   # N galaxies
    # shared= and priors= default to PSD; override for other hierarchies
)

forward = ForwardModel.build(population=pop, observation=obs)

# Inference is the same as for a single galaxy: one Hamiltonian path,
# Fitter sees a (N_gal, n_filters)-shaped batched prediction and
# minimizes chi^2 + xi^T xi over the joint latent space.
fitter = Fitter(forward)          # auto-extracts (data, noise) from pop.galaxies
posterior = fitter.run('vi')
```

The PSD priors live on the ``PopulationSEDModel`` construction — not on a
separate ``HierarchicalFitter`` — so there is one place that
parameterizes the hierarchy.

Inference routes through the standard
:class:`tengri.Fitter` machinery natively (PRs #241–#246, 2026-05).
There is **one** information-Hamiltonian path —
``Fitter(forward, ...).run('vi')`` — whether ``forward`` holds an
:class:`SEDModel` (single galaxy), a :class:`PopulationSEDModel`
(hierarchical), or :class:`SpatialSEDModel`. The
:class:`PopulationSEDModel` publishes its batched axes
(``{'galaxy': 0}``) and the spec view publishes per-param shapes
(``(N_gal,)`` for per-galaxy, ``()`` for shared); the Fitter's
existing inference backends consume the batched output without any
type-specific code.

The legacy :class:`tengri.PopulationFitter` direct API remains
importable but emits a one-shot ``DeprecationWarning`` pointing at
this canonical path; the legacy class will be removed in v1.0.

## Composing SubModels

The SubModel lattice composes — every variant either contains the
others or runs alongside them, but each is a strict ``SubModel`` from
``ForwardModel``'s perspective:

| SubModel | Used when |
|---|---|
| `SEDModel` | one galaxy, SED only |
| `SpatialModel` | one galaxy, morphology-only fit (e.g. resolved imaging) |
| `SpatialSEDModel` | one galaxy, joint spatial + SED (e.g. SDSS-fiber spec + photometry) |
| `PopulationSEDModel` | many galaxies, hierarchical shared parameters (PSD) |
| `PopulationSpatialSED` *(far future)* | many galaxies with shared parameters and morphology |

Adding a new SubModel is one Python file plus one entry in the
ForwardModel-build kwargs table — the inference layer doesn't change.

## Forward chain — the SED physics

The differentiable forward chain inside `SEDModel` follows the standard
SED-fitting cascade:

```
star formation history (SFH)
        ↓
simple stellar populations (SSP) — DSPS interpolation
        ↓
nebular and AGN emission  (added)
        ↓
dust attenuation and re-emission  (transforms + adds)
        ↓
IGM absorption  (transforms)
        ↓
observed-frame photometry / spectroscopy via filter / LSF convolution
```

Each step is a pure JAX function, so the whole chain is JIT-compiled,
gradient-traceable, and batchable through `vmap`.

## Spine notebooks

| Notebook | Focus |
|----------|-------|
| [`00_quickstart`](../spine/00_quickstart) | One-screen end-to-end fit |
| [`02_sed_anatomy`](../spine/02_sed_anatomy) | The panchromatic SED, component by component |
| [`04_building_models`](../spine/04_building_models) | Building models with `Parameters`; swapping SFH families, dust laws, IR templates |
| [`05_fitting_photometry`](../spine/05_fitting_photometry) | Photometry-only MAP/NUTS fits |
| [`06_fitting_spectroscopy`](../spine/06_fitting_spectroscopy) | Spectroscopy-only fits with calibration |
| [`07_joint_photo_spec`](../spine/07_joint_photo_spec) | Combining photometry and spectroscopy in one likelihood |
| [`08_emission_lines`](../spine/08_emission_lines) | Nebular line fluxes, BPT diagnostics, Hα-derived SFR |

The pedagogical order in the spine is *SFH → dust → nebular → AGN →
multi-wavelength*, which mirrors the order of decisions a user typically
makes rather than the strict internal call order.
