# Forward Model Architecture

> Status: **design doc, pre-implementation.** Captured 2026-05-21 from a
> design conversation. Section 2 depends on the in-flight
> `SEDModelComponent` refactor; reconcile before treating this doc as
> canonical. Multi-population namespace decision is captured separately in
> [ADR-0012](../adr/0012-forward-model-population.md).

This document describes the target architecture of the `tengri` forward
model — the thin shell around the physics modules that inference talks
to. It is the reference picture for the next several refactors.

## TL;DR — what changes, in one paragraph

`SEDModel` today is *both* the outer shell and the SED-physics chain.
This document splits those two roles. The outer shell becomes a new
class `ForwardModel`, whose only job is to compose physics sub-models
(SED, spatial, joint spatial-SED, and eventually stellar atmospheres in
a forked repo) with an `Observation` model, and to expose a single
`.predict(params)` method to inference. The SED-physics chain stays
inside an `SEDModel` sub-model, unchanged at the component level. A
mirror-symmetric `SpatialModel` is added so that joint
spectrophotometric fits can model the aperture mismatch between
spectroscopy (one fiber-sized region of the galaxy) and photometry
(the whole galaxy) *physically*, not by flat-slab scaling. From the
start, `ForwardModel` holds **multiple populations** (e.g. AGN
point source + Sérsic bulge + exponential disc), each with its own
SED and spatial model.

## 1. The big picture

The forward model is layered. Each layer has exactly one
responsibility and one Protocol-shaped contract with the layer below.

```
┌──────────────────────────────────────────────────────────────┐
│  Inference          (VI, MCMC, MAP, NSS, …)                  │
│    ↳ asks ForwardModel for log p(data | params)              │
└──────────────────────────────────────────────────────────────┘
            │  knows nothing about physics
            ▼
┌──────────────────────────────────────────────────────────────┐
│  Likelihood         (Gaussian / StudentT / GP / Composite)   │
│    ↳ scores Observation.predict(params) against data         │
└──────────────────────────────────────────────────────────────┘
            ▲
            │  reads dict of predictions
┌──────────────────────────────────────────────────────────────┐
│  ForwardModel       (the outer shell — talks to Observation, │
│                      owns Parameters, holds populations)     │
│    populations: tuple[Population, ...]                       │
│      each Population:                                        │
│        ├── sed     : SEDModel         (chain of SEDModelComponents)
│        └── spatial : SpatialModel     (chain of SpatialModelComponents,
│                                        or None for SED-only populations)
│                                                              │
│    .predict(params) → dict ───────────────────► Observation  │
└──────────────────────────────────────────────────────────────┘
            ▲
            │  threads ForwardState through components
┌──────────────────────────────────────────────────────────────┐
│  Components         (the atomic physics units)               │
│    SEDComponent      — Protocol (substrate, exists today)    │
│      ↳ SEDModelComponent — astronomer-facing base            │
│    SpatialComponent  — Protocol (substrate, new)             │
│      ↳ SpatialModelComponent — astronomer-facing base        │
│      each owns its parameters, precompute, apply(state)      │
└──────────────────────────────────────────────────────────────┘

  ┌─ Observation (sibling of ForwardModel, owned by it) ───────┐
  │   Photometry / Spectroscopy / Joint / Imaging / FiberSpec  │
  │   reads ForwardState, returns prediction dict              │
  └────────────────────────────────────────────────────────────┘
```

**The crisp rule that the whole architecture follows:** *physics lives
in components; instruments live in observation.* A fiber aperture
correction belongs in `FiberSpectroscopyObservation`, not in any
`SpatialComponent`. A per-age spatial profile (when it lands) belongs
in `SpatialComponent`, not in `ImagingObservation`.

## 2. Why this matters (the "physically correct joint spec-phot" story)

The single most common SED-fitting setup is photometry + a fiber
spectrum (SDSS, DESI, MOONS, MaNGA single-fiber extractions). The
spectrum samples *part* of the galaxy through the fiber footprint;
the photometry integrates the *whole* galaxy. Almost every public
SED-fitting code reconciles this by scaling the spectrum by a single
multiplicative factor — equivalent to assuming the galaxy is a uniform
slab. This is the flat-slab approximation.

A real galaxy has a spatial profile. The fiber sees the inner Sérsic
core; the broadband photometry sees core + envelope. The right answer
to the aperture mismatch is to model the spatial profile, integrate it
through the fiber footprint analytically (or numerically), and only
*then* compare to data. The architecture in this document makes that
joint fit a one-liner at the `ForwardModel` level rather than a hack
inside an SED component.

This is the primary scientific motivation for promoting spatial to a
first-class concept now rather than deferring it indefinitely. The
fact that the same machinery also unlocks IFU fitting later
(`docs/dev/spatial_model_extension.md`) is a bonus, not the reason.

## 3. Component contracts (the atomic units)

Two parallel Protocols sit at the bottom of the stack — one for
spectral physics, one for spatial physics. The Protocol is the
substrate; astronomers normally write subclasses of the convenience
base classes one level up.

> **Section 3.1 depends on the in-flight `SEDModelComponent` refactor.**
> The exact signature shown here may shift before that PR lands. The
> Protocol layer (`SEDComponent`) is stable.

### 3.1 `SEDModelComponent` — astronomer-facing SED base

```python
class ModifiedBlackbody(SEDModelComponent):
    name = "dust_ir"
    parameter_prefix = "dust_"

    # Free params — Distribution-typed class attrs, auto-discovered
    T    = Uniform(20.0, 80.0, "dust temperature",      units="K")
    beta = Uniform( 1.0,  3.0, "dust emissivity index", units="")

    # Cross-component contract
    reads     = {"L_absorbed": "erg/s"}
    publishes = {"L_ir": "erg/s"}

    def load(self, wave):                                # optional precompute
        return None

    def predict(self, p, sed_in, wave, *, L_absorbed):   # pure JAX
        addition = modified_blackbody_lnu(wave, L_absorbed, p["T"], p["beta"])
        L_ir = trapz_freq(addition, wave)
        return sed_in + addition, {"L_ir": L_ir}
```

- `p` is the parameter dict with prefix stripped.
- `sed_in` is the running rest-frame L_ν built up by upstream
  components (erg/s/Hz); zeros if this component runs first.
- `wave` is the rest-frame wavelength grid (Å). Under
  `approx=WavePrecomp()`, the framework calls `predict` a second
  time with filter effective wavelengths; the same function works
  for both paths.
- `**reads` kwargs are auto-supplied from `state.derived`.
- Return: `(sed_out, published)` — new rest-frame L_ν plus the dict
  of keys this component publishes.

### 3.2 `SpatialModelComponent` — astronomer-facing spatial base

Mirror-symmetric. Same Protocol shape, different state keys.

```python
class Sersic(SpatialModelComponent):
    name = "sersic"
    parameter_prefix = "spatial_"

    log_re_kpc  = Uniform(-1.0, 2.0, "log effective radius", units="dex(kpc)")
    n           = Uniform( 0.5, 6.0, "Sérsic index",         units="")
    axis_ratio  = Uniform( 0.1, 1.0, "axis ratio b/a",       units="")
    pa_deg      = Uniform(-90., 90., "position angle",       units="deg")

    reads     = {}
    publishes = {"spatial_profile_2d": ""}

    def load(self, grid_kpc):
        return None

    def predict(self, p, profile_in, grid_kpc):
        profile = sersic_profile_2d(grid_kpc, p["log_re_kpc"], p["n"], …)
        return profile, {"spatial_profile_2d": profile}
```

Concrete adapters shipped in v1: `Sersic`, `Exponential`,
`FlatSlab`, `BulgeDisk` (Sérsic + exponential, additive). Each is one
file under `components/spatial/`. Adding a new spatial profile is
exactly the same workflow as adding a new dust law — Protocol +
registry, no factory edits.

A `GPSpatialField` component (correlated-field prior over the spatial
plane) is the same Protocol with many free parameters declared via a
`CorrelatedField` prior, the same way stochastic SFH already works on
the spectral side. No special-casing required.

### 3.3 The B-seam — what spatial publishes today, what it can grow to

Today (A path): `SpatialComponent`s publish `spatial_profile_2d` only.
The profile is wavelength- and age-independent. A single Sérsic +
fiber aperture is enough to correctly handle the joint spec-phot
aperture mismatch.

Reserved (B path): `spatial_profile_per_age` (shape `(n_age, ny, nx)`)
and `spatial_profile_per_wave` (shape `(ny, nx, n_wave)`). These keys
are *named now* in the contract so that:

- A future `PerAgeSersic` component declares
  `reads = {"lnu_age": "erg/s/Hz"}` and the existing publish/require
  validator catches a missing producer at build time.
- Observation models can opt into the richer profile if available
  (`ImagingObservation` prefers per-age if present, falls back to
  uniform-colour 2D if not).

This is purely a contract-shape decision today. No code is written
against the B keys until colour gradients are needed.

## 4. Sub-model layer

Each sub-model is a thin composer over a list of components. They
all satisfy one 2-method Protocol:

```python
class SubModel(Protocol):
    name: str                                            # "sed", "spatial", "spatial_sed"

    def declared_parameters() -> list[ParamDeclaration]: ...    # aggregated
    def run(state, params) -> ForwardState: ...                 # pure JAX
```

`ForwardModel` doesn't care which sub-model class it's holding — only
that it satisfies `SubModel`. This is what makes the forked
stellar-atmospheres repo straightforward (see §7).

### 4.1 `SEDModel` — the spectral mode

```python
sed = SEDModel(components=[
    Stellar(...), DustAttenuation(...), NebularCue(...), IGM(...),
])
```

- Validates the `reads`/`publishes` graph via the existing
  `validate_pipeline` machinery (ADR-0009, ADR-0007 typed bundle).
- Aggregates `declared_parameters()` across components, enforces
  prefix discipline (`tools/check_param_prefixes.py`).
- `run(state, params)` threads `ForwardState` through components in
  topologically-valid order. Publishes `state.sed_intrinsic`,
  `state.sed_attenuated`, `state.sed_observed`, line dicts, derived
  diagnostics.

This is today's `SEDModel`, with the outer-shell responsibilities
factored out into `ForwardModel`.

### 4.2 `SpatialModel` — the spatial mode

```python
spatial = SpatialModel(components=[Sersic(...)])
```

Same validation + aggregation as `SEDModel`, on spatial components.
After `run`, `state.derived["spatial_profile_2d"]` is populated.

Spatial-only fits (morphology benchmarks, imaging-only Sérsic
fitting) work by handing `ForwardModel` a `SpatialModel` and an
`ImagingObservation` with no SED at all.

### 4.3 `SpatialSEDModel` — the joint mode (scientific main path)

```python
spatial_sed = SpatialSEDModel(
    sed     = SEDModel(components=[Stellar(...), Dust(...), Nebular(...)]),
    spatial = SpatialModel(components=[Sersic(...)]),
)
```

- Composes — does **no physics of its own** (~30 lines).
- `declared_parameters()` is the union, no shared params.
- `run(state, params)` calls `sed.run(...)` first, then
  `spatial.run(...)`.

**Order policy.** SED → Spatial. This permits `SpatialComponent`s to
read `state.derived` keys produced by SED components (mass-size
relation, per-age profiles). The reverse direction (SED components
reading spatial state — needed for spatially-varying attenuation or
spaxel-by-spaxel SFH fitting) is a known extension point reserved for
a future `ResolvedSEDModel` mode and is not supported in v1.

## 5. `ForwardModel` — the outer shell

The class inference talks to. Composes everything. Multi-population
from Day 1.

```python
@dataclass(frozen=True)
class ForwardModel:
    populations: tuple[Population, ...]
    observation: ObservationModel
    spec:        Parameters                # aggregated + namespaced

    @classmethod
    def build(cls, *, populations=None, sed=None, spatial=None,
              observation, **param_overrides) -> "ForwardModel":
        # populations=     → multi-population, explicit
        # sed=, spatial=   → single-population sugar; auto-wraps into one Population
        ...

    def predict(self, params) -> Mapping[str, jnp.ndarray]:
        # Run each population's sub-model into a per-population ForwardState,
        # collect into a populations-dict on the outer state,
        # then delegate to observation.predict, which sums in linear flux.
        ...
```

`ForwardModel.predict(params)` is the **only** API inference uses.
Whether the sub-model inside is `SEDModel`, `SpatialModel`, or
`SpatialSEDModel`, and whether there is one population or three, is
invisible at the inference layer. This is the JAX-purity story: the
Python object holds all structural state; `predict` is a pure function
of the traced `params` dict.

## 6. Multi-population — namespacing and summing

Three populations (AGN point source + Sérsic bulge + exponential
disc) is a realistic Day 1 case. The architecture has to support it
from the start because the parameter-naming decision is irreversible
once users have notebooks and saved fits.

### 6.1 Parameter names

Population name is the outer namespace; component prefix is the inner
namespace. Separator: `.`.

```
disc.sfh_dpl_alpha
bulge.sfh_dpl_alpha
agn.disc_log_lbol
```

The prefix CI guard (`tools/check_param_prefixes.py`) runs after
stripping the population namespace. Components remain unchanged.

### 6.2 State keys

Same convention for cross-population state:

```
disc.L_ir              published by disc dust
bulge.lnu_age          published by bulge stellar
agn.L_disc             published by agn disc — read by torus
```

The `publishes`/`requires` validator stays population-local by
default (a component cannot accidentally read another population's
state). Cross-population reads require explicit opt-in via a
fully-namespaced key in `reads`. This is a rare, advanced case (e.g.
AGN dust heating of the host disc) and should stay deliberate.

### 6.3 Summing

`Observation.predict` sums per-population contributions in **linear
flux**, not magnitudes, before returning the prediction dict. For
photometry: per-filter flux sum. For fiber spectroscopy: per-population
spatial integral inside the fiber × per-population SED, then sum.
Single-population fits skip the sum (degenerate one-element case).

### 6.4 The convenience kwargs

The common single-population case stays trivial:

```python
forward = ForwardModel.build(
    sed=SEDModel(components=[Stellar, Dust, Neb]),
    spatial=SpatialModel(components=[Sersic]),
    observation=...,
)
```

`build` wraps this into `populations=(Population("default", sed, spatial),)`
under the hood. Multi-population fits use the explicit form:

```python
forward = ForwardModel.build(
    populations=[
        Population("agn",   sed=SEDModel(...), spatial=SpatialModel([PointSource])),
        Population("bulge", sed=SEDModel(...), spatial=SpatialModel([Sersic(n=4)])),
        Population("disc",  sed=SEDModel(...), spatial=SpatialModel([Exponential()])),
    ],
    observation=...,
)
```

See [ADR-0012](../adr/0012-forward-model-population.md) for the full
namespace-collision rationale.

## 7. The forked stellar-atmospheres repo seam

Long-term, individual-star atmosphere fitting (Teff, log g, [Fe/H],
v sin i, v_rad, high-R spectroscopy) is planned as a separate
repository that forks `tengri`'s core. The architecture above makes
this a small lift:

The forked repo adds one new class —

```python
class StellarAtmosphereModel:                 # satisfies SubModel
    name = "atmosphere"

    def declared_parameters(self) -> list[ParamDeclaration]:
        return [Teff, log_g, FeH, vsini, vrad, ...]

    def run(self, state, params) -> ForwardState:
        # spectrum at high R from a stellar-atmosphere grid (Korg, FERRE,
        # The Cannon, or an NN emulator); write to state.sed_observed.
        ...
```

— and reuses `ForwardModel`, `Observation`, `Likelihood`, and all of
`inference/` unchanged. No core changes in `tengri` are required for
the fork to work. This is the cleanest expression of
"tengri-as-a-platform" in the architecture.

## 8. JAX purity — what's static, what's traced

A JAX-pure function isn't a function without associated data. It's a
function whose output depends only on its **traced inputs**.

| Kind | Lives where | When set | JAX-visible? |
|---|---|---|---|
| **Structural / static** | Python attribute on the object | Once, at `build` time | No — held as Python state |
| **Traced values** | Function argument to `predict` | Every call | Yes — flows through `jit`/`grad`/`vmap` |

The `ForwardModel` Python object can hold all the static structure
it wants — the populations, components, observation config, parameter
spec, filter curves, SSP grids. JAX never sees these because they are
not arguments to the traced function. The pure thing is
`predict(params) → dict`, where `params` is the JAX-traced dict.

In code:

```python
# Build time — eager Python. Components declare; ForwardModel collects.
forward = ForwardModel.build(...)
# forward.spec.free_params → [...]

# Inference time — pure JAX. Only `params` is traced.
@jax.jit
def loss(params):
    prediction = forward.predict(params)
    return -likelihood.log_prob(prediction)
```

The `forward` object is a Python closure variable. JIT compiles
against it the first time, then reuses. This is exactly how
`SEDModel` works today; nothing in the architecture changes the JAX
story.

## 9. Spatial model extension path (forward-looking)

Spatial in this architecture is what unlocks **joint spec-phot done
physically**, the simplest case being a Sérsic plus a fiber aperture
(§2). IFU is the natural generalisation, not the primary use case.

When the resolved / IFU case lands (post-v1), the new pieces are:

- `ResolvedSEDModel` — a new sub-model class that runs SED *after*
  Spatial, so SED components can read spatial keys. Different from
  `SpatialSEDModel`.
- `SpaxelImagingObservation` / `IFUSpectroscopyObservation` — new
  ObservationModel adapters that consume the per-age / per-wave
  spatial cubes.
- No core change to `ForwardModel`, `Population`, `SubModel`
  Protocol, or the component bases.

## 10. What changes vs the current codebase

Not a rewrite — a relayering. The bottom three layers (components,
observation, likelihood) already exist as Protocols. The work is at
the top:

| Layer | Today | Target |
|---|---|---|
| Outer shell | `SEDModel` (~2957 lines, mixes shell + chain) | `ForwardModel` (composer) + `SEDModel` (sub-model, slimmer) |
| Sub-model Protocol | Implicit | Explicit `SubModel` Protocol |
| Spatial | Sketch in `spatial_model_extension.md` | Implemented as a peer of `SEDModel` |
| Multi-population | Single | `populations: tuple[Population, ...]` |
| Parameter namespace | Flat `<prefix>_<param>` | `<population>.<prefix>_<param>` |
| Inference entry point | `SEDModel.predict` (today) → `Fitter` helpers | `ForwardModel.predict` (only) |

Implementation order, breaking changes, and migration story are out
of scope for this design doc — they will be captured in an
implementation plan after this design is approved.

## 11. Open dependencies

Before this document is treated as the canonical architecture:

- **`SEDModelComponent` refactor lands** (in-flight, separate PR by a
  parallel agent). Section 3.1 may need to be reconciled with the
  final signature. The Protocol layer (`SEDComponent`) is stable
  either way.
- **ADR-0012 ratified** — the multi-population namespace decision.
- **Naming sanity check** — `Population` is a generic word; verify
  it doesn't collide with existing astrophysics jargon in the
  codebase (e.g. stellar populations as a synonym for SSPs).
  `Population` vs `Component` vs `Region` vs `Source` is worth a
  5-minute check before code lands.

## References

- [ADR-0009](../adr/0009-typed-pipeline-contract.md) — Typed
  publish/require contract for cross-component data.
- [ADR-0010](../adr/0010-inference-backend-protocol.md) — Inference
  backend Protocol.
- [ADR-0012](../adr/0012-forward-model-population.md) —
  Forward-model populations and parameter namespacing.
- `docs/dev/NAMING_CONTRACT.md` — Free-parameter prefix discipline.
- `docs/dev/where-things-live.md` — Directory map.
