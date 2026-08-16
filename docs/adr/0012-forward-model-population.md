# ADR-0012: Forward-model populations and parameter namespacing

- **Status:** Accepted (2026-05-22)
- **Stakeholders:** Suchetha; future contributors adding multi-component
  galaxy models (AGN + host, bulge + disc, mergers).
- **Related:** [Forward Model Architecture](../dev/archive/forward-model-architecture.md);
  ADR-0009 (typed publish/require contract); NAMING_CONTRACT §3.2
  (free-parameter prefix discipline).

## Context

`tengri` today fits a galaxy as a single physics chain: one stellar
component, one dust attenuation, one nebular block, optionally one
AGN. Parameter names are flat: `sfh_dpl_alpha`, `dust_tau_v`,
`agn_log_lbol`. The prefix discipline (NAMING_CONTRACT §3.2) is
enforced by `tools/check_param_prefixes.py`.

A real galaxy decomposition often needs multiple co-existing
populations:

- AGN as a **point source** (accretion disc + torus + NLR + corona)
  with its own SED and a point-source spatial profile.
- Host **bulge** with an older, more metal-rich stellar population
  and a Sérsic (n ≈ 4) spatial profile.
- Host **disc** with a younger, dust-attenuated stellar population
  and an exponential spatial profile.

Each population has its own SED chain and its own spatial profile.
At the observation, the three populations sum in linear flux —
through the same filters, through the same fiber footprint — before
the likelihood is evaluated.

Three stellar components in three populations means three
`sfh_dpl_alpha`s. The current flat prefix system collides.

## Decision

`ForwardModel` holds a tuple of populations from Day 1, and parameter
names are namespaced by population.

```python
@dataclass(frozen=True)
class Population:
    name: str                              # "agn", "bulge", "disc", …
    sed:     SEDModel
    spatial: SpatialModel | None = None    # None for SED-only populations
```

```python
@dataclass(frozen=True)
class ForwardModel:
    populations: tuple[Population, ...]
    observation: ObservationModel
    spec:        Parameters
    ...
```

### Parameter name shape

```
<population_name>.<component_prefix>_<param>
```

Examples:

```
disc.sfh_dpl_alpha
bulge.sfh_dpl_alpha
bulge.dust_tau_v
agn.disc_log_lbol
agn.torus_cos_theta
```

The separator is `.`. The population name is *outside* the prefix
discipline; the prefix CI guard runs on the part after the first `.`.
Components remain unchanged — they do not know which population they
are in.

### State (`ForwardState.derived`) keys

Same namespace convention applies to cross-component state keys
published into `state.derived`:

```
disc.L_ir              published by disc dust component
bulge.lnu_age          published by bulge stellar component
agn.L_disc             published by agn disc component → read by torus
```

The publish/require validator (ADR-0009) checks consistency within a
population by default. A component cannot accidentally read another
population's key — its `reads = {"L_ir": "erg/s"}` resolves to
`<my_population>.L_ir` automatically.

Cross-population reads are an opt-in advanced case (e.g. AGN dust
heating of the host disc). They use a fully-namespaced key in `reads`:

```python
class HostHeatedDust(SEDModelComponent):
    reads = {"agn.L_bolometric": "erg/s"}
```

This is deliberately verbose so cross-population coupling is always
explicit at the call site.

### Single-population convenience

The 90 % case is one population. The build API stays one-liner:

```python
forward = ForwardModel.build(
    sed=SEDModel(components=[Stellar, Dust, Neb]),
    spatial=SpatialModel(components=[Sersic]),
    observation=...,
)
```

`build` wraps this into `populations=(Population("default", sed, spatial),)`.
Parameter names omit the namespace when there is only one population —
`sfh_dpl_alpha`, not `default.sfh_dpl_alpha` — to preserve backward
compatibility with v0.x notebooks. Once a second population is added,
all names gain their namespace.

### Observation-side summing

`Observation.predict` runs each population's sub-model into a
per-population `ForwardState`, then sums per-population contributions
**in linear flux** before returning the prediction dict the likelihood
consumes. Photometric sum: per-filter `Σ_pop F_ν^pop`. Fiber-spec
sum: per-population spatial integral inside the fiber × per-population
SED, summed.

### CI guard changes

`tools/check_param_prefixes.py` learns one new rule: if a parameter
name contains `.`, strip everything up to and including the first
`.`, then apply the existing prefix check on the remainder.

## Rationale — why Day 1, not "later"

The parameter-naming decision is **irreversible** once notebooks and
saved fits exist in users' hands. Punting it costs migration work
later that lands square on top of users' work. The single-population
case stays trivial under the new scheme; only the architecture grows.

Alternatives considered:

- **Y. Single population for v1, multi as breaking change later.**
  Cheaper to ship; expensive to fix. Rejected.
- **Z. Allow multi-component at the spatial layer only — one big
  SEDModel with AGN + stellar physics, multiple spatial profiles
  with hand-coded knowledge of which spatial profile belongs to
  which SED block.** Smears coupling across two registries; the
  CIGALE failure mode. Rejected.
- **`__` (double underscore) as the namespace separator** instead of
  `.`. Python-attribute-friendly but visually noisy in fit summaries
  and posterior tables. `.` reads better and the dict lookup
  doesn't care. Adopted.

## Consequences

- `ForwardModel.build` accepts both `populations=` (explicit) and
  `sed=`/`spatial=` (single-population sugar).
- `Parameters` grows a population-aware view; flat-name access keeps
  working for single-population fits.
- The prefix CI guard learns the namespace-strip rule.
- The publish/require validator (ADR-0009) treats keys as
  population-local by default; fully-qualified keys are required for
  cross-population reads.
- `Population` is a new name in the codebase. Verify no collision
  with existing astrophysics jargon (in particular, stellar
  populations as a synonym for SSPs) before code lands.

## Status notes

- Proposed alongside the forward-model architecture redesign
  (2026-05-21); accepted and implemented 2026-05-22.
- Implementation: `ForwardModel.build(populations=[...])` accepts
  N > 1 populations; `ForwardModel.predict` slices parameters by the
  `"<name>."` prefix before handing each population's SubModel its
  slice; `JointObservation.predict_summed` runs each child observation
  on every population's state and sums the resulting channel dicts in
  linear flux; `tools/check_param_prefixes.py` strips the namespace
  before applying the prefix discipline.
- Implementation plan: `docs/internal/plans/2026-05-22-multi-population-namespacing.md`.
