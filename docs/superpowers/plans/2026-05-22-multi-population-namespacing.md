# Multi-population namespacing (ADR-0012) implementation plan

> Item #3 of the post-tracer-bullet architecture sequence. Closes the
> last load-bearing piece of the architecture spec
> (`docs/dev/archive/forward-model-architecture.md` §6 + ADR-0012).

**Status:** Plan only. **Do not start implementation until the user reviews this document.** The parameter-namespace decision is irreversible once users have notebooks and saved fits in this format; the user should validate the namespace separator, error-message wording, and `ForwardModel.build` ergonomics before code lands.

**Depends on:** Items #1, #2, #4, #5, #6 (all merged or in flight). Specifically requires `Population` (#169), `SpatialModel`/`SpatialSEDModel` (#185), and `ForwardModel.build(spatial=...)` (#188).

**Goal:** A user can write

```python
forward = ForwardModel.build(
    populations=[
        Population("agn",   sed=SEDModel.build(...), spatial=SpatialModel(components=[PointSource()])),
        Population("bulge", sed=SEDModel.build(...), spatial=SpatialModel(components=[Sersic(n=4)])),
        Population("disc",  sed=SEDModel.build(...), spatial=SpatialModel(components=[Exponential()])),
    ],
    observation=...,
)
params = {
    "agn.disc_log_lbol":   44.5,
    "bulge.sfh_dpl_alpha": 4.0,
    "disc.sfh_dpl_alpha":  1.5,
    "redshift":            0.05,
}
forward.predict(params)
```

and the prediction is the linear-flux sum of all three populations.

## Open design choices (REQUIRES USER REVIEW)

### 1. Separator: `.` vs `__` vs `:`

The ADR-0012 default is `.`. Alternatives:

- **`.`** (current ADR) — reads naturally in posterior tables, easy to type, distinct from prefix-internal `_`.
- **`__`** (double underscore) — Python-attribute-friendly (could enable `params.disc__sfh_dpl_alpha`), visually noisier.
- **`:`** — common in scientific notation but clashes with dict-key parsing in some YAML configs.

**Recommendation: `.`** — matches ADR-0012; the only friction is that some existing code may treat parameter names as identifier-like (use them as variable names). A grep of the codebase for `params[<name>]` patterns will confirm no string-to-identifier conversions are silently broken.

### 2. Back-compat: omit namespace when N=1?

ADR-0012 §6.4 commits to "parameter names omit the namespace when there is only one population." Two reads:

- **(a)** Single-pop fit: `sfh_dpl_alpha` (no prefix); multi-pop fit: `disc.sfh_dpl_alpha`. The user pays the namespacing cost only when they need it.
- **(b)** Always-namespaced, even for `Population(name="default")`. Cleaner but adds friction to the 90% case.

**Recommendation: (a)** per the ADR. Single-population users see no change.

### 3. Cross-population reads — verbose or implicit?

When a component in one population reads a derived key from another (e.g. dust in the disc heated by AGN bolometric luminosity), the ADR makes it **explicit**:

```python
class HostHeatedDust(SEDModelComponent):
    reads = {"agn.L_bolometric": "erg/s"}
```

Confirm this is the desired ergonomics (vs e.g. an implicit "broadcast" of derived keys from a designated reference population).

**Recommendation: per the ADR.** Explicit cross-pop reads make the data flow visible at the call site.

## Out of scope (post-v1)

- IFU / spaxel-by-spaxel multi-population (each spaxel its own population) — needs different state shape.
- Population types beyond SED+Spatial (e.g. transient supernova population overlaid on a galaxy host). The framework supports it; concrete adapters land separately.

---

## File structure

| File | Action | Responsibility |
|---|---|---|
| `src/tengri/forward/forward_model.py` | Modify | Accept >1 populations, sum at observation layer, namespace params on the way in. |
| `src/tengri/forward/population.py` | Modify | Validate `name` (no `.`, non-empty); add `prefix_param_dict(params)` helper that produces `{"<name>.<orig_name>": value}`. |
| `src/tengri/observation/joint_observation.py` | Modify | New `predict_summed(per_pop_states, params)` method for linear-flux sum across populations. |
| `tools/check_param_prefixes.py` | Modify | Strip everything up to and including the first `.` before applying the prefix check. |
| `src/tengri/parameters/parameters.py` | Modify | `Parameters` learns a `populations` kwarg or accepts namespaced names. |
| `docs/adr/0012-forward-model-population.md` | Update | Status from Proposed to Accepted with the implementation date. |
| `docs/dev/where-things-live.md` | Modify | Multi-population entry. |
| `CHANGELOG.md` | Modify | Multi-population entry. |
| `notebooks/20_multi_population_decomposition.py` | Create | The AGN+bulge+disc demo notebook. |
| New tests | Create | `tests/unit/forward/test_multi_population.py` |

---

## Task breakdown

### Task 1: Population validates its name

`name` must be non-empty (already enforced) and must not contain `.` (would break the namespace split).

```python
def __post_init__(self) -> None:
    if not self.name:
        raise ValueError("Population.name must be a non-empty string.")
    if "." in self.name:
        raise ValueError(
            f"Population.name {self.name!r} contains '.'; this is reserved "
            f"as the namespace separator. Use underscores or hyphens."
        )
```

Tests: existing 4 + 1 new for the `.` rejection.

### Task 2: ForwardModel.build accepts N>1 populations

Drop the `len(pops) > 1` `NotImplementedError`. Add validation:
- All `Population.name`s must be distinct (no duplicates).
- The observation must have a `predict_summed` method (or fall back gracefully — see Task 4).

```python
if len({p.name for p in pops}) != len(pops):
    duplicates = [n for n in (p.name for p in pops) if list(p.name for p in pops).count(n) > 1]
    raise ValueError(f"Population names must be distinct; got duplicates: {set(duplicates)}")
```

Tests: 2-pop construction passes; duplicate-name raises; single-pop convenience kwargs (`sed=`, `spatial=`) still produces 1-element tuple.

### Task 3: ForwardModel.predict threads multi-pop

Per architecture spec §6, each population runs into its own state; the observation sums in linear flux:

```python
def predict(self, params):
    per_pop_states: dict[str, ForwardState] = {}
    for pop in self.populations:
        pop_params = self._slice_params_for_population(params, pop)
        full_params = self._merge_fixed_values(pop_params, pop)
        state = ForwardState(wave=jnp.zeros(1))
        state = pop.sed.run(state, full_params)
        if pop.spatial is not None:
            state = pop.spatial.run(state, full_params)
        per_pop_states[pop.name] = state

    if len(per_pop_states) == 1:
        # Convenience: single-pop path stays flat
        (only,) = per_pop_states.values()
        return self.observation.predict(only, params)

    return self.observation.predict_summed(per_pop_states, params)
```

The `_slice_params_for_population` helper: strip `<name>.` prefix from any namespaced param; pass non-namespaced params (like `redshift`) through unchanged.

```python
def _slice_params_for_population(self, params, pop):
    prefix = f"{pop.name}."
    out = {}
    for k, v in params.items():
        if k.startswith(prefix):
            out[k[len(prefix):]] = v
        elif "." not in k:
            out[k] = v  # cross-population (e.g. redshift)
    return out
```

Tests:
- 3-pop fit: each population sees only its own namespaced params + bare names.
- Param `disc.sfh_dpl_alpha` ends up as `sfh_dpl_alpha` inside the disc's sed.run.
- Bare names like `redshift` reach every population unchanged.
- Cross-population reads (e.g. `agn.L_bolometric` consumed by `disc.dust`) work via the typed bundle's namespace-aware lookup (Task 5).

### Task 4: JointObservation.predict_summed

The composer gains a `predict_summed` method:

```python
def predict_summed(self, per_pop_states, params):
    """Sum each child observation's prediction across populations.

    For each child observation, run its predict on each population's
    state, then sum the resulting per-population dicts key-by-key
    (linear flux sum). Returns the merged dict.
    """
    summed: dict[str, jnp.ndarray] = {}
    for child in self.children:
        per_pop_pred = {name: child.predict(state, params) for name, state in per_pop_states.items()}
        # Linear flux sum across populations, per channel key
        all_keys = set().union(*(p.keys() for p in per_pop_pred.values()))
        for key in all_keys:
            contributions = [p[key] for p in per_pop_pred.values() if key in p]
            if key not in summed:
                summed[key] = sum(contributions[1:], contributions[0])
            else:
                # Children disagree on this key — last one wins for now;
                # may want to error in the future.
                summed[key] = sum(contributions[1:], contributions[0])
    return summed
```

For non-JointObservation observations that don't have `predict_summed`, `ForwardModel.predict` can synthesize one by summing children manually. Simplest is to require all multi-pop observations to expose `predict_summed`; document this in the Observation Protocol.

Tests:
- 2-pop, photometry-only: phot_fnu = phot_pop_a + phot_pop_b
- 2-pop, joint phot+spec: each channel sums correctly
- 2-pop, FiberSpec: each pop's spectrum is aperture-scaled before the sum (correct — the fiber sees the aperture-weighted sum of all populations)

### Task 5: Typed-bundle namespace-aware lookups

Currently `DerivedBundle.get("L_ir", 0.0)` is a direct field lookup. For multi-population, components in one population should be able to read keys from other populations via namespaced names. Two approaches:

**A.** Have `ForwardModel.predict` populate each population's state with **all** populations' derived keys (with namespaced names like `agn.L_bolometric`) before running components. Components that have `reads = {"agn.L_bolometric": "erg/s"}` find the key normally.

**B.** Add a namespace-aware `get` helper on `DerivedBundle` that splits on `.` and routes to a per-population sub-bundle.

**Recommendation: A.** Simpler, no new bundle API surface, and the per-population state already has a `derived` attribute that can hold the namespaced keys.

Implementation: after running each population, copy its `derived` keys into a shared multi-pop derived bundle with namespaced names; merge that into each population's state.derived before observation.predict.

Tests:
- Disc dust component declares `reads = {"agn.L_bol": "erg/s"}` and the AGN population publishes `L_bol` — the disc receives the AGN's value.
- A component reading an unnamespaced key (e.g. its own population's `L_ir`) still works.

### Task 6: Parameter-prefix CI guard learns the namespace strip

`tools/check_param_prefixes.py`: if a parameter name contains `.`, strip everything up to and including the first `.` before applying the existing prefix check.

Tests: add a synthetic component test that registers a namespaced free param and verify the guard accepts it.

### Task 7: ADR-0012 → Accepted

Update `docs/adr/0012-forward-model-population.md` status from Proposed → Accepted with the merge date.

### Task 8: where-things-live + CHANGELOG entries

Standard documentation updates.

### Task 9: Multi-population demo notebook

`notebooks/20_multi_population_decomposition.py`: AGN + bulge + disc decomposition, joint MAP fit (or just aperture-fraction demo if SSP data unavailable), comparison to single-pop fit. Shows the recovered bulge and disc stellar masses are degenerate when the spatial profile is ignored but separable when it's modeled.

### Task 10: Self-review + push + PR

Standard.

---

## Risk assessment

**Highest risk:** Task 3 + Task 5 — the param-slicing and derived-key namespacing have many edge cases. Suggested mitigations:
- Comprehensive unit tests (Task 3 lists ~5 cases per sub-test).
- Run on the existing single-population fixtures to ensure no regression.

**Second-highest:** Task 6 — the prefix CI guard runs on every PR; if it's too strict (or too lenient), every contributor feels it. Suggested mitigation: add 4-5 test cases covering edge cases (namespaced names, bare names, names with multiple dots, empty namespaces).

**Lowest:** Task 7, 8 — pure documentation.

---

## Self-review checklist

- [ ] Single-population fits remain backward compatible (no parameter renames).
- [ ] Multi-population fits work with both `populations=[...]` explicit and via convenience kwargs (the latter still produces N=1).
- [ ] Cross-population derived-key reads work via `reads = {"<pop>.<key>": "<units>"}` syntax.
- [ ] Parameter-prefix CI guard accepts namespaced names.
- [ ] ADR-0012 status updated.
- [ ] No existing test fails.
- [ ] Demo notebook runs.
