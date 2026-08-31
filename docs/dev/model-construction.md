# Model construction — the one narrative

This is the canonical guide to how a `tengri` SED model is built: the
user-facing construction API, the single dispatch table that turns a grammar
choice into a physics component, and the one recipe for adding a new physics
variant. It supersedes the scattered "how do I build/extend a model" notes that
predate the ADR-0011/0019 migration.

- **Building a model** (user-facing): [The build path](#the-build-path).
- **How a grammar choice becomes physics** (internals): [One registry, one
  dispatch](#one-registry-one-dispatch).
- **Adding a new physics variant** (contributor): [The add-a-model
  recipe](#the-add-a-model-recipe).
- **Where a given knob lives**: [`where-things-live.md`](where-things-live.md).

The design decisions behind this page are recorded in
[ADR-0011](../adr/0011-sed-model-component-base.md) (the `SEDModelComponent`
base) and [ADR-0019](../adr/0019-unified-component-dispatch.md) (one authoring
unit, one dispatch table, no shape taxonomy).

---

## The build path

A model is a `SEDModel`. The **recommended** way to construct one is
`SEDModel.build(...)`, the nested-dict grammar shipped in 2026-05
(`parameters/groups.py`): one dict per physics group, each declaring a
structural `type`, an `'all_params'` free/fixed wildcard, and per-parameter
overrides.

```python
from tengri import SEDModel, FREE, Fixed, DEFAULT, Uniform, recipes

# From a curated recipe (each recipe's docstring states its SSP requirement):
model = SEDModel.build(ssp_data=ssp, observation=obs,
                       **recipes.star_forming_photometry())

# Or hand-rolled with the nested-dict grammar:
model = SEDModel.build(
    ssp_data=ssp, observation=obs,
    sfh={'type': 'dpl', 'all_params': FREE, 'beta': Uniform(1, 3)},
    dust_attenuation={'type': 'two_component', 'law': 'calzetti',
                      'tau_bc': 0.5, 'other_params': Fixed(DEFAULT)},
    dust_emission={'type': 'dale2014', 'all_params': Fixed(DEFAULT)},
    neb={'type': 'cue', 'all_params': Fixed(DEFAULT)},
    shock={'norm': 'frac', 'frac': Uniform(0, 1)},   # composes with neb
    redshift=Fixed(0.05),
)

model.spec.summary()          # provenance-tagged: [user]/[all_params FREE]/[all_params Fixed(DEFAULT)]/[default]
groups = model.spec.to_groups()   # round-trip back to the grammar for editing
```

### Grammar rules

- **Groups** are the physics blocks: `sfh`, `stellar`, `dust`, `neb`, `shock`,
  `agn`, `igm`, `radio`, `xray` (plus top-level settings `redshift`,
  `apply_igm`, `n_grid`).
- Each group dict accepts:
  - `'type'` — the structural choice (which variant), validated against the
    domain's registered names.
  - `'all_params'` — the wildcard: `FREE` or `Fixed(DEFAULT)` (default
    `Fixed(DEFAULT)`). Its exact synonym `'other_params'` is preferred once the
    group also has explicit per-parameter entries (`'other_params'` written
    last, meaning "the others"); giving both spellings in one dict raises. The
    wildcard cascades over the group's parameters. For groups whose bucket
    params default to `Fixed` (e.g. `radio`, `shock`), `'all_params': FREE` is
    a no-op — use explicit priors instead (`shock={'frac': Uniform(0, 1)}`).
  - **Per-parameter short-forms** — a bare parameter name inside the group
    resolves to the full prefixed name (`'beta'` in the `sfh` group →
    `sfh_dpl_beta`; `'frac'` in `shock` → `shock_frac`). The full prefixed
    name also works.
  - **Other structural keys** — a few groups accept extra non-parameter
    settings. The `sfh` group takes `'bin_edges_gyr'` (non-parametric bin
    layout) and `'age_kernel'` (below).
- **`sfh={'age_kernel': ...}`** picks how the SFH is integrated onto the SSP
  age grid — the one place the two implementations differ numerically:
  - `'cic'` (the default) evaluates the SFH on a 16x denser integrand and
    splits each `SFR(t)·dt` parcel between its bracketing SSP nodes with
    log-age cloud-in-cell weights.
  - `'dsps'` hands the coarse per-SSP-age table to DSPS's histogram kernel.
    It **zeroes the first SSP node older than the SFH start** (3.8 % of the
    mass for a delayed-tau at age = 5 Gyr) and biases the optical CSP +1.2 %
    vs FSPS / bagpipes / a dense reference ([#964]). Offered for cross-code
    comparison against DSPS-native pipelines and pre-#964 tengri, not for
    science. Pre-#964 equivalence is **exact**, verified against the pre-fix
    source: same `sfr_on_ssp`, same `_build_dsps_sfh_table(...,
    add_young_knot=True)`, same `.weights`, same #821 youngest-bin multiplier.
    The one deliberate difference is a `jnp.maximum(sum, 1e-300)` floor on the
    normalization, so a degenerate all-zero SFH yields zero instead of NaN.
  - Leaving it unset auto-selects: `'cic'` on the parametric path, `'dsps'`
    on the GP-field path (whose draw lives on its own coarse grid, so there
    is no dense integrand to cloud-in-cell). Asking for `'cic'` together with
    a field SFH raises rather than silently returning DSPS weights.
  - It is **not** a speed knob, and `'dsps'` is the slower of the two.
    Measured on `predict_photometry` gradients (interleaved reps, medians, an
    A/A control to fix the noise floor): `'cic'` is **3.5 % faster on the exact
    path** and **13 % faster under `WavePrecomp()`**.

    The cause is **not** that DSPS does more arithmetic — by compiled-HLO cost
    analysis it does ~1 % *fewer* FLOPs and touches fewer bytes. It compiles to
    **twice as many `while` loops** (14 vs 7 exact, 13 vs 6 under WavePrecomp)
    and ~40 % more fusion regions: sequential, latency-bound work that does not
    vectorize. Precompute shrinks the vectorizable part (cic fusions 356 → 212)
    but leaves the loops alone (dsps whiles 14 → 13), so DSPS's fixed
    sequential share grows and the gap widens. This is a **CPU wall-clock**
    effect driven by op structure, so the ordering is not guaranteed to hold on
    GPU — re-measure there rather than assuming.

    Do **not** judge this by micro-benchmarking
    `compute_dsps_age_weights` — that helper has no call sites on the model
    path, so its timing says nothing about `apply()`.

[#964]: https://github.com/suchethac/tengri/issues/964
- **Sub-blocks** nest a dict with its own `'type'`/`'all_params'`/per-param keys:
  `dust.emission`, and the six composable AGN selectors `agn.disc`,
  `agn.torus`, `agn.nlr`, `agn.blr`, `agn.feii`, `agn.atten` (the deprecated
  `agn.lines` alias expands to an `nlr`/`blr` pair)
  ([ADR-0018](../adr/0018-composable-agn-grammar.md)).
- **Sentinels** `FREE` / `DEFAULT` are singletons exported from `tengri`.
  `FREE` defers a parameter to the registry's default prior; `DEFAULT` is
  legal only as `Fixed(DEFAULT)`, pinning a parameter at the registry default
  value. The old `FIXED` sentinel is removed (pre-1.0 break, no shim); pin a
  parameter with `Fixed(v)` for your own value or `Fixed(DEFAULT)` for the
  registry default.
- **Recipes** (`tengri.recipes.*`) are five curated starting points —
  `star_forming_photometry`, `quiescent_z0`, `agn_panchromatic`,
  `stochastic_sfh_jwst`, `mock_recovery_minimal`.

### The expert escape hatch

The flat-kwarg `Parameters(...)` constructor is still supported and is what the
grammar lowers to internally, but it is **not** the recommended user surface.
Prefer `SEDModel.build`; reach for `Parameters(...)` only when you need a knob
the grammar does not yet expose.

See [`api_migration_v0.x.md`](api_migration_v0.x.md) for the full grammar
reference and `notebooks/04_building_models.py` for a worked example.

---

## One registry, one dispatch

Under the hood, `SEDModel.build` lowers the grammar to a `Parameters` spec, and
the forward model turns that spec into an **ordered list of physics
components** via `forward/component_factory.py:build_components()`. This is the
single seam where a grammar `type` string becomes a concrete component:

```
grammar type  ──►  _resolve_registry_component(domain, type)  ──►  _REGISTRY[type]  ──►  component instance
```

- **`SEDModelComponent` is the single authoring unit.** Its
  `__init_subclass__` auto-discovers class-attribute `Distribution` priors into
  `_priors`, registers `_REGISTRY[name] = cls` (collision-checked), and wires
  the `inputs`/`optional_inputs`/`outputs` dicts into typed `DerivedKey` tuples
  (validated by [ADR-0009](../adr/0009-typed-pipeline-contract.md)).
- **`_REGISTRY` is the single dispatch table.** `_resolve_registry_component`
  looks the `type` up in `_REGISTRY` at **construction time only** — never
  traced through JAX ([ADR-0010](../adr/0010-inference-backend-protocol.md)).
  A miss **raises** with the list of registered names; there is no silent
  fallback.
- **No shape taxonomy.** There is no "emitter vs screen vs composite" base
  hierarchy. Whether a component adds light (`sed_in + emission`) or multiplies
  transmission (`sed_in * T`) is one line in its `predict()`.
- **Pipeline order is derived, not hand-coded.** The orchestrator
  topologically sorts components by their declared `inputs`/`outputs` so, e.g.,
  dust runs after the nebular + shock components it reddens.

### What stays orthogonal (not in `_REGISTRY`)

These are **data/function registries**, not physics-component dispatch, and
remain by design (ADR-0019 §6):

- `DUST_LAWS` — pure `k(λ)` attenuation functions with no free params.
- `SFH_REGISTRY` / `MET_REGISTRY` — builder-driven, live in `parameters/`.
- `RADIO_MODELS` / `XRAY_MODELS` — variant catalogs consulted *by* the
  registered radio/xray components (the component dispatches through
  `_REGISTRY`; the catalog just names its internal variants).

---

## The add-a-model recipe

For any model with free parameters, a wavelength-dependent emission or
transformation, and optionally a pre-computed library or trained emulator —
attenuation laws, dust IR libraries, AGN torus libraries, nebular emulators,
shock templates — write **one file** subclassing `SEDModelComponent`:

```python
class MyModel(SEDModelComponent):
    name = "my_model"               # registry key → grammar type='my_model'
    parameter_prefix = "my_"

    T    = Uniform(20.0, 80.0, "temperature", units="K")
    beta = Uniform(1.0, 3.0, "emissivity index", units="")

    inputs  = {"L_absorbed": "erg/s"}   # cross-component reads (optional)
    outputs = {"L_ir": "erg/s"}         # cross-component publishes (optional)

    def load(self, wave):               # optional: load atlas/weights → self.data
        return None

    def predict(self, p, sed_in, wave, *, L_absorbed):
        sed = my_emission_formula(wave, L_absorbed, p["T"], p["beta"])
        return sed_in + sed, {"L_ir": trapz_freq(sed, wave)}
```

`__init_subclass__` does the rest: it discovers the class-level priors,
registers `(name, cls)` so `SEDModel.build(dust_emission={'type': 'my_model'})` finds
it, fills `inputs()`/`outputs()` from the dicts, and provides sensible default
`apply()`/`precompute()`. The astronomer writes physics only.

**The contract:**

- `p` — parameter dict, **prefix stripped** (`p["T"]`, not `p["my_T"]`).
- `sed_in` — rest-frame Lν from upstream (erg/s/Hz); zeros if this is the first
  emitter.
- `wave` — rest-frame grid in Å (or filter effective wavelengths under
  `WavePrecomp`).
- `**inputs` — keyword args auto-supplied from `state.derived`.
- Return `(sed_out, published)` — the new SED plus a dict matching the
  `outputs` keys. Publish only names that are typed `DerivedState` fields (the
  strict spillover guard rejects undeclared extras).

**Then** import the module in the domain `__init__.py` so the class body runs
and registers at package import (a registered-but-unimported component is a
silent no-op — this is exactly how the shock component went dead before #851).

### Full references

- [`sed-model-components.md`](sed-model-components.md) — full how-to with three
  worked examples (closed-form, library, NN emulator).
- [`forward-model-architecture.md`](archive/forward-model-architecture.md) —
  architectural context for the forward pipeline.
- [ADR-0011](../adr/0011-sed-model-component-base.md) — the base-class design.
- [`src/tengri/components/dust/wg00_model.py`](../../src/tengri/components/dust/wg00_model.py)
  — canonical small closed-form component.
- [`src/tengri/components/agn/skirtor_model.py`](../../src/tengri/components/agn/skirtor_model.py)
  — canonical template-library component.

### The advanced fallback

Reserve the bare `SEDComponent` Protocol (five-method) path for models that do
not fit the `predict(p, sed_in, wave, **inputs)` shape — typically Stellar
(SFH + SSP + age weights + nine derived publishes) and IGM (observer-frame
transformation). The pattern lives at `src/tengri/protocols/component.py`; the
canonical reference is `src/tengri/components/radio/component.py`.
