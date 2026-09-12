# Cache key generation from object attributes

## Why

Every cache key in tengri used to be a hand-written field list. The same omission shipped four times: `n_subbands` missing from the z-table key (#1122), seven AGN and X-ray fields missing from the structural signature (#1462), the cosmology latent in the z-table key (#2145), and the live dust shape set missing from the structural signature (#2237). Each one silently served a cached computation to a model with different physics.

`Distribution.jit_cache_key` already solved this the other way round: iterate `vars(self)` and exclude a named set rather than include a named set. The `tengri._cache_keys` module generalizes that pattern to any object.

## The rule

An object's cache key is derived from **all** of its attributes under a written **policy ledger**. Each attribute is classified as one of three modes:

- **`content`** — hash the attribute's full value. Use this when the value decides which computation runs, or when a constant is baked into the compiled kernel.
- **`shape`** — hash only shape and dtype (for arrays only). Use this for per-galaxy data: the shape fixes the program structure, but the values are runtime input. `None` keys as `None` (an optional array attribute that is legitimately absent, e.g. `Spectroscopy.covariance`).
- **`exclude`** — omit the attribute from the key. Use this when an attribute is derived from another keyed attribute, or when it is a memo cache or internal working variable.

An attribute not classified in the policy is included by default in `content` mode and reported by `classify()`, so forgetting a classification fails a test instead of shipping silently wrong numbers.

Two failure modes to know:
- **Wrong `exclude`** (the #2163 bug class): two models that differ only in an excluded attribute share one compiled kernel, and the second silently runs the first's physics.
- **Wrong `content` on per-galaxy data**: the key changes with every galaxy's measurements, so a catalog fit recompiles the forward model per galaxy.

## What `baked` does to a value

The `baked()` function reduces a value to a hashable, equality-stable stand-in. The logic is:

| Value type | Result |
|---|---|
| `None`, bool, int, float, str, bytes | unchanged |
| tuple, list | tuple of baked items (recursively) |
| set, frozenset | tuple of sorted, baked items |
| dict, MappingProxyType | tuple of sorted (key_str, baked_value) pairs |
| object with `cache_key()` | `("cache_key", type_qualname, cache_key())` |
| object with `jit_cache_key()` | `("jit_cache_key", type_qualname, jit_cache_key())` |
| frozen dataclass | `(type_qualname, ((field_name, baked_value), …))` |
| `functools.partial` | `("partial", baked(func), baked(args), baked(keywords))` |
| closure (non-empty `__closure__`) | `("callable", "module.qualname", tuple(baked captured content))` |
| bound method (non-class `__self__`) | `("callable", "module.qualname", baked(instance))` |
| plain function, class, builtin | `("callable", "module.qualname")` |
| array-like (shape + dtype) | array-specific: `("array", shape_tuple, str(dtype), digest)` — see below |
| Enum member | `("enum", type_qualname, member_name)` |
| custom `__repr__` | `("repr", type_qualname, repr(value))` |
| otherwise | `TypeError` — no address-bearing keys, ever |

**Array hashing:** JAX arrays (immutable, identified by `id()`) are memoized via weakref for O(1) memo hits; the cached key is reused if the weakref is alive and `is` holds. NumPy arrays are mutable and are hashed on every call (a modified array yields a different key). The digest is BLAKE2b (deterministic across Python processes, unlike `hash()` which is per-process salted).

**Closures and partials:** Closures (functions with a non-empty `__closure__` capturing constants), `functools.partial` objects, and bound methods are keyed by their **captured content or instance state**, never by address. Two closures from the same factory with different captured constants produce different keys—this prevents the silent misuse bug where a second `Fitter(..., extra_log_prior=hook)` with a hook from one factory shares a compiled loss with a first `Fitter` whose hook captured a different threshold. Bound methods include the instance's state (via its `cache_key()` or `__repr__`), so two calls to the same method on different instances produce different keys. Empty cells in a closure (an error to access) key as `("empty_cell",)`. If a closure captures a value that cannot be baked (no `cache_key()`, `__repr__`, or other fallback), the TypeError names the type and propagates—an intentional design: a hook capturing an unkeyable object must expose `cache_key()` to be usable in a cache.

## The tail

Session state no object holds — the JAX x64 flag, the backend name — is passed explicitly as `tail=` when calling `derive_key()`, as named pairs that are baked and appended after the attribute rows:

```python
key = derive_key(
    self,
    policy,
    tail=(("x64", bool(jax.config.jax_enable_x64)), ("backend", jax.default_backend())),
)
```

The `Parameters` key also carries the fixed and free parameter NAME sets in its tail because the prior objects themselves are excluded from the policy (the bounds and fixed values are runtime inputs of the structural kernel; the Fitter engine key carries them, #1972).

## Where the ledgers live

The canonical ledgers and their locations:

| Object | Module | Constant |
|---|---|---|
| `FilterCurve` | `observation.photometry` | `_FILTER_CURVE_CACHE_KEY_POLICY` |
| `Photometry` | `observation.photometry_config` | `_PHOTOMETRY_CACHE_KEY_POLICY` |
| `Spectroscopy` | `observation.spectroscopy` | `_SPECTROSCOPY_CACHE_KEY_POLICY` |
| `Observation` | `observation.observation` | `_OBSERVATION_CACHE_KEY_POLICY` |
| `LineFluxData` | `observation.line_flux_data` | `_LINE_FLUX_DATA_CACHE_KEY_POLICY` |
| `LineRatioData` | `observation.line_ratio_data` | `_LINE_RATIO_DATA_CACHE_KEY_POLICY` |
| `SpectralIndexData` | `observation.spectral_indices` | `_SPECTRAL_INDEX_DATA_CACHE_KEY_POLICY` |
| `LineList` | `observation.line_list` | `_LINE_LIST_CACHE_KEY_POLICY` |
| `DoubletConstraint` | `observation.line_list` | `_DOUBLET_CONSTRAINT_CACHE_KEY_POLICY` |
| `NoiseModel` | `observation.noise_model` | `_NOISE_MODEL_CACHE_KEY_POLICY` |
| `SSPData` | `components.stellar.sps.dsps_wrapper` | `_SSP_CACHE_KEY_POLICY` |
| `Parameters` | `parameters.parameters` | `_PARAMETERS_CACHE_KEY_POLICY` |
| `ZTableRequest` | `components.stellar.sps.precompute` | version constant + all fields |
| `SubbandRequest` / `SubbandBandRequest` | `components.igm._subband_cache` | version constant + all fields |
| `SEDModel.compile_signature()` | `forward._signature_policy` | `SIGNATURE_POLICY` (+ `SIGNATURE_VERSION`) |
| `Fitter._engine_cache_key()` | `inference._engine_policy` | `ENGINE_POLICY` (+ `ENGINE_VERSION`) |
| `_data_fingerprint()` | `inference._engine_policy` | `FINGERPRINT_POLICY` (+ `FINGERPRINT_VERSION`) |
| MCMC backend adaptation `method_key` (`nuts`/`hmc`/`dynamic_hmc`/`chees`/`ghmc`/`mclmc`/`adjusted_mclmc`/`first_order`) | `inference.backends.mcmc._shared` | `adaptation_method_key()` + `_ADAPT_IRRELEVANT` |
| `PreconditionedProblem.cache_key` | `inference.preconditioning` | `_PRECONDITIONED_PROBLEM_CACHE_KEY_POLICY` |

**The engine key and the data fingerprint are complementary, not independent** (#2163 E.5). `Fitter._engine_cache_key()` and `_data_fingerprint()` key the SAME object (a `Fitter`) for two different purposes: the engine key identifies the compiled *program* (which loss function two Fitters can share), the fingerprint identifies the *target* (which cached MAP or tuned adaptation two Fitters can share). Every Fitter attribute is `content`/`shape` in **at most one** of `ENGINE_POLICY` / `FINGERPRINT_POLICY`: the four per-galaxy arrays (`data`, `noise`, `data_mask`, `presence`) are `shape` rows in the engine key (only the length matters for compiling) and `content` rows in the fingerprint (the values decide where a MAP or warmup lands); `_runtime_redshift` is excluded from the engine key (it rides `data_args` as a traced input, #1316) and is the fingerprint's fifth content row for exactly that reason. Everything else the engine key needs as content (spec, free names, calibration/e-line settings, the per-fit override, …) is structure, so the fingerprint excludes it. `adaptation_method_key()`'s `_ADAPT_IRRELEVANT` ledger is a third, narrower kind of exclusion list: it does not classify a whole object's attributes, only which of a sampler runner's OWN keyword arguments do not change what warmup tunes (`init_from`, `n_samples`, `n_chains`, …), replacing a hand-built tuple every backend had separately maintained and extended by name.

**Frozen dataclasses as request objects:** `ZTableRequest`, `SubbandRequest`, and `SubbandBandRequest` are frozen dataclasses whose every field is a key dimension. Each has a `version` constant (bump it when the table structure changes). All fields are hashable (tuples, arrays via `array_key`, scalars, strings).

**`SEDModel.compile_signature()`:** the model's structural ledger, memoized on the instance and invalidated by the two structural mutators `_validate_and_freeze_param_map` and `enable_fast_nebular`. Two `SEDModel` instances with the same structure (observation layout, parameter structure, component configurations, precision) share one compiled kernel; prior bounds and fixed values are not structure and live in the Fitter engine key.

## How to classify a new attribute

When you add an attribute to a class with an existing ledger:

1. Add the attribute to the class.
2. Run the policy-complete test for its class in `tests/contract/test_nested_cache_keys.py` (or `tests/contract/test_compile_signature_invariants.py` for the model). It names the unclassified attributes.
3. **Decide the mode** by answering two questions in order:
   - *Does the compiled program change when this value changes?* → `content`
   - *Is this a per-galaxy measurement?* → `shape`
   - *Is it computed from something already keyed, or a memo cache?* → `exclude` (name the origin)
4. Add the row to the policy dict with a one-line reason.
5. Add a test to verify the row does what it claims:
   - `content`: perturb the value, verify the key changes.
   - `shape`: same shape with different values, verify the key is equal; different shape, verify the key changes.
   - `exclude`: verify the key is equal whether the attribute is present or absent (or present but changed).

## Disk caches

Frozen dataclasses used as request keys (`ZTableRequest`, `SubbandRequest`, `SubbandBandRequest`) follow this pattern:

- **Every field** is a key dimension and must be hashable.
- **One version constant** per cache (`_ZTABLE_CACHE_VERSION`, `_CACHE_VERSION` in the subband module, `_IONSPEC_CACHE_VERSION` for the ionizing-spectrum table). Bump it with any change in what the stored table means; every existing entry is then recomputed once.
- **Perturbation test:** the test suite derives its population from `dataclasses.fields()` and perturbs each field, verifying the cache key changes.
- **Cosmology as a tripwire** (#2145): `ZTableRequest` includes `cosmology: tuple` (baked `DEFAULT_COSMO` or equivalent). Changing any cosmology constant invalidates every cached z-table.

## See also

- `docs/dev/where-things-live.md` — the `src/tengri/` layout
- `docs/performance/compilation.md` — compile time and cache reuse
