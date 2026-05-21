# Testing with chex

`chex` is DeepMind's JAX-native testing toolkit. We use it for three things:

1. **Shape assertions** that work on traced arrays without forcing concretisation.
2. **Finiteness checks** over PyTrees in one call, instead of per-leaf loops.
3. **Tree-allclose comparisons** for whole `PipelineState.derived` dicts and per-component outputs.

It is a **test-only** dependency. Never `import chex` from `src/tengri/`.

## When to reach for chex

| Want to check | Use |
|---------------|-----|
| One array has a specific shape | `chex.assert_shape(x, (n_wave,))` |
| Two arrays share a shape | `chex.assert_equal_shape([a, b])` |
| A PyTree contains no NaN/Inf | `chex.assert_tree_all_finite(tree)` |
| Two PyTrees agree within tolerance | `chex.assert_trees_all_close(a, b, rtol=1e-6)` |
| One scalar within tolerance | leave as `np.testing.assert_allclose` — chex adds nothing |

## Tolerances

We share three tolerance conventions across the suite:

| Context | rtol | Where |
|---------|------|-------|
| JIT-vs-eager parity (bit-for-bit modulo XLA) | `1e-6` | `tests/integration/test_variant_parity.py`, JIT/eager pairs |
| Cross-library crossval (bagpipes / FSPS / Cue) | `1e-3` | `tests/crossval/test_*` |
| Scalar physics (Eddington L, virial masses, etc.) | `1e-2` | individual tests, keep `assert_allclose` |

Pass `atol` only when the expected value can pass through zero (e.g. derived
luminosities at off-band wavelengths). For dimensionless ratios `rtol` is
sufficient.

## Conversion recipes

### Shape + finiteness

```python
# BEFORE
assert result.shape == wavelength.shape
assert jnp.all(jnp.isfinite(result))

# AFTER
chex.assert_equal_shape([result, wavelength])
chex.assert_tree_all_finite(result)
```

### Manual relative-error loop

```python
# BEFORE
numerator = np.abs(a - b)
denominator = np.maximum(np.abs(a), np.abs(b), 1e-30)
np.testing.assert_array_less(numerator / denominator, 1e-3,
                             err_msg="precompute vs runtime drift")

# AFTER
chex.assert_trees_all_close(a, b, rtol=1e-3, atol=1e-30,
                            custom_message="precompute vs runtime drift")
```

### Per-key derived-state loop

```python
# BEFORE
for k in expected.derived:
    np.testing.assert_allclose(actual.derived[k], expected.derived[k],
                               rtol=1e-6)

# AFTER
chex.assert_trees_all_close(actual.derived, expected.derived, rtol=1e-6)
```

## JIT-vs-eager variant tests

The canonical pattern lives in `tests/integration/test_variant_parity.py`.
For ad-hoc one-offs:

```python
import chex, jax, pytest

@pytest.mark.parametrize("variant", ["eager", "jit"])
def test_my_component_finite(variant, component, params, state0):
    run = jax.jit(component.apply) if variant == "jit" else component.apply
    chex.assert_tree_all_finite(run(state0, params))


def test_my_component_jit_matches_eager(component, params, state0):
    eager = component.apply(state0, params)
    jitted = jax.jit(component.apply)(state0, params)
    chex.assert_trees_all_close(eager, jitted, rtol=1e-6)
```

Use the `@pytest.mark.parametrize("variant", ...)` form rather than
`chex.TestCase` / `@chex.variants`. The rest of the suite is plain pytest
functions and the parametrize form composes cleanly with our session fixtures.

## What chex will *not* fix

- Scalar comparisons — `np.testing.assert_allclose(x, expected, rtol=...)`
  stays exactly as it is.
- Physics-content assertions (line ratios, integrals, normalisation) — these
  are domain checks, not infrastructure checks.
- Compile-time regressions. `chex.assert_max_traces` exists but is flaky
  under our shared JIT cache; not in use today.

## Anti-patterns

- Importing `chex` from `src/tengri/`. Forbidden — chex is a test dep only.
- Wrapping forward-pass code in `chex.assert_*` guards. Even disabled they
  clutter the call graph; assertions live in tests.
- Using `chex.assert_trees_all_close` on trees whose leaves are Python
  scalars — fall back to `np.testing.assert_allclose`.
