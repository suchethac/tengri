# Testing Contract

This document is the contract for what counts as a meaningful test in
`tengri`. Every new test must satisfy it. PRs that add tests outside
this contract should be sent back at review.

The bar: a test exists because **a physicist would want it to**, not
because an LLM wrote it for the sake of coverage.

## Physics-first taxonomy

Every test under `tests/physics/`, `tests/regression/`, or
`tests/components/` MUST declare exactly one of these markers, either
at the file level (`pytestmark = pytest.mark.<marker>`) or on the
function. CI enforces this via `tools/check_test_markers.py`.

| Marker | What it asserts | Required in docstring |
|---|---|---|
| `conservation` | Energy / mass / photon balance across a transformation (dust-absorbed = dust-emitted; ∫SFR dt = M\*; ∑photons in = ∑photons out within tol). | The conservation law (one line). |
| `bounds` | Non-negativity (L_ν ≥ 0), unit interval (0 ≤ T_dust ≤ 1), monotonicity (τ_λ falls with λ in attenuation regime), credible-interval ordering. | The bound + why it holds physically. |
| `limit` | Zero-input → known output: τ_BC=0 → intrinsic SED; f_AGN=0 → stellar only; z=0 → rest frame; t→0 → initial condition. | The limit + the analytical answer. |
| `regression_paper` | Numerical match (within stated tol) to a published equation or table. | Paper citation + equation number + tolerance. |
| `regression_bug` | Frozen output for a previously-fixed bug. | ADR ID / commit SHA / issue link. |
| `gradient` | `jax.grad` is finite, has correct sign, and (where applicable) matches finite differences within tol. | The parameter being differentiated + expected sign. |
| `crossval` | Match against bagpipes / FSPS / Synthesizer within stated tol. Already gated by `-m crossval`. | The reference code + version + tolerance. |
| `contract` | Public-API surface check, deprecation alias still resolves, kernel registry round-trip. One file per surface, **not** per symbol. | The surface being protected. |

The existing markers `slow` and `benchmark` remain unchanged and are
orthogonal to the taxonomy above.

## Anti-patterns — REJECT AT REVIEW

A test exhibiting any of these is a `request-changes` until it's
removed or rewritten:

1. **Private-attribute assertions.** `assert model._compositional is
   not None` — test the public effect instead.
2. **Shape-only assertions.** `assert out.shape == (N,)` without any
   value check. Always pair shape with at least one physics assertion.
3. **Self-mocking.** Mocking the code path you claim to exercise.
4. **Parametrize explosions over trivial value variations.** Sixteen
   `@parametrize` cases that all exercise the same code path. Keep one
   representative plus the boundary cases.
5. **Import-only files.** Stand-alone `test_can_import_X.py` /
   `test_can_instantiate_X.py`. Collapse into one `contract` test per
   public surface.
6. **Fixture round-trips.** Setting a value in a fixture and asserting
   `getter() == that_value`. Tautology.
7. **Signature introspection.** `inspect.signature(...).parameters`
   checks. Brittle to kwarg reordering; type checkers do this better.
8. **Tests of constants.** `assert MIN_ENTRY_SIZE_BYTES == 0` when
   the test file itself sets it to 0 on import.

## Good test recipes

### Conservation

```python
import pytest
import jax.numpy as jnp

pytestmark = pytest.mark.conservation


def test_dust_absorbed_equals_dust_emitted(model, params):
    """Energy absorbed by dust must re-radiate (Kirchhoff)."""
    intrinsic = model.predict_intrinsic(params)
    attenuated = model.predict_attenuated(params)
    dust_emission = model.predict_dust_emission(params)

    absorbed = jnp.trapz(intrinsic - attenuated, model.nu)
    emitted = jnp.trapz(dust_emission, model.nu)

    # 1% tolerance: numerical quadrature on a coarse rest-frame grid
    assert jnp.allclose(absorbed, emitted, rtol=1e-2)
```

### Limit

```python
import pytest
import jax.numpy as jnp

pytestmark = pytest.mark.limit


def test_zero_tau_recovers_intrinsic_sed(model, params):
    """τ_BC=0 and τ_ISM=0 must return the intrinsic stellar SED exactly."""
    params_no_dust = params.update(dust_tau_bc=0.0, dust_tau_ism=0.0)
    attenuated = model.predict_attenuated(params_no_dust)
    intrinsic = model.predict_intrinsic(params_no_dust)
    assert jnp.allclose(attenuated, intrinsic, rtol=1e-6)
```

### Paper regression

```python
import pytest

pytestmark = pytest.mark.regression_paper


def test_calzetti2000_attenuation_eq3():
    """Reproduce Calzetti et al. 2000, ApJ 533, 682 — Eq. 3, k(λ).

    Tolerance: 1% (analytical formula, no fit residual).
    """
    from tengri.dust import calzetti_k

    # k(0.55 µm) = 4.05 by construction (Eq. 4 normalization)
    assert abs(calzetti_k(5500.0) - 4.05) < 0.05
```

### Gradient

```python
import pytest
import jax

pytestmark = pytest.mark.gradient


def test_grad_sed_wrt_logmstar_is_positive(model, params):
    """∂L_ν / ∂log M* > 0 — more mass means more light."""

    def loss(logmstar):
        return model.predict(params.update(logmstar=logmstar)).sum()

    g = jax.grad(loss)(10.0)
    assert jnp.isfinite(g)
    assert g > 0
```

## Tree layout

```
tests/
  contract/          # public-API surface, deprecation, registry round-trip
  physics/
    conservation/    # ∫SFR = M*, dust absorbed = dust emitted, photon counts
    bounds/          # L_ν ≥ 0, monotonicity, credible-interval ordering
    limits/          # zero-input recovers analytical answer
    gradients/       # jax.grad sanity + finite-difference checks
  regression/
    paper/           # cite paper + equation
    bug/             # cite ADR / commit / issue
  components/        # per-block physics (sfh/, dust/, agn/, nebular/, …)
  inference/         # backend conformance, posterior arithmetic
  integration/       # full-pipeline scenarios
  crossval/          # gated by `-m crossval`
  conftest.py
  TESTING.md
```

## Reviewer checklist

When reviewing a PR that adds or modifies tests:

- [ ] Every new test file has a `pytestmark` (or per-function marker).
- [ ] No assertion touches a `_private` attribute.
- [ ] Every `shape ==` assertion is paired with a physics assertion.
- [ ] `@parametrize` cases are distinct code paths or named boundaries.
- [ ] No `inspect.signature` checks.
- [ ] No `test_can_import_X` / `test_X_is_callable` files.
- [ ] `regression_paper` tests cite a paper + equation in the docstring.
- [ ] `regression_bug` tests cite an ADR / commit / issue.
- [ ] Tolerances are physically motivated, not "whatever makes the test pass".
