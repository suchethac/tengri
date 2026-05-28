# ADR-0016: Separate filter-integration from flux dimming in the photometry path

**Status:** Proposed (2026-05-26)

**Stakeholders:** Suchetha; AGN/nebular component maintainers; anyone touching `observation/photometry.py` or the `_phot_lnu_precomp` publish contract.

**Refs:** #398 (parent — DSPS observer-frame unification), #402 (the unified `shift_to_obs_frame` kernel), `docs/dev/audits/2026-05-26-redshift-sites-audit.md` (post-merge reassessment).

## Context

The audit for #398 surfaced one architectural smell that wasn't a wave-shift duplicate: **the AGN and nebular components apply a flux-dimming factor and immediately undo it** to produce the `_phot_lnu_precomp` tensors they publish for the precompute photometry path.

The pattern, verbatim from `components/agn/component.py:328–351`:

```python
# Filter-integrate L_agn at the source's z, dl_cm=1.
# compute_flux_density returns F_nu = (1+z)/(4π·dl²) · Lν_filter,
# so undo the cosmology factor to recover the bare rest-frame Lν
# — matches the convention of stellar_phot_lnu_precomp.
inv_cosmology = 4.0 * jnp.pi * 1.0**2 / (1.0 + z)
agn_phot_lnu_precomp = (
    jnp.asarray([
        compute_flux_density(L_agn, state.wave, fw, ft, redshift=z, dl_cm=jnp.asarray(1.0))
        for fw, ft in zip(self._state.filter_waves, self._state.filter_trans, strict=False)
    ])
    * inv_cosmology
)
```

`components/nebular/component.py:685–705` does the same.

### What the pattern does

1. Call `compute_flux_density(L_agn, wave, filter_wave, filter_trans, z, dl_cm=1)` — which computes:

   ```
   F_ν = (1+z) / (4π × 1²) × ∫ L_ν(λ_obs) T(λ) λ dλ  /  ∫ T(λ) λ dλ
   ```

2. Multiply by `inv_cosmology = 4π × 1² / (1+z)` to cancel the leading factor — leaving:

   ```
   L_ν_filter = ∫ L_ν(λ_obs) T(λ) λ dλ  /  ∫ T(λ) λ dλ
   ```

The output is the **filter-integrated rest-frame L_ν** — exactly what `stellar_phot_lnu_precomp` publishes. The cosmology cancellation is intentional, not a bug.

### Why this is a smell

- The components are forced to call `compute_flux_density` (which always returns F_ν) and then explicitly undo half its work.
- The intermediate `F_ν` value never reaches a consumer; it's pure ceremony to compensate for an over-broad helper.
- The "real" operation the components want — "filter-integrate this L_ν spectrum, respecting the wavelength redshift" — has no named function. Every component re-derives the inverse-cosmology dance.
- Future contributors adding a new component (say a radio or X-ray `_phot_lnu_precomp` publisher) have to either (a) read the existing two examples and copy the pattern, or (b) discover the smell mid-debug. Both are a tax on extensibility.
- It makes the audit of "where does the (1+z) factor live" harder than it needs to be — these sites are technically `(1+z)`-touching but the factor cancels exactly.

### Why it exists today

`compute_flux_density` was added before the precompute path needed filter-integrated L_ν as a publishable derived quantity. The stellar precompute (which publishes `stellar_phot_lnu_precomp`) bypassed the helper entirely with its own filter-integration loop. When AGN and nebular needed the same logical output, the natural-looking option was to reuse `compute_flux_density` and cancel the factor — cheaper than introducing a new helper, but architecturally noisy.

## Decision

**Factor out a `lnu_filter_integral` helper from `compute_flux_density`. The two functions become:**

```python
def lnu_filter_integral(
    L_nu_rest: jnp.ndarray,         # shape (..., n_wave)
    wave_rest: jnp.ndarray,         # shape (n_wave,)
    filter_wave: jnp.ndarray,       # shape (n_filt_wave,)
    filter_trans: jnp.ndarray,      # shape (n_filt_wave,)
    redshift: jnp.ndarray,
) -> jnp.ndarray:
    """Filter-weighted rest-frame L_ν on the observed-frame filter grid.

    Convention: ``∫ L_ν(λ_rest = λ_obs/(1+z)) T(λ_obs) λ_obs dλ_obs / ∫ T λ dλ``.
    Returns L_ν in erg/s/Hz — no cosmological dimming. Use ``lnu_to_fnu``
    or ``shift_to_obs_frame`` to go to F_ν.
    """
    wave_obs = wave_rest * (1.0 + redshift)
    L_on_filter = jnp.interp(filter_wave, wave_obs, L_nu_rest, left=0.0, right=0.0)
    num = jnp.trapezoid(L_on_filter * filter_trans * filter_wave, filter_wave)
    den = jnp.trapezoid(filter_trans * filter_wave, filter_wave)
    return num / jnp.maximum(den, 1e-30)


def compute_flux_density(
    L_nu_rest, wave_rest, filter_wave, filter_trans, redshift, dl_cm,
) -> jnp.ndarray:
    """Observed-frame F_ν through a filter — composed of the L_ν integral
    plus the standard ``lnu_to_fnu`` dimming."""
    L_filter = lnu_filter_integral(L_nu_rest, wave_rest, filter_wave, filter_trans, redshift)
    return lnu_to_fnu(L_filter, dl_cm, redshift)
```

**AGN and nebular `apply()` switch to `lnu_filter_integral` directly.** No more `inv_cosmology` ceremony. The `_phot_lnu_precomp` tensors are now computed by the function whose docstring matches the intent.

### Acceptance criteria

1. `components/agn/component.py` and `components/nebular/component.py` no longer reference `inv_cosmology` or pass `dl_cm=1.0` to a flux-dimming function.
2. Bit-exact output: `agn_phot_lnu_precomp` and `nebular_phot_lnu_precomp` produce numerically-identical photometry to the current code (formula-equivalent refactor). Regression test pinned at <1e-12 relative.
3. `compute_flux_density` continues to return identical F_ν for all existing callers (no signature change at its public boundary).
4. A unit test asserts the algebraic identity: `lnu_to_fnu(lnu_filter_integral(...), dl_cm, z) == compute_flux_density(..., dl_cm, z)` to floating-point precision.
5. The stellar precompute path's own inline filter-integration (the historical reason this helper didn't exist) is converted to use `lnu_filter_integral` in the same PR, so the convention has **one** definition site.

### Why now

- The unified kernel from #402 set the precedent: one canonical function per logical operation, with parity tests as the convention lock.
- The audit's reassessment (#405) showed this is the **only remaining structural cleanup** worth doing in the #398 work. Smaller migrations don't apply; bigger ones aren't needed.
- AGN + nebular precompute is on the path for any future component that wants to publish `_phot_lnu_precomp` (radio, X-ray re-emission). Fixing the convention before that growth keeps the smell from spreading.

## Consequences

### Positive

- **One named function per operation.** Filter-integration and flux-dimming are separately citable. The implicit dance becomes explicit composition.
- **AGN / nebular code shrinks ~25 lines each.** `inv_cosmology` and the per-filter loop with `dl_cm=1.0` collapse to a single `lnu_filter_integral` call inside a comprehension.
- **Documentation honesty.** The `_phot_lnu_precomp` publish contract becomes describable in one sentence ("the filter-weighted rest-frame L_ν") rather than "F_ν with cosmology factor inverted".
- **Future-proofs new precompute publishers.** Radio / X-ray re-emission components that want to publish their own `_phot_lnu_precomp` get the convention for free.
- **Audit-grep-friendly.** No more sites that look like `(1+z)` factors but cancel exactly.

### Negative

- One new public-ish helper to maintain. Mitigated: `lnu_filter_integral` is a thin function and `compute_flux_density` becomes its trivial composition.
- Existing callers of `compute_flux_density` see no change, but the test suite gains a small composition-identity test. Cheap.
- One regression test per migrated site (AGN, nebular, stellar precompute). Bit-exact, so the tests are short.

### Out of scope for this ADR

- **Cosmology-parameter handling.** `compute_flux_density` still takes `dl_cm` as an opaque scalar; whether to switch to a `cosmo`-based API is the territory of issue #401 and not changed here.
- **The `shift_to_obs_frame` kernel itself.** This ADR composes with it but does not modify it.
- **Precompute interface redesign.** The `_phot_lnu_precomp` publish/consume contract is unchanged — only the internal computation of what gets published changes.

## Alternatives considered

### A. Add an `apply_dimming: bool = True` flag to `compute_flux_density`

Callers needing rest-frame L_ν would pass `apply_dimming=False` and skip the `dl_cm` argument. **Rejected** — flag-based behavior is exactly the moving part we're trying to remove. `compute_flux_density` would still need to know about both code paths internally; the dance moves into the function instead of out.

### B. Pass `dl_cm = sqrt((1+z) / (4π))` to make the leading factor cancel

The current code uses `dl_cm=1.0` then multiplies by `4π/(1+z)`. An alternative is to pre-compute `dl_cm` to make the leading factor exactly 1. **Rejected** — opaque magic constant, harder to read than the explicit dance it would replace.

### C. Leave it. Document the pattern in `compute_flux_density`'s docstring

**Rejected** — the docstring would have to say "and here's how to call this if you don't want cosmology", which is the wrong factoring. Future contributors would still hit the smell. Net documentation tax is higher than introducing the helper.

### D. Define `lnu_filter_integral` privately inside `observation/photometry.py` and don't expose it

**Considered.** The helper is genuinely a building block, not a top-level user API. But the consequence is that AGN and nebular have to `from tengri.observation.photometry import _lnu_filter_integral`, which is the kind of private-import you regret when refactoring later. **Decided:** expose it. The function is small, well-typed, and has obvious physics meaning; it earns its public surface.

## Migration path

Single PR (#398.e):

1. Add `lnu_filter_integral` to `observation/photometry.py`.
2. Reduce `compute_flux_density` to its composition (`lnu_to_fnu(lnu_filter_integral(...))`).
3. Add the algebraic-identity test to `tests/contract/test_photometry_filter_integral.py`.
4. Migrate AGN's `apply()` — drop the `inv_cosmology` block, call `lnu_filter_integral` directly.
5. Migrate nebular's `apply()` — same surgery.
6. Audit the stellar precompute path (`components/stellar/sps/precompute.py`); if its inline filter-integration is bit-exact-equivalent, migrate it too. If it diverges in some convention detail (e.g. handling of zero-weight bins), document the divergence in a comment and leave it.
7. Add bit-exact regression tests for the three migrated `_phot_lnu_precomp` paths (rtol < 1e-12 vs the current main).

No public API change. No signature change at any public boundary. Pure internal refactor with one new helper exposed.
