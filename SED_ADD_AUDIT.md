# SED-Add Audit

## Summary
Audit of `state.sed_intrinsic` conditional patterns in `src/tengri/components/`. Classifies occurrences as (a) simple-additive → migrate to `add_intrinsic()` helper, or (b) variation → skip with rationale.

## Migrable Sites (Simple Additive)

Pattern: `if state.sed_intrinsic is None: new_sed = L_component else: new_sed = state.sed_intrinsic + L_component`

### 1. `src/tengri/components/agn/component.py` line 183
```python
if state.sed_intrinsic is None:
    new_sed = L_agn
else:
    new_sed = state.sed_intrinsic + L_agn
state = state.with_(sed_intrinsic=new_sed, ...)
```
**Status**: MIGRATE. Pure additive, no initialization overhead.

### 2. `src/tengri/components/agn/grahsp/component.py` line 424
```python
if state.sed_intrinsic is None:
    new_sed = L_nu
else:
    new_sed = state.sed_intrinsic + L_nu
```
**Status**: MIGRATE. Pure additive, no initialization overhead.

### 3. `src/tengri/components/dust/emission_component.py` line 548
```python
if state.sed_intrinsic is None:
    new_sed = jnp.zeros_like(state.wave) + L_dust_lnu
else:
    new_sed = state.sed_intrinsic + L_dust_lnu
state = state.with_(sed_intrinsic=new_sed, ...)
```
**Status**: MIGRATE. The `jnp.zeros_like(...) + L_dust_lnu` in the None branch is mathematically equivalent to just `L_dust_lnu`; can be simplified in the helper or left as-is (redundant but harmless).

### 4. `src/tengri/components/radio/component.py` line 272
```python
if state.sed_intrinsic is None:
    new_sed = L_radio
else:
    new_sed = state.sed_intrinsic + L_radio
state = state.with_(sed_intrinsic=new_sed, ...)
```
**Status**: MIGRATE. Pure additive, no initialization overhead.

### 5. `src/tengri/components/nebular/component.py` line 388 (ternary, shock backend)
```python
new_sed = (
    nebular_sed if state.sed_intrinsic is None else state.sed_intrinsic + nebular_sed
)
state = state.with_(sed_intrinsic=new_sed, ...)
```
**Status**: MIGRATE. One-liner ternary form; logically identical to if-else block above.

### 6. `src/tengri/components/nebular/component.py` line 556 (photoionised backend)
```python
if state.sed_intrinsic is None:
    new_sed = nebular_sed
else:
    new_sed = state.sed_intrinsic + nebular_sed
state = state.with_(sed_intrinsic=new_sed, ...)
```
**Status**: MIGRATE. Pure additive, no initialization overhead.

### 7. `src/tengri/components/xray/component.py` line 200
```python
if state.sed_intrinsic is None:
    new_sed = L_xray
else:
    new_sed = state.sed_intrinsic + L_xray
state = state.with_(sed_intrinsic=new_sed, ...)
```
**Status**: MIGRATE. Pure additive, no initialization overhead.

## Skipped Sites (Variations)

### 1. `src/tengri/components/dust/two_component.py` line 324
```python
if state.sed_intrinsic is None:
    non_stellar_pre_dust = jnp.zeros_like(wave)
else:
    non_stellar_pre_dust = state.sed_intrinsic - sed_intrinsic_stellar
sed_total = non_stellar_pre_dust + sed_attenuated + sed_ir
state = state.with_(sed_intrinsic=sed_total, ...)
```
**Reason**: This does SUBTRACTION before addition (subtracts stellar from intrinsic, then adds back attenuated+IR). The helper doesn't apply cleanly because the None branch initializes to zero (not the component), then subtracts a reference stellar spectrum. Skip this variation; it's a special case that stays inline.

### 2. `src/tengri/components/dust/component.py` line 200
```python
if state.sed_intrinsic is None:
    return state
# ... (early return)
```
**Reason**: This is EARLY RETURN on None, not additive accumulation. It's validation logic ("if no intrinsic spectrum to attenuate, nothing to do"). The helper doesn't apply; keep as-is.

## Total Count

- **Migrable**: 7 sites
- **Skipped**: 2 sites (subtraction + early-return variants)
- **Post-migration expected grep count**: 2 remaining occurrences of `state.sed_intrinsic is None` (the skipped sites)

## Verification Gate

After migration:
```bash
grep -rc 'state.sed_intrinsic is None' src/tengri/components/
# MUST output exactly 2 (dust/two_component.py and dust/component.py)

grep -rn 'state.add_intrinsic(' src/tengri/components/
# MUST output exactly 7 (the migrated sites)
```
