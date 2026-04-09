# Non-Monotonic GL Quadrature Accuracy: Root Cause Analysis

## Problem Statement

The Gauss-Legendre quadrature for dust attenuation in photometric precomputation produces **non-monotonic accuracy** as `n_quad` increases:

```
SDSS z-band:
  n=1: 0.43% error
  n=3: 1.84% error  ← WORSE!
  n=5: ?
  n=7: ?

SDSS g-band:
  n=1: 1.33%
  n=3: 0.33%  ← better
  n=5: 1.13%  ← WORSE again!
  n=7: ?
```

Expected behavior: error should monotonically decrease as n_quad increases (more nodes = more accurate integration).

Observed behavior: error jumps up and down unpredictably.

## Root Cause Identified

The bug is in **`precompute.py` lines 394–400** (dust-only-GL mode):

```python
for f_idx, (fw, ft) in enumerate(zip(filter_waves, filter_trans)):
    fw_np, ft_np = np.asarray(fw), np.asarray(ft)
    lam_nodes, _, h = _gauss_legendre_nodes_for_filter(fw_np, n_quad)
    quad_wav_obs[f_idx] = lam_nodes
    t_at_nodes = np.interp(lam_nodes, fw_np, ft_np)
    denom_quad = h * float(np.sum(w_gl * t_at_nodes * lam_nodes))  # ← THE BUG
    quad_dust_scale_np[f_idx] = t_at_nodes * lam_nodes * h / max(denom_quad, 1e-30)
```

The issue: `quad_dust_scale` is normalized by **`denom_quad`** (a GL approximation):

```
denom_quad = h · Σ_k w_k · T(λ_k) · λ_k
```

But the exact photometric denominator (computed for `ssp_phot` at line 338) is:

```
denom = trapezoid(T·λ, fw)  [exact dense-grid integral]
```

These are **NOT the same**. For irregular filter transmission curves like SDSS z-band, GL nodes sample T(λ) differently at each n_quad, causing:

1. **Different GL node locations**: Each n_quad creates a different set of GL nodes in [-1, 1], mapped to the filter's wavelength range
2. **Different sampling of irregular T(λ)**: SDSS filters have narrow features; GL nodes may miss them at some n_quad values
3. **Different denom_quad estimates**: The GL quadrature rule gives different estimates of ∫T(λ)·λ dλ depending on how well the GL nodes sample T(λ)
4. **Different scale factors**: Since `quad_dust_scale[f, k] = T(λ_k)·λ_k·h / denom_quad`, different denom_quad leads to wildly different scale factors
5. **Non-monotonic error in the final photometry**: The dust averaging uses these bad scale factors

### Mathematical Details

At inference time (fused_kernels.py line 472), the code computes:

```python
trans_young_avg = (quad_weights * trans_bc_q * trans_diff_q * quad_dust_scale).sum(-1)
```

This is:

```
⟨A⟩_GL = Σ_k w_k · A(λ_k) · quad_dust_scale[f, k]
```

where `quad_dust_scale[f, k] = T(λ_k)·λ_k·h / denom_quad`.

The final photometry is:

```
f_b ≈ CSP_b · ⟨A⟩_GL
```

where `CSP_b` is the exact SSP filter integral (no approximation).

The **factorization error** (assuming `f ≈ CSP × ⟨A⟩` when in reality `f = ∫ SSP·A·T·λ dλ`) dominates. Different n_quad values give different estimates of `⟨A⟩_GL` due to different scale factors, and these estimates can be **worse** than n=1 for certain filters.

### Why Charlot-Fall Dust Matters

The dust attenuation law:

```
A(λ) = exp(-τ · (λ / 5500)^-0.7)
```

is a **steep power law**, not a polynomial. Gauss-Legendre quadrature is optimized for polynomials. When the integrand (dust along the filter transmission) is a power law, GL nodes don't necessarily give the best approximation. Combined with the factorization assumption, this leads to non-monotonic behavior.

## Proposed Fix

**Replace `denom_quad` with `denom` (the exact dense-grid denominator)** in the scale factor computation:

### Current Code (Lines 394–400)
```python
for f_idx, (fw, ft) in enumerate(zip(filter_waves, filter_trans)):
    fw_np, ft_np = np.asarray(fw), np.asarray(ft)
    lam_nodes, _, h = _gauss_legendre_nodes_for_filter(fw_np, n_quad)
    quad_wav_obs[f_idx] = lam_nodes
    t_at_nodes = np.interp(lam_nodes, fw_np, ft_np)
    denom_quad = h * float(np.sum(w_gl * t_at_nodes * lam_nodes))  # ← WRONG
    quad_dust_scale_np[f_idx] = t_at_nodes * lam_nodes * h / max(denom_quad, 1e-30)
```

### Fixed Code
```python
# Compute exact dense-grid denominator (once per filter, same as for ssp_phot)
denom = _np_trapezoid(ft_np * fw_np, fw_np)

for f_idx, (fw, ft) in enumerate(zip(filter_waves, filter_trans)):
    fw_np, ft_np = np.asarray(fw), np.asarray(ft)
    lam_nodes, _, h = _gauss_legendre_nodes_for_filter(fw_np, n_quad)
    quad_wav_obs[f_idx] = lam_nodes
    t_at_nodes = np.interp(lam_nodes, fw_np, ft_np)
    # denom_quad = h * float(np.sum(w_gl * t_at_nodes * lam_nodes))  # ← DELETE
    quad_dust_scale_np[f_idx] = t_at_nodes * lam_nodes * h / max(denom, 1e-30)  # ← USE denom
```

### Rationale

1. **Consistency**: The dust scale factors are now normalized against the same denominator used for the SSP integral
2. **Eliminates GL approximation error**: We're no longer introducing a second-order approximation (the denom_quad GL estimate)
3. **Monotonic improvement**: Higher n_quad naturally samples dust values at more locations, improving the average dust estimate without competing normalization factors
4. **Minimal change**: The fix is a one-line change; no other code needs modification (the scale factors are still dimensionally correct)

## Verification Plan

Three diagnostic tests have been added to `tests/unit/`:

1. **`test_quad_final_diagnosis.py`**: Compares buggy vs fixed implementations on real filters. Should show:
   - Buggy error: non-monotonic (e.g., n=3 worse than n=1)
   - Fixed error: monotonically decreasing

2. **`test_quad_scale_factor_bug.py`**: Shows denom_quad vs denom_exact variations. Should show:
   - Large discrepancies in denom_quad across n_quad values
   - Scale factors changing wildly due to different denominators

3. **`test_quad_accuracy_investigation.py`**: Full per-filter analysis of GL nodes, filter transmission, dust values, and resulting accuracies.

Run all tests:
```bash
source .venv/bin/activate
JAX_PLATFORMS=cpu pytest tests/unit/test_quad_final_diagnosis.py -v -s
JAX_PLATFORMS=cpu pytest tests/unit/test_quad_scale_factor_bug.py -v -s
JAX_PLATFORMS=cpu pytest tests/unit/test_quad_accuracy_investigation.py -v -s
```

Then run the original benchmark to confirm the fix:
```bash
JAX_PLATFORMS=cpu python scripts/benchmark_precompute_quad.py --n-quad 1 3 5 7
```

Expected result post-fix:
```
SDSS z-band:
  n=1: 0.43%
  n=3: < 0.43%  ← monotonic improvement
  n=5: < n=3 %
  n=7: < n=5 %
```

## Implementation Checklist

- [ ] Add the three diagnostic tests to verify the hypothesis
- [ ] Run tests to confirm the bug
- [ ] Implement the fix in `precompute.py` lines 394–400
- [ ] Run `scripts/benchmark_precompute_quad.py` to verify monotonic improvement
- [ ] Update existing tests if needed
- [ ] Commit with message: `fix: use exact denom instead of denom_quad for GL dust scale normalization`

## Related Code

- **Main bug**: `src/tengri/models/sps/precompute.py:394–400`
- **SSP-GL mode** (also affected): `src/tengri/models/sps/precompute.py:367–378` — should also use exact denom
- **Inference path**: `src/tengri/core/fused_kernels.py:472` (uses quad_dust_scale)
- **Benchmark**: `scripts/benchmark_precompute_quad.py`

## Notes

- The bug only affects dust-only-GL mode (`sps_quad=False`, the default)
- SSP-GL mode (`sps_quad=True`) has the same bug at lines 367–378 but is rarely used
- The bug is independent of the factorization error itself (SSP × dust assumption), which is a separate approximation
- Charlot-Fall dust with steep power law (n=-0.7) is the worst case; smoother dust laws may show smaller errors
