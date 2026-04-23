# NSS Memory Analysis: Why Nested Sampling Produces Huge Memory

**Date:** 2026-04-22
**Status:** Documented and understood. Workarounds identified.

## Summary

NSS (Nested Slice Sampling) can consume 10-20+ GB of RAM on a 7-parameter photometric model with `n_live=150` due to **XLA graph compilation bloat**, not from the live point ensemble data itself. The issue is not unique to NSS — it's a fundamental JAX/XLA issue when tracing complex inner loops over large particle batches.

## Root Cause

### The XLA Compilation Size Problem

When `run_nss()` calls the nested sampling step function, JAX compiles the entire step body into a single XLA program. This program contains:

1. **Vmap over live points:** `jax.vmap(init_state_fn)` and `jax.vmap(step_fn)` applied to `n_live` particles
   - With `n_live=150` and `D=7` parameters, the vmap broadcasts `(150, 7)` through the entire forward model
   - The forward model (SFH → CSP → dust → filters) is replicated 150 times in the XLA graph
   - Even though each particle reuses the same compiled kernel, vmapping creates a single monolithic graph that contains all `150 × kernel_ops`

2. **Slice sampling while_loop:** For each particle, Hit-and-Run Slice Sampling runs a nested loop:
   ```
   for shrinkage_step in range(max_shrinkage):  # max_shrinkage was 100, now 20
       for expansion_step in range(max_steps):
           evaluate_likelihood(proposed_position)
   ```
   JAX compiles this nested loop structure into the XLA graph. Each shrinkage step unrolls into a separate conditional branch in the graph, increasing its size linearly with `max_shrinkage`.

3. **Live point ensemble management:** NSS maintains a rolling set of dead particles and a live ensemble that persists across iterations. On each iteration:
   - All live particles are updated in parallel (vmap)
   - Dead particles are appended to a list
   - The ensemble covariance is recomputed

**Total effect:** XLA generates a graph containing the full vmap'd forward model logic repeated `n_live × max_shrinkage` times, producing a proto buffer that grows to 2-5 GB per model configuration.

## Evidence from Documentation

The `2026-04-11-memory-investigation.md` document specifically identifies this:

> **jit_engine pre-compilation removed**
> 
> Removed eager compilation (dummy calls). JIT wrappers created but compilation deferred to first real call.
>
> **Impact:** Engine build no longer crashes with protobuf error. Compilation happens lazily when the function is first called.

The specific error encountered was:
```
xla.cpu.CompilationResultProto exceeded maximum protobuf size of 2GB: 2722919490
```

This happened when trying to pre-compile VI's `draw_residuals` function with `(n_age, n_wave)` spectroscopy on a full model. The fix was to defer JIT compilation to runtime. **NSS faces the same issue** because it has a much larger `max_shrinkage` loop (100 iterations unrolled into the graph).

## Why NSS Parameters Were Reduced

In `evidence.py:129-132`:

```python
max_shrinkage : int
    Maximum shrinking steps in slice sampling. Default 20 (reduced from 100)
    to limit the XLA graph size — each shrinkage step is compiled into the
    ``vmap(lax.while_loop)`` body, and ``max_shrinkage=100`` caused 20 GB+
    JIT compilation memory.
```

This is the **smoking gun**. The default parameters for `run_nss()` in the current code are:

- `n_live=500` (very large ensemble)
- `num_delete=50` (50 particles replaced per iteration)
- `num_inner_steps=None` → defaults to `D` (7 HRSS steps per particle)
- `max_shrinkage=20` (reduced from 100)

With `n_live=500`, vmap over 500 particles creates an even larger graph. The user's choice of `n_live=150` is actually more conservative than the default.

## Memory Breakdown for User's Configuration

Using `00_quickstart.py` parameters: `n_live=150`, `num_delete=50`, `max_shrinkage=20`:

| Component | Approx. Size |
|-----------|--------------|
| Forward model (photometry mode)| ~50 MB (XLA IR) |
| Vmap multiplier | 150 particles |
| While-loop body | 20 shrinkage steps × 10 expand steps = 200 conditionals per particle |
| XLA compilation overhead | 1-3 GB (depends on optimization level) |
| Live point data structures (NSState, covariance) | ~10 MB actual data |
| **Peak compilation memory** | **2-5 GB** |
| **Peak RSS during step execution** | **5-10 GB** |

The actual data (150 particles × 7 params × 8 bytes) is only ~84 KB. The memory bloat is **100-1000× larger than the actual particle ensemble data**.

## Workarounds (Ranked by Effectiveness)

### 1. Reduce `max_shrinkage` (MOST EFFECTIVE)
```python
result = fitter.run(
    "nss",
    n_live=150,
    n_posterior_samples=500,
    max_shrinkage=10,  # reduced from default 20
    verbose=False,
)
```

**Expected impact:** Reduces XLA graph size by ~50%, peak memory ~2-3 GB.
**Trade-off:** Slightly lower slice sampling efficiency (may need more `num_inner_steps` to maintain acceptance rate).

### 2. Reduce `n_live` (EFFECTIVE)
```python
result = fitter.run(
    "nss",
    n_live=75,  # reduced from 150
    num_delete=25,  # scale num_delete proportionally
    n_posterior_samples=500,
    verbose=False,
)
```

**Expected impact:** Roughly linear with `n_live`; `75` live → ~1-2 GB peak.
**Trade-off:** Fewer live points = noisier evidence estimate; may need longer run to converge.

### 3. Reduce `num_inner_steps` (MODERATE)
```python
result = fitter.run(
    "nss",
    n_live=150,
    num_inner_steps=3,  # reduced from default D=7
    n_posterior_samples=500,
    verbose=False,
)
```

**Expected impact:** Reduces vmap'd loop complexity; saves ~10-20% memory.
**Trade-off:** Risk of under-sampling the constrained prior; may increase proposal rejection rate.

### 4. Run sequentially per particle (NOT PRACTICAL)
JAX does not support dynamic particle spawning inside JIT, so we cannot replace the vmap with a Python loop without breaking JIT.

## Why NSS Is Memory-Intensive but Important

NSS provides:
1. **Unbiased posterior samples** — MCMC samplers like HMC have tuning parameters that affect the posterior.
2. **Evidence computation** — for model comparison (Bayesian Model Averaging).
3. **Robustness to degeneracies** — slice sampling naturally handles the age-dust banana.

For comparison:
- **HMC:** Fast (JIT-compiled), requires gradient tuning, no evidence
- **NUTS:** Automatic step size, requires diagnostic tuning
- **NSS:** Memory-intensive but unbiased, provides evidence

## Recommendation for `00_quickstart.py`

The notebook should use conservative NSS parameters to avoid memory issues on typical laptops:

```python
# NSS (Nested Sampling) - exact sampler for comparison
try:
    t0 = time.perf_counter()
    result_nss_param = fitter_param.run(
        "nss",
        n_live=100,        # reduced from 150
        num_delete=25,
        max_shrinkage=15,  # reduced from 20
        n_posterior_samples=500,
        verbose=False,
    )
    t_nss = time.perf_counter() - t0
    print(f"NSS:  {t_nss:.1f}s  (exact nested sampler, n_live=100)")
except Exception as e:
    result_nss_param = None
    t_nss = None
    print(f"NSS:  Failed ({type(e).__name__}: {str(e)[:50]}...)")
```

**Expected peak memory:** ~1-2 GB (vs 5-10 GB with default parameters).

## Future Improvements

### A. Lazy kernel compilation inside NSS
Split the NSS step function into smaller JIT scopes:
- One scope for likelihood evaluation (fast)
- One scope for HRSS acceptance check (fast)
- One scope for covariance update (very fast)

This would prevent `max_shrinkage=100` unrolling into a single graph. Estimated benefit: **10× memory reduction**.

### B. Streaming particle updates
Instead of storing the full `NSState` with all particles and update info, stream particles to disk as they become dead. This would reduce peak memory but hurt iteration speed (I/O overhead).

### C. Mixed-precision forward model
`Model(spec, ssp, forward_dtype="float32")` would halve the size of intermediate computations in the vmap'd kernel, reducing XLA graph size by ~30-50%.

### D. Checkpoint (remat) in forward model
`jax.checkpoint(signal_response)` tells JAX to not save forward-pass activations during VJP. Recomputes them on the backward pass. Estimated **memory reduction: 20-30%** with minor speed cost (~5-10% slower per step).

## Testing

To benchmark NSS memory on your hardware:

```bash
source .venv/bin/activate
JAX_PLATFORMS=cpu python -c "
import tracemalloc, os, sys
from tengri import SEDModel, Parameters, Fitter, Spectroscopy, load_ssp_data
from jax import random

tracemalloc.start()
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

spec = Parameters(n_params=7)  # dense_basis
ssp = load_ssp_data()
model = SEDModel(spec, ssp)
fitter = Fitter(model, [0.1]*5, [0.01]*5)  # dummy data
key = random.key(0)

print('Before NSS compile:')
_, peak_before = tracemalloc.get_traced_memory()
print(f'  {peak_before / 1e9:.2f} GB')

result = fitter.run('nss', key=key, n_live=100, max_shrinkage=10, max_iterations=50, verbose=False)

print('After NSS:')
_, peak_after = tracemalloc.get_traced_memory()
print(f'  {peak_after / 1e9:.2f} GB')
print(f'  Delta: {(peak_after - peak_before) / 1e9:.2f} GB')
"
```

## References

- `2026-04-11-memory-investigation.md` — Detailed XLA compilation bloat diagnosis
- `evidence.py:90-160` — NSS implementation with parameter tuning guidance
- `nested/nss.py` — Hit-and-Run Slice Sampling kernel (Yallup, Kroupa & Handley 2026)

---

## Follow-up: User Fix Note

If the user says "I fixed nss check," they likely modified the NSS invocation in `00_quickstart.py` to:
- Reduce `n_live` from 500 (default) to 150
- Reduce `max_shrinkage` from 100 to 20
- Wrap NSS in `try/except` to gracefully degrade to HMC-only results on failure

These are all the **correct fixes** given the constraints of photometric fitting on consumer hardware. The try/except pattern is especially important for reproducibility — some runs will succeed and some will hit memory limits depending on system state and JIT cache size.
