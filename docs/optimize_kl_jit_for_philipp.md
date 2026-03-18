# JIT-Compiled optimize_kl: 500x Speedup for Small Problems

**Summary for Philipp Frank** — March 2025

## The Finding

For small-to-moderate problems (D~137), NIFTy's `optimize_kl` spends
**99.8% of its time in Python overhead**, not in linear algebra.
By reimplementing the geoVI algorithm (same math) using only JAX
primitives (`jax.lax.scan`, `jax.lax.while_loop`, `jax.vmap`),
we achieved a **500x speedup** on the optimization loop and
**100,000x** on posterior sample drawing.

## Problem Context

- Differentiable SED fitting with DSPS (galaxy spectral energy distributions)
- D = 137 parameters (9 physical + 128 GP latent vector `ξ`)
- 5 photometric data points (SDSS ugriz)
- Forward model: 0.04ms (JIT'd) — **not the bottleneck**

## Profiling Results

### Per-Iteration Breakdown (NIFTy optimize_kl)

```
Iter  draw_smp   kl_min    total    mode
   1    1.80s     1.69s     3.49s   MGVI  (compile)
   2    0.01s     1.31s     1.32s   MGVI
   6    4.07s     1.36s     5.43s   geoVI (compile)
   7    0.02s     1.39s     1.41s   geoVI
```

KL minimization dominates at **1.35s/iter** after compilation.
Each iteration: ~10 Newton steps × ~7 CG iterations × 6 samples
= ~420 metric-vector products. At 0.1ms per JVP+VJP pair,
that's ~42ms compute but 1,350ms wall clock → **~30x Python overhead**.

### draw_linear_residual Breakdown

```
Estimated JAX compute:  ~2ms per sample
Actual wall clock:      540ms per sample
Python overhead:        270x
```

Sources of overhead:
- `partial()` / `Partial()` wrapping per call
- `assert_arithmetics()` type checking
- `jax.debug.callback()` logging in CG loop body
- Pytree dict manipulation for Vector state
- `_process_point_estimate()` processing

### Forward Model Is Not the Bottleneck

```
Forward pass (JIT'd):  0.04 ms
Gradient:              0.07 ms
JVP:                   0.05 ms
VJP:                   0.06 ms
```

## Our Solution

Reimplemented Algorithm 2 (geoVI) using flat arrays and JAX primitives:

### 1. Flat Array Representation

Instead of `jft.Vector` dicts, all parameters stored as `jnp.float64[137]`.
Eliminates all pytree manipulation.

### 2. CG Solver via `jax.lax.while_loop`

```python
def cg_solve(mat_fn, b, x0, maxiter=30, miniter=6, absdelta=1e-4):
    r = mat_fn(x0) - b
    d, gamma = r, jnp.dot(r, r)
    energy = jnp.dot((r - b) / 2, x0)
    init = (x0, r, d, gamma, energy, jnp.int32(-2), jnp.int32(0))

    def body(s):
        x, r, d, pg, pe, info, i = s
        i = i + 1
        q = mat_fn(d)
        curv = jnp.dot(d, q)
        alpha = pg / curv
        # Curvature check (from _static_cg)
        info = jnp.where(curv <= 0.0, jnp.int32(-1), info)
        alpha = jnp.where(curv <= 0.0, 0.0, alpha)
        x = x - alpha * d
        # Periodic reset every 20 iters (from _static_cg)
        r = jnp.where((i % 20 == 0) & (info < -1), mat_fn(x) - b, r - alpha * q)
        gamma = jnp.dot(r, r)
        # Energy-based convergence (from _static_cg)
        energy = jnp.dot((r - b) / 2, x)
        ed = pe - energy
        info = jnp.where(ed < -eps * jnp.abs(energy), jnp.int32(-1), info)
        info = jnp.where((ed < absdelta) & (i >= miniter) & (info < -1), jnp.int32(0), info)
        info = jnp.where((i >= maxiter) & (info < -1), i, info)
        d = d * jnp.maximum(0.0, gamma / (pg + 1e-30)) + r
        return (x, r, d, gamma, energy, info, i)

    return jax.lax.while_loop(lambda s: s[5] < -1, body, init)[0]
```

Adapted from NIFTy's `_static_cg` — keeps energy-based convergence,
curvature checks, and periodic reset, but removes `jax.debug.callback`
logging and dict-based state.

### 3. Full Optimization via `jax.lax.scan`

```python
def run_evi(init_pos, key, n_iterations, n_samples):
    def scan_body(carry, iteration_key):
        m = carry
        m = evi_step(m, iteration_key, n_samples)
        return m, None
    keys = jax.random.split(key, n_iterations)
    m_final, _ = jax.lax.scan(scan_body, init_pos, keys)
    return m_final
```

XLA compiles the entire 10-iteration loop into a single fused computation.

### 4. Vmapped Sample Operations

```python
# KL gradient over samples
def kl_vg(m, residuals):
    vals, grads = jax.vmap(lambda r: H_vg(m + r))(residuals)
    return jnp.mean(vals), jnp.mean(grads, axis=0)

# Metric over samples
def kl_metric(m, residuals, v):
    return jnp.mean(jax.vmap(lambda r: metric_vec(m + r, v))(residuals), axis=0)
```

### Results

| Component | NIFTy | Custom JIT | Speedup |
|-----------|-------|------------|---------|
| 10 KL iterations | 20s | **0.04s** | **500x** |
| 2000 posterior samples | 1080s | **0.01s** | **100,000x** |
| Full pipeline (cached) | ~70s | **1.7s** | **41x** |
| JIT compilation (one-time) | — | 3s | — |

## Why This Works for Our Problem

- **D = 137**: Small enough for XLA to compile the full loop.
  For D > 10^4 (NIFTy's typical radio imaging), compilation time
  and memory would be prohibitive.
- **5 data points**: The Jacobian J is 5×137, so J^T N^{-1} J
  is cheap. For spectroscopy with 10^4 wavelength pixels,
  the metric-vector product would dominate.
- **Static shapes**: All parameter shapes are known at compile time.
  NIFTy's generality (supporting dynamic shapes, point estimates,
  multi-GPU sharding) adds overhead that's unnecessary for our case.

## Compatibility

The custom JIT implementation is the default for `fitter.run("evi")`.
NIFTy's `optimize_kl` remains available via `fitter.run("geovi")` or
`fitter.run("mgvi")` for validation and compatibility.

## Suggestions for NIFTy

The `jit_metric` parameter in the `blackjax-interface` branch's
`draw_linear_residual` addresses part of this issue. For the full
benefit on small problems, the key changes would be:

1. **Option to disable logging callbacks in CG**: The `jax.debug.callback`
   in `_static_cg` is the main overhead source for single-call CG.
   A `name=None` already skips it, but the callback setup still happens.

2. **Flat-array fast path**: For problems where all parameters have the
   same dtype and the domain is known at construction time, operating
   on flat arrays avoids pytree manipulation entirely.

3. **`jax.lax.scan` option for the optimization loop**: The `OptimizeVI.run()`
   method uses a Python for-loop. A `jit_loop=True` option that uses
   `jax.lax.scan` (with `jax.debug.print` for logging) would give
   the same speedup while remaining compatible with the existing API.

These changes would be backward-compatible and only affect
performance, not correctness.
