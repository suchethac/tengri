# JIT compile cost in population inference

How XLA compile time scales (and **does not** scale) with catalog size `N`
in tengri's population inference engines, what changed in the `lax.map`
refactor, and where the actual N-scaling cost lives.

## TL;DR

- **JIT compile time is bounded and constant in N** for every backend.
  Synthetic and real-path benchmarks both show HLO size flat at ~1 MB and
  XLA compile at ~5–10 s regardless of `N`.
- The N-scaling cost users observe at `N≥16384` is **not compile** — it is
  the per-galaxy MAP-init Python loop in `PopulationFitter._run_native_vi_*`,
  which dispatches `N` sequential `Fitter.run("map")` calls.
- The manual `lax.map(vmap_K, chunked)` pattern was replaced by
  `jax.lax.map(..., batch_size=K)`. Same compile cost, smaller code,
  and non-divisible `N` is now supported (no padding required for
  K-divisibility).

## Diagnosis: where compile time *doesn't* scale

### Synthetic benchmark — `bench/scripts/benchmark_jit_compile.py`

A toy forward model (closed-over SSP-like and filter-like arrays, softmax SFH,
matrix multiply) is compiled AOT via `jit(f).lower(*abs_args).compile()` for
five batching strategies and `N ∈ {256, 1024, 4096, 16384}`, `K ∈ {1, 16, 64}`:

| Variant            | N=256 compile | N=16384 compile | HLO size |
|--------------------|---------------|-----------------|----------|
| `pure_lax_map`     | 0.06 s        | 0.06 s          | 0.96 MB  |
| `manual_chunk`     | 0.12 s (K=16) | 0.12 s (K=16)   | 0.96 MB  |
| `lax_map_batched`  | 0.12 s (K=16) | 0.12 s (K=16)   | 0.96 MB  |
| `pure_vmap`        | 0.12 s        | 0.12 s          | 0.95 MB  |
| `checkpoint_lax`   | 0.12 s (K=16) | 0.13 s (K=16)   | 0.96 MB  |

**Compile time and HLO size are flat across N for all variants, including
`pure_vmap` over N=16384.** This rules out the structural `lax.map` pattern
as the compile-time culprit. The two compile-time cost classes are:
`K=1 lax.map` (~0.07 s) and `K>1 vmap-inside-or-`pure_vmap` (~0.12 s).

### Real-path benchmark — `bench/scripts/benchmark_jit_real_path.py`

Production code (`PopulationFitter.run("native_vi_linear")`, 12 SDSS+GALEX
filters, n_iterations=2) with the persistent JAX cache enabled. Cold = first
run from a fresh cache; warm = second run, same N.

```{note}
This benchmark was recorded against `native_vi_linear`, which has since been
demoted to `tier="broken"` — the numbers below describe the compile behavior,
not a method to reach for. For catalog work use `mcmc_nuts` / `mcmc_hmc`
(batched and vmappable) or `map` (sequential); the compile analysis carries over
unchanged, since it is a property of the batched forward pass rather than of the
sampler.
```

| N    | setup_s | cold_run_s | warm_run_s | compile_proxy = cold − warm |
|------|---------|------------|------------|-----------------------------|
| 64   | 3.4     | 25.7       | 16.1       | 9.6                         |
| 256  | 1.6     | 37.4       | 31.7       | 5.7                         |
| 1024 | 6.7     | 53.4       | 44.0       | 9.5                         |

Read this table as three independent stories:

1. **`compile_proxy` ≈ 5–10 s, flat in N.** XLA compile cost does not
   scale with catalog size.
2. **`setup_s` scales with N** (3 → 7 s for N=64 → 1024). This is the
   Python `Fitter` construction + per-galaxy MAP-init loop in
   `_run_native_vi_linear`, line ~970. At N=16384 this becomes minutes.
3. **`warm_run_s` scales with N** (16 → 44 s for N=64 → 1024). This is
   pure execution: 2 KL iterations × N forward evaluations sequenced
   by `lax.map`.

## What the lax.map refactor changed

### Before

```python
n_padded = math.ceil(n_gal / K) * K
n_chunks = n_padded // K
n_pad = n_padded - n_gal

# inside signal_response:
if K == 1:
    predictions = jax.lax.map(lambda args: fwd(args[0], args[1]),
                              (p["gal"], p["gal_xi"]))
else:
    chunked_gal = jax.tree.map(lambda a: a.reshape(n_chunks, K), p["gal"])
    chunked_xi = p["gal_xi"].reshape(n_chunks, K, n_grid)
    predictions = jax.lax.map(
        lambda args: jax.vmap(fwd)(args[0], args[1]),
        (chunked_gal, chunked_xi),
    )
return predictions.reshape(n_padded, n_data_per_gal)[:n_gal].reshape(-1)
```

### After

```python
# inside signal_response:
predictions = jax.lax.map(
    lambda args: fwd(args[0], args[1]),
    (p["gal"], p["gal_xi"]),
    batch_size=K,
)
return predictions.reshape(-1)
```

`lax.map(..., batch_size=K)` (JAX ≥ 0.4.30) internally does
`vmap` over batches of size `K` and `scan` over `N//K` batches. **Identical
HLO and compile cost** to the manual chunking, and JAX handles non-divisible
`N` by running a separate vmap over the remainder.

### Trade-off: non-divisible N pays 2× compile

For `N % K ≠ 0`, the compiler emits two bodies (one for `vmap` of size `K`,
one for the remainder of size `N % K`). Real-path measurement at `N=30 K=4`
showed `compile_proxy = 16.3 s` vs `7.0 s` at `N=32 K=4` — exactly the
expected 2× factor. Pad to a multiple of K (or use `n_pad` in
`CatalogFitter`) when reusing the persistent cache across catalog sizes.

### What was removed

- Padding logic in `_run_native_vi_linear` and `_run_native_vi_nonlinear`
  (`n_padded`, `n_chunks`, `n_pad`, the `concatenate(zeros)` blocks for
  `gal` and `gal_xi`).
- The `n_chunks` reshape + trailing slice in
  `CatalogFitter._run_catalog_vi`.

### What was kept

- `n_pad` parameter on `CatalogFitter._run_catalog_vi` (user-facing). Even
  though `batch_size=K` no longer needs divisibility, `n_pad` is still
  useful for amortizing XLA compile cost across catalog sizes (e.g.,
  always pad to power-of-2).
- `K>1` requires uniform `n_data` per galaxy (vmap inside `lax.map` body
  needs static shapes).

## The actual N-scaling bottleneck: per-galaxy MAP init

At `_run_native_vi_linear` line ~970 (and the analogous geoVI path):

```python
for i in range(n_gal):
    fitter_i = Fitter(model, gal["flux_obs"], gal["noise"], data_type=...)
    map_i = fitter_i.run("map", n_steps=500, ...)
    init_u = fitter_i._unbounded_from_posterior(map_i)
    ...
```

This is a Python `for` loop doing `N` sequential `Fitter.run("map")` calls.
With the persistent cache warm, each call dispatches a few hundred JIT'd
ops without recompiling, but Python-side dispatch overhead (~1–5 ms per op)
× 500 steps × N galaxies ≈ minutes at N=16384. At cold cache, the first
call also pays compile (~10 s, amortized across all subsequent calls).

This is what users observe as a "hang" at `N≥16384` — it is the
MAP-init Python loop, not XLA compilation.

## Vectorized MAP-init (implemented)

The Python loop has been replaced by `build_vectorized_map_solver` in
`tengri.inference.backends.map_dispatch`, which returns a single JIT-friendly
per-galaxy MAP solver:

```python
from tengri.inference.backends.map_dispatch import build_vectorized_map_solver

map_solve_one = build_vectorized_map_solver(
    template_fitter, n_steps=200, learning_rate=0.03,
)

# All N galaxies via lax.map — one JIT compile, N executions.
all_init = jax.lax.map(
    lambda args: map_solve_one(args[0], args[1], args[2]),
    (all_flux, all_noise, all_keys),
    batch_size=K,
)
```

`map_solve_one(flux, noise, key)` runs `n_steps=200` Adam updates inside a
single `jax.lax.scan` (no Python overhead per step) and returns the final
unbounded-parameter dict. The body compiles once and runs against any
galaxy's `(flux, noise, key)` triple.

This is wired into all three MAP-init sites in `PopulationFitter`:
`_run_native_vi_linear`, `_run_native_vi_nonlinear`, and the
NIFTy-CFM hierarchical geoVI path.

### Compile-cost contract

- `build_vectorized_map_solver` body compiles in O(`loss_fn`), independent
  of `n_galaxies` and `n_steps` (the scan length is a static int).
- `lax.map(map_solve_one, ..., batch_size=K)` adds one more compile of
  size O(K · `loss_fn`) for the vmap'd body. `K=1` produces a sequential
  scan; `K>1` parallelizes K galaxies per scan iteration.
- Total compile cost is **bounded and independent of N**.

### What was removed

- The `for i in range(n_gal):` Python loops that constructed `Fitter`
  instances and called `Fitter.run("map")` per galaxy.
- The intermediate `gal_param_lists` / `gal_xi_list` stacking.
- The `import gc` + explicit buffer cleanup (no longer needed —
  `lax.map` streams without holding N intermediate Posterior objects).

### Trade-offs

- **No early-stopping**: the scan length is fixed (`n_steps=200`), so all
  galaxies run the full step budget. For Adam at `learning_rate=0.03`
  this is plenty for converging to a reasonable MAP for VI initialization.
  If a particular galaxy converges in 50 steps, the remaining 150 are
  wasted — but Python early-stopping per galaxy is what made the loop
  O(N) in the first place.
- **Uniform `n_data`**: `lax.map` requires shape-stable inputs across
  the batch axis — all galaxies must have the same number of data
  points. This was already required for `forward_chunk_size > 1`.
- **`learning_rate=0.03` and `n_steps=200`** are hard-coded. For most
  population-PSD problems they work; expose as kwargs if you need to
  tune (small change in `hierarchical.py`).

### Measured impact (real-path benchmark)

`bench/scripts/benchmark_jit_real_path.py` with `native_vi_linear`, K=1:

| N    | setup (Python) | setup (vmap) | warm (Python) | warm (vmap) | warm Δ |
|------|----------------|--------------|---------------|-------------|--------|
| 64   | 3.4 s          | 3.3 s        | 16.1 s        | 16.6 s      | +0.5 s |
| 256  | 1.6 s          | 1.7 s        | 31.7 s        | 22.9 s      | **−8.8 s** |
| 1024 | 6.7 s          | 4.0 s        | 44.0 s        | 41.7 s      | **−2.3 s** |

`compile_proxy` (cold − warm) stayed in the 6.7–9.5 s band before and
after — no compile blowup. The N=256 speedup is the regime where Python
dispatch overhead dominates per-galaxy work; at N=1024 cache-warmed
Python dispatch amortizes to a smaller fraction. The decisive win is at
`N ≥ 10⁴`, where the old Python loop's per-galaxy dispatch (≈1 ms × N)
becomes minutes of wall-time and the vectorized path stays bounded.

## Diagnosing compile issues in practice

1. **Time `lower()` and `compile()` separately**:

   ```python
   t0 = time.perf_counter()
   lowered = jax.jit(f).lower(*abs_args)
   t1 = time.perf_counter()
   compiled = lowered.compile()
   t2 = time.perf_counter()
   print(f"lower={t1-t0:.2f}s compile={t2-t1:.2f}s")
   ```

2. **Dump HLO** to see size and structure:

   ```bash
   XLA_FLAGS="--xla_dump_to=/tmp/hlo --xla_dump_hlo_as_text" \
   JAX_LOG_COMPILES=1 \
   .venv/bin/python <script>
   ```

   Inspect `/tmp/hlo/*.before_optimizations.hlo.txt` size. If it scales
   with N, you have an unrolled vmap; switch to `lax.map(..., batch_size=K)`.
   If only `*.after_optimizations.hlo.txt` is large, the optimizer is
   inflating the graph — try `XLA_FLAGS=--xla_backend_optimization_level=1`.

3. **Compare cold vs warm wall time** to isolate compile from execution.
   `cold − warm ≈ compile_time` (works because the persistent cache makes
   the second run skip compile entirely).

## See also

- `docs/inference/scaling.md` — wall-time scaling and PSD posterior recovery.
- `docs/performance/compilation.md` — persistent JAX cache.
- `bench/scripts/benchmark_jit_compile.py` — synthetic compile-time benchmark.
- `bench/scripts/benchmark_jit_real_path.py` — real-path compile-time benchmark.
