# Persistent compilation cache

Every JIT-compiled function in tengri (every backend, every fitter) goes
through XLA. First compile of a given shape is expensive — geoVI compiles
in ~75 s, MGVI's `lax.while_loop` in ~10 s, MCMC warmups in tens of
seconds. The compiled artifact is fast to *load* but expensive to
*build*.

JAX's persistent on-disk compilation cache turns "build" into "load" for
every later invocation: the HLO is serialized to disk on first compile and
loaded in ~100 ms by every later subprocess, notebook restart, slurm
worker, or sweep re-run.

tengri auto-enables this cache on `import tengri`.

## Default behavior

```python
import tengri  # cache auto-enabled at ~/.cache/tengri_jax_cache
```

- Cache directory: `~/.cache/tengri_jax_cache` (matches the legacy path,
  so existing user caches are preserved).
- Min compile time threshold: 5 s — below this, compiles are not
  persisted. This keeps the cache focused on the genuinely expensive
  compiles (geoVI, MGVI, MCMC warmups) and skips the long tail of small
  SSP/dust kernels that compile in 1–2 s and would otherwise bloat disk.

## Configuration

Override the cache directory via environment variable:

```bash
export TENGRI_JAX_CACHE_DIR=/scratch/$USER/jax_cache
```

Useful when `~/.cache` is on a small/quota'd filesystem (clusters,
shared workstations).

Disable entirely:

```bash
export TENGRI_DISABLE_JAX_CACHE=1
```

Programmatic control (for notebook cells where setting env vars before
import is awkward):

```python
import tengri
tengri.enable_persistent_cache(
    "/scratch/jax_cache",
    min_compile_time_secs=5.0,
)
```

`enable_persistent_cache` is idempotent — calling it again with the same
directory is a no-op.

## Cache key

JAX's persistent cache hashes:

- The jaxpr (function structure)
- Abstract input shapes / dtypes
- `static_argnames` values (e.g. `n_samples` in `native_vi_*`)
- JAX / jaxlib version
- CPU class (or GPU model)

Implications:

- **Different RNG seeds hit the same cache** — `PRNGKey` concrete values
  are not part of the key. Re-running with `key=jax.random.PRNGKey(42)`
  vs `PRNGKey(43)` is free after the first compile.
- **Different `n_samples` triggers a fresh compile** — once. Subsequent
  runs at the same `n_samples` hit the cache.
- **A new JAX/jaxlib install invalidates everything.** Wipe the cache
  after `pip install -U jax`:

  ```python
  import tengri
  tengri.clear_cache()
  ```

## When to wipe

| Trigger | Action |
|---------|--------|
| `pip install -U jax` / `jaxlib` | `tengri.clear_cache()` |
| Moved machine / different CPU class | `tengri.clear_cache()` |
| Disk usage too high (`du -sh ~/.cache/tengri_jax_cache`) | `tengri.clear_cache()` then re-run |

Inspect current size:

```python
import tengri
print(tengri.cache_size_bytes() / 1024**2, "MB")
```

## Use cases

- **Notebook iteration.** Restart kernel → cache hit → first cell of a
  fit runs in seconds instead of minutes.
- **Slurm arrays.** Each task is a fresh process; without persistent
  cache each pays full compile, possibly hundreds of times. With it:
  pay once globally.
- **Benchmark sweeps.** `bench/scripts/benchmark_vi_xlarge.py` spawns one
  subprocess per `(N, K)` cell for clean peak-RSS measurement; the
  cache amortizes compile cost across the entire grid.
- **Resume after crash.** Cache entries are persisted at compile
  *finalization* (before run starts). A worker that finishes compile
  and then crashes during run still leaves a usable artifact for the
  next attempt.

## What it doesn't fix

- **First-compile cost** at any new shape — that work still has to be
  done once globally.
- **The XLA compile cliff** at very large shapes (e.g. MGVI K=1
  N≥16384, geoVI K=1 N≥8192) where XLA's optimizer goes from O(10 s)
  to unbounded. The cache helps *subsequent* runs once any worker has
  built the artifact, but the first build can still hang. Tracked
  separately.
- **Different `n_samples`** — `n_samples` is a `static_argname`, so
  changing it is a different cache key. The benchmark grid holds it
  fixed at 6.

## Three-layer cache architecture (Phase B, 2026-05)

The on-disk persistent cache described above is one of three independent
caches that work together. Each handles a different reuse boundary:

| Layer | Lives in | Reuse boundary | Cleared by |
|-------|----------|----------------|------------|
| L1 — structural prediction kernels | RAM, `_STRUCTURAL_KERNEL_CACHE` (LRU=4) | Across `SEDModel` instances with the same `compile_signature()` | `tengri.gc()` |
| L2 — loss / grad / log-density / log-likelihood | RAM, `_SHARED_*_CACHE` keyed on `(compile_sig, mode)` | Across `Fitter` instances with the same fingerprint | `tengri.gc()` (kept by surgical lean) |
| L3 — inference scan body (HMC leapfrog, NUTS tree, VI loop) | RAM, `_SHARED_ENGINE_CACHE` (LRU=2) | Across `Fitter.run` calls with the same `(compile_sig, method)` — kept by smart lean | Smart `lean()` drops only stale entries; `tengri.gc()` drops all |
| Disk cache | `~/.cache/tengri_jax_cache` | Across processes, notebook restarts, slurm tasks | `tengri.clear_cache()` |

`Fitter.run(lean=True)` (the default) calls
`clear_shared_caches(scope="inference_body", keep_sig=(self.compile_signature(), method))`
*before* the run. This drops only L3 entries whose key does **not** match
the current run — so:

- **Multi-phase notebook** (MAP → HMC → posterior-predictive): the prior
  phase's L3 entry is dropped (different `mode`), the current phase's
  entry is kept. Peak RSS stays at one inference body, never two.
- **CatalogFitter loop** (100 galaxies, same model + method): every run
  has the same `(compile_sig, method)`, so the entry is preserved and
  the leapfrog compile is paid once for the whole catalog. No
  `persistent()` context needed for the common case.

`tengri.gc()` calls `clear_shared_caches(scope="all", drop_xla=True)` —
nukes L1 + L2 + L3 + JAX's internal caches. Use between iteration loops
that build many slightly-different SEDModel / Fitter configurations.

`tengri.persistent()` is now only useful for the rare case where you
want to keep a stale L3 entry alive across phases (e.g. running MAP and
HMC repeatedly in alternation and reusing both compiles). The default
smart-lean path covers the catalog and multi-phase cases without it.

The disk cache underneath all three means even a full `tengri.gc()` on a
warm process does not recompile from scratch on the next call: XLA reads
the already-optimized binary from `~/.cache/tengri_jax_cache` (~3 s
versus the ~80 s cold compile).
