# Compilation: caching and diagnostics

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

- Cache directory: `~/.cache/tengri_jax_cache`.
- Compiles faster than `min_compile_time_secs` are not persisted. The
  default is tuned so that every kernel that costs real wall time — from
  per-filter photometry kernels up to geoVI — survives a restart; see
  `tengri.enable_persistent_cache` for the knob.

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

Programmatic control (for notebook cells):

```python
import tengri
tengri.enable_persistent_cache("/scratch/jax_cache")
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
- **Different `n_samples` triggers one fresh compile.** Subsequent runs at
  the same `n_samples` hit the cache.
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
- **Resume after crash.** Cache entries are persisted after compile
  completes. A worker that crashes mid-run leaves a usable artifact for
  the next attempt.

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

## Three-layer cache architecture

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

### Tuning the in-memory caches

The L1–L3 caches above are also tunable from the environment, which is
useful when a process is memory-bound rather than compile-bound:

```bash
export TENGRI_ENGINE_CACHE_MAXSIZE=1   # default 2; L3 entries held before eviction
export TENGRI_DISABLE_SHARED_CACHES=1  # never populate the shared caches at all
export TENGRI_LEAN=1                   # force lean mode process-wide
export TENGRI_PERSISTENT=1             # opt out of the lean default
```

Each L3 entry holds a compiled XLA executable, so lowering
`TENGRI_ENGINE_CACHE_MAXSIZE` to 1 trades one extra compile for a
smaller resident set. `TENGRI_DISABLE_SHARED_CACHES=1` is the strongest
guarantee — every `Fitter` compiles fresh, and nothing is retained
between fits — at the cost of paying the compile on every run.

The in-process equivalents are `tengri.lean()` / `tengri.persistent()`
(context managers) and `tengri.gc()` / `tengri.clear_shared_caches()`.
All four are importable from the top-level package, though they are not
listed in `tengri.__all__`.

The disk cache underneath all three means even a full `tengri.gc()` on a
warm process does not recompile from scratch on the next call: XLA reads
the already-optimized binary from `~/.cache/tengri_jax_cache` (~3 s
versus the ~80 s cold compile).

## Diagnosing recompilations

The cache above makes a *repeated* compile cheap. When wall time is
still dominated by compilation, the question is which kernels are
recompiling and why — that is what the compile-event tracer answers.

When working with JAX inference on large models, cold compilation can
dominate wall-clock time. The compile-event tracer shows what is recompiling
and why.

### Quick Start

#### 1. Enable logging

Set the environment variable before running your notebook:

```bash
export TENGRI_LOG_COMPILES=1
```

#### 2. Run your notebook

Execute your notebook as normal. Events will be logged to
`~/.cache/tengri_jax_cache/compile.log` (or override via
`TENGRI_COMPILE_LOG_PATH`).

#### 3. Analyze the log

```bash
python scripts/analyze_compile_log.py
```

or specify a custom log path:

```bash
python scripts/analyze_compile_log.py --log /path/to/compile.log
```

### What the Analysis Shows

The report includes:

- **Total compile events**: How many JIT compilations occurred
- **Total wall time**: Aggregate compilation time (seconds)
- **Cache-hit ratio**: Proportion of fast (cached) vs. slow (cold) compiles
- **Per-method breakdown**: Count, total, mean, and max duration for each inference method
- **Spurious recompiles**: Consecutive events with different signatures
  (indicates unnecessary recompilation)

Example output:

```
================================================================================
TENGRI COMPILE LOG ANALYSIS
================================================================================

SUMMARY
-------
Total compile events:        14
Total compile wall time:     42.53 s
Cache hits (inferred):       8
Cache misses (inferred):     6
Hit ratio:                   57.1%

PER-METHOD BREAKDOWN
-------
Method                 Count      Total (s)      Mean (s)      Max (s)
-------
geovi                      2         15.23          7.61         8.10
vi                         6          8.54          1.42         2.31
unknown                    6         18.76          3.13         6.54

SPURIOUS RECOMPILES (consecutive events with different signatures)
-------
[2→3] signal_response (None) → run_evi (vi)
  sig[2]: ((...shape_sig..., ...model_sig...),)
  sig[3]: ((...different_model_sig...,),)

```

### Configuration

#### Environment Variables

- `TENGRI_LOG_COMPILES=1` – Enable compile logging (default: off)
- `TENGRI_COMPILE_LOG_PATH=/custom/path.log` – Override log file location

#### Disabling Logging

By default, the logger is completely disabled and adds zero overhead. To confirm:

```python
from tengri.utils.compile_log import is_enabled
print(is_enabled())  # False if TENGRI_LOG_COMPILES not set
```

### Cache-Hit Heuristic

The log marks events as "cache hits" if they complete in < 1.0 s. This is a rough approximation:

- **True hits** (file on disk): typically 0.1–0.5 s
- **Cold compiles**: typically 5–30+ seconds (depends on graph size)
- **Hybrid cases** (e.g., warm iteration with some tracing): 1–5 s

On fast hardware, the heuristic may flag slow loads as hits. On
network-mounted storage, it may misclassify cache loads as cold compiles.

### Integration

The tracer hooks into:

- `get_or_build_signal_response()` – Physics kernel compilation
- `build_jit_engine()` – Inference engine (VI, geoVI, etc.)

Each compilation site is wrapped with a context manager that records timing
and metadata. No code changes required.

### Notes

- Logging is **thread-safe**: multiple threads can write events simultaneously
- The log file grows indefinitely; manually clean it up as needed
- Timestamps are UTC ISO 8601 format
- Signatures are stringified tuples for easy diffing to detect spurious recompiles
