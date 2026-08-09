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
- **The sub-band IGM fold on a free-redshift model.** This is a
  build-time constant, not a compile, so neither the JAX cache nor the
  photometry z-table cache covers it — and it is re-paid on every
  `SEDModel.build`, including repeat builds in one process.

## Build cost: the free-redshift cliff

Compilation is not the only cost before your first number. Building a
free-redshift model with `approx=WavePrecomp()` folds the IGM
transmission into the sub-band quadrature weights, and that fold
dominates:

| build | time |
| --- | --- |
| free redshift, `WavePrecomp()` | ~9 s (~37 s on the first build of a new SSP × filter × z-grid combination, which also pays for the z-table) |
| free redshift, `WavePrecomp()`, `igm={'type': 'none'}` | ~0.3 s |
| fixed redshift, `WavePrecomp()` | ~0.4 s |
| free redshift, `approx=None` | negligible |

Two consequences worth planning around:

- **Fixing the redshift is the cheapest way out** when your science
  allows it. A photometric-redshift fit does not, and pays the fold.
- **The cost is per build, not per process.** Building the same
  configuration three times in one session pays it three times. If you
  are looping over models that differ only in parameters, build once and
  vary the parameters — they are runtime inputs, not build inputs.

Absolute times vary with the machine, band set, and z-grid density; the
ratios are the durable part.

> **Measuring this yourself:** omitting `redshift=` does *not* give you a
> free redshift — see the default below — so a benchmark that leaves it
> out measures the fixed-z path and will not reproduce any of this.

## Runtime cost: free redshift is ~7x a gradient, not ~1x

Build cost is paid once. The gradient cost is paid every leapfrog step, and
for a sampler that is the number that sets your wall clock.

Measured on the compiled gradient graph (`cost_analysis`, so these are exact
and do not depend on the machine), five SDSS bands, `WavePrecomp(n_z=250)`
over z ∈ [0.5, 12]:

| model | FLOPs per gradient |
| --- | --- |
| fixed redshift | 2.7 M |
| free redshift, `igm={'type': 'none'}` | 17.9 M |
| free redshift, with IGM | 18.8 M |

Free redshift interpolates the SSP × filter table, the sub-band quadrature
tensors and the sub-band IGM transmission to the sampled z on every call. That
is ~7x a fixed-z gradient, and it is the dominant term in any photometric-
redshift fit.

**Budgeting a sampler.** Multiply it out before you launch:

```
gradients = (n_warmup + n_samples) * n_leapfrog_steps
```

`fit_interim(n_warmup=300, n_samples=1000, n_leapfrog_steps=100)` is 130,000
gradients **per galaxy**. At a free-redshift gradient of a few milliseconds
that is minutes to tens of minutes per galaxy — so an eight-galaxy population
fit is hours, and it is hours whether or not anything is wrong. If a fit is
taking longer than you expect, compute this product first: it usually explains
the whole thing, and no amount of profiling will make 130,000 gradients cheap.

**`n_z` is not the speed knob it looks like.** It sets the accuracy of the z
interpolation (the default 250 is calibrated for <1% error), and it no longer
buys speed: the interpolation contracts only the five grid nodes the triweight
kernel actually supports, so the per-gradient cost is the same at `n_z=250` as
at `n_z=10` — byte for byte in the compiled graph. Lower it for build time and
memory if you need to, not for gradient throughput.

> **A note on reading `cost_analysis`.** Its `bytes accessed` field counts
> logical operand bytes per operation, not DRAM traffic after fusion. The
> windowing change above cut FLOPs 6.8x and `bytes accessed` 5.1x and moved
> the clock about 1.9x. Use these counters to find where cost lives; use a
> stopwatch to find out what it costs.

## The redshift default is fixed, not free

`SEDModel.build(...)` with no `redshift=` argument pins the redshift to
`Fixed(0.1)`. It does not leave it free.

```python
model = SEDModel.build(ssp_data=ssp, observation=obs, sfh={...})
# -> redshift is Fixed(0.1); the galaxy sits at z = 0.1

model = SEDModel.build(ssp_data=ssp, observation=obs, sfh={...},
                       redshift=Uniform(0.0, 3.0))   # free, and fitted
```

State it explicitly whenever the redshift matters — either
`redshift=Fixed(z)` at your galaxy's known redshift, or a prior to fit
it. Two things follow from the default that are easy to miss:

- A model you meant to be free-redshift is silently at z = 0.1, so the
  fit reports whatever the other parameters can do at that distance.
- Build-cost measurements taken without an explicit free prior exercise
  the fixed-z path, where the IGM fold above costs ~0.4 s instead of
  ~9 s.

## Three-layer cache architecture

The on-disk persistent cache described above is one of three independent
caches that work together. Each handles a different reuse boundary:

| Layer | Lives in | Reuse boundary | Cleared by |
|-------|----------|----------------|------------|
| L1 — structural prediction kernels | RAM, `_STRUCTURAL_KERNEL_CACHE` (LRU=4) | Across `SEDModel` instances with the same `compile_signature()` | `tengri.gc()` |
| L2 — loss / grad / log-density / log-likelihood | RAM, `_SHARED_*_CACHE` keyed on `(compile_sig, mode)` | Across `Fitter` instances with the same fingerprint | `tengri.gc()` — no cache policy touches it |
| L3 — inference scan body (HMC leapfrog, NUTS tree, VI loop) | RAM, `_SHARED_ENGINE_CACHE` (LRU=2) | Across `Fitter.run` calls with the same `(compile_sig, method)` | `iterate` (default) drops only non-matching entries; `sweep` drops all; `tengri.gc()` drops all |
| Disk cache | `~/.cache/tengri_jax_cache` | Across processes, notebook restarts, slurm tasks | `tengri.clear_cache()` |

Each fit picks one of three cache policies before it runs. The default
is **iterate**, which clears L3 with
`keep_sig=(compile_signature(), method)` — dropping only entries whose
key does **not** match the current run. So:

- **Multi-phase notebook** (MAP → HMC → posterior-predictive): the prior
  phase's L3 entry is dropped (different `mode`), the current phase's
  entry is kept. Peak RSS stays at one inference body, never two.
- **Catalog loop** (100 galaxies, same model + method): every run
  has the same `(compile_sig, method)`, so the entry is preserved and
  the leapfrog compile is paid once for the whole catalog. No
  `persistent()` context needed for the common case.

The other two policies are opt-in, and both are reached through a
context manager rather than a keyword:

| Policy | How to select it | L3 effect |
|---|---|---|
| `iterate` (default) | nothing to do | keep the matching entry, drop stale ones |
| `sweep` | `tengri.lean()` or `TENGRI_LEAN=1` | drop every L3 entry |
| `persistent` | `tengri.persistent()` or `TENGRI_PERSISTENT=1` | keep everything, matching or not |

The older `fit(..., lean=True)` keyword is retired. It is still honored
for back-compat (`lean=True` → `sweep`, `lean=False` → `iterate`) but
emits a `DeprecationWarning`; prefer the context managers.

`tengri.gc()` calls `clear_shared_caches(scope="all", drop_xla=True)` —
nukes L1 + L2 + L3 + JAX's internal caches. Use between iteration loops
that build many slightly-different SEDModel / Fitter configurations.

`tengri.persistent()` is only useful for the rare case where you want to
keep a *stale* L3 entry alive across phases (e.g. running MAP and HMC
repeatedly in alternation and reusing both compiles). The default
keep-matching path covers the catalog and multi-phase cases without it.

### Tuning the in-memory caches

The L1–L3 caches above are also tunable from the environment, which is
useful when a process is memory-bound rather than compile-bound:

```bash
export TENGRI_ENGINE_CACHE_MAXSIZE=1   # default 2; L3 entries held before eviction
export TENGRI_DISABLE_SHARED_CACHES=1  # never populate the shared caches at all
export TENGRI_LEAN=1                   # force the sweep policy process-wide
export TENGRI_PERSISTENT=1             # keep every L3 entry, stale included
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
