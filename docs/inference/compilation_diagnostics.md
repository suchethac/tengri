# Diagnosing Recompilations

When working with JAX inference on large models, cold compilation can dominate wall-clock time in tutorial notebooks. The compile-event tracer helps you understand exactly what is recompiling and why.

## Quick Start

### 1. Enable logging

Set the environment variable before running your notebook:

```bash
export TENGRI_LOG_COMPILES=1
```

### 2. Run your notebook

Execute your notebook as normal. Events will be logged to `~/.cache/tengri_jax_cache/compile.log` (or override via `TENGRI_COMPILE_LOG_PATH`).

### 3. Analyze the log

```bash
python scripts/analyze_compile_log.py
```

or specify a custom log path:

```bash
python scripts/analyze_compile_log.py --log /path/to/compile.log
```

## What the Analysis Shows

The report includes:

- **Total compile events**: How many JIT compilations occurred
- **Total wall time**: Aggregate compilation time (seconds)
- **Cache-hit ratio**: Proportion of fast (cached) vs. slow (cold) compiles
- **Per-method breakdown**: Count, total, mean, and max duration for each inference method
- **Spurious recompiles**: Consecutive events with different signatures (indicates unnecessary recompilation)

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

## Configuration

### Environment Variables

- `TENGRI_LOG_COMPILES=1` – Enable compile logging (default: off)
- `TENGRI_COMPILE_LOG_PATH=/custom/path.log` – Override log file location

### Disabling Logging

By default, the logger is completely disabled and adds zero overhead. To confirm:

```python
from tengri.utils.compile_log import is_enabled
print(is_enabled())  # False if TENGRI_LOG_COMPILES not set
```

## Cache-Hit Heuristic

The log marks events as "cache hits" if they complete in < 1.0 s. This is a rough approximation:

- **True hits** (file on disk): typically 0.1–0.5 s
- **Cold compiles**: typically 5–30+ seconds (depends on graph size)
- **Hybrid cases** (e.g., warm iteration with some tracing): 1–5 s

The heuristic may have false positives on very fast hardware or false negatives on network-mounted storage.

## Integration

The tracer is automatically hooked into:

- `get_or_build_signal_response()` – Physics kernel compilation
- `build_jit_engine()` – Inference engine (VI, geoVI, etc.)

Each major compilation site is wrapped with a context manager that records timing and metadata. No changes are needed to your code.

## Notes

- Logging is **thread-safe**: multiple threads can write events simultaneously
- The log file grows indefinitely; manually clean it up as needed
- Timestamps are UTC ISO 8601 format
- Signatures are stringified tuples for easy diffing to detect spurious recompiles
