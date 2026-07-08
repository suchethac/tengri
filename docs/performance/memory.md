# Memory expectations

Tengri's forward model is light on memory; the heavy hitters are JIT
compilation graphs and inference-backend internals. This page lists the
peak RSS you should expect for the common cases, the two recurring OOM
patterns and how to avoid them, and a watchdog you can run on machines
with < 32 GB if you're batch-authoring notebooks.

## What to expect

| Workload | Steady-state RSS | Peak RSS during compile |
|---|---:|---:|
| Smooth D = 7 photometric fit (NUTS) | ~100 MB | 3–6 GB |
| Smooth D = 7 photometric fit (MAP / Laplace / Pathfinder) | ~100 MB | ~1 GB |
| Stochastic D ≈ 137 SFH (geoVI) | ~1.5 GB | 5–6 GB |
| Spectroscopy (1000-pix optical, NUTS) | ~300 MB | 4–8 GB |
| Joint photo + spec (NUTS) | ~500 MB | 6–10 GB |

NUTS warmup with `dense_mass=True` peaks 3–6× higher than steady state due
to `vmap(vmap(...))` tracing. D ≥ 8 with `dense_basis` can hit 20+ GB.
**Multi-fit notebooks need `dense_mass=False`** — see [Multiple NUTS
fits](#pattern-multiple-nuts-fits-per-process).

These numbers are CPU on Apple M-series; GPU peaks are typically lower
because XLA can fuse more aggressively, but VRAM ceilings are tighter.

## Pattern: multiple NUTS fits per process

**Symptom:** First fit OK, second compiles slowly, third OOMs.

**Cause:** Each NUTS warmup compiles `vmap(vmap(...))` of the full predict
graph (~4 GB per fit). The JIT cache amortizes next *calls* but not
*traces*. Different `Observation` or `n_warmup` invalidates the key,
re-paying compile cost.

**Fixes (in order):**

1. **One NUTS fit per notebook.** Use MAP for cheap "before" fits.
2. Share state: same model, same observation *type*, only data changes. JIT cache survives.
3. Drop dense mass: `fitter.run("mcmc_nuts", dense_mass=False, ...)`. Cuts
   compile ~3× (costs ~2× autocorrelation; run more samples).
4. Use HMC: `fitter.run("mcmc_hmc", ...)`. Smaller JIT graph, no binary-tree expansion.

## Pattern: background compile + macOS jetsam

`Fitter` pre-compiles every backend at construction so the first
user-facing call is fast. On models with heavy template blocks, that
background compile alone can reach several GB — on macOS enough to
trip the `jetsam` memory killer before the first call runs.

**Fix:** Disable background compile before `import tengri`:

```python
import os
os.environ.setdefault("TENGRI_NO_BACKGROUND_COMPILE", "1")
import tengri
```

All spine notebooks do this; copy the pattern for OOM-prone configurations.

## JAX persistent compile cache

JAX recompiles XLA on every cold start. Tengri auto-enables a persistent
cache at `~/.cache/tengri_jax_cache` to avoid re-compiling.

If the directory is empty after a notebook run:
- Check writability.
- Check `TENGRI_JAX_CACHE_DIR` if set.
- Compiles faster than `min_compile_time_secs` are skipped by design.

After upgrading JAX:

```python
import tengri
tengri.clear_cache()
```

See [compilation_cache.md](https://github.com/suchethac/tengri/blob/main/docs/inference/compilation_cache.md) for details.

## A safety-net watchdog

If you batch-author notebooks on a < 32 GB machine, running a simple
watchdog as a background process is cheap insurance. The default
threshold is 20 GB — comfortable headroom above the worst-case
single-NUTS-fit peak, low enough to catch a runaway:

```bash
THRESHOLD_KB=20971520  # 20 GB; lower to 10 GB if your machine is tight
while true; do
  ps -axo pid=,rss=,comm= \
    | awk -v t=$THRESHOLD_KB '$2>t && $3 ~ /python/ {print $1, $2, $3}' \
    | while read pid rss cmd; do
        echo "$(date) KILL $pid rss=${rss}KB cmd=$cmd" >> /tmp/oom_killer.log
        kill -9 $pid 2>/dev/null
      done
  sleep 5
done
```

Logs go to `/tmp/oom_killer.log` so you can see what got killed. Drop
the threshold only if your heaviest legitimate fit stays under it —
NUTS on a model with template dust emission can briefly cross 10 GB
during compile.

## When to file a bug

A single fit peaking far above the table at the top of this page is a
regression — file it with the model config and the observed peak RSS.

See [`docs/dev/notebook_orchestration_oom.md`](https://github.com/suchethac/tengri/blob/main/docs/dev/notebook_orchestration_oom.md) for technical root causes and subagent-zombie patterns.
