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

NUTS warmup with `dense_mass=True` peaks 3–6× higher than steady state
because it traces a `vmap(vmap(...))` of the full predict graph. On D ≥ 8
with `mean_sfh_type="dense_basis"` we have observed peaks of 20+ GB.
**Multi-fit notebooks need `dense_mass=False`** unless you have explicit
headroom — see [Pattern: multiple NUTS fits](#pattern-multiple-nuts-fits-per-process).

These numbers are CPU on Apple M-series; GPU peaks are typically lower
because XLA can fuse more aggressively, but VRAM ceilings are tighter.

## Pattern: multiple NUTS fits per process

**Symptom.** First fit runs fine, the second fit takes much longer to
compile, the third fit OOMs or thrashes swap.

**Cause.** Each NUTS warmup compiles a dense-mass-matrix `vmap(vmap(...))`
that pulls the entire `predict_photometry` (and `predict_spectrum`)
graph into one program. Peak compile-time RSS is ~4 GB per fit on a
typical pipeline; the JIT cache amortises the *next call*, not the
next *trace*. A different `Observation` or `n_warmup` invalidates the
cache key, so each fit re-pays the compile cost.

**Fixes, in order of preference.**

1. **One NUTS fit per notebook**. Use MAP for cheaper "before" fits,
   NUTS only for the headline posterior.
2. If you genuinely need multiple posteriors (notebook 07 fits photo,
   spec, and joint), share as much state as possible: same model, same
   observation *type*, only the data array changes. The JIT cache
   survives different data values for the same observation type.
3. Drop the dense mass matrix:

   ```python
   fitter.run("mcmc_nuts", dense_mass=False, ...)
   ```

   This cuts compile peak by ~3× at the cost of ~2× sample
   autocorrelation. Run more samples to compensate.
4. Switch to plain HMC (`"mcmc_hmc"`) — same posterior target, much
   smaller JIT graph (no doubling-binary-tree expansion in the trace).

## Pattern: background compile + macOS jetsam

`Fitter` pre-compiles every inference backend (NUTS, MAP, geoVI,
raytrace, NSS) on construction so the first user-facing call is fast.
With `dust_emission="dale2014"` the geoVI compile alone pushes peak RSS
to ~6 GB. On macOS this can trip the kernel's `jetsam` memory-pressure
killer before the first call ever runs.

**Fix.** Disable background compile *before* `import tengri`:

```python
import os
os.environ.setdefault("TENGRI_NO_BACKGROUND_COMPILE", "1")
import tengri  # <-- after the env var is set
```

Every shipped spine notebook in `notebooks/` does this; if you author
your own notebook for an OOM-prone configuration, copy the pattern.

## JAX persistent compile cache

JAX recompiles XLA programs on every cold start. Tengri auto-enables a
persistent on-disk cache at `~/.cache/tengri_jax_cache` so subsequent
runs hit it instead of re-compiling.

```bash
ls -lh ~/.cache/tengri_jax_cache/ | head -10        # confirm it's filling
```

If the directory is empty after a notebook run:

- Check the directory is writable.
- Check `TENGRI_JAX_CACHE_DIR` if you set a custom location.
- The default `min_compile_time_secs=5.0` skips small kernels — that's
  intentional and not a bug.

After upgrading JAX:

```python
import tengri
tengri.clear_cache()
```

stale entries are evicted. See
[compilation_cache.md](../inference/compilation_cache.md) for details.

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
the threshold to 10 GB only if you're not running NUTS on the
`dale2014` pipeline — that legitimate workload can briefly cross 10 GB
during compile.

## When to file a bug

If a single notebook with a single fit and `dust_emission="dale2014"`
peaks above 8 GB on your machine, that's a regression worth reporting.
The post-Phase-II-2 baseline is ~5 GB peak compile, ~1 GB steady state.

The contributor-side write-up of these patterns lives at
[`docs/dev/notebook_orchestration_oom.md`](https://github.com/suchethac/tengri/blob/main/docs/dev/notebook_orchestration_oom.md);
read that for the technical root cause and the subagent-zombie patterns
that affect AI-assisted development.
