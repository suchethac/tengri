# Notebook orchestration: OOMs and long compiles (2026-05-06)

Companion to `quickstart_oom_diagnosis.md`. That doc covers the
*technical* root cause of the energy-balance compile cost. This doc
covers the **operational** patterns that make those costs hurt during
notebook authoring, batch runs, and subagent-driven development.

## TL;DR — three rules

1. **Never run more than one NUTS fit per notebook process** unless you
   have a reason. Each NUTS warmup peaks at 3–6 GB RSS during the dense
   mass-matrix `vmap` compile. Three sequential warmups in one process
   regularly hits 15+ GB transient peak even with the JIT cache warm.
2. **Rejecting a subagent does not kill its `python` child processes.**
   Always `ps -axo pid,rss,comm | grep python` after rejecting a
   subagent that was running a notebook.
3. **Run a watchdog when batch-authoring notebooks** that drive multiple
   subagent fits. A simple `ps | awk` loop killing python processes
   above 20 GB has saved this project from a wedged kernel more than
   once. Threshold 10 GB is more aggressive but cuts into legitimate
   single-NUTS-fit workloads.

## Pattern: subagent-launched zombie

**Symptom.** System feels sluggish. `vm_stat` shows high active+wired.
There are no obvious foreground processes running. `top` shows a
`python notebooks/*.py` at multi-GB RSS that you do not remember
launching.

**Cause.** When a subagent runs a notebook via the `Bash` tool and the
parent rejects/kills the agent, the agent's tool-use turn terminates
but the `python` subprocess it spawned in the foreground keeps running
to completion. The agent is gone; the JIT compile is not. Multi-fit
notebooks like `07_joint_photo_spec.py` (three NUTS fits) can climb
past 5 GB before they finish — alone, not enough to fire a 20 GB
watchdog, but enough to add real pressure to a developer machine
already running Claude Code, browsers, and other JAX processes.

**Fix.** After rejecting any subagent that touched a notebook:

```bash
ps -axo pid,rss,user,command | grep -i 'python' | grep -v ipykernel | sort -k2 -nr | head
# Kill anything you don't recognise:
kill -9 <pid>
```

**Prevention.** Use `run_in_background: true` on subagent `Bash` calls
that execute notebooks, OR have the subagent capture a PID and tee
stdout to a file so you can later track what's still running.

## Pattern: multiple NUTS fits per notebook

**Symptom.** Notebook runs fine for the first fit, second fit takes
much longer to compile, third fit either OOMs or thrashes swap.

**Cause.** NUTS warmup compiles a dense-mass-matrix vmap that pulls
the entire `predict_photometry` (and `predict_spectrum`) graph into a
nested `vmap(vmap(...))`. The peak compile-time RSS for one fit on
the dale2014 + nebular pipeline is ~4 GB. The JIT cache amortises the
*next* call but not the next *trace*: a different `Observation` object
or different `n_warmup` invalidates the cache, so each fit re-pays
the compile cost.

**Fix.**

- Prefer ONE NUTS fit per notebook. Use MAP for the cheaper "before"
  fits, NUTS only for the headline result.
- If you genuinely need multiple posteriors for comparison (e.g.
  notebook 07 phot vs spec vs joint), share as much state as possible:
  same model, same observation type, just different `data` arg passed
  to `Fitter`. The Fitter's `data` is a pytree leaf, not a static
  arg — the JIT cache *does* survive different data values for the
  same observation type.
- When that's not possible, drop dense mass matrix:
  `fitter.run("mcmc_nuts", dense_mass=False, ...)` cuts compile peak
  by ~3× at the cost of ~2× sample autocorrelation.
- Or use `mcmc_hmc` (plain HMC) instead of NUTS — same posterior, much
  smaller JIT graph (no doubling-binary-tree expansion).

## Pattern: closure-captured SSP grid

Documented in detail in `quickstart_oom_diagnosis.md`. Brief recap:

- The 114 MB SSP flux grid is captured by closure inside the JIT'd
  graph, blowing up XLA constant-folding cost.
- Phase II-2 fixes (landed 2026-05-03) reduced this dramatically.
- If you see `predict_photometry` compile time > 30 s on a model that
  doesn't use `dust_emission`, suspect the closure path is back. File
  a bug.

## Pattern: `dust_emission="dale2014"` + macOS jetsam

The `Fitter` background-compile thread pre-JITs every inference
backend (NUTS, MAP, geoVI, raytrace, NSS) on construction. With
dale2014 IR, the geoVI compile alone pushes peak RSS ~6 GB on macOS,
which can trip the kernel's jetsam memory pressure killer.

**Fix.**

```python
import os
os.environ.setdefault("TENGRI_NO_BACKGROUND_COMPILE", "1")  # BEFORE import tengri
```

Every notebook in `notebooks/` should set this. Verify by:

```bash
grep -L 'TENGRI_NO_BACKGROUND_COMPILE' notebooks/*.py
```

Any file in the output is a bug.

## Pattern: long warm-cache compile despite the cache

**Symptom.** You ran the notebook once, compile cost 75 s. Re-run is
*also* 75 s. Cache is supposed to make warm runs ~100 ms.

**Cause.** Default JAX persistent cache has `min_compile_time_secs=5.0`
— anything that compiled in under 5 s gets thrown away. Tengri's
`__init__` overrides to 5.0 by default, which is correct. But if you
override to a higher value, or if your cache directory was cleared
(e.g. after `pip install -U jax`), every compile re-pays.

**Fix.** Verify cache dir contents:

```bash
ls -lh ~/.cache/tengri_jax_cache/ | head -10
```

If it's empty after you've run a notebook, check
`tengri.config.JAX_PERSISTENT_CACHE_DIR` and make sure the dir is
writable. After upgrading JAX:

```python
import tengri
tengri.clear_cache()  # nuke stale entries
```

## Pattern: the watchdog itself

We use a 5-second-poll bash loop:

```bash
THRESHOLD_KB=20971520  # 20 GB
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

20 GB is comfortable; 10 GB cuts into legitimate single-NUTS-fit
workloads on the dale2014 pipeline. If your machine has < 32 GB total,
consider running this watchdog whenever you batch-author notebooks.

## When to escalate

If a single notebook with a single fit and `dust_emission="dale2014"`
peaks above 8 GB on your machine, that's a regression worth filing. The
post-Phase-II-2 baseline is ~5 GB peak compile, ~1 GB steady state.

See also:

- `docs/dev/quickstart_oom_diagnosis.md` — technical root-cause for the
  energy-balance compile cost.
- `CLAUDE.md` — `TENGRI_NO_BACKGROUND_COMPILE` and `tengri.clear_cache()`.
- `src/tengri/__init__.py` — JAX persistent cache config.
