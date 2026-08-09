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
# Kill anything you don't recognize:
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
the dale2014 + nebular pipeline is ~4 GB. The JIT cache amortizes the
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
  `fitter.run("mcmc_nuts", dense_mass_matrix=False, ...)` cuts compile peak
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

Three scripts cover the three failure shapes:

- `scripts/run_with_oom_monitor.sh` — wraps **one command** and
  SIGKILLs its whole process tree if the tree's summed RSS exceeds
  `LIMIT_GB`. Use for every heavy launch (notebook renders, full
  pytest tiers, NUTS/geoVI fits).
- `scripts/python_oom_guard.sh` — daemon; SIGKILLs any **single**
  python process over `LIMIT_GB` (default 10), whichever session
  spawned it.
- `scripts/python_total_oom_guard.sh` — daemon; watches the
  **machine-wide total** python RSS *and* real OS memory pressure, and
  sheds the most memory-hungry processes first. This is the only guard
  that survives the multi-session case: N parallel worktrees each under
  their own per-tree limit can still jointly exceed physical RAM, and no
  per-tree monitor can see the others.

**Install the total guard as a LaunchAgent — do not background it by
hand:**

```bash
scripts/install_oom_guard_agent.sh
```

`nohup ... & disown` is not enough. It dies at logout and reboot, nothing
restarts it, and its `/tmp` log gets reaped, so the *absence leaves no
trace*. On 2026-08-09 a machine hit ~120 GB of summed python RSS with the
guard installed but dead for weeks; the only clue was a missing log file.
`KeepAlive` in the agent also covers silent death mid-session.

Three traps this guard had to be taught, all of which made it look
healthy while doing nothing:

1. **A sum-RSS limit can sit above what the workload ever reaches.**
   Measured on the incident machine: python summed to 12–16 GB against a
   30 GB limit — never tripping — while the box suffocated. Hence the
   second trigger on `AVAIL_PCT_MIN` (available memory), which is what
   the kernel is actually short of. Do **not** reach for the
   `SWAP_MAX_GB` trigger on macOS: swap there grows on demand and never
   shrinks, so it reads as a permanent emergency on any long-uptime
   machine and would kill 8 GB out of every `pytest -n auto` run.
2. **A `MIN_KILL_MB` floor must never veto the whole candidate list.**
   66% of python RSS on that machine sat below the 512 MB floor. The
   selector now runs a second pass that ignores the floor when the shed
   target cannot otherwise be met.
3. **Don't shed when shedding cannot help.** Pressure trips require
   python to hold at least `PRESSURE_MIN_PYTHON_GB` (default 8),
   otherwise a chronic swap condition re-trips every cooldown and
   eventually kills every python process for no benefit.

For a single-fit limit, 20 GB is comfortable; 10 GB cuts into
legitimate single-NUTS-fit workloads on the dale2014 pipeline. If your
machine has < 32 GB total, run the guards whenever you batch-author
notebooks. Both daemons honor `EXCLUDE_RE` to protect processes you
never want shot; the total guard additionally supports `DRY_RUN=1` to
preview the kill plan a given limit would produce.

## When to escalate

If a single notebook with a single fit and `dust_emission="dale2014"`
peaks above 8 GB on your machine, that's a regression worth filing. The
post-Phase-II-2 baseline is ~5 GB peak compile, ~1 GB steady state.

See also:

- `docs/dev/quickstart_oom_diagnosis.md` — technical root-cause for the
  energy-balance compile cost.
- `CLAUDE.md` — `TENGRI_NO_BACKGROUND_COMPILE` and `tengri.clear_cache()`.
- `src/tengri/__init__.py` — JAX persistent cache config.
