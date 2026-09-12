# The library path costs what the harness costs per gradient; the 28 s of Finding 9 was compile

**Verdict.** `forward.fit(data)` — the `mcmc_nuts_fast` default — now pays the same
per gradient as the bench harness that chose the recipe, on every phase measured
head-to-head from the same start, key, step size and mass matrix: **0.60 vs 0.61 ms
per warmup gradient, 0.64 vs 0.64 ms per chain-gradient in sampling, 0.90 vs 0.86 ms
for one gradient of the objective at identical FLOPs (3.15 M)**. Over six `ctl-dpl`
seeds on a warm compile the library spends **126.4 s** to the harness's **117.2 s**
for 3 % *fewer* serial gradients (158 690 vs 164 340); the 1.5 s per fit that remains
is itemized, behavior-bearing and now the floor: the #1999 dense-step probe (0.3 s),
reinserting the marginalized mass into 1200 draws (0.4 s, down from 1.2 s warm and
5.5 s cold), and an 8-restart MAP seed against the harness's single restart (0.4 s).
Wall differences per seed — the library is faster on three seeds and slower on
three — are adaptation *realizations*: the two implementations reach different step
sizes from the same key (0.158 vs 0.241 on seed 7) because their jitted programs
round differently and NUTS trajectories are chaotic, and gradient counts, not
seconds, are what separate the two.

`bench/reports/2026-09-11_profile_mass_20s.md` Finding 9's **28.1 s** was measured
on a **fresh model per call**, so it included tracing and compilation the harness
excludes by timing every phase twice. On a warm compile the same fit was **17.2 s**
before this branch and is **15.1–16.9 s** after it; in a **fresh process** it is
**24.6 s** with the persistent JAX cache populated (47 s the first time on a box),
of which ~8 s is tracing four programs whose executables the cache serves. That
fresh-process number is what a notebook sees and is the next lever, not a
sampler question.

Platform: AMD Ryzen 9 5900X (24 logical cores), `JAX_PLATFORMS=cpu`, float64,
`XLA_FLAGS=--xla_force_host_platform_device_count=4` (4 CPU devices for the pmapped
chains), library and harness pinned to disjoint 8-core sets (`taskset -c 4-11` /
`12-19`) with nothing else on them; head-to-heads on `taskset -c 0-3`. Git
`34e3bd810` + this branch. Box shared with two other sessions; one swapping episode
during a first sweep contaminated its walls, and that sweep was discarded and re-run
(gradient counts, which the contention cannot touch, were identical between the two).

## What was measured

The library's `forward.fit(data)` with no arguments (`mcmc_nuts_fast`: 4 chains,
150 warmup, 300 draws, `target_accept_rate=0.8`, `profile_mass="auto"`, dense metric
by policy at D = 7, `chain_parallel="auto"` → pmap) against the harness row that
chose it, `benchmark_laplace_nuts_20s.py --arm dense-window --profile-mass
--chain-parallel pmap --n-chains 4 --n-warmup 150 --n-samples 300`, on the same
`ctl-dpl` fixture (D = 8, 14 bands, DPL SFH; D = 7 sampled once the mass is
profiled), the same mock per seed and the same fit key.

The library walls are **warm** walls: the second `fit` call in a process, with the
model's MAP-init and adaptation caches evicted between calls so the second call
re-runs MAP, warmup and sampling on already-compiled programs. That is the harness's
own convention (each phase called twice, the second timed). `n_grad_adapt` /
`n_grad_sample` are new `run_nuts` diagnostics on this branch — the sum of BlackJAX's
`num_integration_steps` over the adaptation and over the kept draws — so the two
paths report the same columns.

## Finding 1 — per-gradient parity on every phase

Head-to-head, `ctl-dpl` seed 7, same model, same start (the library's MAP), same key:

| phase | library | harness | what differs |
|---|---|---|---|
| one gradient of the objective (jit, 2000 calls) | 0.90–1.02 ms | 0.86–1.00 ms | nothing: FLOPs 3 147 102 vs 3 154 999, bytes 11.22 vs 11.26 MB |
| window adaptation, 150 steps, dense | 9.5 s / 15 752 grads = **0.60 ms** | 7.2 s / 11 891 grads = **0.61 ms** | the realization: step 0.124 vs 0.183 from the same key |
| dense-step probe (#1999), 20 NUTS steps | 0.40 s | — | harness has none |
| sampling, 4 pmapped chains × 300 draws | 6.66 s / 41 874 grads = **0.64 ms** per chain-grad | 6.75 s / 42 176 grads = **0.64 ms** | nothing |

The library's `_nuts_warmup_only` and the harness's `_window_adapt_fn` both call
BlackJAX's `window_adaptation`; from the same key they return different step sizes
because the two jitted programs are not the same XLA program (the library filters
the adaptation info and returns the divergence flags and gradient count), and a
last-bit difference in a leapfrog step is a different trajectory a few doublings
later. This is not a defect on either side; it is why a single-seed wall cannot
rank the two paths and why the sweep below is read in gradients.

## Finding 2 — six seeds, warm: the library is at parity in gradients and 1.5 s/fit in wall

`bench/results/2026-09-12_library_path_parity_{library,harness}.jsonl`:

| seed | library warm | cold | grads adapt + sample | step | R-hat | div | harness warm | cold | grads adapt + sample | step | R-hat | div |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 7 | 16.9 s | 24.6 | 13 697 + 28 582 | 0.158 | 1.004 | 31 | 10.3 s | 17.1 | 9 543 + 20 445 | 0.241 | 1.038 | 61 |
| 8 | 18.3 s | 29.6 | 10 747 + 47 000 | 0.121 | 1.035 | 12 | 16.0 s | 23.5 | 9 644 + 43 773 | 0.148 | 1.021 | 15 |
| 9 | 33.5 s | 40.4 | 31 030 + 68 353 | 0.113 | 1.002 | 3 | 19.8 s | 25.9 | 24 299 + 22 294 | 0.210 | 1.016 | 83 |
| 10 | 19.8 s | 27.6 | 10 908 + 41 039 | 0.107 | 1.057 | 155 | 24.6 s | 30.2 | 21 357 + 49 235 | 0.096 | 1.078 | 54 |
| 11 | 22.6 s | 32.8 | 19 499 + 40 002 | 0.167 | 1.019 | 23 | 31.0 s | 34.2 | 17 669 + 98 315 | 0.053 | 1.013 | 1 |
| 12 | 15.4 s | 20.6 | 10 896 + 22 674 | 0.316 | 1.010 | 19 | 15.6 s | 21.4 | 12 589 + 42 894 | 0.123 | 1.008 | 7 |
| **sum** | **126.4 s** | | serial-equivalent **158 690** | | | | **117.2 s** | | serial-equivalent **164 340** | | | |

"Serial-equivalent" gradients = adaptation gradients (one chain) + sampling gradients
/ 4 (four chains on four devices); it is the quantity a wall is proportional to.
The library runs 3 % fewer of them and takes 8 % longer; the difference, 9 s over
six fits, is the fixed cost of Finding 3. Per seed the sign flips — seeds 10, 11 and
12 are faster in the library, 7, 8 and 9 in the harness — with the gradient counts,
which is the realization noise of Finding 1, not overhead. Neither path clears the
strict R-hat < 1.01 bar in 1200 draws on every seed; that is the recipe's known
shape (Finding 5 of the 2026-09-11 report) and identical on both sides.

## Finding 3 — the fixed cost, itemized, and what this branch removed

One warm `forward.fit` on seed 7, every phase wrapped and timed (`benchmark_library_path_parity.py phases`):

| phase | wall | note |
|---|---|---|
| `Fitter.__init__` + `_auto_prewarm` | 0.06 s | compiles are warm |
| MAP init: 8 scipy L-BFGS-B restarts | 0.78 s | harness: 1 restart, 0.9 s |
| warmup (window adaptation) | 8.3–8.5 s | Finding 1: same per gradient |
| dense-step probe (#1999) | 0.24–0.41 s | 20 NUTS steps at the adapted step |
| sampling (pmap, 4 chains) | 5.3 s | Finding 1: same per gradient |
| samples → physical, diagnostics | 0.02 s | |
| finalize (mass reinsertion) | **0.40 s** | **was 1.2 s warm, 5.5 s cold** |
| unitemized | 0.25 s | |

`finalize_profile_mass` vmapped a full forward prediction over the 1200 draws
**without a jit**: every draw's prediction was dispatched op-by-op under batching, and
the closure was re-traced on every fit. It is now one `jax.jit` program cached on
the model per engine key (`_reinsert_mass_fn`), taking the draws, keys, data, noise
and presence mask as traced arguments so it is reused across galaxies. Measured:
1.2 s → 0.4 s warm on seed 7 (the 0.4 s is 1200 forward evaluations), and 5.5 s →
2.0 s the first time in a process. The same eager vmap on a spectroscopy model
(#2310 admits them) materializes `(1200, n_pixels)` intermediates, which is the
memory spike the notebook session saw at the end of a 06 fit; it is gone with the jit.

What is left is behavior: the probe is #1999's guard against a dense metric the
adapted step cannot integrate, the 8 restarts are the seed quality #2311 restored,
and the 0.4 s of reinsertion is the forward model evaluated once per draw. Together
1.5 s on a 15 s fit.

## Finding 4 — the fresh-process wall is tracing, and the persistent cache is what halves it

Library, seed 7, one process each, `TENGRI_HOST_DEVICES=4`:

| process | cold fit | warm fit (caches evicted) | everything cached |
|---|---|---|---|
| first on the box (compile cache empty) | **47.2 s** | 17.2 s | 7.7 s |
| second (cache populated), before the finalize fix | 23.9 s | 15.1 s | 5.9 s |
| clean sweep row, after the fix | 24.6 s | 16.9 s | — |

With `jax._src.compiler` logging on, every heavy program of the fit —
`jit_val_and_grad`, `jit_loss_fn`, `jit__nuts_warmup_only`, `jit__probe_nuts_stability_jit`,
`jit_predict_photometry`, `jit_predict_properties`, the pmapped chain scan — is a
**persistent cache hit** in the second process; the misses are sub-50 ms
micro-kernels (`jit_reshape`, `jit_erf`, `jit_concatenate`) the cache deliberately
does not store. The ~8 s between a cold and a warm fit in the second process is
therefore *tracing* plus executable loading: MAP-init +2.3 s, warmup +1.7 s,
sampling +2.8 s, finalize +2.0 s (before the fix). `_auto_prewarm` also loads
`jit_predict_properties` (1.3 s) before every fit, for post-fit exploration a
notebook may never do. Fewer distinct programs per fit, or a lazy properties
prewarm, is the lever for the notebook-facing number; it is outside this branch.

## Finding 5 — the harness itself needed `profile_mass=False`, and a slow-tier test needed 400 draws

Two consequences of #2281's `profile_mass="auto"` default surfaced on the way:

- `benchmark_laplace_nuts_20s.py` builds its own profiled objective from a
  **full-D** context. Since #2281 its `Fitter(...)` came back with the mass already
  fixed in the spec, and `_mass_free_name` found nothing (`ValueError: not enough
  values to unpack`) — every `--profile-mass` row was a dead row. The harness now
  constructs its context and its MAP with `profile_mass=False`.
- `tests/regression/test_sampler_seed_reproducibility.py::test_nuts_split_warmup_keeps_sampling_quality`
  (`slow`-marked, so not in the PR gate) fails on main at **R-hat 1.1102** against
  a 1.1 bar, bit-identically with and without this branch and at `a0e94dcd5` itself:
  the fit samples the 7-D profiled space now, and the fixed-key realization on an
  all-parameters-free tsnorm fixture with 2 × 200 draws sits on the bar. Measured
  on that fixture and key: profiled 400 draws → **1.031**, profiled with the
  shipped dense auto-policy → 1.019 (1 divergence); `profile_mass=False` with the
  test's diagonal metric → a **dead fit** (100 % divergent at step 0.033), with a
  dense metric → 1.39 on `width_gyr` at 117 k gradients. The unprofiled Hessian at
  the MAP has eigenvalues 0.9–1.07 × 10⁵ (condition 1.2 × 10⁵); profiled, 0.9–124.
  The MAP itself is fine and the same from L-BFGS and Adam (nlp 0.865 vs 0.934), so
  the dead fit is the #2093 pathology — diagonal NUTS on the mass–age ridge — which
  profiling removes. The test keeps its diagonal metric and its purpose and asks
  for 400 draws.

## Caveats

1. Walls are one box, one day, and the two sweeps ran concurrently on disjoint
   core sets. Gradient counts are exact and reproduce to the digit across runs.
2. Six seeds of one fixture. The head-to-heads are one seed; their per-gradient
   numbers are stable to ±3 % across the three repeats each printed.
3. The harness's MAP is one restart (its `--map-restarts` default) and the library's
   is eight; the 0.4 s that costs is counted against the library above.
4. Nothing here changes what a fit computes: the reinsertion draws the same
   conditional masses (the jitted function is the same arithmetic), and the
   gradient counts are diagnostics.

## Reproduce

```bash
export PYTHONPATH=src
# Finding 1: per-gradient head-to-heads (cores 0-3)
for cmd in grad warmup scan phases; do
  XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu taskset -c 0-3 \
    python bench/scripts/benchmark_library_path_parity.py $cmd --seed 7
done
# Finding 2: six seeds, library rows (cores 4-11) and harness rows (cores 12-19)
for s in 7 8 9 10 11 12; do
  XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu taskset -c 4-11 \
    python bench/scripts/benchmark_library_path_parity.py row --seed $s \
    --json bench/results/2026-09-12_library_path_parity_library.jsonl
  JAX_PLATFORMS=cpu taskset -c 12-19 python bench/scripts/benchmark_laplace_nuts_20s.py \
    --notebook ctl-dpl --seed $s --arm dense-window --profile-mass --chain-parallel pmap \
    --n-chains 4 --n-warmup 150 --n-samples 300 \
    --json bench/results/2026-09-12_library_path_parity_harness.jsonl
done
```
