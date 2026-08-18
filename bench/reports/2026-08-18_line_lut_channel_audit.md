# The line-LUT channel matrix: one shipped bias, one veto that no longer pays

**Date:** 2026-08-18
**Platform:** macOS, CPU (`JAX_PLATFORMS=cpu`), x64. FLOPs are off
`jax.jit(jax.grad(...)).lower(...).compile().cost_analysis()["flops"]` — the
*fit objective*, never `predict_photometry`. Measuring the photometry surface
instead of the objective is how #1760 shipped, and #1770 records what it cost.

## Why this was measured

#1925 unbroke `predict_line_fluxes` on the stateless fast line path, which had
been crashing with `AttributeError: 'NoneType' object has no attribute
'derived'`. With the crash gone the line channel could take the LUT again, so
the question was what the LUT is now worth — and whether it is correct.

The answer to the second question turned out to matter more than the first.

## Finding 1 — a 23,093x bias on the default `approx="auto"` path (#1943, fixed)

`predict_line_fluxes` sets `all_waves = grid.wavelengths` in its fast branch,
then the shared redden tail (#1877) replaces `all_lums` with the state's
published `log_line_lums_attenuated` and leaves `all_waves` alone. The two
arrays are then from different catalogs — the grid axis holds the requested
targets (3 entries), the published catalog the backend's full line list (128
for Cue). The target match walks the 3-entry axis and reads the 128-entry
array at indices 0-2: the far-UV 923-937 A lines, returned as Halpha / Hbeta /
[OIII].

Line fluxes at identical params, same model, only `state=` differing:

| call | Halpha [erg/s/cm2] |
|---|---|
| exact model (`approx=None`) | 9.150758e-12 |
| LUT model, `state=None` | 9.150752e-12 |
| **LUT model, `state` supplied** | **3.962955e-16** |

Constant across the prior volume — a fixed factor, not interpolation error:

| `log_total_mass` | exact Halpha | LUT Halpha | deviation |
|---|---|---|---|
| 12.23 | 9.105777e-12 | 3.943475e-16 | 99.9957 % |
| 11.00 | 5.361885e-13 | 2.322092e-17 | 99.9957 % |
| 9.00 | 5.361885e-15 | 2.322092e-19 | 99.9957 % |

The factor closes against the mismatch itself: `10**atten[62] / 10**atten[0]`
= 2.479454e44 / 1.073787e40 = **23,090**, against a measured flux ratio of
**23,093**.

### Effect on the objective, before and after

Same point, identical `data_args` (verified by cross-evaluation:
`nlp_f(p, args_e)` equals `nlp_f(p, args_f)` exactly), dusty Cue model,
photometry + 3 line fluxes.

| cell (`approx="auto"` vs exact) | max rel gradient dev — before | after |
|---|---|---|
| neb=cue, dust=True, lines | **8.04e-01** | **4.64e-07** |
| neb=cue, dust=False, lines | 1.07e-04 | 1.07e-04 |
| neb=none, dust=True, lines | 6.58e-06 | 6.58e-06 |
| neb=none, dust=False, lines | 1.02e-15 | 1.02e-15 |

Gradient FLOPs for the repaired cell move 44,363,920 -> 44,368,040 (**+0.009
%**), so the fix is free.

### Attribution

Run with a `WavePrecomp`-only arm in the same process:

| arm | dObj | max rel gradient dev |
|---|---|---|
| `WavePrecomp()` alone | 1.118e-06 | 4.64e-07 |
| `FeaturePrecomp()` alone | 1.138e+00 | 8.04e-01 |
| the pair | 1.138e+00 | 8.04e-01 |

`WavePrecomp` is innocent; `FeaturePrecomp` alone reproduces the whole effect.
The objective offset scaled 1.138e+00 -> 1.138e+02 for a 10x noise reduction,
i.e. exactly 1/sigma^2 — the signature of a constant forward bias entering the
gradient multiplied by SNR (#1671).

### Why the dust-free arm was clean

All four conditions are needed: a grid must exist, a state must be supplied
(`loss_functions._build_prediction` does, because `_fast_line_measurement` is
`False` while `approx.feature_precomp` is `True`), the state must publish the
attenuated catalog, and `use_grid` must therefore be `False`. **Dust is what
forces the last two** (`must_materialize_sed`, #1281/#1748). A dust-free model
publishes no attenuated catalog, takes the fallback screen against the grid's
own axis, and was correctly paired all along.

That is also why the dust-free control in
`test_bug_1943_fast_line_catalog_mismatch.py` passes against the defect, and
why it is not optional.

## Finding 2 — the `_has_line_adjacent_channel` veto: free with dust, 3.27x without

`_has_line_adjacent_channel` refuses `FeaturePrecomp` for any fit carrying a
`line_ratios` or `spectral_indices` channel. Its stated ground (#1665) is that
the `WavePrecomp + FeaturePrecomp` pair arms the fast nebular grid, which
zeroes the nebular continuum and skips the discrete line-catalog publish.

Measured `use_grid` per arm, reading the four terms of the condition directly
rather than inferring engagement from a number:

| model | arm | `use_grid` |
|---|---|---|
| with dust | wave+feature | **False** (`must_materialize_sed=True`) |
| no dust | wave+feature | **True** |

So the hazard the veto guards can only arise dust-free — and there, #1665's own
fix (requiring `log_restband_per_qh` and publishing the rest-band twin) has
already removed it. Measured on a dusty Cue model, all 13 spectral indices and
both line ratios:

| arm | max index deviation | max ratio deviation |
|---|---|---|
| wave only | 0.0000 % | — |
| feature only | 0.0000 % | — |
| wave+feature | 0.0000 % | 0.000000 % |

Cost of the veto, gradient FLOPs of the fit objective (photometry + line fluxes
+ 2 line ratios):

| model | resolver's choice | forced pair | ratio |
|---|---|---|---|
| dusty | 44,371,320 | 44,367,200 | **1.00x** |
| dust-free | 39,156,648 | 11,969,681 | **3.27x** |

With dust the veto costs nothing — dust disarms the photometry shortcut *and*
the ratio channel forces the state rebuild regardless, so both halves of the
LUT were already off. Dust-free it costs 3.27x, and the objective and gradient
agree with exact to 3.5e-08 and 1.07e-04.

**Not acted on here.** #1665 says of this gate: *"so this gate is load-bearing;
do not relax it for speed."* The measurement above says its premise has since
been repaired, but relaxing a correctness gate is a separate decision from
fixing a correctness bug, and it belongs in its own change with its own guard —
not folded into #1943. Filed for that decision rather than taken.

## Reproduce

The probes live in the issue thread for #1943 rather than in `bench/scripts/`:
each is a dozen lines against the public API, and the durable form of the
result is `tests/regression/bug/test_bug_1943_fast_line_catalog_mismatch.py`,
which pins the correctness half and runs in the default tier.

```bash
.venv/bin/pytest tests/regression/bug/test_bug_1943_fast_line_catalog_mismatch.py -q -n 0
```

## One trap worth recording

Every measurement here was run with `PYTHONPATH` pinned to the worktree's
`src/`. The venv's editable install is a plain `.pth` holding one absolute path
to the **main checkout**, and path-based `.pth` entries land on `sys.path`
after `PYTHONPATH` — so a worktree session that runs `.venv/bin/pytest` with
nothing set tests the main checkout's source while editing its own, greenly and
silently. Verified directly (a throwaway test printing `tengri.__file__`
resolved to the main checkout), not assumed.
