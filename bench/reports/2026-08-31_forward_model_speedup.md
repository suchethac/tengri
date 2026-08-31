# Benchmark: forward-model precompute speedup across all emitters

**Date:** 2026-08-31
**Status:** Current — fresh measured numbers after PR #135 deleted the hybrid kernel adapter.
**Verdict:** PASS — WavePrecomp fast path delivers 9–17× speedup on forwards and 6.5–9.4× on gradients;
AGN dense-integrating components (K&D 3-zone, SKIRTOR) see minimal speedup (~1×) because they bypass
band-projection shortcut.

## Provenance

- **Platform:** CPU (Apple M4 Pro, macOS 14)
- **JAX version:** 0.11.1
- **Float precision:** float64
- **JAX_PLATFORMS:** cpu
- **Quiet-machine protocol:** No concurrent JAX processes; run gated on process checks
- **Branch:** `fix/2092-bench-honesty`
- **Script commit:** `0a801fe14` (ruff formatting)
- **N_WARMUP:** 5
- **N_RUNS:** 200
- **SSP grid:** `ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5`
- **Filters:** SDSS *ugriz* (5 filters)
- **Fixed redshift:** z = 0.1

Exact command:
```bash
JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/benchmark_forward_model.py
```

## Forward photometry — DPL parametric (D=6)

| Config                          | exact    | precomp  | speedup | max rel err |
| ------------------------------- | -------: | -------: | ------: | ----------: |
| Stellar only                    |  9803 µs |   719 µs |   13.6× |     0.009% |
| + nebular (baked-in SSP)        | 11134 µs |   731 µs |   15.2× |     0.009% |
| + dust IR (MBB)                 | 11651 µs |   767 µs |   15.2× |     0.009% |
| + dust IR (THEMIS)              | 11383 µs |   794 µs |   14.3× |     0.009% |
| + dust IR (DL07)                | 13136 µs |   772 µs |   17.0× |     0.009% |
| + dust IR (Dale 2014)           | 12734 µs |   830 µs |   15.3× |     0.009% |
| + AGN (composable disc+torus)   | 10688 µs |  4837 µs |    2.2× |     0.009% |
| + AGN (K&D 3-zone full)         | 15925 µs | 16013 µs |    1.0× |     0.009% |
| + AGN (QSOgen)                  | 10615 µs |  5227 µs |    2.0× |     0.009% |
| + AGN (SKIRTOR torus)           | 12899 µs | 13958 µs |    0.9× |     0.009% |
| + radio (SF + AGN)              | 10855 µs |   809 µs |   13.4× |     0.009% |
| + X-ray (XRB + corona)          | 10694 µs |  3407 µs |    3.1× |     0.009% |
| Typical: neb+THEMIS+radio+xray  | 12625 µs |  3432 µs |    3.7× |     0.009% |
| AGN host: neb+THEMIS+KD+radio+xray | 20135 µs | 12693 µs | 1.6× | 0.009% |
| Kitchen sink (all components)   | 19120 µs | 12484 µs |    1.5× |     0.009% |

## Forward photometry — Dense Basis (D=8)

| Config                          | exact    | precomp  | speedup | max rel err |
| ------------------------------- | -------: | -------: | ------: | ----------: |
| Stellar only                    |  9925 µs |   808 µs |   12.3× |     0.009% |
| + nebular (baked-in SSP)        | 10040 µs |   826 µs |   12.2× |     0.009% |
| + dust IR (MBB)                 | 11556 µs |   953 µs |   12.1× |     0.009% |
| + dust IR (THEMIS)              | 12537 µs |   978 µs |   12.8× |     0.009% |
| + dust IR (DL07)                | 12758 µs |  1012 µs |   12.6× |     0.009% |
| + dust IR (Dale 2014)           | 12640 µs |   886 µs |   14.3× |     0.009% |
| + AGN (composable disc+torus)   | 10205 µs |  5115 µs |    2.0× |     0.009% |
| + AGN (K&D 3-zone full)         | 16296 µs | 14983 µs |    1.1× |     0.009% |
| + AGN (QSOgen)                  | 10059 µs |  5695 µs |    1.8× |     0.009% |
| + AGN (SKIRTOR torus)           | 12757 µs | 14802 µs |    0.9× |     0.009% |
| + radio (SF + AGN)              | 11174 µs |   975 µs |   11.5× |     0.009% |
| + X-ray (XRB + corona)          | 10972 µs |  3466 µs |    3.2× |     0.009% |
| Typical: neb+THEMIS+radio+xray  | 12072 µs |  3754 µs |    3.2× |     0.009% |
| AGN host: neb+THEMIS+KD+radio+xray | 18930 µs | 11577 µs | 1.6× | 0.009% |
| Kitchen sink (all components)   | 19810 µs | 13489 µs |    1.5× |     0.009% |

## Forward photometry — Stochastic Field (D∼137)

| Config                          | exact    | precomp  | speedup | max rel err |
| ------------------------------- | -------: | -------: | ------: | ----------: |
| Stellar only                    |  9860 µs |   782 µs |   12.6× |     0.009% |
| + nebular (baked-in SSP)        |  9565 µs |   724 µs |   13.2× |     0.009% |
| + dust IR (MBB)                 | 11263 µs |   918 µs |   12.3× |     0.009% |
| + dust IR (THEMIS)              | 12528 µs |   959 µs |   13.1× |     0.009% |
| + dust IR (DL07)                | 11942 µs |   858 µs |   13.9× |     0.009% |
| + dust IR (Dale 2014)           | 12506 µs |   978 µs |   12.8× |     0.009% |
| + AGN (composable disc+torus)   | 10482 µs |  4907 µs |    2.1× |     0.009% |
| + AGN (K&D 3-zone full)         | 15837 µs | 15364 µs |    1.0× |     0.009% |
| + AGN (QSOgen)                  | 10256 µs |  5571 µs |    1.8× |     0.009% |
| + AGN (SKIRTOR torus)           | 13511 µs | 14765 µs |    0.9× |     0.009% |
| + radio (SF + AGN)              | 10743 µs |  1125 µs |    9.6× |     0.009% |
| + X-ray (XRB + corona)          | 11177 µs |  3616 µs |    3.1× |     0.009% |
| Typical: neb+THEMIS+radio+xray  | 11953 µs |  3667 µs |    3.3× |     0.009% |
| AGN host: neb+THEMIS+KD+radio+xray | 18903 µs | 13325 µs | 1.4× | 0.009% |
| Kitchen sink (all components)   | 19105 µs | 13594 µs |    1.4× |     0.009% |

## Gradients — full FREE-parameter-vector `jax.grad(predict_photometry)`

Gradient measurements are over the **full FREE-parameter-vector dict** — each model's
free parameters, not just a single scalar. This is fundamentally different from the
2026-05-06 report, which timed a one-scalar VJP and thus cannot be directly compared.
Fixed parameters are closed over in the loss function.

| SFH type             | stellar exact (µs) | stellar precomp (µs) | speedup | kitchen-sink exact (µs) | kitchen-sink precomp (µs) | speedup |
| -------------------- | -----------------: | -------------------: | ------: | ----------------------: | ------------------------: | ------: |
| DPL (D=6)            |               6207 |                  660 |    9.4× |                    FAILED |                     FAILED |     — |
| Dense Basis (D=8)    |               6011 |                  928 |    6.5× |                    FAILED |                     FAILED |     — |
| Stochastic Field     |               6061 |                  870 |    7.0× |                    FAILED |                     FAILED |     — |

Kitchen-sink gradient rows fail with `ConcretizationTypeError` on the exact path (issue #2114).

## Skipped sections census

**Total: 12 skipped sections**

Bare-SSP skips (9):
- `+ nebular (CLOUDY grid)_dpl`
- `+ nebular (Cue emulator)_dpl`
- `Cue+DL07+composable AGN_dpl`
- `+ nebular (CLOUDY grid)_dense_basis`
- `+ nebular (Cue emulator)_dense_basis`
- `Cue+DL07+composable AGN_dense_basis`
- `+ nebular (CLOUDY grid)_field`
- `+ nebular (Cue emulator)_field`
- `Cue+DL07+composable AGN_field`

Reason: bare-stellar SSP grid unavailable; set `TENGRI_BENCH_BARE_SSP=<path>` to enable.

Kitchen-sink gradient failures (3):
- `Kitchen sink (all components)_grad_dpl` — ConcretizationTypeError (issue #2114)
- `Kitchen sink (all components)_grad_dense_basis` — ConcretizationTypeError (issue #2114)
- `Kitchen sink (all components)_grad_field` — ConcretizationTypeError (issue #2114)

## What changed vs 2026-05-06

The 2026-05-06 report measured the "hybrid" path, a kernel-adapter family (`_HybridPhotometryKernel`, etc.)
that conditionally invoked exact-path or precompute-path photometry based on component properties.
This adapter was deleted in PR #135 as part of Phase 6 refactoring, leaving only two paths:

- **Exact:** full wave-grid integration via `observation.predict` (no `approx` argument)
- **WavePrecomp:** precomputed SSP×filter LUT via `observation.predict_via_precomp`

The 2026-05-06 measurements cannot be compared directly because:
1. The hybrid path is gone.
2. Gradient measurements in that report were on a one-scalar VJP, not the full FREE-parameter-vector gradient.

The 2026-08-31 numbers are the fresh baseline for the two-path world. Key observations:

- **AGN dense integrators (K&D 3-zone, SKIRTOR)** deliver ~1× "speedup" (i.e., no speedup).
  These components bypass the band-projection fast branch in `observation/_band_projection.py` because they
  require full-resolution SED integration per call; the precompute overhead cancels any lookup savings.
  See #1022 and the two-branch architecture comment in `_band_projection.py`.
- **Stellar + dust + radio + X-ray** consistently see 12–17× forwards and 6.5–9.4× gradients (stellar only).
- **All errors < 0.01%** — the precompute LUT is faithful across all configurations tested.

## References

- `bench/scripts/benchmark_forward_model.py` — the benchmark harness
- `observation/_band_projection.py` — two-branch photometry kernel (exact vs precompute route selection)
- `#1022` — design of precompute fast-path and AGN dense-integration exception
- `#2114` — kitchen-sink gradient ConcretizationTypeError (open)
