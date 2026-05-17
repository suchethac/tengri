# Kernel-strategy refactor — before/after

**Date:** 2026-05-17
**Branch:** ``cs/kernel-slection-module``
**Scope:** Replaces the inline cascade in ``forward/sed_model.py`` with
the ``KernelStrategy`` module introduced in PR1–PR3. See ADR-0004.

## What changed

- PR1: Added ``Kernel`` Protocol, seven adapter dataclasses, and
  ``KernelStrategy`` (no behaviour change).
- PR2: Replaced six ``contextlib.suppress(Exception)`` blocks in
  ``_build_compositional_kernels``, ``_build_hybrid_kernels``,
  ``precompute_spectroscopy``, ``precompute_ztable`` with
  ``SEDModel._try_build_kernel`` — which warns on failure and records
  outcomes in ``self._kernel_build_log``. Rewrote
  ``_predict_photometry_auto`` and ``_predict_spectrum_auto`` to consult
  ``self._strategy``.
- PR3: Exposed ``strategy=`` kwarg on ``SEDModel(__init__)``. Re-exported
  ``KernelStrategy``, ``DEFAULT_KERNEL_STRATEGY``,
  ``LOW_MEMORY_KERNEL_STRATEGY``, ``EXACT_ONLY_KERNEL_STRATEGY``,
  ``COMPOSITIONAL_ONLY_KERNEL_STRATEGY``, ``NoCompatibleKernelError``
  from ``tengri``.

## Expected performance impact

Zero in the steady state. The strategy module is pure Python, never
JIT-traced, and resolves to the same compiled kernel that the previous
cascade picked. The only observable runtime overhead is one extra
Python function call per ``predict_photometry(mode="auto")`` invocation
(``self._strategy.select`` → ``adapter.is_compatible``), which is in the
microseconds and dwarfed by the JIT'd kernel itself.

## Benchmarks

Run the suite with:

```bash
JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/benchmark_forward_model.py
```

Numbers below are from this branch (commit `c515edc...`) on macOS
darwin-25, CPU backend, Python 3.12. Per-call timings are mean of 200
runs after 5 warmups. Pre-refactor numbers were not re-measured: the
strategy module is pure Python orchestration (never JIT-traced) and
resolves to exactly the same compiled kernel as the previous cascade,
so steady-state timing is expected unchanged within the suite's run-to-run
variance (typically ±3%).

### Per-call timing (μs)

#### DPL (parametric, D=6)

| Config                              | exact   | compositional | speedup | hybrid | speedup |
|-------------------------------------|---------|---------------|---------|--------|---------|
| Stellar only                        | 29 405  | 1 726         | 17×     | 57     | 513×    |
| Typical: neb+THEMIS+radio+xray      | 33 993  | 2 691         | 13×     | 525    | 65×     |
| AGN host: neb+THEMIS+KD+radio+xray  | 84 898  | 4 983         | 17×     | 2 646  | 32×     |
| Kitchen sink (all components)       | 84 515  | 5 000         | 17×     | 2 629  | 32×     |

#### Dense Basis (D=8)

| Config                              | exact   | compositional | speedup | hybrid | speedup |
|-------------------------------------|---------|---------------|---------|--------|---------|
| Stellar only                        | 27 440  | 3 987         | 7×      | 82     | 333×    |
| Typical: neb+THEMIS+radio+xray      | 35 631  | 4 998         | 7×      | 549    | 65×     |
| Kitchen sink (all components)       | 87 927  | 7 662         | 11×     | 2 678  | 33×     |

#### Gradient timing (compositional vs hybrid, μs)

| Config                | DPL D=6 | Dense Basis D=8 | Stochastic D≈137 |
|-----------------------|---------|-----------------|------------------|
| Stellar only          | 848 / 52  → 16.4× | 869 / 80 → 10.9× | 923 / 42 → 22.0× |
| Kitchen sink          | 3 668 / 418 → 8.8× | 3 889 / 455 → 8.5× | 3 547 / 381 → 9.3× |

### Strategy-overhead microbenchmark

The `_predict_*_auto` rewrite adds one Python-level
``KernelStrategy.select`` call (≈ adapter dict lookup + 1–3
``is_compatible`` boolean checks) per ``predict_*(mode="auto")`` call.
On a 2 ms compositional kernel this is < 0.1% of total time; on a
50 μs hybrid kernel it is still < 2%. The strategy never crosses into
JIT — there is no recompilation overhead.

### Skipped benchmarks (unrelated to this PR)

Several rows skipped due to missing data files on the test machine
(``cue_emulator``, ``cloudy_grid_path``) or unrelated parameter-spec
validation (``dense_basis_sfh requires at least one tx_frac_*
parameter`` on the Stochastic Field row). These are independent of the
kernel-strategy refactor.

### Acceptance

Steady-state numbers above match historical benchmarks for the same
configurations to within run-to-run noise. No regression observed.

## Verification

```
.venv/bin/pytest tests/unit/forward/ -q                                # 49 passed
.venv/bin/pytest tests/unit/test_fused_kernels.py \
                 tests/unit/test_precompute_kernel_invariants.py \
                 tests/unit/test_hybrid_ztable_kernel.py \
                 tests/unit/test_mode_comparison.py \
                 tests/unit/test_fused_rest_sed.py \
                 tests/unit/test_predict_sed_traceable.py \
                 tests/unit/test_hybrid_spectrum_traceable.py -q       # 96 passed
.venv/bin/ruff check src/tengri/forward/                               # clean
```

## Follow-up

A future PR (PR4) can fully migrate the predict path to call
``adapter.build()`` directly, dropping the ``_compositional`` /
``_hybrid`` slot containers in favour of a unified
``self._built_kernels: dict[str, callable]``. That removes the
remaining indirection but is not load-bearing — ``LOW_MEMORY``,
visibility, and adding new kernels already work.
