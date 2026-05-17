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

Numbers will be filled in once the worktree's venv has been refreshed.
Expected envelope:

| Recipe                       | Pre-refactor | Post-refactor | Notes        |
|------------------------------|--------------|---------------|--------------|
| star_forming_photometry      | TBD          | TBD           | DEFAULT path |
| star_forming_photometry      | TBD          | TBD           | LOW_MEMORY   |
| agn_panchromatic             | TBD          | TBD           | DEFAULT path |
| star_forming, NUTS warmup    | TBD          | TBD           | regression   |

Variance threshold for accepting numbers: ±3% on steady-state, ±10% on
warmup (cache-cold compile).

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
