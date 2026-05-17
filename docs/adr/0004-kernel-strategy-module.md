# ADR 0004: Kernel strategy as an explicit module

**Status:** Accepted

**Date:** 2026-05-17

## Context

`forward/sed_model.py` (5277 lines) carried the kernel-selection policy
across roughly ten sites:

- four ``contextlib.suppress(Exception)`` blocks around ``build_*`` calls in
  ``_build_compositional_kernels`` / ``_build_hybrid_kernels``,
- two more in ``precompute_spectroscopy`` and ``precompute_ztable``,
- a cascade in ``_predict_photometry_auto`` / ``_predict_spectrum_auto``
  that picked among ``self._compositional`` / ``self._hybrid`` slots,
- inline param-shape checks (``"sfh_t_gyr" not in params``) and state
  checks (``self._met_mode in {"ramp", "chem_evol"}``) deciding what to do
  with each kernel.

This had three failure modes:

1. **Silent failure.** The Phase II-2 ``UnboundLocalError`` in the hybrid
   kernel went undetected for weeks because the surrounding ``suppress``
   block ate the exception and the user got a 114 MB compositional
   closure instead — 40× the expected XLA HLO size.
2. **Untestable policy.** "If hybrid were unavailable, what would you
   pick?" could only be answered by provoking a real OOM.
3. **Adding a fourth kernel** (e.g. the planned preintegrated triweight
   from ``project_preintegrate_design``) required touching ``sed_model.py``
   in many places.

## Decision

Introduce a ``Kernel`` Protocol and a frozen ``KernelStrategy`` dataclass,
both in ``src/tengri/forward/_kernels/`` (a new ``_protocol.py``,
``_adapters.py``, and ``strategy.py``). Each existing ``build_*`` factory
is wrapped by a thin adapter that exposes ``name``, ``product``
(``"rest_sed"`` | ``"photometry"`` | ``"spectrum"``), ``is_compatible``
(state-only predicate), ``is_compatible_with_params`` (param-only
predicate), and ``build``. ``KernelStrategy.select`` iterates the
preferred names and yields compatible adapters; ``_predict_*_auto``
consult the strategy instead of an inline cascade.

Build-time failures are no longer swallowed. ``SEDModel._try_build_kernel``
emits a ``UserWarning`` and records the failure in
``self._kernel_build_log``; ``SEDModel.list_available_kernels()`` returns
that log for inspection.

The strategy lives one level above JIT — it inspects state and params
with plain Python and is never traced.

**Built-in policies:** ``DEFAULT`` (historical cascade), ``LOW_MEMORY``
(skip hybrid), ``EXACT_ONLY`` (force the slow path), ``COMPOSITIONAL_ONLY``.

Public surface: re-exported from ``tengri`` as ``KernelStrategy``,
``DEFAULT_KERNEL_STRATEGY``, ``LOW_MEMORY_KERNEL_STRATEGY``,
``EXACT_ONLY_KERNEL_STRATEGY``, ``COMPOSITIONAL_ONLY_KERNEL_STRATEGY``,
``NoCompatibleKernelError``. ``SEDModel(__init__, strategy=...)`` accepts
a custom strategy.

## Why three kernels coexist

The three families exist for orthogonal reasons, not as redundant
implementations:

- **Exact** (``build_exact_sed``, ``build_fused_rest_sed``) — full
  wavelength resolution, always works, slowest. Bit-exact reference for
  regression tests.
- **Compositional** (``build_fused_tier2_*``) — full wavelength, fused
  JIT, bit-identical to exact via closure-A. Default for photometry and
  spectroscopy.
- **Hybrid** (``build_hybrid_*``) — precomputed SSP×filter einsum
  (~0.4% stellar tolerance) + exact non-stellar. Fastest for photometry
  but biggest XLA HLO; largest OOM risk.

The strategy module pins this hierarchy explicitly so future contributors
do not re-derive it from scattered code.

## The ``is_compatible`` contract

Adapters split predicates into two:

- ``is_compatible(state, model)`` — state-only. Reads filter availability,
  precomputed caches, ``z_fixed``. Decides whether the kernel **can** be
  built at all. Lifted verbatim from the inline checks that gated each
  ``build_*`` call.
- ``is_compatible_with_params(params)`` — param-shape predicate.
  Currently only used to gate hybrid adapters when a tabulated SFH
  (``"sfh_t_gyr"``) is present, since hybrid grids are fixed-size. Future
  param-shape predicates (evolving metallicity, etc.) go here.

Adapters may not look at any other model state. If a predicate needs
more, surface that requirement on ``SEDModelState`` first.

## Consequences

**Positive.**

- The policy is testable in isolation via mock state objects (34 new
  unit tests in ``tests/unit/forward/test_kernel_strategy.py``).
- Build failures are visible: ``model.list_available_kernels()`` returns
  ``{kernel: status}`` instead of silent ``None``.
- Adding a fourth kernel is one adapter file plus an entry in
  ``ALL_ADAPTERS``. No edits to ``sed_model.py``.
- ``LOW_MEMORY`` is now a real, observable behaviour change, not an
  aspirational comment.

**Negative.**

- Param-dependent predicates (``"sfh_t_gyr" not in params``,
  ``met_mode``) are now visible at the adapter layer instead of buried
  in ``_predict_*_auto``. A small new surface, but explicit.
- The strategy must not enter JIT. The Protocol is Python-only.
  Documented in ``strategy.py`` module docstring.

**Risks (mitigated).**

- *Hybrid 0.5% tolerance.* Strategy preserves the historical cascade
  order, so paths that previously took hybrid still take hybrid.
- *``_traceable`` mode (NIFTy VI).* Left untouched. Strategy applies
  only to the JIT'd predict paths.

## References

- ``docs/dev/quickstart_oom_diagnosis.md`` — Phase II-2 silent fallback
  incident that motivated visibility.
- ``project_preintegrate_design`` — the fourth kernel that this design
  is meant to absorb cheaply.
- ``docs/dev/NAMING_CONTRACT.md`` §4 — ``build_*`` factory verb,
  ``*Strategy`` orchestrator suffix.
